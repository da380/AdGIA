/*
  Parallel tests for NonlinearClampedProblem (nonlinear_problem.hpp), on
  the body of TestNonlinearClamped.cpp. Run with 1, 2 and 4 ranks; a
  standalone MPI program that fails if any check does.

  Every rank also builds the serial problem on the whole mesh; the
  parallel and serial solutions are compared through quantities that do
  not depend on the partition (the L2 norm of the displacement, the
  energy, the corrections of each iteration). In parallel as well: the
  residual against differences of the energy and the tangent against
  differences of the residual, at a state that is not the reference; the
  affine patch; and the reference as the solution without loads.
*/

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "AdGIA.hpp"
#include "NonlinearTestCommon.hpp"
#include "mfem.hpp"

using namespace mfem;
using namespace AdGIA;
using namespace nonlinear_test;

namespace {

int num_checks = 0;
int num_fails = 0;

void Check(double err, double tol, const std::string& what) {
  num_checks++;
  if (!(err <= tol)) {
    num_fails++;
    if (Mpi::Root()) {
      std::cout << "FAIL: " << what << "  (err = " << err << ", tol = " << tol
                << ")\n";
    }
  }
}

double RelErr(double a, double b) { return std::abs(a - b) / std::abs(b); }

double L2Norm(const GridFunction& u) {
  auto zero = Vector(u.FESpace()->GetVDim());
  zero = 0.0;
  auto z = VectorConstantCoefficient(zero);
  return u.ComputeL2Error(z);  // global for a ParGridFunction
}

// The pairing of a dual vector in the vdof layout with a field.
double Pair(ParFiniteElementSpace& fes, const Vector& dual,
            const ParGridFunction& field) {
  auto dual_true = Vector(fes.GetTrueVSize());
  fes.GetProlongationMatrix()->MultTranspose(dual, dual_true);
  auto field_true = Vector(fes.GetTrueVSize());
  field.GetTrueDofs(field_true);
  return InnerProduct(fes.GetComm(), dual_true, field_true);
}

void RunCase(int dim, int order, int type, const std::string& label) {
  auto smesh = MakeBody(dim, type, dim == 2 ? 4 : 2);
  auto bottom = Marker(smesh, 1);
  auto top = Marker(smesh, TopAttribute(dim));
  auto fec = H1_FECollection(order, dim);
  auto material = Material(dim);
  auto& law = material.cubic;
  auto traction = VectorConstantCoefficient(TopTraction(dim, 0.04));

  // Serial reference on every rank.
  auto sfes = FiniteElementSpace(&smesh, &fec, dim);
  auto serial = NonlinearClampedProblem(&sfes, law, bottom, traction, top);
  Check(serial.Solve() ? 0.0 : 1.0, 0.0, label + " serial solve");
  const auto u_ref = L2Norm(serial.Displacement());
  const auto energy_ref = serial.Energy();

  auto pmesh = ParMesh(MPI_COMM_WORLD, smesh);
  auto fes = ParFiniteElementSpace(&pmesh, &fec, dim);
  auto problem = NonlinearClampedProblem(&fes, law, bottom, traction, top);
  Check(problem.IsParallel() ? 0.0 : 1.0, 0.0, label + " parallel space");
  Check(problem.Solve() ? 0.0 : 1.0, 0.0, label + " solve");
  Check(RelErr(L2Norm(problem.Displacement()), u_ref), 1e-8, label + " |u|");
  Check(RelErr(problem.Energy(), energy_ref), 1e-8, label + " energy");
  const auto& steps = problem.Corrections();
  const auto& steps_ref = serial.Corrections();
  Check(steps.size() == steps_ref.size() ? 0.0 : 1.0, 0.0,
        label + " iterations");
  for (auto k = 0u; k < steps.size() && k < steps_ref.size(); k++) {
    if (steps_ref[k] > 1e-8 * u_ref) {
      Check(RelErr(steps[k], steps_ref[k]), 1e-4,
            label + " correction " + std::to_string(k + 1));
    }
  }

  // Derivatives at a state that is not the reference, along directions
  // that vanish on the clamped part, by the fourth-order central
  // difference.
  auto zero_vector = Vector(dim);
  zero_vector = 0.0;
  auto zero = VectorConstantCoefficient(zero_vector);
  auto state = VectorFunctionCoefficient(
      dim, [](const Vector& x, Vector& u) { SmoothField(x, u, 0.04, 0.3); });
  auto v_c = VectorFunctionCoefficient(
      dim, [](const Vector& x, Vector& u) { SmoothField(x, u, 1.0, 1.1); });
  auto w_c = VectorFunctionCoefficient(
      dim, [](const Vector& x, Vector& u) { SmoothField(x, u, 1.0, 2.7); });
  auto u0 = ParGridFunction(&fes), v = ParGridFunction(&fes),
       w = ParGridFunction(&fes);
  u0.ProjectCoefficient(state);
  v.ProjectCoefficient(v_c);
  w.ProjectCoefficient(w_c);
  v.ProjectBdrCoefficient(zero, bottom);
  w.ProjectBdrCoefficient(zero, bottom);

  auto& u = problem.Displacement();
  const auto h = 1e-4;
  auto derivative = [&](const ParGridFunction& direction, auto&& value) {
    auto at = [&](double step) {
      u = u0;
      u.Add(step, direction);
      return value();
    };
    return (8.0 * (at(h) - at(-h)) - (at(2.0 * h) - at(-2.0 * h))) /
           (12.0 * h);
  };

  u = u0;
  auto r = Vector();
  problem.Residual(r);
  const auto rv = Pair(fes, r, v);
  const auto dE = derivative(v, [&] { return problem.Energy(); });
  Check(RelErr(dE, rv), 1e-8, label + " residual is the energy's variation");

  u = u0;
  const auto& K = *problem.Tangent().SystemMatrix().As<HypreParMatrix>();
  auto v_true = Vector(fes.GetTrueVSize()), w_true = Vector(fes.GetTrueVSize());
  v.GetTrueDofs(v_true);
  w.GetTrueDofs(w_true);
  auto Kw = Vector(fes.GetTrueVSize());
  K.Mult(w_true, Kw);
  const auto vKw = InnerProduct(fes.GetComm(), v_true, Kw);
  const auto dr = derivative(w, [&] {
    auto at = Vector();
    problem.Residual(at);
    return Pair(fes, at, v);
  });
  Check(RelErr(dr, vKw), 1e-8, label + " tangent is the residual's variation");
}

// With an affine displacement prescribed on the whole boundary of a
// homogeneous body, the solution is that displacement.
void RunAffinePatch(int dim, int order, int type, const std::string& label) {
  auto smesh = MakeBody(dim, type, dim == 2 ? 4 : 2);
  auto pmesh = ParMesh(MPI_COMM_WORLD, smesh);
  auto fec = H1_FECollection(order, dim);
  auto fes = ParFiniteElementSpace(&pmesh, &fec, dim);
  auto all = Array<int>(pmesh.bdr_attributes.Max());
  all = 1;
  auto none = Marker(pmesh, 0);

  auto lambda = ConstantCoefficient(1.2), mu = ConstantCoefficient(0.9);
  auto pressure = ConstantCoefficient(0.3);
  auto C = IsotropicElasticTensorCoefficient(dim, lambda, mu);
  auto Xi = BareElasticTensorCoefficient(dim, C, pressure);
  auto stress = DenseMatrix(dim);
  stress = 0.0;
  for (auto j = 0; j < dim; j++) {
    stress(j, j) = -0.3;
  }
  auto T0 = MatrixConstantCoefficient(stress);
  auto law = QuadraticMaterialLaw(dim, Xi, &T0);

  auto A = DenseMatrix(dim);
  for (auto i = 0; i < dim; i++) {
    for (auto j = 0; j < dim; j++) {
      A(i, j) = 0.12 * std::sin(0.4 + 1.9 * i + 0.8 * j);
    }
  }
  auto affine = VectorFunctionCoefficient(
      dim, [&A](const Vector& x, Vector& u) { A.Mult(x, u); });
  auto no_traction_value = Vector(dim);
  no_traction_value = 0.0;
  auto no_traction = VectorConstantCoefficient(no_traction_value);
  auto problem =
      NonlinearClampedProblem(&fes, law, all, no_traction, none, &affine);
  problem.Options().refresh_contraction = 0.0;
  Check(problem.Solve() ? 0.0 : 1.0, 0.0, label + " solve");
  Check(problem.Displacement().ComputeL2Error(affine), 1e-9,
        label + " the affine displacement");
}

// Without loads the reference is the solution, whatever the equilibrium
// stress.
void RunNoLoad(int dim, int order, int type, const std::string& label) {
  auto smesh = MakeBody(dim, type, dim == 2 ? 4 : 2);
  auto pmesh = ParMesh(MPI_COMM_WORLD, smesh);
  auto fec = H1_FECollection(order, dim);
  auto fes = ParFiniteElementSpace(&pmesh, &fec, dim);
  auto bottom = Marker(pmesh, 1);
  auto top = Marker(pmesh, TopAttribute(dim));
  auto material = Material(dim);
  auto no_traction_value = Vector(dim);
  no_traction_value = 0.0;
  auto no_traction = VectorConstantCoefficient(no_traction_value);
  auto problem = NonlinearClampedProblem(&fes, material.cubic, bottom,
                                         no_traction, top);
  Check(problem.Solve() ? 0.0 : 1.0, 0.0, label + " solve");
  Check(problem.Iterations() == 1 ? 0.0 : 1.0, 0.0, label + " one iteration");
  Check(L2Norm(problem.Displacement()), 0.0, label + " zero displacement");
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();

  RunCase(2, 2, 0, "2d-o2-tri");
  RunCase(2, 1, 1, "2d-o1-quad");
  RunCase(3, 1, 1, "3d-o1-hex");
  RunCase(3, 2, 0, "3d-o2-tet");
  RunAffinePatch(2, 2, 1, "2d-o2-quad patch");
  RunAffinePatch(3, 1, 0, "3d-o1-tet patch");
  RunNoLoad(2, 2, 0, "2d-o2-tri no load");
  RunNoLoad(3, 1, 1, "3d-o1-hex no load");

  if (Mpi::Root()) {
    std::cout << num_checks - num_fails << " of " << num_checks
              << " checks passed.\n";
  }
  return num_fails == 0 ? 0 : 1;
}
