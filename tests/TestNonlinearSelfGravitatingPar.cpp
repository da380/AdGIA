/*
  Parallel tests for NonlinearSelfGravitatingProblem
  (nonlinear_problem.hpp), on the disc of TestNonlinearSelfGravitating.cpp.
  Run with 1, 2 and 4 ranks; a standalone MPI program that fails if any
  check does.

  Every rank also builds the serial problem on the whole mesh; the
  parallel and serial solutions are compared through quantities that do
  not depend on the partition (the L2 norms of the displacement and of
  the potential's change, the corrections of each iteration). In
  parallel as well: the residual against differences of the energy and
  the tangent against differences of the residual, at a state that is
  not the reference; the reference as the solution without a load; and
  a relabelled disc under every kind of load. On more than one rank, the
  first case again with one rank holding the buffer and none of the body.
*/

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <numbers>
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

constexpr double kG = 1.0, kRho = 1.0, kLambda = 5.0, kMu = 5.0;
constexpr int kDtNDegree = 8;

double Pressure(const Vector& x) {
  return std::numbers::pi * kG * kRho * kRho * (1.0 - x * x);
}

double Load(const Vector& x) {
  const double c = x[1] / x.Norml2();
  return 0.03 * (1.0 + 0.5 * (3.0 * c * c - 1.0));
}

double L2Norm(const GridFunction& u) {
  const int vdim = u.FESpace()->GetVDim();
  if (vdim == 1) {
    ConstantCoefficient z(0.0);
    return u.ComputeL2Error(z);
  }
  Vector zero(vdim);
  zero = 0.0;
  VectorConstantCoefficient z(zero);
  return u.ComputeL2Error(z);  // global for a ParGridFunction
}

// The material: the quadratic law about the hydrostatic disc and the law
// with the terms of third order.
struct Laws {
  ConstantCoefficient rho{kRho}, lambda{kLambda}, mu{kMu};
  IsotropicElasticTensorCoefficient Xi{2, lambda, mu};
  MatrixFunctionCoefficient T0{2, [](const Vector& x, DenseMatrix& S) {
                                 S.SetSize(2);
                                 S = 0.0;
                                 S(0, 0) = S(1, 1) = -Pressure(x);
                               }};
  QuadraticMaterialLaw quadratic{2, Xi, &T0};
  ConstantCoefficient nu1{30.0}, nu2{6.0}, nu3{-4.0};
  ThirdOrderIsotropicLaw cubic{quadratic, nu1, nu2, nu3};
};

// With `buffer_rank` the last rank holds the buffer and no element of the
// body, which the others share.
void RunCase(int order, const std::string& label, bool buffer_rank = false) {
  Mesh smesh("../data/disc_buffer_2d.msh", 1, 1);
  Array<int> body_attr({1}), buffer_attr({2});
  Vector lo, hi;
  smesh.GetBoundingBox(lo, hi);
  const double r_out = hi.Normlinf();
  H1_FECollection fec(order, 2);
  Laws laws;
  FunctionCoefficient sigma(Load);

  // Serial reference on every rank.
  double u_ref = 0.0, z_ref = 0.0;
  std::vector<double> steps_ref;
  {
    SubMesh body(SubMesh::CreateFromDomain(smesh, body_attr));
    SubMesh buffer(SubMesh::CreateFromDomain(smesh, buffer_attr));
    FiniteElementSpace fes_u(&body, &fec, 2), fes_zeta(&smesh, &fec);
    FiniteElementSpace fes_buffer(&buffer, &fec, 2);
    auto E = NewRadialVacuumExtension(fes_u, fes_buffer, 1.0, r_out);
    Array<int> surface(body.bdr_attributes.Max());
    surface = 0;
    surface[body.bdr_attributes.Max() - 1] = 1;
    NonlinearSelfGravitatingProblem serial(&fes_u, &fes_zeta, &fes_buffer, *E,
                                           laws.cubic, laws.rho, kG,
                                           kDtNDegree);
    serial.SetSurfaceLoad(sigma, surface);
    serial.Options().refresh_contraction = 0.0;
    Check(serial.Solve() ? 0.0 : 1.0, 0.0, label + " serial solve");
    u_ref = L2Norm(serial.Displacement());
    GridFunction dz(serial.Potential());
    dz -= serial.ReferencePotential();
    z_ref = L2Norm(dz);
    steps_ref = serial.Corrections();
  }

  Array<int> partitioning;
  if (buffer_rank) {
    const int ranks = Mpi::WorldSize();
    partitioning.SetSize(smesh.GetNE());
    int next = 0;
    for (int e = 0; e < smesh.GetNE(); e++) {
      partitioning[e] = smesh.GetAttribute(e) == 2
                            ? ranks - 1
                            : (next++ * (ranks - 1)) / smesh.GetNE();
    }
  }
  ParMesh pmesh(MPI_COMM_WORLD, smesh,
                buffer_rank ? partitioning.GetData() : nullptr);
  ParSubMesh body(ParSubMesh::CreateFromDomain(pmesh, body_attr));
  ParSubMesh buffer(ParSubMesh::CreateFromDomain(pmesh, buffer_attr));
  ParFiniteElementSpace fes_u(&body, &fec, 2), fes_zeta(&pmesh, &fec);
  ParFiniteElementSpace fes_buffer(&buffer, &fec, 2);
  auto E = NewRadialVacuumExtension(fes_u, fes_buffer, 1.0, r_out);
  Array<int> surface(body.bdr_attributes.Max());
  surface = 0;
  surface[body.bdr_attributes.Max() - 1] = 1;
  if (buffer_rank && Mpi::WorldRank() == Mpi::WorldSize() - 1) {
    Check(body.GetNE(), 0.0, label + " the last rank holds no body element");
  }

  {
    NonlinearSelfGravitatingProblem problem(&fes_u, &fes_zeta, &fes_buffer,
                                            *E, laws.cubic, laws.rho, kG,
                                            kDtNDegree);
    problem.SetSurfaceLoad(sigma, surface);
    problem.Options().refresh_contraction = 0.0;
    Check(problem.IsParallel() ? 0.0 : 1.0, 0.0, label + " parallel spaces");
    Check(problem.Solve() ? 0.0 : 1.0, 0.0, label + " solve");
    Check(RelErr(L2Norm(problem.Displacement()), u_ref), 1e-7, label + " |u|");
    ParGridFunction dz(static_cast<ParGridFunction&>(problem.Potential()));
    dz -= problem.ReferencePotential();
    Check(RelErr(L2Norm(dz), z_ref), 1e-7, label + " |zeta - zeta0|");
    const auto& steps = problem.Corrections();
    Check(steps.size() == steps_ref.size() ? 0.0 : 1.0, 0.0,
          label + " iterations");
    for (auto k = 0u; k < steps.size() && k < steps_ref.size(); k++) {
      if (steps_ref[k] > 1e-8 * u_ref) {
        Check(RelErr(steps[k], steps_ref[k]), 1e-4,
              label + " correction " + std::to_string(k + 1));
      }
    }
  }

  // Derivatives at a state that is not the reference, by the fourth-order
  // central difference.
  {
    NonlinearSelfGravitatingProblem problem(&fes_u, &fes_zeta, &fes_buffer,
                                            *E, laws.cubic, laws.rho, kG,
                                            kDtNDegree);
    problem.SetSurfaceLoad(sigma, surface);
    VectorFunctionCoefficient u_c(
        2, [](const Vector& x, Vector& u) { SmoothField(x, u, 0.02, 0.3); });
    VectorFunctionCoefficient v_c(
        2, [](const Vector& x, Vector& u) { SmoothField(x, u, 1.0, 1.1); });
    VectorFunctionCoefficient w_c(
        2, [](const Vector& x, Vector& u) { SmoothField(x, u, 1.0, 2.7); });
    FunctionCoefficient z_c([](const Vector& x) {
      return 0.05 * std::sin(1.7 * x[0] + 0.4) * std::cos(1.1 * x[1] - 0.2);
    });
    FunctionCoefficient chi_c([](const Vector& x) {
      return std::cos(1.3 * x[0] - 0.6) * std::sin(0.9 * x[1] + 0.5);
    });
    FunctionCoefficient psi_c(
        [](const Vector& x) { return std::sin(0.8 * x[0] + 1.9 * x[1]); });
    ParGridFunction u0(&fes_u), v(&fes_u), w(&fes_u);
    ParGridFunction z0(&fes_zeta), chi(&fes_zeta), psi(&fes_zeta);
    u0.ProjectCoefficient(u_c);
    v.ProjectCoefficient(v_c);
    w.ProjectCoefficient(w_c);
    z0.ProjectCoefficient(z_c);
    z0 += problem.ReferencePotential();
    chi.ProjectCoefficient(chi_c);
    psi.ProjectCoefficient(psi_c);
    Vector v_t(fes_u.GetTrueVSize()), w_t(fes_u.GetTrueVSize());
    Vector chi_t(fes_zeta.GetTrueVSize()), psi_t(fes_zeta.GetTrueVSize());
    v.GetTrueDofs(v_t);
    w.GetTrueDofs(w_t);
    chi.GetTrueDofs(chi_t);
    psi.GetTrueDofs(psi_t);
    auto pair = [&](const Vector& a_u, const Vector& a_zeta) {
      return InnerProduct(MPI_COMM_WORLD, a_u, v_t) +
             InnerProduct(MPI_COMM_WORLD, a_zeta, chi_t);
    };
    auto set = [&](double step, const GridFunction& du,
                   const GridFunction& dz) {
      problem.Displacement() = u0;
      problem.Displacement().Add(step, du);
      problem.Potential() = z0;
      problem.Potential().Add(step, dz);
    };
    const double h = 1e-4;
    auto derivative = [&](const GridFunction& du, const GridFunction& dz,
                          auto&& value) {
      auto at = [&](double step) {
        set(step, du, dz);
        return value();
      };
      return (8.0 * (at(h) - at(-h)) - (at(2.0 * h) - at(-2.0 * h))) /
             (12.0 * h);
    };

    set(0.0, v, chi);
    Vector r_u, r_zeta;
    problem.Residual(r_u, r_zeta);
    const double rv = pair(r_u, r_zeta);
    const double dE = derivative(v, chi, [&] { return problem.Energy(); });
    Check(RelErr(dE, rv), 1e-7, label + " residual is the energy's variation");

    set(0.0, v, chi);
    Vector k_u, k_zeta;
    problem.Tangent().ApplyBlockOperator(w_t, psi_t, k_u, k_zeta);
    const double vKw = pair(k_u, k_zeta);
    const double dr = derivative(w, psi, [&] {
      Vector a_u, a_zeta;
      problem.Residual(a_u, a_zeta);
      return pair(a_u, a_zeta);
    });
    Check(RelErr(dr, vKw), 1e-7,
          label + " tangent is the residual's variation");
  }

  // Without a load the reference is the solution.
  {
    NonlinearSelfGravitatingProblem problem(&fes_u, &fes_zeta, &fes_buffer,
                                            *E, laws.cubic, laws.rho, kG,
                                            kDtNDegree);
    Check(problem.Solve() ? 0.0 : 1.0, 0.0, label + " no load: solve");
    Check(problem.Iterations() == 1 ? 0.0 : 1.0, 0.0,
          label + " no load: one iteration");
    Check(problem.Displacement().Normlinf(), 1e-12,
          label + " no load: zero displacement");
  }
}

// A relabelled disc under a surface mass, a body force and a load on the
// potential's row: the equilibrium mapping and the loads in parallel,
// against the serial problem.
void RunRelabelled(const std::string& label) {
  Mesh smesh("../data/disc_buffer_2d.msh", 1, 1);
  Array<int> body_attr({1}), buffer_attr({2});
  Vector lo, hi;
  smesh.GetBoundingBox(lo, hi);
  const double r_out = hi.Normlinf();
  H1_FECollection fec(2, 2);
  Laws laws;
  FunctionCoefficient sigma(Load);

  // The mapping moves the points within r < 0.8.
  auto bump = [](const Vector& x, Vector* gradient) {
    const double q = 1.0 - (x * x) / 0.64;
    if (gradient) {
      gradient->SetSize(2);
      *gradient = 0.0;
      if (q > 0.0) {
        gradient->Add(-6.0 * q * q / 0.64, x);
      }
    }
    return q > 0.0 ? q * q * q : 0.0;
  };
  CallableDiffeomorphism relabelling(
      2,
      [bump](const Vector& x, Vector& y) {
        const double b = bump(x, nullptr);
        y = x;
        y[0] += 0.06 * b;
        y[1] += 0.024 * b;
      },
      [bump](const Vector& x, DenseMatrix& F) {
        Vector g;
        bump(x, &g);
        F.SetSize(2);
        F(0, 0) = 1.0 + 0.06 * g[0];
        F(0, 1) = 0.06 * g[1];
        F(1, 0) = 0.024 * g[0];
        F(1, 1) = 1.0 + 0.024 * g[1];
      });
  TransformedMatrixFunctionCoefficient T0(
      2, relabelling, [](const Vector& x, DenseMatrix& S) {
        S.SetSize(2);
        S = 0.0;
        S(0, 0) = S(1, 1) = -Pressure(x);
      });
  QuadraticMaterialLaw quadratic(2, laws.Xi, &T0);
  ThirdOrderIsotropicLaw cubic(quadratic, laws.nu1, laws.nu2, laws.nu3);
  JacobianCoefficient jacobian(relabelling);
  ProductCoefficient density(kRho, jacobian);
  // Loads with no net force on the body: a force odd in x_0, and a source
  // of the potential's row with no net mass.
  VectorFunctionCoefficient force(2, [](const Vector& x, Vector& f) {
    f.SetSize(2);
    f[0] = 0.02 * std::sin(2.0 * x[0]);
    f[1] = 0.0;
  });
  FunctionCoefficient source([](const Vector& x) {
    return x * x < 1.0 ? 0.05 * x[0] * x[1] : 0.0;
  });

  double u_ref = 0.0, z_ref = 0.0;
  {
    SubMesh body(SubMesh::CreateFromDomain(smesh, body_attr));
    SubMesh buffer(SubMesh::CreateFromDomain(smesh, buffer_attr));
    FiniteElementSpace fes_u(&body, &fec, 2), fes_zeta(&smesh, &fec);
    FiniteElementSpace fes_buffer(&buffer, &fec, 2);
    auto E = NewRadialVacuumExtension(fes_u, fes_buffer, 1.0, r_out);
    Array<int> surface(body.bdr_attributes.Max());
    surface = 0;
    surface[body.bdr_attributes.Max() - 1] = 1;
    NonlinearSelfGravitatingProblem serial(&fes_u, &fes_zeta, &fes_buffer, *E,
                                           cubic, density, kG, kDtNDegree,
                                           &relabelling);
    serial.SetSurfaceLoad(sigma, surface);
    serial.SetBodyLoad(&force, nullptr);
    serial.SetPotentialLoad(&source, nullptr);
    serial.Options().refresh_contraction = 0.0;
    Check(serial.Solve() ? 0.0 : 1.0, 0.0, label + " serial solve");
    u_ref = L2Norm(serial.Displacement());
    GridFunction dz(serial.Potential());
    dz -= serial.ReferencePotential();
    z_ref = L2Norm(dz);
  }

  ParMesh pmesh(MPI_COMM_WORLD, smesh);
  ParSubMesh body(ParSubMesh::CreateFromDomain(pmesh, body_attr));
  ParSubMesh buffer(ParSubMesh::CreateFromDomain(pmesh, buffer_attr));
  ParFiniteElementSpace fes_u(&body, &fec, 2), fes_zeta(&pmesh, &fec);
  ParFiniteElementSpace fes_buffer(&buffer, &fec, 2);
  auto E = NewRadialVacuumExtension(fes_u, fes_buffer, 1.0, r_out);
  Array<int> surface(body.bdr_attributes.Max());
  surface = 0;
  surface[body.bdr_attributes.Max() - 1] = 1;
  NonlinearSelfGravitatingProblem problem(&fes_u, &fes_zeta, &fes_buffer, *E,
                                          cubic, density, kG, kDtNDegree,
                                          &relabelling);
  problem.SetSurfaceLoad(sigma, surface);
  problem.SetBodyLoad(&force, nullptr);
  problem.SetPotentialLoad(&source, nullptr);
  problem.Options().refresh_contraction = 0.0;
  Check(problem.Solve() ? 0.0 : 1.0, 0.0, label + " solve");
  Check(RelErr(L2Norm(problem.Displacement()), u_ref), 1e-6, label + " |u|");
  ParGridFunction dz(static_cast<ParGridFunction&>(problem.Potential()));
  dz -= problem.ReferencePotential();
  Check(RelErr(L2Norm(dz), z_ref), 1e-6, label + " |zeta - zeta0|");
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();

  RunCase(1, "o1");
  RunCase(2, "o2");
  RunRelabelled("relabelled");
  if (Mpi::WorldSize() > 1) {
    RunCase(1, "o1, a rank with the buffer alone", true);
  }

  if (Mpi::Root()) {
    std::cout << num_checks - num_fails << " of " << num_checks
              << " checks passed.\n";
  }
  return num_fails == 0 ? 0 : 1;
}
