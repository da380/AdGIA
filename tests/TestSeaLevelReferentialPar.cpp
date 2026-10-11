/*
  Parallel twin of TestSeaLevelReferential.cpp (WP7 phase A,
  doc/planning/referential_sea_level_plan.md): the referential water
  load on the 3-D two-layer mesh, all-ocean, phi_e = id. Run with 1, 2
  and 4 ranks; a standalone MPI program that exits with status 1 if any
  check fails.

  Every rank also solves the serial problem on the full mesh: serial
  and parallel build the same discrete system, so the global L2 norms
  of displacement and potential and the scalar Phi_g are compared
  directly. The parallel run then re-checks the mass-conservation
  certificate and the inert uniform term on its own integrals.
*/

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <numbers>
#include <string>

#include "mfem.hpp"
#include "AdGIA.hpp"

using namespace mfem;
using namespace AdGIA;

namespace {

constexpr double kG = 0.05;
constexpr double kRho = 1.0;
constexpr double kKappa = 1.0;
constexpr double kMu = 0.5;
constexpr int kDtNDegree = 12;
constexpr double kRhoW = 0.02;

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

double RelErr(double a, double b) {
  return std::abs(a - b) / std::max(1e-30, std::abs(b));
}

double G0() { return 4.0 * std::numbers::pi * kG * kRho / 3.0; }

double Degree2(const Vector& x) {
  const double r = x.Norml2();
  const double c = x[2] / r;
  return 3.0 * c * c - 1.0;
}

// Degree-2 pattern plus a mean, so that Phi_g is active, plus an
// order-1 tesseral part so that the wander rows are driven.
double SigmaWithMean(const Vector& x) {
  const double r2 = x * x;
  return Degree2(x) + 2.0 * x[0] * x[2] / r2 + 0.5;
}

double UniformBallPressure(const Vector& x) {
  const double r2 = x * x;
  return std::max(0.0, 2.0 * std::numbers::pi * kG * kRho * kRho *
                           (1.0 - r2) / 3.0);
}

CallableDiffeomorphism IdentityMap(int dim) {
  return CallableDiffeomorphism(
      dim, [](const Vector& x, Vector& y) { y = x; },
      [](const Vector&, DenseMatrix& F) {
        F = 0.0;
        for (int i = 0; i < F.Height(); i++) {
          F(i, i) = 1.0;
        }
      });
}

double L2Norm(const GridFunction& u) {
  const int vdim = u.FESpace()->GetVDim();
  if (vdim == 1) {
    ConstantCoefficient z(0.0);
    return const_cast<GridFunction&>(u).ComputeL2Error(z);
  }
  Vector zero(vdim);
  zero = 0.0;
  VectorConstantCoefficient z(zero);
  return const_cast<GridFunction&>(u).ComputeL2Error(z);
}

double GlobalSum(double v) {
  double g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return g;
}

// With true, water and the traditional rotation border compose
// (moments away from near-neutral wander, the WP4 lesson).
void RunCase(int order, const std::string& label, bool rotation = false) {
  const char* mesh_file = "../data/coupled_poisson.msh";
  Mesh smesh(mesh_file, 1, 1);
  const int dim = smesh.Dimension();
  Array<int> body_attr({1}), buffer_attr({2});

  Vector bb_min, bb_max;
  smesh.GetBoundingBox(bb_min, bb_max);
  const double r_out = bb_max.Normlinf();

  auto phi = IdentityMap(dim);
  ConstantCoefficient kappa(kKappa), mu(kMu), rho(kRho);
  FunctionCoefficient p0(UniformBallPressure);
  auto C_eff =
      IsotropicElasticTensorCoefficient::FromBulkModulus(dim, kappa, mu);
  BareElasticTensorCoefficient C(dim, C_eff, p0);
  MatrixFunctionCoefficient S(dim, [](const Vector& x, DenseMatrix& Se) {
    Se.SetSize(x.Size());
    Se = 0.0;
    const double p = UniformBallPressure(x);
    for (int i = 0; i < x.Size(); i++) {
      Se(i, i) = -p;
    }
  });
  ReferentialElasticRheology rheology(dim, C, S, phi);
  FunctionCoefficient sigma(SigmaWithMean);
  ConstantCoefficient w(kRhoW / G0());

  Vector moments(3);
  moments[0] = 1.2;
  moments[1] = 1.3;
  moments[2] = 2.0;
  const double Omega = 0.1;

  // Serial reference on every rank.
  double u_ref = 0.0, z_ref = 0.0, g_ref = 0.0;
  Vector w_ref;
  {
    SubMesh body(SubMesh::CreateFromDomain(smesh, body_attr));
    SubMesh buffer(SubMesh::CreateFromDomain(smesh, buffer_attr));
    H1_FECollection fec(order, dim);
    FiniteElementSpace fes_u(&body, &fec, dim), fes_zeta(&smesh, &fec);
    FiniteElementSpace fes_buffer(&buffer, &fec, dim);
    LinearQuasiStaticReferentialSelfGravitatingProblem problem(
        &fes_u, &fes_zeta, rheology, rho, kG, kDtNDegree);
    auto E = NewRadialVacuumExtension(fes_u, fes_buffer, 1.0, r_out);
    problem.SetPrescribedVacuumExtension(fes_buffer, *E);
    Array<int> surface(body.bdr_attributes.Max());
    surface = 0;
    surface[body.bdr_attributes.Max() - 1] = 1;
    problem.SetWaterLoad(w, sigma, surface);
    if (rotation) {
      problem.SetRotation(Omega, moments);
    }
    problem.SetRelTol(1e-12);
    problem.AssembleForce(0.0);
    Check(problem.Solve() ? 0.0 : 1.0, 0.0, label + " serial solve");
    u_ref = L2Norm(problem.Displacement());
    z_ref = L2Norm(problem.Potential());
    g_ref = problem.UniformPotentialTerm();
    if (rotation) {
      w_ref = problem.AngularVelocity();
    }
  }

  // Parallel problem (slab partition, so some ranks can be surface-free
  // at higher rank counts).
  int nxyz[3] = {Mpi::WorldSize(), 1, 1};
  int* partitioning = smesh.CartesianPartitioning(nxyz);
  ParMesh pmesh(MPI_COMM_WORLD, smesh, partitioning);
  delete[] partitioning;
  ParSubMesh body(ParSubMesh::CreateFromDomain(pmesh, body_attr));
  ParSubMesh buffer(ParSubMesh::CreateFromDomain(pmesh, buffer_attr));
  H1_FECollection fec(order, dim);
  ParFiniteElementSpace fes_u(&body, &fec, dim), fes_zeta(&pmesh, &fec);
  ParFiniteElementSpace fes_buffer(&buffer, &fec, dim);
  LinearQuasiStaticReferentialSelfGravitatingProblem problem(
      &fes_u, &fes_zeta, rheology, rho, kG, kDtNDegree);
  auto E = NewRadialVacuumExtension(fes_u, fes_buffer, 1.0, r_out);
  problem.SetPrescribedVacuumExtension(fes_buffer, *E);
  Array<int> surface(body.bdr_attributes.Max());
  surface = 0;
  surface[body.bdr_attributes.Max() - 1] = 1;
  FunctionCoefficient sigma2(SigmaWithMean);
  problem.SetWaterLoad(w, sigma2, surface);
  if (rotation) {
    problem.SetRotation(Omega, moments);
  }
  problem.SetRelTol(1e-12);
  problem.AssembleForce(0.0);
  Check(problem.Solve() ? 0.0 : 1.0, 0.0, label + " parallel solve");
  Check(RelErr(L2Norm(problem.Displacement()), u_ref), 1e-5,
        label + " displacement norm");
  Check(RelErr(L2Norm(problem.Potential()), z_ref), 1e-5,
        label + " potential norm");
  Check(RelErr(problem.UniformPotentialTerm(), g_ref), 1e-5,
        label + " Phi_g");
  if (rotation) {
    Vector dw(problem.AngularVelocity());
    dw -= w_ref;
    Check(dw.Norml2() / w_ref.Norml2(), 1e-5, label + " omega");
  }

  // The parallel run's own mass certificate (route grade, as in the
  // serial suite).
  {
    auto surface_integral = [&](Coefficient& f) {
      H1_FECollection sfec(order, dim);
      ParFiniteElementSpace sfes(&body, &sfec);
      ParLinearForm lf(&sfes);
      lf.AddBoundaryIntegrator(new BoundaryLFIntegrator(f), surface);
      lf.Assemble();
      return GlobalSum(lf.Sum());
    };
    GridFunctionCoefficient zc(
        const_cast<GridFunction*>(&problem.PotentialOnBody()));
    ProductCoefficient wz(w, zc);
    ConstantCoefficient one(1.0);
    ProductCoefficient wone(w, one);
    double total = surface_integral(sigma2) - surface_integral(wz) +
                   problem.UniformPotentialTerm() * surface_integral(wone);
    if (rotation) {
      // The composed load includes -w psi(omega).
      ProductCoefficient wpsi(w, problem.SolutionCentrifugalPotential());
      total -= surface_integral(wpsi);
    }
    Check(std::abs(total), 5e-5, label + " mass certificate");
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();

  RunCase(1, "water o1");
  RunCase(1, "water+rotation o1", true);

  const int fails = num_fails;
  int global_fails = 0;
  MPI_Allreduce(&fails, &global_fails, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (Mpi::Root()) {
    std::cout << (global_fails == 0 ? "PASS" : "FAIL") << " ("
              << num_checks << " checks)\n";
  }
  return global_fails == 0 ? 0 : 1;
}
