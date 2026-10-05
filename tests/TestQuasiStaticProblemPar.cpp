/*
  Parallel tests for LinearQuasiStaticProblemBase /
  LinearQuasiStaticTractionProblem / LinearQuasiStaticClampedProblem on
  ParFiniteElementSpaces. Run with 1, 2 and 4 ranks; a standalone MPI program
  that prints a summary and exits non-zero if any check fails.

  - Clamped problem: every rank also solves the serial problem on the full
    mesh and compares a partition-independent quantity, the L2 norm of the
    displacement (plain, with a relaxation-weight field on an L2 space, and
    after ClearRelaxationWeights).
  - Traction problem: the exact uniaxial strain on every rank's elements,
    at t = 0 and t = 1.
  - The non-natural reference state: the mapped traction problem equals
    the identity-mapped problem on the nodal-image mesh dof for dof, and
    the mapped rigid modes are discrete null vectors (as in the serial
    TestQuasiStaticProblem).
*/

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <numbers>
#include <string>

#include "QuasiStaticTestCommon.hpp"
#include "mfem.hpp"
#include "AdGIA.hpp"

using namespace mfem;
using namespace AdGIA;
using namespace elastic_test;

namespace {

int num_checks = 0;
int num_fails = 0;

double GlobalMax(double v) {
  double g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return g;
}

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

double L2Norm(const GridFunction& u) {
  Vector zero(u.FESpace()->GetVDim());
  zero = 0.0;
  VectorConstantCoefficient z(zero);
  return u.ComputeL2Error(z);  // global for ParGridFunction
}

void RunCase(int dim, int elementType, int order, const std::string& label) {
  auto smesh = MakeSmallMesh(dim, elementType);
  const auto x0_attr = BdrAttributeAt(smesh, 0, 0.0);
  const auto x1_attr = BdrAttributeAt(smesh, 0, 1.0);
  const auto nbdr = smesh.bdr_attributes.Max();

  int nxyz[3] = {Mpi::WorldSize(), 1, 1};
  int* partitioning = smesh.CartesianPartitioning(nxyz);
  ParMesh pmesh(MPI_COMM_WORLD, smesh, partitioning);
  delete[] partitioning;

  H1_FECollection fec(order, dim);
  FiniteElementSpace sfes(&smesh, &fec, dim);
  ParFiniteElementSpace pfes(&pmesh, &fec, dim);

  ConstantCoefficient kappa(Kappa(dim)), mu(kMu), tau(1.0);
  // Maxwell body (mu_inf = 0, one branch): the unrelaxed modulus is mu and
  // a relaxation weight beta gives mu_eff = beta mu.
  auto rheology = IsotropicMaxwellRheology::Maxwell(dim, kappa, mu, tau);
  auto ess_bdr = Marker(nbdr, {x0_attr});
  auto pull_marker = Marker(nbdr, {x1_attr});
  auto uni_marker = Marker(nbdr, {x0_attr, x1_attr});
  VectorFunctionCoefficient pull(dim, PullTraction);
  VectorFunctionCoefficient uni(dim, UniaxialTraction);

  // Clamped: serial reference norm vs parallel norm, then with a
  // relaxation-weight field on an L2 space.
  {
    LinearQuasiStaticClampedProblem serial(&sfes, rheology, ess_bdr, pull,
                                           pull_marker);
    LinearQuasiStaticClampedProblem par(&pfes, rheology, ess_bdr, pull,
                                        pull_marker);
    Check(par.IsParallel() ? 0.0 : 1.0, 0.0, label + ": IsParallel");

    serial.AssembleForce(0.25);
    Check(serial.Solve() ? 0.0 : 1.0, 0.0, label + ": serial solve");
    par.AssembleForce(0.25);
    Check(par.Solve() ? 0.0 : 1.0, 0.0, label + ": parallel solve");
    const auto ns = L2Norm(serial.Displacement());
    const auto np = L2Norm(par.Displacement());
    Check(GlobalMax(std::abs(ns - np) / ns), 1e-8, label + ": clamped norm");

    L2_FECollection l2fec(0, dim);
    FiniteElementSpace ssfes(&smesh, &l2fec);
    ParFiniteElementSpace psfes(&pmesh, &l2fec);
    FunctionCoefficient mu_var(
        [](const Vector& x) { return 0.3 + 0.6 * x[0]; });
    GridFunction smu(&ssfes);
    ParGridFunction pmu(&psfes);
    smu.ProjectCoefficient(mu_var);
    pmu.ProjectCoefficient(mu_var);
    GridFunctionCoefficient smu_c(&smu), pmu_c(&pmu);
    serial.SetRelaxationWeights({&smu_c});
    par.SetRelaxationWeights({&pmu_c});
    serial.AssembleForce(0.25);
    Check(serial.Solve() ? 0.0 : 1.0, 0.0, label + ": serial solve (eff)");
    par.AssembleForce(0.25);
    Check(par.Solve() ? 0.0 : 1.0, 0.0, label + ": parallel solve (eff)");
    const auto ns_e = L2Norm(serial.Displacement());
    const auto np_e = L2Norm(par.Displacement());
    Check(GlobalMax(std::abs(ns_e - np_e) / ns_e), 1e-8,
          label + ": clamped norm, effective modulus");
    Check(std::abs(ns_e - ns) / ns > 1e-3 ? 0.0 : 1.0, 0.0,
          label + ": effective modulus changed the solution");

    par.ClearRelaxationWeights();
    par.AssembleForce(0.25);
    Check(par.Solve() ? 0.0 : 1.0, 0.0, label + ": parallel solve (clear)");
    Check(GlobalMax(std::abs(L2Norm(par.Displacement()) - ns) / ns), 1e-8,
          label + ": clamped norm after Clear");
  }

  // Traction: exact uniaxial strain on every rank's elements.
  {
    LinearQuasiStaticTractionProblem par(&pfes, rheology, uni, uni_marker);
    double exx = 0.0, eyy = 0.0;
    UniaxialStrain(dim, kSigma, exx, eyy);
    par.AssembleForce(0.0);
    Check(par.Solve() ? 0.0 : 1.0, 0.0, label + ": traction solve");
    Check(GlobalMax(MaxStrainError(par.Displacement(), exx, eyy)) / exx, 1e-8,
          label + ": uniaxial strain");
    par.AssembleForce(1.0);
    Check(par.Solve() ? 0.0 : 1.0, 0.0, label + ": traction solve, t = 1");
    Check(GlobalMax(MaxStrainError(par.Displacement(), 2 * exx, 2 * eyy)) / exx,
          1e-8, label + ": uniaxial strain, t = 1");
  }
}

// The non-natural reference state in parallel, as in the serial test: the
// mapped traction problem on the reference ParMesh equals, dof for dof,
// the identity-mapped problem on the nodal-image ParMesh (same partition
// on both sides), and the projector's mapped rotations are discrete null
// vectors of the mapped stiffness.
void RunMappedCase(int dim, int order, const std::string& label) {
  constexpr double kPi = std::numbers::pi;
  auto smesh = MakeSmallMesh(dim, 0);
  smesh.SetCurvature(order);
  int nxyz[3] = {Mpi::WorldSize(), 1, 1};
  int* partitioning = smesh.CartesianPartitioning(nxyz);
  ParMesh pmesh(MPI_COMM_WORLD, smesh, partitioning);
  delete[] partitioning;

  H1_FECollection fec(order, dim);
  ParFiniteElementSpace pfes(&pmesh, &fec, dim);

  const double c = 0.05;
  CallableDiffeomorphism xi(
      dim,
      [c, dim](const Vector& x, Vector& y) {
        for (int i = 0; i < dim; i++) {
          y(i) = x(i) + c * std::sin(kPi * x((i + 1) % dim));
        }
      },
      [c, dim](const Vector& x, DenseMatrix& F) {
        F = 0.0;
        for (int i = 0; i < dim; i++) {
          F(i, i) = 1.0;
          F(i, (i + 1) % dim) = c * kPi * std::cos(kPi * x((i + 1) % dim));
        }
      });
  auto xi_h = Interpolate(xi, pmesh);
  auto mapped = MappedMesh(pmesh, xi);
  ParFiniteElementSpace pfes_mapped(&mapped, &fec, dim);
  CallableDiffeomorphism id(
      dim, [](const Vector& x, Vector& y) { y = x; },
      [dim](const Vector&, DenseMatrix& F) {
        F = 0.0;
        for (int i = 0; i < dim; i++) {
          F(i, i) = 1.0;
        }
      });

  ConstantCoefficient lam(kLambda), mu(kMu);
  IsotropicElasticTensorCoefficient C(dim, lam, mu);
  DenseMatrix zero_mat(dim);
  zero_mat = 0.0;
      MatrixConstantCoefficient S0(zero_mat);
  // The rheology takes the *referential* description: the relabelled
  // tensor (C is constant, so the composition C o xi is C itself).
  RelabelledElasticTensorCoefficient C_rel(dim, C, xi_h);
  ReferentialElasticRheology rheo_ref(dim, C_rel, S0, xi_h);
  ReferentialElasticRheology rheo_map(dim, C, S0, id);

  Vector tv(dim);
  tv = 0.3;
  tv(0) = 1.0;
  VectorConstantCoefficient t_phys(tv);
  NansonAreaCoefficient area(xi_h);
  ScalarVectorProductCoefficient t_ref(area, t_phys);

  Array<int> all_bdr(smesh.bdr_attributes.Max());
  all_bdr = 1;

  LinearQuasiStaticTractionProblem p_ref(&pfes, rheo_ref, t_ref, all_bdr);
  LinearQuasiStaticTractionProblem p_map(&pfes_mapped, rheo_map, t_phys,
                                         all_bdr);
  p_ref.AssembleForce(0.0);
  Check(p_ref.Solve() ? 0.0 : 1.0, 0.0, label + ": mapped solve");
  p_map.AssembleForce(0.0);
  Check(p_map.Solve() ? 0.0 : 1.0, 0.0, label + ": image solve");

  // Same partition on both sides: local dof vectors compare entrywise.
  Vector d(p_ref.Displacement());
  d -= p_map.Displacement();
  const double rel = GlobalMax(d.Normlinf()) /
                     std::max(GlobalMax(p_map.Displacement().Normlinf()),
                              1e-300);
  Check(rel, 1e-8, label + ": mapped identity");

  for (auto r : p_ref.RigidPairResiduals()) {
    Check(GlobalMax(r), 1e-10, label + ": mapped rigid residual");
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();

  for (auto dim : {2, 3}) {
    for (auto elementType : {0, 1}) {
      for (auto order : {1, 2}) {
        auto label = "dim=" + std::to_string(dim) +
                     " et=" + std::to_string(elementType) +
                     " p=" + std::to_string(order);
        RunCase(dim, elementType, order, label);
      }
    }
    for (auto order : {1, 2}) {
      auto label = "mapped dim=" + std::to_string(dim) +
                   " p=" + std::to_string(order);
      RunMappedCase(dim, order, label);
    }
  }

  if (Mpi::Root()) {
    if (num_fails == 0) {
      std::cout << "All " << num_checks << " checks passed on "
                << Mpi::WorldSize() << " ranks.\n";
    } else {
      std::cout << num_fails << " of " << num_checks << " checks FAILED on "
                << Mpi::WorldSize() << " ranks.\n";
    }
  }
  return num_fails == 0 ? 0 : 1;
}
