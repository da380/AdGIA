/*
  Parallel tests for LinearQuasiStaticMixedSelfGravitatingProblem with fluid
  regions, on the three-layer meshes with ONE disconnected ParSubMesh for
  the inner core and the mantle and the outer core as a FluidRegion. Run
  with 1, 2 and 4 ranks; a standalone MPI program returning the number of
  failed checks.

  Every rank also builds the serial problem on the full mesh; the parallel
  and serial solutions are compared through partition-independent
  quantities (global L2 norms of the displacement, the potential and the
  background potential, the rigid-mode and region-rotation residuals, the
  potential-block Ritz values), for both solver types under the combined
  surface and tidal load, for its time scaling, and with a relaxation
  weight.
*/

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "MixedProblemTestCommon.hpp"
#include "mfem.hpp"
#include "AdGIA.hpp"

using namespace mfem;
using namespace AdGIA;
using namespace self_grav_test;

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

// The problem on either a serial or a parallel pair of spaces; owns the
// coefficients.
struct Setup {
  ConstantCoefficient kappa{kKappa}, mu{kMu}, tau{1.0};
  FunctionCoefficient rho_s{SolidDensity}, rho_f{FluidDensity},
      sigma{SurfaceLoad}, psi{TidalPotential};
  std::unique_ptr<IsotropicMaxwellRheology> rheology;
  std::vector<FluidRegion> fluids;
  Array<int> surface, inner_core{Array<int>({1})};
  std::unique_ptr<LinearQuasiStaticMixedSelfGravitatingProblem> problem;

  Setup(FiniteElementSpace& fes_u, FiniteElementSpace& fes_phi,
        Mesh& solid) {
    const int dim = solid.Dimension();
    // Maxwell body: unrelaxed modulus mu, weight beta gives beta mu.
    rheology = std::make_unique<IsotropicMaxwellRheology>(
        IsotropicMaxwellRheology::Maxwell(dim, kappa, mu, tau));
    surface = SurfaceMarker(solid);
    fluids.push_back(OuterCore(solid, rho_f));
    problem = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        &fes_u, &fes_phi, *rheology, rho_s, kG, kDtNDegree, nullptr, fluids);
    problem->SetSurfaceLoad(sigma, surface);
    problem->SetTidalPotential(psi);
    problem->AddRegionRotations(inner_core);
    problem->SetRelTol(1e-11);
  }
};

void RunCase(int dim, int order, const std::string& label) {
  Mesh smesh(ThreeLayerMeshFile(dim).c_str(), 1, 1);
  Array<int> solid_attrs({1, 3});

  // Serial reference on every rank.
  SubMesh ssolid(SubMesh::CreateFromDomain(smesh, solid_attrs));
  H1_FECollection sfec(order, dim);
  FiniteElementSpace sfes_u(&ssolid, &sfec, dim), sfes_phi(&smesh, &sfec);
  Setup ser(sfes_u, sfes_phi, ssolid);
  ser.problem->AssembleForce(0.0);
  Check(ser.problem->Solve() ? 0.0 : 1.0, 0.0, label + " serial solve");
  const double u_ref = L2Norm(ser.problem->Displacement());
  const double phi_ref = L2Norm(ser.problem->Potential());
  const double phi0_ref = L2Norm(ser.problem->BackgroundPotential());
  const auto res_ref = ser.problem->RigidModeResiduals();
  double hi_ref = 0.0;
  const double lo_ref = ser.problem->PotentialBlockMinEigenvalue(30, &hi_ref);

  // Parallel problem.
  ParMesh pmesh(MPI_COMM_WORLD, smesh);
  ParSubMesh solid(ParSubMesh::CreateFromDomain(pmesh, solid_attrs));
  H1_FECollection fec(order, dim);
  ParFiniteElementSpace fes_u(&solid, &fec, dim), fes_phi(&pmesh, &fec);
  Setup par(fes_u, fes_phi, solid);
  auto& p = *par.problem;
  Check(p.IsParallel() ? 0.0 : 1.0, 0.0, label + " parallel spaces");
  Check(p.HasFluidRegions() ? 0.0 : 1.0, 0.0, label + " fluid regions");

  Check(RelErr(L2Norm(p.BackgroundPotential()), phi0_ref), 1e-9,
        label + " background potential");

  const auto res = p.RigidModeResiduals();
  Check(res.size() == res_ref.size() ? 0.0 : 1.0, 0.0,
        label + " number of modes");
  for (size_t i = 0; i < res.size() && i < res_ref.size(); i++) {
    Check(RelErr(res[i], res_ref[i]), 1e-6,
          label + " mode residual " + std::to_string(i));
  }

  // The Lanczos start vector depends on the dof numbering, so the (upper)
  // estimate of the smallest eigenvalue after 30 steps is only loosely
  // comparable; the sign is the point, and the largest Ritz value converges
  // fast.
  double hi = 0.0;
  const double lo = p.PotentialBlockMinEigenvalue(30, &hi);
  Check(lo > 0.0 ? 0.0 : 1.0, 0.0, label + " potential block positive");
  Check(RelErr(hi, hi_ref), 0.2, label + " largest Ritz value");
  Check(lo < 3.0 * lo_ref ? 0.0 : 1.0, 0.0, label + " smallest Ritz value");

  for (auto type :
       {LinearQuasiStaticMixedSelfGravitatingProblem::SolverType::BlockMINRES,
        LinearQuasiStaticMixedSelfGravitatingProblem::SolverType::SchurCG}) {
    const std::string name =
        type == LinearQuasiStaticMixedSelfGravitatingProblem::SolverType::BlockMINRES
            ? " minres"
            : " schur";
    p.SetSolverType(type);
    p.AssembleForce(0.0);
    Check(p.Solve() ? 0.0 : 1.0, 0.0, label + name + " solve");
    Check(RelErr(L2Norm(p.Displacement()), u_ref), 1e-7,
          label + name + " |u|");
    Check(RelErr(L2Norm(p.Potential()), phi_ref), 1e-7,
          label + name + " |phi|");
  }

  // Time scaling of both loads (surface and tidal), in parallel.
  p.SetSolverType(
      LinearQuasiStaticMixedSelfGravitatingProblem::SolverType::BlockMINRES);
  p.AssembleForce(2.0);
  Check(p.Solve() ? 0.0 : 1.0, 0.0, label + " solve at t = 2");
  Check(RelErr(L2Norm(p.Displacement()), 3.0 * u_ref), 1e-8,
        label + " time scaling |u|");
  Check(RelErr(L2Norm(p.Potential()), 3.0 * phi_ref), 1e-8,
        label + " time scaling |phi|");

  // Relaxation weight, compared with the serial problem.
  ConstantCoefficient softer(0.5);
  ser.problem->SetRelaxationWeights({&softer});
  Check(ser.problem->Solve() ? 0.0 : 1.0, 0.0, label + " serial soft solve");
  p.AssembleForce(0.0);
  p.SetRelaxationWeights({&softer});
  Check(p.Solve() ? 0.0 : 1.0, 0.0, label + " soft solve");
  Check(RelErr(L2Norm(p.Displacement()),
               L2Norm(ser.problem->Displacement())),
        1e-8, label + " soft |u|");
  p.ClearRelaxationWeights();
  Check(p.Solve() ? 0.0 : 1.0, 0.0, label + " restored solve");
  Check(RelErr(L2Norm(p.Displacement()), u_ref), 1e-8,
        label + " restored |u|");
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();

  RunCase(2, 1, "2d-o1");
  RunCase(2, 2, "2d-o2");
  RunCase(3, 1, "3d-o1");

  if (Mpi::Root()) {
    if (num_fails == 0) {
      std::cout << "All " << num_checks << " checks passed on "
                << Mpi::WorldSize() << " ranks.\n";
    } else {
      std::cout << num_fails << " of " << num_checks << " checks failed on "
                << Mpi::WorldSize() << " ranks.\n";
    }
  }
  return num_fails == 0 ? 0 : 1;  // an exit status is taken modulo 256
}
