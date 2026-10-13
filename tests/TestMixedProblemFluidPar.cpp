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
  weight. The named CMB treatments (FluidRegion::Treatment) are gated
  against the hand-assembled degradations in parallel.
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

// The named CMB treatments against the hand-assembled degradations, in
// parallel: the enum route builds the same coefficients and assembly, so
// it must match the hand route on the same partition to solver noise.
void RunCmbCase(int dim, int order, const std::string& label) {
  Mesh smesh(ThreeLayerMeshFile(dim).c_str(), 1, 1);
  Array<int> solid_attrs({1, 3});
  ParMesh pmesh(MPI_COMM_WORLD, smesh);
  ParSubMesh solid(ParSubMesh::CreateFromDomain(pmesh, solid_attrs));
  H1_FECollection fec(order, dim);
  ParFiniteElementSpace fes_u(&solid, &fec, dim), fes_phi(&pmesh, &fec);

  ConstantCoefficient kappa(kKappa), mu(kMu), zero(0.0);
  ConstantCoefficient rho_c(1.1);  // the core-top density of the profile
  FunctionCoefficient rho_s(SolidDensity), rho_f(FluidDensity),
      sigma(SurfaceLoad);
  IsotropicElasticRheology rheology(dim, kappa, mu);
  Array<int> surface = SurfaceMarker(solid), inner_core({1});

  auto solve = [&](const std::vector<FluidRegion>& fluids,
                   const std::string& what) {
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        &fes_u, &fes_phi, rheology, rho_s, kG, kDtNDegree, nullptr, fluids);
    p->SetSurfaceLoad(sigma, surface);
    p->AddRegionRotations(inner_core);
    p->SetRelTol(1e-11);
    p->AssembleForce(0.0);
    Check(p->Solve() ? 0.0 : 1.0, 0.0, what + " solve");
    return p;
  };

  for (bool winkler : {false, true}) {
    const std::string name = label + (winkler ? " winkler" : " uniform");
    std::vector<FluidRegion> hand{OuterCore(solid, rho_f)};
    hand[0].density_gradient = &zero;
    hand[0].interface_density = &rho_c;
    hand[0].interface_potential_coupling = !winkler;
    auto p_hand = solve(hand, name + " hand");

    std::vector<FluidRegion> named{OuterCore(solid, rho_f)};
    named[0].treatment = winkler ? FluidRegion::Treatment::Winkler
                                 : FluidRegion::Treatment::UniformCore;
    named[0].uniform_density = 1.1;
    auto p_named = solve(named, name + " named");

    Check(RelErr(L2Norm(p_named->Displacement()),
                 L2Norm(p_hand->Displacement())),
          1e-11, name + " |u|");
    Check(RelErr(L2Norm(p_named->Potential()), L2Norm(p_hand->Potential())),
          1e-11, name + " |phi|");
  }
}

// The dropped-inner-core recipe in parallel: a mantle-only ParSubMesh
// with the whole core folded into one UniformCore FluidRegion must keep
// the background potential of the kept-inner-core problem (the folded
// mass still sources Phi_0) and solve.
void RunDroppedInner(int dim, int order, const std::string& label) {
  Mesh smesh(ThreeLayerMeshFile(dim).c_str(), 1, 1);
  ParMesh pmesh(MPI_COMM_WORLD, smesh);
  H1_FECollection fec(order, dim);
  ParFiniteElementSpace fes_phi(&pmesh, &fec);

  ConstantCoefficient kappa(kKappa), mu(kMu);
  FunctionCoefficient rho_s(SolidDensity), rho_f(FluidDensity),
      sigma(SurfaceLoad);
  IsotropicElasticRheology rheology(dim, kappa, mu);

  // Kept inner core: the standard three-layer problem, UniformCore.
  Array<int> solid_attrs({1, 3});
  ParSubMesh solid(ParSubMesh::CreateFromDomain(pmesh, solid_attrs));
  ParFiniteElementSpace fes_u(&solid, &fec, dim);
  std::vector<FluidRegion> meshed{OuterCore(solid, rho_f)};
  meshed[0].treatment = FluidRegion::Treatment::UniformCore;
  meshed[0].uniform_density = 1.1;
  LinearQuasiStaticMixedSelfGravitatingProblem keep(
      &fes_u, &fes_phi, rheology, rho_s, kG, kDtNDegree, nullptr, meshed);
  keep.SetSurfaceLoad(sigma, SurfaceMarker(solid));
  Array<int> inner_core({1});
  keep.AddRegionRotations(inner_core);
  keep.SetRelTol(1e-11);
  keep.AssembleForce(0.0);
  Check(keep.Solve() ? 0.0 : 1.0, 0.0, label + " keep solve");

  // Dropped inner core: mantle-only SubMesh, the core {1, 2} one region.
  Array<int> mantle_attr({3});
  ParSubMesh mantle(ParSubMesh::CreateFromDomain(pmesh, mantle_attr));
  ParFiniteElementSpace fes_m(&mantle, &fec, dim);
  FunctionCoefficient rho_inner(SolidDensity);
  PWCoefficient rho_core;
  rho_core.UpdateCoefficient(1, rho_inner);
  rho_core.UpdateCoefficient(2, rho_f);
  FluidRegion core;
  core.attributes = Array<int>({1, 2});
  core.density = &rho_core;
  core.treatment = FluidRegion::Treatment::UniformCore;
  core.uniform_density = 1.1;
  core.interface_marker.SetSize(mantle.bdr_attributes.Max());
  core.interface_marker = 0;
  core.interface_marker[1] = 1;  // the CMB; the ICB is unmeshed
  std::vector<FluidRegion> folded{core};
  LinearQuasiStaticMixedSelfGravitatingProblem drop(
      &fes_m, &fes_phi, rheology, rho_s, kG, kDtNDegree, nullptr, folded);
  drop.SetSurfaceLoad(sigma, SurfaceMarker(mantle));
  drop.SetRelTol(1e-11);
  drop.AssembleForce(0.0);
  Check(drop.Solve() ? 0.0 : 1.0, 0.0, label + " drop solve");

  Check(RelErr(L2Norm(drop.BackgroundPotential()),
               L2Norm(keep.BackgroundPotential())),
        1e-8, label + " folded-mass |phi0|");
  Check(RelErr(L2Norm(drop.Potential()), L2Norm(keep.Potential())), 0.5,
        label + " drop-vs-keep |phi| band");
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();

  RunCase(2, 1, "2d-o1");
  RunCase(2, 2, "2d-o2");
  RunCase(3, 1, "3d-o1");
  RunCmbCase(2, 2, "2d-o2 cmb");
  RunCmbCase(3, 1, "3d-o1 cmb");
  RunDroppedInner(2, 2, "2d-o2 dropped-inner");
  RunDroppedInner(3, 1, "3d-o1 dropped-inner");

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
