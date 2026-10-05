/*
  Parallel tests for LinearQuasiStaticMixedSelfGravitatingProblem on
  ParFiniteElementSpaces (ParSubMesh body inside a ParMesh ball). Run with 1, 2
  and 4 ranks; a standalone MPI program returning 0 if every check passes, 1 otherwise.

  Every rank also builds the serial problem on the full mesh; the parallel
  and serial solutions are compared through partition-independent
  quantities (global L2 norms of the displacement, the potential and the
  background potential, the rigid-mode residuals), for both solver types and with a relaxation
  weight, and the time-scaling of the load is checked in parallel.
*/

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>

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

struct SerialCase {
  std::unique_ptr<Mesh> parent;
  std::unique_ptr<SubMesh> body;
  std::unique_ptr<H1_FECollection> fec;
  std::unique_ptr<FiniteElementSpace> fes_u, fes_phi;
  ConstantCoefficient kappa{kKappa}, mu{kMu}, rho{kRho}, tau{1.0};
  std::unique_ptr<IsotropicMaxwellRheology> rheology;
  FunctionCoefficient sigma{SurfaceLoad};
  Array<int> surface;
  std::unique_ptr<LinearQuasiStaticMixedSelfGravitatingProblem> problem;

  SerialCase(Mesh& mesh, int dim, int order) {
    parent = std::make_unique<Mesh>(mesh);
    body = std::make_unique<SubMesh>(
        SubMesh::CreateFromDomain(*parent, BodyMarker(*parent)));
    fec = std::make_unique<H1_FECollection>(order, dim);
    fes_u = std::make_unique<FiniteElementSpace>(body.get(), fec.get(), dim);
    fes_phi = std::make_unique<FiniteElementSpace>(parent.get(), fec.get());
    // Maxwell body: unrelaxed modulus mu, relaxation weight beta gives
    // beta mu.
    rheology = std::make_unique<IsotropicMaxwellRheology>(
        IsotropicMaxwellRheology::Maxwell(dim, kappa, mu, tau));
    surface = SurfaceMarker(*body);
    problem = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        fes_u.get(), fes_phi.get(), *rheology, rho, kG, kDtNDegree);
    problem->SetSurfaceLoad(sigma, surface);
    problem->SetRelTol(1e-11);
  }
};

void RunCase(int dim, int order, const std::string& label) {
  Mesh smesh(MeshFile(dim).c_str(), 1, 1);

  // Serial reference on every rank.
  SerialCase ser(smesh, dim, order);
  ser.problem->AssembleForce(0.0);
  Check(ser.problem->Solve() ? 0.0 : 1.0, 0.0, label + " serial solve");
  const double u_ref = L2Norm(ser.problem->Displacement());
  const double phi_ref = L2Norm(ser.problem->Potential());
  const double phi0_ref = L2Norm(ser.problem->BackgroundPotential());
  const auto res_ref = ser.problem->RigidModeResiduals();

  // Parallel problem.
  ParMesh pmesh(MPI_COMM_WORLD, smesh);
  ParSubMesh body(ParSubMesh::CreateFromDomain(pmesh, BodyMarker(pmesh)));
  H1_FECollection fec(order, dim);
  ParFiniteElementSpace fes_u(&body, &fec, dim);
  ParFiniteElementSpace fes_phi(&pmesh, &fec);
  ConstantCoefficient kappa(kKappa), mu(kMu), rho(kRho), tau(1.0);
  auto rheology = IsotropicMaxwellRheology::Maxwell(dim, kappa, mu, tau);
  FunctionCoefficient sigma(SurfaceLoad);
  Array<int> surface = SurfaceMarker(body);

  LinearQuasiStaticMixedSelfGravitatingProblem p(&fes_u, &fes_phi, rheology, rho, kG,
                                            kDtNDegree);
  p.SetSurfaceLoad(sigma, surface);
  p.SetRelTol(1e-11);
  Check(p.IsParallel() ? 0.0 : 1.0, 0.0, label + " parallel spaces");

  Check(RelErr(L2Norm(p.BackgroundPotential()), phi0_ref), 1e-9,
        label + " background potential");

  const auto res = p.RigidModeResiduals();
  for (size_t i = 0; i < res.size(); i++) {
    Check(RelErr(res[i], res_ref[i]), 1e-6,
          label + " rigid-mode residual " + std::to_string(i));
  }

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
    // Both parallel solvers agree with the serial reference (block MINRES)
    // to solver tolerance: the two solver types solve the same system
    // (measured agreement about 1e-11).
    const double tol = 1e-8;
    Check(RelErr(L2Norm(p.Displacement()), u_ref), tol,
          label + name + " |u|");
    Check(RelErr(L2Norm(p.Potential()), phi_ref), tol,
          label + name + " |phi|");
  }

  // Time scaling of the load, in parallel.
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
