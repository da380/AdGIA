/*
  Parallel tests for the gauged-fluid mode of
  LinearQuasiStaticMixedSelfGravitatingProblem (see TestMixedProblemGauged.cpp
  for the serial suite and the model). Run with 1, 2 and 4 ranks; a
  standalone MPI program returning 0 if every check passes, 1 otherwise.

  Every rank also solves the serial problem on the full mesh; the two solve
  the same discrete system in the same gauge to the solver tolerance, so
  the global L2 norms of the displacement and the potential are compared
  directly, along with the refinement-residual decay.
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

constexpr double kEps = 1.0e-2;
constexpr int kRefine = 3;

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

// The gauged problem on serial or parallel spaces; owns the coefficients.
struct Setup {
  AWBulkModulus kappa;
  FunctionCoefficient mu{GaugedShearModulus};
  ConstantCoefficient mu_gauge{kKappa};
  FunctionCoefficient rho{FullDensity};
  FunctionCoefficient sigma{ZeroMeanSurfaceLoad};
  std::unique_ptr<IsotropicElasticRheology> rheology;
  Array<int> surface, fluid{Array<int>({0, 1, 0})};
  std::unique_ptr<LinearQuasiStaticMixedSelfGravitatingProblem> problem;

  Setup(FiniteElementSpace& fes_u, FiniteElementSpace& fes_phi, Mesh& body) {
    const int dim = body.Dimension();
    rheology = std::make_unique<IsotropicElasticRheology>(dim, kappa, mu);
    surface = SurfaceMarker(body);
    problem = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        &fes_u, &fes_phi, *rheology, rho, kG, kDtNDegree);
    kappa.g = &problem->BackgroundGravity();
    problem->SetGaugedFluid(fluid, mu_gauge, kEps, kRefine);
    problem->SetSurfaceLoad(sigma, surface);
    problem->SetRelTol(1e-11);
  }
};

void RunCase(int dim, int order, const std::string& label) {
  Mesh smesh(ThreeLayerMeshFile(dim).c_str(), 1, 1);
  Array<int> body_attrs({1, 2, 3});

  // Serial reference on every rank.
  SubMesh sbody(SubMesh::CreateFromDomain(smesh, body_attrs));
  H1_FECollection sfec(order, dim);
  FiniteElementSpace sfes_u(&sbody, &sfec, dim), sfes_phi(&smesh, &sfec);
  Setup ser(sfes_u, sfes_phi, sbody);
  ser.problem->AssembleForce(0.0);
  Check(ser.problem->Solve() ? 0.0 : 1.0, 0.0, label + " serial solve");
  const double u_ref = L2Norm(ser.problem->Displacement());
  const double phi_ref = L2Norm(ser.problem->Potential());

  // Parallel problem.
  ParMesh pmesh(MPI_COMM_WORLD, smesh);
  ParSubMesh body(ParSubMesh::CreateFromDomain(pmesh, body_attrs));
  H1_FECollection fec(order, dim);
  ParFiniteElementSpace fes_u(&body, &fec, dim), fes_phi(&pmesh, &fec);
  Setup par(fes_u, fes_phi, body);
  auto& p = *par.problem;
  Check(p.IsParallel() ? 0.0 : 1.0, 0.0, label + " parallel spaces");
  Check(p.HasGaugedFluid() ? 0.0 : 1.0, 0.0, label + " gauged fluid");

  p.AssembleForce(0.0);
  Check(p.Solve() ? 0.0 : 1.0, 0.0, label + " parallel solve");
  Check(RelErr(L2Norm(p.Displacement()), u_ref), 1e-6,
        label + " displacement norm");
  Check(RelErr(L2Norm(p.Potential()), phi_ref), 1e-6,
        label + " potential norm");

  const auto& res = p.GaugeResiduals();
  Check(res.size() == static_cast<size_t>(kRefine) ? 0.0 : 1.0, 0.0,
        label + " residual count");
  if (res.size() >= 2 && res[0] > 0.0) {
    Check(res[1] / res[0], 0.35, label + " residual contraction");
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();

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
