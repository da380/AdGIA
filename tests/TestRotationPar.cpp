/*
  Parallel tests for RotationalFeedback. Run with 1, 2 and 4 ranks; a
  standalone MPI program that exits with status 1 if any check fails.

  Mirrors TestRotation.cpp on the 2-D canned meshes: the solid two-layer
  body and the three-layer fluid-core case. Gates: the component's
  consistency diagnostic, and the independent omega-row residual with
  the test's own inertia matrix and surface pairing.
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

constexpr double kOmega = 0.1;

double RotLoad(const Vector& x, double t) {
  const double r = x.Norml2();
  if (r == 0.0) {
    return 0.0;
  }
  const double c = x[1] / r;
  return 0.02 * (1.0 + 3.0 * c * c) * (1.0 + t);
}

DenseMatrix InertiaMatrix(const Vector& moments) {
  DenseMatrix D(1);
  D(0, 0) = -moments[0];
  return D;
}

double SurfacePair(ParFiniteElementSpace& fes_u, Coefficient& sigma,
                   Coefficient& psi_k, const Array<int>& marker) {
  H1_FECollection fec(fes_u.GetMaxElementOrder(), 2);
  ParFiniteElementSpace s(fes_u.GetParMesh(), &fec);
  ProductCoefficient sp(sigma, psi_k);
  ParLinearForm lf(&s);
  lf.AddBoundaryIntegrator(new BoundaryLFIntegrator(sp),
                           const_cast<Array<int>&>(marker));
  lf.Assemble();
  double v = lf.Sum(), g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return g;
}

double OmegaRowResidual(LinearQuasiStaticMixedSelfGravitatingProblem& p,
                        ParFiniteElementSpace& fes_u, Coefficient& sigma,
                        const Array<int>& marker, const Vector& moments,
                        const Vector& omega) {
  CentrifugalPotential psi0(2, kOmega);
  psi0.SetUnit(0);
  const double tk = p.TidalCoupling(psi0);
  const double dp =
      (InertiaMatrix(moments)(0, 0) + p.TidalTidalCoupling(psi0, psi0)) *
      omega[0];
  const double sk = SurfacePair(fes_u, sigma, psi0, marker);
  return std::abs(tk + dp + sk) /
         (std::max({std::abs(tk), std::abs(dp), std::abs(sk)}) + 1e-300);
}

void SolidCase(int order, const std::string& label) {
  Mesh smesh(MeshFile(2).c_str(), 1, 1);
  int nxyz[3] = {Mpi::WorldSize(), 1, 1};
  int* partitioning = smesh.CartesianPartitioning(nxyz);
  ParMesh pmesh(MPI_COMM_WORLD, smesh, partitioning);
  delete[] partitioning;
  ParSubMesh body(ParSubMesh::CreateFromDomain(pmesh, BodyMarker(pmesh)));
  H1_FECollection fec(order, 2);
  ParFiniteElementSpace fes_u(&body, &fec, 2), fes_phi(&pmesh, &fec);
  ConstantCoefficient kappa(kKappa), mu(kMu), rho(kRho);
  IsotropicElasticRheology rheology(2, kappa, mu);
  LinearQuasiStaticMixedSelfGravitatingProblem p(&fes_u, &fes_phi, rheology,
                                                 rho, kG, kDtNDegree);
  FunctionCoefficient sigma(RotLoad);
  const Array<int> surface = SurfaceMarker(body);
  p.SetSurfaceLoad(sigma, surface);
  p.SetRelTol(1e-11);

  Vector moments(1);
  moments[0] = 2.0;
  RotationalFeedback rot(p, kOmega, moments);
  Check(rot.Solve(0.0, sigma, surface) ? 0.0 : 1.0, 0.0, label + ": solve");
  Check(rot.AngularVelocityPerturbation().Norml2() > 0.0 ? 0.0 : 1.0, 0.0,
        label + ": nonzero omega");
  Check(rot.ConsistencyResidual(sigma, surface), 5e-8,
        label + ": consistency residual");
  Check(OmegaRowResidual(p, fes_u, sigma, surface, moments,
                         rot.AngularVelocityPerturbation()),
        5e-8, label + ": omega-row residual");
}

void FluidCase(int order, const std::string& label) {
  Mesh smesh(ThreeLayerMeshFile(2).c_str(), 1, 1);
  int nxyz[3] = {Mpi::WorldSize(), 1, 1};
  int* partitioning = smesh.CartesianPartitioning(nxyz);
  ParMesh pmesh(MPI_COMM_WORLD, smesh, partitioning);
  delete[] partitioning;
  Array<int> attrs({1, 3});
  ParSubMesh solid(ParSubMesh::CreateFromDomain(pmesh, attrs));
  H1_FECollection fec(order, 2);
  ParFiniteElementSpace fes_u(&solid, &fec, 2), fes_phi(&pmesh, &fec);
  ConstantCoefficient kappa(kKappa), mu(kMu);
  FunctionCoefficient rho_s(SolidDensity), rho_f(FluidDensity);
  IsotropicElasticRheology rheology(2, kappa, mu);
  std::vector<FluidRegion> fluids;
  fluids.push_back(OuterCore(solid, rho_f));
  LinearQuasiStaticMixedSelfGravitatingProblem p(
      &fes_u, &fes_phi, rheology, rho_s, kG, kDtNDegree, nullptr, fluids);
  FunctionCoefficient sigma(RotLoad);
  const Array<int> surface = SurfaceMarker(solid);
  p.SetSurfaceLoad(sigma, surface);
  Array<int> inner_core({1});
  p.AddRegionRotations(inner_core);
  p.SetRelTol(1e-11);

  Vector moments(1);
  moments[0] = 2.0;
  RotationalFeedback rot(p, kOmega, moments);
  Check(rot.Solve(0.0, sigma, surface) ? 0.0 : 1.0, 0.0, label + ": solve");
  CentrifugalPotential u0(2, kOmega);
  u0.SetUnit(0);
  Check(std::abs(p.TidalTidalCoupling(u0, u0)) > 0.0 ? 0.0 : 1.0, 0.0,
        label + ": fluid psi-psi block nonzero");
  Check(rot.ConsistencyResidual(sigma, surface), 5e-8,
        label + ": consistency residual");
  Check(OmegaRowResidual(p, fes_u, sigma, surface, moments,
                         rot.AngularVelocityPerturbation()),
        5e-8, label + ": omega-row residual");
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();

  for (auto order : {1, 2}) {
    SolidCase(order, "solid p=" + std::to_string(order));
    FluidCase(order, "fluid p=" + std::to_string(order));
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
