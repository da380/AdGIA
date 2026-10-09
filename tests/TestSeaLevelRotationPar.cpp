/*
  Parallel tests for the composed water-rotation border (SetWaterLoad +
  SetRotation). Run with 1, 2 and 4 ranks; exits 1 on any failure.

  The 2-D two-layer case: the composed solve succeeds, Omega -> 0
  recovers the water-only solution, and the mass row holds at the
  composed state with psi through its interpolant (the border's
  convention).
*/

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <numbers>
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

constexpr double kRhoW = 0.02;
constexpr double kOmega = 0.1;

double RotLoad(const Vector& x) {
  const double r = x.Norml2();
  const double c = x[1] / r;
  return 0.02 * (1.0 + 3.0 * c * c);
}

double GlobalSum(double v) {
  double g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return g;
}

double GlobalMax(double v) {
  double g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return g;
}

void RunCase(int order, const std::string& label) {
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
  const Array<int> surface = SurfaceMarker(body);
  const double g0 = 2.0 * std::numbers::pi * kG * kRho;
  VectorFunctionCoefficient grad_phi0(2, [g0](const Vector& x, Vector& v) {
    v = x;
    v *= g0;
  });
  FunctionCoefficient sigma(RotLoad);
  ConstantCoefficient w(kRhoW / g0);
  Vector moments(1);
  moments[0] = 2.0;

  auto surface_int = [&](Coefficient& f) {
    H1_FECollection sfec(order, 2);
    ParFiniteElementSpace sfes(&body, &sfec);
    ParLinearForm lf(&sfes);
    lf.AddBoundaryIntegrator(new BoundaryLFIntegrator(f),
                             const_cast<Array<int>&>(surface));
    lf.Assemble();
    return GlobalSum(lf.Sum());
  };
  auto make = [&] {
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        &fes_u, &fes_phi, rheology, rho, kG, kDtNDegree);
    p->SetRelTol(1e-12);
    return p;
  };

  // The composed solve and its mass row.
  auto p = make();
  p->SetWaterLoad(w, sigma, surface);
  p->SetRotation(kOmega, moments);
  p->AssembleForce(0.0);
  Check(p->Solve() ? 0.0 : 1.0, 0.0, label + ": composed solve");
  {
    CentrifugalPotential psi(2, kOmega);
    psi.SetAmplitudes(p->AngularVelocity());
    H1_FECollection sfec(order, 2);
    ParFiniteElementSpace sfes(&body, &sfec);
    ParGridFunction psig(&sfes);
    psig.ProjectCoefficient(psi);
    GridFunctionCoefficient psii(&psig);
    VectorGridFunctionCoefficient uc(&p->Displacement());
    InnerProductCoefficient ug(uc, grad_phi0);
    GridFunctionCoefficient pc(
        const_cast<GridFunction*>(&p->PotentialOnBody()));
    SumCoefficient tau(ug, pc);
    SumCoefficient taupsi(tau, psii);
    ProductCoefficient wtp(w, taupsi);
    ConstantCoefficient one(1.0);
    ProductCoefficient wone(w, one);
    const double total = surface_int(sigma) - surface_int(wtp) +
                         p->UniformPotentialTerm() * surface_int(wone);
    Check(std::abs(total), 1e-5, label + ": composed mass row");
  }

  // Omega -> 0 recovers water-only.
  auto p0 = make();
  p0->SetWaterLoad(w, sigma, surface);
  p0->SetRotation(0.0, moments);
  p0->AssembleForce(0.0);
  Check(p0->Solve() ? 0.0 : 1.0, 0.0, label + ": zero-Omega solve");
  auto q = make();
  q->SetWaterLoad(w, sigma, surface);
  q->AssembleForce(0.0);
  Check(q->Solve() ? 0.0 : 1.0, 0.0, label + ": water-only solve");
  Check(GlobalMax(p0->AngularVelocity().Norml2()), 1e-12,
        label + ": zero-Omega omega");
  Check(std::abs(p0->UniformPotentialTerm() - q->UniformPotentialTerm()),
        1e-10, label + ": zero-Omega uniform term");
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();
  for (auto order : {1, 2}) {
    RunCase(order, "p=" + std::to_string(order));
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
