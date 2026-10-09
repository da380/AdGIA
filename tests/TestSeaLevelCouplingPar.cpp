/*
  Parallel tests of the coupled water load (SetWaterLoad). Run with 1, 2
  and 4 ranks; a standalone MPI program that exits with status 1 if any
  check fails.

  Mirrors the serial rung-0 gates on the 2-D two-layer mesh, all-ocean:
  the spectral identity s = -T/(1 + rho_w T) against the plain-solve
  response, the mass-conservation certificate, the inert uniform term
  for a zero-mean load, and SchurCG == BlockMINRES through the
  Sherman-Morrison border.
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

double Degree2(const Vector& x) {
  const double r = x.Norml2();
  return (x[0] * x[0] - x[1] * x[1]) / (r * r);
}

double GlobalSum(double v) {
  double g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
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
  FunctionCoefficient sigma_data(Degree2);
  ConstantCoefficient w(kRhoW / g0);

  auto surface_integral = [&](Coefficient& f) {
    H1_FECollection sfec(order, 2);
    ParFiniteElementSpace sfes(&body, &sfec);
    ParLinearForm lf(&sfes);
    lf.AddBoundaryIntegrator(new BoundaryLFIntegrator(f),
                             const_cast<Array<int>&>(surface));
    lf.Assemble();
    return GlobalSum(lf.Sum());
  };

  auto make_problem = [&] {
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        &fes_u, &fes_phi, rheology, rho, kG, kDtNDegree);
    p->SetRelTol(1e-12);
    return p;
  };

  auto amplitude = [&](SeaLevelOperator& sea) {
    FunctionCoefficient pat(Degree2);
    ParGridFunction pg(
        static_cast<ParFiniteElementSpace*>(&sea.SurfaceSpace()));
    pg.ProjectCoefficient(pat);
    GridFunctionCoefficient fc(&sea.SeaLevelChangeField()), pc(&pg);
    ProductCoefficient fp(fc, pc), pp(pc, pc);
    return sea.SurfaceIntegral(fp) / sea.SurfaceIntegral(pp);
  };

  // Plain response T.
  auto pt = make_problem();
  pt->SetSurfaceLoad(sigma_data, surface);
  pt->AssembleForce(0.0);
  Check(pt->Solve() ? 0.0 : 1.0, 0.0, label + ": plain solve");
  SeaLevelOperator sea_t(fes_u, surface);
  sea_t.SeaLevelChangeFrom(pt->Displacement(), grad_phi0, pt->Potential(),
                           0.0);
  const double T = -amplitude(sea_t);

  // Monolithic.
  auto p = make_problem();
  p->SetWaterLoad(w, sigma_data, surface);
  p->AssembleForce(0.0);
  Check(p->Solve() ? 0.0 : 1.0, 0.0, label + ": coupled solve");
  SeaLevelOperator sea(fes_u, surface);
  sea.SeaLevelChangeFrom(p->Displacement(), grad_phi0, p->Potential(),
                         p->UniformPotentialTerm());
  const double s = amplitude(sea);
  Check(std::abs(s + T / (1.0 + kRhoW * T)),
        (order == 1 ? 1e-3 : 2e-4) * std::abs(T),
        label + ": spectral identity");
  Check(std::abs(p->UniformPotentialTerm()), 2e-3,
        label + ": inert uniform term");

  // Mass certificate (route grade).
  {
    VectorGridFunctionCoefficient uc(&p->Displacement());
    InnerProductCoefficient ug(uc, grad_phi0);
    GridFunctionCoefficient pc(
        const_cast<GridFunction*>(&p->PotentialOnBody()));
    SumCoefficient tau(ug, pc);
    ProductCoefficient wtau(w, tau);
    ConstantCoefficient one(1.0);
    ProductCoefficient wone(w, one);
    const double total = surface_integral(sigma_data) -
                         surface_integral(wtau) +
                         p->UniformPotentialTerm() * surface_integral(wone);
    Check(std::abs(total), 1e-5, label + ": mass certificate");
  }

  // Solver-type agreement.
  auto b = make_problem();
  b->SetWaterLoad(w, sigma_data, surface);
  b->SetSolverType(
      LinearQuasiStaticMixedSelfGravitatingProblem::SolverType::BlockMINRES);
  b->AssembleForce(0.0);
  Check(b->Solve() ? 0.0 : 1.0, 0.0, label + ": block solve");
  Check(std::abs(b->UniformPotentialTerm() - p->UniformPotentialTerm()),
        1e-8, label + ": solver-type uniform term");
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
