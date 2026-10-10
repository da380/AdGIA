/*
  Parallel tests of shoreline migration (ShorelineMigration). Run with
  1, 2 and 4 ranks; a standalone MPI program that exits with status 1 if
  any check fails.

  Mirrors the serial gates on the 3-D two-layer mesh with the
  fingerprint-style analytical state: the off-switch equals the direct
  frozen-C solve, the Picard loop converges with the migrated mass
  certificate holding, and the composition with SetRotation runs.
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

constexpr double kRhoW = 0.05;
constexpr double kRhoI = 0.045;
constexpr double kDepth = 1.0;
constexpr double kContAmp = 3.0;
constexpr double kContWidth = 0.7;
constexpr double kCapAmp = 4.0;
constexpr double kCapWidth = 0.35;
constexpr double kShore = 5e-3;

double Colat(const Vector& x) {
  const double r = x.Norml2();
  const double c = x[2] / r;
  return std::acos(std::min(1.0, std::max(-1.0, c)));
}

double SL0(const Vector& x) {
  const double t = Colat(x) / kContWidth;
  return kDepth - kContAmp * std::exp(-0.5 * std::pow(t, 6));
}

double Ice0(const Vector& x) {
  const double t = Colat(x) / kCapWidth;
  return kCapAmp * std::exp(-0.5 * t * t);
}

double Fraction0(const Vector& x) {
  const double q = kRhoW * SL0(x) - kRhoI * Ice0(x);
  return 0.5 * (1.0 + std::tanh(q / kShore));
}

double GlobalSum(double v) {
  double g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return g;
}

double GlobalNorm(const ParGridFunction& f) {
  Vector tv(f.ParFESpace()->GetTrueVSize());
  f.GetTrueDofs(tv);
  const double local = tv * tv;
  return std::sqrt(GlobalSum(local));
}

void RunCase(int order, const std::string& label) {
  Mesh smesh(MeshFile(3).c_str(), 1, 1);
  int nxyz[3] = {Mpi::WorldSize(), 1, 1};
  int* partitioning = smesh.CartesianPartitioning(nxyz);
  ParMesh pmesh(MPI_COMM_WORLD, smesh, partitioning);
  delete[] partitioning;
  ParSubMesh body(ParSubMesh::CreateFromDomain(pmesh, BodyMarker(pmesh)));
  H1_FECollection fec(order, 3);
  ParFiniteElementSpace fes_u(&body, &fec, 3), fes_phi(&pmesh, &fec);
  ConstantCoefficient kappa(kKappa), mu(kMu), rho(kRho);
  IsotropicElasticRheology rheology(3, kappa, mu);
  const Array<int> surface = SurfaceMarker(body);
  const double g0 = 4.0 * std::numbers::pi * kG * kRho / 3.0;
  VectorFunctionCoefficient grad_phi0(3, [g0](const Vector& x, Vector& v) {
    v = x;
    v *= g0;
  });
  FunctionCoefficient sl0(SL0), ice0(Ice0);
  FunctionCoefficient dice([](const Vector& x) { return -0.5 * Ice0(x); });

  auto surface_integral = [&](Coefficient& f) {
    H1_FECollection sfec(order, 3);
    ParFiniteElementSpace sfes(&body, &sfec);
    ParLinearForm lf(&sfes);
    lf.AddBoundaryIntegrator(new BoundaryLFIntegrator(f),
                             const_cast<Array<int>&>(surface));
    lf.Assemble();
    return GlobalSum(lf.Sum());
  };

  auto make_problem = [&](bool rotate) {
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        &fes_u, &fes_phi, rheology, rho, kG, kDtNDegree);
    p->SetRelTol(1e-12);
    if (rotate) {
      Vector moments(3);
      moments[0] = 1.2;
      moments[1] = 1.3;
      moments[2] = 2.0;
      p->SetRotation(0.1, moments);
    }
    return p;
  };

  auto migrate = [&](LinearQuasiStaticMixedSelfGravitatingProblem& p,
                     bool moving, double inexact = 0.1,
                     double guard = 0.5) {
    ShorelineMigration::Options opt;
    opt.max_iterations = moving ? 12 : 0;
    opt.tol = 1e-6;
    opt.shore = kShore;
    opt.inexact = inexact;
    opt.guard = guard;
    auto m = std::make_unique<ShorelineMigration>(p, grad_phi0, sl0, ice0,
                                                  dice, kRhoW, kRhoI,
                                                  surface, opt);
    Check(m->Solve(0.0) ? 0.0 : 1.0, 0.0, label + ": migration solve");
    return m;
  };

  // 1. The off-switch against the direct frozen-C wiring (pointwise
  //    g = g0 |x|, the orchestrator's own route).
  auto pf = make_problem(false);
  auto mf = migrate(*pf, false);
  FunctionCoefficient w0([g0](const Vector& x) {
    return kRhoW * Fraction0(x) / (g0 * x.Norml2());
  });
  FunctionCoefficient sd0([](const Vector& x) {
    return kRhoI * (1.0 - Fraction0(x)) * (-0.5 * Ice0(x));
  });
  auto q = make_problem(false);
  q->SetWaterLoad(w0, sd0, surface);
  q->AssembleForce(0.0);
  Check(q->Solve() ? 0.0 : 1.0, 0.0, label + ": direct solve");
  {
    ParGridFunction du(static_cast<const ParGridFunction&>(
        pf->Displacement()));
    du -= q->Displacement();
    const double rel =
        GlobalNorm(du) /
        (GlobalNorm(static_cast<const ParGridFunction&>(q->Displacement())) +
         1e-30);
    Check(rel, 1e-9, label + ": off-switch equals frozen C");
    Check(std::abs(pf->UniformPotentialTerm() - q->UniformPotentialTerm()),
          1e-10, label + ": off-switch Phi_g");
    Check(mf->Iterations(), 0.0, label + ": off-switch iterations");
  }

  // 2. Migration converges and conserves mass with the final fraction.
  auto pm = make_problem(false);
  auto mm = migrate(*pm, true);
  Check(mm->Iterations() > 0 ? 0.0 : 1.0, 0.0,
        label + ": migration iterates");
  Check(mm->LastShorelineChange(), 1e-6, label + ": migration converged");
  {
    Coefficient& C = mm->OceanFraction();
    VectorGridFunctionCoefficient uc(&pm->Displacement());
    InnerProductCoefficient ug(uc, grad_phi0);
    GridFunctionCoefficient pc(
        const_cast<GridFunction*>(&pm->PotentialOnBody()));
    SumCoefficient tau(ug, pc);
    FunctionCoefficient ginv([g0](const Vector& x) {
      return kRhoW / (g0 * x.Norml2());
    });
    ProductCoefficient w(ginv, C);
    ProductCoefficient wtau(w, tau);
    FunctionCoefficient one_c([](const Vector&) { return 1.0; });
    SumCoefficient one_minus_C(one_c, C, 1.0, -1.0);
    FunctionCoefficient itot(
        [](const Vector& x) { return Ice0(x) - 0.5 * Ice0(x); });
    ProductCoefficient i_term(one_minus_C, itot);
    FunctionCoefficient i0_c(Ice0);
    FunctionCoefficient c0_c(Fraction0);
    SumCoefficient one_minus_C0(one_c, c0_c, 1.0, -1.0);
    ProductCoefficient i0_term(one_minus_C0, i0_c);
    SumCoefficient ice_diff(i_term, i0_term, kRhoI, -kRhoI);
    SumCoefficient C_minus_C0(C, c0_c, 1.0, -1.0);
    FunctionCoefficient sl0_c(SL0);
    ProductCoefficient water_diff0(C_minus_C0, sl0_c);
    SumCoefficient sigma_d(ice_diff, water_diff0, 1.0, kRhoW);
    const double total = surface_integral(sigma_d) - surface_integral(wtau) +
                         pm->UniformPotentialTerm() * surface_integral(w);
    Check(std::abs(total), 1e-4, label + ": migrated mass conservation");
  }

  // 3. Inexact Picard equals the all-tight route at polish grade, and
  //    does not cost more outer iterations.
  auto pt = make_problem(false);
  {
    auto mt = migrate(*pt, true, 0.0);
    ParGridFunction du(
        static_cast<const ParGridFunction&>(pm->Displacement()));
    du -= pt->Displacement();
    const double rel =
        GlobalNorm(du) /
        (GlobalNorm(static_cast<const ParGridFunction&>(
             pt->Displacement())) +
         1e-30);
    Check(rel, 1e-5, label + ": inexact equals tight");
    // Iteration-grade slack: at 3-D order 1 the loose and tight routes
    // land within a few iterations of each other.
    Check(mm->TotalOuterIterations() <=
                  mt->TotalOuterIterations() +
                      std::max(5, mt->TotalOuterIterations() / 20)
              ? 0.0
              : 1.0,
          0.0, label + ": inexact economy");
  }

  // 4. The guard's sticky escalation: a vanishing guard fraction trips
  //    on the first measurable pass, abandons inexactness, and must
  //    land on the tight loop's fixed point (the rescue path for
  //    border-poisoned loose passes).
  auto pg = make_problem(false);
  {
    auto mg = migrate(*pg, true, 0.1, 1e-12);
    Check(mg->LastShorelineChange(), 1e-6, label + ": guarded converged");
    ParGridFunction du(
        static_cast<const ParGridFunction&>(pg->Displacement()));
    du -= pt->Displacement();
    const double rel =
        GlobalNorm(du) /
        (GlobalNorm(static_cast<const ParGridFunction&>(
             pt->Displacement())) +
         1e-30);
    Check(rel, 1e-5, label + ": guard escalates to tight");
  }

  // 5. Composition with rotation.
  auto pr = make_problem(true);
  auto mr = migrate(*pr, true);
  Check(mr->LastShorelineChange(), 1e-6, label + ": rotating converged");
  Check(pr->AngularVelocity().Norml2() > 0.0 ? 0.0 : 1.0, 0.0,
        label + ": rotating spin");
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();

  // Order 1 alone: the parallel twin exercises the plumbing; the order
  // axis lives in the serial suite.
  RunCase(1, "o1");

  const int fails = static_cast<int>(GlobalSum(num_fails)) / 1;
  if (Mpi::Root()) {
    std::cout << (fails == 0 ? "PASS" : "FAIL") << " (" << num_checks
              << " checks per rank)\n";
  }
  return fails == 0 ? 0 : 1;
}
