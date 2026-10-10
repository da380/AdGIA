/*
  Parallel tests of the viscoelastic composition of the sea-level
  machinery (WP6 of doc/planning/sea_level_plan.md). Run with 1, 2 and
  4 ranks; a standalone MPI program that exits with status 1 if any
  check fails.

  Mirrors the serial gates on the 2-D two-layer mesh with the
  fingerprint-style analytical state and frozen shorelines: the
  instantaneous Maxwell response equals the elastic water-load solve,
  the Phi_g mass row holds at every step of a Heaviside loading
  history while the body relaxes, and the full Maxwell + water +
  rotation stack composes.
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
int num_failures = 0;

void Check(double value, double bound, const std::string& label) {
  num_checks++;
  const bool ok = std::abs(value) <= bound;
  if (!ok) {
    num_failures++;
  }
  if (Mpi::Root()) {
    std::cout << (ok ? "  ok  " : "  FAIL") << "  " << label << ": |"
              << value << "| <= " << bound << "\n";
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
constexpr double kTauM = 1.0;  // Maxwell time, problem units
constexpr double kMelt = 0.5;

double Colat(const Vector& x) {
  const double r = x.Norml2();
  const double c = x[1] / r;
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

double MeltLoad(const Vector& x) {
  return -(1.0 - Fraction0(x)) * kRhoI * kMelt * Ice0(x);
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
  Mesh smesh(MeshFile(2).c_str(), 1, 1);
  int nxyz[3] = {Mpi::WorldSize(), 1, 1};
  int* partitioning = smesh.CartesianPartitioning(nxyz);
  ParMesh pmesh(MPI_COMM_WORLD, smesh, partitioning);
  delete[] partitioning;
  ParSubMesh body(ParSubMesh::CreateFromDomain(pmesh, BodyMarker(pmesh)));
  H1_FECollection fec(order, 2);
  ParFiniteElementSpace fes_u(&body, &fec, 2), fes_phi(&pmesh, &fec);
  ConstantCoefficient kappa(kKappa), mu(kMu), rho(kRho), tau(kTauM);
  IsotropicElasticRheology elastic(2, kappa, mu);
  auto maxwell = IsotropicMaxwellRheology::Maxwell(2, kappa, mu, tau);
  const Array<int> surface = SurfaceMarker(body);
  const double g0 = 2.0 * std::numbers::pi * kG * kRho;
  VectorFunctionCoefficient grad_phi0(2, [g0](const Vector& x, Vector& v) {
    v = x;
    v *= g0;
  });
  FunctionCoefficient w([g0](const Vector& x) {
    return kRhoW * Fraction0(x) / (g0 * x.Norml2());
  });
  FunctionCoefficient sigma_d(MeltLoad);

  auto surface_integral = [&](Coefficient& f) {
    H1_FECollection sfec(order, 2);
    ParFiniteElementSpace sfes(&body, &sfec);
    ParLinearForm lf(&sfes);
    lf.AddBoundaryIntegrator(new BoundaryLFIntegrator(f),
                             const_cast<Array<int>&>(surface));
    lf.Assemble();
    return GlobalSum(lf.Sum());
  };

  auto make_problem = [&](Rheology& rheology, bool rotate) {
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        &fes_u, &fes_phi, rheology, rho, kG, kDtNDegree);
    p->SetRelTol(1e-12);
    p->SetWaterLoad(w, sigma_d, surface);
    if (rotate) {
      Vector moments(1);
      moments[0] = 2.0;
      p->SetRotation(0.1, moments);
    }
    return p;
  };

  auto mass_residual =
      [&](LinearQuasiStaticMixedSelfGravitatingProblem& p) {
        VectorGridFunctionCoefficient uc(&p.Displacement());
        InnerProductCoefficient ug(uc, grad_phi0);
        GridFunctionCoefficient pc(
            const_cast<GridFunction*>(&p.PotentialOnBody()));
        SumCoefficient tau_field(ug, pc);
        ProductCoefficient wtau(w, tau_field);
        return surface_integral(sigma_d) - surface_integral(wtau) +
               p.UniformPotentialTerm() * surface_integral(w);
      };

  // 1. Instantaneous Maxwell response == elastic water-load solve.
  auto pe = make_problem(elastic, false);
  pe->AssembleForce(0.0);
  Check(pe->Solve() ? 0.0 : 1.0, 0.0, label + ": elastic solve");
  auto pv = make_problem(maxwell, false);
  {
    ViscoelasticOperator visco(*pv);
    Vector m(visco.Height());
    m = 0.0;
    Check(visco.SolveElastic(m, 0.0) ? 0.0 : 1.0, 0.0,
          label + ": instantaneous solve");
    ParGridFunction du(
        static_cast<const ParGridFunction&>(pv->Displacement()));
    du -= pe->Displacement();
    const double rel =
        GlobalNorm(du) /
        (GlobalNorm(static_cast<const ParGridFunction&>(
             pe->Displacement())) +
         1e-30);
    Check(rel, 1e-9, label + ": instantaneous equals elastic");
    Check(pv->UniformPotentialTerm() - pe->UniformPotentialTerm(),
          1e-10 * std::abs(pe->UniformPotentialTerm()) + 1e-14,
          label + ": instantaneous Phi_g");
  }

  // 2. Mass conservation over a Heaviside loading history, with
  //    visible relaxation.
  auto ph = make_problem(maxwell, false);
  {
    ViscoelasticOperator visco(*ph);
    ExponentialTrapezoidSolver ode;
    ode.Init(visco);
    Vector m(visco.Height());
    m = 0.0;
    double t = 0.0;
    Check(visco.SolveElastic(m, t) ? 0.0 : 1.0, 0.0,
          label + ": history elastic response");
    const double u0 = GlobalNorm(
        static_cast<const ParGridFunction&>(ph->Displacement()));
    // Route-grade re-integration: ~1e-5 at order 1, ~1e-6 at order 2.
    const double mass_tol = order == 1 ? 1e-4 : 2e-6;
    Check(mass_residual(*ph), mass_tol, label + ": mass at t = 0");
    const int n = 8;
    double dt = 2.0 * kTauM / n;
    for (int k = 0; k < n; k++) {
      ode.Step(m, t, dt);
      Check(visco.SolveElastic(m, t) ? 0.0 : 1.0, 0.0,
            label + ": history solve");
      Check(mass_residual(*ph), mass_tol,
            label + ": mass along the history");
    }
    const double u1 = GlobalNorm(
        static_cast<const ParGridFunction&>(ph->Displacement()));
    Check(std::abs(u1 - u0) / u0 > 1e-2 ? 0.0 : 1.0, 0.0,
          label + ": the body relaxed");
  }

  // 3. The full stack: Maxwell + water + rotation.
  auto pr = make_problem(maxwell, true);
  {
    ViscoelasticOperator visco(*pr);
    ExponentialTrapezoidSolver ode;
    ode.Init(visco);
    Vector m(visco.Height());
    m = 0.0;
    double t = 0.0;
    Check(visco.SolveElastic(m, t) ? 0.0 : 1.0, 0.0,
          label + ": rotating stack solve");
    double dt = 0.5 * kTauM;
    for (int k = 0; k < 2; k++) {
      ode.Step(m, t, dt);
    }
    Check(visco.SolveElastic(m, t) ? 0.0 : 1.0, 0.0,
          label + ": rotating stack steps");
    Check(mass_residual(*pr), 1e-4, label + ": rotating mass");
    const double om = pr->AngularVelocity().Norml2();
    Check(std::isfinite(om) && om > 0.0 ? 0.0 : 1.0, 0.0,
          label + ": finite spin");
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();
  for (int order : {1, 2}) {
    RunCase(order, "order " + std::to_string(order));
  }
  if (Mpi::Root()) {
    std::cout << num_checks << " checks, " << num_failures
              << " failures\n";
  }
  return num_failures == 0 ? 0 : 1;
}
