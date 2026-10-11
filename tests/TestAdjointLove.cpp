/*
  Gate GA3: the eta kernel's 1-D cross-check against the Love-number
  closed form — an INDEPENDENT route, not a finite difference of the
  same discretisation.

  On the homogeneous self-gravitating Maxwell sphere, a held surface
  load with one unit (l, 0) harmonic coefficient relaxes with the
  closed-form degree history (Wu & Peltier 1982, by correspondence;
  examples/viscoelastic_love_numbers.cpp):

      f_l(t) = 1 - A e^{-t/tau_l},  A = mu_l / (1 + mu_l),
      tau_l  = (1 + mu_l) eta / mu,
      mu_l   = (2 l^2 + 4 l + 3) mu / (l rho g a).

  Any fixed linear observation J that extracts the (l, 0) response is
  J(t) = C f_l(t) with C set by the elastic constants alone — so eta
  enters ONLY through tau_l, and

      dJ/d eta = J(t1) * (df_l/d eta)(t1) / f_l(t1),
      df_l/d eta = -A e^{-t1/tau_l} (t1 / tau_l^2) (1 + mu_l) / mu,

  with NO normalisation conventions anywhere: the measured J(t1)
  carries them all. The gate compares the adjoint eta kernel's
  directional derivative along a uniform delta-eta against this
  analytic value — the kernel machinery against the correspondence
  principle. Two supporting checks:

   - the structure sanity J(0+)/J(t1) = f_l(0)/f_l(t1), which
     calibrates how closely this compressible, discretised sphere
     follows the incompressible closed form (kappa/mu = 2000; the
     example's caveats on order apply and the thresholds below encode
     the measured grades);
   - a RADIAL-PROFILE leg: the kernel integrated against a radial
     viscosity bump versus central finite differences of full reruns
     (self-consistent, but it exercises the production pattern of
     radial inversions — a 3-D kernel paired with 1-D directions).

  The run is the heaviest adjoint gate (3-D gravitating,
  near-incompressible); the order and step count encode the measured
  cost/grade trade recorded in the comments.
*/

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <numbers>

#include "AdGIA.hpp"
#include "mfem.hpp"

using namespace mfem;
using namespace AdGIA;

namespace {

constexpr real_t kPi = std::numbers::pi_v<real_t>;
constexpr real_t kGconst = 1.0;
constexpr real_t kRho = 1.0;
constexpr real_t kKappa = 1000.0;
constexpr real_t kMu = 0.5;
constexpr real_t kEta = 0.5;  // tau = eta / mu = 1
constexpr int kDegree = 2;
constexpr int kDtN = 12;

// The (2,0) observation window, unnormalised: J = int Y (u . rhat) dS.
void RadialY20(const Vector& x, Vector& v) {
  const real_t r = x.Norml2();
  const real_t c = x[2] / r;
  const real_t y = 3.0 * c * c - 1.0;
  v.SetSize(3);
  v = x;
  v *= y / r;
}

}  // namespace

struct LoveRun {
  real_t sanity = 0.0;      // (J0/J1) / (f(0)/f(t1))
  real_t closed = 0.0;      // Directional(all) / dJ_closed
  real_t radial_fd = 0.0;   // Directional(radial) / FD   (0 if skipped)
};

static LoveRun LoveGate(int order, int n_steps, bool with_fd) {
  const real_t t1 = 3.0;  // Maxwell times; near the degree's tau_l
  LoveRun out;

  Mesh smesh("../data/coupled_poisson.msh", 1, 1);
  EXPECT_EQ(smesh.Dimension(), 3);
  Mesh& parent = smesh;
  Array<int> body_marker(parent.attributes.Max());
  body_marker = 0;
  body_marker[0] = 1;
  auto body = SubMesh::CreateFromDomain(parent, body_marker);
  Array<int> surface(body.bdr_attributes.Max());
  surface = 0;
  surface[body.bdr_attributes.Max() - 1] = 1;

  H1_FECollection fec(order, 3);
  FiniteElementSpace fes_u(&body, &fec, 3), fes_phi(&parent, &fec);
  FiniteElementSpace sfes(&body, &fec);
  ConstantCoefficient kappa0(kKappa), mu0(kMu), rho0(kRho), eta0(kEta);
  auto tau0 = RatioCoefficient(eta0, mu0);

  // The pure (2,0) surface load through the harmonic layer.
  using BHC = BoundaryHarmonicCoefficients;
  BHC radial(fes_u, surface, kDegree, BHC::Component::Radial);
  const real_t a = radial.Radius();
  Vector unit(radial.Basis().Size());
  unit = 0.0;
  unit[radial.Basis().Index(kDegree, 0)] = 1.0;
  auto sigma = radial.Expansion(unit, false);

  // Surface gravity from the mesh's own mass; the closed-form constants.
  LinearForm mass(&sfes);
  mass.AddDomainIntegrator(new DomainLFIntegrator(rho0));
  mass.Assemble();
  const real_t g_surf = kGconst * mass.Sum() / (a * a);
  const real_t mu_l = (2.0 * kDegree * kDegree + 4.0 * kDegree + 3.0) * kMu /
                      (kDegree * kRho * g_surf * a);
  const real_t tau_l = (1.0 + mu_l) * kEta / kMu;
  const real_t A = mu_l / (1.0 + mu_l);
  auto f_of = [&](real_t t) { return 1.0 - A * std::exp(-t / tau_l); };
  // df/d eta at t1: tau_l = (1 + mu_l) eta / mu.
  const real_t df_deta = -A * std::exp(-t1 / tau_l) * (t1 / (tau_l * tau_l)) *
                         (1.0 + mu_l) / kMu;

  // The observation dual.
  auto h = VectorFunctionCoefficient(3, RadialY20);
  auto g_dual = LinearForm(&fes_u);
  g_dual.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(h), surface);
  g_dual.Assemble();

  const real_t dt = t1 / n_steps;
  // One forward run; returns (J at 0+, J at t1), optionally recording.
  auto run_forward = [&](Coefficient& eta, ForwardTrajectory* keep,
                         real_t* J0_out) {
    auto tau = RatioCoefficient(eta, mu0);
    auto rheology = IsotropicMaxwellRheology::Maxwell(3, kappa0, mu0, tau);
    auto prob = LinearQuasiStaticMixedSelfGravitatingProblem(
        &fes_u, &fes_phi, rheology, rho0, kGconst, kDtN);
    prob.SetSurfaceLoad(*sigma, surface);
    prob.SetRelTol(1e-9);
    auto visco = ViscoelasticOperator(prob);
    ExponentialTrapezoidSolver ode;
    ode.Init(visco);
    Vector m(visco.Height());
    m = 0.0;
    real_t t = 0.0, dt_step = dt;
    EXPECT_TRUE(visco.SolveElastic(m, t));
    if (J0_out) {
      *J0_out = g_dual * prob.Displacement();
    }
    if (keep) {
      keep->Record(m, t);
    }
    for (int n = 1; n <= n_steps; n++) {
      ode.Step(m, t, dt_step);
      if (keep) {
        keep->Record(m, t);
      }
    }
    EXPECT_TRUE(visco.SolveElastic(m, t));
    return g_dual * prob.Displacement();
  };

  // The base run on persistent operators (the trajectory must outlive
  // the sweep).
  auto rheology0 = IsotropicMaxwellRheology::Maxwell(3, kappa0, mu0, tau0);
  auto prob_fwd = LinearQuasiStaticMixedSelfGravitatingProblem(
      &fes_u, &fes_phi, rheology0, rho0, kGconst, kDtN);
  prob_fwd.SetSurfaceLoad(*sigma, surface);
  prob_fwd.SetRelTol(1e-9);
  auto visco_fwd = ViscoelasticOperator(prob_fwd);
  auto trajectory = ForwardTrajectory(visco_fwd, true);
  real_t J0 = 0.0, J1 = 0.0;
  {
    ExponentialTrapezoidSolver ode;
    ode.Init(visco_fwd);
    Vector m(visco_fwd.Height());
    m = 0.0;
    real_t t = 0.0, dt_step = dt;
    EXPECT_TRUE(visco_fwd.SolveElastic(m, t));
    J0 = g_dual * prob_fwd.Displacement();
    trajectory.Record(m, t);
    for (int n = 1; n <= n_steps; n++) {
      ode.Step(m, t, dt_step);
      trajectory.Record(m, t);
    }
    EXPECT_TRUE(visco_fwd.SolveElastic(m, t));
    J1 = g_dual * prob_fwd.Displacement();
  }

  out.sanity = (J0 / J1) / (f_of(0.0) / f_of(t1));

  // The adjoint sweep on the zero-load twin.
  auto prob_adj = LinearQuasiStaticMixedSelfGravitatingProblem(
      &fes_u, &fes_phi, rheology0, rho0, kGconst, kDtN);
  prob_adj.SetRelTol(1e-9);
  auto visco_adj = ViscoelasticOperator(prob_adj);

  auto K_eta = IsotropicKernel(IsotropicKernel::Parameter::Eta, sfes, &tau0);
  auto chi_all = ConstantCoefficient(1.0);
  auto chi_radial = FunctionCoefficient([](const Vector& x) {
    const real_t r = x.Norml2();
    const real_t d = (r - 0.35) / 0.2;
    return std::exp(-d * d);
  });
  auto d_all = ProductCoefficient(eta0, chi_all);
  auto d_rad = ProductCoefficient(eta0, chi_radial);
  K_eta.AddDirection(d_all);
  K_eta.AddDirection(d_rad);

  AdjointForce force;
  force.AddObservation(trajectory.Time(n_steps), g_dual);
  auto solver = AdjointSolver(visco_adj, trajectory);
  solver.SetForce(force);
  solver.AddKernel(K_eta);
  EXPECT_TRUE(solver.Run());

  // The closed-form cross-check: the kernel's uniform-eta derivative
  // against the correspondence-principle value, built from the
  // MEASURED J(t1) so no normalisation convention enters:
  //     dJ[eta0 * 1] = eta0 * J(t1) * (df/deta)(t1) / f(t1).
  const real_t dJ_closed = kEta * J1 * df_deta / f_of(t1);
  out.closed = K_eta.Directional(0) / dJ_closed;

  if (with_fd) {
    // The radial-profile leg against central FD of full reruns
    // (self-consistent; the radial-inversion pairing pattern).
    const real_t eps = 1e-3;
    real_t fd = 0.0;
    for (const real_t s : {eps, -eps}) {
      auto one_plus = SumCoefficient(1.0, chi_radial, 1.0, s);
      auto eta_p = ProductCoefficient(eta0, one_plus);
      fd += (s > 0 ? 1.0 : -1.0) * run_forward(eta_p, nullptr, nullptr);
    }
    fd /= 2.0 * eps;
    out.radial_fd = K_eta.Directional(1) / fd;
  }
  return out;
}

// GATE A (GA3 proper), at order 3: measured sanity 1.0004 and
// closed-form ratio 1.0037 (n = 12; ~80 s) — the adjoint eta kernel
// against the correspondence principle at the half-percent grade.
// Order 2 is unusable here: the relaxed state is badly resolved
// (sanity 0.95) and the DERIVATIVE amplifies the structure error
// (closed-form ratio 1.17 measured) — the example's order caveat,
// seen through the adjoint.
TEST(Adjoint, LoveNumberClosedForm) {
  auto r = LoveGate(3, 12, false);
  EXPECT_NEAR(r.sanity, 1.0, 5e-3);
  EXPECT_NEAR(r.closed, 1.0, 2e-2);
}

// GATE B: the radial-profile FD leg at order 2 (self-consistent, so
// the order-2 structure drift cancels; measured 1.1e-3 at n = 24,
// and 1.1e-2 at order 3 where the smaller directional value exposes
// the same absolute floors — order 2 is the sharper instrument here).
TEST(Adjoint, LoveNumberRadialProfile) {
  auto r = LoveGate(2, 24, true);
  EXPECT_NEAR(r.radial_fd, 1.0, 5e-3);
}
