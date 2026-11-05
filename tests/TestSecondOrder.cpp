/*
  Tests for the second-order adjoint (second_order.hpp): the derivative
  of the first-order kernels along a model direction v — the Hessian-
  vector product — against central finite differences of the first-order
  kernels themselves (two adjoint sweeps at x ± eps v), on the adjoint
  tests' unit-square configuration, static and viscoelastic, for kappa,
  mu and eta directions and kernels. The sweep is the exact derivative of
  the first-order kernels as computed, so the agreement is at the FD
  level whatever the first-order sweep's own distance from the discrete
  gradient. The symmetry defect <H v, w> − <H w, v> of this flavour is
  O(dt²) (the continuous first-order adjoint is not an exact transpose);
  it is not checked here — TestDiscreteSecondOrder gates it at rounding
  for the discrete flavour.
*/

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <numbers>
#include <vector>

#include "AdGIA.hpp"
#include "mfem.hpp"

using namespace mfem;
using namespace AdGIA;

namespace {

constexpr real_t kKappa0 = 5.4;
constexpr real_t kMu0 = 2.7;
constexpr real_t kEta0 = 2.7;
constexpr real_t kP0 = 0.05;
constexpr real_t kPi = std::numbers::pi;

void Traction2D(const Vector& x, Vector& t) {
  t.SetSize(2);
  t[0] = 0.0;
  t[1] = -kP0 * std::cos(2.0 * kPi * x[0]);
}

void Station2D(const Vector& x, Vector& v) {
  constexpr real_t s = 0.08;
  auto window = [](real_t x0, real_t c) {
    return std::exp(-(x0 - c) * (x0 - c) / (s * s)) / (s * std::sqrt(kPi));
  };
  v.SetSize(2);
  v[0] = 0.0;
  v[1] = window(x[0], 0.55) - window(x[0], 0.2);
}

FunctionCoefficient MakeBlob(real_t cx, real_t cy) {
  return FunctionCoefficient([cx, cy](const Vector& x) {
    const real_t r2 = (x[0] - cx) * (x[0] - cx) + (x[1] - cy) * (x[1] - cy);
    return std::exp(-r2 / 0.015);
  });
}

struct Bumped {
  std::unique_ptr<SumCoefficient> one_plus;
  std::unique_ptr<ProductCoefficient> value;
};
Bumped Bump(Coefficient& m0, Coefficient& chi, real_t s) {
  Bumped b;
  b.one_plus = std::make_unique<SumCoefficient>(1.0, chi, 1.0, s);
  b.value = std::make_unique<ProductCoefficient>(m0, *b.one_plus);
  return b;
}

struct Square {
  Mesh mesh;
  H1_FECollection fec;
  FiniteElementSpace fes, sfes;
  Array<int> top, bottom;
  VectorFunctionCoefficient traction, station;
  LinearForm g;
  Vector zero_vec;
  VectorConstantCoefficient zero_traction;

  Square(int n_el, int order)
      : mesh(Mesh::MakeCartesian2D(n_el, n_el, Element::QUADRILATERAL)),
        fec(order, 2),
        fes(&mesh, &fec, 2),
        sfes(&mesh, &fec),
        top(mesh.bdr_attributes.Max()),
        bottom(mesh.bdr_attributes.Max()),
        traction(2, Traction2D),
        station(2, Station2D),
        g(&fes),
        zero_vec(2),
        zero_traction((zero_vec = 0.0, zero_vec)) {
    top = 0;
    bottom = 0;
    top[2] = 1;
    bottom[0] = 1;
    g.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(station), bottom);
    g.Assemble();
  }
};

// A direction in (kappa, mu, eta) as relative patterns: m0 (1 + s chi).
struct Direction {
  Coefficient* chi_kappa = nullptr;
  Coefficient* chi_mu = nullptr;
  Coefficient* chi_eta = nullptr;
};

}  // namespace

// Static: the first-order kappa/mu kernels along the directions
// (kappa blob, mu all) differentiated along v = (mu blob) and v = (kappa
// all), against FD of the kernels.
TEST(SecondOrder, Static2D) {
  Square s(12, 2);
  auto chi_a = MakeBlob(0.4, 0.6);
  auto chi_b = MakeBlob(0.6, 0.3);
  auto chi_all = ConstantCoefficient(1.0);
  auto kappa0 = ConstantCoefficient(kKappa0);
  auto mu0 = ConstantCoefficient(kMu0);

  // The first-order kernels at (kappa, mu): directional values along the
  // two delta directions.
  struct Grad {
    real_t kappa_a = 0.0, mu_all = 0.0;
  };
  auto gradient = [&](Coefficient& kappa, Coefficient& mu) {
    IsotropicElasticRheology rheology(2, kappa, mu);
    LinearQuasiStaticTractionProblem prob(&s.fes, rheology, s.traction, s.top);
    prob.AssembleForce(0.0);
    EXPECT_TRUE(prob.Solve());
    LinearQuasiStaticTractionProblem twin(&s.fes, rheology, s.zero_traction,
                                          s.top);
    AdjointForce force;
    force.AddObservation(0.0, s.g);
    IsotropicKernel K_kappa(IsotropicKernel::Parameter::Kappa, s.sfes);
    IsotropicKernel K_mu(IsotropicKernel::Parameter::Mu, s.sfes);
    ProductCoefficient d_kappa(kappa0, chi_a), d_mu(mu0, chi_all);
    K_kappa.AddDirection(d_kappa);
    K_mu.AddDirection(d_mu);
    AdjointSolver adjoint(twin, prob.Displacement());
    adjoint.SetForce(force);
    adjoint.AddKernel(K_kappa);
    adjoint.AddKernel(K_mu);
    EXPECT_TRUE(adjoint.Run());
    return Grad{K_kappa.Directional(0), K_mu.Directional(0)};
  };

  // The base state, its tangent along v and the second-order sweep.
  IsotropicElasticRheology rheology0(2, kappa0, mu0);
  LinearQuasiStaticTractionProblem prob(&s.fes, rheology0, s.traction, s.top);
  prob.AssembleForce(0.0);
  ASSERT_TRUE(prob.Solve());
  GridFunction u(&s.fes);
  u = prob.Displacement();
  AdjointForce force;
  force.AddObservation(0.0, s.g);
  ProductCoefficient d_kappa(kappa0, chi_a), d_mu(mu0, chi_all);

  struct V {
    const char* name;
    Coefficient* v_kappa;
    Coefficient* v_mu;
    Coefficient* chi_k;
    Coefficient* chi_m;
  };
  ProductCoefficient v_mu_b(mu0, chi_b), v_kappa_all(kappa0, chi_all);
  V vs[] = {{"v = mu blob", nullptr, &v_mu_b, nullptr, &chi_b},
            {"v = kappa all", &v_kappa_all, nullptr, &chi_all, nullptr}};
  const real_t eps = 1e-3;
  for (const V& v : vs) {
    TangentDirection dir;
    dir.kappa = v.v_kappa;
    dir.mu_longterm = v.v_mu;
    LinearQuasiStaticTractionProblem tan_twin(&s.fes, rheology0,
                                              s.zero_traction, s.top);
    TangentLinearSolver tangent(tan_twin, u);
    tangent.SetDirection(dir);
    ASSERT_TRUE(tangent.Run());

    LinearQuasiStaticTractionProblem adj_twin(&s.fes, rheology0,
                                              s.zero_traction, s.top);
    LinearQuasiStaticTractionProblem der_twin(&s.fes, rheology0,
                                              s.zero_traction, s.top);
    IsotropicKernel H_kappa(IsotropicKernel::Parameter::Kappa, s.sfes);
    IsotropicKernel H_mu(IsotropicKernel::Parameter::Mu, s.sfes);
    H_kappa.AddDirection(d_kappa);
    H_mu.AddDirection(d_mu);
    IsotropicKernel G_kappa(IsotropicKernel::Parameter::Kappa, s.sfes);
    G_kappa.AddDirection(d_kappa);
    SecondOrderAdjointSolver second(adj_twin, der_twin, u,
                                    tangent.Displacement(), dir);
    second.SetForce(force);
    second.AddKernel(H_kappa);
    second.AddKernel(H_mu);
    second.AddFirstOrderKernel(G_kappa);
    ASSERT_TRUE(second.Run());

    // FD of the gradient along v.
    Grad plus, minus;
    for (const real_t sg : {eps, -eps}) {
      auto kp = Bump(kappa0, v.chi_k ? *v.chi_k : chi_all,
                     v.chi_k ? sg : 0.0);
      auto mp = Bump(mu0, v.chi_m ? *v.chi_m : chi_all, v.chi_m ? sg : 0.0);
      (sg > 0 ? plus : minus) = gradient(*kp.value, *mp.value);
    }
    const real_t fd_kappa = (plus.kappa_a - minus.kappa_a) / (2.0 * eps);
    const real_t fd_mu = (plus.mu_all - minus.mu_all) / (2.0 * eps);
    const Grad base = gradient(kappa0, mu0);
    if (std::getenv("ADGIA_TEST_VERBOSE")) {
      std::cout << v.name << ": H kappa " << H_kappa.Directional(0) << " fd "
                << fd_kappa << " | H mu " << H_mu.Directional(0) << " fd "
                << fd_mu << " | first-order kappa " << G_kappa.Directional(0)
                << " vs " << base.kappa_a << "\n";
    }
    EXPECT_NEAR(H_kappa.Directional(0) / fd_kappa, 1.0, 1e-5) << v.name;
    EXPECT_NEAR(H_mu.Directional(0) / fd_mu, 1.0, 1e-5) << v.name;
    EXPECT_NEAR(G_kappa.Directional(0) / base.kappa_a, 1.0, 1e-10) << v.name;
  }
}

// Viscoelastic: the kappa/mu/eta kernels of a final-time and of a
// continuous observation differentiated along v = (eta blob) and
// v = (mu blob, at fixed eta), against FD of the kernels; the symmetry
// defect reported.
TEST(SecondOrder, Viscoelastic2D) {
  const int n_el = 12, order = 2, n_steps = 32;
  const real_t t_final = 2.0;
  Square s(n_el, order);
  auto chi_a = MakeBlob(0.4, 0.6);
  auto chi_b = MakeBlob(0.6, 0.3);
  auto chi_all = ConstantCoefficient(1.0);
  auto kappa0 = ConstantCoefficient(kKappa0);
  auto mu0 = ConstantCoefficient(kMu0);
  auto eta0 = ConstantCoefficient(kEta0);
  const real_t dt = t_final / n_steps;
  auto cont_weight = [t_final](real_t t) { return t / t_final; };

  // Kernel directions (deltas): kappa blob a, mu all, eta blob a.
  ProductCoefficient d_kappa(kappa0, chi_a), d_mu(mu0, chi_all),
      d_eta(eta0, chi_a);

  struct Grad {
    real_t kappa = 0.0, mu = 0.0, eta = 0.0;
  };
  // The forward run + the first-order sweep at a parameter set, for one
  // force; returns the three directional values.
  auto gradient = [&](Coefficient& kappa, Coefficient& mu, Coefficient& eta,
                      bool continuous) {
    RatioCoefficient tau(eta, mu);
    auto rheology = IsotropicMaxwellRheology::Maxwell(2, kappa, mu, tau);
    LinearQuasiStaticTractionProblem prob(&s.fes, rheology, s.traction, s.top);
    ViscoelasticOperator visco(prob);
    ForwardTrajectory traj(visco, true);
    {
      ExponentialTrapezoidSolver ode;
      ode.Init(visco);
      Vector m(visco.Height());
      m = 0.0;
      real_t t = 0.0, dts = dt;
      traj.Record(m, t);
      for (int n = 1; n <= n_steps; n++) {
        ode.Step(m, t, dts);
        traj.Record(m, t);
      }
    }
    LinearQuasiStaticTractionProblem twin(&s.fes, rheology, s.zero_traction,
                                          s.top);
    ViscoelasticOperator visco_adj(twin);
    AdjointForce force;
    if (continuous) {
      force.AddContinuous([&](real_t t, Vector& f) { f.Add(cont_weight(t), s.g); });
    } else {
      force.AddObservation(traj.Time(n_steps), s.g);
    }
    IsotropicKernel K_kappa(IsotropicKernel::Parameter::Kappa, s.sfes);
    IsotropicKernel K_mu(IsotropicKernel::Parameter::Mu, s.sfes);
    IsotropicKernel K_eta(IsotropicKernel::Parameter::Eta, s.sfes, &tau);
    K_kappa.AddDirection(d_kappa);
    K_mu.AddDirection(d_mu);
    K_eta.AddDirection(d_eta);
    AdjointSolver adjoint(visco_adj, traj);
    adjoint.SetForce(force);
    adjoint.AddKernel(K_kappa);
    adjoint.AddKernel(K_mu);
    adjoint.AddKernel(K_eta);
    EXPECT_TRUE(adjoint.Run());
    return Grad{K_kappa.Directional(0), K_mu.Directional(0),
                K_eta.Directional(0)};
  };

  // The base trajectory.
  RatioCoefficient tau0(eta0, mu0);
  auto rheology0 = IsotropicMaxwellRheology::Maxwell(2, kappa0, mu0, tau0);
  LinearQuasiStaticTractionProblem prob_fwd(&s.fes, rheology0, s.traction,
                                            s.top);
  ViscoelasticOperator visco_fwd(prob_fwd);
  ForwardTrajectory trajectory(visco_fwd, true);
  {
    ExponentialTrapezoidSolver ode;
    ode.Init(visco_fwd);
    Vector m(visco_fwd.Height());
    m = 0.0;
    real_t t = 0.0, dts = dt;
    trajectory.Record(m, t);
    for (int n = 1; n <= n_steps; n++) {
      ode.Step(m, t, dts);
      trajectory.Record(m, t);
    }
  }
  AdjointForce force_end, force_cont;
  force_end.AddObservation(trajectory.Time(n_steps), s.g);
  force_cont.AddContinuous([&](real_t t, Vector& f) { f.Add(cont_weight(t), s.g); });

  struct V {
    const char* name;
    Coefficient* v_mu;
    Coefficient* v_eta;
    Coefficient* chi;
  };
  ProductCoefficient v_eta_b(eta0, chi_b), v_mu_b(mu0, chi_b);
  V vs[] = {{"v = eta blob", nullptr, &v_eta_b, &chi_b},
            {"v = mu blob", &v_mu_b, nullptr, &chi_b}};
  const real_t eps = 1e-3;
  for (const V& v : vs) {
    TangentDirection dir;
    dir.mu.assign(1, v.v_mu);
    dir.eta.assign(1, v.v_eta);
    LinearQuasiStaticTractionProblem tan_twin(&s.fes, rheology0,
                                              s.zero_traction, s.top);
    TangentLinearSolver tangent(tan_twin, trajectory);
    tangent.SetDirection(dir);
    ASSERT_TRUE(tangent.Run());

    for (int which = 0; which < 2; which++) {
      const bool continuous = which == 1;
      const AdjointForce& force = continuous ? force_cont : force_end;
      LinearQuasiStaticTractionProblem adj_twin(&s.fes, rheology0,
                                                s.zero_traction, s.top);
      LinearQuasiStaticTractionProblem der_twin(&s.fes, rheology0,
                                                s.zero_traction, s.top);
      ViscoelasticOperator visco_adj(adj_twin);
      IsotropicKernel H_kappa(IsotropicKernel::Parameter::Kappa, s.sfes);
      IsotropicKernel H_mu(IsotropicKernel::Parameter::Mu, s.sfes);
      IsotropicKernel H_eta(IsotropicKernel::Parameter::Eta, s.sfes, &tau0);
      H_kappa.AddDirection(d_kappa);
      H_mu.AddDirection(d_mu);
      H_eta.AddDirection(d_eta);
      IsotropicKernel G_eta(IsotropicKernel::Parameter::Eta, s.sfes, &tau0);
      G_eta.AddDirection(d_eta);
      SecondOrderAdjointSolver second(visco_adj, der_twin, trajectory,
                                      tangent.Trajectory(), dir);
      second.SetForce(force);
      second.AddKernel(H_kappa);
      second.AddKernel(H_mu);
      second.AddKernel(H_eta);
      second.AddFirstOrderKernel(G_eta);
      ASSERT_TRUE(second.Run());

      Grad plus, minus;
      for (const real_t sg : {eps, -eps}) {
        auto mp = Bump(mu0, *v.chi, v.v_mu ? sg : 0.0);
        auto ep = Bump(eta0, *v.chi, v.v_eta ? sg : 0.0);
        (sg > 0 ? plus : minus) =
            gradient(kappa0, *mp.value, *ep.value, continuous);
      }
      const Grad base = gradient(kappa0, mu0, eta0, continuous);
      const real_t fd_kappa = (plus.kappa - minus.kappa) / (2.0 * eps);
      const real_t fd_mu = (plus.mu - minus.mu) / (2.0 * eps);
      const real_t fd_eta = (plus.eta - minus.eta) / (2.0 * eps);
      if (std::getenv("ADGIA_TEST_VERBOSE")) {
        std::cout << v.name << (continuous ? ", continuous" : ", final-time")
                  << ": H kappa " << H_kappa.Directional(0) << " fd " << fd_kappa
                  << " | H mu " << H_mu.Directional(0) << " fd " << fd_mu
                  << " | H eta " << H_eta.Directional(0) << " fd " << fd_eta
                  << " | first-order eta " << G_eta.Directional(0) << " vs "
                  << base.eta << "\n";
      }
      // Each row against FD on the SCALE of the Hessian rows (the mu row
      // along v = eta is two orders below its neighbours, 1.2e-8 against
      // 7e-6, and its plain relative FD error is the noise floor of a
      // small cross term, 4e-4 measured; every other row
      // agrees to 1e-6).
      const real_t scale = std::max({std::abs(fd_kappa), std::abs(fd_mu),
                                     std::abs(fd_eta)});
      EXPECT_LT(std::abs(H_kappa.Directional(0) - fd_kappa) / scale, 1e-5)
          << v.name;
      EXPECT_LT(std::abs(H_mu.Directional(0) - fd_mu) / scale, 1e-5)
          << v.name;
      EXPECT_LT(std::abs(H_eta.Directional(0) - fd_eta) / scale, 1e-5)
          << v.name;
      EXPECT_NEAR(H_kappa.Directional(0) / fd_kappa, 1.0, 1e-3) << v.name;
      EXPECT_NEAR(H_mu.Directional(0) / fd_mu, 1.0, 1e-3) << v.name;
      EXPECT_NEAR(H_eta.Directional(0) / fd_eta, 1.0, 1e-3) << v.name;
      EXPECT_NEAR(G_eta.Directional(0) / base.eta, 1.0, 1e-9) << v.name;
    }
  }
}
