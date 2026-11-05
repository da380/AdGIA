/*
  Tests for the tangent-linear problem (tangent.hpp): the derivative of a
  static or quasi-static viscoelastic solution along moduli and viscosity
  directions, gated against central finite differences of fresh forward
  solves and against the first-order adjoint's kernels (adjoint.hpp) on
  the adjoint tests' configuration: a free unit square (traction problem,
  rigid modes projected) loaded on the top edge by a zero-mean, zero-torque
  normal traction, observed through a windowed two-station baseline on the
  bottom edge; Maxwell body, exponential trapezoid stepping.

  Checks:
   - static: the tangent along kappa and mu directions against FD and
     against the adjoint's directional kernels; the load direction gives
     the solution itself (linearity);
   - viscoelastic: the tangent along kappa, mu and eta directions against
     FD for a final-time, an interior-time and a continuous-in-time
     observation, and against the adjoint sweeps. The tangent is the exact
     derivative of the discrete trajectory, so the continuous-observation
     rows are as sharp as the point ones (the adjoint's carry an O(dt^2)
     time-quadrature floor there, see TestAdjoint).
*/

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <numbers>
#include <vector>

#include "AdGIA.hpp"
#include "MixedProblemTestCommon.hpp"
#include "mfem.hpp"

using namespace mfem;
using namespace AdGIA;

namespace {

constexpr real_t kKappa0 = 5.4;
constexpr real_t kMu0 = 2.7;
constexpr real_t kEta0 = 2.7;  // tau = eta / mu = 1
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

FunctionCoefficient MakeBlob() {
  return FunctionCoefficient([](const Vector& x) {
    const real_t r2 = (x[0] - 0.4) * (x[0] - 0.4) + (x[1] - 0.6) * (x[1] - 0.6);
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

}  // namespace

// Static: kappa and mu directions against FD and against the adjoint
// kernels; the load direction reproduces the solution.
TEST(Tangent, Static2D) {
  Square s(12, 2);
  auto chi_blob = MakeBlob();
  auto chi_all = ConstantCoefficient(1.0);
  auto kappa0 = ConstantCoefficient(kKappa0);
  auto mu0 = ConstantCoefficient(kMu0);

  auto measure = [&](Coefficient& kappa, Coefficient& mu,
                     GridFunction* keep = nullptr) {
    auto rheology = IsotropicElasticRheology(2, kappa, mu);
    auto prob = LinearQuasiStaticTractionProblem(&s.fes, rheology, s.traction,
                                                 s.top);
    prob.AssembleForce(0.0);
    EXPECT_TRUE(prob.Solve());
    if (keep) *keep = prob.Displacement();
    return s.g * prob.Displacement();
  };
  GridFunction u(&s.fes);
  const real_t J0 = measure(kappa0, mu0, &u);

  auto rheology0 = IsotropicElasticRheology(2, kappa0, mu0);
  auto twin = LinearQuasiStaticTractionProblem(&s.fes, rheology0,
                                               s.zero_traction, s.top);
  AdjointForce force;
  force.AddObservation(0.0, s.g);

  // The adjoint's directional kernels for the same four directions.
  auto K_kappa = IsotropicKernel(IsotropicKernel::Parameter::Kappa, s.sfes);
  auto K_mu = IsotropicKernel(IsotropicKernel::Parameter::Mu, s.sfes);
  ProductCoefficient d_kappa(kappa0, chi_blob), d_kappa_all(kappa0, chi_all),
      d_mu(mu0, chi_blob), d_mu_all(mu0, chi_all);
  K_kappa.AddDirection(d_kappa);
  K_kappa.AddDirection(d_kappa_all);
  K_mu.AddDirection(d_mu);
  K_mu.AddDirection(d_mu_all);
  {
    auto adjoint = AdjointSolver(twin, u);
    adjoint.SetForce(force);
    adjoint.AddKernel(K_kappa);
    adjoint.AddKernel(K_mu);
    ASSERT_TRUE(adjoint.Run());
  }

  const real_t eps = 1e-3;
  auto fd = [&](bool is_kappa, Coefficient& chi) {
    auto kp = Bump(kappa0, chi, is_kappa ? eps : 0.0);
    auto mp = Bump(mu0, chi, is_kappa ? 0.0 : eps);
    auto km = Bump(kappa0, chi, is_kappa ? -eps : 0.0);
    auto mm = Bump(mu0, chi, is_kappa ? 0.0 : -eps);
    return (measure(*kp.value, *mp.value) - measure(*km.value, *mm.value)) /
           (2.0 * eps);
  };
  auto tangent = [&](Coefficient* dk, Coefficient* dm) {
    TangentLinearSolver tan(twin, u);
    TangentDirection dir;
    dir.kappa = dk;
    dir.mu_longterm = dm;
    tan.SetDirection(dir);
    EXPECT_TRUE(tan.Run());
    return tan.Pair(force);
  };
  struct Row {
    Coefficient *dk, *dm;
    bool is_kappa;
    Coefficient* chi;
    real_t adjoint;
  } rows[] = {{&d_kappa, nullptr, true, &chi_blob, K_kappa.Directional(0)},
              {&d_kappa_all, nullptr, true, &chi_all, K_kappa.Directional(1)},
              {nullptr, &d_mu, false, &chi_blob, K_mu.Directional(0)},
              {nullptr, &d_mu_all, false, &chi_all, K_mu.Directional(1)}};
  for (const auto& r : rows) {
    const real_t dJ = tangent(r.dk, r.dm);
    const real_t ref = fd(r.is_kappa, *r.chi);
    if (std::getenv("ADGIA_TEST_VERBOSE")) {
      std::cout << "static: tangent " << dJ << " fd " << ref << " adjoint "
                << r.adjoint << "\n";
    }
    // Measured: FD agreement at the adjoint's level (1e-6),
    // tangent/adjoint agreement to solver tolerance.
    EXPECT_NEAR(dJ / ref, 1.0, 1e-5);
    EXPECT_NEAR(dJ / r.adjoint, 1.0, 1e-8);
  }

  // The load direction: a twin loaded by the traction itself, no moduli
  // direction, reproduces J (the problem is linear in the load).
  {
    auto loaded = LinearQuasiStaticTractionProblem(&s.fes, rheology0,
                                                   s.traction, s.top);
    TangentLinearSolver tan(loaded, u);
    tan.SetDirection(TangentDirection{});
    ASSERT_TRUE(tan.Run());
    EXPECT_NEAR(tan.Pair(force) / J0, 1.0, 1e-10);
  }
}

// Viscoelastic: kappa, mu and eta directions for a final-time, an
// interior and a continuous observation, against FD and the adjoint.
TEST(Tangent, Viscoelastic2D) {
  const int n_el = 12, order = 2, n_steps = 32;
  const real_t t_final = 2.0;
  Square s(n_el, order);
  auto chi_blob = MakeBlob();
  auto chi_all = ConstantCoefficient(1.0);
  auto kappa0 = ConstantCoefficient(kKappa0);
  auto mu0 = ConstantCoefficient(kMu0);
  auto eta0 = ConstantCoefficient(kEta0);
  const real_t dt = t_final / n_steps;
  const int n_mid = n_steps / 2;
  auto cont_weight = [t_final](real_t t) { return t / t_final; };

  struct Measured {
    real_t J_end = 0.0, J_mid = 0.0, J_cont = 0.0;
  };
  auto run_forward = [&](Coefficient& kappa, Coefficient& mu,
                         Coefficient& eta) {
    auto tau = RatioCoefficient(eta, mu);
    auto rheology = IsotropicMaxwellRheology::Maxwell(2, kappa, mu, tau);
    auto prob = LinearQuasiStaticTractionProblem(&s.fes, rheology, s.traction,
                                                 s.top);
    auto visco = ViscoelasticOperator(prob);
    ExponentialTrapezoidSolver ode;
    ode.Init(visco);
    Vector m(visco.Height());
    m = 0.0;
    real_t t = 0.0, dt_step = dt;
    Measured out;
    auto sample = [&](int n) {
      EXPECT_TRUE(visco.SolveElastic(m, t));
      const real_t Jt = s.g * prob.Displacement();
      const real_t w = (n == 0 || n == n_steps) ? 0.5 * dt : dt;
      out.J_cont += w * cont_weight(t) * Jt;
      if (n == n_mid) out.J_mid = Jt;
      if (n == n_steps) out.J_end = Jt;
    };
    sample(0);
    for (int n = 1; n <= n_steps; n++) {
      ode.Step(m, t, dt_step);
      sample(n);
    }
    return out;
  };

  // The base forward trajectory.
  auto tau0 = RatioCoefficient(eta0, mu0);
  auto rheology0 = IsotropicMaxwellRheology::Maxwell(2, kappa0, mu0, tau0);
  auto prob_fwd = LinearQuasiStaticTractionProblem(&s.fes, rheology0,
                                                   s.traction, s.top);
  auto visco_fwd = ViscoelasticOperator(prob_fwd);
  auto trajectory = ForwardTrajectory(visco_fwd, true);
  Measured base;
  {
    ExponentialTrapezoidSolver ode;
    ode.Init(visco_fwd);
    Vector m(visco_fwd.Height());
    m = 0.0;
    real_t t = 0.0, dt_step = dt;
    trajectory.Record(m, t);
    for (int n = 1; n <= n_steps; n++) {
      ode.Step(m, t, dt_step);
      trajectory.Record(m, t);
    }
    base = run_forward(kappa0, mu0, eta0);
  }

  // The twin, the three functionals.
  auto prob_twin = LinearQuasiStaticTractionProblem(&s.fes, rheology0,
                                                    s.zero_traction, s.top);
  AdjointForce force_end, force_mid, force_cont;
  force_end.AddObservation(trajectory.Time(n_steps), s.g);
  force_mid.AddObservation(trajectory.Time(n_mid), s.g);
  force_cont.AddContinuous([&](real_t t, Vector& f) {
    f.Add(cont_weight(t), s.g);
  });

  // The adjoint sweeps (one per functional), six directions each.
  std::vector<ProductCoefficient> dirs;
  dirs.emplace_back(kappa0, chi_blob);
  dirs.emplace_back(kappa0, chi_all);
  dirs.emplace_back(mu0, chi_blob);
  dirs.emplace_back(mu0, chi_all);
  dirs.emplace_back(eta0, chi_blob);
  dirs.emplace_back(eta0, chi_all);
  auto visco_adj = ViscoelasticOperator(prob_twin);
  struct Sweep {
    std::unique_ptr<IsotropicKernel> kappa, mu, eta;
  };
  auto sweep = [&](const AdjointForce& force) {
    Sweep sw;
    sw.kappa = std::make_unique<IsotropicKernel>(
        IsotropicKernel::Parameter::Kappa, s.sfes);
    sw.mu = std::make_unique<IsotropicKernel>(IsotropicKernel::Parameter::Mu,
                                              s.sfes);
    sw.eta = std::make_unique<IsotropicKernel>(IsotropicKernel::Parameter::Eta,
                                               s.sfes, &tau0);
    sw.kappa->AddDirection(dirs[0]);
    sw.kappa->AddDirection(dirs[1]);
    sw.mu->AddDirection(dirs[2]);
    sw.mu->AddDirection(dirs[3]);
    sw.eta->AddDirection(dirs[4]);
    sw.eta->AddDirection(dirs[5]);
    auto solver = AdjointSolver(visco_adj, trajectory);
    solver.SetForce(force);
    solver.AddKernel(*sw.kappa);
    solver.AddKernel(*sw.mu);
    solver.AddKernel(*sw.eta);
    EXPECT_TRUE(solver.Run());
    return sw;
  };
  Sweep sw_end = sweep(force_end), sw_mid = sweep(force_mid),
        sw_cont = sweep(force_cont);

  // The tangent runs: one per direction, all three functionals read off.
  Measured tan[3][2];
  for (int p = 0; p < 3; p++) {
    for (int c = 0; c < 2; c++) {
      TangentLinearSolver tangent(prob_twin, trajectory);
      TangentDirection dir;
      dir.mu.assign(1, nullptr);
      dir.eta.assign(1, nullptr);
      if (p == 0) dir.kappa = &dirs[c];
      if (p == 1) dir.mu[0] = &dirs[2 + c];
      if (p == 2) dir.eta[0] = &dirs[4 + c];
      tangent.SetDirection(dir);
      ASSERT_TRUE(tangent.Run());
      tan[p][c].J_end = tangent.Pair(force_end);
      tan[p][c].J_mid = tangent.Pair(force_mid);
      tan[p][c].J_cont = tangent.Pair(force_cont);
    }
  }

  // The FD table.
  const real_t eps = 1e-3;
  Coefficient* chi_of[] = {&chi_blob, &chi_all};
  Measured fd[3][2];
  for (int p = 0; p < 3; p++) {
    for (int c = 0; c < 2; c++) {
      Measured plus, minus;
      for (const real_t sg : {eps, -eps}) {
        auto kp = Bump(kappa0, *chi_of[c], p == 0 ? sg : 0.0);
        auto mp = Bump(mu0, *chi_of[c], p == 1 ? sg : 0.0);
        auto ep = Bump(eta0, *chi_of[c], p == 2 ? sg : 0.0);
        auto out = run_forward(*kp.value, *mp.value, *ep.value);
        (sg > 0 ? plus : minus) = out;
      }
      fd[p][c].J_end = (plus.J_end - minus.J_end) / (2.0 * eps);
      fd[p][c].J_mid = (plus.J_mid - minus.J_mid) / (2.0 * eps);
      fd[p][c].J_cont = (plus.J_cont - minus.J_cont) / (2.0 * eps);
    }
  }

  const char* names[] = {"kappa", "mu", "eta"};
  for (int p = 0; p < 3; p++) {
    for (int c = 0; c < 2; c++) {
      const IsotropicKernel* Ke[] = {sw_end.kappa.get(), sw_end.mu.get(),
                                     sw_end.eta.get()};
      const IsotropicKernel* Km[] = {sw_mid.kappa.get(), sw_mid.mu.get(),
                                     sw_mid.eta.get()};
      const IsotropicKernel* Kc[] = {sw_cont.kappa.get(), sw_cont.mu.get(),
                                     sw_cont.eta.get()};
      if (std::getenv("ADGIA_TEST_VERBOSE")) {
        std::cout << names[p] << (c ? " all" : " blob") << ": end "
                  << tan[p][c].J_end << " fd " << fd[p][c].J_end << " adj "
                  << Ke[p]->Directional(c) << " | mid " << tan[p][c].J_mid
                  << " fd " << fd[p][c].J_mid << " adj "
                  << Km[p]->Directional(c) << " | cont " << tan[p][c].J_cont
                  << " fd " << fd[p][c].J_cont << " adj "
                  << Kc[p]->Directional(c) << "\n";
      }
      // Against FD (the adjoint test's thresholds; the continuous rows
      // included, since the tangent differentiates the sampled sum).
      EXPECT_NEAR(tan[p][c].J_end / fd[p][c].J_end, 1.0, 2e-5)
          << names[p] << " end";
      EXPECT_NEAR(tan[p][c].J_mid / fd[p][c].J_mid, 1.0, 2e-5)
          << names[p] << " mid";
      EXPECT_NEAR(tan[p][c].J_cont / fd[p][c].J_cont, 1.0, 2e-5)
          << names[p] << " cont";
      // Against the adjoint: the point functionals pair to solver
      // tolerance; the continuous one carries the adjoint's O(dt^2)
      // time-quadrature floor (measured: 3e-6 on kappa and
      // mu, 1e-3 on eta, where TestAdjoint records 9.8e-4), while the
      // tangent's continuous rows sit at the FD level with the others.
      EXPECT_NEAR(tan[p][c].J_end / Ke[p]->Directional(c), 1.0, 1e-6)
          << names[p] << " end vs adjoint";
      EXPECT_NEAR(tan[p][c].J_mid / Km[p]->Directional(c), 1.0, 1e-6)
          << names[p] << " mid vs adjoint";
      EXPECT_NEAR(tan[p][c].J_cont / Kc[p]->Directional(c), 1.0,
                  p == 2 ? 5e-3 : 2e-5)
          << names[p] << " cont vs adjoint";
    }
  }

  // The load direction: the twin loaded by the traction reproduces the
  // forward functionals (linearity in the load).
  {
    auto loaded = LinearQuasiStaticTractionProblem(&s.fes, rheology0,
                                                   s.traction, s.top);
    TangentLinearSolver tangent(loaded, trajectory);
    tangent.SetDirection(TangentDirection{});
    ASSERT_TRUE(tangent.Run());
    EXPECT_NEAR(tangent.Pair(force_end) / base.J_end, 1.0, 1e-9);
    EXPECT_NEAR(tangent.Pair(force_mid) / base.J_mid, 1.0, 1e-9);
    EXPECT_NEAR(tangent.Pair(force_cont) / base.J_cont, 1.0, 1e-9);
  }
}

// The Born force on the mixed self-gravitating class (the elastic form on
// the solid SubMesh, the gravity terms unchanged): the static tangent
// along kappa and mu on the canned two-layer disc against FD, with a
// windowed station on the surface as the observation.
TEST(Tangent, GravitatingStatic2D) {
  using namespace self_grav_test;
  const int dim = 2, order = 2;
  Mesh parent(MeshFile(dim).c_str(), 1, 1);
  SubMesh body(SubMesh::CreateFromDomain(parent, BodyMarker(parent)));
  H1_FECollection fec(order, dim);
  FiniteElementSpace fes_u(&body, &fec, dim), fes_phi(&parent, &fec);
  ConstantCoefficient rho(kRho), kappa0(kKappa), mu0(kMu);
  FunctionCoefficient sigma(SurfaceLoad);
  Array<int> surface = SurfaceMarker(body);
  // A radial station at the angle 0.3.
  VectorFunctionCoefficient station(dim, [](const Vector& x, Vector& v) {
    const real_t th = std::atan2(x[1], x[0]), w = 0.15;
    const real_t g = std::exp(-(th - 0.3) * (th - 0.3) / (w * w));
    const real_t r = std::max(x.Norml2(), 1e-300);
    v.SetSize(2);
    v[0] = g * x[0] / r;
    v[1] = g * x[1] / r;
  });
  LinearForm g(&fes_u);
  g.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(station), surface);
  g.Assemble();

  auto measure = [&](Coefficient& kappa, Coefficient& mu, bool load,
                     GridFunction* keep = nullptr) {
    IsotropicElasticRheology rheology(dim, kappa, mu);
    LinearQuasiStaticMixedSelfGravitatingProblem prob(
        &fes_u, &fes_phi, rheology, rho, kG, kDtNDegree);
    if (load) prob.SetSurfaceLoad(sigma, surface);
    prob.SetRelTol(1e-12);
    prob.AssembleForce(0.0);
    EXPECT_TRUE(prob.Solve());
    if (keep) *keep = prob.Displacement();
    return g * prob.Displacement();
  };
  GridFunction u(&fes_u);
  measure(kappa0, mu0, true, &u);

  auto chi = FunctionCoefficient([](const Vector& x) {
    return std::exp(-((x[0] - 0.3) * (x[0] - 0.3) + (x[1] - 0.5) * (x[1] - 0.5)) /
                    0.05);
  });
  ProductCoefficient d_kappa(kappa0, chi), d_mu(mu0, chi);
  AdjointForce force;
  force.AddObservation(0.0, g);
  const real_t eps = 1e-3;
  for (int which = 0; which < 2; which++) {
    IsotropicElasticRheology rheology0(dim, kappa0, mu0);
    LinearQuasiStaticMixedSelfGravitatingProblem twin(
        &fes_u, &fes_phi, rheology0, rho, kG, kDtNDegree);
    twin.SetRelTol(1e-12);
    TangentLinearSolver tan(twin, u);
    TangentDirection dir;
    (which == 0 ? dir.kappa : dir.mu_longterm) = which == 0 ? &d_kappa : &d_mu;
    tan.SetDirection(dir);
    ASSERT_TRUE(tan.Run());
    const real_t dJ = tan.Pair(force);
    auto kp = Bump(kappa0, chi, which == 0 ? eps : 0.0);
    auto mp = Bump(mu0, chi, which == 0 ? 0.0 : eps);
    auto km = Bump(kappa0, chi, which == 0 ? -eps : 0.0);
    auto mm = Bump(mu0, chi, which == 0 ? 0.0 : -eps);
    const real_t ref = (measure(*kp.value, *mp.value, true) -
                        measure(*km.value, *mm.value, true)) /
                       (2.0 * eps);
    if (std::getenv("ADGIA_TEST_VERBOSE")) {
      std::cout << "gravitating static " << (which ? "mu" : "kappa")
                << ": tangent " << dJ << " fd " << ref << "\n";
    }
    EXPECT_NEAR(dJ / ref, 1.0, 1e-4);
  }
}

// ============================================================================
// Two-branch generalised Maxwell body (the adjoint tests' Prony
// configuration): the tangent along kappa, mu_inf and each branch's mu
// and eta against FD and against the per-branch adjoint kernels.
TEST(Tangent, Prony2D) {
  const int n_el = 12, order = 2, n_steps = 64;
  const real_t t_final = 2.0;
  Square s(n_el, order);
  auto chi_blob = MakeBlob();
  auto chi_all = ConstantCoefficient(1.0);
  auto kappa0 = ConstantCoefficient(5.4);
  auto mu_inf0 = ConstantCoefficient(1.0);
  auto mu10 = ConstantCoefficient(1.0);
  auto eta10 = ConstantCoefficient(1.0);
  auto mu20 = ConstantCoefficient(0.7);
  auto eta20 = ConstantCoefficient(0.21);
  const real_t dt = t_final / n_steps;

  auto run_forward = [&](Coefficient& kappa, Coefficient& mu_inf,
                         Coefficient& mu1, Coefficient& eta1,
                         Coefficient& mu2, Coefficient& eta2) -> real_t {
    auto tau1 = RatioCoefficient(eta1, mu1);
    auto tau2 = RatioCoefficient(eta2, mu2);
    std::vector<MaxwellBranch> branches = {{&mu1, &tau1, nullptr},
                                           {&mu2, &tau2, nullptr}};
    auto rheology = IsotropicMaxwellRheology(2, kappa, mu_inf, branches);
    auto prob = LinearQuasiStaticTractionProblem(&s.fes, rheology, s.traction,
                                                 s.top);
    auto visco = ViscoelasticOperator(prob);
    ExponentialTrapezoidSolver ode;
    ode.Init(visco);
    Vector m(visco.Height());
    m = 0.0;
    real_t t = 0.0, dt_step = dt;
    EXPECT_TRUE(visco.SolveElastic(m, t));
    for (int n = 1; n <= n_steps; n++) ode.Step(m, t, dt_step);
    EXPECT_TRUE(visco.SolveElastic(m, t));
    return s.g * prob.Displacement();
  };

  auto tau10 = RatioCoefficient(eta10, mu10);
  auto tau20 = RatioCoefficient(eta20, mu20);
  std::vector<MaxwellBranch> branches0 = {{&mu10, &tau10, nullptr},
                                          {&mu20, &tau20, nullptr}};
  auto rheology0 = IsotropicMaxwellRheology(2, kappa0, mu_inf0, branches0);
  auto prob_fwd = LinearQuasiStaticTractionProblem(&s.fes, rheology0,
                                                   s.traction, s.top);
  auto visco_fwd = ViscoelasticOperator(prob_fwd);
  auto trajectory = ForwardTrajectory(visco_fwd, true);
  {
    ExponentialTrapezoidSolver ode;
    ode.Init(visco_fwd);
    Vector m(visco_fwd.Height());
    m = 0.0;
    real_t t = 0.0, dt_step = dt;
    trajectory.Record(m, t);
    for (int n = 1; n <= n_steps; n++) {
      ode.Step(m, t, dt_step);
      trajectory.Record(m, t);
    }
  }
  auto prob_twin = LinearQuasiStaticTractionProblem(&s.fes, rheology0,
                                                    s.zero_traction, s.top);
  auto visco_adj = ViscoelasticOperator(prob_twin);

  using P = IsotropicKernel::Parameter;
  auto K_kappa = IsotropicKernel(P::Kappa, s.sfes);
  auto K_muinf = IsotropicKernel(P::MuLongTerm, s.sfes);
  auto K_mu1 = IsotropicKernel(P::Mu, s.sfes, nullptr, 0);
  auto K_eta1 = IsotropicKernel(P::Eta, s.sfes, &tau10, 0);
  auto K_mu2 = IsotropicKernel(P::Mu, s.sfes, nullptr, 1);
  auto K_eta2 = IsotropicKernel(P::Eta, s.sfes, &tau20, 1);
  IsotropicKernel* K[6] = {&K_kappa, &K_muinf, &K_mu1,
                           &K_eta1,  &K_mu2,   &K_eta2};
  Coefficient* m0_of[6] = {&kappa0, &mu_inf0, &mu10, &eta10, &mu20, &eta20};
  Coefficient* chi_of[2] = {&chi_blob, &chi_all};
  std::vector<std::unique_ptr<ProductCoefficient>> dirs;
  for (int p = 0; p < 6; p++) {
    for (int c = 0; c < 2; c++) {
      dirs.push_back(
          std::make_unique<ProductCoefficient>(*m0_of[p], *chi_of[c]));
      K[p]->AddDirection(*dirs.back());
    }
  }
  AdjointForce force;
  force.AddObservation(trajectory.Time(n_steps), s.g);
  {
    auto solver = AdjointSolver(visco_adj, trajectory);
    solver.SetForce(force);
    for (auto* k : K) solver.AddKernel(*k);
    ASSERT_TRUE(solver.Run());
  }

  const real_t eps = 1e-3;
  const char* names[6] = {"kappa", "mu_inf", "mu_1", "eta_1", "mu_2",
                          "eta_2"};
  for (int p = 0; p < 6; p++) {
    for (int c = 0; c < 2; c++) {
      TangentLinearSolver tangent(prob_twin, trajectory);
      TangentDirection dir;
      dir.mu.assign(2, nullptr);
      dir.eta.assign(2, nullptr);
      Coefficient* d = dirs[2 * p + c].get();
      switch (p) {
        case 0: dir.kappa = d; break;
        case 1: dir.mu_longterm = d; break;
        case 2: dir.mu[0] = d; break;
        case 3: dir.eta[0] = d; break;
        case 4: dir.mu[1] = d; break;
        default: dir.eta[1] = d; break;
      }
      tangent.SetDirection(dir);
      ASSERT_TRUE(tangent.Run());
      const real_t dJ = tangent.Pair(force);
      real_t fd = 0.0;
      for (const real_t sg : {eps, -eps}) {
        auto bk = Bump(kappa0, *chi_of[c], p == 0 ? sg : 0.0);
        auto bmi = Bump(mu_inf0, *chi_of[c], p == 1 ? sg : 0.0);
        auto bm1 = Bump(mu10, *chi_of[c], p == 2 ? sg : 0.0);
        auto be1 = Bump(eta10, *chi_of[c], p == 3 ? sg : 0.0);
        auto bm2 = Bump(mu20, *chi_of[c], p == 4 ? sg : 0.0);
        auto be2 = Bump(eta20, *chi_of[c], p == 5 ? sg : 0.0);
        const real_t J = run_forward(*bk.value, *bmi.value, *bm1.value,
                                     *be1.value, *bm2.value, *be2.value);
        fd += (sg > 0 ? J : -J);
      }
      fd /= 2.0 * eps;
      if (std::getenv("ADGIA_TEST_VERBOSE")) {
        std::cout << names[p] << (c ? " all" : " blob") << ": tangent " << dJ
                  << " fd " << fd << " adjoint " << K[p]->Directional(c)
                  << "\n";
      }
      // The tangent differentiates the discrete scheme: FD agreement at
      // the FD level on every branch; the adjoint rows carry the O(dt^2)
      // floor TestAdjoint records (up to 5.5e-3 on the fast branch).
      EXPECT_NEAR(dJ / fd, 1.0, 2e-5) << names[p] << (c ? " all" : " blob");
      EXPECT_NEAR(dJ / K[p]->Directional(c), 1.0, p == 4 ? 2e-2 : 3e-3)
          << names[p] << " vs adjoint";
    }
  }
}

// ============================================================================
// The gravitating Maxwell body on the canned two-layer disc: the tangent
// along kappa, mu and eta for a final-time displacement observation and
// a final-time POTENTIAL observation, against FD and the adjoint.
namespace {

real_t StationWindow(const Vector& x) {
  constexpr real_t s = 0.1;
  const real_t phi = std::atan2(x[1], x[0]);
  auto window = [phi](real_t centre) {
    real_t d = phi - centre;
    while (d > kPi) d -= 2.0 * kPi;
    while (d < -kPi) d += 2.0 * kPi;
    return std::exp(-d * d / (s * s)) / (s * std::sqrt(kPi));
  };
  return window(1.4) - window(0.3);
}

struct GravCase {
  Mesh parent;
  SubMesh body;
  H1_FECollection fec;
  FiniteElementSpace fes_u, fes_phi, sfes;
  ConstantCoefficient rho;
  FunctionCoefficient sigma;
  Array<int> surface, parent_surface;
  VectorFunctionCoefficient h_u;
  FunctionCoefficient h_phi;
  LinearForm g_u, g_phi;

  explicit GravCase(int order)
      : parent(self_grav_test::MeshFile(2).c_str(), 1, 1),
        body(SubMesh::CreateFromDomain(parent,
                                       self_grav_test::BodyMarker(parent))),
        fec(order, 2),
        fes_u(&body, &fec, 2),
        fes_phi(&parent, &fec),
        sfes(&body, &fec),
        rho(self_grav_test::kRho),
        sigma([](const Vector& x) { return self_grav_test::SurfaceLoad(x, 0.0); }),
        surface(self_grav_test::SurfaceMarker(body)),
        parent_surface(parent.bdr_attributes.Max()),
        h_u(2, [](const Vector& x, Vector& v) {
          const real_t r = x.Norml2(), w = StationWindow(x);
          v.SetSize(2);
          v[0] = w * x[0] / r;
          v[1] = w * x[1] / r;
        }),
        h_phi([](const Vector& x) { return StationWindow(x); }),
        g_u(&fes_u),
        g_phi(&fes_phi) {
    parent_surface = 0;
    parent_surface[0] = 1;
    g_u.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(h_u), surface);
    g_u.Assemble();
    g_phi.AddBoundaryIntegrator(new BoundaryLFIntegrator(h_phi),
                                parent_surface);
    g_phi.Assemble();
  }

  std::unique_ptr<LinearQuasiStaticMixedSelfGravitatingProblem> Problem(
      const Rheology& rheology, bool with_load) {
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        &fes_u, &fes_phi, rheology, rho, self_grav_test::kG,
        self_grav_test::kDtNDegree);
    if (with_load) p->SetSurfaceLoad(sigma, surface);
    p->SetRelTol(1e-11);
    return p;
  }
};

}  // namespace

TEST(Tangent, GravitatingViscoelastic2D) {
  const int order = 2, n_steps = 24;
  const real_t t_final = 2.0;
  GravCase s(order);
  auto kappa0 = ConstantCoefficient(self_grav_test::kKappa);
  auto mu0 = ConstantCoefficient(self_grav_test::kMu);
  auto eta0 = ConstantCoefficient(self_grav_test::kMu);  // tau = 1
  auto chi_blob = FunctionCoefficient([](const Vector& x) {
    const real_t dx = x[0] - 0.3, dy = x[1] - 0.35;
    return std::exp(-(dx * dx + dy * dy) / 0.02);
  });
  auto chi_all = ConstantCoefficient(1.0);
  const real_t dt = t_final / n_steps;

  struct Measured {
    real_t J_u = 0.0, J_phi = 0.0;
  };
  auto run_forward = [&](Coefficient& kappa, Coefficient& mu,
                         Coefficient& eta) {
    auto tau = RatioCoefficient(eta, mu);
    auto rheology = IsotropicMaxwellRheology::Maxwell(2, kappa, mu, tau);
    auto prob = s.Problem(rheology, true);
    auto visco = ViscoelasticOperator(*prob);
    ExponentialTrapezoidSolver ode;
    ode.Init(visco);
    Vector m(visco.Height());
    m = 0.0;
    real_t t = 0.0, dt_step = dt;
    EXPECT_TRUE(visco.SolveElastic(m, t));
    for (int n = 1; n <= n_steps; n++) ode.Step(m, t, dt_step);
    EXPECT_TRUE(visco.SolveElastic(m, t));
    Measured out;
    out.J_u = s.g_u * prob->Displacement();
    out.J_phi = s.g_phi * prob->Potential();
    return out;
  };

  auto tau0 = RatioCoefficient(eta0, mu0);
  auto rheology0 = IsotropicMaxwellRheology::Maxwell(2, kappa0, mu0, tau0);
  auto prob_fwd = s.Problem(rheology0, true);
  auto visco_fwd = ViscoelasticOperator(*prob_fwd);
  auto trajectory = ForwardTrajectory(visco_fwd, true);
  {
    ExponentialTrapezoidSolver ode;
    ode.Init(visco_fwd);
    Vector m(visco_fwd.Height());
    m = 0.0;
    real_t t = 0.0, dt_step = dt;
    trajectory.Record(m, t);
    for (int n = 1; n <= n_steps; n++) {
      ode.Step(m, t, dt_step);
      trajectory.Record(m, t);
    }
  }
  auto prob_twin = s.Problem(rheology0, false);
  auto visco_adj = ViscoelasticOperator(*prob_twin);

  std::vector<ProductCoefficient> dirs;
  dirs.emplace_back(kappa0, chi_blob);
  dirs.emplace_back(kappa0, chi_all);
  dirs.emplace_back(mu0, chi_blob);
  dirs.emplace_back(mu0, chi_all);
  dirs.emplace_back(eta0, chi_blob);
  dirs.emplace_back(eta0, chi_all);
  struct Sweep {
    std::unique_ptr<IsotropicKernel> kappa, mu, eta;
  };
  auto make_sweep = [&](const AdjointForce& force) {
    Sweep sw;
    sw.kappa = std::make_unique<IsotropicKernel>(
        IsotropicKernel::Parameter::Kappa, s.sfes);
    sw.mu = std::make_unique<IsotropicKernel>(IsotropicKernel::Parameter::Mu,
                                              s.sfes);
    sw.eta = std::make_unique<IsotropicKernel>(IsotropicKernel::Parameter::Eta,
                                               s.sfes, &tau0);
    sw.kappa->AddDirection(dirs[0]);
    sw.kappa->AddDirection(dirs[1]);
    sw.mu->AddDirection(dirs[2]);
    sw.mu->AddDirection(dirs[3]);
    sw.eta->AddDirection(dirs[4]);
    sw.eta->AddDirection(dirs[5]);
    auto solver = AdjointSolver(visco_adj, trajectory);
    solver.SetForce(force);
    solver.AddKernel(*sw.kappa);
    solver.AddKernel(*sw.mu);
    solver.AddKernel(*sw.eta);
    EXPECT_TRUE(solver.Run());
    return sw;
  };
  AdjointForce force_u, force_phi;
  force_u.AddObservation(trajectory.Time(n_steps), s.g_u);
  force_phi.AddPotentialObservation(trajectory.Time(n_steps), s.g_phi);
  auto sweep_u = make_sweep(force_u);
  auto sweep_phi = make_sweep(force_phi);

  const real_t eps = 1e-3;
  Coefficient* chi_of[] = {&chi_blob, &chi_all};
  const char* names[] = {"kappa", "mu", "eta"};
  for (int p = 0; p < 3; p++) {
    for (int c = 0; c < 2; c++) {
      TangentLinearSolver tangent(*prob_twin, trajectory);
      TangentDirection dir;
      dir.mu.assign(1, nullptr);
      dir.eta.assign(1, nullptr);
      if (p == 0) dir.kappa = &dirs[c];
      if (p == 1) dir.mu[0] = &dirs[2 + c];
      if (p == 2) dir.eta[0] = &dirs[4 + c];
      tangent.SetDirection(dir);
      ASSERT_TRUE(tangent.Run());
      const real_t dJ_u = tangent.Pair(force_u), dJ_phi = tangent.Pair(force_phi);
      Measured plus, minus;
      for (const real_t sg : {eps, -eps}) {
        auto kp = Bump(kappa0, *chi_of[c], p == 0 ? sg : 0.0);
        auto mp = Bump(mu0, *chi_of[c], p == 1 ? sg : 0.0);
        auto ep = Bump(eta0, *chi_of[c], p == 2 ? sg : 0.0);
        (sg > 0 ? plus : minus) = run_forward(*kp.value, *mp.value, *ep.value);
      }
      const real_t fd_u = (plus.J_u - minus.J_u) / (2.0 * eps);
      const real_t fd_phi = (plus.J_phi - minus.J_phi) / (2.0 * eps);
      const IsotropicKernel* Ku[] = {sweep_u.kappa.get(), sweep_u.mu.get(),
                                     sweep_u.eta.get()};
      const IsotropicKernel* Kp[] = {sweep_phi.kappa.get(), sweep_phi.mu.get(),
                                     sweep_phi.eta.get()};
      if (std::getenv("ADGIA_TEST_VERBOSE")) {
        std::cout << names[p] << (c ? " all" : " blob") << ": u " << dJ_u
                  << " fd " << fd_u << " adj " << Ku[p]->Directional(c)
                  << " | phi " << dJ_phi << " fd " << fd_phi << " adj "
                  << Kp[p]->Directional(c) << "\n";
      }
      // FD on the curved disc: the adjoint test records rows at 1.6e-4
      // and the mu blob rows at 1.3e-3 (the strain-map floor); the
      // tangent is held to the FD level.
      // Measured: every row within 3e-5 of FD (the eta blob
      // rows at 3.0e-5 and 2.7e-5), the curved disc's FD floor.
      EXPECT_NEAR(dJ_u / fd_u, 1.0, 1e-4) << names[p] << " u";
      EXPECT_NEAR(dJ_phi / fd_phi, 1.0, 1e-4) << names[p] << " phi";
      EXPECT_NEAR(dJ_u / Ku[p]->Directional(c), 1.0, 5e-3)
          << names[p] << " u vs adjoint";
      EXPECT_NEAR(dJ_phi / Kp[p]->Directional(c), 1.0, 5e-3)
          << names[p] << " phi vs adjoint";
    }
  }
}

// ============================================================================
// The frozen-shoreline water load with rotation on the canned coarse
// ball (SetWaterLoad is 3-D only): the bordered system is linear, so the
// twin carrying the same water feedback with zero data plus the Born
// force is the tangent. Observations: a cap-windowed radial displacement,
// and the strict sea-level functional through all four slots (its
// uniform-term and omega parts exercise the border pairing). Directions:
// eta blob, and an ICE-LOAD direction (the twin's own data load).
namespace {
constexpr real_t kRhoW = 1.0;
constexpr real_t kOmega = 0.4;
const real_t kG0 = 4.0 * kPi * self_grav_test::kG * self_grav_test::kRho / 3.0;

real_t MeltCap(const Vector& x, const Vector& n) {
  constexpr real_t s2 = 0.08;
  Vector d(x);
  d /= x.Norml2();
  d -= n;
  return 0.02 * std::exp(-(d * d) / s2);
}
Vector UnitDir(real_t a, real_t b, real_t c) {
  Vector n(3);
  n[0] = a;
  n[1] = b;
  n[2] = c;
  n /= n.Norml2();
  return n;
}
}  // namespace

TEST(Tangent, WaterLoadFrozen3D) {
  const int order = 2, n_steps = 8;
  const real_t t1 = 1.5;
  Mesh parent(self_grav_test::MeshFile(3).c_str(), 1, 1);
  SubMesh body(SubMesh::CreateFromDomain(parent,
                                         self_grav_test::BodyMarker(parent)));
  H1_FECollection fec(order, 3);
  FiniteElementSpace fes_u(&body, &fec, 3), fes_phi(&parent, &fec);
  ConstantCoefficient rho(self_grav_test::kRho);
  Array<int> surface = self_grav_test::SurfaceMarker(body);
  Array<int> parent_surface(parent.bdr_attributes.Max());
  parent_surface = 0;
  parent_surface[0] = 1;
  Vector moments(3);
  moments[0] = 1.2;
  moments[1] = 1.3;
  moments[2] = 2.0;
  ConstantCoefficient w(kRhoW / kG0);
  auto kappa0 = ConstantCoefficient(self_grav_test::kKappa);
  auto mu0 = ConstantCoefficient(self_grav_test::kMu);
  auto eta0 = ConstantCoefficient(self_grav_test::kMu);
  auto tau0 = RatioCoefficient(eta0, mu0);

  auto sigma_data = FunctionCoefficient([](const Vector& x) {
    const real_t r = x.Norml2(), c = x[2] / r;
    return 0.02 * (3.0 * c * c - 0.5 + 0.5 * x[0] * x[2] / (r * r));
  });
  auto zero_data = ConstantCoefficient(0.0);
  auto n_ice = UnitDir(0.5, -0.3, 0.8);
  auto delta_data = FunctionCoefficient(
      [n_ice](const Vector& x) { return MeltCap(x, n_ice); });

  // Duals: radial displacement cap at n1; the strict sea-level
  // functional at the cap n3 through all four slots.
  auto n1 = UnitDir(0.3, 0.5, 0.81), n3 = UnitDir(0.1, -0.6, 0.79);
  auto h_u = VectorFunctionCoefficient(3, [n1](const Vector& x, Vector& v) {
    v = x;
    v *= MeltCap(x, n1) / x.Norml2();
  });
  LinearForm g_u(&fes_u);
  g_u.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(h_u), surface);
  g_u.Assemble();
  auto h_sl = FunctionCoefficient([n3](const Vector& x) { return MeltCap(x, n3); });
  auto h_sl_u = VectorFunctionCoefficient(3, [n3](const Vector& x, Vector& v) {
    v = x;
    v *= -MeltCap(x, n3);
  });
  LinearForm g_sl_u(&fes_u);
  g_sl_u.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(h_sl_u), surface);
  g_sl_u.Assemble();
  auto h_sl_phi = ProductCoefficient(-1.0 / kG0, h_sl);
  LinearForm g_sl_phi(&fes_phi);
  g_sl_phi.AddBoundaryIntegrator(new BoundaryLFIntegrator(h_sl_phi),
                                 parent_surface);
  g_sl_phi.Assemble();
  const real_t sl_uniform = -g_sl_phi.Sum();
  Vector sl_omega(3);
  for (int k = 0; k < 3; k++) {
    CentrifugalPotential psi_k(3, kOmega);
    psi_k.SetUnit(k);
    ProductCoefficient hpsi(h_sl, psi_k);
    LinearForm lf(&fes_phi);
    lf.AddBoundaryIntegrator(new BoundaryLFIntegrator(hpsi), parent_surface);
    lf.Assemble();
    sl_omega[k] = -lf.Sum() / kG0;
  }

  const real_t dt = t1 / n_steps;
  bool use_water = true, use_rotation = true;
  auto make_problem = [&](const Rheology& rheology, Coefficient& data) {
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        &fes_u, &fes_phi, rheology, rho, self_grav_test::kG,
        self_grav_test::kDtNDegree);
    if (use_water) {
      p->SetWaterLoad(w, data, surface);
    } else {
      p->SetSurfaceLoad(data, surface);
    }
    if (use_rotation) p->SetRotation(kOmega, moments);
    p->SetRelTol(1e-11);
    return p;
  };
  struct Measured {
    real_t J_u = 0.0, J_sl = 0.0;
  };
  auto run_forward = [&](Coefficient& kappa, Coefficient& mu,
                         Coefficient& eta, Coefficient& data) {
    auto tau = RatioCoefficient(eta, mu);
    auto rheology = IsotropicMaxwellRheology::Maxwell(3, kappa, mu, tau);
    auto prob = make_problem(rheology, data);
    auto visco = ViscoelasticOperator(*prob);
    ExponentialTrapezoidSolver ode;
    ode.Init(visco);
    Vector m(visco.Height());
    m = 0.0;
    real_t t = 0.0, dt_step = dt;
    EXPECT_TRUE(visco.SolveElastic(m, t));
    for (int n = 1; n <= n_steps; n++) ode.Step(m, t, dt_step);
    EXPECT_TRUE(visco.SolveElastic(m, t));
    Measured out;
    out.J_u = g_u * prob->Displacement();
    out.J_sl = g_sl_u * prob->Displacement() + g_sl_phi * prob->Potential() +
               sl_uniform * prob->UniformPotentialTerm();
    if (prob->AngularVelocity().Size() == 3) {
      out.J_sl += sl_omega * prob->AngularVelocity();
    }
    return out;
  };

  auto rheology0 = IsotropicMaxwellRheology::Maxwell(3, kappa0, mu0, tau0);
  auto chi_blob3 = FunctionCoefficient([](const Vector& x) {
    const real_t dx = x[0] - 0.3, dy = x[1] - 0.35, dz = x[2] - 0.4;
    return std::exp(-(dx * dx + dy * dy + dz * dz) / 0.03);
  });
  ProductCoefficient d_eta(eta0, chi_blob3);
  const real_t eps = 1e-3;

  // The eta direction, the DISPLACEMENT observation only, under water
  // alone, rotation alone and both: localises what the bordered solves
  // do to the tangent.
  for (int cfg = 0; cfg < 3; cfg++) {
    use_water = cfg != 1;
    use_rotation = cfg != 0;
    auto prob_c = make_problem(rheology0, sigma_data);
    auto visco_c = ViscoelasticOperator(*prob_c);
    auto traj_c = ForwardTrajectory(visco_c, true);
    {
      ExponentialTrapezoidSolver ode;
      ode.Init(visco_c);
      Vector m(visco_c.Height());
      m = 0.0;
      real_t t = 0.0, dt_step = dt;
      traj_c.Record(m, t);
      for (int n = 1; n <= n_steps; n++) {
        ode.Step(m, t, dt_step);
        traj_c.Record(m, t);
      }
    }
    AdjointForce fu;
    fu.AddObservation(traj_c.Time(n_steps), g_u);
    auto twin = make_problem(rheology0, zero_data);
    TangentLinearSolver tangent(*twin, traj_c);
    TangentDirection dir;
    dir.eta.assign(1, &d_eta);
    tangent.SetDirection(dir);
    ASSERT_TRUE(tangent.Run());
    const real_t dJ = tangent.Pair(fu);
    Measured plus, minus;
    for (const real_t sg : {eps, -eps}) {
      auto ep = Bump(eta0, chi_blob3, sg);
      (sg > 0 ? plus : minus) = run_forward(kappa0, mu0, *ep.value, sigma_data);
    }
    const real_t fd = (plus.J_u - minus.J_u) / (2.0 * eps);
    if (std::getenv("ADGIA_TEST_VERBOSE")) {
      std::cout << "eta blob, water " << use_water << " rotation "
                << use_rotation << ": tangent " << dJ << " fd " << fd
                << " ratio " << dJ / fd << "\n";
    }
  }
  use_water = true;
  use_rotation = true;

  auto prob_fwd = make_problem(rheology0, sigma_data);
  auto visco_fwd = ViscoelasticOperator(*prob_fwd);
  auto trajectory = ForwardTrajectory(visco_fwd, true);
  {
    ExponentialTrapezoidSolver ode;
    ode.Init(visco_fwd);
    Vector m(visco_fwd.Height());
    m = 0.0;
    real_t t = 0.0, dt_step = dt;
    trajectory.Record(m, t);
    for (int n = 1; n <= n_steps; n++) {
      ode.Step(m, t, dt_step);
      trajectory.Record(m, t);
    }
  }

  AdjointForce force_u, force_sl;
  force_u.AddObservation(trajectory.Time(n_steps), g_u);
  AdjointLoad sl;
  sl.u = &g_sl_u;
  sl.phi = &g_sl_phi;
  sl.uniform = sl_uniform;
  sl.omega = &sl_omega;
  force_sl.AddObservation(trajectory.Time(n_steps), sl);

  // (a) eta blob on the zero-data twin.
  {
    auto twin = make_problem(rheology0, zero_data);
    TangentLinearSolver tangent(*twin, trajectory);
    TangentDirection dir;
    dir.eta.assign(1, &d_eta);
    tangent.SetDirection(dir);
    ASSERT_TRUE(tangent.Run());
    const real_t dJ_u = tangent.Pair(force_u), dJ_sl = tangent.Pair(force_sl);
    Measured plus, minus;
    for (const real_t sg : {eps, -eps}) {
      auto ep = Bump(eta0, chi_blob3, sg);
      (sg > 0 ? plus : minus) = run_forward(kappa0, mu0, *ep.value, sigma_data);
    }
    const real_t fd_u = (plus.J_u - minus.J_u) / (2.0 * eps);
    const real_t fd_sl = (plus.J_sl - minus.J_sl) / (2.0 * eps);
    if (std::getenv("ADGIA_TEST_VERBOSE")) {
      std::cout << "water, eta blob: u " << dJ_u << " fd " << fd_u << " | sl "
                << dJ_sl << " fd " << fd_sl << "\n";
    }
    EXPECT_NEAR(dJ_u / fd_u, 1.0, 1e-4);
    EXPECT_NEAR(dJ_sl / fd_sl, 1.0, 1e-4);
  }
  // (b) the ice-load direction: the twin's data load is delta sigma.
  {
    auto twin = make_problem(rheology0, delta_data);
    TangentLinearSolver tangent(*twin, trajectory);
    tangent.SetDirection(TangentDirection{});
    ASSERT_TRUE(tangent.Run());
    const real_t dJ_u = tangent.Pair(force_u), dJ_sl = tangent.Pair(force_sl);
    Measured plus, minus;
    for (const real_t sg : {eps, -eps}) {
      SumCoefficient data(sigma_data, delta_data, 1.0, sg);
      (sg > 0 ? plus : minus) = run_forward(kappa0, mu0, eta0, data);
    }
    const real_t fd_u = (plus.J_u - minus.J_u) / (2.0 * eps);
    const real_t fd_sl = (plus.J_sl - minus.J_sl) / (2.0 * eps);
    if (std::getenv("ADGIA_TEST_VERBOSE")) {
      std::cout << "water, ice load: u " << dJ_u << " fd " << fd_u << " | sl "
                << dJ_sl << " fd " << fd_sl << "\n";
    }
    // Linear in the load: FD is exact to round-off and solver tolerance.
    EXPECT_NEAR(dJ_u / fd_u, 1.0, 1e-7);
    EXPECT_NEAR(dJ_sl / fd_sl, 1.0, 1e-7);
  }
}

// The 3-D ball (curved tets, where the internal space does NOT resolve
// the strain exactly, so the effective solve and the unrelaxed form
// differ): the three directions against FD. This is the case that
// caught the tangent differentiating the unrelaxed form instead of the
// assembled effective one (mu and eta rows 0.5 % off at 8
// steps, halving with the step count; exact once the Born force carries
// the weights and their derivative).
TEST(Tangent, CurvedElements3D) {
  const int order = 2;
  Mesh parent(self_grav_test::MeshFile(3).c_str(), 1, 1);
  SubMesh body(SubMesh::CreateFromDomain(parent,
                                         self_grav_test::BodyMarker(parent)));
  H1_FECollection fec(order, 3);
  FiniteElementSpace fes_u(&body, &fec, 3), fes_phi(&parent, &fec);
  ConstantCoefficient rho(self_grav_test::kRho);
  Array<int> surface = self_grav_test::SurfaceMarker(body);
  auto kappa0 = ConstantCoefficient(self_grav_test::kKappa);
  auto mu0 = ConstantCoefficient(self_grav_test::kMu);
  auto eta0 = ConstantCoefficient(self_grav_test::kMu);
  auto sigma_data = FunctionCoefficient([](const Vector& x) {
    const real_t r = x.Norml2(), c = x[2] / r;
    return 0.02 * (3.0 * c * c - 0.5 + 0.5 * x[0] * x[2] / (r * r));
  });
  auto zero_data = ConstantCoefficient(0.0);
  auto n1 = UnitDir(0.3, 0.5, 0.81);
  auto h_u = VectorFunctionCoefficient(3, [n1](const Vector& x, Vector& v) {
    v = x;
    v *= MeltCap(x, n1) / x.Norml2();
  });
  LinearForm g_u(&fes_u);
  g_u.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(h_u), surface);
  g_u.Assemble();
  auto chi_blob3 = FunctionCoefficient([](const Vector& x) {
    const real_t dx = x[0] - 0.3, dy = x[1] - 0.35, dz = x[2] - 0.4;
    return std::exp(-(dx * dx + dy * dy + dz * dz) / 0.03);
  });
  auto chi_all = ConstantCoefficient(1.0);
  const real_t eps = 1e-3, t1 = 1.5;
  for (int n_steps : {8}) {
    const real_t dt = t1 / n_steps;
    auto make_problem = [&](const Rheology& rheology, Coefficient& data) {
      auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
          &fes_u, &fes_phi, rheology, rho, self_grav_test::kG,
          self_grav_test::kDtNDegree);
      p->SetSurfaceLoad(data, surface);
      p->SetRelTol(1e-11);
      return p;
    };
    auto run_forward = [&](Coefficient& kappa, Coefficient& mu,
                           Coefficient& eta) {
      auto tau = RatioCoefficient(eta, mu);
      auto rheology = IsotropicMaxwellRheology::Maxwell(3, kappa, mu, tau);
      auto prob = make_problem(rheology, sigma_data);
      auto visco = ViscoelasticOperator(*prob);
      ExponentialTrapezoidSolver ode;
      ode.Init(visco);
      Vector m(visco.Height());
      m = 0.0;
      real_t t = 0.0, dt_step = dt;
      EXPECT_TRUE(visco.SolveElastic(m, t));
      for (int n = 1; n <= n_steps; n++) ode.Step(m, t, dt_step);
      EXPECT_TRUE(visco.SolveElastic(m, t));
      return g_u * prob->Displacement();
    };
    auto tau0 = RatioCoefficient(eta0, mu0);
    auto rheology0 = IsotropicMaxwellRheology::Maxwell(3, kappa0, mu0, tau0);
    auto prob_fwd = make_problem(rheology0, sigma_data);
    auto visco_fwd = ViscoelasticOperator(*prob_fwd);
    auto trajectory = ForwardTrajectory(visco_fwd, true);
    {
      ExponentialTrapezoidSolver ode;
      ode.Init(visco_fwd);
      Vector m(visco_fwd.Height());
      m = 0.0;
      real_t t = 0.0, dt_step = dt;
      trajectory.Record(m, t);
      for (int n = 1; n <= n_steps; n++) {
        ode.Step(m, t, dt_step);
        trajectory.Record(m, t);
      }
    }
    AdjointForce fu;
    fu.AddObservation(trajectory.Time(n_steps), g_u);
    const char* names[] = {"kappa", "mu", "eta"};
    for (int p = 0; p < 3; p++) {
      for (int c = 0; c < 2; c++) {
        Coefficient& chi = c ? static_cast<Coefficient&>(chi_all)
                             : static_cast<Coefficient&>(chi_blob3);
        ProductCoefficient d(p == 0 ? kappa0 : p == 1 ? mu0 : eta0, chi);
        auto twin = make_problem(rheology0, zero_data);
        TangentLinearSolver tangent(*twin, trajectory);
        TangentDirection dir;
        dir.mu.assign(1, nullptr);
        dir.eta.assign(1, nullptr);
        if (p == 0) dir.kappa = &d;
        if (p == 1) dir.mu[0] = &d;
        if (p == 2) dir.eta[0] = &d;
        tangent.SetDirection(dir);
        ASSERT_TRUE(tangent.Run());
        const real_t dJ = tangent.Pair(fu);
        real_t fd = 0.0;
        for (const real_t sg : {eps, -eps}) {
          auto kp = Bump(kappa0, chi, p == 0 ? sg : 0.0);
          auto mp = Bump(mu0, chi, p == 1 ? sg : 0.0);
          auto ep = Bump(eta0, chi, p == 2 ? sg : 0.0);
          const real_t J = run_forward(*kp.value, *mp.value, *ep.value);
          fd += sg > 0 ? J : -J;
        }
        fd /= 2.0 * eps;
        if (std::getenv("ADGIA_TEST_VERBOSE")) {
          std::cout << "curved 3-D n=" << n_steps << " " << names[p]
                    << (c ? " all" : " blob") << ": tangent " << dJ << " fd "
                    << fd << " ratio " << dJ / fd << "\n";
        }
        EXPECT_NEAR(dJ / fd, 1.0, 1e-5) << names[p] << (c ? " all" : " blob");
      }
    }
  }
}
