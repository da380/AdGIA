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
    // Measured 5 Nov 2026: FD agreement at the adjoint's level (1e-6),
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
      // time-quadrature floor (measured 5 Nov 2026: 3e-6 on kappa and
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
