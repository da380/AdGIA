/*
  Tests for the common adjoint structure (adjoint.hpp): the finite-
  difference gates of gate GA0 run through AdjointForce /
  ForwardTrajectory / AdjointSolver / IsotropicKernel instead of the
  hand-rolled loops of the two adjoint examples.

  The configuration is self-contained: a free unit square (traction
  problem, rigid modes projected), loaded on the top edge by a zero-mean,
  zero-torque normal traction p0 cos(2 pi x) (stepped on at t = 0 for the
  viscoelastic cases), observed through a windowed two-station vertical
  baseline on the bottom edge. The observation functional need not be
  rigid-invariant for the gates to be clean: both the forward and the
  finite-difference solves run through the same rigid projector, and the
  elastic form annihilates rigid motions, so the discrete kernel identity
  holds for the projected functional exactly (the examples' remark, made
  precise in doc/planning/adjoint_note.md).

  Gates (thresholds encode measurements on this configuration; see each
  check):
   - static kappa/mu kernels against central finite differences, 2-D
     (order 2) and 3-D (order 1, coarse hexes — the dimension-general
     component contractions);
   - viscoelastic kappa/mu/eta kernels against central FD for a final-time
     observation (the viscoelastic_adjoint table's configuration), an
     INTERIOR observation time (the trapezoid weight split across the m+
     jump — not exercised by the examples), and a CONTINUOUS-in-time
     observation int w(t) g.u dt driven through the force hook;
   - the store-all and re-solve trajectory policies give identical
     kernels (the slaved-state identity: u is recovered from m by one
     solve, so the two paths differ only through solver roundoff);
   - the dual export and display field are consistent with the
     directional values (interpolation-grade, not identity-grade: the
     directional path pairs the density against the direction by
     quadrature, the dual pairs it against interpolants).
*/

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <memory>
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

// The traction on the top edge/face: normal, zero mean and zero torque.
void Traction2D(const Vector& x, Vector& t) {
  t.SetSize(2);
  t[0] = 0.0;
  t[1] = -kP0 * std::cos(2.0 * M_PI * x[0]);
}

void Traction3D(const Vector& x, Vector& t) {
  t.SetSize(3);
  t = 0.0;
  t[2] = -kP0 * std::cos(2.0 * M_PI * x[0]) * std::cos(2.0 * M_PI * x[1]);
}

// The windowed two-station baseline on the bottom edge/face: normalised
// Gaussian windows at x = 0.3 and 0.7 weighting the vertical displacement
// with opposite signs.
void Station2D(const Vector& x, Vector& v) {
  constexpr real_t s = 0.08;
  auto window = [](real_t x0, real_t c) {
    return std::exp(-(x0 - c) * (x0 - c) / (s * s)) / (s * std::sqrt(M_PI));
  };
  v.SetSize(2);
  v[0] = 0.0;
  v[1] = window(x[0], 0.55) - window(x[0], 0.2);
}

void Station3D(const Vector& x, Vector& v) {
  constexpr real_t s = 0.12;
  auto window = [](real_t x0, real_t y0, real_t cx) {
    return std::exp(-((x0 - cx) * (x0 - cx) + (y0 - 0.5) * (y0 - 0.5)) /
                    (s * s)) /
           (s * s * M_PI);
  };
  v.SetSize(3);
  v = 0.0;
  v[2] = window(x[0], x[1], 0.55) - window(x[0], x[1], 0.2);
}

// The perturbation patterns.
FunctionCoefficient MakeBlob(int dim) {
  return FunctionCoefficient([dim](const Vector& x) {
    real_t r2 = (x[0] - 0.4) * (x[0] - 0.4) + (x[1] - 0.6) * (x[1] - 0.6);
    if (dim == 3) {
      r2 += (x[2] - 0.5) * (x[2] - 0.5);
    }
    return std::exp(-r2 / 0.015);
  });
}

// m0 (1 + s chi), owned.
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

}  // namespace

// Static kappa/mu kernels against central finite differences, through the
// common structure, in two and three dimensions.
static void StaticKernelGate(int dim, int n_el, int order, real_t tol,
                             real_t dual_tol) {
  auto mesh = dim == 2 ? Mesh::MakeCartesian2D(n_el, n_el,
                                               Element::QUADRILATERAL)
                       : Mesh::MakeCartesian3D(n_el, n_el, n_el,
                                               Element::HEXAHEDRON);
  auto fec = H1_FECollection(order, dim);
  auto fes = FiniteElementSpace(&mesh, &fec, dim);
  auto sfes = FiniteElementSpace(&mesh, &fec);

  // Cartesian boundary attributes: bottom is 1, top is 3 (2-D) / bottom 1,
  // top 6 (3-D).
  auto top = Array<int>(mesh.bdr_attributes.Max());
  auto bottom = Array<int>(mesh.bdr_attributes.Max());
  top = 0;
  bottom = 0;
  top[dim == 2 ? 2 : 5] = 1;
  bottom[0] = 1;

  auto traction =
      VectorFunctionCoefficient(dim, dim == 2 ? Traction2D : Traction3D);
  auto station =
      VectorFunctionCoefficient(dim, dim == 2 ? Station2D : Station3D);
  auto g = LinearForm(&fes);
  g.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(station), bottom);
  g.Assemble();

  auto chi_blob = MakeBlob(dim);
  auto chi_all = ConstantCoefficient(1.0);
  auto kappa0 = ConstantCoefficient(kKappa0);
  auto mu0 = ConstantCoefficient(kMu0);

  auto measure = [&](Coefficient& kappa, Coefficient& mu,
                     GridFunction* keep = nullptr) {
    auto rheology = IsotropicElasticRheology(dim, kappa, mu);
    auto prob = LinearQuasiStaticTractionProblem(&fes, rheology, traction,
                                                 top);
    prob.AssembleForce(0.0);
    EXPECT_TRUE(prob.Solve());
    if (keep) {
      *keep = prob.Displacement();
    }
    return g * prob.Displacement();
  };

  auto u = GridFunction(&fes);
  measure(kappa0, mu0, &u);

  // The adjoint through the common structure: the zero-load twin, one
  // point term, kappa and mu accumulators with both directions.
  auto rheology0 = IsotropicElasticRheology(dim, kappa0, mu0);
  auto zero_vec = Vector(dim);
  zero_vec = 0.0;
  auto zero_traction = VectorConstantCoefficient(zero_vec);
  auto adj_prob = LinearQuasiStaticTractionProblem(&fes, rheology0,
                                                   zero_traction, top);
  AdjointForce force;
  force.AddObservation(0.0, g);

  auto K_kappa = IsotropicKernel(IsotropicKernel::Parameter::Kappa, sfes);
  auto K_mu = IsotropicKernel(IsotropicKernel::Parameter::Mu, sfes);
  auto d_kappa = ProductCoefficient(kappa0, chi_blob);
  auto d_kappa_all = ProductCoefficient(kappa0, chi_all);
  auto d_mu = ProductCoefficient(mu0, chi_blob);
  auto d_mu_all = ProductCoefficient(mu0, chi_all);
  K_kappa.AddDirection(d_kappa);
  K_kappa.AddDirection(d_kappa_all);
  K_mu.AddDirection(d_mu);
  K_mu.AddDirection(d_mu_all);
  K_kappa.SetDualSpace(sfes);

  auto solver = AdjointSolver(adj_prob, u);
  solver.SetForce(force);
  solver.AddKernel(K_kappa);
  solver.AddKernel(K_mu);
  ASSERT_TRUE(solver.Run());

  // Central finite differences of fresh forward solves.
  const real_t eps = 1e-3;
  auto fd = [&](bool is_kappa, Coefficient& chi) {
    auto kp = Bump(kappa0, chi, is_kappa ? eps : 0.0);
    auto mp = Bump(mu0, chi, is_kappa ? 0.0 : eps);
    auto km = Bump(kappa0, chi, is_kappa ? -eps : 0.0);
    auto mm = Bump(mu0, chi, is_kappa ? 0.0 : -eps);
    return (measure(*kp.value, *mp.value) - measure(*km.value, *mm.value)) /
           (2.0 * eps);
  };
  EXPECT_NEAR(K_kappa.Directional(0) / fd(true, chi_blob), 1.0, tol);
  EXPECT_NEAR(K_kappa.Directional(1) / fd(true, chi_all), 1.0, tol);
  EXPECT_NEAR(K_mu.Directional(0) / fd(false, chi_blob), 1.0, tol);
  EXPECT_NEAR(K_mu.Directional(1) / fd(false, chi_all), 1.0, tol);

  // Dual-export consistency: pairing the exported dual with the
  // INTERPOLATED blob direction reproduces the directional value at
  // interpolation grade (a constant direction would be exact and prove
  // nothing). Measured 6e-4 in 2-D, 2e-2 on the coarse 3-D hexes.
  Vector j;
  K_kappa.TrueDual(j);
  auto chi_gf = GridFunction(&sfes);
  chi_gf.ProjectCoefficient(d_kappa);
  EXPECT_NEAR((j * chi_gf) / K_kappa.Directional(0), 1.0, dual_tol);
}

TEST(Adjoint, StaticKernels2D) {
  // Measured on this configuration (12x12 quads, order 2): rel. diffs
  // 2.6e-7 (blob) and 1.1e-6 (body-wide).
  StaticKernelGate(2, 12, 2, 1e-5, 5e-3);
}

TEST(Adjoint, StaticKernels3D) {
  // Coarse hexes, order 1: the gate is on the dimension-general
  // component contractions, not on sharpness. Measured 4.1e-3 (blob),
  // 1.1e-6 (body-wide); the dual check is scale-only here (0.25
  // measured: the blob is unresolved at h = 1/6 > its width, so its
  // nodal interpolant is poor — the directional path does not
  // interpolate and stays sharp).
  StaticKernelGate(3, 6, 1, 1e-2, 4e-1);
}

// The viscoelastic gates: one forward trajectory serves a final-time
// observation, an interior-time observation and a continuous-in-time
// observation; the FD loop re-runs the forward problem once per
// perturbation sign and evaluates all three functionals on the way.
TEST(Adjoint, ViscoelasticKernels) {
  const int dim = 2, n_el = 12, order = 2, n_steps = 32;
  const real_t t_final = 2.0;
  auto mesh = Mesh::MakeCartesian2D(n_el, n_el, Element::QUADRILATERAL);
  auto fec = H1_FECollection(order, dim);
  auto fes = FiniteElementSpace(&mesh, &fec, dim);
  auto sfes = FiniteElementSpace(&mesh, &fec);

  auto top = Array<int>(mesh.bdr_attributes.Max());
  auto bottom = Array<int>(mesh.bdr_attributes.Max());
  top = 0;
  bottom = 0;
  top[2] = 1;
  bottom[0] = 1;
  auto traction = VectorFunctionCoefficient(dim, Traction2D);
  auto station = VectorFunctionCoefficient(dim, Station2D);
  auto g = LinearForm(&fes);
  g.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(station), bottom);
  g.Assemble();

  auto chi_blob = MakeBlob(dim);
  auto chi_all = ConstantCoefficient(1.0);
  auto kappa0 = ConstantCoefficient(kKappa0);
  auto mu0 = ConstantCoefficient(kMu0);
  auto eta0 = ConstantCoefficient(kEta0);

  const real_t dt = t_final / n_steps;
  const int n_mid = n_steps / 2;
  // The continuous observation weight w(t) = t / t_final.
  auto cont_weight = [t_final](real_t t) { return t / t_final; };

  // One forward run: step to t_final; returns (J at t_final, J at the
  // interior node, the trapezoid quadrature of w(t) g.u(t)); optionally
  // records the trajectory (twice: both store policies).
  struct Measured {
    real_t J_end = 0.0, J_mid = 0.0, J_cont = 0.0;
  };
  auto run_forward = [&](Coefficient& kappa, Coefficient& mu,
                         Coefficient& eta, ForwardTrajectory* keep,
                         ForwardTrajectory* keep_lean) {
    auto tau = RatioCoefficient(eta, mu);
    auto rheology = IsotropicMaxwellRheology::Maxwell(dim, kappa, mu, tau);
    auto prob = LinearQuasiStaticTractionProblem(&fes, rheology, traction,
                                                 top);
    auto visco = ViscoelasticOperator(prob);
    ExponentialTrapezoidSolver ode;
    ode.Init(visco);
    Vector m(visco.Height());
    m = 0.0;
    real_t t = 0.0, dt_step = dt;
    Measured out;
    auto sample = [&](int n) {
      EXPECT_TRUE(visco.SolveElastic(m, t));
      const real_t Jt = g * prob.Displacement();
      const real_t w = (n == 0 || n == n_steps) ? 0.5 * dt : dt;
      out.J_cont += w * cont_weight(t) * Jt;
      if (n == n_mid) {
        out.J_mid = Jt;
      }
      if (n == n_steps) {
        out.J_end = Jt;
      }
      if (keep) {
        keep->Record(m, t);
      }
      if (keep_lean) {
        keep_lean->Record(m, t);
      }
    };
    sample(0);
    for (int n = 1; n <= n_steps; n++) {
      ode.Step(m, t, dt_step);
      sample(n);
    }
    return out;
  };

  // === The base forward run and the three adjoint sweeps ===
  auto tau0 = RatioCoefficient(eta0, mu0);
  auto rheology0 = IsotropicMaxwellRheology::Maxwell(dim, kappa0, mu0, tau0);
  auto prob_fwd = LinearQuasiStaticTractionProblem(&fes, rheology0, traction,
                                                   top);
  auto visco_fwd = ViscoelasticOperator(prob_fwd);
  auto trajectory = ForwardTrajectory(visco_fwd, true);
  auto trajectory_lean = ForwardTrajectory(visco_fwd, false);
  {
    // The base run records both trajectory policies on the persistent
    // forward operator (Record solves and verifies internally).
    ExponentialTrapezoidSolver ode;
    ode.Init(visco_fwd);
    Vector m(visco_fwd.Height());
    m = 0.0;
    real_t t = 0.0, dt_step = dt;
    trajectory.Record(m, t);
    trajectory_lean.Record(m, t);
    for (int n = 1; n <= n_steps; n++) {
      ode.Step(m, t, dt_step);
      trajectory.Record(m, t);
      trajectory_lean.Record(m, t);
    }
  }

  // The zero-load adjoint twin.
  auto zero_vec = Vector(dim);
  zero_vec = 0.0;
  auto zero_traction = VectorConstantCoefficient(zero_vec);
  auto prob_adj = LinearQuasiStaticTractionProblem(&fes, rheology0,
                                                   zero_traction, top);
  auto visco_adj = ViscoelasticOperator(prob_adj);

  struct Sweep {
    std::unique_ptr<IsotropicKernel> kappa, mu, eta;
    // directions: 0 = blob, 1 = all, per parameter
  };
  auto make_sweep = [&](ForwardTrajectory& traj, const AdjointForce& force,
                        std::vector<ProductCoefficient>& dirs) {
    Sweep s;
    s.kappa = std::make_unique<IsotropicKernel>(
        IsotropicKernel::Parameter::Kappa, sfes);
    s.mu = std::make_unique<IsotropicKernel>(IsotropicKernel::Parameter::Mu,
                                             sfes);
    s.eta = std::make_unique<IsotropicKernel>(IsotropicKernel::Parameter::Eta,
                                              sfes, &tau0);
    s.kappa->AddDirection(dirs[0]);
    s.kappa->AddDirection(dirs[1]);
    s.mu->AddDirection(dirs[2]);
    s.mu->AddDirection(dirs[3]);
    s.eta->AddDirection(dirs[4]);
    s.eta->AddDirection(dirs[5]);
    auto solver = AdjointSolver(visco_adj, traj);
    solver.SetForce(force);
    solver.AddKernel(*s.kappa);
    solver.AddKernel(*s.mu);
    solver.AddKernel(*s.eta);
    EXPECT_TRUE(solver.Run());
    return s;
  };
  std::vector<ProductCoefficient> dirs;
  dirs.emplace_back(kappa0, chi_blob);
  dirs.emplace_back(kappa0, chi_all);
  dirs.emplace_back(mu0, chi_blob);
  dirs.emplace_back(mu0, chi_all);
  dirs.emplace_back(eta0, chi_blob);
  dirs.emplace_back(eta0, chi_all);

  AdjointForce force_end;
  force_end.AddObservation(trajectory.Time(n_steps), g);
  AdjointForce force_mid;
  force_mid.AddObservation(trajectory.Time(n_mid), g);
  AdjointForce force_cont;
  force_cont.AddContinuous([&](real_t t, Vector& f) {
    f.Add(cont_weight(t), g);
  });

  auto sweep_end = make_sweep(trajectory, force_end, dirs);
  auto sweep_mid = make_sweep(trajectory, force_mid, dirs);
  auto sweep_cont = make_sweep(trajectory, force_cont, dirs);
  // The re-solve (lean) trajectory policy on the final-time sweep.
  auto sweep_lean = make_sweep(trajectory_lean, force_end, dirs);

  // Store-all and re-solve give the same kernels through solver roundoff
  // (the slaved-state identity; measured 1.1e-11).
  for (int i = 0; i < 2; i++) {
    EXPECT_NEAR(sweep_lean.kappa->Directional(i) /
                    sweep_end.kappa->Directional(i),
                1.0, 1e-9);
    EXPECT_NEAR(sweep_lean.mu->Directional(i) / sweep_end.mu->Directional(i),
                1.0, 1e-9);
    EXPECT_NEAR(sweep_lean.eta->Directional(i) / sweep_end.eta->Directional(i),
                1.0, 1e-9);
  }

  // === The FD table: one rerun per sign and direction, all three
  // functionals sampled on the way ===
  const real_t eps = 1e-3;
  Coefficient* m0_of[] = {&kappa0, &mu0, &eta0};
  Coefficient* chi_of[] = {&chi_blob, &chi_all};
  // fd[param][dir] per functional
  Measured fd[3][2];
  for (int p = 0; p < 3; p++) {
    for (int c = 0; c < 2; c++) {
      Measured plus, minus;
      for (const real_t s : {eps, -eps}) {
        auto kp = Bump(kappa0, *chi_of[c], p == 0 ? s : 0.0);
        auto mp = Bump(mu0, *chi_of[c], p == 1 ? s : 0.0);
        auto ep = Bump(eta0, *chi_of[c], p == 2 ? s : 0.0);
        auto out =
            run_forward(*kp.value, *mp.value, *ep.value, nullptr, nullptr);
        (s > 0 ? plus : minus) = out;
      }
      fd[p][c].J_end = (plus.J_end - minus.J_end) / (2.0 * eps);
      fd[p][c].J_mid = (plus.J_mid - minus.J_mid) / (2.0 * eps);
      fd[p][c].J_cont = (plus.J_cont - minus.J_cont) / (2.0 * eps);
    }
  }

  // Thresholds encode measurements on this configuration (uniform
  // parameters on affine quads: the nodal-sampling floor of the examples
  // is absent, and the point/interior rows sit at 3.3e-6 or below). The
  // continuous-observation eta rows carry an O(dt^2) time-quadrature
  // floor — the forward J is a trapezoid sample while the stepper's
  // effective solve is not its exact discrete adjoint — measured 9.8e-4
  // at n = 32 and 2.4e-4 at n = 64 (the clean factor 4).
  auto gate = [&](Sweep& s, auto member, real_t tol_km, real_t tol_eta,
                  const char* what) {
    const IsotropicKernel* K[] = {s.kappa.get(), s.mu.get(), s.eta.get()};
    for (int p = 0; p < 3; p++) {
      const real_t tol = p == 2 ? tol_eta : tol_km;
      EXPECT_NEAR(K[p]->Directional(0) / (fd[p][0].*member), 1.0, tol)
          << what << ", parameter " << p << ", blob";
      EXPECT_NEAR(K[p]->Directional(1) / (fd[p][1].*member), 1.0, tol)
          << what << ", parameter " << p << ", all";
    }
  };
  gate(sweep_end, &Measured::J_end, 2e-5, 2e-5, "final-time observation");
  gate(sweep_mid, &Measured::J_mid, 2e-5, 2e-5, "interior observation");
  gate(sweep_cont, &Measured::J_cont, 2e-5, 5e-3, "continuous observation");
}

// ============================================================================
// Rung 3: self-gravity on the mixed class (WP8.3).
//
// The configuration is the canned two-layer disc of the mixed-problem
// tests (MixedProblemTestCommon.hpp: body attribute 1 of radius 1 inside
// a buffer shell; kG = 0.05 makes the coupling O(1)). The load is the
// degree-2 surface mass pattern, held constant in time; the observations
// are windowed STATION duals at the angles 0.3 and 1.4 — chosen off every
// mirror image of the load's symmetries (x -> -x and y -> -y), the
// symmetry trap of the file header — on the displacement (radial
// component, body surface) and on the POTENTIAL (parent mesh, boundary
// attribute 1), the latter through the new PotentialForceSlot.

namespace {

// Windowed two-station angular weight at the angles 0.3 and 1.4.
real_t StationWindow(const Vector& x) {
  constexpr real_t s = 0.1;
  const real_t phi = std::atan2(x[1], x[0]);
  auto window = [phi](real_t centre) {
    real_t d = phi - centre;
    while (d > M_PI) d -= 2.0 * M_PI;
    while (d < -M_PI) d += 2.0 * M_PI;
    return std::exp(-d * d / (s * s)) / (s * std::sqrt(M_PI));
  };
  return window(1.4) - window(0.3);
}

// Radial-displacement station dual on the body surface.
void RadialStations(const Vector& x, Vector& v) {
  const real_t r = x.Norml2();
  const real_t w = StationWindow(x);
  v.SetSize(2);
  v[0] = w * x[0] / r;
  v[1] = w * x[1] / r;
}

// The surface mass load, constant in time (the fixture's pattern at
// t = 0).
real_t MassLoad(const Vector& x) { return self_grav_test::SurfaceLoad(x, 0.0); }

struct GravCase {
  std::unique_ptr<Mesh> parent;
  std::unique_ptr<SubMesh> body;
  std::unique_ptr<H1_FECollection> fec;
  std::unique_ptr<FiniteElementSpace> fes_u, fes_phi, sfes;
  std::unique_ptr<ConstantCoefficient> rho;
  std::unique_ptr<FunctionCoefficient> sigma;
  Array<int> surface, parent_surface;
  std::unique_ptr<LinearForm> g_u;   // displacement station dual (body)
  std::unique_ptr<LinearForm> g_phi; // potential station dual (parent)
  std::unique_ptr<VectorFunctionCoefficient> h_u;
  std::unique_ptr<FunctionCoefficient> h_phi;

  explicit GravCase(int order) {
    parent = std::make_unique<Mesh>(self_grav_test::MeshFile(2).c_str(), 1, 1);
    body = std::make_unique<SubMesh>(SubMesh::CreateFromDomain(
        *parent, self_grav_test::BodyMarker(*parent)));
    fec = std::make_unique<H1_FECollection>(order, 2);
    fes_u = std::make_unique<FiniteElementSpace>(body.get(), fec.get(), 2);
    fes_phi = std::make_unique<FiniteElementSpace>(parent.get(), fec.get());
    sfes = std::make_unique<FiniteElementSpace>(body.get(), fec.get());
    rho = std::make_unique<ConstantCoefficient>(self_grav_test::kRho);
    sigma = std::make_unique<FunctionCoefficient>(MassLoad);
    surface = self_grav_test::SurfaceMarker(*body);
    parent_surface.SetSize(parent->bdr_attributes.Max());
    parent_surface = 0;
    parent_surface[0] = 1;  // the body surface on the parent mesh

    h_u = std::make_unique<VectorFunctionCoefficient>(2, RadialStations);
    g_u = std::make_unique<LinearForm>(fes_u.get());
    g_u->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(*h_u), surface);
    g_u->Assemble();
    h_phi = std::make_unique<FunctionCoefficient>(
        [](const Vector& x) { return StationWindow(x); });
    g_phi = std::make_unique<LinearForm>(fes_phi.get());
    g_phi->AddBoundaryIntegrator(new BoundaryLFIntegrator(*h_phi),
                                 parent_surface);
    g_phi->Assemble();
  }

  std::unique_ptr<LinearQuasiStaticMixedSelfGravitatingProblem> Problem(
      const Rheology& rheology, bool with_load) {
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        fes_u.get(), fes_phi.get(), rheology, *rho, self_grav_test::kG,
        self_grav_test::kDtNDegree);
    if (with_load) {
      p->SetSurfaceLoad(*sigma, surface);
    }
    p->SetRelTol(1e-11);
    return p;
  }
};

}  // namespace

// GA1 on the static gravitating solve: the solve is its own adjoint.
// (a) displacement-displacement reciprocity g.u(f) = f.u(g);
// (b) displacement-potential reciprocity h_phi . phi(f) = f . u(h_phi),
//     which pins the sign and scale of the PotentialForceSlot.
TEST(Adjoint, GravitatingReciprocity) {
  GravCase s(2);
  auto kappa = ConstantCoefficient(self_grav_test::kKappa);
  auto mu = ConstantCoefficient(self_grav_test::kMu);
  auto rheology = IsotropicElasticRheology(2, kappa, mu);
  // All solves on the load-free problem; the duals supply the loads.
  auto twin = s.Problem(rheology, false);

  // A second displacement station dual for the u-u identity.
  auto h2 = VectorFunctionCoefficient(2, [](const Vector& x, Vector& v) {
    const real_t r = x.Norml2();
    const real_t phi = std::atan2(x[1], x[0]);
    const real_t w = std::exp(-(phi - 2.2) * (phi - 2.2) / 0.01);
    v.SetSize(2);
    v[0] = w * x[0] / r;
    v[1] = w * x[1] / r;
  });
  auto g2 = LinearForm(s.fes_u.get());
  g2.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(h2), s.surface);
  g2.Assemble();

  // (a) displacement-displacement: g2 . u(g_u) = g_u . u(g2).
  twin->AssembleForce(0.0);
  twin->AddForce(*s.g_u);
  ASSERT_TRUE(twin->Solve());
  const real_t a1 = g2 * twin->Displacement();
  twin->AssembleForce(0.0);
  twin->AddForce(g2);
  ASSERT_TRUE(twin->Solve());
  const real_t a2 = *s.g_u * twin->Displacement();
  EXPECT_NEAR(a1 / a2, 1.0, 2e-9);

  // (b) displacement-potential: g_u . u(h_phi) = h_phi . phi(g_u) — the
  // sign and scale certificate of the PotentialForceSlot.
  twin->AssembleForce(0.0);
  twin->AddPotentialForce(*s.g_phi);
  ASSERT_TRUE(twin->Solve());
  const real_t b1 = *s.g_u * twin->Displacement();
  twin->AssembleForce(0.0);
  twin->AddForce(*s.g_u);
  ASSERT_TRUE(twin->Solve());
  const real_t b2 = *s.g_phi * twin->Potential();
  EXPECT_NEAR(b1 / b2, 1.0, 2e-9);
}

// The rung-3 FD gate: kappa/mu/eta kernels on the gravitating Maxwell
// body, for a final-time DISPLACEMENT observation and a final-time
// POTENTIAL observation (the new slot end to end).
TEST(Adjoint, GravitatingViscoelasticKernels) {
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
    for (int n = 1; n <= n_steps; n++) {
      ode.Step(m, t, dt_step);
    }
    EXPECT_TRUE(visco.SolveElastic(m, t));
    Measured out;
    out.J_u = *s.g_u * prob->Displacement();
    out.J_phi = *s.g_phi * prob->Potential();
    return out;
  };

  // The base run on persistent operators, recording the trajectory.
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

  auto prob_adj = s.Problem(rheology0, false);
  auto visco_adj = ViscoelasticOperator(*prob_adj);

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
        IsotropicKernel::Parameter::Kappa, *s.sfes);
    sw.mu = std::make_unique<IsotropicKernel>(IsotropicKernel::Parameter::Mu,
                                              *s.sfes);
    sw.eta = std::make_unique<IsotropicKernel>(IsotropicKernel::Parameter::Eta,
                                               *s.sfes, &tau0);
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

  AdjointForce force_u;
  force_u.AddObservation(trajectory.Time(n_steps), *s.g_u);
  AdjointForce force_phi;
  force_phi.AddPotentialObservation(trajectory.Time(n_steps), *s.g_phi);
  auto sweep_u = make_sweep(force_u);
  auto sweep_phi = make_sweep(force_phi);

  // FD: one rerun per sign and direction, both functionals sampled.
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
      fd[p][c].J_u = (plus.J_u - minus.J_u) / (2.0 * eps);
      fd[p][c].J_phi = (plus.J_phi - minus.J_phi) / (2.0 * eps);
    }
  }

  // Thresholds encode measurements: every row is at or below 1.6e-4
  // except the mu blob rows at 1.3e-3 — the O(h) strain-map floor of
  // the disc's curved elements, exactly the examples' spatial floor
  // (kappa blob 1.1e-5, kappa all 5e-7 / 1.1e-4, eta 3e-5..1.5e-4;
  // both observation types the same grades).
  auto gate = [&](Sweep& sw, auto member, real_t tol, const char* what) {
    const IsotropicKernel* K[] = {sw.kappa.get(), sw.mu.get(), sw.eta.get()};
    for (int p = 0; p < 3; p++) {
      for (int c = 0; c < 2; c++) {
        EXPECT_NEAR(K[p]->Directional(c) / (fd[p][c].*member), 1.0, tol)
            << what << ", parameter " << p << (c == 0 ? ", blob" : ", all");
      }
    }
  };
  gate(sweep_u, &Measured::J_u, 5e-3, "displacement observation");
  gate(sweep_phi, &Measured::J_phi, 5e-3, "potential observation");
}

// Gate GA8 (the rheology axis, R1): per-branch kernels on a two-branch
// generalised Maxwell body with a long-term shear modulus — the
// relaxation-spectrum kernels. The parameterisation is
// (kappa, mu_inf, {mu_k, eta_k}) with tau_k = eta_k / mu_k; each
// parameter's kernel is FD-gated for a final-time observation on the
// unit-square configuration of the tests above.
TEST(Adjoint, PronyKernels) {
  const int dim = 2, n_el = 12, order = 2, n_steps = 64;
  const real_t t_final = 2.0;
  auto mesh = Mesh::MakeCartesian2D(n_el, n_el, Element::QUADRILATERAL);
  auto fec = H1_FECollection(order, dim);
  auto fes = FiniteElementSpace(&mesh, &fec, dim);
  auto sfes = FiniteElementSpace(&mesh, &fec);
  auto top = Array<int>(mesh.bdr_attributes.Max());
  auto bottom = Array<int>(mesh.bdr_attributes.Max());
  top = 0;
  bottom = 0;
  top[2] = 1;
  bottom[0] = 1;
  auto traction = VectorFunctionCoefficient(dim, Traction2D);
  auto station = VectorFunctionCoefficient(dim, Station2D);
  auto g = LinearForm(&fes);
  g.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(station), bottom);
  g.Assemble();

  auto chi_blob = MakeBlob(dim);
  auto chi_all = ConstantCoefficient(1.0);
  // Base model: kappa 5.4, mu_inf 1.0, branches (mu, eta) = (1.0, 1.0)
  // and (0.7, 0.21): tau_1 = 1, tau_2 = 0.3 — distinct spectra.
  auto kappa0 = ConstantCoefficient(5.4);
  auto mu_inf0 = ConstantCoefficient(1.0);
  auto mu10 = ConstantCoefficient(1.0);
  auto eta10 = ConstantCoefficient(1.0);
  auto mu20 = ConstantCoefficient(0.7);
  auto eta20 = ConstantCoefficient(0.21);
  const real_t dt = t_final / n_steps;

  // One forward run for a parameter set; returns J = g . u(t1).
  auto run_forward = [&](Coefficient& kappa, Coefficient& mu_inf,
                         Coefficient& mu1, Coefficient& eta1,
                         Coefficient& mu2, Coefficient& eta2,
                         ForwardTrajectory* keep,
                         ViscoelasticOperator** keep_op) -> real_t {
    auto tau1 = RatioCoefficient(eta1, mu1);
    auto tau2 = RatioCoefficient(eta2, mu2);
    std::vector<MaxwellBranch> branches = {{&mu1, &tau1, nullptr},
                                           {&mu2, &tau2, nullptr}};
    auto rheology =
        IsotropicMaxwellRheology(dim, kappa, mu_inf, branches);
    auto prob =
        LinearQuasiStaticTractionProblem(&fes, rheology, traction, top);
    auto visco = ViscoelasticOperator(prob);
    ExponentialTrapezoidSolver ode;
    ode.Init(visco);
    Vector m(visco.Height());
    m = 0.0;
    real_t t = 0.0, dt_step = dt;
    EXPECT_TRUE(visco.SolveElastic(m, t));
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
    (void)keep_op;
    return g * prob.Displacement();
  };

  // The base run on PERSISTENT operators (the trajectory must outlive
  // the sweep), plus the zero-load adjoint twin.
  auto tau10 = RatioCoefficient(eta10, mu10);
  auto tau20 = RatioCoefficient(eta20, mu20);
  std::vector<MaxwellBranch> branches0 = {{&mu10, &tau10, nullptr},
                                          {&mu20, &tau20, nullptr}};
  auto rheology0 = IsotropicMaxwellRheology(dim, kappa0, mu_inf0, branches0);
  auto prob_fwd =
      LinearQuasiStaticTractionProblem(&fes, rheology0, traction, top);
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
  auto zero_vec = Vector(dim);
  zero_vec = 0.0;
  auto zero_traction = VectorConstantCoefficient(zero_vec);
  auto prob_adj = LinearQuasiStaticTractionProblem(&fes, rheology0,
                                                   zero_traction, top);
  auto visco_adj = ViscoelasticOperator(prob_adj);

  // The six kernels: Kappa, MuLongTerm, and per-branch Mu/Eta.
  using P = IsotropicKernel::Parameter;
  auto K_kappa = IsotropicKernel(P::Kappa, sfes);
  auto K_muinf = IsotropicKernel(P::MuLongTerm, sfes);
  auto K_mu1 = IsotropicKernel(P::Mu, sfes, nullptr, 0);
  auto K_mu2 = IsotropicKernel(P::Mu, sfes, nullptr, 1);
  auto K_eta1 = IsotropicKernel(P::Eta, sfes, &tau10, 0);
  auto K_eta2 = IsotropicKernel(P::Eta, sfes, &tau20, 1);
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
  force.AddObservation(trajectory.Time(n_steps), g);
  auto solver = AdjointSolver(visco_adj, trajectory);
  solver.SetForce(force);
  for (auto* k : K) {
    solver.AddKernel(*k);
  }
  ASSERT_TRUE(solver.Run());

  // FD per parameter and direction.
  const real_t eps = 1e-3;
  const char* names[6] = {"kappa", "mu_inf", "mu_1", "eta_1", "mu_2",
                          "eta_2"};
  for (int p = 0; p < 6; p++) {
    for (int c = 0; c < 2; c++) {
      real_t fd = 0.0;
      for (const real_t sg : {eps, -eps}) {
        auto bk = Bump(kappa0, *chi_of[c], p == 0 ? sg : 0.0);
        auto bmi = Bump(mu_inf0, *chi_of[c], p == 1 ? sg : 0.0);
        auto bm1 = Bump(mu10, *chi_of[c], p == 2 ? sg : 0.0);
        auto be1 = Bump(eta10, *chi_of[c], p == 3 ? sg : 0.0);
        auto bm2 = Bump(mu20, *chi_of[c], p == 4 ? sg : 0.0);
        auto be2 = Bump(eta20, *chi_of[c], p == 5 ? sg : 0.0);
        const real_t J =
            run_forward(*bk.value, *bmi.value, *bm1.value, *be1.value,
                        *bm2.value, *be2.value, nullptr, nullptr);
        fd += (sg > 0 ? J : -J);
      }
      fd /= 2.0 * eps;
      // Measured at n = 64 (uniform parameters, affine quads): kappa
      // 1.5e-6, mu_inf 1.3e-4, mu_1 6e-5, eta_1 1.5e-4, eta_2 7e-4,
      // mu_2 5.5e-3 — every row is the O(dt^2) time floor, scaling
      // with (dt/tau_k)^2 (the n = 32 values were exactly 4x larger),
      // so the fast branch (tau_2 = 0.3) dominates.
      const real_t tol = p == 4 ? 2e-2 : 3e-3;
      EXPECT_NEAR(K[p]->Directional(c) / fd, 1.0, tol)
          << names[p] << (c == 0 ? ", blob" : ", all");
    }
  }
}
