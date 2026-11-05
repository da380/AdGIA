/*
  Tests for the discrete second-order adjoint (discrete_second_order.hpp):
  the Hessian-vector product of the discrete objective. Gates on the
  adjoint tests' unit-square Maxwell configuration: SYMMETRY <H v, w> =
  <H w, v> to rounding (the discrete transpose), and agreement with
  central finite differences of the discrete gradient (DiscreteAdjoint
  Solver::Directional at x +- eps v) at FD accuracy, for v, w among kappa,
  mu (at fixed eta) and eta blobs, final-time and continuous observations;
  and the bordered solves: the frozen-shoreline water load with rotation
  on the coarse 3-D ball (TestDiscreteAdjoint's WaterLoadFrozen3D), for a
  displacement and a strict four-slot sea-level observation, along eta
  and mu blobs.
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
constexpr real_t kKappa0 = 5.4, kMu0 = 2.7, kEta0 = 2.7, kP0 = 0.05;
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

TEST(DiscreteSecondOrder, SymmetricAndMatchesFD2D) {
  const int n_el = 12, order = 2, n_steps = 16;
  const real_t t_final = 2.0;
  Mesh mesh = Mesh::MakeCartesian2D(n_el, n_el, Element::QUADRILATERAL);
  H1_FECollection fec(order, 2);
  FiniteElementSpace fes(&mesh, &fec, 2);
  Array<int> top(mesh.bdr_attributes.Max()), bottom(mesh.bdr_attributes.Max());
  top = 0;
  bottom = 0;
  top[2] = 1;
  bottom[0] = 1;
  VectorFunctionCoefficient traction(2, Traction2D), station(2, Station2D);
  LinearForm g(&fes);
  g.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(station), bottom);
  g.Assemble();
  Vector zero_vec(2);
  zero_vec = 0.0;
  VectorConstantCoefficient zero_traction(zero_vec);
  ConstantCoefficient kappa0(kKappa0), mu0(kMu0), eta0(kEta0);
  FunctionCoefficient chi_a([](const Vector& x) {
    const real_t r2 = (x[0] - 0.4) * (x[0] - 0.4) + (x[1] - 0.6) * (x[1] - 0.6);
    return std::exp(-r2 / 0.03);
  });
  FunctionCoefficient chi_b([](const Vector& x) {
    const real_t r2 = (x[0] - 0.6) * (x[0] - 0.3) + (x[1] - 0.3) * (x[1] - 0.3);
    return std::exp(-r2 / 0.03);
  });
  const real_t dt = t_final / n_steps;

  // One forward run + discrete first-order adjoint at a parameter set:
  // the gradient along the three delta directions.
  struct Grad {
    real_t k = 0.0, m = 0.0, e = 0.0;
  };
  auto trajectory_of = [&](const Rheology& rheology,
                           LinearQuasiStaticTractionProblem& prob,
                           ViscoelasticOperator& visco) {
    auto traj = std::make_unique<ForwardTrajectory>(visco, true);
    ExponentialTrapezoidSolver ode;
    ode.Init(visco);
    Vector m(visco.Height());
    m = 0.0;
    real_t t = 0.0, dts = dt;
    traj->Record(m, t);
    for (int n = 1; n <= n_steps; n++) {
      ode.Step(m, t, dts);
      traj->Record(m, t);
    }
    return traj;
  };
  // Directions: v, w in {kappa blob a, mu blob b, eta blob a}; deltas the
  // same set.
  ProductCoefficient d_kappa(kappa0, chi_a), d_mu(mu0, chi_b), d_eta(eta0, chi_a);
  auto make_dir = [&](int p) {
    TangentDirection d;
    d.mu.assign(1, nullptr);
    d.eta.assign(1, nullptr);
    if (p == 0) d.kappa = &d_kappa;
    if (p == 1) d.mu[0] = &d_mu;
    if (p == 2) d.eta[0] = &d_eta;
    return d;
  };
  Coefficient* chi_of[] = {&chi_a, &chi_b, &chi_a};
  const char* names[] = {"kappa", "mu", "eta"};

  for (int which = 0; which < 2; which++) {
    const bool continuous = which == 1;
    auto make_force = [&](const ForwardTrajectory& traj) {
      AdjointForce f;
      if (continuous) {
        f.AddContinuous([&](real_t t, Vector& v) { v.Add(t / t_final, g); });
      } else {
        f.AddObservation(traj.Time(n_steps), g);
      }
      return f;
    };
    auto gradient = [&](Coefficient& kappa, Coefficient& mu, Coefficient& eta) {
      RatioCoefficient tau(eta, mu);
      auto rheology = IsotropicMaxwellRheology::Maxwell(2, kappa, mu, tau);
      LinearQuasiStaticTractionProblem prob(&fes, rheology, traction, top);
      ViscoelasticOperator visco(prob);
      auto traj = trajectory_of(rheology, prob, visco);
      AdjointForce force = make_force(*traj);
      LinearQuasiStaticTractionProblem twin(&fes, rheology, zero_traction, top);
      LinearQuasiStaticTractionProblem stwin(&fes, rheology, zero_traction, top);
      DiscreteAdjointSolver discrete(twin, stwin, *traj);
      discrete.SetForce(force);
      EXPECT_TRUE(discrete.Run());
      return Grad{discrete.Directional(make_dir(0)), discrete.Directional(make_dir(1)),
                  discrete.Directional(make_dir(2))};
    };

    // The base: forward, tangents along the three directions, the discrete
    // first-order adjoint, the second order along each v.
    RatioCoefficient tau0(eta0, mu0);
    auto rheology0 = IsotropicMaxwellRheology::Maxwell(2, kappa0, mu0, tau0);
    LinearQuasiStaticTractionProblem prob_fwd(&fes, rheology0, traction, top);
    ViscoelasticOperator visco_fwd(prob_fwd);
    auto trajectory = trajectory_of(rheology0, prob_fwd, visco_fwd);
    AdjointForce force = make_force(*trajectory);
    LinearQuasiStaticTractionProblem twin(&fes, rheology0, zero_traction, top);
    LinearQuasiStaticTractionProblem stwin(&fes, rheology0, zero_traction, top);
    DiscreteAdjointSolver discrete(twin, stwin, *trajectory);
    discrete.SetForce(force);
    ASSERT_TRUE(discrete.Run());
    std::vector<std::unique_ptr<TangentLinearSolver>> tangents;
    LinearQuasiStaticTractionProblem tan_twin(&fes, rheology0, zero_traction, top);
    for (int p = 0; p < 3; p++) {
      tangents.push_back(std::make_unique<TangentLinearSolver>(tan_twin, *trajectory));
      tangents.back()->SetDirection(make_dir(p));
      ASSERT_TRUE(tangents.back()->Run());
    }
    real_t H[3][3];
    for (int p = 0; p < 3; p++) {
      LinearQuasiStaticTractionProblem dtwin(&fes, rheology0, zero_traction, top);
      LinearQuasiStaticTractionProblem svtwin(&fes, rheology0, zero_traction, top);
      DiscreteSecondOrderSolver second(discrete, dtwin, svtwin, *trajectory,
                                       tangents[p]->Trajectory(), make_dir(p));
      ASSERT_TRUE(second.Run());
      for (int q = 0; q < 3; q++) {
        LinearQuasiStaticTractionProblem sdtwin(&fes, rheology0, zero_traction, top);
        H[p][q] = second.Directional(make_dir(q), sdtwin);
      }
    }
    // FD of the discrete gradient along each v.
    const real_t eps = 1e-3;
    for (int p = 0; p < 3; p++) {
      Grad plus, minus;
      for (const real_t sg : {eps, -eps}) {
        auto kp = Bump(kappa0, *chi_of[p], p == 0 ? sg : 0.0);
        auto mp = Bump(mu0, *chi_of[p], p == 1 ? sg : 0.0);
        auto ep = Bump(eta0, *chi_of[p], p == 2 ? sg : 0.0);
        (sg > 0 ? plus : minus) = gradient(*kp.value, *mp.value, *ep.value);
      }
      const real_t fd[3] = {(plus.k - minus.k) / (2 * eps), (plus.m - minus.m) / (2 * eps),
                            (plus.e - minus.e) / (2 * eps)};
      real_t scale = 0.0;
      for (int q = 0; q < 3; q++) scale = std::max(scale, std::abs(fd[q]));
      for (int q = 0; q < 3; q++) {
        if (std::getenv("ADGIA_TEST_VERBOSE")) {
          std::cout << (continuous ? "continuous" : "final-time") << " H[" << names[p]
                    << "," << names[q] << "] " << H[p][q] << " fd " << fd[q]
                    << " | H[" << names[q] << "," << names[p] << "] " << H[q][p] << "\n";
        }
        EXPECT_LT(std::abs(H[p][q] - fd[q]) / scale, 1e-5)
            << names[p] << " " << names[q];
        EXPECT_LT(std::abs(H[p][q] - H[q][p]) / scale, 1e-9)
            << "symmetry " << names[p] << " " << names[q];
      }
    }
  }
}

// Static: the discrete Hessian of the elastic unit square along kappa
// and mu blob directions, symmetric to rounding and matching FD of the
// static discrete gradient.
TEST(DiscreteSecondOrder, Static2D) {
  const int n_el = 12, order = 2;
  Mesh mesh = Mesh::MakeCartesian2D(n_el, n_el, Element::QUADRILATERAL);
  H1_FECollection fec(order, 2);
  FiniteElementSpace fes(&mesh, &fec, 2);
  Array<int> top(mesh.bdr_attributes.Max()), bottom(mesh.bdr_attributes.Max());
  top = 0;
  bottom = 0;
  top[2] = 1;
  bottom[0] = 1;
  VectorFunctionCoefficient traction(2, Traction2D), station(2, Station2D);
  LinearForm g(&fes);
  g.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(station), bottom);
  g.Assemble();
  Vector zero_vec(2);
  zero_vec = 0.0;
  VectorConstantCoefficient zero_traction(zero_vec);
  ConstantCoefficient kappa0(kKappa0), mu0(kMu0);
  FunctionCoefficient chi_a([](const Vector& x) {
    const real_t r2 = (x[0] - 0.4) * (x[0] - 0.4) + (x[1] - 0.6) * (x[1] - 0.6);
    return std::exp(-r2 / 0.03);
  });
  FunctionCoefficient chi_b([](const Vector& x) {
    const real_t r2 = (x[0] - 0.6) * (x[0] - 0.6) + (x[1] - 0.3) * (x[1] - 0.3);
    return std::exp(-r2 / 0.03);
  });
  ProductCoefficient d_kappa(kappa0, chi_a), d_mu(mu0, chi_b);
  auto make_dir = [&](int p) {
    TangentDirection d;
    if (p == 0) d.kappa = &d_kappa; else d.mu_longterm = &d_mu;
    return d;
  };
  Coefficient* chi_of[] = {&chi_a, &chi_b};
  AdjointForce force;
  force.AddObservation(0.0, g);
  struct Grad { real_t k = 0.0, m = 0.0; };
  auto gradient = [&](Coefficient& kappa, Coefficient& mu, GridFunction* keep) {
    IsotropicElasticRheology rheology(2, kappa, mu);
    LinearQuasiStaticTractionProblem prob(&fes, rheology, traction, top);
    prob.AssembleForce(0.0);
    EXPECT_TRUE(prob.Solve());
    if (keep) *keep = prob.Displacement();
    LinearQuasiStaticTractionProblem twin(&fes, rheology, zero_traction, top);
    DiscreteAdjointSolver discrete(twin, prob.Displacement());
    discrete.SetForce(force);
    EXPECT_TRUE(discrete.Run());
    return Grad{discrete.Directional(make_dir(0)), discrete.Directional(make_dir(1))};
  };
  GridFunction u(&fes);
  gradient(kappa0, mu0, &u);
  IsotropicElasticRheology rheology0(2, kappa0, mu0);
  LinearQuasiStaticTractionProblem twin(&fes, rheology0, zero_traction, top);
  DiscreteAdjointSolver discrete(twin, u);
  discrete.SetForce(force);
  ASSERT_TRUE(discrete.Run());
  real_t H[2][2];
  for (int p = 0; p < 2; p++) {
    LinearQuasiStaticTractionProblem tan_twin(&fes, rheology0, zero_traction, top);
    TangentLinearSolver tangent(tan_twin, u);
    tangent.SetDirection(make_dir(p));
    ASSERT_TRUE(tangent.Run());
    LinearQuasiStaticTractionProblem dtwin(&fes, rheology0, zero_traction, top);
    DiscreteSecondOrderSolver second(discrete, dtwin, u, tangent.Displacement(),
                                     make_dir(p));
    ASSERT_TRUE(second.Run());
    for (int q = 0; q < 2; q++) H[p][q] = second.Directional(make_dir(q), dtwin);
  }
  const real_t eps = 1e-3;
  for (int p = 0; p < 2; p++) {
    Grad plus, minus;
    for (const real_t sg : {eps, -eps}) {
      auto kp = Bump(kappa0, *chi_of[p], p == 0 ? sg : 0.0);
      auto mp = Bump(mu0, *chi_of[p], p == 1 ? sg : 0.0);
      (sg > 0 ? plus : minus) = gradient(*kp.value, *mp.value, nullptr);
    }
    const real_t fd[2] = {(plus.k - minus.k) / (2 * eps), (plus.m - minus.m) / (2 * eps)};
    const real_t scale = std::max(std::abs(fd[0]), std::abs(fd[1]));
    for (int q = 0; q < 2; q++) {
      if (std::getenv("ADGIA_TEST_VERBOSE")) {
        std::cout << "static H[" << p << "," << q << "] " << H[p][q] << " fd " << fd[q]
                  << " | H[" << q << "," << p << "] " << H[q][p] << "\n";
      }
      EXPECT_LT(std::abs(H[p][q] - fd[q]) / scale, 1e-5);
      EXPECT_LT(std::abs(H[p][q] - H[q][p]) / scale, 1e-9);
    }
  }
}

// The bordered solves: water load (frozen shoreline) and rotation on the
// coarse 3-D ball, the Maxwell body of TestDiscreteAdjoint's
// WaterLoadFrozen3D, observed by a radial displacement cap and by the
// strict sea-level functional through all four slots (u, phi, uniform,
// omega) at the final time. H[v, delta] for v, delta among an eta blob
// and a mu blob (at fixed eta) against central finite differences of the
// discrete gradient, and the symmetry of the block.
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

TEST(DiscreteSecondOrder, WaterLoadFrozen3D) {
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
  ConstantCoefficient kappa0(self_grav_test::kKappa);
  ConstantCoefficient mu0(self_grav_test::kMu);
  ConstantCoefficient eta0(self_grav_test::kMu);

  FunctionCoefficient sigma_data([](const Vector& x) {
    const real_t r = x.Norml2(), c = x[2] / r;
    return 0.02 * (3.0 * c * c - 0.5 + 0.5 * x[0] * x[2] / (r * r));
  });
  ConstantCoefficient zero_data(0.0);

  const Vector n1 = UnitDir(0.3, 0.5, 0.81), n3 = UnitDir(0.1, -0.6, 0.79);
  VectorFunctionCoefficient h_u(3, [n1](const Vector& x, Vector& v) {
    v = x;
    v *= MeltCap(x, n1) / x.Norml2();
  });
  LinearForm g_u(&fes_u);
  g_u.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(h_u), surface);
  g_u.Assemble();
  FunctionCoefficient h_sl([n3](const Vector& x) { return MeltCap(x, n3); });
  VectorFunctionCoefficient h_sl_u(3, [n3](const Vector& x, Vector& v) {
    v = x;
    v *= -MeltCap(x, n3);
  });
  LinearForm g_sl_u(&fes_u);
  g_sl_u.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(h_sl_u), surface);
  g_sl_u.Assemble();
  ProductCoefficient h_sl_phi(-1.0 / kG0, h_sl);
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

  auto make_problem = [&](const Rheology& rheology, Coefficient& data) {
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        &fes_u, &fes_phi, rheology, rho, self_grav_test::kG,
        self_grav_test::kDtNDegree);
    p->SetWaterLoad(w, data, surface);
    p->SetRotation(kOmega, moments);
    p->SetRelTol(1e-12);
    return p;
  };
  auto trajectory_of = [&](ViscoelasticOperator& visco) {
    auto traj = std::make_unique<ForwardTrajectory>(visco, true);
    ExponentialTrapezoidSolver ode;
    ode.Init(visco);
    Vector m(visco.Height());
    m = 0.0;
    real_t t = 0.0, dts = dt;
    traj->Record(m, t);
    for (int n = 1; n <= n_steps; n++) {
      ode.Step(m, t, dts);
      traj->Record(m, t);
    }
    return traj;
  };
  auto make_forces = [&](const ForwardTrajectory& traj,
                         AdjointForce& force_u, AdjointForce& force_sl) {
    force_u.AddObservation(traj.Time(n_steps), g_u);
    AdjointLoad sl;
    sl.u = &g_sl_u;
    sl.phi = &g_sl_phi;
    sl.uniform = sl_uniform;
    sl.omega = &sl_omega;
    force_sl.AddObservation(traj.Time(n_steps), sl);
  };
  const char* fnames[] = {"displacement", "sea level"};

  FunctionCoefficient chi_blob3([](const Vector& x) {
    const real_t dx = x[0] - 0.3, dy = x[1] - 0.35, dz = x[2] - 0.4;
    return std::exp(-(dx * dx + dy * dy + dz * dz) / 0.03);
  });
  ProductCoefficient d_eta(eta0, chi_blob3), d_mu(mu0, chi_blob3);
  auto make_dir = [&](int p) {
    TangentDirection d;
    d.mu.assign(1, nullptr);
    d.eta.assign(1, nullptr);
    if (p == 0) d.eta[0] = &d_eta;
    if (p == 1) d.mu[0] = &d_mu;
    return d;
  };
  const char* names[] = {"eta", "mu"};

  // The discrete gradient of both functionals along both directions at a
  // parameter set (mu perturbed at fixed eta).
  auto gradient = [&](Coefficient& mu, Coefficient& eta, real_t g[2][2]) {
    RatioCoefficient tau(eta, mu);
    auto rheology = IsotropicMaxwellRheology::Maxwell(3, kappa0, mu, tau);
    auto prob = make_problem(rheology, sigma_data);
    ViscoelasticOperator visco(*prob);
    auto traj = trajectory_of(visco);
    AdjointForce force_u, force_sl;
    make_forces(*traj, force_u, force_sl);
    const AdjointForce* forces[] = {&force_u, &force_sl};
    auto twin = make_problem(rheology, zero_data);
    auto stwin = make_problem(rheology, zero_data);
    DiscreteAdjointSolver discrete(*twin, *stwin, *traj);
    for (int f = 0; f < 2; f++) {
      discrete.SetForce(*forces[f]);
      ASSERT_TRUE(discrete.Run());
      for (int q = 0; q < 2; q++) g[f][q] = discrete.Directional(make_dir(q));
    }
  };

  // The base: forward, tangents, first order per functional, second order
  // per (functional, v).
  RatioCoefficient tau0(eta0, mu0);
  auto rheology0 = IsotropicMaxwellRheology::Maxwell(3, kappa0, mu0, tau0);
  auto prob_fwd = make_problem(rheology0, sigma_data);
  ViscoelasticOperator visco_fwd(*prob_fwd);
  auto trajectory = trajectory_of(visco_fwd);
  AdjointForce force_u, force_sl;
  make_forces(*trajectory, force_u, force_sl);
  const AdjointForce* forces[] = {&force_u, &force_sl};
  std::vector<std::unique_ptr<TangentLinearSolver>> tangents;
  auto tan_twin = make_problem(rheology0, zero_data);
  for (int p = 0; p < 2; p++) {
    tangents.push_back(std::make_unique<TangentLinearSolver>(*tan_twin, *trajectory));
    tangents.back()->SetDirection(make_dir(p));
    ASSERT_TRUE(tangents.back()->Run());
  }
  real_t H[2][2][2];
  auto twin = make_problem(rheology0, zero_data);
  auto stwin = make_problem(rheology0, zero_data);
  auto dtwin = make_problem(rheology0, zero_data);
  auto svtwin = make_problem(rheology0, zero_data);
  auto sdtwin = make_problem(rheology0, zero_data);
  for (int f = 0; f < 2; f++) {
    DiscreteAdjointSolver discrete(*twin, *stwin, *trajectory);
    discrete.SetForce(*forces[f]);
    ASSERT_TRUE(discrete.Run());
    for (int p = 0; p < 2; p++) {
      DiscreteSecondOrderSolver second(discrete, *dtwin, *svtwin, *trajectory,
                                       tangents[p]->Trajectory(), make_dir(p));
      ASSERT_TRUE(second.Run());
      for (int q = 0; q < 2; q++) {
        H[f][p][q] = second.Directional(make_dir(q), *sdtwin);
      }
    }
  }

  // FD of the discrete gradient along each v: 1e-3 relative bumps.
  const real_t eps = 1e-3;
  for (int p = 0; p < 2; p++) {
    real_t plus[2][2], minus[2][2];
    for (const real_t sg : {eps, -eps}) {
      auto mp = Bump(mu0, chi_blob3, p == 1 ? sg : 0.0);
      auto ep = Bump(eta0, chi_blob3, p == 0 ? sg : 0.0);
      gradient(*mp.value, *ep.value, sg > 0 ? plus : minus);
    }
    for (int f = 0; f < 2; f++) {
      real_t fd[2], scale = 0.0;
      for (int q = 0; q < 2; q++) {
        fd[q] = (plus[f][q] - minus[f][q]) / (2 * eps);
        scale = std::max(scale, std::abs(fd[q]));
      }
      for (int q = 0; q < 2; q++) {
        if (std::getenv("ADGIA_TEST_VERBOSE")) {
          std::cout << "water, " << fnames[f] << ": H[" << names[p] << ","
                    << names[q] << "] " << H[f][p][q] << " fd " << fd[q]
                    << " | H[" << names[q] << "," << names[p] << "] "
                    << H[f][q][p] << "\n";
        }
        EXPECT_LT(std::abs(H[f][p][q] - fd[q]) / scale, 1e-5)
            << fnames[f] << " " << names[p] << " " << names[q];
        EXPECT_LT(std::abs(H[f][p][q] - H[f][q][p]) / scale, 1e-7)
            << "symmetry " << fnames[f] << " " << names[p] << " " << names[q];
      }
    }
  }
}
