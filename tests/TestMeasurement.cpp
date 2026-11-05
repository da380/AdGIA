/*
  Tests for the measurement layer (measurement.hpp): observables that
  yield both their value and their adjoint dual. Gates: the duals equal
  the hand-built ones of the adjoint tests (station windows, harmonic
  coefficients, the four-slot sea-level functional); the value is the
  pairing of the dual with the state; the adjoint identity per
  measurement, <dJ_i, du>_tangent = d/d(eps) J_i(x + eps delta) by finite
  differences, on the rotating water-loaded ball for the displacement
  cap, the sea-level cap, polar motion, the length of day and harmonic
  coefficients of the potential and of the radial displacement; and the
  time-integrated form on the unit-square Maxwell body, against the
  tangent and the discrete adjoint.
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
constexpr real_t kPi = std::numbers::pi;

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
Vector UnitDir(real_t a, real_t b, real_t c) {
  Vector n(3);
  n[0] = a;
  n[1] = b;
  n[2] = c;
  n /= n.Norml2();
  return n;
}
real_t RelDiff(const Vector& a, const Vector& b) {
  Vector d(a);
  d -= b;
  return d.Norml2() / std::max(b.Norml2(), 1e-300);
}
}  // namespace

// The rotating water-loaded ball of the adjoint tests: the measurement
// duals against the hand-built ones, and the identity per measurement.
TEST(Measurement, WaterLoadedBall3D) {
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
  const real_t kRhoW = 1.0, kOmega = 0.4;
  const real_t g0 = 4.0 * kPi * self_grav_test::kG * self_grav_test::kRho / 3.0;
  ConstantCoefficient w(kRhoW / g0);
  ConstantCoefficient kappa0(self_grav_test::kKappa);
  ConstantCoefficient mu0(self_grav_test::kMu);
  ConstantCoefficient eta0(self_grav_test::kMu);
  FunctionCoefficient sigma_data([](const Vector& x) {
    const real_t r = x.Norml2(), c = x[2] / r;
    return 0.02 * (3.0 * c * c - 0.5 + 0.5 * x[0] * x[2] / (r * r));
  });
  ConstantCoefficient zero_data(0.0);
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

  // The measurements: a radial displacement cap, a sea-level cap, polar
  // motion m_0, the length of day, the (2,0) potential coefficient and
  // the (2,1) radial-displacement coefficient.
  const Vector n1 = UnitDir(0.3, 0.5, 0.81), n3 = UnitDir(0.1, -0.6, 0.79);
  CapWindowCoefficient cap1(n1, std::sqrt(0.08), 0.02), cap3(n3, std::sqrt(0.08), 0.02);
  RadialVectorCoefficient radial_cap1(3, cap1);
  SurfaceDisplacementMeasurement m_u(radial_cap1, surface, "radial cap");
  SeaLevelMeasurement m_sl(cap3, surface, parent_surface);
  RotationMeasurement m_pm(RotationMeasurement::Kind::PolarMotion, 0);
  RotationMeasurement m_lod(RotationMeasurement::Kind::LengthOfDay);
  HarmonicCoefficientMeasurement m_phi20(
      HarmonicCoefficientMeasurement::Field::Potential, parent_surface, 2, 0);
  HarmonicCoefficientMeasurement m_u21(
      HarmonicCoefficientMeasurement::Field::RadialDisplacement, surface, 2, 1);
  const Measurement* ms[] = {&m_u, &m_sl, &m_pm, &m_lod, &m_phi20, &m_u21};
  const int M = 6;

  RatioCoefficient tau0(eta0, mu0);
  auto rheology0 = IsotropicMaxwellRheology::Maxwell(3, kappa0, mu0, tau0);
  auto prob_fwd = make_problem(rheology0, sigma_data);
  MeasurementSet set;
  for (int i = 0; i < M; i++) set.Add(*ms[i], t1);
  set.Assemble(*prob_fwd);

  // --- the duals against the hand-built ones ---------------------------------
  {
    // The radial cap: the test's VectorBoundaryLFIntegrator of h x_hat.
    VectorFunctionCoefficient h_u(3, [&](const Vector& x, Vector& v) {
      Vector d(x);
      d /= x.Norml2();
      d -= n1;
      v = x;
      v *= 0.02 * std::exp(-(d * d) / 0.08) / x.Norml2();
    });
    LinearForm g_u(&fes_u);
    g_u.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(h_u), surface);
    g_u.Assemble();
    EXPECT_LT(RelDiff(set.Dual(0).u, g_u), 1e-13) << "radial cap dual";
    EXPECT_EQ(set.Dual(0).phi.Size(), 0);

    // The sea level: the four slots with g = g0 of the homogeneous ball;
    // the measurement uses the discrete |grad Phi0|, so they agree to the
    // background's discretisation error on this coarse mesh.
    auto melt3 = [&](const Vector& x) {
      Vector d(x);
      d /= x.Norml2();
      d -= n3;
      return 0.02 * std::exp(-(d * d) / 0.08);
    };
    FunctionCoefficient h_sl(melt3);
    VectorFunctionCoefficient h_sl_u(3, [&](const Vector& x, Vector& v) {
      v = x;
      v *= -melt3(x);
    });
    LinearForm g_sl_u(&fes_u);
    g_sl_u.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(h_sl_u), surface);
    g_sl_u.Assemble();
    ProductCoefficient h_sl_phi(-1.0 / g0, h_sl);
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
      sl_omega[k] = -lf.Sum() / g0;
    }
    const MeasurementDual& d = set.Dual(1);
    const real_t e_u = RelDiff(d.u, g_sl_u), e_phi = RelDiff(d.phi, g_sl_phi),
                 e_un = std::abs(d.uniform / sl_uniform - 1.0),
                 e_om = RelDiff(d.omega, sl_omega);
    if (std::getenv("ADGIA_TEST_VERBOSE")) {
      std::cout << "sea-level dual vs hand-built (discrete g vs g0): u " << e_u
                << " phi " << e_phi << " uniform " << e_un << " omega " << e_om
                << "\n";
    }
    EXPECT_LT(e_u, 2e-2);
    EXPECT_LT(e_phi, 2e-2);
    EXPECT_LT(e_un, 2e-2);
    EXPECT_LT(e_om, 2e-2);

    // Rotation: the slots by definition.
    EXPECT_DOUBLE_EQ(set.Dual(2).omega[0], 1.0 / kOmega);
    EXPECT_DOUBLE_EQ(set.Dual(3).omega[2], -1.0 / kOmega);

    // Harmonics: LoadVector of the unit coefficient.
    using BHC = BoundaryHarmonicCoefficients;
    BHC radial(fes_u, surface, 2, BHC::Component::Radial);
    Vector unit(radial.Size()), b;
    unit = 0.0;
    unit[radial.Basis().Index(2, 1)] = 1.0;
    radial.LoadVector(unit, b);
    b /= radial.Radius() * radial.Radius();
    EXPECT_LT(RelDiff(set.Dual(5).u, b), 1e-13) << "harmonic dual";
  }

  // --- the data of the forward run, and the value as a pairing -----------------
  ViscoelasticOperator visco_fwd(*prob_fwd);
  auto trajectory = trajectory_of(visco_fwd);
  Vector d0;
  set.Data(*trajectory, d0);
  {
    // After Data() the problem holds the final state: the harmonic
    // measurement is the pairing of its dual with it, R^{1-d} times the
    // LoadVector pairing, which is exactly the analysis coefficient c_i
    // of the same state (the analysis is the transpose of LoadVector). R
    // is the radius the operator measures on the discrete boundary (the
    // area-weighted mean over its quadrature points), R - 1 = -2.45e-4 on
    // this coarse curved ball, so the unscaled pairing differs from c_i
    // by 4.9e-4. Measured (ADGIA_TEST_VERBOSE=1): |c_i / value - 1| at
    // rounding.
    using BHC = BoundaryHarmonicCoefficients;
    BHC radial(fes_u, surface, 2, BHC::Component::Radial);
    Vector unit(radial.Size()), b;
    unit = 0.0;
    unit[radial.Basis().Index(2, 1)] = 1.0;
    radial.LoadVector(unit, b);
    const real_t by_hand =
        (b * prob_fwd->Displacement()) / (radial.Radius() * radial.Radius());
    EXPECT_NEAR(d0[5], by_hand, 1e-12 * std::abs(by_hand));
    Vector c;
    radial.Coefficients(prob_fwd->Displacement(), c);
    const real_t analysed = c[radial.Basis().Index(2, 1)];
    if (std::getenv("ADGIA_TEST_VERBOSE")) {
      std::cout << "harmonic value " << d0[5]
                << ", R - 1 = " << radial.Radius() - 1.0
                << ", |c_i / value - 1| = "
                << std::abs(analysed / d0[5] - 1.0) << "\n";
    }
    EXPECT_NEAR(d0[5], analysed, 1e-12 * std::abs(by_hand));
    EXPECT_NEAR(d0[2], prob_fwd->AngularVelocity()[0] / kOmega, 1e-15);
  }

  // --- the identity per measurement: tangent against FD of the data ------------
  FunctionCoefficient chi_blob3([](const Vector& x) {
    const real_t dx = x[0] - 0.3, dy = x[1] - 0.35, dz = x[2] - 0.4;
    return std::exp(-(dx * dx + dy * dy + dz * dz) / 0.03);
  });
  ProductCoefficient d_eta(eta0, chi_blob3);
  TangentDirection dir;
  dir.eta.assign(1, &d_eta);
  auto tan_twin = make_problem(rheology0, zero_data);
  TangentLinearSolver tangent(*tan_twin, *trajectory);
  tangent.SetDirection(dir);
  ASSERT_TRUE(tangent.Run());
  const real_t eps = 1e-3;
  Vector dplus, dminus;
  for (const real_t sg : {eps, -eps}) {
    auto ep = Bump(eta0, chi_blob3, sg);
    RatioCoefficient tau(*ep.value, mu0);
    auto rheology = IsotropicMaxwellRheology::Maxwell(3, kappa0, mu0, tau);
    auto prob = make_problem(rheology, sigma_data);
    ViscoelasticOperator visco(*prob);
    auto traj = trajectory_of(visco);
    set.Data(*traj, sg > 0 ? dplus : dminus);
  }
  for (int i = 0; i < M; i++) {
    Vector e(M);
    e = 0.0;
    e[i] = 1.0;
    AdjointForce force = set.Force(&e);
    const real_t dJ_tan = tangent.Pair(force);
    const real_t dJ_fd = (dplus[i] - dminus[i]) / (2 * eps);
    if (std::getenv("ADGIA_TEST_VERBOSE")) {
      std::cout << ms[i]->Name() << ": J " << d0[i] << " dJ tangent " << dJ_tan
                << " fd " << dJ_fd << " | ratio - 1 = " << dJ_tan / dJ_fd - 1.0
                << "\n";
    }
    EXPECT_NEAR(dJ_tan / dJ_fd, 1.0, 1e-4) << ms[i]->Name();
  }
  // The whole set at once (c = ones) is the sum.
  {
    AdjointForce force = set.Force();
    real_t sum = 0.0;
    for (int i = 0; i < M; i++) sum += (dplus[i] - dminus[i]) / (2 * eps);
    EXPECT_NEAR(tangent.Pair(force) / sum, 1.0, 1e-4) << "merged observation";
  }
}

// The unit-square Maxwell body: a point station and a time-integrated
// station, data against the hand-computed trapezoid sum, and the
// identity against the tangent and the discrete adjoint.
TEST(Measurement, TimeIntegrated2D) {
  const int n_el = 12, order = 2, n_steps = 16;
  const real_t t_final = 2.0, dt = t_final / n_steps;
  Mesh mesh = Mesh::MakeCartesian2D(n_el, n_el, Element::QUADRILATERAL);
  H1_FECollection fec(order, 2);
  FiniteElementSpace fes(&mesh, &fec, 2);
  Array<int> top(mesh.bdr_attributes.Max()), bottom(mesh.bdr_attributes.Max());
  top = 0;
  bottom = 0;
  top[2] = 1;
  bottom[0] = 1;
  VectorFunctionCoefficient traction(2, [](const Vector& x, Vector& t) {
    t.SetSize(2);
    t[0] = 0.0;
    t[1] = -0.05 * std::cos(2.0 * kPi * x[0]);
  });
  VectorFunctionCoefficient station(2, [](const Vector& x, Vector& v) {
    constexpr real_t s = 0.08;
    auto window = [](real_t x0, real_t c) {
      return std::exp(-(x0 - c) * (x0 - c) / (s * s)) / (s * std::sqrt(kPi));
    };
    v.SetSize(2);
    v[0] = 0.0;
    v[1] = window(x[0], 0.55) - window(x[0], 0.2);
  });
  Vector zero_vec(2);
  zero_vec = 0.0;
  VectorConstantCoefficient zero_traction(zero_vec);
  ConstantCoefficient kappa0(5.4), mu0(2.7), eta0(2.7);
  FunctionCoefficient chi([](const Vector& x) {
    const real_t r2 = (x[0] - 0.4) * (x[0] - 0.4) + (x[1] - 0.6) * (x[1] - 0.6);
    return std::exp(-r2 / 0.03);
  });
  auto weight = [t_final](real_t t) { return t / t_final; };

  SurfaceDisplacementMeasurement m_station(station, bottom, "station");
  MeasurementSet set;
  set.Add(m_station, t_final);
  set.AddIntegrated(m_station, weight);

  bool assembled = false;
  auto run = [&](Coefficient& kappa, ForwardTrajectory** keep,
                 std::unique_ptr<LinearQuasiStaticTractionProblem>* keep_prob,
                 std::unique_ptr<ViscoelasticOperator>* keep_visco) {
    RatioCoefficient tau(eta0, mu0);
    auto rheology = IsotropicMaxwellRheology::Maxwell(2, kappa, mu0, tau);
    auto prob = std::make_unique<LinearQuasiStaticTractionProblem>(&fes, rheology, traction, top);
    auto visco = std::make_unique<ViscoelasticOperator>(*prob);
    auto traj = new ForwardTrajectory(*visco, true);
    ExponentialTrapezoidSolver ode;
    ode.Init(*visco);
    Vector m(visco->Height());
    m = 0.0;
    real_t t = 0.0, dts = dt;
    traj->Record(m, t);
    for (int n = 1; n <= n_steps; n++) {
      ode.Step(m, t, dts);
      traj->Record(m, t);
    }
    if (!assembled) {
      set.Assemble(*prob);
      assembled = true;
    }
    Vector d;
    set.Data(*traj, d);
    if (keep) {
      *keep = traj;
      *keep_prob = std::move(prob);
      *keep_visco = std::move(visco);
    } else {
      delete traj;
    }
    return d;
  };

  ForwardTrajectory* traj0 = nullptr;
  std::unique_ptr<LinearQuasiStaticTractionProblem> prob0;
  std::unique_ptr<ViscoelasticOperator> visco0;
  Vector d0 = run(kappa0, &traj0, &prob0, &visco0);
  std::unique_ptr<ForwardTrajectory> traj_owner(traj0);

  // The data by hand: g . u at the last node, and the trapezoid sum.
  {
    LinearForm g(&fes);
    g.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(station), bottom);
    g.Assemble();
    real_t J_point = g * traj0->Displacement(n_steps), J_int = 0.0;
    for (int n = 0; n <= n_steps; n++) {
      const real_t wq = (n == 0 || n == n_steps) ? 0.5 * dt : dt;
      J_int += wq * weight(traj0->Time(n)) * (g * traj0->Displacement(n));
    }
    // Data() re-solves each node: equal to the stored fields at the
    // solver tolerance (1.6e-12 relative here).
    EXPECT_NEAR(d0[0], J_point, 1e-9 * std::abs(J_point));
    EXPECT_NEAR(d0[1], J_int, 1e-9 * std::abs(J_int));
  }

  // The identity: tangent and discrete adjoint along a kappa blob against
  // FD of the data, for the point and the integrated observation.
  ProductCoefficient d_kappa(kappa0, chi);
  TangentDirection dir;
  dir.kappa = &d_kappa;
  dir.mu.assign(1, nullptr);
  dir.eta.assign(1, nullptr);
  const real_t eps = 1e-3;
  auto kp = Bump(kappa0, chi, eps), km = Bump(kappa0, chi, -eps);
  Vector dplus = run(*kp.value, nullptr, nullptr, nullptr);
  Vector dminus = run(*km.value, nullptr, nullptr, nullptr);
  RatioCoefficient tau0(eta0, mu0);
  auto rheology0 = IsotropicMaxwellRheology::Maxwell(2, kappa0, mu0, tau0);
  LinearQuasiStaticTractionProblem tan_twin(&fes, rheology0, zero_traction, top);
  TangentLinearSolver tangent(tan_twin, *traj0);
  tangent.SetDirection(dir);
  ASSERT_TRUE(tangent.Run());
  LinearQuasiStaticTractionProblem twin(&fes, rheology0, zero_traction, top);
  LinearQuasiStaticTractionProblem stwin(&fes, rheology0, zero_traction, top);
  DiscreteAdjointSolver discrete(twin, stwin, *traj0);
  const char* names[] = {"point station", "integrated station"};
  for (int i = 0; i < 2; i++) {
    Vector e(2);
    e = 0.0;
    e[i] = 1.0;
    AdjointForce force = set.Force(&e);
    const real_t dJ_tan = tangent.Pair(force);
    const real_t dJ_fd = (dplus[i] - dminus[i]) / (2 * eps);
    discrete.SetForce(force);
    ASSERT_TRUE(discrete.Run());
    const real_t dJ_disc = discrete.Directional(dir);
    if (std::getenv("ADGIA_TEST_VERBOSE")) {
      std::cout << names[i] << ": J " << d0[i] << " tangent " << dJ_tan
                << " fd " << dJ_fd << " discrete " << dJ_disc << "\n";
    }
    EXPECT_NEAR(dJ_tan / dJ_fd, 1.0, 1e-5) << names[i];
    EXPECT_NEAR(dJ_disc / dJ_tan, 1.0, 1e-8) << names[i];
  }
  // The static data: the problem's current state.
  {
    MeasurementSet s;
    s.Add(m_station, 0.0);
    s.Assemble(*prob0);
    Vector d;
    s.Data(*prob0, d);
    EXPECT_NEAR(d[0], Evaluate(s.Dual(0), *prob0), 0.0);
  }
}
