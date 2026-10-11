#include <numbers>

#include "MixedProblemTestCommon.hpp"
#include "TestCommon.hpp"

/*
  WP7 phase-A gates of the referential water load
  (doc/planning/referential_sea_level_plan.md; quasi_static_models.tex,
  the referential sea-level subsection): SetWaterLoad on the welded
  referential class — the zeta-zeta collapse — on the canned two-layer
  ball, all-ocean (C0 = 1), small water density, phi_e = id. 3-D only,
  by the sea-level rule.

  1. Monolithic == fixed point (G3): the pyslfp-style iteration with
     the water load evaluated from zeta directly. In the referential
     variables the instrument and the operator integrate the SAME
     field (no normal-projected vs full-gradient split as in the mixed
     class), so the agreement is instrument-quadrature grade.
  2. The spectral identity (G1): with T the zeta/g amplitude response
     per unit degree-2 surface mass (one plain solve), the monolithic
     sea-level amplitude satisfies s = -T/(1 + rho_w T).
  3. Referential == mixed (G2): the same physical problem through both
     classes (the hydrostatic cross-check pattern with water on):
     u directly, zeta against phi + u.grad Phi0, Phi_g against Phi_g,
     at the two discretisations' common accuracy, improving with
     order.
  4. Mass conservation and the uniform term (G4): the re-integrated
     certificate at the solution; Phi_g inert for a zero-mean load.
*/

namespace {

using namespace self_grav_test;

constexpr double kRhoW = 0.02;

// Zonal degree-2, zero-mean pattern on the unit sphere.
double Degree2(const Vector& x) {
  const double r = x.Norml2();
  const double c = x[2] / r;
  return 3.0 * c * c - 1.0;
}

double G0() { return 4.0 * std::numbers::pi * kG * kRho / 3.0; }

// 3-D uniform-ball hydrostatic pressure,
// p0(r) = (2/3) pi G rho^2 (1 - r^2).
double UniformBallPressure(const Vector& x) {
  const double r2 = x * x;
  return 2.0 * std::numbers::pi * kG * kRho * kRho * (1.0 - r2) / 3.0;
}

CallableDiffeomorphism IdentityMap(int dim) {
  return CallableDiffeomorphism(
      dim, [](const Vector& x, Vector& y) { y = x; },
      [](const Vector&, DenseMatrix& F) {
        F = 0.0;
        for (int i = 0; i < F.Height(); i++) {
          F(i, i) = 1.0;
        }
      });
}

// The iterated water load of the fixed-point instrument, referential
// form: sigma_w = w (c - zeta), zeta from a stored copy of the last
// solution's body potential.
class IteredZetaLoad : public Coefficient {
 public:
  explicit IteredZetaLoad(Coefficient& w) : w_(&w) {}

  void Update(const GridFunction& zeta_body, double uniform) {
    zeta_ = std::make_unique<GridFunction>(zeta_body);
    c_ = uniform;
  }

  double Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    if (!zeta_) {
      return 0.0;
    }
    return w_->Eval(T, ip) * (c_ - zeta_->GetValue(T, ip));
  }

 private:
  Coefficient* w_;
  std::unique_ptr<GridFunction> zeta_;
  double c_ = 0.0;
};

// Both classes on one parent/body pair, sharing fes_u and the parent
// scalar space, with the referential hydrostatic reference state at
// phi_e = id (the HydrostaticCrossCheck construction) and the radial
// vacuum extension.
struct RefCase {
  std::unique_ptr<Mesh> parent;
  std::unique_ptr<SubMesh> body;
  std::unique_ptr<H1_FECollection> fec;
  std::unique_ptr<FiniteElementSpace> fes_u, fes_scalar;
  std::unique_ptr<SubMesh> buffer;
  std::unique_ptr<FiniteElementSpace> fes_buffer;
  std::unique_ptr<SparseMatrix> E;
  double r_out = 0.0;

  ConstantCoefficient kappa{kKappa}, mu{kMu}, rho{kRho};
  std::unique_ptr<IsotropicElasticRheology> e_rheology;
  CallableDiffeomorphism map;
  FunctionCoefficient p0{UniformBallPressure};
  std::unique_ptr<IsotropicElasticTensorCoefficient> C_eff;
  std::unique_ptr<BareElasticTensorCoefficient> C;
  std::unique_ptr<MatrixFunctionCoefficient> S;
  std::unique_ptr<ReferentialElasticRheology> r_rheology;

  Array<int> surface;
  std::unique_ptr<ConstantCoefficient> w;

  explicit RefCase(int order) : map(IdentityMap(3)) {
    const int dim = 3;
    parent = std::make_unique<Mesh>(MeshFile(dim).c_str(), 1, 1);
    body = std::make_unique<SubMesh>(
        SubMesh::CreateFromDomain(*parent, BodyMarker(*parent)));
    fec = std::make_unique<H1_FECollection>(order, dim);
    fes_u = std::make_unique<FiniteElementSpace>(body.get(), fec.get(), dim);
    fes_scalar = std::make_unique<FiniteElementSpace>(parent.get(), fec.get());
    Array<int> buffer_attr({2});
    buffer = std::make_unique<SubMesh>(
        SubMesh::CreateFromDomain(*parent, buffer_attr));
    fes_buffer =
        std::make_unique<FiniteElementSpace>(buffer.get(), fec.get(), dim);
    Vector bb_min, bb_max;
    parent->GetBoundingBox(bb_min, bb_max);
    r_out = bb_max.Normlinf();
    E = NewRadialVacuumExtension(*fes_u, *fes_buffer, 1.0, r_out);

    e_rheology = std::make_unique<IsotropicElasticRheology>(dim, kappa, mu);
    C_eff = std::make_unique<IsotropicElasticTensorCoefficient>(
        IsotropicElasticTensorCoefficient::FromBulkModulus(dim, kappa, mu));
    C = std::make_unique<BareElasticTensorCoefficient>(dim, *C_eff, p0);
    S = std::make_unique<MatrixFunctionCoefficient>(
        dim, [](const Vector& x, DenseMatrix& Se) {
          Se.SetSize(x.Size());
          Se = 0.0;
          const double p = UniformBallPressure(x);
          for (int i = 0; i < x.Size(); i++) {
            Se(i, i) = -p;
          }
        });
    r_rheology =
        std::make_unique<ReferentialElasticRheology>(dim, *C, *S, map);

    surface = SurfaceMarker(*body);
    w = std::make_unique<ConstantCoefficient>(kRhoW / G0());
  }

  std::unique_ptr<LinearQuasiStaticReferentialSelfGravitatingProblem>
  Referential() {
    auto p =
        std::make_unique<LinearQuasiStaticReferentialSelfGravitatingProblem>(
            fes_u.get(), fes_scalar.get(), *r_rheology, rho, kG, kDtNDegree);
    p->SetPrescribedVacuumExtension(*fes_buffer, *E);
    p->SetRelTol(1e-12);
    return p;
  }

  std::unique_ptr<LinearQuasiStaticMixedSelfGravitatingProblem> Mixed() {
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        fes_u.get(), fes_scalar.get(), *e_rheology, rho, kG, kDtNDegree);
    p->SetRelTol(1e-12);
    return p;
  }

  // int f dS over the body surface.
  double SurfaceIntegral(Coefficient& f) {
    H1_FECollection sfec(fes_u->GetMaxElementOrder(), body->Dimension());
    FiniteElementSpace sfes(body.get(), &sfec);
    LinearForm lf(&sfes);
    lf.AddBoundaryIntegrator(new BoundaryLFIntegrator(f), surface);
    lf.Assemble();
    return lf.Sum();
  }

  // Amplitude of the Degree2 pattern in a body-evaluable coefficient.
  double PatternAmplitude(Coefficient& f) {
    FunctionCoefficient pat(Degree2);
    ProductCoefficient fp(f, pat), pp(pat, pat);
    return SurfaceIntegral(fp) / SurfaceIntegral(pp);
  }
};

using Param = int;  // order; 3-D only

class SeaLevelReferentialTest : public testing::TestWithParam<Param> {};

TEST_P(SeaLevelReferentialTest, MonolithicMatchesFixedPoint) {
  const int order = GetParam();
  RefCase c(order);
  FunctionCoefficient sigma_data(Degree2);

  auto p = c.Referential();
  p->SetWaterLoad(*c.w, sigma_data, c.surface);
  p->AssembleForce(0.0);
  ASSERT_TRUE(p->Solve());
  const double phi_g = p->UniformPotentialTerm();

  // Fixed point on a fresh problem; uniform from mass conservation
  // each pass.
  auto q = c.Referential();
  IteredZetaLoad water(*c.w);
  q->SetSurfaceLoad(sigma_data, c.surface);
  q->SetSurfaceLoad(water, c.surface);
  const double m = [&] {
    ConstantCoefficient one(1.0);
    ProductCoefficient wone(*c.w, one);
    return c.SurfaceIntegral(wone);
  }();
  const double Sd = c.SurfaceIntegral(sigma_data);
  double uniform = 0.0;
  for (int it = 0; it < 20; it++) {
    q->AssembleForce(0.0);
    ASSERT_TRUE(q->Solve());
    GridFunctionCoefficient zc(
        const_cast<GridFunction*>(&q->PotentialOnBody()));
    ProductCoefficient wz(*c.w, zc);
    uniform = (c.SurfaceIntegral(wz) - Sd) / m;
    water.Update(q->PotentialOnBody(), uniform);
  }
  // Same field on both routes: instrument-quadrature grade.
  EXPECT_NEAR(uniform, phi_g, 2e-4 * std::max(1.0, std::abs(phi_g)));
  GridFunction du(p->Displacement());
  du -= q->Displacement();
  EXPECT_LT(L2Norm(du), 1e-3 * (L2Norm(p->Displacement()) + 1e-30));
}

TEST_P(SeaLevelReferentialTest, SpectralIdentityAndMass) {
  const int order = GetParam();
  RefCase c(order);
  FunctionCoefficient sigma_data(Degree2);

  // T: zeta/g amplitude per unit degree-2 surface mass (plain solve).
  auto pt = c.Referential();
  pt->SetSurfaceLoad(sigma_data, c.surface);
  pt->AssembleForce(0.0);
  ASSERT_TRUE(pt->Solve());
  const double T = [&] {
    GridFunctionCoefficient zc(
        const_cast<GridFunction*>(&pt->PotentialOnBody()));
    return c.PatternAmplitude(zc) / G0();
  }();

  // Monolithic sea-level amplitude s = -amp(zeta)/g (the uniform term
  // has no degree-2 component).
  auto p = c.Referential();
  p->SetWaterLoad(*c.w, sigma_data, c.surface);
  p->AssembleForce(0.0);
  ASSERT_TRUE(p->Solve());
  const double s = [&] {
    GridFunctionCoefficient zc(
        const_cast<GridFunction*>(&p->PotentialOnBody()));
    return -c.PatternAmplitude(zc) / G0();
  }();
  EXPECT_NEAR(s, -T / (1.0 + kRhoW * T), 2e-3 * std::abs(T));
  // Zero-mean load, all ocean: Phi_g carries only the pattern's
  // quadrature-level mean on the curved boundary.
  EXPECT_LT(std::abs(p->UniformPotentialTerm()), 2e-3);

  // Mass conservation of the total load at the solution,
  // re-integrated through the certificate's own quadrature route.
  GridFunctionCoefficient zc(
      const_cast<GridFunction*>(&p->PotentialOnBody()));
  ProductCoefficient wz(*c.w, zc);
  const double total = c.SurfaceIntegral(sigma_data) -
                       c.SurfaceIntegral(wz) +
                       p->UniformPotentialTerm() * [&] {
                         ConstantCoefficient one(1.0);
                         ProductCoefficient wone(*c.w, one);
                         return c.SurfaceIntegral(wone);
                       }();
  EXPECT_LT(std::abs(total), 5e-5);
}

TEST(SeaLevelReferential, MatchesMixedWaterSolve) {
  // G2: the hydrostatic cross-check with water on, at both orders:
  // common-discretisation agreement, improving with order.
  std::vector<double> u_diff, z_diff, g_diff;
  for (int order : {1, 2}) {
    RefCase c(order);
    // A load with mean, so that Phi_g is active on both sides.
    FunctionCoefficient sigma_data(
        [](const Vector& x) { return Degree2(x) + 0.5; });

    auto pm = c.Mixed();
    pm->SetWaterLoad(*c.w, sigma_data, c.surface);
    pm->AssembleForce(0.0);
    ASSERT_TRUE(pm->Solve());

    auto pr = c.Referential();
    pr->SetWaterLoad(*c.w, sigma_data, c.surface);
    pr->AssembleForce(0.0);
    ASSERT_TRUE(pr->Solve());

    {
      GridFunction du(pr->Displacement());
      du -= pm->Displacement();
      u_diff.push_back(L2Norm(du) / L2Norm(pm->Displacement()));
    }
    {
      // zeta1 = phi1 + u.grad Phi0 on the body.
      VectorGridFunctionCoefficient uc(&pm->Displacement());
      auto grad_phi0 = VectorFunctionCoefficient(
          3, [](const Vector& x, Vector& v) {
            v = x;
            v *= G0();
          });
      InnerProductCoefficient advect(uc, grad_phi0);
      GridFunctionCoefficient phi1(
          const_cast<GridFunction*>(&pm->PotentialOnBody()));
      SumCoefficient zeta_expected(phi1, advect);
      GridFunction d(pr->PotentialOnBody());
      GridFunction z(d);
      z.ProjectCoefficient(zeta_expected);
      d -= z;
      z_diff.push_back(L2Norm(d) /
                       std::max(1e-30, L2Norm(pr->PotentialOnBody())));
    }
    g_diff.push_back(std::abs(pr->UniformPotentialTerm() -
                              pm->UniformPotentialTerm()) /
                     std::max(1e-30,
                              std::abs(pm->UniformPotentialTerm())));
    {
      // The SeaLevelOperator's referential overload against the mixed
      // route, on one operator (same surface space): SL1 from
      // (u, phi, Phi_g) and from (zeta, Phi_g) agree at the classes'
      // common-discretisation grade.
      auto grad_phi0 = VectorFunctionCoefficient(
          3, [](const Vector& x, Vector& v) {
            v = x;
            v *= G0();
          });
      SeaLevelOperator sea(*c.fes_u, c.surface);
      sea.SeaLevelChangeFrom(pm->Displacement(), grad_phi0,
                             pm->Potential(), pm->UniformPotentialTerm());
      GridFunction sl_mixed(sea.SeaLevelChangeField());
      sea.SeaLevelChangeFrom(pr->PotentialOnBody(), grad_phi0,
                             pr->UniformPotentialTerm());
      GridFunction d(sea.SeaLevelChangeField());
      d -= sl_mixed;
      EXPECT_LT(L2Norm(d), 6e-2 * (L2Norm(sl_mixed) + 1e-30))
          << "order " << order;
    }
  }
  EXPECT_LT(u_diff[1], 5e-2);
  EXPECT_LT(u_diff[1], 0.7 * u_diff[0]);
  EXPECT_LT(z_diff[1], 5e-2);
  EXPECT_LT(z_diff[1], 0.7 * z_diff[0]);
  EXPECT_LT(g_diff[1], 5e-2);
}

TEST(SeaLevelReferential, RotationMatchesMixed) {
  // G6: the traditional rotation border through both classes on the
  // same spherical model, rotation-only and composed with water.
  // The load carries degree-2 order-1 content so the wander rows are
  // genuinely driven, and the moments sit away from near-neutral
  // wander (both WP4 test-trap lessons).
  const double Omega = 0.1;
  Vector moments(3);
  moments[0] = 1.2;
  moments[1] = 1.3;
  moments[2] = 2.0;
  std::vector<double> w_only, w_comp, u_comp;
  for (int order : {1, 2}) {
    RefCase c(order);
    FunctionCoefficient sigma_data([](const Vector& x) {
      const double r2 = x * x;
      return Degree2(x) + 2.0 * x[0] * x[2] / r2 + 0.5;
    });

    // Rotation only.
    auto pm = c.Mixed();
    pm->SetRotation(Omega, moments);
    pm->SetSurfaceLoad(sigma_data, c.surface);
    pm->AssembleForce(0.0);
    ASSERT_TRUE(pm->Solve());
    auto pr = c.Referential();
    pr->SetRotation(Omega, moments);
    pr->SetSurfaceLoad(sigma_data, c.surface);
    pr->AssembleForce(0.0);
    ASSERT_TRUE(pr->Solve());
    {
      Vector dw(pr->AngularVelocity());
      dw -= pm->AngularVelocity();
      w_only.push_back(dw.Norml2() / pm->AngularVelocity().Norml2());
    }

    // Composed with water.
    auto qm = c.Mixed();
    qm->SetWaterLoad(*c.w, sigma_data, c.surface);
    qm->SetRotation(Omega, moments);
    qm->AssembleForce(0.0);
    ASSERT_TRUE(qm->Solve());
    auto qr = c.Referential();
    qr->SetWaterLoad(*c.w, sigma_data, c.surface);
    qr->SetRotation(Omega, moments);
    qr->AssembleForce(0.0);
    ASSERT_TRUE(qr->Solve());
    {
      Vector dw(qr->AngularVelocity());
      dw -= qm->AngularVelocity();
      w_comp.push_back(dw.Norml2() / qm->AngularVelocity().Norml2());
      GridFunction du(qr->Displacement());
      du -= qm->Displacement();
      u_comp.push_back(L2Norm(du) / L2Norm(qm->Displacement()));
    }

    // Omega -> 0 reduces to the water-only solve (tight: the columns
    // and the row data vanish with psi).
    if (order == 1) {
      auto z = c.Referential();
      z->SetWaterLoad(*c.w, sigma_data, c.surface);
      z->SetRotation(0.0, moments);
      z->AssembleForce(0.0);
      ASSERT_TRUE(z->Solve());
      EXPECT_LT(z->AngularVelocity().Norml2(), 1e-12);
      auto w0 = c.Referential();
      w0->SetWaterLoad(*c.w, sigma_data, c.surface);
      w0->AssembleForce(0.0);
      ASSERT_TRUE(w0->Solve());
      GridFunction du(z->Displacement());
      du -= w0->Displacement();
      EXPECT_LT(L2Norm(du), 1e-9 * (L2Norm(w0->Displacement()) + 1e-30));
    }
  }
  // Common-discretisation agreement, improving with order.
  EXPECT_LT(w_only[1], 5e-2);
  EXPECT_LT(w_only[1], 0.7 * w_only[0]);
  EXPECT_LT(w_comp[1], 5e-2);
  EXPECT_LT(w_comp[1], 0.7 * w_comp[0]);
  EXPECT_LT(u_comp[1], 5e-2);
  EXPECT_LT(u_comp[1], 0.7 * u_comp[0]);
}

// --- Shoreline migration on the referential class ---------------------------
// The migrating state of TestShorelineMigration.cpp (polar continent,
// polar cap, half the cap melting), at its densities.
constexpr double kMigRhoW = 0.05;
constexpr double kMigRhoI = 0.045;

double Colat(const Vector& x) {
  const double r = x.Norml2();
  const double c = x[2] / r;
  return std::acos(std::min(1.0, std::max(-1.0, c)));
}

double SL0(const Vector& x) {
  const double t = Colat(x) / 0.7;
  return 1.0 - 3.0 * std::exp(-0.5 * std::pow(t, 6));
}

double Ice0(const Vector& x) {
  const double t = Colat(x) / 0.35;
  return 4.0 * std::exp(-0.5 * t * t);
}

double Fraction0(const Vector& x) {
  const double q = kMigRhoW * SL0(x) - kMigRhoI * Ice0(x);
  return 0.5 * (1.0 + std::tanh(q / 5e-3));
}

TEST(SeaLevelReferential, ShorelineMigrationMatchesMixed) {
  // WP7.3: the shared orchestrator drives the referential class.
  // (a) The off-switch reduces to the direct frozen-C referential
  // water solve (the C = C0 reduction of the load law), tight.
  // (b) The migrating referential run agrees with the migrating mixed
  // run at the classes' common-discretisation grade.
  const int order = 1;
  RefCase c(order);
  auto grad_phi0 =
      VectorFunctionCoefficient(3, [](const Vector& x, Vector& v) {
        v = x;
        v *= G0();
      });
  FunctionCoefficient sl0(SL0), ice0(Ice0);
  FunctionCoefficient dice([](const Vector& x) { return -0.5 * Ice0(x); });

  auto run = [&](auto& problem, int max_iterations, double shore) {
    ShorelineMigration::Options opt;
    opt.max_iterations = max_iterations;
    opt.tol = 1e-6;
    opt.shore = shore;
    opt.inexact = 0.0;
    ShorelineMigration mig(problem, grad_phi0, sl0, ice0, dice, kMigRhoW,
                           kMigRhoI, c.surface, opt);
    EXPECT_TRUE(mig.Solve(0.0));
    return mig.Iterations();
  };

  // (a) Off-switch == direct frozen-C referential solve.
  {
    auto p = c.Referential();
    const int its = run(*p, 0, 5e-3);
    EXPECT_EQ(its, 0);
    FunctionCoefficient w0([](const Vector& x) {
      return kMigRhoW * Fraction0(x) / (G0() * x.Norml2());
    });
    FunctionCoefficient sd0([](const Vector& x) {
      return kMigRhoI * (1.0 - Fraction0(x)) * (-0.5 * Ice0(x));
    });
    auto q = c.Referential();
    q->SetWaterLoad(w0, sd0, c.surface);
    q->AssembleForce(0.0);
    ASSERT_TRUE(q->Solve());
    GridFunction du(p->Displacement());
    du -= q->Displacement();
    EXPECT_LT(L2Norm(du), 1e-9 * (L2Norm(q->Displacement()) + 1e-30));
    EXPECT_NEAR(p->UniformPotentialTerm(), q->UniformPotentialTerm(),
                1e-10);
  }

  // (b) Migrating referential == migrating mixed: the classes'
  // common-discretisation agreement, improving with order (the weak
  // migrating state makes the o1 difference percent-grade, like the
  // other cross-class gates' first rungs).
  std::vector<double> u_diff;
  for (int o : {1, 2}) {
    RefCase cb(o);
    auto pr = cb.Referential();
    run(*pr, 12, 5e-2);
    auto pm = cb.Mixed();
    run(*pm, 12, 5e-2);
    GridFunction du(pr->Displacement());
    du -= pm->Displacement();
    u_diff.push_back(L2Norm(du) / (L2Norm(pm->Displacement()) + 1e-30));
    if (o == 2) {
      EXPECT_NEAR(
          pr->UniformPotentialTerm(), pm->UniformPotentialTerm(),
          6e-2 * std::max(1.0, std::abs(pm->UniformPotentialTerm())));
    }
  }
  EXPECT_LT(u_diff[1], 6e-2);
  EXPECT_LT(u_diff[1], 0.7 * u_diff[0]);
}

INSTANTIATE_TEST_SUITE_P(SeaLevelReferential, SeaLevelReferentialTest,
                         testing::Values(1, 2));

}  // namespace
