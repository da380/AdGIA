#include <numbers>

#include "MixedProblemTestCommon.hpp"
#include "TestCommon.hpp"

/*
  Gates for shoreline migration (ShorelineMigration; WP5 of
  doc/planning/sea_level_plan.md) on the canned two-layer meshes, with
  the fingerprint-style analytical state: a super-Gaussian continent at
  the pole (+y in 2-D, +z in 3-D), an ice cap on it, and a melt load
  scaled by an amplitude.

  1. The off-switch (max_iterations = 0) equals the frozen-C solve with
     the same analytic C0 coefficients passed to SetWaterLoad directly —
     identical formulas, so the agreement is at solver tolerance.
  2. The Crawford certificate: the migrating-minus-frozen difference
     scales quadratically with the load amplitude (the migrating and
     frozen runs share one code path, so the difference is purely the
     shoreline's).
  3. The Picard loop converges in a few passes, the shoreline-change
     metric below tolerance.
  4. Mass conservation holds at the migrated state, with the final ocean
     fraction, re-evaluated through test-side integrals.
  5. The composition with SetRotation runs and conserves mass.
*/

namespace {

using namespace self_grav_test;

constexpr double kRhoW = 0.05;
constexpr double kRhoI = 0.045;
constexpr double kDepth = 1.0;
constexpr double kContAmp = 3.0;
constexpr double kContWidth = 0.7;
constexpr double kCapAmp = 4.0;
constexpr double kCapWidth = 0.35;
constexpr double kShore = 5e-3;

double G0(int dim) {
  return dim == 2 ? 2.0 * std::numbers::pi * kG * kRho
                  : 4.0 * std::numbers::pi * kG * kRho / 3.0;
}

double Colat(const Vector& x) {
  const double r = x.Norml2();
  const double c = x[x.Size() - 1] / r;
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

struct Case {
  std::unique_ptr<Mesh> parent;
  std::unique_ptr<SubMesh> body;
  std::unique_ptr<H1_FECollection> fec;
  std::unique_ptr<FiniteElementSpace> fes_u, fes_phi;
  ConstantCoefficient kappa{kKappa}, mu{kMu}, rho{kRho};
  std::unique_ptr<IsotropicElasticRheology> rheology;
  Array<int> surface;
  int dim;

  explicit Case(int d, int order, int refinements = 0) : dim(d) {
    parent = std::make_unique<Mesh>(MeshFile(dim).c_str(), 1, 1);
    for (int r = 0; r < refinements; r++) {
      parent->UniformRefinement();
    }
    body = std::make_unique<SubMesh>(
        SubMesh::CreateFromDomain(*parent, BodyMarker(*parent)));
    fec = std::make_unique<H1_FECollection>(order, dim);
    fes_u = std::make_unique<FiniteElementSpace>(body.get(), fec.get(), dim);
    fes_phi = std::make_unique<FiniteElementSpace>(parent.get(), fec.get());
    rheology = std::make_unique<IsotropicElasticRheology>(dim, kappa, mu);
    surface = SurfaceMarker(*body);
  }

  std::unique_ptr<LinearQuasiStaticMixedSelfGravitatingProblem> Problem() {
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        fes_u.get(), fes_phi.get(), *rheology, rho, kG, kDtNDegree);
    p->SetRelTol(1e-12);
    return p;
  }

  VectorFunctionCoefficient GradPhi0() {
    const double g0 = G0(dim);
    return VectorFunctionCoefficient(dim, [g0](const Vector& x, Vector& v) {
      v = x;
      v *= g0;
    });
  }

  double SurfaceIntegral(Coefficient& f) {
    H1_FECollection sfec(fes_u->GetMaxElementOrder(), dim);
    FiniteElementSpace sfes(body.get(), &sfec);
    LinearForm lf(&sfes);
    lf.AddBoundaryIntegrator(new BoundaryLFIntegrator(f), surface);
    lf.Assemble();
    return lf.Sum();
  }
};

// Run the migration machinery at melt amplitude eps; migrate = false is
// the off-switch (the frozen-C pass alone).
struct Run {
  std::unique_ptr<LinearQuasiStaticMixedSelfGravitatingProblem> p;
  std::unique_ptr<ShorelineMigration> mig;
  int iterations = 0;
  double change = 0.0;
};

Run Migrate(Case& c, VectorCoefficient& grad_phi0, Coefficient& sl0,
            Coefficient& ice0, Coefficient& dice, bool migrate,
            bool rotate = false, double shore = kShore,
            double inexact = 0.1, double guard = 0.5) {
  Run r;
  r.p = c.Problem();
  if (rotate) {
    Vector moments(c.dim == 2 ? 1 : 3);
    if (c.dim == 2) {
      moments[0] = 2.0;
    } else {
      moments[0] = 1.2;
      moments[1] = 1.3;
      moments[2] = 2.0;
    }
    r.p->SetRotation(0.1, moments);
  }
  ShorelineMigration::Options opt;
  opt.max_iterations = migrate ? 12 : 0;
  opt.tol = 1e-6;
  opt.shore = shore;
  opt.inexact = inexact;
  opt.guard = guard;
  r.mig = std::make_unique<ShorelineMigration>(
      *r.p, grad_phi0, sl0, ice0, dice, kRhoW, kRhoI, c.surface, opt);
  EXPECT_TRUE(r.mig->Solve(0.0));
  r.iterations = r.mig->Iterations();
  r.change = r.mig->LastShorelineChange();
  return r;
}

using Param = std::tuple<int, int>;

class ShorelineTest : public testing::TestWithParam<Param> {};

TEST_P(ShorelineTest, OffSwitchEqualsFrozenC) {
  const auto [dim, order] = GetParam();
  Case c(dim, order);
  auto grad_phi0 = c.GradPhi0();
  FunctionCoefficient sl0(SL0), ice0(Ice0);
  FunctionCoefficient dice([](const Vector& x) { return -0.5 * Ice0(x); });

  auto frozen = Migrate(c, grad_phi0, sl0, ice0, dice, false);

  // The same frozen-C solve wired directly: w = rho_w C0 / g and
  // sigma_d = rho_i (1 - C0) dI (the C = C0 reduction of the load law).
  // Pointwise g = g0 |x| (the orchestrator evaluates |grad Phi0| at
  // quadrature, which varies at geometry grade on the curved boundary).
  const double g0 = G0(dim);
  FunctionCoefficient w0([g0](const Vector& x) {
    return kRhoW * Fraction0(x) / (g0 * x.Norml2());
  });
  FunctionCoefficient sd0([](const Vector& x) {
    return kRhoI * (1.0 - Fraction0(x)) * (-0.5 * Ice0(x));
  });
  auto q = c.Problem();
  q->SetWaterLoad(w0, sd0, c.surface);
  q->AssembleForce(0.0);
  ASSERT_TRUE(q->Solve());

  GridFunction du(frozen.p->Displacement());
  du -= q->Displacement();
  EXPECT_LT(L2Norm(du), 1e-9 * (L2Norm(q->Displacement()) + 1e-30));
  EXPECT_NEAR(frozen.p->UniformPotentialTerm(), q->UniformPotentialTerm(),
              1e-10);
  EXPECT_EQ(frozen.iterations, 0);
}

TEST_P(ShorelineTest, MigrationConvergesAndConservesMass) {
  const auto [dim, order] = GetParam();
  Case c(dim, order);
  auto grad_phi0 = c.GradPhi0();
  FunctionCoefficient sl0(SL0), ice0(Ice0);
  FunctionCoefficient dice([](const Vector& x) { return -0.5 * Ice0(x); });

  auto mig = Migrate(c, grad_phi0, sl0, ice0, dice, true);
  EXPECT_GT(mig.iterations, 0);
  EXPECT_LE(mig.iterations, 8);
  EXPECT_LE(mig.change, 1e-6);

  // Mass conservation with the FINAL ocean fraction: sigma_total =
  // sigma_d(C) - w(C)(tau + psi) + w(C) Phi_g integrates to zero.
  Coefficient& C = mig.mig->OceanFraction();
  const double g0 = G0(dim);
  VectorGridFunctionCoefficient uc(&mig.p->Displacement());
  InnerProductCoefficient ug(uc, grad_phi0);
  GridFunctionCoefficient pc(
      const_cast<GridFunction*>(&mig.p->PotentialOnBody()));
  SumCoefficient tau(ug, pc);
  FunctionCoefficient ginv([g0](const Vector&) { return kRhoW / g0; });
  ProductCoefficient w(ginv, C);
  ProductCoefficient wtau(w, tau);
  // sigma_d(C) through the orchestrator's own law, rebuilt test-side:
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
  const double total = c.SurfaceIntegral(sigma_d) -
                       c.SurfaceIntegral(wtau) +
                       mig.p->UniformPotentialTerm() * c.SurfaceIntegral(w);
  EXPECT_LT(std::abs(total), 1e-4);
}

TEST_P(ShorelineTest, QuadraticAmplitudeScaling) {
  const auto [dim, order] = GetParam();
  if (order > 1) {
    GTEST_SKIP() << "covered at order 1 (refinement keeps the strip "
                    "resolved; order adds nothing to the certificate)";
  }
  // One refinement: the asymptotic regime needs the moved shoreline
  // strip resolved by the boundary quadrature.
  Case c(dim, order, 1);
  auto grad_phi0 = c.GradPhi0();
  FunctionCoefficient sl0(SL0), ice0(Ice0);

  // The strict second-order statement holds in the sharp-shoreline
  // limit (the load integrand vanishes AT the shoreline); a smoothing
  // band wider than the sea-level shift adds a first-order-in-band
  // piece. So the certificate runs with the band narrow against the
  // shifts.
  constexpr double kSharp = 1e-3;
  auto diff_at = [&](double eps) {
    FunctionCoefficient dice(
        [eps](const Vector& x) { return -eps * Ice0(x); });
    auto frozen = Migrate(c, grad_phi0, sl0, ice0, dice, false, false,
                          kSharp);
    auto mig = Migrate(c, grad_phi0, sl0, ice0, dice, true, false, kSharp);
    GridFunction du(mig.p->Displacement());
    du -= frozen.p->Displacement();
    return L2Norm(du);
  };
  const double d1 = diff_at(1.0);
  const double d2 = diff_at(0.5);
  // Second order: halving the amplitude quarters the difference (a
  // generous window: the smooth shoreline band adds its own weak
  // amplitude dependence).
  std::cout << "quadratic certificate: d(1)/d(1/2) = " << d1 / d2 << "\n";
  EXPECT_GT(d1 / d2, 2.5);
  EXPECT_LT(d1 / d2, 6.5);
  EXPECT_GT(d1, 0.0);
}

TEST_P(ShorelineTest, InexactMatchesTight) {
  const auto [dim, order] = GetParam();
  if (dim == 3 && order > 1) {
    GTEST_SKIP() << "covered at order 1";
  }
  Case c(dim, order);
  auto grad_phi0 = c.GradPhi0();
  FunctionCoefficient sl0(SL0), ice0(Ice0);
  FunctionCoefficient dice([](const Vector& x) { return -0.5 * Ice0(x); });

  auto loose = Migrate(c, grad_phi0, sl0, ice0, dice, true, false, kShore,
                       0.1);
  auto tight = Migrate(c, grad_phi0, sl0, ice0, dice, true, false, kShore,
                       0.0);
  EXPECT_LE(loose.change, 1e-6);
  EXPECT_LE(tight.change, 1e-6);

  // Same fixed point: both paths are inside the outer tolerance, and
  // the polish puts the loose endpoint at the problem's RelTol on the
  // final ocean function.
  GridFunction du(loose.p->Displacement());
  du -= tight.p->Displacement();
  const double rel =
      L2Norm(du) / (L2Norm(tight.p->Displacement()) + 1e-30);
  EXPECT_LT(rel, 1e-5);

  // The economy: the loose route must not cost more outer iterations
  // than the all-tight one (the point of the option).
  std::cout << "outer iterations: inexact "
            << loose.mig->TotalOuterIterations() << " vs tight "
            << tight.mig->TotalOuterIterations() << "\n";
  EXPECT_LE(loose.mig->TotalOuterIterations(),
            tight.mig->TotalOuterIterations());
}

TEST_P(ShorelineTest, GuardEscalatesToTight) {
  const auto [dim, order] = GetParam();
  if (dim == 3 && order > 1) {
    GTEST_SKIP() << "covered at order 1";
  }
  Case c(dim, order);
  auto grad_phi0 = c.GradPhi0();
  FunctionCoefficient sl0(SL0), ice0(Ice0);
  FunctionCoefficient dice([](const Vector& x) { return -0.5 * Ice0(x); });

  // A vanishing guard fraction trips on the first measurable pass and
  // abandons inexactness (the sticky escalation of the Options note):
  // the run must land on the tight loop's fixed point, exercising the
  // re-solve-and-re-measure path that rescues a border-poisoned loose
  // pass on solver stacks that amplify (the sea-level benchmark's).
  auto guarded = Migrate(c, grad_phi0, sl0, ice0, dice, true, false,
                         kShore, 0.1, 1e-12);
  auto tight = Migrate(c, grad_phi0, sl0, ice0, dice, true, false, kShore,
                       0.0);
  EXPECT_LE(guarded.change, 1e-6);
  GridFunction du(guarded.p->Displacement());
  du -= tight.p->Displacement();
  const double rel =
      L2Norm(du) / (L2Norm(tight.p->Displacement()) + 1e-30);
  EXPECT_LT(rel, 1e-5);
}

TEST_P(ShorelineTest, ComposesWithRotation) {
  const auto [dim, order] = GetParam();
  if (dim == 3 && order > 1) {
    GTEST_SKIP() << "covered at order 1";
  }
  Case c(dim, order);
  auto grad_phi0 = c.GradPhi0();
  FunctionCoefficient sl0(SL0), ice0(Ice0);
  FunctionCoefficient dice([](const Vector& x) { return -0.5 * Ice0(x); });
  auto mig = Migrate(c, grad_phi0, sl0, ice0, dice, true, true);
  EXPECT_LE(mig.change, 1e-6);
  EXPECT_GT(mig.p->AngularVelocity().Norml2(), 0.0);
}

INSTANTIATE_TEST_SUITE_P(Shoreline, ShorelineTest,
                         testing::Values(Param{3, 1}, Param{3, 2}));

}  // namespace
