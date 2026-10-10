#include <numbers>

#include "MixedProblemTestCommon.hpp"
#include "TestCommon.hpp"

/*
  Gates for the viscoelastic composition of the sea-level machinery
  (WP6 of doc/planning/sea_level_plan.md) on the canned two-layer
  meshes, with the fingerprint-style analytical state and frozen
  shorelines. The composition is by design: SetWaterLoad()'s
  displacement block rides the registered stiffness integrators, so
  SetRelaxationWeights() reassembles it with each effective modulus,
  while the potential block, the coupling and the Phi_g border are
  modulus-independent; every stepper solve routes through the bordered
  SolveLinearSystem().

  1. The instantaneous (t = 0+) response of a Maxwell body with the
     water load equals the purely elastic problem with the unrelaxed
     modulus and the same water load, at solver grade.
  2. Over a Heaviside loading history the mass-conservation identity
     of the Phi_g border holds at every step, re-evaluated through
     test-side surface integrals (route grade), and the body visibly
     relaxes.
  3. The full stack composes: Maxwell + water + rotation steps run,
     with mass held and a finite angular velocity.
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
constexpr double kTau = 1.0;  // Maxwell time, problem units
constexpr double kMelt = 0.5;

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

double MeltLoad(const Vector& x) {
  return -(1.0 - Fraction0(x)) * kRhoI * kMelt * Ice0(x);
}

struct Case {
  std::unique_ptr<Mesh> parent;
  std::unique_ptr<SubMesh> body;
  std::unique_ptr<H1_FECollection> fec;
  std::unique_ptr<FiniteElementSpace> fes_u, fes_phi;
  ConstantCoefficient kappa{kKappa}, mu{kMu}, rho{kRho}, tau{kTau};
  std::unique_ptr<IsotropicElasticRheology> elastic;
  std::unique_ptr<IsotropicMaxwellRheology> maxwell;
  Array<int> surface;
  int dim;

  explicit Case(int d, int order) : dim(d) {
    parent = std::make_unique<Mesh>(MeshFile(dim).c_str(), 1, 1);
    body = std::make_unique<SubMesh>(
        SubMesh::CreateFromDomain(*parent, BodyMarker(*parent)));
    fec = std::make_unique<H1_FECollection>(order, dim);
    fes_u = std::make_unique<FiniteElementSpace>(body.get(), fec.get(), dim);
    fes_phi = std::make_unique<FiniteElementSpace>(parent.get(), fec.get());
    elastic = std::make_unique<IsotropicElasticRheology>(dim, kappa, mu);
    maxwell = std::make_unique<IsotropicMaxwellRheology>(
        IsotropicMaxwellRheology::Maxwell(dim, kappa, mu, tau));
    surface = SurfaceMarker(*body);
  }

  std::unique_ptr<LinearQuasiStaticMixedSelfGravitatingProblem> Problem(
      bool viscous) {
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        fes_u.get(), fes_phi.get(),
        viscous ? static_cast<Rheology&>(*maxwell)
                : static_cast<Rheology&>(*elastic),
        rho, kG, kDtNDegree);
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

// The water-load coefficients of the frozen state (pointwise
// g = g0 |x|, the convention of the sea-level tests).
struct WaterLoad {
  FunctionCoefficient w, sigma_d;
  explicit WaterLoad(int dim)
      : w([g0 = G0(dim)](const Vector& x) {
          return kRhoW * Fraction0(x) / (g0 * x.Norml2());
        }),
        sigma_d(MeltLoad) {}
};

// The mass-conservation identity of the Phi_g border at the problem's
// current fields: int sigma_d - int w (u.grad Phi0 + phi) +
// Phi_g int w = 0.
double MassResidual(Case& c,
                    LinearQuasiStaticMixedSelfGravitatingProblem& p,
                    VectorCoefficient& grad_phi0, WaterLoad& load) {
  VectorGridFunctionCoefficient uc(&p.Displacement());
  InnerProductCoefficient ug(uc, grad_phi0);
  GridFunctionCoefficient pc(
      const_cast<GridFunction*>(&p.PotentialOnBody()));
  SumCoefficient tau_field(ug, pc);
  ProductCoefficient wtau(load.w, tau_field);
  return c.SurfaceIntegral(load.sigma_d) - c.SurfaceIntegral(wtau) +
         p.UniformPotentialTerm() * c.SurfaceIntegral(load.w);
}

using Param = std::tuple<int, int>;

class SeaLevelViscoTest : public testing::TestWithParam<Param> {};

TEST_P(SeaLevelViscoTest, InstantaneousMatchesElastic) {
  const auto [dim, order] = GetParam();
  Case c(dim, order);
  auto grad_phi0 = c.GradPhi0();
  WaterLoad load(dim);

  auto pe = c.Problem(false);
  pe->SetWaterLoad(load.w, load.sigma_d, c.surface);
  pe->AssembleForce(0.0);
  ASSERT_TRUE(pe->Solve());

  auto pv = c.Problem(true);
  pv->SetWaterLoad(load.w, load.sigma_d, c.surface);
  ViscoelasticOperator visco(*pv);
  Vector m(visco.Height());
  m = 0.0;
  ASSERT_TRUE(visco.SolveElastic(m, 0.0));

  GridFunction du(pv->Displacement());
  du -= pe->Displacement();
  const double rel =
      L2Norm(du) / (L2Norm(pe->Displacement()) + 1e-30);
  EXPECT_LT(rel, 1e-9);
  EXPECT_NEAR(pv->UniformPotentialTerm(), pe->UniformPotentialTerm(),
              1e-10 * std::abs(pe->UniformPotentialTerm()) + 1e-14);
}

TEST_P(SeaLevelViscoTest, ConservesMassOverHistory) {
  const auto [dim, order] = GetParam();
  if (dim == 3 && order > 1) {
    GTEST_SKIP() << "covered at order 1";
  }
  Case c(dim, order);
  auto grad_phi0 = c.GradPhi0();
  WaterLoad load(dim);

  auto p = c.Problem(true);
  p->SetWaterLoad(load.w, load.sigma_d, c.surface);
  ViscoelasticOperator visco(*p);
  ExponentialTrapezoidSolver ode;
  ode.Init(visco);
  Vector m(visco.Height());
  m = 0.0;
  double t = 0.0;
  ASSERT_TRUE(visco.SolveElastic(m, t));
  const double u0 = L2Norm(p->Displacement());
  // Route-grade re-integration (pointwise coefficients against the
  // assembled identity): ~1e-5 at order 1, ~1e-6 at order 2 (the WP3
  // hierarchy); it grows mildly with the relaxing fields' amplitude.
  const double mass_tol = order == 1 ? 1e-4 : 2e-6;
  EXPECT_LT(std::abs(MassResidual(c, *p, grad_phi0, load)), mass_tol);

  const int n = 8;
  double dt = 2.0 * kTau / n;
  for (int k = 0; k < n; k++) {
    ode.Step(m, t, dt);
    ASSERT_TRUE(visco.SolveElastic(m, t));
    // The border's mass row holds at every step of the history
    // (route-grade re-integration; the assembled identity is exact).
    EXPECT_LT(std::abs(MassResidual(c, *p, grad_phi0, load)), mass_tol);
  }
  // The body relaxed: the Maxwell mantle keeps deforming after the
  // elastic response (the stepping genuinely engaged).
  const double u1 = L2Norm(p->Displacement());
  EXPECT_GT(std::abs(u1 - u0) / u0, 1e-2);
}

TEST_P(SeaLevelViscoTest, FullStackComposes) {
  const auto [dim, order] = GetParam();
  if (order > 1) {
    GTEST_SKIP() << "covered at order 1";
  }
  Case c(dim, order);
  auto grad_phi0 = c.GradPhi0();
  WaterLoad load(dim);

  auto p = c.Problem(true);
  p->SetWaterLoad(load.w, load.sigma_d, c.surface);
  Vector moments(dim == 2 ? 1 : 3);
  if (dim == 2) {
    moments[0] = 2.0;
  } else {
    moments[0] = 1.2;
    moments[1] = 1.3;
    moments[2] = 2.0;
  }
  p->SetRotation(0.1, moments);
  ViscoelasticOperator visco(*p);
  ExponentialTrapezoidSolver ode;
  ode.Init(visco);
  Vector m(visco.Height());
  m = 0.0;
  double t = 0.0;
  ASSERT_TRUE(visco.SolveElastic(m, t));
  double dt = 0.5 * kTau;
  for (int k = 0; k < 2; k++) {
    ode.Step(m, t, dt);
  }
  ASSERT_TRUE(visco.SolveElastic(m, t));
  // 3-D order 1 sits at the faceted-sphere route grade (the WP3
  // hierarchy: ~1e-3 there, ~1e-4 in 2-D).
  EXPECT_LT(std::abs(MassResidual(c, *p, grad_phi0, load)),
            dim == 3 ? 2e-3 : 1e-4);
  const double om = p->AngularVelocity().Norml2();
  EXPECT_TRUE(std::isfinite(om));
  EXPECT_GT(om, 0.0);
}

INSTANTIATE_TEST_SUITE_P(SeaLevelViscoelastic, SeaLevelViscoTest,
                         testing::Values(Param{2, 1}, Param{2, 2},
                                         Param{3, 1}));

}  // namespace
