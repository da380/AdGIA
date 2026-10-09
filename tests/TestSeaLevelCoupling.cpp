#include <numbers>

#include "MixedProblemTestCommon.hpp"
#include "TestCommon.hpp"

/*
  Rung-0 tests of the coupled water load (SetWaterLoad: the condensed
  sea-level equation with frozen shoreline; doc/planning/sea_level_plan.md,
  "The WP3 derivation") on the canned two-layer meshes, all-ocean
  (C0 = 1) with a small water density.

  1. Monolithic == fixed point: the pyslfp-style iteration — assumed sea
     level -> water load -> solve -> new sea level, with the uniform term
     from mass conservation — on a fresh problem converges to the
     monolithic solution (displacement, potential, uniform term). The
     central gate: the two routes share nothing but the physics.
  2. The spectral identity: with T the measured tau/g amplitude response
     per unit surface mass (one plain solve), the monolithic degree-2
     amplitude satisfies s = -T sigma / (1 + rho_w T) to solver accuracy.
  3. Mass conservation: int sigma_total dS = 0 at the solution, by
     construction of the border - the discrete certificate.
  4. The uniform term: inert for a zero-mean load (Phi_g ~ 0), active for
     a load with mean, with (3) still holding.
  5. Solver-type invariance: SchurCG and BlockMINRES agree through the
     Sherman-Morrison wrap.
*/

namespace {

using namespace self_grav_test;

constexpr double kRhoW = 0.02;

// Degree-2, zero-mean surface pattern (2-D: cos 2theta; 3-D: the zonal
// 3c^2 - 1), optionally with a mean part.
double Degree2(const Vector& x) {
  const double r = x.Norml2();
  if (x.Size() == 2) {
    return (x[0] * x[0] - x[1] * x[1]) / (r * r);
  }
  const double c = x[2] / r;
  return 3.0 * c * c - 1.0;
}

double G0(int dim) {
  return dim == 2 ? 2.0 * std::numbers::pi * kG * kRho
                  : 4.0 * std::numbers::pi * kG * kRho / 3.0;
}

// The iterated water load of the fixed-point instrument:
// sigma_w = -w (u.grad Phi0 + phi) + w c, evaluated from stored copies
// of the last solution's body fields.
class IteredWaterLoad : public Coefficient {
 public:
  IteredWaterLoad(Coefficient& w, VectorCoefficient& grad_phi0)
      : w_(&w), grad_phi0_(&grad_phi0) {}

  void Update(const GridFunction& u, const GridFunction& phi_body,
              double uniform) {
    u_ = std::make_unique<GridFunction>(u);
    phi_ = std::make_unique<GridFunction>(phi_body);
    c_ = uniform;
  }

  double Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    if (!u_) {
      return 0.0;
    }
    Vector uv, g;
    u_->GetVectorValue(T, ip, uv);
    grad_phi0_->Eval(g, T, ip);
    const double tau = (uv * g) + phi_->GetValue(T, ip);
    return w_->Eval(T, ip) * (c_ - tau);
  }

 private:
  Coefficient* w_;
  VectorCoefficient* grad_phi0_;
  std::unique_ptr<GridFunction> u_, phi_;
  double c_ = 0.0;
};

struct Case {
  std::unique_ptr<Mesh> parent;
  std::unique_ptr<SubMesh> body;
  std::unique_ptr<H1_FECollection> fec;
  std::unique_ptr<FiniteElementSpace> fes_u, fes_phi;
  ConstantCoefficient kappa{kKappa}, mu{kMu}, rho{kRho};
  std::unique_ptr<IsotropicElasticRheology> rheology;
  Array<int> surface;
  double w0;  // rho_w / g on the unit surface
  std::unique_ptr<ConstantCoefficient> w;

  explicit Case(int dim, int order) {
    parent = std::make_unique<Mesh>(MeshFile(dim).c_str(), 1, 1);
    body = std::make_unique<SubMesh>(
        SubMesh::CreateFromDomain(*parent, BodyMarker(*parent)));
    fec = std::make_unique<H1_FECollection>(order, dim);
    fes_u = std::make_unique<FiniteElementSpace>(body.get(), fec.get(), dim);
    fes_phi = std::make_unique<FiniteElementSpace>(parent.get(), fec.get());
    rheology = std::make_unique<IsotropicElasticRheology>(dim, kappa, mu);
    surface = SurfaceMarker(*body);
    w0 = kRhoW / G0(dim);
    w = std::make_unique<ConstantCoefficient>(w0);
  }

  std::unique_ptr<LinearQuasiStaticMixedSelfGravitatingProblem> Problem() {
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        fes_u.get(), fes_phi.get(), *rheology, rho, kG, kDtNDegree);
    p->SetRelTol(1e-12);
    return p;
  }

  VectorFunctionCoefficient GradPhi0(int dim) {
    const double g0 = G0(dim);
    return VectorFunctionCoefficient(dim, [g0](const Vector& x, Vector& v) {
      v = x;
      v *= g0;
    });
  }

  // int f dS over the body surface.
  double SurfaceIntegral(Coefficient& f) {
    H1_FECollection sfec(fes_u->GetMaxElementOrder(),
                         body->Dimension());
    FiniteElementSpace sfes(body.get(), &sfec);
    LinearForm lf(&sfes);
    lf.AddBoundaryIntegrator(new BoundaryLFIntegrator(f), surface);
    lf.Assemble();
    return lf.Sum();
  }
};

// Amplitude of the Degree2 pattern in a surface field of `sea`.
double PatternAmplitude(SeaLevelOperator& sea) {
  FunctionCoefficient pat(Degree2);
  GridFunction p(&sea.SurfaceSpace());
  p.ProjectCoefficient(pat);
  GridFunctionCoefficient fc(&sea.SeaLevelChangeField());
  GridFunctionCoefficient pc(&p);
  ProductCoefficient fp(fc, pc), pp(pc, pc);
  return sea.SurfaceIntegral(fp) / sea.SurfaceIntegral(pp);
}

using Param = std::tuple<int, int>;

class SeaLevelCouplingTest : public testing::TestWithParam<Param> {};

TEST_P(SeaLevelCouplingTest, MonolithicMatchesFixedPoint) {
  const auto [dim, order] = GetParam();
  Case c(dim, order);
  auto grad_phi0 = c.GradPhi0(dim);
  FunctionCoefficient sigma_data(Degree2);

  // Monolithic.
  auto p = c.Problem();
  p->SetWaterLoad(*c.w, sigma_data, c.surface);
  p->AssembleForce(0.0);
  ASSERT_TRUE(p->Solve());
  const double phi_g = p->UniformPotentialTerm();

  // Fixed point on a fresh problem: load sigma_data + the iterated
  // water load; uniform from mass conservation each pass.
  auto q = c.Problem();
  IteredWaterLoad water(*c.w, grad_phi0);
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
    // c = (int w tau dS - int sigma_data dS) / m with tau from the
    // current state.
    VectorGridFunctionCoefficient uc(&q->Displacement());
    InnerProductCoefficient ug(uc, grad_phi0);
    GridFunctionCoefficient pc(
        const_cast<GridFunction*>(&q->PotentialOnBody()));
    SumCoefficient tau(ug, pc);
    ProductCoefficient wtau(*c.w, tau);
    uniform = (c.SurfaceIntegral(wtau) - Sd) / m;
    water.Update(q->Displacement(), q->PotentialOnBody(), uniform);
  }
  // The two routes differ by the operator's normal-projected q_g
  // against the instrument's full grad Phi0 in the water load — a
  // curved-geometry discretisation difference, not an algebra one.
  EXPECT_NEAR(uniform, phi_g,
              2e-4 * std::max(1.0, std::abs(phi_g)));
  GridFunction du(p->Displacement());
  du -= q->Displacement();
  EXPECT_LT(L2Norm(du), 1e-3 * (L2Norm(p->Displacement()) + 1e-30));
}

TEST_P(SeaLevelCouplingTest, SpectralIdentityAndMass) {
  const auto [dim, order] = GetParam();
  Case c(dim, order);
  auto grad_phi0 = c.GradPhi0(dim);
  FunctionCoefficient sigma_data(Degree2);

  // T: tau/g amplitude per unit degree-2 surface mass (plain solve).
  auto pt = c.Problem();
  pt->SetSurfaceLoad(sigma_data, c.surface);
  pt->AssembleForce(0.0);
  ASSERT_TRUE(pt->Solve());
  SeaLevelOperator sea_t(*c.fes_u, c.surface);
  sea_t.SeaLevelChangeFrom(pt->Displacement(), grad_phi0,
                           pt->Potential(), 0.0);
  const double T = -PatternAmplitude(sea_t);  // tau/g amplitude per sigma

  // Monolithic degree-2 amplitude.
  auto p = c.Problem();
  p->SetWaterLoad(*c.w, sigma_data, c.surface);
  p->AssembleForce(0.0);
  ASSERT_TRUE(p->Solve());
  SeaLevelOperator sea(*c.fes_u, c.surface);
  sea.SeaLevelChangeFrom(p->Displacement(), grad_phi0, p->Potential(),
                         p->UniformPotentialTerm());
  const double s = PatternAmplitude(sea);
  // s = -T sigma_hat / (1 + rho_w T), sigma_hat = 1, to the
  // geometry-grade difference of the q_g-projected operator route
  // against the full-gradient load route (the exact-identity gate is
  // MonolithicMatchesFixedPoint).
  // Measured route differences: 1.2e-6 at 2-D order 2, ~1e-4 at 2-D
  // order 1, ~1.1e-3 on the coarse faceted 3-D order-1 surface.
  EXPECT_NEAR(s, -T / (1.0 + kRhoW * T), 2e-3 * std::abs(T));
  // Zero-mean load, all ocean: the uniform term carries only the
  // pattern's quadrature-level mean on the curved boundary.
  EXPECT_LT(std::abs(p->UniformPotentialTerm()), 2e-3);

  // Mass conservation of the total load at the solution.
  VectorGridFunctionCoefficient uc(&p->Displacement());
  InnerProductCoefficient ug(uc, grad_phi0);
  GridFunctionCoefficient pc(
      const_cast<GridFunction*>(&p->PotentialOnBody()));
  SumCoefficient tau(ug, pc);
  ProductCoefficient wtau(*c.w, tau);
  const double total = c.SurfaceIntegral(sigma_data) -
                       c.SurfaceIntegral(wtau) +
                       p->UniformPotentialTerm() * [&] {
                         ConstantCoefficient one(1.0);
                         ProductCoefficient wone(*c.w, one);
                         return c.SurfaceIntegral(wone);
                       }();
  // The certificate re-integrates w tau through its own quadrature
  // route; the assembled identity is exact, this one is route-grade.
  EXPECT_LT(std::abs(total), 1e-5);
}

TEST_P(SeaLevelCouplingTest, UniformTermActivatesWithMeanLoad) {
  const auto [dim, order] = GetParam();
  Case c(dim, order);
  auto grad_phi0 = c.GradPhi0(dim);
  // A load with a mean: constant + degree 2.
  FunctionCoefficient sigma_data(
      [](const Vector& x) { return 0.5 + Degree2(x); });

  auto p = c.Problem();
  p->SetWaterLoad(*c.w, sigma_data, c.surface);
  p->AssembleForce(0.0);
  ASSERT_TRUE(p->Solve());
  EXPECT_GT(std::abs(p->UniformPotentialTerm()), 1e-3);

  VectorGridFunctionCoefficient uc(&p->Displacement());
  InnerProductCoefficient ug(uc, grad_phi0);
  GridFunctionCoefficient pc(
      const_cast<GridFunction*>(&p->PotentialOnBody()));
  SumCoefficient tau(ug, pc);
  ProductCoefficient wtau(*c.w, tau);
  ConstantCoefficient one(1.0);
  ProductCoefficient wone(*c.w, one);
  const double total = c.SurfaceIntegral(sigma_data) -
                       c.SurfaceIntegral(wtau) +
                       p->UniformPotentialTerm() * c.SurfaceIntegral(wone);
  EXPECT_LT(std::abs(total),
            1e-6 * std::max(1.0, std::abs(c.SurfaceIntegral(sigma_data))));
}

TEST_P(SeaLevelCouplingTest, SolverTypesAgree) {
  const auto [dim, order] = GetParam();
  if (dim == 3) {
    GTEST_SKIP() << "2-D covers the solver axis";
  }
  Case c(dim, order);
  FunctionCoefficient sigma_data(Degree2);
  auto a = c.Problem();
  a->SetWaterLoad(*c.w, sigma_data, c.surface);
  a->SetSolverType(
      LinearQuasiStaticMixedSelfGravitatingProblem::SolverType::SchurCG);
  a->AssembleForce(0.0);
  ASSERT_TRUE(a->Solve());
  auto b = c.Problem();
  b->SetWaterLoad(*c.w, sigma_data, c.surface);
  b->SetSolverType(
      LinearQuasiStaticMixedSelfGravitatingProblem::SolverType::BlockMINRES);
  b->AssembleForce(0.0);
  ASSERT_TRUE(b->Solve());
  EXPECT_NEAR(a->UniformPotentialTerm(), b->UniformPotentialTerm(), 1e-9);
  GridFunction d(a->Displacement());
  d -= b->Displacement();
  EXPECT_LT(L2Norm(d), 1e-8 * (L2Norm(a->Displacement()) + 1e-30));
}

// A land case: the ocean is the southern half (C0 = indicator), the
// shoreline cutting through boundary elements as coefficient data. No
// closed form; the gate is monolithic == fixed point, land included.
TEST(SeaLevelCouplingLand, MonolithicMatchesFixedPoint) {
  Case c(2, 2);
  auto grad_phi0 = c.GradPhi0(2);
  FunctionCoefficient sigma_data(Degree2);
  const double w0 = c.w0;
  FunctionCoefficient w_land(
      [w0](const Vector& x) { return x[1] < 0.0 ? w0 : 0.0; });

  auto p = c.Problem();
  p->SetWaterLoad(w_land, sigma_data, c.surface);
  p->AssembleForce(0.0);
  ASSERT_TRUE(p->Solve());
  const double phi_g = p->UniformPotentialTerm();

  auto q = c.Problem();
  IteredWaterLoad water(w_land, grad_phi0);
  q->SetSurfaceLoad(sigma_data, c.surface);
  q->SetSurfaceLoad(water, c.surface);
  ConstantCoefficient one(1.0);
  ProductCoefficient wone(w_land, one);
  const double m = c.SurfaceIntegral(wone);
  const double Sd = c.SurfaceIntegral(sigma_data);
  double uniform = 0.0;
  for (int it = 0; it < 20; it++) {
    q->AssembleForce(0.0);
    ASSERT_TRUE(q->Solve());
    VectorGridFunctionCoefficient uc(&q->Displacement());
    InnerProductCoefficient ug(uc, grad_phi0);
    GridFunctionCoefficient pc(
        const_cast<GridFunction*>(&q->PotentialOnBody()));
    SumCoefficient tau(ug, pc);
    ProductCoefficient wtau(w_land, tau);
    uniform = (c.SurfaceIntegral(wtau) - Sd) / m;
    water.Update(q->Displacement(), q->PotentialOnBody(), uniform);
  }
  EXPECT_NEAR(uniform, phi_g, 2e-4 * std::max(1.0, std::abs(phi_g)));
  GridFunction du(p->Displacement());
  du -= q->Displacement();
  EXPECT_LT(L2Norm(du), 1e-3 * (L2Norm(p->Displacement()) + 1e-30));
  // With land the uniform term is genuinely active.
  EXPECT_GT(std::abs(phi_g), 1e-4);
}

// The water load over a Dahlen fluid core: the same monolithic ==
// fixed-point gate on the three-layer model (solid inner core, fluid
// outer core, mantle). The instrument pairs with the class's own
// background gravity, as the operator does; the water boundary blocks,
// the border and the S-M solves must all compose with the fluid-region
// machinery (paper Appendix A: fluid regions leave the form unchanged).
TEST(SeaLevelCouplingFluid, MonolithicMatchesFixedPoint) {
  const int dim = 2, order = 2;
  Mesh parent(ThreeLayerMeshFile(dim).c_str(), 1, 1);
  Array<int> attrs({1, 3});
  SubMesh solid(SubMesh::CreateFromDomain(parent, attrs));
  H1_FECollection fec(order, dim);
  FiniteElementSpace fes_u(&solid, &fec, dim), fes_phi(&parent, &fec);
  ConstantCoefficient kappa(kKappa), mu(kMu);
  FunctionCoefficient rho_s(SolidDensity), rho_f(FluidDensity);
  IsotropicElasticRheology rheology(dim, kappa, mu);
  std::vector<FluidRegion> fluids;
  fluids.push_back(OuterCore(solid, rho_f));
  const Array<int> surface = SurfaceMarker(solid);
  FunctionCoefficient sigma_data(Degree2);
  ConstantCoefficient w(0.05);
  Array<int> inner_core({1});

  auto make = [&] {
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        &fes_u, &fes_phi, rheology, rho_s, kG, kDtNDegree, nullptr, fluids);
    p->AddRegionRotations(inner_core);
    p->SetRelTol(1e-12);
    return p;
  };
  auto surface_int = [&](Coefficient& f) {
    H1_FECollection sfec(order, dim);
    FiniteElementSpace sfes(&solid, &sfec);
    LinearForm lf(&sfes);
    lf.AddBoundaryIntegrator(new BoundaryLFIntegrator(f),
                             const_cast<Array<int>&>(surface));
    lf.Assemble();
    return lf.Sum();
  };

  auto p = make();
  p->SetWaterLoad(w, sigma_data, surface);
  p->AssembleForce(0.0);
  ASSERT_TRUE(p->Solve());
  const double phi_g = p->UniformPotentialTerm();

  auto q = make();
  IteredWaterLoad water(w, q->BackgroundGravity());
  q->SetSurfaceLoad(sigma_data, surface);
  q->SetSurfaceLoad(water, surface);
  ConstantCoefficient one(1.0);
  ProductCoefficient wone(w, one);
  const double m = surface_int(wone);
  const double Sd = surface_int(sigma_data);
  double uniform = 0.0;
  for (int it = 0; it < 20; it++) {
    q->AssembleForce(0.0);
    ASSERT_TRUE(q->Solve());
    VectorGridFunctionCoefficient uc(&q->Displacement());
    InnerProductCoefficient ug(uc, q->BackgroundGravity());
    GridFunctionCoefficient pc(
        const_cast<GridFunction*>(&q->PotentialOnBody()));
    SumCoefficient tau(ug, pc);
    ProductCoefficient wtau(w, tau);
    uniform = (surface_int(wtau) - Sd) / m;
    water.Update(q->Displacement(), q->PotentialOnBody(), uniform);
  }
  // Route-grade agreement (the operator's normal-projected q_g against
  // the instrument's full gradient), as in the solid gates.
  EXPECT_NEAR(uniform, phi_g, 1e-5 * std::max(1.0, std::abs(phi_g)));
  GridFunction du(p->Displacement());
  du -= q->Displacement();
  EXPECT_LT(L2Norm(du), 1e-5 * (L2Norm(p->Displacement()) + 1e-30));
}

INSTANTIATE_TEST_SUITE_P(SeaLevelCoupling, SeaLevelCouplingTest,
                         testing::Values(Param{2, 1}, Param{2, 2},
                                         Param{3, 1}));

}  // namespace
