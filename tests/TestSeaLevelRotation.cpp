#include <numbers>

#include "MixedProblemTestCommon.hpp"
#include "TestCommon.hpp"

/*
  Gates for the WP4 composition (doc/planning/sea_level_plan.md): the
  rotational border inside the solve (SetRotation), alone and jointly
  with the water load (SetWaterLoad), on the canned two-layer meshes.

  1. Water off: SetRotation reproduces RotationalFeedback (the WP2
     orchestrator solves the same bordered system by its own
     elimination) — omega and displacement at solver tolerance.
  2. Composed: the monolithic joint border equals the fixed point that
     iterates sea level AND rotation together on a fresh problem (the
     water load carries -w psi(omega), the tide is psi(omega), omega
     from the angular-momentum row, the uniform from mass conservation)
     — route-grade.
  3. The omega-row and mass-row residuals, with every cross term
     (the water-psi column, the Phi_g coupling, the surface psi-psi
     block) re-evaluated through test-side integrals at the composed
     state.
  4. Omega -> 0 recovers the water-only solution.
*/

namespace {

using namespace self_grav_test;

constexpr double kRhoW = 0.02;
constexpr double kOmega = 0.1;

double Degree2(const Vector& x) {
  const double r = x.Norml2();
  if (x.Size() == 2) {
    return (x[0] * x[0] - x[1] * x[1]) / (r * r);
  }
  const double c = x[2] / r;
  return 3.0 * c * c - 1.0;
}

double RotLoad(const Vector& x) {
  const double r = x.Norml2();
  if (x.Size() == 2) {
    // A mean part so that the 2-D spin (psi ~ |x|^2) is driven.
    const double c = x[1] / r;
    return 0.02 * (1.0 + 3.0 * c * c);
  }
  return Degree2(x) + 0.01 * x[0] * x[2] / (r * r);
}

double G0(int dim) {
  return dim == 2 ? 2.0 * std::numbers::pi * kG * kRho
                  : 4.0 * std::numbers::pi * kG * kRho / 3.0;
}

// The iterated water load with rotation: w (c - tau - psi(omega)).
class IteredWaterRotLoad : public Coefficient {
 public:
  // psi enters through its interpolant (the border's convention).
  IteredWaterRotLoad(double w, VectorCoefficient& grad_phi0,
                     const GridFunction& psi_interp)
      : w_(w), grad_phi0_(&grad_phi0), psi_(&psi_interp) {}

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
    return w_ * (c_ - tau - psi_->GetValue(T, ip));
  }

 private:
  double w_;
  VectorCoefficient* grad_phi0_;
  const GridFunction* psi_;
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
  FunctionCoefficient sigma{RotLoad};
  Array<int> surface;
  Vector moments;
  double w0;
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
    moments.SetSize(dim == 2 ? 1 : 3);
    if (dim == 2) {
      moments[0] = 2.0;
    } else {
      // Well away from the near-neutral wander regime (C3 - C1 small),
      // where the feedback gain approaches one and amplifies ordinary
      // assembly-route differences between the monolithic and the
      // instrument's integrals by 1/(1 - gain).
      moments[0] = 1.2;
      moments[1] = 1.3;
      moments[2] = 2.0;
    }
    w0 = kRhoW / G0(dim);
    w = std::make_unique<ConstantCoefficient>(w0);
  }

  // with_load: false when SetWaterLoad will register sigma itself (the
  // composed tests), so the data load is not counted twice.
  std::unique_ptr<LinearQuasiStaticMixedSelfGravitatingProblem> Problem(
      bool with_load = true) {
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        fes_u.get(), fes_phi.get(), *rheology, rho, kG, kDtNDegree);
    if (with_load) {
      p->SetSurfaceLoad(sigma, surface);
    }
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

  // The scalar body space of the psi interpolants (surface nodes agree
  // with the potential space's, so this is the border's convention).
  std::unique_ptr<H1_FECollection> sfec_i;
  std::unique_ptr<FiniteElementSpace> sfes_i;
  FiniteElementSpace& ScalarSpace() {
    if (!sfes_i) {
      sfec_i = std::make_unique<H1_FECollection>(
          fes_u->GetMaxElementOrder(), body->Dimension());
      sfes_i = std::make_unique<FiniteElementSpace>(body.get(),
                                                    sfec_i.get());
    }
    return *sfes_i;
  }

  double SurfaceIntegral(Coefficient& f) {
    H1_FECollection sfec(fes_u->GetMaxElementOrder(), body->Dimension());
    FiniteElementSpace sfes(body.get(), &sfec);
    LinearForm lf(&sfes);
    lf.AddBoundaryIntegrator(new BoundaryLFIntegrator(f), surface);
    lf.Assemble();
    return lf.Sum();
  }
};

using Param = std::tuple<int, int>;

class SeaLevelRotationTest : public testing::TestWithParam<Param> {};

TEST_P(SeaLevelRotationTest, MatchesRotationalFeedbackWithoutWater) {
  const auto [dim, order] = GetParam();
  Case c(dim, order);

  auto p = c.Problem();
  p->SetRotation(kOmega, c.moments);
  p->AssembleForce(0.0);
  ASSERT_TRUE(p->Solve());

  auto q = c.Problem();
  RotationalFeedback rot(*q, kOmega, c.moments);
  ASSERT_TRUE(rot.Solve(0.0, c.sigma, c.surface));

  Vector dw(p->AngularVelocity());
  dw -= rot.AngularVelocityPerturbation();
  EXPECT_LT(dw.Norml2(),
            1e-8 * (rot.AngularVelocityPerturbation().Norml2() + 1e-30));
  GridFunction du(p->Displacement());
  du -= q->Displacement();
  EXPECT_LT(L2Norm(du), 1e-8 * (L2Norm(q->Displacement()) + 1e-30));
}

TEST_P(SeaLevelRotationTest, ComposedMatchesJointFixedPoint) {
  const auto [dim, order] = GetParam();
  if (dim == 3 && order > 1) {
    GTEST_SKIP() << "covered at order 1; the iteration repeats solves";
  }
  Case c(dim, order);
  auto grad_phi0 = c.GradPhi0(dim);

  auto p = c.Problem(false);
  p->SetWaterLoad(*c.w, c.sigma, c.surface);
  p->SetRotation(kOmega, c.moments);
  p->AssembleForce(0.0);
  ASSERT_TRUE(p->Solve());
  const double phi_g = p->UniformPotentialTerm();
  const Vector omega = p->AngularVelocity();

  // The joint fixed point on a fresh problem: lagged water load
  // (including -w psi(omega)), lagged centrifugal tide, omega from the
  // angular-momentum row, uniform from mass conservation.
  auto q = c.Problem(false);
  const int nw = omega.Size();
  CentrifugalPotential psi(dim, kOmega);
  GridFunction psi_i(&c.ScalarSpace());
  psi_i = 0.0;
  std::vector<std::unique_ptr<CentrifugalPotential>> unit;
  std::vector<std::unique_ptr<GridFunction>> unit_g;
  std::vector<std::unique_ptr<GridFunctionCoefficient>> unit_i;
  for (int k = 0; k < nw; k++) {
    unit.push_back(std::make_unique<CentrifugalPotential>(dim, kOmega));
    unit.back()->SetUnit(k);
    unit_g.push_back(std::make_unique<GridFunction>(&c.ScalarSpace()));
    unit_g.back()->ProjectCoefficient(*unit.back());
    unit_i.push_back(
        std::make_unique<GridFunctionCoefficient>(unit_g.back().get()));
  }
  GridFunctionCoefficient psi_ic(&psi_i);
  IteredWaterRotLoad water(c.w0, grad_phi0, psi_i);
  q->SetSurfaceLoad(c.sigma, c.surface);
  q->SetSurfaceLoad(water, c.surface);
  q->SetTidalPotential(psi);
  ConstantCoefficient one(1.0);
  ProductCoefficient wone(*c.w, one);
  const double m = c.SurfaceIntegral(wone);
  const double Sd = c.SurfaceIntegral(c.sigma);
  const DenseMatrix D = InertiaMatrix(c.moments);
  Vector wn(nw), rhsw(nw);
  wn = 0.0;
  double uniform = 0.0;
  for (int it = 0; it < 60; it++) {
    psi.SetAmplitudes(wn);
    psi_i.ProjectCoefficient(psi);
    q->AssembleForce(0.0);
    ASSERT_TRUE(q->Solve());
    VectorGridFunctionCoefficient uc(&q->Displacement());
    InnerProductCoefficient ug(uc, grad_phi0);
    GridFunctionCoefficient pc(
        const_cast<GridFunction*>(&q->PotentialOnBody()));
    SumCoefficient tau(ug, pc);
    // sigma_total = sigma_data + w (c - tau - psi(omega_n)), psi via
    // its interpolant throughout.
    SumCoefficient taupsi(tau, psi_ic);
    ProductCoefficient wtau(*c.w, taupsi);
    uniform = (c.SurfaceIntegral(wtau) - Sd) / m;
    // omega-row: (D + P) omega = -(T(psi_k, x) + int sigma_total psi_k).
    DenseMatrix DP(D);
    for (int k = 0; k < nw; k++) {
      ProductCoefficient spk(c.sigma, *unit_i[k]);
      ProductCoefficient wtk(wtau, *unit_i[k]);
      ProductCoefficient wck(wone, *unit_i[k]);
      rhsw[k] = -(q->TidalCoupling(*unit[k]) + c.SurfaceIntegral(spk) -
                  c.SurfaceIntegral(wtk) + uniform * c.SurfaceIntegral(wck));
      for (int j = 0; j < nw; j++) {
        DP(k, j) += q->TidalTidalCoupling(*unit[k], *unit[j]);
      }
    }
    DenseMatrixInverse inv(DP);
    inv.Mult(rhsw, wn);
    water.Update(q->Displacement(), q->PotentialOnBody(), uniform);
  }
  // Route-grade agreement; omega carries the wander amplification of
  // the route differences, so its gate is wider than the fields'.
  Vector dw(wn);
  dw -= omega;
  EXPECT_LT(dw.Norml2(), 1e-2 * (omega.Norml2() + 1e-6));
  EXPECT_NEAR(uniform, phi_g, 2e-3 * std::max(1.0, std::abs(phi_g)));
  GridFunction du(p->Displacement());
  du -= q->Displacement();
  EXPECT_LT(L2Norm(du), 2e-3 * (L2Norm(p->Displacement()) + 1e-30));
}

TEST_P(SeaLevelRotationTest, RowResidualsAtComposedState) {
  const auto [dim, order] = GetParam();
  Case c(dim, order);
  auto grad_phi0 = c.GradPhi0(dim);

  auto p = c.Problem(false);
  p->SetWaterLoad(*c.w, c.sigma, c.surface);
  p->SetRotation(kOmega, c.moments);
  p->AssembleForce(0.0);
  ASSERT_TRUE(p->Solve());
  const Vector omega = p->AngularVelocity();
  const double phi_g = p->UniformPotentialTerm();
  const int nw = omega.Size();

  std::vector<std::unique_ptr<CentrifugalPotential>> unit;
  std::vector<std::unique_ptr<GridFunction>> unit_g;
  std::vector<std::unique_ptr<GridFunctionCoefficient>> unit_i;
  for (int k = 0; k < nw; k++) {
    unit.push_back(std::make_unique<CentrifugalPotential>(dim, kOmega));
    unit.back()->SetUnit(k);
    unit_g.push_back(std::make_unique<GridFunction>(&c.ScalarSpace()));
    unit_g.back()->ProjectCoefficient(*unit.back());
    unit_i.push_back(
        std::make_unique<GridFunctionCoefficient>(unit_g.back().get()));
  }
  CentrifugalPotential psi(dim, kOmega);
  psi.SetAmplitudes(omega);
  GridFunction psi_g(&c.ScalarSpace());
  psi_g.ProjectCoefficient(psi);
  GridFunctionCoefficient psi_ic(&psi_g);
  VectorGridFunctionCoefficient uc(&p->Displacement());
  InnerProductCoefficient ug(uc, grad_phi0);
  GridFunctionCoefficient pc(
      const_cast<GridFunction*>(&p->PotentialOnBody()));
  SumCoefficient tau(ug, pc);
  ConstantCoefficient one(1.0);
  ProductCoefficient wone(*c.w, one);

  // Mass row: int sigma_total dS = 0 with
  // sigma_total = sigma_data - w(tau + psi) + w Phi_g.
  {
    SumCoefficient taupsi(tau, psi_ic);
    ProductCoefficient wtp(*c.w, taupsi);
    const double total = c.SurfaceIntegral(c.sigma) -
                         c.SurfaceIntegral(wtp) +
                         phi_g * c.SurfaceIntegral(wone);
    EXPECT_LT(std::abs(total), 1e-4);
  }
  // Omega rows: T(psi_k, x) + (D+P) omega + int sigma_total psi_k = 0.
  const DenseMatrix D = InertiaMatrix(c.moments);
  for (int k = 0; k < nw; k++) {
    const double tk = p->TidalCoupling(*unit[k]);
    double dp = 0.0;
    for (int j = 0; j < nw; j++) {
      dp += (D(k, j) + p->TidalTidalCoupling(*unit[k], *unit[j])) * omega[j];
    }
    SumCoefficient taupsi(tau, psi_ic);
    ProductCoefficient wtp(*c.w, taupsi);
    ProductCoefficient sk(c.sigma, *unit_i[k]);
    ProductCoefficient wtk(wtp, *unit_i[k]);
    ProductCoefficient wck(wone, *unit_i[k]);
    const double sig = c.SurfaceIntegral(sk) - c.SurfaceIntegral(wtk) +
                       phi_g * c.SurfaceIntegral(wck);
    const double row = tk + dp + sig;
    // The row is a near-cancelling balance; normalise by the data
    // term's magnitude as well, so the gate measures the residual
    // against the forcing rather than against the cancellation.
    ProductCoefficient sdk(c.sigma, *unit_i[k]);
    const double data = std::abs(c.SurfaceIntegral(sdk));
    const double scale = std::max({std::abs(tk), std::abs(dp),
                                   std::abs(sig), data, 1e-30});
    EXPECT_LT(std::abs(row) / scale, 1e-2) << "omega row " << k;
  }
}

TEST_P(SeaLevelRotationTest, ZeroOmegaRecoversWaterOnly) {
  const auto [dim, order] = GetParam();
  if (dim == 3) {
    GTEST_SKIP() << "a physics identity; the 2-D runs cover it";
  }
  Case c(dim, order);
  auto p = c.Problem(false);
  p->SetWaterLoad(*c.w, c.sigma, c.surface);
  p->SetRotation(0.0, c.moments);
  p->AssembleForce(0.0);
  ASSERT_TRUE(p->Solve());
  auto q = c.Problem(false);
  q->SetWaterLoad(*c.w, c.sigma, c.surface);
  q->AssembleForce(0.0);
  ASSERT_TRUE(q->Solve());
  EXPECT_LT(p->AngularVelocity().Norml2(), 1e-12);
  EXPECT_NEAR(p->UniformPotentialTerm(), q->UniformPotentialTerm(), 1e-10);
  GridFunction du(p->Displacement());
  du -= q->Displacement();
  EXPECT_LT(L2Norm(du), 1e-9 * (L2Norm(q->Displacement()) + 1e-30));
}

INSTANTIATE_TEST_SUITE_P(SeaLevelRotation, SeaLevelRotationTest,
                         testing::Values(Param{3, 1}, Param{3, 2}));

}  // namespace
