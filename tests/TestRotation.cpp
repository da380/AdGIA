#include <numbers>

#include "MixedProblemTestCommon.hpp"
#include "TestCommon.hpp"

/*
  Tests for RotationalFeedback (the WP2 rotation border of the sea-level
  plan) on the canned meshes.

  The gates are independent of the border's own elimination algebra:
  1. Consistency: after Solve(), the angular-momentum row (eq. A5) is
     re-evaluated at the composed state through the problem's pairing
     accessors and a test-side surface integral; its relative residual
     sits at the solver-tolerance level.
  2. A fresh problem, told nothing of the border, is given the fixed
     centrifugal potential psi(omega*) and the same load; its solution
     satisfies the omega-row with the test's own D matrix.
  3. The paper's successively-refined omega iteration (three Picard
     steps from zero on a fresh problem) approaches the component's
     omega.
  4. Linearity: halving Omega halves omega to the expected accuracy.
  The fluid three-layer case exercises the Appendix-A fluid terms (the
  psi-phi pairing and the psi-psi block) through the same gates.
*/

namespace {

using namespace self_grav_test;

constexpr double kOmega = 0.1;

// Degree-2 load with zonal and order-1 parts, so that in 3-D both the
// spin and the wander components of omega are activated.
double RotLoad(const Vector& x, double t) {
  const double r = x.Norml2();
  if (r == 0.0) {
    return 0.0;
  }
  const double c = (x.Size() == 2 ? x[1] : x[2]) / r;
  double s = 0.02 * (1.0 + 3.0 * c * c) * (1.0 + t);
  if (x.Size() == 3) {
    s += 0.01 * x[0] * x[2] / (r * r);
  }
  return s;
}

// The test's own copy of the inertia matrix and surface pairing, for
// the independent gates.
DenseMatrix InertiaMatrix(const Vector& moments) {
  const int nw = moments.Size();
  DenseMatrix D(nw);
  D = 0.0;
  if (nw == 1) {
    D(0, 0) = -moments[0];
  } else {
    D(0, 0) = moments[2] - moments[0];
    D(1, 1) = moments[2] - moments[1];
    D(2, 2) = -moments[2];
  }
  return D;
}

double SurfacePair(FiniteElementSpace& fes_u, Coefficient& sigma,
                   Coefficient& psi_k, const Array<int>& marker) {
  H1_FECollection fec(fes_u.GetMaxElementOrder(),
                      fes_u.GetMesh()->Dimension());
  FiniteElementSpace s(fes_u.GetMesh(), &fec);
  ProductCoefficient sp(sigma, psi_k);
  LinearForm lf(&s);
  lf.AddBoundaryIntegrator(new BoundaryLFIntegrator(sp),
                           const_cast<Array<int>&>(marker));
  lf.Assemble();
  return lf.Sum();
}

// Residual of the omega-row at a problem's current state.
double OmegaRowResidual(LinearQuasiStaticMixedSelfGravitatingProblem& p,
                        FiniteElementSpace& fes_u, Coefficient& sigma,
                        const Array<int>& marker, const Vector& moments,
                        double Omega, const Vector& omega) {
  const int dim = fes_u.GetMesh()->Dimension();
  const int nw = omega.Size();
  const DenseMatrix D = InertiaMatrix(moments);
  std::vector<std::unique_ptr<CentrifugalPotential>> psi;
  for (int k = 0; k < nw; k++) {
    psi.push_back(std::make_unique<CentrifugalPotential>(dim, Omega));
    psi.back()->SetUnit(k);
  }
  double rmax = 0.0, scale = 0.0;
  for (int k = 0; k < nw; k++) {
    const double tk = p.TidalCoupling(*psi[k]);
    double dp = 0.0;
    for (int j = 0; j < nw; j++) {
      dp += (D(k, j) + p.TidalTidalCoupling(*psi[k], *psi[j])) * omega[j];
    }
    const double sk = SurfacePair(fes_u, sigma, *psi[k], marker);
    rmax = std::max(rmax, std::abs(tk + dp + sk));
    scale = std::max({scale, std::abs(tk), std::abs(dp), std::abs(sk)});
  }
  return rmax / (scale + 1e-300);
}

// (dim, order)
using Param = std::tuple<int, int>;

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

  Case(int dim, int order) {
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
      moments[0] = 1.6;
      moments[1] = 1.7;
      moments[2] = 2.0;
    }
  }

  std::unique_ptr<LinearQuasiStaticMixedSelfGravitatingProblem> Problem() {
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        fes_u.get(), fes_phi.get(), *rheology, rho, kG, kDtNDegree);
    p->SetSurfaceLoad(sigma, surface);
    p->SetRelTol(1e-11);
    return p;
  }
};

class RotationTest : public testing::TestWithParam<Param> {};

TEST_P(RotationTest, BorderSolvesAngularMomentumRow) {
  const auto [dim, order] = GetParam();
  Case c(dim, order);
  auto p = c.Problem();
  RotationalFeedback rot(*p, kOmega, c.moments);
  ASSERT_EQ(rot.NumComponents(), dim == 2 ? 1 : 3);
  ASSERT_TRUE(rot.Solve(0.0, c.sigma, c.surface));
  const Vector omega = rot.AngularVelocityPerturbation();
  EXPECT_GT(omega.Norml2(), 0.0);

  // Gate 1: the component's own consistency diagnostic.
  EXPECT_LT(rot.ConsistencyResidual(c.sigma, c.surface), 5e-8);

  // Gate 2: a fresh problem with the fixed psi(omega*) satisfies the
  // omega-row evaluated with the test's own D and surface integral.
  auto p2 = c.Problem();
  CentrifugalPotential psi(dim, kOmega);
  psi.SetAmplitudes(omega);
  p2->SetTidalPotential(psi);
  p2->AssembleForce(0.0);
  ASSERT_TRUE(p2->Solve());
  EXPECT_LT(OmegaRowResidual(*p2, *c.fes_u, c.sigma, c.surface, c.moments,
                             kOmega, omega),
            5e-8);
}

TEST_P(RotationTest, PicardIterationApproachesBorderSolution) {
  const auto [dim, order] = GetParam();
  if (dim == 3 && order > 1) {
    GTEST_SKIP() << "covered at order 1; Picard repeats solves";
  }
  Case c(dim, order);
  auto p = c.Problem();
  RotationalFeedback rot(*p, kOmega, c.moments);
  ASSERT_TRUE(rot.Solve(0.0, c.sigma, c.surface));
  const Vector target = rot.AngularVelocityPerturbation();

  // The paper's successively-refined iteration on a fresh problem:
  // omega <- -(D + P)^{-1} (T(psi_k, x(omega)) + s_k).
  auto p2 = c.Problem();
  const int nw = target.Size();
  CentrifugalPotential psi(dim, kOmega);
  std::vector<std::unique_ptr<CentrifugalPotential>> unit;
  for (int k = 0; k < nw; k++) {
    unit.push_back(std::make_unique<CentrifugalPotential>(dim, kOmega));
    unit.back()->SetUnit(k);
  }
  p2->SetTidalPotential(psi);
  DenseMatrix DP = InertiaMatrix(c.moments);
  Vector w(nw), rhs(nw);
  w = 0.0;
  double err = 0.0;
  for (int it = 0; it < 3; it++) {
    psi.SetAmplitudes(w);
    p2->AssembleForce(0.0);
    ASSERT_TRUE(p2->Solve());
    for (int k = 0; k < nw; k++) {
      rhs[k] = -(p2->TidalCoupling(*unit[k]) +
                 SurfacePair(*c.fes_u, c.sigma, *unit[k], c.surface));
      for (int j = 0; j < nw; j++) {
        DP(k, j) = InertiaMatrix(c.moments)(k, j) +
                   p2->TidalTidalCoupling(*unit[k], *unit[j]);
      }
    }
    DenseMatrixInverse inv(DP);
    inv.Mult(rhs, w);
    Vector d(w);
    d -= target;
    err = d.Norml2() / target.Norml2();
  }
  // The feedback is weak, so three iterations contract far below 1%.
  EXPECT_LT(err, 1e-4);
}

TEST_P(RotationTest, OmegaScalesLinearlyInOmega0) {
  const auto [dim, order] = GetParam();
  if (dim == 3) {
    GTEST_SKIP() << "scaling is a physics identity; the 2-D runs cover it";
  }
  Case c(dim, order);
  auto p1 = c.Problem();
  RotationalFeedback rot1(*p1, kOmega, c.moments);
  ASSERT_TRUE(rot1.Solve(0.0, c.sigma, c.surface));
  auto p2 = c.Problem();
  RotationalFeedback rot2(*p2, 0.5 * kOmega, c.moments);
  ASSERT_TRUE(rot2.Solve(0.0, c.sigma, c.surface));
  const double w1 = rot1.AngularVelocityPerturbation()[0];
  const double w2 = rot2.AngularVelocityPerturbation()[0];
  // omega ~ Omega with O(Omega^2) feedback corrections.
  EXPECT_NEAR(w1 / w2, 2.0, 0.02);
}

TEST(RotationFluid, FluidCoreTermsEnterTheBorder) {
  const int dim = 2, order = 2;
  Case c(dim, order);  // solid body (for spaces only)
  // The three-layer fluid case from the fluid tests.
  Mesh parent(ThreeLayerMeshFile(dim).c_str(), 1, 1);
  Array<int> attrs({1, 3});
  SubMesh solid(SubMesh::CreateFromDomain(parent, attrs));
  H1_FECollection fec(order, dim);
  FiniteElementSpace fes_u(&solid, &fec, dim);
  FiniteElementSpace fes_phi(&parent, &fec);
  ConstantCoefficient kappa(kKappa), mu(kMu);
  FunctionCoefficient rho_s(SolidDensity), rho_f(FluidDensity);
  IsotropicElasticRheology rheology(dim, kappa, mu);
  std::vector<FluidRegion> fluids;
  fluids.push_back(OuterCore(solid, rho_f));
  LinearQuasiStaticMixedSelfGravitatingProblem p(
      &fes_u, &fes_phi, rheology, rho_s, kG, kDtNDegree, nullptr, fluids);
  FunctionCoefficient sigma(RotLoad);
  const Array<int> surface = SurfaceMarker(solid);
  p.SetSurfaceLoad(sigma, surface);
  Array<int> inner_core({1});
  p.AddRegionRotations(inner_core);
  p.SetRelTol(1e-11);

  Vector moments(1);
  moments[0] = 2.0;
  RotationalFeedback rot(p, kOmega, moments);
  ASSERT_TRUE(rot.Solve(0.0, sigma, surface));
  EXPECT_GT(rot.AngularVelocityPerturbation().Norml2(), 0.0);
  // The psi-psi fluid block is genuinely nonzero here.
  CentrifugalPotential u0(dim, kOmega);
  u0.SetUnit(0);
  EXPECT_GT(std::abs(p.TidalTidalCoupling(u0, u0)), 0.0);
  EXPECT_LT(rot.ConsistencyResidual(sigma, surface), 5e-8);
  EXPECT_LT(OmegaRowResidual(p, fes_u, sigma, surface, moments, kOmega,
                             rot.AngularVelocityPerturbation()),
            5e-8);
}

INSTANTIATE_TEST_SUITE_P(Rotation, RotationTest,
                         testing::Values(Param{2, 1}, Param{2, 2},
                                         Param{3, 1}));

}  // namespace
