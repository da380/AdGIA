#include <numbers>

#include "MixedProblemTestCommon.hpp"
#include "TestCommon.hpp"

/*
  Tests for RotationalFeedback (the rotation border of the sea-level
  machinery) on the canned meshes.

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
  // psi through its interpolant: the convention of the tidal columns
  // and of the border (see RotationalFeedback::SurfacePairing).
  H1_FECollection fec(fes_u.GetMaxElementOrder(),
                      fes_u.GetMesh()->Dimension());
  FiniteElementSpace s(fes_u.GetMesh(), &fec);
  GridFunction psig(&s);
  psig.ProjectCoefficient(psi_k);
  GridFunctionCoefficient psii(&psig);
  ProductCoefficient sp(sigma, psii);
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
  const DenseMatrix inertia = InertiaMatrix(c.moments);
  DenseMatrix DP(inertia);
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
        DP(k, j) = inertia(k, j) +
                   p2->TidalTidalCoupling(*unit[k], *unit[j]);
      }
    }
    DenseMatrixInverse inv(DP);
    inv.Mult(rhs, w);
    Vector d(w);
    d -= target;
    err = d.Norml2() / target.Norml2();
  }
  // The feedback is weak, so three iterations contract to a relative
  // error below 1e-4.
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

// The level border's centrifugal coupling (the closure's omega
// columns; doc/dahlen_closure.tex, "Rotation"): on the fluid-core ball
// with Earth-like near-neutral moments, (a) polar wander is pure
// degree 2 and must be INERT to the border, while (b) the omega_3
// (length-of-day) response reads the fluid's degree-0 inertia
// perturbation, whose -f' part only the border supplies: under a
// degree-0-bearing load the bordered omega_3 differs finitely from the
// naked reduction's — the G9 contamination, measured.
TEST(RotationFluid, LevelBorderCentrifugalCoupling) {
  const int dim = 3, order = 1;
  Mesh parent(TwoLayerMeshFile(dim).c_str(), 1, 1);
  Array<int> solid_attr({2});
  SubMesh solid(SubMesh::CreateFromDomain(parent, solid_attr));
  H1_FECollection fec(order, dim);
  FiniteElementSpace fes_u(&solid, &fec, dim);
  FiniteElementSpace fes_phi(&parent, &fec);
  ConstantCoefficient kappa(kKappa), mu(kMu), kappa_f(kKappa);
  ConstantCoefficient rho_s(kRho);
  FunctionCoefficient rho_f(FluidDensity);
  IsotropicElasticRheology rheology(dim, kappa, mu);
  auto make_fluids = [&]() {
    FluidRegion core;
    core.attributes = Array<int>({1});
    core.density = &rho_f;
    core.interface_marker.SetSize(solid.bdr_attributes.Max());
    core.interface_marker = 0;
    core.interface_marker[0] = 1;  // the CMB
    core.bulk_modulus = &kappa_f;
    return std::vector<FluidRegion>{core};
  };
  const Array<int> surface = SurfaceMarker(solid);
  Vector moments(3);
  moments[0] = 1.5262;
  moments[1] = 1.5262;
  moments[2] = 1.5312;

  auto solve = [&](Coefficient& sigma, int levels) {
    auto fluids = make_fluids();
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        &fes_u, &fes_phi, rheology, rho_s, kG, kDtNDegree, nullptr, fluids);
    if (levels > 0) {
      p->SetLevelBorder(levels);
    }
    p->SetRotation(0.1, moments);
    p->SetSurfaceLoad(sigma, surface);
    p->SetRelTol(1e-11);
    p->AssembleForce(0.0);
    EXPECT_TRUE(p->Solve());
    return p;
  };

  // (a) A tesseral (m = 1) degree-2 load drives the wander, which is
  // border-inert: no degree-0 content anywhere in its loop.
  FunctionCoefficient tesseral([](const Vector& x) {
    const double r2 = x * x;
    return r2 == 0.0 ? 0.0 : 0.02 * x[0] * x[2] / r2;
  });
  auto pw0 = solve(tesseral, 0);
  auto pw1 = solve(tesseral, 12);
  const Vector &w0 = pw0->AngularVelocity(), &w1 = pw1->AngularVelocity();
  ASSERT_GT(std::hypot(w0[0], w0[1]), 1e-12);  // the wander is driven
  // Inert to the discrete-foliation floor: the level sets are spheres
  // only to facet accuracy, so the border leaks O(1e-6) into degree 2
  // on this mesh (the Love-number inertness is 4e-5 at h = 0.2).
  EXPECT_LT(std::abs(w1[0] - w0[0]) / std::hypot(w0[0], w0[1]), 1e-5);
  EXPECT_LT(std::abs(w1[1] - w0[1]) / std::hypot(w0[0], w0[1]), 1e-5);

  // (b) A uniform (degree-0) load: the naked reduction's omega_3 is
  // contaminated by the arbitrary degree-0 family member; the border
  // changes it by a finite, physical amount.
  ConstantCoefficient uniform(0.02);
  auto p0 = solve(uniform, 0);
  auto p1 = solve(uniform, 12);
  const double w3_naked = p0->AngularVelocity()[2];
  const double w3_border = p1->AngularVelocity()[2];
  ASSERT_GT(std::abs(w3_naked), 1e-12);
  const double shift = std::abs(w3_border - w3_naked) / std::abs(w3_naked);
  std::cout << "omega_3: naked " << w3_naked << ", bordered " << w3_border
            << " (relative shift " << shift << ")\n";
  EXPECT_GT(shift, 1e-3);  // the contamination is real ...
  EXPECT_LT(shift, 10.0);  // ... and finite

  // (c) The generalised door's ingredients on this class. The
  // whole-body model inertia (solid SubMesh + fluid region on the
  // parent) against the radial analytic value, I = (8 pi / 3)
  // int rho(r) r^4 dr; the coarse faceted ball carries its geometry
  // error, and the tensor is isotropic to the same grade (the full
  // model-inertia door rightly refuses such a degenerate model — its
  // physics gate lives on the oblate figures).
  {
    auto fluids = make_fluids();
    LinearQuasiStaticMixedSelfGravitatingProblem p(
        &fes_u, &fes_phi, rheology, rho_s, kG, kDtNDegree, nullptr, fluids);
    const DenseMatrix I0 = p.ModelInertiaTensor();
    const double A = 1.2 + 0.1 * kRIcb / (kRCmb - kRIcb);
    const double B = 0.1 / (kRCmb - kRIcb);
    const double rc = kRCmb;
    const double radial = A * std::pow(rc, 5) / 5.0 -
                          B * std::pow(rc, 6) / 6.0 +
                          kRho * (1.0 - std::pow(rc, 5)) / 5.0;
    const double exact = 8.0 * std::numbers::pi / 3.0 * radial;
    const double trace = (I0(0, 0) + I0(1, 1) + I0(2, 2)) / 3.0;
    EXPECT_LT(std::abs(trace - exact) / exact, 3e-2);
    double offdiag = 0.0;
    for (int i = 0; i < 3; i++) {
      for (int j = i + 1; j < 3; j++) {
        offdiag = std::max(offdiag, std::abs(I0(i, j)));
      }
    }
    EXPECT_LT(offdiag / trace, 3e-2);
  }

  // (d) The Tisserand gauge (the door's reading of the mass-weighted
  // gauge; solid-restricted automatically here) pins the rigid
  // representative only: omega is gauge-invariant, so the bordered
  // omega_3 must not move.
  {
    auto fluids = make_fluids();
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        &fes_u, &fes_phi, rheology, rho_s, kG, kDtNDegree, nullptr, fluids);
    p->SetLevelBorder(12);
    p->SetRotation(0.1, moments);
    p->SetTisserandGauge(true);
    p->SetSurfaceLoad(uniform, surface);
    p->SetRelTol(1e-11);
    p->AssembleForce(0.0);
    EXPECT_TRUE(p->Solve());
    EXPECT_LT(std::abs(p->AngularVelocity()[2] - w3_border) /
                  std::abs(w3_border),
              1e-7);
  }
}

INSTANTIATE_TEST_SUITE_P(Rotation, RotationTest,
                         testing::Values(Param{2, 1}, Param{2, 2},
                                         Param{3, 1}));

}  // namespace
