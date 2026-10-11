/*
  Gate GA1b (WP8.4 opening): the sea-level reciprocity theorems of
  Al-Attar, Syvret, Crawford, Mitrovica & Lloyd (2024,
  doc/Elasticity/ggad434.pdf) as certificates on the water-loaded mixed
  solve — the elastic fingerprint problem that SetWaterLoad implements
  (condensed sea-level equation, frozen shoreline; here all-ocean).

  The theorem: with two direct loads zeta_1, zeta_2 sharing the ocean
  function, the fingerprint responses satisfy

      int SL_1 zeta_2 dS = int SL_2 zeta_1 dS          (paper eq. 17)

  — self-adjointness of the fingerprint operator zeta -> SL — and the
  SAME identity holds with rotational feedbacks, with SL read on the
  equipotential of the combined gravitational and centrifugal
  potentials (eqs. 52-54). Three legs, on the canned 3-D two-layer
  ball (SetWaterLoad is 3-D only):

  1. spherically symmetric moduli (the classically tested case);
  2. LATERALLY HETEROGENEOUS 3-D elastic structure — per the paper's
     closing discussion this case has not previously been tested
     numerically; the discrete identity must hold regardless, because
     the assembled system stays symmetric whatever kappa(x), mu(x);
  3. rotation + 3-D structure jointly (the generalized theorem, read
     with psi from the solved rotational feedback).

  The pairings go through the POSTPROCESSED sea-level fields
  (SeaLevelOperator), i.e. the route tier in the WP7.6 sense: the
  algebraic solve is symmetric to solver grade, and the residual is the
  projection/quadrature gap of the readout. Thresholds encode
  measurements (see each leg).

  The melt loads are Gaussian caps at generic, non-symmetric positions;
  their masses are nonzero (the physical fingerprint case) — the
  uniform terms drop from the identity through the solve's mass row,
  as in the paper's proof.
*/

#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include "AdGIA.hpp"
#include "MixedProblemTestCommon.hpp"
#include "mfem.hpp"

using namespace mfem;
using namespace AdGIA;

namespace {

constexpr double kRhoW = 0.02;
constexpr double kOmega = 0.1;

// Gaussian melt cap centred at the unit direction n (s = angular width).
double MeltCap(const Vector& x, const Vector& n) {
  constexpr double s2 = 0.08;
  Vector d(x);
  d /= x.Norml2();
  d -= n;
  return 0.02 * std::exp(-(d * d) / s2);
}

Vector UnitDir(double a, double b, double c) {
  Vector n(3);
  n[0] = a;
  n[1] = b;
  n[2] = c;
  n /= n.Norml2();
  return n;
}

// Smooth lateral moduli variations, nonzero through degree 2 in the
// Cartesian harmonics (well-defined at the origin, no shared symmetry
// with the caps or with each other).
double KappaLateral(const Vector& x) {
  return self_grav_test::kKappa *
         (1.0 + 0.3 * (x[0] * x[0] - x[1] * x[1]) + 0.2 * x[0] * x[2]);
}
double MuLateral(const Vector& x) {
  return self_grav_test::kMu *
         (1.0 - 0.25 * (x[1] * x[1] - x[2] * x[2]) + 0.2 * x[0] * x[1]);
}

struct Fixture {
  std::unique_ptr<Mesh> parent;
  std::unique_ptr<SubMesh> body;
  std::unique_ptr<H1_FECollection> fec;
  std::unique_ptr<FiniteElementSpace> fes_u, fes_phi;
  ConstantCoefficient rho{self_grav_test::kRho};
  Array<int> surface;
  Vector moments;
  std::unique_ptr<ConstantCoefficient> w;

  explicit Fixture(int order) {
    parent =
        std::make_unique<Mesh>(self_grav_test::MeshFile(3).c_str(), 1, 1);
    body = std::make_unique<SubMesh>(SubMesh::CreateFromDomain(
        *parent, self_grav_test::BodyMarker(*parent)));
    fec = std::make_unique<H1_FECollection>(order, 3);
    fes_u = std::make_unique<FiniteElementSpace>(body.get(), fec.get(), 3);
    fes_phi = std::make_unique<FiniteElementSpace>(parent.get(), fec.get());
    surface = self_grav_test::SurfaceMarker(*body);
    moments.SetSize(3);
    moments[0] = 1.2;
    moments[1] = 1.3;
    moments[2] = 2.0;
    const double g0 =
        4.0 * M_PI * self_grav_test::kG * self_grav_test::kRho / 3.0;
    w = std::make_unique<ConstantCoefficient>(kRhoW / g0);
  }

  VectorFunctionCoefficient GradPhi0() {
    const double g0 =
        4.0 * M_PI * self_grav_test::kG * self_grav_test::kRho / 3.0;
    return VectorFunctionCoefficient(3, [g0](const Vector& x, Vector& v) {
      v = x;
      v *= g0;
    });
  }
};

// One fingerprint solve: the melt load zeta through the water border,
// optional rotation; the sea-level field copied out of the shared
// SeaLevelOperator.
GridFunction Fingerprint(Fixture& f, const Rheology& rheology,
                         Coefficient& zeta, bool rotation,
                         SeaLevelOperator& sea,
                         VectorCoefficient& grad_phi0) {
  LinearQuasiStaticMixedSelfGravitatingProblem p(
      f.fes_u.get(), f.fes_phi.get(), rheology, f.rho, self_grav_test::kG,
      self_grav_test::kDtNDegree);
  p.SetRelTol(1e-12);
  p.SetWaterLoad(*f.w, zeta, f.surface);
  if (rotation) {
    p.SetRotation(kOmega, f.moments);
  }
  p.AssembleForce(0.0);
  EXPECT_TRUE(p.Solve());
  Coefficient* psi = rotation ? &p.SolutionCentrifugalPotential() : nullptr;
  sea.SeaLevelChangeFrom(p.Displacement(), grad_phi0, p.Potential(),
                         p.UniformPotentialTerm(), psi);
  GridFunction sl(&sea.SurfaceSpace());
  sl = sea.SeaLevelChangeField();
  return sl;
}

// The reciprocity defect |<SL_1, z_2> - <SL_2, z_1>| / max(|.|,|.|).
double ReciprocityDefect(Fixture& f, const Rheology& rheology,
                         bool rotation) {
  auto grad_phi0 = f.GradPhi0();
  auto n1 = UnitDir(0.3, 0.5, 0.81);
  auto n2 = UnitDir(-0.7, 0.25, 0.55);
  auto z1 = FunctionCoefficient(
      [n1](const Vector& x) { return MeltCap(x, n1); });
  auto z2 = FunctionCoefficient(
      [n2](const Vector& x) { return MeltCap(x, n2); });

  SeaLevelOperator sea(*f.fes_u, f.surface);
  auto sl1 = Fingerprint(f, rheology, z1, rotation, sea, grad_phi0);
  auto sl2 = Fingerprint(f, rheology, z2, rotation, sea, grad_phi0);

  auto pair = [&](GridFunction& sl, Coefficient& z) {
    GridFunctionCoefficient slc(&sl);
    ProductCoefficient p(slc, z);
    return sea.SurfaceIntegral(p);
  };
  const double p12 = pair(sl1, z2);
  const double p21 = pair(sl2, z1);
  return std::abs(p12 - p21) / std::max(std::abs(p12), std::abs(p21));
}

}  // namespace

TEST(AdjointFingerprint, SphericalReciprocity) {
  Fixture f(1);
  auto kappa = ConstantCoefficient(self_grav_test::kKappa);
  auto mu = ConstantCoefficient(self_grav_test::kMu);
  auto rheology = IsotropicElasticRheology(3, kappa, mu);
  // Measured 2.5e-5 at order 1 and 7.6e-6 at order 2: the defect is
  // the readout's projection/quadrature gap and converges with order
  // (the algebraic solve is symmetric to solver grade), so the gate
  // checks both the size and the convergence.
  const double d1 = ReciprocityDefect(f, rheology, false);
  EXPECT_LT(d1, 1e-4);
  Fixture f2(2);
  const double d2 = ReciprocityDefect(f2, rheology, false);
  EXPECT_LT(d2, 3e-5);
  EXPECT_LT(d2, d1);
}

TEST(AdjointFingerprint, LateralStructureReciprocity) {
  // The never-previously-tested case: laterally heterogeneous kappa(x),
  // mu(x). The discrete identity holds regardless of the structure —
  // the assembled system stays symmetric — so the threshold is the
  // same solver-plus-readout grade as the spherical leg.
  Fixture f(1);
  auto kappa = FunctionCoefficient(KappaLateral);
  auto mu = FunctionCoefficient(MuLateral);
  auto rheology = IsotropicElasticRheology(3, kappa, mu);
  // Measured 3.5e-5 — the same readout grade as the spherical leg:
  // the first numerical test of fingerprint reciprocity with 3-D
  // elastic structure.
  EXPECT_LT(ReciprocityDefect(f, rheology, false), 1e-4);
}

TEST(AdjointFingerprint, RotationWithStructureReciprocity) {
  // The generalized theorem (paper §3.2, eq. 54): rotation composed
  // with the water border, jointly with the 3-D structure; SL read on
  // the combined equipotential (psi from the solved feedback).
  Fixture f(1);
  auto kappa = FunctionCoefficient(KappaLateral);
  auto mu = FunctionCoefficient(MuLateral);
  auto rheology = IsotropicElasticRheology(3, kappa, mu);
  // Measured 1.9e-5 — the generalized theorem at the same grade.
  EXPECT_LT(ReciprocityDefect(f, rheology, true), 1e-4);
}
