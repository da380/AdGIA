#include <numbers>

#include "MixedProblemTestCommon.hpp"
#include "TestCommon.hpp"

/*
  Gates for the profile border (SetProfileBorder), the dual form of the
  closure of an eliminated fluid region, on the canned two-layer ball
  with a fluid core.

  The core's density is rho_c - a r^2, for which gravity and
  d rho / d Phi0 are closed forms that stay finite at the centre, and its
  bulk modulus is (1 + delta) times the neutral value -rho^2 / rho', so
  that delta is the excess of the Bullen parameter over one: delta = 0 is
  a neutral core, delta > 0 a stably stratified one.

  The level border (SetLevelBorder) solves the primal form of the same
  closure with a different discretisation, so the two agree to
  discretisation error; the two solution routes of one border agree at
  solver grade.
*/

namespace {

using namespace self_grav_test;

constexpr double kRhoC = 1.2;
constexpr double kDrop = 0.1;  // rho_c - rho at the wall

double CoreA() { return kDrop / (kMeshRCmb * kMeshRCmb); }

double CoreDensity(const Vector& x) { return kRhoC - CoreA() * (x * x); }

// d rho / d Phi0 = (d rho / d r) / g, g = 4 pi G (rho_c r / 3 - a r^3 / 5).
double CoreDensityGradient(const Vector& x) {
  const double a = CoreA();
  return -2.0 * a /
         (4.0 * std::numbers::pi * kG * (kRhoC / 3.0 - a * (x * x) / 5.0));
}

struct Case {
  Mesh parent;
  Array<int> solid_attr;
  SubMesh body;
  H1_FECollection fec;
  FiniteElementSpace fes_u, fes_phi;
  ConstantCoefficient kappa{kKappa}, mu{kMu}, rho{kRho};
  IsotropicElasticRheology rheology;
  FunctionCoefficient rho_f{CoreDensity}, rho_prime{CoreDensityGradient};
  double delta;
  FunctionCoefficient kappa_f;
  std::vector<FluidRegion> fluids;
  Array<int> surface;

  Case(int order, double delta_)
      : parent(TwoLayerMeshFile(3).c_str(), 1, 1),
        solid_attr({2}),
        body(SubMesh::CreateFromDomain(parent, solid_attr)),
        fec(order, 3),
        fes_u(&body, &fec, 3),
        fes_phi(&parent, &fec),
        rheology(3, kappa, mu),
        delta(delta_),
        kappa_f([this](const Vector& x) {
          const double r = CoreDensity(x);
          return -(1.0 + delta) * r * r / CoreDensityGradient(x);
        }) {
    FluidRegion core;
    core.attributes = Array<int>({1});
    core.density = &rho_f;
    core.density_gradient = &rho_prime;
    core.interface_marker.SetSize(body.bdr_attributes.Max());
    core.interface_marker = 0;
    core.interface_marker[0] = 1;  // the wall of the core
    core.bulk_modulus = &kappa_f;
    fluids = {core};
    surface = SurfaceMarker(body);
  }

  // border: 0 none, 1 level, 2 profile.
  std::unique_ptr<LinearQuasiStaticMixedSelfGravitatingProblem> Solve(
      int border, Coefficient& sigma, bool monolithic = false,
      const Vector* moments = nullptr,
      const ProfileBorderOptions& options = {}) {
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        &fes_u, &fes_phi, rheology, rho, kG, kDtNDegree, nullptr, fluids);
    p->SetSolverType(
        LinearQuasiStaticMixedSelfGravitatingProblem::SolverType::BlockMINRES);
    p->SetRelTol(1e-12);
    if (border == 1) {
      p->SetLevelBorder(16);
    } else if (border == 2) {
      p->SetProfileBorder(options);
    }
    if (moments) {
      p->SetRotation(0.1, *moments);
    }
    if (monolithic) {
      p->SetMonolithicBorder();
    }
    p->SetSurfaceLoad(sigma, surface);
    p->AssembleForce(0.0);
    EXPECT_TRUE(p->Solve());
    return p;
  }
};

double RelDiff(const GridFunction& a, const GridFunction& b) {
  GridFunction d(a);
  d -= b;
  return d.Norml2() / (b.Norml2() + 1e-300);
}

double Degree2(const Vector& x) {
  const double c = x[2] / x.Norml2();
  return 0.02 * (3.0 * c * c - 1.0);
}

// The two forms of the closure agree under a uniform load, which excites
// the border fully (the reduction without a border is wrong by a factor
// of about two here). Measured at order 1: 5e-4 (neutral) and 7e-4
// (stratified) in the displacement, 2e-5 in the potential. The level
// border's answer moves by 1e-3 to 4e-3 with its number of nodes, so the
// gate is at that size.
class ProfileBorderAgainstLevel : public testing::TestWithParam<double> {};

TEST_P(ProfileBorderAgainstLevel, UniformLoad) {
  Case c(1, GetParam());
  ConstantCoefficient sigma(0.02);
  auto naked = c.Solve(0, sigma);
  auto level = c.Solve(1, sigma);
  auto profile = c.Solve(2, sigma);
  EXPECT_LT(RelDiff(profile->Displacement(), level->Displacement()), 3e-3);
  EXPECT_LT(RelDiff(profile->Potential(), level->Potential()), 2e-4);
  EXPECT_GT(RelDiff(naked->Displacement(), level->Displacement()), 0.5);
  const auto& B = profile->ProfileBorderData(0);
  ASSERT_EQ(B.layers.size(), 1u);
  EXPECT_EQ(B.layers[0].neutral, GetParam() == 0.0);
  EXPECT_EQ(B.size, GetParam() == 0.0 ? 1 : 7);
}

INSTANTIATE_TEST_SUITE_P(ProfileBorder, ProfileBorderAgainstLevel,
                         testing::Values(0.0, 0.5));

// Weak stratification is a regular perturbation of neutrality: the
// response departs from the neutral one in proportion to delta.
// Measured: 2.7e-4 at delta = 1e-3 and 2.7e-3 at 1e-2.
TEST(ProfileBorder, NeutralLimit) {
  ConstantCoefficient sigma(0.02);
  Case neutral(1, 0.0), weak(1, 1e-3), weaker(1, 1e-2);
  auto pn = neutral.Solve(2, sigma);
  auto p3 = weak.Solve(2, sigma);
  auto p2 = weaker.Solve(2, sigma);
  const double d3 = RelDiff(p3->Displacement(), pn->Displacement());
  const double d2 = RelDiff(p2->Displacement(), pn->Displacement());
  EXPECT_LT(d3, 1e-3);
  EXPECT_NEAR(d2 / d3, 10.0, 1.0);
}

// On a sphere a load of degree two leaves the level geometry unchanged
// and the border is not excited, up to the asphericity of the mesh.
// Measured: the unknown is 1.7e-4 of its value under the uniform load
// and the displacement is within 2e-4 of the unclosed reduction's.
TEST(ProfileBorder, InertAtDegreeTwo) {
  Case c(1, 0.0);
  ConstantCoefficient sigma(0.02);
  FunctionCoefficient sigma2(Degree2);
  auto p0 = c.Solve(2, sigma);
  auto naked = c.Solve(0, sigma2);
  auto p2 = c.Solve(2, sigma2);
  EXPECT_LT(std::abs(p2->ProfileBorderData(0).y[0]),
            1e-3 * std::abs(p0->ProfileBorderData(0).y[0]));
  EXPECT_LT(RelDiff(p2->Displacement(), naked->Displacement()), 1e-3);
}

// The response does not depend on the polynomial degree once it
// resolves the profile. Measured: 2e-7 between degrees 2 and 6.
TEST(ProfileBorder, PolynomialDegree) {
  Case c(1, 0.5);
  ConstantCoefficient sigma(0.02);
  ProfileBorderOptions low;
  low.degree = 2;
  auto p2 = c.Solve(2, sigma, false, nullptr, low);
  auto p6 = c.Solve(2, sigma);
  EXPECT_EQ(p2->ProfileBorderData(0).size, 3);
  EXPECT_LT(RelDiff(p2->Displacement(), p6->Displacement()), 1e-5);
}

// The mass of the region is conserved: the Eulerian change of density
// in the fluid, rho' phi - d(rho gamma)/dc, integrates to the mass that
// the wall sweeps out, int rho (m . u) dS with m the solid's outward
// normal. Evaluated here from the solution, independently of the
// border's rows.
TEST(ProfileBorder, MassOfTheRegionIsConserved) {
  for (double delta : {0.0, 0.5}) {
    Case c(1, delta);
    ConstantCoefficient sigma(0.02);
    auto p = c.Solve(2, sigma);
    const auto& B = p->ProfileBorderData(0);
    LinearForm wall(&c.fes_u);
    wall.AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(c.rho_f),
                               c.fluids[0].interface_marker);
    wall.Assemble();
    const double swept = wall(p->Displacement());
    const GridFunction& phi = p->Potential();
    const GridFunction& phi0 = p->BackgroundPotential();
    const int qorder = 2 * c.fes_phi.GetMaxElementOrder() + 2;
    double dm = 0.0;
    Vector b, db;
    for (int e = 0; e < c.parent.GetNE(); e++) {
      if (c.parent.GetAttribute(e) != 1) {
        continue;
      }
      ElementTransformation* T = c.parent.GetElementTransformation(e);
      const IntegrationRule& ir = IntRules.Get(T->GetGeometryType(), qorder);
      for (int q = 0; q < ir.GetNPoints(); q++) {
        const IntegrationPoint& ip = ir.IntPoint(q);
        T->SetIntPoint(&ip);
        B.Basis(phi0.GetValue(*T, ip), b, db);
        const double gamma = b * B.y, dgamma = db * B.y;
        const double rho = c.rho_f.Eval(*T, ip);
        const double rp = c.rho_prime.Eval(*T, ip);
        dm += ip.weight * T->Weight() *
              (rp * phi.GetValue(*T, ip) - rp * gamma - rho * dgamma);
      }
    }
    EXPECT_GT(std::abs(swept), 1e-6);
    EXPECT_NEAR(dm, swept, 1e-8 * std::abs(swept)) << "delta " << delta;
  }
}

// The elimination and the monolithic route solve one bordered system.
TEST(ProfileBorder, MonolithicMatchesElimination) {
  Case c(1, 0.5);
  ConstantCoefficient sigma(0.02);
  auto pe = c.Solve(2, sigma, false);
  auto pm = c.Solve(2, sigma, true);
  EXPECT_LT(RelDiff(pm->Displacement(), pe->Displacement()), 1e-7);
  EXPECT_LT(RelDiff(pm->Potential(), pe->Potential()), 1e-7);
}

// With rotation the centrifugal potential enters the border's rows: the
// two routes agree at solver grade, and the angular velocity agrees
// with the level border's to discretisation error (measured 4e-4, the
// displacement 7e-4).
TEST(ProfileBorder, RotationComposes) {
  Case c(1, 0.5);
  ConstantCoefficient sigma(0.02);
  Vector moments(3);
  moments[0] = 1.5262;
  moments[1] = 1.5262;
  moments[2] = 1.5312;
  auto pe = c.Solve(2, sigma, false, &moments);
  auto pm = c.Solve(2, sigma, true, &moments);
  auto pl = c.Solve(1, sigma, false, &moments);
  const double w = pe->AngularVelocity().Norml2();
  Vector dw(pm->AngularVelocity());
  dw -= pe->AngularVelocity();
  EXPECT_LT(RelDiff(pm->Displacement(), pe->Displacement()), 1e-7);
  EXPECT_LT(dw.Norml2() / w, 1e-7);
  dw = pl->AngularVelocity();
  dw -= pe->AngularVelocity();
  EXPECT_LT(RelDiff(pl->Displacement(), pe->Displacement()), 3e-3);
  EXPECT_LT(dw.Norml2() / w, 3e-3);
}

// A region may supply its stratification: the declared value takes the
// place of the one computed from the density gradient and the bulk
// modulus. Declaring the computed value changes nothing; declaring zero
// on a weakly stratified model gives one unknown and the neutral answer.
TEST(ProfileBorder, DeclaredStratification) {
  ConstantCoefficient sigma(0.02);
  Case computed(1, 0.5), declared(1, 0.5);
  ConstantCoefficient half(0.5), zero(0.0);
  declared.fluids[0].stratification = &half;
  auto pc = computed.Solve(2, sigma);
  auto pd = declared.Solve(2, sigma);
  EXPECT_LT(RelDiff(pd->Displacement(), pc->Displacement()), 1e-9);
  Case weak(1, 5e-3), neutral(1, 0.0);
  weak.fluids[0].stratification = &zero;
  auto pw = weak.Solve(2, sigma);
  auto pn = neutral.Solve(2, sigma);
  EXPECT_EQ(pw->ProfileBorderData(0).size, 1);
  EXPECT_LT(RelDiff(pw->Displacement(), pn->Displacement()), 1e-10);
}

// A region that declares its stratification and gives no density
// gradient has the gradient derived from the declaration, the density
// and the bulk modulus, which here is the analytic gradient exactly.
TEST(ProfileBorder, DensityGradientFromDeclaration) {
  ConstantCoefficient sigma(0.02);
  Case analytic(1, 0.5), derived(1, 0.5);
  ConstantCoefficient half(0.5);
  derived.fluids[0].stratification = &half;
  derived.fluids[0].density_gradient = nullptr;
  auto pa = analytic.Solve(2, sigma);
  auto pd = derived.Solve(2, sigma);
  EXPECT_LT(RelDiff(pd->Displacement(), pa->Displacement()), 1e-9);
  EXPECT_LT(RelDiff(pd->Potential(), pa->Potential()), 1e-9);
}

// An unstably stratified region is refused; with the option it is
// treated as neutral, and since a neutral layer does not consult the
// bulk modulus the answer is the neutral model's.
TEST(ProfileBorder, UnstableStratification) {
  ConstantCoefficient sigma(0.02);
  Case unstable(1, -0.02), neutral(1, 0.0);
  EXPECT_DEATH(unstable.Solve(2, sigma), "unstably stratified");
  ProfileBorderOptions options;
  options.unstable_as_neutral = true;
  auto pu = unstable.Solve(2, sigma, false, nullptr, options);
  auto pn = neutral.Solve(2, sigma);
  EXPECT_LT(RelDiff(pu->Displacement(), pn->Displacement()), 1e-10);
}

}  // namespace
