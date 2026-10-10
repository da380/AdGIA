#include <numbers>

#include "MixedProblemTestCommon.hpp"
#include "TestCommon.hpp"

/*
  Gates for the monolithic bordered solve (SetMonolithicBorder: one
  MINRES on [K B; B^T Db] with the preconditioner-probed |Schur| border
  block and true-residual refinement) against the block-elimination
  reference, on the canned two-layer meshes with the fingerprint-style
  analytical state. The two paths solve the same system, so agreement is
  at solver grade; the rotating case runs at NEAR-NEUTRAL moments
  (C1 = C2 close to C3, the Earth-like regime that amplifies border
  error), which is exactly where the probed preconditioner must hold.
  3-D only: in 2-D the compatibility relocation (the log-DtN monopole
  gauge) pairs the border rows and columns asymmetrically by design,
  so the monolithic path refuses the dimension.
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
  ConstantCoefficient kappa{kKappa}, mu{kMu}, rho{kRho};
  std::unique_ptr<IsotropicElasticRheology> rheology;
  Array<int> surface;
  int dim;

  explicit Case(int d, int order) : dim(d) {
    parent = std::make_unique<Mesh>(MeshFile(dim).c_str(), 1, 1);
    body = std::make_unique<SubMesh>(
        SubMesh::CreateFromDomain(*parent, BodyMarker(*parent)));
    fec = std::make_unique<H1_FECollection>(order, dim);
    fes_u = std::make_unique<FiniteElementSpace>(body.get(), fec.get(), dim);
    fes_phi = std::make_unique<FiniteElementSpace>(parent.get(), fec.get());
    rheology = std::make_unique<IsotropicElasticRheology>(dim, kappa, mu);
    surface = SurfaceMarker(*body);
  }

  std::unique_ptr<LinearQuasiStaticMixedSelfGravitatingProblem> Problem(
      bool monolithic) {
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        fes_u.get(), fes_phi.get(), *rheology, rho, kG, kDtNDegree);
    p->SetSolverType(
        LinearQuasiStaticMixedSelfGravitatingProblem::SolverType::
            BlockMINRES);
    p->SetRelTol(1e-12);
    if (monolithic) {
      p->SetMonolithicBorder();
    }
    return p;
  }

  mfem::Vector Moments() const {
    // Near-neutral, Earth-like ratios: the hard regime for the border.
    mfem::Vector m(3);
    m[0] = 1.5262;
    m[1] = 1.5262;
    m[2] = 1.5312;
    return m;
  }
};

double RelDiff(const GridFunction& a, const GridFunction& b) {
  GridFunction d(a);
  d -= b;
  return d.Norml2() / (b.Norml2() + 1e-300);
}

using Param = std::tuple<int, int>;

class MonolithicBorderTest : public testing::TestWithParam<Param> {};

TEST_P(MonolithicBorderTest, WaterMatchesElimination) {
  const auto [dim, order] = GetParam();
  Case c(dim, order);
  FunctionCoefficient w([dim](const Vector& x) {
    return kRhoW * Fraction0(x) / (G0(dim) * x.Norml2());
  });
  FunctionCoefficient sigma_d(MeltLoad);

  auto run = [&](bool monolithic) {
    auto p = c.Problem(monolithic);
    p->SetWaterLoad(w, sigma_d, c.surface);
    p->AssembleForce(0.0);
    EXPECT_TRUE(p->Solve());
    return p;
  };
  auto pe = run(false);
  auto pm = run(true);
  EXPECT_LT(RelDiff(pm->Displacement(), pe->Displacement()), 1e-8);
  EXPECT_LT(RelDiff(pm->Potential(), pe->Potential()), 1e-8);
  EXPECT_NEAR(pm->UniformPotentialTerm(), pe->UniformPotentialTerm(),
              1e-8 * std::abs(pe->UniformPotentialTerm()));
}

TEST_P(MonolithicBorderTest, RotationMatchesElimination) {
  const auto [dim, order] = GetParam();
  if (order > 1) {
    GTEST_SKIP() << "covered at order 1";
  }
  Case c(dim, order);
  FunctionCoefficient w([dim](const Vector& x) {
    return kRhoW * Fraction0(x) / (G0(dim) * x.Norml2());
  });
  FunctionCoefficient sigma_d(MeltLoad);

  auto run = [&](bool monolithic, bool water) {
    auto p = c.Problem(monolithic);
    if (water) {
      p->SetWaterLoad(w, sigma_d, c.surface);
    } else {
      p->SetSurfaceLoad(sigma_d, c.surface);
    }
    p->SetRotation(0.1, c.Moments());
    p->AssembleForce(0.0);
    EXPECT_TRUE(p->Solve());
    return p;
  };
  for (bool water : {false, true}) {
    auto pe = run(false, water);
    auto pm = run(true, water);
    EXPECT_LT(RelDiff(pm->Displacement(), pe->Displacement()), 1e-8);
    Vector dw(pm->AngularVelocity());
    dw -= pe->AngularVelocity();
    EXPECT_LT(dw.Norml2() / (pe->AngularVelocity().Norml2() + 1e-300),
              1e-7);
  }
}

INSTANTIATE_TEST_SUITE_P(MonolithicBorder, MonolithicBorderTest,
                         testing::Values(Param{3, 1}, Param{3, 2}));

}  // namespace
