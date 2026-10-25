#include <cstdlib>
#include <iostream>

#include "MixedProblemTestCommon.hpp"
#include "TestCommon.hpp"

/*
  Tests for the gauged-fluid mode of LinearQuasiStaticMixedSelfGravitatingProblem
  on the three-layer meshes: the displacement SubMesh carries all three
  layers, the outer core has its bulk modulus and no shear, and
  SetFluid's gauge treatment supplies the penalty and the Tikhonov refinement;
  no FluidRegions, so the interface terms (F2)-(F3) and the fluid mass term
  never enter (doc/gauged_fluid.md).

  The reference is the Dahlen path of TestMixedProblemFluid. The two
  formulations describe the same physics when the fluid is materially
  barotropic (doc/gauged_fluid.md, "Equivalence and the Adams-Williamson
  condition"), so
  the fluid's bulk modulus is built from the Adams-Williamson condition
  kappa = rho^2 |grad Phi0| / |drho/dr| on the problem's own discrete
  background gravity (N^2 = 0); with the test's linear density profile the
  agreement is then to the two discretisations' own error.

  - Solid displacement and potential match the Dahlen solution under a
    ZERO-MEAN surface load (a degree-2-type load leaves both solutions'
    rigid content at the rigid-mode-residual level, so the different gauges
    do not enter at the tolerance compared). The degree-0 component is
    excluded deliberately: there the two treatments differ by design --
    Dahlen's fluid never sees the fluid's bulk modulus, the gauged fluid
    does (as does pyslfp at l = 0) -- and a separate test asserts that
    difference is present under a uniform load.
  - The two solver types agree within the gauged problem.
  - The refinement residuals contract by O(eps mu_g / mu).
  - A uniform tidal gradient psi = a.x loads exactly a rigid mode of the
    whole body, fluid included, so the response in the solid must stay
    small: a sign check on the fluid's gravity and coupling terms. This is
    the worst case for the gauged fluid (doc/gauged_fluid.md, "Gauge
    fixing: penalty plus iterated refinement"): the leakage is ~3% of a
    comparable surface-load response, bounded here by 5%, not the
    rigid-residual level the Dahlen path reaches.
  - The 3-D gauged problem solves and its refinement residuals contract.
*/

namespace {

using namespace self_grav_test;

constexpr double kEps = 1.0e-2;
constexpr int kRefine = 3;

double UniformTidal(const Vector& x, double /*t*/) {
  return 0.01 * (x.Size() == 2 ? x[1] : x[2]);
}

// The gauged problem on the whole body.
struct GaugedCase {
  std::unique_ptr<Mesh> parent;
  std::unique_ptr<SubMesh> body;
  std::unique_ptr<H1_FECollection> fec;
  std::unique_ptr<FiniteElementSpace> fes_u, fes_phi;
  AWBulkModulus kappa;
  FunctionCoefficient mu{GaugedShearModulus};
  ConstantCoefficient mu_gauge{kKappa};
  FunctionCoefficient rho{FullDensity};
  std::unique_ptr<IsotropicElasticRheology> rheology;
  FunctionCoefficient sigma;
  Array<int> surface, fluid;
  std::unique_ptr<LinearQuasiStaticMixedSelfGravitatingProblem> problem;

  GaugedCase(int dim, int order, bool zero_mean_load = false)
      : sigma(zero_mean_load
                  ? std::function<double(const Vector&, double)>(
                        ZeroMeanSurfaceLoad)
                  : std::function<double(const Vector&, double)>(
                        SurfaceLoad)) {
    parent = std::make_unique<Mesh>(ThreeLayerMeshFile(dim).c_str(), 1, 1);
    Array<int> attrs({1, 2, 3});
    body = std::make_unique<SubMesh>(SubMesh::CreateFromDomain(*parent, attrs));
    fec = std::make_unique<H1_FECollection>(order, dim);
    fes_u = std::make_unique<FiniteElementSpace>(body.get(), fec.get(), dim);
    fes_phi = std::make_unique<FiniteElementSpace>(parent.get(), fec.get());
    rheology = std::make_unique<IsotropicElasticRheology>(dim, kappa, mu);
    surface = SurfaceMarker(*body);
    fluid = Array<int>({0, 1, 0});
    problem = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        fes_u.get(), fes_phi.get(), *rheology, rho, kG, kDtNDegree);
    kappa.g = &problem->BackgroundGravity();
    problem->SetFluid(fluid, mu_gauge, GaugePenaltyOptions{kEps, kRefine});
    problem->SetSurfaceLoad(sigma, surface);
    problem->SetRelTol(1e-11);
  }
};

// The Dahlen reference (as in TestMixedProblemFluid).
struct DahlenCase {
  std::unique_ptr<Mesh> parent;
  std::unique_ptr<SubMesh> solid;
  std::unique_ptr<H1_FECollection> fec;
  std::unique_ptr<FiniteElementSpace> fes_u, fes_phi;
  ConstantCoefficient kappa{kKappa}, mu{kMu};
  FunctionCoefficient rho_s{SolidDensity}, rho_f{FluidDensity};
  std::unique_ptr<IsotropicElasticRheology> rheology;
  FunctionCoefficient sigma;
  Array<int> surface;
  std::vector<FluidRegion> fluids;
  std::unique_ptr<LinearQuasiStaticMixedSelfGravitatingProblem> problem;

  DahlenCase(int dim, int order, bool zero_mean_load = false)
      : sigma(zero_mean_load
                  ? std::function<double(const Vector&, double)>(
                        ZeroMeanSurfaceLoad)
                  : std::function<double(const Vector&, double)>(
                        SurfaceLoad)) {
    parent = std::make_unique<Mesh>(ThreeLayerMeshFile(dim).c_str(), 1, 1);
    Array<int> attrs({1, 3});
    solid =
        std::make_unique<SubMesh>(SubMesh::CreateFromDomain(*parent, attrs));
    fec = std::make_unique<H1_FECollection>(order, dim);
    fes_u = std::make_unique<FiniteElementSpace>(solid.get(), fec.get(), dim);
    fes_phi = std::make_unique<FiniteElementSpace>(parent.get(), fec.get());
    rheology = std::make_unique<IsotropicElasticRheology>(dim, kappa, mu);
    surface = SurfaceMarker(*solid);
    fluids.push_back(OuterCore(*solid, rho_f));
    problem = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        fes_u.get(), fes_phi.get(), *rheology, rho_s, kG, kDtNDegree, nullptr,
        fluids);
    problem->SetSurfaceLoad(sigma, surface);
    problem->AddRegionRotations(Array<int>({1}));
    problem->SetRelTol(1e-11);
  }
};

double RelDiff(const GridFunction& a, const GridFunction& b) {
  GridFunction d(a);
  d -= b;
  return self_grav_test::L2Norm(d) / self_grav_test::L2Norm(b);
}

// The Dahlen path closed by the level border (SetLevelBorder) under the
// uniform load: the outer core carries the same Adams-Williamson bulk
// modulus as the gauged body, so at degree 0 the two must now agree
// (doc/dahlen_closure.tex, "The closure in the cumulative-volume
// variable": at l = 0 the bordered system is the welded compressible
// fluid in a different assembly).
struct DahlenBorderCase {
  std::unique_ptr<Mesh> parent;
  std::unique_ptr<SubMesh> solid;
  std::unique_ptr<H1_FECollection> fec;
  std::unique_ptr<FiniteElementSpace> fes_u, fes_phi;
  ConstantCoefficient kappa{kKappa}, mu{kMu};
  AWBulkModulus kappa_f;
  FunctionCoefficient rho_s{SolidDensity}, rho_f{FluidDensity};
  std::unique_ptr<IsotropicElasticRheology> rheology;
  FunctionCoefficient sigma{SurfaceLoad};
  Array<int> surface;
  std::vector<FluidRegion> fluids;
  std::unique_ptr<LinearQuasiStaticMixedSelfGravitatingProblem> problem;
  // The background gravity on the PARENT mesh, where the border evaluates
  // the fluid's bulk modulus (BackgroundGravity() lives on the body).
  std::unique_ptr<GradientGridFunctionCoefficient> g_parent;

  DahlenBorderCase(int dim, int order, int levels) {
    parent = std::make_unique<Mesh>(ThreeLayerMeshFile(dim).c_str(), 1, 1);
    Array<int> attrs({1, 3});
    solid =
        std::make_unique<SubMesh>(SubMesh::CreateFromDomain(*parent, attrs));
    fec = std::make_unique<H1_FECollection>(order, dim);
    fes_u = std::make_unique<FiniteElementSpace>(solid.get(), fec.get(), dim);
    fes_phi = std::make_unique<FiniteElementSpace>(parent.get(), fec.get());
    rheology = std::make_unique<IsotropicElasticRheology>(dim, kappa, mu);
    surface = SurfaceMarker(*solid);
    fluids.push_back(OuterCore(*solid, rho_f));
    fluids.back().bulk_modulus = &kappa_f;
    problem = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        fes_u.get(), fes_phi.get(), *rheology, rho_s, kG, kDtNDegree, nullptr,
        fluids);
    g_parent = std::make_unique<GradientGridFunctionCoefficient>(
        &problem->BackgroundPotential());
    kappa_f.g = g_parent.get();
    problem->SetSurfaceLoad(sigma, surface);
    problem->AddRegionRotations(Array<int>({1}));
    problem->SetRelTol(1e-11);
    problem->SetLevelBorder(levels);
  }
};

void ExpectDegree0Agreement(int dim, int order, int levels, double tol) {
  GaugedCase g(dim, order);
  DahlenBorderCase d(dim, order, levels);
  g.problem->AssembleForce(0.0);
  EXPECT_TRUE(g.problem->Solve());
  d.problem->AssembleForce(0.0);
  EXPECT_TRUE(d.problem->Solve());
  const double dphi = RelDiff(g.problem->Potential(), d.problem->Potential());
  FiniteElementSpace fes_parent(g.parent.get(), g.fec.get(), dim);
  GridFunction on_parent(&fes_parent);
  on_parent = 0.0;
  SubMesh::Transfer(g.problem->Displacement(), on_parent);
  GridFunction u_g(d.fes_u.get());
  SubMesh::Transfer(on_parent, u_g);
  const double du = RelDiff(u_g, d.problem->Displacement());
  if (std::getenv("ADGIA_TEST_VERBOSE")) {
    Vector c, U, s, nu, lambda;
    d.problem->LevelBorderSolution(0, c, U, s, nu, lambda);
    std::cout << "degree-0 border vs gauged, dim " << dim << ": potential "
              << dphi << ", displacement " << du << "; wall multipliers";
    for (int w = 0; w < lambda.Size(); w++) {
      std::cout << " " << lambda[w];
    }
    std::cout << "\n  diagnostics (U, flux per wall; bulk, wall mass): ";
    d.problem->LevelBorderDiagnostics(0).Print(std::cout, 8);
    std::cout << "  U: ";
    U.Print(std::cout, 12);
    std::cout << "  s: ";
    s.Print(std::cout, 12);
    std::cout << "  nu: ";
    nu.Print(std::cout, 12);
  }
  EXPECT_LT(dphi, tol);
  EXPECT_LT(du, tol);
}

// The agreement is at the two discretisations' own level, and the gauged
// twin carries its O(eps) bias; on the coarse canned meshes it tightens
// with the order (3-D: potential 4.7% -> 0.4%, displacement 13% -> 2.7%
// from order 1 to 2). The order-2 case is the gate; the others record
// the coarse levels (and 2-D carries the class's constant-mode
// regularisation, which is a degree-0 matter). Compare with
// Degree0DiffersFromDahlen2D, where the reduction alone is 1e-2 off and
// more.
TEST(MixedProblemGauged, Degree0MatchesDahlenWithLevelBorder2D) {
  ExpectDegree0Agreement(2, 2, 16, 1.5e-1);
}

TEST(MixedProblemGauged, Degree0MatchesDahlenWithLevelBorder3D) {
  ExpectDegree0Agreement(3, 1, 12, 1.5e-1);
}

TEST(MixedProblemGauged, Degree0MatchesDahlenWithLevelBorder3DOrder2) {
  ExpectDegree0Agreement(3, 2, 16, 5.0e-2);
}

TEST(MixedProblemGauged, MatchesDahlenZeroMeanLoad2D) {
  const int dim = 2, order = 2;
  GaugedCase g(dim, order, true);
  DahlenCase d(dim, order, true);

  g.problem->AssembleForce(0.0);
  EXPECT_TRUE(g.problem->Solve());
  d.problem->AssembleForce(0.0);
  EXPECT_TRUE(d.problem->Solve());

  // Potential on the shared parent space.
  EXPECT_LT(RelDiff(g.problem->Potential(), d.problem->Potential()), 2.0e-2);

  // Solid displacement: gauged body -> parent -> solid SubMesh.
  FiniteElementSpace fes_parent(g.parent.get(), g.fec.get(), dim);
  GridFunction on_parent(&fes_parent);
  on_parent = 0.0;
  SubMesh::Transfer(g.problem->Displacement(), on_parent);
  GridFunction u_g(d.fes_u.get());
  SubMesh::Transfer(on_parent, u_g);
  EXPECT_LT(RelDiff(u_g, d.problem->Displacement()), 2.0e-2);

  // Refinement residuals contract.
  const auto& res = g.problem->GaugeResiduals();
  ASSERT_EQ(static_cast<int>(res.size()), kRefine);
  EXPECT_LT(res[1] / res[0], 0.35);
}

// Under a uniform (degree-0) load the two treatments MUST differ: Dahlen's
// fluid is described by the potential alone and never sees the fluid's
// bulk modulus at l = 0, while the gauged fluid carries it (as pyslfp
// does). This pins the known degree-0 gap to the fluid treatment.
TEST(MixedProblemGauged, Degree0DiffersFromDahlen2D) {
  const int dim = 2, order = 2;
  GaugedCase g(dim, order);
  DahlenCase d(dim, order);
  g.problem->AssembleForce(0.0);
  EXPECT_TRUE(g.problem->Solve());
  d.problem->AssembleForce(0.0);
  EXPECT_TRUE(d.problem->Solve());
  EXPECT_GT(RelDiff(g.problem->Potential(), d.problem->Potential()), 1.0e-2);
}

TEST(MixedProblemGauged, SolversAgree2D) {
  const int dim = 2, order = 2;
  GaugedCase g(dim, order);
  auto& p = *g.problem;

  p.AssembleForce(0.0);
  EXPECT_TRUE(p.Solve());
  GridFunction u1(p.Displacement());
  GridFunction phi1(p.Potential());

  p.SetSolverType(LinearQuasiStaticMixedSelfGravitatingProblem::SolverType::SchurCG);
  p.AssembleForce(0.0);
  EXPECT_TRUE(p.Solve());
  EXPECT_LT(RelDiff(p.Displacement(), u1), 1.0e-6);
  EXPECT_LT(RelDiff(p.Potential(), phi1), 1.0e-6);
}

TEST(MixedProblemGauged, UniformTidalGradientIsRigid2D) {
  const int dim = 2, order = 2;
  GaugedCase g(dim, order);
  auto& p = *g.problem;
  FunctionCoefficient psi(UniformTidal);
  p.SetTidalPotential(psi);

  // Solve for the surface load alone, then with the tidal term: the
  // uniform gradient loads a rigid mode of the whole body (fluid
  // included), so the difference stays at the rigid-residual level.
  p.AssembleForce(0.0);
  EXPECT_TRUE(p.Solve());
  GridFunction u_load(p.Displacement());

  // A rough scale for the tidal load's size: |psi| ~ 1e-2.
  const double u_scale = self_grav_test::L2Norm(u_load);
  EXPECT_GT(u_scale, 0.0);

  GaugedCase g2(dim, order);
  FunctionCoefficient psi2(UniformTidal);
  g2.problem->SetTidalPotential(psi2);
  // Tidal load only: drop the surface load by solving at t = -1 (the load
  // scales with 1 + t while the uniform gradient's rigidity is exact at
  // any amplitude).
  g2.problem->AssembleForce(-1.0);
  EXPECT_TRUE(g2.problem->Solve());
  // Compare on the solid: the fluid part of the response to a rigid-mode
  // load is relabelling junk, bounded by the penalty but amplified by
  // 1/eps relative to the discretisation residual that sources it.
  FiniteElementSpace fes_parent(g2.parent.get(), g2.fec.get(), dim);
  Array<int> solid_attrs({1, 3});
  auto solid = std::make_unique<SubMesh>(
      SubMesh::CreateFromDomain(*g2.parent, solid_attrs));
  FiniteElementSpace fes_s(solid.get(), g2.fec.get(), dim);
  GridFunction on_parent(&fes_parent);
  on_parent = 0.0;
  SubMesh::Transfer(g2.problem->Displacement(), on_parent);
  GridFunction u_s(&fes_s);
  SubMesh::Transfer(on_parent, u_s);
  const double u_tidal = self_grav_test::L2Norm(u_s);
  // A rigid-mode load is the worst case for the gauged fluid: its
  // discretisation residual excites the fluid near-kernel, whose response
  // is amplified by 1/eps and grows with the refinement count
  // (semi-convergence of iterated Tikhonov) -- hence moderate eps and few
  // refinements, and the leakage decreasing with h. At the recommended
  // eps = 1e-2 with 3 refinements the leakage into the solid is ~3% here;
  // the Dahlen path, with no fluid near-kernel, reaches the rigid-residual
  // level instead.
  EXPECT_LT(u_tidal / u_scale, 5.0e-2);
}

TEST(MixedProblemGauged, Runs3D) {
  const int dim = 3, order = 1;
  GaugedCase g(dim, order);
  auto& p = *g.problem;
  p.AssembleForce(0.0);
  EXPECT_TRUE(p.Solve());
  EXPECT_GT(self_grav_test::L2Norm(p.Displacement()), 0.0);
  const auto& res = p.GaugeResiduals();
  ASSERT_EQ(static_cast<int>(res.size()), kRefine);
  EXPECT_LT(res[1] / res[0], 0.5);
}

}  // namespace
