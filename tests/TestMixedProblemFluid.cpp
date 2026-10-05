#include "MixedProblemTestCommon.hpp"
#include "TestCommon.hpp"

/*
  Tests for LinearQuasiStaticMixedSelfGravitatingProblem with fluid regions on
  the canned three-layer meshes (solid inner core, fluid outer core, solid
  mantle; one disconnected solid SubMesh) and the two-layer disc (fluid
  core, mantle).

  - The Schur-complement CG and the block MINRES solvers agree to solver
    tolerance, as without fluids (both solve the same restricted system);
    likewise on the two-layer disc, where the fluid is enclosed by the
    solid only.
  - The global rigid modes remain near-null with the interface and fluid
    mass terms in place (a sign slip in any of them would make the residual
    O(1)); the residuals decrease with the order. The inner core's
    rotations are near-null too; its translation is not.
  - Without AddRegionRotations() the solution differs from the projected
    one by an inner-core rotation only.
  - The tidal load: without fluids it equals the directly assembled
    -int rho grad psi . v; with fluids the response is linear in psi and the
    two solvers agree.
  - A supplied rho'_F (BarotropicDensityGradientCoefficient) reproduces the
    default one.
  - Potential-block diagnostic: the smallest Ritz value is positive for the
    model, lies below that of the solid-only block (the fluid mass term
    lowers the spectrum), and drops further for a steep fluid density
    gradient.
  - The viscoelastic operator runs on the problem.
  - The load's own potential does not feel the fluid mass term: it is the
    same with rho'_F set to zero.
  - The approximate CMB conditions ('uniform', 'winkler') keep the operator
    symmetric and change the answer.
*/

namespace {

using namespace self_grav_test;

using Param = std::tuple<int, int>;

struct Case {
  std::unique_ptr<Mesh> parent;
  std::unique_ptr<SubMesh> solid;
  std::unique_ptr<H1_FECollection> fec;
  std::unique_ptr<FiniteElementSpace> fes_u, fes_phi;
  ConstantCoefficient kappa{kKappa}, mu{kMu};
  FunctionCoefficient rho_s{SolidDensity}, rho_f{FluidDensity};
  std::unique_ptr<IsotropicElasticRheology> rheology;
  FunctionCoefficient sigma{SurfaceLoad};
  Array<int> surface;
  std::vector<FluidRegion> fluids;

  Case(int dim, int order) {
    parent = std::make_unique<Mesh>(ThreeLayerMeshFile(dim).c_str(), 1, 1);
    EXPECT_EQ(parent->Dimension(), dim);
    Array<int> attrs({1, 3});
    solid = std::make_unique<SubMesh>(SubMesh::CreateFromDomain(*parent, attrs));
    EXPECT_EQ(solid->bdr_attributes.Max(), 3);
    fec = std::make_unique<H1_FECollection>(order, dim);
    fes_u = std::make_unique<FiniteElementSpace>(solid.get(), fec.get(), dim);
    fes_phi = std::make_unique<FiniteElementSpace>(parent.get(), fec.get());
    rheology = std::make_unique<IsotropicElasticRheology>(dim, kappa, mu);
    surface = SurfaceMarker(*solid);
    fluids.push_back(OuterCore(*solid, rho_f));
  }

  std::unique_ptr<LinearQuasiStaticMixedSelfGravitatingProblem> Problem(
      bool with_load = true, bool region_rotations = true,
      const std::vector<FluidRegion>* regions = nullptr,
      Coefficient* solid_density = nullptr, double G = kG) {
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        fes_u.get(), fes_phi.get(), *rheology,
        solid_density ? *solid_density : rho_s, G, kDtNDegree, nullptr,
        regions ? *regions : fluids);
    if (with_load) {
      p->SetSurfaceLoad(sigma, surface);
    }
    if (region_rotations) {
      Array<int> inner_core({1});
      p->AddRegionRotations(inner_core);
    }
    p->SetRelTol(1e-11);
    return p;
  }

  // A rigid mode restricted to the inner core, on true dofs.
  Vector InnerCoreMode(VectorCoefficient& mode) {
    GridFunction gf(fes_u.get());
    gf.ProjectCoefficient(mode);
    Array<int> vdofs;
    Array<int> in(fes_u->GetVSize());
    in = 0;
    for (int e = 0; e < solid->GetNE(); e++) {
      if (solid->GetAttribute(e) == 1) {
        fes_u->GetElementVDofs(e, vdofs);
        for (int v : vdofs) {
          in[FiniteElementSpace::DecodeDof(v)] = 1;
        }
      }
    }
    for (int i = 0; i < in.Size(); i++) {
      if (!in[i]) {
        gf[i] = 0.0;
      }
    }
    Vector t;
    gf.GetTrueDofs(t);
    return t;
  }
};

double RelDiff(const GridFunction& a, const GridFunction& b) {
  GridFunction d(a);
  d -= b;
  return L2Norm(d) / L2Norm(b);
}

class MixedProblemFluidTest : public testing::TestWithParam<Param> {};

TEST_P(MixedProblemFluidTest, SchurAndMinresAgree) {
  const auto [dim, order] = GetParam();
  Case s(dim, order);

  auto schur = s.Problem();
  schur->SetSolverType(
      LinearQuasiStaticMixedSelfGravitatingProblem::SolverType::SchurCG);
  schur->AssembleForce(0.0);
  ASSERT_TRUE(schur->Solve());

  auto minres = s.Problem();
  minres->SetSolverType(
      LinearQuasiStaticMixedSelfGravitatingProblem::SolverType::BlockMINRES);
  minres->AssembleForce(0.0);
  ASSERT_TRUE(minres->Solve());

  EXPECT_TRUE(minres->HasFluidRegions());
  {
    const auto res = minres->RigidModeResiduals();
    std::cout << "dim " << dim << " order " << order << " residuals:";
    for (double r : res) std::cout << " " << r;
    std::cout << "\n  schur/minres diff u "
              << RelDiff(schur->Displacement(), minres->Displacement())
              << " phi " << RelDiff(schur->Potential(), minres->Potential())
              << "  outer its " << schur->LastOuterIterations() << " / "
              << minres->LastOuterIterations() << "\n";
  }
  EXPECT_GT(L2Norm(schur->Displacement()), 0.0);
  EXPECT_GT(L2Norm(schur->Potential()), 0.0);
  // Both solve the same restricted system: solver tolerance only.
  const double tol = 1e-7;
  EXPECT_LT(RelDiff(schur->Displacement(), minres->Displacement()), tol);
  EXPECT_LT(RelDiff(schur->Potential(), minres->Potential()), tol);
}

TEST(MixedProblemFluidModes, ResidualsDecreaseWithOrder) {
  for (int dim : {2, 3}) {
    std::vector<double> worst_global, worst_region, translation;
    for (int order : {1, 2}) {
      Case s(dim, order);
      auto p = s.Problem();
      const auto res = p->RigidModeResiduals();
      const int n_global = dim * (dim + 1) / 2;
      const int n_region = dim == 2 ? 1 : 3;
      ASSERT_EQ(static_cast<int>(res.size()), n_global + n_region);
      double wg = 0.0, wr = 0.0;
      for (int i = 0; i < n_global; i++) {
        EXPECT_LT(res[i], 2e-2);
        wg = std::max(wg, res[i]);
      }
      for (int i = n_global; i < n_global + n_region; i++) {
        EXPECT_LT(res[i], 2e-2);
        wr = std::max(wr, res[i]);
      }
      worst_global.push_back(wg);
      worst_region.push_back(wr);
      RigidTranslation tr(dim, dim - 1);
      translation.push_back(p->ModeResidual(s.InnerCoreMode(tr)));
      std::cout << "dim " << dim << " order " << order
                << ": worst global residual " << wg << ", region rotation "
                << wr << ", inner-core translation " << translation.back()
                << "\n";
    }
    EXPECT_LT(worst_global[1], 0.2 * worst_global[0]);
    EXPECT_LT(worst_region[1], 0.5 * worst_region[0]);

    // The inner core's translation is restored gravitationally (Slichter):
    // a soft mode, not a null one. Every residual scales with G (the
    // elastic terms vanish on rigid modes), so the distinction is in the
    // convergence: the rotations' residual vanishes with the order, the
    // translation's converges to its physical value, and their ratio grows.
    const double ratio0 = translation[0] / worst_region[0];
    const double ratio1 = translation[1] / worst_region[1];
    EXPECT_GT(ratio1, 5.0 * ratio0);
    EXPECT_GT(ratio1, 1.0);
  }
}

TEST(MixedProblemFluidModes, UnprojectedSolutionDiffersByRotation) {
  const int dim = 2, order = 2;
  Case s(dim, order);
  auto with = s.Problem(true, true);
  with->AssembleForce(0.0);
  ASSERT_TRUE(with->Solve());
  auto without = s.Problem(true, false);
  without->AssembleForce(0.0);
  ASSERT_TRUE(without->Solve());

  // Remove the inner-core rotation from the unprojected solution.
  RigidRotation rot(2, 2);
  Vector r = s.InnerCoreMode(rot);
  r /= std::sqrt(r * r);
  Vector U;
  without->Displacement().GetTrueDofs(U);
  U.Add(-(U * r), r);
  GridFunction u(s.fes_u.get());
  u.SetFromTrueDofs(U);
  const auto res = with->RigidModeResiduals();
  const double tol = 100.0 * *std::max_element(res.begin(), res.end());
  EXPECT_LT(RelDiff(u, with->Displacement()), tol);
  EXPECT_LT(RelDiff(without->Potential(), with->Potential()), tol);
}

TEST_P(MixedProblemFluidTest, TidalLoad) {
  const auto [dim, order] = GetParam();
  Case s(dim, order);
  FunctionCoefficient psi(TidalPotential);

  // Without fluids: the same as the directly assembled -rho grad psi_h . v
  // with psi_h the interpolant (on the SubMesh: nodal interpolation is
  // local, so it coincides with the parent's), up to quadrature.
  {
    std::vector<FluidRegion> none;
    auto p = s.Problem(false, false, &none);
    p->SetTidalPotential(psi);
    p->AssembleForce(0.0);
    ASSERT_TRUE(p->Solve());

    FiniteElementSpace fes_s(s.solid.get(), s.fec.get());
    GridFunction psi_h(&fes_s);
    psi_h.ProjectCoefficient(psi);
    GradientGridFunctionCoefficient grad_psi(&psi_h);
    ProductCoefficient minus_rho(-1.0, s.rho_s);
    ScalarVectorProductCoefficient load(minus_rho, grad_psi);
    auto q = s.Problem(false, false, &none);
    q->ExternalLoad().AddDomainIntegrator(new VectorDomainLFIntegrator(load));
    q->AssembleForce(0.0);
    ASSERT_TRUE(q->Solve());
    EXPECT_GT(L2Norm(p->Displacement()), 0.0);
    EXPECT_LT(RelDiff(p->Displacement(), q->Displacement()), 1e-6);
    EXPECT_LT(RelDiff(p->Potential(), q->Potential()), 1e-6);
  }

  // With fluids: linear in psi (time scaling), and the solvers agree.
  {
    auto p = s.Problem(false);
    p->SetTidalPotential(psi);
    p->AssembleForce(0.0);
    ASSERT_TRUE(p->Solve());
    GridFunction u0(p->Displacement()), phi0(p->Potential());
    EXPECT_GT(L2Norm(u0), 0.0);
    p->AssembleForce(1.0);
    ASSERT_TRUE(p->Solve());
    u0 *= 2.0;
    phi0 *= 2.0;
    EXPECT_LT(RelDiff(p->Displacement(), u0), 1e-7);
    EXPECT_LT(RelDiff(p->Potential(), phi0), 1e-7);

    auto q = s.Problem(false);
    q->SetSolverType(
        LinearQuasiStaticMixedSelfGravitatingProblem::SolverType::SchurCG);
    q->SetTidalPotential(psi);
    q->AssembleForce(1.0);
    ASSERT_TRUE(q->Solve());
    const double tol = 1e-7;
    EXPECT_LT(RelDiff(q->Displacement(), p->Displacement()), tol);
    EXPECT_LT(RelDiff(q->Potential(), p->Potential()), tol);
  }
}

TEST_P(MixedProblemFluidTest, SuppliedDensityGradient) {
  const auto [dim, order] = GetParam();
  Case s(dim, order);
  auto p = s.Problem();
  p->AssembleForce(0.0);
  ASSERT_TRUE(p->Solve());

  // rho'_F from the user's side: the L2 projection of the fluid density and
  // the problem's own background potential.
  L2_FECollection l2fec(order, dim);
  FiniteElementSpace l2(s.parent.get(), &l2fec);
  GridFunction rho_l2(&l2);
  rho_l2.ProjectCoefficient(s.rho_f);
  BarotropicDensityGradientCoefficient rho_prime(rho_l2,
                                                 p->BackgroundPotential());
  std::vector<FluidRegion> regions{OuterCore(*s.solid, s.rho_f, &rho_prime)};
  auto q = s.Problem(true, true, &regions);
  q->AssembleForce(0.0);
  ASSERT_TRUE(q->Solve());
  EXPECT_LT(RelDiff(q->Displacement(), p->Displacement()), 1e-9);
  EXPECT_LT(RelDiff(q->Potential(), p->Potential()), 1e-9);
}

TEST(MixedProblemFluidBlock, PositivityDiagnostic) {
  for (int dim : {2, 3}) {
    Case s(dim, 1);
    auto p = s.Problem(false);
    double hi = 0.0;
    const double lo = p->PotentialBlockMinEigenvalue(40, &hi);
    std::cout << "dim " << dim << ": potential block Ritz values " << lo
              << " .. " << hi << "\n";
    EXPECT_GT(lo, 0.0);
    EXPECT_GT(hi, lo);

    std::vector<FluidRegion> none;
    auto solid = s.Problem(false, false, &none);
    const double lo_solid = solid->PotentialBlockMinEigenvalue(40);
    EXPECT_GT(lo_solid, 0.0);
    EXPECT_LT(lo, lo_solid);  // the fluid mass term lowers the spectrum

    FunctionCoefficient steep(SteepFluidDensity);
    std::vector<FluidRegion> regions{OuterCore(*s.solid, steep)};
    auto q = s.Problem(false, false, &regions);
    const double lo_steep = q->PotentialBlockMinEigenvalue(40);
    std::cout << "dim " << dim << ": steep fluid, smallest Ritz value "
              << lo_steep << "\n";
    EXPECT_LT(lo_steep, lo);
  }
}

TEST(MixedProblemFluidTwoLayer, SchurAndMinresAgree) {
  // Fluid core, solid mantle: the fluid is enclosed by the solid only.
  Mesh parent("../data/elastogravity_two_layer_2d.msh", 1, 1);
  ASSERT_EQ(parent.Dimension(), 2);
  Array<int> attrs({2});
  SubMesh mantle(SubMesh::CreateFromDomain(parent, attrs));
  ASSERT_EQ(mantle.bdr_attributes.Max(), 2);
  const int order = 2;
  H1_FECollection fec(order, 2);
  FiniteElementSpace fes_u(&mantle, &fec, 2), fes_phi(&parent, &fec);
  ConstantCoefficient kappa(kKappa), mu(kMu), rho_s(1.0);
  FunctionCoefficient rho_f(FluidDensity), sigma(SurfaceLoad);
  auto rheology = IsotropicElasticRheology(2, kappa, mu);
  FluidRegion core;
  core.attributes = Array<int>({1});
  core.density = &rho_f;
  core.interface_marker = Array<int>({1, 0});
  std::vector<FluidRegion> fluids{core};
  Array<int> surface({0, 1});

  auto make = [&](LinearQuasiStaticMixedSelfGravitatingProblem::SolverType type) {
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        &fes_u, &fes_phi, rheology, rho_s, kG, kDtNDegree, nullptr, fluids);
    p->SetSurfaceLoad(sigma, surface);
    p->SetRelTol(1e-11);
    p->SetSolverType(type);
    p->AssembleForce(0.0);
    EXPECT_TRUE(p->Solve());
    return p;
  };
  auto schur =
      make(LinearQuasiStaticMixedSelfGravitatingProblem::SolverType::SchurCG);
  auto minres =
      make(LinearQuasiStaticMixedSelfGravitatingProblem::SolverType::BlockMINRES);
  EXPECT_GT(L2Norm(schur->Displacement()), 0.0);
  EXPECT_LT(RelDiff(schur->Displacement(), minres->Displacement()), 1e-7);
  EXPECT_LT(RelDiff(schur->Potential(), minres->Potential()), 1e-7);
  for (double r : minres->RigidModeResiduals()) {
    EXPECT_LT(r, 1e-3);
  }
}

TEST_P(MixedProblemFluidTest, ViscoelasticCreep) {
  const auto [dim, order] = GetParam();
  Case s(dim, order);
  ConstantCoefficient tau(1.0);
  IsotropicMaxwellRheology maxwell =
      IsotropicMaxwellRheology::Maxwell(dim, s.kappa, s.mu, tau);
  LinearQuasiStaticMixedSelfGravitatingProblem p(s.fes_u.get(), s.fes_phi.get(),
                                            maxwell, s.rho_s, kG, kDtNDegree,
                                            nullptr, s.fluids);
  p.SetSurfaceLoad(s.sigma, s.surface);
  Array<int> inner_core({1});
  p.AddRegionRotations(inner_core);
  p.SetRelTol(1e-10);

  ViscoelasticOperator op(p);
  ExponentialTrapezoidSolver stepper;
  stepper.Init(op);
  Vector m(op.Height());
  m = 0.0;
  double t = 0.0;
  double dt = 0.5;
  p.AssembleForce(0.0);
  ASSERT_TRUE(p.Solve());
  double previous = L2Norm(p.Displacement());
  EXPECT_GT(previous, 0.0);
  for (int step = 0; step < 3; step++) {
    stepper.Step(m, t, dt);
    ASSERT_TRUE(op.SolveElastic(m, t));
    const double now = L2Norm(p.Displacement());
    EXPECT_TRUE(std::isfinite(now));
    EXPECT_GT(now, previous);
    previous = now;
  }
}

TEST_P(MixedProblemFluidTest, LoadPotentialIgnoresTheFluidMass) {
  const auto [dim, order] = GetParam();
  Case c(dim, order);
  ConstantCoefficient zero(0.0);
  std::vector<FluidRegion> unstratified{OuterCore(*c.solid, c.rho_f, &zero)};
  auto with = c.Problem();
  auto without = c.Problem(true, true, &unstratified);
  GridFunction phi_with(&with->PotentialSpaceOnBody());
  GridFunction phi_without(&without->PotentialSpaceOnBody());
  with->AssembleForce(0.0);
  without->AssembleForce(0.0);
  EXPECT_TRUE(with->SolveLoadPotential(phi_with));
  EXPECT_TRUE(without->SolveLoadPotential(phi_without));
  EXPECT_LT(RelDiff(phi_with, phi_without), 1e-8);
  // and the problem's own solves still use the full potential block
  ASSERT_TRUE(with->Solve());
  ASSERT_TRUE(without->Solve());
  EXPECT_GT(RelDiff(with->Potential(), without->Potential()), 1e-4);
}

// The approximate CMB conditions of the GIA literature are degenerate
// cases of the FluidRegion machinery (doc/self_gravitation.md, "The CMB
// approximations of the GIA literature"; the ladder is defined in
// doc/quasi_static_models.tex, "Fluid regions and the ladder of CMB
// approximations"): 'uniform' = constant interface density with the fluid
// mass term dropped, 'winkler' = uniform without the interface potential
// coupling (F3). Both stay symmetric (the two solvers agree), and both
// move the answer.
TEST(MixedProblemFluidCMB, ApproximateConditions) {
  Case c(2, 2);
  ConstantCoefficient zero(0.0);
  ConstantCoefficient rho_c(1.1);  // the core-top density of the profile

  auto full = c.Problem();
  full->AssembleForce(0.0);
  ASSERT_TRUE(full->Solve());
  GridFunction phi_full(full->Potential());

  auto make = [&](bool coupling) {
    std::vector<FluidRegion> regions;
    auto f = OuterCore(*c.solid, c.rho_f);
    f.density_gradient = &zero;
    f.interface_density = &rho_c;
    f.interface_potential_coupling = coupling;
    regions.push_back(f);
    return c.Problem(true, true, &regions);
  };

  auto uniform = make(true);
  uniform->AssembleForce(0.0);
  ASSERT_TRUE(uniform->Solve());
  GridFunction phi_uniform(uniform->Potential());

  auto winkler = make(false);
  winkler->AssembleForce(0.0);
  ASSERT_TRUE(winkler->Solve());
  GridFunction phi_winkler(winkler->Potential());

  auto rel_diff = [](const GridFunction& a, const GridFunction& b) {
    GridFunction d(a);
    d -= b;
    return L2Norm(d) / L2Norm(b);
  };

  // Dropping (F3) must remove both halves: dropping only one would break
  // the symmetry the two solvers rely on, so their agreement is the
  // sharpest check.
  winkler->SetSolverType(
      LinearQuasiStaticMixedSelfGravitatingProblem::SolverType::SchurCG);
  winkler->AssembleForce(0.0);
  ASSERT_TRUE(winkler->Solve());
  EXPECT_LT(rel_diff(winkler->Potential(), phi_winkler), 1e-6);

  // The approximations are not no-ops on the stratified test model.
  EXPECT_GT(rel_diff(phi_full, phi_uniform), 1e-3);
  EXPECT_GT(rel_diff(phi_uniform, phi_winkler), 1e-3);
}

INSTANTIATE_TEST_SUITE_P(MixedProblemFluid, MixedProblemFluidTest,
                         testing::Values(Param{2, 1}, Param{2, 2},
                                         Param{3, 1}));

}  // namespace
