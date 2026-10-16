#include "QuasiStaticTestCommon.hpp"
#include "TestCommon.hpp"

/*
  Tests for the pressure-split (u, p) solve of the clamped quasi-static
  problem (SetPressureSplitSolver; see doc/planning/solvers.md,
  "Near-incompressible and high-contrast effective operators").

  - Constant strain: with u = A x prescribed on the whole boundary the
    Taylor-Hood pair represents the state exactly, so the split solve
    reproduces the strain and the pressure p = kappa tr(A) to solver
    tolerance — the sharp gate on the block system, the boundary lift and
    the sign conventions.
  - Agreement with the primal solve: on a smooth inhomogeneous state (a
    clamped beam under a surface pull) the two discretisations differ at
    discretisation-error level, and their gap shrinks under refinement.
  - Modulus switches: setting, clearing and re-setting relaxation weights
    reassembles the blocks through the live effective coefficient; the
    cleared solve matches a never-weighted one to solver tolerance (the
    stale-cache gate for the bypassed flavour slots), on a two-region
    composite.
  - High contrast: with the weak region's branch weight at 1e-6 (the
    collapsed effective shear of dt >> tau) the split solve converges and
    agrees with the primal one; iteration counts are printed for the
    measured comparison in solvers.md.
  - Viscoelastic stepping: under a prescribed uniform strain the
    displacement stays u = A x at every time (constant stress divergence),
    so backward Euler through the split route is exact in space — the gate
    on the stepping integration (ImplicitSolve's weight switches ride the
    split reassembly).
*/

namespace {

using namespace elastic_test;

using Param = std::tuple<int, int>;  // (dim, elementType)

double RelMaxDiff(const Vector& a, const Vector& b) {
  Vector d(a);
  d -= b;
  return d.Normlinf() / (b.Normlinf() + 1e-300);
}

// u = A x on the whole boundary (and, where exact, everywhere).
DenseMatrix StrainMatrix(int dim) {
  DenseMatrix A(dim);
  A = 0.0;
  A(0, 0) = 0.01;
  A(0, 1) = 0.004;
  A(1, 0) = -0.002;
  A(1, 1) = -0.006;
  if (dim == 3) {
    A(2, 2) = 0.003;
    A(0, 2) = 0.005;
  }
  return A;
}

double MaxFieldError(const GridFunction& u, const DenseMatrix& A) {
  VectorFunctionCoefficient exact(
      A.Height(), [&](const Vector& x, Vector& v) { A.Mult(x, v); });
  return u.ComputeMaxError(exact);
}

// Two horizontal layers: element attribute 1 below mid-height, 2 above.
void TwoLayerAttributes(Mesh& mesh) {
  const int dim = mesh.Dimension();
  Vector c;
  for (int e = 0; e < mesh.GetNE(); e++) {
    auto* T = mesh.GetElementTransformation(e);
    T->Transform(Geometries.GetCenter(T->GetGeometryType()), c);
    mesh.SetAttribute(e, c[dim - 1] < 0.5 ? 1 : 2);
  }
  mesh.SetAttributes();
}

class PressureSplitTest : public ::testing::TestWithParam<Param> {
 protected:
  void SetUp() override {
    std::tie(dim, elementType) = GetParam();
    mesh = std::make_unique<Mesh>(MakeSmallMesh(dim, elementType));
    fec = std::make_unique<H1_FECollection>(kOrder, dim);
    fes = std::make_unique<FiniteElementSpace>(mesh.get(), fec.get(), dim);
    kappa = std::make_unique<ConstantCoefficient>(Kappa(dim));
    mu = std::make_unique<ConstantCoefficient>(kMu);
    bottom_attr = BdrAttributeAt(*mesh, dim - 1, 0.0);
    top_attr = BdrAttributeAt(*mesh, dim - 1, 1.0);
    nbdr = mesh->bdr_attributes.Max();
  }

  static constexpr int kOrder = 2;

  int dim = 2, elementType = 0, bottom_attr = -1, top_attr = -1, nbdr = 0;
  std::unique_ptr<Mesh> mesh;
  std::unique_ptr<FiniteElementCollection> fec;
  std::unique_ptr<FiniteElementSpace> fes;
  std::unique_ptr<ConstantCoefficient> kappa, mu;
};

TEST_P(PressureSplitTest, ConstantStrainExact) {
  IsotropicElasticRheology rheology(dim, *kappa, *mu);
  const DenseMatrix A = StrainMatrix(dim);
  VectorFunctionCoefficient dirichlet(
      dim, [&](const Vector& x, Vector& u) { A.Mult(x, u); });
  Vector zero(dim);
  zero = 0.0;
  VectorConstantCoefficient no_traction(zero);
  Array<int> all(nbdr), none(nbdr);
  all = 1;
  none = 0;
  LinearQuasiStaticClampedProblem problem(fes.get(), rheology, all,
                                          no_traction, none, &dirichlet);
  problem.SetPressureSplitSolver();
  EXPECT_TRUE(problem.UsingPressureSplit());
  problem.AssembleForce(0.0);
  ASSERT_TRUE(problem.Solve());
  EXPECT_LT(MaxFieldError(problem.Displacement(), A), 1e-9);

  // The pressure is kappa tr(A) everywhere (the sign-convention gate).
  double tr = 0.0;
  for (int i = 0; i < dim; i++) {
    tr += A(i, i);
  }
  ConstantCoefficient p_exact(Kappa(dim) * tr);
  ASSERT_NE(problem.Pressure(), nullptr);
  EXPECT_LT(problem.Pressure()->ComputeMaxError(p_exact),
            1e-9 * std::abs(Kappa(dim) * tr));
}

TEST_P(PressureSplitTest, MatchesPrimalUnderRefinement) {
  // A clamped beam under a surface pull: smooth and inhomogeneous, so the
  // two discretisations differ at discretisation-error level only, and
  // the gap shrinks under refinement.
  auto gap = [&](int refinements) {
    Mesh m(MakeSmallMesh(dim, elementType));
    for (int r = 0; r < refinements; r++) {
      m.UniformRefinement();
    }
    H1_FECollection fc(kOrder, dim);
    FiniteElementSpace fs(&m, &fc, dim);
    IsotropicElasticRheology rheology(dim, *kappa, *mu);
    VectorFunctionCoefficient pull(dim, PullTraction);
    const Array<int> clamp = Marker(nbdr, {bottom_attr});
    const Array<int> top = Marker(nbdr, {top_attr});

    LinearQuasiStaticClampedProblem primal(&fs, rheology, clamp, pull, top);
    primal.SetRelTol(1e-12);
    primal.AssembleForce(0.0);
    EXPECT_TRUE(primal.Solve());

    LinearQuasiStaticClampedProblem split(&fs, rheology, clamp, pull, top);
    split.SetPressureSplitSolver();
    split.SetRelTol(1e-12);
    split.AssembleForce(0.0);
    EXPECT_TRUE(split.Solve());

    return RelMaxDiff(split.Displacement(), primal.Displacement());
  };
  const double coarse = gap(0);
  const double fine = gap(1);
  EXPECT_LT(coarse, 0.05);
  EXPECT_LT(fine, 0.7 * coarse);
}

// Two-region Maxwell composite under a surface pull, with per-branch
// weights beta: the modulus-switch and high-contrast gates.
TEST_P(PressureSplitTest, ModulusSwitchesAndHighContrast) {
  TwoLayerAttributes(*mesh);
  ConstantCoefficient mu_inf(0.0), mu1(kMu), mu2(0.6 * kMu);
  ConstantCoefficient tau1(1.0), tau2(2.0);
  std::vector<MaxwellBranch> b1{{&mu1, &tau1}}, b2{{&mu2, &tau2}};
  IsotropicMaxwellRheology r1(dim, *kappa, mu_inf, b1);
  IsotropicMaxwellRheology r2(dim, *kappa, mu_inf, b2);
  const int na = mesh->attributes.Max();
  std::vector<RheologyRegion> regions{{Marker(na, {1}), &r1, "healthy"},
                                      {Marker(na, {2}), &r2, "weak"}};
  CompositeRheology rheology(dim, regions);

  VectorFunctionCoefficient pull(dim, PullTraction);
  const Array<int> clamp = Marker(nbdr, {bottom_attr});
  const Array<int> top = Marker(nbdr, {top_attr});
  auto make = [&](bool split_route) {
    auto p = std::make_unique<LinearQuasiStaticClampedProblem>(
        fes.get(), rheology, clamp, pull, top);
    if (split_route) {
      p->SetPressureSplitSolver();
    }
    p->SetRelTol(1e-12);
    return p;
  };
  auto solve = [&](LinearQuasiStaticProblemBase& p) {
    p.AssembleForce(0.0);
    EXPECT_TRUE(p.Solve());
    return Vector(p.Displacement());
  };

  auto split = make(true);
  auto primal = make(false);
  const Vector u0 = solve(*split);
  const Vector u0_primal = solve(*primal);
  EXPECT_LT(RelMaxDiff(u0, u0_primal), 0.05);

  // Moderate contrast: both discretisations are sound, so they agree at
  // discretisation level.
  ConstantCoefficient beta_healthy(1.0), beta_mild(0.3), beta_weak(1e-6);
  const std::vector<Coefficient*> mild{&beta_healthy, &beta_mild};
  split->SetRelaxationWeights(mild);
  primal->SetRelaxationWeights(mild);
  // 0.1, not the solver floor: the Taylor-Hood pressure is continuous
  // across the layer interface where the true pressure jumps, a genuine
  // discretisation difference on this deliberately coarse mesh.
  EXPECT_LT(RelMaxDiff(solve(*split), solve(*primal)), 0.10);

  // Extreme contrast (the collapsed effective shear of dt >> tau): the
  // primal discretisation LOCKS at kappa/mu_eff ~ 1e6+ — the pathology
  // the split removes — so no agreement gate here, only robustness; the
  // iteration counts feed solvers.md.
  const std::vector<Coefficient*> beta{&beta_healthy, &beta_weak};
  split->SetRelaxationWeights(beta);
  primal->SetRelaxationWeights(beta);
  const long its_before_split = split->TotalIterations();
  const long its_before_primal = primal->TotalIterations();
  const Vector ue = solve(*split);
  solve(*primal);
  std::cout << "contrast 1e6, dim " << dim << ", elementType " << elementType
            << ": split " << split->TotalIterations() - its_before_split
            << " iterations, primal "
            << primal->TotalIterations() - its_before_primal << "\n";

  // Cleared weights reproduce the unrelaxed solve (the stale-cache gate),
  // and re-set ones the effective solve, to the linear-solver floor
  // (1e-5: headroom over the serial Jacobi-MINRES floor; a stale matrix
  // would miss by O(1)).
  split->ClearRelaxationWeights();
  EXPECT_LT(RelMaxDiff(solve(*split), u0), 1e-5);
  split->SetRelaxationWeights(beta);
  EXPECT_LT(RelMaxDiff(solve(*split), ue), 1e-5);
}

// A purely elastic region beside a viscoelastic one — the production
// shape (elastic lithosphere over a Maxwell mantle): the composite's
// piecewise split must carry the elastic region's fixed (kappa, mu) next
// to the Maxwell region's live effective modulus.
TEST_P(PressureSplitTest, ElasticBesideViscoelastic) {
  TwoLayerAttributes(*mesh);
  ConstantCoefficient mu_inf(0.0), mu1(kMu), tau1(1.0), mu_lid(1.5 * kMu);
  std::vector<MaxwellBranch> b1{{&mu1, &tau1}};
  IsotropicMaxwellRheology mantle(dim, *kappa, mu_inf, b1);
  IsotropicElasticRheology lid(dim, *kappa, mu_lid);
  const int na = mesh->attributes.Max();
  std::vector<RheologyRegion> regions{{Marker(na, {1}), &mantle, "mantle"},
                                      {Marker(na, {2}), &lid, "lid"}};
  CompositeRheology rheology(dim, regions);

  VectorFunctionCoefficient pull(dim, PullTraction);
  const Array<int> clamp = Marker(nbdr, {bottom_attr});
  const Array<int> top = Marker(nbdr, {top_attr});
  auto make = [&](bool split_route) {
    auto p = std::make_unique<LinearQuasiStaticClampedProblem>(
        fes.get(), rheology, clamp, pull, top);
    if (split_route) {
      p->SetPressureSplitSolver();
    }
    p->SetRelTol(1e-12);
    return p;
  };
  auto solve = [&](LinearQuasiStaticProblemBase& p) {
    p.AssembleForce(0.0);
    EXPECT_TRUE(p.Solve());
    return Vector(p.Displacement());
  };
  auto split = make(true);
  auto primal = make(false);
  const Vector u0 = solve(*split);
  EXPECT_LT(RelMaxDiff(u0, solve(*primal)), 0.10);
  // The relaxed mantle under the untouched elastic lid (one weight: the
  // composite has one branch; mild, so the primal does not lock), and back.
  ConstantCoefficient beta(0.3);
  const std::vector<Coefficient*> weights{&beta};
  split->SetRelaxationWeights(weights);
  primal->SetRelaxationWeights(weights);
  // 0.2: the softened mantle sharpens the interface pressure jump the
  // continuous Taylor-Hood pressure smears on this coarse mesh.
  EXPECT_LT(RelMaxDiff(solve(*split), solve(*primal)), 0.20);
  split->ClearRelaxationWeights();
  EXPECT_LT(RelMaxDiff(solve(*split), u0), 1e-5);
}

TEST_P(PressureSplitTest, ViscoelasticSteppingExact) {
  // Prescribed uniform strain: the displacement stays u = A x at every
  // time, exactly representable, so the stepping chain through the split
  // route (ImplicitSolve's weight switches included) is exact in space.
  ConstantCoefficient mu_inf(0.3), mu1(0.7), tau1(1.0);
  std::vector<MaxwellBranch> branches{{&mu1, &tau1}};
  IsotropicMaxwellRheology rheology(dim, *kappa, mu_inf, branches);
  const DenseMatrix A = StrainMatrix(dim);
  VectorFunctionCoefficient dirichlet(
      dim, [&](const Vector& x, Vector& u) { A.Mult(x, u); });
  Vector zero(dim);
  zero = 0.0;
  VectorConstantCoefficient no_traction(zero);
  Array<int> all(nbdr), none(nbdr);
  all = 1;
  none = 0;
  LinearQuasiStaticClampedProblem problem(fes.get(), rheology, all,
                                          no_traction, none, &dirichlet);
  problem.SetPressureSplitSolver();
  problem.SetRelTol(1e-12);
  ViscoelasticOperator visco(problem);
  BackwardEulerSolver ode;
  ode.Init(visco);
  Vector m(visco.Height());
  m = 0.0;
  double t = 0.0, dt = 0.5;
  for (int step = 0; step < 4; step++) {
    ode.Step(m, t, dt);
  }
  ASSERT_TRUE(visco.SolveElastic(m, t));
  EXPECT_LT(MaxFieldError(problem.Displacement(), A), 1e-9);
}

// The route refuses what it does not carry.
TEST_P(PressureSplitTest, Refusals) {
  auto rheology = IsotropicMaxwellRheology::Maxwell(dim, *kappa, *mu, *mu);
  VectorFunctionCoefficient traction(dim, [](const Vector&, Vector& f) {
    f = 0.0;
  });
  const Array<int> marker = Marker(nbdr, {top_attr});
  LinearQuasiStaticTractionProblem traction_problem(fes.get(), rheology,
                                                    traction, marker);
  EXPECT_DEATH(traction_problem.SetPressureSplitSolver(), "clamped solid");

  // Order 1 is not a Taylor-Hood pair.
  H1_FECollection fec1(1, dim);
  FiniteElementSpace fes1(mesh.get(), &fec1, dim);
  IsotropicElasticRheology elastic(dim, *kappa, *mu);
  Array<int> all(nbdr), none(nbdr);
  all = 1;
  none = 0;
  LinearQuasiStaticClampedProblem low(&fes1, elastic, all, traction, none);
  EXPECT_DEATH(low.SetPressureSplitSolver(), "order");

  // An anisotropic rheology has no isotropic split.
  auto C = IsotropicElasticTensorCoefficient::FromBulkModulus(dim, *kappa, *mu);
  AnisotropicElasticRheology aniso(dim, C);
  LinearQuasiStaticClampedProblem tensor(fes.get(), aniso, all, traction, none);
  EXPECT_DEATH(tensor.SetPressureSplitSolver(), "isotropic");
}

INSTANTIATE_TEST_SUITE_P(
    PressureSplit, PressureSplitTest,
    ::testing::Combine(::testing::Values(2, 3), ::testing::Values(0, 1)),
    [](const ::testing::TestParamInfo<Param>& info) {
      return std::to_string(std::get<0>(info.param)) + "D_" +
             (std::get<1>(info.param) == 0 ? "simplex" : "tensor");
    });

}  // namespace
