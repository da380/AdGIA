#include <numbers>

#include "QuasiStaticTestCommon.hpp"
#include "TestCommon.hpp"

/*
  Tests for LinearQuasiStaticProblemBase, LinearQuasiStaticTractionProblem and
  LinearQuasiStaticClampedProblem, against direct MFEM assemblies and solves
  and the exact uniaxial-stress solution.

  - The bulk/deviatoric split assembled by the base class equals MFEM's
    ElasticityIntegrator(lambda, mu) with lambda = kappa - 2 mu / d.
  - LinearQuasiStaticClampedProblem reproduces a direct MFEM assembly and solve.
  - SetRelaxationWeights (constant and piecewise-constant L2 field) on a
    Maxwell rheology reproduces a direct assembly with the modulus beta mu;
    Clear restores mu_U.
  - Preconditioner reuse across reassemblies keeps the solutions exact and
    reduces the number of setups.
  - The anisotropic rheology with an isotropic tensor reproduces the
    isotropic problem, relaxed or not.
  - LinearQuasiStaticTractionProblem under uniaxial stress gives the exact
    constant strain.
  - SetMassWeightedGauge selects the solution with zero rho-weighted linear
    and angular momentum (the Euclidean-gauge solution does not have it),
    and SetEuclideanGauge restores the Euclidean one.
  - Loads scale with time through the registered coefficients, and
    AddForce superposes exactly like an integrator on the load.
  - The non-natural reference state (ReferentialElasticRheology with an
    equilibrium mapping): the traction problem on the reference mesh with
    the interpolated mapping and the Nanson-composed traction equals, dof
    for dof, the identity-mapped problem on the nodal-image mesh (the
    discrete change-of-variables identity at problem level), and the
    projector's mapped rotations are discrete null vectors of the mapped
    stiffness (RigidPairResiduals).
*/

namespace {

using namespace elastic_test;

// (dim, elementType, order)
using Param = std::tuple<int, int, int>;

class ElasticProblemTest : public testing::TestWithParam<Param> {
 protected:
  void SetUp() override {
    std::tie(dim, elementType, order) = GetParam();
    mesh = std::make_unique<Mesh>(MakeSmallMesh(dim, elementType));
    fec = std::make_unique<H1_FECollection>(order, dim);
    fes = std::make_unique<FiniteElementSpace>(mesh.get(), fec.get(), dim);
    kappa = std::make_unique<ConstantCoefficient>(Kappa(dim));
    mu = std::make_unique<ConstantCoefficient>(kMu);
    lambda = std::make_unique<ConstantCoefficient>(kLambda);
    rheology = std::make_unique<IsotropicElasticRheology>(dim, *kappa, *mu);
    x0_attr = BdrAttributeAt(*mesh, 0, 0.0);
    x1_attr = BdrAttributeAt(*mesh, 0, 1.0);
    nbdr = mesh->bdr_attributes.Max();
  }

  // Direct MFEM solve of the clamped problem with (lambda, mu) and the
  // load b at time t (b already assembled), to tight tolerance.
  GridFunction DirectClamped(Coefficient& lam, Coefficient& m,
                             const Array<int>& ess_bdr, LinearForm& b) {
    BilinearForm a(fes.get());
    a.AddDomainIntegrator(new ElasticityIntegrator(lam, m));
    a.Assemble();
    Array<int> ess_bdr_copy(ess_bdr), ess_tdof;
    fes->GetEssentialTrueDofs(ess_bdr_copy, ess_tdof);
    GridFunction u(fes.get());
    u = 0.0;
    SparseMatrix A;
    Vector X, B;
    a.FormLinearSystem(ess_tdof, u, b, A, X, B);
    GSSmoother prec(A);
    CGSolver cg;
    cg.SetPreconditioner(prec);
    cg.SetOperator(A);
    cg.SetRelTol(1e-14);
    cg.SetAbsTol(0.0);
    cg.SetMaxIter(20000);
    cg.SetPrintLevel(0);
    cg.Mult(B, X);
    EXPECT_TRUE(cg.GetConverged());
    a.RecoverFEMSolution(X, b, u);
    return u;
  }

  static double RelMaxDiff(const Vector& a, const Vector& b) {
    Vector d(a);
    d -= b;
    return d.Normlinf() / (b.Normlinf() + 1e-300);
  }

  int dim = 2, elementType = 0, order = 1, x0_attr = -1, x1_attr = -1, nbdr = 0;
  std::unique_ptr<Mesh> mesh;
  std::unique_ptr<FiniteElementCollection> fec;
  std::unique_ptr<FiniteElementSpace> fes;
  std::unique_ptr<ConstantCoefficient> kappa, mu, lambda;
  std::unique_ptr<IsotropicElasticRheology> rheology;
};

TEST_P(ElasticProblemTest, SplitIdentity) {
  BilinearForm split(fes.get());
  split.AddDomainIntegrator(new ElasticityIntegrator(*kappa, 1.0, 0.0));
  split.AddDomainIntegrator(new ElasticityIntegrator(*mu, -2.0 / dim, 1.0));
  split.Assemble();
  split.Finalize();

  BilinearForm ref(fes.get());
  ref.AddDomainIntegrator(new ElasticityIntegrator(rheology->Lame(), *mu));
  ref.Assemble();
  ref.Finalize();

  EXPECT_LT(MaxDiff(split.SpMat(), ref.SpMat()), 1e-13 * ref.SpMat().MaxNorm());
}

TEST_P(ElasticProblemTest, ClampedMatchesDirect) {
  auto ess_bdr = Marker(nbdr, {x0_attr});
  auto pull_marker = Marker(nbdr, {x1_attr});
  VectorFunctionCoefficient traction(dim, PullTraction);

  LinearQuasiStaticClampedProblem problem(fes.get(), *rheology, ess_bdr,
                                          traction, pull_marker);
  EXPECT_EQ(&problem.DisplacementSpace(), fes.get());
  EXPECT_EQ(&problem.Rheology(), rheology.get());
  EXPECT_TRUE(problem.SupportsRelaxationWeights());
  EXPECT_FALSE(problem.IsParallel());

  problem.AssembleForce(0.0);
  ASSERT_TRUE(problem.Solve());

  LinearForm b(fes.get());
  traction.SetTime(0.0);
  b.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(traction),
                          pull_marker);
  b.Assemble();
  auto u_ref = DirectClamped(*lambda, *mu, ess_bdr, b);
  ASSERT_GT(u_ref.Normlinf(), 0.0);
  EXPECT_LT(RelMaxDiff(problem.Displacement(), u_ref), 1e-8);

  // A second solve at the same time warm-starts and must return the same.
  problem.AssembleForce(0.0);
  ASSERT_TRUE(problem.Solve());
  EXPECT_LT(RelMaxDiff(problem.Displacement(), u_ref), 1e-8);
}

TEST_P(ElasticProblemTest, RelaxationWeights) {
  auto ess_bdr = Marker(nbdr, {x0_attr});
  auto pull_marker = Marker(nbdr, {x1_attr});
  VectorFunctionCoefficient traction(dim, PullTraction);
  // Maxwell body: mu_inf = 0, one branch mu, so mu_eff = beta mu.
  ConstantCoefficient tau(1.0);
  auto maxwell = IsotropicMaxwellRheology::Maxwell(dim, *kappa, *mu, tau);
  LinearQuasiStaticClampedProblem problem(fes.get(), maxwell, ess_bdr, traction,
                                          pull_marker);
  EXPECT_FALSE(problem.Stiffness().IsRelaxed());

  LinearForm b(fes.get());
  traction.SetTime(0.0);
  b.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(traction),
                          pull_marker);
  b.Assemble();

  // Constant weight.
  ConstantCoefficient beta(0.4), mu_eff(0.4 * kMu);
  SumCoefficient lambda_eff(*kappa, mu_eff, 1.0, -2.0 / dim);
  problem.SetRelaxationWeights({&beta});
  EXPECT_TRUE(problem.Stiffness().IsRelaxed());
  problem.AssembleForce(0.0);
  ASSERT_TRUE(problem.Solve());
  auto u_ref = DirectClamped(lambda_eff, mu_eff, ess_bdr, b);
  EXPECT_LT(RelMaxDiff(problem.Displacement(), u_ref), 1e-8);

  // Piecewise-constant weight given as a GridFunctionCoefficient on an L2
  // space, as the viscoelastic layer supplies it.
  L2_FECollection l2fec(0, dim);
  FiniteElementSpace sfes(mesh.get(), &l2fec);
  GridFunction beta_field(&sfes);
  FunctionCoefficient beta_var(
      [](const Vector& x) { return 0.3 + 0.6 * x[0]; });
  beta_field.ProjectCoefficient(beta_var);
  GridFunctionCoefficient beta_gf(&beta_field);
  ProductCoefficient mu_eff_gf(beta_gf, *mu);
  SumCoefficient lambda_eff_gf(*kappa, mu_eff_gf, 1.0, -2.0 / dim);
  problem.SetRelaxationWeights({&beta_gf});
  problem.AssembleForce(0.0);
  ASSERT_TRUE(problem.Solve());
  auto u_ref_gf = DirectClamped(lambda_eff_gf, mu_eff_gf, ess_bdr, b);
  EXPECT_LT(RelMaxDiff(problem.Displacement(), u_ref_gf), 1e-8);
  EXPECT_GT(RelMaxDiff(u_ref_gf, u_ref), 1e-3);  // the two cases differ

  // Clear restores mu_U.
  problem.ClearRelaxationWeights();
  EXPECT_FALSE(problem.Stiffness().IsRelaxed());
  problem.AssembleForce(0.0);
  ASSERT_TRUE(problem.Solve());
  auto u_ref_u = DirectClamped(*lambda, *mu, ess_bdr, b);
  EXPECT_LT(RelMaxDiff(problem.Displacement(), u_ref_u), 1e-8);
}

TEST_P(ElasticProblemTest, PreconditionerReuse) {
  auto ess_bdr = Marker(nbdr, {x0_attr});
  auto pull_marker = Marker(nbdr, {x1_attr});
  VectorFunctionCoefficient traction(dim, PullTraction);
  ConstantCoefficient tau(1.0);
  auto maxwell = IsotropicMaxwellRheology::Maxwell(dim, *kappa, *mu, tau);
  LinearForm b(fes.get());
  traction.SetTime(0.0);
  b.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(traction),
                          pull_marker);
  b.Assemble();

  for (double reuse : {2.0, 1.0}) {
    LinearQuasiStaticClampedProblem problem(fes.get(), maxwell, ess_bdr,
                                            traction, pull_marker);
    problem.SetPreconditionerReuse(reuse);
    problem.AssembleForce(0.0);
    ASSERT_TRUE(problem.Solve());
    EXPECT_EQ(problem.NumPreconditionerSetups(), 1);
    // Small drifts of the weights: the preconditioner is kept (or rebuilt
    // every time without reuse); the solutions are the exact ones.
    int assemblies = 1;
    for (double bv : {0.9, 0.8, 0.7}) {
      ConstantCoefficient beta(bv), mu_eff(bv * kMu);
      SumCoefficient lambda_eff(*kappa, mu_eff, 1.0, -2.0 / dim);
      problem.SetRelaxationWeights({&beta});
      ASSERT_TRUE(problem.Solve());
      assemblies++;
      auto u_ref = DirectClamped(lambda_eff, mu_eff, ess_bdr, b);
      EXPECT_LT(RelMaxDiff(problem.Displacement(), u_ref), 1e-8);
    }
    if (reuse > 1.0) {
      EXPECT_LT(problem.NumPreconditionerSetups(), assemblies);
    } else {
      EXPECT_EQ(problem.NumPreconditionerSetups(), assemblies);
    }
    // A large drift makes the iteration count grow (with reuse, the
    // preconditioner is rebuilt at the following assembly); the solution
    // must still be the exact one.
    ConstantCoefficient tiny(0.01), mu_tiny(0.01 * kMu);
    SumCoefficient lambda_tiny(*kappa, mu_tiny, 1.0, -2.0 / dim);
    problem.SetRelaxationWeights({&tiny});
    ASSERT_TRUE(problem.Solve());
    auto u_ref = DirectClamped(lambda_tiny, mu_tiny, ess_bdr, b);
    EXPECT_LT(RelMaxDiff(problem.Displacement(), u_ref), 1e-8);
    // The next assembly, a small further drift that would not by itself
    // force a rebuild, rebuilds the preconditioner: the large drift marked
    // it stale.
    const int setups = problem.NumPreconditionerSetups();
    ConstantCoefficient near(0.011), mu_near(0.011 * kMu);
    SumCoefficient lambda_near(*kappa, mu_near, 1.0, -2.0 / dim);
    problem.SetRelaxationWeights({&near});
    ASSERT_TRUE(problem.Solve());
    EXPECT_EQ(problem.NumPreconditionerSetups(), setups + 1);
    auto u_near = DirectClamped(lambda_near, mu_near, ess_bdr, b);
    EXPECT_LT(RelMaxDiff(problem.Displacement(), u_near), 1e-8);
  }
}

TEST_P(ElasticProblemTest, AnisotropicMatchesIsotropic) {
  auto ess_bdr = Marker(nbdr, {x0_attr});
  auto pull_marker = Marker(nbdr, {x1_attr});
  VectorFunctionCoefficient traction(dim, PullTraction);
  ConstantCoefficient tau(1.0), beta(0.4);

  auto iso = IsotropicMaxwellRheology::Maxwell(dim, *kappa, *mu, tau);
  auto C = IsotropicElasticTensorCoefficient::FromBulkModulus(dim, *kappa, *mu);
  auto aniso = AnisotropicMaxwellRheology::DeviatoricMaxwell(dim, C, tau);
  EXPECT_FALSE(aniso.TraceFreeInternalVariables());
  EXPECT_EQ(aniso.NumBranches(), 1);

  LinearQuasiStaticClampedProblem a(fes.get(), iso, ess_bdr, traction,
                                    pull_marker);
  LinearQuasiStaticClampedProblem b(fes.get(), aniso, ess_bdr, traction,
                                    pull_marker);
  for (int relaxed = 0; relaxed < 2; relaxed++) {
    if (relaxed) {
      a.SetRelaxationWeights({&beta});
      b.SetRelaxationWeights({&beta});
    }
    a.AssembleForce(0.0);
    b.AssembleForce(0.0);
    ASSERT_TRUE(a.Solve());
    ASSERT_TRUE(b.Solve());
    EXPECT_GT(a.Displacement().Normlinf(), 0.0);
    EXPECT_LT(RelMaxDiff(b.Displacement(), a.Displacement()), 1e-10);
  }
}

TEST_P(ElasticProblemTest, TractionUniaxialStrain) {
  auto marker = Marker(nbdr, {x0_attr, x1_attr});
  VectorFunctionCoefficient traction(dim, UniaxialTraction);
  LinearQuasiStaticTractionProblem problem(fes.get(), *rheology, traction,
                                           marker);

  double exx = 0.0, eyy = 0.0;
  UniaxialStrain(dim, kSigma, exx, eyy);

  problem.AssembleForce(0.0);
  ASSERT_TRUE(problem.Solve());
  EXPECT_LT(MaxStrainError(problem.Displacement(), exx, eyy), 1e-8 * exx);

  // Doubling the load (t = 1) doubles the strain, from a warm start.
  problem.AssembleForce(1.0);
  ASSERT_TRUE(problem.Solve());
  EXPECT_LT(MaxStrainError(problem.Displacement(), 2.0 * exx, 2.0 * eyy),
            1e-8 * exx);
}

TEST_P(ElasticProblemTest, TractionMassWeightedGauge) {
  auto marker = Marker(nbdr, {x0_attr, x1_attr});
  VectorFunctionCoefficient traction(dim, UniaxialTraction);
  LinearQuasiStaticTractionProblem problem(fes.get(), *rheology, traction,
                                           marker);
  problem.AssembleForce(0.0);
  ASSERT_TRUE(problem.Solve());
  GridFunction u_euclid(problem.Displacement());

  FunctionCoefficient rho([](const Vector& x) { return 1.0 + x[0]; });
  problem.SetMassWeightedGauge(&rho);
  ASSERT_TRUE(problem.Solve());
  const GridFunction& u = problem.Displacement();
  double exx = 0.0, eyy = 0.0;
  UniaxialStrain(dim, kSigma, exx, eyy);
  EXPECT_LT(MaxStrainError(u, exx, eyy), 1e-8 * exx);

  // Zero momentum and angular momentum against the rigid modes.
  BilinearForm mass(fes.get());
  mass.AddDomainIntegrator(new VectorMassIntegrator(rho));
  mass.Assemble();
  mass.Finalize();
  Vector Mu(u.Size());
  mass.SpMat().Mult(u, Mu);
  auto P = MakeRigidModeProjector(*fes);
  double moment = 0.0, moment_euclid = 0.0;
  Vector Mue(u.Size());
  mass.SpMat().Mult(u_euclid, Mue);
  for (int i = 0; i < P->Size(); i++) {
    moment = std::max(moment, std::abs(P->Dot(Mu, P->Basis(i))));
    moment_euclid =
        std::max(moment_euclid, std::abs(P->Dot(Mue, P->Basis(i))));
  }
  EXPECT_LT(moment, 1e-12 * Mu.Norml2());
  EXPECT_GT(moment_euclid, 1e-6 * Mue.Norml2());

  problem.SetEuclideanGauge();
  ASSERT_TRUE(problem.Solve());
  GridFunction d(problem.Displacement());
  d -= u_euclid;
  EXPECT_LT(d.Normlinf(), 1e-12 * u_euclid.Normlinf());
}

TEST_P(ElasticProblemTest, AddForceSuperposes) {
  auto ess_bdr = Marker(nbdr, {x0_attr});
  auto pull_marker = Marker(nbdr, {x1_attr});
  VectorFunctionCoefficient traction(dim, PullTraction);
  Vector g(dim);
  g = 0.0;
  g[0] = 0.1;
  VectorConstantCoefficient body(g);

  // Problem 1: body force through AddForce.
  LinearQuasiStaticClampedProblem p1(fes.get(), *rheology, ess_bdr, traction,
                                     pull_marker);
  LinearForm extra(fes.get());
  extra.AddDomainIntegrator(new VectorDomainLFIntegrator(body));
  extra.Assemble();
  p1.AssembleForce(0.5);
  p1.AddForce(extra);
  ASSERT_TRUE(p1.Solve());

  // Problem 2: the same body force as a load integrator.
  LinearQuasiStaticClampedProblem p2(fes.get(), *rheology, ess_bdr, traction,
                                     pull_marker);
  p2.ExternalLoad().AddDomainIntegrator(new VectorDomainLFIntegrator(body));
  p2.AssembleForce(0.5);
  ASSERT_TRUE(p2.Solve());
  EXPECT_LT(RelMaxDiff(p1.Displacement(), p2.Displacement()), 1e-8);

  // AssembleForce clears the increment: p1 at t = 0.5 without the extra
  // force must differ from the above and equal p2 without it.
  p1.AssembleForce(0.5);
  ASSERT_TRUE(p1.Solve());
  EXPECT_GT(RelMaxDiff(p1.Displacement(), p2.Displacement()), 1e-3);
}

INSTANTIATE_TEST_SUITE_P(ElasticProblem, ElasticProblemTest,
                         testing::Combine(testing::Values(2, 3),
                                          testing::Values(0, 1),
                                          testing::Values(1, 2)));

// The traction problem with a non-natural reference state: both sides use
// the same rheology class (identity mapping on the mapped-mesh side) so
// the integrator classes and quadrature defaults coincide, the mapping is
// the geometric-space interpolant, and the traction on the reference side
// carries the Nanson area factor — then the two discrete problems are the
// same linear system and the solutions agree dof for dof. The mapped
// rotations of the projector are discrete null vectors of the mapped
// stiffness (translations exactly, rotations by the identity).
TEST(TractionProblemMapped, MatchesMappedMeshAndRigidModes) {
  constexpr double kPi = std::numbers::pi;
  for (int dim = 2; dim <= 3; dim++) {
    for (int order = 1; order <= 2; order++) {
      auto mesh = MakeSmallMesh(dim, 0);
      mesh.SetCurvature(order);
      H1_FECollection fec(order, dim);
      FiniteElementSpace fes(&mesh, &fec, dim);

      const double c = 0.05;
      CallableDiffeomorphism xi(
          dim,
          [c, dim](const Vector& x, Vector& y) {
            for (int i = 0; i < dim; i++) {
              y(i) = x(i) + c * std::sin(kPi * x((i + 1) % dim));
            }
          },
          [c, dim](const Vector& x, DenseMatrix& F) {
            F = 0.0;
            for (int i = 0; i < dim; i++) {
              F(i, i) = 1.0;
              F(i, (i + 1) % dim) = c * kPi * std::cos(kPi * x((i + 1) % dim));
            }
          });
      auto xi_h = Interpolate(xi, mesh);
      auto mapped = MappedMesh(mesh, xi);
      FiniteElementSpace fes_mapped(&mapped, &fec, dim);
      CallableDiffeomorphism id(
          dim, [](const Vector& x, Vector& y) { y = x; },
          [dim](const Vector&, DenseMatrix& F) {
            F = 0.0;
            for (int i = 0; i < dim; i++) {
              F(i, i) = 1.0;
            }
          });

      ConstantCoefficient lam(kLambda), mu_c(kMu);
      IsotropicElasticTensorCoefficient C(dim, lam, mu_c);
      DenseMatrix zero_mat(dim);
      zero_mat = 0.0;
      MatrixConstantCoefficient S0(zero_mat);
      // The rheology takes the *referential* description: the relabelled
      // tensor (C is constant, so the composition C o xi is C itself).
      RelabelledElasticTensorCoefficient C_rel(dim, C, xi_h);
      ReferentialElasticRheology rheo_ref(dim, C_rel, S0, xi_h);
      ReferentialElasticRheology rheo_map(dim, C, S0, id);

      Vector tv(dim);
      tv = 0.3;
      tv(0) = 1.0;
      VectorConstantCoefficient t_phys(tv);
      NansonAreaCoefficient area(xi_h);
      ScalarVectorProductCoefficient t_ref(area, t_phys);

      Array<int> all_bdr(mesh.bdr_attributes.Max());
      all_bdr = 1;

      LinearQuasiStaticTractionProblem p_ref(&fes, rheo_ref, t_ref, all_bdr);
      LinearQuasiStaticTractionProblem p_map(&fes_mapped, rheo_map, t_phys,
                                             all_bdr);
      p_ref.AssembleForce(0.0);
      ASSERT_TRUE(p_ref.Solve());
      p_map.AssembleForce(0.0);
      ASSERT_TRUE(p_map.Solve());

      Vector d(p_ref.Displacement());
      d -= p_map.Displacement();
      EXPECT_LT(d.Normlinf() / p_map.Displacement().Normlinf(), 1e-8)
          << "dim " << dim << " order " << order;

      for (auto r : p_ref.RigidPairResiduals()) {
        EXPECT_LT(r, 1e-10) << "dim " << dim << " order " << order;
      }
    }
  }
}

}  // namespace
