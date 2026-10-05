#include "MixedProblemTestCommon.hpp"
#include "TestCommon.hpp"

/*
  Tests for LinearQuasiStaticMixedSelfGravitatingProblem on the canned two-layer
  meshes (2-D disc in a disc, 3-D ball in a ball), see
  MixedProblemTestCommon.hpp.

  - The Schur-complement CG and the block MINRES solvers give the same
    displacement and potential to solver tolerance (both solve the system
    restricted to the complement of the rigid modes).
  - With the mass-weighted rigid gauge the two solvers return the same
    representative, which carries no linear or angular momentum and
    differs from the Euclidean-gauge solution by a rigid motion only.
  - The rigid-mode residuals are small and decrease with the order.
  - With zero density the problem reduces to pure elasticity: the solution
    equals a direct projected CG solve of the traction problem's stiffness
    (the coupling, the gravity terms and the background potential all
    vanish). On these curved meshes the rigid rotations are exact null
    vectors only for order >= 2, so the result depends on the rigid-mode
    regularisation: the reference solves P A P x = P b with the class's
    own projector P.
  - The response is linear in the loads: time scaling of the surface load,
    and superposition of a surface load and an AddForce() increment.
  - Repeated and out-of-order solves (warm starts) reproduce the cold
    solutions to solver tolerance.
  - On a Maxwell rheology, a unit relaxation weight leaves the solution
    unchanged, a smaller one changes it and ClearRelaxationWeights()
    restores it.
  - A supplied background potential equal to the solved one gives the same
    solution.
  - The viscoelastic operator runs on top of the problem (creep under a
    sustained surface load increases the displacement).
*/

namespace {

using namespace self_grav_test;

// (dim, order)
using Param = std::tuple<int, int>;

struct Case {
  std::unique_ptr<Mesh> parent;
  std::unique_ptr<SubMesh> body;
  std::unique_ptr<H1_FECollection> fec_u, fec_phi;
  std::unique_ptr<FiniteElementSpace> fes_u, fes_phi;
  std::unique_ptr<ConstantCoefficient> kappa, mu, rho;
  std::unique_ptr<IsotropicElasticRheology> rheology;
  std::unique_ptr<FunctionCoefficient> sigma;
  Array<int> surface;

  Case(int dim, int order, double density = kRho) {
    parent = std::make_unique<Mesh>(MeshFile(dim).c_str(), 1, 1);
    EXPECT_EQ(parent->Dimension(), dim);
    body = std::make_unique<SubMesh>(
        SubMesh::CreateFromDomain(*parent, BodyMarker(*parent)));
    fec_u = std::make_unique<H1_FECollection>(order, dim);
    fec_phi = std::make_unique<H1_FECollection>(order, dim);
    fes_u = std::make_unique<FiniteElementSpace>(body.get(), fec_u.get(), dim);
    fes_phi = std::make_unique<FiniteElementSpace>(parent.get(), fec_phi.get());
    kappa = std::make_unique<ConstantCoefficient>(kKappa);
    mu = std::make_unique<ConstantCoefficient>(kMu);
    rho = std::make_unique<ConstantCoefficient>(density);
    rheology = std::make_unique<IsotropicElasticRheology>(dim, *kappa, *mu);
    sigma = std::make_unique<FunctionCoefficient>(SurfaceLoad);
    surface = SurfaceMarker(*body);
  }

  std::unique_ptr<LinearQuasiStaticMixedSelfGravitatingProblem> Problem(
      bool with_load = true) {
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        fes_u.get(), fes_phi.get(), *rheology, *rho, kG, kDtNDegree);
    if (with_load) {
      p->SetSurfaceLoad(*sigma, surface);
    }
    p->SetRelTol(1e-11);
    return p;
  }
};

double RelDiff(const GridFunction& a, const GridFunction& b) {
  GridFunction d(a);
  d -= b;
  return L2Norm(d) / L2Norm(b);
}

class MixedProblemTest : public testing::TestWithParam<Param> {};

TEST_P(MixedProblemTest, SchurAndMinresAgree) {
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

  const double tol = 1e-7;
  EXPECT_GT(L2Norm(schur->Displacement()), 0.0);
  EXPECT_GT(L2Norm(schur->Potential()), 0.0);
  EXPECT_LT(RelDiff(schur->Displacement(), minres->Displacement()), tol);
  EXPECT_LT(RelDiff(schur->Potential(), minres->Potential()), tol);
  EXPECT_GT(schur->LastInnerIterations(), minres->LastInnerIterations());
}

TEST_P(MixedProblemTest, MassWeightedGaugeAgrees) {
  const auto [dim, order] = GetParam();
  Case s(dim, order);

  // With the same physical gauge the two solvers return the same
  // representative directly, and it carries no momentum.
  auto schur = s.Problem();
  schur->SetSolverType(
      LinearQuasiStaticMixedSelfGravitatingProblem::SolverType::SchurCG);
  schur->SetMassWeightedGauge();
  schur->AssembleForce(0.0);
  ASSERT_TRUE(schur->Solve());

  auto minres = s.Problem();
  minres->SetMassWeightedGauge();
  minres->AssembleForce(0.0);
  ASSERT_TRUE(minres->Solve());
  EXPECT_LT(RelDiff(schur->Displacement(), minres->Displacement()), 1e-7);
  EXPECT_LT(RelDiff(schur->Potential(), minres->Potential()), 1e-7);

  BilinearForm mass(s.fes_u.get());
  mass.AddDomainIntegrator(new VectorMassIntegrator(*s.rho));
  mass.Assemble();
  mass.Finalize();
  const GridFunction& u = schur->Displacement();
  Vector Mu(u.Size());
  mass.SpMat().Mult(u, Mu);
  const auto& P = schur->RigidModes();
  double moment = 0.0;
  for (int i = 0; i < P.Size(); i++) {
    moment = std::max(moment, std::abs(P.Dot(Mu, P.Basis(i))));
  }
  EXPECT_LT(moment, 1e-10 * Mu.Norml2());

  // Differs from the Euclidean gauge by a rigid motion only.
  auto plain = s.Problem();
  plain->AssembleForce(0.0);
  ASSERT_TRUE(plain->Solve());
  GridFunction d(u);
  d -= plain->Displacement();
  const double dnorm = L2Norm(d);
  Vector dt;
  d.GetTrueDofs(dt);
  P.Project(dt);
  d.SetFromTrueDofs(dt);
  EXPECT_LT(L2Norm(d), 1e-6 * dnorm);
}

TEST(MixedProblemRigidModes, ResidualsDecreaseWithOrder) {
  for (int dim : {2, 3}) {
    std::vector<double> worst;
    for (int order : {1, 2}) {
      Case s(dim, order);
      auto p = s.Problem();
      const auto res = p->RigidModeResiduals();
      ASSERT_EQ(static_cast<int>(res.size()), dim * (dim + 1) / 2);
      double w = 0.0;
      for (auto r : res) {
        EXPECT_LT(r, 1e-2);
        w = std::max(w, r);
      }
      worst.push_back(w);
    }
    EXPECT_LT(worst[1], 0.2 * worst[0]);
  }
}

TEST_P(MixedProblemTest, ZeroDensityIsTractionProblem) {
  const auto [dim, order] = GetParam();
  Case s(dim, order, 0.0);

  // A self-equilibrated traction on the surface (in the sense that the
  // rigid-body projection removes any net force/torque either way).
  VectorFunctionCoefficient traction(dim, [](const Vector& x, Vector& f) {
    const double r = x.Norml2();
    const double c = (x.Size() == 2 ? x[1] : x[2]) / r;
    f = x;
    f *= -0.1 * (1.0 + 3.0 * c * c) / r;
  });

  auto p = s.Problem(false);
  p->ExternalLoad().AddBoundaryIntegrator(
      new VectorBoundaryLFIntegrator(traction), s.surface);
  p->AssembleForce(0.0);
  ASSERT_TRUE(p->Solve());
  EXPECT_LT(L2Norm(p->Potential()), 1e-12);
  EXPECT_LT(L2Norm(p->BackgroundPotential()), 1e-12);

  // Reference: P A P x = P b with the same rigid modes, A the elastic
  // stiffness of a LinearQuasiStaticTractionProblem on the same space.
  LinearQuasiStaticTractionProblem ref(s.fes_u.get(), *s.rheology, traction,
                                       s.surface);
  ref.AssembleForce(0.0);
  const SparseMatrix& A = *ref.SystemMatrix().As<SparseMatrix>();
  LinearForm b(s.fes_u.get());
  b.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(traction), s.surface);
  b.Assemble();
  const NullSpaceProjector& P = p->RigidModes();
  ProjectedOperator PAP(A, P);
  GSSmoother prec(A);
  CGSolver cg;
  cg.SetOperator(PAP);
  cg.SetPreconditioner(prec);
  cg.SetRelTol(1e-13);
  cg.SetAbsTol(0.0);
  cg.SetMaxIter(20000);
  cg.SetPrintLevel(0);
  Vector B(b), X(b.Size());
  P.Project(B);
  X = 0.0;
  cg.Mult(B, X);
  ASSERT_TRUE(cg.GetConverged());
  P.Project(X);
  GridFunction u_ref(s.fes_u.get());
  u_ref.SetFromTrueDofs(X);

  EXPECT_GT(L2Norm(u_ref), 0.0);
  EXPECT_LT(RelDiff(p->Displacement(), u_ref), 1e-7);
}

TEST_P(MixedProblemTest, LinearInTheLoads) {
  const auto [dim, order] = GetParam();
  Case s(dim, order);

  // Time scaling: sigma(t) = (1 + t) sigma(0).
  auto p = s.Problem();
  p->AssembleForce(0.0);
  ASSERT_TRUE(p->Solve());
  GridFunction u0(p->Displacement()), phi0(p->Potential());
  p->AssembleForce(1.0);
  ASSERT_TRUE(p->Solve());
  GridFunction u1(p->Displacement()), phi1(p->Potential());
  u0 *= 2.0;
  phi0 *= 2.0;
  EXPECT_LT(RelDiff(u1, u0), 1e-7);
  EXPECT_LT(RelDiff(phi1, phi0), 1e-7);

  // Superposition: surface load plus an increment f on the displacement.
  VectorFunctionCoefficient body_force(dim, [](const Vector& x, Vector& f) {
    f = 0.0;
    f[0] = 0.05 * x[0];
    f[1] = -0.03 * x[0] * x[0];
  });
  LinearForm f(s.fes_u.get());
  f.AddDomainIntegrator(new VectorDomainLFIntegrator(body_force));
  f.Assemble();

  auto q = s.Problem(false);
  q->AssembleForce(0.0);
  q->AddForce(f);
  ASSERT_TRUE(q->Solve());
  GridFunction uf(q->Displacement()), phif(q->Potential());

  p->AssembleForce(0.0);
  p->AddForce(f);
  ASSERT_TRUE(p->Solve());
  u0 *= 0.5;
  phi0 *= 0.5;
  u0 += uf;
  phi0 += phif;
  EXPECT_LT(RelDiff(p->Displacement(), u0), 1e-7);
  EXPECT_LT(RelDiff(p->Potential(), phi0), 1e-7);
}

TEST_P(MixedProblemTest, WarmStartsReproduceColdSolves) {
  const auto [dim, order] = GetParam();
  Case s(dim, order);

  for (auto type :
       {LinearQuasiStaticMixedSelfGravitatingProblem::SolverType::SchurCG,
        LinearQuasiStaticMixedSelfGravitatingProblem::SolverType::BlockMINRES}) {
    auto p = s.Problem();
    p->SetSolverType(type);
    p->AssembleForce(0.0);
    ASSERT_TRUE(p->Solve());
    GridFunction u0(p->Displacement()), phi0(p->Potential());

    // Same load again: converged already, nothing should move.
    ASSERT_TRUE(p->Solve());
    EXPECT_LT(RelDiff(p->Displacement(), u0), 1e-9);
    EXPECT_LT(RelDiff(p->Potential(), phi0), 1e-9);

    // A different load, then back.
    p->AssembleForce(3.0);
    ASSERT_TRUE(p->Solve());
    p->AssembleForce(0.0);
    ASSERT_TRUE(p->Solve());
    EXPECT_LT(RelDiff(p->Displacement(), u0), 1e-8);
    EXPECT_LT(RelDiff(p->Potential(), phi0), 1e-8);
  }
}

TEST_P(MixedProblemTest, RelaxationWeights) {
  const auto [dim, order] = GetParam();
  Case s(dim, order);

  auto p = s.Problem();
  p->AssembleForce(0.0);
  ASSERT_TRUE(p->Solve());
  GridFunction u0(p->Displacement()), phi0(p->Potential());

  ConstantCoefficient tau(1.0);
  auto maxwell = IsotropicMaxwellRheology::Maxwell(dim, *s.kappa, *s.mu, tau);
  auto q = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
      s.fes_u.get(), s.fes_phi.get(), maxwell, *s.rho, kG, kDtNDegree);
  q->SetSurfaceLoad(*s.sigma, s.surface);
  q->SetRelTol(1e-11);
  ASSERT_TRUE(q->SupportsRelaxationWeights());

  ConstantCoefficient one(1.0), half(0.5);
  q->SetRelaxationWeights({&one});
  q->AssembleForce(0.0);
  ASSERT_TRUE(q->Solve());
  EXPECT_LT(RelDiff(q->Displacement(), u0), 1e-8);
  EXPECT_LT(RelDiff(q->Potential(), phi0), 1e-8);

  q->SetRelaxationWeights({&half});
  ASSERT_TRUE(q->Solve());
  EXPECT_GT(RelDiff(q->Displacement(), u0), 1e-2);

  q->ClearRelaxationWeights();
  ASSERT_TRUE(q->Solve());
  EXPECT_LT(RelDiff(q->Displacement(), u0), 1e-8);
  EXPECT_LT(RelDiff(q->Potential(), phi0), 1e-8);
}

TEST_P(MixedProblemTest, SuppliedBackgroundPotential) {
  const auto [dim, order] = GetParam();
  Case s(dim, order);

  auto p = s.Problem();
  p->AssembleForce(0.0);
  ASSERT_TRUE(p->Solve());

  GridFunctionCoefficient phi0_coeff(&p->BackgroundPotential());
  auto q = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
      s.fes_u.get(), s.fes_phi.get(), *s.rheology, *s.rho, kG, kDtNDegree,
      &phi0_coeff);
  q->SetSurfaceLoad(*s.sigma, s.surface);
  q->SetRelTol(1e-11);
  q->AssembleForce(0.0);
  ASSERT_TRUE(q->Solve());

  EXPECT_LT(RelDiff(q->BackgroundPotential(), p->BackgroundPotential()),
            1e-12);
  EXPECT_LT(RelDiff(q->Displacement(), p->Displacement()), 1e-8);
  EXPECT_LT(RelDiff(q->Potential(), p->Potential()), 1e-8);
}

TEST_P(MixedProblemTest, ViscoelasticCreep) {
  const auto [dim, order] = GetParam();
  Case s(dim, order);

  ConstantCoefficient tau(1.0);
  IsotropicMaxwellRheology maxwell =
      IsotropicMaxwellRheology::Maxwell(dim, *s.kappa, *s.mu, tau);
  LinearQuasiStaticMixedSelfGravitatingProblem p(s.fes_u.get(), s.fes_phi.get(),
                                            maxwell, *s.rho, kG, kDtNDegree);
  p.SetSurfaceLoad(*s.sigma, s.surface);
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
  for (int step = 0; step < 4; step++) {
    stepper.Step(m, t, dt);
    ASSERT_TRUE(op.SolveElastic(m, t));
    const double now = L2Norm(p.Displacement());
    EXPECT_TRUE(std::isfinite(now));
    EXPECT_GT(now, previous);
    previous = now;
  }
}

INSTANTIATE_TEST_SUITE_P(MixedProblem, MixedProblemTest,
                         testing::Values(Param{2, 1}, Param{2, 2},
                                         Param{3, 2}));

}  // namespace
