#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <tuple>

#include "QuasiStaticTestCommon.hpp"

/*
  Tests for the rigid-mode null-space handling in null_space.hpp:

  - MakeRigidModeProjector() holds d(d+1)/2 orthonormal vectors that are
    (on a Cartesian mesh, exactly) in the null space of the free stiffness;
    adding the modes again adds nothing; a dependent vector is dropped.
  - ProjectedSolver + CG on the singular stiffness of a free unit square
    (cube) converges for a load with net force and torque; the solution is
    orthogonal to every basis vector, satisfies the projected equations,
    and equals the solution for the projected load.
  - A warm start carrying a rigid component gives the same solution.
  - ProjectedSolver::SetGauge() with a mass matrix gives zero net momentum
    and angular momentum, and differs from the Euclidean-gauge solution by a
    rigid motion only.
*/

namespace {

using namespace elastic_test;
using namespace AdGIA;

using Param = std::tuple<int, int, int>;  // (dim, elementType, order)

class NullSpaceTest : public testing::TestWithParam<Param> {
 protected:
  void SetUp() override {
    std::tie(dim, elementType, order) = GetParam();
    mesh = std::make_unique<Mesh>(MakeSmallMesh(dim, elementType));
    fec = std::make_unique<H1_FECollection>(order, dim);
    fes = std::make_unique<FiniteElementSpace>(mesh.get(), fec.get(), dim);
    lambda = std::make_unique<ConstantCoefficient>(kLambda);
    mu = std::make_unique<ConstantCoefficient>(kMu);

    // Free stiffness (no essential conditions) and a pull on the x = 1 face:
    // net force and, about the centroid, net torque.
    a = std::make_unique<BilinearForm>(fes.get());
    a->AddDomainIntegrator(new ElasticityIntegrator(*lambda, *mu));
    a->Assemble();
    Array<int> empty;
    a->FormSystemMatrix(empty, A);

    const auto x1_attr = BdrAttributeAt(*mesh, 0, 1.0);
    auto marker = Marker(mesh->bdr_attributes.Max(), {x1_attr});
    pull = std::make_unique<VectorFunctionCoefficient>(dim, PullTraction);
    b = std::make_unique<LinearForm>(fes.get());
    b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(*pull), marker);
    b->Assemble();

    P = MakeRigidModeProjector(*fes);
  }

  // CG on P A P with the Gauss-Seidel preconditioner P M P, wrapped in a
  // ProjectedSolver; returns the wrapper (the pieces live in the fixture).
  ProjectedSolver& MakeSolver() {
    smoother = std::make_unique<GSSmoother>(A);
    projected_prec = std::make_unique<ProjectedSolver>(*P);
    projected_prec->SetSolver(*smoother);
    cg = std::make_unique<CGSolver>();
    cg->SetRelTol(1e-13);
    cg->SetAbsTol(0.0);
    cg->SetMaxIter(10000);
    solver = std::make_unique<ProjectedSolver>(*P);
    solver->SetSolver(*cg);
    // Operator before preconditioner: SetOperator is forwarded to the
    // preconditioner, and the smoother only accepts a SparseMatrix.
    solver->SetOperator(A);
    cg->SetPreconditioner(*projected_prec);
    return *solver;
  }

  double MaxBasisComponent(const Vector& x) const {
    double m = 0.0;
    for (int i = 0; i < P->Size(); i++) {
      m = std::max(m, std::abs(P->Dot(x, P->Basis(i))));
    }
    return m;
  }

  int dim, elementType, order;
  std::unique_ptr<Mesh> mesh;
  std::unique_ptr<H1_FECollection> fec;
  std::unique_ptr<FiniteElementSpace> fes;
  std::unique_ptr<Coefficient> lambda, mu;
  std::unique_ptr<BilinearForm> a;
  SparseMatrix A;
  std::unique_ptr<VectorCoefficient> pull;
  std::unique_ptr<LinearForm> b;
  std::unique_ptr<NullSpaceProjector> P;
  std::unique_ptr<GSSmoother> smoother;
  std::unique_ptr<ProjectedSolver> projected_prec, solver;
  std::unique_ptr<CGSolver> cg;
};

TEST_P(NullSpaceTest, RigidModeProjectorBasis) {
  EXPECT_EQ(P->Size(), dim * (dim + 1) / 2);
  Vector An(A.Height());
  for (int i = 0; i < P->Size(); i++) {
    for (int j = 0; j < P->Size(); j++) {
      const auto g = P->Dot(P->Basis(i), P->Basis(j));
      EXPECT_NEAR(g, i == j ? 1.0 : 0.0, 1e-12);
    }
    // The rigid modes lie in the space exactly on a Cartesian mesh, so they
    // are exact null vectors of the assembled stiffness.
    A.Mult(P->Basis(i), An);
    EXPECT_LE(An.Normlinf(), 1e-12 * A.MaxNorm());
  }
  // Nothing new to add; a combination of basis vectors is dropped.
  EXPECT_EQ(AddRigidModes(*P, *fes), 0);
  EXPECT_EQ(P->Size(), dim * (dim + 1) / 2);
  Vector v(P->Basis(0));
  v.Add(2.0, P->Basis(P->Size() - 1));
  EXPECT_FALSE(P->Add(v));
  EXPECT_EQ(P->Size(), dim * (dim + 1) / 2);
}

TEST_P(NullSpaceTest, ProjectedSolveOfFreeBody) {
  auto& S = MakeSolver();
  Vector x(A.Height());
  x = 0.0;
  S.Mult(*b, x);
  EXPECT_TRUE(cg->GetConverged());
  const auto xnorm = x.Norml2();
  EXPECT_GT(xnorm, 0.0);
  EXPECT_LE(MaxBasisComponent(x), 1e-12 * xnorm);

  // The projected equations hold: P (A x - b) = 0.
  Vector r(A.Height()), Pb(A.Height());
  A.Mult(x, r);
  r -= *b;
  P->Project(r);
  P->Project(*b, Pb);
  EXPECT_LE(r.Norml2(), 1e-9 * Pb.Norml2());

  // The projected load gives the same displacement.
  Vector y(A.Height());
  y = 0.0;
  S.Mult(Pb, y);
  y -= x;
  EXPECT_LE(y.Norml2(), 1e-9 * xnorm);
}

TEST_P(NullSpaceTest, WarmStartWithRigidComponent) {
  auto& S = MakeSolver();
  Vector x(A.Height());
  x = 0.0;
  S.Mult(*b, x);
  ASSERT_TRUE(cg->GetConverged());

  // Restart from the solution plus a rigid motion: the wrapper projects the
  // initial guess, CG sees a converged start and the result is unchanged.
  // As in the problem layer, a warm start needs an absolute tolerance (the
  // relative one is measured against the restart residual).
  Vector z(A.Height());
  projected_prec->Mult(*b, z);
  cg->SetAbsTol(1e-13 * std::sqrt(P->Dot(*b, z)));
  S.iterative_mode = true;
  Vector y(x);
  y.Add(5.0, P->Basis(0));
  y.Add(-3.0, P->Basis(P->Size() - 1));
  S.Mult(*b, y);
  EXPECT_TRUE(cg->GetConverged());
  EXPECT_LE(cg->GetNumIterations(), 1);
  y -= x;
  EXPECT_LE(y.Norml2(), 1e-9 * x.Norml2());
}

TEST_P(NullSpaceTest, MassWeightedGauge) {
  auto& S = MakeSolver();
  Vector x(A.Height());
  x = 0.0;
  S.Mult(*b, x);
  ASSERT_TRUE(cg->GetConverged());

  // Unit-density vector mass matrix.
  BilinearForm mass(fes.get());
  mass.AddDomainIntegrator(new VectorMassIntegrator());
  mass.Assemble();
  mass.Finalize();
  const SparseMatrix& M = mass.SpMat();
  S.SetGauge(&M);
  Vector y(A.Height());
  y = 0.0;
  S.Mult(*b, y);
  ASSERT_TRUE(cg->GetConverged());

  // Zero momentum and angular momentum: n_i . M y = 0; the difference from
  // the Euclidean representative is a rigid motion.
  Vector My(A.Height());
  M.Mult(y, My);
  EXPECT_LE(MaxBasisComponent(My), 1e-12 * My.Norml2());
  EXPECT_GT(MaxBasisComponent(y), 1e-6 * y.Norml2());  // not Euclidean
  Vector d(y);
  d -= x;
  const auto dnorm = d.Norml2();
  P->Project(d);
  EXPECT_LE(d.Norml2(), 1e-12 * dnorm);
  // Same strains: the residual is unchanged.
  Vector r(A.Height()), Pb(A.Height());
  A.Mult(y, r);
  r -= *b;
  P->Project(r);
  P->Project(*b, Pb);
  EXPECT_LE(r.Norml2(), 1e-9 * Pb.Norml2());

  // Back to the Euclidean gauge.
  S.SetGauge(nullptr);
  y = 0.0;
  S.Mult(*b, y);
  y -= x;
  EXPECT_LE(y.Norml2(), 1e-12 * x.Norml2());
}

INSTANTIATE_TEST_SUITE_P(NullSpace, NullSpaceTest,
                         testing::Combine(testing::Values(2, 3),
                                          testing::Values(0, 1),
                                          testing::Values(1, 2)));

}  // namespace
