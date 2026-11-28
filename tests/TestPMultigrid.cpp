/*
  Serial tests for PMultigridPreconditioner (p_multigrid.hpp) and for the
  problem classes with it (SetPreconditionerPMultigrid).

  The orders of the levels. The preconditioner by itself, on the
  elasticity matrix of a clamped body and of a free one with a mass
  term, on triangles and tetrahedra at orders 3 to 5: the cycle is
  symmetric and positive, it leaves the essential dofs alone, the coarse
  matrices are the Galerkin products, and CG with it reaches the
  solution CG reaches with a Gauss-Seidel sweep, in fewer iterations.
  The clamped and the traction problems with the switch on: the
  displacement of the default preconditioner.
*/

#include <gtest/gtest.h>

#include <cmath>
#include <tuple>
#include <vector>

#include "AdGIA.hpp"
#include "mfem.hpp"

namespace {

using namespace mfem;
using namespace AdGIA;

Mesh MakeBody(int dim) {
  return dim == 2 ? Mesh::MakeCartesian2D(4, 4, Element::TRIANGLE)
                  : Mesh::MakeCartesian3D(2, 2, 2, Element::TETRAHEDRON);
}

// A traction on the faces x = 0 and x = 1 with no net force or moment.
void Pull(const Vector& x, Vector& t) {
  t = 0.0;
  if (std::abs(x[0] - 0.5) > 0.49) {
    t[0] = 2.0 * x[0] - 1.0;
  }
}

double L2Norm(const GridFunction& u) {
  auto zero = Vector(u.FESpace()->GetVDim());
  zero = 0.0;
  auto z = VectorConstantCoefficient(zero);
  return u.ComputeL2Error(z);
}

// CG on A x = b with a preconditioner; the number of iterations, or -1.
int Solve(const Operator& A, Solver& prec, const Vector& b, Vector& x) {
  auto cg = CGSolver();
  cg.SetOperator(A);
  cg.SetPreconditioner(prec);
  cg.SetRelTol(1e-10);
  cg.SetMaxIter(2000);
  cg.SetPrintLevel(-1);
  x = 0.0;
  cg.Mult(b, x);
  return cg.GetConverged() ? cg.GetNumIterations() : -1;
}

TEST(PMultigridOrders, HalveDownToTheCoarseOrder) {
  EXPECT_EQ(PMultigridOrders(3, 2), (std::vector<int>{2, 3}));
  EXPECT_EQ(PMultigridOrders(4, 2), (std::vector<int>{2, 4}));
  EXPECT_EQ(PMultigridOrders(5, 2), (std::vector<int>{2, 3, 5}));
  EXPECT_EQ(PMultigridOrders(6, 2), (std::vector<int>{2, 3, 6}));
  EXPECT_EQ(PMultigridOrders(8, 2), (std::vector<int>{2, 4, 8}));
  EXPECT_EQ(PMultigridOrders(3, 1), (std::vector<int>{1, 2, 3}));
  EXPECT_EQ(PMultigridOrders(2, 2), (std::vector<int>{2}));
  EXPECT_EQ(PMultigridOrders(1, 2), (std::vector<int>{1}));
}

// (dimension, order, clamped)
class PMultigridTest
    : public testing::TestWithParam<std::tuple<int, int, bool>> {};

TEST_P(PMultigridTest, IsASymmetricPreconditionerOfTheMatrix) {
  const auto [dim, order, clamped] = GetParam();
  auto mesh = MakeBody(dim);
  auto fec = H1_FECollection(order, dim);
  auto fes = FiniteElementSpace(&mesh, &fec, dim);

  auto lambda = ConstantCoefficient(1.5), mu = ConstantCoefficient(1.0);
  auto shift = ConstantCoefficient(1e-2);
  auto a = BilinearForm(&fes);
  a.AddDomainIntegrator(new ElasticityIntegrator(lambda, mu));
  if (!clamped) {
    a.AddDomainIntegrator(new VectorMassIntegrator(shift));
  }
  a.Assemble();
  auto ess_bdr = Array<int>(mesh.bdr_attributes.Max());
  ess_bdr = 0;
  if (clamped) {
    ess_bdr[0] = 1;
  }
  auto ess = Array<int>();
  fes.GetEssentialTrueDofs(ess_bdr, ess);
  auto A = SparseMatrix();
  a.FormSystemMatrix(ess, A);

  auto pmg = PMultigridPreconditioner(fes, A, ess, {});
  EXPECT_EQ(pmg.Orders(), PMultigridOrders(order, 2));

  // The coarsest matrix is that of the form on the space of order 2,
  // which the Galerkin products of nested spaces reproduce.
  {
    auto fec2 = H1_FECollection(2, dim);
    auto fes2 = FiniteElementSpace(&mesh, &fec2, dim);
    auto a2 = BilinearForm(&fes2);
    a2.AddDomainIntegrator(new ElasticityIntegrator(lambda, mu));
    if (!clamped) {
      a2.AddDomainIntegrator(new VectorMassIntegrator(shift));
    }
    a2.Assemble();
    auto ess2 = Array<int>();
    fes2.GetEssentialTrueDofs(ess_bdr, ess2);
    auto A2 = SparseMatrix();
    a2.FormSystemMatrix(ess2, A2);
    const auto& coarse = pmg.LevelMatrix(0);
    ASSERT_EQ(coarse.Height(), A2.Height());
    auto x = Vector(A2.Height()), y = Vector(A2.Height());
    auto y2 = Vector(A2.Height());
    x.Randomize(3);
    x.SetSubVector(ess2, 0.0);  // off the essential dofs (their diagonal
                                // is the level's own)
    coarse.Mult(x, y);
    A2.Mult(x, y2);
    y.SetSubVector(ess2, 0.0);
    y2.SetSubVector(ess2, 0.0);
    y -= y2;
    EXPECT_LT(y.Norml2(), 1e-10 * y2.Norml2());
  }

  // Symmetric and positive; nothing at the essential dofs from a
  // right-hand side that has nothing there.
  auto x = Vector(A.Height()), y = Vector(A.Height());
  auto Mx = Vector(A.Height()), My = Vector(A.Height());
  x.Randomize(1);
  y.Randomize(2);
  x.SetSubVector(ess, 0.0);
  y.SetSubVector(ess, 0.0);
  pmg.Mult(x, Mx);
  pmg.Mult(y, My);
  EXPECT_NEAR(x * My, y * Mx, 1e-10 * std::abs(x * Mx));
  EXPECT_GT(x * Mx, 0.0);
  for (const auto i : ess) {
    EXPECT_EQ(Mx[i], 0.0);
  }

  // CG against a Gauss-Seidel sweep.
  auto b = Vector(A.Height());
  b.Randomize(7);
  b.SetSubVector(ess, 0.0);
  auto gs = GSSmoother(A);
  auto x_gs = Vector(A.Height()), x_pmg = Vector(A.Height());
  const auto its_gs = Solve(A, gs, b, x_gs);
  const auto its_pmg = Solve(A, pmg, b, x_pmg);
  ASSERT_GT(its_gs, 0);
  ASSERT_GT(its_pmg, 0);
  x_pmg -= x_gs;
  EXPECT_LT(x_pmg.Norml2(), 1e-7 * x_gs.Norml2());
  EXPECT_LT(its_pmg, its_gs);
  std::cout << "dim " << dim << ", order " << order
            << (clamped ? ", clamped: " : ", free: ") << its_pmg
            << " iterations, " << its_gs << " with a Gauss-Seidel sweep\n";
}

INSTANTIATE_TEST_SUITE_P(
    Bodies, PMultigridTest,
    testing::Values(std::tuple{2, 3, true}, std::tuple{2, 3, false},
                    std::tuple{2, 4, true}, std::tuple{2, 4, false},
                    std::tuple{2, 5, true}, std::tuple{2, 5, false},
                    std::tuple{3, 3, true}, std::tuple{3, 3, false},
                    std::tuple{3, 4, true}, std::tuple{3, 4, false}));

class PMultigridProblemTest : public testing::TestWithParam<int> {};

TEST_P(PMultigridProblemTest, GivesTheDisplacementOfTheDefault) {
  const auto dim = GetParam();
  const auto order = 3;
  auto mesh = MakeBody(dim);
  auto fec = H1_FECollection(order, dim);
  auto fes = FiniteElementSpace(&mesh, &fec, dim);
  auto kappa = ConstantCoefficient(2.0), mu = ConstantCoefficient(1.0);
  auto rheology = IsotropicElasticRheology(dim, kappa, mu);
  auto pull = VectorFunctionCoefficient(dim, Pull);
  auto all = Array<int>(mesh.bdr_attributes.Max());
  all = 1;
  auto ess_bdr = Array<int>(mesh.bdr_attributes.Max());
  ess_bdr = 0;
  ess_bdr[0] = 1;
  auto loaded = Array<int>(all);
  loaded[0] = 0;

  const auto run = [&](LinearQuasiStaticProblemBase& problem, bool pmg,
                       double& norm, long& its) {
    problem.SetPreconditionerPMultigrid(pmg);
    EXPECT_EQ(problem.PreconditionerPMultigrid(), pmg);
    problem.SetRelTol(1e-11);
    problem.AssembleForce(0.0);
    const auto before = problem.TotalIterations();
    EXPECT_TRUE(problem.Solve());
    its = problem.TotalIterations() - before;
    norm = L2Norm(problem.Displacement());
  };

  {
    auto with_gs =
        LinearQuasiStaticClampedProblem(&fes, rheology, ess_bdr, pull, loaded);
    auto with_pmg =
        LinearQuasiStaticClampedProblem(&fes, rheology, ess_bdr, pull, loaded);
    double n_gs, n_pmg;
    long i_gs, i_pmg;
    run(with_gs, false, n_gs, i_gs);
    run(with_pmg, true, n_pmg, i_pmg);
    EXPECT_NEAR(n_pmg, n_gs, 1e-8 * n_gs);
    EXPECT_GT(i_pmg, 0);
    EXPECT_LT(i_pmg, i_gs);
  }
  {
    auto with_gs = LinearQuasiStaticTractionProblem(&fes, rheology, pull, all);
    auto with_pmg = LinearQuasiStaticTractionProblem(&fes, rheology, pull, all);
    double n_gs, n_pmg;
    long i_gs, i_pmg;
    run(with_gs, false, n_gs, i_gs);
    run(with_pmg, true, n_pmg, i_pmg);
    EXPECT_NEAR(n_pmg, n_gs, 1e-8 * n_gs);
    EXPECT_GT(i_pmg, 0);
    EXPECT_LT(i_pmg, i_gs);
  }
}

INSTANTIATE_TEST_SUITE_P(Dimensions, PMultigridProblemTest,
                         testing::Values(2, 3));

}  // namespace
