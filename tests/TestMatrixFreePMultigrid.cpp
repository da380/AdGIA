#include <cstdlib>
#include <memory>
#include <numbers>

#include "TestCommon.hpp"

/*
  MatrixFreePMultigrid (p_multigrid.hpp), serial: the elastic tensor form
  of a transversely isotropic body on a curved mesh of hexahedra, clamped
  on a face or free with a mass term, at orders 3 and 4.

  - The finest level's operator is the assembled matrix of the form, with
    the essential rows and columns eliminated.
  - The cycle is symmetric and positive, and CG with it on the finest
    level's operator reaches the solution of the assembled system in a
    bounded number of iterations.

  - A problem class with SetMatrixFree gives the displacement it gives by
    default.

  Set ADGIA_TEST_DEVICE (e.g. to cuda) to run on a device.
*/

namespace {

constexpr double kPi = std::numbers::pi;

class DeviceEnvironment : public testing::Environment {
 public:
  void SetUp() override {
    const char* name = std::getenv("ADGIA_TEST_DEVICE");
    device_ = std::make_unique<Device>(name ? name : "cpu");
  }
  void TearDown() override { device_.reset(); }

 private:
  std::unique_ptr<Device> device_;
};
const auto* const device_environment =
    testing::AddGlobalTestEnvironment(new DeviceEnvironment);

void Jiggle(const Vector& x, Vector& y) {
  y = x;
  const double s = 0.06;
  y[0] += s * std::sin(kPi * x[0]) * std::sin(kPi * x[1]);
  y[1] += s * std::cos(0.5 * kPi * x[0]) * std::sin(kPi * x[1]);
  y[2] += s * std::sin(kPi * x[2]) * std::cos(kPi * x[0]);
}

using Param = std::tuple<int, bool>;  // (order, clamped)

class MatrixFreePMultigridTest : public testing::TestWithParam<Param> {};

TEST_P(MatrixFreePMultigridTest, PreconditionsTheForm) {
  const auto [order, clamped] = GetParam();
  auto mesh = Mesh::MakeCartesian3D(3, 3, 3, Element::HEXAHEDRON);
  mesh.SetCurvature(2);
  mesh.Transform(Jiggle);
  H1_FECollection fec(order, 3);
  FiniteElementSpace fes(&mesh, &fec, 3);

  ConstantCoefficient A(3.1), C(2.7), F(1.1), L(0.9), N(1.2), shift(0.5);
  Vector x0(3);
  x0 = -0.3;
  RadialUnitVectorCoefficient axis(3, x0);
  TransverselyIsotropicElasticTensorCoefficient ti(3, A, C, F, L, N, axis);
  auto integrators = [&](BilinearForm& form) {
    form.AddDomainIntegrator(new ElasticTensorIntegrator(ti));
    if (!clamped) {
      form.AddDomainIntegrator(new VectorMassIntegrator(shift));
    }
  };
  Array<int> ess_bdr(mesh.bdr_attributes.Max());
  ess_bdr = 0;
  if (clamped) {
    ess_bdr[0] = 1;
  }

  MatrixFreePMultigrid mg(fes, ess_bdr, integrators, {});
  EXPECT_EQ(mg.Orders().back(), order);
  EXPECT_EQ(mg.Orders().front(), 2);
  const Operator& op = mg.FineOperator();
  ASSERT_EQ(op.Height(), fes.GetTrueVSize());

  // The assembled system.
  BilinearForm a(&fes);
  integrators(a);
  a.Assemble();
  SparseMatrix Amat;
  a.FormSystemMatrix(mg.FineEssentialTrueDofs(), Amat);
  const double scale = Amat.MaxNorm();

  const int n = fes.GetTrueVSize();
  // Off the essential dofs, where the two keep different diagonals.
  auto x = RandomVector(n), y = RandomVector(n);
  for (const int i : mg.FineEssentialTrueDofs()) {
    x[i] = 0.0;
    y[i] = 0.0;
  }
  Vector t1(n), t2(n);
  op.Mult(x, t1);
  Amat.Mult(x, t2);
  t1 -= t2;
  EXPECT_LT(t1.Normlinf(), 1e-10 * scale * x.Normlinf());

  // Symmetric and positive.
  mg.Mult(x, t1);
  mg.Mult(y, t2);
  const double xMy = x * t2, yMx = y * t1, xMx = x * t1;
  EXPECT_LT(std::abs(xMy - yMx), 1e-9 * (std::abs(xMy) + std::abs(yMx)));
  EXPECT_GT(xMx, 0.0);

  // CG with the cycle, against the assembled system solved tightly.
  auto b = RandomVector(n);
  for (const int i : mg.FineEssentialTrueDofs()) {
    b[i] = 0.0;
  }
  Vector u(n), u_ref(n);
  u = 0.0;
  u_ref = 0.0;
  CGSolver cg;
  cg.SetOperator(op);
  cg.SetPreconditioner(mg);
  cg.SetRelTol(1e-10);
  cg.SetAbsTol(0.0);
  cg.SetMaxIter(200);
  cg.SetPrintLevel(-1);
  cg.Mult(b, u);
  EXPECT_TRUE(cg.GetConverged());
  EXPECT_LE(cg.GetNumIterations(), 60) << "order " << order;

  GSSmoother gs(Amat);
  CGSolver ref;
  ref.SetOperator(Amat);
  ref.SetPreconditioner(gs);
  ref.SetRelTol(1e-13);
  ref.SetAbsTol(0.0);
  ref.SetMaxIter(20000);
  ref.SetPrintLevel(-1);
  ref.Mult(b, u_ref);
  EXPECT_TRUE(ref.GetConverged());
  u -= u_ref;
  EXPECT_LT(u.Norml2(), 1e-7 * u_ref.Norml2());
  std::cout << "  order " << order << (clamped ? ", clamped: " : ", free: ")
            << cg.GetNumIterations() << " iterations\n";
}

// A problem class with its stiffness held with no matrix (SetMatrixFree):
// a free elastic body under traction, the displacement of the default.
TEST(MatrixFreeProblem, GivesTheDisplacementOfTheDefault) {
  auto mesh = Mesh::MakeCartesian3D(3, 3, 3, Element::HEXAHEDRON);
  H1_FECollection fec(3, 3);
  FiniteElementSpace fes(&mesh, &fec, 3);
  ConstantCoefficient kappa(2.0), mu(1.0);
  IsotropicElasticRheology rheology(3, kappa, mu);
  VectorFunctionCoefficient pull(3, [](const Vector& x, Vector& t) {
    t = 0.0;
    if (std::abs(x[0] - 0.5) > 0.49) {
      t[0] = 2.0 * x[0] - 1.0;
    }
  });
  Array<int> all(mesh.bdr_attributes.Max());
  all = 1;
  Vector zero(3);
  zero = 0.0;
  VectorConstantCoefficient z(zero);
  double norm[2];
  for (const int mf : {0, 1}) {
    LinearQuasiStaticTractionProblem problem(&fes, rheology, pull, all);
    problem.SetMatrixFree(mf == 1);
    problem.SetRelTol(1e-11);
    problem.AssembleForce(0.0);
    EXPECT_TRUE(problem.Solve());
    norm[mf] = problem.Displacement().ComputeL2Error(z);
  }
  EXPECT_GT(norm[0], 0.0);
  EXPECT_LT(std::abs(norm[1] - norm[0]), 1e-8 * norm[0]);
}

INSTANTIATE_TEST_SUITE_P(Cases, MatrixFreePMultigridTest,
                         testing::Combine(testing::Values(3, 4),
                                          testing::Bool()));

}  // namespace
