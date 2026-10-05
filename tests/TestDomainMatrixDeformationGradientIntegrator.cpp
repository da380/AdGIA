#include "TestCommon.hpp"

class BilinearFormIntegratorTests
    : public ::testing::TestWithParam<DimOrderTypeTuple> {};

// int q v : grad u for a linear field u = B x and a constant matrix field
// v = A (random, non-symmetric; components in column-major order), against
// DomainLFDeformationGradientIntegrator with the coefficient q A: checks
// the component ordering of the matrix space and the index convention.
TEST_P(BilinearFormIntegratorTests, DomainMatrixDeformationGradientIntegrator) {
  const auto& current_tuple = GetParam();

  auto dim = std::get<0>(current_tuple);
  auto order = std::get<1>(current_tuple);
  auto elementType = std::get<2>(current_tuple);

  auto mesh = MakeMesh(dim, elementType);

  auto L2 = L2_FECollection(order, dim);
  auto H1 = H1_FECollection(order, dim);

  auto vector_fes = FiniteElementSpace(&mesh, &H1, dim);
  auto matrix_fes = FiniteElementSpace(&mesh, &L2, dim * dim);

  auto q = FunctionCoefficient([](const Vector& x) { return x.Norml2(); });

  auto b = MixedBilinearForm(&vector_fes, &matrix_fes);
  b.AddDomainIntegrator(new DomainMatrixDeformationGradientIntegrator(q));
  b.Assemble();

  auto B = RandomMatrix(dim);
  auto uF = VectorFunctionCoefficient(
      dim, [B](const Vector& x, Vector& y) { B.Mult(x, y); });
  auto u = GridFunction(&vector_fes);
  u.ProjectCoefficient(uF);

  auto A = RandomMatrix(dim);
  auto a = Vector(A.GetData(), dim * dim);

  auto vF = VectorConstantCoefficient(a);
  auto v = GridFunction(&matrix_fes);
  v.ProjectCoefficient(vF);

  auto w = GridFunction(&matrix_fes);
  b.Mult(u, w);

  auto value1 = v * w;

  auto AF = MatrixConstantCoefficient(A);
  auto MF = ScalarMatrixProductCoefficient(q, AF);
  auto l = LinearForm(&vector_fes);
  l.AddDomainIntegrator(new DomainLFDeformationGradientIntegrator(MF));
  l.Assemble();

  auto value2 = l * u;

  EXPECT_NEAR(value1, value2, 1.e-6 * std::abs(value1));
}

INSTANTIATE_TEST_SUITE_P(DimensionOrderElementType, BilinearFormIntegratorTests,
                         ::testing::Values(std::make_tuple(1, 2, 0),
                                           std::make_tuple(2, 2, 0),
                                           std::make_tuple(2, 2, 1),
                                           std::make_tuple(3, 2, 0),
                                           std::make_tuple(3, 2, 1)));

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
