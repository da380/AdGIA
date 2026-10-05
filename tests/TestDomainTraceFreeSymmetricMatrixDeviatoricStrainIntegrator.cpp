#include "TestCommon.hpp"

class BilinearFormIntegratorTests
    : public ::testing::TestWithParam<DimOrderTypeTuple> {};

// int q v : dev(sym(grad u)) for u = B x and a constant symmetric
// trace-free field v = A, stored without its last diagonal component,
// against DomainLFDeformationGradientIntegrator with q A (equal because A
// is symmetric and trace-free): checks the trace-free packing, in which
// the omitted diagonal entry is minus the sum of the others.
TEST_P(BilinearFormIntegratorTests,
       DomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator) {
  const auto& current_tuple = GetParam();

  auto dim = std::get<0>(current_tuple);
  auto order = std::get<1>(current_tuple);
  auto elementType = std::get<2>(current_tuple);

  auto mesh = MakeMesh(dim, elementType);

  auto L2 = L2_FECollection(order, dim);
  auto H1 = H1_FECollection(order, dim);

  auto vector_fes = FiniteElementSpace(&mesh, &H1, dim);
  auto deviatoric_strain_fes = FiniteElementSpace(
      &mesh, &L2, SymmetricComponentOrder::TraceFreeCount(dim));

  auto q = FunctionCoefficient([](const Vector& x) { return x.Norml2(); });

  auto b = MixedBilinearForm(&vector_fes, &deviatoric_strain_fes);
  b.AddDomainIntegrator(
      new DomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator(q));
  b.Assemble();

  auto B = RandomMatrix(dim);
  auto uF = VectorFunctionCoefficient(
      dim, [B](const Vector& x, Vector& y) { B.Mult(x, y); });
  auto u = GridFunction(&vector_fes);
  u.ProjectCoefficient(uF);

  auto A = RandomMatrix(dim);
  A.Symmetrize();
  auto trace = A.Trace();
  for (auto i = 0; i < dim; i++) {
    A(i, i) -= trace / dim;
  }

  auto a = Vector(SymmetricComponentOrder::TraceFreeCount(dim));
  auto k = 0;
  for (auto j = 0; j < dim - 1; j++) {
    for (auto i = j; i < dim; i++) {
      a(k++) = A(i, j);
    }
  }
  auto vF = VectorConstantCoefficient(a);

  auto v = GridFunction(&deviatoric_strain_fes);
  v.ProjectCoefficient(vF);

  auto w = GridFunction(&deviatoric_strain_fes);
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
