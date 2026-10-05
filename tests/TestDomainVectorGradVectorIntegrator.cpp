#include "TestCommon.hpp"

class BilinearFormIntegratorTests
    : public ::testing::TestWithParam<DimOrderTypeTuple> {};

// int q v . grad(w . u) for q = |x|, w = x and u = x, so grad(w . u) =
// 2 x, against MFEM's VectorMassIntegrator with weight q between v = x and
// the interpolant of 2 x. The nodal interpolant of w . u = |x|^2 is exact
// at order 2, so the two agree to quadrature.
TEST_P(BilinearFormIntegratorTests, DomainVectorGradVectorIntegrator) {
  const auto& current_tuple = GetParam();

  auto dim = std::get<0>(current_tuple);
  auto order = std::get<1>(current_tuple);
  auto elementType = std::get<2>(current_tuple);

  auto mesh = MakeMesh(dim, elementType);

  auto L2 = L2_FECollection(order, dim);
  auto H1 = H1_FECollection(order, dim);

  auto scalar_fes = FiniteElementSpace(&mesh, &L2);
  auto vector_fes1 = FiniteElementSpace(&mesh, &H1, dim);
  auto vector_fes2 = FiniteElementSpace(&mesh, &L2, dim);

  auto q = FunctionCoefficient([](const Vector& x) { return x.Norml2(); });
  auto qv =
      VectorFunctionCoefficient(dim, [](const Vector& x, Vector& y) { y = x; });

  auto b = MixedBilinearForm(&vector_fes1, &vector_fes2);
  b.AddDomainIntegrator(new DomainVectorGradVectorIntegrator(qv, q));
  b.Assemble();

  auto uF =
      VectorFunctionCoefficient(dim, [](const Vector& x, Vector& y) { y = x; });
  auto u = GridFunction(&vector_fes1);
  u.ProjectCoefficient(uF);

  auto vF =
      VectorFunctionCoefficient(dim, [](const Vector& x, Vector& y) { y = x; });
  auto v = GridFunction(&vector_fes2);
  v.ProjectCoefficient(vF);

  auto w = GridFunction(&vector_fes2);
  b.Mult(u, w);

  auto value1 = v * w;

  auto yF = VectorFunctionCoefficient(
      dim, [](const Vector& x, Vector& y) { y.Set(2, x); });
  auto y = GridFunction(&vector_fes2);
  y.ProjectCoefficient(yF);

  auto c = BilinearForm(&vector_fes2);
  c.AddDomainIntegrator(new VectorMassIntegrator(q));
  c.Assemble();
  auto value2 = c.InnerProduct(y, v);

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
