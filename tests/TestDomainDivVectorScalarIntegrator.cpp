#include "TestCommon.hpp"

class BilinearFormIntegratorTests
    : public ::testing::TestWithParam<DimOrderTypeTuple> {};

// int q (div v) u for v = x (div v = dim) and u the L2 interpolant of |x|,
// against MFEM's DomainLFIntegrator with weight q dim acting on u: the
// two routes integrate the same polynomial-times-coefficient integrand.
TEST_P(BilinearFormIntegratorTests, DomainDivVectorScalarIntegrator) {
  const auto& current_tuple = GetParam();

  auto dim = std::get<0>(current_tuple);
  auto order = std::get<1>(current_tuple);
  auto elementType = std::get<2>(current_tuple);

  auto mesh = MakeMesh(dim, elementType);

  auto L2 = L2_FECollection(order, dim);
  auto H1 = H1_FECollection(order, dim);

  auto scalar_fes = FiniteElementSpace(&mesh, &L2);
  auto vector_fes = FiniteElementSpace(&mesh, &H1, dim);

  auto q = FunctionCoefficient([](const Vector& x) { return x.Norml2(); });
  auto b = MixedBilinearForm(&scalar_fes, &vector_fes);
  b.AddDomainIntegrator(new DomainDivVectorScalarIntegrator(q));
  b.Assemble();

  auto uF = FunctionCoefficient([](const Vector& x) { return x.Norml2(); });
  auto u = GridFunction(&scalar_fes);
  u.ProjectCoefficient(uF);

  auto vF =
      VectorFunctionCoefficient(dim, [](const Vector& x, Vector& y) { y = x; });
  auto v = GridFunction(&vector_fes);
  v.ProjectCoefficient(vF);

  auto w = GridFunction(&vector_fes);
  b.Mult(u, w);

  auto value1 = v * w;

  auto xF = FunctionCoefficient([](const Vector& x) { return x.Size(); });
  auto yF = ProductCoefficient(q, xF);

  auto l = LinearForm(&scalar_fes);
  l.AddDomainIntegrator(new DomainLFIntegrator(yF));
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
