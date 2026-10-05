#include "TestCommon.hpp"

class BilinearFormIntegratorTests
    : public ::testing::TestWithParam<DimOrderTypeTuple> {};

// int (q . v) div u for q = x, v = x and u = |x| x, against
// DomainDivVectorScalarIntegrator applied to the L2 interpolant of q . v:
// the vector coefficient must enter only through its product with v.
TEST_P(BilinearFormIntegratorTests, DomainVectorDivVectorIntegrator) {
  const auto& current_tuple = GetParam();

  auto dim = std::get<0>(current_tuple);
  auto order = std::get<1>(current_tuple);
  auto elementType = std::get<2>(current_tuple);

  auto mesh = MakeMesh(dim, elementType);

  auto L2 = L2_FECollection(order, dim);
  auto H1 = H1_FECollection(order, dim);

  auto scalar_fes = FiniteElementSpace(&mesh, &L2);
  auto vector_fes0 = FiniteElementSpace(&mesh, &H1, dim);
  auto vector_fes1 = FiniteElementSpace(&mesh, &L2, dim);

  auto q =
      VectorFunctionCoefficient(dim, [](const Vector& x, Vector& y) { y = x; });

  auto b = MixedBilinearForm(&vector_fes0, &vector_fes1);
  b.AddDomainIntegrator(new DomainVectorDivVectorIntegrator(q));
  b.Assemble();

  auto uF = VectorFunctionCoefficient(dim, [](const Vector& x, Vector& y) {
    auto dim = x.Size();
    y = x;
    y *= x.Norml2();
  });
  auto u = GridFunction(&vector_fes0);
  u.ProjectCoefficient(uF);

  auto vF =
      VectorFunctionCoefficient(dim, [](const Vector& x, Vector& y) { y = x; });
  auto v = GridFunction(&vector_fes1);
  v.ProjectCoefficient(vF);

  auto z = GridFunction(&vector_fes1);
  b.Mult(u, z);
  auto value1 = z * v;

  auto wF = InnerProductCoefficient(q, vF);
  auto w = GridFunction(&scalar_fes);
  w.ProjectCoefficient(wF);

  auto c = MixedBilinearForm(&scalar_fes, &vector_fes0);
  c.AddDomainIntegrator(new DomainDivVectorScalarIntegrator());
  c.Assemble();

  c.Mult(w, z);

  auto value2 = u * z;

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
