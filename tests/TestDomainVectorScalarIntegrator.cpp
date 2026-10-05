#include "TestCommon.hpp"

class BilinearFormIntegratorTests
    : public ::testing::TestWithParam<DimOrderTypeTuple> {};

// int (q . v) u for q = x, v = x and u the H1 interpolant of |x|, against
// MFEM's VectorDomainLFIntegrator with the coefficient |x| q acting on v
// (the exact |x| in place of its interpolant, so the two agree to
// interpolation error).
TEST_P(BilinearFormIntegratorTests, DomainVectorScalarIntegrator) {
  const auto& current_tuple = GetParam();

  auto dim = std::get<0>(current_tuple);
  auto order = std::get<1>(current_tuple);
  auto elementType = std::get<2>(current_tuple);

  auto mesh = MakeMesh(dim, elementType);

  auto L2 = L2_FECollection(order, dim);
  auto H1 = H1_FECollection(order, dim);

  auto scalar_fes = FiniteElementSpace(&mesh, &H1);
  auto vector_fes = FiniteElementSpace(&mesh, &L2, dim);

  auto qv = VectorFunctionCoefficient(dim, [](const Vector& x, Vector& y) {
    y.SetSize(x.Size());
    y = x;
  });

  auto b = MixedBilinearForm(&scalar_fes, &vector_fes);
  b.AddDomainIntegrator(new DomainVectorScalarIntegrator(qv));
  b.Assemble();

  auto u = GridFunction(&scalar_fes);
  auto uF = FunctionCoefficient([](const Vector& x) { return x.Norml2(); });
  u.ProjectCoefficient(uF);

  auto v = GridFunction(&vector_fes);
  auto vF = VectorFunctionCoefficient(dim, [](const Vector& x, Vector& y) {
    y.SetSize(x.Size());
    y = x;
  });
  v.ProjectCoefficient(vF);

  auto w = GridFunction(&vector_fes);

  b.Mult(u, w);
  auto value1 = v * w;

  auto rv = ScalarVectorProductCoefficient(uF, qv);
  auto l = LinearForm(&vector_fes);
  l.AddDomainIntegrator(new VectorDomainLFIntegrator(rv));
  l.Assemble();

  auto value2 = l * v;

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
