#include "TestCommon.hpp"

class BilinearFormIntegratorTests
    : public ::testing::TestWithParam<DimOrderTypeTuple> {};

// int q div u div v for u = x and v = (sin x_i), against MFEM's mass form
// with weight q between the L2 interpolants of div u and div v: the
// integrator must reproduce the form built from independently computed
// divergences, up to interpolation error.
TEST_P(BilinearFormIntegratorTests, DomainDivVectorDivVectorIntegrator) {
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
  auto b = BilinearForm(&vector_fes);
  b.AddDomainIntegrator(new DomainDivVectorDivVectorIntegrator(q));
  b.Assemble();

  auto uF =

      VectorFunctionCoefficient(dim, [](const Vector& x, Vector& y) { y = x; });
  auto u = GridFunction(&vector_fes);
  u.ProjectCoefficient(uF);

  auto vF = VectorFunctionCoefficient(dim, [](const Vector& x, Vector& y) {
    auto dim = x.Size();
    y.SetSize(dim);
    for (auto i = 0; i < dim; i++) {
      y(i) = std::sin(x(i));
    }
  });
  auto v = GridFunction(&vector_fes);
  v.ProjectCoefficient(vF);

  auto value1 = b.InnerProduct(u, v);

  auto DivuF = FunctionCoefficient([](const Vector& x) { return x.Size(); });
  auto DivvF = FunctionCoefficient([](const Vector& x) {
    auto dim = x.Size();
    auto sum = real_t{0};
    for (auto i = 0; i < dim; i++) {
      sum += std::cos(x(i));
    }
    return sum;
  });

  auto Divu = GridFunction(&scalar_fes);
  Divu.ProjectCoefficient(DivuF);

  auto Divv = GridFunction(&scalar_fes);
  Divv.ProjectCoefficient(DivvF);

  auto c = BilinearForm(&scalar_fes);
  c.AddDomainIntegrator(new MassIntegrator(q));
  c.Assemble();

  auto value2 = c.InnerProduct(Divu, Divv);

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
