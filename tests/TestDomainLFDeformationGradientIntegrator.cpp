
#include "TestCommon.hpp"

class LinearFormIntegratorTests
    : public ::testing::TestWithParam<DimOrderTypeTuple> {};

// int m : grad u for the matrix coefficient m_ij = (i+1)(j+2) x_i x_j and
// u = |x|^2 x, against the integral of the L2 interpolant of the
// pointwise contraction m : grad u, with grad u = 2 x x^T + |x|^2 1
// computed by hand; the two agree to interpolation error.
TEST_P(LinearFormIntegratorTests, DomainLFDeformationGradientIntegrator) {
  const auto& current_tuple = GetParam();

  auto dim = std::get<0>(current_tuple);
  auto order = std::get<1>(current_tuple);
  auto elementType = std::get<2>(current_tuple);

  auto mesh = MakeMesh(dim, elementType);

  auto H1 = H1_FECollection(order, dim);
  auto vector_fes = FiniteElementSpace(&mesh, &H1, dim);

  auto m = MatrixFunctionCoefficient(dim, [](const Vector& x, DenseMatrix& m) {
    auto dim = x.Size();
    m.SetSize(dim);
    for (auto j = 0; j < dim; j++) {
      for (auto i = 0; i < dim; i++) {
        m(i, j) = (i + 1) * (j + 2) * x(i) * x(j);
      }
    }
  });

  auto b = LinearForm(&vector_fes);
  b.AddDomainIntegrator(
      new AdGIA::DomainLFDeformationGradientIntegrator(m));
  b.Assemble();

  auto f = VectorFunctionCoefficient(dim, [](const Vector& x, Vector& y) {
    y = x;
    y *= x * x;
  });

  auto x = GridFunction(&vector_fes);
  x.ProjectCoefficient(f);
  auto value1 = b(x);

  auto L2 = L2_FECollection(order, dim);
  auto scalar_fes = FiniteElementSpace(&mesh, &L2);

  auto c = LinearForm(&scalar_fes);
  auto one = ConstantCoefficient(1);
  c.AddDomainIntegrator(new DomainLFIntegrator(one));
  c.Assemble();

  auto h = FunctionCoefficient([](const Vector& x) {
    auto dim = x.Size();
    auto f = x * x;
    auto sum = real_t{0};
    for (auto j = 0; j < dim; j++) {
      for (auto i = 0; i < dim; i++) {
        sum += (i + 1) * (j + 2) * x(i) * x(j) *
               (2 * x(i) * x(j) + f * (i == j ? 1 : 0));
      }
    }
    return sum;
  });

  auto y = GridFunction(&scalar_fes);
  y.ProjectCoefficient(h);
  auto value2 = c(y);

  EXPECT_NEAR(value1, value2, 1.e-6 * std::abs(value1));
}

// A MatrixDeltaCoefficient turns the integrator into a moment-tensor point
// source through MFEM's delta machinery: for ANY grid function v the
// assembled linear form gives exactly s * M : grad v(x_c).
TEST_P(LinearFormIntegratorTests, DeformationGradientPointSource) {
  const auto& current_tuple = GetParam();

  auto dim = std::get<0>(current_tuple);
  auto order = std::get<1>(current_tuple);
  auto elementType = std::get<2>(current_tuple);
  if (dim == 1) {
    GTEST_SKIP() << "point source test is for dim >= 2";
  }

  auto mesh = MakeMesh(dim, elementType);
  auto H1 = H1_FECollection(order, dim);
  auto vector_fes = FiniteElementSpace(&mesh, &H1, dim);

  // A generic source point strictly inside an element (the mesh is
  // Cartesian with spacing 0.05).
  Vector xc(dim);
  xc(0) = 0.31417;
  xc(1) = 0.27183;
  if (dim == 3) {
    xc(2) = 0.16180;
  }
  const real_t scale = 1.7;
  auto M = RandomMatrix(dim);
  auto md = dim == 2 ? AdGIA::MatrixDeltaCoefficient(M, xc(0), xc(1),
                                                              scale)
                     : AdGIA::MatrixDeltaCoefficient(
                           M, xc(0), xc(1), xc(2), scale);

  auto b = LinearForm(&vector_fes);
  b.AddDomainIntegrator(
      new AdGIA::DomainLFDeformationGradientIntegrator(md));
  b.Assemble();

  auto f = VectorFunctionCoefficient(dim, [](const Vector& x, Vector& y) {
    y = x;
    y *= std::sin(x(0)) + x * x;
  });
  auto v = GridFunction(&vector_fes);
  v.ProjectCoefficient(f);

  // The exact pairing: s * M : grad v_h at the source point.
  DenseMatrix pt(dim, 1);
  for (int d = 0; d < dim; d++) {
    pt(d, 0) = xc(d);
  }
  Array<int> elem;
  Array<IntegrationPoint> ips;
  ASSERT_EQ(mesh.FindPoints(pt, elem, ips), 1);
  auto* tr = mesh.GetElementTransformation(elem[0]);
  tr->SetIntPoint(&ips[0]);
  DenseMatrix grad(dim, dim);
  v.GetVectorGradient(*tr, grad);
  real_t expected = 0.0;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      expected += M(i, j) * grad(i, j);
    }
  }
  expected *= scale;

  EXPECT_NEAR(b(v), expected, 1e-12 * std::abs(expected));

  // The scale is live through the wrapped DeltaCoefficient.
  md.SetScale(2.0 * scale);
  b.Assemble();
  EXPECT_NEAR(b(v), 2.0 * expected, 1e-12 * std::abs(expected));
}

INSTANTIATE_TEST_SUITE_P(DimensionOrderElementType, LinearFormIntegratorTests,
                         ::testing::Values(std::make_tuple(1, 2, 0),
                                           std::make_tuple(2, 2, 0),
                                           std::make_tuple(2, 2, 1),
                                           std::make_tuple(3, 2, 0),
                                           std::make_tuple(3, 2, 1)));

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}