#include <sched.h>

#include "TestCommon.hpp"

class BilinearFormIntegratorTests
    : public ::testing::TestWithParam<DimOrderTypeTuple> {
 public:
  void SetUp() {
    const auto& current_tuple = GetParam();

    dim = std::get<0>(current_tuple);
    order = std::get<1>(current_tuple);
    elementType = std::get<2>(current_tuple);

    mesh = MakeMesh(dim, elementType);

    L2 = new L2_FECollection(order, dim);
    H1 = new H1_FECollection(order, dim);

    scalar_fes = new FiniteElementSpace(&mesh, H1);
    vector_fes = new FiniteElementSpace(&mesh, L2, dim);
  }

  void TearDown() {
    // The spaces first: they refer to the collections.
    delete vector_fes;
    delete scalar_fes;
    delete H1;
    delete L2;
  }

  int dim, order, elementType;
  Mesh mesh;
  FiniteElementCollection *L2, *H1;
  FiniteElementSpace *scalar_fes, *vector_fes;
};

// int v . (q grad u) for u = |x| and the three coefficient types, each
// against MFEM's DomainLFGradIntegrator with the vector coefficient q^T v
// acting on u. Scalar q = |x|, v = (1, ..., 1).
TEST_P(BilinearFormIntegratorTests,
       DomainVectorGradScalarIntegrator_ScalarCoefficient) {
  const auto& current_tuple = GetParam();

  auto q = FunctionCoefficient([](const Vector& x) { return x.Norml2(); });

  auto b = MixedBilinearForm(scalar_fes, vector_fes);
  b.AddDomainIntegrator(new DomainVectorGradScalarIntegrator(q));
  b.Assemble();

  auto u = GridFunction(scalar_fes);
  auto uF = FunctionCoefficient([](const Vector& x) { return x.Norml2(); });
  u.ProjectCoefficient(uF);

  auto v = GridFunction(vector_fes);
  auto vF = VectorFunctionCoefficient(dim, [](const Vector& x, Vector& y) {
    y.SetSize(x.Size());
    y = 1.;
  });
  v.ProjectCoefficient(vF);

  auto w = GridFunction(vector_fes);
  b.Mult(u, w);
  auto value1 = v * w;

  auto zF = ScalarVectorProductCoefficient(q, vF);
  auto l = LinearForm(scalar_fes);
  l.AddDomainIntegrator(new DomainLFGradIntegrator(zF));
  l.Assemble();
  auto value2 = l * u;

  EXPECT_NEAR(value1, value2, 1.e-6 * std::abs(value1));
}

// Vector q = a (a diagonal matrix), v = x.
TEST_P(BilinearFormIntegratorTests,
       DomainVectorGradScalarIntegrator_VectorCoefficient) {
  auto a = RandomVector(dim);
  auto qv = VectorConstantCoefficient(a);

  auto b = MixedBilinearForm(scalar_fes, vector_fes);
  b.AddDomainIntegrator(new DomainVectorGradScalarIntegrator(qv));
  b.Assemble();

  auto u = GridFunction(scalar_fes);
  auto uF = FunctionCoefficient([](const Vector& x) { return x.Norml2(); });
  u.ProjectCoefficient(uF);

  auto v = GridFunction(vector_fes);
  auto vF = VectorFunctionCoefficient(dim, [](const Vector& x, Vector& y) {
    y.SetSize(x.Size());
    y = x;
  });
  v.ProjectCoefficient(vF);

  auto w = GridFunction(vector_fes);
  b.Mult(u, w);
  auto value1 = v * w;

  auto zF = VectorFunctionCoefficient(dim, [a](const Vector& x, Vector& y) {
    auto dim = x.Size();
    y.SetSize(dim);
    for (auto i = 0; i < dim; i++) {
      y(i) = a(i) * x(i);
    }
  });

  auto l = mfem::LinearForm(scalar_fes);
  l.AddDomainIntegrator(new mfem::DomainLFGradIntegrator(zF));
  l.Assemble();
  auto value2 = l * u;

  EXPECT_NEAR(value1, value2, 1.e-6 * std::abs(value1));
}

// Full matrix q = A (random, non-symmetric), v = x: the reference uses
// A^T v, which checks the side on which the matrix acts.
TEST_P(BilinearFormIntegratorTests,
       DomainVectorGradScalarIntegrator_MatrixCoefficient) {
  auto A = RandomMatrix(dim);
  auto qm = MatrixConstantCoefficient(A);

  auto b = MixedBilinearForm(scalar_fes, vector_fes);
  b.AddDomainIntegrator(new DomainVectorGradScalarIntegrator(qm));
  b.Assemble();

  auto u = mfem::GridFunction(scalar_fes);
  auto uF = mfem::FunctionCoefficient(
      [](const mfem::Vector& x) { return x.Norml2(); });
  u.ProjectCoefficient(uF);

  auto v = mfem::GridFunction(vector_fes);
  auto vF = mfem::VectorFunctionCoefficient(
      dim, [](const mfem::Vector& x, mfem::Vector& y) {
        y.SetSize(x.Size());
        y = x;
      });
  v.ProjectCoefficient(vF);

  auto w = mfem::GridFunction(vector_fes);
  b.Mult(u, w);
  auto value1 = v * w;

  auto qmt = TransposeMatrixCoefficient(qm);
  auto zF = MatrixVectorProductCoefficient(qmt, vF);

  auto l = mfem::LinearForm(scalar_fes);
  l.AddDomainIntegrator(new mfem::DomainLFGradIntegrator(zF));
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
