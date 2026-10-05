#include "TestCommon.hpp"

/*
  Tests for the discrete interpolators of bilininteg.hpp, which map an H1
  displacement to an L2 field of its deformation gradient, strain or
  deviatoric strain. A linear displacement u = A x has constant gradient,
  so each interpolated field must equal the exact constant (in the
  library's component order) to round-off on every element type.
*/

class InterpolatorTests : public testing::TestWithParam<DimOrderTypeTuple> {
 protected:
  void SetUp() {
    const auto& current_tuple = GetParam();

    dim = std::get<0>(current_tuple);
    order = std::get<1>(current_tuple);
    auto elementType = std::get<2>(current_tuple);

    mesh = MakeMesh(dim, elementType);

    L2 = std::make_unique<L2_FECollection>(order, dim);
    H1 = std::make_unique<H1_FECollection>(order + 1, dim);

    scalar_fes = std::make_unique<FiniteElementSpace>(&mesh, H1.get());
    vector_fes = std::make_unique<FiniteElementSpace>(&mesh, H1.get(), dim);
    matrix_fes =
        std::make_unique<FiniteElementSpace>(&mesh, L2.get(), dim * dim);
    strain_fes = std::make_unique<FiniteElementSpace>(
        &mesh, L2.get(), SymmetricComponentOrder::Count(dim));
    deviatoric_strain_fes = std::make_unique<FiniteElementSpace>(
        &mesh, L2.get(), SymmetricComponentOrder::TraceFreeCount(dim));

    A = RandomMatrix(dim);

    uF = std::make_unique<VectorFunctionCoefficient>(
        dim, [this](const Vector& x, Vector& u) {
          u.SetSize(x.Size());
          A.Mult(x, u);
        });

    FF = std::make_unique<VectorFunctionCoefficient>(
        dim * dim, [this](const Vector& x, Vector& m) {
          auto dim = x.Size();
          m.SetSize(dim * dim);
          auto k = 0;
          for (auto j = 0; j < dim; j++) {
            for (auto i = 0; i < dim; i++) {
              m(k++) = A(i, j);
            }
          }
        });

    EF = std::make_unique<VectorFunctionCoefficient>(
        SymmetricComponentOrder::Count(dim),
        [this](const Vector& x, Vector& m) {
          auto dim = x.Size();
          m.SetSize(SymmetricComponentOrder::Count(dim));
          auto k = 0;
          for (auto j = 0; j < dim; j++) {
            for (auto i = j; i < dim; i++) {
              m(k++) = 0.5 * (A(i, j) + A(j, i));
            }
          }
        });

    DF = std::make_unique<VectorFunctionCoefficient>(
        SymmetricComponentOrder::TraceFreeCount(dim),
        [this](const Vector& x, Vector& m) {
          auto dim = x.Size();
          m.SetSize(SymmetricComponentOrder::TraceFreeCount(dim));
          auto trace = A.Trace();
          auto k = 0;
          for (auto j = 0; j < dim - 1; j++) {
            for (auto i = j; i < dim; i++) {
              m(k) = 0.5 * (A(i, j) + A(j, i));
              if (i == j) {
                m(k) -= trace / dim;
              }
              k++;
            }
          }
        });
  }

  int order, dim;
  DenseMatrix A;
  Mesh mesh;
  std::unique_ptr<FiniteElementCollection> L2, H1;
  std::unique_ptr<FiniteElementSpace> scalar_fes, vector_fes, matrix_fes,
      strain_fes, deviatoric_strain_fes;
  std::unique_ptr<VectorFunctionCoefficient> uF, FF, EF, DF;
};

// Du = A, stored column-major (dim * dim components).
TEST_P(InterpolatorTests, DeformationGradientInterpolator) {
  auto u = GridFunction(vector_fes.get());
  u.ProjectCoefficient(*uF);

  auto b = DiscreteLinearOperator(vector_fes.get(), matrix_fes.get());
  b.AddDomainInterpolator(
      new AdGIA::DeformationGradientInterpolator());
  b.Assemble();

  auto F = GridFunction(matrix_fes.get());
  b.Mult(u, F);

  auto error = F.ComputeL2Error(*FF);
  EXPECT_TRUE(error < 1.e-8);
}

// sym A, lower triangle column by column (SymmetricComponentOrder).
TEST_P(InterpolatorTests, StrainInterpolator) {
  auto u = GridFunction(vector_fes.get());
  u.ProjectCoefficient(*uF);

  auto b = DiscreteLinearOperator(vector_fes.get(), strain_fes.get());
  b.AddDomainInterpolator(new AdGIA::StrainInterpolator());
  b.Assemble();

  auto E = GridFunction(strain_fes.get());
  b.Mult(u, E);

  auto error = E.ComputeL2Error(*EF);
  EXPECT_TRUE(error < 1.e-8);
}

// dev sym A, with the last diagonal component dropped (it is fixed by the
// zero trace).
TEST_P(InterpolatorTests, DeviatoricStrainInterpolator) {
  auto u = GridFunction(vector_fes.get());
  u.ProjectCoefficient(*uF);

  auto b =
      DiscreteLinearOperator(vector_fes.get(), deviatoric_strain_fes.get());
  b.AddDomainInterpolator(new AdGIA::DeviatoricStrainInterpolator());
  b.Assemble();

  auto D = GridFunction(deviatoric_strain_fes.get());
  b.Mult(u, D);

  auto error = D.ComputeL2Error(*DF);
  EXPECT_LT(error, 1.e-8);
}

INSTANTIATE_TEST_SUITE_P(DimensionOrderElementType, InterpolatorTests,
                         ::testing::Values(std::make_tuple(1, 1, 0),
                                           std::make_tuple(2, 1, 0),
                                           std::make_tuple(2, 1, 1),
                                           std::make_tuple(3, 1, 0),
                                           std::make_tuple(3, 1, 1)));

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
