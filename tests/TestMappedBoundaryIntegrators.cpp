// Pull-back identity tests (doc/mappings.md, "The discrete
// change-of-variables identity") for the boundary (Nanson) machinery: for a smooth non-polynomial mapping interpolated on
// the mesh's geometric space, referential coefficients and one
// integration rule on both sides, each mapped boundary term assembled on
// the reference mesh must equal its standard counterpart on the mapped
// mesh to round-off. (F2) and (F3) are the fluid-interface terms of
// doc/self_gravitation.md, "Bilinear form". Covered:
//  - BoundaryNormalScalarIntegrator (the (F3) coupling): m dS -> nu dS
//    exactly;
//  - BoundaryNormalNormalIntegrator with the full (F2) composition
//    (referential rho times MappedBoundaryNormalDotCoefficient):
//    (nu.u)(nu.u')/|nu| dS, exercising the adjacent-element gradient of
//    GridFunctionDiffeomorphism on boundary transformations;
//  - the stock boundary linear forms with NansonAreaCoefficient (scalar
//    load) and its ScalarVectorProduct composition (vector load).
#include <numbers>

#include "TestCommon.hpp"

namespace {

constexpr real_t pi = std::numbers::pi_v<real_t>;

Mesh SmallMesh(int dim, int elementType) {
  if (dim == 2) {
    return Mesh::MakeCartesian2D(
        6, 6, elementType == 0 ? Element::TRIANGLE : Element::QUADRILATERAL);
  }
  return Mesh::MakeCartesian3D(
      4, 4, 4, elementType == 0 ? Element::TETRAHEDRON : Element::HEXAHEDRON);
}

// A smooth, non-polynomial diffeomorphism with exact gradient that moves
// the boundary: xi_i = x_i + c sin(pi x_j), j = (i + 1) mod dim.
CallableDiffeomorphism SmoothMap(int dim, real_t c) {
  return CallableDiffeomorphism(
      dim,
      [c, dim](const Vector& x, Vector& y) {
        for (int i = 0; i < dim; i++) {
          y(i) = x(i) + c * std::sin(pi * x((i + 1) % dim));
        }
      },
      [c, dim](const Vector& x, DenseMatrix& F) {
        F = 0.0;
        for (int i = 0; i < dim; i++) {
          F(i, i) = 1.0;
          F(i, (i + 1) % dim) = c * pi * std::cos(pi * x((i + 1) % dim));
        }
      });
}

}  // namespace

class MappedBoundaryIntegratorsTest
    : public ::testing::TestWithParam<DimOrderTypeTuple> {};

TEST_P(MappedBoundaryIntegratorsTest, MatchesMappedMesh) {
  const auto [dim, order, elementType] = GetParam();
  H1_FECollection fec(order, dim);

  auto mesh = SmallMesh(dim, elementType);
  mesh.SetCurvature(order);
  FiniteElementSpace sfes(&mesh, &fec);
  FiniteElementSpace vfes(&mesh, &fec, dim);

  auto xi = SmoothMap(dim, 0.05);
  auto xi_h = Interpolate(xi, mesh);
  auto mapped = MappedMesh(mesh, xi);
  FiniteElementSpace sfes_m(&mapped, &fec);
  FiniteElementSpace vfes_m(&mapped, &fec, dim);

  const IntegrationRule& ir =
      IntRules.Get(mesh.GetBdrElementGeometry(0), 2 * order + 3);

  // Physical coefficients and their referential expressions.
  auto rho_fn = [dim](const Vector& x) {
    real_t s = 0.0;
    for (int i = 0; i < dim; i++) s += (i + 1) * x(i);
    return 1.5 + 0.5 * std::sin(s);
  };
  auto g_fn = [dim](const Vector& x, Vector& g) {
    for (int i = 0; i < dim; i++) {
      g(i) = 0.4 + std::cos(x(i) - 0.5 * x((i + 1) % dim)) + 0.1 * i;
    }
  };
  FunctionCoefficient rho_phys(rho_fn);
  VectorFunctionCoefficient g_phys(dim, g_fn);
  TransformedFunctionCoefficient rho_ref(xi_h, rho_fn);
  TransformedVectorFunctionCoefficient g_ref(xi_h, g_fn);

  // (F3): int q p (m.v) dS.
  {
    MixedBilinearForm a_ref(&sfes, &vfes);
    a_ref.AddBoundaryIntegrator(
        new BoundaryNormalScalarIntegrator(rho_ref, xi_h, &ir));
    a_ref.Assemble();
    a_ref.Finalize();

    MixedBilinearForm a_map(&sfes_m, &vfes_m);
    a_map.AddBoundaryIntegrator(
        new BoundaryNormalScalarIntegrator(rho_phys, &ir));
    a_map.Assemble();
    a_map.Finalize();

    const real_t scale = a_map.SpMat().MaxNorm();
    ASSERT_GT(scale, 0.0);
    EXPECT_LT(MaxDiff(a_ref.SpMat(), a_map.SpMat()), 1e-12 * scale)
        << "BoundaryNormalScalar";
  }

  // (F2) composition: int rho (m.g)(m.u)(m.u') dS.
  {
    MappedBoundaryNormalDotCoefficient mg_ref(g_ref, xi_h);
    ProductCoefficient q_ref(rho_ref, mg_ref);
    BoundaryNormalDotCoefficient mg_phys(g_phys);
    ProductCoefficient q_phys(rho_phys, mg_phys);

    BilinearForm a_ref(&vfes);
    a_ref.AddBoundaryIntegrator(
        new BoundaryNormalNormalIntegrator(q_ref, xi_h, &ir));
    a_ref.Assemble();
    a_ref.Finalize();

    BilinearForm a_map(&vfes_m);
    a_map.AddBoundaryIntegrator(
        new BoundaryNormalNormalIntegrator(q_phys, &ir));
    a_map.Assemble();
    a_map.Finalize();

    const real_t scale = a_map.SpMat().MaxNorm();
    ASSERT_GT(scale, 0.0);
    EXPECT_LT(MaxDiff(a_ref.SpMat(), a_map.SpMat()), 1e-12 * scale)
        << "BoundaryNormalNormal (F2 composition)";
  }

  // Scalar load: int sigma p dS via NansonAreaCoefficient.
  NansonAreaCoefficient area(xi_h);
  {
    ProductCoefficient sig_ref(rho_ref, area);

    LinearForm b_ref(&sfes);
    auto* lfi_ref = new BoundaryLFIntegrator(sig_ref);
    lfi_ref->SetIntRule(&ir);
    b_ref.AddBoundaryIntegrator(lfi_ref);
    b_ref.Assemble();

    LinearForm b_map(&sfes_m);
    auto* lfi_map = new BoundaryLFIntegrator(rho_phys);
    lfi_map->SetIntRule(&ir);
    b_map.AddBoundaryIntegrator(lfi_map);
    b_map.Assemble();

    const real_t scale = b_map.Normlinf();
    ASSERT_GT(scale, 0.0);
    b_ref -= b_map;
    EXPECT_LT(b_ref.Normlinf(), 1e-12 * scale) << "BoundaryLF with area";
  }

  // Vector load: int f . v dS via the ScalarVectorProduct composition.
  {
    ScalarVectorProductCoefficient f_ref(area, g_ref);

    LinearForm b_ref(&vfes);
    auto* lfi_ref = new VectorBoundaryLFIntegrator(f_ref);
    lfi_ref->SetIntRule(&ir);
    b_ref.AddBoundaryIntegrator(lfi_ref);
    b_ref.Assemble();

    LinearForm b_map(&vfes_m);
    auto* lfi_map = new VectorBoundaryLFIntegrator(g_phys);
    lfi_map->SetIntRule(&ir);
    b_map.AddBoundaryIntegrator(lfi_map);
    b_map.Assemble();

    const real_t scale = b_map.Normlinf();
    ASSERT_GT(scale, 0.0);
    b_ref -= b_map;
    EXPECT_LT(b_ref.Normlinf(), 1e-12 * scale) << "VectorBoundaryLF with area";
  }
}

INSTANTIATE_TEST_SUITE_P(
    MappedBoundaryIntegrators, MappedBoundaryIntegratorsTest,
    ::testing::Combine(::testing::Values(2, 3), ::testing::Values(1, 2, 3),
                       ::testing::Values(0, 1)));
