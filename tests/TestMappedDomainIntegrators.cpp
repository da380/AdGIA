// Pull-back identity tests (doc/mappings.md, "The discrete
// change-of-variables identity") for the generalised domain integrators of bilininteg.hpp: for a smooth
// non-polynomial mapping interpolated on the mesh's geometric space, with
// referential coefficients (physical fields composed with the interpolated
// mapping) and one integration rule on both sides, each mapped integrator
// assembled on the reference mesh must equal its standard counterpart
// assembled on the mapped mesh to round-off — at every order and element
// type, in 2-D and 3-D. Covered: DomainVectorScalar, DomainVectorGradScalar,
// DomainDivVectorScalar, DomainDivVectorDivVector, DomainVectorGradVector
// and DomainVectorDivVector.
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

// A smooth, non-polynomial diffeomorphism with exact gradient:
// xi_i = x_i + c sin(pi x_j), j = (i + 1) mod dim.
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

std::unique_ptr<SparseMatrix> AssembleMixed(FiniteElementSpace& trial,
                                            FiniteElementSpace& test,
                                            BilinearFormIntegrator* bi) {
  MixedBilinearForm a(&trial, &test);
  a.AddDomainIntegrator(bi);
  a.Assemble();
  a.Finalize();
  return std::make_unique<SparseMatrix>(a.SpMat());
}

}  // namespace

class MappedDomainIntegratorsTest
    : public ::testing::TestWithParam<DimOrderTypeTuple> {};

TEST_P(MappedDomainIntegratorsTest, MatchesMappedMesh) {
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
      IntRules.Get(mesh.GetTypicalElementGeometry(), 2 * order + 3);

  // Physical coefficients and their referential expressions through the
  // interpolated mapping.
  auto q_fn = [dim](const Vector& x) {
    real_t s = 0.0;
    for (int i = 0; i < dim; i++) s += (i + 1) * x(i);
    return 1.5 + std::sin(s);
  };
  auto qv_fn = [dim](const Vector& x, Vector& v) {
    for (int i = 0; i < dim; i++) {
      v(i) = 0.5 + std::cos(x(i) + 0.3 * x((i + 1) % dim)) + 0.2 * i;
    }
  };
  FunctionCoefficient q_phys(q_fn);
  VectorFunctionCoefficient qv_phys(dim, qv_fn);
  TransformedFunctionCoefficient q_ref(xi_h, q_fn);
  TransformedVectorFunctionCoefficient qv_ref(xi_h, qv_fn);

  auto Check = [&](std::unique_ptr<SparseMatrix> A_ref,
                   std::unique_ptr<SparseMatrix> A_map, const char* name) {
    const real_t scale = A_map->MaxNorm();
    ASSERT_GT(scale, 0.0) << name;
    EXPECT_LT(MaxDiff(*A_ref, *A_map), 1e-12 * scale) << name;
  };

  // 1. int q . v u dx  (no derivatives: the J factor alone).
  Check(AssembleMixed(sfes, vfes,
                      new DomainVectorScalarIntegrator(qv_ref, xi_h, &ir)),
        AssembleMixed(sfes_m, vfes_m,
                      new DomainVectorScalarIntegrator(qv_phys, &ir)),
        "DomainVectorScalar");

  // 2. int q v . grad u dx  (trial gradient mapped).
  Check(AssembleMixed(sfes, vfes,
                      new DomainVectorGradScalarIntegrator(q_ref, xi_h, &ir)),
        AssembleMixed(sfes_m, vfes_m,
                      new DomainVectorGradScalarIntegrator(q_phys, &ir)),
        "DomainVectorGradScalar");

  // 3. int q div(v) u dx  (test divergence mapped).
  Check(AssembleMixed(sfes, vfes,
                      new DomainDivVectorScalarIntegrator(q_ref, xi_h, &ir)),
        AssembleMixed(sfes_m, vfes_m,
                      new DomainDivVectorScalarIntegrator(q_phys, &ir)),
        "DomainDivVectorScalar");

  // 4. int q div(v) div(u) dx  (both divergences mapped).
  Check(AssembleMixed(vfes, vfes,
                      new DomainDivVectorDivVectorIntegrator(q_ref, xi_h, &ir)),
        AssembleMixed(vfes_m, vfes_m,
                      new DomainDivVectorDivVectorIntegrator(q_phys, &ir)),
        "DomainDivVectorDivVector");

  // 5. int q v . grad(w . u) dx  (gradient of the nodally interpolated
  //    scalar mapped; w evaluated at trial nodes, whose physical positions
  //    coincide on the two sides).
  Check(AssembleMixed(vfes, vfes,
                      new DomainVectorGradVectorIntegrator(qv_ref, q_ref, xi_h,
                                                           &ir)),
        AssembleMixed(vfes_m, vfes_m,
                      new DomainVectorGradVectorIntegrator(qv_phys, q_phys,
                                                           &ir)),
        "DomainVectorGradVector");

  // 6. int (q . v) div(u) dx  (trial divergence mapped).
  Check(AssembleMixed(vfes, vfes,
                      new DomainVectorDivVectorIntegrator(qv_ref, xi_h, &ir)),
        AssembleMixed(vfes_m, vfes_m,
                      new DomainVectorDivVectorIntegrator(qv_phys, &ir)),
        "DomainVectorDivVector");
}

INSTANTIATE_TEST_SUITE_P(
    MappedDomainIntegrators, MappedDomainIntegratorsTest,
    ::testing::Combine(::testing::Values(2, 3), ::testing::Values(1, 2, 3),
                       ::testing::Values(0, 1)));
