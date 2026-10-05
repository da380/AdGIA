// Tests for TransformedDiffusionIntegrator (bilininteg.hpp): the pull-back of the
// Laplace form under a diffeomorphism xi(x). Three families of checks:
//
//  1. The four ways of specifying the mapping agree. For a radial map
//     xi = f(x) x with f affine, the scalar path (f given), the Diffeomorphism
//     path (exact F) and the matrix path (a = J F^{-1} F^{-T} given
//     analytically) must coincide to round-off at every order; the vector path
//     (xi given as a plain VectorCoefficient, F from its trial-space
//     interpolant) joins them once xi = f x lies in the trial space
//     (order >= 2). The comparison is sensitive to the index order in the
//     scalar-path Jacobian F(j,k) = x_j d_k f.
//
//  2. The discrete change-of-variables identity (doc/mappings.md, "The
//     discrete change-of-variables identity"):
//     for an affine map the transformed form on the reference mesh equals the
//     ordinary DiffusionIntegrator on the mapped mesh outright; for a smooth
//     non-polynomial map the same holds — for the stiffness, the J rho mass
//     term and the J rho load — with the mapping interpolated on the mesh's
//     geometric space and one integration rule passed to both sides, at the
//     matrix level and through a full solve.
//
//  3. Exact-F convergence: with the mapping analytic the solved pull-back
//     converges to the pulled-back exact solution under refinement.
#include <numbers>

#include "TestCommon.hpp"

namespace {

Mesh SmallMesh(int dim, int elementType) {
  if (dim == 2) {
    return Mesh::MakeCartesian2D(
        6, 6, elementType == 0 ? Element::TRIANGLE : Element::QUADRILATERAL);
  }
  return Mesh::MakeCartesian3D(
      4, 4, 4, elementType == 0 ? Element::TETRAHEDRON : Element::HEXAHEDRON);
}

// Affine radial scale f(x) = f0 + g . x.
struct AffineRadialMap {
  real_t f0;
  Vector g;
  real_t f(const Vector& x) const { return f0 + (g * x); }
  // F = f I + x g^T, then a = det(F) F^{-1} F^{-T}.
  void a(const Vector& x, DenseMatrix& A) const {
    const int dim = x.Size();
    DenseMatrix F(dim);
    for (int j = 0; j < dim; j++) {
      for (int k = 0; k < dim; k++) {
        F(j, k) = x(j) * g(k);
      }
      F(j, j) += f(x);
    }
    const real_t J = F.Det();
    F.Invert();
    A.SetSize(dim);
    MultABt(F, F, A);
    A *= J;
  }
};

class AffineRadialMatrixCoefficient : public MatrixCoefficient {
 public:
  AffineRadialMatrixCoefficient(int dim, const AffineRadialMap& map)
      : MatrixCoefficient(dim), map_(map), x_(dim) {}
  void Eval(DenseMatrix& K, ElementTransformation& T,
            const IntegrationPoint& ip) override {
    T.Transform(ip, x_);
    map_.a(x_, K);
  }

 private:
  AffineRadialMap map_;
  Vector x_;
};

std::unique_ptr<SparseMatrix> Assemble(FiniteElementSpace& fes,
                                       BilinearFormIntegrator* integ) {
  BilinearForm a(&fes);
  a.AddDomainIntegrator(integ);
  a.Assemble();
  a.Finalize();
  return std::make_unique<SparseMatrix>(a.SpMat());
}

}  // namespace

class TransformedDiffusionTest
    : public ::testing::TestWithParam<DimOrderTypeTuple> {};

// 1. Scalar / vector / matrix specifications of the same radial map agree.
TEST_P(TransformedDiffusionTest, MappingPathsAgree) {
  const auto [dim, order, elementType] = GetParam();
  auto mesh = SmallMesh(dim, elementType);
  H1_FECollection fec(order, dim);
  FiniteElementSpace fes(&mesh, &fec);

  AffineRadialMap map;
  map.f0 = 1.0;
  map.g.SetSize(dim);
  map.g(0) = 0.3;
  map.g(1) = -0.2;
  if (dim == 3) map.g(2) = 0.15;

  FunctionCoefficient f_coeff(
      [map](const Vector& x) -> real_t { return map.f(x); });
  VectorConstantCoefficient g_coeff(map.g);
  RadialDiffeomorphism xi_exact(dim, f_coeff, g_coeff);
  VectorFunctionCoefficient xi_coeff(dim, [map](const Vector& x, Vector& y) {
    y = x;
    y *= map.f(x);
  });
  AffineRadialMatrixCoefficient a_coeff(dim, map);

  auto A_scalar = Assemble(fes, new TransformedDiffusionIntegrator(f_coeff));
  auto A_vector = Assemble(fes, new TransformedDiffusionIntegrator(xi_coeff));
  auto A_exact = Assemble(fes, new TransformedDiffusionIntegrator(xi_exact));
  auto A_matrix = Assemble(fes, new TransformedDiffusionIntegrator(a_coeff));

  const real_t scale = A_matrix->MaxNorm();
  ASSERT_GT(scale, 0.0);
  const real_t tol = 1e-12 * scale;

  // f is affine, so its nodal interpolant is exact at every order: the scalar
  // path must reproduce the analytic a(x) to round-off.
  EXPECT_LT(MaxDiff(*A_scalar, *A_matrix), tol);

  // The Diffeomorphism path has F analytically at every order.
  EXPECT_LT(MaxDiff(*A_exact, *A_matrix), tol);

  // xi = f x is quadratic; its nodal interpolant is exact for order >= 2.
  if (order >= 2) {
    EXPECT_LT(MaxDiff(*A_vector, *A_matrix), tol);
  }

  // The mapped Laplacian must still be symmetric.
  std::unique_ptr<SparseMatrix> At(Transpose(*A_scalar));
  EXPECT_LT(MaxDiff(*A_scalar, *At), tol);
}

// 2. Affine maps: pull-back on the reference mesh == Laplacian on the mapped
//    mesh. Uses a uniform scaling through the scalar path and a general
//    affine map through the vector path.
TEST_P(TransformedDiffusionTest, MatchesMappedMesh) {
  const auto [dim, order, elementType] = GetParam();
  H1_FECollection fec(order, dim);

  // (a) xi = c x through the scalar path.
  {
    const real_t c = 1.7;
    auto mesh = SmallMesh(dim, elementType);
    FiniteElementSpace fes(&mesh, &fec);
    ConstantCoefficient c_coeff(c);
    auto A_ref = Assemble(fes, new TransformedDiffusionIntegrator(c_coeff));

    auto mapped = SmallMesh(dim, elementType);
    mapped.SetCurvature(order);
    VectorFunctionCoefficient scale(dim, [c](const Vector& x, Vector& y) {
      y = x;
      y *= c;
    });
    mapped.Transform(scale);
    FiniteElementSpace fes_mapped(&mapped, &fec);
    ConstantCoefficient one(1.0);
    auto A_mapped = Assemble(fes_mapped, new DiffusionIntegrator(one));

    EXPECT_LT(MaxDiff(*A_ref, *A_mapped), 1e-12 * A_mapped->MaxNorm());
  }

  // (b) xi = M x + b through the vector path, M a fixed non-symmetric matrix
  //     with positive determinant.
  {
    DenseMatrix M(dim);
    M = 0.0;
    for (int i = 0; i < dim; i++) M(i, i) = 1.0 + 0.2 * i;
    M(0, 1) = 0.4;
    M(1, 0) = -0.1;
    if (dim == 3) {
      M(0, 2) = 0.25;
      M(2, 1) = 0.3;
    }
    ASSERT_GT(M.Det(), 0.0);
    Vector b(dim);
    b = 0.5;

    VectorFunctionCoefficient xi(dim, [M, b](const Vector& x, Vector& y) {
      y.SetSize(x.Size());
      M.Mult(x, y);
      y += b;
    });

    auto mesh = SmallMesh(dim, elementType);
    FiniteElementSpace fes(&mesh, &fec);
    auto A_ref = Assemble(fes, new TransformedDiffusionIntegrator(xi));

    auto mapped = SmallMesh(dim, elementType);
    mapped.SetCurvature(order);
    mapped.Transform(xi);
    FiniteElementSpace fes_mapped(&mapped, &fec);
    ConstantCoefficient one(1.0);
    auto A_mapped = Assemble(fes_mapped, new DiffusionIntegrator(one));

    EXPECT_LT(MaxDiff(*A_ref, *A_mapped), 1e-12 * A_mapped->MaxNorm());
  }
}

INSTANTIATE_TEST_SUITE_P(
    TransformedDiffusion, TransformedDiffusionTest,
    ::testing::Combine(::testing::Values(2, 3), ::testing::Values(1, 2, 3),
                       ::testing::Values(0, 1)));

namespace {

constexpr real_t pi = std::numbers::pi_v<real_t>;

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

Vector AssembleLF(FiniteElementSpace& fes, LinearFormIntegrator* integ) {
  LinearForm b(&fes);
  b.AddDomainIntegrator(integ);
  b.Assemble();
  return Vector(b);
}

}  // namespace

// 2. (continued) The discrete change-of-variables identity for a smooth
//    non-polynomial map: interpolate the mapping on the mesh's geometric
//    space, pass one integration rule to both sides, and the stiffness,
//    the J rho mass term and the J rho load on the reference mesh equal
//    their standard counterparts on the mapped mesh to round-off.
TEST_P(TransformedDiffusionTest, NonAffineMatchesMappedMesh) {
  const auto [dim, order, elementType] = GetParam();
  H1_FECollection fec(order, dim);

  auto mesh = SmallMesh(dim, elementType);
  mesh.SetCurvature(order);
  FiniteElementSpace fes(&mesh, &fec);

  auto xi = SmoothMap(dim, 0.05);
  auto xi_h = Interpolate(xi, mesh);
  auto mapped = MappedMesh(mesh, xi);
  FiniteElementSpace fes_mapped(&mapped, &fec);

  // One rule for both sides: the defaults need not coincide.
  const Geometry::Type geom = mesh.GetTypicalElementGeometry();
  const IntegrationRule& ir = IntRules.Get(geom, 2 * order + 3);

  // A physical density, and its referential expression rho o xi with the
  // Jacobian factor.
  auto rho_fn = [dim](const Vector& x) {
    real_t s = 0.0;
    for (int i = 0; i < dim; i++) s += (i + 1) * x(i);
    return 2.0 + std::sin(s);
  };
  FunctionCoefficient rho_phys(rho_fn);
  TransformedFunctionCoefficient rho_ref(xi_h, rho_fn);
  JacobianCoefficient J_h(xi_h);
  ProductCoefficient Jrho_ref(J_h, rho_ref);

  // Stiffness.
  auto A_ref = Assemble(fes, new TransformedDiffusionIntegrator(xi_h, &ir));
  ConstantCoefficient one(1.0);
  auto A_mapped = Assemble(fes_mapped, new DiffusionIntegrator(one, &ir));
  EXPECT_LT(MaxDiff(*A_ref, *A_mapped), 1e-12 * A_mapped->MaxNorm());

  // Mass with J rho.
  auto M_ref = Assemble(fes, new MassIntegrator(Jrho_ref, &ir));
  auto M_mapped = Assemble(fes_mapped, new MassIntegrator(rho_phys, &ir));
  EXPECT_LT(MaxDiff(*M_ref, *M_mapped), 1e-12 * M_mapped->MaxNorm());

  // Load with J rho.
  auto b_ref = AssembleLF(fes, new DomainLFIntegrator(Jrho_ref, &ir));
  auto b_mapped = AssembleLF(fes_mapped, new DomainLFIntegrator(rho_phys, &ir));
  b_ref -= b_mapped;
  EXPECT_LT(b_ref.Normlinf(), 1e-12 * b_mapped.Normlinf());
}

// 2. (continued) The identity through a full Dirichlet solve: with the
//    interpolated mapping, the pull-back solved on the reference mesh and
//    the standard problem solved on the mapped mesh give the same dofs.
TEST(TransformedDiffusionSolve, MatchesMappedMeshSolve) {
  for (int dim = 2; dim <= 3; dim++) {
    const int order = 2;
    H1_FECollection fec(order, dim);

    auto mesh = SmallMesh(dim, 0);
    mesh.SetCurvature(order);
    FiniteElementSpace fes(&mesh, &fec);

    auto xi = SmoothMap(dim, 0.05);
    auto xi_h = Interpolate(xi, mesh);
    auto mapped = MappedMesh(mesh, xi);
    FiniteElementSpace fes_mapped(&mapped, &fec);

    const Geometry::Type geom = mesh.GetTypicalElementGeometry();
    const IntegrationRule& ir = IntRules.Get(geom, 2 * order + 3);

    Array<int> bdr_marker(mesh.bdr_attributes.Max());
    bdr_marker = 1;
    Array<int> ess_tdof_list;
    fes.GetEssentialTrueDofs(bdr_marker, ess_tdof_list);

    // Dirichlet data: the physical g and its pull-back; the nodal values
    // agree because the mapped mesh's nodes are the images of the
    // reference mesh's.
    auto g_fn = [dim](const Vector& x) { return x(0) * x(dim - 1); };
    FunctionCoefficient g_phys(g_fn);
    TransformedFunctionCoefficient g_ref(xi_h, g_fn);

    auto Solve = [&](FiniteElementSpace& s, BilinearFormIntegrator* integ,
                     Coefficient& g) {
      GridFunction u(&s);
      u.ProjectCoefficient(g);
      BilinearForm a(&s);
      a.AddDomainIntegrator(integ);
      a.Assemble();
      LinearForm b(&s);
      b.Assemble();
      SparseMatrix A;
      Vector B, X;
      a.FormLinearSystem(ess_tdof_list, u, b, A, X, B);
      GSSmoother P(A);
      CGSolver cg;
      cg.SetRelTol(1e-13);
      cg.SetMaxIter(5000);
      cg.SetPrintLevel(0);
      cg.SetPreconditioner(P);
      cg.SetOperator(A);
      cg.Mult(B, X);
      a.RecoverFEMSolution(X, b, u);
      return u;
    };

    auto zeta = Solve(fes, new TransformedDiffusionIntegrator(xi_h, &ir),
                      g_ref);
    ConstantCoefficient one(1.0);
    auto u = Solve(fes_mapped, new DiffusionIntegrator(one, &ir), g_phys);

    zeta -= u;
    EXPECT_LT(zeta.Normlinf(), 1e-8 * u.Normlinf());
  }
}

// 3. Exact-F convergence: solving the pull-back with an analytic mapping,
//    the error against the pulled-back harmonic solution drops under
//    refinement (order 2: L2 error ~ h^3, so halving h gains ~8x; 0.3 is
//    a safe bound).
TEST(TransformedDiffusionSolve, ExactMappingConverges) {
  for (int dim = 2; dim <= 3; dim++) {
    const int order = 2;
    H1_FECollection fec(order, dim);

    auto xi = SmoothMap(dim, 0.05);
    auto g_fn = [dim](const Vector& x) { return x(dim - 2) * x(dim - 1); };
    TransformedFunctionCoefficient g_ref(xi, g_fn);

    auto ErrorAt = [&](int n) {
      auto mesh = dim == 2 ? Mesh::MakeCartesian2D(n, n, Element::TRIANGLE)
                           : Mesh::MakeCartesian3D(n, n, n,
                                                   Element::TETRAHEDRON);
      FiniteElementSpace fes(&mesh, &fec);

      Array<int> bdr_marker(mesh.bdr_attributes.Max());
      bdr_marker = 1;
      Array<int> ess_tdof_list;
      fes.GetEssentialTrueDofs(bdr_marker, ess_tdof_list);

      GridFunction zeta(&fes);
      zeta.ProjectCoefficient(g_ref);
      BilinearForm a(&fes);
      a.AddDomainIntegrator(new TransformedDiffusionIntegrator(xi));
      a.Assemble();
      LinearForm b(&fes);
      b.Assemble();
      SparseMatrix A;
      Vector B, X;
      a.FormLinearSystem(ess_tdof_list, zeta, b, A, X, B);
      GSSmoother P(A);
      CGSolver cg;
      cg.SetRelTol(1e-13);
      cg.SetMaxIter(5000);
      cg.SetPrintLevel(0);
      cg.SetPreconditioner(P);
      cg.SetOperator(A);
      cg.Mult(B, X);
      a.RecoverFEMSolution(X, b, zeta);
      return zeta.ComputeL2Error(g_ref);
    };

    const real_t coarse = ErrorAt(dim == 2 ? 8 : 4);
    const real_t fine = ErrorAt(dim == 2 ? 16 : 8);
    EXPECT_LT(fine, 0.3 * coarse);
  }
}
