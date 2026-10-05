#include <numbers>

#include "TestCommon.hpp"

/*
  Tests for the boundary integrators and coefficients used by the fluid–solid
  interface terms of the self-gravitating problem (doc/self_gravitation.md,
  "Fluid–solid interface conditions"):

  - BoundaryNormalNormalIntegrator  (u, v) -> int q (n.u)(n.v) dS
  - BoundaryNormalScalarIntegrator  (v, p) -> int q p (n.v) dS
  - BoundaryNormalDotCoefficient    V.n on boundary elements
  - BarotropicDensityGradientCoefficient  grad rho . grad Phi0 / |grad Phi0|^2

  On the unit square/cube (flat faces, exact normals) the values on
  interpolated polynomials are checked against closed forms to round-off,
  the mixed integrator against MFEM's VectorBoundaryFluxLFIntegrator, and
  the TransposeIntegrator against the explicit transpose. On the canned
  order-2 gmsh meshes the values on a sphere are checked against the
  analytic surface integrals to the geometric error.

  Orientation: the normal is the SubMesh's outward normal on an inherited
  interior interface (the three-layer disc: ICB from the inner core, CMB
  from the mantle; the 3-D shell's inner sphere) and on a cut interface
  (a Cartesian mesh split at x = 0.5), from either side.
*/

namespace {

constexpr double kPi = std::numbers::pi;

double LinearQ(const Vector& x) {
  double v = 1.0;
  for (int i = 0; i < x.Size(); i++) {
    v += (i + 1) * x[i];
  }
  return v;
}

void Position(const Vector& x, Vector& v) { v = x; }

// Value of the interpolant of (v)^T A (u) on L-vectors.
double FormValue(const SparseMatrix& A, const GridFunction& v,
                 const GridFunction& u) {
  Vector Au(A.Height());
  A.Mult(u, Au);
  return Au * v;
}

struct Spaces {
  std::unique_ptr<H1_FECollection> fec;
  std::unique_ptr<FiniteElementSpace> vec, scal;
  Spaces(Mesh& mesh, int order) {
    const int dim = mesh.Dimension();
    fec = std::make_unique<H1_FECollection>(order, dim);
    vec = std::make_unique<FiniteElementSpace>(&mesh, fec.get(), dim);
    scal = std::make_unique<FiniteElementSpace>(&mesh, fec.get());
  }
};

Array<int> AllBoundaries(Mesh& mesh) {
  Array<int> m(mesh.bdr_attributes.Max());
  m = 1;
  return m;
}

Array<int> OneBoundary(Mesh& mesh, int attr) {
  Array<int> m(mesh.bdr_attributes.Max());
  m = 0;
  m[attr - 1] = 1;
  return m;
}

// (dim, elementType, order)
class CartesianTest : public testing::TestWithParam<DimOrderTypeTuple> {};

TEST_P(CartesianTest, NormalNormalExact) {
  const auto [dim, elementType, order] = GetParam();
  auto mesh = MakeMesh(dim, elementType);
  Spaces sp(mesh, order);
  FunctionCoefficient q(LinearQ);

  BilinearForm a(sp.vec.get());
  a.AddBoundaryIntegrator(new BoundaryNormalNormalIntegrator(q));
  a.Assemble();
  a.Finalize();
  const SparseMatrix& A = a.SpMat();

  // Symmetry.
  std::unique_ptr<SparseMatrix> At(Transpose(A));
  EXPECT_LT(MaxDiff(A, *At), 1e-14 * A.MaxNorm());

  // u = x, v = a: n.x = 1 on the faces x_d = 1 and 0 on x_d = 0, so the
  // value is sum_d a_d int_{x_d = 1} q, and q is linear so each face
  // integral is q at the face centre (unit faces).
  VectorFunctionCoefficient xc(dim, Position);
  GridFunction u(sp.vec.get()), v(sp.vec.get());
  u.ProjectCoefficient(xc);
  Vector avec = RandomVector(dim);
  VectorConstantCoefficient ac(avec);
  v.ProjectCoefficient(ac);
  double expected = 0.0;
  for (int d = 0; d < dim; d++) {
    Vector c(dim);
    c = 0.5;
    c[d] = 1.0;
    expected += avec[d] * LinearQ(c);
  }
  EXPECT_NEAR(FormValue(A, v, u), expected, 1e-12 * std::abs(expected));
}

TEST_P(CartesianTest, NormalScalarExact) {
  const auto [dim, elementType, order] = GetParam();
  auto mesh = MakeMesh(dim, elementType);
  Spaces sp(mesh, order);
  FunctionCoefficient q(LinearQ);

  MixedBilinearForm c(sp.scal.get(), sp.vec.get());
  c.AddBoundaryIntegrator(new BoundaryNormalScalarIntegrator(q));
  c.Assemble();
  c.Finalize();
  const SparseMatrix& C = c.SpMat();

  // p = 1, v = a: sum over all faces of q (n.a) = sum_d a_d (q at the
  // centre of x_d = 1 minus q at the centre of x_d = 0) = sum_d a_d (d+1).
  GridFunction p(sp.scal.get()), v(sp.vec.get());
  p = 1.0;
  Vector avec = RandomVector(dim);
  VectorConstantCoefficient ac(avec);
  v.ProjectCoefficient(ac);
  double expected = 0.0;
  for (int d = 0; d < dim; d++) {
    expected += avec[d] * (d + 1);
  }
  EXPECT_NEAR(FormValue(C, v, p), expected, 1e-12 * (1.0 + std::abs(expected)));

  // C times the constant one equals MFEM's (q, n.v) boundary flux form.
  Vector C1(C.Height());
  C.Mult(p, C1);
  LinearForm l(sp.vec.get());
  l.AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(q));
  l.Assemble();
  C1 -= l;
  EXPECT_LT(C1.Normlinf(), 1e-13 * l.Normlinf());

  // The transpose integrator gives the transposed matrix.
  MixedBilinearForm ct(sp.vec.get(), sp.scal.get());
  ct.AddBoundaryIntegrator(
      new TransposeIntegrator(new BoundaryNormalScalarIntegrator(q)));
  ct.Assemble();
  ct.Finalize();
  std::unique_ptr<SparseMatrix> Ct(Transpose(C));
  EXPECT_LT(MaxDiff(ct.SpMat(), *Ct), 1e-14 * C.MaxNorm());
}

TEST_P(CartesianTest, NormalDotCoefficient) {
  const auto [dim, elementType, order] = GetParam();
  auto mesh = MakeMesh(dim, elementType);
  Spaces sp(mesh, order);

  // q = x.n is 1 on the faces x_d = 1 and 0 on x_d = 0, so with p = 1 and
  // v = a the mixed form gives sum_d a_d. The same through the gradient
  // of Phi0 = |x|^2 / 2 (exact for order >= 2), exercising the
  // boundary-element gradient path.
  VectorFunctionCoefficient xc(dim, Position);
  BoundaryNormalDotCoefficient q(xc);
  auto check = [&](Coefficient& coef, double tol) {
    MixedBilinearForm c(sp.scal.get(), sp.vec.get());
    c.AddBoundaryIntegrator(new BoundaryNormalScalarIntegrator(coef));
    c.Assemble();
    c.Finalize();
    GridFunction p(sp.scal.get()), v(sp.vec.get());
    p = 1.0;
    Vector avec = RandomVector(dim);
    VectorConstantCoefficient ac(avec);
    v.ProjectCoefficient(ac);
    EXPECT_NEAR(FormValue(c.SpMat(), v, p), avec.Sum(), tol);
  };
  check(q, 1e-11);

  if (order >= 2) {
    FunctionCoefficient half_r2(
        [](const Vector& x) { return 0.5 * (x * x); });
    GridFunction phi0(sp.scal.get());
    phi0.ProjectCoefficient(half_r2);
    GradientGridFunctionCoefficient grad(&phi0);
    BoundaryNormalDotCoefficient qg(grad);
    check(qg, 1e-11);
  }
}

TEST_P(CartesianTest, BarotropicDensityGradient) {
  const auto [dim, elementType, order] = GetParam();
  if (order < 2) {
    return;  // Phi0 quadratic
  }
  auto mesh = MakeMesh(dim, elementType);
  Spaces sp(mesh, order);
  L2_FECollection l2fec(order, dim);
  FiniteElementSpace l2(&mesh, &l2fec);

  // rho = 1 + 2 x0 (L2), Phi0 = x0^2 + x1 (H1): rho' = 4 x0 / (4 x0^2 + 1).
  FunctionCoefficient rho_c([](const Vector& x) { return 1.0 + 2.0 * x[0]; });
  FunctionCoefficient phi_c(
      [](const Vector& x) { return x[0] * x[0] + x[1]; });
  GridFunction rho(&l2), phi0(sp.scal.get());
  rho.ProjectCoefficient(rho_c);
  phi0.ProjectCoefficient(phi_c);
  BarotropicDensityGradientCoefficient rp(rho, phi0);

  GradientGridFunctionCoefficient grad_rho(&rho), grad_phi0(&phi0);
  BarotropicDensityGradientCoefficient rp2(grad_rho, grad_phi0);

  for (int e = 0; e < mesh.GetNE(); e += 7) {
    ElementTransformation* T = mesh.GetElementTransformation(e);
    const IntegrationPoint& ip =
        Geometries.GetCenter(mesh.GetElementGeometry(e));
    T->SetIntPoint(&ip);
    Vector x(dim);
    T->Transform(ip, x);
    const double expected = 4.0 * x[0] / (4.0 * x[0] * x[0] + 1.0);
    EXPECT_NEAR(rp.Eval(*T, ip), expected, 1e-12);
    EXPECT_NEAR(rp2.Eval(*T, ip), expected, 1e-12);
  }
}

INSTANTIATE_TEST_SUITE_P(
    BoundaryNormal, CartesianTest,
    testing::Combine(testing::Values(2, 3), testing::Values(0, 1),
                     testing::Values(1, 2, 3)));

// --- spheres ---------------------------------------------------------------

// The canned two-layer meshes: attribute 1 the unit body, attribute 2 the
// buffer shell (out to radius 1.2 in 2-D, 2 in 3-D); bdr attribute 1 the
// unit sphere, 2 the outer one.
std::string TwoLayerMesh(int dim) {
  return dim == 2 ? "../data/elastogravity_2d.msh"
                  : "../data/coupled_poisson.msh";
}

double SphereArea(int dim, double r) {
  return dim == 2 ? 2.0 * kPi * r : 4.0 * kPi * r * r;
}

TEST(BoundaryNormalSphere, AnalyticValues) {
  for (int dim : {2, 3}) {
    Mesh parent(TwoLayerMesh(dim).c_str(), 1, 1);
    ASSERT_EQ(parent.Dimension(), dim);
    Array<int> body({1});
    SubMesh mesh(SubMesh::CreateFromDomain(parent, body));
    const int order = 2;
    Spaces sp(mesh, order);
    VectorFunctionCoefficient xc(dim, Position);
    GridFunction u(sp.vec.get()), p(sp.scal.get());
    u.ProjectCoefficient(xc);
    p = 1.0;
    // Order-2 geometry: the boundary is a piecewise-quadratic sphere; the
    // 3-D mesh is coarse (h ~ 0.2 on the unit sphere).
    const double tol = dim == 2 ? 1e-4 : 2e-3;

    // int (n.x)^2 dS = |S|.
    BilinearForm a(sp.vec.get());
    a.AddBoundaryIntegrator(new BoundaryNormalNormalIntegrator());
    a.Assemble();
    a.Finalize();
    EXPECT_NEAR(FormValue(a.SpMat(), u, u), SphereArea(dim, 1.0),
                tol * SphereArea(dim, 1.0));

    // int (x.n) (n.x)^2 dS = |S| with q = x.n.
    BoundaryNormalDotCoefficient q(xc);
    BilinearForm aq(sp.vec.get());
    aq.AddBoundaryIntegrator(new BoundaryNormalNormalIntegrator(q));
    aq.Assemble();
    aq.Finalize();
    EXPECT_NEAR(FormValue(aq.SpMat(), u, u), SphereArea(dim, 1.0),
                tol * SphereArea(dim, 1.0));

    // int 1 (n.x) dS = |S|; with q = x.n the same.
    MixedBilinearForm c(sp.scal.get(), sp.vec.get());
    c.AddBoundaryIntegrator(new BoundaryNormalScalarIntegrator());
    c.Assemble();
    c.Finalize();
    EXPECT_NEAR(FormValue(c.SpMat(), u, p), SphereArea(dim, 1.0),
                tol * SphereArea(dim, 1.0));
  }
}

// --- orientation on SubMesh interfaces ------------------------------------

// int (m.x) dS over the boundary attribute attr of the SubMesh, with m the
// integrator's normal: p = 1, v = x through the mixed integrator.
double NormalFlux(SubMesh& mesh, int attr, int order) {
  Spaces sp(mesh, order);
  VectorFunctionCoefficient xc(mesh.Dimension(), Position);
  GridFunction u(sp.vec.get()), p(sp.scal.get());
  u.ProjectCoefficient(xc);
  p = 1.0;
  Array<int> marker = OneBoundary(mesh, attr);
  MixedBilinearForm c(sp.scal.get(), sp.vec.get());
  c.AddBoundaryIntegrator(new BoundaryNormalScalarIntegrator(), marker);
  c.Assemble();
  c.Finalize();
  return FormValue(c.SpMat(), u, p);
}

TEST(BoundaryNormalOrientation, InheritedInterfaces2D) {
  // Three-layer disc: attributes 1 (inner core), 2 (outer core), 3
  // (mantle), 4 (shell); boundary attributes 1 (ICB), 2 (CMB), 3 (surface),
  // 4 (outer). One disconnected SubMesh for the two solid regions.
  Mesh parent("../data/elastogravity_three_layer_2d.msh", 1, 1);
  ASSERT_EQ(parent.Dimension(), 2);
  Array<int> solid({1, 3});
  SubMesh mesh(SubMesh::CreateFromDomain(parent, solid));
  ASSERT_EQ(mesh.bdr_attributes.Size(), 3);
  ASSERT_EQ(mesh.bdr_attributes.Max(), 3);

  // Radii from the mesh vertices on each boundary attribute.
  auto radius = [&](int attr) {
    for (int b = 0; b < mesh.GetNBE(); b++) {
      if (mesh.GetBdrAttribute(b) == attr) {
        Array<int> v;
        mesh.GetBdrElementVertices(b, v);
        Vector x(mesh.GetVertex(v[0]), 2);
        return x.Norml2();
      }
    }
    return 0.0;
  };
  const double r_icb = radius(1), r_cmb = radius(2), r_s = radius(3);
  ASSERT_LT(r_icb, r_cmb);
  ASSERT_LT(r_cmb, r_s);

  // Order 2 = the geometric order of the mesh, so that the interpolant of
  // x is the mesh geometry itself; at order 1 the interpolant follows the
  // chords and the fluxes are short by the sagitta (about 1% here).
  {
    const int order = 2;
    const double tol = 1e-4;
    // ICB from the inner core: m = +r_hat; CMB from the mantle: m = -r_hat.
    EXPECT_NEAR(NormalFlux(mesh, 1, order), 2.0 * kPi * r_icb * r_icb,
                tol * 2.0 * kPi * r_icb * r_icb);
    EXPECT_NEAR(NormalFlux(mesh, 2, order), -2.0 * kPi * r_cmb * r_cmb,
                tol * 2.0 * kPi * r_cmb * r_cmb);
    EXPECT_NEAR(NormalFlux(mesh, 3, order), 2.0 * kPi * r_s * r_s,
                tol * 2.0 * kPi * r_s * r_s);
  }
}

TEST(BoundaryNormalOrientation, InheritedInterface3D) {
  // The shell (attribute 2) of the ball-in-ball mesh: its inner sphere
  // (bdr attribute 1, radius 1) is an inherited interior interface with
  // m = -r_hat; the outer sphere (attribute 2, radius 2) has m = +r_hat.
  Mesh parent("../data/coupled_poisson.msh", 1, 1);
  ASSERT_EQ(parent.Dimension(), 3);
  Array<int> shell({2});
  SubMesh mesh(SubMesh::CreateFromDomain(parent, shell));
  ASSERT_EQ(mesh.bdr_attributes.Max(), 2);
  const double tol = 2e-3;
  EXPECT_NEAR(NormalFlux(mesh, 1, 2), -4.0 * kPi, tol * 4.0 * kPi);
  EXPECT_NEAR(NormalFlux(mesh, 2, 2), 4.0 * kPi * 8.0, tol * 4.0 * kPi * 8.0);
}

TEST(BoundaryNormalOrientation, CutInterface) {
  // Cartesian mesh (even element counts) split at x = 0.5 (attribute 2 for
  // x < 0.5, 1 above); the cut carries bdr attribute max + 1 on either
  // SubMesh. From the left m = +e_x so int m.x = 0.5 |cut| = 0.5; from the
  // right -0.5.
  for (int dim : {2, 3}) {
    for (int elementType : {0, 1}) {
      auto parent =
          dim == 2 ? Mesh::MakeCartesian2D(4, 4,
                                           elementType == 0
                                               ? Element::TRIANGLE
                                               : Element::QUADRILATERAL)
                   : Mesh::MakeCartesian3D(4, 4, 4,
                                           elementType == 0
                                               ? Element::TETRAHEDRON
                                               : Element::HEXAHEDRON);
      for (int i = 0; i < parent.GetNE(); i++) {
        Vector c(dim);
        parent.GetElementCenter(i, c);
        parent.SetAttribute(i, c[0] < 0.5 ? 2 : 1);
      }
      parent.SetAttributes();
      const int cut = parent.bdr_attributes.Max() + 1;
      for (int order : {1, 2}) {
        Array<int> left({2}), right({1});
        SubMesh ml(SubMesh::CreateFromDomain(parent, left));
        SubMesh mr(SubMesh::CreateFromDomain(parent, right));
        ASSERT_EQ(ml.bdr_attributes.Max(), cut);
        ASSERT_EQ(mr.bdr_attributes.Max(), cut);
        EXPECT_NEAR(NormalFlux(ml, cut, order), 0.5, 1e-13);
        EXPECT_NEAR(NormalFlux(mr, cut, order), -0.5, 1e-13);
      }
    }
  }
}

}  // namespace
