#include <numbers>

#include "MixedProblemTestCommon.hpp"
#include "TestCommon.hpp"

/*
  Tests for spherical_harmonics.hpp: SurfaceHarmonics,
  HarmonicExpansionCoefficient and BoundaryHarmonicCoefficients, on the
  surface of the body SubMesh of the self-gravitating test meshes (order-2
  gmsh spheres/circles) and on the parent mesh's outer boundary.

  - Index map: Index(l, m) inverts Degree/Order; 2-D and 3-D sizes.
  - Degree one against x_hat, to full relative accuracy next to the polar
    axis (where 1 - cos^2 would lose the sine).
  - Surface gradient: EvalWithGradient against central differences of the
    solid harmonics r^l Y_i, at generic points, on the polar axis and at the
    centre.
  - Orthonormality: the coefficients of the synthesised harmonic Y_i on the
    curved surface are e_i to the geometry's accuracy.
  - Against PoissonDtNOperator::HarmonicCoefficients on the outer boundary
    (the same basis; the 2-D DtN operator carries no degree-zero term).
  - Radial component: u = Y_i n gives e_i, from a VectorCoefficient and from
    its interpolant on the vector space.
  - Tangential component: u = grad_1 Y_i gives e_i, whatever radial part
    and rotation are added, and nothing at degree zero.
  - LoadVector against a BoundaryLFIntegrator of the same expansion.
  - Interior harmonic continuation carries the factor (r/R)^l.
*/

namespace {

using namespace self_grav_test;

int MaxDegree(int dim) { return dim == 2 ? 6 : 3; }
// Tolerances on the order-2 curved test meshes: boundary integrals of
// exact fields (geometry only), and of their order-2 interpolants.
double GeomTol(int dim) { return dim == 2 ? 1e-6 : 3e-3; }
double InterpTol(int dim) { return dim == 2 ? 1e-3 : 5e-2; }

struct HarmonicCase {
  std::unique_ptr<Mesh> parent;
  std::unique_ptr<SubMesh> body;
  H1_FECollection fec;
  std::unique_ptr<FiniteElementSpace> scalar, vector, phi;
  Array<int> surface, outer;
  int dim, L;

  explicit HarmonicCase(int d)
      : fec(2, d), dim(d), L(MaxDegree(d)) {
    parent = std::make_unique<Mesh>(MeshFile(dim).c_str(), 1, 1);
    body = std::make_unique<SubMesh>(
        SubMesh::CreateFromDomain(*parent, BodyMarker(*parent)));
    scalar = std::make_unique<FiniteElementSpace>(body.get(), &fec);
    vector = std::make_unique<FiniteElementSpace>(body.get(), &fec, dim);
    phi = std::make_unique<FiniteElementSpace>(parent.get(), &fec);
    surface = SurfaceMarker(*body);
    outer = ExternalBoundaryMarker(parent.get());
  }
};

Vector Unit(int n, int i) {
  Vector e(n);
  e = 0.0;
  e[i] = 1.0;
  return e;
}

class SphericalHarmonicsTest : public testing::TestWithParam<int> {};

TEST_P(SphericalHarmonicsTest, IndexMap) {
  const int dim = GetParam();
  SurfaceHarmonics basis(dim, 4);
  EXPECT_EQ(basis.Size(), dim == 2 ? 9 : 25);
  for (int i = 0; i < basis.Size(); i++) {
    EXPECT_EQ(basis.Index(basis.Degree(i), basis.Order(i)), i);
    EXPECT_LE(std::abs(basis.Order(i)), basis.Degree(i));
  }
  Vector x(dim), Y;
  x = 0.3;
  x[dim - 1] = 0.9;
  basis.Eval(x, Y);
  EXPECT_EQ(Y.Size(), basis.Size());
  EXPECT_NEAR(Y[0], 1.0 / std::sqrt(dim == 2 ? 2.0 * std::numbers::pi : 4.0 * std::numbers::pi),
              1e-14);
  // Direction only.
  Vector Y2;
  x *= 3.0;
  basis.Eval(x, Y2);
  Y2 -= Y;
  EXPECT_LT(Y2.Normlinf(), 1e-13);
}

TEST(SphericalHarmonicsAccuracy, DegreeOneNearThePolarAxis) {
  // Y_{1,0} = c z/r, Y_{1,1} = -c x/r, Y_{1,-1} = -c y/r, c = sqrt(3/(4 pi)).
  SurfaceHarmonics basis(3, 4);
  const real_t c = std::sqrt(3.0 / (4.0 * std::numbers::pi));
  for (real_t pole : {1.0, -1.0}) {
    for (real_t eps : {1e-3, 1e-6, 1e-9}) {
      Vector x(3), Y;
      x[0] = eps;
      x[1] = -2.0 * eps;
      x[2] = pole;
      basis.Eval(x, Y);
      const real_t r = x.Norml2();
      EXPECT_NEAR(Y[basis.Index(1, 0)], c * x[2] / r, 1e-14);
      EXPECT_NEAR(Y[basis.Index(1, 1)] / (-c * x[0] / r), 1.0, 1e-13)
          << "eps = " << eps;
      EXPECT_NEAR(Y[basis.Index(1, -1)] / (-c * x[1] / r), 1.0, 1e-13)
          << "eps = " << eps;
    }
  }
}

TEST_P(SphericalHarmonicsTest, SurfaceGradient) {
  const int dim = GetParam();
  const int L = 5;
  SurfaceHarmonics basis(dim, L);
  const int n = basis.Size();

  // The solid harmonics r^l Y_i are polynomials, so smooth everywhere.
  auto solid = [&](const Vector& x, Vector& S) {
    basis.Eval(x, S);
    const real_t r = x.Norml2();
    for (int i = 0; i < n; i++) {
      S[i] *= std::pow(r, basis.Degree(i));
    }
  };

  std::vector<Vector> points;
  for (int k = 0; k < 4; k++) {
    Vector x(dim);
    for (int d = 0; d < dim; d++) {
      x[d] = std::sin(1.7 * k + 2.3 * d + 0.4);
    }
    points.push_back(x);
  }
  // The polar axis (both poles) and the centre.
  for (real_t z : {0.8, -1.3, 0.0}) {
    Vector x(dim);
    x = 0.0;
    x[dim - 1] = z;
    points.push_back(x);
  }

  // Sixth-order central differences: exact for these polynomials (degree
  // <= 5), so the step can be large and the comparison sharp.
  const real_t h = 0.1;
  const real_t stencil[3] = {3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0};
  for (const auto& x : points) {
    Vector Y, Sp, Sm;
    DenseMatrix gradY;
    basis.EvalWithGradient(x, Y, gradY);
    ASSERT_EQ(gradY.Height(), dim);
    ASSERT_EQ(gradY.Width(), n);
    Vector Y_only;
    basis.Eval(x, Y_only);
    const real_t r = x.Norml2();
    for (int i = 0; i < n; i++) {
      EXPECT_EQ(Y[i], Y_only[i]);
      const int l = basis.Degree(i);
      real_t radial = 0.0;
      for (int d = 0; d < dim; d++) {
        real_t fd = 0.0;
        for (int k = 1; k <= 3; k++) {
          Vector xp(x), xm(x);
          xp[d] += k * h;
          xm[d] -= k * h;
          solid(xp, Sp);
          solid(xm, Sm);
          fd += stencil[k - 1] * (Sp[i] - Sm[i]) / h;
        }
        // grad(r^l Y) = r^(l-1) (l Y x_hat + grad_1 Y); at the centre only
        // l = 1 survives, and x_hat is the direction theta = 0 that Eval
        // takes there (the polar axis in 3-D, the x axis in 2-D).
        const int axis = dim == 2 ? 0 : 2;
        real_t xhat = r > 0 ? x[d] / r : (d == axis ? 1.0 : 0.0);
        real_t rpow = l == 0 ? 0.0 : std::pow(r, l - 1);
        EXPECT_NEAR(rpow * (l * Y[i] * xhat + gradY(d, i)), fd, 1e-11)
            << "i = " << i << ", d = " << d << ", r = " << r;
        radial += xhat * gradY(d, i);
      }
      EXPECT_NEAR(radial, 0.0, 1e-13) << "i = " << i;
    }
  }
}

TEST_P(SphericalHarmonicsTest, Orthonormality) {
  HarmonicCase s(GetParam());
  BoundaryHarmonicCoefficients bhc(*s.scalar, s.surface, s.L,
                                   BoundaryHarmonicCoefficients::Component::Scalar);
  EXPECT_NEAR(bhc.Radius(), 1.0, 1e-3);
  const int n = bhc.Size();
  double err = 0.0;
  for (int i = 0; i < n; i++) {
    auto f = bhc.Expansion(Unit(n, i));
    Vector c;
    bhc.Coefficients(*f, c);
    for (int j = 0; j < n; j++) {
      err = std::max(err, std::abs(c[j] - (i == j ? 1.0 : 0.0)));
    }
    // Through the interpolant on the order-2 space: interpolation error on
    // top, worst for the highest degree.
    GridFunction g(s.scalar.get());
    g.ProjectCoefficient(*f);
    bhc.Coefficients(g, c);
    EXPECT_NEAR(c[i], 1.0, InterpTol(s.dim)) << "harmonic " << i;
  }
  EXPECT_LT(err, GeomTol(s.dim));
}

TEST_P(SphericalHarmonicsTest, MatchesDtNOperator) {
  HarmonicCase s(GetParam());
  const int dim = s.dim, L = s.L;
  PoissonDtNOperator dtn(s.phi.get(), L);
  dtn.Assemble();
  BoundaryHarmonicCoefficients bhc(*s.phi, s.outer, L,
                                   BoundaryHarmonicCoefficients::Component::Scalar,
                                   dtn.Centroid());
  // The DtN radius comes from the vertices, ours from the quadrature points.
  EXPECT_NEAR(bhc.Radius(), dtn.BoundaryRadius(), 1e-4);

  FunctionCoefficient f([dim](const Vector& x) {
    return std::exp(0.5 * x[0]) * (1.0 + 0.3 * x[1]) +
           (dim == 3 ? 0.7 * x[2] * x[0] : 0.0);
  });
  GridFunction g(s.phi.get());
  g.ProjectCoefficient(f);
  Vector c_dtn, c;
  dtn.HarmonicCoefficients(g, c_dtn);
  bhc.Coefficients(g, c);
  double scale = 0.0;
  for (int i = 0; i < c.Size(); i++) {
    scale = std::max(scale, std::abs(c[i]));
  }
  ASSERT_GT(scale, 0.0);
  // The DtN operator weights each quadrature point by its own radius; the
  // difference is the radius spread of the curved faces, so the tolerance
  // depends on the mesh (the canned meshes sit well inside it).
  const double tol = (dim == 2 ? 5e-6 : 1e-4) * scale;
  // Same ordering and normalisation; the 2-D DtN operator has no
  // degree-zero term and returns that coefficient as zero.
  ASSERT_EQ(c_dtn.Size(), c.Size());
  for (int i = 0; i < c.Size(); i++) {
    if (dim == 2 && i == 0) {
      EXPECT_EQ(c_dtn[i], 0.0);
    } else {
      EXPECT_NEAR(c_dtn[i], c[i], tol) << "i = " << i;
    }
  }
}

TEST_P(SphericalHarmonicsTest, RadialComponent) {
  HarmonicCase s(GetParam());
  BoundaryHarmonicCoefficients bhc(*s.vector, s.surface, s.L,
                                   BoundaryHarmonicCoefficients::Component::Radial);
  const int n = bhc.Size(), dim = s.dim;
  const auto& basis = bhc.Basis();
  double err = 0.0, err_gf = 0.0;
  for (int i = 0; i < n; i++) {
    // u = Y_i(x^) x^ + a tangential part that must not contribute.
    VectorFunctionCoefficient u(dim, [&](const Vector& x, Vector& v) {
      Vector Y;
      basis.Eval(x, Y);
      const double r = x.Norml2();
      v = x;
      v *= Y[i] / r;
      // tangential: rotate x in the (0,1) plane.
      v[0] += -x[1] / r * 0.4;
      v[1] += x[0] / r * 0.4;
    });
    Vector c;
    bhc.Coefficients(u, c);
    for (int j = 0; j < n; j++) {
      err = std::max(err, std::abs(c[j] - (i == j ? 1.0 : 0.0)));
    }
    GridFunction g(s.vector.get());
    g.ProjectCoefficient(u);
    bhc.Coefficients(g, c);
    err_gf = std::max(err_gf, std::abs(c[i] - 1.0));
  }
  EXPECT_LT(err, GeomTol(dim));
  EXPECT_LT(err_gf, InterpTol(dim));
}

TEST_P(SphericalHarmonicsTest, TangentialComponent) {
  HarmonicCase s(GetParam());
  BoundaryHarmonicCoefficients bhc(
      *s.vector, s.surface, s.L,
      BoundaryHarmonicCoefficients::Component::Tangential);
  const int n = bhc.Size(), dim = s.dim;
  const auto& basis = bhc.Basis();
  double err = 0.0, err_gf = 0.0;
  for (int i = 0; i < n; i++) {
    // u = grad_1 Y_i + a radial part and a rotation, neither of which must
    // contribute.
    VectorFunctionCoefficient u(dim, [&](const Vector& x, Vector& v) {
      Vector Y;
      DenseMatrix gradY;
      basis.EvalWithGradient(x, Y, gradY);
      const double r = x.Norml2();
      gradY.GetColumn(i, v);
      v.Add(0.7 * Y[n - 1] / r, x);
      v[0] += -x[1] / r * 0.4;
      v[1] += x[0] / r * 0.4;
    });
    Vector c;
    bhc.Coefficients(u, c);
    const bool has_gradient = basis.Degree(i) > 0;
    for (int j = 0; j < n; j++) {
      err = std::max(err,
                     std::abs(c[j] - (i == j && has_gradient ? 1.0 : 0.0)));
    }
    GridFunction g(s.vector.get());
    g.ProjectCoefficient(u);
    bhc.Coefficients(g, c);
    err_gf = std::max(err_gf, std::abs(c[i] - (has_gradient ? 1.0 : 0.0)));
  }
  EXPECT_LT(err, GeomTol(dim));
  EXPECT_LT(err_gf, InterpTol(dim));
}

TEST_P(SphericalHarmonicsTest, LoadVectorMatchesLinearForm) {
  HarmonicCase s(GetParam());
  BoundaryHarmonicCoefficients bhc(*s.scalar, s.surface, s.L,
                                   BoundaryHarmonicCoefficients::Component::Scalar);
  const int n = bhc.Size();
  Vector c(n);
  for (int i = 0; i < n; i++) {
    c[i] = std::cos(1.3 * i + 0.2);
  }
  Vector b;
  bhc.LoadVector(c, b);
  auto f = bhc.Expansion(c);
  LinearForm lf(s.scalar.get());
  lf.AddBoundaryIntegrator(new BoundaryLFIntegrator(*f, 4, 4), s.surface);
  lf.Assemble();
  EXPECT_EQ(b.Size(), lf.Size());
  Vector d(b);
  d -= lf;
  EXPECT_GT(lf.Normlinf(), 0.0);
  EXPECT_LT(d.Normlinf(), 1e-7 * lf.Normlinf());  // different rules

  // Duality: g . b = R^{d-1} c . coefficients(g).
  GridFunction g(s.scalar.get());
  FunctionCoefficient gc([](const Vector& x) { return 1.0 + x[0] * x[1]; });
  g.ProjectCoefficient(gc);
  Vector cg;
  bhc.Coefficients(g, cg);
  EXPECT_NEAR(g * b, std::pow(bhc.Radius(), s.dim - 1) * (c * cg),
              1e-12 * std::abs(g * b));
}

TEST_P(SphericalHarmonicsTest, InteriorHarmonic) {
  HarmonicCase s(GetParam());
  const int L = s.L;
  SurfaceHarmonics basis(s.dim, L);
  Vector c(basis.Size());
  for (int i = 0; i < c.Size(); i++) {
    c[i] = 0.1 * (i + 1);
  }
  Vector centre(s.dim);
  centre = 0.0;
  HarmonicExpansionCoefficient surface(basis, c, centre, 1.0, false);
  HarmonicExpansionCoefficient interior(basis, c, centre, 1.0, true);
  FunctionCoefficient exact([&](const Vector& x) {
    Vector Y;
    basis.Eval(x, Y);
    const double r = x.Norml2();
    double f = 0.0;
    for (int i = 0; i < c.Size(); i++) {
      f += c[i] * Y[i] * std::pow(r, basis.Degree(i));
    }
    return f;
  });
  // At the quadrature points of the body's elements.
  double err = 0.0, err_surface = 0.0;
  for (int e = 0; e < s.body->GetNE(); e++) {
    auto* T = s.body->GetElementTransformation(e);
    const auto& ir = IntRules.Get(T->GetGeometryType(), 2);
    for (int q = 0; q < ir.GetNPoints(); q++) {
      const auto& ip = ir.IntPoint(q);
      T->SetIntPoint(&ip);
      err = std::max(err, std::abs(interior.Eval(*T, ip) - exact.Eval(*T, ip)));
      Vector x;
      T->Transform(ip, x);
      Vector Y;
      basis.Eval(x, Y);
      err_surface = std::max(err_surface, std::abs(surface.Eval(*T, ip) - (c * Y)));
    }
  }
  EXPECT_LT(err, 1e-13);
  EXPECT_LT(err_surface, 1e-13);
}

INSTANTIATE_TEST_SUITE_P(Harmonics, SphericalHarmonicsTest,
                         testing::Values(2, 3));

}  // namespace
