// Tests for the mapping layer (mappings.hpp).
//
//  1. The analytic mappings return exact deformation gradients: the radial
//     map through both constructors, and the derived coefficients (J, the
//     pulled-back diffusion matrix J F^{-1} F^{-T}, the pulled-back
//     gradient F^{-T} v) against closed forms.
//  2. GridFunctionDiffeomorphism reproduces a polynomial map lying in its
//     space to round-off.
//  3. The discrete change-of-variables identity (doc/mappings.md, "The
//     discrete change-of-variables identity") on which the mapped
//     integrator tests rest: for a smooth non-polynomial mapping, the
//     element transformations of MappedMesh(mesh, xi) factor through the
//     reference ones with the gradient of Interpolate(xi, mesh) —
//     Jacobians and weights agree at every quadrature point to round-off,
//     at every geometric order.
//  4. MaxIdentityDeviation: zero for the identity and for mappings that are
//     the identity on the boundary, |c| for a translation by c.
#include <numbers>

#include "TestCommon.hpp"

namespace {

constexpr real_t pi = std::numbers::pi_v<real_t>;

Mesh SmallMesh(int dim, int elementType = 0) {
  if (dim == 2) {
    return Mesh::MakeCartesian2D(
        6, 6, elementType == 0 ? Element::TRIANGLE : Element::QUADRILATERAL);
  }
  return Mesh::MakeCartesian3D(
      4, 4, 4, elementType == 0 ? Element::TETRAHEDRON : Element::HEXAHEDRON);
}

real_t MaxEntryDiff(const DenseMatrix& A, const DenseMatrix& B) {
  real_t d = 0.0;
  for (int i = 0; i < A.Height(); i++) {
    for (int j = 0; j < A.Width(); j++) {
      d = std::max(d, std::abs(A(i, j) - B(i, j)));
    }
  }
  return d;
}

// A smooth, non-polynomial diffeomorphism of the unit square/cube with
// exact gradient: xi_i = x_i + c sin(pi x_j), j = (i + 1) mod dim.
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

// 1. Exact gradients of the radial map, both constructors, plus the
//    derived coefficients.
TEST(Mappings, RadialGradientsAreExact) {
  for (int dim = 2; dim <= 3; dim++) {
    auto mesh = SmallMesh(dim);

    // Radial profile f(r) = 1 + a r^2, so grad f = 2 a x and
    // F = f I + 2 a x x^T.
    const real_t a = 0.3;
    RadialDiffeomorphism profile(
        dim, [a](real_t r) { return 1.0 + a * r * r; },
        [a](real_t r) { return 2.0 * a * r; });

    // The same map through the coefficient constructor.
    FunctionCoefficient f_coeff(
        [a](const Vector& x) { return 1.0 + a * (x * x); });
    VectorFunctionCoefficient g_coeff(dim, [a](const Vector& x, Vector& g) {
      g = x;
      g *= 2.0 * a;
    });
    RadialDiffeomorphism coeffs(dim, f_coeff, g_coeff);

    JacobianCoefficient J_coeff(profile);
    PullbackDiffusionCoefficient a_coeff(profile);
    VectorConstantCoefficient v_coeff([dim] {
      Vector v(dim);
      v = 0.7;
      v(0) = -0.2;
      return v;
    }());
    PullbackGradientCoefficient pg_coeff(profile, v_coeff);

    Vector x(dim), v(dim), w(dim), expect_w(dim);
    DenseMatrix F(dim), expectF(dim), G(dim), K(dim), expectK(dim);
    for (int e = 0; e < mesh.GetNE(); e++) {
      ElementTransformation* T = mesh.GetElementTransformation(e);
      const IntegrationRule& ir = IntRules.Get(T->GetGeometryType(), 4);
      for (int q = 0; q < ir.GetNPoints(); q++) {
        const IntegrationPoint& ip = ir.IntPoint(q);
        T->Transform(ip, x);

        // Closed form F = (1 + a x.x) I + 2 a x x^T.
        for (int i = 0; i < dim; i++) {
          for (int A = 0; A < dim; A++) {
            expectF(i, A) = 2.0 * a * x(i) * x(A);
          }
          expectF(i, i) += 1.0 + a * (x * x);
        }

        profile.EvalGradient(F, *T, ip);
        EXPECT_LT(MaxEntryDiff(F, expectF), 1e-13);
        coeffs.EvalGradient(G, *T, ip);
        EXPECT_LT(MaxEntryDiff(G, expectF), 1e-13);

        // J = det F.
        EXPECT_NEAR(profile.Jacobian(*T, ip), expectF.Det(), 1e-13);
        EXPECT_NEAR(J_coeff.Eval(*T, ip), expectF.Det(), 1e-13);

        // a = J F^{-1} F^{-T}.
        const real_t J = expectF.Det();
        DenseMatrix Finv(expectF);
        Finv.Invert();
        MultABt(Finv, Finv, expectK);
        expectK *= J;
        a_coeff.Eval(K, *T, ip);
        EXPECT_LT(MaxEntryDiff(K, expectK), 1e-12);

        // F^{-T} v.
        v_coeff.Eval(v, *T, ip);
        Finv.MultTranspose(v, expect_w);
        pg_coeff.Eval(w, *T, ip);
        w -= expect_w;
        EXPECT_LT(w.Norml2(), 1e-13);
      }
    }
  }
}

// 2. A polynomial map lying in the displacement space is reproduced to
//    round-off: h_i = (b . x) x_i, quadratic, in an order-2 space.
TEST(Mappings, GridFunctionMapReproducesPolynomial) {
  for (int dim = 2; dim <= 3; dim++) {
    auto mesh = SmallMesh(dim);
    H1_FECollection fec(2, dim);
    FiniteElementSpace fes(&mesh, &fec, dim);

    Vector b(dim);
    b = 0.25;
    b(0) = -0.1;

    VectorFunctionCoefficient h_coeff(dim, [b](const Vector& x, Vector& h) {
      h = x;
      h *= (b * x);
    });
    GridFunction h(&fes);
    h.ProjectCoefficient(h_coeff);
    GridFunctionDiffeomorphism xi(h);

    Vector x(dim), y(dim), expect_y(dim);
    DenseMatrix F(dim), expectF(dim);
    for (int e = 0; e < mesh.GetNE(); e++) {
      ElementTransformation* T = mesh.GetElementTransformation(e);
      const IntegrationRule& ir = IntRules.Get(T->GetGeometryType(), 4);
      for (int q = 0; q < ir.GetNPoints(); q++) {
        const IntegrationPoint& ip = ir.IntPoint(q);
        T->Transform(ip, x);

        // xi = x + (b.x) x, grad h = x b^T + (b.x) I.
        expect_y = x;
        expect_y *= 1.0 + (b * x);
        xi.Eval(y, *T, ip);
        y -= expect_y;
        EXPECT_LT(y.Norml2(), 1e-12);

        for (int i = 0; i < dim; i++) {
          for (int A = 0; A < dim; A++) {
            expectF(i, A) = x(i) * b(A);
          }
          expectF(i, i) += 1.0 + (b * x);
        }
        xi.EvalGradient(F, *T, ip);
        EXPECT_LT(MaxEntryDiff(F, expectF), 1e-12);
      }
    }
  }
}

// 3. The discrete change-of-variables identity: the mapped mesh's element
//    transformations factor exactly through the interpolated mapping.
TEST(Mappings, InterpolateMatchesMappedMesh) {
  for (int dim = 2; dim <= 3; dim++) {
    for (int order = 1; order <= 3; order++) {
      auto mesh = SmallMesh(dim);
      mesh.SetCurvature(order);

      auto xi = SmoothMap(dim, 0.05);
      auto xi_h = Interpolate(xi, mesh);
      auto mapped = MappedMesh(mesh, xi);

      DenseMatrix F(dim), expected(dim);
      for (int e = 0; e < mesh.GetNE(); e++) {
        ElementTransformation* Tref = mesh.GetElementTransformation(e);
        ElementTransformation* Tmap = mapped.GetElementTransformation(e);
        const IntegrationRule& ir =
            IntRules.Get(Tref->GetGeometryType(), 2 * order + 2);
        for (int q = 0; q < ir.GetNPoints(); q++) {
          const IntegrationPoint& ip = ir.IntPoint(q);
          Tref->SetIntPoint(&ip);
          Tmap->SetIntPoint(&ip);

          // Jacobian factorisation dxi_h/dr = F_h dx/dr ...
          xi_h.EvalGradient(F, *Tref, ip);
          Mult(F, Tref->Jacobian(), expected);
          EXPECT_LT(MaxEntryDiff(Tmap->Jacobian(), expected), 1e-12);

          // ... and the induced weight identity.
          EXPECT_NEAR(Tmap->Weight(), F.Det() * Tref->Weight(), 1e-12);
        }
      }
    }
  }
}

// 4. MaxIdentityDeviation on the unit square: zero for the identity and
//    for boundary-fixing maps, |c| for a translation.
TEST(Mappings, MaxIdentityDeviation) {
  const int dim = 2;
  auto mesh = SmallMesh(dim);
  auto bdr_marker = AllBoundariesMarker(&mesh);

  CallableDiffeomorphism identity(
      dim, [](const Vector& x, Vector& y) { y = x; },
      [](const Vector&, DenseMatrix& F) {
        F = 0.0;
        F(0, 0) = F(1, 1) = 1.0;
      });
  EXPECT_LT(MaxIdentityDeviation(identity, mesh, bdr_marker), 1e-15);

  Vector c(dim);
  c(0) = 0.3;
  c(1) = -0.1;
  CallableDiffeomorphism translation(
      dim,
      [c](const Vector& x, Vector& y) {
        y = x;
        y += c;
      },
      [](const Vector&, DenseMatrix& F) {
        F = 0.0;
        F(0, 0) = F(1, 1) = 1.0;
      });
  EXPECT_NEAR(MaxIdentityDeviation(translation, mesh, bdr_marker), c.Norml2(),
              1e-14);

  // A bump vanishing on the boundary of [0,1]^2.
  CallableDiffeomorphism bump(
      dim,
      [](const Vector& x, Vector& y) {
        y = x;
        const real_t s = 0.1 * std::sin(pi * x(0)) * std::sin(pi * x(1));
        y(0) += s;
        y(1) += s;
      },
      [](const Vector& x, DenseMatrix& F) {
        F(0, 0) = 1.0 + 0.1 * pi * std::cos(pi * x(0)) * std::sin(pi * x(1));
        F(0, 1) = 0.1 * pi * std::sin(pi * x(0)) * std::cos(pi * x(1));
        F(1, 0) = 0.1 * pi * std::cos(pi * x(0)) * std::sin(pi * x(1));
        F(1, 1) = 1.0 + 0.1 * pi * std::sin(pi * x(0)) * std::cos(pi * x(1));
      });
  EXPECT_LT(MaxIdentityDeviation(bump, mesh, bdr_marker), 1e-14);
}
