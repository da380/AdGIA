#pragma once

/*
  Shared pieces of the finite-strain tests: the material of the test
  body, with a law whose tangent is not constant, the body itself, a
  smooth configuration with its gradient and an interior relabelling.
*/

#include <cmath>
#include <memory>
#include <numbers>

#include "AdGIA.hpp"
#include "mfem.hpp"

namespace nonlinear_test {

using namespace mfem;
using namespace AdGIA;

// The equilibrium pressure of the test body: positive inside and zero on
// the sides and the top, which are free or loaded, so that the stress
// does not push on them.
inline real_t Pressure(const Vector& x) {
  const auto dim = x.Size();
  auto p = 0.15 * (1.0 - x[dim - 1]);
  for (auto j = 0; j < dim - 1; j++) {
    p *= 4.0 * x[j] * (1.0 - x[j]);
  }
  return p;
}

// The material of the test body: an isotropic tensor that varies over
// the body and a hydrostatic equilibrium stress -p0 1 that varies too;
// the second elastic tensor of the law is the bare tensor of the two,
// as it is for an Earth model. The quadratic law, and that law with the
// isotropic terms of third order, whose tangent depends on the strain.
struct Material {
  FunctionCoefficient lambda{[](const Vector& x) { return 1.0 + 0.3 * x[0]; }};
  FunctionCoefficient mu{[](const Vector& x) { return 0.8 + 0.2 * x[1]; }};
  FunctionCoefficient p0{Pressure};
  IsotropicElasticTensorCoefficient C;
  BareElasticTensorCoefficient Xi;
  MatrixFunctionCoefficient T0;
  QuadraticMaterialLaw quadratic;
  ConstantCoefficient nu1{3.0}, nu2{0.8}, nu3{-0.5};
  ThirdOrderIsotropicLaw cubic;

  explicit Material(int dim)
      : C(dim, lambda, mu),
        Xi(dim, C, p0),
        T0(dim,
           [dim](const Vector& x, DenseMatrix& S) {
             S.SetSize(dim);
             S = 0.0;
             for (auto j = 0; j < dim; j++) {
               S(j, j) = -Pressure(x);
             }
           }),
        quadratic(dim, Xi, &T0),
        cubic(quadratic, nu1, nu2, nu3) {}
};

// A smooth displacement of amplitude `a`, not zero anywhere in
// particular.
inline void SmoothField(const Vector& x, Vector& u, real_t a, real_t phase) {
  const auto dim = x.Size();
  u.SetSize(dim);
  for (auto i = 0; i < dim; i++) {
    auto value = std::sin(phase + 1.3 * (i + 1));
    for (auto j = 0; j < dim; j++) {
      value += std::sin(std::numbers::pi * (0.7 + 0.3 * i) * x[j] + phase +
                        0.4 * j);
    }
    u[i] = a * value;
  }
}

// The gradient of SmoothField, G_ij = d u_i / d x_j.
inline void SmoothGradient(const Vector& x, DenseMatrix& G, real_t a,
                           real_t phase) {
  const auto dim = x.Size();
  G.SetSize(dim);
  for (auto i = 0; i < dim; i++) {
    const auto k = std::numbers::pi * (0.7 + 0.3 * i);
    for (auto j = 0; j < dim; j++) {
      G(i, j) = a * k * std::cos(k * x[j] + phase + 0.4 * j);
    }
  }
}

// The configuration x + u(x) of SmoothField, with its exact gradient.
inline CallableDiffeomorphism SmoothConfiguration(int dim, real_t a,
                                                  real_t phase) {
  return CallableDiffeomorphism(
      dim,
      [a, phase](const Vector& x, Vector& y) {
        SmoothField(x, y, a, phase);
        y += x;
      },
      [a, phase](const Vector& x, DenseMatrix& F) {
        SmoothGradient(x, F, a, phase);
        for (auto i = 0; i < F.Height(); i++) {
          F(i, i) += 1.0;
        }
      });
}

// A relabelling of the unit square or cube that moves its interior and
// leaves its boundary where it is, to first order as well:
// x -> x + c b(x) d with b the product of sin^2(pi x_j) and d a fixed
// direction.
inline CallableDiffeomorphism InteriorRelabelling(int dim, real_t c) {
  auto bump = [](const Vector& x, Vector* gradient) {
    const auto dim = x.Size();
    auto b = real_t{1};
    for (auto j = 0; j < dim; j++) {
      const auto s = std::sin(std::numbers::pi * x[j]);
      b *= s * s;
    }
    if (gradient) {
      gradient->SetSize(dim);
      for (auto j = 0; j < dim; j++) {
        auto g = std::numbers::pi * std::sin(2.0 * std::numbers::pi * x[j]);
        for (auto k = 0; k < dim; k++) {
          if (k != j) {
            const auto s = std::sin(std::numbers::pi * x[k]);
            g *= s * s;
          }
        }
        (*gradient)[j] = g;
      }
    }
    return b;
  };
  auto direction = [](int dim, int i) { return i == 0 ? 1.0 : 0.6 / dim; };
  return CallableDiffeomorphism(
      dim,
      [c, bump, direction](const Vector& x, Vector& y) {
        const auto b = bump(x, nullptr);
        y = x;
        for (auto i = 0; i < x.Size(); i++) {
          y[i] += c * b * direction(x.Size(), i);
        }
      },
      [c, bump, direction](const Vector& x, DenseMatrix& F) {
        auto g = Vector();
        bump(x, &g);
        const auto dim = x.Size();
        F.SetSize(dim);
        F = 0.0;
        for (auto i = 0; i < dim; i++) {
          F(i, i) = 1.0;
          for (auto j = 0; j < dim; j++) {
            F(i, j) += c * direction(dim, i) * g[j];
          }
        }
      });
}

// The material of the test body as a relabelled body has it: the fields
// of Material at the image of each point of the mesh.
struct RelabelledMaterial {
  TransformedFunctionCoefficient lambda, mu, p0;
  IsotropicElasticTensorCoefficient C;
  BareElasticTensorCoefficient Xi;
  TransformedMatrixFunctionCoefficient T0;
  QuadraticMaterialLaw quadratic;
  ConstantCoefficient nu1{3.0}, nu2{0.8}, nu3{-0.5};
  ThirdOrderIsotropicLaw cubic;

  RelabelledMaterial(int dim, Diffeomorphism& map)
      : lambda(map, [](const Vector& x) { return 1.0 + 0.3 * x[0]; }),
        mu(map, [](const Vector& x) { return 0.8 + 0.2 * x[1]; }),
        p0(map, Pressure),
        C(dim, lambda, mu),
        Xi(dim, C, p0),
        T0(dim, map,
           [dim](const Vector& x, DenseMatrix& S) {
             S.SetSize(dim);
             S = 0.0;
             for (auto j = 0; j < dim; j++) {
               S(j, j) = -Pressure(x);
             }
           }),
        quadratic(dim, Xi, &T0),
        cubic(quadratic, nu1, nu2, nu3) {}
};

// The unit square or cube with n elements a side; elementType 0 gives
// simplices, 1 tensor-product elements. Boundary attribute 1 is the
// bottom (y = 0 in 2-D, z = 0 in 3-D) and the top is attribute 3 in 2-D
// and 6 in 3-D.
inline Mesh MakeBody(int dim, int elementType, int n) {
  return dim == 2
             ? Mesh::MakeCartesian2D(n, n,
                                     elementType == 0 ? Element::TRIANGLE
                                                      : Element::QUADRILATERAL)
             : Mesh::MakeCartesian3D(n, n, n,
                                     elementType == 0 ? Element::TETRAHEDRON
                                                      : Element::HEXAHEDRON);
}

inline Array<int> Marker(const Mesh& mesh, int attribute) {
  auto marker = Array<int>(mesh.bdr_attributes.Max());
  marker = 0;
  if (attribute > 0) {
    marker[attribute - 1] = 1;
  }
  return marker;
}

inline int TopAttribute(int dim) { return dim == 2 ? 3 : 6; }

// A traction on the top: a shear and a pull, of size `s`.
inline Vector TopTraction(int dim, real_t s) {
  auto t = Vector(dim);
  t = 0.0;
  t[0] = 0.6 * s;
  t[dim - 1] = s;
  return t;
}

}  // namespace nonlinear_test
