#pragma once

/*
  Shared pieces of the finite-strain tests: a law whose tangent is not
  constant, the material of the test body, and the body itself.
*/

#include <cmath>
#include <memory>
#include <numbers>

#include "AdGIA.hpp"
#include "mfem.hpp"

namespace nonlinear_test {

using namespace mfem;
using namespace AdGIA;

// A law with a cubic volumetric term added to another,
// W = W_base + nu (tr E)^3 / 6: the stress gains nu (tr E)^2 / 2 times
// the identity and the tangent nu tr E times 1 1^T, so the tangent
// depends on the strain.
class CubicVolumetricLaw : public MaterialLaw {
 public:
  CubicVolumetricLaw(MaterialLaw& base, real_t nu)
      : MaterialLaw(base.SpaceDim()), base_(&base), nu_(nu) {}

  real_t Evaluate(ElementTransformation& T, const IntegrationPoint& ip,
                  const Vector& e, Vector* S, DenseMatrix* tangent) override {
    const auto dim = SpaceDim();
    auto energy = base_->Evaluate(T, ip, e, S, tangent);
    auto trace = real_t{0};
    for (auto j = 0; j < dim; j++) {
      trace += e[SymmetricTensorBasis::Index(dim, j, j)];
    }
    energy += nu_ * trace * trace * trace / 6.0;
    for (auto j = 0; j < dim; j++) {
      const auto a = SymmetricTensorBasis::Index(dim, j, j);
      if (S) {
        (*S)[a] += 0.5 * nu_ * trace * trace;
      }
      if (tangent) {
        for (auto k = 0; k < dim; k++) {
          (*tangent)(a, SymmetricTensorBasis::Index(dim, k, k)) += nu_ * trace;
        }
      }
    }
    return energy;
  }

 private:
  MaterialLaw* base_;
  real_t nu_;
};

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
// as it is for an Earth model. The quadratic law, and that law with a
// cubic volumetric term.
struct Material {
  FunctionCoefficient lambda{[](const Vector& x) { return 1.0 + 0.3 * x[0]; }};
  FunctionCoefficient mu{[](const Vector& x) { return 0.8 + 0.2 * x[1]; }};
  FunctionCoefficient p0{Pressure};
  IsotropicElasticTensorCoefficient C;
  BareElasticTensorCoefficient Xi;
  MatrixFunctionCoefficient T0;
  QuadraticMaterialLaw quadratic;
  CubicVolumetricLaw cubic;

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
        cubic(quadratic, 3.0) {}
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
