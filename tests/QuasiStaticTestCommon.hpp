#pragma once

/*
  Helpers shared by the quasi-static-problem tests (serial gtests and MPI
  programs): small Cartesian meshes, boundary-attribute lookup, the loads of
  the reference problems and the exact uniaxial-stress strain.
*/

#include <cmath>

#include "mfem.hpp"
#include "AdGIA.hpp"

namespace elastic_test {

using namespace mfem;

inline Mesh MakeSmallMesh(int dim, int elementType) {
  return dim == 2
             ? Mesh::MakeCartesian2D(6, 6,
                                     elementType == 0 ? Element::TRIANGLE
                                                      : Element::QUADRILATERAL)
             : Mesh::MakeCartesian3D(3, 3, 3,
                                     elementType == 0 ? Element::TETRAHEDRON
                                                      : Element::HEXAHEDRON);
}

// Boundary attribute of the boundary elements with x[coord] == value.
inline int BdrAttributeAt(Mesh& mesh, int coord, double value) {
  const auto dim = mesh.Dimension();
  for (auto i = 0; i < mesh.GetNBE(); i++) {
    auto* tr = mesh.GetBdrElementTransformation(i);
    Vector c(dim);
    tr->Transform(Geometries.GetCenter(mesh.GetBdrElementGeometry(i)), c);
    if (std::abs(c[coord] - value) < 1e-12) {
      return mesh.GetBdrAttribute(i);
    }
  }
  return -1;
}

inline Array<int> Marker(int size, std::initializer_list<int> attrs) {
  Array<int> m(size);
  m = 0;
  for (auto a : attrs) {
    m[a - 1] = 1;
  }
  return m;
}

// Material of the tests: lambda = 1.3, mu = 0.8, kappa = lambda + 2 mu / d.
constexpr double kLambda = 1.3;
constexpr double kMu = 0.8;
inline double Kappa(int dim) { return kLambda + 2.0 * kMu / dim; }

// Traction of the uniaxial-stress test: +sigma e_x on x = 1, -sigma e_x on
// x = 0, scaled by (1 + t).
constexpr double kSigma = 0.3;
inline void UniaxialTraction(const Vector& x, real_t t, Vector& f) {
  f = 0.0;
  if (x[0] > 1.0 - 1e-8) {
    f[0] = kSigma * (1.0 + t);
  } else if (x[0] < 1e-8) {
    f[0] = -kSigma * (1.0 + t);
  }
}

// Exact (small-strain) uniaxial-stress strain for the library's continuum
// convention in each dimension.
inline void UniaxialStrain(int dim, double sigma, double& exx, double& eyy) {
  const auto l = kLambda, m = kMu;
  if (dim == 2) {
    exx = sigma * (l + 2.0 * m) / (4.0 * m * (l + m));
    eyy = -l / (l + 2.0 * m) * exx;
  } else {
    const auto E = m * (3.0 * l + 2.0 * m) / (l + m);
    const auto nu = l / (2.0 * (l + m));
    exx = sigma / E;
    eyy = -nu * exx;
  }
}

// Max over local elements of |sym(grad u)(centre) - diag(exx, eyy, eyy)|.
inline double MaxStrainError(const GridFunction& u, double exx, double eyy) {
  auto* fes = u.FESpace();
  const auto dim = fes->GetMesh()->Dimension();
  double err = 0.0;
  DenseMatrix grad(dim);
  for (auto e = 0; e < fes->GetNE(); e++) {
    auto* T = fes->GetElementTransformation(e);
    T->SetIntPoint(&Geometries.GetCenter(T->GetGeometryType()));
    u.GetVectorGradient(*T, grad);
    for (auto i = 0; i < dim; i++) {
      for (auto j = 0; j < dim; j++) {
        const auto eij = 0.5 * (grad(i, j) + grad(j, i));
        const auto exact = i != j ? 0.0 : (i == 0 ? exx : eyy);
        err = std::max(err, std::abs(eij - exact));
      }
    }
  }
  return err;
}

// Clamped-beam traction: a downward pull scaled by (1 + t).
inline void PullTraction(const Vector& x, real_t t, Vector& f) {
  f = 0.0;
  f[f.Size() - 1] = -0.05 * (1.0 + t);
}

}  // namespace elastic_test
