// Device kernels of ElasticTensorIntegrator's partial assembly on
// hexahedra: the form applied element by element by sum factorisation with
// the tensor kept at the quadrature points, and its diagonal. One loop over
// the elements (mfem::forall), so the same code runs on the host and on a
// device.
//
// This file is compiled by the device's compiler in a device build of mfem
// (src/CMakeLists.txt), whose dialect may be older than the library's: it
// includes mfem and AdGIA/detail/pa_kernels.hpp alone.
//
// Kept at a point (see pa_kernels.hpp): the 9 entries of the inverse
// Jacobian, d xi_a / d x_j at 3 a + j, then the material's own data, which
// carry the quadrature weight and the Jacobian's determinant. Strains and
// stresses at the point are in Voigt order (11, 22, 33, 23, 13, 12) with
// engineering shear strains.
#include "AdGIA/detail/pa_kernels.hpp"
#include "mfem.hpp"

namespace AdGIA::detail::pa {

using mfem::real_t;

namespace {

/// PackedIndex of pa_kernels.hpp, for the device.
MFEM_HOST_DEVICE inline int Packed(int i, int j) {
  return i <= j ? i * 6 - (i * (i - 1)) / 2 + (j - i)
                : j * 6 - (j * (j - 1)) / 2 + (i - j);
}

/// Voigt index of the component (i, j).
MFEM_HOST_DEVICE inline int Voigt(int i, int j) {
  return i == j ? i : 6 - i - j;
}

/// A full elastic tensor: the 21 entries of its Voigt matrix.
struct GeneralTensor {
  static constexpr int Size = kGeneralTensorSize;
  template <class Data>
  MFEM_HOST_DEVICE static inline real_t Entry(const Data& d, int q, int e,
                                              int i, int j) {
    return d(9 + Packed(i, j), q, e);
  }
  template <class Data>
  MFEM_HOST_DEVICE static inline void Stress(const Data& d, int q, int e,
                                             const real_t* eps, real_t* sig) {
    for (int i = 0; i < 6; i++) {
      real_t s = 0.0;
      for (int j = 0; j < 6; j++) {
        s += d(9 + Packed(i, j), q, e) * eps[j];
      }
      sig[i] = s;
    }
  }
};

/// An isotropic tensor: lambda and mu.
struct IsotropicTensor {
  static constexpr int Size = kIsotropicTensorSize;
  template <class Data>
  MFEM_HOST_DEVICE static inline real_t Entry(const Data& d, int q, int e,
                                              int i, int j) {
    const real_t l = d(9, q, e), m = d(10, q, e);
    if (i < 3 && j < 3) {
      return i == j ? l + 2.0 * m : l;
    }
    return i == j ? m : 0.0;
  }
  template <class Data>
  MFEM_HOST_DEVICE static inline void Stress(const Data& d, int q, int e,
                                             const real_t* eps, real_t* sig) {
    const real_t l = d(9, q, e), m = d(10, q, e);
    const real_t tr = l * (eps[0] + eps[1] + eps[2]);
    sig[0] = tr + 2.0 * m * eps[0];
    sig[1] = tr + 2.0 * m * eps[1];
    sig[2] = tr + 2.0 * m * eps[2];
    sig[3] = m * eps[3];
    sig[4] = m * eps[4];
    sig[5] = m * eps[5];
  }
};

/// y += A x on the element vectors (dofs lexicographic, then component,
/// then element). T_D and T_Q fix the 1-D sizes at compile time; zero
/// leaves them to d1d and q1d within mfem's limits.
template <class Material, int T_D = 0, int T_Q = 0>
void Apply3D(const int NE, const mfem::Array<real_t>& b_,
             const mfem::Array<real_t>& g_, const mfem::Vector& data_,
             const mfem::Vector& x_, mfem::Vector& y_, const int d1d = 0,
             const int q1d = 0) {
  const int D = T_D ? T_D : d1d;
  const int Q = T_Q ? T_Q : q1d;
  constexpr int MD = T_D ? T_D : mfem::DofQuadLimits::MAX_D1D;
  constexpr int MQ = T_Q ? T_Q : mfem::DofQuadLimits::MAX_Q1D;
  MFEM_VERIFY(D <= MD && Q <= MQ, "elastic tensor kernel: order too high.");
  const auto B = mfem::Reshape(b_.Read(), Q, D);
  const auto G = mfem::Reshape(g_.Read(), Q, D);
  const auto data = mfem::Reshape(data_.Read(), 9 + Material::Size, Q * Q * Q, NE);
  const auto X = mfem::Reshape(x_.Read(), D, D, D, 3, NE);
  auto Y = mfem::Reshape(y_.ReadWrite(), D, D, D, 3, NE);
  mfem::forall(NE, [=] MFEM_HOST_DEVICE(int e) {
    // The reference gradients of the three components at the points.
    real_t grad[MQ][MQ][MQ][9];
    for (int qz = 0; qz < Q; qz++) {
      for (int qy = 0; qy < Q; qy++) {
        for (int qx = 0; qx < Q; qx++) {
          for (int i = 0; i < 9; i++) {
            grad[qz][qy][qx][i] = 0.0;
          }
        }
      }
    }
    for (int c = 0; c < 3; c++) {
      for (int dz = 0; dz < D; dz++) {
        real_t gradXY[MQ][MQ][3];
        for (int qy = 0; qy < Q; qy++) {
          for (int qx = 0; qx < Q; qx++) {
            gradXY[qy][qx][0] = 0.0;
            gradXY[qy][qx][1] = 0.0;
            gradXY[qy][qx][2] = 0.0;
          }
        }
        for (int dy = 0; dy < D; dy++) {
          real_t gradX[MQ][2];
          for (int qx = 0; qx < Q; qx++) {
            gradX[qx][0] = 0.0;
            gradX[qx][1] = 0.0;
          }
          for (int dx = 0; dx < D; dx++) {
            const real_t s = X(dx, dy, dz, c, e);
            for (int qx = 0; qx < Q; qx++) {
              gradX[qx][0] += s * B(qx, dx);
              gradX[qx][1] += s * G(qx, dx);
            }
          }
          for (int qy = 0; qy < Q; qy++) {
            const real_t wy = B(qy, dy), wDy = G(qy, dy);
            for (int qx = 0; qx < Q; qx++) {
              gradXY[qy][qx][0] += gradX[qx][1] * wy;
              gradXY[qy][qx][1] += gradX[qx][0] * wDy;
              gradXY[qy][qx][2] += gradX[qx][0] * wy;
            }
          }
        }
        for (int qz = 0; qz < Q; qz++) {
          const real_t wz = B(qz, dz), wDz = G(qz, dz);
          for (int qy = 0; qy < Q; qy++) {
            for (int qx = 0; qx < Q; qx++) {
              grad[qz][qy][qx][3 * c + 0] += gradXY[qy][qx][0] * wz;
              grad[qz][qy][qx][3 * c + 1] += gradXY[qy][qx][1] * wz;
              grad[qz][qy][qx][3 * c + 2] += gradXY[qy][qx][2] * wDz;
            }
          }
        }
      }
    }
    // At the point: physical gradient, strain, stress, and back.
    for (int qz = 0; qz < Q; qz++) {
      for (int qy = 0; qy < Q; qy++) {
        for (int qx = 0; qx < Q; qx++) {
          const int q = qx + Q * (qy + Q * qz);
          real_t* H = grad[qz][qy][qx];
          real_t Gp[3][3];
          for (int c = 0; c < 3; c++) {
            for (int j = 0; j < 3; j++) {
              Gp[c][j] = H[3 * c] * data(j, q, e) +
                         H[3 * c + 1] * data(3 + j, q, e) +
                         H[3 * c + 2] * data(6 + j, q, e);
            }
          }
          const real_t eps[6] = {Gp[0][0],            Gp[1][1],
                                 Gp[2][2],            Gp[1][2] + Gp[2][1],
                                 Gp[0][2] + Gp[2][0], Gp[0][1] + Gp[1][0]};
          real_t sig[6];
          Material::Stress(data, q, e, eps, sig);
          const real_t S[3][3] = {{sig[0], sig[5], sig[4]},
                                  {sig[5], sig[1], sig[3]},
                                  {sig[4], sig[3], sig[2]}};
          for (int c = 0; c < 3; c++) {
            for (int a = 0; a < 3; a++) {
              H[3 * c + a] = S[c][0] * data(3 * a, q, e) +
                             S[c][1] * data(3 * a + 1, q, e) +
                             S[c][2] * data(3 * a + 2, q, e);
            }
          }
        }
      }
    }
    for (int c = 0; c < 3; c++) {
      for (int qz = 0; qz < Q; qz++) {
        real_t gradXY[MD][MD][3];
        for (int dy = 0; dy < D; dy++) {
          for (int dx = 0; dx < D; dx++) {
            gradXY[dy][dx][0] = 0.0;
            gradXY[dy][dx][1] = 0.0;
            gradXY[dy][dx][2] = 0.0;
          }
        }
        for (int qy = 0; qy < Q; qy++) {
          real_t gradX[MD][3];
          for (int dx = 0; dx < D; dx++) {
            gradX[dx][0] = 0.0;
            gradX[dx][1] = 0.0;
            gradX[dx][2] = 0.0;
          }
          for (int qx = 0; qx < Q; qx++) {
            const real_t gX = grad[qz][qy][qx][3 * c + 0];
            const real_t gY = grad[qz][qy][qx][3 * c + 1];
            const real_t gZ = grad[qz][qy][qx][3 * c + 2];
            for (int dx = 0; dx < D; dx++) {
              const real_t wx = B(qx, dx), wDx = G(qx, dx);
              gradX[dx][0] += gX * wDx;
              gradX[dx][1] += gY * wx;
              gradX[dx][2] += gZ * wx;
            }
          }
          for (int dy = 0; dy < D; dy++) {
            const real_t wy = B(qy, dy), wDy = G(qy, dy);
            for (int dx = 0; dx < D; dx++) {
              gradXY[dy][dx][0] += gradX[dx][0] * wy;
              gradXY[dy][dx][1] += gradX[dx][1] * wDy;
              gradXY[dy][dx][2] += gradX[dx][2] * wy;
            }
          }
        }
        for (int dz = 0; dz < D; dz++) {
          const real_t wz = B(qz, dz), wDz = G(qz, dz);
          for (int dy = 0; dy < D; dy++) {
            for (int dx = 0; dx < D; dx++) {
              Y(dx, dy, dz, c, e) +=
                  (gradXY[dy][dx][0] + gradXY[dy][dx][1]) * wz +
                  gradXY[dy][dx][2] * wDz;
            }
          }
        }
      }
    }
  });
}

/// diag += the diagonal of A, on the element vectors.
template <class Material, int T_D = 0, int T_Q = 0>
void Diagonal3D(const int NE, const mfem::Array<real_t>& b_,
                const mfem::Array<real_t>& g_, const mfem::Vector& data_,
                mfem::Vector& diag_, const int d1d = 0, const int q1d = 0) {
  const int D = T_D ? T_D : d1d;
  const int Q = T_Q ? T_Q : q1d;
  constexpr int MD = T_D ? T_D : mfem::DofQuadLimits::MAX_D1D;
  constexpr int MQ = T_Q ? T_Q : mfem::DofQuadLimits::MAX_Q1D;
  MFEM_VERIFY(D <= MD && Q <= MQ, "elastic tensor kernel: order too high.");
  const auto B = mfem::Reshape(b_.Read(), Q, D);
  const auto G = mfem::Reshape(g_.Read(), Q, D);
  const auto data = mfem::Reshape(data_.Read(), 9 + Material::Size, Q * Q * Q, NE);
  auto Y = mfem::Reshape(diag_.ReadWrite(), D, D, D, 3, NE);
  mfem::forall(NE, [=] MFEM_HOST_DEVICE(int e) {
    for (int c = 0; c < 3; c++) {
      for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
          // sum over the points of D_ab(q) F_x F_y F_z, with
          // D_ab = Jinv(a, j) C_{c j c l} Jinv(b, l) and F the products
          // of the 1-D values or derivatives the pair (a, b) calls for.
          real_t T2[MQ][MD][MD];
          for (int qz = 0; qz < Q; qz++) {
            real_t T1[MQ][MD];
            for (int qy = 0; qy < Q; qy++) {
              for (int dx = 0; dx < D; dx++) {
                T1[qy][dx] = 0.0;
              }
              for (int qx = 0; qx < Q; qx++) {
                const int q = qx + Q * (qy + Q * qz);
                real_t dab = 0.0;
                for (int j = 0; j < 3; j++) {
                  for (int l = 0; l < 3; l++) {
                    dab += data(3 * a + j, q, e) *
                           Material::Entry(data, q, e, Voigt(c, j), Voigt(c, l)) *
                           data(3 * b + l, q, e);
                  }
                }
                for (int dx = 0; dx < D; dx++) {
                  const real_t fa = a == 0 ? G(qx, dx) : B(qx, dx);
                  const real_t fb = b == 0 ? G(qx, dx) : B(qx, dx);
                  T1[qy][dx] += dab * fa * fb;
                }
              }
            }
            for (int dy = 0; dy < D; dy++) {
              for (int dx = 0; dx < D; dx++) {
                real_t s = 0.0;
                for (int qy = 0; qy < Q; qy++) {
                  const real_t fa = a == 1 ? G(qy, dy) : B(qy, dy);
                  const real_t fb = b == 1 ? G(qy, dy) : B(qy, dy);
                  s += T1[qy][dx] * fa * fb;
                }
                T2[qz][dy][dx] = s;
              }
            }
          }
          for (int dz = 0; dz < D; dz++) {
            for (int dy = 0; dy < D; dy++) {
              for (int dx = 0; dx < D; dx++) {
                real_t s = 0.0;
                for (int qz = 0; qz < Q; qz++) {
                  const real_t fa = a == 2 ? G(qz, dz) : B(qz, dz);
                  const real_t fb = b == 2 ? G(qz, dz) : B(qz, dz);
                  s += T2[qz][dy][dx] * fa * fb;
                }
                Y(dx, dy, dz, c, e) += s;
              }
            }
          }
        }
      }
    }
  });
}


template <class Material>
void Apply(int ne, const mfem::DofToQuad& maps, const mfem::Vector& data,
           const mfem::Vector& x, mfem::Vector& y) {
  const int D = maps.ndof, Q = maps.nqpt;
  switch (10 * D + Q) {
    case 23: return Apply3D<Material, 2, 3>(ne, maps.B, maps.G, data, x, y);
    case 34: return Apply3D<Material, 3, 4>(ne, maps.B, maps.G, data, x, y);
    case 45: return Apply3D<Material, 4, 5>(ne, maps.B, maps.G, data, x, y);
    case 56: return Apply3D<Material, 5, 6>(ne, maps.B, maps.G, data, x, y);
    case 67: return Apply3D<Material, 6, 7>(ne, maps.B, maps.G, data, x, y);
    default: return Apply3D<Material>(ne, maps.B, maps.G, data, x, y, D, Q);
  }
}

template <class Material>
void Diagonal(int ne, const mfem::DofToQuad& maps, const mfem::Vector& data,
              mfem::Vector& diag) {
  const int D = maps.ndof, Q = maps.nqpt;
  switch (10 * D + Q) {
    case 23: return Diagonal3D<Material, 2, 3>(ne, maps.B, maps.G, data, diag);
    case 34: return Diagonal3D<Material, 3, 4>(ne, maps.B, maps.G, data, diag);
    case 45: return Diagonal3D<Material, 4, 5>(ne, maps.B, maps.G, data, diag);
    case 56: return Diagonal3D<Material, 5, 6>(ne, maps.B, maps.G, data, diag);
    case 67: return Diagonal3D<Material, 6, 7>(ne, maps.B, maps.G, data, diag);
    default: return Diagonal3D<Material>(ne, maps.B, maps.G, data, diag, D, Q);
  }
}

}  // namespace

void ApplyElasticTensor(bool isotropic, int ne, const mfem::DofToQuad& maps,
                        const mfem::Vector& data, const mfem::Vector& x,
                        mfem::Vector& y) {
  if (isotropic) {
    Apply<IsotropicTensor>(ne, maps, data, x, y);
  } else {
    Apply<GeneralTensor>(ne, maps, data, x, y);
  }
}

void DiagonalElasticTensor(bool isotropic, int ne, const mfem::DofToQuad& maps,
                           const mfem::Vector& data, mfem::Vector& diag) {
  if (isotropic) {
    Diagonal<IsotropicTensor>(ne, maps, data, diag);
  } else {
    Diagonal<GeneralTensor>(ne, maps, data, diag);
  }
}

}  // namespace AdGIA::detail::pa
