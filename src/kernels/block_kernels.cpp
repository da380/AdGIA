// Device kernels of the blocks' partial assembly: the displacement block
// (the elastic tensor form, alone or with a mass term or with the gravity
// terms and a mass term) and the potential block (a diffusion form and a
// mass term), applied element by element from data at the quadrature
// points, and their diagonals. Sum-factorised on hexahedra, dense in the
// element otherwise. One loop over the elements (mfem::forall), so the
// same code runs on the host and on a device.
//
// This file is compiled by the device's compiler in a device build of mfem
// (src/CMakeLists.txt), whose dialect may be older than the library's: it
// includes mfem and AdGIA/detail/pa_kernels.hpp alone, which says what the
// data at a point are. Strains and stresses at the point are in Voigt
// order (11, 22, 33, 23, 13, 12) with engineering shear strains.
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

/// A form, to the kernels below: its components (NC), the data it keeps
/// at a point (Size), whether it takes values as well as gradients, the
/// work at a point, and the coefficients of its diagonal.
///
/// The displacement block. In Point, on entry H is the reference
/// gradient of the displacement, H[3 c + a] = d u_c / d xi_a, and u its
/// value (read only with extras). On return H is what is tested with the
/// reference gradient of v, and f what is tested with v (written only
/// with extras).
///
/// With extras NX = 13, g and G as in pa_kernels.hpp and du_ki = d_i u_k:
///   tested with v_i:     g_k du_ki + (G_ik + G_ki) u_k - g_i div u + s u_i
///   tested with d_i v_k: (C : eps(u))_ki + g_k u_i - delta_ki (g . u)
template <class Material, int NX>
struct DisplacementForm {
  static constexpr int NC = 3;
  static constexpr int Size = 9 + Material::Size + NX;
  static constexpr bool Values = NX > 0;

  template <class Data>
  MFEM_HOST_DEVICE static inline void Point(const Data& d, int q, int e, const real_t* u,
                                   real_t* H, real_t* f) {
  real_t Gu[3][3];
  for (int c = 0; c < 3; c++) {
    for (int j = 0; j < 3; j++) {
      Gu[c][j] = H[3 * c] * d(j, q, e) + H[3 * c + 1] * d(3 + j, q, e) +
                 H[3 * c + 2] * d(6 + j, q, e);
    }
  }
  const real_t eps[6] = {Gu[0][0],            Gu[1][1],
                         Gu[2][2],            Gu[1][2] + Gu[2][1],
                         Gu[0][2] + Gu[2][0], Gu[0][1] + Gu[1][0]};
  real_t sig[6];
  Material::Stress(d, q, e, eps, sig);
  real_t P[3][3] = {{sig[0], sig[5], sig[4]},
                    {sig[5], sig[1], sig[3]},
                    {sig[4], sig[3], sig[2]}};
  const int o = 9 + Material::Size;
  if (NX == kMassExtras) {
    const real_t s = d(o, q, e);
    for (int a = 0; a < 3; a++) {
      f[a] = s * u[a];
    }
  } else if (NX == kGravityExtras) {
    const real_t g[3] = {d(o, q, e), d(o + 1, q, e), d(o + 2, q, e)};
    const real_t s = d(o + 12, q, e);
    const real_t divu = Gu[0][0] + Gu[1][1] + Gu[2][2];
    const real_t gdotu = g[0] * u[0] + g[1] * u[1] + g[2] * u[2];
    for (int a = 0; a < 3; a++) {
      real_t t = g[0] * Gu[0][a] + g[1] * Gu[1][a] + g[2] * Gu[2][a] -
                 g[a] * divu + s * u[a];
      for (int k = 0; k < 3; k++) {
        t += (d(o + 3 + 3 * a + k, q, e) + d(o + 3 + 3 * k + a, q, e)) * u[k];
        P[k][a] += g[k] * u[a];
      }
      f[a] = t;
      P[a][a] -= gdotu;
    }
  }
  for (int c = 0; c < 3; c++) {
    for (int a = 0; a < 3; a++) {
      H[3 * c + a] = P[c][0] * d(3 * a, q, e) + P[c][1] * d(3 * a + 1, q, e) +
                     P[c][2] * d(3 * a + 2, q, e);
    }
  }
}

/// The coefficient of u_c v_c at a point: the diagonal of what is tested
/// with v against u's value. (The terms that pair a value with a gradient
/// cancel on the diagonal.)
  template <class Data>
  MFEM_HOST_DEVICE static inline real_t ValueDiagonal(const Data& d, int q, int e,
                                             int c) {
  const int o = 9 + Material::Size;
  if (NX == kMassExtras) {
    return d(o, q, e);
  }
  if (NX == kGravityExtras) {
    return 2.0 * d(o + 3 + 4 * c, q, e) + d(o + 12, q, e);
  }
  return 0.0;
}

/// The coefficient of (d u_c / d xi_b) (d v_c / d xi_a) at a point.
  template <class Data>
  MFEM_HOST_DEVICE static inline real_t GradientDiagonal(const Data& d, int q, int e,
                                                int c, int a, int b) {
  real_t t = 0.0;
  for (int j = 0; j < 3; j++) {
    for (int l = 0; l < 3; l++) {
      t += d(3 * a + j, q, e) *
           Material::Entry(d, q, e, Voigt(c, j), Voigt(c, l)) *
           d(3 * b + l, q, e);
    }
  }
  return t;
}
};

/// The potential block: (phi, psi) -> int k grad phi . grad psi + s phi
/// psi. At a point: the symmetric matrix k w Jinv Jinv^T in the reference
/// directions (entries 00, 01, 02, 11, 12, 22), then s w.
struct PotentialForm {
  static constexpr int NC = 1;
  static constexpr int Size = kPotentialDataSize;
  static constexpr bool Values = true;

  MFEM_HOST_DEVICE static inline int Sym(int a, int b) {
    const int lo = a < b ? a : b, hi = a < b ? b : a;
    return lo == 0 ? hi : lo + hi + 1;
  }
  template <class Data>
  MFEM_HOST_DEVICE static inline void Point(const Data& d, int q, int e,
                                            const real_t* u, real_t* H,
                                            real_t* f) {
    const real_t h0 = H[0], h1 = H[1], h2 = H[2];
    H[0] = d(0, q, e) * h0 + d(1, q, e) * h1 + d(2, q, e) * h2;
    H[1] = d(1, q, e) * h0 + d(3, q, e) * h1 + d(4, q, e) * h2;
    H[2] = d(2, q, e) * h0 + d(4, q, e) * h1 + d(5, q, e) * h2;
    f[0] = d(6, q, e) * u[0];
  }
  template <class Data>
  MFEM_HOST_DEVICE static inline real_t ValueDiagonal(const Data& d, int q,
                                                      int e, int) {
    return d(6, q, e);
  }
  template <class Data>
  MFEM_HOST_DEVICE static inline real_t GradientDiagonal(const Data& d, int q,
                                                         int e, int, int a,
                                                         int b) {
    return d(Sym(a, b), q, e);
  }
};

// ---------------------------------------------------------------------------
// Dense in the element: any element, its basis in mfem's FULL mode.

/// The most dofs and quadrature points of an element the dense kernels
/// take (a tetrahedron of order 9 has 220 dofs): a block's values are
/// kept on the stack.
constexpr int kMaxDenseDofs = 220;
constexpr int kMaxDensePoints = 400;

// Two forms of the dense kernel. The first is plain loops, for any
// compiler and for a device. The second takes a block of elements as one
// vector of the widest register, with the compilers' vector types; it is
// the faster by a factor of two and more where there are instructions for
// it (AVX2, AVX-512), which the library is compiled with only when asked
// (ADGIA_NATIVE_KERNELS), and slower where there are not.

/// The elements the plain form takes at a time.
constexpr int kDenseRowsBlock = 4;

template <class Form>
void ApplyDenseRows(const int NE, const int ND, const int NQ,
                const mfem::DofToQuad& maps, const mfem::Vector& data_,
                const mfem::Vector& x_, mfem::Vector& y_) {
  constexpr int NC = Form::NC;
  MFEM_VERIFY(ND <= kMaxDenseDofs && NQ <= kMaxDensePoints,
              "displacement block kernel: too many dofs or quadrature "
              "points in an element (" << ND << ", " << NQ << ").");
  // Both layouts of the basis: points fastest (B, G) to go from the dofs
  // to the points, dofs fastest (Bt, Gt) to come back.
  const auto B = mfem::Reshape(maps.B.Read(), NQ, ND);
  const auto G = mfem::Reshape(maps.G.Read(), NQ, 3, ND);
  const auto Bt = mfem::Reshape(maps.Bt.Read(), ND, NQ);
  const auto Gt = mfem::Reshape(maps.Gt.Read(), ND, NQ, 3);
  const auto data =
      mfem::Reshape(data_.Read(), Form::Size, NQ, NE);
  const auto X = mfem::Reshape(x_.Read(), ND, NC, NE);
  auto Y = mfem::Reshape(y_.ReadWrite(), ND, NC, NE);
  // The loops are arranged so that the innermost adds a multiple of one
  // contiguous row to another (to the points: over the points; back: over
  // the dofs). A sum carried along the innermost loop would run at the
  // speed of one addition after another. The elements go a block at a
  // time so that a row of the basis, once read, serves the block.
  const int blocks = (NE + kDenseRowsBlock - 1) / kDenseRowsBlock;
  mfem::forall(blocks, [=] MFEM_HOST_DEVICE(int block) {
    const int e0 = block * kDenseRowsBlock;
    const int nb = NE - e0 < kDenseRowsBlock ? NE - e0 : kDenseRowsBlock;
    // At the points, for each element and component: the three reference
    // derivatives (rows 0 to 2) and the value (row 3).
    real_t w[kDenseRowsBlock][NC][4][kMaxDensePoints];
    real_t yl[kDenseRowsBlock][NC][kMaxDenseDofs];
    for (int k = 0; k < nb; k++) {
      for (int c = 0; c < NC; c++) {
        for (int r = 0; r < 4; r++) {
          for (int q = 0; q < NQ; q++) {
            w[k][c][r][q] = 0.0;
          }
        }
        for (int i = 0; i < ND; i++) {
          yl[k][c][i] = 0.0;
        }
      }
    }
    // To the points, four dofs at a time: each row at the points is then
    // written once for four rows of the basis read.
    const int ND4 = ND - ND % 4;
    for (int i = 0; i < ND4; i += 4) {
      for (int k = 0; k < nb; k++) {
        for (int c = 0; c < NC; c++) {
          const real_t x0 = X(i, c, e0 + k), x1 = X(i + 1, c, e0 + k),
                       x2 = X(i + 2, c, e0 + k), x3 = X(i + 3, c, e0 + k);
          for (int r = 0; r < 4; r++) {
            const real_t* t0 = r < 3 ? &G(0, r, i) : &B(0, i);
            const real_t* t1 = r < 3 ? &G(0, r, i + 1) : &B(0, i + 1);
            const real_t* t2 = r < 3 ? &G(0, r, i + 2) : &B(0, i + 2);
            const real_t* t3 = r < 3 ? &G(0, r, i + 3) : &B(0, i + 3);
            real_t* wr = w[k][c][r];
            for (int q = 0; q < NQ; q++) {
              wr[q] += t0[q] * x0 + t1[q] * x1 + t2[q] * x2 + t3[q] * x3;
            }
          }
        }
      }
    }
    for (int i = ND4; i < ND; i++) {
      for (int k = 0; k < nb; k++) {
        for (int c = 0; c < NC; c++) {
          const real_t x = X(i, c, e0 + k);
          for (int r = 0; r < 4; r++) {
            const real_t* t = r < 3 ? &G(0, r, i) : &B(0, i);
            real_t* wr = w[k][c][r];
            for (int q = 0; q < NQ; q++) {
              wr[q] += t[q] * x;
            }
          }
        }
      }
    }
    for (int k = 0; k < nb; k++) {
      for (int q = 0; q < NQ; q++) {
        real_t H[9], u[3], f[3] = {0.0, 0.0, 0.0};
        for (int c = 0; c < NC; c++) {
          H[3 * c] = w[k][c][0][q];
          H[3 * c + 1] = w[k][c][1][q];
          H[3 * c + 2] = w[k][c][2][q];
          u[c] = w[k][c][3][q];
        }
        Form::Point(data, q, e0 + k, u, H, f);
        for (int c = 0; c < NC; c++) {
          w[k][c][0][q] = H[3 * c];
          w[k][c][1][q] = H[3 * c + 1];
          w[k][c][2][q] = H[3 * c + 2];
          w[k][c][3][q] = f[c];
        }
      }
    }
    for (int q = 0; q < NQ; q++) {
      const real_t* b = &Bt(0, q);
      const real_t* g0 = &Gt(0, q, 0);
      const real_t* g1 = &Gt(0, q, 1);
      const real_t* g2 = &Gt(0, q, 2);
      for (int k = 0; k < nb; k++) {
        for (int c = 0; c < NC; c++) {
          const real_t h0 = w[k][c][0][q], h1 = w[k][c][1][q],
                       h2 = w[k][c][2][q], fc = w[k][c][3][q];
          real_t* y = yl[k][c];
          for (int i = 0; i < ND; i++) {
            y[i] += g0[i] * h0 + g1[i] * h1 + g2[i] * h2 + b[i] * fc;
          }
        }
      }
    }
    for (int k = 0; k < nb; k++) {
      for (int c = 0; c < NC; c++) {
        for (int i = 0; i < ND; i++) {
          Y(i, c, e0 + k) += yl[k][c][i];
        }
      }
    }
  });
}

#if (defined(__AVX2__) || defined(__AVX512F__)) && !defined(__CUDACC__) && \
    (defined(__GNUC__) || defined(__clang__))
#define ADGIA_DENSE_VECTOR_KERNEL 1

/// The elements the vector form takes at a time: a register's worth.
#if defined(__AVX512F__)
constexpr int kDenseBlock = 8;
#else
constexpr int kDenseBlock = 4;
#endif

template <class Form>
void ApplyDenseVector(const int NE, const int ND, const int NQ,
                const mfem::DofToQuad& maps, const mfem::Vector& data_,
                const mfem::Vector& x_, mfem::Vector& y_) {
  constexpr int NC = Form::NC;
  MFEM_VERIFY(ND <= kMaxDenseDofs,
              "displacement block kernel: too many dofs in an element ("
                  << ND << ").");
  // The transposed maps: dofs run fastest.
  const auto Bt = mfem::Reshape(maps.Bt.Read(), ND, NQ);
  const auto Gt = mfem::Reshape(maps.Gt.Read(), ND, NQ, 3);
  const auto data =
      mfem::Reshape(data_.Read(), Form::Size, NQ, NE);
  const auto X = mfem::Reshape(x_.Read(), ND, NC, NE);
  auto Y = mfem::Reshape(y_.ReadWrite(), ND, NC, NE);
  // A block of elements at a time, and the block is the vector: at a
  // point and a dof the same number of the basis multiplies the block's
  // values, so every operation below is on kDenseBlock elements at once.
  // The sums over the dofs are four vectors held in registers side by
  // side, so that none waits on the one before. The block's element
  // vectors are on the stack, dof by dof.
  const int blocks = (NE + kDenseBlock - 1) / kDenseBlock;
  mfem::forall(blocks, [=] MFEM_HOST_DEVICE(int block) {
    const int e0 = block * kDenseBlock;
    const int nb = NE - e0 < kDenseBlock ? NE - e0 : kDenseBlock;
    real_t xl[NC][kMaxDenseDofs][kDenseBlock], yl[NC][kMaxDenseDofs][kDenseBlock];
    for (int c = 0; c < NC; c++) {
      for (int i = 0; i < ND; i++) {
        for (int k = 0; k < kDenseBlock; k++) {
          xl[c][i][k] = k < nb ? X(i, c, e0 + k) : 0.0;
          yl[c][i][k] = 0.0;
        }
      }
    }
    typedef real_t vec __attribute__((vector_size(kDenseBlock * sizeof(real_t)), aligned(8)));
    for (int q = 0; q < NQ; q++) {
      const real_t* b = &Bt(0, q);
      const real_t* g0 = &Gt(0, q, 0);
      const real_t* g1 = &Gt(0, q, 1);
      const real_t* g2 = &Gt(0, q, 2);
      real_t w[NC][4][kDenseBlock];
      for (int c = 0; c < NC; c++) {
        vec a0 = {}, a1 = {}, a2 = {}, a3 = {};
        for (int i = 0; i < ND; i++) {
          const vec x = *reinterpret_cast<const vec*>(xl[c][i]);
          a0 += g0[i] * x;
          a1 += g1[i] * x;
          a2 += g2[i] * x;
          a3 += b[i] * x;
        }
        *reinterpret_cast<vec*>(w[c][0]) = a0;
        *reinterpret_cast<vec*>(w[c][1]) = a1;
        *reinterpret_cast<vec*>(w[c][2]) = a2;
        *reinterpret_cast<vec*>(w[c][3]) = a3;
      }
      for (int k = 0; k < nb; k++) {
        real_t H[9], u[3], f[3] = {0.0, 0.0, 0.0};
        for (int c = 0; c < NC; c++) {
          H[3 * c] = w[c][0][k];
          H[3 * c + 1] = w[c][1][k];
          H[3 * c + 2] = w[c][2][k];
          u[c] = w[c][3][k];
        }
        Form::Point(data, q, e0 + k, u, H, f);
        for (int c = 0; c < NC; c++) {
          w[c][0][k] = H[3 * c];
          w[c][1][k] = H[3 * c + 1];
          w[c][2][k] = H[3 * c + 2];
          w[c][3][k] = f[c];
        }
      }
      for (int c = 0; c < NC; c++) {
        const vec a0 = *reinterpret_cast<const vec*>(w[c][0]);
        const vec a1 = *reinterpret_cast<const vec*>(w[c][1]);
        const vec a2 = *reinterpret_cast<const vec*>(w[c][2]);
        const vec a3 = *reinterpret_cast<const vec*>(w[c][3]);
        for (int i = 0; i < ND; i++) {
          vec* y = reinterpret_cast<vec*>(yl[c][i]);
          *y += g0[i] * a0 + g1[i] * a1 + g2[i] * a2 + b[i] * a3;
        }
      }
    }
    for (int c = 0; c < NC; c++) {
      for (int i = 0; i < ND; i++) {
        for (int k = 0; k < nb; k++) {
          Y(i, c, e0 + k) += yl[c][i][k];
        }
      }
    }
  });
}
#endif  // the vector form

template <class Form>
void DiagonalDense(const int NE, const int ND, const int NQ,
                   const mfem::Array<real_t>& bt_,
                   const mfem::Array<real_t>& gt_, const mfem::Vector& data_,
                   mfem::Vector& diag_) {
  constexpr int NC = Form::NC;
  const auto Bt = mfem::Reshape(bt_.Read(), ND, NQ);
  const auto Gt = mfem::Reshape(gt_.Read(), ND, NQ, 3);
  const auto data =
      mfem::Reshape(data_.Read(), Form::Size, NQ, NE);
  auto Y = mfem::Reshape(diag_.ReadWrite(), ND, NC, NE);
  mfem::forall(NE, [=] MFEM_HOST_DEVICE(int e) {
    for (int q = 0; q < NQ; q++) {
      for (int c = 0; c < NC; c++) {
        real_t D[3][3];
        for (int a = 0; a < 3; a++) {
          for (int b = 0; b < 3; b++) {
            D[a][b] = Form::GradientDiagonal(data, q, e, c, a, b);
          }
        }
        const real_t m = Form::ValueDiagonal(data, q, e, c);
        for (int i = 0; i < ND; i++) {
          const real_t g[3] = {Gt(i, q, 0), Gt(i, q, 1), Gt(i, q, 2)};
          real_t t = m * Bt(i, q) * Bt(i, q);
          for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) {
              t += g[a] * D[a][b] * g[b];
            }
          }
          Y(i, c, e) += t;
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Sum-factorised: hexahedra, the basis in mfem's TENSOR mode. T_D and T_Q
// fix the 1-D sizes at compile time; zero leaves them to d1d and q1d
// within mfem's limits.

template <class Form, int T_D = 0, int T_Q = 0>
void ApplyTensor(const int NE, const mfem::Array<real_t>& b_,
                 const mfem::Array<real_t>& g_, const mfem::Vector& data_,
                 const mfem::Vector& x_, mfem::Vector& y_, const int d1d = 0,
                 const int q1d = 0) {
  constexpr int NC = Form::NC;
  const int D = T_D ? T_D : d1d;
  const int Q = T_Q ? T_Q : q1d;
  constexpr int MD = T_D ? T_D : mfem::DofQuadLimits::MAX_D1D;
  constexpr int MQ = T_Q ? T_Q : mfem::DofQuadLimits::MAX_Q1D;
  MFEM_VERIFY(D <= MD && Q <= MQ,
              "displacement block kernel: order too high.");
  const auto B = mfem::Reshape(b_.Read(), Q, D);
  const auto G = mfem::Reshape(g_.Read(), Q, D);
  const auto data =
      mfem::Reshape(data_.Read(), Form::Size, Q * Q * Q, NE);
  const auto X = mfem::Reshape(x_.Read(), D, D, D, NC, NE);
  auto Y = mfem::Reshape(y_.ReadWrite(), D, D, D, NC, NE);
  mfem::forall(NE, [=] MFEM_HOST_DEVICE(int e) {
    // At the points, for each component: the three reference derivatives
    // and (with extras) the value, at index 3.
    real_t grad[MQ][MQ][MQ][4 * NC];
    for (int qz = 0; qz < Q; qz++) {
      for (int qy = 0; qy < Q; qy++) {
        for (int qx = 0; qx < Q; qx++) {
          for (int i = 0; i < 4 * NC; i++) {
            grad[qz][qy][qx][i] = 0.0;
          }
        }
      }
    }
    for (int c = 0; c < NC; c++) {
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
              grad[qz][qy][qx][4 * c + 0] += gradXY[qy][qx][0] * wz;
              grad[qz][qy][qx][4 * c + 1] += gradXY[qy][qx][1] * wz;
              grad[qz][qy][qx][4 * c + 2] += gradXY[qy][qx][2] * wDz;
              if (Form::Values) {
                grad[qz][qy][qx][4 * c + 3] += gradXY[qy][qx][2] * wz;
              }
            }
          }
        }
      }
    }
    for (int qz = 0; qz < Q; qz++) {
      for (int qy = 0; qy < Q; qy++) {
        for (int qx = 0; qx < Q; qx++) {
          const int q = qx + Q * (qy + Q * qz);
          real_t* w = grad[qz][qy][qx];
          real_t H[9], u[3], f[3] = {0.0, 0.0, 0.0};
          for (int c = 0; c < NC; c++) {
            H[3 * c] = w[4 * c];
            H[3 * c + 1] = w[4 * c + 1];
            H[3 * c + 2] = w[4 * c + 2];
            u[c] = w[4 * c + 3];
          }
          Form::Point(data, q, e, u, H, f);
          for (int c = 0; c < NC; c++) {
            w[4 * c] = H[3 * c];
            w[4 * c + 1] = H[3 * c + 1];
            w[4 * c + 2] = H[3 * c + 2];
            w[4 * c + 3] = f[c];
          }
        }
      }
    }
    for (int c = 0; c < NC; c++) {
      for (int qz = 0; qz < Q; qz++) {
        real_t gradXY[MD][MD][4];
        for (int dy = 0; dy < D; dy++) {
          for (int dx = 0; dx < D; dx++) {
            gradXY[dy][dx][0] = 0.0;
            gradXY[dy][dx][1] = 0.0;
            gradXY[dy][dx][2] = 0.0;
            gradXY[dy][dx][3] = 0.0;
          }
        }
        for (int qy = 0; qy < Q; qy++) {
          real_t gradX[MD][4];
          for (int dx = 0; dx < D; dx++) {
            gradX[dx][0] = 0.0;
            gradX[dx][1] = 0.0;
            gradX[dx][2] = 0.0;
            gradX[dx][3] = 0.0;
          }
          for (int qx = 0; qx < Q; qx++) {
            const real_t gX = grad[qz][qy][qx][4 * c + 0];
            const real_t gY = grad[qz][qy][qx][4 * c + 1];
            const real_t gZ = grad[qz][qy][qx][4 * c + 2];
            const real_t fV = grad[qz][qy][qx][4 * c + 3];
            for (int dx = 0; dx < D; dx++) {
              const real_t wx = B(qx, dx), wDx = G(qx, dx);
              gradX[dx][0] += gX * wDx;
              gradX[dx][1] += gY * wx;
              gradX[dx][2] += gZ * wx;
              gradX[dx][3] += fV * wx;
            }
          }
          for (int dy = 0; dy < D; dy++) {
            const real_t wy = B(qy, dy), wDy = G(qy, dy);
            for (int dx = 0; dx < D; dx++) {
              gradXY[dy][dx][0] += gradX[dx][0] * wy;
              gradXY[dy][dx][1] += gradX[dx][1] * wDy;
              gradXY[dy][dx][2] += gradX[dx][2] * wy;
              gradXY[dy][dx][3] += gradX[dx][3] * wy;
            }
          }
        }
        for (int dz = 0; dz < D; dz++) {
          const real_t wz = B(qz, dz), wDz = G(qz, dz);
          for (int dy = 0; dy < D; dy++) {
            for (int dx = 0; dx < D; dx++) {
              Y(dx, dy, dz, c, e) +=
                  (gradXY[dy][dx][0] + gradXY[dy][dx][1] +
                   gradXY[dy][dx][3]) * wz +
                  gradXY[dy][dx][2] * wDz;
            }
          }
        }
      }
    }
  });
}

template <class Form, int T_D = 0, int T_Q = 0>
void DiagonalTensor(const int NE, const mfem::Array<real_t>& b_,
                    const mfem::Array<real_t>& g_, const mfem::Vector& data_,
                    mfem::Vector& diag_, const int d1d = 0,
                    const int q1d = 0) {
  constexpr int NC = Form::NC;
  const int D = T_D ? T_D : d1d;
  const int Q = T_Q ? T_Q : q1d;
  constexpr int MD = T_D ? T_D : mfem::DofQuadLimits::MAX_D1D;
  constexpr int MQ = T_Q ? T_Q : mfem::DofQuadLimits::MAX_Q1D;
  MFEM_VERIFY(D <= MD && Q <= MQ,
              "displacement block kernel: order too high.");
  const auto B = mfem::Reshape(b_.Read(), Q, D);
  const auto G = mfem::Reshape(g_.Read(), Q, D);
  const auto data =
      mfem::Reshape(data_.Read(), Form::Size, Q * Q * Q, NE);
  auto Y = mfem::Reshape(diag_.ReadWrite(), D, D, D, NC, NE);
  mfem::forall(NE, [=] MFEM_HOST_DEVICE(int e) {
    for (int c = 0; c < NC; c++) {
      // The nine pairs (a, b) of reference directions, then (pair 9) the
      // values: the sum over the points of a coefficient times the
      // products of the 1-D values or derivatives the pair calls for.
      for (int pair = 0; pair < (Form::Values ? 10 : 9); pair++) {
        const int a = pair < 9 ? pair / 3 : -1;
        const int b = pair < 9 ? pair % 3 : -1;
        real_t T2[MQ][MD][MD];
        for (int qz = 0; qz < Q; qz++) {
          real_t T1[MQ][MD];
          for (int qy = 0; qy < Q; qy++) {
            for (int dx = 0; dx < D; dx++) {
              T1[qy][dx] = 0.0;
            }
            for (int qx = 0; qx < Q; qx++) {
              const int q = qx + Q * (qy + Q * qz);
              const real_t coefficient =
                  pair < 9
                      ? Form::GradientDiagonal(data, q, e, c, a, b)
                      : Form::ValueDiagonal(data, q, e, c);
              for (int dx = 0; dx < D; dx++) {
                const real_t fa = a == 0 ? G(qx, dx) : B(qx, dx);
                const real_t fb = b == 0 ? G(qx, dx) : B(qx, dx);
                T1[qy][dx] += coefficient * fa * fb;
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
  });
}

// ---------------------------------------------------------------------------
// The launches.

template <class Form>
void Apply(int ne, const mfem::DofToQuad& maps, const mfem::Vector& data,
           const mfem::Vector& x, mfem::Vector& y) {
  const int D = maps.ndof, Q = maps.nqpt;
  if (maps.mode != mfem::DofToQuad::TENSOR) {
#ifdef ADGIA_DENSE_VECTOR_KERNEL
    return ApplyDenseVector<Form>(ne, D, Q, maps, data, x, y);
#else
    return ApplyDenseRows<Form>(ne, D, Q, maps, data, x, y);
#endif
  }
  switch (10 * D + Q) {
    case 23: return ApplyTensor<Form, 2, 3>(ne, maps.B, maps.G, data, x, y);
    case 34: return ApplyTensor<Form, 3, 4>(ne, maps.B, maps.G, data, x, y);
    case 45: return ApplyTensor<Form, 4, 5>(ne, maps.B, maps.G, data, x, y);
    case 56: return ApplyTensor<Form, 5, 6>(ne, maps.B, maps.G, data, x, y);
    case 67: return ApplyTensor<Form, 6, 7>(ne, maps.B, maps.G, data, x, y);
    default: return ApplyTensor<Form>(ne, maps.B, maps.G, data, x, y, D, Q);
  }
}

template <class Form>
void Diagonal(int ne, const mfem::DofToQuad& maps, const mfem::Vector& data,
              mfem::Vector& diag) {
  const int D = maps.ndof, Q = maps.nqpt;
  if (maps.mode != mfem::DofToQuad::TENSOR) {
    return DiagonalDense<Form>(ne, D, Q, maps.Bt, maps.Gt, data, diag);
  }
  switch (10 * D + Q) {
    case 23: return DiagonalTensor<Form, 2, 3>(ne, maps.B, maps.G, data, diag);
    case 34: return DiagonalTensor<Form, 3, 4>(ne, maps.B, maps.G, data, diag);
    case 45: return DiagonalTensor<Form, 4, 5>(ne, maps.B, maps.G, data, diag);
    case 56: return DiagonalTensor<Form, 5, 6>(ne, maps.B, maps.G, data, diag);
    case 67: return DiagonalTensor<Form, 6, 7>(ne, maps.B, maps.G, data, diag);
    default: return DiagonalTensor<Form>(ne, maps.B, maps.G, data, diag, D, Q);
  }
}

}  // namespace

void ApplyDisplacementBlock(bool isotropic, int extras, int ne,
                            const mfem::DofToQuad& maps,
                            const mfem::Vector& data, const mfem::Vector& x,
                            mfem::Vector& y) {
  MFEM_VERIFY(extras == kNoExtras || extras == kMassExtras ||
                  extras == kGravityExtras,
              "displacement block kernel: extras.");
#define ADGIA_LAUNCH(NX)                                   \
  if (extras == NX) {                                      \
    if (isotropic) {                                       \
      Apply<DisplacementForm<IsotropicTensor, NX>>(ne, maps, data, x, y);    \
    } else {                                               \
      Apply<DisplacementForm<GeneralTensor, NX>>(ne, maps, data, x, y);      \
    }                                                      \
  }
  ADGIA_LAUNCH(kNoExtras)
  ADGIA_LAUNCH(kMassExtras)
  ADGIA_LAUNCH(kGravityExtras)
#undef ADGIA_LAUNCH
}

void DiagonalDisplacementBlock(bool isotropic, int extras, int ne,
                               const mfem::DofToQuad& maps,
                               const mfem::Vector& data, mfem::Vector& diag) {
  MFEM_VERIFY(extras == kNoExtras || extras == kMassExtras ||
                  extras == kGravityExtras,
              "displacement block kernel: extras.");
#define ADGIA_LAUNCH(NX)                                   \
  if (extras == NX) {                                      \
    if (isotropic) {                                       \
      Diagonal<DisplacementForm<IsotropicTensor, NX>>(ne, maps, data, diag); \
    } else {                                               \
      Diagonal<DisplacementForm<GeneralTensor, NX>>(ne, maps, data, diag);   \
    }                                                      \
  }
  ADGIA_LAUNCH(kNoExtras)
  ADGIA_LAUNCH(kMassExtras)
  ADGIA_LAUNCH(kGravityExtras)
#undef ADGIA_LAUNCH
}

void ApplyPotentialBlock(int ne, const mfem::DofToQuad& maps,
                         const mfem::Vector& data, const mfem::Vector& x,
                         mfem::Vector& y) {
  Apply<PotentialForm>(ne, maps, data, x, y);
}

void DiagonalPotentialBlock(int ne, const mfem::DofToQuad& maps,
                            const mfem::Vector& data, mfem::Vector& diag) {
  Diagonal<PotentialForm>(ne, maps, data, diag);
}

}  // namespace AdGIA::detail::pa
