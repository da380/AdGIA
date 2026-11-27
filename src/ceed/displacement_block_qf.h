/* libCEED QFunctions of the displacement block: the elastic tensor form,
 * alone or with a mass term or with the gravity terms of the
 * self-gravitating problem and a mass term, at the quadrature points.
 * Reference gradient -> physical gradient -> strain -> stress -> back.
 *
 * This file is C, and libCEED's code-generating backends read it again at
 * run time: it includes <ceed/types.h> alone and is found through the path
 * the library was configured with (ADGIA_CEED_QFUNCTION_DIR).
 *
 * The data at a point are those of AdGIA/detail/pa_kernels.hpp: the 9
 * entries of the inverse Jacobian, d xi_a / d x_j at 3 a + j; the tensor
 * times the quadrature weight and the Jacobian's determinant, as lambda
 * and mu (isotropic) or as the upper triangle of its Voigt matrix
 * (ordering 11, 22, 33, 23, 13, 12); then the extras: none, or s, or g
 * (3), G by rows (9) and s. Gradients are [a][c][point], the derivative
 * of component c along the reference direction a; values [c][point].
 *
 * With g = (rho / 2) grad Phi0, G = (rho / 2) grad grad Phi0, s the mass
 * term's coefficient and du_ki = d_i u_k:
 *   tested with v_i:     g_k du_ki + (G_ik + G_ki) u_k - g_i div u + s u_i
 *   tested with d_i v_k: (C : eps(u))_ki + g_k u_i - delta_ki (g . u) */
#ifndef ADGIA_DISPLACEMENT_BLOCK_QF_H
#define ADGIA_DISPLACEMENT_BLOCK_QF_H

#include <ceed/types.h>

/* Gu[c][j] = d_j u_c from the reference gradient ug and the inverse
 * Jacobian. */
#define ADGIA_PHYSICAL_GRADIENT                                                                      \
  CeedScalar Gu[3][3], P[3][3];                                                                      \
  for (CeedInt c = 0; c < 3; c++) {                                                                  \
    for (CeedInt j = 0; j < 3; j++) {                                                                \
      Gu[c][j] = ug[0][c][i] * qd[j][i] + ug[1][c][i] * qd[3 + j][i] + ug[2][c][i] * qd[6 + j][i];   \
    }                                                                                                \
  }

/* P = C : eps(u), isotropic: lambda at 9, mu at 10. */
#define ADGIA_STRESS_ISOTROPIC                                                \
  {                                                                           \
    const CeedScalar lambda = qd[9][i], mu = qd[10][i];                       \
    const CeedScalar trace  = lambda * (Gu[0][0] + Gu[1][1] + Gu[2][2]);      \
    for (CeedInt c = 0; c < 3; c++) {                                         \
      for (CeedInt j = 0; j < 3; j++) P[c][j] = mu * (Gu[c][j] + Gu[j][c]);   \
      P[c][c] += trace;                                                       \
    }                                                                         \
  }

/* P = C : eps(u), the 21 entries of the Voigt matrix from 9. */
#define ADGIA_STRESS_GENERAL                                                                                              \
  {                                                                                                                       \
    const CeedScalar eps[6] = {Gu[0][0], Gu[1][1], Gu[2][2], Gu[1][2] + Gu[2][1], Gu[0][2] + Gu[2][0], Gu[0][1] + Gu[1][0]}; \
    CeedScalar       sig[6];                                                                                              \
    for (CeedInt r = 0; r < 6; r++) {                                                                                     \
      CeedScalar t = 0.0;                                                                                                 \
      for (CeedInt q = 0; q < 6; q++) {                                                                                   \
        const CeedInt lo = r <= q ? r : q, hi = r <= q ? q : r;                                                           \
        t += qd[9 + lo * 6 - (lo * (lo - 1)) / 2 + (hi - lo)][i] * eps[q];                                                \
      }                                                                                                                   \
      sig[r] = t;                                                                                                         \
    }                                                                                                                     \
    P[0][0] = sig[0]; P[0][1] = sig[5]; P[0][2] = sig[4];                                                                 \
    P[1][0] = sig[5]; P[1][1] = sig[1]; P[1][2] = sig[3];                                                                 \
    P[2][0] = sig[4]; P[2][1] = sig[3]; P[2][2] = sig[2];                                                                 \
  }

/* The mass term, its coefficient at index o. */
#define ADGIA_MASS(o)                                              \
  for (CeedInt a = 0; a < 3; a++) vv[a][i] = qd[(o)][i] * uu[a][i];

/* The gravity terms and the mass term, their data from index o. */
#define ADGIA_GRAVITY(o)                                                                               \
  {                                                                                                    \
    const CeedScalar g[3]  = {qd[(o)][i], qd[(o) + 1][i], qd[(o) + 2][i]};                             \
    const CeedScalar s     = qd[(o) + 12][i];                                                          \
    const CeedScalar u[3]  = {uu[0][i], uu[1][i], uu[2][i]};                                           \
    const CeedScalar divu  = Gu[0][0] + Gu[1][1] + Gu[2][2];                                           \
    const CeedScalar gdotu = g[0] * u[0] + g[1] * u[1] + g[2] * u[2];                                  \
    for (CeedInt a = 0; a < 3; a++) {                                                                  \
      CeedScalar f = g[0] * Gu[0][a] + g[1] * Gu[1][a] + g[2] * Gu[2][a] - g[a] * divu + s * u[a];     \
      for (CeedInt k = 0; k < 3; k++) {                                                                \
        f += (qd[(o) + 3 + 3 * a + k][i] + qd[(o) + 3 + 3 * k + a][i]) * u[k];                         \
        P[k][a] += g[k] * u[a];                                                                        \
      }                                                                                                \
      vv[a][i] = f;                                                                                    \
      P[a][a] -= gdotu;                                                                                \
    }                                                                                                  \
  }

/* What is tested with the reference gradient of v. */
#define ADGIA_REFERENCE_FLUX                                                                              \
  for (CeedInt c = 0; c < 3; c++) {                                                                       \
    for (CeedInt a = 0; a < 3; a++) {                                                                     \
      vg[a][c][i] = P[c][0] * qd[3 * a][i] + P[c][1] * qd[3 * a + 1][i] + P[c][2] * qd[3 * a + 2][i];     \
    }                                                                                                     \
  }

/* Fields of a form of gradients alone, and of values and gradients. */
#define ADGIA_FIELDS_GRADIENTS                                                          \
  const CeedScalar(*ug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0];     \
  const CeedScalar(*qd)[CEED_Q_VLA]    = (const CeedScalar(*)[CEED_Q_VLA])in[1];        \
  CeedScalar(*vg)[3][CEED_Q_VLA]       = (CeedScalar(*)[3][CEED_Q_VLA])out[0];
#define ADGIA_FIELDS_VALUES_GRADIENTS                                                   \
  const CeedScalar(*uu)[CEED_Q_VLA]    = (const CeedScalar(*)[CEED_Q_VLA])in[0];        \
  const CeedScalar(*ug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[1];     \
  const CeedScalar(*qd)[CEED_Q_VLA]    = (const CeedScalar(*)[CEED_Q_VLA])in[2];        \
  CeedScalar(*vv)[CEED_Q_VLA]          = (CeedScalar(*)[CEED_Q_VLA])out[0];             \
  CeedScalar(*vg)[3][CEED_Q_VLA]       = (CeedScalar(*)[3][CEED_Q_VLA])out[1];

CEED_QFUNCTION(AdGIAElasticIsotropic)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  ADGIA_FIELDS_GRADIENTS
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    ADGIA_PHYSICAL_GRADIENT
    ADGIA_STRESS_ISOTROPIC
    ADGIA_REFERENCE_FLUX
  }
  return CEED_ERROR_SUCCESS;
}

CEED_QFUNCTION(AdGIAElasticGeneral)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  ADGIA_FIELDS_GRADIENTS
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    ADGIA_PHYSICAL_GRADIENT
    ADGIA_STRESS_GENERAL
    ADGIA_REFERENCE_FLUX
  }
  return CEED_ERROR_SUCCESS;
}

CEED_QFUNCTION(AdGIAElasticMassIsotropic)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  ADGIA_FIELDS_VALUES_GRADIENTS
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    ADGIA_PHYSICAL_GRADIENT
    ADGIA_STRESS_ISOTROPIC
    ADGIA_MASS(11)
    ADGIA_REFERENCE_FLUX
  }
  return CEED_ERROR_SUCCESS;
}

CEED_QFUNCTION(AdGIAElasticMassGeneral)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  ADGIA_FIELDS_VALUES_GRADIENTS
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    ADGIA_PHYSICAL_GRADIENT
    ADGIA_STRESS_GENERAL
    ADGIA_MASS(30)
    ADGIA_REFERENCE_FLUX
  }
  return CEED_ERROR_SUCCESS;
}

CEED_QFUNCTION(AdGIAElasticGravityIsotropic)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  ADGIA_FIELDS_VALUES_GRADIENTS
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    ADGIA_PHYSICAL_GRADIENT
    ADGIA_STRESS_ISOTROPIC
    ADGIA_GRAVITY(11)
    ADGIA_REFERENCE_FLUX
  }
  return CEED_ERROR_SUCCESS;
}

CEED_QFUNCTION(AdGIAElasticGravityGeneral)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  ADGIA_FIELDS_VALUES_GRADIENTS
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    ADGIA_PHYSICAL_GRADIENT
    ADGIA_STRESS_GENERAL
    ADGIA_GRAVITY(30)
    ADGIA_REFERENCE_FLUX
  }
  return CEED_ERROR_SUCCESS;
}

#endif
