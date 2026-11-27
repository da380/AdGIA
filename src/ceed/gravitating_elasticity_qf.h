/* libCEED QFunctions of the displacement block of the self-gravitating
 * problem (GravitatingElasticityIntegrator): the elastic tensor form, the
 * symmetrised gravity terms and a mass term, at the quadrature points in
 * one pass.
 *
 * With g = (rho / 2) grad Phi0 and G = (rho / 2) grad grad Phi0, both
 * times the quadrature weight and the Jacobian's determinant, s the mass
 * term's coefficient likewise, and du_ki = d_i u_k:
 *
 *   tested with v_i:     g_k du_ki + (G_ik + G_ki) u_k - g_i div u + s u_i
 *   tested with d_i v_k: (C : eps(u))_ki + g_k u_i - delta_ki (g . u)
 *
 * This file is C, and libCEED's code-generating backends read it again at
 * run time (see elastic_tensor_qf.h). The data at a point: the 9 entries
 * of the inverse Jacobian, d xi_a / d x_j at 3 a + j; the elastic tensor
 * as in elastic_tensor_qf.h (2 or 21 entries); then g (3), G by rows (9)
 * and s (1). Gradients are [a][c][point], values [c][point]. */
#ifndef ADGIA_GRAVITATING_ELASTICITY_QF_H
#define ADGIA_GRAVITATING_ELASTICITY_QF_H

#include <ceed/types.h>

/* The gravity and mass terms, added to f (tested with v) and P (tested
 * with the physical gradient of v), from u, its physical gradient Gu
 * (Gu[c][j] = d_j u_c) and the data from index o on. */
#define ADGIA_GRAVITY_POINT(o)                                                                   \
  {                                                                                              \
    const CeedScalar g[3]  = {qd[(o)][i], qd[(o) + 1][i], qd[(o) + 2][i]};                       \
    const CeedScalar s     = qd[(o) + 12][i];                                                    \
    const CeedScalar u[3]  = {uu[0][i], uu[1][i], uu[2][i]};                                     \
    const CeedScalar divu  = Gu[0][0] + Gu[1][1] + Gu[2][2];                                     \
    const CeedScalar gdotu = g[0] * u[0] + g[1] * u[1] + g[2] * u[2];                            \
    for (CeedInt a = 0; a < 3; a++) {                                                            \
      CeedScalar f = g[0] * Gu[0][a] + g[1] * Gu[1][a] + g[2] * Gu[2][a] - g[a] * divu + s * u[a]; \
      for (CeedInt k = 0; k < 3; k++) f += (qd[(o) + 3 + 3 * a + k][i] + qd[(o) + 3 + 3 * k + a][i]) * u[k]; \
      vv[a][i] = f;                                                                              \
      for (CeedInt k = 0; k < 3; k++) P[k][a] += g[k] * u[a];                                    \
      P[a][a] -= gdotu;                                                                          \
    }                                                                                            \
  }

CEED_QFUNCTION(AdGIAGravitatingElasticityIsotropic)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*uu)[CEED_Q_VLA]    = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*ug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[1];
  const CeedScalar(*qd)[CEED_Q_VLA]    = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  CeedScalar(*vv)[CEED_Q_VLA]          = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*vg)[3][CEED_Q_VLA]       = (CeedScalar(*)[3][CEED_Q_VLA])out[1];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar Gu[3][3], P[3][3];
    for (CeedInt c = 0; c < 3; c++) {
      for (CeedInt j = 0; j < 3; j++) {
        Gu[c][j] = ug[0][c][i] * qd[j][i] + ug[1][c][i] * qd[3 + j][i] + ug[2][c][i] * qd[6 + j][i];
      }
    }
    const CeedScalar lambda = qd[9][i], mu = qd[10][i];
    const CeedScalar trace = lambda * (Gu[0][0] + Gu[1][1] + Gu[2][2]);
    for (CeedInt c = 0; c < 3; c++) {
      for (CeedInt j = 0; j < 3; j++) P[c][j] = mu * (Gu[c][j] + Gu[j][c]);
      P[c][c] += trace;
    }
    ADGIA_GRAVITY_POINT(11)
    for (CeedInt c = 0; c < 3; c++) {
      for (CeedInt a = 0; a < 3; a++) {
        vg[a][c][i] = P[c][0] * qd[3 * a][i] + P[c][1] * qd[3 * a + 1][i] + P[c][2] * qd[3 * a + 2][i];
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

CEED_QFUNCTION(AdGIAGravitatingElasticityGeneral)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*uu)[CEED_Q_VLA]    = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*ug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[1];
  const CeedScalar(*qd)[CEED_Q_VLA]    = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  CeedScalar(*vv)[CEED_Q_VLA]          = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*vg)[3][CEED_Q_VLA]       = (CeedScalar(*)[3][CEED_Q_VLA])out[1];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar Gu[3][3], P[3][3];
    for (CeedInt c = 0; c < 3; c++) {
      for (CeedInt j = 0; j < 3; j++) {
        Gu[c][j] = ug[0][c][i] * qd[j][i] + ug[1][c][i] * qd[3 + j][i] + ug[2][c][i] * qd[6 + j][i];
      }
    }
    const CeedScalar eps[6] = {Gu[0][0], Gu[1][1], Gu[2][2], Gu[1][2] + Gu[2][1], Gu[0][2] + Gu[2][0], Gu[0][1] + Gu[1][0]};
    CeedScalar       sig[6];
    for (CeedInt r = 0; r < 6; r++) {
      CeedScalar t = 0.0;
      for (CeedInt q = 0; q < 6; q++) {
        const CeedInt lo = r <= q ? r : q, hi = r <= q ? q : r;
        t += qd[9 + lo * 6 - (lo * (lo - 1)) / 2 + (hi - lo)][i] * eps[q];
      }
      sig[r] = t;
    }
    P[0][0] = sig[0]; P[0][1] = sig[5]; P[0][2] = sig[4];
    P[1][0] = sig[5]; P[1][1] = sig[1]; P[1][2] = sig[3];
    P[2][0] = sig[4]; P[2][1] = sig[3]; P[2][2] = sig[2];
    ADGIA_GRAVITY_POINT(30)
    for (CeedInt c = 0; c < 3; c++) {
      for (CeedInt a = 0; a < 3; a++) {
        vg[a][c][i] = P[c][0] * qd[3 * a][i] + P[c][1] * qd[3 * a + 1][i] + P[c][2] * qd[3 * a + 2][i];
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

#endif
