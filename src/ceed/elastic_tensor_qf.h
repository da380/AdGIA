/* libCEED QFunctions of the elastic tensor form, (u, v) -> int grad v : C :
 * grad u: the work at the quadrature points. Reference gradient ->
 * physical gradient -> strain -> stress -> back.
 *
 * This file is C, and libCEED's code-generating backends read it again at
 * run time: it includes <ceed/types.h> alone and is found through the path
 * the library was configured with (ADGIA_CEED_QFUNCTION_DIR).
 *
 * The data at a point are those of AdGIA/detail/pa_kernels.hpp: the 9
 * entries of the inverse Jacobian, d xi_a / d x_j at 3 a + j, then the
 * tensor times the quadrature weight and the Jacobian's determinant, as
 * lambda and mu (isotropic) or as the upper triangle of its Voigt matrix
 * (ordering 11, 22, 33, 23, 13, 12). The gradients are [a][c][point]: the
 * derivative of component c along the reference direction a. */
#ifndef ADGIA_ELASTIC_TENSOR_QF_H
#define ADGIA_ELASTIC_TENSOR_QF_H

#include <ceed/types.h>

CEED_QFUNCTION(AdGIAElasticTensorIsotropic)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*ug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0];
  const CeedScalar(*qd)[CEED_Q_VLA]    = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  CeedScalar(*vg)[3][CEED_Q_VLA]       = (CeedScalar(*)[3][CEED_Q_VLA])out[0];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar G[3][3];
    for (CeedInt c = 0; c < 3; c++) {
      for (CeedInt j = 0; j < 3; j++) {
        G[c][j] = ug[0][c][i] * qd[j][i] + ug[1][c][i] * qd[3 + j][i] + ug[2][c][i] * qd[6 + j][i];
      }
    }
    const CeedScalar lambda = qd[9][i], mu = qd[10][i];
    const CeedScalar trace = lambda * (G[0][0] + G[1][1] + G[2][2]);
    CeedScalar       S[3][3];
    for (CeedInt c = 0; c < 3; c++) {
      for (CeedInt j = 0; j < 3; j++) S[c][j] = mu * (G[c][j] + G[j][c]);
      S[c][c] += trace;
    }
    for (CeedInt c = 0; c < 3; c++) {
      for (CeedInt a = 0; a < 3; a++) {
        vg[a][c][i] = S[c][0] * qd[3 * a][i] + S[c][1] * qd[3 * a + 1][i] + S[c][2] * qd[3 * a + 2][i];
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

CEED_QFUNCTION(AdGIAElasticTensorGeneral)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*ug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0];
  const CeedScalar(*qd)[CEED_Q_VLA]    = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  CeedScalar(*vg)[3][CEED_Q_VLA]       = (CeedScalar(*)[3][CEED_Q_VLA])out[0];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar G[3][3];
    for (CeedInt c = 0; c < 3; c++) {
      for (CeedInt j = 0; j < 3; j++) {
        G[c][j] = ug[0][c][i] * qd[j][i] + ug[1][c][i] * qd[3 + j][i] + ug[2][c][i] * qd[6 + j][i];
      }
    }
    const CeedScalar eps[6] = {G[0][0], G[1][1], G[2][2], G[1][2] + G[2][1], G[0][2] + G[2][0], G[0][1] + G[1][0]};
    CeedScalar       sig[6];
    for (CeedInt r = 0; r < 6; r++) {
      CeedScalar s = 0.0;
      for (CeedInt t = 0; t < 6; t++) {
        /* packed index of the upper triangle of the symmetric 6 x 6 matrix */
        const CeedInt lo = r <= t ? r : t, hi = r <= t ? t : r;
        s += qd[9 + lo * 6 - (lo * (lo - 1)) / 2 + (hi - lo)][i] * eps[t];
      }
      sig[r] = s;
    }
    const CeedScalar S[3][3] = {
        {sig[0], sig[5], sig[4]},
        {sig[5], sig[1], sig[3]},
        {sig[4], sig[3], sig[2]}
    };
    for (CeedInt c = 0; c < 3; c++) {
      for (CeedInt a = 0; a < 3; a++) {
        vg[a][c][i] = S[c][0] * qd[3 * a][i] + S[c][1] * qd[3 * a + 1][i] + S[c][2] * qd[3 * a + 2][i];
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

#endif
