/* libCEED QFunction of the potential block: a diffusion form and a mass
 * term on a scalar space, at the quadrature points.
 *
 * This file is C, and libCEED's code-generating backends read it again at
 * run time: it includes <ceed/types.h> alone and is found through the path
 * the library was configured with (ADGIA_CEED_QFUNCTION_DIR).
 *
 * The data at a point are those of AdGIA/detail/pa_kernels.hpp: the
 * symmetric matrix k w J^{-1} J^{-T} in the reference directions (entries
 * 00, 01, 02, 11, 12, 22), then s w. Gradients are [a][point], the
 * derivative along the reference direction a. */
#ifndef ADGIA_POTENTIAL_BLOCK_QF_H
#define ADGIA_POTENTIAL_BLOCK_QF_H

#include <ceed/types.h>

CEED_QFUNCTION(AdGIAPotential)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *u                 = in[0];
  const CeedScalar(*ug)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*qd)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  CeedScalar *v                       = out[0];
  CeedScalar(*vg)[CEED_Q_VLA]         = (CeedScalar(*)[CEED_Q_VLA])out[1];
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar h0 = ug[0][i], h1 = ug[1][i], h2 = ug[2][i];
    vg[0][i] = qd[0][i] * h0 + qd[1][i] * h1 + qd[2][i] * h2;
    vg[1][i] = qd[1][i] * h0 + qd[3][i] * h1 + qd[4][i] * h2;
    vg[2][i] = qd[2][i] * h0 + qd[4][i] * h1 + qd[5][i] * h2;
    v[i]     = qd[6][i] * u[i];
  }
  return CEED_ERROR_SUCCESS;
}

#endif
