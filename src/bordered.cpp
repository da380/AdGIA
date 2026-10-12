#include "AdGIA/bordered.hpp"

namespace AdGIA {

using mfem::DenseMatrix;
using mfem::DenseMatrixInverse;
using mfem::real_t;
using mfem::Vector;

bool BorderedElimination::Eliminate(const std::vector<State>& columns,
                                    const DenseMatrix& Db, const Vector& r,
                                    int version, real_t tol, State& x,
                                    Vector& y) {
  const int nb = static_cast<int>(columns.size());
  MFEM_VERIFY(static_cast<int>(x.size()) == nc_,
              "Eliminate: the state must have the configured components");
  const bool cached =
      cache_version_ == version && cache_nb_ == nb && tol >= cache_tol_;
  if (!cached) {
    bool ok = true;
    Y_.assign(nb, State(nc_));
    for (int j = 0; j < nb; j++) {
      MFEM_VERIFY(static_cast<int>(columns[j].size()) == nc_,
                  "Eliminate: every column must have the configured "
                  "components");
      ok = solve_(columns[j], Y_[j]) && ok;
    }
    BtY_.SetSize(nb);
    for (int i = 0; i < nb; i++) {
      for (int j = 0; j < nb; j++) {
        real_t d = 0.0;
        for (int c = 0; c < nc_; c++) {
          d += dot_(columns[i][c], Y_[j][c]);
        }
        BtY_(i, j) = d;
      }
    }
    cache_ok_ = ok;
    cache_version_ = version;
    cache_nb_ = nb;
    cache_tol_ = tol;
  }

  // (Db - B^T Y) y = r - B^T x0.
  DenseMatrix S(nb);
  Vector rhs(nb);
  for (int i = 0; i < nb; i++) {
    rhs[i] = r[i];
    for (int c = 0; c < nc_; c++) {
      rhs[i] -= dot_(columns[i][c], x[c]);
    }
    for (int j = 0; j < nb; j++) {
      S(i, j) = Db(i, j) - BtY_(i, j);
    }
  }
  y.SetSize(nb);
  DenseMatrixInverse Sinv(S);
  Sinv.Mult(rhs, y);
  for (int j = 0; j < nb; j++) {
    for (int c = 0; c < nc_; c++) {
      x[c].Add(-y[j], Y_[j][c]);
    }
  }
  return cache_ok_;
}

}  // namespace AdGIA
