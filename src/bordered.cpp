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
  // Most-recently-used first: a hit on the second entry is swapped to
  // the front, a miss rebuilds the back entry and swaps it to the
  // front, so two alternating operator versions each keep their
  // column set.
  if (!(cache_[0].version == version && cache_[0].nb == nb &&
        tol >= cache_[0].tol)) {
    if (cache_[1].version == version && cache_[1].nb == nb &&
        tol >= cache_[1].tol) {
      std::swap(cache_[0], cache_[1]);
    } else {
      Entry& e = cache_[1];
      bool ok = true;
      e.Y.assign(nb, State(nc_));
      for (int j = 0; j < nb; j++) {
        MFEM_VERIFY(static_cast<int>(columns[j].size()) == nc_,
                    "Eliminate: every column must have the configured "
                    "components");
        // A zero column (a border unknown coupled only within the dense
        // block) has the zero response: no solve.
        real_t nrm2 = 0.0;
        for (int c = 0; c < nc_; c++) {
          nrm2 += dot_(columns[j][c], columns[j][c]);
        }
        if (nrm2 == 0.0) {
          for (int c = 0; c < nc_; c++) {
            e.Y[j][c].SetSize(columns[j][c].Size());
            e.Y[j][c] = 0.0;
          }
          continue;
        }
        ok = solve_(columns[j], e.Y[j]) && ok;
      }
      e.BtY.SetSize(nb);
      for (int i = 0; i < nb; i++) {
        for (int j = 0; j < nb; j++) {
          real_t d = 0.0;
          for (int c = 0; c < nc_; c++) {
            d += dot_(columns[i][c], e.Y[j][c]);
          }
          e.BtY(i, j) = d;
        }
      }
      e.ok = ok;
      e.version = version;
      e.nb = nb;
      e.tol = tol;
      std::swap(cache_[0], cache_[1]);
    }
  }
  const std::vector<State>& Y_ = cache_[0].Y;
  const DenseMatrix& BtY_ = cache_[0].BtY;

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
  return cache_[0].ok;
}

}  // namespace AdGIA
