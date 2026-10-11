#ifndef ADGIA_BORDERED_HPP
#define ADGIA_BORDERED_HPP

#include <functional>
#include <vector>

#include "mfem.hpp"

namespace AdGIA {

/**
 * @brief Block elimination for bordered systems
 * @f$[K\ B;\ B^T\ D_b]\,[x;y] = [b;r]@f$: the base solve, then one
 * solve per border column and a dense reduction
 * (doc/planning/sea_level_plan.md, "The condensed system").
 *
 * The problem class supplies the base solve and the (parallel-safe)
 * dot product; a column — and the state @f$x@f$ — is a fixed number of
 * true-dof vectors (the mixed class: the displacement system and the
 * potential; the referential class: one monolithic vector), so the
 * class-side conventions (compatibility fixes, masked coefficients)
 * stay inside its solve callback. The columns are load-independent and
 * are cached across solves, keyed on the operator version and the
 * tolerance they were built at (the viscoelastic stepping economy:
 * doc/planning/solvers.md, "Bordered solves").
 */
class BorderedElimination {
 public:
  /** @brief A state or column: one true-dof vector per component. */
  using State = std::vector<mfem::Vector>;
  /** @brief Solve @f$K x = \mathrm{rhs}@f$ (the callee sizes @p x). */
  using Solve = std::function<bool(const State& rhs, State& x)>;
  /** @brief Parallel-safe dot of two same-component vectors. */
  using Dot =
      std::function<mfem::real_t(const mfem::Vector&, const mfem::Vector&)>;

  /** @param components Vectors per state (equal in every column).
   * @param solve The base solve against the current operator.
   * @param dot The dot product of the owning class. */
  BorderedElimination(int components, Solve solve, Dot dot)
      : nc_(components), solve_(std::move(solve)), dot_(std::move(dot)) {}

  /**
   * @brief Eliminate the border: on entry @p x is the base solution
   * @f$x_0@f$ of @f$K x_0 = b@f$; on return @f$x = x_0 - Y y@f$ with
   * @f$(D_b - B^T Y)\,y = r - B^T x_0@f$.
   *
   * @param columns The border columns @f$B_j@f$, each sized like @p x.
   * @param Db The dense border block.
   * @param r The border right-hand side.
   * @param version The operator version the columns were assembled
   * against; a change invalidates the cached column solves.
   * @param tol The solver tolerance: cached columns are reused only
   * for solves at this tolerance or looser.
   * @param x In: @f$x_0@f$; out: the corrected state.
   * @param y Out: the border unknowns (sized here).
   * @return Whether every column solve converged (cached columns count
   * as converged).
   */
  bool Eliminate(const std::vector<State>& columns,
                 const mfem::DenseMatrix& Db, const mfem::Vector& r,
                 int version, mfem::real_t tol, State& x, mfem::Vector& y);

  /** @brief Drop the cached column solves. */
  void InvalidateCache() { cache_nb_ = -1; }

 private:
  int nc_;
  Solve solve_;
  Dot dot_;
  std::vector<State> Y_;
  int cache_version_ = -1;
  int cache_nb_ = -1;
  mfem::real_t cache_tol_ = 0.0;
};

}  // namespace AdGIA

#endif
