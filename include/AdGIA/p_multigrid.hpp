#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "mfem.hpp"

namespace AdGIA {

/**
 * @brief The orders of the levels of a p-multigrid for a space of order
 * @p order, coarsest first: @p coarse_order, then the orders reached from
 * @p order by halving (rounded up) while they stay above it. Order 3
 * over 2 gives {2, 3}; 4 gives {2, 4}; 5 gives {2, 3, 5}. One entry
 * (@p order itself) when @p order is not above @p coarse_order.
 */
std::vector<int> PMultigridOrders(int order, int coarse_order);

/**
 * @brief A p-multigrid V-cycle for the assembled matrix of an H1 space of
 * order above two: a preconditioner.
 *
 * The levels are H1 spaces on the same mesh of the orders of
 * PMultigridOrders(), with the vector dimension and ordering of the
 * given space. Between consecutive levels the prolongation is the
 * interpolation of the lower order's functions in the higher's basis,
 * as a matrix; the matrix of a coarser level is the Galerkin product
 * @f$P^T A P@f$. At essential dofs the matrix is taken to have the rows
 * and columns of an eliminated system, a diagonal entry alone: the
 * prolongation's rows there are removed, with the coarse dofs that reach
 * them, which become the essential dofs of the coarser level with a
 * unit diagonal.
 *
 * One cycle is a smoothing step on each level going down, a solve on the
 * coarsest and a smoothing step on each level going up.
 *
 * - Parallel: Chebyshev smoothing about the diagonal
 *   (mfem::HypreSmoother), of order Options::smoothing_order; the
 *   coarsest level's solver from the factory given, one BoomerAMG cycle
 *   by default.
 * - Serial: a symmetric Gauss-Seidel sweep on each level; on the
 *   coarsest the factory's solver, Options::coarse_sweeps symmetric
 *   Gauss-Seidel sweeps by default.
 *
 * With a symmetric positive definite matrix and a symmetric coarse
 * solver the cycle is symmetric positive definite. The matrix and the
 * space are not owned and must outlive the preconditioner; the matrix
 * is not to change under it.
 */
class PMultigridPreconditioner : public mfem::Solver {
 public:
  struct Options {
    int coarse_order = 2;     ///< order of the coarsest level
    int smoothing_order = 2;  ///< parallel: order of the Chebyshev smoother
    int coarse_sweeps = 4;    ///< serial default coarse solver: its sweeps
  };

  /// A solver for the coarsest level, from its space and its matrix (a
  /// SparseMatrix in serial, a HypreParMatrix in parallel). Both outlive
  /// the solver returned.
  using CoarseSolverFactory = std::function<std::unique_ptr<mfem::Solver>(
      mfem::FiniteElementSpace&, mfem::Operator&)>;

  /**
   * @param fes The H1 space of the matrix, of one order, above
   * Options::coarse_order.
   * @param A The assembled matrix on its vdofs, essential rows and
   * columns eliminated.
   * @param ess_dofs The essential vdofs.
   */
  PMultigridPreconditioner(mfem::FiniteElementSpace& fes,
                           const mfem::SparseMatrix& A,
                           const mfem::Array<int>& ess_dofs, Options options,
                           CoarseSolverFactory coarse = {});

#ifdef MFEM_USE_MPI
  /**
   * @param fes The H1 space of the matrix, of one order, above
   * Options::coarse_order.
   * @param A The assembled matrix on its true dofs, essential rows and
   * columns eliminated.
   * @param ess_tdofs This rank's essential true dofs.
   */
  PMultigridPreconditioner(mfem::ParFiniteElementSpace& fes,
                           mfem::HypreParMatrix& A,
                           const mfem::Array<int>& ess_tdofs, Options options,
                           CoarseSolverFactory coarse = {});
#endif

  ~PMultigridPreconditioner() override;

  /// The matrix is given at construction: not to be reset.
  void SetOperator(const mfem::Operator& op) override;

  /// One V-cycle from a zero start.
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

  /// The orders of the levels, coarsest first.
  const std::vector<int>& Orders() const { return orders_; }

  /// The matrix of a level (0 the coarsest).
  const mfem::Operator& LevelMatrix(int level) const {
    return *matrices_[level];
  }

 private:
  void MakeCycle();

  std::vector<int> orders_;
  std::vector<std::unique_ptr<mfem::FiniteElementCollection>> collections_;
  std::vector<std::unique_ptr<mfem::FiniteElementSpace>> spaces_;  // coarse
  std::vector<std::unique_ptr<mfem::Operator>> owned_matrices_;    // coarse
  std::vector<mfem::Operator*> matrices_;  // every level, coarsest first
  std::vector<std::unique_ptr<mfem::Operator>> prolongations_;
  std::vector<std::unique_ptr<mfem::Solver>> smoothers_;  // every level
  std::unique_ptr<mfem::Multigrid> cycle_;
};

}  // namespace AdGIA
