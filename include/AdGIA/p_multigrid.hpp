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

/**
 * @brief A p-multigrid V-cycle with no matrix above the coarsest level: a
 * preconditioner for forms applied by partial assembly.
 *
 * The levels are H1 spaces on the same mesh of the orders of
 * PMultigridOrders(), with the vector dimension, ordering and basis of
 * the given space. On every level the form is made again from its
 * integrators (given as a function that adds them to a form): by partial
 * assembly above the coarsest level, as a matrix on the coarsest. The
 * levels are therefore discretisations of the form at their orders, not
 * Galerkin products. Between levels mfem's transfer of the space
 * hierarchy, applied with no matrix on tensor-product elements.
 *
 * One cycle is a Chebyshev step about the form's diagonal
 * (Options::smoothing_order) on each level going down, a solve on the
 * coarsest and a Chebyshev step on each level going up. The coarsest
 * level's solver comes from the factory given: by default one BoomerAMG
 * cycle in parallel (with Chebyshev smoothing when hypre runs on a
 * device, where its own default is l1-Jacobi) and Options::coarse_sweeps
 * symmetric Gauss-Seidel sweeps in serial.
 *
 * Essential dofs are those of the boundary attributes marked, on every
 * level. The integrators must support partial assembly and its diagonal
 * on the mesh's elements (for ElasticTensorIntegrator: hexahedra). Above
 * the coarsest level everything runs on the configured mfem::Device.
 *
 * The space and what the integrators refer to are not owned and must
 * outlive the preconditioner.
 */
class MatrixFreePMultigrid : public mfem::Solver {
 public:
  struct Options {
    int coarse_order = 2;     ///< order of the coarsest level
    int smoothing_order = 2;  ///< order of the Chebyshev smoother
    int coarse_sweeps = 4;    ///< serial default coarse solver: its sweeps
  };

  /// Adds the form's integrators to the form of a level.
  using Integrators = std::function<void(mfem::BilinearForm&)>;

  using CoarseSolverFactory = PMultigridPreconditioner::CoarseSolverFactory;

  /**
   * @param fes The H1 space of the problem, of one order, above
   * Options::coarse_order; serial or parallel.
   * @param ess_bdr Marker of the essential boundary attributes; empty or
   * all zero for none.
   * @param integrators Adds the form's integrators to a level's form.
   */
  MatrixFreePMultigrid(mfem::FiniteElementSpace& fes,
                       const mfem::Array<int>& ess_bdr,
                       Integrators integrators, Options options,
                       CoarseSolverFactory coarse = {});

  ~MatrixFreePMultigrid() override;

  /// The forms are made at construction: not to be reset.
  void SetOperator(const mfem::Operator& op) override;

  /// One V-cycle from a zero start.
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

  /// The orders of the levels, coarsest first.
  const std::vector<int>& Orders() const { return orders_; }

  /// The operator of the finest level on the true dofs, essential rows
  /// and columns eliminated: the form of @p fes by partial assembly.
  const mfem::Operator& FineOperator() const;

  /// The essential true dofs of the finest level.
  const mfem::Array<int>& FineEssentialTrueDofs() const;

 private:
  class Cycle;
  std::vector<int> orders_;
  std::vector<std::unique_ptr<mfem::FiniteElementCollection>> collections_;
  std::unique_ptr<mfem::FiniteElementSpaceHierarchy> hierarchy_;
  std::unique_ptr<mfem::Operator> coarse_matrix_;
  std::unique_ptr<mfem::Solver> coarse_solver_;
  std::unique_ptr<Cycle> cycle_;
};

}  // namespace AdGIA
