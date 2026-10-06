#ifndef ADGIA_DESCENT_HPP
#define ADGIA_DESCENT_HPP

#include <functional>
#include <memory>

#include "mfem.hpp"

#include "AdGIA/riesz.hpp"

namespace AdGIA {

/**
 * @brief The descent toolkit of the equilibrium-figures programme
 * (doc/equilibrium_figures.tex §6, §8): a constrained metric — Riesz
 * identification, one linear constraint held by metric-orthogonal
 * projection, an optional quadratic prior — and two loops over it,
 * nonlinear conjugate gradients and Levenberg–Marquardt Gauss–Newton.
 * The loops see a problem only through DescentFunctional, so any
 * control field with a dual-valued derivative (density today, shape
 * parameters later) runs through the same machinery. Serial and
 * parallel: the loops work on true-dof vectors and every reduction is
 * a RieszMap pairing.
 */

/**
 * @brief What a descent loop needs of a problem. @c evaluate returns
 * the value at @p c and, when @p dual is non-null, assembles the
 * derivative against the control space. @c gauss_newton (needed by
 * LevenbergMarquardt only) applies the Gauss–Newton Hessian of the
 * state LAST EVALUATED to a direction — the caller's closure holds
 * that state, so no re-evaluation is forced on the matrix-vector
 * products.
 */
struct DescentFunctional {
  std::function<mfem::real_t(const mfem::Vector& c, mfem::Vector* dual)>
      evaluate;
  std::function<void(const mfem::Vector& d, mfem::Vector& Hd)> gauss_newton;
};

/**
 * @brief The metric of a descent loop: a Riesz map identifying duals
 * with gradients, at most one linear constraint @f$\ell\cdot c =
 * \ell\cdot c_0@f$ (the fixed fluid mass) held by metric-orthogonal
 * projection, and an optional quadratic prior
 * @f$\tfrac{\lambda}{2}\,c^T K c@f$ whose Gram the caller assembles
 * (the roughness seminorm of the reference document — the selection
 * within the near-null directions). NOTE: with the prior on, the
 * Levenberg–Marquardt inner solve is mesh-independent only when its
 * preconditioner is spectrally equivalent to the damped system
 * @f$\lambda G + \lambda_p K@f$ — a Sobolev Riesz map, supplied
 * through DescentOptions::inner_preconditioner; the default (the
 * metric's own Riesz map) leaves an unbounded @f$\lambda_p M^{-1}K@f$
 * term whose iteration count grows under refinement (reference
 * document §9). Everything is borrowed and must outlive the metric.
 */
class ConstrainedMetric {
 public:
  explicit ConstrainedMetric(RieszMap& riesz);

  /** @brief The constraint's dual @f$\ell@f$ (e.g. the assembled mass
   * functional); its Riesz representative is formed once. */
  void SetConstraint(const mfem::Vector& ell);

  /** @brief The prior's Gram @p K (true-dof operator) and weight. */
  void SetPrior(const mfem::Operator& K, mfem::real_t weight);

  RieszMap& Riesz() const { return *riesz_; }
  bool HasPrior() const { return prior_ != nullptr && weight_ != 0.0; }
  mfem::real_t PriorWeight() const { return weight_; }

  /** @brief The prior's value at @p c (zero without one). */
  mfem::real_t PriorValue(const mfem::Vector& c) const;

  /** @brief Adds the prior's derivative to the dual @p j. */
  void AddPriorDual(const mfem::Vector& c, mfem::Vector& j) const;

  /** @brief Adds the prior's Gauss–Newton (= exact) Hessian action. */
  void AddPriorHessian(const mfem::Vector& d, mfem::Vector& Hd) const;

  /** @brief The gradient of a dual: Riesz, then the constraint-plane
   * projection (the projector is metric-self-adjoint, so pairings of
   * raw duals with projected fields are the manifold's own inner
   * products). */
  void Gradient(const mfem::Vector& dual, mfem::Vector& g) const;

  /** @brief Projects a field onto the constraint plane. */
  void Project(mfem::Vector& v) const;

 private:
  RieszMap* riesz_;
  const mfem::Operator* prior_ = nullptr;
  mfem::real_t weight_ = 0.0;
  // Whether a constraint is set is a flag, NOT ell_.Size() != 0: the
  // local true-dof count is legitimately zero on a rank owning no
  // elements of the control space, and an early return there deserts
  // the collective pairing the other ranks enter (measured deadlock at
  // 8 ranks on a submesh control).
  bool has_constraint_ = false;
  mfem::Vector ell_, ell_gradient_;
  mfem::real_t ell_scale_ = 0.0;
};

/** @brief Common controls of the loops. The callback (when set) sees
 * every accepted iteration: (iteration, value, step-or-lambda). */
struct DescentOptions {
  int max_iterations = 100;
  /** The Levenberg–Marquardt inner CG's preconditioner (null: the
   * metric's Riesz map; see the ConstrainedMetric note on priors). */
  RieszMap* inner_preconditioner = nullptr;
  /** Stop when the value falls below this fraction of its start;
   * non-positive disables the test (run to the iteration or gradient
   * floor). */
  mfem::real_t tolerance = 1e-3;
  bool print = false;
  std::function<void(int, mfem::real_t, mfem::real_t)> monitor;
  /** Optional stopping predicate, consulted after each accepted
   * iteration (after the monitor, so it may read the functional's
   * state at the accepted iterate): return true to stop, reported as
   * converged. This is how a discrepancy-style rule — stop at a
   * dimensionless infeasibility floor rather than at a fraction of the
   * starting value — reaches the loops (reference document §9). */
  std::function<bool(int, mfem::real_t)> stop;
};

struct DescentResult {
  int iterations = 0;
  mfem::real_t initial = 0.0, final = 0.0;
  bool converged = false;
};

/**
 * @brief Projected Polak–Ribière+ nonlinear CG with a forward-tracking
 * line search (the objective is smooth and, for the feasibility
 * functional, an exact quartic along rays: the accepted step is within
 * a factor two of the ray's minimiser, warm-started across
 * iterations). The prior, when the metric carries one, is part of the
 * descended objective; the reported values are the functional's own.
 */
DescentResult NonlinearCG(const DescentFunctional& f,
                          const ConstrainedMetric& metric, mfem::Vector& c,
                          const DescentOptions& options = {});

/**
 * @brief Levenberg–Marquardt on the Gauss–Newton model: the inner
 * preconditioned CG solves @f$(H_{GN} + \lambda\,G + \lambda_p K)\,s
 * = -j@f$ — @p damping is the Gram @f$G@f$ of the damping term
 * (typically the control's mass matrix), the metric's Riesz map
 * preconditions, and @f$\lambda@f$ adapts multiplicatively on the
 * acceptance test. The damping is iterated-Tikhonov regularisation; at
 * the functional's floor Gauss–Newton is exact and the convergence is
 * superlinear with the prior on (reference document §8).
 */
DescentResult LevenbergMarquardt(const DescentFunctional& f,
                                 const ConstrainedMetric& metric,
                                 const mfem::Operator& damping,
                                 mfem::Vector& c,
                                 const DescentOptions& options = {});

}  // namespace AdGIA

#endif  // ADGIA_DESCENT_HPP
