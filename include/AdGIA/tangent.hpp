/**
 * @file tangent.hpp
 * @brief The tangent-linear (forward-mode) problem: the derivative of a
 * static or quasi-static viscoelastic solution along a direction in the
 * model (bulk and shear moduli, branch moduli and viscosities) and in the
 * load, as one more march of the same stepping with source terms built
 * from the stored forward trajectory. The forward-mode counterpart of
 * adjoint.hpp, and the first ingredient of second-order adjoints and of
 * Gauss-Newton (Hessians plan, 5 November 2026, item A2).
 */
#pragma once
#include <functional>
#include <memory>
#include <vector>

#include "mfem.hpp"
#include "AdGIA/adjoint.hpp"
#include "AdGIA/quasi_static_problem.hpp"
#include "AdGIA/viscoelastic.hpp"

namespace AdGIA {

/**
 * @brief A direction in the isotropic generalised Maxwell model
 * (kappa, mu_inf, {mu_k, eta_k}) of the library's parameter convention
 * (IsotropicKernelDensity): absolute perturbations as coefficients, any
 * of them null for zero. A branch's shear modulus is perturbed at FIXED
 * eta_k (so its relaxation time moves with it), eta_k through the
 * evolution alone, as the kernels are defined. The LOAD direction is
 * not here: it is the tangent twin problem's own load (its
 * SetSurfaceLoad, traction or tidal potential), since the forward
 * problem is linear in the load.
 */
struct TangentDirection {
  mfem::Coefficient* kappa = nullptr;        ///< delta kappa
  mfem::Coefficient* mu_longterm = nullptr;  ///< delta mu_inf
  std::vector<mfem::Coefficient*> mu;        ///< delta mu_k per branch
  std::vector<mfem::Coefficient*> eta;       ///< delta eta_k per branch

  mfem::Coefficient* Mu(int k) const {
    return k < static_cast<int>(mu.size()) ? mu[k] : nullptr;
  }
  mfem::Coefficient* Eta(int k) const {
    return k < static_cast<int>(eta.size()) ? eta[k] : nullptr;
  }
  bool HasElasticModuli() const;
};

/**
 * @brief The Born force of a moduli perturbation: minus the elastic form
 * assembled with the perturbation (delta kappa, delta mu_U) applied to a
 * displacement, in the AddForce layout. The integrators are those of the
 * isotropic rheology (kappa div-div + 2 mu dev-dev) with the same default
 * rules, so the force is the exact derivative of the assembled stiffness
 * along the direction. delta mu_U = delta mu_inf + sum_k delta mu_k (the
 * unrelaxed modulus is what the stiffness carries).
 */
class ModuliBornForce {
 public:
  ModuliBornForce(mfem::FiniteElementSpace& fes, const TangentDirection& dir,
                  int num_branches);
  bool Empty() const { return !form_; }
  /** @brief f += -dK u (L-vector layout). */
  void AddTo(const mfem::GridFunction& u, mfem::Vector& f);

 private:
  std::vector<std::unique_ptr<mfem::Coefficient>> owned_;
  std::unique_ptr<mfem::BilinearForm> form_;
  mfem::Vector buf_;
};

/**
 * @brief The recorded tangent trajectory: per forward node the tangent
 * internal state, displacement and (on a gravitating twin) potential.
 */
class TangentTrajectory {
 public:
  int Size() const { return static_cast<int>(t_.size()); }
  mfem::real_t Time(int n) const { return t_[n]; }
  const mfem::Vector& State(int n) const { return m_[n]; }
  const mfem::GridFunction& Displacement(int n) const { return *u_[n]; }
  /** @brief The tangent potential at node @p n, null when the twin has
   * none. */
  const mfem::GridFunction* Potential(int n) const {
    return n < static_cast<int>(phi_.size()) ? phi_[n].get() : nullptr;
  }
  void Clear();
  void Record(mfem::real_t t, const mfem::Vector& m,
              const mfem::GridFunction& u, const mfem::GridFunction* phi);

 private:
  std::vector<mfem::real_t> t_;
  std::vector<mfem::Vector> m_;
  std::vector<std::unique_ptr<mfem::GridFunction>> u_, phi_;
};

/**
 * @brief The tangent of the exponential-trapezoid viscoelastic evolution:
 * the forward operator's own stepping on the zero-load (or
 * load-direction) twin problem, with the derivative's source terms
 * injected at every elastic solve through the force hook and the
 * internal-variable update completed after each step.
 *
 * Differentiating the step @f$m_k^{n+1} = e_k m_k^n + a_k d^n + b_k
 * d^{n+1}@f$ (weights at @f$h_k = dt/\tau_k@f$) and the solve @f$K_U u =
 * f + \sum_k B^T C_k m_k@f$ along a direction gives the same step for
 * @f$(\delta u, \delta m)@f$ with
 *
 *  - the Born force @f$-\delta K_U\,u@f$ of the moduli perturbation on
 *    the forward displacement (ModuliBornForce),
 *  - the branch force @f$\sum_k B^T (2\delta\mu_k) m_k@f$ of the
 *    forward internal variables,
 *  - the relaxation-time source @f$s_k = \delta h_k\,(-e_k m_k^n + a_k'
 *    d^n + b_k' d^{n+1})@f$, @f$\delta h_k = dt\,\delta(1/\tau_k)@f$,
 *    @f$\delta(1/\tau_k) = (1/\tau_k)(\delta\mu_k/\mu_k -
 *    \delta\eta_k/\eta_k)@f$, which enters the effective solve of the
 *    step as @f$\sum_k B^T C_k s_k@f$ and the new state as
 *    @f$\delta m_k^{n+1} \mathrel{+}= s_k@f$.
 *
 * The forward trajectory must come from ExponentialTrapezoidStep on a
 * LINEAR isotropic rheology (the adjoint's assumptions); the twin shares
 * the forward problem's spaces and rheology and carries the load
 * direction as its own load. The operator owns the force hook.
 */
class TangentViscoelasticOperator : public ViscoelasticOperator {
 public:
  /** @param twin The tangent problem (zero load, or the load direction).
   * @param forward The forward operator: its internal order and strain
   * map are adopted. */
  TangentViscoelasticOperator(LinearQuasiStaticProblem& twin,
                              const ViscoelasticOperator& forward);

  void SetDirection(const TangentDirection& dir);

  /** @brief March the tangent problem along @p forward's grid and record
   * it in @p out; false if a solve fails. */
  bool Run(ForwardTrajectory& forward, TangentTrajectory& out);

 private:
  void Inject(mfem::real_t t);
  int NodeAt(mfem::real_t t) const;
  void RelaxationSource(int n, mfem::real_t dt);

  std::unique_ptr<ModuliBornForce> born_;
  std::vector<mfem::Vector> dmu_, ditau_;  ///< nodal, branch layout
  std::vector<bool> has_dmu_, has_ditau_;
  std::vector<mfem::Vector> s_;  ///< the relaxation source per branch
  ForwardTrajectory* fwd_ = nullptr;
  bool in_step_ = false;
  int target_ = 0;
  mfem::Vector fbuf_, dn_, dn1_, zbuf_;
};

/**
 * @brief The tangent-linear driver, the forward-mode twin of
 * AdjointSolver: a STATIC configuration (one solve of the twin with the
 * Born force of the forward displacement) and a QUASI-STATIC one
 * (TangentViscoelasticOperator on the forward trajectory). Pair()
 * evaluates the derivative of an observation functional given as an
 * AdjointForce: the point and continuous terms applied to the tangent
 * fields with the trapezoid weights of the forward grid, which is the
 * exact derivative of the functional the forward run samples.
 */
class TangentLinearSolver {
 public:
  /** @brief Static: @p twin is the zero-load (or load-direction) twin,
   * @p forward_displacement the solved forward field. */
  TangentLinearSolver(LinearQuasiStaticProblem& twin,
                      const mfem::GridFunction& forward_displacement);
  /** @brief Quasi-static: the tangent operator is built on @p twin with
   * the layout of @p forward's operator. */
  TangentLinearSolver(LinearQuasiStaticProblem& twin,
                      ForwardTrajectory& forward);

  void SetDirection(const TangentDirection& dir) { dir_ = dir; }
  bool Run();

  /** @brief The tangent displacement: the twin's (static), or the last
   * node's (evolution). */
  const mfem::GridFunction& Displacement() const;
  const TangentTrajectory& Trajectory() const { return traj_; }
  TangentViscoelasticOperator* Operator() { return op_.get(); }

  /** @brief dJ along the direction for the functional @p force describes
   * (global in parallel). Border terms (uniform, omega) are refused. */
  mfem::real_t Pair(const AdjointForce& force) const;

 private:
  mfem::real_t Dot(const mfem::Vector& dual, const mfem::GridFunction& u,
                   mfem::FiniteElementSpace& fes) const;

  LinearQuasiStaticProblem& twin_;
  const mfem::GridFunction* u_forward_ = nullptr;
  ForwardTrajectory* forward_ = nullptr;
  std::unique_ptr<TangentViscoelasticOperator> op_;
  TangentDirection dir_;
  TangentTrajectory traj_;
};

}  // namespace AdGIA
