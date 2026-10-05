/**
 * @file viscoelastic.hpp
 * @brief Quasi-static generalised Maxwell viscoelasticity as a
 * TimeDependentOperator on top of a LinearQuasiStaticProblem, with
 * explicit, implicit and exponential time stepping.
 */

#pragma once

#include <memory>
#include <vector>

#include "mfem.hpp"
#include "AdGIA/quasi_static_problem.hpp"

namespace AdGIA {

namespace detail {

/**
 * @brief Weights of the exponential trapezoid step for @f$h = dt/\tau@f$:
 * @f$m \leftarrow e^{-h} m + \alpha\, d^n + \beta\, d^{n+1}@f$ with
 * @f$\alpha = (1-e^{-h})/h - e^{-h}@f$ and @f$\beta = 1 - (1-e^{-h})/h@f$
 * (series for small @p h).
 */
void ExponentialTrapezoidWeights(mfem::real_t h, mfem::real_t& e,
                                 mfem::real_t& alpha, mfem::real_t& beta);

}  // namespace detail

/**
 * @brief Generalised Maxwell viscoelastic evolution operator.
 *
 * For every Prony branch @f$k@f$ of the quasi-static problem's rheology, a
 * symmetric tensor internal variable @f$m_k@f$ lives on a discontinuous
 * (L2) nodal space on the displacement mesh: trace-free when the rheology
 * is isotropic (Rheology::TraceFreeInternalVariables()), a full symmetric
 * tensor for an anisotropic one. The ODE state is the concatenation of the
 * @f$m_k@f$; the displacement is slaved to it through the quasi-static
 * solve
 * @f[
 *   K_U u = f_{ext}(t) + \sum_k B^T (C_k m_k), \qquad
 *   \dot m_k = (D u - m_k)/\tau_k ,
 * @f]
 * with @f$B_{IJ} = \int \Phi_I : \varepsilon(\phi_J)@f$ (deviatoric
 * strain in the trace-free case) assembled once with a unit coefficient,
 * all material weighting applied pointwise at the internal-variable nodes
 * (@f$C_k m_k = 2\mu_k m_k@f$ for the isotropic body, a sampled Mandel
 * tensor otherwise), the rows of @f$B@f$ restricted per branch to the
 * nodes the branch lives on (Rheology::BranchMarker(): a branch of a
 * CompositeRheology exists on its region only, and its state block holds
 * just those nodes), and @f$D@f$ the strain map @f$u \mapsto
 * \varepsilon(u)@f$ (or its deviator) at those nodes, either the Galerkin
 * projection (default; adjoint-consistent) or nodal interpolation. The
 * projection is @f$(G^{-1} \otimes M^{-1}) B@f$ with @f$M@f$ the scalar L2
 * mass matrix and @f$G_{cc'} = E_c : E_{c'}@f$ the Frobenius metric of the
 * basis tensors @f$E_c@f$ (not orthonormal: the off-diagonal ones have
 * norm 2, and in the trace-free basis the diagonal ones overlap).
 *
 * **State-dependent relaxation times.** A branch may carry a RelaxationLaw
 * (Rheology::Law(k)), @f$\tau_k = \tau_{k0} F_k(\varepsilon, \sigma, m_k)@f$
 * evaluated at the internal nodes from the current state; the elastic part
 * stays linear and the global structure is unchanged. Explicit stepping and
 * ETD1 use the times of the current state (ETD1: frozen over the step,
 * first order). The exponential trapezoid and the implicit stages use a
 * predictor–corrector: the effective modulus is built with the times of
 * the start state (trapezoid) or the last evaluated times (implicit
 * stage), the step is taken, the times are re-evaluated at the midpoint
 * (trapezoid) or end (backward Euler) state and the step repeated, up to
 * SetCorrectorIterations() times or until the times stop changing. Each
 * pass is one elastic solve with a different effective operator (see
 * LinearQuasiStaticProblemBase::SetPreconditionerReuse()). For a linear
 * rheology (Rheology::IsLinear()) the times are fixed: they are never
 * re-evaluated and no corrector pass is taken.
 *
 * Time stepping:
 *  - Mult(): the explicit right-hand side, for any explicit MFEM ODESolver
 *    (stable only for @f$dt \lesssim 2.8\,\tau_{min}@f$ with RK4).
 *  - ExponentialEulerStep(): first order, exact relaxation with the strain
 *    frozen; one solve per step; no step restriction.
 *  - ExponentialTrapezoidStep(): second order, exact for a strain varying
 *    linearly in time; one solve per step with an effective modulus; no
 *    step restriction. The recommended workhorse.
 *  - ImplicitSolve(): backward Euler / SDIRK stages through MFEM's implicit
 *    ODESolvers, also via an effective modulus.
 * The last two need the quasi-static problem to support SetRelaxationWeights()
 * (the effective modulus is @f$C_\infty + \sum_k \beta_k C_k@f$ with nodal
 * weights @f$\beta_k@f$); the operator switches the problem between its
 * unrelaxed and effective operators lazily, so mixing schemes is allowed
 * but costs a reassembly at each switch. doc/viscoelasticity.md, "Time
 * stepping" and "Choosing a scheme", tabulates the schemes and when to use
 * each.
 *
 * Observation: after a step the displacement held by the quasi-static problem
 * is consistent with the new state for the trapezoid and backward-Euler
 * schemes; after an SDIRK step it belongs to the last stage state, and after
 * the explicit or exponential-Euler ones it is stale. SolveElastic() gives a
 * consistent displacement for any (m, t), re-solving only when needed;
 * SyncFields() copies the state into the registered output GridFunctions.
 *
 * The internal-variable spaces use Ordering::byNODES, so component @f$c@f$
 * of a branch occupies @f$[c\,n_d, (c+1)\,n_d)@f$, with the component
 * convention of TraceFreeSymmetricMatrixIndex / SymmetricMatrixIndex. Works on
 * serial and parallel problems alike (all operations are element-local; the
 * forces are L-vectors, which is what AddForce expects).
 */
class ViscoelasticOperator : public mfem::TimeDependentOperator {
 public:
  enum class StrainMap { Galerkin, Interpolation };

  /**
   * @param problem The quasi-static problem; must outlive this operator.
   * @param internal_order Polynomial order of the L2 internal-variable
   * spaces; < 0 means the smallest order resolving eps(u) exactly:
   * displacement order - 1 on simplices, the displacement order on
   * tensor-product elements. With an exactly resolving order the Galerkin
   * strain map is the identity on representable strains, the two strain
   * maps coincide, and the effective-modulus elimination is exact.
   * @param map The strain map u -> d at the internal-variable nodes.
   */
  ViscoelasticOperator(LinearQuasiStaticProblem& problem,
                       int internal_order = -1,
                       StrainMap map = StrainMap::Galerkin);

  // --- state layout ---------------------------------------------------------

  int NumBranches() const { return static_cast<int>(itau_.size()); }
  /** @brief Block offsets of the state, one block per branch. */
  const mfem::Array<int>& Offsets() const { return offsets_; }
  int BranchOffset(int k) const { return offsets_[k]; }
  /** @brief Size of branch @p k's state block: NumComponents() times its
   * active nodes, laid out component-major (c * NumBranchNodes(k) + q). */
  int BranchSize(int k) const { return offsets_[k + 1] - offsets_[k]; }
  /** @brief The scalar dofs of InternalScalarSpace() branch @p k lives on:
   * all of them unless the rheology restricts the branch to a region. */
  const mfem::Array<int>& BranchNodes(int k) const { return nodes_[k]; }
  int NumBranchNodes(int k) const { return nodes_[k].Size(); }
  /** @brief Tensor components of the internal variables (n_s - 1
   * trace-free, n_s full). */
  int NumComponents() const { return nc_; }
  bool TraceFree() const { return tracefree_; }
  /** @brief Aliasing view (no copy) of branch @p k in @p m. */
  mfem::Vector Branch(const mfem::Vector& m, int k) const;
  /** @brief Scatter branch @p k of @p m into the layout of
   * InternalVariableSpace() (zero off its nodes). */
  void BranchToFull(const mfem::Vector& m, int k, mfem::Vector& full) const;

  mfem::FiniteElementSpace& InternalVariableSpace() { return *dfes_; }
  mfem::FiniteElementSpace& InternalScalarSpace() { return *sfes_; }
  /** @brief Output view of branch @p k; see SyncFields(). */
  const mfem::GridFunction& InternalVariable(int k) const { return *m_out_[k]; }
  StrainMap Map() const { return map_; }
  LinearQuasiStaticProblem& Problem() { return problem_; }

  // --- ODE interface --------------------------------------------------------

  /** @brief Explicit right-hand side at GetTime(): one elastic solve with the
   * unrelaxed operator, then the pointwise rates. */
  void Mult(const mfem::Vector& m, mfem::Vector& k) const override;

  /** @brief Backward-Euler stage: k such that k = f(m + dt k, GetTime()),
   * via the effective modulus C_inf + sum_k C_k/(1 + dt/tau_k). */
  void ImplicitSolve(mfem::real_t dt, const mfem::Vector& m,
                     mfem::Vector& k) override;

  /** @brief One exponential Euler (ETD1) step: solve at (m, t), freeze the
   * strain, relax exactly. */
  void ExponentialEulerStep(mfem::Vector& m, mfem::real_t& t, mfem::real_t dt);

  /** @brief One exponential trapezoid step (second order, implicit in the
   * new strain through the effective modulus; predictor–corrector on the
   * relaxation times when they are state-dependent). Also forms the ETD1
   * prediction of the same step for ErrorEstimate(). */
  void ExponentialTrapezoidStep(mfem::Vector& m, mfem::real_t& t,
                                mfem::real_t dt);

  /**
   * @brief Local error estimate of the last ExponentialTrapezoidStep(): the
   * RMS over the state of @f$(m - \hat m) / (atol + rtol\,\max(|m^n|,
   * |m^{n+1}|))@f$ with @f$\hat m@f$ the first-order ETD1 prediction (global
   * in parallel). Below one means the step met the tolerances.
   */
  mfem::real_t ErrorEstimate(mfem::real_t rtol, mfem::real_t atol) const;

  // --- nonlinear controls ---------------------------------------------------

  /** @brief Corrector passes of the trapezoid and implicit steps for
   * state-dependent relaxation times (default 1), stopping early when the
   * relative change of the nodal times falls below @p tol. */
  void SetCorrectorIterations(int max_passes, mfem::real_t tol = 1e-3) {
    max_corrector_ = max_passes;
    corrector_tol_ = tol;
  }
  int LastCorrectorPasses() const { return last_passes_; }

  /** @brief True if the rheology is linear (no state-dependent law). */
  bool IsLinear() const { return linear_; }

  /** @brief Re-evaluate the relaxation times at the state (@p d strain,
   * @p m full state vector); no-op for a linear rheology. */
  void EvaluateRelaxationTimes(const mfem::Vector& d,
                               const mfem::Vector& m) const;

  /** @brief Current nodal @f$1/\tau_k@f$ (state-dependent laws: as of the
   * last evaluation). */
  const mfem::Vector& InverseRelaxationTimes(int k) const { return itau_[k]; }

  // --- observation ----------------------------------------------------------

  /** @brief Make the problem's displacement consistent with (m, t) using the
   * unrelaxed operator; skips the solve when it already is. */
  bool SolveElastic(const mfem::Vector& m, mfem::real_t t);

  /** @brief Forget that the problem's displacement is consistent with the
   * last (m, t). Needed when the external load jumps at a step boundary:
   * a step ending at the jump is taken with the load's left limit, the
   * next one must start from the right limit, and the cache keyed on
   * (m, t) cannot tell the two apart. */
  void InvalidateDisplacement() { cache_valid_ = false; }

  /** @brief Copy the state into the output GridFunctions. */
  void SyncFields(const mfem::Vector& m);

  /** @brief Register the problem's fields and the internal variables. */
  void RegisterFields(mfem::DataCollection& dc);

  /** @brief Smallest relaxation time over all nodes and branches (global in
   * parallel); the explicit stability limit is a small multiple of it. */
  mfem::real_t MinRelaxationTime() const;

  /** @brief The strain map: d = D u at the internal nodes. */
  void ComputeStrain(const mfem::GridFunction& u, mfem::Vector& d) const;

  /** @brief The geometric coupling form B. */
  const mfem::MixedBilinearForm& CouplingForm() const { return *B_; }

  /** @brief Restore the problem's unrelaxed operator (done lazily anyway). */
  void UseUnrelaxedOperator() const;

 protected:
  enum class Scheme { None, BackwardEuler, ExponentialTrapezoid };

  /** @brief Pointwise rate of branch @p k: (d - m_k) / tau_k. */
  virtual void Rate(int k, const mfem::Vector& m_k, const mfem::Vector& d,
                    mfem::Vector& k_out) const;

  /** @brief Exact relaxation of branch @p k over @p dt with @p d frozen. */
  virtual void LocalExponentialUpdate(int k, mfem::real_t dt, mfem::Vector& m_k,
                                      const mfem::Vector& d) const;

  /** @brief y = C_k x pointwise at the internal nodes (2 mu_k x for the
   * isotropic body, the sampled tensor otherwise). */
  void ApplyBranchModulus(int k, const mfem::Vector& x, mfem::Vector& y) const;

  /** @brief Push B_k^T zeta (zeta in branch @p k's layout) into the
   * problem. */
  void AddCoupledForce(int k, const mfem::Vector& zeta) const;

  /** @brief AssembleForce(t), the branch forces sum_k B^T(C_k m_k), and
   * Solve(). */
  bool ElasticUpdate(const mfem::Vector& m, mfem::real_t t) const;

  /** @brief d_ = D u for the problem's current displacement. */
  void ComputeCurrentStrain() const;

  /** @brief Set the relaxation weights for (dt, scheme), reassembling only
   * when they change. */
  void SetEffectiveModulus(mfem::real_t dt, Scheme scheme) const;

  bool CacheMatches(const mfem::Vector& m, mfem::real_t t) const;
  void UpdateCache(const mfem::Vector& m, mfem::real_t t) const;

  /** @brief Max over nodes and branches of the relative change of 1/tau
   * between @p old_itau and the current values (global). */
  mfem::real_t RelaxationTimeChange(
      const std::vector<mfem::Vector>& old_itau) const;
  mfem::real_t GlobalMax(mfem::real_t v) const;

  LinearQuasiStaticProblem& problem_;
  StrainMap map_;

  // Internal-variable discretisation.
  mfem::FiniteElementSpace* ufes_ = nullptr;
  std::unique_ptr<mfem::FiniteElementCollection> fec_;
  std::unique_ptr<mfem::FiniteElementSpace> dfes_;  ///< tensor space
  std::unique_ptr<mfem::FiniteElementSpace> sfes_;  ///< scalar companion
  std::unique_ptr<mfem::MixedBilinearForm> B_;      ///< geometric coupling
  std::unique_ptr<mfem::DiscreteLinearOperator> D_interp_;
  std::unique_ptr<mfem::SparseMatrix> Minv_;  ///< block-diagonal L2 M^{-1}
  mfem::DenseMatrix Ginv_;  ///< inverse Frobenius metric of the basis tensors
  bool tracefree_ = true;   ///< trace-free (isotropic) internal variables
  int nd_ = 0;              ///< scalar dofs
  int nc_ = 0;              ///< tensor components
  int ns_ = 0;              ///< d(d+1)/2
  mfem::Array<int> offsets_;
  // Per branch: its active scalar nodes, the inverse map (node -> position
  // in the block, -1 off the branch), and the rows of B for its nodes
  // (B_ itself for a whole-mesh branch). Nodal material data below is
  // stored per branch at its active nodes, in block order.
  std::vector<mfem::Array<int>> nodes_, slot_;
  std::vector<const mfem::SparseMatrix*> Bk_;
  std::vector<std::unique_ptr<mfem::SparseMatrix>> Bk_owned_;

  // Material data sampled at the internal nodes.
  /// Isotropic: nodal 2 mu_k. Anisotropic: per node the n_s x n_s matrix
  /// W with sigma_s = sum_t W_st m_t on unscaled tensor components
  /// (row-major, node-major: W(p)(s,t) at p ns ns + s ns + t).
  std::vector<mfem::Vector> branch_modulus_;
  mfem::Vector CU_;  ///< nodal W-form of C_U (for the stress of state-dependent laws)
  std::vector<mfem::Vector> itau0_;         ///< nodal 1 / tau_k0
  mutable std::vector<mfem::Vector> itau_;  ///< current nodal 1 / tau_k
  bool linear_ = true;                      ///< no state-dependent law
  std::vector<const RelaxationLaw*> law_;   ///< per branch, may be null
  std::vector<mfem::Vector> law_params_;    ///< per branch, node-major
  std::vector<int> num_params_;
  std::vector<std::unique_ptr<mfem::GridFunction>> beta_;  ///< nodal weights
  std::vector<std::unique_ptr<mfem::GridFunctionCoefficient>> beta_coef_;
  std::vector<mfem::Coefficient*> beta_ptrs_;
  std::vector<std::unique_ptr<mfem::GridFunction>> m_out_;
  mutable mfem::Vector d_, d_prev_, dual_, force_, zeta_;  ///< scratch

  mutable Scheme scheme_ = Scheme::None;
  mutable mfem::real_t effective_dt_ = -1.0;
  mutable long effective_version_ = -1;  ///< tau version the weights hold
  mutable long tau_version_ = 0;         ///< bumped by EvaluateRelaxationTimes
  int max_corrector_ = 1;
  mfem::real_t corrector_tol_ = 1e-3;
  int last_passes_ = 0;

  // ETD1 predictor of the last trapezoid step, for the error estimate.
  mfem::Vector predictor_diff_, m_scale_;

  // (m, t) for which d_ and the problem's displacement are consistent, if
  // cache_valid_.
  mutable mfem::Vector cached_m_;
  mutable mfem::real_t cached_t_ = 0.0;
  mutable bool cache_valid_ = false;

  bool parallel_ = false;
#ifdef MFEM_USE_MPI
  MPI_Comm comm_ = MPI_COMM_NULL;
#endif
};

/**
 * @brief ODESolver adaptor for ViscoelasticOperator::ExponentialEulerStep().
 */
class ExponentialEulerSolver : public mfem::ODESolver {
 public:
  void Init(mfem::TimeDependentOperator& f) override;
  void Step(mfem::Vector& x, mfem::real_t& t, mfem::real_t& dt) override;

 private:
  ViscoelasticOperator* op_ = nullptr;
};

/**
 * @brief ODESolver adaptor for
 * ViscoelasticOperator::ExponentialTrapezoidStep().
 */
class ExponentialTrapezoidSolver : public mfem::ODESolver {
 public:
  void Init(mfem::TimeDependentOperator& f) override;
  void Step(mfem::Vector& x, mfem::real_t& t, mfem::real_t& dt) override;

 private:
  ViscoelasticOperator* op_ = nullptr;
};

/**
 * @brief Adaptive exponential trapezoid stepping: the embedded ETD1 /
 * trapezoid pair gives a free local error estimate
 * (ViscoelasticOperator::ErrorEstimate()), and a standard controller
 * @f$dt \leftarrow dt\,\min(grow, \max(shrink, safety\,err^{-1/2}))@f$
 * shrinks and retries a step whose estimate exceeds one. Serves linear
 * bodies (where the effective operator depends on dt) and state-dependent
 * relaxation times alike; there is no stability limit, only accuracy.
 * The estimate is the local error of the first-order companion, so the
 * tolerance is conservative for the second-order solution that is
 * propagated (by a factor of order @f$\tau/dt@f$); choose rtol
 * accordingly.
 *
 * Step(x, t, dt) takes a step of at most @p dt (never below the lower bound)
 * and returns in @p dt the step proposed next; Integrate() runs to a final
 * time, hitting it exactly.
 */
class AdaptiveExponentialTrapezoidSolver : public mfem::ODESolver {
 public:
  void Init(mfem::TimeDependentOperator& f) override;
  void Step(mfem::Vector& x, mfem::real_t& t, mfem::real_t& dt) override;

  /** @brief Integrate from @p t to @p t_final with the current step @p dt
   * (updated on return); returns the number of accepted steps. */
  int Integrate(mfem::Vector& x, mfem::real_t& t, mfem::real_t t_final,
                mfem::real_t& dt);

  void SetTolerances(mfem::real_t rtol, mfem::real_t atol) {
    rtol_ = rtol;
    atol_ = atol;
  }
  void SetStepBounds(mfem::real_t dt_min, mfem::real_t dt_max) {
    dt_min_ = dt_min;
    dt_max_ = dt_max;
  }
  void SetStepFactors(mfem::real_t shrink, mfem::real_t grow,
                      mfem::real_t safety) {
    shrink_ = shrink;
    grow_ = grow;
    safety_ = safety;
  }
  int NumAcceptedSteps() const { return accepted_; }
  int NumRejectedSteps() const { return rejected_; }
  mfem::real_t LastErrorEstimate() const { return last_err_; }

 private:
  ViscoelasticOperator* op_ = nullptr;
  mfem::real_t rtol_ = 1e-4, atol_ = 1e-10;
  mfem::real_t dt_min_ = 0.0, dt_max_ = mfem::infinity();
  mfem::real_t shrink_ = 0.2, grow_ = 4.0, safety_ = 0.9;
  int accepted_ = 0, rejected_ = 0;
  mfem::real_t last_err_ = 0.0;
  mfem::Vector trial_;
};

}  // namespace AdGIA
