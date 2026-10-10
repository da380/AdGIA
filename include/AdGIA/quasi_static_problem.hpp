/**
 * @file quasi_static_problem.hpp
 * @brief Linear quasi-static problems: the abstract interface used by the
 * viscoelastic layer, a base class owning the shared bookkeeping (serial and
 * parallel), and two reference problems (pure traction, clamped).
 *
 * A problem is the equilibrium of a body at a time @f$t@f$: geometry,
 * boundary conditions, loads and solver. Whether the body is elastic or
 * viscoelastic is decided by its Rheology; the problem assembles the
 * rheology's (effective) elastic stiffness and never sees the internal
 * variables, which the viscoelastic layer evolves around it.
 *
 * **Naming convention for problem classes.** Names are built from fixed
 * axis slots, left to right:
 * @verbatim
 *   [Linear|Nonlinear] [QuasiStatic|Dynamic]
 *     [Mixed|Referential]? [SelfGravitating]? [variant] Problem
 * @endverbatim
 * - Linearity and time regime always appear (Static needs no slot of its
 *   own: it is QuasiStatic at fixed data).
 * - The elasticity is *referential in every class*; the formulation slot
 *   says how the *gravity* is described, so it exists only for
 *   self-gravitating problems: `Mixed` = referential displacement with
 *   the spatial (Eulerian) potential perturbation (the Dahlen-style
 *   organisation, mixed_problem.hpp); `Referential` = fully referential,
 *   including the potential (referential_problem.hpp). Without gravity
 *   there is nothing to mix and the slot is omitted: the classes in this
 *   file are referential, trivially so when the body is unstressed.
 * - Properties of the *reference state* — hydrostatic vs non-hydrostatic
 *   equilibrium stress, and natural vs non-natural particle labels in the
 *   sense of Al-Attar & Crawford 2016 (natural: the label *is* the
 *   equilibrium position, @f$\varphi_e = \mathrm{id}@f$) — are carried by
 *   the Rheology, not by class names; a class that *requires* a special
 *   reference state (the mixed formulation requires hydrostatic +
 *   natural) says so in its documentation.
 * - The variant tail names a boundary-condition or interface
 *   specialisation (`Traction`, `Clamped`, `Slip`), and a derived class
 *   extends its parent's name rightward.
 */

#pragma once

#include <memory>
#include <vector>

#include "mfem.hpp"
#include "AdGIA/rheology.hpp"
#include "AdGIA/null_space.hpp"

namespace AdGIA {

class Diffeomorphism;

/**
 * @brief Options of the Maxwell relaxation (secular static) solve; see
 * LinearQuasiStaticProblemBase::SetFluid.
 *
 * The per-step effective fluid shear is @f$\mu_c/(1+\beta)@f$ with
 * @f$\beta = \Delta t/\tau@f$: @p dt_over_tau is the physical phase's
 * @f$\beta@f$, and after stagnation @f$\beta@f$ is multiplied by
 * @p escalate per step up to @p beta_max (the regulariser-continuation
 * endgame; @f$\beta \to \infty@f$ recovers the unregularised static
 * fluid operator, so the cap is a conditioning guard for iterative
 * solvers).
 */
struct MaxwellRelaxationOptions {
  /** @brief @f$\beta@f$ of the physical phase. The default 3 resolves
   * the plateau — the safe choice at @f$N^2 < 0@f$, where larger
   * values coarsen the plateau and degrade the returned band-level
   * answer (measured on the fc model: quality falls monotonically
   * from beta 3 to 100). On @f$N^2 \ge 0@f$ models 10–30 is
   * typically ~3x cheaper and converges with no escalation at all;
   * ~100 overshoots (the per-step shear starts too small for the
   * preconditioner). */
  mfem::real_t dt_over_tau = 3.0;
  mfem::real_t tol = 1e-6;  ///< relative solid-increment convergence stop
                            ///< (well below mesh error; tighten per need)
  mfem::real_t stag_ratio = 0.9;  ///< stagnation: delta > ratio * previous
  int min_steps = 4;    ///< steps before stagnation may fire
  int max_steps = 200;  ///< hard cap on relaxation steps
  mfem::real_t escalate = 10.0;  ///< beta multiplier per post-stagnation step
  mfem::real_t beta_max = 1e8;   ///< beta cap (per-step shear floor)
  /** @brief Escalation economy: once stagnated, a step whose linear
   * solve costs more than iter_budget times the first step's
   * iterations stops the escalation there (the best state is
   * returned); 0 disables. */
  mfem::real_t iter_budget = 20.0;
  /** @brief Inexact stepping: each step's linear solve runs at
   * relative tolerance inexact * (the previous solid increment),
   * clamped to [the problem's RelTol(), inexact_max] — a fixed-point
   * iteration is self-correcting, so solver digits beyond the next
   * increment are wasted. The returned state is always polished by
   * one full-tolerance solve, so the endpoint accuracy is the
   * problem's RelTol() either way. 0 runs every step at full
   * tolerance (the reproducible-path mode). */
  mfem::real_t inexact = 0.1;
  mfem::real_t inexact_max = 1e-3;  ///< loosest per-step tolerance
  /** @brief Anderson-mixing depth on the memory-displacement
   * fixed-point map (0: plain backward Euler). Mixing accelerates the
   * configurational cascade at the PHYSICAL beta — on a linear map it
   * behaves like GMRES on the fixed-point system, so it can converge
   * growing (Rayleigh–Taylor) content too — without driving the
   * per-step shear toward the conditioning floor. The window restarts
   * at every operator (beta) change. Measured at depth 4 (aw/fc):
   * ~40 % fewer steps everywhere; wall time -10-35 % in serial 2-D
   * and on the 3-D ball (347 vs 539 iterations at the default beta,
   * matching the beta0 = 30 economy without the N^2 < 0-unsafe
   * knob), with improved fc increment floors — but +70 % on a small
   * parallel 2-D case (the mixed updates degrade warm starts and
   * tighten the inexact tolerances sooner). Problem-dependent, so
   * OPT-IN; the solver-calibration harness is the place to decide it
   * per problem. Destroys the physical-time reading of the
   * trajectory, so plateau_mode ignores it and steps plainly. */
  int anderson = 0;
  bool plateau_mode = false;     ///< stop at the physical plateau instead
};

/** @brief The gauge penalty's form: Deviatoric (a fluid: dev-dev
 * shear) or Harmonic (a vacuum-extension field: full-gradient
 * @f$\epsilon\mu_g\nabla u:\nabla v@f$). */
enum class GaugePenalty { Deviatoric, Harmonic };

/** @brief Options of the gauge-penalty fluid treatment (see
 * LinearQuasiStaticProblemBase::SetFluid). */
struct GaugePenaltyOptions {
  mfem::real_t epsilon = 1e-2;  ///< penalty factor
  int refinements = 2;          ///< Tikhonov refinements per Solve()
  GaugePenalty form = GaugePenalty::Deviatoric;
};

/**
 * @brief The deviatoric fluid operator @f$\epsilon\mu Q@f$ — the one
 * engine under both fluid treatments and the vacuum-extension
 * regularisation. One assembled matrix, two readings: the
 * relabelling-gauge penalty of the gauge treatment
 * (doc/gauged_fluid.md), and the per-step effective shear
 * @f$\gamma\hat Q@f$ of the Maxwell step at
 * @f$\epsilon = 1/(1+\beta)@f$ (the structural identity of
 * doc/static_fluid_core.tex, "The Maxwell relaxation method").
 *
 * Owns the attribute marker, the scaled coefficients, the penalty
 * integrator (borrowed, not owned, by the problem's combined solver
 * form), the assembled @f$\epsilon\mu Q@f$ and the
 * unit-@f$\epsilon@f$ cache that turns an epsilon-only change into a
 * scaled sparse copy (Rescale) instead of a finite element assembly.
 * The problem class owns what sits on top: the composition
 * @f$A + \epsilon\mu Q@f$, the solvers, and the two drivers (the
 * Tikhonov refinement loop and the Maxwell relaxation).
 */
class FluidRegionOperator {
 public:
  /** @brief Configure for the marked element attributes (sized to
   * attributes.Max(); copied): the penalty integrator at scale
   * epsilon * mu, assembled covariantly through @p map for the
   * Deviatoric form when one is given (a non-identity map is refused
   * for Harmonic — gauge data is shared, not mapped). Replaces any
   * previous configuration; nothing is assembled yet. @p mu is not
   * owned and must outlive the operator. */
  void Configure(mfem::FiniteElementSpace& fes,
                 const mfem::Array<int>& marker, mfem::Coefficient& mu,
                 mfem::real_t epsilon, GaugePenalty form,
                 Diffeomorphism* map);
  void Clear();
  bool Active() const { return integ_ != nullptr; }

  /** @brief Change the scale (the coefficient value only; the caller
   * chooses Assemble or Rescale). */
  void SetEpsilon(mfem::real_t epsilon);
  /** @brief The current scale (0 when inactive). */
  mfem::real_t Epsilon() const;
  bool HasUnitCache() const { return unit_.Ptr() != nullptr; }

  /** @brief Fresh finite element assembly of @f$\epsilon\mu Q@f$ at
   * the current epsilon (unconstrained: the drivers zero essential
   * rows of their residuals instead), refreshing the unit cache. */
  void Assemble(mfem::FiniteElementSpace& fes);
  /** @brief Rebuild @f$\epsilon\mu Q@f$ from the unit cache by a
   * scaled sparse copy — the epsilon-only fast path. */
  void Rescale();

  /** @brief The assembled @f$\epsilon\mu Q@f$ on true dofs. */
  const mfem::OperatorHandle& Q() const { return Q_; }
  /** @brief The penalty integrator, for the problem's combined
   * solver form to borrow (the borrower owns nothing). */
  mfem::BilinearFormIntegrator* Integrator() { return integ_; }
  const mfem::Array<int>& Marker() const { return marker_; }
  /** @brief Non-const access for borrowers that store a marker
   * reference (mfem's AddDomainIntegrator); the marker lives as long
   * as the configuration. */
  mfem::Array<int>& Marker() { return marker_; }

  /** @brief @f$r = \epsilon\mu Q\,x@f$ (r sized by the caller). */
  void ApplyQ(const mfem::Vector& x, mfem::Vector& r) const;

 private:
  mfem::Array<int> marker_;
  std::unique_ptr<mfem::ConstantCoefficient> eps_;
  std::unique_ptr<mfem::ProductCoefficient> mu_eps_;
  // The mapped Deviatoric penalty's pulled-back tensor (lambda = -2mu/d).
  std::unique_ptr<mfem::Coefficient> lambda_eps_;
  std::unique_ptr<mfem::MatrixCoefficient> Cdev_;
  std::unique_ptr<mfem::BilinearForm> integrators_;  ///< owns integ_
  mfem::BilinearFormIntegrator* integ_ = nullptr;
  std::unique_ptr<mfem::BilinearForm> q_form_;
  mfem::OperatorHandle Q_, unit_;
  bool parallel_ = false;
};

/**
 * @brief Diagnostics of the last Maxwell relaxation solve.
 *
 * @c stop: "converged" (solid increment below tolerance), "stagnation"
 * (plateau_mode stop at the physical plateau), "solver_floor" (the
 * escalation hit the iterative solver's conditioning floor),
 * "beta_max" (the escalation stalled at its beta cap) or "max_steps".
 * On the last three the returned state is the BEST one seen — the
 * lowest solid increment, restored by one warm re-solve when the
 * escalation had overshot it — and @c delta_returned is its increment;
 * on "solver_floor"/"beta_max" that is typically the physical plateau,
 * the honest answer when the continuation cannot be completed.
 */
struct MaxwellRelaxationReport {
  int steps = 0;      ///< relaxation steps taken
  int operators = 0;  ///< distinct per-step operators (beta changes)
  int stag_step = -1;  ///< step where stagnation fired (-1: never)
  const char* stop = "none";
  mfem::real_t delta_returned = -1.0;  ///< increment of the state returned
                                       ///< (before the final polish solve)
  std::vector<mfem::real_t> t_over_tau;   ///< physical time per step
  std::vector<mfem::real_t> delta_solid;  ///< relative solid increment
  std::vector<long> iterations;  ///< linear-solver iterations per step
};

/**
 * @brief Abstract interface for linear quasi-static problems.
 *
 * Per evaluation time @f$t@f$ the protocol is
 * @code
 *   AssembleForce(t);        // all time-dependent data to t; external loads;
 *                            // increments cleared
 *   AddForce(f); ...         // superpose dual vectors on the displacement
 *   Solve();                 // displacement <- K^{-1}(external + increments)
 * @endcode
 *
 * - AssembleForce(t) is called at every stage of a time integrator, with
 *   possibly non-monotone t; it must be cheap and idempotent at fixed t.
 * - AddForce(f) takes the vdof (L-vector) layout of DisplacementSpace(),
 *   i.e. the layout of a LinearForm on that space before FormLinearSystem.
 *   In parallel the problem applies the prolongation transpose inside
 *   Solve(); callers never handle true dofs. AddForce accumulates.
 * - Solve() may be internally iterative or nonlinear, but is a black box to
 *   callers; linearity in the *forces* is part of the contract. It returns
 *   false if the linear solver did not converge.
 * - Problems carrying more unknowns than the displacement (a gravitational
 *   potential, say) keep them internal: the interface only ever refers to
 *   the displacement. There is one displacement field; several solid
 *   regions share it on a (possibly disconnected) SubMesh, and regional
 *   material differences are the rheology's business.
 *
 * Implicit and exponential-trapezoid viscoelastic stepping eliminate the
 * internal variables and need the stiffness reassembled with the
 * *effective* modulus @f$C_\infty + \sum_k \beta_k C_k@f$, with pointwise
 * relaxation weights @f$\beta_k@f$ (see ElasticStiffness); problems that
 * can do so advertise it through SupportsRelaxationWeights().
 */
class LinearQuasiStaticProblem {
 public:
  virtual ~LinearQuasiStaticProblem() = default;

  /** @brief The (vector) displacement space. */
  virtual mfem::FiniteElementSpace& DisplacementSpace() = 0;

  /** @brief Read-only access to the displacement. */
  virtual const mfem::GridFunction& Displacement() const = 0;

  /** @brief The rheology the operator was assembled with. */
  virtual const AdGIA::Rheology& Rheology() const = 0;

  /** @brief Bring all time-dependent data to time @p t and reset forcing. */
  virtual void AssembleForce(mfem::real_t t) = 0;

  /** @brief Superpose a dual vector (LinearForm layout) on the displacement. */
  virtual void AddForce(const mfem::Vector& f) = 0;

  /** @brief Solve for the displacement(s); false on solver failure. */
  virtual bool Solve() = 0;

  /** @brief Whether SetRelaxationWeights() is available. */
  virtual bool SupportsRelaxationWeights() const { return false; }

  /**
   * @brief Reassemble the stiffness with @f$C_\infty + \sum_k
   * \beta_k C_k@f$, one weight coefficient per branch of the rheology
   * (typically nodal fields on the internal-variable mesh). The problem
   * must invalidate its solver setup and reassemble on every call (the same
   * coefficient objects may carry new values). The coefficients must outlive
   * the next call to SetRelaxationWeights() or ClearRelaxationWeights().
   */
  virtual void SetRelaxationWeights(
      const std::vector<mfem::Coefficient*>& /*beta*/) {
    MFEM_ABORT("relaxation weights not supported by this problem");
  }

  /** @brief Restore the unrelaxed modulus @f$C_U@f$. */
  virtual void ClearRelaxationWeights() {}

  /** @brief Register output fields with a DataCollection. */
  virtual void RegisterFields(mfem::DataCollection& dc) = 0;
};

/**
 * @brief Base class implementing the interface on a serial or parallel
 * displacement space.
 *
 * The stiffness integrators come from the rheology's ElasticStiffness
 * (two split mfem::ElasticityIntegrators for the isotropic body, one
 * ElasticTensorIntegrator for an anisotropic one), assembled with the
 * unrelaxed modulus or, after SetRelaxationWeights(), the effective one.
 * The operator is (re)assembled lazily in Solve() whenever it is out of
 * date.
 *
 * Serial and parallel are handled in one class: the space decides. Forms,
 * grid function and system matrix are created through their parallel
 * variants when the space is a ParFiniteElementSpace.
 *
 * A derived class:
 *  1. adds loads to ExternalLoad() (and registers time-dependent
 *     coefficients with RegisterTimeDependent) in its constructor;
 *  2. optionally calls SetEssentialBoundary() and overrides
 *     UpdateBoundaryValues();
 *  3. optionally adds further integrators to StiffnessIntegrators();
 *  4. optionally overrides SetupSolver()/SolveLinearSystem() (the defaults
 *     are preconditioned CG with Gauss-Seidel or BoomerAMG).
 */
class LinearQuasiStaticProblemBase : public LinearQuasiStaticProblem {
 public:
  /**
   * @param fes Displacement space (vdim = space dimension); serial or
   * parallel; not owned.
   * @param rheology The material; not owned, must outlive the problem.
   */
  LinearQuasiStaticProblemBase(mfem::FiniteElementSpace* fes,
                               const AdGIA::Rheology& rheology);

  mfem::FiniteElementSpace& DisplacementSpace() override { return *fes_; }
  const mfem::GridFunction& Displacement() const override { return *u_; }
  const AdGIA::Rheology& Rheology() const override {
    return *rheology_;
  }

  void AssembleForce(mfem::real_t t) override;
  void AddForce(const mfem::Vector& f) override;
  bool Solve() override;

  bool SupportsRelaxationWeights() const override { return true; }
  void SetRelaxationWeights(
      const std::vector<mfem::Coefficient*>& beta) override;
  void ClearRelaxationWeights() override;

  void RegisterFields(mfem::DataCollection& dc) override;

  /** @brief True if the displacement space is a ParFiniteElementSpace. */
  bool IsParallel() const;

  /** @brief Time of the most recent AssembleForce(). */
  mfem::real_t Time() const { return t_; }

  /** @brief The external load; add integrators here (before the first
   * AssembleForce). */
  mfem::LinearForm& ExternalLoad() { return *b_; }

  /** @brief Integrators of the stiffness; add further ones here (before the
   * first Solve). The rheology's integrators are already present. */
  mfem::BilinearForm& StiffnessIntegrators() { return *integrators_; }

  /** @brief The rheology's stiffness object of this problem (its relaxation
   * state). */
  const ElasticStiffness& Stiffness() const { return *stiffness_; }

  /** @brief Register a coefficient whose SetTime() AssembleForce() calls. */
  void RegisterTimeDependent(mfem::Coefficient& c) { td_coefs_.push_back(&c); }
  void RegisterTimeDependent(mfem::VectorCoefficient& c) {
    td_vcoefs_.push_back(&c);
  }

  // --- fluid regions --------------------------------------------------------
  //
  // One engine, two treatments. Both treatments of an inviscid fluid
  // region run on the same deviatoric fluid operator eps mu Q (the
  // rheology supplies the fluid's bulk modulus; the shear term below
  // is artificial), read two ways:
  //   - MAXWELL RELAXATION (SetFluid with MaxwellRelaxationOptions,
  //     the DEFAULT and the general method): eps mu Q is the per-step
  //     effective shear of an artificial Maxwell solid, and Solve()
  //     relaxes the Heaviside-loaded system to the secular static
  //     response (doc/static_fluid_core.tex, "The Maxwell relaxation
  //     method"). Honest at any stratification, any geometry, any
  //     number of fluid regions.
  //   - GAUGE PENALTY (SetFluid with GaugePenaltyOptions): eps mu Q is
  //     a relabelling-gauge fixing term, removed by iterated Tikhonov
  //     refinement (doc/gauged_fluid.md). The cheaper specialist for
  //     neutral or near-neutral models with a validated epsilon
  //     window; also the Maxwell treatment's own per-step engine (the
  //     refinement iteration is exactly the beta -> infinity limit of
  //     the Maxwell step).

  /** @brief The penalty form, at namespace scope since the engine
   * moved there; this alias keeps the historical nested spelling. */
  using GaugePenalty = AdGIA::GaugePenalty;

  /** @brief The penalty options, at namespace scope like the enum;
   * this alias keeps the nested spelling. */
  using GaugePenaltyOptions = AdGIA::GaugePenaltyOptions;

  /**
   * @brief Declare the marked element attributes an inviscid fluid
   * region, treated by MAXWELL RELAXATION — the default general
   * method. The rheology supplies the fluid's physical stiffness (its
   * bulk modulus, with zero shear); this call makes the fluid an
   * artificial Maxwell solid and Solve() the SECULAR relaxation: the
   * load is applied as a Heaviside step and the quasi-static Maxwell
   * system is stepped by backward Euler until the SOLID region stops
   * moving — the secular static response, computed with nothing but
   * welded elastic solves (doc/static_fluid_core.tex, "The Maxwell
   * relaxation method"; honest at any stratification, on any
   * geometry, for any number of fluid regions).
   *
   * The problem is linear, so the Maxwell internal variable is the
   * deviatoric fluid strain of a memory displacement @f$w@f$, and one
   * backward Euler step is
   * @f[
   *   (A + \gamma\hat Q)\,u_{n+1} = f + \gamma\hat Q\,w_n, \qquad
   *   w_{n+1} = (w_n + \beta u_{n+1})/(1+\beta),
   * @f]
   * with @f$\beta = \Delta t/\tau@f$, @f$\gamma = \mu_c/(1+\beta)@f$
   * and @f$\hat Q@f$ the unit-shear deviatoric fluid stiffness. The
   * fixed point is the @f$A@f$-solution — independent of @f$\mu_c@f$,
   * @f$\tau@f$ and the schedule — and the per-step operator is the
   * engine's @f$A + \epsilon\mu_c Q@f$ at @f$\epsilon = 1/(1+\beta)@f$
   * (the gauge treatment's refinement iteration is exactly the
   * @f$\beta \to \infty@f$ case; GaugeRefine()). The stepper runs the
   * physical @f$\beta@f$ until the solid increment stagnates (the
   * configurational cascade, and at @f$N^2 < 0@f$ Rayleigh–Taylor
   * growth), then escalates @f$\beta@f$ geometrically — backward
   * Euler is L-stable, so growing modes are damped: regulariser
   * continuation toward the static welded solve — guarded by the
   * best-state checkpoint and the increment-validated inexact solves
   * (MaxwellRelaxationOptions). The stopping metric is the SOLID
   * displacement increment only: the fluid displacement does not
   * converge at @f$N^2 \ne 0@f$ and must not be monitored.
   * Diagnostics in MaxwellReport(); Solve() returns false if a step's
   * linear solve failed with no fallback state.
   *
   * Replaces any previous fluid treatment. Incompatible with
   * SetGaugePreconditionerOnly(). Essential boundary conditions must
   * not touch the marked attributes.
   *
   * @param fluid_marker Element attributes of the fluid regions
   * (sized to attributes.Max(); copied).
   * @param mu_scale Artificial fluid shear scale @f$\mu_c@f$ (a
   * natural choice is a neighbouring solid's shear modulus or the
   * fluid's bulk modulus; the limit is exactly independent of it);
   * not owned, must outlive the problem.
   * @param opts Schedule and stopping options.
   * @param map Optional mapping for the covariant deviatoric form
   * (see the penalty overload); not owned. Derived classes may
   * default it (ConfigureFluidOperator).
   */
  void SetFluid(const mfem::Array<int>& fluid_marker,
                mfem::Coefficient& mu_scale,
                const MaxwellRelaxationOptions& opts = {},
                Diffeomorphism* map = nullptr);

  /**
   * @brief As above, but treated by the GAUGE PENALTY: the engine's
   * term is read as the gauge-fixing shear penalty
   * @f$\epsilon\,2\mu_g\,\mathrm{dev}\,\varepsilon(u):\mathrm{dev}\,
   * \varepsilon(u')@f$ of the relabelling formulation, on the *solver*
   * operator only, and Solve() removes the @f$O(\epsilon)@f$ bias from
   * the observables by iterated Tikhonov refinement (GaugeRefine();
   * doc/gauged_fluid.md, "Gauge fixing: penalty plus iterated
   * refinement"). The specialist treatment for neutral or
   * near-neutral models with a validated epsilon window; the physical
   * operator (SystemMatrix()) is unchanged, and the fluid displacement
   * is gauge (the solid displacement and fields derived from
   * @f$\mathrm{div}(\rho u)@f$ or interface normal displacements are
   * the observables).
   *
   * With @p map non-null (the identity included) the Deviatoric form
   * is assembled covariantly, as ElasticTensorIntegrator(C, map) with
   * @f$C@f$ the isotropic tensor of @f$\lambda = -2\epsilon\mu_g/d@f$,
   * @f$\mu = \epsilon\mu_g@f$ pulled back through the map, so that a
   * relabelled problem's penalty is the exact pull-back of the
   * unmapped one. The Harmonic form is vacuum-extension gauge data,
   * shared rather than mapped, and refuses a non-identity map.
   * Essential boundary conditions must not touch the marked
   * attributes.
   */
  void SetFluid(const mfem::Array<int>& fluid_marker,
                mfem::Coefficient& mu_scale,
                const GaugePenaltyOptions& opts,
                Diffeomorphism* map = nullptr);

  /** @brief Remove the fluid treatment and its engine. */
  void ClearFluid();

  bool HasMaxwellFluid() const {
    return fluid_treatment_ == FluidTreatment::Maxwell;
  }

  /** @brief Diagnostics of the last Maxwell relaxation Solve(). */
  const MaxwellRelaxationReport& MaxwellReport() const {
    return maxwell_report_;
  }

  bool HasGaugedFluid() const { return fluid_op_.Active(); }

  /** @brief Change @f$\epsilon@f$ (marks the operator stale). */
  void SetGaugeEpsilon(mfem::real_t epsilon);
  mfem::real_t GaugeEpsilon() const;

  /** @brief The solid-everywhere preconditioner experiment
   * (doc/planning/solvers.md, "Solid-everywhere preconditioner for the
   * clean gauged system"): the Krylov operator becomes the CLEAN
   * @f$A@f$ — no penalty in the operator — while the preconditioner is
   * built on @f$A + \epsilon_{\mathrm{prec}} Q@f$ with
   * @f$\epsilon_{\mathrm{prec}}@f$ of order one (the model made solid
   * everywhere). The gauge refinements are disabled: there is no
   * @f$O(\epsilon)@f$ bias to remove, and the regularisation moves into
   * the STOPPING RULE — MINRES ignores the exact gauge kernel (zero
   * right-hand side under an SPD preconditioner), but the near-gauge
   * modes keep eigenvalue @f$\lambda \sim h^p@f$, so the residual is
   * expected to plateau at the near-kernel content of the load and
   * drift semi-convergently beyond it; run at a matched tolerance.
   * Call after SetFluid(GaugePenaltyOptions); supported by the block-MINRES paths
   * only. (The slipping class overrides it: there the fluid gauge is
   * SetFluidGauge(), the AL constraint penalty stays in the operator,
   * and the per-sweep Tikhonov term is dropped.) */
  virtual void SetGaugePreconditionerOnly(mfem::real_t eps_prec);
  bool GaugePreconditionerOnly() const { return gauge_prec_only_; }

  /** @brief Stagnation (plateau) stop for the SetGaugePreconditionerOnly
   * mode: the block solve runs as warm-started restarts of @p chunk
   * iterations and stops when the residual of a chunk exceeds @p ratio
   * times the previous chunk's — the plateau at the load's near-kernel
   * content, which IS the intended stopping point (early stopping is
   * the regulariser there; iterating past it semi-converges). With the
   * stop active one tight relative tolerance serves every load: where
   * no plateau exists the tolerance fires, where one does the
   * stagnation fires first. */
  void SetGaugePlateauStop(mfem::real_t ratio = 0.5, int chunk = 25);

  void SetGaugeRefinements(int n) { gauge_refinements_ = n; }
  int GaugeRefinements() const { return gauge_refinements_; }

  /** @brief Diagnostic: the assembled gauge penalty eps Q applied to a
   * displacement true-dof vector (zero without a gauged fluid). */
  void ApplyGaugePenalty(const mfem::Vector& u_true, mfem::Vector& r);

  /** @brief Norms of the physical residual @f$\|\epsilon Q\,\delta\|@f$ at
   * the start of each refinement step of the last Solve(); their decay is
   * the observed contraction factor. */
  const std::vector<mfem::real_t>& GaugeResiduals() const {
    return gauge_residuals_;
  }

  /** @brief The regularised matrix @f$A + \epsilon Q@f$ the solver runs on
   * (assembling if needed); equals SystemMatrix() without a gauged fluid. */
  const mfem::OperatorHandle& RegularizedMatrix();

  /** @brief Relative tolerance of the linear solves: each stops at
   * rel_tol times the preconditioned norm of its right-hand side (see
   * SetWarmStartTolerance()). */
  void SetRelTol(mfem::real_t rel_tol) { rel_tol_ = rel_tol; }
  mfem::real_t RelTol() const { return rel_tol_; }

  /** @brief A counter every operator (re)assembly or gauge rescale
   * bumps: caches derived from the assembled operator (border columns,
   * unit responses) are valid exactly while it stands still. */
  int OperatorVersion() const { return operator_version_; }

  /** @brief Print level of the default CG solver (quiet by default); takes
   * effect at the next operator assembly. */
  void SetPrintLevel(mfem::IterativeSolver::PrintLevel level) {
    print_level_ = level;
  }

  /**
   * @brief Keep the preconditioner across reassemblies of the stiffness
   * (a change of relaxation weights, e.g. every step under a variable dt or
   * state-dependent relaxation times) as long as the solver's iteration
   * count stays within @p factor of its count when the preconditioner was
   * built; then rebuild it. @p factor <= 1 rebuilds at every assembly.
   * Default 2: the BoomerAMG setup, the dominant cost of an assembly, is
   * then paid only when the operator has drifted far. The matrix itself is
   * always the current one.
   */
  void SetPreconditionerReuse(mfem::real_t factor) { prec_reuse_ = factor; }
  mfem::real_t PreconditionerReuse() const { return prec_reuse_; }

  /** @brief Number of preconditioner setups so far. */
  int NumPreconditionerSetups() const { return prec_setups_; }

  /** @brief Number of operator assemblies so far. */
  int NumAssemblies() const { return assemblies_; }

  /** @brief Number of Solve() calls so far. */
  int NumSolves() const { return solves_; }

  /** @brief Outer solver iterations accumulated over all solves. */
  long TotalIterations() const { return total_its_; }

  /** @brief The current (eliminated) system matrix, assembling if needed. */
  const mfem::OperatorHandle& SystemMatrix();

 protected:
  /** @brief Impose essential conditions on the marked boundary attributes
   * (all components). */
  void SetEssentialBoundary(const mfem::Array<int>& ess_bdr);

  /** @brief Refresh Dirichlet values in the solution at time @p t; called by
   * AssembleForce(). Default: nothing. */
  virtual void UpdateBoundaryValues(mfem::real_t /*t*/) {}

  /** @brief Build/refresh preconditioner and solver for a new operator.
   * Default: SetupDefaultCG(). */
  virtual void SetupSolver(mfem::OperatorHandle& A);

  /** @brief Solve A X = B with the solver from SetupSolver(); return
   * convergence. Default: warm-started preconditioned CG. */
  virtual bool SolveLinearSystem(const mfem::Vector& B, mfem::Vector& X);

  /** @brief Preconditioned CG (Gauss-Seidel serial, BoomerAMG with elasticity
   * options in parallel) in prec_/cg_, warm-started; the preconditioner is
   * reused per SetPreconditionerReuse(). Equivalent to
   * SetupDefaultPreconditioner(A) followed by SetupCG(*A.Ptr(), *prec_). */
  void SetupDefaultCG(mfem::OperatorHandle& A);

  /** @brief The default preconditioner on A in prec_, rebuilt or reused per
   * SetPreconditionerReuse(). */
  void SetupDefaultPreconditioner(mfem::OperatorHandle& A);

  /** @brief A fresh CG in cg_ on @p op preconditioned by @p prec, with the
   * problem's tolerance and print level, in iterative mode. The operator is
   * set before the preconditioner so that the latter is not reset onto it. */
  void SetupCG(const mfem::Operator& op, mfem::Solver& prec);

  /** @brief Record a solve's iteration count against the preconditioner's
   * baseline; marks it stale when the count has grown past the reuse
   * factor. Derived solvers call this after their outer solve. */
  void NoteIterations(int its);

  /**
   * @brief Make a warm-started solve converge to the same target as a cold
   * one: sets an absolute tolerance rel_tol * sqrt((M B, B)) in the
   * preconditioner norm, which is what a cold start would use. Returns false
   * when B vanishes (solution zero, no solve needed).
   */
  bool SetWarmStartTolerance(mfem::IterativeSolver& solver, mfem::Solver& prec,
                             const mfem::Vector& B) const;

  /** @brief Inner product, global in parallel. */
  mfem::real_t Dot(const mfem::Vector& x, const mfem::Vector& y) const;

  /** @brief The vector mass matrix @f$\int \rho\, u \cdot v@f$ on the
   * displacement space (unit weight when @p rho is null), as a true-dof
   * operator. Returns the form, which must outlive @p M. */
  std::unique_ptr<mfem::BilinearForm> AssembleMassOperator(
      mfem::Coefficient* rho, mfem::OperatorHandle& M);

  /** @brief (Re)assemble the stiffness and set up the solver. */
  void AssembleOperator();

  /**
   * @brief The engine's configuration hook, called by both SetFluid
   * overloads (and by internal users such as the vacuum extension and
   * the gauge-KKT mode). The base forwards to
   * FluidRegionOperator::Configure; a derived class customises — the
   * referential class defaults the Deviatoric map to its rheology's
   * equilibrium mapping, the slipping class refuses (its fluid has its
   * own space and SetFluidGauge()).
   */
  virtual void ConfigureFluidOperator(const mfem::Array<int>& marker,
                                      mfem::Coefficient& mu,
                                      mfem::real_t epsilon,
                                      GaugePenalty form,
                                      Diffeomorphism* map);

  /**
   * @brief The Tikhonov refinement loop of a penalty-treatment
   * Solve(): the @f$\beta = \infty@f$, fixed-@f$\epsilon@f$,
   * no-escalation case of the Maxwell step — after the first
   * regularised solve has put its solution in @p X, each refinement is
   * the full solve @f$(A + \epsilon Q)\,u_{k+1} = f + \epsilon Q\,
   * u_k@f$ through the virtual SolveLinearSystem(), so problems whose
   * solves carry further unknowns (a potential block) inherit it
   * unchanged, companion loads reapplied each time (equivalent, by
   * linearity, to the historical zero-companion-load increment
   * accumulation). Residuals @f$\|\epsilon Q\,\delta_k\|@f$ are
   * recorded as before (GaugeResiduals()).
   */
  bool GaugeRefine(mfem::Vector& X);

  /** @brief The Maxwell relaxation loop of Solve() (the Maxwell
   * treatment of SetFluid):
   * owns the whole solve, including the per-step operator updates and
   * the final RecoverFEMSolution. Runs through the virtual
   * SolveLinearSystem(), so problems carrying further unknowns (a
   * potential block) inherit it unchanged. */
  bool MaxwellSolve();

  /** @brief True dofs supported on NON-fluid elements (interface dofs
   * included) — the stopping metric's dofs — into
   * maxwell_solid_tdofs_. */
  void BuildMaxwellSolidDofs();

  /** @brief Global max of @f$|x_i|@f$ over maxwell_solid_tdofs_. */
  mfem::real_t MaxOverSolidDofs(const mfem::Vector& x) const;

  /** @brief Assemble the operator if it is out of date. When only the
   * gauge epsilon has changed since the last full assembly, the update
   * is the cheap RescaleGaugeOperator() path. */
  void EnsureOperator();

  /** @brief Fast operator update when only the gauge epsilon changed:
   * the physical @f$A@f$ is untouched, and @f$\epsilon Q@f$ and
   * @f$A + \epsilon Q@f$ are rebuilt from the cached unit-penalty
   * matrix by scaled sparse addition — no FEM reassembly. Ends with
   * SetupSolver() on the new operator. */
  void RescaleGaugeOperator();

  mfem::FiniteElementSpace* fes_;
#ifdef MFEM_USE_MPI
  mfem::ParFiniteElementSpace* pfes_ = nullptr;
#endif
  const AdGIA::Rheology* rheology_;
  std::unique_ptr<ElasticStiffness> stiffness_;

  std::unique_ptr<mfem::BilinearForm> integrators_;  ///< owns integrators
  std::unique_ptr<mfem::BilinearForm> a_;            ///< assembled stiffness
  std::unique_ptr<mfem::LinearForm> b_;              ///< external load
  std::unique_ptr<mfem::GridFunction> u_;            ///< displacement
  mfem::Vector increment_, rhs_;
  mfem::Array<int> ess_tdof_list_;
  mfem::OperatorHandle A_;
  mfem::Vector X_, B_;

  std::unique_ptr<mfem::Solver> prec_;
  std::unique_ptr<mfem::CGSolver> cg_;
  // Preconditioner reuse: the form and matrix the preconditioner was built
  // on are kept alive while it is reused.
  mfem::real_t prec_reuse_ = 2.0;
  bool prec_stale_ = true;
  int prec_baseline_its_ = -1;
  int prec_setups_ = 0;
  int assemblies_ = 0;
  int solves_ = 0;
  long total_its_ = 0;
  std::unique_ptr<mfem::BilinearForm> prec_form_;
  mfem::OperatorHandle prec_A_;

  // Fluid regions: the engine (the deviatoric fluid operator, with
  // its marker, coefficients, integrator and matrices) lives in
  // fluid_op_; the treatment slot says which driver Solve() runs on
  // top of it. The problem class keeps the composition A + eps mu Q
  // (a_solve_form_ / A_solve_) and the solvers.
  FluidRegionOperator fluid_op_;
  enum class FluidTreatment { None, Penalty, Maxwell };
  FluidTreatment fluid_treatment_ = FluidTreatment::None;
  int gauge_refinements_ = 2;
  bool gauge_prec_only_ = false;  // clean-A operator, A + eps_prec Q prec
  mfem::real_t gauge_plateau_ratio_ = 0.0;  // 0: no stagnation stop
  int gauge_plateau_chunk_ = 25;

  /** @brief The chunked solve of SetGaugePlateauStop: repeated
   * warm-started @p krylov restarts through @p outer until convergence
   * or the per-chunk residual ratio exceeds gauge_plateau_ratio_.
   * Returns the total iterations; @p converged reports tolerance OR
   * plateau (both are intended stops). Restores the solver's
   * iteration cap. */
  int PlateauMult(mfem::Solver& outer, mfem::IterativeSolver& krylov,
                  const mfem::Vector& B, mfem::Vector& X, bool& converged);
  std::unique_ptr<mfem::BilinearForm> a_solve_form_;
  mfem::OperatorHandle A_solve_;
  std::vector<mfem::real_t> gauge_residuals_;
  bool eps_only_dirty_ = false;

  /** @brief The engine's assembled @f$\epsilon\mu Q@f$ (for the
   * derived drivers' residuals and blocks). */
  const mfem::OperatorHandle& GaugeQ() const { return fluid_op_.Q(); }

  // Maxwell (secular) treatment state; see SetFluid.
  MaxwellRelaxationOptions maxwell_opts_;
  MaxwellRelaxationReport maxwell_report_;
  mfem::Array<int> maxwell_solid_tdofs_;

  /** @brief The epsilon-window tripwire shared by every GaugeRefine
   * implementation: the refinement corrections contract at
   * @f$\sim\epsilon/(\lambda+\epsilon)@f$, so a rate above 0.9 is the
   * semi-convergence signature (epsilon too small for this mesh/model)
   * and a rate above 0.2 leaves a bias @f$\sim\mathrm{rate}^k@f$ beyond
   * the refinement budget (epsilon too large, or too few refinements).
   * Warn only; the thresholds and the epsilon window are discussed in
   * doc/gauged_fluid.md, "Gauge fixing: penalty plus iterated
   * refinement". */
  void WarnGaugeContraction() const;

  mfem::real_t t_ = 0.0;
  mfem::real_t rel_tol_ = 1e-12;
  mfem::IterativeSolver::PrintLevel print_level_;
  bool operator_dirty_ = true;
  /** Bumped by every operator (re)assembly or rescale: anything derived
   * from the assembled operator (cached border columns, unit responses)
   * is valid exactly while this number stands still. */
  int operator_version_ = 0;
  std::vector<mfem::Coefficient*> td_coefs_;
  std::vector<mfem::VectorCoefficient*> td_vcoefs_;
};

/**
 * @brief Pure traction (Neumann) problem: a traction is applied on the marked
 * boundary attributes; no essential conditions.
 *
 * The stiffness retains the rigid-body null space. CG runs on @f$P A P@f$
 * with the preconditioner @f$P M P@f$, where @f$P@f$ is the Euclidean
 * projector orthogonal to the rigid modes (MakeRigidModeProjector()); the
 * load and the warm start are projected before the solve and the solution
 * after it, so any net force or torque is removed and the displacement is
 * orthogonal to the rigid modes in the true-dof inner product.
 *
 * **Reference state.** The class is reference-state aware through its
 * rheology: with a ReferentialElasticRheology the stiffness is the mapped
 * material + geometric split for the general @f$(\hat C, \mathbf{S}_e,
 * \varphi_e)@f$ (the non-gravitating referential problem), and the rigid
 * rotations of the projector are those of the *mapped* positions
 * @f$W\varphi_e@f$ (Rheology::EquilibriumMapping()); translations are
 * exact null modes either way, rotations null through moment balance of
 * the background state — RigidPairResiduals() verifies both. The
 * traction is per unit *referential* area; for a spatial traction on a
 * mapped boundary compose with NansonAreaCoefficient (mappings.hpp) —
 * composition is the problem layer's/driver's business, coefficients
 * stay referential.
 */
class LinearQuasiStaticTractionProblem : public LinearQuasiStaticProblemBase {
 public:
  /**
   * @param fes Displacement space; serial or parallel; not owned.
   * @param rheology The material; not owned, must outlive the problem.
   * @param traction Boundary traction; registered as time-dependent.
   * @param bdr_marker Boundary attributes it acts on (copied).
   */
  LinearQuasiStaticTractionProblem(mfem::FiniteElementSpace* fes,
                                   const AdGIA::Rheology& rheology,
                                   mfem::VectorCoefficient& traction,
                                   const mfem::Array<int>& bdr_marker);

  /**
   * @brief Fix the rigid gauge of the solution by zero net momentum and
   * angular momentum, @f$\int \rho\, u \cdot (a + b \times x) = 0@f$
   * (unit @f$\rho@f$ when null), instead of orthogonality to the rigid
   * modes in the true-dof inner product (the default). Only the rigid
   * component of the displacement changes. May be called at any time.
   * With a non-natural reference state the rotational condition reads
   * @f$b \times \varphi_e(x)@f$ (the projector's mapped modes) with the
   * *referential* density @f$\rho@f$ — the referential statement of zero
   * angular momentum, no extra Jacobian factor.
   */
  void SetMassWeightedGauge(mfem::Coefficient* rho = nullptr);
  /** @brief Back to the true-dof (Euclidean) gauge. */
  void SetEuclideanGauge();

  /**
   * @brief Diagnostic: @f$\|A n\| / (\|A\|_{\max}\|n\|)@f$ for each rigid
   * mode of the projector under the assembled stiffness (assembles if
   * needed). Translations are exact discrete null vectors (round-off);
   * with a pre-stressed/mapped reference state the mapped rotations are
   * near-null through moment balance, decreasing with refinement.
   */
  std::vector<mfem::real_t> RigidPairResiduals();

 protected:
  void SetupSolver(mfem::OperatorHandle& A) override;
  bool SolveLinearSystem(const mfem::Vector& B, mfem::Vector& X) override;

 private:
  /** @brief The rigid-mode projector (built on first use; the space does
   * not change). */
  const NullSpaceProjector& RigidModes();

  mfem::Array<int> marker_;
  std::unique_ptr<mfem::BilinearForm> gauge_form_;
  mfem::OperatorHandle gauge_M_;  ///< mass-weighted gauge, if set
  std::unique_ptr<NullSpaceProjector> projector_;
  std::unique_ptr<ProjectedOperator> projected_op_;    ///< P A P
  std::unique_ptr<ProjectedSolver> projected_prec_;    ///< P M P
  std::unique_ptr<ProjectedSolver> projected_;         ///< wraps cg_
};

/**
 * @brief Essential/natural problem: the displacement is prescribed on one
 * set of boundary attributes and a traction applied on another; all other
 * boundaries are traction-free. ("Mixed" in the boundary-condition sense
 * only — no relation to the mixed *formulation* of the self-gravitating
 * classes.)
 *
 * Reference-state aware exactly as LinearQuasiStaticTractionProblem (the
 * stiffness is the rheology's; prescribed values and tractions are
 * referential fields), with no null-space machinery to adapt: the
 * essential conditions remove the rigid modes.
 */
class LinearQuasiStaticClampedProblem : public LinearQuasiStaticProblemBase {
 public:
  /**
   * @param fes Displacement space; serial or parallel; not owned.
   * @param rheology The material; not owned, must outlive the problem.
   * @param ess_bdr Boundary attributes with prescribed displacement (copied).
   * @param traction Boundary traction; registered as time-dependent.
   * @param traction_marker Boundary attributes it acts on (copied).
   * @param dirichlet Prescribed displacement (registered as time-dependent);
   * nullptr means homogeneous.
   */
  LinearQuasiStaticClampedProblem(mfem::FiniteElementSpace* fes,
                                  const AdGIA::Rheology& rheology,
                                  const mfem::Array<int>& ess_bdr,
                                  mfem::VectorCoefficient& traction,
                                  const mfem::Array<int>& traction_marker,
                                  mfem::VectorCoefficient* dirichlet = nullptr);

 protected:
  void UpdateBoundaryValues(mfem::real_t t) override;

 private:
  mfem::Array<int> ess_bdr_, marker_;
  std::unique_ptr<mfem::VectorConstantCoefficient> zero_;
  mfem::VectorCoefficient* dirichlet_;
};

}  // namespace AdGIA
