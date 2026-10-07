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

  // --- gauged fluid regions -------------------------------------------------

  /** @brief The gauge penalty's form: Deviatoric (a fluid: dev-dev shear)
   * or Harmonic (a vacuum-extension field: full-gradient
   * @f$\epsilon\mu_g\nabla u:\nabla v@f$). */
  enum class GaugePenalty { Deviatoric, Harmonic };

  /**
   * @brief Treat the marked element attributes as an inviscid fluid in the
   * gauged (relabelling) formulation: the rheology supplies the fluid's
   * physical stiffness (its bulk modulus, with zero shear), and this call
   * adds the gauge-fixing shear penalty @f$\epsilon\,2\mu_g\,
   * \mathrm{dev}\,\varepsilon(u):\mathrm{dev}\,\varepsilon(u')@f$ on those
   * attributes to the *solver* operator only. Solve() then removes the
   * @f$O(\epsilon)@f$ bias from the observables by iterated Tikhonov
   * refinement: each step solves the regularised system for the residual of
   * the physical one, and the residual after an exact step is
   * @f$\epsilon Q\,\delta@f$ with @f$\delta@f$ the last increment, so the
   * error contracts by @f$O(\epsilon\,\mu_g/\mu_{\text{solid}})@f$ per
   * step. The main solve is warm-started from the previous solution; the
   * refinement solves are cold-started (they solve for the increment). The
   * physical operator (SystemMatrix()) is unchanged. See
   * doc/gauged_fluid.md, "Gauge fixing: penalty plus iterated refinement"
   * and "Implementation", for the formulation and its verification.
   *
   * The fluid displacement is gauge-dependent (determined only up to a
   * linearised relabelling); the solid displacement and any field derived
   * from @f$\mathrm{div}(\rho u)@f$ or interface normal displacements are
   * observables. Essential boundary conditions must not touch the marked
   * attributes (their elimination is not folded into the penalty).
   *
   * With @p map non-null (the identity included) the Deviatoric penalty is
   * assembled covariantly, as ElasticTensorIntegrator(C, map) with @f$C@f$
   * the isotropic tensor of @f$\lambda = -2\epsilon\mu_g/d@f$,
   * @f$\mu = \epsilon\mu_g@f$ pulled back through the map, so that a
   * relabelled problem's penalty is the exact pull-back of the unmapped one
   * and both sides of a change-of-variables identity use the same
   * integrator class. The Harmonic form is vacuum-extension gauge data,
   * shared rather than mapped, and refuses a non-identity map.
   *
   * @param fluid_marker Element attributes of the fluid (sized to
   * attributes.Max(); copied).
   * @param mu_gauge Gauge shear scale @f$\mu_g@f$ (a natural choice is the
   * fluid's own bulk modulus); not owned, must outlive the problem.
   * @param epsilon Penalty factor @f$\epsilon@f$ (typically about 1e-2,
   * with 2-3 refinements).
   * @param refinements Tikhonov refinement steps per Solve() (each costs
   * one linear solve on top of the first).
   * @param penalty Form of the penalty (Deviatoric for a fluid).
   * @param map Optional mapping for the covariant Deviatoric penalty; not
   * owned, must outlive the problem.
   */
  virtual void SetGaugedFluid(const mfem::Array<int>& fluid_marker,
                              mfem::Coefficient& mu_gauge,
                              mfem::real_t epsilon, int refinements = 2,
                              GaugePenalty penalty = GaugePenalty::Deviatoric,
                              Diffeomorphism* map = nullptr);

  /** @brief Remove the gauge penalty and the refinement loop. */
  void ClearGaugedFluid();

  bool HasGaugedFluid() const { return gauge_integrators_ != nullptr; }

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
   * Call after SetGaugedFluid(); supported by the block-MINRES paths
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
   * @brief The Tikhonov refinement loop of a gauged-fluid Solve(): after
   * the first regularised solve has put its solution in @p X, repeatedly
   * solve @f$(A + \epsilon Q)\,\delta = \epsilon Q\,\delta_{\text{prev}}@f$
   * through SolveLinearSystem() and accumulate. Overridden by problems
   * whose SolveLinearSystem() carries further unknowns alongside the
   * displacement.
   */
  virtual bool GaugeRefine(mfem::Vector& X);

  /** @brief Assemble the operator if it is out of date. */
  void EnsureOperator();

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

  // Gauged fluid regions: the template form owning the penalty integrator
  // (borrowed by q_form_ and appended to a_solve_form_), the eliminated
  // penalty matrix eps Q for the refinement residuals, and the regularised
  // matrix A + eps Q the solver runs on.
  mfem::Array<int> gauge_marker_;
  std::unique_ptr<mfem::ConstantCoefficient> gauge_eps_coef_;
  std::unique_ptr<mfem::ProductCoefficient> gauge_mu_eps_;
  // The mapped Deviatoric penalty's pulled-back tensor (lambda = -2mu/d).
  std::unique_ptr<mfem::Coefficient> gauge_lambda_eps_;
  std::unique_ptr<mfem::MatrixCoefficient> gauge_Cdev_;
  std::unique_ptr<mfem::BilinearForm> gauge_integrators_;
  mfem::BilinearFormIntegrator* gauge_integ_ = nullptr;
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
  std::unique_ptr<mfem::BilinearForm> q_form_, a_solve_form_;
  mfem::OperatorHandle Q_, A_solve_;
  std::vector<mfem::real_t> gauge_residuals_;

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
