/**
 * @file adjoint.hpp
 * @brief The common adjoint structure (doc/planning/adjoint_note.md): the
 * adjoint force (continuous- and point-in-time terms), the forward
 * trajectory store, kernel accumulators, and the adjoint driver for the
 * static and quasi-static viscoelastic rungs.
 */

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "mfem.hpp"
#include "AdGIA/mixed_problem.hpp"
#include "AdGIA/quasi_static_problem.hpp"
#include "AdGIA/sea_level.hpp"
#include "AdGIA/viscoelastic.hpp"

namespace AdGIA {

/**
 * @brief The adjoint force: the data of the objective functional's
 * derivatives, carried as two kinds of terms that are handled differently
 * (doc/planning/adjoint_note.md, "The common structure").
 *
 * POINT-IN-TIME terms are observations J depending on the state at one
 * forward time t: each carries the dual h of J with respect to the
 * displacement (the AddForce L-vector layout), and triggers, at the
 * reversed time, one impulsive solve and the adjoint internal-variable
 * jump. A static adjoint is the degenerate case of point terms alone.
 *
 * CONTINUOUS-IN-TIME terms are ordinary time-dependent dual loads: a
 * filler that ADDS the dual at a given forward time into a buffer. The
 * driver applies them through ViscoelasticOperator::SetForceHook() at
 * every elastic solve of the reversed stepping, mapping the adjoint's
 * reversed time back to forward time — callers never apply the time
 * reversal themselves.
 *
 * All times are FORWARD times; the driver owns the reversal. Observation
 * times must coincide with nodes of the forward trajectory's time grid
 * (the driver verifies). Two slots are served: the DISPLACEMENT dual
 * (every rung) and the POTENTIAL dual (D_phi J, for problems exposing
 * PotentialForceSlot — the gravitating rungs); an observation touching
 * both rows is added as two terms at the same time, which is exact by
 * linearity. The water and rotation slots of the full dictionary are
 * added by those rungs' specialisations.
 */
/**
 * @brief The slot data of one observation's dual: any subset of the
 * displacement rows (AddForce layout), the potential rows
 * (PotentialForceSlot layout), and the border rows — the uniform-term
 * and angular-velocity components (BorderForceSlot). A sea-level
 * observation J = int h SL1 dS populates all four:
 * u = -(1/g) h grad(Phi0) dual, phi = -(1/g) h dual,
 * uniform = (1/g) int h dS, omega_k = -(1/g) int h psi_k dS.
 */
struct AdjointLoad {
  const mfem::Vector* u = nullptr;
  const mfem::Vector* phi = nullptr;
  mfem::real_t uniform = 0.0;
  const mfem::Vector* omega = nullptr;
};

class AdjointForce {
 public:
  /** @brief A point-in-time observation at forward time @p t with dual
   * @p dual on the displacement rows (copied; AddForce layout). */
  void AddObservation(mfem::real_t t, const mfem::Vector& dual);

  /** @brief A point-in-time observation at forward time @p t with dual
   * @p dual on the POTENTIAL rows (copied; AddPotentialForce layout of
   * the problem's PotentialForceSlot). */
  void AddPotentialObservation(mfem::real_t t, const mfem::Vector& dual);

  /** @brief A point-in-time observation at forward time @p t loading
   * any subset of the slots (all data copied). */
  void AddObservation(mfem::real_t t, const AdjointLoad& load);

  /** @brief A continuous-in-time term: @p filler(t, f) ADDS the dual at
   * forward time @p t into @p f (already sized and initialised). */
  void AddContinuous(std::function<void(mfem::real_t, mfem::Vector&)> filler);

  /** @brief A continuous-in-time term on the potential rows. */
  void AddContinuousPotential(
      std::function<void(mfem::real_t, mfem::Vector&)> filler);

  int NumObservations() const { return static_cast<int>(obs_.size()); }
  /** @brief Observation times, sorted ascending. */
  mfem::real_t ObservationTime(int i) const { return obs_[i].t; }
  /** @brief Slot data of observation @p i (empty vectors: slot unused). */
  const mfem::Vector& ObservationDualU(int i) const { return obs_[i].du; }
  const mfem::Vector& ObservationDualPhi(int i) const { return obs_[i].dphi; }
  mfem::real_t ObservationUniform(int i) const { return obs_[i].uniform; }
  const mfem::Vector& ObservationOmega(int i) const { return obs_[i].omega; }
  /** @brief Whether observation @p i touches the potential rows. */
  bool ObservationOnPotential(int i) const {
    return obs_[i].dphi.Size() > 0;
  }
  /** @brief Whether observation @p i touches the border rows. */
  bool ObservationOnBorder(int i) const {
    return obs_[i].uniform != 0.0 || obs_[i].omega.Size() > 0;
  }

  bool HasContinuous() const { return !continuous_.empty(); }
  bool HasContinuousPotential() const { return !continuous_phi_.empty(); }
  /** @brief Adds every continuous term's dual at forward time @p t into
   * @p f (caller sizes and zeroes). */
  void EvalContinuous(mfem::real_t t, mfem::Vector& f) const;
  void EvalContinuousPotential(mfem::real_t t, mfem::Vector& f) const;

 private:
  struct Observation {
    mfem::real_t t;
    mfem::Vector du, dphi, omega;  // empty: slot unused
    mfem::real_t uniform = 0.0;
  };
  void Insert(mfem::real_t t, Observation&& o);

  std::vector<Observation> obs_;  // kept sorted by time
  std::vector<std::function<void(mfem::real_t, mfem::Vector&)>> continuous_;
  std::vector<std::function<void(mfem::real_t, mfem::Vector&)>>
      continuous_phi_;
};

/**
 * @brief The forward trajectory store: the checkpoint interface of the
 * adjoint sweep (doc/planning/adjoint_note.md, "The trajectory store").
 *
 * The quasi-static state is the internal-variable vector m alone — the
 * displacement is slaved to (m, load(t)) through one elastic solve — so
 * the store records m (and the time) per step and recovers the fields on
 * demand. With @p keep_fields (the store-all policy of the first rungs)
 * the displacement is memoised at Record() and no re-solves happen; without
 * it, Displacement() re-solves through the forward operator — the seed of
 * the revolve-style schedules, and a measured identity against the
 * memoised path (the two differ only through the solver tolerance).
 *
 * Record() is called once per step of the forward run, including the
 * initial state; it makes the forward displacement consistent via
 * SolveElastic (free when the step already left it so). On the re-solve
 * path the reference returned by Displacement() is the forward problem's
 * field and is invalidated by the next Displacement() call.
 */
class ForwardTrajectory {
 public:
  explicit ForwardTrajectory(ViscoelasticOperator& op, bool keep_fields = true);

  /** @brief Record the state @p m at time @p t (after the step). */
  void Record(const mfem::Vector& m, mfem::real_t t);

  /** @brief Number of recorded states (steps + 1). */
  int Size() const { return static_cast<int>(t_.size()); }
  mfem::real_t Time(int n) const { return t_[n]; }
  const mfem::Vector& State(int n) const { return m_[n]; }

  /** @brief The displacement slaved to State(n): memoised, or recovered by
   * one forward solve (see the class comment for the reference's
   * lifetime). */
  const mfem::GridFunction& Displacement(int n);

  /** @brief The strain d = D u at the internal-variable nodes for
   * State(n), in the full internal layout. */
  void Strain(int n, mfem::Vector& d);

  bool KeepsFields() const { return keep_fields_; }
  ViscoelasticOperator& Op() const { return op_; }

 private:
  ViscoelasticOperator& op_;
  bool keep_fields_;
  std::vector<mfem::real_t> t_;
  std::vector<mfem::Vector> m_;
  std::vector<std::unique_ptr<mfem::GridFunction>> u_;
};

/**
 * @brief One pairing of forward and adjoint fields, as the driver hands it
 * to the kernel accumulators. The internal-variable entries are
 * PER-BRANCH (the general linear Prony series; empty on the static
 * rung); @p mdag is also empty for the impulsive terms at observation
 * times, where the adjoint factor is the purely elastic response and
 * the internal variables are bounded.
 */
struct AdjointPair {
  const mfem::GridFunction& u;     ///< forward displacement
  const mfem::GridFunction* d;     ///< forward strain (internal space)
  std::vector<const mfem::GridFunction*> m;     ///< forward m_k per branch
  const mfem::GridFunction& udag;  ///< adjoint displacement
  std::vector<const mfem::GridFunction*> mdag;  ///< adjoint m+_k per branch
  mfem::real_t t = 0.0;            ///< FORWARD time of the pairing
  bool impulse = false;  ///< an observation's impulsive (weight-one) pairing
  /** The adjoint problem whose solved state udag belongs to (its
   * potential, uniform term and angular velocity are current for this
   * pairing) — for accumulators reading more than the elastic fields. */
  LinearQuasiStaticProblem* problem = nullptr;
};

/**
 * @brief Interface of a kernel accumulator: receives every (forward,
 * adjoint) pairing of the sweep with its time-quadrature weight — the
 * impulsive terms at observation times arrive with weight one and a null
 * @p mdag — and accumulates whatever it is for (directional derivatives,
 * a dual vector, a display field). Kernels accumulate DURING the adjoint
 * sweep; the adjoint trajectory is never stored.
 */
class KernelAccumulator {
 public:
  virtual ~KernelAccumulator() = default;
  virtual void AddPair(mfem::real_t w, const AdjointPair& p) = 0;
};

/**
 * @brief The pointwise kernel density of the isotropic generalised
 * Maxwell parameterisation (kappa, mu_inf, {mu_k, eta_k}), for one
 * (forward, adjoint) pairing and one branch
 * (doc/planning/adjoint_note.md, "The kernels" and "The rheology axis"):
 *
 *   Kappa:      - div(u) div(u+)
 *   MuLongTerm: - 2 dev eps(u) : dev eps(u+)
 *   Mu (branch k, at fixed eta_k):
 *               - 2 [ dev eps(u) : dev eps(u+) - m_k : dev eps(u+)
 *                     - (d - m_k) : m+_k ]
 *   Eta (branch k):
 *               - (2 / tau_k) (d - m_k) : m+_k
 *
 * The split tracks where each parameter enters the stress
 * sigma = kappa tr(eps) 1 + 2 mu_inf d + sum_k 2 mu_k (d - m_k):
 * mu_inf only through the stiffness (the elastic density alone); a
 * branch mu_k through the stiffness, its branch force and — at fixed
 * eta_k — its relaxation time; eta_k through the evolution alone. For
 * the classical single-branch Maxwell body (mu_inf = 0) Mu and Eta
 * reduce to the (kappa, mu, eta) kernels of the examples.
 *
 * The dev eps factors are pointwise from the displacement gradients
 * (that is how the moduli enter the discrete elastic form); d, m_k and
 * m+_k are the trace-free component fields of the internal-variable
 * space, contracted through the library component order (index.hpp) in
 * any dimension. Null internal fields are zero: with all of them null
 * the Mu/Kappa densities are the static elastic kernels, and with mdag
 * null they are the impulsive parts at an observation time.
 */
class IsotropicKernelDensity : public mfem::Coefficient {
 public:
  enum class Parameter { Kappa, MuLongTerm, Mu, Eta };

  /** @param tau The relaxation-time coefficient; required for Eta. */
  explicit IsotropicKernelDensity(Parameter parameter,
                                  mfem::Coefficient* tau = nullptr);

  /** @brief The pairing to evaluate; null pointers mean zero fields. */
  void SetPair(const mfem::GridFunction& u, const mfem::GridFunction& udag,
               const mfem::GridFunction* d = nullptr,
               const mfem::GridFunction* m = nullptr,
               const mfem::GridFunction* mdag = nullptr);

  mfem::real_t Eval(mfem::ElementTransformation& T,
                    const mfem::IntegrationPoint& ip) override;

  Parameter Param() const { return parameter_; }

 private:
  Parameter parameter_;
  mfem::Coefficient* tau_;
  const mfem::GridFunction* u_ = nullptr;
  const mfem::GridFunction* udag_ = nullptr;
  const mfem::GridFunction* d_ = nullptr;
  const mfem::GridFunction* m_ = nullptr;
  const mfem::GridFunction* mdag_ = nullptr;
  mfem::DenseMatrix Du_, Dv_, A_, B_;
  mfem::Vector av_, bv_;
};

/**
 * @brief Kernel accumulator for one isotropic parameter. Three outputs,
 * each optional:
 *
 * - DIRECTIONAL derivatives: AddDirection(delta) registers an absolute
 *   model perturbation, and Directional(i) returns dJ = int K delta dV
 *   accumulated with the sweep's quadrature (the assembly rule of the
 *   elastic forms — the faithful pairing; global in parallel). This is
 *   the finite-difference gate's quantity and the derivative-along-a-
 *   direction products an inversion wrapper asks for.
 * - The DUAL vector: with SetDualSpace, j_i = int K phi_i accumulates on
 *   that scalar space (same mesh as the problem); Dual() is the local
 *   assembled layout, TrueDual() the true-dof vector (the export of the
 *   pygeoinf contract — never pre-Riesz'd here).
 * - A display FIELD: with SetField, the L2 projection of the density
 *   accumulated with the sweep's weights (tutorial/plot grade).
 *
 * @p sfes hosts the directional quadrature (a scalar space on the
 * problem's mesh, typically of the displacement order). The Eta kernel
 * needs @p tau (that branch's relaxation time) and skips impulsive
 * pairings (its integrand is bounded). @p branch selects the Prony
 * branch of the Mu and Eta kernels (Kappa and MuLongTerm ignore it).
 */
class IsotropicKernel : public KernelAccumulator {
 public:
  using Parameter = IsotropicKernelDensity::Parameter;

  IsotropicKernel(Parameter parameter, mfem::FiniteElementSpace& sfes,
                  mfem::Coefficient* tau = nullptr, int branch = 0);

  /** @brief Register a perturbation direction; returns its index. */
  int AddDirection(mfem::Coefficient& delta);
  /** @brief dJ along direction @p i (global in parallel). */
  mfem::real_t Directional(int i) const;

  /** @brief Accumulate the dual on @p fes (same mesh as the problem),
   * restricted to @p attr_marker when given. */
  void SetDualSpace(mfem::FiniteElementSpace& fes,
                    const mfem::Array<int>* attr_marker = nullptr);
  /** @brief The accumulated dual in the assembled (local) layout. */
  const mfem::Vector& Dual() const { return dual_; }
  /** @brief The dual on true dofs (serial: a copy). */
  void TrueDual(mfem::Vector& j) const;

  /** @brief Accumulate a display field on the (L2) space @p fes. */
  void SetField(mfem::FiniteElementSpace& fes);
  const mfem::GridFunction& Field() const { return *field_; }

  void AddPair(mfem::real_t w, const AdjointPair& p) override;

 private:
  mfem::real_t Integrate(mfem::Coefficient& c) const;

  IsotropicKernelDensity density_;
  mfem::FiniteElementSpace* sfes_;
  int branch_;
  std::vector<mfem::Coefficient*> directions_;
  std::vector<mfem::real_t> values_;  // local partial sums
  mfem::FiniteElementSpace* dual_fes_ = nullptr;
  mfem::Array<int> dual_marker_;  // empty: the whole mesh
  mfem::Vector dual_;
  std::unique_ptr<mfem::GridFunction> field_, field_step_;
};

/**
 * @brief The sea-level (data-load) kernel accumulator: everything the
 * ice and initial-sea-level kernels need, read off the adjoint sea
 * level (doc/planning/adjoint_note.md, "The kernels"). From the
 * adjoint identity, the derivative of J with respect to the DATA
 * surface load is
 *
 *     delta J = -g int int delta sigma(x, t) SL1+(x, reversed t) dS dt
 *
 * (Yu et al. 2025, eq. 63 rearranged): the kernel of any sigma_data
 * route is -g times the adjoint sea level, with the observation
 * impulses' singular parts contributing atoms read from the
 * generalized fingerprint solves. Per sweep pairing the accumulator
 * computes SL1+ of the adjoint state (displacement, potential, uniform
 * term and, with rotation, psi(omega+)) through SeaLevelOperator, and
 * accumulates:
 *
 *  - DIRECTIONAL derivatives along registered time-aware delta-sigma
 *    coefficients (the FD-gate quantities; an ice-history direction
 *    delta-I enters through the load law as
 *    delta sigma = rho_i (1 - C(t)) delta I(t) - rho_i (1 - C0)
 *    delta I(0), etc.);
 *  - the QUELLED ice kernels of the rate parameterisation (eq. 84),
 *    when enabled: the I0 kernel field
 *    D_I0(x) = rho_i g int (C(t) - C0) SL1+ dt (live only when the
 *    ocean function moves) and the Idot stack — the per-node fields
 *    -rho_i g (1 - C(t_n)) SL1+_n whose backward cumulative
 *    time-integral (Finalize()) is D_Idot(x, t_n), with only finite
 *    jumps at the observation times (the raw kernel's atoms are kept
 *    separately: decision 4 — the raw derivative stays exportable).
 *
 * The accumulator is a specialisation piece of the mixed gravitating
 * class (it reads Potential(), UniformPotentialTerm(),
 * SolutionCentrifugalPotential()); the referential twin follows the
 * same pattern on zeta when those rungs open.
 */
class SeaLevelLoadKernel : public KernelAccumulator {
 public:
  /**
   * @param adjoint_problem The adjoint twin (the driver's problem).
   * @param sea The surface layer (readout spaces).
   * @param grad_phi0 The equilibrium gravity (g formed from it).
   * @param g_surf The surface gravity (the -g factor of the pairing).
   * @param rotation Read psi(omega+) into the adjoint sea level.
   */
  SeaLevelLoadKernel(LinearQuasiStaticMixedSelfGravitatingProblem&
                         adjoint_problem,
                     SeaLevelOperator& sea,
                     mfem::VectorCoefficient& grad_phi0, mfem::real_t g_surf,
                     bool rotation);

  /** @brief Register a time-aware direction delta-sigma(x, t); returns
   * its index. */
  int AddDirection(mfem::Coefficient& dsigma);
  /** @brief delta J along direction @p i (global in parallel). */
  mfem::real_t Directional(int i) const;

  /** @brief Accumulate the quelled ice kernels: @p C the (stored or
   * prescribed) forward ocean function as a time-aware coefficient,
   * @p C0 its initial state, @p rho_i the ice density. */
  void EnableQuelled(mfem::Coefficient& C, mfem::Coefficient& C0,
                     mfem::real_t rho_i);
  /** @brief D_I0(x) (after the sweep; zero without EnableQuelled). */
  const mfem::GridFunction& I0Kernel() const { return *i0_kernel_; }
  /** @brief Turn the accumulated Idot stack into D_Idot(x, t_n) by the
   * backward cumulative time integral; call once after Run(). */
  void Finalize();
  int NumTimes() const { return static_cast<int>(times_.size()); }
  mfem::real_t Time(int n) const { return times_[n]; }
  /** @brief D_Idot(x, t_n) after Finalize() (the smooth part; the
   * observation atoms are added into every node earlier than their
   * time, so the stack carries the finite jumps of eq. 84). */
  const mfem::GridFunction& IdotKernel(int n) const { return *idot_[n]; }

  void AddPair(mfem::real_t w, const AdjointPair& p) override;

 private:
  void ReadAdjointSeaLevel();  // into sl_ (the operator's field copy)

  LinearQuasiStaticMixedSelfGravitatingProblem& problem_;
  SeaLevelOperator& sea_;
  mfem::VectorCoefficient& grad_phi0_;
  mfem::real_t g_surf_;
  bool rotation_;
  std::vector<mfem::Coefficient*> directions_;
  std::vector<mfem::real_t> values_;
  mfem::Coefficient* C_ = nullptr;
  mfem::Coefficient* C0_ = nullptr;
  mfem::real_t rho_i_ = 0.0;
  std::unique_ptr<mfem::GridFunction> i0_kernel_, sl_;
  // The Idot stack: per smooth node the integrand field
  // -rho_i g (1 - C(t_n)) SL1+_n and its time; atoms are folded in by
  // AddPair as finite jumps. Finalize() turns it into the backward
  // cumulative integral.
  std::vector<mfem::real_t> times_;
  std::vector<std::unique_ptr<mfem::GridFunction>> idot_;
  std::vector<mfem::real_t> atoms_t_;
  std::vector<std::unique_ptr<mfem::GridFunction>> atoms_;
  bool finalized_ = false;
};

/**
 * @brief The adjoint driver: runs the time-reversed adjoint problem
 * against a recorded forward trajectory and feeds every registered
 * kernel accumulator (doc/planning/adjoint_note.md, "The adjoint
 * driver"). Two configurations:
 *
 * - STATIC (rung 1): the adjoint of a solved static problem. The adjoint
 *   problem is the same problem with zero external load (caller-built,
 *   sharing the space and rheology); each point term costs one solve,
 *   whose displacement pairs with the given forward one.
 * - QUASI-STATIC (rung 2): the adjoint evolution on a zero-external-load
 *   twin of the forward problem, wrapped in its own ViscoelasticOperator.
 *   The sweep walks the forward grid in reversed time: between
 *   observations it steps with the exponential trapezoid scheme and pairs
 *   each node's fields with the stored forward ones under trapezoid
 *   weights; at an observation time it solves the impulsive problem
 *   (the observation dual on the unrelaxed operator), hands the impulsive
 *   pairing to the kernels, jumps the adjoint internal variable by
 *   (1/tau) times the impulsive strain, and splits the node's weight
 *   across the jump. Continuous force terms enter every elastic solve
 *   through the operator's force hook with the time mapped back to
 *   forward time. The driver owns the displacement-cache invalidation at
 *   the jumps.
 *
 * Observation times must lie on the trajectory's grid. The adjoint
 * problem's own load coefficients should be zero: the adjoint force
 * carries all the loading.
 */
class AdjointSolver {
 public:
  /** @brief Static configuration: @p adjoint_problem is the zero-load
   * twin; @p forward_displacement the solved forward field. */
  AdjointSolver(LinearQuasiStaticProblem& adjoint_problem,
                const mfem::GridFunction& forward_displacement);

  /** @brief Quasi-static configuration: @p adjoint_op wraps the zero-load
   * twin; @p forward is the recorded trajectory (not owned). */
  AdjointSolver(ViscoelasticOperator& adjoint_op, ForwardTrajectory& forward);

  void SetForce(const AdjointForce& force) { force_ = &force; }
  void AddKernel(KernelAccumulator& kernel) { kernels_.push_back(&kernel); }

  /** @brief Synchronise time-dependent problem data with the sweep: the
   * callback receives the FORWARD time and runs before every elastic
   * solve of the reversed stepping (through the operator's force hook)
   * and before every impulsive observation solve. The C† playback of a
   * migrating/moving-ocean forward is the canonical use: set the
   * adjoint water coefficient's time and call RefreshWaterLoad(), so
   * the adjoint runs as the linear problem with the forward's
   * time-dependent coefficients played back in reverse
   * (doc/planning/adjoint_note.md, "The migrating-forward adjoint");
   * the nonlinear tangent adjoint's trajectory feed (R2) will ride the
   * same hook. Static adjoints ignore it.
   */
  void SetTimeSync(std::function<void(mfem::real_t)> sync) {
    time_sync_ = std::move(sync);
  }

  /** @brief Run the adjoint and accumulate the kernels; false if a solve
   * fails. */
  bool Run();

 private:
  bool RunStatic();
  bool RunEvolution();
  void Pair(mfem::real_t w, const AdjointPair& p);
  bool ImpulseSolve(int obs, mfem::real_t s);
  void ApplyJump(const mfem::GridFunction& udag_g, mfem::Vector& mdag);
  /** @brief Resolve the potential and border slots the force needs
   * (refuses on problems without them). */
  void ResolveSlots(LinearQuasiStaticProblem& problem);
  /** @brief Apply observation @p obs's slot data to the assembled
   * problem. */
  void ApplyObservation(int obs);

  LinearQuasiStaticProblem* problem_ = nullptr;      // static configuration
  const mfem::GridFunction* u_forward_ = nullptr;
  ViscoelasticOperator* op_ = nullptr;               // evolution configuration
  ForwardTrajectory* forward_ = nullptr;
  const AdjointForce* force_ = nullptr;
  PotentialForceSlot* phi_slot_ = nullptr;
  BorderForceSlot* border_slot_ = nullptr;
  std::function<void(mfem::real_t)> time_sync_;
  std::vector<KernelAccumulator*> kernels_;
  mfem::Vector strain_, fbuf_, fbuf_phi_;
};

}  // namespace AdGIA
