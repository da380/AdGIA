/**
 * @file discrete_adjoint.hpp
 * @brief The discrete first-order adjoint: the exact transpose of the
 * exponential-trapezoid scheme (the transpose of tangent.hpp), giving the
 * exact gradient of the discrete objective with respect to the moduli and
 * to the data surface load. The parity with the continuous adjoint of
 * adjoint.hpp that sl3dpp has; the equations are in doc/adjoints.md,
 * section 5.
 */
#pragma once
#include <memory>
#include <vector>

#include "mfem.hpp"
#include "AdGIA/adjoint.hpp"
#include "AdGIA/tangent.hpp"
#include "AdGIA/viscoelastic.hpp"

namespace AdGIA {

namespace detail {
/** @brief out[q] = int phi_q density dV over the internal scalar space's
 * basis @p sfes (L2 nodal), with the forward stiffness's own rule
 * (ElasticityIntegrator: order 2 OrderGrad of the displacement element
 * of @p ufes), so that a nodal direction paired with @p out equals the
 * assembled elastic form with that direction's interpolant; the rule is
 * taken from the first element (a mesh of one element type). */
void IntegrateAgainstInternalBasis(mfem::Coefficient& density,
                                   mfem::FiniteElementSpace& ufes,
                                   mfem::FiniteElementSpace& sfes,
                                   mfem::Vector& out);
}  // namespace detail

/**
 * @brief The transposed recursion on the zero-load twin: reversed over
 * the forward grid, the dual of the internal state mu_k^n = e_k (mu_k^{n+1}
 * + C_k B_k lambda^{n+1}) and the dual solve K_eff lambda^n = g_n +
 * B^T P sum_k [a_k (C_k B_k lambda^{n+1} + mu_k^{n+1}) + b_k mu_k^n] with
 * the step's own effective operator (unrelaxed at n = 0), g_n the
 * observation duals at node n. Whole-mesh branches, linear isotropic
 * rheologies, the Galerkin strain map. On the mixed gravitating class the
 * dual solve also returns the duals of the potential, uniform-term and
 * angular-velocity rows (the twin's Potential(), UniformPotentialTerm()
 * and AngularVelocity(): the bordered system is symmetric), kept per node
 * beside lambda^n for the load derivative.
 */
class DiscreteAdjointOperator : public ViscoelasticOperator {
 public:
  DiscreteAdjointOperator(LinearQuasiStaticProblem& twin,
                          const ViscoelasticOperator& forward);

  /** @brief Run the transposed sweep for @p force on @p forward; the
   * duals lambda^n, mu^n are kept per node. */
  bool Run(ForwardTrajectory& forward, const AdjointForce& force);

  int Size() const { return static_cast<int>(lambda_.size()); }
  const mfem::GridFunction& Lambda(int n) const { return *lambda_[n]; }
  const mfem::Vector& Mu(int n) const { return mu_[n]; }
  /** @brief The dual of the potential rows at node @p n (on the parent
   * mesh's potential space); null unless the twin is the mixed
   * gravitating class. */
  const mfem::GridFunction* PotentialDual(int n) const {
    return phi_dual_[n].get();
  }
  /** @brief The dual of the uniform-term row at node @p n (0 without a
   * water load). */
  mfem::real_t UniformDual(int n) const { return uniform_dual_[n]; }
  /** @brief The dual of the angular-velocity rows at node @p n (empty
   * without rotation). */
  const mfem::Vector& OmegaDual(int n) const { return omega_dual_[n]; }

  /** @brief y = B^T P x (the transpose of the strain map), x in the full
   * internal layout, y the displacement L-vector. */
  void StrainMapTranspose(const mfem::Vector& x, mfem::Vector& y) const;

  // --- the scheme's data, for the kernels -----------------------------------

  /** @brief Engage the effective modulus of a step of size @p dt (the
   * weights' values follow); the unrelaxed operator for dt <= 0. */
  void EngageStep(mfem::real_t dt);
  /** @brief The weight coefficient beta_k of the engaged step, as the
   * assembled stiffness carries it. */
  mfem::Coefficient& BetaCoefficient(int k) { return *beta_ptrs_[k]; }
  /** @brief Nodal 2 mu_k at the internal nodes. */
  const mfem::Vector& NodalBranchModulus(int k) const {
    return branch_modulus_[k];
  }
  /** @brief Nodal 1/tau_k (linear rheology: fixed). */
  const mfem::Vector& NodalInverseRelaxationTime(int k) const {
    return itau0_[k];
  }
  int NumNodes() const { return nd_; }
  /** @brief zeta = B lambda (full internal layout). */
  void Coupling(const mfem::GridFunction& lambda, mfem::Vector& zeta) const;

 private:
  void NodeDual(ForwardTrajectory& forward, const AdjointForce& force, int n,
                mfem::real_t w, mfem::Vector& g);

  std::vector<std::unique_ptr<mfem::GridFunction>> lambda_;
  std::vector<mfem::Vector> mu_;
  std::vector<std::unique_ptr<mfem::GridFunction>> phi_dual_;
  std::vector<mfem::real_t> uniform_dual_;
  std::vector<mfem::Vector> omega_dual_;
  PotentialForceSlot* phi_slot_ = nullptr;
  BorderForceSlot* border_slot_ = nullptr;
  mutable mfem::Vector px_, zb_;
};

/**
 * @brief The discrete adjoint driver: the duals and the directional
 * derivatives of the objective along TangentDirections, by pairing the
 * duals with the tangent's sources (the derivative of the scheme), exact
 * to solver tolerance. Two configurations, as TangentLinearSolver:
 *
 * - STATIC: per point observation i, one solve of the twin with the
 *   observation's duals on its slots (displacement, potential, border),
 *   giving lambda_i; then dJ = sum_i <lambda_i, -dK u> with -dK u the
 *   Born force of the direction on the forward displacement
 *   (ModuliBornForce). The twin's operator is symmetric, so this is the
 *   transpose of the static tangent solve.
 * - QUASI-STATIC: the transposed sweep of DiscreteAdjointOperator on the
 *   twin, paired with the sources of TangentViscoelasticOperator on a
 *   second twin.
 *
 * The kernels as dual vectors on the internal space: AddKernel(),
 * AssembleKernels(), Dual(), Pair() (quasi-static configuration).
 *
 * The DATA SURFACE LOAD (frozen shoreline; both configurations): the
 * problem is linear in its load, so dJ along a load direction delta sigma
 * is the sum over the solves of the dual state paired with the full
 * assembled load of delta sigma at the solve's time,
 *
 *     dJ = sum_n [ <lambda^n, f_u^n> + <phi^n_dual, l_phi^n>
 *                  + Phi_g^n_dual S^n + omega^n_dual . r^n ],
 *
 * every row the load touches (displacement force, potential load, the
 * uniform-term datum int delta sigma dS, the omega-row data
 * -int delta sigma Pi psi_k dS) against its dual: DirectionalLoad() reads
 * them off a load twin, LoadDual() is the transpose of that assembly on
 * the body's scalar trace space, int psi_j g SL1(dual) dS with
 * g SL1(dual) = -(lambda . grad Phi0 + phi_dual + psi(omega_dual)
 * - Phi_g_dual), the discrete counterpart of SeaLevelLoadKernel. Static
 * configuration: n runs over the point observations, the load at t = 0.
 * The migrating ocean and the quelled ice kernels are not built here.
 */
class DiscreteAdjointSolver {
 public:
  /** @brief Static: @p twin is the zero-load twin of the solved forward
   * problem (same spaces and rheology), @p forward_displacement the
   * solved forward field. */
  DiscreteAdjointSolver(LinearQuasiStaticProblem& twin,
                        const mfem::GridFunction& forward_displacement);
  /** @brief Quasi-static. @param twin The zero-load twin of the forward
   * problem (the duals' solves). @param source_twin A second zero-load
   * twin for the source builder (its effective modulus is engaged
   * independently). */
  DiscreteAdjointSolver(LinearQuasiStaticProblem& twin,
                        LinearQuasiStaticProblem& source_twin,
                        ForwardTrajectory& forward);
  void SetForce(const AdjointForce& force) { force_ = &force; }
  bool Run();
  /** @brief dJ along @p dir (global in parallel). */
  mfem::real_t Directional(const TangentDirection& dir);
  /** @brief The transposed sweep (quasi-static configuration only). */
  DiscreteAdjointOperator& Operator();
  /** @brief The static duals: one per point observation, in the order
   * of the force's observations (static configuration, after Run()). */
  int NumStaticDuals() const { return static_cast<int>(lambda_.size()); }
  const mfem::GridFunction& StaticDual(int i) const { return *lambda_[i]; }

  // --- the data surface load ------------------------------------------------

  /**
   * @brief dJ along the load of @p load_twin (global in parallel): a twin
   * of the forward problem whose own load is the direction (on the mixed
   * class its water-load DATA is delta sigma(x, t), time-aware through
   * AssembleForce(t) as in the forward), with no moduli direction — the
   * tangent's load direction. Per solve of the forward (quasi-static: the
   * unrelaxed start at node 0 and the effective solve of each step,
   * paired with lambda^n; static: each point observation's dual, the load
   * at t = 0) the twin's AssembleForce(t_n) is read row by row and paired
   * with the dual state: the external load with lambda, and on the mixed
   * class PotentialLoad(), UniformRowLoad() and AngularVelocityRowLoad()
   * with the potential, uniform-term and omega duals; on a class without
   * a potential, the external load alone. A tidal potential on the load
   * twin and the referential class are refused (their rows are not
   * read); Dirichlet data are not read.
   */
  mfem::real_t DirectionalLoad(LinearQuasiStaticProblem& load_twin);

  /**
   * @brief Assemble the load duals of the mixed gravitating class (after
   * Run()): per solve, the transpose of the surface-load assembly on the
   * body boundary marked in @p surface_marker (SubMesh attributes, as
   * SetWaterLoad takes them) applied to the dual state,
   * j_n = int psi_j g SL1(dual_n) dS on LoadDualSpace(), each term by the
   * quadrature rule of the row it transposes (2 p_u for the displacement
   * rows' VectorBoundaryLFIntegrator, the default BoundaryLFIntegrator
   * rule of LoadDualSpace() for the potential, mass and omega rows), so
   * that sum_n PairLoad(LoadDual(n), delta_sigma(t_n)) = DirectionalLoad()
   * to rounding for a load read as a GridFunctionCoefficient of
   * LoadDualSpace() fields.
   */
  void AssembleLoadDuals(const mfem::Array<int>& surface_marker);
  int NumLoadDuals() const { return static_cast<int>(load_duals_.size()); }
  /** @brief The load dual of solve @p n (quasi-static: node n; static:
   * observation n), in the LinearForm layout of LoadDualSpace() (the
   * rank's partial sums; pair with PairLoad()). */
  const mfem::Vector& LoadDual(int n) const { return load_duals_[n]; }
  /** @brief Sum over the solves of LoadDual(n): the dual for a load
   * constant in time (the discrete objective samples the load at the
   * nodes, so the sum carries no time weights). */
  const mfem::Vector& IntegratedLoadDual() const { return load_dual_sum_; }
  /** @brief The space of the load duals: the twin's
   * PotentialSpaceOnBody(), the body's scalar space on which the class
   * assembles its surface-load forms. */
  mfem::FiniteElementSpace& LoadDualSpace();
  /** @brief <dual, delta> for a load dual and a load @p delta on
   * LoadDualSpace() (global in parallel, shared dofs once). */
  mfem::real_t PairLoad(const mfem::Vector& dual,
                        const mfem::GridFunction& delta) const;

  // --- the kernels as dual vectors (stage 2) --------------------------------

  using Parameter = IsotropicKernelDensity::Parameter;

  /** @brief Register a kernel: the dual of J with respect to the nodal
   * values of the parameter on the internal scalar space
   * (DiscreteAdjointOperator::InternalScalarSpace(), L2 nodal: the dofs
   * are true dofs), @p branch for Mu (at fixed eta_k) and Eta. Returns
   * its index. */
  int AddKernel(Parameter parameter, int branch = 0);
  /** @brief Assemble the registered kernels from the duals of Run(). The
   * elastic densities of kappa, mu_inf and the branch moduli (with the
   * steps' weights beta_k on the latter) are integrated against the
   * internal space's basis; the nodal terms of the branch sources, the
   * weights' derivatives and the state source are added pointwise. */
  void AssembleKernels();
  /** @brief The assembled dual of kernel @p i (local layout; L2: true). */
  const mfem::Vector& Dual(int i) const { return kernels_[i].dual; }
  /** @brief The pairing of kernel @p i with a nodal direction on the
   * internal scalar space (global in parallel): equals Directional()
   * along that direction read as a GridFunctionCoefficient. */
  mfem::real_t Pair(int i, const mfem::GridFunction& delta) const;

 private:
  bool RunStatic();
  mfem::real_t DirectionalStatic(const TangentDirection& dir);
  mfem::real_t DirectionalEvolution(const TangentDirection& dir);
  /** @brief <dual, u> for an L-vector @p dual (the rank's partial sums)
   * and a consistent @p u: global in parallel, shared dofs once. */
  mfem::real_t Dot(const mfem::Vector& dual, const mfem::GridFunction& u,
                   mfem::FiniteElementSpace& fes) const;
  /** @brief The sum over ranks of a local pairing on @p fes's
   * element-local (L2) dofs; the identity in serial. */
  mfem::real_t SumOverRanks(mfem::real_t local,
                            mfem::FiniteElementSpace& fes) const;
  /** @brief The assembled load of @p load_twin (after its AssembleForce)
   * paired with one dual state, every row the load touches. */
  mfem::real_t PairAssembledLoad(LinearQuasiStaticProblem& load_twin,
                                 const mfem::GridFunction& lambda,
                                 const mfem::GridFunction* phi,
                                 mfem::real_t uniform,
                                 const mfem::Vector& omega) const;
  /** @brief The dual states of the solves (static or quasi-static). */
  int NumSolves() const;
  void DualState(int n, const mfem::GridFunction*& lambda,
                 const mfem::GridFunction*& phi, mfem::real_t& uniform,
                 const mfem::Vector*& omega) const;
  mfem::real_t SolveTime(int n) const;

  struct Kernel {
    Parameter parameter;
    int branch;
    mfem::Vector dual;
  };
  void AddElastic(Kernel& kernel, const mfem::GridFunction& u,
                  const mfem::GridFunction& lambda, mfem::Coefficient* weight,
                  const mfem::Vector* nodal_scale);

  // Static configuration.
  LinearQuasiStaticProblem* twin_ = nullptr;
  const mfem::GridFunction* u_forward_ = nullptr;
  std::vector<std::unique_ptr<mfem::GridFunction>> lambda_;
  std::vector<std::unique_ptr<mfem::GridFunction>> phi_dual_;
  std::vector<mfem::real_t> uniform_dual_;
  std::vector<mfem::Vector> omega_dual_;
  // Quasi-static configuration.
  ForwardTrajectory* forward_ = nullptr;
  std::unique_ptr<DiscreteAdjointOperator> op_;
  std::unique_ptr<TangentViscoelasticOperator> src_;
  const AdjointForce* force_ = nullptr;
  std::vector<Kernel> kernels_;
  std::vector<mfem::Vector> load_duals_;
  mfem::Vector load_dual_sum_;
};

}  // namespace AdGIA
