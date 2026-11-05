/**
 * @file discrete_adjoint.hpp
 * @brief The discrete first-order adjoint: the exact transpose of the
 * exponential-trapezoid scheme (the transpose of tangent.hpp), giving the
 * exact gradient of the discrete objective. The parity with the continuous
 * adjoint of adjoint.hpp that sl3dpp has; design and equations in
 * doc/planning/adjoint_note.md, "The discrete first-order adjoint".
 */
#pragma once
#include <memory>
#include <vector>

#include "mfem.hpp"
#include "AdGIA/adjoint.hpp"
#include "AdGIA/tangent.hpp"
#include "AdGIA/viscoelastic.hpp"

namespace AdGIA {

/**
 * @brief The transposed recursion on the zero-load twin: reversed over
 * the forward grid, the dual of the internal state mu_k^n = e_k (mu_k^{n+1}
 * + C_k B_k lambda^{n+1}) and the dual solve K_eff lambda^n = g_n +
 * B^T P sum_k [a_k (C_k B_k lambda^{n+1} + mu_k^{n+1}) + b_k mu_k^n] with
 * the step's own effective operator (unrelaxed at n = 0), g_n the
 * observation duals at node n. Whole-mesh branches, linear isotropic
 * rheologies, the Galerkin strain map.
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
  // Quasi-static configuration.
  ForwardTrajectory* forward_ = nullptr;
  std::unique_ptr<DiscreteAdjointOperator> op_;
  std::unique_ptr<TangentViscoelasticOperator> src_;
  const AdjointForce* force_ = nullptr;
  std::vector<Kernel> kernels_;
};

}  // namespace AdGIA
