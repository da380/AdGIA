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
 * @brief The discrete adjoint driver: the sweep on the twin and the
 * directional derivatives of the objective along TangentDirections, by
 * pairing the duals with the tangent's sources (the derivative of the
 * scheme), exact to solver tolerance. Kernel fields come later (stage 2).
 */
class DiscreteAdjointSolver {
 public:
  /** @param twin The zero-load twin of the forward problem (the duals'
   * solves). @param source_twin A second zero-load twin for the source
   * builder (its effective modulus is engaged independently). */
  DiscreteAdjointSolver(LinearQuasiStaticProblem& twin,
                        LinearQuasiStaticProblem& source_twin,
                        ForwardTrajectory& forward);
  void SetForce(const AdjointForce& force) { force_ = &force; }
  bool Run();
  /** @brief dJ along @p dir (global in parallel). */
  mfem::real_t Directional(const TangentDirection& dir);
  DiscreteAdjointOperator& Operator() { return *op_; }

 private:
  mfem::real_t Dot(const mfem::Vector& a, const mfem::Vector& b,
                   mfem::FiniteElementSpace* fes) const;

  ForwardTrajectory& forward_;
  LinearQuasiStaticProblem& source_twin_;
  std::unique_ptr<DiscreteAdjointOperator> op_;
  std::unique_ptr<TangentViscoelasticOperator> src_;
  const AdjointForce* force_ = nullptr;
};

}  // namespace AdGIA
