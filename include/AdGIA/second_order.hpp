/**
 * @file second_order.hpp
 * @brief The second-order adjoint: the derivative along a model direction
 * v of the first-order kernels (the action of the objective's Hessian on
 * v), by the tangent of the adjoint sweep — the forward-mode twin of the
 * first-order adjoint applied to the adjoint itself. sl3dpp's continuous
 * second-order sweep is the model; the method is in doc/methods/adjoints.md.
 */
#pragma once
#include <memory>
#include <vector>

#include "mfem.hpp"
#include "AdGIA/adjoint.hpp"
#include "AdGIA/tangent.hpp"

namespace AdGIA {

/**
 * @brief A window of adjoint nodes as TangentData: the reversed-time
 * states and displacements the derivative state's step reads (the start
 * and the end of one reversed step, or the post-jump node at an
 * observation). Fields are copied in; strains come from the adjoint
 * operator's map.
 */
class AdjointWindow : public TangentData {
 public:
  explicit AdjointWindow(ViscoelasticOperator& op) : op_(op) {}
  void Clear() { nodes_.clear(); }
  /** @brief Append a node; returns its index. */
  int Push(mfem::real_t s, const mfem::Vector& m, const mfem::GridFunction& u);
  int Size() const override { return static_cast<int>(nodes_.size()); }
  mfem::real_t Time(int n) const override { return nodes_[n].s; }
  const mfem::Vector& State(int n) const override { return nodes_[n].m; }
  const mfem::GridFunction& Displacement(int n) override {
    return *nodes_[n].u;
  }
  void Strain(int n, mfem::Vector& d) override {
    op_.ComputeStrain(*nodes_[n].u, d);
  }

 private:
  struct Node {
    mfem::real_t s;
    mfem::Vector m;
    std::unique_ptr<mfem::GridFunction> u;
  };
  ViscoelasticOperator& op_;
  std::vector<Node> nodes_;
};

/**
 * @brief The second-order adjoint driver: runs the first-order adjoint
 * sweep of AdjointSolver and, alongside it, the derivative of that sweep
 * along the direction v — the second-order adjoint state — and feeds the
 * kernel accumulators the derivative of every pairing, which is the
 * Hessian-vector product J''(x) v in the kernels' own form (directional
 * values along the accumulators' directions, dual vectors, fields).
 *
 * For an observation functional linear in the response, the gradient
 * is the sum over the sweep of pairings K(u, u†) of forward and adjoint
 * fields; its derivative along v is K(δu, u†) + K(u, δu†) plus the
 * kernel densities' explicit dependence on the parameters (the Eta
 * density's 1/τ). δu is v's tangent trajectory (TangentLinearSolver);
 * δu† is the tangent of the adjoint sweep: between observations the
 * derivative state is stepped by TangentViscoelasticOperator on the
 * adjoint's reversed trajectory (the adjoint evolution is the forward
 * operator on the zero-load twin, so its tangent is the same tangent
 * step with the adjoint fields as data); at an observation the
 * impulsive solve's derivative is the Born-forced solve
 * K_U δu†_imp = −δK_U u†_imp, and the derivative state jumps by
 * (1/τ) D δu†_imp + δ(1/τ) D u†_imp; the observation duals themselves
 * carry no model dependence. The impulsive pairings and the trapezoid
 * weights are the first-order sweep's.
 *
 * This is the CONTINUOUS second-order adjoint, consistent with the
 * library's first-order one: the exact derivative of the first-order
 * kernels as computed (gate: central differences of the kernels along
 * v), which inherit the first-order sweep's O(dt²) distance from the
 * exact discrete gradient.
 *
 * Two configurations, as AdjointSolver: STATIC (the adjoint twin, the
 * derivative twin, the forward and tangent displacements) and
 * QUASI-STATIC (the adjoint operator, the derivative twin, the forward
 * and tangent trajectories). Both twins are zero-load problems sharing
 * the forward's spaces and rheology. Linear isotropic rheologies, whole-
 * mesh branches, no time sync (playback), as the first-order driver.
 */
class SecondOrderAdjointSolver {
 public:
  SecondOrderAdjointSolver(LinearQuasiStaticProblem& adjoint_problem,
                           LinearQuasiStaticProblem& derivative_twin,
                           const mfem::GridFunction& forward_displacement,
                           const mfem::GridFunction& tangent_displacement,
                           const TangentDirection& v);
  SecondOrderAdjointSolver(ViscoelasticOperator& adjoint_op,
                           LinearQuasiStaticProblem& derivative_twin,
                           ForwardTrajectory& forward,
                           const TangentTrajectory& tangent,
                           const TangentDirection& v);

  void SetForce(const AdjointForce& force) { force_ = &force; }
  /** @brief Accumulators of the DERIVATIVE pairings (the Hessian-vector
   * product). */
  void AddKernel(KernelAccumulator& kernel) { kernels_.push_back(&kernel); }
  /** @brief Accumulators of the first-order pairings (the gradient, a
   * by-product of the same sweep). */
  void AddFirstOrderKernel(KernelAccumulator& kernel) {
    first_.push_back(&kernel);
  }

  bool Run();

 private:
  bool RunStatic();
  bool RunEvolution();
  void ResolveSlots(LinearQuasiStaticProblem& problem);
  void ApplyObservation(int obs);
  /** @brief Feed the pairings of one node or impulse: first-order
   * (u, u†) to the first-order accumulators, (δu, u†) + (u, δu†) and
   * the explicit Eta term to the derivative accumulators. */
  void PairAll(mfem::real_t w, const AdjointPair& first,
               const AdjointPair& d_forward, const AdjointPair& d_adjoint);
  void BuildEtaScale();

  LinearQuasiStaticProblem* adjoint_ = nullptr;  // static configuration
  const mfem::GridFunction* u_forward_ = nullptr;
  const mfem::GridFunction* du_forward_ = nullptr;
  ViscoelasticOperator* op_ = nullptr;           // evolution configuration
  ForwardTrajectory* forward_ = nullptr;
  const TangentTrajectory* tangent_ = nullptr;
  LinearQuasiStaticProblem& twin_;
  TangentDirection v_;
  std::unique_ptr<TangentViscoelasticOperator> dop_;  // the derivative state
  std::unique_ptr<AdjointWindow> window_;
  std::unique_ptr<mfem::GridFunction> eta_scale_gf_;
  std::unique_ptr<mfem::GridFunctionCoefficient> eta_scale_;
  const AdjointForce* force_ = nullptr;
  PotentialForceSlot* phi_slot_ = nullptr;
  BorderForceSlot* border_slot_ = nullptr;
  std::vector<KernelAccumulator*> kernels_, first_;
  mfem::Vector strain_, fbuf_;
};

}  // namespace AdGIA
