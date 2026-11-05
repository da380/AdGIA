/**
 * @file discrete_second_order.hpp
 * @brief The discrete second-order adjoint: the derivative of the
 * discrete gradient (discrete_adjoint.hpp) along a model direction v,
 * i.e. the exact Hessian-vector product of the discrete objective,
 * symmetric to rounding (doc/adjoints.md, section 5).
 */
#pragma once
#include <memory>
#include <vector>

#include "mfem.hpp"
#include "AdGIA/discrete_adjoint.hpp"
#include "AdGIA/tangent.hpp"

namespace AdGIA {

/** @brief The tangent trajectory of a direction as TangentData (the
 * tangent's sources evaluated on it give the forward-field part of the
 * sources' derivative). */
class TangentTrajectoryData : public TangentData {
 public:
  TangentTrajectoryData(const TangentTrajectory& t, ViscoelasticOperator& op)
      : t_(t), op_(op) {}
  int Size() const override { return t_.Size(); }
  mfem::real_t Time(int n) const override { return t_.Time(n); }
  const mfem::Vector& State(int n) const override { return t_.State(n); }
  const mfem::GridFunction& Displacement(int n) override {
    return t_.Displacement(n);
  }
  void Strain(int n, mfem::Vector& d) override {
    op_.ComputeStrain(t_.Displacement(n), d);
  }

 private:
  const TangentTrajectory& t_;
  ViscoelasticOperator& op_;
};

/** @brief The discrete duals (lambda, mu) as TangentData: the tangent's
 * effective Born form of a direction applied to lambda gives delta_v K_eff
 * lambda, the source of the derivative recursion. */
class DualData : public TangentData {
 public:
  DualData(DiscreteAdjointOperator& op, ForwardTrajectory& forward)
      : op_(op), forward_(forward) {}
  int Size() const override { return op_.Size(); }
  mfem::real_t Time(int n) const override { return forward_.Time(n); }
  const mfem::Vector& State(int n) const override { return op_.Mu(n); }
  const mfem::GridFunction& Displacement(int n) override {
    return op_.Lambda(n);
  }
  void Strain(int n, mfem::Vector& d) override {
    op_.ComputeStrain(op_.Lambda(n), d);
  }

 private:
  DiscreteAdjointOperator& op_;
  ForwardTrajectory& forward_;
};

/**
 * @brief The discrete second-order adjoint driver. Given a run of the
 * discrete first-order adjoint (the duals lambda^n, mu^n on the twin) and
 * the tangent trajectory of a direction v, it runs the derivative of the
 * transposed recursion along v (the same recursion, one effective solve
 * per reversed step, with sources from the recursion's own v-dependence
 * applied to the first-order duals) and then evaluates the Hessian
 * along any direction delta: H[v, delta] = the derivative duals paired
 * with delta's sources on the forward trajectory, plus the first-order
 * duals paired with delta's sources on v's tangent trajectory, plus the
 * parameter cross terms (the scheme's second derivative in the model).
 * Whole-mesh branches, linear isotropic rheologies. The pairings are
 * global in parallel (shared displacement dofs counted once, the
 * internal-space sums reduced over ranks), as in DiscreteAdjointSolver.
 */
class DiscreteSecondOrderSolver {
 public:
  /** @brief Static: @p first is a run static discrete adjoint (its
   * duals lambda_i), @p derivative_twin a zero-load twin for the
   * derivative duals, @p tangent_displacement v's static tangent. The
   * derivative dual solves K dlambda_i = -K(v) lambda_i and
   * H[v, delta] = sum_i <dlambda_i, -K(delta) u> + <lambda_i, -K(delta) du>. */
  DiscreteSecondOrderSolver(DiscreteAdjointSolver& first,
                            LinearQuasiStaticProblem& derivative_twin,
                            const mfem::GridFunction& forward_displacement,
                            const mfem::GridFunction& tangent_displacement,
                            const TangentDirection& v);
  /** @param first The run discrete first-order adjoint (its duals and
   * twin are read; its source twin is reused for the sources).
   * @param derivative_twin A zero-load twin for the derivative duals'
   * solves (a third twin: the first-order twin keeps its state).
   * @param source_twin_v A zero-load twin for v's source builder.
   * @param tangent_v The tangent trajectory of v on the forward grid.
   * @param v The direction. */
  DiscreteSecondOrderSolver(DiscreteAdjointSolver& first,
                            LinearQuasiStaticProblem& derivative_twin,
                            LinearQuasiStaticProblem& source_twin_v,
                            ForwardTrajectory& forward,
                            const TangentTrajectory& tangent_v,
                            const TangentDirection& v);

  /** @brief Run the derivative recursion; false if a solve fails. */
  bool Run();
  /** @brief H[v, delta]. The source twin is unused in the static
   * configuration (pass any problem). */
  mfem::real_t Directional(const TangentDirection& delta,
                           LinearQuasiStaticProblem& source_twin_delta);

  const mfem::GridFunction& DeltaLambda(int n) const { return *dlambda_[n]; }
  const mfem::Vector& DeltaMu(int n) const { return dmu_[n]; }

 private:
  // Nodal delta(1/tau) of a direction and its second derivative with
  // another, on branch k (eta = mu / itau).
  void SecondInverseRelaxationTime(int k, const TangentViscoelasticOperator& a,
                                   const TangentViscoelasticOperator& b,
                                   mfem::Vector& d2itau) const;

  bool RunStatic();
  mfem::real_t DirectionalStatic(const TangentDirection& delta);
  mfem::real_t Dot(const mfem::Vector& dual, const mfem::GridFunction& u,
                   mfem::FiniteElementSpace& fes) const;
  // The sum over ranks of a local pairing on the element-local (L2)
  // internal dofs; the identity in serial.
  mfem::real_t SumOverRanks(mfem::real_t local,
                            mfem::FiniteElementSpace& fes) const;

  DiscreteAdjointSolver& first_;
  // Static configuration.
  LinearQuasiStaticProblem* twin_ = nullptr;
  const mfem::GridFunction* u_forward_ = nullptr;
  const mfem::GridFunction* du_forward_ = nullptr;
  std::vector<std::unique_ptr<mfem::GridFunction>> dlambda_static_;
  // Quasi-static configuration.
  DiscreteAdjointOperator* op_ = nullptr;
  ForwardTrajectory* forward_ = nullptr;
  const TangentTrajectory* tangent_v_ = nullptr;
  TangentDirection v_;
  std::unique_ptr<DiscreteAdjointOperator> dop_;  // the derivative duals' solves
  std::unique_ptr<TangentViscoelasticOperator> src_v_;
  std::vector<std::unique_ptr<mfem::GridFunction>> dlambda_;
  std::vector<mfem::Vector> dmu_;
};

}  // namespace AdGIA
