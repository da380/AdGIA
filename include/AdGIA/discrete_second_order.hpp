/**
 * @file discrete_second_order.hpp
 * @brief The discrete second-order adjoint: the derivative of the
 * discrete gradient (discrete_adjoint.hpp) along a model direction v,
 * i.e. the exact Hessian-vector product of the discrete objective,
 * symmetric to rounding (doc/methods/adjoints.md, section 5).
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
 *
 * THE DUAL VECTOR OF H[v, .] (quasi-static configuration): AddKernel(),
 * AssembleKernels(), Dual(), Pair(), on the internal scalar space as
 * DiscreteAdjointSolver's first-order kernels. Every term of
 * H[v, delta] is linear in delta, so H[v, delta] = sum_q j_q p_q for a
 * direction with nodal values p_q of one parameter (q an internal node,
 * phi_q its L2 basis function, the direction's interpolant
 * sum_q p_q phi_q). Notation: on the step n -> n+1 of size dt and at
 * node q, h = dt/tau_k, the weights e, a, b (beta = 1 - b) and their
 * h-derivatives e', a', b', e'', a'', b''; mu, eta, 1/tau the nodal
 * branch-k values; rho_kappa(u, w) = -div u div w and rho_mu(u, w) =
 * -2 dev eps(u) : dev eps(w); int_q f = int phi_q f dV with the
 * forward stiffness's rule (detail::IntegrateAgainstInternalBasis);
 * lambda^n, mu^n the first-order duals and dlambda^n, dmu^n their
 * derivatives along v (Run()); u, m, d = Du the forward trajectory and
 * du, dm, D du v's tangent trajectory; v's nodal data dmu_v,
 * d(1/tau)_v, dh_v = dt d(1/tau)_v, dbeta_v = -b' dh_v and
 * deta_v = eta (dmu_v/mu - tau d(1/tau)_v). Per parameter p of
 * branch k (mu_k at fixed eta_k, or eta_k) the nodal factors are
 *
 *     chi = dmu/dp          = 1                           | 0
 *     c   = d(1/tau)/dp     = (1/tau)/mu                  | -(1/tau)/eta
 *     g   = d2(1/tau)/dp dv = -deta_v/eta^2               | (2 mu deta_v/eta - dmu_v)/eta^2
 *
 * (1/tau = mu/eta: its second derivatives are 0 in mu mu, -1/eta^2 in
 * mu eta and 2 mu/eta^3 in eta eta). With T(U, M, D; L, X) the
 * first-order dual of DiscreteAdjointSolver::AssembleKernels for a data
 * trajectory (U, M, D) and duals (L, X):
 *
 *     node 0:  int_q rho_p(U^0, L^0)                       (kappa; mu_inf, mu_k: rho_mu)
 *              + chi sum_c (B L^0)_c 2 M^0_c
 *     step:    int_q rho_p(U^{n+1}, L^{n+1})               (kappa, mu_inf)
 *              chi int_q beta rho_mu - b' dt c int_q mu_k rho_mu   (mu_k, eta_k; at U^{n+1}, L^{n+1})
 *              + sum_c (B L^{n+1})_c [2 chi (e M^n + a D^n) + 2 mu dt c (e' M^n + a' D^n)]_c
 *              + sum_c X^{n+1}_c dt c (e' M^n + a' D^n + b' D^{n+1})_c
 *
 * the dual is
 *
 *     j_q = T(u, m, d; dlambda, dmu) + T(du, dm, D du; lambda, mu)
 *         + sum over the steps (mu_k, eta_k), with r1 = e' m^n + a' d^n,
 *           r2 = e'' m^n + a'' d^n and rho_1 = rho_mu(u^{n+1}, lambda^{n+1}):
 *             chi int_q dbeta_v rho_1 - b' dt c int_q dmu_v rho_1
 *             + (-b'' dt c dh_v - b' dt g) int_q mu_k rho_1
 *             + sum_c (B lambda^{n+1})_c [2 chi dh_v r1 + 2 dmu_v dt c r1
 *                                         + 2 mu dt (g r1 + c dh_v r2)]_c
 *             + sum_c mu^{n+1}_c dt [g (r1 + b' d^{n+1}) + c dh_v (r2 + b'' d^{n+1})]_c
 *
 * The first two are the source pairings of Directional() (the
 * derivative duals against delta's sources on the forward trajectory,
 * the first-order duals against them on v's tangent trajectory); the
 * cross terms are the Born cross form, with coefficient dbeta_v dmu_delta
 * + dbeta_delta dmu_v + d2beta mu_k, d2beta = -b'' dh_v dh_delta - b' dt
 * d2(1/tau)[v, delta], and the derivatives along v of the known-part
 * branch source and of the state source, each with
 * dmu_delta = chi p, d(1/tau)_delta = c p, d2(1/tau)[v, delta] = g p.
 * Pair(i, delta) equals Directional() along delta read as a
 * GridFunctionCoefficient to rounding (TestDiscreteSecondOrder,
 * KernelDualsPairExactly2D).
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

  // --- the dual vector of H[v, .] -------------------------------------------

  using Parameter = DiscreteAdjointSolver::Parameter;

  /** @brief Register a kernel: the dual of H[v, .] with respect to the
   * nodal values of the parameter on the internal scalar space
   * (DiscreteAdjointOperator::InternalScalarSpace(), L2 nodal: the dofs
   * are true dofs), @p branch for Mu (at fixed eta_k) and Eta. Returns
   * its index (quasi-static configuration only). */
  int AddKernel(Parameter parameter, int branch = 0);
  /** @brief Assemble the registered kernels after Run(): the duals j_q
   * of the class comment. */
  void AssembleKernels();
  /** @brief The assembled dual of kernel @p i (local layout; L2: true). */
  const mfem::Vector& Dual(int i) const { return kernels_[i].dual; }
  /** @brief The pairing of kernel @p i with a nodal direction on the
   * internal scalar space (global in parallel): equals Directional()
   * along that direction read as a GridFunctionCoefficient. */
  mfem::real_t Pair(int i, const mfem::GridFunction& delta) const;

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

  struct Kernel {
    Parameter parameter;
    int branch;
    mfem::Vector dual;
  };
  std::vector<Kernel> kernels_;
};

}  // namespace AdGIA
