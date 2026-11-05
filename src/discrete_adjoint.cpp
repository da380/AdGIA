#include "AdGIA/discrete_adjoint.hpp"

#include <cmath>

#include "AdGIA/detail/fem_factory.hpp"

namespace AdGIA {

using namespace mfem;

namespace {
int InternalOrder(const ViscoelasticOperator& op) {
  return const_cast<ViscoelasticOperator&>(op).InternalScalarSpace().FEColl()
      ->GetOrder();
}
}  // namespace

DiscreteAdjointOperator::DiscreteAdjointOperator(
    LinearQuasiStaticProblem& twin, const ViscoelasticOperator& forward)
    : ViscoelasticOperator(twin, InternalOrder(forward), forward.Map()) {
  MFEM_VERIFY(IsLinear() && TraceFree(),
              "DiscreteAdjointOperator: linear isotropic rheologies.");
  MFEM_VERIFY(Map() == StrainMap::Galerkin,
              "DiscreteAdjointOperator: the Galerkin strain map.");
  for (int k = 0; k < NumBranches(); k++) {
    MFEM_VERIFY(NumBranchNodes(k) == nd_,
                "DiscreteAdjointOperator: whole-mesh branches only.");
  }
}

void DiscreteAdjointOperator::StrainMapTranspose(const Vector& x,
                                                 Vector& y) const {
  // P = G^{-1} (x) M^{-1} is symmetric: apply G^{-1} across components,
  // then M^{-1} per component, then B^T.
  px_.SetSize(nd_ * nc_);
  zb_.SetSize(nd_ * nc_);
  for (int c = 0; c < nc_; c++) {
    for (int p = 0; p < nd_; p++) {
      real_t v = 0.0;
      for (int cp = 0; cp < nc_; cp++) v += Ginv_(c, cp) * x[cp * nd_ + p];
      px_[c * nd_ + p] = v;
    }
  }
  Vector tc, pc;
  for (int c = 0; c < nc_; c++) {
    tc.MakeRef(zb_, c * nd_, nd_);
    pc.MakeRef(px_, c * nd_, nd_);
    Minv_->Mult(pc, tc);
  }
  y.SetSize(ufes_->GetVSize());
  B_->MultTranspose(zb_, y);
}

void DiscreteAdjointOperator::Coupling(const GridFunction& lambda,
                                       Vector& zeta) const {
  zeta.SetSize(nd_ * nc_);
  B_->Mult(lambda, zeta);
}

void DiscreteAdjointOperator::NodeDual(ForwardTrajectory& forward,
                                       const AdjointForce& force, int n,
                                       real_t w, Vector& g) {
  // The observation duals at node n: point terms at its time, continuous
  // terms with the trapezoid weight w. Potential and border duals go to
  // the slots directly (the solve is pending).
  const real_t t = forward.Time(n);
  const int N = forward.Size() - 1;
  const real_t span = forward.Time(N) - forward.Time(0);
  const real_t tol = 1e-8 * (span > 0.0 ? span : 1.0);
  g.SetSize(ufes_->GetVSize());
  g = 0.0;
  for (int i = 0; i < force.NumObservations(); i++) {
    if (std::abs(force.ObservationTime(i) - t) > tol) continue;
    if (force.ObservationDualU(i).Size() > 0) g += force.ObservationDualU(i);
    if (force.ObservationOnPotential(i)) {
      MFEM_VERIFY(phi_slot_, "DiscreteAdjointOperator: no potential slot.");
      phi_slot_->AddPotentialForce(force.ObservationDualPhi(i));
    }
    if (force.ObservationOnBorder(i)) {
      MFEM_VERIFY(border_slot_, "DiscreteAdjointOperator: no border slot.");
      const Vector& om = force.ObservationOmega(i);
      border_slot_->AddBorderForce(force.ObservationUniform(i),
                                   om.Size() > 0 ? &om : nullptr);
    }
  }
  if (force.HasContinuous() && w > 0.0) {
    Vector f(g.Size());
    f = 0.0;
    force.EvalContinuous(t, f);
    g.Add(w, f);
  }
  if (force.HasContinuousPotential() && w > 0.0) {
    MFEM_VERIFY(phi_slot_, "DiscreteAdjointOperator: no potential slot.");
    Vector f(phi_slot_->PotentialForceSpace().GetVSize());
    f = 0.0;
    force.EvalContinuousPotential(t, f);
    f *= w;
    phi_slot_->AddPotentialForce(f);
  }
}

bool DiscreteAdjointOperator::Run(ForwardTrajectory& forward,
                                  const AdjointForce& force) {
  const int N = forward.Size() - 1;
  MFEM_VERIFY(N >= 1, "DiscreteAdjointOperator: fewer than two states.");
  MFEM_VERIFY(Height() == forward.Op().Height(),
              "DiscreteAdjointOperator: the twin's layout differs.");
  phi_slot_ = dynamic_cast<PotentialForceSlot*>(&problem_);
  border_slot_ = dynamic_cast<BorderForceSlot*>(&problem_);
  lambda_.clear();
  lambda_.resize(N + 1);
  mu_.assign(N + 1, Vector());
  const int K = NumBranches();
  const int size = nd_ * nc_;
  Vector g, rhs, acc(size), ck(size), tmp(size), cb(size);
  // mu^N = 0.
  mu_[N].SetSize(Height());
  mu_[N] = 0.0;
  for (int n = N; n >= 0; n--) {
    const real_t wR = n < N ? 0.5 * (forward.Time(n + 1) - forward.Time(n))
                            : 0.0;
    const real_t wL = n > 0 ? 0.5 * (forward.Time(n) - forward.Time(n - 1))
                            : 0.0;
    // The operator of the solve: the step ending at n, unrelaxed at 0.
    if (n > 0) {
      SetEffectiveModulus(forward.Time(n) - forward.Time(n - 1),
                          Scheme::ExponentialTrapezoid);
    } else {
      UseUnrelaxedOperator();
    }
    problem_.AssembleForce(forward.Time(n));
    NodeDual(forward, force, n, wL + wR, g);
    // The internal duals of node n and the transported branch terms.
    acc = 0.0;
    if (n < N) {
      const real_t dt = forward.Time(n + 1) - forward.Time(n);
      EnsureTrapezoidWeights(dt);
      Coupling(*lambda_[n + 1], cb);  // B lambda^{n+1}
      mu_[n].SetSize(Height());
      for (int k = 0; k < K; k++) {
        ApplyBranchModulus(k, cb, ck);  // C_k B lambda^{n+1}
        Vector mu_next = Branch(mu_[n + 1], k);
        Vector mu_k = Branch(mu_[n], k);
        for (int c = 0; c < nc_; c++) {
          for (int q = 0; q < nd_; q++) {
            const int i = c * nd_ + q;
            const real_t s = ck[i] + mu_next[i];
            mu_k[i] = etw_e_[k][q] * s;
            acc[i] += etw_alpha_[k][q] * s;
          }
        }
      }
    }
    if (n > 0) {
      const real_t dt = forward.Time(n) - forward.Time(n - 1);
      EnsureTrapezoidWeights(dt);  // b_k of the step ending at n
      for (int k = 0; k < K; k++) {
        Vector mu_k = Branch(mu_[n], k);
        for (int c = 0; c < nc_; c++) {
          for (int q = 0; q < nd_; q++) {
            acc[c * nd_ + q] += etw_beta_[k][q] * mu_k[c * nd_ + q];
          }
        }
      }
    }
    StrainMapTranspose(acc, rhs);
    rhs += g;
    problem_.AddForce(rhs);
    if (!problem_.Solve()) return false;
    lambda_[n] = detail::MakeGridFunction(ufes_);
    *lambda_[n] = problem_.Displacement();
  }
  UseUnrelaxedOperator();
  return true;
}

// --- DiscreteAdjointSolver ------------------------------------------------

DiscreteAdjointSolver::DiscreteAdjointSolver(
    LinearQuasiStaticProblem& twin, LinearQuasiStaticProblem& source_twin,
    ForwardTrajectory& forward)
    : forward_(forward), source_twin_(source_twin) {
  op_ = std::make_unique<DiscreteAdjointOperator>(twin, forward.Op());
  src_ = std::make_unique<TangentViscoelasticOperator>(source_twin,
                                                       forward.Op());
}

bool DiscreteAdjointSolver::Run() {
  MFEM_VERIFY(force_, "DiscreteAdjointSolver: no force set.");
  return op_->Run(forward_, *force_);
}

real_t DiscreteAdjointSolver::Dot(const Vector& a, const Vector& b,
                                  FiniteElementSpace* fes) const {
  real_t v = a * b;
#ifdef MFEM_USE_MPI
  if (auto* pfes = dynamic_cast<ParFiniteElementSpace*>(fes)) {
    MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                  pfes->GetComm());
  }
#endif
  return v;
}

real_t DiscreteAdjointSolver::Directional(const TangentDirection& dir) {
  MFEM_VERIFY(op_->Size() > 0, "DiscreteAdjointSolver: Run() first.");
  src_->SetDirection(dir);
  ForwardTrajectoryData data(forward_);
  const int N = forward_.Size() - 1;
  const int K = op_->NumBranches();
  Vector f_u, cb;
  std::vector<Vector> zeta;
  real_t dJ = 0.0;
  // The dot products of L-vectors against duals: lambda and the
  // sources are both L-vectors of one space; in parallel the
  // displacement pairing <lambda, f> needs the true-dof reduction of
  // an L-vector force against a consistent lambda, which the
  // GridFunction/LinearForm pairing gives per rank (shared dofs
  // counted once by assembly of f). Serial here; parallel pending.
  // The unrelaxed start.
  src_->ComputeSources(data, 0, false, f_u, zeta);
  dJ += f_u * op_->Lambda(0);
  op_->Coupling(op_->Lambda(0), cb);
  for (int k = 0; k < K; k++) {
    if (zeta[k].Size() > 0) dJ += zeta[k] * cb;
  }
  for (int n = 0; n < N; n++) {
    src_->PrepareStep(data, n, n + 1);
    src_->ComputeSources(data, n + 1, true, f_u, zeta);
    dJ += f_u * op_->Lambda(n + 1);
    op_->Coupling(op_->Lambda(n + 1), cb);
    for (int k = 0; k < K; k++) {
      if (zeta[k].Size() > 0) dJ += zeta[k] * cb;
      const Vector& s = src_->StateSource(k);
      if (s.Size() > 0) {
        Vector mu_k = op_->Branch(op_->Mu(n + 1), k);
        dJ += s * mu_k;
      }
    }
  }
  return dJ;
}

}  // namespace AdGIA
