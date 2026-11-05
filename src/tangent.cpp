#include "AdGIA/tangent.hpp"

#include <algorithm>
#include <cmath>

#include "AdGIA/detail/fem_factory.hpp"
#include "AdGIA/mixed_problem.hpp"

namespace AdGIA {

using namespace mfem;

// --- TangentDirection ----------------------------------------------------

bool TangentDirection::HasElasticModuli() const {
  if (kappa || mu_longterm) {
    return true;
  }
  return std::any_of(mu.begin(), mu.end(), [](auto* c) { return c != nullptr; });
}

// --- ModuliBornForce ------------------------------------------------------

ModuliBornForce::ModuliBornForce(FiniteElementSpace& fes,
                                 const TangentDirection& dir,
                                 int num_branches) {
  if (!dir.HasElasticModuli()) {
    return;
  }
  const int dim = fes.GetMesh()->Dimension();
  // delta mu_U = delta mu_inf + sum_k delta mu_k, as an owned chain.
  Coefficient* dmu = dir.mu_longterm;
  for (int k = 0; k < num_branches; k++) {
    if (!dir.Mu(k)) {
      continue;
    }
    if (!dmu) {
      dmu = dir.Mu(k);
    } else {
      owned_.push_back(std::make_unique<SumCoefficient>(*dmu, *dir.Mu(k)));
      dmu = owned_.back().get();
    }
  }
  form_ = detail::MakeBilinearForm(&fes);
  // The isotropic split's own integrators (rheology.cpp,
  // AddIsotropicIntegrators), so the rules coincide with the forward's.
  if (dir.kappa) {
    form_->AddDomainIntegrator(new ElasticityIntegrator(*dir.kappa, 1.0, 0.0));
  }
  if (dmu) {
    form_->AddDomainIntegrator(
        new ElasticityIntegrator(*dmu, -2.0 / dim, 1.0));
  }
  form_->Assemble();
  form_->Finalize();
}

void ModuliBornForce::AddTo(const GridFunction& u, Vector& f) {
  if (!form_) {
    return;
  }
  buf_.SetSize(u.Size());
  form_->Mult(u, buf_);  // the local action on the L-vector (parallel: the
                         // rank's partial sums, which AddForce assembles)
  f.Add(-1.0, buf_);
}

// --- TangentTrajectory ----------------------------------------------------

void TangentTrajectory::Clear() {
  t_.clear();
  m_.clear();
  u_.clear();
  phi_.clear();
}

void TangentTrajectory::Record(real_t t, const Vector& m, const GridFunction& u,
                               const GridFunction* phi) {
  t_.push_back(t);
  m_.emplace_back(m);
  u_.push_back(detail::MakeGridFunction(
      const_cast<FiniteElementSpace*>(u.FESpace())));
  *u_.back() = u;
  if (phi) {
    phi_.push_back(detail::MakeGridFunction(
        const_cast<FiniteElementSpace*>(phi->FESpace())));
    *phi_.back() = *phi;
  }
}

// --- TangentViscoelasticOperator -----------------------------------------

namespace {

int InternalOrder(const ViscoelasticOperator& op) {
  return const_cast<ViscoelasticOperator&>(op).InternalScalarSpace().FEColl()
      ->GetOrder();
}

// d/dh of the exponential-trapezoid weights (e, alpha, beta) at h.
void TrapezoidWeightDerivatives(real_t h, real_t& de, real_t& dalpha,
                                real_t& dbeta) {
  const real_t e = std::exp(-h);
  real_t dphi1;  // d/dh (1 - e^{-h}) / h
  if (h < 1e-3) {
    dphi1 = -0.5 + h / 3.0 - h * h / 8.0 + h * h * h / 30.0;
  } else {
    dphi1 = (e * (1.0 + h) - 1.0) / (h * h);
  }
  de = -e;
  dalpha = dphi1 + e;
  dbeta = -dphi1;
}

const GridFunction* PotentialOf(LinearQuasiStaticProblem& p) {
  if (auto* g = dynamic_cast<LinearQuasiStaticMixedSelfGravitatingProblem*>(&p)) {
    return &g->Potential();
  }
  return nullptr;
}

}  // namespace

TangentViscoelasticOperator::TangentViscoelasticOperator(
    LinearQuasiStaticProblem& twin, const ViscoelasticOperator& forward)
    : ViscoelasticOperator(twin, InternalOrder(forward), forward.Map()) {
  MFEM_VERIFY(IsLinear(),
              "TangentViscoelasticOperator: a linear rheology (the tangent "
              "of a state-dependent law is a different operator).");
  MFEM_VERIFY(TraceFree(),
              "TangentViscoelasticOperator: isotropic branches only.");
  MFEM_VERIFY(Height() == forward.Height(),
              "TangentViscoelasticOperator: the twin's internal layout "
              "differs from the forward's (same rheology and spaces?).");
  SetForceHook([this](real_t t) { Inject(t); });
}

void TangentViscoelasticOperator::SetDirection(const TangentDirection& dir) {
  const int K = NumBranches();
  born_ = std::make_unique<ModuliBornForce>(problem_.DisplacementSpace(), dir, K);
  dmu_.assign(K, Vector());
  ditau_.assign(K, Vector());
  has_dmu_.assign(K, false);
  has_ditau_.assign(K, false);
  s_.assign(K, Vector());
  GridFunction sample(sfes_.get());
  for (int k = 0; k < K; k++) {
    const Array<int>& nodes = nodes_[k];
    const int nd = nodes.Size();
    dmu_[k].SetSize(nd);
    ditau_[k].SetSize(nd);
    dmu_[k] = 0.0;
    ditau_[k] = 0.0;
    Vector deta(nd);
    deta = 0.0;
    if (dir.Mu(k)) {
      sample.ProjectCoefficient(*dir.Mu(k));
      for (int q = 0; q < nd; q++) dmu_[k][q] = sample[nodes[q]];
      has_dmu_[k] = true;
    }
    if (dir.Eta(k)) {
      sample.ProjectCoefficient(*dir.Eta(k));
      for (int q = 0; q < nd; q++) deta[q] = sample[nodes[q]];
    }
    if (has_dmu_[k] || dir.Eta(k)) {
      // 1/tau = mu/eta: delta(1/tau) = (1/tau)(dmu/mu - deta/eta), with
      // the nodal mu_k = branch_modulus_/2 and eta_k = mu_k tau_k.
      for (int q = 0; q < nd; q++) {
        const real_t mu = 0.5 * branch_modulus_[k][q];
        const real_t itau = itau0_[k][q];
        const real_t eta = mu / itau;
        ditau_[k][q] = itau * (dmu_[k][q] / mu - deta[q] / eta);
      }
      has_ditau_[k] = true;
    }
  }
}

int TangentViscoelasticOperator::NodeAt(real_t t) const {
  const int N = fwd_->Size();
  int best = 0;
  for (int n = 1; n < N; n++) {
    if (std::abs(fwd_->Time(n) - t) < std::abs(fwd_->Time(best) - t)) {
      best = n;
    }
  }
  const real_t span = fwd_->Time(N - 1) - fwd_->Time(0);
  MFEM_VERIFY(std::abs(fwd_->Time(best) - t) <= 1e-8 * (span > 0 ? span : 1.0),
              "TangentViscoelasticOperator: a solve off the forward grid.");
  return best;
}

void TangentViscoelasticOperator::Inject(real_t t) {
  const int n = in_step_ ? target_ : NodeAt(t);
  // The Born force of the moduli on the forward displacement.
  if (born_ && !born_->Empty()) {
    fbuf_.SetSize(problem_.DisplacementSpace().GetVSize());
    fbuf_ = 0.0;
    born_->AddTo(fwd_->Displacement(n), fbuf_);
    problem_.AddForce(fbuf_);
  }
  // The branch-modulus perturbation on the forward internal variables,
  // and, inside a step, the relaxation source of the step ending at n.
  for (int k = 0; k < NumBranches(); k++) {
    const int nd = nodes_[k].Size();
    if (has_dmu_[k]) {
      Vector mk = Branch(fwd_->State(n), k);
      zbuf_.SetSize(nd * nc_);
      for (int c = 0; c < nc_; c++) {
        for (int q = 0; q < nd; q++) {
          zbuf_[c * nd + q] = 2.0 * dmu_[k][q] * mk[c * nd + q];
        }
      }
      AddCoupledForce(k, zbuf_);
    }
    if (in_step_ && has_ditau_[k]) {
      ApplyBranchModulus(k, s_[k], zbuf_);
      AddCoupledForce(k, zbuf_);
    }
  }
}

void TangentViscoelasticOperator::RelaxationSource(int n, real_t dt) {
  fwd_->Strain(n, dn_);
  fwd_->Strain(n + 1, dn1_);
  for (int k = 0; k < NumBranches(); k++) {
    const Array<int>& nodes = nodes_[k];
    const int nd = nodes.Size();
    s_[k].SetSize(nd * nc_);
    s_[k] = 0.0;
    if (!has_ditau_[k]) {
      continue;
    }
    Vector mk = Branch(fwd_->State(n), k);
    for (int q = 0; q < nd; q++) {
      const real_t h = dt * itau0_[k][q];
      real_t de, da, db;
      TrapezoidWeightDerivatives(h, de, da, db);
      const real_t dh = dt * ditau_[k][q];
      const int p = nodes[q];
      for (int c = 0; c < nc_; c++) {
        s_[k][c * nd + q] = dh * (de * mk[c * nd + q] + da * dn_[c * nd_ + p] +
                                  db * dn1_[c * nd_ + p]);
      }
    }
  }
}

bool TangentViscoelasticOperator::Run(ForwardTrajectory& forward,
                                      TangentTrajectory& out) {
  MFEM_VERIFY(born_ || !dmu_.empty(), "TangentViscoelasticOperator: call "
                                       "SetDirection() first.");
  fwd_ = &forward;
  out.Clear();
  const int N = forward.Size() - 1;
  MFEM_VERIFY(N >= 0, "TangentViscoelasticOperator: an empty trajectory.");
  Vector dm(Height());
  dm = 0.0;
  real_t t = forward.Time(0);
  in_step_ = false;
  InvalidateDisplacement();
  if (!SolveElastic(dm, t)) {
    return false;
  }
  out.Record(t, dm, problem_.Displacement(), PotentialOf(problem_));
  for (int n = 0; n < N; n++) {
    const real_t dt = forward.Time(n + 1) - forward.Time(n);
    RelaxationSource(n, dt);
    target_ = n + 1;
    in_step_ = true;
    ExponentialTrapezoidStep(dm, t, dt);
    in_step_ = false;
    // Complete the state with the relaxation source: the solve of the
    // step already carried it, so the displacement is consistent with
    // the completed state.
    for (int k = 0; k < NumBranches(); k++) {
      if (has_ditau_[k]) {
        Vector mk = Branch(dm, k);
        mk += s_[k];
      }
    }
    t = forward.Time(n + 1);  // no drift against the forward grid
    SetTime(t);
    UpdateCache(dm, t);
    out.Record(t, dm, problem_.Displacement(), PotentialOf(problem_));
  }
  return true;
}

// --- TangentLinearSolver ---------------------------------------------------

TangentLinearSolver::TangentLinearSolver(
    LinearQuasiStaticProblem& twin, const GridFunction& forward_displacement)
    : twin_(twin), u_forward_(&forward_displacement) {}

TangentLinearSolver::TangentLinearSolver(LinearQuasiStaticProblem& twin,
                                         ForwardTrajectory& forward)
    : twin_(twin), forward_(&forward) {
  op_ = std::make_unique<TangentViscoelasticOperator>(twin, forward.Op());
}

bool TangentLinearSolver::Run() {
  if (op_) {
    op_->SetDirection(dir_);
    return op_->Run(*forward_, traj_);
  }
  // Static: one solve with the Born force (the twin's own load is the
  // load direction).
  ModuliBornForce born(twin_.DisplacementSpace(), dir_,
                       twin_.Rheology().NumBranches());
  twin_.AssembleForce(0.0);
  if (!born.Empty()) {
    Vector f(twin_.DisplacementSpace().GetVSize());
    f = 0.0;
    born.AddTo(*u_forward_, f);
    twin_.AddForce(f);
  }
  if (!twin_.Solve()) {
    return false;
  }
  traj_.Clear();
  Vector none;
  traj_.Record(0.0, none, twin_.Displacement(), PotentialOf(twin_));
  return true;
}

const GridFunction& TangentLinearSolver::Displacement() const {
  MFEM_VERIFY(traj_.Size() > 0, "TangentLinearSolver: Run() first.");
  return traj_.Displacement(traj_.Size() - 1);
}

real_t TangentLinearSolver::Dot(const Vector& dual, const GridFunction& u,
                                FiniteElementSpace& fes) const {
#ifdef MFEM_USE_MPI
  if (auto* pfes = dynamic_cast<ParFiniteElementSpace*>(&fes)) {
    Vector dt(pfes->GetTrueVSize()), ut(pfes->GetTrueVSize());
    pfes->GetProlongationMatrix()->MultTranspose(dual, dt);
    u.GetTrueDofs(ut);
    return InnerProduct(pfes->GetComm(), dt, ut);
  }
#endif
  return dual * u;
}

real_t TangentLinearSolver::Pair(const AdjointForce& force) const {
  MFEM_VERIFY(traj_.Size() > 0, "TangentLinearSolver: Run() first.");
  const int N = traj_.Size() - 1;
  const real_t span = traj_.Time(N) - traj_.Time(0);
  const real_t tol = 1e-8 * (span > 0.0 ? span : 1.0);
  FiniteElementSpace& fes_u = twin_.DisplacementSpace();
  auto* phi_slot = dynamic_cast<PotentialForceSlot*>(&twin_);
  real_t dJ = 0.0;
  for (int i = 0; i < force.NumObservations(); i++) {
    MFEM_VERIFY(!force.ObservationOnBorder(i),
                "TangentLinearSolver::Pair: border observations (uniform "
                "term, omega) are not paired yet.");
    const real_t t = force.ObservationTime(i);
    int n = 0;
    for (int k = 1; k <= N; k++) {
      if (std::abs(traj_.Time(k) - t) < std::abs(traj_.Time(n) - t)) n = k;
    }
    MFEM_VERIFY(std::abs(traj_.Time(n) - t) <= tol,
                "TangentLinearSolver::Pair: an observation off the grid.");
    if (force.ObservationDualU(i).Size() > 0) {
      dJ += Dot(force.ObservationDualU(i), traj_.Displacement(n), fes_u);
    }
    if (force.ObservationOnPotential(i)) {
      MFEM_VERIFY(phi_slot && traj_.Potential(n),
                  "TangentLinearSolver::Pair: a potential observation on a "
                  "twin without a potential.");
      dJ += Dot(force.ObservationDualPhi(i), *traj_.Potential(n),
                phi_slot->PotentialForceSpace());
    }
  }
  if (force.HasContinuous() || force.HasContinuousPotential()) {
    Vector f, fphi;
    for (int n = 0; n <= N; n++) {
      const real_t wR = n < N ? 0.5 * (traj_.Time(n + 1) - traj_.Time(n)) : 0.0;
      const real_t wL = n > 0 ? 0.5 * (traj_.Time(n) - traj_.Time(n - 1)) : 0.0;
      const real_t w = wL + wR;
      if (force.HasContinuous()) {
        f.SetSize(fes_u.GetVSize());
        f = 0.0;
        force.EvalContinuous(traj_.Time(n), f);
        dJ += w * Dot(f, traj_.Displacement(n), fes_u);
      }
      if (force.HasContinuousPotential()) {
        MFEM_VERIFY(phi_slot && traj_.Potential(n),
                    "TangentLinearSolver::Pair: a continuous potential term "
                    "on a twin without a potential.");
        fphi.SetSize(phi_slot->PotentialForceSpace().GetVSize());
        fphi = 0.0;
        force.EvalContinuousPotential(traj_.Time(n), fphi);
        dJ += w * Dot(fphi, *traj_.Potential(n),
                      phi_slot->PotentialForceSpace());
      }
    }
  }
  return dJ;
}

}  // namespace AdGIA
