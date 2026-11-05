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
  uniform_.clear();
  omega_.clear();
}

void TangentTrajectory::Record(real_t t, const Vector& m,
                               LinearQuasiStaticProblem& twin) {
  t_.push_back(t);
  m_.emplace_back(m);
  const GridFunction& u = twin.Displacement();
  u_.push_back(detail::MakeGridFunction(
      const_cast<FiniteElementSpace*>(u.FESpace())));
  *u_.back() = u;
  real_t uniform = 0.0;
  Vector omega;
  if (auto* g = dynamic_cast<LinearQuasiStaticMixedSelfGravitatingProblem*>(
          &twin)) {
    const GridFunction& phi = g->Potential();
    phi_.push_back(detail::MakeGridFunction(
        const_cast<FiniteElementSpace*>(phi.FESpace())));
    *phi_.back() = phi;
    uniform = g->UniformPotentialTerm();
    omega = g->AngularVelocity();
  }
  uniform_.push_back(uniform);
  omega_.emplace_back(omega);
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
  dir_ = dir;
  born_ = std::make_unique<ModuliBornForce>(problem_.DisplacementSpace(), dir, K);
  born_eff_.reset();
  born_eff_dt_ = -1.0;
  dmu_.assign(K, Vector());
  ditau_.assign(K, Vector());
  has_dmu_.assign(K, false);
  has_ditau_.assign(K, false);
  s_.assign(K, Vector());
  known_.assign(K, Vector());
  dbeta_.clear();
  for (int k = 0; k < K; k++) {
    dbeta_.push_back(detail::MakeGridFunction(sfes_.get()));
    *dbeta_[k] = 0.0;
  }
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

void TangentViscoelasticOperator::BuildEffectiveBorn() {
  // delta mu_eff = delta mu_inf + sum_k [beta_k delta mu_k + delta beta_k mu_k],
  // with beta_k the operator's own weight coefficients (the ones the
  // forward's effective stiffness is assembled with).
  eff_owned_.clear();
  born_eff_.reset();
  const int K = NumBranches();
  const int dim = problem_.DisplacementSpace().GetMesh()->Dimension();
  Coefficient* dmu = dir_.mu_longterm;
  auto add = [&](Coefficient* term) {
    if (!term) return;
    if (!dmu) {
      dmu = term;
    } else {
      eff_owned_.push_back(std::make_unique<SumCoefficient>(*dmu, *term));
      dmu = eff_owned_.back().get();
    }
  };
  for (int k = 0; k < K; k++) {
    if (dir_.Mu(k)) {
      eff_owned_.push_back(
          std::make_unique<ProductCoefficient>(*beta_ptrs_[k], *dir_.Mu(k)));
      add(eff_owned_.back().get());
    }
    if (has_ditau_[k]) {
      eff_owned_.push_back(
          std::make_unique<GridFunctionCoefficient>(dbeta_[k].get()));
      Coefficient* dbeta_c = eff_owned_.back().get();
      eff_owned_.push_back(std::make_unique<ProductCoefficient>(
          *dbeta_c, problem_.Rheology().BranchShearModulus(k)));
      add(eff_owned_.back().get());
    }
  }
  if (!dir_.kappa && !dmu) {
    born_eff_dt_ = step_dt_;
    return;
  }
  born_eff_ = detail::MakeBilinearForm(&problem_.DisplacementSpace());
  if (dir_.kappa) {
    born_eff_->AddDomainIntegrator(
        new ElasticityIntegrator(*dir_.kappa, 1.0, 0.0));
  }
  if (dmu) {
    born_eff_->AddDomainIntegrator(
        new ElasticityIntegrator(*dmu, -2.0 / dim, 1.0));
  }
  born_eff_->Assemble();
  born_eff_->Finalize();
  born_eff_dt_ = step_dt_;
}

void TangentViscoelasticOperator::ComputeSources(TangentData& data, int n,
                                                 bool in_step, Vector& f_u,
                                                 std::vector<Vector>& zeta) {
  data_ = &data;
  f_u.SetSize(problem_.DisplacementSpace().GetVSize());
  f_u = 0.0;
  zeta.assign(NumBranches(), Vector());
  if (in_step) {
    // The effective solve: the Born force of the moduli and of the
    // weights on u^{n+1}, and the step's known-part branch sources.
    if (born_eff_dt_ != step_dt_) {
      BuildEffectiveBorn();
    }
    if (born_eff_) {
      const GridFunction& u = data.Displacement(n);
      bbuf_.SetSize(u.Size());
      born_eff_->Mult(u, bbuf_);
      f_u.Add(-1.0, bbuf_);
    }
    for (int k = 0; k < NumBranches(); k++) {
      if (known_[k].Size() > 0) {
        zeta[k] = known_[k];
      }
    }
    return;
  }
  // An unrelaxed solve: the Born force of the moduli on the forward
  // displacement and the branch-modulus perturbation on the forward
  // internal variables at this node.
  if (born_ && !born_->Empty()) {
    born_->AddTo(data.Displacement(n), f_u);
  }
  for (int k = 0; k < NumBranches(); k++) {
    const int nd = nodes_[k].Size();
    if (has_dmu_[k]) {
      Vector mk = Branch(data.State(n), k);
      zeta[k].SetSize(nd * nc_);
      for (int c = 0; c < nc_; c++) {
        for (int q = 0; q < nd; q++) {
          zeta[k][c * nd + q] = 2.0 * dmu_[k][q] * mk[c * nd + q];
        }
      }
    }
  }
}

void TangentViscoelasticOperator::PrepareStep(TangentData& data, int n0,
                                              int n1) {
  data_ = &data;
  SetEffectiveModulus(data.Time(n1) - data.Time(n0),
                      Scheme::ExponentialTrapezoid);
  RelaxationSource(data, n0, n1);
}

void TangentViscoelasticOperator::Inject(real_t) {
  std::vector<Vector> zeta;
  ComputeSources(*data_, node_, in_step_, fbuf_, zeta);
  problem_.AddForce(fbuf_);
  for (int k = 0; k < NumBranches(); k++) {
    if (zeta[k].Size() > 0) {
      AddCoupledForce(k, zeta[k]);
    }
  }
}

void TangentViscoelasticOperator::RelaxationSource(TangentData& data, int n0,
                                                   int n1) {
  const real_t dt = data.Time(n1) - data.Time(n0);
  data.Strain(n0, dn_);
  data.Strain(n1, dn1_);
  const bool new_dt = dt != step_dt_ || born_eff_dt_ < 0.0;
  step_dt_ = dt;
  for (int k = 0; k < NumBranches(); k++) {
    const Array<int>& nodes = nodes_[k];
    const int nd = nodes.Size();
    s_[k].SetSize(nd * nc_);
    s_[k] = 0.0;
    known_[k].SetSize(0);
    if (!has_ditau_[k] && !has_dmu_[k]) {
      continue;
    }
    Vector mk = Branch(data.State(n0), k);
    // The known part of the step's branch force, 2 dmu (e m^n + a d^n)
    // + C_k (de m^n + da d^n) dh, and the state source s_k.
    Vector rel(nd * nc_);
    rel = 0.0;
    known_[k].SetSize(nd * nc_);
    known_[k] = 0.0;
    for (int q = 0; q < nd; q++) {
      const real_t h = dt * itau0_[k][q];
      real_t e, alpha, beta, de, da, db;
      detail::ExponentialTrapezoidWeights(h, e, alpha, beta);
      TrapezoidWeightDerivatives(h, de, da, db);
      const real_t dh = has_ditau_[k] ? dt * ditau_[k][q] : 0.0;
      const int p = nodes[q];
      if (has_ditau_[k] && new_dt) {
        // delta beta_k = -delta b_k = -b_k' delta h_k, nodal.
        (*dbeta_[k])[p] = -db * dh;
      }
      for (int c = 0; c < nc_; c++) {
        const int i = c * nd + q, full = c * nd_ + p;
        s_[k][i] = dh * (de * mk[i] + da * dn_[full] + db * dn1_[full]);
        rel[i] = dh * (de * mk[i] + da * dn_[full]);
        if (has_dmu_[k]) {
          known_[k][i] = 2.0 * dmu_[k][q] * (e * mk[i] + alpha * dn_[full]);
        }
      }
    }
    if (has_ditau_[k]) {
      ApplyBranchModulus(k, rel, zbuf_);
      known_[k] += zbuf_;
    }
  }
  if (new_dt) {
    born_eff_dt_ = -1.0;  // rebuild the effective Born form for this dt
  }
}

bool TangentViscoelasticOperator::Start(const Vector& dm, real_t t,
                                        TangentData& data, int node) {
  MFEM_VERIFY(born_ || !dmu_.empty(), "TangentViscoelasticOperator: call "
                                       "SetDirection() first.");
  data_ = &data;
  node_ = node;
  in_step_ = false;
  InvalidateDisplacement();
  return SolveElastic(dm, t);
}

bool TangentViscoelasticOperator::Step(Vector& dm, real_t& t, TangentData& data,
                                       int n0, int n1) {
  data_ = &data;
  // The step reuses the start's displacement; if the state was not
  // started on this data (or was changed), start it here.
  if (!CacheMatches(dm, t)) {
    if (!Start(dm, t, data, n0)) {
      return false;
    }
  }
  const real_t dt = data.Time(n1) - data.Time(n0);
  RelaxationSource(data, n0, n1);
  node_ = n1;
  in_step_ = true;
  ExponentialTrapezoidStep(dm, t, dt);
  in_step_ = false;
  // Complete the state with the relaxation source: the solve of the step
  // already carried it, so the displacement is consistent with the
  // completed state.
  for (int k = 0; k < NumBranches(); k++) {
    if (has_ditau_[k]) {
      Vector mk = Branch(dm, k);
      mk += s_[k];
    }
  }
  t = data.Time(n1);  // no drift against the data's grid
  SetTime(t);
  UpdateCache(dm, t);
  return true;
}

bool TangentViscoelasticOperator::Run(ForwardTrajectory& forward,
                                      TangentTrajectory& out) {
  ForwardTrajectoryData data(forward);
  out.Clear();
  const int N = forward.Size() - 1;
  MFEM_VERIFY(N >= 0, "TangentViscoelasticOperator: an empty trajectory.");
  Vector dm(Height());
  dm = 0.0;
  real_t t = forward.Time(0);
  if (!Start(dm, t, data, 0)) {
    return false;
  }
  out.Record(t, dm, problem_);
  for (int n = 0; n < N; n++) {
    if (!Step(dm, t, data, n, n + 1)) {
      return false;
    }
    out.Record(t, dm, problem_);
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
  traj_.Record(0.0, none, twin_);
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
    // The border data: scalars replicated on every rank.
    dJ += force.ObservationUniform(i) * traj_.Uniform(n);
    const Vector& om = force.ObservationOmega(i);
    if (om.Size() > 0) {
      MFEM_VERIFY(traj_.Omega(n).Size() == om.Size(),
                  "TangentLinearSolver::Pair: an omega observation on a "
                  "twin without rotation.");
      dJ += om * traj_.Omega(n);
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
