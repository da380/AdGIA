#include "AdGIA/adjoint.hpp"

#include <algorithm>
#include <cmath>

#include "AdGIA/detail/fem_factory.hpp"
#include "AdGIA/index.hpp"

namespace AdGIA {

using namespace mfem;

namespace {

real_t GlobalSum(const FiniteElementSpace& fes, real_t local) {
#ifdef MFEM_USE_MPI
  if (auto* pfes = dynamic_cast<const ParFiniteElementSpace*>(&fes)) {
    real_t global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                  pfes->GetComm());
    return global;
  }
#endif
  return local;
}

// Reconstruct a symmetric matrix from component values in the library
// order (index.hpp); a trace-free component vector (one short of the full
// count) closes the trace through the dropped diagonal.
void ReconstructSymmetric(int dim, const Vector& a, DenseMatrix& M) {
  const int ns = SymmetricComponentOrder::Count(dim);
  MFEM_ASSERT(a.Size() == ns || a.Size() == ns - 1,
              "ReconstructSymmetric: component count does not match the "
              "dimension.");
  M.SetSize(dim);
  for (int k = 0; k < dim; k++) {
    for (int j = k; j < dim; j++) {
      const int o = SymmetricComponentOrder::Offset(dim, j, k);
      const real_t v = o < a.Size() ? a[o] : 0.0;
      M(j, k) = v;
      M(k, j) = v;
    }
  }
  if (a.Size() == ns - 1) {
    real_t tr = 0.0;
    for (int i = 0; i + 1 < dim; i++) {
      tr += M(i, i);
    }
    M(dim - 1, dim - 1) = -tr;
  }
}

real_t FrobeniusDot(const DenseMatrix& A, const DenseMatrix& B) {
  real_t s = 0.0;
  for (int j = 0; j < A.Height(); j++) {
    for (int i = 0; i < A.Height(); i++) {
      s += A(i, j) * B(i, j);
    }
  }
  return s;
}

}  // namespace

// --- AdjointForce ------------------------------------------------------------

void AdjointForce::Insert(real_t t, Observation&& o) {
  o.t = t;
  auto it = std::upper_bound(
      obs_.begin(), obs_.end(), t,
      [](real_t a, const Observation& b) { return a < b.t; });
  obs_.insert(it, std::move(o));
}

void AdjointForce::AddObservation(real_t t, const Vector& dual) {
  Observation o;
  o.du = dual;
  Insert(t, std::move(o));
}

void AdjointForce::AddPotentialObservation(real_t t, const Vector& dual) {
  Observation o;
  o.dphi = dual;
  Insert(t, std::move(o));
}

void AdjointForce::AddObservation(real_t t, const AdjointLoad& load) {
  Observation o;
  if (load.u) {
    o.du = *load.u;
  }
  if (load.phi) {
    o.dphi = *load.phi;
  }
  o.uniform = load.uniform;
  if (load.omega) {
    o.omega = *load.omega;
  }
  Insert(t, std::move(o));
}

void AdjointForce::AddContinuous(
    std::function<void(real_t, Vector&)> filler) {
  continuous_.push_back(std::move(filler));
}

void AdjointForce::AddContinuousPotential(
    std::function<void(real_t, Vector&)> filler) {
  continuous_phi_.push_back(std::move(filler));
}

void AdjointForce::EvalContinuous(real_t t, Vector& f) const {
  for (const auto& filler : continuous_) {
    filler(t, f);
  }
}

void AdjointForce::EvalContinuousPotential(real_t t, Vector& f) const {
  for (const auto& filler : continuous_phi_) {
    filler(t, f);
  }
}

// --- ForwardTrajectory -------------------------------------------------------

ForwardTrajectory::ForwardTrajectory(ViscoelasticOperator& op,
                                     bool keep_fields)
    : op_(op), keep_fields_(keep_fields) {}

void ForwardTrajectory::Record(const Vector& m, real_t t) {
  MFEM_VERIFY(op_.SolveElastic(m, t),
              "ForwardTrajectory::Record: forward solve failed.");
  t_.push_back(t);
  m_.emplace_back(m);
  if (keep_fields_) {
    u_.push_back(detail::MakeGridFunction(&op_.Problem().DisplacementSpace()));
    *u_.back() = op_.Problem().Displacement();
  }
}

const GridFunction& ForwardTrajectory::Displacement(int n) {
  if (keep_fields_) {
    return *u_[n];
  }
  MFEM_VERIFY(op_.SolveElastic(m_[n], t_[n]),
              "ForwardTrajectory::Displacement: forward re-solve failed.");
  return op_.Problem().Displacement();
}

void ForwardTrajectory::Strain(int n, Vector& d) {
  op_.ComputeStrain(Displacement(n), d);
}

// --- IsotropicKernelDensity --------------------------------------------------

IsotropicKernelDensity::IsotropicKernelDensity(Parameter parameter,
                                               Coefficient* tau)
    : parameter_(parameter), tau_(tau) {
  MFEM_VERIFY(parameter_ != Parameter::Eta || tau_,
              "IsotropicKernelDensity: the Eta density needs tau.");
}

void IsotropicKernelDensity::SetPair(const GridFunction& u,
                                     const GridFunction& udag,
                                     const GridFunction* d,
                                     const GridFunction* m,
                                     const GridFunction* mdag) {
  u_ = &u;
  udag_ = &udag;
  d_ = d;
  m_ = m;
  mdag_ = mdag;
}

real_t IsotropicKernelDensity::Eval(ElementTransformation& T,
                                    const IntegrationPoint& ip) {
  T.SetIntPoint(&ip);
  const int dim = T.GetSpaceDim();
  if (parameter_ == Parameter::Eta) {
    // -(2/tau_k) (d - m_k) : mdag_k; zero without the viscous factors.
    if (!d_ || !mdag_) {
      return 0.0;
    }
    d_->GetVectorValue(T, ip, av_);
    if (m_) {
      m_->GetVectorValue(T, ip, bv_);
      av_ -= bv_;
    }
    mdag_->GetVectorValue(T, ip, bv_);
    ReconstructSymmetric(dim, av_, A_);
    ReconstructSymmetric(dim, bv_, B_);
    return -(2.0 / tau_->Eval(T, ip)) * FrobeniusDot(A_, B_);
  }
  u_->GetVectorGradient(T, Du_);
  udag_->GetVectorGradient(T, Dv_);
  real_t divu = 0.0, divv = 0.0;
  for (int i = 0; i < dim; i++) {
    divu += Du_(i, i);
    divv += Dv_(i, i);
  }
  if (parameter_ == Parameter::Kappa) {
    return -divu * divv;
  }
  // The elastic pairing dev eps(u) : dev eps(udag), pointwise from the
  // displacement gradients (that is how the shear moduli enter the
  // discrete elastic form).
  real_t ee = 0.0;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      ee += 0.25 * (Du_(i, j) + Du_(j, i)) * (Dv_(i, j) + Dv_(j, i));
    }
  }
  const real_t dev_pair = ee - divu * divv / dim;
  if (parameter_ == Parameter::MuLongTerm) {
    // mu_inf enters the stiffness alone.
    return -2.0 * dev_pair;
  }
  // Mu (branch k, fixed eta_k): -2 [ dev_pair - m_k : dev eps(udag)
  //                                  - (d - m_k) : mdag_k ].
  real_t m_pair = 0.0;
  if (m_) {
    // m trace-free: m : dev eps(udag) = m : eps(udag).
    m_->GetVectorValue(T, ip, av_);
    ReconstructSymmetric(dim, av_, A_);
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        m_pair += A_(i, j) * 0.5 * (Dv_(i, j) + Dv_(j, i));
      }
    }
  }
  real_t visc = 0.0;
  if (d_ && mdag_) {
    d_->GetVectorValue(T, ip, av_);
    if (m_) {
      m_->GetVectorValue(T, ip, bv_);
      av_ -= bv_;
    }
    mdag_->GetVectorValue(T, ip, bv_);
    ReconstructSymmetric(dim, av_, A_);
    ReconstructSymmetric(dim, bv_, B_);
    visc = FrobeniusDot(A_, B_);
  }
  return -2.0 * (dev_pair - m_pair - visc);
}

// --- IsotropicKernel ---------------------------------------------------------

IsotropicKernel::IsotropicKernel(Parameter parameter,
                                 FiniteElementSpace& sfes, Coefficient* tau,
                                 int branch)
    : density_(parameter, tau), sfes_(&sfes), branch_(branch) {}

int IsotropicKernel::AddDirection(Coefficient& delta) {
  directions_.push_back(&delta);
  values_.push_back(0.0);
  return static_cast<int>(directions_.size()) - 1;
}

real_t IsotropicKernel::Directional(int i) const {
  return GlobalSum(*sfes_, values_[i]);
}

void IsotropicKernel::SetDualSpace(FiniteElementSpace& fes,
                                   const Array<int>* attr_marker) {
  MFEM_VERIFY(fes.GetMesh() == sfes_->GetMesh(),
              "IsotropicKernel::SetDualSpace: the dual space must live on "
              "the problem's mesh (buffered metrics transfer downstream).");
  dual_fes_ = &fes;
  if (attr_marker) {
    attr_marker->Copy(dual_marker_);
  } else {
    dual_marker_.DeleteAll();
  }
  dual_.SetSize(fes.GetVSize());
  dual_ = 0.0;
}

void IsotropicKernel::TrueDual(Vector& j) const {
  MFEM_VERIFY(dual_fes_, "IsotropicKernel::TrueDual: no dual space set.");
  j.SetSize(dual_fes_->GetTrueVSize());
  if (const Operator* P = dual_fes_->GetProlongationMatrix()) {
    P->MultTranspose(dual_, j);
  } else {
    j = dual_;
  }
}

void IsotropicKernel::SetField(FiniteElementSpace& fes) {
  field_ = detail::MakeGridFunction(&fes);
  field_step_ = detail::MakeGridFunction(&fes);
  *field_ = 0.0;
}

real_t IsotropicKernel::Integrate(Coefficient& c) const {
  auto lf = detail::MakeLinearForm(sfes_);
  lf->AddDomainIntegrator(new DomainLFIntegrator(c));
  lf->Assemble();
  // The H1 nodal basis is a partition of unity, so the assembled sum is
  // the integral of c at the forms' quadrature (local; reduced by the
  // caller).
  return lf->Sum();
}

void IsotropicKernel::AddPair(real_t w, const AdjointPair& p) {
  const GridFunction* m =
      branch_ < static_cast<int>(p.m.size()) ? p.m[branch_] : nullptr;
  const GridFunction* mdag =
      branch_ < static_cast<int>(p.mdag.size()) ? p.mdag[branch_] : nullptr;
  if (density_.Param() == Parameter::Eta && (!p.d || !mdag)) {
    return;  // the Eta integrand is bounded: no impulsive or static part
  }
  density_.SetPair(p.u, p.udag, p.d, m, mdag);
  for (std::size_t i = 0; i < directions_.size(); i++) {
    auto weighted = ProductCoefficient(density_, *directions_[i]);
    values_[i] += w * Integrate(weighted);
  }
  if (dual_fes_) {
    auto lf = detail::MakeLinearForm(dual_fes_);
    if (dual_marker_.Size() > 0) {
      lf->AddDomainIntegrator(new DomainLFIntegrator(density_), dual_marker_);
    } else {
      lf->AddDomainIntegrator(new DomainLFIntegrator(density_));
    }
    lf->Assemble();
    dual_.Add(w, *lf);
  }
  if (field_) {
    field_step_->ProjectCoefficient(density_);
    field_->Add(w, *field_step_);
  }
}

// --- SeaLevelLoadKernel ------------------------------------------------------

SeaLevelLoadKernel::SeaLevelLoadKernel(
    LinearQuasiStaticMixedSelfGravitatingProblem& adjoint_problem,
    SeaLevelOperator& sea, VectorCoefficient& grad_phi0, real_t g_surf,
    bool rotation)
    : problem_(adjoint_problem),
      sea_(sea),
      grad_phi0_(grad_phi0),
      g_surf_(g_surf),
      rotation_(rotation) {
  sl_ = detail::MakeGridFunction(&sea_.SurfaceSpace());
}

int SeaLevelLoadKernel::AddDirection(Coefficient& dsigma) {
  directions_.push_back(&dsigma);
  values_.push_back(0.0);
  return static_cast<int>(directions_.size()) - 1;
}

real_t SeaLevelLoadKernel::Directional(int i) const {
  return values_[i];  // SurfaceIntegral is already global
}

void SeaLevelLoadKernel::EnableQuelled(Coefficient& C, Coefficient& C0,
                                       real_t rho_i) {
  C_ = &C;
  C0_ = &C0;
  rho_i_ = rho_i;
  i0_kernel_ = detail::MakeGridFunction(&sea_.SurfaceSpace());
  *i0_kernel_ = 0.0;
}

void SeaLevelLoadKernel::ReadAdjointSeaLevel() {
  Coefficient* psi =
      rotation_ ? &problem_.SolutionCentrifugalPotential() : nullptr;
  sea_.SeaLevelChangeFrom(problem_.Displacement(), grad_phi0_,
                          problem_.Potential(),
                          problem_.UniformPotentialTerm(), psi);
  *sl_ = sea_.SeaLevelChangeField();
}

void SeaLevelLoadKernel::AddPair(real_t w, const AdjointPair& p) {
  MFEM_VERIFY(!finalized_, "SeaLevelLoadKernel: already finalized.");
  ReadAdjointSeaLevel();
  GridFunctionCoefficient slc(sl_.get());

  // Directional pairings: delta J = +g int dsigma(t) SL1+ dS in the
  // LIBRARY's load/readout conventions (the FD gate fixed the sign;
  // the paper's eq. 63 carries -g in its own conventions), the
  // impulses with weight one (the fingerprint solves' singular parts).
  for (std::size_t i = 0; i < directions_.size(); i++) {
    directions_[i]->SetTime(p.t);
    ProductCoefficient ds_sl(*directions_[i], slc);
    values_[i] += w * g_surf_ * sea_.SurfaceIntegral(ds_sl);
  }

  if (!C_) {
    return;
  }
  C_->SetTime(p.t);
  // D_I0 += -w rho_i g (C(t) - C0) SL1+ (eq. 84's second kernel in the
  // library's sign conventions; zero with a frozen ocean function).
  {
    SumCoefficient dC(*C_, *C0_, 1.0, -1.0);
    ProductCoefficient dC_sl(dC, slc);
    GridFunction tmp(&sea_.SurfaceSpace());
    tmp.ProjectCoefficient(dC_sl);
    i0_kernel_->Add(-w * rho_i_ * g_surf_, tmp);
  }
  // The Idot stack: per node the integrand +rho_i g (1 - C(t)) SL1+
  // (library signs); an impulsive pairing is an ATOM — a finite jump
  // of D_Idot — folded into the node's field with weight one.
  {
    SumCoefficient oneC(1.0, *C_, 1.0, -1.0);
    ProductCoefficient i_sl(oneC, slc);
    GridFunction tmp(&sea_.SurfaceSpace());
    tmp.ProjectCoefficient(i_sl);
    tmp *= rho_i_ * g_surf_;
    if (p.impulse) {
      atoms_t_.push_back(p.t);
      atoms_.push_back(std::make_unique<GridFunction>(tmp));
      return;
    }
    if (!times_.empty() && times_.back() == p.t) {
      *idot_[times_.size() - 1] = tmp;  // the node's latest smooth value
    } else {
      times_.push_back(p.t);
      idot_.push_back(std::make_unique<GridFunction>(tmp));
    }
  }
}

void SeaLevelLoadKernel::Finalize() {
  if (finalized_ || !C_) {
    finalized_ = true;
    return;
  }
  // idot_[n] currently holds the integrand f(t_n); arrival order is
  // DESCENDING forward time. Replace it by the backward cumulative
  // trapezoid D(t_n) = int_{t_n}^{t_1} f ds, plus every atom at or
  // after t_n.
  const int K = static_cast<int>(times_.size());
  GridFunction cum(&sea_.SurfaceSpace());
  cum = 0.0;
  std::unique_ptr<GridFunction> prev;
  for (int k = 0; k < K; k++) {
    if (prev) {
      const real_t dt = times_[k - 1] - times_[k];
      cum.Add(0.5 * dt, *prev);
      cum.Add(0.5 * dt, *idot_[k]);
    }
    prev = std::make_unique<GridFunction>(*idot_[k]);
    *idot_[k] = cum;
    for (std::size_t a = 0; a < atoms_.size(); a++) {
      if (atoms_t_[a] >= times_[k]) {
        idot_[k]->Add(1.0, *atoms_[a]);
      }
    }
  }
  finalized_ = true;
}

// --- AdjointSolver -----------------------------------------------------------

AdjointSolver::AdjointSolver(LinearQuasiStaticProblem& adjoint_problem,
                             const GridFunction& forward_displacement)
    : problem_(&adjoint_problem), u_forward_(&forward_displacement) {}

AdjointSolver::AdjointSolver(ViscoelasticOperator& adjoint_op,
                             ForwardTrajectory& forward)
    : op_(&adjoint_op), forward_(&forward) {}

bool AdjointSolver::Run() {
  MFEM_VERIFY(force_, "AdjointSolver::Run: no adjoint force set.");
  return op_ ? RunEvolution() : RunStatic();
}

void AdjointSolver::Pair(real_t w, const AdjointPair& p) {
  for (auto* kernel : kernels_) {
    kernel->AddPair(w, p);
  }
}

void AdjointSolver::ResolveSlots(LinearQuasiStaticProblem& problem) {
  bool needs_phi = force_->HasContinuousPotential();
  bool needs_border = false;
  for (int i = 0; i < force_->NumObservations(); i++) {
    needs_phi = needs_phi || force_->ObservationOnPotential(i);
    needs_border = needs_border || force_->ObservationOnBorder(i);
  }
  if (needs_phi) {
    phi_slot_ = dynamic_cast<PotentialForceSlot*>(&problem);
    MFEM_VERIFY(phi_slot_,
                "AdjointSolver: the force carries potential-row terms but "
                "the problem exposes no PotentialForceSlot.");
  }
  if (needs_border) {
    border_slot_ = dynamic_cast<BorderForceSlot*>(&problem);
    MFEM_VERIFY(border_slot_,
                "AdjointSolver: the force carries border-row terms but the "
                "problem exposes no BorderForceSlot.");
  }
}

void AdjointSolver::ApplyObservation(int obs) {
  auto& prob = problem_ ? *problem_ : op_->Problem();
  const Vector& du = force_->ObservationDualU(obs);
  if (du.Size() > 0) {
    prob.AddForce(du);
  }
  const Vector& dphi = force_->ObservationDualPhi(obs);
  if (dphi.Size() > 0) {
    phi_slot_->AddPotentialForce(dphi);
  }
  if (force_->ObservationOnBorder(obs)) {
    const Vector& om = force_->ObservationOmega(obs);
    border_slot_->AddBorderForce(force_->ObservationUniform(obs),
                                 om.Size() > 0 ? &om : nullptr);
  }
}

bool AdjointSolver::RunStatic() {
  MFEM_VERIFY(!force_->HasContinuous() && !force_->HasContinuousPotential(),
              "AdjointSolver: a static adjoint takes point terms only.");
  ResolveSlots(*problem_);
  for (int i = 0; i < force_->NumObservations(); i++) {
    problem_->AssembleForce(force_->ObservationTime(i));
    ApplyObservation(i);
    if (!problem_->Solve()) {
      return false;
    }
    Pair(1.0, AdjointPair{*u_forward_, nullptr, {},
                          problem_->Displacement(), {},
                          force_->ObservationTime(i), true, problem_});
  }
  return true;
}

bool AdjointSolver::ImpulseSolve(int obs, real_t s) {
  // The impulsive (singular) response: the observation dual on the
  // unrelaxed elastic operator alone — no branch forces, and no
  // continuous loads (they are bounded and carry no delta part). The
  // time sync runs first so the operator carries this time's
  // coefficients (the playback's fingerprint solve).
  op_->UseUnrelaxedOperator();
  auto& prob = op_->Problem();
  prob.AssembleForce(s);
  if (time_sync_) {
    time_sync_(forward_->Time(forward_->Size() - 1) - s);
  }
  ApplyObservation(obs);
  return prob.Solve();
}

void AdjointSolver::ApplyJump(const GridFunction& udag_g, Vector& mdag) {
  // [m+] = (1/tau) eps-projection of the impulsive response, per branch at
  // its nodes (adjoint_note.md, eq. 80 of the paper).
  op_->ComputeStrain(udag_g, strain_);
  const int nd_full = op_->InternalScalarSpace().GetVSize();
  for (int b = 0; b < op_->NumBranches(); b++) {
    const Vector& itau = op_->InverseRelaxationTimes(b);
    const Array<int>& nodes = op_->BranchNodes(b);
    const int nd = nodes.Size();
    const int off = op_->BranchOffset(b);
    for (int c = 0; c < op_->NumComponents(); c++) {
      for (int q = 0; q < nd; q++) {
        mdag[off + c * nd + q] += itau[q] * strain_[c * nd_full + nodes[q]];
      }
    }
  }
}

bool AdjointSolver::RunEvolution() {
  auto& op = *op_;
  auto& fwd = *forward_;
  auto& prob = op.Problem();
  const int N = fwd.Size() - 1;
  MFEM_VERIFY(N >= 1, "AdjointSolver: the trajectory holds fewer than two "
                      "states.");
  MFEM_VERIFY(op.IsLinear(),
              "AdjointSolver: the adjoint evolution assumes a linear "
              "rheology (state-dependent laws are a different adjoint).");
  if (!kernels_.empty()) {
    for (int b = 0; b < op.NumBranches(); b++) {
      MFEM_VERIFY(op.NumBranchNodes(b) ==
                      op.InternalScalarSpace().GetVSize(),
                  "AdjointSolver: kernel accumulators support whole-mesh "
                  "branches for now (region-restricted branches need "
                  "marker-restricted kernel quadrature).");
    }
  }

  // Observations onto grid nodes.
  const real_t t1 = fwd.Time(N);
  const real_t span = t1 - fwd.Time(0);
  const real_t tol = 1e-8 * (span > 0.0 ? span : 1.0);
  std::vector<std::vector<int>> node_obs(N + 1);
  for (int i = 0; i < force_->NumObservations(); i++) {
    const real_t t = force_->ObservationTime(i);
    int best = 0;
    for (int n = 1; n <= N; n++) {
      if (std::abs(fwd.Time(n) - t) < std::abs(fwd.Time(best) - t)) {
        best = n;
      }
    }
    MFEM_VERIFY(std::abs(fwd.Time(best) - t) <= tol,
                "AdjointSolver: observation times must lie on the forward "
                "trajectory's grid.");
    node_obs[best].push_back(i);
  }

  // Continuous terms and the time sync through the force hook,
  // reversed time mapped back (the sync runs FIRST, so refreshed
  // operators see this time's coefficients before any dual lands).
  ResolveSlots(prob);
  const bool has_continuous =
      force_->HasContinuous() || force_->HasContinuousPotential();
  const bool hook_needed = has_continuous || bool(time_sync_);
  if (hook_needed) {
    op.SetForceHook([this, t1, &prob](real_t s) {
      if (time_sync_) {
        time_sync_(t1 - s);
      }
      if (force_->HasContinuous()) {
        fbuf_.SetSize(prob.DisplacementSpace().GetVSize());
        fbuf_ = 0.0;
        force_->EvalContinuous(t1 - s, fbuf_);
        prob.AddForce(fbuf_);
      }
      if (force_->HasContinuousPotential()) {
        fbuf_phi_.SetSize(phi_slot_->PotentialForceSpace().GetVSize());
        fbuf_phi_ = 0.0;
        force_->EvalContinuousPotential(t1 - s, fbuf_phi_);
        phi_slot_->AddPotentialForce(fbuf_phi_);
      }
    });
  }

  // Pairing fields: wrappers on the internal spaces, reassigned per node;
  // one forward and one adjoint internal field per Prony branch.
  auto& fop = fwd.Op();
  const int nb = op.NumBranches();
  auto d_f = detail::MakeGridFunction(&fop.InternalVariableSpace());
  std::vector<std::unique_ptr<GridFunction>> m_f(nb), mdag_f(nb);
  std::vector<const GridFunction*> m_ptr(nb), mdag_ptr(nb);
  for (int b = 0; b < nb; b++) {
    m_f[b] = detail::MakeGridFunction(&fop.InternalVariableSpace());
    mdag_f[b] = detail::MakeGridFunction(&op.InternalVariableSpace());
    m_ptr[b] = m_f[b].get();
    mdag_ptr[b] = mdag_f[b].get();
  }
  auto udag_g = detail::MakeGridFunction(&prob.DisplacementSpace());
  Vector dbuf, mbuf;

  ExponentialTrapezoidSolver ode;
  ode.Init(op);
  Vector mdag(op.Height());
  mdag = 0.0;
  real_t s = 0.0;
  // Before the first observation the adjoint state is identically zero
  // unless continuous terms drive it; skip those nodes entirely.
  bool active = has_continuous;
  bool ok = true;

  for (int k = 0; k <= N && ok; k++) {
    const int n = N - k;
    const real_t s_k = t1 - fwd.Time(n);
    const bool has_obs = !node_obs[n].empty();
    const bool arrived_active = active;
    if (!active && !has_obs) {
      s = s_k;
      continue;
    }
    if (active && k > 0) {
      real_t dt = s_k - s;
      ode.Step(mdag, s, dt);
      s = s_k;  // no accumulation drift against the forward grid
    } else {
      s = s_k;
    }
    if (active) {
      ok = op.SolveElastic(mdag, s);
      if (!ok) {
        break;
      }
    }

    // Forward fields at node n and the trapezoid weights of its two
    // subintervals (adjoint_note.md: at an observation the node weight
    // splits across the jump — the forward-right part pairs the arrived
    // state, the forward-left part the post-jump one).
    const GridFunction& u_f = fwd.Displacement(n);
    fwd.Strain(n, dbuf);
    *d_f = dbuf;
    for (int b = 0; b < nb; b++) {
      fop.BranchToFull(fwd.State(n), b, mbuf);
      *m_f[b] = mbuf;
    }
    const real_t wR = n < N ? 0.5 * (fwd.Time(n + 1) - fwd.Time(n)) : 0.0;
    const real_t wL = n > 0 ? 0.5 * (fwd.Time(n) - fwd.Time(n - 1)) : 0.0;

    auto pair_state = [&](real_t w) {
      if (w <= 0.0) {
        return;
      }
      for (int b = 0; b < nb; b++) {
        op.BranchToFull(mdag, b, mbuf);
        *mdag_f[b] = mbuf;
      }
      Pair(w, AdjointPair{u_f, d_f.get(), m_ptr, prob.Displacement(),
                          mdag_ptr, fwd.Time(n), false, &prob});
    };

    if (!has_obs) {
      pair_state(wL + wR);
      continue;
    }
    if (arrived_active) {
      pair_state(wR);
    }
    for (int i : node_obs[n]) {
      ok = ImpulseSolve(i, s);
      if (!ok) {
        break;
      }
      *udag_g = prob.Displacement();
      // The impulsive pairing: the adjoint factor is the purely elastic
      // response; the m+_k are bounded, so mdag is empty here. The
      // problem's state is the impulse solve's (the generalized
      // fingerprint fields), which the sea-level accumulators read.
      Pair(1.0, AdjointPair{u_f, d_f.get(), m_ptr, *udag_g, {},
                            fwd.Time(n), true, &prob});
      ApplyJump(*udag_g, mdag);
    }
    if (!ok) {
      break;
    }
    // The post-jump smooth field (the driver owns the cache invalidation:
    // the impulse solve left the problem's displacement singular).
    op.InvalidateDisplacement();
    ok = op.SolveElastic(mdag, s);
    if (!ok) {
      break;
    }
    active = true;
    pair_state(wL);
  }

  if (hook_needed) {
    op.SetForceHook(nullptr);
  }
  return ok;
}

}  // namespace AdGIA
