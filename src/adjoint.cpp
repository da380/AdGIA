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

void AdjointForce::Insert(real_t t, const Vector& dual, bool on_potential) {
  auto it = std::upper_bound(
      obs_.begin(), obs_.end(), t,
      [](real_t a, const Observation& b) { return a < b.t; });
  obs_.insert(it, Observation{t, Vector(dual), on_potential});
}

void AdjointForce::AddObservation(real_t t, const Vector& dual) {
  Insert(t, dual, false);
}

void AdjointForce::AddPotentialObservation(real_t t, const Vector& dual) {
  Insert(t, dual, true);
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
    // -(2/tau) (d - m) : mdag; zero without the viscous factors (static
    // pairings and the impulsive terms, where m+ is bounded).
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
  // Mu: -2 [ dev eps(u) : dev eps(udag) - m : dev eps(udag) - (d - m) : mdag ],
  // the dev eps factors pointwise from the displacement gradients (that is
  // how mu enters the discrete elastic form).
  real_t ee = 0.0;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      ee += 0.25 * (Du_(i, j) + Du_(j, i)) * (Dv_(i, j) + Dv_(j, i));
    }
  }
  const real_t dev_pair = ee - divu * divv / dim;
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
                                 FiniteElementSpace& sfes, Coefficient* tau)
    : density_(parameter, tau), sfes_(&sfes) {}

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
  if (density_.Param() == Parameter::Eta && (!p.d || !p.mdag)) {
    return;  // the Eta integrand is bounded: no impulsive or static part
  }
  density_.SetPair(p.u, *&p.udag, p.d, p.m, p.mdag);
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

void AdjointSolver::ResolvePotentialSlot(LinearQuasiStaticProblem& problem) {
  bool needs = force_->HasContinuousPotential();
  for (int i = 0; i < force_->NumObservations() && !needs; i++) {
    needs = force_->ObservationOnPotential(i);
  }
  if (!needs) {
    return;
  }
  phi_slot_ = dynamic_cast<PotentialForceSlot*>(&problem);
  MFEM_VERIFY(phi_slot_,
              "AdjointSolver: the force carries potential-row terms but the "
              "problem exposes no PotentialForceSlot.");
}

bool AdjointSolver::RunStatic() {
  MFEM_VERIFY(!force_->HasContinuous() && !force_->HasContinuousPotential(),
              "AdjointSolver: a static adjoint takes point terms only.");
  ResolvePotentialSlot(*problem_);
  for (int i = 0; i < force_->NumObservations(); i++) {
    problem_->AssembleForce(force_->ObservationTime(i));
    if (force_->ObservationOnPotential(i)) {
      phi_slot_->AddPotentialForce(force_->ObservationDual(i));
    } else {
      problem_->AddForce(force_->ObservationDual(i));
    }
    if (!problem_->Solve()) {
      return false;
    }
    Pair(1.0, AdjointPair{*u_forward_, nullptr, nullptr,
                          problem_->Displacement(), nullptr});
  }
  return true;
}

bool AdjointSolver::ImpulseSolve(int obs, real_t s) {
  // The impulsive (singular) response: the observation dual on the
  // unrelaxed elastic operator alone — no branch forces, and no
  // continuous loads (they are bounded and carry no delta part).
  op_->UseUnrelaxedOperator();
  auto& prob = op_->Problem();
  prob.AssembleForce(s);
  if (force_->ObservationOnPotential(obs)) {
    phi_slot_->AddPotentialForce(force_->ObservationDual(obs));
  } else {
    prob.AddForce(force_->ObservationDual(obs));
  }
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
    MFEM_VERIFY(op.NumBranches() == 1 &&
                    op.NumBranchNodes(0) ==
                        op.InternalScalarSpace().GetVSize(),
                "AdjointSolver: kernel accumulators support a single "
                "whole-mesh branch for now.");
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

  // Continuous terms through the force hook, reversed time mapped back.
  ResolvePotentialSlot(prob);
  const bool has_continuous =
      force_->HasContinuous() || force_->HasContinuousPotential();
  if (has_continuous) {
    op.SetForceHook([this, t1, &prob](real_t s) {
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

  // Pairing fields: wrappers on the internal spaces, reassigned per node.
  auto& fop = fwd.Op();
  auto d_f = detail::MakeGridFunction(&fop.InternalVariableSpace());
  auto m_f = detail::MakeGridFunction(&fop.InternalVariableSpace());
  auto mdag_f = detail::MakeGridFunction(&op.InternalVariableSpace());
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
    fop.BranchToFull(fwd.State(n), 0, mbuf);
    *m_f = mbuf;
    const real_t wR = n < N ? 0.5 * (fwd.Time(n + 1) - fwd.Time(n)) : 0.0;
    const real_t wL = n > 0 ? 0.5 * (fwd.Time(n) - fwd.Time(n - 1)) : 0.0;

    auto pair_state = [&](real_t w) {
      if (w <= 0.0) {
        return;
      }
      op.BranchToFull(mdag, 0, mbuf);
      *mdag_f = mbuf;
      Pair(w, AdjointPair{u_f, d_f.get(), m_f.get(), prob.Displacement(),
                          mdag_f.get()});
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
      // response; m+ is bounded, so mdag is null here.
      Pair(1.0, AdjointPair{u_f, d_f.get(), m_f.get(), *udag_g, nullptr});
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

  if (has_continuous) {
    op.SetForceHook(nullptr);
  }
  return ok;
}

}  // namespace AdGIA
