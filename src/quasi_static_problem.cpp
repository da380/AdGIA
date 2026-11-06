/**
 * @file quasi_static_problem.cpp
 * @brief Implementation of LinearQuasiStaticProblemBase,
 * LinearQuasiStaticTractionProblem and LinearQuasiStaticClampedProblem.
 */

#include "AdGIA/quasi_static_problem.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <string_view>
#include <vector>

#include "AdGIA/bilininteg.hpp"
#include "AdGIA/detail/fem_factory.hpp"
#include "AdGIA/elastic_tensor.hpp"
#include "AdGIA/mappings.hpp"

namespace AdGIA {

using namespace mfem;

// ---------------------------------------------------------------------------
// FluidRegionOperator

void FluidRegionOperator::Configure(FiniteElementSpace& fes,
                                    const Array<int>& marker,
                                    Coefficient& mu, real_t epsilon,
                                    GaugePenalty form, Diffeomorphism* map) {
  MFEM_VERIFY(marker.Size() == fes.GetMesh()->attributes.Max(),
              "FluidRegionOperator: the marker must be sized to "
              "attributes.Max().");
  MFEM_VERIFY(epsilon > 0.0,
              "FluidRegionOperator: epsilon must be positive.");
  // A supplied map — the identity included — switches the Deviatoric
  // branch to ElasticTensorIntegrator(C, map), so that the two sides of
  // a change-of-variables identity assemble with the SAME integrator
  // class and quadrature rule. Only a non-identity map is refused for
  // Harmonic.
  const bool mapped = map != nullptr;
  MFEM_VERIFY(!(mapped && !map->IsIdentity()) ||
                  form == GaugePenalty::Deviatoric,
              "FluidRegionOperator: the Harmonic penalty is gauge data, "
              "shared rather than mapped.");
  Clear();
  marker_ = marker;
  eps_ = std::make_unique<ConstantCoefficient>(epsilon);
  mu_eps_ = std::make_unique<ProductCoefficient>(*eps_, mu);
  const int dim = fes.GetMesh()->Dimension();
  integrators_ = detail::MakeBilinearForm(&fes);
  BilinearFormIntegrator* integ;
  if (mapped && form == GaugePenalty::Deviatoric) {
    // The covariant form of the Deviatoric branch: the isotropic tensor
    // of lambda = -2 eps mu_g / d, mu = eps mu_g, pulled back through
    // the map by ElasticTensorIntegrator, so the penalty of a
    // relabelled problem is the exact pull-back of the unmapped one.
    lambda_eps_ = std::make_unique<ProductCoefficient>(-2.0 / dim, *mu_eps_);
    Cdev_ = std::make_unique<IsotropicElasticTensorCoefficient>(
        dim, *lambda_eps_, *mu_eps_);
    integ = new ElasticTensorIntegrator(*Cdev_, *map);
  } else if (form == GaugePenalty::Deviatoric) {
    integ = new ElasticityIntegrator(*mu_eps_, -2.0 / dim, 1.0);
  } else {
    integ = new VectorDiffusionIntegrator(*mu_eps_);
  }
  integrators_->AddDomainIntegrator(integ, marker_);
  integ_ = integ;
#ifdef MFEM_USE_MPI
  parallel_ = dynamic_cast<ParFiniteElementSpace*>(&fes) != nullptr;
#endif
}

void FluidRegionOperator::Clear() {
  integrators_.reset();
  integ_ = nullptr;
  q_form_.reset();
  mu_eps_.reset();
  lambda_eps_.reset();
  Cdev_.reset();
  eps_.reset();
  Q_.Clear();
  unit_.Clear();
}

void FluidRegionOperator::SetEpsilon(real_t epsilon) {
  MFEM_VERIFY(eps_, "FluidRegionOperator: not configured.");
  MFEM_VERIFY(epsilon > 0.0,
              "FluidRegionOperator: epsilon must be positive.");
  eps_->constant = epsilon;
}

real_t FluidRegionOperator::Epsilon() const {
  return eps_ ? eps_->constant : 0.0;
}

void FluidRegionOperator::Assemble(FiniteElementSpace& fes) {
  MFEM_VERIFY(Active(), "FluidRegionOperator: not configured.");
#ifdef MFEM_USE_MPI
  if (parallel_) {
    Q_.SetType(Operator::Hypre_ParCSR);
  }
#endif
  // eps mu Q on true dofs, unconstrained: the drivers zero the
  // essential rows of their residuals instead.
  q_form_ = detail::MakeBilinearForm(&fes, integrators_.get());
  q_form_->Assemble();
  Array<int> empty;
  q_form_->FormSystemMatrix(empty, Q_);
  // The unit-epsilon cache, so an epsilon-only change is Rescale().
  const real_t e = eps_->constant;
  unit_.Clear();
#ifdef MFEM_USE_MPI
  if (parallel_) {
    auto* Qh = Q_.As<HypreParMatrix>();
    unit_.Reset(mfem::Add(1.0 / e, *Qh, 0.0, *Qh), true);
  } else
#endif
  {
    auto* Qu = new SparseMatrix(*Q_.As<SparseMatrix>());
    *Qu *= 1.0 / e;
    unit_.Reset(Qu, true);
  }
}

void FluidRegionOperator::Rescale() {
  MFEM_VERIFY(HasUnitCache(), "FluidRegionOperator: no unit cache.");
  const real_t e = eps_->constant;
#ifdef MFEM_USE_MPI
  if (parallel_) {
    auto* Qu = unit_.As<HypreParMatrix>();
    Q_.Reset(mfem::Add(e, *Qu, 0.0, *Qu), true);
  } else
#endif
  {
    auto* q = new SparseMatrix(*unit_.As<SparseMatrix>());
    *q *= e;
    Q_.Reset(q, true);
  }
}

void FluidRegionOperator::ApplyQ(const Vector& x, Vector& r) const {
  Q_.Ptr()->Mult(x, r);
}

// ---------------------------------------------------------------------------

LinearQuasiStaticProblemBase::LinearQuasiStaticProblemBase(
    FiniteElementSpace* fes, const AdGIA::Rheology& rheology)
    : fes_(fes),
      rheology_(&rheology),
      stiffness_(rheology.MakeStiffness()),
      A_(Operator::MFEM_SPARSEMAT) {
  const int dim = fes_->GetMesh()->Dimension();
  MFEM_VERIFY(
      fes_->GetVDim() == dim,
      "LinearQuasiStaticProblemBase: the displacement space must have vdim "
      "equal to the space dimension.");
  MFEM_VERIFY(
      rheology.SpaceDim() == dim,
      "LinearQuasiStaticProblemBase: rheology and mesh dimensions differ.");
#ifdef MFEM_USE_MPI
  pfes_ = dynamic_cast<ParFiniteElementSpace*>(fes_);
  if (pfes_) {
    A_.SetType(Operator::Hypre_ParCSR);
  }
#endif

  // The stiffness integrators live in a template form that is never
  // assembled; each assembly builds a fresh form borrowing them, so that a
  // change of modulus never has to reuse a matrix pattern.
  integrators_ = detail::MakeBilinearForm(fes_);
  stiffness_->AddIntegrators(*integrators_);
  // Everything a derived class adds later (gravity, interface and load
  // terms) comes after these; the preconditioner may leave it out.
  n_rheology_domain_ = integrators_->GetDBFI()->Size();
  n_rheology_boundary_ = integrators_->GetBBFI()->Size();

  b_ = detail::MakeLinearForm(fes_);
  u_ = detail::MakeGridFunction(fes_);
  *u_ = 0.0;
  increment_.SetSize(fes_->GetVSize());
  increment_ = 0.0;
}

bool LinearQuasiStaticProblemBase::IsParallel() const {
#ifdef MFEM_USE_MPI
  return pfes_ != nullptr;
#else
  return false;
#endif
}

void LinearQuasiStaticProblemBase::SetEssentialBoundary(
    const Array<int>& ess_bdr) {
  Array<int> marker(ess_bdr);
  fes_->GetEssentialTrueDofs(marker, ess_tdof_list_);
  MarkOperatorDirty();
}

void LinearQuasiStaticProblemBase::AssembleForce(real_t t) {
  t_ = t;
  for (auto* c : td_coefs_) {
    c->SetTime(t);
  }
  for (auto* c : td_vcoefs_) {
    c->SetTime(t);
  }
  // LinearForm::Assemble() zeroes before assembling: idempotent at fixed t.
  b_->Assemble();
  increment_ = 0.0;
  UpdateBoundaryValues(t);
}

void LinearQuasiStaticProblemBase::AddForce(const Vector& f) {
  MFEM_VERIFY(f.Size() == increment_.Size(),
              "AddForce: expected a dual vector in the vdof layout of "
              "DisplacementSpace().");
  increment_ += f;
}

void LinearQuasiStaticProblemBase::SetRelaxationWeights(
    const std::vector<Coefficient*>& beta) {
  // The same coefficient objects may carry new values, so the relaxed
  // flavour always reassembles; the unrelaxed state does not depend on
  // the weights and is stashed for ClearRelaxationWeights to restore.
  if (!stiffness_->IsRelaxed()) {
    StashFlavor(0);
  }
  ReleaseFlavorSlot(1);
  stiffness_->SetRelaxationWeights(beta);
  relax_beta_ = beta;
  operator_dirty_ = true;
}

void LinearQuasiStaticProblemBase::ClearRelaxationWeights() {
  if (!stiffness_->IsRelaxed()) {
    return;
  }
  StashFlavor(1);
  stiffness_->ClearRelaxationWeights();
  if (!TryRestoreFlavor(0)) {
    operator_dirty_ = true;
  }
}

bool LinearQuasiStaticProblemBase::RestoreRelaxationWeights() {
  if (stiffness_->IsRelaxed()) {
    return true;  // engaged, and the caller promises the values unchanged
  }
  if (!flavor_[1].valid || relax_beta_.empty() ||
      (fluid_op_.Active() &&
       fluid_op_.Epsilon() != flavor_[1].gauge_eps)) {
    return false;  // nothing restorable: fall back to SetRelaxationWeights
  }
  StashFlavor(0);
  stiffness_->SetRelaxationWeights(relax_beta_);
  if (!TryRestoreFlavor(1)) {
    // StashFlavor may invalidate the slot it writes, never slot 1;
    // unreachable after the checks above, but stay correct.
    operator_dirty_ = true;
  }
  return true;
}

void LinearQuasiStaticProblemBase::MarkOperatorDirty() {
  operator_dirty_ = true;
  InvalidateFlavorSlots();
}

void LinearQuasiStaticProblemBase::MarkGaugeEpsilonDirty() {
  eps_only_dirty_ = true;
  InvalidateFlavorSlots();
}

void LinearQuasiStaticProblemBase::InvalidateFlavorSlots() {
  ReleaseFlavorSlot(0);
  ReleaseFlavorSlot(1);
}

void LinearQuasiStaticProblemBase::ReleaseFlavorSlot(int which) {
  FlavorSlot& s = flavor_[which];
  const bool referent = prec_ref_slot_ == which;
  if (referent) {
    // The live preconditioner was built on this slot's matrices: they
    // move into the shared keep-alive (the AssembleOperator capture's
    // home) instead of dying with the slot.
    if (s.a_solve_form || s.A_solve.Ptr()) {
      prec_form_ = std::move(s.a_solve_form);
      MoveHandle(s.A_solve, prec_A_);
      s.a.reset();
      s.A.Clear();
    } else {
      prec_form_ = std::move(s.a);
      MoveHandle(s.A, prec_A_);
    }
    prec_ref_slot_ = -1;
  } else {
    s.a.reset();
    s.a_solve_form.reset();
    s.A.Clear();
    s.A_solve.Clear();
  }
  s.valid = false;
  OnFlavorRelease(which, referent);
}

void LinearQuasiStaticProblemBase::MoveHandle(OperatorHandle& from,
                                              OperatorHandle& to) {
  to.Clear();
  if (from.Ptr()) {
    to.Reset(from.Ptr(), from.OwnsOperator());
    from.SetOperatorOwner(false);
  }
  from.Clear();
}

void LinearQuasiStaticProblemBase::StashFlavor(int which) {
  if (flavor_[which].valid || flavor_[which].a) {
    ReleaseFlavorSlot(which);  // retained storage dies by the one rule
  }
  FlavorSlot& s = flavor_[which];
  if (pressure_split_ || !a_ || operator_dirty_ || eps_only_dirty_) {
    // No clean assembled state to keep — or the pressure-split route is
    // on, whose block matrices the slots do not hold: a restore would
    // clear the dirty flag over stale blocks.
    s.valid = false;
    return;
  }
  // The shared preconditioner stays live (its reuse policy is
  // flavour-blind); when it was built on the matrices being stashed
  // and is not yet captured, the slot becomes its keep-alive.
  if (prec_ && !prec_form_ && !prec_A_.Ptr() && prec_ref_slot_ < 0) {
    prec_ref_slot_ = which;
  }
  s.a = std::move(a_);
  s.a_solve_form = std::move(a_solve_form_);
  MoveHandle(A_, s.A);
  MoveHandle(A_solve_, s.A_solve);
  s.version = operator_version_;
  s.gauge_eps = fluid_op_.Active() ? fluid_op_.Epsilon() : -1.0;
  s.valid = true;
  OnFlavorStash(which);
}

bool LinearQuasiStaticProblemBase::TryRestoreFlavor(int which) {
  FlavorSlot& s = flavor_[which];
  if (pressure_split_ || !s.valid ||
      (fluid_op_.Active() ? fluid_op_.Epsilon() : -1.0) != s.gauge_eps) {
    if (s.valid) {
      ReleaseFlavorSlot(which);
    }
    return false;
  }
  a_ = std::move(s.a);
  a_solve_form_ = std::move(s.a_solve_form);
  MoveHandle(s.A, A_);
  MoveHandle(s.A_solve, A_solve_);
  if (prec_ref_slot_ == which) {
    prec_ref_slot_ = -1;  // the preconditioner's matrices are live again
  }
  // Reviving the version revives the version-keyed caches (border
  // columns, unit responses) built against this flavour — valid
  // because any other operator mutation drops the slots
  // (MarkOperatorDirty).
  operator_version_ = s.version;
  s.valid = false;
  operator_dirty_ = false;
  eps_only_dirty_ = false;
  OnFlavorRestore(which);
  SetupSolver(HasGaugedFluid() ? A_solve_ : A_);
  return true;
}

bool LinearQuasiStaticProblemBase::PreconditionerAwaitsCapture() const {
  // A preconditioner whose matrices sit in a flavour slot is already
  // kept alive there (prec_ref_slot_); the current matrices are not
  // its referents.
  return a_ && prec_ && !prec_stale_ && prec_reuse_ > 1.0 && !prec_form_ &&
         !prec_A_.Ptr() && prec_ref_slot_ < 0;
}

void LinearQuasiStaticProblemBase::CapturePreconditionerMatrix(
    OperatorHandle& src) {
  prec_A_ = src;
  prec_A_.SetOperatorOwner(src.OwnsOperator());
  src.SetOperatorOwner(false);
}

void LinearQuasiStaticProblemBase::AssembleOperator() {
  operator_version_++;
  if (PreconditionerAwaitsCapture()) {
    // The preconditioner was built on the current solver matrix and stays on
    // it: keep that form and matrix alive while the preconditioner is
    // reused. (Later reassemblies leave prec_form_ alone; their matrices
    // go.) With a gauged fluid the solver matrix is the regularised one.
    // A matrix already captured by the RescaleGaugeOperator fast path
    // (prec_A_ set with no form behind it) is likewise left alone.
    if (a_solve_form_) {
      prec_form_ = std::move(a_solve_form_);
      CapturePreconditionerMatrix(A_solve_);
    } else {
      prec_form_ = std::move(a_);
      CapturePreconditionerMatrix(A_);
    }
  }
  a_ = detail::MakeBilinearForm(fes_, integrators_.get());
  a_->Assemble();
  a_->FormSystemMatrix(ess_tdof_list_, A_);
  a_solve_form_.reset();
  if (HasGaugedFluid()) {
#ifdef MFEM_USE_MPI
    if (pfes_) {
      A_solve_.SetType(Operator::Hypre_ParCSR);
    }
#endif
    fluid_op_.Assemble(*fes_);
    // A + eps mu Q in one assembly: borrow the physical integrators and
    // append the engine's penalty integrator (the borrowing form owns
    // none of them).
    a_solve_form_ = detail::MakeBilinearForm(fes_, integrators_.get());
    a_solve_form_->AddDomainIntegrator(fluid_op_.Integrator(),
                                       fluid_op_.Marker());
    a_solve_form_->Assemble();
    a_solve_form_->FormSystemMatrix(ess_tdof_list_, A_solve_);
  }
  SetupSolver(HasGaugedFluid() ? A_solve_ : A_);
  operator_dirty_ = false;
  eps_only_dirty_ = false;
  assemblies_++;
}

void LinearQuasiStaticProblemBase::RescaleGaugeOperator() {
  operator_version_++;
  MFEM_VERIFY(HasGaugedFluid() && fluid_op_.HasUnitCache(),
              "RescaleGaugeOperator: no cached unit penalty.");
  // Keep the matrix the (reused) preconditioner was built on alive, as
  // in AssembleOperator; here there is no form to move, the matrix
  // handle alone is captured.
  if (PreconditionerAwaitsCapture()) {
    CapturePreconditionerMatrix(A_solve_);
  }
  fluid_op_.Rescale();
#ifdef MFEM_USE_MPI
  if (pfes_) {
    A_solve_.Reset(mfem::Add(1.0, *A_.As<HypreParMatrix>(), 1.0,
                             *fluid_op_.Q().As<HypreParMatrix>()),
                   true);
  } else
#endif
  {
    A_solve_.Reset(mfem::Add(1.0, *A_.As<SparseMatrix>(), 1.0,
                             *fluid_op_.Q().As<SparseMatrix>()),
                   true);
  }
  SetupSolver(A_solve_);
  eps_only_dirty_ = false;
}

void LinearQuasiStaticProblemBase::WarnGaugeContraction() const {
  if (gauge_residuals_.size() < 2) {
    return;
  }
  // Corrections at the linear-solver tolerance floor are solver noise:
  // the recorded residuals ||eps Q delta_k|| stop shrinking there and the
  // ratio sits near 1 although the refinement has converged. The noise
  // scale is rel_tol_ times the first residual ||eps Q X|| (delta noise
  // ~ rel_tol_ ||X||), so below a safe multiple of it the contraction
  // estimate is meaningless and no warning is due.
  const real_t floor = 10.0 * rel_tol_ * gauge_residuals_.front();
  if (gauge_residuals_.back() <= floor) {
    return;
  }
  const real_t r0 = gauge_residuals_[gauge_residuals_.size() - 2];
  const real_t rate = gauge_residuals_.back() / std::max(r0, real_t{1e-300});
  bool root = true;
#ifdef MFEM_USE_MPI
  if (pfes_) {
    root = pfes_->GetMyRank() == 0;
  }
#endif
  if (root && rate > real_t{0.9}) {
    mfem::out << "GaugeRefine WARNING: refinement contraction " << rate
              << " >= 0.9 — semi-convergence regime, the gauge penalty "
                 "epsilon is too small for this mesh/model.\n";
  } else if (root && rate > real_t{0.2}) {
    mfem::out << "GaugeRefine note: refinement contraction " << rate
              << " > 0.2 — residual gauge bias ~rate^k may remain; "
                 "consider a smaller epsilon or more refinements.\n";
  }
}

void LinearQuasiStaticProblemBase::NoteIterations(int its) {
  total_its_ += its;
  if (prec_baseline_its_ < 0) {
    prec_baseline_its_ = its;
  } else if (its > prec_reuse_ * prec_baseline_its_) {
    prec_stale_ = true;
  }
}

void LinearQuasiStaticProblemBase::EnsureOperator() {
  if (operator_dirty_) {
    AssembleOperator();
  } else if (eps_only_dirty_) {
    RescaleGaugeOperator();
  }
}

const OperatorHandle& LinearQuasiStaticProblemBase::SystemMatrix() {
  EnsureOperator();
  return A_;
}

bool LinearQuasiStaticProblemBase::Solve() {
  solves_++;
  if (fluid_treatment_ == FluidTreatment::Maxwell) {
    return MaxwellSolve();
  }
  if (pressure_split_ && !PressureSplitInClassSolve()) {
    return PressureSplitSolve();
  }
  EnsureOperator();
  rhs_ = *b_;
  rhs_ += increment_;
  // Fold the boundary data into the reduced system on the scratch copy rhs_,
  // keeping the assembled external load pristine. copy_interior = 1 keeps
  // the interior of u_ in X_ so that solvers in iterative_mode warm start.
  a_->FormLinearSystem(ess_tdof_list_, *u_, rhs_, A_, X_, B_, 1);
  bool ok = SolveLinearSystem(B_, X_);
  if (HasGaugedFluid()) {
    ok = GaugeRefine(X_) && ok;
  }
  a_->RecoverFEMSolution(X_, rhs_, *u_);
  return ok;
}

void LinearQuasiStaticProblemBase::ConfigureFluidOperator(
    const Array<int>& marker, Coefficient& mu, real_t epsilon,
    GaugePenalty form, Diffeomorphism* map) {
  fluid_op_.Configure(*fes_, marker, mu, epsilon, form, map);
}

void LinearQuasiStaticProblemBase::SetFluid(
    const Array<int>& fluid_marker, Coefficient& mu_scale,
    const MaxwellRelaxationOptions& opts, Diffeomorphism* map) {
  MFEM_VERIFY(!gauge_prec_only_,
              "SetFluid: the Maxwell treatment is incompatible with "
              "SetGaugePreconditionerOnly.");
  MFEM_VERIFY(!pressure_split_,
              "SetFluid: no fluid treatments on the pressure-split route.");
  MFEM_VERIFY(opts.dt_over_tau > 0.0 && opts.escalate >= 1.0 &&
                  opts.beta_max >= opts.dt_over_tau && opts.max_steps > 0 &&
                  opts.prec_eps >= 0.0,
              "SetFluid: invalid Maxwell schedule options.");
  maxwell_opts_ = opts;
  if (opts.plateau_mode) {
    // The physical plateau needs the plain trajectory: Anderson mixing
    // is ignored in plateau mode.
    maxwell_opts_.anderson = 0;
  }
  // The per-step operator is the engine at eps = 1/(1 + beta), with no
  // Tikhonov refinements (the Maxwell term is the regulariser, and
  // MaxwellSolve owns the iteration). The configuration dispatches
  // through the virtual hook, so a derived class's covariant default
  // (or refusal) applies.
  ConfigureFluidOperator(fluid_marker, mu_scale,
                         1.0 / (1.0 + opts.dt_over_tau),
                         GaugePenalty::Deviatoric, map);
  fluid_treatment_ = FluidTreatment::Maxwell;
  gauge_refinements_ = 0;
  maxwell_solid_tdofs_.SetSize(0);
  MarkOperatorDirty();
}

void LinearQuasiStaticProblemBase::SetFluid(
    const Array<int>& fluid_marker, Coefficient& mu_scale,
    const GaugePenaltyOptions& opts, Diffeomorphism* map) {
  MFEM_VERIFY(!pressure_split_,
              "SetFluid: no fluid treatments on the pressure-split route.");
  ConfigureFluidOperator(fluid_marker, mu_scale, opts.epsilon, opts.form,
                         map);
  fluid_treatment_ = FluidTreatment::Penalty;
  gauge_refinements_ = opts.refinements;
  MarkOperatorDirty();
}

void LinearQuasiStaticProblemBase::ApplyGaugePenalty(const Vector& u_true,
                                                     Vector& r) {
  r.SetSize(u_true.Size());
  r = 0.0;
  if (!HasGaugedFluid()) {
    return;
  }
  EnsureOperator();
  fluid_op_.ApplyQ(u_true, r);
}

void LinearQuasiStaticProblemBase::ClearFluid() {
  fluid_treatment_ = FluidTreatment::None;  // both drivers ride the engine
  fluid_op_.Clear();
  a_solve_form_.reset();
  A_solve_.Clear();
  eps_only_dirty_ = false;
  gauge_residuals_.clear();
  maxwell_solid_tdofs_.SetSize(0);
  maxwell_report_ = MaxwellRelaxationReport();
  MarkOperatorDirty();
}

void LinearQuasiStaticProblemBase::SetGaugeEpsilon(real_t epsilon) {
  MFEM_VERIFY(fluid_op_.Active(),
              "SetGaugeEpsilon: no gauged fluid is set.");
  if (fluid_op_.Epsilon() == epsilon) {
    return;
  }
  fluid_op_.SetEpsilon(epsilon);
  if (fluid_op_.HasUnitCache()) {
    // Only the penalty scale changed: the cheap rescale path serves.
    MarkGaugeEpsilonDirty();
  } else {
    MarkOperatorDirty();
  }
}

real_t LinearQuasiStaticProblemBase::GaugeEpsilon() const {
  return fluid_op_.Epsilon();
}

void LinearQuasiStaticProblemBase::SetGaugePreconditionerOnly(
    real_t eps_prec) {
  MFEM_VERIFY(fluid_op_.Active(),
              "SetGaugePreconditionerOnly: no gauged fluid is set.");
  MFEM_VERIFY(fluid_treatment_ != FluidTreatment::Maxwell,
              "SetGaugePreconditionerOnly: incompatible with "
              "the Maxwell fluid treatment.");
  // The penalty scale now serves the PRECONDITIONER matrix A + eps_prec Q
  // only; the solver operator is the clean A, and without an O(eps) bias
  // in the operator the Tikhonov refinements have nothing to remove.
  fluid_op_.SetEpsilon(eps_prec);  // verifies positivity
  gauge_refinements_ = 0;
  gauge_prec_only_ = true;
  MarkOperatorDirty();
}

void LinearQuasiStaticProblemBase::SetGaugePlateauStop(real_t ratio,
                                                       int chunk) {
  MFEM_VERIFY(gauge_prec_only_,
              "SetGaugePlateauStop: only meaningful with "
              "SetGaugePreconditionerOnly (the penalty path has no "
              "plateau to detect).");
  MFEM_VERIFY(ratio > 0.0 && ratio < 1.0 && chunk > 0,
              "SetGaugePlateauStop: 0 < ratio < 1 and chunk > 0.");
  gauge_plateau_ratio_ = ratio;
  gauge_plateau_chunk_ = chunk;
}

int LinearQuasiStaticProblemBase::PlateauMult(Solver& outer,
                                              IterativeSolver& krylov,
                                              const Vector& B, Vector& X,
                                              bool& converged) {
  const int chunk = gauge_plateau_chunk_;
  const int cap = 10000;
  krylov.SetMaxIter(chunk);
  int total = 0;
  real_t prev = 0.0;
  converged = false;
  for (int c = 0; total < cap; ++c) {
    outer.Mult(B, X);
    total += krylov.GetNumIterations();
    if (krylov.GetConverged()) {
      converged = true;
      break;
    }
    const real_t r = krylov.GetFinalNorm();
    if (c > 0 && r > gauge_plateau_ratio_ * prev) {
      // Stagnation: the residual has reached the load's near-kernel
      // plateau — the intended stopping point of the clean-operator
      // mode (iterating on semi-converges into gauge junk).
      converged = true;
      break;
    }
    prev = r;
  }
  krylov.SetMaxIter(cap);
  return total;
}

const OperatorHandle& LinearQuasiStaticProblemBase::RegularizedMatrix() {
  EnsureOperator();
  return HasGaugedFluid() ? A_solve_ : A_;
}

bool LinearQuasiStaticProblemBase::GaugeRefine(Vector& X) {
  // The beta = infinity, fixed-epsilon, no-escalation case of the
  // Maxwell step: each refinement is the FULL solve
  // (A + eps Q) u_{k+1} = f + eps Q u_k through the virtual
  // SolveLinearSystem, so companion blocks (a potential) ride along
  // with their loads reapplied (equivalent, by linearity, to
  // accumulating zero-companion-load increments). The residuals
  // ||eps Q delta_k|| are recorded.
  gauge_residuals_.clear();
  Vector r(X.Size()), Bn, Xprev, delta(X);
  bool ok = true;
  for (int k = 0; k < gauge_refinements_; ++k) {
    // After an exact regularised solve the physical residual is
    // f - A U = eps Q delta, with delta the last step (the first
    // "step" being the solution itself).
    fluid_op_.ApplyQ(delta, r);
    if (ess_tdof_list_.Size() > 0) {
      r.SetSubVector(ess_tdof_list_, 0.0);
    }
    gauge_residuals_.push_back(std::sqrt(Dot(r, r)));
    fluid_op_.ApplyQ(X, r);
    if (ess_tdof_list_.Size() > 0) {
      r.SetSubVector(ess_tdof_list_, 0.0);
    }
    Bn = B_;
    Bn += r;
    Xprev = X;
    ok = SolveLinearSystem(Bn, X) && ok;
    delta = X;
    delta -= Xprev;
  }
  WarnGaugeContraction();
  return ok;
}

void LinearQuasiStaticProblemBase::BuildMaxwellSolidDofs() {
  // L-dof indicator of the elements OUTSIDE the fluid marker; a dof
  // shared with a fluid element (the interface trace) counts as solid.
  Vector ind(fes_->GetVSize());
  ind = 0.0;
  Array<int> vdofs;
  const Array<int>& marker = fluid_op_.Marker();
  for (int e = 0; e < fes_->GetNE(); ++e) {
    const int attr = fes_->GetMesh()->GetAttribute(e);
    if (attr <= marker.Size() && marker[attr - 1]) {
      continue;
    }
    fes_->GetElementVDofs(e, vdofs);
    for (int j : vdofs) {
      ind[j < 0 ? -1 - j : j] = 1.0;
    }
  }
  maxwell_solid_tdofs_.SetSize(0);
  maxwell_solid_tdofs_.Reserve(fes_->GetTrueVSize());
  const Operator* P = fes_->GetProlongationMatrix();
  if (P) {
    // P^T assembles the indicator onto true dofs across ranks: positive
    // wherever any sharing element is solid.
    Vector t(fes_->GetTrueVSize());
    P->MultTranspose(ind, t);
    for (int i = 0; i < t.Size(); ++i) {
      if (t[i] > 0.5) {
        maxwell_solid_tdofs_.Append(i);
      }
    }
  } else {
    for (int i = 0; i < ind.Size(); ++i) {
      if (ind[i] > 0.5) {
        maxwell_solid_tdofs_.Append(i);
      }
    }
  }
}

real_t LinearQuasiStaticProblemBase::MaxOverSolidDofs(const Vector& x) const {
  real_t m = 0.0;
  for (int i = 0; i < maxwell_solid_tdofs_.Size(); ++i) {
    m = std::max(m, std::abs(x[maxwell_solid_tdofs_[i]]));
  }
#ifdef MFEM_USE_MPI
  if (pfes_) {
    real_t g = 0.0;
    MPI_Allreduce(&m, &g, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                  pfes_->GetComm());
    return g;
  }
#endif
  return m;
}

bool LinearQuasiStaticProblemBase::MaxwellSolve() {
  const MaxwellRelaxationOptions& o = maxwell_opts_;
  maxwell_report_ = MaxwellRelaxationReport();

  real_t beta = o.dt_over_tau;
  real_t eps_cur = -1.0;
  Vector w, r, Bn, prev;
  real_t ref = 0.0, prev_delta = -1.0, t_phys = 0.0;
  bool stagnated = false, ok = true, first = true;

  // Inexact stepping: the fixed-point iteration is self-correcting, so
  // each step is solved only to a fraction of the increment it
  // produces. The solver objects stay configured at the base
  // tolerance (set_operator restores it around every setup); the
  // loosening acts purely through the per-solve absolute tolerance of
  // SetWarmStartTolerance, and each step VALIDATES its tolerance
  // against the increment it measured, tightening and redoing (a
  // warm-started continuation of the same solve) when the increment
  // was not resolved — without this, a warm start under a loose
  // tolerance can return unmoved and fake convergence.
  const real_t rel_tol_base = rel_tol_;

  // The best state seen (lowest solid increment at a successful step).
  // The iterate is a function of (w, eps) plus a warm start, so the
  // checkpoint is the memory vector alone and a restore is one warm
  // re-solve — which also restores any companion blocks a derived
  // SolveLinearSystem carries (a potential, say) to consistency.
  Vector w_best;
  real_t eps_best = -1.0, delta_best = infinity(), beta_best = 0.0;

  // Switch the per-step operator to eps (rescale or reassembly), and
  // rebuild the reduced system around the current u_ so warm starts
  // survive.
  auto set_operator = [&](real_t eps) {
    if (!first) {
      a_->RecoverFEMSolution(X_, rhs_, *u_);
    }
    const real_t rt = rel_tol_;
    rel_tol_ = rel_tol_base;  // solvers are configured at base tolerance
    SetGaugeEpsilon(eps);
    EnsureOperator();
    rel_tol_ = rt;
    rhs_ = *b_;
    rhs_ += increment_;
    a_->FormLinearSystem(ess_tdof_list_, *u_, rhs_, A_, X_, B_, 1);
    eps_cur = eps;
    maxwell_report_.operators++;
  };
  // One backward Euler step at the current operator:
  // (A + gamma Qhat) u = f + gamma Qhat w, with Q_ = eps mu_c Qhat =
  // gamma Qhat at eps = 1/(1 + beta).
  auto step_solve = [&]() {
    fluid_op_.ApplyQ(w, r);
    if (ess_tdof_list_.Size() > 0) {
      r.SetSubVector(ess_tdof_list_, 0.0);
    }
    Bn = B_;
    Bn += r;
    return SolveLinearSystem(Bn, X_);
  };

  // Anderson mixing on the memory fixed-point map w -> G(w) (depth
  // o.anderson): histories of iterates g_i = G(w_i) and residuals
  // r_i = g_i - w_i, the mixed update w <- g_k - dG gamma with gamma
  // from the least squares min ||r_k - dR gamma|| (normal equations;
  // the window restarts at every operator change).
  std::vector<Vector> aa_g, aa_r;
  auto aa_mix = [&](const Vector& g) {
    aa_r.emplace_back(g);
    aa_r.back() -= w;
    aa_g.emplace_back(g);
    const int keep = o.anderson + 1;
    while (static_cast<int>(aa_g.size()) > keep) {
      aa_g.erase(aa_g.begin());
      aa_r.erase(aa_r.begin());
    }
    const int m = static_cast<int>(aa_g.size()) - 1;
    if (m < 1) {
      w = g;
      return;
    }
    // dR_j = r_k - r_j, dG_j = g_k - g_j over the window.
    DenseMatrix N(m);
    Vector rhs(m), gamma(m);
    const Vector& rk = aa_r.back();
    std::vector<Vector> dr(m, rk);
    for (int i = 0; i < m; ++i) {
      dr[i] -= aa_r[i];
    }
    for (int i = 0; i < m; ++i) {
      for (int j = i; j < m; ++j) {
        const real_t v = Dot(dr[i], dr[j]);
        N(i, j) = v;
        N(j, i) = v;
      }
      rhs[i] = Dot(dr[i], rk);
    }
    for (int i = 0; i < m; ++i) {
      N(i, i) += 1e-12 * (N(i, i) + 1.0);  // near-collinearity guard
    }
    DenseMatrixInverse Ninv(N);
    Ninv.Mult(rhs, gamma);
    // w = g_k - sum_j gamma_j (g_k - g_j)
    w = aa_g.back();
    for (int j = 0; j < m; ++j) {
      w.Add(-gamma[j], aa_g.back());
      w.Add(gamma[j], aa_g[j]);
    }
  };

  // Frozen preconditioner (prec_eps > 0): assemble the operator at the
  // moderate frozen epsilon and build the preconditioner on it, then
  // pin it for the whole continuation. The spike heuristic would
  // otherwise rebuild on the current — ever worse conditioned —
  // small-eps operator once the iteration count doubles.
  const real_t reuse_saved = prec_reuse_;
  if (o.prec_eps > 0.0) {
    prec_stale_ = true;
    if (fluid_op_.Epsilon() == o.prec_eps && !operator_dirty_ &&
        !eps_only_dirty_) {
      // A clean operator already at prec_eps gives EnsureOperator
      // nothing to do: force one setup through the available path.
      if (fluid_op_.HasUnitCache()) {
        MarkGaugeEpsilonDirty();
      } else {
        MarkOperatorDirty();
      }
    }
    set_operator(o.prec_eps);
    prec_reuse_ = infinity();
  }

  const char* stop = "max_steps";
  real_t delta = infinity();
  long its_first = -1;
  int at_cap = 0;
  for (int n = 1; n <= o.max_steps; ++n) {
    if (1.0 / (1.0 + beta) != eps_cur) {
      set_operator(1.0 / (1.0 + beta));
      aa_g.clear();
      aa_r.clear();
    }
    if (first) {
      // Outside the operator-change guard: with the frozen
      // preconditioner the loop may open at eps_cur == 1/(1 + beta).
      w.SetSize(X_.Size());
      w = 0.0;
      r.SetSize(X_.Size());
      BuildMaxwellSolidDofs();
      first = false;
    }
    // The step, with its tolerance validated against the increment it
    // measures (the stopping metric: the relative per-step SOLID
    // displacement increment — the fluid displacement does not
    // converge at N^2 != 0 and is not monitored).
    const long its_before = total_its_;
    real_t tol_n = o.inexact > 0.0
                       ? std::min(std::max(o.inexact * delta, rel_tol_base),
                                  o.inexact_max)
                       : rel_tol_base;
    bool step_ok;
    for (;;) {
      rel_tol_ = tol_n;
      step_ok = step_solve();
      if (!step_ok) {
        break;
      }
      if (n == 1) {
        ref = MaxOverSolidDofs(X_);
        if (ref <= 0.0) {
          ref = 1.0;
        }
        delta = infinity();
      } else {
        r = X_;
        r -= prev;
        delta = MaxOverSolidDofs(r) / ref;
      }
      if (tol_n <= rel_tol_base ||
          tol_n <= std::max(o.inexact * delta, rel_tol_base)) {
        break;  // the increment is resolved at this tolerance
      }
      tol_n = std::max(rel_tol_base,
                       std::min(tol_n / 10.0, o.inexact * delta));
    }
    if (!step_ok) {
      // The iterative solver's conditioning floor (the per-step fluid
      // shear got too small for the preconditioner). The best state is
      // restored below; the failed increment is discarded.
      stop = "solver_floor";
      delta = infinity();
      break;
    }
    const long its_step = total_its_ - its_before;
    if (its_first < 0) {
      its_first = std::max(its_step, 1L);
    }
    if (o.anderson > 0) {
      Vector g(w);
      g.Add(beta, X_);
      g *= 1.0 / (1.0 + beta);
      aa_mix(g);
    } else {
      w.Add(beta, X_);
      w *= 1.0 / (1.0 + beta);
    }
    t_phys += beta;
    prev = X_;
    maxwell_report_.t_over_tau.push_back(t_phys);
    maxwell_report_.delta_solid.push_back(delta);
    maxwell_report_.iterations.push_back(its_step);

    if (delta < delta_best) {
      delta_best = delta;
      w_best = w;
      eps_best = eps_cur;
      beta_best = beta;
    }
    if (delta <= o.tol) {
      stop = "converged";
      break;
    }
    if (!stagnated && n > o.min_steps && prev_delta > 0.0 &&
        delta > o.stag_ratio * prev_delta) {
      // The solid increment has stopped contracting: the physical
      // plateau (the configurational cascade and, at N^2 < 0, the
      // onset of Rayleigh-Taylor growth). Stop here in plateau mode;
      // otherwise escalate beta — backward Euler is L-stable, so the
      // growing modes are damped and the iteration continues toward
      // the fixed point (regulariser continuation), guarded by the
      // best-state checkpoint.
      stagnated = true;
      maxwell_report_.stag_step = n;
      if (o.plateau_mode) {
        stop = "stagnation";
        break;
      }
    }
    if (stagnated) {
      if (o.iter_budget > 0.0 && its_step > o.iter_budget * its_first) {
        // The escalated solves have become disproportionately
        // expensive: the solver is approaching its conditioning floor.
        // Stop here; the best state is restored below.
        stop = "solver_floor";
        break;
      }
      if (beta >= o.beta_max) {
        // At the cap the iteration is a fixed-regulariser solve; stop
        // when it stagnates by the same criterion as the physical
        // phase (a couple of steps to settle after the operator
        // change).
        if (++at_cap >= 3 && delta > o.stag_ratio * prev_delta) {
          stop = "beta_max";
          break;
        }
      } else {
        beta = std::min(beta * o.escalate, o.beta_max);
      }
    }
    if (delta < infinity()) {
      prev_delta = delta;
    }
  }

  const std::string_view stop_sv(stop);
  maxwell_report_.stop = stop;
  maxwell_report_.delta_returned = delta;
  rel_tol_ = rel_tol_base;  // the endgame runs at full tolerance
  bool polished = false;
  if (stop_sv != "converged" && stop_sv != "stagnation") {
    if (eps_best < 0.0) {
      ok = false;  // no successful step to return
    } else if (delta > delta_best) {
      // The escalation overshot (noise injection past the solver's
      // floor, or drift at the cap): restore the best state with one
      // warm re-solve — one more backward Euler step from (w_best,
      // eps_best), which only improves on it, at full tolerance.
      w = w_best;
      set_operator(eps_best);
      ok = step_solve() && ok;
      w.Add(beta_best, X_);
      w *= 1.0 / (1.0 + beta_best);
      maxwell_report_.delta_returned = delta_best;
      polished = true;
    }
  }
  if (!polished && o.inexact > 0.0 && eps_best >= 0.0) {
    // Inexact steps built the state: polish it with one full-tolerance
    // backward Euler step, so the endpoint accuracy is RelTol()'s.
    ok = step_solve() && ok;
    w.Add(beta, X_);
    w *= 1.0 / (1.0 + beta);
  }
  maxwell_report_.steps =
      static_cast<int>(maxwell_report_.t_over_tau.size());
  prec_reuse_ = reuse_saved;
  a_->RecoverFEMSolution(X_, rhs_, *u_);
  return ok;
}

bool LinearQuasiStaticProblemBase::HasNonElasticIntegrators() {
  return integrators_->GetDBFI()->Size() > n_rheology_domain_ ||
         integrators_->GetBBFI()->Size() > n_rheology_boundary_ ||
         integrators_->GetFBFI()->Size() > 0 ||
         integrators_->GetBFBFI()->Size() > 0;
}

#ifdef MFEM_USE_MPI
void LinearQuasiStaticProblemBase::AssembleElasticPreconditionerMatrix() {
  // A form borrowing the rheology's integrators only (and, with a gauged
  // fluid, the penalty that makes the fluid block definite, as in
  // AssembleOperator's solve form).
  prec_elastic_form_ = detail::MakeBilinearForm(fes_);
  prec_elastic_form_->UseExternalIntegrators();
  auto& dom = *integrators_->GetDBFI();
  auto& dom_mk = *integrators_->GetDBFI_Marker();
  for (int i = 0; i < n_rheology_domain_; i++) {
    if (dom_mk[i]) {
      prec_elastic_form_->AddDomainIntegrator(dom[i], *dom_mk[i]);
    } else {
      prec_elastic_form_->AddDomainIntegrator(dom[i]);
    }
  }
  auto& bdr = *integrators_->GetBBFI();
  auto& bdr_mk = *integrators_->GetBBFI_Marker();
  for (int i = 0; i < n_rheology_boundary_; i++) {
    if (bdr_mk[i]) {
      prec_elastic_form_->AddBoundaryIntegrator(bdr[i], *bdr_mk[i]);
    } else {
      prec_elastic_form_->AddBoundaryIntegrator(bdr[i]);
    }
  }
  if (HasGaugedFluid()) {
    prec_elastic_form_->AddDomainIntegrator(fluid_op_.Integrator(),
                                            fluid_op_.Marker());
  }
  prec_elastic_form_->Assemble();
  prec_elastic_A_.SetType(Operator::Hypre_ParCSR);
  prec_elastic_form_->FormSystemMatrix(ess_tdof_list_, prec_elastic_A_);
}

bool LinearQuasiStaticProblemBase::FreeDisplacementBody() const {
  if (!pfes_) return false;
  long long n = ess_tdof_list_.Size(), ng = 0;
  MPI_Allreduce(&n, &ng, 1, MPI_LONG_LONG, MPI_SUM, pfes_->GetComm());
  return ng == 0;
}

std::unique_ptr<HypreParMatrix> LinearQuasiStaticProblemBase::MassShiftedMatrix(
    HypreParMatrix& A, real_t factor) const {
  ParBilinearForm m(pfes_);
  m.AddDomainIntegrator(new VectorMassIntegrator);
  m.Assemble();
  m.Finalize();
  std::unique_ptr<HypreParMatrix> M(m.ParallelAssemble());
  // eps ~ the stiffness of the smoothest non-rigid deformation, mu / L^2:
  // tr(A) / tr(M) is the mean eigenvalue ratio, ~ mu / h^2, and h^2 / L^2
  // ~ N^(-2/d) for N nodes in d dimensions. The iteration count is
  // insensitive to eps over decades, so the estimate need not be sharp.
  Vector dA, dM;
  A.GetDiag(dA);
  M->GetDiag(dM);
  real_t loc[2] = {dA.Sum(), dM.Sum()}, glob[2];
  MPI_Allreduce(loc, glob, 2, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                pfes_->GetComm());
  const real_t nodes =
      static_cast<real_t>(pfes_->GlobalTrueVSize()) / pfes_->GetVDim();
  const int d = pfes_->GetParMesh()->Dimension();
  const real_t eps = factor * kPrecMassShift * glob[0] / glob[1] *
                     std::pow(nodes, -2.0 / d);
  return std::unique_ptr<HypreParMatrix>(Add(1.0, A, eps, *M));
}
#endif

std::unique_ptr<IterativeSolver> LinearQuasiStaticProblemBase::MakeBlockKrylov()
    const {
  std::unique_ptr<IterativeSolver> k;
#ifdef MFEM_USE_MPI
  if (pfes_) {
    if (block_gmres_) {
      auto g = std::make_unique<FGMRESSolver>(pfes_->GetComm());
      g->SetKDim(200);
      k = std::move(g);
    } else {
      k = std::make_unique<MINRESSolver>(pfes_->GetComm());
    }
    return k;
  }
#endif
  if (block_gmres_) {
    auto g = std::make_unique<FGMRESSolver>();
    g->SetKDim(200);
    k = std::move(g);
  } else {
    k = std::make_unique<MINRESSolver>();
  }
  return k;
}

bool LinearQuasiStaticProblemBase::SetBlockKrylovTolerance(
    IterativeSolver& solver, Solver& prec, const Vector& B) const {
  if (!block_gmres_) {
    return SetWarmStartTolerance(solver, prec, B);
  }
  const real_t nb = std::sqrt(Dot(B, B));
  solver.SetRelTol(0.0);
  solver.SetAbsTol(rel_tol_ * nb);
  return nb > 0.0;
}

void LinearQuasiStaticProblemBase::SetupDefaultCG(OperatorHandle& A) {
  SetupDefaultPreconditioner(A);
  SetupCG(*A.Ptr(), *prec_);
}

void LinearQuasiStaticProblemBase::SetupDefaultPreconditioner(
    OperatorHandle& A) {
  const bool rebuild = !prec_ || prec_stale_ || prec_reuse_ <= 1.0;
  if (rebuild) {
#ifdef MFEM_USE_MPI
    if (pfes_) {
      HypreParMatrix* Amat = A.As<HypreParMatrix>();
      prec_.reset();  // the matrices it was built on are replaced below
      prec_shift_mat_.reset();
      prec_elastic_A_.Clear();
      prec_elastic_form_.reset();
      if (HasNonElasticIntegrators() && !prec_full_operator_) {
        // SetPreconditionerFullOperator(false): the AMG is built on the
        // rheology's stiffness alone, the added terms (gravity, fluid
        // folds) left to the Krylov method. The default builds it on the
        // full operator, shifted below to cover its (small, rigid-mode)
        // negative directions: an AMG on an indefinite matrix is an
        // indefinite preconditioner, which block MINRES cannot use.
        AssembleElasticPreconditionerMatrix();
        Amat = prec_elastic_A_.As<HypreParMatrix>();
      }
      if (FreeDisplacementBody() ||
          (prec_full_operator_ && HasNonElasticIntegrators())) {
        // A free body's operator is singular (rigid modes, region
        // rotations): AMG on it amplifies the kernel components of its
        // input without bound and the V-cycle stops being positive
        // definite in floating point (MINRES breaks down; CG stalls).
        // The AMG is therefore built on A + eps M (vector mass), which is
        // definite; the Krylov method still solves with A.
        prec_shift_mat_ = MassShiftedMatrix(
            *Amat, prec_full_operator_ ? prec_full_shift_ : 1.0);
        Amat = prec_shift_mat_.get();
      }
      auto amg = std::make_unique<HypreBoomerAMG>(*Amat);
      // Unknown-based systems AMG; see SetSystemsAMGOptions for why
      // nodal coarsening is not used. The coarsest level gets one
      // l1-Gauss-Seidel sweep instead of hypre's default Gaussian
      // elimination (which SetRelaxType installs): free-body and fluid
      // operators are singular, and a direct solve on a singular coarse
      // grid makes the V-cycle indefinite (MINRES then breaks down).
      detail::SetSystemsAMGOptions(*amg, *pfes_);
      HYPRE_BoomerAMGSetCycleRelaxType(*amg, 8, 3);
      amg->SetPrintLevel(0);
      prec_ = std::move(amg);
    } else
#endif
    {
      prec_ = std::make_unique<GSSmoother>(*A.As<SparseMatrix>());
    }
    prec_form_.reset();
    prec_A_.Clear();
    prec_ref_slot_ = -1;  // the fresh preconditioner sits on live matrices
    prec_stale_ = false;
    prec_baseline_its_ = -1;
    prec_setups_++;
  }
}

void LinearQuasiStaticProblemBase::SetupCG(const Operator& op, Solver& prec) {
#ifdef MFEM_USE_MPI
  if (pfes_) {
    cg_ = std::make_unique<CGSolver>(pfes_->GetComm());
  } else
#endif
  {
    cg_ = std::make_unique<CGSolver>();
  }
  // Operator before preconditioner: SetOperator would otherwise reset the
  // (reused) preconditioner onto the new operator.
  cg_->SetOperator(op);
  cg_->SetPreconditioner(prec);
  cg_->SetRelTol(rel_tol_);
  cg_->SetAbsTol(0.0);
  cg_->SetMaxIter(10000);
  cg_->SetPrintLevel(print_level_);
  cg_->iterative_mode = true;
}

void LinearQuasiStaticProblemBase::SetupSolver(OperatorHandle& A) {
  MFEM_VERIFY(!gauge_prec_only_,
              "SetGaugePreconditionerOnly: supported by the block-MINRES "
              "paths only, not the base CG solver.");
  SetupDefaultCG(A);
}

bool LinearQuasiStaticProblemBase::SolveLinearSystem(const Vector& B,
                                                     Vector& X) {
  if (!SetWarmStartTolerance(*cg_, *prec_, B)) {
    X = 0.0;
    return true;
  }
  cg_->Mult(B, X);
  NoteIterations(cg_->GetNumIterations());
  return cg_->GetConverged();
}

#ifdef MFEM_USE_MPI
namespace {
// base^{-1} plus the coarse correction Z S_c^{-1} Z^T (additive, so SPD
// when base is): the region-mean pressure modes solved exactly.
class CoarseCorrectedSolver : public Solver {
 public:
  CoarseCorrectedSolver(Solver& base, std::vector<Vector> Z,
                        DenseMatrix Sc_inv, MPI_Comm comm)
      : Solver(base.Height()),
        base_(base),
        Z_(std::move(Z)),
        Sc_inv_(std::move(Sc_inv)),
        comm_(comm) {}
  void Mult(const Vector& x, Vector& y) const override {
    base_.Mult(x, y);
    const int R = static_cast<int>(Z_.size());
    Vector loc(R), c(R), d(R);
    for (int r = 0; r < R; r++) {
      loc(r) = Z_[r] * x;
    }
    MPI_Allreduce(loc.GetData(), c.GetData(), R,
                  MPITypeMap<real_t>::mpi_type, MPI_SUM, comm_);
    Sc_inv_.Mult(c, d);
    for (int r = 0; r < R; r++) {
      y.Add(d(r), Z_[r]);
    }
  }
  void SetOperator(const Operator&) override {}

 private:
  Solver& base_;
  std::vector<Vector> Z_;
  DenseMatrix Sc_inv_;
  MPI_Comm comm_;
};
}  // namespace
#endif

void LinearQuasiStaticProblemBase::SetPressureSplitSolver(real_t theta) {
  if (pressure_split_) {
    return;
  }
  MFEM_VERIFY(SupportsPressureSplit(),
              "SetPressureSplitSolver: this class's solve carries machinery "
              "(rigid-mode projection, gravity blocks, interface "
              "constraints) the pressure-split route does not; it is "
              "supported on the clamped and the free (traction) solid.");
  MFEM_VERIFY(fluid_treatment_ == FluidTreatment::None && !gauge_prec_only_,
              "SetPressureSplitSolver: no fluid treatments on the "
              "pressure-split route.");
  Coefficient* kappa = nullptr;
  Coefficient* mu = nullptr;
  MFEM_VERIFY(stiffness_->IsotropicSplit(kappa, mu),
              "SetPressureSplitSolver: needs an isotropic(-split) rheology "
              "(a bulk and a shear modulus).");
  MFEM_VERIFY(!Rheology().EquilibriumMapping(),
              "SetPressureSplitSolver: non-natural reference states are not "
              "carried by the prototype.");
  const int dim = fes_->GetMesh()->Dimension();
  const int order = fes_->GetMaxElementOrder();
  MFEM_VERIFY(order >= 2,
              "SetPressureSplitSolver: Taylor-Hood needs displacement order "
              ">= 2 (the pressure runs one order below).");
  real_t ness = ess_tdof_list_.Size();
#ifdef MFEM_USE_MPI
  if (pfes_) {
    MPI_Allreduce(MPI_IN_PLACE, &ness, 1, MPITypeMap<real_t>::mpi_type,
                  MPI_SUM, pfes_->GetComm());
  }
#endif
  MFEM_VERIFY(ness > 0 || PressureSplitNullSpace(),
              "SetPressureSplitSolver: a free body needs its class to "
              "supply the rigid-mode projector (PressureSplitNullSpace).");

  MFEM_VERIFY(theta >= 0.0 && theta <= 1.0,
              "SetPressureSplitSolver: theta must lie in [0, 1].");
  pressure_split_ = true;
  psplit_theta_ = theta;
  psplit_kappa_ = kappa;
  psplit_mu_ = mu;
  if (psplit_per_region_) {
    MFEM_VERIFY(pfes_, "SetPressureSplitPerRegion: parallel builds only.");
    // The L2 host carries the assembled blocks; the unknowns are the
    // region copies of the continuous space (ProjectPerRegionPressure).
    psplit_fec_ = std::make_unique<L2_FECollection>(order - 1, dim);
    psplit_qfec_ = std::make_unique<H1_FECollection>(order - 1, dim);
    psplit_qfes_ = detail::MakeFESpace(*fes_, psplit_qfec_.get());
  } else {
    psplit_fec_ = std::make_unique<H1_FECollection>(order - 1, dim);
  }
  psplit_fes_ = detail::MakeFESpace(*fes_, psplit_fec_.get());
  psplit_p_ = detail::MakeGridFunction(psplit_fes_.get());
  *psplit_p_ = 0.0;
  // The route assembles its own blocks; the flavour slots hold primal
  // matrices only and stay disabled while it is on (StashFlavor /
  // TryRestoreFlavor guards), so a weight switch can never restore a
  // stale primal state over the split's reassembly.
  InvalidateFlavorSlots();
  MarkOperatorDirty();
}

void LinearQuasiStaticProblemBase::ClearPressureSplitSolver() {
  if (!pressure_split_) {
    return;
  }
  pressure_split_ = false;
  psplit_minres_.reset();
  psplit_schur_cc_.reset();
  psplit_proj_op_.reset();
  psplit_proj_prec_.reset();
#ifdef MFEM_USE_MPI
  psplit_shift_mat_.reset();
#endif
  psplit_tri_prec_.reset();
  psplit_neg_schur_.reset();
  psplit_prec_.reset();
  psplit_op_.reset();
  psplit_amg_.reset();
  psplit_schur_prec_.reset();
  psplit_BT_.reset();
  psplit_Amu_.Clear();
  psplit_Aprec_.Clear();
  psplit_B_.Clear();
  psplit_M_.Clear();
  psplit_Mw_.Clear();
  psplit_amu_form_.reset();
  psplit_aprec_form_.reset();
  psplit_b_form_.reset();
  psplit_mass_form_.reset();
  psplit_schur_form_.reset();
  psplit_coefs_.clear();
  psplit_p_.reset();
  psplit_fes_.reset();
  psplit_fec_.reset();
  psplit_qfes_.reset();
  psplit_qfec_.reset();
  psplit_pP_.Destroy();
#ifdef MFEM_USE_MPI
  psplit_P_.reset();
  psplit_PT_.reset();
#endif
  psplit_kappa_ = nullptr;
  psplit_mu_ = nullptr;
  psplit_theta_ = 0.0;
  psplit_version_ = -1;
  MarkOperatorDirty();
}

void LinearQuasiStaticProblemBase::AssemblePressureSplitBlocks() {
  const int dim = fes_->GetMesh()->Dimension();
  // Re-read the split each assembly: the pointers track the live
  // effective modulus (a redirectable, or a piecewise dispatch over the
  // regions' redirectables), so a weight switch needs no re-wiring here,
  // only this reassembly.
  Coefficient* kappa = nullptr;
  Coefficient* mu = nullptr;
  MFEM_VERIFY(stiffness_->IsotropicSplit(kappa, mu),
              "AssemblePressureSplitOperator: the rheology lost its "
              "isotropic split.");
  psplit_kappa_ = kappa;
  psplit_mu_ = mu;
  psplit_coefs_.clear();
  // p = lambda_p div u, lambda_p = kappa - theta 2 mu / d (floored at a
  // small multiple of kappa so that 1/lambda_p stays bounded: theta = 1
  // needs Poisson's ratio > 0); the displacement block carries the rest
  // of the volumetric term, lambda_A = (kappa - 2 mu / d) - lambda_p, so
  // the reformulation is exact for every theta.
  const real_t theta = psplit_theta_;
  const real_t two_over_d = 2.0 / dim;
  psplit_coefs_.push_back(std::make_unique<TransformedCoefficient>(
      mu, kappa, [theta, two_over_d](real_t m, real_t k) {
        return std::max(k - theta * two_over_d * m, real_t{1e-3} * k);
      }));
  Coefficient* lambda_p = psplit_coefs_.back().get();
  psplit_lambda_p_ = lambda_p;
  psplit_coefs_.push_back(std::make_unique<TransformedCoefficient>(
      mu, kappa, [theta, two_over_d](real_t m, real_t k) {
        const real_t lp =
            std::max(k - theta * two_over_d * m, real_t{1e-3} * k);
        return (k - two_over_d * m) - lp;
      }));
  Coefficient* lambda_A = psplit_coefs_.back().get();
  psplit_coefs_.push_back(std::make_unique<RatioCoefficient>(1.0, *lambda_p));
  Coefficient* inv_kappa = psplit_coefs_.back().get();  // 1 / lambda_p
  // The Schur weight inverts the effective shear, so it is floored at a
  // tiny multiple of kappa (the bulk modulus is the natural local scale):
  // the clamped weight interpolant can still touch zero between internal
  // nodes on a rapidly varying relaxation time, and 1/(2 mu) would turn
  // that into an unbounded (or, unclamped, indefinite) mass weight. The
  // floor lives in the PRECONDITIONER only — the operator blocks use mu
  // as assembled — so it costs preconditioner quality at the affected
  // points, never correctness.
  psplit_coefs_.push_back(std::make_unique<TransformedCoefficient>(
      mu, kappa, [](real_t m, real_t k) {
        return std::max(m, real_t{1e-10} * k);
      }));
  Coefficient* mu_floored = psplit_coefs_.back().get();
  psplit_coefs_.push_back(std::make_unique<RatioCoefficient>(0.5, *mu_floored));
  Coefficient* inv_mu2 = psplit_coefs_.back().get();
  psplit_coefs_.push_back(std::make_unique<SumCoefficient>(
      *inv_kappa, *inv_mu2, psplit_schur_scale_, psplit_schur_scale_));
  Coefficient* schur_w = psplit_coefs_.back().get();

  auto set_type = [this](OperatorHandle& h) {
    h.Clear();
#ifdef MFEM_USE_MPI
    if (pfes_) {
      h.SetType(Operator::Hypre_ParCSR);
      return;
    }
#endif
    h.SetType(Operator::MFEM_SPARSEMAT);
  };
  Array<int> none;

  // A = 2 mu eps : eps + lambda_A div div (theta = 0: 2 mu dev eps : dev
  // eps), essential dofs eliminated (DIAG_ONE, as the primal path).
  psplit_amu_form_ = detail::MakeBilinearForm(fes_);
  psplit_amu_form_->AddDomainIntegrator(
      new ElasticityIntegrator(*lambda_A, *mu));
  psplit_amu_form_->Assemble();
  set_type(psplit_Amu_);
  psplit_amu_form_->FormSystemMatrix(ess_tdof_list_, psplit_Amu_);

  // B = (q, div u); the essential trial columns are eliminated, their
  // boundary data returning through the lift per solve.
  psplit_b_form_ = detail::MakeMixedBilinearForm(fes_, psplit_fes_.get());
  psplit_b_form_->AddDomainIntegrator(new VectorDivergenceIntegrator());
  psplit_b_form_->Assemble();
  set_type(psplit_B_);
  psplit_b_form_->FormRectangularSystemMatrix(ess_tdof_list_, none,
                                              psplit_B_);

  // The compressibility block, assembled positively as (1/kappa) M_p and
  // applied with weight -1 in the block operator.
  psplit_mass_form_ = detail::MakeBilinearForm(psplit_fes_.get());
  psplit_mass_form_->AddDomainIntegrator(new MassIntegrator(*inv_kappa));
  psplit_mass_form_->Assemble();
  set_type(psplit_M_);
  psplit_mass_form_->FormSystemMatrix(none, psplit_M_);

  // The preconditioner blocks: the house AMG on the kappa-free
  // full-gradient operator 2 mu eps : eps, and the diagonal of the
  // pressure mass weighted by 1/kappa + 1/(2 mu) (spectrally equivalent
  // to the Schur complement, Stokes-fashion). The essential diagonal is
  // KEPT at its assembled value, not set to one: with a collapsed
  // effective shear the clamped boundary can sit in a region where the
  // kappa-free block's entries are many orders below one, and unit
  // essential rows put an artificial jump there that derails the AMG
  // (MINRES dies on a NaN preconditioner application). The operator
  // blocks stay DIAG_ONE; a preconditioner only needs to stay SPD, and
  // the kept diagonal is the assembled (positive) one.
  psplit_aprec_form_ = detail::MakeBilinearForm(fes_);
  psplit_aprec_form_->AddDomainIntegrator(
      new ElasticityIntegrator(*mu, 0.0, 1.0));
  psplit_aprec_form_->SetDiagonalPolicy(Operator::DIAG_KEEP);
  psplit_aprec_form_->Assemble();
#ifdef MFEM_USE_MPI
  if (pfes_) {
    // The legacy parallel elimination hard-codes unit diagonals, so keep
    // them by hand: save the assembled diagonal at the essential dofs,
    // eliminate, and write it back (the diagonal block is rank-local).
    auto* pform = static_cast<ParBilinearForm*>(psplit_aprec_form_.get());
    pform->Finalize();
    HypreParMatrix* Afull = pform->ParallelAssemble();
    SparseMatrix dblk;
    Afull->GetDiag(dblk);
    Vector keep(ess_tdof_list_.Size());
    for (int i = 0; i < ess_tdof_list_.Size(); i++) {
      keep[i] = dblk(ess_tdof_list_[i], ess_tdof_list_[i]);
    }
    delete Afull->EliminateRowsCols(ess_tdof_list_);
    Afull->GetDiag(dblk);
    for (int i = 0; i < ess_tdof_list_.Size(); i++) {
      dblk(ess_tdof_list_[i], ess_tdof_list_[i]) = keep[i];
    }
    psplit_Aprec_.Reset(Afull, true);
  } else
#endif
  {
    set_type(psplit_Aprec_);
    psplit_aprec_form_->FormSystemMatrix(ess_tdof_list_, psplit_Aprec_);
  }

  psplit_schur_form_ = detail::MakeBilinearForm(psplit_fes_.get());
  psplit_schur_form_->AddDomainIntegrator(new MassIntegrator(*schur_w));
  psplit_schur_form_->Assemble();
  set_type(psplit_Mw_);
  psplit_schur_form_->FormSystemMatrix(none, psplit_Mw_);
#ifdef MFEM_USE_MPI
  if (psplit_per_region_) {
    ProjectPerRegionPressure();
  }
#endif

#ifdef MFEM_USE_MPI
  if (pfes_) {
    psplit_amg_.reset();
    psplit_shift_mat_.reset();
    HypreParMatrix* Ap = psplit_Aprec_.As<HypreParMatrix>();
    if (PressureSplitNullSpace()) {
      // Free body: AMG on the definite A + eps M (as the primal route).
      psplit_shift_mat_ = MassShiftedMatrix(*Ap);
      Ap = psplit_shift_mat_.get();
    }
    auto amg = std::make_unique<HypreBoomerAMG>(*Ap);
    // The systems settings of SetupDefaultPreconditioner.
    detail::SetSystemsAMGOptions(*amg, *pfes_);
    HYPRE_BoomerAMGSetCycleRelaxType(*amg, 8, 3);
    amg->SetPrintLevel(0);
    psplit_amg_ = std::move(amg);
    if (psplit_cheb_order_ > 0) {
      Vector diag;
      psplit_Mw_.As<HypreParMatrix>()->GetDiag(diag);
      Array<int> no_ess;
      psplit_schur_prec_ = std::make_unique<OperatorChebyshevSmoother>(
          *psplit_Mw_.Ptr(), diag, no_ess, psplit_cheb_order_,
          pfes_->GetComm());
    } else {
      psplit_schur_prec_ =
          std::make_unique<HypreDiagScale>(*psplit_Mw_.As<HypreParMatrix>());
    }
  } else
#endif
  {
    // Serial: Jacobi on both blocks (MINRES needs a symmetric positive
    // definite preconditioner, which rules the serial Gauss-Seidel out);
    // the serial path is kept working, not tuned.
    psplit_amg_ =
        std::make_unique<DSmoother>(*psplit_Aprec_.As<SparseMatrix>());
    psplit_schur_prec_ =
        std::make_unique<DSmoother>(*psplit_Mw_.As<SparseMatrix>());
  }

}

LinearQuasiStaticProblemBase::PressureSplitParts
LinearQuasiStaticProblemBase::AssemblePressureSplitParts() {
  AssemblePressureSplitBlocks();
  PressureSplitParts parts;
  parts.B = psplit_B_.Ptr();
  parts.M = psplit_M_.Ptr();
  parts.amg = psplit_amg_.get();
  parts.schur = psplit_schur_prec_.get();
  parts.lambda_p = psplit_lambda_p_;
  parts.np = psplit_fes_->GetTrueVSize();
#ifdef MFEM_USE_MPI
  if (psplit_per_region_) {
    parts.np = psplit_P_->Width();
  }
#endif
  return parts;
}

void LinearQuasiStaticProblemBase::AssemblePressureSplitOperator() {
  operator_version_++;
  AssemblePressureSplitBlocks();
  psplit_offsets_.SetSize(3);
  psplit_offsets_[0] = 0;
  psplit_offsets_[1] = fes_->GetTrueVSize();
  int np = psplit_fes_->GetTrueVSize();
#ifdef MFEM_USE_MPI
  if (psplit_per_region_) {
    np = psplit_P_->Width();
  }
#endif
  psplit_offsets_[2] = psplit_offsets_[1] + np;
  psplit_op_ = std::make_unique<BlockOperator>(psplit_offsets_);
  // Free body: project the rigid modes out of the displacement block and
  // its preconditioner (an unprojected AMG amplifies their round-off).
  psplit_proj_op_.reset();
  psplit_proj_prec_.reset();
  Operator* a00 = psplit_Amu_.Ptr();
  Solver* p00 = psplit_amg_.get();
  if (const NullSpaceProjector* P = PressureSplitNullSpace()) {
    psplit_proj_op_ = std::make_unique<ProjectedOperator>(*a00, *P);
    psplit_proj_prec_ = std::make_unique<ProjectedSolver>(*P);
    psplit_proj_prec_->SetSolver(*psplit_amg_);
    a00 = psplit_proj_op_.get();
    p00 = psplit_proj_prec_.get();
  }
  psplit_op_->SetBlock(0, 0, a00);
  psplit_BT_ = std::make_unique<TransposeOperator>(psplit_B_.Ptr());
  psplit_op_->SetBlock(0, 1, psplit_BT_.get());
  psplit_op_->SetBlock(1, 0, psplit_B_.Ptr());
  psplit_op_->SetBlock(1, 1, psplit_M_.Ptr(), -1.0);

  // Optional coarse correction of the Schur block by the region-mean
  // pressures (SetPressureSplitCoarse).
  psplit_schur_cc_.reset();
  Solver* sprec = psplit_schur_prec_.get();
#ifdef MFEM_USE_MPI
  if (psplit_coarse_ && pfes_) {
    MPI_Comm comm = pfes_->GetComm();
    Mesh* mesh = fes_->GetMesh();
    const Array<int>& attrs = mesh->attributes;
    std::vector<Vector> Z;
    Array<int> vdofs;
    for (int r = 0; r < attrs.Size(); r++) {
      Vector z(np);
      z = 0.0;
      if (psplit_per_region_) {
        // copy r's live dofs: where P^T (host indicator of region r) > 0
        Vector ind(psplit_P_->Height());
        ind = 0.0;
        for (int e = 0; e < mesh->GetNE(); e++) {
          if (mesh->GetAttribute(e) != attrs[r]) continue;
          psplit_fes_->GetElementVDofs(e, vdofs);
          for (int v : vdofs) {
            const int t = static_cast<ParFiniteElementSpace*>(psplit_fes_.get())
                              ->GetLocalTDofNumber(v >= 0 ? v : -1 - v);
            if (t >= 0) ind(t) = 1.0;
          }
        }
        Vector w(np);
        psplit_PT_->Mult(ind, w);
        for (int i = 0; i < np; i++) z(i) = w(i) > 0.0 ? 1.0 : 0.0;
      } else {
        auto* qf = static_cast<ParFiniteElementSpace*>(psplit_fes_.get());
        for (int e = 0; e < mesh->GetNE(); e++) {
          if (mesh->GetAttribute(e) != attrs[r]) continue;
          qf->GetElementVDofs(e, vdofs);
          for (int v : vdofs) {
            const int t = qf->GetLocalTDofNumber(v >= 0 ? v : -1 - v);
            if (t >= 0) z(t) = 1.0;
          }
        }
      }
      Z.push_back(std::move(z));
    }
    // S_c = Z^T (M_c + B A^{-1} B^T) Z, A^{-1} by AMG-preconditioned CG
    const int R = static_cast<int>(Z.size());
    CGSolver cg(comm);
    cg.SetOperator(*a00);
    cg.SetPreconditioner(*p00);
    cg.SetRelTol(1e-10);
    cg.SetAbsTol(0.0);
    cg.SetMaxIter(5000);
    cg.SetPrintLevel(0);
    std::vector<Vector> SZ(R);
    for (int r = 0; r < R; r++) {
      Vector t(psplit_offsets_[1]), y(psplit_offsets_[1]);
      psplit_B_->MultTranspose(Z[r], t);
      if (const NullSpaceProjector* P = PressureSplitNullSpace()) {
        P->Project(t);
      }
      y = 0.0;
      cg.Mult(t, y);
      SZ[r].SetSize(np);
      psplit_B_->Mult(y, SZ[r]);
      Vector mz(np);
      psplit_M_->Mult(Z[r], mz);
      SZ[r] += mz;
    }
    DenseMatrix Sc(R);
    for (int a = 0; a < R; a++) {
      for (int b = 0; b < R; b++) {
        real_t loc = Z[a] * SZ[b], glob = 0.0;
        MPI_Allreduce(&loc, &glob, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                      comm);
        Sc(a, b) = glob;
      }
    }
    Sc.Symmetrize();
    Sc.Invert();
    psplit_schur_cc_ = std::make_unique<CoarseCorrectedSolver>(
        *psplit_schur_prec_, std::move(Z), Sc, comm);
    sprec = psplit_schur_cc_.get();
  }
#endif
  psplit_prec_ =
      std::make_unique<BlockDiagonalPreconditioner>(psplit_offsets_);
  psplit_prec_->SetDiagonalBlock(0, p00);
  psplit_prec_->SetDiagonalBlock(1, sprec);

  // Experimental alternative: the block lower-triangular preconditioner
  // [[P_A, 0], [B, -P_S]]^{-1} (the exact factor of the saddle with the
  // Schur complement S = M_c + B A^{-1} B^T) with flexible GMRES.
  psplit_tri_prec_.reset();
  psplit_neg_schur_.reset();
  if (psplit_triangular_) {
    psplit_neg_schur_ =
        std::make_unique<ScaledOperator>(sprec, -1.0);
    psplit_tri_prec_ =
        std::make_unique<BlockLowerTriangularPreconditioner>(psplit_offsets_);
    psplit_tri_prec_->SetDiagonalBlock(0, p00);
    psplit_tri_prec_->SetBlock(1, 0, psplit_B_.Ptr());
    psplit_tri_prec_->SetDiagonalBlock(1, psplit_neg_schur_.get());
  }
#ifdef MFEM_USE_MPI
  if (pfes_) {
    if (psplit_triangular_) {
      auto gm = std::make_unique<FGMRESSolver>(pfes_->GetComm());
      gm->SetKDim(100);
      psplit_minres_ = std::move(gm);
    } else {
      psplit_minres_ = std::make_unique<MINRESSolver>(pfes_->GetComm());
    }
  } else
#endif
  {
    if (psplit_triangular_) {
      auto gm = std::make_unique<FGMRESSolver>();
      gm->SetKDim(100);
      psplit_minres_ = std::move(gm);
    } else {
      psplit_minres_ = std::make_unique<MINRESSolver>();
    }
  }
  // Operator before preconditioner, as in SetupCG.
  psplit_minres_->SetOperator(*psplit_op_);
  if (psplit_triangular_) {
    psplit_minres_->SetPreconditioner(*psplit_tri_prec_);
  } else {
    psplit_minres_->SetPreconditioner(*psplit_prec_);
  }
  psplit_minres_->SetRelTol(rel_tol_);
  psplit_minres_->SetAbsTol(0.0);
  psplit_minres_->SetMaxIter(10000);
  psplit_minres_->SetPrintLevel(print_level_);
  psplit_minres_->iterative_mode = true;

  psplit_version_ = operator_version_;
  operator_dirty_ = false;
  eps_only_dirty_ = false;
  assemblies_++;
  prec_setups_++;
}

bool LinearQuasiStaticProblemBase::PressureSplitSolve() {
  if (operator_dirty_ || eps_only_dirty_ ||
      psplit_version_ != operator_version_) {
    AssemblePressureSplitOperator();
  }
  rhs_ = *b_;
  rhs_ += increment_;
  // u row: fold the boundary data into the reduced system on the scratch
  // copy rhs_, warm start from the interior of u_ (copy_interior = 1).
  Vector Xu, Bu;
  psplit_amu_form_->FormLinearSystem(ess_tdof_list_, *u_, rhs_, psplit_Amu_,
                                     Xu, Bu, 1);
  // p row: the reduced right-hand side is minus the divergence coupling
  // of the boundary lift (the equation's own data is zero).
  Vector zero_p(psplit_fes_->GetVSize());
  zero_p = 0.0;
  Array<int> none;
  Vector Xtmp, Bp;
  // A scratch handle: the block operator's psplit_B_ may be a projection
  // of the form's matrix (per-region pressure) and must not be replaced.
  OperatorHandle B_scratch(psplit_B_.Type());
  psplit_b_form_->FormRectangularLinearSystem(ess_tdof_list_, none, *u_,
                                              zero_p, B_scratch, Xtmp, Bp);
  BlockVector X(psplit_offsets_), B(psplit_offsets_);
  X.GetBlock(0) = Xu;
  B.GetBlock(0) = Bu;
#ifdef MFEM_USE_MPI
  if (psplit_per_region_) {
    if (psplit_pP_.Size() != psplit_P_->Width()) {
      psplit_pP_.SetSize(psplit_P_->Width());
      psplit_pP_ = 0.0;
    }
    X.GetBlock(1) = psplit_pP_;  // the pressure warm start
    psplit_PT_->Mult(Bp, B.GetBlock(1));
  } else
#endif
  {
    Vector Xp;
    psplit_p_->GetTrueDofs(Xp);
    X.GetBlock(1) = Xp;  // the pressure warm start
    B.GetBlock(1) = Bp;
  }
  const NullSpaceProjector* nullspace = PressureSplitNullSpace();
  if (nullspace) {
    nullspace->Project(B.GetBlock(0));
    nullspace->Project(X.GetBlock(0));
  }
  bool ok = true;
  bool solve = true;
  if (psplit_triangular_) {
    // FGMRES stops on the unpreconditioned residual; the warm start's
    // target is the cold start's, rel_tol ||B||.
    const real_t nb = std::sqrt(Dot(B, B));
    solve = nb > 0.0;
    psplit_minres_->SetRelTol(0.0);
    psplit_minres_->SetAbsTol(rel_tol_ * nb);
  } else {
    solve = SetWarmStartTolerance(*psplit_minres_, *psplit_prec_, B);
  }
  if (solve) {
    psplit_minres_->Mult(B, X);
    NoteIterations(psplit_minres_->GetNumIterations());
    ok = psplit_minres_->GetConverged();
  } else {
    X = 0.0;
  }
  // Copy the solved block back into the FormLinearSystem vector BEFORE the
  // recovery: in serial Xu ALIASES u_'s data and RecoverFEMSolution is a
  // no-op that assumes the solver wrote through it; in parallel the
  // recovery prolongates whatever vector it is given.
  if (nullspace) {
    nullspace->Project(X.GetBlock(0));
  }
  Xu = X.GetBlock(0);
  psplit_amu_form_->RecoverFEMSolution(Xu, rhs_, *u_);
#ifdef MFEM_USE_MPI
  if (psplit_per_region_) {
    psplit_pP_ = X.GetBlock(1);
    Vector pl(psplit_P_->Height());
    psplit_P_->Mult(psplit_pP_, pl);
    psplit_p_->SetFromTrueDofs(pl);
  } else
#endif
  {
    psplit_p_->SetFromTrueDofs(X.GetBlock(1));
  }
  return ok;
}

#ifdef MFEM_USE_MPI
void LinearQuasiStaticProblemBase::ProjectPerRegionPressure() {
  auto* lfes = static_cast<ParFiniteElementSpace*>(psplit_fes_.get());
  auto* qfes = static_cast<ParFiniteElementSpace*>(psplit_qfes_.get());
  if (!psplit_P_) {  // the spaces never change: build P once
    ParDiscreteLinearOperator inj(qfes, lfes);
    inj.AddDomainInterpolator(new IdentityInterpolator);
    inj.Assemble();
    inj.Finalize();
    std::unique_ptr<HypreParMatrix> G(inj.ParallelAssemble());
    Mesh* mesh = lfes->GetMesh();
    const Array<int>& attrs = mesh->attributes;  // global, sorted
    Array2D<const HypreParMatrix*> blocks(1, attrs.Size());
    std::vector<std::unique_ptr<HypreParMatrix>> gr;
    Array<int> vdofs;
    for (int r = 0; r < attrs.Size(); r++) {
      Vector d(lfes->GetTrueVSize());
      d = 0.0;
      for (int e = 0; e < mesh->GetNE(); e++) {
        if (mesh->GetAttribute(e) != attrs[r]) {
          continue;
        }
        lfes->GetElementVDofs(e, vdofs);
        for (int v : vdofs) {
          const int t = lfes->GetLocalTDofNumber(v >= 0 ? v : -1 - v);
          if (t >= 0) {
            d(t) = 1.0;
          }
        }
      }
      gr.push_back(std::make_unique<HypreParMatrix>(*G));
      gr.back()->ScaleRows(d);
      blocks(0, r) = gr.back().get();
    }
    psplit_P_.reset(HypreParMatrixFromBlocks(blocks));
    psplit_PT_.reset(psplit_P_->Transpose());
  }
  // Region copies with no support in their region (the continuous dofs
  // away from it) have empty rows and columns: an identity there makes
  // them decoupled zeros.
  auto project = [this](HypreParMatrix& M, Vector* dead_in, Vector& dead) {
    std::unique_ptr<HypreParMatrix> MP(RAP(&M, psplit_P_.get()));
    if (!dead_in) {
      Vector diag;
      MP->GetDiag(diag);
      dead.SetSize(diag.Size());
      for (int i = 0; i < diag.Size(); i++) {
        dead(i) = diag(i) == 0.0 ? 1.0 : 0.0;
      }
    }
    SparseMatrix id(dead.Size());
    for (int i = 0; i < dead.Size(); i++) {
      id.Add(i, i, dead(i));
    }
    id.Finalize();
    HypreParMatrix idp(MP->GetComm(), MP->GetGlobalNumRows(), MP->RowPart(),
                       &id);
    return std::unique_ptr<HypreParMatrix>(Add(1.0, *MP, 1.0, idp));
  };
  Vector dead;
  auto M = project(*psplit_M_.As<HypreParMatrix>(), nullptr, dead);
  auto Mw = project(*psplit_Mw_.As<HypreParMatrix>(), &dead, dead);
  HypreParMatrix* BP = ParMult(psplit_PT_.get(), psplit_B_.As<HypreParMatrix>());
  psplit_M_.Reset(M.release(), true);
  psplit_Mw_.Reset(Mw.release(), true);
  psplit_B_.Reset(BP, true);
}
#endif

std::unique_ptr<BilinearForm> LinearQuasiStaticProblemBase::AssembleMassOperator(
    Coefficient* rho, OperatorHandle& M) {
  auto form = detail::MakeBilinearForm(fes_);
  form->AddDomainIntegrator(rho ? new VectorMassIntegrator(*rho)
                                : new VectorMassIntegrator());
  form->Assemble();
#ifdef MFEM_USE_MPI
  if (pfes_) {
    M.SetType(Operator::Hypre_ParCSR);
  }
#endif
  Array<int> empty;
  form->FormSystemMatrix(empty, M);
  return form;
}

real_t LinearQuasiStaticProblemBase::Dot(const Vector& x,
                                         const Vector& y) const {
#ifdef MFEM_USE_MPI
  if (pfes_) {
    return InnerProduct(pfes_->GetComm(), x, y);
  }
#endif
  return InnerProduct(x, y);
}

bool LinearQuasiStaticProblemBase::SetWarmStartTolerance(
    IterativeSolver& solver, Solver& prec, const Vector& B) const {
  warm_z_.SetSize(B.Size());
  prec.Mult(B, warm_z_);
  const real_t nom = Dot(B, warm_z_);
  if (!(nom > 0.0)) {
    return false;
  }
  solver.SetAbsTol(rel_tol_ * std::sqrt(nom));
  return true;
}

void LinearQuasiStaticProblemBase::RegisterFields(DataCollection& dc) {
  dc.RegisterField("displacement", u_.get());
}

// ---------------------------------------------------------------------------

LinearQuasiStaticTractionProblem::LinearQuasiStaticTractionProblem(
    FiniteElementSpace* fes, const AdGIA::Rheology& rheology,
    VectorCoefficient& traction, const Array<int>& bdr_marker)
    : LinearQuasiStaticProblemBase(fes, rheology), marker_(bdr_marker) {
  RegisterTimeDependent(traction);
  b_->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(traction), marker_);
}

const NullSpaceProjector& LinearQuasiStaticTractionProblem::RigidModes() {
  if (!projector_) {
    // Rotations of the mapped positions when the rheology carries a
    // non-natural reference state; identical to the plain rotations at
    // the identity (nullptr for a natural reference state).
    projector_ = MakeRigidModeProjector(*fes_, Rheology().EquilibriumMapping());
  }
  return *projector_;
}

std::vector<real_t> LinearQuasiStaticTractionProblem::RigidPairResiduals() {
  EnsureOperator();
  real_t a_max = 0.0;
#ifdef MFEM_USE_MPI
  if (pfes_) {
    auto* hyp = A_.As<HypreParMatrix>();
    SparseMatrix diag, offd;
    HYPRE_BigInt* cmap = nullptr;
    hyp->GetDiag(diag);
    hyp->GetOffd(offd, cmap);
    real_t local = std::max(diag.MaxNorm(), offd.MaxNorm());
    MPI_Allreduce(&local, &a_max, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                  pfes_->GetComm());
  } else
#endif
  {
    a_max = A_.As<SparseMatrix>()->MaxNorm();
  }
  const auto& P = RigidModes();
  std::vector<real_t> out;
  Vector r(A_.Ptr()->Height());
  for (int i = 0; i < P.Size(); i++) {
    const Vector& n = P.Basis(i);
    A_.Ptr()->Mult(n, r);
    const real_t norm = std::sqrt(P.Dot(n, n));
    out.push_back(std::sqrt(P.Dot(r, r)) /
                  (a_max * std::max(norm, real_t{1e-300})));
  }
  return out;
}

void LinearQuasiStaticTractionProblem::SetupSolver(OperatorHandle& A) {
  const auto& P = RigidModes();
  SetupDefaultPreconditioner(A);
  // CG on P A P with the preconditioner P M P: an unprojected preconditioner
  // amplifies the round-off component along the (near-)null rigid modes.
  projected_prec_ = std::make_unique<ProjectedSolver>(P);
  projected_prec_->SetSolver(*prec_);
  projected_op_ = std::make_unique<ProjectedOperator>(*A.Ptr(), P);
  SetupCG(*projected_op_, *projected_prec_);
  // The outer wrapper projects the load and the warm start before, and the
  // solution after, the CG solve.
  projected_ = std::make_unique<ProjectedSolver>(P);
  projected_->SetSolver(*cg_);
  projected_->iterative_mode = true;
  projected_->SetGauge(gauge_M_.Ptr());
}

void LinearQuasiStaticTractionProblem::SetMassWeightedGauge(Coefficient* rho) {
  gauge_M_.Clear();
  gauge_form_ = AssembleMassOperator(rho, gauge_M_);
  if (projected_) {
    projected_->SetGauge(gauge_M_.Ptr());
  }
}

void LinearQuasiStaticTractionProblem::SetEuclideanGauge() {
  gauge_M_.Clear();
  gauge_form_.reset();
  if (projected_) {
    projected_->SetGauge(nullptr);
  }
}

bool LinearQuasiStaticTractionProblem::SolveLinearSystem(const Vector& B,
                                                         Vector& X) {
  // (P M P B, B) = (M P B, P B): the cold-start norm of the projected load.
  if (!SetWarmStartTolerance(*cg_, *projected_prec_, B)) {
    X = 0.0;
    return true;
  }
  projected_->Mult(B, X);
  NoteIterations(cg_->GetNumIterations());
  return cg_->GetConverged();
}

// ---------------------------------------------------------------------------

LinearQuasiStaticClampedProblem::LinearQuasiStaticClampedProblem(
    FiniteElementSpace* fes, const AdGIA::Rheology& rheology,
    const Array<int>& ess_bdr, VectorCoefficient& traction,
    const Array<int>& traction_marker, VectorCoefficient* dirichlet)
    : LinearQuasiStaticProblemBase(fes, rheology),
      ess_bdr_(ess_bdr),
      marker_(traction_marker),
      dirichlet_(dirichlet) {
  SetEssentialBoundary(ess_bdr_);
  if (!dirichlet_) {
    Vector zero(fes_->GetVDim());
    zero = 0.0;
    zero_ = std::make_unique<VectorConstantCoefficient>(zero);
    dirichlet_ = zero_.get();
  } else {
    RegisterTimeDependent(*dirichlet_);
  }
  RegisterTimeDependent(traction);
  b_->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(traction), marker_);
}

void LinearQuasiStaticClampedProblem::UpdateBoundaryValues(real_t /*t*/) {
  u_->ProjectBdrCoefficient(*dirichlet_, ess_bdr_);
}

}  // namespace AdGIA
