/**
 * @file quasi_static_problem.cpp
 * @brief Implementation of LinearQuasiStaticProblemBase,
 * LinearQuasiStaticTractionProblem and LinearQuasiStaticClampedProblem.
 */

#include "AdGIA/quasi_static_problem.hpp"

#include <cmath>

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
  operator_dirty_ = true;
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
  // Always reassemble: the same coefficient objects may carry new values.
  stiffness_->SetRelaxationWeights(beta);
  operator_dirty_ = true;
}

void LinearQuasiStaticProblemBase::ClearRelaxationWeights() {
  if (stiffness_->IsRelaxed()) {
    stiffness_->ClearRelaxationWeights();
    operator_dirty_ = true;
  }
}

void LinearQuasiStaticProblemBase::AssembleOperator() {
  if (a_ && prec_ && !prec_stale_ && prec_reuse_ > 1.0 && !prec_form_ &&
      !prec_A_.Ptr()) {
    // The preconditioner was built on the current solver matrix and stays on
    // it: keep that form and matrix alive while the preconditioner is
    // reused. (Later reassemblies leave prec_form_ alone; their matrices
    // go.) With a gauged fluid the solver matrix is the regularised one.
    // A matrix already captured by the RescaleGaugeOperator fast path
    // (prec_A_ set with no form behind it) is likewise left alone.
    if (a_solve_form_) {
      prec_form_ = std::move(a_solve_form_);
      prec_A_ = A_solve_;
      prec_A_.SetOperatorOwner(A_solve_.OwnsOperator());
      A_solve_.SetOperatorOwner(false);
    } else {
      prec_form_ = std::move(a_);
      prec_A_ = A_;
      prec_A_.SetOperatorOwner(A_.OwnsOperator());
      A_.SetOperatorOwner(false);
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
  MFEM_ASSERT(HasGaugedFluid() && fluid_op_.HasUnitCache(),
              "RescaleGaugeOperator: no cached unit penalty.");
  // Keep the matrix the (reused) preconditioner was built on alive, as
  // in AssembleOperator; here there is no form to move, the matrix
  // handle alone is captured.
  if (a_ && prec_ && !prec_stale_ && prec_reuse_ > 1.0 && !prec_form_ &&
      !prec_A_.Ptr()) {
    prec_A_ = A_solve_;
    prec_A_.SetOperatorOwner(A_solve_.OwnsOperator());
    A_solve_.SetOperatorOwner(false);
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
  MFEM_VERIFY(opts.dt_over_tau > 0.0 && opts.escalate >= 1.0 &&
                  opts.beta_max >= opts.dt_over_tau && opts.max_steps > 0,
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
  operator_dirty_ = true;
}

void LinearQuasiStaticProblemBase::SetFluid(
    const Array<int>& fluid_marker, Coefficient& mu_scale,
    const GaugePenaltyOptions& opts, Diffeomorphism* map) {
  ConfigureFluidOperator(fluid_marker, mu_scale, opts.epsilon, opts.form,
                         map);
  fluid_treatment_ = FluidTreatment::Penalty;
  gauge_refinements_ = opts.refinements;
  operator_dirty_ = true;
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
  operator_dirty_ = true;
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
    eps_only_dirty_ = true;
  } else {
    operator_dirty_ = true;
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
  operator_dirty_ = true;
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
  // with their loads reapplied — equivalent, by linearity, to the
  // historical zero-companion-load increment accumulation, with the
  // same residuals ||eps Q delta_k|| recorded.
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
  for (int e = 0; e < fes_->GetNE(); ++e) {
    const int attr = fes_->GetMesh()->GetAttribute(e);
    const Array<int>& marker = fluid_op_.Marker();
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
  maxwell_report_.stop = "max_steps";

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
    for (int i = 0; i < m; ++i) {
      Vector dri(rk);
      dri -= aa_r[i];
      for (int j = i; j < m; ++j) {
        Vector drj(rk);
        drj -= aa_r[j];
        const real_t v = Dot(dri, drj);
        N(i, j) = v;
        N(j, i) = v;
      }
      rhs[i] = Dot(dri, rk);
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

  const char* stop = "max_steps";
  real_t delta = infinity();
  long its_first = -1;
  int at_cap = 0;
  for (int n = 1; n <= o.max_steps; ++n) {
    if (1.0 / (1.0 + beta) != eps_cur) {
      set_operator(1.0 / (1.0 + beta));
      aa_g.clear();
      aa_r.clear();
      if (first) {
        w.SetSize(X_.Size());
        w = 0.0;
        r.SetSize(X_.Size());
        BuildMaxwellSolidDofs();
        first = false;
      }
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

  maxwell_report_.stop = stop;
  maxwell_report_.delta_returned = delta;
  rel_tol_ = rel_tol_base;  // the endgame runs at full tolerance
  bool polished = false;
  if (stop != std::string("converged") && stop != std::string("stagnation")) {
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
  a_->RecoverFEMSolution(X_, rhs_, *u_);
  return ok;
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
      auto amg = std::make_unique<HypreBoomerAMG>(*A.As<HypreParMatrix>());
      // Systems AMG with the nodal coarsening and coarse-grid smoother of
      // SetElasticityOptions, but without its rigid-body interpolation,
      // which needs Ordering::byVDIM (MFEM 4.10 verifies this; the
      // displacement spaces here are byNODES).
      amg->SetSystemsOptions(pfes_->GetVDim(), true);
      HYPRE_BoomerAMGSetNodal(*amg, 4);
      HYPRE_BoomerAMGSetNodalDiag(*amg, 1);
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
  Vector z(B.Size());
  prec.Mult(B, z);
  const real_t nom = Dot(B, z);
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
