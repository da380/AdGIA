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
  if (a_ && prec_ && !prec_stale_ && prec_reuse_ > 1.0 && !prec_form_) {
    // The preconditioner was built on the current solver matrix and stays on
    // it: keep that form and matrix alive while the preconditioner is
    // reused. (Later reassemblies leave prec_form_ alone; their matrices
    // go.) With a gauged fluid the solver matrix is the regularised one.
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
  q_form_.reset();
  if (HasGaugedFluid()) {
#ifdef MFEM_USE_MPI
    if (pfes_) {
      Q_.SetType(Operator::Hypre_ParCSR);
      A_solve_.SetType(Operator::Hypre_ParCSR);
    }
#endif
    // eps Q on true dofs, unconstrained: GaugeRefine() zeroes the essential
    // rows of its residuals instead.
    q_form_ = detail::MakeBilinearForm(fes_, gauge_integrators_.get());
    q_form_->Assemble();
    Array<int> empty;
    q_form_->FormSystemMatrix(empty, Q_);
    // A + eps Q in one assembly: borrow the physical integrators and append
    // the penalty (the borrowing form owns none of them).
    a_solve_form_ = detail::MakeBilinearForm(fes_, integrators_.get());
    a_solve_form_->AddDomainIntegrator(gauge_integ_, gauge_marker_);
    a_solve_form_->Assemble();
    a_solve_form_->FormSystemMatrix(ess_tdof_list_, A_solve_);
  }
  SetupSolver(HasGaugedFluid() ? A_solve_ : A_);
  operator_dirty_ = false;
  assemblies_++;
}

void LinearQuasiStaticProblemBase::WarnGaugeContraction() const {
  if (gauge_residuals_.size() < 2) {
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
  }
}

const OperatorHandle& LinearQuasiStaticProblemBase::SystemMatrix() {
  EnsureOperator();
  return A_;
}

bool LinearQuasiStaticProblemBase::Solve() {
  solves_++;
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

void LinearQuasiStaticProblemBase::SetGaugedFluid(
    const Array<int>& fluid_marker, Coefficient& mu_gauge, real_t epsilon,
    int refinements, GaugePenalty penalty, Diffeomorphism* map) {
  MFEM_VERIFY(fluid_marker.Size() == fes_->GetMesh()->attributes.Max(),
              "SetGaugedFluid: the fluid marker must be sized to "
              "attributes.Max().");
  MFEM_VERIFY(epsilon > 0.0, "SetGaugedFluid: epsilon must be positive.");
  // A supplied map — the identity included — switches the Deviatoric
  // branch to ElasticTensorIntegrator(C, map), so that the two sides of a
  // change-of-variables identity assemble with the SAME integrator class
  // and quadrature rule. Only a non-identity map is refused for Harmonic.
  const bool mapped = map != nullptr;
  MFEM_VERIFY(!(mapped && !map->IsIdentity()) ||
                  penalty == GaugePenalty::Deviatoric,
              "SetGaugedFluid: the Harmonic penalty is gauge data, "
              "shared rather than mapped.");
  gauge_marker_ = fluid_marker;
  gauge_eps_coef_ = std::make_unique<ConstantCoefficient>(epsilon);
  gauge_mu_eps_ =
      std::make_unique<ProductCoefficient>(*gauge_eps_coef_, mu_gauge);
  const int dim = fes_->GetMesh()->Dimension();
  gauge_integrators_ = detail::MakeBilinearForm(fes_);
  BilinearFormIntegrator* integ;
  if (mapped && penalty == GaugePenalty::Deviatoric) {
    // The covariant form of the Deviatoric branch: the isotropic tensor
    // of lambda = -2 eps mu_g / d, mu = eps mu_g, pulled back through the
    // map by ElasticTensorIntegrator, so the penalty of a relabelled
    // problem is the exact pull-back of the unmapped one.
    gauge_lambda_eps_ =
        std::make_unique<ProductCoefficient>(-2.0 / dim, *gauge_mu_eps_);
    gauge_Cdev_ = std::make_unique<IsotropicElasticTensorCoefficient>(
        dim, *gauge_lambda_eps_, *gauge_mu_eps_);
    integ = new ElasticTensorIntegrator(*gauge_Cdev_, *map);
  } else if (penalty == GaugePenalty::Deviatoric) {
    integ = new ElasticityIntegrator(*gauge_mu_eps_, -2.0 / dim, 1.0);
  } else {
    integ = new VectorDiffusionIntegrator(*gauge_mu_eps_);
  }
  gauge_integrators_->AddDomainIntegrator(integ, gauge_marker_);
  gauge_integ_ = integ;
  gauge_refinements_ = refinements;
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
  Q_.Ptr()->Mult(u_true, r);
}

void LinearQuasiStaticProblemBase::ClearGaugedFluid() {
  gauge_integrators_.reset();
  gauge_integ_ = nullptr;
  gauge_mu_eps_.reset();
  gauge_eps_coef_.reset();
  q_form_.reset();
  a_solve_form_.reset();
  Q_.Clear();
  A_solve_.Clear();
  gauge_residuals_.clear();
  operator_dirty_ = true;
}

void LinearQuasiStaticProblemBase::SetGaugeEpsilon(real_t epsilon) {
  MFEM_VERIFY(gauge_eps_coef_, "SetGaugeEpsilon: no gauged fluid is set.");
  MFEM_VERIFY(epsilon > 0.0, "SetGaugeEpsilon: epsilon must be positive.");
  gauge_eps_coef_->constant = epsilon;
  operator_dirty_ = true;
}

real_t LinearQuasiStaticProblemBase::GaugeEpsilon() const {
  return gauge_eps_coef_ ? gauge_eps_coef_->constant : 0.0;
}

const OperatorHandle& LinearQuasiStaticProblemBase::RegularizedMatrix() {
  EnsureOperator();
  return HasGaugedFluid() ? A_solve_ : A_;
}

bool LinearQuasiStaticProblemBase::GaugeRefine(Vector& X) {
  gauge_residuals_.clear();
  Vector r(X.Size()), d(X.Size()), prev;
  bool ok = true;
  for (int k = 0; k < gauge_refinements_; ++k) {
    // After an exact regularised solve the physical residual is
    // f - A U = eps Q delta, with delta the last increment (the first
    // "increment" being the solution itself). Each increment is solved
    // cold from zero.
    Q_.Ptr()->Mult(k == 0 ? X : prev, r);
    if (ess_tdof_list_.Size() > 0) {
      r.SetSubVector(ess_tdof_list_, 0.0);
    }
    gauge_residuals_.push_back(std::sqrt(Dot(r, r)));
    d = 0.0;
    ok = SolveLinearSystem(r, d) && ok;
    X += d;
    prev = d;
  }
  WarnGaugeContraction();
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
