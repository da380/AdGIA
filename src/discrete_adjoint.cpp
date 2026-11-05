#include "AdGIA/discrete_adjoint.hpp"

#include <cmath>

#include "AdGIA/detail/fem_factory.hpp"
#include "AdGIA/mixed_problem.hpp"
#include "AdGIA/referential_problem.hpp"

namespace AdGIA {

using namespace mfem;

namespace {
int InternalOrder(const ViscoelasticOperator& op) {
  return const_cast<ViscoelasticOperator&>(op).InternalScalarSpace().FEColl()
      ->GetOrder();
}

// The potential and border components of a solved twin: the duals of
// those rows (the bordered system is symmetric), read as
// TangentTrajectory::Record reads the tangent's.
void ReadRowDuals(LinearQuasiStaticProblem& twin,
                  std::unique_ptr<GridFunction>& phi, real_t& uniform,
                  Vector& omega) {
  phi.reset();
  uniform = 0.0;
  omega.SetSize(0);
  if (auto* g =
          dynamic_cast<LinearQuasiStaticMixedSelfGravitatingProblem*>(&twin)) {
    const GridFunction& p = g->Potential();
    phi =
        detail::MakeGridFunction(const_cast<FiniteElementSpace*>(p.FESpace()));
    *phi = p;
    uniform = g->UniformPotentialTerm();
    omega = g->AngularVelocity();
  }
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

void DiscreteAdjointOperator::EngageStep(real_t dt) {
  if (dt > 0.0) {
    SetEffectiveModulus(dt, Scheme::ExponentialTrapezoid);
  } else {
    UseUnrelaxedOperator();
  }
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
  phi_dual_.clear();
  phi_dual_.resize(N + 1);
  uniform_dual_.assign(N + 1, 0.0);
  omega_dual_.assign(N + 1, Vector());
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
    ReadRowDuals(problem_, phi_dual_[n], uniform_dual_[n], omega_dual_[n]);
  }
  UseUnrelaxedOperator();
  return true;
}

// --- DiscreteAdjointSolver ------------------------------------------------

DiscreteAdjointSolver::DiscreteAdjointSolver(
    LinearQuasiStaticProblem& twin, const GridFunction& forward_displacement)
    : twin_(&twin), u_forward_(&forward_displacement) {}

DiscreteAdjointSolver::DiscreteAdjointSolver(
    LinearQuasiStaticProblem& twin, LinearQuasiStaticProblem& source_twin,
    ForwardTrajectory& forward)
    : forward_(&forward) {
  op_ = std::make_unique<DiscreteAdjointOperator>(twin, forward.Op());
  src_ = std::make_unique<TangentViscoelasticOperator>(source_twin,
                                                       forward.Op());
}

DiscreteAdjointOperator& DiscreteAdjointSolver::Operator() {
  MFEM_VERIFY(op_, "DiscreteAdjointSolver: the static configuration has no "
                   "transposed sweep.");
  return *op_;
}

bool DiscreteAdjointSolver::Run() {
  MFEM_VERIFY(force_, "DiscreteAdjointSolver: no force set.");
  // The load duals belong to the previous duals.
  load_duals_.clear();
  load_dual_sum_.SetSize(0);
  return op_ ? op_->Run(*forward_, *force_) : RunStatic();
}

bool DiscreteAdjointSolver::RunStatic() {
  // One solve of the twin per point observation, its duals on the slots
  // as AdjointSolver::ApplyObservation applies them: the twin's operator
  // is symmetric, so the solution is the transpose of the static tangent
  // solve applied to the observation.
  MFEM_VERIFY(!force_->HasContinuous() && !force_->HasContinuousPotential(),
              "DiscreteAdjointSolver: a static adjoint takes point terms "
              "only.");
  auto* phi_slot = dynamic_cast<PotentialForceSlot*>(twin_);
  auto* border_slot = dynamic_cast<BorderForceSlot*>(twin_);
  lambda_.clear();
  phi_dual_.clear();
  uniform_dual_.clear();
  omega_dual_.clear();
  for (int i = 0; i < force_->NumObservations(); i++) {
    twin_->AssembleForce(force_->ObservationTime(i));
    const Vector& du = force_->ObservationDualU(i);
    if (du.Size() > 0) {
      twin_->AddForce(du);
    }
    if (force_->ObservationOnPotential(i)) {
      MFEM_VERIFY(phi_slot,
                  "DiscreteAdjointSolver: the force carries potential-row "
                  "terms but the problem exposes no PotentialForceSlot.");
      phi_slot->AddPotentialForce(force_->ObservationDualPhi(i));
    }
    if (force_->ObservationOnBorder(i)) {
      MFEM_VERIFY(border_slot,
                  "DiscreteAdjointSolver: the force carries border-row "
                  "terms but the problem exposes no BorderForceSlot.");
      const Vector& om = force_->ObservationOmega(i);
      border_slot->AddBorderForce(force_->ObservationUniform(i),
                                  om.Size() > 0 ? &om : nullptr);
    }
    if (!twin_->Solve()) {
      return false;
    }
    lambda_.push_back(detail::MakeGridFunction(&twin_->DisplacementSpace()));
    *lambda_.back() = twin_->Displacement();
    phi_dual_.emplace_back();
    uniform_dual_.push_back(0.0);
    omega_dual_.emplace_back();
    ReadRowDuals(*twin_, phi_dual_.back(), uniform_dual_.back(),
                 omega_dual_.back());
  }
  return true;
}

real_t DiscreteAdjointSolver::Dot(const Vector& dual, const GridFunction& u,
                                  FiniteElementSpace& fes) const {
  // The dual is the rank's partial sums (an unassembled L-vector, as
  // AddForce takes it): restrict it to true dofs through the transpose
  // of the prolongation, which assembles the shared dofs once, and pair
  // it with u's true dofs (TangentLinearSolver::Dot).
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

real_t DiscreteAdjointSolver::SumOverRanks(real_t local,
                                           FiniteElementSpace& fes) const {
  // The internal space is discontinuous: its dofs are element-local and
  // never shared, so the ranks' local pairings add up to the global one.
#ifdef MFEM_USE_MPI
  if (auto* pfes = dynamic_cast<ParFiniteElementSpace*>(&fes)) {
    MPI_Allreduce(MPI_IN_PLACE, &local, 1, MPITypeMap<real_t>::mpi_type,
                  MPI_SUM, pfes->GetComm());
  }
#else
  (void)fes;
#endif
  return local;
}

real_t DiscreteAdjointSolver::Directional(const TangentDirection& dir) {
  return op_ ? DirectionalEvolution(dir) : DirectionalStatic(dir);
}

real_t DiscreteAdjointSolver::DirectionalStatic(const TangentDirection& dir) {
  MFEM_VERIFY(force_ && static_cast<int>(lambda_.size()) ==
                            force_->NumObservations(),
              "DiscreteAdjointSolver: Run() first.");
  // The source of the static tangent solve: the Born force -dK u of the
  // direction on the forward displacement (TangentLinearSolver::Run).
  FiniteElementSpace& fes = twin_->DisplacementSpace();
  ModuliBornForce born(fes, dir, twin_->Rheology().NumBranches());
  if (born.Empty()) {
    return 0.0;
  }
  Vector f(fes.GetVSize());
  f = 0.0;
  born.AddTo(*u_forward_, f);
  real_t dJ = 0.0;
  for (const auto& lambda : lambda_) {
    dJ += Dot(f, *lambda, fes);
  }
  return dJ;
}

real_t DiscreteAdjointSolver::DirectionalEvolution(
    const TangentDirection& dir) {
  MFEM_VERIFY(op_->Size() > 0, "DiscreteAdjointSolver: Run() first.");
  src_->SetDirection(dir);
  ForwardTrajectoryData data(*forward_);
  const int N = forward_->Size() - 1;
  const int K = op_->NumBranches();
  FiniteElementSpace& fes_u = op_->Problem().DisplacementSpace();
  Vector f_u, cb;
  std::vector<Vector> zeta;
  // The displacement-row sources are L-vectors of partial sums against
  // the consistent lambda (Dot, global); the branch and state sources
  // live on the internal space's element-local dofs, so their pairings
  // are local sums, reduced once at the end.
  real_t dJ = 0.0, internal = 0.0;
  // The unrelaxed start.
  src_->ComputeSources(data, 0, false, f_u, zeta);
  dJ += Dot(f_u, op_->Lambda(0), fes_u);
  op_->Coupling(op_->Lambda(0), cb);
  for (int k = 0; k < K; k++) {
    if (zeta[k].Size() > 0) internal += zeta[k] * cb;
  }
  for (int n = 0; n < N; n++) {
    src_->PrepareStep(data, n, n + 1);
    src_->ComputeSources(data, n + 1, true, f_u, zeta);
    dJ += Dot(f_u, op_->Lambda(n + 1), fes_u);
    op_->Coupling(op_->Lambda(n + 1), cb);
    for (int k = 0; k < K; k++) {
      if (zeta[k].Size() > 0) internal += zeta[k] * cb;
      const Vector& s = src_->StateSource(k);
      if (s.Size() > 0) {
        Vector mu_k = op_->Branch(op_->Mu(n + 1), k);
        internal += s * mu_k;
      }
    }
  }
  return dJ + SumOverRanks(internal, op_->InternalScalarSpace());
}

// --- the data surface load ---------------------------------------------------

int DiscreteAdjointSolver::NumSolves() const {
  return op_ ? op_->Size() : static_cast<int>(lambda_.size());
}

real_t DiscreteAdjointSolver::SolveTime(int n) const {
  // Static: the load of the solved state, assembled at t = 0 as
  // TangentLinearSolver's static solve assembles it.
  return op_ ? forward_->Time(n) : 0.0;
}

void DiscreteAdjointSolver::DualState(int n, const GridFunction*& lambda,
                                      const GridFunction*& phi, real_t& uniform,
                                      const Vector*& omega) const {
  if (op_) {
    lambda = &op_->Lambda(n);
    phi = op_->PotentialDual(n);
    uniform = op_->UniformDual(n);
    omega = &op_->OmegaDual(n);
  } else {
    lambda = lambda_[n].get();
    phi = phi_dual_[n].get();
    uniform = uniform_dual_[n];
    omega = &omega_dual_[n];
  }
}

real_t DiscreteAdjointSolver::PairAssembledLoad(
    LinearQuasiStaticProblem& load_twin, const GridFunction& lambda,
    const GridFunction* phi, real_t uniform, const Vector& omega) const {
  MFEM_VERIFY(
      !dynamic_cast<LinearQuasiStaticReferentialSelfGravitatingProblem*>(
          &load_twin),
      "DiscreteAdjointSolver::DirectionalLoad: the referential "
      "class's potential rows are not read.");
  auto* base = dynamic_cast<LinearQuasiStaticProblemBase*>(&load_twin);
  MFEM_VERIFY(base,
              "DiscreteAdjointSolver::DirectionalLoad: the load twin must "
              "expose its assembled load (LinearQuasiStaticProblemBase).");
  // The displacement rows: the external load form, the rank's partial
  // sums (the Solve() right-hand side before the AddForce increments).
  real_t dJ = Dot(base->ExternalLoad(), lambda, load_twin.DisplacementSpace());
  auto* mixed =
      dynamic_cast<LinearQuasiStaticMixedSelfGravitatingProblem*>(&load_twin);
  if (!mixed) {
    return dJ;
  }
  MFEM_VERIFY(!mixed->HasTidalPotential(),
              "DiscreteAdjointSolver::DirectionalLoad: a tidal potential on "
              "the load twin is not read.");
  MFEM_VERIFY(phi,
              "DiscreteAdjointSolver::DirectionalLoad: the dual twin "
              "carries no potential (the mixed class expected).");
  // The potential rows on true dofs (global), the border rows' data as
  // replicated scalars.
  const Vector& lphi = mixed->PotentialLoad();
  Vector pt(lphi.Size());
  MFEM_VERIFY(phi->FESpace()->GetTrueVSize() == lphi.Size(),
              "DiscreteAdjointSolver::DirectionalLoad: the potential layouts "
              "differ.");
  phi->GetTrueDofs(pt);
  real_t dphi = lphi * pt;
#ifdef MFEM_USE_MPI
  if (auto* pf = dynamic_cast<const ParFiniteElementSpace*>(phi->FESpace())) {
    dphi = InnerProduct(pf->GetComm(), lphi, pt);
  }
#endif
  dJ += dphi + mixed->UniformRowLoad() * uniform;
  const Vector& r = mixed->AngularVelocityRowLoad();
  if (r.Size() > 0) {
    MFEM_VERIFY(omega.Size() == r.Size(),
                "DiscreteAdjointSolver::DirectionalLoad: the load twin "
                "rotates but the dual twin does not.");
    dJ += r * omega;
  }
  return dJ;
}

real_t DiscreteAdjointSolver::DirectionalLoad(
    LinearQuasiStaticProblem& load_twin) {
  const int ns = NumSolves();
  MFEM_VERIFY(ns > 0, "DiscreteAdjointSolver: Run() first.");
  // The load is the right-hand side of each solve: the unrelaxed start
  // (node 0) and the effective solve (S1) of every step, each paired
  // with its own dual, as DirectionalEvolution pairs the Born forces.
  real_t dJ = 0.0;
  for (int n = 0; n < ns; n++) {
    const GridFunction *lambda, *phi;
    const Vector* omega;
    real_t uniform;
    DualState(n, lambda, phi, uniform, omega);
    load_twin.AssembleForce(SolveTime(n));
    dJ += PairAssembledLoad(load_twin, *lambda, phi, uniform, *omega);
  }
  return dJ;
}

namespace {

// int psi_j c dS with the rule of the displacement rows' surface load
// (VectorBoundaryLFIntegrator's default, order 2 p on the displacement
// space's boundary element of the same index), so that the dual of the
// u-row load is its transpose to rounding.
class DisplacementRuleBoundaryLFIntegrator : public LinearFormIntegrator {
 public:
  DisplacementRuleBoundaryLFIntegrator(Coefficient& c,
                                       const FiniteElementSpace& ufes)
      : c_(c), ufes_(ufes) {}
  using LinearFormIntegrator::AssembleRHSElementVect;
  void AssembleRHSElementVect(const FiniteElement& el,
                              ElementTransformation& Tr,
                              Vector& elvect) override {
    const int dof = el.GetDof();
    shape_.SetSize(dof);
    elvect.SetSize(dof);
    elvect = 0.0;
    const int order = 2 * ufes_.GetBE(Tr.ElementNo)->GetOrder();
    const IntegrationRule& ir = IntRules.Get(el.GetGeomType(), order);
    for (int i = 0; i < ir.GetNPoints(); i++) {
      const IntegrationPoint& ip = ir.IntPoint(i);
      Tr.SetIntPoint(&ip);
      const real_t val = Tr.Weight() * c_.Eval(Tr, ip);
      el.CalcShape(ip, shape_);
      elvect.Add(ip.weight * val, shape_);
    }
  }

 private:
  Coefficient& c_;
  const FiniteElementSpace& ufes_;
  Vector shape_;
};

LinearQuasiStaticMixedSelfGravitatingProblem& MixedTwin(
    LinearQuasiStaticProblem& twin) {
  auto* mixed =
      dynamic_cast<LinearQuasiStaticMixedSelfGravitatingProblem*>(&twin);
  MFEM_VERIFY(mixed,
              "DiscreteAdjointSolver: the load duals are built on "
              "the mixed gravitating class.");
  return *mixed;
}

}  // namespace

FiniteElementSpace& DiscreteAdjointSolver::LoadDualSpace() {
  return MixedTwin(op_ ? op_->Problem() : *twin_).PotentialSpaceOnBody();
}

void DiscreteAdjointSolver::AssembleLoadDuals(
    const Array<int>& surface_marker) {
  const int ns = NumSolves();
  MFEM_VERIFY(ns > 0, "DiscreteAdjointSolver: Run() first.");
  auto& mixed = MixedTwin(op_ ? op_->Problem() : *twin_);
  FiniteElementSpace& sfes = mixed.PotentialSpaceOnBody();
  FiniteElementSpace& ufes = mixed.DisplacementSpace();
  FiniteElementSpace& pfes = mixed.PotentialSpace();
  MFEM_VERIFY(sfes.GetMesh() == ufes.GetMesh(),
              "DiscreteAdjointSolver::AssembleLoadDuals: the trace space and "
              "the displacement space live on different meshes.");
  MFEM_VERIFY(surface_marker.Size() == sfes.GetMesh()->bdr_attributes.Max(),
              "DiscreteAdjointSolver::AssembleLoadDuals: the marker must be "
              "sized to the body mesh's bdr_attributes.Max().");
  Array<int> marker(surface_marker);
  // The interpolants of the unit centrifugal potentials on the potential
  // space, as AssembleForce reads the omega rows off the potential load.
  const int nw = mixed.AngularVelocity().Size();
  std::vector<std::unique_ptr<GridFunction>> psi(nw);
  for (int k = 0; k < nw; k++) {
    psi[k] = detail::MakeGridFunction(&pfes);
    psi[k]->ProjectCoefficient(mixed.UnitCentrifugalPotential(k));
  }
  auto parent = detail::MakeGridFunction(&pfes);
  auto cs = detail::MakeGridFunction(&sfes);
  load_duals_.assign(ns, Vector());
  load_dual_sum_.SetSize(sfes.GetVSize());
  load_dual_sum_ = 0.0;
  for (int n = 0; n < ns; n++) {
    const GridFunction *lambda, *phi;
    const Vector* omega;
    real_t uniform;
    DualState(n, lambda, phi, uniform, omega);
    MFEM_VERIFY(phi,
                "DiscreteAdjointSolver::AssembleLoadDuals: no potential "
                "dual recorded.");
    MFEM_VERIFY(omega->Size() == nw,
                "DiscreteAdjointSolver::AssembleLoadDuals: the omega duals "
                "do not match the rotation border.");
    // The potential, mass and omega rows' transpose: the nodal field
    // -(phi_dual + Pi psi(omega_dual) - Phi_g_dual) on the body, against
    // the trace basis with the surface-load forms' own rule.
    *parent = *phi;
    for (int k = 0; k < nw; k++) parent->Add((*omega)[k], *psi[k]);
    mixed.Injection().MultTranspose(*parent, *cs);
    *cs -= uniform;
    *cs *= -1.0;
    GridFunctionCoefficient cs_c(cs.get());
    // The displacement rows' transpose: -grad Phi0 . lambda with their
    // rule.
    VectorGridFunctionCoefficient lam_c(lambda);
    InnerProductCoefficient g_lam(mixed.BackgroundGravity(), lam_c);
    ProductCoefficient cu(-1.0, g_lam);
    auto lf = detail::MakeLinearForm(&sfes);
    lf->AddBoundaryIntegrator(new BoundaryLFIntegrator(cs_c), marker);
    lf->AddBoundaryIntegrator(
        new DisplacementRuleBoundaryLFIntegrator(cu, ufes), marker);
    lf->Assemble();
    load_duals_[n] = *lf;
    load_dual_sum_ += *lf;
  }
}

real_t DiscreteAdjointSolver::PairLoad(const Vector& dual,
                                       const GridFunction& delta) const {
  MFEM_VERIFY(dual.Size() == delta.Size(),
              "DiscreteAdjointSolver::PairLoad: the load must live on "
              "LoadDualSpace().");
  return Dot(dual, delta, *const_cast<FiniteElementSpace*>(delta.FESpace()));
}

// --- the kernels (stage 2) --------------------------------------------------

int DiscreteAdjointSolver::AddKernel(Parameter parameter, int branch) {
  kernels_.push_back(Kernel{parameter, branch, Vector()});
  return static_cast<int>(kernels_.size()) - 1;
}

namespace {

// A kernel density times an optional weight coefficient, for the
// integration against the L2 basis (int phi_q weight(x) density(x) dV,
// then scaled per node) with the forward stiffness's own rule.
class ScaledDensity : public Coefficient {
 public:
  ScaledDensity(Coefficient& density, Coefficient* weight)
      : density_(density), weight_(weight) {}
  real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    real_t v = density_.Eval(T, ip);
    if (weight_) v *= weight_->Eval(T, ip);
    return v;
  }

 private:
  Coefficient& density_;
  Coefficient* weight_;
};

}  // namespace

void detail::IntegrateAgainstInternalBasis(Coefficient& density,
                                           FiniteElementSpace& ufes,
                                           FiniteElementSpace& sfes,
                                           Vector& out) {
  // The forward stiffness's own rule (ElasticityIntegrator: order
  // 2 OrderGrad of the displacement element), so that the dual paired
  // with a nodal direction is the derivative of the ASSEMBLED stiffness
  // along it, to rounding; a uniform mesh is assumed for the one rule.
  auto lf = detail::MakeLinearForm(&sfes);
  ElementTransformation& T0 = *ufes.GetElementTransformation(0);
  const int order = 2 * T0.OrderGrad(ufes.GetFE(0));
  auto* integ = new DomainLFIntegrator(density);
  integ->SetIntRule(
      &IntRules.Get(sfes.GetMesh()->GetElementGeometry(0), order));
  lf->AddDomainIntegrator(integ);
  lf->Assemble();
  out.SetSize(lf->Size());
  out = *lf;
}

void DiscreteAdjointSolver::AddElastic(Kernel& kernel, const GridFunction& u,
                                       const GridFunction& lambda,
                                       Coefficient* weight,
                                       const Vector* nodal_scale) {
  // The elastic density of the parameter (IsotropicKernelDensity's
  // Kappa: -div u div lambda; MuLongTerm: -2 dev eps(u) : dev eps(lambda)),
  // times the weight coefficient, against the internal basis.
  const bool is_kappa = kernel.parameter == Parameter::Kappa;
  IsotropicKernelDensity density(is_kappa ? Parameter::Kappa
                                          : Parameter::MuLongTerm);
  density.SetPair(u, lambda);
  ScaledDensity scaled(density, weight);
  Vector lf;
  detail::IntegrateAgainstInternalBasis(
      scaled, *const_cast<FiniteElementSpace*>(u.FESpace()),
      op_->InternalScalarSpace(), lf);
  if (nodal_scale) {
    for (int q = 0; q < lf.Size(); q++) {
      kernel.dual[q] += lf[q] * (*nodal_scale)[q];
    }
  } else {
    kernel.dual += lf;
  }
}

void DiscreteAdjointSolver::AssembleKernels() {
  MFEM_VERIFY(op_->Size() > 0, "DiscreteAdjointSolver: Run() first.");
  const int N = forward_->Size() - 1;
  const int K = op_->NumBranches();
  const int nd = op_->NumNodes(), nc = op_->NumComponents();
  const Rheology& rh = op_->Problem().Rheology();
  for (auto& kernel : kernels_) {
    kernel.dual.SetSize(nd);
    kernel.dual = 0.0;
  }
  Vector cb, dn, dn1, scale(nd);
  // Node 0: the unrelaxed solve (beta = 1, no weight derivative) and the
  // branch term <B lambda^0, 2 delta mu_k m_k^0>.
  op_->EngageStep(0.0);
  op_->Coupling(op_->Lambda(0), cb);
  for (auto& kernel : kernels_) {
    const int k = kernel.branch;
    switch (kernel.parameter) {
      case Parameter::Kappa:
      case Parameter::MuLongTerm:
        AddElastic(kernel, forward_->Displacement(0), op_->Lambda(0), nullptr,
                   nullptr);
        break;
      case Parameter::Mu: {
        AddElastic(kernel, forward_->Displacement(0), op_->Lambda(0), nullptr,
                   nullptr);
        Vector m0 = op_->Branch(forward_->State(0), k);
        for (int q = 0; q < nd; q++) {
          real_t v = 0.0;
          for (int c = 0; c < nc; c++) v += cb[c * nd + q] * 2.0 * m0[c * nd + q];
          kernel.dual[q] += v;
        }
        break;
      }
      case Parameter::Eta:
        break;
    }
  }
  // The steps n -> n+1: the effective solve at n+1 (its elastic
  // densities with the weights beta_k and the weights' derivative), the
  // branch sources paired with B lambda^{n+1}, the state source with
  // mu^{n+1}.
  for (int n = 0; n < N; n++) {
    const real_t dt = forward_->Time(n + 1) - forward_->Time(n);
    op_->EngageStep(dt);
    const GridFunction& u1 = forward_->Displacement(n + 1);
    const GridFunction& lam = op_->Lambda(n + 1);
    forward_->Strain(n, dn);
    forward_->Strain(n + 1, dn1);
    op_->Coupling(lam, cb);
    for (auto& kernel : kernels_) {
      const int k = kernel.branch;
      if (kernel.parameter == Parameter::Kappa ||
          kernel.parameter == Parameter::MuLongTerm) {
        AddElastic(kernel, u1, lam, nullptr, nullptr);
        continue;
      }
      const Vector& two_mu = op_->NodalBranchModulus(k);
      const Vector& itau = op_->NodalInverseRelaxationTime(k);
      Vector mn = op_->Branch(forward_->State(n), k);
      Vector mu1 = op_->Branch(op_->Mu(n + 1), k);
      const bool is_mu = kernel.parameter == Parameter::Mu;
      // d(1/tau_q)/d(param_q): itau/mu for mu (fixed eta), -itau/eta for eta,
      // with eta = mu / itau.
      for (int q = 0; q < nd; q++) {
        const real_t mu = 0.5 * two_mu[q];
        scale[q] = is_mu ? itau[q] / mu : -itau[q] * itau[q] / mu;
      }
      if (is_mu) {
        // The branch modulus in the effective stiffness: beta_k times the
        // elastic density.
        AddElastic(kernel, u1, lam, &op_->BetaCoefficient(k), nullptr);
      }
      // The weights' derivative in the stiffness: delta beta_k =
      // -b' dt delta(1/tau): -2 mu_k dev:dev density, nodally scaled.
      Vector dbeta_scale(nd);
      for (int q = 0; q < nd; q++) {
        const real_t h = dt * itau[q];
        real_t de, da, db;
        detail::TrapezoidWeightDerivatives(h, de, da, db);
        dbeta_scale[q] = -db * dt * scale[q];
      }
      {
        Kernel tmp{Parameter::MuLongTerm, k, Vector(nd)};
        tmp.dual = 0.0;
        AddElastic(tmp, u1, lam, &rh.BranchShearModulus(k), &dbeta_scale);
        kernel.dual += tmp.dual;
      }
      // The nodal terms: the known-part branch source paired with
      // B lambda, the state source with mu.
      for (int q = 0; q < nd; q++) {
        const real_t h = dt * itau[q];
        real_t e, alpha, beta, de, da, db;
        detail::ExponentialTrapezoidWeights(h, e, alpha, beta);
        detail::TrapezoidWeightDerivatives(h, de, da, db);
        real_t v = 0.0;
        for (int c = 0; c < nc; c++) {
          const int i = c * nd + q;
          const real_t rel = de * mn[i] + da * dn[i];
          const real_t st = rel + db * dn1[i];
          // zeta: 2 delta mu (e m + a d) [mu only] + 2 mu dt dh' rel
          if (is_mu) v += cb[i] * 2.0 * (e * mn[i] + alpha * dn[i]);
          v += cb[i] * two_mu[q] * dt * scale[q] * rel;
          // s: dt dh' (de m + da d + db d1)
          v += mu1[i] * dt * scale[q] * st;
        }
        kernel.dual[q] += v;
      }
    }
  }
  op_->EngageStep(0.0);
}

real_t DiscreteAdjointSolver::Pair(int i, const GridFunction& delta) const {
  const Vector& d = kernels_[i].dual;
  MFEM_VERIFY(d.Size() == delta.Size(),
              "DiscreteAdjointSolver::Pair: the direction must live on the "
              "internal scalar space.");
  // The internal space is discontinuous: a local pairing, reduced.
  return SumOverRanks(d * delta,
                      *const_cast<FiniteElementSpace*>(delta.FESpace()));
}

}  // namespace AdGIA
