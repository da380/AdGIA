#include "AdGIA/second_order.hpp"

#include <cmath>

#include "AdGIA/detail/fem_factory.hpp"

namespace AdGIA {

using namespace mfem;

// --- AdjointWindow ---------------------------------------------------------

int AdjointWindow::Push(real_t s, const Vector& m, const GridFunction& u) {
  Node n;
  n.s = s;
  n.m = m;
  n.u = detail::MakeGridFunction(const_cast<FiniteElementSpace*>(u.FESpace()));
  *n.u = u;
  nodes_.push_back(std::move(n));
  return static_cast<int>(nodes_.size()) - 1;
}

// --- SecondOrderAdjointSolver ----------------------------------------------

SecondOrderAdjointSolver::SecondOrderAdjointSolver(
    LinearQuasiStaticProblem& adjoint_problem,
    LinearQuasiStaticProblem& derivative_twin,
    const GridFunction& forward_displacement,
    const GridFunction& tangent_displacement, const TangentDirection& v)
    : adjoint_(&adjoint_problem),
      u_forward_(&forward_displacement),
      du_forward_(&tangent_displacement),
      twin_(derivative_twin),
      v_(v) {}

SecondOrderAdjointSolver::SecondOrderAdjointSolver(
    ViscoelasticOperator& adjoint_op, LinearQuasiStaticProblem& derivative_twin,
    ForwardTrajectory& forward, const TangentTrajectory& tangent,
    const TangentDirection& v)
    : op_(&adjoint_op),
      forward_(&forward),
      tangent_(&tangent),
      twin_(derivative_twin),
      v_(v) {
  dop_ = std::make_unique<TangentViscoelasticOperator>(derivative_twin,
                                                       adjoint_op);
  dop_->SetDirection(v_);
  window_ = std::make_unique<AdjointWindow>(adjoint_op);
  BuildEtaScale();
}

void SecondOrderAdjointSolver::BuildEtaScale() {
  // delta(1/tau_k) / (1/tau_k) as a nodal field on the internal scalar
  // space: the Eta density's explicit factor. Whole-mesh branches only
  // (the kernels' own restriction); one field serves every branch the
  // kernels read through their branch index — the accumulators are
  // per branch, so the scale is per branch too.
  // Built per branch lazily in PairAll; here the buffers.
  eta_scale_gf_.reset();
  eta_scale_.reset();
}

bool SecondOrderAdjointSolver::Run() {
  MFEM_VERIFY(force_, "SecondOrderAdjointSolver: no force set.");
  return op_ ? RunEvolution() : RunStatic();
}

void SecondOrderAdjointSolver::ResolveSlots(LinearQuasiStaticProblem& problem) {
  phi_slot_ = nullptr;
  border_slot_ = nullptr;
  bool needs_phi = force_->HasContinuousPotential();
  bool needs_border = false;
  for (int i = 0; i < force_->NumObservations(); i++) {
    needs_phi = needs_phi || force_->ObservationOnPotential(i);
    needs_border = needs_border || force_->ObservationOnBorder(i);
  }
  if (needs_phi) {
    phi_slot_ = dynamic_cast<PotentialForceSlot*>(&problem);
    MFEM_VERIFY(phi_slot_, "SecondOrderAdjointSolver: the force carries "
                           "potential-row terms but the adjoint problem "
                           "exposes no PotentialForceSlot.");
  }
  if (needs_border) {
    border_slot_ = dynamic_cast<BorderForceSlot*>(&problem);
    MFEM_VERIFY(border_slot_, "SecondOrderAdjointSolver: the force carries "
                              "border-row terms but the adjoint problem "
                              "exposes no BorderForceSlot.");
  }
}

void SecondOrderAdjointSolver::ApplyObservation(int obs) {
  auto& prob = adjoint_ ? *adjoint_ : op_->Problem();
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

void SecondOrderAdjointSolver::PairAll(real_t w, const AdjointPair& first,
                                       const AdjointPair& d_forward,
                                       const AdjointPair& d_adjoint) {
  for (auto* k : first_) {
    k->AddPair(w, first);
  }
  for (auto* k : kernels_) {
    k->AddPair(w, d_forward);
    k->AddPair(w, d_adjoint);
  }
  // The Eta density's explicit factor: -(2/tau)(d - m):m+ differentiated
  // in 1/tau gives (delta(1/tau)/(1/tau)) times the density, on the
  // first-order pair. Fed to the Eta accumulators alone, per branch.
  if (!dop_ || first.impulse) {
    return;
  }
  for (auto* k : kernels_) {
    auto* ik = dynamic_cast<IsotropicKernel*>(k);
    if (!ik || ik->Param() != IsotropicKernel::Parameter::Eta) {
      continue;
    }
    const int b = ik->Branch();
    if (!dop_->HasDeltaInverseRelaxationTime(b)) {
      continue;
    }
    auto& sfes = op_->InternalScalarSpace();
    if (!eta_scale_gf_) {
      eta_scale_gf_ = detail::MakeGridFunction(&sfes);
      eta_scale_ =
          std::make_unique<GridFunctionCoefficient>(eta_scale_gf_.get());
    }
    const Vector& ditau = dop_->DeltaInverseRelaxationTime(b);
    const Vector& itau = op_->InverseRelaxationTimes(b);
    const Array<int>& nodes = op_->BranchNodes(b);
    *eta_scale_gf_ = 0.0;
    for (int q = 0; q < nodes.Size(); q++) {
      (*eta_scale_gf_)[nodes[q]] = ditau[q] / itau[q];
    }
    AdjointPair scaled = first;
    scaled.scale = eta_scale_.get();
    ik->AddPair(w, scaled);
  }
}

bool SecondOrderAdjointSolver::RunStatic() {
  MFEM_VERIFY(!force_->HasContinuous() && !force_->HasContinuousPotential(),
              "SecondOrderAdjointSolver: a static sweep takes point terms "
              "only.");
  ResolveSlots(*adjoint_);
  ModuliBornForce born(twin_.DisplacementSpace(), v_,
                       twin_.Rheology().NumBranches());
  for (int i = 0; i < force_->NumObservations(); i++) {
    const real_t t = force_->ObservationTime(i);
    adjoint_->AssembleForce(t);
    ApplyObservation(i);
    if (!adjoint_->Solve()) {
      return false;
    }
    // The derivative of the adjoint solve along v: the Born force on u+.
    twin_.AssembleForce(t);
    if (!born.Empty()) {
      fbuf_.SetSize(twin_.DisplacementSpace().GetVSize());
      fbuf_ = 0.0;
      born.AddTo(adjoint_->Displacement(), fbuf_);
      twin_.AddForce(fbuf_);
    }
    if (!twin_.Solve()) {
      return false;
    }
    PairAll(1.0,
            AdjointPair{*u_forward_, nullptr, {}, adjoint_->Displacement(), {},
                        t, true, adjoint_},
            AdjointPair{*du_forward_, nullptr, {}, adjoint_->Displacement(), {},
                        t, true, adjoint_},
            AdjointPair{*u_forward_, nullptr, {}, twin_.Displacement(), {}, t,
                        true, &twin_});
  }
  return true;
}

bool SecondOrderAdjointSolver::RunEvolution() {
  auto& op = *op_;
  auto& fwd = *forward_;
  auto& prob = op.Problem();
  const int N = fwd.Size() - 1;
  MFEM_VERIFY(N >= 1, "SecondOrderAdjointSolver: the trajectory holds fewer "
                      "than two states.");
  MFEM_VERIFY(tangent_->Size() == fwd.Size(),
              "SecondOrderAdjointSolver: the tangent trajectory must be "
              "recorded on the forward grid.");
  MFEM_VERIFY(op.IsLinear(), "SecondOrderAdjointSolver: linear rheologies.");
  for (int b = 0; b < op.NumBranches(); b++) {
    MFEM_VERIFY(op.NumBranchNodes(b) == op.InternalScalarSpace().GetVSize(),
                "SecondOrderAdjointSolver: whole-mesh branches only.");
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
      if (std::abs(fwd.Time(n) - t) < std::abs(fwd.Time(best) - t)) best = n;
    }
    MFEM_VERIFY(std::abs(fwd.Time(best) - t) <= tol,
                "SecondOrderAdjointSolver: observation times must lie on "
                "the forward grid.");
    node_obs[best].push_back(i);
  }

  // The first-order sweep's continuous terms through the adjoint
  // operator's hook (the derivative state has none: the duals carry no
  // model dependence).
  ResolveSlots(prob);
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
        Vector fphi(phi_slot_->PotentialForceSpace().GetVSize());
        fphi = 0.0;
        force_->EvalContinuousPotential(t1 - s, fphi);
        phi_slot_->AddPotentialForce(fphi);
      }
    });
  }

  // Pairing fields: wrappers on the internal spaces.
  auto& fop = fwd.Op();
  const int nb = op.NumBranches();
  auto d_f = detail::MakeGridFunction(&fop.InternalVariableSpace());
  auto dd_f = detail::MakeGridFunction(&fop.InternalVariableSpace());
  std::vector<std::unique_ptr<GridFunction>> m_f(nb), dm_f(nb), mdag_f(nb),
      wdag_f(nb);
  std::vector<const GridFunction*> m_ptr(nb), dm_ptr(nb), mdag_ptr(nb),
      wdag_ptr(nb);
  for (int b = 0; b < nb; b++) {
    m_f[b] = detail::MakeGridFunction(&fop.InternalVariableSpace());
    dm_f[b] = detail::MakeGridFunction(&fop.InternalVariableSpace());
    mdag_f[b] = detail::MakeGridFunction(&op.InternalVariableSpace());
    wdag_f[b] = detail::MakeGridFunction(&op.InternalVariableSpace());
    m_ptr[b] = m_f[b].get();
    dm_ptr[b] = dm_f[b].get();
    mdag_ptr[b] = mdag_f[b].get();
    wdag_ptr[b] = wdag_f[b].get();
  }
  auto udag_imp = detail::MakeGridFunction(&prob.DisplacementSpace());
  auto wdag_imp = detail::MakeGridFunction(&twin_.DisplacementSpace());
  Vector dbuf, mbuf;
  ModuliBornForce born(twin_.DisplacementSpace(), v_, nb);

  ExponentialTrapezoidSolver ode;
  ode.Init(op);
  Vector mdag(op.Height()), wdag(op.Height());
  mdag = 0.0;
  wdag = 0.0;
  real_t s = 0.0;
  bool active = has_continuous;
  bool ok = true;
  // The window holds the previous node's post-jump state (index 0) for
  // the derivative step; refreshed per node.
  window_->Clear();
  int prev_node = -1;

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
      // The first-order step, then the derivative step on the window
      // [previous post-jump node, this arrived node].
      real_t dt = s_k - s;
      ode.Step(mdag, s, dt);
      s = s_k;
      ok = op.SolveElastic(mdag, s);
      if (!ok) break;
      MFEM_VERIFY(prev_node == 0, "SecondOrderAdjointSolver: window state.");
      window_->Push(s, mdag, prob.Displacement());
      real_t sd = window_->Time(0);
      ok = dop_->Step(wdag, sd, *window_, 0, 1);
      if (!ok) break;
    } else {
      s = s_k;
      if (active) {
        // The first active node (continuous forcing from the start):
        // both states start here.
        ok = op.SolveElastic(mdag, s);
        if (!ok) break;
        window_->Clear();
        window_->Push(s, mdag, prob.Displacement());
        prev_node = 0;
        ok = dop_->Start(wdag, s, *window_, 0);
        if (!ok) break;
      }
    }

    // Forward and tangent fields at node n.
    const GridFunction& u_f = fwd.Displacement(n);
    fwd.Strain(n, dbuf);
    *d_f = dbuf;
    const GridFunction& du_f = tangent_->Displacement(n);
    fop.ComputeStrain(du_f, dbuf);
    *dd_f = dbuf;
    for (int b = 0; b < nb; b++) {
      fop.BranchToFull(fwd.State(n), b, mbuf);
      *m_f[b] = mbuf;
      fop.BranchToFull(tangent_->State(n), b, mbuf);
      *dm_f[b] = mbuf;
    }
    const real_t wR = n < N ? 0.5 * (fwd.Time(n + 1) - fwd.Time(n)) : 0.0;
    const real_t wL = n > 0 ? 0.5 * (fwd.Time(n) - fwd.Time(n - 1)) : 0.0;

    auto pair_state = [&](real_t w) {
      if (w <= 0.0) return;
      for (int b = 0; b < nb; b++) {
        op.BranchToFull(mdag, b, mbuf);
        *mdag_f[b] = mbuf;
        op.BranchToFull(wdag, b, mbuf);
        *wdag_f[b] = mbuf;
      }
      PairAll(w,
              AdjointPair{u_f, d_f.get(), m_ptr, prob.Displacement(), mdag_ptr,
                          fwd.Time(n), false, &prob},
              AdjointPair{du_f, dd_f.get(), dm_ptr, prob.Displacement(),
                          mdag_ptr, fwd.Time(n), false, &prob},
              AdjointPair{u_f, d_f.get(), m_ptr, twin_.Displacement(), wdag_ptr,
                          fwd.Time(n), false, &twin_});
    };

    if (!has_obs) {
      pair_state(wL + wR);
      // Roll the window: this node becomes the next step's start.
      window_->Clear();
      window_->Push(s, mdag, prob.Displacement());
      prev_node = 0;
      continue;
    }
    if (arrived_active) {
      pair_state(wR);
    }
    for (int i : node_obs[n]) {
      // The impulsive solve and its derivative along v.
      op.UseUnrelaxedOperator();
      prob.AssembleForce(s);
      ApplyObservation(i);
      ok = prob.Solve();
      if (!ok) break;
      *udag_imp = prob.Displacement();
      twin_.ClearRelaxationWeights();
      twin_.AssembleForce(s);
      if (!born.Empty()) {
        fbuf_.SetSize(twin_.DisplacementSpace().GetVSize());
        fbuf_ = 0.0;
        born.AddTo(*udag_imp, fbuf_);
        twin_.AddForce(fbuf_);
      }
      ok = twin_.Solve();
      if (!ok) break;
      *wdag_imp = twin_.Displacement();
      PairAll(1.0,
              AdjointPair{u_f, d_f.get(), m_ptr, *udag_imp, {}, fwd.Time(n),
                          true, &prob},
              AdjointPair{du_f, dd_f.get(), dm_ptr, *udag_imp, {}, fwd.Time(n),
                          true, &prob},
              AdjointPair{u_f, d_f.get(), m_ptr, *wdag_imp, {}, fwd.Time(n),
                          true, &twin_});
      // The jumps: [m+] = (1/tau) D u+_imp; [w+] = (1/tau) D w_imp +
      // delta(1/tau) D u+_imp.
      Vector strain_w;
      op.ComputeStrain(*udag_imp, strain_);
      op.ComputeStrain(*wdag_imp, strain_w);
      const int nd_full = op.InternalScalarSpace().GetVSize();
      for (int b = 0; b < nb; b++) {
        const Vector& itau = op.InverseRelaxationTimes(b);
        const Array<int>& nodes = op.BranchNodes(b);
        const int nd = nodes.Size(), off = op.BranchOffset(b);
        const bool has_d = dop_->HasDeltaInverseRelaxationTime(b);
        const Vector* ditau = has_d ? &dop_->DeltaInverseRelaxationTime(b)
                                    : nullptr;
        for (int c = 0; c < op.NumComponents(); c++) {
          for (int q = 0; q < nd; q++) {
            const int full = c * nd_full + nodes[q], idx = off + c * nd + q;
            mdag[idx] += itau[q] * strain_[full];
            wdag[idx] += itau[q] * strain_w[full];
            if (has_d) wdag[idx] += (*ditau)[q] * strain_[full];
          }
        }
      }
    }
    if (!ok) break;
    // The post-jump smooth fields of both states.
    op.InvalidateDisplacement();
    ok = op.SolveElastic(mdag, s);
    if (!ok) break;
    window_->Clear();
    window_->Push(s, mdag, prob.Displacement());
    prev_node = 0;
    ok = dop_->Start(wdag, s, *window_, 0);
    if (!ok) break;
    active = true;
    pair_state(wL);
  }

  if (has_continuous) {
    op.SetForceHook(nullptr);
  }
  return ok;
}

}  // namespace AdGIA
