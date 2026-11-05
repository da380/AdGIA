#include "AdGIA/discrete_second_order.hpp"

#include <cmath>

#include "AdGIA/detail/fem_factory.hpp"

namespace AdGIA {

using namespace mfem;

DiscreteSecondOrderSolver::DiscreteSecondOrderSolver(
    DiscreteAdjointSolver& first, LinearQuasiStaticProblem& derivative_twin,
    const GridFunction& forward_displacement,
    const GridFunction& tangent_displacement, const TangentDirection& v)
    : first_(first),
      twin_(&derivative_twin),
      u_forward_(&forward_displacement),
      du_forward_(&tangent_displacement),
      v_(v) {
  MFEM_VERIFY(first.NumStaticDuals() > 0,
              "DiscreteSecondOrderSolver: run the static first-order adjoint "
              "first.");
}

DiscreteSecondOrderSolver::DiscreteSecondOrderSolver(
    DiscreteAdjointSolver& first, LinearQuasiStaticProblem& derivative_twin,
    LinearQuasiStaticProblem& source_twin_v, ForwardTrajectory& forward,
    const TangentTrajectory& tangent_v, const TangentDirection& v)
    : first_(first),
      op_(&first.Operator()),
      forward_(&forward),
      tangent_v_(&tangent_v),
      v_(v) {
  MFEM_VERIFY(op_->Size() == forward.Size(),
              "DiscreteSecondOrderSolver: run the first-order adjoint first.");
  MFEM_VERIFY(tangent_v.Size() == forward.Size(),
              "DiscreteSecondOrderSolver: the tangent must be on the forward "
              "grid.");
  dop_ = std::make_unique<DiscreteAdjointOperator>(derivative_twin,
                                                   forward.Op());
  src_v_ = std::make_unique<TangentViscoelasticOperator>(source_twin_v,
                                                         forward.Op());
  src_v_->SetDirection(v_);
}

void DiscreteSecondOrderSolver::SecondInverseRelaxationTime(
    int k, const TangentViscoelasticOperator& a,
    const TangentViscoelasticOperator& b, Vector& d2itau) const {
  // 1/tau = mu / eta: d2(1/tau)[a, b] = -(dmu_a deta_b + dmu_b deta_a)/eta^2
  // + 2 mu deta_a deta_b / eta^3, with deta recovered from the stored
  // d(1/tau) and dmu: d(1/tau) = (1/tau)(dmu/mu - deta/eta).
  const int nd = op_->NumNodes();
  const Vector& two_mu = op_->NodalBranchModulus(k);
  const Vector& itau = op_->NodalInverseRelaxationTime(k);
  d2itau.SetSize(nd);
  d2itau = 0.0;
  const bool ha = a.HasDeltaInverseRelaxationTime(k),
             hb = b.HasDeltaInverseRelaxationTime(k);
  if (!ha || !hb) return;
  for (int q = 0; q < nd; q++) {
    const real_t mu = 0.5 * two_mu[q], eta = mu / itau[q];
    const real_t dmu_a = a.HasDeltaBranchModulus(k) ? a.DeltaBranchModulus(k)[q]
                                                    : 0.0;
    const real_t dmu_b = b.HasDeltaBranchModulus(k) ? b.DeltaBranchModulus(k)[q]
                                                    : 0.0;
    const real_t deta_a = eta * (dmu_a / mu - a.DeltaInverseRelaxationTime(k)[q] / itau[q]);
    const real_t deta_b = eta * (dmu_b / mu - b.DeltaInverseRelaxationTime(k)[q] / itau[q]);
    d2itau[q] = -(dmu_a * deta_b + dmu_b * deta_a) / (eta * eta) +
                2.0 * mu * deta_a * deta_b / (eta * eta * eta);
  }
}

real_t DiscreteSecondOrderSolver::Dot(const Vector& dual, const GridFunction& u,
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

real_t DiscreteSecondOrderSolver::SumOverRanks(real_t local,
                                               FiniteElementSpace& fes) const {
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

bool DiscreteSecondOrderSolver::RunStatic() {
  // K dlambda_i = -K(v) lambda_i: the Born force of v on each dual.
  FiniteElementSpace& fes = twin_->DisplacementSpace();
  ModuliBornForce born(fes, v_, twin_->Rheology().NumBranches());
  dlambda_static_.clear();
  Vector f(fes.GetVSize());
  for (int i = 0; i < first_.NumStaticDuals(); i++) {
    twin_->AssembleForce(0.0);
    f = 0.0;
    born.AddTo(first_.StaticDual(i), f);
    twin_->AddForce(f);
    if (!twin_->Solve()) return false;
    dlambda_static_.push_back(detail::MakeGridFunction(&fes));
    *dlambda_static_.back() = twin_->Displacement();
  }
  return true;
}

real_t DiscreteSecondOrderSolver::DirectionalStatic(
    const TangentDirection& delta) {
  MFEM_VERIFY(dlambda_static_.size() == static_cast<std::size_t>(
                                            first_.NumStaticDuals()),
              "DiscreteSecondOrderSolver: Run() first.");
  // H[v, delta] = sum_i <dlambda_i, -K(delta) u> + <lambda_i, -K(delta) du>;
  // the static stiffness is linear in the moduli (no cross term).
  FiniteElementSpace& fes = twin_->DisplacementSpace();
  ModuliBornForce born(fes, delta, twin_->Rheology().NumBranches());
  if (born.Empty()) return 0.0;
  Vector fu(fes.GetVSize()), fdu(fes.GetVSize());
  fu = 0.0;
  fdu = 0.0;
  born.AddTo(*u_forward_, fu);
  born.AddTo(*du_forward_, fdu);
  real_t H = 0.0;
  for (int i = 0; i < first_.NumStaticDuals(); i++) {
    H += Dot(fu, *dlambda_static_[i], fes) + Dot(fdu, first_.StaticDual(i), fes);
  }
  return H;
}

bool DiscreteSecondOrderSolver::Run() {
  if (!op_) return RunStatic();
  const int N = forward_->Size() - 1;
  const int K = op_->NumBranches();
  const int nd = op_->NumNodes(), nc = op_->NumComponents();
  const int size = nd * nc;
  auto& prob = dop_->Problem();
  DualData duals(*op_, *forward_);
  dlambda_.clear();
  dlambda_.resize(N + 1);
  dmu_.assign(N + 1, Vector());
  dmu_[N].SetSize(op_->Height());
  dmu_[N] = 0.0;
  Vector acc(size), cb(size), dcb(size), rhs, f_u;
  std::vector<Vector> zeta;
  for (int n = N; n >= 0; n--) {
    acc = 0.0;
    if (n < N) {
      const real_t dt = forward_->Time(n + 1) - forward_->Time(n);
      op_->Coupling(op_->Lambda(n + 1), cb);
      op_->Coupling(*dlambda_[n + 1], dcb);
      dmu_[n].SetSize(op_->Height());
      for (int k = 0; k < K; k++) {
        const Vector& two_mu = op_->NodalBranchModulus(k);
        const Vector& itau = op_->NodalInverseRelaxationTime(k);
        const bool hd = src_v_->HasDeltaInverseRelaxationTime(k);
        const bool hm = src_v_->HasDeltaBranchModulus(k);
        Vector mu_next = op_->Branch(op_->Mu(n + 1), k);
        Vector dmu_next = op_->Branch(dmu_[n + 1], k);
        Vector dmu_k = op_->Branch(dmu_[n], k);
        for (int q = 0; q < nd; q++) {
          const real_t h = dt * itau[q];
          real_t e, alpha, beta, de, da, db;
          detail::ExponentialTrapezoidWeights(h, e, alpha, beta);
          detail::TrapezoidWeightDerivatives(h, de, da, db);
          const real_t dh = hd ? dt * src_v_->DeltaInverseRelaxationTime(k)[q]
                               : 0.0;
          const real_t dmu_v = hm ? src_v_->DeltaBranchModulus(k)[q] : 0.0;
          for (int c = 0; c < nc; c++) {
            const int i = c * nd + q;
            const real_t s1 = two_mu[q] * dcb[i] + dmu_next[i];   // C B dl + dmu
            const real_t s0 = two_mu[q] * cb[i] + mu_next[i];     // C B l + mu
            const real_t sc = 2.0 * dmu_v * cb[i];                // dC B l
            dmu_k[i] = e * s1 + de * dh * s0 + e * sc;
            acc[i] += alpha * s1 + da * dh * s0 + alpha * sc;
          }
        }
      }
    }
    if (n > 0) {
      const real_t dt = forward_->Time(n) - forward_->Time(n - 1);
      for (int k = 0; k < K; k++) {
        const Vector& itau = op_->NodalInverseRelaxationTime(k);
        const bool hd = src_v_->HasDeltaInverseRelaxationTime(k);
        Vector mu_k = op_->Branch(op_->Mu(n), k);
        Vector dmu_k = op_->Branch(dmu_[n], k);
        for (int q = 0; q < nd; q++) {
          const real_t h = dt * itau[q];
          real_t e, alpha, beta, de, da, db;
          detail::ExponentialTrapezoidWeights(h, e, alpha, beta);
          detail::TrapezoidWeightDerivatives(h, de, da, db);
          const real_t dh = hd ? dt * src_v_->DeltaInverseRelaxationTime(k)[q]
                               : 0.0;
          for (int c = 0; c < nc; c++) {
            const int i = c * nd + q;
            acc[i] += beta * dmu_k[i] + db * dh * mu_k[i];
          }
        }
      }
    }
    dop_->StrainMapTranspose(acc, rhs);
    // -delta_v K_eff lambda^n (the effective Born form of v on lambda;
    // unrelaxed at n = 0).
    if (n > 0) {
      src_v_->PrepareStep(duals, n - 1, n);
      src_v_->ComputeSources(duals, n, true, f_u, zeta);
      dop_->EngageStep(forward_->Time(n) - forward_->Time(n - 1));
    } else {
      src_v_->ComputeSources(duals, 0, false, f_u, zeta);
      dop_->EngageStep(0.0);
    }
    rhs += f_u;
    prob.AssembleForce(forward_->Time(n));
    prob.AddForce(rhs);
    if (!prob.Solve()) return false;
    dlambda_[n] = detail::MakeGridFunction(&prob.DisplacementSpace());
    *dlambda_[n] = prob.Displacement();
  }
  dop_->EngageStep(0.0);
  return true;
}

real_t DiscreteSecondOrderSolver::Directional(
    const TangentDirection& delta, LinearQuasiStaticProblem& source_twin_delta) {
  if (!op_) return DirectionalStatic(delta);
  MFEM_VERIFY(!dlambda_.empty(), "DiscreteSecondOrderSolver: Run() first.");
  const int N = forward_->Size() - 1;
  const int K = op_->NumBranches();
  const int nd = op_->NumNodes(), nc = op_->NumComponents();
  const int dim = forward_->Op().Problem().DisplacementSpace().GetMesh()
                      ->Dimension();
  TangentViscoelasticOperator src_d(source_twin_delta, forward_->Op());
  src_d.SetDirection(delta);
  ForwardTrajectoryData fdata(*forward_);
  TangentTrajectoryData tdata(*tangent_v_, forward_->Op());
  const Rheology& rh = op_->Problem().Rheology();
  FiniteElementSpace& fes_u = op_->Problem().DisplacementSpace();
  Vector f_u, cb, dcb, dn, dn1, d2itau;
  std::vector<Vector> zeta;
  // H: the displacement-row pairings, each global (Dot); internal: the
  // rank's local sums on the element-local internal dofs, reduced once
  // at the end.
  real_t H = 0.0, internal = 0.0;

  // The derivative duals against delta's sources on the forward
  // trajectory, and the first-order duals against delta's sources on
  // v's tangent trajectory (the sources are linear in the data).
  auto pair_sources = [&](TangentData& data, auto lambda_of, auto mu_of) {
    src_d.ComputeSources(data, 0, false, f_u, zeta);
    real_t s = Dot(f_u, lambda_of(0), fes_u);
    op_->Coupling(lambda_of(0), cb);
    for (int k = 0; k < K; k++) {
      if (zeta[k].Size() > 0) internal += zeta[k] * cb;
    }
    for (int n = 0; n < N; n++) {
      src_d.PrepareStep(data, n, n + 1);
      src_d.ComputeSources(data, n + 1, true, f_u, zeta);
      s += Dot(f_u, lambda_of(n + 1), fes_u);
      op_->Coupling(lambda_of(n + 1), cb);
      for (int k = 0; k < K; k++) {
        if (zeta[k].Size() > 0) internal += zeta[k] * cb;
        const Vector& st = src_d.StateSource(k);
        if (st.Size() > 0) {
          Vector mu_k = op_->Branch(mu_of(n + 1), k);
          internal += st * mu_k;
        }
      }
    }
    return s;
  };
  H += pair_sources(fdata, [&](int n) -> const GridFunction& { return *dlambda_[n]; },
                    [&](int n) -> const Vector& { return dmu_[n]; });
  H += pair_sources(tdata, [&](int n) -> const GridFunction& { return op_->Lambda(n); },
                    [&](int n) -> const Vector& { return op_->Mu(n); });

  // The parameter cross terms, per step, with the first-order duals.
  for (int n = 0; n < N; n++) {
    const real_t dt = forward_->Time(n + 1) - forward_->Time(n);
    src_v_->PrepareStep(fdata, n, n + 1);
    src_d.PrepareStep(fdata, n, n + 1);
    forward_->Strain(n, dn);
    forward_->Strain(n + 1, dn1);
    const GridFunction& lam = op_->Lambda(n + 1);
    const GridFunction& u1 = forward_->Displacement(n + 1);
    op_->Coupling(lam, cb);
    // The Born cross term: -<lambda, K(sum_k [dbeta_v dmu_d + dbeta_d dmu_v
    // + d2beta mu_k]) u^{n+1}>.
    std::vector<std::unique_ptr<Coefficient>> owned;
    std::vector<std::unique_ptr<GridFunction>> fields;
    Coefficient* total = nullptr;
    auto add = [&](Coefficient* c) {
      if (!total) {
        total = c;
      } else {
        owned.push_back(std::make_unique<SumCoefficient>(*total, *c));
        total = owned.back().get();
      }
    };
    for (int k = 0; k < K; k++) {
      const Vector& two_mu = op_->NodalBranchModulus(k);
      const Vector& itau = op_->NodalInverseRelaxationTime(k);
      const bool hv = src_v_->HasDeltaInverseRelaxationTime(k);
      const bool hd = src_d.HasDeltaInverseRelaxationTime(k);
      if (hv && delta.Mu(k)) {
        owned.push_back(std::make_unique<GridFunctionCoefficient>(
            &src_v_->DeltaBeta(k)));
        owned.push_back(std::make_unique<ProductCoefficient>(
            *owned.back(), *delta.Mu(k)));
        add(owned.back().get());
      }
      if (hd && v_.Mu(k)) {
        owned.push_back(std::make_unique<GridFunctionCoefficient>(
            &src_d.DeltaBeta(k)));
        owned.push_back(std::make_unique<ProductCoefficient>(
            *owned.back(), *v_.Mu(k)));
        add(owned.back().get());
      }
      if (hv && hd) {
        SecondInverseRelaxationTime(k, *src_v_, src_d, d2itau);
        fields.push_back(detail::MakeGridFunction(&op_->InternalScalarSpace()));
        GridFunction& d2beta = *fields.back();
        for (int q = 0; q < nd; q++) {
          const real_t h = dt * itau[q];
          real_t de, da, db, d2e, d2a, d2b;
          detail::TrapezoidWeightDerivatives(h, de, da, db);
          detail::TrapezoidWeightSecondDerivatives(h, d2e, d2a, d2b);
          const real_t dh_v = dt * src_v_->DeltaInverseRelaxationTime(k)[q];
          const real_t dh_d = dt * src_d.DeltaInverseRelaxationTime(k)[q];
          d2beta[q] = -d2b * dh_v * dh_d - db * dt * d2itau[q];
        }
        owned.push_back(std::make_unique<GridFunctionCoefficient>(&d2beta));
        owned.push_back(std::make_unique<ProductCoefficient>(
            *owned.back(), rh.BranchShearModulus(k)));
        add(owned.back().get());
      }
      // The nodal cross terms of the branch and state sources.
      if (!(hv || hd)) continue;
      Vector mn = op_->Branch(forward_->State(n), k);
      Vector mu1 = op_->Branch(op_->Mu(n + 1), k);
      real_t sum = 0.0;
      for (int q = 0; q < nd; q++) {
        const real_t h = dt * itau[q];
        real_t de, da, db, d2e, d2a, d2b;
        detail::TrapezoidWeightDerivatives(h, de, da, db);
        detail::TrapezoidWeightSecondDerivatives(h, d2e, d2a, d2b);
        const real_t dh_v = hv ? dt * src_v_->DeltaInverseRelaxationTime(k)[q] : 0.0;
        const real_t dh_d = hd ? dt * src_d.DeltaInverseRelaxationTime(k)[q] : 0.0;
        const real_t d2h = (hv && hd) ? dt * d2itau[q] : 0.0;
        const real_t dmu_v = src_v_->HasDeltaBranchModulus(k)
                                 ? src_v_->DeltaBranchModulus(k)[q] : 0.0;
        const real_t dmu_d = src_d.HasDeltaBranchModulus(k)
                                 ? src_d.DeltaBranchModulus(k)[q] : 0.0;
        for (int c = 0; c < nc; c++) {
          const int i = c * nd + q;
          const real_t r1 = de * mn[i] + da * dn[i];
          const real_t r2 = d2e * mn[i] + d2a * dn[i];
          // zeta cross: 2 dmu_d (dv e m + dv a d) + 2 dmu_v dh_d r1
          //             + C [d2h r1 + dh_d dh_v r2]
          real_t z = 2.0 * dmu_d * dh_v * r1 + 2.0 * dmu_v * dh_d * r1 +
                     two_mu[q] * (d2h * r1 + dh_d * dh_v * r2);
          sum += cb[i] * z;
          // s cross: d2h (r1 + db d1) + dh_d dh_v (r2 + d2b d1)
          sum += mu1[i] * (d2h * (r1 + db * dn1[i]) +
                           dh_d * dh_v * (r2 + d2b * dn1[i]));
        }
      }
      internal += sum;
    }
    if (total) {
      auto form = detail::MakeBilinearForm(&forward_->Op().Problem().DisplacementSpace());
      form->AddDomainIntegrator(new ElasticityIntegrator(*total, -2.0 / dim, 1.0));
      form->Assemble();
      form->Finalize();
      Vector ku(u1.Size());
      form->Mult(u1, ku);
      H -= Dot(ku, lam, fes_u);
    }
  }
  return H + SumOverRanks(internal, op_->InternalScalarSpace());
}

// --- the dual vector of H[v, .] -------------------------------------------

int DiscreteSecondOrderSolver::AddKernel(Parameter parameter, int branch) {
  MFEM_VERIFY(op_,
              "DiscreteSecondOrderSolver: the kernel duals are built in "
              "the quasi-static configuration.");
  kernels_.push_back(Kernel{parameter, branch, Vector()});
  return static_cast<int>(kernels_.size()) - 1;
}

void DiscreteSecondOrderSolver::AssembleKernels() {
  MFEM_VERIFY(op_,
              "DiscreteSecondOrderSolver: the kernel duals are built in "
              "the quasi-static configuration.");
  MFEM_VERIFY(!dlambda_.empty(), "DiscreteSecondOrderSolver: Run() first.");
  const int N = forward_->Size() - 1;
  const int K = op_->NumBranches();
  const int nd = op_->NumNodes(), nc = op_->NumComponents();
  const Rheology& rh = op_->Problem().Rheology();
  ViscoelasticOperator& fop = forward_->Op();
  FiniteElementSpace& ufes = op_->Problem().DisplacementSpace();
  FiniteElementSpace& sfes = op_->InternalScalarSpace();
  // Which integrals against the basis the registered kernels need.
  bool need_kappa = false, need_mu_inf = false, need_shear0 = false;
  std::vector<bool> need_branch(K, false), need_mu(K, false);
  for (auto& kernel : kernels_) {
    kernel.dual.SetSize(nd);
    kernel.dual = 0.0;
    switch (kernel.parameter) {
      case Parameter::Kappa:
        need_kappa = true;
        break;
      case Parameter::MuLongTerm:
        need_mu_inf = need_shear0 = true;
        break;
      case Parameter::Mu:
        MFEM_VERIFY(kernel.branch >= 0 && kernel.branch < K,
                    "DiscreteSecondOrderSolver: no such branch.");
        need_shear0 = true;
        need_branch[kernel.branch] = need_mu[kernel.branch] = true;
        break;
      case Parameter::Eta:
        MFEM_VERIFY(kernel.branch >= 0 && kernel.branch < K,
                    "DiscreteSecondOrderSolver: no such branch.");
        need_branch[kernel.branch] = true;
        break;
    }
  }
  auto integrate = [&](Coefficient& density, Vector& out) {
    detail::IntegrateAgainstInternalBasis(density, ufes, sfes, out);
  };
  // The elastic densities: (A) the forward field against the derivative
  // duals, (B) v's tangent field against the first-order duals (summed:
  // the two source pairings share their weights), (1) the forward field
  // against the first-order duals (the cross terms).
  using IKD = IsotropicKernelDensity;
  IKD kA(Parameter::Kappa), kB(Parameter::Kappa);
  IKD sA(Parameter::MuLongTerm), sB(Parameter::MuLongTerm),
      s1(Parameter::MuLongTerm);
  SumCoefficient kAB(kA, kB), sAB(sA, sB);
  Vector cb, dcb, dn, dn1, tdn, tdn1;
  Vector I_kappa, I_shear, I_beta, I_mu, I_mu1, I_dmu1;

  // Node 0, the unrelaxed solve: the Born force (delta mu_U = delta mu_inf
  // + sum_k delta mu_k) and the branch term 2 delta mu_k m_k^0, on the
  // forward fields against the derivative duals and on v's tangent
  // fields against the first-order duals.
  {
    const GridFunction& lam = op_->Lambda(0);
    const GridFunction& dlam = *dlambda_[0];
    const GridFunction& du = tangent_v_->Displacement(0);
    const GridFunction& u = forward_->Displacement(0);
    kA.SetPair(u, dlam);
    kB.SetPair(du, lam);
    sA.SetPair(u, dlam);
    sB.SetPair(du, lam);
    if (need_kappa) integrate(kAB, I_kappa);
    if (need_shear0) integrate(sAB, I_shear);
    op_->Coupling(lam, cb);
    op_->Coupling(dlam, dcb);
    for (auto& kernel : kernels_) {
      const int k = kernel.branch;
      switch (kernel.parameter) {
        case Parameter::Kappa:
          kernel.dual += I_kappa;
          break;
        case Parameter::MuLongTerm:
          kernel.dual += I_shear;
          break;
        case Parameter::Mu: {
          kernel.dual += I_shear;
          Vector m0 = op_->Branch(forward_->State(0), k);
          Vector dm0 = op_->Branch(tangent_v_->State(0), k);
          for (int q = 0; q < nd; q++) {
            real_t v = 0.0;
            for (int c = 0; c < nc; c++) {
              const int i = c * nd + q;
              v += 2.0 * (dcb[i] * m0[i] + cb[i] * dm0[i]);
            }
            kernel.dual[q] += v;
          }
          break;
        }
        case Parameter::Eta:
          break;
      }
    }
  }

  // The steps n -> n+1: the effective solve at n+1, the known-part branch
  // sources and the state source (both source pairings), and the cross
  // terms with the first-order duals.
  for (int n = 0; n < N; n++) {
    const real_t dt = forward_->Time(n + 1) - forward_->Time(n);
    op_->EngageStep(dt);
    forward_->Strain(n, dn);
    forward_->Strain(n + 1, dn1);
    fop.ComputeStrain(tangent_v_->Displacement(n), tdn);
    fop.ComputeStrain(tangent_v_->Displacement(n + 1), tdn1);
    // Read last: in the re-solve policy the reference holds until the
    // next Displacement() or Strain() call.
    const GridFunction& u = forward_->Displacement(n + 1);
    const GridFunction& du = tangent_v_->Displacement(n + 1);
    const GridFunction& lam = op_->Lambda(n + 1);
    const GridFunction& dlam = *dlambda_[n + 1];
    op_->Coupling(lam, cb);
    op_->Coupling(dlam, dcb);
    kA.SetPair(u, dlam);
    kB.SetPair(du, lam);
    sA.SetPair(u, dlam);
    sB.SetPair(du, lam);
    s1.SetPair(u, lam);
    if (need_kappa) integrate(kAB, I_kappa);
    if (need_mu_inf) integrate(sAB, I_shear);
    for (auto& kernel : kernels_) {
      if (kernel.parameter == Parameter::Kappa) kernel.dual += I_kappa;
      if (kernel.parameter == Parameter::MuLongTerm) kernel.dual += I_shear;
    }
    for (int k = 0; k < K; k++) {
      if (!need_branch[k]) continue;
      const Vector& two_mu = op_->NodalBranchModulus(k);
      const Vector& itau = op_->NodalInverseRelaxationTime(k);
      const bool hv = src_v_->HasDeltaInverseRelaxationTime(k);
      const bool hmv = src_v_->HasDeltaBranchModulus(k);
      Coefficient* vmu = v_.Mu(k);
      Coefficient& mu_k = rh.BranchShearModulus(k);
      // int_q mu_k rho_AB (the weights' derivative in both source
      // pairings); int_q mu_k rho_1 and int_q dmu_v rho_1 (the cross
      // terms' d2beta mu_k and dbeta_delta dmu_v).
      ProductCoefficient mu_AB(mu_k, sAB);
      integrate(mu_AB, I_mu);
      if (hv) {
        ProductCoefficient mu_1(mu_k, s1);
        integrate(mu_1, I_mu1);
      }
      if (vmu) {
        ProductCoefficient dmu_1(*vmu, s1);
        integrate(dmu_1, I_dmu1);
      }
      if (need_mu[k]) {
        // int_q [beta_k rho_AB + dbeta_v rho_1]: the branch modulus in the
        // effective stiffness, and the cross term dbeta_v dmu_delta.
        ProductCoefficient beta_AB(op_->BetaCoefficient(k), sAB);
        if (hv) {
          auto dbeta_v = detail::MakeGridFunction(&sfes);
          for (int q = 0; q < nd; q++) {
            real_t de, da, db;
            detail::TrapezoidWeightDerivatives(dt * itau[q], de, da, db);
            (*dbeta_v)[q] = -db * dt * src_v_->DeltaInverseRelaxationTime(k)[q];
          }
          GridFunctionCoefficient dbeta_c(dbeta_v.get());
          ProductCoefficient dbeta_1(dbeta_c, s1);
          SumCoefficient total(beta_AB, dbeta_1);
          integrate(total, I_beta);
        } else {
          integrate(beta_AB, I_beta);
        }
      }
      Vector mn = op_->Branch(forward_->State(n), k);
      Vector tmn = op_->Branch(tangent_v_->State(n), k);
      Vector mu1 = op_->Branch(op_->Mu(n + 1), k);
      Vector dmu1 = op_->Branch(dmu_[n + 1], k);
      for (auto& kernel : kernels_) {
        if (kernel.branch != k || (kernel.parameter != Parameter::Mu &&
                                   kernel.parameter != Parameter::Eta)) {
          continue;
        }
        const bool is_mu = kernel.parameter == Parameter::Mu;
        const real_t chi = is_mu ? 1.0 : 0.0;
        for (int q = 0; q < nd; q++) {
          const real_t mu = 0.5 * two_mu[q], eta = mu / itau[q];
          const real_t h = dt * itau[q];
          real_t e, alpha, beta, de, da, db, d2e, d2a, d2b;
          detail::ExponentialTrapezoidWeights(h, e, alpha, beta);
          detail::TrapezoidWeightDerivatives(h, de, da, db);
          detail::TrapezoidWeightSecondDerivatives(h, d2e, d2a, d2b);
          // v's nodal data.
          const real_t ditau_v =
              hv ? src_v_->DeltaInverseRelaxationTime(k)[q] : 0.0;
          const real_t dmu_v = hmv ? src_v_->DeltaBranchModulus(k)[q] : 0.0;
          const real_t dh_v = dt * ditau_v;
          const real_t deta_v = eta * (dmu_v / mu - ditau_v / itau[q]);
          // The nodal factors: c = d(1/tau)/dp, g = d2(1/tau)/dp dv.
          const real_t cf = is_mu ? itau[q] / mu : -itau[q] / eta;
          const real_t g =
              is_mu ? -deta_v / (eta * eta)
                    : (2.0 * mu * deta_v / eta - dmu_v) / (eta * eta);
          // The elastic terms.
          real_t v = -db * dt * cf * I_mu[q];
          if (is_mu) v += I_beta[q];
          if (vmu) v += -db * dt * cf * I_dmu1[q];
          if (hv) v += (-d2b * dt * cf * dh_v - db * dt * g) * I_mu1[q];
          // The nodal terms.
          for (int c = 0; c < nc; c++) {
            const int i = c * nd + q;
            // The source pairings: the forward data against (dlambda,
            // dmu), v's tangent data against (lambda, mu).
            const real_t r1 = de * mn[i] + da * dn[i];
            const real_t rt = de * tmn[i] + da * tdn[i];
            v += dcb[i] * (2.0 * chi * (e * mn[i] + alpha * dn[i]) +
                           two_mu[q] * dt * cf * r1);
            v += dmu1[i] * dt * cf * (r1 + db * dn1[i]);
            v += cb[i] * (2.0 * chi * (e * tmn[i] + alpha * tdn[i]) +
                          two_mu[q] * dt * cf * rt);
            v += mu1[i] * dt * cf * (rt + db * tdn1[i]);
            // The cross terms of the branch and state sources.
            const real_t r2 = d2e * mn[i] + d2a * dn[i];
            v += cb[i] * (2.0 * chi * dh_v * r1 + 2.0 * dmu_v * dt * cf * r1 +
                          two_mu[q] * dt * (g * r1 + cf * dh_v * r2));
            v += mu1[i] * dt *
                 (g * (r1 + db * dn1[i]) + cf * dh_v * (r2 + d2b * dn1[i]));
          }
          kernel.dual[q] += v;
        }
      }
    }
  }
  op_->EngageStep(0.0);
}

real_t DiscreteSecondOrderSolver::Pair(int i, const GridFunction& delta) const {
  const Vector& d = kernels_[i].dual;
  MFEM_VERIFY(d.Size() == delta.Size(),
              "DiscreteSecondOrderSolver::Pair: the direction must live on "
              "the internal scalar space.");
  // The internal space is discontinuous: a local pairing, reduced.
  return SumOverRanks(d * delta,
                      *const_cast<FiniteElementSpace*>(delta.FESpace()));
}

}  // namespace AdGIA
