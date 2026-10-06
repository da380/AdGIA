#include "AdGIA/background.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numbers>
#include <utility>

#include "AdGIA/detail/fem_factory.hpp"
#include "AdGIA/null_space.hpp"

namespace AdGIA {

using namespace mfem;

RadialHydrostaticState::RadialHydrostaticState(
    int dim, std::function<real_t(real_t)> rho, real_t G, real_t radius,
    int samples)
    : dim_(dim), rho_(std::move(rho)), G_(G), R_(radius) {
  MFEM_VERIFY(dim == 2 || dim == 3, "dimension must be 2 or 3");
  MFEM_VERIFY(radius > 0.0 && samples > 1, "invalid radius or sample count");
  h_ = R_ / samples;
  const real_t four_pi_G = 4.0 * std::numbers::pi * G_;

  // Cumulative trapezoid for m(r) = int_0^r rho s^{d-1} ds, then
  // g = 4 pi G m / r^{d-1}.
  g_.assign(samples + 1, 0.0);
  real_t m = 0.0;
  auto integrand = [&](real_t r) {
    return rho_(r) * (dim_ == 2 ? r : r * r);
  };
  real_t prev = integrand(0.0);
  for (int i = 1; i <= samples; i++) {
    const real_t r = i * h_;
    const real_t cur = integrand(r);
    m += 0.5 * h_ * (prev + cur);
    prev = cur;
    g_[i] = four_pi_G * m / (dim_ == 2 ? r : r * r);
  }

  // Cumulative trapezoid inward for p(r) = int_r^R rho g ds, p(R) = 0.
  p_.assign(samples + 1, 0.0);
  for (int i = samples - 1; i >= 0; i--) {
    const real_t f0 = rho_(i * h_) * g_[i];
    const real_t f1 = rho_((i + 1) * h_) * g_[i + 1];
    p_[i] = p_[i + 1] + 0.5 * h_ * (f0 + f1);
  }
}

real_t RadialHydrostaticState::Gravity(real_t r) const {
  if (r >= R_) {
    // Exterior field of the total mass.
    const real_t g_R = g_.back();
    return dim_ == 2 ? g_R * (R_ / r) : g_R * (R_ / r) * (R_ / r);
  }
  const real_t s = std::max(real_t{0}, r) / h_;
  const int i =
      std::min(static_cast<int>(s), static_cast<int>(g_.size()) - 2);
  const real_t w = s - i;
  return (1.0 - w) * g_[i] + w * g_[i + 1];
}

real_t RadialHydrostaticState::Pressure(real_t r) const {
  if (r >= R_) {
    return 0.0;
  }
  const real_t s = std::max(real_t{0}, r) / h_;
  const int i =
      std::min(static_cast<int>(s), static_cast<int>(p_.size()) - 2);
  const real_t w = s - i;
  return (1.0 - w) * p_[i] + w * p_[i + 1];
}

real_t RadialHydrostaticState::Density(real_t r) const {
  return r > R_ ? 0.0 : rho_(std::max(real_t{0}, r));
}

// Hidden from Doxygen, which cannot match the unqualified parameter types
// of this constructor to its declaration.
/// @cond
RadialHydrostaticBackground::RadialHydrostaticBackground(
    int dim, RadialFunc rho, RadialFunc kappa, RadialFunc mu, real_t G,
    real_t radius, int samples)
    : dim_(dim),
      kappa_fn_(std::move(kappa)),
      mu_fn_(std::move(mu)),
      state_(dim, std::move(rho), G, radius, samples),
      rho_coef_([this](const Vector& x) { return state_.Density(x.Norml2()); }),
      kappa_coef_([this](const Vector& x) { return kappa_fn_(x.Norml2()); }),
      mu_coef_([this](const Vector& x) { return mu_fn_(x.Norml2()); }),
      p0_coef_([this](const Vector& x) { return state_.Pressure(x.Norml2()); }),
      C_eff_(IsotropicElasticTensorCoefficient::FromBulkModulus(
          dim, kappa_coef_, mu_coef_)),
      C_(dim, C_eff_, p0_coef_),
      minus_p0_(-1.0, p0_coef_),
      identity_(dim),
      S_(minus_p0_, identity_),
      phi_e_(dim),
      rheology_(dim, C_, S_, phi_e_) {}
/// @endcond

RelabelledBackground::RelabelledBackground(RadialHydrostaticBackground& base,
                                           Diffeomorphism& xi)
    : base_(&base),
      xi_(&xi),
      rho_xi_(xi,
              [this](const Vector& y) { return base_->DensityAt(y.Norml2()); }),
      kappa_xi_(xi,
                [this](const Vector& y) { return base_->KappaAt(y.Norml2()); }),
      mu_xi_(xi, [this](const Vector& y) { return base_->MuAt(y.Norml2()); }),
      p0_xi_(xi,
             [this](const Vector& y) { return base_->PressureAt(y.Norml2()); }),
      C_eff_xi_(IsotropicElasticTensorCoefficient::FromBulkModulus(
          base.SpaceDim(), kappa_xi_, mu_xi_)),
      C_comp_(base.SpaceDim(), C_eff_xi_, p0_xi_),
      C_rel_(base.SpaceDim(), C_comp_, xi),
      S_comp_(base.SpaceDim(), xi,
              [this](const Vector& y, DenseMatrix& S) {
                S.SetSize(y.Size());
                S = 0.0;
                const real_t p = base_->PressureAt(y.Norml2());
                for (int i = 0; i < y.Size(); i++) {
                  S(i, i) = -p;
                }
              }),
      S_rel_(base.SpaceDim(), S_comp_, xi),
      jac_(xi),
      rho_rel_(rho_xi_, jac_),
      rheology_(base.SpaceDim(), C_rel_, S_rel_, xi) {}

namespace {

// A pressure signed per boundary element so that it always pairs with
// the mesh's OUTWARD normal: a SubMesh inherits its interface boundary
// elements with the parent's stored orientation, which points outward
// of the region on ONE side only (measured: centre-outward on the
// layered meshes — the inner core's outward normal, the mantle's inward
// one). The sign compares the stored normal with the line from the
// adjacent element's centre to the face's.
class OutwardSignedCoefficient : public Coefficient {
 public:
  OutwardSignedCoefficient(Mesh& mesh, Coefficient& p) : p_(&p) {
    const int dim = mesh.SpaceDimension();
    signs_.SetSize(mesh.GetNBE());
    Vector nor(dim), fc(dim), ec(dim);
    for (int b = 0; b < mesh.GetNBE(); b++) {
      int el, info;
      mesh.GetBdrElementAdjacentElement(b, el, info);
      ElementTransformation* bt = mesh.GetBdrElementTransformation(b);
      const IntegrationPoint& bip =
          Geometries.GetCenter(mesh.GetBdrElementGeometry(b));
      bt->SetIntPoint(&bip);
      CalcOrtho(bt->Jacobian(), nor);
      bt->Transform(bip, fc);
      ElementTransformation* et = mesh.GetElementTransformation(el);
      et->Transform(Geometries.GetCenter(mesh.GetElementGeometry(el)), ec);
      fc -= ec;
      signs_[b] = (nor * fc) >= 0.0 ? 1.0 : -1.0;
    }
  }

  real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    // Boundary integrators evaluate on the boundary element's own
    // transformation, whose ElementNo is the boundary index.
    return signs_[T.ElementNo] * p_->Eval(T, ip);
  }

 private:
  Coefficient* p_;
  Vector signs_;
};

// True-dof assembly of a linear form: B = P^T L (the parallel reduction;
// the identity in serial).
void AssembleTrueRHS(FiniteElementSpace& fes, LinearForm& lf, Vector& B) {
  lf.Assemble();
  B.SetSize(fes.GetTrueVSize());
  const Operator* P = fes.GetProlongationMatrix();
  if (P) {
    P->MultTranspose(lf, B);
  } else {
    B = lf;
  }
}

std::unique_ptr<BilinearForm> MakeBilinearForm(FiniteElementSpace& fes) {
#ifdef MFEM_USE_MPI
  if (auto* pfes = dynamic_cast<ParFiniteElementSpace*>(&fes)) {
    return std::make_unique<ParBilinearForm>(pfes);
  }
#endif
  return std::make_unique<BilinearForm>(&fes);
}

std::unique_ptr<MixedBilinearForm> MakeMixedBilinearForm(
    FiniteElementSpace& trial, FiniteElementSpace& test) {
#ifdef MFEM_USE_MPI
  auto* ptrial = dynamic_cast<ParFiniteElementSpace*>(&trial);
  auto* ptest = dynamic_cast<ParFiniteElementSpace*>(&test);
  if (ptrial && ptest) {
    return std::make_unique<ParMixedBilinearForm>(ptrial, ptest);
  }
#endif
  return std::make_unique<MixedBilinearForm>(&trial, &test);
}

// A GS (serial) or AMG (parallel) preconditioner for an assembled
// operator handle.
std::unique_ptr<Solver> MakePreconditioner(const OperatorHandle& A) {
#ifdef MFEM_USE_MPI
  if (A.Type() == Operator::Hypre_ParCSR) {
    auto amg = std::make_unique<HypreBoomerAMG>(
        *const_cast<OperatorHandle&>(A).As<HypreParMatrix>());
    amg->SetPrintLevel(0);
    return amg;
  }
#endif
  return std::make_unique<GSSmoother>(
      *const_cast<OperatorHandle&>(A).As<SparseMatrix>());
}

#ifdef MFEM_USE_MPI
MPI_Comm CommOf(FiniteElementSpace& fes, bool& parallel) {
  if (auto* pfes = dynamic_cast<ParFiniteElementSpace*>(&fes)) {
    parallel = true;
    return pfes->GetComm();
  }
  parallel = false;
  return MPI_COMM_NULL;
}
#endif

// The rigid kernel of the (possibly pulled-back) generator problems:
// translations, and rotations of the mapped positions.
std::unique_ptr<NullSpaceProjector> MakeGeneratorProjector(
    FiniteElementSpace& fes, Diffeomorphism* map) {
  if (!map) {
    return MakeRigidModeProjector(fes);
  }
  std::unique_ptr<NullSpaceProjector> P;
#ifdef MFEM_USE_MPI
  if (auto* pfes = dynamic_cast<ParFiniteElementSpace*>(&fes)) {
    P = std::make_unique<NullSpaceProjector>(pfes->GetComm());
  } else
#endif
  {
    P = std::make_unique<NullSpaceProjector>();
  }
  const int dim = fes.GetMesh()->SpaceDimension();
  auto gf = detail::MakeGridFunction(&fes);
  Vector t;
  auto add = [&](VectorCoefficient& c) {
    gf->ProjectCoefficient(c);
    gf->GetTrueDofs(t);
    P->Add(t);
  };
  for (int c = 0; c < dim; c++) {
    Vector e(dim);
    e = 0.0;
    e[c] = 1.0;
    VectorConstantCoefficient tc(e);
    add(tc);
  }
  if (dim == 2) {
    MappedRotation rot(*map, 2);
    add(rot);
  } else {
    for (int c = 0; c < 3; c++) {
      MappedRotation rot(*map, c);
      add(rot);
    }
  }
  return P;
}

// The pulled-back stress at a point: T o phi = 2 mu sym(G F^{-1}) - p 1,
// then S = J F^{-1} (T o phi) F^{-T} (the identity map when map is null).
void PullbackStressAt(Diffeomorphism* map, ElementTransformation& T,
                      const IntegrationPoint& ip, const DenseMatrix& G,
                      real_t two_mu, real_t p, DenseMatrix& F, DenseMatrix& Fi,
                      DenseMatrix& A, DenseMatrix& S, DenseMatrix& tmp,
                      DenseMatrix& K) {
  const int d = G.Height();
  K.SetSize(d);
  if (!map) {
    for (int i = 0; i < d; i++) {
      for (int j = 0; j < d; j++) {
        K(i, j) = 0.5 * two_mu * (G(i, j) + G(j, i));
      }
      K(i, i) -= p;
    }
    return;
  }
  map->EvalGradient(F, T, ip);
  Fi.SetSize(d);
  CalcInverse(F, Fi);
  const real_t J = F.Det();
  A.SetSize(d);
  Mult(G, Fi, A);
  S.SetSize(d);
  for (int i = 0; i < d; i++) {
    for (int j = 0; j < d; j++) {
      S(i, j) = 0.5 * two_mu * (A(i, j) + A(j, i));
    }
    S(i, i) -= p;
  }
  tmp.SetSize(d);
  Mult(Fi, S, tmp);
  MultABt(tmp, Fi, K);
  K *= J;
}

}  // namespace

MinimumNormEquilibriumStress::MinimumNormEquilibriumStress(
    FiniteElementSpace& fes, VectorCoefficient& body_force, Coefficient* mu,
    Diffeomorphism* map)
    : MatrixCoefficient(fes.GetMesh()->SpaceDimension()),
      fes_(&fes),
      mu_(mu),
      map_(map),
      half_(0.5) {
  const int dim = fes.GetMesh()->SpaceDimension();
  MFEM_VERIFY(fes.GetVDim() == dim,
              "MinimumNormEquilibriumStress: a vector space is needed");
  Coefficient& m = mu_ ? *mu_ : half_;

  // AW10 eq. (56): Div(2 mu grad_s u) = f, traction-free; weakly
  // int 2 mu e(u):e(v) = -(f, v), rigid modes projected. In mapped mode
  // the form is pulled back with the standard relabelling recipe.
  ConstantCoefficient zero(0.0);
  IsotropicElasticTensorCoefficient C_iso(dim, zero, m);
  auto a = MakeBilinearForm(fes);
  a->AddDomainIntegrator(map_ ? new ElasticTensorIntegrator(C_iso, *map_)
                              : new ElasticTensorIntegrator(C_iso));
  a->Assemble();
  OperatorHandle A;
  Array<int> empty;
  a->FormSystemMatrix(empty, A);

  LinearForm lf(&fes);
  std::unique_ptr<JacobianCoefficient> jac;
  std::unique_ptr<ScalarVectorProductCoefficient> fJ;
  if (map_) {
    jac = std::make_unique<JacobianCoefficient>(*map_);
    fJ = std::make_unique<ScalarVectorProductCoefficient>(*jac, body_force);
    lf.AddDomainIntegrator(new VectorDomainLFIntegrator(*fJ));
  } else {
    lf.AddDomainIntegrator(new VectorDomainLFIntegrator(body_force));
  }
  Vector B;
  AssembleTrueRHS(fes, lf, B);
  B *= -1.0;

  auto projector = MakeGeneratorProjector(fes, map_);
  auto prec = MakePreconditioner(A);
  std::unique_ptr<CGSolver> cg;
#ifdef MFEM_USE_MPI
  bool parallel = false;
  MPI_Comm comm = CommOf(fes, parallel);
  cg = parallel ? std::make_unique<CGSolver>(comm)
                : std::make_unique<CGSolver>();
#else
  cg = std::make_unique<CGSolver>();
#endif
  ProjectedOperator op(*A.Ptr(), *projector);
  ProjectedSolver prec_p(*projector);
  prec_p.SetSolver(*prec);
  cg->SetOperator(op);
  cg->SetPreconditioner(prec_p);
  cg->SetRelTol(1e-12);
  cg->SetAbsTol(0.0);
  cg->SetMaxIter(20000);
  cg->SetPrintLevel(0);
  Vector X(B.Size());
  X = 0.0;
  ProjectedSolver solver(*projector);
  solver.SetSolver(*cg);
  solver.Mult(B, X);
  MFEM_VERIFY(cg->GetConverged(),
              "MinimumNormEquilibriumStress: the elastic solve did not "
              "converge (is the body force self-equilibrated?)");
  iterations_ = cg->GetNumIterations();

  u_ = detail::MakeGridFunction(&fes);
  u_->SetFromTrueDofs(X);
}

void MinimumNormEquilibriumStress::Eval(DenseMatrix& K,
                                        ElementTransformation& T,
                                        const IntegrationPoint& ip) {
  T.SetIntPoint(&ip);
  u_->GetVectorGradient(T, G_);
  const real_t two_mu = 2.0 * (mu_ ? mu_->Eval(T, ip) : 0.5);
  PullbackStressAt(map_, T, ip, G_, two_mu, 0.0, F_, Fi_, A_, S_, tmp_, K);
}

// Everything the saddle keeps assembled: the system, its blocks, the
// preconditioner, the null-space projector, the solver chain, and —
// with rigid boundary groups — the border (lifted fields, columns,
// their precomputed base solves and the reduced dense system).
struct StokesSaddleSolver::Impl {
  mfem::ConstantCoefficient half{0.5};
  mfem::Array<int> ess_tdofs, offsets;
  std::unique_ptr<BilinearForm> a, mp;
  std::unique_ptr<MixedBilinearForm> g_form;
  OperatorHandle A, G, Mp;
  std::unique_ptr<TransposeOperator> Gt;
  std::unique_ptr<BlockOperator> block_op;
  std::unique_ptr<Solver> prec_u, prec_p;
  std::unique_ptr<BlockDiagonalPreconditioner> block_prec;
  std::unique_ptr<NullSpaceProjector> projector;
  std::unique_ptr<MINRESSolver> minres;
  std::unique_ptr<ProjectedOperator> op;
  std::unique_ptr<ProjectedSolver> prec_proj, solver;

  FiniteElementSpace* fes_u = nullptr;
  bool parallel = false;
#ifdef MFEM_USE_MPI
  MPI_Comm comm = MPI_COMM_NULL;
#endif

  // The rigid border (one group per enclosed solid component).
  int rigid_modes = 0;
  std::unique_ptr<BilinearForm> a_full_form;
  std::unique_ptr<MixedBilinearForm> g_full_form;
  OperatorHandle A_full, G_full;
  std::vector<std::unique_ptr<GridFunction>> rigid_fields;
  std::vector<Vector> R;      // the lifted fields' true dofs
  std::vector<Vector> B_col;  // border columns [ess-zeroed A_full R; -G^T R]
  std::vector<Vector> Y;      // S^{-1} B_col, precomputed
  DenseMatrix reduced;        // D - B^T Y
  DenseMatrixInverse reduced_inv;

  real_t GlobalDot(const Vector& x, const Vector& y) const {
    real_t d = x * y;
#ifdef MFEM_USE_MPI
    if (parallel) {
      real_t global = 0.0;
      MPI_Allreduce(&d, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
      d = global;
    }
#endif
    return d;
  }
};

std::function<void(const Vector&, Vector&)> RigidMode(int dim, int mode) {
  return [dim, mode](const Vector& x, Vector& v) {
    v.SetSize(dim);
    v = 0.0;
    if (dim == 2) {
      if (mode < 2) {
        v[mode] = 1.0;
      } else {
        v[0] = -x[1];
        v[1] = x[0];
      }
    } else {
      if (mode < 3) {
        v[mode] = 1.0;
      } else {
        const int i = mode - 3;
        // e_i x x
        v[(i + 1) % 3] = -x[(i + 2) % 3];
        v[(i + 2) % 3] = x[(i + 1) % 3];
      }
    }
  };
}

StokesSaddleSolver::StokesSaddleSolver(
    FiniteElementSpace& fes_u, FiniteElementSpace& fes_p, Coefficient* mu,
    Diffeomorphism* map, const Array<int>* essential_bdr,
    const std::vector<Array<int>>* rigid_bdr)
    : impl_(std::make_unique<Impl>()) {
  Impl& s = *impl_;
  s.fes_u = &fes_u;
  const int dim = fes_u.GetMesh()->SpaceDimension();
  MFEM_VERIFY(fes_u.GetVDim() == dim && fes_p.GetVDim() == 1 &&
                  fes_u.GetMesh() == fes_p.GetMesh(),
              "StokesSaddleSolver: vector velocity and scalar pressure "
              "spaces on one mesh are needed");
  MFEM_VERIFY(
      fes_p.FEColl()->GetOrder() < fes_u.FEColl()->GetOrder(),
      "StokesSaddleSolver: the pressure space must sit at least one "
      "polynomial order below the velocity space (Taylor-Hood); "
      "equal-order interpolation violates the inf-sup (LBB) condition and "
      "produces spurious pressure modes.");
  Coefficient& m = mu ? *mu : s.half;

  // With an essential marker the velocity is clamped there (the
  // fluid-only feasibility variant); the true dofs are eliminated from
  // the saddle in the usual way, with zero data. A rigid group's
  // boundary is clamped too — its rigid motions return through the
  // border below. (The groups' boundaries are assumed disjoint from
  // each other and from the clamped set, as concentric interfaces
  // are.)
  const int n_bdr = fes_u.GetMesh()->bdr_attributes.Size()
                        ? fes_u.GetMesh()->bdr_attributes.Max()
                        : 0;
  Array<int> clamp(n_bdr);
  clamp = 0;
  if (essential_bdr) {
    for (int i = 0; i < std::min(n_bdr, essential_bdr->Size()); i++) {
      clamp[i] = (*essential_bdr)[i];
    }
  }
  if (rigid_bdr) {
    for (const auto& group : *rigid_bdr) {
      for (int i = 0; i < std::min(n_bdr, group.Size()); i++) {
        if (group[i]) {
          clamp[i] = 1;
        }
      }
    }
  }
  const bool any_clamped = [&]() {
    for (int i = 0; i < clamp.Size(); i++) {
      if (clamp[i]) {
        return true;
      }
    }
    return false;
  }();
  if (any_clamped) {
    fes_u.GetEssentialTrueDofs(clamp, s.ess_tdofs);
  }

  // AW10 eqs. (73)-(74): the steady incompressible Stokes problem with
  // traction boundary conditions, as the symmetric saddle system
  //   [ A  -G   ] [u]   [-F]
  //   [-G^T  0  ] [p] = [ 0],   A = int 2 mu e(u):e(v),  G = (p, div v),
  // each form pulled back with the relabelling recipe in mapped mode.
  ConstantCoefficient zero(0.0);
  IsotropicElasticTensorCoefficient C_iso(dim, zero, m);
  s.a = MakeBilinearForm(fes_u);
  s.a->AddDomainIntegrator(map ? new ElasticTensorIntegrator(C_iso, *map)
                               : new ElasticTensorIntegrator(C_iso));
  s.a->Assemble();
  Array<int> empty;
  s.a->FormSystemMatrix(s.ess_tdofs, s.A);

  ConstantCoefficient one(1.0);
  s.g_form = MakeMixedBilinearForm(fes_p, fes_u);
  s.g_form->AddDomainIntegrator(
      map ? new DomainDivVectorScalarIntegrator(*map)
          : new DomainDivVectorScalarIntegrator());
  s.g_form->Assemble();
  s.g_form->FormRectangularSystemMatrix(empty, s.ess_tdofs, s.G);
  s.Gt = std::make_unique<TransposeOperator>(*s.G.Ptr());

  s.offsets.SetSize(3);
  s.offsets[0] = 0;
  s.offsets[1] = fes_u.GetTrueVSize();
  s.offsets[2] = fes_p.GetTrueVSize();
  s.offsets.PartialSum();
  s.block_op = std::make_unique<BlockOperator>(s.offsets);
  s.block_op->SetBlock(0, 0, s.A.Ptr());
  s.block_op->SetBlock(0, 1, s.G.Ptr(), -1.0);
  s.block_op->SetBlock(1, 0, s.Gt.get(), -1.0);

  // Pressure-block preconditioner: the mass matrix (the Stokes Schur
  // complement up to the mu weight).
  s.mp = MakeBilinearForm(fes_p);
  s.mp->AddDomainIntegrator(new MassIntegrator(one));
  s.mp->Assemble();
  s.mp->FormSystemMatrix(empty, s.Mp);
  s.prec_u = MakePreconditioner(s.A);
  s.prec_p = MakePreconditioner(s.Mp);
  s.block_prec = std::make_unique<BlockDiagonalPreconditioner>(s.offsets);
  s.block_prec->SetDiagonalBlock(0, s.prec_u.get());
  s.block_prec->SetDiagonalBlock(1, s.prec_p.get());

  // Null space. Traction everywhere: the rigid modes of u alone (mapped
  // rotations in mapped mode), and the pressure has NO constant
  // ambiguity. Any clamping kills the rigid modes; clamping on the
  // WHOLE boundary instead gives the pressure its classical constant
  // mode (G 1_p has entries over interior test functions only, each the
  // integral of a divergence with vanishing trace).
#ifdef MFEM_USE_MPI
  bool parallel = false;
  MPI_Comm comm = CommOf(fes_u, parallel);
  s.parallel = parallel;
  s.comm = comm;
  s.projector = parallel ? std::make_unique<NullSpaceProjector>(comm)
                         : std::make_unique<NullSpaceProjector>();
#else
  s.projector = std::make_unique<NullSpaceProjector>();
#endif
  if (!any_clamped) {
    auto rigid = MakeGeneratorProjector(fes_u, map);
    BlockVector n(s.offsets);
    for (int i = 0; i < rigid->Size(); i++) {
      n.GetBlock(0) = rigid->Basis(i);
      n.GetBlock(1) = 0.0;
      s.projector->Add(n);
    }
  } else {
    bool whole_boundary = true;
    const Array<int>& bdr = fes_u.GetMesh()->bdr_attributes;
    for (int i = 0; i < bdr.Size(); i++) {
      if (bdr[i] > clamp.Size() || !clamp[bdr[i] - 1]) {
        whole_boundary = false;
      }
    }
    if (whole_boundary) {
      // The pressure's constant mode survives the border: the lifted
      // fields' fluxes through their own boundary vanish.
      BlockVector n(s.offsets);
      n.GetBlock(0) = 0.0;
      n.GetBlock(1) = 1.0;  // H1 Lagrange: the constant's true dofs
      s.projector->Add(n);
    }
  }

#ifdef MFEM_USE_MPI
  s.minres = parallel ? std::make_unique<MINRESSolver>(comm)
                      : std::make_unique<MINRESSolver>();
#else
  s.minres = std::make_unique<MINRESSolver>();
#endif
  s.op = std::make_unique<ProjectedOperator>(*s.block_op, *s.projector);
  s.prec_proj = std::make_unique<ProjectedSolver>(*s.projector);
  s.prec_proj->SetSolver(*s.block_prec);
  s.minres->SetOperator(*s.op);
  s.minres->SetPreconditioner(*s.prec_proj);
  s.minres->SetRelTol(1e-11);
  s.minres->SetAbsTol(0.0);
  s.minres->SetMaxIter(50000);
  s.minres->SetPrintLevel(0);
  s.solver = std::make_unique<ProjectedSolver>(*s.projector);
  s.solver->SetSolver(*s.minres);

  // --- The rigid border -----------------------------------------------------
  // Per group: lifted fields R_k (rigid trace on the group's boundary,
  // zero on every other clamped one), the uneliminated operators for
  // the border blocks, the columns B_k = [A R_k (free rows); -G^T R_k],
  // their base solves Y_k = S^{-1} B_k — paid once here — and the
  // reduced dense system D - B^T Y. A bordered Solve then costs one
  // base solve plus dense algebra.
  if (rigid_bdr && !rigid_bdr->empty()) {
    MFEM_VERIFY(!map, "StokesSaddleSolver: rigid boundary groups are not "
                      "implemented in mapped mode.");
    const int modes_per = dim == 3 ? 6 : 3;
    s.rigid_modes = modes_per * static_cast<int>(rigid_bdr->size());

    // The uneliminated operators (fresh assemblies; the eliminated
    // ones above were modified in place).
    IsotropicElasticTensorCoefficient C_full(dim, zero, m);
    s.a_full_form = MakeBilinearForm(fes_u);
    s.a_full_form->AddDomainIntegrator(new ElasticTensorIntegrator(C_full));
    s.a_full_form->Assemble();
    Array<int> none;
    s.a_full_form->FormSystemMatrix(none, s.A_full);
    s.g_full_form = MakeMixedBilinearForm(fes_p, fes_u);
    s.g_full_form->AddDomainIntegrator(new DomainDivVectorScalarIntegrator());
    s.g_full_form->Assemble();
    s.g_full_form->FormRectangularSystemMatrix(none, none, s.G_full);

    std::vector<Vector> AR_full;
    for (const auto& group : *rigid_bdr) {
      // The other clamped boundaries: everything clamped minus this
      // group (disjointness assumed).
      Array<int> others(clamp);
      for (int i = 0; i < std::min(others.Size(), group.Size()); i++) {
        if (group[i]) {
          others[i] = 0;
        }
      }
      Array<int> other_tdofs;
      fes_u.GetEssentialTrueDofs(others, other_tdofs);

      for (int mode = 0; mode < modes_per; mode++) {
        VectorFunctionCoefficient rigid(dim, RigidMode(dim, mode));
        auto field = detail::MakeGridFunction(&fes_u);
        field->ProjectCoefficient(rigid);
        Vector Rk(fes_u.GetTrueVSize());
        field->GetTrueDofs(Rk);
        Rk.SetSubVector(other_tdofs, 0.0);
        field->SetFromTrueDofs(Rk);

        Vector ARk(Rk.Size());
        s.A_full.Ptr()->Mult(Rk, ARk);
        BlockVector col(s.offsets);
        col.GetBlock(0) = ARk;
        col.GetBlock(0).SetSubVector(s.ess_tdofs, 0.0);
        s.G_full.Ptr()->MultTranspose(Rk, col.GetBlock(1));
        col.GetBlock(1) *= -1.0;

        s.rigid_fields.push_back(std::move(field));
        s.R.push_back(std::move(Rk));
        AR_full.push_back(std::move(ARk));
        s.B_col.push_back(col);
      }
    }

    // The base solves of the columns, and the reduced system.
    s.Y.resize(s.rigid_modes);
    for (int k = 0; k < s.rigid_modes; k++) {
      s.Y[k].SetSize(s.offsets.Last());
      s.Y[k] = 0.0;
      s.solver->Mult(s.B_col[k], s.Y[k]);
      MFEM_VERIFY(s.minres->GetConverged(),
                  "StokesSaddleSolver: a border base solve did not "
                  "converge.");
    }
    s.reduced.SetSize(s.rigid_modes);
    for (int k = 0; k < s.rigid_modes; k++) {
      for (int l = 0; l < s.rigid_modes; l++) {
        s.reduced(k, l) = s.GlobalDot(s.R[k], AR_full[l]) -
                          s.GlobalDot(s.B_col[k], s.Y[l]);
      }
    }
    s.reduced_inv.Factor(s.reduced);
  }
}

StokesSaddleSolver::~StokesSaddleSolver() = default;

int StokesSaddleSolver::RigidModes() const { return impl_->rigid_modes; }

const GridFunction& StokesSaddleSolver::RigidField(int k) const {
  return *impl_->rigid_fields.at(k);
}

real_t StokesSaddleSolver::Energy(const GridFunction& u) const {
  const Impl& s = *impl_;
  MFEM_VERIFY(s.A_full.Ptr(),
              "StokesSaddleSolver::Energy: the uneliminated operator "
              "exists only with rigid boundary groups.");
  Vector U(s.fes_u->GetTrueVSize()), AU(U.Size());
  u.GetTrueDofs(U);
  s.A_full.Ptr()->Mult(U, AU);
  return 0.5 * s.GlobalDot(U, AU);
}

int StokesSaddleSolver::Solve(const Vector& F, GridFunction& u,
                              GridFunction& p, const Vector& rigid_rhs,
                              Vector* rigid_coefficients) const {
  const Impl& s = *impl_;
  BlockVector B(s.offsets);
  B.GetBlock(0) = F;
  B.GetBlock(0) *= -1.0;
  B.GetBlock(0).SetSubVector(s.ess_tdofs, 0.0);
  B.GetBlock(1) = 0.0;
  BlockVector X(s.offsets);
  X = 0.0;
  s.solver->Mult(B, X);
  MFEM_VERIFY(s.minres->GetConverged(),
              "StokesSaddleSolver: the Stokes solve did not converge (is "
              "the body force self-equilibrated?)");
  const int iterations = s.minres->GetNumIterations();

  if (s.rigid_modes == 0) {
    u.SetFromTrueDofs(X.GetBlock(0));
    p.SetFromTrueDofs(X.GetBlock(1));
    return iterations;
  }

  // The border elimination: a = (D - B^T Y)^{-1} (b_a - B^T X),
  // X <- X - Y a, with b_a[k] = -R_k . F + rigid_rhs[k] (the enclosed
  // component's own load; see the class notes).
  MFEM_VERIFY(rigid_rhs.Size() == 0 || rigid_rhs.Size() == s.rigid_modes,
              "StokesSaddleSolver: rigid_rhs size mismatch.");
  Vector b_a(s.rigid_modes), a(s.rigid_modes);
  for (int k = 0; k < s.rigid_modes; k++) {
    b_a[k] = -s.GlobalDot(s.R[k], F) +
             (rigid_rhs.Size() ? rigid_rhs[k] : 0.0) -
             s.GlobalDot(s.B_col[k], X);
  }
  s.reduced_inv.Mult(b_a, a);
  for (int k = 0; k < s.rigid_modes; k++) {
    X.Add(-a[k], s.Y[k]);
  }

  // The total velocity: the clamped part plus the rigid combination.
  Vector u_tot(X.GetBlock(0));
  for (int k = 0; k < s.rigid_modes; k++) {
    u_tot.Add(a[k], s.R[k]);
  }
  u.SetFromTrueDofs(u_tot);
  p.SetFromTrueDofs(X.GetBlock(1));
  if (rigid_coefficients) {
    *rigid_coefficients = a;
  }
  return iterations;
}

MinimumDeviatoricEquilibriumStress::MinimumDeviatoricEquilibriumStress(
    FiniteElementSpace& fes_u, FiniteElementSpace& fes_p,
    VectorCoefficient& body_force, Coefficient* mu, Diffeomorphism* map,
    const Array<int>* essential_bdr, const StokesSaddleSolver* solver,
    const Vector& rigid_rhs, Vector* rigid_coefficients,
    Coefficient* interface_pressure, const Array<int>* interface_bdr)
    : MatrixCoefficient(fes_u.GetMesh()->SpaceDimension()),
      fes_u_(&fes_u),
      fes_p_(&fes_p),
      mu_(mu),
      map_(map),
      half_(0.5) {
  MFEM_VERIFY(!(map && interface_pressure),
              "MinimumDeviatoricEquilibriumStress: the interface pressure "
              "is not available in mapped mode.");
  if (!solver) {
    own_solver_ = std::make_unique<StokesSaddleSolver>(fes_u, fes_p, mu, map,
                                                       essential_bdr);
    solver = own_solver_.get();
  }

  LinearForm lf(&fes_u);
  std::unique_ptr<JacobianCoefficient> jac;
  std::unique_ptr<ScalarVectorProductCoefficient> fJ;
  if (map_) {
    jac = std::make_unique<JacobianCoefficient>(*map_);
    fJ = std::make_unique<ScalarVectorProductCoefficient>(*jac, body_force);
    lf.AddDomainIntegrator(new VectorDomainLFIntegrator(*fJ));
  } else {
    lf.AddDomainIntegrator(new VectorDomainLFIntegrator(body_force));
  }
  std::unique_ptr<OutwardSignedCoefficient> signed_p;
  if (interface_pressure) {
    // The pressure load T n = -p n against the OUTWARD normal: the
    // stored boundary orientation is per-element corrected
    // (OutwardSignedCoefficient), and the saddle's weak form carries
    // the boundary term with the sign that makes the (f, v . n)
    // integrator at s = +1 the load for T n = -p n — pinned by the
    // interface-continuity check (an enclosed solid's hydrostatic
    // pressure must EQUAL the loading fluid's at the interface, not
    // its negative).
    signed_p = std::make_unique<OutwardSignedCoefficient>(
        *fes_u.GetMesh(), *interface_pressure);
    auto* integrator = new VectorBoundaryFluxLFIntegrator(*signed_p, 1.0);
    if (interface_bdr) {
      lf.AddBoundaryIntegrator(integrator,
                               const_cast<Array<int>&>(*interface_bdr));
    } else {
      lf.AddBoundaryIntegrator(integrator);
    }
  }
  Vector F;
  AssembleTrueRHS(fes_u, lf, F);

  u_ = detail::MakeGridFunction(&fes_u);
  p_ = detail::MakeGridFunction(&fes_p);
  iterations_ = solver->Solve(F, *u_, *p_, rigid_rhs, rigid_coefficients);
}

void MinimumDeviatoricEquilibriumStress::Eval(DenseMatrix& K,
                                              ElementTransformation& T,
                                              const IntegrationPoint& ip) {
  T.SetIntPoint(&ip);
  u_->GetVectorGradient(T, G_);
  const real_t two_mu = 2.0 * (mu_ ? mu_->Eval(T, ip) : 0.5);
  const real_t p = p_->GetValue(T, ip);
  PullbackStressAt(map_, T, ip, G_, two_mu, p, F_, Fi_, A_, S_, tmp_, K);
}

GridFunctionDiffeomorphism NewHarmonicExtensionMapping(
    Mesh& parent, int order, Diffeomorphism& xi,
    const Array<int>& body_attributes, const Array<int>& buffer_attributes) {
  const int dim = parent.SpaceDimension();
  MFEM_VERIFY(xi.GetVDim() == dim && order >= 1,
              "NewHarmonicExtensionMapping: dimension mismatch or bad order");

  auto fec = std::make_unique<H1_FECollection>(order, dim);
  auto fes = std::make_unique<FiniteElementSpace>(&parent, fec.get(), dim);
  auto h = std::make_unique<GridFunction>(fes.get());
  *h = 0.0;

  // Body: the nodal interpolant of the displacement xi - id.
  Array<int> battrs(body_attributes);
  SubMesh body(SubMesh::CreateFromDomain(parent, battrs));
  FiniteElementSpace fes_body(&body, fec.get(), dim);
  GridFunction h_body(&fes_body);
  VectorFunctionCoefficient pos(dim, [](const Vector& x, Vector& y) { y = x; });
  VectorSumCoefficient disp(xi, pos, 1.0, -1.0);
  h_body.ProjectCoefficient(disp);
  SubMesh::Transfer(h_body, *h);

  // Buffer: harmonic, Dirichlet on every buffer boundary (the body trace
  // on the shared surface, zero on the outer sphere).
  Array<int> vattrs(buffer_attributes);
  SubMesh buffer(SubMesh::CreateFromDomain(parent, vattrs));
  FiniteElementSpace fes_buffer(&buffer, fec.get(), dim);
  GridFunction h_buffer(&fes_buffer);
  h_buffer = 0.0;
  SubMesh::Transfer(*h, h_buffer);

  Array<int> ess_bdr(buffer.bdr_attributes.Size() ? buffer.bdr_attributes.Max()
                                                  : 0);
  ess_bdr = 1;
  Array<int> ess_tdof;
  fes_buffer.GetEssentialTrueDofs(ess_bdr, ess_tdof);
  MFEM_VERIFY(ess_tdof.Size() > 0,
              "NewHarmonicExtensionMapping: the buffer has no boundary dofs");

  ConstantCoefficient one(1.0);
  BilinearForm a(&fes_buffer);
  a.AddDomainIntegrator(new VectorDiffusionIntegrator(one));
  a.Assemble();
  LinearForm b(&fes_buffer);
  b.Assemble();
  OperatorPtr A;
  Vector X, B;
  a.FormLinearSystem(ess_tdof, h_buffer, b, A, X, B);
  GSSmoother prec(*A.As<SparseMatrix>());
  CGSolver cg;
  cg.SetOperator(*A);
  cg.SetPreconditioner(prec);
  cg.SetRelTol(1e-12);
  cg.SetMaxIter(2000);
  cg.SetPrintLevel(0);
  cg.Mult(B, X);
  a.RecoverFEMSolution(X, b, h_buffer);
  SubMesh::Transfer(h_buffer, *h);

  return GridFunctionDiffeomorphism(std::move(fec), std::move(fes),
                                    std::move(h));
}

#ifdef MFEM_USE_MPI
GridFunctionDiffeomorphism NewHarmonicExtensionMapping(
    ParMesh& parent, int order, Diffeomorphism& xi,
    const Array<int>& body_attributes, const Array<int>& buffer_attributes) {
  const int dim = parent.SpaceDimension();
  MFEM_VERIFY(xi.GetVDim() == dim && order >= 1,
              "NewHarmonicExtensionMapping: dimension mismatch or bad order");

  auto fec = std::make_unique<H1_FECollection>(order, dim);
  auto fes = std::make_unique<ParFiniteElementSpace>(&parent, fec.get(), dim);
  auto h = std::make_unique<ParGridFunction>(fes.get());
  *h = 0.0;

  Array<int> battrs(body_attributes);
  ParSubMesh body(ParSubMesh::CreateFromDomain(parent, battrs));
  ParFiniteElementSpace fes_body(&body, fec.get(), dim);
  ParGridFunction h_body(&fes_body);
  VectorFunctionCoefficient pos(dim, [](const Vector& x, Vector& y) { y = x; });
  VectorSumCoefficient disp(xi, pos, 1.0, -1.0);
  h_body.ProjectCoefficient(disp);
  ParSubMesh::Transfer(h_body, *h);

  Array<int> vattrs(buffer_attributes);
  ParSubMesh buffer(ParSubMesh::CreateFromDomain(parent, vattrs));
  ParFiniteElementSpace fes_buffer(&buffer, fec.get(), dim);
  ParGridFunction h_buffer(&fes_buffer);
  h_buffer = 0.0;
  ParSubMesh::Transfer(*h, h_buffer);

  Array<int> ess_bdr(buffer.bdr_attributes.Size() ? buffer.bdr_attributes.Max()
                                                  : 0);
  ess_bdr = 1;
  Array<int> ess_tdof;
  fes_buffer.GetEssentialTrueDofs(ess_bdr, ess_tdof);

  ConstantCoefficient one(1.0);
  ParBilinearForm a(&fes_buffer);
  a.AddDomainIntegrator(new VectorDiffusionIntegrator(one));
  a.Assemble();
  ParLinearForm b(&fes_buffer);
  b.Assemble();
  OperatorPtr A;
  Vector X, B;
  a.FormLinearSystem(ess_tdof, h_buffer, b, A, X, B);
  HypreBoomerAMG prec(*A.As<HypreParMatrix>());
  prec.SetPrintLevel(0);
  prec.SetSystemsOptions(dim);
  CGSolver cg(parent.GetComm());
  cg.SetOperator(*A);
  cg.SetPreconditioner(prec);
  cg.SetRelTol(1e-12);
  cg.SetMaxIter(2000);
  cg.SetPrintLevel(0);
  cg.Mult(B, X);
  a.RecoverFEMSolution(X, b, h_buffer);
  ParSubMesh::Transfer(h_buffer, *h);

  return GridFunctionDiffeomorphism(std::move(fec), std::move(fes),
                                    std::move(h));
}
#endif

}  // namespace AdGIA
