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

MinimumDeviatoricEquilibriumStress::MinimumDeviatoricEquilibriumStress(
    FiniteElementSpace& fes_u, FiniteElementSpace& fes_p,
    VectorCoefficient& body_force, Coefficient* mu, Diffeomorphism* map)
    : MatrixCoefficient(fes_u.GetMesh()->SpaceDimension()),
      fes_u_(&fes_u),
      fes_p_(&fes_p),
      mu_(mu),
      map_(map),
      half_(0.5) {
  const int dim = fes_u.GetMesh()->SpaceDimension();
  MFEM_VERIFY(fes_u.GetVDim() == dim && fes_p.GetVDim() == 1 &&
                  fes_u.GetMesh() == fes_p.GetMesh(),
              "MinimumDeviatoricEquilibriumStress: vector velocity and "
              "scalar pressure spaces on one mesh are needed");
  MFEM_VERIFY(
      fes_p.FEColl()->GetOrder() < fes_u.FEColl()->GetOrder(),
      "MinimumDeviatoricEquilibriumStress: the pressure space must sit at "
      "least one polynomial order below the velocity space (Taylor-Hood); "
      "equal-order interpolation violates the inf-sup (LBB) condition and "
      "produces spurious pressure modes.");
  Coefficient& m = mu_ ? *mu_ : half_;

  // AW10 eqs. (73)-(74): the steady incompressible Stokes problem with
  // traction boundary conditions, as the symmetric saddle system
  //   [ A  -G   ] [u]   [-F]
  //   [-G^T  0  ] [p] = [ 0],   A = int 2 mu e(u):e(v),  G = (p, div v),
  // each form pulled back with the relabelling recipe in mapped mode.
  ConstantCoefficient zero(0.0);
  IsotropicElasticTensorCoefficient C_iso(dim, zero, m);
  auto a = MakeBilinearForm(fes_u);
  a->AddDomainIntegrator(map_ ? new ElasticTensorIntegrator(C_iso, *map_)
                              : new ElasticTensorIntegrator(C_iso));
  a->Assemble();
  OperatorHandle A;
  Array<int> empty;
  a->FormSystemMatrix(empty, A);

  ConstantCoefficient one(1.0);
  auto g_form = MakeMixedBilinearForm(fes_p, fes_u);
  g_form->AddDomainIntegrator(map_
                                  ? new DomainDivVectorScalarIntegrator(*map_)
                                  : new DomainDivVectorScalarIntegrator());
  g_form->Assemble();
  OperatorHandle G;
  g_form->FormRectangularSystemMatrix(empty, empty, G);
  TransposeOperator Gt(*G.Ptr());

  Array<int> offsets({0, fes_u.GetTrueVSize(), fes_p.GetTrueVSize()});
  offsets.PartialSum();
  BlockOperator block_op(offsets);
  block_op.SetBlock(0, 0, A.Ptr());
  block_op.SetBlock(0, 1, G.Ptr(), -1.0);
  block_op.SetBlock(1, 0, &Gt, -1.0);

  // Pressure-block preconditioner: the mass matrix (the Stokes Schur
  // complement up to the mu weight).
  auto mp = MakeBilinearForm(fes_p);
  mp->AddDomainIntegrator(new MassIntegrator(one));
  mp->Assemble();
  OperatorHandle Mp;
  mp->FormSystemMatrix(empty, Mp);
  auto prec_u = MakePreconditioner(A);
  auto prec_pr = MakePreconditioner(Mp);
  BlockDiagonalPreconditioner block_prec(offsets);
  block_prec.SetDiagonalBlock(0, prec_u.get());
  block_prec.SetDiagonalBlock(1, prec_pr.get());

  // Null space: the rigid modes of u alone (mapped rotations in mapped
  // mode). With the traction boundary condition the pressure has NO
  // constant ambiguity.
  std::unique_ptr<NullSpaceProjector> projector;
#ifdef MFEM_USE_MPI
  bool parallel = false;
  MPI_Comm comm = CommOf(fes_u, parallel);
  projector = parallel ? std::make_unique<NullSpaceProjector>(comm)
                       : std::make_unique<NullSpaceProjector>();
#else
  projector = std::make_unique<NullSpaceProjector>();
#endif
  {
    auto rigid = MakeGeneratorProjector(fes_u, map_);
    BlockVector n(offsets);
    for (int i = 0; i < rigid->Size(); i++) {
      n.GetBlock(0) = rigid->Basis(i);
      n.GetBlock(1) = 0.0;
      projector->Add(n);
    }
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
  BlockVector B(offsets);
  AssembleTrueRHS(fes_u, lf, B.GetBlock(0));
  B.GetBlock(0) *= -1.0;
  B.GetBlock(1) = 0.0;

  std::unique_ptr<MINRESSolver> minres;
#ifdef MFEM_USE_MPI
  minres = parallel ? std::make_unique<MINRESSolver>(comm)
                    : std::make_unique<MINRESSolver>();
#else
  minres = std::make_unique<MINRESSolver>();
#endif
  ProjectedOperator op(block_op, *projector);
  ProjectedSolver prec_proj(*projector);
  prec_proj.SetSolver(block_prec);
  minres->SetOperator(op);
  minres->SetPreconditioner(prec_proj);
  minres->SetRelTol(1e-11);
  minres->SetAbsTol(0.0);
  minres->SetMaxIter(50000);
  minres->SetPrintLevel(0);
  BlockVector X(offsets);
  X = 0.0;
  ProjectedSolver solver(*projector);
  solver.SetSolver(*minres);
  solver.Mult(B, X);
  MFEM_VERIFY(minres->GetConverged(),
              "MinimumDeviatoricEquilibriumStress: the Stokes solve did "
              "not converge (is the body force self-equilibrated?)");
  iterations_ = minres->GetNumIterations();

  u_ = detail::MakeGridFunction(&fes_u);
  u_->SetFromTrueDofs(X.GetBlock(0));
  p_ = detail::MakeGridFunction(&fes_p);
  p_->SetFromTrueDofs(X.GetBlock(1));
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
