#include "AdGIA/equilibrium_figures.hpp"

#include <memory>
#include <numbers>

#include "AdGIA/detail/fem_factory.hpp"
#include "AdGIA/poisson.hpp"

namespace AdGIA {

using namespace mfem;

namespace {

// True-dof assembly of a linear form: B = P^T L (the parallel
// reduction; the identity in serial).
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

// Transfer a parent GridFunction to one on a SubMesh of the same parent
// (or back: the direction is read off the meshes).
void Transfer(const GridFunction& src, GridFunction& dst) {
#ifdef MFEM_USE_MPI
  if (dynamic_cast<ParSubMesh*>(src.FESpace()->GetMesh()) ||
      dynamic_cast<ParSubMesh*>(dst.FESpace()->GetMesh())) {
    ParSubMesh::Transfer(static_cast<const ParGridFunction&>(src),
                         static_cast<ParGridFunction&>(dst));
    return;
  }
#endif
  SubMesh::Transfer(src, dst);
}

real_t GlobalSum(FiniteElementSpace& fes, real_t v) {
#ifdef MFEM_USE_MPI
  if (auto* pfes = dynamic_cast<ParFiniteElementSpace*>(&fes)) {
    real_t global = 0.0;
    MPI_Allreduce(&v, &global, 1, MPI_DOUBLE, MPI_SUM, pfes->GetComm());
    return global;
  }
#endif
  return v;
}

// The whole-mesh Poisson operator with the DtN closure, kept assembled
// so that the potential and the adjoint solves share it.
class PoissonSolver {
 public:
  PoissonSolver(FiniteElementSpace& fes, int dtn_degree) : fes_(&fes) {
    a_ = detail::MakeBilinearForm(&fes);
    a_->AddDomainIntegrator(new DiffusionIntegrator());
    a_->Assemble();
    Array<int> empty;
    a_->FormSystemMatrix(empty, A_);

    // The mass shift makes the preconditioner positive definite
    // (examples/poisson_dtn.cpp).
    eps_ = std::make_unique<ConstantCoefficient>(0.01);
    as_ = detail::MakeBilinearForm(&fes);
    as_->AddDomainIntegrator(new DiffusionIntegrator());
    as_->AddDomainIntegrator(new MassIntegrator(*eps_));
    as_->Assemble();
    as_->FormSystemMatrix(empty, As_);
    prec_ = detail::MakePreconditioner(As_);

#ifdef MFEM_USE_MPI
    if (auto* pfes = dynamic_cast<ParFiniteElementSpace*>(&fes)) {
      dtn_ = std::make_unique<PoissonDtNOperator>(pfes->GetComm(), pfes,
                                                  dtn_degree);
      dtn_->Assemble();
      rap_ = std::make_unique<RAPOperator>(dtn_->RAP());
      op_ = std::make_unique<SumOperator>(A_.Ptr(), 1.0, rap_.get(), 1.0,
                                          false, false);
      cg_ = std::make_unique<CGSolver>(pfes->GetComm());
    }
#endif
    if (!op_) {
      dtn_ = std::make_unique<PoissonDtNOperator>(&fes, dtn_degree);
      dtn_->Assemble();
      op_ = std::make_unique<SumOperator>(A_.Ptr(), 1.0, dtn_.get(), 1.0,
                                          false, false);
      cg_ = std::make_unique<CGSolver>();
    }
    cg_->SetOperator(*op_);
    cg_->SetPreconditioner(*prec_);
    cg_->SetRelTol(1e-12);
    cg_->SetAbsTol(0.0);
    cg_->SetMaxIter(10000);
    cg_->SetPrintLevel(0);
  }

  // Solves (K + DtN) x = B for the true-dof vector B; returns the
  // iteration count.
  int Solve(const Vector& B, GridFunction& x) const {
    Vector X(fes_->GetTrueVSize());
    X = 0.0;
    cg_->Mult(B, X);
    MFEM_VERIFY(cg_->GetConverged(),
                "DensityFeasibility: the Poisson solve did not converge.");
    x.SetFromTrueDofs(X);
    return cg_->GetNumIterations();
  }

 private:
  FiniteElementSpace* fes_;
  std::unique_ptr<BilinearForm> a_, as_;
  OperatorHandle A_, As_;
  std::unique_ptr<ConstantCoefficient> eps_;
  std::unique_ptr<Solver> prec_;
  std::unique_ptr<PoissonDtNOperator> dtn_;
#ifdef MFEM_USE_MPI
  std::unique_ptr<RAPOperator> rap_;
#endif
  std::unique_ptr<SumOperator> op_;
  std::unique_ptr<CGSolver> cg_;
};

}  // namespace

// The persistent, density-independent machinery.
struct DensityFeasibilityProblem::Impl {
  FiniteElementSpace *fes_phi, *fes_u, *fes_p;
  Coefficient* mu;
  const Array<int>* essential_bdr;
  real_t G;
  Array<int> stokes_marker;

  std::unique_ptr<PoissonSolver> poisson;
  std::unique_ptr<StokesSaddleSolver> saddle;

  // The potential's and the velocity's companion spaces.
  std::unique_ptr<H1_FECollection> h1_sub, h1_u_parent;
  std::unique_ptr<FiniteElementSpace> fes_phi_sub, fes_u_parent;

  // The enclosed solid components: the saddle's rigid groups, each
  // component's parent-side region marker, modes per group, and the
  // all-ones true-dof vector of the potential space (for the load
  // integrals).
  const std::vector<RigidComponent>* rigid = nullptr;
  std::vector<Array<int>> rigid_markers;      // for the saddle
  std::vector<Array<int>> component_markers;  // parent regions
  int modes_per = 0;
  Vector ones_phi;

  // int_{marked region} coeff, as the marked dual dotted with ones.
  real_t IntegrateOver(const Array<int>& marker, Coefficient& coeff) const {
    auto lf = detail::MakeLinearForm(fes_phi);
    Array<int>& m = const_cast<Array<int>&>(marker);
    lf->AddDomainIntegrator(new DomainLFIntegrator(coeff), m);
    Vector dual;
    AssembleTrueRHS(*fes_phi, *lf, dual);
    real_t v = dual * ones_phi;
#ifdef MFEM_USE_MPI
    if (auto* pfes = dynamic_cast<ParFiniteElementSpace*>(fes_phi)) {
      real_t global = 0.0;
      MPI_Allreduce(&v, &global, 1, MPI_DOUBLE, MPI_SUM, pfes->GetComm());
      v = global;
    }
#endif
    return v;
  }

  // The border loads W_k = int_{component} rho grad(Phi) . r_k.
  Vector ComponentLoads(Coefficient& rho_parent,
                        const GridFunction& phi) const {
    const int dim = fes_phi->GetMesh()->Dimension();
    Vector W(modes_per * static_cast<int>(rigid->size()));
    GradientGridFunctionCoefficient grad_phi(
        const_cast<GridFunction*>(&phi));
    int k = 0;
    for (const auto& marker : component_markers) {
      for (int mode = 0; mode < modes_per; mode++, k++) {
        VectorFunctionCoefficient r(dim, RigidMode(dim, mode));
        InnerProductCoefficient g_dot_r(grad_phi, r);
        ProductCoefficient integrand(rho_parent, g_dot_r);
        W[k] = IntegrateOver(marker, integrand);
      }
    }
    return W;
  }
};

DensityFeasibilityProblem::DensityFeasibilityProblem(
    FiniteElementSpace& fes_phi, int dtn_degree, real_t G,
    const Array<int>& stokes_attributes, FiniteElementSpace& fes_u,
    FiniteElementSpace& fes_p, Coefficient* mu,
    const Array<int>* essential_bdr,
    const std::vector<RigidComponent>* rigid)
    : impl_(std::make_unique<Impl>()) {
  MFEM_VERIFY(fes_phi.GetVDim() == 1,
              "DensityFeasibility: a scalar potential space is needed.");
  Impl& s = *impl_;
  s.fes_phi = &fes_phi;
  s.fes_u = &fes_u;
  s.fes_p = &fes_p;
  s.mu = mu;
  s.essential_bdr = essential_bdr;
  s.G = G;

  s.stokes_marker.SetSize(fes_phi.GetMesh()->attributes.Max());
  s.stokes_marker = 0;
  for (int i = 0; i < stokes_attributes.Size(); i++) {
    s.stokes_marker[stokes_attributes[i] - 1] = 1;
  }

  s.poisson = std::make_unique<PoissonSolver>(fes_phi, dtn_degree);
  if (rigid && !rigid->empty()) {
    s.rigid = rigid;
    s.modes_per = fes_u.GetMesh()->Dimension() == 3 ? 6 : 3;
    for (const auto& component : *rigid) {
      s.rigid_markers.push_back(component.fluid_bdr_marker);
      Array<int> marker(fes_phi.GetMesh()->attributes.Max());
      marker = 0;
      for (int i = 0; i < component.parent_attributes.Size(); i++) {
        marker[component.parent_attributes[i] - 1] = 1;
      }
      s.component_markers.push_back(marker);
    }
    auto ones = detail::MakeGridFunction(&fes_phi);
    *ones = 1.0;
    s.ones_phi.SetSize(fes_phi.GetTrueVSize());
    ones->GetTrueDofs(s.ones_phi);
  }
  s.saddle = std::make_unique<StokesSaddleSolver>(
      fes_u, fes_p, mu, nullptr, essential_bdr,
      s.rigid ? &s.rigid_markers : nullptr);

  s.h1_sub = std::make_unique<H1_FECollection>(
      fes_phi.FEColl()->GetOrder(), fes_u.GetMesh()->Dimension());
  s.fes_phi_sub = detail::MakeFESpace(fes_u, s.h1_sub.get());
  s.h1_u_parent = std::make_unique<H1_FECollection>(
      fes_u.FEColl()->GetOrder(), fes_phi.GetMesh()->Dimension());
  s.fes_u_parent =
      detail::MakeFESpace(fes_phi, s.h1_u_parent.get(), fes_u.GetVDim());
}

DensityFeasibilityProblem::~DensityFeasibilityProblem() = default;

std::unique_ptr<DensityFeasibility> DensityFeasibilityProblem::Evaluate(
    Coefficient& rho_parent, Coefficient& rho_stokes) const {
  return std::unique_ptr<DensityFeasibility>(
      new DensityFeasibility(*this, rho_parent, rho_stokes));
}

DensityFeasibility::DensityFeasibility(
    const DensityFeasibilityProblem& problem, Coefficient& rho_parent,
    Coefficient& rho_stokes)
    : problem_(&problem) {
  Solve(rho_parent, rho_stokes);
}

DensityFeasibility::DensityFeasibility(
    FiniteElementSpace& fes_phi, int dtn_degree, real_t G,
    Coefficient& rho_parent, const Array<int>& stokes_attributes,
    FiniteElementSpace& fes_u, FiniteElementSpace& fes_p,
    Coefficient& rho_stokes, Coefficient* mu,
    const Array<int>* essential_bdr) {
  own_problem_ = std::make_unique<DensityFeasibilityProblem>(
      fes_phi, dtn_degree, G, stokes_attributes, fes_u, fes_p, mu,
      essential_bdr);
  problem_ = own_problem_.get();
  Solve(rho_parent, rho_stokes);
}

void DensityFeasibility::Solve(Coefficient& rho_parent,
                               Coefficient& rho_stokes) {
  const DensityFeasibilityProblem::Impl& s = *problem_->impl_;
  const real_t four_pi_G = 4.0 * std::numbers::pi * s.G;

  // 1. The self-consistent potential on the parent mesh:
  //    (K + DtN) Phi = -4 pi G (rho, v), and the energy
  //    E = 1/2 int rho Phi from the unscaled density dual.
  {
    auto b = detail::MakeLinearForm(s.fes_phi);
    b->AddDomainIntegrator(new DomainLFIntegrator(rho_parent));
    Vector B;
    AssembleTrueRHS(*s.fes_phi, *b, B);
    Vector rho_dual(B);
    B *= -four_pi_G;
    phi_ = detail::MakeGridFunction(s.fes_phi);
    phi_iterations_ = s.poisson->Solve(B, *phi_);

    Vector Phi(s.fes_phi->GetTrueVSize());
    phi_->GetTrueDofs(Phi);
    energy_ = GlobalSum(*s.fes_phi, 0.5 * (rho_dual * Phi));
  }

  // 2. The potential on the Stokes mesh (an H1 dof transfer), the load
  //    rho grad Phi, and the Stokes saddle (the problem's assembled
  //    solver, through the stress generator).
  phi_sub_ = detail::MakeGridFunction(s.fes_phi_sub.get());
  Transfer(*phi_, *phi_sub_);

  GradientGridFunctionCoefficient grad_phi(phi_sub_.get());
  ScalarVectorProductCoefficient body_force(rho_stokes, grad_phi);
  Vector W;
  if (s.rigid) {
    // The enclosed components' loads on the border (+W: the sign the
    // force-balance derivation fixes; doc/equilibrium_figures.tex §2).
    W = s.ComponentLoads(rho_parent, *phi_);
  }
  stress_ = std::make_unique<MinimumDeviatoricEquilibriumStress>(
      *s.fes_u, *s.fes_p, body_force, s.mu, nullptr, s.essential_bdr,
      s.saddle.get(), W, s.rigid ? &rigid_a_ : nullptr);

  // 3. The value function: J = -1/2 F . u without the border; with it,
  //    the convention-free energy of the total multiplier.
  if (s.rigid) {
    value_ = s.saddle->Energy(stress_->Auxiliary());
  } else {
    auto lf = detail::MakeLinearForm(s.fes_u);
    lf->AddDomainIntegrator(new VectorDomainLFIntegrator(body_force));
    Vector F, U(s.fes_u->GetTrueVSize());
    AssembleTrueRHS(*s.fes_u, *lf, F);
    stress_->Auxiliary().GetTrueDofs(U);
    value_ = GlobalSum(*s.fes_u, -0.5 * (F * U));
  }

  // 4. The adjoint potential: (K + DtN) w = r, with the source
  //    r(v) = int_S rho u . grad v assembled on the parent against the
  //    Stokes region's attributes (the velocity transferred to a parent
  //    H1 field, exact there by the dof identification).
  {
    u_parent_ = detail::MakeGridFunction(s.fes_u_parent.get());
    *u_parent_ = 0.0;
    Transfer(stress_->Auxiliary(), *u_parent_);
    VectorGridFunctionCoefficient u_coeff(u_parent_.get());
    ScalarVectorProductCoefficient rho_u(rho_parent, u_coeff);

    auto r = detail::MakeLinearForm(s.fes_phi);
    Array<int>& marker = const_cast<Array<int>&>(s.stokes_marker);
    r->AddDomainIntegrator(new DomainLFGradIntegrator(rho_u), marker);
    // With rigid components the adjoint "velocity" extends into each
    // core as its rigid multiplier motion — the border load's own
    // Phi-chain, derived with the envelope theorem like the rest.
    std::vector<std::unique_ptr<VectorFunctionCoefficient>> core_motion;
    std::vector<std::unique_ptr<ScalarVectorProductCoefficient>> core_rho_u;
    if (s.rigid) {
      const int dim = s.fes_phi->GetMesh()->Dimension();
      for (std::size_t g = 0; g < s.component_markers.size(); g++) {
        std::vector<real_t> a(s.modes_per);
        for (int mode = 0; mode < s.modes_per; mode++) {
          // MINUS: the border multiplier pairs with its constraint with
          // the opposite sign to the field multiplier's pattern (the FD
          // harness pinned it).
          a[mode] = -rigid_a_[static_cast<int>(g) * s.modes_per + mode];
        }
        core_motion.push_back(std::make_unique<VectorFunctionCoefficient>(
            dim, [dim, a](const Vector& x, Vector& v) {
              v.SetSize(dim);
              v = 0.0;
              Vector mode_v(dim);
              for (int mode = 0; mode < static_cast<int>(a.size());
                   mode++) {
                RigidMode(dim, mode)(x, mode_v);
                v.Add(a[mode], mode_v);
              }
            }));
        core_rho_u.push_back(std::make_unique<ScalarVectorProductCoefficient>(
            rho_parent, *core_motion.back()));
        r->AddDomainIntegrator(
            new DomainLFGradIntegrator(*core_rho_u.back()),
            const_cast<Array<int>&>(s.component_markers[g]));
      }
    }
    Vector R;
    AssembleTrueRHS(*s.fes_phi, *r, R);
    w_ = detail::MakeGridFunction(s.fes_phi);
    w_iterations_ = s.poisson->Solve(R, *w_);
  }

  // 5. The adjoint potential on the Stokes mesh, for the derivative.
  w_sub_ = detail::MakeGridFunction(s.fes_phi_sub.get());
  Transfer(*w_, *w_sub_);
}

DensityFeasibility::~DensityFeasibility() = default;

const GridFunction& DensityFeasibility::Velocity() const {
  return stress_->Auxiliary();
}

void DensityFeasibility::Derivative(FiniteElementSpace& fes_rho,
                                    Vector& dual) const {
  const DensityFeasibilityProblem::Impl& s = *problem_->impl_;
  MFEM_VERIFY(fes_rho.GetMesh() == s.fes_u->GetMesh(),
              "DensityFeasibility: the control space must live on the "
              "Stokes mesh.");
  const real_t four_pi_G = 4.0 * std::numbers::pi * s.G;

  // dJ[drho] = int drho ( -u . grad Phi + 4 pi G w ) over the Stokes
  // mesh (doc/equilibrium_figures.tex, the envelope-theorem derivative).
  VectorGridFunctionCoefficient u_coeff(&stress_->Auxiliary());
  GradientGridFunctionCoefficient grad_phi(phi_sub_.get());
  InnerProductCoefficient advective(u_coeff, grad_phi);
  GridFunctionCoefficient w_coeff(w_sub_.get());
  SumCoefficient integrand(advective, w_coeff, -1.0, four_pi_G);

  auto lf = detail::MakeLinearForm(&fes_rho);
  lf->AddDomainIntegrator(new DomainLFIntegrator(integrand));
  AssembleTrueRHS(fes_rho, *lf, dual);
}

void DensityFeasibility::HessianAction(
    Coefficient& rho_parent, Coefficient& rho_stokes,
    Coefficient& drho_parent, Coefficient& drho_stokes,
    FiniteElementSpace& fes_rho, Vector& dual, bool gn_only) const {
  const DensityFeasibilityProblem::Impl& s = *problem_->impl_;
  MFEM_VERIFY(fes_rho.GetMesh() == s.fes_u->GetMesh(),
              "DensityFeasibility: the control space must live on the "
              "Stokes mesh.");
  const real_t four_pi_G = 4.0 * std::numbers::pi * s.G;
  Array<int>& marker = const_cast<Array<int>&>(s.stokes_marker);

  // The potential's sensitivity: (K + DtN) dPhi = -4 pi G (drho, v).
  auto dphi = detail::MakeGridFunction(s.fes_phi);
  {
    auto b = detail::MakeLinearForm(s.fes_phi);
    b->AddDomainIntegrator(new DomainLFIntegrator(drho_parent));
    Vector B;
    AssembleTrueRHS(*s.fes_phi, *b, B);
    B *= -four_pi_G;
    s.poisson->Solve(B, *dphi);
  }
  auto dphi_sub = detail::MakeGridFunction(s.fes_phi_sub.get());
  Transfer(*dphi, *dphi_sub);

  // The state's sensitivity: the saddle with the load derivative
  // dF = (drho grad Phi + rho grad dPhi, v); with rigid components the
  // border loads' own derivative dW_k = int rho grad(dPhi) . r_k rides
  // along (the direction vanishes on the solid, so its direct term
  // does not appear).
  auto du = detail::MakeGridFunction(s.fes_u);
  auto dp = detail::MakeGridFunction(s.fes_p);
  Vector da;
  {
    GradientGridFunctionCoefficient grad_phi(phi_sub_.get());
    GradientGridFunctionCoefficient grad_dphi(dphi_sub.get());
    ScalarVectorProductCoefficient term_a(drho_stokes, grad_phi);
    ScalarVectorProductCoefficient term_b(rho_stokes, grad_dphi);
    VectorSumCoefficient df(term_a, term_b);
    auto lf = detail::MakeLinearForm(s.fes_u);
    lf->AddDomainIntegrator(new VectorDomainLFIntegrator(df));
    Vector F;
    AssembleTrueRHS(*s.fes_u, *lf, F);
    Vector dW;
    if (s.rigid) {
      dW = s.ComponentLoads(rho_parent, *dphi);
    }
    s.saddle->Solve(F, *du, *dp, dW, s.rigid ? &da : nullptr);
  }

  // The Gauss-Newton dual is the gradient assembly with (u, w)
  // replaced by (du, dw): by the saddle identity
  // du2^T A du1 = -du2 . dF(d1). The residual term -u^T d2F[., d]
  // adds -(., u . grad dPhi) and its own adjoint potential; the two
  // adjoint sources are linear, so one Poisson solve carries their sum
  // — a full product costs one saddle and two Poisson solves, the same
  // as a Gauss-Newton one.
  auto du_parent = detail::MakeGridFunction(s.fes_u_parent.get());
  *du_parent = 0.0;
  Transfer(*du, *du_parent);
  auto w_adj = detail::MakeGridFunction(s.fes_phi);
  {
    VectorGridFunctionCoefficient du_coeff(du_parent.get());
    ScalarVectorProductCoefficient rho_du(rho_parent, du_coeff);
    auto r = detail::MakeLinearForm(s.fes_phi);
    r->AddDomainIntegrator(new DomainLFGradIntegrator(rho_du), marker);
    // The residual term's source, int drho u . grad v, merged in.
    VectorGridFunctionCoefficient up_coeff(u_parent_.get());
    ScalarVectorProductCoefficient drho_u(drho_parent, up_coeff);
    if (!gn_only) {
      r->AddDomainIntegrator(new DomainLFGradIntegrator(drho_u), marker);
    }
    // The sensitivity's core extension, as in the gradient's adjoint
    // (the border load's Phi-chain, with the SENSITIVITY coefficients;
    // the border load is linear in the fluid density, so no residual
    // core term exists).
    std::vector<std::unique_ptr<VectorFunctionCoefficient>> core_motion;
    std::vector<std::unique_ptr<ScalarVectorProductCoefficient>> core_rho;
    if (s.rigid) {
      const int dim = s.fes_phi->GetMesh()->Dimension();
      for (std::size_t g = 0; g < s.component_markers.size(); g++) {
        std::vector<real_t> a(s.modes_per);
        for (int mode = 0; mode < s.modes_per; mode++) {
          // MINUS, as in the gradient's core extension.
          a[mode] = -da[static_cast<int>(g) * s.modes_per + mode];
        }
        core_motion.push_back(std::make_unique<VectorFunctionCoefficient>(
            dim, [dim, a](const Vector& x, Vector& v) {
              v.SetSize(dim);
              v = 0.0;
              Vector mode_v(dim);
              for (int mode = 0; mode < static_cast<int>(a.size());
                   mode++) {
                RigidMode(dim, mode)(x, mode_v);
                v.Add(a[mode], mode_v);
              }
            }));
        core_rho.push_back(std::make_unique<ScalarVectorProductCoefficient>(
            rho_parent, *core_motion.back()));
        r->AddDomainIntegrator(
            new DomainLFGradIntegrator(*core_rho.back()),
            const_cast<Array<int>&>(s.component_markers[g]));
      }
    }
    Vector R;
    AssembleTrueRHS(*s.fes_phi, *r, R);
    s.poisson->Solve(R, *w_adj);
  }
  auto w_adj_sub = detail::MakeGridFunction(s.fes_phi_sub.get());
  Transfer(*w_adj, *w_adj_sub);

  VectorGridFunctionCoefficient du_stokes(du.get());
  GradientGridFunctionCoefficient grad_phi(phi_sub_.get());
  InnerProductCoefficient gn_advective(du_stokes, grad_phi);
  GridFunctionCoefficient w_adj_coeff(w_adj_sub.get());
  SumCoefficient gn_integrand(gn_advective, w_adj_coeff, -1.0, four_pi_G);

  auto lf = detail::MakeLinearForm(&fes_rho);
  lf->AddDomainIntegrator(new DomainLFIntegrator(gn_integrand));

  VectorGridFunctionCoefficient u_coeff(&stress_->Auxiliary());
  GradientGridFunctionCoefficient grad_dphi_sub(dphi_sub.get());
  InnerProductCoefficient res_advective(u_coeff, grad_dphi_sub);
  ProductCoefficient res_negated(-1.0, res_advective);
  if (!gn_only) {
    lf->AddDomainIntegrator(new DomainLFIntegrator(res_negated));
  }

  AssembleTrueRHS(fes_rho, *lf, dual);
}

void DensityFeasibility::DerivativeOnParent(FiniteElementSpace& fes,
                                            Vector& dual) const {
  const DensityFeasibilityProblem::Impl& s = *problem_->impl_;
  MFEM_VERIFY(fes.GetMesh() == s.fes_phi->GetMesh(),
              "DensityFeasibility: a space on the parent mesh is needed.");
  const real_t four_pi_G = 4.0 * std::numbers::pi * s.G;

  // The same integrand as Derivative(), in its parent-side fields,
  // under the Stokes region's attribute marker (the transferred
  // velocity equals the Stokes one there).
  VectorGridFunctionCoefficient u_coeff(u_parent_.get());
  GradientGridFunctionCoefficient grad_phi(phi_.get());
  InnerProductCoefficient advective(u_coeff, grad_phi);
  GridFunctionCoefficient w_coeff(w_.get());
  SumCoefficient integrand(advective, w_coeff, -1.0, four_pi_G);

  auto lf = detail::MakeLinearForm(&fes);
  Array<int>& marker = const_cast<Array<int>&>(s.stokes_marker);
  lf->AddDomainIntegrator(new DomainLFIntegrator(integrand), marker);
  AssembleTrueRHS(fes, *lf, dual);
}

}  // namespace AdGIA
