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

// The whole-mesh Poisson operator with the DtN closure, kept assembled
// so that the potential and the adjoint solve share it.
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
    prec_ = MakePreconditioner(As_);

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

DensityFeasibility::DensityFeasibility(
    FiniteElementSpace& fes_phi, int dtn_degree, real_t G,
    Coefficient& rho_parent, const Array<int>& stokes_attributes,
    FiniteElementSpace& fes_u, FiniteElementSpace& fes_p,
    Coefficient& rho_stokes, Coefficient* mu,
    const Array<int>* essential_bdr)
    : fes_phi_(&fes_phi), fes_u_(&fes_u), G_(G) {
  MFEM_VERIFY(fes_phi.GetVDim() == 1,
              "DensityFeasibility: a scalar potential space is needed.");
  const real_t four_pi_G = 4.0 * std::numbers::pi * G_;

  // 1. The self-consistent potential on the parent mesh:
  //    (K + DtN) Phi = -4 pi G (rho, v).
  PoissonSolver poisson(fes_phi, dtn_degree);
  {
    auto b = detail::MakeLinearForm(&fes_phi);
    b->AddDomainIntegrator(new DomainLFIntegrator(rho_parent));
    Vector B;
    AssembleTrueRHS(fes_phi, *b, B);
    B *= -four_pi_G;
    phi_ = detail::MakeGridFunction(&fes_phi);
    phi_iterations_ = poisson.Solve(B, *phi_);
  }

  // 2. The potential on the Stokes mesh (an H1 dof transfer), the load
  //    rho grad Phi, and the Stokes saddle.
  h1_sub_ = std::make_unique<H1_FECollection>(
      fes_phi.FEColl()->GetOrder(), fes_u.GetMesh()->Dimension());
  fes_phi_sub_ = detail::MakeFESpace(fes_u, h1_sub_.get());
  phi_sub_ = detail::MakeGridFunction(fes_phi_sub_.get());
  Transfer(*phi_, *phi_sub_);

  GradientGridFunctionCoefficient grad_phi(phi_sub_.get());
  ScalarVectorProductCoefficient body_force(rho_stokes, grad_phi);
  stress_ = std::make_unique<MinimumDeviatoricEquilibriumStress>(
      fes_u, fes_p, body_force, mu, nullptr, essential_bdr);

  // 3. The value function J = -1/2 F . u, with F the load assembled
  //    exactly as the saddle assembled it.
  {
    auto lf = detail::MakeLinearForm(&fes_u);
    lf->AddDomainIntegrator(new VectorDomainLFIntegrator(body_force));
    Vector F, U(fes_u.GetTrueVSize());
    AssembleTrueRHS(fes_u, *lf, F);
    stress_->Auxiliary().GetTrueDofs(U);
    real_t j = -0.5 * (F * U);
#ifdef MFEM_USE_MPI
    if (detail::IsParallel(fes_u)) {
      real_t global = 0.0;
      MPI_Allreduce(&j, &global, 1, MPI_DOUBLE, MPI_SUM,
                    dynamic_cast<ParFiniteElementSpace&>(fes_u).GetComm());
      j = global;
    }
#endif
    value_ = j;
  }

  // 4. The adjoint potential: (K + DtN) w = r, with the source
  //    r(v) = int_S rho u . grad v assembled on the parent against the
  //    Stokes region's attributes (the velocity transferred to a parent
  //    H1 field, exact there by the dof identification).
  {
    auto vec_fec = std::make_unique<H1_FECollection>(
        fes_u.FEColl()->GetOrder(), fes_phi.GetMesh()->Dimension());
    auto fes_u_parent = detail::MakeFESpace(fes_phi, vec_fec.get(),
                                            fes_u.GetVDim());
    auto u_parent = detail::MakeGridFunction(fes_u_parent.get());
    *u_parent = 0.0;
    Transfer(stress_->Auxiliary(), *u_parent);
    VectorGridFunctionCoefficient u_coeff(u_parent.get());
    ScalarVectorProductCoefficient rho_u(rho_parent, u_coeff);

    Array<int> marker(fes_phi.GetMesh()->attributes.Max());
    marker = 0;
    for (int i = 0; i < stokes_attributes.Size(); i++) {
      marker[stokes_attributes[i] - 1] = 1;
    }
    auto r = detail::MakeLinearForm(&fes_phi);
    r->AddDomainIntegrator(new DomainLFGradIntegrator(rho_u), marker);
    Vector R;
    AssembleTrueRHS(fes_phi, *r, R);
    w_ = detail::MakeGridFunction(&fes_phi);
    w_iterations_ = poisson.Solve(R, *w_);
  }

  // 5. The adjoint potential on the Stokes mesh, for the derivative.
  w_sub_ = detail::MakeGridFunction(fes_phi_sub_.get());
  Transfer(*w_, *w_sub_);
}

DensityFeasibility::~DensityFeasibility() = default;

const GridFunction& DensityFeasibility::Velocity() const {
  return stress_->Auxiliary();
}

void DensityFeasibility::Derivative(FiniteElementSpace& fes_rho,
                                    Vector& dual) const {
  MFEM_VERIFY(fes_rho.GetMesh() == fes_u_->GetMesh(),
              "DensityFeasibility: the control space must live on the "
              "Stokes mesh.");
  const real_t four_pi_G = 4.0 * std::numbers::pi * G_;

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

}  // namespace AdGIA
