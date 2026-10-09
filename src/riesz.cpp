#include "AdGIA/riesz.hpp"

#include "AdGIA/detail/fem_factory.hpp"

namespace AdGIA {

using namespace mfem;

namespace {

std::unique_ptr<CGSolver> MakeCG(FiniteElementSpace& fes) {
#ifdef MFEM_USE_MPI
  if (auto* pfes = dynamic_cast<ParFiniteElementSpace*>(&fes)) {
    return std::make_unique<CGSolver>(pfes->GetComm());
  }
#endif
  return std::make_unique<CGSolver>();
}

real_t GlobalDot(FiniteElementSpace& fes, const Vector& x, const Vector& y) {
  real_t d = x * y;
#ifdef MFEM_USE_MPI
  if (auto* pfes = dynamic_cast<ParFiniteElementSpace*>(&fes)) {
    real_t global = 0.0;
    MPI_Allreduce(&d, &global, 1, MPI_DOUBLE, MPI_SUM, pfes->GetComm());
    d = global;
  }
#endif
  return d;
}

}  // namespace

real_t RieszMap::Pair(const Vector& dual, const Vector& x) const {
  return GlobalDot(*fes_, dual, x);
}

namespace {

// The mass and stiffness integrators of a space, scalar or (component-
// wise) vector by its vdim.
BilinearFormIntegrator* MassOf(FiniteElementSpace& fes, Coefficient* q) {
  if (fes.GetVDim() > 1) {
    return q ? new VectorMassIntegrator(*q) : new VectorMassIntegrator();
  }
  return q ? new MassIntegrator(*q) : new MassIntegrator();
}

BilinearFormIntegrator* DiffusionOf(FiniteElementSpace& fes,
                                    Coefficient& q) {
  if (fes.GetVDim() > 1) {
    return new VectorDiffusionIntegrator(q);
  }
  return new DiffusionIntegrator(q);
}

// On a vector space the parallel AMG needs the system structure (the
// spaces here are byNODES-ordered).
void SetSystemsOptions(Solver& prec, FiniteElementSpace& fes) {
#ifdef MFEM_USE_MPI
  if (fes.GetVDim() > 1) {
    if (auto* amg = dynamic_cast<HypreBoomerAMG*>(&prec)) {
      amg->SetSystemsOptions(fes.GetVDim(), true);
    }
  }
#endif
  (void)prec;
  (void)fes;
}

}  // namespace

L2RieszMap::L2RieszMap(FiniteElementSpace& fes) : RieszMap(fes) {
  m_ = detail::MakeBilinearForm(&fes);
  m_->AddDomainIntegrator(MassOf(fes, nullptr));
  m_->Assemble();
  Array<int> empty;
  m_->FormSystemMatrix(empty, M_);
  prec_ = detail::MakePreconditioner(M_);
  SetSystemsOptions(*prec_, fes);
  cg_ = MakeCG(fes);
  cg_->SetOperator(*M_.Ptr());
  cg_->SetPreconditioner(*prec_);
  cg_->SetRelTol(1e-13);
  cg_->SetAbsTol(0.0);
  cg_->SetMaxIter(2000);
  cg_->SetPrintLevel(0);
}

L2RieszMap::~L2RieszMap() = default;

void L2RieszMap::Mult(const Vector& dual, Vector& g) const {
  g.SetSize(dual.Size());
  g = 0.0;
  cg_->Mult(dual, g);
  MFEM_VERIFY(cg_->GetConverged(),
              "L2RieszMap: the mass solve did not converge.");
}

SobolevRieszMap::SobolevRieszMap(FiniteElementSpace& fes, real_t alpha,
                                 real_t beta, int order,
                                 const Array<int>* dirichlet_bdr)
    : RieszMap(fes), order_(order) {
  MFEM_VERIFY(order >= 1, "SobolevRieszMap: the order must be positive.");
  if (dirichlet_bdr) {
    Array<int> marker(*dirichlet_bdr);
    fes.GetEssentialTrueDofs(marker, ess_tdofs_);
  }

  ConstantCoefficient a_coeff(alpha), b_coeff(beta);
  a_ = detail::MakeBilinearForm(&fes);
  a_->AddDomainIntegrator(MassOf(fes, &a_coeff));
  a_->AddDomainIntegrator(DiffusionOf(fes, b_coeff));
  a_->Assemble();
  a_->FormSystemMatrix(ess_tdofs_, A_);

  m_ = detail::MakeBilinearForm(&fes);
  m_->AddDomainIntegrator(MassOf(fes, nullptr));
  m_->Assemble();
  Array<int> empty;
  m_->FormSystemMatrix(empty, M_);

  prec_ = detail::MakePreconditioner(A_);
  SetSystemsOptions(*prec_, fes);
  cg_ = MakeCG(fes);
  cg_->SetOperator(*A_.Ptr());
  cg_->SetPreconditioner(*prec_);
  cg_->SetRelTol(1e-13);
  cg_->SetAbsTol(0.0);
  cg_->SetMaxIter(5000);
  cg_->SetPrintLevel(0);
}

SobolevRieszMap::~SobolevRieszMap() = default;

void SobolevRieszMap::EliminateEssential(Vector& v) const {
  v.SetSubVector(ess_tdofs_, 0.0);
}

void SobolevRieszMap::Mult(const Vector& dual, Vector& g) const {
  // g = (A^-1 M)^{s-1} A^-1 j, every intermediate re-eliminated: the
  // metric space is the Dirichlet-constrained subspace, and a dual's
  // content on the eliminated dofs is annihilated.
  Vector rhs(dual);
  EliminateEssential(rhs);
  g.SetSize(dual.Size());
  Vector x(dual.Size());
  for (int k = 0; k < order_; k++) {
    x = 0.0;
    cg_->Mult(rhs, x);
    MFEM_VERIFY(cg_->GetConverged(),
                "SobolevRieszMap: the Helmholtz solve did not converge.");
    if (k + 1 < order_) {
      M_.Ptr()->Mult(x, rhs);
      EliminateEssential(rhs);
    }
  }
  g = x;
}

}  // namespace AdGIA
