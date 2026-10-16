#include "AdGIA/fault_problem.hpp"

#include "AdGIA/detail/fem_factory.hpp"
#include "AdGIA/lininteg.hpp"

using namespace mfem;

namespace AdGIA {

namespace {

// u_w + chi_HW s_hat at a point of an element; the attribute of the
// element transformation selects the side.
class PhysicalDisplacementCoefficient : public VectorCoefficient {
 public:
  PhysicalDisplacementCoefficient(const GridFunction& u,
                                  const GridFunction& slip,
                                  const Array<int>& hanging_wall)
      : VectorCoefficient(u.FESpace()->GetVDim()),
        u_(u),
        slip_(slip),
        hanging_wall_(hanging_wall) {}

  void Eval(Vector& V, ElementTransformation& T,
            const IntegrationPoint& ip) override {
    MFEM_ASSERT(T.ElementType == ElementTransformation::ELEMENT,
                "PhysicalDisplacement: element transformations only.");
    u_.GetVectorValue(T, ip, V);
    if (hanging_wall_[T.Attribute - 1]) {
      slip_.GetVectorValue(T, ip, w_);
      V += w_;
    }
  }

  using VectorCoefficient::Eval;

 private:
  const GridFunction& u_;
  const GridFunction& slip_;
  const Array<int>& hanging_wall_;
  Vector w_;
};

}  // namespace

LinearQuasiStaticClampedFaultProblem::LinearQuasiStaticClampedFaultProblem(
    FiniteElementSpace* fes, const AdGIA::Rheology& rheology,
    const Array<int>& ess_bdr, VectorCoefficient& traction,
    const Array<int>& traction_marker, const Array<int>& hanging_wall_marker,
    const Array<int>& fault_marker, const Array<int>& fault_edge_marker)
    : LinearQuasiStaticClampedProblem(fes, rheology, ess_bdr, traction,
                                      traction_marker),
      hanging_wall_(hanging_wall_marker),
      fault_(fault_marker),
      edge_(fault_edge_marker) {
  const auto& mesh = *fes_->GetMesh();
  MFEM_VERIFY(hanging_wall_.Size() == mesh.attributes.Max(),
              "LinearQuasiStaticClampedFaultProblem: the hanging-wall marker "
              "must be sized to attributes.Max().");
  MFEM_VERIFY(fault_.Size() == mesh.bdr_attributes.Max() &&
                  edge_.Size() == mesh.bdr_attributes.Max(),
              "LinearQuasiStaticClampedFaultProblem: the fault and edge "
              "markers must be sized to bdr_attributes.Max().");
  for (int a = 0; a < fault_.Size(); a++) {
    MFEM_VERIFY(!(fault_[a] && edge_[a]),
                "LinearQuasiStaticClampedFaultProblem: boundary attribute "
                    << a + 1 << " is marked both fault and fault edge.");
  }

  // The slip-carrying dofs: the fault's minus the edge's, as a vdof mask
  // and as true dofs. Both queries are synchronised across ranks, so a
  // rank holding a fault dof without the fault face marks it too.
  auto fault_vdofs = Array<int>();
  auto edge_vdofs = Array<int>();
  fes_->GetEssentialVDofs(fault_, fault_vdofs);
  fes_->GetEssentialVDofs(edge_, edge_vdofs);
  slip_vdof_mask_.SetSize(fault_vdofs.Size());
  for (int i = 0; i < fault_vdofs.Size(); i++) {
    slip_vdof_mask_[i] = fault_vdofs[i] != 0 && edge_vdofs[i] == 0;
  }
  auto fault_tdofs = Array<int>();
  auto edge_tdofs = Array<int>();
  fes_->GetEssentialTrueDofs(fault_, fault_tdofs);
  fes_->GetEssentialTrueDofs(edge_, edge_tdofs);
  auto carries = Array<int>(fes_->GetTrueVSize());
  carries = 0;
  for (auto i : fault_tdofs) {
    carries[i] = 1;
  }
  for (auto i : edge_tdofs) {
    carries[i] = 0;
  }
  for (int i = 0; i < carries.Size(); i++) {
    if (carries[i]) {
      slip_tdofs_.Append(i);
    }
  }

  slip_ = detail::MakeGridFunction(fes_);
  *slip_ = 0.0;
  b_->AddDomainIntegrator(
      new DomainLFFormActionIntegrator(*integrators_, *slip_, -1.0),
      hanging_wall_);
  physical_ = std::make_unique<PhysicalDisplacementCoefficient>(*u_, *slip_,
                                                                hanging_wall_);
}

void LinearQuasiStaticClampedFaultProblem::MaskSlip() {
  for (int i = 0; i < slip_->Size(); i++) {
    if (!slip_vdof_mask_[i]) {
      (*slip_)(i) = 0.0;
    }
  }
}

void LinearQuasiStaticClampedFaultProblem::SetSlip(VectorCoefficient& slip) {
  *slip_ = 0.0;
  slip_->ProjectBdrCoefficient(slip, fault_);
  MaskSlip();
}

void LinearQuasiStaticClampedFaultProblem::SetSlipTrueDofs(
    const Vector& slip_true) {
  MFEM_VERIFY(slip_true.Size() == fes_->GetTrueVSize(),
              "SetSlipTrueDofs: expected a true-dof vector of the "
              "displacement space.");
  slip_->SetFromTrueDofs(slip_true);
  MaskSlip();
}

void LinearQuasiStaticClampedFaultProblem::ClearSlip() { *slip_ = 0.0; }

void LinearQuasiStaticClampedFaultProblem::FaultReaction(Vector& r_true) {
  if (!reaction_form_) {
    reaction_field_ = detail::MakeGridFunction(fes_);
    reaction_form_ = detail::MakeLinearForm(fes_);
    reaction_form_->AddDomainIntegrator(
        new DomainLFFormActionIntegrator(*integrators_, *reaction_field_),
        hanging_wall_);
  }
  add(*u_, *slip_, *reaction_field_);
  reaction_form_->Assemble();
  auto full = Vector(fes_->GetTrueVSize());
  if (const auto* P = fes_->GetProlongationMatrix()) {
    P->MultTranspose(*reaction_form_, full);
  } else {
    full = *reaction_form_;
  }
  r_true.SetSize(full.Size());
  r_true = 0.0;
  for (auto i : slip_tdofs_) {
    r_true(i) = full(i);
  }
}

void LinearQuasiStaticClampedFaultProblem::RegisterFields(DataCollection& dc) {
  LinearQuasiStaticClampedProblem::RegisterFields(dc);
  dc.RegisterField("slip", slip_.get());
}

FaultToSurfaceMap::FaultToSurfaceMap(
    LinearQuasiStaticClampedFaultProblem& problem,
    const FiniteElementSpace& fault_fes, const FiniteElementSpace& surface_fes)
    : Operator(surface_fes.GetTrueVSize(), fault_fes.GetTrueVSize()),
      problem_(&problem),
      fault_injection_(fault_fes, problem.DisplacementSpace()),
      surface_injection_(surface_fes, problem.DisplacementSpace()) {
#ifdef MFEM_USE_MPI
  if (problem.IsParallel()) {
    pi_fault_ = fault_injection_.NewTrueDofMatrix();
    pi_surface_ = surface_injection_.NewTrueDofMatrix();
    return;
  }
#endif
  pi_fault_ = fault_injection_.NewSparseMatrix();
  pi_surface_ = surface_injection_.NewSparseMatrix();
}

void FaultToSurfaceMap::Mult(const Vector& slip, Vector& u) const {
  const auto& fes = problem_->DisplacementSpace();
  parent_true_.SetSize(fes.GetTrueVSize());
  pi_fault_->Mult(slip, parent_true_);
  problem_->SetSlipTrueDofs(parent_true_);
  problem_->AssembleForce(problem_->Time());
  converged_ = problem_->Solve();
  problem_->Displacement().GetTrueDofs(parent_true_);
  u.SetSize(Height());
  pi_surface_->MultTranspose(parent_true_, u);
}

void FaultToSurfaceMap::MultTranspose(const Vector& w, Vector& r) const {
  const auto& fes = problem_->DisplacementSpace();
  parent_true_.SetSize(fes.GetTrueVSize());
  pi_surface_->Mult(w, parent_true_);
  // The surface load as an L-vector dual whose prolongation transpose is
  // the true-dof load: each value on the dof's owner only.
  parent_l_.SetSize(fes.GetVSize());
  if (const auto* R = fes.GetRestrictionMatrix()) {
    R->MultTranspose(parent_true_, parent_l_);
  } else {
    parent_l_ = parent_true_;
  }
  problem_->ClearSlip();
  problem_->AssembleForce(problem_->Time());
  problem_->AddForce(parent_l_);
  converged_ = problem_->Solve();
  problem_->FaultReaction(parent_true_);
  r.SetSize(Width());
  pi_fault_->MultTranspose(parent_true_, r);
  r.Neg();
}

}  // namespace AdGIA
