#include "AdGIA/lininteg.hpp"

namespace AdGIA {

namespace {

// The scalar coefficient handed to the DeltaLFIntegrator base: the wrapped
// DeltaCoefficient when M is a MatrixDeltaCoefficient (so that IsDelta(),
// GetDeltaCenter() and the linear form's delta machinery engage), and an
// arbitrary non-delta coefficient otherwise (the base only dynamic_casts
// it; nothing is stored).
mfem::Coefficient& DeltaScalarOf(mfem::MatrixCoefficient& M) {
  static mfem::ConstantCoefficient not_a_delta(0.0);
  if (auto* md = dynamic_cast<MatrixDeltaCoefficient*>(&M)) {
    return md->GetDeltaCoefficient();
  }
  return not_a_delta;
}

}  // namespace

DomainLFDeformationGradientIntegrator::DomainLFDeformationGradientIntegrator(
    mfem::MatrixCoefficient& M, const mfem::IntegrationRule* ir)
    : mfem::DeltaLFIntegrator(DeltaScalarOf(M), ir),
      M_{M},
      delta_M_{dynamic_cast<MatrixDeltaCoefficient*>(&M)} {}

void DomainLFDeformationGradientIntegrator::AssembleDeltaElementVect(
    const mfem::FiniteElement& fe, mfem::ElementTransformation& Trans,
    mfem::Vector& elvect) {
  using namespace mfem;
  MFEM_ASSERT(delta_M_, "coefficient must be a MatrixDeltaCoefficient");

  const auto dof = fe.GetDof();
  const auto space_dim = Trans.GetSpaceDim();
#ifdef MFEM_THREAD_SAFE
  DenseMatrix dshape, m;
  Vector v;
#endif
  dshape.SetSize(dof, space_dim);
  m.SetSize(space_dim, space_dim);
  fe.CalcPhysDShape(Trans, dshape);
  delta_M_->EvalDelta(m, Trans, Trans.GetIntPoint());

  elvect.SetSize(dof * space_dim);
  auto vm = DenseMatrix(elvect.GetData(), dof, space_dim);
  MultABt(dshape, m, vm);
}

void DomainLFDeformationGradientIntegrator::AssembleRHSElementVect(
    const mfem::FiniteElement& el, mfem::ElementTransformation& Trans,
    mfem::Vector& elvect) {
  using namespace mfem;

  auto dof = el.GetDof();
  auto space_dim = Trans.GetSpaceDim();
  MFEM_ASSERT(M_.GetHeight() == space_dim && M_.GetWidth() == space_dim,
              "Width of matrix coefficient must equal spatial dimension");

#ifdef MFEM_THREAD_SAFE
  DenseMatrix dshape, m;
  Vector v;
#endif
  dshape.SetSize(dof, space_dim);
  m.SetSize(space_dim, space_dim);
  elvect.SetSize(dof * space_dim);

  elvect = 0.0;
  v.SetSize(dof * space_dim);
  auto vm = DenseMatrix(v.GetData(), dof, space_dim);

  const auto* ir = GetIntegrationRule(el, Trans);
  if (ir == nullptr) {
    int intorder = 2 * el.GetOrder() + Trans.OrderW();
    ir = &IntRules.Get(el.GetGeomType(), intorder);
  }

  for (int i = 0; i < ir->GetNPoints(); i++) {
    const auto& ip = ir->IntPoint(i);
    Trans.SetIntPoint(&ip);
    auto factor = Trans.Weight() * ip.weight;
    el.CalcPhysDShape(Trans, dshape);
    M_.Eval(m, Trans, ip);
    MultABt(dshape, m, vm);
    elvect.Add(factor, v);
  }
}

}  // namespace AdGIA