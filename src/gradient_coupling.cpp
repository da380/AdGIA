#include "AdGIA/gradient_coupling.hpp"

#include "AdGIA/detail/pa_kernels.hpp"

namespace AdGIA {

using namespace mfem;

GradientCouplingOperator::GradientCouplingOperator(
    FiniteElementSpace& scalar_fes, FiniteElementSpace& vector_fes,
    Coefficient& c, const IntegrationRule* ir)
    : Operator(vector_fes.GetVSize(), scalar_fes.GetVSize()) {
  Mesh& mesh = *scalar_fes.GetMesh();
  MFEM_VERIFY(vector_fes.GetMesh() == &mesh,
              "GradientCouplingOperator: the two spaces are on one mesh.");
  MFEM_VERIFY(mesh.Dimension() == 3 && mesh.SpaceDimension() == 3 &&
                  scalar_fes.GetVDim() == 1 && vector_fes.GetVDim() == 3,
              "GradientCouplingOperator: a scalar and a vector space in "
              "3-D.");
  Array<Geometry::Type> geoms;
  mesh.GetGeometries(3, geoms);
  MFEM_VERIFY(geoms.Size() <= 1,
              "GradientCouplingOperator: a mesh of one element geometry.");
  ne_ = mesh.GetNE();
  if (ne_ == 0) {
    return;
  }
  const FiniteElement& scalar_fe = *scalar_fes.GetTypicalFE();
  const FiniteElement& vector_fe = *vector_fes.GetTypicalFE();
  if (ir == nullptr) {
    // The rule of mfem's GradientIntegrator.
    ElementTransformation& T0 = *mesh.GetTypicalElementTransformation();
    ir = &IntRules.Get(scalar_fe.GetGeomType(),
                       T0.OrderGrad(&scalar_fe) + vector_fe.GetOrder() +
                           T0.OrderJ());
  }
  const bool tensor = UsesTensorBasis(scalar_fes) && UsesTensorBasis(vector_fes);
  const DofToQuad::Mode mode = tensor ? DofToQuad::TENSOR : DofToQuad::FULL;
  scalar_maps_ = &scalar_fe.GetDofToQuad(*ir, mode);
  vector_maps_ = &vector_fe.GetDofToQuad(*ir, mode);
  const ElementDofOrdering ordering = tensor ? ElementDofOrdering::LEXICOGRAPHIC
                                             : ElementDofOrdering::NATIVE;
  scalar_restriction_ = scalar_fes.GetElementRestriction(ordering);
  vector_restriction_ = vector_fes.GetElementRestriction(ordering);
  MFEM_VERIFY(scalar_restriction_ && vector_restriction_,
              "GradientCouplingOperator: no element restriction.");
  scalar_e_.SetSize(scalar_restriction_->Height());
  vector_e_.SetSize(vector_restriction_->Height());
  scalar_e_.UseDevice(true);
  vector_e_.UseDevice(true);

  const int nq = ir->GetNPoints();
  const int n = detail::pa::kCouplingDataSize;
  data_.SetSize(n * nq * ne_);
  data_.UseDevice(true);
  real_t* d = data_.HostWrite();
  for (int e = 0; e < ne_; e++) {
    ElementTransformation& T = *mesh.GetElementTransformation(e);
    for (int q = 0; q < nq; q++) {
      const IntegrationPoint& ip = ir->IntPoint(q);
      T.SetIntPoint(&ip);
      const DenseMatrix& Jinv = T.InverseJacobian();
      const real_t cw = ip.weight * T.Weight() * c.Eval(T, ip);
      real_t* p = d + static_cast<std::size_t>(n) *
                          (q + static_cast<std::size_t>(nq) * e);
      for (int a = 0; a < 3; a++) {
        for (int j = 0; j < 3; j++) {
          p[3 * a + j] = cw * Jinv(a, j);
        }
      }
    }
  }
}

void GradientCouplingOperator::Mult(const Vector& phi, Vector& u) const {
  if (ne_ == 0) {
    u = 0.0;
    return;
  }
  scalar_restriction_->Mult(phi, scalar_e_);
  vector_e_ = 0.0;
  detail::pa::ApplyGradientCoupling(false, ne_, *scalar_maps_, *vector_maps_,
                                    data_, scalar_e_, vector_e_);
  vector_restriction_->MultTranspose(vector_e_, u);
}

void GradientCouplingOperator::MultTranspose(const Vector& u,
                                             Vector& phi) const {
  if (ne_ == 0) {
    phi = 0.0;
    return;
  }
  vector_restriction_->Mult(u, vector_e_);
  scalar_e_ = 0.0;
  detail::pa::ApplyGradientCoupling(true, ne_, *scalar_maps_, *vector_maps_,
                                    data_, vector_e_, scalar_e_);
  scalar_restriction_->MultTranspose(scalar_e_, phi);
}

}  // namespace AdGIA
