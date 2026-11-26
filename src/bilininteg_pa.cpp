// Partial assembly of the bilinear form integrators: the forms applied
// element by element from data kept at the quadrature points, without an
// element matrix. Here the data are made, on the host; the kernels that
// apply them are in src/kernels/.
#include "AdGIA/bilininteg.hpp"
#include "AdGIA/detail/pa_kernels.hpp"
#include "AdGIA/elastic_tensor.hpp"

namespace AdGIA {

void ElasticTensorIntegrator::AssemblePA(const mfem::FiniteElementSpace& fes) {
  using namespace mfem;
  Mesh& mesh = *fes.GetMesh();
  MFEM_VERIFY(mesh.Dimension() == 3 && mesh.SpaceDimension() == 3 &&
                  fes.GetVDim() == 3,
              "ElasticTensorIntegrator: partial assembly is for 3-D.");
  Array<Geometry::Type> geoms;
  mesh.GetGeometries(3, geoms);
  MFEM_VERIFY(geoms.Size() == 0 ||
                  (geoms.Size() == 1 && geoms[0] == Geometry::CUBE &&
                   UsesTensorBasis(fes)),
              "ElasticTensorIntegrator: partial assembly is for hexahedra "
              "with a tensor basis.");
  MFEM_VERIFY(C_->GetHeight() == 6 && C_->GetWidth() == 6,
              "ElasticTensorIntegrator: the tensor coefficient must be "
              "6 x 6.");
  const FiniteElement& el = *fes.GetTypicalFE();
  // The rule of the element matrices, so that the two assemblies give
  // one operator.
  const IntegrationRule* ir = IntRule;
  if (ir == nullptr) {
    ElementTransformation& T0 = *mesh.GetTypicalElementTransformation();
    ir = &IntRules.Get(el.GetGeomType(), 2 * T0.OrderGrad(&el));
  }
  pa_maps_ = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
  pa_ne_ = fes.GetNE();
  pa_isotropic_ =
      dynamic_cast<IsotropicElasticTensorCoefficient*>(C_) != nullptr;
  const int nq = ir->GetNPoints();
  const int n = 9 + (pa_isotropic_ ? detail::pa::kIsotropicTensorSize
                                   : detail::pa::kGeneralTensorSize);
  pa_data_.SetSize(n * nq * pa_ne_);
  pa_data_.UseDevice(true);
  real_t* d = pa_data_.HostWrite();

  // The coefficients are evaluated here, on the host, once.
  DenseMatrix Jinv(3), F(3), JF(3), Cm(6), Cv(6);
  const int i00 = SymmetricTensorBasis::Index(3, 0, 0);
  const int i11 = SymmetricTensorBasis::Index(3, 1, 1);
  const int i01 = SymmetricTensorBasis::Index(3, 0, 1);
  for (int e = 0; e < pa_ne_; e++) {
    ElementTransformation& T = *mesh.GetElementTransformation(e);
    for (int q = 0; q < nq; q++) {
      const IntegrationPoint& ip = ir->IntPoint(q);
      T.SetIntPoint(&ip);
      Jinv = T.InverseJacobian();
      real_t w = ip.weight * T.Weight();
      if (map_) {
        // Pull-back: derivatives with respect to the mapped coordinates
        // and the mapping's Jacobian in the weight.
        map_->EvalGradient(F, T, ip);
        w *= F.Det();
        F.Invert();
        Mult(Jinv, F, JF);
        Jinv = JF;
      }
      C_->Eval(Cm, T, ip);
      real_t* p = d + static_cast<std::size_t>(n) * (q + static_cast<std::size_t>(nq) * e);
      for (int a = 0; a < 3; a++) {
        for (int j = 0; j < 3; j++) {
          p[3 * a + j] = Jinv(a, j);
        }
      }
      if (pa_isotropic_) {
        p[9] = w * Cm(i00, i11);
        p[10] = w * 0.5 * Cm(i01, i01);
      } else {
        SymmetricTensorBasis::ToVoigt(3, Cm, Cv);
        for (int i = 0; i < 6; i++) {
          for (int j = i; j < 6; j++) {
            p[9 + detail::pa::PackedIndex(i, j)] = w * Cv(i, j);
          }
        }
      }
    }
  }
}

void ElasticTensorIntegrator::AddMultPA(const mfem::Vector& x,
                                        mfem::Vector& y) const {
  MFEM_VERIFY(pa_maps_, "ElasticTensorIntegrator: AssemblePA not called.");
  if (pa_ne_ == 0) {
    return;
  }
  detail::pa::ApplyElasticTensor(pa_isotropic_, pa_ne_, *pa_maps_, pa_data_, x,
                                 y);
}

void ElasticTensorIntegrator::AddMultTransposePA(const mfem::Vector& x,
                                                 mfem::Vector& y) const {
  AddMultPA(x, y);  // the form is symmetric
}

void ElasticTensorIntegrator::AssembleDiagonalPA(mfem::Vector& diag) {
  MFEM_VERIFY(pa_maps_, "ElasticTensorIntegrator: AssemblePA not called.");
  if (pa_ne_ == 0) {
    return;
  }
  detail::pa::DiagonalElasticTensor(pa_isotropic_, pa_ne_, *pa_maps_,
                                    pa_data_, diag);
}

}  // namespace AdGIA
