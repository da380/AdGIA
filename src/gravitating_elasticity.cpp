// GravitatingElasticityIntegrator: the displacement block of the
// self-gravitating problem as one integrator (bilininteg.hpp).
#include "AdGIA/bilininteg.hpp"
#include "AdGIA/detail/ceed_operators.hpp"
#include "AdGIA/detail/pa_kernels.hpp"
#include "AdGIA/elastic_tensor.hpp"

namespace AdGIA {

using namespace mfem;

GravitatingElasticityIntegrator::GravitatingElasticityIntegrator(
    MatrixCoefficient& C, Coefficient& rho, VectorCoefficient& grad_phi0,
    MatrixCoefficient& hessian_phi0, const IntegrationRule* ir)
    : BilinearFormIntegrator(ir),
      C_(&C),
      rho_(&rho),
      grad_phi0_(&grad_phi0),
      hessian_phi0_(&hessian_phi0),
      half_rho_(0.5, rho),
      minus_half_rho_(-0.5, rho),
      minus_half_rho_grad_(minus_half_rho_, grad_phi0),
      elastic_(C),
      gradient_(grad_phi0, half_rho_),
      divergence_(minus_half_rho_grad_) {
  gradient_.SetGradient(hessian_phi0_);
}

void GravitatingElasticityIntegrator::SetMassTerm(Coefficient* s) {
  mass_coefficient_ = s;
  mass_ = s ? std::make_unique<VectorMassIntegrator>(*s) : nullptr;
}

void GravitatingElasticityIntegrator::AssembleElementMatrix(
    const FiniteElement& el, ElementTransformation& Trans,
    DenseMatrix& elmat) {
  const IntegrationRule* ir = IntRule;
  if (ir == nullptr) {
    ir = &IntRules.Get(el.GetGeomType(), 2 * Trans.OrderGrad(&el));
  }
  elastic_.SetIntRule(ir);
  gradient_.SetIntRule(ir);
  divergence_.SetIntRule(ir);

  elastic_.AssembleElementMatrix(el, Trans, elmat);
  const int n = elmat.Height();
  const auto add_symmetrised = [&]() {
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < n; i++) {
        elmat(i, j) += part_(i, j) + part_(j, i);
      }
    }
  };
  gradient_.AssembleElementMatrix(el, Trans, part_);
  add_symmetrised();
  divergence_.AssembleElementMatrix(el, Trans, part_);
  add_symmetrised();
  if (mass_) {
    mass_->SetIntRule(ir);
    mass_->AssembleElementMatrix(el, Trans, part_);
    elmat += part_;
  }
}

void GravitatingElasticityIntegrator::AssemblePA(
    const FiniteElementSpace& fes) {
  MFEM_VERIFY(DeviceCanUseCeed(),
              "GravitatingElasticityIntegrator: partial assembly needs a "
              "device that uses libCEED (mfem built with MFEM_USE_CEED; "
              "e.g. the device ceed-cpu).");
  Mesh& mesh = *fes.GetMesh();
  MFEM_VERIFY(mesh.Dimension() == 3 && mesh.SpaceDimension() == 3 &&
                  fes.GetVDim() == 3,
              "GravitatingElasticityIntegrator: partial assembly is for "
              "3-D.");
  Array<Geometry::Type> geoms;
  mesh.GetGeometries(3, geoms);
  MFEM_VERIFY(geoms.Size() <= 1,
              "GravitatingElasticityIntegrator: partial assembly is for "
              "meshes of one element geometry.");
  MFEM_VERIFY(C_->GetHeight() == 6 && C_->GetWidth() == 6,
              "GravitatingElasticityIntegrator: the tensor coefficient "
              "must be 6 x 6.");
  pa_ceed_.reset();
  const int ne = fes.GetNE();
  if (ne == 0) {
    return;
  }
  const FiniteElement& el = *fes.GetTypicalFE();
  const IntegrationRule* ir = IntRule;
  if (ir == nullptr) {
    ElementTransformation& T0 = *mesh.GetTypicalElementTransformation();
    ir = &IntRules.Get(el.GetGeomType(), 2 * T0.OrderGrad(&el));
  }
  const bool isotropic =
      dynamic_cast<IsotropicElasticTensorCoefficient*>(C_) != nullptr;
  const int nq = ir->GetNPoints();
  const int nc = isotropic ? detail::pa::kIsotropicTensorSize
                           : detail::pa::kGeneralTensorSize;
  const int n = 9 + nc + detail::kGravityDataSize;
  Vector data(n * nq * ne);
  real_t* d = data.HostWrite();

  DenseMatrix Cm(6), Cv(6), H(3);
  Vector g(3);
  const int i00 = SymmetricTensorBasis::Index(3, 0, 0);
  const int i11 = SymmetricTensorBasis::Index(3, 1, 1);
  const int i01 = SymmetricTensorBasis::Index(3, 0, 1);
  for (int e = 0; e < ne; e++) {
    ElementTransformation& T = *mesh.GetElementTransformation(e);
    for (int q = 0; q < nq; q++) {
      const IntegrationPoint& ip = ir->IntPoint(q);
      T.SetIntPoint(&ip);
      const DenseMatrix& Jinv = T.InverseJacobian();
      const real_t w = ip.weight * T.Weight();
      real_t* p = d + static_cast<std::size_t>(n) *
                          (q + static_cast<std::size_t>(nq) * e);
      for (int a = 0; a < 3; a++) {
        for (int j = 0; j < 3; j++) {
          p[3 * a + j] = Jinv(a, j);
        }
      }
      C_->Eval(Cm, T, ip);
      if (isotropic) {
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
      real_t* pg = p + 9 + nc;
      const real_t half_rho = 0.5 * w * rho_->Eval(T, ip);
      grad_phi0_->Eval(g, T, ip);
      hessian_phi0_->Eval(H, T, ip);
      for (int k = 0; k < 3; k++) {
        pg[k] = half_rho * g(k);
        for (int l = 0; l < 3; l++) {
          pg[3 + 3 * k + l] = half_rho * H(k, l);
        }
      }
      pg[12] = mass_coefficient_ ? w * mass_coefficient_->Eval(T, ip) : 0.0;
    }
  }
  pa_ceed_ =
      detail::MakeGravitatingElasticityCeedOperator(fes, *ir, data, isotropic);
}

void GravitatingElasticityIntegrator::AddMultPA(const Vector& x,
                                                Vector& y) const {
  if (pa_ceed_) {
    pa_ceed_->AddMult(x, y);
  }
}

void GravitatingElasticityIntegrator::AddMultTransposePA(const Vector& x,
                                                         Vector& y) const {
  AddMultPA(x, y);  // the block is symmetric
}

void GravitatingElasticityIntegrator::AssembleDiagonalPA(Vector& diag) {
  if (pa_ceed_) {
    pa_ceed_->GetDiagonal(diag);
  }
}

}  // namespace AdGIA
