// Partial assembly of the bilinear form integrators: the forms applied
// element by element from data kept at the quadrature points, without an
// element matrix. Here the data are made, on the host; the kernels that
// apply them are in src/kernels/, and their libCEED counterparts in
// src/ceed/.
#include "AdGIA/bilininteg.hpp"
#include "AdGIA/detail/ceed_operators.hpp"
#include "AdGIA/detail/displacement_block_pa.hpp"
#include "AdGIA/detail/pa_kernels.hpp"
#include "AdGIA/elastic_tensor.hpp"
#include "AdGIA/mappings.hpp"

namespace AdGIA {

using namespace mfem;

namespace detail {

void DisplacementBlockPA::Assemble(const FiniteElementSpace& fes,
                                   const IntegrationRule* ir,
                                   const DisplacementBlockTerms& terms,
                                   const char* who) {
  Mesh& mesh = *fes.GetMesh();
  MFEM_VERIFY(mesh.Dimension() == 3 && mesh.SpaceDimension() == 3 &&
                  fes.GetVDim() == 3,
              who << ": partial assembly is for 3-D.");
  Array<Geometry::Type> geoms;
  mesh.GetGeometries(3, geoms);
  MFEM_VERIFY(geoms.Size() <= 1,
              who << ": partial assembly is for meshes of one element "
                     "geometry.");
  MFEM_VERIFY(terms.tensor && terms.tensor->GetHeight() == 6 &&
                  terms.tensor->GetWidth() == 6,
              who << ": the tensor coefficient must be 6 x 6.");
  const bool gravity = terms.rho != nullptr;
  MFEM_VERIFY(!gravity || (terms.grad_phi0 && terms.hessian_phi0),
              who << ": the gravity terms need the gradient and the Hessian "
                     "of the background potential.");
  MFEM_VERIFY(!(gravity && terms.map),
              who << ": no mapping with the gravity terms.");

  ceed_.reset();
  maps_ = nullptr;
  data_.Destroy();
  ne_ = fes.GetNE();
  isotropic_ = dynamic_cast<IsotropicElasticTensorCoefficient*>(
                   terms.tensor) != nullptr;
  extras_ = gravity ? pa::kGravityExtras
                    : terms.mass ? pa::kMassExtras : pa::kNoExtras;
  if (ne_ == 0) {
    return;
  }
  const FiniteElement& el = *fes.GetTypicalFE();
  // By default the rule of the element matrices, so that the two
  // assemblies give one operator.
  if (ir == nullptr) {
    ElementTransformation& T0 = *mesh.GetTypicalElementTransformation();
    ir = &IntRules.Get(el.GetGeomType(), 2 * T0.OrderGrad(&el));
  }
  const int nq = ir->GetNPoints();
  const int n = pa::DataSize(isotropic_, extras_);
  const int nc = isotropic_ ? pa::kIsotropicTensorSize
                            : pa::kGeneralTensorSize;
  data_.SetSize(n * nq * ne_);
  data_.UseDevice(true);
  real_t* d = data_.HostWrite();

  // The coefficients are evaluated here, on the host, once.
  DenseMatrix Jinv(3), F(3), JF(3), Cm(6), Cv(6), H(3);
  Vector g(3);
  const int i00 = SymmetricTensorBasis::Index(3, 0, 0);
  const int i11 = SymmetricTensorBasis::Index(3, 1, 1);
  const int i01 = SymmetricTensorBasis::Index(3, 0, 1);
  for (int e = 0; e < ne_; e++) {
    ElementTransformation& T = *mesh.GetElementTransformation(e);
    for (int q = 0; q < nq; q++) {
      const IntegrationPoint& ip = ir->IntPoint(q);
      T.SetIntPoint(&ip);
      Jinv = T.InverseJacobian();
      real_t w = ip.weight * T.Weight();
      if (terms.map) {
        // Pull-back: derivatives with respect to the mapped coordinates
        // and the mapping's Jacobian in the weight.
        terms.map->EvalGradient(F, T, ip);
        w *= F.Det();
        F.Invert();
        Mult(Jinv, F, JF);
        Jinv = JF;
      }
      real_t* p = d + static_cast<std::size_t>(n) *
                          (q + static_cast<std::size_t>(nq) * e);
      for (int a = 0; a < 3; a++) {
        for (int j = 0; j < 3; j++) {
          p[3 * a + j] = Jinv(a, j);
        }
      }
      terms.tensor->Eval(Cm, T, ip);
      if (isotropic_) {
        p[9] = w * Cm(i00, i11);
        p[10] = w * 0.5 * Cm(i01, i01);
      } else {
        SymmetricTensorBasis::ToVoigt(3, Cm, Cv);
        for (int i = 0; i < 6; i++) {
          for (int j = i; j < 6; j++) {
            p[9 + pa::PackedIndex(i, j)] = w * Cv(i, j);
          }
        }
      }
      real_t* px = p + 9 + nc;
      const real_t s = terms.mass ? w * terms.mass->Eval(T, ip) : 0.0;
      if (gravity) {
        const real_t half_rho = 0.5 * w * terms.rho->Eval(T, ip);
        terms.grad_phi0->Eval(g, T, ip);
        terms.hessian_phi0->Eval(H, T, ip);
        for (int k = 0; k < 3; k++) {
          px[k] = half_rho * g(k);
          for (int l = 0; l < 3; l++) {
            px[3 + 3 * k + l] = half_rho * H(k, l);
          }
        }
        px[12] = s;
      } else if (terms.mass) {
        px[0] = s;
      }
    }
  }

  if (DeviceCanUseCeed()) {
    ceed_ = MakeDisplacementBlockCeedOperator(fes, *ir, data_, isotropic_,
                                              extras_);
    data_.Destroy();  // libCEED holds its copy
  } else {
    // Sum-factorised on a tensor basis, dense in the element otherwise.
    maps_ = &el.GetDofToQuad(*ir, UsesTensorBasis(fes) ? DofToQuad::TENSOR
                                                       : DofToQuad::FULL);
  }
}

void DisplacementBlockPA::AddMult(const Vector& x, Vector& y) const {
  if (ceed_) {
    ceed_->AddMult(x, y);
  } else if (ne_ > 0) {
    MFEM_VERIFY(maps_, "partial assembly: AssemblePA not called.");
    pa::ApplyDisplacementBlock(isotropic_, extras_, ne_, *maps_, data_, x, y);
  }
}

void DisplacementBlockPA::AddDiagonal(Vector& diag) const {
  if (ceed_) {
    ceed_->GetDiagonal(diag);
  } else if (ne_ > 0) {
    MFEM_VERIFY(maps_, "partial assembly: AssemblePA not called.");
    pa::DiagonalDisplacementBlock(isotropic_, extras_, ne_, *maps_, data_,
                                  diag);
  }
}

}  // namespace detail

// ---------------------------------------------------------------------------
// ElasticTensorIntegrator

void ElasticTensorIntegrator::AssemblePA(const FiniteElementSpace& fes) {
  detail::DisplacementBlockTerms terms;
  terms.tensor = C_;
  terms.map = map_;
  pa_.Assemble(fes, IntRule, terms, "ElasticTensorIntegrator");
}

void ElasticTensorIntegrator::AddMultPA(const Vector& x, Vector& y) const {
  pa_.AddMult(x, y);
}

void ElasticTensorIntegrator::AddMultTransposePA(const Vector& x,
                                                 Vector& y) const {
  pa_.AddMult(x, y);  // the form is symmetric
}

void ElasticTensorIntegrator::AssembleDiagonalPA(Vector& diag) {
  pa_.AddDiagonal(diag);
}

// ---------------------------------------------------------------------------
// GravitatingElasticityIntegrator

GravitatingElasticityIntegrator::GravitatingElasticityIntegrator(
    MatrixCoefficient& C, const IntegrationRule* ir)
    : BilinearFormIntegrator(ir), C_(&C), elastic_(C) {}

GravitatingElasticityIntegrator::GravitatingElasticityIntegrator(
    MatrixCoefficient& C, Coefficient& rho, VectorCoefficient& grad_phi0,
    MatrixCoefficient& hessian_phi0, const IntegrationRule* ir)
    : BilinearFormIntegrator(ir),
      C_(&C),
      rho_(&rho),
      grad_phi0_(&grad_phi0),
      hessian_phi0_(&hessian_phi0),
      elastic_(C) {
  half_rho_ = std::make_unique<ProductCoefficient>(0.5, rho);
  minus_half_rho_ = std::make_unique<ProductCoefficient>(-0.5, rho);
  minus_half_rho_grad_ = std::make_unique<ScalarVectorProductCoefficient>(
      *minus_half_rho_, grad_phi0);
  gradient_ = std::make_unique<DomainVectorGradVectorIntegrator>(grad_phi0,
                                                                 *half_rho_);
  gradient_->SetGradient(hessian_phi0_);
  divergence_ = std::make_unique<DomainVectorDivVectorIntegrator>(
      *minus_half_rho_grad_);
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
  elastic_.AssembleElementMatrix(el, Trans, elmat);
  const int n = elmat.Height();
  const auto add_symmetrised = [&]() {
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < n; i++) {
        elmat(i, j) += part_(i, j) + part_(j, i);
      }
    }
  };
  if (gradient_) {
    gradient_->SetIntRule(ir);
    divergence_->SetIntRule(ir);
    gradient_->AssembleElementMatrix(el, Trans, part_);
    add_symmetrised();
    divergence_->AssembleElementMatrix(el, Trans, part_);
    add_symmetrised();
  }
  if (mass_) {
    mass_->SetIntRule(ir);
    mass_->AssembleElementMatrix(el, Trans, part_);
    elmat += part_;
  }
}

void GravitatingElasticityIntegrator::AssemblePA(
    const FiniteElementSpace& fes) {
  detail::DisplacementBlockTerms terms;
  terms.tensor = C_;
  terms.rho = rho_;
  terms.grad_phi0 = grad_phi0_;
  terms.hessian_phi0 = hessian_phi0_;
  terms.mass = mass_coefficient_;
  pa_.Assemble(fes, IntRule, terms, "GravitatingElasticityIntegrator");
}

void GravitatingElasticityIntegrator::AddMultPA(const Vector& x,
                                                Vector& y) const {
  pa_.AddMult(x, y);
}

void GravitatingElasticityIntegrator::AddMultTransposePA(const Vector& x,
                                                         Vector& y) const {
  pa_.AddMult(x, y);  // the block is symmetric
}

void GravitatingElasticityIntegrator::AssembleDiagonalPA(Vector& diag) {
  pa_.AddDiagonal(diag);
}

// ---------------------------------------------------------------------------
// DiffusionMassIntegrator

DiffusionMassIntegrator::DiffusionMassIntegrator(Coefficient& k, Coefficient* s,
                                                 const IntegrationRule* ir)
    : BilinearFormIntegrator(ir), k_(&k), s_(s), diffusion_(k) {
  if (s) {
    mass_ = std::make_unique<MassIntegrator>(*s);
  }
}

const IntegrationRule& DiffusionMassIntegrator::Rule(
    const FiniteElement& el, ElementTransformation& Trans) const {
  // A rule for the mass term serves both.
  return IntRule ? *IntRule
                 : IntRules.Get(el.GetGeomType(),
                                2 * el.GetOrder() + Trans.OrderW());
}

void DiffusionMassIntegrator::AssembleElementMatrix(
    const FiniteElement& el, ElementTransformation& Trans,
    DenseMatrix& elmat) {
  const IntegrationRule& ir = Rule(el, Trans);
  diffusion_.SetIntRule(&ir);
  diffusion_.AssembleElementMatrix(el, Trans, elmat);
  if (mass_) {
    mass_->SetIntRule(&ir);
    mass_->AssembleElementMatrix(el, Trans, part_);
    elmat += part_;
  }
}

void DiffusionMassIntegrator::AssemblePA(const FiniteElementSpace& fes) {
  Mesh& mesh = *fes.GetMesh();
  MFEM_VERIFY(mesh.Dimension() == 3 && mesh.SpaceDimension() == 3 &&
                  fes.GetVDim() == 1,
              "DiffusionMassIntegrator: partial assembly is for a scalar "
              "space in 3-D.");
  Array<Geometry::Type> geoms;
  mesh.GetGeometries(3, geoms);
  MFEM_VERIFY(geoms.Size() <= 1,
              "DiffusionMassIntegrator: partial assembly is for meshes of "
              "one element geometry.");
  pa_maps_ = nullptr;
  pa_data_.Destroy();
  pa_ceed_.reset();
  pa_ne_ = fes.GetNE();
  if (pa_ne_ == 0) {
    return;
  }
  const FiniteElement& el = *fes.GetTypicalFE();
  const IntegrationRule& ir =
      Rule(el, *mesh.GetTypicalElementTransformation());
  const int nq = ir.GetNPoints();
  const int n = detail::pa::kPotentialDataSize;
  pa_data_.SetSize(n * nq * pa_ne_);
  pa_data_.UseDevice(true);
  real_t* d = pa_data_.HostWrite();
  for (int e = 0; e < pa_ne_; e++) {
    ElementTransformation& T = *mesh.GetElementTransformation(e);
    for (int q = 0; q < nq; q++) {
      const IntegrationPoint& ip = ir.IntPoint(q);
      T.SetIntPoint(&ip);
      const DenseMatrix& Jinv = T.InverseJacobian();
      const real_t w = ip.weight * T.Weight();
      const real_t kw = w * k_->Eval(T, ip);
      real_t* p = d + static_cast<std::size_t>(n) *
                          (q + static_cast<std::size_t>(nq) * e);
      int t = 0;
      for (int a = 0; a < 3; a++) {
        for (int b = a; b < 3; b++) {
          p[t++] = kw * (Jinv(a, 0) * Jinv(b, 0) + Jinv(a, 1) * Jinv(b, 1) +
                         Jinv(a, 2) * Jinv(b, 2));
        }
      }
      p[6] = s_ ? w * s_->Eval(T, ip) : 0.0;
    }
  }
  if (DeviceCanUseCeed()) {
    pa_ceed_ = detail::MakePotentialBlockCeedOperator(fes, ir, pa_data_);
    pa_data_.Destroy();  // libCEED holds its copy
    return;
  }
  pa_maps_ = &el.GetDofToQuad(ir, UsesTensorBasis(fes) ? DofToQuad::TENSOR
                                                       : DofToQuad::FULL);
}

void DiffusionMassIntegrator::AddMultPA(const Vector& x, Vector& y) const {
  if (pa_ceed_) {
    pa_ceed_->AddMult(x, y);
  } else if (pa_ne_ > 0) {
    MFEM_VERIFY(pa_maps_, "DiffusionMassIntegrator: AssemblePA not called.");
    detail::pa::ApplyPotentialBlock(pa_ne_, *pa_maps_, pa_data_, x, y);
  }
}

void DiffusionMassIntegrator::AddMultTransposePA(const Vector& x,
                                                 Vector& y) const {
  AddMultPA(x, y);  // the form is symmetric
}

void DiffusionMassIntegrator::AssembleDiagonalPA(Vector& diag) {
  if (pa_ceed_) {
    pa_ceed_->GetDiagonal(diag);
  } else if (pa_ne_ > 0) {
    MFEM_VERIFY(pa_maps_, "DiffusionMassIntegrator: AssemblePA not called.");
    detail::pa::DiagonalPotentialBlock(pa_ne_, *pa_maps_, pa_data_, diag);
  }
}

}  // namespace AdGIA
