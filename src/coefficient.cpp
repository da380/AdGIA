/**
 * @file coefficient.cpp
 * @brief Implementation of the coefficients in coefficient.hpp.
 */

#include "AdGIA/coefficient.hpp"

namespace AdGIA {

using namespace mfem;

RadialUnitVectorCoefficient::RadialUnitVectorCoefficient(int dim)
    : VectorCoefficient(dim), x0_(dim), x_(dim) {
  x0_ = 0.0;
}

// Hidden from Doxygen, which cannot match the unqualified parameter types
// of these overloaded constructors to their declarations.
/// @cond
RadialUnitVectorCoefficient::RadialUnitVectorCoefficient(int dim,
                                                         const Vector& x0)
    : VectorCoefficient(dim), x0_(x0), x_(dim) {
  MFEM_VERIFY(x0.Size() == dim, "RadialUnitVectorCoefficient: x0 size.");
}
/// @endcond

void RadialUnitVectorCoefficient::Eval(Vector& V, ElementTransformation& T,
                                       const IntegrationPoint& ip) {
  T.Transform(ip, x_);
  V.SetSize(vdim);
  V = x_;
  V -= x0_;
  const real_t r = V.Norml2();
  if (r > 0.0) {
    V /= r;
  } else {
    V = 0.0;
    V[vdim - 1] = 1.0;
  }
}

real_t BoundaryNormalDotCoefficient::Eval(ElementTransformation& T,
                                          const IntegrationPoint& ip) {
  MFEM_VERIFY(T.ElementType == ElementTransformation::BDR_ELEMENT,
              "BoundaryNormalDotCoefficient: only defined on boundary "
              "elements.");
  T.SetIntPoint(&ip);
  n_.SetSize(T.GetSpaceDim());
  CalcOrtho(T.Jacobian(), n_);
  const real_t nrm = n_.Norml2();
  if (nrm <= 0.0) {
    return 0.0;
  }
  V_->Eval(v_, T, ip);
  return (v_ * n_) / nrm;
}

BarotropicDensityGradientCoefficient::BarotropicDensityGradientCoefficient(
    VectorCoefficient& grad_rho, VectorCoefficient& grad_phi0)
    : grad_rho_(&grad_rho), grad_phi0_(&grad_phi0) {}

// Hidden from Doxygen, as for RadialUnitVectorCoefficient above.
/// @cond
BarotropicDensityGradientCoefficient::BarotropicDensityGradientCoefficient(
    const GridFunction& rho, const GridFunction& phi0)
    : grad_rho_from_rho_(
          std::make_unique<GradientGridFunctionCoefficient>(&rho)),
      grad_phi0_from_phi0_(
          std::make_unique<GradientGridFunctionCoefficient>(&phi0)),
      grad_rho_(grad_rho_from_rho_.get()),
      grad_phi0_(grad_phi0_from_phi0_.get()) {
  MFEM_VERIFY(rho.FESpace()->GetMesh() == phi0.FESpace()->GetMesh(),
              "BarotropicDensityGradientCoefficient: the density and the "
              "potential must live on the same mesh.");
}
/// @endcond

real_t BarotropicDensityGradientCoefficient::Eval(ElementTransformation& T,
                                                  const IntegrationPoint& ip) {
  T.SetIntPoint(&ip);
  grad_phi0_->Eval(gp_, T, ip);
  const real_t g2 = gp_ * gp_;
  if (g2 <= 0.0) {
    return 0.0;
  }
  grad_rho_->Eval(gr_, T, ip);
  return (gr_ * gp_) / g2;
}

void MatrixDeltaCoefficient::SetTime(mfem::real_t t) {
  d_.SetTime(t);
  mfem::MatrixCoefficient::SetTime(t);
}

void MatrixDeltaCoefficient::SetMatrix(const mfem::DenseMatrix& M) {
  MFEM_VERIFY(M.Height() == M_.Height() && M.Width() == M_.Width(),
              "MatrixDeltaCoefficient::SetMatrix: dimensions must match.");
  M_ = M;
}

void MatrixDeltaCoefficient::EvalDelta(mfem::DenseMatrix& M,
                                       mfem::ElementTransformation& T,
                                       const mfem::IntegrationPoint& ip) {
  M = M_;
  d_.SetTime(GetTime());
  M *= d_.EvalDelta(T, ip);
}

void MatrixDeltaCoefficient::Eval(mfem::DenseMatrix& /*M*/,
                                  mfem::ElementTransformation& /*T*/,
                                  const mfem::IntegrationPoint& /*ip*/) {
  mfem::mfem_error("MatrixDeltaCoefficient::Eval");
}

}  // namespace AdGIA
