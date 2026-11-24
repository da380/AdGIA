/**
 * @file material_law.cpp
 * @brief Implementation of material_law.hpp.
 */

#include "AdGIA/material_law.hpp"

namespace AdGIA {

using namespace mfem;

void GreenStrain(const DenseMatrix& F, Vector& e) {
  const auto dim = F.Width();
  e.SetSize(SymmetricTensorBasis::Size(dim));
  for (auto j = 0; j < dim; j++) {
    for (auto k = 0; k <= j; k++) {
      auto c = real_t{0};
      for (auto i = 0; i < dim; i++) {
        c += F(i, j) * F(i, k);
      }
      const auto value = 0.5 * (c - (j == k ? 1.0 : 0.0));
      e[SymmetricTensorBasis::Index(dim, j, k)] =
          SymmetricTensorBasis::Scale(j, k) * value;
    }
  }
}

void MandelToMatrix(int dim, const Vector& s, DenseMatrix& S) {
  S.SetSize(dim);
  for (auto j = 0; j < dim; j++) {
    for (auto k = 0; k <= j; k++) {
      const auto value = s[SymmetricTensorBasis::Index(dim, j, k)] /
                         SymmetricTensorBasis::Scale(j, k);
      S(j, k) = value;
      S(k, j) = value;
    }
  }
}

void MatrixToMandel(const DenseMatrix& S, Vector& s) {
  const auto dim = S.Width();
  s.SetSize(SymmetricTensorBasis::Size(dim));
  for (auto j = 0; j < dim; j++) {
    for (auto k = 0; k <= j; k++) {
      s[SymmetricTensorBasis::Index(dim, j, k)] =
          SymmetricTensorBasis::Scale(j, k) * 0.5 * (S(j, k) + S(k, j));
    }
  }
}

QuadraticMaterialLaw::QuadraticMaterialLaw(int dim, MatrixCoefficient& Xi,
                                           MatrixCoefficient* T0)
    : MaterialLaw(dim), Xi_(&Xi), T0_(T0) {
  MFEM_VERIFY(Xi.GetHeight() == StrainSize() && Xi.GetWidth() == StrainSize(),
              "QuadraticMaterialLaw: the elastic tensor must be n_s x n_s "
              "with n_s = d(d+1)/2.");
  MFEM_VERIFY(!T0 || (T0->GetHeight() == dim && T0->GetWidth() == dim),
              "QuadraticMaterialLaw: the equilibrium stress must be d x d.");
}

real_t QuadraticMaterialLaw::Evaluate(ElementTransformation& T,
                                      const IntegrationPoint& ip,
                                      const Vector& e, Vector* S,
                                      DenseMatrix* tangent) {
  Xi_->Eval(Xi_q_, T, ip);
  s_.SetSize(StrainSize());
  Xi_q_.Mult(e, s_);
  auto energy = 0.5 * (s_ * e);
  if (T0_) {
    T0_->Eval(T0_q_, T, ip);
    MatrixToMandel(T0_q_, t0_);
    energy += t0_ * e;
    s_ += t0_;
  }
  if (S) {
    *S = s_;
  }
  if (tangent) {
    *tangent = Xi_q_;
  }
  return energy;
}

namespace {

// The law at the deformation of a mapping: F and the Green strain at the
// point, then the law.
real_t EvaluateAt(MaterialLaw& law, Diffeomorphism& map,
                  ElementTransformation& T, const IntegrationPoint& ip,
                  DenseMatrix& F, Vector& e, Vector* s, DenseMatrix* tangent) {
  F.SetSize(law.SpaceDim());
  map.EvalGradient(F, T, ip);
  GreenStrain(F, e);
  return law.Evaluate(T, ip, e, s, tangent);
}

// The stress of the law at the reference, E = 0.
void ReferenceStress(MaterialLaw& law, ElementTransformation& T,
                     const IntegrationPoint& ip, Vector& e, Vector& s,
                     DenseMatrix& S) {
  e.SetSize(law.StrainSize());
  e = 0.0;
  law.Evaluate(T, ip, e, &s, nullptr);
  MandelToMatrix(law.SpaceDim(), s, S);
}

}  // namespace

StrainEnergyDensityCoefficient::StrainEnergyDensityCoefficient(
    MaterialLaw& law, Diffeomorphism& map, bool subtract_reference)
    : law_(&law), map_(&map), subtract_reference_(subtract_reference) {}

real_t StrainEnergyDensityCoefficient::Eval(ElementTransformation& T,
                                            const IntegrationPoint& ip) {
  auto energy = EvaluateAt(*law_, *map_, T, ip, F_, e_, nullptr, nullptr);
  if (subtract_reference_) {
    ReferenceStress(*law_, T, ip, e_, s_, S_);
    const auto dim = law_->SpaceDim();
    for (auto i = 0; i < dim; i++) {
      for (auto j = 0; j < dim; j++) {
        energy -= S_(i, j) * (F_(i, j) - (i == j ? 1.0 : 0.0));
      }
    }
  }
  return energy;
}

SecondPiolaKirchhoffStressCoefficient::SecondPiolaKirchhoffStressCoefficient(
    MaterialLaw& law, Diffeomorphism& map)
    : MatrixCoefficient(law.SpaceDim()), law_(&law), map_(&map) {}

void SecondPiolaKirchhoffStressCoefficient::Eval(DenseMatrix& S,
                                                 ElementTransformation& T,
                                                 const IntegrationPoint& ip) {
  EvaluateAt(*law_, *map_, T, ip, F_, e_, &s_, nullptr);
  MandelToMatrix(law_->SpaceDim(), s_, S);
}

FirstPiolaKirchhoffStressCoefficient::FirstPiolaKirchhoffStressCoefficient(
    MaterialLaw& law, Diffeomorphism& map, bool subtract_reference)
    : MatrixCoefficient(law.SpaceDim()),
      law_(&law),
      map_(&map),
      subtract_reference_(subtract_reference) {}

void FirstPiolaKirchhoffStressCoefficient::Eval(DenseMatrix& P,
                                                ElementTransformation& T,
                                                const IntegrationPoint& ip) {
  const auto dim = law_->SpaceDim();
  EvaluateAt(*law_, *map_, T, ip, F_, e_, &s_, nullptr);
  MandelToMatrix(dim, s_, S_);
  P.SetSize(dim);
  Mult(F_, S_, P);
  if (subtract_reference_) {
    ReferenceStress(*law_, T, ip, e_, s_, S_);
    P -= S_;
  }
}

MaterialTangentCoefficient::MaterialTangentCoefficient(MaterialLaw& law,
                                                       Diffeomorphism& map)
    : MatrixCoefficient(law.StrainSize()), law_(&law), map_(&map) {}

void MaterialTangentCoefficient::Eval(DenseMatrix& C, ElementTransformation& T,
                                      const IntegrationPoint& ip) {
  C.SetSize(law_->StrainSize());
  EvaluateAt(*law_, *map_, T, ip, F_, e_, nullptr, &C);
}

}  // namespace AdGIA
