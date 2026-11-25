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

ThirdOrderIsotropicLaw::ThirdOrderIsotropicLaw(MaterialLaw& base,
                                               Coefficient& nu1,
                                               Coefficient& nu2,
                                               Coefficient& nu3)
    : MaterialLaw(base.SpaceDim()), base_(&base), nu_{&nu1, &nu2, &nu3} {}

real_t ThirdOrderIsotropicLaw::Evaluate(ElementTransformation& T,
                                        const IntegrationPoint& ip,
                                        const Vector& e, Vector* S,
                                        DenseMatrix* tangent) {
  const auto dim = SpaceDim();
  auto energy = base_->Evaluate(T, ip, e, S, tangent);
  const auto nu1 = nu_[0]->Eval(T, ip), nu2 = nu_[1]->Eval(T, ip),
             nu3 = nu_[2]->Eval(T, ip);
  MandelToMatrix(dim, e, E_);
  E2_.SetSize(dim);
  mfem::Mult(E_, E_, E2_);
  const auto j1 = E_.Trace(), j2 = E2_.Trace();
  auto j3 = real_t{0};
  for (auto i = 0; i < dim; i++) {
    for (auto j = 0; j < dim; j++) {
      j3 += E2_(i, j) * E_(j, i);
    }
  }
  energy += nu1 * j1 * j1 * j1 / 6.0 + nu2 * j1 * j2 + 4.0 * nu3 * j3 / 3.0;
  if (S) {
    // nu1 j1^2 / 2 + nu2 (j2 1 + 2 j1 E) + 4 nu3 E^2
    D_.SetSize(dim);
    D_ = 0.0;
    D_.Add(2.0 * nu2 * j1, E_);
    D_.Add(4.0 * nu3, E2_);
    for (auto i = 0; i < dim; i++) {
      D_(i, i) += 0.5 * nu1 * j1 * j1 + nu2 * j2;
    }
    MatrixToMandel(D_, column_);
    *S += column_;
  }
  if (tangent) {
    // On a symmetric H: nu1 j1 tr(H) 1
    //   + nu2 (2 tr(E H) 1 + 2 tr(H) E + 2 j1 H) + 4 nu3 (E H + H E),
    // column by column on the Mandel basis.
    const auto size = StrainSize();
    unit_.SetSize(size);
    D_.SetSize(dim);
    for (auto a = 0; a < size; a++) {
      unit_ = 0.0;
      unit_[a] = 1.0;
      MandelToMatrix(dim, unit_, H_);
      const auto trH = H_.Trace();
      auto trEH = real_t{0};
      for (auto i = 0; i < dim; i++) {
        for (auto j = 0; j < dim; j++) {
          trEH += E_(i, j) * H_(j, i);
        }
      }
      mfem::Mult(E_, H_, D_);
      D_.Symmetrize();
      D_ *= 8.0 * nu3;
      D_.Add(2.0 * nu2 * trH, E_);
      D_.Add(2.0 * nu2 * j1, H_);
      for (auto i = 0; i < dim; i++) {
        D_(i, i) += nu1 * j1 * trH + 2.0 * nu2 * trEH;
      }
      MatrixToMandel(D_, column_);
      for (auto b = 0; b < size; b++) {
        (*tangent)(b, a) += column_[b];
      }
    }
  }
  return energy;
}

MappedMaterialLaw::MappedMaterialLaw(MaterialLaw& law,
                                     Diffeomorphism& equilibrium)
    : MaterialLaw(law.SpaceDim()), law_(&law), equilibrium_(&equilibrium) {}

real_t MappedMaterialLaw::Evaluate(ElementTransformation& T,
                                   const IntegrationPoint& ip, const Vector& e,
                                   Vector* S, DenseMatrix* tangent) {
  const auto dim = SpaceDim();
  const auto size = StrainSize();
  equilibrium_->EvalGradient(G_, T, ip);
  const auto J = G_.Det();
  G_.Invert();  // G = F_e^{-1}
  // E = G^T E^ G + (G^T G - 1) / 2, in Mandel form Q e^ + e0.
  SymmetricTensorBasis::CongruenceMatrix(dim, G_, Q_, s1_, s2_, s3_);
  GtG_.SetSize(dim);
  MultAtB(G_, G_, GtG_);
  for (auto i = 0; i < dim; i++) {
    GtG_(i, i) -= 1.0;
  }
  GtG_ *= 0.5;
  MatrixToMandel(GtG_, e0_);
  e_.SetSize(size);
  Q_.Mult(e, e_);
  e_ += e0_;
  const auto energy =
      law_->Evaluate(T, ip, e_, S ? &s_ : nullptr, tangent ? &D_ : nullptr);
  if (S) {
    S->SetSize(size);
    Q_.MultTranspose(s_, *S);
    *S *= J;
  }
  if (tangent) {
    QtD_.SetSize(size);
    MultAtB(Q_, D_, QtD_);
    tangent->SetSize(size);
    Mult(QtD_, Q_, *tangent);
    *tangent *= J;
  }
  return J * energy;
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

// The first Piola-Kirchhoff stress of the law at a reference mapping, and
// that mapping's gradient.
void ReferenceStress(MaterialLaw& law, Diffeomorphism& reference,
                     ElementTransformation& T, const IntegrationPoint& ip,
                     Vector& e, Vector& s, DenseMatrix& S, DenseMatrix& F0,
                     DenseMatrix& P0) {
  const auto dim = law.SpaceDim();
  EvaluateAt(law, reference, T, ip, F0, e, &s, nullptr);
  MandelToMatrix(dim, s, S);
  P0.SetSize(dim);
  Mult(F0, S, P0);
}

}  // namespace

StrainEnergyDensityCoefficient::StrainEnergyDensityCoefficient(
    MaterialLaw& law, Diffeomorphism& map, Diffeomorphism* reference)
    : law_(&law), map_(&map), reference_(reference) {}

real_t StrainEnergyDensityCoefficient::Eval(ElementTransformation& T,
                                            const IntegrationPoint& ip) {
  auto energy = EvaluateAt(*law_, *map_, T, ip, F_, e_, nullptr, nullptr);
  if (reference_) {
    ReferenceStress(*law_, *reference_, T, ip, e_, s_, S_, F0_, P0_);
    const auto dim = law_->SpaceDim();
    for (auto i = 0; i < dim; i++) {
      for (auto j = 0; j < dim; j++) {
        energy -= P0_(i, j) * (F_(i, j) - F0_(i, j));
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
    MaterialLaw& law, Diffeomorphism& map, Diffeomorphism* reference)
    : MatrixCoefficient(law.SpaceDim()),
      law_(&law),
      map_(&map),
      reference_(reference) {}

void FirstPiolaKirchhoffStressCoefficient::Eval(DenseMatrix& P,
                                                ElementTransformation& T,
                                                const IntegrationPoint& ip) {
  const auto dim = law_->SpaceDim();
  EvaluateAt(*law_, *map_, T, ip, F_, e_, &s_, nullptr);
  MandelToMatrix(dim, s_, S_);
  P.SetSize(dim);
  Mult(F_, S_, P);
  if (reference_) {
    ReferenceStress(*law_, *reference_, T, ip, e_, s_, S_, F0_, P0_);
    P -= P0_;
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
