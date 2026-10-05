/**
 * @file elastic_tensor.cpp
 * @brief Implementation of SymmetricTensorBasis and the elastic tensor
 * coefficients.
 */

#include "AdGIA/elastic_tensor.hpp"

#include <cmath>
#include <numbers>

namespace AdGIA {

using namespace mfem;

namespace {

inline int Flat(int dim, int i, int j, int k, int l) {
  return ((i * dim + j) * dim + k) * dim + l;
}

}  // namespace

// --- SymmetricTensorBasis ----------------------------------------------------

void SymmetricTensorBasis::Component(int dim, int s, int& j, int& k) {
  for (k = 0; k < dim; k++) {
    const int first = Index(dim, k, k);
    const int last = Index(dim, dim - 1, k);
    if (s >= first && s <= last) {
      j = k + (s - first);
      return;
    }
  }
  MFEM_ABORT("SymmetricTensorBasis::Component: index out of range.");
}

real_t SymmetricTensorBasis::Scale(int j, int k) {
  return j == k ? 1.0 : std::numbers::sqrt2_v<real_t>;
}

int SymmetricTensorBasis::VoigtIndex(int dim, int j, int k) {
  if (j == k) {
    return j;
  }
  if (dim == 2) {
    return 2;
  }
  // 3-D: 23 -> 3, 13 -> 4, 12 -> 5, i.e. the index of the missing axis + 3.
  return 3 + (3 - j - k);
}

void SymmetricTensorBasis::FromVoigt(int dim, const DenseMatrix& Cv,
                                     DenseMatrix& Cm) {
  const int n = Size(dim);
  MFEM_VERIFY(Cv.Height() == n && Cv.Width() == n,
              "SymmetricTensorBasis::FromVoigt: wrong size.");
  Cm.SetSize(n);
  int j, k, l, m;
  for (int s = 0; s < n; s++) {
    Component(dim, s, j, k);
    for (int t = 0; t < n; t++) {
      Component(dim, t, l, m);
      Cm(s, t) = Scale(j, k) * Scale(l, m) *
                 Cv(VoigtIndex(dim, j, k), VoigtIndex(dim, l, m));
    }
  }
}

void SymmetricTensorBasis::ToVoigt(int dim, const DenseMatrix& Cm,
                                   DenseMatrix& Cv) {
  const int n = Size(dim);
  MFEM_VERIFY(Cm.Height() == n && Cm.Width() == n,
              "SymmetricTensorBasis::ToVoigt: wrong size.");
  Cv.SetSize(n);
  int j, k, l, m;
  for (int s = 0; s < n; s++) {
    Component(dim, s, j, k);
    for (int t = 0; t < n; t++) {
      Component(dim, t, l, m);
      Cv(VoigtIndex(dim, j, k), VoigtIndex(dim, l, m)) =
          Cm(s, t) / (Scale(j, k) * Scale(l, m));
    }
  }
}

void SymmetricTensorBasis::Pack(int dim, const real_t* Cijkl, DenseMatrix& Cm) {
  const int n = Size(dim);
  Cm.SetSize(n);
  int j, k, l, m;
  for (int s = 0; s < n; s++) {
    Component(dim, s, j, k);
    for (int t = 0; t < n; t++) {
      Component(dim, t, l, m);
      Cm(s, t) = Scale(j, k) * Scale(l, m) * Cijkl[Flat(dim, j, k, l, m)];
    }
  }
}

void SymmetricTensorBasis::Unpack(int dim, const DenseMatrix& Cm,
                                  real_t* Cijkl) {
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      const int s = Index(dim, i, j);
      for (int k = 0; k < dim; k++) {
        for (int l = 0; l < dim; l++) {
          const int t = Index(dim, k, l);
          Cijkl[Flat(dim, i, j, k, l)] = Cm(s, t) / (Scale(i, j) * Scale(k, l));
        }
      }
    }
  }
}

void SymmetricTensorBasis::Apply(const DenseMatrix& Cm, const Vector& eps,
                                 Vector& sig) {
  const int n = Cm.Height();
  const int dim = n == 6 ? 3 : (n == 3 ? 2 : 1);
  MFEM_VERIFY(Size(dim) == n && eps.Size() == n,
              "SymmetricTensorBasis::Apply: size mismatch.");
  sig.SetSize(n);
  int j, k, l, m;
  for (int s = 0; s < n; s++) {
    Component(dim, s, j, k);
    real_t v = 0.0;
    for (int t = 0; t < n; t++) {
      Component(dim, t, l, m);
      v += Cm(s, t) * Scale(l, m) * eps[t];
    }
    sig[s] = v / Scale(j, k);
  }
}

void SymmetricTensorBasis::VolumetricProjector(int dim, DenseMatrix& P) {
  const int n = Size(dim);
  P.SetSize(n);
  P = 0.0;
  for (int a = 0; a < dim; a++) {
    for (int b = 0; b < dim; b++) {
      P(Index(dim, a, a), Index(dim, b, b)) = 1.0 / dim;
    }
  }
}

void SymmetricTensorBasis::DeviatoricProjector(int dim, DenseMatrix& P) {
  VolumetricProjector(dim, P);
  P.Neg();
  for (int s = 0; s < P.Height(); s++) {
    P(s, s) += 1.0;
  }
}

void SymmetricTensorBasis::RotationMatrix(int dim, const DenseMatrix& R,
                                          DenseMatrix& Q) {
  const int n = Size(dim);
  Q.SetSize(n);
  // Column t of Q: Mandel components of R E_t R^T, with E_t the
  // orthonormal basis tensor of component t.
  DenseMatrix E(dim), RE(dim), RERt(dim);
  int j, k;
  for (int t = 0; t < n; t++) {
    Component(dim, t, j, k);
    E = 0.0;
    if (j == k) {
      E(j, j) = 1.0;
    } else {
      E(j, k) = 1.0 / std::numbers::sqrt2_v<real_t>;
      E(k, j) = 1.0 / std::numbers::sqrt2_v<real_t>;
    }
    Mult(R, E, RE);
    MultABt(RE, R, RERt);
    for (int s = 0; s < n; s++) {
      int a, b;
      Component(dim, s, a, b);
      Q(s, t) = Scale(a, b) * RERt(a, b);
    }
  }
}

void SymmetricTensorBasis::CongruenceMatrix(int dim, const DenseMatrix& F,
                                             DenseMatrix& Q) {
  const int n = Size(dim);
  Q.SetSize(n);
  // Column t of Q: Mandel components of F^T E_t F.
  DenseMatrix E(dim), FtE(dim), FtEF(dim);
  int j, k;
  for (int t = 0; t < n; t++) {
    Component(dim, t, j, k);
    E = 0.0;
    if (j == k) {
      E(j, j) = 1.0;
    } else {
      E(j, k) = 1.0 / std::numbers::sqrt2_v<real_t>;
      E(k, j) = 1.0 / std::numbers::sqrt2_v<real_t>;
    }
    MultAtB(F, E, FtE);
    Mult(FtE, F, FtEF);
    for (int s = 0; s < n; s++) {
      int a, b;
      Component(dim, s, a, b);
      Q(s, t) = Scale(a, b) * FtEF(a, b);
    }
  }
}

RelabelledElasticTensorCoefficient::RelabelledElasticTensorCoefficient(
    int dim, MatrixCoefficient& C_composed, Diffeomorphism& xi)
    : ElasticTensorCoefficient(dim), C_(&C_composed), xi_(&xi) {
  MFEM_VERIFY(C_composed.GetHeight() == SymmetricTensorBasis::Size(dim) &&
                  C_composed.GetWidth() == SymmetricTensorBasis::Size(dim),
              "RelabelledElasticTensorCoefficient: size mismatch.");
}

void RelabelledElasticTensorCoefficient::Eval(DenseMatrix& K,
                                              ElementTransformation& T,
                                              const IntegrationPoint& ip) {
  xi_->EvalGradient(F_, T, ip);
  const auto J = F_.Det();
  F_.Invert();
  SymmetricTensorBasis::CongruenceMatrix(dim_, F_, Qi_);  // eps -> F^-T eps F^-1
  C_->Eval(Cq_, T, ip);
  const int n = SymmetricTensorBasis::Size(dim_);
  tmp_.SetSize(n);
  K.SetSize(n);
  MultAtB(Qi_, Cq_, tmp_);
  Mult(tmp_, Qi_, K);
  K *= J;
}

// --- Isotropic ---------------------------------------------------------------

IsotropicElasticTensorCoefficient::IsotropicElasticTensorCoefficient(
    int dim, Coefficient& lambda, Coefficient& mu)
    : ElasticTensorCoefficient(dim), lambda_or_kappa_(&lambda), mu_(&mu) {}

IsotropicElasticTensorCoefficient
IsotropicElasticTensorCoefficient::FromBulkModulus(int dim, Coefficient& kappa,
                                                   Coefficient& mu) {
  IsotropicElasticTensorCoefficient c(dim, kappa, mu);
  c.use_bulk_ = true;
  return c;
}

void IsotropicElasticTensorCoefficient::Eval(DenseMatrix& K,
                                             ElementTransformation& T,
                                             const IntegrationPoint& ip) {
  const real_t mu = mu_->Eval(T, ip);
  real_t lambda = lambda_or_kappa_->Eval(T, ip);
  if (use_bulk_) {
    lambda -= 2.0 * mu / dim_;
  }
  const int n = SymmetricTensorBasis::Size(dim_);
  K.SetSize(n);
  K = 0.0;
  for (int a = 0; a < dim_; a++) {
    for (int b = 0; b < dim_; b++) {
      K(SymmetricTensorBasis::Index(dim_, a, a),
        SymmetricTensorBasis::Index(dim_, b, b)) = lambda;
    }
  }
  for (int s = 0; s < n; s++) {
    K(s, s) += 2.0 * mu;
  }
}

// --- Transversely isotropic --------------------------------------------------

TransverselyIsotropicElasticTensorCoefficient::
    TransverselyIsotropicElasticTensorCoefficient(
        int dim, Coefficient& A, Coefficient& C, Coefficient& F, Coefficient& L,
        Coefficient& N, VectorCoefficient& axis)
    : ElasticTensorCoefficient(dim),
      A_(&A),
      C_(&C),
      F_(&F),
      L_(&L),
      N_(&N),
      axis_(&axis),
      n_(dim),
      full_(dim * dim * dim * dim) {
  MFEM_VERIFY(axis.GetVDim() == dim,
              "TransverselyIsotropicElasticTensorCoefficient: the axis must "
              "have the space dimension.");
}

TransverselyIsotropicElasticTensorCoefficient
TransverselyIsotropicElasticTensorCoefficient::FromVelocities(
    int dim, Coefficient& rho, Coefficient& vpv, Coefficient& vph,
    Coefficient& vsv, Coefficient& vsh, Coefficient& eta,
    VectorCoefficient& axis) {
  std::vector<std::unique_ptr<Coefficient>> owned;
  auto rho_v2 = [&](Coefficient& v) -> Coefficient& {
    owned.push_back(std::make_unique<PowerCoefficient>(v, 2.0));
    owned.push_back(std::make_unique<ProductCoefficient>(rho, *owned.back()));
    return *owned.back();
  };
  Coefficient& A = rho_v2(vph);
  Coefficient& C = rho_v2(vpv);
  Coefficient& N = rho_v2(vsh);
  Coefficient& L = rho_v2(vsv);
  owned.push_back(std::make_unique<SumCoefficient>(A, L, 1.0, -2.0));
  owned.push_back(std::make_unique<ProductCoefficient>(eta, *owned.back()));
  Coefficient& F = *owned.back();
  TransverselyIsotropicElasticTensorCoefficient c(dim, A, C, F, L, N, axis);
  c.owned_ = std::move(owned);
  return c;
}

void TransverselyIsotropicElasticTensorCoefficient::Build(int dim, real_t A,
                                                          real_t C, real_t F,
                                                          real_t L, real_t N,
                                                          const Vector& n,
                                                          DenseMatrix& Cm) {
  std::vector<real_t> full(dim * dim * dim * dim);
  const real_t c1 = A - 2.0 * N;
  const real_t c2 = N;
  const real_t c3 = F - A + 2.0 * N;
  const real_t c4 = L - N;
  const real_t c5 = A + C - 2.0 * F - 4.0 * L;
  auto d = [](int a, int b) { return a == b ? 1.0 : 0.0; };
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      for (int k = 0; k < dim; k++) {
        for (int l = 0; l < dim; l++) {
          full[Flat(dim, i, j, k, l)] =
              c1 * d(i, j) * d(k, l) +
              c2 * (d(i, k) * d(j, l) + d(i, l) * d(j, k)) +
              c3 * (d(i, j) * n[k] * n[l] + n[i] * n[j] * d(k, l)) +
              c4 * (d(i, k) * n[j] * n[l] + d(i, l) * n[j] * n[k] +
                    d(j, k) * n[i] * n[l] + d(j, l) * n[i] * n[k]) +
              c5 * n[i] * n[j] * n[k] * n[l];
        }
      }
    }
  }
  SymmetricTensorBasis::Pack(dim, full.data(), Cm);
}

void TransverselyIsotropicElasticTensorCoefficient::Eval(
    DenseMatrix& K, ElementTransformation& T, const IntegrationPoint& ip) {
  const real_t A = A_->Eval(T, ip), C = C_->Eval(T, ip), F = F_->Eval(T, ip),
               L = L_->Eval(T, ip), N = N_->Eval(T, ip);
  MFEM_ASSERT(L > 0.0 && N > 0.0 && C > 0.0 && A > N && (A - N) * C > F * F,
              "TransverselyIsotropicElasticTensorCoefficient: the moduli "
              "violate the TI stability conditions.");
  axis_->Eval(n_, T, ip);
  const real_t norm = n_.Norml2();
  MFEM_VERIFY(norm > 0.0,
              "TransverselyIsotropicElasticTensorCoefficient: zero axis.");
  n_ /= norm;
  Build(dim_, A, C, F, L, N, n_, K);
}

// --- Voigt input -------------------------------------------------------------

VoigtElasticTensorCoefficient::VoigtElasticTensorCoefficient(
    int dim, MatrixCoefficient& C_voigt)
    : ElasticTensorCoefficient(dim), Cv_(&C_voigt) {
  MFEM_VERIFY(C_voigt.GetHeight() == SymmetricTensorBasis::Size(dim) &&
                  C_voigt.GetWidth() == SymmetricTensorBasis::Size(dim),
              "VoigtElasticTensorCoefficient: the Voigt matrix must be "
              "n_s x n_s.");
}

void VoigtElasticTensorCoefficient::Eval(DenseMatrix& K,
                                         ElementTransformation& T,
                                         const IntegrationPoint& ip) {
  Cv_->Eval(tmp_, T, ip);
  SymmetricTensorBasis::FromVoigt(dim_, tmp_, K);
}

// --- Rotated frame -----------------------------------------------------------

RotatedElasticTensorCoefficient::RotatedElasticTensorCoefficient(
    MatrixCoefficient& C_local, MatrixCoefficient& R)
    : ElasticTensorCoefficient(R.GetHeight()), C_(&C_local), R_(&R) {
  MFEM_VERIFY(R.GetHeight() == R.GetWidth() &&
                  C_local.GetHeight() == SymmetricTensorBasis::Size(dim_) &&
                  C_local.GetWidth() == SymmetricTensorBasis::Size(dim_),
              "RotatedElasticTensorCoefficient: size mismatch.");
}

void RotatedElasticTensorCoefficient::Eval(DenseMatrix& K,
                                           ElementTransformation& T,
                                           const IntegrationPoint& ip) {
  C_->Eval(Cl_, T, ip);
  R_->Eval(Rm_, T, ip);
  // C' = Q C Q^T with Q the Mandel representation of the rotation; this
  // equals R_ia R_jb R_kc R_ld C_abcd.
  SymmetricTensorBasis::RotationMatrix(dim_, Rm_, Q_);
  const int n = SymmetricTensorBasis::Size(dim_);
  tmp_.SetSize(n);
  K.SetSize(n);
  Mult(Q_, Cl_, tmp_);
  MultABt(tmp_, Q_, K);
}

// --- Deviatoric split --------------------------------------------------------

DeviatoricProjectionElasticTensorCoefficient::
    DeviatoricProjectionElasticTensorCoefficient(int dim, MatrixCoefficient& C,
                                                 bool deviatoric_part)
    : ElasticTensorCoefficient(dim), C_(&C), deviatoric_part_(deviatoric_part) {
  MFEM_VERIFY(C.GetHeight() == SymmetricTensorBasis::Size(dim) &&
                  C.GetWidth() == SymmetricTensorBasis::Size(dim),
              "DeviatoricProjectionElasticTensorCoefficient: size mismatch.");
  SymmetricTensorBasis::DeviatoricProjector(dim, P_);
}

void DeviatoricProjectionElasticTensorCoefficient::Eval(
    DenseMatrix& K, ElementTransformation& T, const IntegrationPoint& ip) {
  C_->Eval(Cq_, T, ip);
  const int n = SymmetricTensorBasis::Size(dim_);
  tmp_.SetSize(n);
  K.SetSize(n);
  Mult(P_, Cq_, tmp_);
  Mult(tmp_, P_, K);
  if (!deviatoric_part_) {
    K.Neg();
    K += Cq_;
  }
}

BareElasticTensorCoefficient::BareElasticTensorCoefficient(
    int dim, MatrixCoefficient& C_eff, Coefficient& p0)
    : ElasticTensorCoefficient(dim), C_(&C_eff), p_(&p0) {
  MFEM_VERIFY(C_eff.GetHeight() == SymmetricTensorBasis::Size(dim) &&
                  C_eff.GetWidth() == SymmetricTensorBasis::Size(dim),
              "BareElasticTensorCoefficient: size mismatch.");
}

void BareElasticTensorCoefficient::Eval(DenseMatrix& K,
                                        ElementTransformation& T,
                                        const IntegrationPoint& ip) {
  C_->Eval(K, T, ip);
  const auto p = p_->Eval(T, ip);
  const int n = SymmetricTensorBasis::Size(dim_);
  // C_eff - p 1^ 1^T + 2 p I in Mandel form (the identity's Mandel vector
  // has entry 1 on the diagonal components and 0 on the shears).
  for (int j = 0; j < dim_; j++) {
    const int sj = SymmetricTensorBasis::Index(dim_, j, j);
    for (int k = 0; k < dim_; k++) {
      K(sj, SymmetricTensorBasis::Index(dim_, k, k)) -= p;
    }
  }
  for (int s = 0; s < n; s++) {
    K(s, s) += 2.0 * p;
  }
}

}  // namespace AdGIA
