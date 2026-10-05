/**
 * @file spherical_harmonics.cpp
 * @brief Implementation of spherical_harmonics.hpp.
 */

#include "AdGIA/spherical_harmonics.hpp"

#include <cmath>
#include <numbers>

namespace AdGIA {

using namespace mfem;

namespace {
constexpr real_t kInvSqrtPi = std::numbers::inv_sqrtpi_v<real_t>;
constexpr real_t kSqrt2 = std::numbers::sqrt2_v<real_t>;
}  // namespace

// ---------------------------------------------------------------------------
// SurfaceHarmonics

SurfaceHarmonics::SurfaceHarmonics(int dim, int max_degree)
    : dim_(dim), lmax_(max_degree) {
  MFEM_VERIFY(dim == 2 || dim == 3, "SurfaceHarmonics: dim must be 2 or 3.");
  MFEM_VERIFY(max_degree >= 0, "SurfaceHarmonics: negative degree.");
  size_ = dim == 2 ? 2 * lmax_ + 1 : (lmax_ + 1) * (lmax_ + 1);
  degree_.resize(size_);
  order_.resize(size_);
  if (dim == 2) {
    degree_[0] = 0;
    order_[0] = 0;
    for (int k = 1; k <= lmax_; k++) {
      degree_[2 * k - 1] = k;
      order_[2 * k - 1] = k;
      degree_[2 * k] = k;
      order_[2 * k] = -k;
    }
  } else {
    for (int l = 0; l <= lmax_; l++) {
      for (int m = -l; m <= l; m++) {
        degree_[l * l + l + m] = l;
        order_[l * l + l + m] = m;
      }
    }
    sqrt_.SetSize(2 * lmax_ + 2);
    isqrt_.SetSize(2 * lmax_ + 2);
    sqrt_[0] = 0.0;
    isqrt_[0] = 0.0;
    for (int k = 1; k <= 2 * lmax_ + 1; k++) {
      sqrt_[k] = std::sqrt(static_cast<real_t>(k));
      isqrt_[k] = 1 / sqrt_[k];
    }
#ifndef MFEM_THREAD_SAFE
    p_.SetSize(lmax_ + 1);
    pm1_.SetSize(lmax_ + 1);
    cos_.SetSize(lmax_ + 1);
    sin_.SetSize(lmax_ + 1);
#endif
  }
}

std::pair<real_t, real_t> SurfaceHarmonics::RecursionCoefficients(int l,
                                                                  int m) const {
  const real_t alpha =
      sqrt_[2 * l + 1] * sqrt_[2 * l - 1] * isqrt_[l + m] * isqrt_[l - m];
  const real_t beta = l > 1 ? sqrt_[l - 1 + m] * sqrt_[l - 1 - m] *
                                  isqrt_[2 * l - 1] * isqrt_[2 * l - 3]
                            : real_t{0};
  return {alpha, beta};
}

int SurfaceHarmonics::Index(int l, int m) const {
  MFEM_VERIFY(l >= 0 && l <= lmax_ && std::abs(m) <= l,
              "SurfaceHarmonics::Index: (l, m) out of range.");
  if (dim_ == 2) {
    MFEM_VERIFY(l == 0 || std::abs(m) == l,
                "SurfaceHarmonics::Index: in 2-D m = +l (cos) or -l (sin).");
    return l == 0 ? 0 : (m > 0 ? 2 * l - 1 : 2 * l);
  }
  return l * l + l + m;
}

void SurfaceHarmonics::Eval(const Vector& x, Vector& Y) const {
  EvalImpl(x, Y, nullptr);
}

void SurfaceHarmonics::EvalWithGradient(const Vector& x, Vector& Y,
                                        DenseMatrix& gradY) const {
  EvalImpl(x, Y, &gradY);
}

void SurfaceHarmonics::EvalImpl(const Vector& x, Vector& Y,
                                DenseMatrix* gradY) const {
  Y.SetSize(size_);
  if (gradY) {
    gradY->SetSize(dim_, size_);
  }
  if (dim_ == 2) {
    const real_t r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
    const real_t c = r > 0 ? x[0] / r : 1.0, s = r > 0 ? x[1] / r : 0.0;
    Y[0] = kInvSqrtPi / kSqrt2;
    if (gradY) {
      (*gradY)(0, 0) = 0.0;
      (*gradY)(1, 0) = 0.0;
    }
    const real_t fac = kInvSqrtPi;
    real_t ck = 1.0, sk = 0.0;  // cos k theta, sin k theta
    for (int k = 1; k <= lmax_; k++) {
      const real_t ck1 = ck * c - sk * s, sk1 = sk * c + ck * s;
      ck = ck1;
      sk = sk1;
      Y[2 * k - 1] = fac * ck;
      Y[2 * k] = fac * sk;
      if (gradY) {
        // theta_hat = (-s, c).
        const real_t dc = -fac * k * sk, ds = fac * k * ck;
        (*gradY)(0, 2 * k - 1) = -s * dc;
        (*gradY)(1, 2 * k - 1) = c * dc;
        (*gradY)(0, 2 * k) = -s * ds;
        (*gradY)(1, 2 * k) = c * ds;
      }
    }
    return;
  }

#ifdef MFEM_THREAD_SAFE
  Vector p_(lmax_ + 1), pm1_(lmax_ + 1), cos_(lmax_ + 1), sin_(lmax_ + 1);
#endif
  // At the centre the direction is arbitrary; take theta = 0.
  const real_t r = x.Norml2();
  const real_t cos_theta = r > 0 ? x[2] / r : 1.0;
  const real_t rxy = std::sqrt(x[0] * x[0] + x[1] * x[1]);
  const real_t sin_theta = r > 0 ? rxy / r : 0.0;
  const real_t c = rxy > 0 ? x[0] / rxy : 1.0;
  const real_t s = rxy > 0 ? x[1] / rxy : 0.0;

  // d/dtheta of Y and (1/sin theta) d/dphi of Y, on the local frame
  // theta_hat = (cos_theta c, cos_theta s, -sin_theta), phi_hat = (-s, c, 0).
  auto set_gradient = [&](int i, real_t dtheta, real_t dphi) {
    (*gradY)(0, i) = cos_theta * c * dtheta - s * dphi;
    (*gradY)(1, i) = cos_theta * s * dtheta + c * dphi;
    (*gradY)(2, i) = -sin_theta * dtheta;
  };

  cos_[0] = 1.0;
  sin_[0] = 0.0;
  pm1_ = 0.0;
  p_ = 0.0;
  // The sectoral functions by X_{ll} = -sqrt((2l + 1) / (2l)) sin(theta)
  // X_{l-1,l-1} from X_{00} = 1 / sqrt(4 pi); sin(theta) comes from the
  // coordinates, so nothing is lost near the polar axis.
  real_t sectoral = kInvSqrtPi / 2;
  p_[0] = sectoral;
  Y[0] = p_[0];
  if (gradY) {
    set_gradient(0, 0.0, 0.0);
  }
  for (int l = 1; l <= lmax_; l++) {
    cos_[l] = cos_[l - 1] * c - sin_[l - 1] * s;
    sin_[l] = sin_[l - 1] * c + cos_[l - 1] * s;
    // X_{l,m} from X_{l-1,m} (p_) and X_{l-2,m} (pm1_), then X_{l,l}.
    for (int m = 0; m < l; m++) {
      const auto [alpha, beta] = RecursionCoefficients(l, m);
      pm1_[m] = alpha * (cos_theta * p_[m] - beta * pm1_[m]);
    }
    sectoral *= -sqrt_[2 * l + 1] * isqrt_[2 * l] * sin_theta;
    pm1_[l] = sectoral;
    p_[l] = 0.0;
    std::swap(p_, pm1_);
    const int base = l * l + l;
    Y[base] = p_[0];
    for (int m = 1; m <= l; m++) {
      Y[base + m] = kSqrt2 * p_[m] * cos_[m];
      Y[base - m] = kSqrt2 * p_[m] * sin_[m];
    }
    if (!gradY) {
      continue;
    }
    // dX_{lm}/dtheta from X_{l,m+1} and X_{l,m-1} (X_{l,l+1} = 0).
    auto dtheta = [&](int m) {
      if (m == 0) {
        return sqrt_[l] * sqrt_[l + 1] * p_[1];
      }
      const real_t up =
          m < l ? sqrt_[l - m] * sqrt_[l + m + 1] * p_[m + 1] : real_t{0};
      return real_t{0.5} * (up - sqrt_[l + m] * sqrt_[l - m + 1] * p_[m - 1]);
    };
    set_gradient(base, dtheta(0), 0.0);
    for (int m = 1; m <= l; m++) {
      const real_t dX = dtheta(m);
      // X_{lm} / sin theta; on the polar axis only m = 1 survives, with
      // the limit cos_theta dX_{l1}/dtheta.
      const real_t X_over_sin = sin_theta > 0
                                    ? p_[m] / sin_theta
                                    : (m == 1 ? cos_theta * dX : real_t{0});
      set_gradient(base + m, kSqrt2 * dX * cos_[m],
                   -kSqrt2 * m * X_over_sin * sin_[m]);
      set_gradient(base - m, kSqrt2 * dX * sin_[m],
                   kSqrt2 * m * X_over_sin * cos_[m]);
    }
  }
}

// ---------------------------------------------------------------------------
// HarmonicExpansionCoefficient

HarmonicExpansionCoefficient::HarmonicExpansionCoefficient(
    const SurfaceHarmonics& basis, const Vector& coefficients,
    const Vector& centre, real_t radius, bool interior_harmonic)
    : basis_(&basis),
      c_(coefficients),
      x0_(centre),
      R_(radius),
      interior_(interior_harmonic) {
  MFEM_VERIFY(c_.Size() == basis_->Size(),
              "HarmonicExpansionCoefficient: coefficient vector size.");
  if (x0_.Size() == 0) {
    x0_.SetSize(basis_->Dim());
    x0_ = 0.0;
  }
  MFEM_VERIFY(x0_.Size() == basis_->Dim(),
              "HarmonicExpansionCoefficient: centre dimension.");
  MFEM_VERIFY(R_ > 0.0, "HarmonicExpansionCoefficient: radius.");
  x_.SetSize(basis_->Dim());
}

void HarmonicExpansionCoefficient::SetCoefficients(const Vector& c) {
  MFEM_VERIFY(c.Size() == basis_->Size(),
              "HarmonicExpansionCoefficient: coefficient vector size.");
  c_ = c;
}

real_t HarmonicExpansionCoefficient::Eval(ElementTransformation& T,
                                          const IntegrationPoint& ip) {
  T.Transform(ip, x_);
  x_ -= x0_;
  const real_t r = x_.Norml2();
  if (!(r > 0.0)) {
    // Only the constant survives at the centre (interior); a surface
    // field is undefined there, take the constant too.
    return c_[0] * (basis_->Dim() == 2 ? kInvSqrtPi / kSqrt2 : kInvSqrtPi / 2);
  }
  basis_->Eval(x_, Y_);
  real_t f = 0.0;
  if (interior_) {
    const real_t q = r / R_;
    real_t ql = 1.0;
    int l_prev = 0;
    for (int i = 0; i < c_.Size(); i++) {
      const int l = basis_->Degree(i);
      while (l_prev < l) {
        ql *= q;
        l_prev++;
      }
      f += c_[i] * Y_[i] * ql;
    }
  } else {
    f = c_ * Y_;
  }
  return f;
}

// ---------------------------------------------------------------------------
// BoundaryHarmonicCoefficients

BoundaryHarmonicCoefficients::BoundaryHarmonicCoefficients(
    FiniteElementSpace& fes, const Array<int>& bdr_marker, int max_degree,
    Component component, const Vector& centre, real_t radius_tolerance)
    : fes_(&fes),
      dim_(fes.GetMesh()->Dimension()),
      component_(component),
      marker_(bdr_marker),
      x0_(centre),
      basis_(fes.GetMesh()->Dimension(), max_degree) {
  MFEM_VERIFY(marker_.Size() == fes_->GetMesh()->bdr_attributes.Max(),
              "BoundaryHarmonicCoefficients: the marker must be sized to "
              "the mesh's bdr_attributes.Max().");
  if (component_ == Component::Scalar) {
    MFEM_VERIFY(fes_->GetVDim() == 1,
                "BoundaryHarmonicCoefficients: Scalar needs vdim 1.");
  } else {
    MFEM_VERIFY(fes_->GetVDim() == dim_,
                "BoundaryHarmonicCoefficients: Radial and Tangential need "
                "vdim = dim.");
  }
  // int |grad_1 Y_i|^2 over the unit sphere (circle).
  norm_.SetSize(basis_.Size());
  for (int i = 0; i < basis_.Size(); i++) {
    const int l = basis_.Degree(i);
    norm_[i] = dim_ == 2 ? l * l : l * (l + 1);
  }
  if (x0_.Size() == 0) {
    x0_.SetSize(dim_);
    x0_ = 0.0;
  }
  MFEM_VERIFY(x0_.Size() == dim_, "BoundaryHarmonicCoefficients: centre.");
#ifdef MFEM_USE_MPI
  if (auto* pfes = dynamic_cast<ParFiniteElementSpace*>(fes_)) {
    parallel_ = true;
    comm_ = pfes->GetComm();
  }
#endif
  MeasureRadius(radius_tolerance);
  Assemble();
}

template <class F>
void BoundaryHarmonicCoefficients::ForEachQuadraturePoint(F visit) const {
  Mesh* mesh = fes_->GetMesh();
  Vector x(dim_), Y;
  DenseMatrix gradY;
  const bool tangential = component_ == Component::Tangential;
  const real_t scale = std::pow(R_, 1 - dim_);
  for (int b = 0; b < fes_->GetNBE(); b++) {
    if (!marker_[mesh->GetBdrAttribute(b) - 1]) {
      continue;
    }
    const FiniteElement* fe = fes_->GetBE(b);
    ElementTransformation* T = fes_->GetBdrElementTransformation(b);
    const int order = 2 * fe->GetOrder() + T->OrderW() + basis_.MaxDegree();
    const IntegrationRule& ir = IntRules.Get(fe->GetGeomType(), order);
    for (int q = 0; q < ir.GetNPoints(); q++) {
      const IntegrationPoint& ip = ir.IntPoint(q);
      T->SetIntPoint(&ip);
      T->Transform(ip, x);
      x -= x0_;
      if (tangential) {
        // grad_1 Y_i over its norm, so that the coefficients are those of
        // the expansion in grad_1 Y_i; nothing at degree zero.
        basis_.EvalWithGradient(x, Y, gradY);
        for (int i = 0; i < gradY.Width(); i++) {
          const real_t s = norm_[i] > 0.0 ? 1.0 / norm_[i] : 0.0;
          for (int c = 0; c < dim_; c++) {
            gradY(c, i) *= s;
          }
        }
      } else {
        basis_.Eval(x, Y);
      }
      visit(b, *fe, *T, ip, x, Y, gradY, ip.weight * T->Weight() * scale);
    }
  }
}

void BoundaryHarmonicCoefficients::MeasureRadius(real_t tolerance) {
  // Area-weighted mean radius and the spread about it.
  Mesh* mesh = fes_->GetMesh();
  Vector x(dim_);
  real_t sums[2] = {0.0, 0.0};
  real_t rmin = INFINITY, rmax = 0.0;
  for (int b = 0; b < fes_->GetNBE(); b++) {
    if (!marker_[mesh->GetBdrAttribute(b) - 1]) {
      continue;
    }
    ElementTransformation* T = fes_->GetBdrElementTransformation(b);
    const IntegrationRule& ir =
        IntRules.Get(T->GetGeometryType(), 2 * T->OrderW() + 2);
    for (int q = 0; q < ir.GetNPoints(); q++) {
      const IntegrationPoint& ip = ir.IntPoint(q);
      T->SetIntPoint(&ip);
      T->Transform(ip, x);
      x -= x0_;
      const real_t r = x.Norml2(), w = ip.weight * T->Weight();
      sums[0] += w;
      sums[1] += w * r;
      rmin = std::min(rmin, r);
      rmax = std::max(rmax, r);
    }
  }
#ifdef MFEM_USE_MPI
  if (parallel_) {
    MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                  comm_);
    MPI_Allreduce(MPI_IN_PLACE, &rmin, 1, MPITypeMap<real_t>::mpi_type, MPI_MIN,
                  comm_);
    MPI_Allreduce(MPI_IN_PLACE, &rmax, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                  comm_);
  }
#endif
  MFEM_VERIFY(sums[0] > 0.0,
              "BoundaryHarmonicCoefficients: the marked boundary is empty.");
  R_ = sums[1] / sums[0];
  MFEM_VERIFY(rmax - rmin <= tolerance * R_,
              "BoundaryHarmonicCoefficients: the marked boundary is not a "
              "sphere about the centre (radii from "
                  << rmin << " to " << rmax << ").");
}

void BoundaryHarmonicCoefficients::Assemble() {
  const int n = basis_.Size();
  M_ = SparseMatrix(fes_->GetVSize(), n);
  Array<int> vdofs, cols(n);
  for (int i = 0; i < n; i++) {
    cols[i] = i;
  }
  Vector shape;
  DenseMatrix elmat;
  int current = -1;
  auto flush = [&]() {
    if (current >= 0) {
      M_.AddSubMatrix(vdofs, cols, elmat);
    }
  };
  ForEachQuadraturePoint([&](int b, const FiniteElement& fe,
                             ElementTransformation& T,
                             const IntegrationPoint& ip, const Vector& x,
                             const Vector& Y, const DenseMatrix& gradY,
                             real_t w) {
    if (b != current) {
      flush();
      current = b;
      fes_->GetBdrElementVDofs(b, vdofs);
      elmat.SetSize(vdofs.Size(), n);
      elmat = 0.0;
      shape.SetSize(fe.GetDof());
    }
    fe.CalcShape(ip, shape);
    const int dof = fe.GetDof();
    if (component_ == Component::Scalar) {
      for (int j = 0; j < dof; j++) {
        for (int i = 0; i < n; i++) {
          elmat(j, i) += w * shape[j] * Y[i];
        }
      }
    } else if (component_ == Component::Tangential) {
      for (int c = 0; c < dim_; c++) {
        for (int j = 0; j < dof; j++) {
          for (int i = 0; i < n; i++) {
            elmat(c * dof + j, i) += w * shape[j] * gradY(c, i);
          }
        }
      }
    } else {
      const real_t r = x.Norml2();
      for (int c = 0; c < dim_; c++) {
        const real_t nc = x[c] / r;
        for (int j = 0; j < dof; j++) {
          for (int i = 0; i < n; i++) {
            elmat(c * dof + j, i) += w * nc * shape[j] * Y[i];
          }
        }
      }
    }
  });
  flush();
  M_.Finalize();
}

void BoundaryHarmonicCoefficients::Reduce(Vector& c) const {
#ifdef MFEM_USE_MPI
  if (parallel_) {
    MPI_Allreduce(MPI_IN_PLACE, c.GetData(), c.Size(),
                  MPITypeMap<real_t>::mpi_type, MPI_SUM, comm_);
  }
#endif
}

// Hidden from Doxygen, which cannot match these overloads to their
// declarations: the unqualified mfem types here do not resolve without the
// MFEM headers in its input.
/// @cond
void BoundaryHarmonicCoefficients::Coefficients(const GridFunction& f,
                                                Vector& c) const {
  MFEM_VERIFY(f.Size() == M_.Height(),
              "BoundaryHarmonicCoefficients: the field is not on the space.");
  c.SetSize(Size());
  M_.MultTranspose(f, c);
  Reduce(c);
}

void BoundaryHarmonicCoefficients::Coefficients(Coefficient& f,
                                                Vector& c) const {
  MFEM_VERIFY(component_ == Component::Scalar,
              "BoundaryHarmonicCoefficients: scalar coefficient on a Radial "
              "operator.");
  c.SetSize(Size());
  c = 0.0;
  ForEachQuadraturePoint(
      [&](int, const FiniteElement&, ElementTransformation& T,
          const IntegrationPoint& ip, const Vector&, const Vector& Y,
          const DenseMatrix&, real_t w) { c.Add(w * f.Eval(T, ip), Y); });
  Reduce(c);
}

void BoundaryHarmonicCoefficients::Coefficients(VectorCoefficient& f,
                                                Vector& c) const {
  MFEM_VERIFY(component_ != Component::Scalar,
              "BoundaryHarmonicCoefficients: vector coefficient on a Scalar "
              "operator.");
  c.SetSize(Size());
  c = 0.0;
  Vector v(dim_);
  ForEachQuadraturePoint([&](int, const FiniteElement&,
                             ElementTransformation& T,
                             const IntegrationPoint& ip, const Vector& x,
                             const Vector& Y, const DenseMatrix& gradY,
                             real_t w) {
    f.Eval(v, T, ip);
    if (component_ == Component::Tangential) {
      gradY.AddMultTranspose(v, c, w);
    } else {
      c.Add(w * (v * x) / x.Norml2(), Y);
    }
  });
  Reduce(c);
}

/// @endcond

void BoundaryHarmonicCoefficients::LoadVector(const Vector& c,
                                              Vector& b) const {
  MFEM_VERIFY(c.Size() == Size(), "BoundaryHarmonicCoefficients: size.");
  MFEM_VERIFY(component_ != Component::Tangential,
              "BoundaryHarmonicCoefficients: no load vector for the "
              "Tangential component.");
  b.SetSize(M_.Height());
  M_.Mult(c, b);
  b *= std::pow(R_, dim_ - 1);
}

std::unique_ptr<HarmonicExpansionCoefficient>
BoundaryHarmonicCoefficients::Expansion(const Vector& c,
                                        bool interior_harmonic) const {
  return std::make_unique<HarmonicExpansionCoefficient>(basis_, c, x0_, R_,
                                                        interior_harmonic);
}

}  // namespace AdGIA
