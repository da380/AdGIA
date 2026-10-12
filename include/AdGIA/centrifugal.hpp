/**
 * @file centrifugal.hpp
 * @brief The centrifugal potential perturbation of the traditional
 * rotational theory, shared by the rotational-feedback machinery
 * (rotation.hpp, LinearQuasiStaticMixedSelfGravitatingProblem::
 * SetRotation).
 */

#pragma once

#include "mfem.hpp"

namespace AdGIA {

/**
 * @brief @f$\psi = -(\Omega\times x)\cdot(\omega\times x)@f$ with the
 * equilibrium rotation @f$\Omega@f$ about @f$e_3@f$ (out-of-plane in
 * 2-D) and amplitudes @f$\omega@f$: in 3-D @f$\psi = \Omega(\omega_1
 * x_3x_1 + \omega_2 x_3x_2 - \omega_3(x_1^2+x_2^2))@f$, in 2-D
 * @f$\psi = -\Omega\,\omega_1|x|^2@f$.
 */
class CentrifugalPotential : public mfem::Coefficient {
 public:
  CentrifugalPotential(int dim, mfem::real_t Omega)
      : dim_(dim), Omega_(Omega), w_(dim == 2 ? 1 : 3) {
    w_ = 0.0;
  }
  void SetAmplitudes(const mfem::Vector& w) {
    MFEM_ASSERT(w.Size() == w_.Size(), "CentrifugalPotential: wrong "
                                       "number of amplitudes.");
    w_ = w;
  }
  void SetUnit(int k) {
    w_ = 0.0;
    w_[k] = 1.0;
  }
  void SetZero() { w_ = 0.0; }
  int NumComponents() const { return w_.Size(); }

  mfem::real_t Eval(mfem::ElementTransformation& T,
                    const mfem::IntegrationPoint& ip) override {
    T.Transform(ip, x_);
    if (dim_ == 2) {
      return -Omega_ * w_[0] * (x_ * x_);
    }
    return Omega_ * (w_[0] * x_[2] * x_[0] + w_[1] * x_[2] * x_[1] -
                     w_[2] * (x_[0] * x_[0] + x_[1] * x_[1]));
  }

 private:
  int dim_;
  mfem::real_t Omega_;
  mfem::Vector w_;
  mfem::Vector x_;  ///< scratch: the physical point
};

/**
 * @brief @f$\nabla\psi@f$ of CentrifugalPotential, analytically: in 3-D
 * @f$\nabla\psi = \Omega(\omega_1 x_3 - 2\omega_3 x_1,\;
 * \omega_2 x_3 - 2\omega_3 x_2,\; \omega_1 x_1 + \omega_2 x_2)@f$, in
 * 2-D @f$-2\Omega\,\omega_1 x@f$. The referential classes' rotation
 * machinery assembles its volume loads from this (their tidal door
 * takes the physical gradient as analytic data).
 */
class CentrifugalPotentialGradient : public mfem::VectorCoefficient {
 public:
  CentrifugalPotentialGradient(int dim, mfem::real_t Omega)
      : mfem::VectorCoefficient(dim),
        dim_(dim),
        Omega_(Omega),
        w_(dim == 2 ? 1 : 3) {
    w_ = 0.0;
  }
  void SetAmplitudes(const mfem::Vector& w) {
    MFEM_ASSERT(w.Size() == w_.Size(),
                "CentrifugalPotentialGradient: wrong number of amplitudes.");
    w_ = w;
  }
  void SetUnit(int k) {
    w_ = 0.0;
    w_[k] = 1.0;
  }

  void Eval(mfem::Vector& V, mfem::ElementTransformation& T,
            const mfem::IntegrationPoint& ip) override {
    T.Transform(ip, x_);
    V.SetSize(dim_);
    if (dim_ == 2) {
      V = x_;
      V *= -2.0 * Omega_ * w_[0];
      return;
    }
    V[0] = Omega_ * (w_[0] * x_[2] - 2.0 * w_[2] * x_[0]);
    V[1] = Omega_ * (w_[1] * x_[2] - 2.0 * w_[2] * x_[1]);
    V[2] = Omega_ * (w_[0] * x_[0] + w_[1] * x_[1]);
  }

 private:
  int dim_;
  mfem::real_t Omega_;
  mfem::Vector w_;
  mfem::Vector x_;  ///< scratch: the physical point
};

/** @brief The inertia matrix of the traditional theory (Yu et al. 2025,
 * eq. 16): @f$\mathrm{diag}(C_3-C_1, C_3-C_2, -C_3)@f$ in 3-D, the spin
 * row @f$-C_3@f$ alone in 2-D. The moments are data, not derived from
 * the model (paper §2.4). */
inline mfem::DenseMatrix InertiaMatrix(const mfem::Vector& moments) {
  const int nw = moments.Size();
  mfem::DenseMatrix D(nw);
  D = 0.0;
  if (nw == 1) {
    D(0, 0) = -moments[0];
  } else {
    D(0, 0) = moments[2] - moments[0];
    D(1, 1) = moments[2] - moments[1];
    D(2, 2) = -moments[2];
  }
  return D;
}

}  // namespace AdGIA
