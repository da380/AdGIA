#pragma once

#include <cmath>
#include <numbers>

#include "mfem.hpp"

// Exact gravitational potential of a uniform disk (2-D) or ball (3-D) of
// unit density and radius r centred at x, with G = 1 and the sign
// convention ∇²φ = 4πρ (φ < 0 in 3-D, zero at infinity; in 2-D φ is fixed
// only up to a constant and is zero at the centre). Used by poisson_dtn as
// the reference solution.
class UniformSphereSolution {
 private:
  static constexpr mfem::real_t pi = std::numbers::pi_v<mfem::real_t>;

  int dim_;
  mfem::real_t r_;

  mfem::Vector x_;

 public:
  UniformSphereSolution(int dim, const mfem::Vector& x, mfem::real_t r)
      : dim_{dim}, r_{r}, x_{x} {}

  // The potential φ0 of the uniform body. The returned coefficient refers to
  // this object, which must outlive it.
  mfem::FunctionCoefficient Coefficient() const {
    using namespace mfem;
    if (dim_ == 2) {
      return FunctionCoefficient([this](const Vector& x) {
        auto r = x.DistanceTo(x_);
        if (r <= r_) {
          return pi * r * r;
        } else {
          return 2 * pi * r_ * r_ * log(r / r_) + pi * r_ * r_;
        }
      });
    } else {
      return FunctionCoefficient([this](const Vector& x) {
        auto r = x.DistanceTo(x_);
        if (r <= r_) {
          return -2 * pi * (3 * r_ * r_ - r * r) / 3;
        } else {
          return -4 * pi * pow(r_, 3) / (3 * r);
        }
      });
    }
  }

  // The potential perturbation -a·∇φ0 caused by a rigid translation a of
  // the body, written with the offset d = x - x0 (no division by r at the
  // centre): inside, -2π d·a (2-D) or -(4π/3) d·a (3-D); outside,
  // -2π R² d·a / r² (2-D) or -(4π/3) R³ d·a / r³ (3-D). The coefficient
  // keeps its own copy of a.
  mfem::FunctionCoefficient LinearisedCoefficient(const mfem::Vector& a) const {
    using namespace mfem;
    const real_t c = dim_ == 2 ? 2 * pi : 4 * pi / 3;
    return FunctionCoefficient([this, a, c](const Vector& x) {
      Vector d(x);
      d -= x_;
      const real_t r = d.Norml2();
      const real_t da = d * a;
      if (r <= r_) {
        return -c * da;
      }
      return dim_ == 2 ? -c * r_ * r_ * da / (r * r)
                       : -c * std::pow(r_, 3) * da / (r * r * r);
    });
  }
};
