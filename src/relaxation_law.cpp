/**
 * @file relaxation_law.cpp
 * @brief Implementation of LocalState and PowerLawRelaxation.
 */

#include "AdGIA/relaxation_law.hpp"

#include <cmath>

#include "AdGIA/elastic_tensor.hpp"

namespace AdGIA {

using namespace mfem;

LocalState::LocalState(int d)
    : dim(d),
      strain(SymmetricTensorBasis::Size(d)),
      stress(SymmetricTensorBasis::Size(d)),
      m(SymmetricTensorBasis::Size(d)) {
  strain = 0.0;
  stress = 0.0;
  m = 0.0;
}

real_t LocalState::DeviatoricNorm(int dim, const real_t* c) {
  const int ns = SymmetricTensorBasis::Size(dim);
  real_t tr = 0.0;
  for (int j = 0; j < dim; j++) {
    tr += c[SymmetricTensorBasis::Index(dim, j, j)];
  }
  real_t s = 0.0;
  for (int i = 0; i < ns; i++) {
    int j, k;
    SymmetricTensorBasis::Component(dim, i, j, k);
    const real_t v = c[i] - (j == k ? tr / dim : 0.0);
    s += (j == k ? 1.0 : 2.0) * v * v;
  }
  return std::sqrt(s);
}

real_t LocalState::DeviatoricStressNorm() const {
  return DeviatoricNorm(dim, stress.GetData());
}

PowerLawRelaxation::PowerLawRelaxation(Coefficient& gamma, Coefficient& n,
                                       Coefficient& mu0)
    : gamma_(&gamma), n_(&n), mu0_(&mu0) {}

Coefficient& PowerLawRelaxation::Parameter(int i) const {
  switch (i) {
    case 0:
      return *gamma_;
    case 1:
      return *n_;
    default:
      return *mu0_;
  }
}

real_t PowerLawRelaxation::Factor(const real_t* p, const LocalState& s) const {
  const real_t gamma = p[0], n = p[1], mu0 = p[2];
  if (gamma == 0.0 || n == 1.0) {
    return 1.0;
  }
  const real_t x = s.DeviatoricStressNorm() / (2.0 * mu0);
  return 1.0 / (1.0 + gamma * std::pow(x, n - 1.0));
}

void PowerLawRelaxation::Gradient(const real_t* p, const LocalState& s,
                                  Vector& g) const {
  // F = 1 / (1 + gamma x^(n-1)), x = |T| / 2 mu0, T = dev sigma;
  // dF/dsigma_s = dF/dx * dx/d|T| * d|T|/dsigma_s, with d|T|/dsigma_s =
  // w_s T_s / |T| (w_s = 2 off the diagonal; the deviator's derivative is
  // absorbed since T : I = 0).
  const int ns = SymmetricTensorBasis::Size(s.dim);
  g.SetSize(ns);
  g = 0.0;
  const real_t gamma = p[0], n = p[1], mu0 = p[2];
  const real_t T = s.DeviatoricStressNorm();
  if (gamma == 0.0 || n == 1.0 || T <= 0.0) {
    return;
  }
  const real_t x = T / (2.0 * mu0);
  const real_t xp = std::pow(x, n - 1.0);
  const real_t F = 1.0 / (1.0 + gamma * xp);
  const real_t dF_dx = -F * F * gamma * (n - 1.0) * xp / x;
  const real_t dF_dT = dF_dx / (2.0 * mu0);
  real_t tr = 0.0;
  for (int j = 0; j < s.dim; j++) {
    tr += s.stress[SymmetricTensorBasis::Index(s.dim, j, j)];
  }
  for (int i = 0; i < ns; i++) {
    int j, k;
    SymmetricTensorBasis::Component(s.dim, i, j, k);
    const real_t Ts = s.stress[i] - (j == k ? tr / s.dim : 0.0);
    g[i] = dF_dT * (j == k ? 1.0 : 2.0) * Ts / T;
  }
}

}  // namespace AdGIA
