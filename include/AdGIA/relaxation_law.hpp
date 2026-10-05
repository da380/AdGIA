/**
 * @file relaxation_law.hpp
 * @brief State-dependent relaxation times for the generalised Maxwell body:
 * a pointwise law tau_k = tau_k0 * Factor(state) evaluated at the
 * internal-variable nodes, and the power law of Crawford et al. (2017,
 * Appendix A).
 */

#pragma once

#include "mfem.hpp"

namespace AdGIA {

/**
 * @brief The local state at an internal-variable node, as full symmetric
 * tensor components (unscaled, SymmetricTensorBasis ordering: lower
 * triangle column-major, 11 12 13 22 23 33 in 3-D): the strain, the
 * stress and the branch's own internal variable @f$m_k@f$. With full
 * (anisotropic) internal variables the strain is @f$\varepsilon@f$ and the
 * stress @f$\sigma = C_U\varepsilon - \sum_j C_j m_j@f$. With trace-free
 * (isotropic) ones the strain is @f$d = \mathrm{dev}\,\varepsilon@f$ and
 * the stress is the full deviatoric stress @f$2\mu_\infty d + \sum_j
 * 2\mu_j (d - m_j)@f$ over the branches present at the node (computed as
 * @f$C_U d - \sum_j 2\mu_j m_j@f$), and the dropped diagonal component of
 * each tensor is filled in from the trace-free condition.
 */
struct LocalState {
  int dim = 0;
  mfem::Vector strain, stress, m;

  explicit LocalState(int d);

  /// |dev sigma|, sqrt(T : T) with T = dev sigma, from the stress.
  mfem::real_t DeviatoricStressNorm() const;

  /// Frobenius norm of the deviator of a component vector.
  static mfem::real_t DeviatoricNorm(int dim, const mfem::real_t* comps);
};

/**
 * @brief A pointwise relaxation law @f$\tau_k = \tau_{k0}(x)\,F(x;
 * \varepsilon, \sigma, m_k)@f$. Parameters are coefficients that the
 * viscoelastic operator samples once at the internal nodes and passes back
 * as an array in the order of Parameter(i).
 *
 * A law with IsStateDependent() false is treated as linear: the operator
 * never evaluates it and uses @f$\tau_{k0}@f$, exactly as for a branch
 * without a law (@f$F = 1@f$).
 */
class RelaxationLaw {
 public:
  virtual ~RelaxationLaw() = default;

  virtual bool IsStateDependent() const = 0;
  virtual int NumParameters() const = 0;
  virtual mfem::Coefficient& Parameter(int i) const = 0;

  /** @brief @f$F = \tau_k/\tau_{k0}@f$ from the sampled parameters and the
   * local state. */
  virtual mfem::real_t Factor(const mfem::real_t* params,
                              const LocalState& s) const = 0;

  /** @brief Whether Gradient() is implemented (for adjoint problems). */
  virtual bool HasGradient() const { return false; }

  /** @brief @f$\partial F/\partial\sigma_s@f$ with respect to the stress
   * components (same layout as LocalState::stress). */
  virtual void Gradient(const mfem::real_t* /*params*/, const LocalState& /*s*/,
                        mfem::Vector& /*dF_dstress*/) const {
    MFEM_ABORT("RelaxationLaw::Gradient: not available for this law.");
  }
};

/**
 * @brief Crawford et al. (2017, eq. A11):
 * @f[
 *   \tau_k = \frac{\tau_{k0}}{1 + \gamma_k\,(\|\mathrm{dev}\,\sigma\| /
 *   2\mu_0)^{\,n_k - 1}},
 * @f]
 * a composite Newtonian / power-law body: Newtonian (diffusion creep) at
 * stresses small against @f$\tau_e = 2\mu_0\gamma^{-1/(n-1)}@f$, power law
 * with exponent @f$n@f$ (dislocation creep for @f$n = 3@f$) above it.
 * @f$\gamma@f$, @f$n@f$ and @f$\mu_0@f$ are coefficients (non-owning);
 * @f$\gamma = 0@f$ or @f$n = 1@f$ is linear.
 */
class PowerLawRelaxation : public RelaxationLaw {
 public:
  PowerLawRelaxation(mfem::Coefficient& gamma, mfem::Coefficient& n,
                     mfem::Coefficient& mu0);

  bool IsStateDependent() const override { return true; }
  int NumParameters() const override { return 3; }
  mfem::Coefficient& Parameter(int i) const override;
  mfem::real_t Factor(const mfem::real_t* params,
                      const LocalState& s) const override;
  bool HasGradient() const override { return true; }
  void Gradient(const mfem::real_t* params, const LocalState& s,
                mfem::Vector& dF_dstress) const override;

 private:
  mfem::Coefficient* gamma_;
  mfem::Coefficient* n_;
  mfem::Coefficient* mu0_;
};

}  // namespace AdGIA
