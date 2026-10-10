/**
 * @file rotation.hpp
 * @brief Rotational feedback on a loaded self-gravitating problem: the
 * angular-velocity border of the traditional rotational theory, solved
 * by block elimination over the problem's own tidal machinery.
 */

#pragma once

#include <memory>
#include <vector>

#include "mfem.hpp"
#include "AdGIA/centrifugal.hpp"
#include "AdGIA/mixed_problem.hpp"

namespace AdGIA {

/**
 * @brief Rotational feedback for a loaded mixed self-gravitating
 * problem (Yu, Al-Attar, Syvret & Lloyd 2025, §2.3–2.4 and Appendix A3;
 * doc/planning/sea_level_plan.md, WP2).
 *
 * Surface loading perturbs the inertia tensor; conservation of angular
 * momentum perturbs the rotation, whose centrifugal potential
 * @f$\psi(\omega)@f$ feeds back on the deformation. The coupled system
 * is the problem's own plus a small symmetric border (3 components in
 * 3-D, 1 — the spin rate — in 2-D):
 * @f[
 *   T(\psi_k, x) + \sum_j (D + P)_{kj}\,\omega_j
 *   + \int_{\partial M}\sigma\,\psi_k\,dS = 0 ,
 * @f]
 * with @f$T@f$ the problem's TidalCoupling() pairing (eq. A5 including
 * its fluid-region and interface terms), @f$P@f$ its
 * TidalTidalCoupling() fluid block (eq. A6), @f$D = \mathrm{diag}(C_3 -
 * C_1, C_3 - C_2, -C_3)@f$ the equilibrium inertia matrix (eq. 16;
 * @f$-C_3@f$ alone in 2-D), and @f$\sigma@f$ the applied surface load.
 *
 * The border is solved exactly by block elimination: the unit
 * centrifugal responses @f$Y_k@f$ (one problem solve each, through
 * SetTidalPotential()) are computed once and reused across loads and
 * times by linearity — the coupling block @f$T(\psi_k, Y_j)@f$ is
 * load-independent. A Solve() then costs one base solve, a dense
 * @f$N_\omega\times N_\omega@f$ solve, and one final composed solve
 * that leaves the problem's fields at the full solution. This replaces
 * the successively-refined @f$\omega@f$ iteration the paper mentions
 * (kept here as the test-side cross-check).
 *
 * The equilibrium principal moments are DATA, not derived from the
 * model: the traditional theory assumes @f$C_1 \le C_2 < C_3@f$, which
 * a spherically symmetric model violates (its own moments are equal and
 * the equatorial rows of @f$D@f$ degenerate) — real calculations use
 * observed values (paper §2.4, including the implicit
 * isotropic-inertia caveat for 3-D models).
 *
 * The component claims the problem's tidal-potential slot
 * (SetTidalPotential); a physical tide alongside the feedback is a
 * later composition. After any change to the problem's operator
 * (material, mesh, options), call InvalidateResponses().
 */
class RotationalFeedback {
 public:
  /**
   * @param problem The configured problem (its loads as set by the
   * caller); must outlive the component.
   * @param Omega Equilibrium rotation rate about @f$e_3@f$.
   * @param principal_moments @f$[C_1, C_2, C_3]@f$ in 3-D,
   * @f$[C_3]@f$ in 2-D.
   */
  RotationalFeedback(LinearQuasiStaticMixedSelfGravitatingProblem& problem,
                     mfem::real_t Omega,
                     const mfem::Vector& principal_moments);

  int NumComponents() const { return nw_; }

  /**
   * @brief Solve the problem at time @p t with rotational feedback.
   * @param sigma The applied surface load (the @f$\int\sigma\psi_k@f$
   * data of the border row) and @p surface_marker its boundary marker —
   * the same pair handed to SetSurfaceLoad().
   */
  bool Solve(mfem::real_t t, mfem::Coefficient& sigma,
             const mfem::Array<int>& surface_marker);

  /** @brief The angular-velocity perturbation of the last Solve(). */
  const mfem::Vector& AngularVelocityPerturbation() const { return omega_; }

  /**
   * @brief Residual of the angular-momentum row at the problem's
   * current (composed) state, relative to the largest of its terms —
   * an independent consistency diagnostic, at the solver-tolerance
   * level when all is well.
   */
  mfem::real_t ConsistencyResidual(mfem::Coefficient& sigma,
                                   const mfem::Array<int>& surface_marker);

  /** @brief Forget the cached unit responses (call after the problem's
   * operator changes). */
  void InvalidateResponses() { cached_ = false; }

 private:
  /** @brief @f$\int_{\partial M} \sigma\,\psi_k\,dS@f$. */
  mfem::real_t SurfacePairing(mfem::Coefficient& sigma,
                              const mfem::Array<int>& marker, int k);
  void EnsureResponses(mfem::real_t t);
  mfem::real_t GlobalSum(mfem::real_t v) const;

  LinearQuasiStaticMixedSelfGravitatingProblem& problem_;
  int dim_, nw_;
  mfem::real_t Omega_;
  mfem::DenseMatrix D_;                        ///< inertia matrix + fluid block
  CentrifugalPotential psi_;                   ///< the solved amplitude
  std::vector<std::unique_ptr<CentrifugalPotential>> psi_unit_;
  mfem::DenseMatrix S_;                        ///< D + P + T(psi_k, Y_j)
  bool cached_ = false;
  mfem::Vector omega_;

  // Scalar space on the body mesh for the surface pairings.
  std::unique_ptr<mfem::FiniteElementCollection> sfec_;
  std::unique_ptr<mfem::FiniteElementSpace> sfes_;
  bool parallel_ = false;
#ifdef MFEM_USE_MPI
  MPI_Comm comm_ = MPI_COMM_NULL;
#endif
};

}  // namespace AdGIA
