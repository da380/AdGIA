/**
 * @file profile_border.hpp
 * @brief The profile border of the Dahlen closure: the closure of an
 * eliminated hydrostatic fluid region in its dual form, in which the one
 * unknown per region is the perturbation of the fluid's pressure profile,
 * a function of the level of the background potential
 * (doc/methods/profile_border.md).
 *
 * With @f$c = \Phi_0@f$ the level, @f$\hat\rho(c)@f$ and
 * @f$\hat\kappa(c)@f$ the fluid's density and bulk modulus, and the
 * stratification
 * @f[
 *   \beta = -\Bigl(\frac{d\hat\rho}{dc} + \frac{\hat\rho^2}{\hat\kappa}\Bigr)
 *   \ge 0,
 * @f]
 * the perturbation of the pressure profile at a fixed level is written
 * @f$\varpi = \hat\rho\,\gamma@f$, with @f$\gamma@f$ the change of the
 * fluid's Bernoulli function. The linearised static problem is the
 * stationarity of the functional of the mixed class with three terms
 * added,
 * @f[
 *   -\tfrac12\,a(\varpi,\varpi)
 *   + \int_\Sigma \varpi\,(\mathbf m\cdot\mathbf u)\,dS
 *   - \int_\Omega \frac{d\varpi}{dc}\,\phi\,dV,
 *   \qquad
 *   a(\varpi,\psi) = \int_\Omega\Bigl[
 *     \frac{\partial_a\varpi\,\partial_a\psi}{\beta}
 *     + \frac{\varpi\,\psi}{\hat\kappa}\Bigr]dV,
 * @f]
 * where @f$\Omega@f$ is the fluid region, @f$\Sigma@f$ its wall with
 * @f$\mathbf m@f$ the outward normal of the solid, and
 * @f$\partial_a\varpi = d\varpi/dc + (\hat\rho/\hat\kappa)\varpi
 * = \hat\rho\,\gamma' - \beta\gamma@f$. In a neutral layer
 * (@f$\beta = 0@f$) @f$\gamma@f$ is constant and the first term of
 * @f$a@f$ is absent; a region that is neutral throughout carries one
 * number, which expresses the conservation of its mass.
 *
 * Every integral is a quadrature over the fluid elements or over the
 * wall, with the profile evaluated at the level of each point. The
 * hypsometry of the level surfaces is not needed, and nothing
 * distinguishes a level wall from a wall that cuts the level surfaces.
 * The level border (level_border.hpp) is the primal form of the same
 * closure, in which the unknown is the cumulative volume beneath each
 * level.
 */

#pragma once

#include <list>
#include <vector>

#include "mfem.hpp"
#include "AdGIA/quasi_static_problem.hpp"

namespace AdGIA {

/** @brief Options of the profile border. */
struct ProfileBorderOptions {
  /// Polynomial degree of @f$\gamma@f$ on each stratified layer. A layer
  /// that spans fewer level bins than this takes the smaller number.
  int degree = 6;
  /// Level bins of the stratification survey that divides each region
  /// into neutral and stratified layers.
  int bins = 48;
  /// A bin is neutral when its mean of @f$\beta\hat\kappa/\hat\rho^2@f$
  /// (the excess of the Bullen parameter over one) is within this of
  /// zero, stratified when above, unstable when below.
  mfem::real_t neutral_tol = 1e-4;
  /// Treat unstable bins as neutral. Off, an unstable bin is refused:
  /// the closure has no solution that a relaxation would reach, and the
  /// model should be made stable first.
  bool unstable_as_neutral = false;
};

/**
 * @brief One fluid region's profile border: the layers, the dense block
 * and the coupling columns. Built by BuildProfileBorders(); the owning
 * problem class scatters the columns into its bordered solve.
 *
 * The region is a stack of layers in the level, each neutral or
 * stratified. @f$\gamma@f$ is continuous, constant on a neutral layer
 * and a polynomial on a stratified one: the unknowns are its values at
 * the ends of the layers (the two ends of a neutral layer sharing one)
 * followed by the interior polynomial modes of the stratified layers.
 * The basis functions are scaled to unit diagonal of @f$a@f$.
 */
struct ProfileBorder {
  struct Layer {
    mfem::real_t c0 = 0.0, c1 = 0.0;  ///< the level interval
    bool neutral = false;
    int degree = 0;    ///< polynomial degree (0 on a neutral layer)
    int v0 = 0, v1 = 0;  ///< unknowns of the lower and upper end values
    int bubble = 0;    ///< first interior mode (degree - 1 of them)
  };
  std::vector<Layer> layers;
  int size = 0;              ///< the number of unknowns
  mfem::Vector scale;        ///< the scaling of each basis function
  mfem::DenseMatrix Db;      ///< the dense block, @f$-a@f$ on the basis
  std::vector<mfem::Vector> phi_cols;  ///< columns on potential true dofs
  std::vector<mfem::Vector> u_cols;    ///< columns on displacement true dofs
  mfem::Vector y;            ///< the last solve's unknowns
  /// The survey: the centre of each level bin and its mean excess of
  /// the Bullen parameter over one (zero in a bin without fluid).
  mfem::Vector bin_level, bin_excess;

  /** @brief The scaled basis functions of @f$\gamma@f$ and their
   * derivatives with respect to the level, at the level @p c (clamped
   * to the region's range). */
  void Basis(mfem::real_t c, mfem::Vector& b, mfem::Vector& db) const;

  /** @brief @f$\gamma@f$ of the last solve at the level @p c. */
  mfem::real_t Gamma(mfem::real_t c) const;
};

/**
 * @brief The class-side fields the builder reads: the solid
 * displacement space, the parent potential space and the background
 * potential on the parent and on the solid shadow. With @p pot0_coef set
 * the background is read from it, as a function of position, in place
 * of the projected fields. Parallelism is read off the spaces.
 */
struct ProfileBorderContext {
  mfem::FiniteElementSpace* fes_u = nullptr;    ///< solid (SubMesh)
  mfem::FiniteElementSpace* fes_pot = nullptr;  ///< potential (parent)
  mfem::GridFunction* pot0 = nullptr;           ///< background, parent
  mfem::GridFunction* pot0_shadow = nullptr;    ///< on the solid shadow
  mfem::Coefficient* pot0_coef = nullptr;       ///< exact levels
  ProfileBorderOptions options;
};

/**
 * @brief Build one ProfileBorder per fluid region: the level range, the
 * survey of the stratification and the layers, the dense block and the
 * potential columns on the same quadrature points (so that the discrete
 * conservation of the region's mass is an identity), and the wall
 * columns.
 *
 * @param fluids One Treatment::Full region per entry, each with its
 * bulk_modulus; @p fluid_markers its parent-attribute markers,
 * @p rho_prime its @f$d\hat\rho/dc@f$ coefficients (one per region).
 */
std::vector<ProfileBorder> BuildProfileBorders(
    const ProfileBorderContext& ctx, const std::vector<FluidRegion>& fluids,
    const std::list<mfem::Array<int>>& fluid_markers,
    const std::vector<mfem::Coefficient*>& rho_prime);

}  // namespace AdGIA
