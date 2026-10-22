/**
 * @file level_border.hpp
 * @brief The level border of the Dahlen closure: the one-dimensional
 * bordered extension that closes the eliminated hydrostatic fluid's
 * per-level mass budget (doc/dahlen_closure.tex, "The closure in the
 * cumulative-volume variable" and "Implementation"). The builder is
 * shared by the mixed class (Eulerian potential) and the referential
 * class (which conjugates the columns through its potential glue); the
 * bordered algebra itself runs on the shared elimination engine
 * (bordered.hpp) or the monolithic route.
 */

#pragma once

#include <list>
#include <memory>
#include <vector>

#include "mfem.hpp"
#include "AdGIA/quasi_static_problem.hpp"

namespace AdGIA {

/**
 * @brief One fluid region's level border: the 1-D grid in the level
 * value, the per-element level-mean coefficients, the dense block
 * @f$D_b@f$ (local ordering U | s | nu | lambda), and the coupling
 * columns — the @f$\nu@f$ columns on the potential's true dofs and the
 * wall columns on the solid displacement's true dofs. Built by
 * BuildLevelBorders(); the owning problem class scatters the columns
 * into its bordered solve.
 */
struct LevelBorder {
  int n = 0;                 ///< elements of the 1-D grid (n+1 nodes)
  bool centre = false;       ///< the region contains a centre: U_0 = 0
  mfem::Vector c;            ///< node levels, c[0] < ... < c[n]
  mfem::Vector vol, mass, kap;   ///< per-element volume, mass, kappa-volume
  mfem::DenseMatrix Db;      ///< the 1-D block, local ordering below
  std::vector<mfem::Vector> nu_phi;   ///< the nu columns on potential true dofs
  std::vector<mfem::Vector> nu_u;     ///< glue-conjugated u parts of the nu
                                      ///< columns (referential class only)
  std::vector<mfem::Vector> wall_u;   ///< the wall columns on u true dofs
  std::vector<int> wall_node;         ///< node of each wall (n: outer)
  int nU = 0, oU = 0, os = 0, onu = 0, ol = 0, size = 0;
  mfem::Vector y;            ///< the last solve's border unknowns
  int UIndex(int node) const { return centre ? node - 1 : node; }

  /** @brief Unpack the last solve's border unknowns onto the grid
   * (U_0 = 0 restored at a centre). */
  void Solution(mfem::Vector& c_out, mfem::Vector& U, mfem::Vector& s,
                mfem::Vector& nu, mfem::Vector& lambda) const;
};

/**
 * @brief The class-side fields the builder reads: the solid
 * displacement space, the parent potential space, and the background
 * potential with its solid-shadow copy and the two gradients. For the
 * mixed class the potential is the Eulerian @f$\Phi_0@f$; for the
 * referential class it is @f$\zeta^0@f$, equal to @f$\Phi_0@f$ at the
 * natural reference state the glue requires. Parallelism is read off
 * the spaces.
 */
struct LevelBorderContext {
  mfem::FiniteElementSpace* fes_u = nullptr;    ///< solid (SubMesh)
  mfem::FiniteElementSpace* fes_pot = nullptr;  ///< potential (parent)
  mfem::GridFunction* pot0 = nullptr;           ///< background, parent
  mfem::GridFunction* pot0_shadow = nullptr;    ///< on the solid shadow
  mfem::VectorCoefficient* grad_pot0 = nullptr;         ///< parent
  mfem::VectorCoefficient* grad_pot0_shadow = nullptr;  ///< solid shadow
  int n = 15;  ///< 1-D grid elements (nodes n+1)
};

/**
 * @brief Build one LevelBorder per fluid region: wall classification by
 * the sign of the wall-normal background gravity, the foliation
 * assertion (@f$|\nabla\Phi_0|@f$ bounded away from zero off a centre),
 * the grid uniform in the level, the level-mean coefficients and hat
 * blocks on the fluid quadrature points, the @f$\nu@f$ columns on the
 * SAME quadrature points (the discrete mass balance is then an
 * identity), the wall columns, and the dense block.
 *
 * @param fluids One Treatment::Full region per entry, each with its
 * bulk_modulus; @p fluid_markers its parent-attribute markers,
 * @p rho_prime its @f$\rho_\Phi@f$ coefficients (one per region).
 * @param coefs Receives coefficients the borders reference; must
 * outlive them.
 */
std::vector<LevelBorder> BuildLevelBorders(
    const LevelBorderContext& ctx, const std::vector<FluidRegion>& fluids,
    const std::list<mfem::Array<int>>& fluid_markers,
    const std::vector<mfem::Coefficient*>& rho_prime,
    std::vector<std::unique_ptr<mfem::Coefficient>>& coefs);

}  // namespace AdGIA
