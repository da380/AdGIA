/**
 * @file sea_level.hpp
 * @brief Surface layer of the sea-level machinery: a boundary SubMesh of
 * the body's exterior surface, scalar fields on it, trace restriction
 * from body (and ancestor-mesh) fields, the sea-level-change
 * postprocessing, and surface/ocean integrals.
 */

#pragma once

#include <memory>
#include <string>

#include "mfem.hpp"
#include "AdGIA/mixed_problem.hpp"

namespace AdGIA {

/**
 * @brief Surface fields and sea-level postprocessing on the body's
 * exterior surface (doc/planning/sea_level_plan.md, WP1).
 *
 * The operator owns a boundary SubMesh of the surface marked on the body
 * mesh, a scalar H1 space on it (sea level is physically continuous),
 * and the named surface fields: the initial sea level @f$SL_0@f$, the
 * ice thickness @f$I@f$, and the sea-level change @f$SL_1@f$. Fields of
 * the body mesh — or of a mesh the body is itself a SubMesh of, such as
 * the potential of the mixed self-gravitating class — are restricted to
 * the surface by (Par)SubMesh transfers. When the body is a SubMesh,
 * the surface is cut from the body's parent mesh (whose interface
 * boundary elements carry the same attributes), because MFEM's
 * CreateFromBoundary cannot take a SubMesh parent; body fields then hop
 * up through the parent and down to the surface.
 *
 * The sea-level change of a solved state is the postprocessing
 * @f[
 *   SL_1 = -\frac{1}{g}\left(\mathbf{u}\cdot\nabla\Phi_0 + \phi + \psi\right)
 *          + c ,
 * @f]
 * (Yu et al. 2025, eq. 34; physicist sign convention: @f$\Phi_0@f$
 * increases outward, @f$g = |\nabla\Phi_0|@f$), with the uniform
 * @f$c@f$ chosen so that the ocean's water mass changes by a prescribed
 * amount (zero by default: mass-conserving redistribution). The
 * combination @f$\mathbf{u}\cdot\nabla\Phi_0 + \phi@f$ and @f$g@f$ are
 * interpolated nodally on the body and transferred; @f$\psi@f$ (a
 * centrifugal or tidal potential) must be evaluable on the surface mesh
 * (an analytic coefficient is).
 *
 * The ocean function @f$C@f$ of the initial state (eq. 29:
 * @f$C = 1@f$ where @f$\rho_w SL_0 - \rho_i I > 0@f$) enters integrals
 * pointwise at quadrature from the surface fields — never through a
 * projection — per the plan's representation split. Until
 * SetInitialState() is called the whole surface is ocean
 * (@f$C \equiv 1@f$), the all-ocean configuration of the rung-0 tests.
 * A shoreline crossing an element makes the integrand discontinuous
 * there; the quadrature error is the documented price of
 * shorelines-as-data.
 *
 * Serial and parallel in one source: the surface SubMesh and all
 * transfers follow the body space's parallelism. In parallel a rank may
 * own no surface elements; every method is safe to call on such ranks
 * (the integrals are collective over the body communicator).
 *
 * WP2 grows this class by the feedback boundary forms and the
 * @f$\Phi_g@f$/@f$\omega@f$ borders; nothing here depends on a problem
 * class.
 */
class SeaLevelOperator {
 public:
  /**
   * @param body_fes A space on the body mesh (typically the
   * displacement space); fixes the mesh and the parallelism. Must
   * outlive the operator.
   * @param surface_marker Boundary marker of the exterior surface,
   * sized to the body mesh's bdr_attributes.Max().
   * @param order Polynomial order of the surface fields; < 0 means the
   * body space's maximum element order (the trace order).
   */
  SeaLevelOperator(mfem::FiniteElementSpace& body_fes,
                   const mfem::Array<int>& surface_marker, int order = -1);

  mfem::Mesh& SurfaceMesh() { return *surf_mesh_; }
  mfem::FiniteElementSpace& SurfaceSpace() { return *sfes_; }
  bool IsParallel() const { return parallel_; }

  // --- data -----------------------------------------------------------------

  /** @brief Water and ice densities entering the ocean function and the
   * mass bookkeeping (defaults 1.0 and 0.9; non-dimensional, like the
   * models). */
  void SetDensities(mfem::real_t rho_water, mfem::real_t rho_ice) {
    rho_water_ = rho_water;
    rho_ice_ = rho_ice;
  }
  mfem::real_t WaterDensity() const { return rho_water_; }
  mfem::real_t IceDensity() const { return rho_ice_; }

  /** @brief Project the initial sea level and ice thickness onto the
   * surface fields; they define the ocean function. The coefficients
   * must be evaluable on the surface mesh (analytic ones are). */
  void SetInitialState(mfem::Coefficient& sea_level,
                       mfem::Coefficient& ice_thickness);
  bool HasInitialState() const { return sl0_ != nullptr; }
  mfem::GridFunction& InitialSeaLevel();
  mfem::GridFunction& IceThickness();

  /** @brief Nodal indicator of the ocean function into @p c (a surface
   * GridFunction) — for display; integrals use the pointwise rule. */
  void OceanFunction(mfem::GridFunction& c) const;

  // --- restriction ----------------------------------------------------------

  /**
   * @brief Trace of the scalar field @p scalar into the surface field
   * @p out. The field may live on the surface mesh (copied), on the
   * body mesh (one transfer), or on a mesh the body is a SubMesh of
   * (two transfers, e.g. the mixed class's potential on its parent
   * mesh). The field's space must have the operator's order.
   */
  void Restrict(const mfem::GridFunction& scalar,
                mfem::GridFunction& out) const;

  // --- sea-level postprocessing ---------------------------------------------

  struct SeaLevelChangeInfo {
    mfem::real_t uniform = 0.0;     ///< the mass-fixing constant c
    mfem::real_t ocean_area = 0.0;  ///< @f$\int_O dS@f$
  };

  /**
   * @brief Compute the sea-level change field of a solved state (see
   * the class comment), into SeaLevelChangeField().
   *
   * @param u Displacement on the body mesh.
   * @param grad_phi0 The equilibrium @f$\nabla\Phi_0@f$ (outward;
   * @f$g = |\nabla\Phi_0|@f$ is formed from it), evaluable on the body.
   * @param phi The Eulerian potential perturbation, on the body mesh or
   * an ancestor (see Restrict()).
   * @param psi Optional centrifugal/tidal potential, evaluable on the
   * surface mesh.
   * @param water_mass_change Prescribed change of the ocean's water
   * mass (0: redistribution only); fixes the uniform constant.
   */
  SeaLevelChangeInfo SeaLevelChange(const mfem::GridFunction& u,
                                    mfem::VectorCoefficient& grad_phi0,
                                    const mfem::GridFunction& phi,
                                    mfem::Coefficient* psi = nullptr,
                                    mfem::real_t water_mass_change = 0.0);
  mfem::GridFunction& SeaLevelChangeField();

  /**
   * @brief The sea-level change of a coupled water-load solve:
   * @f$SL_1 = -(u\cdot\nabla\Phi_0 + \phi + \psi)/g + \Phi_g/g@f$ with
   * the uniform term the solve determined
   * (LinearQuasiStaticMixedSelfGravitatingProblem::UniformPotentialTerm),
   * into SeaLevelChangeField(). No mass bookkeeping here — the solve's
   * mass row already fixed it.
   */
  void SeaLevelChangeFrom(const mfem::GridFunction& u,
                          mfem::VectorCoefficient& grad_phi0,
                          const mfem::GridFunction& phi, mfem::real_t phi_g,
                          mfem::Coefficient* psi = nullptr);

  // --- integrals ------------------------------------------------------------

  /**
   * @brief Write a surface field's nodal values with their coordinates
   * to @p path as CSV (header `x,y[,z],value`; one row per global node,
   * written by rank 0 in parallel) — the exchange format of
   * `postprocess/surface_to_netcdf.py`, which grids the nodes onto a regular
   * lon–lat array for NetCDF / pyshtools / cartopy. Exact nodal data,
   * no resampling on the C++ side.
   */
  void WriteSurfaceField(const mfem::GridFunction& f,
                         const std::string& path) const;

  /** @brief @f$\int_{\partial M} f\,dS@f$ over the surface (global in
   * parallel). */
  mfem::real_t SurfaceIntegral(mfem::Coefficient& f) const;
  /** @brief @f$\int_{\partial M} C f\,dS@f$ with the pointwise ocean
   * function (all-ocean before SetInitialState()). */
  mfem::real_t OceanIntegral(mfem::Coefficient& f) const;
  mfem::real_t OceanArea() const;

 private:
  void TransferPair(const mfem::GridFunction& src,
                    mfem::GridFunction& dst) const;
  mfem::real_t GlobalSum(mfem::real_t v) const;
  mfem::GridFunction& BodyScratch() const;
  mfem::GridFunction& TopScratch() const;

  mfem::FiniteElementSpace* body_fes_ = nullptr;
  mfem::Mesh* body_mesh_ = nullptr;
  // The mesh the surface SubMesh is cut from: the body's own parent when
  // the body is itself a SubMesh (MFEM's CreateFromBoundary cannot take
  // a SubMesh parent), the body mesh otherwise.
  mfem::Mesh* top_mesh_ = nullptr;
  bool parallel_ = false;
#ifdef MFEM_USE_MPI
  MPI_Comm comm_ = MPI_COMM_NULL;
#endif
  int order_ = 0;

  std::unique_ptr<mfem::FiniteElementCollection> fec_;
  std::unique_ptr<mfem::Mesh> surf_mesh_;  ///< SubMesh or ParSubMesh
  std::unique_ptr<mfem::FiniteElementSpace> sfes_;

  // Scalar hop spaces for restrictions and nodal products: on the top
  // mesh always, and on the body when it differs.
  std::unique_ptr<mfem::FiniteElementCollection> body_fec_;
  std::unique_ptr<mfem::FiniteElementSpace> top_sfes_, body_sfes_;
  mutable std::unique_ptr<mfem::GridFunction> body_scratch_, top_scratch_;

  std::unique_ptr<mfem::GridFunction> sl0_, ice_, sl1_;
  mfem::real_t rho_water_ = 1.0, rho_ice_ = 0.9;
};

namespace detail {
class MigratingOceanState;
}

/**
 * @brief Shoreline migration for the coupled water load
 * (doc/planning/sea_level_plan.md, WP5): required machinery for
 * ice-age work, with frozen shorelines as the switchable first-order
 * control.
 *
 * The ocean function of eq. (29) follows the evolving sea level,
 * @f$C = C(\rho_w SL - \rho_i I > 0)@f$ with @f$SL = SL_0 + SL_1@f$
 * and @f$SL_1@f$ read from the problem's own solution (including
 * @f$\psi(\omega)@f$ when SetRotation() is on). The load law (eq. 32)
 * splits as in the frozen-C derivation but with the CURRENT C:
 * the ocean weight is @f$w = \rho_w C/g@f$ and the data load
 * @f$\sigma_d = \rho_w(C - C_0)SL_0 +
 * \rho_i[(1-C)(I_0+\Delta I) - (1-C_0)I_0]@f$, which reduces to the
 * frozen form at @f$C = C_0@f$. Both are owned here as mutable
 * coefficients handed to SetWaterLoad() at construction; each Picard
 * pass updates the state copies, calls
 * RefreshWaterLoad() (boundary-only rebuilds) and re-solves, until the
 * shoreline stops moving (@f$\int|\Delta C|\,dS@f$ below a fraction
 * of the initial ocean area) or the iteration cap. The first pass IS
 * the frozen-C solve, so `max_iterations = 0` is the off-switch.
 *
 * Shoreline migration is strictly second order in the instantaneous
 * perturbation (the plan's Crawford note) — the certificate is that
 * the migrating-minus-frozen difference scales quadratically with the
 * load amplitude — but over ice-age histories the excursions are
 * finite, hence this machinery.
 *
 * The state coefficients (initial sea level, initial ice, ice change)
 * must be evaluable on the body mesh (analytic ones are); the smooth
 * shoreline width is in units of @f$\rho_w \times@f$ length, as in
 * the examples.
 */
class ShorelineMigration {
 public:
  struct Options {
    int max_iterations = 12;  ///< 0: frozen shorelines (the off-switch)
    mfem::real_t tol = 1e-3;  ///< stop: relative SL1 increment
    mfem::real_t shore = 1e-5;  ///< smoothing width of the ocean fraction
    /** @brief Inexact Picard (the quasi-static stepper's pattern): pass
     * k's linear solve runs at relative tolerance inexact * (the
     * previous SL1 increment), clamped to [the problem's RelTol(),
     * inexact_max]; the Picard seed (the frozen-C pass) runs at
     * inexact_max — a fixed-point iteration is self-correcting, so
     * solver digits beyond the next shoreline correction are wasted.
     * The returned state is always polished by one full-tolerance
     * solve whose operator carries the FINAL ocean function, so the
     * endpoint is the problem's RelTol() either way. 0 runs every pass
     * at full tolerance (the reproducible-path mode); the off-switch
     * (max_iterations = 0) always solves at full tolerance. */
    mfem::real_t inexact = 0.1;
    mfem::real_t inexact_max = 1e-3;  ///< loosest per-pass tolerance
  };

  ShorelineMigration(LinearQuasiStaticMixedSelfGravitatingProblem& problem,
                     mfem::VectorCoefficient& grad_phi0,
                     mfem::Coefficient& initial_sea_level,
                     mfem::Coefficient& initial_ice,
                     mfem::Coefficient& ice_change, mfem::real_t rho_water,
                     mfem::real_t rho_ice,
                     const mfem::Array<int>& surface_marker,
                     Options options);
  ~ShorelineMigration();

  /** @brief Solve at time @p t: the frozen-C pass, then Picard on the
   * shoreline until it stops moving. The problem's fields end at the
   * final (migrated) solution. */
  bool Solve(mfem::real_t t);

  int Iterations() const { return iterations_; }
  /** @brief Outer solver iterations summed over every pass of the last
   * Solve(), the polish included — the inexact option's economy shows
   * here. */
  int TotalOuterIterations() const { return outer_its_total_; }
  /** @brief The relative surface-L2 increment of @f$SL_1@f$ between
   * the last two passes — the loop's convergence measure (a direct
   * @f$\Delta C@f$ snapshot cannot resolve a shoreline strip narrower
   * than the mesh). */
  mfem::real_t LastShorelineChange() const { return last_change_; }
  /** @brief The current ocean fraction (for inspection/plotting). */
  mfem::Coefficient& OceanFraction();

 private:
  mfem::real_t ShorelineChange();
  void UpdateState();

  LinearQuasiStaticMixedSelfGravitatingProblem& problem_;
  Options options_;
  mfem::Array<int> marker_;
  std::unique_ptr<detail::MigratingOceanState> state_;
  std::unique_ptr<mfem::Coefficient> weight_, data_, frac_coef_;
  mfem::VectorCoefficient* grad_phi0_ = nullptr;
  std::unique_ptr<mfem::FiniteElementCollection> sfec_;
  std::unique_ptr<mfem::FiniteElementSpace> sfes_, pfes_;
  mfem::real_t last_change_ = 0.0;
  int iterations_ = 0, outer_its_total_ = 0;
  bool parallel_ = false;
#ifdef MFEM_USE_MPI
  MPI_Comm comm_ = MPI_COMM_NULL;
#endif
};

}  // namespace AdGIA
