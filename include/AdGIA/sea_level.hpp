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

}  // namespace AdGIA
