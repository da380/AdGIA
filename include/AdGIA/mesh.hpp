/**
 * @file mesh.hpp
 * @brief Mesh queries, serial and parallel: attribute marker arrays, the
 * centroid of a (sub)domain, detection of a spherical boundary and its
 * radius (SphericalMeshHelper), and the communicator of the ranks owning a
 * boundary.
 */

#pragma once

#include <tuple>
#include <vector>

#include "mfem.hpp"

namespace AdGIA {

/**
 * @brief Generates a marker array for the external boundary of a mesh.
 *
 * This function creates an `mfem::Array<int>` suitable for marking
 * all boundary elements that are part of the mesh's external boundary
 * (mfem::Mesh::MarkExternalBoundaries): attributes carried by external
 * boundary elements are marked with \f$1\f$, the others (interior
 * interfaces) with \f$0\f$.
 *
 * @param mesh Pointer to the `mfem::Mesh` object.
 * @return An `mfem::Array<int>` where each entry corresponds to a
 * boundary attribute. The size of the array is `mesh->bdr_attributes.Max()`.
 */
mfem::Array<int> ExternalBoundaryMarker(mfem::Mesh* mesh);

/**
 * @brief Generates a marker array for all domain attributes of a mesh.
 *
 * This function creates an `mfem::Array<int>` that marks all existing
 * domain attributes in the mesh. Useful for selecting all elements in the
 * domain.
 *
 * @param mesh Pointer to the `mfem::Mesh` object.
 * @return An `mfem::Array<int>` where each entry corresponds to a
 * domain attribute. The size of the array is `mesh->attributes.Max()`.
 */
mfem::Array<int> AllDomainsMarker(mfem::Mesh* mesh);

/**
 * @brief Generates a marker array for all boundary attributes of a mesh.
 *
 * This function creates an `mfem::Array<int>` that marks all existing
 * boundary attributes in the mesh. Useful for selecting all boundary elements.
 *
 * @param mesh Pointer to the `mfem::Mesh` object.
 * @return An `mfem::Array<int>` where each entry corresponds to a
 * boundary attribute. The size of the array is `mesh->bdr_attributes.Max()`.
 */
mfem::Array<int> AllBoundariesMarker(mfem::Mesh* mesh);

/**
 * @brief Determines if an indicated boundary is spherical and returns its
 * radius.
 *
 * The test is on the vertices of the marked boundary elements (local to
 * this rank for a ParMesh passed as a Mesh): all must lie at the distance
 * of the first one from @p x0, to a relative tolerance of 1e-6.
 *
 * @param mesh Pointer to the `mfem::Mesh` object.
 * @param bdr_marker An `mfem::Array<int>` marking which boundary attributes
 * (1 for inclusion, 0 for exclusion) to consider.
 * @param x0 The origin (center) from which the radius is measured.
 * @return A `std::tuple` containing:
 * - `int`: Equal to 1 if the boundary is non-empty, 0 otherwise.
 * - `int`: Equal to 1 if the radii of all boundary points are (approximately)
 * equal.
 * - `mfem::real_t`: The radius found (-1 if the boundary is empty).
 *    Meaningful only if the first two return values equal 1.
 *
 * The sphericity verdict compares the maximum relative spread of the
 * boundary nodes' distances from x0 against kSphericalBoundaryRTol;
 * @p deviation, when given, receives that spread. The tolerance must
 * absorb the quadrature error of a computed centroid on coarse meshes
 * while still rejecting a genuinely non-spherical boundary: measured
 * spreads on the house meshes reach 3.6e-6 (the coarse
 * ball_buffer_coarse_3d ball), so 1e-3 keeps two to three orders of margin
 * on both sides.
 */
inline constexpr mfem::real_t kSphericalBoundaryRTol = 1e-3;

std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::Mesh* mesh, const mfem::Array<int>& bdr_marker,
    const mfem::Vector& x0, mfem::real_t* deviation = nullptr);

/**
 * @brief Determines if an indicated boundary is spherical and returns its
 * radius, measured from the coordinate origin.
 *
 * This overload takes `x0 = 0`; pass MeshCentroid() explicitly to measure
 * from the centroid.
 *
 * @param mesh Pointer to the mfem::Mesh object.
 * @param bdr_marker An `mfem::Array<int>` marking which boundary attributes
 * (1 for inclusion, 0 for exclusion) to consider.
 * @return A `std::tuple` containing:
 * - `int`: Equal to 1 if the boundary is non-empty, 0 otherwise.
 * - `int`: Equal to 1 if the radii of all boundary points are (approximately)
 * equal.
 * - `mfem::real_t`: The radius found. Meaningful only if the first two
 * return values equal 1.
 */
std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::Mesh* mesh, const mfem::Array<int>& bdr_marker);

/**
 * @brief Determines if the external boundary is spherical and returns its
 * radius from a given origin.
 *
 * This overload automatically marks the external boundary of the mesh and
 * uses it for the radius computation.
 *
 * @param mesh Pointer to the mfem::Mesh object.
 * @param x0 The origin (center) from which the radius is measured.
 * @return A `std::tuple` containing:
 * - `int`: Equal to 1 if the boundary is non-empty, 0 otherwise.
 * - `int`: Equal to 1 if the radii of all boundary points are (approximately)
 * equal.
 * - `mfem::real_t`: The radius found. Meaningful only if the first two
 * return values equal 1.
 */
std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::Mesh* mesh, const mfem::Vector& x0);

/**
 * @brief Determines if the external boundary is spherical and returns its
 * radius, measured from the coordinate origin.
 *
 * This overload automatically marks the external boundary and takes
 * `x0 = 0`.
 *
 * @param mesh Pointer to the mfem::Mesh object.
 * @return A `std::tuple` containing:
 * - `int`: Equal to 1 if the boundary is non-empty, 0 otherwise.
 * - `int`: Equal to 1 if the radii of all boundary points are (approximately)
 * equal.
 * - `mfem::real_t`: The radius found. Meaningful only if the first two
 * return values equal 1.
 */
std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(mfem::Mesh* mesh);

#ifdef MFEM_USE_MPI

/**
 * @brief Determines if an indicated boundary is spherical and returns its
 * radius for a parallel mesh.
 *
 * This function is the parallel counterpart of the serial
 * `SphericalBoundaryRadius` function, collective on the mesh's communicator.
 * Each rank tests its own boundary elements; the radius returned is that of
 * the lowest rank holding part of the boundary (-1 if none does), and the
 * second return value is 1 only if every rank holding part of it found its
 * part spherical, all with the same radius to a relative tolerance of 1e-6.
 *
 * @param mesh Pointer to the mfem::ParMesh object.
 * @param bdr_marker An `mfem::Array<int>` marking which boundary attributes
 * (1 for inclusion, 0 for exclusion) to consider.
 * @param x0 The origin (center) from which the radius is measured.
 * @return A `std::tuple` containing:
 * - `int`: Equal to 1 if the boundary is non-empty, 0 otherwise.
 * - `int`: Equal to 1 if the radii of all boundary points are (approximately)
 * equal.
 * - `mfem::real_t`: The global radius found. Meaningful only if the first two
 * return values equal 1.
 */
std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::ParMesh* mesh, const mfem::Array<int>& bdr_marker,
    const mfem::Vector& x0, mfem::real_t* deviation = nullptr);

/**
 * @brief Determines if an indicated boundary is spherical and returns its
 * radius for a parallel mesh, measured from the coordinate origin.
 *
 * This parallel overload takes `x0 = 0`; pass MeshCentroid() explicitly to
 * measure from the centroid.
 *
 * @param mesh Pointer to the mfem::ParMesh object.
 * @param bdr_marker An `mfem::Array<int>` marking which boundary attributes
 * (1 for inclusion, 0 for exclusion) to consider.
 * @return A `std::tuple` containing:
 * - `int`: Equal to 1 if the boundary is non-empty, 0 otherwise.
 * - `int`: Equal to 1 if the radii of all boundary points are (approximately)
 * equal.
 * - `mfem::real_t`: The global radius found. Meaningful only if the first two
 * return values equal 1.
 */
std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::ParMesh* mesh, const mfem::Array<int>& bdr_marker);
/**
 * @brief Determines if the external boundary is spherical and returns its
 * radius from a given origin for a parallel mesh.
 *
 * This parallel overload automatically marks the external boundary globally and
 * uses the provided origin.
 *
 * @param mesh Pointer to the mfem::ParMesh object.
 * @param x0 The origin (center) from which the radius is measured.
 * @return A `std::tuple` containing:
 * - `int`: Equal to 1 if the boundary is non-empty, 0 otherwise.
 * - `int`: Equal to 1 if the radii of all boundary points are (approximately)
 * equal.
 * - `mfem::real_t`: The global radius found. Meaningful only if the first two
 * return values equal 1.
 */
std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::ParMesh* mesh, const mfem::Vector& x0);

/**
 * @brief Determines if the external boundary is spherical and returns its
 * radius from the coordinate origin for a parallel mesh.
 *
 * This parallel overload automatically marks the external boundary globally and
 * takes `x0 = 0`.
 *
 * @param mesh Pointer to the mfem::ParMesh object.
 * @return A `std::tuple` containing:
 * - `int`: Equal to 1 if the boundary is non-empty, 0 otherwise.
 * - `int`: Equal to 1 if the radii of all boundary points are (approximately)
 * equal.
 * - `mfem::real_t`: The global radius found. Meaningful only if the first two
 * return values equal 1.
 */
std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(mfem::ParMesh* mesh);
#endif

/**
 * @brief Computes the centroid of a mesh, optionally for a subset of domain
 * attributes.
 *
 * The centroid is computed by integrating the position vector over the
 * specified domain(s) and dividing by the total volume. The integrals are
 * taken as linear forms on an auxiliary L2 space of the given order, which
 * sets the quadrature degree.
 *
 * @param mesh Pointer to the mfem::Mesh object.
 * @param dom_marker An `mfem::Array<int>` marking which domain attributes
 * (1 for inclusion, 0 for exclusion) to consider.
 * @param order The order of the auxiliary L2 space (sets the quadrature
 * degree).
 * @return An `mfem::Vector` representing the coordinates of the computed
 * centroid.
 */
mfem::Vector MeshCentroid(mfem::Mesh* mesh, const mfem::Array<int>& dom_marker,
                          int order = 1);

/**
 * @brief Computes the centroid of the entire mesh.
 *
 * This overload computes the centroid considering all domain attributes.
 *
 * @param mesh Pointer to the mfem::Mesh object.
 * @param order The order of the auxiliary L2 space (sets the quadrature
 * degree).
 * @return An `mfem::Vector` representing the coordinates of the computed
 * centroid.
 */
mfem::Vector MeshCentroid(mfem::Mesh* mesh, int order = 1);

#ifdef MFEM_USE_MPI
/**
 * @brief Computes the global centroid of a parallel mesh, optionally for a
 * subset of domain attributes.
 *
 * This parallel overload computes the centroid by accumulating contributions
 * from all processors.
 *
 * @param mesh Pointer to the mfem::ParMesh object.
 * @param dom_marker An `mfem::Array<int>` marking which domain attributes
 * (1 for inclusion, 0 for exclusion) to consider.
 * @param order The order of the auxiliary L2 space (sets the quadrature
 * degree).
 * @return An `mfem::Vector` representing the global coordinates of the computed
 * centroid.
 */
mfem::Vector MeshCentroid(mfem::ParMesh* mesh,
                          const mfem::Array<int>& dom_marker, int order = 1);

/**
 * @brief Computes the global centroid of the entire parallel mesh.
 *
 * This parallel overload computes the centroid considering all domain
 * attributes across all processors.
 *
 * @param mesh Pointer to the mfem::ParMesh object.
 * @param order The order of the auxiliary L2 space (sets the quadrature
 * degree).
 * @return An `mfem::Vector` representing the global coordinates of the computed
 * centroid.
 */
mfem::Vector MeshCentroid(mfem::ParMesh* mesh, int order = 1);
#endif

/**
 * @brief Struct providing utilities for a mesh with a spherical external
 * boundary.
 *
 * This helper struct encapsulates properties and methods relevant to meshes
 * that are known to have an external boundary that lies on a spherical surface.
 * The centre is taken as the centroid of the whole mesh, and the external
 * boundary must be a sphere about it; this is checked with MFEM_VERIFY.
 */
struct SphericalMeshHelper {
  /** @brief The radius of the spherical external boundary. */
  mfem::real_t bdr_radius_;
  /** @brief The centre of the spherical boundary: the mesh centroid. */
  mfem::Vector x0_;
  /** @brief Marker array identifying the external boundary attributes. */
  mfem::Array<int> bdr_marker_;

  /**
   * @brief Determines and sets the external boundary marker for a serial mesh.
   *
   * This method populates `bdr_marker_`, `bdr_radius_`, and `x0_` by
   * analyzing the provided serial mesh.
   * @param mesh Pointer to the mfem::Mesh object.
   */
  void SetBoundaryMarker(mfem::Mesh* mesh);

#ifdef MFEM_USE_MPI
  /**
   * @brief Determines and sets the external boundary marker for a parallel
   * mesh.
   *
   * This method populates `bdr_marker_`, `bdr_radius_`, and `x0_` by
   * analyzing the provided parallel mesh, performing necessary MPI
   * communication to ensure global consistency.
   * @param mesh Pointer to the mfem::ParMesh object.
   */
  void SetBoundaryMarker(mfem::ParMesh* mesh);
#endif
};

#ifdef MFEM_USE_MPI

/**
 * @brief Splits a communicator, creating a new one only for ranks owning
 * boundary elements.
 *
 * This function identifies all MPI ranks that own at least one boundary
 * element matching the provided boundary marker. It then calls
 * `MPI_Comm_split` to create a new communicator (`bdr_comm`) containing
 * only these "boundary" ranks. Collective on the mesh's communicator; the
 * caller owns the new communicator.
 *
 * @param mesh Pointer to the mfem::ParMesh object.
 * @param bdr_marker An `mfem::Array<int>` marking which boundary attributes
 * (1 for inclusion, 0 for exclusion) to consider.
 *
 * @return A `std::tuple` containing:
 * - `MPI_Comm`: The new communicator. Ranks not owning the boundary will
 * receive `MPI_COMM_NULL`.
 * - `bool`: True if the current rank is part of the new communicator,
 * false otherwise.
 * - `int`: The *global* rank (in the parent communicator) of the root
 * (rank 0) of the new boundary communicator. This is needed for
 * `MPI_Bcast` operations from the boundary group to all ranks.
 */
std::tuple<MPI_Comm, bool, int> SplitBoundaryCommunicator(
    mfem::ParMesh* mesh, const mfem::Array<int>& bdr_marker);

/**
 * @brief Splits a communicator for ranks owning *external* boundary
 * elements.
 *
 * This overload automatically uses `ExternalBoundaryMarker` to identify
 * ranks owning the external boundary.
 *
 * @param mesh Pointer to the mfem::ParMesh object.
 * @return A `std::tuple` containing:
 * - `MPI_Comm`: The new communicator (`MPI_COMM_NULL` for non-boundary
 * ranks).
 * - `bool`: True if the current rank is part of the new communicator.
 * - `int`: The *global* rank of the new communicator's root.
 */
std::tuple<MPI_Comm, bool, int> SplitBoundaryCommunicator(mfem::ParMesh* mesh);

#endif

/**
 * @brief Options for RefineToShoreline.
 *
 * The defaults are a two-pass refinement marking a band of ocean-function
 * values about the flotation threshold, with new nodes snapped back onto
 * the unit sphere. The band matters because shorelines migrate during a
 * run: the refined region must contain every shoreline the run will see,
 * and in practice migration stays within a modest band about the initial
 * shoreline, so one pre-run refinement suffices (no mid-run AMR).
 */
struct ShorelineRefinementOptions {
  /** @brief Marking + refinement passes to perform. */
  int levels = 2;
  /** @brief Element sampling subdivisions (the shaper miniapp's `sd`):
   * each element is sampled on its `subdivisions`-refined lattice. */
  int subdivisions = 2;
  /** @brief Ocean where the ocean function exceeds this. */
  mfem::real_t threshold = 0.5;
  /** @brief Half-width, in ocean-function units, of the marked band about
   * the threshold: an element is marked when its sampled values meet
   * [threshold - band, threshold + band]. Zero marks only elements the
   * shoreline crosses. Ignored when width > 0. */
  mfem::real_t band = 0.2;
  /** @brief GEOMETRIC half-width of the marked band: the target angular
   * distance (radians of arc on the surface sphere) from the shoreline
   * within which elements are marked. When positive it replaces the
   * ocean-function band: an element is marked when it straddles the
   * threshold or when its estimated angular distance to the shoreline —
   * min |C - threshold| over the samples, divided by the element's
   * sampled C-gradient (range over angular extent) — is below width.
   * This is the direct knob: the refined zone's width no longer depends
   * on how steep the ocean function is at the coast.
   *
   * NOTE the division of labour: width (and band) set the marked zone's
   * EXTENT; how FINE the elements get inside it is target_h (or, with
   * target_h unset, the number of levels). A width below the current
   * element size cannot show — marking then collapses to the elements
   * the shoreline crosses. */
  mfem::real_t width = 0.0;
  /** @brief Target element size at the shoreline: when positive, only
   * elements larger than this mark, and passes continue (up to levels,
   * which becomes a safety cap — raise it alongside) until the marked
   * zone's elements are at or below the target. The size measure is
   * mfem::Mesh::GetElementSize's default, the same one the marking
   * tests against, so the stop is exact by construction. */
  mfem::real_t target_h = 0.0;
  /** @brief Radial window: only samples with r_min <= |x| <= r_max count,
   * confining refinement to a shell about the shoreline rather than the
   * full radial cone of the ocean function's transition. */
  mfem::real_t r_min = 0.8;
  /** @brief See r_min. */
  mfem::real_t r_max = 1.2;
  /** @brief Spheres (radii about the origin) that refinement must keep
   * spherical: after each pass, nodes within snap_tol (relative) of a
   * listed radius are projected onto it. Refinement interpolates the
   * existing geometry, so without snapping new surface nodes land on the
   * old facets and the sphere keeps its original faceting. */
  std::vector<mfem::real_t> snap_radii = {1.0};
  /** @brief Relative capture window about each snap radius. Chord
   * midpoints sit at relative depth h^2 / (8 R^2) — about 5e-3 for the
   * house ball meshes — so 2e-2 captures them without grabbing genuinely
   * interior nodes. */
  mfem::real_t snap_tol = 2e-2;
};

/**
 * @brief Conforming local refinement of a mesh toward the shoreline of an
 * ocean function (the MFEM `shaper` miniapp's pattern, specialised).
 *
 * Each pass samples the ocean function on every element's refined lattice
 * (within the radial window) and marks the elements whose values meet the
 * band about the flotation threshold; `Mesh::GeneralRefinement` then
 * refines the marked set — conforming on the house tet meshes, so the
 * downstream submesh and interface machinery is untouched — and nodes
 * near the listed radii are snapped back onto their spheres. Works on a
 * serial `Mesh` or a `ParMesh` alike (the virtual refinement call), and
 * is a PRE-RUN utility: spaces, submeshes and operators are built after.
 *
 * The ocean function is evaluated at 3-D points of the shell, so a
 * surface function C(x-hat) must be supplied as its radial extension
 * (read the direction x/|x| in the coefficient).
 *
 * @param mesh The mesh to refine in place (serial or parallel).
 * @param ocean The ocean-function coefficient, 1 at sea and 0 on land.
 * @param options See ShorelineRefinementOptions.
 * @return Elements refined over all passes; the global count in parallel.
 */
int RefineToShoreline(
    mfem::Mesh& mesh, mfem::Coefficient& ocean,
    const ShorelineRefinementOptions& options = ShorelineRefinementOptions());

}  // namespace AdGIA
