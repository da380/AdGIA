/**
 * @file mesh.hpp
 * @brief Mesh queries, serial and parallel: attribute marker arrays, the
 * centroid of a (sub)domain, detection of a spherical boundary and its
 * radius (SphericalMeshHelper), and the communicator of the ranks owning a
 * boundary.
 */

#pragma once

#include <optional>
#include <tuple>

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
 */
std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::Mesh* mesh, const mfem::Array<int>& bdr_marker,
    const mfem::Vector& x0);

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
    const mfem::Vector& x0);

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
 * boundary must be a sphere about it; this is checked by an assert, so only
 * in builds without NDEBUG.
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
 * only these "boundary" ranks.
 *
 * @param mesh Pointer to the mfem::ParMesh object.
 * @param bdr_marker An `mfem::Array<int>` marking which boundary attributes
 * (1 for inclusion, 0 for exclusion) to consider.
 * Collective on the mesh's communicator; the caller owns the new
 * communicator.
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

}  // namespace AdGIA
