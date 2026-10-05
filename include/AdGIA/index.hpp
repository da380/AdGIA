/**
 * @file index.hpp
 * @brief Component ordering and indexing for vector, matrix and symmetric
 * tensor fields.
 *
 * Two layers. SymmetricComponentOrder fixes, in one place, the library-wide
 * ordering of the independent components of a symmetric second-order tensor
 * and the trace-free convention derived from it; it is dimension-only and
 * constexpr. The Index classes build on it (and on the plain column-major
 * order for full matrices) to map a node and a component to a position in an
 * element or nodal dof vector with Ordering::byNODES, and so also carry the
 * number of degrees of freedom per component.
 *
 * Everything in the library that speaks about "component (j, k) of a
 * symmetric tensor" goes through this header: the integrators and
 * interpolators of bilininteg.hpp, the Mandel-basis algebra of
 * elastic_tensor.hpp, and the internal-variable fields of viscoelastic.hpp.
 */

#pragma once

namespace AdGIA {

/**
 * @brief The library-wide ordering of the independent components of a
 * symmetric second-order tensor.
 *
 * Components are taken from the lower triangle, column-major: (11, 12, 13,
 * 22, 23, 33) in 3-D and (11, 12, 22) in 2-D, so that component (j, k) with
 * j >= k has offset j + k d - k(k+1)/2. A trace-free tensor drops the last
 * diagonal component, which in this ordering is also the last component
 * overall.
 *
 * SymmetricMatrixIndex, TraceFreeSymmetricMatrixIndex and
 * SymmetricTensorBasis all delegate here, so the conventions cannot drift.
 */
struct SymmetricComponentOrder {
  /// Number of independent components, d(d+1)/2.
  static constexpr int Count(int dim) { return dim * (dim + 1) / 2; }

  /// Offset of component (j, k), in either order.
  static constexpr int Offset(int dim, int j, int k) {
    return j < k ? Offset(dim, k, j) : j + k * dim - k * (k + 1) / 2;
  }

  /// Number of independent components of a trace-free tensor, Count - 1.
  static constexpr int TraceFreeCount(int dim) { return Count(dim) - 1; }

  /// Offset of the diagonal component (d-1, d-1) that the trace-free
  /// representation drops; equal to TraceFreeCount(dim).
  static constexpr int TraceFreeDropped(int dim) {
    return Offset(dim, dim - 1, dim - 1);
  }
};

static_assert(SymmetricComponentOrder::TraceFreeDropped(2) ==
              SymmetricComponentOrder::TraceFreeCount(2));
static_assert(SymmetricComponentOrder::TraceFreeDropped(3) ==
              SymmetricComponentOrder::TraceFreeCount(3));

/**
 * @brief Base class for indexing vector, matrix, and tensor fields.
 *
 * This abstract base class provides common functionalities for indexing
 * components within vector, matrix, and tensor fields  defined
 * over a finite element space. It stores the spatial dimension (`dim_`)
 * and the number of degrees of freedom per component (`dof_`).
 */
class Index {
 private:
  int dim_; /**< The spatial dimension (e.g., 2 for 2D, 3 for 3D). */
  int dof_; /**< The number of degrees of freedom per component (e.g., number of
               nodes in an element). */

 public:
  /**
   * @brief Constructor for the Index class.
   * @param dim The spatial dimension of the field.
   * @param dof The number of degrees of freedom associated with each component.
   */
  Index(int dim, int dof) : dim_{dim}, dof_{dof} {}

  /**
   * @brief Returns the spatial dimension of the field.
   * @return The dimension.
   */
  int Dim() const { return dim_; }

  /**
   * @brief Returns the number of degrees of freedom per component.
   * @return The degrees of freedom.
   */
  int Dof() const { return dof_; }

  /**
   * @brief Pure virtual method to return the number of components in the field
   * type.
   * @return The size of one component (e.g., `Dim()` for a vector,
   * `Dim()*Dim()` for a matrix).
   */
  virtual int ComponentSize() const = 0;

  /**
   * @brief Returns the total size of the field, i.e., `dof_ * ComponentSize()`.
   * @return The total size of the field.
   */
  int Size() const { return dof_ * ComponentSize(); }
};

/**
 * @brief Class for indexing vector fields.
 *
 * This class extends the base `Index` class to provide specific indexing
 * for vector fields, where components are typically stored contiguously
 * for each spatial dimension across all degrees of freedom.
 */
class VectorIndex : public Index {
 public:
  /**
   * @brief Constructor for the VectorIndex class.
   * @param dim The spatial dimension of the vector.
   * @param dof The number of degrees of freedom.
   */
  VectorIndex(int dim, int dof) : Index(dim, dof) {}

  /**
   * @brief Returns the offset to the start of the `j`-th component's block of
   * data.
   * @param j The component index (0, 1, ..., Dim()-1).
   * @return The offset.
   */
  int Offset(int j) const { return j * Dof(); }

  /**
   * @brief Overloaded operator to get the global index for the `i`-th node
   * and `j`-th component of the vector field.
   * @param i The node index.
   * @param j The component index.
   * @return The global index.
   */
  int operator()(int i, int j) const { return i + Offset(j); }

  /**
   * @brief Returns the number of components in a vector field, which is equal
   * to `Dim()`.
   * @return The number of components.
   */
  int ComponentSize() const override { return Dim(); }
};

/**
 * @brief Class for indexing matrix fields.
 *
 * This class extends the base `Index` class to provide specific indexing
 * for general (dense) matrix fields, assuming a column-major storage
 * for the components.
 */
class MatrixIndex : public Index {
 public:
  /**
   * @brief Constructor for the MatrixIndex class.
   * @param dim The spatial dimension of the matrix (e.g., for a Dim x Dim
   * matrix).
   * @param dof The number of degrees of freedom.
   */
  MatrixIndex(int dim, int dof) : Index(dim, dof) {}

  /**
   * @brief Returns the offset for the `(j,k)`-th component within the matrix.
   * This assumes column-major ordering: `j + Dim() * k`.
   * @param j The row index.
   * @param k The column index.
   * @return The component offset.
   */
  virtual int ComponentOffset(int j, int k) const { return j + Dim() * k; }

  /**
   * @brief Returns the offset to the `(j,k)`-th component's block of data.
   * @param j The row index.
   * @param k The column index.
   * @return The offset to the block.
   */
  int Offset(int j, int k) const { return ComponentOffset(j, k) * Dof(); }

  /**
   * @brief Overloaded operator to get the global index for the `i`-th node
   * and `(j,k)`-th component of the matrix field.
   * @param i The node index.
   * @param j The row index.
   * @param k The column index.
   * @return The global index.
   */
  int operator()(int i, int j, int k) const { return i + Offset(j, k); }

  /**
   * @brief Returns the number of components in a full matrix field, which is
   * `Dim() * Dim()`.
   * @return The number of components.
   */
  int ComponentSize() const override { return Dim() * Dim(); }
};

/**
 * @brief Class for indexing symmetric matrix fields.
 *
 * This class extends `MatrixIndex` to provide indexing specifically for
 * symmetric matrix fields. It stores only the unique components, typically
 * the lower triangle in a column-major fashion.
 */
class SymmetricMatrixIndex : public MatrixIndex {
 public:
  /**
   * @brief Constructor for the SymmetricMatrixIndex class.
   * @param dim The spatial dimension of the symmetric matrix.
   * @param dof The number of degrees of freedom.
   */
  SymmetricMatrixIndex(int dim, int dof) : MatrixIndex(dim, dof) {}

  /**
   * @brief Returns the offset for the `(j,k)`-th component within the symmetric
   * matrix.
   *
   * This method handles symmetry, ensuring that `ComponentOffset(j,k)` is the
   * same as `ComponentOffset(k,j)`. It calculates the offset assuming a storage
   * order that keeps only the unique elements (e.g., lower triangle in
   * column-major).
   *
   * @param j The row index.
   * @param k The column index.
   * @return The component offset.
   */
  int ComponentOffset(int j, int k) const override {
    return SymmetricComponentOrder::Offset(Dim(), j, k);
  }

  /**
   * @brief Returns the number of unique components in a symmetric matrix, which
   * is `Dim() * (Dim() + 1) / 2`.
   * @return The number of components.
   */
  int ComponentSize() const override {
    return SymmetricComponentOrder::Count(Dim());
  }
};

/**
 * @brief Class for indexing trace-free symmetric matrices.
 *
 * This class extends `SymmetricMatrixIndex`. The indexing is identical
 * to that for symmetric matrices, with the implicit understanding that
 * the final diagonal element (component (d-1, d-1), i.e. 33 in 3D and 22 in
 * 2D, at offset `SymmetricComponentOrder::TraceFreeDropped`) is removed from the basis to
 * enforce the trace-free condition. This removal is not explicitly checked
 * in calls to the indexing or offset functions.
 */
class TraceFreeSymmetricMatrixIndex : public SymmetricMatrixIndex {
 public:
  /**
   * @brief Constructor for the TraceFreeSymmetricMatrixIndex class.
   * @param dim The spatial dimension.
   * @param dof The number of degrees of freedom.
   */
  TraceFreeSymmetricMatrixIndex(int dim, int dof)
      : SymmetricMatrixIndex(dim, dof) {}

  /**
   * @brief Returns the number of components in a trace-free symmetric matrix.
   * This is one less than a full symmetric matrix.
   * @return The number of components.
   */
  int ComponentSize() const override {
    return SymmetricComponentOrder::TraceFreeCount(Dim());
  }
};

}  // namespace AdGIA
