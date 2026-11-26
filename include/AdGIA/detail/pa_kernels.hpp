/**
 * @file detail/pa_kernels.hpp
 * @brief The device kernels of the integrators' partial assembly, as plain
 * functions. Their sources (src/kernels/) are compiled by the device's
 * compiler in a device build of mfem, whose dialect may be older than the
 * library's; this header is what the two sides share, and it includes
 * mfem alone.
 */
#pragma once

#include "mfem.hpp"

namespace AdGIA::detail::pa {

/**
 * @name The elastic tensor form on hexahedra
 * The data at a quadrature point, point by point and then element by
 * element: the 9 entries of the inverse Jacobian (of the mesh, and of the
 * mapping if there is one), @f$\partial\xi_a/\partial x_j@f$ at 3 a + j;
 * then the tensor times the quadrature weight and the Jacobian's
 * determinant, as the upper triangle of its Voigt matrix (ordering 11, 22,
 * 33, 23, 13, 12; entry (i, j) at 9 + PackedIndex(i, j)), or, isotropic,
 * as lambda and mu.
 */
///@{
inline constexpr int kGeneralTensorSize = 21;
inline constexpr int kIsotropicTensorSize = 2;

/// Packed index of the upper triangle of a symmetric 6 x 6 matrix.
constexpr int PackedIndex(int i, int j) {
  return i <= j ? i * 6 - (i * (i - 1)) / 2 + (j - i)
                : j * 6 - (j * (j - 1)) / 2 + (i - j);
}

/// y += A x on element vectors (dofs lexicographic, component, element).
void ApplyElasticTensor(bool isotropic, int ne, const mfem::DofToQuad& maps,
                        const mfem::Vector& data, const mfem::Vector& x,
                        mfem::Vector& y);

/// diag += the diagonal of A, on element vectors.
void DiagonalElasticTensor(bool isotropic, int ne,
                           const mfem::DofToQuad& maps,
                           const mfem::Vector& data, mfem::Vector& diag);
///@}

}  // namespace AdGIA::detail::pa
