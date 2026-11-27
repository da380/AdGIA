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
 * @name The displacement block
 * The elastic tensor form, alone or with a mass term or with the gravity
 * terms of the self-gravitating problem and a mass term
 * (GravitatingElasticityIntegrator says which terms), applied element by
 * element from data at the quadrature points, in 3-D.
 *
 * The data at a point, point by point and then element by element:
 * - the 9 entries of the inverse Jacobian (of the mesh, and of the mapping
 *   if there is one), @f$\partial\xi_a/\partial x_j@f$ at 3 a + j;
 * - the tensor times the quadrature weight and the Jacobian's
 *   determinant, as the upper triangle of its Voigt matrix (ordering 11,
 *   22, 33, 23, 13, 12; entry (i, j) at PackedIndex(i, j)), or,
 *   isotropic, as lambda and mu;
 * - the extras, weighted likewise: none; or the mass term's coefficient
 *   s; or @f$g = \tfrac12\rho\nabla\Phi^0@f$ (3),
 *   @f$G = \tfrac12\rho\nabla\nabla\Phi^0@f$ by rows (9) and s.
 *
 * The element's basis is given by @p maps: in mfem's TENSOR mode the
 * kernels are sum-factorised (hexahedra, dofs lexicographic); in FULL
 * mode they are dense in the element (any element, dofs in its own
 * order). Element vectors are dof, then component, then element.
 */
///@{
inline constexpr int kGeneralTensorSize = 21;
inline constexpr int kIsotropicTensorSize = 2;
inline constexpr int kNoExtras = 0;
inline constexpr int kMassExtras = 1;
inline constexpr int kGravityExtras = 13;

/// Packed index of the upper triangle of a symmetric 6 x 6 matrix.
constexpr int PackedIndex(int i, int j) {
  return i <= j ? i * 6 - (i * (i - 1)) / 2 + (j - i)
                : j * 6 - (j * (j - 1)) / 2 + (i - j);
}

/// The data at a point.
constexpr int DataSize(bool isotropic, int extras) {
  return 9 + (isotropic ? kIsotropicTensorSize : kGeneralTensorSize) + extras;
}

/// y += A x on element vectors.
void ApplyDisplacementBlock(bool isotropic, int extras, int ne,
                            const mfem::DofToQuad& maps,
                            const mfem::Vector& data, const mfem::Vector& x,
                            mfem::Vector& y);

/// diag += the diagonal of A, on element vectors.
void DiagonalDisplacementBlock(bool isotropic, int extras, int ne,
                               const mfem::DofToQuad& maps,
                               const mfem::Vector& data, mfem::Vector& diag);
///@}

/**
 * @name The potential block
 * A diffusion form and a mass term on a scalar space,
 * @f$(\phi,\psi)\mapsto\int k\,\nabla\phi\cdot\nabla\psi + s\,\phi\psi@f$
 * (DiffusionMassIntegrator), in 3-D, with the same two kinds of kernel.
 * The data at a point: the symmetric matrix
 * @f$k\,w\,J^{-1}J^{-T}@f$ in the reference directions (entries 00, 01,
 * 02, 11, 12, 22), then @f$s\,w@f$, with w the quadrature weight times
 * the Jacobian's determinant.
 */
///@{
inline constexpr int kPotentialDataSize = 7;

/// y += A x on element vectors.
void ApplyPotentialBlock(int ne, const mfem::DofToQuad& maps,
                         const mfem::Vector& data, const mfem::Vector& x,
                         mfem::Vector& y);

/// diag += the diagonal of A, on element vectors.
void DiagonalPotentialBlock(int ne, const mfem::DofToQuad& maps,
                            const mfem::Vector& data, mfem::Vector& diag);
///@}

/**
 * @name The coupling of a scalar's gradient to a vector's value
 * @f$(\\phi, v)\\mapsto\\int c\\,\\nabla\\phi\\cdot v@f$
 * (GradientCouplingOperator) and its transpose, in 3-D, between a scalar
 * space and a vector space that share the quadrature rule and may differ
 * in order; sum-factorised where both maps are TENSOR, dense in the
 * element otherwise. The data at a point: @f$c\\,w\\,J^{-1}@f$, entry
 * (a, j) at 3 a + j.
 */
///@{
inline constexpr int kCouplingDataSize = 9;

/// y += C x (the scalar's element vector to the vector's), or with
/// @p transpose y += C^T x (the vector's to the scalar's).
void ApplyGradientCoupling(bool transpose, int ne,
                           const mfem::DofToQuad& scalar_maps,
                           const mfem::DofToQuad& vector_maps,
                           const mfem::Vector& data, const mfem::Vector& x,
                           mfem::Vector& y);
///@}

}  // namespace AdGIA::detail::pa
