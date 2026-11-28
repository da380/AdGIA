/**
 * @file detail/ceed_operators.hpp
 * @brief The integrators' forms as libCEED operators, for an mfem built
 * with libCEED (MFEM_USE_CEED) and a device that uses it (e.g.
 * `ceed-cpu`, `ceed-cuda:/gpu/cuda/gen`). libCEED supplies the element
 * restriction and the basis for tensor-product and simplex elements
 * alike; the work at the quadrature points is in src/ceed/.
 */
#pragma once

#include <memory>

#include "mfem.hpp"

namespace AdGIA::detail {

/// True if the library was built with libCEED.
bool HasCeed();

/**
 * @brief The displacement block on @p fes as a libCEED operator on the
 * space's local vectors: the form, and the data at the quadrature points
 * of @p ir, of pa_kernels.hpp (laid out component, then point, then
 * element; they are copied). One element geometry. Aborts if the library
 * was built without libCEED.
 */
std::unique_ptr<mfem::ceed::Operator> MakeDisplacementBlockCeedOperator(
    const mfem::FiniteElementSpace& fes, const mfem::IntegrationRule& ir,
    const mfem::Vector& data, bool isotropic, int extras);

/**
 * @brief The potential block (a diffusion form and a mass term) on the
 * scalar space @p fes as a libCEED operator on the space's local vectors,
 * from the data at the quadrature points of @p ir of pa_kernels.hpp
 * (copied). One element geometry. Aborts if the library was built
 * without libCEED.
 *
 * The geometry reaches libCEED in those data alone. mfem's own libCEED
 * diffusion and mass forms hand it the mesh's nodes and their basis
 * instead, and with `/gpu/cuda/gen` (libCEED 1.1.2) that gave a wrong
 * operator where the rule had fewer points a direction than the mesh
 * nodes (measured: geometry of order 4, space of order 3, 4 points).
 */
std::unique_ptr<mfem::ceed::Operator> MakePotentialBlockCeedOperator(
    const mfem::FiniteElementSpace& fes, const mfem::IntegrationRule& ir,
    const mfem::Vector& data);

}  // namespace AdGIA::detail
