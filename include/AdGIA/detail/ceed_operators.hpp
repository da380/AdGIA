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

}  // namespace AdGIA::detail
