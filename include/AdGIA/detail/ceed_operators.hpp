/**
 * @file detail/ceed_operators.hpp
 * @brief The integrators' forms as libCEED operators, for an mfem built
 * with libCEED (MFEM_USE_CEED) and a device that uses it (e.g.
 * `ceed-cpu`, `ceed-cuda:/gpu/cuda/gen`). libCEED supplies the element
 * restriction and the basis for tensor-product and simplex elements
 * alike; the work at the quadrature points is in src/ceed/.
 *
 * Each operator acts on the space's local vectors, takes the data at the
 * quadrature points of the rule laid out component, then point, then
 * element, and copies them. One element geometry. The functions abort if
 * the library was built without libCEED.
 */
#pragma once

#include <memory>

#include "mfem.hpp"

namespace AdGIA::detail {

/// True if the library was built with libCEED.
bool HasCeed();

/// The elastic tensor form; the data of pa_kernels.hpp.
std::unique_ptr<mfem::ceed::Operator> MakeElasticTensorCeedOperator(
    const mfem::FiniteElementSpace& fes, const mfem::IntegrationRule& ir,
    const mfem::Vector& data, bool isotropic);

/// Data at a point of the gravity and mass terms that follow the elastic
/// tensor's in GravitatingElasticityIntegrator's: g (3), G by rows (9),
/// s (1).
inline constexpr int kGravityDataSize = 13;

/// The displacement block of the self-gravitating problem
/// (GravitatingElasticityIntegrator): the data of the elastic tensor
/// form followed by kGravityDataSize entries (src/ceed/
/// gravitating_elasticity_qf.h).
std::unique_ptr<mfem::ceed::Operator> MakeGravitatingElasticityCeedOperator(
    const mfem::FiniteElementSpace& fes, const mfem::IntegrationRule& ir,
    const mfem::Vector& data, bool isotropic);

}  // namespace AdGIA::detail
