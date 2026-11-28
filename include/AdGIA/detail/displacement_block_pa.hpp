/**
 * @file detail/displacement_block_pa.hpp
 * @brief The partial assembly of the displacement block, shared by the
 * integrators that are it or part of it (ElasticTensorIntegrator,
 * GravitatingElasticityIntegrator): the data at the quadrature points
 * made once on the host, and the form applied from them by the library's
 * kernels (detail/pa_kernels.hpp) or, on a device that uses libCEED, by a
 * libCEED operator (detail/ceed_operators.hpp).
 */
#pragma once

#include <memory>

#include "mfem.hpp"

namespace AdGIA {
class Diffeomorphism;
}

namespace AdGIA::detail {

/// The terms of the block, by their coefficients; null for a term that is
/// absent. The gravity terms take the density, the gradient of the
/// background potential and its Hessian together.
struct DisplacementBlockTerms {
  mfem::MatrixCoefficient* tensor = nullptr;  ///< Mandel form, 6 x 6
  Diffeomorphism* map = nullptr;  ///< pull-back of the elastic form
  mfem::Coefficient* rho = nullptr;
  mfem::VectorCoefficient* grad_phi0 = nullptr;
  mfem::MatrixCoefficient* hessian_phi0 = nullptr;
  mfem::Coefficient* mass = nullptr;
};

class DisplacementBlockPA {
 public:
  /// Makes the data and what applies them. @p ir null: the rule of
  /// ElasticTensorIntegrator's element matrices. @p who names the
  /// integrator in refusals. 3-D, one element geometry; no mapping with
  /// the gravity terms.
  void Assemble(const mfem::FiniteElementSpace& fes,
                const mfem::IntegrationRule* ir,
                const DisplacementBlockTerms& terms, const char* who);

  /// y += A x: on element vectors, or on the space's local vectors where
  /// the device uses libCEED (mfem's convention for an integrator).
  void AddMult(const mfem::Vector& x, mfem::Vector& y) const;

  /// diag += the diagonal of A, on the same vectors.
  void AddDiagonal(mfem::Vector& diag) const;

 private:
  const mfem::DofToQuad* maps_ = nullptr;
  mfem::Vector data_;
  int ne_ = 0;
  bool isotropic_ = false;
  int extras_ = 0;
  std::unique_ptr<mfem::ceed::Operator> ceed_;
};

}  // namespace AdGIA::detail
