/**
 * @file gradient_coupling.hpp
 * @brief The coupling of a scalar field's gradient to a vector field,
 * applied with no matrix.
 */
#pragma once

#include "mfem.hpp"

namespace AdGIA {

/**
 * @brief The form
 * @f$(\phi, v) \mapsto \int c\,\nabla\phi\cdot v@f$ between a scalar H1
 * space and a vector H1 space on one mesh, as an operator with no matrix:
 * Mult() takes the scalar's local vector to the vector's, MultTranspose()
 * the other way.
 *
 * It is the matrix of mfem's GradientIntegrator (trial the scalar, test
 * the vector) on the same integration rule, which is the default here
 * too. The coefficient and the geometry are evaluated once at the
 * quadrature points, on the host; the form is applied element by element
 * by kernels of the library's own on the configured mfem::Device,
 * sum-factorised where both spaces have a tensor basis (hexahedra) and
 * dense in the element otherwise. 3-D, one element geometry; the two
 * spaces may differ in order.
 *
 * The spaces and the coefficient are used at construction and the spaces
 * afterwards: the spaces must outlive the operator.
 */
class GradientCouplingOperator : public mfem::Operator {
 public:
  GradientCouplingOperator(mfem::FiniteElementSpace& scalar_fes,
                           mfem::FiniteElementSpace& vector_fes,
                           mfem::Coefficient& c,
                           const mfem::IntegrationRule* ir = nullptr);

  /// u = C phi, on local vectors.
  void Mult(const mfem::Vector& phi, mfem::Vector& u) const override;

  /// phi = C^T u, on local vectors.
  void MultTranspose(const mfem::Vector& u, mfem::Vector& phi) const override;

 private:
  const mfem::Operator* scalar_restriction_ = nullptr;
  const mfem::Operator* vector_restriction_ = nullptr;
  const mfem::DofToQuad* scalar_maps_ = nullptr;
  const mfem::DofToQuad* vector_maps_ = nullptr;
  mfem::Vector data_;
  int ne_ = 0;
  mutable mfem::Vector scalar_e_, vector_e_;
};

}  // namespace AdGIA
