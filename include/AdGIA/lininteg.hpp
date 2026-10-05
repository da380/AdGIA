/**
 * @file lininteg.hpp
 * @brief Linear form integrators: the pairing of a matrix coefficient with
 * the deformation gradient of a vector test field.
 */

#pragma once

#include "mfem.hpp"
#include "AdGIA/coefficient.hpp"

namespace AdGIA {

/**
 * @brief A LinearFormIntegrator that evaluates an integral involving a matrix
 * coefficient and the deformation gradient of a vector field.
 *
 * This integrator computes the integral:
 * \f[
 * \bvec{u} \mapsto \int_{\Omega} \bvec{m} : \deriv \bvec{u} \dd x =
 * \int_{\Omega} m_{ij} \frac{\partial u_{i}}{\partial x_{j}} \dd x,
 * \f]
 * where \f$\Omega\f$ is the computational domain, \f$\bvec{u}\f$ is a vector
 * field,
 * \f$\deriv \bvec{u}\f$ is its deformation gradient with components
 * \f$\frac{\partial u_{i}}{\partial x_{j}}\f$, and \f$\bvec{m}\f$ is a given
 * matrix coefficient with components \f$m_{ij}\f$.
 *
 * It is assumed that \f$\bvec{u}\f$ is defined on a product of scalar finite
 * element spaces for which the gradient operator is well-defined.
 *
 * It is also assumed that the matrix coefficient \f$\bvec{m}\f$ is square with
 * its dimension equal to the spatial dimension of the finite-element space.
 *
 * A typical use is a prescribed-stress (stress-glut) source. Given a
 * MatrixDeltaCoefficient it follows MFEM's
 * delta-function machinery instead (mfem::DeltaLFIntegrator): the linear
 * form assembles the point source
 * \f$\bvec{u} \mapsto s\, m_{ij}\, \partial_j u_i(\bvec{x}_c)\f$
 * at the delta's center \f$\bvec{x}_c\f$ with its scale \f$s\f$ — a
 * moment-tensor point source (the equivalent force of the seismic moment
 * tensor \f$\bvec{m}\f$), e.g. for post-seismic deformation studies.
 */
class DomainLFDeformationGradientIntegrator : public mfem::DeltaLFIntegrator {
 private:
  /** @brief The matrix coefficient \f$\bvec{m}\f$ (components \f$m_{ij}\f$)
   * used in the integral. */
  mfem::MatrixCoefficient& M_;

  /** @brief Set when \f$\bvec{m}\f$ is a MatrixDeltaCoefficient: the
   * integrator then acts as a point source through MFEM's delta path. */
  MatrixDeltaCoefficient* delta_M_ = nullptr;

#ifndef MFEM_THREAD_SAFE
  /** @brief Workspace vector for element vector computations (non-thread-safe).
   */
  mfem::Vector v;
  /** @brief Workspace for the shape function gradients (non-thread-safe). */
  mfem::DenseMatrix dshape;
  /** @brief Workspace for the coefficient matrix evaluation (non-thread-safe).
   */
  mfem::DenseMatrix m;
#endif

 public:
  /**
   * @brief Constructs a DomainLFDeformationGradientIntegrator.
   * @param M The matrix coefficient \f$\bvec{m}\f$ used in the integral; a
   * MatrixDeltaCoefficient selects the point-source (delta) path.
   * @param ir An optional integration rule. If `nullptr`, a rule of order
   * 2 el.GetOrder() + Trans.OrderW() is used.
   */
  DomainLFDeformationGradientIntegrator(
      mfem::MatrixCoefficient& M, const mfem::IntegrationRule* ir = nullptr);

  /**
   * @brief Assembles the element vector for a given finite element.
   *
   * This method calculates the local contribution of the integral
   * \f$\int_T m_{ij} \frac{\partial u_{i}}{\partial x_{j}} dx\f$ over an
   * element \f$T\f$ to the right-hand side vector.
   *
   * @param el The finite element for which to assemble the element vector.
   * @param Trans The element transformation mapping reference coordinates to
   * physical coordinates.
   * @param elvect The output element vector, overwritten and resized to
   * `d*el.GetDof()` (\f$d\f$ the space dimension, byNODES ordering).
   */
  void AssembleRHSElementVect(const mfem::FiniteElement& el,
                              mfem::ElementTransformation& Trans,
                              mfem::Vector& elvect) override;

  /**
   * @brief Point-source assembly at the delta center (requires a
   * MatrixDeltaCoefficient): \f$\mathrm{elvect} = s\, m_{ij}\,\partial_j
   * \phi_a(\bvec{x}_c)\f$ with the integration point of @p Trans set to
   * the center by the linear form's delta machinery.
   * @param fe The finite element containing the center.
   * @param Trans The element transformation.
   * @param elvect The output element vector, resized to `d*fe.GetDof()`.
   */
  void AssembleDeltaElementVect(const mfem::FiniteElement& fe,
                                mfem::ElementTransformation& Trans,
                                mfem::Vector& elvect) override;

  /**
   * @brief Inherit other overloads of AssembleRHSElementVect from base class.
   * This ensures that other base-class `AssembleRHSElementVect` methods (if
   * any) are also accessible.
   */
  using mfem::LinearFormIntegrator::AssembleRHSElementVect;
};

}  // namespace AdGIA
