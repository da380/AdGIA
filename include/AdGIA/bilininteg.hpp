/**
 * @file bilininteg.hpp
 * @brief Bilinear form integrators and discrete interpolators: couplings
 * between scalar, vector and matrix fields on nodal spaces, the general
 * (anisotropic) elasticity integrator, the geometric and material
 * stiffnesses and the gravity blocks of the linearised referential system,
 * the transformed diffusion integrator, the strain interpolators, the
 * boundary normal integrators of the fluid–solid problems, and the
 * one-sided kernels of the slipping-interface forms.
 */

#pragma once

#include <array>

#include "mfem.hpp"
#include "AdGIA/index.hpp"
#include "AdGIA/mappings.hpp"

namespace AdGIA {

/**
 * @brief BilinearFormIntegrator that acts on a test vector field,
 * \f$\bvec{v}\f$, and a trial scalar field, \f$u\f$ according to:
 * \f[
 *   (\mathbf{v},u) \mapsto \int_{\Omega} \mathbf{q}\cdot \mathbf{v} \, u \dd x
 * \f]
 * where \f$\Omega\f$ is the domain and \f$\bvec{q}\f$ is a vector
 * coefficient.
 *
 * It is assumed that the vector field is defined on a finite element space
 * formed from the product of a scalar nodal space.
 *
 * With a `Diffeomorphism` the integrator assembles the pull-back of the
 * form through the mapping (doc/mappings.md): the volume element gains
 * the Jacobian, while coefficients stay referential.
 */
class DomainVectorScalarIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::VectorCoefficient*
      QV; /**< Pointer to the vector coefficient \f$\bvec{q}\f$. */
  Diffeomorphism* map_ = nullptr; /**< Optional mapping (pull-back form). */

#ifndef MFEM_THREAD_SAFE
  mfem::Vector trial_shape, test_shape,
      qv; /**< Internal buffers for shape functions and coefficient values. */
  mfem::DenseMatrix
      part_elmat; /**< Internal buffer for partial element matrix. */
#endif

 public:
  /**
   * @brief Constructor for DomainVectorScalarIntegrator.
   *
   * To define an instance, a vector coefficient is provided along, optionally,
   * with an integration rule. The vector coefficient must return values with
   * size equal to the spatial dimension of the finite-element space.
   *
   * @param qv A reference to the `mfem::VectorCoefficient` \f$\bvec{q}\f$.
   * @param ir An optional pointer to an `mfem::IntegrationRule`. If `nullptr`,
   * a default rule will be used.
   */
  DomainVectorScalarIntegrator(mfem::VectorCoefficient& qv,
                               const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), QV{&qv} {}

  /**
   * @brief Constructor for the pull-back of the form through a mapping.
   * @param qv The referential vector coefficient \f$\bvec{q}\f$.
   * @param map The mapping (not owned).
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainVectorScalarIntegrator(mfem::VectorCoefficient& qv, Diffeomorphism& map,
                               const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), QV{&qv}, map_{&map} {}

  /**
   * @brief Sets the default integration rule.
   *
   * The orders of the trial space, test space, and element transformation are
   * taken into account. Variations in the coefficient are not considered.
   *
   * @param trial_fe The trial finite element.
   * @param test_fe The test finite element.
   * @param Trans The element transformation.
   * @return A constant reference to the chosen `mfem::IntegrationRule`.
   */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& Trans);

  /**
   * @brief Implementation of the element level assembly for the bilinear form.
   *
   * @param trial_fe The trial finite element for the scalar field \f$u\f$.
   * @param test_fe The test finite element for the vector field
   * \f$\bvec{v}\f$.
   * @param Trans The element transformation.
   * @param elmat The output dense matrix representing the element stiffness
   * matrix.
   */
  void AssembleElementMatrix2(const mfem::FiniteElement& trial_fe,
                              const mfem::FiniteElement& test_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;

 protected:
  /**
   * @brief Protected method to get the default integration rule.
   * @param trial_fe The trial finite element.
   * @param test_fe The test finite element.
   * @param trans The element transformation.
   * @return A constant pointer to the chosen `mfem::IntegrationRule`.
   */
  const mfem::IntegrationRule* GetDefaultIntegrationRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& trans) const {
    return &GetRule(trial_fe, test_fe, trans);
  }
};

/**
 * @brief BilinearFormIntegrator that acts on a test vector field,
 * \f$\bvec{v}\f$, and a trial scalar field, \f$u\f$ according to:
 *
 *   \f[
 *     (\bvec{v},u) \mapsto \int_{\Omega} \bvec{v} \cdot \bvec{q} \cdot \grad u
 * \dd x,
 *   \f]
 * where \f$\Omega\f$ is the domain and \f$\bvec{q}\f$ is a matrix coefficient.
 *
 * The coefficient \f$\bvec{q}\f$ can be set as a scalar, in which case the
 * matrix coefficient is proportional to the identity matrix. It can also be set
 * as a vector, this corresponding to the matrix coefficient being diagonal.
 *
 * It is assumed that the vector field is defined on a finite element space
 * formed from the product of a scalar nodal space. The scalar field must be
 * defined on a nodal space on which the gradient operator is defined.
 */
class DomainVectorGradScalarIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::Coefficient* Q = nullptr; /**< Pointer to a scalar coefficient. If not
                                     null, \f$q_{ij} = Q \delta_{ij}\f$. */
  mfem::VectorCoefficient* QV = nullptr; /**< Pointer to a vector coefficient.
                                            If not null, \f$q_{ij} = QV_i
                                            \delta_{ij}\f$. */
  mfem::MatrixCoefficient* QM =
      nullptr; /**< Pointer to a matrix coefficient. If not null, \f$q_{ij}\f$
                  is directly from QM. */
  Diffeomorphism* map_ = nullptr; /**< Optional mapping (pull-back form). */

#ifndef MFEM_THREAD_SAFE
  mfem::Vector test_shape, qv; /**< Internal buffers for test shape functions
                                  and vector coefficient values. */
  mfem::DenseMatrix trial_dshape, part_elmat, qm,
      tm; /**< Internal buffers for trial derivative shape functions, partial
             element matrix, and matrix coefficient values. */
  mfem::DenseMatrix F_, dtmp_; /**< Mapping buffers. */
#endif

 public:
  /**
   * @brief Constructor for DomainVectorGradScalarIntegrator with an identity
   * matrix coefficient.
   * @param ir An optional pointer to an `mfem::IntegrationRule`. If `nullptr`,
   * a default rule will be used.
   */
  DomainVectorGradScalarIntegrator(const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir) {}

  /**
   * @brief Constructor for DomainVectorGradScalarIntegrator with a scalar
   * coefficient.
   *
   * The matrix coefficient is proportional to the identity matrix (\f$q_{ij} =
   * Q
   * \delta_{ij}\f$).
   *
   * @param q A reference to the `mfem::Coefficient`, Q.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainVectorGradScalarIntegrator(mfem::Coefficient& q,
                                   const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q} {}

  /**
   * @brief Constructor for DomainVectorGradScalarIntegrator with a vector
   * coefficient.
   *
   * The matrix coefficient is diagonal (\f$q_{ij} = QV_i \delta_{ij}\f$).
   *
   * @param qv A reference to the `mfem::VectorCoefficient`, QV.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainVectorGradScalarIntegrator(mfem::VectorCoefficient& qv,
                                   const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), QV{&qv} {}

  /**
   * @brief Constructor for DomainVectorGradScalarIntegrator with a matrix
   * coefficient.
   *
   * @param qm A reference to the `mfem::MatrixCoefficient`, q.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainVectorGradScalarIntegrator(mfem::MatrixCoefficient& qm,
                                   const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), QM{&qm} {}

  /** @brief Pull-back of the form through a mapping (identity coefficient). */
  DomainVectorGradScalarIntegrator(Diffeomorphism& map,
                                   const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), map_{&map} {}

  /** @brief Pull-back of the form through a mapping (referential scalar
   * coefficient). */
  DomainVectorGradScalarIntegrator(mfem::Coefficient& q, Diffeomorphism& map,
                                   const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q}, map_{&map} {}

  /** @brief Pull-back of the form through a mapping (referential vector
   * coefficient). */
  DomainVectorGradScalarIntegrator(mfem::VectorCoefficient& qv,
                                   Diffeomorphism& map,
                                   const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), QV{&qv}, map_{&map} {}

  /** @brief Pull-back of the form through a mapping (referential matrix
   * coefficient). */
  DomainVectorGradScalarIntegrator(mfem::MatrixCoefficient& qm,
                                   Diffeomorphism& map,
                                   const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), QM{&qm}, map_{&map} {}

  /**
   * @brief Sets the default integration rule.
   *
   * The orders of the trial space, test space, and element transformation are
   * taken into account, with one order removed to account for the spatial
   * derivatives. Variations in the matrix coefficient are not considered.
   *
   * @param trial_fe The trial finite element for the scalar field \f$u\f$.
   * @param test_fe The test finite element for the vector field \f$\bvec{v}\f$.
   * @param Trans The element transformation.
   * @return A constant reference to the chosen `mfem::IntegrationRule`.
   */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& Trans);

  /**
   * @brief Implementation of the element level assembly for the bilinear form.
   *
   * @param trial_fe The trial finite element.
   * @param test_fe The test finite element.
   * @param Trans The element transformation.
   * @param elmat The output dense matrix representing the element stiffness
   * matrix.
   */
  void AssembleElementMatrix2(const mfem::FiniteElement& trial_fe,
                              const mfem::FiniteElement& test_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;

 protected:
  /**
   * @brief Protected method to get the default integration rule.
   * @param trial_fe The trial finite element.
   * @param test_fe The test finite element.
   * @param trans The element transformation.
   * @return A constant pointer to the chosen `mfem::IntegrationRule`.
   */
  const mfem::IntegrationRule* GetDefaultIntegrationRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& trans) const override {
    return &GetRule(trial_fe, test_fe, trans);
  }
};

/**
 * @brief BilinearFormIntegrator that acts on a test vector field,
 * \f$\bvec{v}\f$, and a trial scalar field, \f$u\f$ according to:
 * \f[
 *    (\bvec{v},u) \mapsto \int_{\Omega} q \divg \bvec{v}\, u \dd x,
 * \f]
 * where \f$\Omega\f$ is the domain and \f$q\f$ is a scalar coefficient.
 *
 * It is assumed that the vector field is defined on a finite element space
 * formed from the product of a scalar nodal space on which the gradient
 * operator is defined.
 */
class DomainDivVectorScalarIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::Coefficient* Q = nullptr; /**< Pointer to the scalar coefficient \f$q\f$. */
  Diffeomorphism* map_ = nullptr; /**< Optional mapping (pull-back form). */

#ifndef MFEM_THREAD_SAFE
  mfem::DenseMatrix test_dshape,
      part_elmat; /**< Internal buffers for test derivative shape functions and
                     partial element matrix. */
  mfem::Vector trial_shape; /**< Internal buffer for trial shape functions. */
  mfem::DenseMatrix F_, dtmp_; /**< Mapping buffers. */
#endif

 public:
  /**
   * @brief Constructor for DomainDivVectorScalarIntegrator.
   *
   * The scalar coefficient is taken equal to the constant 1 if no coefficient
   * is provided.
   *
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainDivVectorScalarIntegrator(const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{nullptr} {}

  /**
   * @brief Constructor for DomainDivVectorScalarIntegrator with a scalar
   * coefficient.
   * @param q A reference to the `mfem::Coefficient` \f$q\f$.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainDivVectorScalarIntegrator(mfem::Coefficient& q,
                                  const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q} {}

  /** @brief Pull-back of the form through a mapping (unit coefficient). */
  DomainDivVectorScalarIntegrator(Diffeomorphism& map,
                                  const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), map_{&map} {}

  /** @brief Pull-back of the form through a mapping (referential scalar
   * coefficient). */
  DomainDivVectorScalarIntegrator(mfem::Coefficient& q, Diffeomorphism& map,
                                  const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q}, map_{&map} {}

  /**
   * @brief Sets the default integration rule.
   *
   * The orders of the trial space, test space, and element transformation are
   * taken into account, with one order removed to account for the spatial
   * derivatives. Variations in the coefficient are not considered.
   *
   * @param trial_fe The trial finite element for the scalar field \f$u\f$.
   * @param test_fe The test finite element for the vector field \f$\bvec{v}\f$.
   * @param Trans The element transformation.
   * @return A constant reference to the chosen `mfem::IntegrationRule`.
   */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& Trans);

  /**
   * @brief Implementation of element level calculations for the bilinear form.
   *
   * @param trial_fe The trial finite element.
   * @param test_fe The test finite element.
   * @param Trans The element transformation.
   * @param elmat The output dense matrix representing the element stiffness
   * matrix.
   */
  void AssembleElementMatrix2(const mfem::FiniteElement& trial_fe,
                              const mfem::FiniteElement& test_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;

 protected:
  /**
   * @brief Protected method to get the default integration rule.
   * @param trial_fe The trial finite element.
   * @param test_fe The test finite element.
   * @param trans The element transformation.
   * @return A constant pointer to the chosen `mfem::IntegrationRule`.
   */
  const mfem::IntegrationRule* GetDefaultIntegrationRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& trans) const override {
    return &GetRule(trial_fe, test_fe, trans);
  }
};

/**
 * @brief BilinearFormIntegrator that acts on a test vector field,
 * \f$\bvec{v}\f$, and a trial vector field, \f$\bvec{u}\f$ according to:
 * \f[
 *    (\bvec{v},\bvec{u}) \mapsto \int_{\Omega} q \divg \bvec{v}\, \divg
 * \bvec{u} \dd x
 * \f]
 * where \f$\Omega\f$ is the domain and \f$q\f$ is a scalar coefficient.
 *
 * It is assumed that both vector fields are defined on finite element spaces
 * formed from the product of scalar nodal spaces on which the gradient operator
 * is defined.
 */
class DomainDivVectorDivVectorIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::Coefficient* Q =
      nullptr; /**< Pointer to the scalar coefficient \f$q\f$. */
  Diffeomorphism* map_ = nullptr; /**< Optional mapping (pull-back form). */

#ifndef MFEM_THREAD_SAFE
  mfem::DenseMatrix trial_dshape,
      test_dshape; /**< Internal buffers for trial and test derivative shape
                      functions. */
  mfem::DenseMatrix F_, dtmp_; /**< Mapping buffers. */
#endif

 public:
  /**
   * @brief Constructor for DomainDivVectorDivVectorIntegrator.
   *
   * The coefficient is taken to be the constant 1 if no coefficient is
   * provided.
   *
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainDivVectorDivVectorIntegrator(const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir) {}

  /**
   * @brief Constructor for DomainDivVectorDivVectorIntegrator with a scalar
   * coefficient.
   * @param q A reference to the `mfem::Coefficient` \f$q\f$.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainDivVectorDivVectorIntegrator(mfem::Coefficient& q,
                                     const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q} {}

  /** @brief Pull-back of the form through a mapping (unit coefficient). */
  DomainDivVectorDivVectorIntegrator(Diffeomorphism& map,
                                     const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), map_{&map} {}

  /** @brief Pull-back of the form through a mapping (referential scalar
   * coefficient). */
  DomainDivVectorDivVectorIntegrator(mfem::Coefficient& q, Diffeomorphism& map,
                                     const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q}, map_{&map} {}

  /**
   * @brief Sets the default integration rule.
   *
   * The orders of the trial space, test space, and element transformation are
   * taken into account, with two orders removed to account for the spatial
   * derivatives. Variations in the coefficient are not considered.
   *
   * @param trial_fe The trial finite element for the vector field
   * \f$\bvec{u}\f$.
   * @param test_fe The test finite element for the vector field
   * \f$\bvec{v}\f$.
   * @param Trans The element transformation.
   * @return A constant reference to the chosen `mfem::IntegrationRule`.
   */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& Trans);

  /**
   * @brief Implementation of the element level calculations for the bilinear
   * form.
   *
   * @param trial_fe The trial finite element.
   * @param test_fe The test finite element.
   * @param Trans The element transformation.
   * @param elmat The output dense matrix representing the element stiffness
   * matrix.
   */
  void AssembleElementMatrix2(const mfem::FiniteElement& trial_fe,
                              const mfem::FiniteElement& test_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat);

  /**
   * @brief Assembly method when the trial and test spaces are equal.
   * @param fe The finite element for both trial and test spaces.
   * @param Trans The element transformation.
   * @param elmat The output dense matrix representing the element stiffness
   * matrix.
   */
  void AssembleElementMatrix(const mfem::FiniteElement& fe,
                             mfem::ElementTransformation& Trans,
                             mfem::DenseMatrix& elmat) override {
    AssembleElementMatrix2(fe, fe, Trans, elmat);
  }

 protected:
  /**
   * @brief Protected method to get the default integration rule.
   * @param trial_fe The trial finite element.
   * @param test_fe The test finite element.
   * @param trans The element transformation.
   * @return A constant pointer to the chosen `mfem::IntegrationRule`.
   */
  const mfem::IntegrationRule* GetDefaultIntegrationRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& trans) const override {
    return &GetRule(trial_fe, test_fe, trans);
  }
};

/**
 * @brief BilinearFormIntegrator acting on a test vector field, \f$\bvec{v}\f$,
 * and a trial vector field, \f$\bvec{u}\f$ according to:
 * \f[
 *    (\bvec{v},\bvec{u}) \mapsto \int_{\Omega} q \,\bvec{v}\cdot
 *    \grad(\bvec{w}\cdot \bvec{u}) \dd x,
 * \f]
 * where \f$\Omega\f$ is the domain,  \f$q\f$ is a scalar coefficient and
 * \f$\bvec{w}\f$ is a vector coefficient.
 *
 * It is assumed that the vector fields are defined on finite element spaces
 * formed from the product of scalar nodal spaces. On the trial space, the
 * gradient operator must be defined.
 *
 * The product \f$\bvec{w}\cdot\bvec{u}\f$ is replaced by its nodal
 * interpolant on the trial space: \f$\bvec{w}\f$ is evaluated at the
 * trial element's nodes, and the gradient is that of the interpolant. The
 * trial element must therefore be nodal.
 */
class DomainVectorGradVectorIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::Coefficient* Q =
      nullptr; /**< Pointer to the scalar coefficient \f$q\f$. */
  mfem::VectorCoefficient*
      QV; /**< Pointer to the vector coefficient \f$\bvec{w}\f$. */
  Diffeomorphism* map_ = nullptr; /**< Optional mapping (pull-back form). */

#ifndef MFEM_THREAD_SAFE
  mfem::Vector qv, test_shape; /**< Internal buffers for vector coefficient
                                  values and test shape functions. */
  mfem::DenseMatrix trial_dshape, left_elmat, rigth_elmat_trans,
      part_elmat; /**< Internal buffers for trial derivative shape functions and
                     various intermediate matrices. */
  mfem::DenseMatrix F_, dtmp_; /**< Mapping buffers. */
#endif

 public:
  /**
   * @brief Constructor for DomainVectorGradVectorIntegrator with a vector
   * coefficient and default scalar coefficient (1).
   * @param qv A reference to the `mfem::VectorCoefficient` \f$\bvec{w}\f$.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainVectorGradVectorIntegrator(mfem::VectorCoefficient& qv,
                                   const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), QV{&qv} {}

  /**
   * @brief Constructor for DomainVectorGradVectorIntegrator with both vector
   * and scalar coefficients.
   * @param qv A reference to the `mfem::VectorCoefficient` \f$\bvec{w}\f$.
   * @param q A reference to the `mfem::Coefficient` \f$q\f$.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainVectorGradVectorIntegrator(mfem::VectorCoefficient& qv,
                                   mfem::Coefficient& q,
                                   const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q}, QV{&qv} {}

  /** @brief Pull-back of the form through a mapping (referential
   * coefficients). */
  DomainVectorGradVectorIntegrator(mfem::VectorCoefficient& qv,
                                   Diffeomorphism& map,
                                   const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), QV{&qv}, map_{&map} {}

  /** @brief Pull-back of the form through a mapping (referential
   * coefficients, with a scalar factor). */
  DomainVectorGradVectorIntegrator(mfem::VectorCoefficient& qv,
                                   mfem::Coefficient& q, Diffeomorphism& map,
                                   const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q}, QV{&qv}, map_{&map} {}

  /**
   * @brief Sets the default integration rule.
   *
   * The orders of the trial space, test space, and element transformation are
   * taken into account, with one order removed to account for the spatial
   * derivative. Variations in the coefficients are not considered.
   *
   * @param trial_fe The trial finite element for the vector field
   * \f$\bvec{u}\f$.
   * @param test_fe The test finite element for the vector field \f$\bvec{v}\f$.
   * @param Trans The element transformation.
   * @return A constant reference to the chosen `mfem::IntegrationRule`.
   */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& Trans);

  /**
   * @brief Implementation of element level calculations for the bilinear form.
   * @param trial_fe The trial finite element.
   * @param test_fe The test finite element.
   * @param Trans The element transformation.
   * @param elmat The output dense matrix representing the element stiffness
   * matrix.
   */
  void AssembleElementMatrix2(const mfem::FiniteElement& trial_fe,
                              const mfem::FiniteElement& test_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;

  /**
   * @brief Assembly method when the trial and test spaces are equal.
   * @param el The finite element for both trial and test spaces.
   * @param Trans The element transformation.
   * @param elmat The output dense matrix representing the element stiffness
   * matrix.
   */
  void AssembleElementMatrix(const mfem::FiniteElement& el,
                             mfem::ElementTransformation& Trans,
                             mfem::DenseMatrix& elmat) override {
    AssembleElementMatrix2(el, el, Trans, elmat);
  }

 protected:
  /**
   * @brief Protected method to get the default integration rule.
   * @param trial_fe The trial finite element.
   * @param test_fe The test finite element.
   * @param trans The element transformation.
   * @return A constant pointer to the chosen `mfem::IntegrationRule`.
   */
  const mfem::IntegrationRule* GetDefaultIntegrationRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& trans) const override {
    return &GetRule(trial_fe, test_fe, trans);
  }
};

/**
 * @brief BilinearFormIntegrator acting on a test vector field, \f$\bvec{v}\f$,
 * and a trial vector field, \f$\bvec{u}\f$ according to:
 * \f[
 *    (\bvec{v},\bvec{u}) \mapsto \int_{\Omega} \bvec{q} \cdot \bvec{v}\,
 * \divg\bvec{u}  \dd x,
 * \f]
 * where \f$\Omega\f$ is the domain and \f$\bvec{q}\f$ is a vector coefficient.
 *
 * This integrator assumes that the test vector field \f$\bvec{v}\f$ is defined
 * on a finite element space formed from the product of a scalar nodal space,
 * and the trial vector field \f$\bvec{u}\f$ is defined on a finite element
 * space formed from the product of a scalar nodal space on which the gradient
 * operator is defined.
 */
class DomainVectorDivVectorIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::VectorCoefficient* QV =
      nullptr; /**< Pointer to the vector coefficient \f$\bvec{q}\f$. */
  Diffeomorphism* map_ = nullptr; /**< Optional mapping (pull-back form). */

#ifndef MFEM_THREAD_SAFE
  mfem::Vector qv, test_shape; /**< Internal buffers for vector coefficient
                                  values and test shape functions. */
  mfem::DenseMatrix trial_dshape,
      part_elmat; /**< Internal buffers for trial derivative shape functions and
                     partial element matrix. */
  mfem::DenseMatrix F_, dtmp_; /**< Mapping buffers. */
#endif

 public:
  /**
   * @brief Constructor for DomainVectorDivVectorIntegrator.
   * @param qv A reference to the `mfem::VectorCoefficient` \f$\bvec{q}\f$.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainVectorDivVectorIntegrator(mfem::VectorCoefficient& qv,
                                  const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), QV{&qv} {}

  /** @brief Pull-back of the form through a mapping (referential vector
   * coefficient). */
  DomainVectorDivVectorIntegrator(mfem::VectorCoefficient& qv,
                                  Diffeomorphism& map,
                                  const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), QV{&qv}, map_{&map} {}

  /**
   * @brief Sets the default integration rule.
   *
   * The orders of the trial space, test space, and element transformation are
   * taken into account, with one order removed to account for the spatial
   * derivative. Variations in the coefficients are not considered.
   *
   * @param trial_fe The trial finite element for \f$\bvec{u}\f$.
   * @param test_fe The test finite element for the vector field \f$\bvec{v}\f$.
   * @param Trans The element transformation.
   * @return A constant reference to the chosen `mfem::IntegrationRule`.
   */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& Trans);

  /**
   * @brief Implementation of element level calculations for the bilinear form.
   * @param trial_fe The trial finite element.
   * @param test_fe The test finite element.
   * @param Trans The element transformation.
   * @param elmat The output dense matrix representing the element stiffness
   * matrix.
   */
  void AssembleElementMatrix2(const mfem::FiniteElement& trial_fe,
                              const mfem::FiniteElement& test_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;

  /**
   * @brief Assembly method when the trial and test spaces are equal.
   * @param el The finite element for both trial and test spaces.
   * @param Trans The element transformation.
   * @param elmat The output dense matrix representing the element stiffness
   * matrix.
   */
  void AssembleElementMatrix(const mfem::FiniteElement& el,
                             mfem::ElementTransformation& Trans,
                             mfem::DenseMatrix& elmat) override {
    AssembleElementMatrix2(el, el, Trans, elmat);
  }

 protected:
  /**
   * @brief Protected method to get the default integration rule.
   * @param trial_fe The trial finite element.
   * @param test_fe The test finite element.
   * @param trans The element transformation.
   * @return A constant pointer to the chosen `mfem::IntegrationRule`.
   */
  const mfem::IntegrationRule* GetDefaultIntegrationRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& trans) const override {
    return &GetRule(trial_fe, test_fe, trans);
  }
};

/**
 * @brief BilinearFormIntegrator acting on a test matrix field, \f$\bvec{v}\f$,
 * and a trial vector field, \f$\bvec{u}\f$ according to:
 * \f[
 *   (\bvec{v},\bvec{u}) \mapsto \int_{\Omega} q \,\bvec{v}: \deriv \bvec{u}
 * \dd x = \int_{\Omega} q\, v_{ij} \frac{\partial u_{i}}{\partial x_{j}} \dd x,
 * \f]
 * where \f$\Omega\f$ is the domain and \f$q\f$ is a scalar coefficient.
 *
 * The matrix field must be defined on a nodal finite element space formed from
 * the product of a scalar space. The ordering of the matrix components
 * corresponds to a dense matrix using column-major storage (i.e., \f$v_{00},
 * v_{10}, v_{20}, v_{01}, \dots\f$). The vector field must be defined on a
 * nodal finite element space formed from the product of a scalar space for
 * which the gradient operator is defined. The vector and matrix fields need to
 * have compatible dimensions.
 */
class DomainMatrixDeformationGradientIntegrator
    : public mfem::BilinearFormIntegrator {
 private:
  mfem::Coefficient* Q =
      nullptr; /**< Pointer to the scalar coefficient \f$q\f$. */

#ifndef MFEM_THREAD_SAFE
  mfem::Vector test_shape; /**< Internal buffer for test shape functions. */
  mfem::DenseMatrix trial_dshape,
      part_elmat; /**< Internal buffers for trial derivative shape functions and
                     partial element matrix. */
#endif

 public:
  /**
   * @brief Constructor for DomainMatrixDeformationGradientIntegrator.
   *
   * The scalar coefficient is taken equal to the constant 1 if no coefficient
   * is provided.
   *
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainMatrixDeformationGradientIntegrator(
      const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir) {}

  /**
   * @brief Constructor for DomainMatrixDeformationGradientIntegrator with a
   * scalar coefficient.
   * @param q A reference to the `mfem::Coefficient` \f$q\f$.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainMatrixDeformationGradientIntegrator(
      mfem::Coefficient& q, const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q} {}

  /**
   * @brief Sets the default integration rule.
   *
   * The orders of the trial space, test space, and element transformation are
   * taken into account, with one order removed to account for the spatial
   * derivative. Variations in the coefficients are not considered.
   *
   * @param trial_fe The trial finite element for the vector field
   * \f$\bvec{u}\f$.
   * @param test_fe The test finite element for the matrix field \f$\bvec{v}\f$.
   * @param Trans The element transformation.
   * @return A constant reference to the chosen `mfem::IntegrationRule`.
   */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& Trans);

  /**
   * @brief Implementation of element-level calculations for the bilinear form.
   * @param trial_fe The trial finite element.
   * @param test_fe The test finite element.
   * @param Trans The element transformation.
   * @param elmat The output dense matrix representing the element stiffness
   * matrix.
   */
  void AssembleElementMatrix2(const mfem::FiniteElement& trial_fe,
                              const mfem::FiniteElement& test_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;

 protected:
  /**
   * @brief Protected method to get the default integration rule.
   * @param trial_fe The trial finite element.
   * @param test_fe The test finite element.
   * @param trans The element transformation.
   * @return A constant pointer to the chosen `mfem::IntegrationRule`.
   */
  const mfem::IntegrationRule* GetDefaultIntegrationRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& trans) const override {
    return &GetRule(trial_fe, test_fe, trans);
  }
};

/**
 * @brief BilinearFormIntegrator acting on a test symmetric matrix field,
 * \f$\bvec{v}\f$, and a trial vector field, \f$\bvec{u}\f$ according to:
 * \f[
 *   (\bvec{v},\bvec{u}) \mapsto \int_{\Omega} q \,\bvec{v}: \deriv \bvec{u}
 * \dd x = \frac{1}{2}\int_{\Omega} q\, v_{ij} \left(\frac{\partial
 *         u_{i}}{\partial x_{j}} + \frac{\partial
 *         u_{j}}{\partial x_{i}}\right)\dd x,
 * \f]
 * where \f$\Omega\f$ is the domain and \f$q\f$ is a scalar coefficient.
 *
 * The matrix field must be defined on a nodal finite element space formed from
 * the product of a scalar space. The ordering of the matrix components
 * corresponds to a dense matrix using column-major storage but storing only the
 * lower triangle. The vector field must be defined on a nodal finite element
 * space formed from the product of a scalar space for which the gradient
 * operator is defined. The vector and matrix fields need to have compatible
 * dimensions.
 */
class DomainSymmetricMatrixStrainIntegrator
    : public mfem::BilinearFormIntegrator {
 private:
  mfem::Coefficient* Q = nullptr; /**< Pointer to the scalar coefficient \f$q\f$. */

#ifndef MFEM_THREAD_SAFE
  mfem::Vector test_shape; /**< Internal buffer for test shape functions. */
  mfem::DenseMatrix part_elmat,
      trial_dshape; /**< Internal buffers for partial element matrix and trial
                       derivative shape functions. */
#endif

 public:
  /**
   * @brief Constructor for DomainSymmetricMatrixStrainIntegrator.
   *
   * The scalar coefficient is taken equal to the constant 1 if no coefficient
   * is provided.
   *
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainSymmetricMatrixStrainIntegrator(
      const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir) {}

  /**
   * @brief Constructor for DomainSymmetricMatrixStrainIntegrator with a scalar
   * coefficient.
   * @param q A reference to the `mfem::Coefficient` \f$q\f$.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainSymmetricMatrixStrainIntegrator(
      mfem::Coefficient& q, const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q} {}

  /**
   * @brief Sets the default integration rule.
   *
   * The orders of the trial space, test space, and element transformation are
   * taken into account, with one order removed to account for the spatial
   * derivative. Variations in the coefficients are not considered.
   *
   * @param trial_fe The trial finite element for the vector field
   * \f$\bvec{u}\f$.
   * @param test_fe The test finite element for the symmetric matrix field
   * \f$\bvec{v}\f$.
   * @param Trans The element transformation.
   * @return A constant reference to the chosen `mfem::IntegrationRule`.
   */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& Trans);

  /**
   * @brief Implementation of element-level calculations for the bilinear form.
   * @param trial_fe The trial finite element.
   * @param test_fe The test finite element.
   * @param Trans The element transformation.
   * @param elmat The output dense matrix representing the element stiffness
   * matrix.
   */
  void AssembleElementMatrix2(const mfem::FiniteElement& trial_fe,
                              const mfem::FiniteElement& test_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;

 protected:
  /**
   * @brief Protected method to get the default integration rule.
   * @param trial_fe The trial finite element.
   * @param test_fe The test finite element.
   * @param trans The element transformation.
   * @return A constant pointer to the chosen `mfem::IntegrationRule`.
   */
  const mfem::IntegrationRule* GetDefaultIntegrationRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& trans) const override {
    return &GetRule(trial_fe, test_fe, trans);
  }
};

/**
 * @brief BilinearFormIntegrator acting on a test trace-free symmetric matrix
 * field, \f$\bvec{v}\f$, and a trial vector field, \f$\bvec{u}\f$  according
 * to:
 * \f[
 *   (\bvec{v},\bvec{u}) \mapsto \int_{\Omega} q \,\bvec{v}: \deriv \bvec{u}
 * \dd x = \frac{1}{2}\int_{\Omega} q\, v_{ij} \left(
 *         \frac{\partial u_{i}}{\partial x_{j}}
 *        + \frac{\partial u_{j}}{\partial x_{i}}
 *        - \frac{2}{n}\frac{\partial u_{k}}{\partial x_{k}}\delta_{ij}
 *        \right)\dd x,
 * \f]
 * where \f$\Omega\f$ is the domain, \f$q\f$ is a scalar coefficient, and
 * \f$n\f$ the spatial dimension.
 *
 * The matrix field must be defined on a nodal finite element space formed from
 * the product of a scalar space. The ordering of the matrix components
 * corresponds to a dense matrix using column-major storage but storing only the
 * lower triangle and explicitly handling the trace-free nature. The vector
 * field must be defined on a nodal finite element space formed from the product
 * of a scalar space for which the gradient operator is defined. The vector and
 * matrix fields need to have compatible dimensions.
 */
class DomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator
    : public mfem::BilinearFormIntegrator {
 private:
  mfem::Coefficient* Q = nullptr; /**< Pointer to the scalar coefficient \f$q\f$. */

#ifndef MFEM_THREAD_SAFE
  mfem::Vector test_shape; /**< Internal buffer for test shape functions. */
  mfem::DenseMatrix part_elmat,
      trial_dshape; /**< Internal buffers for partial element matrix and trial
                       derivative shape functions. */
#endif

 public:
  /**
   * @brief Constructor for
   * DomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator.
   *
   * The scalar coefficient is taken equal to the constant 1 if no coefficient
   * is provided.
   *
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator(
      const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir) {}

  /**
   * @brief Constructor for
   * DomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator with a scalar
   * coefficient.
   * @param q A reference to the `mfem::Coefficient` \f$q\f$.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator(
      mfem::Coefficient& q, const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q} {}

  /**
   * @brief Sets the default integration rule.
   *
   * The orders of the trial space, test space, and element transformation are
   * taken into account, with one order removed to account for the spatial
   * derivative. Variations in the coefficients are not considered.
   *
   * @param trial_fe The trial finite element for the vector field
   * \f$\bvec{u}\f$.
   * @param test_fe The test finite element for the trace-free symmetric matrix
   * field \f$\bvec{v}\f$.
   * @param Trans The element transformation.
   * @return A constant reference to the chosen `mfem::IntegrationRule`.
   */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& Trans);

  /**
   * @brief Implementation of element-level calculations for the bilinear form.
   * @param trial_fe The trial finite element.
   * @param test_fe The test finite element.
   * @param Trans The element transformation.
   * @param elmat The output dense matrix representing the element stiffness
   * matrix.
   */
  void AssembleElementMatrix2(const mfem::FiniteElement& trial_fe,
                              const mfem::FiniteElement& test_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;

 protected:
  /**
   * @brief Protected method to get the default integration rule.
   * @param trial_fe The trial finite element.
   * @param test_fe The test finite element.
   * @param trans The element transformation.
   * @return A constant pointer to the chosen `mfem::IntegrationRule`.
   */
  const mfem::IntegrationRule* GetDefaultIntegrationRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& trans) const override {
    return &GetRule(trial_fe, test_fe, trans);
  }
};

/**
 * @brief Bilinear form integrator for general linear elasticity,
 * \f[
 * (\bvec{u}, \bvec{v}) \mapsto \int_{\Omega} \bvec{\varepsilon}(\bvec{v})
 * : \bvec{C} : \bvec{\varepsilon}(\bvec{u}) \, \mathrm{d}x,
 * \f]
 * where \f$\bvec{\varepsilon}\f$ is the symmetric strain and \f$\bvec{C}\f$ an
 * arbitrary elasticity tensor.
 *
 * The tensor is supplied as an \f$n_s \times n_s\f$ `mfem::MatrixCoefficient`,
 * \f$n_s = d(d+1)/2\f$, in Mandel form and `SymmetricMatrixIndex` component
 * ordering (lower triangle, column-major, with off-diagonal components scaled
 * by \f$\sqrt{2}\f$). This is the representation produced by the
 * `ElasticTensorCoefficient` classes of `elastic_tensor.hpp`; sums and scalar
 * products of them through MFEM's matrix-coefficient algebra are fine too.
 *
 * The vector field must be defined on a nodal H1 finite element space with
 * `Ordering::byNODES`. The element matrix has the layout
 * `elmat(dof c + i, dof c' + i')` and the default quadrature order is
 * `2 OrderGrad(el)`, as for `mfem::ElasticityIntegrator`. Per quadrature
 * point the reduced strain-displacement matrix \f$B\f$
 * (\f$\hat{\varepsilon} = B u\f$) is formed and \f$w B^T C B\f$ added.
 * Manifold elements are not supported; in 2-D the semantics are plane strain.
 *
 * With a `Diffeomorphism` the integrator assembles the pull-back of the
 * form through the mapping (doc/mappings.md): the shape gradients become
 * derivatives with respect to the mapped coordinates
 * (\f$\mathrm{gshape} \to \mathrm{gshape}\,\bvec{F}^{-1}\f$) and the weight
 * gains the Jacobian (\f$w \to J w\f$), after which the assembly is
 * unchanged. The coefficient stays the *referential* tensor (21 components
 * in 3-D); the pulled-back tensor \f$c'_{iAkB} = J c_{ijkl} F^{-1}_{Aj}
 * F^{-1}_{Bl}\f$, with only its major symmetry, is never formed. For the
 * discrete change-of-variables identity against the mapped mesh, pass the
 * interpolated mapping and one integration rule to both sides.
 */
class ElasticTensorIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::MatrixCoefficient* C_; /**< Pointer to the elasticity tensor
                                  coefficient in Mandel form. */
  Diffeomorphism* map_ = nullptr; /**< Optional mapping (pull-back form). */

#ifndef MFEM_THREAD_SAFE
  mfem::DenseMatrix dshape_, gshape_, B_, Cq_,
      CB_; /**< Internal buffers: reference and physical shape gradients,
              strain-displacement matrix, tensor at the point and C B. */
  mfem::DenseMatrix F_, gshape_map_; /**< Mapping buffers: deformation
              gradient and mapped shape gradients. */
#endif

 public:
  /**
   * @brief Constructor for ElasticTensorIntegrator.
   * @param C The \f$n_s \times n_s\f$ Mandel-form elasticity tensor
   * coefficient.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  explicit ElasticTensorIntegrator(mfem::MatrixCoefficient& C,
                                   const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), C_(&C) {}

  /**
   * @brief Constructor for the pull-back of the form through a mapping.
   * @param C The *referential* Mandel-form elasticity tensor coefficient.
   * @param map The mapping (not owned).
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  ElasticTensorIntegrator(mfem::MatrixCoefficient& C, Diffeomorphism& map,
                          const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), C_(&C), map_(&map) {}

  /**
   * @brief Implementation of element-level calculations for the bilinear form.
   * @param el The finite element (shared by trial and test spaces).
   * @param Trans The element transformation.
   * @param elmat The output dense matrix representing the element stiffness
   * matrix.
   */
  void AssembleElementMatrix(const mfem::FiniteElement& el,
                             mfem::ElementTransformation& Trans,
                             mfem::DenseMatrix& elmat) override;

  /**
   * @brief Builds the reduced strain-displacement matrix \f$B\f$ (size
   * \f$n_s \times d\,\mathrm{dof}\f$) from the physical shape-function
   * gradients.
   * @param dim The spatial dimension.
   * @param gshape The physical gradients of the shape functions
   * (\f$\mathrm{dof} \times d\f$).
   * @param B The output matrix, resized as needed.
   */
  static void StrainDisplacementMatrix(int dim, const mfem::DenseMatrix& gshape,
                                       mfem::DenseMatrix& B);
};

/**
 * @brief The geometric (initial-stress) stiffness of the total-Lagrangian
 * split about an equilibrium (doc/gravitating_elasticity.md §2):
 * @f[
 *   (u, v) \mapsto \int_B S_{AB}\,\partial_A u_k\,\partial_B v_k\,dV,
 * @f]
 * with @f$\mathbf{S}@f$ the (symmetric) second Piola–Kirchhoff equilibrium
 * stress as a @f$d \times d@f$ MatrixCoefficient. The initial stress acts
 * on the full displacement gradient — this is the term through which a
 * pre-stressed state stiffens (or destabilises) the linearised operator,
 * and at a natural reference @f$\mathbf{S} = \mathbf{T}^0@f$, the
 * equilibrium Cauchy stress.
 *
 * The optional trailing Diffeomorphism is the *relabelling* pull-back of
 * the Domain* family (derivatives with respect to the mapped coordinates,
 * Jacobian in the weight; doc/mappings.md) — the equilibrium-mapping role
 * belongs to MaterialStiffnessIntegrator, and the geometric term itself
 * carries no @f$F_e@f$.
 */
class GeometricStiffnessIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::MatrixCoefficient* S_;
  Diffeomorphism* map_ = nullptr;

#ifndef MFEM_THREAD_SAFE
  mfem::DenseMatrix dshape_, gshape_, Sq_, tmp_, G_;
  mfem::DenseMatrix F_, gshape_map_;
#endif

 public:
  /**
   * @param S The @f$d \times d@f$ equilibrium stress coefficient.
   * @param ir An optional integration rule; the default has order
   * 2 OrderGrad(el).
   */
  explicit GeometricStiffnessIntegrator(
      mfem::MatrixCoefficient& S, const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), S_(&S) {}

  /** @brief Relabelling pull-back through @p map (not owned). */
  GeometricStiffnessIntegrator(mfem::MatrixCoefficient& S, Diffeomorphism& map,
                               const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), S_(&S), map_(&map) {}

  /** @brief Element matrix: one copy of the scalar block
   * @f$\int S_{AB}\,\partial_A\phi_a\,\partial_B\phi_b@f$ per displacement
   * component (byNODES layout). */
  void AssembleElementMatrix(const mfem::FiniteElement& el,
                             mfem::ElementTransformation& Trans,
                             mfem::DenseMatrix& elmat) override;
};

/**
 * @brief The material stiffness of the total-Lagrangian split about an
 * equilibrium mapping @f$\varphi_e@f$ (doc/gravitating_elasticity.md §2):
 * @f[
 *   (u, v) \mapsto \int_B \bigl\langle \hat C\,
 *     \widehat{\mathrm{sym}(F_e^T Du)},\,
 *     \widehat{\mathrm{sym}(F_e^T Dv)} \bigr\rangle\,dV,
 * @f]
 * with @f$\hat C@f$ the second elastic tensor at equilibrium
 * (@f$n_s \times n_s@f$ Mandel, classical symmetries) and
 * @f$\mathrm{sym}(F_e^T Du)@f$ the linearised Green strain. There is no
 * Jacobian factor: the strain energy is per referential volume. With the
 * identity mapping this is ElasticTensorIntegrator.
 *
 * The Diffeomorphism here is the *equilibrium mapping* (exact or
 * interpolated per the object), not the relabelling pull-back of the
 * other mapped integrators: a relabelling composes into the mapping and
 * transforms the coefficients (Al-Attar et al. 2018, "AC18" of
 * doc/gravitating_elasticity.md, eqs. 134–136; done by the
 * background-state layer), leaving this integrator's form unchanged.
 * Consequently this is the plain pull-back of the elastic form only when
 * @f$\hat C@f$ is the relabelled tensor; the relabelling pull-back of an
 * unrelabelled tensor (e.g. the covariant gauge penalty) is
 * ElasticTensorIntegrator(C, map) (doc/mappings.md, "Pitfalls").
 */
class MaterialStiffnessIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::MatrixCoefficient* C_;
  Diffeomorphism* map_;

#ifndef MFEM_THREAD_SAFE
  mfem::DenseMatrix dshape_, gshape_, B_, Cq_, CB_, F_;
#endif

 public:
  /**
   * @param C The Mandel-form second elastic tensor at equilibrium.
   * @param phi_e The equilibrium mapping (not owned).
   * @param ir An optional integration rule; the default has order
   * 2 OrderGrad(el).
   */
  MaterialStiffnessIntegrator(mfem::MatrixCoefficient& C,
                              Diffeomorphism& phi_e,
                              const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), C_(&C), map_(&phi_e) {}

  void AssembleElementMatrix(const mfem::FiniteElement& el,
                             mfem::ElementTransformation& Trans,
                             mfem::DenseMatrix& elmat) override;

  /**
   * @brief The strain-displacement matrix of the linearised Green strain:
   * @f$B@f$ (size @f$n_s \times d\,\mathrm{dof}@f$) such that @f$B\,u@f$
   * is the Mandel vector of @f$\mathrm{sym}(\mathbf{F}^T Du)@f$. With
   * @f$\mathbf{F} = \mathbf{1}@f$ this is
   * ElasticTensorIntegrator::StrainDisplacementMatrix.
   * @param dim The spatial dimension.
   * @param gshape The physical shape-function gradients
   * (@f$\mathrm{dof} \times d@f$).
   * @param F The deformation gradient @f$\mathbf{F}@f$ at the point.
   * @param B The output matrix, resized as needed.
   */
  static void StrainDisplacementMatrix(int dim, const mfem::DenseMatrix& gshape,
                                       const mfem::DenseMatrix& F,
                                       mfem::DenseMatrix& B);
};

/**
 * @brief The gravity–gravity displacement block of the linearised
 * referential system (doc/gravitating_elasticity.md §3.1):
 * @f[
 *   (u, v) \mapsto s \int_B \langle a''(u,v)\,\mathbf{g}_0,
 *   \mathbf{g}_0\rangle\,dV,
 * @f]
 * with @f$a(F) = J F^{-1}F^{-T}@f$ evaluated on the equilibrium mapping,
 * @f$a''@f$ its second derivative in the directions @f$Du, Dv@f$, and
 * @f$\mathbf{g}_0 = \nabla\zeta^0@f$ the referential gradient of the
 * background potential. The problem layer passes the scale
 * @f$s = 1/8\pi G@f$. Assembled through the rank-one structure of
 * @f$H = F_e^{-1}Du@f$ per basis function — no fourth-order coefficient
 * is formed. The default integration rule has order 2 OrderGrad(el).
 */
class ReferentialGravityIntegrator : public mfem::BilinearFormIntegrator {
 private:
  Diffeomorphism* map_;
  mfem::VectorCoefficient* g0_;
  mfem::real_t scale_;

#ifndef MFEM_THREAD_SAFE
  mfem::DenseMatrix dshape_, gshape_, F_, a_, M_, P_, ag_;
  mfem::Vector g0v_, w_, beta_, gamma_;
#endif

 public:
  /**
   * @param phi_e The equilibrium mapping (not owned).
   * @param grad_zeta0 The referential background-potential gradient
   * @f$\mathbf{g}_0 = \nabla\zeta^0@f$.
   * @param scale The factor @f$s@f$.
   * @param ir An optional integration rule.
   */
  ReferentialGravityIntegrator(Diffeomorphism& phi_e,
                               mfem::VectorCoefficient& grad_zeta0,
                               mfem::real_t scale = 1.0,
                               const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir),
        map_(&phi_e),
        g0_(&grad_zeta0),
        scale_(scale) {}

  void AssembleElementMatrix(const mfem::FiniteElement& el,
                             mfem::ElementTransformation& Trans,
                             mfem::DenseMatrix& elmat) override;
};

/**
 * @brief The displacement–potential coupling of the linearised
 * referential system (doc/gravitating_elasticity.md §3.1):
 * @f[
 *   (\zeta, v) \mapsto
 *   s \int_B \langle a'(v)\,\mathbf{g}_0, \nabla\zeta\rangle\,dV,
 *   \qquad a'(v) = (\mathrm{tr}H)a_e - H a_e - a_e H^T,\
 *   H = F_e^{-1}Dv,
 * @f]
 * with @f$a_e@f$, @f$\mathbf{g}_0@f$ as in ReferentialGravityIntegrator,
 * as a MixedBilinearForm integrator with the *scalar* potential
 * @f$\zeta@f$ as trial and the *vector* displacement @f$v@f$ as test;
 * the transpose of the assembled matrix gives the potential-row block.
 * The problem layer passes @f$s = 1/4\pi G@f$. The default integration
 * rule has order trial + test order + OrderGrad(test_fe).
 */
class ReferentialGravityCouplingIntegrator
    : public mfem::BilinearFormIntegrator {
 private:
  Diffeomorphism* map_;
  mfem::VectorCoefficient* g0_;
  mfem::real_t scale_;

#ifndef MFEM_THREAD_SAFE
  mfem::DenseMatrix dshape_u_, gshape_u_, dshape_p_, gshape_p_, F_, a_, M_,
      agp_;
  mfem::Vector g0v_, w_, beta_, gamma_, fw_;
#endif

 public:
  /**
   * @param phi_e The equilibrium mapping (not owned).
   * @param grad_zeta0 The referential background-potential gradient
   * @f$\mathbf{g}_0 = \nabla\zeta^0@f$.
   * @param scale The factor @f$s@f$.
   * @param ir An optional integration rule.
   */
  ReferentialGravityCouplingIntegrator(
      Diffeomorphism& phi_e, mfem::VectorCoefficient& grad_zeta0,
      mfem::real_t scale = 1.0, const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir),
        map_(&phi_e),
        g0_(&grad_zeta0),
        scale_(scale) {}

  /** @brief Rows: vector test dofs; columns: scalar trial dofs. */
  void AssembleElementMatrix2(const mfem::FiniteElement& trial_fe,
                              const mfem::FiniteElement& test_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;
};

/**
 * @brief BilinearFormIntegrator for the transformed Laplace integrator.
 *
 * The bilinear form acts on a pair of scalar fields through
 * \f[
 *  (v,u) \mapsto \int_{\Omega} \grad v \cdot \bvec{a} \cdot \grad u \dd x,
 * \f]
 * with \f$\Omega\f$ the domain, and where the symmetric matrix field,
 * \f$\bvec{a}\f$, takes the form
 * \f[
 * \bvec{a} = J \bvec{C}^{-1}  = J \bvec{F}^{-1} \bvec{F}^{-T},
 * \f]
 * with \f$\bvec{F} = \deriv \boldsymbol{\xi}\f$ for a diffeomorphism,
 * \f$\boldsymbol{\xi}\f$, on
 * \f$\Omega\f$.
 *
 *
 * The diffeomorphism and/or resulting matrix field can be specified in four
 * ways:
 *
 * -# A `Diffeomorphism` is provided: \f$\bvec{F}\f$ comes from its
 * EvalGradient() at each quadrature point — exact for the analytic
 * mappings, the gradient of the interpolant for a
 * GridFunctionDiffeomorphism (the mode of the discrete change-of-variables
 * identity, see doc/mappings.md).
 * -# A `mfem::Coefficient` is provided which specifies the scalar part,
 * \f$f\f$, of a radial mapping \f$\boldsymbol{\xi}(\bvec{x}) = f(\bvec{x})
 * \bvec{x}\f$. Then \f$\bvec{F} = f\,\bvec{1} + \bvec{x}\otimes\nabla
 * f_h\f$, with \f$f\f$ evaluated at the quadrature point and
 * \f$\nabla f_h\f$ the gradient of the interpolant of \f$f\f$ on the trial
 * space.
 * -# A `mfem::VectorCoefficient` which directly specifies
 * \f$\boldsymbol{\xi}\f$ is provided. \f$\bvec{F}\f$ is that of the
 * interpolant of \f$\boldsymbol{\xi}\f$ on the trial space.
 * -# A `mfem::MatrixCoefficient` specifying \f$\bvec{a}\f$ is given directly.
 *
 * There is also a constructor for which no coefficients are provided, this
 * corresponding to the identity transformation.
 *
 * Modes 2 and 3 sample the coefficient at the trial element's nodes, so the
 * trial element must be nodal.
 *
 * The remaining pulled-back Poisson pieces need no dedicated integrators: the
 * volume terms \f$\int J \rho\, \phi \phi' \dd x\f$ and \f$\int J \rho\,
 * \phi' \dd x\f$ are mfem::MassIntegrator and mfem::DomainLFIntegrator with
 * the coefficient \f$\rho\f$ multiplied by JacobianCoefficient.
 */
class TransformedDiffusionIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::Coefficient* Q =
      nullptr; /**< Scalar part \f$f\f$ of a radial mapping. */
  mfem::VectorCoefficient* QV =
      nullptr; /**< The mapping \f$\boldsymbol{\xi}\f$. */
  mfem::MatrixCoefficient* QM =
      nullptr; /**< The matrix field \f$\bvec{a}\f$ given directly. */
  Diffeomorphism* D =
      nullptr; /**< The mapping supplying \f$\bvec{F}\f$ directly. */

#ifndef MFEM_THREAD_SAFE
  mfem::Vector fs, df, x;
  mfem::DenseMatrix trial_dshape, test_dshape, xis, F, a,
      trial_dshape_trans; /**< Internal buffers for shape function derivatives
and intermediate matrices during integration. */
#endif

 public:
  /**
   * @brief Constructor for the identity mapping.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  TransformedDiffusionIntegrator(const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir) {}

  /**
   * @brief Constructor from a Diffeomorphism: \f$\bvec{F}\f$ from its
   * EvalGradient() at each quadrature point.
   * @param d The mapping (not owned).
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  TransformedDiffusionIntegrator(Diffeomorphism& d,
                                 const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), D{&d} {}

  /**
   * @brief Constructor for a radial mapping specified by a scalar
   * function.
   * @param q The scalar part \f$f\f$ of the radial mapping.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  TransformedDiffusionIntegrator(mfem::Coefficient& q,
                                 const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q} {}

  /**
   * @brief Constructor for a general transformation specified by a
   * VectorCoefficient.
   * @param qv The mapping \f$\boldsymbol{\xi}\f$.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  TransformedDiffusionIntegrator(mfem::VectorCoefficient& qv,
                                 const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), QV{&qv} {}

  /**
   * @brief Constructor for which the matrix \f$\bvec{a}\f$ is provided
   * directly.
   * @param qm The matrix field \f$\bvec{a}\f$.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  TransformedDiffusionIntegrator(mfem::MatrixCoefficient& qm,
                                 const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), QM{&qm} {}

  /**
   * @brief Sets the default integration rule.
   *
   * The orders of the trial space, test space, and element transformation are
   * taken into account. Variations in the coefficient are not considered.
   *
   * @param trial_fe The trial finite element.
   * @param test_fe The test finite element.
   * @param Trans The element transformation.
   * @return A constant reference to the chosen `mfem::IntegrationRule`.
   */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& Trans);

  /**
   * @brief Implementation of the element level assembly for the bilinear form.
   * @param trial_fe The trial finite element.
   * @param test_fe The test finite element.
   * @param Trans The element transformation.
   * @param elmat The output dense matrix representing the element stiffness
   * matrix.
   */
  void AssembleElementMatrix2(const mfem::FiniteElement& trial_fe,
                              const mfem::FiniteElement& test_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;

  /**
   * @brief Assembly method when the trial and test spaces are equal.
   * @param fe The finite element for both trial and test spaces.
   * @param Trans The element transformation.
   * @param elmat The output dense matrix representing the element stiffness
   * matrix.
   */
  void AssembleElementMatrix(const mfem::FiniteElement& fe,
                             mfem::ElementTransformation& Trans,
                             mfem::DenseMatrix& elmat) override {
    AssembleElementMatrix2(fe, fe, Trans, elmat);
  }

 protected:
  /**
   * @brief Protected method to get the default integration rule.
   * @param trial_fe The trial finite element.
   * @param test_fe The test finite element.
   * @param trans The element transformation.
   * @return A constant pointer to the chosen `mfem::IntegrationRule`.
   */
  const mfem::IntegrationRule* GetDefaultIntegrationRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& trans) const {
    return &GetRule(trial_fe, test_fe, trans);
  }
};

/**
 * @brief DiscreteInterpolator that acts on a vector field, \f$\bvec{u}\f$, to
 * return the matrix field, \f$\deriv \bvec{u}\f$, with components:
 * \f[
 * (\deriv \bvec{u})_{ij} = \frac{\partial u_{i}}{\partial x_{j}},
 * \f]
 * The resulting matrix field components are stored using the column-major
 * format.
 *
 * The input vector field \f$\bvec{u}\f$ must be defined on a nodal finite
 * element space formed from the product of a scalar space on which the gradient
 * operator is defined. The output matrix field \f$\deriv \bvec{u}\f$ must be
 * defined on a nodal finite element space formed from the product of a scalar
 * space.
 */
class DeformationGradientInterpolator : public mfem::DiscreteInterpolator {
 private:
#ifndef MFEM_THREAD_SAFE
  mfem::DenseMatrix
      dshape; /**< Internal buffer for derivative shape functions. */
#endif

 public:
  /**
   * @brief Constructs a DeformationGradientInterpolator object.
   * This default constructor initializes the interpolator without specific
   * parameters.
   */
  DeformationGradientInterpolator() {}

  /**
   * @brief Assembles the element matrix for the interpolation of \f$\deriv
   * \bvec{u}\f$.
   *
   * This method computes the local element interpolation matrix that maps
   * degrees of freedom of the input vector field \f$u\f$ to the degrees of
   * freedom of the output matrix field \f$v = \deriv u\f$.
   *
   * @param in_fe The input finite element for the vector field.
   * @param out_fe The output finite element for the matrix field.
   * @param Trans The element transformation.
   * @param elmat The output dense matrix representing the element interpolation
   * matrix, of size `d*d*out_fe.GetDof()` by `d*in_fe.GetDof()`, with `d`
   * the dimension of `in_fe`.
   */
  void AssembleElementMatrix2(const mfem::FiniteElement& in_fe,
                              const mfem::FiniteElement& out_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;
};

/**
 * @brief DiscreteInterpolator that acts on a vector displacement field,
 * \f$\bvec{u}\f$, to return the symmetric strain tensor field,
 * \f$\bvec{e}\f$, with components:
 * \f[
 * e_{ij} = \frac{1}{2}\left(\frac{\partial u_{i}}{\partial x_{j}} +
 * \frac{\partial u_{j}}{\partial x_{i}}\right).
 * \f]
 *
 * The components of the symmetric matrix field \f$\bvec{e}\f$ are
 * stored in column-major format, keeping only elements in the lower triangle
 * due to symmetry.
 * - In 3D spaces, this implies the ordering: \f$e_{00}\f$,
 * \f$e_{10}\f$, \f$e_{20}\f$,
 * \f$e_{11}\f$, \f$e_{21}\f$, \f$e_{22}\f$.
 * - In 2D spaces, this implies the ordering: \f$e_{00}\f$,
 * \f$e_{10}\f$, \f$e_{11}\f$.
 *
 * The input vector field \f$\bvec{u}\f$ must be defined on a nodal finite
 * element space formed from the product of a scalar space on which the gradient
 * operator is defined. The output symmetric matrix field
 * \f$\bvec{e}\f$ must be defined on a nodal finite element space
 * formed from the product of a scalar space.
 */
class StrainInterpolator : public mfem::DiscreteInterpolator {
 private:
#ifndef MFEM_THREAD_SAFE
  mfem::DenseMatrix
      dshape; /**< Internal buffer for derivative shape functions. */
#endif
 public:
  /**
   * @brief Constructs a StrainInterpolator object.
   * This default constructor initializes the interpolator.
   */
  StrainInterpolator() {}

  /**
   * @brief Assembles the element matrix for the interpolation of the strain
   * tensor.
   *
   * This method computes the local element interpolation matrix that maps
   * degrees of freedom of the input displacement field \f$\bvec{u}\f$ to the
   * degrees of freedom of the output symmetric strain field \f$\bvec{e}\f$.
   *
   * @param in_fe The input finite element for the vector field.
   * @param out_fe The output finite element for the symmetric matrix field.
   * @param Trans The element transformation.
   * @param elmat The output dense matrix representing the element interpolation
   * matrix, of size `d*(d+1)/2*out_fe.GetDof()` by `d*in_fe.GetDof()`,
   * with `d` the dimension of `in_fe`.
   */
  void AssembleElementMatrix2(const mfem::FiniteElement& in_fe,
                              const mfem::FiniteElement& out_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;
};

/**
 * @brief DiscreteInterpolator that maps a vector displacement field,
 * \f$\bvec{u}\f$, into a trace-free symmetric matrix field, \f$\bvec{v}\f$.
 *
 * The components of the output trace-free symmetric matrix \f$\bvec{v}\f$ are
 * given by:
 * \f[
 * v_{ij} = \frac{1}{2}\left(\frac{\partial u_{i}}{\partial x_{j}} +
 * \frac{\partial u_{j}}{\partial x_{i}}\right) - \frac{1}{n}
 * \frac{\partial u_{k}}{\partial x_{k}} \delta_{ij}
 * \f]
 * where \f$n\f$ is the spatial dimension.
 *
 * The input vector field \f$\bvec{u}\f$ must be defined on a nodal finite
 * element space formed from the product of a scalar space on which the gradient
 * operator is defined. The output trace-free symmetric matrix field
 * \f$\bvec{v}\f$ must be defined on a nodal finite element space formed from
 * the product of a scalar space. Its components are those of
 * `TraceFreeSymmetricMatrixIndex`: the lower triangle in column-major order
 * without the last diagonal entry, which is minus the sum of the others.
 */
class DeviatoricStrainInterpolator : public mfem::DiscreteInterpolator {
 private:
#ifndef MFEM_THREAD_SAFE
  mfem::DenseMatrix
      dshape; /**< Internal buffer for derivative shape functions. */
#endif
 public:
  /**
   * @brief Constructs a DeviatoricStrainInterpolator object.
   * This default constructor initializes the interpolator.
   */
  DeviatoricStrainInterpolator() {}

  /**
   * @brief Assembles the element matrix for the interpolation of the deviatoric
   * strain tensor.
   *
   * This method computes the local element interpolation matrix that maps
   * degrees of freedom of the input displacement field \f$u\f$ to the degrees
   * of freedom of the output trace-free symmetric matrix field \f$v\f$.
   *
   * @param in_fe The input finite element for the vector field.
   * @param out_fe The output finite element for the trace-free symmetric matrix
   * field.
   * @param Trans The element transformation.
   * @param elmat The output dense matrix representing the element interpolation
   * matrix, of size `(d*(d+1)/2-1)*out_fe.GetDof()` by `d*in_fe.GetDof()`,
   * with `d` the dimension of `in_fe`.
   */
  void AssembleElementMatrix2(const mfem::FiniteElement& in_fe,
                              const mfem::FiniteElement& out_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;
};

/**
 * @brief Boundary integrator
 * \f[
 *   (\bvec{u},\bvec{v}) \mapsto \int_{\Gamma} q\,(\bvec{n}\cdot\bvec{u})
 *   (\bvec{n}\cdot\bvec{v}) \dd S,
 * \f]
 * on boundary elements, for a vector field on a nodal (H1-type) space with
 * vdim equal to the space dimension and Ordering::byNODES, and a scalar
 * coefficient \f$q\f$.
 *
 * \f$\bvec{n}\f$ is the unit normal of the boundary element obtained from
 * mfem::CalcOrtho of its transformation's Jacobian, i.e. the outward normal
 * of the mesh for consistently oriented boundary elements (the normal that
 * mfem::BoundaryNormalLFIntegrator uses). On a (Par)SubMesh this is the
 * outward normal of the submesh, on inherited and on cut boundaries alike.
 *
 * Used for the fluid–solid interface term of self-gravitating problems,
 * \f$-\int \rho_F\,(\bvec{m}\cdot\nabla\Phi_0)(\bvec{m}\cdot\bvec{u})
 * (\bvec{m}\cdot\bvec{v})\f$, with \f$q\f$ built from a
 * BoundaryNormalDotCoefficient.
 */
class BoundaryNormalNormalIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::Coefficient* Q = nullptr; /**< Pointer to the coefficient \f$q\f$. */
  Diffeomorphism* map_ = nullptr; /**< Optional mapping (pull-back form). */

#ifndef MFEM_THREAD_SAFE
  mfem::Vector shape, normal, nshape; /**< Internal buffers. */
  mfem::Vector nu_;                   /**< Mapping buffer. */
#endif

 public:
  /**
   * @param q The scalar coefficient (optional; unit when null).
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  explicit BoundaryNormalNormalIntegrator(
      mfem::Coefficient* q = nullptr, const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{q} {}

  explicit BoundaryNormalNormalIntegrator(
      mfem::Coefficient& q, const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q} {}

  /**
   * @brief Pull-back of the form through a mapping (doc/mappings.md):
   * with \f$\nu = \mathrm{cof}(F)\mathbf{n}\f$, the integrand
   * \f$q\,(\mathbf{m}\cdot\mathbf{u})(\mathbf{m}\cdot\mathbf{u}')\,dS\f$
   * becomes \f$q\,(\nu\cdot\mathbf{u})(\nu\cdot\mathbf{u}')/|\nu|\,dS\f$
   * (two unit normals, one measure). The coefficient stays referential;
   * a \f$\tilde{\mathbf{m}}\cdot\tilde\nabla\tilde\Phi_0\f$ factor
   * belongs in it via MappedBoundaryNormalDotCoefficient.
   */
  explicit BoundaryNormalNormalIntegrator(
      Diffeomorphism& map, const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), map_{&map} {}

  BoundaryNormalNormalIntegrator(mfem::Coefficient& q, Diffeomorphism& map,
                                 const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q}, map_{&map} {}

  /** @brief Default rule: order 2 el.GetOrder() + Trans.OrderW(). */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& el, const mfem::ElementTransformation& Trans);

  void AssembleElementMatrix(const mfem::FiniteElement& el,
                             mfem::ElementTransformation& Trans,
                             mfem::DenseMatrix& elmat) override;
};

/**
 * @brief The one-sided kernel of the slip-interface pressure form
 * (doc/slip_interface.tex, proposition "The interface form"): on boundary
 * elements of one (solid-side) vector space,
 * \f[
 *   G(\bvec{u}, \bvec{w}) = \oint_\Sigma \pi\;
 *     \bnu\cdot\nabla_\Sigma \bvec{u}\,
 *     [\,P_T F_e^{-1}(1 - \hat\bnu\otimes\hat\bnu)\,\bvec{w}\,]\,\dd S,
 * \f]
 * with \f$\bnu = \mathrm{cof}(F_e)\,\bvec{n}\f$ the unnormalised Nanson
 * normal of the mapping (identity map: \f$\bnu = \bvec{n}\f$),
 * \f$\nabla_\Sigma\f$ the tangential (surface shape-function) gradient,
 * and \f$P_T\f$ the tangential projector. The direction slot is
 * tangential on the slip constraint; off it the discrete form needs a
 * sanitiser, and the \f$\bnu\f$-orthogonal projection BEFORE
 * \f$F_e^{-1}\f$ is the pullback of the deformed face's own tangential
 * projector -- the unique choice invariant under face-fixing shears of
 * \f$F_e\f$, i.e. the one that preserves the discrete
 * change-of-variables identity (TestSlipInterface,
 * InterfaceKernelsFaceShearCovariance). Unmapped, it is contained in
 * \f$P_T\f$ and changes nothing. NON-symmetric by design: \f$\bvec{u}\f$
 * occupies the test (row) slot and \f$\bvec{w}\f$ the trial (column)
 * slot, and the symmetrised two-field interface blocks are built from
 * \f$G\f$ and the pairing by NewSlipInterfaceMatrix
 * (doc/slip_interface.tex, "One-sided assembly of the interface forms").
 * \f$\bvec{n}\f$ is the boundary element's unit normal (CalcOrtho), i.e.
 * the outward normal of the solid SubMesh. The space must be a nodal
 * vector space with vdim equal to the space dimension and
 * Ordering::byNODES.
 */
class SlipInterfacePressureIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::Coefficient* pi_;
  Diffeomorphism* map_ = nullptr;

#ifndef MFEM_THREAD_SAFE
  mfem::Vector shape_, normal_, nu_;
  mfem::DenseMatrix dshape_, gshape_, Jt_, JtJ_, F_, Fi_, PT_, dir_;
#endif

 public:
  /** @param pi The referential pressure on the interface.
   *  @param ir Optional integration rule. The mapping defaults to the
   *  identity. */
  explicit SlipInterfacePressureIntegrator(
      mfem::Coefficient& pi, const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), pi_{&pi} {}

  /** @brief With the equilibrium mapping @p map (not owned) supplying
   * \f$F_e\f$ and \f$\bnu\f$. */
  SlipInterfacePressureIntegrator(mfem::Coefficient& pi, Diffeomorphism& map,
                                  const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), pi_{&pi}, map_{&map} {}

  /** @brief Default rule: 2 el.GetOrder() + Trans.OrderW(). */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& el, const mfem::ElementTransformation& Trans);

  void AssembleElementMatrix(const mfem::FiniteElement& el,
                             mfem::ElementTransformation& Trans,
                             mfem::DenseMatrix& elmat) override;
};

/**
 * @brief The one-sided VECTOR kernel of the broken-\f$\zeta\f$ gravity
 * interface form (doc/slip_interface.tex, proposition "The gravity
 * interface form"): on boundary elements of one (solid-side) vector space,
 * \f[
 *   G_A(\bvec{u}, \bvec{w}) = \oint_\Sigma
 *     \mathbf{A}\cdot\nabla_\Sigma \bvec{u}\,
 *     [\,P_T F_e^{-1}(1 - \hat\bnu\otimes\hat\bnu)\,\bvec{w}\,]\,\dd S,
 *   \qquad
 *   \mathbf{A} = \frac{|\mathbf{b}|^2}{8\pi G}\,\bnu
 *              - \frac{\mathbf{b}\cdot\bnu}{4\pi G}\,\mathbf{b},
 * \f]
 * with \f$\mathbf{b} = F_e^{-T}\nabla\zeta^0\f$ computed from the
 * supplied referential gradient \f$\nabla\zeta^0\f$ and the mapping,
 * and \f$\bnu = \mathrm{cof}(F_e)\bvec{n}\f$. Conventions (tangential
 * projector, non-symmetry, space requirements) exactly as
 * SlipInterfacePressureIntegrator; the symmetrised blocks are built by
 * NewSlipGravityInterfaceMatrix.
 */
class SlipInterfaceGravityIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::VectorCoefficient* grad_zeta0_;
  mfem::real_t G_;
  Diffeomorphism* map_ = nullptr;

#ifndef MFEM_THREAD_SAFE
  mfem::Vector shape_, normal_, nu_, gz_, b_, A_;
  mfem::DenseMatrix dshape_, gshape_, Jt_, JtJ_, F_, Fi_, PT_, dir_;
#endif

 public:
  /** @param grad_zeta0 The referential background-potential gradient
   *  \f$\nabla\zeta^0\f$ on the interface.
   *  @param G The gravitational constant.
   *  @param ir Optional integration rule; identity mapping. */
  SlipInterfaceGravityIntegrator(mfem::VectorCoefficient& grad_zeta0,
                                 mfem::real_t G,
                                 const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), grad_zeta0_{&grad_zeta0}, G_{G} {}

  /** @brief With the equilibrium mapping @p map (not owned). */
  SlipInterfaceGravityIntegrator(mfem::VectorCoefficient& grad_zeta0,
                                 mfem::real_t G, Diffeomorphism& map,
                                 const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir),
        grad_zeta0_{&grad_zeta0},
        G_{G},
        map_{&map} {}

  /** @brief Default rule: 2 el.GetOrder() + Trans.OrderW(). */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& el, const mfem::ElementTransformation& Trans);

  void AssembleElementMatrix(const mfem::FiniteElement& el,
                             mfem::ElementTransformation& Trans,
                             mfem::DenseMatrix& elmat) override;
};

/**
 * @brief The one-sided SCALAR--vector kernel of the broken-\f$\zeta\f$
 * gravity interface form: mixed boundary integrator with scalar trial
 * \f$\zeta\f$ (which carries the surface gradient) and vector test
 * \f$\bvec{w}\f$ (the slip slot),
 * \f[
 *   G_q(\zeta, \bvec{w}) = \oint_\Sigma
 *     q\,\nabla_\Sigma\zeta\cdot\bigl(P_T F_e^{-1}
 *     (1 - \hat\bnu\otimes\hat\bnu)\,\bvec{w}\bigr)\,\dd S,
 *   \qquad q = \frac{\mathbf{b}\cdot\bnu}{4\pi G},
 * \f]
 * coefficients as SlipInterfaceGravityIntegrator. Rows (test) are the
 * vector dofs, columns (trial) the scalar dofs. Assembled on the solid
 * side; the four rectangular \f$(\bvec{v}, \zeta)\f$ blocks are built by
 * NewSlipGravityInterfaceMatrix.
 */
class SlipInterfaceGravityScalarIntegrator
    : public mfem::BilinearFormIntegrator {
 private:
  mfem::VectorCoefficient* grad_zeta0_;
  mfem::real_t G_;
  Diffeomorphism* map_ = nullptr;

#ifndef MFEM_THREAD_SAFE
  mfem::Vector shape_, normal_, nu_, gz_, b_;
  mfem::DenseMatrix dshape_, gshape_, Jt_, JtJ_, F_, Fi_, PT_, dir_, T_;
#endif

 public:
  /** @param grad_zeta0 The referential background-potential gradient
   *  \f$\nabla\zeta^0\f$ on the interface.
   *  @param G The gravitational constant.
   *  @param ir Optional integration rule; identity mapping. */
  SlipInterfaceGravityScalarIntegrator(
      mfem::VectorCoefficient& grad_zeta0, mfem::real_t G,
      const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), grad_zeta0_{&grad_zeta0}, G_{G} {}

  /** @brief With the equilibrium mapping @p map (not owned). */
  SlipInterfaceGravityScalarIntegrator(
      mfem::VectorCoefficient& grad_zeta0, mfem::real_t G, Diffeomorphism& map,
      const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir),
        grad_zeta0_{&grad_zeta0},
        G_{G},
        map_{&map} {}

  /** @brief Default rule: trial + test order + Trans.OrderW(). */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& Trans);

  void AssembleElementMatrix2(const mfem::FiniteElement& trial_fe,
                              const mfem::FiniteElement& test_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;
};

/**
 * @brief Mixed boundary integrator acting on a scalar trial field \f$p\f$
 * and a vector test field \f$\bvec{v}\f$:
 * \f[
 *   (\bvec{v}, p) \mapsto \int_{\Gamma} q\, p\,(\bvec{n}\cdot\bvec{v})
 *   \dd S,
 * \f]
 * on boundary elements, with \f$\bvec{n}\f$ the boundary element's unit
 * normal as in BoundaryNormalNormalIntegrator. The vector space must be a
 * nodal space with vdim equal to the space dimension and Ordering::byNODES.
 *
 * Its transpose (vector trial, scalar test) is obtained with
 * mfem::TransposeIntegrator. Used for the interface coupling
 * \f$-\int \rho_F\,\phi\,(\bvec{m}\cdot\bvec{v})\f$ of self-gravitating
 * fluid–solid problems.
 */
class BoundaryNormalScalarIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::Coefficient* Q = nullptr; /**< Pointer to the coefficient \f$q\f$. */
  Diffeomorphism* map_ = nullptr; /**< Optional mapping (pull-back form). */

#ifndef MFEM_THREAD_SAFE
  mfem::Vector trial_shape, test_shape, normal, nshape; /**< Buffers. */
  mfem::Vector nu_;                                     /**< Mapping buffer. */
#endif

 public:
  explicit BoundaryNormalScalarIntegrator(
      mfem::Coefficient* q = nullptr, const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{q} {}

  explicit BoundaryNormalScalarIntegrator(
      mfem::Coefficient& q, const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q} {}

  /**
   * @brief Pull-back of the form through a mapping (doc/mappings.md):
   * Nanson's relation exactly, \f$q\,p\,(\mathbf{m}\cdot\mathbf{v})\,dS
   * \to q\,p\,(\nu\cdot\mathbf{v})\,dS\f$ with \f$\nu =
   * \mathrm{cof}(F)\mathbf{n}\f$ — the measure and normalisation factors
   * cancel. The coefficient stays referential.
   */
  explicit BoundaryNormalScalarIntegrator(
      Diffeomorphism& map, const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), map_{&map} {}

  BoundaryNormalScalarIntegrator(mfem::Coefficient& q, Diffeomorphism& map,
                                 const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q}, map_{&map} {}

  /** @brief Default rule: order trial + test + Trans.OrderW(). */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& Trans);

  void AssembleElementMatrix2(const mfem::FiniteElement& trial_fe,
                              const mfem::FiniteElement& test_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;
};

/**
 * @brief Mixed boundary integrator acting on a scalar trial field
 * \f$p\f$ and a vector test field \f$\bvec{v}\f$:
 * \f[
 *   (\bvec{v}, p) \mapsto \oint_\Gamma p\,(\mathbf{c}\cdot\bvec{v})
 *   \,\dd S,
 * \f]
 * with \f$\mathbf{c}\f$ a given vector coefficient (evaluated as
 * supplied — any mapping factors belong in the coefficient). The
 * direction-agnostic sibling of BoundaryNormalScalarIntegrator, used
 * for the broken-\f$\zeta\f$ scalar-jump constraint kernel
 * \f$\oint \chi\,(\mathbf{b}\cdot\bvec{u})\f$. Space requirements as
 * there (nodal vector space, vdim = space dimension, byNODES).
 */
class BoundaryVectorScalarIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::VectorCoefficient* c_;

#ifndef MFEM_THREAD_SAFE
  mfem::Vector trial_shape, test_shape, cvec_, cshape_;
#endif

 public:
  explicit BoundaryVectorScalarIntegrator(
      mfem::VectorCoefficient& c, const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), c_{&c} {}

  /** @brief Default rule: order trial + test + Trans.OrderW(). */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& Trans);

  void AssembleElementMatrix2(const mfem::FiniteElement& trial_fe,
                              const mfem::FiniteElement& test_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;
};

}  // namespace AdGIA
