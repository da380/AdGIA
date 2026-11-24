/**
 * @file material_law.hpp
 * @brief Hyperelastic laws about a given equilibrium state, and the
 * coefficients that evaluate them at a deformation
 * (doc/methods/finite_strain.md).
 *
 * The reference configuration is the given equilibrium. With
 * @f$\mathbf F@f$ the deformation gradient from it and
 * @f$\mathbf E = \tfrac12(\mathbf F^T\mathbf F - \mathbf 1)@f$ the Green
 * strain, a law gives the strain energy per unit reference volume
 * @f$W(X, \mathbf E)@f$, the second Piola--Kirchhoff stress
 * @f$\mathbf S = \partial W/\partial\mathbf E@f$ and the tensor
 * @f$\partial\mathbf S/\partial\mathbf E@f$. The first Piola--Kirchhoff
 * stress is @f$\mathbf P = \mathbf F\mathbf S@f$. No stress-free state
 * appears: at @f$\mathbf E = 0@f$ the stress is the equilibrium stress
 * of the reference configuration and the tensor is the second elastic
 * tensor there.
 *
 * Symmetric tensors are passed as Mandel vectors in the ordering of
 * SymmetricTensorBasis (components scaled by 1 on the diagonal and
 * @f$\sqrt2@f$ off it), in which the contraction of two tensors is the
 * dot product of their vectors and a fourth-order tensor with the
 * classical symmetries is a symmetric matrix.
 */

#pragma once

#include <memory>

#include "AdGIA/elastic_tensor.hpp"
#include "AdGIA/mappings.hpp"
#include "mfem.hpp"

namespace AdGIA {

/** @brief The Green strain @f$\tfrac12(\mathbf F^T\mathbf F - \mathbf 1)@f$
 * of a deformation gradient, as a Mandel vector. */
void GreenStrain(const mfem::DenseMatrix& F, mfem::Vector& e);

/** @brief The symmetric d x d matrix of a Mandel vector. */
void MandelToMatrix(int dim, const mfem::Vector& s, mfem::DenseMatrix& S);

/** @brief The Mandel vector of a symmetric d x d matrix (its symmetric
 * part is taken). */
void MatrixToMandel(const mfem::DenseMatrix& S, mfem::Vector& s);

/**
 * @brief A hyperelastic law about a given equilibrium: at a point, the
 * energy density, the second Piola--Kirchhoff stress and its derivative
 * with respect to the Green strain.
 */
class MaterialLaw {
 public:
  explicit MaterialLaw(int dim)
      : dim_(dim), size_(SymmetricTensorBasis::Size(dim)) {}
  virtual ~MaterialLaw() = default;

  int SpaceDim() const { return dim_; }
  /** @brief The number of components of a symmetric tensor, d(d+1)/2. */
  int StrainSize() const { return size_; }

  /**
   * @brief Evaluate the law at a point.
   * @param e The Green strain, a Mandel vector.
   * @param S The second Piola--Kirchhoff stress, a Mandel vector; not
   * computed where null.
   * @param tangent The derivative of the stress with respect to the
   * strain, a Mandel matrix; not computed where null.
   * @return The energy density, per unit reference volume.
   */
  virtual mfem::real_t Evaluate(mfem::ElementTransformation& T,
                                const mfem::IntegrationPoint& ip,
                                const mfem::Vector& e, mfem::Vector* S,
                                mfem::DenseMatrix* tangent) = 0;

 private:
  int dim_, size_;
};

/**
 * @brief The quadratic law with a prescribed equilibrium stress,
 * @f[
 *   W = \mathbf T_0 : \mathbf E + \tfrac12\,\mathbf E : \Xi : \mathbf E,
 *   \qquad \mathbf S = \mathbf T_0 + \Xi : \mathbf E,
 * @f]
 * with @f$\mathbf T_0@f$ the equilibrium stress of the reference
 * configuration and @f$\Xi@f$ its second elastic tensor, the bare tensor
 * (BareElasticTensorCoefficient converts seismological moduli). It keeps
 * every geometric effect of a finite deformation and makes no statement
 * about the dependence of the moduli on strain.
 */
class QuadraticMaterialLaw : public MaterialLaw {
 public:
  /**
   * @param Xi The second elastic tensor, an n_s x n_s Mandel matrix
   * coefficient; not owned.
   * @param T0 The equilibrium stress, a symmetric d x d matrix
   * coefficient; not owned; null for a natural reference state.
   */
  QuadraticMaterialLaw(int dim, mfem::MatrixCoefficient& Xi,
                       mfem::MatrixCoefficient* T0 = nullptr);

  mfem::real_t Evaluate(mfem::ElementTransformation& T,
                        const mfem::IntegrationPoint& ip,
                        const mfem::Vector& e, mfem::Vector* S,
                        mfem::DenseMatrix* tangent) override;

 private:
  mfem::MatrixCoefficient* Xi_;
  mfem::MatrixCoefficient* T0_;
  mfem::DenseMatrix Xi_q_, T0_q_;
  mfem::Vector t0_, s_;
};

/**
 * @brief The energy density of a law at the deformation of a mapping,
 * @f$W(X, \mathbf E(\mathbf F))@f$ with @f$\mathbf F@f$ the mapping's
 * gradient. With @p subtract_reference the term linear in the
 * displacement gradient at the reference, @f$\mathbf S(0) : (\mathbf F -
 * \mathbf 1)@f$, is taken off, which gives the energy whose first
 * variation is the incremental residual.
 */
class StrainEnergyDensityCoefficient : public mfem::Coefficient {
 public:
  StrainEnergyDensityCoefficient(MaterialLaw& law, Diffeomorphism& map,
                                 bool subtract_reference = false);
  mfem::real_t Eval(mfem::ElementTransformation& T,
                    const mfem::IntegrationPoint& ip) override;

 private:
  MaterialLaw* law_;
  Diffeomorphism* map_;
  bool subtract_reference_;
  mfem::DenseMatrix F_, S_;
  mfem::Vector e_, s_;
};

/**
 * @brief The second Piola--Kirchhoff stress of a law at the deformation
 * of a mapping, a symmetric d x d matrix: the stress of the geometric
 * stiffness of the tangent (GeometricStiffnessIntegrator).
 */
class SecondPiolaKirchhoffStressCoefficient : public mfem::MatrixCoefficient {
 public:
  SecondPiolaKirchhoffStressCoefficient(MaterialLaw& law, Diffeomorphism& map);
  void Eval(mfem::DenseMatrix& S, mfem::ElementTransformation& T,
            const mfem::IntegrationPoint& ip) override;

 private:
  MaterialLaw* law_;
  Diffeomorphism* map_;
  mfem::DenseMatrix F_;
  mfem::Vector e_, s_;
};

/**
 * @brief The first Piola--Kirchhoff stress of a law at the deformation
 * of a mapping, @f$\mathbf P = \mathbf F\mathbf S@f$, a d x d matrix with
 * @f$P_{iA}@f$ the force in direction i per unit reference area of
 * normal A: the coefficient of the elastic residual
 * @f$\int \mathbf P : \nabla\mathbf v@f$
 * (DomainLFDeformationGradientIntegrator). With @p subtract_reference
 * its value at the reference, @f$\mathbf S(0)@f$, is taken off, which
 * gives the incremental residual.
 */
class FirstPiolaKirchhoffStressCoefficient : public mfem::MatrixCoefficient {
 public:
  FirstPiolaKirchhoffStressCoefficient(MaterialLaw& law, Diffeomorphism& map,
                                       bool subtract_reference = false);
  void Eval(mfem::DenseMatrix& P, mfem::ElementTransformation& T,
            const mfem::IntegrationPoint& ip) override;

 private:
  MaterialLaw* law_;
  Diffeomorphism* map_;
  bool subtract_reference_;
  mfem::DenseMatrix F_, S_;
  mfem::Vector e_, s_;
};

/**
 * @brief The derivative of the second Piola--Kirchhoff stress with
 * respect to the Green strain at the deformation of a mapping, an
 * n_s x n_s Mandel matrix: the tensor of the material stiffness of the
 * tangent (MaterialStiffnessIntegrator).
 */
class MaterialTangentCoefficient : public mfem::MatrixCoefficient {
 public:
  MaterialTangentCoefficient(MaterialLaw& law, Diffeomorphism& map);
  void Eval(mfem::DenseMatrix& C, mfem::ElementTransformation& T,
            const mfem::IntegrationPoint& ip) override;

 private:
  MaterialLaw* law_;
  Diffeomorphism* map_;
  mfem::DenseMatrix F_;
  mfem::Vector e_;
};

}  // namespace AdGIA
