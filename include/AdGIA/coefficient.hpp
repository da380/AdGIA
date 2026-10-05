/**
 * @file coefficient.hpp
 * @brief General-purpose coefficients: the unit radial vector and, for the
 * self-gravitating fluid–solid problems, the normal component of a vector
 * coefficient on boundary elements and the barotropic density gradient
 * @f$d\rho/d\Phi_0@f$ of a fluid; and a matrix-valued delta function for
 * moment-tensor point sources.
 *
 * Coefficients tied to one subsystem live with it: the elasticity tensors in
 * elastic_tensor.hpp, harmonic expansions in spherical_harmonics.hpp, the
 * rigid modes in null_space.hpp, mappings and their pull-backs in
 * mappings.hpp.
 */

#pragma once

#include <functional>
#include <memory>

#include "mfem.hpp"

namespace AdGIA {

/**
 * @brief The unit radial vector (x - x0)/|x - x0|; e_d at x = x0.
 */
class RadialUnitVectorCoefficient : public mfem::VectorCoefficient {
 public:
  explicit RadialUnitVectorCoefficient(int dim);
  RadialUnitVectorCoefficient(int dim, const mfem::Vector& x0);

  void Eval(mfem::Vector& V, mfem::ElementTransformation& T,
            const mfem::IntegrationPoint& ip) override;

 private:
  mfem::Vector x0_, x_;
};

/**
 * @brief @f$\mathbf{V}\cdot\mathbf{n}@f$ on boundary elements, with
 * @f$\mathbf{n}@f$ the boundary element's unit normal (mfem::CalcOrtho of the
 * boundary transformation's Jacobian, normalised: the outward normal for
 * consistently oriented boundary elements, on a SubMesh the submesh's
 * outward normal).
 *
 * Only defined for boundary-element transformations (ElementType ==
 * BDR_ELEMENT); evaluation on any other transformation aborts. The vector
 * coefficient is evaluated on the boundary transformation, which is fine for
 * mfem::GradientGridFunctionCoefficient (MFEM evaluates the gradient in the
 * adjacent element) and for coefficients of position.
 *
 * With @f$\mathbf{V} = \nabla\Phi_0@f$ this gives @f$\mathbf{n}\cdot
 * \nabla\Phi_0 = \pm g@f$ on a fluid–solid interface, the sign selecting
 * between a fluid below and a fluid above the solid.
 */
class BoundaryNormalDotCoefficient : public mfem::Coefficient {
 public:
  explicit BoundaryNormalDotCoefficient(mfem::VectorCoefficient& V) : V_(&V) {}

  mfem::real_t Eval(mfem::ElementTransformation& T,
                    const mfem::IntegrationPoint& ip) override;

 private:
  mfem::VectorCoefficient* V_;
  mfem::Vector v_, n_;
};

/**
 * @brief The barotropic density gradient of a hydrostatic fluid,
 * @f[
 *   \rho'_F = \frac{d\rho}{d\Phi_0}
 *           = \frac{\nabla\rho\cdot\nabla\Phi_0}{|\nabla\Phi_0|^2}
 *           = g^{-1}\,\partial_r\rho \quad\text{(radial models)},
 * @f]
 * built from any two vector coefficients for @f$\nabla\rho@f$ and
 * @f$\nabla\Phi_0@f$, or from a density and a background potential given as
 * grid functions on the same mesh (their gradients are element-local, so a
 * discontinuous density in an L2 space is handled cleanly). Returns zero
 * where @f$|\nabla\Phi_0|@f$ vanishes.
 *
 * The coefficient @f$\rho'_F \phi\phi'@f$ is the fluid mass term of the
 * hydrostatic Poisson equation, eq. (2.8) of Al-Attar & Tromp (2014); it is
 * negative wherever density increases downward. Analytic radial models can
 * supply @f$g^{-1}\partial_r\rho@f$ directly instead of using this class.
 */
class BarotropicDensityGradientCoefficient : public mfem::Coefficient {
 public:
  /** @brief From the two gradients; neither is owned. */
  BarotropicDensityGradientCoefficient(mfem::VectorCoefficient& grad_rho,
                                       mfem::VectorCoefficient& grad_phi0);

  /** @brief From a density and a background potential on the same mesh
   * (scalar grid functions; the gradient coefficients are owned here). */
  BarotropicDensityGradientCoefficient(const mfem::GridFunction& rho,
                                       const mfem::GridFunction& phi0);

  mfem::real_t Eval(mfem::ElementTransformation& T,
                    const mfem::IntegrationPoint& ip) override;

 private:
  std::unique_ptr<mfem::GradientGridFunctionCoefficient> grad_rho_from_rho_,
      grad_phi0_from_phi0_;
  mfem::VectorCoefficient* grad_rho_;
  mfem::VectorCoefficient* grad_phi0_;
  mfem::Vector gr_, gp_;
};

/**
 * @brief A matrix-valued delta function @f$\mathbf{M}\,\delta(\mathbf{x} -
 * \mathbf{x}_c)@f$: a constant matrix (a moment tensor, or a point stress
 * glut) times an mfem::DeltaCoefficient, following the pattern of
 * mfem::VectorDeltaCoefficient.
 *
 * Passed to a DomainLFDeformationGradientIntegrator it assembles the point
 * source @f$v \mapsto M_{ij}\,\partial_j v_i(\mathbf{x}_c)@f$ (times the
 * delta's scale and optional weight), the weak form of the equivalent body
 * force @f$-\mathrm{Div}[\mathbf{M}\,\delta(\mathbf{x}-\mathbf{x}_c)]@f$ of
 * a moment-tensor point source. The center, scale and time dependence are
 * the wrapped DeltaCoefficient's. Like its scalar and vector counterparts
 * it cannot be Eval()uated pointwise.
 *
 * In parallel the source is assembled on the rank whose local mesh contains
 * the center (MFEM's delta machinery); a center placed exactly on a shared
 * element boundary may be found by more than one rank, so keep point
 * sources strictly inside elements.
 */
class MatrixDeltaCoefficient : public mfem::MatrixCoefficient {
 public:
  /** @brief A unit delta at the origin times @p M (square, its dimension
   * the space dimension); @p M is copied. */
  explicit MatrixDeltaCoefficient(const mfem::DenseMatrix& M)
      : mfem::MatrixCoefficient(M.Height()), M_(M) {}

  /** @brief 2-D: @f$s\,\mathbf{M}\,\delta(\mathbf{x} - (x,y))@f$. */
  MatrixDeltaCoefficient(const mfem::DenseMatrix& M, mfem::real_t x,
                         mfem::real_t y, mfem::real_t s)
      : mfem::MatrixCoefficient(M.Height()), M_(M), d_(x, y, s) {}

  /** @brief 3-D: @f$s\,\mathbf{M}\,\delta(\mathbf{x} - (x,y,z))@f$. */
  MatrixDeltaCoefficient(const mfem::DenseMatrix& M, mfem::real_t x,
                         mfem::real_t y, mfem::real_t z, mfem::real_t s)
      : mfem::MatrixCoefficient(M.Height()), M_(M), d_(x, y, z, s) {}

  /** @brief Set the time in the wrapped DeltaCoefficient (for a
   * time-dependent weight). */
  void SetTime(mfem::real_t t) override;

  /** @brief The wrapped scalar DeltaCoefficient (center, scale, weight). */
  mfem::DeltaCoefficient& GetDeltaCoefficient() { return d_; }

  void SetScale(mfem::real_t s) { d_.SetScale(s); }
  void SetDeltaCenter(const mfem::Vector& center) {
    d_.SetDeltaCenter(center);
  }
  void GetDeltaCenter(mfem::Vector& center) { d_.GetDeltaCenter(center); }

  /** @brief Replace the matrix (same dimensions). */
  void SetMatrix(const mfem::DenseMatrix& M);

  /** @brief The matrix @f$\mathbf{M}@f$. */
  const mfem::DenseMatrix& Matrix() const { return M_; }

  /** @brief @f$\mathbf{M}@f$ times DeltaCoefficient::EvalDelta() of the
   * wrapped delta. */
  virtual void EvalDelta(mfem::DenseMatrix& M, mfem::ElementTransformation& T,
                         const mfem::IntegrationPoint& ip);

  /** @brief A delta function cannot be evaluated pointwise: calling this
   * is an MFEM error, as for mfem::VectorDeltaCoefficient. */
  void Eval(mfem::DenseMatrix& M, mfem::ElementTransformation& T,
            const mfem::IntegrationPoint& ip) override;

 private:
  mfem::DenseMatrix M_;
  mfem::DeltaCoefficient d_;
};

}  // namespace AdGIA
