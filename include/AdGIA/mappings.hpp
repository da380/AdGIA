/**
 * @file mappings.hpp
 * @brief The mapping (relabelling) layer: diffeomorphisms of the reference
 * domain and their deformation gradients, evaluated at quadrature points.
 *
 * A Diffeomorphism supplies the mapping @f$\boldsymbol{\xi}@f$ (through the
 * mfem::VectorCoefficient interface) and its deformation gradient
 * @f$F_{iA} = \partial\xi_i/\partial x_A@f$; everything a pulled-back
 * bilinear or linear form needs follows from the pair @f$(\xi, F)@f$. See
 * doc/mappings.md for the pulled-back forms and the assembly recipe, and
 * TransformedDiffusionIntegrator for the Poisson instance.
 *
 * Concrete mappings: IdentityDiffeomorphism (exact identity),
 * CallableDiffeomorphism (analytic, exact gradient), RadialDiffeomorphism
 * (@f$\xi = f\,\mathbf{x}@f$ with exact gradient), TaperedDiffeomorphism
 * (the radial blend of a mapping to the identity) and
 * GridFunctionDiffeomorphism (@f$\xi = \mathbf{x} + \mathbf{h}@f$ for a
 * displacement grid function: interpolated mappings and mappings supplied
 * in discretised form).
 *
 * Free functions: Interpolate (the nodal interpolant of a mapping on a
 * mesh's geometric space), MappedMesh (the element-by-element image mesh;
 * with Interpolate it realises the discrete change-of-variables identity),
 * and MaxIdentityDeviation (asserts the convention that mappings are the
 * identity on and outside the DtN sphere).
 */

#pragma once

#include <functional>
#include <memory>

#include "mfem.hpp"

namespace AdGIA {

/**
 * @brief A diffeomorphism @f$\boldsymbol{\xi}@f$ of the reference domain,
 * with its deformation gradient.
 *
 * The mfem::VectorCoefficient interface (Eval) returns
 * @f$\boldsymbol{\xi}@f$ itself, so a Diffeomorphism can be used directly
 * wherever a coefficient of position goes: TransformedFunctionCoefficient,
 * mfem::Mesh::Transform, projection onto a nodal space. EvalGradient
 * returns
 * @f[
 *   F_{iA} = \frac{\partial \xi_i}{\partial x_A},
 * @f]
 * and the Jacobian is @f$J = \det F@f$, required positive.
 *
 * Convention of the layer (see doc/mappings.md): mappings are the identity
 * pointwise on and outside the DtN sphere, so that the outer boundary, the
 * DtN map, SurfaceHarmonics and the submesh coupling act on genuine
 * spheres unchanged. This is a precondition of the consumers, asserted in
 * drivers via MaxIdentityDeviation, not enforced here.
 */
class Diffeomorphism : public mfem::VectorCoefficient {
 public:
  explicit Diffeomorphism(int dim) : mfem::VectorCoefficient(dim) {}

  /**
   * @brief The deformation gradient at an integration point.
   * @param F The output matrix, resized to dim x dim.
   * @param T The element transformation locating the point.
   * @param ip The integration point.
   */
  virtual void EvalGradient(mfem::DenseMatrix& F,
                            mfem::ElementTransformation& T,
                            const mfem::IntegrationPoint& ip) = 0;

  /**
   * @brief Whether this mapping is the identity, exactly. False by
   * default; IdentityDiffeomorphism overrides. Used by assembly paths
   * that are only valid at @f$\varphi_e = \mathrm{id}@f$ to refuse a
   * mapped background instead of silently assembling the wrong form
   * (an identity-valued CallableDiffeomorphism reports false — use
   * IdentityDiffeomorphism for an exact identity).
   */
  virtual bool IsIdentity() const { return false; }

  /**
   * @brief The Jacobian @f$J = \det F@f$ at an integration point.
   */
  mfem::real_t Jacobian(mfem::ElementTransformation& T,
                        const mfem::IntegrationPoint& ip);

  /**
   * @brief Nanson's relation at an integration point: for a reference
   * unit normal @f$\mathbf{n}@f$ returns @f$\nu = \mathrm{cof}(F)\,
   * \mathbf{n} = J F^{-T}\mathbf{n}@f$, so that the physical surface
   * element and unit normal are @f$d\tilde S = |\nu|\,dS@f$ and
   * @f$\tilde{\mathbf{m}} = \nu/|\nu|@f$. Formed through the adjugate,
   * with no inversion of F.
   *
   * On a boundary-element transformation of a GridFunctionDiffeomorphism
   * the gradient is evaluated in the adjacent volume element (via
   * mfem::GridFunction::GetVectorGradient), which is what makes the
   * discrete change-of-variables identity hold on boundary terms: the
   * mapped mesh's boundary geometry is the trace of its volume geometry.
   */
  void MapNormal(const mfem::Vector& n, mfem::ElementTransformation& T,
                 const mfem::IntegrationPoint& ip, mfem::Vector& nu);

 private:
  mfem::DenseMatrix F_, adj_;
};

/**
 * @brief The identity mapping, exactly: @f$\boldsymbol{\xi}(\mathbf{x}) =
 * \mathbf{x}@f$, @f$F = I@f$. The default equilibrium mapping of every
 * unmapped problem, as a first-class object.
 */
class IdentityDiffeomorphism : public Diffeomorphism {
 public:
  explicit IdentityDiffeomorphism(int dim) : Diffeomorphism(dim) {}

  void Eval(mfem::Vector& V, mfem::ElementTransformation& T,
            const mfem::IntegrationPoint& ip) override {
    T.Transform(ip, V);
  }

  void EvalGradient(mfem::DenseMatrix& F, mfem::ElementTransformation&,
                    const mfem::IntegrationPoint&) override {
    F.SetSize(vdim);
    F = 0.0;
    for (int i = 0; i < vdim; i++) {
      F(i, i) = 1.0;
    }
  }

  bool IsIdentity() const override { return true; }
};

/**
 * @brief A diffeomorphism given by callables for @f$\boldsymbol{\xi}@f$ and
 * @f$F@f$ as functions of position: analytic mappings with exact
 * derivatives, the production and convergence-benchmark mode.
 */
class CallableDiffeomorphism : public Diffeomorphism {
 public:
  /// xi(x, out): the mapping as a function of position.
  using MapFunc = std::function<void(const mfem::Vector&, mfem::Vector&)>;
  /// F(x, out): its deformation gradient, F(i, A) = dxi_i/dx_A.
  using GradientFunc =
      std::function<void(const mfem::Vector&, mfem::DenseMatrix&)>;

  CallableDiffeomorphism(int dim, MapFunc xi, GradientFunc F);

  void Eval(mfem::Vector& V, mfem::ElementTransformation& T,
            const mfem::IntegrationPoint& ip) override;

  void EvalGradient(mfem::DenseMatrix& F, mfem::ElementTransformation& T,
                    const mfem::IntegrationPoint& ip) override;

 private:
  MapFunc xi_;
  GradientFunc grad_;
  mfem::Vector x_;
};

/**
 * @brief The radial diffeomorphism @f$\boldsymbol{\xi}(\mathbf{x}) =
 * f(\mathbf{x})\,\mathbf{x}@f$, position measured from the origin, with the
 * exact deformation gradient
 * @f[
 *   F = f I + \mathbf{x}\,(\nabla f)^{T}.
 * @f]
 *
 * Both @f$f@f$ and @f$\nabla f@f$ must be supplied: the analytic classes
 * are exact by design, and a caller with @f$f@f$ alone should go through
 * Interpolate() instead, which states honestly that the gradient is then a
 * discrete one.
 */
class RadialDiffeomorphism : public Diffeomorphism {
 public:
  /**
   * @brief From coefficients for @f$f@f$ and @f$\nabla f@f$ (not owned).
   */
  RadialDiffeomorphism(int dim, mfem::Coefficient& f,
                       mfem::VectorCoefficient& grad_f);

  /**
   * @brief From a radial profile: @f$f = f(r)@f$ and its derivative
   * @f$f'(r)@f$, so that @f$\nabla f = f'(r)\,\mathbf{x}/r@f$.
   *
   * For the mapping to be smooth at the origin the profile needs
   * @f$f'(0) = 0@f$ (or the origin outside the domain); at @f$r = 0@f$ the
   * gradient is taken as zero.
   */
  RadialDiffeomorphism(int dim, std::function<mfem::real_t(mfem::real_t)> f,
                       std::function<mfem::real_t(mfem::real_t)> df);

  void Eval(mfem::Vector& V, mfem::ElementTransformation& T,
            const mfem::IntegrationPoint& ip) override;

  void EvalGradient(mfem::DenseMatrix& F, mfem::ElementTransformation& T,
                    const mfem::IntegrationPoint& ip) override;

 private:
  mfem::Coefficient* f_ = nullptr;
  mfem::VectorCoefficient* grad_f_ = nullptr;
  std::function<mfem::real_t(mfem::real_t)> fr_, dfr_;
  mfem::Vector x_, g_;

  mfem::real_t Scalar(mfem::ElementTransformation& T,
                      const mfem::IntegrationPoint& ip);
};

/**
 * @brief The radial blend of a mapping to the identity: the buffer-taper
 * rule for an equilibrium mapping that is non-trivial on the physical
 * surface (doc/gravitating_elasticity.md §3, "Gravity: three
 * formulations"; NewHarmonicExtensionMapping is the elliptic alternative
 * for a mapping known only on the body).
 *
 * @f[
 *   \boldsymbol{\varphi}(\mathbf{x}) = \mathbf{x}
 *     + t(r)\,\bigl(\boldsymbol{\xi}(\mathbf{x}) - \mathbf{x}\bigr),
 *   \qquad
 *   F = I + t\,(F_\xi - I)
 *     + \frac{t'(r)}{r}\,(\boldsymbol{\xi}-\mathbf{x})\otimes\mathbf{x},
 * @f]
 * with the cubic smoothstep @f$t = 1 - s^2(3-2s)@f$,
 * @f$s = (r - r_{\mathrm{inner}})/(r_{\mathrm{outer}} -
 * r_{\mathrm{inner}})@f$ clamped to @f$[0,1]@f$: @f$\varphi = \xi@f$
 * exactly (values and gradient) for @f$r \le r_{\mathrm{inner}}@f$, the
 * identity exactly for @f$r \ge r_{\mathrm{outer}}@f$, and @f$C^1@f$
 * across both seams (@f$t' = 0@f$ at both ends), so the DtN-sphere
 * convention holds with @f$F = I@f$ there as well. The inner mapping
 * must be evaluable through the taper region and is not owned.
 * Diffeomorphy of the blend is the caller's responsibility (moderate
 * displacements; the consumers' @f$J > 0@f$ checks apply).
 */
class TaperedDiffeomorphism : public Diffeomorphism {
 public:
  TaperedDiffeomorphism(Diffeomorphism& xi, mfem::real_t r_inner,
                        mfem::real_t r_outer);

  void Eval(mfem::Vector& V, mfem::ElementTransformation& T,
            const mfem::IntegrationPoint& ip) override;

  void EvalGradient(mfem::DenseMatrix& F, mfem::ElementTransformation& T,
                    const mfem::IntegrationPoint& ip) override;

 private:
  Diffeomorphism* xi_;
  mfem::real_t r0_, r1_;
  mfem::Vector x_, y_;
  mfem::DenseMatrix Fx_;

  /// t(r) and t'(r) of the clamped smoothstep.
  void Taper(mfem::real_t r, mfem::real_t& t, mfem::real_t& dt) const;
};

/**
 * @brief The diffeomorphism @f$\boldsymbol{\xi}(\mathbf{x}) = \mathbf{x} +
 * \mathbf{h}(\mathbf{x})@f$ for a displacement grid function, with
 * @f$F = I + \nabla\mathbf{h}@f$ evaluated through the element
 * transformation.
 *
 * The discrete representative of the layer: the interpolated mode of the
 * change-of-variables identity (see Interpolate), mappings supplied in
 * discretised form (e.g. by planetmodel), and the output of
 * NewHarmonicExtensionMapping.
 */
class GridFunctionDiffeomorphism : public Diffeomorphism {
 public:
  /**
   * @brief From a vector displacement grid function (not owned) with
   * vdim equal to the space dimension.
   */
  explicit GridFunctionDiffeomorphism(const mfem::GridFunction& h);

  /**
   * @brief Owning constructor: takes the collection, space and
   * displacement built elsewhere (used by Interpolate).
   */
  GridFunctionDiffeomorphism(
      std::unique_ptr<mfem::FiniteElementCollection> fec,
      std::unique_ptr<mfem::FiniteElementSpace> fes,
      std::unique_ptr<mfem::GridFunction> h);

  GridFunctionDiffeomorphism(GridFunctionDiffeomorphism&&) = default;

  /// The displacement @f$\mathbf{h}@f$.
  const mfem::GridFunction& Displacement() const { return *h_; }

  void Eval(mfem::Vector& V, mfem::ElementTransformation& T,
            const mfem::IntegrationPoint& ip) override;

  void EvalGradient(mfem::DenseMatrix& F, mfem::ElementTransformation& T,
                    const mfem::IntegrationPoint& ip) override;

 private:
  std::unique_ptr<mfem::FiniteElementCollection> owned_fec_;
  std::unique_ptr<mfem::FiniteElementSpace> owned_fes_;
  std::unique_ptr<mfem::GridFunction> owned_h_;
  const mfem::GridFunction* h_;
  mfem::Vector x_;
};

/**
 * @brief The pull-back @f$f \circ \boldsymbol{\xi}@f$ of a function of
 * position by a mapping @f$\boldsymbol{\xi}@f$ (not owned): a field given
 * on the physical domain, evaluated on the reference domain.
 */
class TransformedFunctionCoefficient : public mfem::Coefficient {
 public:
  TransformedFunctionCoefficient(
      mfem::VectorCoefficient& xi,
      std::function<mfem::real_t(const mfem::Vector&)> f)
      : xi_{&xi}, f_{std::move(f)} {}

  mfem::real_t Eval(mfem::ElementTransformation& T,
                    const mfem::IntegrationPoint& ip) override;

 private:
  mfem::VectorCoefficient* xi_;
  std::function<mfem::real_t(const mfem::Vector&)> f_;
};

/**
 * @brief The pull-back @f$\mathbf{f} \circ \boldsymbol{\xi}@f$ of a vector
 * function of position by a mapping @f$\boldsymbol{\xi}@f$ (not owned): a
 * vector field given on the physical domain, evaluated on the reference
 * domain, components untouched (relabelling does not rotate ambient
 * components). For a field that is physically a gradient, use
 * PullbackGradientCoefficient instead.
 */
class TransformedVectorFunctionCoefficient : public mfem::VectorCoefficient {
 public:
  TransformedVectorFunctionCoefficient(
      mfem::VectorCoefficient& xi,
      std::function<void(const mfem::Vector&, mfem::Vector&)> f)
      : mfem::VectorCoefficient(xi.GetVDim()), xi_{&xi}, f_{std::move(f)} {}

  void Eval(mfem::Vector& V, mfem::ElementTransformation& T,
            const mfem::IntegrationPoint& ip) override;

 private:
  mfem::VectorCoefficient* xi_;
  std::function<void(const mfem::Vector&, mfem::Vector&)> f_;
  mfem::Vector y_;
};

/**
 * @brief The matrix analogue of TransformedFunctionCoefficient:
 * @f$M(\xi(x))@f$ for a matrix-valued function of physical position (the
 * referential expression of a physical matrix field, e.g. a stress).
 */
class TransformedMatrixFunctionCoefficient : public mfem::MatrixCoefficient {
 public:
  TransformedMatrixFunctionCoefficient(
      int dim, mfem::VectorCoefficient& xi,
      std::function<void(const mfem::Vector&, mfem::DenseMatrix&)> f)
      : mfem::MatrixCoefficient(dim), xi_{&xi}, f_{std::move(f)} {}

  void Eval(mfem::DenseMatrix& M, mfem::ElementTransformation& T,
            const mfem::IntegrationPoint& ip) override;

 private:
  mfem::VectorCoefficient* xi_;
  std::function<void(const mfem::Vector&, mfem::DenseMatrix&)> f_;
  mfem::Vector y_;
};

/**
 * @brief The relabelling (Piola) transformation of a symmetric referential
 * stress: @f$\tilde S = J_\xi F_\xi^{-1} (S\circ\xi) F_\xi^{-T}@f$ — the
 * second Piola–Kirchhoff equilibrium stress seen from a relabelled
 * reference (doc/gravitating_elasticity.md §1). The inner coefficient
 * must already be the referential expression @f$S\circ\xi@f$. For
 * @f$S = -p\mathbf{1}@f$ this is @f$-p\,J C^{-1}@f$, the pull-back
 * diffusion tensor scaled by the pressure.
 */
class PullbackStressCoefficient : public mfem::MatrixCoefficient {
 public:
  PullbackStressCoefficient(int dim, mfem::MatrixCoefficient& S_composed,
                            Diffeomorphism& xi)
      : mfem::MatrixCoefficient(dim), S_(&S_composed), xi_(&xi) {}

  void Eval(mfem::DenseMatrix& M, mfem::ElementTransformation& T,
            const mfem::IntegrationPoint& ip) override;

 private:
  mfem::MatrixCoefficient* S_;
  Diffeomorphism* xi_;
  mfem::DenseMatrix F_, Sq_, tmp_;
};

/**
 * @brief The Jacobian @f$J = \det F@f$ of a mapping (not owned) as a
 * scalar coefficient: the factor of every pulled-back volume term.
 */
class JacobianCoefficient : public mfem::Coefficient {
 public:
  explicit JacobianCoefficient(Diffeomorphism& xi) : xi_(&xi) {}

  mfem::real_t Eval(mfem::ElementTransformation& T,
                    const mfem::IntegrationPoint& ip) override;

 private:
  Diffeomorphism* xi_;
};

/**
 * @brief The matrix @f$a = J C^{-1} = J F^{-1} F^{-T}@f$ of a mapping (not
 * owned): the pulled-back Laplace form
 * @f$\int \nabla\varphi \cdot a \cdot \nabla\varphi' \, dx@f$, consumed by
 * TransformedDiffusionIntegrator.
 */
class PullbackDiffusionCoefficient : public mfem::MatrixCoefficient {
 public:
  explicit PullbackDiffusionCoefficient(Diffeomorphism& xi)
      : mfem::MatrixCoefficient(xi.GetVDim()), xi_(&xi) {}

  void Eval(mfem::DenseMatrix& K, mfem::ElementTransformation& T,
            const mfem::IntegrationPoint& ip) override;

 private:
  Diffeomorphism* xi_;
  mfem::DenseMatrix F_;
};

/**
 * @brief The physical gradient @f$F^{-T} \mathbf{v}@f$ of a referential
 * gradient coefficient @f$\mathbf{v} = \nabla\varphi@f$ (neither owned):
 * how gradients of background fields (e.g. @f$\nabla\Phi_0@f$) enter the
 * pulled-back forms.
 */
class PullbackGradientCoefficient : public mfem::VectorCoefficient {
 public:
  PullbackGradientCoefficient(Diffeomorphism& xi, mfem::VectorCoefficient& v)
      : mfem::VectorCoefficient(xi.GetVDim()), xi_(&xi), v_(&v) {}

  void Eval(mfem::Vector& V, mfem::ElementTransformation& T,
            const mfem::IntegrationPoint& ip) override;

 private:
  Diffeomorphism* xi_;
  mfem::VectorCoefficient* v_;
  mfem::Vector w_;
  mfem::DenseMatrix F_;
};

/**
 * @brief The Nanson area factor @f$|\nu| = |\mathrm{cof}(F)\,\mathbf{n}|
 * = d\tilde S/dS@f$ of a mapping (not owned) on boundary elements: the
 * factor that turns a physical surface density into its referential
 * expression for the stock boundary linear-form integrators (multiply
 * the referential load coefficient by it, as JacobianCoefficient does
 * for volume densities).
 *
 * Only defined on boundary-element transformations; evaluation on any
 * other transformation aborts. Returns zero where the boundary element
 * is degenerate.
 */
class NansonAreaCoefficient : public mfem::Coefficient {
 public:
  explicit NansonAreaCoefficient(Diffeomorphism& xi) : xi_(&xi) {}

  mfem::real_t Eval(mfem::ElementTransformation& T,
                    const mfem::IntegrationPoint& ip) override;

 private:
  Diffeomorphism* xi_;
  mfem::Vector n_, nu_;
};

/**
 * @brief The mapped counterpart of BoundaryNormalDotCoefficient:
 * @f$\tilde{\mathbf{m}}\cdot\mathbf{V} = (\nu\cdot\mathbf{V})/|\nu|@f$
 * on boundary elements, with @f$\tilde{\mathbf{m}}@f$ the *physical*
 * unit normal of the mapped surface and @f$\mathbf{V}@f$ the physical
 * vector field expressed referentially (neither owned; for a field that
 * is physically a gradient, compose with PullbackGradientCoefficient).
 *
 * With @f$\mathbf{V} = F^{-T}\nabla\Phi_0@f$ this gives the
 * @f$\tilde{\mathbf{m}}\cdot\tilde\nabla\tilde\Phi_0@f$ factor of the
 * fluid–solid interface term on a mapped interface. Only defined on
 * boundary-element transformations; aborts otherwise. Returns zero where
 * the boundary element or the mapped normal is degenerate.
 */
class MappedBoundaryNormalDotCoefficient : public mfem::Coefficient {
 public:
  MappedBoundaryNormalDotCoefficient(mfem::VectorCoefficient& V,
                                     Diffeomorphism& xi)
      : V_(&V), xi_(&xi) {}

  mfem::real_t Eval(mfem::ElementTransformation& T,
                    const mfem::IntegrationPoint& ip) override;

 private:
  mfem::VectorCoefficient* V_;
  Diffeomorphism* xi_;
  mfem::Vector n_, nu_, v_;
};

/**
 * @brief The nodal interpolant @f$\boldsymbol{\xi}_h@f$ of a mapping on
 * the mesh's geometric (nodal) space, as a GridFunctionDiffeomorphism
 * owning its displacement @f$\boldsymbol{\xi}_h - \mathbf{x}@f$.
 *
 * The interpolation space copies the mesh's nodal collection, order and
 * ordering (H1 order 1 for a mesh without nodes), so the interpolant's
 * geometry is exactly that of MappedMesh(mesh, xi): assembling mapped
 * integrators on `mesh` with the returned mapping reproduces, element
 * matrix by element matrix, the standard integrators on the mapped mesh,
 * provided the same integration rule is used on both. This is the discrete
 * change-of-variables identity (doc/mappings.md, "The discrete
 * change-of-variables identity").
 *
 * The mesh must outlive the returned object.
 */
GridFunctionDiffeomorphism Interpolate(Diffeomorphism& xi, mfem::Mesh& mesh);

/**
 * @brief A copy of the mesh with its nodes moved to the nodal interpolant
 * of the mapping: the element-by-element image mesh.
 *
 * The copy keeps the mesh's geometric order (a mesh without nodes is given
 * order-1 nodes first), so the image geometry matches Interpolate(xi,
 * mesh) on the original. Set the curvature of the reference mesh *before*
 * calling, to the order the comparison is to be made at.
 */
mfem::Mesh MappedMesh(const mfem::Mesh& mesh, Diffeomorphism& xi);

/**
 * @brief The maximum of @f$|\boldsymbol{\xi}(\mathbf{x}) - \mathbf{x}|@f$
 * over quadrature points of the marked boundary elements.
 *
 * The seatbelt for the layer's convention that mappings are the identity
 * on and outside the DtN sphere: drivers wrap this in an MFEM_VERIFY
 * against a tolerance at setup. Mappings should satisfy the convention by
 * construction; this detects the taper that does not quite reach zero.
 *
 * @param xi The mapping.
 * @param mesh The mesh.
 * @param bdr_marker Boundary attributes to include (1) or exclude (0).
 */
mfem::real_t MaxIdentityDeviation(Diffeomorphism& xi, mfem::Mesh& mesh,
                                  const mfem::Array<int>& bdr_marker);

#ifdef MFEM_USE_MPI

/**
 * @brief Parallel Interpolate: the interpolant lives on a
 * ParFiniteElementSpace over the parallel mesh; otherwise as the serial
 * version.
 */
GridFunctionDiffeomorphism Interpolate(Diffeomorphism& xi,
                                       mfem::ParMesh& mesh);

/**
 * @brief Parallel MappedMesh: a copy of the parallel mesh with its nodes
 * moved to the nodal interpolant of the mapping.
 */
mfem::ParMesh MappedMesh(const mfem::ParMesh& mesh, Diffeomorphism& xi);

/**
 * @brief Parallel MaxIdentityDeviation: the global maximum over the
 * mesh's communicator.
 */
mfem::real_t MaxIdentityDeviation(Diffeomorphism& xi, mfem::ParMesh& mesh,
                                  const mfem::Array<int>& bdr_marker);

#endif

}  // namespace AdGIA
