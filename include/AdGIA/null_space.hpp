/**
 * @file null_space.hpp
 * @brief Singular symmetric systems: the rigid-body modes as
 * VectorCoefficients, an orthonormal basis of a (near-)null space
 * (NullSpaceProjector, MakeRigidModeProjector()), and the projected operator
 * and solver built on it. The rationale (which inner product projects, which
 * gauges, why the projected system is solved) is in doc/null_space.md.
 */

#pragma once

#include <memory>
#include <vector>

#include "mfem.hpp"
#include "AdGIA/mappings.hpp"

namespace AdGIA {

/**
 * @brief Implementation of rigid body translations as a VectorCoefficient.
 *
 * This class represents a rigid body translation for use with MFEM's
 * VectorCoefficient. It defines a constant vector field where only
 * one component is non-zero (specifically, 1.0), representing a translation
 * along a specific axis.
 */
class RigidTranslation : public mfem::VectorCoefficient {
 private:
  int component_; /**< The component (spatial dimension) along which the
                       translation occurs (0 for x, 1 for y, 2 for z). */

 public:
  /**
   * @brief Constructor for the RigidTranslation class.
   *
   * To form a class instance, the spatial dimension and the component
   * of the translation are specified.
   *
   * @param dimension The spatial dimension of the problem (e.g., 2 for 2D, 3
   * for 3D).
   * @param component The index of the component (0, 1, or 2) along which the
   * translation is applied. Must be less than `dimension`.
   */
  RigidTranslation(int dimension, int component);

  /**
   * @brief Resets the component value for the translation.
   * @param component The new component index for the translation.
   */
  void SetComponent(int component);

  /**
   * @brief Overload of the Eval method for VectorCoefficients.
   *
   * This method evaluates the rigid translation vector at a given point.
   * The resulting vector will have a value of 1.0 in the specified `component_`
   * and 0.0 in all other components.
   *
   * @param V The output vector where the evaluated translation will be stored.
   * @param T The element transformation object.
   * @param ip The integration point where the coefficient is evaluated.
   */
  void Eval(mfem::Vector &V, mfem::ElementTransformation &T,
            const mfem::IntegrationPoint &ip) override;
};

/**
 * @brief Implementation of rigid body rotations as a VectorCoefficient.
 *
 * This class represents a rigid body rotation for use with MFEM's
 * VectorCoefficient. It defines a vector field corresponding to a
 * rotation about a specific axis.
 *
 * @note If the spatial dimension is 3, then all three components (x, y, z)
 * of the rotation can be defined. If the spatial dimension is 2, only
 * the 2 (i.e. z) component of the rotation is defined (rotation in the XY
 * plane).
 */
class RigidRotation : public mfem::VectorCoefficient {
 private:
  int component_; /**< The component representing the axis of rotation.
                       In 2D, only 2 (z-axis) is valid. In 3D, 0 (x-axis),
                       1 (y-axis), or 2 (z-axis) are valid. */

#ifndef MFEM_THREAD_SAFE
  mfem::Vector
      x_; /**< Internal buffer for the transformed spatial coordinates. */
#endif

 public:
  /**
   * @brief Constructor for the RigidRotation class.
   *
   * To form a class instance, the spatial dimension and the component
   * of the rotation axis are specified.
   *
   * @param dimension The spatial dimension of the problem (e.g., 2 for 2D, 3
   * for 3D).
   * @param component The index of the rotation axis (0 for x, 1 for y, 2 for
   * z). In 2D, this must be 2 (z-axis rotation).
   */
  RigidRotation(int dimension, int component);

  /**
   * @brief Resets the component value for the rotation axis.
   * @param component The new component index for the rotation axis.
   */
  void SetComponent(int component);

  /**
   * @brief Overload of the Eval method for VectorCoefficients.
   *
   * This method evaluates the rigid rotation vector at a given point `ip`.
   * The rotation vector is computed based on the spatial coordinates and the
   * specified rotation axis.
   *
   * @param V The output vector where the evaluated rotation will be stored.
   * @param T The element transformation object.
   * @param ip The integration point where the coefficient is evaluated.
   */
  void Eval(mfem::Vector &V, mfem::ElementTransformation &T,
            const mfem::IntegrationPoint &ip) override;
};

/**
 * @brief The rigid rotation of the *mapped* positions, @f$W\varphi(x)@f$,
 * about the axis @p component (2-D: the single in-plane rotation, component
 * 2): the strain-free rotational mode of a problem posed on a fixed
 * reference body with equilibrium mapping @f$\varphi@f$. Reduces to
 * RigidRotation at the identity.
 */
class MappedRotation : public mfem::VectorCoefficient {
 public:
  MappedRotation(Diffeomorphism &map, int component)
      : mfem::VectorCoefficient(map.GetVDim()), map_(&map), c_(component) {}

  void Eval(mfem::Vector &V, mfem::ElementTransformation &T,
            const mfem::IntegrationPoint &ip) override {
    map_->Eval(y_, T, ip);
    V.SetSize(vdim);
    if (vdim == 2) {
      V(0) = -y_(1);
      V(1) = y_(0);
    } else {
      // V = e_c x y: V_a = -y_b, V_b = y_a with (c, a, b) cyclic.
      const int a = (c_ + 1) % 3, b = (c_ + 2) % 3;
      V(c_) = 0.0;
      V(a) = -y_(b);
      V(b) = y_(a);
    }
  }

 private:
  Diffeomorphism *map_;
  int c_;
  mfem::Vector y_;
};

/**
 * @brief An orthonormal basis of a (near-)null space of an operator, with the
 * Euclidean projection onto its orthogonal complement; serial or parallel
 * (true-dof vectors, global inner products).
 *
 * Vectors are added one at a time and orthonormalised by modified
 * Gram-Schmidt; a vector that is (numerically) dependent on the ones already
 * present is dropped. Typical null vectors are the rigid modes of a
 * displacement space (AddRigidModes(), MakeRigidModeProjector()), the same
 * modes as block vectors of the coupled displacement/potential system of a
 * self-gravitating body, or the constant potential in two dimensions.
 */
class NullSpaceProjector {
 public:
  NullSpaceProjector() = default;

#ifdef MFEM_USE_MPI
  explicit NullSpaceProjector(MPI_Comm comm) : comm_(comm), parallel_(true) {}
#endif

  /**
   * @brief Add @p v to the basis (orthonormalised against the existing
   * vectors). Returns false, and adds nothing, if the remainder is below
   * @p drop_tol times the norm of @p v.
   */
  bool Add(const mfem::Vector& v, mfem::real_t drop_tol = 1e-10);

  /** @brief Number of basis vectors. */
  int Size() const { return static_cast<int>(basis_.size()); }

  /** @brief Orthonormal basis vector @p i. */
  const mfem::Vector& Basis(int i) const { return *basis_[i]; }

  /** @brief x <- (I - sum_i n_i n_i^T) x. */
  void Project(mfem::Vector& x) const;

  /** @brief y = (I - sum_i n_i n_i^T) x. */
  void Project(const mfem::Vector& x, mfem::Vector& y) const {
    y = x;
    Project(y);
  }

  /** @brief Inner product, global in parallel. */
  mfem::real_t Dot(const mfem::Vector& x, const mfem::Vector& y) const;

 private:
  std::vector<std::unique_ptr<mfem::Vector>> basis_;
#ifdef MFEM_USE_MPI
  MPI_Comm comm_ = MPI_COMM_NULL;
#endif
  bool parallel_ = false;
};

/**
 * @brief Append the linearised rigid modes of the displacement space @p fes
 * (the translations and, in two dimensions, the in-plane rotation, or all
 * three rotations in three dimensions) to @p P as true-dof vectors. Returns
 * the number actually added (a mode already spanned by @p P is dropped).
 *
 * With a non-null @p map the rotations are those of the *mapped* positions
 * @f$W\varphi_e(x)@f$ (MappedRotation) — the strain-free rotational modes
 * of a problem posed on a fixed reference body with a non-natural
 * reference state; translations are unchanged.
 *
 * @p P must use the communicator of @p fes when the space is parallel. The
 * space must have vdim 2 or 3. On a curved mesh the rotations lie in the
 * discrete space only when the displacement order is at least the geometry
 * order (doc/null_space.md, "Element order on curved meshes").
 */
int AddRigidModes(NullSpaceProjector& P, mfem::FiniteElementSpace& fes,
                  Diffeomorphism* map = nullptr);

/**
 * @brief A projector holding exactly the rigid modes of @p fes (serial or
 * parallel, the communicator taken from the space); the null-space handling
 * of a pure traction problem. Rotations are mapped through @p map when one
 * is given (see AddRigidModes).
 */
std::unique_ptr<NullSpaceProjector> MakeRigidModeProjector(
    mfem::FiniteElementSpace& fes, Diffeomorphism* map = nullptr);

/**
 * @brief The operator @f$P A P@f$ for a projector @f$P@f$ from a
 * NullSpaceProjector: @f$A@f$ restricted to the orthogonal complement of the
 * null space. Symmetric when @f$A@f$ is; the natural operator to hand to a
 * Krylov method for a singular or nearly singular symmetric system.
 */
class ProjectedOperator : public mfem::Operator {
 public:
  ProjectedOperator(const mfem::Operator& A, const NullSpaceProjector& P)
      : mfem::Operator(A.Height(), A.Width()), A_(&A), P_(&P) {
    MFEM_VERIFY(A.Height() == A.Width(),
                "ProjectedOperator: the operator must be square.");
  }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

 private:
  const mfem::Operator* A_;
  const NullSpaceProjector* P_;
  mutable mfem::Vector z_;
};

/**
 * @brief Wraps a solver for a (nearly) singular symmetric system: the
 * operator handed to the inner solver is @f$P A P@f$, the right-hand side and
 * (in iterative mode) the initial guess are projected before, and the
 * solution after, the inner solve. With MakeRigidModeProjector() this is the
 * solver of a free body under traction.
 */
class ProjectedSolver : public mfem::Solver {
 public:
  explicit ProjectedSolver(const NullSpaceProjector& P)
      : mfem::Solver(0, false), P_(&P) {}

  /** @brief The inner solver; call before SetOperator(). */
  void SetSolver(mfem::Solver& solver);

  /** @brief Set @f$A@f$: the inner solver receives @f$P A P@f$. */
  void SetOperator(const mfem::Operator& op) override;

  void Mult(const mfem::Vector& b, mfem::Vector& x) const override;

  /**
   * @brief Choose the representative of the solution: after each solve,
   * remove the component of @f$x@f$ in the span of the basis so that
   * @f$n_i^T M x = 0@f$ for every basis vector (with @f$M@f$ the
   * @f$\rho@f$-weighted vector mass matrix and the basis the rigid modes:
   * zero net momentum and angular momentum), instead of the Euclidean
   * @f$n_i^T x = 0@f$ of the projector. The inner solve is unchanged (it
   * needs the Euclidean projector to keep @f$PAP@f$ symmetric); only the
   * null-space component of the result differs. @p M must be symmetric
   * positive semi-definite; basis vectors with zero @f$M@f$-norm (the
   * constant potential of a coupled system, say) keep the Euclidean
   * gauge, the others must be independent under @f$M@f$. Null restores the
   * Euclidean gauge. Call after the basis is complete.
   */
  void SetGauge(const mfem::Operator* M);

 private:
  void ApplyGauge(mfem::Vector& x) const;

  const NullSpaceProjector* P_;
  mfem::Solver* solver_ = nullptr;
  std::unique_ptr<ProjectedOperator> projected_;
  mutable mfem::Vector b_;
  std::vector<int> gauge_idx_;    ///< basis vectors with nonzero M-norm
  std::vector<mfem::Vector> Mn_;  ///< M n_i for those
  mfem::DenseMatrix Ginv_;        ///< (n_i . M n_j)^{-1} over those
};

}  // namespace AdGIA