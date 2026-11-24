/**
 * @file nonlinear_problem.hpp
 * @brief Static problems at finite strain about a given equilibrium
 * (doc/methods/finite_strain.md).
 */

#pragma once

#include <memory>
#include <vector>

#include "AdGIA/material_law.hpp"
#include "AdGIA/quasi_static_problem.hpp"
#include "AdGIA/referential_problem.hpp"
#include "mfem.hpp"

namespace AdGIA {

/** @brief The options of a Newton solve with a frozen or refreshed
 * tangent. */
struct NewtonOptions {
  /// The iteration stops when the L2 norm of a correction is at most
  /// rel_tol times that of the displacement, or at most abs_tol.
  mfem::real_t rel_tol = 1e-10;
  mfem::real_t abs_tol = 0.0;
  /// The most iterations taken.
  int max_iterations = 25;
  /// The tangent is assembled at the state of the first iteration and
  /// kept until a correction exceeds this fraction of the one before it.
  /// That correction is not taken, and from then on the tangent is
  /// assembled at every iteration. Zero or less assembles it at every
  /// iteration from the start, which is Newton's method proper.
  mfem::real_t refresh_contraction = 0.1;
  /// Relative tolerance of the linear solves.
  mfem::real_t linear_rel_tol = 1e-12;
  /// Print one line for each iteration, on the root rank.
  bool print = false;
};

/**
 * @brief The static deformation at finite strain of a body without
 * gravity, clamped on part of its boundary, under dead loads.
 *
 * The reference configuration is a given equilibrium, with the
 * displacement @f$\mathbf u@f$ from it the unknown and
 * @f$\mathbf F = \mathbf 1 + \nabla\mathbf u@f$. The body has the strain
 * energy of a MaterialLaw. The loads are dead: a traction per unit
 * reference area on part of the boundary and a force per unit reference
 * volume, both fixed in direction and magnitude. The problem is the
 * stationarity of
 * @f[
 *   \Pi[\mathbf u] = \int_B \bigl[W(X, \mathbf E) - \mathbf S_0 :
 *   \nabla\mathbf u\bigr]\,dV - \int_B \mathbf b\cdot\mathbf u\,dV
 *   - \int_{\Gamma_t} \mathbf t\cdot\mathbf u\,dS
 * @f]
 * over displacements with the prescribed values on the clamped part,
 * where @f$\mathbf S_0@f$ is the law's stress at the reference. The term
 * in @f$\mathbf S_0@f$ makes the residual incremental,
 * @f[
 *   r[\mathbf u](\mathbf v) = \int_B (\mathbf F\mathbf S - \mathbf S_0) :
 *   \nabla\mathbf v\,dV - \int_B \mathbf b\cdot\mathbf v\,dV
 *   - \int_{\Gamma_t} \mathbf t\cdot\mathbf v\,dS,
 * @f]
 * so that the reference is a solution without loads whatever the
 * equilibrium stress of the law, which the discrete fields balance only
 * to the accuracy of the mesh.
 *
 * The tangent is a LinearQuasiStaticClampedProblem owned by this one,
 * with the material and geometric stiffness of the law at the current
 * state. Solve() iterates: the residual at the current state, one solve
 * with the tangent, the correction added. A correction takes on the
 * clamped part what the displacement lacks of its prescribed value, so
 * that from the reference the first iteration is the linear problem with
 * the boundary data. The tangent is assembled at the first state and
 * kept while the corrections contract as fast as
 * NewtonOptions::refresh_contraction asks; after that it is assembled at
 * every iteration.
 *
 * The tangent must be positive definite on displacements that vanish on
 * the clamped part; the linear solver does not detect that it is not.
 * With an equilibrium stress this asks that the stress be the
 * equilibrium of the body under its own loads: where the boundary is
 * free or carries a dead load, a stress that pushes on it (a pressure,
 * say) is held in the true problem by a load that follows the surface,
 * which this class does not have, and the tangent without it can lose
 * definiteness.
 *
 * Serial or parallel by the space.
 */
class NonlinearClampedProblem {
 public:
  /**
   * @param fes Displacement space; serial or parallel; not owned.
   * @param law The material; not owned, must outlive the problem.
   * @param ess_bdr Boundary attributes with prescribed displacement
   * (copied).
   * @param traction The traction per unit reference area; not owned.
   * @param traction_marker Boundary attributes it acts on (copied).
   * @param dirichlet The prescribed displacement; nullptr means zero.
   */
  NonlinearClampedProblem(mfem::FiniteElementSpace* fes, MaterialLaw& law,
                          const mfem::Array<int>& ess_bdr,
                          mfem::VectorCoefficient& traction,
                          const mfem::Array<int>& traction_marker,
                          mfem::VectorCoefficient* dirichlet = nullptr);
  ~NonlinearClampedProblem();

  /** @brief A force per unit reference volume; not owned. Call before
   * the first evaluation. */
  void SetBodyForce(mfem::VectorCoefficient& force);

  NewtonOptions& Options() { return options_; }
  const NewtonOptions& Options() const { return options_; }

  /** @brief True if the displacement space is a ParFiniteElementSpace. */
  bool IsParallel() const;

  mfem::FiniteElementSpace& DisplacementSpace() { return *fes_; }

  /** @brief The displacement: the solution after Solve(), and the state
   * at which Energy(), Residual() and Tangent() are evaluated. Writing
   * to it sets that state, and Solve() starts from it. */
  mfem::GridFunction& Displacement() { return *u_; }
  const mfem::GridFunction& Displacement() const { return *u_; }

  /** @brief The functional @f$\Pi@f$ at the current displacement. */
  mfem::real_t Energy();

  /** @brief The residual at the current displacement, a dual vector in
   * the vdof (L-vector) layout of the displacement space: its pairing
   * with a displacement that vanishes on the clamped part is the first
   * variation of Energy(). */
  void Residual(mfem::Vector& r);

  /** @brief The tangent at the current displacement: the linear problem
   * owned by this one, its operator marked for assembly at this state. */
  LinearQuasiStaticClampedProblem& Tangent();

  /** @brief Solve from the current displacement. Returns false if the
   * iteration did not meet its tolerance or a linear solve failed. */
  bool Solve();

  /** @brief Corrections taken by the last Solve(). */
  int Iterations() const { return static_cast<int>(corrections_.size()); }
  /** @brief The L2 norms of the corrections taken by the last Solve(), in
   * order. */
  const std::vector<mfem::real_t>& Corrections() const { return corrections_; }
  /** @brief Assemblies of the tangent so far. */
  int TangentAssemblies() const;

 private:
  mfem::real_t Norm(const mfem::GridFunction& v) const;

  mfem::FiniteElementSpace* fes_;
  MaterialLaw* law_;
  mfem::Array<int> ess_bdr_, traction_marker_;
  mfem::VectorCoefficient* traction_;
  mfem::VectorCoefficient* dirichlet_;
  mfem::VectorCoefficient* body_force_ = nullptr;
  NewtonOptions options_;

  std::unique_ptr<mfem::GridFunction> u_;
  std::unique_ptr<GridFunctionDiffeomorphism> map_;
  std::unique_ptr<StrainEnergyDensityCoefficient> energy_c_;
  std::unique_ptr<FirstPiolaKirchhoffStressCoefficient> stress_c_;
  std::unique_ptr<SecondPiolaKirchhoffStressCoefficient> tangent_stress_c_;
  std::unique_ptr<MaterialTangentCoefficient> tangent_c_;
  std::unique_ptr<ReferentialElasticRheology> rheology_;
  std::unique_ptr<mfem::VectorConstantCoefficient> zero_;
  std::unique_ptr<mfem::VectorGridFunctionCoefficient> state_c_;
  std::unique_ptr<mfem::VectorSumCoefficient> lacking_c_;
  std::unique_ptr<LinearQuasiStaticClampedProblem> tangent_;
  std::unique_ptr<mfem::LinearForm> internal_, load_;
  bool load_assembled_ = false;
  std::vector<mfem::real_t> corrections_;
};

}  // namespace AdGIA
