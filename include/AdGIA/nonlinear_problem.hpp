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
   * @param equilibrium The mapping of the mesh onto the equilibrium
   * configuration; nullptr is the identity. With a mapping the mesh is a
   * reference of labels: the law is taken about the image of the mesh
   * (MappedMaterialLaw), the displacement is from that image, and the
   * loads are per unit area and volume of the mesh. Not owned.
   */
  NonlinearClampedProblem(mfem::FiniteElementSpace* fes, MaterialLaw& law,
                          const mfem::Array<int>& ess_bdr,
                          mfem::VectorCoefficient& traction,
                          const mfem::Array<int>& traction_marker,
                          mfem::VectorCoefficient* dirichlet = nullptr,
                          Diffeomorphism* equilibrium = nullptr);
  ~NonlinearClampedProblem();

  /** @brief A force per unit reference volume; not owned. Call before
   * the first evaluation. */
  void SetBodyForce(mfem::VectorCoefficient& force);

  /** @brief A dead load given by a stress field @f$\mathbf M@f$,
   * @f$\int_B \mathbf M : \nabla\mathbf v\,dV@f$: the body force
   * @f$-\operatorname{Div}\mathbf M@f$ with the traction
   * @f$\mathbf M\mathbf N@f$ on the boundary, in one. Not owned. Once,
   * before the first evaluation. */
  void SetStressLoad(mfem::MatrixCoefficient& stress);

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
  mfem::MatrixCoefficient* stress_load_ = nullptr;
  NewtonOptions options_;

  std::unique_ptr<mfem::GridFunction> u_;
  std::unique_ptr<IdentityDiffeomorphism> identity_;
  std::unique_ptr<MappedMaterialLaw> mapped_law_;
  std::unique_ptr<DisplacedDiffeomorphism> map_;
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

/**
 * @brief The static deformation at finite strain of a self-gravitating
 * solid body, in the referential form.
 *
 * The body @f$B@f$ sits in a ball whose outer sphere carries the exterior
 * potential's Dirichlet-to-Neumann condition, with a buffer shell of
 * vacuum between. The reference configuration is a given equilibrium of
 * referential density @f$\rho@f$. The unknowns are the displacement
 * @f$\mathbf u@f$ on the body and the referential potential
 * @f$\zeta@f$ on the ball, the potential at the place a point of the
 * reference has moved to. With
 * @f$\mathbf F = \mathbf 1 + \nabla\mathbf u@f$, continued through the
 * buffer by a prescribed extension of the displacement, and
 * @f$\mathbf a(\mathbf F) = J\,\mathbf F^{-1}\mathbf F^{-T}@f$, the
 * problem is the stationarity in both of
 * @f[
 *   V[\mathbf u, \zeta] = \int_B W(X, \mathbf E)\,dV
 *   + \int_B \rho\,\zeta\,dV
 *   + \frac{1}{8\pi G}\int \langle\mathbf a(\mathbf F)\nabla\zeta,
 *     \nabla\zeta\rangle\,dV + \int_{\partial B}\sigma\,\zeta\,dS ,
 * @f]
 * the third integral over the ball with the exterior's energy on its
 * sphere (doc/theory/gravitating_elasticity.md, §3.1). The load is a
 * surface mass @f$\sigma@f$ per unit reference area, carried by the
 * points of the surface: it enters through the potential alone, and that
 * is exact at finite strain. Stationarity in @f$\zeta@f$ is Poisson's
 * equation pulled back to the reference; it is linear in @f$\zeta@f$.
 *
 * The residual is incremental: the residual of the reference state
 * @f$(\mathbf 0, \zeta_0)@f$ without load is subtracted from both rows,
 * @f$\zeta_0@f$ being the discrete potential of the reference. In the
 * potential's row that removes the density's source identically; in the
 * displacement's it removes the imbalance of the given equilibrium
 * stress and gravity on the mesh, so that without load the reference is
 * the solution.
 *
 * The tangent is the linear referential class at the current state,
 * LinearQuasiStaticReferentialSelfGravitatingProblem with the current
 * configuration its mapping, the law's stress and tangent its rheology
 * and the current potential its background; its block solve gives the
 * corrections of both unknowns. A new object of that class is made at
 * each state at which the residual is wanted, and the gravity terms of
 * the residual are read from the potential's column of its operator, so
 * that the tangent is the derivative of the residual as assembled; what
 * only a solve needs is assembled only where the object is solved. Solve() keeps or refreshes
 * the tangent as NonlinearClampedProblem does (NewtonOptions).
 *
 * Built for a solid body (no fluid regions). Serial or parallel by the
 * spaces. See doc/methods/finite_strain.md.
 */
class NonlinearSelfGravitatingProblem {
 public:
  /**
   * @param fes_u The displacement space, on a (Par)SubMesh of the ball
   * that is the body.
   * @param fes_zeta The potential's space, on the ball.
   * @param fes_buffer A displacement space on the (Par)SubMesh that is
   * the buffer, of @p fes_u's collection: it holds the extension.
   * @param extension The prescribed vacuum extension, body to buffer
   * (NewRadialVacuumExtension): vdofs in serial. It must give the
   * identity mapping on the outer sphere.
   * @param law The material law of the body.
   * @param density @f$\rho@f$, on the body.
   * @param density @f$\rho@f$ is per unit volume of the mesh.
   * @param equilibrium The mapping of the mesh onto the equilibrium
   * configuration; nullptr is the identity. With a mapping the mesh is a
   * reference of labels: the law is taken about the image of the mesh
   * (MappedMaterialLaw), the displacement is from that image, and the
   * density and the loads are per unit volume and area of the mesh. The
   * mapping must be the identity on the outer sphere.
   *
   * None of the arguments is owned.
   */
  NonlinearSelfGravitatingProblem(mfem::FiniteElementSpace* fes_u,
                                  mfem::FiniteElementSpace* fes_zeta,
                                  mfem::FiniteElementSpace* fes_buffer,
                                  const mfem::SparseMatrix& extension,
                                  MaterialLaw& law, mfem::Coefficient& density,
                                  mfem::real_t gravitational_constant,
                                  int dtn_degree,
                                  Diffeomorphism* equilibrium = nullptr);
#ifdef MFEM_USE_MPI
  /** @brief Parallel overload: @p extension on true dofs. */
  NonlinearSelfGravitatingProblem(mfem::ParFiniteElementSpace* fes_u,
                                  mfem::ParFiniteElementSpace* fes_zeta,
                                  mfem::ParFiniteElementSpace* fes_buffer,
                                  const mfem::HypreParMatrix& extension,
                                  MaterialLaw& law, mfem::Coefficient& density,
                                  mfem::real_t gravitational_constant,
                                  int dtn_degree,
                                  Diffeomorphism* equilibrium = nullptr);
#endif
  ~NonlinearSelfGravitatingProblem();

  /** @brief The surface mass @f$\sigma@f$ per unit reference area on the
   * marked boundary attributes of the body's mesh; not owned. Once,
   * before the first evaluation. */
  void SetSurfaceLoad(mfem::Coefficient& sigma, const mfem::Array<int>& marker);

  /** @brief Dead loads on the displacement,
   * @f$\int_B \mathbf b\cdot\mathbf v\,dV +
   * \int_B \mathbf M : \nabla\mathbf v\,dV@f$: a force per unit volume
   * of the mesh and a load given by a stress field. Either may be null;
   * not owned. Once, before the first evaluation. */
  void SetBodyLoad(mfem::VectorCoefficient* force,
                   mfem::MatrixCoefficient* stress);

  /** @brief A load on the potential's row,
   * @f$\ell_\zeta(\chi) = \int q\,\chi\,dV +
   * \int \mathbf g\cdot\nabla\chi\,dV@f$ over the ball, on the right of
   * the potential's equation as the surface mass's
   * @f$-\int\sigma\chi\,dS@f$ is: a mass of density @f$m@f$ is
   * @f$q = -m@f$. Either may be null; not owned. Once, before the first
   * evaluation. */
  void SetPotentialLoad(mfem::Coefficient* source,
                        mfem::VectorCoefficient* flux);

  NewtonOptions& Options() { return options_; }
  const NewtonOptions& Options() const { return options_; }

  bool IsParallel() const;

  mfem::FiniteElementSpace& DisplacementSpace() { return *fes_u_; }
  mfem::FiniteElementSpace& PotentialSpace() { return *fes_zeta_; }

  /** @brief The state, at which Energy(), Residual() and Tangent() are
   * evaluated and from which Solve() starts, and the solution after it.
   * Before anything is written to them the state is the reference:
   * no displacement, and the potential ReferencePotential(). */
  mfem::GridFunction& Displacement();
  mfem::GridFunction& Potential();

  /** @brief @f$\zeta_0@f$, the discrete potential of the reference. */
  const mfem::GridFunction& ReferencePotential();

  /** @brief @f$V@f$ at the state, less its value's part that the
   * incremental residual drops: its first variation is Residual(). */
  mfem::real_t Energy();

  /** @brief The residual at the state, on the true dofs of the two
   * spaces. */
  void Residual(mfem::Vector& r_u, mfem::Vector& r_zeta);

  /** @brief The tangent at the state: a linear problem made there, owned
   * by this one until the next call or Solve(). */
  LinearQuasiStaticReferentialSelfGravitatingProblem& Tangent();

  /** @brief Solve from the state. Returns false if the iteration did not
   * meet its tolerance or a linear solve failed. */
  bool Solve();

  /** @brief Corrections taken by the last Solve(). */
  int Iterations() const { return static_cast<int>(corrections_.size()); }
  /** @brief The L2 norms of the displacement corrections taken by the
   * last Solve(), in order. */
  const std::vector<mfem::real_t>& Corrections() const { return corrections_; }
  /** @brief Tangents used by Solve() so far. */
  int TangentAssemblies() const { return tangents_; }

 private:
  void Initialise(Diffeomorphism* equilibrium);
  void EnsureReference();
  // The copies of the displacement on the buffer and on the ball.
  void SyncFields();
  std::unique_ptr<LinearQuasiStaticReferentialSelfGravitatingProblem>
  NewLinear(bool reference);
  // The residual at the state from a linear problem made there, and the
  // potential row of its operator applied to the potential.
  void StateResidual(LinearQuasiStaticReferentialSelfGravitatingProblem& at,
                     mfem::Vector& r_u, mfem::Vector& r_zeta,
                     mfem::Vector* a_zeta = nullptr);
  mfem::real_t Dot(const mfem::Vector& a, const mfem::Vector& b) const;
  mfem::real_t Norm(const mfem::GridFunction& v) const;

  mfem::FiniteElementSpace* fes_u_;
  mfem::FiniteElementSpace* fes_zeta_;
  mfem::FiniteElementSpace* fes_buffer_;
  const mfem::SparseMatrix* extension_ = nullptr;
#ifdef MFEM_USE_MPI
  const mfem::HypreParMatrix* pextension_ = nullptr;
#endif
  MaterialLaw* law_;
  mfem::Coefficient* rho_;
  mfem::real_t G_;
  int dtn_degree_;
  mfem::Coefficient* sigma_ = nullptr;
  mfem::Array<int> surface_marker_;
  mfem::VectorCoefficient* body_force_ = nullptr;
  mfem::MatrixCoefficient* stress_load_ = nullptr;
  mfem::Coefficient* potential_source_ = nullptr;
  mfem::VectorCoefficient* potential_flux_ = nullptr;
  NewtonOptions options_;

  std::unique_ptr<IdentityDiffeomorphism> identity_;
  std::unique_ptr<MappedMaterialLaw> mapped_law_;
  std::unique_ptr<mfem::FiniteElementSpace> fes_ball_;  // displacement, ball
  std::unique_ptr<mfem::GridFunction> u_, u_buffer_, u_ball_, zeta_, zeta0_;
  std::unique_ptr<DisplacedDiffeomorphism> map_;
  std::unique_ptr<StrainEnergyDensityCoefficient> energy_c_;
  std::unique_ptr<FirstPiolaKirchhoffStressCoefficient> stress_c_;
  std::unique_ptr<SecondPiolaKirchhoffStressCoefficient> tangent_stress_c_;
  std::unique_ptr<MaterialTangentCoefficient> tangent_c_;
  std::unique_ptr<ReferentialElasticRheology> rheology_;
  std::unique_ptr<mfem::GridFunctionCoefficient> zeta_c_;
  std::unique_ptr<mfem::VectorConstantCoefficient> zero_;
  std::unique_ptr<mfem::LinearForm> internal_;
  // The one rule of the energy, the residual and the tangent.
  const mfem::IntegrationRule* rule_ = nullptr;

  // The reference: its residual's gravity parts and the loads of the two
  // rows, true dofs.
  bool reference_ready_ = false;
  mfem::Vector reference_u_, reference_zeta_, load_zeta_, load_u_;

  std::unique_ptr<LinearQuasiStaticReferentialSelfGravitatingProblem> tangent_;
  int tangents_ = 0;
  std::vector<mfem::real_t> corrections_;
};

}  // namespace AdGIA
