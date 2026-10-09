/**
 * @file background.hpp
 * @brief The background-state module: generation of the equilibrium
 * fields @f$(\rho, \hat C, \mathbf{S}_e, \varphi_e)@f$ that the
 * referential problem consumes (doc/gravitating_elasticity.md §1,
 * "Notation and kinematics", §2, "The elastic tensor dictionary", and §6,
 * "Equilibrium stress in general models").
 * The moduli conversion and the relabelling transformation laws live
 * here; drivers never hand-roll them.
 */

#pragma once

#include <functional>
#include <vector>

#include "mfem.hpp"
#include "AdGIA/elastic_tensor.hpp"
#include "AdGIA/mappings.hpp"
#include "AdGIA/referential_problem.hpp"

namespace AdGIA {

/**
 * @brief The hydrostatic balance of a radial density profile: gravity
 * @f$g(r)@f$ and pressure @f$p^0(r)@f$ by cumulative quadrature, in the
 * codebase's Poisson convention @f$\nabla^2\Phi_0 = 4\pi G\rho@f$ for
 * either dimension,
 * @f[
 *   g(r) = \frac{4\pi G}{r^{d-1}} \int_0^r \rho(s)\, s^{d-1}\, ds,
 *   \qquad \frac{dp^0}{dr} = -\rho g, \qquad p^0(R) = 0,
 * @f]
 * (2-D: the gravitating-disc convention of the test models). The profile
 * is sampled on a uniform grid and interpolated linearly; piecewise-smooth
 * densities (PREM) are fine. Outside the radius @f$p^0 = 0@f$ and
 * @f$g@f$ is the exterior field of the total mass.
 */
class RadialHydrostaticState {
 public:
  /**
   * @param dim Space dimension (2 or 3).
   * @param rho Radial density @f$\rho(r)@f$.
   * @param G Gravitational constant in the units of the problem.
   * @param radius The (referential) surface radius @f$R@f$.
   * @param samples Grid intervals for the cumulative quadrature.
   */
  RadialHydrostaticState(int dim,
                         std::function<mfem::real_t(mfem::real_t)> rho,
                         mfem::real_t G, mfem::real_t radius,
                         int samples = 4000);

  mfem::real_t Gravity(mfem::real_t r) const;
  mfem::real_t Pressure(mfem::real_t r) const;
  /** @brief The profile itself (0 outside the radius). */
  mfem::real_t Density(mfem::real_t r) const;

  mfem::real_t Radius() const { return R_; }
  int SpaceDim() const { return dim_; }

 private:
  int dim_;
  std::function<mfem::real_t(mfem::real_t)> rho_;
  mfem::real_t G_, R_, h_;
  std::vector<mfem::real_t> g_, p_;
};

/**
 * @brief The hydrostatic equilibrium of a radial model as a referential
 * background at @f$\varphi_e = \mathrm{id}@f$: owns the full coefficient
 * chain the referential problem consumes — the referential density
 * @f$\rho(|\mathbf{x}|)@f$, the *bare* second elastic tensor
 * @f$\hat C@f$ (the radial moduli are SEISMOLOGICAL @f$(\kappa,\mu)@f$;
 * BareElasticTensorCoefficient applies the conversion with the
 * hydrostatic @f$p^0@f$ computed here), the equilibrium stress
 * @f$\mathbf{S}_e = -p^0\mathbf{1}@f$, and the identity mapping — and
 * the assembled ReferentialElasticRheology.
 *
 * Non-copyable: the members reference one another.
 */
class RadialHydrostaticBackground {
 public:
  using RadialFunc = std::function<mfem::real_t(mfem::real_t)>;

  RadialHydrostaticBackground(int dim, RadialFunc rho, RadialFunc kappa,
                              RadialFunc mu, mfem::real_t G,
                              mfem::real_t radius, int samples = 4000);

  RadialHydrostaticBackground(const RadialHydrostaticBackground&) = delete;
  RadialHydrostaticBackground& operator=(const RadialHydrostaticBackground&) =
      delete;

  ReferentialElasticRheology& Rheology() { return rheology_; }
  mfem::Coefficient& Density() { return rho_coef_; }
  /** @brief The bare tensor @f$\hat C@f$ (Mandel). */
  mfem::MatrixCoefficient& ElasticTensor() { return C_; }
  mfem::MatrixCoefficient& EquilibriumStress() { return S_; }
  Diffeomorphism& EquilibriumMapping() { return phi_e_; }
  mfem::Coefficient& Pressure() { return p0_coef_; }

  const RadialHydrostaticState& State() const { return state_; }
  int SpaceDim() const { return dim_; }

  mfem::real_t DensityAt(mfem::real_t r) const { return state_.Density(r); }
  mfem::real_t KappaAt(mfem::real_t r) const { return kappa_fn_(r); }
  mfem::real_t MuAt(mfem::real_t r) const { return mu_fn_(r); }
  mfem::real_t PressureAt(mfem::real_t r) const { return state_.Pressure(r); }
  mfem::real_t GravityAt(mfem::real_t r) const { return state_.Gravity(r); }

 private:
  int dim_;
  RadialFunc kappa_fn_, mu_fn_;
  RadialHydrostaticState state_;
  mfem::FunctionCoefficient rho_coef_, kappa_coef_, mu_coef_, p0_coef_;
  IsotropicElasticTensorCoefficient C_eff_;
  BareElasticTensorCoefficient C_;
  mfem::ProductCoefficient minus_p0_;
  mfem::IdentityMatrixCoefficient identity_;
  mfem::ScalarMatrixProductCoefficient S_;
  IdentityDiffeomorphism phi_e_;
  ReferentialElasticRheology rheology_;
};

/**
 * @brief The same physical equilibrium described from a relabelled
 * reference body: given a radial hydrostatic base and a relabelling
 * @f$\boldsymbol{\xi}@f$ (the identity at and outside the surface for the
 * present problem class), owns the transformed chain
 * @f[
 *   \tilde\rho = J_\xi\,\rho\circ\xi, \qquad
 *   \tilde C = J_\xi\, Q_\xi^{-T} (\hat C\circ\xi)\, Q_\xi^{-1}, \qquad
 *   \tilde S = J_\xi\, F_\xi^{-1} (\mathbf{S}_e\circ\xi)\, F_\xi^{-T},
 *   \qquad \varphi_e = \xi,
 * @f]
 * (doc/gravitating_elasticity.md §1, doc/mappings.md). The solution must
 * reproduce the base solution under composition, which makes the pair a
 * self-benchmark of the mapped assembly (RelabelledEquilibrium2D in
 * TestReferentialProblem). Base and @f$\boldsymbol{\xi}@f$ are not owned
 * and must outlive this object. Non-copyable.
 */
class RelabelledBackground {
 public:
  RelabelledBackground(RadialHydrostaticBackground& base, Diffeomorphism& xi);

  RelabelledBackground(const RelabelledBackground&) = delete;
  RelabelledBackground& operator=(const RelabelledBackground&) = delete;

  ReferentialElasticRheology& Rheology() { return rheology_; }
  /** @brief @f$\tilde\rho = J_\xi\,\rho\circ\xi@f$. */
  mfem::Coefficient& Density() { return rho_rel_; }
  mfem::MatrixCoefficient& ElasticTensor() { return C_rel_; }
  mfem::MatrixCoefficient& EquilibriumStress() { return S_rel_; }
  /** @brief The referential pressure @f$\tilde\pi = p^0\circ\xi@f$ (the
   * interface-pressure input of the slip problems). */
  mfem::Coefficient& Pressure() { return p0_xi_; }
  Diffeomorphism& EquilibriumMapping() { return *xi_; }

 private:
  RadialHydrostaticBackground* base_;
  Diffeomorphism* xi_;
  TransformedFunctionCoefficient rho_xi_, kappa_xi_, mu_xi_, p0_xi_;
  IsotropicElasticTensorCoefficient C_eff_xi_;
  BareElasticTensorCoefficient C_comp_;
  RelabelledElasticTensorCoefficient C_rel_;
  TransformedMatrixFunctionCoefficient S_comp_;
  PullbackStressCoefficient S_rel_;
  JacobianCoefficient jac_;
  mfem::ProductCoefficient rho_rel_;
  ReferentialElasticRheology rheology_;
};

/**
 * @brief The minimum equilibrium stress field of Al-Attar & Woodhouse
 * (2010, GJI 181, 567; §3.3): among all symmetric stress fields with
 * @f$\mathrm{Div}\,\mathbf{T} = \mathbf{f}@f$ in the body and
 * @f$\mathbf{T}\hat{\mathbf{n}} = 0@f$ on its surface, the one of
 * smallest norm @f$\int \tfrac{1}{2\mu}\,\mathbf{T}:\mathbf{T}\,dV@f$.
 * By their eqs. (55)–(57) it has the form @f$\mathbf{T} =
 * 2\mu\nabla_s\mathbf{u}@f$ with @f$\mathbf{u}@f$ solving the static
 * elastic problem @f$\mathrm{Div}(2\mu\nabla_s\mathbf{u}) =
 * \mathbf{f}@f$, traction-free (an elastic material with shear modulus
 * @f$\mu@f$ and @f$\lambda = 0@f$); the solve happens in the
 * constructor, projected onto the complement of the rigid modes (the
 * body force must be self-equilibrated: zero net force and torque —
 * their eq. 58). The stress is independent of the absolute scale of
 * @f$\mu@f$; only spatial variations matter (a priori weighting, e.g.
 * down-weighting lithospheric stress), so @p mu defaults to a constant.
 *
 * The object *is* the stress: a MatrixCoefficient usable directly as
 * the @f$\mathbf{S}_e@f$ of ReferentialElasticRheology, evaluable on
 * the mesh of @p fes. For @f$\mathbf{f} = \rho\nabla\Phi_0@f$ this is
 * an equilibrium stress consistent with self-gravity in the sign
 * convention of the codebase (@f$\mathrm{Div}(-p\mathbf{1}) =
 * \rho\nabla\Phi_0@f$ hydrostatically). Serial and parallel.
 *
 * **Relabelled (mapped) mode**: with a Diffeomorphism @p map, the whole
 * problem is pulled back to the fixed reference body: the elastic form
 * becomes @f$\int 2\mu\,\mathrm{sym}(\nabla u F^{-1}) :
 * \mathrm{sym}(\nabla v F^{-1})\, J\,dx@f$ (the standard relabelling
 * recipe of the mapped integrators), the rigid kernel becomes
 * translations plus MappedRotation, @p body_force must then be the
 * *composed* physical force @f$\mathbf{f}\circ\varphi@f$ (a coefficient
 * on the reference mesh; the class supplies the Jacobian weight), and
 * Eval() returns the **second Piola–Kirchhoff pullback**
 * @f$\mathbf{S} = J F^{-1} (\mathbf{T}\circ\varphi) F^{-T}@f$ — exactly
 * the @f$\mathbf{S}_e@f$ the referential problem consumes. With
 * @f$F@f$ explicit in every form, derivatives with respect to the
 * mapping can be formed analytically.
 */
class MinimumNormEquilibriumStress : public mfem::MatrixCoefficient {
 public:
  /**
   * @param fes Vector H1 space on the body (SubMesh); not owned.
   * @param body_force @f$\mathbf{f} = \rho\nabla\Phi_0@f$ (in mapped
   * mode: the composed @f$\mathbf{f}\circ\varphi@f$); not owned, used
   * only during construction.
   * @param mu Optional positive weight field; constant when null.
   * @param map Optional relabelling: solve pulled back on the reference
   * body (see the class notes); not owned, must outlive the object.
   */
  MinimumNormEquilibriumStress(mfem::FiniteElementSpace& fes,
                               mfem::VectorCoefficient& body_force,
                               mfem::Coefficient* mu = nullptr,
                               Diffeomorphism* map = nullptr);

  void Eval(mfem::DenseMatrix& K, mfem::ElementTransformation& T,
            const mfem::IntegrationPoint& ip) override;

  /** @brief The Lagrange-multiplier field @f$\mathbf{u}@f$ (a formal
   * displacement; the stress is its symmetric gradient). */
  const mfem::GridFunction& Auxiliary() const { return *u_; }
  int SolverIterations() const { return iterations_; }

 private:
  mfem::FiniteElementSpace* fes_;
  mfem::Coefficient* mu_;
  Diffeomorphism* map_;
  mfem::ConstantCoefficient half_;
  std::unique_ptr<mfem::GridFunction> u_;
  mfem::DenseMatrix G_, F_, Fi_, A_, S_, tmp_;
  int iterations_ = 0;
};

/**
 * @brief The assembled minimum-deviatoric Stokes saddle, kept alive so
 * that repeated solves pay assembly and preconditioner setup once. No
 * operator here depends on the density — only the load does — so an
 * optimisation loop (doc/equilibrium_figures.tex) holds one of these
 * for its every value, gradient and Hessian-action solve.
 *
 * The system, null-space handling, optional viscosity field, mapped
 * mode and essential-velocity option are exactly those of
 * MinimumDeviatoricEquilibriumStress, which delegates to this class
 * (and can borrow a shared instance). Serial and parallel.
 */
/** @brief The mode-th rigid-motion field of @p dim dimensions
 * (translations first, then rotations about the origin), as a function
 * for a VectorFunctionCoefficient — shared by the saddle's lifted
 * border fields and the enclosed components' load integrals, so the
 * two cannot drift apart. */
std::function<void(const mfem::Vector&, mfem::Vector&)> RigidMode(int dim,
                                                                  int mode);

class StokesSaddleSolver {
 public:
  /** Parameters as in MinimumDeviatoricEquilibriumStress; the spaces,
   * coefficients, map and marker are borrowed and must outlive the
   * solver.
   * @param rigid_bdr Optional rigid boundary groups
   * (doc/equilibrium_figures.tex §2, the disconnected-solid
   * correction): one boundary-attribute marker per solid component
   * enclosed by the fluid (e.g. the ICB). The velocity space is then
   * clamped there EXCEPT for that boundary's rigid motions — six
   * extra unknowns per group in 3-D (three in 2-D, rotations about
   * the origin) — which add the component's force and torque balance
   * to the constraint set the saddle enforces. Implemented by
   * bordering: the lifted rigid fields' base solves are precomputed at
   * construction, so a bordered Solve costs one base solve plus dense
   * algebra. */
  StokesSaddleSolver(
      mfem::FiniteElementSpace& fes_u, mfem::FiniteElementSpace& fes_p,
      mfem::Coefficient* mu = nullptr, Diffeomorphism* map = nullptr,
      const mfem::Array<int>* essential_bdr = nullptr,
      const std::vector<mfem::Array<int>>* rigid_bdr = nullptr);
  ~StokesSaddleSolver();

  /** @brief The number of rigid border unknowns (groups times modes). */
  int RigidModes() const;

  /** @brief The k-th lifted rigid field (a GridFunction on the
   * velocity space): its trace is the rigid motion on its group's
   * boundary and zero on the others. */
  const mfem::GridFunction& RigidField(int k) const;

  /** @brief The energy @f$\tfrac12 u^T A u@f$ of a velocity field in
   * the UNELIMINATED form — with rigid groups, the value function of
   * the bordered saddle, convention-free (only available when groups
   * are present, which is when the uneliminated operator exists). */
  mfem::real_t Energy(const mfem::GridFunction& u) const;

  /**
   * @brief Solves the saddle for the velocity-space load dual
   * @f$F = (\mathbf{f}, \mathbf{v})@f$ (true dofs; the class applies
   * the system's sign and the essential zeroing). Returns the solver's
   * iteration count. With rigid groups, @p rigid_rhs (length
   * RigidModes(), or empty for zero) is the ADDITIONAL border
   * right-hand side — the enclosed component's own load paired with
   * the rigid modes — beyond the fluid load's pairing, which the
   * class applies itself; on return @p u contains the TOTAL velocity
   * (the clamped part plus the rigid combination), and
   * @p rigid_coefficients (when given) the modes' coefficients.
   */
  int Solve(const mfem::Vector& F, mfem::GridFunction& u,
            mfem::GridFunction& p,
            const mfem::Vector& rigid_rhs = mfem::Vector(),
            mfem::Vector* rigid_coefficients = nullptr) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/**
 * @brief The minimum *deviatoric* equilibrium stress field of Al-Attar
 * & Woodhouse (2010, §3.4): the equilibrium stress whose deviatoric
 * part has the smallest norm. By their eqs. (70)–(74) it is
 * @f$\mathbf{T} = -p\mathbf{1} + 2\mu\nabla_s\mathbf{u}@f$ with
 * @f$(\mathbf{u}, p)@f$ the steady incompressible Stokes problem
 * @f[
 *   -\nabla p + \mathrm{Div}(2\mu\nabla_s\mathbf{u}) = \mathbf{f},
 *   \qquad \mathrm{div}\,\mathbf{u} = 0,
 *   \qquad [-p\hat{\mathbf{n}} +
 *     2\mu\hat{\mathbf{n}}\cdot\nabla_s\mathbf{u}] = 0
 *   \ \text{on}\ \partial B,
 * @f]
 * solved here with **Taylor–Hood** elements: the pressure space must be
 * (at least) one polynomial order below the velocity space, since
 * equal-order interpolation violates the inf–sup (LBB) condition and
 * produces spurious pressure modes — the constructor refuses it. With
 * the traction (all-Neumann) boundary condition the pressure carries
 * *no* constant ambiguity; the only kernel is the rigid modes of
 * @f$\mathbf{u}@f$, projected. In regions admitting a hydrostatic
 * state the field reduces to @f$-p^0\mathbf{1}@f$; on an aspherical
 * body it quantifies the deviatoric stress that no equilibrium field
 * can avoid. (Their further claim that this field also minimises
 * stress-induced anisotropy, eq. 79, rested on Dahlen's pre-stress
 * decomposition of the elastic tensor, which Maitra & Al-Attar 2021
 * showed to be incomplete — it is not relied on here.) Serial and
 * parallel.
 *
 * **Relabelled (mapped) mode**, as for MinimumNormEquilibriumStress:
 * with @p map the Stokes problem is pulled back to the fixed reference
 * body (mapped elastic block, mapped divergence coupling
 * @f$\int \mathrm{tr}(\nabla u F^{-1})\, q\, J\,dx@f$, kernel =
 * translations + MappedRotation; the natural traction condition pulls
 * back exactly through Nanson), @p body_force is the composed
 * @f$\mathbf{f}\circ\varphi@f$, and Eval() returns the second
 * Piola–Kirchhoff pullback @f$J F^{-1}(-p\mathbf{1} +
 * 2\mu\,\mathrm{sym}(\nabla u F^{-1})) F^{-T}@f$. Taylor–Hood
 * stability survives the (bi-Lipschitz) relabelling, with the inf–sup
 * constant degrading with the map's condition number.
 */
class MinimumDeviatoricEquilibriumStress : public mfem::MatrixCoefficient {
 public:
  /**
   * @param fes_u Vector H1 velocity space on the body; not owned.
   * @param fes_p Scalar H1 pressure space on the same mesh, one order
   * below @p fes_u (Taylor–Hood); not owned.
   * @param body_force @f$\mathbf{f} = \rho\nabla\Phi_0@f$
   * (self-equilibrated; in mapped mode: the composed
   * @f$\mathbf{f}\circ\varphi@f$); not owned, used during construction.
   * @param mu Optional positive weight field; constant when null.
   * @param map Optional relabelling (see the class notes); not owned,
   * must outlive the object.
   * @param essential_bdr Optional boundary-attribute marker on which the
   * velocity is clamped, @f$\mathbf{u} = \mathbf{0}@f$, in place of the
   * natural traction condition — the multiplier condition of the
   * fluid-only feasibility functional (doc/equilibrium_figures.tex),
   * whose constraint set leaves the interface traction free. The
   * constraint @f$\mathrm{Div}\,\mathbf{T} + \mathbf{f} = \mathbf{0}@f$
   * then holds on the mesh with no condition on @f$\mathbf{T}
   * \hat{\mathbf{n}}@f$ over the marked part, so @p body_force need not
   * be self-equilibrated there. Clamping removes the rigid kernel; when
   * the marker covers *every* boundary attribute the pressure acquires
   * the classical constant ambiguity instead, projected here.
   * @param solver Optional shared StokesSaddleSolver (which must have
   * been built with the same spaces and options); when null the object
   * assembles its own. A loop evaluating many densities shares one.
   * @param rigid_rhs With a solver carrying rigid boundary groups: the
   * border right-hand side (the enclosed components' loads), passed
   * through to StokesSaddleSolver::Solve; @p rigid_coefficients
   * returns the modes' coefficients when given.
   * @param interface_pressure Optional pressure load on the boundary
   * marked by @p interface_bdr (all of it when null): the natural
   * condition becomes @f$\mathbf{T}\hat{\mathbf{n}} =
   * -p\,\hat{\mathbf{n}}@f$ with @f$\hat{\mathbf{n}}@f$ the mesh's
   * outward normal — a solid component loaded by the pressure of the
   * fluid outside that boundary. A constant added to @f$p@f$ on a
   * CLOSED marked boundary has no net force or torque and shifts the
   * stress isotropically, so the deviatoric field does not see the
   * enclosed fluid's pressure gauge. Not available in mapped mode.
   */
  MinimumDeviatoricEquilibriumStress(
      mfem::FiniteElementSpace& fes_u, mfem::FiniteElementSpace& fes_p,
      mfem::VectorCoefficient& body_force, mfem::Coefficient* mu = nullptr,
      Diffeomorphism* map = nullptr,
      const mfem::Array<int>* essential_bdr = nullptr,
      const StokesSaddleSolver* solver = nullptr,
      const mfem::Vector& rigid_rhs = mfem::Vector(),
      mfem::Vector* rigid_coefficients = nullptr,
      mfem::Coefficient* interface_pressure = nullptr,
      const mfem::Array<int>* interface_bdr = nullptr);

  void Eval(mfem::DenseMatrix& K, mfem::ElementTransformation& T,
            const mfem::IntegrationPoint& ip) override;

  const mfem::GridFunction& Pressure() const { return *p_; }
  const mfem::GridFunction& Auxiliary() const { return *u_; }
  int SolverIterations() const { return iterations_; }

 private:
  mfem::FiniteElementSpace *fes_u_, *fes_p_;
  mfem::Coefficient* mu_;
  Diffeomorphism* map_;
  mfem::ConstantCoefficient half_;
  std::unique_ptr<StokesSaddleSolver> own_solver_;
  std::unique_ptr<mfem::GridFunction> u_, p_;
  mfem::DenseMatrix G_, F_, Fi_, A_, S_, tmp_;
  int iterations_ = 0;
};

/**
 * @brief The elliptic buffer-taper rule: extend an equilibrium mapping,
 * given (at least) on the body, to the whole ball by a one-off harmonic
 * solve in the buffer (doc/gravitating_elasticity.md §3, "Gravity: three
 * formulations"; the
 * alternative to the analytic TaperedDiffeomorphism when the mapping is
 * only known on the body, e.g.\ discrete or planetmodel-supplied).
 *
 * The displacement @f$\mathbf{h} = \boldsymbol{\xi} - \mathrm{id}@f$ is
 * interpolated on the body (nodal, at @p order — the interpolated mode
 * of doc/mappings.md, "The discrete change-of-variables identity"),
 * extended into the buffer by the vector
 * Laplace equation with Dirichlet data @f$\mathbf{h}@f$ on the shared
 * body surface and @f$\mathbf{0}@f$ on the outer (DtN) sphere, and
 * returned as an owning GridFunctionDiffeomorphism on the parent mesh.
 * The DtN-sphere identity convention holds pointwise (values); by the
 * maximum principle the extension is bounded by its surface trace.
 * Diffeomorphy for moderate displacements is the caller's business, as
 * for every mapping.
 */
GridFunctionDiffeomorphism NewHarmonicExtensionMapping(
    mfem::Mesh& parent, int order, Diffeomorphism& xi,
    const mfem::Array<int>& body_attributes,
    const mfem::Array<int>& buffer_attributes);

#ifdef MFEM_USE_MPI
/** @brief Parallel overload: ParSubMesh transfers and an AMG-CG buffer
 * solve; semantics as the serial builder. */
GridFunctionDiffeomorphism NewHarmonicExtensionMapping(
    mfem::ParMesh& parent, int order, Diffeomorphism& xi,
    const mfem::Array<int>& body_attributes,
    const mfem::Array<int>& buffer_attributes);
#endif

}  // namespace AdGIA
