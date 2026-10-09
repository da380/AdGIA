/**
 * @file mixed_problem.hpp
 * @brief Quasi-static elasticity of a self-gravitating body in the *mixed*
 * referential–spatial formulation: the (referential) displacement on a
 * SubMesh of the body coupled to the *spatial* (Eulerian) gravitational
 * potential perturbation on the enclosing ball, with a
 * Dirichlet-to-Neumann outer condition.
 *
 * On the naming axes (see quasi_static_problem.hpp): the formulation slot
 * distinguishes how the *gravity* is described — the elasticity is
 * referential in every problem class. Mixed is the Dahlen-style
 * organisation, and it *requires* a hydrostatic, natural reference state
 * (Al-Attar & Crawford 2016: the particle label is the equilibrium
 * position). The fully referential classes
 * (referential_problem.hpp) lift both restrictions.
 */

#pragma once

#include <list>
#include <memory>
#include <vector>

#include "mfem.hpp"
#include "AdGIA/coefficient.hpp"
#include "AdGIA/poisson.hpp"
#include "AdGIA/quasi_static_problem.hpp"
#include "AdGIA/null_space.hpp"
#include "AdGIA/submesh.hpp"

namespace AdGIA {

/**
 * @brief A fluid region of a self-gravitating body (see
 * LinearQuasiStaticMixedSelfGravitatingProblem): a set of parent-mesh attributes
 * carrying no displacement unknown, its density, and its interfaces with the
 * solid.
 *
 * - @p attributes: the parent-mesh attributes of the fluid (not part of the
 *   displacement SubMesh).
 * - @p density: @f$\rho_F@f$, evaluated on the parent's fluid elements (for
 *   the background potential and, through the fallback below, for
 *   @f$\rho'_F@f$) and, unless @p interface_density is given, on the
 *   *SubMesh's boundary elements* of the interfaces: a coefficient of
 *   position (FunctionCoefficient) serves both; a coefficient keyed by
 *   domain attribute or a GridFunctionCoefficient on the parent does not.
 * - @p density_gradient: @f$\rho'_F = d\rho/d\Phi_0 = g^{-1}\partial_r\rho@f$
 *   on the fluid elements (BarotropicDensityGradientCoefficient, or an
 *   analytic form for radial models). When null it is computed from the
 *   element-wise L2 projection of @p density and the discrete
 *   @f$\nabla\Phi_0@f$.
 * - @p interface_marker: the SubMesh boundary attributes of the fluid's
 *   interfaces with the solid (sized to the SubMesh's bdr_attributes.Max()).
 *   Nothing distinguishes a fluid below the solid from a fluid above: the
 *   sign of @f$\mathbf{m}\cdot\nabla\Phi_0@f$ does that.
 * - @p interface_density: @f$\rho_F@f$ on those boundary elements; default
 *   @p density.
 * - @p interface_potential_coupling: when false, the interface coupling
 *   (F3) is dropped — both halves, so the operator stays symmetric — and
 *   the fluid presses on the solid through the buoyancy term (F2) alone
 *   (a "Winkler foundation"). With @p density_gradient zero and a constant
 *   @p interface_density this reproduces, in steps, the approximate CMB
 *   conditions of the GIA literature (see doc/self_gravitation.md, "The
 *   CMB approximations of the GIA literature", doc/quasi_static_models.tex,
 *   "Fluid regions and the ladder of CMB approximations", and the
 *   benchmark option `-cmb`).
 */
struct FluidRegion {
  mfem::Array<int> attributes;
  mfem::Coefficient* density = nullptr;
  mfem::Coefficient* density_gradient = nullptr;
  mfem::Array<int> interface_marker;
  mfem::Coefficient* interface_density = nullptr;
  bool interface_potential_coupling = true;
};

/**
 * @brief Self-gravitating linear quasi-static problem in the mixed
 * referential–spatial formulation (eq. 3 of Yu, Al-Attar, Syvret & Lloyd
 * 2025, with the fluid regions of their Appendix A / Al-Attar & Tromp 2014
 * eq. 2.52), implementing LinearQuasiStaticProblem so that the
 * viscoelastic layer runs on it unchanged.
 *
 * **Formulation and reference state.** The displacement is referential;
 * the potential perturbation @f$\phi@f$ is the *Eulerian* (spatial) one.
 * The derivation assumes the reference state is **hydrostatic** (the
 * equilibrium stress is @f$-p_0 I@f$, entering through @f$\Phi_0@f$ and
 * the fluid terms) and **natural** in the sense of Al-Attar & Crawford
 * 2016 — the particle label is the equilibrium position
 * (@f$\varphi_e = \mathrm{id}@f$). These are requirements of this class,
 * not defaults: for a non-hydrostatic or non-natural reference state use
 * LinearQuasiStaticReferentialSelfGravitatingProblem, which takes the
 * general @f$(\hat C, \mathbf{S}_e, \varphi_e)@f$.
 *
 * **Geometry.** The body @f$M = M_S \cup M_F@f$ is a region of a ball
 * @f$B@f$ whose external boundary is a sphere (a circle in 2-D). The
 * displacement @f$u@f$ lives on a (Par)SubMesh of the *solid* regions
 * @f$M_S@f$ (possibly disconnected, e.g. inner core and mantle), the
 * potential perturbation @f$\phi@f$ on @f$B@f$, and the exterior of @f$B@f$
 * is represented by a PoissonDtNOperator. Fluid regions @f$M_F@f$ (a liquid
 * core) carry no displacement; they enter through their density, the
 * hydrostatic Poisson term, and the interface terms below.
 *
 * **Weak form** (with @f$G@f$ the gravitational constant in the chosen
 * units, @f$\rho@f$ the density and @f$\Phi_0@f$ the background potential,
 * @f$\mathbf{m}@f$ the outward normal of the solid on the fluid–solid
 * interfaces @f$\Sigma_F@f$, @f$\rho_F@f$ the fluid-side density there,
 * and @f$\rho'_F = d\rho/d\Phi_0@f$ in the fluid):
 * @f[
 *   a(u,v) + \tfrac12\int_{M_S} \rho\,[v\cdot\nabla(u\cdot\nabla\Phi_0)
 *     + u\cdot\nabla(v\cdot\nabla\Phi_0) -
 * (v\cdot\nabla\Phi_0)\,\mathrm{div}\,u
 *     - (u\cdot\nabla\Phi_0)\,\mathrm{div}\,v]
 *     - \int_{\Sigma_F} \rho_F\,(\mathbf{m}\cdot\nabla\Phi_0)
 *       (\mathbf{m}\cdot u)(\mathbf{m}\cdot v)
 *     + c(\phi, v) = \ell_u(v) - c(\psi, v),
 * @f]
 * @f[
 *   \frac{1}{4\pi G}\Big[\int_B \nabla\phi\cdot\nabla\chi
 *     + \mathrm{DtN}(\phi,\chi)\Big] + \int_{M_F} \rho'_F\,\phi\,\chi
 *     + c(\chi, u) = \ell_\phi(\chi) - \int_{M_F} \rho'_F\,\psi\,\chi,
 * @f]
 * with the coupling form
 * @f$c(\phi, v) = \int_{M_S} \rho\,\nabla\phi\cdot v
 *   - \int_{\Sigma_F} \rho_F\,\phi\,(\mathbf{m}\cdot v)@f$,
 * @f$a@f$ the bulk/deviatoric elastic form of LinearQuasiStaticProblemBase, and
 * @f$\psi@f$ an optional applied (tidal) potential. Without fluid regions
 * the interface and @f$\rho'_F@f$ terms are absent and this is eq. 3 of the
 * paper. A surface mass load @f$\sigma@f$ (mass per unit area, positive
 * when mass is added) on marked boundary attributes of the SubMesh
 * contributes @f$\ell_u(v) = -\int \sigma\,\nabla\Phi_0\cdot v@f$ and
 * @f$\ell_\phi(\chi) = -\int \sigma\,\chi@f$. Further displacement loads
 * can be added to ExternalLoad() and further potential loads to
 * ExternalPotentialLoad(); AddForce() acts on the displacement as usual.
 *
 * **Background potential.** Solved once from the density of the solid
 * (injected from the SubMesh) and of the fluids (on the parent) with the
 * plain Poisson–DtN operator, unless a coefficient is supplied, e.g. from a
 * radial model. The fluid mass term @f$\int_{M_F}\rho'_F\phi\chi@f$ is
 * assembled afterwards (it may depend on @f$\nabla\Phi_0@f$) and added to
 * the potential block.
 *
 * **Solvers** (SetSolverType()):
 * - BlockMINRES (default): MINRES on the @f$[u;\phi]@f$ block system with a
 *   block-diagonal preconditioner (Gauss-Seidel or BoomerAMG on the
 *   displacement block and on the shifted Laplacian), restricted by a
 *   ProjectedSolver to displacements orthogonal to the rigid modes (and,
 *   in 2-D, potentials orthogonal to the constant). Typically an order of
 *   magnitude cheaper than SchurCG.
 * - SchurCG: the potential is eliminated and CG runs on the displacement
 *   Schur complement @f$S = A_{uu} - C A_{\phi\phi}^{-1} C^T@f$, which is
 *   symmetric and, for a gravitationally stable body, positive on the
 *   complement of the rigid modes; each application costs one inner
 *   Poisson solve. The reference against which BlockMINRES is checked.
 * Both warm start across solves. The inner potential solves use CG; with
 * fluid regions the potential block is @f$(K + \mathrm{DtN})/4\pi G +
 * M_F@f$ with @f$M_F \le 0@f$ where density increases downward, positive
 * for Earth-like models but not by a wide margin —
 * PotentialBlockMinEigenvalue() reports it.
 *
 * **Null space and gauge.** The global rigid modes, paired with their
 * potential @f$-u_r\cdot\nabla\Phi_0@f$ extended through the fluid, are
 * near-null vectors of the coupled system (see RigidModeResiduals()), so
 * the system is regularised by restricting the displacement to the
 * orthogonal complement of the rigid modes in the Euclidean true-dof inner
 * product. Both solvers solve exactly that restricted system (the block
 * form of BlockMINRES has the Schur complement of SchurCG), so they agree
 * to solver tolerance and share the gauge: no rigid-body component in the
 * displacement, the potential the discrete solution of its equation for
 * that displacement. A solid region enclosed by fluid (an inner core) has,
 * for a spherically symmetric model, its own near-null rotations;
 * AddRegionRotations() removes them as well. Its translations are restored
 * gravitationally (the Slichter mode) and must not be projected; being
 * soft, they are where discretisation error shows first.
 *
 * In two dimensions the constant potential is a null vector of the
 * Laplace–DtN block; potential loads are made compatible by subtracting a
 * uniform flux through the outer boundary, and the potential is restricted
 * to the complement of the constant (every potential solve runs on
 * @f$P A_{\phi\phi} P@f$, the block solver restricts the potential block
 * likewise). With fluid regions this is a regularisation: the constant is
 * not null for @f$M_F \ne 0@f$, and the 2-D problem is not gauge invariant
 * (a constant potential shift loads the interfaces through the fluid
 * pressure), so 2-D fluid results are for testing on small meshes only.
 *
 * **Serial and parallel** in one class: pass ParFiniteElementSpaces on a
 * ParSubMesh and its ParMesh for the parallel path. The solid density is
 * evaluated on the SubMesh; fluid densities on the parent (see FluidRegion).
 *
 * SetRelaxationWeights() reassembles the displacement block only (the
 * interface term is reassembled with it, harmlessly); the potential block,
 * the coupling and the DtN are built once.
 *
 * **Gauged fluid regions** (the alternative to Dahlen's treatment above;
 * see doc/gauged_fluid.md). Instead of FluidRegions, the fluid attributes
 * join the displacement SubMesh: the rheology gives them their bulk modulus
 * with zero shear, @p density covers them, and SetFluid() adds the
 * gauge shear penalty and the Tikhonov refinement of
 * LinearQuasiStaticProblemBase. Every operator of this class then extends
 * over the fluid on its own: the coupling and the gravity terms integrate
 * over the whole SubMesh, and the interface terms (F2)-(F3), the fluid mass
 * term @f$M_F@f$ and its near-indefiniteness, and the 2-D constant-mode
 * inconsistency never arise, because the class sees no FluidRegions. The
 * refinement solves the regularised coupled system against the physical
 * residual @f$[\epsilon Q\,\delta_u; 0]@f$ with zero potential load and no
 * tidal term through the shared refinement driver. The two treatments agree exactly
 * when the fluid is materially barotropic, i.e. its bulk modulus satisfies
 * the Adams-Williamson condition @f$\kappa = \rho^2 g / |d\rho/dr|@f$
 * (@f$N^2 = 0@f$); for a two-parameter fluid the static
 * @f$\kappa@f$-response differs from Dahlen's secular one, and the gauge
 * freedom itself shrinks to label-preserving relabellings. Do not project
 * inner-core rotations in this mode (AddRegionRotations()): with the fluid
 * meshed they cost @f$O(\epsilon)@f$ energy and the penalty handles them.
 */
class LinearQuasiStaticMixedSelfGravitatingProblem
    : public LinearQuasiStaticProblemBase {
 public:
  /** BlockCG runs CG on the same projected block system as BlockMINRES:
   * admissible because the assembled symmetric system is congruent to a
   * positive-definite one for a gravitationally stable body (the
   * physical functional's saddle sign does not survive the
   * symmetrisation); CG breakdown would indicate indefiniteness. */
  enum class SolverType { SchurCG, BlockMINRES, BlockCG };

  /**
   * @brief KKT (exact-gauge) alternative to the penalised gauged fluid:
   * minimise @f$\tfrac12(Qu,u)@f$ subject to the full coupled
   * equations, with @f$Q@f$ the UNIT-scale deviatoric gauge form on
   * @p fluid_marker. One MINRES on the doubled saddle
   * @f$[\mathrm{diag}(Q,0), A; A, 0]@f$: the physical equations hold
   * exactly — no @f$\epsilon@f$, no bias, no Tikhonov refinements, no
   * semi-convergence — and the gauge is fixed as the minimum-Q-energy
   * representative. The auxiliary field @f$v@f$ is determined up to
   * @f$\ker A@f$, which MINRES on the consistent system tolerates.
   * Internally configures the fluid engine at @f$\epsilon = 1@f$ (zero
   * refinements), whose @f$A+Q@f$ assembly doubles as the
   * preconditioner of both displacement slots (of the primal slot only
   * when @p mgw_prec is set). The KKT solve is always MINRES; the call is
   * refused under SchurCG. Not the production path: under the available
   * block-diagonal preconditioners MINRES reaches mesh-level physics but
   * does not converge algebraically at acceptable cost (see
   * doc/quasi_static_models.tex, "The exact-gauge KKT alternative").
   *
   * @param fluid_marker Element attributes of the fluid (as for
   * SetFluid()).
   * @param mu_gauge Gauge shear scale @f$\mu_g@f$; not owned, must outlive
   * the problem.
   * @param mgw_prec Precondition the multiplier displacement slot by the
   * Murphy-Golub-Wathen composite @f$P(A)\,(A+Q)\,P(A)@f$, with
   * @f$P(A)@f$ a preconditioner of the physical @f$A_{uu}@f$, instead of
   * the @f$A+Q@f$ preconditioner. Worthwhile only when @f$P(A)@f$ is of
   * AMG quality; with a Gauss-Seidel smoother it degrades the solve.
   */
  void EnableGaugeKKT(const mfem::Array<int>& fluid_marker,
                      mfem::Coefficient& mu_gauge, bool mgw_prec = false);

  /**
   * @param fes_u Displacement space (vdim = dim) on a (Par)SubMesh of the
   * ball; not owned.
   * @param fes_phi Scalar potential space on the parent (Par)Mesh; not
   * owned.
   * @param rheology The material of the body; must outlive the problem.
   * @param density Density on the body; evaluated on the SubMesh.
   * @param gravitational_constant @f$G@f$ in the units of the problem.
   * @param dtn_degree Truncation degree of the DtN expansion.
   * @param background_potential Optional @f$\Phi_0@f$ (projected onto
   * @p fes_phi); solved from the densities when null.
   * @param fluids Fluid regions (copied); their coefficients must outlive
   * the problem. Empty for a solid body.
   */
  LinearQuasiStaticMixedSelfGravitatingProblem(
      mfem::FiniteElementSpace* fes_u, mfem::FiniteElementSpace* fes_phi,
      const AdGIA::Rheology& rheology, mfem::Coefficient& density,
      mfem::real_t gravitational_constant, int dtn_degree,
      mfem::Coefficient* background_potential = nullptr,
      const std::vector<FluidRegion>& fluids = {});

  // --- loads ----------------------------------------------------------------

  /**
   * @brief Add a surface mass load @f$\sigma@f$ (registered as
   * time-dependent) on the SubMesh boundary attributes marked in
   * @p bdr_marker (sized to the SubMesh's bdr_attributes.Max(); copied).
   */
  void SetSurfaceLoad(mfem::Coefficient& sigma,
                      const mfem::Array<int>& bdr_marker);

  /**
   * @brief Potential load @f$\ell_\phi@f$ as a linear form on the shadow of
   * the potential space on the SubMesh; add integrators before the first
   * AssembleForce(). Markers refer to the SubMesh.
   */
  mfem::LinearForm& ExternalPotentialLoad() { return *b_phi_; }

  /**
   * @brief Apply a (tidal) potential @f$\psi@f$ (registered as
   * time-dependent; interpolated on the potential space at each
   * AssembleForce()). Its load is @f$-c(\psi, v)@f$ on the displacement and
   * @f$-\int_{M_F}\rho'_F\psi\chi@f$ on the potential, i.e. the coupling and
   * fluid-mass operators applied to the interpolant.
   */
  void SetTidalPotential(mfem::Coefficient& psi);

  /**
   * @brief The symmetric tidal pairing of @p psi with the problem's
   * current solution: @f$T(\psi, (u,\phi)) = c(\psi, u) +
   * \int_{M_F}\rho'_F\,\psi\,\phi@f$ — the coupling and fluid-mass
   * operators applied to the interpolant of @f$\psi@f$ and paired with
   * the state. This is the @f$\omega@f$-row functional of the
   * rotational-feedback border (Yu et al. 2025, eqs. A5–A6): by the
   * symmetry of the combined weak form it is exactly minus the tidal
   * load of @f$\psi@f$ dotted with the solution, so its sign and its
   * fluid/interface bookkeeping are inherited from the verified tidal
   * machinery. Global in parallel.
   */
  mfem::real_t TidalCoupling(mfem::Coefficient& psi);

  /** @brief The fluid block of the same pairing between two potentials:
   * @f$\int_{M_F}\rho'_F\,\psi_a\,\psi_b@f$ (zero without fluid
   * regions) — the @f$\psi\psi'@f$ term of eq. A6. */
  mfem::real_t TidalTidalCoupling(mfem::Coefficient& a, mfem::Coefficient& b);

  /**
   * @brief Gravitationally self-consistent water load with a frozen
   * shoreline — the condensed sea-level equation
   * (doc/planning/sea_level_plan.md, "The WP3 derivation").
   *
   * With the ocean weight @f$w = \rho_w C_0/g@f$ and the surface trace
   * @f$\tau = u\cdot\nabla\Phi_0 + \phi@f$, the load
   * @f$\sigma = -w\,\tau + w\,\Phi_g + \sigma_{data}@f$ is folded into
   * the operator: the symmetric boundary blocks
   * @f$-\int w\,\tau'\tau\,dS@f$ (added to the stiffness, the coupling
   * and the potential blocks, with @f$\nabla\Phi_0@f$ in its
   * normal-projected form @f$(m\cdot\nabla\Phi_0)\,m@f$ — exact on an
   * equipotential surface), and the rank-one @f$\Phi_g@f$ border with
   * the mass-conservation row @f$\int\sigma\,dS = 0@f$, eliminated by
   * Sherman–Morrison inside the solve (one extra inner solve per
   * Solve()). @p sigma_data (e.g. @f$\rho_i(1-C_0)I_1@f$, may be
   * time-dependent) is registered as an ordinary surface load, and its
   * surface integral feeds the mass row.
   *
   * Call once, after construction and before the first Solve();
   * incompatible with the gauge-KKT saddle. The uniform term of the
   * last solve is UniformPotentialTerm(); the sea-level change is the
   * postprocessing @f$SL_1 = -\tau/g + \Phi_g/g@f$
   * (SeaLevelOperator). Shoreline migration is strictly second order
   * (Crawford), so the frozen @f$C_0@f$ here is the first-order-exact
   * production path.
   */
  void SetWaterLoad(mfem::Coefficient& ocean_weight,
                    mfem::Coefficient& sigma_data,
                    const mfem::Array<int>& surface_marker);

  /** @brief @f$\Phi_g@f$ of the last Solve() (0 before). */
  mfem::real_t UniformPotentialTerm() const { return phi_g_; }

  /**
   * @brief Rotational feedback inside the solve (the WP4 composition of
   * doc/planning/sea_level_plan.md): the angular-velocity border of the
   * traditional theory, solved jointly with the water load's
   * @f$\Phi_g@f$ border by a small block elimination.
   *
   * The border columns are the centrifugal coupling @f$T(\psi_k,
   * \cdot)@f$ (the tidal operators applied to the unit potentials) and,
   * with SetWaterLoad(), the water's @f$-\int w\,\tau'\psi_k\,dS@f$
   * columns; the dense block is @f$D + P@f$ (inertia + the fluid
   * @f$\psi\psi'@f$ term) minus the surface @f$\int w\psi_k\psi_j@f$
   * block, bordered by @f$\int w\psi_k@f$ against @f$\Phi_g@f$; the
   * @f$\omega@f$-row data @f$\int\sigma\psi_k\,dS@f$ is read off the
   * assembled surface-load form, so every registered surface load
   * enters. Works with or without SetWaterLoad(); without it this
   * reproduces RotationalFeedback exactly (the agreement is a test).
   * The principal moments are data, not model-derived (paper §2.4).
   * Call after construction, before the first Solve(); incompatible
   * with the gauge-KKT saddle and with SetTidalPotential() (the
   * centrifugal potential owns that slot's physics here).
   */
  void SetRotation(mfem::real_t Omega,
                   const mfem::Vector& principal_moments);

  /** @brief @f$\omega@f$ of the last Solve() (zero before; empty
   * without SetRotation()). */
  const mfem::Vector& AngularVelocity() const { return omega_; }

  // --- solver controls ------------------------------------------------------

  void SetSolverType(SolverType type);
  SolverType GetSolverType() const { return type_; }

  /** @brief Relative tolerance of the inner potential solves (default: a
   * hundredth of RelTol(); never below 1e-13, the round-off floor of CG's
   * squared-residual criterion). */
  void SetInnerRelTol(mfem::real_t tol);

  /** @brief Shift @f$\epsilon@f$ of the potential preconditioner
   * @f$(K + \epsilon M)/4\pi G@f$ (default 1e-3). */
  void SetPreconditionerShift(mfem::real_t eps);

  /** @brief Print level of the inner potential solves (quiet by default). */
  void SetInnerPrintLevel(mfem::IterativeSolver::PrintLevel level);
  /**
   * @brief Fix the rigid gauge of the displacement by zero net momentum and
   * angular momentum with the body's density, @f$\int \rho\, u \cdot
   * (a + b \times x) = 0@f$ for every projected mode (including region
   * rotations), instead of orthogonality in the true-dof inner product
   * (the default). Both solvers then return the same representative. The
   * potential follows the displacement in either case.
   */
  void SetMassWeightedGauge(bool on = true);

  /** @brief Replace the background potential and mark the operator stale. */
  void SetBackgroundPotential(mfem::Coefficient& phi0);

  /**
   * @brief Project out the rigid rotations of the solid region(s) with the
   * given SubMesh attributes (a solid inner core enclosed by fluid), which
   * are near-null vectors when density and background potential are
   * invariant under those rotations. May be called at any time before or
   * between solves; translations of the region are never projected.
   */
  void AddRegionRotations(const mfem::Array<int>& solid_attributes);

  // --- outputs --------------------------------------------------------------

  mfem::FiniteElementSpace& PotentialSpace() { return *fes_phi_; }

  /** @brief Potential perturbation on the ball. */
  const mfem::GridFunction& Potential() const { return *phi_; }

  /** @brief Potential perturbation restricted to the body (shadow space). */
  const mfem::GridFunction& PotentialOnBody() const { return *phi_shadow_; }
  /** @brief The shadow of the potential space on the body's SubMesh. */
  mfem::FiniteElementSpace& PotentialSpaceOnBody() { return *shadow_phi_; }
  /**
   * @brief The potential of the assembled potential loads (the surface mass
   * loads) on their own, as if the body were not there: the Laplace-DtN
   * equation solved for the potential load, without the displacement and
   * without the fluid mass term, both of which are the body's response. On
   * the shadow space. For load Love numbers this is the load's own
   * potential on the same discretisation as the solution. Call after
   * AssembleForce(); returns the inner solver's convergence.
   */
  bool SolveLoadPotential(mfem::GridFunction& phi_body);

  const mfem::GridFunction& BackgroundPotential() const { return *phi0_; }
  const mfem::GridFunction& BackgroundPotentialOnBody() const {
    return *phi0_shadow_;
  }

  /** @brief @f$\nabla\Phi_0@f$ on the body, as used in the operator. */
  mfem::VectorCoefficient& BackgroundGravity() const {
    return *grad_phi0_shadow_;
  }

  const PoissonDtNOperator& DtN() const { return *dtn_; }

  /** @brief The coupling operator @f$C@f$ (potential true dofs to
   * displacement true dofs) of the weak form. */
  const mfem::Operator& Coupling() const { return *C_op_; }

  /** @brief The potential block @f$A_{\phi\phi} = (K + \mathrm{DtN})/4\pi G
   * + M_F@f$ on true dofs. */
  const mfem::Operator& PotentialOperator() const { return *A_phiphi_; }

  /** @brief The potential load @f$\ell_\phi@f$ of the last AssembleForce()
   * on true dofs (tidal part and 2-D compatibility included). */
  const mfem::Vector& PotentialLoad() const { return B_phi_; }
  const SubMeshDofInjection& Injection() const { return *injection_; }
  mfem::real_t GravitationalConstant() const { return G_; }
  mfem::Coefficient& Density() const { return *rho_; }

  /** @brief Iterations of the outer solver in the last Solve(). */
  int LastOuterIterations() const { return outer_its_; }

  /** @brief Inner potential-solve iterations accumulated over the last
   * Solve() (zero for BlockMINRES). */
  int LastInnerIterations() const { return inner_its_; }

  /**
   * @brief Diagnostic: how well the discretisation preserves the rigid-body
   * null space. Returns, for each rigid mode @f$u_r@f$ (translations then
   * rotations), @f$\|S u_r\| / (\|A_{uu}\|_{\max} \|u_r\|)@f$ with
   * @f$S@f$ the displacement Schur complement; zero for the continuous
   * problem, and decreasing with refinement for the discrete one. The
   * rigid modes are unit vectors in the Euclidean true-dof norm. Assembles
   * the operator if needed; costs one inner solve per mode.
   */
  std::vector<mfem::real_t> RigidModeResiduals();

  /** @brief The residual of RigidModeResiduals() for an arbitrary true-dof
   * displacement @p u (normalised by its norm): @f$\|S u\| /
   * (\|A_{uu}\|_{\max} \|u\|)@f$. */
  mfem::real_t ModeResidual(const mfem::Vector& u);

  /** @brief Rigid-body modes (orthonormal, true dofs), translations then
   * rotations then any region rotations; the null space projected out of
   * the displacement. */
  const NullSpaceProjector& RigidModes() const { return *projector_u_; }

  /**
   * @brief Diagnostic: the extreme Ritz values of the potential block
   * @f$A_{\phi\phi} = (K + \mathrm{DtN})/4\pi G + M_F@f$ (Euclidean
   * inner product) after @p lanczos_steps Lanczos steps from a fixed
   * pseudo-random start. Returns the smallest; the largest is stored in
   * @p largest when given. In 2-D the constant (projected out of every
   * potential solve) is deflated. Negative means the fluid mass term has
   * made the block indefinite (the inner CG solves then cannot be trusted).
   */
  mfem::real_t PotentialBlockMinEigenvalue(int lanczos_steps = 40,
                                           mfem::real_t* largest = nullptr);

  bool HasFluidRegions() const { return !fluids_.empty(); }
  const std::vector<FluidRegion>& FluidRegions() const { return fluids_; }

  // --- interface overrides --------------------------------------------------

  void AssembleForce(mfem::real_t t) override;

  /** @brief Registers "displacement", "potential" and
   * "background_potential" (the latter two on the body). */
  void RegisterFields(mfem::DataCollection& dc) override;

  /** @brief Gauged-fluid mode requires an empty FluidRegion list (the
   * fluid lives inside the displacement SubMesh; see the class notes):
   * verified here, on the engine's configuration hook. */
  void ConfigureFluidOperator(const mfem::Array<int>& marker,
                              mfem::Coefficient& mu, mfem::real_t epsilon,
                              GaugePenalty form,
                              Diffeomorphism* map) override;

 protected:
  void SetupSolver(mfem::OperatorHandle& A) override;
  bool SolveLinearSystem(const mfem::Vector& B, mfem::Vector& X) override;

  /** @brief The system solve without the water-load border (the body of
   * SolveLinearSystem() before SetWaterLoad()). */
  bool SolveUnbordered(const mfem::Vector& B, mfem::Vector& X);

 private:
  /** @brief @f$S x = A_{uu} x - C A_{\phi\phi}^{-1} C^T x@f$. */
  class SchurOperator : public mfem::Operator {
   public:
    SchurOperator(const LinearQuasiStaticMixedSelfGravitatingProblem& p,
                  const mfem::Operator& A_uu);
    void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

   private:
    const LinearQuasiStaticMixedSelfGravitatingProblem* p_;
    const mfem::Operator* A_uu_;
    mutable mfem::Vector t_, w_, cw_;
  };

  void SetupPotentialOperators();
  void SetupPotentialPreconditioner();
  void SetupPotentialSolver();
  void SetupFluidMass();
  void SetupCoupling();
  void SetupGravityIntegrators();
  void SetupRigidModes();
  void SetupCoupledNullSpace();
  void ComputeBackgroundPotential(mfem::Coefficient* phi0);
  void SetupSchur(mfem::OperatorHandle& A);
  void SetupMinres(mfem::OperatorHandle& A);

  /** @brief Cold-started inner solve of @f$A_{\phi\phi} x = b@f$ on true
   * dofs; returns convergence, accumulates inner_its_. */
  bool SolvePotential(const mfem::Vector& b, mfem::Vector& x) const;

  /** @brief Dual L-vector of @p fes to true dofs (P^T). */
  void ToTrueDofs(const mfem::FiniteElementSpace& fes, const mfem::Vector& L,
                  mfem::Vector& T) const;

  /** @brief In 2-D: make a potential load compatible with the constant. */
  void MakeCompatible(mfem::Vector& B_phi) const;

  /** @brief Push true-dof potential @p Phi into phi_ and phi_shadow_. */
  void DistributePotential(const mfem::Vector& Phi);

  /** @brief Interpolate the tidal potential at the current time and form
   * its loads C Psi and M_F Psi on true dofs. */
  void AssembleTidalLoad();

  /** @brief @f$\rho_F@f$ on the interfaces of region @p f (its
   * interface_density or density). */
  mfem::Coefficient& InterfaceDensity(const FluidRegion& f) const {
    return f.interface_density ? *f.interface_density : *f.density;
  }

  bool ParallelPotential() const;

  // geometry and spaces
  int dim_;
  mfem::FiniteElementSpace* fes_phi_;
#ifdef MFEM_USE_MPI
  mfem::ParFiniteElementSpace* pfes_phi_ = nullptr;
#endif
  std::unique_ptr<mfem::FiniteElementSpace> shadow_phi_;
  std::unique_ptr<SubMeshDofInjection> injection_;

  // physics
  mfem::Coefficient* rho_;
  mfem::real_t G_, four_pi_G_;
  int dtn_degree_;
  mfem::ConstantCoefficient one_, inv_four_pi_G_, shift_coef_;
  std::unique_ptr<mfem::ProductCoefficient> half_rho_, minus_half_rho_;
  std::unique_ptr<mfem::ScalarVectorProductCoefficient> minus_half_rho_grad_;

  // potential fields
  std::unique_ptr<mfem::GridFunction> phi0_, phi0_shadow_, phi_, phi_shadow_;
  std::unique_ptr<mfem::GradientGridFunctionCoefficient> grad_phi0_,
      grad_phi0_shadow_;
  mfem::Vector Phi_true_;

  // fluid regions
  std::vector<FluidRegion> fluids_;
  std::list<mfem::Array<int>> fluid_markers_;  ///< parent-attribute markers
  std::vector<std::unique_ptr<mfem::Coefficient>> fluid_coefs_;
  std::unique_ptr<mfem::L2_FECollection> l2_fec_;
  std::unique_ptr<mfem::FiniteElementSpace> l2_fes_;
  std::vector<std::unique_ptr<mfem::GridFunction>> rho_fluid_l2_;
  std::unique_ptr<BoundaryNormalDotCoefficient> m_dot_grad_phi0_;

  // potential operators: A_lap = (K + DtN)/4piG; A_phiphi = A_lap + M_F
  std::unique_ptr<PoissonDtNOperator> dtn_;
#ifdef MFEM_USE_MPI
  std::unique_ptr<mfem::RAPOperator> dtn_rap_;
#endif
  const mfem::Operator* dtn_op_ = nullptr;
  std::unique_ptr<mfem::BilinearForm> k_phi_form_, k_shift_form_,
      m_fluid_form_;
  mfem::OperatorHandle K_phi_, K_shift_, M_fluid_;
  std::unique_ptr<mfem::SumOperator> A_lap_, A_full_;
  const mfem::Operator* A_phiphi_ = nullptr;
  std::unique_ptr<mfem::Solver> prec_phi_;
  std::unique_ptr<mfem::CGSolver> cg_phi_;
  // 2-D: the constant is projected out, CG runs on P A_phiphi P with the
  // preconditioner P M P.
  std::unique_ptr<NullSpaceProjector> projector_c_;
  std::unique_ptr<ProjectedOperator> projected_phi_op_;
  std::unique_ptr<ProjectedSolver> projected_phi_, projected_prec_phi_;
  mfem::Solver* phi_solver_ = nullptr;

  // coupling C (trial phi on B, test u on M) and its transpose, true dofs
  std::unique_ptr<mfem::MixedBilinearForm> c_form_;
  mfem::OperatorHandle C_;
  std::unique_ptr<mfem::Operator> Ct_owned_;
  const mfem::Operator* C_op_ = nullptr;
  const mfem::Operator* Ct_op_ = nullptr;

  // loads
  std::unique_ptr<mfem::LinearForm> b_phi_;
  mfem::Vector B_phi_;
  std::list<mfem::Array<int>> load_markers_;
  std::vector<std::unique_ptr<mfem::Coefficient>> load_coefs_;
  std::vector<std::unique_ptr<mfem::VectorCoefficient>> load_vcoefs_;
  mfem::Coefficient* psi_ = nullptr;
  std::unique_ptr<mfem::GridFunction> psi_gf_;
  mfem::Vector Psi_true_, tidal_u_, B_eff_;

  // The water-load feedback (SetWaterLoad): the ocean weight, its
  // derived coefficients (kept alive), the sea phi-phi boundary matrix
  // summed onto the potential block, the Phi_g border vectors
  // (c = int w tau' dS on each row; the compatibilised phi copy is the
  // inner-solve rhs in 2-D), the mass-row scalars and the last solve's
  // uniform term.
  bool sea_enabled_ = false;
  mfem::Coefficient* sea_w_ = nullptr;
  mfem::Coefficient* sea_sigma_data_ = nullptr;
  mfem::Array<int> sea_marker_;
  std::vector<std::unique_ptr<mfem::Coefficient>> sea_coefs_;
  std::unique_ptr<mfem::BilinearForm> sea_phiphi_form_;
  mfem::OperatorHandle S_sea_;
  std::unique_ptr<mfem::SumOperator> A_sea_sum_;
  mfem::Vector sea_cu_, sea_cphi_, sea_cphi_compat_;
  mfem::real_t sea_m_ = 0.0, sea_Sd_ = 0.0, phi_g_ = 0.0;
  mfem::Vector sea_yu_, sea_yphi_;

  // The rotational border (SetRotation): the unit centrifugal
  // potentials, the inertia matrix, the water cross-column vectors and
  // surface couplings (built when both features are on), the omega-row
  // data of the current assembly, and the last solve's omega.
  bool rot_enabled_ = false;
  mfem::real_t rot_Omega_ = 0.0;
  mfem::DenseMatrix rot_D_;
  std::vector<std::unique_ptr<mfem::Coefficient>> rot_psi_;
  std::vector<mfem::Vector> rot_cpsi_u_, rot_cpsi_phi_;
  mfem::DenseMatrix rot_Pw_;   ///< int w psi_k psi_j dS
  mfem::Vector rot_mpsi_;     ///< int w psi_k dS
  mfem::Vector rot_r_;        ///< -int sigma psi_k dS per assembly
  mfem::Vector omega_;

  /** @brief Rebuild the water-rotation cross data (both features on). */
  void BuildBorderCrossData();
  /** @brief The surface-mass column of density @p s: the tau'-pairing
   * vectors on the two rows. */
  void BuildSurfaceMassColumn(mfem::Coefficient& s, mfem::Vector& cu,
                              mfem::Vector& cphi);
  /** @brief @f$\int_{\partial M} f\,dS@f$ over the water marker. */
  mfem::real_t SurfaceIntegralOnMarker(mfem::Coefficient& f);

  // 2-D compatibility
  mfem::Vector ones_, L_outer_;
  mfem::real_t outer_length_ = 0.0;

  // null spaces
  std::unique_ptr<NullSpaceProjector> projector_u_, projector_block_;
  bool mass_gauge_ = false;
  std::unique_ptr<mfem::BilinearForm> gauge_form_;
  mfem::OperatorHandle gauge_M_;                   ///< rho-weighted mass
  std::unique_ptr<mfem::BlockOperator> gauge_block_;  ///< diag(M, 0)
  int num_global_modes_ = 0;

  // outer solvers
  SolverType type_ = SolverType::BlockMINRES;
  std::unique_ptr<SchurOperator> schur_;
  std::unique_ptr<ProjectedOperator> projected_op_;
  std::unique_ptr<ProjectedSolver> projected_;
  mfem::Array<int> offsets_;
  std::unique_ptr<mfem::BlockOperator> block_op_;
  std::unique_ptr<mfem::BlockDiagonalPreconditioner> block_prec_;
  std::unique_ptr<ProjectedSolver> projected_prec_;  ///< P M P, both solvers
  std::unique_ptr<mfem::IterativeSolver> minres_;  ///< MINRES or block CG

  // ---- KKT gauge (EnableGaugeKKT): the doubled saddle solver ----
  bool gauge_kkt_ = false;
  mfem::Array<int> offsets_kkt4_;
  std::unique_ptr<NullSpaceProjector> projector_kkt4_;
  std::unique_ptr<mfem::BlockOperator> block_op_kkt4_;
  std::unique_ptr<mfem::BlockDiagonalPreconditioner> block_prec_kkt4_;
  std::unique_ptr<ProjectedOperator> projected_op_kkt4_;
  std::unique_ptr<ProjectedSolver> projected_kkt4_, projected_prec_kkt4_;
  std::unique_ptr<mfem::MINRESSolver> minres_kkt4_;
  std::unique_ptr<mfem::BlockVector> Xk4_, Bk4_;
  bool kkt_mgw_ = false;
  std::unique_ptr<mfem::Solver> prec_clean_u_;  ///< on the CLEAN A_uu
  std::unique_ptr<mfem::Solver> prec_vu_;       ///< MGW composite
  void SetupKKTGauge();

 public:
  /** @brief Diagnostic: relative physical residual
   * @f$\|A(u,\phi) - f\| / \|f\|@f$ of the last gauge-KKT iterate (the
   * constraint rows of the saddle), and the stationarity residual
   * @f$\|Qu + Av\|@f$ relative to @f$\|Qu\|@f$, as a pair. */
  std::pair<mfem::real_t, mfem::real_t> KKTResiduals() const;

 private:
  std::unique_ptr<mfem::BlockVector> X_block_, B_block_;
  mfem::Vector rhs_s_, w_;

  // tolerances and statistics
  mfem::real_t inner_rel_tol_ = 1e-13;
  bool inner_tol_set_ = false;
  mfem::real_t shift_ = 1e-3;
  mfem::IterativeSolver::PrintLevel inner_print_level_;
  int outer_its_ = 0;
  mutable int inner_its_ = 0;
};

}  // namespace AdGIA
