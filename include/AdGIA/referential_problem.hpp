/**
 * @file referential_problem.hpp
 * @brief The general linearised referential problem: quasi-static
 * elasticity about an arbitrarily pre-stressed equilibrium of a general
 * reference body, with referential self-gravity
 * (doc/gravitating_elasticity.md).
 */

#pragma once

#include <array>
#include <list>
#include <memory>
#include <vector>

#include "mfem.hpp"
#include "AdGIA/bilininteg.hpp"
#include "AdGIA/mappings.hpp"
#include "AdGIA/null_space.hpp"
#include "AdGIA/poisson.hpp"
#include "AdGIA/quasi_static_problem.hpp"
#include "AdGIA/rheology.hpp"
#include "AdGIA/submesh.hpp"

namespace AdGIA {

/**
 * @brief A prescribed radial vacuum-extension operator @f$E@f$: buffer
 * displacement vdofs from body displacement vdofs
 * (doc/gravitating_elasticity.md, "3.1 The linearised referential
 * system", option (b)).
 *
 * Buffer dofs shared with the body (the @f$\partial B@f$ trace) copy their
 * body values exactly (through the SubMesh dof pairing), so
 * @f$u_{\mathrm{ext}}|_{\partial B} = u|_{\partial B}@f$ holds to
 * round-off. Interior buffer nodes at radius @f$r@f$ take the tapered
 * radial interpolation @f$t(r)\,u(x_s)@f$ with
 * @f$t = ((r_{\mathrm{out}} - r)/(r_{\mathrm{out}} - r_b))^p@f$,
 * @f$p@f$ = @p taper_power (default 2) — for @f$p > 1@f$ both @f$t@f$ and
 * @f$t'@f$ vanish at the outer (DtN) sphere, keeping @f$a = 1@f$ there —
 * and @f$x_s@f$ the radial projection onto the body
 * surface (pulled inside by @p pullback for robust point location; the
 * interior rule is a gauge choice, so the pull-back is harmless). Any
 * other smooth extension is an equally valid gauge: agreement of
 * observables between two @f$E@f$s is a gauge-invariance test.
 *
 * Both spaces must share the body space's FiniteElementCollection object
 * and live on SubMeshes of one parent. Serial.
 */
std::unique_ptr<mfem::SparseMatrix> NewRadialVacuumExtension(
    mfem::FiniteElementSpace& body_fes, mfem::FiniteElementSpace& buffer_fes,
    mfem::real_t r_body, mfem::real_t r_outer, mfem::real_t taper_power = 2.0,
    mfem::real_t pullback = 0.999);

#ifdef MFEM_USE_MPI
/**
 * @brief Parallel radial vacuum extension on true dofs: trace rows through
 * NewSubMeshPairingTrueDofMatrix (cross-rank), interior rows through a
 * query–reply exchange (surface projections gathered to every rank, each
 * locating what its local body elements contain and replying with
 * global-column interpolation rows; unresolved points retry at smaller
 * radii, as in serial). Semantics as the serial overload.
 */
std::unique_ptr<mfem::HypreParMatrix> NewRadialVacuumExtension(
    mfem::ParFiniteElementSpace& body_fes,
    mfem::ParFiniteElementSpace& buffer_fes, mfem::real_t r_body,
    mfem::real_t r_outer, mfem::real_t taper_power = 2.0,
    mfem::real_t pullback = 0.999);
#endif

/**
 * @brief A prescribed radial fluid-extension operator @f$E@f$: fluid
 * displacement vdofs from solid displacement vdofs — the smooth
 * solid-side extension @f$\tilde v@f$ of the gauged gravity treatment
 * for slipping interfaces (doc/slip_interface.tex, "Gravity: single-valued
 * zeta, and two vanishing results"), for a fluid CORE inside a solid
 * shell.
 *
 * Fluid dofs shared with the solid (the interface trace @f$\Sigma@f$)
 * copy their solid values exactly through the SubMesh dof pairing, so
 * @f$\tilde v|_\Sigma = v_s|_\Sigma@f$ holds to round-off — the property
 * the mismatch @f$w = v_f - \tilde v@f$ and the vanishing lemmas rely
 * on. Interior fluid nodes at radius @f$r@f$ take
 * @f$t(r)\,v_s(x_\Sigma)@f$ with @f$t = (r/r_c)^p@f$, @f$p@f$ =
 * @p taper_power (vanishing at the centre, where the radial direction is
 * undefined; centre nodes keep zero rows) and @f$x_\Sigma@f$
 * the radial projection onto the interface, pushed slightly outward
 * into the solid by @p pushout for robust point location. The interior
 * rule is a gauge choice: agreement of observables between two
 * @f$E@f$s is a gauge-invariance test. The extension is one-sided:
 * nested shells (a fluid layer between two solids) are not supported.
 *
 * Both spaces must share the solid space's FiniteElementCollection
 * object and live on SubMeshes of one parent. Returns fluid vsize by
 * solid vsize. Serial.
 */
std::unique_ptr<mfem::SparseMatrix> NewRadialFluidExtension(
    mfem::FiniteElementSpace& solid_fes, mfem::FiniteElementSpace& fluid_fes,
    mfem::real_t r_interface, mfem::real_t taper_power = 2.0,
    mfem::real_t pushout = 1.001);

#ifdef MFEM_USE_MPI
/**
 * @brief Parallel radial fluid extension on true dofs: trace rows through
 * NewSubMeshPairingTrueDofMatrix (cross-rank), interior rows through the
 * same query–reply exchange as the parallel vacuum extension, with the
 * interface projections pushed *outward* into the solid (retry factors
 * above one, mirroring the serial overload). Semantics as the serial
 * overload; returns fluid true dofs by solid true dofs.
 */
std::unique_ptr<mfem::HypreParMatrix> NewRadialFluidExtension(
    mfem::ParFiniteElementSpace& solid_fes,
    mfem::ParFiniteElementSpace& fluid_fes, mfem::real_t r_interface,
    mfem::real_t taper_power = 2.0, mfem::real_t pushout = 1.001);
#endif

/**
 * @brief The symmetrised slip-interface pressure blocks
 * (doc/slip_interface.tex, "The second variation", proposition "The
 * interface form"): with @f$G@f$ the one-sided
 * kernel of SlipInterfacePressureIntegrator assembled on the solid
 * side, the pairing @f$J@f$, the sum map @f$S = [I, J]@f$ and the jump
 * map @f$D = [I, -J]@f$, the interface bilinear form on the broken pair
 * @f$(u_s, u_f)@f$ is
 * @f[
 *   B_\Sigma = -\tfrac12\,(S^T G D + D^T G^T S),
 * @f]
 * the minus because @f$G@f$ is assembled with the SOLID submesh's
 * outward boundary normal, which is @f$-N@f$ of the derivation
 * (@f$N@f$ out of the fluid) — pinned by the discrete-vs-quadrature
 * cross-check of TestSlipInterface. Returned as four blocks in the
 * (solid, fluid) ordering; @f$B@f$ is symmetric, and it annihilates
 * welded pairs in the quadratic-form sense.
 */
struct SlipInterfaceBlocks {
  std::unique_ptr<mfem::SparseMatrix> ss, sf, fs, ff;
};

/**
 * @param fes_s The solid-side shadow space (SubMeshDofInjection).
 * @param J The dof pairing (solid vdofs x fluid vdofs,
 * NewSubMeshPairingMatrix).
 * @param interface_marker Boundary attributes of @f$\Sigma@f$ on the
 * solid SubMesh.
 * @param pi The referential pressure on the interface.
 * @param map The equilibrium mapping (Nanson normal); pass an
 * IdentityDiffeomorphism at @f$\varphi_e = \mathrm{id}@f$. Serial.
 */
SlipInterfaceBlocks NewSlipInterfaceMatrix(
    mfem::FiniteElementSpace& fes_s, const mfem::SparseMatrix& J,
    const mfem::Array<int>& interface_marker, mfem::Coefficient& pi,
    Diffeomorphism& map);

#ifdef MFEM_USE_MPI
/** @brief The slip-interface blocks on true dofs. */
struct ParSlipInterfaceBlocks {
  std::unique_ptr<mfem::HypreParMatrix> ss, sf, fs, ff;
};

/**
 * @brief Parallel overload of NewSlipInterfaceMatrix: the one-sided
 * kernel is assembled by a ParBilinearForm on the solid side and the
 * blocks are formed by hypre products with the true-dof pairing
 * @p J (NewSubMeshPairingTrueDofMatrix, solid x fluid). Semantics and
 * the normal-convention minus as the serial builder.
 */
ParSlipInterfaceBlocks NewSlipInterfaceMatrix(
    mfem::ParFiniteElementSpace& fes_s, const mfem::HypreParMatrix& J,
    const mfem::Array<int>& interface_marker, mfem::Coefficient& pi,
    Diffeomorphism& map);
#endif

/**
 * @brief The symmetrised broken-@f$\zeta@f$ gravity interface blocks
 * (doc/slip_interface.tex, "Gravity: the broken-zeta organisation",
 * proposition "The gravity interface form"). With @f$G_A@f$ the one-sided
 * vector kernel of
 * SlipInterfaceGravityIntegrator and @f$M_q@f$ the one-sided
 * scalar--vector kernel of SlipInterfaceGravityScalarIntegrator, both
 * assembled on the solid side, the polarised form on the broken pairs
 * @f$(u_s, u_f)@f$, @f$(\zeta_s, \zeta_f)@f$ is
 * @f[
 *   G_\Sigma = \tfrac12\,(S^T G_A D + D^T G_A^T S)
 *   \;+\; \Bigl[\tfrac12\,D^T M_q S_\zeta + \text{transpose}\Bigr],
 * @f]
 * with @f$S = [I, J]@f$, @f$D = [I, -J]@f$ on the vector pairing and
 * @f$S_\zeta = [I, J_\zeta]@f$ on the scalar pairing. The signs are
 * PLUS here (against B_Sigma's minus): the solid-side normal is
 * @f$-N@f$ of the derivation and both @f$\boldsymbol{\nu}@f$-linear
 * coefficients flip with it, cancelling the slip-slot minus of
 * @f$\mathbf{s} = -F_e^{-1}\jump{\mathbf{v}}@f$ — pinned by the
 * discrete-vs-quadrature cross-check of TestSlipInterface.
 *
 * Vector–vector blocks in the (solid, fluid) ordering; the vz blocks
 * are the (vector-row, scalar-column) rectangles, the (scalar, vector)
 * blocks being their transposes. The full quadratic form on
 * @f$(v, \zeta)@f$ is @f$v^T [vv] v + 2\, v^T [vz] \zeta@f$.
 */
struct SlipGravityInterfaceBlocks {
  std::unique_ptr<mfem::SparseMatrix> ss, sf, fs, ff;
  std::unique_ptr<mfem::SparseMatrix> vz_ss, vz_sf, vz_fs, vz_ff;
};

/**
 * @param fes_s The solid-side vector shadow space.
 * @param fes_zs The solid-side scalar space (same mesh and collection).
 * @param J The vector dof pairing (solid vdofs x fluid vdofs).
 * @param Jz The scalar dof pairing (solid dofs x fluid dofs).
 * @param interface_marker Boundary attributes of @f$\Sigma@f$ on the
 * solid SubMesh.
 * @param grad_zeta0 The referential @f$\nabla\zeta^0@f$ on the
 * interface.
 * @param G The gravitational constant.
 * @param map The equilibrium mapping. Serial.
 */
SlipGravityInterfaceBlocks NewSlipGravityInterfaceMatrix(
    mfem::FiniteElementSpace& fes_s, mfem::FiniteElementSpace& fes_zs,
    const mfem::SparseMatrix& J, const mfem::SparseMatrix& Jz,
    const mfem::Array<int>& interface_marker,
    mfem::VectorCoefficient& grad_zeta0, mfem::real_t G, Diffeomorphism& map);

#ifdef MFEM_USE_MPI
/** @brief The broken-@f$\zeta@f$ gravity interface blocks on true dofs. */
struct ParSlipGravityInterfaceBlocks {
  std::unique_ptr<mfem::HypreParMatrix> ss, sf, fs, ff;
  std::unique_ptr<mfem::HypreParMatrix> vz_ss, vz_sf, vz_fs, vz_ff;
};

/**
 * @brief Parallel overload of NewSlipGravityInterfaceMatrix: kernels
 * assembled by Par(Mixed)BilinearForm on the solid side, blocks formed
 * by hypre products with the true-dof pairings. Semantics and sign
 * conventions as the serial builder.
 */
ParSlipGravityInterfaceBlocks NewSlipGravityInterfaceMatrix(
    mfem::ParFiniteElementSpace& fes_s, mfem::ParFiniteElementSpace& fes_zs,
    const mfem::HypreParMatrix& J, const mfem::HypreParMatrix& Jz,
    const mfem::Array<int>& interface_marker,
    mfem::VectorCoefficient& grad_zeta0, mfem::real_t G, Diffeomorphism& map);
#endif

/**
 * @brief The constitutive state of the linearised referential problem as a
 * Rheology: the second elastic tensor @f$\hat C@f$ at equilibrium (Mandel,
 * classical symmetries), the second Piola–Kirchhoff equilibrium stress
 * @f$\mathbf{S}_e@f$, and the equilibrium mapping @f$\varphi_e@f$. Its
 * stiffness is the total-Lagrangian split
 * MaterialStiffnessIntegrator + GeometricStiffnessIntegrator
 * (doc/gravitating_elasticity.md §2). No branches: the rheology is
 * purely elastic and relaxation weights are no-ops.
 *
 * All inputs are non-owning and must outlive the rheology. Remember the
 * moduli convention: @f$\hat C@f$ is the *bare* tensor
 * (BareElasticTensorCoefficient converts seismological moduli).
 */
class ReferentialElasticRheology : public Rheology {
 public:
  ReferentialElasticRheology(int dim, mfem::MatrixCoefficient& C,
                             mfem::MatrixCoefficient& S_e,
                             Diffeomorphism& phi_e);

  int SpaceDim() const override { return dim_; }
  int NumBranches() const override { return 0; }
  mfem::Coefficient& RelaxationTime(int k) const override;
  const RelaxationLaw* Law(int k) const override;
  bool TraceFreeInternalVariables() const override { return true; }
  void BranchModulus(int k, mfem::ElementTransformation& T,
                     const mfem::IntegrationPoint& ip,
                     mfem::DenseMatrix& Ck) const override;
  /** @brief The Mandel tensor @f$\hat C@f$ (the equilibrium-mapping
   * weighting is the integrator's business, not the modulus'). */
  void UnrelaxedModulus(mfem::ElementTransformation& T,
                        const mfem::IntegrationPoint& ip,
                        mfem::DenseMatrix& CU) const override;
  std::unique_ptr<ElasticStiffness> MakeStiffness() const override;

  mfem::MatrixCoefficient& ElasticTensor() const { return *C_; }
  mfem::MatrixCoefficient& EquilibriumStress() const { return *S_; }
  /** @brief Never null here: the rheology always carries a mapping
   * (identity for a natural reference state). */
  Diffeomorphism* EquilibriumMapping() const override { return map_; }

 private:
  int dim_;
  mfem::MatrixCoefficient* C_;
  mfem::MatrixCoefficient* S_;
  Diffeomorphism* map_;
};

/**
 * @brief The linearised quasi-static problem of a self-gravitating,
 * arbitrarily pre-stressed elastic body in the fully referential
 * formulation (doc/gravitating_elasticity.md): both the displacement and
 * the potential perturbation @f$\zeta^1@f$ are referential fields on the
 * fixed reference body.
 *
 * **Reference state.** The class takes the general constitutive state
 * @f$(\hat C, \mathbf{S}_e, \varphi_e)@f$ through its
 * ReferentialElasticRheology: hydrostatic (@f$\mathbf{S}_e = -p_0 I@f$)
 * and natural (@f$\varphi_e = \mathrm{id}@f$; Al-Attar & Crawford 2016)
 * reference states are special *values*, not special cases of the code.
 * The mixed formulation (mixed_problem.hpp) requires both restrictions;
 * this class is where they are lifted.
 *
 * **Geometry.** As for LinearQuasiStaticMixedSelfGravitatingProblem: the body
 * on a (Par)SubMesh of a ball whose outer boundary carries the DtN
 * condition; the displacement @f$u@f$ on the SubMesh, the *referential*
 * potential perturbation @f$\zeta^1@f$ on the ball. The reference body is
 * fixed; the physics of shape lives in the equilibrium mapping
 * @f$\varphi_e@f$ (a Diffeomorphism on the ball, tapering to the identity
 * at the DtN sphere — asserted, not enforced) and the referential fields.
 *
 * **Weak form** (doc/gravitating_elasticity.md §3.1, with
 * @f$a_e = J_e F_e^{-1} F_e^{-T}@f$, @f$\mathbf{g}_0 = \nabla\zeta^0@f$):
 * @f[
 *   \int_B \langle \hat C\,\widehat{\mathrm{sym}(F_e^TDu)},
 *     \widehat{\mathrm{sym}(F_e^TDv)}\rangle
 *   + \mathbf{S}_e : (Du^T Dv)
 *   + \tfrac{1}{8\pi G}\langle a''(u,v)\mathbf{g}_0, \mathbf{g}_0\rangle\,dV
 *   + c(\zeta^1, v) = \ell_u(v),
 * @f]
 * @f[
 *   \tfrac{1}{4\pi G}\Bigl[\int_{B_R} \langle a_e\nabla\zeta^1,
 *     \nabla\chi\rangle + \mathrm{DtN}(\zeta^1,\chi)\Bigr]
 *   + c(\chi, u) = \ell_\zeta(\chi),
 * @f]
 * with the coupling @f$c(\zeta^1, v) = \tfrac{1}{4\pi G}\int_B
 * \langle a'(v)\mathbf{g}_0, \nabla\zeta^1\rangle@f$. The background
 * @f$\zeta^0@f$ solves the referential Poisson equation with the fixed
 * referential density (solved at construction unless supplied).
 *
 * **Loads.** A surface mass load @f$\sigma@f$ (mass per unit *referential*
 * area) loads the potential row alone,
 * @f$\ell_\zeta(\chi) = -\int_{\partial B}\sigma\chi\,dS@f$: the Eulerian
 * form's @f$-\int\sigma\nabla\Phi_0\cdot v@f$ is absorbed by the change of
 * variables (doc/gravitating_elasticity.md §3.1). Further loads via
 * ExternalLoad() /
 * ExternalPotentialLoad() / AddForce().
 *
 * **Null space.** Rigid modes carry *no* potential partners: the null
 * pairs are @f$(t, 0)@f$ — exact discretely, every term vanishing
 * pointwise — and @f$(W\varphi_e, 0)@f$, exactly strain-free and
 * gravitationally near-null. Both are projected from the displacement in
 * the true-dof inner product; the potential needs no projection (2-D:
 * the constant is handled as in the Eulerian class).
 *
 * **Solver**: block MINRES on the symmetric saddle system with the
 * projected block-diagonal preconditioner (AMG/GS on the displacement
 * block and on the shifted mapped Laplacian). Serial and parallel in one
 * class, as throughout.
 */
class LinearQuasiStaticReferentialSelfGravitatingProblem
    : public LinearQuasiStaticProblemBase {
 public:
  /**
   * @param fes_u Displacement space (vdim = dim), not owned: either on a
   * (Par)SubMesh of the ball (the buffer terms then need a prescribed
   * extension; see SetVacuumExtension for the alternative), or on the
   * ball itself ("ball-wide" mode): the displacement carries the gauge
   * vacuum-extension field through the buffer, the constitutive
   * coefficients must vanish there, and SetVacuumExtension() supplies its
   * regularisation (doc/gravitating_elasticity.md §3.1).
   * @param fes_zeta Scalar referential-potential space on the parent
   * (Par)Mesh; not owned.
   * @param rheology The constitutive state (C, S_e, phi_e); must outlive
   * the problem. Its equilibrium mapping must extend over the ball and be
   * the identity at the DtN sphere.
   * @param density Referential density on the body (SubMesh); not owned.
   * @param gravitational_constant @f$G@f$ in the units of the problem.
   * @param dtn_degree Truncation degree of the DtN expansion.
   * @param background_zeta0 Optional @f$\zeta^0@f$ (projected onto
   * @p fes_zeta); solved from the density when null.
   */
  LinearQuasiStaticReferentialSelfGravitatingProblem(
      mfem::FiniteElementSpace* fes_u, mfem::FiniteElementSpace* fes_zeta,
      const ReferentialElasticRheology& rheology, mfem::Coefficient& density,
      mfem::real_t gravitational_constant, int dtn_degree,
      mfem::Coefficient* background_zeta0 = nullptr);

  /** @brief Surface mass load @f$\sigma@f$ per referential area
   * (registered as time-dependent) on the marked boundary attributes of
   * the displacement mesh: a potential-row load only (see the class
   * notes). */
  void SetSurfaceLoad(mfem::Coefficient& sigma,
                      const mfem::Array<int>& bdr_marker);

  /**
   * @brief Tidal (applied-potential) load from the PHYSICAL gradient
   * @f$\nabla\psi@f$ of an external potential, evaluated at the
   * equilibrium-mapped position: a displacement-row load only,
   * @f$\ell_u(v) = -\int_B \tilde\rho\,(\nabla\psi\circ\varphi_e)
   * \cdot v\,dV@f$ with @f$\tilde\rho@f$ the referential density.
   * This is the whole referential statement: by the chain rule
   * @f$F_e^{-T}\nabla_X(\psi\circ\varphi_e) = (\nabla\psi)\circ
   * \varphi_e@f$, so composing analytic physical data with the mapping
   * (HarmonicExpansionGradientCoefficient takes the mapping directly) is
   * the same as differentiating the referential tidal potential — and
   * unlike the Eulerian classes nothing loads the potential row, whose
   * fluid term belongs to the mixed formulation's fluid-density unknown.
   * The centrifugal potential's force fits the same door (its
   * degree-zero @f$r^2@f$ part is not an interior harmonic, so pass its
   * gradient as plain analytic data), which is where rotational
   * feedbacks will later enter (Maitra & Al-Attar 2024). Registered as
   * time-dependent; call before the first AssembleForce().
   */
  void SetTidalLoad(mfem::VectorCoefficient& grad_psi);

  /**
   * @brief Ball-wide mode: regularise the pure-gauge vacuum-extension
   * field with the harmonic penalty @f$\epsilon\mu_g\int \nabla u :
   * \nabla v@f$ on the marked (buffer) attributes, through the fluid
   * engine's Harmonic form with @p refinements Tikhonov
   * refinements. Unlike the gauged fluid, the refinement is not reliable
   * here: the buffer carries no physical stiffness, the generalised
   * spectrum of the (operator, penalty) pair degenerates like @f$h^2@f$,
   * and the refinement diverges under mesh and order refinement
   * (doc/gauge_penalty_iteration.tex, "When it must fail: the spectral
   * condition"). Use @p refinements = 0 (a single biased solve, with an
   * @f$O(\epsilon)@f$ error in the observables), or the prescribed
   * extension of SetPrescribedVacuumExtension(), which is exact. Call
   * before the first Solve().
   */
  void SetVacuumExtension(const mfem::Array<int>& buffer_marker,
                          mfem::Coefficient& mu_gauge, mfem::real_t epsilon,
                          int refinements = 3);

  /** @brief As the base, but the Deviatoric form is assembled
   * covariantly through the rheology's equilibrium mapping when the
   * caller passes no map of its own, so that a relabelled problem's
   * penalty is the exact pull-back of the unmapped one. */
  void ConfigureFluidOperator(const mfem::Array<int>& marker,
                              mfem::Coefficient& mu, mfem::real_t epsilon,
                              GaugePenalty form,
                              Diffeomorphism* map) override;

  /**
   * @brief SubMesh mode: supply the buffer's gravity terms through a
   * *prescribed* extension @f$u_{\mathrm{ext}} = E\,u@f$ on a buffer
   * displacement space (option (b) of doc/gravitating_elasticity.md
   * §3.1, the accurate route): the buffer's gravity-gravity block and
   * coupling are assembled on @p fes_buffer and folded through @f$E@f$
   * by sparse products. Exact for any smooth extension (the choice is
   * gauge); no extra unknowns and no penalty. @p fes_buffer must live on
   * a (Sub)Mesh of the ball sharing this space's collection; @p E maps
   * body vdofs to buffer vdofs (NewRadialVacuumExtension). Serial. Call
   * once, before the first Solve().
   */
  void SetPrescribedVacuumExtension(mfem::FiniteElementSpace& fes_buffer,
                                    const mfem::SparseMatrix& E);

#ifdef MFEM_USE_MPI
  /** @brief Parallel overload: @p E on true dofs
   * (NewRadialVacuumExtension parallel overload); the folds run through
   * hypre RAP/ParMult. */
  void SetPrescribedVacuumExtension(mfem::ParFiniteElementSpace& fes_buffer,
                                    const mfem::HypreParMatrix& E);
#endif

  /** @brief True when the displacement lives on the ball itself. */
  bool BallWide() const { return ball_wide_; }

  /**
   * @brief Zero the stored solution iterate. The block solvers
   * warm-start from the previous Solve(), and the gauge refinement's
   * right-hand side deliberately carries the previous solution's
   * fluid-gauge component forward — right within a continuation, wrong
   * across INDEPENDENT forcings (a benchmark sweeping degrees), where
   * the carried gauge component accumulates. Call between independent
   * solves.
   */
  virtual void ResetSolution();

  /** @brief Potential-row load as a linear form on the shadow of the
   * potential space; add integrators before the first AssembleForce(). */
  mfem::LinearForm& ExternalPotentialLoad() { return *b_zeta_; }

  mfem::FiniteElementSpace& PotentialSpace() { return *fes_zeta_; }

  /** @brief Referential potential perturbation @f$\zeta^1@f$ on the
   * ball. */
  const mfem::GridFunction& Potential() const { return *zeta_; }
  const mfem::GridFunction& PotentialOnBody() const {
    return ball_wide_ ? *zeta_ : *zeta_shadow_;
  }
  mfem::FiniteElementSpace& PotentialSpaceOnBody() { return *shadow_zeta_; }

  const mfem::GridFunction& BackgroundPotential() const { return *zeta0_; }
  const mfem::GridFunction& BackgroundPotentialOnBody() const {
    return ball_wide_ ? *zeta0_ : *zeta0_shadow_;
  }

  /** @brief @f$\nabla\zeta^0@f$ on the body, as used in the operator. */
  mfem::VectorCoefficient& BackgroundGravity() const {
    return *grad_zeta0_shadow_;
  }

  const PoissonDtNOperator& DtN() const { return *dtn_; }
  mfem::real_t GravitationalConstant() const { return G_; }

  /** @brief Iterations of the outer solver in the last Solve(). */
  int LastOuterIterations() const { return outer_its_; }

  /** @brief Rigid-body modes (orthonormal, true dofs): translations, then
   * rotations of the *mapped* positions @f$W\varphi_e@f$. */
  const NullSpaceProjector& RigidModes() const { return *projector_u_; }

  /**
   * @brief Diagnostic: @f$\|A_{\mathrm{blk}}(u, 0)\| /
   * (\|A\|_{\max}\,\|u\|)@f$ for an arbitrary candidate null pair
   * @f$(u, 0)@f$ (true dofs) under the full block operator. Assembles
   * if needed. The sharp test of the tensor dictionary: rigid modes,
   * and in fluid regions linearised relabellings (where the material
   * @f$\mu_b = p^0@f$ term, the geometric @f$-p^0@f$ term and the
   * gravity terms must annihilate the direction jointly, through the
   * equilibrium condition).
   */
  mfem::real_t NullPairResidual(const mfem::Vector& u_true);

  /** @brief Diagnostic: the assembled block operator applied to a
   * (u, zeta) true-dof pair — for entrywise cross-checks between two
   * problems sharing a dof layout (the relabelled identity's probes). */
  void ApplyBlockOperator(const mfem::Vector& u_true,
                          const mfem::Vector& z_true, mfem::Vector& r_u,
                          mfem::Vector& r_z);

  /**
   * @brief Diagnostic: @f$\|A_{\mathrm{blk}} n\| / \|A\|_{\max}@f$ for
   * each rigid null pair @f$(u_r, 0)@f$ under the full block operator.
   * Without a vacuum extension, or with one that carries the body's rigid
   * motions into the buffer, translations are exact discrete null vectors
   * (round-off); the tapered NewRadialVacuumExtension does not, and makes
   * them near-null. Rotations are near-null, decreasing with refinement.
   * Assembles if needed.
   */
  std::vector<mfem::real_t> RigidPairResiduals();

  void AssembleForce(mfem::real_t t) override;
  void RegisterFields(mfem::DataCollection& dc) override;

 protected:
  void SetupSolver(mfem::OperatorHandle& A) override;
  bool SolveLinearSystem(const mfem::Vector& B, mfem::Vector& X) override;

 protected:
  // Protected (not private) so that the slip-interface subclass can reuse
  // the potential machinery, the coupling and the extension folds.
  void SetupPotentialOperators();
  void SetupCoupling();
  void SetupGravityIntegrators();
  void SetupRigidModes();
  void ComputeBackgroundPotential(mfem::Coefficient* zeta0);
  void ToTrueDofs(const mfem::FiniteElementSpace& fes, const mfem::Vector& L,
                  mfem::Vector& T) const;
  void MakeCompatible(mfem::Vector& B_zeta) const;
  void DistributePotential(const mfem::Vector& Z);
  bool ParallelPotential() const;

  // geometry and spaces
  bool ball_wide_ = false;
  int dim_;
  mfem::FiniteElementSpace* fes_zeta_;
#ifdef MFEM_USE_MPI
  mfem::ParFiniteElementSpace* pfes_zeta_ = nullptr;
#endif
  std::unique_ptr<mfem::FiniteElementSpace> shadow_zeta_;
  std::unique_ptr<SubMeshDofInjection> injection_;

  // physics
  const ReferentialElasticRheology* ref_rheology_;
  mfem::Coefficient* rho_;
  std::vector<std::unique_ptr<mfem::Coefficient>> tidal_scalar_coefs_;
  std::vector<std::unique_ptr<mfem::VectorCoefficient>> tidal_coefs_;
  mfem::real_t G_, four_pi_G_;
  int dtn_degree_;
  mfem::ConstantCoefficient one_, inv_four_pi_G_, shift_coef_;

  // potential fields
  std::unique_ptr<mfem::GridFunction> zeta0_, zeta0_shadow_, zeta_,
      zeta_shadow_;
  std::unique_ptr<mfem::GradientGridFunctionCoefficient> grad_zeta0_shadow_;
  mfem::Vector Zeta_true_;

  // potential operators: A_zeta = (K_a + DtN)/4piG
  std::unique_ptr<PoissonDtNOperator> dtn_;
#ifdef MFEM_USE_MPI
  std::unique_ptr<mfem::RAPOperator> dtn_rap_;
#endif
  const mfem::Operator* dtn_op_ = nullptr;
  std::unique_ptr<mfem::BilinearForm> k_zeta_form_, k_shift_form_;
  mfem::OperatorHandle K_zeta_, K_shift_;
  std::unique_ptr<mfem::SumOperator> A_zeta_op_;
  const mfem::Operator* A_zeta_ = nullptr;
  std::unique_ptr<mfem::Solver> prec_zeta_;
  std::unique_ptr<mfem::CGSolver> cg_zeta_;
  std::unique_ptr<NullSpaceProjector> projector_c_;
  std::unique_ptr<ProjectedOperator> projected_zeta_op_;
  std::unique_ptr<ProjectedSolver> projected_zeta_, projected_prec_zeta_;
  mfem::Solver* zeta_solver_ = nullptr;

  // coupling (trial zeta on the ball, test u on the body), true dofs
  std::unique_ptr<mfem::MixedBilinearForm> c_form_;
  mfem::OperatorHandle C_;
  std::unique_ptr<mfem::Operator> Ct_owned_;
  const mfem::Operator* C_op_ = nullptr;
  const mfem::Operator* Ct_op_ = nullptr;

  // prescribed vacuum extension (SubMesh mode, serial)
  std::unique_ptr<mfem::FiniteElementSpace> shadow_zeta_buffer_;
  std::unique_ptr<mfem::GridFunction> zeta0_buffer_;
  std::unique_ptr<mfem::GradientGridFunctionCoefficient> grad_zeta0_buffer_;
  std::unique_ptr<mfem::SparseMatrix> ext_EtGE_, ext_C_total_, ext_Ct_total_;
#ifdef MFEM_USE_MPI
  std::unique_ptr<mfem::HypreParMatrix> pext_EtGE_, pext_C_total_,
      pext_Ct_total_;
#endif
  mfem::OperatorHandle A_aug_;
  mfem::OperatorHandle A_aug_clean_;  // clean-A augmentation of the
                                      // SetGaugePreconditionerOnly mode
  // Keep-alive: the fold matrix the REUSED preconditioner was built on
  // (SetupSolver rebuilds A_aug_ on every call; without the capture a
  // reused AMG would reference the freed previous one).
  mfem::OperatorHandle prec_A_aug_;

  // loads
  std::unique_ptr<mfem::LinearForm> b_zeta_;
  mfem::Vector B_zeta_;
  std::list<mfem::Array<int>> load_markers_;
  std::vector<std::unique_ptr<mfem::Coefficient>> load_coefs_;

  // 2-D compatibility (as in the Eulerian class)
  mfem::Vector ones_, L_outer_;
  mfem::real_t outer_length_ = 0.0;

  // null space and solver
  std::unique_ptr<NullSpaceProjector> projector_u_, projector_block_;
  mfem::Array<int> offsets_;
  std::unique_ptr<mfem::BlockOperator> block_op_;
  std::unique_ptr<mfem::BlockDiagonalPreconditioner> block_prec_;
  std::unique_ptr<ProjectedOperator> projected_op_;
  std::unique_ptr<ProjectedSolver> projected_, projected_prec_;
  std::unique_ptr<mfem::MINRESSolver> minres_;
  std::unique_ptr<mfem::BlockVector> X_block_, B_block_;
  mfem::real_t shift_ = 1e-3;
  int outer_its_ = 0;
};

/**
 * @brief The linearised quasi-static problem with a genuinely *slipping*
 * fluid–solid interface in the referential formulation
 * (doc/slip_interface.tex): the broken displacement pair
 * @f$(u_s, u_f)@f$ on solid and fluid SubMeshes of the ball and the
 * single-valued referential potential @f$\zeta^1@f$ on the ball — the
 * three-block system of doc/slip_interface.tex, "The collected operator".
 * EnableBrokenZeta() switches to the four-block broken-@f$\zeta@f$
 * organisation (doc/slip_interface.tex, "Gravity: the broken-zeta
 * organisation").
 *
 * **Blocks.**
 * - Solid row: the base class's material + geometric + referential-gravity
 *   operators on @f$u_s@f$ and the prescribed vacuum-extension folds,
 *   *plus* the fluid-region gravity assembled with the extension field
 *   @f$\tilde v = E_f u_s@f$ (the @f$a''@f$ and @f$a'@f$ forms folded
 *   through @f$E_f@f$) and the mismatch folds.
 * - Fluid row: the fluid's own material + geometric stiffness (the
 *   dictionary @f$\mu_b = \pi@f$ is the rheology's/background module's
 *   business, the note's @f$\mu_b = \pi@f$ lemma) and the mismatch
 *   gravity pieces in
 *   @f$w = u_f - E_f u_s@f$: the @f$\rho\,\mathrm{sym}(D\mathbf{g}_0)@f$
 *   mass term, the compensated @f$\tilde v@f$-coupling @f$G - M@f$, and
 *   the mismatch–potential coupling @f$\int\rho\,w\cdot\nabla\zeta^1@f$.
 * - Potential row: the base class's mapped Poisson + DtN block.
 * - Interface: the pressure form @f$B_\Sigma@f$ (NewSlipInterfaceMatrix)
 *   plus the penalty @f$\theta[B_n, -B_nJ; -J^TB_n, J^TB_nJ]@f$ on the
 *   normal-jump constraint @f$\nu\cdot[\![u]\!] = 0@f$, with
 *   augmented-Lagrangian iterations driving the jump to zero (or a
 *   multiplier block, EnableKKT()), interleaved with the Tikhonov
 *   refinement of the fluid gauge penalty (the fluid displacement is
 *   determined only up to linearised relabellings, exactly as in the
 *   welded gauged formulation).
 *
 * **Null space**: common translations @f$(t, t, 0)@f$ and *independent*
 * mapped rotations of shell and core (a frictionless axisymmetric
 * interface transmits no torque), plus the 2-D potential constant. With
 * the tapered extensions the translations are near-null rather than
 * exact (as for the vacuum extension).
 *
 * Serial and parallel in one class, as throughout: in parallel the
 * blocks are hypre products on true dofs with the cross-rank pairing
 * and extension operators.
 *
 * **Limitations**: the single-valued organisation assembles its
 * *mismatch* gravity pieces at @f$\varphi_e = \mathrm{id}@f$ only and
 * refuses a non-identity equilibrium mapping
 * (Diffeomorphism::IsIdentity); its elastic, @f$B_\Sigma@f$ and
 * referential gravity terms are fully mapped. Mapped and aspherical
 * backgrounds use the broken-@f$\zeta@f$ organisation
 * (EnableBrokenZeta()), assembled mapped throughout. One fluid region
 * inside a solid shell (the radial fluid extension is one-sided).
 */
class LinearQuasiStaticReferentialSelfGravitatingSlipProblem
    : public LinearQuasiStaticReferentialSelfGravitatingProblem {
 public:
  /** @brief Refused: the slipping classes do not yet carry the tidal
   * body force on the fluid displacement row, and a solid-row-only
   * load under-drives a fluid core by tens of percent. Welded
   * referential and maxwell solve tides (the base class). */
  void SetTidalLoad(mfem::VectorCoefficient&) {
    MFEM_ABORT(
        "SetTidalLoad: not yet supported by the slipping classes (the "
        "fluid row carries no tidal force); use the welded referential "
        "or maxwell treatment for tides.");
  }

  /**
   * @param fes_s Solid displacement space on a SubMesh of the ball; the
   * base class's displacement space.
   * @param fes_f Fluid displacement space on a SubMesh of the ball,
   * sharing @p fes_s's FiniteElementCollection.
   * @param fes_zeta Scalar potential space on the ball.
   * @param rheology The constitutive state, position-based coefficients
   * valid on *both* regions (a fluid region carries @f$\mu_b = \pi@f$
   * automatically through the bare conversion; RadialHydrostaticBackground
   * supplies exactly this).
   * @param density Referential density on both regions.
   * @param interface_pressure The referential pressure @f$\pi@f$ on
   * @f$\Sigma@f$ (the background's pressure field).
   * @param interface_marker Boundary attributes of @f$\Sigma@f$ on the
   * *solid* SubMesh.
   * @param gravitational_constant @f$G@f$ in the units of the problem.
   * @param dtn_degree Truncation degree of the DtN expansion.
   * @param background_zeta0 Optional @f$\zeta^0@f$; when null it is
   * solved from the density over solid *and* fluid.
   */
  LinearQuasiStaticReferentialSelfGravitatingSlipProblem(
      mfem::FiniteElementSpace* fes_s, mfem::FiniteElementSpace* fes_f,
      mfem::FiniteElementSpace* fes_zeta,
      const ReferentialElasticRheology& rheology, mfem::Coefficient& density,
      mfem::Coefficient& interface_pressure,
      const mfem::Array<int>& interface_marker,
      mfem::real_t gravitational_constant, int dtn_degree,
      mfem::Coefficient* background_zeta0 = nullptr);

  /**
   * @brief The prescribed fluid extension @f$\tilde v = E\,u_s@f$
   * (NewRadialFluidExtension): fluid vdofs from solid vdofs, interface
   * trace exact. Required before the first Solve(); the interior rule is
   * gauge, and agreement of observables between two @f$E@f$s is the
   * built-in invariance test. Copied.
   */
  void SetFluidExtension(const mfem::SparseMatrix& E);

#ifdef MFEM_USE_MPI
  /** @brief Parallel overload: @p E on true dofs
   * (NewRadialFluidExtension parallel overload). */
  void SetFluidExtension(const mfem::HypreParMatrix& E);
#endif

  /**
   * @brief The fluid gauge penalty @f$\epsilon\,2\mu_g\,\mathrm{dev}\,
   * \varepsilon(u_f):\mathrm{dev}\,\varepsilon(u_f')@f$ on the whole
   * fluid space (solver operator only; the interleaved refinement removes
   * the @f$O(\epsilon)@f$ bias). Assembled covariantly as
   * ElasticTensorIntegrator(C, map) with the deviatoric tensor and the
   * rheology's equilibrium mapping, so that a relabelled problem's
   * penalty is the pull-back of the unmapped one. Required before the
   * first Solve().
   */
  void SetFluidGauge(mfem::Coefficient& mu_gauge, mfem::real_t epsilon);

  /** @brief The solid-everywhere preconditioner experiment for the
   * slipping organisations: the Krylov operator keeps the CLEAN fluid
   * block (physics + the AL constraint penalty, no gauge term) while
   * the fluid-block preconditioner stays on the
   * @f$+\epsilon_{\mathrm{prec}} Q_f@f$ matrix, and the per-sweep
   * Tikhonov term @f$\epsilon Q u_f@f$ of the AL right-hand side is
   * dropped (no operator bias to remove). @p eps_prec replaces the
   * SetFluidGauge() epsilon for the preconditioner matrix. AL paths
   * only (EnableKKT refuses the mode); combine with
   * SetGaugePlateauStop() for the per-sweep stagnation stop. */
  void SetGaugePreconditionerOnly(mfem::real_t eps_prec) override;

  /** @brief Constraint penalty @f$\theta@f$ (default 100) and the number
   * of augmented-Lagrangian iterations per Solve() (default 8; each
   * interleaves one Tikhonov refinement of the fluid gauge). On the KKT
   * path the count is the number of gauge-refinement solves, and
   * @f$\theta@f$ the augmentation of the displacement blocks. */
  void SetConstraint(mfem::real_t theta, int al_iterations);

  /**
   * @brief KKT (monolithic saddle-point) enforcement of the normal-jump
   * constraint, replacing the penalty + augmented-Lagrangian iterations
   * of the single-valued organisation: a multiplier @f$\lambda@f$ on the
   * interface trace joins the unknowns, the constraint row is the
   * Nanson-exact flux kernel @f$\oint \lambda\,(\nu\cdot
   * [\![\mathbf{v}]\!])\,dS@f$ (BoundaryNormalScalarIntegrator — exact
   * under mappings, all area factors cancelling), and one MINRES solves
   * @f$[S, C^T; C, 0]@f$ with the @f$\theta@f$-augmented blocks kept in
   * @f$S@f$ (Golub–Greif) and the @f$1/\theta@f$-scaled lumped interface
   * mass preconditioning the multiplier block. The outer loop shrinks to
   * the gauge refinements (no multiplier updates); SetConstraint()'s
   * iteration count sets the number of those refinement solves, all of
   * which run. Single-valued organisation only (EnableBrokenZeta() and
   * EnableKKT() are mutually exclusive). See doc/slip_interface.tex,
   * "Constraint enforcement", "KKT enforcement".
   *
   * @param fes_scalar_solid Scalar space on the solid SubMesh; the
   * multiplier is its restriction to the interface boundary dofs. Equal
   * order with the displacement is the strictest constraint space (used
   * by the serial cross-check test). One order below is the mortar-style
   * choice: cheaper at an endpoint shift well below the mesh error, and
   * the recommended setting in 3-D, but not robust everywhere — with
   * quadratic displacements in 2-D it can make the gauge refinements
   * diverge. With a lower-order multiplier, check that
   * NormalJumpHistory() decreases (doc/slip_interface.tex, "The
   * multiplier space"). Not owned.
   */
  void EnableKKT(mfem::FiniteElementSpace* fes_scalar_solid);

  /** @brief Whether KKT enforcement is active. */
  bool KKTEnabled() const { return kkt_; }

  /** @brief The multiplier (interface normal-traction perturbation
   * paired with @f$\nu@f$) of the last KKT Solve(), on the interface
   * dofs of the space passed to EnableKKT(). */
  const mfem::Vector& KKTMultiplier() const { return lambda_; }

  /**
   * @brief Inexact sweeps (the AL sweeps of both organisations and the
   * KKT path's gauge refinements): the inner tolerance of the early
   * sweeps is relaxed to @p loose_rel and tightens geometrically to the
   * solver's relative tolerance, the final sweep always running at full
   * tolerance (doc/slip_interface.tex, "The sweep schedule"). Roughly
   * halves the cost on the 3-D benchmarks at a solver-endpoint shift an
   * order below the mesh error (doc/benchmarks.tex, "Solver settings:
   * measured behaviour"). The default @p loose_rel = 0 keeps every sweep
   * at full tolerance — the reproducible-endpoint mode that strict
   * comparisons and finite-difference studies must use.
   */
  void SetSweepTolerance(mfem::real_t loose_rel) {
    sweep_loose_rel_ = loose_rel;
  }

  /**
   * @brief Switch to the broken-@f$\zeta@f$ organisation
   * (doc/slip_interface.tex, "Gravity: the broken-zeta organisation"):
   * the potential is composed
   * region-wise, @f$(\zeta_o, \zeta_f)@f$ on the outer (solid + buffer)
   * and fluid regions, the gravity sources are exact — the fluid
   * extension and every mismatch term drop, so SetFluidExtension() is
   * not required — and their place is taken by the interface form
   * @f$G_\Sigma@f$ (NewSlipGravityInterfaceMatrix) and the scalar-jump
   * constraint @f$\jump{\zeta^1} = \mathbf{b}\cdot\jump{\mathbf{v}}@f$,
   * imposed by penalty + augmented Lagrangian alongside the normal-jump
   * constraint. The vacuum extension (the outer region's own buffer
   * continuation) is still required. The DtN stays on the ball space
   * and is folded through the outer injection.
   *
   * @param fes_zeta_outer Scalar space on a SubMesh of the ball
   * covering every attribute EXCEPT the fluid (solid + buffer), sharing
   * @c fes_zeta's FiniteElementCollection (a shadow space,
   * SubMeshDofInjection::MakeShadowSpace).
   * @param theta_zeta Penalty for the scalar-jump constraint; the AL
   * iteration count is shared with SetConstraint().
   */
  void EnableBrokenZeta(mfem::FiniteElementSpace* fes_zeta_outer,
                        mfem::real_t theta_zeta);

  /** @brief Whether the broken-@f$\zeta@f$ organisation is active. */
  bool BrokenZetaEnabled() const { return broken_zeta_; }

  /**
   * @brief DIAGNOSTIC (the interface-shift study): override the field
   * @f$\mathbf{b} = F^{-T}\nabla\zeta^0@f$ used in the broken-@f$\zeta@f$
   * scalar-jump constraint kernels (Kvz, Pb) by an externally supplied
   * coefficient — e.g. the analytically composed physical gravity,
   * which is continuous across @f$\Sigma@f$ even when @f$F@f$ jumps
   * there, where the default's one-sided discrete composition is not.
   * Null restores the default. Call before the first Solve(); not
   * owned.
   */
  void SetBrokenConstraintGravity(mfem::VectorCoefficient* b) {
    b_override_ = b;
  }

  /** @brief DIAGNOSTIC (interface-shift study): scale the whole
   * @f$G_\Sigma@f$ gravity-interface block family by @p s (0 drops it
   * symmetrically — the solve then targets a DIFFERENT physical
   * problem; only solver-structure comparisons are meaningful). */
  void ScaleBrokenGravityInterface(mfem::real_t s) { gs_scale_ = s; }

  /** @brief Scalar-jump energies @f$\sqrt{(c, c)_\Sigma}@f$,
   * @f$c = \jump{\zeta^1} - \mathbf{b}\cdot\jump{\mathbf{v}}@f$, at the end of
   * each AL iteration of the last broken-@f$\zeta@f$ Solve(). */
  const std::vector<mfem::real_t>& ZetaJumpHistory() const {
    return zeta_jump_history_;
  }

  /** @brief The outer-region potential @f$\zeta_o^1@f$ (broken mode). */
  const mfem::GridFunction& OuterPotential() const { return *zeta_o_gf_; }
  /** @brief The fluid-region potential @f$\zeta_f^1@f$ (broken mode). */
  const mfem::GridFunction& FluidPotential() const { return *zeta_f_gf_; }

  const mfem::GridFunction& FluidDisplacement() const { return *u_f_; }
  mfem::FiniteElementSpace& FluidSpace() { return *fes_f_; }

  void ResetSolution() override;

  /** @brief Normal-jump energies @f$\sqrt{j^T B_n j}@f$,
   * @f$j = u_s - J u_f@f$, at the end of each AL iteration of the last
   * Solve(). */
  const std::vector<mfem::real_t>& NormalJumpHistory() const {
    return jump_history_;
  }

  /** @brief The accumulated AL multiplier (dual vector on the
   * @f$(u_s, u_f)@f$ blocks): the discrete constraint reaction, i.e. the
   * interface normal-traction *perturbation* paired with @f$\nu@f$. */
  const mfem::BlockVector& ConstraintMultiplier() const { return *w_al_; }

  /**
   * @brief Diagnostic: @f$\|A_{\mathrm{blk}}(u_s, u_f, 0)\| /
   * (\|A\|_{\max}\|(u_s,u_f)\|)@f$ under the full three-block operator
   * (physical interface form included, penalty and gauge penalty
   * excluded). Assembles if needed. Single-valued organisation only.
   */
  mfem::real_t BlockNullPairResidual(const mfem::Vector& us_true,
                                     const mfem::Vector& uf_true);

  /** @brief Residuals of the slip null pairs: common translations, then
   * the independent solid and fluid mapped rotations. Single-valued
   * organisation: BlockNullPairResidual() (penalties excluded);
   * broken-@f$\zeta@f$ organisation: the full four-block solver operator,
   * constraint penalties included, normalised by its largest entry. */
  std::vector<mfem::real_t> SlipRigidPairResiduals();

  /** @brief Diagnostic (broken mode, after a Solve): the assembled
   * broken solver block @f$(i, j)@f$ applied to a true-dof vector —
   * the relabelled identity's per-block probes, localising a
   * non-covariant assembly term to its block. Slots: 0 the solid
   * displacement, 1 the fluid displacement, 2 @f$\zeta_o@f$, 3
   * @f$\zeta_f@f$. The blocks carry the constraint penalties
   * (@f$\theta@f$, @f$\theta_\zeta@f$) folded in, as the solver sees
   * them; the (2,2) block EXCLUDES the DtN fold, which differs between
   * a relabelled pair by the mesh-centroid centring alone. Serial and
   * parallel. */
  void ApplyBrokenBlock(int i, int j, const mfem::Vector& x,
                        mfem::Vector& y);

  /** @brief The space of broken slot @p i, for building probe fields
   * (see ApplyBrokenBlock). */
  mfem::FiniteElementSpace& BrokenSpace(int i);

  /** @brief Diagnostic (broken mode, after a Solve): one stored
   * interface kernel applied to a true-dof vector — the second level of
   * the relabelled identity's probes, separating the constraint kernels
   * by their mapping ingredients. Kernels: "Bn" the normal-jump
   * @f$\nu\cdot\,\cdot\;\nu\cdot\,\cdot@f$ (solid u slot; the Nanson
   * @f$\nu@f$ alone), "Pb" the scalar-jump @f$b\otimes b@f$ mass (solid
   * u slot; the mapped @f$b = F^{-T}\nabla\zeta^0@f$), "Kvz" the mixed
   * @f$b@f$-kernel (zeta_s trial, solid u test), and "Mz" the plain
   * interface scalar mass (zeta_s slot; no mapping ingredient); "KvzT"
   * applies the transpose of "Kvz". */
  void ApplyBrokenKernel(const std::string& kernel, const mfem::Vector& x,
                         mfem::Vector& y);

  /** @brief The base-class fluid treatments are not meaningful here
   * (the fluid has its own space); use SetFluidGauge(). */
  void ConfigureFluidOperator(const mfem::Array<int>&, mfem::Coefficient&,
                              mfem::real_t, GaugePenalty,
                              Diffeomorphism*) override;

  void RegisterFields(mfem::DataCollection& dc) override;

 protected:
  void SetupSolver(mfem::OperatorHandle& A) override;
  bool SolveLinearSystem(const mfem::Vector& B, mfem::Vector& X) override;

 private:
  void RecomputeBackgroundPotential();
  void SetupSlipRigidModes();
  void AssembleSlipBlocks(mfem::OperatorHandle& A);
#ifdef MFEM_USE_MPI
  void AssembleSlipBlocksPar(mfem::OperatorHandle& A);
#endif

  // Broken-zeta organisation (EnableBrokenZeta).
  void AssembleBrokenBlocks(mfem::OperatorHandle& A);
#ifdef MFEM_USE_MPI
  void AssembleBrokenBlocksPar(mfem::OperatorHandle& A);
#endif
  void SetupSolverBroken(mfem::OperatorHandle& A);
  bool SolveLinearSystemBroken(const mfem::Vector& B, mfem::Vector& X);
  void SetupBrokenRigidModes();
  void DistributeBrokenPotential();

  mfem::FiniteElementSpace* fes_f_;
#ifdef MFEM_USE_MPI
  mfem::ParFiniteElementSpace* pfes_f_ = nullptr;
#endif
  mfem::Coefficient* pi_;
  mfem::Array<int> interface_marker_;
  std::unique_ptr<mfem::GridFunction> u_f_;

  // interface pairing (solid x fluid; vdofs in serial, true dofs in
  // parallel) and its transpose
  std::unique_ptr<mfem::SparseMatrix> J_, Jt_;
#ifdef MFEM_USE_MPI
  std::unique_ptr<mfem::HypreParMatrix> pJ_, pJt_;
#endif

  // fluid shadow of the potential space, zeta0 and g0 on the fluid
  std::unique_ptr<mfem::FiniteElementSpace> shadow_zeta_fluid_;
  std::unique_ptr<SubMeshDofInjection> injection_fluid_;
  std::unique_ptr<mfem::GridFunction> zeta0_fluid_;
  std::unique_ptr<mfem::GradientGridFunctionCoefficient> grad_zeta0_fluid_;
  std::unique_ptr<mfem::GridFunction> g0_fluid_gf_;

  // fluid extension, gauge and constraint parameters
  std::unique_ptr<mfem::SparseMatrix> Ef_;
#ifdef MFEM_USE_MPI
  std::unique_ptr<mfem::HypreParMatrix> pEf_;
#endif
  mfem::Coefficient* fluid_mu_gauge_ = nullptr;
  mfem::real_t fluid_gauge_eps_ = 0.0;
  // The gauge-free (1,1) solver block of SetGaugePreconditionerOnly:
  // physics + constraint penalty, no eps Q_f (the Krylov operator's
  // fluid block; the preconditioner keeps the +Q_f matrix).
  std::unique_ptr<mfem::SparseMatrix> S11_noq_;
#ifdef MFEM_USE_MPI
  std::unique_ptr<mfem::HypreParMatrix> pS11_noq_;
#endif
  mfem::real_t theta_ = 1.0e2;
  int al_iterations_ = 8;
  mfem::real_t sweep_loose_rel_ = 0.0;  ///< 0: every sweep at full tol

  // assembled physical blocks, the solver blocks with the constraint
  // penalty folded in (S**, A11_solve_), the penalty kernel Bn and the
  // fluid gauge penalty eps Q; serial sparse or hypre per the spaces,
  // with type-generic views (op_*) for the solver and diagnostics
  std::unique_ptr<mfem::SparseMatrix> A00_, A01_, A10_, A11_, A02_, A20_,
      A12_, A21_, S00_, S01_, S10_, A11_solve_, Qf_, Bn_;
#ifdef MFEM_USE_MPI
  std::unique_ptr<mfem::HypreParMatrix> pA00_, pA01_, pA10_, pA11_, pA02_,
      pA20_, pA12_, pA21_, pS00_, pS01_, pS10_, pA11_solve_, pQf_, pBn_;
#endif
  const mfem::Operator *op_A00_ = nullptr, *op_A01_ = nullptr,
                       *op_A10_ = nullptr, *op_A11_ = nullptr,
                       *op_A02_ = nullptr, *op_A20_ = nullptr,
                       *op_A12_ = nullptr, *op_A21_ = nullptr,
                       *op_S00_ = nullptr, *op_S01_ = nullptr,
                       *op_S10_ = nullptr, *op_S11_ = nullptr,
                       *op_Qf_ = nullptr, *op_Bn_ = nullptr,
                       *op_J_ = nullptr, *op_Jt_ = nullptr;

  // ---- KKT enforcement (EnableKKT) ----
  bool kkt_ = false;
  mfem::FiniteElementSpace* fes_lam_ = nullptr;
  mfem::Array<int> lam_dofs_;  ///< interface true dofs of fes_lam_
  // Constraint kernel N (u test x scalar trial) on true dofs; the
  // constraint row is its transpose restricted to lam_dofs_.
  std::unique_ptr<mfem::SparseMatrix> Nlam_;
#ifdef MFEM_USE_MPI
  std::unique_ptr<mfem::HypreParMatrix> pNlam_;
#endif
  const mfem::Operator* op_Nlam_ = nullptr;
  mfem::Vector lambda_, lam_mass_diag_;
  // Consistent theta-augmentation of the KKT solver blocks: theta *
  // N Mhat^{-1} N^T replaces theta * Bn (same kernel as the constraint
  // row, clustering the Schur complement at M_lam / theta).
  std::unique_ptr<mfem::SparseMatrix> SK00_, SK01_, SK10_, SK11_;
#ifdef MFEM_USE_MPI
  std::unique_ptr<mfem::HypreParMatrix> pSK00_, pSK01_, pSK10_, pSK11_;
#endif
  std::unique_ptr<mfem::Solver> prec11_kkt_;
  mfem::Array<int> offsets_kkt_;
  std::unique_ptr<NullSpaceProjector> projector_kkt_;
  std::unique_ptr<mfem::BlockOperator> block_op_kkt_;
  std::unique_ptr<mfem::BlockDiagonalPreconditioner> block_prec_kkt_;
  std::unique_ptr<mfem::Solver> prec_lam_;
  std::vector<std::unique_ptr<mfem::Operator>> kkt_ops_;
  std::unique_ptr<ProjectedOperator> projected_op_kkt_;
  std::unique_ptr<ProjectedSolver> projected_kkt_, projected_prec_kkt_;
  std::unique_ptr<mfem::MINRESSolver> minres_kkt_;
  std::unique_ptr<mfem::BlockVector> Xk_, Bk_;
  void SetupSolverKKT(mfem::OperatorHandle& A);
  bool SolveLinearSystemKKT(const mfem::Vector& B, mfem::Vector& X);

  // three-block solver
  mfem::Array<int> offsets3_;
  std::unique_ptr<NullSpaceProjector> projector3_;
  std::unique_ptr<mfem::BlockOperator> block_op3_;
  std::unique_ptr<mfem::BlockDiagonalPreconditioner> block_prec3_;
  std::unique_ptr<mfem::Solver> prec11_;
  std::unique_ptr<ProjectedOperator> projected_op3_;
  std::unique_ptr<ProjectedSolver> projected3_, projected_prec3_;
  std::unique_ptr<mfem::MINRESSolver> minres3_;
  std::unique_ptr<mfem::BlockVector> X3_, B3_, w_al_;
  std::vector<mfem::real_t> jump_history_;

  // ---- broken-zeta organisation (EnableBrokenZeta) ----
  bool broken_zeta_ = false;
  mfem::real_t theta_zeta_ = 0.0;
  mfem::FiniteElementSpace* fes_zo_ = nullptr;
  mfem::VectorCoefficient* b_override_ = nullptr;
  mfem::real_t gs_scale_ = 1.0;
#ifdef MFEM_USE_MPI
  mfem::ParFiniteElementSpace* pfes_zo_ = nullptr;
  std::unique_ptr<mfem::ParFiniteElementSpace> pfes_zs_solid_;
#endif
  std::unique_ptr<mfem::FiniteElementSpace> fes_zs_solid_;
  std::unique_ptr<mfem::GridFunction> zeta_o_gf_, zeta_f_gf_;
  std::vector<mfem::real_t> zeta_jump_history_;

  // transfers: Pzo (ball x outer injection), Pos (solid-scalar x
  // outer-scalar pairing over the whole solid), Jzsf (solid-scalar x
  // fluid-scalar pairing at Sigma); serial sparse or hypre true-dof.
  std::unique_ptr<mfem::SparseMatrix> Pzo_, Pos_, Jzsf_, PosT_, JzsfT_;
#ifdef MFEM_USE_MPI
  std::unique_ptr<mfem::HypreParMatrix> pPzo_, pPos_, pJzsf_, pPosT_,
      pJzsfT_;
#endif
  const mfem::Operator *op_Pzo_ = nullptr, *op_Pos_ = nullptr,
                       *op_PosT_ = nullptr, *op_Jzsf_ = nullptr,
                       *op_JzsfT_ = nullptr;

  // solver blocks (physical + both penalties + eps Q folded), row-major
  // 4x4; the (2,2) entry is the sparse part, completed by the DtN fold.
  std::array<std::unique_ptr<mfem::SparseMatrix>, 16> bzS_;
#ifdef MFEM_USE_MPI
  std::array<std::unique_ptr<mfem::HypreParMatrix>, 16> pbzS_;
#endif
  // constraint kernels on the solid side: scalar boundary mass Mz,
  // the (vector x scalar) coupling Kvz = oint (b.u) zeta, and the
  // (b x b) vector boundary mass Pb.
  std::unique_ptr<mfem::SparseMatrix> Mz_, Kvz_, Pb_;
#ifdef MFEM_USE_MPI
  std::unique_ptr<mfem::HypreParMatrix> pMz_, pKvz_, pPb_;
#endif
  const mfem::Operator *op_Mz_ = nullptr, *op_Kvz_ = nullptr,
                       *op_Pb_ = nullptr;

  // zeta_o solver block: sparse part + DtN folded through Pzo.
  std::unique_ptr<mfem::Operator> dtn_fold_;
  std::unique_ptr<mfem::SumOperator> S22_op_;
  std::unique_ptr<mfem::Solver> prec22_, prec33_;
  std::unique_ptr<mfem::SparseMatrix> prec22_mat_, prec33_mat_;
#ifdef MFEM_USE_MPI
  std::unique_ptr<mfem::HypreParMatrix> pprec22_mat_, pprec33_mat_;
#endif
  mfem::Vector ones_o_, ones_f_, L_outer_o_;
  mfem::real_t outer_length_o_ = 0.0;

  // four-block solver
  mfem::Array<int> offsets4_;
  std::unique_ptr<NullSpaceProjector> projector4_;
  std::unique_ptr<mfem::BlockOperator> block_op4_;
  std::unique_ptr<mfem::BlockDiagonalPreconditioner> block_prec4_;
  std::unique_ptr<ProjectedOperator> projected_op4_;
  std::unique_ptr<ProjectedSolver> projected4_, projected_prec4_;
  std::unique_ptr<mfem::MINRESSolver> minres4_;
  std::unique_ptr<mfem::BlockVector> X4_, B4_, w_al4_;
};

}  // namespace AdGIA
