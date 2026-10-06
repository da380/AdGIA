#ifndef ADGIA_EQUILIBRIUM_FIGURES_HPP
#define ADGIA_EQUILIBRIUM_FIGURES_HPP

#include <memory>

#include "mfem.hpp"

#include "AdGIA/background.hpp"

namespace AdGIA {

/**
 * @brief The density feasibility functional of the equilibrium-figures
 * programme (doc/equilibrium_figures.tex): does the model possess a
 * valid static state, and how should its fluid density move towards
 * one?
 *
 * Given a density @f$\rho@f$ on a body with fluid regions, the object
 * performs, at construction,
 *
 * 1. the self-consistent potential: one Poisson solve
 *    @f$\nabla^2\Phi = 4\pi G\rho@f$ over the whole mesh (body and
 *    buffer), closed by the matrix-free DtN operator on the spherical
 *    outer boundary;
 * 2. the minimum-deviatoric Stokes saddle
 *    (MinimumDeviatoricEquilibriumStress) on the given Taylor–Hood pair,
 *    driven by the body force @f$\rho\nabla\Phi@f$ — in the *fluid-only*
 *    variant the pair lives on the fluid SubMesh with the velocity
 *    clamped on its whole boundary (@p essential_bdr), in the *weighted
 *    whole-body* variant on the body SubMesh with a two-region
 *    viscosity @p mu;
 *
 * and exposes the value
 * @f[
 *   J \;=\; \int \mu\,|\varepsilon(\mathbf{u})|^2
 *   \;=\; \tfrac12\int \tfrac{1}{2\mu}|\mathrm{dev}\,\mathbf{T}|^2 ,
 * @f]
 * computed discretely as @f$-\tfrac12 F\cdot u@f$ (the saddle's own
 * value function), together with its density derivative by the envelope
 * theorem: @f$\rho@f$ enters the Lagrangian only through the load and
 * the potential, so
 * @f[
 *   \delta J = -\langle \mathbf{u},\,\delta\rho\,\nabla\Phi\rangle
 *              - \langle \mathbf{u},\,\rho\,\nabla\delta\Phi\rangle ,
 * @f]
 * with no transposed saddle solve — the primal multiplier is the
 * adjoint variable — and the self-gravity chain costing one further
 * solve with the *same* Poisson operator: @f$\delta J =
 * \int \delta\rho\,(-\mathbf{u}\cdot\nabla\Phi + 4\pi G\,w)@f$ where
 * @f$w@f$ solves the Poisson problem with the source functional
 * @f$v \mapsto \int \rho\,\mathbf{u}\cdot\nabla v@f$ over the Stokes
 * region. Serial and parallel.
 *
 * The derivative is returned as a **dual vector** against a control
 * space on the Stokes mesh; the identification of a gradient (the L2
 * mass solve, or a Sobolev Riesz map) is deliberately the caller's
 * business — the metric is the outer loop's preconditioner, not part
 * of the functional.
 *
 * The density is supplied twice, as a coefficient on the parent mesh
 * (the Poisson source; zero in the buffer) and one on the Stokes mesh
 * (the load), because a coefficient bound to a GridFunction evaluates
 * only on its own mesh; the caller guarantees the two describe the
 * same field (an analytic coefficient may simply be passed twice).
 *
 * In two dimensions the exterior problem carries the logarithmic
 * infrared structure documented in doc/self_gravitation.md ("2-D
 * caveats"); the class is written for, and verified in, three
 * dimensions.
 */
class DensityFeasibility;

/**
 * @brief The persistent half of the feasibility evaluation: none of the
 * operators — the Poisson system with its DtN closure and
 * preconditioner, the Stokes saddle with its — depends on the density,
 * so an optimisation loop assembles them once here and pays only the
 * solves per evaluation. Parameters as in DensityFeasibility, minus
 * the densities; everything is borrowed and must outlive the problem.
 */
class DensityFeasibilityProblem {
 public:
  DensityFeasibilityProblem(mfem::FiniteElementSpace& fes_phi,
                            int dtn_degree, mfem::real_t G,
                            const mfem::Array<int>& stokes_attributes,
                            mfem::FiniteElementSpace& fes_u,
                            mfem::FiniteElementSpace& fes_p,
                            mfem::Coefficient* mu = nullptr,
                            const mfem::Array<int>* essential_bdr = nullptr);
  ~DensityFeasibilityProblem();

  /** @brief The functional, its fields and its derivative at one
   * density (the two coefficients as in DensityFeasibility). */
  std::unique_ptr<DensityFeasibility> Evaluate(
      mfem::Coefficient& rho_parent, mfem::Coefficient& rho_stokes) const;

 private:
  friend class DensityFeasibility;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class DensityFeasibility {
 public:
  /**
   * @param fes_phi Scalar H1 space on the parent mesh (body and buffer,
   * spherical outer boundary); not owned.
   * @param dtn_degree Truncation degree of the DtN expansion.
   * @param G Gravitational constant, in the model's units.
   * @param rho_parent The density on the parent mesh (zero in the
   * buffer); not owned, used during construction.
   * @param stokes_attributes Parent domain attributes of the Stokes
   * mesh (the fluid regions, or the whole body), for the adjoint
   * source's marker-restricted assembly.
   * @param fes_u Vector H1 velocity space on the Stokes SubMesh of the
   * parent; not owned.
   * @param fes_p Scalar H1 pressure space, one order below (Taylor–
   * Hood); not owned.
   * @param rho_stokes The same density, evaluable on the Stokes mesh;
   * not owned, used during construction.
   * @param mu Optional viscosity weight (the weighted whole-body
   * variant's two-region field); constant when null.
   * @param essential_bdr Optional boundary marker clamping the
   * velocity (the fluid-only variant: every boundary attribute of the
   * fluid SubMesh); forwarded to MinimumDeviatoricEquilibriumStress.
   */
  DensityFeasibility(mfem::FiniteElementSpace& fes_phi, int dtn_degree,
                     mfem::real_t G, mfem::Coefficient& rho_parent,
                     const mfem::Array<int>& stokes_attributes,
                     mfem::FiniteElementSpace& fes_u,
                     mfem::FiniteElementSpace& fes_p,
                     mfem::Coefficient& rho_stokes,
                     mfem::Coefficient* mu = nullptr,
                     const mfem::Array<int>* essential_bdr = nullptr);

  ~DensityFeasibility();

  /** @brief The feasibility value @f$J@f$ (the saddle's value
   * function). */
  mfem::real_t Value() const { return value_; }

  /** @brief The gravitational energy @f$E = \tfrac12\int\rho\,\Phi@f$
   * of the state. The advection flow is E's gradient flow in the
   * dissipation metric with decay rate @f$2J@f$: an advection loop
   * accepts steps on E, not on J, which need not fall monotonically
   * along the flow. */
  mfem::real_t GravitationalEnergy() const { return energy_; }

  /** @brief The self-consistent potential on the parent mesh. */
  const mfem::GridFunction& Potential() const { return *phi_; }

  /** @brief The adjoint potential @f$w@f$ on the parent mesh. */
  const mfem::GridFunction& AdjointPotential() const { return *w_; }

  /** @brief The Stokes multiplier (the relaxation flow) on the Stokes
   * mesh. */
  const mfem::GridFunction& Velocity() const;

  /** @brief The stress generator, for the stress field itself and its
   * diagnostics. */
  const MinimumDeviatoricEquilibriumStress& Stress() const {
    return *stress_;
  }

  /** @brief Solver iterations of the two Poisson solves. */
  int PotentialIterations() const { return phi_iterations_; }
  int AdjointIterations() const { return w_iterations_; }

  /**
   * @brief The derivative @f$J'(\rho)@f$ as a dual vector against
   * @p fes_rho, a (typically L2) space on the Stokes mesh:
   * @f$\mathrm{dual}_j = \int \psi_j\,(-\mathbf{u}\cdot\nabla\Phi
   * + 4\pi G\,w)@f$ over the Stokes mesh. True-dof sized.
   */
  void Derivative(mfem::FiniteElementSpace& fes_rho,
                  mfem::Vector& dual) const;

  /**
   * @brief The same derivative assembled against a space on the
   * PARENT mesh, restricted to the Stokes region's attributes — the
   * extended-domain form the Sobolev metrics consume
   * (doc/equilibrium_figures.tex §6: the parameter lives on the whole
   * mesh, the model reads its restriction, and the dual of the
   * restriction is this marker-restricted assembly).
   */
  void DerivativeOnParent(mfem::FiniteElementSpace& fes,
                          mfem::Vector& dual) const;

 private:
  friend class DensityFeasibilityProblem;
  /** The state constructor the problem's Evaluate uses: the heavy
   * pieces come assembled from @p problem. */
  DensityFeasibility(const DensityFeasibilityProblem& problem,
                     mfem::Coefficient& rho_parent,
                     mfem::Coefficient& rho_stokes);
  void Solve(mfem::Coefficient& rho_parent, mfem::Coefficient& rho_stokes);

  // One-shot compatibility: the public constructor owns its problem.
  std::unique_ptr<DensityFeasibilityProblem> own_problem_;
  const DensityFeasibilityProblem* problem_ = nullptr;

  mfem::real_t value_ = 0.0;
  mfem::real_t energy_ = 0.0;
  int phi_iterations_ = 0, w_iterations_ = 0;

  // The per-state fields: potentials on the parent, their restrictions
  // to the Stokes mesh, the velocity's parent-side twin, the stress.
  std::unique_ptr<mfem::GridFunction> phi_, w_, phi_sub_, w_sub_, u_parent_;
  std::unique_ptr<MinimumDeviatoricEquilibriumStress> stress_;
};

}  // namespace AdGIA

#endif  // ADGIA_EQUILIBRIUM_FIGURES_HPP
