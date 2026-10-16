/**
 * @file fault_problem.hpp
 * @brief A meshed fault with prescribed slip: the clamped quasi-static
 * problem with a displacement discontinuity across an interior fault
 * surface (LinearQuasiStaticClampedFaultProblem), and the linear map from
 * fault slip to the displacement of an observation surface with its
 * adjoint, surface traction to fault traction (FaultToSurfaceMap).
 */

#pragma once

#include <memory>

#include "mfem.hpp"
#include "AdGIA/quasi_static_problem.hpp"
#include "AdGIA/submesh.hpp"

namespace AdGIA {

/**
 * @brief Clamped problem with a prescribed slip across a meshed fault.
 *
 * **Mesh.** The fault is an interior surface of boundary elements (the
 * fault attributes) whose one side is the *hanging wall*, a set of element
 * attributes, the other side the footwall. The hanging-wall elements must
 * include every element on that side touching the fault. The fault edge,
 * where the fault ends inside the body and the displacement is
 * single-valued, is given by boundary attributes whose dofs carry no slip:
 * typically the extension of the fault plane to the outer boundary, which
 * also separates the two sides into volumes (meshes/faulted_box.py).
 *
 * **Method** (the split-node method). The unknown is the displacement
 * @f$u_w@f$ in the ordinary continuous space; the slip @f$\hat s@f$ is a
 * field of the same space carried by the fault dofs, zero on the edge dofs
 * and everywhere else. The physical displacement is
 * @f[
 *   u = u_w + \chi_{\mathrm{HW}}\,\hat s,
 * @f]
 * with @f$\chi_{\mathrm{HW}}@f$ the indicator of the hanging-wall elements,
 * so that @f$[\![u]\!] = u_{\mathrm{HW}} - u_{\mathrm{FW}} = \hat s@f$ on
 * the fault exactly, and @f$u_w@f$ solves the clamped problem with the
 * additional load
 * @f[
 *   b(v) = -a_{\mathrm{HW}}(\hat s, v),
 * @f]
 * the stiffness over the hanging-wall elements acting on the slip
 * (DomainLFFormActionIntegrator in ExternalLoad(), borrowing the
 * problem's own stiffness integrators, so that anisotropic, composite and
 * relaxed rheologies are followed). This is the exact solution in the
 * finite element space broken across the fault (equal to a
 * Lagrange-multiplier solution with an equal-order multiplier), obtained
 * with one solve of the unbroken system. Displacement() is @f$u_w@f$;
 * PhysicalDisplacement() is @f$u@f$. Away from the elements touching the
 * fault the two agree, so the surface displacement of a buried fault is
 * @f$u_w@f$.
 *
 * **Conventions.** The slip is the motion of the hanging wall relative to
 * the footwall. With @f$\nu@f$ the fault normal pointing into the hanging
 * wall, the source is the stress glut @f$C : (\hat s \otimes \nu)@f$ on the
 * fault (for an isotropic body and tangential slip, the moment density
 * @f$\mu(\hat s\nu^T + \nu\hat s^T)@f$).
 *
 * **Fault reaction.** FaultReaction() returns
 * @f$r = A_{\mathrm{HW}}(u_w + \hat s)@f$ on the slip-carrying dofs, the
 * consistent nodal form of @f$-\tau@f$, @f$\tau = \sigma\nu@f$ the traction
 * on the fault (by equilibrium of the footwall, also @f$-A_{\mathrm{FW}}u_w@f$
 * there). As a function of the slip it is the Schur complement of the
 * broken stiffness: symmetric and positive.
 *
 * Serial and parallel in one class: the space decides. The slip is built
 * only through rank-consistent paths (a rank may hold a hanging-wall
 * element touching a fault dof without the fault face).
 *
 * **Limitations.** Viscoelastic stepping on this class is consistent when
 * the elements touching the fault are elastic: the viscoelastic layer's
 * strain map sees @f$u_w@f$, not the strain of the hanging-wall lifting.
 */
class LinearQuasiStaticClampedFaultProblem
    : public LinearQuasiStaticClampedProblem {
 public:
  /**
   * @param fes Displacement space; serial or parallel; not owned.
   * @param rheology The material; not owned, must outlive the problem.
   * @param ess_bdr Boundary attributes with zero displacement (copied).
   * @param traction Boundary traction; registered as time-dependent.
   * @param traction_marker Boundary attributes it acts on (copied).
   * @param hanging_wall_marker Element attributes of the hanging wall,
   * sized to attributes.Max() (copied).
   * @param fault_marker Boundary attributes of the fault, sized to
   * bdr_attributes.Max() (copied).
   * @param fault_edge_marker Boundary attributes whose dofs carry no slip,
   * sized to bdr_attributes.Max() (copied); all zero for a fault that
   * separates the body.
   */
  LinearQuasiStaticClampedFaultProblem(
      mfem::FiniteElementSpace* fes, const AdGIA::Rheology& rheology,
      const mfem::Array<int>& ess_bdr, mfem::VectorCoefficient& traction,
      const mfem::Array<int>& traction_marker,
      const mfem::Array<int>& hanging_wall_marker,
      const mfem::Array<int>& fault_marker,
      const mfem::Array<int>& fault_edge_marker);

  /** @brief The slip from a coefficient, interpolated at the fault dofs
   * (zero on the edge dofs). Takes effect at the next AssembleForce(). */
  void SetSlip(mfem::VectorCoefficient& slip);

  /** @brief The slip from a true-dof vector of the displacement space;
   * entries off the slip-carrying dofs are ignored. Takes effect at the
   * next AssembleForce(). */
  void SetSlipTrueDofs(const mfem::Vector& slip_true);

  /** @brief Zero slip. */
  void ClearSlip();

  /** @brief The slip field @f$\hat s@f$ on the displacement space. */
  const mfem::GridFunction& Slip() const { return *slip_; }

  /** @brief The true dofs that carry slip: the fault's minus the edge's. */
  const mfem::Array<int>& SlipTrueDofs() const { return slip_tdofs_; }

  /** @brief The physical displacement @f$u_w + \chi_{\mathrm{HW}}\hat s@f$
   * of the current state, discontinuous across the fault; project it onto
   * a discontinuous space to show the jump, or sample it. */
  mfem::VectorCoefficient& PhysicalDisplacement() { return *physical_; }

  /** @brief The fault reaction @f$A_{\mathrm{HW}}(u_w + \hat s)@f$ of the
   * current state as a true-dof dual vector, zero off SlipTrueDofs(). */
  void FaultReaction(mfem::Vector& r_true);

  void RegisterFields(mfem::DataCollection& dc) override;

 private:
  mfem::Array<int> hanging_wall_, fault_, edge_;
  mfem::Array<int> slip_vdof_mask_;
  mfem::Array<int> slip_tdofs_;
  std::unique_ptr<mfem::GridFunction> slip_;
  std::unique_ptr<mfem::GridFunction> reaction_field_;
  std::unique_ptr<mfem::LinearForm> reaction_form_;
  std::unique_ptr<mfem::VectorCoefficient> physical_;

  void MaskSlip();
};

/**
 * @brief The linear map from fault slip to the displacement of an
 * observation surface, through a LinearQuasiStaticClampedFaultProblem,
 * and its adjoint.
 *
 * Both spaces are shadow spaces (SubMeshDofInjection::MakeShadowSpace) of
 * the problem's displacement space on boundary SubMeshes: @p fault_fes on
 * the fault, @p surface_fes on the observation surface. Mult() takes a
 * slip in the fault space's true dofs (entries on the edge dofs are
 * ignored), solves, and returns the displacement @f$u_w@f$ restricted to the
 * surface's true dofs (the physical displacement there when the fault is
 * buried). MultTranspose() is its exact adjoint in the Euclidean true-dof
 * pairing: it takes a dual vector on the surface space — the traction of a
 * surface observable, e.g. @f$w = M_{\mathrm{surface}}\,w_{\mathrm{obs}}@f$
 * for @f$\int w_{\mathrm{obs}}\cdot u\,dS@f$ — solves the problem under
 * that surface load with zero slip, and returns the traction it induces on
 * the fault, @f$\int \sigma(y)\nu\cdot\phi_i\,dS@f$ in consistent nodal form
 * (minus the fault reaction), whose tangential part is the sensitivity of
 * the observable to tangential slip (Volterra reciprocity:
 * @f$\int w\cdot u\,dS = \int_\Sigma \hat s\cdot\sigma(y)\nu\,dS@f$).
 *
 * The map drives the problem: after a call the problem holds that call's
 * slip and displacement (zero slip after MultTranspose()). Each call is one
 * solve with the problem's solver (its preconditioner is reused, the
 * operator is assembled once).
 */
class FaultToSurfaceMap : public mfem::Operator {
 public:
  /**
   * @param problem The fault problem; not owned, must outlive the map.
   * @param fault_fes Shadow space of the displacement space on a boundary
   * SubMesh of the fault; not owned.
   * @param surface_fes Shadow space on a boundary SubMesh of the
   * observation surface; not owned.
   */
  FaultToSurfaceMap(LinearQuasiStaticClampedFaultProblem& problem,
                    const mfem::FiniteElementSpace& fault_fes,
                    const mfem::FiniteElementSpace& surface_fes);

  /** @brief Surface displacement (true dofs) of the slip @p slip (fault
   * true dofs). */
  void Mult(const mfem::Vector& slip, mfem::Vector& u) const override;

  /** @brief Fault traction (dual, fault true dofs) of the surface load
   * @p w (dual, surface true dofs). */
  void MultTranspose(const mfem::Vector& w, mfem::Vector& r) const override;

  /** @brief Whether the last application's solve converged. */
  bool Converged() const { return converged_; }

 private:
  LinearQuasiStaticClampedFaultProblem* problem_;
  SubMeshDofInjection fault_injection_, surface_injection_;
  std::unique_ptr<mfem::Operator> pi_fault_, pi_surface_;
  mutable mfem::Vector parent_true_, parent_l_;
  mutable bool converged_ = true;
};

}  // namespace AdGIA
