/**
 * @file submesh.hpp
 * @brief Coupling between a mesh and its (Par)SubMesh: the signed dof
 * injection SubMeshDofInjection and the mixed bilinear forms
 * SubMeshMixedBilinearForm / ParSubMeshMixedBilinearForm whose trial and
 * test spaces live on the two meshes. Background: doc/submesh_coupling.md.
 */

#pragma once

#include <memory>

#include "mfem.hpp"

namespace AdGIA {

/**
 * @brief Signed injection of the vdofs of a space on a (Par)SubMesh into the
 * vdofs of the corresponding space on the parent mesh.
 *
 * Built from mfem::SubMeshUtils::BuildVdofToVdofMap; valid for both
 * SubMesh::From::Domain and SubMesh::From::Boundary submeshes, in serial and
 * (when the spaces are ParFiniteElementSpaces on a ParSubMesh) in parallel.
 *
 * The sub space must be the *shadow* of the parent space: same
 * FiniteElementCollection object, vdim and ordering — use MakeShadowSpace().
 * Restricted to conforming meshes and H1/L2 collections of fixed order.
 *
 * As an mfem::Operator this acts between L-vectors (local vdofs):
 * Mult() is the prolongation-by-zero sub → parent, MultTranspose() the exact
 * restriction parent → sub. NewTrueDofMatrix() provides the corresponding
 * true-dof injection Π for parallel block operators.
 */
class SubMeshDofInjection : public mfem::Operator {
 public:
  /**
   * @brief Build the injection for an existing shadow space.
   *
   * @param sub_fes Space on the SubMesh/ParSubMesh; must share the parent
   * space's FiniteElementCollection object, vdim and ordering.
   * @param parent_fes Space on the exact parent mesh of the submesh.
   */
  SubMeshDofInjection(const mfem::FiniteElementSpace& sub_fes,
                      const mfem::FiniteElementSpace& parent_fes);

  /**
   * @brief Construct the shadow of @p parent_fes on @p submesh (the space this
   * class expects as its sub space). The FiniteElementCollection object is
   * shared with the parent space, which must therefore outlive the result.
   */
  static std::unique_ptr<mfem::FiniteElementSpace> MakeShadowSpace(
      const mfem::FiniteElementSpace& parent_fes, mfem::SubMesh& submesh);

#ifdef MFEM_USE_MPI
  /** @brief Parallel overload of MakeShadowSpace. */
  static std::unique_ptr<mfem::ParFiniteElementSpace> MakeShadowSpace(
      const mfem::ParFiniteElementSpace& parent_fes, mfem::ParSubMesh& submesh);
#endif

  /** @brief Number of sub vdofs ( = Width()). */
  int SubVSize() const { return width; }

  /** @brief Number of parent vdofs ( = Height()). */
  int ParentVSize() const { return height; }

  /** @brief Parent vdof paired with each sub vdof (decoded, always >= 0). */
  const mfem::Array<int>& ParentVDofs() const { return parent_vdof_; }

  /** @brief Sign (+1/-1) relating each sub vdof to its parent vdof. */
  const mfem::Array<mfem::real_t>& Signs() const { return sign_; }

  /** @brief y = P x: scatter sub vdofs into parent vdofs, zero elsewhere. */
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

  /** @brief y = P^T x: exact restriction of parent vdofs to sub vdofs. */
  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override;

  /**
   * @brief P as an explicit sparse matrix (ParentVSize() x SubVSize(),
   * one +-1 entry per column, at most one per row).
   */
  std::unique_ptr<mfem::SparseMatrix> NewSparseMatrix() const;

  /**
   * @brief Given a finalized M with SubVSize() rows, return P M: row i of M
   * moved to row ParentVDofs()[i] and scaled by Signs()[i]; parent rows
   * outside the submesh are empty. O(nnz), no sparse product.
   */
  std::unique_ptr<mfem::SparseMatrix> RemapRows(
      const mfem::SparseMatrix& M) const;

  /**
   * @brief Given a finalized M with SubVSize() columns, return M P^T: column
   * indices remapped to parent vdofs and entries scaled by the signs.
   * O(nnz), no sparse product.
   */
  std::unique_ptr<mfem::SparseMatrix> RemapColumns(
      const mfem::SparseMatrix& M) const;

  /** @brief The space on the SubMesh. */
  const mfem::FiniteElementSpace& SubSpace() const { return *sub_fes_; }

  /** @brief The space on the parent mesh. */
  const mfem::FiniteElementSpace& ParentSpace() const { return *parent_fes_; }

#ifdef MFEM_USE_MPI
  /**
   * @brief The true-dof injection Pi (parent true dofs x sub true dofs) as a
   * HypreParMatrix. Requires both spaces to be ParFiniteElementSpaces.
   *
   * Pi is a boolean (+-1) injection: one entry per column, at most one per
   * row, so Pi^T Pi = I and Pi^T is at once the exact primal restriction
   * parent -> sub and the dual prolongation.
   *
   * Built row-by-row over *owned* sub true dofs through
   * ParFiniteElementSpace::GetGlobalTDofNumber. (Note: the seemingly simpler
   * R_parent * P_loc * P_sub product is wrong when a shared parent dof on the
   * submesh boundary is owned by a rank whose local elements there all lie
   * outside the submesh: that rank's P_loc row is empty and the entry is
   * silently lost. The owner of a *sub* true dof always has the submesh
   * element, so this construction has no such case.)
   */
  std::unique_ptr<mfem::HypreParMatrix> NewTrueDofMatrix() const;
#endif

 private:
  const mfem::FiniteElementSpace* sub_fes_;
  const mfem::FiniteElementSpace* parent_fes_;
  mfem::Array<int> parent_vdof_;
  mfem::Array<mfem::real_t> sign_;
};

/**
 * @brief The signed pairing @f$J = \Pi_a^T \Pi_b@f$ of the vdofs of two
 * spaces on SubMeshes of one parent: @f$J_{ij} = s_i s_j@f$ where the
 * @f$i@f$-th vdof of @p a and the @f$j@f$-th vdof of @p b map to the same
 * parent vdof (with signs @f$s@f$ from the injections), zero elsewhere.
 *
 * For two domain SubMeshes meeting along an interface the shared parent
 * dofs are exactly the interface dofs, so @f$J@f$ identifies the two
 * spaces' traces there nodally: for GridFunctions @f$u_a, u_b@f$ of equal
 * trace on the interface, @f$(J u_b)_i = (u_a)_i@f$ on the interface dofs
 * of @p a. This is the pairing a sliding fluid–solid interface is built
 * from (doc/slip_interface.tex, "Discretisation of the slipping
 * interface"): with @f$B@f$ an interface bilinear form
 * assembled on @p a's boundary elements, the same form on @p b's trace is
 * @f$J^T\!B J@f$ and the cross terms are @f$B J@f$ — one assembly serves
 * all four blocks, and the orientation bookkeeping is the injections'.
 *
 * Both injections must target the same parent space. @f$J J^T@f$ is the
 * identity on the shared dofs of @p a (the pairing is one-to-one).
 */
std::unique_ptr<mfem::SparseMatrix> NewSubMeshPairingMatrix(
    const SubMeshDofInjection& a, const SubMeshDofInjection& b);

#ifdef MFEM_USE_MPI
/**
 * @brief The true-dof pairing @f$J = \Pi_a^T \Pi_b@f$ as a HypreParMatrix
 * (a true dofs x b true dofs), from the two injections' NewTrueDofMatrix().
 * The product is formed by hypre, so the two sides of a shared parent dof
 * may live on different ranks: the pairing works across partition
 * boundaries.
 */
std::unique_ptr<mfem::HypreParMatrix> NewSubMeshPairingTrueDofMatrix(
    const SubMeshDofInjection& a, const SubMeshDofInjection& b);
#endif

namespace detail {

/**
 * @brief The shadow space and injection shared by SubMeshMixedBilinearForm
 * and ParSubMeshMixedBilinearForm; built from the trial/test pair.
 */
struct SubMeshFormSetup {
  SubMeshFormSetup(mfem::FiniteElementSpace* trial_fes,
                   mfem::FiniteElementSpace* test_fes);

  std::unique_ptr<mfem::FiniteElementSpace> shadow;
  std::unique_ptr<SubMeshDofInjection> injection;
  bool parent_is_trial;
};

/**
 * @brief Assemble @p form's integrators on the submesh (via a helper
 * MixedBilinearForm that borrows them) and re-index the parent side.
 * Returns the finalized matrix in the real spaces' vdof numbering.
 */
std::unique_ptr<mfem::SparseMatrix> AssembleOnSubMesh(
    mfem::MixedBilinearForm& form, const SubMeshFormSetup& setup,
    int skip_zeros);

}  // namespace detail

/**
 * @brief A MixedBilinearForm whose trial and test spaces live on a mesh and
 * on a SubMesh of that mesh (either way round).
 *
 * All integrals are taken over the SubMesh: its elements, boundary elements
 * or boundary faces. Every integrator type accepted by MixedBilinearForm is
 * supported except interior-face integrators. Attribute markers refer to the
 * SubMesh's attributes: domain attributes are inherited from the parent,
 * boundary attributes are inherited where the parent had a boundary element
 * and equal max(parent bdr attributes) + 1 on the cut. As for
 * MixedBilinearForm, a marker must be sized to the SubMesh's
 * attributes.Max() / bdr_attributes.Max().
 *
 * Assemble() builds a plain MixedBilinearForm on the SubMesh between the
 * SubMesh-side space and a shadow of the parent-side space (see
 * SubMeshDofInjection::MakeShadowSpace), assembles it with MFEM's own code,
 * and re-indexes the parent side through the SubMeshDofInjection. The result
 * is a matrix in the two real spaces' vdof numbering, so everything else
 * (SpMat, Mult, EliminateTrialDofs, FormRectangularSystemMatrix, ...) is
 * inherited unchanged.
 *
 * @note Assemble() hides the non-virtual MixedBilinearForm::Assemble (as
 * mfem::DiscreteLinearOperator does): call it through this type. It replaces
 * any previously assembled matrix rather than adding to it. Only
 * AssemblyLevel::LEGACY is supported.
 */
class SubMeshMixedBilinearForm : public mfem::MixedBilinearForm {
 public:
  /**
   * @param trial_fes Trial space, on the parent mesh or on the SubMesh.
   * @param test_fes Test space, on the other of the two meshes.
   */
  SubMeshMixedBilinearForm(mfem::FiniteElementSpace* trial_fes,
                           mfem::FiniteElementSpace* test_fes);

  /** @brief Assemble on the SubMesh and re-index; the result is finalized. */
  void Assemble(int skip_zeros = 1);

  /** @brief The injection from the shadow space into the parent-side space. */
  const SubMeshDofInjection& Injection() const { return *setup_.injection; }

  /** @brief The shadow of the parent-side space on the SubMesh. */
  const mfem::FiniteElementSpace& ShadowSpace() const { return *setup_.shadow; }

  /** @brief True if the trial space is the one on the parent mesh. */
  bool ParentIsTrial() const { return setup_.parent_is_trial; }

 private:
  detail::SubMeshFormSetup setup_;
};

#ifdef MFEM_USE_MPI
/**
 * @brief Parallel version of SubMeshMixedBilinearForm, for
 * ParFiniteElementSpaces on a ParMesh and a ParSubMesh of it.
 *
 * Assemble() produces the local (L-vector) matrix in the real spaces'
 * numbering exactly as in serial; the inherited ParallelAssemble() and
 * FormRectangularSystemMatrix() then apply the two spaces' own
 * prolongations, so no parallel-specific assembly is needed. Ranks holding
 * no submesh elements contribute an empty local matrix.
 */
class ParSubMeshMixedBilinearForm : public mfem::ParMixedBilinearForm {
 public:
  ParSubMeshMixedBilinearForm(mfem::ParFiniteElementSpace* trial_fes,
                              mfem::ParFiniteElementSpace* test_fes);

  /** @brief Assemble the local matrix on the ParSubMesh and re-index. */
  void Assemble(int skip_zeros = 1);

  const SubMeshDofInjection& Injection() const { return *setup_.injection; }

  const mfem::ParFiniteElementSpace& ShadowSpace() const {
    return static_cast<const mfem::ParFiniteElementSpace&>(*setup_.shadow);
  }

  bool ParentIsTrial() const { return setup_.parent_is_trial; }

 private:
  detail::SubMeshFormSetup setup_;
};
#endif

}  // namespace AdGIA
