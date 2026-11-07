/**
 * @file fem_factory.hpp
 * @brief Factories creating the parallel variant of a space, form or grid
 * function when the reference space is parallel; used to keep serial and
 * parallel code paths in one class.
 */

#pragma once

#include <memory>

#include "mfem.hpp"

namespace AdGIA {
namespace detail {

/// Returns true if an FE space is parallel.
inline bool IsParallel(const mfem::FiniteElementSpace& fes) {
#ifdef MFEM_USE_MPI
  return dynamic_cast<const mfem::ParFiniteElementSpace*>(&fes) != nullptr;
#else
  (void)fes;
  return false;
#endif
}

/// A space on the same mesh as @p like, parallel iff @p like is.
inline std::unique_ptr<mfem::FiniteElementSpace> MakeFESpace(
    mfem::FiniteElementSpace& like, mfem::FiniteElementCollection* fec,
    int vdim = 1, int ordering = mfem::Ordering::byNODES) {
#ifdef MFEM_USE_MPI
  if (auto* pfes = dynamic_cast<mfem::ParFiniteElementSpace*>(&like)) {
    return std::make_unique<mfem::ParFiniteElementSpace>(pfes->GetParMesh(),
                                                         fec, vdim, ordering);
  }
#endif
  return std::make_unique<mfem::FiniteElementSpace>(like.GetMesh(), fec, vdim,
                                                    ordering);
}

/// A GridFunction for a space, parallel if the space is.
inline std::unique_ptr<mfem::GridFunction> MakeGridFunction(
    mfem::FiniteElementSpace* fes) {
#ifdef MFEM_USE_MPI
  if (auto* pfes = dynamic_cast<mfem::ParFiniteElementSpace*>(fes)) {
    return std::make_unique<mfem::ParGridFunction>(pfes);
  }
#endif
  return std::make_unique<mfem::GridFunction>(fes);
}

/// A LinearForm for a space, parallel if the space is.
inline std::unique_ptr<mfem::LinearForm> MakeLinearForm(
    mfem::FiniteElementSpace* fes) {
#ifdef MFEM_USE_MPI
  if (auto* pfes = dynamic_cast<mfem::ParFiniteElementSpace*>(fes)) {
    return std::make_unique<mfem::ParLinearForm>(pfes);
  }
#endif
  return std::make_unique<mfem::LinearForm>(fes);
}

#ifdef MFEM_USE_MPI
/// Configures @p amg as unknown-based systems AMG for a matrix on the
/// vector space @p fes (no-op for a scalar space), with hypre's
/// function map taken from the space's ordering.
///
/// Nodal coarsening (HYPRE_BoomerAMGSetNodal, and so mfem's
/// SetElasticityOptions with its rigid-body interpolation) is
/// deliberately not used. hypre forms its nodal blocks from consecutive
/// unknowns (byVDIM interleaving) and ignores the function map, so on a
/// byNODES space it couples unrelated unknowns and the hierarchy breaks
/// down (interpolation weights ~1e16, iterations doubling with each
/// halving of h). Even on a byVDIM space it was mesh-dependent in 3-D
/// (Q2 slab, C = 1e6: 1524 PCG iterations at 807k unknowns, against 42
/// for the unknown-based setting here, which stays near mesh- and
/// contrast-independent at orders 1-3 in 2-D and 3-D).
inline void SetSystemsAMGOptions(mfem::HypreBoomerAMG& amg,
                                 const mfem::FiniteElementSpace& fes) {
  if (fes.GetVDim() > 1) {
    amg.SetSystemsOptions(fes.GetVDim(),
                          fes.GetOrdering() == mfem::Ordering::byNODES);
  }
}
#endif

/// A GS (serial) or AMG (parallel) preconditioner for an assembled
/// operator handle.
inline std::unique_ptr<mfem::Solver> MakePreconditioner(
    const mfem::OperatorHandle& A) {
#ifdef MFEM_USE_MPI
  if (A.Type() == mfem::Operator::Hypre_ParCSR) {
    auto amg = std::make_unique<mfem::HypreBoomerAMG>(
        *const_cast<mfem::OperatorHandle&>(A).As<mfem::HypreParMatrix>());
    amg->SetPrintLevel(0);
    return amg;
  }
#endif
  return std::make_unique<mfem::GSSmoother>(
      *const_cast<mfem::OperatorHandle&>(A).As<mfem::SparseMatrix>());
}

/// A mixed bilinear form (trial, test), parallel if the spaces are (both
/// must agree).
inline std::unique_ptr<mfem::MixedBilinearForm> MakeMixedBilinearForm(
    mfem::FiniteElementSpace* trial, mfem::FiniteElementSpace* test) {
#ifdef MFEM_USE_MPI
  auto* ptrial = dynamic_cast<mfem::ParFiniteElementSpace*>(trial);
  auto* ptest = dynamic_cast<mfem::ParFiniteElementSpace*>(test);
  MFEM_VERIFY(!ptrial == !ptest,
              "MakeMixedBilinearForm: trial and test spaces must both be "
              "serial or both parallel.");
  if (ptrial) {
    return std::make_unique<mfem::ParMixedBilinearForm>(ptrial, ptest);
  }
#endif
  return std::make_unique<mfem::MixedBilinearForm>(trial, test);
}

/// A bilinear form on @p fes, borrowing the integrators of @p borrow_from
/// when given (the returned form does not own them). When @p fes is parallel,
/// @p borrow_from must be a ParBilinearForm (it is static_cast).
inline std::unique_ptr<mfem::BilinearForm> MakeBilinearForm(
    mfem::FiniteElementSpace* fes, mfem::BilinearForm* borrow_from = nullptr) {
#ifdef MFEM_USE_MPI
  if (auto* pfes = dynamic_cast<mfem::ParFiniteElementSpace*>(fes)) {
    if (borrow_from) {
      return std::make_unique<mfem::ParBilinearForm>(
          pfes, static_cast<mfem::ParBilinearForm*>(borrow_from));
    }
    return std::make_unique<mfem::ParBilinearForm>(pfes);
  }
#endif
  if (borrow_from) {
    return std::make_unique<mfem::BilinearForm>(fes, borrow_from);
  }
  return std::make_unique<mfem::BilinearForm>(fes);
}

}  // namespace detail
}  // namespace AdGIA
