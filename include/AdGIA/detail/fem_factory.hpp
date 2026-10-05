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
