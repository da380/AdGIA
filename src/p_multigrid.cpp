#include "AdGIA/p_multigrid.hpp"

#include <algorithm>
#include <cmath>

#include "AdGIA/detail/fem_factory.hpp"

namespace AdGIA {

using namespace mfem;

namespace {

// The collection of a level: H1 of the given order with the basis of the
// space the multigrid is for.
std::unique_ptr<FiniteElementCollection> LevelCollection(
    const FiniteElementSpace& fes, int order) {
  const auto* h1 = dynamic_cast<const H1_FECollection*>(fes.FEColl());
  MFEM_VERIFY(h1 != nullptr, "PMultigridPreconditioner: an H1 space.");
  return std::make_unique<H1_FECollection>(order, fes.GetMesh()->Dimension(),
                                           h1->GetBasisType());
}

int UniformOrder(const FiniteElementSpace& fes) {
  MFEM_VERIFY(!fes.IsVariableOrder(),
              "PMultigridPreconditioner: a space of one order.");
  return fes.GetMaxElementOrder();
}

}  // namespace

std::vector<int> PMultigridOrders(int order, int coarse_order) {
  MFEM_VERIFY(coarse_order >= 1, "PMultigridOrders: coarse order >= 1.");
  auto orders = std::vector<int>{order};
  if (order > coarse_order) {
    for (auto o = (order + 1) / 2; o > coarse_order; o = (o + 1) / 2) {
      orders.push_back(o);
    }
    orders.push_back(coarse_order);
  }
  std::reverse(orders.begin(), orders.end());
  return orders;
}

PMultigridPreconditioner::PMultigridPreconditioner(
    FiniteElementSpace& fes, const SparseMatrix& A, const Array<int>& ess_dofs,
    Options options, CoarseSolverFactory coarse)
    : Solver(A.Height()) {
  const auto order = UniformOrder(fes);
  MFEM_VERIFY(order > options.coarse_order,
              "PMultigridPreconditioner: the space's order must be above "
              "the coarse order.");
  MFEM_VERIFY(A.Height() == fes.GetVSize() && A.Width() == fes.GetVSize(),
              "PMultigridPreconditioner: the matrix must be on the space's "
              "vdofs.");
  orders_ = PMultigridOrders(order, options.coarse_order);
  const auto levels = static_cast<int>(orders_.size());
  const auto vdim = fes.GetVDim();
  for (auto l = 0; l + 1 < levels; l++) {
    collections_.push_back(LevelCollection(fes, orders_[l]));
    spaces_.push_back(std::make_unique<FiniteElementSpace>(
        fes.GetMesh(), collections_.back().get(), vdim, fes.GetOrdering()));
  }
  matrices_.assign(levels, nullptr);
  matrices_[levels - 1] = const_cast<SparseMatrix*>(&A);
  owned_matrices_.resize(levels - 1);
  prolongations_.resize(levels - 1);

  auto fine_ess = Array<int>(ess_dofs);
  for (auto l = levels - 2; l >= 0; l--) {
    auto* fine = l + 2 == levels ? &fes : spaces_[l + 1].get();
    auto* coarse_fes = spaces_[l].get();
    auto interpolation = DiscreteLinearOperator(coarse_fes, fine);
    interpolation.AddDomainInterpolator(new IdentityInterpolator(vdim));
    interpolation.Assemble();
    interpolation.Finalize();
    auto P = std::unique_ptr<SparseMatrix>(interpolation.LoseMat());

    // The coarse dofs that reach an essential dof are essential.
    auto coarse_marker = Array<int>(P->Width());
    coarse_marker = 0;
    for (const auto i : fine_ess) {
      const auto* cols = P->GetRowColumns(i);
      const auto* vals = P->GetRowEntries(i);
      for (auto k = 0; k < P->RowSize(i); k++) {
        if (std::abs(vals[k]) > 1e-12) {
          coarse_marker[cols[k]] = 1;
        }
      }
      P->EliminateRow(i);
    }
    P->EliminateCols(coarse_marker);
    const auto& fine_matrix = static_cast<const SparseMatrix&>(*matrices_[l + 1]);
    auto Ac = std::unique_ptr<SparseMatrix>(mfem::RAP(*P, fine_matrix, *P));
    fine_ess.SetSize(0);
    for (auto j = 0; j < coarse_marker.Size(); j++) {
      if (coarse_marker[j]) {
        Ac->EliminateRowCol(j, Operator::DIAG_ONE);
        fine_ess.Append(j);
      }
    }
    matrices_[l] = Ac.get();
    owned_matrices_[l] = std::move(Ac);
    prolongations_[l] = std::move(P);
  }

  smoothers_.resize(levels);
  auto& coarse_matrix = static_cast<SparseMatrix&>(*matrices_[0]);
  if (coarse) {
    smoothers_[0] = coarse(*spaces_[0], coarse_matrix);
  } else {
    smoothers_[0] = std::make_unique<GSSmoother>(coarse_matrix, 0,
                                                 options.coarse_sweeps);
  }
  for (auto l = 1; l < levels; l++) {
    smoothers_[l] = std::make_unique<GSSmoother>(
        static_cast<const SparseMatrix&>(*matrices_[l]), 0, 1);
  }
  MakeCycle();
}

#ifdef MFEM_USE_MPI
PMultigridPreconditioner::PMultigridPreconditioner(
    ParFiniteElementSpace& fes, HypreParMatrix& A, const Array<int>& ess_tdofs,
    Options options, CoarseSolverFactory coarse)
    : Solver(A.Height()) {
  const auto order = UniformOrder(fes);
  MFEM_VERIFY(order > options.coarse_order,
              "PMultigridPreconditioner: the space's order must be above "
              "the coarse order.");
  MFEM_VERIFY(A.Height() == fes.GetTrueVSize() &&
                  A.Width() == fes.GetTrueVSize(),
              "PMultigridPreconditioner: the matrix must be on the space's "
              "true dofs.");
  orders_ = PMultigridOrders(order, options.coarse_order);
  const auto levels = static_cast<int>(orders_.size());
  const auto vdim = fes.GetVDim();
  auto level_space = std::vector<ParFiniteElementSpace*>(levels, nullptr);
  level_space[levels - 1] = &fes;
  for (auto l = 0; l + 1 < levels; l++) {
    collections_.push_back(LevelCollection(fes, orders_[l]));
    auto space = std::make_unique<ParFiniteElementSpace>(
        fes.GetParMesh(), collections_.back().get(), vdim, fes.GetOrdering());
    level_space[l] = space.get();
    spaces_.push_back(std::move(space));
  }
  matrices_.assign(levels, nullptr);
  matrices_[levels - 1] = &A;
  owned_matrices_.resize(levels - 1);
  prolongations_.resize(levels - 1);

  long long local_ess = ess_tdofs.Size(), global_ess = 0;
  MPI_Allreduce(&local_ess, &global_ess, 1, MPI_LONG_LONG, MPI_SUM,
                fes.GetComm());
  auto fine_ess = Array<int>(ess_tdofs);
  for (auto l = levels - 2; l >= 0; l--) {
    auto interpolation =
        ParDiscreteLinearOperator(level_space[l], level_space[l + 1]);
    interpolation.AddDomainInterpolator(new IdentityInterpolator(vdim));
    interpolation.Assemble();
    interpolation.Finalize();
    auto P = std::unique_ptr<HypreParMatrix>(interpolation.ParallelAssemble());
    auto& fine_matrix = static_cast<HypreParMatrix&>(*matrices_[l + 1]);

    auto coarse_ess = Array<int>();
    if (global_ess > 0) {
      // The coarse dofs that reach an essential dof are essential: those
      // with a nonzero in |P|^T applied to the essential dofs' indicator.
      auto Pabs = HypreParMatrix(*P);
      for (auto* part : {hypre_ParCSRMatrixDiag(
                             static_cast<hypre_ParCSRMatrix*>(Pabs)),
                         hypre_ParCSRMatrixOffd(
                             static_cast<hypre_ParCSRMatrix*>(Pabs))}) {
        auto* data = hypre_CSRMatrixData(part);
        const auto nnz = hypre_CSRMatrixNumNonzeros(part);
        for (HYPRE_Int k = 0; k < nnz; k++) {
          data[k] = std::abs(data[k]);
        }
      }
      auto indicator = Vector(P->Height()), reached = Vector(P->Width());
      indicator = 0.0;
      for (const auto i : fine_ess) {
        indicator[i] = 1.0;
      }
      Pabs.MultTranspose(indicator, reached);
      for (auto j = 0; j < reached.Size(); j++) {
        if (reached[j] > 1e-12) {
          coarse_ess.Append(j);
        }
      }
      P->EliminateRows(fine_ess);
      delete P->EliminateCols(coarse_ess);
    }
    auto Ac = std::unique_ptr<HypreParMatrix>(mfem::RAP(&fine_matrix, P.get()));
    if (global_ess > 0) {
      Ac->EliminateBC(coarse_ess, Operator::DIAG_ONE);
    }
    coarse_ess.Copy(fine_ess);
    matrices_[l] = Ac.get();
    owned_matrices_[l] = std::move(Ac);
    prolongations_[l] = std::move(P);
  }

  smoothers_.resize(levels);
  auto& coarse_matrix = static_cast<HypreParMatrix&>(*matrices_[0]);
  if (coarse) {
    smoothers_[0] = coarse(*level_space[0], coarse_matrix);
  } else {
    auto amg = std::make_unique<HypreBoomerAMG>(coarse_matrix);
    detail::SetSystemsAMGOptions(*amg, *level_space[0]);
    detail::SetDeviceAMGOptions(*amg);
    amg->SetPrintLevel(0);
    smoothers_[0] = std::move(amg);
  }
  for (auto l = 1; l < levels; l++) {
    auto smoother = std::make_unique<HypreSmoother>(
        static_cast<HypreParMatrix&>(*matrices_[l]), HypreSmoother::Chebyshev,
        1, 1.0, 1.0, options.smoothing_order, 0.3);
    smoother->SetOperatorSymmetry(true);
    smoothers_[l] = std::move(smoother);
  }
  MakeCycle();
}
#endif

PMultigridPreconditioner::~PMultigridPreconditioner() = default;

void PMultigridPreconditioner::MakeCycle() {
  const auto levels = static_cast<int>(orders_.size());
  auto operators = Array<Operator*>(levels);
  auto smoothers = Array<Solver*>(levels);
  auto prolongations = Array<Operator*>(levels - 1);
  auto own_levels = Array<bool>(levels);
  auto own_prolongations = Array<bool>(levels - 1);
  own_levels = false;
  own_prolongations = false;
  for (auto l = 0; l < levels; l++) {
    operators[l] = matrices_[l];
    smoothers[l] = smoothers_[l].get();
  }
  for (auto l = 0; l + 1 < levels; l++) {
    prolongations[l] = prolongations_[l].get();
  }
  cycle_ = std::make_unique<Multigrid>(operators, smoothers, prolongations,
                                       own_levels, own_levels,
                                       own_prolongations);
}

void PMultigridPreconditioner::SetOperator(const Operator&) {
  MFEM_ABORT("PMultigridPreconditioner: the matrix is given at "
             "construction; make a new preconditioner for a new matrix.");
}

void PMultigridPreconditioner::Mult(const Vector& x, Vector& y) const {
  cycle_->Mult(x, y);
}

// ---------------------------------------------------------------------------
// MatrixFreePMultigrid

// mfem's geometric multigrid with the levels added here.
class MatrixFreePMultigrid::Cycle : public GeometricMultigrid {
 public:
  Cycle(const FiniteElementSpaceHierarchy& hierarchy, const Array<int>& ess_bdr)
      : GeometricMultigrid(hierarchy, ess_bdr) {}

  const Array<int>& EssentialTrueDofs(int level) const {
    return *essentialTrueDofs[level];
  }

  // A level's form, assembled; owned by the cycle.
  BilinearForm& AddForm(FiniteElementSpace& fes, const Integrators& integrators,
                        bool partial) {
    BilinearForm* form = nullptr;
#ifdef MFEM_USE_MPI
    if (auto* pfes = dynamic_cast<ParFiniteElementSpace*>(&fes)) {
      form = new ParBilinearForm(pfes);
    }
#endif
    if (!form) {
      form = new BilinearForm(&fes);
    }
    if (partial) {
      form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
    }
    integrators(*form);
    form->Assemble();
    bfs.Append(form);
    return *form;
  }

  using GeometricMultigrid::AddLevel;
};

MatrixFreePMultigrid::MatrixFreePMultigrid(FiniteElementSpace& fes,
                                           const Array<int>& ess_bdr,
                                           Integrators integrators,
                                           Options options,
                                           CoarseSolverFactory coarse)
    : Solver(fes.GetTrueVSize()) {
  const auto order = UniformOrder(fes);
  MFEM_VERIFY(order > options.coarse_order,
              "MatrixFreePMultigrid: the space's order must be above the "
              "coarse order.");
  MFEM_VERIFY(fes.Conforming(),
              "MatrixFreePMultigrid: a conforming space.");
  orders_ = PMultigridOrders(order, options.coarse_order);
  const auto levels = static_cast<int>(orders_.size());
  const auto vdim = fes.GetVDim();
  const auto ordering = fes.GetOrdering();
  for (auto l = 0; l < levels; l++) {
    collections_.push_back(LevelCollection(fes, orders_[l]));
  }

#ifdef MFEM_USE_MPI
  auto* pfes = dynamic_cast<ParFiniteElementSpace*>(&fes);
  if (pfes) {
    auto* pmesh = pfes->GetParMesh();
    auto* coarse_space = new ParFiniteElementSpace(pmesh, collections_[0].get(),
                                                   vdim, ordering);
    hierarchy_ = std::make_unique<ParFiniteElementSpaceHierarchy>(
        pmesh, coarse_space, false, true);
  } else
#endif
  {
    auto* mesh = fes.GetMesh();
    auto* coarse_space =
        new FiniteElementSpace(mesh, collections_[0].get(), vdim, ordering);
    hierarchy_ = std::make_unique<FiniteElementSpaceHierarchy>(
        mesh, coarse_space, false, true);
  }
  for (auto l = 1; l < levels; l++) {
    hierarchy_->AddOrderRefinedLevel(collections_[l].get(), vdim, ordering);
  }
  MFEM_VERIFY(hierarchy_->GetFinestFESpace().GetTrueVSize() == fes.GetTrueVSize(),
              "MatrixFreePMultigrid: the finest level is not the space.");

  cycle_ = std::make_unique<Cycle>(*hierarchy_, ess_bdr);

  // The coarsest level: a matrix and its solver.
  {
    auto& space = hierarchy_->GetFESpaceAtLevel(0);
    auto& form = cycle_->AddForm(space, integrators, false);
    const auto& ess = cycle_->EssentialTrueDofs(0);
#ifdef MFEM_USE_MPI
    if (pfes) {
      auto A = std::make_unique<HypreParMatrix>();
      form.FormSystemMatrix(ess, *A);
      if (coarse) {
        coarse_solver_ = coarse(space, *A);
      } else {
        auto amg = std::make_unique<HypreBoomerAMG>(*A);
        detail::SetSystemsAMGOptions(*amg, space);
        detail::SetDeviceAMGOptions(*amg);
        amg->SetPrintLevel(0);
        coarse_solver_ = std::move(amg);
      }
      coarse_matrix_ = std::move(A);
    } else
#endif
    {
      auto A = std::make_unique<SparseMatrix>();
      form.FormSystemMatrix(ess, *A);
      if (coarse) {
        coarse_solver_ = coarse(space, *A);
      } else {
        coarse_solver_ =
            std::make_unique<GSSmoother>(*A, 0, options.coarse_sweeps);
      }
      coarse_matrix_ = std::move(A);
    }
    cycle_->AddLevel(coarse_matrix_.get(), coarse_solver_.get(), false, false);
  }

  // The levels above: the form with no matrix, Chebyshev about its diagonal.
  for (auto l = 1; l < levels; l++) {
    auto& space = hierarchy_->GetFESpaceAtLevel(l);
    auto& form = cycle_->AddForm(space, integrators, true);
    const auto& ess = cycle_->EssentialTrueDofs(l);
    OperatorPtr op;
    op.SetType(Operator::ANY_TYPE);
    form.FormSystemMatrix(ess, op);
    op.SetOperatorOwner(false);
    Vector diag(space.GetTrueVSize());
    diag.UseDevice(true);
    form.AssembleDiagonal(diag);
    Solver* smoother = nullptr;
#ifdef MFEM_USE_MPI
    if (pfes) {
      smoother = new OperatorChebyshevSmoother(*op, diag, ess,
                                               options.smoothing_order,
                                               pfes->GetComm());
    }
#endif
    if (!smoother) {
      smoother = new OperatorChebyshevSmoother(*op, diag, ess,
                                               options.smoothing_order);
    }
    cycle_->AddLevel(op.Ptr(), smoother, true, true);
  }
}

MatrixFreePMultigrid::~MatrixFreePMultigrid() = default;

void MatrixFreePMultigrid::SetOperator(const Operator&) {
  MFEM_ABORT("MatrixFreePMultigrid: the forms are made at construction; "
             "make a new preconditioner for a new form.");
}

void MatrixFreePMultigrid::Mult(const Vector& x, Vector& y) const {
  cycle_->Mult(x, y);
}

const Operator& MatrixFreePMultigrid::FineOperator() const {
  return *cycle_->GetOperatorAtFinestLevel();
}

const Array<int>& MatrixFreePMultigrid::FineEssentialTrueDofs() const {
  return cycle_->EssentialTrueDofs(static_cast<int>(orders_.size()) - 1);
}

}  // namespace AdGIA
