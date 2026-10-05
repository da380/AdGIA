/**
 * @file null_space.cpp
 * @brief Implementation of the rigid-mode coefficients, NullSpaceProjector,
 * AddRigidModes(), MakeRigidModeProjector(), ProjectedOperator and
 * ProjectedSolver.
 */

#include "AdGIA/null_space.hpp"

#include <cmath>

#include "AdGIA/detail/fem_factory.hpp"

namespace AdGIA {

RigidTranslation::RigidTranslation(int dimension, int component)
    : mfem::VectorCoefficient(dimension), component_{component} {
  MFEM_ASSERT(component >= 0 && component < dimension,
              "component out of range");
}

void RigidTranslation::SetComponent(int component) { component_ = component; }

void RigidTranslation::Eval(mfem::Vector &V, mfem::ElementTransformation &T,
                            const mfem::IntegrationPoint &ip) {
  V.SetSize(vdim);
  V = 0.;
  V[component_] = 1;
}

RigidRotation::RigidRotation(int dimension, int component)
    : mfem::VectorCoefficient(dimension), component_{component} {
  MFEM_ASSERT(component >= 0 && component < dimension,
              "component out of range");
  MFEM_ASSERT(dimension == 3 || component == 2,
              "In two dimensions only z-rotation defined");
#ifndef MFEM_THREAD_SAFE
  x_.SetSize(dimension);
#endif
}

void RigidRotation::SetComponent(int component) { component_ = component; }

void RigidRotation::Eval(mfem::Vector &V, mfem::ElementTransformation &T,
                         const mfem::IntegrationPoint &ip) {
  V.SetSize(vdim);
#ifdef MFEM_THREAD_SAFE
  mfem::Vector x_(vdim);
#endif
  T.Transform(ip, x_);
  if (component_ == 0) {
    V[0] = 0;
    V[1] = -x_[2];
    V[2] = x_[1];
  } else if (component_ == 1) {
    V[0] = x_[2];
    V[1] = 0;
    V[2] = -x_[0];
  } else {
    V[0] = -x_[1];
    V[1] = x_[0];
    if (vdim == 3) V[2] = 0;
  }
}

// ---------------------------------------------------------------------------

mfem::real_t NullSpaceProjector::Dot(const mfem::Vector &x,
                                     const mfem::Vector &y) const {
#ifdef MFEM_USE_MPI
  if (parallel_) {
    return mfem::InnerProduct(comm_, x, y);
  }
#endif
  return mfem::InnerProduct(x, y);
}

bool NullSpaceProjector::Add(const mfem::Vector &v, mfem::real_t drop_tol) {
  auto n = std::make_unique<mfem::Vector>(v);
  const auto norm0 = std::sqrt(Dot(*n, *n));
  if (!(norm0 > 0.0)) {
    return false;
  }
  for (const auto &b : basis_) {
    n->Add(-Dot(*n, *b), *b);
  }
  const auto norm = std::sqrt(Dot(*n, *n));
  if (norm <= drop_tol * norm0) {
    return false;
  }
  *n /= norm;
  basis_.push_back(std::move(n));
  return true;
}

void NullSpaceProjector::Project(mfem::Vector &x) const {
  for (const auto &b : basis_) {
    x.Add(-Dot(x, *b), *b);
  }
}

int AddRigidModes(NullSpaceProjector &P, mfem::FiniteElementSpace &fes,
                  Diffeomorphism *map) {
  const int dim = fes.GetVDim();
  MFEM_VERIFY(dim == 2 || dim == 3,
              "AddRigidModes: the space must have vdim 2 or 3.");
  auto gf = detail::MakeGridFunction(&fes);
  mfem::Vector t;
  int added = 0;
  auto add = [&](mfem::VectorCoefficient &c) {
    gf->ProjectCoefficient(c);
    gf->GetTrueDofs(t);
    if (P.Add(t)) {
      added++;
    }
  };
  auto add_rotation = [&](int c) {
    if (map) {
      MappedRotation rot(*map, c);
      add(rot);
    } else {
      RigidRotation rot(dim, c);
      add(rot);
    }
  };
  for (int c = 0; c < dim; c++) {
    RigidTranslation tr(dim, c);
    add(tr);
  }
  if (dim == 2) {
    add_rotation(2);
  } else {
    for (int c = 0; c < 3; c++) {
      add_rotation(c);
    }
  }
  return added;
}

std::unique_ptr<NullSpaceProjector> MakeRigidModeProjector(
    mfem::FiniteElementSpace &fes, Diffeomorphism *map) {
  std::unique_ptr<NullSpaceProjector> P;
#ifdef MFEM_USE_MPI
  if (auto *pfes = dynamic_cast<mfem::ParFiniteElementSpace *>(&fes)) {
    P = std::make_unique<NullSpaceProjector>(pfes->GetComm());
  } else
#endif
  {
    P = std::make_unique<NullSpaceProjector>();
  }
  AddRigidModes(*P, fes, map);
  return P;
}

void ProjectedOperator::Mult(const mfem::Vector &x, mfem::Vector &y) const {
  P_->Project(x, z_);
  A_->Mult(z_, y);
  P_->Project(y);
}

void ProjectedSolver::SetSolver(mfem::Solver &solver) {
  solver_ = &solver;
  height = solver_->Height();
  width = solver_->Width();
}

void ProjectedSolver::SetOperator(const mfem::Operator &op) {
  MFEM_VERIFY(solver_, "ProjectedSolver: call SetSolver() first.");
  projected_ = std::make_unique<ProjectedOperator>(op, *P_);
  solver_->SetOperator(*projected_);
  height = op.Height();
  width = op.Width();
}

void ProjectedSolver::Mult(const mfem::Vector &b, mfem::Vector &x) const {
  P_->Project(b, b_);
  solver_->iterative_mode = iterative_mode;
  if (iterative_mode) {
    P_->Project(x);
  }
  solver_->Mult(b_, x);
  P_->Project(x);
  ApplyGauge(x);
}

void ProjectedSolver::SetGauge(const mfem::Operator *M) {
  Mn_.clear();
  gauge_idx_.clear();
  if (!M) {
    return;
  }
  // Basis vectors with a nonzero M-norm take part in the gauge; the others
  // keep their Euclidean condition (x is projected before the gauge, and
  // the basis is Euclidean-orthonormal, so adding multiples of the gauged
  // vectors leaves the components along the others at zero).
  mfem::Vector Mn(M->Height());
  mfem::real_t max_norm = 0.0;
  std::vector<mfem::real_t> norms;
  for (int i = 0; i < P_->Size(); i++) {
    M->Mult(P_->Basis(i), Mn);
    norms.push_back(P_->Dot(P_->Basis(i), Mn));
    max_norm = std::max(max_norm, norms.back());
  }
  for (int i = 0; i < P_->Size(); i++) {
    if (norms[i] > 1e-12 * max_norm) {
      gauge_idx_.push_back(i);
      Mn_.emplace_back(M->Height());
      M->Mult(P_->Basis(i), Mn_.back());
    }
  }
  const int n = static_cast<int>(gauge_idx_.size());
  if (n == 0) {
    return;
  }
  mfem::DenseMatrix G(n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      G(i, j) = P_->Dot(P_->Basis(gauge_idx_[i]), Mn_[j]);
    }
  }
  mfem::DenseMatrixInverse inv(G);
  inv.GetInverseMatrix(Ginv_);
}

void ProjectedSolver::ApplyGauge(mfem::Vector &x) const {
  const int n = static_cast<int>(gauge_idx_.size());
  if (n == 0) {
    return;
  }
  // x <- x - sum_i c_i n_i with G c = (M n . x): then n_i . M x = 0.
  mfem::Vector r(n), c(n);
  for (int i = 0; i < n; i++) {
    r[i] = P_->Dot(Mn_[i], x);
  }
  Ginv_.Mult(r, c);
  for (int i = 0; i < n; i++) {
    x.Add(-c[i], P_->Basis(gauge_idx_[i]));
  }
}

}  // namespace AdGIA