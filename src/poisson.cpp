#include "AdGIA/poisson.hpp"

#include <cmath>

namespace AdGIA {

namespace {

// The element matrices of the three operators share one form: the pairing of
// the shape functions with the harmonics Y_i of a SurfaceHarmonics basis
// about x0, each weighted by a function g_l(r) of its degree and of the
// distance from x0,
//
//   elmat(j, i) = int shape_j g_{l_i}(r) Y_i(x_hat) w,
//
// with radial(r, g) filling g[0..L] and w an optional coefficient.
template <class Radial>
void AssembleHarmonicElementMatrix(const SurfaceHarmonics& basis,
                                   const mfem::Vector& x0,
                                   const mfem::FiniteElement& fe,
                                   mfem::ElementTransformation& Trans,
                                   Radial radial, mfem::Coefficient* weight,
                                   mfem::DenseMatrix& elmat) {
  using namespace mfem;
  const auto dof = fe.GetDof();
  const auto n = basis.Size();

  Vector shape(dof), x(x0.Size()), Y(n), c(n), g(basis.MaxDegree() + 1);
  elmat.SetSize(dof, n);
  elmat = 0.0;

  const auto intorder = fe.GetOrder() + Trans.OrderW();
  const auto& ir = IntRules.Get(fe.GetGeomType(), intorder);

  for (auto j = 0; j < ir.GetNPoints(); j++) {
    const auto& ip = ir.IntPoint(j);
    Trans.SetIntPoint(&ip);
    Trans.Transform(ip, x);
    x -= x0;

    basis.Eval(x, Y);
    radial(x.Norml2(), g);
    for (auto i = 0; i < n; i++) {
      c(i) = g(basis.Degree(i)) * Y(i);
    }

    fe.CalcShape(ip, shape);
    auto w = Trans.Weight() * ip.weight;
    if (weight) {
      w *= weight->Eval(Trans, ip);
    }
    AddMult_a_VWt(w, shape, c, elmat);
  }
}

// The vector counterpart, pairing the components of a vector field with the
// gradients of the solid harmonics up to a degree-dependent factor,
//
//   elmat(a dof + j, i) = int shape_j g_{l_i}(r)
//                             (l_i Y_i x_hat + grad_1 Y_i)_a w,
//
// so that g_l = r^{l-1} gives grad(r^l Y_i).
template <class Radial>
void AssembleHarmonicGradientElementMatrix(
    const SurfaceHarmonics& basis, const mfem::Vector& x0,
    const mfem::FiniteElement& fe, mfem::ElementTransformation& Trans,
    Radial radial, mfem::Coefficient* weight, mfem::DenseMatrix& elmat) {
  using namespace mfem;
  const auto dof = fe.GetDof();
  const auto dim = x0.Size();
  const auto n = basis.Size();

  Vector shape(dof), x(dim), Y(n), c(n), g(basis.MaxDegree() + 1);
  DenseMatrix gradY, part_elmat(dof, n);
  elmat.SetSize(dim * dof, n);
  elmat = 0.0;

  const auto intorder = fe.GetOrder() + Trans.OrderW();
  const auto& ir = IntRules.Get(fe.GetGeomType(), intorder);

  for (auto j = 0; j < ir.GetNPoints(); j++) {
    const auto& ip = ir.IntPoint(j);
    Trans.SetIntPoint(&ip);
    Trans.Transform(ip, x);
    x -= x0;

    basis.EvalWithGradient(x, Y, gradY);
    const auto r = x.Norml2();
    radial(r, g);

    fe.CalcShape(ip, shape);
    auto w = Trans.Weight() * ip.weight;
    if (weight) {
      w *= weight->Eval(Trans, ip);
    }

    for (auto a = 0; a < dim; a++) {
      // At x0 the direction is the one SurfaceHarmonics takes there.
      const real_t x_hat =
          r > 0 ? x(a) / r : (a == (dim == 2 ? 0 : 2) ? 1.0 : 0.0);
      for (auto i = 0; i < n; i++) {
        const auto l = basis.Degree(i);
        c(i) = g(l) * (l * Y(i) * x_hat + gradY(a, i));
      }
      MultVWt(shape, c, part_elmat);
      elmat.AddMatrix(w, part_elmat, a * dof, 0);
    }
  }
}

}  // namespace

/*****************************************************************
******************************************************************
******************************************************************
*****************************************************************/

void PoissonDtNOperator::SetUp() {
  assert(dim_ == 2 || dim_ == 3);

#ifdef MFEM_USE_MPI
  if (parallel_) {
    auto* pmesh = pfes_->GetParMesh();

    SetBoundaryMarker(pmesh);

    auto [comm, has_bdr, root_rank] =
        SplitBoundaryCommunicator(pfes_->GetParMesh(), bdr_marker_);
    bdr_comm_ = comm;
    has_boundary_ = has_bdr;
    bdr_root_rank_ = root_rank;

  } else {
    SetBoundaryMarker(fes_->GetMesh());
  }
#else
  SetBoundaryMarker(fes_->GetMesh());
#endif

#ifndef MFEM_THREAD_SAFE
  c_.SetSize(coeff_dim_);
#endif
}

PoissonDtNOperator::PoissonDtNOperator(mfem::FiniteElementSpace* fes,
                                       int degree)
    : mfem::Operator(fes->GetVSize()),
      fes_{fes},
      dim_{fes->GetMesh()->Dimension()},
      degree_{degree},
      basis_{dim_, degree},
      coeff_dim_{basis_.Size()},
      mat_(fes->GetVSize(), coeff_dim_) {
  SetUp();
}

#ifdef MFEM_USE_MPI
PoissonDtNOperator::PoissonDtNOperator(MPI_Comm comm,
                                       mfem::ParFiniteElementSpace* fes,
                                       int degree)
    : mfem::Operator(fes->GetVSize()),
      parallel_{true},
      comm_{comm},
      pfes_{fes},
      fes_{fes},
      dim_{fes->GetMesh()->Dimension()},
      degree_{degree},
      basis_{dim_, degree},
      coeff_dim_{basis_.Size()},
      mat_(fes->GetVSize(), coeff_dim_) {
  SetUp();
}
#endif

void PoissonDtNOperator::Mult(const mfem::Vector& x, mfem::Vector& y) const {
  using namespace mfem;

#ifdef MFEM_THREAD_SAFE
  Vector c_(coeff_dim_);
#endif

  mat_.MultTranspose(x, c_);

#ifdef MFEM_USE_MPI
  if (parallel_) {
    if (has_boundary_) {
      MPI_Allreduce(MPI_IN_PLACE, c_.GetData(), coeff_dim_, MFEM_MPI_REAL_T,
                    MPI_SUM, bdr_comm_);
    }
  }
#endif

  y.SetSize(x.Size());
  mat_.Mult(c_, y);
}

void PoissonDtNOperator::HarmonicCoefficients(const mfem::Vector& x,
                                              mfem::Vector& y) const {
  using namespace mfem;

  y.SetSize(coeff_dim_);

  mat_.MultTranspose(x, y);

#ifdef MFEM_USE_MPI
  if (parallel_) {
    MPI_Allreduce(MPI_IN_PLACE, y.GetData(), coeff_dim_, MFEM_MPI_REAL_T,
                  MPI_SUM, comm_);
  }
#endif

  // Remove the weights of the DtN factorisation (see AssembleElementMatrix)
  // to leave c_i = R^{1-d} int_S x Y_i dS.
  for (auto i = 0; i < coeff_dim_; i++) {
    const auto l = basis_.Degree(i);
    if (dim_ == 2) {
      y(i) = l > 0 ? y(i) / std::sqrt(static_cast<mfem::real_t>(l)) : 0.0;
    } else {
      y(i) /= std::sqrt(bdr_radius_ * (l + 1));
    }
  }
}

void PoissonDtNOperator::Assemble() {
  using namespace mfem;
  auto* mesh = fes_->GetMesh();

  auto elmat = DenseMatrix();
  auto vdofs = Array<int>();
  auto rows = Array<int>(coeff_dim_);
  for (auto i = 0; i < coeff_dim_; i++) {
    rows[i] = i;
  }

  for (auto i = 0; i < fes_->GetNBE(); i++) {
    const auto elm_attr = mesh->GetBdrAttribute(i);
    if (bdr_marker_[elm_attr - 1] == 1) {
      fes_->GetBdrElementVDofs(i, vdofs);
      const auto* fe = fes_->GetBE(i);
      auto* Trans = fes_->GetBdrElementTransformation(i);

      AssembleElementMatrix(*fe, *Trans, elmat);

      mat_.AddSubMatrix(vdofs, rows, elmat);
    }
  }

  mat_.Finalize();
}

#ifdef MFEM_USE_MPI
mfem::RAPOperator PoissonDtNOperator::RAP() const {
  auto* P = fes_->GetProlongationMatrix();
  return mfem::RAPOperator(*P, *this, *P);
}
#endif

void PoissonDtNOperator::AssembleElementMatrix(
    const mfem::FiniteElement& fe, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  // B = C C^T with C_{ji} = int_S shape_j g_l(r) Y_i dS and, with the radius
  // taken pointwise, g_k = sqrt(k) / r (2-D), g_l = sqrt(l + 1) r^{-3/2}
  // (3-D).
  const auto dim = dim_;
  auto radial = [dim](mfem::real_t r, mfem::Vector& g) {
    const auto scale = dim == 2 ? 1 / r : 1 / (r * std::sqrt(r));
    for (auto l = 0; l < g.Size(); l++) {
      g(l) = scale * std::sqrt(static_cast<mfem::real_t>(dim == 2 ? l : l + 1));
    }
  };
  AssembleHarmonicElementMatrix(basis_, x0_, fe, Trans, radial, nullptr, elmat);
}

/*****************************************************************
******************************************************************
******************************************************************
*****************************************************************/

void PoissonMultipoleOperator::SetUp() {
  assert(tr_fes_->GetMesh() == te_fes_->GetMesh());
#ifdef MFEM_USE_MPI
  if (parallel_) {
    SetBoundaryMarker(tr_pfes_->GetParMesh());
  } else {
    SetBoundaryMarker(tr_fes_->GetMesh());
  }
#else
  SetBoundaryMarker(tr_fes_->GetMesh());
#endif

#ifndef MFEM_THREAD_SAFE
  c_.SetSize(coeff_dim_);
#endif
}

PoissonMultipoleOperator::PoissonMultipoleOperator(
    mfem::FiniteElementSpace* tr_fes, mfem::FiniteElementSpace* te_fes,
    int degree, const mfem::Array<int>& dom_marker)
    : mfem::Operator(te_fes->GetVSize(), tr_fes->GetVSize()),
      tr_fes_{tr_fes},
      te_fes_{te_fes},
      dim_{tr_fes->GetMesh()->Dimension()},
      degree_{degree},
      basis_{dim_, degree},
      coeff_dim_{basis_.Size()},
      dom_marker_{dom_marker},
      lmat_(te_fes->GetVSize(), coeff_dim_),
      rmat_(tr_fes->GetVSize(), coeff_dim_) {
  SetUp();
}

#ifdef MFEM_USE_MPI
PoissonMultipoleOperator::PoissonMultipoleOperator(
    MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
    mfem::ParFiniteElementSpace* te_fes, int degree,
    const mfem::Array<int>& dom_marker)
    : mfem::Operator(te_fes->GetVSize(), tr_fes->GetVSize()),
      parallel_{true},
      comm_{comm},
      tr_fes_{tr_fes},
      te_fes_{te_fes},
      tr_pfes_{tr_fes},
      te_pfes_{te_fes},
      dim_{tr_fes->GetMesh()->Dimension()},
      degree_{degree},
      basis_{dim_, degree},
      coeff_dim_{basis_.Size()},
      dom_marker_{dom_marker},
      lmat_(te_fes->GetVSize(), coeff_dim_),
      rmat_(tr_fes->GetVSize(), coeff_dim_) {
  SetUp();
}
#endif

void PoissonMultipoleOperator::Mult(const mfem::Vector& x,
                                    mfem::Vector& y) const {
  using namespace mfem;

#ifdef MFEM_THREAD_SAFE
  Vector c_(coeff_dim_);
#endif

  rmat_.MultTranspose(x, c_);

#ifdef MFEM_USE_MPI
  if (parallel_) {
    MPI_Allreduce(MPI_IN_PLACE, c_.GetData(), coeff_dim_, MFEM_MPI_REAL_T,
                  MPI_SUM, comm_);
  }
#endif

  y.SetSize(lmat_.Height());
  lmat_.Mult(c_, y);
}

void PoissonMultipoleOperator::MultTranspose(const mfem::Vector& x,
                                             mfem::Vector& y) const {
  using namespace mfem;

#ifdef MFEM_THREAD_SAFE
  Vector c_(coeff_dim_);
#endif

  lmat_.MultTranspose(x, c_);

#ifdef MFEM_USE_MPI
  if (parallel_) {
    MPI_Allreduce(MPI_IN_PLACE, c_.GetData(), coeff_dim_, MFEM_MPI_REAL_T,
                  MPI_SUM, comm_);
  }
#endif

  y.SetSize(rmat_.Height());
  rmat_.Mult(c_, y);
}

void PoissonMultipoleOperator::Assemble() {
  auto* mesh = tr_fes_->GetMesh();

  auto elmat = mfem::DenseMatrix();
  auto vdofs = mfem::Array<int>();
  auto cdofs = mfem::Array<int>(coeff_dim_);
  for (auto i = 0; i < coeff_dim_; i++) {
    cdofs[i] = i;
  }

  for (auto i = 0; i < te_fes_->GetNBE(); i++) {
    const auto elm_attr = mesh->GetBdrAttribute(i);
    if (bdr_marker_[elm_attr - 1] == 1) {
      te_fes_->GetBdrElementVDofs(i, vdofs);
      const auto* fe = te_fes_->GetBE(i);
      auto* Trans = te_fes_->GetBdrElementTransformation(i);

      AssembleLeftElementMatrix(*fe, *Trans, elmat);

      lmat_.AddSubMatrix(vdofs, cdofs, elmat);
    }
  }

  lmat_.Finalize();

  for (auto i = 0; i < tr_fes_->GetNE(); i++) {
    const auto elm_attr = mesh->GetAttribute(i);
    if (dom_marker_[elm_attr - 1] == 1) {
      tr_fes_->GetElementVDofs(i, vdofs);
      const auto* fe = tr_fes_->GetFE(i);
      auto* Trans = tr_fes_->GetElementTransformation(i);

      AssembleRightElementMatrix(*fe, *Trans, elmat);

      rmat_.AddSubMatrix(vdofs, cdofs, elmat);
    }
  }

  rmat_.Finalize();
}

#ifdef MFEM_USE_MPI
mfem::RAPOperator PoissonMultipoleOperator::RAP() const {
  auto* P_te = te_fes_->GetProlongationMatrix();
  auto* P_tr = tr_fes_->GetProlongationMatrix();
  return mfem::RAPOperator(*P_te, *this, *P_tr);
}
#endif

void PoissonMultipoleOperator::AssembleLeftElementMatrix(
    const mfem::FiniteElement& fe, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  // int_S shape_j Y_i dS.
  auto radial = [](mfem::real_t, mfem::Vector& g) { g = 1.0; };
  AssembleHarmonicElementMatrix(basis_, x0_, fe, Trans, radial, nullptr, elmat);
}

void PoissonMultipoleOperator::AssembleRightElementMatrix(
    const mfem::FiniteElement& fe, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  // Interior harmonics of the source scaled to the boundary radius b:
  // g_0 = 1 / b and g_k = (r/b)^k / (2 b) in 2-D,
  // g_l = (l + 1) / (2 l + 1) (r/b)^l / b^2 in 3-D.
  const auto dim = dim_;
  const auto b = bdr_radius_;
  auto radial = [dim, b](mfem::real_t r, mfem::Vector& g) {
    const auto ratio = r / b;
    auto rfac = dim == 2 ? 1 / b : 1 / (b * b);
    for (auto l = 0; l < g.Size(); l++) {
      if (dim == 2) {
        g(l) = l == 0 ? rfac : rfac / 2;
      } else {
        g(l) = rfac * (l + 1) / (2 * l + 1);
      }
      rfac *= ratio;
    }
  };
  AssembleHarmonicElementMatrix(basis_, x0_, fe, Trans, radial, nullptr, elmat);
}

/*****************************************************************
******************************************************************
******************************************************************
*****************************************************************/

void PoissonLinearisedMultipoleOperator::SetUp() {
  assert(tr_fes_->GetMesh() == te_fes_->GetMesh());
  assert(tr_fes_->GetMesh()->Dimension() == tr_fes_->GetVDim());
#ifdef MFEM_USE_MPI
  if (parallel_) {
    SetBoundaryMarker(tr_pfes_->GetParMesh());
  } else {
    SetBoundaryMarker(tr_fes_->GetMesh());
  }
#else
  SetBoundaryMarker(tr_fes_->GetMesh());
#endif

#ifndef MFEM_THREAD_SAFE
  c_.SetSize(coeff_dim_);
#endif
}

PoissonLinearisedMultipoleOperator::PoissonLinearisedMultipoleOperator(
    mfem::FiniteElementSpace* tr_fes, mfem::FiniteElementSpace* te_fes,
    mfem::Coefficient& density, int degree, const mfem::Array<int>& dom_marker)
    : mfem::Operator(te_fes->GetVSize(), tr_fes->GetVSize()),
      tr_fes_{tr_fes},
      te_fes_{te_fes},
      density_{&density},
      dim_{tr_fes->GetMesh()->Dimension()},
      degree_{degree},
      basis_{dim_, degree},
      coeff_dim_{basis_.Size()},
      dom_marker_{dom_marker},
      lmat_(te_fes->GetVSize(), coeff_dim_),
      rmat_(tr_fes->GetVSize(), coeff_dim_) {
  SetUp();
}

PoissonLinearisedMultipoleOperator::PoissonLinearisedMultipoleOperator(
    mfem::FiniteElementSpace* tr_fes, mfem::FiniteElementSpace* te_fes,
    int degree, const mfem::Array<int>& dom_marker)
    : mfem::Operator(te_fes->GetVSize(), tr_fes->GetVSize()),
      tr_fes_{tr_fes},
      te_fes_{te_fes},
      dim_{tr_fes->GetMesh()->Dimension()},
      degree_{degree},
      basis_{dim_, degree},
      coeff_dim_{basis_.Size()},
      dom_marker_{dom_marker},
      lmat_(te_fes->GetVSize(), coeff_dim_),
      rmat_(tr_fes->GetVSize(), coeff_dim_) {
  SetUp();
}

#ifdef MFEM_USE_MPI

PoissonLinearisedMultipoleOperator::PoissonLinearisedMultipoleOperator(
    MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
    mfem::ParFiniteElementSpace* te_fes, mfem::Coefficient& density, int degree,
    const mfem::Array<int>& dom_marker)
    : mfem::Operator(te_fes->GetVSize(), tr_fes->GetVSize()),
      parallel_{true},
      comm_{comm},
      tr_fes_{tr_fes},
      te_fes_{te_fes},
      tr_pfes_{tr_fes},
      te_pfes_{te_fes},
      density_{&density},
      dim_{tr_fes->GetMesh()->Dimension()},
      degree_{degree},
      basis_{dim_, degree},
      coeff_dim_{basis_.Size()},
      dom_marker_{dom_marker},
      lmat_(te_fes->GetVSize(), coeff_dim_),
      rmat_(tr_fes->GetVSize(), coeff_dim_) {
  SetUp();
}

PoissonLinearisedMultipoleOperator::PoissonLinearisedMultipoleOperator(
    MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
    mfem::ParFiniteElementSpace* te_fes, int degree,
    const mfem::Array<int>& dom_marker)
    : mfem::Operator(te_fes->GetVSize(), tr_fes->GetVSize()),
      parallel_{true},
      comm_{comm},
      tr_fes_{tr_fes},
      te_fes_{te_fes},
      tr_pfes_{tr_fes},
      te_pfes_{te_fes},
      dim_{tr_fes->GetMesh()->Dimension()},
      degree_{degree},
      basis_{dim_, degree},
      coeff_dim_{basis_.Size()},
      dom_marker_{dom_marker},
      lmat_(te_fes->GetVSize(), coeff_dim_),
      rmat_(tr_fes->GetVSize(), coeff_dim_) {
  SetUp();
}
#endif

void PoissonLinearisedMultipoleOperator::Mult(const mfem::Vector& x,
                                              mfem::Vector& y) const {
  using namespace mfem;

#ifdef MFEM_THREAD_SAFE
  Vector c_(coeff_dim_);
#endif

  rmat_.MultTranspose(x, c_);

#ifdef MFEM_USE_MPI
  if (parallel_) {
    MPI_Allreduce(MPI_IN_PLACE, c_.GetData(), coeff_dim_, MFEM_MPI_REAL_T,
                  MPI_SUM, comm_);
  }
#endif

  y.SetSize(lmat_.Height());
  lmat_.Mult(c_, y);
}

void PoissonLinearisedMultipoleOperator::MultTranspose(const mfem::Vector& x,
                                                       mfem::Vector& y) const {
  using namespace mfem;

#ifdef MFEM_THREAD_SAFE
  Vector c_(coeff_dim_);
#endif

  lmat_.MultTranspose(x, c_);

#ifdef MFEM_USE_MPI
  if (parallel_) {
    MPI_Allreduce(MPI_IN_PLACE, c_.GetData(), coeff_dim_, MFEM_MPI_REAL_T,
                  MPI_SUM, comm_);
  }
#endif

  y.SetSize(rmat_.Height());
  rmat_.Mult(c_, y);
}

void PoissonLinearisedMultipoleOperator::Assemble() {
  auto* mesh = tr_fes_->GetMesh();

  auto elmat = mfem::DenseMatrix();
  auto vdofs = mfem::Array<int>();
  auto cdofs = mfem::Array<int>(coeff_dim_);
  for (auto i = 0; i < coeff_dim_; i++) {
    cdofs[i] = i;
  }

  for (auto i = 0; i < te_fes_->GetNBE(); i++) {
    const auto elm_attr = mesh->GetBdrAttribute(i);
    if (bdr_marker_[elm_attr - 1] == 1) {
      te_fes_->GetBdrElementVDofs(i, vdofs);
      const auto* fe = te_fes_->GetBE(i);
      auto* Trans = te_fes_->GetBdrElementTransformation(i);

      AssembleLeftElementMatrix(*fe, *Trans, elmat);

      lmat_.AddSubMatrix(vdofs, cdofs, elmat);
    }
  }

  lmat_.Finalize();

  for (auto i = 0; i < tr_fes_->GetNE(); i++) {
    const auto elm_attr = mesh->GetAttribute(i);
    if (dom_marker_[elm_attr - 1] == 1) {
      tr_fes_->GetElementVDofs(i, vdofs);
      const auto* fe = tr_fes_->GetFE(i);
      auto* Trans = tr_fes_->GetElementTransformation(i);

      AssembleRightElementMatrix(*fe, *Trans, elmat);

      rmat_.AddSubMatrix(vdofs, cdofs, elmat);
    }
  }

  rmat_.Finalize();
}

#ifdef MFEM_USE_MPI
mfem::RAPOperator PoissonLinearisedMultipoleOperator::RAP() const {
  auto* P_te = te_fes_->GetProlongationMatrix();
  auto* P_tr = tr_fes_->GetProlongationMatrix();
  return mfem::RAPOperator(*P_te, *this, *P_tr);
}
#endif

void PoissonLinearisedMultipoleOperator::AssembleLeftElementMatrix(
    const mfem::FiniteElement& fe, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  // int_S shape_j Y_i dS.
  auto radial = [](mfem::real_t, mfem::Vector& g) { g = 1.0; };
  AssembleHarmonicElementMatrix(basis_, x0_, fe, Trans, radial, nullptr, elmat);
}

void PoissonLinearisedMultipoleOperator::AssembleRightElementMatrix(
    const mfem::FiniteElement& fe, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  // The gradients of the interior harmonics of the static operator,
  // grad(r^l Y_i) = r^{l-1} (l Y_i x_hat + grad_1 Y_i), so with
  // g_k = (r/b)^{k-1} / (2 b^2) in 2-D and
  // g_l = (l + 1) / (2 l + 1) (r/b)^{l-1} / b^3 in 3-D; degree zero drops out.
  const auto dim = dim_;
  const auto b = bdr_radius_;
  auto radial = [dim, b](mfem::real_t r, mfem::Vector& g) {
    const auto ratio = r / b;
    auto rfac = dim == 2 ? 1 / (2 * b * b) : 1 / (b * b * b);
    g(0) = 0.0;
    for (auto l = 1; l < g.Size(); l++) {
      g(l) = dim == 2 ? rfac : rfac * (l + 1) / (2 * l + 1);
      rfac *= ratio;
    }
  };
  AssembleHarmonicGradientElementMatrix(basis_, x0_, fe, Trans, radial,
                                        density_, elmat);
}

}  // namespace AdGIA
