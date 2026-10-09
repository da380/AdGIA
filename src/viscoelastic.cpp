/**
 * @file viscoelastic.cpp
 * @brief Implementation of ViscoelasticOperator and the exponential ODE
 * solver adaptors.
 */

#include "AdGIA/viscoelastic.hpp"

#include <algorithm>
#include <cmath>

#include "AdGIA/bilininteg.hpp"
#include "AdGIA/detail/fem_factory.hpp"
#include "AdGIA/elastic_tensor.hpp"
#include "AdGIA/relaxation_law.hpp"

namespace AdGIA {

using namespace mfem;

namespace detail {

void ExponentialTrapezoidWeights(real_t h, real_t& e, real_t& alpha,
                                 real_t& beta) {
  e = std::exp(-h);
  real_t phi1;  // (1 - e^{-h}) / h
  if (h < 1e-3) {
    phi1 =
        1.0 - h / 2.0 + h * h / 6.0 - h * h * h / 24.0 + h * h * h * h / 120.0;
  } else {
    phi1 = (1.0 - e) / h;
  }
  alpha = phi1 - e;
  beta = 1.0 - phi1;
}

}  // namespace detail

namespace {

// Block-diagonal inverse of the L2 mass matrix of a scalar space, as a
// sparse matrix (exact: the mass matrix is element-block-diagonal).
std::unique_ptr<SparseMatrix> BuildBlockMassInverse(FiniteElementSpace& sfes) {
  BilinearForm mass(&sfes);
  mass.AddDomainIntegrator(new MassIntegrator());
  auto Minv = std::make_unique<SparseMatrix>(sfes.GetVSize(), sfes.GetVSize());
  Array<int> dofs;
  DenseMatrix elmat, inv;
  for (int e = 0; e < sfes.GetNE(); e++) {
    sfes.GetElementDofs(e, dofs);
    mass.ComputeElementMatrix(e, elmat);
    DenseMatrixInverse elinv(elmat);
    elinv.GetInverseMatrix(inv);
    Minv->SetSubMatrix(dofs, dofs, inv, 0);
  }
  Minv->Finalize(0);
  return Minv;
}

// Inverse of the Frobenius metric G_{cc'} = E_c : E_{c'} of the symmetric
// basis tensors in the SymmetricComponentOrder layout (component (j, k),
// j >= k, lower triangle column-major) with the last diagonal entry dropped
// in the trace-free case. E = e_j e_k^T + e_k e_j^T off the
// diagonal; on it E = e_j e_j^T (full) or e_j e_j^T - e_{d-1} e_{d-1}^T
// (trace-free).
DenseMatrix MetricInverse(int dim, bool tracefree) {
  const int nc = tracefree ? SymmetricComponentOrder::TraceFreeCount(dim)
                           : SymmetricComponentOrder::Count(dim);
  std::vector<DenseMatrix> E;
  for (int k = 0; k < dim; k++) {
    for (int j = k; j < dim; j++) {
      if (tracefree && SymmetricComponentOrder::Offset(dim, j, k) ==
                           SymmetricComponentOrder::TraceFreeDropped(dim)) {
        continue;
      }
      DenseMatrix T(dim);
      T = 0.0;
      if (j == k) {
        T(j, j) = 1.0;
        if (tracefree) {
          T(dim - 1, dim - 1) = -1.0;
        }
      } else {
        T(j, k) = 1.0;
        T(k, j) = 1.0;
      }
      E.push_back(T);
    }
  }
  MFEM_VERIFY(static_cast<int>(E.size()) == nc, "symmetric basis size");
  DenseMatrix G(nc);
  for (int c = 0; c < nc; c++) {
    for (int cp = 0; cp < nc; cp++) {
      double s = 0.0;
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
          s += E[c](i, j) * E[cp](i, j);
        }
      }
      G(c, cp) = s;
    }
  }
  DenseMatrix Ginv(nc);
  DenseMatrixInverse inv(G);
  inv.GetInverseMatrix(Ginv);
  return Ginv;
}

// Nodal values of a coefficient at the L2 nodes of sfes.
Vector NodalValues(FiniteElementSpace& sfes, Coefficient& c) {
  GridFunction g(&sfes);
  g.ProjectCoefficient(c);
  Vector v(g);
  return v;
}

// Branch tensor C_k sampled at the L2 nodes of sfes, stored per node as the
// n_s x n_s matrix W acting on unscaled tensor components: sigma_s =
// sum_t W_st m_t, i.e. W_st = C^_st w_t / (a_s a_t) with the Mandel scales
// a and the multiplicity w_t (2 off the diagonal) of the symmetric pair.
template <class Eval>
Vector NodalTensors(FiniteElementSpace& sfes, Eval eval) {
  const int dim = sfes.GetMesh()->Dimension();
  const int ns = SymmetricTensorBasis::Size(dim);
  Vector W(sfes.GetVSize() * ns * ns);
  W = 0.0;
  std::vector<real_t> a(ns), w(ns);
  for (int s = 0; s < ns; s++) {
    int j, l;
    SymmetricTensorBasis::Component(dim, s, j, l);
    a[s] = SymmetricTensorBasis::Scale(j, l);
    w[s] = j == l ? 1.0 : 2.0;
  }
  DenseMatrix Cm(ns);
  Array<int> dofs;
  for (int e = 0; e < sfes.GetNE(); e++) {
    const FiniteElement* fe = sfes.GetFE(e);
    ElementTransformation* T = sfes.GetElementTransformation(e);
    const IntegrationRule& nodes = fe->GetNodes();
    sfes.GetElementDofs(e, dofs);
    for (int i = 0; i < dofs.Size(); i++) {
      const IntegrationPoint& ip = nodes.IntPoint(i);
      T->SetIntPoint(&ip);
      eval(*T, ip, Cm);
      real_t* Wp = W.GetData() + static_cast<std::size_t>(dofs[i]) * ns * ns;
      for (int s = 0; s < ns; s++) {
        for (int t = 0; t < ns; t++) {
          Wp[s * ns + t] = Cm(s, t) * w[t] / (a[s] * a[t]);
        }
      }
    }
  }
  return W;
}

Vector NodalBranchTensors(FiniteElementSpace& sfes, const Rheology& rh,
                          int k) {
  return NodalTensors(sfes, [&](ElementTransformation& T,
                                const IntegrationPoint& ip, DenseMatrix& C) {
    rh.BranchModulus(k, T, ip, C);
  });
}

Vector NodalUnrelaxedTensors(FiniteElementSpace& sfes, const Rheology& rh) {
  return NodalTensors(sfes, [&](ElementTransformation& T,
                                const IntegrationPoint& ip, DenseMatrix& C) {
    rh.UnrelaxedModulus(T, ip, C);
  });
}

}  // namespace

ViscoelasticOperator::ViscoelasticOperator(LinearQuasiStaticProblem& problem,
                                           int internal_order, StrainMap map)
    : TimeDependentOperator(0, 0.0, TimeDependentOperator::EXPLICIT),
      problem_(problem),
      map_(map) {
  ufes_ = &problem_.DisplacementSpace();
  Mesh* mesh = ufes_->GetMesh();
  const int dim = mesh->Dimension();
  const auto& rh = problem_.Rheology();
  MFEM_VERIFY(rh.SpaceDim() == dim,
              "ViscoelasticOperator: rheology/mesh dimension mismatch.");

  parallel_ = detail::IsParallel(*ufes_);
#ifdef MFEM_USE_MPI
  if (parallel_) {
    comm_ = static_cast<ParFiniteElementSpace*>(ufes_)->GetComm();
  }
#endif

  // Default internal order: the smallest L2 order containing eps(u)
  // exactly, p - 1 on simplices and p on tensor-product elements (the
  // gradient of a Q_p field has Q_{p,p-1} components).
  int order = internal_order;
  if (order < 0) {
    const int p = ufes_->GetMaxElementOrder();
    const bool tensor = mesh->HasGeometry(Geometry::SQUARE) ||
                        mesh->HasGeometry(Geometry::CUBE) ||
                        mesh->HasGeometry(Geometry::PRISM) ||
                        mesh->HasGeometry(Geometry::PYRAMID);
    order = tensor ? p : std::max(p - 1, 0);
#ifdef MFEM_USE_MPI
    if (parallel_) {
      int g = order;
      MPI_Allreduce(&order, &g, 1, MPI_INT, MPI_MAX, comm_);
      order = g;
    }
#endif
  }
  fec_ = std::make_unique<L2_FECollection>(order, dim);
  tracefree_ = rh.TraceFreeInternalVariables();
  ns_ = SymmetricComponentOrder::Count(dim);
  nc_ = tracefree_ ? SymmetricComponentOrder::TraceFreeCount(dim) : ns_;
  dfes_ = detail::MakeFESpace(*ufes_, fec_.get(), nc_, Ordering::byNODES);
  sfes_ = detail::MakeFESpace(*ufes_, fec_.get(), 1);
  nd_ = sfes_->GetVSize();
  MFEM_VERIFY(dfes_->GetVSize() == nd_ * nc_, "layout");

  // Geometric coupling, unit coefficient: all material weighting is
  // applied pointwise at the internal nodes. Trace-free deviatoric
  // strain for the isotropic body, the full strain otherwise.
  B_ = std::make_unique<MixedBilinearForm>(ufes_, dfes_.get());
  if (tracefree_) {
    B_->AddDomainIntegrator(
        new DomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator());
  } else {
    B_->AddDomainIntegrator(new DomainSymmetricMatrixStrainIntegrator());
  }
  B_->Assemble();
  B_->Finalize();

  if (map_ == StrainMap::Interpolation) {
    D_interp_ = std::make_unique<DiscreteLinearOperator>(ufes_, dfes_.get());
    if (tracefree_) {
      D_interp_->AddDomainInterpolator(new DeviatoricStrainInterpolator());
    } else {
      D_interp_->AddDomainInterpolator(new StrainInterpolator());
    }
    D_interp_->Assemble();
    D_interp_->Finalize();
  } else {
    Minv_ = BuildBlockMassInverse(*sfes_);
    Ginv_ = MetricInverse(dim, tracefree_);
  }

  // Material data sampled once at the internal nodes (L2 nodes are
  // element-interior, so attribute-wise data is sampled correctly), then
  // restricted to each branch's active nodes: all of them, or those of the
  // elements the rheology marks for the branch.
  const int K = rh.NumBranches();
  branch_modulus_.resize(K);
  itau0_.resize(K);
  itau_.resize(K);
  law_.resize(K, nullptr);
  law_params_.resize(K);
  num_params_.resize(K, 0);
  nodes_.resize(K);
  slot_.resize(K);
  Bk_.resize(K, nullptr);
  linear_ = rh.IsLinear();
  offsets_.SetSize(1);
  offsets_[0] = 0;
  const SparseMatrix& Bmat = B_->SpMat();
  int max_block = nd_ * nc_;
  bool all_marked = K > 0;
  for (int k = 0; k < K; k++) {
    const Array<int>* marker = rh.BranchMarker(k);
    Array<int>& nodes = nodes_[k];
    if (!marker) {
      all_marked = false;
      nodes.SetSize(nd_);
      for (int p = 0; p < nd_; p++) {
        nodes[p] = p;
      }
    } else {
      MFEM_VERIFY(marker->Size() >= mesh->attributes.Max(),
                  "ViscoelasticOperator: branch marker too small.");
      Array<int> dofs;
      for (int e = 0; e < mesh->GetNE(); e++) {
        if ((*marker)[mesh->GetAttribute(e) - 1]) {
          sfes_->GetElementDofs(e, dofs);
          nodes.Append(dofs);
        }
      }
    }
    const int ndk = nodes.Size();
    slot_[k].SetSize(nd_);
    slot_[k] = -1;
    for (int q = 0; q < ndk; q++) {
      slot_[k][nodes[q]] = q;
    }
    auto gather = [&](const Vector& full, int per_node) {
      Vector v(ndk * per_node);
      for (int q = 0; q < ndk; q++) {
        for (int i = 0; i < per_node; i++) {
          v[q * per_node + i] = full[nodes[q] * per_node + i];
        }
      }
      return v;
    };
    if (tracefree_) {
      branch_modulus_[k] =
          gather(NodalValues(*sfes_, rh.BranchShearModulus(k)), 1);
      branch_modulus_[k] *= 2.0;
    } else {
      branch_modulus_[k] = gather(NodalBranchTensors(*sfes_, rh, k), ns_ * ns_);
    }
    itau0_[k] = gather(NodalValues(*sfes_, rh.RelaxationTime(k)), 1);
    for (int q = 0; q < ndk; q++) {
      MFEM_VERIFY(itau0_[k][q] > 0.0,
                  "ViscoelasticOperator: relaxation times must be positive.");
      itau0_[k][q] = 1.0 / itau0_[k][q];
    }
    itau_[k] = itau0_[k];
    const RelaxationLaw* law = rh.Law(k);
    if (law && law->IsStateDependent()) {
      law_[k] = law;
      const int np = law->NumParameters();
      num_params_[k] = np;
      law_params_[k].SetSize(ndk * np);
      for (int i = 0; i < np; i++) {
        Vector v = NodalValues(*sfes_, law->Parameter(i));
        for (int q = 0; q < ndk; q++) {
          law_params_[k][q * np + i] = v[nodes[q]];
        }
      }
    }
    m_out_.push_back(detail::MakeGridFunction(dfes_.get()));
    *m_out_.back() = 0.0;
    offsets_.Append(offsets_.Last() + ndk * nc_);
    max_block = std::max(max_block, ndk * nc_);

    beta_.push_back(detail::MakeGridFunction(sfes_.get()));
    *beta_.back() = 1.0;
    beta_coef_.push_back(
        std::make_unique<GridFunctionCoefficient>(beta_.back().get()));
    beta_ptrs_.push_back(beta_coef_.back().get());

    // The rows of B for the branch's nodes (an L2 node's row involves its
    // own element only, so these are the rows a form assembled on the
    // region alone would have).
    if (!marker) {
      Bk_[k] = &Bmat;
    } else {
      auto Bk = std::make_unique<SparseMatrix>(ndk * nc_, Bmat.Width());
      Array<int> cols;
      Vector vals;
      for (int c = 0; c < nc_; c++) {
        for (int q = 0; q < ndk; q++) {
          Bmat.GetRow(c * nd_ + nodes[q], cols, vals);
          if (cols.Size() > 0) {
            Bk->AddRow(c * ndk + q, cols, vals);
          }
        }
      }
      Bk->Finalize();
      Bk_[k] = Bk.get();
      Bk_owned_.push_back(std::move(Bk));
    }
  }

  // When every branch is confined to a region, the per-step strain is
  // needed only at the union of the branch nodes. The union is
  // element-closed (branch nodes are gathered element-wise), so the row
  // restriction of the strain map reproduces the full map's values there
  // exactly: an L2 node's B and M rows involve its own element only, and
  // G^{-1} acts node-locally. Decided per rank in parallel — a rank whose
  // elements all carry branches keeps the whole-mesh path; no collectives
  // are involved either way.
  if (all_marked) {
    Array<int> uni;
    for (int k = 0; k < K; k++) {
      uni.Append(nodes_[k]);
    }
    uni.Sort();
    uni.Unique();
    if (uni.Size() < nd_) {
      strain_restricted_ = true;
      union_nodes_ = uni;
      const int nu = union_nodes_.Size();
      if (nu > 0) {
        // Row restriction (columns unchanged): rows c nd_ + p -> c nu + q.
        auto restrict_rows = [&](const SparseMatrix& A) {
          auto R = std::make_unique<SparseMatrix>(nu * nc_, A.Width());
          Array<int> cols;
          Vector vals;
          for (int c = 0; c < nc_; c++) {
            for (int q = 0; q < nu; q++) {
              A.GetRow(c * nd_ + union_nodes_[q], cols, vals);
              if (cols.Size() > 0) {
                R->AddRow(c * nu + q, cols, vals);
              }
            }
          }
          R->Finalize();
          return R;
        };
        if (map_ == StrainMap::Interpolation) {
          D_union_ = restrict_rows(D_interp_->SpMat());
        } else {
          B_union_ = restrict_rows(Bmat);
          // M^{-1} with rows and columns in union numbering; the columns
          // of a union row all lie in the union (element closure). GetRow
          // of a finalized matrix aliases its internal arrays, so the
          // columns are remapped in a copy.
          Array<int> uslot(nd_);
          uslot = -1;
          for (int q = 0; q < nu; q++) {
            uslot[union_nodes_[q]] = q;
          }
          auto Mu = std::make_unique<SparseMatrix>(nu, nu);
          Array<int> cols, mapped;
          Vector vals;
          for (int q = 0; q < nu; q++) {
            Minv_->GetRow(union_nodes_[q], cols, vals);
            mapped.SetSize(cols.Size());
            for (int i = 0; i < cols.Size(); i++) {
              mapped[i] = uslot[cols[i]];
              MFEM_VERIFY(mapped[i] >= 0,
                          "ViscoelasticOperator: branch-node union is not "
                          "element-closed.");
            }
            if (mapped.Size() > 0) {
              Mu->AddRow(q, mapped, vals);
            }
          }
          Mu->Finalize();
          Minv_union_ = std::move(Mu);
        }
      }
    }
  }

  if (!linear_) {
    CU_ = NodalUnrelaxedTensors(*sfes_, rh);
  }

  d_.SetSize(nd_ * nc_);
  d_prev_.SetSize(nd_ * nc_);
  // The restricted strain path writes union rows only; the others must
  // read as zero.
  d_ = 0.0;
  d_prev_ = 0.0;
  dual_.SetSize(max_block);
  zeta_.SetSize(max_block);
  force_.SetSize(ufes_->GetVSize());

  height = width = offsets_.Last();
}

Vector ViscoelasticOperator::Branch(const Vector& m, int k) const {
  Vector v;
  v.MakeRef(const_cast<Vector&>(m), BranchOffset(k), BranchSize(k));
  return v;
}

void ViscoelasticOperator::BranchToFull(const Vector& m, int k,
                                        Vector& full) const {
  full.SetSize(nd_ * nc_);
  full = 0.0;
  const Vector b = Branch(m, k);
  const Array<int>& nodes = nodes_[k];
  const int ndk = nodes.Size();
  for (int c = 0; c < nc_; c++) {
    for (int q = 0; q < ndk; q++) {
      full[c * nd_ + nodes[q]] = b[c * ndk + q];
    }
  }
}

void ViscoelasticOperator::ComputeStrain(const GridFunction& u,
                                         Vector& d) const {
  d.SetSize(nd_ * nc_);
  if (map_ == StrainMap::Interpolation) {
    D_interp_->Mult(u, d);
    return;
  }
  // d = (G^{-1} (x) M^{-1}) B u: scalar mass inverse per component, then the
  // inverse basis-tensor metric across components.
  dual_.SetSize(nd_ * nc_);
  zeta_.SetSize(nd_ * nc_);
  B_->Mult(u, dual_);
  Vector tc, duc;
  for (int c = 0; c < nc_; c++) {
    tc.MakeRef(zeta_, c * nd_, nd_);
    duc.MakeRef(dual_, c * nd_, nd_);
    Minv_->Mult(duc, tc);
  }
  for (int c = 0; c < nc_; c++) {
    for (int p = 0; p < nd_; p++) {
      real_t v = 0.0;
      for (int cp = 0; cp < nc_; cp++) {
        v += Ginv_(c, cp) * zeta_[cp * nd_ + p];
      }
      d[c * nd_ + p] = v;
    }
  }
}

void ViscoelasticOperator::ComputeStrainRestricted(const GridFunction& u,
                                                   Vector& d) const {
  MFEM_ASSERT(d.Size() == nd_ * nc_, "ComputeStrainRestricted: layout");
  const int nu = union_nodes_.Size();
  if (nu == 0) {
    return;  // no branch nodes on this rank; d stays zero
  }
  dual_.SetSize(nu * nc_);
  if (map_ == StrainMap::Interpolation) {
    D_union_->Mult(u, dual_);
    for (int c = 0; c < nc_; c++) {
      for (int q = 0; q < nu; q++) {
        d[c * nd_ + union_nodes_[q]] = dual_[c * nu + q];
      }
    }
    return;
  }
  // The union rows of d = (G^{-1} (x) M^{-1}) B u, as in ComputeStrain().
  zeta_.SetSize(nu * nc_);
  B_union_->Mult(u, dual_);
  Vector tc, duc;
  for (int c = 0; c < nc_; c++) {
    tc.MakeRef(zeta_, c * nu, nu);
    duc.MakeRef(dual_, c * nu, nu);
    Minv_union_->Mult(duc, tc);
  }
  for (int c = 0; c < nc_; c++) {
    for (int q = 0; q < nu; q++) {
      real_t v = 0.0;
      for (int cp = 0; cp < nc_; cp++) {
        v += Ginv_(c, cp) * zeta_[cp * nu + q];
      }
      d[c * nd_ + union_nodes_[q]] = v;
    }
  }
}

void ViscoelasticOperator::ComputeCurrentStrain() const {
  if (strain_restricted_) {
    ComputeStrainRestricted(problem_.Displacement(), d_);
  } else {
    ComputeStrain(problem_.Displacement(), d_);
  }
}

void ViscoelasticOperator::ApplyBranchModulus(int k, const Vector& x,
                                              Vector& y) const {
  y.SetSize(x.Size());
  const Vector& w = branch_modulus_[k];
  const int nd = nodes_[k].Size();
  MFEM_ASSERT(x.Size() == nd * nc_, "ApplyBranchModulus: branch layout");
  if (tracefree_) {
    for (int c = 0; c < nc_; c++) {
      const int o = c * nd;
      for (int q = 0; q < nd; q++) {
        y[o + q] = w[q] * x[o + q];
      }
    }
    return;
  }
  const int ns = ns_;
  for (int q = 0; q < nd; q++) {
    const real_t* Wq = w.GetData() + static_cast<std::size_t>(q) * ns * ns;
    for (int s = 0; s < ns; s++) {
      real_t v = 0.0;
      for (int t = 0; t < ns; t++) {
        v += Wq[s * ns + t] * x[t * nd + q];
      }
      y[s * nd + q] = v;
    }
  }
}

void ViscoelasticOperator::EvaluateRelaxationTimes(const Vector& d,
                                                   const Vector& m) const {
  if (linear_) {
    return;
  }
  const int dim = ufes_->GetMesh()->Dimension();
  const int ns = ns_, nc = nc_, nd = nd_;
  const int K = NumBranches();
  LocalState s(dim);
  std::vector<Vector> mk(K);
  for (int k = 0; k < K; k++) {
    mk[k] = Branch(m, k);
  }
  // Index of the dropped diagonal component in the trace-free layout.
  const int last_diag = SymmetricComponentOrder::TraceFreeDropped(dim);
  auto complete = [&](Vector& full) {
    if (tracefree_) {
      real_t tr = 0.0;
      for (int j = 0; j < dim - 1; j++) {
        tr += full[SymmetricComponentOrder::Offset(dim, j, j)];
      }
      full[last_diag] = -tr;
    }
  };
  // Full-layout vector at node p; block of branch j at its position q.
  auto expand = [&](const Vector& x, int p, Vector& full) {
    for (int c = 0; c < nc; c++) {
      full[c] = x[c * nd + p];
    }
    complete(full);
  };
  auto expand_branch = [&](int j, int q, Vector& full) {
    const int ndj = nodes_[j].Size();
    for (int c = 0; c < nc; c++) {
      full[c] = mk[j][c * ndj + q];
    }
    complete(full);
  };
  Vector tmp(ns);
  for (int k = 0; k < K; k++) {
    if (!law_[k]) {
      continue;
    }
    const Array<int>& nodes = nodes_[k];
    for (int q = 0; q < nodes.Size(); q++) {
      const int p = nodes[q];
      expand(d, p, s.strain);
      // Stress at p: C_U eps - sum_j C_j m_j over the branches living at
      // p, through the nodal unrelaxed tensor and the branch moduli. With
      // trace-free variables the strain is dev(eps), so C_U eps is
      // 2 mu_U dev(eps) and the result is the full deviatoric stress
      // 2 mu_inf dev(eps) + sum_j 2 mu_j (dev(eps) - m_j); the branch
      // moduli are then the scalars 2 mu_j.
      const real_t* Wu =
          CU_.GetData() + static_cast<std::size_t>(p) * ns * ns;
      for (int a = 0; a < ns; a++) {
        real_t v = 0.0;
        for (int b = 0; b < ns; b++) {
          v += Wu[a * ns + b] * s.strain[b];
        }
        s.stress[a] = v;
      }
      for (int j = 0; j < K; j++) {
        const int qj = slot_[j][p];
        if (qj < 0) {
          continue;
        }
        expand_branch(j, qj, tmp);
        if (tracefree_) {
          const real_t two_mu = branch_modulus_[j][qj];
          for (int a = 0; a < ns; a++) {
            s.stress[a] -= two_mu * tmp[a];
          }
        } else {
          const real_t* Wj = branch_modulus_[j].GetData() +
                             static_cast<std::size_t>(qj) * ns * ns;
          for (int a = 0; a < ns; a++) {
            real_t v = 0.0;
            for (int b = 0; b < ns; b++) {
              v += Wj[a * ns + b] * tmp[b];
            }
            s.stress[a] -= v;
          }
        }
      }
      expand_branch(k, q, s.m);
      const real_t* params = law_params_[k].GetData() + q * num_params_[k];
      const real_t F = law_[k]->Factor(params, s);
      MFEM_ASSERT(F > 0.0, "relaxation law factor must be positive");
      itau_[k][q] = itau0_[k][q] / F;
    }
  }
  tau_version_++;
}

real_t ViscoelasticOperator::RelaxationTimeChange(
    const std::vector<Vector>& old_itau) const {
  real_t change = 0.0;
  if (!linear_) {
    for (int k = 0; k < NumBranches(); k++) {
      for (int p = 0; p < itau_[k].Size(); p++) {
        const real_t a = old_itau[k][p], b = itau_[k][p];
        change = std::max(change, std::abs(b - a) / std::max(a, b));
      }
    }
  }
  return GlobalMax(change);
}

real_t ViscoelasticOperator::GlobalMax(real_t v) const {
#ifdef MFEM_USE_MPI
  if (parallel_) {
    real_t g = 0.0;
    MPI_Allreduce(&v, &g, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX, comm_);
    return g;
  }
#endif
  return v;
}

void ViscoelasticOperator::AddCoupledForce(int k, const Vector& zeta) const {
  force_.SetSize(ufes_->GetVSize());
  Bk_[k]->MultTranspose(zeta, force_);
  problem_.AddForce(force_);
}

bool ViscoelasticOperator::ElasticUpdate(const Vector& m, real_t t) const {
  problem_.AssembleForce(t);
  for (int k = 0; k < NumBranches(); k++) {
    ApplyBranchModulus(k, Branch(m, k), zeta_);
    AddCoupledForce(k, zeta_);
  }
  return problem_.Solve();
}

void ViscoelasticOperator::Rate(int k, const Vector& m_k, const Vector& d,
                                Vector& k_out) const {
  const Array<int>& nodes = nodes_[k];
  const int nd = nodes.Size();
  for (int c = 0; c < nc_; c++) {
    const int o = c * nd, of = c * nd_;
    for (int q = 0; q < nd; q++) {
      k_out[o + q] = (d[of + nodes[q]] - m_k[o + q]) * itau_[k][q];
    }
  }
}

void ViscoelasticOperator::LocalExponentialUpdate(int k, real_t dt, Vector& m_k,
                                                  const Vector& d) const {
  const Array<int>& nodes = nodes_[k];
  const int nd = nodes.Size();
  for (int q = 0; q < nd; q++) {
    const real_t a = std::exp(-dt * itau_[k][q]);
    const real_t b = 1.0 - a;
    const int p = nodes[q];
    for (int c = 0; c < nc_; c++) {
      m_k[c * nd + q] = a * m_k[c * nd + q] + b * d[c * nd_ + p];
    }
  }
}

void ViscoelasticOperator::SetEffectiveModulus(real_t dt, Scheme scheme) const {
  if (scheme == scheme_ && dt == effective_dt_ &&
      effective_version_ == tau_version_) {
    return;
  }
  MFEM_VERIFY(problem_.SupportsRelaxationWeights(),
              "ViscoelasticOperator: this scheme needs a quasi-static problem "
              "supporting SetRelaxationWeights(); use an explicit solver or "
              "ExponentialEulerSolver instead.");
  // Nodal weights beta_k: the effective modulus is C_inf + sum_k beta_k C_k
  // after eliminating m_k^{n+1}.
  for (int k = 0; k < NumBranches(); k++) {
    GridFunction& beta_k = *beta_[k];
    const Array<int>& nodes = nodes_[k];
    for (int q = 0; q < nodes.Size(); q++) {
      const real_t h = dt * itau_[k][q];
      if (scheme == Scheme::BackwardEuler) {
        beta_k[nodes[q]] = 1.0 / (1.0 + h);
      } else {
        real_t e, alpha, beta;
        detail::ExponentialTrapezoidWeights(h, e, alpha, beta);
        beta_k[nodes[q]] = 1.0 - beta;
      }
    }
  }
  problem_.SetRelaxationWeights(beta_ptrs_);
  scheme_ = scheme;
  effective_dt_ = dt;
  effective_version_ = tau_version_;
}

void ViscoelasticOperator::UseUnrelaxedOperator() const {
  if (scheme_ != Scheme::None) {
    problem_.ClearRelaxationWeights();
    scheme_ = Scheme::None;
    effective_dt_ = -1.0;
  }
}

bool ViscoelasticOperator::CacheMatches(const Vector& m, real_t t) const {
  // Equal to round-off: an implicit step stores (m + h d) / (1 + h) as the
  // key, while the ODE solver forms m + dt k, and output times may be
  // accumulated by repeated addition; the two agree to a few ulps. A
  // relative tolerance far below any solver tolerance accepts them, so
  // the displacement is not solved again for the same state.
  constexpr real_t rel = 1e-12;
  int ok = cache_valid_ && m.Size() == cached_m_.Size() &&
           std::abs(t - cached_t_) <=
               rel * std::max<real_t>(1.0, std::abs(cached_t_));
  if (ok) {
    real_t scale = 0.0;
    for (int j = 0; j < m.Size(); j++) {
      scale = std::max(scale, std::abs(cached_m_[j]));
    }
#ifdef MFEM_USE_MPI
    if (parallel_) {
      MPI_Allreduce(MPI_IN_PLACE, &scale, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_MAX, comm_);
    }
#endif
    const real_t tol = rel * scale;
    for (int j = 0; j < m.Size(); j++) {
      if (std::abs(m[j] - cached_m_[j]) > tol) {
        ok = 0;
        break;
      }
    }
  }
#ifdef MFEM_USE_MPI
  if (parallel_) {
    int g = 0;
    MPI_Allreduce(&ok, &g, 1, MPI_INT, MPI_LAND, comm_);
    ok = g;
  }
#endif
  return ok != 0;
}

void ViscoelasticOperator::UpdateCache(const Vector& m, real_t t) const {
  cached_m_ = m;
  cached_t_ = t;
  cache_valid_ = true;
}

// --- ODE interface -----------------------------------------------------------

void ViscoelasticOperator::Mult(const Vector& m, Vector& k) const {
  const real_t t = GetTime();
  UseUnrelaxedOperator();
  MFEM_VERIFY(ElasticUpdate(m, t),
              "ViscoelasticOperator::Mult: elastic solve failed.");
  ComputeCurrentStrain();
  k.SetSize(m.Size());
  EvaluateRelaxationTimes(d_, m);
  for (int b = 0; b < NumBranches(); b++) {
    Vector kb = Branch(k, b);
    Rate(b, Branch(m, b), d_, kb);
  }
  UpdateCache(m, t);
}

void ViscoelasticOperator::ImplicitSolve(real_t dt, const Vector& m,
                                         Vector& k) {
  const real_t t = GetTime();
  // Eliminating m^{n+1} = (m + h d^{n+1}) / (1 + h) leaves the effective
  // operator with the force sum_k B^T C_k m_k / (1 + h_k). With
  // state-dependent times the h_k of the end state are found by
  // predictor–corrector: the current (last evaluated) times first, then
  // those of the end state, repeated.
  k.SetSize(m.Size());
  cached_m_.SetSize(m.Size());
  last_passes_ = 0;
  for (int pass = 0;; pass++) {
    SetEffectiveModulus(dt, Scheme::BackwardEuler);
    problem_.AssembleForce(t);
    for (int b = 0; b < NumBranches(); b++) {
      Vector mb = Branch(m, b);
      const int nd = nodes_[b].Size();
      dual_.SetSize(nd * nc_);
      for (int c = 0; c < nc_; c++) {
        const int o = c * nd;
        for (int q = 0; q < nd; q++) {
          const real_t h = dt * itau_[b][q];
          dual_[o + q] = mb[o + q] / (1.0 + h);
        }
      }
      ApplyBranchModulus(b, dual_, zeta_);
      AddCoupledForce(b, zeta_);
    }
    MFEM_VERIFY(problem_.Solve(),
                "ViscoelasticOperator::ImplicitSolve: elastic solve failed.");
    ComputeCurrentStrain();

    for (int b = 0; b < NumBranches(); b++) {
      Vector mb = Branch(m, b);
      Vector kb = Branch(k, b);
      Vector cb = Branch(cached_m_, b);
      const Array<int>& nodes = nodes_[b];
      const int nd = nodes.Size();
      for (int c = 0; c < nc_; c++) {
        const int o = c * nd, of = c * nd_;
        for (int q = 0; q < nd; q++) {
          const real_t h = dt * itau_[b][q];
          const real_t m_new =
              (mb[o + q] + h * d_[of + nodes[q]]) / (1.0 + h);
          kb[o + q] = (m_new - mb[o + q]) / dt;
          cb[o + q] = m_new;
        }
      }
    }
    last_passes_ = pass;
    if (linear_ || pass >= max_corrector_) {
      break;
    }
    // Corrector: times at the end state (d^{n+1}, m^{n+1}).
    const std::vector<Vector> old_itau = itau_;
    EvaluateRelaxationTimes(d_, cached_m_);
    if (RelaxationTimeChange(old_itau) < corrector_tol_) {
      break;
    }
  }
  // The displacement is consistent with the new state m + dt k (exactly
  // when the internal order resolves eps(u), so that B^T beta C D =
  // K(beta C)).
  cached_t_ = t;
  cache_valid_ = true;
}

void ViscoelasticOperator::ExponentialEulerStep(Vector& m, real_t& t,
                                                real_t dt) {
  SetTime(t);
  if (!CacheMatches(m, t)) {
    UseUnrelaxedOperator();
    MFEM_VERIFY(ElasticUpdate(m, t),
                "ViscoelasticOperator::ExponentialEulerStep: elastic solve "
                "failed.");
    ComputeCurrentStrain();
  }
  EvaluateRelaxationTimes(d_, m);  // frozen over the step
  for (int b = 0; b < NumBranches(); b++) {
    Vector mb = Branch(m, b);
    LocalExponentialUpdate(b, dt, mb, d_);
  }
  t += dt;
  SetTime(t);
  cache_valid_ = false;  // u is not consistent with the new state
}

void ViscoelasticOperator::ExponentialTrapezoidStep(Vector& m, real_t& t,
                                                    real_t dt) {
  SetTime(t);
  // d^n = D u^n with the unrelaxed operator; reuse if the previous step (or
  // a SolveElastic) left it consistent.
  if (!CacheMatches(m, t)) {
    UseUnrelaxedOperator();
    MFEM_VERIFY(ElasticUpdate(m, t),
                "ViscoelasticOperator::ExponentialTrapezoidStep: elastic "
                "solve failed.");
    ComputeCurrentStrain();
  }
  d_prev_ = d_;
  EvaluateRelaxationTimes(d_prev_, m);  // predictor: start

  // ETD1 prediction of the step (times and strain frozen at the start),
  // kept for the error estimate.
  const Vector m_start(m);
  predictor_diff_ = m;
  m_scale_.SetSize(m.Size());
  for (int b = 0; b < NumBranches(); b++) {
    Vector pb = Branch(predictor_diff_, b);
    LocalExponentialUpdate(b, dt, pb, d_prev_);
  }

  last_passes_ = 0;
  for (int pass = 0;; pass++) {
    SetEffectiveModulus(dt, Scheme::ExponentialTrapezoid);
    problem_.AssembleForce(t + dt);
    for (int b = 0; b < NumBranches(); b++) {
      Vector mb = Branch(m_start, b);
      const Array<int>& nodes = nodes_[b];
      const int nd = nodes.Size();
      dual_.SetSize(nd * nc_);
      for (int q = 0; q < nd; q++) {
        real_t e, alpha, beta;
        detail::ExponentialTrapezoidWeights(dt * itau_[b][q], e, alpha, beta);
        for (int c = 0; c < nc_; c++) {
          dual_[c * nd + q] =
              e * mb[c * nd + q] + alpha * d_prev_[c * nd_ + nodes[q]];
        }
      }
      ApplyBranchModulus(b, dual_, zeta_);
      AddCoupledForce(b, zeta_);
    }
    MFEM_VERIFY(problem_.Solve(),
                "ViscoelasticOperator::ExponentialTrapezoidStep: elastic "
                "solve failed.");
    ComputeCurrentStrain();  // d^{n+1}

    for (int b = 0; b < NumBranches(); b++) {
      Vector mb = Branch(m, b);
      Vector m0 = Branch(m_start, b);
      const Array<int>& nodes = nodes_[b];
      const int nd = nodes.Size();
      for (int q = 0; q < nd; q++) {
        real_t e, alpha, beta;
        detail::ExponentialTrapezoidWeights(dt * itau_[b][q], e, alpha, beta);
        const int p = nodes[q];
        for (int c = 0; c < nc_; c++) {
          const int idx = c * nd + q, full = c * nd_ + p;
          mb[idx] = e * m0[idx] + alpha * d_prev_[full] + beta * d_[full];
        }
      }
    }
    last_passes_ = pass;
    if (linear_ || pass >= max_corrector_) {
      break;
    }
    // Corrector: times at the midpoint state.
    const std::vector<Vector> old_itau = itau_;
    Vector m_mid(m);
    m_mid += m_start;
    m_mid *= 0.5;
    Vector d_mid(d_);
    d_mid += d_prev_;
    d_mid *= 0.5;
    EvaluateRelaxationTimes(d_mid, m_mid);
    if (RelaxationTimeChange(old_itau) < corrector_tol_) {
      break;
    }
  }

  for (int j = 0; j < m.Size(); j++) {
    predictor_diff_[j] = m[j] - predictor_diff_[j];
    m_scale_[j] = std::max(std::abs(m_start[j]), std::abs(m[j]));
  }
  t += dt;
  SetTime(t);
  // By construction u^{n+1} satisfies K_U u = f + sum B^T C m^{n+1}
  // (exactly when the internal order resolves eps(u)).
  UpdateCache(m, t);
}

real_t ViscoelasticOperator::ErrorEstimate(real_t rtol, real_t atol) const {
  real_t sum = 0.0;
  for (int j = 0; j < predictor_diff_.Size(); j++) {
    const real_t w = atol + rtol * m_scale_[j];
    const real_t r = predictor_diff_[j] / w;
    sum += r * r;
  }
  real_t n = static_cast<real_t>(predictor_diff_.Size());
#ifdef MFEM_USE_MPI
  if (parallel_) {
    real_t buf[2] = {sum, n}, g[2] = {0.0, 0.0};
    MPI_Allreduce(buf, g, 2, MPITypeMap<real_t>::mpi_type, MPI_SUM, comm_);
    sum = g[0];
    n = g[1];
  }
#endif
  return n > 0.0 ? std::sqrt(sum / n) : 0.0;
}

// --- observation -------------------------------------------------------------

bool ViscoelasticOperator::SolveElastic(const Vector& m, real_t t) {
  SetTime(t);
  if (CacheMatches(m, t)) {
    return true;
  }
  UseUnrelaxedOperator();
  const bool ok = ElasticUpdate(m, t);
  if (ok) {
    ComputeCurrentStrain();
    UpdateCache(m, t);
  }
  return ok;
}

void ViscoelasticOperator::SyncFields(const Vector& m) {
  Vector full;
  for (int b = 0; b < NumBranches(); b++) {
    BranchToFull(m, b, full);
    *m_out_[b] = full;
  }
}

void ViscoelasticOperator::RegisterFields(DataCollection& dc) {
  problem_.RegisterFields(dc);
  // One branch with the default label keeps the plain name; otherwise the
  // rheology's label (e.g. "<region>_branch<j>" for a CompositeRheology).
  for (int b = 0; b < NumBranches(); b++) {
    std::string name = "internal_variable";
    const std::string label = problem_.Rheology().BranchLabel(b);
    if (NumBranches() > 1 || label != "branch0") {
      name += "_" + label;
    }
    dc.RegisterField(name, m_out_[b].get());
  }
}

real_t ViscoelasticOperator::MinRelaxationTime() const {
  real_t itau_max = 0.0;
  for (const auto& it : itau_) {
    if (it.Size() > 0) {
      itau_max = std::max(itau_max, it.Max());
    }
  }
#ifdef MFEM_USE_MPI
  if (parallel_) {
    real_t g = 0.0;
    MPI_Allreduce(&itau_max, &g, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                  comm_);
    itau_max = g;
  }
#endif
  return itau_max > 0.0 ? 1.0 / itau_max : infinity();
}

// --- ODESolver adaptors ------------------------------------------------------

void ExponentialEulerSolver::Init(TimeDependentOperator& f) {
  ODESolver::Init(f);
  op_ = dynamic_cast<ViscoelasticOperator*>(&f);
  MFEM_VERIFY(op_, "ExponentialEulerSolver requires a ViscoelasticOperator.");
}

void ExponentialEulerSolver::Step(Vector& x, real_t& t, real_t& dt) {
  op_->ExponentialEulerStep(x, t, dt);
}

void ExponentialTrapezoidSolver::Init(TimeDependentOperator& f) {
  ODESolver::Init(f);
  op_ = dynamic_cast<ViscoelasticOperator*>(&f);
  MFEM_VERIFY(op_,
              "ExponentialTrapezoidSolver requires a ViscoelasticOperator.");
}

void ExponentialTrapezoidSolver::Step(Vector& x, real_t& t, real_t& dt) {
  op_->ExponentialTrapezoidStep(x, t, dt);
}

void AdaptiveExponentialTrapezoidSolver::Init(TimeDependentOperator& f) {
  ODESolver::Init(f);
  op_ = dynamic_cast<ViscoelasticOperator*>(&f);
  MFEM_VERIFY(op_, "AdaptiveExponentialTrapezoidSolver requires a "
                   "ViscoelasticOperator.");
}

void AdaptiveExponentialTrapezoidSolver::Step(Vector& x, real_t& t,
                                              real_t& dt) {
  real_t h = std::min(dt, dt_max_);
  h = std::max(h, dt_min_);
  for (;;) {
    trial_ = x;
    real_t t_trial = t;
    op_->ExponentialTrapezoidStep(trial_, t_trial, h);
    last_err_ = op_->ErrorEstimate(rtol_, atol_);
    if (last_err_ <= 1.0 || h <= dt_min_) {
      x = trial_;
      t = t_trial;
      accepted_++;
      break;
    }
    rejected_++;
    h = std::max(dt_min_,
                 h * std::max(shrink_, safety_ / std::sqrt(last_err_)));
  }
  const real_t factor =
      last_err_ > 0.0
          ? std::min(grow_, std::max(shrink_, safety_ / std::sqrt(last_err_)))
          : grow_;
  dt = std::min(dt_max_, std::max(dt_min_, h * factor));
}

int AdaptiveExponentialTrapezoidSolver::Integrate(Vector& x, real_t& t,
                                                  real_t t_final, real_t& dt) {
  int steps = 0;
  while (t < t_final - 1e-14 * std::max<real_t>(1.0, std::abs(t_final))) {
    real_t h = std::min(dt, t_final - t);
    // Avoid a tiny last step: split the remainder in two if needed.
    if (t + h < t_final && t_final - (t + h) < 0.1 * h) {
      h = 0.5 * (t_final - t);
    }
    Step(x, t, h);
    dt = h;
    steps++;
  }
  return steps;
}

}  // namespace AdGIA
