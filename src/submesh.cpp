/**
 * @file submesh.cpp
 * @brief Implementation of SubMeshDofInjection and the SubMesh mixed
 * bilinear forms.
 */

#include "AdGIA/submesh.hpp"

namespace AdGIA {

namespace {

// The submesh facts SubMeshDofInjection needs, read from either a SubMesh or
// a ParSubMesh (which are unrelated types in MFEM).
struct SubMeshInfo {
  const mfem::Mesh* parent;
  mfem::SubMesh::From from;
  const mfem::Array<int>* parent_element_ids;
};

SubMeshInfo GetSubMeshInfo(const mfem::Mesh* mesh) {
  if (auto* sm = dynamic_cast<const mfem::SubMesh*>(mesh)) {
    return {sm->GetParent(), sm->GetFrom(), &sm->GetParentElementIDMap()};
  }
#ifdef MFEM_USE_MPI
  if (auto* psm = dynamic_cast<const mfem::ParSubMesh*>(mesh)) {
    return {psm->GetParent(), psm->GetFrom(), &psm->GetParentElementIDMap()};
  }
#endif
  MFEM_ABORT("SubMeshDofInjection: sub space is not on a (Par)SubMesh.");
  return {nullptr, mfem::SubMesh::From::Domain, nullptr};
}

}  // namespace

SubMeshDofInjection::SubMeshDofInjection(
    const mfem::FiniteElementSpace& sub_fes,
    const mfem::FiniteElementSpace& parent_fes)
    : mfem::Operator(parent_fes.GetVSize(), sub_fes.GetVSize()),
      sub_fes_(&sub_fes),
      parent_fes_(&parent_fes) {
  using namespace mfem;

  auto info = GetSubMeshInfo(sub_fes.GetMesh());
  MFEM_VERIFY(info.parent == parent_fes.GetMesh(),
              "SubMeshDofInjection: parent_fes is not defined on the parent "
              "mesh of the submesh.");
  MFEM_VERIFY(sub_fes.FEColl() == parent_fes.FEColl(),
              "SubMeshDofInjection: the sub space must share the parent "
              "space's FiniteElementCollection object (use MakeShadowSpace).");
  MFEM_VERIFY(sub_fes.GetVDim() == parent_fes.GetVDim() &&
                  sub_fes.GetOrdering() == parent_fes.GetOrdering(),
              "SubMeshDofInjection: vdim/ordering mismatch between sub and "
              "parent spaces.");
  MFEM_VERIFY(!sub_fes.IsVariableOrder() && !parent_fes.IsVariableOrder(),
              "SubMeshDofInjection: variable-order spaces are not supported.");
  const auto cont = parent_fes.FEColl()->GetContType();
  MFEM_VERIFY(cont == FiniteElementCollection::CONTINUOUS ||
                  cont == FiniteElementCollection::DISCONTINUOUS,
              "SubMeshDofInjection: only H1 and L2 collections are supported "
              "(H(div)/H(curl) need face-orientation corrections).");
  MFEM_VERIFY(
      parent_fes.GetMesh()->Conforming() && sub_fes.GetMesh()->Conforming(),
      "SubMeshDofInjection: nonconforming meshes are not supported.");

  Array<int> raw_map;
  SubMeshUtils::BuildVdofToVdofMap(sub_fes, parent_fes, info.from,
                                   *info.parent_element_ids, raw_map);

  const int n = sub_fes.GetVSize();
  parent_vdof_.SetSize(n);
  sign_.SetSize(n);
  for (int i = 0; i < n; i++) {
    real_t s = 1.0;
    parent_vdof_[i] = FiniteElementSpace::DecodeDof(raw_map[i], s);
    sign_[i] = s;
  }
}

std::unique_ptr<mfem::FiniteElementSpace> SubMeshDofInjection::MakeShadowSpace(
    const mfem::FiniteElementSpace& parent_fes, mfem::SubMesh& submesh) {
  return std::make_unique<mfem::FiniteElementSpace>(
      &submesh, parent_fes.FEColl(), parent_fes.GetVDim(),
      parent_fes.GetOrdering());
}

#ifdef MFEM_USE_MPI
std::unique_ptr<mfem::ParFiniteElementSpace>
SubMeshDofInjection::MakeShadowSpace(
    const mfem::ParFiniteElementSpace& parent_fes, mfem::ParSubMesh& submesh) {
  return std::make_unique<mfem::ParFiniteElementSpace>(
      &submesh, parent_fes.FEColl(), parent_fes.GetVDim(),
      parent_fes.GetOrdering());
}
#endif

void SubMeshDofInjection::Mult(const mfem::Vector& x, mfem::Vector& y) const {
  y = 0.0;
  for (int i = 0; i < width; i++) {
    y[parent_vdof_[i]] = sign_[i] * x[i];
  }
}

void SubMeshDofInjection::MultTranspose(const mfem::Vector& x,
                                        mfem::Vector& y) const {
  for (int i = 0; i < width; i++) {
    y[i] = sign_[i] * x[parent_vdof_[i]];
  }
}

std::unique_ptr<mfem::SparseMatrix> SubMeshDofInjection::NewSparseMatrix()
    const {
  using namespace mfem;
  const int m = height, n = width;

  int* I = new int[m + 1]();
  for (int i = 0; i < n; i++) {
    I[parent_vdof_[i] + 1]++;
  }
  for (int i = 0; i < m; i++) {
    I[i + 1] += I[i];
  }

  int* J = new int[n];
  real_t* data = new real_t[n];
  for (int i = 0; i < n; i++) {
    // Injective map: each parent row receives at most one entry, so I[row]
    // is its exact position.
    const int pos = I[parent_vdof_[i]];
    J[pos] = i;
    data[pos] = sign_[i];
  }

  return std::make_unique<SparseMatrix>(I, J, data, m, n, true, true, true);
}

std::unique_ptr<mfem::SparseMatrix> SubMeshDofInjection::RemapRows(
    const mfem::SparseMatrix& M) const {
  using namespace mfem;
  MFEM_VERIFY(M.Finalized(),
              "SubMeshDofInjection::RemapRows: M must be "
              "finalized.");
  MFEM_VERIFY(M.Height() == width,
              "SubMeshDofInjection::RemapRows: M must have SubVSize() rows.");

  const int* MI = M.GetI();
  const int* MJ = M.GetJ();
  const real_t* MA = M.GetData();
  const int nnz = MI[width];

  int* I = new int[height + 1]();
  for (int i = 0; i < width; i++) {
    I[parent_vdof_[i] + 1] = MI[i + 1] - MI[i];
  }
  for (int i = 0; i < height; i++) {
    I[i + 1] += I[i];
  }

  int* J = new int[nnz];
  real_t* data = new real_t[nnz];
  for (int i = 0; i < width; i++) {
    int pos = I[parent_vdof_[i]];
    for (int k = MI[i]; k < MI[i + 1]; k++, pos++) {
      J[pos] = MJ[k];
      data[pos] = sign_[i] * MA[k];
    }
  }

  return std::make_unique<SparseMatrix>(I, J, data, height, M.Width(), true,
                                        true, false);
}

std::unique_ptr<mfem::SparseMatrix> SubMeshDofInjection::RemapColumns(
    const mfem::SparseMatrix& M) const {
  using namespace mfem;
  MFEM_VERIFY(M.Finalized(),
              "SubMeshDofInjection::RemapColumns: M must be "
              "finalized.");
  MFEM_VERIFY(M.Width() == width,
              "SubMeshDofInjection::RemapColumns: M must "
              "have SubVSize() columns.");

  const int* MI = M.GetI();
  const int* MJ = M.GetJ();
  const real_t* MA = M.GetData();
  const int m = M.Height();
  const int nnz = MI[m];

  int* I = new int[m + 1];
  std::copy(MI, MI + m + 1, I);
  int* J = new int[nnz];
  real_t* data = new real_t[nnz];
  for (int k = 0; k < nnz; k++) {
    J[k] = parent_vdof_[MJ[k]];
    data[k] = sign_[MJ[k]] * MA[k];
  }

  auto res =
      std::make_unique<SparseMatrix>(I, J, data, m, height, true, true, false);
  res->SortColumnIndices();
  return res;
}

#ifdef MFEM_USE_MPI
std::unique_ptr<mfem::HypreParMatrix> SubMeshDofInjection::NewTrueDofMatrix()
    const {
  using namespace mfem;

  auto* sub_pfes = dynamic_cast<const ParFiniteElementSpace*>(sub_fes_);
  auto* parent_pfes = dynamic_cast<const ParFiniteElementSpace*>(parent_fes_);
  MFEM_VERIFY(sub_pfes && parent_pfes,
              "SubMeshDofInjection::NewTrueDofMatrix: both spaces must be "
              "ParFiniteElementSpaces.");

  // Build Pi^T (sub true dofs x parent true dofs) with exactly one +-1 per
  // row: the owner of each sub true dof reads off the *global* parent true
  // dof of its representative local vdof. GetGlobalTDofNumber is valid for
  // unowned parent ldofs too (conforming spaces), so it does not matter which
  // rank owns the parent dof.
  const int nrows = sub_pfes->TrueVSize();
  Array<int> I(nrows + 1);
  Array<HYPRE_BigInt> J(std::max(nrows, 1));
  Vector data(std::max(nrows, 1));
  J = HYPRE_BigInt(-1);

  for (int l = 0; l < width; l++) {
    const int lt = sub_pfes->GetLocalTDofNumber(l);
    if (lt < 0) {
      continue;
    }
    MFEM_ASSERT(J[lt] == -1, "duplicate true dof");
    J[lt] = parent_pfes->GetGlobalTDofNumber(parent_vdof_[l]);
    data[lt] = sign_[l];
  }
  for (int i = 0; i <= nrows; i++) {
    I[i] = i;
  }
  for (int i = 0; i < nrows; i++) {
    MFEM_VERIFY(J[i] >= 0,
                "SubMeshDofInjection::NewTrueDofMatrix: sub true "
                "dof without a parent image.");
  }

  HypreParMatrix Pi_t(
      sub_pfes->GetComm(), nrows, sub_pfes->GlobalTrueVSize(),
      parent_pfes->GlobalTrueVSize(), I.GetData(), J.GetData(), data.GetData(),
      const_cast<ParFiniteElementSpace*>(sub_pfes)->GetTrueDofOffsets(),
      const_cast<ParFiniteElementSpace*>(parent_pfes)->GetTrueDofOffsets());

  return std::unique_ptr<HypreParMatrix>(Pi_t.Transpose());
}
#endif

std::unique_ptr<mfem::SparseMatrix> NewSubMeshPairingMatrix(
    const SubMeshDofInjection& a, const SubMeshDofInjection& b) {
  using namespace mfem;
  MFEM_VERIFY(&a.ParentSpace() == &b.ParentSpace(),
              "NewSubMeshPairingMatrix: the two injections must target the "
              "same parent space.");
  // parent vdof -> a vdof; the injection is one-to-one into the parent.
  Array<int> to_a(a.ParentVSize());
  to_a = -1;
  for (int i = 0; i < a.SubVSize(); i++) {
    to_a[a.ParentVDofs()[i]] = i;
  }
  auto J = std::make_unique<SparseMatrix>(a.SubVSize(), b.SubVSize());
  for (int j = 0; j < b.SubVSize(); j++) {
    const int i = to_a[b.ParentVDofs()[j]];
    if (i >= 0) {
      J->Set(i, j, a.Signs()[i] * b.Signs()[j]);
    }
  }
  J->Finalize();
  return J;
}

#ifdef MFEM_USE_MPI
std::unique_ptr<mfem::HypreParMatrix> NewSubMeshPairingTrueDofMatrix(
    const SubMeshDofInjection& a, const SubMeshDofInjection& b) {
  using namespace mfem;
  MFEM_VERIFY(&a.ParentSpace() == &b.ParentSpace(),
              "NewSubMeshPairingTrueDofMatrix: the two injections must "
              "target the same parent space.");
  auto Pi_a = a.NewTrueDofMatrix();
  auto Pi_b = b.NewTrueDofMatrix();
  std::unique_ptr<HypreParMatrix> Pi_a_t(Pi_a->Transpose());
  return std::unique_ptr<HypreParMatrix>(ParMult(Pi_a_t.get(), Pi_b.get()));
}
#endif

namespace {

// True if `mesh` is a (Par)SubMesh whose parent is exactly `parent`.
bool IsSubMeshOf(const mfem::Mesh* mesh, const mfem::Mesh* parent) {
  if (auto* sm = dynamic_cast<const mfem::SubMesh*>(mesh)) {
    return sm->GetParent() == parent;
  }
#ifdef MFEM_USE_MPI
  if (auto* psm = dynamic_cast<const mfem::ParSubMesh*>(mesh)) {
    return psm->GetParent() == parent;
  }
#endif
  return false;
}

// The shadow of parent_fes on submesh, serial or parallel as appropriate.
std::unique_ptr<mfem::FiniteElementSpace> MakeShadow(
    const mfem::FiniteElementSpace& parent_fes, mfem::Mesh* submesh) {
#ifdef MFEM_USE_MPI
  if (auto* psm = dynamic_cast<mfem::ParSubMesh*>(submesh)) {
    auto* parent_pfes =
        dynamic_cast<const mfem::ParFiniteElementSpace*>(&parent_fes);
    MFEM_VERIFY(parent_pfes,
                "SubMeshMixedBilinearForm: a ParSubMesh requires a "
                "ParFiniteElementSpace on the parent side.");
    return SubMeshDofInjection::MakeShadowSpace(*parent_pfes, *psm);
  }
#endif
  auto* sm = dynamic_cast<mfem::SubMesh*>(submesh);
  MFEM_VERIFY(sm, "SubMeshMixedBilinearForm: not a SubMesh.");
  return SubMeshDofInjection::MakeShadowSpace(parent_fes, *sm);
}

}  // namespace

detail::SubMeshFormSetup::SubMeshFormSetup(mfem::FiniteElementSpace* trial_fes,
                                           mfem::FiniteElementSpace* test_fes) {
  const bool trial_is_sub =
      IsSubMeshOf(trial_fes->GetMesh(), test_fes->GetMesh());
  const bool test_is_sub =
      IsSubMeshOf(test_fes->GetMesh(), trial_fes->GetMesh());
  MFEM_VERIFY(trial_is_sub != test_is_sub,
              "SubMeshMixedBilinearForm: exactly one of the two spaces must "
              "live on a SubMesh of the other's mesh (use MixedBilinearForm "
              "when both spaces share a mesh).");
  parent_is_trial = test_is_sub;

  auto* parent_fes = parent_is_trial ? trial_fes : test_fes;
  auto* submesh = (parent_is_trial ? test_fes : trial_fes)->GetMesh();
  shadow = MakeShadow(*parent_fes, submesh);
  injection = std::make_unique<SubMeshDofInjection>(*shadow, *parent_fes);
}

std::unique_ptr<mfem::SparseMatrix> detail::AssembleOnSubMesh(
    mfem::MixedBilinearForm& form, const SubMeshFormSetup& setup,
    int skip_zeros) {
  using namespace mfem;
  MFEM_VERIFY(form.GetFBFI()->Size() == 0,
              "SubMeshMixedBilinearForm: interior-face integrators are not "
              "supported.");

  auto* shadow = setup.shadow.get();
  // Borrows form's domain, boundary, trace-face and boundary-trace-face
  // integrators and markers (extern_bfs = 1: it does not delete them).
  MixedBilinearForm helper(setup.parent_is_trial ? shadow : form.TrialFESpace(),
                           setup.parent_is_trial ? form.TestFESpace() : shadow,
                           &form);
  // The borrowing constructor omits boundary-face integrators (MFEM 4.9).
  *helper.GetBFBFI() = *form.GetBFBFI();
  *helper.GetBFBFI_Marker() = *form.GetBFBFI_Marker();

  helper.Assemble(skip_zeros);
  helper.Finalize(skip_zeros);

  const auto& injection = *setup.injection;
  return setup.parent_is_trial ? injection.RemapColumns(helper.SpMat())
                               : injection.RemapRows(helper.SpMat());
}

SubMeshMixedBilinearForm::SubMeshMixedBilinearForm(
    mfem::FiniteElementSpace* trial_fes, mfem::FiniteElementSpace* test_fes)
    : mfem::MixedBilinearForm(trial_fes, test_fes),
      setup_(trial_fes, test_fes) {}

void SubMeshMixedBilinearForm::Assemble(int skip_zeros) {
  MFEM_VERIFY(assembly == mfem::AssemblyLevel::LEGACY && !ext,
              "SubMeshMixedBilinearForm: only AssemblyLevel::LEGACY is "
              "supported.");
  auto m = detail::AssembleOnSubMesh(*this, setup_, skip_zeros);
  delete mat;
  mat = m.release();
  delete mat_e;
  mat_e = nullptr;
}

#ifdef MFEM_USE_MPI
ParSubMeshMixedBilinearForm::ParSubMeshMixedBilinearForm(
    mfem::ParFiniteElementSpace* trial_fes,
    mfem::ParFiniteElementSpace* test_fes)
    : mfem::ParMixedBilinearForm(trial_fes, test_fes),
      setup_(trial_fes, test_fes) {}

void ParSubMeshMixedBilinearForm::Assemble(int skip_zeros) {
  MFEM_VERIFY(assembly == mfem::AssemblyLevel::LEGACY && !ext,
              "ParSubMeshMixedBilinearForm: only AssemblyLevel::LEGACY is "
              "supported.");
  auto m = detail::AssembleOnSubMesh(*this, setup_, skip_zeros);
  delete mat;
  mat = m.release();
  delete mat_e;
  mat_e = nullptr;
  p_mat.Clear();
  p_mat_e.Clear();
}
#endif

}  // namespace AdGIA
