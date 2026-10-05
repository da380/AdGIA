#include "SubMeshTestCommon.hpp"

/*
  Tests for SubMeshDofInjection (serial): the operator against its explicit
  matrix, the algebraic identities of an injection, agreement with
  SubMesh::Transfer, and re-indexing against explicit sparse products.

  Configurations sweep dimension, element type, order, H1/L2, vdim and
  dof ordering. The parent mesh is a Cartesian mesh split into two
  attributes; the submesh is the attribute-2 half (From::Domain) or a
  section of the outer boundary (From::Boundary).
*/

namespace {

// (dim, elementType, order, useL2, useVectorSpace, byVDIM)
using InjectionParam = std::tuple<int, int, int, bool, bool, bool>;

class SubMeshDofInjectionTest : public testing::TestWithParam<InjectionParam> {
 protected:
  void SetUp() override {
    auto [dim_, elementType, order, useL2, vector, byVDIM] = GetParam();
    dim = dim_;
    mesh = std::make_unique<Mesh>(MakeTwoAttributeMesh(dim, elementType));
    if (useL2) {
      fec = std::make_unique<L2_FECollection>(order, dim,
                                              BasisType::GaussLobatto);
    } else {
      fec = std::make_unique<H1_FECollection>(order, dim);
    }
    vdim = vector ? dim : 1;
    ordering = byVDIM ? Ordering::byVDIM : Ordering::byNODES;
    parent_fes = std::make_unique<FiniteElementSpace>(mesh.get(), fec.get(),
                                                      vdim, ordering);
  }

  // The checks shared by the domain and boundary variants.
  void CheckInjection(SubMesh& submesh) {
    auto shadow = SubMeshDofInjection::MakeShadowSpace(*parent_fes, submesh);
    auto injection = SubMeshDofInjection(*shadow, *parent_fes);

    const auto n = injection.SubVSize();
    const auto m = injection.ParentVSize();
    ASSERT_EQ(n, shadow->GetVSize());
    ASSERT_EQ(m, parent_fes->GetVSize());
    ASSERT_GT(n, 0);
    ASSERT_LT(n, m);

    auto P = injection.NewSparseMatrix();

    // Operator Mult/MultTranspose agree with the explicit matrix.
    auto x = RandomVector(n);
    auto w = RandomVector(m);
    Vector y1(m), y2(m), z1(n), z2(n);
    injection.Mult(x, y1);
    P->Mult(x, y2);
    y2 -= y1;
    EXPECT_EQ(y2.Normlinf(), 0.0);
    injection.MultTranspose(w, z1);
    P->MultTranspose(w, z2);
    z2 -= z1;
    EXPECT_EQ(z2.Normlinf(), 0.0);

    // P^T P = I exactly (entries are +-1).
    Vector xr(n);
    injection.Mult(x, y1);
    injection.MultTranspose(y1, xr);
    xr -= x;
    EXPECT_EQ(xr.Normlinf(), 0.0);

    // P P^T is idempotent.
    Vector w1(m), w2(m);
    injection.MultTranspose(w, z1);
    injection.Mult(z1, w1);
    injection.MultTranspose(w1, z2);
    injection.Mult(z2, w2);
    w2 -= w1;
    EXPECT_EQ(w2.Normlinf(), 0.0);

    // MultTranspose of a projected analytic parent function equals
    // SubMesh::Transfer of it into the shadow space.
    auto f = FunctionCoefficient(TestFunction);
    auto fv =
        VectorFunctionCoefficient(vdim, [this](const Vector& x, Vector& u) {
          u.SetSize(vdim);
          for (auto i = 0; i < vdim; i++) {
            u[i] = (i + 1.0) * TestFunction(x);
          }
        });
    auto pgf = GridFunction(parent_fes.get());
    if (vdim == 1) {
      pgf.ProjectCoefficient(f);
    } else {
      pgf.ProjectCoefficient(fv);
    }
    auto sgf = GridFunction(shadow.get());
    SubMesh::Transfer(pgf, sgf);
    Vector sv(n);
    injection.MultTranspose(pgf, sv);
    sv -= sgf;
    EXPECT_LT(sv.Normlinf(), 1e-13);

    // Re-indexing equals the explicit sparse products: M a mass matrix on
    // the submesh with a non-trivial coefficient.
    auto M = SparseMatrix();
    {
      auto blf = BilinearForm(shadow.get());
      auto rho = FunctionCoefficient(
          [](const Vector& x) { return 1.0 + x[0] + 0.5 * x[x.Size() - 1]; });
      if (vdim == 1) {
        blf.AddDomainIntegrator(new MassIntegrator(rho));
      } else {
        blf.AddDomainIntegrator(new VectorMassIntegrator(rho));
      }
      blf.Assemble();
      blf.Finalize();
      M.Swap(blf.SpMat());
    }

    auto PM = injection.RemapRows(M);
    std::unique_ptr<SparseMatrix> PM_ref(mfem::Mult(*P, M));
    EXPECT_EQ(MaxDiff(*PM, *PM_ref), 0.0);

    auto MPt = injection.RemapColumns(M);
    std::unique_ptr<SparseMatrix> Pt(mfem::Transpose(*P));
    std::unique_ptr<SparseMatrix> MPt_ref(mfem::Mult(M, *Pt));
    EXPECT_EQ(MaxDiff(*MPt, *MPt_ref), 0.0);
  }

  int dim = 0, vdim = 1;
  Ordering::Type ordering = Ordering::byNODES;
  std::unique_ptr<Mesh> mesh;
  std::unique_ptr<FiniteElementCollection> fec;
  std::unique_ptr<FiniteElementSpace> parent_fes;
};

TEST_P(SubMeshDofInjectionTest, DomainSubMesh) {
  auto attrs = Array<int>({2});
  auto submesh = SubMesh::CreateFromDomain(*mesh, attrs);
  CheckInjection(submesh);
}

TEST_P(SubMeshDofInjectionTest, BoundarySubMesh) {
  // Trace shadows are only supported for H1 here (the L2 path requires
  // trace-element machinery restricted to GaussLobatto, which the library
  // does not use).
  auto [dim_, elementType, order, useL2, vector, byVDIM] = GetParam();
  if (useL2) {
    GTEST_SKIP();
  }
  auto attrs = Array<int>({1, 2});
  auto submesh = SubMesh::CreateFromBoundary(*mesh, attrs);
  CheckInjection(submesh);
}

INSTANTIATE_TEST_SUITE_P(SubMeshDofInjection, SubMeshDofInjectionTest,
                         testing::Combine(testing::Values(2, 3),
                                          testing::Values(0, 1),
                                          testing::Values(1, 2, 3),
                                          testing::Bool(), testing::Bool(),
                                          testing::Bool()));

}  // namespace
