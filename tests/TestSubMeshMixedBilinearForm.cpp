#include "SubMeshTestCommon.hpp"

/*
  Tests for SubMeshMixedBilinearForm (serial).

  The parent mesh is a Cartesian mesh split into two attributes (2 for
  x < 0.5, 1 for x > 0.5); the domain submesh is the attribute-2 half, the
  boundary submesh is the x = 0 face. The parent-side space is always
  scalar H1 of the given order; the submesh-side space is one of three
  families (scalar H1 with its own collection, vector H1 byVDIM, scalar
  L2 of order - 1). Both orientations (parent space as trial or as test)
  are exercised.

  References:
  - MFEM's own MixedBilinearForm on the *parent* mesh, with the same
    integrators restricted by attribute markers to the submesh region and
    to an exterior boundary section, re-indexed with the explicit
    injection matrix. Exact algebra, so agreement to round-off.
  - SubMesh::Transfer applied to random vectors, against a plain
    MixedBilinearForm on the submesh, for integrators on the internal
    interface (the cut) and on a boundary submesh, where no parent-mesh
    boundary elements exist.
  - An analytic surface integral for the boundary submesh.
*/

namespace {

// (dim, elementType, order, parentIsTrial, subFamily)
using FormParam = std::tuple<int, int, int, bool, int>;

constexpr double kTol = 1e-12;

// Check that two spaces on the same mesh use identical element vdofs.
void ExpectSameElementVDofs(const FiniteElementSpace& a,
                            const FiniteElementSpace& b) {
  ASSERT_EQ(a.GetVSize(), b.GetVSize());
  Array<int> va, vb;
  for (auto e = 0; e < a.GetNE(); e++) {
    a.GetElementVDofs(e, va);
    b.GetElementVDofs(e, vb);
    ASSERT_EQ(va.Size(), vb.Size());
    for (auto i = 0; i < va.Size(); i++) {
      ASSERT_EQ(va[i], vb[i]);
    }
  }
}

double MaxAbs(const SparseMatrix& A) { return A.MaxNorm(); }

class SubMeshMixedBilinearFormTest : public testing::TestWithParam<FormParam> {
 protected:
  void SetUp() override {
    std::tie(dim, elementType, order, parentIsTrial, family) = GetParam();
    mesh = std::make_unique<Mesh>(MakeTwoAttributeMesh(dim, elementType));
    x0_attr = BdrAttributeAtXZero(*mesh);
    ASSERT_GT(x0_attr, 0);
    parent_fec = std::make_unique<H1_FECollection>(order, dim);
    parent_fes =
        std::make_unique<FiniteElementSpace>(mesh.get(), parent_fec.get());
    rho = std::make_unique<FunctionCoefficient>(
        [](const Vector& x) { return 1.0 + x[0] + 0.5 * x[x.Size() - 1]; });
    sigma = std::make_unique<FunctionCoefficient>(
        [](const Vector& x) { return 2.0 - x[x.Size() - 1]; });
  }

  // The submesh-side space for the current family on `m` (which may be the
  // parent mesh itself, for the reference assembly). Owns its collection.
  std::unique_ptr<FiniteElementSpace> MakeSubSpace(Mesh& m) {
    const auto mdim = m.Dimension();
    if (family == 2) {
      fecs.push_back(std::make_unique<L2_FECollection>(order - 1, mdim));
      return std::make_unique<FiniteElementSpace>(&m, fecs.back().get());
    }
    fecs.push_back(std::make_unique<H1_FECollection>(order, mdim));
    if (family == 1) {
      // vdim = the mesh's own dimension: GradientIntegrator sizes its
      // element matrix by the test element's reference dimension.
      return std::make_unique<FiniteElementSpace>(&m, fecs.back().get(), mdim,
                                                  Ordering::byVDIM);
    }
    return std::make_unique<FiniteElementSpace>(&m, fecs.back().get());
  }

  // Add the family's integrators to `f`, whose trial/test spaces are
  // (parent-like, sub-like) or (sub-like, parent-like) according to
  // parentIsTrial. A null marker adds no integrator of that kind.
  void AddIntegrators(MixedBilinearForm& f, Array<int>* dom_marker,
                      Array<int>* bdr_marker) {
    if (dom_marker) {
      BilinearFormIntegrator* integ = nullptr;
      if (family == 1) {
        integ = parentIsTrial
                    ? static_cast<BilinearFormIntegrator*>(
                          new GradientIntegrator(*rho))
                    : new TransposeIntegrator(new GradientIntegrator(*rho));
      } else {
        integ = new MassIntegrator(*rho);
      }
      f.AddDomainIntegrator(integ, *dom_marker);
    }
    // Boundary integrators only for the scalar H1 family: the vector
    // family has no scalar/vector boundary integrator in MFEM, and L2
    // spaces carry no boundary dofs.
    if (bdr_marker && family == 0) {
      f.AddBoundaryIntegrator(new BoundaryMassIntegrator(*sigma), *bdr_marker);
    }
  }

  static Array<int> Marker(int size, int attr) {
    Array<int> m(size);
    m = 0;
    m[attr - 1] = 1;
    return m;
  }

  Vector RandomVectorFor(const FiniteElementSpace& fes) {
    return RandomVector(fes.GetVSize());
  }

  int dim = 0, elementType = 0, order = 1, family = 0, x0_attr = -1;
  bool parentIsTrial = true;
  std::unique_ptr<Mesh> mesh;
  std::unique_ptr<FiniteElementCollection> parent_fec;
  std::unique_ptr<FiniteElementSpace> parent_fes;
  std::vector<std::unique_ptr<FiniteElementCollection>> fecs;
  std::unique_ptr<Coefficient> rho, sigma;
};

// Domain integrator on the submesh region plus a boundary integrator on an
// exterior boundary section, versus MFEM's MixedBilinearForm on the parent
// mesh with the same markers, re-indexed.
TEST_P(SubMeshMixedBilinearFormTest, MatchesParentAssembly) {
  auto attrs = Array<int>({2});
  auto submesh = SubMesh::CreateFromDomain(*mesh, attrs);

  auto B_parent = MakeSubSpace(*mesh);
  auto B_sub = MakeSubSpace(submesh);
  auto B_shadow = SubMeshDofInjection::MakeShadowSpace(*B_parent, submesh);
  ExpectSameElementVDofs(*B_sub, *B_shadow);
  auto PB = SubMeshDofInjection(*B_shadow, *B_parent).NewSparseMatrix();

  auto dom_parent = Marker(mesh->attributes.Max(), 2);
  auto dom_sub = Marker(submesh.attributes.Max(), 2);
  auto bdr_parent = Marker(mesh->bdr_attributes.Max(), x0_attr);
  auto bdr_sub = Marker(submesh.bdr_attributes.Max(), x0_attr);

  // Reference on the parent mesh.
  auto ref =
      MixedBilinearForm(parentIsTrial ? parent_fes.get() : B_parent.get(),
                        parentIsTrial ? B_parent.get() : parent_fes.get());
  AddIntegrators(ref, &dom_parent, &bdr_parent);
  ref.Assemble();
  ref.Finalize();
  std::unique_ptr<SparseMatrix> expected(parentIsTrial
                                             ? TransposeMult(*PB, ref.SpMat())
                                             : mfem::Mult(ref.SpMat(), *PB));

  // The class under test.
  auto form =
      SubMeshMixedBilinearForm(parentIsTrial ? parent_fes.get() : B_sub.get(),
                               parentIsTrial ? B_sub.get() : parent_fes.get());
  EXPECT_EQ(form.ParentIsTrial(), parentIsTrial);
  EXPECT_EQ(form.Injection().ParentVSize(), parent_fes->GetVSize());
  EXPECT_EQ(form.Injection().SubVSize(), form.ShadowSpace().GetVSize());
  AddIntegrators(form, &dom_sub, &bdr_sub);
  form.Assemble();
  form.Finalize();

  EXPECT_EQ(form.Height(), expected->Height());
  EXPECT_EQ(form.Width(), expected->Width());
  const auto scale = MaxAbs(*expected);
  ASSERT_GT(scale, 0.0);
  EXPECT_LT(MaxDiff(form.SpMat(), *expected), kTol * scale);

  // Assemble() replaces rather than accumulates.
  form.Assemble();
  form.Finalize();
  EXPECT_LT(MaxDiff(form.SpMat(), *expected), kTol * scale);
}

// Internal interface: the boundary integrator acts on the cut, which is a
// boundary of the submesh but not of the parent mesh. Reference: a plain
// MixedBilinearForm on the submesh with the shadow space, connected to the
// parent through SubMesh::Transfer.
TEST_P(SubMeshMixedBilinearFormTest, InternalInterfaceViaTransfer) {
  auto attrs = Array<int>({2});
  auto submesh = SubMesh::CreateFromDomain(*mesh, attrs);
  const auto cut_attr = mesh->bdr_attributes.Max() + 1;
  ASSERT_EQ(submesh.bdr_attributes.Max(), cut_attr);

  auto B_sub = MakeSubSpace(submesh);
  auto shadow = SubMeshDofInjection::MakeShadowSpace(*parent_fes, submesh);
  auto dom_sub = Marker(submesh.attributes.Max(), 2);
  auto bdr_sub = Marker(submesh.bdr_attributes.Max(), cut_attr);

  auto ref = MixedBilinearForm(parentIsTrial ? shadow.get() : B_sub.get(),
                               parentIsTrial ? B_sub.get() : shadow.get());
  AddIntegrators(ref, &dom_sub, &bdr_sub);
  ref.Assemble();
  ref.Finalize();

  auto form =
      SubMeshMixedBilinearForm(parentIsTrial ? parent_fes.get() : B_sub.get(),
                               parentIsTrial ? B_sub.get() : parent_fes.get());
  AddIntegrators(form, &dom_sub, &bdr_sub);
  form.Assemble();
  form.Finalize();

  for (auto trial = 0; trial < 3; trial++) {
    if (parentIsTrial) {
      // form x == ref (x restricted to the shadow).
      auto x = RandomVectorFor(*parent_fes);
      GridFunction xg(parent_fes.get());
      xg = x;
      GridFunction xs(shadow.get());
      SubMesh::Transfer(xg, xs);
      Vector y1(form.Height()), y2(ref.Height());
      form.Mult(x, y1);
      ref.Mult(xs, y2);
      y1 -= y2;
      EXPECT_LT(y1.Normlinf(), kTol * (y2.Normlinf() + 1.0));
    } else {
      // form x == (ref x) prolonged by zero to the parent.
      auto x = RandomVectorFor(*B_sub);
      Vector y1(form.Height());
      form.Mult(x, y1);
      GridFunction ys(shadow.get());
      ref.Mult(x, ys);
      GridFunction yp(parent_fes.get());
      yp = 0.0;
      SubMesh::Transfer(ys, yp);
      y1 -= yp;
      EXPECT_LT(y1.Normlinf(), kTol * (yp.Normlinf() + 1.0));
    }
  }
}

// A boundary submesh (From::Boundary). The parent-side shadow is
// the trace space. Checked against SubMesh::Transfer as above and, for the
// scalar family, against an analytic surface integral through a
// BoundaryLFIntegrator on the parent.
TEST_P(SubMeshMixedBilinearFormTest, BoundarySubMesh) {
  if (family == 1) {
    // GradientIntegrator does not support manifold elements (it sizes the
    // Jacobian adjugate by the reference dimension), so there is no MFEM
    // scalar/vector integrator to use here.
    GTEST_SKIP();
  }
  auto attrs = Array<int>({x0_attr});
  auto sigma_mesh = SubMesh::CreateFromBoundary(*mesh, attrs);
  ASSERT_EQ(sigma_mesh.Dimension(), dim - 1);

  auto B_sub = MakeSubSpace(sigma_mesh);
  auto shadow = SubMeshDofInjection::MakeShadowSpace(*parent_fes, sigma_mesh);
  auto dom_sub = Marker(sigma_mesh.attributes.Max(), x0_attr);

  auto ref = MixedBilinearForm(parentIsTrial ? shadow.get() : B_sub.get(),
                               parentIsTrial ? B_sub.get() : shadow.get());
  AddIntegrators(ref, &dom_sub, nullptr);
  ref.Assemble();
  ref.Finalize();

  auto form =
      SubMeshMixedBilinearForm(parentIsTrial ? parent_fes.get() : B_sub.get(),
                               parentIsTrial ? B_sub.get() : parent_fes.get());
  AddIntegrators(form, &dom_sub, nullptr);
  form.Assemble();
  form.Finalize();

  if (parentIsTrial) {
    auto x = RandomVectorFor(*parent_fes);
    GridFunction xg(parent_fes.get());
    xg = x;
    GridFunction xs(shadow.get());
    SubMesh::Transfer(xg, xs);
    Vector y1(form.Height()), y2(ref.Height());
    form.Mult(x, y1);
    ref.Mult(xs, y2);
    y1 -= y2;
    EXPECT_LT(y1.Normlinf(), kTol * (y2.Normlinf() + 1.0));
  } else {
    auto x = RandomVectorFor(*B_sub);
    Vector y1(form.Height());
    form.Mult(x, y1);
    GridFunction ys(shadow.get());
    ref.Mult(x, ys);
    GridFunction yp(parent_fes.get());
    yp = 0.0;
    SubMesh::Transfer(ys, yp);
    y1 -= yp;
    EXPECT_LT(y1.Normlinf(), kTol * (yp.Normlinf() + 1.0));
  }

  if (family != 0) {
    return;
  }

  // Analytic check: f, g polynomials of degree <= order are represented
  // exactly by nodal H1 interpolation, so g^T A f == \int_Sigma rho f g.
  const auto p = order;
  auto f = FunctionCoefficient([p](const Vector& x) {
    auto v = 1.0;
    for (auto i = 0; i < x.Size(); i++) {
      v += (i + 1.0) * std::pow(x[i], p);
    }
    return v;
  });
  auto g = FunctionCoefficient([p](const Vector& x) {
    auto v = 0.5;
    for (auto i = 0; i < x.Size(); i++) {
      v += (2.0 - i) * std::pow(x[i], p) + x[i];
    }
    return v;
  });
  GridFunction fg(parent_fes.get()), gg(B_sub.get());
  fg.ProjectCoefficient(f);
  gg.ProjectCoefficient(g);

  // Same form with a quadrature rule exact for the degree 2p + 1 integrand.
  const auto& ir =
      IntRules.Get(sigma_mesh.GetElementGeometry(0), 2 * order + 4);
  auto aform =
      SubMeshMixedBilinearForm(parentIsTrial ? parent_fes.get() : B_sub.get(),
                               parentIsTrial ? B_sub.get() : parent_fes.get());
  aform.AddDomainIntegrator(new MassIntegrator(*rho, &ir), dom_sub);
  aform.Assemble();
  aform.Finalize();

  Vector Af(aform.Height());
  double lhs = 0.0;
  if (parentIsTrial) {
    aform.Mult(fg, Af);
    lhs = InnerProduct(gg, Af);
  } else {
    aform.Mult(gg, Af);
    lhs = InnerProduct(fg, Af);
  }

  auto rfg = ProductCoefficient(*rho, f);
  auto rfgg = ProductCoefficient(rfg, g);
  auto bdr_parent = Marker(mesh->bdr_attributes.Max(), x0_attr);
  LinearForm lf(parent_fes.get());
  lf.AddBoundaryIntegrator(new BoundaryLFIntegrator(rfgg, 2, 4), bdr_parent);
  lf.Assemble();
  GridFunction one(parent_fes.get());
  one = 1.0;
  const auto rhs = lf(one);

  EXPECT_NEAR(lhs, rhs, 1e-11 * std::abs(rhs));
}

INSTANTIATE_TEST_SUITE_P(SubMeshMixedBilinearForm, SubMeshMixedBilinearFormTest,
                         testing::Combine(testing::Values(2, 3),
                                          testing::Values(0, 1),
                                          testing::Values(1, 2, 3),
                                          testing::Bool(),
                                          testing::Values(0, 1, 2)));

}  // namespace
