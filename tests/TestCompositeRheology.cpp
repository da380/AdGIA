#include "QuasiStaticTestCommon.hpp"
#include "MixedProblemTestCommon.hpp"
#include "TestCommon.hpp"

/*
  Tests for CompositeRheology, against single global rheologies with
  piecewise coefficients. The unit square/cube of the quasi-static tests,
  loaded by the uniaxial traction on x = 0 and x = 1, is split into two
  attribute regions (x < 0.4 and x > 0.4).

  1. A Maxwell bar with the same rheology in both regions equals the unsplit
     bar to round-off for the exponential trapezoid and backward Euler:
     displacement, and each region's internal variable; the state is the
     unsplit body's, each branch living on its region only.
  2. Masking equivalence: an elastic region beside a Maxwell region equals
     the global Maxwell body whose branch modulus vanishes in the elastic
     region (PWCoefficient); a one-branch region beside a two-branch region
     equals the global two-branch body whose second modulus vanishes in the
     first region.
  3. Stiffness: two elastic regions assemble the matrix of one isotropic
     elastic rheology with piecewise moduli; two Maxwell regions with
     relaxation weights assemble the matrix of the piecewise Maxwell body
     with a piecewise weight.
  4. An isotropic Maxwell region beside an anisotropic (deviatoric Maxwell,
     isotropic tensor) region uses full internal variables and reproduces the
     all-isotropic composite.
  5. Smoke: a three-layer self-gravitating body with an elastic inner core,
     a fluid outer core and a Maxwell mantle steps through the viscoelastic
     operator, and the mantle relaxes under the sustained load.
  Plus the bookkeeping: branch numbering, labels, region lookup, the
  masked branch data (relaxation time and branch moduli outside a branch's
  region), and the construction checks (overlap, coverage).
*/

namespace {

using namespace elastic_test;

using Param = std::tuple<int, int, int>;  // (dim, elementType, order)

constexpr double kSplit = 0.4;

// Attribute 1 for x < kSplit, 2 otherwise.
void SplitAttributes(Mesh& mesh) {
  Vector c;
  for (int e = 0; e < mesh.GetNE(); e++) {
    mesh.GetElementCenter(e, c);
    mesh.SetAttribute(e, c[0] < kSplit ? 1 : 2);
  }
  mesh.SetAttributes();
}

Array<int> Region(int attr) {
  Array<int> m(2);
  m = 0;
  m[attr - 1] = 1;
  return m;
}

void ConstantUniaxial(const Vector& x, Vector& f) {
  UniaxialTraction(x, 0.0, f);
}

double RelMaxDiff(const Vector& a, const Vector& b) {
  Vector d(a);
  d -= b;
  return d.Normlinf() / (b.Normlinf() + 1e-300);
}

// Branch k of the state m scattered to the full internal layout.
Vector Full(const ViscoelasticOperator& op, const Vector& m, int k) {
  Vector full;
  op.BranchToFull(m, k, full);
  return full;
}

// Number of scalar internal nodes on the elements with attribute `attr`.
int NodesInRegion(ViscoelasticOperator& op, int attr) {
  auto& sfes = op.InternalScalarSpace();
  int n = 0;
  for (int e = 0; e < sfes.GetNE(); e++) {
    if (sfes.GetMesh()->GetAttribute(e) == attr) {
      n += sfes.GetFE(e)->GetDof();
    }
  }
  return n;
}

// Max |a - b| over the internal nodes of the elements with attribute
// `attr`, for two full-layout (nc x nd, byNODES) branch vectors, relative
// to max |b| there.
double RelMaxDiffInRegion(ViscoelasticOperator& op, const Vector& a,
                          const Vector& b, int attr) {
  auto& sfes = op.InternalScalarSpace();
  Mesh* mesh = sfes.GetMesh();
  const int nd = sfes.GetVSize(), nc = op.NumComponents();
  double diff = 0.0, scale = 0.0;
  Array<int> dofs;
  for (int e = 0; e < mesh->GetNE(); e++) {
    if (mesh->GetAttribute(e) != attr) {
      continue;
    }
    sfes.GetElementDofs(e, dofs);
    for (int p : dofs) {
      for (int c = 0; c < nc; c++) {
        diff = std::max(diff, std::abs(a[c * nd + p] - b[c * nd + p]));
        scale = std::max(scale, std::abs(b[c * nd + p]));
      }
    }
  }
  return diff / (scale + 1e-300);
}

std::unique_ptr<ODESolver> MakeScheme(int scheme) {
  if (scheme == 0) {
    return std::make_unique<ExponentialTrapezoidSolver>();
  }
  return std::make_unique<BackwardEulerSolver>();
}

class CompositeRheologyTest : public testing::TestWithParam<Param> {
 protected:
  void SetUp() override {
    std::tie(dim, elementType, order) = GetParam();
    mesh = std::make_unique<Mesh>(MakeSmallMesh(dim, elementType));
    SplitAttributes(*mesh);
    ASSERT_EQ(mesh->attributes.Max(), 2);
    fec = std::make_unique<H1_FECollection>(order, dim);
    fes = std::make_unique<FiniteElementSpace>(mesh.get(), fec.get(), dim);
    x0_attr = BdrAttributeAt(*mesh, 0, 0.0);
    x1_attr = BdrAttributeAt(*mesh, 0, 1.0);
    marker = Marker(mesh->bdr_attributes.Max(), {x0_attr, x1_attr});
    traction = std::make_unique<VectorFunctionCoefficient>(dim,
                                                           ConstantUniaxial);
  }

  // Piecewise-constant coefficient: v1 on attribute 1, v2 on attribute 2.
  std::unique_ptr<PWCoefficient> Piecewise(double v1, double v2) {
    owned.push_back(std::make_unique<ConstantCoefficient>(v1));
    owned.push_back(std::make_unique<ConstantCoefficient>(v2));
    Array<int> attrs({1, 2});
    Array<Coefficient*> coefs({owned[owned.size() - 2].get(),
                               owned.back().get()});
    return std::make_unique<PWCoefficient>(attrs, coefs);
  }

  // Run `steps` steps of size `dt` of `scheme` (0 exponential trapezoid,
  // 1 backward Euler) from the zero state; returns the state and leaves
  // the displacement of the final state in the operator's problem.
  Vector Run(ViscoelasticOperator& visco, int scheme, int steps, double dt) {
    auto ode = MakeScheme(scheme);
    ode->Init(visco);
    Vector m(visco.Height());
    m = 0.0;
    double t = 0.0;
    for (int s = 0; s < steps; s++) {
      ode->Step(m, t, dt);
    }
    EXPECT_TRUE(visco.SolveElastic(m, t));
    return m;
  }

  int dim = 2, elementType = 0, order = 1, x0_attr = -1, x1_attr = -1;
  std::unique_ptr<Mesh> mesh;
  std::unique_ptr<FiniteElementCollection> fec;
  std::unique_ptr<FiniteElementSpace> fes;
  Array<int> marker;
  std::unique_ptr<VectorCoefficient> traction;
  std::vector<std::unique_ptr<Coefficient>> owned;
  ConstantCoefficient kappa{2.1}, mu{0.8}, mu2{0.5}, tau{1.0}, tau2{3.0};
};

TEST_P(CompositeRheologyTest, Bookkeeping) {
  IsotropicElasticRheology elastic(dim, kappa, mu);
  auto maxwell = IsotropicMaxwellRheology::Maxwell(dim, kappa, mu, tau);
  std::vector<MaxwellBranch> two{{&mu, &tau}, {&mu2, &tau2}};
  IsotropicMaxwellRheology general(dim, kappa, mu, two);

  CompositeRheology c(dim, {{Region(1), &maxwell, "core"},
                            {Region(2), &general, ""}});
  EXPECT_EQ(c.NumRegions(), 2);
  EXPECT_EQ(c.NumBranches(), 3);
  EXPECT_TRUE(c.TraceFreeInternalVariables());
  EXPECT_TRUE(c.IsLinear());
  EXPECT_EQ(c.RegionOf(1), 0);
  EXPECT_EQ(c.RegionOf(2), 1);
  EXPECT_EQ(c.RegionOf(3), -1);
  EXPECT_EQ(c.RegionOf(0), -1);
  EXPECT_EQ(c.RegionBranchOffset(0), 0);
  EXPECT_EQ(c.RegionBranchOffset(1), 1);
  const int region[] = {0, 1, 1}, local[] = {0, 0, 1};
  for (int k = 0; k < 3; k++) {
    EXPECT_EQ(c.BranchRegion(k), region[k]);
    EXPECT_EQ(c.LocalBranch(k), local[k]);
    EXPECT_EQ(&c.RelaxationTime(k), &c.RelaxationTime(k));
  }
  EXPECT_EQ(c.BranchLabel(0), "core_branch0");
  EXPECT_EQ(c.BranchLabel(1), "region1_branch0");
  EXPECT_EQ(c.BranchLabel(2), "region1_branch1");
  EXPECT_EQ(c.RegionName(0), "core");
  EXPECT_EQ(c.RegionName(1), "region1");
  EXPECT_EQ(c.BranchMarker(0), &c.RegionMarker(0));
  EXPECT_EQ(c.BranchMarker(2), &c.RegionMarker(1));
  EXPECT_EQ(maxwell.BranchMarker(0), nullptr);

  // Masked branch data: tau and mu_k of branch 2 seen from each region.
  for (int e = 0; e < mesh->GetNE(); e++) {
    auto* T = mesh->GetElementTransformation(e);
    const auto& ip = Geometries.GetCenter(T->GetGeometryType());
    T->SetIntPoint(&ip);
    const bool inside = mesh->GetAttribute(e) == 2;
    EXPECT_EQ(c.RelaxationTime(2).Eval(*T, ip),
              inside ? 3.0 : CompositeRheology::kOutsideRelaxationTime);
    EXPECT_EQ(c.BranchShearModulus(2).Eval(*T, ip), inside ? 0.5 : 0.0);
    EXPECT_EQ(c.BranchShearModulus(0).Eval(*T, ip), inside ? 0.0 : 0.8);
    DenseMatrix Ck;
    c.BranchModulus(2, *T, ip, Ck);
    EXPECT_EQ(Ck.Height(), SymmetricTensorBasis::Size(dim));
    EXPECT_EQ(Ck.MaxMaxNorm() > 0.0, inside);
  }

  // Overlapping regions are rejected.
  Array<int> both(2);
  both = 1;
  EXPECT_DEATH(
      CompositeRheology(dim, {{Region(1), &maxwell, ""}, {both, &elastic, ""}}),
      "two regions");
  // A region left uncovered on the mesh is rejected when the stiffness is
  // attached.
  CompositeRheology partial(dim, {{Region(1), &maxwell, ""}});
  EXPECT_DEATH(LinearQuasiStaticTractionProblem(fes.get(), partial, *traction,
                                                marker),
               "belongs to no region");
}

TEST_P(CompositeRheologyTest, SplitHomogeneousBody) {
  auto maxwell = IsotropicMaxwellRheology::Maxwell(dim, kappa, mu, tau);
  CompositeRheology split(dim, {{Region(1), &maxwell, ""},
                                {Region(2), &maxwell, ""}});
  ASSERT_EQ(split.NumBranches(), 2);

  for (int scheme = 0; scheme < 2; scheme++) {
    LinearQuasiStaticTractionProblem plain(fes.get(), maxwell, *traction,
                                           marker);
    LinearQuasiStaticTractionProblem comp(fes.get(), split, *traction, marker);
    ViscoelasticOperator v_plain(plain), v_comp(comp);
    ASSERT_EQ(v_comp.NumBranches(), 2);
    // Each branch lives on its region only: the two blocks partition the
    // nodes, and the state is exactly that of the unsplit body.
    ASSERT_EQ(v_comp.NumBranchNodes(0), NodesInRegion(v_comp, 1));
    ASSERT_EQ(v_comp.NumBranchNodes(1), NodesInRegion(v_comp, 2));
    ASSERT_EQ(v_comp.Height(), v_plain.Height());

    const Vector m_plain = Run(v_plain, scheme, 4, 0.7);
    const Vector m_comp = Run(v_comp, scheme, 4, 0.7);
    EXPECT_GT(plain.Displacement().Normlinf(), 0.0);
    EXPECT_LT(RelMaxDiff(comp.Displacement(), plain.Displacement()), 1e-12)
        << "scheme " << scheme;
    const Vector p0 = v_plain.Branch(m_plain, 0);
    EXPECT_LT(RelMaxDiffInRegion(v_plain, Full(v_comp, m_comp, 0), p0, 1),
              1e-12);
    EXPECT_LT(RelMaxDiffInRegion(v_plain, Full(v_comp, m_comp, 1), p0, 2),
              1e-12);
  }
}

TEST_P(CompositeRheologyTest, ElasticRegionMasksBranch) {
  IsotropicElasticRheology elastic(dim, kappa, mu);
  auto maxwell = IsotropicMaxwellRheology::Maxwell(dim, kappa, mu, tau);
  CompositeRheology composite(dim, {{Region(1), &elastic, "elastic"},
                                    {Region(2), &maxwell, "maxwell"}});
  ASSERT_EQ(composite.NumBranches(), 1);
  EXPECT_EQ(composite.BranchLabel(0), "maxwell_branch0");

  // The equivalent global body: mu_inf = mu in region 1, the branch modulus
  // mu in region 2 only.
  auto mu_inf = Piecewise(0.8, 0.0);
  auto mu_k = Piecewise(0.0, 0.8);
  std::vector<MaxwellBranch> branches{{mu_k.get(), &tau}};
  IsotropicMaxwellRheology global(dim, kappa, *mu_inf, branches);

  for (int scheme = 0; scheme < 2; scheme++) {
    LinearQuasiStaticTractionProblem plain(fes.get(), global, *traction,
                                           marker);
    LinearQuasiStaticTractionProblem comp(fes.get(), composite, *traction,
                                          marker);
    ViscoelasticOperator v_plain(plain), v_comp(comp);
    const Vector m_plain = Run(v_plain, scheme, 4, 0.7);
    const Vector m_comp = Run(v_comp, scheme, 4, 0.7);
    EXPECT_GT(plain.Displacement().Normlinf(), 0.0);
    EXPECT_LT(RelMaxDiff(comp.Displacement(), plain.Displacement()), 1e-12)
        << "scheme " << scheme;
    EXPECT_LT(RelMaxDiffInRegion(v_plain, Full(v_comp, m_comp, 0),
                                 v_plain.Branch(m_plain, 0), 2),
              1e-12);
    // The elastic region carries no internal variable at all.
    EXPECT_EQ(v_comp.NumBranchNodes(0), NodesInRegion(v_comp, 2));
    EXPECT_EQ(v_comp.Height(),
              v_comp.NumComponents() * NodesInRegion(v_comp, 2));
  }
}

// With every branch confined to a region the operator computes the
// per-step strain on the union of the branch nodes only; the rate it
// produces must match the one rebuilt from the whole-mesh ComputeStrain(),
// for both strain maps.
TEST_P(CompositeRheologyTest, RestrictedStrainMatchesFull) {
  IsotropicElasticRheology elastic(dim, kappa, mu);
  auto maxwell = IsotropicMaxwellRheology::Maxwell(dim, kappa, mu, tau);
  CompositeRheology composite(dim, {{Region(1), &elastic, "elastic"},
                                    {Region(2), &maxwell, "maxwell"}});

  for (auto map : {ViscoelasticOperator::StrainMap::Galerkin,
                   ViscoelasticOperator::StrainMap::Interpolation}) {
    LinearQuasiStaticTractionProblem prob(fes.get(), composite, *traction,
                                          marker);
    ViscoelasticOperator visco(prob, -1, map);
    EXPECT_TRUE(visco.RestrictedStrainMap());

    Vector m(visco.Height());
    for (int i = 0; i < m.Size(); i++) {
      m[i] = std::sin(1.0 + i);
    }
    Vector k(visco.Height());
    visco.Mult(m, k);  // internally: the restricted strain path
    ASSERT_TRUE(visco.SolveElastic(m, 0.0));
    Vector d_full;
    visco.ComputeStrain(prob.Displacement(), d_full);

    const Array<int>& nodes = visco.BranchNodes(0);
    const int nd = visco.InternalScalarSpace().GetVSize();
    const int nb = visco.NumBranchNodes(0), nc = visco.NumComponents();
    const Vector mb = visco.Branch(m, 0), kb = visco.Branch(k, 0);
    double err = 0.0;
    for (int c = 0; c < nc; c++) {
      for (int q = 0; q < nb; q++) {
        // Rate (d - m) / tau with tau = 1.
        const double rate = d_full[c * nd + nodes[q]] - mb[c * nb + q];
        err = std::max(err, std::abs(kb[c * nb + q] - rate));
      }
    }
    EXPECT_LT(err, 1e-13) << "map " << static_cast<int>(map);
  }

  // A whole-mesh rheology keeps the unrestricted path.
  auto whole = IsotropicMaxwellRheology::Maxwell(dim, kappa, mu, tau);
  LinearQuasiStaticTractionProblem plain(fes.get(), whole, *traction, marker);
  ViscoelasticOperator v_plain(plain);
  EXPECT_FALSE(v_plain.RestrictedStrainMap());
}

TEST_P(CompositeRheologyTest, BranchCountsPerRegion) {
  auto one = IsotropicMaxwellRheology::Maxwell(dim, kappa, mu, tau);
  ConstantCoefficient zero(0.0);
  std::vector<MaxwellBranch> two_branches{{&mu, &tau}, {&mu2, &tau2}};
  IsotropicMaxwellRheology two(dim, kappa, zero, two_branches);
  CompositeRheology composite(dim, {{Region(1), &one, ""},
                                    {Region(2), &two, ""}});
  ASSERT_EQ(composite.NumBranches(), 3);

  auto mu2_pw = Piecewise(0.0, 0.5);
  std::vector<MaxwellBranch> global_branches{{&mu, &tau}, {mu2_pw.get(), &tau2}};
  IsotropicMaxwellRheology global(dim, kappa, zero, global_branches);

  for (int scheme = 0; scheme < 2; scheme++) {
    LinearQuasiStaticTractionProblem plain(fes.get(), global, *traction,
                                           marker);
    LinearQuasiStaticTractionProblem comp(fes.get(), composite, *traction,
                                          marker);
    ViscoelasticOperator v_plain(plain), v_comp(comp);
    const Vector m_plain = Run(v_plain, scheme, 4, 0.7);
    const Vector m_comp = Run(v_comp, scheme, 4, 0.7);
    EXPECT_GT(plain.Displacement().Normlinf(), 0.0);
    EXPECT_LT(RelMaxDiff(comp.Displacement(), plain.Displacement()), 1e-12)
        << "scheme " << scheme;
    // Global branch 2 (region 2's second) against the plain second branch.
    EXPECT_LT(RelMaxDiffInRegion(v_plain, Full(v_comp, m_comp, 2),
                                 v_plain.Branch(m_plain, 1), 2),
              1e-12);
    // State: one branch on region 1, two on region 2.
    EXPECT_EQ(v_comp.Height(),
              v_comp.NumComponents() *
                  (NodesInRegion(v_comp, 1) + 2 * NodesInRegion(v_comp, 2)));
  }
}

TEST_P(CompositeRheologyTest, StiffnessMatchesPiecewiseModuli) {
  auto assemble = [&](const Rheology& r, ElasticStiffness* s = nullptr) {
    auto stiffness = s ? nullptr : r.MakeStiffness();
    BilinearForm a(fes.get());
    (s ? s : stiffness.get())->AddIntegrators(a);
    a.Assemble();
    a.Finalize();
    return SparseMatrix(a.SpMat());
  };

  // Elastic regions.
  {
    ConstantCoefficient k1(2.1), k2(3.3), m1(0.8), m2(0.4);
    IsotropicElasticRheology e1(dim, k1, m1), e2(dim, k2, m2);
    CompositeRheology composite(dim, {{Region(1), &e1, ""},
                                      {Region(2), &e2, ""}});
    auto kpw = Piecewise(2.1, 3.3);
    auto mpw = Piecewise(0.8, 0.4);
    IsotropicElasticRheology global(dim, *kpw, *mpw);
    const auto A = assemble(composite), B = assemble(global);
    EXPECT_GT(B.MaxNorm(), 0.0);
    EXPECT_LT(MaxDiff(A, B), 1e-13 * B.MaxNorm());
  }

  // Maxwell regions with relaxation weights.
  {
    ConstantCoefficient m1(0.8), m2(0.4);
    auto r1 = IsotropicMaxwellRheology::Maxwell(dim, kappa, m1, tau);
    auto r2 = IsotropicMaxwellRheology::Maxwell(dim, kappa, m2, tau2);
    CompositeRheology composite(dim, {{Region(1), &r1, ""},
                                      {Region(2), &r2, ""}});
    auto mpw = Piecewise(0.8, 0.4);
    auto global = IsotropicMaxwellRheology::Maxwell(dim, kappa, *mpw, tau);

    auto sc = composite.MakeStiffness();
    auto sg = global.MakeStiffness();
    EXPECT_LT(MaxDiff(assemble(composite, sc.get()),
                      assemble(global, sg.get())),
              1e-13);

    ConstantCoefficient b1(0.3), b2(0.7);
    auto bpw = Piecewise(0.3, 0.7);
    sc->SetRelaxationWeights({&b1, &b2});
    sg->SetRelaxationWeights({bpw.get()});
    EXPECT_TRUE(sc->IsRelaxed());
    const auto A = assemble(composite, sc.get()),
               B = assemble(global, sg.get());
    EXPECT_LT(MaxDiff(A, B), 1e-13 * B.MaxNorm());
    // The weighted matrix differs from the unrelaxed one.
    sc->ClearRelaxationWeights();
    EXPECT_FALSE(sc->IsRelaxed());
    EXPECT_GT(MaxDiff(assemble(composite, sc.get()), A), 1e-3);
  }
}

TEST_P(CompositeRheologyTest, MixedIsotropicAnisotropicRegions) {
  auto iso = IsotropicMaxwellRheology::Maxwell(dim, kappa, mu, tau);
  auto C = IsotropicElasticTensorCoefficient::FromBulkModulus(dim, kappa, mu);
  auto aniso = AnisotropicMaxwellRheology::DeviatoricMaxwell(dim, C, tau);
  CompositeRheology mixed(dim, {{Region(1), &iso, ""}, {Region(2), &aniso, ""}});
  CompositeRheology all_iso(dim, {{Region(1), &iso, ""}, {Region(2), &iso, ""}});
  EXPECT_FALSE(mixed.TraceFreeInternalVariables());
  EXPECT_TRUE(all_iso.TraceFreeInternalVariables());

  LinearQuasiStaticTractionProblem p_mixed(fes.get(), mixed, *traction, marker);
  LinearQuasiStaticTractionProblem p_iso(fes.get(), all_iso, *traction, marker);
  ViscoelasticOperator v_mixed(p_mixed), v_iso(p_iso);
  EXPECT_FALSE(v_mixed.TraceFree());
  EXPECT_TRUE(v_iso.TraceFree());
  Run(v_mixed, 0, 4, 0.7);
  Run(v_iso, 0, 4, 0.7);
  EXPECT_GT(p_iso.Displacement().Normlinf(), 0.0);
  EXPECT_LT(RelMaxDiff(p_mixed.Displacement(), p_iso.Displacement()), 1e-10);
}

INSTANTIATE_TEST_SUITE_P(Composite, CompositeRheologyTest,
                         testing::Combine(testing::Values(2, 3),
                                          testing::Values(0, 1),
                                          testing::Values(1, 2)));

TEST(CompositeRheologyMixedProblem, ElasticCoreMaxwellMantle) {
  using namespace self_grav_test;
  const int dim = 2, order = 1;
  Mesh parent(ThreeLayerMeshFile(dim).c_str(), 1, 1);
  Array<int> attrs({1, 3});
  SubMesh solid = SubMesh::CreateFromDomain(parent, attrs);
  H1_FECollection fec(order, dim);
  FiniteElementSpace fes_u(&solid, &fec, dim), fes_phi(&parent, &fec);
  ConstantCoefficient kappa(self_grav_test::kKappa), mu(self_grav_test::kMu),
      tau(1.0);
  FunctionCoefficient rho_s(SolidDensity), rho_f(FluidDensity);
  FunctionCoefficient sigma(SurfaceLoad);

  IsotropicElasticRheology core(dim, kappa, mu);
  auto mantle = IsotropicMaxwellRheology::Maxwell(dim, kappa, mu, tau);
  Array<int> core_marker(3), mantle_marker(3);
  core_marker = 0;
  core_marker[0] = 1;
  mantle_marker = 0;
  mantle_marker[2] = 1;
  CompositeRheology composite(dim, {{core_marker, &core, "inner_core"},
                                    {mantle_marker, &mantle, "mantle"}});
  ASSERT_EQ(composite.NumBranches(), 1);
  EXPECT_EQ(composite.BranchLabel(0), "mantle_branch0");

  std::vector<FluidRegion> fluids{OuterCore(solid, rho_f)};
  LinearQuasiStaticMixedSelfGravitatingProblem problem(&fes_u, &fes_phi, composite,
                                                  rho_s, kG, kDtNDegree,
                                                  nullptr, fluids);
  problem.SetSurfaceLoad(sigma, SurfaceMarker(solid));
  Array<int> inner_core({1});
  problem.AddRegionRotations(inner_core);
  problem.SetRelTol(1e-10);

  ViscoelasticOperator visco(problem);
  ExponentialTrapezoidSolver ode;
  ode.Init(visco);
  Vector m(visco.Height());
  m = 0.0;
  double t = 0.0, dt = 0.5;
  ASSERT_TRUE(visco.SolveElastic(m, t));
  const double u0 = L2Norm(problem.Displacement());
  EXPECT_GT(u0, 0.0);
  for (int step = 0; step < 3; step++) {
    ode.Step(m, t, dt);
  }
  ASSERT_TRUE(visco.SolveElastic(m, t));
  const double u1 = L2Norm(problem.Displacement());
  EXPECT_TRUE(std::isfinite(u1));
  EXPECT_GT(u1, u0);  // the mantle relaxes under the sustained load
  EXPECT_GT(m.Normlinf(), 0.0);
}

}  // namespace
