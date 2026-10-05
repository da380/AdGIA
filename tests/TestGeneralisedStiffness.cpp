#include <numbers>

#include "TestCommon.hpp"

/*
  Tests for the building blocks of the generalised (pre-stressed,
  referential) elasticity implementation (doc/gravitating_elasticity.md):

  - BareElasticTensorCoefficient: the PREM -> bare conversion reproduces
    lambda - p, mu + p isotropically, and is the identity at p = 0.
  - GeometricStiffnessIntegrator: with S = s 1 it equals
    mfem::VectorDiffusionIntegrator on a curved mesh (shared quadrature);
    an energy patch test for constant symmetric S and linear fields,
    u^T K v = |Omega| tr(A S B^T); and the pull-back (change-of-variables)
    identity against the mapped mesh (interpolated mapping, one rule).
  - MaterialStiffnessIntegrator: with the identity mapping it equals
    ElasticTensorIntegrator to round-off; for an affine equilibrium
    mapping the energy of linear fields is
    |Omega| <C m(sym F0^T A), m(sym F0^T B)> with m the Mandel vector,
    checked for an isotropic and (3-D) a transversely isotropic tensor.
  - ReferentialGravityIntegrator and ReferentialGravityCouplingIntegrator:
    against second differences of the exact gravitational functional.
*/

namespace {

constexpr real_t pi = std::numbers::pi_v<real_t>;

using Param = std::tuple<int, int, int>;  // (dim, order, elementType)

Mesh SmallMesh(int dim, int elementType) {
  if (dim == 2) {
    return Mesh::MakeCartesian2D(
        5, 5, elementType == 0 ? Element::TRIANGLE : Element::QUADRILATERAL);
  }
  return Mesh::MakeCartesian3D(
      3, 3, 3, elementType == 0 ? Element::TETRAHEDRON : Element::HEXAHEDRON);
}

// A smooth, non-polynomial diffeomorphism with exact gradient (as in
// TestMappedDomainIntegrators).
CallableDiffeomorphism SmoothMap(int dim, real_t c) {
  return CallableDiffeomorphism(
      dim,
      [c, dim](const Vector& x, Vector& y) {
        for (int i = 0; i < dim; i++) {
          y(i) = x(i) + c * std::sin(pi * x((i + 1) % dim));
        }
      },
      [c, dim](const Vector& x, DenseMatrix& F) {
        F = 0.0;
        for (int i = 0; i < dim; i++) {
          F(i, i) = 1.0;
          F(i, (i + 1) % dim) = c * pi * std::cos(pi * x((i + 1) % dim));
        }
      });
}

CallableDiffeomorphism IdentityMap(int dim) {
  return CallableDiffeomorphism(
      dim, [](const Vector& x, Vector& y) { y = x; },
      [](const Vector& /*x*/, DenseMatrix& F) {
        F = 0.0;
        for (int i = 0; i < F.Height(); i++) {
          F(i, i) = 1.0;
        }
      });
}

double MaxDiff(const DenseMatrix& A, const DenseMatrix& B) {
  if (A.Height() != B.Height() || A.Width() != B.Width() || A.Height() == 0) {
    return std::numeric_limits<double>::infinity();
  }
  DenseMatrix D(A);
  D -= B;
  return D.MaxMaxNorm();
}

// The Mandel vector of a symmetric matrix in library ordering.
Vector MandelVector(const DenseMatrix& E) {
  const int dim = E.Height();
  Vector m(SymmetricTensorBasis::Size(dim));
  for (int j = 0; j < dim; j++) {
    for (int k = 0; k <= j; k++) {
      m(SymmetricTensorBasis::Index(dim, j, k)) =
          SymmetricTensorBasis::Scale(j, k) * E(j, k);
    }
  }
  return m;
}

// Interpolate the linear field u = A x on a vector space.
GridFunction LinearField(FiniteElementSpace& fes, const DenseMatrix& A) {
  VectorFunctionCoefficient c(A.Height(), [&A](const Vector& x, Vector& u) {
    A.Mult(x, u);
  });
  GridFunction u(&fes);
  u.ProjectCoefficient(c);
  return u;
}

}  // namespace

TEST(GeneralisedStiffness, BareTensorIsotropicConversion) {
  for (int dim : {2, 3}) {
    auto mesh = SmallMesh(dim, 0);
    auto* T = mesh.GetElementTransformation(0);
    const auto& ip = Geometries.GetCenter(mesh.GetElementGeometry(0));
    T->SetIntPoint(&ip);

    ConstantCoefficient lam(2.3), mu(0.9), p(0.4), zero(0.0);
    ConstantCoefficient lam_b(2.3 - 0.4), mu_b(0.9 + 0.4);
    IsotropicElasticTensorCoefficient C_eff(dim, lam, mu);
    IsotropicElasticTensorCoefficient C_expected(dim, lam_b, mu_b);
    BareElasticTensorCoefficient C_bare(dim, C_eff, p);
    BareElasticTensorCoefficient C_id(dim, C_eff, zero);

    DenseMatrix K1, K2, K3;
    C_bare.Eval(K1, *T, ip);
    C_expected.Eval(K2, *T, ip);
    C_eff.Eval(K3, *T, ip);
    EXPECT_LT(MaxDiff(K1, K2), 1e-14);
    DenseMatrix K4;
    C_id.Eval(K4, *T, ip);
    EXPECT_LT(MaxDiff(K4, K3), 1e-14);
  }
}

class GeneralisedStiffnessTest : public ::testing::TestWithParam<Param> {};

TEST_P(GeneralisedStiffnessTest, GeometricEqualsVectorDiffusionForIsotropicS) {
  const auto [dim, order, elementType] = GetParam();
  auto mesh = SmallMesh(dim, elementType);
  mesh.SetCurvature(order);
  H1_FECollection fec(order, dim);
  FiniteElementSpace fes(&mesh, &fec, dim);
  const IntegrationRule& ir =
      IntRules.Get(mesh.GetTypicalElementGeometry(), 2 * order + 2);

  const real_t s = 0.7;
  DenseMatrix S(dim);
  S = 0.0;
  for (int i = 0; i < dim; i++) {
    S(i, i) = s;
  }
  MatrixConstantCoefficient S_c(S);
  ConstantCoefficient s_c(s);

  BilinearForm a(&fes);
  a.AddDomainIntegrator(new GeometricStiffnessIntegrator(S_c, &ir));
  a.Assemble();
  a.Finalize();

  BilinearForm b(&fes);
  auto* vd = new VectorDiffusionIntegrator(s_c);
  vd->SetIntRule(&ir);
  b.AddDomainIntegrator(vd);
  b.Assemble();
  b.Finalize();

  EXPECT_LT(::MaxDiff(a.SpMat(), b.SpMat()), 1e-12 * a.SpMat().MaxNorm());
}

TEST_P(GeneralisedStiffnessTest, GeometricEnergyPatch) {
  const auto [dim, order, elementType] = GetParam();
  auto mesh = SmallMesh(dim, elementType);
  H1_FECollection fec(order, dim);
  FiniteElementSpace fes(&mesh, &fec, dim);

  auto S = RandomMatrix(dim);
  S.Symmetrize();
  MatrixConstantCoefficient S_c(S);
  auto A = RandomMatrix(dim);
  auto B = RandomMatrix(dim);

  BilinearForm a(&fes);
  a.AddDomainIntegrator(new GeometricStiffnessIntegrator(S_c));
  a.Assemble();
  a.Finalize();

  auto u = LinearField(fes, A);
  auto v = LinearField(fes, B);
  Vector Ku(fes.GetVSize());
  a.SpMat().Mult(u, Ku);
  const real_t energy = Ku * v;

  // |Omega| = 1: int S_AB A_kA B_kB = tr(A S B^T).
  DenseMatrix AS(dim), ASBt(dim);
  Mult(A, S, AS);
  MultABt(AS, B, ASBt);
  EXPECT_NEAR(energy, ASBt.Trace(), 1e-12 * std::abs(ASBt.Trace()) + 1e-14);
}

TEST_P(GeneralisedStiffnessTest, GeometricPullbackMatchesMappedMesh) {
  const auto [dim, order, elementType] = GetParam();
  auto mesh = SmallMesh(dim, elementType);
  mesh.SetCurvature(order);
  H1_FECollection fec(order, dim);
  FiniteElementSpace fes(&mesh, &fec, dim);

  auto xi = SmoothMap(dim, 0.05);
  auto xi_h = Interpolate(xi, mesh);
  auto mapped = MappedMesh(mesh, xi);
  FiniteElementSpace fes_m(&mapped, &fec, dim);

  const IntegrationRule& ir =
      IntRules.Get(mesh.GetTypicalElementGeometry(), 2 * order + 3);

  // A physical symmetric matrix field and its referential expression.
  auto S_fn = [dim](const Vector& x, DenseMatrix& S) {
    S.SetSize(x.Size());
    for (int j = 0; j < dim; j++) {
      for (int k = 0; k <= j; k++) {
        S(j, k) = S(k, j) =
            (j == k ? 2.0 : 0.0) + 0.3 * std::cos(x(j) + 2.0 * x(k));
      }
    }
  };
  MatrixFunctionCoefficient S_phys(dim, S_fn);
  TransformedMatrixFunctionCoefficient S_ref(dim, xi_h, S_fn);

  BilinearForm a_ref(&fes);
  a_ref.AddDomainIntegrator(
      new GeometricStiffnessIntegrator(S_ref, xi_h, &ir));
  a_ref.Assemble();
  a_ref.Finalize();

  BilinearForm a_map(&fes_m);
  a_map.AddDomainIntegrator(new GeometricStiffnessIntegrator(S_phys, &ir));
  a_map.Assemble();
  a_map.Finalize();

  EXPECT_LT(::MaxDiff(a_ref.SpMat(), a_map.SpMat()),
            1e-12 * a_map.SpMat().MaxNorm());
}

TEST_P(GeneralisedStiffnessTest, MaterialIdentityMapEqualsElasticTensor) {
  const auto [dim, order, elementType] = GetParam();
  auto mesh = SmallMesh(dim, elementType);
  mesh.SetCurvature(order);
  H1_FECollection fec(order, dim);
  FiniteElementSpace fes(&mesh, &fec, dim);
  const IntegrationRule& ir =
      IntRules.Get(mesh.GetTypicalElementGeometry(), 2 * order + 2);

  FunctionCoefficient lam([](const Vector& x) { return 1.5 + 0.3 * x(0); });
  FunctionCoefficient mu([](const Vector& x) { return 0.8 + 0.2 * x(1); });
  IsotropicElasticTensorCoefficient C(dim, lam, mu);
  auto id = IdentityMap(dim);

  BilinearForm a(&fes);
  a.AddDomainIntegrator(new MaterialStiffnessIntegrator(C, id, &ir));
  a.Assemble();
  a.Finalize();

  BilinearForm b(&fes);
  b.AddDomainIntegrator(new ElasticTensorIntegrator(C, &ir));
  b.Assemble();
  b.Finalize();

  EXPECT_LT(::MaxDiff(a.SpMat(), b.SpMat()), 1e-12 * b.SpMat().MaxNorm());
}

TEST_P(GeneralisedStiffnessTest, MaterialAffineEnergyPatch) {
  const auto [dim, order, elementType] = GetParam();
  auto mesh = SmallMesh(dim, elementType);
  H1_FECollection fec(order, dim);
  FiniteElementSpace fes(&mesh, &fec, dim);

  // A fixed, orientation-preserving affine equilibrium mapping.
  DenseMatrix F0(dim);
  F0 = 0.0;
  for (int i = 0; i < dim; i++) {
    F0(i, i) = 1.0 + 0.1 * i;
    F0(i, (i + 1) % dim) += 0.2;
  }
  MFEM_VERIFY(F0.Det() > 0.0, "test mapping must be orientation-preserving");
  CallableDiffeomorphism phi(
      dim, [&F0](const Vector& x, Vector& y) { F0.Mult(x, y); },
      [&F0](const Vector& /*x*/, DenseMatrix& F) { F = F0; });

  // The energy for one tensor C.
  auto check = [&](MatrixCoefficient& C, const char* name) {
    BilinearForm a(&fes);
    a.AddDomainIntegrator(new MaterialStiffnessIntegrator(C, phi));
    a.Assemble();
    a.Finalize();

    auto A = RandomMatrix(dim);
    auto B = RandomMatrix(dim);
    auto u = LinearField(fes, A);
    auto v = LinearField(fes, B);
    Vector Ku(fes.GetVSize());
    a.SpMat().Mult(u, Ku);
    const real_t energy = Ku * v;

    // |Omega| = 1: <C m(sym F0^T A), m(sym F0^T B)> with the Mandel
    // matrix evaluated once (constant coefficient).
    auto sym_pull = [&F0, dim](const DenseMatrix& G) {
      DenseMatrix FtG(dim), E(dim);
      MultAtB(F0, G, FtG);
      E = FtG;
      E.Symmetrize();
      return E;
    };
    auto mA = MandelVector(sym_pull(A));
    auto mB = MandelVector(sym_pull(B));
    auto* T = mesh.GetElementTransformation(0);
    const auto& ip = Geometries.GetCenter(mesh.GetElementGeometry(0));
    T->SetIntPoint(&ip);
    DenseMatrix Cm;
    C.Eval(Cm, *T, ip);
    Vector CmB(mB.Size());
    Cm.Mult(mB, CmB);
    const real_t expected = mA * CmB;
    // Scaled by |mA| |C mB| rather than |expected|: the random A and B can
    // make the product nearly cancel.
    EXPECT_NEAR(energy, expected, 1e-12 * mA.Norml2() * CmB.Norml2())
        << name;
  };

  ConstantCoefficient lam(1.3), mu(0.7);
  IsotropicElasticTensorCoefficient iso(dim, lam, mu);
  check(iso, "isotropic");

  // Transversely isotropic (3-D: the class is plane strain in 2-D) with a
  // constant axis tilted off the coordinate axes, so that every Mandel
  // component of the tensor is exercised.
  if (dim == 3) {
    ConstantCoefficient Ac(3.1), Cc(2.6), Fc(1.0), Lc(0.7), Nc(0.9);
    Vector n(3);
    n(0) = 0.3;
    n(1) = -0.4;
    n(2) = 0.866;
    VectorConstantCoefficient axis(n);
    TransverselyIsotropicElasticTensorCoefficient ti(3, Ac, Cc, Fc, Lc, Nc,
                                                     axis);
    check(ti, "transversely isotropic");
  }
}


// The referential gravity blocks against second differences of the exact
// functional T = int <a(F) g, g> dV, a(F) = J F^{-1} F^{-T}, g = g0 + grad
// zeta1, at the equilibrium mapping: the assembled bilinear forms must reproduce the mixed partial
// derivatives of T in the FE directions to finite-difference accuracy.
namespace {

double GravityFunctional(Mesh& mesh, Diffeomorphism& phi,
                         VectorCoefficient& g0, const GridFunction* u,
                         const GridFunction* v, double e, double d,
                         const GridFunction* zeta1, double dz, int rule_order) {
  const int dim = mesh.Dimension();
  DenseMatrix F(dim), Du(dim), Dv(dim), Finv(dim), a(dim);
  Vector g(dim), gz(dim), ag(dim);
  double total = 0.0;
  for (int el = 0; el < mesh.GetNE(); el++) {
    auto* T = mesh.GetElementTransformation(el);
    const auto& ir = IntRules.Get(mesh.GetElementGeometry(el), rule_order);
    for (int q = 0; q < ir.GetNPoints(); q++) {
      const auto& ip = ir.IntPoint(q);
      T->SetIntPoint(&ip);
      phi.EvalGradient(F, *T, ip);
      if (u) {
        u->GetVectorGradient(*T, Du);
        F.Add(e, Du);
      }
      if (v) {
        v->GetVectorGradient(*T, Dv);
        F.Add(d, Dv);
      }
      const double J = F.Det();
      Finv = F;
      Finv.Invert();
      MultAAt(Finv, a);
      a *= J;
      g0.Eval(g, *T, ip);
      if (zeta1) {
        zeta1->GetGradient(*T, gz);
        g.Add(dz, gz);
      }
      a.Mult(g, ag);
      total += ip.weight * T->Weight() * (g * ag);
    }
  }
  return total;
}

}  // namespace

TEST_P(GeneralisedStiffnessTest, ReferentialGravityMatchesFiniteDifference) {
  const auto [dim, order, elementType] = GetParam();
  auto mesh = SmallMesh(dim, elementType);
  mesh.SetCurvature(order);
  H1_FECollection fec(order, dim);
  FiniteElementSpace fes_u(&mesh, &fec, dim);
  FiniteElementSpace fes_p(&mesh, &fec);
  const int rule_order = 2 * order + 3;
  const IntegrationRule& ir =
      IntRules.Get(mesh.GetTypicalElementGeometry(), rule_order);

  auto phi = SmoothMap(dim, 0.07);
  VectorFunctionCoefficient g0(dim, [dim](const Vector& x, Vector& g) {
    for (int i = 0; i < dim; i++) {
      g(i) = 0.4 + 0.3 * std::cos(x(i) + 0.2 * x((i + 1) % dim)) + 0.1 * i;
    }
  });

  VectorFunctionCoefficient u_fn(dim, [dim](const Vector& x, Vector& y) {
    for (int i = 0; i < dim; i++) {
      y(i) = std::sin(1.3 * x(i)) + 0.5 * x((i + 1) % dim);
    }
  });
  VectorFunctionCoefficient v_fn(dim, [dim](const Vector& x, Vector& y) {
    for (int i = 0; i < dim; i++) {
      y(i) = std::cos(0.9 * x((i + 1) % dim)) - 0.3 * x(i) * x(i);
    }
  });
  GridFunction u(&fes_u), v(&fes_u);
  u.ProjectCoefficient(u_fn);
  v.ProjectCoefficient(v_fn);

  // Gravity-gravity block: v^T B u = d^2/de dd T(F + e Du + d Dv).
  {
    BilinearForm b(&fes_u);
    b.AddDomainIntegrator(new ReferentialGravityIntegrator(phi, g0, 1.0, &ir));
    b.Assemble();
    b.Finalize();
    Vector Bu(fes_u.GetVSize());
    b.SpMat().Mult(u, Bu);
    const double assembled = Bu * v;

    const double h = 1e-4;
    auto T = [&](double e, double d) {
      return GravityFunctional(mesh, phi, g0, &u, &v, e, d, nullptr, 0.0,
                               rule_order);
    };
    const double fd =
        (T(h, h) - T(h, -h) - T(-h, h) + T(-h, -h)) / (4.0 * h * h);
    EXPECT_NEAR(assembled, fd, 1e-5 * std::abs(fd) + 1e-9);

    // Symmetry of the assembled block.
    std::unique_ptr<SparseMatrix> Bt(Transpose(b.SpMat()));
    EXPECT_LT(::MaxDiff(b.SpMat(), *Bt), 1e-12 * b.SpMat().MaxNorm());
  }

  // Coupling block: u^T C zeta1 = d^2/de dd of T(F + e Du, g0 + d grad z)/2.
  {
    FunctionCoefficient z_fn([dim](const Vector& x) {
      double s = 0.7;
      for (int i = 0; i < dim; i++) {
        s += std::sin(0.8 * x(i) + 0.1 * i);
      }
      return s;
    });
    GridFunction zeta1(&fes_p);
    zeta1.ProjectCoefficient(z_fn);

    MixedBilinearForm c(&fes_p, &fes_u);
    c.AddDomainIntegrator(
        new ReferentialGravityCouplingIntegrator(phi, g0, 1.0, &ir));
    c.Assemble();
    c.Finalize();
    Vector Cz(fes_u.GetVSize());
    c.SpMat().Mult(zeta1, Cz);
    const double assembled = Cz * u;

    const double h = 1e-4;
    auto T = [&](double e, double d) {
      return 0.5 * GravityFunctional(mesh, phi, g0, &u, nullptr, e, 0.0,
                                     &zeta1, d, rule_order);
    };
    const double fd =
        (T(h, h) - T(h, -h) - T(-h, h) + T(-h, -h)) / (4.0 * h * h);
    EXPECT_NEAR(assembled, fd, 1e-5 * std::abs(fd) + 1e-9);
  }
}

INSTANTIATE_TEST_SUITE_P(DimOrderType, GeneralisedStiffnessTest,
                         ::testing::Values(std::make_tuple(2, 1, 0),
                                           std::make_tuple(2, 2, 1),
                                           std::make_tuple(2, 3, 0),
                                           std::make_tuple(3, 1, 1),
                                           std::make_tuple(3, 2, 0)));
