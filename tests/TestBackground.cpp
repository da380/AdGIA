#include "MixedProblemTestCommon.hpp"
#include "TestCommon.hpp"

/*
  Tests for the background-state module (background.hpp) and the buffer
  tapers of mappings.hpp: the radial hydrostatic balance against analytic
  profiles; the generated coefficient chains against hand-rolled
  references; the minimum-norm and minimum-deviatoric equilibrium-stress
  generators (weak equilibrium, optimality, the hydrostatic disc, the
  ellipse, the mapped mode); and the analytic and harmonic buffer tapers.
  The end-to-end check, a relabelled background reproducing the base
  solution, is ReferentialProblem.RelabelledEquilibrium2D in
  TestReferentialProblem.cpp.
*/

namespace {

using namespace self_grav_test;

constexpr double kPi = std::numbers::pi;

}  // namespace

TEST(Background, UniformStateMatchesAnalytic) {
  for (int dim : {2, 3}) {
    RadialHydrostaticState state(
        dim, [](double) { return kRho; }, kG, 1.0);
    for (double r : {0.0, 0.25, 0.5, 0.75, 1.0}) {
      const double g = dim == 2 ? 2.0 * kPi * kG * kRho * r
                                : 4.0 * kPi * kG * kRho * r / 3.0;
      const double p = dim == 2
                           ? kPi * kG * kRho * kRho * (1.0 - r * r)
                           : 2.0 * kPi * kG * kRho * kRho * (1.0 - r * r) / 3.0;
      EXPECT_NEAR(state.Gravity(r), g, 1e-6);
      EXPECT_NEAR(state.Pressure(r), p, 1e-6);
    }
    // Exterior: the field of the total mass, zero pressure.
    const double g_ext = dim == 2 ? 2.0 * kPi * kG * kRho / 1.5
                                  : 4.0 * kPi * kG * kRho / (3.0 * 1.5 * 1.5);
    EXPECT_NEAR(state.Gravity(1.5), g_ext, 1e-6);
    EXPECT_EQ(state.Pressure(1.5), 0.0);
    EXPECT_EQ(state.Density(1.5), 0.0);
  }
}

TEST(Background, StratifiedStateMatchesAnalytic) {
  // rho = rho0 (1 - r^2/2), R = 1, 3-D:
  // g = 4 pi G rho0 (r/3 - r^3/10),
  // p = 4 pi G rho0^2 (F(1) - F(r)), F(s) = s^2/6 - s^4/15 + s^6/120.
  const double rho0 = 1.3;
  RadialHydrostaticState state(
      3, [rho0](double r) { return rho0 * (1.0 - 0.5 * r * r); }, kG, 1.0);
  auto F = [](double s) {
    const double s2 = s * s;
    return s2 / 6.0 - s2 * s2 / 15.0 + s2 * s2 * s2 / 120.0;
  };
  for (double r : {0.1, 0.4, 0.7, 0.95}) {
    const double g = 4.0 * kPi * kG * rho0 * (r / 3.0 - r * r * r / 10.0);
    const double p = 4.0 * kPi * kG * rho0 * rho0 * (F(1.0) - F(r));
    EXPECT_NEAR(state.Gravity(r), g, 1e-6);
    EXPECT_NEAR(state.Pressure(r), p, 1e-6);
  }
}

// The generated coefficients equal the hand-rolled chain on the uniform
// disc: p0 = pi G rho^2 (1 - r^2), C = Bare(C_eff, p0), S_e = -p0 1.
TEST(Background, HydrostaticCoefficientsMatchHandRolled) {
  const int dim = 2;
  Mesh mesh(MeshFile(dim).c_str(), 1, 1);

  RadialHydrostaticBackground bg(
      dim, [](double) { return kRho; }, [](double) { return kKappa; },
      [](double) { return kMu; }, kG, 1.0);

  ConstantCoefficient kappa(kKappa), mu(kMu);
  auto C_eff =
      IsotropicElasticTensorCoefficient::FromBulkModulus(dim, kappa, mu);
  FunctionCoefficient p0([](const Vector& x) {
    return kPi * kG * kRho * kRho * (1.0 - (x * x));
  });
  BareElasticTensorCoefficient C_ref(dim, C_eff, p0);

  int n_body = 0;
  for (int e = 0; e < mesh.GetNE(); e++) {
    if (mesh.GetAttribute(e) != 1) {
      continue;
    }
    n_body++;
    auto* T = mesh.GetElementTransformation(e);
    const auto& ip = Geometries.GetCenter(mesh.GetElementGeometry(e));
    T->SetIntPoint(&ip);
    Vector x(dim);
    T->Transform(ip, x);

    const double p_ref = kPi * kG * kRho * kRho * (1.0 - (x * x));
    EXPECT_NEAR(bg.Pressure().Eval(*T, ip), p_ref, 1e-6);
    EXPECT_NEAR(bg.Density().Eval(*T, ip), kRho, 1e-14);

    DenseMatrix C1, C2;
    bg.ElasticTensor().Eval(C1, *T, ip);
    C_ref.Eval(C2, *T, ip);
    C1 -= C2;
    EXPECT_LT(C1.MaxMaxNorm(), 1e-6);

    DenseMatrix S;
    bg.EquilibriumStress().Eval(S, *T, ip);
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        EXPECT_NEAR(S(i, j), i == j ? -p_ref : 0.0, 1e-6);
      }
    }
  }
  ASSERT_GT(n_body, 0);

  // The equilibrium mapping is the identity, exactly.
  auto* T = mesh.GetElementTransformation(0);
  const auto& ip = Geometries.GetCenter(mesh.GetElementGeometry(0));
  T->SetIntPoint(&ip);
  Vector x(dim), y(dim);
  T->Transform(ip, x);
  bg.EquilibriumMapping().Eval(y, *T, ip);
  y -= x;
  EXPECT_LT(y.Norml2(), 1e-15);
  DenseMatrix F;
  bg.EquilibriumMapping().EvalGradient(F, *T, ip);
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      EXPECT_NEAR(F(i, j), i == j ? 1.0 : 0.0, 1e-15);
    }
  }
}

// The relabelled chain equals the hand-rolled transformation laws at a
// point, for a non-trivial interior map.
TEST(Background, RelabelledCoefficientsMatchHandRolled) {
  const int dim = 2;
  Mesh mesh(MeshFile(dim).c_str(), 1, 1);

  RadialHydrostaticBackground base(
      dim, [](double) { return kRho; }, [](double) { return kKappa; },
      [](double) { return kMu; }, kG, 1.0);

  // xi = x (1 + c (r (1 - r))^2) inside r < 1, identity beyond.
  const double c = 0.3;
  auto h = [](double r) {
    if (r >= 1.0) {
      return 0.0;
    }
    const double q = r * (1.0 - r);
    return q * q;
  };
  CallableDiffeomorphism xi(
      dim,
      [c, h](const Vector& x, Vector& y) {
        y = x;
        y *= 1.0 + c * h(x.Norml2());
      },
      [c, h](const Vector& x, DenseMatrix& F) {
        const double r = x.Norml2();
        F = 0.0;
        const double f = 1.0 + c * h(r);
        for (int i = 0; i < x.Size(); i++) {
          F(i, i) = f;
        }
        if (r > 0.0 && r < 1.0) {
          const double dh = 2.0 * r * (1.0 - r) * (1.0 - 2.0 * r);
          for (int i = 0; i < x.Size(); i++) {
            for (int j = 0; j < x.Size(); j++) {
              F(i, j) += c * dh / r * x(i) * x(j);
            }
          }
        }
      });
  RelabelledBackground rel(base, xi);

  // Hand-rolled chain: each coefficient composed from its transformation
  // law.
  auto pressure = [&base](const Vector& y) {
    return base.PressureAt(y.Norml2());
  };
  TransformedFunctionCoefficient p0_xi(xi, pressure);
  ConstantCoefficient kappa(kKappa), mu(kMu);
  auto C_eff =
      IsotropicElasticTensorCoefficient::FromBulkModulus(dim, kappa, mu);
  BareElasticTensorCoefficient C_comp(dim, C_eff, p0_xi);
  RelabelledElasticTensorCoefficient C_ref(dim, C_comp, xi);
  TransformedMatrixFunctionCoefficient S_comp(
      dim, xi, [&pressure](const Vector& y, DenseMatrix& S) {
        S.SetSize(y.Size());
        S = 0.0;
        const double p = pressure(y);
        for (int i = 0; i < y.Size(); i++) {
          S(i, i) = -p;
        }
      });
  PullbackStressCoefficient S_ref(dim, S_comp, xi);
  JacobianCoefficient jac(xi);

  int n_body = 0;
  for (int e = 0; e < mesh.GetNE(); e += 3) {
    if (mesh.GetAttribute(e) != 1) {
      continue;
    }
    n_body++;
    auto* T = mesh.GetElementTransformation(e);
    const auto& ip = Geometries.GetCenter(mesh.GetElementGeometry(e));
    T->SetIntPoint(&ip);

    DenseMatrix A, B;
    rel.ElasticTensor().Eval(A, *T, ip);
    C_ref.Eval(B, *T, ip);
    A -= B;
    EXPECT_LT(A.MaxMaxNorm(), 1e-12);

    rel.EquilibriumStress().Eval(A, *T, ip);
    S_ref.Eval(B, *T, ip);
    A -= B;
    EXPECT_LT(A.MaxMaxNorm(), 1e-12);

    const double rho_ref = kRho * jac.Eval(*T, ip);
    EXPECT_NEAR(rel.Density().Eval(*T, ip), rho_ref, 1e-12);
  }
  ASSERT_GT(n_body, 0);
}

namespace {

// Relative weak-form residual of an equilibrium stress: with
// Div T = f and T n = 0, int T : grad v + int f . v must vanish for
// every v; testing on an independent higher-order space measures the
// discretisation error of the generator.
double WeakEquilibriumResidual(MatrixCoefficient& S, VectorCoefficient& f,
                               FiniteElementSpace& fes) {
  LinearForm r(&fes);
  r.AddDomainIntegrator(new DomainLFDeformationGradientIntegrator(S));
  r.AddDomainIntegrator(new VectorDomainLFIntegrator(f));
  r.Assemble();
  LinearForm fn(&fes);
  fn.AddDomainIntegrator(new VectorDomainLFIntegrator(f));
  fn.Assemble();
  return r.Norml2() / fn.Norml2();
}

// Quadrature L2 norms of a stress coefficient: full and deviatoric.
void StressNorms(MatrixCoefficient& S, Mesh& mesh, int order, double& full,
                 double& dev) {
  const int dim = mesh.Dimension();
  DenseMatrix T;
  double full2 = 0.0, dev2 = 0.0;
  for (int e = 0; e < mesh.GetNE(); e++) {
    auto* Tr = mesh.GetElementTransformation(e);
    const auto& ir = IntRules.Get(mesh.GetElementGeometry(e), 2 * order);
    for (int q = 0; q < ir.GetNPoints(); q++) {
      const auto& ip = ir.IntPoint(q);
      Tr->SetIntPoint(&ip);
      const double w = ip.weight * Tr->Weight();
      S.Eval(T, *Tr, ip);
      double tr = 0.0;
      for (int i = 0; i < dim; i++) {
        tr += T(i, i);
      }
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
          const double d = T(i, j) - (i == j ? tr / dim : 0.0);
          full2 += w * T(i, j) * T(i, j);
          dev2 += w * d * d;
        }
      }
    }
  }
  full = std::sqrt(full2);
  dev = std::sqrt(dev2);
}

// Self-gravity of the uniform disc, f = rho grad Phi0 = 2 pi G rho^2 x.
void DiscBodyForce(const Vector& x, Vector& f) {
  f = x;
  f *= 2.0 * kPi * kG * kRho * kRho;
}

double UniformDiscPressure(const Vector& x) {
  return kPi * kG * kRho * kRho * (1.0 - (x * x));
}

}  // namespace

// Al-Attar & Woodhouse (2010, §3.3): the minimum equilibrium stress solves the equilibrium
// equations weakly, at discretisation level improving with order.
TEST(Background, MinimumNormEquilibriumSatisfiesWeakForm) {
  const int dim = 2;
  Mesh parent(MeshFile(dim).c_str(), 1, 1);
  Array<int> body_attr({1});
  SubMesh body(SubMesh::CreateFromDomain(parent, body_attr));
  VectorFunctionCoefficient f(dim, DiscBodyForce);

  H1_FECollection fec3(3, dim);
  FiniteElementSpace fes3(&body, &fec3, dim);
  std::vector<double> res;
  for (int order : {1, 2}) {
    H1_FECollection fec(order, dim);
    FiniteElementSpace fes(&body, &fec, dim);
    MinimumNormEquilibriumStress T(fes, f);
    res.push_back(WeakEquilibriumResidual(T, f, fes3));
  }
  EXPECT_LT(res[1], 0.6 * res[0]);
  EXPECT_LT(res[1], 5e-2);
}

// Al-Attar & Woodhouse (2010, §3.4) on the uniform disc: the minimum deviatoric equilibrium
// stress recovers the hydrostatic state, T = -p0 1 with
// p0 = pi G rho^2 (1 - r^2), essentially free of deviatoric content.
TEST(Background, MinimumDeviatoricRecoversHydrostatic) {
  const int dim = 2;
  Mesh parent(MeshFile(dim).c_str(), 1, 1);
  Array<int> body_attr({1});
  SubMesh body(SubMesh::CreateFromDomain(parent, body_attr));
  VectorFunctionCoefficient f(dim, DiscBodyForce);

  H1_FECollection fec_u(3, dim), fec_p(2, dim);
  FiniteElementSpace fes_u(&body, &fec_u, dim), fes_p(&body, &fec_p);
  MinimumDeviatoricEquilibriumStress T(fes_u, fes_p, f);

  FunctionCoefficient p0(UniformDiscPressure);
  auto& p = const_cast<GridFunction&>(T.Pressure());
  ConstantCoefficient zero(0.0);
  const double p_err = p.ComputeL2Error(p0) / p.ComputeL2Error(zero);
  EXPECT_LT(p_err, 1e-2);

  double full = 0.0, dev = 0.0;
  StressNorms(T, body, 3, full, dev);
  EXPECT_LT(dev, 1e-2 * full);

  H1_FECollection fec4(4, dim);
  FiniteElementSpace fes4(&body, &fec4, dim);
  EXPECT_LT(WeakEquilibriumResidual(T, f, fes4), 5e-2);
}

// The two generators are each optimal in their own functional: the
// minimum-norm field has the smaller full norm, the minimum-deviatoric
// field the smaller deviatoric norm (a discretisation-tolerant check of
// minimality that needs no analytic solution).
TEST(Background, EquilibriumStressOptimalityOrdering) {
  const int dim = 2;
  Mesh parent(MeshFile(dim).c_str(), 1, 1);
  Array<int> body_attr({1});
  SubMesh body(SubMesh::CreateFromDomain(parent, body_attr));
  VectorFunctionCoefficient f(dim, DiscBodyForce);

  H1_FECollection fec2(2, dim), fec1(1, dim);
  FiniteElementSpace fes_u(&body, &fec2, dim), fes_p(&body, &fec1);
  MinimumNormEquilibriumStress T_mn(fes_u, f);
  MinimumDeviatoricEquilibriumStress T_md(fes_u, fes_p, f);

  double full_mn, dev_mn, full_md, dev_md;
  StressNorms(T_mn, body, 2, full_mn, dev_mn);
  StressNorms(T_md, body, 2, full_md, dev_md);
  EXPECT_LT(full_mn, 1.02 * full_md);
  EXPECT_LT(dev_md, 1.02 * dev_mn);
  // And genuinely different fields: the disc's minimum-norm stress is
  // not hydrostatic.
  EXPECT_GT(dev_mn, 0.05 * full_mn);
}

// Love's obstruction, computed: a homogeneous ELLIPSE admits no
// hydrostatic equilibrium (its self-gravity equipotentials are not
// parallel to its surface), so even the minimum-deviatoric equilibrium
// stress carries genuine deviatoric content, of order the ellipticity.
// The interior attraction of the homogeneous elliptical cylinder is
// linear: grad Phi = 4 pi G rho / (a + b) * (b x, a y).
TEST(Background, EllipseNeedsDeviatoricStress) {
  const int dim = 2;
  const double a = 1.2, b = 1.0 / 1.2;
  Mesh parent(MeshFile(dim).c_str(), 1, 1);
  parent.Transform([](const Vector& x, Vector& y) {
    y.SetSize(2);
    y[0] = 1.2 * x[0];
    y[1] = x[1] / 1.2;
  });
  Array<int> body_attr({1});
  SubMesh body(SubMesh::CreateFromDomain(parent, body_attr));
  VectorFunctionCoefficient f(dim, [a, b](const Vector& x, Vector& v) {
    const double c = 4.0 * kPi * kG * kRho * kRho / (a + b);
    v.SetSize(2);
    v[0] = c * b * x[0];
    v[1] = c * a * x[1];
  });

  H1_FECollection fec_u(3, dim), fec_p(2, dim);
  FiniteElementSpace fes_u(&body, &fec_u, dim), fes_p(&body, &fec_p);
  MinimumDeviatoricEquilibriumStress T(fes_u, fes_p, f);

  double full = 0.0, dev = 0.0;
  StressNorms(T, body, 3, full, dev);
  EXPECT_GT(dev, 1e-2 * full);   // unavoidable deviatoric stress
  EXPECT_LT(dev, 0.5 * full);    // still pressure-dominated

  H1_FECollection fec4(4, dim);
  FiniteElementSpace fes4(&body, &fec4, dim);
  EXPECT_LT(WeakEquilibriumResidual(T, f, fes4), 5e-2);
}

// The relabelled (mapped) generator mode: for the exact linear ellipse
// map the pulled-back problems on the reference disc are the SAME
// discrete systems as the unmapped problems on the transformed mesh, so
// the mapped Eval (the second PK pullback S = J F^{-1} T(phi(x)) F^{-T})
// must reproduce the hand-computed pullback of the unmapped stress,
// element by element, to solver tolerance -- the change-of-variables
// identity that underwrites referential shape optimisation.
TEST(Background, MappedGeneratorsMatchTransformedMesh) {
  const int dim = 2;
  const double a = 1.2, b = 1.0 / 1.2;
  Mesh parent(MeshFile(dim).c_str(), 1, 1);
  Array<int> body_attr({1});
  SubMesh body(SubMesh::CreateFromDomain(parent, body_attr));

  // The exact linear map and the physical (ellipse) body force.
  CallableDiffeomorphism phi(
      dim,
      [a, b](const Vector& x, Vector& y) {
        y.SetSize(2);
        y(0) = a * x(0);
        y(1) = b * x(1);
      },
      [a, b](const Vector&, DenseMatrix& F) {
        F = 0.0;
        F(0, 0) = a;
        F(1, 1) = b;
      });
  auto force = [a, b](const Vector& y, Vector& v) {
    const double c = 4.0 * kPi * kG * kRho * kRho / (a + b);
    v.SetSize(2);
    v[0] = c * b * y[0];
    v[1] = c * a * y[1];
  };
  TransformedVectorFunctionCoefficient f_comp(phi, force);
  VectorFunctionCoefficient f_phys(dim, force);

  // Mapped generators on the reference disc.
  H1_FECollection fec_u(3, dim), fec_p(2, dim);
  FiniteElementSpace fes_u(&body, &fec_u, dim), fes_p(&body, &fec_p);
  MinimumNormEquilibriumStress S_mn(fes_u, f_comp, nullptr, &phi);
  MinimumDeviatoricEquilibriumStress S_md(fes_u, fes_p, f_comp, nullptr,
                                          &phi);

  // Unmapped generators on the transformed copy (exact geometry for a
  // linear map; element indices correspond one to one).
  Mesh mapped(body);
  mapped.Transform([](const Vector& x, Vector& y) {
    y.SetSize(2);
    y(0) = 1.2 * x(0);
    y(1) = x(1) / 1.2;
  });
  FiniteElementSpace mfes_u(&mapped, &fec_u, dim), mfes_p(&mapped, &fec_p);
  MinimumNormEquilibriumStress T_mn(mfes_u, f_phys);
  MinimumDeviatoricEquilibriumStress T_md(mfes_u, mfes_p, f_phys);

  DenseMatrix F(dim), Fi(dim);
  F = 0.0;
  F(0, 0) = a;
  F(1, 1) = b;
  Fi = 0.0;
  Fi(0, 0) = 1.0 / a;
  Fi(1, 1) = 1.0 / b;
  const double J = 1.0;  // area-preserving

  double max_mn = 0.0, max_md = 0.0, scale_mn = 0.0, scale_md = 0.0;
  DenseMatrix S1, T1, tmp(dim), E(dim);
  for (int e = 0; e < body.GetNE(); e++) {
    auto* Tr = body.GetElementTransformation(e);
    const auto& ip = Geometries.GetCenter(body.GetElementGeometry(e));
    Tr->SetIntPoint(&ip);
    auto* Tm = mapped.GetElementTransformation(e);
    Tm->SetIntPoint(&ip);

    auto compare = [&](MatrixCoefficient& mapped_gen,
                       MatrixCoefficient& plain_gen, double& max_d,
                       double& scale) {
      mapped_gen.Eval(S1, *Tr, ip);
      plain_gen.Eval(T1, *Tm, ip);
      // Expected: S = J F^{-1} T F^{-T}.
      Mult(Fi, T1, tmp);
      MultABt(tmp, Fi, E);
      E *= J;
      scale = std::max(scale, E.MaxMaxNorm());
      E -= S1;
      max_d = std::max(max_d, E.MaxMaxNorm());
    };
    compare(S_mn, T_mn, max_mn, scale_mn);
    compare(S_md, T_md, max_md, scale_md);
  }
  EXPECT_LT(max_mn, 1e-7 * scale_mn);
  EXPECT_LT(max_md, 1e-7 * scale_md);
}

// The analytic buffer-taper rule: phi = x + t(r)(xi - x) with the cubic
// smoothstep. Exact equality with xi inside, the exact identity outside,
// the closed-form blend in between, and an independent differentiation
// path (the gradient of the nodal interpolant) agreeing at
// interpolation level.
TEST(Background, TaperedDiffeomorphismBlends) {
  const int dim = 2;
  Mesh mesh(MeshFile(dim).c_str(), 1, 1);

  const double c = 0.08;
  auto f = [c](double r) { return 1.0 + c * std::exp(-r * r); };
  auto df = [c](double r) { return -2.0 * r * c * std::exp(-r * r); };
  RadialDiffeomorphism xi(dim, f, df);
  const double r0 = 0.9, r1 = 1.15;
  TaperedDiffeomorphism phi(xi, r0, r1);

  int n_in = 0, n_mid = 0, n_out = 0;
  Vector x(dim), v, xiv(dim);
  DenseMatrix F, E(dim);
  for (int e = 0; e < mesh.GetNE(); e++) {
    auto* T = mesh.GetElementTransformation(e);
    const auto& ip = Geometries.GetCenter(mesh.GetElementGeometry(e));
    T->SetIntPoint(&ip);
    T->Transform(ip, x);
    const double r = x.Norml2();

    phi.Eval(v, *T, ip);
    phi.EvalGradient(F, *T, ip);
    EXPECT_GT(F.Det(), 0.0);

    // The independently coded blend.
    double t = 1.0, dt = 0.0;
    if (r >= r1) {
      t = 0.0;
    } else if (r > r0) {
      const double s = (r - r0) / (r1 - r0);
      t = 1.0 - s * s * (3.0 - 2.0 * s);
      dt = -6.0 * s * (1.0 - s) / (r1 - r0);
    }
    xiv = x;
    xiv *= f(r);
    for (int i = 0; i < dim; i++) {
      EXPECT_NEAR(v(i), x(i) + t * (xiv(i) - x(i)), 1e-13);
      for (int j = 0; j < dim; j++) {
        const double Fxi =
            (i == j ? f(r) : 0.0) + df(r) / r * x(i) * x(j);
        E(i, j) = (i == j ? 1.0 : 0.0) + t * (Fxi - (i == j ? 1.0 : 0.0)) +
                  dt / r * (xiv(i) - x(i)) * x(j);
        EXPECT_NEAR(F(i, j), E(i, j), 1e-13);
      }
    }
    (r < r0 ? n_in : (r > r1 ? n_out : n_mid))++;
  }
  ASSERT_GT(n_in, 0);
  ASSERT_GT(n_mid, 0);
  ASSERT_GT(n_out, 0);

  // Independent differentiation: the interpolant's discrete gradient.
  auto gd = Interpolate(phi, mesh);
  double max_dF = 0.0;
  for (int e = 0; e < mesh.GetNE(); e++) {
    auto* T = mesh.GetElementTransformation(e);
    const auto& ip = Geometries.GetCenter(mesh.GetElementGeometry(e));
    T->SetIntPoint(&ip);
    DenseMatrix Fa, Fd;
    phi.EvalGradient(Fa, *T, ip);
    gd.EvalGradient(Fd, *T, ip);
    Fa -= Fd;
    max_dF = std::max(max_dF, Fa.MaxMaxNorm());
  }
  EXPECT_LT(max_dF, 2e-2);
}

// The elliptic buffer-taper rule: a mapping non-trivial on the physical
// surface, extended harmonically through the buffer. Body values are the
// interpolant of xi, the buffer obeys the maximum principle, the
// displacement dies towards the DtN sphere, and the result is a
// diffeomorphism.
TEST(Background, HarmonicExtensionMapping) {
  const int dim = 2;
  Mesh mesh(MeshFile(dim).c_str(), 1, 1);
  Vector bb_min, bb_max;
  mesh.GetBoundingBox(bb_min, bb_max);
  const double r_out = bb_max.Normlinf();

  // Non-identity ON the surface r = 1 (|h| = c q(1)^2 there).
  const double c = 0.05;
  auto q = [](double r) { return 1.0 - r * r / 1.44; };
  auto f = [c, q](double r) { return 1.0 + c * q(r) * q(r); };
  auto df = [c, q](double r) {
    return c * 2.0 * q(r) * (-2.0 * r / 1.44);
  };
  RadialDiffeomorphism xi(dim, f, df);
  const double trace = c * q(1.0) * q(1.0);  // |h| at r = 1

  Array<int> body_attr({1}), buffer_attr({2});
  auto phi = NewHarmonicExtensionMapping(mesh, 2, xi, body_attr, buffer_attr);
  const GridFunction& h = phi.Displacement();

  double body_err = 0.0, buffer_max = 0.0, inner_buffer_max = 0.0;
  Vector x(dim), hv;
  for (int e = 0; e < mesh.GetNE(); e++) {
    auto* T = mesh.GetElementTransformation(e);
    const auto& ip = Geometries.GetCenter(mesh.GetElementGeometry(e));
    T->SetIntPoint(&ip);
    T->Transform(ip, x);
    const double r = x.Norml2();
    h.GetVectorValue(*T, ip, hv);

    DenseMatrix F;
    phi.EvalGradient(F, *T, ip);
    EXPECT_GT(F.Det(), 0.0);

    if (mesh.GetAttribute(e) == 1) {
      // h = (f(r) - 1) x on the body, to interpolation error.
      for (int i = 0; i < dim; i++) {
        body_err = std::max(body_err,
                            std::abs(hv(i) - (f(r) - 1.0) * x(i)));
      }
    } else {
      buffer_max = std::max(buffer_max, hv.Norml2());
      if (r < 0.5 * (1.0 + r_out)) {
        inner_buffer_max = std::max(inner_buffer_max, hv.Norml2());
      }
    }
  }
  EXPECT_LT(body_err, 1e-4);
  EXPECT_LT(buffer_max, 1.05 * trace);   // maximum principle
  EXPECT_GT(inner_buffer_max, 0.1 * trace);  // non-trivial extension

  // The displacement dies towards the DtN sphere.
  DenseMatrix pts(dim, 8);
  for (int i = 0; i < 8; i++) {
    const double th = 2.0 * kPi * i / 8.0 + 0.05;
    pts(0, i) = 0.999 * r_out * std::cos(th);
    pts(1, i) = 0.999 * r_out * std::sin(th);
  }
  Array<int> elem;
  Array<IntegrationPoint> ips;
  mesh.FindPoints(pts, elem, ips, false);
  int n_found = 0;
  for (int i = 0; i < 8; i++) {
    if (elem[i] < 0) {
      continue;
    }
    n_found++;
    h.GetVectorValue(elem[i], ips[i], hv);
    EXPECT_LT(hv.Norml2(), 0.05 * trace);
  }
  ASSERT_GT(n_found, 4);
}
