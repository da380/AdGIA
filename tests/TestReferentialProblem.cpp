#include "MixedProblemTestCommon.hpp"
#include "TestCommon.hpp"

/*
  Tests for LinearQuasiStaticReferentialSelfGravitatingProblem, the general linearised
  referential problem (doc/gravitating_elasticity.md, §3.1 "The linearised
  referential system"), on the canned uniform-body and two-layer meshes.

  - Translation null pairs (t, 0) are EXACT discrete null vectors of the
    full block operator (every term vanishes pointwise for Du = 0), for a
    non-trivial equilibrium mapping and arbitrary S_e; the rotation pairs
    (W phi_e, 0) are near-null for a consistent hydrostatic background,
    with residuals decreasing with order.
  - Hydrostatic cross-check: at phi_e = id on the uniform disc, with
    S_e = -p0(r) 1 and the bare moduli from BareElasticTensorCoefficient(p0),
    the solution must map onto the mixed class's
    (LinearQuasiStaticMixedSelfGravitatingProblem) under the change of
    variables: u agrees directly (same space, same rigid gauge), and
    zeta1 = phi1 + u.grad(Phi0) on the body — to the two discretisations'
    common accuracy, improving with order. Two different prescribed vacuum
    extensions agree in the body observables (the extension is gauge).
  - The ball-wide vacuum extension (SetVacuumExtension) characterised in
    its biased O(eps) regime.
  - Unit checks of the relabelling transformation coefficients
    (CongruenceMatrix, PullbackStressCoefficient).
  - Fluid relabelling: a divergence-free azimuthal relabelling inside a
    fluid core is a (near-)null direction only with the bare-moduli
    conversion (mu_b = p0 in the fluid).
  - The gauged fluid in the referential class against the mixed class in
    gauged mode, on a two-layer disc.
  - Pre-stress in loading: the minimum-deviatoric equilibrium stress of an
    elliptical body against its quasi-hydrostatic truncation.
  - Relabelled equilibrium: the same hydrostatic physics described from an
    interior-relabelled reference reproduces the phi_e = id solution under
    composition.
*/

namespace {

using namespace self_grav_test;

// 2-D uniform-disc hydrostatic pressure, p0(r) = pi G rho^2 (R^2 - r^2)
// for the 2-D Poisson convention grad^2 Phi = 4 pi G rho.
double UniformDiscPressure(const Vector& x) {
  const double r2 = x * x;
  return std::numbers::pi * kG * kRho * kRho * (1.0 - r2);
}

CallableDiffeomorphism IdentityMap(int dim) {
  return CallableDiffeomorphism(
      dim, [](const Vector& x, Vector& y) { y = x; },
      [](const Vector&, DenseMatrix& F) {
        F = 0.0;
        for (int i = 0; i < F.Height(); i++) {
          F(i, i) = 1.0;
        }
      });
}

// A radial map x -> (1 + c g(r)) x with g smooth, supported in r < 1.1:
// non-trivial on the body, the identity at and beyond the DtN sphere.
CallableDiffeomorphism TaperedMap(int dim, double c) {
  auto g = [](double r) {
    if (r >= 1.1) {
      return 0.0;
    }
    const double s = r / 1.1;
    const double q = 1.0 - s * s;
    return q * q;
  };
  auto dg = [](double r) {
    if (r >= 1.1) {
      return 0.0;
    }
    const double s = r / 1.1;
    return -4.0 * s * (1.0 - s * s) / 1.1;
  };
  return CallableDiffeomorphism(
      dim,
      [c, g](const Vector& x, Vector& y) {
        y = x;
        y *= 1.0 + c * g(x.Norml2());
      },
      [c, g, dg](const Vector& x, DenseMatrix& F) {
        const int dim = x.Size();
        const double r = x.Norml2();
        F = 0.0;
        const double f = 1.0 + c * g(r);
        for (int i = 0; i < dim; i++) {
          F(i, i) = f;
        }
        if (r > 0.0) {
          const double df = c * dg(r) / r;
          for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
              F(i, j) += df * x(i) * x(j);
            }
          }
        }
      });
}

struct Setting {
  std::unique_ptr<Mesh> parent;
  std::unique_ptr<SubMesh> body;
  std::unique_ptr<H1_FECollection> fec;
  std::unique_ptr<FiniteElementSpace> fes_u, fes_zeta;

  Setting(int dim, int order) {
    parent = std::make_unique<Mesh>(MeshFile(dim).c_str(), 1, 1);
    Array<int> attrs({1});
    body = std::make_unique<SubMesh>(SubMesh::CreateFromDomain(*parent, attrs));
    fec = std::make_unique<H1_FECollection>(order, dim);
    fes_u = std::make_unique<FiniteElementSpace>(body.get(), fec.get(), dim);
    fes_zeta = std::make_unique<FiniteElementSpace>(parent.get(), fec.get());
  }
};

}  // namespace

TEST(ReferentialProblem, TranslationsAreExactNullPairs) {
  const int dim = 2, order = 2;
  Setting s(dim, order);

  // A non-trivial equilibrium mapping and a deliberately arbitrary
  // (non-equilibrium) S_e: translations must still be exactly null.
  auto phi = TaperedMap(dim, 0.08);
  ConstantCoefficient lam(1.1), mu(0.6);
  IsotropicElasticTensorCoefficient C(dim, lam, mu);
  MatrixFunctionCoefficient S(dim, [](const Vector& x, DenseMatrix& S) {
    S.SetSize(x.Size());
    S = 0.0;
    S(0, 0) = -0.2 + 0.1 * x(1);
    S(1, 1) = -0.3;
    S(0, 1) = S(1, 0) = 0.05 * x(0);
  });
  ReferentialElasticRheology rheology(dim, C, S, phi);
  ConstantCoefficient rho(kRho);
  LinearQuasiStaticReferentialSelfGravitatingProblem problem(s.fes_u.get(), s.fes_zeta.get(),
                                              rheology, rho, kG, kDtNDegree);

  const auto res = problem.RigidPairResiduals();
  ASSERT_EQ(static_cast<int>(res.size()), dim == 2 ? 3 : 6);
  for (int i = 0; i < dim; i++) {
    EXPECT_LT(res[i], 1e-12);  // translations: exact
  }
}

TEST(ReferentialProblem, RotationResidualDecreasesWithOrder) {
  const int dim = 2;
  std::vector<double> rot;
  for (int order : {1, 2}) {
    Setting s(dim, order);
    auto phi = IdentityMap(dim);
    ConstantCoefficient kappa(kKappa), mu(kMu);
    FunctionCoefficient p0(UniformDiscPressure);
    auto C_eff = IsotropicElasticTensorCoefficient::FromBulkModulus(dim, kappa,
                                                                    mu);
    BareElasticTensorCoefficient C(dim, C_eff, p0);
    MatrixFunctionCoefficient S(dim, [](const Vector& x, DenseMatrix& S) {
      S.SetSize(x.Size());
      S = 0.0;
      const double p = UniformDiscPressure(x);
      for (int i = 0; i < x.Size(); i++) {
        S(i, i) = -p;
      }
    });
    ReferentialElasticRheology rheology(dim, C, S, phi);
    ConstantCoefficient rho(kRho);
    LinearQuasiStaticReferentialSelfGravitatingProblem problem(
        s.fes_u.get(), s.fes_zeta.get(), rheology, rho, kG, kDtNDegree);
    const auto res = problem.RigidPairResiduals();
    rot.push_back(res.back());
  }
  EXPECT_LT(rot[1], 0.5 * rot[0]);
}

TEST(ReferentialProblem, HydrostaticCrossCheck2D) {
  const int dim = 2;
  std::vector<double> u_diff, z_diff;
  for (int order : {1, 2}) {
    Setting s(dim, order);

    // Reference solution from the mixed class (Eulerian potential):
    // PREM-convention moduli go in directly.
    ConstantCoefficient kappa(kKappa), mu(kMu), rho(kRho);
    IsotropicElasticRheology e_rheology(dim, kappa, mu);
    FunctionCoefficient sigma(SurfaceLoad);
    auto surface = SurfaceMarker(*s.body);
    LinearQuasiStaticMixedSelfGravitatingProblem eulerian(
        s.fes_u.get(), s.fes_zeta.get(), e_rheology, rho, kG, kDtNDegree);
    eulerian.SetSurfaceLoad(sigma, surface);
    eulerian.SetRelTol(1e-11);
    eulerian.AssembleForce(0.0);
    ASSERT_TRUE(eulerian.Solve());

    // Referential problem at phi_e = id: bare moduli and S_e = -p0 1.
    auto phi = IdentityMap(dim);
    FunctionCoefficient p0(UniformDiscPressure);
    auto C_eff =
        IsotropicElasticTensorCoefficient::FromBulkModulus(dim, kappa, mu);
    BareElasticTensorCoefficient C(dim, C_eff, p0);
    MatrixFunctionCoefficient S(dim, [](const Vector& x, DenseMatrix& S) {
      S.SetSize(x.Size());
      S = 0.0;
      const double p = UniformDiscPressure(x);
      for (int i = 0; i < x.Size(); i++) {
        S(i, i) = -p;
      }
    });
    ReferentialElasticRheology r_rheology(dim, C, S, phi);
    Setting s2(dim, order);

    // The buffer space and the prescribed radial extension (option (b)).
    Array<int> buffer_attr({2});
    SubMesh buffer(SubMesh::CreateFromDomain(*s2.parent, buffer_attr));
    FiniteElementSpace fes_buffer(&buffer, s2.fec.get(), dim);
    Vector bb_min, bb_max;
    s2.parent->GetBoundingBox(bb_min, bb_max);
    const double r_out = bb_max.Normlinf();

    LinearQuasiStaticReferentialSelfGravitatingProblem referential(
        s2.fes_u.get(), s2.fes_zeta.get(), r_rheology, rho, kG, kDtNDegree);
    auto E = NewRadialVacuumExtension(*s2.fes_u, fes_buffer, 1.0, r_out);
    referential.SetPrescribedVacuumExtension(fes_buffer, *E);
    FunctionCoefficient sigma2(SurfaceLoad);
    auto surface2 = SurfaceMarker(*s2.body);
    referential.SetSurfaceLoad(sigma2, surface2);
    referential.SetRelTol(1e-11);
    referential.AssembleForce(0.0);
    ASSERT_TRUE(referential.Solve());

    // A second, different extension (steeper taper): a gauge choice, so
    // the body observables must agree at the discretisation level or
    // better.
    if (order == 2) {
      Setting s3(dim, order);
      SubMesh buffer3(SubMesh::CreateFromDomain(*s3.parent, buffer_attr));
      FiniteElementSpace fes_buffer3(&buffer3, s3.fec.get(), dim);
      LinearQuasiStaticReferentialSelfGravitatingProblem ref3(
          s3.fes_u.get(), s3.fes_zeta.get(), r_rheology, rho, kG,
          kDtNDegree);
      auto E3 =
          NewRadialVacuumExtension(*s3.fes_u, fes_buffer3, 1.0, r_out, 3.0);
      ref3.SetPrescribedVacuumExtension(fes_buffer3, *E3);
      FunctionCoefficient sigma3(SurfaceLoad);
      auto surface3 = SurfaceMarker(*s3.body);
      ref3.SetSurfaceLoad(sigma3, surface3);
      ref3.SetRelTol(1e-11);
      ref3.AssembleForce(0.0);
      ASSERT_TRUE(ref3.Solve());
      GridFunction du(referential.Displacement());
      du -= ref3.Displacement();
      const double e_gauge =
          L2Norm(du) / L2Norm(referential.Displacement());
      EXPECT_LT(e_gauge, 2e-2);
    }

    // u agrees directly (same spaces, same rigid gauge at phi_e = id).
    {
      GridFunction d(referential.Displacement());
      d -= eulerian.Displacement();
      u_diff.push_back(L2Norm(d) / L2Norm(eulerian.Displacement()));
    }
    // zeta1 = phi1 + u . grad(Phi0) on the body.
    {
      VectorGridFunctionCoefficient u_c(&eulerian.Displacement());
      InnerProductCoefficient advect(u_c, eulerian.BackgroundGravity());
      GridFunctionCoefficient phi1(&eulerian.PotentialOnBody());
      SumCoefficient zeta_expected(phi1, advect);
      GridFunction d(referential.PotentialOnBody());
      GridFunction z(d);
      z.ProjectCoefficient(zeta_expected);
      d -= z;
      // Both potentials carry the 2-D constant gauge, but phi1 + u.g does
      // not: compare modulo the constant.
      d -= d.Sum() / d.Size();
      z_diff.push_back(L2Norm(d) /
                       std::max(1e-30, L2Norm(referential.PotentialOnBody())));
    }
  }
  // With the prescribed vacuum extension the cross-check holds:
  // agreement at the discretisation level, improving with order.
  EXPECT_LT(u_diff[1], 5e-2);
  EXPECT_LT(u_diff[1], 0.6 * u_diff[0]);
  EXPECT_LT(z_diff[1], 5e-2);
  EXPECT_LT(z_diff[1], 0.6 * z_diff[0]);
}

namespace {

// A Mandel tensor restricted to attribute 1 (zero on the buffer).
class BodyOnlyTensor : public MatrixCoefficient {
 public:
  BodyOnlyTensor(int dim, MatrixCoefficient& inner)
      : MatrixCoefficient(inner.GetHeight()), inner_(&inner) {}
  void Eval(DenseMatrix& K, ElementTransformation& T,
            const IntegrationPoint& ip) override {
    if (T.Attribute == 1) {
      inner_->Eval(K, T, ip);
    } else {
      K.SetSize(height, width);
      K = 0.0;
    }
  }

 private:
  MatrixCoefficient* inner_;
};

}  // namespace

// Ball-wide mode (option (a) of doc/gravitating_elasticity.md §3.1): the
// displacement carries the gauge vacuum extension through the buffer under
// a small harmonic stiffness (no refinement sweeps), and the gravity terms
// extend to the DtN sphere. The mode is accurate only in its biased O(eps)
// regime, so the test bounds the error against the mixed-class solution
// and checks that the bias is present and depends on the vacuum epsilon
// (the body displacement at two epsilons, eps_gauge_diff; see the comment
// at the end).
TEST(ReferentialProblem, HydrostaticCrossCheckBallWide2D) {
  const int dim = 2;
  std::vector<double> u_diff, z_diff;
  double eps_gauge_diff = 0.0;
  for (int order : {1, 2}) {
    Setting s(dim, order);

    // Eulerian reference on the body SubMesh.
    ConstantCoefficient kappa(kKappa), mu(kMu), rho(kRho);
    IsotropicElasticRheology e_rheology(dim, kappa, mu);
    FunctionCoefficient sigma(SurfaceLoad);
    auto surface = SurfaceMarker(*s.body);
    LinearQuasiStaticMixedSelfGravitatingProblem eulerian(
        s.fes_u.get(), s.fes_zeta.get(), e_rheology, rho, kG, kDtNDegree);
    eulerian.SetSurfaceLoad(sigma, surface);
    eulerian.SetRelTol(1e-11);
    eulerian.AssembleForce(0.0);
    ASSERT_TRUE(eulerian.Solve());

    // Ball-wide referential problem, on the same parent mesh (the
    // transfers below require it).
    Mesh& parent2 = *s.parent;
    FiniteElementSpace fes_u2(&parent2, s.fec.get(), dim);
    FiniteElementSpace fes_zeta2(&parent2, s.fec.get());

    auto phi = IdentityMap(dim);
    FunctionCoefficient p0(UniformDiscPressure);
    auto C_eff =
        IsotropicElasticTensorCoefficient::FromBulkModulus(dim, kappa, mu);
    BareElasticTensorCoefficient C_bare(dim, C_eff, p0);
    BodyOnlyTensor C(dim, C_bare);
    MatrixFunctionCoefficient S(dim, [](const Vector& x, DenseMatrix& S) {
      S.SetSize(x.Size());
      S = 0.0;
      const double p = std::max(0.0, UniformDiscPressure(x));
      for (int i = 0; i < x.Size(); i++) {
        S(i, i) = -p;
      }
    });
    ReferentialElasticRheology r_rheology(dim, C, S, phi);
    Vector rho_vals(parent2.attributes.Max());
    rho_vals = 0.0;
    rho_vals(0) = kRho;  // body attribute 1; vacuum density zero
    PWConstCoefficient rho_pw(rho_vals);

    auto solve_ball = [&](double eps, GridFunction& u_out,
                          GridFunction& z_out) {
      LinearQuasiStaticReferentialSelfGravitatingProblem problem(
          &fes_u2, &fes_zeta2, r_rheology, rho_pw, kG, kDtNDegree);
      Array<int> buffer(parent2.attributes.Max());
      buffer = 0;
      buffer[1] = 1;  // attribute 2: the buffer shell
      ConstantCoefficient mu_gauge(kMu);
      problem.SetVacuumExtension(buffer, mu_gauge, eps, 0);
      FunctionCoefficient sigma2(SurfaceLoad);
      Array<int> surf2(parent2.bdr_attributes.Max());
      surf2 = 0;
      surf2[0] = 1;  // the body surface
      problem.SetSurfaceLoad(sigma2, surf2);
      problem.SetRelTol(1e-11);
      problem.AssembleForce(0.0);
      ASSERT_TRUE(problem.Solve());
      u_out = problem.Displacement();
      z_out = problem.Potential();
    };
    GridFunction u_ball(&fes_u2), z_ball(&fes_zeta2);
    solve_ball(2e-1, u_ball, z_ball);
    if (order == 2) {
      GridFunction u2(&fes_u2), z2(&fes_zeta2);
      solve_ball(6e-1, u2, z2);
      // Body displacement at a second vacuum epsilon.
      GridFunction du(u_ball);
      du -= u2;
      GridFunction du_body(s.fes_u.get());
      SubMesh::Transfer(du, du_body);
      eps_gauge_diff = L2Norm(du_body) / L2Norm(eulerian.Displacement());
    }

    // Restrict the ball-wide displacement to the body and compare.
    {
      GridFunction u_body(s.fes_u.get());
      SubMesh::Transfer(u_ball, u_body);
      GridFunction d(u_body);
      d -= eulerian.Displacement();
      u_diff.push_back(L2Norm(d) / L2Norm(eulerian.Displacement()));
    }
    // zeta1 = phi1 + u . grad(Phi0) on the body.
    {
      VectorGridFunctionCoefficient u_c(&eulerian.Displacement());
      InnerProductCoefficient advect(u_c, eulerian.BackgroundGravity());
      GridFunctionCoefficient phi1(&eulerian.PotentialOnBody());
      SumCoefficient zeta_expected(phi1, advect);
      GridFunction z_body(&eulerian.PotentialSpaceOnBody());
      SubMesh::Transfer(z_ball, z_body);
      GridFunction z(&eulerian.PotentialSpaceOnBody());
      z.ProjectCoefficient(zeta_expected);
      z_body -= z;
      z_diff.push_back(L2Norm(z_body) / std::max(1e-30, L2Norm(z)));
    }
  }
  // The ball-wide vacuum extension works only in the biased O(eps) regime
  // (doc/gravitating_elasticity.md §3.1; doc/gauge_penalty_iteration.tex,
  // "When it must fail: the spectral condition"). Iterated Tikhonov
  // cannot remove the bias here: the physical operator has no buffer
  // stiffness to compare with (only the zeroth-order gravity terms),
  // while the harmonic penalty scales like 1/h^2, so the contraction
  // factor eps * ||A_S^{-1} Q|| grows with refinement of mesh or order
  // (the refinement diverges at order 2). Contrast the gauged fluid, where
  // penalty and physical stiffness share the h-scaling. The assertions
  // below characterise the biased regime at eps = 2e-1; the
  // prescribed-extension route (option (b)) is the accurate one.
  EXPECT_LT(z_diff[1], 0.25);
  EXPECT_LT(u_diff[1], 1.0);
  EXPECT_GT(u_diff[1] + z_diff[1], 5e-2);  // the bias is present
  // The bias depends on the vacuum epsilon: the body displacement moves
  // by a few per cent between eps = 2e-1 and 6e-1 (about 5e-2 measured).
  EXPECT_GT(eps_gauge_diff, 1e-2);
  EXPECT_LT(eps_gauge_diff, 0.2);
}

namespace {

// An interior relabelling: xi = x (1 + c h(r)) with h = (r(1-r))^2 for
// r < 1 and zero beyond -- the identity on the body surface and the whole
// buffer, C^1 at r = 1, so the physical problem is untouched while every
// volume integrator sees F_e != 1.
CallableDiffeomorphism InteriorMap(int dim, double c) {
  auto h = [](double r) {
    if (r >= 1.0) {
      return 0.0;
    }
    const double q = r * (1.0 - r);
    return q * q;
  };
  auto dh_over_r = [](double r) {
    if (r >= 1.0) {
      return 0.0;
    }
    return 2.0 * (1.0 - r) * (1.0 - 2.0 * r) * r;  // h'(r) = 2r(1-r)(1-2r)
  };
  return CallableDiffeomorphism(
      dim,
      [c, h](const Vector& x, Vector& y) {
        y = x;
        y *= 1.0 + c * h(x.Norml2());
      },
      [c, h, dh_over_r](const Vector& x, DenseMatrix& F) {
        const int dim = x.Size();
        const double r = x.Norml2();
        F = 0.0;
        const double f = 1.0 + c * h(r);
        for (int i = 0; i < dim; i++) {
          F(i, i) = f;
        }
        if (r > 0.0) {
          const double df = c * dh_over_r(r) / (r * r);  // h'(r)/r
          for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
              F(i, j) += df * x(i) * x(j);
            }
          }
        }
      });
}

// Evaluate a grid function at a physical point of its mesh.
bool EvalAt(const GridFunction& g, Mesh& m, const Vector& x, Vector& out) {
  DenseMatrix pt(x.Size(), 1);
  for (int d = 0; d < x.Size(); d++) {
    pt(d, 0) = x(d);
  }
  Array<int> elem;
  Array<IntegrationPoint> ips;
  if (m.FindPoints(pt, elem, ips, false) != 1 || elem[0] < 0) {
    return false;
  }
  const int vdim = g.FESpace()->GetVDim();
  out.SetSize(vdim);
  if (vdim == 1) {
    out(0) = g.GetValue(elem[0], ips[0]);
  } else {
    g.GetVectorValue(elem[0], ips[0], out);
  }
  return true;
}

}  // namespace

// Unit checks of the relabelling transformation coefficients.
TEST(ReferentialProblem, TransformationLawCoefficients) {
  const int dim = 2;
  Setting s(dim, 2);
  auto* T = s.body->GetElementTransformation(0);
  const auto& ip = Geometries.GetCenter(s.body->GetElementGeometry(0));
  T->SetIntPoint(&ip);

  // CongruenceMatrix at an orthogonal F equals RotationMatrix(F^T).
  {
    const double th = 0.7;
    DenseMatrix R(dim);
    R(0, 0) = std::cos(th);
    R(0, 1) = -std::sin(th);
    R(1, 0) = std::sin(th);
    R(1, 1) = std::cos(th);
    DenseMatrix Q1, Q2, Rt(dim);
    Rt.Transpose(R);
    SymmetricTensorBasis::CongruenceMatrix(dim, Rt, Q1);
    SymmetricTensorBasis::RotationMatrix(dim, R, Q2);
    DenseMatrix D(Q1);
    D -= Q2;
    EXPECT_LT(D.MaxMaxNorm(), 1e-14);
  }

  // PullbackStressCoefficient of -p 1 equals -p J C^{-1} (the pull-back
  // diffusion tensor scaled by the pressure).
  {
    auto xi = TaperedMap(dim, 0.08);
    ConstantCoefficient p(0.37);
    MatrixFunctionCoefficient S(dim, [](const Vector& x, DenseMatrix& S) {
      S.SetSize(x.Size());
      S = 0.0;
      for (int i = 0; i < x.Size(); i++) {
        S(i, i) = -0.37;
      }
    });
    PullbackStressCoefficient S_rel(dim, S, xi);
    PullbackDiffusionCoefficient a(xi);
    DenseMatrix S1, A;
    S_rel.Eval(S1, *T, ip);
    a.Eval(A, *T, ip);
    A *= -0.37;
    DenseMatrix D(S1);
    D -= A;
    EXPECT_LT(D.MaxMaxNorm(), 1e-13);
  }
}

// The fluid dictionary test: on a two-layer disc with a uniform
// fluid core, a linearised relabelling supported inside the core (w =
// curl psi, div w = 0 exactly) is a null pair (w, 0) of the full block
// operator -- but only JOINTLY: the material term (whose bare shear
// modulus in the fluid is mu_b = p0, NOT zero -- the seismological
// conversion), the geometric -p0 term and the gravity terms annihilate
// the direction through the equilibrium condition. The residual falls
// with order; dropping the bare conversion (seismological moduli used
// directly) breaks the cancellation by an order of magnitude.
TEST(ReferentialProblem, FluidRelabellingNullPair) {
  const int dim = 2;
  const double r_cmb = 3483.0 / 6371.0;

  // The candidate relabelling: w = curl(psi z) with psi a C^2 radial bump
  // strictly inside the core -- div w = 0 exactly, azimuthal (so it
  // preserves any radial stratification), zero near the CMB.
  const double r0 = 0.15, r1 = 0.85 * r_cmb;
  auto dpsi = [r0, r1](double r) {
    if (r <= r0 || r >= r1) {
      return 0.0;
    }
    const double q = (r - r0) * (r1 - r);
    return 3.0 * q * q * (r1 + r0 - 2.0 * r);
  };
  VectorFunctionCoefficient w_coeff(dim,
                                    [&dpsi](const Vector& x, Vector& w) {
                                      const double r = x.Norml2();
                                      w.SetSize(2);
                                      if (r < 1e-12) {
                                        w = 0.0;
                                        return;
                                      }
                                      const double d = dpsi(r) / r;
                                      w(0) = d * x(1);
                                      w(1) = -d * x(0);
                                    });

  // Part 1: the u-row identity with fully analytic coefficients. The
  // material term (bare moduli: in the fluid mu_b = p0, from the
  // seismological conversion), the geometric -p0 term and the gravity
  // second-variation term must annihilate w JOINTLY, through the
  // equilibrium condition grad p0 = -rho grad zeta0. Dropping the
  // conversion (seismological moduli used directly) leaves an O(p0)
  // defect that does not converge away.
  {
    FunctionCoefficient p0(UniformDiscPressure);
    FunctionCoefficient kappa_c([](const Vector&) { return kKappa; });
    FunctionCoefficient mu_c([r_cmb](const Vector& x) {
      return x.Norml2() < r_cmb ? 0.0 : kMu;
    });
    auto C_seis =
        IsotropicElasticTensorCoefficient::FromBulkModulus(dim, kappa_c,
                                                           mu_c);
    BareElasticTensorCoefficient C_bare(dim, C_seis, p0);
    MatrixFunctionCoefficient S_e(dim, [](const Vector& x, DenseMatrix& S) {
      S.SetSize(x.Size());
      S = 0.0;
      const double p = std::max(0.0, UniformDiscPressure(x));
      for (int i = 0; i < x.Size(); i++) {
        S(i, i) = -p;
      }
    });
    // Uniform disc: grad zeta0 = 2 pi G rho x, analytically.
    VectorFunctionCoefficient g0(dim, [](const Vector& x, Vector& g) {
      g = x;
      g *= 2.0 * std::numbers::pi * kG * kRho;
    });
    IdentityDiffeomorphism id(dim);
    const double scale = 1.0 / (8.0 * std::numbers::pi * kG);

    std::vector<double> res, res_wrong;
    for (int order : {1, 2, 3}) {
      Mesh parent("../data/elastogravity_two_layer_2d.msh", 1, 1);
      Array<int> body_attr({1, 2});
      SubMesh body(SubMesh::CreateFromDomain(parent, body_attr));
      H1_FECollection fec(order, dim);
      FiniteElementSpace fes_u(&body, &fec, dim);
      GridFunction w_gf(&fes_u);
      w_gf.ProjectCoefficient(w_coeff);

      // The metric is the energy along the relabelling orbit,
      // |w K w| normalised by the plain elastic energy of w: for the
      // right dictionary it converges to zero (the orbit is flat); for
      // the wrong one it converges to the finite defect
      // int 2 p0 |sym Dw|^2 -- an h-independent separation.
      ConstantCoefficient kappa_n(kKappa);
      BilinearForm dform(&fes_u);
      dform.AddDomainIntegrator(new VectorDiffusionIntegrator(kappa_n));
      dform.Assemble();
      dform.Finalize();
      Vector tmp(w_gf.Size());
      dform.SpMat().Mult(w_gf, tmp);
      const double energy_scale = InnerProduct(w_gf, tmp);

      auto orbit_energy = [&](MatrixCoefficient& C) {
        BilinearForm k(&fes_u);
        k.AddDomainIntegrator(new MaterialStiffnessIntegrator(C, id));
        k.AddDomainIntegrator(new GeometricStiffnessIntegrator(S_e));
        k.AddDomainIntegrator(new ReferentialGravityIntegrator(id, g0, scale));
        k.Assemble();
        k.Finalize();
        Vector r(w_gf.Size());
        k.SpMat().Mult(w_gf, r);
        return std::abs(InnerProduct(w_gf, r)) / energy_scale;
      };
      res.push_back(orbit_energy(C_bare));
      res_wrong.push_back(orbit_energy(C_seis));
    }
    for (size_t i = 0; i < res.size(); i++) {
      std::cout << "order " << i + 1 << ": orbit energy = " << res[i]
                << ", without conversion = " << res_wrong[i] << "\n";
    }
    EXPECT_LT(res[1], 0.35 * res[0]);         // converging to a flat orbit
    EXPECT_LT(res[2], 0.5 * res[1]);
    EXPECT_GT(res_wrong[2], 10.0 * res[2]);   // the conversion is load-bearing
  }

  // Part 2: the same direction under the FULL block operator of the
  // general class (background module coefficients, discrete zeta0):
  // (w, 0) is near-null with the residual falling with order.
  {
    std::vector<double> res;
    for (int order : {1, 2}) {
      Mesh parent("../data/elastogravity_two_layer_2d.msh", 1, 1);
      Array<int> body_attr({1, 2});
      SubMesh body(SubMesh::CreateFromDomain(parent, body_attr));
      H1_FECollection fec(order, dim);
      FiniteElementSpace fes_u(&body, &fec, dim), fes_zeta(&parent, &fec);
      RadialHydrostaticBackground bg(
          dim, [](double) { return kRho; }, [](double) { return kKappa; },
          [r_cmb](double r) { return r < r_cmb ? 0.0 : kMu; }, kG, 1.0);
      LinearQuasiStaticReferentialSelfGravitatingProblem problem(
          &fes_u, &fes_zeta, bg.Rheology(), bg.Density(), kG, kDtNDegree);
      GridFunction w_gf(&fes_u);
      w_gf.ProjectCoefficient(w_coeff);
      Vector w_true;
      w_gf.GetTrueDofs(w_true);
      res.push_back(problem.NullPairResidual(w_true));
    }
    EXPECT_LT(res[1], 0.7 * res[0]);
  }
}

// The gauged fluid in the general class: the same two-layer physical
// problem through the general referential class (base-class
// SetGaugedFluid, prescribed vacuum extension) and through the mixed
// self-gravitating class (Eulerian potential) in gauged mode. Solid displacement agrees
// directly; the potential through the change of variables
// zeta1 = phi1 + u.grad Phi0 (modulo the 2-D constant).
TEST(ReferentialProblem, GaugedFluidCrossCheck2D) {
  const int dim = 2, order = 2;
  const double r_cmb = 3483.0 / 6371.0;
  const double eps = 1e-2;
  const int nref = 3;

  Mesh parent("../data/elastogravity_two_layer_2d.msh", 1, 1);
  Array<int> body_attr({1, 2}), buffer_attr({3});
  SubMesh body(SubMesh::CreateFromDomain(parent, body_attr));
  SubMesh buffer(SubMesh::CreateFromDomain(parent, buffer_attr));
  H1_FECollection fec(order, dim);
  FiniteElementSpace fes_u(&body, &fec, dim), fes_phi(&parent, &fec);
  FiniteElementSpace fes_u2(&body, &fec, dim), fes_zeta(&parent, &fec);
  FiniteElementSpace fes_buffer(&buffer, &fec, dim);
  Vector bb_min, bb_max;
  parent.GetBoundingBox(bb_min, bb_max);
  const double r_out = bb_max.Normlinf();

  Array<int> fluid_marker(body.attributes.Max());
  fluid_marker = 0;
  fluid_marker[0] = 1;
  Array<int> surface(body.bdr_attributes.Max());
  surface = 0;
  surface[body.bdr_attributes.Max() - 1] = 1;
  ConstantCoefficient rho(kRho), mu_gauge(kKappa);
  FunctionCoefficient sigma([](const Vector& x) {
    const double r = x.Norml2();
    const double c = x[1] / r;
    return 0.02 * (1.0 + (2.0 * c * c - 1.0));
  });

  // Mixed class, gauged mode: seismological moduli, fluid = kappa + no
  // shear.
  FunctionCoefficient kappa_c([](const Vector&) { return kKappa; });
  FunctionCoefficient mu_c([r_cmb](const Vector& x) {
    return x.Norml2() < r_cmb ? 0.0 : kMu;
  });
  IsotropicElasticRheology e_rheology(dim, kappa_c, mu_c);
  LinearQuasiStaticMixedSelfGravitatingProblem eulerian(
      &fes_u, &fes_phi, e_rheology, rho, kG, kDtNDegree);
  eulerian.SetGaugedFluid(fluid_marker, mu_gauge, eps, nref);
  eulerian.SetSurfaceLoad(sigma, surface);
  eulerian.SetRelTol(1e-11);
  eulerian.AssembleForce(0.0);
  ASSERT_TRUE(eulerian.Solve());

  // General referential class: background module (bare conversion,
  // S_e = -p0 1), base-class gauge on the fluid attribute.
  RadialHydrostaticBackground bg(
      dim, [](double) { return kRho; }, [](double) { return kKappa; },
      [r_cmb](double r) { return r < r_cmb ? 0.0 : kMu; }, kG, 1.0);
  LinearQuasiStaticReferentialSelfGravitatingProblem referential(
      &fes_u2, &fes_zeta, bg.Rheology(), bg.Density(), kG, kDtNDegree);
  auto E = NewRadialVacuumExtension(fes_u2, fes_buffer, 1.0, r_out);
  referential.SetPrescribedVacuumExtension(fes_buffer, *E);
  referential.SetGaugedFluid(fluid_marker, mu_gauge, eps, nref);
  referential.SetSurfaceLoad(sigma, surface);
  referential.SetRelTol(1e-11);
  referential.AssembleForce(0.0);
  ASSERT_TRUE(referential.Solve());

  // Solid (mantle) displacement: sample points away from the CMB and the
  // surface; the fluid displacement is gauge and is not compared.
  double du2 = 0.0, un2 = 0.0;
  int n_pts = 0;
  Vector x(dim);
  for (int i = 0; i < 32; i++) {
    const double r = r_cmb + 0.1 + (0.85 - r_cmb) * (i % 8) / 7.0;
    const double th = 2.0 * std::numbers::pi * i / 32.0 + 0.05;
    x(0) = r * std::cos(th);
    x(1) = r * std::sin(th);
    Vector ue, ur;
    if (!EvalAt(eulerian.Displacement(), body, x, ue) ||
        !EvalAt(referential.Displacement(), body, x, ur)) {
      continue;
    }
    n_pts++;
    for (int d = 0; d < dim; d++) {
      du2 += (ur(d) - ue(d)) * (ur(d) - ue(d));
      un2 += ue(d) * ue(d);
    }
  }
  ASSERT_GT(n_pts, 24);
  EXPECT_LT(std::sqrt(du2 / un2), 2e-2);

  // Potential through the change of variables, modulo the 2-D constant.
  {
    VectorGridFunctionCoefficient u_c(&eulerian.Displacement());
    InnerProductCoefficient advect(u_c, eulerian.BackgroundGravity());
    GridFunctionCoefficient phi1(&eulerian.PotentialOnBody());
    SumCoefficient zeta_expected(phi1, advect);
    GridFunction d(referential.PotentialOnBody());
    GridFunction z(d);
    z.ProjectCoefficient(zeta_expected);
    d -= z;
    d -= d.Sum() / d.Size();
    ConstantCoefficient zc(0.0);
    const double rel =
        d.ComputeL2Error(zc) /
        std::max(1e-30, const_cast<GridFunction&>(
                            referential.PotentialOnBody())
                            .ComputeL2Error(zc));
    EXPECT_LT(rel, 2e-2);
  }
}

// Pre-stress in loading (the ellipse benchmark of
// examples/prestress_loading.cpp, in miniature): the loading response
// with the full minimum-deviatoric equilibrium stress versus the
// quasi-hydrostatic approximation (its pressure part only) must differ,
// with the difference growing with ellipticity, and remain a
// perturbation.
TEST(ReferentialProblem, EllipticalPrestressLoading) {
  const int dim = 2, order = 2;
  const double G = 0.1;
  std::vector<double> du_rel;
  for (double e : {0.03, 0.1}) {
    const double a = 1.0 + e, b = 1.0 / a;
    Setting s(dim, order);
    Array<int> buffer_attr({2});
    SubMesh buffer(SubMesh::CreateFromDomain(*s.parent, buffer_attr));
    FiniteElementSpace fes_buffer(&buffer, s.fec.get(), dim);
    Vector bb_min, bb_max;
    s.parent->GetBoundingBox(bb_min, bb_max);
    const double r_out = bb_max.Normlinf();

    // The exact linear ellipse map (for the generator) and the
    // parameter-blended taper phi = (a(r) x, y/a(r)) for the problem.
    CallableDiffeomorphism lin(
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
    auto af = [e, r_out](double r, double& av, double& da) {
      if (r <= 1.0) {
        av = 1.0 + e;
        da = 0.0;
        return;
      }
      if (r >= r_out) {
        av = 1.0;
        da = 0.0;
        return;
      }
      const double w = r_out - 1.0;
      const double t = (r - 1.0) / w;
      av = 1.0 + e * (1.0 - t * t * (3.0 - 2.0 * t));
      da = -e * 6.0 * t * (1.0 - t) / w;
    };
    CallableDiffeomorphism phi_e(
        dim,
        [af](const Vector& x, Vector& y) {
          double av, da;
          af(x.Norml2(), av, da);
          y.SetSize(2);
          y(0) = av * x(0);
          y(1) = x(1) / av;
        },
        [af](const Vector& x, DenseMatrix& F) {
          const double r = x.Norml2();
          double av, da;
          af(r, av, da);
          F.SetSize(2);
          F = 0.0;
          F(0, 0) = av;
          F(1, 1) = 1.0 / av;
          if (r > 0.0 && da != 0.0) {
            F(0, 0) += da * x(0) * x(0) / r;
            F(0, 1) = da * x(0) * x(1) / r;
            F(1, 0) = -da * x(0) * x(1) / (av * av * r);
            F(1, 1) -= da * x(1) * x(1) / (av * av * r);
          }
        });

    // The realistic stress (mapped min-deviatoric generator, exact
    // elliptical-cylinder gravity) and its quasi-hydrostatic truncation.
    auto force = [a, b, G](const Vector& y, Vector& v) {
      const double c = 4.0 * std::numbers::pi * G * kRho * kRho / (a + b);
      v.SetSize(2);
      v[0] = c * b * y[0];
      v[1] = c * a * y[1];
    };
    TransformedVectorFunctionCoefficient f_comp(lin, force);
    H1_FECollection fec_gu(order + 1, dim), fec_gp(order, dim);
    FiniteElementSpace fes_gu(s.body.get(), &fec_gu, dim),
        fes_gp(s.body.get(), &fec_gp);
    MinimumDeviatoricEquilibriumStress S_full(fes_gu, fes_gp, f_comp,
                                              nullptr, &lin);
    GridFunctionCoefficient p_c(&S_full.Pressure());
    ProductCoefficient minus_p(-1.0, p_c);
    PullbackDiffusionCoefficient a_e(lin);
    ScalarMatrixProductCoefficient S_hydro(minus_p, a_e);

    ConstantCoefficient kappa(kKappa), mu(kMu), rho(kRho);
    auto C =
        IsotropicElasticTensorCoefficient::FromBulkModulus(dim, kappa, mu);
    TransformedFunctionCoefficient sigma_comp(
        lin, [](const Vector& y) {
          return 0.02 * (1.0 + y(1) * y(1) / (y * y));
        });
    NansonAreaCoefficient nu(lin);
    ProductCoefficient sigma_ref(sigma_comp, nu);
    auto surface = SurfaceMarker(*s.body);

    auto run = [&](MatrixCoefficient& S, GridFunction& u_out) {
      ReferentialElasticRheology rheology(dim, C, S, phi_e);
      LinearQuasiStaticReferentialSelfGravitatingProblem problem(
          s.fes_u.get(), s.fes_zeta.get(), rheology, rho, G, kDtNDegree);
      auto E = NewRadialVacuumExtension(*s.fes_u, fes_buffer, 1.0, r_out);
      problem.SetPrescribedVacuumExtension(fes_buffer, *E);
      problem.SetSurfaceLoad(sigma_ref, surface);
      problem.SetRelTol(1e-10);
      problem.AssembleForce(0.0);
      ASSERT_TRUE(problem.Solve());
      u_out = problem.Displacement();
    };
    GridFunction uA(s.fes_u.get()), uB(s.fes_u.get());
    run(S_full, uA);
    run(S_hydro, uB);
    GridFunction du(uA);
    du -= uB;
    du_rel.push_back(du.Norml2() / uA.Norml2());
  }
  // The deviatoric pre-stress is a genuine, ellipticity-scaled effect.
  EXPECT_GT(du_rel[0], 1e-4);
  EXPECT_LT(du_rel[1], 0.2);
  EXPECT_GT(du_rel[1], 1.5 * du_rel[0]);
}

// The same spherical hydrostatic physics described from a relabelled
// reference (interior-only relabelling: identity on the surface and the
// buffer). The transformed coefficients (C~, S~, rho~) with phi_e = xi
// must reproduce the phi_e = id solution under composition,
// u~(x) = u(xi(x)), zeta~ = zeta o xi (modulo the 2-D constant), at the
// level of the mesh's geometric-interpolation floor (see the comment at
// the end). Both legs run on the background-state module (background.hpp),
// whose generators supply the moduli conversion and the transformation
// laws (their coefficient-level checks are in TestBackground).
TEST(ReferentialProblem, RelabelledEquilibrium2D) {
  const int dim = 2;
  std::vector<double> u_err, z_err;
  RadialHydrostaticBackground bg(
      dim, [](double) { return kRho; }, [](double) { return kKappa; },
      [](double) { return kMu; }, kG, 1.0);
  for (int order : {1, 2}) {
    Array<int> buffer_attr({2});

    // Reference: phi_e = id.
    Setting s(dim, order);
    SubMesh buffer(SubMesh::CreateFromDomain(*s.parent, buffer_attr));
    FiniteElementSpace fes_buffer(&buffer, s.fec.get(), dim);
    Vector bb_min, bb_max;
    s.parent->GetBoundingBox(bb_min, bb_max);
    const double r_out = bb_max.Normlinf();
    LinearQuasiStaticReferentialSelfGravitatingProblem ref(s.fes_u.get(), s.fes_zeta.get(),
                                            bg.Rheology(), bg.Density(), kG,
                                            kDtNDegree);
    auto E = NewRadialVacuumExtension(*s.fes_u, fes_buffer, 1.0, r_out);
    ref.SetPrescribedVacuumExtension(fes_buffer, *E);
    FunctionCoefficient sigma(SurfaceLoad);
    auto surface = SurfaceMarker(*s.body);
    ref.SetSurfaceLoad(sigma, surface);
    ref.SetRelTol(1e-11);
    ref.AssembleForce(0.0);
    ASSERT_TRUE(ref.Solve());

    // Relabelled: phi_e = xi, coefficients through the generator.
    Setting s2(dim, order);
    auto xi = InteriorMap(dim, 0.3);
    RelabelledBackground rel_bg(bg, xi);
    SubMesh buffer2(SubMesh::CreateFromDomain(*s2.parent, buffer_attr));
    FiniteElementSpace fes_buffer2(&buffer2, s2.fec.get(), dim);
    LinearQuasiStaticReferentialSelfGravitatingProblem rel(
        s2.fes_u.get(), s2.fes_zeta.get(), rel_bg.Rheology(),
        rel_bg.Density(), kG, kDtNDegree);
    auto E2 = NewRadialVacuumExtension(*s2.fes_u, fes_buffer2, 1.0, r_out);
    rel.SetPrescribedVacuumExtension(fes_buffer2, *E2);
    FunctionCoefficient sigma2(SurfaceLoad);
    auto surface2 = SurfaceMarker(*s2.body);
    rel.SetSurfaceLoad(sigma2, surface2);
    rel.SetRelTol(1e-11);
    rel.AssembleForce(0.0);
    ASSERT_TRUE(rel.Solve());

    // Compare at sample points: u~(x) = u(xi(x)), zeta~ = zeta o xi
    // (modulo the constant).
    double du2 = 0.0, un2 = 0.0;
    std::vector<double> dz;
    double zn2 = 0.0;
    Vector x(dim), y(dim), a, b;
    int n_pts = 0;
    for (int i = 0; i < 40; i++) {
      const double r = 0.15 + 0.75 * (i % 8) / 7.0;
      const double th = 2.0 * std::numbers::pi * i / 40.0 + 0.1;
      x(0) = r * std::cos(th);
      x(1) = r * std::sin(th);
      {  // y = xi(x), the same formula as InteriorMap(dim, 0.3)
        const double q = r * (1.0 - r);
        y = x;
        y *= 1.0 + 0.3 * q * q;
      }
      Vector ur, zr, urel, zrel;
      if (!EvalAt(rel.Displacement(), *s2.body, x, urel) ||
          !EvalAt(ref.Displacement(), *s.body, y, ur) ||
          !EvalAt(rel.Potential(), *s2.parent, x, zrel) ||
          !EvalAt(ref.Potential(), *s.parent, y, zr)) {
        continue;
      }
      n_pts++;
      for (int d = 0; d < dim; d++) {
        du2 += (urel(d) - ur(d)) * (urel(d) - ur(d));
        un2 += ur(d) * ur(d);
      }
      dz.push_back(zrel(0) - zr(0));
      zn2 += zr(0) * zr(0);
    }
    ASSERT_GT(n_pts, 30);
    double dz_mean = 0.0;
    for (double v : dz) {
      dz_mean += v;
    }
    dz_mean /= dz.size();
    double dz2 = 0.0;
    for (double v : dz) {
      dz2 += (v - dz_mean) * (v - dz_mean);
    }
    u_err.push_back(std::sqrt(du2 / un2));
    z_err.push_back(std::sqrt(dz2 / zn2));
  }
  // Agreement at the half-percent level at both orders. The error does
  // not fall with the FIELD order because the mesh geometry is fixed at
  // order 2 and the exact analytic map carries the geometric
  // interpolation error (the exact-F-versus-interpolated-F effect of
  // doc/mappings.md, "The discrete change-of-variables identity"): the
  // floor is the mesh's, not the fields'.
  EXPECT_LT(u_err[0], 1.5e-2);
  EXPECT_LT(u_err[1], 1.5e-2);
  EXPECT_LT(z_err[0], 1.5e-2);
  EXPECT_LT(z_err[1], 1.5e-2);
}
