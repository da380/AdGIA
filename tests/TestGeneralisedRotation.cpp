#include <numbers>

#include "MixedProblemTestCommon.hpp"
#include "TestCommon.hpp"

/*
  Checks of the generalised rotational theory: the traditional theory's
  rigid-gauge sensitivity, shown quantitatively on a model with an
  anisotropic inertia tensor.

  1. The delta-I identity (the note's first identity): the assembled
     omega-row functional of a displacement u equals -omega'.dI(u).W
     with dI the linearised inertia perturbation — checked by
     quadrature for rigid rotations u = b x x, where it reduces to the
     closed form Delta(w') = -w'.[ad_b, I] W.
  2. The gauge shift: re-gauging the solved displacement by b x x
     shifts the traditional omega by exactly D^{-1} [ad_b, I] W — zero
     iff the model's inertia is isotropic (the paper's eq. 20). On the
     triaxial-density model the shift is O(1) of omega; on the
     spherical model it vanishes at quadrature grade. This is the
     existence argument of Yu et al. §2.4 made concrete.
  3. The Tisserand gauge (the mass-weighted gauge) pins the
     representative: at the gauged solution the relative angular
     momentum int rho x cross u dV vanishes, so the omega of the
     generalised theory is the row's value at THE representative — no
     ambiguity remains.
*/

namespace {

using namespace self_grav_test;

// Triaxial density on the unit ball: distinct principal moments along
// the coordinate axes.
double TriaxialRho(const Vector& x) {
  const double r2 = x * x;
  return 1.0 + 0.3 * (3.0 * x[2] * x[2] / r2 - 1.0) +
         0.2 * (x[0] * x[0] - x[1] * x[1]) / r2;
}

double G0() { return 4.0 * std::numbers::pi * kG * kRho / 3.0; }

double Degree2Mean(const Vector& x) {
  const double r2 = x * x;
  const double c2 = x[2] * x[2] / r2;
  return 3.0 * c2 - 1.0 + 2.0 * x[0] * x[2] / r2 + 0.5;
}

// Quadrature over the body mesh: the inertia tensor of a density, the
// omega-row pairing int rho u . grad psi_k dV of a displacement field,
// and the relative angular momentum int rho x cross u dV.
struct Instrument {
  Mesh* body;
  Coefficient* rho;
  int order;

  DenseMatrix Inertia() {
    DenseMatrix I(3);
    I = 0.0;
    Quad([&](const Vector& x, const Vector&, double w) {
      const double r2 = x * x;
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          I(i, j) += w * ((i == j ? r2 : 0.0) - x[i] * x[j]);
        }
      }
    });
    return I;
  }

  // row_k(u) = int rho u . grad psi_k dV (unit psi_k at Omega).
  Vector Row(VectorCoefficient& u, double Omega) {
    Vector r(3);
    r = 0.0;
    Quad([&](const Vector& x, const Vector& uv, double w) {
      // grad psi_k, k = 1..3 (centrifugal.hpp conventions).
      r[0] += w * Omega * (uv[0] * x[2] + uv[2] * x[0]);
      r[1] += w * Omega * (uv[1] * x[2] + uv[2] * x[1]);
      r[2] += w * Omega * (-2.0 * (uv[0] * x[0] + uv[1] * x[1]));
    },
         &u);
    return r;
  }

  Vector AngularMomentum(VectorCoefficient& u) {
    Vector h(3);
    h = 0.0;
    Quad([&](const Vector& x, const Vector& uv, double w) {
      h[0] += w * (x[1] * uv[2] - x[2] * uv[1]);
      h[1] += w * (x[2] * uv[0] - x[0] * uv[2]);
      h[2] += w * (x[0] * uv[1] - x[1] * uv[0]);
    },
         &u);
    return h;
  }

  template <class F>
  void Quad(F&& f, VectorCoefficient* u = nullptr) {
    Vector x(3), uv(3);
    uv = 0.0;
    for (int e = 0; e < body->GetNE(); e++) {
      auto* T = body->GetElementTransformation(e);
      const auto& ir =
          IntRules.Get(body->GetElementGeometry(e), 2 * order + 4);
      for (int q = 0; q < ir.GetNPoints(); q++) {
        const auto& ip = ir.IntPoint(q);
        T->SetIntPoint(&ip);
        T->Transform(ip, x);
        if (u) {
          u->Eval(uv, *T, ip);
        }
        const double w = ip.weight * T->Weight() * rho->Eval(*T, ip);
        f(x, uv, w);
      }
    }
  }
};

// [ad_b, I] W = b x (I W) - I (b x W).
Vector Commutator(const DenseMatrix& I, const Vector& b, const Vector& W) {
  Vector IW(3), bW(3), IbW(3), c(3);
  I.Mult(W, IW);
  bW[0] = b[1] * W[2] - b[2] * W[1];
  bW[1] = b[2] * W[0] - b[0] * W[2];
  bW[2] = b[0] * W[1] - b[1] * W[0];
  I.Mult(bW, IbW);
  c[0] = b[1] * IW[2] - b[2] * IW[1] - IbW[0];
  c[1] = b[2] * IW[0] - b[0] * IW[2] - IbW[1];
  c[2] = b[0] * IW[1] - b[1] * IW[0] - IbW[2];
  return c;
}

TEST(GeneralisedRotation, GaugeShiftMatchesClosedForm) {
  const int order = 2;
  Mesh parent(MeshFile(3).c_str(), 1, 1);
  SubMesh body(SubMesh::CreateFromDomain(parent, BodyMarker(parent)));
  FunctionCoefficient rho_tri(TriaxialRho);
  ConstantCoefficient rho_sph(1.0);
  const double Omega = 0.1;
  Vector b(3);
  b[0] = 0.3;
  b[1] = -0.2;
  b[2] = 0.1;
  Vector W(3);
  W = 0.0;
  W[2] = Omega;

  // Rigid rotation u = b x x as a coefficient.
  VectorFunctionCoefficient bx(3, [&b](const Vector& x, Vector& v) {
    v.SetSize(3);
    v[0] = b[1] * x[2] - b[2] * x[1];
    v[1] = b[2] * x[0] - b[0] * x[2];
    v[2] = b[0] * x[1] - b[1] * x[0];
  });

  // 1. The closed form: row(b x x) = -[ad_b, I] W, by quadrature on
  // the same mesh — the delta-I identity for the rigid field. The
  // triaxial shift is O(1); the spherical one vanishes relative to it.
  Instrument tri{&body, &rho_tri, order};
  Instrument sph{&body, &rho_sph, order};
  {
    const Vector row = tri.Row(bx, Omega);
    Vector c = Commutator(tri.Inertia(), b, W);
    c *= -1.0;
    Vector d(row);
    d -= c;
    EXPECT_LT(d.Norml2(), 1e-10 * std::max(1.0, c.Norml2()))
        << "triaxial closed form";
    EXPECT_GT(c.Norml2(), 1e-3);  // the shift is genuinely O(1)
    const Vector row_s = sph.Row(bx, Omega);
    // The spherical mesh is faceted, so its discrete inertia is only
    // isotropic to geometry grade; gate against the triaxial scale.
    EXPECT_LT(row_s.Norml2(), 2e-2 * c.Norml2()) << "spherical shift";
  }

  // 2. The solve-level demonstration: the traditional omega of the
  // triaxial model shifts under a rigid re-gauge of u by exactly
  // D^{-1} [ad_b, I] W.
  H1_FECollection fec(order, 3);
  FiniteElementSpace fes_u(&body, &fec, 3), fes_phi(&parent, &fec);
  ConstantCoefficient kappa(kKappa), mu(kMu);
  IsotropicElasticRheology rheology(3, kappa, mu);
  Array<int> surface = SurfaceMarker(body);
  LinearQuasiStaticMixedSelfGravitatingProblem p(&fes_u, &fes_phi, rheology,
                                                 rho_tri, kG, kDtNDegree);
  p.SetRelTol(1e-12);
  Vector moments(3);
  moments[0] = 1.2;
  moments[1] = 1.3;
  moments[2] = 2.0;
  p.SetRotation(Omega, moments);
  FunctionCoefficient sigma(Degree2Mean);
  p.SetSurfaceLoad(sigma, surface);
  p.AssembleForce(0.0);
  ASSERT_TRUE(p.Solve());

  // The row evaluated at u and at u + b x x through the instrument;
  // sigma's row is common and cancels in the shift.
  VectorGridFunctionCoefficient uc(&p.Displacement());
  const Vector row_u = tri.Row(uc, Omega);
  GridFunction shifted(p.Displacement());
  {
    GridFunction bgf(&fes_u);
    bgf.ProjectCoefficient(bx);
    shifted += bgf;
  }
  VectorGridFunctionCoefficient sc(&shifted);
  const Vector row_shifted = tri.Row(sc, Omega);
  // delta(D omega) = row(u + bx) - row(u) = -[ad_b, I]W; the projected
  // b x x is the interpolant, so the agreement is interpolation grade.
  Vector d_row(row_shifted);
  d_row -= row_u;
  Vector c = Commutator(tri.Inertia(), b, W);
  c *= -1.0;
  Vector dd(d_row);
  dd -= c;
  EXPECT_LT(dd.Norml2(), 1e-3 * c.Norml2()) << "solve-level shift";
  // The implied omega shift through D^{-1} is O(1) of omega itself:
  // the ambiguity has no small parameter.
  DenseMatrix D = InertiaMatrix(moments);
  DenseMatrixInverse Dinv(D);
  Vector dw(3);
  Dinv.Mult(c, dw);
  EXPECT_GT(dw.Norml2(), 0.05 * p.AngularVelocity().Norml2());

  // 3. The Tisserand gauge pins the representative. Two certificates:
  // (a) with the mass-weighted gauge the solution carries no
  // relative angular momentum, measured against the O(1) rigid scale
  // |I b| (the momentum a rigid rotation of that size carries);
  // (b) re-pinning after an arbitrary rigid shift recovers the same
  // omega row — the generalised datum is representative-independent.
  LinearQuasiStaticMixedSelfGravitatingProblem q(&fes_u, &fes_phi, rheology,
                                                 rho_tri, kG, kDtNDegree);
  q.SetRelTol(1e-12);
  q.SetMassWeightedGauge(true);
  q.SetRotation(Omega, moments);
  q.SetSurfaceLoad(sigma, surface);
  q.AssembleForce(0.0);
  ASSERT_TRUE(q.Solve());
  VectorGridFunctionCoefficient qc(&q.Displacement());
  const DenseMatrix I_tri = tri.Inertia();
  Vector Ib(3);
  I_tri.Mult(b, Ib);
  const Vector h = tri.AngularMomentum(qc);
  // (a): instrument-quadrature grade against the rigid scale.
  EXPECT_LT(h.Norml2(), 1e-5 * Ib.Norml2());

  // (b): pin u and u + b x x by the instrument's own Tisserand
  // projection (subtract the rigid rotation with I b_fit = h) and
  // compare the rows.
  DenseMatrixInverse Iinv(I_tri);
  auto pinned_row = [&](const GridFunction& g) {
    VectorGridFunctionCoefficient gc(const_cast<GridFunction*>(&g));
    Vector hb = tri.AngularMomentum(gc);
    Vector bf(3);
    Iinv.Mult(hb, bf);
    // row(u - bf x x) = row(u) - row(bf x x); the rigid row through
    // the closed form (part 1 certified it).
    Vector r = tri.Row(gc, Omega);
    Vector cf = Commutator(I_tri, bf, W);
    cf *= -1.0;  // row(bf x x) = -[ad_bf, I] W
    r -= cf;
    return r;
  };
  const Vector r1 = pinned_row(q.Displacement());
  const Vector r2 = pinned_row(shifted);  // u + b x x, re-pinned
  const Vector r3 = pinned_row(p.Displacement());
  Vector d21(r2), d23(r2);
  d21 -= r1;
  d23 -= r3;
  // The shifted field re-pins to ITS OWN representative (u's): r2 and
  // r3 agree to interpolation grade (the projected b x x is the
  // interpolant); r1 belongs to the gauged solve of the SAME system,
  // agreeing at solver-plus-route grade.
  EXPECT_LT(d23.Norml2(), 1e-3 * std::max(r3.Norml2(), 1e-12));
  EXPECT_LT(d21.Norml2(), 1e-3 * std::max(r1.Norml2(), 1e-12));
}

TEST(GeneralisedRotation, FluidRelabellingCertificate) {
  // Fluid-core relabelling: on the two-layer ball's fluid core with its
  // radial density, (a) a compactly-supported fluid relabelling rho v = curl W
  // leaves the omega row invariant (the row of v vanishes) while
  // carrying relative angular momentum 2 int W dV — so the whole-body
  // Tisserand condition is gauge-dependent on fluid-core models; (b)
  // the core's rigid swirls are themselves relabellings, with zero row
  // (the core's inertia is isotropic at geometry grade) and momentum
  // I_core b. Quadrature identities: no solve.
  const int order = 2;
  Mesh parent(TwoLayerMeshFile(3).c_str(), 1, 1);
  Array<int> core_attr({1});
  SubMesh core(SubMesh::CreateFromDomain(parent, core_attr));
  FunctionCoefficient rho_f(FluidDensity);
  const double Omega = 0.1;
  Vector W3(3);
  W3 = 0.0;
  W3[2] = Omega;
  Instrument fl{&core, &rho_f, order};

  // (a) rho v = curl(q(r) c), q = (1 - (r/R0)^2)^3 inside r < R0,
  // compactly supported in the open core.
  const double R0 = 0.9 * kRCmb;
  Vector cdir(3);
  cdir[0] = 0.3;
  cdir[1] = 0.5;
  cdir[2] = -0.2;
  auto q = [R0](double r) {
    const double s = 1.0 - (r / R0) * (r / R0);
    return s > 0.0 ? s * s * s : 0.0;
  };
  auto dq = [R0](double r) {
    const double s = 1.0 - (r / R0) * (r / R0);
    return s > 0.0 ? -6.0 * r / (R0 * R0) * s * s : 0.0;
  };
  // v = curl(W)/rho = (q'(r)/r) (x cross c) / rho.
  VectorFunctionCoefficient v_frl(3, [&](const Vector& x, Vector& v) {
    const double r = x.Norml2();
    const double a =
        r > 0.0 ? dq(r) / (r * FluidDensity(x)) : 0.0;
    v.SetSize(3);
    v[0] = a * (x[1] * cdir[2] - x[2] * cdir[1]);
    v[1] = a * (x[2] * cdir[0] - x[0] * cdir[2]);
    v[2] = a * (x[0] * cdir[1] - x[1] * cdir[0]);
  });
  const Vector row_v = fl.Row(v_frl, Omega);
  const Vector h_v = fl.AngularMomentum(v_frl);
  // The momentum identity h = 2 int W dV = 2 c int q dV, by the same
  // quadrature.
  double qint = 0.0;
  {
    ConstantCoefficient one(1.0);
    Instrument unit{&core, &one, order};
    unit.Quad([&](const Vector& x, const Vector&, double w) {
      qint += w * q(x.Norml2());
    });
  }
  Vector h_pred(cdir);
  h_pred *= 2.0 * qint;
  const double scale = h_pred.Norml2();  // the O(1) rigid scale of v
  ASSERT_GT(scale, 1e-6);
  {
    Vector d(h_v);
    d -= h_pred;
    // Both identities hold through integration by parts, so the two
    // quadratures' errors do not cancel: the grade is the 2*order+4
    // rule on the C^2 bump (measured 6.4e-5 and 1.0e-5 of the scales).
    EXPECT_LT(d.Norml2(), 2e-4 * scale) << "momentum identity";
    // The row of the relabelling vanishes against the O(1) momentum it
    // carries: the omega data are gauge-safe, the Tisserand condition
    // is not — four orders between the two is the certificate.
    EXPECT_LT(row_v.Norml2(), 5e-5 * Omega * scale) << "row invariance";
  }

  // (b) The core swirl b x x: a relabelling of the radial-density
  // spherical core, zero row at the faceted core's isotropy grade,
  // momentum I_core b exactly.
  Vector b(3);
  b[0] = 0.4;
  b[1] = -0.3;
  b[2] = 0.2;
  VectorFunctionCoefficient bx(3, [&b](const Vector& x, Vector& v) {
    v.SetSize(3);
    v[0] = b[1] * x[2] - b[2] * x[1];
    v[1] = b[2] * x[0] - b[0] * x[2];
    v[2] = b[0] * x[1] - b[1] * x[0];
  });
  const DenseMatrix I_core = fl.Inertia();
  Vector Ib(3);
  I_core.Mult(b, Ib);
  const Vector row_s = fl.Row(bx, Omega);
  const Vector h_s = fl.AngularMomentum(bx);
  {
    Vector d(h_s);
    d -= Ib;
    EXPECT_LT(d.Norml2(), 1e-10 * Ib.Norml2()) << "swirl momentum";
    // row(b x x) = -[ad_b, I_core] W: zero at the discrete core's
    // isotropy grade (measured ~2e-5 of the O(1) scale |I b| Omega).
    EXPECT_LT(row_s.Norml2(), 1e-4 * Omega * Ib.Norml2())
        << "swirl row (isotropy grade)";
  }
}

TEST(GeneralisedRotation, FluidCoreMantleTisserand) {
  // On a fluid-core model the Tisserand gauge restricts its metric to the
  // SOLID attributes — the mantle frame — because the core's relative
  // angular momentum is pure relabelling gauge. The generalised door
  // here is the input-moments form (the radial model's own moments are
  // rightly degenerate): SetRotation(Omega, moments) +
  // SetTisserandGauge(true), composed with the gauged fluid core.
  // Certificates: (i) the mantle momentum int_mantle rho x cross u dV
  // vanishes at the gauged solution and is ACTIVE (the ungauged twin's
  // is orders larger); (ii) omega is stable when the fluid
  // representative changes (a different gauge penalty epsilon), at the
  // penalty-bias grade.
  const int order = 1, dim = 3;
  const double r_cmb = kRCmb;
  Mesh parent(TwoLayerMeshFile(dim).c_str(), 1, 1);
  Array<int> body_attr({1, 2}), buffer_attr({3});
  SubMesh body(SubMesh::CreateFromDomain(parent, body_attr));
  SubMesh buffer(SubMesh::CreateFromDomain(parent, buffer_attr));
  H1_FECollection fec(order, dim);
  FiniteElementSpace fes_u(&body, &fec, dim), fes_zeta(&parent, &fec);
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

  // The canned two-layer densities (MixedProblemTestCommon), radially.
  auto fluid_rho = [](double r) {
    return 1.2 - 0.1 * (r - kRIcb) / (kRCmb - kRIcb);
  };
  RadialHydrostaticBackground bg(
      dim,
      [r_cmb, fluid_rho](double r) {
        return r < r_cmb ? fluid_rho(r) : 1.0;
      },
      [](double) { return kKappa; },
      [r_cmb](double r) { return r < r_cmb ? 0.0 : kMu; }, kG, 1.0);
  ConstantCoefficient mu_gauge(kKappa);
  FunctionCoefficient sigma([](const Vector& x) {
    const double r2 = x * x;
    const double c2 = x[2] * x[2] / r2;
    return 0.02 * (3.0 * c2 - 1.0 + 2.0 * x[0] * x[2] / r2 + 0.5);
  });
  const double Omega = 0.1;
  Vector moments(3);
  moments[0] = 1.2;
  moments[1] = 1.3;
  moments[2] = 2.0;
  auto E = NewRadialVacuumExtension(fes_u, fes_buffer, 1.0, r_out);

  auto solve = [&](double eps, bool gauged) {
    auto p =
        std::make_unique<LinearQuasiStaticReferentialSelfGravitatingProblem>(
            &fes_u, &fes_zeta, bg.Rheology(), bg.Density(), kG, kDtNDegree);
    p->SetPrescribedVacuumExtension(fes_buffer, *E);
    p->SetFluid(fluid_marker, mu_gauge, GaugePenaltyOptions{eps, 3});
    p->SetSurfaceLoad(sigma, surface);
    p->SetRotation(Omega, moments);
    p->SetTisserandGauge(gauged);
    p->SetRelTol(1e-11);
    p->AssembleForce(0.0);
    EXPECT_TRUE(p->Solve());
    return p;
  };

  // Mantle-restricted momentum and inertia by quadrature (attribute 2).
  auto mantle_h = [&](const GridFunction& u, double* trace_out = nullptr) {
    VectorGridFunctionCoefficient uc(const_cast<GridFunction*>(&u));
    Vector h(3), x(3), uv(3);
    h = 0.0;
    double trace = 0.0;
    for (int e = 0; e < body.GetNE(); e++) {
      if (body.GetAttribute(e) != 2) {
        continue;
      }
      auto* T = body.GetElementTransformation(e);
      const auto& ir =
          IntRules.Get(body.GetElementGeometry(e), 2 * order + 4);
      for (int q = 0; q < ir.GetNPoints(); q++) {
        const auto& ip = ir.IntPoint(q);
        T->SetIntPoint(&ip);
        T->Transform(ip, x);
        uc.Eval(uv, *T, ip);
        const double w = ip.weight * T->Weight();  // mantle rho = 1
        h[0] += w * (x[1] * uv[2] - x[2] * uv[1]);
        h[1] += w * (x[2] * uv[0] - x[0] * uv[2]);
        h[2] += w * (x[0] * uv[1] - x[1] * uv[0]);
        trace += w * 2.0 * (x * x);
      }
    }
    if (trace_out) {
      *trace_out = trace;
    }
    return h;
  };

  auto p_g = solve(1e-2, true);
  auto p_u = solve(1e-2, false);
  double tr_m = 0.0;
  const Vector h_g = mantle_h(p_g->Displacement(), &tr_m);
  const Vector h_u = mantle_h(p_u->Displacement());
  // (i) The mantle frame is pinned (measured 7.1e-9 at o1: the mantle
  // density is constant, so the restricted metric is assembled
  // exactly), and the pinning is active (the ungauged twin measured
  // 333x larger).
  EXPECT_LT(h_g.Norml2(), 5e-8 * tr_m);
  EXPECT_GT(h_u.Norml2(), 1e2 * std::max(h_g.Norml2(), 1e-14 * tr_m));

  // (ii) omega across fluid representatives (eps 1e-2 vs 3e-3): stable
  // at the penalty-bias grade of the refined gauge penalty.
  auto p_g2 = solve(3e-3, true);
  Vector dw(p_g->AngularVelocity());
  dw -= p_g2->AngularVelocity();
  // Measured 3.1e-7: the rows are relabelling-invariant and the
  // refined penalty debiases, so omega barely feels the representative.
  EXPECT_LT(dw.Norml2(), 3e-6 * p_g->AngularVelocity().Norml2());
}

}  // namespace
