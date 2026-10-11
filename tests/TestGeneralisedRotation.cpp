#include <numbers>

#include "MixedProblemTestCommon.hpp"
#include "TestCommon.hpp"

/*
  WP7 phase-B gates of the generalised rotational theory
  (doc/planning/generalised_rotation_note.md) — the demonstration
  deliverable of the plan's decision 3: the traditional theory's
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
  3. The Tisserand gauge (the mass-weighted gauge, F4) pins the
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

 private:
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
  // (a) with the mass-weighted gauge (F4) the solution carries no
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

}  // namespace
