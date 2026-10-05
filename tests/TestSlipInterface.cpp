#include <array>
#include <cmath>
#include <functional>
#include <numbers>
#include <vector>

#include "gtest/gtest.h"
#include "mfem.hpp"
#include "AdGIA.hpp"

/*
  Finite-difference verification of the slip-interface second variation
  (doc/slip_interface.tex): the interface bilinear form

      B_Sigma(v, v) = - oint_Sigma pi nu . grad_Sigma(v_s + v_f)[s] dS,
      s = -F_e^{-1} [v],   nu = cof(F_e) N,

  is pinned (sign, factors, polarisation) against the EXACT energy of a
  constrained broken motion. No finite elements anywhere: the slip
  family is exact at finite epsilon (rotations of the circular
  referential interface), all fields are analytic, and every integral
  is high-order quadrature, so the comparison is limited only by the
  FD step and quadrature resolution.

  Construction (2-D, gravity-free, constant-referential-pressure
  equilibria -- exact by the Piola identity div cof F = 0):
    - referential interface Sigma = circle radius rc; fluid inside,
      solid outside with fields supported below r = 0.9 (no outer
      boundary terms);
    - solid energy W = S_e : E_rel + (1/2) E_rel : C : E_rel with
      E_rel = (F^T F - F_e^T F_e)/2 and S_e = -pi J_e C_e^{-1} (the
      pullback of -pi 1), so the equilibrium stress is -pi cof F_e,
      divergence-free, with interface traction -pi nu;
    - fluid energy V(x, J) = -pi (J - J_e) + (kappa/2)(J - J_e)^2;
    - slip family sigma_eps = rotation of Sigma by the angle
      eps a(theta); extended into the fluid by R_eps = rotation by
      eps a(theta) g(r) with g = (r/rc)^3 (analytic in x), and
      phi_f = phi_s o R_eps: the attachment holds EXACTLY at finite
      eps, and the family's own curvature reproduces the sigma_2
      normal term of the derivation.

  The test asserts d^2E/deps^2 (numerical) = Q_vol(v_s, v_f) + B_Sigma
  to FD accuracy, that B_Sigma is genuinely load-bearing (dropping or
  flipping it breaks the match by orders of magnitude), and the
  polarised identity via a mixed second difference of a two-parameter
  family. Case 1: phi_e = id. Case 2: a curved radial phi_e
  (nu != N, exercising the full Nanson and trace-derivative content).
*/

namespace {

using Vec2 = std::array<double, 2>;
using Mat2 = std::array<std::array<double, 2>, 2>;

constexpr double kPi = std::numbers::pi;
double rc = 0.6;  // referential interface radius (a variable: the
                  // discrete cross-check resets it to the mesh's)
constexpr double rOut = 0.9;    // support limit of the solid field
constexpr double piP = 0.3;     // referential pressure
constexpr double lam = 1.1, mu = 0.7, kap = 2.0;

Mat2 MatMul(const Mat2& A, const Mat2& B) {
  Mat2 C{};
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      C[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j];
    }
  }
  return C;
}
Mat2 MatT(const Mat2& A) { return {{{A[0][0], A[1][0]}, {A[0][1], A[1][1]}}}; }
double Det(const Mat2& A) { return A[0][0] * A[1][1] - A[0][1] * A[1][0]; }
double Tr(const Mat2& A) { return A[0][0] + A[1][1]; }
Mat2 Inv(const Mat2& A) {
  const double d = Det(A);
  return {{{A[1][1] / d, -A[0][1] / d}, {-A[1][0] / d, A[0][0] / d}}};
}
Mat2 Add(const Mat2& A, const Mat2& B, double b = 1.0) {
  Mat2 C{};
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      C[i][j] = A[i][j] + b * B[i][j];
    }
  }
  return C;
}
double Dot(const Mat2& A, const Mat2& B) {
  double s = 0.0;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      s += A[i][j] * B[i][j];
    }
  }
  return s;
}
constexpr Mat2 kI{{{1.0, 0.0}, {0.0, 1.0}}};

// --- The equilibrium mapping (case-dependent) -----------------------------
// phi_e = f(r) x with f = f0 - f2 r^2 (f2 = 0 gives the identity).
struct EquilibriumMap {
  double f0 = 1.0, f2 = 0.0;
  Vec2 Map(const Vec2& x) const {
    const double f = f0 - f2 * (x[0] * x[0] + x[1] * x[1]);
    return {f * x[0], f * x[1]};
  }
  Mat2 Grad(const Vec2& x) const {
    const double f = f0 - f2 * (x[0] * x[0] + x[1] * x[1]);
    Mat2 F = {{{f, 0.0}, {0.0, f}}};
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        F[i][j] += -2.0 * f2 * x[i] * x[j];
      }
    }
    return F;
  }
};

// --- The independent solid-side field V (global analytic) -----------------
// V = h(r) b(x), h = ((rOut^2 - r^2)/(rOut^2 - rc^2))^2 (polynomial in x).
struct Field {
  double c1, c2, c3, c4, c5;  // generic coefficients of b(x)
  Vec2 Eval(const Vec2& x) const {
    const double q = (rOut * rOut - x[0] * x[0] - x[1] * x[1]) /
                     (rOut * rOut - rc * rc);
    if (q <= 0.0) {
      return {0.0, 0.0};
    }
    const double h = q * q;
    return {h * (c1 + c2 * x[1] + c3 * x[0] * x[0]),
            h * (c4 * x[0] + c5 * x[0] * x[1])};
  }
  Mat2 Grad(const Vec2& x) const {
    const double denom = rOut * rOut - rc * rc;
    const double q = (rOut * rOut - x[0] * x[0] - x[1] * x[1]) / denom;
    if (q <= 0.0) {
      return Mat2{};
    }
    const double h = q * q;
    const Vec2 dh = {-4.0 * q * x[0] / denom, -4.0 * q * x[1] / denom};
    const Vec2 b = {c1 + c2 * x[1] + c3 * x[0] * x[0],
                    c4 * x[0] + c5 * x[0] * x[1]};
    const Mat2 db = {{{2.0 * c3 * x[0], c2}, {c4 + c5 * x[1], c5 * x[0]}}};
    Mat2 G{};
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        G[i][j] = dh[j] * b[i] + h * db[i][j];
      }
    }
    return G;
  }
};

// --- The slip flow --------------------------------------------------------
// Rotation by the angle chi(x) = A(x), A = (a1 cos + a2 sin 2theta) (r/rc)^3
// as a polynomial-in-x expression: A = [a1 x r^2 + a2 (2 x y) r] / rc^3.
// (r = |x|; the a2 term uses sin(2theta) r^3 = 2 x y r, smooth away from 0
// and O(r^3) at 0.)
struct SlipFlow {
  double a1, a2;
  double A(const Vec2& x) const {
    const double r2 = x[0] * x[0] + x[1] * x[1];
    const double r = std::sqrt(r2);
    return (a1 * x[0] * r2 + a2 * 2.0 * x[0] * x[1] * r) / (rc * rc * rc);
  }
  Vec2 GradA(const Vec2& x) const {
    const double r2 = x[0] * x[0] + x[1] * x[1];
    const double r = std::sqrt(r2);
    Vec2 g = {a1 * (3.0 * x[0] * x[0] + x[1] * x[1]),
              a1 * 2.0 * x[0] * x[1]};
    if (r > 1e-14) {
      g[0] += a2 * (2.0 * x[1] * r + 2.0 * x[0] * x[0] * x[1] / r);
      g[1] += a2 * (2.0 * x[0] * r + 2.0 * x[0] * x[1] * x[1] / r);
    }
    g[0] /= rc * rc * rc;
    g[1] /= rc * rc * rc;
    return g;
  }
  // R_eps(x) = Rot(eps A(x)) x and its gradient.
  Vec2 Map(double eps, const Vec2& x) const {
    const double c = std::cos(eps * A(x)), s = std::sin(eps * A(x));
    return {c * x[0] - s * x[1], s * x[0] + c * x[1]};
  }
  Mat2 Grad(double eps, const Vec2& x) const {
    const double chi = eps * A(x);
    const double c = std::cos(chi), s = std::sin(chi);
    const Mat2 R = {{{c, -s}, {s, c}}};
    const Vec2 g = GradA(x);
    const Vec2 zx = {-x[1], x[0]};  // z cross x
    Mat2 M = kI;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        M[i][j] += zx[i] * eps * g[j];
      }
    }
    return MatMul(R, M);
  }
};

// --- Energies -------------------------------------------------------------
struct Model {
  EquilibriumMap phi;

  double SolidDensity(const Vec2& x, const Mat2& F) const {
    const Mat2 Fe = phi.Grad(x);
    const Mat2 Ce = MatMul(MatT(Fe), Fe);
    const double Je = Det(Fe);
    const Mat2 E = Add(MatMul(MatT(F), F), Ce, -1.0);  // 2 E_rel
    // W = S_e : E_rel + (lam/2) tr(E)^2 + mu |E|^2, S_e = -pi Je Ce^{-1}.
    const Mat2 Se = Inv(Ce);
    const double SeE = -piP * Je * 0.5 * Dot(Se, E);
    const double trE = 0.5 * Tr(E);
    double EE = 0.0;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        EE += 0.25 * E[i][j] * E[i][j];
      }
    }
    return SeE + 0.5 * lam * trE * trE + mu * EE;
  }
  double FluidDensity(const Vec2& x, const Mat2& F) const {
    const double J = Det(F);
    const double Je = Det(phi.Grad(x));
    return -piP * (J - Je) + 0.5 * kap * (J - Je) * (J - Je);
  }

  // Exact energy of the constrained two-parameter family.
  double Energy(double e1, double e2, const Field& V1, const Field& V2,
                const SlipFlow& S1, const SlipFlow& S2, int Nr,
                int Nt) const {
    // Combined solid path phi_s = phi_e + e1 V1 + e2 V2; combined slip
    // rotation angle e1 A1 + e2 A2 (a valid exact family).
    auto rot = [&](const Vec2& x) -> Vec2 {
      const double chi = e1 * S1.A(x) + e2 * S2.A(x);
      const double c = std::cos(chi), s = std::sin(chi);
      return {c * x[0] - s * x[1], s * x[0] + c * x[1]};
    };
    auto drot = [&](const Vec2& x) -> Mat2 {
      const double chi = e1 * S1.A(x) + e2 * S2.A(x);
      const double c = std::cos(chi), s = std::sin(chi);
      const Mat2 R = {{{c, -s}, {s, c}}};
      const Vec2 g1 = S1.GradA(x), g2 = S2.GradA(x);
      const Vec2 zx = {-x[1], x[0]};
      Mat2 M = kI;
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
          M[i][j] += zx[i] * (e1 * g1[j] + e2 * g2[j]);
        }
      }
      return MatMul(R, M);
    };

    auto Fs = [&](const Vec2& x) -> Mat2 {
      return Add(Add(phi.Grad(x), V1.Grad(x), e1), V2.Grad(x), e2);
    };

    double E = 0.0;
    // Solid annulus rc..rOut.
    for (int i = 0; i < Nr; i++) {
      const double r = rc + (rOut - rc) * (i + 0.5) / Nr;
      const double wr = (rOut - rc) / Nr * r;
      for (int j = 0; j < Nt; j++) {
        const double th = 2.0 * kPi * (j + 0.5) / Nt;
        const Vec2 x = {r * std::cos(th), r * std::sin(th)};
        E += wr * (2.0 * kPi / Nt) * SolidDensity(x, Fs(x));
      }
    }
    // Fluid disc 0..rc: phi_f = phi_s o R.
    for (int i = 0; i < Nr; i++) {
      const double r = rc * (i + 0.5) / Nr;
      const double wr = rc / Nr * r;
      for (int j = 0; j < Nt; j++) {
        const double th = 2.0 * kPi * (j + 0.5) / Nt;
        const Vec2 x = {r * std::cos(th), r * std::sin(th)};
        const Vec2 y = rot(x);
        const Mat2 Ff = MatMul(Fs(y), drot(x));
        E += wr * (2.0 * kPi / Nt) * FluidDensity(x, Ff);
      }
    }
    return E;
  }

  // Predicted volume Hessian (the volume Hessians of doc/slip_interface.tex,
  // "The second variation", per region, with each region's own
  // first-order field).
  double SolidHess(const Vec2& x, const Mat2& Dv) const {
    const Mat2 Fe = phi.Grad(x);
    const Mat2 Ce = MatMul(MatT(Fe), Fe);
    const double Je = Det(Fe);
    // material: lam tr(M)^2 + 2 mu |M|^2, M = sym(Fe^T Dv)
    const Mat2 FtD = MatMul(MatT(Fe), Dv);
    const Mat2 M = {{{FtD[0][0], 0.5 * (FtD[0][1] + FtD[1][0])},
                     {0.5 * (FtD[0][1] + FtD[1][0]), FtD[1][1]}}};
    const double mat = lam * Tr(M) * Tr(M) + 2.0 * mu * Dot(M, M);
    // geometric: S_e : (Dv^T Dv) = -pi Je tr(Ce^{-1} Dv^T Dv)
    const Mat2 G = MatMul(MatT(Dv), Dv);
    const double geo = -piP * Je * Dot(Inv(Ce), G);
    return mat + geo;
  }
  double FluidHess(const Vec2& x, const Mat2& Dv) const {
    const Mat2 Fe = phi.Grad(x);
    const double Je = Det(Fe);
    const Mat2 H = MatMul(Inv(Fe), Dv);
    const double trH = Tr(H);
    // tr(H^2) = sum_ij H[i][j] H[j][i]
    const double trHH =
        H[0][0] * H[0][0] + 2.0 * H[0][1] * H[1][0] + H[1][1] * H[1][1];
    return kap * Je * Je * trH * trH - piP * Je * (trH * trH - trHH);
  }
};

// The first-order fields of the family: v_s = V; v_f = V + F_e s_ext with
// s_ext the slip-flow generator, s_ext = A(x) (z cross x).
Vec2 SExt(const SlipFlow& S, const Vec2& x) {
  const double A = S.A(x);
  return {-A * x[1], A * x[0]};
}
Mat2 DSExt(const SlipFlow& S, const Vec2& x) {
  const Vec2 g = S.GradA(x);
  const double A = S.A(x);
  Mat2 G = {{{0.0, -A}, {A, 0.0}}};
  const Vec2 zx = {-x[1], x[0]};
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      G[i][j] += zx[i] * g[j];
    }
  }
  return G;
}

// v_f and its gradient (chain rule at eps = 0: v_f = V + Dphi_e[s_ext]
// evaluated referentially; its gradient D(V + F_e s_ext)).
Vec2 VFluid(const Model& m, const Field& V, const SlipFlow& S,
            const Vec2& x) {
  const Vec2 v = V.Eval(x);
  const Vec2 se = SExt(S, x);
  const Mat2 Fe = m.phi.Grad(x);
  return {v[0] + Fe[0][0] * se[0] + Fe[0][1] * se[1],
          v[1] + Fe[1][0] * se[0] + Fe[1][1] * se[1]};
}
Mat2 DVFluid(const Model& m, const Field& V, const SlipFlow& S,
             const Vec2& x, double h = 1e-6) {
  // D(F_e s_ext) by an accurate central difference of the analytic
  // product (avoids hand-coding D^2 phi_e; h-error ~1e-12).
  Mat2 G = V.Grad(x);
  for (int j = 0; j < 2; j++) {
    Vec2 xp = x, xm = x;
    xp[j] += h;
    xm[j] -= h;
    const Vec2 sp = SExt(S, xp), sm = SExt(S, xm);
    const Mat2 Fp = m.phi.Grad(xp), Fm = m.phi.Grad(xm);
    for (int i = 0; i < 2; i++) {
      const double fp = Fp[i][0] * sp[0] + Fp[i][1] * sp[1];
      const double fm = Fm[i][0] * sm[0] + Fm[i][1] * sm[1];
      G[i][j] += (fp - fm) / (2.0 * h);
    }
  }
  return G;
}

// Predicted quadratic forms.
double QVol(const Model& m, const Field& Va, const SlipFlow& Sa,
            const Field& Vb, const SlipFlow& Sb, int Nr, int Nt) {
  // Polarised volume Hessian via the quadratic-form parallelogram.
  auto quad = [&](const Field& V, const SlipFlow& S, double sgnb,
                  const Field& V2, const SlipFlow& S2) {
    double E = 0.0;
    for (int i = 0; i < Nr; i++) {
      const double r = rc + (rOut - rc) * (i + 0.5) / Nr;
      const double wr = (rOut - rc) / Nr * r;
      for (int j = 0; j < Nt; j++) {
        const double th = 2.0 * kPi * (j + 0.5) / Nt;
        const Vec2 x = {r * std::cos(th), r * std::sin(th)};
        Mat2 D = Add(V.Grad(x), V2.Grad(x), sgnb);
        E += wr * (2.0 * kPi / Nt) * m.SolidHess(x, D);
      }
    }
    for (int i = 0; i < Nr; i++) {
      const double r = rc * (i + 0.5) / Nr;
      const double wr = rc / Nr * r;
      for (int j = 0; j < Nt; j++) {
        const double th = 2.0 * kPi * (j + 0.5) / Nt;
        const Vec2 x = {r * std::cos(th), r * std::sin(th)};
        Mat2 D = Add(DVFluid(m, V, S, x), DVFluid(m, V2, S2, x), sgnb);
        E += wr * (2.0 * kPi / Nt) * m.FluidHess(x, D);
      }
    }
    return E;
  };
  // B(a,b) = [Q(a+b) - Q(a-b)]/4.
  return 0.25 * (quad(Va, Sa, 1.0, Vb, Sb) - quad(Va, Sa, -1.0, Vb, Sb));
}

double BSigma(const Model& m, const Field& Va, const SlipFlow& Sa,
              const Field& Vb, const SlipFlow& Sb, int Nt) {
  // B_Sigma(a, b) = -1/2 oint pi { nu . d/dth(vs_a + vf_a) * s_b(th)
  //                              + nu . d/dth(vs_b + vf_b) * s_a(th) } dth,
  // with the tangential slip parameter along theta: s = a(th) d(pos)/dth,
  // so grad_Sigma(.)[s] = a(th) d(.)/dth, and a(th) = A(x(th)) since
  // g(rc) = 1.
  auto sum_trace = [&](const Field& V, const SlipFlow& S,
                       double th) -> Vec2 {
    const Vec2 x = {rc * std::cos(th), rc * std::sin(th)};
    const Vec2 vs = V.Eval(x);
    const Vec2 vf = VFluid(m, V, S, x);
    return {vs[0] + vf[0], vs[1] + vf[1]};
  };
  double B = 0.0;
  const int N = Nt;
  const double dth = 2.0 * kPi / N;
  for (int j = 0; j < N; j++) {
    const double th = j * dth;
    const Vec2 x = {rc * std::cos(th), rc * std::sin(th)};
    // nu = cof(F_e) N, N = outward radial unit.
    const Mat2 Fe = m.phi.Grad(x);
    const Mat2 cof = {{{Fe[1][1], -Fe[1][0]}, {-Fe[0][1], Fe[0][0]}}};
    const Vec2 Nref = {std::cos(th), std::sin(th)};
    const Vec2 nu = {cof[0][0] * Nref[0] + cof[0][1] * Nref[1],
                     cof[1][0] * Nref[0] + cof[1][1] * Nref[1]};
    // centered d/dth of the trace sums: h independent of the grid (the
    // traces are analytic in theta), so the derivative is ~1e-10 exact.
    auto ddth = [&](const Field& V, const SlipFlow& S) -> Vec2 {
      const double h = 1e-5;
      const Vec2 p = sum_trace(V, S, th + h), q = sum_trace(V, S, th - h);
      return {(p[0] - q[0]) / (2.0 * h), (p[1] - q[1]) / (2.0 * h)};
    };
    const Vec2 da = ddth(Va, Sa), db = ddth(Vb, Sb);
    const double aa = Sa.A(x), ab = Sb.A(x);
    // dS = rc dtheta on the referential circle.
    B += -0.5 * piP *
         ((nu[0] * da[0] + nu[1] * da[1]) * ab +
          (nu[0] * db[0] + nu[1] * db[1]) * aa) *
         rc * dth;
  }
  return B;
}

void RunCase(const EquilibriumMap& phi, const char* label) {
  Model m{phi};
  const Field V1{0.31, 0.62, -0.41, 0.53, 0.27};
  const Field V2{-0.22, 0.35, 0.18, -0.47, 0.39};
  const SlipFlow S1{0.8, 0.0};
  const SlipFlow S2{0.0, 0.6};

  const int Nr = 400, Nt = 600;
  const double eps = 2e-4;

  // Diagonal: d^2/deps^2 of the exact energy vs Q_vol + B_Sigma.
  const double Ep = m.Energy(eps, 0.0, V1, V2, S1, S2, Nr, Nt);
  const double E0 = m.Energy(0.0, 0.0, V1, V2, S1, S2, Nr, Nt);
  const double Em = m.Energy(-eps, 0.0, V1, V2, S1, S2, Nr, Nt);
  const double d2 = (Ep - 2.0 * E0 + Em) / (eps * eps);

  const double qvol = QVol(m, V1, S1, V1, S1, Nr, Nt);
  const double bsig = BSigma(m, V1, S1, V1, S1, Nt);
  const double scale = std::abs(qvol) + std::abs(bsig) + 1e-30;

  SCOPED_TRACE(label);
  // The match, and that B_Sigma is load-bearing.
  EXPECT_NEAR(d2, qvol + bsig, 2e-4 * scale)
      << "d2=" << d2 << " qvol=" << qvol << " B=" << bsig;
  EXPECT_GT(std::abs(bsig), 50.0 * std::abs(d2 - qvol - bsig))
      << "interface term not resolved: B=" << bsig
      << " residual=" << d2 - qvol - bsig;

  // Polarisation: mixed second difference vs the polarised forms.
  const double Epp = m.Energy(eps, eps, V1, V2, S1, S2, Nr, Nt);
  const double Epm = m.Energy(eps, -eps, V1, V2, S1, S2, Nr, Nt);
  const double Emp = m.Energy(-eps, eps, V1, V2, S1, S2, Nr, Nt);
  const double Emm = m.Energy(-eps, -eps, V1, V2, S1, S2, Nr, Nt);
  const double d2mix = (Epp - Epm - Emp + Emm) / (4.0 * eps * eps);
  // The mixed second derivative of E equals the POLARISED Hessian
  // directly (E is quadratic to leading order), so no factor of two.
  const double qmix = QVol(m, V1, S1, V2, S2, Nr, Nt);
  const double bmix = BSigma(m, V1, S1, V2, S2, Nt);
  const double scale2 = std::abs(qmix) + std::abs(bmix) + 1e-30;
  EXPECT_NEAR(d2mix, qmix + bmix, 3e-4 * scale2)
      << "d2mix=" << d2mix << " qmix=" << qmix << " Bmix=" << bmix;
}

}  // namespace

TEST(SlipInterface, SecondVariationIdentity) {
  RunCase(EquilibriumMap{1.0, 0.0}, "phi_e = id");
}

TEST(SlipInterface, SecondVariationMapped) {
  // f = 1.15 - (0.15/0.81) r^2: curved radial map, f(0.9) = 1 - ish,
  // J_e and nu genuinely non-trivial on Sigma.
  RunCase(EquilibriumMap{1.15, 0.15 / 0.81}, "phi_e radial");
}


// ---------------------------------------------------------------------------
// The discrete interface matrix (SlipInterfacePressureIntegrator +
// NewSlipInterfaceMatrix) against the FD-verified quadrature form: the
// same analytic fields interpolated onto the broken FE pair must
// reproduce B_Sigma(v, v) at interpolation accuracy, improving with
// order; a welded pair (equal traces) is annihilated in the quadratic
// form to round-off. Both the identity and the curved radial mapping.
TEST(SlipInterface, DiscreteMatrixMatchesQuadrature) {
  using namespace mfem;
  using namespace AdGIA;

  const double rc_saved = rc;
  rc = 3483.0 / 6371.0;  // the two-layer mesh's interface radius

  const Field V1{0.31, 0.62, -0.41, 0.53, 0.27};
  const SlipFlow S1{0.8, 0.0};

  struct Case {
    EquilibriumMap phi;
    const char* label;
  };
  const std::vector<Case> cases = {{EquilibriumMap{1.0, 0.0}, "id"},
                                   {EquilibriumMap{1.15, 0.15 / 0.81},
                                    "radial"}};

  for (const auto& cs : cases) {
    SCOPED_TRACE(cs.label);
    Model m{cs.phi};
    const double ref = BSigma(m, V1, S1, V1, S1, 4096);

    // The discrete mapping object matching EquilibriumMap.
    const double f0 = cs.phi.f0, f2 = cs.phi.f2;
    CallableDiffeomorphism map(
        2,
        [f0, f2](const Vector& x, Vector& y) {
          const double f = f0 - f2 * (x * x);
          y.SetSize(2);
          y(0) = f * x(0);
          y(1) = f * x(1);
        },
        [f0, f2](const Vector& x, DenseMatrix& F) {
          const double f = f0 - f2 * (x * x);
          F.SetSize(2);
          F = 0.0;
          F(0, 0) = f;
          F(1, 1) = f;
          for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
              F(i, j) += -2.0 * f2 * x(i) * x(j);
            }
          }
        });

    VectorFunctionCoefficient vs_coeff(2, [&](const Vector& x, Vector& v) {
      const Vec2 val = V1.Eval({x(0), x(1)});
      v.SetSize(2);
      v(0) = val[0];
      v(1) = val[1];
    });
    VectorFunctionCoefficient vf_coeff(2, [&](const Vector& x, Vector& v) {
      const Vec2 val = VFluid(m, V1, S1, {x(0), x(1)});
      v.SetSize(2);
      v(0) = val[0];
      v(1) = val[1];
    });

    std::vector<double> rel;
    double welded = 0.0, scale = 0.0;
    for (int order : {1, 2}) {
      Mesh parent("../data/elastogravity_two_layer_2d.msh", 1, 1);
      Array<int> fluid_attr({1}), solid_attr({2});
      SubMesh solid(SubMesh::CreateFromDomain(parent, solid_attr));
      SubMesh fluid(SubMesh::CreateFromDomain(parent, fluid_attr));
      H1_FECollection fec(order, 2);
      FiniteElementSpace fes_parent(&parent, &fec, 2);
      auto fes_s = SubMeshDofInjection::MakeShadowSpace(fes_parent, solid);
      auto fes_f = SubMeshDofInjection::MakeShadowSpace(fes_parent, fluid);
      SubMeshDofInjection inj_s(*fes_s, fes_parent),
          inj_f(*fes_f, fes_parent);
      auto J = NewSubMeshPairingMatrix(inj_s, inj_f);

      // Interface marker: the solid-submesh boundary attributes whose
      // elements sit at the CMB radius.
      Array<int> marker(solid.bdr_attributes.Max());
      marker = 0;
      for (int i = 0; i < solid.GetNBE(); i++) {
        auto* tr = solid.GetBdrElementTransformation(i);
        Vector c(2);
        tr->Transform(Geometries.GetCenter(solid.GetBdrElementGeometry(i)),
                      c);
        const double r = c.Norml2();
        if (r > 0.9 * rc && r < 1.1 * rc) {
          marker[solid.GetBdrAttribute(i) - 1] = 1;
        }
      }

      ConstantCoefficient piC(piP);
      auto B = NewSlipInterfaceMatrix(*fes_s, *J, marker, piC, map);

      GridFunction us(fes_s.get()), uf(fes_f.get());
      us.ProjectCoefficient(vs_coeff);
      uf.ProjectCoefficient(vf_coeff);

      auto quadform = [&](const GridFunction& a, const GridFunction& b) {
        Vector t(us.Size());
        double v = 0.0;
        B.ss->Mult(a, t);
        v += InnerProduct(a, t);
        B.sf->Mult(b, t);
        v += 2.0 * InnerProduct(a, t);
        Vector tf(uf.Size());
        B.ff->Mult(b, tf);
        v += InnerProduct(b, tf);
        return v;
      };
      const double val = quadform(us, uf);
      rel.push_back(std::abs(val - ref) / std::abs(ref));

      if (order == 2) {
        // Welded pair: same field on both sides -> quadratic form ~ 0.
        GridFunction uw(fes_f.get());
        uw.ProjectCoefficient(vs_coeff);
        welded = std::abs(quadform(us, uw));
        scale = std::abs(ref);
      }
    }
    EXPECT_LT(rel[1], 0.05);
    EXPECT_LT(rel[1], rel[0]);
    EXPECT_LT(welded, 1e-10 * scale);
  }

  rc = rc_saved;
}


// ---------------------------------------------------------------------------
// Gravity for the broken motion (doc/slip_interface.tex, "The gravity
// Hessian of the broken motion, explicitly"): with a rigid stress-free
// solid (rho_s = 0 and the hydrostatic fluid pressure vanishing at the
// interface, pi(rc) = 0), B_Sigma and the extension terms switch off,
// isolating the volume pieces specific to the broken motion: the mismatch coupling (through its zeta1-eliminated
// stationary value) and the rho w.grad grad zeta0 w term. Exact
// disc-preserving families (radial squeeze, rotation, composition);
// the density stays radial, so the exact gravitational self-energy is
// a single 1-D integral by the shell theorem,
//     E_g = 2 G int_0^rc ln R(r) M_enc(r) rho 2 pi r dr.
// Checks: equilibrium stationarity, the radial-family identity, the
// relabelling-null identity WITH gravity (azimuthal family: elastic
// pi(r)-terms, the 2 pi G rho^2 |w|^2 term and the eliminated-zeta1
// part cancel jointly), and the mixed family.
namespace gravity_fd {

constexpr double Gg = 0.5;
constexpr double rho = 1.0;
constexpr double kapF = 2.0;

double PiHydro(double r2, double rc2) {
  return kPi * Gg * rho * rho * (rc2 - r2);  // pi(rc) = 0
}

struct GravityFamily {
  double beta;    // radial-squeeze amplitude: v_rad = m(r) x
  double a1;      // theta-DEPENDENT rotation amplitude (chi ~ cos th)
  double arot;    // AXISYMMETRIC rotation amplitude (chi = arot (r/rc)^3
                  // -- volume preserving: the true relabelling)
  double mfun(double r) const {
    return beta * (rc * rc - r * r) / (rc * rc);
  }
  double chi_axi(double r) const {
    return arot * r * r * r / (rc * rc * rc);
  }
  // phi_eps(x) = (1 + eps m(r)) Rot(eps [A(x) + chi_axi(r)]) x, exact
  // disc-preserving.
  Vec2 Map(double eps, const Vec2& x) const {
    const SlipFlow S{a1, 0.0};
    const double r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
    const double chi = eps * (S.A(x) + chi_axi(r));
    const double c = std::cos(chi), sn = std::sin(chi);
    const Vec2 y = {c * x[0] - sn * x[1], sn * x[0] + c * x[1]};
    const double f = 1.0 + eps * mfun(r);
    return {f * y[0], f * y[1]};
  }
  Mat2 Grad(double eps, const Vec2& x) const {
    const SlipFlow S{a1, 0.0};
    const double r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
    const double chi = eps * (S.A(x) + chi_axi(r));
    const double c = std::cos(chi), sn = std::sin(chi);
    const Mat2 R = {{{c, -sn}, {sn, c}}};
    Vec2 gchi = S.GradA(x);
    if (r > 1e-14) {
      const double dchi = 3.0 * arot * r / (rc * rc * rc);
      gchi[0] += dchi * x[0];
      gchi[1] += dchi * x[1];
    }
    const Vec2 zx = {-x[1], x[0]};
    Mat2 M = kI;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        M[i][j] += zx[i] * eps * gchi[j];
      }
    }
    const Mat2 Dy = MatMul(R, M);
    const Vec2 y = {c * x[0] - sn * x[1], sn * x[0] + c * x[1]};
    const double f = 1.0 + eps * mfun(r);
    const Vec2 gm = {-2.0 * beta * x[0] / (rc * rc),
                     -2.0 * beta * x[1] / (rc * rc)};
    Mat2 F{};
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        F[i][j] = f * Dy[i][j] + eps * y[i] * gm[j];
      }
    }
    return F;
  }
  // First-order field v = m(r) x + [A(x) + chi_axi(r)] (z cross x).
  Vec2 V(const Vec2& x) const {
    const SlipFlow S{a1, 0.0};
    const double r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
    const double A = S.A(x) + chi_axi(r);
    return {mfun(r) * x[0] - A * x[1], mfun(r) * x[1] + A * x[0]};
  }
  Mat2 DV(const Vec2& x) const {
    const SlipFlow S{a1, 0.0};
    const double r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
    const Vec2 gm = {-2.0 * beta * x[0] / (rc * rc),
                     -2.0 * beta * x[1] / (rc * rc)};
    Mat2 F = DSExt(S, x);
    // axisymmetric rotation part: chi_axi (z cross x)
    const double A0 = chi_axi(r);
    Vec2 gA0 = {0.0, 0.0};
    if (r > 1e-14) {
      const double d = 3.0 * arot * r / (rc * rc * rc);
      gA0 = {d * x[0], d * x[1]};
    }
    const Vec2 zx = {-x[1], x[0]};
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        F[i][j] += zx[i] * gA0[j] + (i == 0 && j == 1 ? -A0 : 0.0) +
                   (i == 1 && j == 0 ? A0 : 0.0);
      }
    }
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        F[i][j] += (i == j ? mfun(r) : 0.0) + x[i] * gm[j];
      }
    }
    return F;
  }
};

// Exact elastic energy of the fluid (hydrostatic pi(r), J_e = 1).
double ElasticEnergy(const GravityFamily& fam, double eps, int Nr, int Nt) {
  double E = 0.0;
  const double rc2 = rc * rc;
  for (int i = 0; i < Nr; i++) {
    const double r = rc * (i + 0.5) / Nr;
    const double wr = rc / Nr * r;
    for (int j = 0; j < Nt; j++) {
      const double th = 2.0 * kPi * (j + 0.5) / Nt;
      const Vec2 x = {r * std::cos(th), r * std::sin(th)};
      const double J = Det(fam.Grad(eps, x));
      const double p = PiHydro(r * r, rc2);
      E += wr * (2.0 * kPi / Nt) *
           (-p * (J - 1.0) + 0.5 * kapF * (J - 1.0) * (J - 1.0));
    }
  }
  return E;
}

// Exact gravitational self-energy: the deformed density is radial with
// image radius R(r) = r (1 + eps m(r)), so by the 2-D shell theorem
// E_g = 2 G int ln R(r) M_enc(r) dm(r).
double GravityEnergy(const GravityFamily& fam, double eps, int N) {
  double E = 0.0;
  for (int i = 0; i < N; i++) {
    const double r = rc * (i + 0.5) / N;
    const double dr = rc / N;
    const double R = r * (1.0 + eps * fam.mfun(r));
    const double Menc = rho * kPi * r * r;
    E += 2.0 * Gg * std::log(R) * Menc * rho * 2.0 * kPi * r * dr;
  }
  return E;
}

double TotalEnergy(const GravityFamily& fam, double eps, int Nr, int Nt) {
  return ElasticEnergy(fam, eps, Nr, Nt) + GravityEnergy(fam, eps, 40000);
}

// Predicted Hessian: elastic + the two gravity pieces.
double Prediction(const GravityFamily& fam, int Nr, int Nt) {
  const double rc2 = rc * rc;
  double Qel = 0.0, wnorm2 = 0.0;
  for (int i = 0; i < Nr; i++) {
    const double r = rc * (i + 0.5) / Nr;
    const double wr = rc / Nr * r;
    for (int j = 0; j < Nt; j++) {
      const double th = 2.0 * kPi * (j + 0.5) / Nt;
      const Vec2 x = {r * std::cos(th), r * std::sin(th)};
      const Mat2 H = fam.DV(x);
      const double trH = Tr(H);
      const double trHH = H[0][0] * H[0][0] + 2.0 * H[0][1] * H[1][0] +
                          H[1][1] * H[1][1];
      const double p = PiHydro(r * r, rc2);
      Qel += wr * (2.0 * kPi / Nt) *
             (kapF * trH * trH - p * (trH * trH - trHH));
      const Vec2 v = fam.V(x);
      wnorm2 += wr * (2.0 * kPi / Nt) * (v[0] * v[0] + v[1] * v[1]);
    }
  }
  // rho w . grad grad zeta0 . w = 2 pi G rho^2 |w|^2 inside the disc.
  const double Qgg = 2.0 * kPi * Gg * rho * rho * wnorm2;
  // The eliminated-zeta1 stationary value: NEGATIVE definite (the
  // induced potential perturbation lowers the energy -- self-gravity
  // destabilises). For the radial source div(rho w) the interior
  // solution gives |value| = 8 pi^2 G rho^2 int m^2 r^3 dr.
  double stat = 0.0;
  const int N1 = 40000;
  for (int i = 0; i < N1; i++) {
    const double r = rc * (i + 0.5) / N1;
    const double dr = rc / N1;
    const double m = fam.mfun(r);
    stat += 8.0 * kPi * kPi * Gg * rho * rho * m * m * r * r * r * dr;
  }
  return Qel + Qgg - stat;
}

}  // namespace gravity_fd

TEST(SlipInterface, GravityHessianIdentity) {
  using namespace gravity_fd;
  rc = 0.6;
  const int Nr = 400, Nt = 600;
  const double eps = 2e-4;

  struct Case {
    GravityFamily fam;
    const char* label;
    bool expect_null;
  };
  const std::vector<Case> cases = {
      {{0.4, 0.0, 0.0}, "radial", false},
      {{0.0, 0.8, 0.0}, "azimuthal (non-volume-preserving)", false},
      {{0.0, 0.0, 0.8}, "axisymmetric rotation (relabelling)", true},
      {{0.4, 0.8, 0.5}, "mixed", false},
  };

  for (const auto& cs : cases) {
    SCOPED_TRACE(cs.label);
    const double Ep = TotalEnergy(cs.fam, eps, Nr, Nt);
    const double E0 = TotalEnergy(cs.fam, 0.0, Nr, Nt);
    const double Em = TotalEnergy(cs.fam, -eps, Nr, Nt);

    // Equilibrium: the first variation vanishes along every family.
    const double d1 = (Ep - Em) / (2.0 * eps);
    const double d2 = (Ep - 2.0 * E0 + Em) / (eps * eps);
    const double pred = Prediction(cs.fam, Nr, Nt);
    const double scale = std::abs(pred) + std::abs(d2) + 1e-12;

    EXPECT_LT(std::abs(d1), 1e-5 * scale * std::max(1.0, std::abs(E0)));
    if (cs.expect_null) {
      // The relabelling-null identity WITH gravity: the pi(r)-elastic
      // terms, the grad grad zeta0 term and the (vanishing) stationary
      // part cancel jointly; both the exact energy and the assembled
      // prediction must see it.
      const double term_scale =
          2.0 * kPi * Gg * rho * rho;  // the size of the players
      EXPECT_LT(std::abs(d2), 5e-4 * term_scale);
      EXPECT_LT(std::abs(pred), 5e-4 * term_scale);
    } else {
      EXPECT_NEAR(d2, pred, 3e-4 * scale)
          << "d2=" << d2 << " pred=" << pred;
    }
  }

  // The gravity-slip cross terms vanish (doc/slip_interface.tex, lemma
  // "Second order: gravity stays volume-only"): the mixed
  // second derivative across (radial, rotation) equals the purely
  // elastic cross term; gravity contributes nothing.
  {
    const GravityFamily fr{0.4, 0.0, 0.0}, fo{0.0, 0.8, 0.0};
    auto Etwo = [&](double e1, double e2) {
      // phi = (1 + e1 m) Rot(e2 A) x: an exact two-parameter family.
      GravityFamily f{0.0, 0.0};
      (void)f;
      double E = 0.0;
      const double rc2 = rc * rc;
      for (int i = 0; i < Nr; i++) {
        const double r = rc * (i + 0.5) / Nr;
        const double wr = rc / Nr * r;
        for (int j = 0; j < Nt; j++) {
          const double th = 2.0 * kPi * (j + 0.5) / Nt;
          const Vec2 x = {r * std::cos(th), r * std::sin(th)};
          const SlipFlow S{fo.a1, 0.0};
          const Vec2 y = S.Map(e2, x);
          const Mat2 Dy = S.Grad(e2, x);
          const double fscale = 1.0 + e1 * fr.mfun(r);
          const Vec2 gm = {-2.0 * fr.beta * x[0] / rc2,
                           -2.0 * fr.beta * x[1] / rc2};
          Mat2 F{};
          for (int a = 0; a < 2; a++) {
            for (int b = 0; b < 2; b++) {
              F[a][b] = fscale * Dy[a][b] + e1 * y[a] * gm[b];
            }
          }
          const double J = Det(F);
          const double p = PiHydro(r * r, rc2);
          E += wr * (2.0 * kPi / Nt) *
               (-p * (J - 1.0) + 0.5 * kapF * (J - 1.0) * (J - 1.0));
        }
      }
      // Gravity: density depends on the radial part only.
      GravityFamily frad = fr;
      E += GravityEnergy(frad, e1, 40000);
      return E;
    };
    const double Epp = Etwo(eps, eps), Epm = Etwo(eps, -eps);
    const double Emp = Etwo(-eps, eps), Emm = Etwo(-eps, -eps);
    const double d2mix = (Epp - Epm - Emp + Emm) / (4.0 * eps * eps);

    // Elastic-only cross prediction (gravity cross = 0 by the lemma).
    double Qcross = 0.0;
    const double rc2 = rc * rc;
    for (int i = 0; i < Nr; i++) {
      const double r = rc * (i + 0.5) / Nr;
      const double wr = rc / Nr * r;
      for (int j = 0; j < Nt; j++) {
        const double th = 2.0 * kPi * (j + 0.5) / Nt;
        const Vec2 x = {r * std::cos(th), r * std::sin(th)};
        const Mat2 Ha = fr.DV(x), Hb = fo.DV(x);
        const double tra = Tr(Ha), trb = Tr(Hb);
        const double trab = Ha[0][0] * Hb[0][0] + Ha[0][1] * Hb[1][0] +
                            Ha[1][0] * Hb[0][1] + Ha[1][1] * Hb[1][1];
        const double p = PiHydro(r * r, rc2);
        Qcross += wr * (2.0 * kPi / Nt) *
                  (kapF * tra * trb - p * (tra * trb - trab));
      }
    }
    const double scale = std::abs(Qcross) + std::abs(d2mix) + 1e-12;
    EXPECT_NEAR(d2mix, Qcross, 5e-4 * scale)
        << "d2mix=" << d2mix << " elastic cross=" << Qcross;
  }
}

// ---------------------------------------------------------------------------
// The broken-zeta organisation (doc/slip_interface.tex, "Gravity: the
// broken-zeta organisation"): region-wise composition makes both gravity
// sources exact, the entire mismatch machinery vanishes, and its place is
// taken by the jump condition [[zeta1]] = b.[[v]] and ONE additional
// interface form (at phi_e = id, b = grad zeta0, q = b.nu / 4 pi G):
//
//   G_Sigma = oint { (|b|^2 / 8 pi G) nu . grad_S(vs + vf)[s]
//                  + q ( grad_S(zs1 + zf1)[s]
//                      - b . grad_S(vs + vf)[s] ) } dS .
//
// Same rigid-solid hydrostatic-disc setup as GravityHessianIdentity
// (rho_s = 0, pi(rc) = 0 so B_Sigma is off; the solid is rigid, so
// v_s = 0 and zeta_s1 = Phi_1). The prediction is assembled ENTIRELY
// in the broken organisation: the per-region a'/a''-machinery with
// the region's own field, the exact Eulerian first-order potential
// Phi_1 in closed form (a radial part from the squeeze, a sin(theta)
// harmonic from the azimuthal family), the zeta1 Dirichlet terms
// (interior + exterior), and G_Sigma -- no extension, no mismatch
// coupling, no grad grad zeta0 mass term anywhere.
//
// The exact energy must include the theta-dependent part of the
// deformed density's self-interaction, which the radial shell
// shortcut omits. (In GravityHessianIdentity that omission cancels
// exactly against the omitted theta-part of the eliminated zeta1 --
// both sides of that test drop the same term; here Phi_1 is explicit,
// so the m = 1 multipole energy is added.) For image angle
// theta + eps (a(r) cos theta + chi(r)) it is exact:
//   E_theta = -4 pi^2 G rho^2 II (r</r>) J1(eps a(r)) J1(eps a(r'))
//             cos(eps (chi - chi')) r r' dr dr',
// separable by the cos/sin split of the chi-phase, hence two 1-D
// prefix-sum integrals.
//
// The axisymmetric relabelling is the sharp null check: in closed
// form Q_el, the a''-term and G_Sigma's vector part cancel as
// (-2/5) + (-3/5) + 1 (times pi^2 G rho^2 arot^2 rc^4), jointly
// pinning G_Sigma's sign, the 1/8piG factor and the q-coefficient.
namespace broken_zeta_fd {

using gravity_fd::Gg;
using gravity_fd::GravityFamily;
using gravity_fd::PiHydro;
using gravity_fd::rho;

double BesselJ1Small(double x) {
  const double x2 = x * x;
  return 0.5 * x * (1.0 - x2 / 8.0 + x2 * x2 / 192.0);  // |x| << 1
}

// The m = 1 multipole part of the exact gravitational energy (see the
// header comment). The (R</R>) kernel is taken at eps = 0: the
// correction is odd in eps at leading order and cancels in the
// symmetric second difference.
double ThetaEnergy(const GravityFamily& fam, double eps, int N) {
  if (fam.a1 == 0.0 || eps == 0.0) {
    return 0.0;
  }
  double Ic = 0.0, Is = 0.0, Pc = 0.0, Ps = 0.0;
  for (int i = 0; i < N; i++) {
    const double r = rc * (i + 0.5) / N;
    const double dr = rc / N;
    const double a = fam.a1 * r * r * r / (rc * rc * rc);
    const double chi = fam.chi_axi(r);
    const double j1 = BesselJ1Small(eps * a);
    const double u = j1 * std::cos(eps * chi);
    const double s = j1 * std::sin(eps * chi);
    // II (r</r>) u(r) u(r') r r' dr dr'
    //   = 2 int u(r) [ int_{r'<r} u(r') r'^2 dr' ] dr  (+ diagonal cell)
    Ic += 2.0 * u * (Pc + 0.5 * u * r * r * dr) * dr;
    Is += 2.0 * s * (Ps + 0.5 * s * r * r * dr) * dr;
    Pc += u * r * r * dr;
    Ps += s * r * r * dr;
  }
  return -4.0 * kPi * kPi * Gg * rho * rho * (Ic + Is);
}

// Closed-form Eulerian first-order potential Phi_1 (constants
// dropped; only gradients and theta-derivatives enter).
//   radial (squeeze, m(r) = beta (1 - r^2/rc^2)):
//     Phi_1r = pi G rho beta (r^4/rc^2 - 2 r^2), constant outside;
//   sin(theta) (azimuthal, rho_1 = rho a1 sin(theta) r^3/rc^3):
//     f_in = k (r^5 - 3 rc^4 r), f_out = -2 k rc^6 / r,
//     k = pi G rho a1 / (6 rc^3), C^1 at rc.
struct Phi1 {
  double beta, a1;
  double kcoef() const { return kPi * Gg * rho * a1 / (6.0 * rc * rc * rc); }
  double f(double r) const {  // r <= rc
    return kcoef() * (r * r * r * r * r - 3.0 * rc * rc * rc * rc * r);
  }
  double fp(double r) const {
    return kcoef() * (5.0 * r * r * r * r - 3.0 * rc * rc * rc * rc);
  }
  double Val(const Vec2& x) const {
    const double r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
    const double radial =
        kPi * Gg * rho * beta * (r * r * r * r / (rc * rc) - 2.0 * r * r);
    const double sth = (r > 1e-14) ? x[1] / r : 0.0;
    return radial + f(r) * sth;
  }
  Vec2 Grad(const Vec2& x) const {
    const double r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
    if (r < 1e-14) {
      return {0.0, fp(0.0)};  // grad(f sin th) -> f'(0) e_y at the centre
    }
    const Vec2 er = {x[0] / r, x[1] / r};
    const Vec2 et = {-x[1] / r, x[0] / r};
    const double sth = x[1] / r, cth = x[0] / r;
    const double dr_radial =
        4.0 * kPi * Gg * rho * beta * (r * r * r / (rc * rc) - r);
    const double gr = dr_radial + fp(r) * sth;
    const double gt = (f(r) / r) * cth;
    return {gr * er[0] + gt * et[0], gr * er[1] + gt * et[1]};
  }
};

struct BrokenPrediction {
  double total;
  double gsigma;
};

BrokenPrediction Predict(const GravityFamily& fam, int Nr, int Nt) {
  const Phi1 phi1{fam.beta, fam.a1};
  const double rc2 = rc * rc;
  const double twoPiGrho = 2.0 * kPi * Gg * rho;
  double Qel = 0.0, Ta = 0.0, Tc = 0.0, Tq = 0.0;
  for (int i = 0; i < Nr; i++) {
    const double r = rc * (i + 0.5) / Nr;
    const double wr = rc / Nr * r;
    for (int j = 0; j < Nt; j++) {
      const double th = 2.0 * kPi * (j + 0.5) / Nt;
      const Vec2 x = {r * std::cos(th), r * std::sin(th)};
      const double wq = wr * (2.0 * kPi / Nt);
      const Mat2 D = fam.DV(x);
      const Vec2 v = fam.V(x);
      // Elastic Hessian, as in the single-valued test.
      const double trH = Tr(D);
      const double trHH =
          D[0][0] * D[0][0] + 2.0 * D[0][1] * D[1][0] + D[1][1] * D[1][1];
      const double p = PiHydro(r * r, rc2);
      Qel += wq *
             (gravity_fd::kapF * trH * trH - p * (trH * trH - trHH));
      // Background field b = grad zeta0 = 2 pi G rho x, and the broken
      // fluid potential zeta_f1 = Phi_1 + b . v.
      const Vec2 g0 = {twoPiGrho * x[0], twoPiGrho * x[1]};
      const Vec2 Dtx = {D[0][0] * x[0] + D[1][0] * x[1],
                        D[0][1] * x[0] + D[1][1] * x[1]};
      const Vec2 gp = phi1.Grad(x);
      const Vec2 gz = {gp[0] + twoPiGrho * (v[0] + Dtx[0]),
                       gp[1] + twoPiGrho * (v[1] + Dtx[1])};
      // a'(v) g0 = tr(D) g0 - D g0 - D^T g0.
      const Vec2 Dg0 = {D[0][0] * g0[0] + D[0][1] * g0[1],
                        D[1][0] * g0[0] + D[1][1] * g0[1]};
      const Vec2 Dtg0 = {D[0][0] * g0[0] + D[1][0] * g0[1],
                         D[0][1] * g0[0] + D[1][1] * g0[1]};
      const Vec2 apg0 = {trH * g0[0] - Dg0[0] - Dtg0[0],
                         trH * g0[1] - Dg0[1] - Dtg0[1]};
      // <a''(v,v) g0, g0>
      //   = 2 [ det(D)|g0|^2 - tr(D) g0.(D + D^T)g0
      //        + g0.(D^2 + (D^T)^2 + D D^T)g0 ],
      // with g0.D^2 g0 = g0.(D^T)^2 g0 = (D^T g0).(D g0) and
      // g0.(D D^T)g0 = |D^T g0|^2.
      const double g0n2 = g0[0] * g0[0] + g0[1] * g0[1];
      const double detD = Det(D);
      const double gDDg = Dtg0[0] * Dg0[0] + Dtg0[1] * Dg0[1];
      const double gDDtg = Dtg0[0] * Dtg0[0] + Dtg0[1] * Dtg0[1];
      const double gSymg = g0[0] * (Dg0[0] + Dtg0[0]) +
                           g0[1] * (Dg0[1] + Dtg0[1]);
      const double aQuad =
          2.0 * (detD * g0n2 - trH * gSymg + 2.0 * gDDg + gDDtg);
      Ta += wq * aQuad / (8.0 * kPi * Gg);
      Tc += wq * (apg0[0] * gz[0] + apg0[1] * gz[1]) / (2.0 * kPi * Gg);
      Tq += wq * (gz[0] * gz[0] + gz[1] * gz[1]) / (4.0 * kPi * Gg);
    }
  }
  // Exterior Dirichlet energy of Phi_1: the radial part is constant
  // outside; the sin-theta part gamma/r gives gamma^2 / (4 G rc^2).
  const double gamma = -2.0 * phi1.kcoef() * rc2 * rc2 * rc2;
  Tq += gamma * gamma / (4.0 * Gg * rc2);
  // The outer region contributes nothing else: v = 0 there (rigid
  // solid, zero extension), so its a'/a''-terms vanish and
  // zeta_s1 = Phi_1.

  // G_Sigma by a theta loop on the interface circle (traces analytic
  // in theta; centered differences as in BSigma).
  double GS = 0.0;
  {
    const int N = 4096;
    const double dth = 2.0 * kPi / N;
    const double h = 1e-5;
    const double babs = twoPiGrho * rc;        // |b| on Sigma
    const double q = babs / (4.0 * kPi * Gg);  // = rho rc / 2
    auto traceVec = [&](double th) -> Vec2 {  // v_s + v_f (v_s = 0)
      return fam.V({rc * std::cos(th), rc * std::sin(th)});
    };
    auto traceScal = [&](double th) -> double {  // zeta_s1 + zeta_f1
      const Vec2 x = {rc * std::cos(th), rc * std::sin(th)};
      const Vec2 vf = fam.V(x);
      return 2.0 * phi1.Val(x) + twoPiGrho * (x[0] * vf[0] + x[1] * vf[1]);
    };
    for (int j = 0; j < N; j++) {
      const double th = j * dth;
      const Vec2 er = {std::cos(th), std::sin(th)};
      const Vec2 et = {-er[1], er[0]};
      const Vec2 vf = fam.V({rc * er[0], rc * er[1]});
      // s = alpha (z cross x), grad_S(.)[s] = alpha d(.)/dth.
      const double alpha = (vf[0] * et[0] + vf[1] * et[1]) / rc;
      const Vec2 vp = traceVec(th + h), vm = traceVec(th - h);
      const Vec2 dv = {(vp[0] - vm[0]) / (2.0 * h),
                       (vp[1] - vm[1]) / (2.0 * h)};
      const double ds =
          (traceScal(th + h) - traceScal(th - h)) / (2.0 * h);
      const double nudv = er[0] * dv[0] + er[1] * dv[1];
      GS += ((babs * babs / (8.0 * kPi * Gg) - q * babs) * nudv + q * ds) *
            alpha * rc * dth;
    }
  }
  return {Qel + Ta + Tc + Tq + GS, GS};
}

}  // namespace broken_zeta_fd

TEST(SlipInterface, BrokenZetaGravityIdentity) {
  using namespace broken_zeta_fd;
  rc = 0.6;
  const int Nr = 400, Nt = 600;
  const double eps = 2e-4;

  struct Case {
    GravityFamily fam;
    const char* label;
    bool expect_null;    // the joint relabelling null
    bool gs_loadbearing; // G_Sigma must be resolved against the residual
  };
  const std::vector<Case> cases = {
      {{0.4, 0.0, 0.0}, "radial (no slip: G_Sigma = 0)", false, false},
      {{0.0, 0.8, 0.0}, "azimuthal", false, true},
      {{0.0, 0.0, 0.8}, "axisymmetric relabelling (joint null)", true,
       true},
      {{0.4, 0.8, 0.5}, "mixed", false, true},
  };

  auto total = [&](const GravityFamily& fam, double e) {
    return gravity_fd::ElasticEnergy(fam, e, Nr, Nt) +
           gravity_fd::GravityEnergy(fam, e, 40000) +
           ThetaEnergy(fam, e, 40000);
  };

  for (const auto& cs : cases) {
    SCOPED_TRACE(cs.label);
    const double Ep = total(cs.fam, eps);
    const double E0 = total(cs.fam, 0.0);
    const double Em = total(cs.fam, -eps);
    const double d1 = (Ep - Em) / (2.0 * eps);
    const double d2 = (Ep - 2.0 * E0 + Em) / (eps * eps);
    const BrokenPrediction pred = Predict(cs.fam, Nr, Nt);
    const double scale = std::abs(pred.total) + std::abs(d2) + 1e-12;

    // Equilibrium survives the broken organisation (first-order
    // interface terms cancel between the Maxwell traction and the
    // jump flux).
    EXPECT_LT(std::abs(d1), 1e-5 * scale * std::max(1.0, std::abs(E0)));

    if (cs.expect_null) {
      const double term_scale = 2.0 * kPi * Gg * rho * rho;
      EXPECT_LT(std::abs(d2), 5e-4 * term_scale);
      EXPECT_LT(std::abs(pred.total), 5e-4 * term_scale);
      // The null is a genuine three-way cancellation: G_Sigma is a
      // large player in it ((-2/5) + (-3/5) + 1 = 0).
      EXPECT_GT(std::abs(pred.gsigma), 50.0 * std::abs(pred.total));
    } else {
      EXPECT_NEAR(d2, pred.total, 3e-4 * scale)
          << "d2=" << d2 << " pred=" << pred.total
          << " G_Sigma=" << pred.gsigma;
    }
    if (!cs.gs_loadbearing) {
      // No slip on Sigma: the interface form vanishes identically.
      EXPECT_LT(std::abs(pred.gsigma), 1e-12 * scale);
    } else if (!cs.expect_null) {
      EXPECT_GT(std::abs(pred.gsigma),
                30.0 * std::abs(d2 - pred.total))
          << "interface term not resolved: G=" << pred.gsigma
          << " residual=" << d2 - pred.total;
    }
  }
}

// ---------------------------------------------------------------------------
// Discrete realisation of the gravity pieces (pinned like B_Sigma
// above): the fluid-side extension operator
// (NewRadialFluidExtension) and the volume integrators that realise
// the mismatch/Hessian terms at phi_e = id are cross-checked against
// analytic quadrature on the two-layer mesh:
//   (a) E reproduces the solid trace on the interface to round-off
//       (the property w = v_f - tilde-v and the vanishing lemmas need);
//   (b) rho w . grad grad zeta0 . w' via VectorMassIntegrator with the
//       analytic uniform-disc MatrixConstantCoefficient 2 pi G rho^2 I;
//   (c) H_zeta-w = 2 int rho w . grad zeta1 via
//       DomainVectorGradScalarIntegrator(rho) (scalar trial);
//   (d) the tilde-v term -2 int rho grad zeta0 . D tilde-v [w]. NOTE
//       the semantics pinned here: DomainVectorGradVectorIntegrator
//       nodally interpolates the scalar w-bar . u and differentiates
//       the PRODUCT, so its form is int q w . grad(g0 . tilde-v)
//       = int q g0 . D tilde-v [w] + int q w^T (grad g0)^T tilde-v,
//       and the needed contraction is the compensated combination
//       G - M(rho grad grad zeta0) -- the SAME mass matrix as (b),
//       since grad g0 = grad grad zeta0. tilde-v = E u_s is folded
//       through the extension E and referenced against the analytic
//       extension rule t(r) u_s(rc x-hat), t = (r/rc)^2.
TEST(SlipInterface, DiscreteGravityPiecesMatchQuadrature) {
  using namespace mfem;
  using namespace AdGIA;
  using gravity_fd::Gg;
  using gravity_fd::rho;

  const double rc_saved = rc;
  rc = 3483.0 / 6371.0;  // the two-layer mesh's interface radius

  const gravity_fd::GravityFamily fam{0.4, 0.8, 0.5};  // generic fluid w
  const Field US{0.31, 0.62, -0.41, 0.53, 0.27};       // solid-side u_s

  // A generic smooth zeta1 test scalar and its gradient.
  auto zeta = [](const Vec2& x) {
    return 0.4 * x[0] - 0.7 * x[1] + 0.9 * x[0] * x[1] +
           0.5 * (x[0] * x[0] - x[1] * x[1]) + 0.3 * x[0] * x[0] * x[1];
  };
  auto grad_zeta = [](const Vec2& x) -> Vec2 {
    return {0.4 + 0.9 * x[1] + 1.0 * x[0] + 0.6 * x[0] * x[1],
            -0.7 + 0.9 * x[0] - 1.0 * x[1] + 0.3 * x[0] * x[0]};
  };
  // grad zeta0 = 2 pi G rho x inside the uniform disc.
  auto grad_zeta0 = [&](const Vec2& x) -> Vec2 {
    return {2.0 * kPi * Gg * rho * x[0], 2.0 * kPi * Gg * rho * x[1]};
  };

  // The analytic image of the extension rule.
  auto vtil = [&](const Vec2& x) -> Vec2 {
    const double r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
    if (r < 1e-8 * rc) {
      return {0.0, 0.0};
    }
    const double t = (r / rc) * (r / rc);
    const Vec2 u = US.Eval({rc * x[0] / r, rc * x[1] / r});
    return {t * u[0], t * u[1]};
  };
  auto Dvtil = [&](const Vec2& x) -> Mat2 {
    const double h = 1e-6;
    Mat2 D{};
    for (int j = 0; j < 2; j++) {
      Vec2 xp = x, xm = x;
      xp[j] += h;
      xm[j] -= h;
      const Vec2 vp = vtil(xp), vm = vtil(xm);
      for (int i = 0; i < 2; i++) {
        D[i][j] = (vp[i] - vm[i]) / (2.0 * h);
      }
    }
    return D;
  };

  // Quadrature references over the fluid disc.
  double refMass = 0.0, refZeta = 0.0, refVtil = 0.0;
  {
    const int Nr = 600, Nt = 800;
    for (int i = 0; i < Nr; i++) {
      const double r = rc * (i + 0.5) / Nr;
      const double wr = rc / Nr * r;
      for (int j = 0; j < Nt; j++) {
        const double th = 2.0 * kPi * (j + 0.5) / Nt;
        const Vec2 x = {r * std::cos(th), r * std::sin(th)};
        const double wq = wr * (2.0 * kPi / Nt);
        const Vec2 w = fam.V(x);
        refMass +=
            wq * 2.0 * kPi * Gg * rho * rho * (w[0] * w[0] + w[1] * w[1]);
        const Vec2 gz = grad_zeta(x);
        refZeta += wq * 2.0 * rho * (w[0] * gz[0] + w[1] * gz[1]);
        const Vec2 g0 = grad_zeta0(x);
        const Mat2 D = Dvtil(x);
        refVtil += wq * (-2.0) * rho *
                   (g0[0] * (D[0][0] * w[0] + D[0][1] * w[1]) +
                    g0[1] * (D[1][0] * w[0] + D[1][1] * w[1]));
      }
    }
  }
  // The references must be load-bearing.
  ASSERT_GT(std::abs(refMass), 1e-3);
  ASSERT_GT(std::abs(refZeta), 1e-3);
  ASSERT_GT(std::abs(refVtil), 1e-4);

  VectorFunctionCoefficient wC(2, [&](const Vector& x, Vector& v) {
    const Vec2 val = fam.V({x(0), x(1)});
    v.SetSize(2);
    v(0) = val[0];
    v(1) = val[1];
  });
  VectorFunctionCoefficient usC(2, [&](const Vector& x, Vector& v) {
    const Vec2 val = US.Eval({x(0), x(1)});
    v.SetSize(2);
    v(0) = val[0];
    v(1) = val[1];
  });
  FunctionCoefficient zC(
      [&](const Vector& x) { return zeta({x(0), x(1)}); });
  VectorFunctionCoefficient g0C(2, [&](const Vector& x, Vector& v) {
    const Vec2 val = grad_zeta0({x(0), x(1)});
    v.SetSize(2);
    v(0) = val[0];
    v(1) = val[1];
  });
  ConstantCoefficient rhoC(rho);
  DenseMatrix hess0(2);
  hess0 = 0.0;
  hess0(0, 0) = hess0(1, 1) = 2.0 * kPi * Gg * rho * rho;
  MatrixConstantCoefficient hess0C(hess0);

  std::vector<double> relMass, relZeta, relVtil;
  for (int order : {1, 2}) {
    SCOPED_TRACE(order);
    Mesh parent("../data/elastogravity_two_layer_2d.msh", 1, 1);
    Array<int> fluid_attr({1}), solid_attr({2});
    SubMesh solid(SubMesh::CreateFromDomain(parent, solid_attr));
    SubMesh fluid(SubMesh::CreateFromDomain(parent, fluid_attr));
    H1_FECollection fec(order, 2);
    FiniteElementSpace fes_parent(&parent, &fec, 2);
    auto fes_s = SubMeshDofInjection::MakeShadowSpace(fes_parent, solid);
    auto fes_f = SubMeshDofInjection::MakeShadowSpace(fes_parent, fluid);
    FiniteElementSpace fes_z(&fluid, &fec);

    GridFunction us(fes_s.get()), w(fes_f.get()), z(&fes_z);
    us.ProjectCoefficient(usC);
    w.ProjectCoefficient(wC);
    z.ProjectCoefficient(zC);

    // (a) The extension: trace rows exact, interior rows tapered.
    auto E = NewRadialFluidExtension(*fes_s, *fes_f, rc);
    GridFunction vt(fes_f.get());
    E->Mult(us, vt);
    {
      SubMeshDofInjection inj_s(*fes_s, fes_parent),
          inj_f(*fes_f, fes_parent);
      auto J = NewSubMeshPairingMatrix(inj_f, inj_s);  // fluid x solid
      Vector traced(fes_f->GetVSize());
      J->Mult(us, traced);
      double err = 0.0, scale = us.Normlinf() + 1e-30;
      for (int r = 0; r < J->Height(); r++) {
        if (J->RowSize(r) > 0) {
          err = std::max(err, std::abs(vt(r) - traced(r)));
        }
      }
      EXPECT_LT(err, 1e-12 * scale);
    }

    // (b) The grad grad zeta0 term.
    BilinearForm M(fes_f.get());
    M.AddDomainIntegrator(new VectorMassIntegrator(hess0C));
    M.Assemble();
    M.Finalize();
    {
      Vector t(w.Size());
      M.Mult(w, t);
      relMass.push_back(std::abs(InnerProduct(w, t) - refMass) /
                        std::abs(refMass));
    }

    // (c) The mismatch coupling H_zeta-w (scalar trial).
    {
      MixedBilinearForm A(&fes_z, fes_f.get());
      A.AddDomainIntegrator(new DomainVectorGradScalarIntegrator(rhoC));
      A.Assemble();
      A.Finalize();
      Vector t(w.Size());
      A.Mult(z, t);
      relZeta.push_back(std::abs(2.0 * InnerProduct(w, t) - refZeta) /
                        std::abs(refZeta));
    }

    // (d) The tilde-v term, folded through E: the compensated
    // combination G - M (product-rule semantics, see the header note).
    {
      BilinearForm G(fes_f.get());
      G.AddDomainIntegrator(
          new DomainVectorGradVectorIntegrator(g0C, rhoC));
      G.Assemble();
      G.Finalize();
      Vector t(w.Size()), tm(w.Size());
      G.Mult(vt, t);
      M.Mult(vt, tm);
      const double val = -2.0 * (InnerProduct(w, t) - InnerProduct(w, tm));
      relVtil.push_back(std::abs(val - refVtil) / std::abs(refVtil));
    }
  }

  // Measured relative errors: order 1 ~4e-2, order 2 2--5e-4 on every
  // piece (a two-orders drop); the bounds leave 10x headroom.
  EXPECT_LT(relMass[1], 5e-3);
  EXPECT_LT(relMass[1], relMass[0]);
  EXPECT_LT(relZeta[1], 5e-3);
  EXPECT_LT(relZeta[1], relZeta[0]);
  EXPECT_LT(relVtil[1], 5e-3);
  EXPECT_LT(relVtil[1], relVtil[0]);

  rc = rc_saved;
}

// ---------------------------------------------------------------------------
// The discrete broken-zeta gravity interface blocks
// (SlipInterfaceGravityIntegrator + SlipInterfaceGravityScalarIntegrator
// + NewSlipGravityInterfaceMatrix) against an analytic quadrature
// reference of G_Sigma: generic UNCONSTRAINED vector and scalar trace
// fields (the kernels' tangential projector makes the form well
// defined off the constraint, and the reference projects identically),
// a generic vector field in the grad-zeta0 slot, both the identity and
// the curved radial mapping; interpolation accuracy improving with
// order, and annihilation of a welded vector pair (zero slip kills the
// vector AND the scalar parts of the quadratic form) to round-off.
TEST(SlipInterface, DiscreteGravityInterfaceMatchesQuadrature) {
  using namespace mfem;
  using namespace AdGIA;
  using gravity_fd::Gg;

  const double rc_saved = rc;
  rc = 3483.0 / 6371.0;  // the two-layer mesh's interface radius

  const Field VS{0.31, 0.62, -0.41, 0.53, 0.27};
  const Field VF{-0.22, 0.35, 0.18, -0.47, 0.39};
  auto zs_fun = [](const Vec2& x) {
    return 0.3 + 0.7 * x[0] - 0.4 * x[1] + 0.5 * x[0] * x[1] +
           0.2 * x[0] * x[0];
  };
  auto zf_fun = [](const Vec2& x) {
    return -0.2 + 0.4 * x[0] + 0.6 * x[1] - 0.3 * x[0] * x[1] +
           0.1 * x[1] * x[1];
  };
  auto g0_fun = [](const Vec2& x) -> Vec2 {
    return {0.5 + 0.3 * x[1] + 0.4 * x[0] * x[0],
            -0.7 + 0.2 * x[0] + 0.1 * x[0] * x[1]};
  };

  struct Case {
    EquilibriumMap phi;
    const char* label;
  };
  const std::vector<Case> cases = {
      {EquilibriumMap{1.0, 0.0}, "id"},
      {EquilibriumMap{1.15, 0.15 / 0.81}, "radial"}};

  for (const auto& cs : cases) {
    SCOPED_TRACE(cs.label);
    Model m{cs.phi};

    // Quadrature reference on the referential circle.
    double ref = 0.0;
    {
      const int N = 4096;
      const double dth = 2.0 * kPi / N;
      const double h = 1e-5;
      auto vsum = [&](double th) -> Vec2 {
        const Vec2 x = {rc * std::cos(th), rc * std::sin(th)};
        const Vec2 a = VS.Eval(x), b = VF.Eval(x);
        return {a[0] + b[0], a[1] + b[1]};
      };
      auto zsum = [&](double th) -> double {
        const Vec2 x = {rc * std::cos(th), rc * std::sin(th)};
        return zs_fun(x) + zf_fun(x);
      };
      for (int j = 0; j < N; j++) {
        const double th = j * dth;
        const Vec2 er = {std::cos(th), std::sin(th)};
        const Vec2 et = {-er[1], er[0]};
        const Vec2 x = {rc * er[0], rc * er[1]};
        const Mat2 F = m.phi.Grad(x);
        const Mat2 Fi = Inv(F);
        const Mat2 cof = {{{F[1][1], -F[1][0]}, {-F[0][1], F[0][0]}}};
        const Vec2 nu = {cof[0][0] * er[0] + cof[0][1] * er[1],
                         cof[1][0] * er[0] + cof[1][1] * er[1]};
        const Vec2 g0 = g0_fun(x);
        // b = F^{-T} g0.
        const Vec2 b = {Fi[0][0] * g0[0] + Fi[1][0] * g0[1],
                        Fi[0][1] * g0[0] + Fi[1][1] * g0[1]};
        const double b2 = b[0] * b[0] + b[1] * b[1];
        const double q = (b[0] * nu[0] + b[1] * nu[1]) / (4.0 * kPi * Gg);
        // s = -P_T F^{-1} [[v]] = sigma t-hat.
        const Vec2 avs = VS.Eval(x), avf = VF.Eval(x);
        const Vec2 jump = {avs[0] - avf[0], avs[1] - avf[1]};
        const Vec2 Fij = {Fi[0][0] * jump[0] + Fi[0][1] * jump[1],
                          Fi[1][0] * jump[0] + Fi[1][1] * jump[1]};
        const double sigma = -(et[0] * Fij[0] + et[1] * Fij[1]);
        // grad_S(.)[s] = (sigma / rc) d(.)/dth; dS = rc dth.
        const Vec2 vp = vsum(th + h), vm = vsum(th - h);
        const Vec2 dv = {(vp[0] - vm[0]) / (2.0 * h),
                         (vp[1] - vm[1]) / (2.0 * h)};
        const double ds = (zsum(th + h) - zsum(th - h)) / (2.0 * h);
        const Vec2 A = {b2 * nu[0] / (8.0 * kPi * Gg) -
                            q * b[0],
                        b2 * nu[1] / (8.0 * kPi * Gg) - q * b[1]};
        ref += ((A[0] * dv[0] + A[1] * dv[1]) + q * ds) * sigma * dth;
      }
    }
    ASSERT_GT(std::abs(ref), 1e-4);  // load-bearing reference

    // The discrete mapping object matching EquilibriumMap.
    const double f0 = cs.phi.f0, f2 = cs.phi.f2;
    CallableDiffeomorphism map(
        2,
        [f0, f2](const Vector& x, Vector& y) {
          const double f = f0 - f2 * (x * x);
          y.SetSize(2);
          y(0) = f * x(0);
          y(1) = f * x(1);
        },
        [f0, f2](const Vector& x, DenseMatrix& F) {
          const double f = f0 - f2 * (x * x);
          F.SetSize(2);
          F = 0.0;
          F(0, 0) = f;
          F(1, 1) = f;
          for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
              F(i, j) += -2.0 * f2 * x(i) * x(j);
            }
          }
        });

    VectorFunctionCoefficient vsC(2, [&](const Vector& x, Vector& v) {
      const Vec2 val = VS.Eval({x(0), x(1)});
      v.SetSize(2);
      v(0) = val[0];
      v(1) = val[1];
    });
    VectorFunctionCoefficient vfC(2, [&](const Vector& x, Vector& v) {
      const Vec2 val = VF.Eval({x(0), x(1)});
      v.SetSize(2);
      v(0) = val[0];
      v(1) = val[1];
    });
    FunctionCoefficient zsC(
        [&](const Vector& x) { return zs_fun({x(0), x(1)}); });
    FunctionCoefficient zfC(
        [&](const Vector& x) { return zf_fun({x(0), x(1)}); });
    VectorFunctionCoefficient g0C(2, [&](const Vector& x, Vector& v) {
      const Vec2 val = g0_fun({x(0), x(1)});
      v.SetSize(2);
      v(0) = val[0];
      v(1) = val[1];
    });

    std::vector<double> rel;
    double welded = 0.0;
    for (int order : {1, 2}) {
      Mesh parent("../data/elastogravity_two_layer_2d.msh", 1, 1);
      Array<int> fluid_attr({1}), solid_attr({2});
      SubMesh solid(SubMesh::CreateFromDomain(parent, solid_attr));
      SubMesh fluid(SubMesh::CreateFromDomain(parent, fluid_attr));
      H1_FECollection fec(order, 2);
      FiniteElementSpace fes_parent(&parent, &fec, 2);
      FiniteElementSpace fes_parent_z(&parent, &fec);
      auto fes_s = SubMeshDofInjection::MakeShadowSpace(fes_parent, solid);
      auto fes_f = SubMeshDofInjection::MakeShadowSpace(fes_parent, fluid);
      auto fes_zs =
          SubMeshDofInjection::MakeShadowSpace(fes_parent_z, solid);
      auto fes_zf =
          SubMeshDofInjection::MakeShadowSpace(fes_parent_z, fluid);
      SubMeshDofInjection inj_s(*fes_s, fes_parent),
          inj_f(*fes_f, fes_parent);
      SubMeshDofInjection inj_zs(*fes_zs, fes_parent_z),
          inj_zf(*fes_zf, fes_parent_z);
      auto J = NewSubMeshPairingMatrix(inj_s, inj_f);
      auto Jz = NewSubMeshPairingMatrix(inj_zs, inj_zf);

      Array<int> marker(solid.bdr_attributes.Max());
      marker = 0;
      for (int i = 0; i < solid.GetNBE(); i++) {
        auto* tr = solid.GetBdrElementTransformation(i);
        Vector c(2);
        tr->Transform(Geometries.GetCenter(solid.GetBdrElementGeometry(i)),
                      c);
        const double r = c.Norml2();
        if (r > 0.9 * rc && r < 1.1 * rc) {
          marker[solid.GetBdrAttribute(i) - 1] = 1;
        }
      }

      auto B = NewSlipGravityInterfaceMatrix(*fes_s, *fes_zs, *J, *Jz,
                                             marker, g0C, Gg, map);

      GridFunction us(fes_s.get()), uf(fes_f.get());
      GridFunction zs(fes_zs.get()), zf(fes_zf.get());
      us.ProjectCoefficient(vsC);
      uf.ProjectCoefficient(vfC);
      zs.ProjectCoefficient(zsC);
      zf.ProjectCoefficient(zfC);

      auto quadform = [&](const GridFunction& a, const GridFunction& b,
                          const GridFunction& za, const GridFunction& zb) {
        double v = 0.0;
        Vector t(a.Size());
        B.ss->Mult(a, t);
        v += InnerProduct(a, t);
        B.sf->Mult(b, t);
        v += 2.0 * InnerProduct(a, t);
        Vector tf(b.Size());
        B.ff->Mult(b, tf);
        v += InnerProduct(b, tf);
        // Scalar part: 2 v^T [vz] zeta.
        B.vz_ss->Mult(za, t);
        v += 2.0 * InnerProduct(a, t);
        B.vz_sf->Mult(zb, t);
        v += 2.0 * InnerProduct(a, t);
        B.vz_fs->Mult(za, tf);
        v += 2.0 * InnerProduct(b, tf);
        B.vz_ff->Mult(zb, tf);
        v += 2.0 * InnerProduct(b, tf);
        return v;
      };
      const double val = quadform(us, uf, zs, zf);
      rel.push_back(std::abs(val - ref) / std::abs(ref));

      if (order == 2) {
        // Welded vector pair: zero slip annihilates the whole form,
        // scalar parts included, whatever the zeta traces.
        GridFunction uw(fes_f.get());
        uw.ProjectCoefficient(vsC);
        welded = std::abs(quadform(us, uw, zs, zf)) / std::abs(ref);
      }
    }
    EXPECT_LT(rel[1], 0.05);
    EXPECT_LT(rel[1], rel[0]);
    EXPECT_LT(welded, 1e-10);
  }

  rc = rc_saved;
}
