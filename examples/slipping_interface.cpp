// ============================================================================
// slipping_interface.cpp
//
// The slipping fluid-solid interface of a self-gravitating body,
// LinearQuasiStaticReferentialSelfGravitatingSlipProblem, with every
// option of the class exposed and its diagnostics printed, on a spherical
// or an ASPHERICAL reference mesh.
//
// The physical model is a spherically symmetric two-layer body of unit
// radius: a uniform compressible fluid core below r_c and a solid mantle
// above it (the constants of self_gravitating_solvers), in hydrostatic
// equilibrium (RadialHydrostaticBackground), under a static surface mass
// load of one harmonic degree l (zonal Y_l0 in 3-D, cos(l theta) in 2-D,
// unit coefficient of the orthonormal harmonic). The core displacement u_f
// and the mantle displacement u_s are separate fields that share only
// their normal component on the core boundary Sigma, nu . [[u]] = 0: the
// fluid slips along Sigma (doc/slip_interface.tex, "Kinematics of the
// slip"; doc/quasi_static_models.tex, "The slipping-interface problem").
//
// The reference body. On a spherical mesh the reference body is the
// physical body and the equilibrium mapping phi_e is the identity. On an
// aspherical mesh (meshes/aspherical_body.py --fluid-core) the mesh is a
// REFERENCE body of a different shape, the radial stretch
//   D(x) = (r + h) x^,  h = eps r s(theta, phi),  s = P2(cos theta)
//          + beta sin^2(theta) cos(2 phi),
// tapered to the identity across the buffer, and phi_e = D^{-1} maps it
// back onto the spherical physical body (exactly: the radial inversion
// is done per point, and F = (grad D)^{-1} analytically). The physics is
// the same; only its description changes: the background is the
// RelabelledBackground of the radial one (density J rho o phi_e, the
// pulled-back tensors), and the load is the physical harmonic pulled back
// with the Nanson area factor. The shape parameters are read from the
// mesh's manifest. An aspherical mesh therefore tests whether the mapped
// assembly is covariant: every observable must be that of the spherical
// model, and any response at other degrees is discretisation error.
//
// Organisations and enforcement (doc/slip_interface.tex, "Constraint
// enforcement" and "Implementation"):
//   -zeta single   one potential zeta^1 on the ball, the fluid's gravity
//                  through a prescribed extension of u_s into the core;
//                  assembled at phi_e = id only, so refused on an
//                  aspherical mesh;
//   -zeta broken   a potential on each side of Sigma with the scalar jump
//                  [[zeta^1]] = b . [[u]] as a second constraint; mapped
//                  throughout, the route for aspherical meshes;
//   -enforce al    penalty theta plus augmented-Lagrangian sweeps (both
//                  organisations), each sweep one projected-MINRES solve
//                  and one Tikhonov refinement of the fluid gauge;
//   -enforce kkt   a multiplier on Sigma and one saddle-point MINRES per
//                  gauge refinement (single-valued only). -kkt-order sets
//                  the multiplier order: one below the displacement order
//                  can make the gauge refinements diverge on these 2-D
//                  meshes, so watch the printed normal-jump history
//                  (doc/slip_interface.tex, "The multiplier space").
// The fluid displacement is determined only up to linearised
// relabellings; the gauge penalty eps Q (-geps) fixes it, with the
// Tikhonov refinements removing its bias.
//
// What is printed: the per-sweep constraint histories (normal jump, and
// the zeta jump of the broken organisation), iterations and wall time,
// the multiplier norm (KKT), the residuals of the slip null pairs under
// the assembled operator (common translations, independent rotations of
// mantle and core), the largest tangential slip and normal jump on Sigma,
// and a harmonic analysis on the PHYSICAL surface: the coefficients of the
// radial displacement u.x^ and of the Eulerian potential perturbation
// phi1 = zeta1 - b.u (b = F^{-T} grad zeta0) by degree up to -lmax,
// integrated over the reference surface at the physical points phi_e(x)
// with the Nanson area weight (the usual analysis on a sphere), and from
// them the load Love numbers
//   h' = -g u_l / phi_sigma,  l' = -g v_l / phi_sigma,
//   k' = phi_l / phi_sigma - 1,
// with g the surface gravity and phi_sigma the load's own potential on
// the surface in closed form (-4 pi G / (2l + 1) in 3-D, -2 pi G / l in
// 2-D, for the unit harmonic), and "spurious", the largest coefficient
// of any other harmonic relative to the loaded one (in 2-D without the
// potential's degree 0, a constant the 2-D problem leaves free), and the
// same without degree 1, where a residual translation of the body also
// shows. Spurious measures the departure from spherical symmetry:
// discretisation error on a spherical mesh, and on an aspherical mesh
// also any lack of covariance of the mapped assembly. With -compare the
// same problem is solved by the welded gauged referential problem (same
// mesh, map and background) and by
// Dahlen's mixed problem with an eliminated fluid region on the spherical
// counterpart mesh, and the three are tabulated side by side; on a
// spherical mesh the mantle displacements are also compared as fields.
//
// Outputs: the CSV table of the constraint history (slipping_interface.csv;
// plot with python3 plot_csv.py slipping_interface.csv), and GLVis windows
// (-vis, the default) drawn on the PHYSICAL body (copies of the meshes
// moved by phi_e): the mantle displacement, the core displacement (gauge,
// not an observable), the core's pressure perturbation -kappa div u (the
// gauge-invariant fluid observable, Lagrangian), the tangential slip on
// Sigma (drawn on the mantle, zero off Sigma; the computed slip depends
// on the fluid's gauge, which the penalty fixes, so it is not an
// observable here and differs between organisations — which slips a
// relabelling can remove is discussed in doc/gauged_fluid.md, "Tangential
// slip and the welded space"), and zeta^1
// (both sides for the broken organisation).
//
// One source serves the serial and the parallel build.
//
// Options (defaults in brackets):
//   -m          two-layer mesh with a buffer (domains 1 core, 2
//               mantle, 3 buffer; boundaries 1 core boundary, 2
//               surface, 3 outer), spherical or aspherical
//               (meshes/aspherical_body.py --fluid-core); an
//               aspherical mesh is the reference body of the spherical
//               physical model
//               [../data/aspherical_fluid_core_buffer_2d.mesh].
//   -o          finite element order [2].
//   -l          degree of the surface load (2 or more): Y_l0 in 3-D,
//               cos(l theta) in 2-D [2].
//   -deg        truncation degree of the Dirichlet-to-Neumann map
//               [12].
//   -rt         relative tolerance of the linear solves [1e-10].
//   -zeta       potential organisation: 'single' (one zeta on the
//               ball, fluid extension; identity map only, so refused
//               on an aspherical mesh) or 'broken' (zeta on each side
//               of Sigma, scalar-jump constraint; mapped) [broken].
//   -enforce    constraint enforcement: 'al' (penalty + augmented
//               Lagrangian) or 'kkt' (multiplier, saddle-point MINRES;
//               -zeta single only) [al].
//   -theta      constraint penalty theta (the normal jump and, with
//               -zeta broken, the zeta jump) [1e2].
//   -al         AL sweeps per solve (-enforce al), or gauge
//               refinements (-enforce kkt) [-1: 8 for al, 3 for kkt].
//   -sweep-tol  loose relative tolerance of the early AL sweeps,
//               tightening to -rt (SetSweepTolerance); 0: every sweep
//               at -rt, the reproducible endpoint [0].
//   -kkt-order  order of the KKT multiplier space (0: the displacement
//               order). Below the displacement order the multiplier
//               misses part of the normal jump, and on these 2-D
//               meshes the gauge refinements then amplify it from one
//               to the next: watch the printed history [0].
//   -geps       fluid gauge penalty epsilon (SetFluidGauge, with
//               mu_g = kappa) [1e-2].
//   -gref       gauge refinements of the welded comparison (-compare);
//               the slipping solver interleaves one per sweep [3].
//   -lmax       highest degree of the surface analysis [-1:
//               max(4, l + 2)].
//   -eps        shape amplitude of an aspherical mesh; a flag that
//               contradicts the mesh manifest is refused
//               [manifest, else 0].
//   -beta       elliptical part of the shape [manifest, else 0.5].
//   -buffer     buffer thickness of the shape's taper
//               [manifest, else 0.2].
//   -rc         physical core radius [manifest, else 3483/6371].
//   -compare / -no-compare   also solve the welded gauged referential
//               problem on the same mesh and Dahlen's mixed problem on
//               the spherical counterpart mesh, and compare [off].
//   -mref       spherical mesh for Dahlen's problem with -compare
//               (default: the mesh's name with 'aspherical_' ->
//               'spherical_', or the mesh itself if spherical) [""].
//   -csv        CSV table of the constraint history ("": none)
//               [slipping_interface.csv].
//   -vis / -no-vis   show the fields in GLVis [on].
//
// Sample runs (with mpiexec -np N in front in a parallel build):
//    ./slipping_interface                                  (aspherical, broken)
//    ./slipping_interface -compare
//    ./slipping_interface -m ../data/spherical_fluid_core_buffer_2d.mesh \
//        -zeta single -enforce kkt
//    ./slipping_interface -l 3 -sweep-tol 1e-4
//    ./slipping_interface -m ../data/aspherical_fluid_core_buffer_3d.mesh -o 1
// ============================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

#include "AdGIA.hpp"
#include "visualisation.hpp"

using namespace mfem;
using namespace AdGIA;

namespace {

#ifdef MFEM_USE_MPI
using MeshType = ParMesh;
using SubMeshType = ParSubMesh;
using SpaceType = ParFiniteElementSpace;
using FieldType = ParGridFunction;
bool Root() { return Mpi::Root(); }
#else
using MeshType = Mesh;
using SubMeshType = SubMesh;
using SpaceType = FiniteElementSpace;
using FieldType = GridFunction;
bool Root() { return true; }
#endif

// Global reductions (serial: no-ops).
void SumAll(Vector& v) {
#ifdef MFEM_USE_MPI
  MPI_Allreduce(MPI_IN_PLACE, v.GetData(), v.Size(),
                MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
#endif
}
real_t MaxAll(real_t x) {
#ifdef MFEM_USE_MPI
  MPI_Allreduce(MPI_IN_PLACE, &x, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                MPI_COMM_WORLD);
#endif
  return x;
}
real_t MinAll(real_t x) { return -MaxAll(-x); }

using Clock = std::chrono::steady_clock;
double Seconds(Clock::time_point since) {
  return std::chrono::duration<double>(Clock::now() - since).count();
}

constexpr real_t kPi = std::numbers::pi_v<real_t>;

// The two-layer model (non-dimensional, those of self_gravitating_solvers):
// unit radius, uniform density, fluid core below r_c.
constexpr real_t kG = 0.05;
constexpr real_t kRho = 1.0;
constexpr real_t kKappa = 1.0;
constexpr real_t kMu = 0.5;
constexpr real_t kRcDefault = 3483.0 / 6371.0;

// Attributes of the meshes of meshes/aspherical_body.py --fluid-core (and
// of layered_earth.py --layers 2): domains 1 core, 2 mantle, 3 buffer;
// boundaries 1 core boundary, 2 surface, 3 outer sphere.
constexpr int kCoreBoundary = 1;
constexpr int kSurface = 2;

// phi_e = D^{-1}: the exact inverse of the generator's radial stretch
// (as in benchmarks/relabelling/aspherical_reference.cpp). D moves the
// point x of the physical body to the reference point
//   (r + h) x^,  h = eps r s            for r <= 1,
//                h = eps s q^2, q = (1+B-r)/B   for 1 <= r <= 1+B,
// the identity beyond. The direction is unchanged, so s is evaluated at
// the reference point and the radius inverted per point.
class InverseShape : public Diffeomorphism {
 public:
  InverseShape(int dim, real_t eps, real_t beta, real_t buffer)
      : Diffeomorphism(dim), eps_(eps), beta_(beta), B_(buffer) {}

  void Eval(Vector& V, ElementTransformation& T,
            const IntegrationPoint& ip) override {
    Vector y(vdim);
    T.Transform(ip, y);
    Map(y, V);
  }

  void EvalGradient(DenseMatrix& F, ElementTransformation& T,
                    const IntegrationPoint& ip) override {
    Vector y(vdim), x(vdim);
    T.Transform(ip, y);
    Map(y, x);
    DenseMatrix FD(vdim);
    ForwardGradient(x, FD);
    F.SetSize(vdim);
    CalcInverse(FD, F);
  }

 private:
  // The physical radius of the reference radius rt along the direction
  // with angular factor s: r + h(r) = rt solved for r.
  real_t Radius(real_t rt, real_t s) const {
    const real_t inside = rt / (1.0 + eps_ * s);
    if (inside <= 1.0) {
      return inside;
    }
    if (rt >= 1.0 + B_) {
      return rt;
    }
    // r + eps s q^2 = rt: with u = 1+B-r, a = eps s / B^2, c = 1+B-rt,
    // a u^2 - u + c = 0, the root tending to u = c as a -> 0, written
    // without cancellation.
    const real_t a = eps_ * s / (B_ * B_), c = 1.0 + B_ - rt;
    const real_t disc = std::max(real_t(0), 1.0 - 4.0 * a * c);
    return 1.0 + B_ - 2.0 * c / (1.0 + std::sqrt(disc));
  }

  // cos theta (from the last axis), cos 2 phi, sin 2 phi of a direction.
  void Angles(const Vector& y, real_t& ct, real_t& c2p, real_t& s2p) const {
    const real_t r = y.Norml2();
    ct = vdim == 3 ? y[2] / std::max(r, real_t(1e-300)) : 0.0;
    const real_t rho2 = vdim == 3 ? y[0] * y[0] + y[1] * y[1] : r * r;
    c2p = 1.0;
    s2p = 0.0;
    if (rho2 > 1e-300) {
      c2p = (y[0] * y[0] - y[1] * y[1]) / rho2;
      s2p = 2.0 * y[0] * y[1] / rho2;
    }
  }

  real_t Shape(const Vector& y) const {
    real_t ct, c2p, s2p;
    Angles(y, ct, c2p, s2p);
    return 0.5 * (3.0 * ct * ct - 1.0) + beta_ * (1.0 - ct * ct) * c2p;
  }

  void Map(const Vector& y, Vector& x) const {
    const real_t rt = y.Norml2();
    x = y;
    if (rt > 1e-300) {
      x *= Radius(rt, Shape(y)) / rt;
    }
  }

  // grad D at the physical point x, pole-regular in Cartesian form:
  //   F_D = (1 + dh/dr) r^ r^T + (1 + h/r)(I - r^ r^T) + r^ (x) v_t,
  // v_t = (c1/r) grad_1 s, h = c1(r) s.
  // The radial derivative jumps at the surface (the taper's knot). On
  // the surface itself the body's side is taken, the side every surface
  // integral here is taken from: the quadrature points of the curved
  // facets land on the unit sphere only to their geometric error, so
  // points within kKnot of it count as on it (buffer quadrature points
  // lie far further out).
  static constexpr real_t kKnot = 1e-4;
  void ForwardGradient(const Vector& x, DenseMatrix& FD) const {
    const int dim = vdim;
    const real_t r = x.Norml2();
    FD.SetSize(dim);
    FD = 0.0;
    for (int i = 0; i < dim; i++) {
      FD(i, i) = 1.0;
    }
    if (r < 1e-300 || r >= 1.0 + B_) {
      return;
    }
    real_t ct, c2p, s2p;
    Angles(x, ct, c2p, s2p);
    const real_t s = Shape(x);
    real_t h, hr, c1;
    if (r <= 1.0 + kKnot) {
      h = eps_ * r * s;
      hr = eps_ * s;
      c1 = eps_ * r;
    } else {
      const real_t q = (1.0 + B_ - r) / B_;
      h = eps_ * s * q * q;
      hr = -2.0 * eps_ * s * q / B_;
      c1 = eps_ * q * q;
    }
    Vector er(x);
    er /= r;
    Vector vt(dim);
    vt = 0.0;
    if (dim == 3) {
      // grad_1 s = A (sin th th^) - 2 beta s2p (sin th ph^), with
      // A = ct (2 beta c2p - 3); both brackets smooth on the axis.
      const real_t st2 = std::max(real_t(0), 1.0 - ct * ct);
      const real_t A = ct * (2.0 * beta_ * c2p - 3.0);
      const real_t stth[3] = {ct * x[0] / r, ct * x[1] / r, -st2};
      const real_t stph[3] = {-x[1] / r, x[0] / r, 0.0};
      for (int i = 0; i < 3; i++) {
        vt[i] = (c1 / r) * (A * stth[i] - 2.0 * beta_ * s2p * stph[i]);
      }
    } else {
      // The plane theta = pi/2: s = -1/2 + beta cos 2phi.
      const real_t stph[2] = {-x[1] / r, x[0] / r};
      for (int i = 0; i < 2; i++) {
        vt[i] = (c1 / r) * (-2.0 * beta_ * s2p * stph[i]);
      }
    }
    const real_t Frr = 1.0 + hr, Ftt = 1.0 + h / r;
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        FD(i, j) = Ftt * ((i == j) - er[i] * er[j]) + Frr * er[i] * er[j] +
                   er[i] * vt[j];
      }
    }
  }

  real_t eps_, beta_, B_;
};

// One orthonormal surface harmonic of the PHYSICAL direction phi_e(x).
class HarmonicCoefficient : public Coefficient {
 public:
  HarmonicCoefficient(const SurfaceHarmonics& basis, int index,
                      Diffeomorphism& map)
      : basis_(basis), i_(index), map_(map) {}
  real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    map_.Eval(x_, T, ip);
    basis_.Eval(x_, Y_);
    return Y_[i_];
  }

 private:
  const SurfaceHarmonics& basis_;
  int i_;
  Diffeomorphism& map_;
  Vector x_, Y_;
};

// The fluid's pressure perturbation p = -kappa div u, the divergence
// taken in the physical frame: tr(Du F^{-1}).
class FluidPressureCoefficient : public Coefficient {
 public:
  FluidPressureCoefficient(const GridFunction& u, Diffeomorphism& map,
                           real_t kappa)
      : u_(u), map_(map), kappa_(kappa) {}
  real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    T.SetIntPoint(&ip);
    u_.GetVectorGradient(T, Du_);
    map_.EvalGradient(F_, T, ip);
    Fi_.SetSize(F_.Height());
    CalcInverse(F_, Fi_);
    real_t div = 0.0;
    for (int i = 0; i < Du_.Height(); i++) {
      for (int j = 0; j < Du_.Width(); j++) {
        div += Du_(i, j) * Fi_(j, i);
      }
    }
    return -kappa_ * div;
  }

 private:
  const GridFunction& u_;
  Diffeomorphism& map_;
  real_t kappa_;
  DenseMatrix Du_, F_, Fi_;
};

// The tangential (or, with normal = true, the normal) part of the jump
// u_s - u_f on Sigma, against the PHYSICAL radial direction.
class InterfaceJumpCoefficient : public Coefficient {
 public:
  InterfaceJumpCoefficient(const GridFunction& us, const GridFunction& uf,
                           Diffeomorphism& map, bool normal)
      : us_(us), uf_(uf), map_(map), normal_(normal) {}
  real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    us_.GetVectorValue(T, ip, a_);
    uf_.GetVectorValue(T, ip, b_);
    a_ -= b_;
    map_.Eval(x_, T, ip);
    x_ /= x_.Norml2();
    const real_t dn = a_ * x_;
    if (normal_) {
      return std::abs(dn);
    }
    a_.Add(-dn, x_);
    return a_.Norml2();
  }

 private:
  const GridFunction &us_, &uf_;
  Diffeomorphism& map_;
  bool normal_;
  Vector a_, b_, x_;
};

// The surface harmonic analysis of one solution and its Love numbers.
struct Analysis {
  Vector cu, cv, cphi;  // coefficients of u.x^, the tangential part, phi1
  real_t h = NAN, l = NAN, k = NAN;
  // The largest other coefficient relative to the loaded one: over every
  // other harmonic, and over those not of degree 1 (where a residual
  // translation of the body, a frame matter, also shows).
  real_t spurious = NAN, spurious_no1 = NAN;
};

// Coefficients on the PHYSICAL surface, by quadrature over the reference
// surface (boundary attribute kSurface of `mesh`, the mesh of u): each
// integrand at the physical point phi_e(x), the weight the physical area
// element |cof(F) n| dS. Fields are evaluated in the adjacent volume
// element. `grad_zeta0` (referential gradient of the background
// potential) converts a referential zeta1 to the Eulerian phi1 =
// zeta1 - (F^{-T} grad zeta0) . u; null when `pot` is Eulerian already.
Analysis Analyse(Mesh& mesh, const GridFunction& u, const GridFunction& pot,
                 VectorCoefficient* grad_zeta0, Diffeomorphism& map,
                 const SurfaceHarmonics& basis, int main, real_t g,
                 real_t phi_sigma, int qorder) {
  const int dim = mesh.Dimension();
  const int n = basis.Size();
  Analysis a;
  a.cu.SetSize(n);
  a.cv.SetSize(n);
  a.cphi.SetSize(n);
  a.cu = 0.0;
  a.cv = 0.0;
  a.cphi = 0.0;
  Vector x(dim), uv(dim), gz(dim), bvec(dim), nor(dim), nu(dim), Y;
  DenseMatrix F(dim), adj(dim), Fi(dim), gradY;
  for (int be = 0; be < mesh.GetNBE(); be++) {
    if (mesh.GetBdrAttribute(be) != kSurface) {
      continue;
    }
    FaceElementTransformations* ft = mesh.GetBdrFaceTransformations(be);
    if (!ft) {
      continue;
    }
    const IntegrationRule& ir = IntRules.Get(ft->GetGeometryType(), qorder);
    for (int q = 0; q < ir.GetNPoints(); q++) {
      const IntegrationPoint& ip = ir.IntPoint(q);
      ft->SetAllIntPoints(&ip);
      ElementTransformation& Te = *ft->Elem1;
      const IntegrationPoint& eip = ft->GetElement1IntPoint();
      // Nanson: the physical area element is |cof(F) n| dS.
      CalcOrtho(ft->Jacobian(), nor);
      map.EvalGradient(F, Te, eip);
      CalcAdjugate(F, adj);
      adj.MultTranspose(nor, nu);
      const real_t w = ip.weight * nu.Norml2();
      map.Eval(x, Te, eip);
      basis.EvalWithGradient(x, Y, gradY);
      u.GetVectorValue(Te, eip, uv);
      real_t phi = pot.GetValue(Te, eip);
      if (grad_zeta0) {
        grad_zeta0->Eval(gz, Te, eip);
        CalcInverse(F, Fi);
        Fi.MultTranspose(gz, bvec);
        phi -= bvec * uv;
      }
      const real_t r = x.Norml2();
      const real_t ur = (uv * x) / r;
      for (int j = 0; j < n; j++) {
        a.cu[j] += w * ur * Y[j];
        a.cphi[j] += w * phi * Y[j];
        real_t ut = 0.0;
        for (int d = 0; d < dim; d++) {
          ut += uv[d] * gradY(d, j);
        }
        a.cv[j] += w * ut;
      }
    }
  }
  SumAll(a.cu);
  SumAll(a.cv);
  SumAll(a.cphi);
  // The tangential coefficients: divided by int |grad_1 Y|^2 = l(l+1)
  // (l^2 in 2-D), zero at degree 0.
  for (int j = 0; j < n; j++) {
    const int l = basis.Degree(j);
    const real_t norm = dim == 3 ? l * (l + 1.0) : real_t(l * l);
    a.cv[j] = l > 0 ? a.cv[j] / norm : 0.0;
  }
  a.h = -g * a.cu[main] / phi_sigma;
  a.l = -g * a.cv[main] / phi_sigma;
  a.k = a.cphi[main] / phi_sigma - 1.0;
  // In 2-D the potential is determined only up to a constant (a null
  // mode of the 2-D problem), so its degree-0 coefficient is gauge and
  // left out.
  real_t m = 0.0, m1 = 0.0;
  for (int j = 0; j < n; j++) {
    if (j != main) {
      const bool gauge = dim == 2 && basis.Degree(j) == 0;
      const real_t c = std::max(
          std::abs(a.cu[j]) / std::abs(a.cu[main]),
          gauge ? 0.0 : std::abs(a.cphi[j]) / std::abs(a.cphi[main]));
      m = std::max(m, c);
      if (basis.Degree(j) != 1) {
        m1 = std::max(m1, c);
      }
    }
  }
  a.spurious = m;
  a.spurious_no1 = m1;
  return a;
}

// A mesh with its regions as SubMeshes.
struct Model {
  std::unique_ptr<MeshType> parent;
  std::unique_ptr<SubMeshType> fluid, solid, buffer, body, outer;
  int dim = 0;
  real_t r_out = 0.0;

  explicit Model(const std::string& file) {
    Mesh smesh(file.c_str(), 1, 1);
#ifdef MFEM_USE_MPI
    parent = std::make_unique<ParMesh>(MPI_COMM_WORLD, smesh);
#else
    parent = std::make_unique<Mesh>(std::move(smesh));
#endif
    dim = parent->Dimension();
    Array<int> fluid_attr({1}), solid_attr({2}), buffer_attr({3}),
        body_attr({1, 2}), outer_attr({2, 3});
    fluid = std::make_unique<SubMeshType>(
        SubMeshType::CreateFromDomain(*parent, fluid_attr));
    solid = std::make_unique<SubMeshType>(
        SubMeshType::CreateFromDomain(*parent, solid_attr));
    buffer = std::make_unique<SubMeshType>(
        SubMeshType::CreateFromDomain(*parent, buffer_attr));
    body = std::make_unique<SubMeshType>(
        SubMeshType::CreateFromDomain(*parent, body_attr));
    outer = std::make_unique<SubMeshType>(
        SubMeshType::CreateFromDomain(*parent, outer_attr));
    Vector bb_min, bb_max;
    parent->GetBoundingBox(bb_min, bb_max);
    r_out = bb_max.Normlinf();
  }
};

Array<int> Marker(Mesh& mesh, int attribute) {
  Array<int> m(std::max(mesh.bdr_attributes.Max(), attribute));
  m = 0;
  m[attribute - 1] = 1;
  return m;
}

// A copy of a mesh with its nodes moved by phi_e: the physical body.
std::unique_ptr<MeshType> Physical(MeshType& mesh, Diffeomorphism& map) {
  auto copy = std::make_unique<MeshType>(mesh);
  copy->Transform(map);
  return copy;
}

// The relative L2 difference a - b of two mantle displacements with the
// rigid modes of `projector` removed (true dofs).
real_t RigidFreeDifference(const FieldType& a, const FieldType& b,
                           const NullSpaceProjector& projector) {
  FieldType d(a), ref(b);
  d -= b;
  Vector t;
  d.GetTrueDofs(t);
  projector.Project(t);
  d.SetFromTrueDofs(t);
  ref.GetTrueDofs(t);
  projector.Project(t);
  ref.SetFromTrueDofs(t);
  const int dim = a.FESpace()->GetVDim();
  Vector zero(dim);
  zero = 0.0;
  VectorConstantCoefficient z(zero);
  return d.ComputeL2Error(z) / ref.ComputeL2Error(z);
}

// A displacement on another SubMesh (the welded body, the core) carried to
// the mantle through the parent: where the two meet, its own values.
std::unique_ptr<FieldType> OnMantle(const FieldType& u, SpaceType& fes_s,
                                    MeshType& parent,
                                    const FiniteElementCollection& fec) {
  const int dim = parent.Dimension();
  SpaceType parent_v(&parent, &fec, dim);
  FieldType on_parent(&parent_v);
  on_parent = 0.0;
  auto out = std::make_unique<FieldType>(&fes_s);
  SubMeshType::Transfer(u, on_parent);
  SubMeshType::Transfer(on_parent, *out);
  return out;
}

// One row of the comparison table.
struct Row {
  std::string name;
  Analysis a;
  int iterations = 0;
  double seconds = 0.0;
  real_t mantle_difference = NAN;  // vs the slipping solution
};

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif

  const char* mesh_file = "../data/aspherical_fluid_core_buffer_2d.mesh";
  const char* mref_file = "";
  const char* zeta_mode = "broken";
  const char* enforce = "al";
  const char* csv_file = "slipping_interface.csv";
  int order = 2, degree = 2, dtn_degree = 12, lmax = -1;
  int al_iterations = -1, kkt_order = 0, gauge_refinements = 3;
  real_t rel_tol = 1e-10, theta = 1e2, sweep_tol = 0.0, gauge_eps = 1e-2;
  real_t eps = NAN, beta = NAN, buffer = NAN, rc = NAN;
  bool compare = false, visualization = true;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
                 "Two-layer mesh with a buffer (domains 1 core, 2 mantle, 3 "
                 "buffer; boundaries 1 core boundary, 2 surface, 3 outer), "
                 "spherical or aspherical (meshes/aspherical_body.py "
                 "--fluid-core). An aspherical mesh is the reference body "
                 "of the spherical physical model.");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&degree, "-l", "--load-degree",
                 "Degree of the surface load (2 or more): Y_l0 in 3-D, "
                 "cos(l theta) in 2-D.");
  args.AddOption(&dtn_degree, "-deg", "--dtn-degree",
                 "Truncation degree of the Dirichlet-to-Neumann map.");
  args.AddOption(&rel_tol, "-rt", "--rel-tol",
                 "Relative tolerance of the linear solves.");
  args.AddOption(&zeta_mode, "-zeta", "--zeta",
                 "Potential organisation: 'single' (one zeta on the ball, "
                 "fluid extension; identity map only) or 'broken' (zeta on "
                 "each side of Sigma, scalar-jump constraint; mapped).");
  args.AddOption(&enforce, "-enforce", "--enforce",
                 "Constraint enforcement: 'al' (penalty + augmented "
                 "Lagrangian) or 'kkt' (multiplier, saddle-point MINRES; "
                 "single-valued zeta only).");
  args.AddOption(&theta, "-theta", "--theta",
                 "Constraint penalty theta (normal jump and, broken, the "
                 "zeta jump).");
  args.AddOption(&al_iterations, "-al", "--al-iterations",
                 "AL sweeps per solve (al), or gauge refinements (kkt); "
                 "default 8 (al), 3 (kkt).");
  args.AddOption(&sweep_tol, "-sweep-tol", "--sweep-tolerance",
                 "Loose relative tolerance of the early AL sweeps, "
                 "tightening to -rt (SetSweepTolerance); 0: every sweep at "
                 "-rt, the reproducible endpoint.");
  args.AddOption(&kkt_order, "-kkt-order", "--kkt-order",
                 "Order of the KKT multiplier space (0: the displacement "
                 "order). Below the displacement order the multiplier "
                 "misses part of the normal jump, and on these meshes the "
                 "gauge refinements then amplify it from one to the next: "
                 "watch the printed history.");
  args.AddOption(&gauge_eps, "-geps", "--gauge-epsilon",
                 "Fluid gauge penalty epsilon (SetFluidGauge, mu_g = "
                 "kappa).");
  args.AddOption(&gauge_refinements, "-gref", "--gauge-refinements",
                 "Gauge refinements of the welded comparison (-compare); "
                 "the slipping solver interleaves one per sweep.");
  args.AddOption(&lmax, "-lmax", "--max-degree",
                 "Highest degree of the surface analysis (default "
                 "max(4, l + 2)).");
  args.AddOption(&eps, "-eps", "--epsilon",
                 "Shape amplitude of an aspherical mesh; read from the "
                 "mesh manifest (a contradicting value is refused), 0 "
                 "without one.");
  args.AddOption(&beta, "-beta", "--beta",
                 "Elliptical part of the shape (manifest, else 0.5).");
  args.AddOption(&buffer, "-buffer", "--buffer",
                 "Buffer thickness of the shape's taper (manifest, else "
                 "0.2).");
  args.AddOption(&rc, "-rc", "--core-radius",
                 "Physical core radius (manifest, else 3483/6371).");
  args.AddOption(&compare, "-compare", "--compare", "-no-compare",
                 "--no-compare",
                 "Also solve the welded gauged referential problem on the "
                 "same mesh and Dahlen's mixed problem on the spherical "
                 "counterpart mesh, and compare.");
  args.AddOption(&mref_file, "-mref", "--reference-mesh",
                 "Spherical mesh for Dahlen's problem with -compare "
                 "(default: the mesh's name with 'aspherical_' -> "
                 "'spherical_', or the mesh itself if spherical).");
  args.AddOption(&csv_file, "-csv", "--csv",
                 "CSV table of the constraint history (\"\" for none).");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization", "Show the fields in GLVis.");
  args.Parse();
  if (!args.Good()) {
    if (Root()) {
      args.PrintUsage(std::cout);
    }
    return 1;
  }

  // Refusals that need no mesh.
  const std::string zeta_s(zeta_mode), enforce_s(enforce);
  auto refuse = [](const std::string& why) {
    if (Root()) {
      std::cout << "slipping_interface: " << why << "\n";
    }
    return 2;
  };
  if (zeta_s != "single" && zeta_s != "broken") {
    return refuse("-zeta must be 'single' or 'broken'.");
  }
  if (enforce_s != "al" && enforce_s != "kkt") {
    return refuse("-enforce must be 'al' or 'kkt'.");
  }
  const bool broken = zeta_s == "broken", kkt = enforce_s == "kkt";
  if (kkt && broken) {
    return refuse(
        "KKT enforcement is implemented for the single-valued organisation "
        "only; use -enforce al with -zeta broken, or -zeta single (on a "
        "spherical mesh) with -enforce kkt.");
  }
  if (degree < 2) {
    return refuse(
        "-l must be 2 or more (degree 0 changes the mass of the body, "
        "degree 1 moves its centre: both need a frame convention this "
        "example does not make).");
  }
  if (al_iterations < 0) {
    al_iterations = kkt ? 3 : 8;
  }
  if (lmax < 0) {
    lmax = std::max(4, degree + 2);
  }
  lmax = std::max(lmax, degree);

  // The shape of the mesh, from its manifest when it has one (the meta
  // aspherical_body.py writes): authoritative; a flag that contradicts it
  // is refused.
  {
    std::string mpath(mesh_file);
    const auto dot = mpath.rfind('.');
    mpath = mpath.substr(0, dot) + ".json";
    bool contradiction = false;
    std::string message;
    if (dot != std::string::npos && std::ifstream(mpath).good()) {
      const MeshManifest manifest(mpath);
      auto adopt = [&](const char* key, real_t& value) {
        if (!manifest.HasMeta(key)) {
          return;
        }
        const real_t recorded = manifest.MetaNumber(key);
        if (!std::isnan(value) && std::abs(value - recorded) > 1e-12) {
          contradiction = true;
          message += std::string(" -") + key + " " + std::to_string(value) +
                     " (manifest: " + std::to_string(recorded) + ")";
        }
        value = recorded;
      };
      adopt("eps", eps);
      adopt("beta", beta);
      adopt("buffer", buffer);
      adopt("r_c", rc);
    }
    if (contradiction) {
      return refuse("the flags contradict the mesh manifest " + mpath +
                    ":" + message + "; drop them, the manifest is "
                    "authoritative.");
    }
    if (std::isnan(eps)) eps = 0.0;
    if (std::isnan(beta)) beta = 0.5;
    if (std::isnan(buffer)) buffer = 0.2;
    if (std::isnan(rc)) rc = kRcDefault;
  }
  const bool mapped = eps != 0.0;
  if (mapped && !broken) {
    std::string spherical(mesh_file);
    const auto pos = spherical.rfind("aspherical_");
    spherical = pos == std::string::npos
                    ? std::string("a spherical mesh")
                    : "the spherical counterpart, -m " +
                          spherical.replace(pos, 11, "spherical_");
    return refuse(
        "the single-valued organisation (-zeta single) assembles its "
        "gravity mismatch terms at phi_e = id only, and this mesh is an "
        "aspherical reference body (eps = " + std::to_string(eps) +
        "). Use -zeta broken, which is mapped throughout, or " + spherical +
        ".");
  }
  if (Root()) {
    args.PrintOptions(std::cout);
  }

  Model model(mesh_file);
  const int dim = model.dim;
  MFEM_VERIFY(model.parent->attributes.Max() == 3,
              "slipping_interface: the mesh must have three domain "
              "attributes (core, mantle, buffer).");
  MeshType& parent = *model.parent;
  SubMeshType &solid = *model.solid, &fluid = *model.fluid,
              &buffer_mesh = *model.buffer;
  H1_FECollection fec(order, dim);
  SpaceType fes_s(&solid, &fec, dim), fes_f(&fluid, &fec, dim);
  SpaceType fes_b(&buffer_mesh, &fec, dim), fes_zeta(&parent, &fec);

  // The equilibrium mapping: exact inverse stretch, or the identity.
  std::unique_ptr<Diffeomorphism> phi_e;
  if (mapped) {
    phi_e = std::make_unique<InverseShape>(dim, eps, beta, buffer);
  } else {
    phi_e = std::make_unique<IdentityDiffeomorphism>(dim);
  }
  Diffeomorphism& map = *phi_e;

  // Seatbelts: the map must be the identity on the DtN sphere, carry the
  // reference surface onto the unit sphere and keep the core's
  // quadrature points below r_c and the mantle's above (the shear modulus
  // is a function of the physical radius).
  {
    real_t d_out = 0.0, d_surf = 0.0;
    Vector x(dim), y(dim);
    // The outer sphere is a boundary of the parent, the surface one of the
    // mantle (on the parent it is an interior face).
    auto boundary_points = [&](Mesh& mesh, int attribute, auto visit) {
      for (int be = 0; be < mesh.GetNBE(); be++) {
        if (mesh.GetBdrAttribute(be) != attribute) {
          continue;
        }
        auto* T = mesh.GetBdrElementTransformation(be);
        const auto& ir =
            IntRules.Get(mesh.GetBdrElementGeometry(be), 2 * order + 2);
        for (int q = 0; q < ir.GetNPoints(); q++) {
          T->SetIntPoint(&ir.IntPoint(q));
          map.Eval(x, *T, ir.IntPoint(q));
          T->Transform(ir.IntPoint(q), y);
          visit();
        }
      }
    };
    boundary_points(parent, 3, [&]() {
      y -= x;
      d_out = std::max(d_out, y.Norml2());
    });
    boundary_points(solid, kSurface, [&]() {
      d_surf = std::max(d_surf, std::abs(x.Norml2() - 1.0));
    });
    real_t r_core = 0.0, r_mantle = 1e300;
    for (int e = 0; e < parent.GetNE(); e++) {
      const int attr = parent.GetAttribute(e);
      if (attr == 3) {
        continue;
      }
      auto* T = parent.GetElementTransformation(e);
      const auto& ir = IntRules.Get(parent.GetElementGeometry(e),
                                    2 * order + 2);
      for (int q = 0; q < ir.GetNPoints(); q++) {
        T->SetIntPoint(&ir.IntPoint(q));
        map.Eval(x, *T, ir.IntPoint(q));
        const real_t r = x.Norml2();
        if (attr == 1) {
          r_core = std::max(r_core, r);
        } else {
          r_mantle = std::min(r_mantle, r);
        }
      }
    }
    d_out = MaxAll(d_out);
    d_surf = MaxAll(d_surf);
    r_core = MaxAll(r_core);
    r_mantle = MinAll(r_mantle);
    if (Root()) {
      std::cout << "\nReference body: "
                << (mapped ? "aspherical, eps " + std::to_string(eps) +
                                 ", beta " + std::to_string(beta) +
                                 " (phi_e = exact inverse stretch)"
                           : std::string("spherical (phi_e = identity)"))
                << "\n  phi_e on the DtN sphere: identity to " << d_out
                << "\n  surface lands on the unit sphere to " << d_surf
                << " (geometric error of the curved facets)"
                << "\n  core quadrature points up to r = " << r_core
                << ", mantle from r = " << r_mantle << " (r_c = " << rc
                << ")\n";
    }
    MFEM_VERIFY(d_out < 1e-8, "phi_e must be the identity on the DtN sphere.");
    MFEM_VERIFY(d_surf < 2e-2, "the mesh does not match the shape "
                               "parameters (eps, beta).");
    MFEM_VERIFY(r_core < rc && r_mantle > rc,
                "a core or mantle quadrature point lands on the wrong side "
                "of r_c: the mesh and -rc disagree.");
  }

  // The background: hydrostatic radial model, relabelled when mapped.
  RadialHydrostaticBackground bg(
      dim, [](real_t) { return kRho; }, [](real_t) { return kKappa; },
      [rc](real_t r) { return r < rc ? 0.0 : kMu; }, kG, 1.0);
  std::unique_ptr<RelabelledBackground> rel;
  if (mapped) {
    rel = std::make_unique<RelabelledBackground>(bg, map);
  }
  ReferentialElasticRheology& rheology = rel ? rel->Rheology()
                                             : bg.Rheology();
  Coefficient& density = rel ? rel->Density() : bg.Density();
  Coefficient& pressure = rel ? rel->Pressure() : bg.Pressure();
  const real_t g_surface = bg.GravityAt(1.0);

  // The load: the unit physical harmonic, per referential area (Nanson).
  SurfaceHarmonics basis(dim, lmax);
  const int main_index =
      dim == 3 ? basis.Index(degree, 0) : basis.Index(degree, degree);
  HarmonicCoefficient Y_load(basis, main_index, map);
  NansonAreaCoefficient nanson(map);
  ProductCoefficient sigma(Y_load, nanson);
  const real_t phi_sigma = dim == 3 ? -4.0 * kPi * kG / (2.0 * degree + 1.0)
                                    : -2.0 * kPi * kG / degree;
  const int qorder = 2 * order + 4;
  ConstantCoefficient mu_gauge(kKappa);

  // ---- The slipping problem.
  auto interface_s = Marker(solid, kCoreBoundary);
  auto surface_s = Marker(solid, kSurface);
  LinearQuasiStaticReferentialSelfGravitatingSlipProblem slip(
      &fes_s, &fes_f, &fes_zeta, rheology, density, pressure, interface_s,
      kG, dtn_degree);
  auto Evac = NewRadialVacuumExtension(fes_s, fes_b, 1.0, model.r_out);
  slip.SetPrescribedVacuumExtension(fes_b, *Evac);
  slip.SetFluidGauge(mu_gauge, gauge_eps);
  slip.SetConstraint(theta, al_iterations);
  slip.SetSweepTolerance(sweep_tol);
  // Only one of these is used, by organisation and enforcement; they must
  // outlive the problem.
  std::unique_ptr<SpaceType> fes_zo, fes_lam;
  std::unique_ptr<H1_FECollection> fec_lam;
  decltype(NewRadialFluidExtension(fes_s, fes_f, rc)) Ef;
  if (broken) {
    fes_zo = SubMeshDofInjection::MakeShadowSpace(fes_zeta, *model.outer);
    slip.EnableBrokenZeta(fes_zo.get(), theta);
  } else {
    Ef = NewRadialFluidExtension(fes_s, fes_f, rc);
    slip.SetFluidExtension(*Ef);
  }
  if (kkt) {
    fec_lam = std::make_unique<H1_FECollection>(
        kkt_order > 0 ? kkt_order : order, dim);
    fes_lam = std::make_unique<SpaceType>(&solid, fec_lam.get());
    slip.EnableKKT(fes_lam.get());
  }
  slip.SetSurfaceLoad(sigma, surface_s);
  slip.SetRelTol(rel_tol);

  if (Root()) {
    std::cout << "\nSlipping interface: " << (broken ? "broken" : "single-valued")
              << " zeta, " << (kkt ? "KKT" : "penalty + AL") << ", theta "
              << theta << ", " << al_iterations
              << (kkt ? " gauge refinements" : " sweeps")
              << ", gauge eps " << gauge_eps << "; load degree " << degree
              << ", order " << order << "\n";
  }
  auto t0 = Clock::now();
  slip.AssembleForce(0.0);
  const bool converged = slip.Solve();
  const double slip_seconds = Seconds(t0);
  const int slip_its = slip.LastOuterIterations();

  // Constraint histories: per AL sweep (per gauge refinement for KKT).
  const auto& jumps = slip.NormalJumpHistory();
  const auto& zjumps = slip.ZetaJumpHistory();
  std::vector<std::string> columns{"iteration", "normal_jump"};
  if (broken) {
    columns.push_back("zeta_jump");
  }
  examples::CsvTable table(csv_file, columns);
  table.Meta("title", std::string("Slipping interface: constraint history (") +
                          (broken ? "broken" : "single") + " zeta, " +
                          (kkt ? "KKT" : "AL") + ")")
      .Meta("note", kkt ? "normal-jump energy after each gauge refinement"
                        : "constraint energies after each AL sweep")
      .Meta("xlabel", kkt ? "gauge refinement" : "AL sweep")
      .Meta("ylabel", "constraint energy")
      .Meta("logy", "true");
  if (Root()) {
    std::cout << "\n  " << (kkt ? "refinement" : "sweep") << "   normal jump"
              << (broken ? "     zeta jump" : "") << "\n";
  }
  for (std::size_t k = 0; k < jumps.size(); k++) {
    std::vector<double> row{double(k + 1), jumps[k]};
    if (broken) {
      row.push_back(k < zjumps.size() ? zjumps[k] : NAN);
    }
    table.Row(row);
    if (Root()) {
      std::cout << std::setw(8) << k + 1 << std::setw(14)
                << std::scientific << std::setprecision(3) << jumps[k];
      if (broken && k < zjumps.size()) {
        std::cout << std::setw(14) << zjumps[k];
      }
      std::cout << std::defaultfloat << "\n";
    }
  }
  table.Write();
  real_t lambda_norm = NAN;
  if (kkt) {
    const Vector& lam = slip.KKTMultiplier();
    Vector s2(1);
    s2[0] = lam * lam;
    SumAll(s2);
    lambda_norm = std::sqrt(s2[0]);
  }
  const auto pairs = slip.SlipRigidPairResiduals();
  if (Root()) {
    std::cout << "  " << slip_its << " MINRES iterations in all, "
              << std::setprecision(3) << slip_seconds << " s"
              << (converged ? "" : "  (NOT CONVERGED)") << "\n";
    if (kkt) {
      std::cout << "  KKT multiplier norm (true dofs) " << lambda_norm
                << ", constraint residual (final normal jump) "
                << (jumps.empty() ? NAN : jumps.back()) << "\n";
    }
    std::cout << "  slip null pairs under the assembled operator "
                 "(translations, then mantle and core rotations):";
    for (const real_t r : pairs) {
      std::cout << " " << std::scientific << std::setprecision(2) << r;
    }
    std::cout << std::defaultfloat << std::setprecision(6) << "\n";
  }

  // The jump on Sigma, as scalar fields on the mantle that carry its
  // values on the interface dofs (zero elsewhere): u_f comes across
  // through the parent, landing on the interface dofs.
  auto uf_on_solid =
      OnMantle(static_cast<const FieldType&>(slip.FluidDisplacement()), fes_s,
               parent, fec);
  SpaceType fes_s_scalar(&solid, &fec);
  FieldType slip_field(&fes_s_scalar), normal_field(&fes_s_scalar);
  InterfaceJumpCoefficient tangential_c(slip.Displacement(), *uf_on_solid,
                                        map, false),
      normal_c(slip.Displacement(), *uf_on_solid, map, true);
  slip_field = 0.0;
  normal_field = 0.0;
  slip_field.ProjectBdrCoefficient(tangential_c, interface_s);
  normal_field.ProjectBdrCoefficient(normal_c, interface_s);
  const real_t max_slip = MaxAll(slip_field.Normlinf());
  const real_t max_normal = MaxAll(normal_field.Normlinf());

  std::vector<Row> rows;
  {
    Row r;
    r.name = std::string("slip, ") + (broken ? "broken" : "single") + ", " +
             (kkt ? "KKT" : "AL");
    r.a = Analyse(solid, slip.Displacement(), slip.PotentialOnBody(),
                  &slip.BackgroundGravity(), map, basis, main_index,
                  g_surface, phi_sigma, qorder);
    r.iterations = slip_its;
    r.seconds = slip_seconds;
    rows.push_back(r);
  }
  if (Root()) {
    const Analysis& a = rows[0].a;
    std::cout << "\nOn Sigma (at the nodes): largest normal jump " << max_normal
              << ", largest tangential slip " << max_slip
              << "\n  (the tangential slip depends on the fluid's gauge, fixed "
                 "here by the penalty,\n  so it is not an observable and "
                 "differs between organisations; the normal\n  jump is the "
                 "constraint)\n"
              << "\nPhysical surface, coefficients by degree (largest |c| "
                 "over the orders of each degree):\n"
              << "   l        u.x^           phi1\n";
    for (int l = 0; l <= lmax; l++) {
      real_t mu = 0.0, mp = 0.0;
      for (int j = 0; j < basis.Size(); j++) {
        if (basis.Degree(j) == l) {
          mu = std::max(mu, std::abs(a.cu[j]));
          mp = std::max(mp, std::abs(a.cphi[j]));
        }
      }
      std::cout << std::setw(4) << l << std::scientific << std::setprecision(4)
                << std::setw(15) << mu << std::setw(15) << mp
                << (l == degree ? "   <- loaded" : "") << "\n";
    }
    std::cout << std::defaultfloat << std::setprecision(6)
              << "Love numbers at l = " << degree << ": h' " << a.h
              << ", l' " << a.l << ", k' " << a.k << "\nspurious " << a.spurious
              << " (all other harmonics), " << a.spurious_no1
              << " (degree 1 excluded)\n  (phi_sigma = " << phi_sigma
              << ", the closed form for the unit harmonic; g = " << g_surface
              << ")\n";
  }

  // ---- Comparisons.
  std::unique_ptr<FieldType> welded_mantle, dahlen_mantle;
  std::unique_ptr<NullSpaceProjector> rigid;
  if (compare) {
    rigid = MakeRigidModeProjector(fes_s, &map);
    FieldType slip_mantle(&fes_s);
    slip_mantle = slip.Displacement();

    // (a) The welded gauged referential problem, same mesh, map and
    // background: the fluid welded to the mantle, its gauge fixed by the
    // covariant penalty. On this spherically symmetric model it agrees
    // with the slipping solution to discretisation level
    // (doc/gauged_fluid.md, "Tangential slip and the welded space").
    {
      SpaceType fes_body(model.body.get(), &fec, dim);
      LinearQuasiStaticReferentialSelfGravitatingProblem welded(
          &fes_body, &fes_zeta, rheology, density, kG, dtn_degree);
      auto Evac_w = NewRadialVacuumExtension(fes_body, fes_b, 1.0,
                                             model.r_out);
      welded.SetPrescribedVacuumExtension(fes_b, *Evac_w);
      Array<int> fluid_marker(model.body->attributes.Max());
      fluid_marker = 0;
      fluid_marker[0] = 1;
      welded.SetFluid(fluid_marker, mu_gauge,
                      GaugePenaltyOptions{gauge_eps, gauge_refinements});
      welded.SetSurfaceLoad(sigma, Marker(*model.body, kSurface));
      welded.SetRelTol(rel_tol);
      t0 = Clock::now();
      welded.AssembleForce(0.0);
      welded.Solve();
      Row r;
      r.name = "welded, gauged";
      r.seconds = Seconds(t0);
      r.iterations = welded.LastOuterIterations();
      r.a = Analyse(*model.body, welded.Displacement(),
                    welded.PotentialOnBody(), &welded.BackgroundGravity(),
                    map, basis, main_index, g_surface, phi_sigma, qorder);
      welded_mantle = OnMantle(
          static_cast<const FieldType&>(welded.Displacement()), fes_s,
          parent, fec);
      r.mantle_difference =
          RigidFreeDifference(*welded_mantle, slip_mantle, *rigid);
      if (Root()) {
        std::cout << "\nWelded gauged referential: gauge residuals";
        for (const real_t g : welded.GaugeResiduals()) {
          std::cout << " " << std::scientific << std::setprecision(2) << g;
        }
        std::cout << std::defaultfloat << std::setprecision(6) << "\n";
      }
      rows.push_back(r);
    }

    // (b) Dahlen's mixed problem with an eliminated fluid region, on the
    // spherical counterpart (it requires phi_e = id).
    {
      std::string ref(mref_file);
      if (ref.empty()) {
        ref = mesh_file;
        if (mapped) {
          const auto pos = ref.rfind("aspherical_");
          MFEM_VERIFY(pos != std::string::npos,
                      "-compare: name the spherical counterpart with "
                      "-mref.");
          ref.replace(pos, std::string("aspherical_").size(), "spherical_");
        }
      }
      const bool same = ref == std::string(mesh_file);
      std::unique_ptr<Model> other;
      if (!same) {
        other = std::make_unique<Model>(ref);
      }
      Model& m = same ? model : *other;
      SpaceType ds(m.solid.get(), &fec, dim), dphi(m.parent.get(), &fec);
      SpaceType& fes_d = same ? fes_s : ds;
      SpaceType& fes_phi = same ? fes_zeta : dphi;
      ConstantCoefficient rho_c(kRho), kappa_c(kKappa), mu_c(kMu), zero(0.0);
      IsotropicElasticRheology elastic(dim, kappa_c, mu_c);
      FluidRegion core;
      core.attributes = Array<int>({1});
      core.density = &rho_c;
      core.density_gradient = &zero;
      core.interface_marker = Marker(*m.solid, kCoreBoundary);
      std::vector<FluidRegion> fluids{core};
      LinearQuasiStaticMixedSelfGravitatingProblem dahlen(
          &fes_d, &fes_phi, elastic, rho_c, kG, dtn_degree, nullptr, fluids);
      IdentityDiffeomorphism id(dim);
      HarmonicCoefficient Y_sphere(basis, main_index, id);
      dahlen.SetSurfaceLoad(Y_sphere, Marker(*m.solid, kSurface));
      dahlen.SetRelTol(rel_tol);
      t0 = Clock::now();
      dahlen.AssembleForce(0.0);
      dahlen.Solve();
      Row r;
      r.name = std::string("Dahlen, ") + (same ? "same mesh" : "spherical");
      r.seconds = Seconds(t0);
      r.iterations = dahlen.LastOuterIterations();
      r.a = Analyse(*m.solid, dahlen.Displacement(),
                    dahlen.PotentialOnBody(), nullptr, id, basis,
                    main_index, g_surface, phi_sigma, qorder);
      if (same) {
        dahlen_mantle = std::make_unique<FieldType>(&fes_s);
        *dahlen_mantle = dahlen.Displacement();
        r.mantle_difference =
            RigidFreeDifference(*dahlen_mantle, slip_mantle, *rigid);
      }
      if (Root() && !same) {
        std::cout << "Dahlen's problem on " << ref << "\n";
      }
      rows.push_back(r);
    }

    if (Root()) {
      std::cout << "\n  method                   h'           l'           "
                   "k'   spurious   (l!=1)    its   time s  mantle diff\n";
      for (const auto& r : rows) {
        std::cout << "  " << std::left << std::setw(22) << r.name
                  << std::right << std::fixed << std::setprecision(6)
                  << std::setw(11) << r.a.h << std::setw(13) << r.a.l
                  << std::setw(13) << r.a.k << std::scientific
                  << std::setprecision(2) << std::setw(11) << r.a.spurious
                  << std::setw(10) << r.a.spurious_no1 << std::setw(7)
                  << r.iterations << std::fixed << std::setprecision(2)
                  << std::setw(9) << r.seconds << std::scientific
                  << std::setw(13);
        if (std::isnan(r.mantle_difference)) {
          std::cout << "-";
        } else {
          std::cout << r.mantle_difference;
        }
        std::cout << std::defaultfloat << "\n";
      }
      std::cout << std::setprecision(6) << "\n  relative to the slipping "
                                           "solution:\n";
      for (std::size_t i = 1; i < rows.size(); i++) {
        auto rel_d = [](real_t a, real_t b) {
          return std::abs(a - b) / std::abs(b);
        };
        std::cout << "  " << std::left << std::setw(22) << rows[i].name
                  << std::right << std::scientific << std::setprecision(2)
                  << "  dh' " << rel_d(rows[i].a.h, rows[0].a.h) << "  dl' "
                  << rel_d(rows[i].a.l, rows[0].a.l) << "  dk' "
                  << rel_d(rows[i].a.k, rows[0].a.k) << std::defaultfloat
                  << "\n";
      }
      std::cout << std::setprecision(6)
                << "  (mantle diff: relative L2 difference of the mantle "
                   "displacement from the slipping one, rigid modes "
                   "removed; same mesh only)\n";
    }
  }

  // ---- GLVis, on the physical body.
  if (visualization) {
    const std::string keys = examples::DefaultKeys(dim);
    const std::string where = mapped ? " (physical body)" : "";
    auto solid_p = Physical(solid, map);
    auto fluid_p = Physical(fluid, map);
    examples::GLVisWindow("mantle displacement" + where, keys)
        .Send(*solid_p, slip.Displacement());
    examples::GLVisWindow("core displacement (gauge)" + where, keys)
        .Send(*fluid_p, slip.FluidDisplacement());
    // The pressure perturbation, discontinuous L2 of one order less.
    L2_FECollection fec_p(std::max(order - 1, 0), dim);
    SpaceType fes_p(&fluid, &fec_p);
    FieldType p(&fes_p);
    FluidPressureCoefficient p_c(slip.FluidDisplacement(), map, kKappa);
    p.ProjectCoefficient(p_c);
    examples::GLVisWindow("core pressure -kappa div u" + where, keys)
        .Send(*fluid_p, p);
    examples::GLVisWindow(
        "tangential slip on Sigma, gauge-dependent (zero off Sigma)" + where, keys)
        .Send(*solid_p, slip_field);
    if (broken) {
      auto outer_p = Physical(*model.outer, map);
      examples::GLVisWindow("zeta1, mantle and buffer" + where, keys)
          .Send(*outer_p, slip.OuterPotential());
      examples::GLVisWindow("zeta1, core" + where, keys)
          .Send(*fluid_p, slip.FluidPotential());
    } else {
      auto parent_p = Physical(parent, map);
      examples::GLVisWindow("zeta1" + where, keys)
          .Send(*parent_p, slip.Potential());
    }
  }
  return 0;
}
