#pragma once

/*
  Helpers shared by the self-gravitating tests (serial gtests and MPI
  programs): the canned two-layer meshes in ../data, the attribute
  conventions of those meshes, and a degree-2 surface load.

  Both meshes come from the scripts in meshes/: domain attribute 1 is the
  body (radius 1), attribute 2 the buffer shell; the body surface carries
  boundary attribute 1 and the outer sphere boundary attribute 2. On the
  body SubMesh the surface is the only boundary and keeps attribute 1.
*/

#include <cmath>
#include <memory>
#include <string>

#include "mfem.hpp"
#include "AdGIA.hpp"

namespace self_grav_test {

using namespace mfem;
using namespace AdGIA;

inline std::string MeshFile(int dim) {
  return dim == 2 ? "../data/elastogravity_2d.msh"
                  : "../data/coupled_poisson.msh";
}

// Non-dimensional constants of the tests: unit radius, G such that the
// coupling is strong enough to matter (rho g R / mu = O(1)).
constexpr double kG = 0.05;
constexpr double kRho = 1.0;
constexpr double kKappa = 1.0;
constexpr double kMu = 0.5;
constexpr int kDtNDegree = 12;

// Body marker on the parent mesh.
inline Array<int> BodyMarker(Mesh& parent) {
  Array<int> m(parent.attributes.Max());
  m = 0;
  m[0] = 1;
  return m;
}

// Surface marker on the body SubMesh (its largest boundary attribute).
inline Array<int> SurfaceMarker(Mesh& body) {
  // On the three-layer solid SubMesh the surface is attribute 3, the
  // largest, as on the single-body SubMesh.
  Array<int> m(body.bdr_attributes.Max());
  m = 0;
  m[body.bdr_attributes.Max() - 1] = 1;
  return m;
}

// Surface mass load: a degree-2 pattern in the polar angle, scaled by
// (1 + t) so that time dependence can be checked.
inline double SurfaceLoad(const Vector& x, double t) {
  const double r = x.Norml2();
  if (r == 0.0) {
    return 0.0;
  }
  const double c = (x.Size() == 2 ? x[1] : x[2]) / r;
  return 0.02 * (1.0 + 3.0 * c * c) * (1.0 + t);
}

// L2 norm of a (Par)GridFunction; global in parallel.
inline double L2Norm(const GridFunction& u) {
  const int vdim = u.FESpace()->GetVDim();
  if (vdim == 1) {
    ConstantCoefficient z(0.0);
    return u.ComputeL2Error(z);
  }
  Vector zero(vdim);
  zero = 0.0;
  VectorConstantCoefficient z(zero);
  return u.ComputeL2Error(z);
}

// --- three-layer models with a fluid outer core -----------------------------
//
// data/elastogravity_three_layer_{2d,3d}.msh: attributes 1 inner core,
// 2 outer core (fluid), 3 mantle, 4 buffer; boundary attributes 1 ICB,
// 2 CMB, 3 surface, 4 outer. Radii 0.1931 / 0.5467 / 1 / 1.2 (Earth's
// ratios). The solid SubMesh {1, 3} is disconnected and inherits the
// boundary attributes 1, 2, 3.
//
// data/elastogravity_two_layer_2d.msh: attributes 1 fluid core, 2 mantle,
// 3 buffer; boundary attributes 1 CMB, 2 surface, 3 outer.
//
// Non-dimensional densities: inner core 1.3, fluid from 1.2 (ICB) to 1.1
// (CMB), mantle 1.0; with G = 0.05 the fluid mass term is comfortably
// inside the positivity margin of the potential block (k R_CMB ~ 0.7
// against pi/2; see doc/self_gravitation.md, "Solvers", the paragraph
// "Definiteness of the potential block").

constexpr double kRIcb = 0.1931;
constexpr double kRCmb = 0.5467;

inline std::string ThreeLayerMeshFile(int dim) {
  return dim == 2 ? "../data/elastogravity_three_layer_2d.msh"
                  : "../data/elastogravity_three_layer_3d.msh";
}

inline std::string TwoLayerMeshFile(int dim) {
  return dim == 2 ? "../data/elastogravity_two_layer_2d.msh"
                  : "../data/elastogravity_two_layer_3d.msh";
}

inline double SolidDensity(const Vector& x) {
  return x.Norml2() < 0.5 * (kRIcb + kRCmb) ? 1.3 : 1.0;
}

inline double FluidDensity(const Vector& x) {
  const double r = x.Norml2();
  return 1.2 - 0.1 * (r - kRIcb) / (kRCmb - kRIcb);
}

// A fluid whose density gradient is steep enough to make the potential
// block indefinite (k R_CMB ~ 2 against pi/2).
inline double SteepFluidDensity(const Vector& x) {
  const double r = x.Norml2();
  return 1.3 - 0.8 * (r - kRIcb) / (kRCmb - kRIcb);
}

// A degree-2 tidal potential, scaled by (1 + t).
inline double TidalPotential(const Vector& x, double t) {
  const double last = x.Size() == 2 ? x[1] : x[2];
  return 0.01 * (3.0 * last * last - (x * x)) * (1.0 + t);
}

inline void TidalGravity(const Vector& x, double t, Vector& g) {
  g = x;
  g *= -2.0;
  const int last = x.Size() - 1;
  g[last] += 6.0 * x[last];
  g *= 0.01 * (1.0 + t);
}

// The fluid region of the three-layer model, with markers sized to the
// solid SubMesh; density evaluated on the parent and on the interfaces.
inline FluidRegion OuterCore(Mesh& solid, Coefficient& rho_f,
                             Coefficient* rho_f_prime = nullptr) {
  FluidRegion f;
  f.attributes = Array<int>({2});
  f.density = &rho_f;
  f.density_gradient = rho_f_prime;
  f.interface_marker.SetSize(solid.bdr_attributes.Max());
  f.interface_marker = 0;
  f.interface_marker[0] = 1;  // ICB
  f.interface_marker[1] = 1;  // CMB
  return f;
}

// --- gauged-fluid model on the three-layer meshes ---------------------------
//
// The whole body (all three layers) carries the displacement; the outer
// core has the Adams-Williamson bulk modulus kappa = rho^2 |grad Phi0| /
// |drho/dr| built on the problem's own discrete background gravity
// (N^2 = 0, so the gauged formulation must agree with Dahlen's away from
// degree 0) and zero shear.

constexpr double kDrhoDr = 0.1 / (kRCmb - kRIcb);

inline double FullDensity(const Vector& x) {
  const double r = x.Norml2();
  if (r < 0.5 * (kRIcb + kRCmb)) {
    return r < kRIcb ? 1.3 : FluidDensity(x);
  }
  return r < kRCmb ? FluidDensity(x) : 1.0;
}

// kappa on the body SubMesh: Adams-Williamson in the fluid (attribute 2),
// kKappa in the solid. The background gravity is wired after the problem
// is constructed (it is evaluated only at operator assembly).
struct AWBulkModulus : public Coefficient {
  VectorCoefficient* g = nullptr;

  double Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    if (T.Attribute != 2) {
      return kKappa;
    }
    MFEM_ASSERT(g, "background gravity not wired");
    Vector gv(T.GetSpaceDim()), x(T.GetSpaceDim());
    g->Eval(gv, T, ip);
    T.Transform(ip, x);
    const double rho = FluidDensity(x);
    return rho * rho * gv.Norml2() / kDrhoDr;
  }
};

// Shear: zero in the fluid, kMu in the solid layers.
inline double GaugedShearModulus(const Vector& x) {
  const double r = x.Norml2();
  return (r > kRIcb && r < kRCmb) ? 0.0 : kMu;
}

// SurfaceLoad minus its spherical mean: no degree-0 content (at degree 0
// the gauged and Dahlen fluids differ by design).
inline double ZeroMeanSurfaceLoad(const Vector& x, double t) {
  const double r = x.Norml2();
  if (r == 0.0) {
    return 0.0;
  }
  const double c = (x.Size() == 2 ? x[1] : x[2]) / r;
  const double mean_c2 = x.Size() == 2 ? 0.5 : 1.0 / 3.0;
  return 0.02 * 3.0 * (c * c - mean_c2) * (1.0 + t);
}

}  // namespace self_grav_test
