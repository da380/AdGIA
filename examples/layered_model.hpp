// ============================================================================
// layered_model.hpp
//
// The PREM-like layered Earth models shared by the elastogravity_layered and
// self_gravitating_relaxation examples.
//
// Three models, recognised from the mesh's number of domain attributes
// (SetModel):
//   uniform     (attributes 1 body, 2 buffer; boundary attributes 1 surface,
//                2 outer), a solid of constant density and moduli whatever
//                its shape
//   two-layer   (attributes 1 fluid core, 2 mantle, 3 buffer;
//                boundary attributes 1 CMB, 2 surface, 3 outer)
//   three-layer (attributes 1 inner core, 2 fluid outer core, 3 mantle,
//                4 buffer; boundary attributes 1 ICB, 2 CMB, 3 surface,
//                4 outer)
// the layered ones with radii 1230/6371 (ICB), 3483/6371 (CMB) and 1
// (surface), as produced by meshes/layered_earth.py.
//
// Profiles of the layered models are piecewise linear in radius
// (dimensional values below), with a degree-2 polar and an azimuthal
// perturbation of the mantle moduli and of the load.
// Non-dimensionalisation: L = 6371 km, rho = 5000 kg/m^3, T = 1/sqrt(G rho),
// so that the non-dimensional gravitational constant is 1.
// ============================================================================

#pragma once

#include <cmath>

#include "common.hpp"

namespace layered {

using namespace mfem;

inline Nondimensionalisation ND(6371e3, 1.0 / std::sqrt(Constants::G * 5000.0),
                                5000.0);

constexpr real_t kRIcb = 1230.0 / 6371.0;
constexpr real_t kRCmb = 3483.0 / 6371.0;
constexpr real_t kRSurface = 1.0;

// Set from the mesh (SetModel) before the coefficients are used.
inline bool uniform = false;
inline bool inner_core = false;
// Density [kg/m^3] and moduli [Pa] of the uniform model (-rho, -kappa, -mu).
inline real_t uniform_density = 5500.0;
inline real_t uniform_bulk_modulus = 300e9;
inline real_t uniform_shear_modulus = 150e9;
// The surface-load amplitude factor (-load) and whether the fluid is
// treated as solid (-solid-core).
inline real_t load_factor = 1.0;
inline bool solid_core = false;

// Choose the model from the mesh's number of domain attributes; false when
// the mesh is none of the three.
inline bool SetModel(const Mesh& mesh) {
  const int n = mesh.attributes.Max();
  uniform = n == 2;
  inner_core = n == 4;
  return n >= 2 && n <= 4;
}

inline const char* ModelName() {
  return uniform ? "Uniform" : inner_core ? "Three-layer" : "Two-layer";
}

inline real_t Linear(real_t a, real_t b, real_t s) { return a + (b - a) * s; }

inline real_t Azimuthal(const Vector& x) {
  return x.Size() == 2 ? 0.0 : std::sin(2.0 * std::atan2(x[1], x[0]));
}

inline real_t Polar(const Vector& x) {
  const real_t r = x.Norml2();
  if (r == 0.0) {
    return 0.0;
  }
  const real_t theta = std::acos((x.Size() == 2 ? x[1] : x[2]) / r);
  return 0.015 * (1.0 + std::cos(2.0 * theta));
}

inline real_t MantlePerturbation(const Vector& x) {
  return (1.0 + Polar(x)) * (1.0 + 0.05 * Azimuthal(x));
}

// Density of the whole body (kg/m^3, dimensional), by radius.
inline real_t DensityDim(real_t r) {
  if (uniform) {
    return uniform_density;
  }
  if (r > kRSurface) {
    return 0.0;
  }
  if (inner_core) {
    if (r < kRIcb) {
      return Linear(13.1e3, 12.8e3, r / kRIcb);
    }
    if (r < kRCmb) {
      return Linear(12.2e3, 9.9e3, (r - kRIcb) / (kRCmb - kRIcb));
    }
    return Linear(5.6e3, 3.3e3, (r - kRCmb) / (kRSurface - kRCmb));
  }
  if (r < kRCmb) {
    return Linear(13.4e3, 9.9e3, r / kRCmb);
  }
  return Linear(5.6e3, 3.3e3, (r - kRCmb) / (kRSurface - kRCmb));
}

// rho on the solid (and, evaluated on the parent, everywhere).
inline real_t Density(const Vector& x) {
  return ND.ScaleDensity(DensityDim(x.Norml2()));
}

// rho_F of the fluid, clamped to the fluid's radii so that it is right on
// the interfaces as seen from the solid.
inline real_t FluidDensity(const Vector& x) {
  real_t r = x.Norml2();
  const real_t r_in = inner_core ? kRIcb : 0.0;
  r = std::min(std::max(r, r_in), kRCmb);
  return ND.ScaleDensity(DensityDim(r));
}

inline real_t ShearModulus(const Vector& x) {
  const real_t r = x.Norml2();
  if (uniform) {
    return ND.ScaleStress(uniform_shear_modulus);
  }
  if (r > kRSurface) {
    return 0.0;
  }
  real_t mu;
  if (inner_core && r < kRIcb) {
    mu = Linear(176e9, 156e9, r / kRIcb);
  } else if (r < kRCmb) {
    mu = solid_core ? 100e9 : 0.0;
  } else if (inner_core) {
    mu = Linear(294e9, 68e9, (r - kRCmb) / (kRSurface - kRCmb)) *
         MantlePerturbation(x);
  } else {
    mu = Linear(280e9, 70e9, (r - kRCmb) / (kRSurface - kRCmb)) *
         MantlePerturbation(x);
  }
  return ND.ScaleStress(mu);
}

inline real_t Lame(const Vector& x) {
  const real_t r = x.Norml2();
  if (uniform) {
    return ND.ScaleStress(uniform_bulk_modulus -
                          2.0 * uniform_shear_modulus / x.Size());
  }
  if (r > kRSurface) {
    return 0.0;
  }
  real_t lambda;
  if (inner_core && r < kRIcb) {
    lambda = Linear(1.31e12, 1.24e12, r / kRIcb);
  } else if (r < kRCmb) {
    lambda = solid_core ? 300e9 : 0.0;
  } else if (inner_core) {
    lambda = Linear(461e9, 86e9, (r - kRCmb) / (kRSurface - kRCmb)) *
             MantlePerturbation(x);
  } else {
    const real_t s = (r - kRCmb) / (kRSurface - kRCmb);
    lambda = (Linear(650e9, 130e9, s) - 2.0 * Linear(280e9, 70e9, s) / 3.0) *
             MantlePerturbation(x);
  }
  return ND.ScaleStress(lambda);
}

// Bulk modulus of the library's split form, kappa = lambda + 2 mu / dim
// (this reproduces ElasticityIntegrator(lambda, mu) in either dimension).
inline real_t BulkModulus(const Vector& x) {
  return Lame(x) + 2.0 * ShearModulus(x) / x.Size();
}

// Surface mass load sigma (non-dimensional; positive = mass added): a
// degree-2 pressure pattern between 0.1 and 1 MPa (times load_factor) with
// a 20% azimuthal perturbation, converted to mass per area with the
// gravity scale.
inline real_t SurfaceLoad(const Vector& x) {
  const real_t r = x.Norml2();
  if (r == 0.0) {
    return 0.0;
  }
  const real_t theta = std::acos((x.Size() == 2 ? x[1] : x[2]) / r);
  const real_t p_high = 10e6, p_low = 1e6;
  const real_t pressure = (0.5 * (p_low + p_high) +
                           0.5 * (p_high - p_low) * std::cos(2.0 * theta)) *
                          (1.0 + 0.2 * Azimuthal(x)) * 0.1 * load_factor;
  const real_t sigma_dim = pressure / ND.Gravity();
  return sigma_dim / (ND.Density() * ND.Length());
}

// Degree-2 tidal potential psi = A (r/a)^2 P_2(cos theta), amplitude A in
// m^2/s^2 (dimensional) set through tidal_amplitude.
inline real_t tidal_amplitude = 0.0;
inline real_t TidalPotential(const Vector& x) {
  const real_t r2 = x * x;
  const real_t z = x.Size() == 2 ? x[1] : x[2];
  return ND.ScaleGravityPotential(tidal_amplitude) * 0.5 * (3.0 * z * z - r2);
}

// The mean over radius of the mantle's shear modulus [Pa].
inline real_t MantleMeanShearModulusDim() {
  return inner_core ? 0.5 * (294e9 + 68e9) : 0.5 * (280e9 + 70e9);
}

// Markers. The solid SubMesh inherits the interface and surface attributes.
inline Array<int> SolidAttributes() {
  if (uniform) {
    return Array<int>({1});
  }
  return inner_core ? (solid_core ? Array<int>({1, 2, 3}) : Array<int>({1, 3}))
                    : (solid_core ? Array<int>({1, 2}) : Array<int>({2}));
}

inline int FluidAttribute() { return inner_core ? 2 : 1; }
inline int InnerCoreAttribute() { return 1; }
inline int MantleAttribute() { return inner_core ? 3 : 2; }

inline Array<int> SurfaceMarker(Mesh& solid) {
  Array<int> m(solid.bdr_attributes.Max());
  m = 0;
  m[(uniform ? 1 : inner_core ? 3 : 2) - 1] = 1;
  return m;
}

inline Array<int> InterfaceMarker(Mesh& solid) {
  Array<int> m(solid.bdr_attributes.Max());
  m = 0;
  if (inner_core) {
    m[0] = 1;  // ICB
    m[1] = 1;  // CMB
  } else {
    m[0] = 1;  // CMB
  }
  return m;
}

inline real_t L2Norm(const GridFunction& u) {
  const int vdim = u.FESpace()->GetVDim();
  Vector zero(vdim);
  zero = 0.0;
  VectorConstantCoefficient z(zero);
  ConstantCoefficient zs(0.0);
  return vdim == 1 ? u.ComputeL2Error(zs) : u.ComputeL2Error(z);
}

}  // namespace layered
