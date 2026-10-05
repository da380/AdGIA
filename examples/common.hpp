// Non-dimensionalisation helpers shared by the examples built on
// layered_model.hpp.

#pragma once

#include <iostream>

#include "mfem.hpp"

// Scales of length, time and density, and the conversion of the quantities
// the examples use between SI and the units those scales define.
class Nondimensionalisation {
 private:
  mfem::real_t L;    // Length scale [m]
  mfem::real_t T;    // Time scale [s]
  mfem::real_t RHO;  // Density scale [kg/m^3]

 public:
  Nondimensionalisation(mfem::real_t length_scale, mfem::real_t time_scale,
                        mfem::real_t density_scale)
      : L(length_scale), T(time_scale), RHO(density_scale) {}

  mfem::real_t Length() const { return L; }
  mfem::real_t Time() const { return T; }
  mfem::real_t Density() const { return RHO; }

  // Derived scales
  mfem::real_t Pressure() const { return RHO * L * L / (T * T); }  // [Pa]
  mfem::real_t Gravity() const { return L / (T * T); }
  mfem::real_t Potential() const { return L * L / (T * T); }

  mfem::real_t ScaleDensity(mfem::real_t rho) const { return rho / RHO; }
  mfem::real_t ScaleGravityPotential(mfem::real_t phi) const {
    return phi / Potential();
  }
  mfem::real_t ScaleStress(mfem::real_t sigma) const {
    return sigma / Pressure();
  }

  void UnscaleGravityPotential(mfem::GridFunction& phi_gf) const {
    phi_gf *= Potential();
  }
  void UnscaleDisplacement(mfem::GridFunction& u_gf) const { u_gf *= L; }

  void Print() const {
    std::cout << "Scaling parameters:\n";
    std::cout << "  Length scale: " << L << " m\n";
    std::cout << "  Time scale: " << T << " s\n";
    std::cout << "  Density scale: " << RHO << " kg/m^3\n";
    std::cout << "  Gravity potential scale: " << Potential() << " m^2/s^2\n";
  }
};

// Physical constants in SI units.
struct Constants {
  // Newtonian gravitational constant [m^3 kg^-1 s^-2] (CODATA 2018).
  static constexpr mfem::real_t G = 6.6743e-11;
};
