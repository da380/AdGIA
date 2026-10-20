// ============================================================================
// kernel_sampling.hpp
//
// Sampling of a scalar volume field on a ball (an adjoint viscosity kernel,
// say) for the map chain. Both examples that export such kernels
// (gia_kernels.cpp, ice_age_loading.cpp) gather the field to one file pair
// (mesh + .gf), reload it serially on the root rank and call these functions
// there, so a serial run and an MPI run write the same files in the same
// format:
//
//   WriteDepthShells       the field on spherical shells r = const, at the
//                          nodes of the ball's boundary surface
//                          (<prefix><r>.csv with the columns x,y,z,value:
//                          the format of SeaLevelOperator::WriteSurfaceField,
//                          which <build>/earth_data/plot_fingerprint_kernel
//                          reads);
//   WriteGreatCircleSlice  the field on the great-circle plane through a
//                          given direction and the +z pole, on a regular
//                          (angle, radius) grid (columns alpha_deg,r,value).
//
// Conventions: the mesh is a SERIAL ball in units of its outer radius; the
// highest boundary attribute of the mesh is its outer surface; points the
// mesh does not contain (a shell outside the body, a fluid-core gap that is
// not part of the mesh) are written as 0. These are plain functions of a
// serial Mesh and GridFunction, with no MPI.
//
// The CSVs are for quick looks; the interchange form of a kernel is the
// planetmodel model file of AdGIA/model_export.hpp, which both examples
// write beside them (visc_kernel.nc).
// ============================================================================

#pragma once

#include <cmath>
#include <fstream>
#include <iomanip>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

#include "mfem.hpp"

namespace examples {

// The field on n_shells shells between radii r_lo and r_hi (a single shell
// sits at r_hi). `order` is the order of the surface node set (use the
// finite element order of the run). Returns the written file names, shell
// by shell in increasing radius.
inline std::vector<std::string> WriteDepthShells(
    mfem::Mesh& ball, const mfem::GridFunction& field, int order,
    int n_shells, double r_lo, double r_hi,
    const std::string& prefix = "visc_kernel_r") {
  std::vector<std::string> names;
  if (n_shells <= 0) {
    return names;
  }
  // The surface node set: the nodes of the ball's outer boundary.
  mfem::Array<int> surface_attr({ball.bdr_attributes.Max()});
  auto surface = mfem::SubMesh::CreateFromBoundary(ball, surface_attr);
  mfem::H1_FECollection fec(order, 2);
  mfem::FiniteElementSpace sfes(&surface, &fec);
  mfem::FiniteElementSpace vfes(&surface, &fec, 3);
  mfem::GridFunction xyz(&vfes);
  mfem::VectorFunctionCoefficient identity(
      3, [](const mfem::Vector& x, mfem::Vector& v) { v = x; });
  xyz.ProjectCoefficient(identity);
  const int nd = sfes.GetVSize();

  for (int k = 0; k < n_shells; k++) {
    const double rk =
        n_shells == 1 ? r_hi : r_lo + (r_hi - r_lo) * k / (n_shells - 1.0);
    mfem::DenseMatrix pts(3, nd);
    for (int i = 0; i < nd; i++) {
      mfem::Vector dir(3);
      dir[0] = xyz[i];
      dir[1] = xyz[i + nd];
      dir[2] = xyz[i + 2 * nd];
      dir /= dir.Norml2();
      for (int c = 0; c < 3; c++) {
        pts(c, i) = rk * dir[c];
      }
    }
    mfem::Array<int> elems;
    mfem::Array<mfem::IntegrationPoint> ips;
    ball.FindPoints(pts, elems, ips, false);
    std::ostringstream name;
    name << prefix << std::fixed << std::setprecision(2) << rk << ".csv";
    std::ofstream out(name.str());
    out << "x,y,z,value\n";
    out.precision(16);
    for (int i = 0; i < nd; i++) {
      out << xyz[i] << "," << xyz[i + nd] << "," << xyz[i + 2 * nd] << ","
          << (elems[i] >= 0 ? field.GetValue(elems[i], ips[i]) : 0.0)
          << "\n";
    }
    names.push_back(name.str());
  }
  return names;
}

// The field on the great-circle plane through the unit direction e1 and the
// +z pole: n_alpha angles alpha in [-180, 180] degrees measured from e1
// towards the in-plane northward direction, and n_r radii in [r_in, r_out].
// Returns the file name.
inline std::string WriteGreatCircleSlice(
    mfem::Mesh& ball, const mfem::GridFunction& field, const mfem::Vector& e1,
    double r_in, double r_out, int n_alpha = 181, int n_r = 48,
    const std::string& file = "visc_kernel_slice.csv") {
  mfem::Vector ez(3), e2(3);
  ez = 0.0;
  ez[2] = 1.0;
  e2 = ez;
  e2.Add(-(ez * e1), e1);
  e2 /= e2.Norml2();
  auto radius = [&](int ir) {
    return n_r == 1 ? r_out : r_in + (r_out - r_in) * ir / (n_r - 1.0);
  };
  mfem::DenseMatrix pts(3, n_alpha * n_r);
  for (int ia = 0; ia < n_alpha; ia++) {
    const double alpha =
        -std::numbers::pi + 2.0 * std::numbers::pi * ia / (n_alpha - 1);
    for (int ir = 0; ir < n_r; ir++) {
      const int col = ia * n_r + ir;
      for (int c = 0; c < 3; c++) {
        pts(c, col) =
            radius(ir) * (std::cos(alpha) * e1[c] + std::sin(alpha) * e2[c]);
      }
    }
  }
  mfem::Array<int> elems;
  mfem::Array<mfem::IntegrationPoint> ips;
  ball.FindPoints(pts, elems, ips, false);
  std::ofstream slice(file);
  slice << "alpha_deg,r,value\n";
  slice.precision(16);
  for (int ia = 0; ia < n_alpha; ia++) {
    const double alpha_deg = -180.0 + 360.0 * ia / (n_alpha - 1);
    for (int ir = 0; ir < n_r; ir++) {
      const int col = ia * n_r + ir;
      slice << alpha_deg << "," << radius(ir) << ","
            << (elems[col] >= 0 ? field.GetValue(elems[col], ips[col]) : 0.0)
            << "\n";
    }
  }
  return file;
}

}  // namespace examples
