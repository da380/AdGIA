/*
  Parallel tests for RefineToShoreline (mesh.hpp): the same wavy-polar
  shoreline configuration as the serial TestShorelineRefinement, run on a
  ParMesh with 1, 2 and 4 ranks. A standalone MPI program that exits
  non-zero if any check fails.

  Checks: refinement happens and grows the global mesh; conforming local
  bisection stays valid (orientation clean, uniform refinement composes
  after); the refined band's surface geometry improves past the measured
  grade; snapped nodes sit on the unit sphere exactly. The thresholds are
  the serial measurements' grades (4.1e-4 -> 2.16e-5 at two passes,
  node snap 4.4e-16).
*/

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

#include "AdGIA.hpp"
#include "mfem.hpp"

using namespace mfem;
using namespace AdGIA;

namespace {

int num_fails = 0;

void Check(bool ok, const std::string& what) {
  int local = ok ? 0 : 1;
  int global = 0;
  MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (global) {
    num_fails++;
    if (Mpi::Root()) {
      std::cout << "FAIL: " << what << "\n";
    }
  } else if (Mpi::Root()) {
    std::cout << "ok:   " << what << "\n";
  }
}

double ShorelineColatitude(double phi) {
  return 0.9 + 0.15 * std::sin(3.0 * phi);
}

double Ocean(const Vector& x) {
  const double r = x.Norml2();
  const double th = std::acos(std::min(1.0, std::max(-1.0, x[2] / r)));
  const double q = th / ShorelineColatitude(std::atan2(x[1], x[0]));
  return 1.0 - std::exp(-q * q * q * q);
}

double BandSurfaceDeviation(ParMesh& mesh) {
  double dev = 0.0;
  for (int be = 0; be < mesh.GetNBE(); be++) {
    if (mesh.GetBdrAttribute(be) != 1) {
      continue;
    }
    ElementTransformation* T = mesh.GetBdrElementTransformation(be);
    const IntegrationRule& ir =
        IntRules.Get(mesh.GetBdrElementGeometry(be), 4);
    Vector p;
    double d = 0.0;
    bool in_band = false;
    for (int j = 0; j < ir.GetNPoints(); j++) {
      T->Transform(ir.IntPoint(j), p);
      const double r = p.Norml2();
      d = std::max(d, std::abs(r - 1.0));
      const double th = std::acos(std::min(1.0, std::max(-1.0, p[2] / r)));
      if (std::abs(th - ShorelineColatitude(std::atan2(p[1], p[0]))) < 0.1) {
        in_band = true;
      }
    }
    if (in_band) {
      dev = std::max(dev, d);
    }
  }
  double gdev = 0.0;
  MPI_Allreduce(&dev, &gdev, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return gdev;
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();

  Mesh smesh("../data/ball_buffer_coarse_3d.msh", 1, 1);
  ParMesh mesh(MPI_COMM_WORLD, smesh);
  smesh.Clear();

  const auto ne0 = mesh.GetGlobalNE();
  const double dev0 = BandSurfaceDeviation(mesh);
  FunctionCoefficient ocean(Ocean);
  ShorelineRefinementOptions opt;
  opt.levels = 2;
  const int refined = RefineToShoreline(mesh, ocean, opt);

  Check(refined > 0, "elements were refined (global count)");
  Check(mesh.GetGlobalNE() > ne0, "global element count grew");
  Check(mesh.Conforming(), "mesh stays conforming");
  Check(mesh.CheckElementOrientation(false) == 0, "no inverted elements");
  // Short-circuit before the collective is safe: dev0 is globally reduced,
  // so every rank takes the same branch.
  Check(dev0 > 1e-4 && BandSurfaceDeviation(mesh) < 5e-5,
        "band surface geometry improved past the measured grade");

  const GridFunction* nodes = mesh.GetNodes();
  double node_dev = 0.0;
  if (nodes) {
    const FiniteElementSpace& nfes = *nodes->FESpace();
    for (int i = 0; i < nfes.GetNDofs(); i++) {
      Vector p(3);
      for (int c = 0; c < 3; c++) {
        p[c] = (*nodes)(nfes.DofToVDof(i, c));
      }
      const double r = p.Norml2();
      if (std::abs(r - 1.0) < 5e-3) {
        node_dev = std::max(node_dev, std::abs(r - 1.0));
      }
    }
  }
  Check(nodes != nullptr && node_dev < 1e-14,
        "snapped nodes sit on the unit sphere exactly");

  mesh.UniformRefinement();
  Check(mesh.CheckElementOrientation(false) == 0,
        "uniform refinement composes after");

  if (Mpi::Root()) {
    std::cout << (num_fails ? "FAILED" : "tests passed") << "\n";
  }
  return num_fails ? 1 : 0;
}
