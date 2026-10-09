/*
  Parallel tests for SeaLevelOperator on ParSubMesh surfaces. Run with 1,
  2 and 4 ranks; a standalone MPI program that exits with status 1 if any
  check fails.

  The checks mirror TestSeaLevel.cpp: nodally exact restriction (one hop
  from the body, two hops from the parent), the manufactured sea-level
  change against its closed form, and the half-flooded ocean area. Each
  case runs twice: with the default Cartesian slab partition, and with
  every parent element on rank 0 — so ranks owning no surface (or no
  body) elements exercise every collective path.
*/

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

#include "MixedProblemTestCommon.hpp"
#include "mfem.hpp"
#include "AdGIA.hpp"

using namespace mfem;
using namespace AdGIA;
using namespace self_grav_test;

namespace {

int num_checks = 0;
int num_fails = 0;

double GlobalMax(double v) {
  double g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return g;
}

void Check(double err, double tol, const std::string& what) {
  num_checks++;
  if (!(err <= tol)) {
    num_fails++;
    if (Mpi::Root()) {
      std::cout << "FAIL: " << what << "  (err = " << err << ", tol = " << tol
                << ")\n";
    }
  }
}

constexpr double kA = 0.3, kB = 0.7, kC = 0.2;

double Smooth(const Vector& x) {
  double s = 1.0;
  for (int i = 0; i < x.Size(); i++) {
    s += (i + 1) * x[i] + 0.3 * std::sin(x[i]);
  }
  return s;
}

void RunCase(int dim, int order, bool rank0_partition,
             const std::string& label) {
  Mesh smesh(MeshFile(dim).c_str(), 1, 1);
  std::unique_ptr<ParMesh> pmesh;
  if (rank0_partition) {
    std::vector<int> part(smesh.GetNE(), 0);
    pmesh = std::make_unique<ParMesh>(MPI_COMM_WORLD, smesh, part.data());
  } else {
    int nxyz[3] = {Mpi::WorldSize(), 1, 1};
    int* partitioning = smesh.CartesianPartitioning(nxyz);
    pmesh = std::make_unique<ParMesh>(MPI_COMM_WORLD, smesh, partitioning);
    delete[] partitioning;
  }
  ParSubMesh body(ParSubMesh::CreateFromDomain(*pmesh, BodyMarker(*pmesh)));
  H1_FECollection fec_u(order, dim), fec(order, dim);
  ParFiniteElementSpace fes_u(&body, &fec_u, dim);
  const Array<int> surface = SurfaceMarker(body);

  SeaLevelOperator sea(fes_u, surface);
  Check(sea.IsParallel() ? 0.0 : 1.0, 0.0, label + ": parallel surface");

  // Restriction: one hop and two hops, nodally exact.
  FunctionCoefficient f(Smooth);
  ParGridFunction direct(
      static_cast<ParFiniteElementSpace*>(&sea.SurfaceSpace()));
  direct.ProjectCoefficient(f);
  {
    ParFiniteElementSpace body_s(&body, &fec);
    ParGridFunction on_body(&body_s);
    on_body.ProjectCoefficient(f);
    ParGridFunction restricted(direct);
    sea.Restrict(on_body, restricted);
    restricted -= direct;
    Check(GlobalMax(restricted.Normlinf()), 1e-12,
          label + ": one-hop restriction");

    ParFiniteElementSpace parent_s(pmesh.get(), &fec);
    ParGridFunction on_parent(&parent_s);
    on_parent.ProjectCoefficient(f);
    sea.Restrict(on_parent, restricted);
    restricted -= direct;
    Check(GlobalMax(restricted.Normlinf()), 1e-12,
          label + ": two-hop restriction");
  }

  // Manufactured sea-level change: SL1 = -kB x0 after the mass shift.
  {
    ParGridFunction u(&fes_u);
    VectorFunctionCoefficient uc(dim, [](const Vector& x, Vector& v) {
      v = x;
      v *= kA;
    });
    u.ProjectCoefficient(uc);
    ParFiniteElementSpace parent_s(pmesh.get(), &fec);
    ParGridFunction phi(&parent_s);
    FunctionCoefficient phic([](const Vector& x) { return kB * x[0]; });
    phi.ProjectCoefficient(phic);
    VectorFunctionCoefficient grad_phi0(
        dim, [](const Vector& x, Vector& v) { v = x; });
    ConstantCoefficient psi(kC);
    auto info = sea.SeaLevelChange(u, grad_phi0, phi, &psi);
    FunctionCoefficient exact([](const Vector& x) { return -kB * x[0]; });
    // Nodal comparison (as in the serial test): ComputeMaxError would
    // sample the faceted order-1 sphere at quadrature points and report
    // the O(h^2) geometry error instead.
    ParGridFunction ex(
        static_cast<ParFiniteElementSpace*>(&sea.SurfaceSpace()));
    ex.ProjectCoefficient(exact);
    ex -= sea.SeaLevelChangeField();
    Check(GlobalMax(ex.Normlinf()), 1e-3, label + ": manufactured sea level");
    Check(std::abs(info.uniform - (kA + kC)), 2e-3,
          label + ": uniform shift");
  }

  // The CSV export gathers one row per GLOBAL node on the root.
  {
    ParGridFunction field(
        static_cast<ParFiniteElementSpace*>(&sea.SurfaceSpace()));
    field.ProjectCoefficient(f);
    const std::string path = "sea_level_export_par.csv";
    sea.WriteSurfaceField(field, path);
    double rows = 0.0;
    if (Mpi::Root()) {
      std::ifstream in(path);
      std::string line;
      std::getline(in, line);  // header
      while (std::getline(in, line)) {
        rows += 1.0;
      }
      std::remove(path.c_str());
    }
    MPI_Bcast(&rows, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const double expected =
        static_cast<ParFiniteElementSpace&>(sea.SurfaceSpace())
            .GlobalTrueVSize();
    Check(std::abs(rows - expected), 0.0, label + ": export row count");
  }

  // Half-flooded ocean.
  {
    const double exact_full =
        dim == 2 ? 2.0 * std::numbers::pi : 4.0 * std::numbers::pi;
    Check(std::abs(sea.OceanArea() - exact_full), 1e-2 * exact_full,
          label + ": all-ocean area");
    FunctionCoefficient sl0([](const Vector& x) { return -x[x.Size() - 1]; });
    ConstantCoefficient no_ice(0.0);
    sea.SetInitialState(sl0, no_ice);
    Check(std::abs(sea.OceanArea() - 0.5 * exact_full), 0.05 * exact_full,
          label + ": half-flooded area");
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();

  for (auto dim : {2, 3}) {
    for (auto order : {1, 2}) {
      for (auto rank0 : {false, true}) {
        auto label = "dim=" + std::to_string(dim) +
                     " p=" + std::to_string(order) +
                     (rank0 ? " rank0-partition" : "");
        RunCase(dim, order, rank0, label);
      }
    }
  }

  if (Mpi::Root()) {
    if (num_fails == 0) {
      std::cout << "All " << num_checks << " checks passed on "
                << Mpi::WorldSize() << " ranks.\n";
    } else {
      std::cout << num_fails << " of " << num_checks << " checks FAILED on "
                << Mpi::WorldSize() << " ranks.\n";
    }
  }
  return num_fails == 0 ? 0 : 1;
}
