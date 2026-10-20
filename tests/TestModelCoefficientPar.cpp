/*
  Parallel tests for model_coefficient.hpp: the coefficients of a model
  file evaluate on a partitioned mesh exactly as in serial. Run with 1, 2
  and 4 ranks; a standalone MPI program that exits with status 1 if any
  check fails.

  The root writes the model file of TestModelCoefficient (a polynomial
  radial density on every body layer of the three-layer Earth, a lateral
  viscosity on the mantle); every rank reads it (the fields are immutable
  and evaluation is local, nothing collective), and the quadrature-point
  errors against the closed forms are reduced over the ranks.
*/

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "AdGIA.hpp"
#include "planetmodel/error.hpp"
#include "planetmodel/modelfile.hpp"

using namespace mfem;
using namespace AdGIA;

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

double RhoExact(double r) { return 2.0 - r * r; }

double ViscosityExact(double r, double theta, double phi) {
  return r * (1.0 + 0.1 * std::sin(theta) * std::cos(phi));
}

}  // namespace

int main(int argc, char* argv[]) try {
  Mpi::Init(argc, argv);

  const planetmodel::MeshManifest manifest("../data/earth_innercore_3d.json");
  const std::filesystem::path dir =
      "model_coefficient_par_np" + std::to_string(Mpi::WorldSize());
  const auto model_path = dir / "test.model.nc";
  if (Mpi::Root()) {
    std::filesystem::remove_all(dir);
    std::filesystem::create_directory(dir);
    std::vector<double> boundaries{manifest.Layers().front().r_inner};
    for (const int a : manifest.BodyAttributes()) {
      boundaries.push_back(
          manifest.Layers()[static_cast<std::size_t>(a - 1)].r_outer);
    }
    auto file = planetmodel::ModelFile(planetmodel::Skeleton(boundaries));
    const int last = static_cast<int>(file.GetSkeleton().NLayers()) - 1;
    file.AddPart(0, last,
                 planetmodel::ElementEdges(file.GetSkeleton(), 0, last, 0.1),
                 5, planetmodel::GaussLegendre(32));
    const auto rho =
        planetmodel::LayerSource::Radial([](double r) { return RhoExact(r); });
    file.AddField("rho", {{0, rho}, {1, rho}, {2, rho}});
    file.AddField("viscosity",
                  {{last, planetmodel::LayerSource::Lateral(ViscosityExact)}},
                  {.lateral = planetmodel::LateralRule::Cubic});
    planetmodel::WriteModelFile(model_path, file);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  const auto file = planetmodel::ReadModelFile(model_path);
  ModelScalarCoefficient rho(file, "rho");
  ModelScalarCoefficient viscosity(file, "viscosity");

  Mesh serial = LoadMesh(manifest);
  ParMesh mesh(MPI_COMM_WORLD, serial);
  serial.Clear();

  std::vector<std::pair<double, double>> intervals;
  for (const auto& layer : manifest.Layers()) {
    intervals.emplace_back(layer.r_inner, layer.r_outer);
  }

  Vector x;
  double rho_error = 0.0, viscosity_error = 0.0;
  for (int e = 0; e < mesh.GetNE(); e++) {
    const int attribute = mesh.GetAttribute(e);
    if (!rho.HasLayer(attribute)) {
      continue;
    }
    ElementTransformation* T = mesh.GetElementTransformation(e);
    const IntegrationRule& ir = IntRules.Get(T->GetGeometryType(), 4);
    for (int q = 0; q < ir.GetNPoints(); q++) {
      const IntegrationPoint& ip = ir.IntPoint(q);
      T->SetIntPoint(&ip);
      T->Transform(ip, x);
      const auto& [r_inner, r_outer] =
          intervals[static_cast<std::size_t>(attribute - 1)];
      const double r_mesh = x.Norml2();
      const double r = std::clamp(r_mesh, r_inner, r_outer);
      rho_error =
          std::max(rho_error, std::abs(rho.Eval(*T, ip) - RhoExact(r)));
      if (viscosity.HasLayer(attribute)) {
        const double theta = std::acos(std::clamp(x[2] / r_mesh, -1.0, 1.0));
        const double phi = std::atan2(x[1], x[0]);
        viscosity_error = std::max(
            viscosity_error, std::abs(viscosity.Eval(*T, ip) -
                                      ViscosityExact(r, theta, phi)));
      }
    }
  }
  Check(GlobalMax(rho_error), 1e-12, "radial field exact on the partition");
  Check(GlobalMax(viscosity_error), 1e-3,
        "lateral field interpolates on the partition");

  MPI_Barrier(MPI_COMM_WORLD);
  if (Mpi::Root()) {
    std::filesystem::remove_all(dir);
    std::cout << (num_fails == 0 ? "PASS" : "FAIL") << ": " << num_checks
              << " checks, " << num_fails << " failures\n";
  }
  return num_fails == 0 ? 0 : 1;
} catch (const planetmodel::Error& e) {
  std::cerr << e.what() << "\n";
  return 1;
}
