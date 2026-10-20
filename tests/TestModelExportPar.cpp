/*
  Parallel tests for model_export.hpp: the gather-then-write pattern of
  the kernel examples. Run with 1, 2 and 4 ranks; a standalone MPI
  program that exits with status 1 if any check fails.

  A smooth analytic "kernel" lives as a ParGridFunction on the
  partitioned three-layer-Earth mesh; it is gathered to the root
  (GetSerialGridFunction, the way gia_kernels.cpp gathers for the CSV
  samplers), the root writes the model file and reads it back, and the
  pairing integral against a test perturbation is checked against the
  field's own — the same number on every rank count, since the gathered
  field is the same.
*/

#include <mpi.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <numbers>
#include <string>

#include "mfem.hpp"
#include "AdGIA.hpp"
#include "planetmodel/error.hpp"
#include "planetmodel/modelfile.hpp"

using namespace mfem;
using namespace AdGIA;

namespace {

int num_checks = 0;
int num_fails = 0;

void Check(bool ok, const std::string& what) {
  num_checks++;
  if (!ok) {
    num_fails++;
    if (Mpi::Root()) {
      std::cout << "FAIL: " << what << "\n";
    }
  }
}

double KernelExact(const Vector& x) {
  const double r = x.Norml2();
  return std::sin(std::numbers::pi * r) *
         (1.0 + 0.3 * x[2] / std::max(r, 1e-12));
}

}  // namespace

int main(int argc, char* argv[]) try {
  Mpi::Init(argc, argv);

  const planetmodel::MeshManifest manifest("../data/earth_innercore_3d.json");
  Mesh serial = LoadMesh(manifest);
  ParMesh mesh(MPI_COMM_WORLD, serial);

  H1_FECollection fec(2, mesh.Dimension());
  ParFiniteElementSpace fes(&mesh, &fec);
  ParGridFunction kernel(&fes);
  FunctionCoefficient kernel_c(
      [](const Vector& x) { return KernelExact(x); });
  kernel.ProjectCoefficient(kernel_c);

  // The gathered pair: GetSerialGridFunction lives on the RECONSTRUCTED
  // serial mesh (GetSerialMesh's ordering), not the one the ParMesh was
  // partitioned from.
  Mesh gathered_mesh = mesh.GetSerialMesh(0);
  const GridFunction gathered = kernel.GetSerialGridFunction(0, gathered_mesh);

  double pairing_error = 0.0;
  if (Mpi::Root()) {
    const std::filesystem::path dir =
        "model_export_par_np" + std::to_string(Mpi::WorldSize());
    std::filesystem::remove_all(dir);
    std::filesystem::create_directory(dir);
    const auto path = dir / "kernel.nc";
    WriteKernelModelFile(path, manifest,
                         {{.name = "K_eta",
                           .field = &gathered,
                           .dimensions = {.mass = -1, .length = -1,
                                          .time = 1},
                           .meaning = "the gathered synthetic kernel"}},
                         {.radial_size = 0.1, .lmax = 16});
    const auto file = planetmodel::ReadModelFile(path);
    CheckModelMatchesManifest(file, manifest);
    ModelScalarCoefficient read_back(file, "K_eta");
    FunctionCoefficient perturbation_c(
        [](const Vector& x) { return 1.0 + x[0] * x[1]; });
    Vector x;
    double pairing_file = 0.0, pairing_field = 0.0;
    for (int e = 0; e < gathered_mesh.GetNE(); e++) {
      if (!read_back.HasLayer(gathered_mesh.GetAttribute(e))) {
        continue;
      }
      ElementTransformation* T = gathered_mesh.GetElementTransformation(e);
      const IntegrationRule& ir = IntRules.Get(T->GetGeometryType(), 4);
      for (int q = 0; q < ir.GetNPoints(); q++) {
        const IntegrationPoint& ip = ir.IntPoint(q);
        T->SetIntPoint(&ip);
        const double w = ip.weight * T->Weight();
        const double dm = perturbation_c.Eval(*T, ip);
        pairing_file += w * read_back.Eval(*T, ip) * dm;
        pairing_field += w * gathered.GetValue(e, ip) * dm;
      }
    }
    pairing_error = std::abs(pairing_file - pairing_field) /
                    std::abs(pairing_field);
    std::filesystem::remove_all(dir);
  }
  MPI_Bcast(&pairing_error, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  Check(pairing_error < 2e-2, "round-trip pairing on the gathered kernel");

  if (Mpi::Root()) {
    std::cout << (num_fails == 0 ? "PASS" : "FAIL") << ": " << num_checks
              << " checks, " << num_fails << " failures (pairing error "
              << pairing_error << ")\n";
  }
  return num_fails == 0 ? 0 : 1;
} catch (const planetmodel::Error& e) {
  std::cerr << e.what() << "\n";
  return 1;
}
