#include <cmath>
#include <filesystem>
#include <numbers>

#include "TestCommon.hpp"
#include "planetmodel/error.hpp"
#include "planetmodel/modelfile.hpp"

/*
  Tests for model_export.hpp: fields on a mesh written as a model file.

  A smooth analytic "kernel" is projected onto an H1 field on the
  three-layer-Earth mesh of the build and written out
  (WriteKernelModelFile), then read back through ModelScalarCoefficient.

  - The file passes CheckModelMatchesManifest (the skeleton is the
    manifest's body layers) and carries the stated dimensions, the layer
    and interface names.
  - The round trip reproduces the field pointwise within the file's
    sampling error, and the integral of the kernel against a smooth test
    perturbation — the pairing a kernel exists for — matches the field's
    own integral much more closely.
  - A field naming a shell is refused.
*/

namespace {

const char* const kSourceManifest = "../data/earth_innercore_3d.json";

double KernelExact(const Vector& x) {
  const double r = x.Norml2();
  return std::sin(std::numbers::pi * r) *
         (1.0 + 0.3 * x[2] / std::max(r, 1e-12));
}

double PerturbationExact(const Vector& x) {
  return 1.0 + x[0] * x[1];
}

struct Files {
  std::filesystem::path dir;
  Files() {
    const auto* info = testing::UnitTest::GetInstance()->current_test_info();
    dir = std::string("model_export_test_") + info->test_suite_name() + "_" +
          info->name();
    std::filesystem::remove_all(dir);
    std::filesystem::create_directory(dir);
  }
  ~Files() { std::filesystem::remove_all(dir); }
};

}  // namespace

TEST(ModelExport, RoundTripsAKernel) {
  const Files files;
  const planetmodel::MeshManifest manifest(kSourceManifest);
  Mesh mesh = LoadMesh(manifest);
  H1_FECollection fec(2, mesh.Dimension());
  FiniteElementSpace fes(&mesh, &fec);
  GridFunction kernel(&fes);
  FunctionCoefficient kernel_c(
      [](const Vector& x) { return KernelExact(x); });
  kernel.ProjectCoefficient(kernel_c);

  // The pairing the kernel exists for, int K dm dV, with K from the file
  // against K from the field itself. The file's layout is linear between
  // its radial nodes, so the pairing error is the systematic
  // second-derivative bias of that rule: it falls with the square of
  // radial_size, which is the gate.
  FunctionCoefficient perturbation_c(
      [](const Vector& x) { return PerturbationExact(x); });
  auto pairing_error = [&](double radial_size, const char* file_name) {
    const auto path = files.dir / file_name;
    WriteKernelModelFile(
        path, manifest,
        {{.name = "K_eta",
          .field = &kernel,
          .dimensions = {.mass = -1, .length = -1, .time = 1},
          .meaning = "a synthetic kernel of the round-trip test"}},
        {.radial_size = radial_size, .lmax = 16, .name = "round trip"});
    const auto file = planetmodel::ReadModelFile(path);
    CheckModelMatchesManifest(file, manifest);
    EXPECT_EQ(file.LayerNames()[0], "inner_core");
    EXPECT_EQ(file.InterfaceNames().back(), "surface");
    const auto& field = file.FieldOn(0, "K_eta");
    EXPECT_TRUE(field.GetDimensions().has_value() &&
                field.GetDimensions()->length == -1);

    ModelScalarCoefficient read_back(file, "K_eta");
    Vector x;
    double max_error = 0.0, max_value = 0.0;
    double pairing_file = 0.0, pairing_field = 0.0;
    for (int e = 0; e < mesh.GetNE(); e++) {
      if (!read_back.HasLayer(mesh.GetAttribute(e))) {
        continue;
      }
      ElementTransformation* T = mesh.GetElementTransformation(e);
      const IntegrationRule& ir = IntRules.Get(T->GetGeometryType(), 4);
      for (int q = 0; q < ir.GetNPoints(); q++) {
        const IntegrationPoint& ip = ir.IntPoint(q);
        T->SetIntPoint(&ip);
        T->Transform(ip, x);
        const double from_file = read_back.Eval(*T, ip);
        const double from_field = kernel.GetValue(e, ip);
        max_error = std::max(max_error, std::abs(from_file - from_field));
        max_value = std::max(max_value, std::abs(from_field));
        const double w = ip.weight * T->Weight();
        const double dm = perturbation_c.Eval(*T, ip);
        pairing_file += w * from_file * dm;
        pairing_field += w * from_field * dm;
      }
    }
    EXPECT_GT(max_error, 0.0);
    EXPECT_LT(max_error / max_value, 10.0 * radial_size * radial_size);
    return std::abs(pairing_file - pairing_field) /
           std::abs(pairing_field);
  };

  const double coarse = pairing_error(0.1, "kernel_coarse.nc");
  const double fine = pairing_error(0.05, "kernel_fine.nc");
  EXPECT_LT(coarse, 2e-2);
  EXPECT_LT(fine, 0.4 * coarse);
}

TEST(ModelExport, RefusesAShell) {
  const Files files;
  const planetmodel::MeshManifest manifest(kSourceManifest);
  Mesh mesh = LoadMesh(manifest);
  L2_FECollection fec(0, mesh.Dimension());
  FiniteElementSpace fes(&mesh, &fec);
  GridFunction kernel(&fes);
  kernel = 1.0;
  EXPECT_THROW(
      WriteKernelModelFile(files.dir / "bad.nc", manifest,
                           {{.name = "K",
                             .field = &kernel,
                             .attributes = {4}}}),
      planetmodel::Error);
}
