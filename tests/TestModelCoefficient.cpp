#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "TestCommon.hpp"
#include "planetmodel/error.hpp"
#include "planetmodel/modelfile.hpp"

/*
  Tests for model_coefficient.hpp: model-file fields as MFEM coefficients.

  A model file is written here (planetmodel's own writer) on the skeleton
  of the three-layer-Earth mesh of the build, with a radial density that
  is a polynomial (so the file's GLL interpolation reproduces it exactly)
  and a viscosity on the mantle alone that varies with direction; a copy
  of the mesh's manifest is patched with the model record.

  - LoadModelFile reads the recorded file and holds it to the export's
    contract (layer n of the file is attribute n + 1): a manifest without
    a record, and a file whose skeleton does not match the layers, are
    refused.
  - The radial field evaluates at the quadrature points of the body's
    elements to the exact polynomial (machine precision; the radius of a
    curved facet's point is clamped into the layer).
  - The lateral field interpolates its smooth closed form; a layer
    without the field refuses to evaluate, and HasLayer says so.
*/

namespace {

const char* const kSourceManifest = "../data/earth_innercore_3d.json";

double RhoExact(double r) { return 2.0 - r * r; }

double ViscosityExact(double r, double theta, double phi) {
  return r * (1.0 + 0.1 * std::sin(theta) * std::cos(phi));
}

// The model file, a patched manifest recording it, and a mismatched pair,
// in a directory of the test's own (ctest runs tests concurrently).
struct Files {
  std::filesystem::path dir;
  std::string manifest, mismatched_manifest;
  Files() {
    const auto* info = testing::UnitTest::GetInstance()->current_test_info();
    dir = std::string("model_coefficient_test_") + info->test_suite_name() +
          "_" + info->name();
    std::filesystem::remove_all(dir);
    std::filesystem::create_directory(dir);

    const planetmodel::MeshManifest source(kSourceManifest);
    std::vector<double> boundaries{source.Layers().front().r_inner};
    for (const int a : source.BodyAttributes()) {
      boundaries.push_back(
          source.Layers()[static_cast<std::size_t>(a - 1)].r_outer);
    }
    auto file = planetmodel::ModelFile(planetmodel::Skeleton(boundaries));
    const auto& skeleton = file.GetSkeleton();
    const int last = static_cast<int>(skeleton.NLayers()) - 1;
    file.AddPart(0, last,
                 planetmodel::ElementEdges(skeleton, 0, last, 0.1), 5,
                 planetmodel::GaussLegendre(32));
    const auto rho =
        planetmodel::LayerSource::Radial([](double r) { return RhoExact(r); });
    file.AddField("rho", {{0, rho}, {1, rho}, {2, rho}});
    file.AddField("viscosity",
                  {{last, planetmodel::LayerSource::Lateral(ViscosityExact)}},
                  {.lateral = planetmodel::LateralRule::Cubic});
    planetmodel::WriteModelFile(dir / "test.model.nc", file);

    auto mismatched = planetmodel::ModelFile(
        planetmodel::Skeleton({0.0, boundaries.back()}));
    mismatched.AddPart(0, 0,
                       planetmodel::ElementEdges(mismatched.GetSkeleton(), 0,
                                                 0, 0.1),
                       5, planetmodel::GaussLegendre(8));
    mismatched.AddField(
        "rho", {{0, planetmodel::LayerSource::Radial(
                        [](double) { return 1.0; })}});
    planetmodel::WriteModelFile(dir / "wrong.model.nc", mismatched);

    manifest = Patched("case.json", "test.model.nc");
    mismatched_manifest = Patched("wrong.json", "wrong.model.nc");
  }
  ~Files() { std::filesystem::remove_all(dir); }

  // The source manifest with a model record naming `model`: the writer's
  // empty "model": null is replaced (manifests from older writers lack the
  // key, so it is then inserted before "schema").
  std::string Patched(const char* name, const char* model) const {
    std::ifstream in(kSourceManifest);
    std::stringstream buffer;
    buffer << in.rdbuf();
    std::string text = std::move(buffer).str();
    const std::string record = std::string("\"model\": {\"file\": \"") + model +
                               "\", \"schema\": \"planetmodel.model/1\"}";
    const std::string empty = "\"model\": null";
    if (const auto pos = text.find(empty); pos != std::string::npos) {
      text.replace(pos, empty.size(), record);
    } else {
      const auto schema_pos = text.rfind("\"schema\"");
      EXPECT_NE(schema_pos, std::string::npos);
      text.insert(schema_pos, record + ",\n  ");
    }
    const std::string path = (dir / name).string();
    std::ofstream(path) << text;
    return path;
  }
};

// The layer intervals by attribute, for clamping the reference values the
// way the coefficient clamps the field's.
std::vector<std::pair<double, double>> Intervals(
    const planetmodel::MeshManifest& manifest) {
  std::vector<std::pair<double, double>> out;
  for (const auto& layer : manifest.Layers()) {
    out.emplace_back(layer.r_inner, layer.r_outer);
  }
  return out;
}

}  // namespace

TEST(ModelCoefficient, LoadModelFileHoldsTheContract) {
  const Files files;
  const planetmodel::MeshManifest manifest(files.manifest);
  const auto file = LoadModelFile(manifest);
  EXPECT_EQ(file.GetSkeleton().NLayers(), 3u);

  const planetmodel::MeshManifest source(kSourceManifest);
  EXPECT_THROW(source.ModelFile(), planetmodel::Error);

  const planetmodel::MeshManifest mismatched(files.mismatched_manifest);
  EXPECT_THROW(LoadModelFile(mismatched), planetmodel::Error);
}

TEST(ModelCoefficient, RadialFieldMatchesExactly) {
  const Files files;
  const planetmodel::MeshManifest manifest(files.manifest);
  const auto file = LoadModelFile(manifest);
  ModelScalarCoefficient rho(file, "rho");
  // The mesh from the source manifest: the patched copy sits in the test's
  // directory, away from the mesh it names.
  Mesh mesh = LoadMesh(planetmodel::MeshManifest(kSourceManifest));
  const auto intervals = Intervals(manifest);
  Vector x;
  double max_error = 0.0;
  for (int e = 0; e < mesh.GetNE(); e++) {
    const int attribute = mesh.GetAttribute(e);
    if (!rho.HasLayer(attribute)) {
      continue;  // the buffer shell has no model data
    }
    ElementTransformation* T = mesh.GetElementTransformation(e);
    const IntegrationRule& ir = IntRules.Get(T->GetGeometryType(), 4);
    for (int q = 0; q < ir.GetNPoints(); q++) {
      const IntegrationPoint& ip = ir.IntPoint(q);
      T->SetIntPoint(&ip);
      T->Transform(ip, x);
      const auto& [r_inner, r_outer] =
          intervals[static_cast<std::size_t>(attribute - 1)];
      const double r = std::clamp(double(x.Norml2()), r_inner, r_outer);
      max_error = std::max(max_error,
                           std::abs(rho.Eval(*T, ip) - RhoExact(r)));
    }
  }
  EXPECT_LT(max_error, 1e-12);
}

TEST(ModelCoefficient, LateralFieldInterpolates) {
  const Files files;
  const planetmodel::MeshManifest manifest(files.manifest);
  const auto file = LoadModelFile(manifest);
  ModelScalarCoefficient viscosity(file, "viscosity");
  EXPECT_TRUE(viscosity.HasLayer(3));
  EXPECT_FALSE(viscosity.HasLayer(1));
  EXPECT_FALSE(viscosity.HasLayer(4));
  EXPECT_THROW(ModelScalarCoefficient(file, "mu"), planetmodel::Error);

  Mesh mesh = LoadMesh(planetmodel::MeshManifest(kSourceManifest));
  const auto intervals = Intervals(manifest);
  Vector x;
  double max_error = 0.0;
  bool threw_off_layer = false;
  for (int e = 0; e < mesh.GetNE(); e++) {
    const int attribute = mesh.GetAttribute(e);
    ElementTransformation* T = mesh.GetElementTransformation(e);
    const IntegrationRule& ir = IntRules.Get(T->GetGeometryType(), 4);
    const IntegrationPoint& ip = ir.IntPoint(0);
    T->SetIntPoint(&ip);
    if (attribute == 1 && !threw_off_layer) {
      EXPECT_THROW(viscosity.Eval(*T, ip), planetmodel::Error);
      threw_off_layer = true;
      continue;
    }
    if (attribute != 3) {
      continue;
    }
    for (int q = 0; q < ir.GetNPoints(); q++) {
      const IntegrationPoint& point = ir.IntPoint(q);
      T->SetIntPoint(&point);
      T->Transform(point, x);
      const auto& [r_inner, r_outer] =
          intervals[static_cast<std::size_t>(attribute - 1)];
      const double r_mesh = x.Norml2();
      const double r = std::clamp(r_mesh, r_inner, r_outer);
      const double theta = std::acos(std::clamp(x[2] / r_mesh, -1.0, 1.0));
      const double phi = std::atan2(x[1], x[0]);
      max_error = std::max(
          max_error,
          std::abs(viscosity.Eval(*T, point) - ViscosityExact(r, theta, phi)));
    }
  }
  EXPECT_TRUE(threw_off_layer);
  EXPECT_GT(max_error, 0.0);
  EXPECT_LT(max_error, 1e-3);
}
