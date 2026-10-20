/**
 * @file model_export.cpp
 * @brief Implementation of model_export.hpp.
 */

#include "AdGIA/model_export.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <map>

#include "planetmodel/error.hpp"

namespace AdGIA {

using namespace mfem;

namespace {

// The values of `field` at the layer's sample points. A point the
// discrete mesh misses (a curved facet at a layer boundary) is nudged
// radially towards the layer's middle and retried.
std::vector<double> SampleLayer(const planetmodel::LayerPoints& points,
                                const GridFunction& field, double r_inner,
                                double r_outer, const std::string& name) {
  auto& mesh = *field.FESpace()->GetMesh();
  const auto n = points.Size();
  const auto r_mid = 0.5 * (r_inner + r_outer);

  // `pull` is the fraction of the way from the point's radius to the
  // layer's middle: a curved facet of order k sits O(h^{k+1}) off the
  // exact sphere, so the ladder reaches ~1e-2 of the half-span.
  auto fill = [&](DenseMatrix& pts, const std::vector<std::size_t>& which,
                  double pull) {
    for (std::size_t j = 0; j < which.size(); j++) {
      auto p = points.At(which[j]);
      const auto r = p.r + pull * (r_mid - p.r);
      const auto s = std::sin(p.theta);
      pts(0, static_cast<int>(j)) = r * s * std::cos(p.phi);
      pts(1, static_cast<int>(j)) = r * s * std::sin(p.phi);
      pts(2, static_cast<int>(j)) = r * std::cos(p.theta);
    }
  };

  auto values = std::vector<double>(n);
  auto missing = std::vector<std::size_t>(n);
  for (std::size_t i = 0; i < n; i++) {
    missing[i] = i;
  }
  Array<int> elems;
  Array<IntegrationPoint> ips;
  for (const double pull : {0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2}) {
    DenseMatrix pts(3, static_cast<int>(missing.size()));
    fill(pts, missing, pull);
    mesh.FindPoints(pts, elems, ips, false);
    auto still = std::vector<std::size_t>{};
    for (std::size_t j = 0; j < missing.size(); j++) {
      if (elems[static_cast<int>(j)] >= 0) {
        values[missing[j]] = field.GetValue(elems[static_cast<int>(j)],
                                            ips[static_cast<int>(j)]);
      } else {
        still.push_back(missing[j]);
      }
    }
    missing = std::move(still);
    if (missing.empty()) {
      return values;
    }
  }
  throw planetmodel::Error(
      "WriteKernelModelFile", name,
      std::format("{} of {} sample points of the layer over [{}, {}] lie "
                  "off the mesh",
                  missing.size(), n, r_inner, r_outer));
}

}  // namespace

void WriteKernelModelFile(const std::filesystem::path& path,
                          const planetmodel::MeshManifest& manifest,
                          const std::vector<ExportedField>& fields,
                          const ModelExportOptions& options) {
  const auto body = manifest.BodyAttributes();
  std::vector<double> boundaries{manifest.Layers().front().r_inner};
  for (const int a : body) {
    boundaries.push_back(
        manifest.Layers()[static_cast<std::size_t>(a - 1)].r_outer);
  }
  auto file = planetmodel::ModelFile(planetmodel::Skeleton(boundaries));
  if (!options.name.empty()) {
    file.SetName(options.name);
  }
  for (std::size_t i = 0; i < body.size(); i++) {
    file.SetLayerName(
        static_cast<int>(i),
        manifest.Layers()[static_cast<std::size_t>(body[i] - 1)].name);
  }
  // The file's interfaces are the body's boundaries, inner first — the
  // manifest's first entries, since the skeleton is the body's layers.
  for (std::size_t i = 0; i < file.InterfaceRadii().size(); i++) {
    file.SetInterfaceName(static_cast<int>(i),
                          manifest.Interfaces()[i].name);
  }
  const auto& skeleton = file.GetSkeleton();
  const int last = static_cast<int>(skeleton.NLayers()) - 1;
  file.AddPart(0, last,
               planetmodel::ElementEdges(skeleton, 0, last,
                                         options.radial_size),
               2, planetmodel::GaussLegendre(options.lmax));

  for (const auto& exported : fields) {
    if (!exported.field) {
      throw planetmodel::Error("WriteKernelModelFile", exported.name,
                               "has no field");
    }
    auto& mesh = *exported.field->FESpace()->GetMesh();
    auto attributes = exported.attributes;
    if (attributes.empty()) {
      for (const int a : mesh.attributes) {
        if (std::ranges::contains(body, a)) {
          attributes.push_back(a);
        }
      }
    }
    if (attributes.empty()) {
      throw planetmodel::Error(
          "WriteKernelModelFile", exported.name,
          "is on no body layer of the manifest's mesh");
    }
    auto sources = std::map<int, planetmodel::LayerSource>{};
    for (const int a : attributes) {
      const auto it = std::ranges::find(body, a);
      if (it == body.end()) {
        throw planetmodel::Error(
            "WriteKernelModelFile", exported.name,
            std::format("names attribute {}, not a body layer", a));
      }
      const int layer = static_cast<int>(it - body.begin());
      const auto [r_inner, r_outer] =
          skeleton.Interval(static_cast<std::size_t>(layer));
      sources.emplace(
          layer, planetmodel::LayerSource::LateralValues(
                     SampleLayer(file.PointsOf(layer), *exported.field,
                                 r_inner, r_outer, exported.name),
                     planetmodel::Frame::Cartesian));
    }
    const auto spec = planetmodel::FieldSpec{
        .character = planetmodel::SCALAR,
        .dimensions = exported.dimensions,
        .meaning = exported.meaning};
    file.AddField(exported.name, spec, sources,
                  {.lateral = planetmodel::LateralRule::Linear});
  }
  planetmodel::WriteModelFile(path, file);
}

}  // namespace AdGIA
