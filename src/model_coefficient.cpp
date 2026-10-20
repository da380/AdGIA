/**
 * @file model_coefficient.cpp
 * @brief Implementation of model_coefficient.hpp.
 */

#include "AdGIA/model_coefficient.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <numbers>

#include "planetmodel/error.hpp"

namespace AdGIA {

using namespace mfem;

void CheckModelMatchesManifest(const planetmodel::ModelFile& file,
                               const planetmodel::MeshManifest& manifest) {
  const auto& skeleton = file.GetSkeleton();
  const auto body = manifest.BodyAttributes();
  if (skeleton.NLayers() != body.size()) {
    throw planetmodel::Error(
        "CheckModelMatchesManifest", file.Source(),
        std::format("holds {} layers for a mesh of {} body layers; layer n "
                    "of the file is attribute n + 1 of the mesh",
                    skeleton.NLayers(), body.size()));
  }
  const auto tolerance = file.Rtol() * skeleton.Boundaries().back();
  for (std::size_t i = 0; i < body.size(); i++) {
    const auto& layer =
        manifest.Layers()[static_cast<std::size_t>(body[i] - 1)];
    const auto [r_inner, r_outer] = skeleton.Interval(i);
    if (std::abs(layer.r_inner - r_inner) > tolerance ||
        std::abs(layer.r_outer - r_outer) > tolerance) {
      throw planetmodel::Error(
          "CheckModelMatchesManifest", file.Source(),
          std::format("layer {} runs over [{}, {}] but the mesh's layer "
                      "{} over [{}, {}]: not the mesh's model",
                      i, r_inner, r_outer, body[i], layer.r_inner,
                      layer.r_outer));
    }
  }
}

planetmodel::ModelFile LoadModelFile(
    const planetmodel::MeshManifest& manifest) {
  auto file = planetmodel::ReadModelFile(manifest.ModelFile());
  CheckModelMatchesManifest(file, manifest);
  return file;
}

ModelScalarCoefficient::ModelScalarCoefficient(
    const planetmodel::ModelFile& file, const std::string& name)
    : name_(name) {
  const auto layers = file.GetSkeleton().NLayers();
  fields_.resize(layers);
  for (std::size_t i = 0; i < layers; i++) {
    if (file.HasField(static_cast<int>(i), name)) {
      fields_[i] = file.FieldOn(static_cast<int>(i), name).AsScalar();
    }
  }
  if (std::ranges::none_of(fields_,
                           [](const auto& f) { return f.has_value(); })) {
    throw planetmodel::Error("ModelScalarCoefficient", file.Source(),
                             std::format("holds no field {}", name));
  }
}

real_t ModelScalarCoefficient::Eval(ElementTransformation& T,
                                    const IntegrationPoint& ip) {
  const int attribute = T.Attribute;
  if (!HasLayer(attribute)) {
    throw planetmodel::Error(
        "ModelScalarCoefficient", name_,
        std::format("is not on the layer of attribute {}", attribute));
  }
  const auto& field = *fields_[static_cast<std::size_t>(attribute - 1)];
  T.Transform(ip, x_);
  const auto dim = x_.Size();
  const auto r_mesh = static_cast<double>(x_.Norml2());
  const auto [r_inner, r_outer] = field.Underlying().Interval();
  const auto r = std::clamp(r_mesh, r_inner, r_outer);
  double theta = std::numbers::pi / 2.0, phi = 0.0;
  if (r_mesh > 0.0) {
    if (dim == 3) {
      theta = std::acos(std::clamp(x_[2] / r_mesh, -1.0, 1.0));
    }
    phi = std::atan2(x_[1], x_[0]);
  }
  return static_cast<real_t>(field.Value(r, theta, phi));
}

}  // namespace AdGIA
