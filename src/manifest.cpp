/**
 * @file manifest.cpp
 * @brief Implementation of manifest.hpp.
 */

#include "AdGIA/manifest.hpp"

#include <format>
#include <fstream>

#include "planetmodel/error.hpp"

namespace AdGIA {

using namespace mfem;

Array<int> ToArray(const std::vector<int>& attributes) {
  Array<int> out(static_cast<int>(attributes.size()));
  for (int i = 0; i < out.Size(); i++) {
    out[i] = attributes[static_cast<std::size_t>(i)];
  }
  return out;
}

Array<int> Marker(const std::vector<int>& attributes, int size) {
  return ToArray(planetmodel::MeshManifest::Marker(attributes, size));
}

Array<int> Marker(const Array<int>& attributes, int size) {
  return Marker(
      std::vector<int>(attributes.begin(), attributes.end()), size);
}

Mesh LoadMesh(const planetmodel::MeshManifest& manifest) {
  const auto file = manifest.MeshFile().string();
  if (!std::ifstream(file).good()) {
    throw planetmodel::Error("LoadMesh", file, "cannot be opened");
  }
  const auto& options = manifest.Options();
  return Mesh(file.c_str(), options.generate_edges, options.refine,
              options.fix_orientation);
}

std::unique_ptr<GridFunction> LoadField(
    const planetmodel::MeshManifest& manifest, Mesh& mesh,
    const std::string& name) {
  const auto& field = manifest.FieldNamed(name);
  const auto file = manifest.FieldFile(name).string();
  auto in = std::ifstream(file);
  if (!in.good()) {
    throw planetmodel::Error("LoadField", file, "cannot be opened");
  }
  auto gf = std::make_unique<GridFunction>(&mesh, in);
  if (field.fe_space != gf->FESpace()->FEColl()->Name() ||
      field.vdim != gf->FESpace()->GetVDim()) {
    throw planetmodel::Error(
        "LoadField", file,
        std::format("holds {} components in {}, the manifest says {} in {}",
                    gf->FESpace()->GetVDim(), gf->FESpace()->FEColl()->Name(),
                    field.vdim, field.fe_space));
  }
  return gf;
}

void Print(const planetmodel::MeshManifest& manifest, std::ostream& os) {
  os << "Manifest " << manifest.Path().string() << "\n  mesh "
     << manifest.MeshFile().filename().string() << " (" << manifest.Format()
     << ", " << manifest.Nodes() << " nodes)\n  layers\n";
  for (const auto& l : manifest.Layers()) {
    os << "    " << l.attribute << "  " << l.name << "  [" << l.r_inner
       << ", " << l.r_outer << "]";
    if (!l.in_geometry) {
      os << "  shell";
    } else if (l.fluid) {
      os << "  " << (*l.fluid ? "fluid" : "solid");
    }
    os << "\n";
  }
  os << "  interfaces\n";
  for (const auto& f : manifest.Interfaces()) {
    os << "    " << f.attribute << "  " << f.name << "  radius " << f.radius;
    if (f.kind) {
      os << "  " << *f.kind;
    }
    os << "\n";
  }
  if (!manifest.Fields().empty()) {
    os << "  fields\n";
    for (const auto& f : manifest.Fields()) {
      os << "    " << f.name << "  " << f.file << "  " << f.fe_space
         << " vdim " << f.vdim << "\n";
    }
  }
  if (manifest.HasScales()) {
    const auto& scales = manifest.GetScales();
    os << "  scales: length " << scales.Length() << " m, mass "
       << scales.Mass() << " kg, time " << scales.Time() << " s\n";
  }
  for (const auto& [key, value] : manifest.Constants()) {
    os << "  " << key << " = " << value << "\n";
  }
}

}  // namespace AdGIA
