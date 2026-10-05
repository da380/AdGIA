/**
 * @file mesh_manifest.cpp
 * @brief Implementation of mesh_manifest.hpp.
 */

#include "AdGIA/mesh_manifest.hpp"

#include "AdGIA/json.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <utility>

namespace AdGIA {

using namespace mfem;

namespace {

// The current schema, and the previous one, which differs only in lacking
// the model-derived records (layers[].fluid, interfaces[].kind and .values,
// fields[].radial_degree).
const char* const kSchema = "planetmodel.mesh.manifest/5";
const char* const kOldSchema = "planetmodel.mesh.manifest/4";

// Typed access to the members of an object, each failing with the file and
// the member's name.
class Record {
 public:
  Record(const Json& json, std::string where, const std::string& file)
      : json_(json), where_(std::move(where)), file_(file) {
    MFEM_VERIFY(json.type == Json::Type::Object,
                "MeshManifest: " << file_ << ": " << where_
                                 << " is not an object.");
  }

  const Json& Member(const std::string& key, Json::Type type) const {
    const Json* v = json_.Find(key);
    MFEM_VERIFY(v && v->type == type, "MeshManifest: "
                                          << file_ << ": " << where_ << "."
                                          << key
                                          << " is missing or of another type.");
    return *v;
  }

  const Json* Optional(const std::string& key) const {
    const Json* v = json_.Find(key);
    return v && !v->IsNull() ? v : nullptr;
  }

  real_t Number(const std::string& key) const {
    return Member(key, Json::Type::Number).number;
  }
  int Integer(const std::string& key) const {
    const Json& v = Member(key, Json::Type::Number);
    MFEM_VERIFY(v.IsInteger(), "MeshManifest: " << file_ << ": " << where_
                                                << "." << key
                                                << " is not an integer.");
    return static_cast<int>(v.number);
  }
  bool Bool(const std::string& key) const {
    return Member(key, Json::Type::Bool).boolean;
  }
  const std::string& String(const std::string& key) const {
    return Member(key, Json::Type::String).string;
  }
  std::vector<int> Integers(const std::string& key) const {
    const Json& v = Member(key, Json::Type::Array);
    std::vector<int> out;
    for (const Json& e : v.array) {
      MFEM_VERIFY(e.IsInteger(), "MeshManifest: " << file_ << ": " << where_
                                                  << "." << key
                                                  << " holds a non-integer.");
      out.push_back(static_cast<int>(e.number));
    }
    return out;
  }

 private:
  const Json& json_;
  std::string where_;
  const std::string& file_;
};

std::string Indexed(const char* what, std::size_t i) {
  return std::string(what) + "[" + std::to_string(i) + "]";
}

Array<int> ToArray(const std::vector<int>& v) {
  Array<int> a(static_cast<int>(v.size()));
  for (int i = 0; i < a.Size(); i++) {
    a[i] = v[i];
  }
  return a;
}

}  // namespace

// ---------------------------------------------------------------------------
// MeshManifest

MeshManifest::MeshManifest(const std::string& path) : path_(path) {
  const auto slash = path.find_last_of('/');
  directory_ = slash == std::string::npos ? "" : path.substr(0, slash + 1);

  std::ifstream in(path);
  MFEM_VERIFY(in.good(), "MeshManifest: cannot open " << path << ".");
  std::stringstream buffer;
  buffer << in.rdbuf();
  const std::string text = buffer.str();
  const Json root_json = JsonReader(text, path).Read();
  const Record root(root_json, "manifest", path);

  const std::string& schema = root.String("schema");
  MFEM_VERIFY(schema == kSchema || schema == kOldSchema,
              "MeshManifest: " << path << " has schema " << schema
                               << ", expected " << kSchema << " (or "
                               << kOldSchema << ").");

  {
    const Record m(root.Member("mesh", Json::Type::Object), "mesh", path);
    mesh_file_ = m.String("file");
    mesh_format_ = m.String("format");
    physical_nodes_ = m.String("nodes") == "physical";
    const Json& options_json = m.Member("read_options", Json::Type::Object);
    const Record options(options_json, "mesh.read_options", path);
    if (options.Optional("generate_edges")) {
      generate_edges_ = options.Integer("generate_edges");
    }
    if (options.Optional("refine")) {
      refine_ = options.Integer("refine");
    }
    if (options.Optional("fix_orientation")) {
      fix_orientation_ = options.Bool("fix_orientation");
    }
  }

  const auto& layers = root.Member("layers", Json::Type::Array).array;
  MFEM_VERIFY(!layers.empty(), "MeshManifest: " << path << " lists no layers.");
  for (std::size_t i = 0; i < layers.size(); i++) {
    const Record r(layers[i], Indexed("layers", i), path);
    Layer l;
    l.attribute = r.Integer("attribute");
    l.name = r.String("name");
    l.r_inner = r.Number("r_inner");
    l.r_outer = r.Number("r_outer");
    l.in_geometry = r.Bool("in_geometry");
    if (r.Optional("fluid")) {
      l.fluid = r.Bool("fluid");
    }
    MFEM_VERIFY(l.attribute == static_cast<int>(i) + 1,
                "MeshManifest: " << path << ": layers are not numbered from "
                                 << "the centre.");
    layers_.push_back(l);
  }

  const int n_layers = static_cast<int>(layers_.size());
  const auto& faces = root.Member("interfaces", Json::Type::Array).array;
  for (std::size_t i = 0; i < faces.size(); i++) {
    const Record r(faces[i], Indexed("interfaces", i), path);
    Interface f;
    f.attribute = r.Integer("attribute");
    f.name = r.String("name");
    f.radius = r.Number("radius");
    const auto between = r.Integers("between_layers");
    MFEM_VERIFY(between.size() == 2 && between[0] < n_layers &&
                    between[1] < n_layers,
                "MeshManifest: " << path << ": " << Indexed("interfaces", i)
                                 << ".between_layers is not a pair of layers.");
    // 0-based with -1 for the outside in the file; attributes with 0 here.
    f.below = between[0] + 1;
    f.above = between[1] + 1;
    if (r.Optional("kind")) {
      f.kind = r.String("kind");
    }
    if (const Json* values = r.Optional("values")) {
      MFEM_VERIFY(values->type == Json::Type::Object,
                  "MeshManifest: " << path << ": " << Indexed("interfaces", i)
                                   << ".values is not an object.");
      for (const auto& [key, pair] : values->object) {
        MFEM_VERIFY(
            pair.type == Json::Type::Array && pair.array.size() == 2,
            "MeshManifest: " << path << ": " << Indexed("interfaces", i)
                             << ".values." << key << " is not a pair.");
        auto side = [&](const Json& e) {
          MFEM_VERIFY(e.type == Json::Type::Number || e.IsNull(),
                      "MeshManifest: " << path << ": "
                                       << Indexed("interfaces", i)
                                       << ".values." << key
                                       << " holds a non-number.");
          return e.IsNull() ? std::numeric_limits<real_t>::quiet_NaN()
                            : static_cast<real_t>(e.number);
        };
        f.values[key] = {side(pair.array[0]), side(pair.array[1])};
      }
    }
    interfaces_.push_back(f);
  }

  for (const Json& e : root.Member("fields", Json::Type::Array).array) {
    const Record r(e, Indexed("fields", fields_.size()), path);
    Field f;
    f.name = r.String("name");
    f.file = r.String("file");
    f.fe_space = r.String("fe_space");
    f.ordering = r.String("ordering");
    f.unit = r.String("unit");
    f.vdim = r.Integer("vdim");
    f.rank = r.Integer("rank");
    f.weight = r.Integer("weight");
    f.voigt = r.Bool("voigt");
    f.layers = r.Integers("layers");
    if (r.Optional("radial_degree")) {
      f.radial_degree = r.Integer("radial_degree");
    }
    fields_.push_back(f);
  }

  if (const Json* s = root.Optional("scales")) {
    const Record r(*s, "scales", path);
    has_scales_ = true;
    length_ = r.Number("length");
    mass_ = r.Number("mass");
    time_ = r.Number("time");
  }

  if (const Json* c = root.Optional("constants")) {
    const Record r(*c, "constants", path);
    for (const auto& [key, value] : c->object) {
      constants_[key] = r.Number(key);
    }
  }

  if (const Json* m = root.Optional("meta")) {
    const Record r(*m, "meta", path);
    for (const auto& [key, value] : m->object) {
      if (value.type == Json::Type::Number) {
        meta_numbers_[key] = value.number;
      } else if (value.type == Json::Type::String) {
        meta_strings_[key] = value.string;
      } else if (value.type == Json::Type::Array) {
        bool integers = true;
        for (const Json& e : value.array) {
          integers = integers && e.IsInteger();
        }
        if (integers) {
          meta_integers_[key] = r.Integers(key);
        }
      }
    }
  }

  if (HasMeta("fluid_layers")) {
    for (const int a : MetaIntegers("fluid_layers")) {
      MFEM_VERIFY(a >= 1 && a <= n_layers && layers_[a - 1].in_geometry,
                  "MeshManifest: " << path << ": meta.fluid_layers names "
                                   << a << ", not a layer of the body.");
      layers_[a - 1].fluid = true;
    }
  }
}

real_t MeshManifest::Interface::ValueBeside(const std::string& field,
                                            int attribute) const {
  MFEM_VERIFY(attribute == below || attribute == above,
              "MeshManifest: layer " << attribute << " is not a side of the "
                                     << "interface " << name << ".");
  const auto it = values.find(field);
  const real_t value =
      it == values.end()
          ? std::numeric_limits<real_t>::quiet_NaN()
          : (attribute == below ? it->second.first : it->second.second);
  MFEM_VERIFY(std::isfinite(value),
              "MeshManifest: the interface "
                  << name << " holds no value of " << field
                  << " on the side of layer " << attribute
                  << "; the manifest was written without the model "
                  << "(planetmodel 1.2.3 or later writes it).");
  return value;
}

const MeshManifest::Layer& MeshManifest::LayerNamed(
    const std::string& name) const {
  for (const Layer& l : layers_) {
    if (l.name == name) {
      return l;
    }
  }
  MFEM_ABORT("MeshManifest: " << path_ << " has no layer named " << name
                              << ".");
}

const MeshManifest::Interface& MeshManifest::InterfaceNamed(
    const std::string& name) const {
  for (const Interface& f : interfaces_) {
    if (f.name == name) {
      return f;
    }
  }
  MFEM_ABORT("MeshManifest: " << path_ << " has no interface named " << name
                              << ".");
}

bool MeshManifest::HasField(const std::string& name) const {
  for (const Field& f : fields_) {
    if (f.name == name) {
      return true;
    }
  }
  return false;
}

const MeshManifest::Field& MeshManifest::FieldNamed(
    const std::string& name) const {
  for (const Field& f : fields_) {
    if (f.name == name) {
      return f;
    }
  }
  MFEM_ABORT("MeshManifest: " << path_ << " has no field named " << name
                              << ".");
}

Array<int> MeshManifest::SolidAttributes() const {
  Array<int> a;
  for (const Layer& l : layers_) {
    if (l.in_geometry && !l.fluid) {
      a.Append(l.attribute);
    }
  }
  return a;
}

Array<int> MeshManifest::FluidAttributes() const {
  Array<int> a;
  for (const Layer& l : layers_) {
    if (l.in_geometry && l.fluid) {
      a.Append(l.attribute);
    }
  }
  return a;
}

Array<int> MeshManifest::BodyAttributes() const {
  Array<int> a;
  for (const Layer& l : layers_) {
    if (l.in_geometry) {
      a.Append(l.attribute);
    }
  }
  return a;
}

Array<int> MeshManifest::ShellAttributes() const {
  Array<int> a;
  for (const Layer& l : layers_) {
    if (!l.in_geometry) {
      a.Append(l.attribute);
    }
  }
  return a;
}

Array<int> MeshManifest::FluidSolidInterfaces() const {
  Array<int> a;
  auto body = [&](int attr) { return attr > 0 && layers_[attr - 1].in_geometry; };
  for (const Interface& f : interfaces_) {
    if (body(f.below) && body(f.above) &&
        layers_[f.below - 1].fluid != layers_[f.above - 1].fluid) {
      a.Append(f.attribute);
    }
  }
  return a;
}

Array<int> MeshManifest::FluidSolidInterfaces(int fluid_attribute) const {
  Array<int> a;
  auto solid = [&](int attr) {
    return attr > 0 && layers_[attr - 1].in_geometry &&
           !layers_[attr - 1].fluid;
  };
  for (const Interface& f : interfaces_) {
    if ((f.below == fluid_attribute && solid(f.above)) ||
        (f.above == fluid_attribute && solid(f.below))) {
      a.Append(f.attribute);
    }
  }
  return a;
}

int MeshManifest::SurfaceAttribute() const {
  int outermost = 0;
  for (const Layer& l : layers_) {
    if (l.in_geometry) {
      outermost = l.attribute;
    }
  }
  for (const Interface& f : interfaces_) {
    if (f.below == outermost) {
      return f.attribute;
    }
  }
  MFEM_ABORT("MeshManifest: " << path_ << " lists no surface of the body.");
}

int MeshManifest::OuterAttribute() const {
  for (const Interface& f : interfaces_) {
    if (f.above == 0) {
      return f.attribute;
    }
  }
  MFEM_ABORT("MeshManifest: " << path_ << " lists no outer boundary.");
}

real_t MeshManifest::SurfaceRadius() const {
  return interfaces_[SurfaceAttribute() - interfaces_.front().attribute].radius;
}

real_t MeshManifest::OuterRadius() const { return layers_.back().r_outer; }

Array<int> MeshManifest::Marker(const Array<int>& attributes, int size) {
  Array<int> marker(size);
  marker = 0;
  for (const int a : attributes) {
    MFEM_VERIFY(a >= 1, "MeshManifest::Marker: attributes start at one.");
    if (a <= size) {
      marker[a - 1] = 1;
    }
  }
  return marker;
}

real_t MeshManifest::ScaleFactor(int mass, int length, int time) const {
  return std::pow(mass_, mass) * std::pow(length_, length) *
         std::pow(time_, time);
}

bool MeshManifest::HasConstant(const std::string& name) const {
  return constants_.count(name) > 0;
}

real_t MeshManifest::Constant(const std::string& name) const {
  const auto it = constants_.find(name);
  MFEM_VERIFY(it != constants_.end(), "MeshManifest: " << path_
                                                       << " has no constant "
                                                       << name << ".");
  return it->second;
}

bool MeshManifest::HasMeta(const std::string& key) const {
  return meta_numbers_.count(key) || meta_strings_.count(key) ||
         meta_integers_.count(key);
}

real_t MeshManifest::MetaNumber(const std::string& key) const {
  const auto it = meta_numbers_.find(key);
  MFEM_VERIFY(it != meta_numbers_.end(),
              "MeshManifest: " << path_ << " has no number meta." << key
                               << ".");
  return it->second;
}

const std::string& MeshManifest::MetaString(const std::string& key) const {
  const auto it = meta_strings_.find(key);
  MFEM_VERIFY(it != meta_strings_.end(),
              "MeshManifest: " << path_ << " has no string meta." << key
                               << ".");
  return it->second;
}

const std::vector<int>& MeshManifest::MetaIntegers(
    const std::string& key) const {
  const auto it = meta_integers_.find(key);
  MFEM_VERIFY(it != meta_integers_.end(),
              "MeshManifest: " << path_ << " has no list of integers meta."
                               << key << ".");
  return it->second;
}

std::string MeshManifest::MeshFile() const { return directory_ + mesh_file_; }

Mesh MeshManifest::LoadMesh() const {
  const std::string file = MeshFile();
  std::ifstream in(file);
  MFEM_VERIFY(in.good(), "MeshManifest: cannot open the mesh " << file << ".");
  in.close();
  return Mesh(file.c_str(), generate_edges_, refine_, fix_orientation_);
}

std::unique_ptr<GridFunction> MeshManifest::LoadField(
    Mesh& mesh, const std::string& name) const {
  const Field& field = FieldNamed(name);
  const std::string file = directory_ + field.file;
  std::ifstream in(file);
  MFEM_VERIFY(in.good(), "MeshManifest: cannot open the field " << file << ".");
  auto gf = std::make_unique<GridFunction>(&mesh, in);
  MFEM_VERIFY(field.fe_space == gf->FESpace()->FEColl()->Name() &&
                  field.vdim == gf->FESpace()->GetVDim(),
              "MeshManifest: " << file << " holds " << gf->FESpace()->GetVDim()
                               << " components in "
                               << gf->FESpace()->FEColl()->Name()
                               << ", the manifest says " << field.vdim
                               << " in " << field.fe_space << ".");
  return gf;
}

void MeshManifest::Print(std::ostream& os) const {
  os << "Manifest " << path_ << "\n  mesh " << mesh_file_ << " ("
     << mesh_format_ << ", " << (physical_nodes_ ? "physical" : "reference")
     << " nodes)\n  layers\n";
  for (const Layer& l : layers_) {
    os << "    " << l.attribute << "  " << l.name << "  [" << l.r_inner << ", "
       << l.r_outer << "]  "
       << (!l.in_geometry ? "shell" : l.fluid ? "fluid" : "solid") << "\n";
  }
  os << "  interfaces\n";
  for (const Interface& f : interfaces_) {
    os << "    " << f.attribute << "  " << f.name << "  radius " << f.radius;
    if (!f.kind.empty()) {
      os << "  " << f.kind;
    }
    os << "\n";
  }
  if (!fields_.empty()) {
    os << "  fields\n";
    for (const Field& f : fields_) {
      os << "    " << f.name << "  " << f.file << "  " << f.fe_space
         << " vdim " << f.vdim << "\n";
    }
  }
  if (has_scales_) {
    os << "  scales: length " << length_ << " m, mass " << mass_
       << " kg, time " << time_ << " s\n";
  }
  for (const auto& [key, value] : constants_) {
    os << "  " << key << " = " << value << "\n";
  }
}

}  // namespace AdGIA
