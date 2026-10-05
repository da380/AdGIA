/**
 * @file mesh_manifest.hpp
 * @brief The manifest that planetmodel writes beside a mesh: what each
 * attribute of the mesh means, which files hold which fields of the model,
 * and the units and constants they are in.
 *
 * A mesh file carries numbered attributes and nothing else. The manifest is
 * a JSON file of the same basename (schema `planetmodel.mesh.manifest/5`;
 * /4 is also read) listing the layers (element attributes 1..N from the
 * centre, the shells outside the body last) and the interfaces (boundary
 * attributes 1..M in the same order) with their names and radii, the fields
 * written as GridFunctions beside the mesh, the scales of the model's units
 * in SI and the model's constants in its units. MeshManifest reads it,
 * answers questions of the kind "which attributes are solid" or "which
 * boundary is the surface" as the attribute lists and markers the problem
 * classes take, and opens the mesh and the fields the way the manifest says.
 *
 * Schema 5 carries what only the exported model can say, null where no
 * model has said: `layers[].fluid`, `interfaces[].kind`, the one-sided
 * `interfaces[].values` of the exported scalar radial fields, and
 * `fields[].radial_degree`; schema /4 lacks them. A layer is fluid when
 * its `layers[].fluid` is true or its attribute is listed in
 * `meta.fluid_layers` (which must name layers of the body); a manifest
 * with neither has no fluid layers.
 */

#pragma once

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mfem.hpp"

namespace AdGIA {

/**
 * @brief A manifest read from its JSON file. Radii and lengths are in the
 * mesh's own units.
 */
class MeshManifest {
 public:
  /** @brief A layer of the mesh: element attribute, name and radii. A layer
   * outside the geometry is a shell (the buffer). */
  struct Layer {
    int attribute = 0;
    std::string name;
    mfem::real_t r_inner = 0.0, r_outer = 0.0;
    bool in_geometry = true;
    bool fluid = false;
  };

  /** @brief A boundary of the mesh: boundary attribute, name, radius and
   * the attributes of the layers below and above it (0 for the outside of
   * the mesh). `kind` classifies it from the model's fluidity
   * ("solid-solid", "fluid-solid", "fluid-fluid", "free", "outer"), empty
   * where no model has said; `values` holds, per exported scalar radial
   * field, its one-sided {below, above} values at the interface, NaN on a
   * side without one, and is empty where no model has said. */
  struct Interface {
    int attribute = 0;
    std::string name;
    mfem::real_t radius = 0.0;
    int below = 0, above = 0;
    std::string kind;
    std::map<std::string, std::pair<mfem::real_t, mfem::real_t>> values;

    /** @brief The one-sided value of the named field on the side of the
     * layer of @p attribute (aborts when the manifest holds none). */
    mfem::real_t ValueBeside(const std::string& field, int attribute) const;
  };

  /** @brief A field written beside the mesh: the file, the space to read
   * it into, and the attributes of the layers on which its values mean
   * anything (elsewhere the file holds zeros). `radial_degree` is the
   * degree in r that reproduces a scalar field within each layer holding
   * it, -1 where no model (or no single degree) has said. */
  struct Field {
    std::string name, file, fe_space, ordering, unit;
    int vdim = 1, rank = 0, weight = 0;
    bool voigt = false;
    std::vector<int> layers;
    int radial_degree = -1;
  };

  /** @brief Read the manifest at @p path; aborts with a message naming the
   * file on a missing file, malformed JSON or another schema. */
  explicit MeshManifest(const std::string& path);

  const std::string& Path() const { return path_; }

  const std::vector<Layer>& Layers() const { return layers_; }
  const std::vector<Interface>& Interfaces() const { return interfaces_; }
  const std::vector<Field>& Fields() const { return fields_; }

  /** @brief The layer, interface or field of the given name (aborts when
   * there is none). */
  const Layer& LayerNamed(const std::string& name) const;
  const Interface& InterfaceNamed(const std::string& name) const;
  const Field& FieldNamed(const std::string& name) const;
  bool HasField(const std::string& name) const;

  /// @name Attribute lists, in increasing order.
  ///@{
  /** @brief Element attributes of the solid layers of the body. */
  mfem::Array<int> SolidAttributes() const;
  /** @brief Element attributes of the fluid layers of the body. */
  mfem::Array<int> FluidAttributes() const;
  /** @brief Element attributes of the body, solid and fluid. */
  mfem::Array<int> BodyAttributes() const;
  /** @brief Element attributes of the shells outside the body. */
  mfem::Array<int> ShellAttributes() const;
  /** @brief Boundary attributes of the interfaces with a fluid layer on one
   * side and a solid layer on the other. */
  mfem::Array<int> FluidSolidInterfaces() const;
  /** @brief Boundary attributes of the interfaces bounding the fluid layer
   * of the given element attribute against a solid layer. */
  mfem::Array<int> FluidSolidInterfaces(int fluid_attribute) const;
  ///@}

  /** @brief Boundary attribute of the surface of the body: the outer
   * boundary of the outermost layer of the geometry. */
  int SurfaceAttribute() const;
  /** @brief Boundary attribute of the outer boundary of the mesh. */
  int OuterAttribute() const;
  /** @brief Radius of the surface of the body. */
  mfem::real_t SurfaceRadius() const;
  /** @brief Outer radius of the mesh, shells included. */
  mfem::real_t OuterRadius() const;

  /** @brief A 0/1 marker of length @p size with the entries of
   * @p attributes set. @p size is `attributes.Max()` or
   * `bdr_attributes.Max()` of the mesh the marker is for; an attribute
   * above it is ignored, so that a marker for a SubMesh can be made from
   * the attributes of the parent. */
  static mfem::Array<int> Marker(const mfem::Array<int>& attributes, int size);

  /// @name Units and constants.
  ///@{
  /** @brief Whether the manifest carries scales (a model was exported). */
  bool HasScales() const { return has_scales_; }
  /** @brief What one unit of length, mass and time is in SI. */
  mfem::real_t LengthScale() const { return length_; }
  mfem::real_t MassScale() const { return mass_; }
  mfem::real_t TimeScale() const { return time_; }
  /** @brief The SI size of one unit of a quantity of dimensions
   * mass^a length^b time^c. */
  mfem::real_t ScaleFactor(int mass, int length, int time) const;
  bool HasConstant(const std::string& name) const;
  /** @brief A constant of the model, in the model's units. */
  mfem::real_t Constant(const std::string& name) const;
  /** @brief The gravitational constant, in the model's units. */
  mfem::real_t G() const { return Constant("G"); }
  ///@}

  /// @name The `meta` block: numbers, strings and lists of integers.
  ///@{
  bool HasMeta(const std::string& key) const;
  mfem::real_t MetaNumber(const std::string& key) const;
  const std::string& MetaString(const std::string& key) const;
  const std::vector<int>& MetaIntegers(const std::string& key) const;
  ///@}

  /// @name Files.
  ///@{
  /** @brief The mesh file, in the manifest's directory. */
  std::string MeshFile() const;
  /** @brief Whether the mesh's nodes are physical coordinates (true) or
   * reference coordinates (false). */
  bool PhysicalNodes() const { return physical_nodes_; }
  /** @brief The mesh, opened with the manifest's read options. The dof
   * numbering of the fields is that of the mesh opened this way. */
  mfem::Mesh LoadMesh() const;
  /** @brief The named field on @p mesh, which must be LoadMesh()'s. The
   * GridFunction owns its space and collection. */
  std::unique_ptr<mfem::GridFunction> LoadField(mfem::Mesh& mesh,
                                                const std::string& name) const;
  ///@}

  /** @brief A readable summary: layers, interfaces, fields, scales. */
  void Print(std::ostream& os) const;

 private:
  std::string path_, directory_;
  std::string mesh_file_, mesh_format_;
  bool physical_nodes_ = false;
  int generate_edges_ = 1, refine_ = 1;
  bool fix_orientation_ = true;
  std::vector<Layer> layers_;
  std::vector<Interface> interfaces_;
  std::vector<Field> fields_;
  bool has_scales_ = false;
  mfem::real_t length_ = 1.0, mass_ = 1.0, time_ = 1.0;
  std::map<std::string, mfem::real_t> constants_;
  std::map<std::string, mfem::real_t> meta_numbers_;
  std::map<std::string, std::string> meta_strings_;
  std::map<std::string, std::vector<int>> meta_integers_;
};

}  // namespace AdGIA
