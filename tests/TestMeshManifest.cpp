#include <cmath>
#include <filesystem>
#include <fstream>

#include "TestCommon.hpp"

/*
  Tests for mesh_manifest.hpp.

  - A schema-5 manifest written here beside a small mesh and a field: the
    layers, interfaces and fields come back as written; layers[].fluid
    gives the solid, fluid, body and shell attributes and the fluid-solid
    interfaces; the interfaces' kinds and one-sided values are read;
    scales, constants and meta are read; the mesh and the field are opened
    and the field has the values saved.
  - A schema-4 manifest without the model-derived records: still read, the
    fluid layers from the meta.fluid_layers fallback.
  - Marker() sets the attributes within the size and ignores those above.
  - A manifest generated with the meshes of the build (three-layer Earth):
    names, radii, the surface and the outer boundary, no fluid declared.
*/

namespace {

const char* const kManifest = R"({
  "mesh": {"file": "manifest_test.mesh", "format": "mfem",
           "nodes": "reference",
           "read_options": {"generate_edges": 1, "refine": 0,
                            "fix_orientation": false},
           "displacement": null},
  "layers": [
    {"attribute": 1, "name": "inner_core", "r_inner": 0.0, "r_outer": 0.2,
     "in_geometry": true, "fluid": false},
    {"attribute": 2, "name": "outer_core", "r_inner": 0.2, "r_outer": 0.5,
     "in_geometry": true, "fluid": true},
    {"attribute": 3, "name": "mantle", "r_inner": 0.5, "r_outer": 1.0,
     "in_geometry": true, "fluid": false},
    {"attribute": 4, "name": "buffer", "r_inner": 1.0, "r_outer": 1.2e0,
     "in_geometry": false, "fluid": null}],
  "interfaces": [
    {"attribute": 1, "name": "icb", "radius": 0.2, "between_layers": [0, 1],
     "kind": "fluid-solid", "values": {"rho": [12.9, 12.2]}},
    {"attribute": 2, "name": "cmb", "radius": 0.5, "between_layers": [1, 2],
     "kind": "fluid-solid", "values": {"rho": [11.0, 5.5]}},
    {"attribute": 3, "name": "surface", "radius": 1.0,
     "between_layers": [2, 3], "kind": "free",
     "values": {"rho": [3.4, null]}},
    {"attribute": 4, "name": "interface_4", "radius": 1.2,
     "between_layers": [3, -1], "kind": "outer", "values": null}],
  "fields": [
    {"name": "rho", "file": "manifest_test.rho.gf", "fe_space": "L2_3D_P1",
     "vdim": 1, "ordering": "byNODES", "rank": 0, "weight": 1,
     "voigt": false, "unit": "1", "layers": [1, 2, 3],
     "radial_degree": 1}],
  "scales": {"length": 6371000.0, "mass": 1.5e24, "time": 1700.0},
  "constants": {"G": 1.25},
  "meta": {"model": "a \"test\"", "factor": -2.5e-1},
  "schema": "planetmodel.mesh.manifest/5"
})";

// The same mesh under schema 4, which lacks the model-derived records: the fluid layers then come from meta.fluid_layers.
const char* const kOldManifest = R"({
  "mesh": {"file": "manifest_test.mesh", "format": "mfem",
           "nodes": "reference",
           "read_options": {"generate_edges": 1, "refine": 0,
                            "fix_orientation": false},
           "displacement": null},
  "layers": [
    {"attribute": 1, "name": "core", "r_inner": 0.0, "r_outer": 0.5,
     "in_geometry": true},
    {"attribute": 2, "name": "mantle", "r_inner": 0.5, "r_outer": 1.0,
     "in_geometry": true}],
  "interfaces": [
    {"attribute": 1, "name": "cmb", "radius": 0.5, "between_layers": [0, 1]},
    {"attribute": 2, "name": "surface", "radius": 1.0,
     "between_layers": [1, -1]}],
  "fields": [],
  "scales": null,
  "constants": {},
  "meta": {"fluid_layers": [1]},
  "schema": "planetmodel.mesh.manifest/4"
})";

// The manifest, mesh and field of a test, in a directory of its own (named
// after the test): ctest runs the tests as concurrent processes in one
// working directory, and shared file names let one test's clean-up delete
// the files another is reading. The manifest names its mesh and field
// relative to its own directory, so the names inside stay fixed.
struct Files {
  std::filesystem::path dir;
  std::string manifest, old_manifest;
  Files() {
    const auto* info = testing::UnitTest::GetInstance()->current_test_info();
    dir = std::string("manifest_test_") + info->test_suite_name() + "_" +
          info->name();
    std::filesystem::remove_all(dir);
    std::filesystem::create_directory(dir);
    manifest = (dir / "manifest_test.json").string();
    old_manifest = (dir / "manifest_test_v4.json").string();
    Mesh mesh = Mesh::MakeCartesian3D(2, 2, 2, Element::TETRAHEDRON);
    L2_FECollection fec(1, 3);
    FiniteElementSpace fes(&mesh, &fec);
    GridFunction rho(&fes);
    for (int i = 0; i < rho.Size(); i++) {
      rho[i] = 1.0 + 0.5 * i;
    }
    std::ofstream mesh_out(dir / "manifest_test.mesh");
    mesh_out.precision(16);
    mesh.Print(mesh_out);
    std::ofstream rho_out(dir / "manifest_test.rho.gf");
    rho_out.precision(16);
    rho.Save(rho_out);
    std::ofstream(manifest) << kManifest;
    std::ofstream(old_manifest) << kOldManifest;
  }
  ~Files() { std::filesystem::remove_all(dir); }
};

void ExpectArray(const Array<int>& a, std::initializer_list<int> want) {
  ASSERT_EQ(a.Size(), static_cast<int>(want.size()));
  int i = 0;
  for (const int w : want) {
    EXPECT_EQ(a[i++], w);
  }
}

}  // namespace

TEST(MeshManifest, ReadsWhatWasWritten) {
  const Files files;
  const MeshManifest m(files.manifest);

  ASSERT_EQ(m.Layers().size(), 4u);
  EXPECT_EQ(m.Layers()[1].name, "outer_core");
  EXPECT_DOUBLE_EQ(m.Layers()[1].r_inner, 0.2);
  EXPECT_DOUBLE_EQ(m.Layers()[3].r_outer, 1.2);
  EXPECT_FALSE(m.Layers()[3].in_geometry);
  EXPECT_TRUE(m.Layers()[1].fluid);
  EXPECT_FALSE(m.Layers()[0].fluid);

  ASSERT_EQ(m.Interfaces().size(), 4u);
  EXPECT_EQ(m.InterfaceNamed("cmb").attribute, 2);
  EXPECT_EQ(m.InterfaceNamed("cmb").below, 2);
  EXPECT_EQ(m.InterfaceNamed("cmb").above, 3);
  EXPECT_EQ(m.Interfaces()[3].above, 0);
  EXPECT_EQ(m.LayerNamed("mantle").attribute, 3);

  EXPECT_EQ(m.InterfaceNamed("cmb").kind, "fluid-solid");
  EXPECT_EQ(m.InterfaceNamed("interface_4").kind, "outer");
  // The fluid side of the cmb is the outer core, attribute 2, below it.
  EXPECT_DOUBLE_EQ(m.InterfaceNamed("cmb").ValueBeside("rho", 2), 11.0);
  EXPECT_DOUBLE_EQ(m.InterfaceNamed("cmb").ValueBeside("rho", 3), 5.5);
  EXPECT_TRUE(std::isnan(m.InterfaceNamed("surface").values.at("rho").second));
  EXPECT_TRUE(m.InterfaceNamed("interface_4").values.empty());

  ExpectArray(m.SolidAttributes(), {1, 3});
  ExpectArray(m.FluidAttributes(), {2});
  ExpectArray(m.BodyAttributes(), {1, 2, 3});
  ExpectArray(m.ShellAttributes(), {4});
  ExpectArray(m.FluidSolidInterfaces(), {1, 2});
  ExpectArray(m.FluidSolidInterfaces(2), {1, 2});
  EXPECT_EQ(m.FluidSolidInterfaces(1).Size(), 0);
  EXPECT_EQ(m.SurfaceAttribute(), 3);
  EXPECT_EQ(m.OuterAttribute(), 4);
  EXPECT_DOUBLE_EQ(m.SurfaceRadius(), 1.0);
  EXPECT_DOUBLE_EQ(m.OuterRadius(), 1.2);

  EXPECT_TRUE(m.HasScales());
  EXPECT_DOUBLE_EQ(m.LengthScale(), 6371000.0);
  EXPECT_DOUBLE_EQ(m.TimeScale(), 1700.0);
  // a density: mass length^-3
  EXPECT_DOUBLE_EQ(m.ScaleFactor(1, -3, 0), 1.5e24 / std::pow(6371000.0, 3));
  EXPECT_DOUBLE_EQ(m.G(), 1.25);
  EXPECT_FALSE(m.HasConstant("c"));

  EXPECT_EQ(m.MetaString("model"), "a \"test\"");
  EXPECT_DOUBLE_EQ(m.MetaNumber("factor"), -0.25);
  EXPECT_FALSE(m.HasMeta("absent"));

  EXPECT_TRUE(m.HasField("rho"));
  EXPECT_FALSE(m.HasField("mu"));
  EXPECT_EQ(m.FieldNamed("rho").fe_space, "L2_3D_P1");
  EXPECT_EQ(m.FieldNamed("rho").layers.size(), 3u);
  EXPECT_EQ(m.FieldNamed("rho").radial_degree, 1);
  EXPECT_FALSE(m.PhysicalNodes());
}

TEST(MeshManifest, OldSchemaWithMetaFallback) {
  const Files files;
  const MeshManifest m(files.old_manifest);
  ExpectArray(m.FluidAttributes(), {1});
  ExpectArray(m.SolidAttributes(), {2});
  ExpectArray(m.FluidSolidInterfaces(), {1});
  EXPECT_TRUE(m.InterfaceNamed("cmb").kind.empty());
  EXPECT_TRUE(m.InterfaceNamed("cmb").values.empty());
  EXPECT_FALSE(m.HasScales());
}

TEST(MeshManifest, OpensTheMeshAndTheFields) {
  const Files files;
  const MeshManifest m(files.manifest);
  Mesh mesh = m.LoadMesh();
  EXPECT_EQ(mesh.GetNE(), 48);
  const auto rho = m.LoadField(mesh, "rho");
  ASSERT_EQ(rho->Size(), 4 * 48);
  for (int i = 0; i < rho->Size(); i++) {
    EXPECT_DOUBLE_EQ((*rho)[i], 1.0 + 0.5 * i);
  }
}

TEST(MeshManifest, Marker) {
  const Array<int> marker = MeshManifest::Marker(Array<int>({1, 3, 5}), 4);
  ExpectArray(marker, {1, 0, 1, 0});
}

TEST(MeshManifest, GeneratedManifest) {
  const MeshManifest m("../data/elastogravity_three_layer_3d.json");
  ASSERT_EQ(m.Layers().size(), 4u);
  EXPECT_EQ(m.Layers()[0].name, "inner_core");
  EXPECT_EQ(m.InterfaceNamed("surface").attribute, m.SurfaceAttribute());
  EXPECT_DOUBLE_EQ(m.SurfaceRadius(), 1.0);
  EXPECT_NEAR(m.InterfaceNamed("cmb").radius, 3483.0 / 6371.0, 1e-14);
  EXPECT_EQ(m.OuterAttribute(), 4);
  EXPECT_EQ(m.FluidAttributes().Size(), 0);
  EXPECT_FALSE(m.HasScales());
  Mesh mesh = m.LoadMesh();
  EXPECT_EQ(mesh.attributes.Max(), 4);
  EXPECT_EQ(mesh.bdr_attributes.Max(), 4);
}
