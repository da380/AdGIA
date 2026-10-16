# Meshes

The gmsh meshes the examples and tests read are made here (and the
aspherical meshes in MFEM's own format), with
[planetmodel](https://github.com/da380/planetmodel) driving gmsh (the
faulted box, which is not a planetary model, drives gmsh directly), and
land in the build tree's `data/` directory; none is kept in the
repository. Each script builds one kind of mesh and is short enough to
read in a minute; copy the nearest one and change the skeleton, the
sizing or the shells to make a mesh of your own.

## In the build

`CMakeLists.txt` here runs the scripts as part of the build when
`GENERATE_MESHES` is on (the default whenever examples or tests are
built). One custom command per script writes its `.msh` and `.json`
files into `<build>/data`, so a mesh is generated once and again only
when its script or `common.py` changes. A new mesh is one `meshes_add`
line there. The cache variable `MESHES_SCALE` (default 1) multiplies
every script's element sizes — `cmake -DMESHES_SCALE=0.5 <build>`
regenerates the whole set at double resolution on the next build, and
setting it back regenerates the canned set.

The Python used is `MESHES_PYTHON` if set, else the one CMake finds (3.12
or later). It must import `planetmodel.mesh3d` and `mfem.ser` (PyMFEM, which
writes the aspherical meshes) and have planetmodel 1.2.4 or later; if the
found Python does not, a virtual environment is made at
`<build>/meshes-venv` and `planetmodel[meshing,mfem]>=1.2.4` installed into
it with pip (the requirement is the cache variable
`PLANETMODEL_REQUIREMENT`).
To reuse an environment you already have:

```
cmake -S . -B build -DMESHES_PYTHON=/path/to/python ...
```

## By hand

The canned meshes are deliberately coarse, so that the build is quick
and the tests on them cheap. Any script also runs by hand — to make a
higher-resolution version of its mesh, or a variant through its own
options. Every script takes `--scale`, a factor on all
its element sizes (smaller is finer), beside its own options (`-h`
lists them). Each runs in the shared poetry environment of the
repository root (one environment for the meshes, the benchmarks and the
earth-data tools; see INSTALL.md), made once:

```
cd <repo>
poetry install                       # planetmodel[meshing,mfem]>=1.2.4 among others
poetry run python meshes/homogeneous_body.py -h    # any one script; -h lists its options
poetry run python meshes/make_all.py --out /some/dir
poetry run python meshes/layered_earth.py --all --scale 0.5 --out <build>/data
```

The last writes half-size (so roughly 4x in 2-D, 8x in 3-D) versions of
the sixteen layered-Earth meshes (four spherical, twelve aspherical) over
the canned ones in the build tree,
and the examples and tests on them then run at that resolution; the
build puts the coarse version back only when the script next changes.
`make_all.py` forwards `--scale` to every script it runs, for a finer
full set. `layered_earth.py` and `aspherical_body.py` also take
`--name`, a suffix on the file names, for keeping several versions side
by side.

From any other directory, `poetry -P <repo> run python
meshes/<script>.py ...` does the same (`-P` names the poetry project and
leaves the working directory alone, so relative paths and outputs
resolve where you are; `-C` would change into the project and break both).
If the build created a `meshes-venv` (see above), its
`<build>/meshes-venv/bin/python` runs the scripts directly too.

Python 3.12 or later. No compiler is needed: gmsh and PyMFEM (the `mfem`
extra, needed by `aspherical_body.py`) come as wheels from PyPI. A script writes to the current directory unless
given `--out`, so run from the repository root without `--out` it would put
its meshes there; the repository's `data/` holds only the three tracked
MFEM meshes, and generated `*.msh`, `*.json` and `*.mesh` files in it are
gitignored. Pass `--out <build>/data` to replace the canned meshes.

## The scripts

| script | writes | read by (examples unless stated) |
|---|---|---|
| `homogeneous_body.py --preset disc` | `disc_2d.msh` | transformed_diffusion |
| `homogeneous_body.py --preset ball` | `ball_3d.msh` | transformed_diffusion, anisotropic_elasticity |
| `offset_disc.py` | `offset_disc_2d.msh` | poisson_dtn, submesh_injection |
| `faulted_box.py --all` | `faulted_box_{2d,3d}.msh` (the coseismic box under a mountain-range topography over the hanging wall, cut by a planar fault extended to the boundary: footwall and hanging wall as two volumes, the fault patch and its extension as interior boundaries, refined toward the fault and its tips; the JSON beside it is the script's own record, not a planetmodel manifest) | coseismic_fault; tests |
| `faulted_box.py --all --topography flat --name flat` | `faulted_box_{2d,3d}_flat.msh` (the same with a flat free surface, as in coseismic_deformation) | coseismic_fault through `-m` |
| `homogeneous_body.py --preset disc-buffer` | `disc_buffer_2d.msh` | elastogravity_layered, equilibrium_stress, lateral_viscosity, love_numbers, referential_elastogravity, rotational_feedback; tests |
| `homogeneous_body.py --preset disc-buffer-wide` | `disc_buffer_wide_2d.msh` (buffer out to radius 2, for strongly non-trivial equilibrium mappings) | prestress_loading |
| `homogeneous_body.py --preset ball-buffer` | `ball_buffer_3d.msh` | gia_kernels |
| `homogeneous_body.py --preset ball-buffer-coarse` | `ball_buffer_coarse_3d.msh` | coupled_poisson, elastogravity_layered, generalised_rotation, ice_age_loading, love_numbers, referential_elastogravity, shoreline_refinement, viscoelastic_love_numbers; tests |
| `layered_earth.py --all --shape none` | `earth_fluidcore_2d.msh` | adjoint_elasticity, elastogravity_layered, equilibrium_density, gauged_fluid_cavity, relabelled_elasticity, self_gravitating_relaxation, self_gravitating_solvers, sliding_fluid_ellipse, viscoelastic_adjoint; tests |
| | `earth_innercore_2d.msh` | elastogravity_layered, equilibrium_density, self_gravitating_relaxation; tests |
| | `earth_fluidcore_3d.msh` | equilibrium_density |
| | `earth_innercore_3d.msh` | elastogravity_layered, equilibrium_density, gauged_fluid_cavity, self_gravitating_relaxation; tests |
| `layered_earth.py --all --shape shaped` | `earth_{fluidcore,innercore}_flat_{2d,3d}.mesh` (every interface flattened, `--flattening`), `earth_{fluidcore,innercore}_cmbtopo_{2d,3d}.mesh` (oscillatory CMB topography, `--amplitude`, `--degree`), `earth_{fluidcore,innercore}_cmbbump_{2d,3d}.mesh` (one Gaussian CMB bump, `--amplitude`, `--width`) | equilibrium_density (its `-shape flat\|cmb\|bump` defaults) |
| `aspherical_body.py --all` | `aspherical_{2d,3d}.mesh`, `aspherical_buffer_{2d,3d}.mesh` | any example given `-m` (the script's docstring lists runs); `aspherical_buffer_3d.mesh`: `benchmarks/relabelling/aspherical_reference.cpp` and `benchmarks/campaign.py` |
| | `aspherical_fluid_core_buffer_{2d,3d}.mesh` (fluid core and mantle, both stretched, buffer) and `spherical_fluid_core_buffer_{2d,3d}.mesh` (the same unstretched) | slipping_interface |
| `earth_coastlines.py --field earth` | `earth_coastlines.msh` (the ball in a thin buffer, surface-refined along the real shorelines of the committed lmax-64 ICE-7G topography; see below) | ice_age_loading and sea_level_fingerprint through `-m` |
| `earth_coastlines.py --field analytic` | `fingerprint_ring_3d.msh` (the same ball, refined in a ring about the analytic shoreline of the fingerprint example's default state) | sea_level_fingerprint (its default mesh) |
| `make_all.py` | runs every script above, the `--all` forms included: the set the build makes | |

The benchmarks mesh
their own models (`benchmarks/common/make_case.py`,
`benchmarks/viscoelastic/sphere/meshes.py`) and do not read these except
where listed.

`common.py` holds what the scripts share: the command line (`--out DIR`,
`--verbose`, `--scale`), the layered Earth's radii, element sizes and
geometry (`layered_earth.py` builds the spherical and aspherical layered
models on them), the sizing of a thin-buffer ball, the export of an aspherical
mesh to MFEM, and `SeaLevelBand`, the refinement of the coastline meshes.

## The coastline mesh

`earth_coastlines.py --field earth` refines the ball along the real shorelines. The
shoreline is pre-processed at export: `earth_data/topography_grid.py`
writes, beside the truncated topography, a `coast_rad` field — the
great-circle distance to the coast of the near-native-resolution
topography (its sign changes, plus the ocean within `--band` metres of
the surface, the shallow shelves where the flotation criterion is
delicate). The raw field cannot define the coast evenly (cliff coasts
and shelf seas have wildly different gradients; low-lying interiors
masquerade as coast), while the distance field has unit gradient
everywhere. The mesh is finest within `--coast-width` radians of that
coast and radially near the surface, and the size then grows at the
bounded rate `--gradation` — bounded so that the advancing front
cannot step over a fine band narrower than the background size, which
is what loses whole coastlines. No polylines are isolated: it is one
sizing field (`common.SeaLevelBand`, which `--field analytic`
reuses for the analytic shoreline ring of the fingerprint example).

The default grid is **committed beside the script**
(`ice7g_topography_lmax64.npz`: topography at lmax 64, coastline at
lmax 180, shelf band 400 m, ~430 KB), so the build generates
`data/earth_coastlines.msh` like any other mesh, with no pyslfp
anywhere in the loop. Examples take it with
`-m ../data/earth_coastlines.msh`.

For another date, shelf band or truncation, regenerate the grid with
`earth_data/topography_grid.py` (pyslfp; the one-time setup is in
`earth_data/README.md`) and point the script at it:

```
cd <build>/earth_data
./topography_grid --date 21 --plot       # writes topography.npz + a map here
cd <source>
poetry -P <repo> run python meshes/earth_coastlines.py \
    --topography <build>/earth_data/topography.npz --out <build>/data
```

The committed default grid is regenerated with

```
<build>/earth_data/topography_grid --lmax 64 \
    -o <source>/meshes/ice7g_topography_lmax64.npz
```

(the other options at their defaults: ICE7G, `--date 0`, `--shore-lmax 180`,
`--band 400`). What makes
the coasts *visually* clear in GLVis or gmsh is `--coast-size`
(default 0.035, ~220 km; halving it quadruples the surface detail at
the usual cubic volume cost), with `--coast-width` setting how far the
fine band extends.

## How a mesh is described

A **skeleton** is the list of boundary radii, from the centre outwards. A
**geometry** is a skeleton with names for its layers and interfaces. A
**shell** is a layer appended outside the geometry, used here as the buffer
region on whose outer boundary the far-field condition is applied. The
**sizing** gives every interface a target element size, a size far from
the interface and the distance over which one grows into the other. A
`MeshSpec` collects these with the dimension (2 for a disc, 3 for a ball)
and the element order, and `build_layered_mesh` does the rest: gmsh
geometry, physical groups, sizing fields, meshing, orientation, curving,
validation and the files on disk.

The offset disc is not spherically layered, so it has its own function,
`build_offset_mesh`, with the same sizing rules and the same output.

A **mapping** gives a geometry a non-spherical shape. `aspherical_body.py`
shows the pattern: a radial displacement `h(r, theta, phi)` from a
formula, wrapped in `CallableDisplacement` and attached with
`Geometry.stretched`. gmsh still meshes the sphere; `export_mfem_mesh`
moves the nodes when it writes the mesh in MFEM's own format, and refuses
a mapping that folds an element. With a buffer shell the displacement is
tapered to zero across it so that the outer boundary stays spherical for
the DtN and multipole conditions, and the surface, where the taper
starts, is declared a knot of the displacement. These meshes are
`.mesh` files and need the `mfem` extra (PyMFEM).

## Who owns what

- **planetmodel** owns the model and the mesher: skeletons, geometries and
  mappings, the sizing rules, shells, the manifest and the MFEM export.
  Anything geometric that a planetary mesh here needs and planetmodel
  lacks is a planetmodel change, not a script here. The one exception is
  `faulted_box.py`: a box with an embedded fault is not a planetary
  model, so the script drives gmsh's own Python API and writes its own
  JSON record (box, fault frame, sizes, attribute dictionary). planetmodel 1.2.4 or later is
  required: it writes manifest schema 5 and runs gmsh's linear-mesh
  optimiser before curving, which removes the slivers the Delaunay mesher
  leaves between interfaces.
- **`meshes/`** owns the recipes: which skeleton, which sizes, which
  shells, which file name.
- **The C++ side** reads the mesh files. `MeshManifest`
  (`mesh_manifest.hpp`) reads the manifest and answers questions such as
  "which attributes are solid" or "which boundary is the surface" as the
  attribute lists and markers the problem classes take; the benchmarks and
  `tests/TestMeshManifest.cpp` use it. The examples identify layers by
  position instead (for instance `attributes.Max()` tells the two- and
  three-layer Earth apart).

A realistic Earth skeleton comes from planetmodel's `PREM()`, coarsened as
needed, through a script like `layered_earth.py`.

## What the files carry

Every gmsh mesh is MSH 2.2 with physical groups, which MFEM reads
directly; the aspherical ones are MFEM's native format with curved nodes.
Domain attributes number the layers 1..N from the centre, the buffer
last; boundary attributes number the interfaces 1..M in the same order,
so attribute 1 of a boundary element is the innermost interface and the
largest boundary attribute is the outer boundary of the domain. The
elements come out consistently oriented, so MFEM has nothing to fix on
load.

Beside each mesh file is a JSON **manifest** of the same name (schema
`planetmodel.mesh.manifest/5`; `MeshManifest` also reads /4) listing the
layers and interfaces with their attributes, names and radii, and, for an
exported model, which layers are fluid, the fields written beside the mesh
and the model's units and constants. It is what a reader consults instead
of guessing what attribute 3 means.

## Sizes and quality

Element sizes are in the mesh's own units, the unit radius here. The
sizes in the scripts keep the meshes coarse, so that they build quickly and
the tests on them are cheap; `--scale` refines them uniformly (see "By
hand"). planetmodel checks every mesh before writing it and refuses one
with an inverted element; a warning is printed when the worst element is
usable but badly shaped. If a new sizing meets either, nudging the size by
a few per cent is usually enough.

Everything a script can do is documented in planetmodel; its tutorial
`04_a_mesh_for_mfem.py` walks through the same steps, including
topography through a mapping.
