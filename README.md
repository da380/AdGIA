# AdGIA

[![CI](https://github.com/da380/AdGIA/actions/workflows/ci.yml/badge.svg)](https://github.com/da380/AdGIA/actions/workflows/ci.yml)

An open-source finite-element library for modelling glacial isostatic
adjustment (GIA) and related processes of the solid Earth, such as
post-seismic deformation. It is built on [MFEM](https://mfem.org) and aims to
provide both the forward problem and its adjoints, to first and second order,
so that the sensitivity of observables to the Earth model, the load or the
source can be computed and used in inversion.

**Status.** The solid-Earth part of the forward problem is in place:
quasi-static deformation of a self-gravitating, elastic or viscoelastic body,
with fluid regions, laterally varying and anisotropic structure, general
(non-hydrostatic, non-natural) reference states and slipping fluid–solid
interfaces, in serial and in parallel (MPI). Current work is on benchmarking
it against independent reference solutions (`benchmarks/`,
`doc/benchmarks.tex`) and on refining its solvers. Planned for the near
future, and not yet in the code, are the sea-level equation and rotational
feedbacks, which complete the GIA forward problem, and the first- and
second-order adjoint machinery.

## Library contents

The main pieces, by header in `include/AdGIA/` (`include/AdGIA.hpp`
includes them all), are

- the component ordering and node/component indexing of vector, matrix and
  symmetric tensor fields (`index.hpp`);
- integrators (`bilininteg.hpp`, `lininteg.hpp`): mixed bilinear/linear form
  integrators between vector, scalar and tensor nodal spaces and the strain
  interpolators; a general (anisotropic) elasticity integrator; the
  initial-stress stiffness split (material and geometric stiffness) and the
  referential gravity terms; the boundary integrators of fluid–solid and
  slipping interfaces; and the transformed diffusion integrator for Poisson
  problems on a reference domain. Most take an optional mapping and then
  assemble the pulled-back form. General-purpose coefficients are in
  `coefficient.hpp`;
- elastic tensor coefficients in the Mandel convention (`elastic_tensor.hpp`):
  isotropic, transversely isotropic (radially anisotropic), Voigt-matrix,
  rotated and deviatoric-projection tensors, the relabelling transformation of
  a tensor, and the conversion of seismological (PREM) moduli to the bare
  tensor of the general theory;
- the exterior Poisson machinery: a matrix-free Dirichlet-to-Neumann operator
  on a spherical outer boundary and multipole operators (`poisson.hpp`), built
  on real orthonormal harmonics on a circle or sphere, with synthesis of
  fields from coefficients and analysis of a finite-element field (scalar, or
  the radial or tangential part of a vector) on any spherical boundary
  (`spherical_harmonics.hpp`), and mesh queries (`mesh.hpp`);
- coupling of forms between a mesh and its `SubMesh`es through a signed dof
  injection, and the dof pairing of sibling SubMeshes (`submesh.hpp`);
- a linear quasi-static problem interface with traction and clamped
  reference implementations and the gauge penalty of gauged fluid regions
  (`quasi_static_problem.hpp`); elastic and generalised Maxwell rheologies,
  isotropic or anisotropic, composite by region, with optional
  state-dependent relaxation (`rheology.hpp`, `relaxation_law.hpp`); and a
  viscoelastic time-dependent operator with explicit, implicit and
  exponential time stepping (`viscoelastic.hpp`);
- the self-gravitating problems, which implement the same interface so the
  viscoelastic layer runs on them unchanged: the mixed (Eulerian-potential)
  problem with Dahlen or gauged fluid regions (`mixed_problem.hpp`), and the
  fully referential problem about a general reference state, with its
  slipping-interface variant (`referential_problem.hpp`);
- background states for the referential problems (`background.hpp`): the
  hydrostatic state of a radial model, its relabelled description, and
  minimum-norm and minimum-deviatoric equilibrium stress fields for
  aspherical bodies;
- the mapping (relabelling) layer (`mappings.hpp`): diffeomorphisms of the
  reference domain (identity, analytic, radial, tapered, grid-function),
  their interpolation, mapped meshes, and pull-back and Nanson coefficients;
- rigid-body and general null-space projectors and projected Krylov solvers
  for singular systems (`null_space.hpp`);
- the manifest that planetmodel writes beside a mesh, read into the
  attribute lists and markers the problems take, with the mesh and the
  fields of the model opened as it says (`mesh_manifest.hpp`).

Serial and parallel (MPI) paths are provided throughout.
`examples/love_numbers.cpp` shows the self-gravitating machinery end to end:
load and tidal Love numbers read off one solve per degree.

## Documentation

The API is documented in the headers (Doxygen; build with `BUILD_DOCS`). The
documents in the source tree are of two kinds.

**Reference** — how the library works and why:

| Document | Subject |
|---|---|
| `doc/quasi_static_models.tex` (PDF beside it) | every problem class: model, assumptions, discretisation, solvers, verification |
| `doc/gravitating_elasticity.md` | the linearised theory of self-gravitating, pre-stressed elasticity behind the referential classes and the background module |
| `doc/self_gravitation.md` | the mixed problem with Dahlen fluid regions: weak form, fluid–solid interface conditions and CMB approximations, solvers, null space, 2-D caveats, verification |
| `doc/gauged_fluid.md` | the gauged treatment of fluid regions |
| `doc/slip_interface.tex` (PDF) | the slipping fluid–solid interface: derivation, discretisation, constraint enforcement, implementation, verification |
| `doc/gauge_penalty_iteration.tex` (PDF) | gauge penalties and their iterated (Tikhonov) refinement |
| `doc/mappings.md` | the mapping (relabelling) layer: pulled-back forms, assembly recipe, change-of-variables identity |
| `doc/submesh_coupling.md` | forms between a mesh and its SubMesh: the dof injection, its parallel construction, constraints |
| `doc/viscoelasticity.md` | the quasi-static problem interface, rheologies, time stepping, strain maps, state-dependent relaxation, composite rheologies |
| `doc/elastic_tensors.md` | the Mandel convention, the elastic tensor coefficients and the anisotropic integrator |
| `doc/null_space.md` | projected solvers for singular systems, the two gauges, element order on curved meshes |
| `doc/mfem_notes.md` | MFEM facts and pitfalls |
| `doc/benchmarks.tex` (PDF) | the benchmark families and their results |
| `doc/BenchmarkPapers/code_survey.md` | a survey of the published 3-D GIA codes and benchmarks, set against the library's treatments |
| `INSTALL.md` | prerequisites, configuration, build options |
| `examples/README.md` | how to run the examples, and what each one does |
| `meshes/README.md` | mesh generation, the files written and the attribute conventions |
| `benchmarks/README.md` | the benchmark families: comparisons with independent reference solutions |

**Planning** — open issues and future work, not a description of the code:
`doc/planning/`, indexed by its `README.md` (`open_issues.md`, `solvers.md`,
`future_work.md`, `equilibrium_figures.md`).

## Installation

MFEM must be built first (a parallel MFEM, with hypre and METIS, for the MPI
build). The project uses CMake; in-source builds are refused.

**Serial:**
```bash
cmake -S . -B build_serial \
      -DCMAKE_PREFIX_PATH=/path/to/mfem_serial_build \
      -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON
cmake --build build_serial -j
```

**Parallel:**
```bash
cmake -S . -B build_parallel \
      -DUSE_MPI=ON \
      -DCMAKE_PREFIX_PATH=/path/to/mfem_parallel_build \
      -DCMAKE_CXX_COMPILER=/path/to/mpic++ \
      -DMPI_C_COMPILER=/path/to/mpicc \
      -DMPI_CXX_COMPILER=/path/to/mpic++ \
      -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON
cmake --build build_parallel -j
```

`INSTALL.md` is the reference for the build: it lists every option
(`USE_MPI`, `BUILD_EXAMPLES`, `BUILD_TESTS`, `BUILD_BENCHMARKS`,
`BUILD_DOCS`, `GENERATE_MESHES`, and the Python interpreters used by the
mesh and benchmark scripts) with its default, explains how the MPI launcher
for the tests is chosen, and describes installation.

## Examples

The programs in `examples/` demonstrate the library on small problems, most
of them checked against an exact solution, a closed form or a second
formulation. They are run from the build's `examples/` directory, show
their fields in [GLVis](https://glvis.org) and write their curves as CSV
tables plotted by the `plot_csv.py` script the build puts beside them.
**`examples/README.md`** explains how to run them and look at the results,
and describes each one.

### Meshes

The gmsh meshes the examples and tests read are not in the repository. They
are generated into the build's `data/` directory, at build time, by the
Python scripts in `meshes/`, which use the
[planetmodel](https://pypi.org/project/planetmodel/) package to drive gmsh.
The scripts are short and meant to be copied and changed; `meshes/README.md`
lists them, the files they write and the attribute conventions. Each mesh
file comes with a JSON manifest beside it saying which attribute is which
layer or interface and at what radius. `meshes/aspherical_body.py` shows how
a body gets a non-spherical shape from a formula, with and without a buffer
shell, and any example accepts the result through `-m`.

Generation is on whenever examples or tests are built (`GENERATE_MESHES`,
default follows those two options). It needs a Python 3.12 or later that can
import `planetmodel.mesh3d` (planetmodel 1.2.3 or later) and `mfem.ser`
(PyMFEM, through which the aspherical meshes are written). CMake uses the one
named by `MESHES_PYTHON` if given, otherwise the Python it finds, and if that
does not qualify it creates a virtual environment `meshes-venv` under the
build directory and installs `planetmodel[meshing,mfem]>=1.2.3` into it (about
100 MB, once per build directory). A fresh build spends a few minutes
generating meshes, most of it on the 3-D ones; later builds regenerate a mesh
only when its script changes.
