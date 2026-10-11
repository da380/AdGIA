# Installing AdGIA

## Building a minimal MFEM first

AdGIA needs nothing of MFEM beyond its core: no PETSc, no SUNDIALS, no
GPU backends. The quickest route is the script the CI uses, which runs
equally well on a workstation:

```bash
# serial: MFEM alone
.github/ci/install_mfem.sh serial  ~/mfem-serial

# parallel: hypre + MFEM with one MPI's wrappers; METIS from the system
sudo apt install libmetis-dev        # or set METIS_DIR
MPICC=mpicc MPICXX=mpicxx .github/ci/install_mfem.sh parallel ~/mfem-parallel
```

It builds the reference versions (MFEM v4.10, hypre 3.1.0; overridable
through `MFEM_VERSION`/`HYPRE_VERSION`) into the given prefix, which is
then the `MFEM_DIR` below. By hand, the minimum is:

- **serial** — MFEM with its defaults:
  `cmake -S mfem -B mfem/build -DCMAKE_BUILD_TYPE=Release` and build;
- **parallel** — hypre and MFEM compiled with the *same* MPI compiler
  wrappers (mixing MPIs is the usual cause of link and launch
  failures), plus a serial METIS 5:
  `cmake -S mfem -B mfem/build -DMFEM_USE_MPI=YES
  -DCMAKE_CXX_COMPILER=mpicxx -DHYPRE_DIR=<hypre prefix>
  -DMETIS_DIR=/usr`.

MFEM v4.10 is the reference (what the CI tests against); v4.9 is also
known to work — the full test suite passes on both.

## Prerequisites

1. **CMake** 3.15 or later.
2. **A C++20 compiler** (GCC 10, Clang 11 or later).
3. **MFEM**, built or installed. For the parallel build, a parallel MFEM
   (with hypre and METIS). See the [MFEM build
   instructions](https://mfem.org/building/). CMake needs the directory
   holding `MFEMConfig.cmake`, normally the MFEM build or install directory.
   The continuous integration tests against MFEM 4.10 and hypre 3.1.0;
   `.github/ci/install_mfem.sh serial|parallel <prefix>` builds and installs
   that MFEM (and hypre) with the MPI compiler wrappers it is given, and
   works on a workstation too.
4. **MPI** (optional), for the parallel build: the same MPI that hypre and
   MFEM were built with, for the compiler and for `mpiexec`.
5. **Python 3.12 or later** (examples, tests, benchmarks and earth-data
   tools only). All of the Python tooling — the mesh generation in
   `meshes/`, the benchmark scripts in `benchmarks/` (MPI builds only)
   and the earth-data tools in `earth_data/` — shares **one**
   environment, the poetry project of the repository root, made once:

   ```bash
   cd <repo>
   poetry env use python3.12   # only if your python3 is older than 3.12
   poetry install              # planetmodel (gmsh, PyMFEM), pyslfp,
                               # matplotlib, netCDF4, ...
   ```

   Poetry needs a Python ≥ 3.12 and does not pick one up by itself: on a
   machine whose `python3` is older, the `poetry env use` line is needed
   first — without it `poetry install` reports the mismatch and creates
   nothing. The configure step finds the environment (the line
   `Python tooling: the poetry environment ...`), so a build directory
   configured before the environment existed is configured again
   (`cmake .` there) to pick it up; until then the benchmark and
   earth-data launchers refuse to run, with this recipe in their message.
   `-DMESHES_PYTHON`, `-DBENCHMARKS_PYTHON` and `-DEARTH_DATA_PYTHON`
   each override the shared environment with one of your own.

   Mesh generation alone does not need poetry: without the environment, a
   qualifying Python that CMake finds is used, and failing that a virtual
   environment `meshes-venv` is created under the build directory and
   `planetmodel[meshing,mfem]>=1.2.3` installed there with pip
   (`-DGENERATE_MESHES=OFF` skips generation). See `meshes/README.md`,
   `benchmarks/README.md` and `earth_data/README.md`.
6. **Doxygen** (optional), for the API documentation.

## Configure

In-source builds are refused; configure into a separate directory.

Serial:

```bash
cmake -S . -B build_serial \
      -DMFEM_DIR=/path/to/mfem_serial_build \
      -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON
```

Parallel:

```bash
cmake -S . -B build_parallel \
      -DUSE_MPI=ON \
      -DMFEM_DIR=/path/to/mfem_parallel_build \
      -DCMAKE_CXX_COMPILER=/path/to/mpic++ \
      -DMPI_C_COMPILER=/path/to/mpicc \
      -DMPI_CXX_COMPILER=/path/to/mpic++ \
      -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON
```

`CMAKE_PREFIX_PATH` may be given instead of `MFEM_DIR`. Naming the MPI
compiler wrappers matters when several MPI installations are present (use
those MFEM was built with). The launcher for the MPI tests and benchmarks is
`MPIEXEC_EXECUTABLE` if set; otherwise the `mpiexec` in the directory of the
C++ compiler (`CMAKE_CXX_COMPILER`) if there is one; otherwise whatever
CMake's `FindMPI` finds. Setting the C++ compiler to the MPI wrapper, as
above, therefore picks the `mpiexec` of the same MPI. `CMAKE_CXX_COMPILER`
takes effect only on a fresh build directory.

| Option | Default | Effect |
|---|---|---|
| `USE_MPI` | `OFF` | build against a parallel MFEM; enables the parallel classes, examples and tests |
| `BUILD_EXAMPLES` | `OFF` | build the programs in `examples/` |
| `BUILD_TESTS` | `OFF` | build the test suite in `tests/` (googletest is fetched at configure time) |
| `BUILD_BENCHMARKS` | `OFF` | build the benchmark drivers and the launchers of their scripts into `<build>/benchmarks`; needs `USE_MPI` (skipped with a message otherwise) |
| `BENCHMARKS_PYTHON` | the root poetry environment | the Python interpreter the benchmark launchers use; falls back to `python3`, with a warning, when neither is available |
| `EARTH_DATA_PYTHON` | the root poetry environment | the Python interpreter the earth-data launchers use; falls back to `python3`, with a message, when neither is available |
| `BUILD_DOCS` | `OFF` | generate the Doxygen documentation (HTML in `<build>/doc/html`) as part of the build |
| `GENERATE_MESHES` | on with examples or tests | generate the gmsh meshes into `<build>/data` |
| `MESHES_PYTHON` | the root poetry environment, else found or created | the Python interpreter used for mesh generation |
| `MESHES_SCALE` | `1` | factor on every generated mesh's element sizes (smaller is finer); changing it regenerates the whole set |

The usual CMake variables (`CMAKE_BUILD_TYPE`, `CMAKE_INSTALL_PREFIX`)
apply.

## Build, test, install

```bash
cmake --build build_serial -j 8
(cd build_serial && ctest)             # if BUILD_TESTS=ON
cmake --install build_serial           # optional
```

A first build with examples or tests spends a few minutes generating
meshes; later builds regenerate a mesh only when its script changes. The MPI
tests run at 1, 2 and 4 ranks.

Installation puts the library in `lib/`, the headers in `include/`, and the
CMake package files in `lib/cmake/AdGIA`. With `BUILD_DOCS` the
generated documentation directory `<build>/doc` is installed as
`share/AdGIA/html/doc` (the HTML pages in its `html/`
subdirectory).

## Using the installed library

```cmake
find_package(AdGIA REQUIRED)
target_link_libraries(my_program PRIVATE AdGIA)
```
