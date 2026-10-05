# Installing AdGIA

## Prerequisites

1. **CMake** 3.15 or later.
2. **A C++20 compiler** (GCC 10, Clang 11 or later).
3. **MFEM**, built or installed. For the parallel build, a parallel MFEM
   (with hypre and METIS). See the [MFEM build
   instructions](https://mfem.org/building/). CMake needs the directory
   holding `MFEMConfig.cmake`, normally the MFEM build or install directory.
   The continuous integration tests against MFEM 4.10 and hypre 3.1.0;
   `.github/ci/build_mfem.sh serial|parallel <prefix>` builds and installs
   that MFEM (and hypre) with the MPI compiler wrappers it is given, and
   works on a workstation too.
4. **MPI** (optional), for the parallel build: the same MPI that hypre and
   MFEM were built with, for the compiler and for `mpiexec`.
5. **Python 3.12 or later** (examples and tests only). The gmsh meshes the
   examples and tests read are generated at build time by the scripts in
   `meshes/`, which need a Python that can import `planetmodel.mesh3d`, with
   planetmodel 1.2.3 or later, and `mfem.ser` (PyMFEM, used to write the
   aspherical meshes) — i.e. the package `planetmodel[meshing,mfem]>=1.2.3`.
   If the Python that CMake finds does not qualify, a virtual environment
   `meshes-venv` is created under the build directory and the package
   installed there with pip. Pass `-DMESHES_PYTHON=<python>` to use an
   environment of your own, or `-DGENERATE_MESHES=OFF` to skip generation.
   See `meshes/README.md`.
6. **Benchmarks** (optional, MPI builds only): the benchmark scripts need a
   Python with planetmodel, pyslfp and matplotlib, normally the poetry
   environment of `benchmarks/` (`poetry install` there). See
   `benchmarks/README.md`.
7. **Doxygen** (optional), for the API documentation.

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
| `BENCHMARKS_PYTHON` | the poetry environment of `benchmarks/` | the Python interpreter the benchmark launchers use; falls back to `python3`, with a warning, when neither is available |
| `BUILD_DOCS` | `OFF` | generate the Doxygen documentation (HTML in `<build>/doc/html`) as part of the build |
| `GENERATE_MESHES` | on with examples or tests | generate the gmsh meshes into `<build>/data` |
| `MESHES_PYTHON` | found or created | the Python interpreter used for mesh generation |

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
