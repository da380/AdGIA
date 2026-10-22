# Benchmarks: drivers that solve a case written by the Python scripts beside
# them and write results for comparison with a reference solution. They are
# parallel programs and are built with USE_MPI only.
#
# The build makes the drivers and, beside them, launchers of the scripts:
# <build>/benchmarks/<benchmark>/run and plot start the scripts of the source
# tree with the Python environment, the drivers and the MPI launcher of this
# build, so that a benchmark is run from the build tree and writes there
# (see README.md here). BENCHMARKS_PYTHON names a Python with planetmodel,
# pyslfp and matplotlib; when empty, the root poetry environment
# (ADGIA_POETRY_PYTHON, found by the top-level CMakeLists) is used.

if(NOT USE_MPI)
    message(STATUS "Benchmarks: skipped (they need USE_MPI)")
    return()
endif()

set(BENCHMARKS_PYTHON "" CACHE FILEPATH
    "Python interpreter with planetmodel, pyslfp and matplotlib; the root poetry environment if empty")
if(NOT BENCHMARKS_PYTHON)
    set(BENCHMARKS_PYTHON "${ADGIA_POETRY_PYTHON}")
endif()
set(BENCHMARKS_PYTHON_FALLBACK 0)
if(BENCHMARKS_PYTHON)
    message(STATUS "Benchmarks: scripts run with ${BENCHMARKS_PYTHON}")
else()
    message(WARNING
        "Benchmarks: no Python environment found for the scripts. Run "
        "`poetry install` at the repository root (with "
        "`poetry env use python3.12` first if your python3 is older) and "
        "configure again, or pass -DBENCHMARKS_PYTHON=<python>; the "
        "launchers fall back on `python3` and refuse to run until then.")
    set(BENCHMARKS_PYTHON "python3")
    set(BENCHMARKS_PYTHON_FALLBACK 1)
endif()

# The benchmarks are organised by family, one sub-folder each with its
# own scripts and outputs (see README.md); common/ holds what they
# share. The drivers of every family collect in <build>/benchmarks/bin,
# and each family's launchers sit in <build>/benchmarks/<family>/, meant
# to be started there so that the outputs land beside them. The master
# script's launcher is <build>/benchmarks/campaign.

# benchmark_driver(<family> <driver>): the program <family>/<driver>.cpp,
# built into <build>/benchmarks/bin with common/ on its include path.
function(benchmark_driver family driver)
    add_executable(${driver} ${family}/${driver}.cpp)
    target_link_libraries(${driver} AdGIA)
    target_include_directories(${driver} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/common")
    set_target_properties(${driver} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
endfunction()

# benchmark_launcher(<source> <destination> <script> [<argument>...]):
# the launcher <build>/benchmarks/<destination>/<script> of the script
# <source>/<script>.py, with the given arguments before the user's.
function(benchmark_launcher source destination script)
    string(JOIN " " BENCHMARK_ARGUMENTS ${ARGN})
    set(BENCHMARK_SCRIPT
        "${CMAKE_CURRENT_SOURCE_DIR}/${source}/${script}.py")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/launcher.sh.in"
                   "${CMAKE_CURRENT_BINARY_DIR}/${destination}/${script}"
                   @ONLY
                   FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                                    GROUP_READ GROUP_EXECUTE
                                    WORLD_READ WORLD_EXECUTE)
endfunction()

set(BENCHMARK_LAUNCHER_ARGUMENTS
    "--programs \"${CMAKE_CURRENT_BINARY_DIR}/bin\""
    "--mpiexec \"${MPIEXEC_EXECUTABLE}\"")

# --- Love numbers -----------------------------------------------------------

foreach(driver love_benchmark field_benchmark partition_case)
    benchmark_driver(love_numbers ${driver})
endforeach()
benchmark_launcher(love_numbers love_numbers run
    ${BENCHMARK_LAUNCHER_ARGUMENTS})
benchmark_launcher(love_numbers love_numbers plot)
benchmark_launcher(love_numbers love_numbers cmb_report)
benchmark_launcher(common love_numbers make_case)

# --- Elastic, non-gravitating (E1 of the cross-code hierarchy) --------------

# The exchange files this driver writes record which commit computed them.
execute_process(COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE ADGIA_GIT_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(NOT ADGIA_GIT_COMMIT)
    set(ADGIA_GIT_COMMIT "unknown")
endif()
benchmark_driver(elastic/nongrav traction_benchmark)
target_compile_definitions(traction_benchmark PRIVATE
    ADGIA_GIT_COMMIT="${ADGIA_GIT_COMMIT}")
benchmark_launcher(elastic/nongrav elastic/nongrav reference)
benchmark_launcher(elastic/nongrav elastic/nongrav compare)
# pyslfp_reference.py runs from the pyslfp checkout's own environment
# (see its docstring), so it gets no launcher here.

# --- Elastic, gravitating (E2): conventional Love numbers across the codes --

benchmark_launcher(elastic/grav elastic/grav from_results)
benchmark_launcher(elastic/nongrav elastic/grav compare)
# pyslfp_reference.py again runs from the pyslfp checkout.

# --- Disc (the 2-D family; rung 0 is gravity-free) --------------------------

benchmark_driver(sea_level sea_level_benchmark)
benchmark_launcher(sea_level sea_level run ${BENCHMARK_LAUNCHER_ARGUMENTS})
benchmark_launcher(common sea_level make_case)

benchmark_driver(disc disc_benchmark)
benchmark_launcher(disc disc check ${BENCHMARK_LAUNCHER_ARGUMENTS})
benchmark_launcher(disc disc explore)

# --- Relabelling ------------------------------------------------------------

foreach(driver relabelled_identity aspherical_reference)
    benchmark_driver(relabelling ${driver})
endforeach()
benchmark_launcher(relabelling relabelling plot)

# --- Perturbation -----------------------------------------------------------

benchmark_launcher(perturbation perturbation perturbation_check
    ${BENCHMARK_LAUNCHER_ARGUMENTS})
benchmark_launcher(perturbation perturbation plot)

# --- Viscoelastic -----------------------------------------------------------

# Four sub-families, each with its launchers in
# <build>/benchmarks/viscoelastic/<sub-family>/ (viscoelastic/README.md):
# the stepper survey on the beam example, the gravitating Love-number
# histories against pyslfp, and the non-gravitating box and sphere
# benchmarks with exact references.

benchmark_launcher(viscoelastic/stepping viscoelastic/stepping survey)

benchmark_driver(viscoelastic/love viscoelastic_love)
benchmark_launcher(viscoelastic/love viscoelastic/love laplace_reference)
benchmark_launcher(viscoelastic/love viscoelastic/love compare)
benchmark_launcher(common viscoelastic/love make_case)
# The V1 cross-code flow: the JSON-to-exchange converter; exchange files
# are compared with elastic/nongrav's `compare` launcher, and
# pyslfp_histories.py runs from the pyslfp checkout.
benchmark_launcher(viscoelastic/love viscoelastic/love from_results)

benchmark_driver(viscoelastic/box viscoelastic_box)
benchmark_launcher(viscoelastic/box viscoelastic/box cases)
benchmark_launcher(viscoelastic/box viscoelastic/box reference)
benchmark_launcher(viscoelastic/box viscoelastic/box compare)
benchmark_launcher(viscoelastic/box viscoelastic/box study
    ${BENCHMARK_LAUNCHER_ARGUMENTS})

benchmark_driver(viscoelastic/sphere viscoelastic_sphere)
benchmark_launcher(viscoelastic/sphere viscoelastic/sphere meshes)
benchmark_launcher(viscoelastic/sphere viscoelastic/sphere cases)
benchmark_launcher(viscoelastic/sphere viscoelastic/sphere reference)
benchmark_launcher(viscoelastic/sphere viscoelastic/sphere study
    ${BENCHMARK_LAUNCHER_ARGUMENTS})

# --- The master script ------------------------------------------------------

benchmark_launcher(. . campaign ${BENCHMARK_LAUNCHER_ARGUMENTS})
benchmark_launcher(. . collect)
