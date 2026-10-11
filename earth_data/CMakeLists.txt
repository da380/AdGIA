# Earth data: Python scripts that live in the source tree and run from
# the build tree (the house pattern of meshes/ and benchmarks/) —
# <build>/earth_data/<script> starts the script of the source tree with
# the shared Python environment, so outputs land in the build.
# EARTH_DATA_PYTHON names a Python with numpy, scipy, netCDF4 and
# pyslfp; when empty, the root poetry environment (ADGIA_POETRY_PYTHON,
# found by the top-level CMakeLists) is used.

set(EARTH_DATA_PYTHON "" CACHE FILEPATH
    "Python interpreter with numpy, scipy, netCDF4 and pyslfp; the root poetry environment if empty")
if(NOT EARTH_DATA_PYTHON)
    set(EARTH_DATA_PYTHON "${ADGIA_POETRY_PYTHON}")
endif()
set(EARTH_DATA_PYTHON_FALLBACK 0)
if(EARTH_DATA_PYTHON)
    message(STATUS "Earth data: scripts run with ${EARTH_DATA_PYTHON}")
else()
    message(STATUS
        "Earth data: no Python environment found; run `poetry install` at "
        "the repository root (with `poetry env use python3.12` first if "
        "your python3 is older) and configure again, or pass "
        "-DEARTH_DATA_PYTHON=<python>; the launchers fall back on `python3` "
        "and refuse to run until then.")
    set(EARTH_DATA_PYTHON "python3")
    set(EARTH_DATA_PYTHON_FALLBACK 1)
endif()

# earth_data_launcher(<script>): <build>/earth_data/<script> starting
# <source>/earth_data/<script>.py.
function(earth_data_launcher script)
    set(EARTH_DATA_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/${script}.py")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/launcher.sh.in"
                   "${CMAKE_CURRENT_BINARY_DIR}/${script}"
                   @ONLY
                   FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                                    GROUP_READ GROUP_EXECUTE
                                    WORLD_READ WORLD_EXECUTE)
endfunction()

earth_data_launcher(surface_to_netcdf)
earth_data_launcher(ice_ng_to_surface)
earth_data_launcher(topography_grid)
