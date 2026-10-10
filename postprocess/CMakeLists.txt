# Post-processing: Python scripts that live in the source tree and run
# from the build tree (the house pattern of meshes/ and benchmarks/) —
# <build>/postprocess/<script> starts the script of the source tree with
# this directory's poetry environment, so outputs land in the build.
# POSTPROCESS_PYTHON names a Python with numpy, scipy and netCDF4; when
# empty, the poetry environment of this directory is looked for.

set(POSTPROCESS_PYTHON "" CACHE FILEPATH
    "Python interpreter with numpy, scipy and netCDF4; the poetry environment of postprocess/ if empty")
if(NOT POSTPROCESS_PYTHON)
    find_program(POETRY_EXECUTABLE poetry)
    if(POETRY_EXECUTABLE)
        execute_process(
            COMMAND "${POETRY_EXECUTABLE}" -C "${CMAKE_CURRENT_SOURCE_DIR}"
                    env info -p
            OUTPUT_VARIABLE _venv OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET RESULT_VARIABLE _result)
        if(_result EQUAL 0 AND EXISTS "${_venv}/bin/python")
            set(POSTPROCESS_PYTHON "${_venv}/bin/python" CACHE FILEPATH
                "Python interpreter with numpy, scipy and netCDF4; the poetry environment of postprocess/ if empty"
                FORCE)
        endif()
    endif()
endif()
if(POSTPROCESS_PYTHON)
    message(STATUS "Post-processing: scripts run with ${POSTPROCESS_PYTHON}")
else()
    message(STATUS
        "Post-processing: no Python environment found; run `poetry install` "
        "in ${CMAKE_CURRENT_SOURCE_DIR} and configure again, or pass "
        "-DPOSTPROCESS_PYTHON=<python>; the launchers fall back on `python3`.")
    set(POSTPROCESS_PYTHON "python3")
endif()

# postprocess_launcher(<script>): <build>/postprocess/<script> starting
# <source>/postprocess/<script>.py.
function(postprocess_launcher script)
    set(POSTPROCESS_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/${script}.py")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/launcher.sh.in"
                   "${CMAKE_CURRENT_BINARY_DIR}/${script}"
                   @ONLY
                   FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                                    GROUP_READ GROUP_EXECUTE
                                    WORLD_READ WORLD_EXECUTE)
endfunction()

postprocess_launcher(surface_to_netcdf)
postprocess_launcher(ice_ng_to_surface)
