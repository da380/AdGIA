# The gmsh meshes the examples and tests read are generated here, into the
# build tree's data/ directory, by the Python scripts in this directory
# (planetmodel driving gmsh). Nothing is written to the source tree.
#
# A Python able to `import planetmodel.mesh3d` and `import mfem.ser` (the
# aspherical meshes are written through PyMFEM) is needed, with planetmodel
# at least PLANETMODEL_MINIMUM_VERSION (1.2.3 first wrote manifest schema 5
# and optimised the linear volume mesh before curving). MESHES_PYTHON
# names one; otherwise the Python found by CMake is tried, and failing
# that a virtual environment is created under the build directory and
# planetmodel[meshing,mfem] is installed into it with pip (once).

set(MESHES_PYTHON "" CACHE FILEPATH
    "Python interpreter with planetmodel[meshing]; found or created if empty")
set(PLANETMODEL_MINIMUM_VERSION "1.2.3")
set(PLANETMODEL_REQUIREMENT
    "planetmodel[meshing,mfem]>=${PLANETMODEL_MINIMUM_VERSION}" CACHE STRING
    "pip requirement installed into the build's venv when no suitable Python is found")

function(meshes_python_works python out_var)
    string(REPLACE "." ", " _minimum "${PLANETMODEL_MINIMUM_VERSION}")
    execute_process(
        COMMAND "${python}" -c "
import re
from importlib.metadata import version
import planetmodel.mesh3d, mfem.ser
found = tuple(int(n) for n in re.findall(r'\\d+', version('planetmodel'))[:3])
raise SystemExit(0 if found >= (${_minimum}) else 1)"
        RESULT_VARIABLE result OUTPUT_QUIET ERROR_QUIET)
    if(result EQUAL 0)
        set(${out_var} TRUE PARENT_SCOPE)
    else()
        set(${out_var} FALSE PARENT_SCOPE)
    endif()
endfunction()

if(NOT MESHES_PYTHON)
    find_package(Python3 3.12 COMPONENTS Interpreter REQUIRED)
    meshes_python_works("${Python3_EXECUTABLE}" _works)
    if(_works)
        set(_python "${Python3_EXECUTABLE}")
    else()
        set(_venv "${CMAKE_BINARY_DIR}/meshes-venv")
        set(_python "${_venv}/bin/python")
        if(NOT EXISTS "${_python}")
            message(STATUS "Creating ${_venv} and installing ${PLANETMODEL_REQUIREMENT} "
                           "(no Python with planetmodel found; set MESHES_PYTHON to use one)")
            execute_process(COMMAND "${Python3_EXECUTABLE}" -m venv "${_venv}"
                            RESULT_VARIABLE _result)
            if(NOT _result EQUAL 0)
                message(FATAL_ERROR "could not create a virtual environment at ${_venv}")
            endif()
        endif()
        meshes_python_works("${_python}" _works)
        if(NOT _works)
            execute_process(COMMAND "${_python}" -m pip install --quiet
                                    "${PLANETMODEL_REQUIREMENT}"
                            RESULT_VARIABLE _result)
            if(NOT _result EQUAL 0)
                message(FATAL_ERROR
                    "pip could not install ${PLANETMODEL_REQUIREMENT} into ${_venv}; "
                    "install it yourself and pass -DMESHES_PYTHON=<python>, "
                    "or configure with -DGENERATE_MESHES=OFF")
            endif()
        endif()
    endif()
    set(MESHES_PYTHON "${_python}" CACHE FILEPATH
        "Python interpreter with planetmodel[meshing]; found or created if empty" FORCE)
endif()

meshes_python_works("${MESHES_PYTHON}" _works)
if(NOT _works)
    message(FATAL_ERROR "${MESHES_PYTHON} cannot import planetmodel.mesh3d (version "
                        "${PLANETMODEL_MINIMUM_VERSION} or later) and mfem.ser; install "
                        "${PLANETMODEL_REQUIREMENT} for it or unset MESHES_PYTHON")
endif()
message(STATUS "Meshes: generated with ${MESHES_PYTHON}")

set(MESHES_DIR "${CMAKE_BINARY_DIR}/data")
file(MAKE_DIRECTORY "${MESHES_DIR}")

# One knob for the whole set's resolution: every script is run with
# --scale MESHES_SCALE (a factor on all element sizes, smaller is
# finer). Changing it regenerates every mesh: the value is written to a
# stamp file (only when it changes) that every mesh depends on.
set(MESHES_SCALE "1" CACHE STRING
    "factor on every generated mesh's element sizes (smaller is finer)")
set(MESHES_SCALE_STAMP "${CMAKE_CURRENT_BINARY_DIR}/meshes_scale.txt")
set(_scale_stamp_content "")
if(EXISTS "${MESHES_SCALE_STAMP}")
    file(READ "${MESHES_SCALE_STAMP}" _scale_stamp_content)
endif()
if(NOT _scale_stamp_content STREQUAL "${MESHES_SCALE}\n")
    file(WRITE "${MESHES_SCALE_STAMP}" "${MESHES_SCALE}\n")
endif()

# meshes_add(<script> <output basename>... [MFEM] [ARGS <script arguments>...]
#            [DEPENDS <extra input files>...])
# One command per script run; its outputs are <basename>.msh and .json, or
# <basename>.mesh and .json for a script that writes MFEM's own format.
# DEPENDS names data files of this directory the script reads, so the mesh
# is regenerated when they change.
function(meshes_add script)
    cmake_parse_arguments(M "MFEM" "" "ARGS;DEPENDS" ${ARGN})
    if(M_MFEM)
        set(ext ".mesh")
    else()
        set(ext ".msh")
    endif()
    string(JOIN " " names ${M_UNPARSED_ARGUMENTS})
    set(outputs)
    foreach(name IN LISTS M_UNPARSED_ARGUMENTS)
        list(APPEND outputs "${MESHES_DIR}/${name}${ext}" "${MESHES_DIR}/${name}.json")
    endforeach()
    set(extra_deps)
    foreach(dep IN LISTS M_DEPENDS)
        list(APPEND extra_deps "${CMAKE_CURRENT_SOURCE_DIR}/${dep}")
    endforeach()
    add_custom_command(
        OUTPUT ${outputs}
        COMMAND "${MESHES_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/${script}"
                ${M_ARGS} --scale "${MESHES_SCALE}" --out "${MESHES_DIR}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${script}"
                "${CMAKE_CURRENT_SOURCE_DIR}/common.py"
                "${MESHES_SCALE_STAMP}"
                ${extra_deps}
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        COMMENT "Generating ${names} with ${script}"
        VERBATIM)
    set(MESHES_OUTPUTS ${MESHES_OUTPUTS} ${outputs} PARENT_SCOPE)
endfunction()

meshes_add(unit_disc.py        disk)
meshes_add(earth_coastlines.py earth_coastlines
                               DEPENDS ice7g_topography_lmax64.npz)
meshes_add(fingerprint_coastline.py sea_level_fingerprint)
meshes_add(unit_ball.py        ball)
meshes_add(offset_disc.py      circular_offset)
meshes_add(disc_with_buffer.py elastogravity_2d)
meshes_add(disc_with_wide_buffer.py elastogravity_2d_wide)
meshes_add(ball_with_buffer.py coupled_poisson)
meshes_add(layered_earth.py    elastogravity_two_layer_2d
                               elastogravity_three_layer_2d
                               elastogravity_two_layer_3d
                               elastogravity_three_layer_3d
                               ARGS --all)
meshes_add(equilibrium_bodies.py
                               flattened_two_layer_2d
                               flattened_two_layer_3d
                               flattened_three_layer_2d
                               flattened_three_layer_3d
                               cmb_topo_two_layer_2d
                               cmb_topo_two_layer_3d
                               cmb_topo_three_layer_2d
                               cmb_topo_three_layer_3d
                               cmb_bump_two_layer_2d
                               cmb_bump_two_layer_3d
                               cmb_bump_three_layer_2d
                               cmb_bump_three_layer_3d
                               MFEM ARGS --all)
meshes_add(aspherical_body.py  aspherical_2d aspherical_buffer_2d
                               aspherical_3d aspherical_buffer_3d
                               aspherical_fluid_core_buffer_2d
                               spherical_fluid_core_buffer_2d
                               aspherical_fluid_core_buffer_3d
                               spherical_fluid_core_buffer_3d
                               MFEM ARGS --all)

add_custom_target(meshes ALL DEPENDS ${MESHES_OUTPUTS})
