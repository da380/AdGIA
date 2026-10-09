cmake_minimum_required(VERSION 3.15)

project(
    AdGIA
    VERSION 0.0.1
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prohibit in-source builds
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "In-source builds are prohibited.")
endif ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Import MFEM. The following variables can be used to help CMake find MFEM:
#  * MFEM_DIR - absolute path to the MFEM build or install prefix.
#  * mfem_DIR - absolute path to where MFEMConfig.cmake is.
message(STATUS "Looking for mfem ...")
set(MFEM_DIR "" CACHE PATH "Path to the MFEM build or install prefix.")
if (MFEM_DIR)
    find_package(mfem REQUIRED NAMES MFEM HINTS "${MFEM_DIR}"
                "${mfem_DIR}/lib/cmake/mfem" NO_DEFAULT_PATH)
else()
    find_package(mfem CONFIG REQUIRED NAMES MFEM)
endif()
message(STATUS "Found mfem config in: ${mfem_DIR} (version ${MFEM_VERSION})")
# v4.10 is the reference version (CI); v4.9 is the oldest known to pass the
# full test suite (see README, Installation).
if(MFEM_VERSION VERSION_LESS 4.9)
    message(FATAL_ERROR "MFEM ${MFEM_VERSION} is too old: AdGIA needs MFEM "
                        ">= 4.9 (v4.10 is the reference version).")
endif()



# Add MPI if needed.
option(USE_MPI "Use MPI library" OFF)
if(USE_MPI)
    enable_language(C)
    # Prefer the mpiexec that sits next to the MPI compiler wrapper (e.g. a
    # PETSc-installed MPICH), so that ctest launches with the same MPI the
    # code links against rather than whatever mpiexec is first on PATH.
    if(NOT MPIEXEC_EXECUTABLE)
        get_filename_component(_mpi_bin_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
        if(EXISTS "${_mpi_bin_dir}/mpiexec")
            set(MPIEXEC_EXECUTABLE "${_mpi_bin_dir}/mpiexec" CACHE FILEPATH
                "MPI launcher (taken from the compiler wrapper's directory)")
        endif()
    endif()
    find_package(MPI REQUIRED)        
    if (NOT CMAKE_CXX_COMPILER AND MFEM_CXX_COMPILER)  
        set(CMAKE_CXX_COMPILER "${MFEM_CXX_COMPILER}")      
    endif()
endif()


# Set up the library 
add_library(${PROJECT_NAME} STATIC)

# Add in linked libraries.
if(USE_MPI)
    target_link_libraries(${PROJECT_NAME} PUBLIC mfem MPI::MPI_CXX )
else()
    target_link_libraries(${PROJECT_NAME} PUBLIC mfem)
endif()


# Add in the source files. 
add_subdirectory(src)

# Optionally build examples. 
option(BUILD_EXAMPLES "Build the example programs" OFF)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Optionally build the benchmarks.
option(BUILD_BENCHMARKS "Build the benchmark programs" OFF)
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Post-processing launchers (Python; scripts stay in source, run in build).
add_subdirectory(postprocess)

# Optionally build tests
option(BUILD_TESTS "Build the test programs" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()



# The examples and tests read their meshes from <build>/data: the small
# MFEM-native meshes are copied there from data/, and the gmsh meshes are
# generated there by the Python scripts in meshes/ (see meshes/CMakeLists.txt).
if(BUILD_EXAMPLES OR BUILD_TESTS)
    set(_generate_meshes_default ON)
else()
    set(_generate_meshes_default OFF)
endif()
option(GENERATE_MESHES
       "Generate the gmsh meshes in <build>/data with planetmodel (needs Python)"
       ${_generate_meshes_default})

if(BUILD_EXAMPLES OR BUILD_TESTS)
    add_custom_target(CopyData
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/data
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/data ${CMAKE_BINARY_DIR}/data
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            )
    add_dependencies(AdGIA CopyData)
endif()
if(GENERATE_MESHES)
    add_subdirectory(meshes)
    if(TARGET CopyData)
        add_dependencies(CopyData meshes)
    endif()
endif()

# Add standard install locations. 
include(GNUInstallDirs)

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Install the library (headers and library file)
install(TARGETS AdGIA 
        EXPORT AdGIATargets
        DESTINATION lib)
install(DIRECTORY include/
        DESTINATION include)


# Create the CMake package configuration file
install(EXPORT AdGIATargets        
        FILE AdGIATargets.cmake
        DESTINATION lib/cmake/AdGIA)

include(CMakePackageConfigHelpers)        


set(INCLUDE_INSTALL_DIRS ${PROJECT_BINARY_DIR})

set(LIB_INSTALL_DIR ${PROJECT_BINARY_DIR})
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/AdGIAConfig.cmake"
  INSTALL_DESTINATION "lib/cmake/AdGIA"
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO  
  )

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/AdGIAConfigVersion.cmake"
  VERSION "${AdGIA_VERSION_MAJOR}.${AdGIA_VERSION_MINOR}"
  COMPATIBILITY AnyNewerVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/AdGIAConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/AdGIAConfigVersion.cmake
  DESTINATION lib/cmake/AdGIA
  )

export(EXPORT AdGIATargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/AdGIATargets.cmake"
)


option(BUILD_DOCS "Build documentation" OFF)

if(BUILD_DOCS)

find_package(Doxygen)

if (DOXYGEN_FOUND)
    # The documentation is written to <build>/doc.
    set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/doc")

    # The MathJax configuration is copied next to the HTML output.
    set(MATHJAX_CONFIG_JS_SRC "${CMAKE_SOURCE_DIR}/doc/mathjax_config.js")
    set(MATHJAX_CONFIG_JS_DST "${DOXYGEN_OUTPUT_DIRECTORY}/mathjax_config.js")

    add_custom_command(
        OUTPUT "${MATHJAX_CONFIG_JS_DST}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${DOXYGEN_OUTPUT_DIRECTORY}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${MATHJAX_CONFIG_JS_SRC}" "${MATHJAX_CONFIG_JS_DST}"
        COMMENT "Copying MathJax configuration file."
        VERBATIM
    )    

    # Doxygen PREDEFINED macros; the parallel classes are documented only
    # in an MPI build.
    set(DOXYGEN_PREDEFINED_MACROS "DOXYGEN_SHOULD_PARSE_THIS")
    set(DOXYGEN_PREDEFINED_MACROS "${DOXYGEN_PREDEFINED_MACROS} DOXYGEN_ONLY")
    
    if(USE_MPI)
        set(DOXYGEN_PREDEFINED_MACROS "${DOXYGEN_PREDEFINED_MACROS} MFEM_USE_MPI=1")
    endif()

    configure_file(
        "${CMAKE_SOURCE_DIR}/doc/Doxyfile.in"
        "${CMAKE_BINARY_DIR}/Doxyfile"
        @ONLY
    )

    # Target `doc`, part of the default build when BUILD_DOCS is on.
    add_custom_target(
        doc
        ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "${DOXYGEN_OUTPUT_DIRECTORY}"
        COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating Doxygen documentation..."
        DEPENDS "${MATHJAX_CONFIG_JS_DST}"
    )

    install(DIRECTORY "${DOXYGEN_OUTPUT_DIRECTORY}"
            DESTINATION "${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/html"
            COMPONENT Development
    )

else()
    message(STATUS "Doxygen not found. Documentation will not be generated.")
endif()

endif()




