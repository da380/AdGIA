target_sources(${PROJECT_NAME} PRIVATE null_space.cpp
                                       background.cpp
                                       bordered.cpp
                                       equilibrium_figures.cpp
                                       riesz.cpp
                                       descent.cpp
                                       bilininteg.cpp
                                       bilininteg_pa.cpp
                                       gradient_coupling.cpp
                                       kernels/block_kernels.cpp
                                       ceed/ceed_operators.cpp
                                       coefficient.cpp
                                       lininteg.cpp
                                       mappings.cpp
                                       mesh.cpp
                                       p_multigrid.cpp
                                       manifest.cpp
                                       model_coefficient.cpp
                                       model_export.cpp
                                       poisson.cpp
                                       spherical_harmonics.cpp
                                       rheology.cpp
                                       relaxation_law.cpp
                                       quasi_static_problem.cpp
                                       fault_problem.cpp
                                       referential_problem.cpp
    level_border.cpp
    profile_border.cpp
    mixed_problem.cpp
                                       material_law.cpp
                                       nonlinear_problem.cpp
                                       viscoelastic.cpp
                                       adjoint.cpp
                                       tangent.cpp
                                       second_order.cpp
                                       discrete_adjoint.cpp
                                       discrete_second_order.cpp
                                       measurement.cpp
                                       elastic_tensor.cpp
                                       rotation.cpp
                                       sea_level.cpp
                                       submesh.cpp
                                       )

# The sources with device kernels (mfem::forall bodies), in kernels/. They
# include mfem and AdGIA/detail/pa_kernels.hpp alone, so the CUDA compiler's
# dialect (CMAKE_CUDA_STANDARD, top level) binds nothing else.
if(MFEM_USE_CUDA)
    set_source_files_properties(kernels/block_kernels.cpp
        TARGET_DIRECTORY ${PROJECT_NAME} PROPERTIES
        LANGUAGE CUDA
        COMPILE_OPTIONS "--expt-extended-lambda;--expt-relaxed-constexpr")
endif()

# The host's own vector instructions for the kernels (off by default: the
# library then runs on any machine of the architecture). On, the kernel
# sources alone are compiled with -march=native, and the dense kernel takes
# its vector form where that gives AVX2 or AVX-512: two to three times
# the speed on tetrahedra. The binaries then need the machine they were
# built on, or one like it.
option(ADGIA_NATIVE_KERNELS
       "Compile the kernels in src/kernels/ for this machine (-march=native)"
       OFF)
if(ADGIA_NATIVE_KERNELS AND NOT MFEM_USE_CUDA)
    set_source_files_properties(kernels/block_kernels.cpp
        TARGET_DIRECTORY ${PROJECT_NAME} PROPERTIES
        COMPILE_OPTIONS "-march=native")
endif()

# With libCEED in mfem (MFEM_USE_CEED) the forms are also libCEED operators
# (ceed/). Their QFunction sources are read again at run time by libCEED's
# code-generating backends, from this directory.
set(ADGIA_CEED_QFUNCTION_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ceed" CACHE PATH
    "Where libCEED finds AdGIA's QFunction sources at run time")
set_source_files_properties(ceed/ceed_operators.cpp
    TARGET_DIRECTORY ${PROJECT_NAME} PROPERTIES
    COMPILE_DEFINITIONS "ADGIA_CEED_QFUNCTION_DIR=\"${ADGIA_CEED_QFUNCTION_DIR}\"")
