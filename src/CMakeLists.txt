target_sources(${PROJECT_NAME} PRIVATE null_space.cpp
                                       background.cpp
                                       bordered.cpp
                                       equilibrium_figures.cpp
                                       riesz.cpp
                                       descent.cpp
                                       bilininteg.cpp
                                       bilininteg_pa.cpp
                                       kernels/elastic_tensor_kernels.cpp
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
    set_source_files_properties(kernels/elastic_tensor_kernels.cpp
        TARGET_DIRECTORY ${PROJECT_NAME} PROPERTIES
        LANGUAGE CUDA
        COMPILE_OPTIONS "--expt-extended-lambda;--expt-relaxed-constexpr")
endif()
