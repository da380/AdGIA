include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG main
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)


add_executable(TestDomainLFDeformationGradientIntegrator TestDomainLFDeformationGradientIntegrator.cpp)
target_link_libraries(TestDomainLFDeformationGradientIntegrator PRIVATE  AdGIA gtest_main)
gtest_discover_tests(TestDomainLFDeformationGradientIntegrator)

add_executable(TestInterpolators TestInterpolators.cpp)
target_link_libraries(TestInterpolators PRIVATE  AdGIA gtest_main)
gtest_discover_tests(TestInterpolators)

add_executable(TestDomainVectorScalarIntegrator TestDomainVectorScalarIntegrator.cpp)
target_link_libraries(TestDomainVectorScalarIntegrator PRIVATE  AdGIA gtest_main)
gtest_discover_tests(TestDomainVectorScalarIntegrator)

add_executable(TestDomainVectorGradScalarIntegrator TestDomainVectorGradScalarIntegrator.cpp)
target_link_libraries(TestDomainVectorGradScalarIntegrator PRIVATE  AdGIA gtest_main)
gtest_discover_tests(TestDomainVectorGradScalarIntegrator)

add_executable(TestDomainDivVectorScalarIntegrator TestDomainDivVectorScalarIntegrator.cpp)
target_link_libraries(TestDomainDivVectorScalarIntegrator PRIVATE  AdGIA gtest_main)
gtest_discover_tests(TestDomainDivVectorScalarIntegrator)

add_executable(TestDomainDivVectorDivVectorIntegrator TestDomainDivVectorDivVectorIntegrator.cpp)
target_link_libraries(TestDomainDivVectorDivVectorIntegrator PRIVATE  AdGIA gtest_main)
gtest_discover_tests(TestDomainDivVectorDivVectorIntegrator)

add_executable(TestDomainVectorGradVectorIntegrator TestDomainVectorGradVectorIntegrator.cpp)
target_link_libraries(TestDomainVectorGradVectorIntegrator PRIVATE  AdGIA gtest_main)
gtest_discover_tests(TestDomainVectorGradVectorIntegrator)

add_executable(TestDomainVectorDivVectorIntegrator TestDomainVectorDivVectorIntegrator.cpp)
target_link_libraries(TestDomainVectorDivVectorIntegrator PRIVATE  AdGIA gtest_main)
gtest_discover_tests(TestDomainVectorDivVectorIntegrator)

add_executable(TestDomainMatrixDeformationGradientIntegrator TestDomainMatrixDeformationGradientIntegrator.cpp)
target_link_libraries(TestDomainMatrixDeformationGradientIntegrator PRIVATE  AdGIA gtest_main)
gtest_discover_tests(TestDomainMatrixDeformationGradientIntegrator)

add_executable(TestDomainSymmetricMatrixStrainIntegrator TestDomainSymmetricMatrixStrainIntegrator.cpp)
target_link_libraries(TestDomainSymmetricMatrixStrainIntegrator PRIVATE  AdGIA gtest_main)
gtest_discover_tests(TestDomainSymmetricMatrixStrainIntegrator)


add_executable(TestDomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator TestDomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator.cpp)
target_link_libraries(TestDomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator PRIVATE  AdGIA gtest_main)
gtest_discover_tests(TestDomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator)

add_executable(TestElasticTensorIntegrator TestElasticTensorIntegrator.cpp)
target_link_libraries(TestElasticTensorIntegrator PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestElasticTensorIntegrator)

add_executable(TestBoundaryNormalIntegrators TestBoundaryNormalIntegrators.cpp)
target_link_libraries(TestBoundaryNormalIntegrators PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestBoundaryNormalIntegrators)

add_executable(TestSubMeshDofInjection TestSubMeshDofInjection.cpp)
target_link_libraries(TestSubMeshDofInjection PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestSubMeshDofInjection)

if(USE_MPI)
    add_executable(TestSubMeshDofInjectionPar TestSubMeshDofInjectionPar.cpp)
    target_link_libraries(TestSubMeshDofInjectionPar PRIVATE AdGIA MPI::MPI_CXX)
    add_executable(TestSubMeshMixedBilinearFormPar TestSubMeshMixedBilinearFormPar.cpp)
    target_link_libraries(TestSubMeshMixedBilinearFormPar PRIVATE AdGIA MPI::MPI_CXX)
    add_executable(TestQuasiStaticProblemPar TestQuasiStaticProblemPar.cpp)
    target_link_libraries(TestQuasiStaticProblemPar PRIVATE AdGIA MPI::MPI_CXX)
    add_executable(TestViscoelasticPar TestViscoelasticPar.cpp)
    target_link_libraries(TestViscoelasticPar PRIVATE AdGIA MPI::MPI_CXX)
    add_executable(TestMixedProblemPar TestMixedProblemPar.cpp)
    target_link_libraries(TestMixedProblemPar PRIVATE AdGIA MPI::MPI_CXX)
    add_executable(TestBoundaryNormalIntegratorsPar TestBoundaryNormalIntegratorsPar.cpp)
    target_link_libraries(TestBoundaryNormalIntegratorsPar PRIVATE AdGIA MPI::MPI_CXX)
    add_executable(TestMixedProblemFluidPar TestMixedProblemFluidPar.cpp)
    target_link_libraries(TestMixedProblemFluidPar PRIVATE AdGIA MPI::MPI_CXX)
    add_executable(TestFluidGaugePar TestFluidGaugePar.cpp)
    target_link_libraries(TestFluidGaugePar PRIVATE AdGIA MPI::MPI_CXX)
    add_executable(TestMixedProblemGaugedPar TestMixedProblemGaugedPar.cpp)
    target_link_libraries(TestMixedProblemGaugedPar PRIVATE AdGIA MPI::MPI_CXX)
    add_executable(TestSlidingInterfacePar TestSlidingInterfacePar.cpp)
    target_link_libraries(TestSlidingInterfacePar PRIVATE AdGIA MPI::MPI_CXX)
    add_executable(TestReferentialProblemPar TestReferentialProblemPar.cpp)
    target_link_libraries(TestReferentialProblemPar PRIVATE AdGIA MPI::MPI_CXX)
    add_executable(TestNullSpacePar TestNullSpacePar.cpp)
    target_link_libraries(TestNullSpacePar PRIVATE AdGIA MPI::MPI_CXX)
    add_executable(TestCompositeRheologyPar TestCompositeRheologyPar.cpp)
    target_link_libraries(TestCompositeRheologyPar PRIVATE AdGIA MPI::MPI_CXX)
    add_executable(TestSeaLevelPar TestSeaLevelPar.cpp)
    target_link_libraries(TestSeaLevelPar PRIVATE AdGIA MPI::MPI_CXX)
    add_executable(TestRotationPar TestRotationPar.cpp)
    target_link_libraries(TestRotationPar PRIVATE AdGIA MPI::MPI_CXX)
    add_executable(TestSeaLevelCouplingPar TestSeaLevelCouplingPar.cpp)
    target_link_libraries(TestSeaLevelCouplingPar PRIVATE AdGIA MPI::MPI_CXX)
    add_executable(TestSphericalHarmonicsPar TestSphericalHarmonicsPar.cpp)
    target_link_libraries(TestSphericalHarmonicsPar PRIVATE AdGIA MPI::MPI_CXX)
    add_executable(TestEquilibriumFiguresPar TestEquilibriumFiguresPar.cpp)
    target_link_libraries(TestEquilibriumFiguresPar PRIVATE AdGIA MPI::MPI_CXX)
    add_executable(TestRieszPar TestRieszPar.cpp)
    target_link_libraries(TestRieszPar PRIVATE AdGIA MPI::MPI_CXX)
    add_executable(TestDescentPar TestDescentPar.cpp)
    target_link_libraries(TestDescentPar PRIVATE AdGIA MPI::MPI_CXX)
    foreach(np 1 2 4)
        add_test(NAME TestSubMeshDofInjectionPar_np${np}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                         $<TARGET_FILE:TestSubMeshDofInjectionPar>)
        add_test(NAME TestSubMeshMixedBilinearFormPar_np${np}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                         $<TARGET_FILE:TestSubMeshMixedBilinearFormPar>)
        add_test(NAME TestQuasiStaticProblemPar_np${np}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                         $<TARGET_FILE:TestQuasiStaticProblemPar>)
        add_test(NAME TestViscoelasticPar_np${np}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                         $<TARGET_FILE:TestViscoelasticPar>)
        add_test(NAME TestMixedProblemPar_np${np}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                         $<TARGET_FILE:TestMixedProblemPar>)
        add_test(NAME TestBoundaryNormalIntegratorsPar_np${np}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                         $<TARGET_FILE:TestBoundaryNormalIntegratorsPar>)
        add_test(NAME TestMixedProblemFluidPar_np${np}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                         $<TARGET_FILE:TestMixedProblemFluidPar>)
        add_test(NAME TestFluidGaugePar_np${np}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                         $<TARGET_FILE:TestFluidGaugePar>)
        add_test(NAME TestMixedProblemGaugedPar_np${np}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                         $<TARGET_FILE:TestMixedProblemGaugedPar>)
        add_test(NAME TestSlidingInterfacePar_np${np}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                         $<TARGET_FILE:TestSlidingInterfacePar>)
        add_test(NAME TestReferentialProblemPar_np${np}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                         $<TARGET_FILE:TestReferentialProblemPar>)
        add_test(NAME TestNullSpacePar_np${np}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                         $<TARGET_FILE:TestNullSpacePar>)
        add_test(NAME TestCompositeRheologyPar_np${np}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                         $<TARGET_FILE:TestCompositeRheologyPar>)
        add_test(NAME TestSeaLevelPar_np${np}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                         $<TARGET_FILE:TestSeaLevelPar>)
        add_test(NAME TestRotationPar_np${np}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                         $<TARGET_FILE:TestRotationPar>)
        add_test(NAME TestSeaLevelCouplingPar_np${np}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                         $<TARGET_FILE:TestSeaLevelCouplingPar>)
        add_test(NAME TestSphericalHarmonicsPar_np${np}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                         $<TARGET_FILE:TestSphericalHarmonicsPar>)
        add_test(NAME TestEquilibriumFiguresPar_np${np}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                         $<TARGET_FILE:TestEquilibriumFiguresPar>)
        add_test(NAME TestRieszPar_np${np}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                         $<TARGET_FILE:TestRieszPar>)
        add_test(NAME TestDescentPar_np${np}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                         $<TARGET_FILE:TestDescentPar>)
    endforeach()
endif()


add_executable(TestNullSpace TestNullSpace.cpp)
target_link_libraries(TestNullSpace PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestNullSpace)

add_executable(TestRelaxationLaw TestRelaxationLaw.cpp)
target_link_libraries(TestRelaxationLaw PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestRelaxationLaw)

add_executable(TestMeshManifest TestMeshManifest.cpp)
target_link_libraries(TestMeshManifest PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestMeshManifest)

add_executable(TestSphericalHarmonics TestSphericalHarmonics.cpp)
target_link_libraries(TestSphericalHarmonics PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestSphericalHarmonics)

add_executable(TestPoissonOperators TestPoissonOperators.cpp)
target_link_libraries(TestPoissonOperators PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestPoissonOperators)

add_executable(TestCompositeRheology TestCompositeRheology.cpp)
target_link_libraries(TestCompositeRheology PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestCompositeRheology)

add_executable(TestRheology TestRheology.cpp)
target_link_libraries(TestRheology PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestRheology)

add_executable(TestQuasiStaticProblem TestQuasiStaticProblem.cpp)
target_link_libraries(TestQuasiStaticProblem PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestQuasiStaticProblem)

add_executable(TestElasticTensor TestElasticTensor.cpp)
target_link_libraries(TestElasticTensor PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestElasticTensor)

add_executable(TestViscoelastic TestViscoelastic.cpp)
target_link_libraries(TestViscoelastic PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestViscoelastic)

add_executable(TestSubMeshMixedBilinearForm TestSubMeshMixedBilinearForm.cpp)
target_link_libraries(TestSubMeshMixedBilinearForm PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestSubMeshMixedBilinearForm)

add_executable(TestMappedBoundaryIntegrators TestMappedBoundaryIntegrators.cpp)
target_link_libraries(TestMappedBoundaryIntegrators PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestMappedBoundaryIntegrators)

add_executable(TestMappedDomainIntegrators TestMappedDomainIntegrators.cpp)
target_link_libraries(TestMappedDomainIntegrators PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestMappedDomainIntegrators)

add_executable(TestMappings TestMappings.cpp)
target_link_libraries(TestMappings PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestMappings)

add_executable(TestTransformedDiffusionIntegrator TestTransformedDiffusionIntegrator.cpp)
target_link_libraries(TestTransformedDiffusionIntegrator PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestTransformedDiffusionIntegrator)

add_executable(TestMixedProblem TestMixedProblem.cpp)
target_link_libraries(TestMixedProblem PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestMixedProblem)

add_executable(TestMixedProblemFluid TestMixedProblemFluid.cpp)
target_link_libraries(TestMixedProblemFluid PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestMixedProblemFluid)

add_executable(TestFluidGauge TestFluidGauge.cpp)
target_link_libraries(TestFluidGauge PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestFluidGauge)

add_executable(TestSeaLevel TestSeaLevel.cpp)
target_link_libraries(TestSeaLevel PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestSeaLevel)

add_executable(TestRotation TestRotation.cpp)
target_link_libraries(TestRotation PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestRotation)

add_executable(TestSeaLevelCoupling TestSeaLevelCoupling.cpp)
target_link_libraries(TestSeaLevelCoupling PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestSeaLevelCoupling)

add_executable(TestSlipInterface TestSlipInterface.cpp)
target_link_libraries(TestSlipInterface PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestSlipInterface)

add_executable(TestSlipProblem TestSlipProblem.cpp)
target_link_libraries(TestSlipProblem PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestSlipProblem)

add_executable(TestBackground TestBackground.cpp)
target_link_libraries(TestBackground PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestBackground)

add_executable(TestEquilibriumFigures TestEquilibriumFigures.cpp)
target_link_libraries(TestEquilibriumFigures PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestEquilibriumFigures)

add_executable(TestRiesz TestRiesz.cpp)
target_link_libraries(TestRiesz PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestRiesz)

add_executable(TestDescent TestDescent.cpp)
target_link_libraries(TestDescent PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestDescent)

add_executable(TestReferentialProblem TestReferentialProblem.cpp)
target_link_libraries(TestReferentialProblem PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestReferentialProblem)

add_executable(TestGeneralisedStiffness TestGeneralisedStiffness.cpp)
target_link_libraries(TestGeneralisedStiffness PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestGeneralisedStiffness)

add_executable(TestSlidingInterface TestSlidingInterface.cpp)
target_link_libraries(TestSlidingInterface PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestSlidingInterface)

add_executable(TestMixedProblemGauged TestMixedProblemGauged.cpp)
target_link_libraries(TestMixedProblemGauged PRIVATE AdGIA gtest_main)
gtest_discover_tests(TestMixedProblemGauged)
