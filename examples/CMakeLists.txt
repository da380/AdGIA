# The examples: one source per example, for both builds — a parallel
# program when MFEM has MPI, a serial one otherwise, with the genuine
# serial/parallel differences marked by #ifdef MFEM_USE_MPI blocks in the
# sources. The examples read their meshes from <build>/data, which the
# top-level CMakeLists.txt fills (CopyData and the meshes/ scripts).

set(EXAMPLES
    poisson_dtn                 # Poisson on the whole space: Neumann / DtN / multipole
    transformed_diffusion       # pull-back Laplacian on a mapped domain
    relabelled_elasticity       # one aspherical body, natural vs relabelled description
    adjoint_elasticity          # first-order sensitivity kernels (kappa, mu) + FD check + Riesz gradients
    viscoelastic_adjoint        # final-time observation kernels (kappa, mu, eta) via the time-reversed adjoint
    submesh_injection           # tour of SubMeshDofInjection
    coupled_poisson             # coupled Poisson pair across mesh / submesh
    elastogravity_layered       # LinearQuasiStaticMixedSelfGravitatingProblem: uniform, fluid core (+ inner core)
    quasi_static_elasticity     # quasi_static_problem.hpp driver
    gauged_fluid_cavity         # gauged fluid region, purely elastic (SetGaugedFluid)
    sliding_fluid_ellipse       # sliding interface (displacement discontinuity) on an elliptical body
    slipping_interface          # self-gravitating slipping fluid core: every option, spherical or aspherical reference
    equilibrium_stress          # AW10 minimum-norm / minimum-deviatoric equilibrium stress generators
    equilibrium_density         # density restoration: the feasibility functional driven to its floor
    prestress_loading           # does the deviatoric pre-stress matter for loading? (ellipse sweep)
    referential_elastogravity   # general referential problem vs the Eulerian one (self-benchmark)
    viscoelasticity             # ViscoelasticOperator driver
    viscoelastic_column         # two-layer Maxwell column vs exact: interface on a face or inside an element
    viscoelastic_loading        # load on, load off: layered Maxwell box, no gravity
    viscoelastic_schemes        # cost/accuracy table of the time integrators
    self_gravitating_relaxation # Maxwell mantle + fluid core under a Heaviside load
    lateral_viscosity           # self-gravitating disc relaxing faster on one side: transient asymmetry
    self_gravitating_solvers    # one problem, every linear-solver architecture: cost/agreement table
    love_numbers                # load and tidal Love numbers by degree, one solve each
    viscoelastic_love_numbers   # load Love-number histories of a Maxwell sphere vs the closed form
    anisotropic_elasticity      # ElasticTensorIntegrator, radial TI
    coseismic_deformation       # moment-tensor (stress-glut) point/finite-fault sources
    postseismic_deformation     # the same fault over an SLS: relaxation to the long-term limit
)

foreach(ex IN LISTS EXAMPLES)
    add_executable(${ex} ${ex}.cpp)
    target_link_libraries(${ex} AdGIA)
endforeach()


# The examples write their curves as CSV tables (visualisation.hpp) into
# the directory they run in; the plotting script sits beside them.
configure_file(plot_csv.py ${CMAKE_CURRENT_BINARY_DIR}/plot_csv.py COPYONLY)
