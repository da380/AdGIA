// ============================================================================
// elastogravity_layered.cpp
//
// Self-gravitating elastic deformation of a body under a surface mass load
// or a tidal potential, as a driver of
// LinearQuasiStaticMixedSelfGravitatingProblem (AdGIA/mixed_problem.hpp).
// The models and loads are those of layered_model.hpp, chosen by the mesh:
//
//   uniform      a solid of constant density and moduli (-rho, -kappa, -mu),
//                of any shape
//   two-layer    a fluid core under a mantle
//   three-layer  a solid inner core, a fluid outer core and a mantle
//
// The solid regions (the mantle, and the inner core when the mesh has one)
// form ONE displacement SubMesh, disconnected for the three-layer model; the
// fluid outer core enters through a FluidRegion (its density, the
// hydrostatic Poisson term and the fluid–solid interface terms). The
// inner core's near-null rotations are projected out.
//
// One source serves the serial and the parallel build: against an MFEM with
// MPI the mesh, the spaces and the fields are the parallel ones and the
// program runs under mpiexec, on any number of ranks.
//
// Meshes (../data):
//   disc_buffer_2d.msh, ball_buffer_coarse_3d.msh          uniform, 2-D and 3-D
//                   (meshes/disc_with_buffer.py, meshes/ball_with_buffer.py)
//   aspherical_buffer_2d.mesh, aspherical_buffer_3d.mesh         uniform
//                   (meshes/aspherical_body.py)
//   earth_fluidcore_2d.msh, earth_innercore_2d.msh,
//   earth_innercore_3d.msh                   (meshes/layered_earth.py)
//
// Options (defaults in brackets):
//   -m      mesh file [../data/earth_fluidcore_2d.msh]; the
//           layering of the model follows the mesh (see above).
//   -o      finite element order [1].
//   -s      solver: 0 Schur-complement CG, 1 block MINRES, 2 both [1];
//           with 2 the difference of the two solutions is reported.
//   -deg    truncation degree of the DtN expansion [16].
//   -rt     relative solver tolerance [1e-10].
//   -load   factor on the surface mass load [1]; 0 switches it off.
//   -tidal  amplitude A of the degree-2 tidal potential A (r/a)^2 P_2,
//           in m^2/s^2 [0]; with -load 0 it is the only forcing.
//   -rho    density of the uniform model [5500 kg/m^3].
//   -kappa  bulk modulus of the uniform model [300e9 Pa].
//   -mu     shear modulus of the uniform model [150e9 Pa].
//   -solid-core / -fluid-core   treat the outer core as a solid with
//           constant moduli, for comparison with the fluid physics
//           [fluid].
//   -diag / -no-diag   print the rigid-mode residuals and, with a
//           fluid, the extreme Ritz values of the potential block
//           [off].
//   -vis / -no-vis   GLVis windows on or off [off].
//   -no-fluid-mass / -fluid-mass   drop the hydrostatic Poisson term
//           rho'_F phi (the fluid becomes unstratified in the Eulerian
//           sense), for experiments [off: the term is kept]. For the
//           PREM-like core it changes the response by a factor of about
//           three, because the potential block is not far from losing its
//           positivity (doc/self_gravitation.md, "3. Solvers",
//           "Definiteness of the potential block").
//   -pv / -no-pv   write ParaView data collections [off]: ParaView/
//           elastogravity_layered (displacement, on the solid SubMesh) and
//           ParaView/elastogravity_layered_potential (potentials, on the
//           parent mesh).
//
// Sample runs (with mpiexec -np N in front in a parallel build):
//    ./elastogravity_layered -o 2
//    ./elastogravity_layered -m ../data/disc_buffer_2d.msh -o 2 -s 2 -diag
//    ./elastogravity_layered -m ../data/ball_buffer_coarse_3d.msh -o 2 -mu 50e9
//    ./elastogravity_layered -m ../data/earth_innercore_2d.msh -o 2
//        -s 2 -diag
//    ./elastogravity_layered -m ../data/earth_innercore_3d.msh -o 1
//    ./elastogravity_layered -o 2 -load 0 -tidal 1.0
//    ./elastogravity_layered -o 2 -solid-core
//    ./elastogravity_layered -m ../data/earth_innercore_3d.msh -o 1
//        -no-fluid-mass
//
// Output: iterations, time and norms of each solve, the maximum displacement
// in metres, and with -vis the displacement and the potential perturbation
// in GLVis (-pv: ParaView collections).
//
// Mesh requirements: 2-D or 3-D, origin-centred, in the units of the body
// (the radii are those of layered_model.hpp; the mesh's outer radius is the
// surface). The layering is read from the number of attributes and the
// attribute table is, for the meshes of meshes/layered_earth.py (radii in
// units of the surface radius 6371 km, the core-mantle boundary at
// r = 3483/6371):
//
//   attribute   uniform         two-layer       three-layer
//      1        body            fluid core      inner core
//      2        buffer          mantle          fluid outer core
//      3        -               buffer (to 1.2) mantle
//      4        -               -               buffer
//
//   boundary    1 CMB (two-layer), 2 surface, 3 outer boundary of the
//               buffer; the three-layer mesh adds the inner-core boundary.
//
// The buffer (to r = 1.2 for the layered meshes) carries the potential
// only; the displacement lives on the solid SubMesh of the body. The
// uniform meshes (disc_with_buffer.py, ball_with_buffer.py,
// aspherical_body.py) have the body as attribute 1 and the buffer as 2,
// the surface being the body's outer boundary. Dimensional parameters are
// in SI units and are nondimensionalised internally.
// ============================================================================

#include <chrono>
#include <iostream>
#include <memory>
#include <numbers>
#include <vector>

#include "AdGIA.hpp"
#include "layered_model.hpp"
#include "visualisation.hpp"

using namespace mfem;
using namespace AdGIA;
using namespace layered;

namespace {

#ifdef MFEM_USE_MPI
using MeshType = ParMesh;
using SubMeshType = ParSubMesh;
using SpaceType = ParFiniteElementSpace;
using FieldType = ParGridFunction;
bool Root() { return Mpi::Root(); }
#else
using MeshType = Mesh;
using SubMeshType = SubMesh;
using SpaceType = FiniteElementSpace;
using FieldType = GridFunction;
bool Root() { return true; }
#endif

using Problem = LinearQuasiStaticMixedSelfGravitatingProblem;

// The largest absolute value of a field, over all ranks.
real_t MaxAbs(const FieldType& f) {
  real_t m = f.Normlinf();
#ifdef MFEM_USE_MPI
  MPI_Allreduce(MPI_IN_PLACE, &m, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                MPI_COMM_WORLD);
#endif
  return m;
}

struct Result {
  FieldType u, phi;
  int outer = 0, inner = 0;
  double seconds = 0.0;
};

Result Run(Problem& p, Problem::SolverType type) {
  p.SetSolverType(type);
  p.AssembleForce(0.0);
  const auto t0 = std::chrono::steady_clock::now();
  const bool ok = p.Solve();
  const auto t1 = std::chrono::steady_clock::now();
  Result r{static_cast<const FieldType&>(p.Displacement()),
           static_cast<const FieldType&>(p.Potential()),
           p.LastOuterIterations(), p.LastInnerIterations(),
           std::chrono::duration<double>(t1 - t0).count()};
  const char* name =
      type == Problem::SolverType::SchurCG ? "Schur CG    " : "Block MINRES";
  // The norms are collective: computed on every rank before printing.
  const real_t u_norm = L2Norm(r.u), phi_norm = L2Norm(r.phi);
  if (Root()) {
    std::cout.precision(10);
    std::cout << name << ": " << (ok ? "converged" : "FAILED") << ", outer "
              << r.outer << ", inner " << r.inner << ", " << r.seconds
              << " s, ||u|| = " << u_norm << ", ||phi|| = " << phi_norm
              << "\n";
  }
  return r;
}

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif

  const char* mesh_file = "../data/earth_fluidcore_2d.msh";
  int order = 1;
  int solver = 1;
  int dtn_degree = 16;
  real_t rel_tol = 1e-10;
  bool diagnostics = false;
  bool visualization = false;
  bool paraview = false;
  bool no_fluid_mass = false;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
                 "Mesh of a uniform, two-layer or three-layer body.");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&solver, "-s", "--solver",
                 "0: Schur-complement CG, 1: block MINRES, 2: both.");
  args.AddOption(&dtn_degree, "-deg", "--dtn-degree",
                 "Truncation degree of the DtN expansion.");
  args.AddOption(&rel_tol, "-rt", "--rel-tol", "Relative solver tolerance.");
  args.AddOption(&load_factor, "-load", "--load-factor",
                 "Factor on the surface mass load (0: none).");
  args.AddOption(&tidal_amplitude, "-tidal", "--tidal-amplitude",
                 "Amplitude of the degree-2 tidal potential [m^2/s^2].");
  args.AddOption(&uniform_density, "-rho", "--density",
                 "Density of the uniform model [kg/m^3].");
  args.AddOption(&uniform_bulk_modulus, "-kappa", "--bulk-modulus",
                 "Bulk modulus of the uniform model [Pa].");
  args.AddOption(&uniform_shear_modulus, "-mu", "--shear-modulus",
                 "Shear modulus of the uniform model [Pa].");
  args.AddOption(&solid_core, "-solid-core", "--solid-core", "-fluid-core",
                 "--fluid-core", "Treat the outer core as solid.");
  args.AddOption(&diagnostics, "-diag", "--diagnostics", "-no-diag",
                 "--no-diagnostics",
                 "Print rigid-mode residuals and potential-block Ritz values.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization", "GLVis visualisation.");
  args.AddOption(&no_fluid_mass, "-no-fluid-mass", "--no-fluid-mass",
                 "-fluid-mass", "--fluid-mass",
                 "Drop the fluid mass term (rho'_F = 0), for experiments.");
  args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                 "Write a ParaView data collection.");
  args.Parse();
  if (!args.Good()) {
    if (Root()) args.PrintUsage(std::cout);
    return 1;
  }
  if (Root()) args.PrintOptions(std::cout);

  Mesh serial(mesh_file, 1, 1);
  const int dim = serial.Dimension();
  if (!SetModel(serial)) {
    if (Root()) {
      std::cerr << "Expected a uniform (2 attributes), two-layer (3) or "
                   "three-layer (4) mesh.\n";
    }
    return 1;
  }
  const bool fluid_core = !uniform && !solid_core;
#ifdef MFEM_USE_MPI
  ParMesh parent(MPI_COMM_WORLD, serial);
  serial.Clear();
#else
  Mesh& parent = serial;
#endif
  if (Root()) {
    std::cout << ModelName() << " model, " << dim << "-D";
    if (!uniform) {
      std::cout << ", " << (solid_core ? "solid" : "fluid") << " outer core";
    }
#ifdef MFEM_USE_MPI
    std::cout << ", " << Mpi::WorldSize() << " ranks";
#endif
    std::cout << "\n";
    ND.Print();
  }

  Array<int> solid_attrs = SolidAttributes();
  SubMeshType solid(SubMeshType::CreateFromDomain(parent, solid_attrs));
  H1_FECollection fec(order, dim);
  SpaceType fes_u(&solid, &fec, dim), fes_phi(&parent, &fec);
#ifdef MFEM_USE_MPI
  // (collective, so outside the root-only block)
  const HYPRE_BigInt n_u = fes_u.GlobalTrueVSize();
  const HYPRE_BigInt n_phi = fes_phi.GlobalTrueVSize();
#else
  const int n_u = fes_u.GetTrueVSize(), n_phi = fes_phi.GetTrueVSize();
#endif
  if (Root()) {
    std::cout << "Displacement unknowns: " << n_u
              << ", potential unknowns: " << n_phi << "\n";
    if (uniform) {
      // With G = 1 the gravity on the surface of a uniform ball (disc) of
      // unit radius is 4 pi rho / dim.
      const real_t rho_nd = ND.ScaleDensity(uniform_density);
      std::cout << "Coupling strength rho g R / mu = "
                << rho_nd * 4.0 * std::numbers::pi_v<real_t> * rho_nd /
                       (dim * ND.ScaleStress(uniform_shear_modulus))
                << "\n";
    }
  }

  FunctionCoefficient rho(Density), rho_f(FluidDensity), kappa(BulkModulus),
      mu(ShearModulus), sigma(SurfaceLoad), psi(TidalPotential);
  ConstantCoefficient zero_gradient(0.0);
  auto rheology = IsotropicElasticRheology(dim, kappa, mu);
  std::vector<FluidRegion> fluids;
  if (fluid_core) {
    FluidRegion f;
    f.attributes = Array<int>({FluidAttribute()});
    f.density = &rho_f;
    if (no_fluid_mass) {
      f.density_gradient = &zero_gradient;
    }
    f.interface_marker = InterfaceMarker(solid);
    fluids.push_back(f);
  }

  Problem problem(&fes_u, &fes_phi, rheology, rho, 1.0, dtn_degree, nullptr,
                  fluids);
  if (load_factor != 0.0) {
    problem.SetSurfaceLoad(sigma, SurfaceMarker(solid));
  }
  if (tidal_amplitude != 0.0) {
    problem.SetTidalPotential(psi);
  }
  if (inner_core && fluid_core) {
    problem.AddRegionRotations(Array<int>({InnerCoreAttribute()}));
  }
  problem.SetRelTol(rel_tol);
  const real_t phi0_norm = L2Norm(problem.BackgroundPotential());
  if (Root()) {
    std::cout.precision(10);
    std::cout << "||Phi0|| = " << phi0_norm << "\n";
  }

  if (diagnostics) {
    const auto res = problem.RigidModeResiduals();
    real_t hi = 0.0;
    const real_t lo =
        fluid_core ? problem.PotentialBlockMinEigenvalue(40, &hi) : 0.0;
    if (Root()) {
      std::cout << "Rigid-mode residuals:";
      for (auto r : res) {
        std::cout << " " << r;
      }
      std::cout << "\n";
      if (fluid_core) {
        std::cout << "Potential block Ritz values: " << lo << " .. " << hi
                  << (lo > 0.0 ? "" : "  (INDEFINITE)") << "\n";
      }
    }
  }

  std::unique_ptr<Result> schur, minres;
  if (solver == 0 || solver == 2) {
    schur = std::make_unique<Result>(
        Run(problem, Problem::SolverType::SchurCG));
  }
  if (solver == 1 || solver == 2) {
    minres = std::make_unique<Result>(
        Run(problem, Problem::SolverType::BlockMINRES));
  }
  if (schur && minres) {
    FieldType du(schur->u), dphi(schur->phi);
    du -= minres->u;
    dphi -= minres->phi;
    const real_t eu = L2Norm(du) / L2Norm(minres->u);
    const real_t ephi = L2Norm(dphi) / L2Norm(minres->phi);
    if (Root()) {
      std::cout << "Relative difference Schur vs MINRES: u " << eu << ", phi "
                << ephi << "\n";
    }
  }

  const Result& r = minres ? *minres : *schur;
  FieldType u_dim(r.u);
  ND.UnscaleDisplacement(u_dim);
  const real_t umax = MaxAbs(u_dim);
  if (Root()) {
    std::cout << "Max displacement: " << umax << " m\n";
  }

  FieldType phi_dim(r.phi);
  ND.UnscaleGravityPotential(phi_dim);

  if (paraview) {
    // Two collections, because a collection has one mesh: the displacement
    // lives on the solid SubMesh, the potentials on the parent mesh.
    // (Fields are registered directly, in the problem's model units, rather
    // than through problem.RegisterFields, which would put the potentials in
    // the solid collection too.)
    FieldType u_out(r.u), phi_out(r.phi),
        phi0_out(static_cast<const FieldType&>(problem.BackgroundPotential()));
    ParaViewDataCollection dc_u("elastogravity_layered", &solid);
    dc_u.SetPrefixPath("ParaView");
    dc_u.SetLevelsOfDetail(order);
    dc_u.SetHighOrderOutput(true);
    dc_u.RegisterField("displacement", &u_out);
    dc_u.SetCycle(0);
    dc_u.SetTime(0.0);
    dc_u.Save();
    ParaViewDataCollection dc_phi("elastogravity_layered_potential", &parent);
    dc_phi.SetPrefixPath("ParaView");
    dc_phi.SetLevelsOfDetail(order);
    dc_phi.SetHighOrderOutput(true);
    dc_phi.RegisterField("potential", &phi_out);
    dc_phi.RegisterField("background_potential", &phi0_out);
    dc_phi.SetCycle(0);
    dc_phi.SetTime(0.0);
    dc_phi.Save();
  }
  if (visualization) {
    // The solid SubMesh can leave ranks empty; GLVisWindow skips them.
    examples::GLVisWindow win_u("Displacement [m]", examples::DefaultKeys(dim));
    examples::GLVisWindow win_phi("Potential perturbation [m^2/s^2]",
                                  examples::DefaultKeys(dim));
    win_u.Send(solid, u_dim);
    win_phi.Send(parent, phi_dim);
  }
  return 0;
}
