// ============================================================================
// self_gravitating_relaxation.cpp
//
// Viscoelastic relaxation of a self-gravitating layered Earth model with a
// fluid outer core: the layered models of layered_model.hpp (two- or
// three-layer meshes, see elastogravity_layered.cpp), a Maxwell mantle with
// a given viscosity, in the three-layer model an elastic inner core (a
// CompositeRheology: the inner core carries an elastic rheology, the mantle
// a Maxwell one), and a surface mass load switched on at t = 0 (a Heaviside
// load: the elastic response is followed by the viscous relaxation towards
// isostasy). ViscoelasticOperator
// runs on LinearQuasiStaticMixedSelfGravitatingProblem unchanged; the potential
// and the fluid core come along for free.
//
// Time is measured in Maxwell times of the mantle, tau = eta / mu evaluated
// with the mantle's mean shear modulus; the run prints, at every step, the
// L2 norms of the displacement and of the potential perturbation and the
// radial surface displacement under the load's maximum (theta = 0).
//
// Outputs: the table on the screen; self_gravitating_relaxation.csv, the
// radial displacement at the pole and the two norms against time (python3
// plot_csv.py self_gravitating_relaxation.csv); with -vis (the default), a
// GLVis animation of the mantle displacement, one frame per output time.
//
// One source serves the serial and the parallel build; the genuine
// differences are the mesh partitioning and the pole observation point,
// which in parallel lives on one rank and is reduced globally.
//
// Options (defaults in brackets):
//   -m      two- or three-layer mesh
//           [../data/elastogravity_two_layer_2d.msh].
//   -o      finite element order [1].
//   -deg    truncation degree of the DtN expansion [16].
//   -rt     relative solver tolerance [1e-9].
//   -eta    mantle viscosity [1e21 Pa s]; with the mantle's mean shear
//           modulus it sets the Maxwell time tau = eta / mu in which
//           the run is measured.
//   -tf     final time, in Maxwell times of the mantle [5].
//   -n      number of steps (output times when adaptive) [20].
//   -rtol   relative tolerance of adaptive stepping [0: fixed dt].
//   -pv / -no-pv   write a ParaView data collection [off].
//   -vis / -no-vis   animate the mantle displacement in GLVis [on].
//   -csv    file for the time history, read by plot_csv.py
//           ["self_gravitating_relaxation.csv"; "": none].
//
// Sample runs (with mpiexec -np N in front in a parallel build):
//    ./self_gravitating_relaxation -o 2 -n 20 -tf 5
//    ./self_gravitating_relaxation -m ../data/elastogravity_three_layer_2d.msh
//    -o 2
//    ./self_gravitating_relaxation -m ../data/elastogravity_three_layer_3d.msh
//    -o 1 -n 10
//    ./self_gravitating_relaxation -o 2 -rtol 1e-3
//    ./self_gravitating_relaxation -o 2 -eta 3e21 -pv
// ============================================================================

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>

#include "layered_model.hpp"
#include "AdGIA.hpp"
#include "visualisation.hpp"

using namespace mfem;
using namespace AdGIA;
using namespace layered;

namespace {

#ifdef MFEM_USE_MPI
using MeshType = ParMesh;
using SubMeshType = ParSubMesh;
using SpaceType = ParFiniteElementSpace;
bool Root() { return Mpi::Root(); }
double GlobalMax(double v) {
  double g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return g;
}
#else
using MeshType = Mesh;
using SubMeshType = SubMesh;
using SpaceType = FiniteElementSpace;
bool Root() { return true; }
double GlobalMax(double v) { return v; }
#endif

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif

  const char* mesh_file = "../data/elastogravity_two_layer_2d.msh";
  int order = 1;
  int dtn_degree = 16;
  real_t rel_tol = 1e-9;
  real_t eta_dim = 1e21;  // mantle viscosity [Pa s]
  real_t t_final = 5.0;   // in Maxwell times
  int n_steps = 20;
  real_t rtol = 0.0;
  bool paraview = false;
  bool visualization = true;
  const char* csv_file = "self_gravitating_relaxation.csv";

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Two- or three-layer mesh.");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&dtn_degree, "-deg", "--dtn-degree",
                 "Truncation degree of the DtN expansion.");
  args.AddOption(&rel_tol, "-rt", "--rel-tol", "Relative solver tolerance.");
  args.AddOption(&eta_dim, "-eta", "--viscosity", "Mantle viscosity [Pa s].");
  args.AddOption(&t_final, "-tf", "--t-final", "Final time [Maxwell times].");
  args.AddOption(&n_steps, "-n", "--n-steps",
                 "Number of steps (output times when adaptive).");
  args.AddOption(&rtol, "-rtol", "--adaptive-rtol",
                 "Relative tolerance of adaptive stepping (0: fixed dt).");
  args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                 "Write a ParaView data collection.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization", "Animate the displacement in GLVis.");
  args.AddOption(&csv_file, "-csv", "--csv",
                 "Table of the history for plot_csv.py (\"\": none).");
  args.Parse();
  if (!args.Good()) {
    if (Root()) {
      args.PrintUsage(std::cout);
    }
    return 1;
  }
  if (Root()) {
    args.PrintOptions(std::cout);
  }

  Mesh smesh(mesh_file, 1, 1);
  const int dim = smesh.Dimension();
  if (!SetModel(smesh) || uniform) {
    if (Root()) {
      std::cerr << "Expected a two-layer (3 attributes) or three-layer "
                   "(4 attributes) mesh.\n";
    }
    return 1;
  }
#ifdef MFEM_USE_MPI
  MeshType parent(MPI_COMM_WORLD, smesh);
  smesh.Clear();
#else
  MeshType& parent = smesh;
#endif
  Array<int> solid_attrs = SolidAttributes();
  auto solid = SubMeshType::CreateFromDomain(parent, solid_attrs);
  H1_FECollection fec(order, dim);
  SpaceType fes_u(&solid, &fec, dim), fes_phi(&parent, &fec);

  // Material. The Maxwell time of the mantle from its mean shear modulus,
  // applied uniformly: tau is constant, so the implied viscosity mu tau
  // varies with radius as the shear modulus does and equals eta only at the
  // mean modulus. With an inner core the rheology is a composite: elastic
  // in the inner core, Maxwell in the mantle (the same kappa and mu
  // coefficients serve both; each region reads its own radii).
  const real_t tau_dim = eta_dim / MantleMeanShearModulusDim();
  const real_t tau_nd = tau_dim / ND.Time();
  FunctionCoefficient rho(Density), rho_f(FluidDensity), kappa(BulkModulus),
      mu(ShearModulus), sigma(SurfaceLoad);
  ConstantCoefficient tau(tau_nd);
  auto mantle = IsotropicMaxwellRheology::Maxwell(dim, kappa, mu, tau);
  IsotropicElasticRheology core(dim, kappa, mu);
  std::vector<RheologyRegion> regions;
  {
    Array<int> mantle_marker(solid.attributes.Max()), core_marker;
    mantle_marker = 0;
    mantle_marker[MantleAttribute() - 1] = 1;
    regions.push_back({mantle_marker, &mantle, "mantle"});
    if (inner_core) {
      core_marker.SetSize(solid.attributes.Max());
      core_marker = 0;
      core_marker[InnerCoreAttribute() - 1] = 1;
      regions.push_back({core_marker, &core, "inner_core"});
    }
  }
  CompositeRheology rheology(dim, regions);
  std::vector<FluidRegion> fluids;
  {
    FluidRegion f;
    f.attributes = Array<int>({FluidAttribute()});
    f.density = &rho_f;
    f.interface_marker = InterfaceMarker(solid);
    fluids.push_back(f);
  }
  LinearQuasiStaticMixedSelfGravitatingProblem problem(
      &fes_u, &fes_phi, rheology, rho, 1.0, dtn_degree, nullptr, fluids);
  // A Heaviside load: the surface load coefficient is constant in time, so
  // switching it on at t = 0 is simply starting from an unloaded state.
  problem.SetSurfaceLoad(sigma, SurfaceMarker(solid));
  if (inner_core) {
    problem.AddRegionRotations(Array<int>({InnerCoreAttribute()}));
  }
  problem.SetRelTol(rel_tol);
  if (Root()) {
    std::cout << (inner_core ? "Three-layer" : "Two-layer") << " model, "
              << dim << "-D; mantle Maxwell time "
              << tau_dim / (365.25 * 86400.0) << " yr (" << tau_nd
              << " time units)\n";
  }

  ViscoelasticOperator visco(problem);
  ExponentialTrapezoidSolver ode;
  ode.Init(visco);
  AdaptiveExponentialTrapezoidSolver adaptive;
  if (rtol > 0.0) {
    adaptive.Init(visco);
    adaptive.SetTolerances(rtol, 1e-14);
  }

  // Observation point: the surface vertex nearest to the pole (theta = 0).
  // In parallel the pole lives on one rank (possibly shared): every rank
  // finds its local best, the global pole is the reduced maximum, and the
  // observed value is reduced with a -inf sentinel from the other ranks.
  int pole = -1;
  real_t pole_z = -std::numeric_limits<real_t>::infinity();
  for (int v = 0; v < solid.GetNV(); v++) {
    const real_t* x = solid.GetVertex(v);
    const real_t z = x[dim - 1];
    if (z > pole_z) {
      pole_z = z;
      pole = v;
    }
  }
  const real_t pole_z_global = GlobalMax(pole_z);
  auto radial_at_pole = [&]() {
    // Order-1 vertex dof = vertex index; for higher orders too, since the
    // vertex dofs come first in H1 spaces.
    real_t v = -std::numeric_limits<real_t>::infinity();
    if (pole >= 0 && pole_z == pole_z_global) {
      const GridFunction& u = problem.Displacement();
      v = u[fes_u.DofToVDof(pole, dim - 1)] * ND.Length();
    }
    return GlobalMax(v);
  };

  ParaViewDataCollection dc("self_gravitating_relaxation", &solid);
  if (paraview) {
    dc.SetPrefixPath("ParaView");
    dc.SetLevelsOfDetail(order);
    dc.SetHighOrderOutput(true);
    visco.RegisterFields(dc);
  }

  Vector m(visco.Height());
  m = 0.0;
  real_t t = 0.0;
  real_t dt = t_final * tau_nd / n_steps;
  real_t dt_adaptive = 0.1 * dt;

  // The elastic response at t = 0+.
  if (!visco.SolveElastic(m, t)) {
    if (Root()) {
      std::cerr << "Elastic solve failed.\n";
    }
    return 2;
  }
  std::cout.precision(6);
  if (Root()) {
    std::cout << "t/tau        ||u||        ||phi||   u_r(pole) [m]\n";
  }
  examples::GLVisWindow window("mantle displacement (Heaviside load)",
                               examples::DefaultKeys(dim));
  examples::CsvTable table(csv_file, {"t/tau", "u_r_pole", "u_L2",
                                      "phi_L2"});
  table.Meta("title", "Relaxation of a self-gravitating layered model")
      .Meta("note", "Maxwell mantle, fluid core, Heaviside surface load")
      .Meta("xlabel", "time / mantle Maxwell time")
      .Meta("y", "u_r_pole|u_L2,phi_L2")
      .Meta("ylabel", "u_r at the pole [m]|L2 norms (non-dim.)");
  auto report = [&](int cycle) {
    visco.SyncFields(m);
    const double un = L2Norm(problem.Displacement());
    const double pn = L2Norm(problem.Potential());
    const double ur = radial_at_pole();
    if (Root()) {
      std::cout << std::setw(6) << t / tau_nd << std::setw(14) << un
                << std::setw(14) << pn << std::setw(14) << ur << "\n";
    }
    table.Row({t / tau_nd, ur, un, pn});
    if (visualization) {
      window.Send(solid, problem.Displacement());
    }
    if (paraview) {
      dc.SetCycle(cycle);
      dc.SetTime(t / tau_nd);
      dc.Save();
    }
  };
  report(0);
  const auto w0 = std::chrono::steady_clock::now();
  for (int step = 1; step <= n_steps; step++) {
    if (rtol > 0.0) {
      adaptive.Integrate(m, t, step * t_final * tau_nd / n_steps, dt_adaptive);
    } else {
      ode.Step(m, t, dt);
    }
    if (!visco.SolveElastic(m, t)) {
      if (Root()) {
        std::cerr << "Elastic solve failed at t = " << t << "\n";
      }
      return 2;
    }
    report(step);
  }
  const auto w1 = std::chrono::steady_clock::now();
  if (Root()) {
    std::cout << "Solves " << problem.NumSolves() << ", assemblies "
              << problem.NumAssemblies() << ", preconditioner setups "
              << problem.NumPreconditionerSetups() << ", "
              << std::chrono::duration<double>(w1 - w0).count() << " s";
    if (rtol > 0.0) {
      std::cout << "; adaptive steps " << adaptive.NumAcceptedSteps()
                << " accepted, " << adaptive.NumRejectedSteps() << " rejected";
    }
    std::cout << "\n";
  }
  table.Write();
  return 0;
}
