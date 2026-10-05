// ============================================================================
// viscoelastic_loading.cpp
//
// Loading and unloading of a layered Maxwell box: a surface load (an "ice
// sheet") presses on the top of a layered viscoelastic box for a while and
// is then removed. Built on the same building blocks as viscoelasticity.cpp
// (AdGIA/quasi_static_problem.hpp, viscoelastic.hpp): a
// LinearQuasiStaticClampedProblem (bottom clamped, load traction on top) with
// an IsotropicMaxwellRheology, stepped by the ViscoelasticOperator.
//
// It is NOT a glacial-isostatic-adjustment model: there is no gravity, so
// nothing restores the surface (no buoyancy, no isostasy). What it shows is
// the viscoelastic response alone:
//   - under the load the surface sinks at once (the elastic response), then
//     keeps sinking at a steady rate as the mantle and the channel flow:
//     with Maxwell layers and no restoring force there is no equilibrium to
//     approach, and by the time the load comes off the subsidence under its
//     centre is about four times the elastic value (defaults);
//   - when the load comes off, the surface springs back by the elastic part
//     only, then creeps up a little more as the bent elastic lithosphere
//     relaxes against the flowing layers beneath; most of the subsidence is
//     permanent.
//
// The box [0,W]x[0,H] (2-D) or [0,W]x[0,W]x[0,H] (3-D, -d 3) is split into
// three layers of element attribute by the depth of each element's centre,
// so the layer boundaries fall on element faces and the thicknesses are
// rounded to the element grid (with the default -ny 10: lithosphere 10% of
// H, asthenosphere 20%, mantle 70%):
//   3 lithosphere   - centre in the top 10% of H, elastic (relaxation time
//                     1e6: it does not relax over the run)
//   2 asthenosphere - centre in the next 15%, short relaxation time tau_a
//                     (a weak channel that lets the lithosphere flex)
//   1 mantle        - the rest, relaxation time tau_m
// all sharing bulk modulus kappa = 1 + 2/dim and shear modulus mu = 1
// (nondimensional), i.e. a classical Maxwell body (mu_inf = 0) per layer
// with a piecewise-constant relaxation time.
//
// The bottom boundary is clamped; a downward traction of amplitude p0
// loads the top boundary over |x - W/2| < a (3-D: also |y - W/2| < a) for
// 0 <= t < t_load, then switches off. The reported quantity is the vertical
// displacement at the surface point directly under the load's centre. The
// switch-off is not treated specially: the step that ends at t_load is
// taken with the load already off at its end, so the jump is spread over
// that one step (an O(dt) error in the history near t_load; see
// ViscoelasticOperator::InvalidateDisplacement for resolving a load jump
// exactly at a step boundary).
//
// Outputs: the history on the screen; viscoelastic_loading.csv (python3
// plot_csv.py viscoelastic_loading.csv); with -vis (the default), a GLVis
// animation of the vertical displacement u_z, one frame per output time,
// on a colour scale fixed from the elastic response so that subsidence
// and spring-back can be compared frame to frame.
//
// One source serves the serial and the parallel build; the genuine
// differences are the mesh partitioning (layer attributes are assigned
// on the serial mesh first, so the layering is identical) and the
// observation point, which lives on one rank and is reduced globally.
//
// Sample runs (with mpiexec -np N in front in a parallel build):
//    ./viscoelastic_loading
//    ./viscoelastic_loading -rtol 1e-3
//    ./viscoelastic_loading -d 3 -nx 12 -ny 4 -o 1 -n 10
// ============================================================================

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>

#include "AdGIA.hpp"
#include "visualisation.hpp"

using namespace std;
using namespace mfem;
using namespace AdGIA;

namespace {

#ifdef MFEM_USE_MPI
using MeshType = ParMesh;
using SpaceType = ParFiniteElementSpace;
using FieldType = ParGridFunction;
bool Root() { return Mpi::Root(); }
double GlobalMin(double v) {
  double g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  return g;
}
double GlobalMax(double v) {
  double g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return g;
}
#else
using MeshType = Mesh;
using SpaceType = FiniteElementSpace;
using FieldType = GridFunction;
bool Root() { return true; }
double GlobalMin(double v) { return v; }
double GlobalMax(double v) { return v; }
#endif

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif

  // Set the default options.
  int dim = 2;
  int order = 2;
  int nx = 40, ny = 10;
  real_t W = 4.0, H = 1.0;
  real_t a = 0.5;
  real_t p0 = 0.05;
  real_t t_load = 5.0;
  real_t t_final = 15.0;
  int n_steps = 60;
  real_t tau_a = 0.1;
  real_t tau_m = 1.0;
  real_t rtol = 0.0;
  bool paraview = false;
  bool visualization = true;
  const char* csv_file = "viscoelastic_loading.csv";

  // Read in command line options and process.
  OptionsParser args(argc, argv);
  args.AddOption(&dim, "-d", "--dimension", "Space dimension (2 or 3).");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order for the displacement.");
  args.AddOption(&nx, "-nx", "--num-elements-x",
                 "Elements across the width (3-D: also across the depth "
                 "direction perpendicular to the load strip).");
  args.AddOption(&ny, "-ny", "--num-elements-y",
                 "Elements through the depth H.");
  args.AddOption(&W, "-W", "--width", "Width of the box.");
  args.AddOption(&H, "-H", "--depth", "Depth of the box.");
  args.AddOption(&a, "-a", "--load-half-width",
                 "Half-width of the surface load patch, |x - W/2| < a.");
  args.AddOption(&p0, "-p0", "--load-amplitude",
                 "Amplitude of the downward surface traction.");
  args.AddOption(&t_load, "-tl", "--load-time",
                 "Duration the load is applied (t < t_load).");
  args.AddOption(&t_final, "-tf", "--t-final", "Final time.");
  args.AddOption(&n_steps, "-n", "--n-steps",
                 "Number of time steps (fixed dt), or output times when "
                 "-rtol > 0.");
  args.AddOption(&tau_a, "-tau-a", "--tau-asthenosphere",
                 "Relaxation time of the asthenosphere channel.");
  args.AddOption(&tau_m, "-tau-m", "--tau-mantle",
                 "Relaxation time of the mantle.");
  args.AddOption(&rtol, "-rtol", "--adaptive-rtol",
                 "Relative tolerance of adaptive stepping (0: fixed dt).");
  args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                 "Save time slices to a ParaView data collection.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization", "Animate u_z in GLVis.");
  args.AddOption(&csv_file, "-csv", "--csv",
                 "Table of the history for plot_csv.py (\"\": none).");
  args.Parse();
  if (!args.Good()) {
    if (Root()) {
      args.PrintUsage(cout);
    }
    return 1;
  }
  if (Root()) {
    args.PrintOptions(cout);
  }
  MFEM_VERIFY(dim == 2 || dim == 3, "Dimension must be 2 or 3.");

  // Build the box mesh: a rectangle in 2-D, a square-based box in 3-D so
  // that the load patch can be centred at (W/2, W/2) on the top face.
  Mesh smesh = dim == 2
                 ? Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL,
                                        false, W, H)
                 : Mesh::MakeCartesian3D(nx, nx, ny, Element::HEXAHEDRON, W,
                                        W, H);

  // Boundary attributes: 2-D bottom/right/top/left = 1/2/3/4; 3-D
  // bottom/front/right/back/left/top = 1/2/3/4/5/6. In both cases the
  // bottom is attribute 1; the top is 3 (2-D) or 6 (3-D).
  const int bottom_attr = 1;
  const int top_attr = dim == 2 ? 3 : 6;

  // Assign the three layers by element-centre depth: lithosphere (top 10%
  // of H, elastic), asthenosphere (next 15%), mantle (remaining 75%).
  // (On the serial mesh, before partitioning: the layering is then
  // identical in both builds.)
  Vector center(dim);
  for (int e = 0; e < smesh.GetNE(); e++) {
    smesh.GetElementCenter(e, center);
    const real_t frac = center(dim - 1) / H;
    int attr;
    if (frac >= 0.9) {
      attr = 3;  // lithosphere
    } else if (frac >= 0.75) {
      attr = 2;  // asthenosphere
    } else {
      attr = 1;  // mantle
    }
    smesh.SetAttribute(e, attr);
  }
  smesh.SetAttributes();
#ifdef MFEM_USE_MPI
  MeshType mesh(MPI_COMM_WORLD, smesh);
  smesh.Clear();
#else
  MeshType& mesh = smesh;
#endif

  // Material: uniform bulk and (unrelaxed branch) shear modulus, a
  // piecewise-constant Maxwell relaxation time by attribute (mantle,
  // asthenosphere, lithosphere). The lithosphere's time is set far beyond
  // t_final so it behaves elastically over the run.
  ConstantCoefficient kappa(1.0 + 2.0 / dim), mu(1.0);
  Vector tau_vals(3);
  tau_vals(0) = tau_m;  // attribute 1: mantle
  tau_vals(1) = tau_a;  // attribute 2: asthenosphere
  tau_vals(2) = 1.0e6;  // attribute 3: lithosphere (effectively elastic)
  PWConstCoefficient tau(tau_vals);
  IsotropicMaxwellRheology rheology =
      IsotropicMaxwellRheology::Maxwell(dim, kappa, mu, tau);

  // Displacement space.
  H1_FECollection fec(order, dim);
  SpaceType fes(&mesh, &fec, dim);

  // Bottom clamped; downward traction on the top over the load patch,
  // switched off at t_load. In 3-D the patch is a square,
  // |x - W/2| < a and |y - W/2| < a.
  Array<int> ess_bdr(mesh.bdr_attributes.Max());
  ess_bdr = 0;
  ess_bdr[bottom_attr - 1] = 1;

  Array<int> traction_marker(mesh.bdr_attributes.Max());
  traction_marker = 0;
  traction_marker[top_attr - 1] = 1;

  VectorFunctionCoefficient traction(
      dim, [dim, W, a, p0, t_load](const Vector& x, real_t t, Vector& f) {
        f = 0.0;
        const bool in_x = std::abs(x(0) - 0.5 * W) < a;
        const bool in_patch = dim == 2 ? in_x : (in_x && std::abs(x(1) - 0.5 * W) < a);
        if (t < t_load && in_patch) {
          f(dim - 1) = -p0;
        }
      });

  LinearQuasiStaticClampedProblem problem(&fes, rheology, ess_bdr, traction,
                                          traction_marker);

  ViscoelasticOperator visco(problem);

  // Find the mesh vertex nearest the surface point under the load centre
  // (W/2, H), or (W/2, W/2, H) in 3-D, once at the start. For H1_FECollection
  // spaces the first NV scalar dofs coincide with the mesh vertices for any
  // order (vertex-based basis functions are numbered first and are nodal
  // there), so the observation vdof is simply fes.DofToVDof(vertex, dim-1);
  // no interpolation is needed.
  Vector target(dim);
  target(0) = 0.5 * W;
  if (dim == 3) {
    target(1) = 0.5 * W;
  }
  target(dim - 1) = H;

  int vbest = -1;
  real_t dmin = numeric_limits<real_t>::infinity();
  for (int v = 0; v < mesh.GetNV(); v++) {
    const real_t* vc = mesh.GetVertex(v);
    real_t dist2 = 0.0;
    for (int c = 0; c < dim; c++) {
      const real_t diff = vc[c] - target(c);
      dist2 += diff * diff;
    }
    if (dist2 < dmin) {
      dmin = dist2;
      vbest = v;
    }
  }
  // In parallel the observation vertex lives on one rank (possibly
  // shared, with equal values): reduce with a -inf sentinel elsewhere.
  const real_t dmin_global = GlobalMin(dmin);
  const bool observer = dmin == dmin_global;
  const int obs_vdof = observer ? fes.DofToVDof(vbest, dim - 1) : -1;
  auto observe = [&]() {
    const real_t v = observer ? problem.Displacement()(obs_vdof)
                              : -numeric_limits<real_t>::infinity();
    return GlobalMax(v);
  };

  // Optional ParaView output.
  ParaViewDataCollection dc("viscoelastic_loading", &mesh);
  if (paraview) {
    dc.SetPrefixPath("ParaView");
    dc.SetLevelsOfDetail(order);
    dc.SetHighOrderOutput(true);
    visco.RegisterFields(dc);
  }

  // The vertical displacement as a scalar field for GLVis (the H1 vertical
  // component, interpolated nodally into the scalar space of the same
  // order: exact), and the history table.
  SpaceType fes_z(&mesh, &fec);
  FieldType u_z(&fes_z);
  VectorGridFunctionCoefficient u_c(&problem.Displacement());
  Vector e_z(dim);
  e_z = 0.0;
  e_z(dim - 1) = 1.0;
  VectorConstantCoefficient e_z_c(e_z);
  InnerProductCoefficient u_z_c(u_c, e_z_c);
  examples::GLVisWindow window("u_z (load on for t < " +
                                   std::to_string(t_load).substr(0, 4) + ")",
                               examples::DefaultKeys(dim));
  examples::CsvTable table(csv_file, {"t", "u_z"});
  table.Meta("title", "Surface displacement under the load centre")
      .Meta("note", "load on for t < " + std::to_string(t_load).substr(0, 4) +
                        ", then removed")
      .Meta("xlabel", "time")
      .Meta("ylabel", "u_z(load centre)");
  auto show = [&](real_t t_now, real_t uz, bool first) {
    table.Row({t_now, uz});
    if (!visualization) {
      return;
    }
    u_z.ProjectCoefficient(u_z_c);
    if (first) {
      // The elastic subsidence sets the scale; relaxation deepens it
      // (about 4x at the defaults), so leave room below.
      window.SetValueRange(5.0 * uz, -0.5 * uz);
    }
    window.Send(mesh, u_z);
  };

  // Time stepping: fixed exponential-trapezoid steps of dt = t_final /
  // n_steps by default, or the adaptive exponential trapezoid solver
  // integrating between the same n_steps output times when -rtol > 0.
  ExponentialTrapezoidSolver ode;
  AdaptiveExponentialTrapezoidSolver adaptive;
  const bool use_adaptive = rtol > 0.0;
  if (use_adaptive) {
    adaptive.Init(visco);
    adaptive.SetTolerances(rtol, 1e-12);
  } else {
    ode.Init(visco);
  }

  real_t t = 0.0;
  real_t dt = t_final / n_steps;
  Vector m(visco.Height());
  m = 0.0;

  if (Root()) {
    cout << "\n  step          t     u_z(load centre)\n"
         << "  ----  ---------  --------------------\n";
  }

  // Initial state: relaxed internal variable, elastic response at t = 0
  // (the load is already active there since t_load > 0).
  if (!visco.SolveElastic(m, t)) {
    if (Root()) {
      cerr << "Elastic solve failed at t = " << t << "\n";
    }
    return 2;
  }
  visco.SyncFields(m);
  cout.precision(6);
  {
    const real_t uz = observe();
    if (Root()) {
      cout << "  " << 0 << "  " << fixed << setw(9) << t << "  " << setw(20)
           << scientific << uz << "\n";
    }
    show(t, uz, true);
  }
  if (paraview) {
    dc.SetCycle(0);
    dc.SetTime(t);
    dc.Save();
  }

  for (int step = 1; step <= n_steps; step++) {
    if (use_adaptive) {
      const real_t t_target = step * t_final / n_steps;
      adaptive.Integrate(m, t, t_target, dt);
    } else {
      ode.Step(m, t, dt);
    }

    if (!visco.SolveElastic(m, t)) {
      if (Root()) {
        cerr << "Elastic solve failed at t = " << t << "\n";
      }
      return 2;
    }
    visco.SyncFields(m);

    const real_t uz = observe();
    if (Root()) {
      cout << "  " << step << "  " << fixed << setw(9) << t << "  "
           << setw(20) << scientific << uz << "\n";
    }
    show(t, uz, false);

    if (paraview) {
      dc.SetCycle(step);
      dc.SetTime(t);
      dc.Save();
    }
  }

  if (Root()) {
    cout << "\nSolves:                 " << problem.NumSolves()
         << "\nAssemblies:              " << problem.NumAssemblies()
         << "\nPreconditioner setups:   " << problem.NumPreconditionerSetups()
         << "\n";
  }
  table.Write();
  return 0;
}
