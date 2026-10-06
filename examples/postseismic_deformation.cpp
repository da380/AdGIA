// ============================================================================
// postseismic_deformation.cpp
//
// Post-seismic relaxation of the coseismic_deformation.cpp problem: the
// same Cartesian box, free surface, elastic structure and finite fault
// (fault_box.hpp), but the medium is an elastic lid over a standard
// linear solid (SLS) and the stress glut, switched on in full at t = 0
// and held, drives a viscoelastic transient. At t = 0 the response is
// the coseismic (unrelaxed) deformation of the coseismic example; as the
// SLS relaxes, the displacement creeps towards the fully relaxed limit —
// the elastic solution with the lid at its unrelaxed modulus and the
// mantle at mu_inf — which the example also solves directly and checks
// against (printed as the distance to the relaxed limit, O(exp(-tf /
// tau_e)) when the run is long enough).
//
// The machinery: a LinearQuasiStaticClampedProblem whose ExternalLoad()
// carries the moment-tensor point sources
// (DomainLFDeformationGradientIntegrator on MatrixDeltaCoefficients),
// an IsotropicMaxwellRheology with mu_inf > 0 and one Prony branch (the
// SLS), stepped by the ViscoelasticOperator with the exponential
// trapezoid solver (adaptive with -rtol > 0).
//
// The SLS is parameterised by its two relaxation times: tau_s (-tau-s),
// the stress-relaxation time of the branch, and tau_e (-tau-e), the
// creep-retardation time, whose ratio fixes the relaxed modulus,
// mu_inf = mu_U tau_s / tau_e (tau_e > tau_s). Both times are scaled by
// ONE parametric factor of the same form as the elastic structure
// (-tbeta, -talpha, -tgamma...), so the relaxation speed varies in space
// while the relaxed modulus keeps tracking the unrelaxed one. The lid is
// the top -nl element ROWS of the mesh — the material discontinuity at
// its base is a surface of element faces by construction, bent with the
// topography — and is made elastic by a huge branch time there
// (PWCoefficient by attribute), exactly the viscoelastic_loading.cpp
// lithosphere pattern. The default lid contains the default fault, so
// the transient is driven purely by relaxation beneath the rupture.
//
// The arguments shared with coseismic_deformation.cpp (see its header
// for the details; same defaults):
//   -d -o                      dimension and order                 [2, 2]
//   -nx -nz -W -D              the box                   [64/24, 32/12, 8, 4]
//   -dip -rake -strike -fd -fx -fl -fw -s -nd -ns -smooth -taper
//            the fault (-smooth > 0: Gaussian densities of that many
//            element widths instead of point sources — no near-fault
//            delta artifacts, same far field; -taper, on by default:
//            cos^2 slip taper to zero at the fault ends, -s the peak)
//   -beta -alpha -gamma -bx -bz -br                 the elastic structure
//   -topo -tx -tr                                   the topography
// and its own (defaults in brackets):
//   -nl      lid thickness in element rows (0: no lid;
//            -1: 3 nz / 8, rounded up)                         [-1]
//   -tau-s   stress-relaxation time tau_s of the SLS           [1]
//   -tau-e   creep-retardation time tau_e (> tau_s); the
//            relaxed modulus is mu_inf = mu_U tau_s / tau_e    [2]
//   -tbeta   both times grow by 1 + tbeta |z| / D              [0]
//   -talpha  both times grow by 1 + talpha x / W               [0]
//   -tgamma  Gaussian weak zone: both times reduced by the
//            factor 1 - tgamma exp(-r^2 / 2 tbr^2)  (< 1)      [0]
//   -tbx     x of the weak-zone centre                         [0]
//   -tbz     depth of the weak-zone centre below z = 0         [1.5]
//   -tbr     Gaussian radius of the weak zone                  [1]
//   -tf      final time                                        [10]
//   -n       number of time steps (fixed dt), or output times
//            when -rtol > 0                                    [50]
//   -rtol    relative tolerance of adaptive stepping (0: fixed
//            dt)                                               [0]
//   -ox      x of the surface observation point (1e30: the
//            coseismic uplift peak)                            [1e30]
//   -pv      save time slices to a ParaView collection         [off]
//   -vis / -no-vis   GLVis windows                             [on]
//   -anim / -no-anim save the u_z frames and a replay script   [on]
//   -csv     history table ("": off)   [postseismic_deformation.csv]
//   -csvp    profiles table ("": off)      [postseismic_profiles.csv]
//
// Output: the history of u_z at the observation point on the screen and
// in the history CSV (t, u_x, u_z); the profiles CSV holds the surface
// u_z at t = 0 (coseismic), at t = tf, and the directly solved relaxed
// limit as the _exact reference (dashed in plot_csv.py); with -vis,
// three GLVis windows sharing one colour scale fixed from the coseismic
// surface peaks times the growth bound tau_e / tau_s: the coseismic u_z
// (t = 0), the relaxation ANIMATION (one frame per output time), and the
// final u_z. A streamed animation cannot be replayed (GLVis keeps only
// the last frame; the space bar pauses a LIVE stream), so with -anim
// (the default) the frames are also saved to postseismic_frames/ with a
// GLVis script:  glvis -run postseismic_deformation.glvs  replays the
// animation at leisure, the space bar playing and pausing it.
//
// One source serves the serial and the parallel build; the genuine
// differences are the mesh partitioning (mesh built, labelled and mapped
// serially first), the solvers inside the problem class (Gauss-Seidel /
// BoomerAMG), and the reductions of the sampled profile.
//
// Sample runs (with mpiexec -np N in front in a parallel build):
//    ./postseismic_deformation
//    ./postseismic_deformation -tau-e 4 -tf 20
//    ./postseismic_deformation -tgamma 0.9 -tbx 1 -tbz 2    (weak zone)
//    ./postseismic_deformation -tbeta 4                     (slow at depth)
//    ./postseismic_deformation -beta 2 -alpha 1 -topo 0.3
//    ./postseismic_deformation -d 3 -o 1 -rake 0 -n 25
// ============================================================================

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "AdGIA.hpp"
#include "fault_box.hpp"
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
  examples::FaultBox box;
  examples::Fault fault;
  int lid_rows = -1;
  real_t tau_s = 1.0, tau_e = 2.0;
  examples::ParametricFactor tau_var;
  tau_var.cz = 1.5;  // a weak zone defaults to below the default lid
  real_t t_final = 10.0;
  int n_steps = 50;
  real_t rtol = 0.0;
  real_t obs_x = 1e30;
  bool paraview = false;
  bool visualization = true;
  bool anim = true;
  const char* csv_file = "postseismic_deformation.csv";
  const char* profiles_file = "postseismic_profiles.csv";

  // Read in command line options and process.
  OptionsParser args(argc, argv);
  args.AddOption(&dim, "-d", "--dimension", "Space dimension (2 or 3).");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order for the displacement.");
  box.AddOptions(args);
  fault.AddOptions(args);
  args.AddOption(&lid_rows, "-nl", "--lid-rows",
                 "Elastic lid thickness in element rows (0: none; -1: "
                 "3 nz / 8, rounded up).");
  args.AddOption(&tau_s, "-tau-s", "--tau-stress",
                 "Stress-relaxation time of the SLS branch.");
  args.AddOption(&tau_e, "-tau-e", "--tau-creep",
                 "Creep-retardation time (> tau-s); sets mu_inf = mu_U "
                 "tau_s / tau_e.");
  args.AddOption(&tau_var.beta, "-tbeta", "--tau-depth",
                 "Both times grow by 1 + tbeta |z| / D.");
  args.AddOption(&tau_var.alpha, "-talpha", "--tau-lateral",
                 "Both times grow by 1 + talpha x / W (|talpha| < 2).");
  args.AddOption(&tau_var.gamma, "-tgamma", "--tau-weak-zone",
                 "Relative reduction of both times at the weak-zone "
                 "centre (< 1).");
  args.AddOption(&tau_var.cx, "-tbx", "--tau-weak-x",
                 "x of the weak-zone centre.");
  args.AddOption(&tau_var.cz, "-tbz", "--tau-weak-depth",
                 "Depth of the weak-zone centre below z = 0.");
  args.AddOption(&tau_var.radius, "-tbr", "--tau-weak-radius",
                 "Gaussian radius of the weak zone.");
  args.AddOption(&t_final, "-tf", "--t-final", "Final time.");
  args.AddOption(&n_steps, "-n", "--n-steps",
                 "Number of time steps (fixed dt), or output times when "
                 "-rtol > 0.");
  args.AddOption(&rtol, "-rtol", "--adaptive-rtol",
                 "Relative tolerance of adaptive stepping (0: fixed dt).");
  args.AddOption(&obs_x, "-ox", "--observation-x",
                 "x of the surface observation point (1e30: the coseismic "
                 "uplift peak).");
  args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                 "Save time slices to a ParaView data collection.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Show the coseismic, animated and final u_z in GLVis.");
  args.AddOption(&anim, "-anim", "--save-animation", "-no-anim",
                 "--no-save-animation",
                 "Save the frames and a GLVis script, for replay with "
                 "glvis -run postseismic_deformation.glvs (space "
                 "plays/pauses).");
  args.AddOption(&csv_file, "-csv", "--csv",
                 "Table of the history for plot_csv.py (\"\": none).");
  args.AddOption(&profiles_file, "-csvp", "--csv-profiles",
                 "Table of the initial/final profiles (\"\": none).");
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
  box.Finalize(dim);
  MFEM_VERIFY(tau_s > 0.0 && tau_e > tau_s,
              "The SLS needs 0 < tau-s < tau-e.");
  tau_var.Verify(box.D, box.h0, "relaxation-time");
  if (lid_rows < 0) {
    lid_rows = (3 * box.nz + 7) / 8;
  }
  const real_t lid_depth = lid_rows * box.D / box.nz;

  // Build the box mesh, mapped and centred, the top lid_rows element
  // rows labelled attribute 2 (on the serial mesh, before partitioning).
  Mesh smesh = box.MakeMesh(dim, lid_rows);
  const int top_attr = box.TopAttribute(dim);
#ifdef MFEM_USE_MPI
  MeshType mesh(MPI_COMM_WORLD, smesh);
  smesh.Clear();
#else
  MeshType& mesh = smesh;
#endif

  // Displacement space.
  H1_FECollection fec(order, dim);
  SpaceType fes(&mesh, &fec, dim);
#ifdef MFEM_USE_MPI
  const auto n_u = fes.GlobalTrueVSize();
#else
  const auto n_u = fes.GetTrueVSize();
#endif
  if (Root()) {
    cout << "Displacement unknowns: " << n_u << "\nLid: " << lid_rows
         << " element rows, depth " << lid_depth << "\n";
  }

  // Material: the SLS  sigma = kappa tr(eps) I + 2 mu_inf dev eps
  // + 2 mu_1 (dev eps - m),  tau m' = dev eps - m,  with every
  // coefficient the elastic-structure factor times a constant:
  // mu_U = mu_inf + mu_1 matches the coseismic example's shear modulus
  // and kappa its bulk modulus, so the t = 0 response IS the coseismic
  // deformation; mu_inf / mu_U = tau_s / tau_e. The branch time is the
  // tau factor times tau_s in the mantle (attribute 1) and effectively
  // infinite in the lid (attribute 2), which therefore stays elastic.
  const real_t lambda0 = 1.0, mu0 = 1.0;
  const real_t relaxed = tau_s / tau_e;  // mu_inf / mu_U
  auto modulus_factor = box.ModulusFactor(dim);
  FunctionCoefficient kappa_c([=](const Vector& x) {
    return (lambda0 + 2.0 * mu0 / dim) * modulus_factor(x);
  });
  FunctionCoefficient mu_inf_c(
      [=](const Vector& x) { return relaxed * mu0 * modulus_factor(x); });
  FunctionCoefficient mu_1_c([=](const Vector& x) {
    return (1.0 - relaxed) * mu0 * modulus_factor(x);
  });
  auto tau_factor = tau_var.Make(dim, box.W, box.D);
  FunctionCoefficient tau_mantle(
      [=](const Vector& x) { return tau_s * tau_factor(x); });
  ConstantCoefficient tau_lid(1e9);
  PWCoefficient tau;
  tau.UpdateCoefficient(1, tau_mantle);
  tau.UpdateCoefficient(2, tau_lid);
  MaxwellBranch branch{&mu_1_c, &tau, nullptr};
  IsotropicMaxwellRheology rheology(dim, kappa_c, mu_inf_c, {branch});

  // Clamp every boundary but the free surface; no surface traction.
  Array<int> ess_bdr(mesh.bdr_attributes.Max());
  ess_bdr = 1;
  ess_bdr[top_attr - 1] = 0;
  Vector zero_vec(dim);
  zero_vec = 0.0;
  VectorConstantCoefficient zero_traction(zero_vec);
  Array<int> no_traction(mesh.bdr_attributes.Max());
  no_traction = 0;
  LinearQuasiStaticClampedProblem problem(&fes, rheology, ess_bdr,
                                          zero_traction, no_traction);

  // The stress glut, on in full from t = 0: the fault's point sources —
  // or its smoothed density, with a rule fine enough for a Gaussian of
  // about an element width — in the problem's external load, with the
  // UNRELAXED local shear modulus in each patch moment.
  auto sources = fault.Build(
      dim, box, [&](const Vector& xp) { return mu0 * modulus_factor(xp); });
  auto add_glut = [&](LinearForm& load) {
    if (sources.smoothed) {
      load.AddDomainIntegrator(new DomainLFDeformationGradientIntegrator(
          *sources.smoothed,
          &IntRules.Get(dim == 2 ? Geometry::SQUARE : Geometry::CUBE,
                        2 * order + 6)));
    } else {
      for (auto& src : sources.patches) {
        load.AddDomainIntegrator(
            new DomainLFDeformationGradientIntegrator(*src));
      }
    }
  };
  add_glut(problem.ExternalLoad());
  if (Root()) {
    cout << "Fault patches: " << fault.nd * (dim == 2 ? 1 : fault.ns)
         << (sources.smoothed ? " (smoothed)" : "")
         << ", total scalar moment M0 = " << sources.moment
         << (dim == 2 ? " (per unit strike length)" : "") << "\n";
  }

  ViscoelasticOperator visco(problem);

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

  // The surface profile sampler, the u_z scalar field and the windows.
  examples::SurfaceProfile profile(mesh, box, dim);
  const int npts = profile.NumPoints();
  SpaceType fes_z(&mesh, &fec);
  FieldType u_z(&fes_z);
  VectorGridFunctionCoefficient u_c(&problem.Displacement());
  Vector e_z(dim);
  e_z = 0.0;
  e_z(dim - 1) = 1.0;
  VectorConstantCoefficient e_z_c(e_z);
  InnerProductCoefficient u_z_c(u_c, e_z_c);

  // Optional ParaView output.
  ParaViewDataCollection dc("postseismic_deformation", &mesh);
  if (paraview) {
    dc.SetPrefixPath("ParaView");
    dc.SetLevelsOfDetail(order);
    dc.SetHighOrderOutput(true);
    visco.RegisterFields(dc);
  }

  // The replay files: u_z frames as single serial files (gathered to the
  // root in parallel) plus a GLVis script. A socket animation cannot be
  // replayed — GLVis renders the stream and keeps only the last frame —
  // but `glvis -run postseismic_deformation.glvs` plays the saved
  // frames, with the space bar as play/pause.
  const string frames_dir = "postseismic_frames";
  const string script_file = "postseismic_deformation.glvs";
  auto frame_name = [&](int step) {
    ostringstream os;
    os << frames_dir << "/u_z." << setw(6) << setfill('0') << step << ".gf";
    return os.str();
  };
  vector<real_t> frame_times;
  auto save_frame = [&](int step, real_t t_now) {
    ofstream ofs;
    if (Root()) {
      ofs.open(frame_name(step));
      ofs.precision(8);
    }
#ifdef MFEM_USE_MPI
    u_z.SaveAsOne(ofs);
#else
    u_z.Save(ofs);
#endif
    frame_times.push_back(t_now);
  };
  if (anim) {
    if (Root()) {
      std::filesystem::create_directories(frames_dir);
    }
    ofstream ofs;
    if (Root()) {
      ofs.open(frames_dir + "/mesh");
      ofs.precision(8);
    }
#ifdef MFEM_USE_MPI
    mesh.PrintAsOne(ofs);
#else
    mesh.Print(ofs);
#endif
  }

  // Initial state: relaxed internal variable, so the t = 0 solve is the
  // coseismic (unrelaxed elastic) response.
  real_t t = 0.0;
  real_t dt = t_final / n_steps;
  Vector m(visco.Height());
  m = 0.0;
  if (!visco.SolveElastic(m, t)) {
    if (Root()) {
      cerr << "Elastic solve failed at t = " << t << "\n";
    }
    return 2;
  }
  visco.SyncFields(m);
  if (sources.smoothed) {
    const double captured = examples::MomentCapture(
        problem.ExternalLoad(), fes, sources.couple, sources.moment_signed);
    if (Root()) {
      cout << "Smoothed sources: sigma = " << sources.sigma << " ("
           << fault.smooth << " element widths), moment captured: "
           << 100.0 * captured << "%\n";
    }
  }

  const auto values0 = profile.Sample(problem.Displacement());
  u_z.ProjectCoefficient(u_z_c);
  const auto [uz0_min, uz0_max] =
      examples::SurfacePeaks(u_z, fes_z, top_attr);
  if (Root()) {
    cout << "Coseismic peak surface uplift:     " << uz0_max
         << "\nCoseismic peak surface subsidence: " << uz0_min << "\n";
  }

  // The observation point: the coseismic uplift peak by default.
  int obs = 0;
  if (obs_x < 1e29) {
    obs = profile.Nearest(obs_x);
  } else {
    for (int i = 1; i < npts; i++) {
      if (std::abs(values0[(dim - 1) * npts + i]) >
          std::abs(values0[(dim - 1) * npts + obs])) {
        obs = i;
      }
    }
  }
  if (Root()) {
    cout << "Observation point: x = " << profile.X(obs) << "\n";
  }

  // One colour scale for the three windows, from the coseismic surface
  // peaks times the growth bound mu_U / mu_inf = tau_e / tau_s (the
  // source region saturates by design).
  const real_t span = (tau_e / tau_s) * 1.2 *
                      std::max({std::abs(uz0_min), std::abs(uz0_max),
                                real_t{1e-30}});
  examples::GLVisWindow win_initial("u_z coseismic (t = 0)",
                                    examples::DefaultKeys(dim));
  examples::GLVisWindow win_anim("u_z (postseismic relaxation)",
                                 examples::DefaultKeys(dim));
  examples::GLVisWindow win_final("u_z relaxed (t = tf)",
                                  examples::DefaultKeys(dim));

  examples::CsvTable history(csv_file, {"t", "u_x", "u_z"});
  history.Meta("title", "Post-seismic displacement at the surface point")
      .Meta("note",
            "x = " + std::to_string(profile.X(obs)).substr(0, 6) +
                "; stress glut held from t = 0")
      .Meta("xlabel", "time")
      .Meta("ylabel", "u");

  auto observe = [&](const std::vector<double>& values, real_t t_now,
                     int step) {
    const double ux = values[0 * npts + obs];
    const double uz = values[(dim - 1) * npts + obs];
    history.Row({t_now, ux, uz});
    if (Root()) {
      cout << "  " << setw(4) << step << "  " << fixed << setw(9) << t_now
           << "  " << setw(20) << scientific << uz << "\n";
    }
    if (visualization || anim) {
      u_z.ProjectCoefficient(u_z_c);
    }
    if (visualization) {
      win_anim.Send(mesh, u_z);
    }
    if (anim) {
      save_frame(step, t_now);
    }
    if (paraview) {
      dc.SetCycle(step);
      dc.SetTime(t_now);
      dc.Save();
    }
  };

  if (Root()) {
    cout << "\n  step          t     u_z(observation)\n"
         << "  ----  ---------  --------------------\n";
  }
  cout.precision(6);
  if (visualization) {
    win_initial.SetValueRange(-span, span);
    win_initial.Send(mesh, u_z);
    win_anim.SetValueRange(-span, span);
  }
  observe(values0, t, 0);

  // Step to t_final.
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
    observe(profile.Sample(problem.Displacement()), t, step);
  }
  const auto values_f = profile.Sample(problem.Displacement());
  if (visualization) {
    u_z.ProjectCoefficient(u_z_c);
    win_final.SetValueRange(-span, span);
    win_final.Send(mesh, u_z);
  }

  // The replay script: the first frame initialises the window; the block
  // sets the view and plays the frames without stopping, so one press of
  // the space bar runs the animation and another pauses it.
  if (anim && Root()) {
    ofstream scr(script_file);
    scr << "# Replay of the post-seismic run: glvis -run " << script_file
        << "\n# The space bar plays / pauses.\n"
        << "solution " << frames_dir << "/mesh " << frame_name(0) << "\n{\n"
        << "keys " << examples::DefaultKeys(dim) << "\n"
        << "autoscale off\nvaluerange " << -span << " " << span << "\n";
    for (size_t i = 0; i < frame_times.size(); i++) {
      scr << "plot_caption 't = " << fixed << setprecision(2)
          << frame_times[i] << "'\n";
      if (i > 0) {
        scr << "solution " << frames_dir << "/mesh " << frame_name(int(i))
            << "\n";
      }
    }
    scr << "}\n";
    cout << "Wrote " << script_file << " and " << frame_times.size()
         << " frames (replay: glvis -run " << script_file
         << "; space plays/pauses)\n";
  }

  // The fully relaxed limit, solved directly: an elastic solid with the
  // lid at its unrelaxed modulus and the mantle at mu_inf, loaded by the
  // same stress glut. The run's final state should approach it as
  // exp(-(t_final / tau_e) / (tau factor)).
  FunctionCoefficient mu_u_c(
      [=](const Vector& x) { return mu0 * modulus_factor(x); });
  PWCoefficient mu_relaxed;
  mu_relaxed.UpdateCoefficient(1, mu_inf_c);
  mu_relaxed.UpdateCoefficient(2, mu_u_c);
  IsotropicElasticRheology relaxed_rheology(dim, kappa_c, mu_relaxed);
  LinearQuasiStaticClampedProblem relaxed_problem(
      &fes, relaxed_rheology, ess_bdr, zero_traction, no_traction);
  add_glut(relaxed_problem.ExternalLoad());
  relaxed_problem.AssembleForce(0.0);
  if (!relaxed_problem.Solve()) {
    if (Root()) {
      cerr << "Relaxed-limit solve failed\n";
    }
    return 2;
  }
  const auto values_r = profile.Sample(relaxed_problem.Displacement());
  {
    GridFunction diff(relaxed_problem.Displacement());
    diff -= problem.Displacement();
    const double rel = GlobalMax(diff.Normlinf()) /
                       GlobalMax(relaxed_problem.Displacement().Normlinf());
    const double uz0 = values0[(dim - 1) * npts + obs];
    const double uzf = values_f[(dim - 1) * npts + obs];
    if (Root()) {
      cout << "\nu_z(observation): coseismic " << scientific << uz0
           << ", final " << uzf << " (growth " << fixed << setprecision(3)
           << uzf / uz0 << ")\n"
           << "Distance to the relaxed limit: " << scientific
           << setprecision(3) << rel << " (relative L-inf)\n";
    }
  }
  if (Root()) {
    cout << "\nSolves:                 " << problem.NumSolves()
         << "\nAssemblies:             " << problem.NumAssemblies()
         << "\nPreconditioner setups:  " << problem.NumPreconditionerSetups()
         << "\n";
  }

  // The profiles: coseismic, final, and the relaxed limit as the _exact
  // reference of the final (dashed in plot_csv.py).
  examples::CsvTable profiles(profiles_file,
                              {"x", "u_z_coseismic", "u_z_final",
                               "u_z_final_exact"});
  profiles.Meta("title", "Surface uplift: coseismic and relaxed")
      .Meta("note", "dashed: the directly solved relaxed limit")
      .Meta("xlabel", "x")
      .Meta("ylabel", "u_z");
  for (int i = 0; i < npts; i++) {
    profiles.Row({profile.X(i), values0[(dim - 1) * npts + i],
                  values_f[(dim - 1) * npts + i],
                  values_r[(dim - 1) * npts + i]});
  }
  history.Write();
  profiles.Write();
  return 0;
}
