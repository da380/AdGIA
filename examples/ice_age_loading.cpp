// ============================================================================
// ice_age_loading.cpp
//
// Viscoelastic sea-level response to a real deglaciation: the ICE-7G
// ice history (data downloaded and cached by pyslfp; nothing is bundled
// here) loaded onto a Maxwell planet with the monolithic water-load
// feedback (SetWaterLoad, frozen shorelines at the linearisation
// instant = the oldest date), stepped through the history — the WP6
// demo of doc/planning/sea_level_plan.md, chaining the surface exchange
// format end to end.
//
// The chain has three steps (the first and last are this program). The
// build's data/earth_coastlines.msh — refined along the real
// coastlines and the shallow shelves (one sizing field, no coastline
// isolation) — suits every run below through
// -m ../data/earth_coastlines.msh; an optional step 0 rebuilds it from
// a sharper or re-dated topography (meshes/README.md, "The coastline
// mesh"):
//
//   0.  <build>/postprocess/topography_grid --lmax 96
//       poetry -P meshes run python meshes/earth_coastlines.py \
//           --topography topography.npz --coast-size 0.03 --out <build>/data
//
//   1.  ./ice_age_loading                 # no -ice: writes the surface
//       node CSV (ice_age_nodes.csv) and prints step 2's commands.
//   2.  <build>/postprocess/ice_ng_to_surface ice_age_nodes.csv \
//           --dates 21,19,17,15,13,11,9,7,5,3,1,0 --length-scale 6.371e6
//       <build>/postprocess/ice_ng_to_surface ice_age_nodes.csv \
//           --field sea_level --dates 21 --length-scale 6.371e6 \
//           -o ice_age_nodes_state.csv
//       (pyslfp samples ICE-7G at the nodes; lengths in planet radii,
//       model time in ka from the oldest date. The CSVs are exact
//       nodal data: the node set depends on the mesh AND the order,
//       so changing -m or -o means re-running steps 1 and 2.)
//   3.  ./ice_age_loading -ice ice_age_nodes_ice.csv \
//                         -state ice_age_nodes_state.csv
//
// Step 3 reads both stacks (IceHistory), freezes the ocean function at
// the first field's flotation, steps the Maxwell body with the
// time-interpolated grounded-ice load sigma_d(t) = rho_i (1 - C0) dI(t)
// inside the water-load solve, prints a per-step table (eustatic
// equivalent, ocean-mean sea level, Phi_g, displacement norm), writes
// the history CSV for plot_csv.py and the final fingerprint CSV for the
// map chain, and opens GLVis windows.
//
// The body is the toy homogeneous planet of the sea-level examples
// (non-dimensional; the real geography rides a model rheology), so the
// output is qualitative: the realistic-Earth runs belong to the
// benchmarks and the server. Time is measured in the stacks' model-time
// unit (ka as exported above); -tau sets the Maxwell time in that unit.
//
// One source serves the serial and the parallel build (np N gives the
// same numbers).
//
// Options (defaults in brackets):
//   -m       body-in-buffer mesh [../data/coupled_poisson.msh]; the
//            ice chain needs the 3-D ball.
//   -o       finite element order [1].
//   -r       uniform refinements [0].
//   -ice     the ice-thickness time stack (step 2) [""].
//   -state   the initial sea-level stack; its FIRST column is the
//            linearisation state [""].
//   -rhow / -rhoi   water and ice density, body units [0.19, 0.17].
//   -shore   flotation smoothing width (rho_w x length units) [1e-4].
//   -tau     Maxwell time of the mantle, stack time units [0.5].
//   -n       time steps over the stack's span [40].
//   -rt      relative solver tolerance [1e-11: the stepped bordered
//            solves accumulate and amplify solver noise, and the early
//            history's small fields drown at looser settings].
//   -nodes-csv   step-1 output [ice_age_nodes.csv].
//   -csv     per-step history table [ice_age_loading.csv].
//   -vis / -no-vis   GLVis windows [on].
//
// Sample runs (with mpiexec -np N in front in a parallel build):
//    ./ice_age_loading
//    ./ice_age_loading -ice ice_age_nodes_ice.csv -state ice_age_nodes_state.csv
//    ./ice_age_loading -ice ... -state ... -tau 0.3 -n 80
// ============================================================================

#include <cmath>
#include <iostream>
#include <memory>
#include <numbers>

#include "mfem.hpp"
#include "AdGIA.hpp"
#include "visualisation.hpp"

using namespace mfem;
using namespace AdGIA;
using namespace examples;

namespace {

constexpr double kG = 0.05;
constexpr double kRho = 1.0;
constexpr double kKappa = 1.0;
constexpr double kMu = 0.5;
constexpr int kDtNDegree = 12;

double L2NormOf(const GridFunction& u) {
  Vector zero(u.FESpace()->GetVDim());
  zero = 0.0;
  VectorConstantCoefficient z(zero);
  return u.ComputeL2Error(z);
}

// The flotation of the linearisation instant: C0 from the stacks' first
// fields (time-independent), smoothed over `shore` in rho_w x length
// units, as everywhere in the sea-level machinery.
struct Flotation {
  Coefficient& sl0;
  Coefficient& i0;
  double rho_w, rho_i, shore;
  double Eval(ElementTransformation& T, const IntegrationPoint& ip) const {
    const double q = rho_w * sl0.Eval(T, ip) - rho_i * i0.Eval(T, ip);
    return 0.5 * (1.0 + std::tanh(q / shore));
  }
};

// sigma_d(t) = rho_i (1 - C0) dI(t): only the ice change follows the
// stepper's time (SetTime is forwarded to it alone — the linearisation
// state stays put by construction).
class IceAgeLoad : public Coefficient {
 public:
  IceAgeLoad(const Flotation& c0, Coefficient& dice)
      : c0_(c0), dice_(dice) {}
  void SetTime(real_t t) override {
    Coefficient::SetTime(t);
    dice_.SetTime(t);
  }
  real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    return c0_.rho_i * (1.0 - c0_.Eval(T, ip)) * dice_.Eval(T, ip);
  }

 private:
  Flotation c0_;
  Coefficient& dice_;
};

// The ocean weight w = rho_w C0 / g with the pointwise g = g0 |x|.
class IceAgeWeight : public Coefficient {
 public:
  IceAgeWeight(const Flotation& c0, double g0) : c0_(c0), g0_(g0) {}
  real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    Vector x;
    T.Transform(ip, x);
    return c0_.rho_w * c0_.Eval(T, ip) / (g0_ * x.Norml2());
  }

 private:
  Flotation c0_;
  double g0_;
};

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif

  const char* mesh_file = "../data/coupled_poisson.msh";
  int order = 1;
  int ref_levels = 0;
  const char* ice_csv = "";
  const char* state_csv = "";
  double rho_w = 0.19;
  double rho_i = 0.17;
  double shore = 1e-4;
  double tau_m = 0.5;
  int n_steps = 40;
  double rel_tol = 1e-11;
  const char* nodes_csv = "ice_age_nodes.csv";
  const char* csv_file = "ice_age_loading.csv";
  bool visualization = true;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
                 "Body-in-buffer mesh (the ice chain needs the 3-D ball).");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&ref_levels, "-r", "--refine",
                 "Uniform mesh refinements before partitioning.");
  args.AddOption(&ice_csv, "-ice", "--ice-stack",
                 "Ice-thickness time stack (ice_ng_to_surface); empty: "
                 "write the node CSV and stop.");
  args.AddOption(&state_csv, "-state", "--state-stack",
                 "Initial sea-level stack; its first column is the "
                 "linearisation state.");
  args.AddOption(&rho_w, "-rhow", "--water-density",
                 "Water density, body units.");
  args.AddOption(&rho_i, "-rhoi", "--ice-density",
                 "Ice density, body units.");
  args.AddOption(&shore, "-shore", "--shoreline-width",
                 "Flotation smoothing width (rho_w x length units).");
  args.AddOption(&tau_m, "-tau", "--maxwell-time",
                 "Maxwell time of the mantle, stack time units.");
  args.AddOption(&n_steps, "-n", "--steps",
                 "Time steps over the stack's span.");
  args.AddOption(&rel_tol, "-rt", "--solver-tolerance",
                 "Relative solver tolerance.");
  args.AddOption(&nodes_csv, "-nodes-csv", "--nodes-csv",
                 "Step-1 output: the surface node CSV.");
  args.AddOption(&csv_file, "-csv", "--history-csv",
                 "Per-step history table for plot_csv.py; \"\": none.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization", "GLVis windows.");
  args.Parse();
  if (!args.Good()) {
    if (IsRoot()) {
      args.PrintUsage(std::cout);
    }
    return 1;
  }

  Mesh smesh(mesh_file, 1, 1);
  for (int l = 0; l < ref_levels; l++) {
    smesh.UniformRefinement();
  }
  const int dim = smesh.Dimension();
#ifdef MFEM_USE_MPI
  ParMesh parent(MPI_COMM_WORLD, smesh);
  smesh.Clear();
#else
  Mesh& parent = smesh;
#endif
  Array<int> body_marker(parent.attributes.Max());
  body_marker = 0;
  body_marker[0] = 1;
#ifdef MFEM_USE_MPI
  ParSubMesh body(ParSubMesh::CreateFromDomain(parent, body_marker));
#else
  SubMesh body(SubMesh::CreateFromDomain(parent, body_marker));
#endif
  H1_FECollection fec(order, dim);
#ifdef MFEM_USE_MPI
  ParFiniteElementSpace fes_u(&body, &fec, dim), fes_phi(&parent, &fec);
#else
  FiniteElementSpace fes_u(&body, &fec, dim), fes_phi(&parent, &fec);
#endif
  Array<int> surface(body.bdr_attributes.Max());
  surface = 0;
  surface[body.bdr_attributes.Max() - 1] = 1;

  SeaLevelOperator sea(fes_u, surface);
  sea.SetDensities(rho_w, rho_i);

  // Step 1: no ice stack — export the node set and say what comes next.
  if (std::string(ice_csv).empty()) {
    GridFunction zero(&sea.SurfaceSpace());
    zero = 0.0;
    sea.WriteSurfaceField(zero, nodes_csv);
    if (IsRoot()) {
      std::cout
          << "wrote " << nodes_csv << " (" << dim << "-D surface nodes)\n"
          << "sample the ICE-7G history at these nodes (pyslfp downloads\n"
          << "and caches the data on first use):\n"
          << "  <build>/postprocess/ice_ng_to_surface " << nodes_csv
          << " --dates 21,19,17,15,13,11,9,7,5,3,1,0 "
             "--length-scale 6.371e6\n"
          << "  <build>/postprocess/ice_ng_to_surface " << nodes_csv
          << " --field sea_level --dates 21 --length-scale 6.371e6 \\\n"
          << "      -o ice_age_nodes_state.csv\n"
          << "then run step 3:\n"
          << "  ./ice_age_loading -ice ice_age_nodes_ice.csv "
             "-state ice_age_nodes_state.csv\n";
    }
    return 0;
  }
  if (dim != 3) {
    if (IsRoot()) {
      std::cout << "the ice chain needs the 3-D ball mesh\n";
    }
    return 1;
  }
  MFEM_VERIFY(!std::string(state_csv).empty(),
              "-ice needs -state (the linearisation sea level)");

  // The stacks and the frozen-C0 load law.
  IceHistory ice(sea, ice_csv);
  IceHistory state(sea, state_csv);
  const double g0 = 4.0 * std::numbers::pi * kG * kRho / 3.0;
  Flotation c0{state.FieldCoefficient(0), ice.FieldCoefficient(0), rho_w,
               rho_i, shore};
  IceAgeWeight w(c0, g0);
  IceAgeLoad sigma_d(c0, ice.Change());
  VectorFunctionCoefficient grad_phi0(dim, [g0](const Vector& x, Vector& v) {
    v = x;
    v *= g0;
  });

  // The Maxwell body with the monolithic water load.
  ConstantCoefficient kappa(kKappa), mu(kMu), rho(kRho), tau(tau_m);
  auto rheology = IsotropicMaxwellRheology::Maxwell(dim, kappa, mu, tau);
  LinearQuasiStaticMixedSelfGravitatingProblem prob(&fes_u, &fes_phi,
                                                    rheology, rho, kG,
                                                    kDtNDegree);
  prob.SetWaterLoad(w, sigma_d, surface);
  prob.SetRelTol(rel_tol);

  // The surface diagnostics use the same frozen state.
  GridFunctionCoefficient sl0_surf(
      const_cast<GridFunction*>(&state.Field(0)));
  GridFunctionCoefficient i0_surf(const_cast<GridFunction*>(&ice.Field(0)));
  sea.SetInitialState(sl0_surf, i0_surf);
  const double ocean_area = sea.OceanArea();

  ViscoelasticOperator visco(prob);
  ExponentialTrapezoidSolver ode;
  ode.Init(visco);
  Vector m(visco.Height());
  m = 0.0;
  double t = ice.Time(0);
  const double t_end = ice.Time(ice.NumTimes() - 1);
  double dt = (t_end - t) / n_steps;

  if (IsRoot()) {
    std::cout << "ICE-7G loading on the Maxwell planet: "
              << ice.NumTimes() << " dates, t = " << t << " .. " << t_end
              << " (tau = " << tau_m << "), " << n_steps << " steps\n"
              << "ocean area " << ocean_area << " (fraction "
              << ocean_area / (4.0 * std::numbers::pi) << ")\n\n";
    std::cout << "t         eustatic      ocean mean    Phi_g         "
                 "||u||\n";
  }
  CsvTable table(csv_file,
                 {"t", "eustatic", "ocean_mean", "phi_g", "u_L2"});
  table.Meta("title", "ICE-7G deglaciation on a Maxwell planet")
      .Meta("note", "frozen shorelines at the oldest date; toy rheology")
      .Meta("xlabel", "model time (stack units)")
      .Meta("y", "eustatic,ocean_mean|phi_g|u_L2")
      .Meta("ylabel", "sea level (planet radii)|Phi_g|displacement L2");

  auto report = [&]() {
    if (!visco.SolveElastic(m, t)) {
      if (IsRoot()) {
        std::cout << "solve failed at t = " << t << "\n";
      }
      return false;
    }
    sea.SeaLevelChangeFrom(prob.Displacement(), grad_phi0, prob.Potential(),
                           prob.UniformPotentialTerm());
    sigma_d.SetTime(t);
    const double melted = -sea.SurfaceIntegral(sigma_d);
    const double eustatic = melted / (rho_w * ocean_area);
    GridFunctionCoefficient slc(&sea.SeaLevelChangeField());
    const double ocean_mean = sea.OceanIntegral(slc) / ocean_area;
    const double un = L2NormOf(prob.Displacement());
    if (IsRoot()) {
      std::cout.precision(6);
      std::cout << std::left << t << "  " << eustatic << "  " << ocean_mean
                << "  " << prob.UniformPotentialTerm() << "  " << un
                << "\n";
    }
    table.Row({t, eustatic, ocean_mean, prob.UniformPotentialTerm(), un});
    return true;
  };

  if (!report()) {
    return 1;
  }
  for (int step = 0; step < n_steps; step++) {
    ode.Step(m, t, dt);
    if (!report()) {
      return 1;
    }
  }

  // The endpoint fingerprint as data, for the map chain.
  sea.WriteSurfaceField(sea.SeaLevelChangeField(), "ice_age_fingerprint.csv");
  if (IsRoot()) {
    std::cout << "\nwrote " << (table.Enabled() ? csv_file : "(no CSV)")
              << " (plot_csv.py) and ice_age_fingerprint.csv — map it "
                 "with\n  <build>/postprocess/surface_to_netcdf "
                 "ice_age_fingerprint.csv --plot\n";
  }

  if (visualization) {
    GLVisWindow wu("displacement", DefaultKeys(dim));
    wu.Send(body, const_cast<GridFunction&>(prob.Displacement()));
    GLVisWindow ws("sea-level change", DefaultKeys(2));
    ws.Send(sea.SurfaceMesh(), sea.SeaLevelChangeField());
    GridFunction c(&sea.SurfaceSpace());
    sea.OceanFunction(c);
    GLVisWindow wc("ocean function", DefaultKeys(2));
    wc.Send(sea.SurfaceMesh(), c);
    GridFunction load(&sea.SurfaceSpace());
    load.ProjectCoefficient(sigma_d);
    GLVisWindow wl("ice load (endpoint)", DefaultKeys(2));
    wl.Send(sea.SurfaceMesh(), load);
  }
  return 0;
}
