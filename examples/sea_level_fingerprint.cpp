// ============================================================================
// sea_level_fingerprint.cpp
//
// An elastic sea-level fingerprint: the gravitationally self-consistent
// ocean response to melting an ice cap, solved MONOLITHICALLY — the
// sea-level equation folded into the elastic operator
// (LinearQuasiStaticMixedSelfGravitatingProblem::SetWaterLoad;
// doc/planning/sea_level_plan.md, "The WP3 derivation") rather than by
// the fixed-point iteration of pseudo-spectral codes. The water-load
// feedback is a symmetric boundary modification of the system plus a
// rank-one border carrying the uniform term and mass conservation, so
// self-consistency costs one extra inner solve, with no outer loop.
// Shorelines are frozen at the initial state, which is first-order
// exact (shoreline migration is strictly second order).
//
// Two geographies, one solve:
//
// SIMPLE (the default): the sea-level benchmark family's state
// (benchmarks/sea_level), at example-sized amplitudes and without the
// pyslfp comparison: smooth analytical topography — a super-Gaussian
// continent at the pole (+z), ocean elsewhere — an
// ice cap on the continent, and a melt that unloads the +x hemisphere
// of the cap through the smooth factor (1 + tanh(x/melt_width))/2
// (regular at the pole), so the fingerprint carries non-zonal structure.
// The default mesh, data/sea_level_fingerprint.msh
// (meshes/fingerprint_coastline.py), is refined in a ring about this
// state's shoreline circle; change the state's options there and here
// together, or run on any ball-in-buffer mesh through -m.
//
// REAL (-earth): the present-day ICE-7G topography and ice sheets,
// sampled at the surface nodes by the ingest chain of
// ice_age_loading.cpp at one date, and a melt of one real ice sheet —
// Greenland or West Antarctica (-melt-region), confined by a smooth
// geographic window — so the classic fingerprint maps of either sheet
// come out of the same monolithic solve. Two steps, like the ice-age
// chain (the first run prints the exact commands):
//
//   ./sea_level_fingerprint -earth -m ../data/earth_coastlines.msh
//   <build>/postprocess/ice_ng_to_surface fingerprint_nodes.csv \
//       --field sea_level --dates 0 --length-scale 6.371e6 \
//       -o fingerprint_state.csv
//   <build>/postprocess/ice_ng_to_surface fingerprint_nodes.csv \
//       --field ice --dates 0 --length-scale 6.371e6 \
//       -o fingerprint_ice.csv
//   ./sea_level_fingerprint -earth -m ../data/earth_coastlines.msh \
//       -state fingerprint_state.csv -ice fingerprint_ice.csv
//
// (data/earth_coastlines.msh is the coastline-refined ball the build
// makes; meshes/README.md, "The coastline mesh". The model stays the
// toy homogeneous planet, so the maps are qualitative — real geography
// on a model body. The CSVs are exact nodal data: the node set depends
// on the mesh AND the order, so changing -m or -o means re-running
// step 1 and the sampling.)
//
// In both geographies the ocean function C0 enters the operator
// pointwise through the ocean weight w = rho_w C0 / g; nothing is
// projected. Every piece of the geometry and load is an option.
//
// Output: the melt load and the sea-level change SL1 on the surface (the fingerprint: a
// far-field rise of roughly the eustatic value, depressed — possibly
// below zero — near the melted cap where gravitational attraction is
// lost and the ground rebounds), the uniform term Phi_g, the eustatic
// equivalent, the ocean area and the mass-conservation certificate. The
// fingerprint is exported with SeaLevelOperator::WriteSurfaceField;
// grid and map it with
//    <build>/postprocess/surface_to_netcdf sea_level_fingerprint.csv --plot
// (cartopy/pyshtools-ready NetCDF; see postprocess/README.md). GLVis
// shows the body displacement and the fingerprint on the surface shell.
//
// 3-D only: the sea-level equation is not wired for 2-D (2-D elastic
// solves with gravity remain useful; a 2-D ocean is not).
//
// One source serves the serial and the parallel build; the only genuine
// differences are the partitioning and typed field copies.
//
// Options (defaults in brackets):
//   -m       mesh file [../data/sea_level_fingerprint.msh, the 3-D ball
//            refined about the simple state's shoreline ring].
//   -o       finite element order [2].
//   -r       uniform mesh refinements [0].
//   -Omega   equilibrium rotation rate about e3 [0: rotational feedback
//            off]; with it the angular-velocity border joins the solve
//            and psi(omega) enters the fingerprint.
//   -C1 -C2 -C3   principal moments [1.2, 1.3, 2.0].
//   -rhow    water density [0.05] (non-dimensional, like the model).
//   -rhoi    ice density [0.045].
//   -ocean-depth   initial ocean depth [1.0].
//   -cont-amp      continent height [3.0].
//   -cont-width    continent angular width (rad) [0.7].
//   -cap-amp       ice-cap thickness [1.0].
//   -cap-width     ice-cap angular width (rad) [0.35].
//   -melt    melted fraction of the cap [0.5].
//   -melt-width    smoothing width of the hemispheric unloading [0.15].
//   -shore   shoreline smoothing width of the ocean fraction [0.02;
//            1e-4 with -earth, whose flotation field is in rho_w x
//            planet-radius units]; a sharp indicator (small width)
//            costs quadrature accuracy in the shoreline-crossing
//            boundary elements.
//   -mig / -no-mig   shoreline migration (Picard on the ocean function
//            via ShorelineMigration; off = frozen shorelines,
//            first-order exact) [off].
//   -earth / -no-earth   real present-day geography from ICE-7G [off];
//            without -state and -ice it writes the node CSV and prints
//            the two sampling commands above.
//   -state -ice    the sampled sea-level and ice-thickness CSVs.
//   -melt-region   greenland | west-antarctica [greenland]; -melt is
//            the melted fraction of the sheet, -melt-width the window's
//            edge smoothing in radians (sharper than the simple
//            geometry's default suits a single sheet, e.g. 0.05).
//   -nodes-csv     step-1 output [fingerprint_nodes.csv].
//   -vis / -no-vis   GLVis windows on or off [on].
//
// The fixed-point route of pseudo-spectral codes is not repeated here;
// the field-level identity monolithic == fixed point is the rung-0 gate
// of tests/TestSeaLevelCoupling.cpp.
//
// Sample runs (with mpiexec -np N in front in a parallel build):
//    ./sea_level_fingerprint
//    ./sea_level_fingerprint -melt 1.0 -melt-width 0.4
//    ./sea_level_fingerprint -r 1 -o 2
//    ./sea_level_fingerprint -Omega 0.1
//    ./sea_level_fingerprint -earth -m ../data/earth_coastlines.msh \
//        -state fingerprint_state.csv -ice fingerprint_ice.csv \
//        -melt-region west-antarctica -melt-width 0.05
// ============================================================================

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <numbers>
#include <string>

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

// The state of the sea-level benchmark family (benchmarks/sea_level),
// at example-sized amplitudes and with every knob an option: a
// super-Gaussian continent at the pole (+z), an ice
// cap on it, a smoothed ocean fraction, and a melt that unloads the +x
// hemisphere of the cap smoothly.
double g_rho_w = 0.05;
double g_rho_i = 0.045;
double g_ocean_depth = 1.0;
double g_cont_amp = 3.0;
double g_cont_width = 0.7;
double g_cap_amp = 1.0;
double g_cap_width = 0.35;
double g_melt = 0.5;
double g_melt_width = 0.15;
double g_shore = 0.02;
int g_dim = 2;

double G0() {
  return g_dim == 2 ? 2.0 * std::numbers::pi * kG * kRho
                    : 4.0 * std::numbers::pi * kG * kRho / 3.0;
}

// Colatitude from the pole (+z).
double Colatitude(const Vector& x) {
  const double r = x.Norml2();
  const double c = x[x.Size() - 1] / r;
  return std::acos(std::min(1.0, std::max(-1.0, c)));
}

double InitialSeaLevel(const Vector& x) {
  const double t = Colatitude(x) / g_cont_width;
  return g_ocean_depth - g_cont_amp * std::exp(-0.5 * std::pow(t, 6));
}

double IceThickness(const Vector& x) {
  const double t = Colatitude(x) / g_cap_width;
  return g_cap_amp * std::exp(-0.5 * t * t);
}

// The ocean fraction: the flooded indicator of eq. (29) smoothed over
// a thin band (the pyslfp trick of smooth analytical states — a sharp
// shoreline through a boundary element costs quadrature accuracy; the
// operator accepts either).
double OceanFraction(const Vector& x) {
  const double q = g_rho_w * InitialSeaLevel(x) - g_rho_i * IceThickness(x);
  return 0.5 * (1.0 + std::tanh(q / g_shore));
}

// The ocean weight w = rho_w C0 / g at the (unit-radius) surface.
double OceanWeight(const Vector& x) {
  return g_rho_w * OceanFraction(x) / G0();
}

// The ice-thickness change -melt H(x) I0 with the smooth hemispheric
// factor H = (1 + tanh(x0/melt_width))/2, regular at the pole
// (Cartesian coordinate, not an angle).
double IceChange(const Vector& x) {
  const double hemi = 0.5 * (1.0 + std::tanh(x[0] / g_melt_width));
  return -g_melt * hemi * IceThickness(x);
}

// The melt load sigma_data = rho_i (1 - C0) dI (the frozen-shoreline
// reduction of the load law; with -migrate the orchestrator rebuilds
// the C-dependent version itself).
double MeltLoad(const Vector& x) {
  return (1.0 - OceanFraction(x)) * g_rho_i * IceChange(x);
}

// --- The -earth leg: the same four roles (ocean fraction, weight, ice
// change, melt load) from the sampled ICE-7G state instead of the
// analytic one. The stacks' coefficients evaluate on the body side
// (IceHistory), so these compose exactly as the functions above.

bool g_greenland = true;

// The smooth geographic window of the melted sheet, 1 inside; edges
// tanh-smoothed over g_melt_width radians. Loose boxes are enough:
// within each, ICE-7G's present-day thickness is the sheet itself.
double RegionMask(const Vector& x) {
  constexpr double deg = std::numbers::pi / 180.0;
  const double r = x.Norml2();
  const double lat = std::asin(std::min(1.0, std::max(-1.0, x[2] / r)));
  double lon = std::atan2(x[1], x[0]);
  if (lon < 0.0) {
    lon += 2.0 * std::numbers::pi;
  }
  auto edge = [](double d) {
    return 0.5 * (1.0 + std::tanh(d / g_melt_width));
  };
  if (g_greenland) {
    return edge(lat - 59.0 * deg) * edge(84.0 * deg - lat) *
           edge(lon - 285.0 * deg) * edge(350.0 * deg - lon);
  }
  // West Antarctica: the Pacific sector, stopped short of the pole so
  // the East Antarctic plateau stays out of the window.
  return edge(lat + 85.0 * deg) * edge(-72.0 * deg - lat) *
         edge(lon - 190.0 * deg) * edge(300.0 * deg - lon);
}

// C0 of the sampled state: the flotation indicator, smoothed over
// g_shore in rho_w x length units (as everywhere in the machinery).
class FieldFlotation : public Coefficient {
 public:
  FieldFlotation(Coefficient& sl0, Coefficient& i0) : sl0_(sl0), i0_(i0) {}
  real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    const double q =
        g_rho_w * sl0_.Eval(T, ip) - g_rho_i * i0_.Eval(T, ip);
    return 0.5 * (1.0 + std::tanh(q / g_shore));
  }

 private:
  Coefficient &sl0_, &i0_;
};

// dI = -melt x window x I0: the chosen sheet loses the melted fraction.
class EarthIceChange : public Coefficient {
 public:
  explicit EarthIceChange(Coefficient& i0) : i0_(i0) {}
  real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    Vector x;
    T.Transform(ip, x);
    return -g_melt * RegionMask(x) * i0_.Eval(T, ip);
  }

 private:
  Coefficient& i0_;
};

class EarthMeltLoad : public Coefficient {
 public:
  EarthMeltLoad(FieldFlotation& c0, Coefficient& dice)
      : c0_(c0), dice_(dice) {}
  real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    return g_rho_i * (1.0 - c0_.Eval(T, ip)) * dice_.Eval(T, ip);
  }

 private:
  FieldFlotation& c0_;
  Coefficient& dice_;
};

class EarthOceanWeight : public Coefficient {
 public:
  explicit EarthOceanWeight(FieldFlotation& c0) : c0_(c0) {}
  real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    return g_rho_w * c0_.Eval(T, ip) / G0();
  }

 private:
  FieldFlotation& c0_;
};

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif

  const char* mesh_file = "../data/sea_level_fingerprint.msh";
  int order = 2;
  int ref_levels = 0;
  double Omega = 0.0;
  double C1 = 1.2, C2 = 1.3, C3 = 2.0;
  bool migrate = false;
  bool earth = false;
  const char* state_csv = "";
  const char* ice_csv = "";
  const char* melt_region = "greenland";
  const char* nodes_csv = "fingerprint_nodes.csv";
  bool visualization = true;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
                 "Body-in-buffer mesh (the 3-D ball, refined about the "
                 "simple state's shoreline by default).");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&ref_levels, "-r", "--refine",
                 "Uniform mesh refinements before partitioning.");
  args.AddOption(&Omega, "-Omega", "--rotation-rate",
                 "Equilibrium rotation rate about e3 (0: no rotational "
                 "feedback).");
  args.AddOption(&C1, "-C1", "--moment-1", "Principal moment C1 (3-D).");
  args.AddOption(&C2, "-C2", "--moment-2", "Principal moment C2 (3-D).");
  args.AddOption(&C3, "-C3", "--moment-3", "Principal moment C3.");
  args.AddOption(&g_rho_w, "-rhow", "--water-density", "Water density.");
  args.AddOption(&g_rho_i, "-rhoi", "--ice-density", "Ice density.");
  args.AddOption(&g_ocean_depth, "-ocean-depth", "--ocean-depth",
                 "Initial ocean depth.");
  args.AddOption(&g_cont_amp, "-cont-amp", "--continent-amplitude",
                 "Continent height (super-Gaussian at the pole).");
  args.AddOption(&g_cont_width, "-cont-width", "--continent-width",
                 "Continent angular width (rad).");
  args.AddOption(&g_cap_amp, "-cap-amp", "--cap-amplitude",
                 "Ice-cap thickness (Gaussian at the pole).");
  args.AddOption(&g_cap_width, "-cap-width", "--cap-width",
                 "Ice-cap angular width (rad).");
  args.AddOption(&g_melt, "-melt", "--melt-fraction",
                 "Melted fraction of the ice cap.");
  args.AddOption(&g_melt_width, "-melt-width", "--melt-width",
                 "Smoothing width of the hemispheric unloading factor "
                 "(1 + tanh(x/width))/2.");
  args.AddOption(&migrate, "-mig", "--migrate", "-no-mig", "--no-migrate",
                 "Shoreline migration (Picard on the ocean function; "
                 "off = frozen shorelines).");
  args.AddOption(&g_shore, "-shore", "--shoreline-width",
                 "Shoreline smoothing width of the ocean fraction.");
  args.AddOption(&earth, "-earth", "--earth", "-no-earth", "--no-earth",
                 "Real present-day geography (ICE-7G at the nodes); "
                 "without -state and -ice, write the node CSV and print "
                 "the sampling commands.");
  args.AddOption(&state_csv, "-state", "--state-csv",
                 "Sampled sea level at the nodes (ice_ng_to_surface "
                 "--field sea_level --dates 0).");
  args.AddOption(&ice_csv, "-ice", "--ice-csv",
                 "Sampled ice thickness at the nodes (--field ice).");
  args.AddOption(&melt_region, "-melt-region", "--melt-region",
                 "greenland or west-antarctica (-earth only; the simple "
                 "geometry melts the cap's +x hemisphere).");
  args.AddOption(&nodes_csv, "-nodes-csv", "--nodes-csv",
                 "Step-1 output: the surface node CSV (-earth only).");
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
  g_dim = smesh.Dimension();
  const int dim = g_dim;
  if (dim != 3) {
    if (IsRoot()) {
      std::cout << "the sea-level equation is 3-D only: use the ball mesh "
                   "(../data/coupled_poisson.msh)\n";
    }
    return 1;
  }
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

  // The surface layer up front: -earth exports its node set here, and
  // the diagnostics at the end use it in both geographies.
  SeaLevelOperator sea(fes_u, surface);
  sea.SetDensities(g_rho_w, g_rho_i);

  g_greenland = std::string(melt_region) != "west-antarctica";
  if (earth && g_shore == 0.02) {
    // The analytic default is far wider than the sampled state's
    // rho_w x length units (planet radii): it would smear C0 to 1/2
    // everywhere. ice_age_loading's default; -shore still overrides.
    g_shore = 1e-4;
  }
  const bool have_stacks =
      !std::string(state_csv).empty() && !std::string(ice_csv).empty();
  if (earth && !have_stacks) {
    // Step 1 of the real-geography chain: the node set, then the exact
    // sampling commands (pyslfp downloads and caches the data).
    GridFunction zero(&sea.SurfaceSpace());
    zero = 0.0;
    sea.WriteSurfaceField(zero, nodes_csv);
    if (IsRoot()) {
      std::cout
          << "wrote " << nodes_csv << " (surface nodes)\n"
          << "sample the present-day ICE-7G state at these nodes:\n"
          << "  <build>/postprocess/ice_ng_to_surface " << nodes_csv
          << " --field sea_level --dates 0 \\\n"
          << "      --length-scale 6.371e6 -o fingerprint_state.csv\n"
          << "  <build>/postprocess/ice_ng_to_surface " << nodes_csv
          << " --field ice --dates 0 \\\n"
          << "      --length-scale 6.371e6 -o fingerprint_ice.csv\n"
          << "then melt a sheet:\n"
          << "  ./sea_level_fingerprint -earth -m " << mesh_file
          << " \\\n      -state fingerprint_state.csv -ice "
             "fingerprint_ice.csv -melt-region "
          << melt_region << "\n";
    }
    return 0;
  }

  ConstantCoefficient kappa(kKappa), mu(kMu), rho(kRho);
  IsotropicElasticRheology rheology(dim, kappa, mu);
  const double g0 = G0();
  VectorFunctionCoefficient grad_phi0(dim, [g0](const Vector& x, Vector& v) {
    v = x;
    v *= g0;
  });

  // The monolithic solve.
  LinearQuasiStaticMixedSelfGravitatingProblem prob(&fes_u, &fes_phi,
                                                    rheology, rho, kG,
                                                    kDtNDegree);

  // The two geographies fill the same roles; pointers pick one set.
  // The analytic functions evaluate anywhere, so one object serves the
  // solve (body side) and the diagnostics (surface side); the sampled
  // state follows ice_age_loading.cpp's split — the stacks' body-side
  // FieldCoefficients feed the water load and the migration, their
  // surface GridFunctions the initial state and the diagnostics.
  FunctionCoefficient w(OceanWeight), sigma_data(MeltLoad);
  FunctionCoefficient sl0(InitialSeaLevel), ice(IceThickness);
  FunctionCoefficient dice(IceChange);
  FunctionCoefficient ocean_ind(OceanFraction);
  Coefficient* w_used = &w;
  Coefficient* sigma_used = &sigma_data;      // the water-load pair
  Coefficient* mig_sl0 = &sl0;
  Coefficient* mig_ice = &ice;
  Coefficient* mig_dice = &dice;              // the migration's state
  Coefficient* init_sl0 = &sl0;
  Coefficient* init_ice = &ice;               // SetInitialState
  Coefficient* ocean_diag = &ocean_ind;
  Coefficient* melt_diag = &sigma_data;       // surface diagnostics
  std::unique_ptr<IceHistory> state_stack, ice_stack;
  std::unique_ptr<GridFunctionCoefficient> sl0_surf, i0_surf;
  std::unique_ptr<FieldFlotation> c0_body, c0_surf;
  std::unique_ptr<EarthIceChange> dice_body, dice_surf;
  std::unique_ptr<EarthMeltLoad> sigma_body, sigma_surf;
  std::unique_ptr<EarthOceanWeight> w_body;
  if (earth) {
    state_stack = std::make_unique<IceHistory>(sea, state_csv);
    ice_stack = std::make_unique<IceHistory>(sea, ice_csv);
    c0_body = std::make_unique<FieldFlotation>(
        state_stack->FieldCoefficient(0), ice_stack->FieldCoefficient(0));
    dice_body = std::make_unique<EarthIceChange>(
        ice_stack->FieldCoefficient(0));
    sigma_body = std::make_unique<EarthMeltLoad>(*c0_body, *dice_body);
    w_body = std::make_unique<EarthOceanWeight>(*c0_body);
    sl0_surf = std::make_unique<GridFunctionCoefficient>(
        const_cast<GridFunction*>(&state_stack->Field(0)));
    i0_surf = std::make_unique<GridFunctionCoefficient>(
        const_cast<GridFunction*>(&ice_stack->Field(0)));
    c0_surf = std::make_unique<FieldFlotation>(*sl0_surf, *i0_surf);
    dice_surf = std::make_unique<EarthIceChange>(*i0_surf);
    sigma_surf = std::make_unique<EarthMeltLoad>(*c0_surf, *dice_surf);
    w_used = w_body.get();
    sigma_used = sigma_body.get();
    mig_sl0 = &state_stack->FieldCoefficient(0);
    mig_ice = &ice_stack->FieldCoefficient(0);
    mig_dice = dice_body.get();
    init_sl0 = sl0_surf.get();
    init_ice = i0_surf.get();
    ocean_diag = c0_surf.get();
    melt_diag = sigma_surf.get();
  }

  // With -mig the orchestrator owns the water load (its constructor
  // wires SetWaterLoad with C-dependent coefficients); otherwise the
  // frozen-C coefficients go in directly.
  std::unique_ptr<ShorelineMigration> mig;
  if (migrate) {
    ShorelineMigration::Options opt;
    opt.shore = g_shore;
    mig = std::make_unique<ShorelineMigration>(prob, grad_phi0, *mig_sl0,
                                               *mig_ice, *mig_dice, g_rho_w,
                                               g_rho_i, surface, opt);
  } else {
    prob.SetWaterLoad(*w_used, *sigma_used, surface);
  }
  if (Omega != 0.0) {
    // Rotational feedback inside the same bordered solve; the moments
    // are data (a spherical model's own would be degenerate).
    Vector moments(dim == 2 ? 1 : 3);
    if (dim == 2) {
      moments[0] = C3;
    } else {
      moments[0] = C1;
      moments[1] = C2;
      moments[2] = C3;
    }
    prob.SetRotation(Omega, moments);
  }
  prob.SetRelTol(1e-11);
  if (migrate) {
    if (!mig->Solve(0.0)) {
      if (IsRoot()) {
        std::cout << "solve failed\n";
      }
      return 1;
    }
    if (IsRoot()) {
      std::cout << "shoreline migration: " << mig->Iterations()
                << " extra passes, final relative SL1 increment "
                << mig->LastShorelineChange() << ", "
                << mig->TotalOuterIterations()
                << " outer iterations in all\n";
    }
  } else {
    prob.AssembleForce(0.0);
    if (!prob.Solve()) {
      if (IsRoot()) {
        std::cout << "solve failed\n";
      }
      return 1;
    }
  }

  // The surface layer: the initial state, the fingerprint, diagnostics.
  sea.SetInitialState(*init_sl0, *init_ice);
  CentrifugalPotential psi(dim, Omega);
  if (Omega != 0.0) {
    psi.SetAmplitudes(prob.AngularVelocity());
  }
  sea.SeaLevelChangeFrom(prob.Displacement(), grad_phi0, prob.Potential(),
                         prob.UniformPotentialTerm(),
                         Omega != 0.0 ? &psi : nullptr);

  // Diagnostics over the same ocean function the solve used (the
  // field-projected indicator differs near the shoreline at
  // interpolation grade).
  ConstantCoefficient one(1.0);
  ProductCoefficient ocean_area_ind(*ocean_diag, one);
  const double area = sea.SurfaceIntegral(ocean_area_ind);
  const double melted_mass = -sea.SurfaceIntegral(*melt_diag);
  const double eustatic = melted_mass / (g_rho_w * area);
  GridFunctionCoefficient slc(&sea.SeaLevelChangeField());
  ProductCoefficient ocean_sl(*ocean_diag, slc);
  const double ocean_mean = sea.SurfaceIntegral(ocean_sl) / area;
  if (IsRoot()) {
    std::cout << "\nsea-level fingerprint ("
              << (earth ? std::string("ICE-7G, melting ") + melt_region
                        : std::string("polar cap"))
              << "; rho_w " << g_rho_w << ", rho_i " << g_rho_i
              << ", melt " << g_melt << ")\n";
    std::cout << "  ocean area                 " << area << "\n";
    std::cout << "  melted ice mass            " << melted_mass << "\n";
    std::cout << "  eustatic equivalent        " << eustatic << "\n";
    std::cout << "  ocean-mean SL change       " << ocean_mean
              << "   (= eustatic when mass is conserved)\n";
    std::cout << "  uniform term Phi_g         "
              << prob.UniformPotentialTerm() << "\n";
    if (Omega != 0.0) {
      std::cout << "  angular-velocity change   ";
      for (int k = 0; k < prob.AngularVelocity().Size(); k++) {
        std::cout << " " << prob.AngularVelocity()[k];
      }
      std::cout << (dim == 2 ? "   (spin)\n" : "   (wander_1 wander_2 spin)\n");
    }
  }

  // The fingerprint as data: nodal CSV for the postprocess tool.
  sea.WriteSurfaceField(sea.SeaLevelChangeField(),
                        "sea_level_fingerprint.csv");
  if (IsRoot()) {
    if (dim == 3) {
      std::cout << "  wrote sea_level_fingerprint.csv — map it with\n"
                << "    <build>/postprocess/surface_to_netcdf "
                   "sea_level_fingerprint.csv --plot\n";
    } else {
      std::cout << "  wrote sea_level_fingerprint.csv (x,y,value rows "
                   "along the circle)\n";
    }
  }

  if (visualization) {
    GLVisWindow wu("displacement", DefaultKeys(dim));
    wu.Send(body, const_cast<GridFunction&>(prob.Displacement()));
    if (dim == 3) {
      GLVisWindow ws("sea-level change", DefaultKeys(2));
      ws.Send(sea.SurfaceMesh(), sea.SeaLevelChangeField());
      GridFunction c(&sea.SurfaceSpace());
      sea.OceanFunction(c);
      GLVisWindow wc("ocean function", DefaultKeys(2));
      wc.Send(sea.SurfaceMesh(), c);
      GridFunction load(&sea.SurfaceSpace());
      load.ProjectCoefficient(*melt_diag);
      GLVisWindow wl("melt load", DefaultKeys(2));
      wl.Send(sea.SurfaceMesh(), load);
    }
  }
  return 0;
}
