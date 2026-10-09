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
// The problem is the sea-level benchmark family's
// (benchmarks/sea_level), at example-sized amplitudes and without the
// pyslfp comparison: smooth analytical topography — a super-Gaussian
// continent at the pole (+z in 3-D, +y in 2-D), ocean elsewhere — an
// ice cap on the continent, and a melt that unloads the +x hemisphere
// of the cap through the smooth factor (1 + tanh(x/melt_width))/2
// (regular at the pole), so the fingerprint carries non-zonal structure.
// The ocean function C0 enters the operator pointwise through the ocean
// weight w = rho_w C0 / g; nothing is projected. Every piece of the
// geometry and load is an option.
//
// Output: the melt load and the sea-level change SL1 on the surface (the fingerprint: a
// far-field rise of roughly the eustatic value, depressed — possibly
// below zero — near the melted cap where gravitational attraction is
// lost and the ground rebounds), the uniform term Phi_g, the eustatic
// equivalent, the ocean area and the mass-conservation certificate. The
// fingerprint is exported with SeaLevelOperator::WriteSurfaceField; in
// 3-D, grid and map it with
//    <build>/postprocess/surface_to_netcdf sea_level_fingerprint.csv --plot
// (cartopy/pyshtools-ready NetCDF; see postprocess/README.md). GLVis
// shows the body displacement and, in 3-D, the fingerprint on the
// surface shell; in 2-D the fingerprint is written as a polar profile
// CSV for plot_csv.py.
//
// One source serves the serial and the parallel build; the only genuine
// differences are the partitioning and typed field copies.
//
// Options (defaults in brackets):
//   -m       mesh file [../data/coupled_poisson.msh, the 3-D ball];
//            ../data/elastogravity_2d.msh runs the 2-D disc (a polar
//            profile rather than a map).
//   -o       finite element order [2].
//   -r       uniform mesh refinements [0].
//   -Omega   equilibrium rotation rate about e3 [0: rotational feedback
//            off]; with it the angular-velocity border joins the solve
//            and psi(omega) enters the fingerprint.
//   -C1 -C2 -C3   principal moments [1.2, 1.3, 2.0]; 2-D uses -C3 only.
//   -rhow    water density [0.05] (non-dimensional, like the model).
//   -rhoi    ice density [0.045].
//   -ocean-depth   initial ocean depth [1.0].
//   -cont-amp      continent height [3.0].
//   -cont-width    continent angular width (rad) [0.7].
//   -cap-amp       ice-cap thickness [1.0].
//   -cap-width     ice-cap angular width (rad) [0.35].
//   -melt    melted fraction of the cap [0.5].
//   -melt-width    smoothing width of the hemispheric unloading [0.15].
//   -shore   shoreline smoothing width of the ocean fraction [0.02];
//            a sharp indicator (small width) costs quadrature accuracy
//            in the shoreline-crossing boundary elements.
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
//    ./sea_level_fingerprint -m ../data/elastogravity_2d.msh
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

// The state of the sea-level benchmark family (benchmarks/sea_level),
// at example-sized amplitudes and with every knob an option: a
// super-Gaussian continent at the pole (+z in 3-D, +y in 2-D), an ice
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

// Colatitude from the pole: +z in 3-D, +y in 2-D.
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

// The melt load sigma_data = -rho_i (1 - C0) melt H(x) I0 with the
// smooth hemispheric factor H = (1 + tanh(x0/melt_width))/2, regular at
// the pole (Cartesian coordinate, not an angle).
double MeltLoad(const Vector& x) {
  const double hemi = 0.5 * (1.0 + std::tanh(x[0] / g_melt_width));
  return -(1.0 - OceanFraction(x)) * g_rho_i * g_melt * hemi *
         IceThickness(x);
}

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif

  const char* mesh_file = "../data/coupled_poisson.msh";
  int order = 2;
  int ref_levels = 0;
  double Omega = 0.0;
  double C1 = 1.2, C2 = 1.3, C3 = 2.0;
  bool visualization = true;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
                 "Body-in-buffer mesh (2-D disc or 3-D ball).");
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
  args.AddOption(&g_shore, "-shore", "--shoreline-width",
                 "Shoreline smoothing width of the ocean fraction.");
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

  ConstantCoefficient kappa(kKappa), mu(kMu), rho(kRho);
  IsotropicElasticRheology rheology(dim, kappa, mu);
  FunctionCoefficient w(OceanWeight), sigma_data(MeltLoad);
  const double g0 = G0();
  VectorFunctionCoefficient grad_phi0(dim, [g0](const Vector& x, Vector& v) {
    v = x;
    v *= g0;
  });

  // The monolithic solve.
  LinearQuasiStaticMixedSelfGravitatingProblem prob(&fes_u, &fes_phi,
                                                    rheology, rho, kG,
                                                    kDtNDegree);
  prob.SetWaterLoad(w, sigma_data, surface);
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
  prob.AssembleForce(0.0);
  if (!prob.Solve()) {
    if (IsRoot()) {
      std::cout << "solve failed\n";
    }
    return 1;
  }

  // The surface layer: the initial state, the fingerprint, diagnostics.
  SeaLevelOperator sea(fes_u, surface);
  sea.SetDensities(g_rho_w, g_rho_i);
  FunctionCoefficient sl0(InitialSeaLevel), ice(IceThickness);
  sea.SetInitialState(sl0, ice);
  CentrifugalPotential psi(dim, Omega);
  if (Omega != 0.0) {
    psi.SetAmplitudes(prob.AngularVelocity());
  }
  sea.SeaLevelChangeFrom(prob.Displacement(), grad_phi0, prob.Potential(),
                         prob.UniformPotentialTerm(),
                         Omega != 0.0 ? &psi : nullptr);

  // Diagnostics over the same analytic ocean function the solve used
  // (the field-projected indicator differs near the shoreline at
  // interpolation grade).
  FunctionCoefficient ocean_ind(OceanFraction);
  ConstantCoefficient one(1.0);
  ProductCoefficient ocean_area_ind(ocean_ind, one);
  const double area = sea.SurfaceIntegral(ocean_area_ind);
  FunctionCoefficient melt(MeltLoad);
  const double melted_mass = -sea.SurfaceIntegral(melt);
  const double eustatic = melted_mass / (g_rho_w * area);
  GridFunctionCoefficient slc(&sea.SeaLevelChangeField());
  ProductCoefficient ocean_sl(ocean_ind, slc);
  const double ocean_mean = sea.SurfaceIntegral(ocean_sl) / area;
  if (IsRoot()) {
    std::cout << "\nsea-level fingerprint (rho_w " << g_rho_w << ", rho_i "
              << g_rho_i << ", melt " << g_melt << ")\n";
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
      load.ProjectCoefficient(melt);
      GLVisWindow wl("melt load", DefaultKeys(2));
      wl.Send(sea.SurfaceMesh(), load);
    } else if (IsRoot()) {
      // 2-D: the surface is a circle; the exported CSV (x, y, value)
      // is the profile — plot theta = atan2(y, x) against value.
      std::cout << "  (2-D: plot the exported CSV as a polar profile, "
                   "theta = atan2(y, x) vs value)\n";
    }
  }
  return 0;
}
