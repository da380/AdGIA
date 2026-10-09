// ============================================================================
// sea_level_benchmark.cpp
//
// The elastic sea-level fingerprint of an ice melt on a spherically
// layered case (benchmark_case.hpp; nothing about the model is set
// here), solved monolithically with the water-load feedback
// (LinearQuasiStaticMixedSelfGravitatingProblem::SetWaterLoad), for
// comparison with pyslfp's pseudo-spectral solution of the same problem
// (run.py drives both sides; doc/planning/sea_level_plan.md, rung 1).
//
// The initial state is analytic and smooth (the pyslfp testing trick),
// axisymmetric about +z, in the case's non-dimensional units:
//
//   SL0(theta) = ocean_depth - cont_amp exp(-((theta/cont_width)^6)/2)
//   I0(theta)  = cap_amp exp(-((theta/cap_width)^2)/2)
//   C0         = (1 + tanh((rho_w SL0 - rho_i I0)/shore)) / 2
//   sigma_data = -(1 - C0) rho_i melt H(x) I0,
//   H(x)       = (1 + tanh(x/melt_width))/2
//
// — the melt unloads the +x hemisphere of the cap smoothly, so the
// fingerprint carries non-zonal harmonics (order m != 0) as well.
//
// run.py passes the same constants to the pyslfp side, so the two
// solvers see one problem. Shorelines are frozen (first-order exact) and
// rotation is off (WP4). The surface gravity entering the ocean weight
// w = rho_w C0 / g is the case's own (G M / a^2).
//
// Output: the sea-level change as a nodal CSV
// (SeaLevelOperator::WriteSurfaceField) and a JSON of scalars (the
// uniform term, ocean area, eustatic equivalent, ocean mean, sizes,
// iterations, times).
//
// Options (defaults in brackets): the case options of
// benchmark_case.hpp (-c, -o, ...; -method must stay dahlen) plus
//   -rhow / -rhoi   water and ice density, case units [0.2, 0.1834]
//   -ocean-depth    [6.3e-4]      -cont-amp   [1.4e-3]
//   -cont-width     [0.7]         -cap-amp    [4.7e-4]
//   -cap-width      [0.35]        -melt       [0.5]
//   -shore          smoothing width of the ocean fraction, in units of
//                   rho_w*SL [1e-5]
//   -slcsv          the fingerprint CSV [sea_level.csv]
//   -out            the scalars JSON [sea_level.json]
//
// Sample run (through run.py normally):
//    mpiexec -np 4 sea_level_benchmark -c <case>/case.json -o 2
// ============================================================================

#include <cmath>
#include <fstream>
#include <iostream>

#include "benchmark_case.hpp"

using namespace mfem;
using namespace AdGIA;
using namespace benchmark;

namespace {

double g_rho_w = 0.2;
double g_rho_i = 0.1834;
double g_ocean_depth = 6.3e-4;
double g_cont_amp = 1.4e-3;
double g_cont_width = 0.7;
double g_cap_amp = 4.7e-4;
double g_cap_width = 0.35;
double g_melt = 0.5;
double g_melt_width = 0.15;
double g_shore = 1e-5;
double g_gravity = 1.0;

double Colatitude(const Vector& x) {
  const double r = x.Norml2();
  return std::acos(std::min(1.0, std::max(-1.0, x[2] / r)));
}

double InitialSeaLevel(const Vector& x) {
  const double t = Colatitude(x) / g_cont_width;
  return g_ocean_depth - g_cont_amp * std::exp(-0.5 * std::pow(t, 6));
}

double IceThickness(const Vector& x) {
  const double t = Colatitude(x) / g_cap_width;
  return g_cap_amp * std::exp(-0.5 * t * t);
}

double OceanFraction(const Vector& x) {
  const double q = g_rho_w * InitialSeaLevel(x) - g_rho_i * IceThickness(x);
  return 0.5 * (1.0 + std::tanh(q / g_shore));
}

double OceanWeight(const Vector& x) {
  return g_rho_w * OceanFraction(x) / g_gravity;
}

// The melt unloads the +x hemisphere of the cap, smoothly: the factor
// (1 + tanh(x0/width))/2 is expressed in the Cartesian coordinate, so
// it is regular at the pole.
double MeltLoad(const Vector& x) {
  const double hemi = 0.5 * (1.0 + std::tanh(x[0] / g_melt_width));
  return -(1.0 - OceanFraction(x)) * g_rho_i * g_melt * hemi *
         IceThickness(x);
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();

  CaseOptions options;
  const char* slcsv = "sea_level.csv";
  const char* out = "sea_level.json";
  OptionsParser args(argc, argv);
  options.Add(args);
  args.AddOption(&g_rho_w, "-rhow", "--water-density",
                 "Water density, case units.");
  args.AddOption(&g_rho_i, "-rhoi", "--ice-density",
                 "Ice density, case units.");
  args.AddOption(&g_ocean_depth, "-ocean-depth", "--ocean-depth",
                 "Initial ocean depth, case units.");
  args.AddOption(&g_cont_amp, "-cont-amp", "--continent-amplitude",
                 "Continent height, case units.");
  args.AddOption(&g_cont_width, "-cont-width", "--continent-width",
                 "Continent angular width (rad).");
  args.AddOption(&g_cap_amp, "-cap-amp", "--cap-amplitude",
                 "Ice-cap thickness, case units.");
  args.AddOption(&g_cap_width, "-cap-width", "--cap-width",
                 "Ice-cap angular width (rad).");
  args.AddOption(&g_melt, "-melt", "--melt-fraction",
                 "Melted fraction of the cap.");
  args.AddOption(&g_melt_width, "-melt-width", "--melt-width",
                 "Smoothing width (case length units) of the hemispheric "
                 "unloading factor (1 + tanh(x/width))/2.");
  args.AddOption(&g_shore, "-shore", "--shoreline-width",
                 "Ocean-fraction smoothing width (rho_w * length units).");
  args.AddOption(&slcsv, "-slcsv", "--sea-level-csv",
                 "Output CSV of the fingerprint's nodal values.");
  args.AddOption(&out, "-out", "--output", "Output JSON of the scalars.");
  args.Parse();
  if (!args.Good()) {
    if (Mpi::Root()) {
      args.PrintUsage(std::cout);
    }
    return 1;
  }
  MFEM_VERIFY(std::string(options.method) == "dahlen" && !options.gauged,
              "sea_level_benchmark: the water load lives on the plain "
              "Eulerian problem (-method dahlen).");

  Case c(options);
  g_gravity = c.gravity;
  const Array<int>& surface = c.analyses[c.surface].radial->Marker();

  FunctionCoefficient w(OceanWeight), sigma_data(MeltLoad);
  c.problem->SetWaterLoad(w, sigma_data, surface);
  const auto t0 = Clock::now();
  c.problem->AssembleForce(0.0);
  MFEM_VERIFY(c.problem->Solve(), "the coupled solve failed");
  const double solve_s = Seconds(t0);

  // The fingerprint: SL1 = -(u.grad Phi0 + phi)/g + Phi_g/g, with
  // grad Phi0 = (g/a) x exact at the surface (only surface nodal values
  // enter the restriction).
  const double ga = c.gravity / c.radius;
  VectorFunctionCoefficient grad_phi0(3, [ga](const Vector& x, Vector& v) {
    v = x;
    v *= ga;
  });
  SeaLevelOperator sea(*c.fes_u, surface);
  sea.SeaLevelChangeFrom(c.problem->Displacement(), grad_phi0,
                         c.problem->Potential(),
                         c.problem->UniformPotentialTerm());
  sea.WriteSurfaceField(sea.SeaLevelChangeField(), slcsv);

  // Scalars, over the same analytic ocean fraction the solve used.
  FunctionCoefficient ocean_ind(OceanFraction);
  ConstantCoefficient one(1.0);
  ProductCoefficient area_ind(ocean_ind, one);
  const double area = sea.SurfaceIntegral(area_ind);
  FunctionCoefficient melt(MeltLoad);
  const double melted_mass = -sea.SurfaceIntegral(melt);
  const double eustatic = melted_mass / (g_rho_w * area);
  GridFunctionCoefficient slc(&sea.SeaLevelChangeField());
  ProductCoefficient ocean_sl(ocean_ind, slc);
  const double ocean_mean = sea.SurfaceIntegral(ocean_sl) / area;

  if (Mpi::Root()) {
    std::ofstream os(out);
    os << "{\n";
    os << "  \"gravity\": " << Num(c.gravity) << ",\n";
    os << "  \"radius\": " << Num(c.radius) << ",\n";
    os << "  \"phi_g\": " << Num(c.problem->UniformPotentialTerm())
       << ",\n";
    os << "  \"ocean_area\": " << Num(area) << ",\n";
    os << "  \"melted_mass\": " << Num(melted_mass) << ",\n";
    os << "  \"eustatic\": " << Num(eustatic) << ",\n";
    os << "  \"ocean_mean\": " << Num(ocean_mean) << ",\n";
    os << "  \"displacement_unknowns\": " << c.displacement_unknowns
       << ",\n";
    os << "  \"iterations\": " << c.problem->TotalIterations() << ",\n";
    os << "  \"solve_seconds\": " << Num(solve_s) << "\n";
    os << "}\n";
    std::cout << "sea level: phi_g " << c.problem->UniformPotentialTerm()
              << ", eustatic " << eustatic << ", ocean mean " << ocean_mean
              << ", wrote " << slcsv << " and " << out << "\n";
  }
  return 0;
}
