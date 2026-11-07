// ============================================================================
// stability_probe: the global gravitational-stability criterion, measured.
//
// For a case made by make_case.py (a gravitating model with its fields),
// builds the mixed self-gravitating problem with the shear modulus of the
// chosen solid attributes scaled by a factor f (default: every solid
// attribute, i.e. "the mantle" of the ladder models) and reports the lowest
// generalised eigenvalues of the Schur complement
//
//     S = A_uu - C A_phiphi^{-1} C^T,   S x = lambda M x,
//
// M the unit vector mass (LinearQuasiStaticMixedSelfGravitatingProblem::
// SchurEigenvalues). The body is globally stable iff the lowest eigenvalue
// on the complement of the rigid modes is positive; the sweep over f
// (-f 1,0.1,0.01,...) locates the effective shear at which the static
// problem of the model stops being physical, i.e. where the local
// (buoyancy-frequency) instability of its sub-adiabatic layers, which the
// unrelaxed shear holds, takes over. The rigid modes are projected out of
// the LOBPCG subspace before the Rayleigh-Ritz values are read; each mode
// carries its residual and, with -region, its mass fraction on the given
// attributes of the displacement mesh.
//
// On a model with a Dahlen fluid core the naked operator (no level border)
// has one negative mode at every f whatever the core's stratification: the
// degree-0 closure defect of the naked fluid, which the level border
// closes in the solves but which no Schur complement of A_uu contains.
// Read the lowest degenerate cluster above it (2l+1 modes of degree l) as
// the physical criterion, or run a model without a fluid core.
//
// The operator is the naked Dahlen one (no water-load border, the fluid
// regions as the case defines them). Output: JSON with the case header,
// then per f the Ritz values, their residuals and fluid fractions.
//
// Sample run:
//    mpiexec -np 4 ./stability_probe -c runs/fluid_core/h0.3/case.json \
//        -o 2 -f 1,0.1,0.01 -nev 10 -out probe.json
// ============================================================================
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "benchmark_case.hpp"

using namespace mfem;
using namespace AdGIA;
using namespace benchmark;

namespace {

std::vector<real_t> ParseList(const std::string& s) {
  std::vector<real_t> v;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) v.push_back(std::stod(item));
  }
  return v;
}

std::vector<int> ParseInts(const std::string& s) {
  std::vector<int> v;
  for (const real_t x : ParseList(s)) v.push_back(static_cast<int>(x));
  return v;
}

}  // namespace

int main(int argc, char* argv[]) try {
  Mpi::Init(argc, argv);
  Hypre::Init();
  const bool root = Mpi::Root();

  CaseOptions options;
  const char* out_file = "stability.json";
  const char* factors = "1";
  const char* attributes = "";
  const char* region = "";
  int nev = 10, max_iter = 400;
  real_t tol = 1e-8;

  OptionsParser args(argc, argv);
  options.Add(args);
  args.AddOption(&out_file, "-out", "--output", "Results file (JSON).");
  args.AddOption(&factors, "-f", "--factors",
                 "Comma-separated shear-modulus factors to sweep.");
  args.AddOption(&attributes, "-attr", "--attributes",
                 "Comma-separated element attributes whose shear is scaled "
                 "(default: every solid attribute).");
  args.AddOption(&region, "-region", "--region",
                 "Comma-separated attributes OF THE DISPLACEMENT MESH on "
                 "which each mode's mass fraction is reported (the Dahlen "
                 "fluid is eliminated, so its attributes are not there; "
                 "with -cmb-keep-inner a meshed fluid is). Default: none.");
  args.AddOption(&nev, "-nev", "--num-eigenvalues",
                 "Eigenvalues per factor (rigid modes included: 3 in 2-D, "
                 "6 in 3-D, plus region rotations).");
  args.AddOption(&max_iter, "-eig-max-iter", "--eig-max-iter",
                 "LOBPCG iteration cap.");
  args.AddOption(&tol, "-eig-tol", "--eig-tol", "LOBPCG tolerance.");
  args.Parse();
  if (!args.Good()) {
    if (root) args.PrintUsage(std::cout);
    return 1;
  }
  if (root) args.PrintOptions(std::cout);
  MFEM_VERIFY(std::string(options.method) == "dahlen",
              "stability_probe: the mixed (Dahlen) class only.");

  const std::vector<real_t> fs = ParseList(factors);
  const std::vector<int> attrs = ParseInts(attributes);
  MFEM_VERIFY(!fs.empty(), "stability_probe: -f needs at least one factor.");

  // The attributes whose shear is scaled: the manifest's solid layers by
  // default (for the ladder models, the mantle and any solid inner core).
  planetmodel::MeshManifest manifest(options.manifest);
  Array<int> scaled_attrs =
      attrs.empty() ? ToArray(manifest.SolidAttributes()) : Array<int>();
  for (const int a : attrs) scaled_attrs.Append(a);
  if (root) {
    std::cout << "Scaling the shear of attribute" << (scaled_attrs.Size() > 1 ? "s" : "");
    for (const int a : scaled_attrs) std::cout << " " << a;
    std::cout << " by each factor.\n";
  }

  std::string header;
  std::ostringstream sweep;
  sweep.precision(17);
  for (std::size_t k = 0; k < fs.size(); k++) {
    const real_t f = fs[k];
    // The scaled rheology: mu f on the chosen attributes, mu elsewhere.
    // The coefficients must outlive the Case, hence the holders.
    std::unique_ptr<PWConstCoefficient> scale;
    std::unique_ptr<ProductCoefficient> mu_scaled;
    std::unique_ptr<IsotropicElasticRheology> rheology;
    Case::RheologyFactory factory = [&](ParSubMesh& mesh, Coefficient& kappa,
                                        Coefficient& mu) -> const Rheology* {
      const int amax = mesh.attributes.Max();
      Vector s(amax);
      s = 1.0;
      for (const int a : scaled_attrs) {
        MFEM_VERIFY(a >= 1 && a <= amax, "stability_probe: attribute "
                                               << a << " is not on the mesh.");
        s(a - 1) = f;
      }
      scale = std::make_unique<PWConstCoefficient>(s);
      mu_scaled = std::make_unique<ProductCoefficient>(mu, *scale);
      rheology = std::make_unique<IsotropicElasticRheology>(
          mesh.Dimension(), kappa, *mu_scaled);
      return rheology.get();
    };
    Case c(options, factory);
    MFEM_VERIFY(c.eulerian, "stability_probe: the mixed class only.");
    if (header.empty()) {
      std::ostringstream h;
      c.WriteHeader(h);
      header = h.str();
    }
    c.problem->AssembleForce(0.0);
    const auto start = Clock::now();
    const std::vector<int> region_list = ParseInts(region);
    Array<int> region_attrs;
    for (const int a : region_list) region_attrs.Append(a);
    const auto eig = c.problem->SchurEigenvalues(
        nev, max_iter, tol, region_attrs.Size() > 0 ? &region_attrs : nullptr);
    const double seconds = Seconds(start);
    const real_t lowest = eig.empty() ? std::numeric_limits<real_t>::quiet_NaN()
                                      : eig.front().value;
    if (root) {
      std::cout << "f = " << f << ": lowest non-rigid eigenvalue " << lowest
                << " (" << eig.size() << " Ritz values, " << seconds
                << " s)\n";
      for (const auto& e : eig) {
        std::cout << "    " << e.value << "  residual " << e.residual
                  << "  region fraction " << e.region_fraction << "\n";
      }
    }
    std::vector<real_t> values, residuals, fractions;
    for (const auto& e : eig) {
      values.push_back(e.value);
      residuals.push_back(e.residual);
      fractions.push_back(e.region_fraction);
    }
    sweep << (k ? "," : "") << "\n    {\"f\": " << Num(f)
          << ", \"lowest_nonrigid\": " << Num(lowest)
          << ", \"seconds\": " << Num(seconds)
          << ",\n     \"eigenvalues\": " << List(values)
          << ",\n     \"residual\": " << List(residuals)
          << ",\n     \"region_fraction\": " << List(fractions) << "}";
  }
  std::ostringstream os;
  os << "{\n" << header << ",\n  \"scaled_attributes\": "
     << List(std::vector<real_t>(scaled_attrs.begin(), scaled_attrs.end()))
     << ",\n  \"factors\": " << List(fs) << ",\n  \"sweep\": ["
     << sweep.str();
  os << "\n  ]\n}\n";
  if (root) {
    std::ofstream out(out_file);
    out << os.str();
    std::cout << "Wrote " << out_file << "\n";
  }
  return 0;
} catch (const std::exception& e) {
  std::cerr << "stability_probe: " << e.what() << "\n";
  return 1;
}
