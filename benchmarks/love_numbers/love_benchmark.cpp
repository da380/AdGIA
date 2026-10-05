// ============================================================================
// love_benchmark.cpp
//
// Load and tidal Love numbers of a spherically layered body, for comparison
// with a radial reference solution. The body is a case written by
// make_case.py (benchmark_case.hpp); nothing about the model is set here.
// The formulation is chosen by -method (benchmark_case.hpp lists them;
// doc/benchmarks.tex, "The Love-number family", compares them).
//
// For each degree l (order 0) the surface load is set to sigma = Y_l0 and,
// separately for l >= 2, the tidal potential to psi = (r/a)^l Y_l0, with a
// the radius of the surface. The displacement, u = U Y r^ + V grad_1 Y, and
// the potential perturbation are analysed into harmonic coefficients on the
// surface and on every other interface that bounds a solid layer. With u_l,
// v_l and phi_l the (l, 0) coefficients of U, V and phi on the surface, g the
// surface gravity and phi_sigma the load's own potential on the surface
// (solved on the same mesh for the load alone, so that its discretisation
// error cancels in k'),
//
//   load:   h'_l = -g u_l / phi_sigma        k'_l = phi_l / phi_sigma - 1
//           l'_l = -g v_l / phi_sigma
//   tidal:  h_l  = -g u_l     l_l = -g v_l   k_l  = phi_l
//
// At degree one the solution is known up to a rigid translation, which the
// solver fixes by leaving the displacement without rigid component in the
// true-dof inner product. The numbers and the coefficients written are
// those of the frame of the centre of mass of the body and its load, in
// which the potential outside has no part of degree one and k'_1 = -1: the
// translation d = phi_1 / g is added to U and V, and g(r) d taken from phi
// on the interface of radius r.
//
// The results are written as JSON, with the coefficients on every interface
// per unit forcing, the radial functions U, V and phi of each solution in
// every layer (the polynomial in the radius that, times the harmonic, is
// closest to the solution there; -no-radial leaves them out), the sizes of
// the problem, the iteration counts and the times. The referential methods
// (referential, slip, slip_broken) solve the load problems only and write
// no radial functions.
//
// With -combined the problem, being linear, is solved twice in all: once
// for the sum of the unit loads of every degree from lmin to lmax and once
// for the sum of the tidal potentials of degrees two and above, the
// numbers of each degree analysed from the one solution as above (the
// analysis returns every coefficient in one pass, and the frame correction
// at degree one touches the degree-one coefficients alone). The discrete
// operator is not exactly rotationally invariant, so each degree's
// response then carries what the others' leak into it, which the solves by
// degree discard; README.md gives the measured size. The solves share one
// residual tolerance, and Dahlen's fluid leaves degree zero out. The cost
// of each combined solve is written once, at the top of the results, and
// the per-degree iteration counts and times are null.
//
// Sample runs:
//    mpiexec -np 8 ./love_benchmark -c cases/homogeneous/case.json -lmax 6
//    mpiexec -np 8 ./love_benchmark -c case.json -lmax 8 -combined
//    mpiexec -np 8 ./love_benchmark -c case.json -o 3 -out results_o3.json
// ============================================================================

#include <algorithm>
#include <fstream>
#include <map>

#include "benchmark_case.hpp"

using namespace mfem;
using namespace AdGIA;
using namespace benchmark;

namespace {

// The largest |c_j| over the harmonics j that were not forced, relative to
// |c_main|.
real_t Spurious(const Vector& c, int main, const std::vector<int>& forced) {
  real_t m = 0.0;
  for (int j = 0; j < c.Size(); j++) {
    if (std::find(forced.begin(), forced.end(), j) == forced.end()) {
      m = std::max(m, std::abs(c[j]));
    }
  }
  return m / (std::abs(c[main]) + 1e-300);
}

// The cost of one solve.
struct SolveCost {
  bool converged = false;
  int outer = 0, inner = 0;
  double seconds = 0.0;
};

// The response to one forcing of one degree.
struct Response {
  SolveCost cost;
  // True when the response was read from a combined solve of many
  // degrees, whose cost is written once for all of them: the per-degree
  // iteration counts and times are then null.
  bool shared = false;
  real_t spurious = 0.0;
  // The translation to the centre-of-mass frame (degree one; else zero).
  real_t shift = 0.0;
  // (l, 0) coefficients of U, V and phi on each analysed interface.
  std::vector<real_t> u, v, phi;
  // The radial functions of the harmonic in each layer.
  std::vector<Case::LayerProfile> profiles;
};

void Write(std::ostream& os, const char* name, const Response& r,
           const std::string& extra) {
  const auto count = [&](int n) {
    return r.shared ? std::string("null") : std::to_string(n);
  };
  os << "     \"" << name << "\": {" << extra
     << "\"converged\": " << (r.cost.converged ? "true" : "false")
     << ", \"outer_iterations\": " << count(r.cost.outer)
     << ", \"inner_iterations\": " << count(r.cost.inner)
     << ", \"seconds\": " << (r.shared ? "null" : Num(r.cost.seconds))
     << ", \"spurious\": " << Num(r.spurious)
     << ", \"shift\": " << Num(r.shift) << ",\n      \"u\": " << List(r.u)
     << ",\n      \"v\": " << List(r.v) << ",\n      \"phi\": " << List(r.phi)
     << ",\n      \"profiles\": [";
  for (std::size_t j = 0; j < r.profiles.size(); j++) {
    const auto& p = r.profiles[j];
    os << (j ? "," : "") << "\n       {\"attribute\": " << p.attribute
       << ", \"radius\": " << List(p.radius) << ", \"u\": " << List(p.u)
       << ", \"v\": " << List(p.v) << ", \"phi\": " << List(p.phi) << "}";
  }
  os << "]}";
}

std::string WriteCost(const SolveCost& s, const std::vector<int>& degrees) {
  return "{\"degrees\": " + List(degrees) + ", \"converged\": " +
         (s.converged ? "true" : "false") +
         ", \"outer_iterations\": " + std::to_string(s.outer) +
         ", \"inner_iterations\": " + std::to_string(s.inner) +
         ", \"seconds\": " + Num(s.seconds) + "}";
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();
  const bool root = Mpi::Root();

  CaseOptions options;
  const char* out_file = "results.json";
  int lmin = 0;
  bool tide = true;
  bool profiles = true;
  bool combined = false;

  OptionsParser args(argc, argv);
  options.Add(args);
  args.AddOption(&out_file, "-out", "--output", "Results file (JSON).");
  args.AddOption(&lmin, "-lmin", "--min-degree", "Lowest harmonic degree.");
  args.AddOption(&tide, "-tide", "--tide", "-no-tide", "--no-tide",
                 "Solve the tidal problems as well as the load problems.");
  // (-profiles is taken by the shared case options: a profiles file.)
  args.AddOption(&profiles, "-radial", "--radial", "-no-radial",
                 "--no-radial",
                 "Write the radial functions of each solution by layer.");
  args.AddOption(&combined, "-combined", "--combined", "-per-degree",
                 "--per-degree",
                 "One solve for the loads of all the degrees together and "
                 "one for the tides, the Love numbers of each degree "
                 "analysed from it; else one solve per degree and forcing.");
  args.Parse();
  if (!args.Good()) {
    if (root) args.PrintUsage(std::cout);
    return 1;
  }
  if (root) args.PrintOptions(std::cout);
  const int lmax = options.lmax;
  MFEM_VERIFY(lmin >= 0 && lmax >= lmin, "Degrees must satisfy 0 <= lmin <= "
                                         "lmax.");

  Case c(options);
  const auto& basis = c.Basis();
  const real_t a = c.radius, g = c.gravity;
  profiles = profiles && c.SupportsProfiles();
  tide = tide && c.SupportsTide();
  if (combined && c.method == "dahlen" && c.HasFluid() && lmin == 0) {
    // Dahlen's fluid differs from the reference at degree zero by design
    // (README.md), and its degree-zero response would be carried into the
    // others' by the discrete coupling of the degrees: leave it out of the
    // combined load.
    lmin = 1;
    if (root) {
      std::cout << "Combined: degree zero left out of the load (Dahlen's "
                   "fluid).\n";
    }
  }

  // The gravity of the background at the radii of the profiles.
  std::vector<Vector> profile_gravity;
  if (profiles) {
    Vector none;
    ParGridFunction zero_u(c.fes_u.get()), zero_phi(c.fes_phi.get());
    zero_u = 0.0;
    zero_phi = 0.0;
    auto layout = c.Profiles(zero_u, zero_phi, 0);
    c.ProfileGravity(layout, profile_gravity);
  }

  // Solve with unit coefficients on the harmonics `forced`.
  auto solve = [&](const std::vector<int>& forced, bool load) {
    Vector coefficients(basis.Size());
    coefficients = 0.0;
    for (const int i : forced) {
      coefficients[i] = 1.0;
    }
    SolveCost s;
    MPI_Barrier(MPI_COMM_WORLD);
    const auto t0 = Clock::now();
    s.converged = c.Solve(coefficients, load);
    s.seconds = Seconds(t0);
    s.outer = c.OuterIterations();
    s.inner = c.InnerIterations();
    return s;
  };

  // The coefficients of the current solution on every analysed interface.
  struct Coefficients {
    std::vector<Vector> u, v, phi;
  };
  auto analyse = [&]() {
    Coefficients k;
    for (const auto& an : c.analyses) {
      k.u.emplace_back();
      k.v.emplace_back();
      k.phi.emplace_back();
      an.radial->Coefficients(c.Displacement(), k.u.back());
      an.tangential->Coefficients(c.Displacement(), k.v.back());
      an.scalar->Coefficients(c.AnalysisPotential(), k.phi.back());
    }
    return k;
  };

  // The response of harmonic i, read from the current solution and its
  // coefficients k; `forced` are the harmonics the solve was forced on.
  auto extract = [&](int i, const Coefficients& k,
                     const std::vector<int>& forced) {
    Response r;
    for (std::size_t j = 0; j < c.analyses.size(); j++) {
      const auto& an = c.analyses[j];
      const Vector &cu = k.u[j], &cv = k.v[j], &cphi = k.phi[j];
      r.u.push_back(cu[i]);
      r.v.push_back(cv[i]);
      // The referential methods analyse zeta1 = phi1 + u . grad zeta0:
      // convert with the interface's own gravity (exact, layer masses).
      r.phi.push_back(c.PotentialIsReferential()
                          ? cphi[i] - an.gravity * cu[i]
                          : cphi[i]);
      if (&an == &c.Surface()) {
        r.spurious = std::max(Spurious(cu, i, forced),
                              Spurious(cphi, i, forced));
      }
    }
    if (profiles) {
      r.profiles = c.Profiles(c.problem->Displacement(),
                              c.problem->Potential(), i);
    }
    // The centre-of-mass frame: a translation is of degree one alone, so
    // the correction touches the (1, 0) coefficients only, and is the same
    // whether the solution holds other degrees or not.
    if (basis.Degree(i) == 1) {
      r.shift = r.phi[c.surface] / g;
      for (std::size_t j = 0; j < r.profiles.size(); j++) {
        auto& p = r.profiles[j];
        for (int s = 0; s < p.radius.Size(); s++) {
          if (p.u.Size()) {
            p.u[s] += r.shift;
            p.v[s] += r.shift;
          }
          p.phi[s] -= profile_gravity[j][s] * r.shift;
        }
      }
      for (std::size_t j = 0; j < c.analyses.size(); j++) {
        r.u[j] += r.shift;
        r.v[j] += r.shift;
        r.phi[j] -= c.analyses[j].gravity * r.shift;
      }
    }
    return r;
  };

  ParGridFunction phi_direct(
      static_cast<ParFiniteElementSpace*>(&c.AnalysisPotentialSpace()));
  // The load's own potential on the surface, coefficients by harmonic.
  auto load_potential = [&]() {
    c.SolveLoadPotential(phi_direct);
    Vector cdirect;
    c.Surface().scalar->Coefficients(phi_direct, cdirect);
    return cdirect;
  };

  // The responses by degree: in combined mode from the two solves, else
  // one solve per degree and forcing in the loop below.
  std::map<int, Response> loads, tides;
  std::map<int, real_t> phi_sigmas;
  SolveCost load_cost, tide_cost;
  std::vector<int> load_degrees, tide_degrees;
  if (combined) {
    std::vector<int> forced;
    for (int l = lmin; l <= lmax; l++) {
      load_degrees.push_back(l);
      forced.push_back(basis.Index(l, 0));
    }
    load_cost = solve(forced, true);
    {
      const Coefficients k = analyse();
      for (int l = lmin; l <= lmax; l++) {
        loads[l] = extract(basis.Index(l, 0), k, forced);
        loads[l].shared = true;
        loads[l].cost.converged = load_cost.converged;
      }
      const Vector cdirect = load_potential();
      for (int l = lmin; l <= lmax; l++) {
        phi_sigmas[l] = cdirect[basis.Index(l, 0)];
      }
    }
    forced.clear();
    for (int l = std::max(lmin, 2); tide && l <= lmax; l++) {
      tide_degrees.push_back(l);
      forced.push_back(basis.Index(l, 0));
    }
    if (!forced.empty()) {
      // Independent forcings: the tide solve starts cold.
      c.ResetSolution();
      tide_cost = solve(forced, false);
      const Coefficients k = analyse();
      for (const int l : tide_degrees) {
        tides[l] = extract(basis.Index(l, 0), k, forced);
        tides[l].shared = true;
        tides[l].cost.converged = tide_cost.converged;
      }
    }
  }

  std::ostringstream degrees;
  if (root) {
    std::cout << std::setprecision(6)
              << "\n  l          h'          l'          k'           h"
              << "           l           k    spurious   phi_s  its     time\n";
  }
  for (int l = lmin; l <= lmax; l++) {
    const int i = basis.Index(l, 0);
    const bool with_tide = tide && l >= 2;
    if (!combined) {
      const std::vector<int> forced{i};
      const SolveCost cost = solve(forced, true);
      loads[l] = extract(i, analyse(), forced);
      loads[l].cost = cost;
      phi_sigmas[l] = load_potential()[i];
      if (with_tide) {
        const SolveCost tide_cost_l = solve(forced, false);
        tides[l] = extract(i, analyse(), forced);
        tides[l].cost = tide_cost_l;
      }
    }
    const Response& load = loads[l];
    const real_t phi_sigma = phi_sigmas[l];
    const real_t phi_exact = -4.0 * kPi * c.G * a / (2.0 * l + 1.0);
    const real_t h_load = -g * load.u[c.surface] / phi_sigma;
    const real_t k_load = load.phi[c.surface] / phi_sigma - 1.0;
    const real_t l_load = -g * load.v[c.surface] / phi_sigma;

    Response tidal;
    real_t h = NAN, k = NAN, l_tide = NAN;
    if (with_tide) {
      tidal = tides[l];
      h = -g * tidal.u[c.surface];
      k = tidal.phi[c.surface];
      l_tide = -g * tidal.v[c.surface];
    }

    if (root) {
      std::cout << std::setw(3) << l << std::setw(12) << h_load
                << std::setw(12) << l_load << std::setw(12) << k_load
                << std::setw(12) << h << std::setw(12) << l_tide
                << std::setw(12) << k << std::setw(12) << std::setprecision(2)
                << std::max(load.spurious, tidal.spurious) << std::setw(8)
                << std::setprecision(4) << phi_sigma / phi_exact;
      if (combined) {
        std::cout << std::setw(5) << "-" << std::setw(9) << "-";
      } else {
        std::cout << std::setw(5) << load.cost.outer << std::setw(9)
                  << std::setprecision(3)
                  << load.cost.seconds + tidal.cost.seconds;
      }
      std::cout << std::setprecision(6)
                << (load.cost.converged &&
                            (!with_tide || tidal.cost.converged)
                        ? ""
                        : "   NOT CONVERGED")
                << "\n";
      degrees << (l > lmin ? ",\n" : "") << "    {\"degree\": " << l << ",\n";
      Write(degrees, "load", load,
            "\"h\": " + Num(h_load) + ", \"l\": " + Num(l_load) +
                ", \"k\": " + Num(k_load) +
                ", \"phi_direct\": " + Num(phi_sigma) +
                ", \"phi_direct_exact\": " + Num(phi_exact) + ", ");
      if (with_tide) {
        degrees << ",\n";
        Write(degrees, "tide", tidal,
              "\"h\": " + Num(h) + ", \"l\": " + Num(l_tide) +
                  ", \"k\": " + Num(k) + ", ");
      }
      degrees << "}";
    }
  }
  if (root && combined) {
    std::cout << std::setprecision(3) << "\nCombined load solve (degrees "
              << lmin << " to " << lmax << "): " << load_cost.outer
              << " iterations, " << load_cost.seconds << " s"
              << (load_cost.converged ? "" : ", NOT CONVERGED") << "\n";
    if (!tide_degrees.empty()) {
      std::cout << "Combined tide solve (degrees " << tide_degrees.front()
                << " to " << lmax << "): " << tide_cost.outer
                << " iterations, " << tide_cost.seconds << " s"
                << (tide_cost.converged ? "" : ", NOT CONVERGED") << "\n";
    }
  }

  if (root) {
    std::ofstream os(out_file);
    MFEM_VERIFY(os.good(), "Cannot write " << out_file << ".");
    os << "{\n";
    c.WriteHeader(os);
    os << ",\n  \"combined\": " << (combined ? "true" : "false");
    if (combined) {
      // The cost of the two solves, written once: the per-degree entries
      // carry null iteration counts and times.
      os << ",\n  \"combined_solves\": {\"load\": "
         << WriteCost(load_cost, load_degrees);
      if (!tide_degrees.empty()) {
        os << ",\n    \"tide\": " << WriteCost(tide_cost, tide_degrees);
      }
      os << "}";
    }
    os << ",\n  \"degrees\": [\n" << degrees.str() << "]\n}\n";
    std::cout << "\nWrote " << out_file << "\n";
  }
  return 0;
}
