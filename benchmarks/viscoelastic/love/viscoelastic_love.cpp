// ============================================================================
// viscoelastic_love.cpp
//
// Viscoelastic load (and tidal) Love-number HISTORIES of a spherically
// layered Maxwell body, for comparison with the correspondence-principle
// reference of laplace_reference.py. The body is a case written by
// make_case.py (benchmark_case.hpp), exactly as for love_benchmark; what this
// driver adds is the rheology and the time.
//
// Every solid layer gets a Maxwell time (-tau, centre outward, one value per
// solid layer or one for all; "inf" keeps a layer elastic), and the problem
// is the case's LinearQuasiStaticMixedSelfGravitatingProblem built with a
// CompositeRheology: per solid layer an IsotropicMaxwellRheology of the
// case's own kappa and mu with that layer's tau, or the elastic rheology. At
// t = 0 a Heaviside surface load sigma = sum_l Y_l0 over the degrees
// lmin..lmax is switched on (one evolution serves every degree: the problem
// is linear; the discrete leakage between degrees is the one love_benchmark
// -combined measures), and ViscoelasticOperator evolves the internal
// variables to the output times. At each output time the surface
// coefficients of the displacement and the potential give, with g the
// surface gravity and phi_sigma the load's own potential,
//
//   h'_l(t) = -g u_l(t) / phi_sigma,  l'_l(t) = -g v_l(t) / phi_sigma,
//   k'_l(t) = phi_l(t) / phi_sigma - 1,
//
// as in love_benchmark (degree one in the centre-of-mass frame). With -tide
// the tidal potential (r/a)^l Y_l0, l >= 2, is evolved as well, in a second
// run from rest.
//
// Time stepping (-scheme): sdirk23 (the stepper survey's best fixed-step
// scheme, two solves per step; the default), exptrap (exponential
// trapezoid), be (backward Euler), etd1 (exponential Euler), adaptive (the
// adaptive exponential trapezoid at -rtol). Fixed-step schemes take, within
// each interval between output times, the same number of equal steps: at
// least -steps, more if needed to keep dt <= -dt-max. The output times are
// therefore hit exactly, are NOT tied to a fixed step grid, and the
// default times (geometric, ratio 2^(5/6), the reference's own) fall at a
// different place in the relaxation each, so that the history metric of
// compare.py (worst error over all output times, the survey's metric) cannot
// be scored stroboscopically.
//
// Times and taus are in the same, arbitrary, unit: the quasi-static problem
// knows time only through the relaxation times.
//
// The results are JSON: the header of the case, the layers and their taus,
// the scheme, per output time the Love numbers per degree (the schema of
// laplace_reference.py's "histories") with the cumulative cost, the elastic
// (t = 0+) numbers, and the totals: steps, elastic solves (stepping and
// observation apart), operator assemblies, preconditioner setups, Krylov
// iterations and seconds.
//
// Sample runs:
//    mpiexec -np 8 ./viscoelastic_love -c case.json -tau 1 -lmax 4
//    mpiexec -np 8 ./viscoelastic_love -c case.json -tau inf,1 -scheme adaptive
//    mpiexec -np 8 ./viscoelastic_love -c case.json -tau 1 -steps 8 -tide
// ============================================================================

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>

#include "benchmark_case.hpp"
#include "viscoelastic_common.hpp"

using namespace mfem;
using namespace AdGIA;
using namespace benchmark;

namespace {

// A comma-separated list of numbers; "inf" is infinity.
std::vector<real_t> ParseList(const std::string& s) {
  std::vector<real_t> out;
  std::size_t pos = 0;
  while (pos < s.size()) {
    std::size_t next = s.find(',', pos);
    if (next == std::string::npos) {
      next = s.size();
    }
    const std::string item = s.substr(pos, next - pos);
    if (!item.empty()) {
      out.push_back(item == "inf" ? std::numeric_limits<real_t>::infinity()
                                  : std::stod(item));
    }
    pos = next + 1;
  }
  return out;
}

using vebench::Counters;

// The Love numbers of one forcing at one time, per degree.
struct Snapshot {
  real_t time = 0.0;
  std::vector<real_t> h, l, k;
  real_t spurious = 0.0;
  int solves = 0;  // cumulative, stepping only
  double seconds = 0.0;
};

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();
  const bool root = Mpi::Root();

  CaseOptions options;
  options.lmax = 4;
  const char* out_file = "viscoelastic.json";
  const char* tau_arg = "1";
  const char* times_arg = "";
  const char* scheme = "sdirk23";
  int lmin = 2;
  int steps = 4;
  real_t dt_max = 0.25;
  real_t rtol = 1e-4, atol = 1e-12;
  bool tide = false;

  OptionsParser args(argc, argv);
  options.Add(args);
  args.AddOption(&out_file, "-out", "--output", "Results file (JSON).");
  args.AddOption(&tau_arg, "-tau", "--maxwell-times",
                 "Maxwell time of each solid layer, centre outward, comma "
                 "separated ('inf': elastic); one value for all.");
  args.AddOption(&times_arg, "-times", "--output-times",
                 "Output times, comma separated, increasing (default: the "
                 "reference's 13, 0.011 * 2^(5k/6) times the smallest "
                 "finite tau, k = 0..12).");
  args.AddOption(&scheme, "-scheme", "--scheme",
                 "sdirk23 (default), exptrap, be, etd1 or adaptive.");
  args.AddOption(&steps, "-steps", "--steps-per-interval",
                 "Fixed-step schemes: least number of equal steps between "
                 "consecutive output times.");
  args.AddOption(&dt_max, "-dt-max", "--max-step",
                 "Fixed-step schemes: largest step, in units of the "
                 "smallest finite tau (0: no cap).");
  args.AddOption(&rtol, "-rtol", "--adaptive-rtol",
                 "Adaptive scheme: relative tolerance.");
  args.AddOption(&atol, "-atol", "--adaptive-atol",
                 "Adaptive scheme: absolute tolerance.");
  args.AddOption(&lmin, "-lmin", "--min-degree", "Lowest harmonic degree.");
  args.AddOption(&tide, "-tide", "--tide", "-no-tide", "--no-tide",
                 "Evolve the tidal forcing as well (a second run).");
  args.Parse();
  if (!args.Good()) {
    if (root) args.PrintUsage(std::cout);
    return 1;
  }
  if (root) args.PrintOptions(std::cout);
  const int lmax = options.lmax;
  MFEM_VERIFY(lmin >= 0 && lmax >= lmin,
              "Degrees must satisfy 0 <= lmin <= lmax.");
  const std::string scheme_name(scheme);
  MFEM_VERIFY(scheme_name == "sdirk23" || scheme_name == "exptrap" ||
                  scheme_name == "be" || scheme_name == "etd1" ||
                  scheme_name == "adaptive",
              "-scheme must be sdirk23, exptrap, be, etd1 or adaptive.");

  // The Maxwell time of every solid layer, by attribute.
  const MeshManifest manifest(options.manifest);
  const Array<int> solids = manifest.SolidAttributes();
  std::vector<real_t> taus = ParseList(tau_arg);
  if (taus.size() == 1) {
    taus.assign(solids.Size(), taus[0]);
  }
  MFEM_VERIFY(static_cast<int>(taus.size()) == solids.Size(),
              "-tau needs one value per solid layer (" << solids.Size()
                                                       << ") or one for all.");
  std::map<int, real_t> tau_of;
  real_t tau_min = std::numeric_limits<real_t>::infinity();
  for (int j = 0; j < solids.Size(); j++) {
    MFEM_VERIFY(taus[j] > 0.0, "Maxwell times must be positive.");
    tau_of[solids[j]] = taus[j];
    tau_min = std::min(tau_min, taus[j]);
  }
  MFEM_VERIFY(std::isfinite(tau_min),
              "Every solid layer is elastic: nothing relaxes.");

  std::vector<real_t> times = ParseList(times_arg);
  if (times.empty()) {
    for (int k = 0; k <= 12; k++) {
      times.push_back(0.011 * tau_min * std::pow(2.0, k * 10.0 / 12.0));
    }
  }
  for (std::size_t j = 0; j < times.size(); j++) {
    MFEM_VERIFY(times[j] > (j ? times[j - 1] : 0.0),
                "Output times must be positive and increasing.");
  }

  // The rheology: per attribute of the displacement mesh a Maxwell body
  // with the layer's tau, or the elastic solid (tau = inf, and the fluid
  // layers of the gauged method, whose mu is zero).
  std::vector<std::unique_ptr<ConstantCoefficient>> tau_c;
  std::vector<std::unique_ptr<IsotropicMaxwellRheology>> maxwell;
  std::unique_ptr<IsotropicElasticRheology> elastic;
  std::unique_ptr<CompositeRheology> composite;
  auto factory = [&](ParSubMesh& sub, Coefficient& kappa,
                     Coefficient& mu) -> const Rheology* {
    const int n = sub.attributes.Max();
    elastic = std::make_unique<IsotropicElasticRheology>(3, kappa, mu);
    std::vector<RheologyRegion> regions;
    for (const int a : sub.attributes) {
      Array<int> marker(n);
      marker = 0;
      marker[a - 1] = 1;
      std::string name = "layer" + std::to_string(a);
      for (const auto& layer : manifest.Layers()) {
        if (layer.attribute == a) name = layer.name;
      }
      const auto it = tau_of.find(a);
      if (it != tau_of.end() && std::isfinite(it->second)) {
        tau_c.push_back(std::make_unique<ConstantCoefficient>(it->second));
        maxwell.push_back(std::make_unique<IsotropicMaxwellRheology>(
            IsotropicMaxwellRheology::Maxwell(3, kappa, mu, *tau_c.back())));
        regions.push_back({marker, maxwell.back().get(), name});
      } else {
        regions.push_back({marker, elastic.get(), name});
      }
    }
    composite = std::make_unique<CompositeRheology>(3, regions);
    return composite.get();
  };

  Case c(options, factory);
  MFEM_VERIFY(c.eulerian, "The viscoelastic driver needs an Eulerian method "
                          "(dahlen or gauged).");
  if (c.method == "dahlen" && c.HasFluid() && lmin == 0) {
    // As love_benchmark -combined: Dahlen's fluid differs from the
    // reference at degree zero by design, and its degree-zero response
    // would leak into the others' through the discrete coupling.
    lmin = 1;
    if (root) {
      std::cout << "Degree zero left out of the load (Dahlen's fluid).\n";
    }
  }
  const auto& basis = c.Basis();
  const real_t g = c.gravity;
  Problem& problem = *c.problem;
  if (root) {
    std::cout << "Layers:";
    for (const auto& layer : manifest.Layers()) {
      if (!layer.in_geometry) continue;
      std::cout << " " << layer.name;
      if (layer.fluid) {
        std::cout << " (fluid)";
      } else {
        std::cout << " (tau " << tau_of[layer.attribute] << ")";
      }
    }
    std::cout << "\nScheme " << scheme_name << ", " << times.size()
              << " output times to " << times.back() << "\n";
  }

  std::vector<int> degrees, forced;
  for (int l = lmin; l <= lmax; l++) {
    degrees.push_back(l);
    forced.push_back(basis.Index(l, 0));
  }

  // The load's own potential, once: time-independent.
  Vector phi_sigma(basis.Size());
  {
    Vector unit(basis.Size());
    unit = 0.0;
    for (const int i : forced) unit[i] = 1.0;
    c.sigma->SetCoefficients(unit);
    // The Eulerian class solves the load potential from its assembled
    // load: assemble it first (love_benchmark does so by solving).
    problem.AssembleForce(0.0);
    ParGridFunction phi_direct(
        static_cast<ParFiniteElementSpace*>(&c.AnalysisPotentialSpace()));
    c.SolveLoadPotential(phi_direct);
    c.Surface().scalar->Coefficients(phi_direct, phi_sigma);
  }

  // The Love numbers of the current solution.
  auto observe = [&](bool load, real_t t) {
    Snapshot s;
    s.time = t;
    Vector cu, cv, cphi;
    const auto& an = c.Surface();
    an.radial->Coefficients(c.Displacement(), cu);
    an.tangential->Coefficients(c.Displacement(), cv);
    an.scalar->Coefficients(c.AnalysisPotential(), cphi);
    for (std::size_t j = 0; j < degrees.size(); j++) {
      const int l = degrees[j], i = forced[j];
      real_t u = cu[i], v = cv[i], phi = cphi[i];
      if (l == 1) {
        // The centre-of-mass frame (love_benchmark).
        const real_t shift = phi / g;
        u += shift;
        v += shift;
        phi -= an.gravity * shift;
      }
      if (load) {
        s.h.push_back(-g * u / phi_sigma[i]);
        s.l.push_back(-g * v / phi_sigma[i]);
        s.k.push_back(phi / phi_sigma[i] - 1.0);
      } else {
        s.h.push_back(-g * u);
        s.l.push_back(-g * v);
        s.k.push_back(phi);
      }
    }
    // The largest unforced surface coefficient relative to the forced
    // ones: the discrete leakage of the combined forcing.
    real_t big = 0.0, small = 0.0;
    for (int i = 0; i < cu.Size(); i++) {
      const bool f = std::find(forced.begin(), forced.end(), i) !=
                     forced.end();
      (f ? big : small) = std::max(f ? big : small, std::abs(cu[i]));
    }
    s.spurious = small / (big + 1e-300);
    return s;
  };

  struct Run {
    Snapshot elastic;
    std::vector<Snapshot> history;
    Counters stepping, observation, total;
    int steps = 0, rejected = 0;
    double seconds = 0.0;
    bool ok = true;
  };

  // One evolution from rest under the forcing (load or tide).
  auto evolve = [&](bool load) {
    Run run;
    Vector unit(basis.Size()), zero(basis.Size());
    unit = 0.0;
    zero = 0.0;
    for (const int i : forced) {
      if (load || basis.Degree(i) >= 2) unit[i] = 1.0;
    }
    c.sigma->SetCoefficients(load ? unit : zero);
    c.psi->SetCoefficients(load ? zero : unit);

    ViscoelasticOperator visco(problem);
    // The shared evolution (viscoelastic_common.hpp): fixed-step schemes
    // take, between consecutive output times, max(-steps, ceil(span / dt))
    // equal steps with dt = -dt-max tau_min (no bound when -dt-max <= 0);
    // the adaptive trapezoid starts from a tenth of the first output time.
    // The load is a Heaviside one, so the history has no breakpoints.
    vebench::StepOptions opt;
    opt.scheme = scheme_name;
    opt.min_steps = std::max(1, steps);
    opt.dt = scheme_name == "adaptive"
                 ? 0.1 * times.front()
                 : (dt_max > 0.0 ? dt_max * tau_min
                                 : std::numeric_limits<real_t>::infinity());
    opt.rtol = rtol;
    opt.atol = atol;
    vebench::LoadHistory constant;
    const vebench::EvolveResult r = vebench::Evolve(
        problem, visco, constant, times, opt,
        [&](const Vector&, real_t t, int k, const vebench::OutputCost& oc) {
          if (k < 0) {
            run.elastic = observe(load, 0.0);
            return;
          }
          Snapshot s = observe(load, t);
          s.solves = oc.cost.solves;
          s.seconds = oc.seconds;
          run.history.push_back(std::move(s));
          if (root) {
            std::cout << std::setw(10) << std::setprecision(4) << t;
            for (std::size_t j = 0; j < degrees.size(); j++) {
              std::cout << std::setw(12) << std::setprecision(6)
                        << run.history.back().h[j];
            }
            std::cout << std::setw(8) << run.history.back().solves
                      << std::setw(9) << std::setprecision(3)
                      << run.history.back().seconds << "\n";
          }
        });
    run.ok = r.ok;
    run.steps = r.steps;
    run.rejected = r.rejected;
    run.seconds = r.seconds;
    run.total = r.total;
    run.observation = r.observation;
    run.stepping = r.stepping;
    return run;
  };

  if (root) {
    std::cout << "\nLoad: h' by degree " << lmin << ".." << lmax
              << "\n         t" << std::string(12 * degrees.size(), ' ')
              << "  solves     time\n";
  }
  const Run load = evolve(true);
  Run tidal;
  const bool with_tide = tide && lmax >= 2;
  if (with_tide) {
    if (root) std::cout << "\nTide: h by degree\n";
    // The load run's operator is gone but its effective modulus is still
    // assembled in the problem, and a fresh operator assumes the
    // unrelaxed one: restore it before starting from rest.
    problem.ClearRelaxationWeights();
    c.ResetSolution();
    tidal = evolve(false);
  }

  if (root) {
    std::cout << std::setprecision(4) << "\nLoad run: " << load.steps
              << " steps" << (load.rejected ? " (+" : "")
              << (load.rejected ? std::to_string(load.rejected) + " rejected)"
                                : "")
              << ", " << load.stepping.solves << " stepping solves + "
              << load.observation.solves << " observation, "
              << load.total.assemblies << " assemblies, "
              << load.total.setups << " preconditioner setups, "
              << load.total.its << " iterations, " << load.seconds << " s"
              << (load.ok ? "" : ", SOME SOLVES FAILED") << "\n";

    auto quantities = [&](std::ostream& os, const Snapshot& s,
                          const Snapshot* tide_s) {
      os << "\"h_load\": " << List(s.h) << ", \"l_load\": " << List(s.l)
         << ", \"k_load\": " << List(s.k);
      if (tide_s) {
        // The tidal numbers exist from degree two: null below.
        std::vector<real_t> h, l, k;
        for (std::size_t j = 0; j < degrees.size(); j++) {
          const bool has = degrees[j] >= 2;
          h.push_back(has ? tide_s->h[j] : NAN);
          l.push_back(has ? tide_s->l[j] : NAN);
          k.push_back(has ? tide_s->k[j] : NAN);
        }
        os << ", \"h_tide\": " << List(h) << ", \"l_tide\": " << List(l)
           << ", \"k_tide\": " << List(k);
      }
    };
    auto cost = [&](const Run& r) {
      std::ostringstream os;
      os << "{\"steps\": " << r.steps << ", \"rejected_steps\": "
         << r.rejected << ", \"stepping_solves\": " << r.stepping.solves
         << ", \"observation_solves\": " << r.observation.solves
         << ", \"assemblies\": " << r.total.assemblies
         << ", \"preconditioner_setups\": " << r.total.setups
         << ", \"iterations\": " << r.total.its
         << ", \"seconds\": " << Num(r.seconds)
         << ", \"converged\": " << (r.ok ? "true" : "false") << "}";
      return os.str();
    };

    std::ofstream os(out_file);
    MFEM_VERIFY(os.good(), "Cannot write " << out_file << ".");
    os << "{\n";
    c.WriteHeader(os);
    os << ",\n  \"layers\": [";
    bool first = true;
    for (const auto& layer : manifest.Layers()) {
      if (!layer.in_geometry) continue;
      os << (first ? "" : ",") << "\n    {\"attribute\": " << layer.attribute
         << ", \"name\": \"" << layer.name << "\", \"fluid\": "
         << (layer.fluid ? "true" : "false") << ", \"tau\": ";
      if (layer.fluid) {
        os << "null";
      } else if (!std::isfinite(tau_of[layer.attribute])) {
        os << "\"inf\"";
      } else {
        os << Num(tau_of[layer.attribute]);
      }
      os << "}";
      first = false;
    }
    os << "],\n  \"scheme\": \"" << scheme_name << "\""
       << ",\n  \"steps_per_interval\": " << steps
       << ",\n  \"dt_max\": " << Num(dt_max * tau_min)
       << ",\n  \"rtol\": " << Num(rtol) << ",\n  \"atol\": " << Num(atol)
       << ",\n  \"tau_min\": " << Num(tau_min)
       << ",\n  \"degree\": " << List(degrees)
       << ",\n  \"times\": " << List(times) << ",\n  \"phi_direct\": [";
    for (std::size_t j = 0; j < degrees.size(); j++) {
      os << (j ? ", " : "") << Num(phi_sigma[forced[j]]);
    }
    os << "],\n  \"elastic\": {";
    quantities(os, load.elastic, with_tide ? &tidal.elastic : nullptr);
    os << "},\n  \"histories\": [";
    for (std::size_t n = 0; n < load.history.size(); n++) {
      const Snapshot& s = load.history[n];
      os << (n ? "," : "") << "\n    {\"time\": " << Num(s.time) << ", ";
      quantities(os, s, with_tide ? &tidal.history[n] : nullptr);
      os << ", \"solves\": " << s.solves << ", \"seconds\": "
         << Num(s.seconds) << ", \"spurious\": " << Num(s.spurious) << "}";
    }
    os << "],\n  \"cost\": {\"load\": " << cost(load);
    if (with_tide) {
      os << ",\n    \"tide\": " << cost(tidal);
    }
    os << "}\n}\n";
    std::cout << "Wrote " << out_file << "\n";
  }
  return 0;
}
