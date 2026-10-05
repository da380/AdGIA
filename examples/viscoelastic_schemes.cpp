// ============================================================================
// viscoelastic_schemes.cpp
//
// Cost and accuracy of the time integrators of ViscoelasticOperator on one
// problem: a clamped beam (data/beam-quad.mesh: attribute 1 clamped,
// attribute 2 pulled) with a Maxwell or standard-linear-solid rheology and a
// pull varying on the relaxation-time scale, p(t) = p0 (1 + 0.5 sin(2 t /
// tau)) at the defaults (-tp sets the period in units of tau, -ph the
// phase). Every scheme is run from t = 0 to t_final for a list of step sizes
// and compared with a reference solution (RK4 at a very small step, which is
// stable there); the table reports, per run,
//
//   error   relative max-norm error of the internal variables at t_final
//   u-error relative max-norm error of the displacement over the HISTORY:
//           the worst of t_final and three interior checkpoints (the
//           quarters of the run), so that a coarse step cannot alias a fast
//           load and still score well on the final state (the checkpoint
//           solves are not counted as cost; a fixed-step run whose step
//           count is not a multiple of four is judged on t_final alone)
//   solves  elastic solves (the cost unit: one linear system each)
//   asm     operator assemblies (a change of effective modulus)
//   pc      preconditioner setups (the expensive part of an assembly)
//   its     CG iterations accumulated over the run
//   time    wall-clock seconds
//
// Schemes: ETD1 (exponential Euler, first order, one solve per step),
// ExpTrap (exponential trapezoid, second order, one solve per step, exact
// for a strain linear in time), BE (backward Euler through MFEM's
// ODESolver, one solve per step), SDIRK23 (MFEM's L-stable two-stage
// variant, second order, two solves per step), RK4 (explicit, four solves
// per step, only stable for dt < ~2.8 tau_min), and the adaptive exponential
// trapezoid at a few tolerances (its dt column is the number of accepted
// steps). With -gamma > 0 the Maxwell branch carries the power-law
// relaxation of Crawford et al. (2017): the times depend on the stress, the
// trapezoid and implicit schemes run their corrector, and "ExpTrap/conv"
// iterates the corrector to convergence. The explicit schemes' stability
// limit then involves the effective (shorter) times, and a run that blows
// up is reported as unstable.
//
// With -targets (a list of relative errors, measured as u-error above) a
// second table answers the practical question: to reach a given accuracy
// over the run, what does each scheme cost? For every target and scheme the
// step is halved (from one step per tau) until the error is met, and the
// cost at that step is reported; for the adaptive solver the tolerance is
// tightened by factors of 2 from 0.1 until the target is met.
//
// Outputs: the tables on the screen; viscoelastic_schemes.csv, the
// history error of the displacement against the number of solves for every
// scheme and step of the first table (python3 plot_csv.py
// viscoelastic_schemes.csv: the efficiency of each scheme at a glance --
// lower left is better, the slope is the order); -out, the cost-to-target
// table as JSON. No fields are shown: the beam's deformation is that of
// viscoelasticity.cpp.
//
// Sample runs:
//    ./viscoelastic_schemes
//    ./viscoelastic_schemes -o 1 -r 2 -tf 8
//    ./viscoelastic_schemes -mu-inf 0.5
//    ./viscoelastic_schemes -gamma 5
//    ./viscoelastic_schemes -targets 1e-2,1e-3,1e-4
//    ./viscoelastic_schemes -gamma 5 -targets 1e-2,1e-3
//    ./viscoelastic_schemes -tau-ratio 100 -targets 1e-2,1e-3,1e-4
//
// -tau-ratio r adds a second Maxwell branch with the relaxation time tau/r:
// the stiff case, where the explicit schemes are bound to dt < 2.8 tau/r
// while the exponential and implicit ones are not.
//
// What the cost-to-target table prints at the defaults (order 2, one
// refinement, serial build), as elastic solves to reach a u-error of
// 1e-2 / 1e-3 / 1e-4:
//   one relaxation time (-targets 1e-2,1e-3,1e-4): RK4 17/17/33,
//     ExpTrap 14/30/126, SDIRK23 17/33/129, adaptive ExpTrap 24/94/225,
//     BE 29/253/4093, ETD1 254/2046/16382 -- nothing is stiff, so the
//     explicit scheme is cheapest and the first-order schemes are not
//     competitive;
//   two relaxation times, ratio 100 (-tau-ratio 100 -targets ...): RK4
//     1025 for any target (stability), SDIRK23 17/65/257, ExpTrap
//     30/254/510, adaptive ExpTrap 40/112/244, BE 29/509/4093, ETD1
//     510/4094/not reached;
//   power law (-gamma 5 -targets 1e-2,1e-3): SDIRK23 33/129, RK4 65/65,
//     ExpTrap 62/126, adaptive ExpTrap 66/269, BE 253/2004.
// So: on the stiff body the implicit and exponential schemes pay for
// themselves, while RK4 pays for stability; at gamma = 5 the stress
// shortens the times only moderately and RK4 stays competitive at
// moderate accuracy. The exponential trapezoid and SDIRK23 cost about the
// same on the single-branch bodies; on the stiff two-branch body under this
// stress-controlled load SDIRK23 is the cheaper fixed-step scheme, since
// the exponential trapezoid's order drops there. Under this sustained
// periodic pull the adaptive trapezoid at best matches the cheapest
// fixed-step scheme (on the stiff body) and otherwise falls behind it: its
// conservative error estimate keeps its steps small. These are the cases
// of doc/viscoelasticity.md, "Choosing a scheme".
//
// One source serves the serial and the parallel build; the genuine
// differences are the mesh partitioning and the global reductions on the
// error metrics (the state vectors are distributed, so max-norms and
// finiteness checks reduce over ranks — every rank must agree on the
// halving decisions). Run with mpiexec -np N in a parallel build.
// ============================================================================

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

#include "AdGIA.hpp"
#include "visualisation.hpp"

using namespace std;
using namespace mfem;
using namespace AdGIA;

namespace {

#ifdef MFEM_USE_MPI
using MeshType = ParMesh;
using SpaceType = ParFiniteElementSpace;
bool Root() { return Mpi::Root(); }
double GlobalMax(double v) {
  double g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return g;
}
double GlobalMin(double v) {
  double g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  return g;
}
#else
using MeshType = Mesh;
using SpaceType = FiniteElementSpace;
bool Root() { return true; }
double GlobalMax(double v) { return v; }
double GlobalMin(double v) { return v; }
#endif

struct Cost {
  int solves = 0, assemblies = 0, setups = 0;
  long its = 0;
  double seconds = 0.0;
};

struct Counters {
  int solves, assemblies, setups;
  long its;
  explicit Counters(const LinearQuasiStaticProblemBase& p)
      : solves(p.NumSolves()),
        assemblies(p.NumAssemblies()),
        setups(p.NumPreconditionerSetups()),
        its(p.TotalIterations()) {}
};

// Global over ranks: the vectors are distributed, and every rank must
// reach the same verdict.
bool IsFinite(const Vector& v) {
  bool ok = true;
  for (int i = 0; i < v.Size(); i++) {
    if (!std::isfinite(v[i])) {
      ok = false;
      break;
    }
  }
  return GlobalMin(ok ? 1.0 : 0.0) > 0.5;
}

double RelMaxDiff(const Vector& a, const Vector& b) {
  Vector d(a);
  d -= b;
  return GlobalMax(d.Normlinf()) / (GlobalMax(b.Normlinf()) + 1e-300);
}

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif

  const char* mesh_file = "../data/beam-quad.mesh";
  int order = 2;
  int ref_levels = 1;
  real_t tau0 = 1.0;
  real_t mu_inf0 = 0.0;
  real_t gamma0 = 0.0;
  real_t tau_ratio = 1.0;
  real_t t_final = 4.0;
  real_t p0 = 0.05;
  // The default period pi tau gives the pull sin(2 t / tau) of the header.
  real_t load_period = std::numbers::pi_v<real_t>;
  real_t load_phase = 0.0;
  int n_ref = 400;  // reference RK4 steps
  const char* steps_arg = "1,2,4,8,16";
  const char* targets_arg = "";
  const char* out_file = "";
  const char* csv_file = "viscoelastic_schemes.csv";
  int max_steps_per_tau = 4096;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&ref_levels, "-r", "--refinement", "Uniform refinements.");
  args.AddOption(&tau0, "-tau", "--relaxation-time", "Relaxation time.");
  args.AddOption(&mu_inf0, "-mu-inf", "--long-term-modulus",
                 "Long-term shear modulus (0: Maxwell body).");
  args.AddOption(&gamma0, "-gamma", "--power-law-gamma",
                 "Power-law nonlinearity of the relaxation (0: linear).");
  args.AddOption(&tau_ratio, "-tau-ratio", "--relaxation-time-ratio",
                 "> 1: a second branch, half the relaxable modulus, with the "
                 "relaxation time tau / ratio (a stiff problem).");
  args.AddOption(&t_final, "-tf", "--t-final", "Final time (in units of tau).");
  args.AddOption(&p0, "-p0", "--load", "Amplitude of the pull.");
  args.AddOption(&n_ref, "-nref", "--reference-steps",
                 "RK4 steps of the reference solution.");
  args.AddOption(&steps_arg, "-steps", "--steps-per-tau",
                 "Comma-separated list of steps per relaxation time.");
  args.AddOption(&targets_arg, "-targets", "--target-errors",
                 "Comma-separated target relative errors of the "
                 "displacement history (u-error) for the "
                 "cost-to-tolerance table (empty: none).");
  args.AddOption(&max_steps_per_tau, "-kmax", "--max-steps-per-tau",
                 "Give up on a target beyond this many steps per tau.");
  args.AddOption(&load_period, "-tp", "--load-period",
                 "Period of the pull's oscillation, in units of tau: "
                 "small is a load-controlled run, large a "
                 "relaxation-controlled one.");
  args.AddOption(&load_phase, "-ph", "--load-phase",
                 "Phase of the pull (radians): keep the checkpoint "
                 "times off the load's nodes, or a coarse step can "
                 "sample an aliased trajectory at matching phases and "
                 "score well stroboscopically.");
  args.AddOption(&csv_file, "-csv", "--csv",
                 "Error against solves for plot_csv.py (\"\": none).");
  args.AddOption(&out_file, "-out", "--output",
                 "Write the cost-to-target table as JSON (with -targets).");
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

  Mesh smesh(mesh_file, 1, 1);
  const int dim = smesh.Dimension();
  for (int l = 0; l < ref_levels; l++) {
    smesh.UniformRefinement();
  }
#ifdef MFEM_USE_MPI
  MeshType mesh(MPI_COMM_WORLD, smesh);
  smesh.Clear();
#else
  MeshType& mesh = smesh;
#endif
  MFEM_VERIFY(mesh.bdr_attributes.Max() >= 2,
              "The mesh needs boundary attributes 1 (clamped) and 2 (pulled).");
  H1_FECollection fec(order, dim);
  SpaceType fes(&mesh, &fec, dim);

  // Material: kappa such that lambda = mu = 1 in the unrelaxed state.
  const bool stiff = tau_ratio > 1.0;
  ConstantCoefficient kappa(1.0 + 2.0 / dim), mu_inf(mu_inf0),
      mu1((1.0 - mu_inf0) * (stiff ? 0.5 : 1.0)), mu2(0.5 * (1.0 - mu_inf0)),
      tau(tau0), tau2(tau0 / tau_ratio), gamma_c(gamma0), n_c(3.0),
      mu0_c(1.0);
  PowerLawRelaxation law(gamma_c, n_c, mu0_c);
  std::vector<MaxwellBranch> branches{
      {&mu1, &tau, gamma0 > 0.0 ? &law : nullptr}};
  if (stiff) {
    branches.push_back({&mu2, &tau2, gamma0 > 0.0 ? &law : nullptr});
  }
  IsotropicMaxwellRheology rheology(dim, kappa, mu_inf, branches);
  const real_t tau_min = stiff ? tau0 / tau_ratio : tau0;

  // Loads: attribute 1 clamped, a pull on attribute 2 varying on the tau
  // scale.
  const real_t omega =
      2.0 * std::numbers::pi_v<real_t> / (load_period * tau0);
  VectorFunctionCoefficient pull(
      dim, [p0, omega, load_phase](const Vector& /*x*/, real_t t,
                                   Vector& f) {
        f = 0.0;
        f[f.Size() - 1] =
            -p0 * (1.0 + 0.5 * std::sin(omega * t + load_phase));
      });
  Array<int> ess_bdr(mesh.bdr_attributes.Max()), marker(mesh.bdr_attributes.Max());
  ess_bdr = 0;
  ess_bdr[0] = 1;
  marker = 0;
  marker[1] = 1;

  // Collective calls on every rank, root prints.
#ifdef MFEM_USE_MPI
  const auto n_el = mesh.GetGlobalNE();
  const auto n_u = fes.GlobalTrueVSize();
#else
  const auto n_el = mesh.GetNE();
  const auto n_u = fes.GetTrueVSize();
#endif
  if (Root()) {
    cout << "Mesh: " << n_el << " elements, " << n_u
         << " displacement unknowns, rheology "
         << (rheology.IsLinear() ? "linear" : "power law (gamma = ")
         << (rheology.IsLinear() ? "" : std::to_string(gamma0) + ")")
         << "\n";
  }

  // One problem per run, so that the counters and warm starts are
  // clean. The displacement is recorded at the n_check - 1 interior
  // checkpoints as well as t_final: the error is judged against the
  // HISTORY, or a coarse step could alias a fast load and still land the
  // final state.
  // The checkpoint solves are pure output and are taken out of the cost.
  const int n_check = 4;
  auto run = [&](const std::string& name, ODESolver& ode, int n_steps,
                 bool adaptive, real_t rtol, Vector& m_out, Vector& u_out,
                 std::vector<Vector>& checks_out, Cost& cost,
                 int& steps_taken, int corrector = 1) {
    LinearQuasiStaticClampedProblem problem(&fes, rheology, ess_bdr, pull,
                                            marker);
    ViscoelasticOperator visco(problem);
    visco.SetCorrectorIterations(corrector, 1e-3);
    ode.Init(visco);
    Vector m(visco.Height());
    m = 0.0;
    real_t t = 0.0;
    real_t dt = t_final * tau0 / n_steps;
    checks_out.clear();
    bool ok = true;
    auto checkpoint = [&]() {
      ok = visco.SolveElastic(m, t) && ok;
      Vector& u = checks_out.emplace_back();
      problem.Displacement().GetTrueDofs(u);
    };
    const Counters c0(problem);
    const auto w0 = chrono::steady_clock::now();
    if (adaptive) {
      auto* ad = dynamic_cast<AdaptiveExponentialTrapezoidSolver*>(&ode);
      ad->SetTolerances(rtol, 1e-12);
      steps_taken = 0;
      for (int c = 1; c <= n_check; c++) {
        const real_t t_end = t_final * tau0 * c / n_check;
        steps_taken += ad->Integrate(m, t, t_end, dt);
        if (c < n_check) {
          checkpoint();
        }
      }
    } else {
      for (int s = 0; s < n_steps; s++) {
        ode.Step(m, t, dt);
        if (s + 1 < n_steps &&
            static_cast<long>(s + 1) * n_check % n_steps == 0) {
          checkpoint();
        }
      }
      steps_taken = n_steps;
    }
    ok = visco.SolveElastic(m, t) && ok;
    const auto w1 = chrono::steady_clock::now();
    const Counters c1(problem);
    cost.solves = c1.solves - c0.solves -
                  static_cast<int>(checks_out.size());
    cost.assemblies = c1.assemblies - c0.assemblies;
    cost.setups = c1.setups - c0.setups;
    cost.its = c1.its - c0.its;
    cost.seconds = chrono::duration<double>(w1 - w0).count();
    m_out = m;
    problem.Displacement().GetTrueDofs(u_out);
    if (!ok || !IsFinite(m) || !IsFinite(u_out)) {
      m_out.SetSize(1);
      m_out = infinity();
    }
    (void)name;
  };

  // Reference: RK4 at a small step (explicit, so every step costs four
  // solves; stable since dt << tau_min).
  Vector m_ref, u_ref;
  std::vector<Vector> checks_ref;
  {
    RK4Solver rk4;
    Cost c;
    int n;
    if (stiff) {
      n_ref = std::max(n_ref, static_cast<int>(std::round(
                                  100.0 * t_final * tau0 / tau_min)));
    }
    // A fast load must be resolved by the reference as well, and the
    // step count lands on the checkpoints.
    n_ref = std::max(n_ref, static_cast<int>(std::round(
                                100.0 * t_final / load_period)));
    n_ref = ((n_ref + n_check - 1) / n_check) * n_check;
    run("reference", rk4, n_ref, false, 0.0, m_ref, u_ref, checks_ref, c,
        n);
    if (Root()) {
      cout << "Reference: RK4, " << n_ref << " steps, " << c.solves
           << " solves, " << std::fixed << std::setprecision(2) << c.seconds
           << " s\n\n";
    }
  }

  auto parse_list = [](const char* arg) {
    std::vector<double> out;
    std::string s(arg);
    size_t pos = 0;
    while (pos < s.size()) {
      size_t next = s.find(',', pos);
      if (next == std::string::npos) {
        next = s.size();
      }
      if (next > pos) {
        out.push_back(std::stod(s.substr(pos, next - pos)));
      }
      pos = next + 1;
    }
    return out;
  };
  std::vector<int> per_tau;
  for (double v : parse_list(steps_arg)) {
    per_tau.push_back(static_cast<int>(std::round(v)));
  }
  const std::vector<double> targets = parse_list(targets_arg);

  if (Root()) {
    cout << std::left << std::setw(14) << "scheme" << std::right
         << std::setw(8) << "dt/tau" << std::setw(11) << "error"
         << std::setw(11) << "u-error" << std::setw(8) << "solves"
         << std::setw(6) << "asm" << std::setw(5) << "pc" << std::setw(8)
         << "its" << std::setw(9) << "time\n";
  }
  // The displacement error over the HISTORY: the worst of the final
  // state and the common checkpoints.
  auto history_error = [&](const Vector& u,
                           const std::vector<Vector>& checks) {
    double e = RelMaxDiff(u, u_ref);
    if (checks.size() == checks_ref.size()) {
      for (size_t i = 0; i < checks.size(); i++) {
        e = std::max(e, RelMaxDiff(checks[i], checks_ref[i]));
      }
    }
    return e;
  };
  examples::CsvTable curve(csv_file, {"scheme", "solves", "u_error", "error"});
  curve.Meta("title", "Time integrators on the viscoelastic beam")
      .Meta("note", "history error of u against cost; slope = order")
      .Meta("x", "solves")
      .Meta("group", "scheme")
      .Meta("y", "u_error")
      .Meta("xlabel", "elastic solves")
      .Meta("ylabel", "relative u error (history)")
      .Meta("logx", "true")
      .Meta("logy", "true");
  auto report = [&](const std::string& name, const std::string& dt_label,
                    const Vector& m, const Vector& u,
                    const std::vector<Vector>& checks, const Cost& c) {
    // The metrics reduce over ranks; every rank calls them, root prints.
    const bool finite = m.Size() > 1 && IsFinite(m);
    const double e_m = finite ? RelMaxDiff(m, m_ref) : 0.0;
    const double e_u = finite ? history_error(u, checks) : 0.0;
    if (!Root()) {
      return;
    }
    cout << std::left << std::setw(14) << name << std::right << std::setw(8)
         << dt_label;
    if (finite) {
      // The adaptive rows ("Adaptive <rtol>") make one curve.
      curve.Row(name.substr(0, name.find(' ')),
                {static_cast<double>(c.solves), e_u, e_m});
      cout << std::setw(11) << std::scientific << std::setprecision(2) << e_m
           << std::setw(11) << e_u;
    } else {
      cout << std::setw(11) << "unstable" << std::setw(11) << "-";
    }
    cout << std::setw(8) << c.solves << std::setw(6) << c.assemblies
         << std::setw(5) << c.setups << std::setw(8) << c.its << std::setw(9)
         << std::fixed << std::setprecision(2) << c.seconds << "\n";
  };

  struct Scheme {
    std::string name;
    std::function<std::unique_ptr<ODESolver>()> make;
    int corrector;
    bool explicit_scheme;
  };
  std::vector<Scheme> schemes;
  schemes.push_back({"ETD1", [] { return std::make_unique<ExponentialEulerSolver>(); }, 1, false});
  schemes.push_back({"ExpTrap", [] { return std::make_unique<ExponentialTrapezoidSolver>(); }, 1, false});
  if (!rheology.IsLinear()) {
    schemes.push_back({"ExpTrap/conv", [] { return std::make_unique<ExponentialTrapezoidSolver>(); }, 20, false});
  }
  schemes.push_back({"BE", [] { return std::make_unique<BackwardEulerSolver>(); }, 1, false});
  schemes.push_back({"SDIRK23", [] { return std::make_unique<SDIRK23Solver>(2); }, 1, false});
  schemes.push_back({"RK4", [] { return std::make_unique<RK4Solver>(); }, 1, true});

  for (const auto& sc : schemes) {
    for (int k : per_tau) {
      const int n_steps = static_cast<int>(std::round(t_final * k));
      const real_t dt = t_final * tau0 / n_steps;
      if (sc.explicit_scheme && dt > 2.8 * tau_min) {
        if (Root()) {
          cout << std::left << std::setw(14) << sc.name << std::right
               << std::setw(8) << ("1/" + std::to_string(k))
               << "   skipped: dt > 2.8 tau_min (unstable)\n";
        }
        continue;
      }
      auto ode = sc.make();
      Vector m, u;
      std::vector<Vector> checks;
      Cost c;
      int n;
      run(sc.name, *ode, n_steps, false, 0.0, m, u, checks, c, n,
          sc.corrector);
      report(sc.name, "1/" + std::to_string(k), m, u, checks, c);
    }
  }
  for (real_t rtol : {1e-2, 1e-3, 1e-4}) {
    AdaptiveExponentialTrapezoidSolver ode;
    Vector m, u;
    std::vector<Vector> checks;
    Cost c;
    int n;
    run("Adaptive", ode, 4, true, rtol, m, u, checks, c, n);
    std::ostringstream label;
    label << n << " st";
    report("Adaptive " + std::to_string(rtol).substr(0, 6), label.str(), m, u,
           checks, c);
  }
  if (Root()) {
    cout << "\nThe reference is RK4 at dt = tau/" << n_ref / t_final
         << "; the adaptive rows' dt column is the number of accepted steps "
            "at the given rtol.\n";
  }

  // --- Cost to reach a target accuracy ------------------------------------
  if (!targets.empty()) {
    if (Root())
      cout << "\nCost to reach a target relative error of the "
            "displacement history (coarsest step, or loosest rtol, that "
            "meets it):\n"
         << std::left << std::setw(14) << "scheme" << std::right
         << std::setw(9) << "target" << std::setw(9) << "dt/tau"
         << std::setw(11) << "u-error" << std::setw(8) << "solves"
         << std::setw(6) << "asm" << std::setw(5) << "pc" << std::setw(8)
         << "its" << std::setw(9) << "time\n";
    struct TargetRow {
      std::string scheme, dt;
      double target, err;
      Cost c;
    };
    std::vector<TargetRow> table;
    auto row = [&](const std::string& name, double target,
                   const std::string& dt_label, double err, const Cost& c) {
      if (!Root()) {
        return;
      }
      table.push_back({name, dt_label, target, err, c});
      cout << std::left << std::setw(14) << name << std::right << std::setw(9)
           << std::scientific << std::setprecision(0) << target
           << std::setw(9) << dt_label;
      if (err >= 0.0) {
        cout << std::setw(11) << std::scientific << std::setprecision(2)
             << err << std::setw(8) << c.solves << std::setw(6)
             << c.assemblies << std::setw(5) << c.setups << std::setw(8)
             << c.its << std::setw(9) << std::fixed << std::setprecision(2)
             << c.seconds << "\n";
      } else {
        cout << "   not reached within " << max_steps_per_tau
             << " steps per tau\n";
      }
    };
    for (double target : targets) {
      for (const auto& sc : schemes) {
        bool found = false;
        for (int k = 1; k <= max_steps_per_tau; k *= 2) {
          const int n_steps = static_cast<int>(std::round(t_final * k));
          const real_t dt = t_final * tau0 / n_steps;
          if (sc.explicit_scheme && dt > 2.8 * tau_min) {
            continue;
          }
          auto ode = sc.make();
          Vector m, u;
          std::vector<Vector> checks;
          Cost c;
          int n;
          run(sc.name, *ode, n_steps, false, 0.0, m, u, checks, c, n,
              sc.corrector);
          if (m.Size() > 1 && IsFinite(m)) {
            const double err = history_error(u, checks);
            if (err <= target) {
              row(sc.name, target, "1/" + std::to_string(k), err, c);
              found = true;
              break;
            }
          }
        }
        if (!found) {
          row(sc.name, target, "-", -1.0, Cost());
        }
      }
      {
        bool found = false;
        for (double rtol = 0.1; rtol > 1e-8; rtol *= 0.5) {
          AdaptiveExponentialTrapezoidSolver ode;
          Vector m, u;
          std::vector<Vector> checks;
          Cost c;
          int n;
          run("Adaptive", ode, 4, true, rtol, m, u, checks, c, n);
          const double err = history_error(u, checks);
          if (err <= target) {
            std::ostringstream label;
            label << n << " st";
            std::ostringstream name;
            name << "Adaptive " << std::scientific << std::setprecision(0)
                 << rtol;
            row(name.str(), target, label.str(), err, c);
            found = true;
            break;
          }
        }
        if (!found) {
          row("Adaptive", target, "-", -1.0, Cost());
        }
      }
      if (Root()) {
        cout << "\n";
      }
    }
    if (Root() && out_file[0] != '\0') {
      std::ofstream os(out_file);
      os << "{\n  \"tau_ratio\": " << tau_ratio
         << ",\n  \"gamma\": " << gamma0
         << ",\n  \"load_period\": " << load_period
         << ",\n  \"t_final\": " << t_final
         << ",\n  \"reference_steps\": " << n_ref
         << ",\n  \"unknowns\": " << n_u << ",\n  \"rows\": [\n";
      for (size_t i = 0; i < table.size(); i++) {
        const auto& r = table[i];
        os << "    {\"scheme\": \"" << r.scheme << "\", \"target\": "
           << std::scientific << r.target << ", \"dt\": \"" << r.dt
           << "\", \"reached\": " << (r.err >= 0.0 ? "true" : "false")
           << ", \"u_error\": " << (r.err >= 0.0 ? r.err : -1.0)
           << ", \"solves\": " << r.c.solves << ", \"assemblies\": "
           << r.c.assemblies << ", \"setups\": " << r.c.setups
           << ", \"iterations\": " << r.c.its << ", \"seconds\": "
           << std::fixed << std::setprecision(3) << r.c.seconds << "}"
           << (i + 1 < table.size() ? "," : "") << "\n";
      }
      os << "  ]\n}\n";
      cout << "Wrote " << out_file << "\n";
    }
  }
  curve.Write();
  return 0;
}
