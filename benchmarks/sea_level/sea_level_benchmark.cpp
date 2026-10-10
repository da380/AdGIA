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
// solvers see one problem. Shorelines are frozen (first-order exact)
// unless -mig runs the WP5 Picard loop (rung 3: against pyslfp's
// nonlinear solver, whose ocean-function updates are SHARP where ours
// are smoothed over -shore — a convention difference that lives at the
// shoreline and shrinks with the smoothing on the resolved ladder).
// Rotation is off unless -Omega (WP4). The surface gravity entering the
// ocean weight w = rho_w C0 / g is the case's own (G M / a^2).
// -no-water applies the same melt load WITHOUT the water feedback (a
// plain elastic loading solve) — the control row of the timing sweep,
// quantifying what the sea-level machinery adds to a solve.
//
// Output: the sea-level change as a nodal CSV
// (SeaLevelOperator::WriteSurfaceField) and a JSON of scalars (the
// uniform term, ocean area, eustatic equivalent, ocean mean, sizes,
// iterations, times, migration statistics).
//
// Options (defaults in brackets): the case options of
// benchmark_case.hpp (-c, -o, ...; -method must stay dahlen) plus
//   -rhow / -rhoi   water and ice density, case units [0.2, 0.1834]
//   -ocean-depth    [6.3e-4]      -cont-amp   [1.4e-3]
//   -cont-width     [0.7]         -cap-amp    [4.7e-4]
//   -cap-width      [0.35]        -melt       [0.5]
//   -shore          smoothing width of the ocean fraction, in units of
//                   rho_w*SL [1e-5]
//   -Omega          rotation rate, case units [0: rotation off]
//   -C1 -C2 -C3     principal moments, case units [0, 0, 0]
//   -mig            shoreline migration (Picard on C; off = frozen C0)
//   -mig-tol        migration stop, relative SL1 increment [1e-3]
//   -mig-iters      migration pass cap [12]
//   -feedback       border | monolithic | picard [border]: the bordered
//                   solve by column elimination; the same system as ONE
//                   MINRES with a block preconditioner whose border
//                   block is the preconditioner-probed Schur complement
//                   (no column solves; BlockMINRES solver type); or the
//                   feedback iterated as explicit loads
//                   around the PLAIN elastic operator (the spectral
//                   codes' route: water as the redistributed-ocean
//                   load, rotation through the tidal slot, the uniform
//                   term by mass bookkeeping, omega by the dense
//                   angular-momentum row). The loop gain is the small
//                   physical feedback, so a handful of warm-started
//                   passes should converge — the per-pass iteration
//                   counts are printed to measure exactly that.
//   -picard-tol     picard stop: relative SL1 increment [1e-8]
//   -picard-max     picard pass cap [30]
//   -no-water       melt load without the water feedback (elastic
//                   control; incompatible with -mig)
//   -slcsv          the fingerprint CSV [sea_level.csv]
//   -out            the scalars JSON [sea_level.json]
//   -export-nodes   write the surface node CSV and exit (run.py --earth
//                   samples the state there; the node set depends on
//                   the mesh AND the order)
//   -state -ice -dice   sampled node CSVs (IceHistory stacks, one
//                   column): the field state replaces the analytic one
//                   — SL0, I0 and the ice change dI, with C0 and the
//                   load law built from them exactly as above. run.py
//                   --earth defines the melt once for both sides, so
//                   no geographic mask lives here. Not with -feedback
//                   picard.
//
// Sample run (through run.py normally):
//    mpiexec -np 4 sea_level_benchmark -c <case>/case.json -o 2
// ============================================================================

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>

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
double g_Omega = 0.0;
double g_C1 = 0.0, g_C2 = 0.0, g_C3 = 0.0;

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
double IceChange(const Vector& x) {
  const double hemi = 0.5 * (1.0 + std::tanh(x[0] / g_melt_width));
  return -g_melt * hemi * IceThickness(x);
}

double MeltLoad(const Vector& x) {
  return (1.0 - OceanFraction(x)) * g_rho_i * IceChange(x);
}

// --- The sampled-state mode (-state/-ice/-dice, run.py --earth): the
// same three roles built from node CSVs (IceHistory; the stacks'
// FieldCoefficients evaluate on the body side, their GridFunctions
// serve the surface-side diagnostics). The ice change comes from its
// own CSV, so run.py defines the melt ONCE for both sides — no
// geographic mask lives here.

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

class FieldWeight : public Coefficient {
 public:
  explicit FieldWeight(FieldFlotation& c0) : c0_(c0) {}
  real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    return g_rho_w * c0_.Eval(T, ip) / g_gravity;
  }

 private:
  FieldFlotation& c0_;
};

class FieldMeltLoad : public Coefficient {
 public:
  FieldMeltLoad(FieldFlotation& c0, Coefficient& dice)
      : c0_(c0), dice_(dice) {}
  real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    return g_rho_i * (1.0 - c0_.Eval(T, ip)) * dice_.Eval(T, ip);
  }

 private:
  FieldFlotation& c0_;
  Coefficient& dice_;
};

// The Picard-feedback load: sigma_data + rho_w C0 SL1, with SL1 the
// previous pass's sea-level change (nodal on the body scalar space;
// zero on the first pass, which is then the plain elastic melt solve).
class PicardLoad : public Coefficient {
 public:
  explicit PicardLoad(GridFunction& sl1) : sl1_(sl1) {}
  real_t Eval(ElementTransformation& T,
              const IntegrationPoint& ip) override {
    Vector x;
    T.Transform(ip, x);
    return MeltLoad(x) + g_rho_w * OceanFraction(x) * sl1_.GetValue(T, ip);
  }

 private:
  GridFunction& sl1_;
};

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();

  CaseOptions options;
  const char* slcsv = "sea_level.csv";
  const char* out = "sea_level.json";
  const char* export_nodes = "";
  const char* state_csv = "";
  const char* ice_csv = "";
  const char* dice_csv = "";
  bool migrate = false;
  bool water = true;
  const char* feedback = "border";
  double picard_tol = 1e-8;
  int picard_max = 30;
  double mig_tol = 1e-3;
  double mig_inexact = 0.1;
  double mig_inexact_max = 1e-3;
  bool mig_verbose = false;
  int mig_iters = 12;
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
  args.AddOption(&g_Omega, "-Omega", "--rotation-rate",
                 "Equilibrium rotation rate about e3, case units (0: "
                 "rotational feedback off).");
  args.AddOption(&g_C1, "-C1", "--moment-1", "Principal moment C1.");
  args.AddOption(&g_C2, "-C2", "--moment-2", "Principal moment C2.");
  args.AddOption(&g_C3, "-C3", "--moment-3", "Principal moment C3.");
  args.AddOption(&migrate, "-mig", "--migrate", "-no-mig", "--no-migrate",
                 "Shoreline migration (Picard on the ocean function; off: "
                 "frozen C0, first-order exact).");
  args.AddOption(&mig_tol, "-mig-tol", "--migration-tolerance",
                 "Migration stop: relative surface-L2 SL1 increment.");
  args.AddOption(&mig_iters, "-mig-iters", "--migration-iterations",
                 "Migration pass cap.");
  args.AddOption(&mig_inexact, "-mig-inexact", "--migration-inexact",
                 "Inexact Picard factor (0: every pass at full tolerance).");
  args.AddOption(&mig_inexact_max, "-mig-inexact-max",
                 "--migration-inexact-max",
                 "Loosest per-pass relative tolerance of the inexact "
                 "Picard.");
  args.AddOption(&mig_verbose, "-mig-verbose", "--migration-verbose",
                 "-no-mig-verbose", "--no-migration-verbose",
                 "Per-pass migration trace (tolerances, increments, "
                 "Phi_g, guard events).");
  args.AddOption(&feedback, "-feedback", "--feedback",
                 "border (monolithic) or picard (feedback as explicit "
                 "loads around the plain elastic operator).");
  args.AddOption(&picard_tol, "-picard-tol", "--picard-tolerance",
                 "Picard stop: relative SL1 increment.");
  args.AddOption(&picard_max, "-picard-max", "--picard-maximum",
                 "Picard pass cap.");
  args.AddOption(&water, "-water", "--water-load", "-no-water",
                 "--no-water-load",
                 "Water-load feedback; off applies the same melt load as a "
                 "plain elastic surface load (the timing control).");
  args.AddOption(&slcsv, "-slcsv", "--sea-level-csv",
                 "Output CSV of the fingerprint's nodal values.");
  args.AddOption(&out, "-out", "--output", "Output JSON of the scalars.");
  args.AddOption(&export_nodes, "-export-nodes", "--export-nodes",
                 "Write the surface node CSV to this path and exit (the "
                 "sampling step of run.py --earth).");
  args.AddOption(&state_csv, "-state", "--state-csv",
                 "Sampled initial sea level at the nodes (with -ice and "
                 "-dice: the field state replaces the analytic one).");
  args.AddOption(&ice_csv, "-ice", "--ice-csv",
                 "Sampled initial ice thickness at the nodes.");
  args.AddOption(&dice_csv, "-dice", "--dice-csv",
                 "Sampled ice-thickness change at the nodes.");
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
  MFEM_VERIFY(water || !migrate,
              "sea_level_benchmark: -mig needs the water load (-water)");
  const bool picard = std::string(feedback) == "picard";
  const bool monolithic = std::string(feedback) == "monolithic";
  MFEM_VERIFY(picard || monolithic || std::string(feedback) == "border",
              "-feedback is border, monolithic or picard");
  MFEM_VERIFY(!picard || (water && !migrate),
              "sea_level_benchmark: -feedback picard iterates the water "
              "feedback (no -no-water, no -mig)");
  const bool fields = !std::string(state_csv).empty();
  MFEM_VERIFY(fields == !std::string(ice_csv).empty() &&
                  fields == !std::string(dice_csv).empty(),
              "sea_level_benchmark: -state, -ice and -dice come together");
  MFEM_VERIFY(!(fields && picard),
              "sea_level_benchmark: -feedback picard is wired for the "
              "analytic state only");

  Case c(options);
  g_gravity = c.gravity;
  if (monolithic) {
    // The same bordered system, solved as one MINRES with the probed
    // border Schur preconditioner instead of by column elimination.
    c.problem->SetMonolithicBorder();
  }
  const Array<int>& surface = c.analyses[c.surface].radial->Marker();

  // grad Phi0 = (g/a) x, exact at the surface (only surface values
  // enter the restriction and the migration's flotation criterion).
  const double ga = c.gravity / c.radius;
  VectorFunctionCoefficient grad_phi0(3, [ga](const Vector& x, Vector& v) {
    v = x;
    v *= ga;
  });

  // The load: with -mig the orchestrator owns it (its constructor wires
  // SetWaterLoad with C-dependent coefficients); frozen C0 otherwise;
  // with -no-water the same melt load enters as a plain surface load.
  FunctionCoefficient w(OceanWeight), sigma_data(MeltLoad);
  FunctionCoefficient sl0(InitialSeaLevel), ice(IceThickness);
  FunctionCoefficient dice(IceChange);

  // The surface layer before the solve: the picard loop needs its
  // fields and transfers; the postprocessing uses it either way.
  SeaLevelOperator sea(*c.fes_u, surface);

  // The sampling step of run.py --earth: the node set, then exit (it
  // depends on the mesh AND the order, so run.py exports it per case).
  if (!std::string(export_nodes).empty()) {
    GridFunction zero(&sea.SurfaceSpace());
    zero = 0.0;
    sea.WriteSurfaceField(zero, export_nodes);
    if (Mpi::Root()) {
      std::cout << "wrote " << export_nodes << " (surface nodes)\n";
    }
    return 0;
  }

  // The sampled state: body-side coefficients (the stacks') for the
  // load and the migration, surface GridFunctions for the diagnostics.
  std::unique_ptr<IceHistory> state_stack, ice_stack, dice_stack;
  std::unique_ptr<FieldFlotation> c0_body, c0_surf;
  std::unique_ptr<FieldWeight> w_body;
  std::unique_ptr<FieldMeltLoad> sigma_body, sigma_surf;
  std::unique_ptr<GridFunctionCoefficient> sl0_surf, i0_surf, dice_surf;
  Coefficient* w_p = &w;
  Coefficient* sigma_p = &sigma_data;
  Coefficient* sl0_p = &sl0;
  Coefficient* ice_p = &ice;
  Coefficient* dice_p = &dice;
  if (fields) {
    state_stack = std::make_unique<IceHistory>(sea, state_csv);
    ice_stack = std::make_unique<IceHistory>(sea, ice_csv);
    dice_stack = std::make_unique<IceHistory>(sea, dice_csv);
    c0_body = std::make_unique<FieldFlotation>(
        state_stack->FieldCoefficient(0), ice_stack->FieldCoefficient(0));
    w_body = std::make_unique<FieldWeight>(*c0_body);
    sigma_body = std::make_unique<FieldMeltLoad>(
        *c0_body, dice_stack->FieldCoefficient(0));
    sl0_surf = std::make_unique<GridFunctionCoefficient>(
        const_cast<GridFunction*>(&state_stack->Field(0)));
    i0_surf = std::make_unique<GridFunctionCoefficient>(
        const_cast<GridFunction*>(&ice_stack->Field(0)));
    dice_surf = std::make_unique<GridFunctionCoefficient>(
        const_cast<GridFunction*>(&dice_stack->Field(0)));
    c0_surf = std::make_unique<FieldFlotation>(*sl0_surf, *i0_surf);
    sigma_surf = std::make_unique<FieldMeltLoad>(*c0_surf, *dice_surf);
    w_p = w_body.get();
    sigma_p = sigma_body.get();
    sl0_p = &state_stack->FieldCoefficient(0);
    ice_p = &ice_stack->FieldCoefficient(0);
    dice_p = &dice_stack->FieldCoefficient(0);
  }
  // SL1 state of the picard loop, on the body scalar space (the
  // benchmark body is a SubMesh of the case's parent, so the space is
  // there); the load coefficient reads its current values each pass.
  std::unique_ptr<ParGridFunction> sl1_body;  // benchmarks are MPI-only
  std::unique_ptr<PicardLoad> picard_load;
  if (picard) {
    sl1_body = std::make_unique<ParGridFunction>(
        static_cast<ParFiniteElementSpace*>(sea.BodyScalarSpace()));
    *sl1_body = 0.0;
    picard_load = std::make_unique<PicardLoad>(*sl1_body);
  }

  std::unique_ptr<ShorelineMigration> mig;
  if (migrate) {
    ShorelineMigration::Options opt;
    opt.shore = g_shore;
    opt.tol = mig_tol;
    opt.max_iterations = mig_iters;
    opt.inexact = mig_inexact;
    opt.inexact_max = mig_inexact_max;
    opt.verbose = mig_verbose;
    mig = std::make_unique<ShorelineMigration>(*c.problem, grad_phi0,
                                               *sl0_p, *ice_p, *dice_p,
                                               g_rho_w, g_rho_i,
                                               surface, opt);
  } else if (picard) {
    c.problem->SetSurfaceLoad(*picard_load, surface);
  } else if (water) {
    c.problem->SetWaterLoad(*w_p, *sigma_p, surface);
  } else {
    c.problem->SetSurfaceLoad(*sigma_p, surface);
  }
  CentrifugalPotential psi_total(3, g_Omega);
  Vector moments(3);
  moments[0] = g_C1;
  moments[1] = g_C2;
  moments[2] = g_C3;
  if (g_Omega != 0.0) {
    // The Case wires the Love machinery's tidal expansion; the
    // rotational physics owns that slot here (the border, or the
    // picard loop's updated centrifugal potential).
    c.problem->ClearTidalPotential();
    if (picard) {
      c.problem->SetTidalPotential(psi_total);
    } else {
      c.problem->SetRotation(g_Omega, moments);
    }
  }

  int picard_passes = 0;
  std::vector<int> pass_its;
  Vector omega_p(3);
  omega_p = 0.0;
  real_t picard_uniform = 0.0;
  const auto t0 = Clock::now();
  if (migrate) {
    MFEM_VERIFY(mig->Solve(0.0), "the migrating solve failed");
  } else if (picard) {
    // The feedback as a fixed point around the plain operator: solve,
    // read SL1 (mass-conserving uniform) and omega (the dense
    // angular-momentum row), update the loads, repeat. The loop gain is
    // the physical feedback (percent-grade for water; the wander rows
    // approach neutrality as C3 - C1 -> 0), and each pass warm-starts
    // the solver from the last.
    const double melted = -sea.SurfaceIntegral(sigma_data);
    // The mass bookkeeping must run over the state's own ocean, not
    // the all-ocean default (the uniform term is a constant: a wrong
    // ocean shifts the whole fingerprint).
    sea.SetDensities(g_rho_w, g_rho_i);
    sea.SetInitialState(sl0, ice);
    std::vector<std::unique_ptr<CentrifugalPotential>> psi_unit;
    DenseMatrix Aw(3);
    if (g_Omega != 0.0) {
      Aw = InertiaMatrix(moments);
      for (int k = 0; k < 3; k++) {
        psi_unit.push_back(std::make_unique<CentrifugalPotential>(
            3, g_Omega));
        psi_unit[k]->SetUnit(k);
        for (int j = 0; j < 3; j++) {
          Aw(k, j) += c.problem->TidalTidalCoupling(*psi_unit[k],
                                                    *psi_unit[j]);
        }
      }
    }
    auto body_surface_integral = [&](Coefficient& f) {
      ParLinearForm lf(
          static_cast<ParFiniteElementSpace*>(sea.BodyScalarSpace()));
      lf.AddBoundaryIntegrator(new BoundaryLFIntegrator(f),
                               const_cast<Array<int>&>(surface));
      lf.Assemble();
      double v = lf.Sum();
      double g = 0.0;
      MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      return g;
    };
    auto surf_l2 = [&](const GridFunction& f) {
      GridFunctionCoefficient fc(const_cast<GridFunction*>(&f));
      ProductCoefficient f2(fc, fc);
      return std::sqrt(std::max(sea.SurfaceIntegral(f2), real_t{0}));
    };
    GridFunction prev(&sea.SurfaceSpace());
    prev = 0.0;
    for (int k = 0; k < picard_max; k++) {
      c.problem->AssembleForce(0.0);
      MFEM_VERIFY(c.problem->Solve(), "the picard solve failed");
      pass_its.push_back(c.problem->LastOuterIterations());
      const auto info = sea.SeaLevelChange(
          c.problem->Displacement(), grad_phi0, c.problem->Potential(),
          g_Omega != 0.0 ? &psi_total : nullptr, melted);
      picard_uniform = info.uniform;
      GridFunction d(sea.SeaLevelChangeField());
      d -= prev;
      const real_t rel = surf_l2(d) /
                         (surf_l2(sea.SeaLevelChangeField()) + 1e-300);
      prev = sea.SeaLevelChangeField();
      sea.Extend(sea.SeaLevelChangeField(), *sl1_body);
      if (g_Omega != 0.0) {
        Vector rhs(3);
        for (int j = 0; j < 3; j++) {
          ProductCoefficient spj(*picard_load, *psi_unit[j]);
          rhs[j] = -(c.problem->TidalCoupling(*psi_unit[j]) +
                     body_surface_integral(spj));
        }
        DenseMatrixInverse Ainv(Aw);
        Ainv.Mult(rhs, omega_p);
        psi_total.SetAmplitudes(omega_p);
      }
      picard_passes++;
      if (Mpi::Root()) {
        std::cout << "picard pass " << picard_passes << ": "
                  << pass_its.back() << " outer its, rel increment "
                  << rel << "\n";
      }
      if (k > 0 && rel <= picard_tol) {
        break;
      }
    }
  } else {
    c.problem->AssembleForce(0.0);
    MFEM_VERIFY(c.problem->Solve(), "the coupled solve failed");
  }
  const double solve_s = Seconds(t0);

  // The fingerprint: SL1 = -(u.grad Phi0 + phi + psi)/g + Phi_g/g. The
  // picard loop already left SeaLevelChangeField() at its converged
  // state (mass-conserving uniform in place of the border's Phi_g).
  CentrifugalPotential psi(3, g_Omega);
  if (!picard) {
    if (g_Omega != 0.0) {
      psi.SetAmplitudes(c.problem->AngularVelocity());
    }
    sea.SeaLevelChangeFrom(c.problem->Displacement(), grad_phi0,
                           c.problem->Potential(),
                           c.problem->UniformPotentialTerm(),
                           g_Omega != 0.0 ? &psi : nullptr);
  }
  sea.WriteSurfaceField(sea.SeaLevelChangeField(), slcsv);

  // Scalars, over the same ocean fraction the solve used (surface-side
  // coefficients in the sampled-state mode).
  FunctionCoefficient ocean_ind(OceanFraction);
  FunctionCoefficient melt(MeltLoad);
  Coefficient& ocean_diag = fields ? static_cast<Coefficient&>(*c0_surf)
                                   : ocean_ind;
  Coefficient& melt_diag = fields ? static_cast<Coefficient&>(*sigma_surf)
                                  : melt;
  ConstantCoefficient one(1.0);
  ProductCoefficient area_ind(ocean_diag, one);
  const double area = sea.SurfaceIntegral(area_ind);
  const double melted_mass = -sea.SurfaceIntegral(melt_diag);
  const double eustatic = melted_mass / (g_rho_w * area);
  GridFunctionCoefficient slc(&sea.SeaLevelChangeField());
  ProductCoefficient ocean_sl(ocean_diag, slc);
  const double ocean_mean = sea.SurfaceIntegral(ocean_sl) / area;

  if (Mpi::Root()) {
    std::ofstream os(out);
    os << "{\n";
    os << "  \"gravity\": " << Num(c.gravity) << ",\n";
    os << "  \"radius\": " << Num(c.radius) << ",\n";
    os << "  \"phi_g\": "
       << Num(picard ? picard_uniform * g_gravity
                     : c.problem->UniformPotentialTerm())
       << ",\n";
    os << "  \"ocean_area\": " << Num(area) << ",\n";
    os << "  \"melted_mass\": " << Num(melted_mass) << ",\n";
    os << "  \"eustatic\": " << Num(eustatic) << ",\n";
    os << "  \"ocean_mean\": " << Num(ocean_mean) << ",\n";
    os << "  \"displacement_unknowns\": " << c.displacement_unknowns
       << ",\n";
    os << "  \"Omega\": " << Num(g_Omega) << ",\n";
    os << "  \"omega\": [";
    if (g_Omega != 0.0) {
      const Vector& om = picard ? omega_p : c.problem->AngularVelocity();
      for (int k = 0; k < om.Size(); k++) {
        os << Num(om[k]) << (k + 1 < om.Size() ? ", " : "");
      }
    }
    os << "],\n";
    os << "  \"feedback\": \"" << feedback << "\",\n";
    if (picard) {
      os << "  \"picard_passes\": " << picard_passes << ",\n";
      os << "  \"picard_pass_iterations\": [";
      for (std::size_t k = 0; k < pass_its.size(); k++) {
        os << pass_its[k] << (k + 1 < pass_its.size() ? ", " : "");
      }
      os << "],\n";
    }
    os << "  \"water\": " << (water ? "true" : "false") << ",\n";
    os << "  \"migrate\": " << (migrate ? "true" : "false") << ",\n";
    if (migrate) {
      os << "  \"mig_passes\": " << mig->Iterations() << ",\n";
      os << "  \"mig_last_change\": " << Num(mig->LastShorelineChange())
         << ",\n";
      os << "  \"mig_outer_iterations\": " << mig->TotalOuterIterations()
         << ",\n";
    }
    os << "  \"iterations\": " << c.problem->TotalIterations() << ",\n";
    os << "  \"solve_seconds\": " << Num(solve_s) << "\n";
    os << "}\n";
    std::cout << "sea level: phi_g " << c.problem->UniformPotentialTerm()
              << ", eustatic " << eustatic << ", ocean mean " << ocean_mean
              << ", wrote " << slcsv << " and " << out << "\n";
  }
  return 0;
}
