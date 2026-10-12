//------------------------------------------------------------------------------
//
// PURPOSE:
//
// This example (viscoelastic_adjoint) extends adjoint_elasticity to
// QUASI-STATIC VISCOELASTICITY: first-order sensitivity kernels of a
// final-time surface observation with respect to the bulk modulus, the
// shear modulus AND the Maxwell viscosity, verified against central
// finite differences of full forward runs. It is non-gravitating and
// first-order only; the theory is the obvious specialisation of Yu,
// Al-Attar, Syvret & Lloyd (2025) with
// gravity, rotation and ocean loading switched off.
//
// It is the tutorial layer over the library's common adjoint structure
// (adjoint.hpp): the forward run RECORDS a
// ForwardTrajectory (the internal-variable state m per step — the
// quasi-static state is m alone, everything else is slaved through one
// solve), the observation is a POINT term of an AdjointForce, and the
// AdjointSolver sweeps the zero-load twin in reversed time — solving the
// impulsive observation problem, jumping the adjoint internal variable
// by (1/tau) times the impulsive strain, stepping between observations
// and feeding every (forward, adjoint) pairing to the IsotropicKernel
// accumulators as it goes (kernels accumulate DURING the sweep; the
// adjoint trajectory is never stored). The library also supports an
// INTERIOR observation time and a CONTINUOUS-in-time observation; this
// example uses a single final-time point term.
//
// The set-up mirrors adjoint_elasticity deliberately. The physical body
// is again the inner layer of the two-layer disc mesh, with kappa, mu
// and the viscosity eta uniform by default (tau = eta / mu); -vd and
// -va add a smooth radial and harmonic-polynomial variation of the
// moduli, and -ba/-bs/-bphi give the VISCOSITY a localised Gaussian
// anomaly of variable amplitude, size and angular position (so tau
// varies), the classic target of a GIA viscosity inversion; the measurement is
// the SAME windowed two-station baseline functional on the surface,
// but now read at the FINAL time t1 — the geodetic analogue: observe
// today, infer the model. The load is a STEP: switched on at t = 0 and
// held, so the run is simple relaxation. It is localised — a
// Gaussian-in-angle radial traction with a trigonometric modulation,
//
//   p(phi) = p0 exp(-(phi - phiL)^2 / sL^2) cos(n (phi - phiL)),
//
// whose oscillation removes the net contributions (the net force
// carries the factor exp(-n^2 sL^2 / 4); a radial traction exerts no
// torque about the centre; the solver's rigid-mode projection absorbs
// the exponentially small remainder).
//
// Forward problem (Maxwell, internal-variable form; m trace-free):
//
//   stress  T = kappa div(u) 1 + 2 mu (dev eps(u) - m),
//   evolution  tau dm/dt + m = dev eps(u),   m(0) = 0,  tau = eta / mu.
//
// The adjoint problem. Because the elastic operator is self-adjoint and
// the Maxwell evolution reverses cleanly, the adjoint fields satisfy
// THE SAME forward equations in reversed time t' = t1 - t with zero
// external load. The final-time measurement enters through the
// quasi-static slaving of u to (m, load): the measurement functional g
// acts as an impulsive load at t' = 0, whose elastic response u+_g
// (one static solve) both terminates the kernels' time integrals and
// JUMPS the adjoint internal variable,
//
//   m+(0+) = dev eps(u+_g) / tau,
//
// after which m+ relaxes freely, driving the adjoint displacement u+.
//
// The kernels (absolute perturbations of kappa, mu, eta; d = dev eps(u),
// all adjoint factors evaluated at the reversed time t1 - t; the u+_g
// terms are the impulsive, purely elastic parts):
//
//   K_kappa = - int_0^t1 div(u) div(u+) dt  -  div(u(t1)) div(u+_g)
//   K_mu    = - int_0^t1 2 (d - m):(d+ - m+) dt - 2 (d - m)(t1) : d+_g
//   K_eta   = - int_0^t1 (2/tau) (d-m):m+ dt
//
// where the eta-kernel follows Yu et al. eq. (61) (there eta = mu tau
// with mu fixed; here (kappa, mu, eta) are the independent parameters,
// so perturbing mu at fixed eta also perturbs tau — the extra term is
// included in K_mu above, and in IsotropicKernelDensity). The relation
// tau m_dot = d - m converts between rate and difference forms.
//
// Checkpointing. The kernels pair the forward trajectory at t with the
// adjoint one at t1 - t. The forward field CANNOT be recovered by
// integrating backwards from t1 — Maxwell relaxation is dissipative and
// the reverse integration is unstable (this is the essential practical
// difference from the seismological adjoint trick). This example records
// the trajectory with the store-all policy (fields memoised); the
// library's re-solve policy (ForwardTrajectory with keep_fields false)
// stores m alone and recovers the fields on demand — the seed of the
// revolve-style schedules — and the two agree up to the solver
// tolerance.
//
// Verification: for perturbation patterns chi (a Gaussian blob and the
// whole body) the predicted change int K m0 chi is compared against the
// central finite difference of full forward re-runs at m0 (1 +/- e chi).
// Unlike the static example, the agreement is not roundoff-limited; two
// discretisation floors remain. The TIME floor (exponential-trapezoid
// stepping and trapezoid quadrature of the kernels, both second order)
// converges away as O(dt^2) with -n. A SPATIAL floor remains in the
// rows that touch the viscous coupling: mu and tau are sampled at the
// internal-variable nodes (and the strain map is inexact on the curved
// elements), while the kernel integrals pair smooth densities — an O(h)
// variational crime that the LOCALISED blob pattern feels most (~1-2%
// for mu at the default resolution, halving per -r refinement; the
// kappa rows and the body-integrated rows sit at 1e-5..1e-7). With -vd
// or -va the floor spreads body-wide — uniform parameters are sampled
// EXACTLY at the nodes, varying ones are not — and every row then
// h-converges under -r. Removing
// it would need the exact discrete adjoint of the nodal sampling, which
// the library does not provide.
//
// Derivative versus gradient, as in adjoint_elasticity — the point is
// the same and matters MORE here, because the raw viscoelastic kernels
// inherit the station and load signatures of both trajectories: the
// accumulated kernel is the L2 REPRESENTATION of the derivative, and a
// sensible gradient for continuous (kappa, mu, eta) is its Riesz
// representative in a space embedding in C0. The example closes by
// applying the library's Riesz maps — L2, and Sobolev metrics of order
// 1 and 2 — to all three kernels' duals, posed on the extension domain
// (body plus buffer) with homogeneous Dirichlet conditions on its far
// boundary; only the H2 representative is mathematically right for
// continuous moduli.
//
// Visualisation (GLVis; -no-vis to skip): the viscosity field eta
// (which shows the -vd/-va variation), and each kernel in the three
// metrics, on the physical sub-domain. The forward and adjoint
// displacements are time-dependent and are not plotted.
//
// One source serves the serial and the parallel build, as throughout.
//
// Mesh requirements. Default ../data/earth_fluidcore_2d.msh,
// generated by meshes/layered_earth.py. Two-dimensional only. Attribute 1
// is the viscoelastic body, a disc of radius kRBody = 3483/6371
// (hard-coded for the model variations); attributes 2 and 3 are the
// Sobolev extension buffer (to r = 1 and r = 1.2) and carry no physics.
// The body's whole boundary is the loaded, observed surface; the mesh's
// outer boundary carries the homogeneous Dirichlet condition of the
// Sobolev metrics. The geometry should be order 2. Lengths are in units
// of the planet radius; (kappa, mu) = (5.4, 2.7), tau = 1 and the
// traction amplitude are nondimensional.
//
// Options (defaults in brackets):
//   -m    mesh file [../data/earth_fluidcore_2d.msh]: the
//         two-layer disc whose inner layer is the body and whose outer
//         layers are the Sobolev buffer.
//   -o    displacement order [2].
//   -r    uniform refinements of the serial mesh [0]: lowers the O(h)
//         spatial floor of the verification (see above).
//   -n    number of time steps [48]: lowers the O(dt^2) time floor.
//   -tf   observation time t1 in units of tau [3]: late enough that
//         the viscous kernels differ visibly from the elastic ones,
//         short of full relaxation.
//   -fd   relative size of the FD perturbations [1e-3].
//   -sl   smoothing length sqrt(beta/alpha) of the Sobolev metrics
//         [0.1], as in adjoint_elasticity.
//   -vd   radial variation of kappa, mu and eta [0]: a term
//         vd (1 - (r/R)^2), smooth at the centre. Nonzero variation
//         activates the nodal sampling floor body-wide (see above).
//   -va   angular variation of kappa and mu [0]: a term
//         va r^2 cos(2 phi) / R^2; a harmonic polynomial, smooth at
//         the centre. Keep |vd| + |va| < 1.
//   -ba   relative amplitude of a localised viscosity anomaly [0]:
//         eta0 (1 + ba exp(-|x - xc|^2 / bs^2)), tau varying with it.
//   -bs   size (Gaussian width) of the viscosity blob [0.15].
//   -bphi angular position of the blob, centred at mid-radius [2].
//   -vis / -no-vis   GLVis windows on or off [on].
//
// Sample runs:  ./viscoelastic_adjoint
//               ./viscoelastic_adjoint -n 96
//               mpiexec -np 4 ./viscoelastic_adjoint -r 1  (parallel)
//
//------------------------------------------------------------------------------

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

#include "AdGIA.hpp"
#include "mfem.hpp"
#include "visualisation.hpp"

using namespace mfem;
using namespace AdGIA;

#ifdef MFEM_USE_MPI
using SubMeshType = ParSubMesh;
using SpaceType = ParFiniteElementSpace;
using FieldType = ParGridFunction;
using LinearFormType = ParLinearForm;
#else
using SubMeshType = SubMesh;
using SpaceType = FiniteElementSpace;
using FieldType = GridFunction;
using LinearFormType = LinearForm;
#endif

namespace {

constexpr real_t kPi = std::numbers::pi_v<real_t>;

bool Root() {
#ifdef MFEM_USE_MPI
  return Mpi::Root();
#else
  return true;
#endif
}

real_t GlobalSum(real_t local) {
#ifdef MFEM_USE_MPI
  real_t global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                MPI_COMM_WORLD);
  return global;
#else
  return local;
#endif
}

// The inner layer of the two-layer disc (attribute 1, radius kRBody) is
// the physical body.
constexpr real_t kRBody = 3483.0 / 6371.0;
constexpr real_t kKappa0 = 5.4, kMu0 = 2.7, kTau0 = 1.0;

// The stations (as in adjoint_elasticity): a short baseline centred at
// the angle kPhi0 with separation kDPhi, observed through angular
// windows of width kSigma.
constexpr real_t kPhi0 = 0.3;
constexpr real_t kDPhi = 0.15;
constexpr real_t kSigma = 0.05;

// The load: localised at kPhiL, Gaussian width kSigmaL, modulation
// wavenumber kNMod (kNMod * kSigmaL = 4 suppresses the net force by
// exp(-4)); stepped on at t = 0 and held.
constexpr real_t kPhiL = 0.5 * kPi + 0.2;
constexpr real_t kSigmaL = 0.2;
constexpr real_t kNMod = 20.0;
constexpr real_t kP0 = 0.05;

void Traction(const Vector& x, Vector& t) {
  const real_t r = x.Norml2();
  real_t dphi = std::atan2(x[1], x[0]) - kPhiL;
  while (dphi > kPi) dphi -= 2.0 * kPi;
  while (dphi < -kPi) dphi += 2.0 * kPi;
  const real_t p = -kP0 * std::exp(-dphi * dphi / (kSigmaL * kSigmaL)) *
                   std::cos(kNMod * dphi);
  t.SetSize(2);
  t[0] = p * x[0] / r;
  t[1] = p * x[1] / r;
}

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif

  const char* mesh_file = "../data/earth_fluidcore_2d.msh";
  int order = 2;
  int refinement = 0;
  int n_steps = 48;
  real_t t_final = 3.0;
  real_t fd_eps = 1e-3;
  real_t smoothing = 0.1;
  real_t var_depth = 0.0;
  real_t var_angle = 0.0;
  real_t blob_amp = 0.0;
  real_t blob_size = 0.15;
  real_t blob_angle = 2.0;
  bool visualization = true;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Two-layer disc mesh.");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&refinement, "-r", "--refinement",
                 "Uniform refinements of the serial mesh.");
  args.AddOption(&n_steps, "-n", "--n-steps", "Number of time steps.");
  args.AddOption(&t_final, "-tf", "--t-final",
                 "Observation time t1 (in units of tau).");
  args.AddOption(&fd_eps, "-fd", "--fd-epsilon",
                 "Relative size of the finite-difference perturbations.");
  args.AddOption(&smoothing, "-sl", "--smoothing-length",
                 "Smoothing length sqrt(beta/alpha) of the Sobolev "
                 "metrics.");
  args.AddOption(&var_depth, "-vd", "--variation-depth",
                 "Radial variation of the base parameters: a term "
                 "vd (1 - (r/R)^2).");
  args.AddOption(&var_angle, "-va", "--variation-angle",
                 "Angular variation of kappa and mu: a term "
                 "va r^2 cos(2 phi) / R^2 (smooth at the centre).");
  args.AddOption(&blob_amp, "-ba", "--blob-amplitude",
                 "Relative amplitude of a localised VISCOSITY anomaly: "
                 "eta0 (1 + ba exp(-|x - xc|^2 / bs^2)) with xc at "
                 "mid-radius; tau varies with it.");
  args.AddOption(&blob_size, "-bs", "--blob-size",
                 "Size (Gaussian width) of the viscosity blob.");
  args.AddOption(&blob_angle, "-bphi", "--blob-angle",
                 "Angular position of the viscosity blob (radians).");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization", "Send the fields to GLVis.");
  args.Parse();
  if (!args.Good()) {
    if (Root()) args.PrintUsage(std::cout);
    return 1;
  }
  if (Root()) args.PrintOptions(std::cout);

  auto serial_mesh = Mesh(mesh_file, 1, 1);
  for (int l = 0; l < refinement; l++) {
    serial_mesh.UniformRefinement();
  }
#ifdef MFEM_USE_MPI
  auto parent = ParMesh(MPI_COMM_WORLD, serial_mesh);
  serial_mesh.Clear();
#else
  Mesh& parent = serial_mesh;
#endif
  Array<int> body_attrs({1});
  auto body = SubMeshType::CreateFromDomain(parent, body_attrs);
  const int dim = body.Dimension();
  MFEM_VERIFY(dim == 2, "This example is two-dimensional.");
  auto fec = H1_FECollection(order, dim);
  auto fes = SpaceType(&body, &fec, dim);

  auto surface = Array<int>(body.bdr_attributes.Max());
  surface = 1;
  auto traction = VectorFunctionCoefficient(dim, Traction);

  // The measurement functional (as in adjoint_elasticity): the windowed
  // two-station baseline, read at the final time.
  auto h = VectorFunctionCoefficient(dim, [](const Vector& x, Vector& v) {
    const real_t phi = std::atan2(x[1], x[0]);
    auto window = [phi](real_t centre) {
      real_t d = phi - centre;
      while (d > kPi) d -= 2.0 * kPi;
      while (d < -kPi) d += 2.0 * kPi;
      return std::exp(-d * d / (kSigma * kSigma)) /
             (kSigma * kRBody * std::sqrt(kPi));
    };
    const real_t w = window(kPhi0 + 0.5 * kDPhi) - window(kPhi0 - 0.5 * kDPhi);
    v.SetSize(2);
    v[0] = -w * std::sin(kPhi0);
    v[1] = w * std::cos(kPhi0);
  });
  auto g = LinearFormType(&fes);
  g.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(h), surface);
  g.Assemble();

  const real_t dt = t_final / n_steps;

  // One full forward run for a given (kappa, mu, eta): build the Maxwell
  // problem (tau = eta / mu), step to t1, return J = g . u(t1). The FD
  // re-runs below need nothing else; the base run keeps its operators
  // alive and records a ForwardTrajectory instead (see there).
  auto run_forward = [&](Coefficient& kappa, Coefficient& mu,
                         Coefficient& eta) {
    auto tau = RatioCoefficient(eta, mu);
    auto rheology = IsotropicMaxwellRheology::Maxwell(dim, kappa, mu, tau);
    auto prob =
        LinearQuasiStaticTractionProblem(&fes, rheology, traction, surface);
    auto visco = ViscoelasticOperator(prob);
    ExponentialTrapezoidSolver ode;
    ode.Init(visco);
    Vector m(visco.Height());
    m = 0.0;
    real_t t = 0.0, dt_step = dt;
    MFEM_VERIFY(visco.SolveElastic(m, t), "forward solve failed");
    for (int n = 1; n <= n_steps; n++) {
      ode.Step(m, t, dt_step);
    }
    MFEM_VERIFY(visco.SolveElastic(m, t), "forward solve failed");
    return GlobalSum(g * prob.Displacement());
  };

  // === The forward run at the base model ===
  // The moduli: uniform, optionally modulated by a smooth two-knob
  // parameterisation (zero by default) — a radial even power and the
  // degree-2 harmonic polynomial r^2 cos(2 phi) = x^2 - y^2, both
  // well-defined at r = 0 (naive r- and phi-based variations are
  // not). Keep |vd| + |va| < 1. The viscosity: the same radial term,
  // plus a LOCALISED anomaly — a Gaussian blob of relative amplitude
  // -ba, width -bs, centred at mid-radius at the angle -bphi — the
  // classic target of a GIA viscosity inversion, and what the eta
  // kernels are for; tau = eta / mu varies with all of it.
  auto modulate = [var_depth, var_angle](real_t base) {
    return FunctionCoefficient([=](const Vector& x) {
      const real_t R2 = kRBody * kRBody;
      const real_t q2 = (x[0] * x[0] + x[1] * x[1]) / R2;
      const real_t c2 = (x[0] * x[0] - x[1] * x[1]) / R2;
      return base * (1.0 + var_depth * (1.0 - q2) + var_angle * c2);
    });
  };
  auto kappa0 = modulate(kKappa0);
  auto mu0 = modulate(kMu0);
  auto eta0 = FunctionCoefficient(
      [var_depth, blob_amp, blob_size, blob_angle](const Vector& x) {
        const real_t R2 = kRBody * kRBody;
        const real_t q2 = (x[0] * x[0] + x[1] * x[1]) / R2;
        const real_t cx = 0.5 * kRBody * std::cos(blob_angle);
        const real_t cy = 0.5 * kRBody * std::sin(blob_angle);
        const real_t dx = x[0] - cx, dy = x[1] - cy;
        const real_t blob =
            std::exp(-(dx * dx + dy * dy) / (blob_size * blob_size));
        return kMu0 * kTau0 * (1.0 + var_depth * (1.0 - q2) + blob_amp * blob);
      });

  // The base run on persistent operators, recording the trajectory
  // (store-all policy: the displacements are memoised at Record; the
  // library's keep_fields = false policy would store m alone and
  // re-solve on demand).
  auto tau0 = RatioCoefficient(eta0, mu0);
  auto rheology0 = IsotropicMaxwellRheology::Maxwell(dim, kappa0, mu0, tau0);
  auto prob_fwd =
      LinearQuasiStaticTractionProblem(&fes, rheology0, traction, surface);
  auto visco_fwd = ViscoelasticOperator(prob_fwd);
  auto trajectory = ForwardTrajectory(visco_fwd, true);
  real_t J0 = 0.0;
  {
    ExponentialTrapezoidSolver ode;
    ode.Init(visco_fwd);
    Vector m(visco_fwd.Height());
    m = 0.0;
    real_t t = 0.0, dt_step = dt;
    trajectory.Record(m, t);
    for (int n = 1; n <= n_steps; n++) {
      ode.Step(m, t, dt_step);
      trajectory.Record(m, t);
    }
    // Record() performs the slaving solve (SolveElastic) for the state it
    // stores, so the forward problem's Displacement() is current here.
    J0 = GlobalSum(g * prob_fwd.Displacement());
  }

  // === The adjoint run, through the common structure ===
  // The zero-load twin wrapped in its own operator; the final-time
  // measurement is one POINT term, triggering the impulsive solve and
  // the m+ jump at the reversed start; the kernels accumulate during the
  // sweep with the trapezoid weights and the impulsive terms.
  auto zero_vec = Vector(dim);
  zero_vec = 0.0;
  auto zero_traction = VectorConstantCoefficient(zero_vec);
  auto prob_adj =
      LinearQuasiStaticTractionProblem(&fes, rheology0, zero_traction,
                                       surface);
  auto visco_adj = ViscoelasticOperator(prob_adj);

  AdjointForce force;
  force.AddObservation(trajectory.Time(n_steps), g);

  auto sfes = SpaceType(&body, &fec);
  auto kfec = L2_FECollection(order - 1, dim);
  auto kfes = SpaceType(&body, &kfec);
  auto K_kappa = IsotropicKernel(IsotropicKernel::Parameter::Kappa, sfes);
  auto K_mu = IsotropicKernel(IsotropicKernel::Parameter::Mu, sfes);
  auto K_eta = IsotropicKernel(IsotropicKernel::Parameter::Eta, sfes, &tau0);
  K_kappa.SetField(kfes);
  K_mu.SetField(kfes);
  K_eta.SetField(kfes);

  // Verification directions: int K m0 chi dV for a Gaussian blob and the
  // whole body.
  auto chi_blob = FunctionCoefficient([](const Vector& x) {
    const real_t dx = x[0] - 0.25, dy = x[1] - 0.1;
    return std::exp(-(dx * dx + dy * dy) / 0.01);
  });
  auto chi_all = ConstantCoefficient(1.0);
  Coefficient* m0_of[] = {&kappa0, &mu0, &eta0};
  Coefficient* chi_of[] = {&chi_blob, &chi_all};
  IsotropicKernel* K_of[] = {&K_kappa, &K_mu, &K_eta};
  std::vector<std::unique_ptr<ProductCoefficient>> directions;
  for (int p = 0; p < 3; p++) {
    for (int c = 0; c < 2; c++) {
      directions.push_back(
          std::make_unique<ProductCoefficient>(*m0_of[p], *chi_of[c]));
      K_of[p]->AddDirection(*directions.back());
    }
  }

  auto solver = AdjointSolver(visco_adj, trajectory);
  solver.SetForce(force);
  solver.AddKernel(K_kappa);
  solver.AddKernel(K_mu);
  solver.AddKernel(K_eta);
  MFEM_VERIFY(solver.Run(), "adjoint sweep failed");

  // === The verification table ===
  struct Case {
    const char* name;
    int param;  // 0 kappa, 1 mu, 2 eta
    int direction;
    Coefficient* chi;
  };
  Case cases[] = {{"kappa, blob", 0, 0, &chi_blob},
                  {"kappa, all", 0, 1, &chi_all},
                  {"mu, blob", 1, 0, &chi_blob},
                  {"mu, all", 1, 1, &chi_all},
                  {"eta, blob", 2, 0, &chi_blob},
                  {"eta, all", 2, 1, &chi_all}};

  if (Root()) {
    std::cout << "\nJ at the base model (station baseline at t1 = " << t_final
              << "): " << J0 << "\n\n"
              << "  perturbation        kernel dJ        central FD dJ"
              << "      rel. diff\n";
  }
  for (const auto& c : cases) {
    auto bump = [&](Coefficient& base, bool on, real_t s) {
      auto one_plus =
          std::make_unique<SumCoefficient>(1.0, *c.chi, 1.0, on ? s : 0.0);
      auto prod = std::make_unique<ProductCoefficient>(base, *one_plus);
      return std::pair(std::move(one_plus), std::move(prod));
    };
    const real_t e = fd_eps;
    real_t dJ_fd = 0.0;
    for (const real_t s : {e, -e}) {
      auto [o1, kp] = bump(kappa0, c.param == 0, s);
      auto [o2, mp] = bump(mu0, c.param == 1, s);
      auto [o3, ep] = bump(eta0, c.param == 2, s);
      const real_t J = run_forward(*kp, *mp, *ep);
      dJ_fd += (s > 0 ? J : -J);
    }
    dJ_fd /= 2.0 * e;
    const real_t dJ_kernel = K_of[c.param]->Directional(c.direction);
    if (Root()) {
      std::cout << "  " << std::left << std::setw(16) << c.name << std::right
                << std::scientific << std::setprecision(6) << std::setw(17)
                << dJ_kernel << std::setw(19) << dJ_fd << std::setprecision(1)
                << std::setw(15) << std::abs(dJ_kernel / dJ_fd - 1.0)
                << std::defaultfloat << std::setprecision(6) << "\n";
    }
  }
  if (Root()) {
    std::cout << "\n(Two floors: the time discretisation, O(dt^2) in -n, "
                 "and the nodal\nsampling of mu and tau at the "
                 "internal-variable nodes, O(h) in -r — the\nlocalised "
                 "blob rows feel the latter most; kappa and body-wide "
                 "rows sit at\n1e-5..1e-7. See the header.)\n\n";
  }

  // === Derivative into gradient: the Riesz maps on the buffered mesh ===
  // As in adjoint_elasticity: the duals assembled from the ACCUMULATED
  // kernel fields (transferred to the parent mesh and integrated against
  // the scalar H1 basis over the body attribute; zero in the buffer).
  auto pfes_s = SpaceType(&parent, &fec);
  auto pkfes = SpaceType(&parent, &kfec);
  auto body_marker = Array<int>(parent.attributes.Max());
  body_marker = 0;
  body_marker[0] = 1;
  auto dual = [&](const GridFunction& K) {
    auto K_parent = FieldType(&pkfes);
    K_parent = 0.0;
    SubMeshType::Transfer(static_cast<const FieldType&>(K), K_parent);
    auto K_coeff = GridFunctionCoefficient(&K_parent);
    auto lf = LinearFormType(&pfes_s);
    lf.AddDomainIntegrator(new DomainLFIntegrator(K_coeff), body_marker);
    lf.Assemble();
    Vector j(pfes_s.GetTrueVSize());
#ifdef MFEM_USE_MPI
    lf.ParallelAssemble(j);
#else
    j = lf;
#endif
    return j;
  };
  const Vector j_kappa = dual(K_kappa.Field());
  const Vector j_mu = dual(K_mu.Field());
  const Vector j_eta = dual(K_eta.Field());

  auto outer = ExternalBoundaryMarker(&parent);
  const real_t beta = smoothing * smoothing;
  auto riesz_l2 = L2RieszMap(pfes_s);
  auto riesz_h1 = SobolevRieszMap(pfes_s, 1.0, beta, 1, &outer);
  auto riesz_h2 = SobolevRieszMap(pfes_s, 1.0, beta, 2, &outer);

  auto gradient_on_body = [&](RieszMap& map, const Vector& j,
                              const char* name) {
    Vector g_true(pfes_s.GetTrueVSize());
    map.Mult(j, g_true);
    auto on_parent = FieldType(&pfes_s);
    on_parent.SetFromTrueDofs(g_true);
    auto on_body = FieldType(&sfes);
    SubMeshType::Transfer(on_parent, on_body);
    // Pair is a global reduction: every rank calls it, the root prints.
    const real_t slope = map.Pair(j, g_true);
    if (Root()) {
      std::cout << "  " << name << ": slope <j, g> = " << slope << "\n";
    }
    return on_body;
  };
  if (Root()) {
    std::cout << "Derivatives into gradients (smoothing length " << smoothing
              << "):\n";
  }
  auto gk_l2 = gradient_on_body(riesz_l2, j_kappa, "kappa, L2");
  auto gk_h1 = gradient_on_body(riesz_h1, j_kappa, "kappa, H1");
  auto gk_h2 = gradient_on_body(riesz_h2, j_kappa, "kappa, H2");
  auto gm_l2 = gradient_on_body(riesz_l2, j_mu, "mu,    L2");
  auto gm_h1 = gradient_on_body(riesz_h1, j_mu, "mu,    H1");
  auto gm_h2 = gradient_on_body(riesz_h2, j_mu, "mu,    H2");
  auto ge_l2 = gradient_on_body(riesz_l2, j_eta, "eta,   L2");
  auto ge_h1 = gradient_on_body(riesz_h1, j_eta, "eta,   H1");
  auto ge_h2 = gradient_on_body(riesz_h2, j_eta, "eta,   H2");
  if (Root()) {
    std::cout << "(each slope is positive — every representative is an "
                 "ascent direction of J\nin its own metric; only the H2 "
                 "one is a function for continuous moduli; the\nplots "
                 "show the physical sub-domain.)\n\n";
  }

  // === Visualisation ===
  if (visualization) {
    const std::string keys = examples::DefaultKeys(dim);
    auto eta_field = FieldType(&sfes);
    eta_field.ProjectCoefficient(eta0);
    examples::GLVisWindow("viscosity eta", keys).Send(body, eta_field);
    examples::GLVisWindow("kappa-sensitivity: L2 (the raw kernel)", keys)
        .Send(body, gk_l2);
    examples::GLVisWindow("kappa-sensitivity: H1 gradient", keys)
        .Send(body, gk_h1);
    examples::GLVisWindow("kappa-sensitivity: H2 gradient", keys)
        .Send(body, gk_h2);
    examples::GLVisWindow("mu-sensitivity: L2 (the raw kernel)", keys)
        .Send(body, gm_l2);
    examples::GLVisWindow("mu-sensitivity: H1 gradient", keys)
        .Send(body, gm_h1);
    examples::GLVisWindow("mu-sensitivity: H2 gradient", keys)
        .Send(body, gm_h2);
    examples::GLVisWindow("eta-sensitivity: L2 (the raw kernel)", keys)
        .Send(body, ge_l2);
    examples::GLVisWindow("eta-sensitivity: H1 gradient", keys)
        .Send(body, ge_h1);
    examples::GLVisWindow("eta-sensitivity: H2 gradient", keys)
        .Send(body, ge_h2);
  }
  return 0;
}
