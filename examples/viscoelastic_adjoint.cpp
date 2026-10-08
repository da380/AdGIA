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
// Al-Attar, Syvret & Lloyd (2025, doc/Elasticity/ggae388.pdf) with
// gravity, rotation and ocean loading switched off.
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
//   K_eta   = - int_0^t1 (2 mu / eta^2) ... = - int (2/tau) (d-m):m+ dt
//
// where the eta-kernel follows Yu et al. eq. (61) (there eta = mu tau
// with mu fixed; here (kappa, mu, eta) are the independent parameters,
// so perturbing mu at fixed eta also perturbs tau — the extra term is
// included in K_mu above). The relation tau m_dot = d - m converts
// between rate and difference forms.
//
// Checkpointing. The kernels pair the forward trajectory at t with the
// adjoint one at t1 - t. The forward field CANNOT be recovered by
// integrating backwards from t1 — Maxwell relaxation is dissipative and
// the reverse integration is unstable (this is the essential practical
// difference from the seismological adjoint trick). This example simply
// STORES the whole forward and adjoint trajectories at the step grid
// (store-all checkpointing); for large problems the same pairing is
// done from windowed or binomial checkpoints with partial re-runs.
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
// h-converges under -r. Chasing
// it further would mean the exact discrete adjoint of the nodal
// sampling — library work for later, not this warm-up.
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
// Options (defaults in brackets):
//   -m    mesh file [../data/elastogravity_two_layer_2d.msh]: the
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
#include <vector>

#include "AdGIA.hpp"
#include "mfem.hpp"
#include "visualisation.hpp"

using namespace mfem;
using namespace AdGIA;

#ifdef MFEM_USE_MPI
using MeshType = ParMesh;
using SubMeshType = ParSubMesh;
using SpaceType = ParFiniteElementSpace;
using FieldType = ParGridFunction;
using LinearFormType = ParLinearForm;
#else
using MeshType = Mesh;
using SubMeshType = SubMesh;
using SpaceType = FiniteElementSpace;
using FieldType = GridFunction;
using LinearFormType = LinearForm;
#endif

namespace {

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
constexpr real_t kPhiL = 0.5 * M_PI + 0.2;
constexpr real_t kSigmaL = 0.2;
constexpr real_t kNMod = 20.0;
constexpr real_t kP0 = 0.05;

void Traction(const Vector& x, Vector& t) {
  const real_t r = x.Norml2();
  real_t dphi = std::atan2(x[1], x[0]) - kPhiL;
  while (dphi > M_PI) dphi -= 2.0 * M_PI;
  while (dphi < -M_PI) dphi += 2.0 * M_PI;
  const real_t p = -kP0 * std::exp(-dphi * dphi / (kSigmaL * kSigmaL)) *
                   std::cos(kNMod * dphi);
  t.SetSize(2);
  t[0] = p * x[0] / r;
  t[1] = p * x[1] / r;
}

// The 2-D trace-free internal-variable basis is E0 = diag(1,-1), E1 =
// offdiag(1,1), with Frobenius metric 2 I: for component fields a, b,
// A : B = 2 (a0 b0 + a1 b1).
real_t TensorDot(const Vector& a, const Vector& b) {
  return 2.0 * (a[0] * b[0] + a[1] * b[1]);
}

// One pairing step's kernel density: forward (u, d, m) at t against
// adjoint (udag, ddag, mdag) at t1 - t. The tensor factors are the
// internal-variable component GridFunctions; the divergences come from
// the displacement gradients.
class DensityCoefficient : public Coefficient {
 public:
  enum class Mode { Kappa, Mu, Eta };

  DensityCoefficient(Mode mode, Coefficient& tau)
      : mode_(mode), tau_(&tau) {}

  void SetPair(const GridFunction& u, const GridFunction& d,
               const GridFunction& m, const GridFunction& udag,
               const GridFunction& ddag, const GridFunction& mdag) {
    u_ = &u, d_ = &d, m_ = &m;
    udag_ = &udag, ddag_ = &ddag, mdag_ = &mdag;
  }

  real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    T.SetIntPoint(&ip);
    if (mode_ == Mode::Kappa) {
      u_->GetVectorGradient(T, Du_);
      udag_->GetVectorGradient(T, Dv_);
      real_t divu = 0.0, divv = 0.0;
      for (int i = 0; i < Du_.Height(); i++) {
        divu += Du_(i, i);
        divv += Dv_(i, i);
      }
      return -divu * divv;
    }
    d_->GetVectorValue(T, ip, dv_);
    m_->GetVectorValue(T, ip, mv_);
    dv_ -= mv_;  // d - m (the projected strain: the evolution's source)
    if (mode_ == Mode::Eta) {
      mdag_->GetVectorValue(T, ip, bv_);
      return -(2.0 / tau_->Eval(T, ip)) * TensorDot(dv_, bv_);
    }
    // K_mu = -2 [ dev eps(u) : dev eps(udag) - m : dev eps(udag)
    //             - (d - m) : mdag ],
    // with the dev eps terms POINTWISE from the displacement gradients
    // (that is how mu enters the discrete elastic form; using the
    // projected strains there leaves an h-dependent floor).
    mdag_->GetVectorValue(T, ip, bv_);
    real_t val = TensorDot(dv_, bv_);  // (d - m) : mdag
    u_->GetVectorGradient(T, Du_);
    udag_->GetVectorGradient(T, Dv_);
    const int d = Du_.Height();
    real_t divu = 0.0, divv = 0.0;
    for (int i = 0; i < d; i++) {
      divu += Du_(i, i);
      divv += Dv_(i, i);
    }
    real_t ee = 0.0;
    for (int i = 0; i < d; i++) {
      for (int j = 0; j < d; j++) {
        ee += 0.25 * (Du_(i, j) + Du_(j, i)) * (Dv_(i, j) + Dv_(j, i));
      }
    }
    const real_t dev_pair = ee - divu * divv / d;  // dev eps : dev eps
    // m : dev eps(udag): reconstruct m from its trace-free components
    // (E0 = diag(1,-1), E1 = offdiag) against the pointwise strain.
    m_->GetVectorValue(T, ip, mv_);
    const real_t m_pair = mv_[0] * (Dv_(0, 0) - Dv_(1, 1)) +
                          mv_[1] * (Dv_(0, 1) + Dv_(1, 0));
    return -2.0 * (dev_pair - m_pair - val);
  }

 private:
  Mode mode_;
  Coefficient* tau_;
  const GridFunction *u_ = nullptr, *d_ = nullptr, *m_ = nullptr;
  const GridFunction *udag_ = nullptr, *ddag_ = nullptr, *mdag_ = nullptr;
  DenseMatrix Du_, Dv_;
  Vector dv_, mv_, bv_;
};

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif

  const char* mesh_file = "../data/elastogravity_two_layer_2d.msh";
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
      while (d > M_PI) d -= 2.0 * M_PI;
      while (d < -M_PI) d += 2.0 * M_PI;
      return std::exp(-d * d / (kSigma * kSigma)) /
             (kSigma * kRBody * std::sqrt(M_PI));
    };
    const real_t w =
        window(kPhi0 + 0.5 * kDPhi) - window(kPhi0 - 0.5 * kDPhi);
    v.SetSize(2);
    v[0] = -w * std::sin(kPhi0);
    v[1] = w * std::cos(kPhi0);
  });
  auto g = LinearFormType(&fes);
  g.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(h), surface);
  g.Assemble();

  const real_t dt = t_final / n_steps;

  // One full forward run for a given (kappa, mu, eta): build the Maxwell
  // problem (tau = eta / mu), step to t1, return J = g . u(t1). With
  // keep != nullptr, also store the trajectory: per step the
  // displacement, the strain d and the internal variable m as
  // component GridFunctions on the internal-variable spaces.
  struct Trajectory {
    std::vector<FieldType> u;   // displacement (its space persists)
    std::vector<Vector> d, m;   // internal-node component vectors: the
                                // per-run internal space does not
                                // outlive the run, so the fields are
                                // rebuilt on the adjoint's space.
  };
  auto run_forward = [&](Coefficient& kappa, Coefficient& mu,
                         Coefficient& eta, Trajectory* keep) {
    auto tau = RatioCoefficient(eta, mu);
    auto rheology = IsotropicMaxwellRheology::Maxwell(dim, kappa, mu, tau);
    auto prob = LinearQuasiStaticTractionProblem(&fes, rheology, traction,
                                                 surface);
    auto visco = ViscoelasticOperator(prob);
    ExponentialTrapezoidSolver ode;
    ode.Init(visco);
    Vector m(visco.Height());
    m = 0.0;
    real_t t = 0.0, dt_step = dt;  // Step takes dt by reference
    Vector d_full;
    auto record = [&]() {
      if (!keep) return;
      auto& ug = keep->u.emplace_back(&fes);
      ug = prob.Displacement();
      visco.ComputeStrain(prob.Displacement(), d_full);
      keep->d.emplace_back(d_full);
      Vector m_full;
      visco.BranchToFull(m, 0, m_full);
      keep->m.emplace_back(std::move(m_full));
    };
    MFEM_VERIFY(visco.SolveElastic(m, t), "forward solve failed");
    record();
    for (int n = 1; n <= n_steps; n++) {
      ode.Step(m, t, dt_step);
      MFEM_VERIFY(visco.SolveElastic(m, t), "forward solve failed");
      record();
    }
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
        return kMu0 * kTau0 *
               (1.0 + var_depth * (1.0 - q2) + blob_amp * blob);
      });
  Trajectory fwd;
  const real_t J0 = run_forward(kappa0, mu0, eta0, &fwd);

  // === The adjoint run ===
  // The same Maxwell problem with ZERO external load. The measurement's
  // impulsive response u+_g (one static solve loaded by g) jumps the
  // adjoint internal variable to m+(0+) = dev eps(u+_g) / tau, after
  // which the operator relaxes it freely in reversed time.
  auto tau0 = RatioCoefficient(eta0, mu0);
  auto rheology0 = IsotropicMaxwellRheology::Maxwell(dim, kappa0, mu0, tau0);
  auto zero_vec = Vector(dim);
  zero_vec = 0.0;
  auto zero_traction = VectorConstantCoefficient(zero_vec);
  auto prob_dag = LinearQuasiStaticTractionProblem(&fes, rheology0,
                                                   zero_traction, surface);
  auto visco_dag = ViscoelasticOperator(prob_dag);
  auto* dfes = static_cast<SpaceType*>(&visco_dag.InternalVariableSpace());

  prob_dag.AssembleForce(0.0);
  prob_dag.AddForce(g);
  MFEM_VERIFY(prob_dag.Solve(), "adjoint impulse solve failed");
  auto udag_g = FieldType(&fes);
  udag_g = prob_dag.Displacement();
  Vector ddag_g_full;
  visco_dag.ComputeStrain(udag_g, ddag_g_full);
  auto ddag_g = FieldType(dfes);
  ddag_g = ddag_g_full;

  // m+(0+): the strain of the impulse response scaled nodewise by 1/tau.
  Vector mdag(visco_dag.Height());
  {
    const Vector& itau = visco_dag.InverseRelaxationTimes(0);
    const int nd = visco_dag.NumBranchNodes(0);
    for (int c = 0; c < visco_dag.NumComponents(); c++) {
      for (int p = 0; p < nd; p++) {
        mdag[c * nd + p] = itau[p] * ddag_g_full[c * nd + p];
      }
    }
  }

  Trajectory adj;
  {
    ExponentialTrapezoidSolver ode;
    ode.Init(visco_dag);
    real_t t = 0.0, dt_step = dt;
    Vector d_full;
    auto record = [&]() {
      auto& ug = adj.u.emplace_back(&fes);
      ug = prob_dag.Displacement();
      visco_dag.ComputeStrain(prob_dag.Displacement(), d_full);
      adj.d.emplace_back(d_full);
      Vector m_full;
      visco_dag.BranchToFull(mdag, 0, m_full);
      adj.m.emplace_back(std::move(m_full));
    };
    // The impulse lives at t' = 0 only: the smooth adjoint displacement
    // there is the response to m+(0+) alone (AssembleForce has already
    // dropped the AddForce increment).
    visco_dag.InvalidateDisplacement();
    MFEM_VERIFY(visco_dag.SolveElastic(mdag, t), "adjoint solve failed");
    record();
    for (int k = 1; k <= n_steps; k++) {
      ode.Step(mdag, t, dt_step);
      MFEM_VERIFY(visco_dag.SolveElastic(mdag, t), "adjoint solve failed");
      record();
    }
  }

  // === The kernels: time-quadrature over the paired trajectories ===
  auto sfes = SpaceType(&body, &fec);
  auto ones = FieldType(&sfes);
  ones = 1.0;
  // The default quadrature matches the assembly rule of the elastic
  // forms: the verification integral then follows the DISCRETE
  // derivative closely (a higher-order rule is a less faithful pairing,
  // not a more accurate one).
  auto integral = [&](Coefficient& c) {
    auto lf = LinearFormType(&sfes);
    lf.AddDomainIntegrator(new DomainLFIntegrator(c));
    lf.Assemble();
    return GlobalSum(lf * ones);
  };

  // The kernel fields, for display: L2 projections accumulated with the
  // same trapezoid weights (an L2 space of the internal order).
  auto kfec = L2_FECollection(order - 1, dim);
  auto kfes = SpaceType(&body, &kfec);
  auto K_kappa = FieldType(&kfes), K_mu = FieldType(&kfes),
       K_eta = FieldType(&kfes);
  K_kappa = 0.0;
  K_mu = 0.0;
  K_eta = 0.0;
  auto Kk_step = FieldType(&kfes);

  DensityCoefficient dens_k(DensityCoefficient::Mode::Kappa, tau0);
  DensityCoefficient dens_m(DensityCoefficient::Mode::Mu, tau0);
  DensityCoefficient dens_e(DensityCoefficient::Mode::Eta, tau0);

  // Verification patterns and accumulators: int K m0 chi dV per case.
  auto chi_blob = FunctionCoefficient([](const Vector& x) {
    const real_t dx = x[0] - 0.25, dy = x[1] - 0.1;
    return std::exp(-(dx * dx + dy * dy) / 0.01);
  });
  auto chi_all = ConstantCoefficient(1.0);
  struct Case {
    const char* name;
    int param;  // 0 kappa, 1 mu, 2 eta
    Coefficient* chi;
    real_t dJ_kernel = 0.0;
  };
  Case cases[] = {{"kappa, blob", 0, &chi_blob}, {"kappa, all", 0, &chi_all},
                  {"mu, blob", 1, &chi_blob},    {"mu, all", 1, &chi_all},
                  {"eta, blob", 2, &chi_blob},   {"eta, all", 2, &chi_all}};
  Coefficient* m0_of[] = {&kappa0, &mu0, &eta0};
  DensityCoefficient* dens_of[] = {&dens_k, &dens_m, &dens_e};

  // The tensor factors as fields on the (persistent) adjoint internal
  // space, reassigned per pairing step.
  auto d_n = FieldType(dfes), m_n = FieldType(dfes);
  auto ddag_k = FieldType(dfes), mdag_k = FieldType(dfes);
  for (int n = 0; n <= n_steps; n++) {
    const int k = n_steps - n;  // adjoint index at t' = t1 - t
    const real_t w = (n == 0 || n == n_steps) ? 0.5 * dt : dt;
    d_n = fwd.d[n];
    m_n = fwd.m[n];
    ddag_k = adj.d[k];
    mdag_k = adj.m[k];
    for (auto* dens : dens_of) {
      dens->SetPair(fwd.u[n], d_n, m_n, adj.u[k], ddag_k, mdag_k);
    }
    for (auto& c : cases) {
      auto weighted = ProductCoefficient(*c.chi, *dens_of[c.param]);
      auto weighted_m0 = ProductCoefficient(weighted, *m0_of[c.param]);
      c.dJ_kernel += w * integral(weighted_m0);
    }
    for (auto [K, dens] : {std::pair{&K_kappa, &dens_k},
                           std::pair{&K_mu, &dens_m},
                           std::pair{&K_eta, &dens_e}}) {
      Kk_step.ProjectCoefficient(*dens);
      K->Add(w, Kk_step);
    }
  }
  // The impulsive (purely elastic) parts: the adjoint delta at t' = 0
  // pairs with the forward state at t1 (K_eta has none: m+ is bounded).
  {
    d_n = fwd.d[n_steps];
    m_n = fwd.m[n_steps];
    // Mode::Mu subtracts mdag from ddag: pass a zero mdag.
    auto zero_m = FieldType(dfes);
    zero_m = 0.0;
    dens_k.SetPair(fwd.u[n_steps], d_n, m_n, udag_g, ddag_g, zero_m);
    dens_m.SetPair(fwd.u[n_steps], d_n, m_n, udag_g, ddag_g, zero_m);
    for (auto& c : cases) {
      if (c.param == 2) continue;
      auto weighted = ProductCoefficient(*c.chi, *dens_of[c.param]);
      auto weighted_m0 = ProductCoefficient(weighted, *m0_of[c.param]);
      c.dJ_kernel += integral(weighted_m0);
    }
    Kk_step.ProjectCoefficient(dens_k);
    K_kappa.Add(1.0, Kk_step);
    Kk_step.ProjectCoefficient(dens_m);
    K_mu.Add(1.0, Kk_step);
  }

  // === The verification table ===
  if (Root()) {
    std::cout << "\nJ at the base model (station baseline at t1 = "
              << t_final << "): " << J0 << "\n\n"
              << "  perturbation        kernel dJ        central FD dJ"
              << "      rel. diff\n";
  }
  for (auto& c : cases) {
    auto bump = [&](Coefficient& base, bool on, real_t s) {
      auto one_plus = std::make_unique<SumCoefficient>(1.0, *c.chi, 1.0,
                                                       on ? s : 0.0);
      auto prod = std::make_unique<ProductCoefficient>(base, *one_plus);
      return std::pair(std::move(one_plus), std::move(prod));
    };
    const real_t e = fd_eps;
    real_t dJ_fd = 0.0;
    for (const real_t s : {e, -e}) {
      auto [o1, kp] = bump(kappa0, c.param == 0, s);
      auto [o2, mp] = bump(mu0, c.param == 1, s);
      auto [o3, ep] = bump(eta0, c.param == 2, s);
      const real_t J = run_forward(*kp, *mp, *ep, nullptr);
      dJ_fd += (s > 0 ? J : -J);
    }
    dJ_fd /= 2.0 * e;
    if (Root()) {
      std::cout << "  " << std::left << std::setw(16) << c.name
                << std::right << std::scientific << std::setprecision(6)
                << std::setw(17) << c.dJ_kernel << std::setw(19) << dJ_fd
                << std::setprecision(1) << std::setw(15)
                << std::abs(c.dJ_kernel / dJ_fd - 1.0) << std::defaultfloat
                << std::setprecision(6) << "\n";
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
  // As in adjoint_elasticity, but with the duals assembled from the
  // ACCUMULATED kernel fields (transferred to the parent mesh and
  // integrated against the scalar H1 basis over the body attribute;
  // zero in the buffer).
  auto pfes_s = SpaceType(&parent, &fec);
  auto pkfes = SpaceType(&parent, &kfec);
  auto body_marker = Array<int>(parent.attributes.Max());
  body_marker = 0;
  body_marker[0] = 1;
  auto dual = [&](FieldType& K) {
    auto K_parent = FieldType(&pkfes);
    K_parent = 0.0;
    SubMeshType::Transfer(K, K_parent);
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
  const Vector j_kappa = dual(K_kappa);
  const Vector j_mu = dual(K_mu);
  const Vector j_eta = dual(K_eta);

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
    std::cout << "Derivatives into gradients (smoothing length "
              << smoothing << "):\n";
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
