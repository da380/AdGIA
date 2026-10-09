//------------------------------------------------------------------------------
//
// PURPOSE:
//
// This example (adjoint_elasticity) is the simplest member of the adjoint
// family: first-order sensitivity kernels for STATIC, non-gravitating,
// isotropic elasticity, with the classic finite-difference verification —
// and, on top, the distinction the inverse-problem literature often
// blurs: the kernel is a DERIVATIVE; a GRADIENT needs a metric.
//
// The geometry. The physical body is the INNER layer of the two-layer
// disc mesh alone (an elastic disc, uniform by default; -vd and -va
// add a smooth radial and harmonic-polynomial variation of the
// moduli); everything outside it —
// the second layer and the mesh's outer shell — serves only as the
// extension domain that the Sobolev metrics at the end need. Plots show
// the physical sub-domain.
//
// The load is localised: a Gaussian-in-angle radial traction with a
// trigonometric modulation,
//
//   p(phi) = p0 exp(-(phi - phiL)^2 / sL^2) cos(n (phi - phiL)),
//
// whose oscillation removes the net contributions (the net force
// carries the factor exp(-n^2 sL^2 / 4); a radial traction exerts no
// torque about the centre; the solver's rigid-mode projection absorbs
// the exponentially small remainder).
//
// The measurement is the intuitive geodetic one: the linearised change
// of DISTANCE between two stations on the surface. The stations sit
// only a few degrees apart — in effect a surface STRAIN measurement:
// with overlapping windows the functional approaches the tangential
// derivative of the windowed displacement. Point values are assembled
// as a SURFACE integral — two narrow angular windows w1, w2
// (normalised, opposite signs) centred on the stations x1, x2,
// weighting the displacement along the separation direction e (the
// surface tangent at the baseline's mid-point):
//
//   J(u) = int_S (w2 - w1) e . u dS  ~  e . (u(x2) - u(x1)).
//
// A boundary integrator keeps the observation exactly ON the surface
// (a raw delta force would also work pointwise, but its centre sits
// ambiguously against a curved discrete boundary, and in 2-D the point
// force's log-singular response is not even in H1 — the windowed
// functional is an honest bounded dual, so the continuum adjoint
// exists). J is translation-invariant exactly (the windows cancel) and
// rotation-invariant to O(window width squared), because a linearised
// rigid rotation moves u(x2) - u(x1) perpendicular to e; the solver's
// rigid-mode projection fixes one gauge for everything, so the
// verification below is clean regardless.
//
// The adjoint. With the symmetric bilinear form a(u, v; kappa, mu) of
// isotropic elasticity, the adjoint problem a(v, u+) = J'(u)[v] is the
// SAME traction problem loaded by the windowed station pair — an almost
// exactly equilibrated load (force zero by the window cancellation,
// torque zero to the window width squared, the rest projected). One
// extra elastic solve buys the sensitivity of J to every model
// parameter at once.
//
// The kernels. CONVENTIONS, stated for checking: the moduli are the
// library's (kappa, mu) pair with lambda = kappa - 2 mu / d, so the
// strain energy density is
//
//   E = (1/2) kappa (div u)^2 + mu dev eps(u) : dev eps(u),
//
// with eps the symmetric gradient and dev the d-dimensional deviator.
// Perturbing the ABSOLUTE moduli,
//
//   delta J = int_B [ K_kappa delta_kappa + K_mu delta_mu ] dV,
//
//   K_kappa(x) = - (div u)(div u+),
//   K_mu(x)    = - 2 dev eps(u) : dev eps(u+).
//
// The verification. For perturbation patterns chi (an inner-disc
// indicator, a Gaussian blob, and the whole body), the predicted change
// int K m0 chi is compared against the central finite difference of
// fresh forward solves at m0 (1 +/- e chi); the printed table agrees to
// a small tolerance that tightens with refinement.
//
// Derivative versus gradient. The kernel K is the L2 REPRESENTATION of
// the derivative: the dual vector j[delta m] = int K delta m identified
// through the L2 inner product. Identifying through a different metric
// gives a different "gradient" — and for CONTINUOUS moduli the honest
// choice is a space that embeds in C0, which in two dimensions H2 does
// with margin while L2 and H1 do not (doc/equilibrium_figures.tex §6;
// riesz.hpp). The example closes by applying the library's Riesz maps —
// L2, and the Sobolev metrics of order 1 (H1-like) and 2 (H2-like) — to
// BOTH kernels' duals. The Sobolev solves are posed on the extension
// domain with homogeneous Dirichlet conditions on its far boundary, so
// the metric's artificial boundary condition decays before it reaches
// the body. Watch the point-force signature: raw/L2 is spiky at the
// observation points, H1 is bounded but still peaked, H2 is the smooth
// field an inversion for continuous moduli should descend along.
//
// Visualisation (GLVis; -no-vis to skip): the forward and adjoint
// displacements, and each modulus's sensitivity in the three metrics.
//
// The gravitating version is deliberately absent here. Its roadmap: for
// MODULI kernels nothing changes beyond the forward and adjoint
// problems being gravitating; a DENSITY kernel must remember that the
// density also sources the EQUILIBRIUM Poisson equation, so the zeroth-
// and first-order dynamics both enter the Lagrangian; and the
// equilibrium stress depends on the density too, non-uniquely — the
// subtle part (after Yu, Al-Attar, Syvret & Lloyd 2025,
// doc/Elasticity/ggae388.pdf). A later example.
//
// One source serves the serial and the parallel build, as throughout.
//
// Options (defaults in brackets):
//   -m    mesh file [../data/elastogravity_two_layer_2d.msh]: the
//         two-layer disc whose inner layer is the body and whose outer
//         layers are the Sobolev buffer.
//   -o    displacement order [2]; raising it tightens the FD table.
//   -r    uniform refinements of the serial mesh [0]; likewise.
//   -fd   relative size of the FD perturbations [1e-3]: small enough
//         that the O(eps^2) truncation is negligible, large enough to
//         stay clear of solver roundoff.
//   -sl   smoothing length sqrt(beta/alpha) of the Sobolev metrics
//         [0.1]: the scale below which the H1/H2 gradients suppress
//         structure; a fraction of the body radius is sensible.
//   -vd   radial variation of the base moduli [0]: a term
//         vd (1 - (r/R)^2), smooth at the centre.
//   -va   angular variation of the base moduli [0]: a term
//         va r^2 cos(2 phi) / R^2 (a harmonic polynomial, smooth at
//         the centre; keep |vd| + |va| < 1).
//   -vis / -no-vis   GLVis windows on or off [on].
//
// Sample runs:  ./adjoint_elasticity
//               ./adjoint_elasticity -r 1 -o 3
//               mpiexec -np 4 ./adjoint_elasticity -r 1  (parallel build)
//
//------------------------------------------------------------------------------

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>

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
// the physical body; attributes 2 and 3 are the Sobolev extension.
constexpr real_t kRBody = 3483.0 / 6371.0;
constexpr real_t kKappa0 = 5.4, kMu0 = 2.7;

// The stations: a short baseline centred at the angle kPhi0 with
// separation kDPhi (a few degrees, strain-like), observed through
// angular windows of width kSigma (radians).
constexpr real_t kPhi0 = 0.3;
constexpr real_t kDPhi = 0.15;
constexpr real_t kSigma = 0.05;

// The load: localised at kPhiL, Gaussian width kSigmaL, modulation
// wavenumber kNMod (kNMod * kSigmaL = 4 suppresses the net force by
// exp(-4)).
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

// A pointwise kernel built from the forward and adjoint displacements:
//   mode Kappa: - (div u)(div u+)
//   mode Mu:    - 2 dev eps(u) : dev eps(u+)
class KernelCoefficient : public Coefficient {
 public:
  enum class Mode { Kappa, Mu };

  KernelCoefficient(const GridFunction& u, const GridFunction& udag,
                    Mode mode)
      : u_(&u), udag_(&udag), mode_(mode) {}

  real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    T.SetIntPoint(&ip);
    u_->GetVectorGradient(T, Du_);
    udag_->GetVectorGradient(T, Dv_);
    const int d = Du_.Height();
    real_t divu = 0.0, divv = 0.0;
    for (int i = 0; i < d; i++) {
      divu += Du_(i, i);
      divv += Dv_(i, i);
    }
    if (mode_ == Mode::Kappa) {
      return -divu * divv;
    }
    // dev eps(u) : dev eps(u+) = eps(u):eps(u+) - (div u)(div u+)/d
    real_t ee = 0.0;
    for (int i = 0; i < d; i++) {
      for (int j = 0; j < d; j++) {
        ee += 0.25 * (Du_(i, j) + Du_(j, i)) * (Dv_(i, j) + Dv_(j, i));
      }
    }
    return -2.0 * (ee - divu * divv / d);
  }

 private:
  const GridFunction *u_, *udag_;
  Mode mode_;
  DenseMatrix Du_, Dv_;
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
  real_t fd_eps = 1e-3;
  real_t smoothing = 0.1;
  real_t var_depth = 0.0;
  real_t var_angle = 0.0;
  bool visualization = true;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Two-layer disc mesh.");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&refinement, "-r", "--refinement",
                 "Uniform refinements of the serial mesh.");
  args.AddOption(&fd_eps, "-fd", "--fd-epsilon",
                 "Relative size of the finite-difference perturbations.");
  args.AddOption(&smoothing, "-sl", "--smoothing-length",
                 "Smoothing length sqrt(beta/alpha) of the Sobolev "
                 "metrics.");
  args.AddOption(&var_depth, "-vd", "--variation-depth",
                 "Radial variation of the base moduli: a term "
                 "vd (1 - (r/R)^2).");
  args.AddOption(&var_angle, "-va", "--variation-angle",
                 "Angular variation of the base moduli: a term "
                 "va r^2 cos(2 phi) / R^2 (smooth at the centre).");
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

  // The body's whole boundary (the former interface circle) is its
  // loaded, observed surface.
  auto surface = Array<int>(body.bdr_attributes.Max());
  surface = 1;
  auto traction = VectorFunctionCoefficient(dim, Traction);

  // === The observation functional: the windowed station pair ===
  // J(u) = int_S (w2 - w1) e . u dS with w1/w2 normalised angular
  // Gaussians at the stations and e the separation direction (for a
  // circle, exactly the surface tangent at the mid-point): assembled
  // once as a dual vector through a BOUNDARY integrator (so the
  // observation is exactly on the surface), it is both the measurement
  // and the adjoint load.
  auto h = VectorFunctionCoefficient(dim, [](const Vector& x, Vector& v) {
    const real_t phi = std::atan2(x[1], x[0]);
    auto window = [phi](real_t centre) {
      real_t d = phi - centre;
      while (d > M_PI) d -= 2.0 * M_PI;
      while (d < -M_PI) d += 2.0 * M_PI;
      // normalised against the arc measure r dphi
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

  // The perturbation patterns chi (dimensionless).
  auto chi_inner = FunctionCoefficient(
      [](const Vector& x) { return x.Norml2() < 0.5 * kRBody ? 1.0 : 0.0; });
  auto chi_blob = FunctionCoefficient([](const Vector& x) {
    const real_t dx = x[0] - 0.25, dy = x[1] - 0.1;
    return std::exp(-(dx * dx + dy * dy) / 0.01);
  });
  auto chi_all = ConstantCoefficient(1.0);

  // One forward solve for a given (kappa, mu); returns J = g(u).
  auto measure = [&](Coefficient& kappa, Coefficient& mu,
                     FieldType* keep_u = nullptr) {
    auto rheology = IsotropicElasticRheology(dim, kappa, mu);
    auto prob = LinearQuasiStaticTractionProblem(&fes, rheology, traction,
                                                 surface);
    prob.AssembleForce(0.0);
    MFEM_VERIFY(prob.Solve(), "forward solve failed");
    const real_t J = GlobalSum(g * prob.Displacement());
    if (keep_u) {
      *keep_u = prob.Displacement();
    }
    return J;
  };

  // === The forward and adjoint solves at the base model ===
  // The base model: uniform moduli, optionally modulated by a smooth
  // two-knob parameterisation (zero by default): a radial even power
  // and the degree-2 harmonic polynomial r^2 cos(2 phi) = x^2 - y^2 —
  // both well-defined at r = 0, unlike naive r- and phi-based
  // variations. Keep |vd| + |va| < 1 so the moduli stay positive.
  auto modulate = [var_depth, var_angle](real_t base) {
    return FunctionCoefficient([=](const Vector& x) {
      const real_t R2 = kRBody * kRBody;
      const real_t q2 = (x[0] * x[0] + x[1] * x[1]) / R2;
      const real_t c2 = (x[0] * x[0] - x[1] * x[1]) / R2;  // r^2 cos 2phi
      return base * (1.0 + var_depth * (1.0 - q2) + var_angle * c2);
    });
  };
  auto kappa0 = modulate(kKappa0);
  auto mu0 = modulate(kMu0);

  auto u = FieldType(&fes);
  const real_t J0 = measure(kappa0, mu0, &u);

  // The adjoint: the SAME traction problem, loaded by the windowed
  // station pair (self-adjointness in action).
  auto rheology0 = IsotropicElasticRheology(dim, kappa0, mu0);
  auto zero_vec = Vector(dim);
  zero_vec = 0.0;
  auto zero_traction = VectorConstantCoefficient(zero_vec);
  auto adjoint = LinearQuasiStaticTractionProblem(&fes, rheology0,
                                                  zero_traction, surface);
  adjoint.AssembleForce(0.0);
  adjoint.AddForce(g);
  MFEM_VERIFY(adjoint.Solve(), "adjoint solve failed");
  auto udag = FieldType(&fes);
  udag = adjoint.Displacement();

  // === The kernels and the verification table ===
  // (Element-by-element coefficients are the lightest route to the
  // kernels; the alternative is mfem's discrete interpolators mapping
  // displacement into strain-type fields, as the library does
  // elsewhere, which pays off once kernels feed a descent loop rather
  // than a plot.)
  auto K_kappa = KernelCoefficient(u, udag, KernelCoefficient::Mode::Kappa);
  auto K_mu = KernelCoefficient(u, udag, KernelCoefficient::Mode::Mu);

  // int_B K m0 chi dV, by quadrature against the H1 partition of unity.
  auto sfes = SpaceType(&body, &fec);
  auto ones = FieldType(&sfes);
  ones = 1.0;
  auto integral = [&](Coefficient& K, Coefficient& m0, Coefficient& chi) {
    auto Km = ProductCoefficient(K, m0);
    auto Kmc = ProductCoefficient(Km, chi);
    auto lf = LinearFormType(&sfes);
    lf.AddDomainIntegrator(new DomainLFIntegrator(Kmc));
    lf.Assemble();
    return GlobalSum(lf * ones);
  };

  struct Case {
    const char* name;
    bool is_kappa;
    Coefficient* chi;
  };
  Case cases[] = {{"kappa, inner", true, &chi_inner},
                  {"kappa, blob", true, &chi_blob},
                  {"kappa, all", true, &chi_all},
                  {"mu, inner", false, &chi_inner},
                  {"mu, blob", false, &chi_blob},
                  {"mu, all", false, &chi_all}};

  if (Root()) {
    std::cout << "\nJ at the base model (baseline change e.(u(x2)-u(x1))): "
              << J0 << "\n\n"
              << "  perturbation        kernel dJ        central FD dJ"
              << "      rel. diff\n";
  }
  for (const auto& c : cases) {
    auto bump = [&](Coefficient& m0_c, real_t s) {
      // m0 (1 + s chi) as nested coefficient objects (owned here).
      auto one_plus = std::make_unique<SumCoefficient>(1.0, *c.chi, 1.0, s);
      auto prod = std::make_unique<ProductCoefficient>(m0_c, *one_plus);
      return std::pair(std::move(one_plus), std::move(prod));
    };
    const real_t e = fd_eps;
    auto [op1, kp] = bump(kappa0, c.is_kappa ? e : 0.0);
    auto [op2, mp] = bump(mu0, c.is_kappa ? 0.0 : e);
    auto [om1, km] = bump(kappa0, c.is_kappa ? -e : 0.0);
    auto [om2, mm] = bump(mu0, c.is_kappa ? 0.0 : -e);
    const real_t Jp = measure(*kp, *mp);
    const real_t Jm = measure(*km, *mm);
    const real_t dJ_fd = (Jp - Jm) / (2.0 * e);
    const real_t dJ_kernel =
        c.is_kappa ? integral(K_kappa, kappa0, *c.chi)
                   : integral(K_mu, mu0, *c.chi);
    if (Root()) {
      std::cout << "  " << std::left << std::setw(16) << c.name
                << std::right << std::scientific << std::setprecision(6)
                << std::setw(17) << dJ_kernel << std::setw(19) << dJ_fd
                << std::setprecision(1) << std::setw(15)
                << std::abs(dJ_kernel / dJ_fd - 1.0) << std::defaultfloat
                << std::setprecision(6) << "\n";
    }
  }
  if (Root()) {
    std::cout << "\n(The agreement improves with -r/-o: the kernels are "
                 "quadrature\napproximations of the exact discrete "
                 "perturbation; the FD error is O(eps^2).)\n";
  }

  // === Derivative into gradient: the Riesz maps on the buffered mesh ===
  //
  // Each modulus's derivative as a dual on a scalar H1 space over the
  // WHOLE parent mesh (the kernel integrand lives on the body attribute
  // alone; the outer layers are the extension domain), then its
  // representative in three metrics, displayed on the body.
  auto pfes_v = SpaceType(&parent, &fec, dim);
  auto u_parent = FieldType(&pfes_v);
  auto udag_parent = FieldType(&pfes_v);
  u_parent = 0.0;
  udag_parent = 0.0;
  SubMeshType::Transfer(u, u_parent);
  SubMeshType::Transfer(udag, udag_parent);
  auto K_kappa_parent = KernelCoefficient(u_parent, udag_parent,
                                          KernelCoefficient::Mode::Kappa);
  auto K_mu_parent = KernelCoefficient(u_parent, udag_parent,
                                       KernelCoefficient::Mode::Mu);
  auto body_marker = Array<int>(parent.attributes.Max());
  body_marker = 0;
  body_marker[0] = 1;

  auto pfes_s = SpaceType(&parent, &fec);
  auto dual = [&](Coefficient& K) {
    auto lf = LinearFormType(&pfes_s);
    lf.AddDomainIntegrator(new DomainLFIntegrator(K), body_marker);
    lf.Assemble();
    Vector j(pfes_s.GetTrueVSize());
#ifdef MFEM_USE_MPI
    lf.ParallelAssemble(j);
#else
    j = lf;
#endif
    return j;
  };
  const Vector j_kappa = dual(K_kappa_parent);
  const Vector j_mu = dual(K_mu_parent);

  auto outer = ExternalBoundaryMarker(&parent);
  const real_t beta = smoothing * smoothing;
  auto riesz_l2 = L2RieszMap(pfes_s);
  auto riesz_h1 = SobolevRieszMap(pfes_s, 1.0, beta, 1, &outer);
  auto riesz_h2 = SobolevRieszMap(pfes_s, 1.0, beta, 2, &outer);

  auto body_sfes = SpaceType(&body, &fec);
  auto gradient_on_body = [&](RieszMap& map, const Vector& j,
                              const char* name) {
    Vector g_true(pfes_s.GetTrueVSize());
    map.Mult(j, g_true);
    auto on_parent = FieldType(&pfes_s);
    on_parent.SetFromTrueDofs(g_true);
    auto on_body = FieldType(&body_sfes);
    SubMeshType::Transfer(on_parent, on_body);
    // Pair is a global reduction: every rank calls it, the root prints.
    const real_t slope = map.Pair(j, g_true);
    if (Root()) {
      std::cout << "  " << name << ": slope <j, g> = " << slope << "\n";
    }
    return on_body;
  };
  if (Root()) {
    std::cout << "\nDerivatives into gradients (smoothing length "
              << smoothing << "):\n";
  }
  auto gk_l2 = gradient_on_body(riesz_l2, j_kappa, "kappa, L2");
  auto gk_h1 = gradient_on_body(riesz_h1, j_kappa, "kappa, H1");
  auto gk_h2 = gradient_on_body(riesz_h2, j_kappa, "kappa, H2");
  auto gm_l2 = gradient_on_body(riesz_l2, j_mu, "mu,    L2");
  auto gm_h1 = gradient_on_body(riesz_h1, j_mu, "mu,    H1");
  auto gm_h2 = gradient_on_body(riesz_h2, j_mu, "mu,    H2");
  if (Root()) {
    std::cout << "(each slope is positive — every representative is an "
                 "ascent direction of J\nin its own metric — while the "
                 "FIELDS differ: only the H2 one is a function\nfor "
                 "continuous moduli; the plots show the physical "
                 "sub-domain.)\n\n";
  }

  // === Visualisation ===
  if (visualization) {
    const std::string keys = examples::DefaultKeys(dim);
    examples::GLVisWindow("forward displacement u", keys).Send(body, u);
    examples::GLVisWindow("adjoint displacement u+ (station pair)", keys)
        .Send(body, udag);
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
  }
  return 0;
}
