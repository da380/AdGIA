/*
  Gate GA1b (WP8.4 opening): the sea-level reciprocity theorems of
  Al-Attar, Syvret, Crawford, Mitrovica & Lloyd (2024,
  doc/Elasticity/ggad434.pdf) as certificates on the water-loaded mixed
  solve — the elastic fingerprint problem that SetWaterLoad implements
  (condensed sea-level equation, frozen shoreline; here all-ocean).

  The theorem: with two direct loads zeta_1, zeta_2 sharing the ocean
  function, the fingerprint responses satisfy

      int SL_1 zeta_2 dS = int SL_2 zeta_1 dS          (paper eq. 17)

  — self-adjointness of the fingerprint operator zeta -> SL — and the
  SAME identity holds with rotational feedbacks, with SL read on the
  equipotential of the combined gravitational and centrifugal
  potentials (eqs. 52-54). Three legs, on the canned 3-D two-layer
  ball (SetWaterLoad is 3-D only):

  1. spherically symmetric moduli (the classically tested case);
  2. LATERALLY HETEROGENEOUS 3-D elastic structure — per the paper's
     closing discussion this case has not previously been tested
     numerically; the discrete identity must hold regardless, because
     the assembled system stays symmetric whatever kappa(x), mu(x);
  3. rotation + 3-D structure jointly (the generalized theorem, read
     with psi from the solved rotational feedback).

  The pairings go through the POSTPROCESSED sea-level fields
  (SeaLevelOperator), i.e. the route tier in the WP7.6 sense: the
  algebraic solve is symmetric to solver grade, and the residual is the
  projection/quadrature gap of the readout. Thresholds encode
  measurements (see each leg).

  The melt loads are Gaussian caps at generic, non-symmetric positions;
  their masses are nonzero (the physical fingerprint case) — the
  uniform terms drop from the identity through the solve's mass row,
  as in the paper's proof.
*/

#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include "AdGIA.hpp"
#include "MixedProblemTestCommon.hpp"
#include "mfem.hpp"

using namespace mfem;
using namespace AdGIA;

namespace {

constexpr double kRhoW = 0.02;
constexpr double kOmega = 0.1;

// Gaussian melt cap centred at the unit direction n (s = angular width).
double MeltCap(const Vector& x, const Vector& n) {
  constexpr double s2 = 0.08;
  Vector d(x);
  d /= x.Norml2();
  d -= n;
  return 0.02 * std::exp(-(d * d) / s2);
}

Vector UnitDir(double a, double b, double c) {
  Vector n(3);
  n[0] = a;
  n[1] = b;
  n[2] = c;
  n /= n.Norml2();
  return n;
}

// Smooth lateral moduli variations, nonzero through degree 2 in the
// Cartesian harmonics (well-defined at the origin, no shared symmetry
// with the caps or with each other).
double KappaLateral(const Vector& x) {
  return self_grav_test::kKappa *
         (1.0 + 0.3 * (x[0] * x[0] - x[1] * x[1]) + 0.2 * x[0] * x[2]);
}
double MuLateral(const Vector& x) {
  return self_grav_test::kMu *
         (1.0 - 0.25 * (x[1] * x[1] - x[2] * x[2]) + 0.2 * x[0] * x[1]);
}

// m0 (1 + s chi), owned (as in TestAdjoint.cpp).
struct Bumped {
  std::unique_ptr<SumCoefficient> one_plus;
  std::unique_ptr<ProductCoefficient> value;
};
Bumped Bump(Coefficient& m0, Coefficient& chi, real_t s) {
  Bumped b;
  b.one_plus = std::make_unique<SumCoefficient>(1.0, chi, 1.0, s);
  b.value = std::make_unique<ProductCoefficient>(m0, *b.one_plus);
  return b;
}

struct Fixture {
  std::unique_ptr<Mesh> parent;
  std::unique_ptr<SubMesh> body;
  std::unique_ptr<H1_FECollection> fec;
  std::unique_ptr<FiniteElementSpace> fes_u, fes_phi;
  ConstantCoefficient rho{self_grav_test::kRho};
  Array<int> surface;
  Vector moments;
  std::unique_ptr<ConstantCoefficient> w;

  explicit Fixture(int order) {
    parent =
        std::make_unique<Mesh>(self_grav_test::MeshFile(3).c_str(), 1, 1);
    body = std::make_unique<SubMesh>(SubMesh::CreateFromDomain(
        *parent, self_grav_test::BodyMarker(*parent)));
    fec = std::make_unique<H1_FECollection>(order, 3);
    fes_u = std::make_unique<FiniteElementSpace>(body.get(), fec.get(), 3);
    fes_phi = std::make_unique<FiniteElementSpace>(parent.get(), fec.get());
    surface = self_grav_test::SurfaceMarker(*body);
    moments.SetSize(3);
    moments[0] = 1.2;
    moments[1] = 1.3;
    moments[2] = 2.0;
    const double g0 =
        4.0 * M_PI * self_grav_test::kG * self_grav_test::kRho / 3.0;
    w = std::make_unique<ConstantCoefficient>(kRhoW / g0);
  }

  VectorFunctionCoefficient GradPhi0() {
    const double g0 =
        4.0 * M_PI * self_grav_test::kG * self_grav_test::kRho / 3.0;
    return VectorFunctionCoefficient(3, [g0](const Vector& x, Vector& v) {
      v = x;
      v *= g0;
    });
  }
};

// One fingerprint solve: the melt load zeta through the water border,
// optional rotation; the sea-level field copied out of the shared
// SeaLevelOperator.
GridFunction Fingerprint(Fixture& f, const Rheology& rheology,
                         Coefficient& zeta, bool rotation,
                         SeaLevelOperator& sea,
                         VectorCoefficient& grad_phi0) {
  LinearQuasiStaticMixedSelfGravitatingProblem p(
      f.fes_u.get(), f.fes_phi.get(), rheology, f.rho, self_grav_test::kG,
      self_grav_test::kDtNDegree);
  p.SetRelTol(1e-12);
  p.SetWaterLoad(*f.w, zeta, f.surface);
  if (rotation) {
    p.SetRotation(kOmega, f.moments);
  }
  p.AssembleForce(0.0);
  EXPECT_TRUE(p.Solve());
  Coefficient* psi = rotation ? &p.SolutionCentrifugalPotential() : nullptr;
  sea.SeaLevelChangeFrom(p.Displacement(), grad_phi0, p.Potential(),
                         p.UniformPotentialTerm(), psi);
  GridFunction sl(&sea.SurfaceSpace());
  sl = sea.SeaLevelChangeField();
  return sl;
}

// The reciprocity defect |<SL_1, z_2> - <SL_2, z_1>| / max(|.|,|.|).
double ReciprocityDefect(Fixture& f, const Rheology& rheology,
                         bool rotation) {
  auto grad_phi0 = f.GradPhi0();
  auto n1 = UnitDir(0.3, 0.5, 0.81);
  auto n2 = UnitDir(-0.7, 0.25, 0.55);
  auto z1 = FunctionCoefficient(
      [n1](const Vector& x) { return MeltCap(x, n1); });
  auto z2 = FunctionCoefficient(
      [n2](const Vector& x) { return MeltCap(x, n2); });

  SeaLevelOperator sea(*f.fes_u, f.surface);
  auto sl1 = Fingerprint(f, rheology, z1, rotation, sea, grad_phi0);
  auto sl2 = Fingerprint(f, rheology, z2, rotation, sea, grad_phi0);

  auto pair = [&](GridFunction& sl, Coefficient& z) {
    GridFunctionCoefficient slc(&sl);
    ProductCoefficient p(slc, z);
    return sea.SurfaceIntegral(p);
  };
  const double p12 = pair(sl1, z2);
  const double p21 = pair(sl2, z1);
  return std::abs(p12 - p21) / std::max(std::abs(p12), std::abs(p21));
}

}  // namespace

TEST(AdjointFingerprint, SphericalReciprocity) {
  Fixture f(1);
  auto kappa = ConstantCoefficient(self_grav_test::kKappa);
  auto mu = ConstantCoefficient(self_grav_test::kMu);
  auto rheology = IsotropicElasticRheology(3, kappa, mu);
  // Measured 2.5e-5 at order 1 and 7.6e-6 at order 2: the defect is
  // the readout's projection/quadrature gap and converges with order
  // (the algebraic solve is symmetric to solver grade), so the gate
  // checks both the size and the convergence.
  const double d1 = ReciprocityDefect(f, rheology, false);
  EXPECT_LT(d1, 1e-4);
  Fixture f2(2);
  const double d2 = ReciprocityDefect(f2, rheology, false);
  EXPECT_LT(d2, 3e-5);
  EXPECT_LT(d2, d1);
}

TEST(AdjointFingerprint, LateralStructureReciprocity) {
  // The never-previously-tested case: laterally heterogeneous kappa(x),
  // mu(x). The discrete identity holds regardless of the structure —
  // the assembled system stays symmetric — so the threshold is the
  // same solver-plus-readout grade as the spherical leg.
  Fixture f(1);
  auto kappa = FunctionCoefficient(KappaLateral);
  auto mu = FunctionCoefficient(MuLateral);
  auto rheology = IsotropicElasticRheology(3, kappa, mu);
  // Measured 3.5e-5 — the same readout grade as the spherical leg:
  // the first numerical test of fingerprint reciprocity with 3-D
  // elastic structure.
  EXPECT_LT(ReciprocityDefect(f, rheology, false), 1e-4);
}

TEST(AdjointFingerprint, RotationWithStructureReciprocity) {
  // The generalized theorem (paper §3.2, eq. 54): rotation composed
  // with the water border, jointly with the 3-D structure; SL read on
  // the combined equipotential (psi from the solved feedback).
  Fixture f(1);
  auto kappa = FunctionCoefficient(KappaLateral);
  auto mu = FunctionCoefficient(MuLateral);
  auto rheology = IsotropicElasticRheology(3, kappa, mu);
  // Measured 1.9e-5 — the generalized theorem at the same grade.
  EXPECT_LT(ReciprocityDefect(f, rheology, true), 1e-4);
}

// Gate GA2: the full-physics kernels — Maxwell viscoelasticity with the
// water border AND the rotational border in the solve (frozen
// shorelines, so C-dagger = C0 and no ocean-function playback arises;
// the production first-order path) — FD-gated for a final-time
// DISPLACEMENT observation and a final-time POTENTIAL observation. The
// potential observation is the gate on AddPotentialForce's mass-like
// border bookkeeping (Yu eq. 58): the datum's omega-row pairings and
// mass-row integral must be fed, or the kernels break against FD.
TEST(AdjointFingerprint, FullPhysicsKernels) {
  const int order = 2, n_steps = 12;
  const real_t t1 = 1.5;  // Maxwell times
  const real_t kOm_ = kOmega;
  Fixture f(order);
  auto kappa0 = ConstantCoefficient(self_grav_test::kKappa);
  auto mu0 = ConstantCoefficient(self_grav_test::kMu);
  auto eta0 = ConstantCoefficient(self_grav_test::kMu);  // tau = 1
  auto tau0 = RatioCoefficient(eta0, mu0);
  auto grad_phi0 = f.GradPhi0();

  // A held surface mass load with a mean part (the uniform term works)
  // and non-zonal structure.
  auto sigma_data = FunctionCoefficient([](const Vector& x) {
    const real_t r = x.Norml2();
    const real_t c = x[2] / r;
    return 0.02 * (3.0 * c * c - 0.5 + 0.5 * x[0] * x[2] / (r * r));
  });
  auto zero_data = ConstantCoefficient(0.0);

  // Observation duals: cap-windowed radial displacement at n1 (body
  // surface) and a potential window at n2 (parent surface).
  auto n1 = UnitDir(0.3, 0.5, 0.81);
  auto n2 = UnitDir(-0.7, 0.25, 0.55);
  auto h_u = VectorFunctionCoefficient(3, [n1](const Vector& x, Vector& v) {
    const real_t w = MeltCap(x, n1);
    v = x;
    v *= w / x.Norml2();
  });
  auto g_u = LinearForm(f.fes_u.get());
  g_u.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(h_u), f.surface);
  g_u.Assemble();
  auto h_phi = FunctionCoefficient(
      [n2](const Vector& x) { return MeltCap(x, n2); });
  Array<int> parent_surface(f.parent->bdr_attributes.Max());
  parent_surface = 0;
  parent_surface[0] = 1;
  auto g_phi = LinearForm(f.fes_phi.get());
  g_phi.AddBoundaryIntegrator(new BoundaryLFIntegrator(h_phi),
                              parent_surface);
  g_phi.Assemble();

  const real_t dt = t1 / n_steps;
  auto make_problem = [&](const Rheology& rheology, Coefficient& data) {
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        f.fes_u.get(), f.fes_phi.get(), rheology, f.rho,
        self_grav_test::kG, self_grav_test::kDtNDegree);
    p->SetWaterLoad(*f.w, data, f.surface);
    p->SetRotation(kOm_, f.moments);
    p->SetRelTol(1e-11);
    return p;
  };

  // The SL observation J = int h SL1 dS, as the STRICT four-part dual
  // pairing of the solution components (SL1 = -(u.grad Phi0 + phi +
  // psi(omega))/g + Phi_g/g with g = g0 on the unit surface): the
  // discrete functional whose exact adjoint the slots build. h is a
  // third cap window.
  const real_t g0 = 4.0 * M_PI * self_grav_test::kG * self_grav_test::kRho /
                    3.0;
  auto n3 = UnitDir(0.1, -0.6, 0.79);
  auto h_sl = FunctionCoefficient(
      [n3](const Vector& x) { return MeltCap(x, n3); });
  auto h_sl_u = VectorFunctionCoefficient(
      3, [n3, g0](const Vector& x, Vector& v) {
        // -(1/g) h grad Phi0 = -h g0 x / g0 = -h x on the unit surface.
        v = x;
        v *= -MeltCap(x, n3);
      });
  auto g_sl_u = LinearForm(f.fes_u.get());
  g_sl_u.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(h_sl_u),
                               f.surface);
  g_sl_u.Assemble();
  auto h_sl_phi = ProductCoefficient(-1.0 / g0, h_sl);
  auto g_sl_phi = LinearForm(f.fes_phi.get());
  g_sl_phi.AddBoundaryIntegrator(new BoundaryLFIntegrator(h_sl_phi),
                                 parent_surface);
  g_sl_phi.Assemble();
  const real_t sl_uniform = -g_sl_phi.Sum();  // +(1/g0) int h dS
  Vector sl_omega(3);
  {
    for (int k = 0; k < 3; k++) {
      CentrifugalPotential psi_k(3, kOm_);
      psi_k.SetUnit(k);
      ProductCoefficient hpsi(h_sl, psi_k);
      LinearForm lf(f.fes_phi.get());
      lf.AddBoundaryIntegrator(new BoundaryLFIntegrator(hpsi),
                               parent_surface);
      lf.Assemble();
      sl_omega[k] = -lf.Sum() / g0;
    }
  }

  struct Measured {
    real_t J_u = 0.0, J_phi = 0.0, J_sl = 0.0;
  };
  auto run_forward = [&](Coefficient& kappa, Coefficient& mu,
                         Coefficient& eta) {
    auto tau = RatioCoefficient(eta, mu);
    auto rheology = IsotropicMaxwellRheology::Maxwell(3, kappa, mu, tau);
    auto prob = make_problem(rheology, sigma_data);
    auto visco = ViscoelasticOperator(*prob);
    ExponentialTrapezoidSolver ode;
    ode.Init(visco);
    Vector m(visco.Height());
    m = 0.0;
    real_t t = 0.0, dt_step = dt;
    EXPECT_TRUE(visco.SolveElastic(m, t));
    for (int n = 1; n <= n_steps; n++) {
      ode.Step(m, t, dt_step);
    }
    EXPECT_TRUE(visco.SolveElastic(m, t));
    Measured out;
    out.J_u = g_u * prob->Displacement();
    out.J_phi = g_phi * prob->Potential();
    out.J_sl = g_sl_u * prob->Displacement() + g_sl_phi * prob->Potential() +
               sl_uniform * prob->UniformPotentialTerm() +
               sl_omega * prob->AngularVelocity();
    return out;
  };

  // The base run on persistent operators.
  auto rheology0 = IsotropicMaxwellRheology::Maxwell(3, kappa0, mu0, tau0);
  auto prob_fwd = make_problem(rheology0, sigma_data);
  auto visco_fwd = ViscoelasticOperator(*prob_fwd);
  auto trajectory = ForwardTrajectory(visco_fwd, true);
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
  }
  auto prob_adj = make_problem(rheology0, zero_data);
  auto visco_adj = ViscoelasticOperator(*prob_adj);

  auto chi_blob3 = FunctionCoefficient([](const Vector& x) {
    const real_t dx = x[0] - 0.3, dy = x[1] - 0.35, dz = x[2] - 0.4;
    return std::exp(-(dx * dx + dy * dy + dz * dz) / 0.03);
  });
  auto chi_all = ConstantCoefficient(1.0);
  FiniteElementSpace sfes(f.body.get(), f.fec.get());
  std::vector<ProductCoefficient> dirs;
  dirs.emplace_back(eta0, chi_blob3);  // 0: eta blob
  dirs.emplace_back(eta0, chi_all);    // 1: eta all
  dirs.emplace_back(mu0, chi_blob3);   // 2: mu blob

  struct Sweep {
    std::unique_ptr<IsotropicKernel> eta, mu;
  };
  auto make_sweep = [&](const AdjointForce& force) {
    Sweep sw;
    sw.eta = std::make_unique<IsotropicKernel>(
        IsotropicKernel::Parameter::Eta, sfes, &tau0);
    sw.mu = std::make_unique<IsotropicKernel>(IsotropicKernel::Parameter::Mu,
                                              sfes);
    sw.eta->AddDirection(dirs[0]);
    sw.eta->AddDirection(dirs[1]);
    sw.mu->AddDirection(dirs[2]);
    auto solver = AdjointSolver(visco_adj, trajectory);
    solver.SetForce(force);
    solver.AddKernel(*sw.eta);
    solver.AddKernel(*sw.mu);
    EXPECT_TRUE(solver.Run());
    return sw;
  };
  AdjointForce force_u;
  force_u.AddObservation(trajectory.Time(n_steps), g_u);
  AdjointForce force_phi;
  force_phi.AddPotentialObservation(trajectory.Time(n_steps), g_phi);
  AdjointForce force_sl;
  {
    AdjointLoad load;
    load.u = &g_sl_u;
    load.phi = &g_sl_phi;
    load.uniform = sl_uniform;
    load.omega = &sl_omega;
    force_sl.AddObservation(trajectory.Time(n_steps), load);
  }
  auto sweep_u = make_sweep(force_u);
  auto sweep_phi = make_sweep(force_phi);
  auto sweep_sl = make_sweep(force_sl);

  // FD: one rerun per sign and direction, both functionals sampled.
  const real_t eps = 1e-3;
  Coefficient* base_of[3] = {&eta0, &eta0, &mu0};
  Coefficient* chi_of[3] = {&chi_blob3, &chi_all, &chi_blob3};
  Measured fd[3];
  for (int d = 0; d < 3; d++) {
    Measured plus, minus;
    for (const real_t sg : {eps, -eps}) {
      auto bk = Bump(kappa0, *chi_of[d], 0.0);
      auto bm = Bump(mu0, *chi_of[d], d == 2 ? sg : 0.0);
      auto be = Bump(eta0, *chi_of[d], d == 2 ? 0.0 : sg);
      auto out = run_forward(*bk.value, *bm.value, *be.value);
      (sg > 0 ? plus : minus) = out;
    }
    fd[d].J_u = (plus.J_u - minus.J_u) / (2.0 * eps);
    fd[d].J_phi = (plus.J_phi - minus.J_phi) / (2.0 * eps);
    fd[d].J_sl = (plus.J_sl - minus.J_sl) / (2.0 * eps);
  }

  // Thresholds encode measurements at order 2, n = 12. The
  // body-integrated eta rows are the structural gates: 8e-5 (u),
  // 9e-7 (phi) and 1.6e-5 (SL, the FOUR-SLOT dual including the
  // border rows — BorderForceSlot's sign and scale certificate) — the
  // phi row is ALSO the certificate that
  // AddPotentialForce loads only the potential block (feeding the
  // border rows as if the dual were a surface load measured 10 %
  // here; the bordered system's symmetry carries the eq.-58
  // bookkeeping by itself). The blob rows carry the coarse ball's
  // spatial floor (nodal sampling + curved tets): eta 2.7e-2/6.4e-2,
  // mu 0.15/0.15 — order 1 is unusable for them (eta blob 0.59, mu
  // blob SIGN-FLIPS at internal order zero), and order 2 restores
  // sign and magnitude; they gate structure, not sharpness.
  auto gate = [&](Sweep& sw, auto member, const char* what) {
    EXPECT_NEAR(sw.eta->Directional(0) / (fd[0].*member), 1.0, 1.5e-1)
        << what << ", eta blob";
    EXPECT_NEAR(sw.eta->Directional(1) / (fd[1].*member), 1.0, 1e-3)
        << what << ", eta all";
    EXPECT_NEAR(sw.mu->Directional(0) / (fd[2].*member), 1.0, 3e-1)
        << what << ", mu blob";
  };
  gate(sweep_u, &Measured::J_u, "full-physics u observation");
  gate(sweep_phi, &Measured::J_phi, "full-physics phi observation");
  // The SL observation exercises ALL FOUR slots at once, including the
  // border rows (its functional reads the uniform term and omega) —
  // the gate on BorderForceSlot's sign and scale.
  gate(sweep_sl, &Measured::J_sl, "full-physics SL observation");
}
