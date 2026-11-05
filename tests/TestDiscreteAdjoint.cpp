/*
  Tests for the discrete first-order adjoint (discrete_adjoint.hpp): the
  transpose of the tangent. The gate is the pairing identity
  <g, du>_tangent = dJ_discrete along every direction, which holds to
  solver tolerance (no finite differences, no time-quadrature floor), on
  the adjoint tests' unit-square Maxwell configuration for a final-time,
  an interior-time and a continuous-in-time observation; and the
  comparison with the continuous adjoint's kernels, which agree to their
  O(dt^2) distance only.

  The static configuration: the elastic unit square (the identity, and
  the static continuous adjoint's kernels, which are the same solve and
  the same assembled form) and the gravitating canned disc. The bordered
  solves: the frozen-shoreline water load with rotation on the coarse
  3-D ball (TestTangent's WaterLoadFrozen3D), for a displacement and a
  strict sea-level observation through all four slots.
*/

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <numbers>
#include <vector>

#include "AdGIA.hpp"
#include "MixedProblemTestCommon.hpp"
#include "mfem.hpp"

using namespace mfem;
using namespace AdGIA;

namespace {
constexpr real_t kKappa0 = 5.4, kMu0 = 2.7, kEta0 = 2.7, kP0 = 0.05;
constexpr real_t kPi = std::numbers::pi;
void Traction2D(const Vector& x, Vector& t) {
  t.SetSize(2);
  t[0] = 0.0;
  t[1] = -kP0 * std::cos(2.0 * kPi * x[0]);
}
void Station2D(const Vector& x, Vector& v) {
  constexpr real_t s = 0.08;
  auto window = [](real_t x0, real_t c) {
    return std::exp(-(x0 - c) * (x0 - c) / (s * s)) / (s * std::sqrt(kPi));
  };
  v.SetSize(2);
  v[0] = 0.0;
  v[1] = window(x[0], 0.55) - window(x[0], 0.2);
}
}  // namespace

TEST(DiscreteAdjoint, PairsWithTangent2D) {
  const int n_el = 12, order = 2, n_steps = 32;
  const real_t t_final = 2.0;
  Mesh mesh = Mesh::MakeCartesian2D(n_el, n_el, Element::QUADRILATERAL);
  H1_FECollection fec(order, 2);
  FiniteElementSpace fes(&mesh, &fec, 2), sfes(&mesh, &fec);
  Array<int> top(mesh.bdr_attributes.Max()), bottom(mesh.bdr_attributes.Max());
  top = 0;
  bottom = 0;
  top[2] = 1;
  bottom[0] = 1;
  VectorFunctionCoefficient traction(2, Traction2D), station(2, Station2D);
  LinearForm g(&fes);
  g.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(station), bottom);
  g.Assemble();
  Vector zero_vec(2);
  zero_vec = 0.0;
  VectorConstantCoefficient zero_traction(zero_vec);
  ConstantCoefficient kappa0(kKappa0), mu0(kMu0), eta0(kEta0), chi_all(1.0);
  FunctionCoefficient chi_blob([](const Vector& x) {
    const real_t r2 = (x[0] - 0.4) * (x[0] - 0.4) + (x[1] - 0.6) * (x[1] - 0.6);
    return std::exp(-r2 / 0.015);
  });
  const real_t dt = t_final / n_steps;
  auto cont_weight = [t_final](real_t t) { return t / t_final; };

  RatioCoefficient tau0(eta0, mu0);
  auto rheology0 = IsotropicMaxwellRheology::Maxwell(2, kappa0, mu0, tau0);
  LinearQuasiStaticTractionProblem prob_fwd(&fes, rheology0, traction, top);
  ViscoelasticOperator visco_fwd(prob_fwd);
  ForwardTrajectory trajectory(visco_fwd, true);
  {
    ExponentialTrapezoidSolver ode;
    ode.Init(visco_fwd);
    Vector m(visco_fwd.Height());
    m = 0.0;
    real_t t = 0.0, dts = dt;
    trajectory.Record(m, t);
    for (int n = 1; n <= n_steps; n++) {
      ode.Step(m, t, dts);
      trajectory.Record(m, t);
    }
  }
  AdjointForce force_end, force_mid, force_cont;
  force_end.AddObservation(trajectory.Time(n_steps), g);
  force_mid.AddObservation(trajectory.Time(n_steps / 2), g);
  force_cont.AddContinuous([&](real_t t, Vector& f) { f.Add(cont_weight(t), g); });
  const AdjointForce* forces[] = {&force_end, &force_mid, &force_cont};
  const char* fnames[] = {"final-time", "interior", "continuous"};

  std::vector<ProductCoefficient> dirs;
  dirs.emplace_back(kappa0, chi_blob);
  dirs.emplace_back(mu0, chi_all);
  dirs.emplace_back(eta0, chi_blob);
  const char* dnames[] = {"kappa blob", "mu all", "eta blob"};

  // The tangents along the three directions, once.
  std::vector<std::unique_ptr<TangentTrajectory>> tangents;
  std::vector<TangentDirection> tdirs(3);
  for (int p = 0; p < 3; p++) {
    tdirs[p].mu.assign(1, nullptr);
    tdirs[p].eta.assign(1, nullptr);
    if (p == 0) tdirs[p].kappa = &dirs[0];
    if (p == 1) tdirs[p].mu[0] = &dirs[1];
    if (p == 2) tdirs[p].eta[0] = &dirs[2];
  }
  LinearQuasiStaticTractionProblem tan_twin(&fes, rheology0, zero_traction, top);
  std::vector<real_t> dJ_tan[3];
  for (int p = 0; p < 3; p++) {
    TangentLinearSolver tangent(tan_twin, trajectory);
    tangent.SetDirection(tdirs[p]);
    ASSERT_TRUE(tangent.Run());
    for (int f = 0; f < 3; f++) dJ_tan[p].push_back(tangent.Pair(*forces[f]));
  }

  LinearQuasiStaticTractionProblem twin(&fes, rheology0, zero_traction, top);
  LinearQuasiStaticTractionProblem source_twin(&fes, rheology0, zero_traction,
                                               top);
  ViscoelasticOperator visco_cont(twin);
  for (int f = 0; f < 3; f++) {
    DiscreteAdjointSolver discrete(twin, source_twin, trajectory);
    discrete.SetForce(*forces[f]);
    ASSERT_TRUE(discrete.Run());
    // The continuous adjoint's kernels for comparison.
    IsotropicKernel K_kappa(IsotropicKernel::Parameter::Kappa, sfes);
    IsotropicKernel K_mu(IsotropicKernel::Parameter::Mu, sfes);
    IsotropicKernel K_eta(IsotropicKernel::Parameter::Eta, sfes, &tau0);
    K_kappa.AddDirection(dirs[0]);
    K_mu.AddDirection(dirs[1]);
    K_eta.AddDirection(dirs[2]);
    {
      LinearQuasiStaticTractionProblem adj_twin(&fes, rheology0,
                                                zero_traction, top);
      ViscoelasticOperator visco_adj(adj_twin);
      AdjointSolver adjoint(visco_adj, trajectory);
      adjoint.SetForce(*forces[f]);
      adjoint.AddKernel(K_kappa);
      adjoint.AddKernel(K_mu);
      adjoint.AddKernel(K_eta);
      ASSERT_TRUE(adjoint.Run());
    }
    const real_t cont[] = {K_kappa.Directional(0), K_mu.Directional(0),
                           K_eta.Directional(0)};
    for (int p = 0; p < 3; p++) {
      const real_t dJ = discrete.Directional(tdirs[p]);
      if (std::getenv("ADGIA_TEST_VERBOSE")) {
        std::cout << fnames[f] << ", " << dnames[p] << ": discrete " << dJ
                  << " tangent " << dJ_tan[p][f] << " continuous " << cont[p]
                  << "\n";
      }
      // The identity: exact to solver tolerance.
      EXPECT_NEAR(dJ / dJ_tan[p][f], 1.0, 1e-8) << fnames[f] << " " << dnames[p];
      // The continuous adjoint within its O(dt^2) distance.
      EXPECT_NEAR(cont[p] / dJ, 1.0, 5e-3) << fnames[f] << " " << dnames[p];
    }
  }
}

// Static, elastic unit square (TestTangent's Static2D): the discrete
// adjoint is one solve of the twin with the station dual, paired with the
// Born force of the direction on the forward displacement. The tangent
// is the transposed solve, so the two agree to solver tolerance; the
// static continuous adjoint is the same solve paired through the kernel
// densities on the elastic forms' own rule, so it agrees as closely.
TEST(DiscreteAdjoint, Static2D) {
  const int n_el = 12, order = 2;
  Mesh mesh = Mesh::MakeCartesian2D(n_el, n_el, Element::QUADRILATERAL);
  H1_FECollection fec(order, 2);
  FiniteElementSpace fes(&mesh, &fec, 2), sfes(&mesh, &fec);
  Array<int> top(mesh.bdr_attributes.Max()), bottom(mesh.bdr_attributes.Max());
  top = 0;
  bottom = 0;
  top[2] = 1;
  bottom[0] = 1;
  VectorFunctionCoefficient traction(2, Traction2D), station(2, Station2D);
  LinearForm g(&fes);
  g.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(station), bottom);
  g.Assemble();
  Vector zero_vec(2);
  zero_vec = 0.0;
  VectorConstantCoefficient zero_traction(zero_vec);
  ConstantCoefficient kappa0(kKappa0), mu0(kMu0), chi_all(1.0);
  FunctionCoefficient chi_blob([](const Vector& x) {
    const real_t r2 = (x[0] - 0.4) * (x[0] - 0.4) + (x[1] - 0.6) * (x[1] - 0.6);
    return std::exp(-r2 / 0.015);
  });

  IsotropicElasticRheology rheology0(2, kappa0, mu0);
  GridFunction u(&fes);
  {
    LinearQuasiStaticTractionProblem prob(&fes, rheology0, traction, top);
    prob.AssembleForce(0.0);
    ASSERT_TRUE(prob.Solve());
    u = prob.Displacement();
  }
  LinearQuasiStaticTractionProblem twin(&fes, rheology0, zero_traction, top);
  AdjointForce force;
  force.AddObservation(0.0, g);

  ProductCoefficient d_kappa(kappa0, chi_blob), d_mu(mu0, chi_all);
  IsotropicKernel K_kappa(IsotropicKernel::Parameter::Kappa, sfes);
  IsotropicKernel K_mu(IsotropicKernel::Parameter::Mu, sfes);
  K_kappa.AddDirection(d_kappa);
  K_mu.AddDirection(d_mu);
  {
    AdjointSolver adjoint(twin, u);
    adjoint.SetForce(force);
    adjoint.AddKernel(K_kappa);
    adjoint.AddKernel(K_mu);
    ASSERT_TRUE(adjoint.Run());
  }

  DiscreteAdjointSolver discrete(twin, u);
  discrete.SetForce(force);
  ASSERT_TRUE(discrete.Run());
  ASSERT_EQ(discrete.NumStaticDuals(), 1);
  const char* dnames[] = {"kappa blob", "mu all"};
  const real_t cont[] = {K_kappa.Directional(0), K_mu.Directional(0)};
  for (int p = 0; p < 2; p++) {
    TangentDirection dir;
    if (p == 0) dir.kappa = &d_kappa;
    if (p == 1) dir.mu_longterm = &d_mu;
    const real_t dJ = discrete.Directional(dir);
    TangentLinearSolver tangent(twin, u);
    tangent.SetDirection(dir);
    ASSERT_TRUE(tangent.Run());
    const real_t dJ_tan = tangent.Pair(force);
    if (std::getenv("ADGIA_TEST_VERBOSE")) {
      std::cout << "static, " << dnames[p] << ": discrete " << dJ
                << " tangent " << dJ_tan << " continuous " << cont[p]
                << " | discrete/tangent - 1 = " << dJ / dJ_tan - 1.0
                << ", discrete/continuous - 1 = " << dJ / cont[p] - 1.0
                << "\n";
    }
    // Measured 5 Nov 2026: discrete/tangent - 1 = 4.5e-10 (kappa blob),
    // -3.6e-12 (mu all); the kappa-blob row is a round-off floor of the
    // two solves (the pairing cancels 25-fold, and tighter solver
    // tolerances, 1e-13 to 1e-15, leave it between 1e-10 and 5e-10).
    // Discrete/continuous - 1 = 1.5e-14 and 4.9e-14: the same solve.
    EXPECT_NEAR(dJ / dJ_tan, 1.0, 1e-9) << dnames[p];
    EXPECT_NEAR(dJ / cont[p], 1.0, 1e-8) << dnames[p];
  }
}

// Static, gravitating (TestTangent's GravitatingStatic2D: the canned
// two-layer disc, the mixed self-gravitating class, a windowed radial
// station on the surface): the identity along kappa and mu blobs.
TEST(DiscreteAdjoint, GravitatingStatic2D) {
  using namespace self_grav_test;
  const int dim = 2, order = 2;
  Mesh parent(MeshFile(dim).c_str(), 1, 1);
  SubMesh body(SubMesh::CreateFromDomain(parent, BodyMarker(parent)));
  H1_FECollection fec(order, dim);
  FiniteElementSpace fes_u(&body, &fec, dim), fes_phi(&parent, &fec);
  ConstantCoefficient rho(kRho), kappa0(kKappa), mu0(kMu);
  FunctionCoefficient sigma(SurfaceLoad);
  Array<int> surface = SurfaceMarker(body);
  // A radial station at the angle 0.3.
  VectorFunctionCoefficient station(dim, [](const Vector& x, Vector& v) {
    const real_t th = std::atan2(x[1], x[0]), w = 0.15;
    const real_t g = std::exp(-(th - 0.3) * (th - 0.3) / (w * w));
    const real_t r = std::max(x.Norml2(), 1e-300);
    v.SetSize(2);
    v[0] = g * x[0] / r;
    v[1] = g * x[1] / r;
  });
  LinearForm g(&fes_u);
  g.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(station), surface);
  g.Assemble();

  IsotropicElasticRheology rheology0(dim, kappa0, mu0);
  GridFunction u(&fes_u);
  {
    LinearQuasiStaticMixedSelfGravitatingProblem prob(
        &fes_u, &fes_phi, rheology0, rho, kG, kDtNDegree);
    prob.SetSurfaceLoad(sigma, surface);
    prob.SetRelTol(1e-12);
    prob.AssembleForce(0.0);
    ASSERT_TRUE(prob.Solve());
    u = prob.Displacement();
  }
  FunctionCoefficient chi([](const Vector& x) {
    const real_t dx = x[0] - 0.3, dy = x[1] - 0.5;
    return std::exp(-(dx * dx + dy * dy) / 0.05);
  });
  ProductCoefficient d_kappa(kappa0, chi), d_mu(mu0, chi);
  AdjointForce force;
  force.AddObservation(0.0, g);

  LinearQuasiStaticMixedSelfGravitatingProblem twin(
      &fes_u, &fes_phi, rheology0, rho, kG, kDtNDegree);
  twin.SetRelTol(1e-12);
  DiscreteAdjointSolver discrete(twin, u);
  discrete.SetForce(force);
  ASSERT_TRUE(discrete.Run());
  const char* dnames[] = {"kappa blob", "mu blob"};
  for (int p = 0; p < 2; p++) {
    TangentDirection dir;
    if (p == 0) dir.kappa = &d_kappa;
    if (p == 1) dir.mu_longterm = &d_mu;
    const real_t dJ = discrete.Directional(dir);
    TangentLinearSolver tangent(twin, u);
    tangent.SetDirection(dir);
    ASSERT_TRUE(tangent.Run());
    const real_t dJ_tan = tangent.Pair(force);
    if (std::getenv("ADGIA_TEST_VERBOSE")) {
      std::cout << "gravitating static, " << dnames[p] << ": discrete " << dJ
                << " tangent " << dJ_tan << " | discrete/tangent - 1 = "
                << dJ / dJ_tan - 1.0 << "\n";
    }
    EXPECT_NEAR(dJ / dJ_tan, 1.0, 1e-8) << dnames[p];
  }
}

// The frozen-shoreline water load with rotation (TestTangent's
// WaterLoadFrozen3D, kept light: the eta blob direction, 8 steps): the
// bordered solve is symmetric, so the transposed sweep carries the
// uniform-term and omega rows unchanged. A radial displacement cap and
// the strict sea-level functional through all four slots (u, phi,
// uniform, omega), both at the final time.
namespace {
constexpr real_t kRhoW = 1.0;
constexpr real_t kOmega = 0.4;
const real_t kG0 = 4.0 * kPi * self_grav_test::kG * self_grav_test::kRho / 3.0;

real_t MeltCap(const Vector& x, const Vector& n) {
  constexpr real_t s2 = 0.08;
  Vector d(x);
  d /= x.Norml2();
  d -= n;
  return 0.02 * std::exp(-(d * d) / s2);
}
Vector UnitDir(real_t a, real_t b, real_t c) {
  Vector n(3);
  n[0] = a;
  n[1] = b;
  n[2] = c;
  n /= n.Norml2();
  return n;
}
}  // namespace

TEST(DiscreteAdjoint, WaterLoadFrozen3D) {
  const int order = 2, n_steps = 8;
  const real_t t1 = 1.5;
  Mesh parent(self_grav_test::MeshFile(3).c_str(), 1, 1);
  SubMesh body(SubMesh::CreateFromDomain(parent,
                                         self_grav_test::BodyMarker(parent)));
  H1_FECollection fec(order, 3);
  FiniteElementSpace fes_u(&body, &fec, 3), fes_phi(&parent, &fec);
  ConstantCoefficient rho(self_grav_test::kRho);
  Array<int> surface = self_grav_test::SurfaceMarker(body);
  Array<int> parent_surface(parent.bdr_attributes.Max());
  parent_surface = 0;
  parent_surface[0] = 1;
  Vector moments(3);
  moments[0] = 1.2;
  moments[1] = 1.3;
  moments[2] = 2.0;
  ConstantCoefficient w(kRhoW / kG0);
  ConstantCoefficient kappa0(self_grav_test::kKappa);
  ConstantCoefficient mu0(self_grav_test::kMu);
  ConstantCoefficient eta0(self_grav_test::kMu);
  RatioCoefficient tau0(eta0, mu0);

  FunctionCoefficient sigma_data([](const Vector& x) {
    const real_t r = x.Norml2(), c = x[2] / r;
    return 0.02 * (3.0 * c * c - 0.5 + 0.5 * x[0] * x[2] / (r * r));
  });
  ConstantCoefficient zero_data(0.0);

  // Duals: radial displacement cap at n1; the strict sea-level
  // functional at the cap n3 through all four slots.
  const Vector n1 = UnitDir(0.3, 0.5, 0.81), n3 = UnitDir(0.1, -0.6, 0.79);
  VectorFunctionCoefficient h_u(3, [n1](const Vector& x, Vector& v) {
    v = x;
    v *= MeltCap(x, n1) / x.Norml2();
  });
  LinearForm g_u(&fes_u);
  g_u.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(h_u), surface);
  g_u.Assemble();
  FunctionCoefficient h_sl([n3](const Vector& x) { return MeltCap(x, n3); });
  VectorFunctionCoefficient h_sl_u(3, [n3](const Vector& x, Vector& v) {
    v = x;
    v *= -MeltCap(x, n3);
  });
  LinearForm g_sl_u(&fes_u);
  g_sl_u.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(h_sl_u), surface);
  g_sl_u.Assemble();
  ProductCoefficient h_sl_phi(-1.0 / kG0, h_sl);
  LinearForm g_sl_phi(&fes_phi);
  g_sl_phi.AddBoundaryIntegrator(new BoundaryLFIntegrator(h_sl_phi),
                                 parent_surface);
  g_sl_phi.Assemble();
  const real_t sl_uniform = -g_sl_phi.Sum();
  Vector sl_omega(3);
  for (int k = 0; k < 3; k++) {
    CentrifugalPotential psi_k(3, kOmega);
    psi_k.SetUnit(k);
    ProductCoefficient hpsi(h_sl, psi_k);
    LinearForm lf(&fes_phi);
    lf.AddBoundaryIntegrator(new BoundaryLFIntegrator(hpsi), parent_surface);
    lf.Assemble();
    sl_omega[k] = -lf.Sum() / kG0;
  }

  const real_t dt = t1 / n_steps;
  auto rheology0 = IsotropicMaxwellRheology::Maxwell(3, kappa0, mu0, tau0);
  // Every problem carries the water feedback and the rotation; the twins
  // with a zero data load. The solves run at the class default tolerance
  // 1e-12 (TestTangent's 1e-11 leaves the sea-level row at 1.0e-8): the
  // identity follows the solver tolerance, measured 5 Nov 2026 at
  // 9.4e-8 / 8.9e-7 (u / sea level) for 1e-10, 2.3e-9 / 1.0e-8 for 1e-11,
  // 1.0e-9 / 4.1e-9 for 1e-12 and 5.4e-11 / 1.5e-9 for 1e-13.
  auto make_problem = [&](Coefficient& data) {
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        &fes_u, &fes_phi, rheology0, rho, self_grav_test::kG,
        self_grav_test::kDtNDegree);
    p->SetWaterLoad(w, data, surface);
    p->SetRotation(kOmega, moments);
    p->SetRelTol(1e-12);
    return p;
  };

  auto prob_fwd = make_problem(sigma_data);
  ViscoelasticOperator visco_fwd(*prob_fwd);
  ForwardTrajectory trajectory(visco_fwd, true);
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

  AdjointForce force_u, force_sl;
  force_u.AddObservation(trajectory.Time(n_steps), g_u);
  AdjointLoad sl;
  sl.u = &g_sl_u;
  sl.phi = &g_sl_phi;
  sl.uniform = sl_uniform;
  sl.omega = &sl_omega;
  force_sl.AddObservation(trajectory.Time(n_steps), sl);
  const AdjointForce* forces[] = {&force_u, &force_sl};
  const char* fnames[] = {"displacement", "sea level"};

  FunctionCoefficient chi_blob3([](const Vector& x) {
    const real_t dx = x[0] - 0.3, dy = x[1] - 0.35, dz = x[2] - 0.4;
    return std::exp(-(dx * dx + dy * dy + dz * dz) / 0.03);
  });
  ProductCoefficient d_eta(eta0, chi_blob3);
  TangentDirection dir;
  dir.eta.assign(1, &d_eta);

  real_t dJ_tan[2];
  {
    auto tan_twin = make_problem(zero_data);
    TangentLinearSolver tangent(*tan_twin, trajectory);
    tangent.SetDirection(dir);
    ASSERT_TRUE(tangent.Run());
    for (int f = 0; f < 2; f++) dJ_tan[f] = tangent.Pair(*forces[f]);
  }

  auto twin = make_problem(zero_data);
  auto source_twin = make_problem(zero_data);
  DiscreteAdjointSolver discrete(*twin, *source_twin, trajectory);
  for (int f = 0; f < 2; f++) {
    discrete.SetForce(*forces[f]);
    ASSERT_TRUE(discrete.Run());
    const real_t dJ = discrete.Directional(dir);
    if (std::getenv("ADGIA_TEST_VERBOSE")) {
      std::cout << "water, eta blob, " << fnames[f] << ": discrete " << dJ
                << " tangent " << dJ_tan[f] << " | discrete/tangent - 1 = "
                << dJ / dJ_tan[f] - 1.0 << "\n";
    }
    EXPECT_NEAR(dJ / dJ_tan[f], 1.0, 1e-8) << fnames[f];
  }
}

// The kernels as dual vectors on the internal scalar space: paired with
// a nodal direction they reproduce Directional() along that direction read
// as a GridFunctionCoefficient — exactly, since both sides then see the
// same function (affine quads: the quadrature is exact).
TEST(DiscreteAdjoint, KernelDualsPairExactly2D) {
  const int n_el = 12, order = 2, n_steps = 16;
  const real_t t_final = 2.0;
  Mesh mesh = Mesh::MakeCartesian2D(n_el, n_el, Element::QUADRILATERAL);
  H1_FECollection fec(order, 2);
  FiniteElementSpace fes(&mesh, &fec, 2);
  Array<int> top(mesh.bdr_attributes.Max()), bottom(mesh.bdr_attributes.Max());
  top = 0;
  bottom = 0;
  top[2] = 1;
  bottom[0] = 1;
  VectorFunctionCoefficient traction(2, Traction2D), station(2, Station2D);
  LinearForm g(&fes);
  g.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(station), bottom);
  g.Assemble();
  Vector zero_vec(2);
  zero_vec = 0.0;
  VectorConstantCoefficient zero_traction(zero_vec);
  ConstantCoefficient kappa0(kKappa0), mu0(kMu0), eta0(kEta0);
  const real_t dt = t_final / n_steps;
  RatioCoefficient tau0(eta0, mu0);
  auto rheology0 = IsotropicMaxwellRheology::Maxwell(2, kappa0, mu0, tau0);
  LinearQuasiStaticTractionProblem prob_fwd(&fes, rheology0, traction, top);
  ViscoelasticOperator visco_fwd(prob_fwd);
  ForwardTrajectory trajectory(visco_fwd, true);
  {
    ExponentialTrapezoidSolver ode;
    ode.Init(visco_fwd);
    Vector m(visco_fwd.Height());
    m = 0.0;
    real_t t = 0.0, dts = dt;
    trajectory.Record(m, t);
    for (int n = 1; n <= n_steps; n++) {
      ode.Step(m, t, dts);
      trajectory.Record(m, t);
    }
  }
  AdjointForce force;
  force.AddObservation(trajectory.Time(n_steps), g);
  force.AddContinuous([&](real_t t, Vector& f) { f.Add(t / t_final, g); });

  LinearQuasiStaticTractionProblem twin(&fes, rheology0, zero_traction, top);
  LinearQuasiStaticTractionProblem source_twin(&fes, rheology0, zero_traction,
                                               top);
  DiscreteAdjointSolver discrete(twin, source_twin, trajectory);
  discrete.SetForce(force);
  ASSERT_TRUE(discrete.Run());
  using P = DiscreteAdjointSolver::Parameter;
  const int i_kappa = discrete.AddKernel(P::Kappa);
  const int i_mu = discrete.AddKernel(P::Mu, 0);
  const int i_eta = discrete.AddKernel(P::Eta, 0);
  discrete.AssembleKernels();

  // Nodal directions on the internal scalar space: a blob interpolated
  // there, read back as a coefficient for Directional().
  FiniteElementSpace& sfes = discrete.Operator().InternalScalarSpace();
  FunctionCoefficient blob([](const Vector& x) {
    const real_t r2 = (x[0] - 0.4) * (x[0] - 0.4) + (x[1] - 0.6) * (x[1] - 0.6);
    return std::exp(-r2 / 0.03);
  });
  GridFunction chi(&sfes);
  chi.ProjectCoefficient(blob);
  GridFunctionCoefficient chi_c(&chi);
  ProductCoefficient d_kappa(kappa0, chi_c), d_mu(mu0, chi_c), d_eta(eta0, chi_c);
  GridFunction dk(&sfes), dm(&sfes), de(&sfes);
  dk.ProjectCoefficient(d_kappa);
  dm.ProjectCoefficient(d_mu);
  de.ProjectCoefficient(d_eta);
  TangentDirection dir_k, dir_m, dir_e;
  dir_k.kappa = &d_kappa;
  dir_m.mu.assign(1, &d_mu);
  dir_m.eta.assign(1, nullptr);
  dir_e.mu.assign(1, nullptr);
  dir_e.eta.assign(1, &d_eta);
  const real_t dJ_k = discrete.Directional(dir_k), pk = discrete.Pair(i_kappa, dk);
  const real_t dJ_m = discrete.Directional(dir_m), pm = discrete.Pair(i_mu, dm);
  const real_t dJ_e = discrete.Directional(dir_e), pe = discrete.Pair(i_eta, de);
  if (std::getenv("ADGIA_TEST_VERBOSE")) {
    std::cout << "kernel duals: kappa " << pk << " vs " << dJ_k << " | mu " << pm
              << " vs " << dJ_m << " | eta " << pe << " vs " << dJ_e << "\n";
  }
  EXPECT_NEAR(pk / dJ_k, 1.0, 1e-9);
  EXPECT_NEAR(pm / dJ_m, 1.0, 1e-9);
  EXPECT_NEAR(pe / dJ_e, 1.0, 1e-9);
}
