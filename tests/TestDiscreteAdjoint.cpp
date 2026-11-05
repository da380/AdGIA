/*
  Tests for the discrete first-order adjoint (discrete_adjoint.hpp): the
  transpose of the tangent. The gate is the pairing identity
  <g, du>_tangent = dJ_discrete along every direction, which holds to
  solver tolerance (no finite differences, no time-quadrature floor), on
  the adjoint tests' unit-square Maxwell configuration for a final-time,
  an interior-time and a continuous-in-time observation; and the
  comparison with the continuous adjoint's kernels, which agree to their
  O(dt^2) distance only.
*/

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <numbers>
#include <vector>

#include "AdGIA.hpp"
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
