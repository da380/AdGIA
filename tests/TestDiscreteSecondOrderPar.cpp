/*
  Parallel test for the discrete second-order adjoint
  (discrete_second_order.hpp). Run with 1, 2 and 4 ranks; a standalone
  MPI program that exits non-zero if any check fails. The serial
  TestDiscreteSecondOrder.SymmetricAndMatchesFD2D on the unit-square
  Maxwell body: H[v, delta] against central finite differences of the
  parallel discrete gradient (DiscreteAdjointSolver::Directional at
  x +- eps v) and the symmetry H[v, w] = H[w, v], for v, w among kappa,
  mu and eta blobs, final-time and continuous observations. The FD
  gradient is global (TestDiscreteAdjointPar), so the agreement checks
  that the second order's pairings count every shared displacement dof
  once and reduce the internal-space sums over ranks on any partition.
*/
#include <mpi.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

#include "AdGIA.hpp"
#include "mfem.hpp"

using namespace mfem;
using namespace AdGIA;

namespace {

int num_checks = 0;
int num_fails = 0;

void Check(double err, double tol, const std::string& what) {
  num_checks++;
  if (!(err <= tol)) {
    num_fails++;
    if (Mpi::Root()) {
      std::cout << "FAIL: " << what << "  (err = " << err << ", tol = " << tol
                << ")\n";
    }
  }
}

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

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();
  const int dim = 2, n_el = 12, order = 2, n_steps = 16;
  const real_t t_final = 2.0;
  const real_t dt = t_final / n_steps;

  auto serial_mesh = Mesh::MakeCartesian2D(n_el, n_el, Element::QUADRILATERAL);
  auto mesh = ParMesh(MPI_COMM_WORLD, serial_mesh);
  serial_mesh.Clear();
  auto fec = H1_FECollection(order, dim);
  auto fes = ParFiniteElementSpace(&mesh, &fec, dim);
  Array<int> top(mesh.bdr_attributes.Max()), bottom(mesh.bdr_attributes.Max());
  top = 0;
  bottom = 0;
  top[2] = 1;
  bottom[0] = 1;
  auto traction = VectorFunctionCoefficient(dim, Traction2D);
  auto station = VectorFunctionCoefficient(dim, Station2D);
  auto g = ParLinearForm(&fes);
  g.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(station), bottom);
  g.Assemble();
  Vector zero_vec(dim);
  zero_vec = 0.0;
  auto zero_traction = VectorConstantCoefficient(zero_vec);
  auto kappa0 = ConstantCoefficient(kKappa0);
  auto mu0 = ConstantCoefficient(kMu0);
  auto eta0 = ConstantCoefficient(kEta0);
  auto chi_a = FunctionCoefficient([](const Vector& x) {
    const real_t r2 = (x[0] - 0.4) * (x[0] - 0.4) + (x[1] - 0.6) * (x[1] - 0.6);
    return std::exp(-r2 / 0.03);
  });
  auto chi_b = FunctionCoefficient([](const Vector& x) {
    const real_t r2 = (x[0] - 0.6) * (x[0] - 0.3) + (x[1] - 0.3) * (x[1] - 0.3);
    return std::exp(-r2 / 0.03);
  });

  auto trajectory_of = [&](ViscoelasticOperator& visco) {
    auto traj = std::make_unique<ForwardTrajectory>(visco, true);
    ExponentialTrapezoidSolver ode;
    ode.Init(visco);
    Vector m(visco.Height());
    m = 0.0;
    real_t t = 0.0, dts = dt;
    traj->Record(m, t);
    for (int n = 1; n <= n_steps; n++) {
      ode.Step(m, t, dts);
      traj->Record(m, t);
    }
    return traj;
  };
  ProductCoefficient d_kappa(kappa0, chi_a), d_mu(mu0, chi_b), d_eta(eta0, chi_a);
  auto make_dir = [&](int p) {
    TangentDirection d;
    d.mu.assign(1, nullptr);
    d.eta.assign(1, nullptr);
    if (p == 0) d.kappa = &d_kappa;
    if (p == 1) d.mu[0] = &d_mu;
    if (p == 2) d.eta[0] = &d_eta;
    return d;
  };
  Coefficient* chi_of[] = {&chi_a, &chi_b, &chi_a};
  const char* names[] = {"kappa", "mu", "eta"};
  struct Grad {
    real_t k = 0.0, m = 0.0, e = 0.0;
  };

  for (int which = 0; which < 2; which++) {
    const bool continuous = which == 1;
    const std::string fname = continuous ? "continuous" : "final-time";
    auto make_force = [&](const ForwardTrajectory& traj) {
      AdjointForce f;
      if (continuous) {
        f.AddContinuous([&](real_t t, Vector& v) { v.Add(t / t_final, g); });
      } else {
        f.AddObservation(traj.Time(n_steps), g);
      }
      return f;
    };
    // The parallel discrete gradient at a parameter set.
    auto gradient = [&](Coefficient& kappa, Coefficient& mu, Coefficient& eta) {
      RatioCoefficient tau(eta, mu);
      auto rheology = IsotropicMaxwellRheology::Maxwell(dim, kappa, mu, tau);
      LinearQuasiStaticTractionProblem prob(&fes, rheology, traction, top);
      ViscoelasticOperator visco(prob);
      auto traj = trajectory_of(visco);
      AdjointForce force = make_force(*traj);
      LinearQuasiStaticTractionProblem twin(&fes, rheology, zero_traction, top);
      LinearQuasiStaticTractionProblem stwin(&fes, rheology, zero_traction, top);
      DiscreteAdjointSolver discrete(twin, stwin, *traj);
      discrete.SetForce(force);
      if (!discrete.Run()) Check(1.0, 0.0, fname + " FD discrete run");
      return Grad{discrete.Directional(make_dir(0)), discrete.Directional(make_dir(1)),
                  discrete.Directional(make_dir(2))};
    };

    RatioCoefficient tau0(eta0, mu0);
    auto rheology0 = IsotropicMaxwellRheology::Maxwell(dim, kappa0, mu0, tau0);
    LinearQuasiStaticTractionProblem prob_fwd(&fes, rheology0, traction, top);
    ViscoelasticOperator visco_fwd(prob_fwd);
    auto trajectory = trajectory_of(visco_fwd);
    AdjointForce force = make_force(*trajectory);
    LinearQuasiStaticTractionProblem twin(&fes, rheology0, zero_traction, top);
    LinearQuasiStaticTractionProblem stwin(&fes, rheology0, zero_traction, top);
    DiscreteAdjointSolver discrete(twin, stwin, *trajectory);
    discrete.SetForce(force);
    if (!discrete.Run()) Check(1.0, 0.0, fname + " discrete run");
    std::vector<std::unique_ptr<TangentLinearSolver>> tangents;
    LinearQuasiStaticTractionProblem tan_twin(&fes, rheology0, zero_traction, top);
    for (int p = 0; p < 3; p++) {
      tangents.push_back(std::make_unique<TangentLinearSolver>(tan_twin, *trajectory));
      tangents.back()->SetDirection(make_dir(p));
      if (!tangents.back()->Run()) Check(1.0, 0.0, fname + " tangent run");
    }
    real_t H[3][3];
    for (int p = 0; p < 3; p++) {
      LinearQuasiStaticTractionProblem dtwin(&fes, rheology0, zero_traction, top);
      LinearQuasiStaticTractionProblem svtwin(&fes, rheology0, zero_traction, top);
      DiscreteSecondOrderSolver second(discrete, dtwin, svtwin, *trajectory,
                                       tangents[p]->Trajectory(), make_dir(p));
      if (!second.Run()) Check(1.0, 0.0, fname + " second-order run");
      for (int q = 0; q < 3; q++) {
        LinearQuasiStaticTractionProblem sdtwin(&fes, rheology0, zero_traction, top);
        H[p][q] = second.Directional(make_dir(q), sdtwin);
      }
    }
    const real_t eps = 1e-3;
    for (int p = 0; p < 3; p++) {
      Grad plus, minus;
      for (const real_t sg : {eps, -eps}) {
        auto kp = Bump(kappa0, *chi_of[p], p == 0 ? sg : 0.0);
        auto mp = Bump(mu0, *chi_of[p], p == 1 ? sg : 0.0);
        auto ep = Bump(eta0, *chi_of[p], p == 2 ? sg : 0.0);
        (sg > 0 ? plus : minus) = gradient(*kp.value, *mp.value, *ep.value);
      }
      const real_t fd[3] = {(plus.k - minus.k) / (2 * eps), (plus.m - minus.m) / (2 * eps),
                            (plus.e - minus.e) / (2 * eps)};
      real_t scale = 0.0;
      for (int q = 0; q < 3; q++) scale = std::max(scale, std::abs(fd[q]));
      for (int q = 0; q < 3; q++) {
        const std::string what =
            fname + " H[" + names[p] + "," + names[q] + "]";
        if (Mpi::Root() && std::getenv("ADGIA_TEST_VERBOSE")) {
          std::cout << what << " " << H[p][q] << " fd " << fd[q] << " | H["
                    << names[q] << "," << names[p] << "] " << H[q][p] << "\n";
        }
        Check(std::abs(H[p][q] - fd[q]) / scale, 1e-5, what + " against FD");
        Check(std::abs(H[p][q] - H[q][p]) / scale, 1e-9, what + " symmetry");
      }
    }
  }

  if (Mpi::Root()) {
    std::cout << "TestDiscreteSecondOrderPar: " << num_checks << " checks, "
              << num_fails << " failures on " << Mpi::WorldSize()
              << " ranks\n";
  }
  return num_fails == 0 ? 0 : 1;
}
