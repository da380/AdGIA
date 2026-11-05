/*
  Parallel tests for the discrete first-order adjoint
  (discrete_adjoint.hpp). Run with 1, 2 and 4 ranks; a standalone MPI
  program that exits non-zero if any check fails. The pairing identity
  of the serial TestDiscreteAdjoint, <g, du>_tangent = dJ_discrete, on
  the unit-square configuration: the static elastic body along kappa
  and mu, and the Maxwell body along kappa, mu and eta for a final-time
  and a continuous-in-time observation. The tangent's pairing is
  global, so the identity checks that the discrete adjoint's pairings
  count every shared degree of freedom once on any partition.
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

real_t GlobalSum(real_t local) {
  real_t global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                MPI_COMM_WORLD);
  return global;
}

constexpr real_t kKappa0 = 5.4;
constexpr real_t kMu0 = 2.7;
constexpr real_t kEta0 = 2.7;
constexpr real_t kP0 = 0.05;
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

void Report(const std::string& what, real_t dJ, real_t dJ_tan) {
  if (Mpi::Root() && std::getenv("ADGIA_TEST_VERBOSE")) {
    std::cout << what << ": discrete " << dJ << " tangent " << dJ_tan
              << " | discrete/tangent - 1 = " << dJ / dJ_tan - 1.0 << "\n";
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();
  const int dim = 2, n_el = 12, order = 2, n_steps = 32;
  const real_t t_final = 2.0;

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
  auto chi_blob = FunctionCoefficient([](const Vector& x) {
    const real_t r2 = (x[0] - 0.4) * (x[0] - 0.4) + (x[1] - 0.6) * (x[1] - 0.6);
    return std::exp(-r2 / 0.015);
  });
  auto chi_all = ConstantCoefficient(1.0);
  auto kappa0 = ConstantCoefficient(kKappa0);
  auto mu0 = ConstantCoefficient(kMu0);
  auto eta0 = ConstantCoefficient(kEta0);

  // --- static -------------------------------------------------------------
  {
    auto rheology0 = IsotropicElasticRheology(dim, kappa0, mu0);
    auto u = ParGridFunction(&fes);
    {
      auto prob = LinearQuasiStaticTractionProblem(&fes, rheology0, traction,
                                                   top);
      prob.AssembleForce(0.0);
      if (!prob.Solve()) Check(1.0, 0.0, "static forward solve");
      u = prob.Displacement();
    }
    auto twin = LinearQuasiStaticTractionProblem(&fes, rheology0,
                                                 zero_traction, top);
    AdjointForce force;
    force.AddObservation(0.0, g);
    auto discrete = DiscreteAdjointSolver(twin, u);
    discrete.SetForce(force);
    if (!discrete.Run()) Check(1.0, 0.0, "static discrete run");
    ProductCoefficient d_kappa(kappa0, chi_blob), d_mu(mu0, chi_all);
    const char* names[] = {"kappa blob", "mu all"};
    for (int p = 0; p < 2; p++) {
      TangentDirection dir;
      if (p == 0) dir.kappa = &d_kappa;
      if (p == 1) dir.mu_longterm = &d_mu;
      const real_t dJ = discrete.Directional(dir);
      TangentLinearSolver tangent(twin, u);
      tangent.SetDirection(dir);
      if (!tangent.Run()) Check(1.0, 0.0, "static tangent run");
      const real_t dJ_tan = tangent.Pair(force);
      Report(std::string("static, ") + names[p], dJ, dJ_tan);
      Check(std::abs(dJ / dJ_tan - 1.0), 1e-8,
            std::string("static identity, ") + names[p]);
      // The same pairing as the ranks' local sums: the Born force is the
      // rank's partial sums and the dual is consistent, so the sum over
      // ranks of the local products counts every shared dof once too.
      ModuliBornForce born(fes, dir, 0);
      Vector f(fes.GetVSize());
      f = 0.0;
      born.AddTo(u, f);
      const real_t dJ_local = GlobalSum(f * discrete.StaticDual(0));
      if (Mpi::Root() && std::getenv("ADGIA_TEST_VERBOSE")) {
        std::cout << "static, " << names[p] << ": local sums " << dJ_local
                  << " | local/discrete - 1 = " << dJ_local / dJ - 1.0
                  << "\n";
      }
      Check(std::abs(dJ_local / dJ - 1.0), 1e-12,
            std::string("static pairing as local sums, ") + names[p]);
    }
  }

  // --- viscoelastic ---------------------------------------------------------
  {
    const real_t dt = t_final / n_steps;
    auto cont_weight = [t_final](real_t t) { return t / t_final; };
    auto tau0 = RatioCoefficient(eta0, mu0);
    auto rheology0 = IsotropicMaxwellRheology::Maxwell(dim, kappa0, mu0, tau0);
    auto prob_fwd = LinearQuasiStaticTractionProblem(&fes, rheology0,
                                                     traction, top);
    auto visco_fwd = ViscoelasticOperator(prob_fwd);
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
    AdjointForce force_end, force_cont;
    force_end.AddObservation(trajectory.Time(n_steps), g);
    force_cont.AddContinuous([&](real_t t, Vector& f) {
      f.Add(cont_weight(t), g);
    });
    const AdjointForce* forces[] = {&force_end, &force_cont};
    const char* fnames[] = {"end", "continuous"};
    std::vector<ProductCoefficient> dirs;
    dirs.emplace_back(kappa0, chi_blob);
    dirs.emplace_back(mu0, chi_all);
    dirs.emplace_back(eta0, chi_blob);
    const char* names[] = {"kappa blob", "mu all", "eta blob"};
    std::vector<TangentDirection> tdirs(3);
    for (int p = 0; p < 3; p++) {
      tdirs[p].mu.assign(1, nullptr);
      tdirs[p].eta.assign(1, nullptr);
      if (p == 0) tdirs[p].kappa = &dirs[0];
      if (p == 1) tdirs[p].mu[0] = &dirs[1];
      if (p == 2) tdirs[p].eta[0] = &dirs[2];
    }

    // The tangents along the three directions, paired with both forces.
    auto tan_twin = LinearQuasiStaticTractionProblem(&fes, rheology0,
                                                     zero_traction, top);
    real_t dJ_tan[3][2];
    for (int p = 0; p < 3; p++) {
      TangentLinearSolver tangent(tan_twin, trajectory);
      tangent.SetDirection(tdirs[p]);
      if (!tangent.Run()) Check(1.0, 0.0, "tangent run");
      for (int f = 0; f < 2; f++) dJ_tan[p][f] = tangent.Pair(*forces[f]);
    }

    auto twin = LinearQuasiStaticTractionProblem(&fes, rheology0,
                                                 zero_traction, top);
    auto source_twin = LinearQuasiStaticTractionProblem(&fes, rheology0,
                                                        zero_traction, top);
    auto discrete = DiscreteAdjointSolver(twin, source_twin, trajectory);
    for (int f = 0; f < 2; f++) {
      discrete.SetForce(*forces[f]);
      if (!discrete.Run()) Check(1.0, 0.0, "discrete run");
      for (int p = 0; p < 3; p++) {
        const real_t dJ = discrete.Directional(tdirs[p]);
        const std::string what =
            std::string("viscoelastic, ") + fnames[f] + ", " + names[p];
        Report(what, dJ, dJ_tan[p][f]);
        Check(std::abs(dJ / dJ_tan[p][f] - 1.0), 1e-8, what + " identity");
      }
    }
  }

  if (Mpi::Root()) {
    std::cout << "TestDiscreteAdjointPar: " << num_checks << " checks, "
              << num_fails << " failures on " << Mpi::WorldSize()
              << " ranks\n";
  }
  return num_fails == 0 ? 0 : 1;
}
