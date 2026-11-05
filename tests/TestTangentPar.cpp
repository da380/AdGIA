/*
  Parallel tests for the tangent-linear problem (tangent.hpp). Run with
  1, 2 and 4 ranks; a standalone MPI program that exits non-zero if any
  check fails. The partition-independent gates of the serial TestTangent
  on the unit-square configuration: the static kappa/mu tangents and the
  viscoelastic kappa/mu/eta tangents for a final-time and a
  continuous-in-time observation, against central finite differences of
  fresh forward solves (the identity holds per run on any partition).
*/
#include <mpi.h>

#include <cmath>
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
  const real_t eps = 1e-3;

  // --- static -------------------------------------------------------------
  {
    auto measure = [&](Coefficient& kappa, Coefficient& mu,
                       GridFunction* keep = nullptr) {
      auto rheology = IsotropicElasticRheology(dim, kappa, mu);
      auto prob = LinearQuasiStaticTractionProblem(&fes, rheology, traction,
                                                   top);
      prob.AssembleForce(0.0);
      if (!prob.Solve()) Check(1.0, 0.0, "static forward solve");
      if (keep) *keep = prob.Displacement();
      return GlobalSum(g * prob.Displacement());
    };
    ParGridFunction u(&fes);
    measure(kappa0, mu0, &u);
    auto rheology0 = IsotropicElasticRheology(dim, kappa0, mu0);
    auto twin = LinearQuasiStaticTractionProblem(&fes, rheology0,
                                                 zero_traction, top);
    AdjointForce force;
    force.AddObservation(0.0, g);
    ProductCoefficient d_kappa(kappa0, chi_blob), d_mu(mu0, chi_all);
    for (int which = 0; which < 2; which++) {
      TangentLinearSolver tan(twin, u);
      TangentDirection dir;
      if (which == 0) dir.kappa = &d_kappa; else dir.mu_longterm = &d_mu;
      tan.SetDirection(dir);
      if (!tan.Run()) Check(1.0, 0.0, "static tangent run");
      const real_t dJ = tan.Pair(force);
      Coefficient& chi = which == 0 ? static_cast<Coefficient&>(chi_blob)
                                    : static_cast<Coefficient&>(chi_all);
      auto kp = Bump(kappa0, chi, which == 0 ? eps : 0.0);
      auto mp = Bump(mu0, chi, which == 0 ? 0.0 : eps);
      auto km = Bump(kappa0, chi, which == 0 ? -eps : 0.0);
      auto mm = Bump(mu0, chi, which == 0 ? 0.0 : -eps);
      const real_t fd = (measure(*kp.value, *mp.value) -
                         measure(*km.value, *mm.value)) / (2.0 * eps);
      Check(std::abs(dJ / fd - 1.0), 1e-5,
            which == 0 ? "static tangent kappa blob" : "static tangent mu all");
    }
  }

  // --- viscoelastic ---------------------------------------------------------
  {
    const real_t dt = t_final / n_steps;
    auto cont_weight = [t_final](real_t t) { return t / t_final; };
    struct Measured {
      real_t J_end = 0.0, J_cont = 0.0;
    };
    auto run_forward = [&](Coefficient& kappa, Coefficient& mu,
                           Coefficient& eta) {
      auto tau = RatioCoefficient(eta, mu);
      auto rheology = IsotropicMaxwellRheology::Maxwell(dim, kappa, mu, tau);
      auto prob = LinearQuasiStaticTractionProblem(&fes, rheology, traction,
                                                   top);
      auto visco = ViscoelasticOperator(prob);
      ExponentialTrapezoidSolver ode;
      ode.Init(visco);
      Vector m(visco.Height());
      m = 0.0;
      real_t t = 0.0, dt_step = dt;
      Measured out;
      auto sample = [&](int n) {
        if (!visco.SolveElastic(m, t)) Check(1.0, 0.0, "forward solve");
        const real_t Jt = GlobalSum(g * prob.Displacement());
        const real_t w = (n == 0 || n == n_steps) ? 0.5 * dt : dt;
        out.J_cont += w * cont_weight(t) * Jt;
        if (n == n_steps) out.J_end = Jt;
      };
      sample(0);
      for (int n = 1; n <= n_steps; n++) {
        ode.Step(m, t, dt_step);
        sample(n);
      }
      return out;
    };
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
    auto prob_twin = LinearQuasiStaticTractionProblem(&fes, rheology0,
                                                      zero_traction, top);
    AdjointForce force_end, force_cont;
    force_end.AddObservation(trajectory.Time(n_steps), g);
    force_cont.AddContinuous([&](real_t t, Vector& f) {
      f.Add(cont_weight(t), g);
    });
    std::vector<ProductCoefficient> dirs;
    dirs.emplace_back(kappa0, chi_blob);
    dirs.emplace_back(mu0, chi_all);
    dirs.emplace_back(eta0, chi_blob);
    const char* names[] = {"kappa blob", "mu all", "eta blob"};
    Coefficient* chi_of[] = {&chi_blob, &chi_all, &chi_blob};
    for (int p = 0; p < 3; p++) {
      TangentLinearSolver tangent(prob_twin, trajectory);
      TangentDirection dir;
      dir.mu.assign(1, nullptr);
      dir.eta.assign(1, nullptr);
      if (p == 0) dir.kappa = &dirs[0];
      if (p == 1) dir.mu[0] = &dirs[1];
      if (p == 2) dir.eta[0] = &dirs[2];
      tangent.SetDirection(dir);
      if (!tangent.Run()) Check(1.0, 0.0, "tangent run");
      const real_t dJ_end = tangent.Pair(force_end);
      const real_t dJ_cont = tangent.Pair(force_cont);
      Measured plus, minus;
      for (const real_t sg : {eps, -eps}) {
        auto kp = Bump(kappa0, *chi_of[p], p == 0 ? sg : 0.0);
        auto mp = Bump(mu0, *chi_of[p], p == 1 ? sg : 0.0);
        auto ep = Bump(eta0, *chi_of[p], p == 2 ? sg : 0.0);
        (sg > 0 ? plus : minus) = run_forward(*kp.value, *mp.value, *ep.value);
      }
      const real_t fd_end = (plus.J_end - minus.J_end) / (2.0 * eps);
      const real_t fd_cont = (plus.J_cont - minus.J_cont) / (2.0 * eps);
      Check(std::abs(dJ_end / fd_end - 1.0), 2e-5,
            std::string("viscoelastic tangent, end, ") + names[p]);
      Check(std::abs(dJ_cont / fd_cont - 1.0), 2e-5,
            std::string("viscoelastic tangent, continuous, ") + names[p]);
    }
  }

  if (Mpi::Root()) {
    std::cout << "TestTangentPar: " << num_checks << " checks, " << num_fails
              << " failures on " << Mpi::WorldSize() << " ranks\n";
  }
  return num_fails == 0 ? 0 : 1;
}
