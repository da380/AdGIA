/*
  Parallel tests for the common adjoint structure (adjoint.hpp). Run with
  1, 2 and 4 ranks; a standalone MPI program that exits non-zero if any
  check fails.

  The checks are the partition-independent finite-difference gates of the
  serial TestAdjoint on the same unit-square configuration: the static
  kappa/mu kernels, and the viscoelastic kappa/mu/eta kernels for a
  final-time observation and a continuous-in-time observation (the force
  hook on the parallel assembly path), plus the store-all/re-solve
  trajectory identity. The kernel-vs-FD identity holds per run on any
  partition, so the thresholds are the serial measurements' grades.
*/

#include <mpi.h>

#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

#include "AdGIA.hpp"
#include "MixedProblemTestCommon.hpp"
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

// Gravitating-leg observation duals (as in the serial TestAdjoint): windowed
// stations at the angles 0.3 and 1.4, off every mirror image of the
// degree-2 load's symmetries.
real_t StationWindow(const Vector& x) {
  constexpr real_t s = 0.1;
  const real_t phi = std::atan2(x[1], x[0]);
  auto window = [phi](real_t centre) {
    real_t d = phi - centre;
    while (d > kPi) d -= 2.0 * kPi;
    while (d < -kPi) d += 2.0 * kPi;
    return std::exp(-d * d / (s * s)) / (s * std::sqrt(kPi));
  };
  return window(1.4) - window(0.3);
}

void RadialStations(const Vector& x, Vector& v) {
  const real_t r = x.Norml2();
  const real_t w = StationWindow(x);
  v.SetSize(2);
  v[0] = w * x[0] / r;
  v[1] = w * x[1] / r;
}

real_t MassLoad(const Vector& x) {
  return self_grav_test::SurfaceLoad(x, 0.0);
}

// The gravitating leg: reciprocity (including the PotentialForceSlot
// sign certificate) and the kappa/mu/eta blob FD gates for displacement
// and potential observations on the mixed Maxwell body. Thresholds are
// the serial measurements' grades (the identities hold per run on any
// partition).
void GravitatingLeg() {
  const int order = 2, n_steps = 24;
  const real_t t_final = 2.0;
  auto serial_mesh = Mesh(self_grav_test::MeshFile(2).c_str(), 1, 1);
  auto parent = ParMesh(MPI_COMM_WORLD, serial_mesh);
  serial_mesh.Clear();
  auto body = ParSubMesh::CreateFromDomain(
      parent, self_grav_test::BodyMarker(parent));
  auto fec = H1_FECollection(order, 2);
  auto fes_u = ParFiniteElementSpace(&body, &fec, 2);
  auto fes_phi = ParFiniteElementSpace(&parent, &fec);
  auto sfes = ParFiniteElementSpace(&body, &fec);
  auto rho = ConstantCoefficient(self_grav_test::kRho);
  auto sigma = FunctionCoefficient(MassLoad);
  auto surface = self_grav_test::SurfaceMarker(body);
  Array<int> parent_surface(parent.bdr_attributes.Max());
  parent_surface = 0;
  parent_surface[0] = 1;

  auto h_u = VectorFunctionCoefficient(2, RadialStations);
  auto g_u = ParLinearForm(&fes_u);
  g_u.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(h_u), surface);
  g_u.Assemble();
  auto h_phi = FunctionCoefficient(
      [](const Vector& x) { return StationWindow(x); });
  auto g_phi = ParLinearForm(&fes_phi);
  g_phi.AddBoundaryIntegrator(new BoundaryLFIntegrator(h_phi),
                              parent_surface);
  g_phi.Assemble();

  auto make_problem = [&](const Rheology& rheology, bool with_load) {
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        &fes_u, &fes_phi, rheology, rho, self_grav_test::kG,
        self_grav_test::kDtNDegree);
    if (with_load) {
      p->SetSurfaceLoad(sigma, surface);
    }
    p->SetRelTol(1e-11);
    return p;
  };

  // --- reciprocity -----------------------------------------------------
  {
    auto kappa = ConstantCoefficient(self_grav_test::kKappa);
    auto mu = ConstantCoefficient(self_grav_test::kMu);
    auto rheology = IsotropicElasticRheology(2, kappa, mu);
    auto twin = make_problem(rheology, false);
    twin->AssembleForce(0.0);
    twin->AddPotentialForce(g_phi);
    if (!twin->Solve()) {
      Check(1.0, 0.0, "gravitating potential-dual solve");
    }
    const real_t b1 = GlobalSum(g_u * twin->Displacement());
    twin->AssembleForce(0.0);
    twin->AddForce(g_u);
    if (!twin->Solve()) {
      Check(1.0, 0.0, "gravitating displacement-dual solve");
    }
    const real_t b2 = GlobalSum(g_phi * twin->Potential());
    Check(std::abs(b1 / b2 - 1.0), 2e-9,
          "gravitating reciprocity u-phi (the potential slot)");
  }

  // --- the FD gates (blob directions) -----------------------------------
  auto kappa0 = ConstantCoefficient(self_grav_test::kKappa);
  auto mu0 = ConstantCoefficient(self_grav_test::kMu);
  auto eta0 = ConstantCoefficient(self_grav_test::kMu);
  auto chi_blob = FunctionCoefficient([](const Vector& x) {
    const real_t dx = x[0] - 0.3, dy = x[1] - 0.35;
    return std::exp(-(dx * dx + dy * dy) / 0.02);
  });
  const real_t dt = t_final / n_steps;

  struct Measured {
    real_t J_u = 0.0, J_phi = 0.0;
  };
  auto run_forward = [&](Coefficient& kappa, Coefficient& mu,
                         Coefficient& eta) {
    auto tau = RatioCoefficient(eta, mu);
    auto rheology = IsotropicMaxwellRheology::Maxwell(2, kappa, mu, tau);
    auto prob = make_problem(rheology, true);
    auto visco = ViscoelasticOperator(*prob);
    ExponentialTrapezoidSolver ode;
    ode.Init(visco);
    Vector m(visco.Height());
    m = 0.0;
    real_t t = 0.0, dt_step = dt;
    if (!visco.SolveElastic(m, t)) {
      Check(1.0, 0.0, "gravitating forward solve");
    }
    for (int n = 1; n <= n_steps; n++) {
      ode.Step(m, t, dt_step);
    }
    if (!visco.SolveElastic(m, t)) {
      Check(1.0, 0.0, "gravitating forward solve");
    }
    Measured out;
    out.J_u = GlobalSum(g_u * prob->Displacement());
    out.J_phi = GlobalSum(g_phi * prob->Potential());
    return out;
  };

  auto tau0 = RatioCoefficient(eta0, mu0);
  auto rheology0 = IsotropicMaxwellRheology::Maxwell(2, kappa0, mu0, tau0);
  auto prob_fwd = make_problem(rheology0, true);
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
  auto prob_adj = make_problem(rheology0, false);
  auto visco_adj = ViscoelasticOperator(*prob_adj);

  std::vector<ProductCoefficient> dirs;
  dirs.emplace_back(kappa0, chi_blob);
  dirs.emplace_back(mu0, chi_blob);
  dirs.emplace_back(eta0, chi_blob);

  struct Sweep {
    std::unique_ptr<IsotropicKernel> K[3];
  };
  auto make_sweep = [&](const AdjointForce& force) {
    Sweep sw;
    sw.K[0] = std::make_unique<IsotropicKernel>(
        IsotropicKernel::Parameter::Kappa, sfes);
    sw.K[1] =
        std::make_unique<IsotropicKernel>(IsotropicKernel::Parameter::Mu,
                                          sfes);
    sw.K[2] = std::make_unique<IsotropicKernel>(
        IsotropicKernel::Parameter::Eta, sfes, &tau0);
    for (int p = 0; p < 3; p++) {
      sw.K[p]->AddDirection(dirs[p]);
    }
    auto solver = AdjointSolver(visco_adj, trajectory);
    solver.SetForce(force);
    for (int p = 0; p < 3; p++) {
      solver.AddKernel(*sw.K[p]);
    }
    if (!solver.Run()) {
      Check(1.0, 0.0, "gravitating adjoint sweep");
    }
    return sw;
  };
  AdjointForce force_u;
  force_u.AddObservation(trajectory.Time(n_steps), g_u);
  AdjointForce force_phi;
  force_phi.AddPotentialObservation(trajectory.Time(n_steps), g_phi);
  auto sweep_u = make_sweep(force_u);
  auto sweep_phi = make_sweep(force_phi);

  const real_t eps = 1e-3;
  const char* names[] = {"kappa", "mu", "eta"};
  for (int p = 0; p < 3; p++) {
    Measured plus, minus;
    for (const real_t sg : {eps, -eps}) {
      auto kp = Bump(kappa0, chi_blob, p == 0 ? sg : 0.0);
      auto mp = Bump(mu0, chi_blob, p == 1 ? sg : 0.0);
      auto ep = Bump(eta0, chi_blob, p == 2 ? sg : 0.0);
      auto out = run_forward(*kp.value, *mp.value, *ep.value);
      (sg > 0 ? plus : minus) = out;
    }
    const real_t fd_u = (plus.J_u - minus.J_u) / (2.0 * eps);
    const real_t fd_phi = (plus.J_phi - minus.J_phi) / (2.0 * eps);
    // Serial grades: mu blob 1.3e-3 (curved-element strain-map floor),
    // the rest 1.6e-4 or below.
    Check(std::abs(sweep_u.K[p]->Directional(0) / fd_u - 1.0), 5e-3,
          std::string("gravitating u-observation ") + names[p] + ", blob");
    Check(std::abs(sweep_phi.K[p]->Directional(0) / fd_phi - 1.0), 5e-3,
          std::string("gravitating phi-observation ") + names[p] + ", blob");
  }
}

}  // namespace

// The two-branch Prony kernels (blob direction; the three modes mu_inf and
// the fast branch's mu_2/eta_2) against FD at n = 32; the thresholds are that
// resolution's measured O(dt^2) floors (mu_2 2.2e-2, eta_2 2.8e-3,
// mu_inf 5.3e-4; the serial gate documents the factor-4 convergence).
void PronyLeg() {
  const int dim = 2, n_el = 12, order = 2, n_steps = 32;
  const real_t t_final = 2.0;
  auto serial_mesh =
      Mesh::MakeCartesian2D(n_el, n_el, Element::QUADRILATERAL);
  auto mesh = ParMesh(MPI_COMM_WORLD, serial_mesh);
  serial_mesh.Clear();
  auto fec = H1_FECollection(order, dim);
  auto fes = ParFiniteElementSpace(&mesh, &fec, dim);
  auto sfes = ParFiniteElementSpace(&mesh, &fec);
  auto top = Array<int>(mesh.bdr_attributes.Max());
  auto bottom = Array<int>(mesh.bdr_attributes.Max());
  top = 0;
  bottom = 0;
  top[2] = 1;
  bottom[0] = 1;
  auto traction = VectorFunctionCoefficient(dim, Traction2D);
  auto station = VectorFunctionCoefficient(dim, Station2D);
  auto g = ParLinearForm(&fes);
  g.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(station), bottom);
  g.Assemble();
  auto chi_blob = FunctionCoefficient([](const Vector& x) {
    const real_t r2 =
        (x[0] - 0.4) * (x[0] - 0.4) + (x[1] - 0.6) * (x[1] - 0.6);
    return std::exp(-r2 / 0.015);
  });
  auto kappa0 = ConstantCoefficient(5.4);
  auto mu_inf0 = ConstantCoefficient(1.0);
  auto mu10 = ConstantCoefficient(1.0);
  auto eta10 = ConstantCoefficient(1.0);
  auto mu20 = ConstantCoefficient(0.7);
  auto eta20 = ConstantCoefficient(0.21);
  const real_t dt = t_final / n_steps;

  auto run_forward = [&](Coefficient& mu_inf, Coefficient& mu2,
                         Coefficient& eta2, real_t* J) {
    auto tau1 = RatioCoefficient(eta10, mu10);
    auto tau2 = RatioCoefficient(eta2, mu2);
    std::vector<MaxwellBranch> branches = {{&mu10, &tau1, nullptr},
                                           {&mu2, &tau2, nullptr}};
    auto rheology = IsotropicMaxwellRheology(dim, kappa0, mu_inf, branches);
    auto prob =
        LinearQuasiStaticTractionProblem(&fes, rheology, traction, top);
    auto visco = ViscoelasticOperator(prob);
    ExponentialTrapezoidSolver ode;
    ode.Init(visco);
    Vector m(visco.Height());
    m = 0.0;
    real_t t = 0.0, dt_step = dt;
    if (!visco.SolveElastic(m, t)) {
      Check(1.0, 0.0, "prony forward solve");
    }
    for (int n = 1; n <= n_steps; n++) {
      ode.Step(m, t, dt_step);
    }
    if (!visco.SolveElastic(m, t)) {
      Check(1.0, 0.0, "prony forward solve");
    }
    *J = GlobalSum(g * prob.Displacement());
  };

  // Base run on persistent operators.
  auto tau10 = RatioCoefficient(eta10, mu10);
  auto tau20 = RatioCoefficient(eta20, mu20);
  std::vector<MaxwellBranch> branches0 = {{&mu10, &tau10, nullptr},
                                          {&mu20, &tau20, nullptr}};
  auto rheology0 = IsotropicMaxwellRheology(dim, kappa0, mu_inf0, branches0);
  auto prob_fwd =
      LinearQuasiStaticTractionProblem(&fes, rheology0, traction, top);
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
  auto zero_vec = Vector(dim);
  zero_vec = 0.0;
  auto zero_traction = VectorConstantCoefficient(zero_vec);
  auto prob_adj = LinearQuasiStaticTractionProblem(&fes, rheology0,
                                                   zero_traction, top);
  auto visco_adj = ViscoelasticOperator(prob_adj);

  using P = IsotropicKernel::Parameter;
  auto K_muinf = IsotropicKernel(P::MuLongTerm, sfes);
  auto K_mu2 = IsotropicKernel(P::Mu, sfes, nullptr, 1);
  auto K_eta2 = IsotropicKernel(P::Eta, sfes, &tau20, 1);
  auto d_mi = ProductCoefficient(mu_inf0, chi_blob);
  auto d_m2 = ProductCoefficient(mu20, chi_blob);
  auto d_e2 = ProductCoefficient(eta20, chi_blob);
  K_muinf.AddDirection(d_mi);
  K_mu2.AddDirection(d_m2);
  K_eta2.AddDirection(d_e2);

  AdjointForce force;
  force.AddObservation(trajectory.Time(n_steps), g);
  auto solver = AdjointSolver(visco_adj, trajectory);
  solver.SetForce(force);
  solver.AddKernel(K_muinf);
  solver.AddKernel(K_mu2);
  solver.AddKernel(K_eta2);
  if (!solver.Run()) {
    Check(1.0, 0.0, "prony adjoint sweep");
  }

  const real_t eps = 1e-3;
  struct Row {
    IsotropicKernel* K;
    int which;  // 0 mu_inf, 1 mu_2, 2 eta_2
    real_t tol;
    const char* name;
  };
  Row rows[] = {{&K_muinf, 0, 3e-3, "prony mu_inf, blob"},
                {&K_mu2, 1, 5e-2, "prony mu_2, blob"},
                {&K_eta2, 2, 1e-2, "prony eta_2, blob"}};
  for (const auto& row : rows) {
    real_t fd = 0.0;
    for (const real_t sg : {eps, -eps}) {
      auto bmi = Bump(mu_inf0, chi_blob, row.which == 0 ? sg : 0.0);
      auto bm2 = Bump(mu20, chi_blob, row.which == 1 ? sg : 0.0);
      auto be2 = Bump(eta20, chi_blob, row.which == 2 ? sg : 0.0);
      real_t J = 0.0;
      run_forward(*bmi.value, *bm2.value, *be2.value, &J);
      fd += (sg > 0 ? J : -J);
    }
    fd /= 2.0 * eps;
    Check(std::abs(row.K->Directional(0) / fd - 1.0), row.tol, row.name);
  }
}

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();

  const int dim = 2, n_el = 12, order = 2, n_steps = 32;
  const real_t t_final = 2.0;
  auto serial_mesh =
      Mesh::MakeCartesian2D(n_el, n_el, Element::QUADRILATERAL);
  auto mesh = ParMesh(MPI_COMM_WORLD, serial_mesh);
  serial_mesh.Clear();
  auto fec = H1_FECollection(order, dim);
  auto fes = ParFiniteElementSpace(&mesh, &fec, dim);
  auto sfes = ParFiniteElementSpace(&mesh, &fec);

  auto top = Array<int>(mesh.bdr_attributes.Max());
  auto bottom = Array<int>(mesh.bdr_attributes.Max());
  top = 0;
  bottom = 0;
  top[2] = 1;
  bottom[0] = 1;
  auto traction = VectorFunctionCoefficient(dim, Traction2D);
  auto station = VectorFunctionCoefficient(dim, Station2D);
  auto g = ParLinearForm(&fes);
  g.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(station), bottom);
  g.Assemble();

  auto chi_blob = FunctionCoefficient([](const Vector& x) {
    const real_t r2 =
        (x[0] - 0.4) * (x[0] - 0.4) + (x[1] - 0.6) * (x[1] - 0.6);
    return std::exp(-r2 / 0.015);
  });
  auto chi_all = ConstantCoefficient(1.0);
  auto kappa0 = ConstantCoefficient(kKappa0);
  auto mu0 = ConstantCoefficient(kMu0);
  auto eta0 = ConstantCoefficient(kEta0);

  // === The static gate ===
  {
    auto measure = [&](Coefficient& kappa, Coefficient& mu,
                       GridFunction* keep = nullptr) {
      auto rheology = IsotropicElasticRheology(dim, kappa, mu);
      auto prob =
          LinearQuasiStaticTractionProblem(&fes, rheology, traction, top);
      prob.AssembleForce(0.0);
      if (!prob.Solve()) {
        Check(1.0, 0.0, "static forward solve");
      }
      if (keep) {
        *keep = prob.Displacement();
      }
      return GlobalSum(g * prob.Displacement());
    };
    auto u = ParGridFunction(&fes);
    measure(kappa0, mu0, &u);

    auto rheology0 = IsotropicElasticRheology(dim, kappa0, mu0);
    auto zero_vec = Vector(dim);
    zero_vec = 0.0;
    auto zero_traction = VectorConstantCoefficient(zero_vec);
    auto adj_prob = LinearQuasiStaticTractionProblem(&fes, rheology0,
                                                     zero_traction, top);
    AdjointForce force;
    force.AddObservation(0.0, g);
    auto K_kappa = IsotropicKernel(IsotropicKernel::Parameter::Kappa, sfes);
    auto K_mu = IsotropicKernel(IsotropicKernel::Parameter::Mu, sfes);
    auto dk_b = ProductCoefficient(kappa0, chi_blob);
    auto dk_a = ProductCoefficient(kappa0, chi_all);
    auto dm_b = ProductCoefficient(mu0, chi_blob);
    auto dm_a = ProductCoefficient(mu0, chi_all);
    K_kappa.AddDirection(dk_b);
    K_kappa.AddDirection(dk_a);
    K_mu.AddDirection(dm_b);
    K_mu.AddDirection(dm_a);
    auto solver = AdjointSolver(adj_prob, u);
    solver.SetForce(force);
    solver.AddKernel(K_kappa);
    solver.AddKernel(K_mu);
    if (!solver.Run()) {
      Check(1.0, 0.0, "static adjoint run");
    }

    const real_t eps = 1e-3;
    auto fd = [&](bool is_kappa, Coefficient& chi) {
      auto kp = Bump(kappa0, chi, is_kappa ? eps : 0.0);
      auto mp = Bump(mu0, chi, is_kappa ? 0.0 : eps);
      auto km = Bump(kappa0, chi, is_kappa ? -eps : 0.0);
      auto mm = Bump(mu0, chi, is_kappa ? 0.0 : -eps);
      return (measure(*kp.value, *mp.value) -
              measure(*km.value, *mm.value)) /
             (2.0 * eps);
    };
    // Serial grade 1.1e-6; the parallel solver path gets the same order.
    Check(std::abs(K_kappa.Directional(0) / fd(true, chi_blob) - 1.0), 1e-5,
          "static kappa kernel, blob");
    Check(std::abs(K_kappa.Directional(1) / fd(true, chi_all) - 1.0), 1e-5,
          "static kappa kernel, all");
    Check(std::abs(K_mu.Directional(0) / fd(false, chi_blob) - 1.0), 1e-5,
          "static mu kernel, blob");
    Check(std::abs(K_mu.Directional(1) / fd(false, chi_all) - 1.0), 1e-5,
          "static mu kernel, all");
  }

  // === The viscoelastic gates ===
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
      auto prob =
          LinearQuasiStaticTractionProblem(&fes, rheology, traction, top);
      auto visco = ViscoelasticOperator(prob);
      ExponentialTrapezoidSolver ode;
      ode.Init(visco);
      Vector m(visco.Height());
      m = 0.0;
      real_t t = 0.0, dt_step = dt;
      Measured out;
      auto sample = [&](int n) {
        if (!visco.SolveElastic(m, t)) {
          Check(1.0, 0.0, "forward solve");
        }
        const real_t Jt = GlobalSum(g * prob.Displacement());
        out.J_cont += ((n == 0 || n == n_steps) ? 0.5 * dt : dt) *
                      cont_weight(t) * Jt;
        if (n == n_steps) {
          out.J_end = Jt;
        }
      };
      sample(0);
      for (int n = 1; n <= n_steps; n++) {
        ode.Step(m, t, dt_step);
        sample(n);
      }
      return out;
    };

    // The base run on persistent operators, recording both policies.
    auto tau0 = RatioCoefficient(eta0, mu0);
    auto rheology0 =
        IsotropicMaxwellRheology::Maxwell(dim, kappa0, mu0, tau0);
    auto prob_fwd =
        LinearQuasiStaticTractionProblem(&fes, rheology0, traction, top);
    auto visco_fwd = ViscoelasticOperator(prob_fwd);
    auto trajectory = ForwardTrajectory(visco_fwd, true);
    auto trajectory_lean = ForwardTrajectory(visco_fwd, false);
    {
      ExponentialTrapezoidSolver ode;
      ode.Init(visco_fwd);
      Vector m(visco_fwd.Height());
      m = 0.0;
      real_t t = 0.0, dt_step = dt;
      trajectory.Record(m, t);
      trajectory_lean.Record(m, t);
      for (int n = 1; n <= n_steps; n++) {
        ode.Step(m, t, dt_step);
        trajectory.Record(m, t);
        trajectory_lean.Record(m, t);
      }
    }

    auto zero_vec = Vector(dim);
    zero_vec = 0.0;
    auto zero_traction = VectorConstantCoefficient(zero_vec);
    auto prob_adj = LinearQuasiStaticTractionProblem(&fes, rheology0,
                                                     zero_traction, top);
    auto visco_adj = ViscoelasticOperator(prob_adj);

    std::vector<ProductCoefficient> dirs;
    dirs.emplace_back(kappa0, chi_blob);
    dirs.emplace_back(kappa0, chi_all);
    dirs.emplace_back(mu0, chi_blob);
    dirs.emplace_back(mu0, chi_all);
    dirs.emplace_back(eta0, chi_blob);
    dirs.emplace_back(eta0, chi_all);

    struct Sweep {
      std::unique_ptr<IsotropicKernel> kappa, mu, eta;
    };
    auto make_sweep = [&](ForwardTrajectory& traj,
                          const AdjointForce& force) {
      Sweep s;
      s.kappa = std::make_unique<IsotropicKernel>(
          IsotropicKernel::Parameter::Kappa, sfes);
      s.mu = std::make_unique<IsotropicKernel>(
          IsotropicKernel::Parameter::Mu, sfes);
      s.eta = std::make_unique<IsotropicKernel>(
          IsotropicKernel::Parameter::Eta, sfes, &tau0);
      s.kappa->AddDirection(dirs[0]);
      s.kappa->AddDirection(dirs[1]);
      s.mu->AddDirection(dirs[2]);
      s.mu->AddDirection(dirs[3]);
      s.eta->AddDirection(dirs[4]);
      s.eta->AddDirection(dirs[5]);
      auto solver = AdjointSolver(visco_adj, traj);
      solver.SetForce(force);
      solver.AddKernel(*s.kappa);
      solver.AddKernel(*s.mu);
      solver.AddKernel(*s.eta);
      if (!solver.Run()) {
        Check(1.0, 0.0, "adjoint sweep");
      }
      return s;
    };

    AdjointForce force_end;
    force_end.AddObservation(trajectory.Time(n_steps), g);
    AdjointForce force_cont;
    force_cont.AddContinuous(
        [&](real_t t, Vector& f) { f.Add(cont_weight(t), g); });

    auto sweep_end = make_sweep(trajectory, force_end);
    auto sweep_cont = make_sweep(trajectory, force_cont);
    auto sweep_lean = make_sweep(trajectory_lean, force_end);

    for (int i = 0; i < 2; i++) {
      Check(std::abs(sweep_lean.kappa->Directional(i) /
                         sweep_end.kappa->Directional(i) -
                     1.0),
            1e-9, "store-all vs re-solve, kappa");
      Check(std::abs(sweep_lean.mu->Directional(i) /
                         sweep_end.mu->Directional(i) -
                     1.0),
            1e-9, "store-all vs re-solve, mu");
      Check(std::abs(sweep_lean.eta->Directional(i) /
                         sweep_end.eta->Directional(i) -
                     1.0),
            1e-9, "store-all vs re-solve, eta");
    }

    const real_t eps = 1e-3;
    Coefficient* chi_of[] = {&chi_blob, &chi_all};
    Measured fd[3][2];
    for (int p = 0; p < 3; p++) {
      for (int c = 0; c < 2; c++) {
        Measured plus, minus;
        for (const real_t s : {eps, -eps}) {
          auto kp = Bump(kappa0, *chi_of[c], p == 0 ? s : 0.0);
          auto mp = Bump(mu0, *chi_of[c], p == 1 ? s : 0.0);
          auto ep = Bump(eta0, *chi_of[c], p == 2 ? s : 0.0);
          auto out = run_forward(*kp.value, *mp.value, *ep.value);
          (s > 0 ? plus : minus) = out;
        }
        fd[p][c].J_end = (plus.J_end - minus.J_end) / (2.0 * eps);
        fd[p][c].J_cont = (plus.J_cont - minus.J_cont) / (2.0 * eps);
      }
    }

    // Serial grades: point rows 3.3e-6, continuous eta rows 9.8e-4 (the
    // O(dt^2) quadrature floor; see TestAdjoint.cpp).
    const char* names[] = {"kappa", "mu", "eta"};
    for (int p = 0; p < 3; p++) {
      const IsotropicKernel* Ke[] = {sweep_end.kappa.get(),
                                     sweep_end.mu.get(),
                                     sweep_end.eta.get()};
      const IsotropicKernel* Kc[] = {sweep_cont.kappa.get(),
                                     sweep_cont.mu.get(),
                                     sweep_cont.eta.get()};
      const real_t tol_cont = p == 2 ? 5e-3 : 2e-5;
      for (int c = 0; c < 2; c++) {
        Check(std::abs(Ke[p]->Directional(c) / fd[p][c].J_end - 1.0), 2e-5,
              std::string("final-time ") + names[p] +
                  (c == 0 ? ", blob" : ", all"));
        Check(std::abs(Kc[p]->Directional(c) / fd[p][c].J_cont - 1.0),
              tol_cont, std::string("continuous ") + names[p] +
                            (c == 0 ? ", blob" : ", all"));
      }
    }
  }

  GravitatingLeg();
  PronyLeg();

  if (Mpi::Root()) {
    std::cout << (num_fails == 0 ? "OK" : "FAILED") << ": " << num_checks
              << " checks, " << num_fails << " failures\n";
  }
  return num_fails == 0 ? 0 : 1;
}
