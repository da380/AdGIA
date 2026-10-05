// ============================================================================
// lateral_viscosity.cpp
//
// A self-gravitating disc whose viscosity varies sideways: one half relaxes
// a hundred times faster than the other. A demonstration (there is no exact
// solution) of what lateral variations do to the relaxation, and of a
// pattern worth knowing: the asymmetry is TRANSIENT.
//
// The body is the homogeneous disc of elastogravity_2d.msh (radius 1, in a
// buffer for the DtN condition), self-gravitating, with a standard linear
// solid rheology (IsotropicMaxwellRheology with a long-term modulus):
//
//   shear modulus mu_inf + mu_1, relaxing to mu_inf with the time tau(x),
//
// so that the relaxed body is still an elastic solid (mu_inf > 0; a fully
// relaxing, compressible, uniform body would carry slowly growing buoyancy
// modes). The relaxation time drops smoothly across x = 0:
//
//   log tau(x) = log tau_slow + (log tau_fast - log tau_slow)
//                                (1 + tanh(x / w)) / 2,
//
// fast on the east (x > 0), slow on the west. A Heaviside surface load
// sigma0 2 cos(2 theta), theta measured from the y axis (positive at the
// poles, negative at the east and west points), is switched on at t = 0
// and held.
//
// The load is symmetric under x -> -x, and so are the elastic response at
// t = 0+ (the moduli do not depend on x) and the fully relaxed one (nor
// does mu_inf). In between the east relaxes first: the east and west points
// part company and then come together again, and the time over which they
// differ spans the two relaxation times. The displacement is reported in
// the centre-of-mass frame (SetMassWeightedGauge): an east-west difference
// is otherwise contaminated by a gauge translation. The unstructured mesh
// is not exactly mirror-symmetric, so even a uniform body (-tauf 1 -taus 1)
// shows an east-west difference of a few 1e-5 (0.3% of u_r): the floor
// against which the lateral effect (2.6e-3 at the defaults) stands out.
//
// Outputs: the table on the screen; lateral_viscosity.csv, u_r at the
// east, west and north points against log time (python3 plot_csv.py
// lateral_viscosity.csv); with -vis, GLVis windows of log10 tau and an
// animation of the radial displacement u_r.
//
// One source serves the serial and the parallel build; the genuine
// differences are the mesh partitioning and the observation points, each
// on one rank and reduced globally.
//
// Sample runs (with mpiexec -np N in front in a parallel build):
//    ./lateral_viscosity
//    ./lateral_viscosity -tauf 1 -taus 1       (uniform: no asymmetry)
//    ./lateral_viscosity -w 0.02 -o 3         (a sharper transition)
// ============================================================================
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "AdGIA.hpp"
#include "visualisation.hpp"

using namespace mfem;
using namespace AdGIA;

namespace {

#ifdef MFEM_USE_MPI
using MeshType = ParMesh;
using SubMeshType = ParSubMesh;
using SpaceType = ParFiniteElementSpace;
using FieldType = ParGridFunction;
bool Root() { return Mpi::Root(); }
real_t GlobalMin(real_t v) {
  real_t g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPITypeMap<real_t>::mpi_type, MPI_MIN,
                MPI_COMM_WORLD);
  return g;
}
real_t GlobalMax(real_t v) {
  real_t g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                MPI_COMM_WORLD);
  return g;
}
#else
using MeshType = Mesh;
using SubMeshType = SubMesh;
using SpaceType = FiniteElementSpace;
using FieldType = GridFunction;
bool Root() { return true; }
real_t GlobalMin(real_t v) { return v; }
real_t GlobalMax(real_t v) { return v; }
#endif

// The radial displacement at the mesh vertex nearest a target point. In
// parallel the vertex lives on one rank (or is shared, with equal values):
// the global nearest distance picks it, and the value is reduced with a
// -inf sentinel from the other ranks.
class Probe {
 public:
  Probe(Mesh& mesh, const FiniteElementSpace& fes, const Vector& target)
      : fes_(fes) {
    const int dim = mesh.Dimension();
    real_t best = std::numeric_limits<real_t>::infinity();
    for (int v = 0; v < mesh.GetNV(); v++) {
      const real_t* x = mesh.GetVertex(v);
      real_t d2 = 0.0;
      for (int c = 0; c < dim; c++) {
        d2 += (x[c] - target[c]) * (x[c] - target[c]);
      }
      if (d2 < best) {
        best = d2;
        vertex_ = v;
        x_.SetSize(dim);
        for (int c = 0; c < dim; c++) {
          x_[c] = x[c];
        }
      }
    }
    owner_ = best == GlobalMin(best);
  }

  // Collective in parallel.
  real_t RadialDisplacement(const GridFunction& u) const {
    real_t v = -std::numeric_limits<real_t>::infinity();
    if (owner_) {
      const int dim = x_.Size();
      v = 0.0;
      for (int c = 0; c < dim; c++) {
        v += u(fes_.DofToVDof(vertex_, c)) * x_[c];
      }
      v /= x_.Norml2();
    }
    return GlobalMax(v);
  }

 private:
  const FiniteElementSpace& fes_;
  int vertex_ = -1;
  Vector x_;
  bool owner_ = false;
};

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif

  const char* mesh_file = "../data/elastogravity_2d.msh";
  int order = 2;
  int dtn_degree = 16;
  real_t G = 0.2, rho = 1.0, kappa = 3.0, mu_inf = 0.2, mu_1 = 0.8;
  real_t tau_slow = 1.0, tau_fast = 0.01, width = 0.1;
  real_t sigma0 = 0.01;
  real_t t_first = 1e-3, t_final = 30.0;
  int n_out = 30, steps_per_out = 2;
  real_t rel_tol = 1e-8;
  bool visualization = true;
  const char* csv_file = "lateral_viscosity.csv";

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
                 "Mesh: a body (attribute 1) in a buffer (attribute 2).");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&dtn_degree, "-deg", "--dtn-degree", "DtN expansion degree.");
  args.AddOption(&G, "-G", "--gravitational-constant", "G.");
  args.AddOption(&kappa, "-kappa", "--bulk-modulus", "Bulk modulus.");
  args.AddOption(&mu_inf, "-mu-inf", "--relaxed-modulus",
                 "Long-term shear modulus.");
  args.AddOption(&mu_1, "-mu1", "--branch-modulus",
                 "Shear modulus that relaxes.");
  args.AddOption(&tau_slow, "-taus", "--tau-slow",
                 "Relaxation time on the west (the time unit).");
  args.AddOption(&tau_fast, "-tauf", "--tau-fast",
                 "Relaxation time on the east.");
  args.AddOption(&width, "-w", "--width", "Width of the transition.");
  args.AddOption(&sigma0, "-s", "--sigma", "Load amplitude.");
  args.AddOption(&t_first, "-t0", "--t-first", "First output time.");
  args.AddOption(&t_final, "-T", "--t-final", "Final time.");
  args.AddOption(&n_out, "-nout", "--outputs",
                 "Output times, spaced logarithmically from -t0 to -T.");
  args.AddOption(&steps_per_out, "-n", "--steps-per-output",
                 "Exponential-trapezoid steps between output times.");
  args.AddOption(&rel_tol, "-rt", "--rel-tol", "Relative solver tolerance.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization", "Show tau and animate u_r in GLVis.");
  args.AddOption(&csv_file, "-csv", "--csv",
                 "Table of the histories for plot_csv.py (\"\": none).");
  args.Parse();
  if (!args.Good() || t_first <= 0.0 || t_final <= t_first) {
    if (Root()) {
      args.PrintUsage(std::cout);
    }
    return 1;
  }
  if (Root()) {
    args.PrintOptions(std::cout);
  }

  Mesh smesh(mesh_file, 1, 1);
  const int dim = smesh.Dimension();
#ifdef MFEM_USE_MPI
  MeshType parent(MPI_COMM_WORLD, smesh);
  smesh.Clear();
#else
  MeshType& parent = smesh;
#endif
  Array<int> body_marker(parent.attributes.Max());
  body_marker = 0;
  body_marker[0] = 1;
  auto body = SubMeshType::CreateFromDomain(parent, body_marker);
  Array<int> surface(body.bdr_attributes.Max());
  surface = 0;
  surface[body.bdr_attributes.Max() - 1] = 1;

  H1_FECollection fec(order, dim);
  SpaceType fes_u(&body, &fec, dim), fes_phi(&parent, &fec), fes_s(&body, &fec);

  // The standard linear solid with the lateral relaxation time.
  ConstantCoefficient kappa_c(kappa), mu_inf_c(mu_inf), mu_1_c(mu_1), rho_c(rho);
  auto tau_of = [&](const Vector& x) {
    const real_t s = 0.5 * (1.0 + std::tanh(x[0] / width));
    return std::exp(std::log(tau_slow) +
                    (std::log(tau_fast) - std::log(tau_slow)) * s);
  };
  FunctionCoefficient tau_c(tau_of);
  IsotropicMaxwellRheology rheology(dim, kappa_c, mu_inf_c,
                                    {MaxwellBranch{&mu_1_c, &tau_c, nullptr}});

  // The load sigma0 2 cos(2 theta), theta measured from the y axis:
  // 2 (2 (y/r)^2 - 1), positive (pressing) at the poles.
  FunctionCoefficient sigma([&](const Vector& x) {
    const real_t c = x[dim - 1] / x.Norml2();
    return sigma0 * 2.0 * (2.0 * c * c - 1.0);
  });
  LinearQuasiStaticMixedSelfGravitatingProblem problem(
      &fes_u, &fes_phi, rheology, rho_c, G, dtn_degree);
  problem.SetSurfaceLoad(sigma, surface);
  problem.SetMassWeightedGauge();
  problem.SetRelTol(rel_tol);

  // Observation points on the surface.
  auto point = [&](real_t x, real_t y) {
    Vector p(dim);
    p = 0.0;
    p[0] = x;
    p[dim - 1] = y;
    return Probe(body, fes_u, p);
  };
  const Probe east = point(1.0, 0.0), west = point(-1.0, 0.0),
              north = point(0.0, 1.0);

  // GLVis: log10 tau once, u_r animated.
  FieldType u_r(&fes_s);
  VectorGridFunctionCoefficient u_c(&problem.Displacement());
  VectorFunctionCoefficient radial(dim, [](const Vector& x, Vector& n) {
    n = x;
    n /= x.Norml2();
  });
  InnerProductCoefficient u_r_c(u_c, radial);
  examples::GLVisWindow u_window("radial displacement u_r",
                                 examples::DefaultKeys(dim));
  if (visualization) {
    FieldType log_tau(&fes_s);
    FunctionCoefficient log_tau_c(
        [&](const Vector& x) { return std::log10(tau_of(x)); });
    log_tau.ProjectCoefficient(log_tau_c);
    examples::GLVisWindow("log10 relaxation time (fast east, slow west)",
                          examples::DefaultKeys(dim))
        .Send(body, log_tau);
  }

  examples::CsvTable table(csv_file, {"t", "east", "west", "north"});
  table.Meta("title", "Lateral viscosity: a disc relaxing faster on one side")
      .Meta("note", [&] {
        std::ostringstream os;
        os << "u_r at the surface; tau = " << tau_fast << " east, "
           << tau_slow << " west";
        return os.str();
      }())
      .Meta("xlabel", "time / slow relaxation time")
      .Meta("y", "east,west|north")
      .Meta("ylabel", "u_r east and west|u_r north (pole)")
      .Meta("logx", "true");
  if (Root()) {
    std::cout << "\n          t      u_r east      u_r west     u_r north"
                 "   east-west\n";
  }
  auto observe = [&](real_t t) {
    const real_t e = east.RadialDisplacement(problem.Displacement());
    const real_t w = west.RadialDisplacement(problem.Displacement());
    const real_t n = north.RadialDisplacement(problem.Displacement());
    table.Row({t, e, w, n});
    if (Root()) {
      std::cout << std::scientific << std::setprecision(3) << std::setw(11)
                << t << std::setw(14) << e << std::setw(14) << w
                << std::setw(14) << n << std::setw(12)
                << std::setprecision(2) << e - w << "\n";
    }
    if (visualization) {
      u_r.ProjectCoefficient(u_r_c);
      u_window.Send(body, u_r);
    }
  };

  // Time stepping: the exponential trapezoid between logarithmically
  // spaced output times (its steps grow with them: the fast side's
  // relaxation is integrated exactly in each step's exponential, so steps
  // much longer than tau_fast are fine once its transient has passed).
  ViscoelasticOperator visco(problem);
  ExponentialTrapezoidSolver ode;
  ode.Init(visco);
  Vector m(visco.Height());
  m = 0.0;
  real_t t = 0.0;
  if (!visco.SolveElastic(m, t)) {
    if (Root()) {
      std::cerr << "Elastic solve failed.\n";
    }
    return 2;
  }
  observe(t);
  const real_t ratio = std::pow(t_final / t_first, 1.0 / (n_out - 1));
  for (int k = 0; k < n_out; k++) {
    const real_t t_out = t_first * std::pow(ratio, k);
    const real_t dt = (t_out - t) / steps_per_out;
    for (int s = 0; s < steps_per_out; s++) {
      real_t h = dt;
      ode.Step(m, t, h);
    }
    t = t_out;
    if (!visco.SolveElastic(m, t)) {
      if (Root()) {
        std::cerr << "Elastic solve failed at t = " << t << "\n";
      }
      return 2;
    }
    observe(t);
  }
  if (Root()) {
    std::cout << "\nSolves " << problem.NumSolves() << ", assemblies "
              << problem.NumAssemblies() << "\n";
  }
  table.Write();
  return 0;
}
