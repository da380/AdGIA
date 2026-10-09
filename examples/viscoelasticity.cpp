// ============================================================================
// viscoelasticity.cpp
//
// Quasi-static generalised Maxwell viscoelasticity with the library's
// ViscoelasticOperator on top of a LinearQuasiStaticProblem
// (AdGIA/quasi_static_problem.hpp, viscoelastic.hpp).
//
// The rheology is a Maxwell body (mu_inf = 0, one branch) or, with -mu-inf,
// a standard linear solid (mu_inf > 0, one branch). With -ti the body is
// transversely isotropic instead (AnisotropicMaxwellRheology, Love's
// constants from the isotropic moduli with an anisotropy factor, symmetry
// axis e_d), relaxing the deviatoric part of the tensor; -ti 1 reproduces
// the isotropic run. The elastic problem is assembled with the unrelaxed
// modulus from the same rheology object the viscoelastic operator reads its
// branch data from.
//
// Time integrators (-s): exponential trapezoid (default; second order, one
// solve per step, no step restriction), exponential Euler, backward Euler
// and SDIRK23 through MFEM's implicit solvers, and explicit RK4 / forward
// Euler (stable only for dt < ~2.8 tau_min).
//
// Options (defaults in brackets):
//   -m        mesh file [../data/star.mesh].
//   -o        finite element order of the displacement [2].
//   -mo       order of the internal-variable space [-1: the smallest
//             order resolving eps(u) exactly, order - 1 on simplices
//             and order on tensor-product elements].
//   -r        uniform mesh refinements [1].
//   -p        problem type [0]: 0 = pure traction on all external
//             boundaries (any mesh), 1 = clamped (needs boundary
//             attributes 1 and 2, e.g. beam-quad.mesh).
//   -s        time integrator [0]: 0 = exponential trapezoid, 1 =
//             exponential Euler, 2 = backward Euler, 3 = SDIRK23,
//             4 = RK4, 5 = forward Euler (4 and 5 are explicit,
//             stable only for dt < ~2.8 tau_min).
//   -map      strain map [0]: 0 = Galerkin projection, 1 =
//             interpolation.
//   -tf       final time [5].
//   -n        number of time steps [50]; with -rtol > 0, the output
//             times between which the adaptive solver steps.
//   -tau      Maxwell relaxation time tau = eta / mu [1].
//   -mu-inf   long-term shear modulus [0]: 0 is a Maxwell body,
//             > 0 a standard linear solid.
//   -gamma    nonlinearity of the power-law relaxation [0: linear];
//             > 0 puts the composite law of Crawford et al. (2017,
//             App. A) on the Maxwell branch.
//   -nexp     exponent n of the power-law relaxation [3].
//   -rtol     relative tolerance of adaptive stepping [0: fixed dt];
//             > 0 replaces the fixed steps by the adaptive
//             exponential trapezoid solver.
//   -ti       anisotropy factor of a transversely isotropic body
//             [0: isotropic; 1 reproduces the isotropic run through
//             the anisotropic path].
//   -pv / -no-pv     ParaView time slices on or off [on].
//   -vis / -no-vis   final displacement to GLVis on or off [on].
//
// Sample runs:
//    ./viscoelasticity -m ../data/star.mesh -o 2 -r 2
//    ./viscoelasticity -m ../data/star.mesh -o 2 -r 2 -s 4 -n 200
//    ./viscoelasticity -m ../data/beam-quad.mesh -p 1 -o 2 -r 1 -tau 0.5
//    ./viscoelasticity -m ../data/beam-quad.mesh -p 1 -mu-inf 0.5 -tf 20
//    ./viscoelasticity -m ../data/beam-quad.mesh -p 1 -ti 1.3
//    ./viscoelasticity -m ../data/beam-quad.mesh -p 1 -gamma 5 -rtol 1e-3
//
// -gamma g (with -nexp n) puts the composite Newtonian / power-law
// relaxation law of Crawford et al. (2017, App. A) on the Maxwell branch,
// tau = tau0 / (1 + g (|dev sigma| / 2 mu0)^(n-1)) with mu0 the unrelaxed
// shear modulus; -rtol r > 0 replaces the fixed steps by the adaptive
// exponential trapezoid solver with that relative tolerance.
//
// One source serves the serial and the parallel build; the genuine
// differences are the mesh partitioning, the global reductions on the
// printed norms and counts, the per-rank native-format save and the
// GLVis stream header. (Run with mpiexec -np N in a parallel build.)
// ============================================================================

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include "AdGIA.hpp"

using namespace std;
using namespace mfem;
using namespace AdGIA;

namespace {

#ifdef MFEM_USE_MPI
using MeshType = ParMesh;
using SpaceType = ParFiniteElementSpace;
bool Root() { return Mpi::Root(); }
double GlobalSum(double v) {
  double g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return g;
}
#else
using MeshType = Mesh;
using SpaceType = FiniteElementSpace;
bool Root() { return true; }
double GlobalSum(double v) { return v; }
#endif

// The internal-variable vector is distributed in parallel: its norm and
// its length need reductions.
double GlobalNorm(const Vector& v) {
  return std::sqrt(GlobalSum(v * v));
}

string RankName(const string& base) {
#ifdef MFEM_USE_MPI
  ostringstream name;
  name << base << "." << setfill('0') << setw(6) << Mpi::WorldRank();
  return name.str();
#else
  return base;
#endif
}

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif

  // Set the default options.
  const char* mesh_file = "../data/star.mesh";
  int order = 2;
  // Internal-variable order; < 0 means order - 1 on simplices and order on
  // tensor-product elements (the smallest order resolving eps(u) exactly).
  int m_order = -1;
  real_t ti_factor = 0.0;
  real_t gamma0 = 0.0, nexp = 3.0, rtol = 0.0;
  int ref_levels = 1;
  int problem_type = 0;
  int solver_type = 0;
  int map_type = 0;
  real_t t_final = 5.0;
  int n_steps = 50;
  real_t tau0 = 1.0;
  real_t mu_inf0 = 0.0;
  bool paraview = true;
  bool visualization = true;

  // Read in command line options and process.
  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order for the displacement.");
  args.AddOption(&m_order, "-mo", "--m-order",
                 "Order of the internal-variable space (< 0: the order "
                 "resolving eps(u) exactly: order - 1 on simplices, order "
                 "on tensor-product elements).");
  args.AddOption(&ref_levels, "-r", "--refinement",
                 "Number of uniform mesh refinements.");
  args.AddOption(&problem_type, "-p", "--problem",
                 "Problem type: 0 = pure traction (any mesh), 1 = clamped "
                 "(needs two boundary attributes, e.g. beam-quad.mesh).");
  args.AddOption(&solver_type, "-s", "--solver",
                 "Time integrator: 0 = exponential trapezoid, 1 = exponential "
                 "Euler, 2 = backward Euler, 3 = SDIRK23, 4 = RK4, "
                 "5 = forward Euler.");
  args.AddOption(&map_type, "-map", "--strain-map",
                 "Strain map: 0 = Galerkin projection, 1 = interpolation.");
  args.AddOption(&t_final, "-tf", "--t-final", "Final time.");
  args.AddOption(&n_steps, "-n", "--n-steps", "Number of time steps.");
  args.AddOption(&tau0, "-tau", "--relaxation-time",
                 "Maxwell relaxation time tau = eta / mu.");
  args.AddOption(&mu_inf0, "-mu-inf", "--long-term-modulus",
                 "Long-term shear modulus (0: Maxwell body).");
  args.AddOption(&gamma0, "-gamma", "--power-law-gamma",
                 "Nonlinearity of the power-law relaxation (0: linear).");
  args.AddOption(&nexp, "-nexp", "--power-law-exponent",
                 "Exponent n of the power-law relaxation.");
  args.AddOption(&rtol, "-rtol", "--adaptive-rtol",
                 "Relative tolerance of adaptive stepping (0: fixed dt).");
  args.AddOption(&ti_factor, "-ti", "--transversely-isotropic",
                 "Anisotropy factor of a transversely isotropic body (0: "
                 "isotropic; 1: isotropic through the anisotropic path).");
  args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                 "Save time slices to a ParaView data collection.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Send the final displacement to a running GLVis server.");
  args.Parse();
  if (!args.Good()) {
    if (Root()) {
      args.PrintUsage(cout);
    }
    return 1;
  }
  if (Root()) {
    args.PrintOptions(cout);
  }

  // Read in the mesh, refine if requested, and partition in parallel.
  Mesh smesh(mesh_file, 1, 1);
  const int dim = smesh.Dimension();
  for (int l = 0; l < ref_levels; l++) {
    smesh.UniformRefinement();
  }
#ifdef MFEM_USE_MPI
  MeshType mesh(MPI_COMM_WORLD, smesh);
  smesh.Clear();
#else
  MeshType& mesh = smesh;
#endif

  // Displacement space and material. kappa = 1 + 2/d so that the unrelaxed
  // state has lambda = mu = 1 when mu_inf + mu_1 = 1.
  H1_FECollection fec(order, dim);
  SpaceType fes(&mesh, &fec, dim);
  ConstantCoefficient kappa(1.0 + 2.0 / dim), mu_inf(mu_inf0),
      mu1(1.0 - mu_inf0), tau(tau0);
  ConstantCoefficient gamma_c(gamma0), n_c(nexp), mu0_c(1.0);
  PowerLawRelaxation power_law(gamma_c, n_c, mu0_c);
  const RelaxationLaw* law = gamma0 > 0.0 ? &power_law : nullptr;
  std::vector<MaxwellBranch> branches{{&mu1, &tau, law}};
  IsotropicMaxwellRheology isotropic(dim, kappa, mu_inf, branches);

  // Transversely isotropic alternative: Love's constants of the isotropic
  // unrelaxed body (lambda = mu = 1: A = C = 3, F = 1, L = N = 1) with the
  // axis-parallel P and S moduli scaled by the factor, axis e_d, relaxing
  // the deviatoric part with mu_inf / (mu_inf + mu_1) of it retained.
  unique_ptr<AnisotropicMaxwellRheology> anisotropic;
  ConstantCoefficient A(3.0), C(3.0 * ti_factor), F(1.0), L(ti_factor),
      N(1.0), tau_ti(tau0);
  Vector axis(dim);
  axis = 0.0;
  axis[dim - 1] = 1.0;
  VectorConstantCoefficient axis_coef(axis);
  TransverselyIsotropicElasticTensorCoefficient C_ti(dim, A, C, F, L, N,
                                                     axis_coef);
  DeviatoricProjectionElasticTensorCoefficient C_dev(dim, C_ti, true),
      C_vol(dim, C_ti, false);
  ConstantCoefficient retained(mu_inf0), relaxable(1.0 - mu_inf0);
  ScalarMatrixProductCoefficient C_inf_dev(retained, C_dev),
      C_1(relaxable, C_dev);
  MatrixSumCoefficient C_inf(C_vol, C_inf_dev, 1.0, 1.0);
  std::vector<AnisotropicBranch> ti_branches{{&C_1, &tau_ti, law}};
  if (ti_factor > 0.0) {
    anisotropic =
        make_unique<AnisotropicMaxwellRheology>(dim, C_inf, ti_branches);
  }
  const Rheology& rheology =
      anisotropic ? static_cast<const Rheology&>(*anisotropic)
                  : static_cast<const Rheology&>(isotropic);

  // Loads. Problem 0: a time-scaled uniform traction t -> (0, 1 + t, ...)
  // on all external boundaries. Problem 1: boundary attribute 1 clamped,
  // a time-scaled pull t -> (0, ..., -0.05 (1 + t)) on attribute 2.
  VectorFunctionCoefficient traction(
      dim, [problem_type](const Vector& /*x*/, real_t t, Vector& f) {
        f = 0.0;
        if (problem_type == 0) {
          f[1] = 1.0 + t;
        } else {
          f[f.Size() - 1] = -0.05 * (1.0 + t);
        }
      });
  Array<int> marker(mesh.bdr_attributes.Max()), ess_bdr;
  marker = 0;

  unique_ptr<LinearQuasiStaticProblemBase> problem;
  if (problem_type == 0) {
    mesh.MarkExternalBoundaries(marker);
    problem = make_unique<LinearQuasiStaticTractionProblem>(&fes, rheology,
                                                            traction, marker);
  } else if (problem_type == 1) {
    MFEM_VERIFY(mesh.bdr_attributes.Max() >= 2,
                "Problem 1 needs boundary attributes 1 (clamped) and 2 "
                "(traction), e.g. data/beam-quad.mesh.");
    ess_bdr.SetSize(mesh.bdr_attributes.Max());
    ess_bdr = 0;
    ess_bdr[0] = 1;
    marker[1] = 1;
    problem = make_unique<LinearQuasiStaticClampedProblem>(
        &fes, rheology, ess_bdr, traction, marker);
  } else {
    if (Root()) {
      cerr << "Unknown problem type: " << problem_type << "\n";
    }
    return 1;
  }
  problem->SetPrintLevel(IterativeSolver::PrintLevel().Summary());

  const auto map = map_type == 0
                       ? ViscoelasticOperator::StrainMap::Galerkin
                       : ViscoelasticOperator::StrainMap::Interpolation;
  ViscoelasticOperator visco(*problem, m_order, map);
  {
    // Collective calls on every rank, root prints.
    const double n_m = GlobalSum(visco.Height());
#ifdef MFEM_USE_MPI
    const auto n_u = fes.GlobalTrueVSize();
#else
    const auto n_u = fes.GetTrueVSize();
#endif
    if (Root()) {
      cout << "Displacement unknowns:      " << n_u << "\n";
      cout << "Internal-variable unknowns: " << static_cast<long>(n_m)
           << "\n";
    }
  }

  // Select the time integrator.
  unique_ptr<ODESolver> ode;
  switch (solver_type) {
    case 0:
      ode = make_unique<ExponentialTrapezoidSolver>();
      break;
    case 1:
      ode = make_unique<ExponentialEulerSolver>();
      break;
    case 2:
      ode = make_unique<BackwardEulerSolver>();
      break;
    case 3:
      ode = make_unique<SDIRK23Solver>(2);  // the L-stable variant
      break;
    case 4:
      ode = make_unique<RK4Solver>();
      break;
    case 5:
      ode = make_unique<ForwardEulerSolver>();
      break;
    default:
      if (Root()) {
        cerr << "Unknown solver type: " << solver_type << "\n";
      }
      return 1;
  }
  ode->Init(visco);
  // Adaptive stepping (-rtol > 0): the n_steps become output times between
  // which the adaptive solver chooses its own steps.
  AdaptiveExponentialTrapezoidSolver adaptive;
  if (rtol > 0.0) {
    adaptive.Init(visco);
    adaptive.SetTolerances(rtol, 1e-12);
  }

  real_t t = 0.0;
  real_t dt = t_final / n_steps;
  real_t dt_adaptive = dt;
  Vector m(visco.Height());
  m = 0.0;

  if (Root() && solver_type >= 4 && dt > 2.5 * visco.MinRelaxationTime()) {
    cout << "Warning: dt = " << dt
         << " exceeds the explicit stability limit of roughly 2.8 tau_min = "
         << 2.8 * visco.MinRelaxationTime()
         << ". Expect blow-up; use an implicit or exponential integrator.\n";
  }

  // Time slices are written through the fields the operator registers.
  ParaViewDataCollection dc("viscoelastic", &mesh);
  if (paraview) {
    dc.SetPrefixPath("ParaView");
    dc.SetLevelsOfDetail(order);
    dc.SetDataFormat(VTKFormat::BINARY);
    dc.SetHighOrderOutput(true);
    visco.RegisterFields(dc);
  }

  // Initial state: relaxed internal variable, elastic response at t = 0.
  if (!visco.SolveElastic(m, t)) {
    if (Root()) {
      cerr << "Elastic solve failed at t = " << t << "\n";
    }
    return 2;
  }
  visco.SyncFields(m);
  if (paraview) {
    dc.SetCycle(0);
    dc.SetTime(t);
    dc.Save();
  }

  // March through time. SolveElastic() makes (u, m) consistent for output;
  // it is free after an exponential-trapezoid or backward-Euler step and
  // costs one solve after an SDIRK, explicit or exponential-Euler one.
  for (int step = 1; step <= n_steps; step++) {
    if (rtol > 0.0) {
      const real_t t_target = step * t_final / n_steps;
      const int n = adaptive.Integrate(m, t, t_target, dt_adaptive);
      if (Root()) {
        cout << "  adaptive: " << n << " steps to t = " << t << ", next dt "
             << dt_adaptive << "\n";
      }
    } else {
      ode->Step(m, t, dt);
    }

    if (!visco.SolveElastic(m, t)) {
      if (Root()) {
        cerr << "Elastic solve failed at t = " << t << "\n";
      }
      return 2;
    }
    visco.SyncFields(m);
    const double m_norm = GlobalNorm(m);
    if (Root()) {
      cout << "step " << step << ", t = " << t << ", ||m||_2 = " << m_norm
           << "\n";
    }

    if (paraview) {
      dc.SetCycle(step);
      dc.SetTime(t);
      dc.Save();
    }
  }

  {
    Vector zero(dim);
    zero = 0.0;
    VectorConstantCoefficient z(zero);
    const double u_norm = problem->Displacement().ComputeL2Error(z);
    if (Root()) {
      cout.precision(10);
      cout << "Final ||u||_L2 = " << u_norm << "\n";
      if (rtol > 0.0) {
        cout << "Adaptive steps: " << adaptive.NumAcceptedSteps()
             << " accepted, " << adaptive.NumRejectedSteps() << " rejected\n";
      }
      cout << "Preconditioner setups: " << problem->NumPreconditionerSetups()
           << "\n";
    }
  }

  // Save the final state in MFEM's native format (one file per rank in
  // parallel).
  {
    ofstream mesh_ofs(RankName("refined.mesh"));
    mesh_ofs.precision(8);
    mesh.Print(mesh_ofs);
    ofstream sol_ofs(RankName("sol.gf"));
    sol_ofs.precision(8);
    problem->Displacement().Save(sol_ofs);
  }

  // Visualise if glvis is open.
  if (visualization) {
    char vishost[] = "localhost";
    int visport = 19916;
    socketstream sol_sock(vishost, visport);
    sol_sock.precision(8);
#ifdef MFEM_USE_MPI
    sol_sock << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank()
             << "\n";
#endif
    sol_sock << "solution\n";
    mesh.Print(sol_sock);
    problem->Displacement().Save(sol_sock);
    sol_sock << flush;
    if (dim == 2) {
      sol_sock << "keys Rjlvvvvvmm\n" << flush;
    } else {
      sol_sock << "keys m\n" << flush;
    }
  }

  return 0;
}
