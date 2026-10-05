// ============================================================================
// quasi_static_elasticity.cpp
//
// Driver for the quasi-static linear elastic problems defined in
// AdGIA/quasi_static_problem.hpp, exercising the AssembleForce /
// AddForce / Solve protocol over a sequence of times. See
// AdGIA/quasi_static_problem.hpp for the interface contract.
//
// One source serves the serial and the parallel build. The problem
// classes are serial/parallel in one class, so the genuine differences
// are only the mesh partitioning, the native-format save (one file per
// rank in parallel) and the GLVis stream header.
//
// Output: the solver summary of each step; the time slices in a ParaView
// collection (ParaView/quasi_static, on by default, -no-pv to skip); the
// final displacement in refined.mesh / sol.gf and, with -vis (on by
// default), in GLVis.
//
// Sample runs (with mpiexec -np N in front in a parallel build):
//    ./quasi_static_elasticity -m ../data/star.mesh -o 2 -r 2
//    ./quasi_static_elasticity -m ../data/star.mesh -o 2 -r 2 -inc
//    ./quasi_static_elasticity -m ../data/beam-quad.mesh -p 1 -o 2 -r 1
// ============================================================================

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include "AdGIA.hpp"

/*----------------------------------------------------------------------------
  Driver
----------------------------------------------------------------------------*/

using namespace std;
using namespace mfem;
using namespace AdGIA;

namespace {

#ifdef MFEM_USE_MPI
using MeshType = ParMesh;
using SpaceType = ParFiniteElementSpace;
bool Root() { return Mpi::Root(); }
#else
using MeshType = Mesh;
using SpaceType = FiniteElementSpace;
bool Root() { return true; }
#endif

// A per-rank filename in parallel, the plain name serially.
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
  int order = 1;
  int ref_levels = 0;
  int problem_type = 0;
  real_t t_final = 1.0;
  int n_steps = 10;
  bool demo_increment = false;
  bool paraview = true;
  bool visualization = true;

  // Read in command line options and process.
  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree).");
  args.AddOption(&ref_levels, "-r", "--refinement",
                 "Number of uniform mesh refinements.");
  args.AddOption(&problem_type, "-p", "--problem",
                 "Problem type: 0 = pure traction (any mesh), 1 = clamped "
                 "(needs two boundary attributes, e.g. beam-quad.mesh).");
  args.AddOption(&t_final, "-tf", "--t-final", "Final time.");
  args.AddOption(&n_steps, "-n", "--n-steps", "Number of time steps.");
  args.AddOption(&demo_increment, "-inc", "--increment", "-no-inc",
                 "--no-increment",
                 "Superpose an extra body force through AddForce() to "
                 "demonstrate the increment protocol.");
  args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                 "Save time slices to a ParaView data collection.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Send the final solution to a running GLVis server.");
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

  // Displacement space and material (lambda = mu = 1, so kappa = 1 + 2/d).
  H1_FECollection fec(order, dim);
  SpaceType fes(&mesh, &fec, dim);
  ConstantCoefficient kappa(1.0 + 2.0 / dim), mu(1.0);
  auto rheology = IsotropicElasticRheology(dim, kappa, mu);

  // Loads. Problem 0: a time-scaled uniform traction t -> (0, 1 + t, ...)
  // on all external boundaries (its net force is removed by the traction
  // problem's rigid-mode projection). Problem 1: boundary attribute 1 clamped,
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

  // Construct the requested problem behind the common interface (the
  // classes detect the parallel space themselves).
  unique_ptr<LinearQuasiStaticProblem> problem;
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
  static_cast<LinearQuasiStaticProblemBase&>(*problem).SetPrintLevel(
      IterativeSolver::PrintLevel().Summary());
  // GlobalTrueVSize is collective on first call: every rank calls it,
  // the root prints.
#ifdef MFEM_USE_MPI
  const auto n_u = fes.GlobalTrueVSize();
#else
  const auto n_u = fes.GetTrueVSize();
#endif
  if (Root()) {
    cout << "Displacement unknowns: " << n_u << "\n";
  }

  // Optional demonstration of the AddForce() protocol: any dual vector
  // assembled against DisplacementSpace() may be superposed on the external
  // load. ViscoelasticOperator uses this slot for the effective
  // internal-variable force B^T(C_k m_k) (B^T(2 mu m) for an isotropic
  // body).
  unique_ptr<VectorConstantCoefficient> extra_coef;
  unique_ptr<LinearForm> extra;
  if (demo_increment) {
    Vector g(dim);
    g = 0.0;
    g[0] = 0.1;
    extra_coef = make_unique<VectorConstantCoefficient>(g);
    extra = make_unique<LinearForm>(&problem->DisplacementSpace());
    extra->AddDomainIntegrator(new VectorDomainLFIntegrator(*extra_coef));
    extra->Assemble();
  }

  // Time slices are written through the fields the problem registers
  // (ParaViewDataCollection is parallel-aware by itself).
  ParaViewDataCollection dc("quasi_static", &mesh);
  if (paraview) {
    dc.SetPrefixPath("ParaView");
    dc.SetLevelsOfDetail(order);
    dc.SetDataFormat(VTKFormat::BINARY);
    dc.SetHighOrderOutput(true);
    problem->RegisterFields(dc);
  }

  // March through time: reset the forcing, superpose increments, solve.
  const real_t dt = t_final / n_steps;
  for (int step = 0; step <= n_steps; step++) {
    const real_t t = step * dt;
    if (Root()) {
      cout << "\nstep " << step << ", t = " << t << "\n";
    }

    problem->AssembleForce(t);
    if (extra) {
      problem->AddForce(*extra);
    }
    if (!problem->Solve()) {
      if (Root()) {
        cerr << "Linear solver failed at t = " << t << "\n";
      }
      return 2;
    }

    if (paraview) {
      dc.SetCycle(step);
      dc.SetTime(t);
      dc.Save();
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
