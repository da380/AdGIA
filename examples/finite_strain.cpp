// ============================================================================
// finite_strain.cpp
//
// A clamped beam bent by a dead load on its end, at finite strain: the
// static problem of AdGIA/nonlinear_problem.hpp with the quadratic law
// of AdGIA/material_law.hpp (doc/methods/finite_strain.md), over a
// ladder of loads. For each load it prints the largest displacement of
// the linear problem and of the nonlinear one, their relative
// difference, the Newton iterations and the assemblies of the tangent.
// The difference grows in proportion to the load: the nonlinear
// correction is of second order.
//
// Mesh requirements: any MFEM mesh, 2-D or 3-D, with boundary attribute
// 1 clamped and boundary attribute 2 loaded; the default is the stock
// ../data/beam-quad.mesh. The material is dimensionless and uniform
// (lambda = 50, mu = 50), with a natural reference state.
//
// One source serves the serial and the parallel build: the problem class
// is serial or parallel by its space, and the only difference here is the
// partitioning of the mesh.
//
// Options (defaults in brackets):
//   -m    mesh file [../data/beam-quad.mesh].
//   -o    finite element order [2].
//   -r    uniform mesh refinements [1].
//   -l    the largest load, a traction on attribute 2 across the beam
//         [0.04].
//   -n    number of loads, each half the one after it [4].
//   -rc   the contraction above which the tangent of the reference is
//         given up for one assembled at every iteration [0.1]; 0
//         assembles it at every iteration from the start.
//   -pv / -no-pv   save the last displacement to a ParaView collection
//         (ParaView/finite_strain) [off].
//
// Sample runs (with mpiexec -np N in front in a parallel build):
//    ./finite_strain
//    ./finite_strain -m ../data/beam-tet.mesh -o 2 -r 0
//    ./finite_strain -rc 0 -l 0.2
// ============================================================================

#include <iomanip>
#include <iostream>
#include <vector>

#include "AdGIA.hpp"

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

// The largest nodal displacement, over every rank.
real_t Largest(const GridFunction& u) {
  auto largest = u.Normlinf();
#ifdef MFEM_USE_MPI
  auto local = largest;
  MPI_Allreduce(&local, &largest, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                MPI_COMM_WORLD);
#endif
  return largest;
}

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif

  const char* mesh_file = "../data/beam-quad.mesh";
  auto order = 2;
  auto ref_levels = 1;
  auto largest_load = real_t{0.04};
  auto n_loads = 4;
  auto refresh = real_t{0.1};
  auto paraview = false;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&ref_levels, "-r", "--refinement",
                 "Number of uniform mesh refinements.");
  args.AddOption(&largest_load, "-l", "--load",
                 "The largest load: a traction across the beam on boundary "
                 "attribute 2.");
  args.AddOption(&n_loads, "-n", "--loads",
                 "Number of loads, each half the one after it.");
  args.AddOption(&refresh, "-rc", "--refresh-contraction",
                 "Contraction above which the tangent of the reference is "
                 "given up for one assembled at every iteration; 0 assembles "
                 "it at every iteration from the start.");
  args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                 "Save the last displacement to a ParaView collection.");
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

  auto serial_mesh = Mesh(mesh_file, 1, 1);
  const auto dim = serial_mesh.Dimension();
  for (auto l = 0; l < ref_levels; l++) {
    serial_mesh.UniformRefinement();
  }
#ifdef MFEM_USE_MPI
  auto mesh = ParMesh(MPI_COMM_WORLD, serial_mesh);
  serial_mesh.Clear();
#else
  auto& mesh = serial_mesh;
#endif
  MFEM_VERIFY(mesh.bdr_attributes.Max() >= 2,
              "finite_strain: the mesh needs boundary attributes 1 "
              "(clamped) and 2 (loaded).");

  auto fec = H1_FECollection(order, dim);
  auto fes = SpaceType(&mesh, &fec, dim);

  // The quadratic law about a natural state.
  auto lambda = ConstantCoefficient(50.0), mu = ConstantCoefficient(50.0);
  auto Xi = IsotropicElasticTensorCoefficient(dim, lambda, mu);
  auto law = QuadraticMaterialLaw(dim, Xi);

  auto clamped = Array<int>(mesh.bdr_attributes.Max());
  auto loaded = Array<int>(mesh.bdr_attributes.Max());
  clamped = 0;
  clamped[0] = 1;
  loaded = 0;
  loaded[1] = 1;

  if (Root()) {
    cout << "\n      load       linear    nonlinear   difference  iterations"
            "  tangents\n";
  }
  for (auto k = n_loads - 1; k >= 0; k--) {
    const auto load = largest_load / (1 << k);
    auto pull = Vector(dim);
    pull = 0.0;
    pull[dim - 1] = -load;
    auto traction = VectorConstantCoefficient(pull);

    // One iteration from the reference is the linear problem.
    auto linear = NonlinearClampedProblem(&fes, law, clamped, traction, loaded);
    linear.Options().max_iterations = 1;
    linear.Solve();

    auto problem =
        NonlinearClampedProblem(&fes, law, clamped, traction, loaded);
    problem.Options().refresh_contraction = refresh;
    const auto converged = problem.Solve();

    auto difference = GridFunction(problem.Displacement());
    difference -= linear.Displacement();
    const auto size = Largest(problem.Displacement());
    if (Root()) {
      cout << scientific << setprecision(3) << setw(10) << load << setw(13)
           << Largest(linear.Displacement()) << setw(13) << size << setw(13)
           << Largest(difference) / size << setw(12) << problem.Iterations()
           << setw(10) << problem.TangentAssemblies()
           << (converged ? "" : "   not converged") << "\n";
    }

    if (paraview && k == 0) {
      auto pv = ParaViewDataCollection("finite_strain", &mesh);
      pv.SetPrefixPath("ParaView");
      pv.SetLevelsOfDetail(order);
      pv.SetHighOrderOutput(true);
      pv.RegisterField("displacement", &problem.Displacement());
      pv.Save();
    }
  }
  return 0;
}
