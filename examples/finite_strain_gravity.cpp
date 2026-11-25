// ============================================================================
// finite_strain_gravity.cpp
//
// A self-gravitating ball under a surface mass, at finite strain: the
// static problem of NonlinearSelfGravitatingProblem
// (AdGIA/nonlinear_problem.hpp, doc/methods/finite_strain.md) on a
// homogeneous body in hydrostatic equilibrium, with the quadratic law
// about that state, over a ladder of loads. For each load it prints the
// mean radial displacement of the surface in the linear problem and in
// the nonlinear one, their difference over the load squared (which tends
// to a constant: the nonlinear correction is of second order), the Newton
// iterations and the tangents used.
//
// The load is a mass per unit reference area carried by the surface,
// uniform plus a part of degree two about the z axis. With -a2 0 the
// problem is radial and tests/radial_nonlinear_solid.py solves it
// independently: at the load 0.05 the displacement is -1.0452e-2, and
// -9.3408e-3 in the linear problem.
//
// Units G = 1, radius 1, density 1; the bare Lame moduli are 5.
//
// Mesh requirements: a 3-D ball of unit radius (attribute 1) inside a
// buffer shell (attribute 2) with a spherical outer boundary; the default
// is ../data/ball_buffer_coarse_3d.msh.
//
// One source serves the serial and the parallel build.
//
// Options (defaults in brackets):
//   -m    mesh file [../data/ball_buffer_coarse_3d.msh].
//   -o    finite element order [2].
//   -l    the largest load [0.05].
//   -n    number of loads, each half the one after it [3].
//   -a2   the degree-two part of the load over its uniform part [0].
//   -rc   the contraction above which a kept tangent is given up for one
//         made at every iteration [0.1]; 0 makes it at every iteration.
//   -p / -no-p   print each Newton iteration, with the time of its
//         residual and of its linear solve [off].
//
// Sample runs (with mpiexec -np N in front in a parallel build):
//    ./finite_strain_gravity
//    ./finite_strain_gravity -a2 0.5 -l 0.1 -n 4
// ============================================================================

#include <iomanip>
#include <iostream>
#include <numbers>

#include "AdGIA.hpp"

using namespace std;
using namespace mfem;
using namespace AdGIA;

namespace {

#ifdef MFEM_USE_MPI
using MeshType = ParMesh;
using SubMeshType = ParSubMesh;
using SpaceType = ParFiniteElementSpace;
using FormType = ParLinearForm;
using FieldType = ParGridFunction;
bool Root() { return Mpi::Root(); }
#else
using MeshType = Mesh;
using SubMeshType = SubMesh;
using SpaceType = FiniteElementSpace;
using FormType = LinearForm;
using FieldType = GridFunction;
bool Root() { return true; }
#endif

constexpr real_t kPi = numbers::pi_v<real_t>;

// The hydrostatic pressure of the homogeneous ball (G = rho = a = 1).
real_t Pressure(const Vector& x) { return 2.0 * kPi * (1.0 - x * x) / 3.0; }

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif

  const char* mesh_file = "../data/ball_buffer_coarse_3d.msh";
  auto order = 2;
  auto largest_load = real_t{0.05};
  auto n_loads = 3;
  auto degree_two = real_t{0.0};
  auto refresh = real_t{0.1};
  auto print = false;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&largest_load, "-l", "--load", "The largest load.");
  args.AddOption(&n_loads, "-n", "--loads",
                 "Number of loads, each half the one after it.");
  args.AddOption(&degree_two, "-a2", "--degree-two",
                 "The degree-two part of the load over its uniform part.");
  args.AddOption(&refresh, "-rc", "--refresh-contraction",
                 "Contraction above which a kept tangent is given up; 0 "
                 "makes the tangent at every iteration.");
  args.AddOption(&print, "-p", "--print", "-no-p", "--no-print",
                 "Print each Newton iteration, with its times.");
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
  MFEM_VERIFY(serial_mesh.Dimension() == 3 && serial_mesh.attributes.Max() == 2,
              "finite_strain_gravity: a 3-D ball (attribute 1) in a buffer "
              "shell (attribute 2).");
  auto lo = Vector(), hi = Vector();
  serial_mesh.GetBoundingBox(lo, hi);
  const auto r_out = hi.Normlinf();
#ifdef MFEM_USE_MPI
  auto mesh = ParMesh(MPI_COMM_WORLD, serial_mesh);
  serial_mesh.Clear();
#else
  auto& mesh = serial_mesh;
#endif
  auto body_attr = Array<int>({1}), buffer_attr = Array<int>({2});
  auto body = SubMeshType(SubMeshType::CreateFromDomain(mesh, body_attr));
  auto buffer = SubMeshType(SubMeshType::CreateFromDomain(mesh, buffer_attr));

  auto fec = H1_FECollection(order, 3);
  auto fes_u = SpaceType(&body, &fec, 3);
  auto fes_zeta = SpaceType(&mesh, &fec);
  auto fes_buffer = SpaceType(&buffer, &fec, 3);
  auto extension = NewRadialVacuumExtension(fes_u, fes_buffer, 1.0, r_out);

  // The quadratic law about the hydrostatic state: the bare tensor and
  // the equilibrium stress -p0 1.
  auto rho = ConstantCoefficient(1.0);
  auto lambda = ConstantCoefficient(5.0), mu = ConstantCoefficient(5.0);
  auto Xi = IsotropicElasticTensorCoefficient(3, lambda, mu);
  auto T0 = MatrixFunctionCoefficient(3, [](const Vector& x, DenseMatrix& S) {
    S.SetSize(3);
    S = 0.0;
    S(0, 0) = S(1, 1) = S(2, 2) = -Pressure(x);
  });
  auto law = QuadraticMaterialLaw(3, Xi, &T0);

  auto surface = Array<int>(body.bdr_attributes.Max());
  surface = 0;
  surface[body.bdr_attributes.Max() - 1] = 1;

  // The mean radial displacement of the surface, int u . n dS / 4 pi.
  auto one = ConstantCoefficient(1.0);
  auto flux = FormType(&fes_u);
  flux.AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(one), surface);
  flux.Assemble();
  auto mean_radial = [&](GridFunction& u) {
    return flux(static_cast<FieldType&>(u)) / (4.0 * kPi);
  };

  if (Root()) {
    cout << "\n      load       linear    nonlinear  difference / load^2"
            "  iterations  tangents\n";
  }
  for (auto k = n_loads - 1; k >= 0; k--) {
    const auto load = largest_load / (1 << k);
    auto sigma = FunctionCoefficient([load, degree_two](const Vector& x) {
      const auto c = x[2] / x.Norml2();
      return load * (1.0 + degree_two * (3.0 * c * c - 1.0));
    });

    // One iteration from the reference is the linear problem.
    auto linear = NonlinearSelfGravitatingProblem(
        &fes_u, &fes_zeta, &fes_buffer, *extension, law, rho, 1.0, 8);
    linear.SetSurfaceLoad(sigma, surface);
    linear.Options().max_iterations = 1;
    linear.Solve();

    auto problem = NonlinearSelfGravitatingProblem(
        &fes_u, &fes_zeta, &fes_buffer, *extension, law, rho, 1.0, 8);
    problem.SetSurfaceLoad(sigma, surface);
    problem.Options().refresh_contraction = refresh;
    problem.Options().print = print;
    const auto converged = problem.Solve();

    const auto u_linear = mean_radial(linear.Displacement());
    const auto u = mean_radial(problem.Displacement());
    if (Root()) {
      cout << scientific << setprecision(4) << setw(11) << load << setw(13)
           << u_linear << setw(13) << u << setw(15)
           << (u - u_linear) / (load * load) << setw(12)
           << problem.Iterations() << setw(10) << problem.TangentAssemblies()
           << (converged ? "" : "   not converged") << "\n";
    }
  }
  return 0;
}
