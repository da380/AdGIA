/******************************************************************************
## Poisson Solver for Whole-Space Problems

Solves the Poisson equation for the gravitational potential (∇²ϕ = 4πGρ,
with G=1) on the whole space by using a finite computational domain with
transparent boundary conditions, and compares the result with the exact
potential of a uniform disk or ball (uniform_sphere.hpp).

One source serves the serial and the parallel build. The genuine
differences, marked below:
the parallel mesh (with optional refinements after partitioning, -pr),
the comm-constructed DtN and multipole operators (the parallel DtN is
applied through its RAP), the AMG preconditioner in place of
Gauss-Seidel, the comm-aware OrthoSolver, and per-rank output files.

---
### Boundary Condition Methods

1.  **Homogeneous Neumann:** A `∂ϕ/∂n = 0` condition is applied on the
    mesh's exterior boundary. This approach is general but is only
    accurate if the boundary is far from the source density.

2.  **Dirichlet-to-Neumann (DtN):** An exact DtN operator is added to the
    system matrix. For 2D problems, an additional correction is also
    applied to the right-hand-side vector. This method requires a
    spherical exterior boundary.

3.  **Multipole Expansion:** The right-hand-side vector is modified based
    on a multipole expansion of the interior density. This method also
    requires a spherical exterior boundary.

---
### Source Terms

1.  **Reference Problem:** A uniform density (ρ=1) within the mesh's
    first attribute and zero elsewhere.

2.  **Linearised Problem:** The density perturbation caused by a rigid
    translation of the reference density distribution.

---
### Command-Line Options

Options (defaults in brackets):
  -m    mesh file [../data/circular_offset.msh]: attribute 1 is the
        source body, as described above; the DtN and multipole
        methods need a spherical exterior boundary.
  -o    finite element polynomial order [1].
  -r    uniform mesh refinements, before partitioning in the
        parallel build [0].
  -pr   further uniform refinements of the partitioned mesh [0];
        parallel build only.
  -deg  truncation degree of the DtN / multipole expansion [8]; used
        by methods 1 and 2 only.
  -res  set to 1 to output the pointwise error against the exact
        uniform-sphere solution instead of the potential [0]; needs
        a spherical source body.
  -mth  solution method [0]: 0 = Neumann, 1 = DtN, 2 = multipole
        (1 and 2 need a spherical exterior boundary).
  -lin  problem type [0]: 0 = reference density, 1 = the linearised
        density perturbation of a rigid translation.

### Sample runs

    ./poisson_dtn -o 2
    ./poisson_dtn -o 2 -mth 1 -res 1     (DtN, error against the exact solution)
    mpiexec -np 4 ./poisson_dtn -o 2 -mth 2 -lin 1   (parallel build)

The solution (or, with -res 1, the error) is written to refined.mesh and
sol.gf (one file per rank in parallel) and sent to GLVis if a server is
running.

*******************************************************************************/

#include <cassert>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <sstream>

#include "mfem.hpp"
#include "AdGIA.hpp"
#include "uniform_sphere.hpp"

using namespace std;
using namespace mfem;
using namespace AdGIA;

namespace {

#ifdef MFEM_USE_MPI
using MeshType = ParMesh;
using SubMeshType = ParSubMesh;
using SpaceType = ParFiniteElementSpace;
using FieldType = ParGridFunction;
using FormType = ParBilinearForm;
using LFType = ParLinearForm;
using MixedFormType = ParMixedBilinearForm;
using MatType = HypreParMatrix;
bool Root() { return Mpi::Root(); }
#else
using MeshType = Mesh;
using SubMeshType = SubMesh;
using SpaceType = FiniteElementSpace;
using FieldType = GridFunction;
using FormType = BilinearForm;
using LFType = LinearForm;
using MixedFormType = MixedBilinearForm;
using MatType = SparseMatrix;
bool Root() { return true; }
#endif

constexpr real_t pi = std::numbers::pi_v<mfem::real_t>;

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

int main(int argc, char *argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif

  // Set default options.
  const char *mesh_file = "../data/circular_offset.msh";
  int order = 1;
  int refinement = 0;
  int parallel_refinement = 0;
  int degree = 8;
  int residual = 0;
  int method = 0;
  int linearised = 0;

  // Deal with options.
  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree).");
  args.AddOption(&refinement, "-r", "--refinement",
                 "number of mesh refinements (before partitioning)");
  args.AddOption(&parallel_refinement, "-pr", "--parallel_refinement",
                 "number of parallel mesh refinements (parallel build only)");
  args.AddOption(&degree, "-deg", "--degree",
                 "Truncation degree of the DtN / multipole expansion.");
  args.AddOption(&residual, "-res", "--residual",
                 "Output the residual from reference solution");
  args.AddOption(&method, "-mth", "--method",
                 "Solution method: 0 = Neumann, 1 = DtN, 2 = multipole.");
  args.AddOption(&linearised, "-lin", "--linearised",
                 "Solve reference (0) or linearised (1) problem.");

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

  // Read in mesh (serial), refine, and partition in the parallel build
  // (with optional further refinements of the partitioned mesh).
  auto smesh = Mesh(mesh_file, 1, 1);
  auto dim = smesh.Dimension();
  for (int l = 0; l < refinement; l++) {
    smesh.UniformRefinement();
  }
#ifdef MFEM_USE_MPI
  MeshType mesh(MPI_COMM_WORLD, smesh);
  smesh.Clear();
  for (int l = 0; l < parallel_refinement; l++) {
    mesh.UniformRefinement();
  }
#else
  MeshType& mesh = smesh;
#endif

  // Properties of the first attribute.
  auto dom_marker = Array<int>(mesh.attributes.Max());
  dom_marker = 0;
  dom_marker[0] = 1;
  auto bdr_marker = Array<int>(mesh.bdr_attributes.Max());
  bdr_marker = 0;
  bdr_marker[0] = 1;
  auto c1 = MeshCentroid(&mesh, dom_marker);
  auto [found1, same1, r1] = SphericalBoundaryRadius(&mesh, bdr_marker, c1);

  // Properties of the full mesh.
  auto c2 = MeshCentroid(&mesh);
  auto [found2, same2, r2] = SphericalBoundaryRadius(&mesh, c2);

  // If residual from exact solution required, check mesh is appropriate.
  if (residual) {
    assert(found1 == 1 && same1 == 1);
  }

  // Set up the finite element spaces.
  auto L2 = L2_FECollection(order - 1, dim);
  auto H1 = H1_FECollection(order, dim);

  // Space for the potential.
  auto fes = SpaceType(&mesh, &H1);
  // GlobalTrueVSize is collective on first call: every rank calls it,
  // the root prints.
#ifdef MFEM_USE_MPI
  const auto n_dofs = fes.GlobalTrueVSize();
#else
  const auto n_dofs = fes.GetTrueVSize();
#endif
  if (Root()) {
    cout << "Number of finite element unknowns: " << n_dofs << endl;
  }

  // For multipole method we need a discontinuous L2 space.
  std::unique_ptr<SpaceType> dfes;
  if (method == 2) {
    dfes = std::make_unique<SpaceType>(&mesh, &L2);
  }

  // For the linearised problem, we need a discontinuous vector L2 space.
  std::unique_ptr<SpaceType> vfes;
  if (linearised == 1) {
    vfes = std::make_unique<SpaceType>(&mesh, &L2, dim);
  }

  // Assemble the bilinear form for Poisson's equation.
  auto a = FormType(&fes);
  a.AddDomainIntegrator(new DiffusionIntegrator());
  a.Assemble();

  // Assemble mass-shifted bilinear form for preconditioning.
  auto eps = ConstantCoefficient(0.01);
  auto as = FormType(&fes);
  as.AddDomainIntegrator(new DiffusionIntegrator());
  as.AddDomainIntegrator(new MassIntegrator(eps));
  as.Assemble();

  // Set the density coefficient.
  auto rho_coeff1 = ConstantCoefficient(1);
  auto rho_coeff = PWCoefficient();
  rho_coeff.UpdateCoefficient(1, rho_coeff1);

  // Set gridfunction for the potential.
  auto x = FieldType(&fes);

  // Set up the linear form for the rhs.
  auto b = LFType(&fes);

  // Set the constant displacement vector for the linearised problem.
  auto uv = Vector(dim);
  uv = 1.0;

  // Project displacement to a gridfunction if necessary.
  std::unique_ptr<FieldType> u;
  if (linearised == 1) {
    auto uCoeff1 = VectorConstantCoefficient(uv);
    auto uCoeff = PWVectorCoefficient(dim);
    uCoeff.UpdateCoefficient(1, uCoeff1);
    u = std::make_unique<FieldType>(vfes.get());
    u->ProjectCoefficient(uCoeff);
  }

  if (linearised == 0) {
    // For reference problem set up the linear form.
    b.AddDomainIntegrator(new DomainLFIntegrator(rho_coeff));
    b.Assemble();

    // For the DtN method in 2D add the additional term to the rhs.
    if (method == 1 and dim == 2) {
      x = 1.0;
      auto mass = b(x);
      auto l = LFType(&fes);
      auto one = ConstantCoefficient(1);
      auto boundary_marker = ExternalBoundaryMarker(&mesh);
      l.AddBoundaryIntegrator(new BoundaryLFIntegrator(one), boundary_marker);
      l.Assemble();
      auto length = l(x);
      b.Add(-mass / length, l);
    }
  } else {
    // For the linearised problem, form the mixed bilinear form and map the
    // displacement to the rhs.
    auto d = MixedFormType(&fes, vfes.get());
    d.AddDomainIntegrator(new DomainVectorGradScalarIntegrator(rho_coeff));
    d.Assemble();
    d.MultTranspose(*u, b);
  }

  // If using the multipole method, modify the rhs. (The parallel
  // operators are constructed on the communicator.)
  if (method == 2) {
    if (linearised == 0) {
#ifdef MFEM_USE_MPI
      auto c =
          PoissonMultipoleOperator(MPI_COMM_WORLD, dfes.get(), &fes, degree);
#else
      auto c = PoissonMultipoleOperator(dfes.get(), &fes, degree);
#endif
      c.Assemble();
      auto rhof = GridFunction(dfes.get());
      rhof.ProjectCoefficient(rho_coeff);
      c.AddMult(rhof, b, -1);
    } else {
#ifdef MFEM_USE_MPI
      auto c = PoissonLinearisedMultipoleOperator(MPI_COMM_WORLD, vfes.get(),
                                                  &fes, rho_coeff, degree);
#else
      auto c = PoissonLinearisedMultipoleOperator(vfes.get(), &fes, rho_coeff,
                                                  degree);
#endif
      c.Assemble();
      c.AddMult(*u, b, -1);
    }
  }

  // Scale the linear form.
  b *= -4 * pi;

  // Set up the linear system
  x = 0.0;
  Array<int> ess_tdof_list{};
  MatType A;
  Vector B, X;
  a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

  // Set up the preconditioner (the mass shift makes it positive
  // definite): Gauss-Seidel serially, BoomerAMG in parallel.
  MatType As;
  as.FormSystemMatrix(ess_tdof_list, As);
#ifdef MFEM_USE_MPI
  auto P = HypreBoomerAMG(As);
  P.SetPrintLevel(0);
  auto solver = CGSolver(MPI_COMM_WORLD);
#else
  auto P = GSSmoother(As);
  auto solver = CGSolver();
#endif
  solver.SetRelTol(1e-12);
  solver.SetMaxIter(10000);
  solver.SetPrintLevel(1);

  const auto t0 = chrono::steady_clock::now();
  if (method == 1) {
    // The DtN operator: in parallel it is constructed on the
    // communicator and applied through its RAP.
#ifdef MFEM_USE_MPI
    auto c = PoissonDtNOperator(MPI_COMM_WORLD, &fes, degree);
    c.Assemble();
    auto C = c.RAP();
    auto D = SumOperator(&A, 1, &C, 1, false, false);
#else
    auto c = PoissonDtNOperator(&fes, degree);
    c.Assemble();
    auto D = SumOperator(&A, 1, &c, 1, false, false);
#endif
    solver.SetOperator(D);
    solver.SetPreconditioner(P);
    if (dim == 2) {
#ifdef MFEM_USE_MPI
      auto orthoSolver = OrthoSolver(MPI_COMM_WORLD);
#else
      auto orthoSolver = OrthoSolver();
#endif
      orthoSolver.SetSolver(solver);
      orthoSolver.Mult(B, X);
    } else {
      solver.Mult(B, X);
    }
  } else {
    solver.SetOperator(A);
    solver.SetPreconditioner(P);
#ifdef MFEM_USE_MPI
    auto orthoSolver = OrthoSolver(MPI_COMM_WORLD);
#else
    auto orthoSolver = OrthoSolver();
#endif
    orthoSolver.SetSolver(solver);
    orthoSolver.Mult(B, X);
  }
  const auto t1 = chrono::steady_clock::now();
  if (Root()) {
    cout << "Solver time: " << chrono::duration<double>(t1 - t0).count()
         << " s" << endl;
  }

  a.RecoverFEMSolution(X, b, x);

  auto exact = UniformSphereSolution(dim, c1, r1);
  auto exact_coeff =
      linearised == 0 ? exact.Coefficient() : exact.LinearisedCoefficient(uv);

  if (residual == 1) {
    // Subtract exact solution.
    auto y = FieldType(&fes);
    y.ProjectCoefficient(exact_coeff);
    x -= y;
  }

  // Remove mean from the solution.
  {
    auto l = LFType(&fes);
    auto z = FieldType(&fes);
    z = 1.0;
    auto one = ConstantCoefficient(1);
    l.AddDomainIntegrator(new DomainLFIntegrator(one));
    l.Assemble();
    auto area = l(z);
    l /= area;
    auto px = l(x);
    x -= px;
  }

  if (residual == 1) {
    // Relative L2 error of the (mean-free) residual on the source body.
    auto submesh = SubMeshType::CreateFromDomain(mesh, dom_marker);
    auto subfes = SpaceType(&submesh, &H1);
    auto subx = FieldType(&subfes);
    submesh.Transfer(x, subx);
    auto zero = ConstantCoefficient(0);
    auto error = subx.ComputeL2Error(zero);
    subx.ProjectCoefficient(exact_coeff);
    auto norm = subx.ComputeL2Error(zero);
    error /= norm;
    if (Root()) {
      cout << "L2 error: " << error << endl;
    }
  }

  // Write to file (one file per rank in parallel).
  ofstream mesh_ofs(RankName("refined.mesh"));
  mesh_ofs.precision(8);
  mesh.Print(mesh_ofs);

  ofstream sol_ofs(RankName("sol.gf"));
  sol_ofs.precision(8);
  x.Save(sol_ofs);

  // Visualise if glvis is open.
  char vishost[] = "localhost";
  int visport = 19916;
  socketstream sol_sock(vishost, visport);
  sol_sock.precision(8);
#ifdef MFEM_USE_MPI
  sol_sock << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank()
           << "\n";
#endif
  sol_sock << "solution\n" << mesh << x << flush;
  if (dim == 2) {
    sol_sock << "keys Rjlbc\n" << flush;
  } else {
    sol_sock << "keys RRRilmc\n" << flush;
  }
}
