// ============================================================================
// anisotropic_elasticity.cpp
//
// Static linear elasticity with a transversely isotropic material whose
// symmetry axis is the radial direction (radial anisotropy, as in PREM),
// taken about the centre of the mesh's bounding box, and assembled with
// AdGIA::ElasticTensorIntegrator. The boundary attribute 1 is
// clamped and a uniform body force is applied.
//
// With -iso the Love constants are set to their isotropic values
// (A = C = lambda + 2 mu, F = lambda, L = N = mu, with lambda = mu = 1),
// overriding -A ... -N, and the solution is
// compared with one assembled by mfem::ElasticityIntegrator; the two agree
// to solver tolerance.
//
// One source serves the serial and the parallel build; the only genuine
// differences are the mesh partitioning, the preconditioner (Gauss-Seidel
// serially, BoomerAMG in parallel) and the global reductions on the
// printed norms.
//
// Output: the L2 norm of the displacement (and with -iso the relative
// maximum difference from the mfem::ElasticityIntegrator solution); with
// -vis (on by default) the displacement in GLVis.
//
// Options (defaults in brackets):
//   -m    mesh file [../data/star.mesh].
//   -o    finite element order [1].
//   -r    number of uniform mesh refinements [0].
//   -A    Love constant A [3.1].
//   -C    Love constant C [2.7].
//   -F    Love constant F [1.1].
//   -L    Love constant L [0.9].
//   -N    Love constant N [1.2].
//   -iso / -no-iso   use the isotropic Love constants (lambda = mu =
//         1), overriding -A ... -N, and compare with the
//         mfem::ElasticityIntegrator solution [off].
//   -vis / -no-vis   GLVis windows on or off [on].
//
// Sample runs (with mpiexec -np N in front in a parallel build):
//    ./anisotropic_elasticity -m ../data/star.mesh -o 2
//    ./anisotropic_elasticity -m ../data/beam-tet.mesh -o 1 -iso
//    ./anisotropic_elasticity -m ../data/ball.msh -o 1 -A 3.0 -C 2.6 -F 1.0
// ============================================================================

#include <iostream>
#include <memory>

#include "AdGIA.hpp"

using namespace std;
using namespace mfem;
using namespace AdGIA;

namespace {

#ifdef MFEM_USE_MPI
using MeshType = ParMesh;
using SpaceType = ParFiniteElementSpace;
using FieldType = ParGridFunction;
using FormType = ParBilinearForm;
using LFType = ParLinearForm;
bool Root() { return Mpi::Root(); }
double GlobalMax(double v) {
  double g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return g;
}
#else
using MeshType = Mesh;
using SpaceType = FiniteElementSpace;
using FieldType = GridFunction;
using FormType = BilinearForm;
using LFType = LinearForm;
bool Root() { return true; }
double GlobalMax(double v) { return v; }
#endif

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif

  const char* mesh_file = "../data/star.mesh";
  int order = 1;
  int ref_levels = 0;
  real_t A = 3.1, C = 2.7, F = 1.1, L = 0.9, N = 1.2;
  bool isotropic = false;
  bool visualization = true;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&ref_levels, "-r", "--refinement",
                 "Number of uniform mesh refinements.");
  args.AddOption(&A, "-A", "--love-A", "Love constant A.");
  args.AddOption(&C, "-C", "--love-C", "Love constant C.");
  args.AddOption(&F, "-F", "--love-F", "Love constant F.");
  args.AddOption(&L, "-L", "--love-L", "Love constant L.");
  args.AddOption(&N, "-N", "--love-N", "Love constant N.");
  args.AddOption(&isotropic, "-iso", "--isotropic", "-no-iso", "--no-isotropic",
                 "Use isotropic Love constants (lambda = mu = 1) and compare "
                 "with mfem::ElasticityIntegrator.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization", "GLVis visualisation.");
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

  H1_FECollection fec(order, dim);
  SpaceType fes(&mesh, &fec, dim);
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

  // Material: TI with the radial axis about the mesh centre.
  const real_t lambda = 1.0, mu = 1.0;
  if (isotropic) {
    A = C = lambda + 2.0 * mu;
    F = lambda;
    L = N = mu;
  }
  ConstantCoefficient cA(A), cC(C), cF(F), cL(L), cN(N);
  Vector centre(dim);
  {
    Vector lo, hi;
    mesh.GetBoundingBox(lo, hi);
    for (int i = 0; i < dim; i++) {
      centre[i] = 0.5 * (lo[i] + hi[i]);
    }
  }
  RadialUnitVectorCoefficient axis(dim, centre);
  TransverselyIsotropicElasticTensorCoefficient tensor(dim, cA, cC, cF, cL, cN,
                                                       axis);

  // Boundary conditions and load.
  Array<int> ess_bdr(mesh.bdr_attributes.Max()), ess_tdof_list;
  ess_bdr = 0;
  ess_bdr[0] = 1;
  fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
  Vector g(dim);
  g = 0.0;
  g[dim - 1] = -0.1;
  VectorConstantCoefficient body(g);

  auto solve = [&](BilinearFormIntegrator* integ, FieldType& u) {
    LFType b(&fes);
    b.AddDomainIntegrator(new VectorDomainLFIntegrator(body));
    b.Assemble();
    FormType a(&fes);
    a.AddDomainIntegrator(integ);
    a.Assemble();
    u = 0.0;
    OperatorPtr Amat;
    Vector X, B;
    a.FormLinearSystem(ess_tdof_list, u, b, Amat, X, B);
    // The one solver difference: Gauss-Seidel serially, AMG in parallel.
#ifdef MFEM_USE_MPI
    HypreBoomerAMG prec(*Amat.As<HypreParMatrix>());
    prec.SetPrintLevel(0);
    prec.SetSystemsOptions(dim);
    CGSolver cg(MPI_COMM_WORLD);
#else
    GSSmoother prec(*Amat.As<SparseMatrix>());
    CGSolver cg;
#endif
    cg.SetPreconditioner(prec);
    cg.SetOperator(*Amat);
    cg.SetRelTol(1e-12);
    cg.SetMaxIter(10000);
    cg.SetPrintLevel(IterativeSolver::PrintLevel().Summary());
    cg.Mult(B, X);
    a.RecoverFEMSolution(X, b, u);
  };

  FieldType u(&fes);
  solve(new ElasticTensorIntegrator(tensor), u);
  Vector zero(dim);
  zero = 0.0;
  VectorConstantCoefficient z(zero);
  const double norm = u.ComputeL2Error(z);  // global through FieldType
  if (Root()) {
    cout << "||u||_L2 (anisotropic integrator) = " << norm << "\n";
  }

  if (isotropic) {
    ConstantCoefficient lam(lambda), m(mu);
    FieldType u_ref(&fes);
    solve(new ElasticityIntegrator(lam, m), u_ref);
    u_ref -= u;
    // Normlinf is rank-local: reduce explicitly.
    const double diff = GlobalMax(u_ref.Normlinf()) / GlobalMax(u.Normlinf());
    if (Root()) {
      cout << "||u - u_ref||_inf / ||u||_inf = " << diff << "\n";
    }
    u_ref += u;
  }

  if (visualization) {
    char vishost[] = "localhost";
    int visport = 19916;
    socketstream sol_sock(vishost, visport);
    sol_sock.precision(8);
#ifdef MFEM_USE_MPI
    sol_sock << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank()
             << "\n";
#endif
    sol_sock << "solution\n" << mesh << u << flush;
    sol_sock << (dim == 2 ? "keys Rjlmvvv\n" : "keys RRRilc\n") << std::flush;
  }
  return 0;
}
