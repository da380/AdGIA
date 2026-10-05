// -----------------------------------------------------------------------------
// Coupled PDE system:
//
// On submesh Ω1 (sphere of radius 1):
//     Δψ1 + ψ2 = f1
//
// On full mesh Ω2 (sphere of radius 2):
//     Δψ2 + ψ1 = f2
//
// Boundary conditions:
//     ψ1 = 0 on ∂Ω1
//     ψ2 = 0 on ∂Ω2
//
// Notes:
// - The system is solved as one block system (CG with a block-diagonal
//   preconditioner), with the cross terms assembled by
//   (Par)SubMeshMixedBilinearForm.
// - Mesh: ../data/coupled_poisson.msh, made by meshes/ball_with_buffer.py.
//   The mesh is deliberately coarse (relative L2 errors of order 0.2 to 0.5)
//   so that it builds and runs quickly; halve the sizes in that script for a
//   converged run.
// - One source serves the serial and the parallel build. The genuine
//   differences, marked below: the parallel mesh, the Par forms and hypre
//   matrices, AMG in place of diagonal smoothing, the global coefficient
//   norms, and the GLVis stream headers.
// - The exact solution is known in closed form (psi1Exact, psi2Exact
//   below, with f1, f2 made from them); the program prints the relative
//   L2 errors of ψ1, of ψ2 and of ψ2 restricted to the submesh. With
//   -vis, ψ1 and ψ2 are shown on the submesh beside their exact values.
//
// Sample runs:  ./coupled_poisson -o 2
//               mpiexec -np 4 ./coupled_poisson -o 2 -vis   (parallel build)
// -----------------------------------------------------------------------------
#include <cmath>

#include "mfem.hpp"
#include "AdGIA.hpp"

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
using SubMixedFormType = ParSubMeshMixedBilinearForm;
using MatType = HypreParMatrix;
bool Root() { return Mpi::Root(); }
double CoeffNorm(Coefficient &c, Mesh &mesh,
                 const IntegrationRule *irs[]) {
  return ComputeGlobalLpNorm(2.0, c, static_cast<ParMesh &>(mesh), irs);
}
#else
using MeshType = Mesh;
using SubMeshType = SubMesh;
using SpaceType = FiniteElementSpace;
using FieldType = GridFunction;
using FormType = BilinearForm;
using LFType = LinearForm;
using SubMixedFormType = SubMeshMixedBilinearForm;
using MatType = SparseMatrix;
bool Root() { return true; }
double CoeffNorm(Coefficient &c, Mesh &mesh,
                 const IntegrationRule *irs[]) {
  return ComputeLpNorm(2.0, c, mesh, irs);
}
#endif

}  // namespace

real_t psi1Exact(const Vector &x) {
  const real_t r = sqrt(x(0) * x(0) + x(1) * x(1) + x(2) * x(2));
  const real_t z = x(2);

  if (r < 1.0) {
    return (1.0 - r * r) * z;
  } else {
    return 0.0;
  }
}

real_t psi2Exact(const Vector &x) {
  const real_t r = sqrt(x(0) * x(0) + x(1) * x(1) + x(2) * x(2));
  const real_t z = x(2);

  if (r < 1.0) {
    return (19.0 / 7.0 - 12.0 / 7.0 * r * r) * z;
  } else {
    return (-1.0 / 7.0 + 8.0 / (7.0 * r * r * r)) * z;
  }
}

real_t f1Exact(const Vector &x) {
  const real_t r = sqrt(x(0) * x(0) + x(1) * x(1) + x(2) * x(2));
  const real_t z = x(2);

  if (r < 1.0) {
    return (-51.0 / 7.0 - 12.0 / 7.0 * r * r) * z;
  } else {
    return (-1.0 / 7.0 + 8.0 / (7.0 * r * r * r)) * z;
  }
}

real_t f2Exact(const Vector &x) {
  const real_t r = sqrt(x(0) * x(0) + x(1) * x(1) + x(2) * x(2));
  const real_t z = x(2);

  if (r < 1.0) {
    return (-113.0 / 7.0 - r * r) * z;
  } else {
    return 0.0;
  }
}

int main(int argc, char *argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif
  StopWatch chrono;

  const char *mesh_file = "../data/coupled_poisson.msh";
  real_t rel_tol = 1e-10;
  int order = 1;
  bool visualization = false;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&rel_tol, "-rt", "--rel-tol", "Relative tolerance.");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization", "Enable or disable GLVis.");
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
  int dim = smesh.Dimension();
#ifdef MFEM_USE_MPI
  MeshType mesh(MPI_COMM_WORLD, smesh);
  smesh.Clear();
#else
  MeshType &mesh = smesh;
#endif

  Array<int> attr_cond;
  attr_cond.Append(1);
  auto mesh_cond = SubMeshType::CreateFromDomain(mesh, attr_cond);

  H1_FECollection fec(order, dim);
  SpaceType fes(&mesh, &fec), fes_cond(&mesh_cond, &fec);

  // Collective calls on every rank, root prints.
#ifdef MFEM_USE_MPI
  const auto n1 = fes_cond.GlobalTrueVSize();
  const auto n2 = fes.GlobalTrueVSize();
#else
  const auto n1 = fes_cond.GetVSize();
  const auto n2 = fes.GetVSize();
#endif
  if (Root()) {
    cout << "Number of psi1-unknowns: " << n1 << endl;
    cout << "Number of psi2-unknowns: " << n2 << endl;
  }

  FieldType psi1_gf(&fes_cond), psi2_gf(&fes), psi2_gf_cond(&fes_cond);
  psi1_gf = 0.0;
  psi2_gf = 0.0;
  psi2_gf_cond = 0.0;

  FunctionCoefficient psi1_exact_coeff(psi1Exact), psi2_exact_coeff(psi2Exact),
      f1_coeff(f1Exact), f2_coeff(f2Exact);

  ConstantCoefficient zero(0.0), one(1.0), minus_one(-1.0);

  Array<int> ess_tdof_list, ess_tdof_list_cond;

  Array<int> bdr_marker(mesh.bdr_attributes.Max());
  bdr_marker = 0;
  bdr_marker[1] = 1;

  Array<int> bdr_marker_cond(mesh_cond.bdr_attributes.Max());
  bdr_marker_cond = 0;
  bdr_marker_cond[0] = 1;

  fes.GetEssentialTrueDofs(bdr_marker, ess_tdof_list);
  fes_cond.GetEssentialTrueDofs(bdr_marker_cond, ess_tdof_list_cond);

  psi1_gf.ProjectBdrCoefficient(zero, bdr_marker_cond);
  psi2_gf.ProjectBdrCoefficient(zero, bdr_marker);

  LFType b1(&fes_cond);
  b1.AddDomainIntegrator(new DomainLFIntegrator(f1_coeff));
  b1.Assemble();
  b1 *= -1.0;

  LFType b2(&fes);
  b2.AddDomainIntegrator(new DomainLFIntegrator(f2_coeff));
  b2.Assemble();
  b2 *= -1.0;

  FormType a11(&fes_cond);
  FormType a22(&fes);

  SubMixedFormType a12(&fes, &fes_cond);
  SubMixedFormType a21(&fes_cond, &fes);

  a11.AddDomainIntegrator(new DiffusionIntegrator(one));
  a11.Assemble();
  a11.Finalize();

  a22.AddDomainIntegrator(new DiffusionIntegrator(one));
  a22.Assemble();
  a22.Finalize();

  a12.AddDomainIntegrator(new MassIntegrator(minus_one));
  a12.Assemble();
  a12.Finalize();

  a21.AddDomainIntegrator(new MassIntegrator(minus_one));
  a21.Assemble();
  a21.Finalize();

  MatType A11, A22;
  Vector X1, B1, X2, B2;

  // B1 and B2 will need further reductions from off-diagonal contributions for
  // non-zero Dirichlet BCs
  a11.FormLinearSystem(ess_tdof_list_cond, psi1_gf, b1, A11, X1, B1);
  a22.FormLinearSystem(ess_tdof_list, psi2_gf, b2, A22, X2, B2);

#ifdef MFEM_USE_MPI
  OperatorHandle A12_handle(Operator::Hypre_ParCSR);
  OperatorHandle A21_handle(Operator::Hypre_ParCSR);
#else
  OperatorHandle A12_handle;
  OperatorHandle A21_handle;
#endif
  a12.FormRectangularSystemMatrix(ess_tdof_list, ess_tdof_list_cond,
                                  A12_handle);
  a21.FormRectangularSystemMatrix(ess_tdof_list_cond, ess_tdof_list,
                                  A21_handle);

  Array<int> block_offsets(3);
  block_offsets[0] = 0;
  block_offsets[1] = X1.Size();
  block_offsets[2] = X2.Size();
  block_offsets.PartialSum();

  BlockVector X(block_offsets), B(block_offsets);
  X = 0.0;
  B = 0.0;

  B.GetBlock(0) = B1;
  B.GetBlock(1) = B2;

  BlockOperator Op(block_offsets);
  Op.SetBlock(0, 0, &A11);
  Op.SetBlock(0, 1, A12_handle.Ptr());
  Op.SetBlock(1, 0, A21_handle.Ptr());
  Op.SetBlock(1, 1, &A22);

  // Diagonal smoothing serially, AMG in parallel.
  BlockDiagonalPreconditioner Prec(block_offsets);
#ifdef MFEM_USE_MPI
  HypreBoomerAMG prec11(A11), prec22(A22);
  prec11.SetPrintLevel(0);
  prec22.SetPrintLevel(0);
  CGSolver solver(MPI_COMM_WORLD);
#else
  DSmoother prec11(A11), prec22(A22);
  CGSolver solver;
#endif
  Prec.SetDiagonalBlock(0, &prec11);
  Prec.SetDiagonalBlock(1, &prec22);

  solver.SetRelTol(rel_tol);
  solver.SetAbsTol(0.0);
  solver.SetMaxIter(3000);
  solver.SetPrintLevel(Root() ? 1 : 0);
  solver.SetOperator(Op);
  solver.SetPreconditioner(Prec);

  chrono.Clear();
  chrono.Start();

  solver.Mult(B, X);

  if (Root()) {
    if (solver.GetConverged()) {
      std::cout << "Converged in " << solver.GetNumIterations()
                << " iterations with a residual norm of "
                << solver.GetFinalNorm() << ".\n";
    } else {
      std::cout << "Did not converge in " << solver.GetNumIterations()
                << " iterations. Residual norm is " << solver.GetFinalNorm()
                << ".\n";
    }
  }

  chrono.Stop();
  if (Root()) {
    cout << "Solver time = " << chrono.RealTime() << " s." << endl;
  }

  a11.RecoverFEMSolution(X.GetBlock(0), b1, psi1_gf);
  a22.RecoverFEMSolution(X.GetBlock(1), b2, psi2_gf);

  mesh_cond.Transfer(psi2_gf, psi2_gf_cond);

  int order_quad = max(2, 2 * order + 1);
  const IntegrationRule *irs[Geometry::NumGeom];
  for (int i = 0; i < Geometry::NumGeom; ++i) {
    irs[i] = &(IntRules.Get(i, order_quad));
  }

  real_t psi1_l2_err = psi1_gf.ComputeL2Error(psi1_exact_coeff, irs) /
                       CoeffNorm(psi1_exact_coeff, mesh_cond, irs);
  real_t psi2_l2_err = psi2_gf.ComputeL2Error(psi2_exact_coeff, irs) /
                       CoeffNorm(psi2_exact_coeff, mesh, irs);
  real_t psi2_cond_l2_err = psi2_gf_cond.ComputeL2Error(psi2_exact_coeff, irs) /
                            CoeffNorm(psi2_exact_coeff, mesh_cond, irs);

  if (Root()) {
    cout << "\nErrors:" << endl;
    cout << "psi1 L2 error = " << psi1_l2_err << endl;
    cout << "psi2 L2 error = " << psi2_l2_err << endl;
    cout << "psi2 (submesh) L2 error = " << psi2_cond_l2_err << endl;
  }

  if (visualization) {
    FieldType psi1_exact_gf(&fes_cond), psi2_exact_gf(&fes),
        psi2_exact_gf_cond(&fes_cond);
    psi1_exact_gf.ProjectCoefficient(psi1_exact_coeff);
    psi2_exact_gf.ProjectCoefficient(psi2_exact_coeff);
    mesh_cond.Transfer(psi2_exact_gf, psi2_exact_gf_cond);

    char vishost[] = "localhost";
    int visport = 19916;

    auto show = [&](const FieldType &f, const char *title) {
      socketstream sock(vishost, visport);
#ifdef MFEM_USE_MPI
      sock << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank()
           << "\n";
#endif
      sock.precision(8);
      sock << "solution\n"
           << mesh_cond << f << "window_title '" << title << "'" << endl;
#ifdef MFEM_USE_MPI
      MPI_Barrier(mesh_cond.GetComm());
#endif
    };
    show(psi1_gf, "psi1 numerical");
    show(psi1_exact_gf, "psi1 exact");
    show(psi2_gf_cond, "psi2 numerical (submesh)");
    show(psi2_exact_gf_cond, "psi2 exact (submesh)");
  }

  return 0;
}
