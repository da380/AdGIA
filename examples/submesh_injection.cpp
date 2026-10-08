// -----------------------------------------------------------------------------
// A tour of SubMeshDofInjection. One source serves the serial and the
// parallel build; the #ifdef blocks below are the tour's actual content —
// they mark exactly what changes in parallel.
//
// Mesh: ../data/circular_offset.msh (meshes/offset_disc.py) — an off-centre
// disk M (attribute 1) inside a larger disk Ω centred on the origin
// (attribute 2 is the surrounding region). Boundary attribute 1 is the
// internal circle ∂M, attribute 2 the outer circle ∂Ω.
//
// Part A. Moving fields between the parent mesh and the submesh.
//
//   The injection is built from a space on the parent mesh and its "shadow"
//   on the submesh (same FE collection object, vdim and ordering — made by
//   MakeShadowSpace). Serially its MultTranspose restricts a parent field
//   to the submesh dof-wise (identical to SubMesh::Transfer, which we
//   verify) and its Mult injects a submesh field back, extended by zero.
//   IN PARALLEL the same roles are played by the true-dof matrix
//
//       Pi = injection.NewTrueDofMatrix()  (parent true dofs × sub true dofs),
//
//   a boolean (±1) HypreParMatrix with Pi^T Pi = I; ParSubMesh inherits
//   the parent's partition, so some ranks may hold no submesh elements,
//   and nothing below needs to care.
//
// Part B. A toy coupled problem, solved through the injection.
//
//   Find φ on Ω and u on M such that
//
//       ∫_Ω ∇φ·∇φ' + ∫_M u φ'  =  ∫_Ω f φ'   for all φ',   φ = 0 on ∂Ω,
//       ∫_M φ u'   - ∫_M u u'  =  0           for all u'.
//
//   The cross terms ∫_M u φ' and ∫_M φ u' are integrals over the submesh in
//   which one field lives on the parent mesh — exactly the structure of the
//   elastogravity coupling ∫_M ρ ∇φ·u'. With the injection they need no
//   custom assembly: with M_sub the plain mass matrix on the submesh,
//
//       serial:    B  = P M_sub  = RemapRows(M_sub)     (pure re-indexing),
//                  Bᵀ = M_sub Pᵀ = RemapColumns(M_sub);
//       parallel:  B  = ParMult(Pi, M̂),  Bᵀ = B->Transpose()
//                  (M̂ the true-dof mass matrix; no communication code),
//
//   and the block system reads
//
//       [ A   B ] [Φ]   [F]
//       [ Bᵀ -M ] [U] = [0],   solved here with MINRES.
//
//   The toy is chosen to be self-checking, in two independent ways:
//
//   1. The second equation says M U = Bᵀ Φ, i.e. u is exactly the dof-wise
//      restriction of φ to the submesh: U = Pᵀ Φ (Pi^T Φ in parallel).
//   2. Eliminating u gives precisely the single-mesh problem "Poisson with
//      a reaction term confined to M": assemble it directly on the parent
//      mesh with an attribute-restricted MassIntegrator and the two
//      solutions must agree to solver tolerance.
//
// Output: the two transfer errors of Part A and, for Part B, the MINRES
// iteration count and the two check errors, all of which should be at
// round-off or solver tolerance; with -vis, φ and u are sent to GLVis.
//
// Options (defaults in brackets):
//   -m    mesh file [../data/circular_offset.msh]: the off-centre
//         disk M (attribute 1) inside the larger disk (attribute 2).
//   -o    finite element order [2].
//   -rt   relative tolerance of the solves [1e-12]: the MINRES block
//         solve and the CG solve of the monolithic check; the two
//         check errors of Part B sit at this level.
//   -vis / -no-vis   send phi and u to GLVis [off].
//
// Sample runs:  ./submesh_injection -o 2 -vis
//               mpiexec -np 4 ./submesh_injection -o 2   (parallel build)
// -----------------------------------------------------------------------------

#include <cmath>
#include <memory>

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
using MatType = HypreParMatrix;
bool Root() { return Mpi::Root(); }
double GlobalMax(double v) {
  return GlobalLpNorm(infinity(), v, MPI_COMM_WORLD);
}
#else
using MeshType = Mesh;
using SubMeshType = SubMesh;
using SpaceType = FiniteElementSpace;
using FieldType = GridFunction;
using FormType = BilinearForm;
using LFType = LinearForm;
using MatType = SparseMatrix;
bool Root() { return true; }
double GlobalMax(double v) { return v; }
#endif

}  // namespace

int main(int argc, char *argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif

  const char *mesh_file = "../data/circular_offset.msh";
  int order = 2;
  real_t rel_tol = 1e-12;
  bool visualization = false;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&rel_tol, "-rt", "--rel-tol", "Solver relative tolerance.");
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

  // ---------------------------------------------------------------------------
  // Meshes and spaces. The u space *is* the shadow space: the restriction of
  // the parent φ space to the submesh.
  // ---------------------------------------------------------------------------
  Mesh smesh(mesh_file, 1, 1);
  const int dim = smesh.Dimension();
#ifdef MFEM_USE_MPI
  MeshType mesh(MPI_COMM_WORLD, smesh);
  smesh.Clear();
#else
  MeshType &mesh = smesh;
#endif

  Array<int> patch_attr({1});
  auto patch = SubMeshType::CreateFromDomain(mesh, patch_attr);

  H1_FECollection fec(order, dim);
  SpaceType fes(&mesh, &fec);
  auto shadow = SubMeshDofInjection::MakeShadowSpace(fes, patch);

  auto injection = SubMeshDofInjection(*shadow, fes);

  // The sizes and, in parallel, the true-dof injection matrix: the one
  // object that replaces the serial dof-wise Mult/MultTranspose.
  const int m = fes.GetTrueVSize();
  const int n = shadow->GetTrueVSize();
#ifdef MFEM_USE_MPI
  auto Pi = injection.NewTrueDofMatrix();
  // Collective on every rank, root prints.
  const auto n_parent_glob = fes.GlobalTrueVSize();
  const auto n_sub_glob = shadow->GlobalTrueVSize();
  if (Root()) {
    cout << "\nGlobal parent true dofs: " << n_parent_glob
         << ",  global submesh true dofs: " << n_sub_glob << endl;
  }
  cout << "  rank " << Mpi::WorldRank() << ": " << m << " parent / " << n
       << " submesh true dofs" << endl;
#else
  cout << "\nParent dofs: " << m << ",  submesh dofs: " << n << endl;
#endif

  // ---------------------------------------------------------------------------
  // Part A: field transfer, both directions, against (Par)SubMesh::Transfer
  // (which works on L-vectors with its own communication; the injection
  // needs none).
  // ---------------------------------------------------------------------------
  auto g_coeff = FunctionCoefficient([](const Vector &x) {
    return sin(3.0 * x[0]) * cos(2.0 * x[1]) + 0.5 * x[0] * x[1];
  });

  FieldType g(&fes);
  g.ProjectCoefficient(g_coeff);
  FieldType g_sub_ref(shadow.get());
  g_sub_ref = 0.0;
  SubMeshType::Transfer(g, g_sub_ref);

#ifdef MFEM_USE_MPI
  // Parallel: everything at true-dof level through Pi.
  Vector g_t(m), g_sub_t(n), g_sub_ref_t(n);
  g.GetTrueDofs(g_t);
  g_sub_ref.GetTrueDofs(g_sub_ref_t);
  Pi->MultTranspose(g_t, g_sub_t);
  g_sub_ref_t -= g_sub_t;
  const double transfer_err = GlobalMax(g_sub_ref_t.Normlinf());

  Vector g_ext_t(m), g_round_t(n);
  Pi->Mult(g_sub_t, g_ext_t);
  Pi->MultTranspose(g_ext_t, g_round_t);
  g_round_t -= g_sub_t;
  const double round_err = GlobalMax(g_round_t.Normlinf());
#else
  // Serial: the injection acts on the grid functions themselves.
  FieldType g_sub(shadow.get());
  injection.MultTranspose(g, g_sub);
  g_sub_ref -= g_sub;
  const double transfer_err = g_sub_ref.Normlinf();

  FieldType g_ext(&fes), g_round(shadow.get());
  injection.Mult(g_sub, g_ext);
  injection.MultTranspose(g_ext, g_round);
  g_round -= g_sub;
  const double round_err = g_round.Normlinf();
#endif

  if (Root()) {
    cout << "\nPart A: field transfer" << endl;
    cout << "  ||P^T g - Transfer(g)||_inf     = " << transfer_err << endl;
    cout << "  ||P^T (P g_sub) - g_sub||_inf   = " << round_err << endl;
  }

  // ---------------------------------------------------------------------------
  // Part B: the coupled toy problem.
  // ---------------------------------------------------------------------------
  auto f_coeff = FunctionCoefficient([](const Vector &x) {
    const real_t dx = x[0] - 0.2, dy = x[1] + 0.3;
    return exp(-4.0 * (dx * dx + dy * dy));
  });
  ConstantCoefficient one(1.0);

  // Dirichlet condition on the outer circle (parent boundary attribute 2).
  Array<int> bdr_marker(mesh.bdr_attributes.Max());
  bdr_marker = 0;
  bdr_marker[1] = 1;
  Array<int> ess_tdof_list;
  fes.GetEssentialTrueDofs(bdr_marker, ess_tdof_list);

  // The elimination of essential dofs below leaves B untouched, which is
  // only right if no essential parent dof lies in the closure of M. That
  // holds for this geometry; check it rather than assume it.
  {
    Array<int> ess_vdofs;
    fes.GetEssentialVDofs(bdr_marker, ess_vdofs);
    for (int i = 0; i < injection.SubVSize(); i++) {
      MFEM_VERIFY(ess_vdofs[injection.ParentVDofs()[i]] == 0,
                  "The submesh touches the essential boundary.");
    }
  }

  // A: stiffness on the parent, with essential elimination.
  FormType a(&fes);
  a.AddDomainIntegrator(new DiffusionIntegrator(one));
  a.Assemble();

  LFType b(&fes);
  b.AddDomainIntegrator(new DomainLFIntegrator(f_coeff));
  b.Assemble();

  FieldType phi(&fes);
  phi = 0.0;

  MatType A;
  Vector Phi, F;
  a.FormLinearSystem(ess_tdof_list, phi, b, A, Phi, F);

  // M: mass on the submesh. B and B^T then come from the injection — by
  // pure re-indexing serially, by hypre products in parallel; no
  // cross-mesh assembly anywhere in either build.
  FormType m_form(shadow.get());
  m_form.AddDomainIntegrator(new MassIntegrator(one));
  m_form.Assemble();
  m_form.Finalize();
#ifdef MFEM_USE_MPI
  unique_ptr<MatType> M_owned(m_form.ParallelAssemble());
  MatType &M = *M_owned;
  unique_ptr<MatType> B(ParMult(Pi.get(), M_owned.get()));  // parent × sub
  unique_ptr<MatType> Bt(B->Transpose());                   // sub × parent
#else
  MatType &M = m_form.SpMat();
  auto B = injection.RemapRows(M);      // = P M   (parent × sub)
  auto Bt = injection.RemapColumns(M);  // = M P^T (sub × parent) = B^T
#endif

  // Block system and MINRES. (The preconditioner is the build's:
  // diagonal smoothing serially, AMG + diagonal scaling in parallel.)
  Array<int> offsets({0, m, m + n});
  BlockOperator block_op(offsets);
  block_op.SetBlock(0, 0, &A);
  block_op.SetBlock(0, 1, B.get());
  block_op.SetBlock(1, 0, Bt.get());
  block_op.SetBlock(1, 1, &M, -1.0);

#ifdef MFEM_USE_MPI
  HypreBoomerAMG prec_A(A);
  prec_A.SetPrintLevel(0);
  HypreDiagScale prec_M(M);
  MINRESSolver minres(MPI_COMM_WORLD);
#else
  DSmoother prec_A(A), prec_M(M);
  MINRESSolver minres;
#endif
  BlockDiagonalPreconditioner prec(offsets);
  prec.SetDiagonalBlock(0, &prec_A);
  prec.SetDiagonalBlock(1, &prec_M);

  BlockVector X(offsets), Rhs(offsets);
  X = 0.0;
  X.GetBlock(0) = Phi;
  Rhs = 0.0;
  Rhs.GetBlock(0) = F;

  minres.SetRelTol(rel_tol);
  minres.SetMaxIter(20000);
  minres.SetPrintLevel(0);
  minres.SetOperator(block_op);
  minres.SetPreconditioner(prec);
  minres.Mult(Rhs, X);

  if (Root()) {
    cout << "\nPart B: block solve" << endl;
    cout << "  MINRES iterations               = "
         << minres.GetNumIterations()
         << (minres.GetConverged() ? "" : "  (NOT converged)") << endl;
  }

  FieldType u(shadow.get());
#ifdef MFEM_USE_MPI
  a.RecoverFEMSolution(X.GetBlock(0), b, phi);
  u.SetFromTrueDofs(X.GetBlock(1));
#else
  // Not RecoverFEMSolution here: in serial legacy assembly the X returned
  // by FormLinearSystem aliases phi's memory and RecoverFEMSolution relies
  // on that aliasing; our solution lives in a BlockVector instead, so copy
  // it back explicitly (conforming serial space: vdofs = tdofs, and the
  // eliminated boundary values were carried through the solve).
  phi = X.GetBlock(0);
  u = X.GetBlock(1);
#endif

  // Check 1: the second block equation forces u to be the dof-wise
  // restriction of φ.
  {
#ifdef MFEM_USE_MPI
    Vector phi_t(m), phi_restricted_t(n);
    phi.GetTrueDofs(phi_t);
    Pi->MultTranspose(phi_t, phi_restricted_t);
    phi_restricted_t -= X.GetBlock(1);
    const double u_err = GlobalMax(phi_restricted_t.Normlinf());
#else
    FieldType phi_restricted(shadow.get());
    injection.MultTranspose(phi, phi_restricted);
    phi_restricted -= u;
    const double u_err = phi_restricted.Normlinf();
#endif
    if (Root()) {
      cout << "  ||u - P^T phi||_inf             = " << u_err
           << "   (solver tolerance)" << endl;
    }
  }

  // Check 2: eliminating u gives the single-mesh problem with the reaction
  // term confined to the patch, assembled here directly on the parent mesh
  // with an attribute marker.
  FieldType phi_mono(&fes);
  phi_mono = 0.0;
  {
    Array<int> patch_marker(mesh.attributes.Max());
    patch_marker = 0;
    patch_marker[0] = 1;

    FormType a_mono(&fes);
    a_mono.AddDomainIntegrator(new DiffusionIntegrator(one));
    a_mono.AddDomainIntegrator(new MassIntegrator(one), patch_marker);
    a_mono.Assemble();

    LFType b_mono(&fes);
    b_mono.AddDomainIntegrator(new DomainLFIntegrator(f_coeff));
    b_mono.Assemble();

    MatType A_mono;
    Vector Phi_mono, F_mono;
    a_mono.FormLinearSystem(ess_tdof_list, phi_mono, b_mono, A_mono, Phi_mono,
                            F_mono);

#ifdef MFEM_USE_MPI
    HypreBoomerAMG prec_mono(A_mono);
    prec_mono.SetPrintLevel(0);
    CGSolver cg(MPI_COMM_WORLD);
#else
    GSSmoother prec_mono(A_mono);
    CGSolver cg;
#endif
    cg.SetRelTol(rel_tol);
    cg.SetMaxIter(20000);
    cg.SetPrintLevel(0);
    cg.SetOperator(A_mono);
    cg.SetPreconditioner(prec_mono);
    cg.Mult(F_mono, Phi_mono);

    a_mono.RecoverFEMSolution(Phi_mono, b_mono, phi_mono);
  }

  {
    FieldType diff(phi_mono);
    diff -= phi;
    const double mono_err = GlobalMax(diff.Normlinf());
    const double phi_norm = GlobalMax(phi.Normlinf());
    if (Root()) {
      cout << "  ||phi - phi_monolithic||_inf    = " << mono_err
           << "   (solver tolerance; ||phi||_inf = " << phi_norm << ")"
           << endl;
    }
  }

  if (visualization) {
    char vishost[] = "localhost";
    int visport = 19916;

    socketstream phi_sock(vishost, visport);
#ifdef MFEM_USE_MPI
    phi_sock << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank()
             << "\n";
#endif
    phi_sock.precision(8);
    phi_sock << "solution\n"
             << mesh << phi << "window_title 'phi on the parent mesh'"
             << flush;
    phi_sock << "keys Rjlbc\n" << flush;

    socketstream u_sock(vishost, visport);
#ifdef MFEM_USE_MPI
    u_sock << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank()
           << "\n";
#endif
    u_sock.precision(8);
    u_sock << "solution\n"
           << patch << u << "window_title 'u on the submesh'" << flush;
    u_sock << "keys Rjlbc\n" << flush;
  }

  return 0;
}
