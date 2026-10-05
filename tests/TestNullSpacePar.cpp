/*
  Parallel tests for the rigid-mode null-space handling in null_space.hpp.
  Run with 1, 2 and 4 ranks; a standalone MPI program returning 1 if any
  check fails, 0 otherwise.

  - MakeRigidModeProjector() on a ParFiniteElementSpace: d(d+1)/2 globally
    orthonormal true-dof vectors, exact null vectors of the free stiffness.
  - ProjectedSolver + CG with a projected BoomerAMG preconditioner converges
    on a free unit square (cube), partitioned in slabs along x, under a
    load with net force and torque; the solution is orthogonal to the rigid
    modes, satisfies the projected equations, and its L2 norm equals the
    serial solve on the full mesh (partition-independent).
  - A warm start carrying a rigid component gives the same solution.
  - ProjectedSolver::SetGauge() with a mass matrix gives zero net momentum
    and angular momentum, and differs from the Euclidean-gauge solution by a
    rigid motion only.
*/

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>

#include "QuasiStaticTestCommon.hpp"
#include "mfem.hpp"
#include "AdGIA.hpp"

using namespace mfem;
using namespace AdGIA;
using namespace elastic_test;

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

void CheckEq(int a, int b, const std::string& what) {
  Check(std::abs(a - b), 0, what + " (" + std::to_string(a) + " vs " +
                                std::to_string(b) + ")");
}

double L2Norm(const GridFunction& u) {
  Vector zero(u.FESpace()->GetVDim());
  zero = 0.0;
  VectorConstantCoefficient z(zero);
  return u.ComputeL2Error(z);  // global for ParGridFunction
}

// Serial reference: the free body solved on the full mesh with the same
// projected CG (Gauss-Seidel preconditioner); returns the L2 norm.
double SerialNorm(Mesh& mesh, FiniteElementCollection& fec, int dim,
                  Coefficient& lambda, Coefficient& mu,
                  VectorCoefficient& pull, Array<int> marker) {
  FiniteElementSpace fes(&mesh, &fec, dim);
  BilinearForm a(&fes);
  a.AddDomainIntegrator(new ElasticityIntegrator(lambda, mu));
  a.Assemble();
  Array<int> empty;
  SparseMatrix A;
  a.FormSystemMatrix(empty, A);
  LinearForm b(&fes);
  b.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(pull), marker);
  b.Assemble();

  auto P = MakeRigidModeProjector(fes);
  GSSmoother M(A);
  ProjectedSolver PM(*P);
  PM.SetSolver(M);
  CGSolver cg;
  cg.SetRelTol(1e-13);
  cg.SetAbsTol(0.0);
  cg.SetMaxIter(10000);
  ProjectedSolver S(*P);
  S.SetSolver(cg);
  S.SetOperator(A);
  cg.SetPreconditioner(PM);
  GridFunction u(&fes);
  u = 0.0;
  Vector X(A.Height());
  X = 0.0;
  S.Mult(b, X);
  u.SetFromTrueDofs(X);
  return L2Norm(u);
}

void RunCase(int dim, int elementType, int order, const std::string& label) {
  auto smesh = MakeSmallMesh(dim, elementType);
  const auto x1_attr = BdrAttributeAt(smesh, 0, 1.0);
  auto marker = Marker(smesh.bdr_attributes.Max(), {x1_attr});

  int nxyz[3] = {Mpi::WorldSize(), 1, 1};
  int* partitioning = smesh.CartesianPartitioning(nxyz);
  ParMesh pmesh(MPI_COMM_WORLD, smesh, partitioning);
  delete[] partitioning;

  H1_FECollection fec(order, dim);
  ParFiniteElementSpace pfes(&pmesh, &fec, dim);
  ConstantCoefficient lambda(kLambda), mu(kMu);
  VectorFunctionCoefficient pull(dim, PullTraction);

  ParBilinearForm a(&pfes);
  a.AddDomainIntegrator(new ElasticityIntegrator(lambda, mu));
  a.Assemble();
  Array<int> empty;
  HypreParMatrix A;
  a.FormSystemMatrix(empty, A);
  ParLinearForm b(&pfes);
  b.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(pull), marker);
  b.Assemble();
  Vector B(pfes.GetTrueVSize());
  b.ParallelAssemble(B);

  // The projector and its basis.
  auto P = MakeRigidModeProjector(pfes);
  CheckEq(P->Size(), dim * (dim + 1) / 2, label + ": number of rigid modes");
  Vector An(A.Height());
  double gram_err = 0.0, null_err = 0.0;
  for (int i = 0; i < P->Size(); i++) {
    for (int j = 0; j < P->Size(); j++) {
      const auto g = P->Dot(P->Basis(i), P->Basis(j));
      gram_err = std::max(gram_err, std::abs(g - (i == j ? 1.0 : 0.0)));
    }
    A.Mult(P->Basis(i), An);
    null_err = std::max(null_err, std::sqrt(P->Dot(An, An)));
  }
  Check(gram_err, 1e-12, label + ": orthonormal basis");
  Check(null_err, 1e-11 * std::sqrt(A.GetGlobalNumRows()),
        label + ": rigid modes are null vectors");
  CheckEq(AddRigidModes(*P, pfes), 0, label + ": modes added twice");

  // Projected CG with a projected BoomerAMG.
  HypreBoomerAMG amg(A);
  amg.SetSystemsOptions(pfes.GetVDim(), true);
  HYPRE_BoomerAMGSetNodal(amg, 4);
  HYPRE_BoomerAMGSetNodalDiag(amg, 1);
  HYPRE_BoomerAMGSetCycleRelaxType(amg, 8, 3);
  amg.SetPrintLevel(0);
  ProjectedSolver PM(*P);
  PM.SetSolver(amg);
  CGSolver cg(MPI_COMM_WORLD);
  cg.SetRelTol(1e-13);
  cg.SetAbsTol(0.0);
  cg.SetMaxIter(10000);
  ProjectedSolver S(*P);
  S.SetSolver(cg);
  S.SetOperator(A);
  cg.SetPreconditioner(PM);

  Vector X(A.Height());
  X = 0.0;
  S.Mult(B, X);
  Check(cg.GetConverged() ? 0.0 : 1.0, 0.0, label + ": CG converged");
  const auto xnorm = std::sqrt(P->Dot(X, X));
  double comp = 0.0;
  for (int i = 0; i < P->Size(); i++) {
    comp = std::max(comp, std::abs(P->Dot(X, P->Basis(i))));
  }
  Check(comp, 1e-12 * xnorm, label + ": solution orthogonal to rigid modes");

  Vector R(A.Height()), PB(A.Height());
  A.Mult(X, R);
  R -= B;
  P->Project(R);
  P->Project(B, PB);
  Check(std::sqrt(P->Dot(R, R)), 1e-9 * std::sqrt(P->Dot(PB, PB)),
        label + ": projected residual");

  ParGridFunction u(&pfes);
  u.SetFromTrueDofs(X);
  const auto serial =
      SerialNorm(smesh, fec, dim, lambda, mu, pull, marker);
  Check(std::abs(L2Norm(u) - serial), 1e-9 * serial,
        label + ": L2 norm matches the serial solve");

  // Mass-weighted gauge: zero momentum, same solution up to a rigid motion.
  {
    ParBilinearForm mass(&pfes);
    mass.AddDomainIntegrator(new VectorMassIntegrator());
    mass.Assemble();
    mass.Finalize();
    std::unique_ptr<HypreParMatrix> M(mass.ParallelAssemble());
    S.SetGauge(M.get());
    S.iterative_mode = false;
    Vector Y(A.Height());
    Y = 0.0;
    S.Mult(B, Y);
    Vector MY(A.Height());
    M->Mult(Y, MY);
    double comp_M = 0.0;
    for (int i = 0; i < P->Size(); i++) {
      comp_M = std::max(comp_M, std::abs(P->Dot(MY, P->Basis(i))));
    }
    Check(comp_M, 1e-12 * std::sqrt(P->Dot(MY, MY)),
          label + ": mass-weighted gauge, zero momentum");
    Vector D(Y);
    D -= X;
    const double dnorm = std::sqrt(P->Dot(D, D));
    P->Project(D);
    Check(std::sqrt(P->Dot(D, D)), 1e-12 * dnorm,
          label + ": mass-weighted gauge differs by a rigid motion");
    S.SetGauge(nullptr);
  }

  // Warm start with a rigid component; absolute tolerance as in the problem
  // layer (the relative one is measured against the restart residual).
  Vector Z(A.Height());
  PM.Mult(B, Z);
  cg.SetAbsTol(1e-13 * std::sqrt(P->Dot(B, Z)));
  S.iterative_mode = true;
  Vector Y(X);
  Y.Add(5.0, P->Basis(0));
  Y.Add(-3.0, P->Basis(P->Size() - 1));
  S.Mult(B, Y);
  Check(cg.GetNumIterations(), 1, label + ": warm start needs no iteration");
  Y -= X;
  Check(std::sqrt(P->Dot(Y, Y)), 1e-9 * xnorm,
        label + ": warm start with rigid component");
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();

  for (auto dim : {2, 3}) {
    for (auto elementType : {0, 1}) {
      for (auto order : {1, 2}) {
        auto label = "dim=" + std::to_string(dim) +
                     " et=" + std::to_string(elementType) +
                     " p=" + std::to_string(order);
        RunCase(dim, elementType, order, label);
      }
    }
  }

  if (Mpi::Root()) {
    if (num_fails == 0) {
      std::cout << "All " << num_checks << " checks passed on "
                << Mpi::WorldSize() << " ranks.\n";
    } else {
      std::cout << num_fails << " of " << num_checks << " checks FAILED on "
                << Mpi::WorldSize() << " ranks.\n";
    }
  }
  return num_fails == 0 ? 0 : 1;
}
