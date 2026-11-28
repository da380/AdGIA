/*
  Parallel tests for PMultigridPreconditioner (p_multigrid.hpp) and for
  the problem classes with it (SetPreconditionerPMultigrid). Run with 1,
  2 and 4 ranks; a standalone MPI program that fails if any check does.

  The preconditioner by itself, on the elasticity matrix of a clamped
  body and of a free one with a mass term, on triangles and tetrahedra
  at orders 3 to 5: the cycle is symmetric and positive, it leaves the
  essential dofs alone, and CG with it reaches the solution CG reaches
  with the AMG, in no more iterations. The clamped and the traction
  problems with the switch on: the displacement of the default
  preconditioner.
*/

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <string>

#include "AdGIA.hpp"
#include "mfem.hpp"

using namespace mfem;
using namespace AdGIA;

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

Mesh MakeBody(int dim) {
  return dim == 2 ? Mesh::MakeCartesian2D(4, 4, Element::TRIANGLE)
                  : Mesh::MakeCartesian3D(2, 2, 2, Element::TETRAHEDRON);
}

// A traction on the faces x = 0 and x = 1 with no net force or moment.
void Pull(const Vector& x, Vector& t) {
  t = 0.0;
  if (std::abs(x[0] - 0.5) > 0.49) {
    t[0] = 2.0 * x[0] - 1.0;
  }
}

double L2Norm(const GridFunction& u) {
  auto zero = Vector(u.FESpace()->GetVDim());
  zero = 0.0;
  auto z = VectorConstantCoefficient(zero);
  return u.ComputeL2Error(z);  // global for a ParGridFunction
}

// CG on A x = b with a preconditioner; the number of iterations, or -1.
int Solve(MPI_Comm comm, const Operator& A, Solver& prec, const Vector& b,
          Vector& x) {
  auto cg = CGSolver(comm);
  cg.SetOperator(A);
  cg.SetPreconditioner(prec);
  cg.SetRelTol(1e-10);
  cg.SetMaxIter(500);
  cg.SetPrintLevel(-1);
  x = 0.0;
  cg.Mult(b, x);
  return cg.GetConverged() ? cg.GetNumIterations() : -1;
}

void RunPreconditioner(int dim, int order, bool clamped,
                       const std::string& label) {
  auto smesh = MakeBody(dim);
  auto pmesh = ParMesh(MPI_COMM_WORLD, smesh);
  auto fec = H1_FECollection(order, dim);
  auto fes = ParFiniteElementSpace(&pmesh, &fec, dim);
  const auto comm = fes.GetComm();

  auto lambda = ConstantCoefficient(1.5), mu = ConstantCoefficient(1.0);
  auto shift = ConstantCoefficient(1e-2);
  auto a = ParBilinearForm(&fes);
  a.AddDomainIntegrator(new ElasticityIntegrator(lambda, mu));
  if (!clamped) {
    a.AddDomainIntegrator(new VectorMassIntegrator(shift));
  }
  a.Assemble();
  auto ess_bdr = Array<int>(pmesh.bdr_attributes.Max());
  ess_bdr = 0;
  if (clamped) {
    ess_bdr[0] = 1;
  }
  auto ess = Array<int>();
  fes.GetEssentialTrueDofs(ess_bdr, ess);
  auto A = HypreParMatrix();
  a.FormSystemMatrix(ess, A);

  auto pmg = PMultigridPreconditioner(fes, A, ess, {});
  const auto expected = PMultigridOrders(order, 2);
  Check(pmg.Orders() == expected ? 0.0 : 1.0, 0.0, label + ": the levels");

  // Symmetric and positive; nothing at the essential dofs from a
  // right-hand side that has nothing there.
  auto x = Vector(A.Height()), y = Vector(A.Height());
  auto Mx = Vector(A.Height()), My = Vector(A.Height());
  x.Randomize(1 + Mpi::WorldRank());
  y.Randomize(101 + Mpi::WorldRank());
  x.SetSubVector(ess, 0.0);
  y.SetSubVector(ess, 0.0);
  pmg.Mult(x, Mx);
  pmg.Mult(y, My);
  const auto xMy = InnerProduct(comm, x, My), yMx = InnerProduct(comm, y, Mx);
  const auto xMx = InnerProduct(comm, x, Mx);
  Check(std::abs(xMy - yMx) / std::abs(xMx), 1e-10, label + ": symmetric");
  Check(xMx > 0.0 ? 0.0 : 1.0, 0.0, label + ": positive");
  auto at_ess = Vector();
  Mx.GetSubVector(ess, at_ess);
  auto local = at_ess.Size() ? at_ess.Normlinf() : 0.0, global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, comm);
  Check(global, 0.0, label + ": essential dofs left alone");

  // CG against the AMG.
  auto b = Vector(A.Height());
  b.Randomize(7 + Mpi::WorldRank());
  b.SetSubVector(ess, 0.0);
  auto amg = HypreBoomerAMG(A);
  amg.SetSystemsOptions(dim, false);
  amg.SetPrintLevel(0);
  auto x_amg = Vector(A.Height()), x_pmg = Vector(A.Height());
  const auto its_amg = Solve(comm, A, amg, b, x_amg);
  const auto its_pmg = Solve(comm, A, pmg, b, x_pmg);
  Check(its_amg > 0 ? 0.0 : 1.0, 0.0, label + ": CG with the AMG converges");
  Check(its_pmg > 0 ? 0.0 : 1.0, 0.0,
        label + ": CG with the p-multigrid converges");
  x_pmg -= x_amg;
  Check(std::sqrt(InnerProduct(comm, x_pmg, x_pmg) /
                  InnerProduct(comm, x_amg, x_amg)),
        1e-7, label + ": the same solution");
  // Measured on 1, 2 and 4 ranks: 18 to 27 iterations on triangles and
  // 22 to 67 on tetrahedra, where the AMG takes 54 to 202.
  Check(its_pmg, its_amg, label + ": no more iterations than the AMG");
  if (Mpi::Root()) {
    std::cout << label << ": " << its_pmg << " iterations, " << its_amg
              << " with the AMG\n";
  }
}

void RunProblems(int dim, int order, const std::string& label) {
  auto smesh = MakeBody(dim);
  auto pmesh = ParMesh(MPI_COMM_WORLD, smesh);
  auto fec = H1_FECollection(order, dim);
  auto fes = ParFiniteElementSpace(&pmesh, &fec, dim);
  auto kappa = ConstantCoefficient(2.0), mu = ConstantCoefficient(1.0);
  auto rheology = IsotropicElasticRheology(dim, kappa, mu);
  auto pull = VectorFunctionCoefficient(dim, Pull);
  auto all = Array<int>(pmesh.bdr_attributes.Max());
  all = 1;
  auto ess_bdr = Array<int>(pmesh.bdr_attributes.Max());
  ess_bdr = 0;
  ess_bdr[0] = 1;
  auto loaded = Array<int>(all);
  loaded[0] = 0;

  const auto run = [&](LinearQuasiStaticProblemBase& problem, bool pmg,
                       const std::string& what, double& norm, long& its) {
    problem.SetPreconditionerPMultigrid(pmg);
    problem.SetRelTol(1e-11);
    problem.AssembleForce(0.0);
    const auto before = problem.TotalIterations();
    Check(problem.Solve() ? 0.0 : 1.0, 0.0, label + ": " + what + " solve");
    its = problem.TotalIterations() - before;
    norm = L2Norm(problem.Displacement());
  };

  {
    auto with_amg =
        LinearQuasiStaticClampedProblem(&fes, rheology, ess_bdr, pull, loaded);
    auto with_pmg =
        LinearQuasiStaticClampedProblem(&fes, rheology, ess_bdr, pull, loaded);
    double n_amg, n_pmg;
    long i_amg, i_pmg;
    run(with_amg, false, "clamped, AMG", n_amg, i_amg);
    run(with_pmg, true, "clamped, p-multigrid", n_pmg, i_pmg);
    Check(std::abs(n_pmg - n_amg) / n_amg, 1e-8, label + ": clamped norm");
    Check(i_pmg > 0 ? 0.0 : 1.0, 0.0, label + ": clamped iterates");
    Check(i_pmg, i_amg, label + ": clamped, no more iterations");
    if (Mpi::Root()) {
      std::cout << label << ": clamped, " << i_pmg << " iterations, " << i_amg
                << " with the AMG\n";
    }
  }
  {
    auto with_amg = LinearQuasiStaticTractionProblem(&fes, rheology, pull, all);
    auto with_pmg = LinearQuasiStaticTractionProblem(&fes, rheology, pull, all);
    double n_amg, n_pmg;
    long i_amg, i_pmg;
    run(with_amg, false, "free, AMG", n_amg, i_amg);
    run(with_pmg, true, "free, p-multigrid", n_pmg, i_pmg);
    Check(std::abs(n_pmg - n_amg) / n_amg, 1e-8, label + ": free norm");
    Check(i_pmg > 0 ? 0.0 : 1.0, 0.0, label + ": free iterates");
    if (Mpi::Root()) {
      std::cout << label << ": free, " << i_pmg << " iterations, " << i_amg
                << " with the AMG\n";
    }
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();

  for (const auto dim : {2, 3}) {
    for (const auto order : {3, 4, 5}) {
      if (dim == 3 && order == 5) {
        continue;
      }
      for (const auto clamped : {true, false}) {
        RunPreconditioner(dim, order, clamped,
                          "dim=" + std::to_string(dim) +
                              " p=" + std::to_string(order) +
                              (clamped ? " clamped" : " free"));
      }
    }
    RunProblems(dim, 3, "problems dim=" + std::to_string(dim) + " p=3");
  }

  if (Mpi::Root()) {
    std::cout << num_checks - num_fails << " of " << num_checks
              << " checks passed\n";
  }
  return num_fails == 0 ? 0 : 1;
}
