/*
  Parallel tests of the problem classes with the stiffness held with no
  matrix (SetMatrixFree). Run with 1, 2 and 4 ranks; a standalone MPI
  program that fails if any check does.

  - A free elastic body under traction, on hexahedra at order 3: the
    displacement of the default route.
  - The self-gravitating mixed problem on the two-layer ball of
    tetrahedra at order 3, by block MINRES and by the Schur complement:
    the displacement and the potential of the default route.

  The device is ADGIA_TEST_DEVICE (cpu if unset; cuda, ceed-cpu, ...).
*/

#include <mpi.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "AdGIA.hpp"
#include "MixedProblemTestCommon.hpp"
#include "mfem.hpp"

using namespace mfem;
using namespace AdGIA;
using namespace self_grav_test;

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

double RelErr(double a, double b) { return std::abs(a - b) / std::abs(b); }

// A traction on the faces x = 0 and x = 1 with no net force or moment.
void Pull(const Vector& x, Vector& t) {
  t = 0.0;
  if (std::abs(x[0] - 0.5) > 0.49) {
    t[0] = 2.0 * x[0] - 1.0;
  }
}

void ElasticBody() {
  const std::string label = "elastic body";
  auto smesh = Mesh::MakeCartesian3D(3, 3, 3, Element::HEXAHEDRON);
  ParMesh pmesh(MPI_COMM_WORLD, smesh);
  H1_FECollection fec(3, 3);
  ParFiniteElementSpace fes(&pmesh, &fec, 3);
  ConstantCoefficient kappa(2.0), mu(1.0);
  IsotropicElasticRheology rheology(3, kappa, mu);
  VectorFunctionCoefficient pull(3, Pull);
  Array<int> all(pmesh.bdr_attributes.Max());
  all = 1;

  double norm[2];
  long its[2];
  for (const int mf : {0, 1}) {
    LinearQuasiStaticTractionProblem problem(&fes, rheology, pull, all);
    problem.SetMatrixFree(mf == 1);
    problem.SetRelTol(1e-11);
    problem.AssembleForce(0.0);
    const auto before = problem.TotalIterations();
    Check(problem.Solve() ? 0.0 : 1.0, 0.0,
          label + (mf ? ": solve with no matrix" : ": default solve"));
    its[mf] = problem.TotalIterations() - before;
    norm[mf] = L2Norm(problem.Displacement());
  }
  Check(RelErr(norm[1], norm[0]), 1e-8, label + ": |u|");
  Check(its[1] > 0 ? 0.0 : 1.0, 0.0, label + ": iterates");
  if (Mpi::Root()) {
    std::cout << label << ": " << its[1] << " iterations with no matrix, "
              << its[0] << " by default\n";
  }
}

void GravitatingBody() {
  const std::string label = "gravitating body";
  const int dim = 3, order = 3;
  Mesh smesh(MeshFile(dim).c_str(), 1, 1);
  ParMesh pmesh(MPI_COMM_WORLD, smesh);
  ParSubMesh body(ParSubMesh::CreateFromDomain(pmesh, BodyMarker(pmesh)));
  H1_FECollection fec(order, dim);
  ParFiniteElementSpace fes_u(&body, &fec, dim);
  ParFiniteElementSpace fes_phi(&pmesh, &fec);
  ConstantCoefficient kappa(kKappa), mu(kMu), rho(kRho);
  IsotropicElasticRheology rheology(dim, kappa, mu);
  FunctionCoefficient sigma(SurfaceLoad);
  Array<int> surface = SurfaceMarker(body);

  using Problem = LinearQuasiStaticMixedSelfGravitatingProblem;
  for (const auto type :
       {Problem::SolverType::BlockMINRES, Problem::SolverType::SchurCG}) {
    const std::string name =
        type == Problem::SolverType::BlockMINRES ? ", minres" : ", schur";
    // 0: the default; 1: no matrix for the displacement; 2: nor for the
    // potential's preconditioner.
    double u[3], phi[3];
    long its[3];
    for (const int mf : {0, 1, 2}) {
      Problem p(&fes_u, &fes_phi, rheology, rho, kG, kDtNDegree);
      p.SetSurfaceLoad(sigma, surface);
      p.SetRelTol(1e-11);
      p.SetSolverType(type);
      p.SetMatrixFree(mf >= 1);
      p.SetPotentialPreconditionerMatrixFree(mf == 2);
      p.AssembleForce(0.0);
      const auto before = p.TotalIterations();
      Check(p.Solve() ? 0.0 : 1.0, 0.0,
            label + name + (mf ? ": solve with no matrix" : ": default solve"));
      its[mf] = p.TotalIterations() - before;
      u[mf] = L2Norm(p.Displacement());
      phi[mf] = L2Norm(p.Potential());
    }
    Check(RelErr(u[1], u[0]), 1e-7, label + name + ": |u|");
    Check(RelErr(phi[1], phi[0]), 1e-7, label + name + ": |phi|");
    Check(RelErr(u[2], u[0]), 1e-7,
          label + name + ": |u|, the potential's preconditioner too");
    Check(RelErr(phi[2], phi[0]), 1e-7,
          label + name + ": |phi|, the potential's preconditioner too");
    if (Mpi::Root()) {
      std::cout << label << name << ": " << its[1]
                << " iterations with no matrix, " << its[0] << " by default\n";
    }
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();
  const char* name = std::getenv("ADGIA_TEST_DEVICE");
  Device device(name ? name : "cpu");
  ElasticBody();
  GravitatingBody();
  if (Mpi::Root()) {
    std::cout << num_checks - num_fails << " of " << num_checks
              << " checks passed\n";
  }
  return num_fails == 0 ? 0 : 1;
}
