/*
  Parallel tests for MatrixFreePMultigrid (p_multigrid.hpp). Run with 1, 2
  and 4 ranks; a standalone MPI program that fails if any check does.

  The elastic tensor form of a transversely isotropic body on a curved
  mesh of hexahedra, clamped on a face or free with a mass term, at
  orders 3 and 4: the finest level's operator is the assembled matrix,
  the cycle is symmetric and positive, and CG with it reaches the
  solution CG reaches with BoomerAMG on the assembled matrix, in a
  bounded number of iterations.

  Set ADGIA_TEST_DEVICE (e.g. to cuda) to run on a device.
*/

#include <mpi.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <string>

#include "AdGIA.hpp"
#include "mfem.hpp"

using namespace mfem;
using namespace AdGIA;

namespace {

constexpr double kPi = std::numbers::pi;

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

void Jiggle(const Vector& x, Vector& y) {
  y = x;
  const double s = 0.06;
  y[0] += s * std::sin(kPi * x[0]) * std::sin(kPi * x[1]);
  y[1] += s * std::cos(0.5 * kPi * x[0]) * std::sin(kPi * x[1]);
  y[2] += s * std::sin(kPi * x[2]) * std::cos(kPi * x[0]);
}

void Case(int order, bool clamped) {
  const std::string tag = "order " + std::to_string(order) +
                          (clamped ? ", clamped: " : ", free: ");
  auto serial = Mesh::MakeCartesian3D(4, 4, 4, Element::HEXAHEDRON);
  serial.SetCurvature(2);
  serial.Transform(Jiggle);
  ParMesh mesh(MPI_COMM_WORLD, serial);
  serial.Clear();
  H1_FECollection fec(order, 3);
  ParFiniteElementSpace fes(&mesh, &fec, 3);

  ConstantCoefficient A(3.1), C(2.7), F(1.1), L(0.9), N(1.2), shift(0.5);
  Vector x0(3);
  x0 = -0.3;
  RadialUnitVectorCoefficient axis(3, x0);
  TransverselyIsotropicElasticTensorCoefficient ti(3, A, C, F, L, N, axis);
  auto integrators = [&](BilinearForm& form) {
    form.AddDomainIntegrator(new ElasticTensorIntegrator(ti));
    if (!clamped) {
      form.AddDomainIntegrator(new VectorMassIntegrator(shift));
    }
  };
  Array<int> ess_bdr(mesh.bdr_attributes.Max());
  ess_bdr = 0;
  if (clamped) {
    ess_bdr[0] = 1;
  }

  MatrixFreePMultigrid mg(fes, ess_bdr, integrators, {});
  const Operator& op = mg.FineOperator();
  const Array<int>& ess = mg.FineEssentialTrueDofs();

  ParBilinearForm a(&fes);
  integrators(a);
  a.Assemble();
  HypreParMatrix Amat;
  a.FormSystemMatrix(ess, Amat);

  const int n = fes.GetTrueVSize();
  auto dot = [&](const Vector& u, const Vector& v) {
    return InnerProduct(MPI_COMM_WORLD, u, v);
  };
  Vector x(n), y(n), t1(n), t2(n);
  x.Randomize(11 + Mpi::WorldRank());
  y.Randomize(57 + Mpi::WorldRank());
  x.HostReadWrite();
  y.HostReadWrite();
  for (const int i : ess) {
    x[i] = 0.0;
    y[i] = 0.0;
  }
  op.Mult(x, t1);
  Amat.Mult(x, t2);
  t2 -= t1;
  Check(std::sqrt(dot(t2, t2) / dot(t1, t1)), 1e-11,
        tag + "the finest operator is the assembled matrix");

  mg.Mult(x, t1);
  mg.Mult(y, t2);
  const double xMy = dot(x, t2), yMx = dot(y, t1);
  // On a device and more than one rank hypre's AMG for systems (the
  // coarsest level's solver) is symmetric to about 1e-5 only, whatever
  // its smoothers: measured, and the same without this library's
  // settings. CG and MINRES converge with it as elsewhere.
  const bool device_in_parallel =
      Device::Allows(Backend::DEVICE_MASK) && Mpi::WorldSize() > 1;
  Check(std::abs(xMy - yMx) / (std::abs(xMy) + std::abs(yMx)),
        device_in_parallel ? 1e-4 : 1e-9, tag + "the cycle is symmetric");
  Check(-dot(x, t1), 0.0, tag + "the cycle is positive");

  Vector b(n), u(n), u_ref(n);
  b.Randomize(5 + Mpi::WorldRank());
  b.HostReadWrite();
  for (const int i : ess) {
    b[i] = 0.0;
  }
  u = 0.0;
  u_ref = 0.0;
  CGSolver cg(MPI_COMM_WORLD);
  cg.SetOperator(op);
  cg.SetPreconditioner(mg);
  cg.SetRelTol(1e-10);
  cg.SetAbsTol(0.0);
  cg.SetMaxIter(300);
  cg.SetPrintLevel(-1);
  cg.Mult(b, u);
  Check(cg.GetConverged() ? 0.0 : 1.0, 0.0, tag + "CG converges");
  Check(cg.GetNumIterations(), 60, tag + "iterations of CG");

  HypreBoomerAMG amg(Amat);
  amg.SetSystemsOptions(3, fes.GetOrdering() == Ordering::byNODES);
  amg.SetPrintLevel(0);
  CGSolver ref(MPI_COMM_WORLD);
  ref.SetOperator(Amat);
  ref.SetPreconditioner(amg);
  ref.SetRelTol(1e-12);
  ref.SetAbsTol(0.0);
  ref.SetMaxIter(2000);
  ref.SetPrintLevel(-1);
  ref.Mult(b, u_ref);
  u -= u_ref;
  Check(std::sqrt(dot(u, u) / dot(u_ref, u_ref)), 1e-7,
        tag + "the solution of the assembled system");
  if (Mpi::Root()) {
    std::cout << "  " << tag << cg.GetNumIterations()
              << " iterations (BoomerAMG on the matrix: "
              << ref.GetNumIterations() << ")\n";
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();
  const char* name = std::getenv("ADGIA_TEST_DEVICE");
  Device device(name ? name : "cpu");
  for (const int order : {3, 4}) {
    for (const bool clamped : {true, false}) {
      Case(order, clamped);
    }
  }
  if (Mpi::Root()) {
    std::cout << num_checks - num_fails << " of " << num_checks
              << " checks passed\n";
  }
  return num_fails == 0 ? 0 : 1;
}
