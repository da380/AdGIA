/*
  Parallel tests for the descent toolkit (descent.hpp): the analytic
  toy of TestDescent.cpp — the constrained quadratic over an L2 space —
  on a ParMesh. Run with 1, 2 and 4 ranks; a standalone MPI program
  that exits non-zero if any check fails. Checks: both loops satisfy
  the constrained optimality system (constraint at round-off, projected
  gradient at the floor), and the K = M prior scales the
  Levenberg-Marquardt solution by 1/(1 + lambda).
*/

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>

#include "MixedProblemTestCommon.hpp"
#include "mfem.hpp"
#include "AdGIA.hpp"

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

double GlobalDot(const Vector& a, const Vector& b) {
  double d = a * b;
  double global = 0.0;
  MPI_Allreduce(&d, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return global;
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();

  Mesh serial("../data/elastogravity_two_layer_2d.msh", 1, 1);
  ParMesh mesh(MPI_COMM_WORLD, serial);
  serial.Clear();
  L2_FECollection l2(1, 2);
  ParFiniteElementSpace fes(&mesh, &l2);

  ParBilinearForm m(&fes);
  m.AddDomainIntegrator(new MassIntegrator());
  m.Assemble();
  Array<int> empty;
  OperatorHandle M;
  m.FormSystemMatrix(empty, M);

  FunctionCoefficient fc(
      [](const Vector& x) { return 1.0 + x[0] + std::sin(2.0 * x[1]); });
  ParLinearForm flf(&fes);
  flf.AddDomainIntegrator(new DomainLFIntegrator(fc));
  flf.Assemble();
  Vector f(fes.GetTrueVSize());
  flf.ParallelAssemble(f);

  ConstantCoefficient one(1.0);
  ParLinearForm ell_lf(&fes);
  ell_lf.AddDomainIntegrator(new DomainLFIntegrator(one));
  ell_lf.Assemble();
  Vector ell(fes.GetTrueVSize());
  ell_lf.ParallelAssemble(ell);

  DescentFunctional toy;
  toy.evaluate = [&](const Vector& c, Vector* dual) {
    Vector Mc(c.Size());
    M.Ptr()->Mult(c, Mc);
    if (dual) {
      *dual = Mc;
      *dual -= f;
    }
    return 0.5 * GlobalDot(Mc, c) - GlobalDot(f, c);
  };
  toy.gauss_newton = [&](const Vector& x, Vector& Hx) {
    Hx.SetSize(x.Size());
    M.Ptr()->Mult(x, Hx);
  };

  L2RieszMap riesz(fes);
  const double f_scale = std::sqrt(GlobalDot(f, f));

  DescentOptions options;
  options.max_iterations = 400;
  options.tolerance = 0.0;

  auto check = [&](ConstrainedMetric& metric, const Vector& c,
                   const std::string& what) {
    Check(std::abs(GlobalDot(ell, c)), 1e-9 * f_scale,
          what + ": constraint holds");
    Vector dual(c.Size()), g(c.Size());
    toy.evaluate(c, &dual);
    metric.Gradient(dual, g);
    Vector Mg(g.Size());
    M.Ptr()->Mult(g, Mg);
    Check(std::sqrt(GlobalDot(Mg, g)), 1e-6 * f_scale,
          what + ": projected gradient at the floor");
  };

  Vector c_plain(fes.GetTrueVSize());
  {
    ConstrainedMetric metric(riesz);
    metric.SetConstraint(ell);
    Vector c(fes.GetTrueVSize());
    c = 0.0;
    NonlinearCG(toy, metric, c, options);
    check(metric, c, "cg");

    c_plain = 0.0;
    LevenbergMarquardt(toy, metric, *M.Ptr(), c_plain, options);
    check(metric, c_plain, "lm");
  }
  {
    const double lambda = 0.7;
    ConstrainedMetric metric(riesz);
    metric.SetConstraint(ell);
    metric.SetPrior(*M.Ptr(), lambda);
    Vector c(fes.GetTrueVSize());
    c = 0.0;
    LevenbergMarquardt(toy, metric, *M.Ptr(), c, options);
    Vector scaled(c_plain);
    scaled *= 1.0 / (1.0 + lambda);
    Vector diff(c);
    diff -= scaled;
    Check(std::sqrt(GlobalDot(diff, diff)) /
              std::sqrt(GlobalDot(scaled, scaled)),
          1e-5, "prior scales the solution by 1/(1 + lambda)");
  }

  if (Mpi::Root()) {
    std::cout << (num_fails == 0 ? "PASS" : "FAIL") << ": " << num_checks
              << " checks, " << num_fails << " failures\n";
  }
  return num_fails == 0 ? 0 : 1;
}
