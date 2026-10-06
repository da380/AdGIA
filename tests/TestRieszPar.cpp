/*
  Parallel tests for the Riesz maps (riesz.hpp) on the 2-D two-layer
  disc with its buffer. Run with 1, 2 and 4 ranks; a standalone MPI
  program that exits non-zero if any check fails.

  The checks compare metric scalars against a serial computation every
  rank performs on the full mesh: with a function-assembled dual j_f,
  the pairing j_f . g (the squared metric norm of the gradient) must
  agree between the serial and the parallel map for the L2 metric and
  the Sobolev metrics of orders 1 and 2; the parallel Sobolev gradient
  must vanish on the Dirichlet (outer) dofs; and the order-2 norm of a
  given dual must not exceed the order-1 norm.
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

double Smooth(const Vector& x) {
  return 1.0 + x[0] + 0.5 * x[0] * x[1] + std::sin(3.0 * x[1]);
}

// j_f . Mult(j_f) for a map on the given space, the dual assembled
// from Smooth.
template <class Map, class... Args>
double MetricNorm2(FiniteElementSpace& fes, Args&&... args) {
  Map riesz(fes, std::forward<Args>(args)...);
  FunctionCoefficient f(Smooth);
  auto j = [&]() {
#ifdef MFEM_USE_MPI
    if (auto* pfes = dynamic_cast<ParFiniteElementSpace*>(&fes)) {
      ParLinearForm lf(pfes);
      lf.AddDomainIntegrator(new DomainLFIntegrator(f));
      lf.Assemble();
      Vector J(pfes->GetTrueVSize());
      lf.ParallelAssemble(J);
      return J;
    }
#endif
    LinearForm lf(&fes);
    lf.AddDomainIntegrator(new DomainLFIntegrator(f));
    lf.Assemble();
    return Vector(lf);
  }();
  Vector g(j.Size());
  riesz.Mult(j, g);
  return riesz.Pair(j, g);
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();

  const char* file = "../data/elastogravity_two_layer_2d.msh";

  // Serial reference on every rank.
  double l2_s, s1_s, s2_s;
  {
    Mesh mesh(file, 1, 1);
    L2_FECollection l2(2, 2);
    H1_FECollection h1(2, 2);
    FiniteElementSpace fes_l2(&mesh, &l2);
    FiniteElementSpace fes_h1(&mesh, &h1);
    Array<int> outer(mesh.bdr_attributes.Max());
    outer = 0;
    outer[mesh.bdr_attributes.Max() - 1] = 1;
    l2_s = MetricNorm2<L2RieszMap>(fes_l2);
    s1_s = MetricNorm2<SobolevRieszMap>(fes_h1, 1.0, 0.04, 1, &outer);
    s2_s = MetricNorm2<SobolevRieszMap>(fes_h1, 1.0, 0.04, 2, &outer);
  }

  // Parallel.
  {
    Mesh serial(file, 1, 1);
    ParMesh mesh(MPI_COMM_WORLD, serial);
    serial.Clear();
    L2_FECollection l2(2, 2);
    H1_FECollection h1(2, 2);
    ParFiniteElementSpace fes_l2(&mesh, &l2);
    ParFiniteElementSpace fes_h1(&mesh, &h1);
    Array<int> outer(mesh.bdr_attributes.Max());
    outer = 0;
    outer[mesh.bdr_attributes.Max() - 1] = 1;

    const double l2_p = MetricNorm2<L2RieszMap>(fes_l2);
    const double s1_p =
        MetricNorm2<SobolevRieszMap>(fes_h1, 1.0, 0.04, 1, &outer);
    const double s2_p =
        MetricNorm2<SobolevRieszMap>(fes_h1, 1.0, 0.04, 2, &outer);

    Check(std::abs(l2_p - l2_s) / l2_s, 1e-9, "parallel L2 metric norm");
    Check(std::abs(s1_p - s1_s) / s1_s, 1e-9, "parallel H1 metric norm");
    Check(std::abs(s2_p - s2_s) / s2_s, 1e-9, "parallel H2 metric norm");
    Check(s2_p <= s1_p ? 0.0 : 1.0, 0.5,
          "order-2 norm bounded by order-1 norm");

    // The Dirichlet condition on the parallel gradient.
    SobolevRieszMap riesz(fes_h1, 1.0, 0.04, 2, &outer);
    FunctionCoefficient f(Smooth);
    ParLinearForm lf(&fes_h1);
    lf.AddDomainIntegrator(new DomainLFIntegrator(f));
    lf.Assemble();
    Vector j(fes_h1.GetTrueVSize());
    lf.ParallelAssemble(j);
    Vector g(j.Size());
    riesz.Mult(j, g);
    Array<int> ess;
    fes_h1.GetEssentialTrueDofs(outer, ess);
    double worst = 0.0;
    for (int i = 0; i < ess.Size(); i++) {
      worst = std::max(worst, std::abs(g[ess[i]]));
    }
    double global = 0.0;
    MPI_Allreduce(&worst, &global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    Check(global, 0.0, "Dirichlet dofs of the parallel gradient vanish");
  }

  if (Mpi::Root()) {
    std::cout << (num_fails == 0 ? "PASS" : "FAIL") << ": " << num_checks
              << " checks, " << num_fails << " failures\n";
  }
  return num_fails == 0 ? 0 : 1;
}
