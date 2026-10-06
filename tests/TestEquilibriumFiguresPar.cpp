/*
  Parallel tests for the density feasibility functional
  (equilibrium_figures.hpp): the fluid-only variant on the 3-D
  three-layer model, run on a ParMesh with the fluid as a ParSubMesh.
  Run with 1, 2 and 4 ranks; a standalone MPI program that exits
  non-zero if any check fails.

  Checks: the parallel value equals the serial value computed by every
  rank on the full mesh; the envelope-theorem derivative paired with a
  deterministic (function-projected) direction matches a central finite
  difference of the parallel value; and that directional derivative
  equals the serial one.
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

constexpr int kDtNDegree = 8;
constexpr double kGravG = 0.05;
constexpr double kStep = 1e-3;

double FluidRho(const Vector& x) {
  const double r2 = x * x;
  return 1.2 - 0.3 * r2 + 0.1 * x[1];
}

// The FD direction: deterministic in space, so that the serial and the
// parallel runs perturb the same density.
double Direction(const Vector& x) {
  return std::sin(3.0 * x[0]) * std::cos(2.0 * x[1]) + 0.3 * x[2];
}

// The density by attribute (as in TestEquilibriumFigures.cpp).
class ModelDensity {
 public:
  ModelDensity()
      : inner_(1.3), fluid_(FluidRho), mantle_(1.0), buffer_(0.0) {
    pw_.UpdateCoefficient(1, inner_);
    pw_.UpdateCoefficient(2, fluid_);
    pw_.UpdateCoefficient(3, mantle_);
    pw_.UpdateCoefficient(4, buffer_);
  }
  Coefficient& Get() { return pw_; }

 private:
  ConstantCoefficient inner_, mantle_, buffer_;
  FunctionCoefficient fluid_;
  PWCoefficient pw_;
};

// J (and optionally the directional derivative against `direction`) of
// the fluid-only variant with the fluid density perturbed by
// eps * direction, serial or parallel by the types of the arguments.
double Evaluate(FiniteElementSpace& fes_phi, FiniteElementSpace& fes_u,
                FiniteElementSpace& fes_p, FiniteElementSpace& fes_rho,
                const Array<int>& ess, double eps, Coefficient& direction,
                double* directional = nullptr) {
  ProductCoefficient dc(eps, direction);
  ModelDensity rho0_parent, rho0_stokes;
  // The perturbation is analytic, so it serves the parent and the
  // Stokes mesh alike, restricted to the fluid by attribute.
  PWCoefficient dc_parent;
  dc_parent.UpdateCoefficient(2, dc);
  SumCoefficient rho_parent(rho0_parent.Get(), dc_parent);
  SumCoefficient rho_stokes(rho0_stokes.Get(), dc);

  DensityFeasibility J(fes_phi, kDtNDegree, kGravG, rho_parent,
                       Array<int>({2}), fes_u, fes_p, rho_stokes, nullptr,
                       &ess);
  if (directional) {
    Vector dual;
    J.Derivative(fes_rho, dual);
    std::unique_ptr<GridFunction> d;
    auto* pfes = dynamic_cast<ParFiniteElementSpace*>(&fes_rho);
    if (pfes) {
      d = std::make_unique<ParGridFunction>(pfes);
    } else {
      d = std::make_unique<GridFunction>(&fes_rho);
    }
    d->ProjectCoefficient(direction);
    Vector d_true(fes_rho.GetTrueVSize());
    d->GetTrueDofs(d_true);
    double dot = dual * d_true;
    if (pfes) {
      double global = 0.0;
      MPI_Allreduce(&dot, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      dot = global;
    }
    *directional = dot;
  }
  return J.Value();
}

// (H d) . d at the base state, with the same deterministic direction:
// the Gauss-Newton machinery's serial-parallel agreement check.
double HessianPairing(FiniteElementSpace& fes_phi,
                      FiniteElementSpace& fes_u, FiniteElementSpace& fes_p,
                      FiniteElementSpace& fes_rho, const Array<int>& ess,
                      Coefficient& direction) {
  ModelDensity rho0;
  DensityFeasibility base(fes_phi, kDtNDegree, kGravG, rho0.Get(),
                          Array<int>({2}), fes_u, fes_p, rho0.Get(),
                          nullptr, &ess);
  // The direction's parent twin vanishes off the fluid.
  PWCoefficient d_parent;
  d_parent.UpdateCoefficient(2, direction);
  Vector Hd;
  base.HessianAction(rho0.Get(), rho0.Get(), d_parent, direction, fes_rho,
                     Hd);

  std::unique_ptr<GridFunction> d;
  auto* pfes = dynamic_cast<ParFiniteElementSpace*>(&fes_rho);
  if (pfes) {
    d = std::make_unique<ParGridFunction>(pfes);
  } else {
    d = std::make_unique<GridFunction>(&fes_rho);
  }
  d->ProjectCoefficient(direction);
  Vector d_true(fes_rho.GetTrueVSize());
  d->GetTrueDofs(d_true);
  double dot = Hd * d_true;
  if (pfes) {
    double global = 0.0;
    MPI_Allreduce(&dot, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    dot = global;
  }
  return dot;
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();

  FunctionCoefficient direction(Direction);

  // Every rank: the serial problem on the full mesh.
  double j_serial = 0.0, dd_serial = 0.0, hdd_serial = 0.0;
  {
    Mesh parent(ThreeLayerMeshFile(3), 1, 1);
    auto fluid = SubMesh::CreateFromDomain(parent, Array<int>({2}));
    H1_FECollection h1_phi(2, 3), h1_u(2, 3), h1_p(1, 3);
    L2_FECollection l2(2, 3);
    FiniteElementSpace fes_phi(&parent, &h1_phi);
    FiniteElementSpace fes_u(&fluid, &h1_u, 3);
    FiniteElementSpace fes_p(&fluid, &h1_p);
    FiniteElementSpace fes_rho(&fluid, &l2);
    Array<int> ess(fluid.bdr_attributes.Max());
    ess = 1;
    j_serial =
        Evaluate(fes_phi, fes_u, fes_p, fes_rho, ess, 0.0, direction,
                 &dd_serial);
    hdd_serial =
        HessianPairing(fes_phi, fes_u, fes_p, fes_rho, ess, direction);
  }

  // The parallel problem.
  {
    Mesh serial(ThreeLayerMeshFile(3), 1, 1);
    ParMesh parent(MPI_COMM_WORLD, serial);
    serial.Clear();
    auto fluid = ParSubMesh::CreateFromDomain(parent, Array<int>({2}));
    H1_FECollection h1_phi(2, 3), h1_u(2, 3), h1_p(1, 3);
    L2_FECollection l2(2, 3);
    ParFiniteElementSpace fes_phi(&parent, &h1_phi);
    ParFiniteElementSpace fes_u(&fluid, &h1_u, 3);
    ParFiniteElementSpace fes_p(&fluid, &h1_p);
    ParFiniteElementSpace fes_rho(&fluid, &l2);
    Array<int> ess(fluid.bdr_attributes.Max());
    ess = 1;

    double dd = 0.0;
    const double j0 =
        Evaluate(fes_phi, fes_u, fes_p, fes_rho, ess, 0.0, direction, &dd);
    Check(std::abs(j0 - j_serial) / j_serial, 1e-8,
          "parallel J equals serial J");
    Check(std::abs(dd - dd_serial) / std::abs(dd_serial), 1e-7,
          "parallel directional derivative equals serial");

    const double jp =
        Evaluate(fes_phi, fes_u, fes_p, fes_rho, ess, kStep, direction);
    const double jm =
        Evaluate(fes_phi, fes_u, fes_p, fes_rho, ess, -kStep, direction);
    const double fd = (jp - jm) / (2.0 * kStep);
    Check(std::abs(fd - dd) / std::max(std::abs(fd), j0), 2e-5,
          "derivative matches the central difference");

    const double hdd =
        HessianPairing(fes_phi, fes_u, fes_p, fes_rho, ess, direction);
    Check(std::abs(hdd - hdd_serial) / std::abs(hdd_serial), 1e-6,
          "parallel Hessian pairing equals serial");
  }

  if (Mpi::Root()) {
    std::cout << (num_fails == 0 ? "PASS" : "FAIL") << ": " << num_checks
              << " checks, " << num_fails << " failures\n";
  }
  return num_fails == 0 ? 0 : 1;
}
