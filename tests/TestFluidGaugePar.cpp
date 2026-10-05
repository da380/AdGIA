/*
  Parallel tests for the gauged-fluid option of LinearQuasiStaticProblemBase
  on purely elastic bodies (see TestFluidGauge.cpp for the serial suite and
  the exact references). Run with 1, 2 and 4 ranks; a standalone MPI program
  returning 0 if every check passes, 1 otherwise.

  The exact Lame solution for uniform external pressure with a fluid core is
  partition-independent, so the global relative L2 error against it is the
  check, in 2-D and 3-D. Also checked: the gauge-residual count and decay
  (under this conformal load the first residual sits at the near-kernel
  noise floor; the contraction rate proper is asserted in the serial
  suite), and the epsilon-independence of the refined solution.
*/

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>

#include "mfem.hpp"
#include "AdGIA.hpp"

using namespace mfem;
using namespace AdGIA;

namespace {

constexpr double kKappaS = 2.0;
constexpr double kMuS = 1.0;
constexpr double kKappaF = 1.0;
constexpr double kP0 = 0.01;
constexpr double kRCmb = 3483.0 / 6371.0;

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

std::string BodyMeshFile(int dim) {
  return dim == 2 ? "../data/elastogravity_two_layer_2d.msh"
                  : "../data/elastogravity_three_layer_3d.msh";
}

void UniformPressureTraction(const Vector& x, Vector& f) {
  f = x;
  f *= -kP0 / x.Norml2();
}

// Exact uniform-pressure solution with a fluid core of radius a (see
// TestFluidGauge.cpp).
struct LameSolution {
  int dim;
  double a, alpha, beta, gamma;

  LameSolution(int dim, double a, double b) : dim(dim), a(a) {
    DenseMatrix M(2);
    Vector rhs(2), sol(2);
    const double ad = std::pow(a, dim), bd = std::pow(b, dim);
    if (dim == 2) {
      M(0, 0) = 2.0 * (kKappaS - kKappaF);
      M(0, 1) = -2.0 * (kMuS + kKappaF) / ad;
      M(1, 0) = 2.0 * kKappaS;
      M(1, 1) = -2.0 * kMuS / bd;
    } else {
      M(0, 0) = 3.0 * (kKappaS - kKappaF);
      M(0, 1) = -(4.0 * kMuS + 3.0 * kKappaF) / ad;
      M(1, 0) = 3.0 * kKappaS;
      M(1, 1) = -4.0 * kMuS / bd;
    }
    rhs(0) = 0.0;
    rhs(1) = -kP0;
    M.Invert();
    M.Mult(rhs, sol);
    alpha = sol(0);
    beta = sol(1);
    gamma = alpha + beta / ad;
  }

  void Eval(const Vector& x, Vector& u) const {
    const double r = x.Norml2();
    u = x;
    u *= r < a ? gamma : alpha + beta / std::pow(r, dim);
  }
};

void RunCase(int dim, double tol, const std::string& label) {
  Mesh smesh(BodyMeshFile(dim).c_str(), 1, 1);
  ParMesh pmesh(MPI_COMM_WORLD, smesh);
  const int n_layers = pmesh.attributes.Max() - 1;  // buffer excluded
  Array<int> attrs(n_layers);
  for (int i = 0; i < n_layers; i++) {
    attrs[i] = i + 1;
  }
  ParSubMesh body(ParSubMesh::CreateFromDomain(pmesh, attrs));
  H1_FECollection fec(2, dim);
  ParFiniteElementSpace fes(&body, &fec, dim);

  // Fluid: attribute 1 (2-D) or 1+2 (3-D, both cores); solid: the mantle.
  Vector kappa_vals(n_layers), mu_vals(n_layers);
  Array<int> fluid(n_layers);
  kappa_vals = kKappaF;
  mu_vals = 0.0;
  fluid = 1;
  kappa_vals(n_layers - 1) = kKappaS;
  mu_vals(n_layers - 1) = kMuS;
  fluid[n_layers - 1] = 0;
  PWConstCoefficient kappa(kappa_vals), mu(mu_vals);
  IsotropicElasticRheology rheology(dim, kappa, mu);
  ConstantCoefficient mu_gauge(kKappaF);

  // The surface is the body SubMesh's largest boundary attribute.
  Array<int> surface(body.bdr_attributes.Max());
  surface = 0;
  surface[body.bdr_attributes.Max() - 1] = 1;

  VectorFunctionCoefficient traction(dim, UniformPressureTraction);
  LinearQuasiStaticTractionProblem prob(&fes, rheology, traction, surface);
  prob.SetGaugedFluid(fluid, mu_gauge, 1.0e-2, 3);
  prob.SetMassWeightedGauge();
  prob.AssembleForce(0.0);
  Check(prob.Solve() ? 0.0 : 1.0, 0.0, label + " solve");
  Check(prob.HasGaugedFluid() ? 0.0 : 1.0, 0.0, label + " gauged fluid");

  LameSolution lame(dim, kRCmb, 1.0);
  VectorFunctionCoefficient exact(
      dim, [&lame](const Vector& x, Vector& u) { lame.Eval(x, u); });
  Vector zero(dim);
  zero = 0.0;
  VectorConstantCoefficient z(zero);
  GridFunction& u = const_cast<GridFunction&>(prob.Displacement());
  const double err = u.ComputeL2Error(exact) / u.ComputeL2Error(z);
  Check(err, tol, label + " Lame error");

  // For the uniform load the fluid response is conformal, so the gauge
  // penalty barely acts: the first residual sits at the near-kernel noise
  // floor (the epsilon-contraction proper is asserted on the degree-2 load
  // in the serial suite).
  const auto& res = prob.GaugeResiduals();
  Check(res.size() == 3 ? 0.0 : 1.0, 0.0, label + " residual count");
  if (res.size() == 3) {
    Check(res[0], 1.0e-6, label + " conformal-response residual");
    Check(res[1] / res[0], 0.5, label + " residual decay");
  }

  // Epsilon-independence of the converged solution.
  prob.SetGaugeEpsilon(1.0e-3);
  prob.AssembleForce(0.0);
  Check(prob.Solve() ? 0.0 : 1.0, 0.0, label + " solve, eps 1e-3");
  GridFunction& u2 = const_cast<GridFunction&>(prob.Displacement());
  const double err2 = u2.ComputeL2Error(exact) / u2.ComputeL2Error(z);
  Check(std::abs(err2 - err), 0.5 * tol, label + " eps independence");
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();

  RunCase(2, 1.0e-4, "2d");
  RunCase(3, 1.0e-3, "3d");

  if (Mpi::Root()) {
    if (num_fails == 0) {
      std::cout << "All " << num_checks << " checks passed on "
                << Mpi::WorldSize() << " ranks.\n";
    } else {
      std::cout << num_fails << " of " << num_checks << " checks failed on "
                << Mpi::WorldSize() << " ranks.\n";
    }
  }
  return num_fails == 0 ? 0 : 1;  // an exit status is taken modulo 256
}
