/*
  Parallel twin of TestAdjointFingerprint (gate GA1b): the sea-level
  reciprocity certificates of Al-Attar et al. (2024,
  doc/Elasticity/ggad434.pdf) on the water-loaded mixed solve — the
  spherical, 3-D-lateral-structure and rotation+structure legs on the
  canned 3-D two-layer ball, at 1, 2 and 4 ranks. A standalone MPI
  program exiting non-zero on failure; the identity is
  partition-independent, so the thresholds are the serial measurements'
  grades (2.5e-5 / 3.5e-5 / 1.9e-5 at order 1 — the readout's
  projection/quadrature gap).
*/

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>

#include "AdGIA.hpp"
#include "MixedProblemTestCommon.hpp"
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

constexpr double kRhoW = 0.02;
constexpr double kOmega = 0.1;

double MeltCap(const Vector& x, const Vector& n) {
  constexpr double s2 = 0.08;
  Vector d(x);
  d /= x.Norml2();
  d -= n;
  return 0.02 * std::exp(-(d * d) / s2);
}

Vector UnitDir(double a, double b, double c) {
  Vector n(3);
  n[0] = a;
  n[1] = b;
  n[2] = c;
  n /= n.Norml2();
  return n;
}

double KappaLateral(const Vector& x) {
  return self_grav_test::kKappa *
         (1.0 + 0.3 * (x[0] * x[0] - x[1] * x[1]) + 0.2 * x[0] * x[2]);
}
double MuLateral(const Vector& x) {
  return self_grav_test::kMu *
         (1.0 - 0.25 * (x[1] * x[1] - x[2] * x[2]) + 0.2 * x[0] * x[1]);
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();

  const int order = 1;
  auto serial_mesh = Mesh(self_grav_test::MeshFile(3).c_str(), 1, 1);
  auto parent = ParMesh(MPI_COMM_WORLD, serial_mesh);
  serial_mesh.Clear();
  auto body = ParSubMesh::CreateFromDomain(
      parent, self_grav_test::BodyMarker(parent));
  auto fec = H1_FECollection(order, 3);
  auto fes_u = ParFiniteElementSpace(&body, &fec, 3);
  auto fes_phi = ParFiniteElementSpace(&parent, &fec);
  auto rho = ConstantCoefficient(self_grav_test::kRho);
  auto surface = self_grav_test::SurfaceMarker(body);
  Vector moments(3);
  moments[0] = 1.2;
  moments[1] = 1.3;
  moments[2] = 2.0;
  const double g0 =
      4.0 * M_PI * self_grav_test::kG * self_grav_test::kRho / 3.0;
  auto w = ConstantCoefficient(kRhoW / g0);
  auto grad_phi0 =
      VectorFunctionCoefficient(3, [g0](const Vector& x, Vector& v) {
        v = x;
        v *= g0;
      });

  auto n1 = UnitDir(0.3, 0.5, 0.81);
  auto n2 = UnitDir(-0.7, 0.25, 0.55);
  auto z1 =
      FunctionCoefficient([n1](const Vector& x) { return MeltCap(x, n1); });
  auto z2 =
      FunctionCoefficient([n2](const Vector& x) { return MeltCap(x, n2); });

  auto defect = [&](const Rheology& rheology, bool rotation) {
    SeaLevelOperator sea(fes_u, surface);
    auto fingerprint = [&](Coefficient& zeta) {
      LinearQuasiStaticMixedSelfGravitatingProblem p(
          &fes_u, &fes_phi, rheology, rho, self_grav_test::kG,
          self_grav_test::kDtNDegree);
      p.SetRelTol(1e-12);
      p.SetWaterLoad(w, zeta, surface);
      if (rotation) {
        p.SetRotation(kOmega, moments);
      }
      p.AssembleForce(0.0);
      if (!p.Solve()) {
        Check(1.0, 0.0, "fingerprint solve");
      }
      Coefficient* psi =
          rotation ? &p.SolutionCentrifugalPotential() : nullptr;
      sea.SeaLevelChangeFrom(p.Displacement(), grad_phi0, p.Potential(),
                             p.UniformPotentialTerm(), psi);
      GridFunction sl(&sea.SurfaceSpace());
      sl = sea.SeaLevelChangeField();
      return sl;
    };
    auto sl1 = fingerprint(z1);
    auto sl2 = fingerprint(z2);
    auto pair = [&](GridFunction& sl, Coefficient& z) {
      GridFunctionCoefficient slc(&sl);
      ProductCoefficient pc(slc, z);
      return sea.SurfaceIntegral(pc);  // global in parallel
    };
    const double p12 = pair(sl1, z2);
    const double p21 = pair(sl2, z1);
    return std::abs(p12 - p21) / std::max(std::abs(p12), std::abs(p21));
  };

  {
    auto kappa = ConstantCoefficient(self_grav_test::kKappa);
    auto mu = ConstantCoefficient(self_grav_test::kMu);
    auto rheology = IsotropicElasticRheology(3, kappa, mu);
    Check(defect(rheology, false), 1e-4, "spherical reciprocity");
  }
  {
    auto kappa = FunctionCoefficient(KappaLateral);
    auto mu = FunctionCoefficient(MuLateral);
    auto rheology = IsotropicElasticRheology(3, kappa, mu);
    Check(defect(rheology, false), 1e-4, "3-D structure reciprocity");
    Check(defect(rheology, true), 1e-4, "rotation + structure reciprocity");
  }

  if (Mpi::Root()) {
    std::cout << (num_fails == 0 ? "OK" : "FAILED") << ": " << num_checks
              << " checks, " << num_fails << " failures\n";
  }
  return num_fails == 0 ? 0 : 1;
}
