/*
  Parallel tests of the profile border (SetProfileBorder), the dual form
  of the closure of an eliminated fluid region. Run with 1, 2 and 4
  ranks; a standalone MPI program that exits with status 1 if any check
  fails.

  Mirrors the serial gates of TestProfileBorder on the two-layer ball
  with a fluid core of density rho_c - a r^2 and bulk modulus
  (1 + delta) times the neutral value: agreement with the level border
  to discretisation error, the elimination against the monolithic route
  at solver grade, with and without rotation, and the conservation of
  the region's mass evaluated from the solution. The border's blocks
  are global sums over the ranks and its columns live on the true dofs,
  which is what these runs exercise.
*/

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <numbers>
#include <string>

#include "MixedProblemTestCommon.hpp"
#include "mfem.hpp"
#include "AdGIA.hpp"

using namespace mfem;
using namespace AdGIA;
using namespace self_grav_test;

namespace {

int num_checks = 0;
int num_failures = 0;

void Check(double value, double bound, const std::string& label) {
  num_checks++;
  const bool ok = std::abs(value) <= bound;
  if (!ok) {
    num_failures++;
  }
  if (Mpi::Root()) {
    std::cout << (ok ? "  ok  " : "  FAIL") << "  " << label << ": |"
              << value << "| <= " << bound << "\n";
  }
}

constexpr double kRhoC = 1.2;
constexpr double kDrop = 0.1;  // rho_c - rho at the wall

double CoreA() { return kDrop / (kMeshRCmb * kMeshRCmb); }

double CoreDensity(const Vector& x) { return kRhoC - CoreA() * (x * x); }

// d rho / d Phi0 = (d rho / d r) / g, g = 4 pi G (rho_c r / 3 - a r^3 / 5).
double CoreDensityGradient(const Vector& x) {
  const double a = CoreA();
  return -2.0 * a /
         (4.0 * std::numbers::pi * kG * (kRhoC / 3.0 - a * (x * x) / 5.0));
}

double GlobalNorm(const ParGridFunction& u) {
  Vector t;
  u.GetTrueDofs(t);
  return std::sqrt(InnerProduct(MPI_COMM_WORLD, t, t));
}

double RelDiff(const GridFunction& a, const GridFunction& b) {
  ParGridFunction d(static_cast<const ParGridFunction&>(a));
  d -= b;
  return GlobalNorm(d) /
         (GlobalNorm(static_cast<const ParGridFunction&>(b)) + 1e-300);
}

double RelDiff(const Vector& a, const Vector& b) {
  Vector d(a);
  d -= b;
  return d.Norml2() / b.Norml2();
}

void RunCase(double delta, const std::string& label) {
  Mesh smesh(TwoLayerMeshFile(3).c_str(), 1, 1);
  ParMesh pmesh(MPI_COMM_WORLD, smesh);
  Array<int> solid_attr({2});
  ParSubMesh body(ParSubMesh::CreateFromDomain(pmesh, solid_attr));
  H1_FECollection fec(1, 3);
  ParFiniteElementSpace fes_u(&body, &fec, 3), fes_phi(&pmesh, &fec);
  ConstantCoefficient kappa(kKappa), mu(kMu), rho(kRho);
  IsotropicElasticRheology rheology(3, kappa, mu);
  FunctionCoefficient rho_f(CoreDensity), rho_prime(CoreDensityGradient);
  FunctionCoefficient kappa_f([delta](const Vector& x) {
    const double r = CoreDensity(x);
    return -(1.0 + delta) * r * r / CoreDensityGradient(x);
  });
  FluidRegion core;
  core.attributes = Array<int>({1});
  core.density = &rho_f;
  core.density_gradient = &rho_prime;
  core.interface_marker.SetSize(body.bdr_attributes.Max());
  core.interface_marker = 0;
  core.interface_marker[0] = 1;  // the wall of the core
  core.bulk_modulus = &kappa_f;
  std::vector<FluidRegion> fluids{core};
  ConstantCoefficient sigma(0.02);
  auto surface = SurfaceMarker(body);
  Vector moments(3);
  moments[0] = 1.5262;
  moments[1] = 1.5262;
  moments[2] = 1.5312;

  // border: 1 level, 2 profile.
  auto run = [&](int border, bool monolithic, bool rotating,
                 const std::string& name) {
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        &fes_u, &fes_phi, rheology, rho, kG, kDtNDegree, nullptr, fluids);
    p->SetSolverType(
        LinearQuasiStaticMixedSelfGravitatingProblem::SolverType::BlockMINRES);
    p->SetRelTol(1e-12);
    if (border == 1) {
      p->SetLevelBorder(16);
    } else {
      p->SetProfileBorder();
    }
    if (rotating) {
      p->SetRotation(0.1, moments);
    }
    if (monolithic) {
      p->SetMonolithicBorder();
    }
    p->SetSurfaceLoad(sigma, surface);
    p->AssembleForce(0.0);
    Check(p->Solve() ? 0.0 : 1.0, 0.0, label + ": " + name + " solve");
    return p;
  };

  // The two forms of the closure, and the two routes of the dual form.
  auto level = run(1, false, false, "level border");
  auto pe = run(2, false, false, "profile border");
  auto pm = run(2, true, false, "profile border, monolithic");
  const auto& B = pe->ProfileBorderData(0);
  Check(B.size - (delta == 0.0 ? 1 : 7), 0.0, label + ": unknowns");
  Check(RelDiff(pe->Displacement(), level->Displacement()), 3e-3,
        label + ": profile vs level displacement");
  Check(RelDiff(pe->Potential(), level->Potential()), 2e-4,
        label + ": profile vs level potential");
  Check(RelDiff(pm->Displacement(), pe->Displacement()), 1e-7,
        label + ": monolithic vs elimination displacement");
  Check(RelDiff(pm->Potential(), pe->Potential()), 1e-7,
        label + ": monolithic vs elimination potential");
  Check(RelDiff(pm->ProfileBorderData(0).y, B.y), 1e-7,
        label + ": monolithic vs elimination unknowns");

  // The mass of the region: int (rho' phi - d(rho gamma)/dc) dV over the
  // fluid against the mass the wall sweeps out, int rho (m . u) dS.
  {
    ParLinearForm wall(&fes_u);
    wall.AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(rho_f),
                               core.interface_marker);
    wall.Assemble();
    const double swept =
        wall(static_cast<const ParGridFunction&>(pe->Displacement()));
    const GridFunction& phi = pe->Potential();
    const GridFunction& phi0 = pe->BackgroundPotential();
    const int qorder = 2 * fes_phi.GetMaxElementOrder() + 2;
    double dm = 0.0;
    Vector b, db;
    for (int e = 0; e < pmesh.GetNE(); e++) {
      if (pmesh.GetAttribute(e) != 1) {
        continue;
      }
      ElementTransformation* T = pmesh.GetElementTransformation(e);
      const IntegrationRule& ir = IntRules.Get(T->GetGeometryType(), qorder);
      for (int q = 0; q < ir.GetNPoints(); q++) {
        const IntegrationPoint& ip = ir.IntPoint(q);
        T->SetIntPoint(&ip);
        B.Basis(phi0.GetValue(*T, ip), b, db);
        const double gamma = b * B.y, dgamma = db * B.y;
        const double rp = rho_prime.Eval(*T, ip);
        dm += ip.weight * T->Weight() *
              (rp * phi.GetValue(*T, ip) - rp * gamma -
               rho_f.Eval(*T, ip) * dgamma);
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, &dm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    Check((dm - swept) / swept, 1e-8, label + ": mass of the region");
  }

  // With rotation.
  auto re = run(2, false, true, "rotating profile border");
  auto rm = run(2, true, true, "rotating profile border, monolithic");
  auto rl = run(1, false, true, "rotating level border");
  Check(RelDiff(rm->Displacement(), re->Displacement()), 1e-7,
        label + ": rotating, monolithic vs elimination displacement");
  Check(RelDiff(rm->AngularVelocity(), re->AngularVelocity()), 1e-7,
        label + ": rotating, monolithic vs elimination angular velocity");
  Check(RelDiff(rl->Displacement(), re->Displacement()), 3e-3,
        label + ": rotating, level vs profile displacement");
  Check(RelDiff(rl->AngularVelocity(), re->AngularVelocity()), 3e-3,
        label + ": rotating, level vs profile angular velocity");
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();
  RunCase(0.0, "neutral core");
  RunCase(0.5, "stratified core");
  if (Mpi::Root()) {
    std::cout << num_checks << " checks, " << num_failures << " failures\n";
  }
  return num_failures == 0 ? 0 : 1;
}
