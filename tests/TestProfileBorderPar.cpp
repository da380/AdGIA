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

  And the layers declared (ProfileBorderOptions::breaks) on a core that is
  stratified below a knot and neutral above it: the level of the knot
  from ValueAtPoint, the layers, the two routes and the mass, with the
  level and the displacement against the serial problem's.
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

// A core stratified (excess 0.5) below a knot and neutral above it, its
// stratification declared and the knot's level given to the border.
// Every rank also solves the serial problem.
void RunDeclaredLayers(const std::string& label) {
  const double delta = 0.5, knot = 0.6 * kMeshRCmb;
  auto excess_at = [=](const Vector& x) {
    return x.Norml2() < knot ? delta : 0.0;
  };
  Mesh smesh(TwoLayerMeshFile(3).c_str(), 1, 1);
  Array<int> solid_attr({2});
  H1_FECollection fec(1, 3);
  ConstantCoefficient kappa(kKappa), mu(kMu), rho(kRho);
  IsotropicElasticRheology rheology(3, kappa, mu);
  FunctionCoefficient rho_f(CoreDensity), rho_prime(CoreDensityGradient);
  FunctionCoefficient kappa_f([&](const Vector& x) {
    const double r = CoreDensity(x);
    return -(1.0 + excess_at(x)) * r * r / CoreDensityGradient(x);
  });
  FunctionCoefficient excess(excess_at);
  ConstantCoefficient sigma(0.02);
  Vector at_knot(3);
  at_knot = 0.0;
  at_knot[2] = knot;

  // One problem on a mesh and its SubMesh, serial or parallel by the
  // spaces; with no break given, the survey's layers.
  auto solve = [&](FiniteElementSpace& fes_u, FiniteElementSpace& fes_phi,
                   Mesh& body, const std::vector<double>& breaks,
                   bool monolithic, const std::string& name) {
    FluidRegion core;
    core.attributes = Array<int>({1});
    core.density = &rho_f;
    core.density_gradient = &rho_prime;
    core.interface_marker.SetSize(body.bdr_attributes.Max());
    core.interface_marker = 0;
    core.interface_marker[0] = 1;
    core.bulk_modulus = &kappa_f;
    core.stratification = &excess;
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        &fes_u, &fes_phi, rheology, rho, kG, kDtNDegree, nullptr,
        std::vector<FluidRegion>{core});
    p->SetSolverType(
        LinearQuasiStaticMixedSelfGravitatingProblem::SolverType::BlockMINRES);
    p->SetRelTol(1e-12);
    ProfileBorderOptions options;
    options.breaks = breaks;
    p->SetProfileBorder(options);
    if (monolithic) {
      p->SetMonolithicBorder();
    }
    auto surface = SurfaceMarker(body);
    p->SetSurfaceLoad(sigma, surface);
    p->AssembleForce(0.0);
    Check(p->Solve() ? 0.0 : 1.0, 0.0, label + ": " + name + " solve");
    return p;
  };

  // Serial reference.
  double level_serial = 0.0, u_serial = 0.0;
  {
    SubMesh body(SubMesh::CreateFromDomain(smesh, solid_attr));
    FiniteElementSpace fes_u(&body, &fec, 3), fes_phi(&smesh, &fec);
    auto surveyed = solve(fes_u, fes_phi, body, {}, false, "serial survey");
    level_serial = ValueAtPoint(surveyed->BackgroundPotential(), at_knot);
    auto declared =
        solve(fes_u, fes_phi, body, {level_serial}, false, "serial declared");
    u_serial = declared->Displacement().Norml2();
  }

  ParMesh pmesh(MPI_COMM_WORLD, smesh);
  ParSubMesh body(ParSubMesh::CreateFromDomain(pmesh, solid_attr));
  ParFiniteElementSpace fes_u(&body, &fec, 3), fes_phi(&pmesh, &fec);
  auto surveyed = solve(fes_u, fes_phi, body, {}, false, "survey");
  const double level = ValueAtPoint(surveyed->BackgroundPotential(), at_knot);
  Check((level - level_serial) / level_serial, 1e-9,
        label + ": the knot's level against serial");
  auto pe = solve(fes_u, fes_phi, body, {level}, false, "declared");
  auto pm = solve(fes_u, fes_phi, body, {level}, true, "declared, monolithic");
  const auto& B = pe->ProfileBorderData(0);
  Check(B.declared ? 0.0 : 1.0, 0.0, label + ": layers declared");
  Check(static_cast<double>(B.layers.size()) - 2.0, 0.0, label + ": two layers");
  Check(B.size - 7, 0.0, label + ": unknowns");
  if (B.layers.size() == 2) {
    Check(B.layers[0].c1 - level, 0.0, label + ": the boundary at the level");
    Check(B.layers[0].neutral ? 1.0 : 0.0, 0.0, label + ": stratified below");
    Check(B.layers[1].neutral ? 0.0 : 1.0, 0.0, label + ": neutral above");
  }
  Check(surveyed->ProfileBorderData(0).declared ? 1.0 : 0.0, 0.0,
        label + ": the survey's layers without a break");
  Check((GlobalNorm(static_cast<const ParGridFunction&>(pe->Displacement())) -
         u_serial) / u_serial,
        1e-7, label + ": displacement against serial");
  Check(RelDiff(pm->Displacement(), pe->Displacement()), 1e-7,
        label + ": monolithic vs elimination displacement");
  Check(RelDiff(pm->ProfileBorderData(0).y, B.y), 1e-7,
        label + ": monolithic vs elimination unknowns");

  // The mass of the region, as in RunCase.
  {
    FluidRegion core;
    core.interface_marker.SetSize(body.bdr_attributes.Max());
    core.interface_marker = 0;
    core.interface_marker[0] = 1;
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
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();
  RunCase(0.0, "neutral core");
  RunCase(0.5, "stratified core");
  RunDeclaredLayers("declared layers");
  if (Mpi::Root()) {
    std::cout << num_checks << " checks, " << num_failures << " failures\n";
  }
  return num_failures == 0 ? 0 : 1;
}
