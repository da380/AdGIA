/*
  Parallel tests of the monolithic bordered solve (SetMonolithicBorder)
  against the block-elimination reference. Run with 1, 2 and 4 ranks; a
  standalone MPI program that exits with status 1 if any check fails.

  Mirrors the serial gates on the 3-D two-layer mesh (the monolithic
  path is 3-D only — the 2-D compatibility relocation is asymmetric by
  design): water,
  rotation-only and the composed border, elimination vs monolithic at
  solver grade — the rank-0-owned border scalars and the collective
  operator/preconditioner applies are exactly what these runs exercise.
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

constexpr double kRhoW = 0.05;
constexpr double kRhoI = 0.045;
constexpr double kDepth = 1.0;
constexpr double kContAmp = 3.0;
constexpr double kContWidth = 0.7;
constexpr double kCapAmp = 4.0;
constexpr double kCapWidth = 0.35;
constexpr double kShore = 5e-3;
constexpr double kMelt = 0.5;

double Colat(const Vector& x) {
  const double r = x.Norml2();
  const double c = x[2] / r;
  return std::acos(std::min(1.0, std::max(-1.0, c)));
}

double SL0(const Vector& x) {
  const double t = Colat(x) / kContWidth;
  return kDepth - kContAmp * std::exp(-0.5 * std::pow(t, 6));
}

double Ice0(const Vector& x) {
  const double t = Colat(x) / kCapWidth;
  return kCapAmp * std::exp(-0.5 * t * t);
}

double Fraction0(const Vector& x) {
  const double q = kRhoW * SL0(x) - kRhoI * Ice0(x);
  return 0.5 * (1.0 + std::tanh(q / kShore));
}

double MeltLoad(const Vector& x) {
  return -(1.0 - Fraction0(x)) * kRhoI * kMelt * Ice0(x);
}

double GlobalNorm(const ParGridFunction& f) {
  Vector tv(f.ParFESpace()->GetTrueVSize());
  f.GetTrueDofs(tv);
  double local = tv * tv, g = 0.0;
  MPI_Allreduce(&local, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return std::sqrt(g);
}

double RelDiff(const GridFunction& a, const GridFunction& b) {
  ParGridFunction d(static_cast<const ParGridFunction&>(a));
  d -= b;
  return GlobalNorm(d) /
         (GlobalNorm(static_cast<const ParGridFunction&>(b)) + 1e-300);
}

void RunCase(int order, const std::string& label) {
  Mesh smesh(MeshFile(3).c_str(), 1, 1);
  int nxyz[3] = {Mpi::WorldSize(), 1, 1};
  int* partitioning = smesh.CartesianPartitioning(nxyz);
  ParMesh pmesh(MPI_COMM_WORLD, smesh, partitioning);
  delete[] partitioning;
  ParSubMesh body(ParSubMesh::CreateFromDomain(pmesh, BodyMarker(pmesh)));
  H1_FECollection fec(order, 3);
  ParFiniteElementSpace fes_u(&body, &fec, 3), fes_phi(&pmesh, &fec);
  ConstantCoefficient kappa(kKappa), mu(kMu), rho(kRho);
  IsotropicElasticRheology rheology(3, kappa, mu);
  const Array<int> surface = SurfaceMarker(body);
  const double g0 = 4.0 * std::numbers::pi * kG * kRho / 3.0;
  FunctionCoefficient w([g0](const Vector& x) {
    return kRhoW * Fraction0(x) / (g0 * x.Norml2());
  });
  FunctionCoefficient sigma_d(MeltLoad);

  // route: 0 elimination (the reference), 1 monolithic, 2 Picard border.
  auto run = [&](int route, bool water, bool rotate) {
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        &fes_u, &fes_phi, rheology, rho, kG, kDtNDegree);
    p->SetSolverType(
        LinearQuasiStaticMixedSelfGravitatingProblem::SolverType::
            BlockMINRES);
    p->SetRelTol(1e-12);
    if (route == 1) {
      p->SetMonolithicBorder();
    } else if (route == 2) {
      p->SetPicardBorder();
    }
    if (water) {
      p->SetWaterLoad(w, sigma_d, surface);
    } else {
      p->SetSurfaceLoad(sigma_d, surface);
    }
    if (rotate) {
      Vector moments(3);
      moments[0] = 1.5262;
      moments[1] = 1.5262;
      moments[2] = 1.5312;
      p->SetRotation(0.1, moments);
    }
    p->AssembleForce(0.0);
    const char* names[] = {": elimination", ": monolithic", ": picard"};
    Check(p->Solve() ? 0.0 : 1.0, 0.0, label + names[route] + " solve");
    return p;
  };

  // Water, rotation-only, and the composed border.
  struct Config {
    bool water, rotate;
    const char* name;
  };
  for (const Config cfg : {Config{true, false, "water"},
                           Config{false, true, "rotation"},
                           Config{true, true, "composed"}}) {
    auto pe = run(0, cfg.water, cfg.rotate);
    for (int route : {1, 2}) {
      auto pm = run(route, cfg.water, cfg.rotate);
      const std::string rname =
          std::string(cfg.name) + (route == 1 ? " monolithic" : " picard");
      Check(RelDiff(pm->Displacement(), pe->Displacement()), 1e-8,
            label + ": " + rname + " displacement");
      Check(RelDiff(pm->Potential(), pe->Potential()), 1e-8,
            label + ": " + rname + " potential");
      if (cfg.water) {
        Check(pm->UniformPotentialTerm() - pe->UniformPotentialTerm(),
              1e-8 * std::abs(pe->UniformPotentialTerm()),
              label + ": " + rname + " Phi_g");
      }
      if (cfg.rotate) {
        Vector dw(pm->AngularVelocity());
        dw -= pe->AngularVelocity();
        Check(dw.Norml2() / (pe->AngularVelocity().Norml2() + 1e-300),
              1e-7, label + ": " + rname + " omega");
      }
      if (route == 2) {
        // Genuinely converged (no elimination fallback), identically on
        // every rank — the pass loop is rank-collective.
        Check(pm->LastPicardPasses() > 0 ? 0.0 : 1.0, 0.0,
              label + ": " + rname + " converged without fallback");
        if (Mpi::Root()) {
          std::cout << "picard passes (" << cfg.name
                    << "): " << pm->LastPicardPasses() << "\n";
        }
      }
    }
  }
}

}  // namespace

// The level border through the monolithic route, parallel: the two
// routes solve the same bordered system (serial gate
// TestMonolithicBorder.LevelBorderMatchesElimination), so the parallel
// fields agree at solver grade across ranks.
void RunLevelBorderCase(const std::string& label) {
  Mesh smesh("../data/earth_fluidcore_3d.msh", 1, 1);
  ParMesh pmesh(MPI_COMM_WORLD, smesh);
  Array<int> solid_attr({2});
  ParSubMesh body(ParSubMesh::CreateFromDomain(pmesh, solid_attr));
  H1_FECollection fec(1, 3);
  ParFiniteElementSpace fes_u(&body, &fec, 3), fes_phi(&pmesh, &fec);
  ConstantCoefficient kappa(kKappa), mu(kMu), rho(kRho), kappa_f(kKappa);
  IsotropicElasticRheology rheology(3, kappa, mu);
  FunctionCoefficient rho_f([](const Vector& x) {
    const double r = x.Norml2();
    return 1.2 - 0.1 * r;
  });
  FluidRegion core;
  core.attributes = Array<int>({1});
  core.density = &rho_f;
  core.interface_marker.SetSize(body.bdr_attributes.Max());
  core.interface_marker = 0;
  core.interface_marker[0] = 1;  // the CMB
  core.bulk_modulus = &kappa_f;
  std::vector<FluidRegion> fluids{core};
  ConstantCoefficient sigma(0.02);
  Array<int> surface(body.bdr_attributes.Max());
  surface = 0;
  surface[body.bdr_attributes.Max() - 1] = 1;

  auto run = [&](bool monolithic) {
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        &fes_u, &fes_phi, rheology, rho, kG, kDtNDegree, nullptr, fluids);
    p->SetSolverType(
        LinearQuasiStaticMixedSelfGravitatingProblem::SolverType::
            BlockMINRES);
    p->SetRelTol(1e-12);
    p->SetLevelBorder(12);
    if (monolithic) {
      p->SetMonolithicBorder();
    }
    p->SetSurfaceLoad(sigma, surface);
    p->AssembleForce(0.0);
    Check(p->Solve() ? 0.0 : 1.0, 0.0,
          label + (monolithic ? ": monolithic solve" : ": elimination solve"));
    return p;
  };
  auto pe = run(false);
  auto pm = run(true);
  Check(RelDiff(pm->Displacement(), pe->Displacement()), 1e-7,
        label + ": displacement");
  Check(RelDiff(pm->Potential(), pe->Potential()), 1e-7,
        label + ": potential");
}

// The level border's own-row sources (AddLevelBorderForce), parallel, on
// the same two-layer ball without a surface load: (1) reciprocity of the
// symmetric bordered system, <x(f), g> = <x(g), f>, pairing a nu-row
// source and the wall row with a potential load and a U-row source with
// a CMB traction (serial gate TestLevelBorderSources.Reciprocity) --- the
// border data are global, the field pairings reduced over the ranks; (2)
// a source on every row kind through the elimination and the monolithic
// routes, which must agree at solver grade. Measured on 1, 2 and 4
// ranks at solver tolerance 1e-12: reciprocity asymmetries 3e-14 to
// 1.8e-12 (gate 1e-10); route differences 2.5e-11 in the displacement,
// 3e-12 in the potential and the border unknowns (gate 1e-7, as for the
// unloaded border above).
void RunLevelBorderSourcesCase(const std::string& label) {
  Mesh smesh("../data/earth_fluidcore_3d.msh", 1, 1);
  ParMesh pmesh(MPI_COMM_WORLD, smesh);
  Array<int> solid_attr({2});
  ParSubMesh body(ParSubMesh::CreateFromDomain(pmesh, solid_attr));
  H1_FECollection fec(1, 3);
  ParFiniteElementSpace fes_u(&body, &fec, 3), fes_phi(&pmesh, &fec);
  ConstantCoefficient kappa(kKappa), mu(kMu), rho(kRho), kappa_f(kKappa);
  IsotropicElasticRheology rheology(3, kappa, mu);
  FunctionCoefficient rho_f([](const Vector& x) {
    const double r = x.Norml2();
    return 1.2 - 0.1 * r;
  });
  FluidRegion core;
  core.attributes = Array<int>({1});
  core.density = &rho_f;
  core.interface_marker.SetSize(body.bdr_attributes.Max());
  core.interface_marker = 0;
  core.interface_marker[0] = 1;  // the CMB
  core.bulk_modulus = &kappa_f;
  std::vector<FluidRegion> fluids{core};

  auto make = [&](bool monolithic) {
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        &fes_u, &fes_phi, rheology, rho, kG, kDtNDegree, nullptr, fluids);
    p->SetSolverType(
        LinearQuasiStaticMixedSelfGravitatingProblem::SolverType::
            BlockMINRES);
    p->SetRelTol(1e-12);
    p->SetLevelBorder(12);
    if (monolithic) {
      p->SetMonolithicBorder();
    }
    return p;
  };
  auto pe = make(false);
  Vector c;
  bool centre;
  std::vector<int> wall_node;
  pe->LevelBorderGrid(0, c, centre, wall_node);
  const int m = c.Size();
  // Nodal data of smooth functions of the level (the hat integrals are
  // not needed for the algebraic identities checked here).
  Vector r_nu(m), r_U(m), r_s(m), r_l(1);
  for (int k = 0; k < m; k++) {
    const double t = static_cast<double>(k) / (m - 1);
    r_nu[k] = 1e-3 * (1.0 + t - 0.5 * t * t);
    r_U[k] = 1e-3 * (0.3 + t * (1.0 - t));
    r_s[k] = 1e-3 * (1.0 - t * t);
  }
  r_l[0] = 1e-2;
  Array<int> fluid_marker(pmesh.attributes.Max()), cmb(body.bdr_attributes.Max());
  fluid_marker = 0;
  fluid_marker[0] = 1;
  cmb = 0;
  cmb[0] = 1;
  FunctionCoefficient q_phi([](const Vector& x) {
    return 1.0 + x[2] * x[2] + 0.3 * x[0];
  });
  ParLinearForm ell(&fes_phi);
  ell.AddDomainIntegrator(new DomainLFIntegrator(q_phi), fluid_marker);
  ell.Assemble();
  VectorFunctionCoefficient q_t(3, [](const Vector& x, Vector& t) {
    const double r = x.Norml2();
    t = x;
    t *= 0.1 * (1.0 + 0.5 * x[2] / r) / r;
    t[0] += 0.02;
  });
  ParLinearForm trac(&fes_u);
  trac.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(q_t), cmb);
  trac.Assemble();
  Vector ell_t(fes_phi.GetTrueVSize()), trac_t(fes_u.GetTrueVSize());
  ell.ParallelAssemble(ell_t);
  trac.ParallelAssemble(trac_t);
  const Vector none;

  struct Out {
    double phi_ell, u_trac;  // field pairings, global
    Vector U, s, nu, lambda;
  };
  auto solve = [&](LinearQuasiStaticMixedSelfGravitatingProblem& p,
                   const Vector& rU, const Vector& rs, const Vector& rnu,
                   const Vector& rl, bool with_ell, bool with_trac) {
    p.AssembleForce(0.0);
    p.AddLevelBorderForce(0, rU, rs, rnu, rl);
    if (with_ell) {
      p.AddPotentialForce(ell);
    }
    if (with_trac) {
      p.AddForce(trac);
    }
    Check(p.Solve() ? 0.0 : 1.0, 0.0, label + ": solve");
    Out o;
    Vector phi_t, u_t;
    p.Potential().GetTrueDofs(phi_t);
    p.Displacement().GetTrueDofs(u_t);
    o.phi_ell = InnerProduct(MPI_COMM_WORLD, phi_t, ell_t);
    o.u_trac = InnerProduct(MPI_COMM_WORLD, u_t, trac_t);
    Vector cc;
    p.LevelBorderSolution(0, cc, o.U, o.s, o.nu, o.lambda);
    return o;
  };
  auto rel = [](double a, double b) {
    return std::abs(a - b) / std::max(std::abs(a), std::abs(b));
  };
  const Out x_nu = solve(*pe, none, none, r_nu, none, false, false);
  const Out x_ell = solve(*pe, none, none, none, none, true, false);
  const Out x_U = solve(*pe, r_U, none, none, none, false, false);
  const Out x_t = solve(*pe, none, none, none, none, false, true);
  const Out x_l = solve(*pe, none, none, none, r_l, false, false);
  Check(rel(x_nu.phi_ell, r_nu * x_ell.nu), 1e-10,
        label + ": nu row / potential load reciprocity");
  Check(rel(x_l.phi_ell, r_l * x_ell.lambda), 1e-10,
        label + ": wall row / potential load reciprocity");
  Check(rel(x_U.u_trac, r_U * x_t.U), 1e-10,
        label + ": U row / CMB traction reciprocity");

  // Every row kind at once, through both routes.
  auto pm = make(true);
  const Out ye = solve(*pe, r_U, r_s, r_nu, r_l, false, false);
  const Out ym = solve(*pm, r_U, r_s, r_nu, r_l, false, false);
  Check(RelDiff(pm->Displacement(), pe->Displacement()), 1e-7,
        label + ": sources, monolithic vs elimination displacement");
  Check(RelDiff(pm->Potential(), pe->Potential()), 1e-7,
        label + ": sources, monolithic vs elimination potential");
  Vector dnu(ye.nu);
  dnu -= ym.nu;
  Check(dnu.Normlinf() / ye.nu.Normlinf(), 1e-7,
        label + ": sources, monolithic vs elimination nu");
  Check(rel(ye.lambda[0], ym.lambda[0]), 1e-7,
        label + ": sources, monolithic vs elimination wall multiplier");
}

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();
  RunCase(1, "order 1");
    RunLevelBorderCase("level border");
  RunLevelBorderSourcesCase("level border sources");

  if (Mpi::Root()) {
    std::cout << num_checks << " checks, " << num_failures
              << " failures\n";
  }
  return num_failures == 0 ? 0 : 1;
}
