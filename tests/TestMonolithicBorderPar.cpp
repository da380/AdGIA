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

  auto run = [&](bool monolithic, bool water, bool rotate) {
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        &fes_u, &fes_phi, rheology, rho, kG, kDtNDegree);
    p->SetSolverType(
        LinearQuasiStaticMixedSelfGravitatingProblem::SolverType::
            BlockMINRES);
    p->SetRelTol(1e-12);
    if (monolithic) {
      p->SetMonolithicBorder();
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
    Check(p->Solve() ? 0.0 : 1.0, 0.0,
          label + (monolithic ? ": monolithic" : ": elimination") +
              " solve");
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
    auto pe = run(false, cfg.water, cfg.rotate);
    auto pm = run(true, cfg.water, cfg.rotate);
    Check(RelDiff(pm->Displacement(), pe->Displacement()), 1e-8,
          label + ": " + cfg.name + " displacement");
    Check(RelDiff(pm->Potential(), pe->Potential()), 1e-8,
          label + ": " + cfg.name + " potential");
    if (cfg.water) {
      Check(pm->UniformPotentialTerm() - pe->UniformPotentialTerm(),
            1e-8 * std::abs(pe->UniformPotentialTerm()),
            label + ": " + cfg.name + " Phi_g");
    }
    if (cfg.rotate) {
      Vector dw(pm->AngularVelocity());
      dw -= pe->AngularVelocity();
      Check(dw.Norml2() / (pe->AngularVelocity().Norml2() + 1e-300), 1e-7,
            label + ": " + cfg.name + " omega");
    }
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();
  RunCase(1, "order 1");
  if (Mpi::Root()) {
    std::cout << num_checks << " checks, " << num_failures
              << " failures\n";
  }
  return num_failures == 0 ? 0 : 1;
}
