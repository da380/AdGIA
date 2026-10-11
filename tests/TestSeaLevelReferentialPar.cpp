/*
  Parallel twin of TestSeaLevelReferential.cpp (WP7 phase A,
  doc/planning/referential_sea_level_plan.md): the referential water
  load on the 3-D two-layer mesh, all-ocean, phi_e = id. Run with 1, 2
  and 4 ranks; a standalone MPI program that exits with status 1 if any
  check fails.

  Every rank also solves the serial problem on the full mesh: serial
  and parallel build the same discrete system, so the global L2 norms
  of displacement and potential and the scalar Phi_g are compared
  directly. The parallel run then re-checks the mass-conservation
  certificate and the inert uniform term on its own integrals.
*/

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <numbers>
#include <string>

#include "mfem.hpp"
#include "AdGIA.hpp"

using namespace mfem;
using namespace AdGIA;

namespace {

constexpr double kG = 0.05;
constexpr double kRho = 1.0;
constexpr double kKappa = 1.0;
constexpr double kMu = 0.5;
constexpr int kDtNDegree = 12;
constexpr double kRhoW = 0.02;

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

double RelErr(double a, double b) {
  return std::abs(a - b) / std::max(1e-30, std::abs(b));
}

double G0() { return 4.0 * std::numbers::pi * kG * kRho / 3.0; }

double Degree2(const Vector& x) {
  const double r = x.Norml2();
  const double c = x[2] / r;
  return 3.0 * c * c - 1.0;
}

// Degree-2 pattern plus a mean, so that Phi_g is active, plus an
// order-1 tesseral part so that the wander rows are driven.
double SigmaWithMean(const Vector& x) {
  const double r2 = x * x;
  return Degree2(x) + 2.0 * x[0] * x[2] / r2 + 0.5;
}

double UniformBallPressure(const Vector& x) {
  const double r2 = x * x;
  return std::max(0.0, 2.0 * std::numbers::pi * kG * kRho * kRho *
                           (1.0 - r2) / 3.0);
}

CallableDiffeomorphism IdentityMap(int dim) {
  return CallableDiffeomorphism(
      dim, [](const Vector& x, Vector& y) { y = x; },
      [](const Vector&, DenseMatrix& F) {
        F = 0.0;
        for (int i = 0; i < F.Height(); i++) {
          F(i, i) = 1.0;
        }
      });
}

double L2Norm(const GridFunction& u) {
  const int vdim = u.FESpace()->GetVDim();
  if (vdim == 1) {
    ConstantCoefficient z(0.0);
    return const_cast<GridFunction&>(u).ComputeL2Error(z);
  }
  Vector zero(vdim);
  zero = 0.0;
  VectorConstantCoefficient z(zero);
  return const_cast<GridFunction&>(u).ComputeL2Error(z);
}

double GlobalSum(double v) {
  double g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return g;
}

// rotation: 0 = water only; 1 = the traditional border composes with
// water (moments away from near-neutral wander, the WP4 lesson); 2 =
// the GENERALISED door (SetRotation(Omega): model inertia + Tisserand
// gauge) on a triaxial density — the parallel leg of WP7 phase B, which
// exercises the ModelInertiaTensor reduction, the gauge block and the
// gauge-preserving solve tail.
void RunCase(int order, const std::string& label, int rotation = 0) {
  const char* mesh_file = "../data/coupled_poisson.msh";
  Mesh smesh(mesh_file, 1, 1);
  const int dim = smesh.Dimension();
  Array<int> body_attr({1}), buffer_attr({2});

  Vector bb_min, bb_max;
  smesh.GetBoundingBox(bb_min, bb_max);
  const double r_out = bb_max.Normlinf();

  auto phi = IdentityMap(dim);
  ConstantCoefficient kappa(kKappa), mu(kMu), rho(kRho);
  FunctionCoefficient p0(UniformBallPressure);
  auto C_eff =
      IsotropicElasticTensorCoefficient::FromBulkModulus(dim, kappa, mu);
  BareElasticTensorCoefficient C(dim, C_eff, p0);
  MatrixFunctionCoefficient S(dim, [](const Vector& x, DenseMatrix& Se) {
    Se.SetSize(x.Size());
    Se = 0.0;
    const double p = UniformBallPressure(x);
    for (int i = 0; i < x.Size(); i++) {
      Se(i, i) = -p;
    }
  });
  ReferentialElasticRheology rheology(dim, C, S, phi);
  FunctionCoefficient sigma(SigmaWithMean);
  ConstantCoefficient w(kRhoW / G0());

  // The generalised door wants genuinely distinct model moments:
  // polynomial triaxiality, as in the serial G8d certificate.
  FunctionCoefficient rho_tri([](const Vector& x) {
    const double r2 = x * x;
    return kRho * (1.0 + 0.1 * (3.0 * x[2] * x[2] - r2) +
                   0.05 * (x[0] * x[0] - x[1] * x[1]));
  });
  Coefficient& rho_used =
      rotation == 2 ? static_cast<Coefficient&>(rho_tri) : rho;

  Vector moments(3);
  moments[0] = 1.2;
  moments[1] = 1.3;
  moments[2] = 2.0;
  const double Omega = 0.1;

  // Serial reference on every rank.
  double u_ref = 0.0, z_ref = 0.0, g_ref = 0.0;
  Vector w_ref;
  {
    SubMesh body(SubMesh::CreateFromDomain(smesh, body_attr));
    SubMesh buffer(SubMesh::CreateFromDomain(smesh, buffer_attr));
    H1_FECollection fec(order, dim);
    FiniteElementSpace fes_u(&body, &fec, dim), fes_zeta(&smesh, &fec);
    FiniteElementSpace fes_buffer(&buffer, &fec, dim);
    LinearQuasiStaticReferentialSelfGravitatingProblem problem(
        &fes_u, &fes_zeta, rheology, rho_used, kG, kDtNDegree);
    auto E = NewRadialVacuumExtension(fes_u, fes_buffer, 1.0, r_out);
    problem.SetPrescribedVacuumExtension(fes_buffer, *E);
    Array<int> surface(body.bdr_attributes.Max());
    surface = 0;
    surface[body.bdr_attributes.Max() - 1] = 1;
    problem.SetWaterLoad(w, sigma, surface);
    if (rotation == 1) {
      problem.SetRotation(Omega, moments);
    } else if (rotation == 2) {
      problem.SetRotation(Omega);
    }
    problem.SetRelTol(1e-12);
    problem.AssembleForce(0.0);
    Check(problem.Solve() ? 0.0 : 1.0, 0.0, label + " serial solve");
    u_ref = L2Norm(problem.Displacement());
    z_ref = L2Norm(problem.Potential());
    g_ref = problem.UniformPotentialTerm();
    if (rotation) {
      w_ref = problem.AngularVelocity();
    }
  }

  // Parallel problem (slab partition, so some ranks can be surface-free
  // at higher rank counts).
  int nxyz[3] = {Mpi::WorldSize(), 1, 1};
  int* partitioning = smesh.CartesianPartitioning(nxyz);
  ParMesh pmesh(MPI_COMM_WORLD, smesh, partitioning);
  delete[] partitioning;
  ParSubMesh body(ParSubMesh::CreateFromDomain(pmesh, body_attr));
  ParSubMesh buffer(ParSubMesh::CreateFromDomain(pmesh, buffer_attr));
  H1_FECollection fec(order, dim);
  ParFiniteElementSpace fes_u(&body, &fec, dim), fes_zeta(&pmesh, &fec);
  ParFiniteElementSpace fes_buffer(&buffer, &fec, dim);
  LinearQuasiStaticReferentialSelfGravitatingProblem problem(
      &fes_u, &fes_zeta, rheology, rho_used, kG, kDtNDegree);
  auto E = NewRadialVacuumExtension(fes_u, fes_buffer, 1.0, r_out);
  problem.SetPrescribedVacuumExtension(fes_buffer, *E);
  Array<int> surface(body.bdr_attributes.Max());
  surface = 0;
  surface[body.bdr_attributes.Max() - 1] = 1;
  FunctionCoefficient sigma2(SigmaWithMean);
  problem.SetWaterLoad(w, sigma2, surface);
  if (rotation == 1) {
    problem.SetRotation(Omega, moments);
  } else if (rotation == 2) {
    problem.SetRotation(Omega);
  }
  problem.SetRelTol(1e-12);
  problem.AssembleForce(0.0);
  Check(problem.Solve() ? 0.0 : 1.0, 0.0, label + " parallel solve");
  Check(RelErr(L2Norm(problem.Displacement()), u_ref), 1e-5,
        label + " displacement norm");
  Check(RelErr(L2Norm(problem.Potential()), z_ref), 1e-5,
        label + " potential norm");
  Check(RelErr(problem.UniformPotentialTerm(), g_ref), 1e-5,
        label + " Phi_g");
  if (rotation) {
    Vector dw(problem.AngularVelocity());
    dw -= w_ref;
    Check(dw.Norml2() / w_ref.Norml2(), 1e-5, label + " omega");
  }

  // The parallel run's own mass certificate (route grade, as in the
  // serial suite).
  {
    auto surface_integral = [&](Coefficient& f) {
      H1_FECollection sfec(order, dim);
      ParFiniteElementSpace sfes(&body, &sfec);
      ParLinearForm lf(&sfes);
      lf.AddBoundaryIntegrator(new BoundaryLFIntegrator(f), surface);
      lf.Assemble();
      return GlobalSum(lf.Sum());
    };
    GridFunctionCoefficient zc(
        const_cast<GridFunction*>(&problem.PotentialOnBody()));
    ProductCoefficient wz(w, zc);
    ConstantCoefficient one(1.0);
    ProductCoefficient wone(w, one);
    double total = surface_integral(sigma2) - surface_integral(wz) +
                   problem.UniformPotentialTerm() * surface_integral(wone);
    if (rotation) {
      // The composed load includes -w psi(omega).
      ProductCoefficient wpsi(w, problem.SolutionCentrifugalPotential());
      total -= surface_integral(wpsi);
    }
    // Route grade tightens with order (the G8d lesson: the assembled
    // data integrate sigma against the interpolated psi/zeta with the
    // forms' rules): measured 2.4e-4 at o1, passing 5e-5 at o2, on the
    // generalised case.
    Check(std::abs(total), rotation == 2 && order == 1 ? 5e-4 : 5e-5,
          label + " mass certificate");
  }

  // The generalised door's own certificate: the Tisserand condition
  // int rho x cross u dV = 0 at the parallel solution, re-integrated by
  // quadrature (rank-local, reduced), against the rigid scale tr I0.
  if (rotation == 2) {
    Vector h(3);
    h = 0.0;
    VectorGridFunctionCoefficient uc(
        const_cast<GridFunction*>(&problem.Displacement()));
    Vector x(3), uv(3);
    for (int e = 0; e < body.GetNE(); e++) {
      auto* T = body.GetElementTransformation(e);
      const auto& ir =
          IntRules.Get(body.GetElementGeometry(e), 2 * order + 4);
      for (int q = 0; q < ir.GetNPoints(); q++) {
        const auto& ip = ir.IntPoint(q);
        T->SetIntPoint(&ip);
        T->Transform(ip, x);
        uc.Eval(uv, *T, ip);
        const double wq = ip.weight * T->Weight() * rho_used.Eval(*T, ip);
        h[0] += wq * (x[1] * uv[2] - x[2] * uv[1]);
        h[1] += wq * (x[2] * uv[0] - x[0] * uv[2]);
        h[2] += wq * (x[0] * uv[1] - x[1] * uv[0]);
      }
    }
    for (int k = 0; k < 3; k++) {
      h[k] = GlobalSum(h[k]);
    }
    const DenseMatrix I0 = problem.ModelInertiaTensor();
    const double trace = I0(0, 0) + I0(1, 1) + I0(2, 2);
    // The discrete gauge pins the ASSEMBLED rho-mass pairing; the
    // certificate re-integrates at 2*order+4, so the gap is the mass
    // form's quadrature on the triaxial density: measured 4.0e-5 at
    // o1, 1e-7-grade at o2 (the serial door gate).
    Check(h.Norml2() / trace, order == 1 ? 1e-4 : 1e-7,
          label + " Tisserand certificate");
  }
}

// The fluid-core mantle-Tisserand leg (G8c phase 2,
// generalised_rotation_note.md "The fluid-core wrinkle"): the
// input-moments generalised door composed with the gauged fluid core
// on the two-layer ball. The gauge metric restricts to the solid
// attributes; the parallel certificates are the serial omega match and
// the mantle momentum re-integrated by rank-local quadrature.
void RunFluidCase(const std::string& label) {
  const int order = 1, dim = 3;
  const double r_cmb = 0.5467, r_icb = 0.1931;
  Mesh smesh("../data/elastogravity_two_layer_3d.msh", 1, 1);
  Array<int> body_attr({1, 2}), buffer_attr({3});
  Vector bb_min, bb_max;
  smesh.GetBoundingBox(bb_min, bb_max);
  const double r_out = bb_max.Normlinf();

  RadialHydrostaticBackground bg(
      dim,
      [=](double r) {
        return r < r_cmb ? 1.2 - 0.1 * (r - r_icb) / (r_cmb - r_icb) : 1.0;
      },
      [](double) { return kKappa; },
      [=](double r) { return r < r_cmb ? 0.0 : kMu; }, kG, 1.0);
  ConstantCoefficient mu_gauge(kKappa);
  FunctionCoefficient sigma(SigmaWithMean);
  Vector moments(3);
  moments[0] = 1.2;
  moments[1] = 1.3;
  moments[2] = 2.0;
  const double Omega = 0.1;

  auto configure = [&](auto& problem, auto& body, auto& fes_buffer,
                       auto& E) {
    Array<int> fluid_marker(body.attributes.Max());
    fluid_marker = 0;
    fluid_marker[0] = 1;
    Array<int> surface(body.bdr_attributes.Max());
    surface = 0;
    surface[body.bdr_attributes.Max() - 1] = 1;
    problem.SetPrescribedVacuumExtension(fes_buffer, *E);
    problem.SetFluid(fluid_marker, mu_gauge,
                     GaugePenaltyOptions{1e-2, 3});
    problem.SetSurfaceLoad(sigma, surface);
    problem.SetRotation(Omega, moments);
    problem.SetTisserandGauge(true);
    problem.SetRelTol(1e-11);
  };

  // Serial reference on every rank.
  double u_ref = 0.0;
  Vector w_ref;
  {
    SubMesh body(SubMesh::CreateFromDomain(smesh, body_attr));
    SubMesh buffer(SubMesh::CreateFromDomain(smesh, buffer_attr));
    H1_FECollection fec(order, dim);
    FiniteElementSpace fes_u(&body, &fec, dim), fes_zeta(&smesh, &fec);
    FiniteElementSpace fes_buffer(&buffer, &fec, dim);
    LinearQuasiStaticReferentialSelfGravitatingProblem problem(
        &fes_u, &fes_zeta, bg.Rheology(), bg.Density(), kG, kDtNDegree);
    auto E = NewRadialVacuumExtension(fes_u, fes_buffer, 1.0, r_out);
    configure(problem, body, fes_buffer, E);
    problem.AssembleForce(0.0);
    Check(problem.Solve() ? 0.0 : 1.0, 0.0, label + " serial solve");
    u_ref = L2Norm(problem.Displacement());
    w_ref = problem.AngularVelocity();
  }

  // Parallel.
  int nxyz[3] = {Mpi::WorldSize(), 1, 1};
  int* partitioning = smesh.CartesianPartitioning(nxyz);
  ParMesh pmesh(MPI_COMM_WORLD, smesh, partitioning);
  delete[] partitioning;
  ParSubMesh body(ParSubMesh::CreateFromDomain(pmesh, body_attr));
  ParSubMesh buffer(ParSubMesh::CreateFromDomain(pmesh, buffer_attr));
  H1_FECollection fec(order, dim);
  ParFiniteElementSpace fes_u(&body, &fec, dim), fes_zeta(&pmesh, &fec);
  ParFiniteElementSpace fes_buffer(&buffer, &fec, dim);
  LinearQuasiStaticReferentialSelfGravitatingProblem problem(
      &fes_u, &fes_zeta, bg.Rheology(), bg.Density(), kG, kDtNDegree);
  auto E = NewRadialVacuumExtension(fes_u, fes_buffer, 1.0, r_out);
  configure(problem, body, fes_buffer, E);
  problem.AssembleForce(0.0);
  Check(problem.Solve() ? 0.0 : 1.0, 0.0, label + " parallel solve");
  Check(RelErr(L2Norm(problem.Displacement()), u_ref), 1e-5,
        label + " displacement norm");
  Vector dw(problem.AngularVelocity());
  dw -= w_ref;
  Check(dw.Norml2() / w_ref.Norml2(), 1e-5, label + " omega");

  // The mantle momentum, re-integrated rank-locally and reduced
  // (mantle rho = 1; the serial gate's grade, constant-density-exact).
  {
    VectorGridFunctionCoefficient uc(
        const_cast<GridFunction*>(&problem.Displacement()));
    Vector h(3), x(3), uv(3);
    h = 0.0;
    double trace = 0.0;
    for (int e = 0; e < body.GetNE(); e++) {
      if (body.GetAttribute(e) != 2) {
        continue;
      }
      auto* T = body.GetElementTransformation(e);
      const auto& ir =
          IntRules.Get(body.GetElementGeometry(e), 2 * order + 4);
      for (int q = 0; q < ir.GetNPoints(); q++) {
        const auto& ip = ir.IntPoint(q);
        T->SetIntPoint(&ip);
        T->Transform(ip, x);
        uc.Eval(uv, *T, ip);
        const double w = ip.weight * T->Weight();
        h[0] += w * (x[1] * uv[2] - x[2] * uv[1]);
        h[1] += w * (x[2] * uv[0] - x[0] * uv[2]);
        h[2] += w * (x[0] * uv[1] - x[1] * uv[0]);
        trace += w * 2.0 * (x * x);
      }
    }
    for (int k = 0; k < 3; k++) {
      h[k] = GlobalSum(h[k]);
    }
    trace = GlobalSum(trace);
    // Measured 3.6e-7: the serial gate's grade scaled by this leg's
    // O(1) load amplitude (h scales with u, the trace does not).
    Check(h.Norml2() / trace, 1e-6, label + " mantle Tisserand");
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();

  RunCase(1, "water o1");
  RunCase(1, "water+rotation o1", 1);
  RunCase(1, "water+generalised o1", 2);
  RunCase(2, "water+generalised o2", 2);
  RunFluidCase("fluid-core mantle-frame o1");

  const int fails = num_fails;
  int global_fails = 0;
  MPI_Allreduce(&fails, &global_fails, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (Mpi::Root()) {
    std::cout << (global_fails == 0 ? "PASS" : "FAIL") << " ("
              << num_checks << " checks)\n";
  }
  return global_fails == 0 ? 0 : 1;
}
