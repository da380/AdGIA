/*
  Parallel tests for LinearQuasiStaticReferentialSelfGravitatingProblem with the
  prescribed vacuum extension (see TestReferentialProblem.cpp for the
  serial suite and the model), for the slip class
  LinearQuasiStaticReferentialSelfGravitatingSlipProblem (single-valued
  and broken zeta), for NewHarmonicExtensionMapping and for the
  equilibrium-stress generators. Run with 1, 2 and 4 ranks; a standalone
  MPI program returning 0 if every check passes, 1 otherwise.

  Every rank also solves the serial problem on the full mesh: serial and
  parallel build the same discrete system (the parallel extension's trace
  rows come from the cross-rank dof pairing and its interior rows from the
  same interpolation), so global L2 norms of the fields are compared
  directly.
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
constexpr double kPi = std::numbers::pi;

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

double RelErr(double a, double b) { return std::abs(a - b) / std::abs(b); }

double Pressure(const Vector& x) {
  const double r2 = x * x;
  const double p = x.Size() == 2
                       ? kPi * kG * kRho * kRho * (1.0 - r2)
                       : 2.0 * kPi * kG * kRho * kRho * (1.0 - r2) / 3.0;
  return std::max(0.0, p);
}

double SurfaceLoad(const Vector& x) {
  const double r = x.Norml2();
  const double c = (x.Size() == 2 ? x[1] : x[2]) / r;
  return 0.02 * (1.0 + 3.0 * c * c);
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

void RunCase(int order, const std::string& label) {
  const char* mesh_file = "../data/elastogravity_2d.msh";
  Mesh smesh(mesh_file, 1, 1);
  const int dim = smesh.Dimension();
  Array<int> body_attr({1}), buffer_attr({2});

  Vector bb_min, bb_max;
  smesh.GetBoundingBox(bb_min, bb_max);
  const double r_out = bb_max.Normlinf();

  auto phi = IdentityMap(dim);
  ConstantCoefficient kappa(kKappa), mu(kMu), rho(kRho);
  FunctionCoefficient p0(Pressure);
  auto C_eff =
      IsotropicElasticTensorCoefficient::FromBulkModulus(dim, kappa, mu);
  BareElasticTensorCoefficient C(dim, C_eff, p0);
  MatrixFunctionCoefficient S(dim, [](const Vector& x, DenseMatrix& S) {
    S.SetSize(x.Size());
    S = 0.0;
    const double p = Pressure(x);
    for (int i = 0; i < x.Size(); i++) {
      S(i, i) = -p;
    }
  });
  ReferentialElasticRheology rheology(dim, C, S, phi);
  FunctionCoefficient sigma(SurfaceLoad);

  // Serial reference on every rank.
  double u_ref = 0.0, z_ref = 0.0;
  {
    SubMesh body(SubMesh::CreateFromDomain(smesh, body_attr));
    SubMesh buffer(SubMesh::CreateFromDomain(smesh, buffer_attr));
    H1_FECollection fec(order, dim);
    FiniteElementSpace fes_u(&body, &fec, dim), fes_zeta(&smesh, &fec);
    FiniteElementSpace fes_buffer(&buffer, &fec, dim);
    LinearQuasiStaticReferentialSelfGravitatingProblem problem(&fes_u, &fes_zeta, rheology,
                                                rho, kG, kDtNDegree);
    auto E = NewRadialVacuumExtension(fes_u, fes_buffer, 1.0, r_out);
    problem.SetPrescribedVacuumExtension(fes_buffer, *E);
    Array<int> surface(body.bdr_attributes.Max());
    surface = 0;
    surface[body.bdr_attributes.Max() - 1] = 1;
    problem.SetSurfaceLoad(sigma, surface);
    problem.SetRelTol(1e-11);
    problem.AssembleForce(0.0);
    Check(problem.Solve() ? 0.0 : 1.0, 0.0, label + " serial solve");
    u_ref = L2Norm(problem.Displacement());
    z_ref = L2Norm(problem.Potential());
  }

  // Parallel problem.
  ParMesh pmesh(MPI_COMM_WORLD, smesh);
  ParSubMesh body(ParSubMesh::CreateFromDomain(pmesh, body_attr));
  ParSubMesh buffer(ParSubMesh::CreateFromDomain(pmesh, buffer_attr));
  H1_FECollection fec(order, dim);
  ParFiniteElementSpace fes_u(&body, &fec, dim), fes_zeta(&pmesh, &fec);
  ParFiniteElementSpace fes_buffer(&buffer, &fec, dim);
  LinearQuasiStaticReferentialSelfGravitatingProblem problem(&fes_u, &fes_zeta, rheology,
                                              rho, kG, kDtNDegree);
  auto E = NewRadialVacuumExtension(fes_u, fes_buffer, 1.0, r_out);
  problem.SetPrescribedVacuumExtension(fes_buffer, *E);
  Array<int> surface(body.bdr_attributes.Max());
  surface = 0;
  surface[body.bdr_attributes.Max() - 1] = 1;
  FunctionCoefficient sigma2(SurfaceLoad);
  problem.SetSurfaceLoad(sigma2, surface);
  problem.SetRelTol(1e-11);
  problem.AssembleForce(0.0);
  Check(problem.Solve() ? 0.0 : 1.0, 0.0, label + " parallel solve");
  Check(RelErr(L2Norm(problem.Displacement()), u_ref), 1e-5,
        label + " displacement norm");
  Check(RelErr(L2Norm(problem.Potential()), z_ref), 1e-5,
        label + " potential norm");
}

// Serial-vs-parallel agreement of the three-block slip solver on the
// two-layer disc (fluid core, solid mantle): the parallel pairing,
// fluid extension, B_Sigma blocks and mismatch folds build the same
// discrete system as the serial path, so the global L2 norms of the
// solid displacement, the fluid displacement and the potential are
// compared directly (the fluid displacement carries the fluid gauge, but
// the eps penalty fixes it in the discrete system).
void RunSlipCase(int order, const std::string& label) {
  const char* mesh_file = "../data/elastogravity_two_layer_2d.msh";
  Mesh smesh(mesh_file, 1, 1);
  const int dim = smesh.Dimension();
  const double r_cmb = 3483.0 / 6371.0;
  Array<int> fluid_attr({1}), solid_attr({2}), buffer_attr({3});

  Vector bb_min, bb_max;
  smesh.GetBoundingBox(bb_min, bb_max);
  const double r_out = bb_max.Normlinf();

  RadialHydrostaticBackground bg(
      dim, [](double) { return kRho; }, [](double) { return kKappa; },
      [r_cmb](double r) { return r < r_cmb ? 0.0 : kMu; }, kG, 1.0);
  FunctionCoefficient sigma(SurfaceLoad);
  ConstantCoefficient mu_gauge(kKappa);

  auto interface_marker = [r_cmb](Mesh& solid) {
    Array<int> marker(solid.bdr_attributes.Max());
    marker = 0;
    for (int i = 0; i < solid.GetNBE(); i++) {
      auto* tr = solid.GetBdrElementTransformation(i);
      Vector c(solid.Dimension());
      tr->Transform(Geometries.GetCenter(solid.GetBdrElementGeometry(i)), c);
      const double r = c.Norml2();
      if (r > 0.9 * r_cmb && r < 1.1 * r_cmb) {
        marker[solid.GetBdrAttribute(i) - 1] = 1;
      }
    }
    return marker;
  };
  auto surface_marker = [](Mesh& solid) {
    Array<int> marker(solid.bdr_attributes.Max());
    marker = 0;
    for (int i = 0; i < solid.GetNBE(); i++) {
      auto* tr = solid.GetBdrElementTransformation(i);
      Vector c(solid.Dimension());
      tr->Transform(Geometries.GetCenter(solid.GetBdrElementGeometry(i)), c);
      if (c.Norml2() > 0.9) {
        marker[solid.GetBdrAttribute(i) - 1] = 1;
      }
    }
    return marker;
  };

  // Serial reference on every rank.
  double us_ref = 0.0, uf_ref = 0.0, z_ref = 0.0;
  {
    SubMesh solid(SubMesh::CreateFromDomain(smesh, solid_attr));
    SubMesh fluid(SubMesh::CreateFromDomain(smesh, fluid_attr));
    SubMesh buffer(SubMesh::CreateFromDomain(smesh, buffer_attr));
    H1_FECollection fec(order, dim);
    FiniteElementSpace fes_s(&solid, &fec, dim), fes_f(&fluid, &fec, dim);
    FiniteElementSpace fes_buffer(&buffer, &fec, dim), fes_zeta(&smesh, &fec);
    auto marker = interface_marker(solid);
    LinearQuasiStaticReferentialSelfGravitatingSlipProblem problem(
        &fes_s, &fes_f, &fes_zeta, bg.Rheology(), bg.Density(), bg.Pressure(),
        marker, kG, kDtNDegree);
    auto Evac = NewRadialVacuumExtension(fes_s, fes_buffer, 1.0, r_out);
    problem.SetPrescribedVacuumExtension(fes_buffer, *Evac);
    auto Ef = NewRadialFluidExtension(fes_s, fes_f, r_cmb);
    problem.SetFluidExtension(*Ef);
    problem.SetFluidGauge(mu_gauge, 1e-2);
    problem.SetConstraint(1e2, 6);
    auto surface = surface_marker(solid);
    problem.SetSurfaceLoad(sigma, surface);
    problem.SetRelTol(1e-11);
    problem.AssembleForce(0.0);
    Check(problem.Solve() ? 0.0 : 1.0, 0.0, label + " serial slip solve");
    us_ref = L2Norm(problem.Displacement());
    uf_ref = L2Norm(problem.FluidDisplacement());
    z_ref = L2Norm(problem.Potential());
  }

  // Parallel problem.
  ParMesh pmesh(MPI_COMM_WORLD, smesh);
  ParSubMesh solid(ParSubMesh::CreateFromDomain(pmesh, solid_attr));
  ParSubMesh fluid(ParSubMesh::CreateFromDomain(pmesh, fluid_attr));
  ParSubMesh buffer(ParSubMesh::CreateFromDomain(pmesh, buffer_attr));
  H1_FECollection fec(order, dim);
  ParFiniteElementSpace fes_s(&solid, &fec, dim), fes_f(&fluid, &fec, dim);
  ParFiniteElementSpace fes_buffer(&buffer, &fec, dim), fes_zeta(&pmesh, &fec);
  auto marker = interface_marker(solid);
  LinearQuasiStaticReferentialSelfGravitatingSlipProblem problem(
      &fes_s, &fes_f, &fes_zeta, bg.Rheology(), bg.Density(), bg.Pressure(),
      marker, kG, kDtNDegree);
  auto Evac = NewRadialVacuumExtension(fes_s, fes_buffer, 1.0, r_out);
  problem.SetPrescribedVacuumExtension(fes_buffer, *Evac);
  auto Ef = NewRadialFluidExtension(fes_s, fes_f, r_cmb);
  problem.SetFluidExtension(*Ef);
  problem.SetFluidGauge(mu_gauge, 1e-2);
  problem.SetConstraint(1e2, 6);
  auto surface = surface_marker(solid);
  FunctionCoefficient sigma2(SurfaceLoad);
  problem.SetSurfaceLoad(sigma2, surface);
  problem.SetRelTol(1e-11);
  problem.AssembleForce(0.0);
  Check(problem.Solve() ? 0.0 : 1.0, 0.0, label + " parallel slip solve");
  Check(RelErr(L2Norm(problem.Displacement()), us_ref), 1e-5,
        label + " slip solid displacement norm");
  Check(RelErr(L2Norm(problem.FluidDisplacement()), uf_ref), 1e-4,
        label + " slip fluid displacement norm");
  Check(RelErr(L2Norm(problem.Potential()), z_ref), 1e-5,
        label + " slip potential norm");
}

// Serial-vs-parallel agreement of the broken-zeta slip solver: the
// four-block system with per-region potentials, the G_Sigma blocks and
// both interface constraints (no fluid extension anywhere) must build
// the same discrete system on 1 and N ranks.
void RunBrokenSlipCase(int order, const std::string& label) {
  const char* mesh_file = "../data/elastogravity_two_layer_2d.msh";
  Mesh smesh(mesh_file, 1, 1);
  const int dim = smesh.Dimension();
  const double r_cmb = 3483.0 / 6371.0;
  Array<int> fluid_attr({1}), solid_attr({2}), buffer_attr({3}),
      outer_attr({2, 3});

  Vector bb_min, bb_max;
  smesh.GetBoundingBox(bb_min, bb_max);
  const double r_out = bb_max.Normlinf();

  RadialHydrostaticBackground bg(
      dim, [](double) { return kRho; }, [](double) { return kKappa; },
      [r_cmb](double r) { return r < r_cmb ? 0.0 : kMu; }, kG, 1.0);
  FunctionCoefficient sigma(SurfaceLoad);
  ConstantCoefficient mu_gauge(kKappa);

  auto interface_marker = [r_cmb](Mesh& solid) {
    Array<int> marker(solid.bdr_attributes.Max());
    marker = 0;
    for (int i = 0; i < solid.GetNBE(); i++) {
      auto* tr = solid.GetBdrElementTransformation(i);
      Vector c(solid.Dimension());
      tr->Transform(Geometries.GetCenter(solid.GetBdrElementGeometry(i)), c);
      const double r = c.Norml2();
      if (r > 0.9 * r_cmb && r < 1.1 * r_cmb) {
        marker[solid.GetBdrAttribute(i) - 1] = 1;
      }
    }
    return marker;
  };
  auto surface_marker = [](Mesh& solid) {
    Array<int> marker(solid.bdr_attributes.Max());
    marker = 0;
    for (int i = 0; i < solid.GetNBE(); i++) {
      auto* tr = solid.GetBdrElementTransformation(i);
      Vector c(solid.Dimension());
      tr->Transform(Geometries.GetCenter(solid.GetBdrElementGeometry(i)), c);
      if (c.Norml2() > 0.9) {
        marker[solid.GetBdrAttribute(i) - 1] = 1;
      }
    }
    return marker;
  };

  // Serial reference on every rank.
  double us_ref = 0.0, uf_ref = 0.0, z_ref = 0.0;
  {
    SubMesh solid(SubMesh::CreateFromDomain(smesh, solid_attr));
    SubMesh fluid(SubMesh::CreateFromDomain(smesh, fluid_attr));
    SubMesh buffer(SubMesh::CreateFromDomain(smesh, buffer_attr));
    SubMesh outer(SubMesh::CreateFromDomain(smesh, outer_attr));
    H1_FECollection fec(order, dim);
    FiniteElementSpace fes_s(&solid, &fec, dim), fes_f(&fluid, &fec, dim);
    FiniteElementSpace fes_buffer(&buffer, &fec, dim), fes_zeta(&smesh, &fec);
    auto marker = interface_marker(solid);
    LinearQuasiStaticReferentialSelfGravitatingSlipProblem problem(
        &fes_s, &fes_f, &fes_zeta, bg.Rheology(), bg.Density(), bg.Pressure(),
        marker, kG, kDtNDegree);
    auto Evac = NewRadialVacuumExtension(fes_s, fes_buffer, 1.0, r_out);
    problem.SetPrescribedVacuumExtension(fes_buffer, *Evac);
    problem.SetFluidGauge(mu_gauge, 1e-2);
    problem.SetConstraint(1e2, 6);
    auto fes_zo = SubMeshDofInjection::MakeShadowSpace(fes_zeta, outer);
    problem.EnableBrokenZeta(fes_zo.get(), 1e2);
    auto surface = surface_marker(solid);
    problem.SetSurfaceLoad(sigma, surface);
    problem.SetRelTol(1e-11);
    problem.AssembleForce(0.0);
    Check(problem.Solve() ? 0.0 : 1.0, 0.0,
          label + " serial broken-zeta solve");
    us_ref = L2Norm(problem.Displacement());
    uf_ref = L2Norm(problem.FluidDisplacement());
    z_ref = L2Norm(problem.Potential());
  }

  // Parallel problem.
  ParMesh pmesh(MPI_COMM_WORLD, smesh);
  ParSubMesh solid(ParSubMesh::CreateFromDomain(pmesh, solid_attr));
  ParSubMesh fluid(ParSubMesh::CreateFromDomain(pmesh, fluid_attr));
  ParSubMesh buffer(ParSubMesh::CreateFromDomain(pmesh, buffer_attr));
  ParSubMesh outer(ParSubMesh::CreateFromDomain(pmesh, outer_attr));
  H1_FECollection fec(order, dim);
  ParFiniteElementSpace fes_s(&solid, &fec, dim), fes_f(&fluid, &fec, dim);
  ParFiniteElementSpace fes_buffer(&buffer, &fec, dim), fes_zeta(&pmesh, &fec);
  auto marker = interface_marker(solid);
  LinearQuasiStaticReferentialSelfGravitatingSlipProblem problem(
      &fes_s, &fes_f, &fes_zeta, bg.Rheology(), bg.Density(), bg.Pressure(),
      marker, kG, kDtNDegree);
  auto Evac = NewRadialVacuumExtension(fes_s, fes_buffer, 1.0, r_out);
  problem.SetPrescribedVacuumExtension(fes_buffer, *Evac);
  problem.SetFluidGauge(mu_gauge, 1e-2);
  problem.SetConstraint(1e2, 6);
  auto fes_zo = SubMeshDofInjection::MakeShadowSpace(fes_zeta, outer);
  problem.EnableBrokenZeta(fes_zo.get(), 1e2);
  auto surface = surface_marker(solid);
  FunctionCoefficient sigma2(SurfaceLoad);
  problem.SetSurfaceLoad(sigma2, surface);
  problem.SetRelTol(1e-11);
  problem.AssembleForce(0.0);
  Check(problem.Solve() ? 0.0 : 1.0, 0.0,
        label + " parallel broken-zeta solve");
  Check(RelErr(L2Norm(problem.Displacement()), us_ref), 1e-5,
        label + " broken-zeta solid displacement norm");
  Check(RelErr(L2Norm(problem.FluidDisplacement()), uf_ref), 1e-4,
        label + " broken-zeta fluid displacement norm");
  Check(RelErr(L2Norm(problem.Potential()), z_ref), 1e-5,
        label + " broken-zeta potential norm");
}

// Serial-vs-parallel agreement of the harmonic buffer extension of a
// mapping that is non-trivial on the physical surface.
void RunHarmonicExtensionCase() {
  const char* mesh_file = "../data/elastogravity_2d.msh";
  Mesh smesh(mesh_file, 1, 1);
  const int dim = smesh.Dimension();

  const double c = 0.05;
  auto q = [](double r) { return 1.0 - r * r / 1.44; };
  auto f = [c, q](double r) { return 1.0 + c * q(r) * q(r); };
  auto df = [c, q](double r) {
    return c * 2.0 * q(r) * (-2.0 * r / 1.44);
  };
  Array<int> body_attr({1}), buffer_attr({2});

  double ref = 0.0;
  {
    RadialDiffeomorphism xi(dim, f, df);
    auto phi =
        NewHarmonicExtensionMapping(smesh, 2, xi, body_attr, buffer_attr);
    ref = L2Norm(phi.Displacement());
  }

  ParMesh pmesh(MPI_COMM_WORLD, smesh);
  RadialDiffeomorphism xi(dim, f, df);
  auto phi = NewHarmonicExtensionMapping(pmesh, 2, xi, body_attr, buffer_attr);
  Check(RelErr(L2Norm(phi.Displacement()), ref), 1e-8,
        "harmonic extension displacement norm");
}

// Serial-vs-parallel agreement of the equilibrium-stress generators of
// Al-Attar & Woodhouse (2010) (doc/gravitating_elasticity.md, §6) (the auxiliary fields are rigid-projected, so they compare
// directly).
void RunEquilibriumStressCase() {
  Mesh smesh("../data/elastogravity_2d.msh", 1, 1);
  const int dim = smesh.Dimension();
  Array<int> body_attr({1});
  VectorFunctionCoefficient f(dim, [](const Vector& x, Vector& v) {
    v = x;
    v *= 2.0 * kPi * kG * kRho * kRho;
  });

  double u_ref = 0.0, p_ref = 0.0;
  {
    SubMesh body(SubMesh::CreateFromDomain(smesh, body_attr));
    H1_FECollection fec2(2, dim), fec1(1, dim);
    FiniteElementSpace fes_u(&body, &fec2, dim), fes_p(&body, &fec1);
    MinimumNormEquilibriumStress T1(fes_u, f);
    MinimumDeviatoricEquilibriumStress T2(fes_u, fes_p, f);
    u_ref = L2Norm(T1.Auxiliary());
    p_ref = L2Norm(T2.Pressure());
  }

  ParMesh pmesh(MPI_COMM_WORLD, smesh);
  ParSubMesh body(ParSubMesh::CreateFromDomain(pmesh, body_attr));
  H1_FECollection fec2(2, dim), fec1(1, dim);
  ParFiniteElementSpace fes_u(&body, &fec2, dim), fes_p(&body, &fec1);
  MinimumNormEquilibriumStress T1(fes_u, f);
  MinimumDeviatoricEquilibriumStress T2(fes_u, fes_p, f);
  Check(RelErr(L2Norm(T1.Auxiliary()), u_ref), 1e-6,
        "minimum-norm auxiliary field norm");
  Check(RelErr(L2Norm(T2.Pressure()), p_ref), 1e-6,
        "minimum-deviatoric pressure norm");
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();

  RunCase(1, "o1");
  RunCase(2, "o2");
  RunSlipCase(1, "slip o1");
  RunSlipCase(2, "slip o2");
  RunBrokenSlipCase(1, "broken o1");
  RunBrokenSlipCase(2, "broken o2");
  RunHarmonicExtensionCase();
  RunEquilibriumStressCase();

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
