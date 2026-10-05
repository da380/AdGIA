#include <numbers>

#include "MixedProblemTestCommon.hpp"
#include "TestCommon.hpp"

/*
  Tests for LinearQuasiStaticReferentialSelfGravitatingSlipProblem (the three-block
  slip-interface solver of doc/slip_interface.tex, "The collected
  operator"; the reductions (a)-(d) checked here are listed there and in
  its "Verification" section) on the two-layer disc: fluid core
  (attribute 1), solid mantle (2), buffer shell (3), DtN sphere.

  - Barotropic cross-check (reduction (b)), the acceptance test of the
    assembly: on the two-layer hydrostatic background the broken
    (slipping) formulation must reproduce the welded gauged solution of
    the same physical problem. For this spherically symmetric model the
    tangential slip that the weld suppresses is absent or removable by
    relabelling (doc/gauged_fluid.md, "Tangential slip and the welded
    space"; not true in general), so the solid displacement (modulo rigid
    modes) and the potential agree at the two discretisations' common
    accuracy. Unlike the gravity-free TestSlidingInterface,
    pi(r_cmb) > 0 here, so B_Sigma, the mismatch gravity pieces and all
    the extension folds are load-bearing.
  - The AL iterations contract the normal jump far below the tangential
    jump (the slip the constraint must leave free).
  - Extension invariance: fluid extensions of taper powers 2 and 3 agree
    on the observables -- the built-in gauge test of the design.
  - KKT enforcement of the normal-jump constraint against the AL
    iterations.
  - Reduction (a): the mixed-class Dahlen solution on the sphere.
  - Reduction (c): a massless, pressure-free background gives the
    gravity-free sliding cavity.
  - Reduction (d): the broken-zeta organisation head to head against the
    single-valued one.
  - Rigid pairs (common translations, independent shell/core rotations)
    are near-null under the physical block operator, in both
    organisations.
  - The broken-zeta solver on a relabelled background reproduces the
    identity description under composition.
*/

namespace {

using namespace self_grav_test;

constexpr double kRc = 3483.0 / 6371.0;
constexpr double kEps = 1.0e-2;
constexpr double kTheta = 1.0e2;
constexpr int kALIterations = 8;

// Marker for the boundary attributes at radius in (r_min, r_max).
Array<int> RadialBdrMarker(Mesh& mesh, double r_min, double r_max) {
  Array<int> marker(mesh.bdr_attributes.Max());
  marker = 0;
  for (int i = 0; i < mesh.GetNBE(); i++) {
    auto* tr = mesh.GetBdrElementTransformation(i);
    Vector c(mesh.Dimension());
    tr->Transform(Geometries.GetCenter(mesh.GetBdrElementGeometry(i)), c);
    const double r = c.Norml2();
    if (r > r_min && r < r_max) {
      marker[mesh.GetBdrAttribute(i) - 1] = 1;
    }
  }
  return marker;
}

// The two-layer setting shared by the tests: parent ball, solid/fluid/
// buffer SubMeshes, spaces on a common collection, and the background.
struct Setting {
  std::unique_ptr<Mesh> parent;
  std::unique_ptr<SubMesh> solid, fluid, buffer, body;
  std::unique_ptr<H1_FECollection> fec;
  std::unique_ptr<FiniteElementSpace> fes_s, fes_f, fes_buffer, fes_zeta,
      fes_body, fes_zeta_ref;
  std::unique_ptr<RadialHydrostaticBackground> bg;
  double r_out = 0.0;

  explicit Setting(int order = 2) {
    parent = std::make_unique<Mesh>("../data/elastogravity_two_layer_2d.msh",
                                    1, 1);
    const int dim = parent->Dimension();
    Array<int> fluid_attr({1}), solid_attr({2}), buffer_attr({3}),
        body_attr({1, 2});
    solid = std::make_unique<SubMesh>(
        SubMesh::CreateFromDomain(*parent, solid_attr));
    fluid = std::make_unique<SubMesh>(
        SubMesh::CreateFromDomain(*parent, fluid_attr));
    buffer = std::make_unique<SubMesh>(
        SubMesh::CreateFromDomain(*parent, buffer_attr));
    body = std::make_unique<SubMesh>(
        SubMesh::CreateFromDomain(*parent, body_attr));
    fec = std::make_unique<H1_FECollection>(order, dim);
    fes_s = std::make_unique<FiniteElementSpace>(solid.get(), fec.get(), dim);
    fes_f = std::make_unique<FiniteElementSpace>(fluid.get(), fec.get(), dim);
    fes_buffer =
        std::make_unique<FiniteElementSpace>(buffer.get(), fec.get(), dim);
    fes_zeta = std::make_unique<FiniteElementSpace>(parent.get(), fec.get());
    fes_body = std::make_unique<FiniteElementSpace>(body.get(), fec.get(), dim);
    fes_zeta_ref =
        std::make_unique<FiniteElementSpace>(parent.get(), fec.get());
    Vector bb_min, bb_max;
    parent->GetBoundingBox(bb_min, bb_max);
    r_out = bb_max.Normlinf();
    bg = std::make_unique<RadialHydrostaticBackground>(
        dim, [](double) { return kRho; }, [](double) { return kKappa; },
        [](double r) { return r < kRc ? 0.0 : kMu; }, kG, 1.0);
  }
};

double SurfaceSigma(const Vector& x) {
  const double r = x.Norml2();
  const double c = x[1] / r;
  return 0.02 * (1.0 + (2.0 * c * c - 1.0));
}

// Pure degree-2 surface load (no degree-0 part): the Dahlen reduction
// must avoid l = 0, where the Dahlen fluid treatment has its known gap
// (the gauged family, and hence the slip solver, sits on the other side
// of it).
double SurfaceSigmaDegree2(const Vector& x) {
  const double r = x.Norml2();
  const double c = x[1] / r;
  return 0.02 * (2.0 * c * c - 1.0);
}

// Radial pressure traction of the gravity-free sliding tests.
void PressureTraction(const Vector& x, Vector& f) {
  const double r = x.Norml2();
  const double c = x[1] / r;
  f = x;
  f *= -0.01 * (1.0 + (2.0 * c * c - 1.0)) / r;
}

// Evaluate a grid function at a physical point of its mesh.
bool EvalAt(const GridFunction& g, Mesh& m, const Vector& x, Vector& out) {
  DenseMatrix pt(x.Size(), 1);
  for (int d = 0; d < x.Size(); d++) {
    pt(d, 0) = x(d);
  }
  Array<int> elem;
  Array<IntegrationPoint> ips;
  if (m.FindPoints(pt, elem, ips, false) != 1 || elem[0] < 0) {
    return false;
  }
  const int vdim = g.FESpace()->GetVDim();
  out.SetSize(vdim);
  if (vdim == 1) {
    out(0) = g.GetValue(elem[0], ips[0]);
  } else {
    g.GetVectorValue(elem[0], ips[0], out);
  }
  return true;
}

// The interface-fixing relabelling of the broken-zeta mapped test:
//   xi(x) = f(r) Rot(alpha(r)) x,
// with f = 1 + c [r (rc - r)(1 - r) / q0]^2 (identity radius at the
// centre, the INTERFACE and the surface, so the referential mesh
// interface is the physical one) and a twist alpha = a [r(1-r)/p0]^2
// that is NONZERO across the interface (a non-radial F_e on Sigma);
// both are identity on and outside the surface (C^1 there).
struct SlipRelabelling {
  double c = 0.02, a = 0.05;
  static constexpr double q0 = 0.055, p0 = 0.25;

  double f(double r) const {
    if (r >= 1.0) {
      return 1.0;
    }
    const double q = r * (kRc - r) * (1.0 - r) / q0;
    return 1.0 + c * q * q;
  }
  double fp(double r) const {
    if (r >= 1.0) {
      return 0.0;
    }
    const double q = r * (kRc - r) * (1.0 - r) / q0;
    const double qp =
        ((kRc - r) * (1.0 - r) - r * (1.0 - r) - r * (kRc - r)) / q0;
    return 2.0 * c * q * qp;
  }
  double alpha(double r) const {
    if (r >= 1.0) {
      return 0.0;
    }
    const double p = r * (1.0 - r) / p0;
    return a * p * p;
  }
  double alphap(double r) const {
    if (r >= 1.0) {
      return 0.0;
    }
    const double p = r * (1.0 - r) / p0;
    return 2.0 * a * p * (1.0 - 2.0 * r) / p0;
  }
  void Map(const Vector& x, Vector& y) const {
    const double r = x.Norml2();
    const double ch = std::cos(alpha(r)), sh = std::sin(alpha(r));
    const double s = f(r);
    y.SetSize(2);
    y(0) = s * (ch * x(0) - sh * x(1));
    y(1) = s * (sh * x(0) + ch * x(1));
  }
  void Grad(const Vector& x, DenseMatrix& F) const {
    const double r = x.Norml2();
    F.SetSize(2);
    const double ch = std::cos(alpha(r)), sh = std::sin(alpha(r));
    const double s = f(r);
    // F = R [ f I + (f' x + f alpha' (z cross x)) otimes x-hat ].
    DenseMatrix M(2);
    M = 0.0;
    M(0, 0) = M(1, 1) = s;
    if (r > 1e-12) {
      const double sp = fp(r), ap = alphap(r);
      const double xh[2] = {x(0) / r, x(1) / r};
      const double zx[2] = {-x(1), x(0)};
      const double xx[2] = {x(0), x(1)};
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
          M(i, j) += (sp * xx[i] + s * ap * zx[i]) * xh[j];
        }
      }
    }
    F(0, 0) = ch * M(0, 0) - sh * M(1, 0);
    F(0, 1) = ch * M(0, 1) - sh * M(1, 1);
    F(1, 0) = sh * M(0, 0) + ch * M(1, 0);
    F(1, 1) = sh * M(0, 1) + ch * M(1, 1);
  }
};

}  // namespace

TEST(SlipProblem, TwoLayerBarotropicCrossCheck) {
  const int order = 2;
  Setting s(order);
  const int dim = 2;

  auto interface = RadialBdrMarker(*s.solid, 0.9 * kRc, 1.1 * kRc);
  auto surface_s = RadialBdrMarker(*s.solid, 0.9, 1.1);
  FunctionCoefficient sigma(SurfaceSigma);
  ConstantCoefficient mu_gauge(kKappa);

  // --- The slipping three-block problem.
  LinearQuasiStaticReferentialSelfGravitatingSlipProblem slip(
      s.fes_s.get(), s.fes_f.get(), s.fes_zeta.get(), s.bg->Rheology(),
      s.bg->Density(), s.bg->Pressure(), interface, kG, kDtNDegree);
  auto Evac = NewRadialVacuumExtension(*s.fes_s, *s.fes_buffer, 1.0, s.r_out);
  slip.SetPrescribedVacuumExtension(*s.fes_buffer, *Evac);
  auto Ef = NewRadialFluidExtension(*s.fes_s, *s.fes_f, kRc);
  slip.SetFluidExtension(*Ef);
  slip.SetFluidGauge(mu_gauge, kEps);
  slip.SetConstraint(kTheta, kALIterations);
  slip.SetSurfaceLoad(sigma, surface_s);
  slip.SetRelTol(1e-10);
  slip.AssembleForce(0.0);
  ASSERT_TRUE(slip.Solve());

  // The AL iterations contract the normal jump (inexact early sweeps;
  // the final sweep runs at the full tolerance).
  const auto& jumps = slip.NormalJumpHistory();
  ASSERT_EQ(static_cast<int>(jumps.size()), kALIterations);
  EXPECT_LT(jumps.back(), 0.05 * jumps.front());
  std::cout << "normal jump: first " << jumps.front() << ", last "
            << jumps.back() << "\n";

  // The tangential jump stays free: well above the normal jump.
  {
    FiniteElementSpace parent_v(s.parent.get(), s.fec.get(), dim);
    SubMeshDofInjection inj_s(*s.fes_s, parent_v), inj_f(*s.fes_f, parent_v);
    auto J = NewSubMeshPairingMatrix(inj_s, inj_f);
    Vector js(s.fes_s->GetVSize());
    J->Mult(slip.FluidDisplacement(), js);
    js -= slip.Displacement();
    ConstantCoefficient one(1.0);
    BilinearForm bn(s.fes_s.get()), mm(s.fes_s.get());
    bn.AddBoundaryIntegrator(new BoundaryNormalNormalIntegrator(one),
                             interface);
    mm.AddBoundaryIntegrator(new VectorMassIntegrator(one), interface);
    bn.Assemble();
    bn.Finalize();
    mm.Assemble();
    mm.Finalize();
    Vector t(js.Size());
    bn.SpMat().Mult(js, t);
    const double normal2 = std::abs(InnerProduct(js, t));
    mm.SpMat().Mult(js, t);
    const double full2 = std::abs(InnerProduct(js, t));
    EXPECT_GT(full2 - normal2, 1e2 * normal2);
  }

  // --- The welded gauged reference: the same physical problem through
  // the verified general class (continuous space over both layers).
  LinearQuasiStaticReferentialSelfGravitatingProblem welded(
      s.fes_body.get(), s.fes_zeta_ref.get(), s.bg->Rheology(),
      s.bg->Density(), kG, kDtNDegree);
  auto Evac_b =
      NewRadialVacuumExtension(*s.fes_body, *s.fes_buffer, 1.0, s.r_out);
  welded.SetPrescribedVacuumExtension(*s.fes_buffer, *Evac_b);
  Array<int> fluid_marker(s.body->attributes.Max());
  fluid_marker = 0;
  fluid_marker[0] = 1;
  welded.SetGaugedFluid(fluid_marker, mu_gauge, kEps, 3);
  auto surface_b = RadialBdrMarker(*s.body, 0.9, 1.1);
  welded.SetSurfaceLoad(sigma, surface_b);
  welded.SetRelTol(1e-10);
  welded.AssembleForce(0.0);
  ASSERT_TRUE(welded.Solve());

  // Solid (mantle) displacement, modulo rigid modes: transfer the welded
  // solution through the parent onto the solid SubMesh.
  {
    FiniteElementSpace parent_v(s.parent.get(), s.fec.get(), dim);
    GridFunction on_parent(&parent_v), on_solid(s.fes_s.get());
    on_parent = 0.0;
    SubMesh::Transfer(welded.Displacement(), on_parent);
    SubMesh::Transfer(on_parent, on_solid);
    Vector d(slip.Displacement());
    d -= on_solid;
    auto proj = MakeRigidModeProjector(*s.fes_s);
    proj->Project(d);
    Vector ref(on_solid);
    proj->Project(ref);
    const double rel = d.Norml2() / ref.Norml2();
    std::cout << "solid displacement, slip vs welded: " << rel << "\n";
    EXPECT_LT(rel, 3e-2);
  }

  // Potential on the SOLID region, modulo the 2-D constant. Off the
  // solid, zeta1 = phi1 + vtil . grad zeta0 carries the extension field
  // vtil, which differs between the two formulations (and between two
  // fluid extensions): zeta1 is an observable only where the motion is,
  // i.e. on the solid.
  auto solid_potential = [&](const GridFunction& zeta_ball) {
    auto shadow = SubMeshDofInjection::MakeShadowSpace(*s.fes_zeta, *s.solid);
    GridFunction z(shadow.get());
    SubMesh::Transfer(zeta_ball, z);
    Vector v(z);
    v -= v.Sum() / v.Size();
    return v;
  };
  {
    Vector zs = solid_potential(slip.Potential());
    Vector zw = solid_potential(welded.Potential());
    Vector d(zs);
    d -= zw;
    const double rel = d.Norml2() / zw.Norml2();
    std::cout << "potential (solid region), slip vs welded: " << rel << "\n";
    EXPECT_LT(rel, 3e-2);
  }

  // --- Extension invariance: a taper-power-3 fluid extension must give
  // the same observables (the interior rule is gauge).
  {
    LinearQuasiStaticReferentialSelfGravitatingSlipProblem slip3(
        s.fes_s.get(), s.fes_f.get(), s.fes_zeta.get(), s.bg->Rheology(),
        s.bg->Density(), s.bg->Pressure(), interface, kG, kDtNDegree);
    slip3.SetPrescribedVacuumExtension(*s.fes_buffer, *Evac);
    auto Ef3 = NewRadialFluidExtension(*s.fes_s, *s.fes_f, kRc, 3.0);
    slip3.SetFluidExtension(*Ef3);
    slip3.SetFluidGauge(mu_gauge, kEps);
    slip3.SetConstraint(kTheta, kALIterations);
    slip3.SetSurfaceLoad(sigma, surface_s);
    slip3.SetRelTol(1e-10);
    slip3.AssembleForce(0.0);
    ASSERT_TRUE(slip3.Solve());

    Vector du(slip.Displacement());
    du -= slip3.Displacement();
    const double rel_u = du.Norml2() / slip.Displacement().Norml2();
    Vector z2 = solid_potential(slip.Potential());
    Vector z3 = solid_potential(slip3.Potential());
    Vector dz(z2);
    dz -= z3;
    const double rel_z = dz.Norml2() / z2.Norml2();
    std::cout << "extension invariance: u " << rel_u << ", zeta " << rel_z
              << "\n";
    EXPECT_LT(rel_u, 1e-2);
    EXPECT_LT(rel_z, 1e-2);
  }
}

// KKT enforcement against the penalty + AL iterations on the same
// two-layer problem: the multiplier block enforces the normal-jump
// constraint within each solve (the jump lands at the solver floor at
// once), and the observables agree with the AL endpoint at the level of
// the AL iteration's own remaining constraint error.
TEST(SlipProblem, KKTMatchesAugmentedLagrangian) {
  const int order = 2;
  Setting s(order);
  auto interface = RadialBdrMarker(*s.solid, 0.9 * kRc, 1.1 * kRc);
  auto surface_s = RadialBdrMarker(*s.solid, 0.9, 1.1);
  FunctionCoefficient sigma(SurfaceSigma);
  ConstantCoefficient mu_gauge(kKappa);
  FiniteElementSpace lam_fes(s.solid.get(), s.fec.get());

  auto solve = [&](bool kkt) {
    auto p = std::make_unique<
        LinearQuasiStaticReferentialSelfGravitatingSlipProblem>(
        s.fes_s.get(), s.fes_f.get(), s.fes_zeta.get(), s.bg->Rheology(),
        s.bg->Density(), s.bg->Pressure(), interface, kG, kDtNDegree);
    auto Evac =
        NewRadialVacuumExtension(*s.fes_s, *s.fes_buffer, 1.0, s.r_out);
    p->SetPrescribedVacuumExtension(*s.fes_buffer, *Evac);
    auto Ef = NewRadialFluidExtension(*s.fes_s, *s.fes_f, kRc);
    p->SetFluidExtension(*Ef);
    p->SetFluidGauge(mu_gauge, kEps);
    p->SetConstraint(kTheta, kkt ? 3 : kALIterations);
    if (kkt) {
      p->EnableKKT(&lam_fes);
    }
    p->SetSurfaceLoad(sigma, surface_s);
    p->SetRelTol(1e-10);
    p->AssembleForce(0.0);
    EXPECT_TRUE(p->Solve());
    return p;
  };

  auto al = solve(false);
  auto kkt = solve(true);

  // Both enforcements land at the same DISCRETE constraint floor: the
  // multiplier zeroes the jump weakly (against the multiplier space),
  // so the L2 jump is the projection remainder — comparable to the
  // converged AL jump, and reached within one solve.
  const double j_al = al->NormalJumpHistory().back();
  const double j_kkt = kkt->NormalJumpHistory().back();
  std::cout << "KKT jump " << j_kkt << " vs AL converged jump " << j_al
            << "\n";
  EXPECT_LT(j_kkt, 5.0 * j_al);

  // Observables agree at the AL endpoint's residual-constraint level.
  Vector du(al->Displacement());
  du -= kkt->Displacement();
  const double rel =
      du.Normlinf() / std::max(al->Displacement().Normlinf(), 1e-300);
  std::cout << "KKT vs AL displacement relative diff " << rel << "\n";
  EXPECT_LT(rel, 1e-3);
}

// Reduction (a) of doc/slip_interface.tex ("The collected operator"): on
// the spherical hydrostatic two-layer background the slip solver must
// reproduce the mixed-class Dahlen solution (fluid displacement
// eliminated, F1-F3 interface terms) under the change of variables
// zeta1 = phi1 + u.grad Phi0. A pure degree-2 load keeps the comparison
// away from the Dahlen degree-0 gap. This is a genuinely two-sided check:
// the two formulations share no interface machinery.
TEST(SlipProblem, ReducesToDahlenOnSphere) {
  const int order = 2;
  Setting s(order);
  const int dim = 2;

  auto interface = RadialBdrMarker(*s.solid, 0.9 * kRc, 1.1 * kRc);
  auto surface_s = RadialBdrMarker(*s.solid, 0.9, 1.1);
  FunctionCoefficient sigma(SurfaceSigmaDegree2);
  ConstantCoefficient mu_gauge(kKappa);

  // The slipping three-block problem.
  LinearQuasiStaticReferentialSelfGravitatingSlipProblem slip(
      s.fes_s.get(), s.fes_f.get(), s.fes_zeta.get(), s.bg->Rheology(),
      s.bg->Density(), s.bg->Pressure(), interface, kG, kDtNDegree);
  auto Evac = NewRadialVacuumExtension(*s.fes_s, *s.fes_buffer, 1.0, s.r_out);
  slip.SetPrescribedVacuumExtension(*s.fes_buffer, *Evac);
  auto Ef = NewRadialFluidExtension(*s.fes_s, *s.fes_f, kRc);
  slip.SetFluidExtension(*Ef);
  slip.SetFluidGauge(mu_gauge, kEps);
  slip.SetConstraint(kTheta, kALIterations);
  slip.SetSurfaceLoad(sigma, surface_s);
  slip.SetRelTol(1e-10);
  slip.AssembleForce(0.0);
  ASSERT_TRUE(slip.Solve());

  // The Dahlen path: displacement on the mantle only, the fluid core as
  // a FluidRegion (uniform density, zero gradient), seismological
  // moduli.
  FiniteElementSpace fes_u_d(s.solid.get(), s.fec.get(), dim);
  FiniteElementSpace fes_phi(s.parent.get(), s.fec.get());
  ConstantCoefficient kappa_c(kKappa), mu_c(kMu), rho_c(kRho), zero(0.0);
  IsotropicElasticRheology e_rheology(dim, kappa_c, mu_c);
  FluidRegion core;
  core.attributes = Array<int>({1});
  core.density = &rho_c;
  core.density_gradient = &zero;
  core.interface_marker = interface;
  std::vector<FluidRegion> fluids{core};
  LinearQuasiStaticMixedSelfGravitatingProblem dahlen(
      &fes_u_d, &fes_phi, e_rheology, rho_c, kG, kDtNDegree, nullptr, fluids);
  FunctionCoefficient sigma_d(SurfaceSigmaDegree2);
  dahlen.SetSurfaceLoad(sigma_d, surface_s);
  dahlen.SetRelTol(1e-10);
  dahlen.AssembleForce(0.0);
  ASSERT_TRUE(dahlen.Solve());

  // Solid displacement, modulo rigid modes (same space layout).
  {
    Vector d(slip.Displacement());
    d -= dahlen.Displacement();
    auto proj = MakeRigidModeProjector(*s.fes_s);
    proj->Project(d);
    Vector ref(dahlen.Displacement());
    proj->Project(ref);
    const double rel = d.Norml2() / ref.Norml2();
    std::cout << "solid displacement, slip vs Dahlen: " << rel << "\n";
    EXPECT_LT(rel, 3e-2);
  }

  // Potential on the solid through the change of variables, modulo the
  // 2-D constant.
  {
    VectorGridFunctionCoefficient u_c(&dahlen.Displacement());
    InnerProductCoefficient advect(u_c, dahlen.BackgroundGravity());
    GridFunctionCoefficient phi1(&dahlen.PotentialOnBody());
    SumCoefficient zeta_expected(phi1, advect);
    GridFunction d(slip.PotentialOnBody());
    GridFunction z(d);
    z.ProjectCoefficient(zeta_expected);
    d -= z;
    d -= d.Sum() / d.Size();
    Vector ref(z);
    ref -= ref.Sum() / ref.Size();
    const double rel = d.Norml2() / ref.Norml2();
    std::cout << "potential (change of variables), slip vs Dahlen: " << rel
              << "\n";
    EXPECT_LT(rel, 3e-2);
  }
}

// Reduction (c) of doc/slip_interface.tex: with a massless background
// (pi = 0, gravity off) the slip solver reduces to the gravity-free sliding interface,
// whose solid solution has the condensed rank-one cavity form (the
// TestSlidingInterface reference): the fluid enters only through its
// bulk modulus against the interface volume change. The potential
// decouples and stays zero.
TEST(SlipProblem, PressureFreeReducesToSlidingCavity) {
  const int order = 2;
  Setting s(order);
  const int dim = 2;

  auto interface = RadialBdrMarker(*s.solid, 0.9 * kRc, 1.1 * kRc);
  auto surface_s = RadialBdrMarker(*s.solid, 0.9, 1.1);
  ConstantCoefficient mu_gauge(kKappa);
  VectorFunctionCoefficient traction(dim, PressureTraction);

  // Massless background: p0 = 0, zeta0 = 0, bare = seismological.
  RadialHydrostaticBackground bg0(
      dim, [](double) { return 0.0; }, [](double) { return kKappa; },
      [](double r) { return r < kRc ? 0.0 : kMu; }, kG, 1.0);

  LinearQuasiStaticReferentialSelfGravitatingSlipProblem slip(
      s.fes_s.get(), s.fes_f.get(), s.fes_zeta.get(), bg0.Rheology(),
      bg0.Density(), bg0.Pressure(), interface, kG, kDtNDegree);
  auto Evac = NewRadialVacuumExtension(*s.fes_s, *s.fes_buffer, 1.0, s.r_out);
  slip.SetPrescribedVacuumExtension(*s.fes_buffer, *Evac);
  auto Ef = NewRadialFluidExtension(*s.fes_s, *s.fes_f, kRc);
  slip.SetFluidExtension(*Ef);
  slip.SetFluidGauge(mu_gauge, kEps);
  slip.SetConstraint(kTheta, kALIterations);
  slip.ExternalLoad().AddBoundaryIntegrator(
      new VectorBoundaryLFIntegrator(traction), surface_s);
  slip.SetRelTol(1e-10);
  slip.AssembleForce(0.0);
  ASSERT_TRUE(slip.Solve());

  // The potential decouples (no density anywhere) and stays zero.
  {
    Vector z(slip.Potential());
    const double uscale = slip.Displacement().Norml2();
    EXPECT_LT(z.Norml2(), 1e-8 * uscale);
  }

  // Condensed rank-one cavity reference on the solid space (the
  // TestSlidingInterface construction): cavity solve plus the fluid's
  // bulk reaction kappa_f / V against the interface volume change.
  {
    ConstantCoefficient kappa_c(kKappa), mu_c(kMu), one(1.0);
    IsotropicElasticRheology rheology(dim, kappa_c, mu_c);
    LinearQuasiStaticTractionProblem cond(s.fes_s.get(), rheology, traction,
                                          surface_s);
    LinearForm b_lf(s.fes_s.get());
    b_lf.AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(one),
                               interface);
    b_lf.Assemble();
    double V = 0.0;
    for (int i = 0; i < s.fluid->GetNE(); i++) {
      V += s.fluid->GetElementVolume(i);
    }
    const double sig = kKappa / V;
    cond.AssembleForce(0.0);
    ASSERT_TRUE(cond.Solve());
    GridFunction u_a(cond.Displacement());
    cond.AssembleForce(0.0);
    Vector delta(b_lf);
    delta -= cond.ExternalLoad();
    cond.AddForce(delta);
    ASSERT_TRUE(cond.Solve());
    GridFunction u_b(cond.Displacement());
    GridFunction u_cond(u_a);
    u_cond.Add(-sig * b_lf(u_a) / (1.0 + sig * b_lf(u_b)), u_b);

    Vector d(slip.Displacement());
    d -= u_cond;
    auto proj = MakeRigidModeProjector(*s.fes_s);
    proj->Project(d);
    Vector ref(u_cond);
    proj->Project(ref);
    const double rel = d.Norml2() / ref.Norml2();
    std::cout << "solid displacement, pi = 0 slip vs condensed cavity: "
              << rel << "\n";
    EXPECT_LT(rel, 1e-2);
  }
}

// The broken-zeta organisation head to head against the single-valued
// one (doc/slip_interface.tex, "Gravity: the broken-zeta organisation",
// and reduction (d)): the
// same physical problem, same meshes and spaces, but the two
// organisations share NO gravity-interface machinery — mismatch volume
// terms + fluid extension on one side, G_Sigma + the scalar-jump
// constraint (and no fluid extension at all) on the other. Agreement of
// the solid displacement and the solid-region potential is the
// designed self-benchmark of both. Both constraints' AL iterations
// must contract their jumps.
TEST(SlipProblem, BrokenZetaHeadToHead) {
  const int order = 2;
  Setting s(order);
  const int dim = 2;

  auto interface = RadialBdrMarker(*s.solid, 0.9 * kRc, 1.1 * kRc);
  auto surface_s = RadialBdrMarker(*s.solid, 0.9, 1.1);
  FunctionCoefficient sigma(SurfaceSigma);
  ConstantCoefficient mu_gauge(kKappa);

  // --- The single-valued reference (the verified path).
  LinearQuasiStaticReferentialSelfGravitatingSlipProblem sv(
      s.fes_s.get(), s.fes_f.get(), s.fes_zeta.get(), s.bg->Rheology(),
      s.bg->Density(), s.bg->Pressure(), interface, kG, kDtNDegree);
  auto Evac = NewRadialVacuumExtension(*s.fes_s, *s.fes_buffer, 1.0, s.r_out);
  sv.SetPrescribedVacuumExtension(*s.fes_buffer, *Evac);
  auto Ef = NewRadialFluidExtension(*s.fes_s, *s.fes_f, kRc);
  sv.SetFluidExtension(*Ef);
  sv.SetFluidGauge(mu_gauge, kEps);
  sv.SetConstraint(kTheta, kALIterations);
  sv.SetSurfaceLoad(sigma, surface_s);
  sv.SetRelTol(1e-10);
  sv.AssembleForce(0.0);
  ASSERT_TRUE(sv.Solve());

  // --- The broken-zeta problem: outer (solid + buffer) scalar region,
  // no fluid extension anywhere.
  Array<int> outer_attr({2, 3});
  SubMesh outer(SubMesh::CreateFromDomain(*s.parent, outer_attr));
  auto fes_zo = SubMeshDofInjection::MakeShadowSpace(*s.fes_zeta, outer);

  LinearQuasiStaticReferentialSelfGravitatingSlipProblem bz(
      s.fes_s.get(), s.fes_f.get(), s.fes_zeta.get(), s.bg->Rheology(),
      s.bg->Density(), s.bg->Pressure(), interface, kG, kDtNDegree);
  bz.SetPrescribedVacuumExtension(*s.fes_buffer, *Evac);
  bz.SetFluidGauge(mu_gauge, kEps);
  bz.SetConstraint(kTheta, kALIterations);
  bz.EnableBrokenZeta(fes_zo.get(), kTheta);
  bz.SetSurfaceLoad(sigma, surface_s);
  bz.SetRelTol(1e-10);
  bz.AssembleForce(0.0);
  ASSERT_TRUE(bz.Solve());

  // Both constraints contract under the AL iterations (inexact early
  // sweeps, as in the single-valued test).
  {
    const auto& jn = bz.NormalJumpHistory();
    const auto& jz = bz.ZetaJumpHistory();
    ASSERT_EQ(static_cast<int>(jn.size()), kALIterations);
    ASSERT_EQ(jn.size(), jz.size());
    EXPECT_LT(jn.back(), 0.05 * jn.front());
    EXPECT_LT(jz.back(), 0.2 * jz.front());
    std::cout << "broken-zeta normal jump: first " << jn.front() << ", last "
              << jn.back() << "\n";
    std::cout << "broken-zeta scalar jump: first " << jz.front() << ", last "
              << jz.back() << "\n";
  }

  // Solid displacement, modulo rigid modes.
  {
    Vector d(bz.Displacement());
    d -= sv.Displacement();
    auto proj = MakeRigidModeProjector(*s.fes_s);
    proj->Project(d);
    Vector ref(sv.Displacement());
    proj->Project(ref);
    const double rel = d.Norml2() / ref.Norml2();
    std::cout << "solid displacement, broken vs single-valued: " << rel
              << "\n";
    EXPECT_LT(rel, 3e-2);
  }

  // Potential on the solid region, modulo the 2-D constant.
  {
    auto solid_potential = [&](const GridFunction& zeta_ball) {
      auto shadow =
          SubMeshDofInjection::MakeShadowSpace(*s.fes_zeta, *s.solid);
      GridFunction z(shadow.get());
      SubMesh::Transfer(zeta_ball, z);
      Vector v(z);
      v -= v.Sum() / v.Size();
      return v;
    };
    Vector zb = solid_potential(bz.Potential());
    Vector zs = solid_potential(sv.Potential());
    Vector d(zb);
    d -= zs;
    const double rel = d.Norml2() / zs.Norml2();
    std::cout << "potential (solid region), broken vs single-valued: " << rel
              << "\n";
    EXPECT_LT(rel, 3e-2);
  }
}

TEST(SlipProblem, RigidPairsNearNull) {
  Setting s(2);
  auto interface = RadialBdrMarker(*s.solid, 0.9 * kRc, 1.1 * kRc);
  ConstantCoefficient mu_gauge(kKappa);

  LinearQuasiStaticReferentialSelfGravitatingSlipProblem slip(
      s.fes_s.get(), s.fes_f.get(), s.fes_zeta.get(), s.bg->Rheology(),
      s.bg->Density(), s.bg->Pressure(), interface, kG, kDtNDegree);
  auto Evac = NewRadialVacuumExtension(*s.fes_s, *s.fes_buffer, 1.0, s.r_out);
  slip.SetPrescribedVacuumExtension(*s.fes_buffer, *Evac);
  auto Ef = NewRadialFluidExtension(*s.fes_s, *s.fes_f, kRc);
  slip.SetFluidExtension(*Ef);
  slip.SetFluidGauge(mu_gauge, kEps);
  slip.SetConstraint(kTheta, kALIterations);

  const auto res = slip.SlipRigidPairResiduals();
  ASSERT_EQ(static_cast<int>(res.size()), 4);  // 2 translations, 2 rotations
  for (size_t i = 0; i < res.size(); i++) {
    std::cout << "slip rigid pair " << i << ": residual " << res[i] << "\n";
    EXPECT_LT(res[i], 5e-2);
  }

  // Broken-zeta organisation: the same rigid pairs must be near-null
  // under the FULL four-block operator, both penalties included (the
  // constraints vanish on rigid pairs: nu and b are physically radial
  // against tangential rotations, and translations have no jump).
  {
    Array<int> outer_attr({2, 3});
    SubMesh outer(SubMesh::CreateFromDomain(*s.parent, outer_attr));
    auto fes_zo = SubMeshDofInjection::MakeShadowSpace(*s.fes_zeta, outer);
    LinearQuasiStaticReferentialSelfGravitatingSlipProblem bz(
        s.fes_s.get(), s.fes_f.get(), s.fes_zeta.get(), s.bg->Rheology(),
        s.bg->Density(), s.bg->Pressure(), interface, kG, kDtNDegree);
    bz.SetPrescribedVacuumExtension(*s.fes_buffer, *Evac);
    bz.SetFluidGauge(mu_gauge, kEps);
    bz.SetConstraint(kTheta, kALIterations);
    bz.EnableBrokenZeta(fes_zo.get(), kTheta);
    const auto res_bz = bz.SlipRigidPairResiduals();
    ASSERT_EQ(static_cast<int>(res_bz.size()), 4);
    for (size_t i = 0; i < res_bz.size(); i++) {
      std::cout << "broken-zeta rigid pair " << i << ": residual "
                << res_bz[i] << "\n";
      EXPECT_LT(res_bz[i], 5e-2);
    }
  }
}

// The broken-zeta solver on a relabelled two-layer background (the
// relabelled-equilibrium self-benchmark of doc/mappings.md, "Verification",
// with a slipping interface; mapped slip problems use broken zeta, the
// single-valued organisation refuses non-identity maps): xi = f(r) Rot(alpha(r)) x fixes the centre, the
// interface radius and the surface, with a twist across Sigma, so the
// referential mesh describes the SAME physical two-layer body while
// F_e is genuinely non-radial on the interface. Every mapped piece of
// the broken organisation acts: the per-region a-form machinery and
// Poisson blocks, the couplings, B_Sigma, G_Sigma, b = F^{-T} grad
// zeta0, and the scalar-jump constraint. The solution must reproduce
// the identity-description broken solution under composition,
// u~(x) = u(xi(x)), zeta~ = zeta o xi (modulo the 2-D constant), on
// the solid region, at the fixed mesh's geometric-interpolation floor
// (the exact-F-versus-interpolated-F effect of doc/mappings.md, "The
// discrete change-of-variables identity").
TEST(SlipProblem, BrokenZetaRelabelledEquilibrium) {
  const int order = 2;
  const int dim = 2;
  FunctionCoefficient sigma(SurfaceSigma);
  ConstantCoefficient mu_gauge(kKappa);
  Array<int> outer_attr({2, 3});

  // Reference: the broken solver at phi_e = id.
  Setting s(order);
  auto interface = RadialBdrMarker(*s.solid, 0.9 * kRc, 1.1 * kRc);
  auto surface_s = RadialBdrMarker(*s.solid, 0.9, 1.1);
  SubMesh outer(SubMesh::CreateFromDomain(*s.parent, outer_attr));
  auto fes_zo = SubMeshDofInjection::MakeShadowSpace(*s.fes_zeta, outer);
  auto Evac = NewRadialVacuumExtension(*s.fes_s, *s.fes_buffer, 1.0, s.r_out);
  LinearQuasiStaticReferentialSelfGravitatingSlipProblem ref(
      s.fes_s.get(), s.fes_f.get(), s.fes_zeta.get(), s.bg->Rheology(),
      s.bg->Density(), s.bg->Pressure(), interface, kG, kDtNDegree);
  ref.SetPrescribedVacuumExtension(*s.fes_buffer, *Evac);
  ref.SetFluidGauge(mu_gauge, kEps);
  ref.SetConstraint(kTheta, kALIterations);
  ref.EnableBrokenZeta(fes_zo.get(), kTheta);
  ref.SetSurfaceLoad(sigma, surface_s);
  ref.SetRelTol(1e-10);
  ref.AssembleForce(0.0);
  ASSERT_TRUE(ref.Solve());

  // Relabelled: phi_e = xi, coefficients through the generator chain.
  Setting s2(order);
  auto interface2 = RadialBdrMarker(*s2.solid, 0.9 * kRc, 1.1 * kRc);
  auto surface2 = RadialBdrMarker(*s2.solid, 0.9, 1.1);
  SubMesh outer2(SubMesh::CreateFromDomain(*s2.parent, outer_attr));
  auto fes_zo2 = SubMeshDofInjection::MakeShadowSpace(*s2.fes_zeta, outer2);
  auto Evac2 =
      NewRadialVacuumExtension(*s2.fes_s, *s2.fes_buffer, 1.0, s2.r_out);
  SlipRelabelling xi_def;
  CallableDiffeomorphism xi(
      dim, [&](const Vector& x, Vector& y) { xi_def.Map(x, y); },
      [&](const Vector& x, DenseMatrix& F) { xi_def.Grad(x, F); });
  RelabelledBackground rel_bg(*s2.bg, xi);
  LinearQuasiStaticReferentialSelfGravitatingSlipProblem rel(
      s2.fes_s.get(), s2.fes_f.get(), s2.fes_zeta.get(), rel_bg.Rheology(),
      rel_bg.Density(), rel_bg.Pressure(), interface2, kG, kDtNDegree);
  rel.SetPrescribedVacuumExtension(*s2.fes_buffer, *Evac2);
  rel.SetFluidGauge(mu_gauge, kEps);
  rel.SetConstraint(kTheta, kALIterations);
  rel.EnableBrokenZeta(fes_zo2.get(), kTheta);
  FunctionCoefficient sigma2(SurfaceSigma);
  rel.SetSurfaceLoad(sigma2, surface2);
  rel.SetRelTol(1e-10);
  rel.AssembleForce(0.0);
  ASSERT_TRUE(rel.Solve());

  // The mapped rigid pairs stay near-null under the mapped four-block
  // operator.
  {
    const auto res = rel.SlipRigidPairResiduals();
    ASSERT_EQ(static_cast<int>(res.size()), 4);
    for (size_t i = 0; i < res.size(); i++) {
      std::cout << "relabelled broken rigid pair " << i << ": residual "
                << res[i] << "\n";
      EXPECT_LT(res[i], 5e-2);
    }
  }

  // Compare at solid sample points: u~(x) = u(xi(x)), and the
  // potential through the same composition modulo the 2-D constant.
  double du2 = 0.0, un2 = 0.0, zn2 = 0.0;
  std::vector<double> dz;
  Vector x(dim), y(dim);
  int n_pts = 0;
  for (int i = 0; i < 48; i++) {
    const double r = kRc + 0.05 + (0.93 - kRc - 0.05) * (i % 8) / 7.0;
    const double th = 2.0 * std::numbers::pi * i / 48.0 + 0.1;
    x(0) = r * std::cos(th);
    x(1) = r * std::sin(th);
    xi_def.Map(x, y);
    Vector ur, urel, zr, zrel;
    if (!EvalAt(rel.Displacement(), *s2.solid, x, urel) ||
        !EvalAt(ref.Displacement(), *s.solid, y, ur) ||
        !EvalAt(rel.Potential(), *s2.parent, x, zrel) ||
        !EvalAt(ref.Potential(), *s.parent, y, zr)) {
      continue;
    }
    n_pts++;
    for (int d = 0; d < dim; d++) {
      du2 += (urel(d) - ur(d)) * (urel(d) - ur(d));
      un2 += ur(d) * ur(d);
    }
    dz.push_back(zrel(0) - zr(0));
    zn2 += zr(0) * zr(0);
  }
  ASSERT_GT(n_pts, 36);
  double dz_mean = 0.0;
  for (double v : dz) {
    dz_mean += v;
  }
  dz_mean /= dz.size();
  double dz2 = 0.0;
  for (double v : dz) {
    dz2 += (v - dz_mean) * (v - dz_mean);
  }
  const double u_err = std::sqrt(du2 / un2);
  const double z_err = std::sqrt(dz2 / zn2);
  std::cout << "relabelled broken-zeta: u " << u_err << ", zeta " << z_err
            << "\n";
  // Measured 5e-4 / 2e-4: two orders below the ~2-5% map amplitude,
  // so the mapped assembly is load-bearing, not trivially passing.
  EXPECT_LT(u_err, 5e-3);
  EXPECT_LT(z_err, 5e-3);
}
