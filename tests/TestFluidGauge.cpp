#include "TestCommon.hpp"

/*
  Tests for the gauged-fluid option of LinearQuasiStaticProblemBase on
  purely elastic bodies (no gravity), where the exact statics of the fluid
  is known independently: a static, gravity-free fluid supports a uniform
  pressure, so its entire response condenses onto the solid as the rank-one
  cavity term (1/K_V) b(u) b(v), b(v) = int_Sigma v.m dS, K_V = V/kappa_f.
  The method is described in doc/gauged_fluid.md ("Gauge fixing: penalty
  plus iterated refinement", "Verification").

  Meshes: the canned two-layer disc (attribute 1 fluid core, 2 mantle) and
  the three-layer ball with the inner and outer core merged into one fluid
  (attributes 1+2 fluid, 3 mantle), both with the buffer excluded from the
  body SubMesh. Order-2 geometry and displacements.

  - Uniform external pressure against the exact Lame solution with a fluid
    core (the fluid's own displacement is gamma x: conformal, so it is also
    the Q-minimal gauge and can be compared). The mass-weighted rigid
    gauge is used, since the exact solution has zero net momentum.
  - A degree-2 pressure against the condensed rank-one problem on the
    solid-only SubMesh (Sherman-Morrison with two projected solves),
    modulo the two problems' different rigid gauges.
  - The solid observables are independent of epsilon after refinement,
    while the refinement residuals contract by O(epsilon mu_g / mu_s).
  - The regularised matrix stays symmetric.
*/

namespace {

constexpr double kKappaS = 2.0;
constexpr double kMuS = 1.0;
constexpr double kKappaF = 1.0;
constexpr double kP0 = 0.01;
constexpr double kP2 = 0.01;
constexpr double kEps = 1.0e-2;
constexpr int kRefine = 3;

std::string BodyMeshFile(int dim) {
  return dim == 2 ? "../data/elastogravity_two_layer_2d.msh"
                  : "../data/elastogravity_three_layer_3d.msh";
}

// Radius of the boundary elements carrying @p attr on @p mesh.
double BdrRadius(Mesh& mesh, int attr) {
  const int dim = mesh.Dimension();
  for (int i = 0; i < mesh.GetNBE(); i++) {
    if (mesh.GetBdrAttribute(i) != attr) {
      continue;
    }
    auto* tr = mesh.GetBdrElementTransformation(i);
    Vector c(dim);
    tr->Transform(Geometries.GetCenter(mesh.GetBdrElementGeometry(i)), c);
    return c.Norml2();
  }
  return -1.0;
}

// Marker for the boundary attributes of @p mesh whose elements lie at
// radius in (r_min, r_max).
Array<int> RadialBdrMarker(Mesh& mesh, double r_min, double r_max) {
  Array<int> marker(mesh.bdr_attributes.Max());
  marker = 0;
  for (int a = 1; a <= mesh.bdr_attributes.Max(); a++) {
    const double r = BdrRadius(mesh, a);
    if (r > r_min && r < r_max) {
      marker[a - 1] = 1;
    }
  }
  return marker;
}

// External pressure P(x) = P0 + P2 (degree-2 pattern); minus P times the
// outward unit normal of the unit sphere/circle.
void PressureTraction(const Vector& x, Vector& f) {
  const double r = x.Norml2();
  const double c = (x.Size() == 2 ? x[1] : x[2]) / r;
  const double P =
      kP0 + kP2 * (x.Size() == 2 ? 2.0 * c * c - 1.0 : 3.0 * c * c - 1.0);
  f = x;
  f *= -P / r;
}

void UniformPressureTraction(const Vector& x, Vector& f) {
  const double r = x.Norml2();
  f = x;
  f *= -kP0 / r;
}

// Exact solution for uniform pressure kP0 on r = b with a fluid core of
// radius a: u = (alpha + beta / r^d) x in the solid, gamma x in the fluid.
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

// The body (fluid + solid, buffer excluded), its displacement space and
// the piecewise material of the tests.
struct Case {
  std::unique_ptr<Mesh> parent;
  std::unique_ptr<SubMesh> body;
  std::unique_ptr<H1_FECollection> fec;
  std::unique_ptr<FiniteElementSpace> fes;
  Vector kappa_vals, mu_vals;
  std::unique_ptr<PWConstCoefficient> kappa, mu;
  std::unique_ptr<IsotropicElasticRheology> rheology;
  ConstantCoefficient mu_gauge{kKappaF};
  Array<int> fluid, surface;
  double r_cmb;

  explicit Case(int dim, int order = 2) {
    parent = std::make_unique<Mesh>(BodyMeshFile(dim).c_str(), 1, 1);
    EXPECT_EQ(parent->Dimension(), dim);
    const int n_layers = parent->attributes.Max() - 1;  // buffer excluded
    Array<int> attrs(n_layers);
    for (int i = 0; i < n_layers; i++) {
      attrs[i] = i + 1;
    }
    body = std::make_unique<SubMesh>(SubMesh::CreateFromDomain(*parent, attrs));
    fec = std::make_unique<H1_FECollection>(order, dim);
    fes = std::make_unique<FiniteElementSpace>(body.get(), fec.get(), dim);

    // Fluid: attribute 1 (2-D) or 1+2 (3-D, both cores); solid: the mantle.
    kappa_vals.SetSize(n_layers);
    mu_vals.SetSize(n_layers);
    fluid.SetSize(n_layers);
    kappa_vals = kKappaF;
    mu_vals = 0.0;
    fluid = 1;
    kappa_vals(n_layers - 1) = kKappaS;
    mu_vals(n_layers - 1) = kMuS;
    fluid[n_layers - 1] = 0;
    kappa = std::make_unique<PWConstCoefficient>(kappa_vals);
    mu = std::make_unique<PWConstCoefficient>(mu_vals);
    rheology = std::make_unique<IsotropicElasticRheology>(dim, *kappa, *mu);

    surface = RadialBdrMarker(*body, 0.9, 1.1);
    r_cmb = 3483.0 / 6371.0;
  }
};

double RelL2Error(GridFunction& u, VectorCoefficient& exact) {
  Vector zero(u.FESpace()->GetVDim());
  zero = 0.0;
  VectorConstantCoefficient z(zero);
  return u.ComputeL2Error(exact) / std::max(1e-30, u.ComputeL2Error(z));
}

TEST(FluidGauge, UniformPressureMatchesLame2D) {
  Case c(2);
  VectorFunctionCoefficient traction(2, UniformPressureTraction);
  LinearQuasiStaticTractionProblem prob(c.fes.get(), *c.rheology, traction,
                                        c.surface);
  prob.SetGaugedFluid(c.fluid, c.mu_gauge, kEps, kRefine);
  // The exact radial solution has zero net momentum; the Euclidean
  // true-dof gauge differs from it by a mesh-asymmetry rigid translation.
  prob.SetMassWeightedGauge();
  prob.AssembleForce(0.0);
  EXPECT_TRUE(prob.Solve());

  LameSolution lame(2, c.r_cmb, 1.0);
  VectorFunctionCoefficient exact(
      2, [&lame](const Vector& x, Vector& u) { lame.Eval(x, u); });
  auto u = prob.Displacement();
  EXPECT_LT(RelL2Error(u, exact), 1.0e-4);
}

TEST(FluidGauge, UniformPressureMatchesLame3D) {
  Case c(3);
  VectorFunctionCoefficient traction(3, UniformPressureTraction);
  LinearQuasiStaticTractionProblem prob(c.fes.get(), *c.rheology, traction,
                                        c.surface);
  prob.SetGaugedFluid(c.fluid, c.mu_gauge, kEps, kRefine);
  prob.SetMassWeightedGauge();
  prob.AssembleForce(0.0);
  EXPECT_TRUE(prob.Solve());

  LameSolution lame(3, c.r_cmb, 1.0);
  VectorFunctionCoefficient exact(
      3, [&lame](const Vector& x, Vector& u) { lame.Eval(x, u); });
  auto u = prob.Displacement();
  EXPECT_LT(RelL2Error(u, exact), 1.0e-3);
}

// Restrict a displacement on the body SubMesh to a GridFunction on the
// solid-only SubMesh, through the shared parent.
void RestrictToSolid(const GridFunction& u_body, Mesh& parent,
                     FiniteElementSpace& fes_parent, GridFunction& u_solid) {
  GridFunction on_parent(&fes_parent);
  on_parent = 0.0;
  SubMesh::Transfer(u_body, on_parent);
  SubMesh::Transfer(on_parent, u_solid);
}

TEST(FluidGauge, MatchesCondensedCavity2D) {
  Case c(2);
  VectorFunctionCoefficient traction(2, PressureTraction);

  // Gauged problem on the body.
  LinearQuasiStaticTractionProblem gauged(c.fes.get(), *c.rheology, traction,
                                          c.surface);
  gauged.SetGaugedFluid(c.fluid, c.mu_gauge, kEps, kRefine);
  gauged.AssembleForce(0.0);
  EXPECT_TRUE(gauged.Solve());

  // Condensed problem on the solid alone: A_s + (1/K_V) b b^T by
  // Sherman-Morrison with two projected solves.
  Array<int> solid_attr({c.body->attributes.Max()});
  auto solid = std::make_unique<SubMesh>(
      SubMesh::CreateFromDomain(*c.parent, solid_attr));
  FiniteElementSpace fes_s(solid.get(), c.fec.get(), 2);
  ConstantCoefficient kappa_s(kKappaS), mu_s(kMuS);
  IsotropicElasticRheology solid_rheology(2, kappa_s, mu_s);
  auto surf_s = RadialBdrMarker(*solid, 0.9, 1.1);
  auto cmb_s = RadialBdrMarker(*solid, 0.9 * c.r_cmb, 1.1 * c.r_cmb);

  LinearQuasiStaticTractionProblem cond(&fes_s, solid_rheology, traction,
                                        surf_s);
  ConstantCoefficient one(1.0);
  LinearForm b_lf(&fes_s);
  b_lf.AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(one), cmb_s);
  b_lf.Assemble();

  double V = 0.0;
  for (int i = 0; i < c.body->GetNE(); i++) {
    if (c.fluid[c.body->GetAttribute(i) - 1]) {
      V += c.body->GetElementVolume(i);
    }
  }
  const double sigma = kKappaF / V;  // 1 / K_V

  cond.AssembleForce(0.0);
  EXPECT_TRUE(cond.Solve());
  GridFunction u_f(cond.Displacement());  // A^{-1} f

  cond.AssembleForce(0.0);
  Vector delta(b_lf);
  delta -= cond.ExternalLoad();
  cond.AddForce(delta);
  EXPECT_TRUE(cond.Solve());
  GridFunction u_b(cond.Displacement());  // A^{-1} b

  GridFunction u_cond(u_f);
  u_cond.Add(-sigma * b_lf(u_f) / (1.0 + sigma * b_lf(u_b)), u_b);

  // Compare on the solid, modulo the two problems' rigid gauges.
  FiniteElementSpace fes_parent(c.parent.get(), c.fec.get(), 2);
  GridFunction u_gauged_s(&fes_s);
  RestrictToSolid(gauged.Displacement(), *c.parent, fes_parent, u_gauged_s);

  GridFunction diff(u_gauged_s);
  diff -= u_cond;
  Vector d_true;
  diff.GetTrueDofs(d_true);
  auto projector = MakeRigidModeProjector(fes_s);
  projector->Project(d_true);
  Vector u_true;
  u_cond.GetTrueDofs(u_true);
  EXPECT_LT(d_true.Norml2() / u_true.Norml2(), 2.0e-3);
}

TEST(FluidGauge, ObservablesIndependentOfEpsilon2D) {
  Case c(2);
  VectorFunctionCoefficient traction(2, PressureTraction);
  LinearQuasiStaticTractionProblem prob(c.fes.get(), *c.rheology, traction,
                                        c.surface);
  FiniteElementSpace fes_parent(c.parent.get(), c.fec.get(), 2);
  Array<int> solid_attr({c.body->attributes.Max()});
  auto solid = std::make_unique<SubMesh>(
      SubMesh::CreateFromDomain(*c.parent, solid_attr));
  FiniteElementSpace fes_s(solid.get(), c.fec.get(), 2);

  prob.SetGaugedFluid(c.fluid, c.mu_gauge, 1.0e-2, 5);
  prob.AssembleForce(0.0);
  EXPECT_TRUE(prob.Solve());
  GridFunction u1(&fes_s);
  RestrictToSolid(prob.Displacement(), *c.parent, fes_parent, u1);

  // The refinement residuals contract by O(eps mu_g / mu_s) per step.
  const auto& res = prob.GaugeResiduals();
  ASSERT_EQ(static_cast<int>(res.size()), 5);
  EXPECT_LT(res[1] / res[0], 0.2);
  EXPECT_LT(res[2] / res[1], 0.2);

  prob.SetGaugeEpsilon(1.0e-3);
  prob.SetGaugeRefinements(kRefine);
  prob.AssembleForce(0.0);
  EXPECT_TRUE(prob.Solve());
  GridFunction u2(&fes_s);
  RestrictToSolid(prob.Displacement(), *c.parent, fes_parent, u2);

  GridFunction diff(u1);
  diff -= u2;
  Vector d_true, u_true;
  diff.GetTrueDofs(d_true);
  u2.GetTrueDofs(u_true);
  EXPECT_LT(d_true.Norml2() / u_true.Norml2(), 1.0e-4);
}

// The Maxwell (secular) relaxation mode: on a gravity-free body the
// relaxed Maxwell core IS the static fluid (no stratification, so no
// band), and the converged solve must reproduce the exact Lame
// solutions through time-stepping alone. The comparison is restricted
// to the SOLID: the fluid displacement is relaxation history (the
// class's stopping metric makes the same restriction), where the
// gauged formulation's refined fluid field happens to be comparable.
TEST(FluidGauge, MaxwellRelaxationMatchesLame2D) {
  Case c(2);
  VectorFunctionCoefficient traction(2, UniformPressureTraction);
  LinearQuasiStaticTractionProblem prob(c.fes.get(), *c.rheology, traction,
                                        c.surface);
  ConstantCoefficient mu_core(kMuS);
  prob.SetFluid(c.fluid, mu_core);  // the default (Maxwell) treatment
  prob.SetMassWeightedGauge();
  prob.AssembleForce(0.0);
  EXPECT_TRUE(prob.Solve());
  const auto& rep = prob.MaxwellReport();
  EXPECT_LT(rep.delta_returned, 1.0e-6);
  EXPECT_LE(rep.steps, 100);

  LameSolution lame(2, c.r_cmb, 1.0);
  VectorFunctionCoefficient exact(
      2, [&lame](const Vector& x, Vector& u) { lame.Eval(x, u); });
  FiniteElementSpace fes_parent(c.parent.get(), c.fec.get(), 2);
  Array<int> solid_attr({c.body->attributes.Max()});
  auto solid = std::make_unique<SubMesh>(
      SubMesh::CreateFromDomain(*c.parent, solid_attr));
  FiniteElementSpace fes_s(solid.get(), c.fec.get(), 2);
  GridFunction u_s(&fes_s);
  RestrictToSolid(prob.Displacement(), *c.parent, fes_parent, u_s);
  EXPECT_LT(RelL2Error(u_s, exact), 1.0e-4);
}

TEST(FluidGauge, MaxwellRelaxationMatchesLame3D) {
  Case c(3);
  VectorFunctionCoefficient traction(3, UniformPressureTraction);
  LinearQuasiStaticTractionProblem prob(c.fes.get(), *c.rheology, traction,
                                        c.surface);
  ConstantCoefficient mu_core(kMuS);
  MaxwellRelaxationOptions opts;
  opts.beta_max = 1.0e3;  // shorten the escalation probe (test economy)
  opts.tol = 1.0e-6;      // the assertion level below (test economy)
  prob.SetMaxwellFluid(c.fluid, mu_core, opts);
  prob.SetMassWeightedGauge();
  prob.AssembleForce(0.0);
  EXPECT_TRUE(prob.Solve());
  EXPECT_LT(prob.MaxwellReport().delta_returned, 1.0e-5);

  LameSolution lame(3, c.r_cmb, 1.0);
  VectorFunctionCoefficient exact(
      3, [&lame](const Vector& x, Vector& u) { lame.Eval(x, u); });
  FiniteElementSpace fes_parent(c.parent.get(), c.fec.get(), 3);
  Array<int> solid_attr({c.body->attributes.Max()});
  auto solid = std::make_unique<SubMesh>(
      SubMesh::CreateFromDomain(*c.parent, solid_attr));
  FiniteElementSpace fes_s(solid.get(), c.fec.get(), 3);
  GridFunction u_s(&fes_s);
  RestrictToSolid(prob.Displacement(), *c.parent, fes_parent, u_s);
  EXPECT_LT(RelL2Error(u_s, exact), 1.0e-3);
}

// Under a non-conformal load the Maxwell limit and the gauged solve must
// agree on the SOLID (both compute the same physical response; the fluid
// displacement is gauge in one and relaxation history in the other).
TEST(FluidGauge, MaxwellMatchesGaugedOnSolid2D) {
  Case c(2);
  VectorFunctionCoefficient traction(2, PressureTraction);
  FiniteElementSpace fes_parent(c.parent.get(), c.fec.get(), 2);
  Array<int> solid_attr({c.body->attributes.Max()});
  auto solid = std::make_unique<SubMesh>(
      SubMesh::CreateFromDomain(*c.parent, solid_attr));
  FiniteElementSpace fes_s(solid.get(), c.fec.get(), 2);

  LinearQuasiStaticTractionProblem gauged(c.fes.get(), *c.rheology, traction,
                                          c.surface);
  gauged.SetGaugedFluid(c.fluid, c.mu_gauge, kEps, kRefine);
  gauged.AssembleForce(0.0);
  EXPECT_TRUE(gauged.Solve());
  GridFunction u_g(&fes_s);
  RestrictToSolid(gauged.Displacement(), *c.parent, fes_parent, u_g);

  LinearQuasiStaticTractionProblem maxwell(c.fes.get(), *c.rheology,
                                           traction, c.surface);
  ConstantCoefficient mu_core(kMuS);
  maxwell.SetMaxwellFluid(c.fluid, mu_core);
  maxwell.AssembleForce(0.0);
  EXPECT_TRUE(maxwell.Solve());
  GridFunction u_m(&fes_s);
  RestrictToSolid(maxwell.Displacement(), *c.parent, fes_parent, u_m);

  // Modulo the two solves' rigid gauges.
  GridFunction diff(u_g);
  diff -= u_m;
  Vector d_true, u_true;
  diff.GetTrueDofs(d_true);
  u_m.GetTrueDofs(u_true);
  auto projector = MakeRigidModeProjector(fes_s);
  projector->Project(d_true);
  // The two formulations agree to the gauged solve's O(eps^k) bias.
  EXPECT_LT(d_true.Norml2() / u_true.Norml2(), 3.0e-4);
}

TEST(FluidGauge, RegularizedMatrixIsSymmetric2D) {
  Case c(2);
  VectorFunctionCoefficient traction(2, PressureTraction);
  LinearQuasiStaticTractionProblem prob(c.fes.get(), *c.rheology, traction,
                                        c.surface);
  prob.SetGaugedFluid(c.fluid, c.mu_gauge, kEps, kRefine);
  const auto& A = prob.RegularizedMatrix();
  const auto* As = A.As<SparseMatrix>();
  std::unique_ptr<SparseMatrix> At(Transpose(*As));
  EXPECT_LT(MaxDiff(*As, *At), 1.0e-14 * As->MaxNorm());
}

}  // namespace
