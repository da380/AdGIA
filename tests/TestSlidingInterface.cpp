#include "TestCommon.hpp"

/*
  Tests for the sliding fluid-solid interface built from the SubMesh dof
  pairing J = Pi_s^T Pi_f (NewSubMeshPairingMatrix; doc/slip_interface.tex,
  "Discretisation of the slipping interface"):
  two displacement fields on the solid and fluid SubMeshes of one parent,
  coupled by a penalty on the normal jump assembled from ONE
  BoundaryNormalNormalIntegrator matrix B on the solid side,

      theta [ B, -B J; -J^T B, J^T B J ],

  with augmented-Lagrangian iterations (w <- w + theta P U) driving the
  normal jump to zero at moderate theta, interleaved with the gauge
  (eps Q) refinement of the fluid.

  On the two-layer disc (fluid core, solid mantle), gravity-free:

  - Pairing identities: J maps the fluid trace to the solid trace nodally
    (machine precision for a common interpolated field) and J J^T is the
    identity on the paired solid dofs (the pairing is one-to-one).
  - The sliding solution matches the condensed rank-one cavity reference on
    the solid, modulo the rigid modes (in the barotropic/neutral setting
    the sliding interface, the continuous-space gauge of TestFluidGauge and
    the condensed problem must all agree).
  - The normal-jump energy contracts with the AL iterations; the tangential
    jump stays free (orders of magnitude above the normal jump under an
    angular load) -- the slip the continuous space cannot represent.

  The null space of the sliding system is larger than the welded one:
  common translations, and *independent* rotations of shell and core (a
  frictionless circular interface transmits no torque); all are projected.
*/

namespace {

constexpr double kKappaS = 2.0;
constexpr double kMuS = 1.0;
constexpr double kKappaF = 1.0;
constexpr double kP0 = 0.01;
constexpr double kP2 = 0.01;
constexpr double kEps = 1.0e-2;
constexpr double kTheta = 1.0e2;
constexpr int kIterations = 8;
constexpr double kRCmb = 3483.0 / 6371.0;

void PressureTraction(const Vector& x, Vector& f) {
  const double r = x.Norml2();
  const double c = x[1] / r;
  f = x;
  f *= -(kP0 + kP2 * (2.0 * c * c - 1.0)) / r;
}

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

// The two-field setting: parent, solid and fluid SubMeshes, shadow vector
// spaces (shared collection, as the injections require) and the pairing.
struct Setting {
  std::unique_ptr<Mesh> parent;
  std::unique_ptr<SubMesh> solid, fluid;
  std::unique_ptr<H1_FECollection> fec;
  std::unique_ptr<FiniteElementSpace> fes_parent;
  std::unique_ptr<FiniteElementSpace> fes_s, fes_f;
  std::unique_ptr<SubMeshDofInjection> inj_s, inj_f;
  std::unique_ptr<SparseMatrix> J;  // solid vdofs x fluid vdofs

  explicit Setting(int order = 2) {
    parent = std::make_unique<Mesh>("../data/elastogravity_two_layer_2d.msh",
                                    1, 1);
    const int dim = parent->Dimension();
    Array<int> fluid_attr({1}), solid_attr({2});
    solid = std::make_unique<SubMesh>(
        SubMesh::CreateFromDomain(*parent, solid_attr));
    fluid = std::make_unique<SubMesh>(
        SubMesh::CreateFromDomain(*parent, fluid_attr));
    fec = std::make_unique<H1_FECollection>(order, dim);
    fes_parent =
        std::make_unique<FiniteElementSpace>(parent.get(), fec.get(), dim);
    fes_s = SubMeshDofInjection::MakeShadowSpace(*fes_parent, *solid);
    fes_f = SubMeshDofInjection::MakeShadowSpace(*fes_parent, *fluid);
    inj_s = std::make_unique<SubMeshDofInjection>(*fes_s, *fes_parent);
    inj_f = std::make_unique<SubMeshDofInjection>(*fes_f, *fes_parent);
    J = NewSubMeshPairingMatrix(*inj_s, *inj_f);
  }
};

TEST(SlidingInterface, PairingIdentities) {
  Setting s;
  const auto& J = *s.J;
  EXPECT_EQ(J.Height(), s.fes_s->GetVSize());
  EXPECT_EQ(J.Width(), s.fes_f->GetVSize());
  EXPECT_GT(J.NumNonZeroElems(), 0);

  // A common smooth field interpolates to equal traces: J carries the
  // fluid values onto the solid's interface dofs exactly.
  VectorFunctionCoefficient f(2, [](const Vector& x, Vector& v) {
    v.SetSize(2);
    v[0] = std::sin(x[0]) + x[1] * x[1];
    v[1] = std::cos(x[1]) - 2.0 * x[0];
  });
  GridFunction u_s(s.fes_s.get()), u_f(s.fes_f.get());
  u_s.ProjectCoefficient(f);
  u_f.ProjectCoefficient(f);
  Vector Ju(J.Height());
  J.Mult(u_f, Ju);
  int paired = 0;
  for (int i = 0; i < J.Height(); i++) {
    if (J.RowSize(i) > 0) {
      paired++;
      EXPECT_NEAR(Ju[i], u_s[i], 1e-13);
    }
  }
  EXPECT_GT(paired, 0);

  // The pairing is one-to-one: J J^T is the identity on the paired solid
  // dofs.
  std::unique_ptr<SparseMatrix> Jt(Transpose(J));
  std::unique_ptr<SparseMatrix> JJt(mfem::Mult(J, *Jt));
  for (int i = 0; i < J.Height(); i++) {
    if (J.RowSize(i) > 0) {
      EXPECT_NEAR((*JJt)(i, i), 1.0, 1e-14);
      EXPECT_EQ(JJt->RowSize(i), 1);
    }
  }
}

TEST(SlidingInterface, CavityMatchesCondensedWithFreeSlip) {
  Setting s;
  const int dim = 2;
  const int ns = s.fes_s->GetVSize(), nf = s.fes_f->GetVSize();

  // Solid and fluid stiffness (the fluid with the gauge shear eps mu_g).
  ConstantCoefficient kappa_s(kKappaS), mu_s(kMuS), kappa_f(kKappaF);
  ConstantCoefficient eps_mu(kEps * kKappaF), one(1.0);
  BilinearForm a_s(s.fes_s.get());
  a_s.AddDomainIntegrator(new ElasticityIntegrator(kappa_s, 1.0, 0.0));
  a_s.AddDomainIntegrator(new ElasticityIntegrator(mu_s, -2.0 / dim, 1.0));
  a_s.Assemble();
  a_s.Finalize();
  BilinearForm a_f(s.fes_f.get());
  a_f.AddDomainIntegrator(new ElasticityIntegrator(kappa_f, 1.0, 0.0));
  a_f.AddDomainIntegrator(new ElasticityIntegrator(eps_mu, -2.0 / dim, 1.0));
  a_f.Assemble();
  a_f.Finalize();
  // The eps Q penalty alone, for the gauge-refinement residuals.
  BilinearForm q_f(s.fes_f.get());
  q_f.AddDomainIntegrator(new ElasticityIntegrator(eps_mu, -2.0 / dim, 1.0));
  q_f.Assemble();
  q_f.Finalize();

  // Interface forms on the SOLID boundary elements alone: B for the normal
  // jump, M for the full jump (the tangential part is their difference).
  auto cmb = RadialBdrMarker(*s.solid, 0.9 * kRCmb, 1.1 * kRCmb);
  BilinearForm b_form(s.fes_s.get());
  b_form.AddBoundaryIntegrator(new BoundaryNormalNormalIntegrator(one), cmb);
  b_form.Assemble();
  b_form.Finalize();
  const SparseMatrix& B = b_form.SpMat();
  BilinearForm m_form(s.fes_s.get());
  m_form.AddBoundaryIntegrator(new VectorMassIntegrator(one), cmb);
  m_form.Assemble();
  m_form.Finalize();

  // Penalty blocks from B and J: [B, -BJ; -J^T B, J^T B J].
  const SparseMatrix& J = *s.J;
  std::unique_ptr<SparseMatrix> Jt(Transpose(J));
  std::unique_ptr<SparseMatrix> BJ(mfem::Mult(B, J));
  std::unique_ptr<SparseMatrix> JtB(mfem::Mult(*Jt, B));
  std::unique_ptr<SparseMatrix> JtBJ(mfem::Mult(*JtB, J));
  std::unique_ptr<SparseMatrix> mBJ(new SparseMatrix(*BJ));
  *mBJ *= -1.0;
  std::unique_ptr<SparseMatrix> mJtB(new SparseMatrix(*JtB));
  *mJtB *= -1.0;

  // The regularised operator A + theta P as one monolithic matrix.
  Array<int> offsets({0, ns, nf});
  offsets.PartialSum();
  BlockMatrix blocks(offsets);
  std::unique_ptr<SparseMatrix> A00(
      Add(1.0, a_s.SpMat(), kTheta, B));
  std::unique_ptr<SparseMatrix> A11(
      Add(1.0, a_f.SpMat(), kTheta, *JtBJ));
  std::unique_ptr<SparseMatrix> A01(new SparseMatrix(*mBJ));
  *A01 *= kTheta;
  std::unique_ptr<SparseMatrix> A10(new SparseMatrix(*mJtB));
  *A10 *= kTheta;
  blocks.SetBlock(0, 0, A00.get());
  blocks.SetBlock(0, 1, A01.get());
  blocks.SetBlock(1, 0, A10.get());
  blocks.SetBlock(1, 1, A11.get());
  std::unique_ptr<SparseMatrix> R(blocks.CreateMonolithic());

  // Null space: common translations, and independent rotations of the
  // shell and of the core (the frictionless circular interface transmits
  // no torque).
  NullSpaceProjector P;
  {
    BlockVector n(offsets);
    GridFunction gs(s.fes_s.get()), gf(s.fes_f.get());
    for (int c = 0; c < dim; c++) {
      Vector e(dim);
      e = 0.0;
      e[c] = 1.0;
      VectorConstantCoefficient t(e);
      gs.ProjectCoefficient(t);
      gf.ProjectCoefficient(t);
      n.GetBlock(0) = gs;
      n.GetBlock(1) = gf;
      P.Add(n);
    }
    RigidRotation rot(dim, 2);
    gs.ProjectCoefficient(rot);
    gf.ProjectCoefficient(rot);
    n.GetBlock(0) = gs;
    n.GetBlock(1) = 0.0;
    P.Add(n);
    n.GetBlock(0) = 0.0;
    n.GetBlock(1) = gf;
    P.Add(n);
  }
  EXPECT_EQ(P.Size(), 4);

  // Pressure load on the solid's outer surface.
  auto surf = RadialBdrMarker(*s.solid, 0.9, 1.1);
  VectorFunctionCoefficient traction(dim, PressureTraction);
  LinearForm lf(s.fes_s.get());
  lf.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(traction), surf);
  lf.Assemble();
  BlockVector F(offsets);
  F.GetBlock(0) = lf;
  F.GetBlock(1) = 0.0;

  // Solve with augmented-Lagrangian iterations for the constraint and
  // Tikhonov refinement for the gauge in one loop:
  //   R U_{k+1} = f - w_k + eps Q u_f,k ;  w_{k+1} = w_k + theta P U_{k+1}.
  // At the fixed point (A + theta P) U = f - w with P U -> 0, and the eps
  // shear drops out of the observables.
  GSSmoother gs_prec(*R);
  ProjectedSolver prec_p(P);
  prec_p.SetSolver(gs_prec);
  ProjectedOperator op_p(*R, P);
  CGSolver cg;
  cg.SetOperator(op_p);
  cg.SetPreconditioner(prec_p);
  cg.SetRelTol(1e-12);
  cg.SetAbsTol(0.0);
  cg.SetMaxIter(20000);
  cg.iterative_mode = true;
  ProjectedSolver solver(P);
  solver.SetSolver(cg);
  solver.iterative_mode = true;

  BlockVector U(offsets), rhs(offsets), w(offsets), t(offsets);
  U = 0.0;
  w = 0.0;
  auto normal_jump2 = [&](const BlockVector& X) {
    // U^T P U = |m.(u_s - J u_f)|^2 on the interface.
    Vector js(ns);
    J.Mult(X.GetBlock(1), js);
    js -= X.GetBlock(0);
    Vector Bj(ns);
    B.Mult(js, Bj);
    return InnerProduct(js, Bj);
  };
  std::vector<double> jumps;
  for (int k = 0; k < kIterations; k++) {
    rhs = F;
    rhs -= w;
    Vector qf(nf);
    q_f.SpMat().Mult(U.GetBlock(1), qf);
    rhs.GetBlock(1) += qf;
    solver.Mult(rhs, U);
    EXPECT_TRUE(cg.GetConverged());
    // w += theta P U.
    Vector js(ns);
    J.Mult(U.GetBlock(1), js);
    js -= U.GetBlock(0);  // -(u_s - J u_f)
    Vector Bj(ns);
    B.Mult(js, Bj);
    t = 0.0;
    t.GetBlock(0) -= Bj;  // theta B (u_s - J u_f) row
    Jt->AddMult(Bj, t.GetBlock(1));
    t *= kTheta;
    w += t;
    jumps.push_back(std::sqrt(normal_jump2(U)));
  }

  // The normal jump contracts with the AL iterations down to the floor
  // set by the interleaved gauge source, and ends far below the normal
  // trace itself (the meaningful, theta-independent statement).
  ASSERT_EQ(static_cast<int>(jumps.size()), kIterations);
  EXPECT_LT(jumps[2] / jumps[0], 0.5);
  {
    Vector Bu(ns);
    B.Mult(U.GetBlock(0), Bu);
    const double trace = std::sqrt(InnerProduct(U.GetBlock(0), Bu));
    EXPECT_LT(jumps.back(), 1e-4 * trace);
    std::cout << "normal jump / normal trace: " << jumps.back() / trace
              << ", decay per AL iteration: "
              << jumps[kIterations - 1] / jumps[kIterations - 2] << "\n";
  }

  // The tangential jump stays free: orders of magnitude above the normal
  // jump under the angular load.
  {
    Vector js(ns);
    J.Mult(U.GetBlock(1), js);
    js -= U.GetBlock(0);
    Vector Mj(ns);
    m_form.SpMat().Mult(js, Mj);
    const double full2 = InnerProduct(js, Mj);
    const double normal2 = normal_jump2(U);
    EXPECT_GT(full2 - normal2, 1e2 * normal2);
  }

  // Condensed rank-one reference on the same solid space (as in
  // TestFluidGauge), compared modulo the solid-space rigid modes.
  {
    IsotropicElasticRheology rheology(dim, kappa_s, mu_s);
    LinearQuasiStaticTractionProblem cond(s.fes_s.get(), rheology, traction,
                                          surf);
    ConstantCoefficient one_c(1.0);
    LinearForm b_lf(s.fes_s.get());
    b_lf.AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(one_c),
                               cmb);
    b_lf.Assemble();
    double V = 0.0;
    for (int i = 0; i < s.fluid->GetNE(); i++) {
      V += s.fluid->GetElementVolume(i);
    }
    const double sigma = kKappaF / V;
    cond.AssembleForce(0.0);
    EXPECT_TRUE(cond.Solve());
    GridFunction u_a(cond.Displacement());
    cond.AssembleForce(0.0);
    Vector delta(b_lf);
    delta -= cond.ExternalLoad();
    cond.AddForce(delta);
    EXPECT_TRUE(cond.Solve());
    GridFunction u_b(cond.Displacement());
    GridFunction u_cond(u_a);
    u_cond.Add(-sigma * b_lf(u_a) / (1.0 + sigma * b_lf(u_b)), u_b);

    Vector d(U.GetBlock(0));
    d -= u_cond;
    auto proj = MakeRigidModeProjector(*s.fes_s);
    proj->Project(d);
    Vector uc(u_cond);
    EXPECT_LT(d.Norml2() / uc.Norml2(), 5.0e-3);
  }
}

}  // namespace
