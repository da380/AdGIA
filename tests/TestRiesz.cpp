#include <random>

#include "MixedProblemTestCommon.hpp"
#include "TestCommon.hpp"

/*
  Tests for the Riesz maps (riesz.hpp) on the 2-D two-layer disc with
  its buffer (the extension-domain shape they are built for): the L2
  map reproduces an exactly representable function from its dual; the
  Sobolev map's gradient satisfies the Helmholtz weak form and the
  Dirichlet condition on the outer boundary; the order-2 map equals the
  hand-composed chain A^-1 M A^-1; and the dual-gradient pairing is a
  positive quantity that matches its assembled value.
*/

namespace {

using namespace self_grav_test;

double Smooth(const Vector& x) { return 1.0 + x[0] + 0.5 * x[0] * x[1]; }

}  // namespace

TEST(Riesz, L2MapSolvesMassSystem) {
  Mesh mesh("../data/elastogravity_two_layer_2d.msh", 1, 1);
  L2_FECollection l2(2, 2);
  FiniteElementSpace fes(&mesh, &l2);

  FunctionCoefficient f(Smooth);
  LinearForm j(&fes);
  j.AddDomainIntegrator(new DomainLFIntegrator(f));
  j.Assemble();

  L2RieszMap riesz(fes);
  Vector g(fes.GetTrueVSize());
  riesz.Mult(j, g);

  // M g = j (the L2 identification), and the pairing is the squared
  // metric norm g^T M g.
  BilinearForm m(&fes);
  m.AddDomainIntegrator(new MassIntegrator());
  m.Assemble();
  m.Finalize();
  Vector Mg(g.Size());
  m.SpMat().Mult(g, Mg);
  Vector res(Mg);
  res -= j;
  EXPECT_LT(res.Norml2() / j.Norml2(), 1e-10);
  GridFunction gf(&fes);
  gf.SetFromTrueDofs(g);
  const double norm2 = m.InnerProduct(gf, gf);
  EXPECT_NEAR(riesz.Pair(j, g), norm2, 1e-10 * norm2);
}

TEST(Riesz, SobolevMapSatisfiesWeakFormAndDirichlet) {
  Mesh mesh("../data/elastogravity_two_layer_2d.msh", 1, 1);
  H1_FECollection h1(2, 2);
  FiniteElementSpace fes(&mesh, &h1);

  Array<int> outer(mesh.bdr_attributes.Max());
  outer = 0;
  outer[mesh.bdr_attributes.Max() - 1] = 1;  // the outer boundary

  const double alpha = 1.0, beta = 0.04;  // smoothing length 0.2
  SobolevRieszMap riesz(fes, alpha, beta, 1, &outer);

  FunctionCoefficient f(Smooth);
  LinearForm j(&fes);
  j.AddDomainIntegrator(new DomainLFIntegrator(f));
  j.Assemble();

  Vector g(fes.GetTrueVSize());
  riesz.Mult(j, g);

  // The weak form: (alpha M + beta K) g equals j on the free dofs.
  ConstantCoefficient ac(alpha), bc(beta);
  BilinearForm a(&fes);
  a.AddDomainIntegrator(new MassIntegrator(ac));
  a.AddDomainIntegrator(new DiffusionIntegrator(bc));
  a.Assemble();
  a.Finalize();
  Vector Ag(g.Size());
  a.SpMat().Mult(g, Ag);
  Array<int> ess;
  fes.GetEssentialTrueDofs(outer, ess);
  Vector res(Ag);
  res -= j;
  res.SetSubVector(ess, 0.0);
  EXPECT_LT(res.Norml2() / j.Norml2(), 1e-9);

  // Dirichlet on the outer boundary.
  for (int i = 0; i < ess.Size(); i++) {
    EXPECT_EQ(g[ess[i]], 0.0);
  }
}

TEST(Riesz, OrderTwoEqualsComposedChain) {
  Mesh mesh("../data/elastogravity_two_layer_2d.msh", 1, 1);
  H1_FECollection h1(2, 2);
  FiniteElementSpace fes(&mesh, &h1);

  Array<int> outer(mesh.bdr_attributes.Max());
  outer = 0;
  outer[mesh.bdr_attributes.Max() - 1] = 1;

  SobolevRieszMap s1(fes, 1.0, 0.04, 1, &outer);
  SobolevRieszMap s2(fes, 1.0, 0.04, 2, &outer);

  std::mt19937 gen(99);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  Vector j(fes.GetTrueVSize());
  for (int i = 0; i < j.Size(); i++) {
    j[i] = dist(gen);
  }

  Vector g1(j.Size()), g2(j.Size()), chain(j.Size());
  s1.Mult(j, g1);
  BilinearForm m(&fes);
  m.AddDomainIntegrator(new MassIntegrator());
  m.Assemble();
  m.Finalize();
  Vector Mg1(j.Size());
  m.SpMat().Mult(g1, Mg1);
  Array<int> ess;
  fes.GetEssentialTrueDofs(outer, ess);
  Mg1.SetSubVector(ess, 0.0);
  s1.Mult(Mg1, chain);
  s2.Mult(j, g2);

  Vector diff(g2);
  diff -= chain;
  EXPECT_LT(diff.Norml2() / chain.Norml2(), 1e-9);

  // The pairing with the gradient is the squared metric norm: positive,
  // and smaller for the smoother (higher-order) gradient of the same
  // rough dual.
  EXPECT_GT(s1.Pair(j, g1), 0.0);
  EXPECT_GT(s2.Pair(j, g2), 0.0);
  EXPECT_LT(s2.Pair(j, g2), s1.Pair(j, g1));
}

TEST(Riesz, VectorSpacesGetComponentwiseMetric) {
  Mesh mesh("../data/elastogravity_two_layer_2d.msh", 1, 1);
  H1_FECollection h1(2, 2);
  FiniteElementSpace fes(&mesh, &h1, 2);

  Array<int> outer(mesh.bdr_attributes.Max());
  outer = 0;
  outer[mesh.bdr_attributes.Max() - 1] = 1;

  const double alpha = 1.0, beta = 0.04;
  SobolevRieszMap riesz(fes, alpha, beta, 1, &outer);

  VectorFunctionCoefficient f(2, [](const Vector& x, Vector& v) {
    v[0] = 1.0 + x[0];
    v[1] = x[1] - 0.5 * x[0] * x[1];
  });
  LinearForm j(&fes);
  j.AddDomainIntegrator(new VectorDomainLFIntegrator(f));
  j.Assemble();

  Vector g(fes.GetTrueVSize());
  riesz.Mult(j, g);

  // The component-wise weak form: (alpha M + beta K) g = j on the free
  // dofs, with the vector integrators.
  ConstantCoefficient ac(alpha), bc(beta);
  BilinearForm a(&fes);
  a.AddDomainIntegrator(new VectorMassIntegrator(ac));
  a.AddDomainIntegrator(new VectorDiffusionIntegrator(bc));
  a.Assemble();
  a.Finalize();
  Vector Ag(g.Size());
  a.SpMat().Mult(g, Ag);
  Array<int> ess;
  fes.GetEssentialTrueDofs(outer, ess);
  Vector res(Ag);
  res -= j;
  res.SetSubVector(ess, 0.0);
  EXPECT_LT(res.Norml2() / j.Norml2(), 1e-9);
  for (int i = 0; i < ess.Size(); i++) {
    EXPECT_EQ(g[ess[i]], 0.0);
  }

  // The L2 map on the same vector space.
  L2RieszMap l2map(fes);
  Vector gl(fes.GetTrueVSize());
  l2map.Mult(j, gl);
  BilinearForm m(&fes);
  m.AddDomainIntegrator(new VectorMassIntegrator());
  m.Assemble();
  m.Finalize();
  Vector Mg(gl.Size());
  m.SpMat().Mult(gl, Mg);
  Vector resl(Mg);
  resl -= j;
  EXPECT_LT(resl.Norml2() / j.Norml2(), 1e-10);
}
