/*
  Tests for NonlinearSelfGravitatingProblem (nonlinear_problem.hpp), the
  static problem of a self-gravitating solid body at finite strain, on a
  homogeneous disc and ball in hydrostatic equilibrium inside a buffer
  shell.

  - Without a load the reference is the solution.
  - The residual is the first variation of the energy, and the tangent
    that of the residual, at a state that is not the reference, by
    fourth-order differences; with a law whose tangent depends on the
    strain.
  - The first iteration from the reference is the linear referential
    problem under the same load, and the difference between the
    nonlinear solution and that one falls as the load squared.
  - With the tangent made at every iteration each correction is bounded
    by a multiple of the square of the one before.
  - A ball under a uniform surface mass against the radial solution of
    the same problem (Lagrangian equations with gravity G m / r^2, no
    potential field), in the mean radial displacement of its surface.
*/

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <numbers>
#include <vector>

#include "AdGIA.hpp"
#include "NonlinearTestCommon.hpp"
#include "mfem.hpp"

namespace {

using namespace mfem;
using namespace AdGIA;
using namespace nonlinear_test;

constexpr double kG = 1.0, kRho = 1.0, kLambda = 5.0, kMu = 5.0;
constexpr int kDtNDegree = 8;

// The hydrostatic pressure of the homogeneous body of unit radius: in
// three dimensions (2 pi / 3) G rho^2 (1 - r^2), in two pi G rho^2
// (1 - r^2).
double Pressure(const Vector& x) {
  const double factor = x.Size() == 3 ? 2.0 / 3.0 : 1.0;
  return factor * std::numbers::pi * kG * kRho * kRho * (1.0 - x * x);
}

// The body, its buffer, the spaces, the extension and the two laws.
struct Body {
  Mesh parent;
  int dim;
  Array<int> body_attr{1}, buffer_attr{2};
  SubMesh body, buffer;
  H1_FECollection fec;
  FiniteElementSpace fes_u, fes_zeta, fes_buffer;
  std::unique_ptr<SparseMatrix> extension;
  ConstantCoefficient rho{kRho}, lambda{kLambda}, mu{kMu};
  IsotropicElasticTensorCoefficient Xi;
  MatrixFunctionCoefficient T0;
  QuadraticMaterialLaw quadratic;
  ConstantCoefficient nu1{30.0}, nu2{6.0}, nu3{-4.0};
  ThirdOrderIsotropicLaw cubic;
  Array<int> surface;

  static Mesh Load(int dim, int refinements) {
    auto mesh = Mesh(dim == 2 ? "../data/disc_buffer_2d.msh"
                              : "../data/ball_buffer_coarse_3d.msh",
                     1, 1);
    for (auto l = 0; l < refinements; l++) {
      mesh.UniformRefinement();
    }
    return mesh;
  }

  Body(int dim_, int order, int refinements = 0)
      : parent(Load(dim_, refinements)),
        dim(dim_),
        body(SubMesh::CreateFromDomain(parent, body_attr)),
        buffer(SubMesh::CreateFromDomain(parent, buffer_attr)),
        fec(order, dim_),
        fes_u(&body, &fec, dim_),
        fes_zeta(&parent, &fec),
        fes_buffer(&buffer, &fec, dim_),
        Xi(dim_, lambda, mu),
        T0(dim_,
           [](const Vector& x, DenseMatrix& S) {
             S.SetSize(x.Size());
             S = 0.0;
             for (int i = 0; i < x.Size(); i++) {
               S(i, i) = -Pressure(x);
             }
           }),
        quadratic(dim_, Xi, &T0),
        cubic(quadratic, nu1, nu2, nu3) {
    Vector lo, hi;
    parent.GetBoundingBox(lo, hi);
    extension =
        NewRadialVacuumExtension(fes_u, fes_buffer, 1.0, hi.Normlinf());
    surface.SetSize(body.bdr_attributes.Max());
    surface = 0;
    surface[body.bdr_attributes.Max() - 1] = 1;
  }

  std::unique_ptr<NonlinearSelfGravitatingProblem> Problem(
      MaterialLaw& law, Coefficient* sigma = nullptr) {
    auto p = std::make_unique<NonlinearSelfGravitatingProblem>(
        &fes_u, &fes_zeta, &fes_buffer, *extension, law, rho, kG, kDtNDegree);
    if (sigma) {
      p->SetSurfaceLoad(*sigma, surface);
    }
    return p;
  }
};

double Degree2Load(const Vector& x) {
  const double c = x[x.Size() - 1] / x.Norml2();
  return 3.0 * c * c - 1.0;
}

double RelDiff(const GridFunction& a, const GridFunction& b) {
  GridFunction d(a);
  d -= b;
  return d.Norml2() / b.Norml2();
}

TEST(NonlinearSelfGravitating, ReferenceWithoutLoad) {
  Body b(2, 2);
  auto p = b.Problem(b.cubic);
  Vector r_u, r_zeta;
  p->Residual(r_u, r_zeta);
  EXPECT_LT(r_u.Normlinf(), 1e-12);
  EXPECT_LT(r_zeta.Normlinf(), 1e-12);
  ASSERT_TRUE(p->Solve());
  EXPECT_EQ(p->Iterations(), 1);
  EXPECT_LT(p->Displacement().Normlinf(), 1e-12);
  EXPECT_LT(RelDiff(p->Potential(), p->ReferencePotential()), 1e-12);
}

class Derivatives : public testing::TestWithParam<std::pair<int, int>> {};

TEST_P(Derivatives, ResidualAndTangentByDifferences) {
  const auto [dim, order] = GetParam();
  Body b(dim, order);
  FunctionCoefficient sigma([](const Vector& x) {
    return 0.03 * (1.0 + 0.5 * Degree2Load(x));
  });
  auto p = b.Problem(b.cubic, &sigma);

  // A state that is not the reference, and directions of both unknowns.
  VectorFunctionCoefficient u_c(
      dim, [](const Vector& x, Vector& u) { SmoothField(x, u, 0.02, 0.3); });
  VectorFunctionCoefficient v_c(
      dim, [](const Vector& x, Vector& u) { SmoothField(x, u, 1.0, 1.1); });
  VectorFunctionCoefficient w_c(
      dim, [](const Vector& x, Vector& u) { SmoothField(x, u, 1.0, 2.7); });
  FunctionCoefficient z_c([](const Vector& x) {
    return 0.05 * std::sin(1.7 * x[0] + 0.4) * std::cos(1.1 * x[1] - 0.2);
  });
  FunctionCoefficient chi_c([](const Vector& x) {
    return std::cos(1.3 * x[0] - 0.6) * std::sin(0.9 * x[1] + 0.5);
  });
  FunctionCoefficient psi_c([](const Vector& x) {
    return std::sin(0.8 * x[0] + 1.9 * x[1]);
  });
  GridFunction u0(&b.fes_u), v(&b.fes_u), w(&b.fes_u);
  GridFunction z0(&b.fes_zeta), chi(&b.fes_zeta), psi(&b.fes_zeta);
  u0.ProjectCoefficient(u_c);
  v.ProjectCoefficient(v_c);
  w.ProjectCoefficient(w_c);
  z0.ProjectCoefficient(z_c);
  z0 += p->ReferencePotential();
  chi.ProjectCoefficient(chi_c);
  psi.ProjectCoefficient(psi_c);

  auto set = [&](double step, const GridFunction& du, const GridFunction& dz) {
    p->Displacement() = u0;
    p->Displacement().Add(step, du);
    p->Potential() = z0;
    p->Potential().Add(step, dz);
  };
  const double h = 1e-4;
  auto derivative = [&](const GridFunction& du, const GridFunction& dz,
                        auto&& value) {
    auto at = [&](double step) {
      set(step, du, dz);
      return value();
    };
    return (8.0 * (at(h) - at(-h)) - (at(2.0 * h) - at(-2.0 * h))) /
           (12.0 * h);
  };

  set(0.0, v, chi);
  Vector r_u, r_zeta;
  p->Residual(r_u, r_zeta);
  const double pairing = r_u * v + r_zeta * chi;
  const double dE = derivative(v, chi, [&] { return p->Energy(); });
  EXPECT_NEAR(dE, pairing, 1e-7 * std::abs(pairing));

  // The tangent applied to (w, psi), paired with (v, chi).
  set(0.0, v, chi);
  Vector k_u, k_zeta;
  p->Tangent().ApplyBlockOperator(w, psi, k_u, k_zeta);
  const double vKw = k_u * v + k_zeta * chi;
  const double dr = derivative(w, psi, [&] {
    Vector a_u, a_zeta;
    p->Residual(a_u, a_zeta);
    return a_u * v + a_zeta * chi;
  });
  EXPECT_NEAR(dr, vKw, 1e-7 * std::abs(vKw));
}

INSTANTIATE_TEST_SUITE_P(Bodies, Derivatives,
                         testing::Values(std::pair{2, 1}, std::pair{2, 2}));

TEST(NonlinearSelfGravitating, LinearLimitAndLoadSquared) {
  Body b(2, 2);
  std::vector<double> differences;
  for (const double amplitude : {0.02, 0.01}) {
    FunctionCoefficient sigma([amplitude](const Vector& x) {
      return amplitude * (1.0 + 0.5 * Degree2Load(x));
    });
    // The linear referential problem about the reference.
    IdentityDiffeomorphism identity(2);
    ReferentialElasticRheology rheology(2, b.Xi, b.T0, identity);
    LinearQuasiStaticReferentialSelfGravitatingProblem linear(
        &b.fes_u, &b.fes_zeta, rheology, b.rho, kG, kDtNDegree);
    linear.SetPrescribedVacuumExtension(b.fes_buffer, *b.extension);
    linear.SetSurfaceLoad(sigma, b.surface);
    linear.SetRelTol(1e-12);
    linear.AssembleForce(0.0);
    ASSERT_TRUE(linear.Solve());

    auto first = b.Problem(b.quadratic, &sigma);
    first->Options().max_iterations = 1;
    first->Solve();
    EXPECT_LT(RelDiff(first->Displacement(), linear.Displacement()), 1e-8);

    auto p = b.Problem(b.quadratic, &sigma);
    ASSERT_TRUE(p->Solve());
    GridFunction d(p->Displacement());
    d -= linear.Displacement();
    differences.push_back(d.Norml2());
  }
  EXPECT_NEAR(differences[0] / differences[1], 4.0, 0.4);
}

TEST(NonlinearSelfGravitating, NewtonConvergesQuadratically) {
  Body b(2, 2);
  FunctionCoefficient sigma([](const Vector& x) {
    return 0.05 * (1.0 + 0.5 * Degree2Load(x));
  });
  auto p = b.Problem(b.cubic, &sigma);
  p->Options().refresh_contraction = 0.0;
  ASSERT_TRUE(p->Solve());
  const auto& steps = p->Corrections();
  ASSERT_GE(steps.size(), 3u);
  EXPECT_EQ(p->TangentAssemblies(), p->Iterations());
  const double scale = steps[0];
  for (std::size_t k = 1; k < steps.size(); k++) {
    if (steps[k] > 1e-11 * scale) {
      EXPECT_LT(steps[k] / scale, 20.0 * std::pow(steps[k - 1] / scale, 2));
    }
  }
}

// At a small load the tangent of the reference serves throughout: the
// same answer as Newton's method proper, from one tangent.
TEST(NonlinearSelfGravitating, KeptTangentAtASmallLoad) {
  Body b(2, 2);
  FunctionCoefficient sigma([](const Vector& x) {
    return 0.002 * (1.0 + 0.5 * Degree2Load(x));
  });
  auto newton = b.Problem(b.quadratic, &sigma);
  newton->Options().refresh_contraction = 0.0;
  ASSERT_TRUE(newton->Solve());
  auto kept = b.Problem(b.quadratic, &sigma);
  ASSERT_TRUE(kept->Solve());
  EXPECT_EQ(kept->TangentAssemblies(), 1);
  EXPECT_GT(kept->Iterations(), 1);
  EXPECT_LT(RelDiff(kept->Displacement(), newton->Displacement()), 1e-8);
}

// The mean radial displacement of the surface, int u . n dS / int dS.
double MeanRadial(Body& b, const GridFunction& u) {
  LinearForm flux(&b.fes_u);
  ConstantCoefficient one(1.0);
  flux.AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(one),
                             b.surface);
  flux.Assemble();
  return flux(u) / (4.0 * std::numbers::pi);
}

// A homogeneous ball under a uniform surface mass 0.05, against the radial
// solution (tests/radial_nonlinear_solid.py; rho = G = a = 1, bare
// lambda = mu = 5): u_r(a) is -1.045206e-2, and -9.340776e-3 in the
// linear problem. On this mesh at order 2 the two are 1.1% and 1.2% off,
// and their difference, which is the nonlinear effect, 2%.
TEST(NonlinearSelfGravitating, UniformLoadOnABall) {
  Body b(3, 2);
  ConstantCoefficient sigma(0.05);
  auto first = b.Problem(b.quadratic, &sigma);
  first->Options().max_iterations = 1;
  first->Solve();
  auto p = b.Problem(b.quadratic, &sigma);
  ASSERT_TRUE(p->Solve());
  const double linear = MeanRadial(b, first->Displacement());
  const double nonlinear = MeanRadial(b, p->Displacement());
  EXPECT_NEAR(linear, -9.340776e-3, 2e-2 * 9.340776e-3);
  EXPECT_NEAR(nonlinear, -1.045206e-2, 2e-2 * 1.045206e-2);
  EXPECT_NEAR(nonlinear - linear, -1.045206e-2 + 9.340776e-3,
              5e-2 * (1.045206e-2 - 9.340776e-3));
}


// ---------------------------------------------------------------------------
// A manufactured solution on the disc.
//
// With w(r) = (1 - r^2)^3 inside the body and zero outside, the fields
//   u* = a w(r) s(x),   zeta* = zeta_0 + c w(r) t(x)
// leave the surface and the buffer as they are, so that the extension of
// u* vanishes and zeta* is the reference potential there. The loads that
// make them the solution are given by their stress and flux: with
// g = grad zeta, v = F^-T g and the reference's g_0 = 2 pi G rho x,
//   M = [P(F*) - S0] + m(F*, g*) - m(1, g_0),
//       m(F, g) = (J / 8 pi G) (|v|^2 F^-T - 2 v (F^-1 v)^T),
//   q = [a(F*) g* - g_0] / 4 pi G,
// m being the derivative in F of the gravity energy density
// J |F^-T g|^2 / 8 pi G. The reference potential is the discrete one of
// the class, taken from a problem without loads, so that the fields above
// solve the discrete problem's continuous counterpart whatever the mesh
// makes of the disc.
//
// u* has no rigid part (its components have opposite parities in the two
// coordinates). The class takes the rigid motions out of every
// correction, as the free body's own loads allow; a dead load is not
// unchanged by a rotation of the body, so that with a u* that held one
// the class's answer would differ from u* by more than a rigid motion,
// at the size of the rotation, at every mesh. Measured: rates of 0.99
// and 1.98 at order 1, and 1.99 and 2.99 at order 2 over three meshes.

constexpr double kAmplitude = 0.02, kPotentialAmplitude = 0.3;

double Window(const Vector& x, Vector* gradient) {
  const double q = 1.0 - x * x;
  if (gradient) {
    gradient->SetSize(x.Size());
    *gradient = 0.0;
    if (q > 0.0) {
      gradient->Add(-6.0 * q * q, x);
    }
  }
  return q > 0.0 ? q * q * q : 0.0;
}

void ExactDisplacement(const Vector& x, Vector& u, DenseMatrix* gradient) {
  Vector dw;
  const double w = Window(x, &dw);
  // Odd in x_0 and even in x_1 for the first component, the other way
  // round for the second: no rigid part.
  const double s[2] = {std::sin(1.3 * x[0]) * std::cos(0.9 * x[1]),
                       std::cos(0.7 * x[0]) * std::sin(1.1 * x[1])};
  const double ds[2][2] = {
      {1.3 * std::cos(1.3 * x[0]) * std::cos(0.9 * x[1]),
       -0.9 * std::sin(1.3 * x[0]) * std::sin(0.9 * x[1])},
      {-0.7 * std::sin(0.7 * x[0]) * std::sin(1.1 * x[1]),
       1.1 * std::cos(0.7 * x[0]) * std::cos(1.1 * x[1])}};
  u.SetSize(2);
  if (gradient) {
    gradient->SetSize(2);
  }
  for (int i = 0; i < 2; i++) {
    u[i] = kAmplitude * w * s[i];
    for (int j = 0; gradient && j < 2; j++) {
      (*gradient)(i, j) = kAmplitude * (s[i] * dw[j] + w * ds[i][j]);
    }
  }
}

double ExactPotentialChange(const Vector& x, Vector* gradient) {
  Vector dw;
  const double w = Window(x, &dw);
  const double a = 0.9 * x[0] + 1.4 * x[1];
  if (gradient) {
    gradient->SetSize(2);
    (*gradient)[0] = 0.9 * std::cos(a);
    (*gradient)[1] = 1.4 * std::cos(a);
    *gradient *= kPotentialAmplitude * w;
    gradient->Add(kPotentialAmplitude * std::sin(a), dw);
  }
  return kPotentialAmplitude * w * std::sin(a);
}

// F* and the gradient of the potential's change at a point.
void ExactState(const Vector& x, DenseMatrix& F, Vector& dg) {
  Vector u;
  ExactDisplacement(x, u, &F);
  F(0, 0) += 1.0;
  F(1, 1) += 1.0;
  ExactPotentialChange(x, &dg);
}

// m(F, g), and a(F) g.
void GravityStress(const DenseMatrix& F, const Vector& g, DenseMatrix& m,
                   Vector& ag) {
  DenseMatrix Fi(F);
  const double J = Fi.Det();
  Fi.Invert();
  Vector v(2), Fiv(2);
  Fi.MultTranspose(g, v);
  Fi.Mult(v, Fiv);
  m.SetSize(2);
  for (int i = 0; i < 2; i++) {
    for (int A = 0; A < 2; A++) {
      m(i, A) = J * ((v * v) * Fi(A, i) - 2.0 * v[i] * Fiv[A]) /
                (8.0 * std::numbers::pi * kG);
    }
  }
  ag = Fiv;
  ag *= J;
}

// m(F*, g_0 + grad delta) - m(1, g_0), with g_0 the gradient of the
// reference potential.
class GravityStressLoad : public MatrixCoefficient {
 public:
  explicit GravityStressLoad(VectorCoefficient& g0)
      : MatrixCoefficient(2), g0_(&g0) {}
  void Eval(DenseMatrix& M, ElementTransformation& T,
            const IntegrationPoint& ip) override {
    Vector x, g0, g, ag;
    DenseMatrix F, m0, one(2);
    T.Transform(ip, x);
    g0_->Eval(g0, T, ip);
    ExactState(x, F, g);
    g += g0;
    GravityStress(F, g, M, ag);
    one = 0.0;
    one(0, 0) = one(1, 1) = 1.0;
    GravityStress(one, g0, m0, ag);
    M -= m0;
  }

 private:
  VectorCoefficient* g0_;
};

// [a(F*) (g_0 + grad delta) - g_0] / 4 pi G.
class PotentialFluxLoad : public VectorCoefficient {
 public:
  explicit PotentialFluxLoad(VectorCoefficient& g0)
      : VectorCoefficient(2), g0_(&g0) {}
  void Eval(Vector& q, ElementTransformation& T,
            const IntegrationPoint& ip) override {
    Vector x, g0, g;
    DenseMatrix F, m;
    T.Transform(ip, x);
    g0_->Eval(g0, T, ip);
    ExactState(x, F, g);
    g += g0;
    GravityStress(F, g, m, q);
    q -= g0;
    q /= 4.0 * std::numbers::pi * kG;
  }

 private:
  VectorCoefficient* g0_;
};

// The strain of a displacement gradient, as a matrix.
void Strain(const DenseMatrix& grad_u, DenseMatrix& E) {
  DenseMatrix F(grad_u);
  F(0, 0) += 1.0;
  F(1, 1) += 1.0;
  E.SetSize(2);
  MultAtB(F, F, E);
  E(0, 0) -= 1.0;
  E(1, 1) -= 1.0;
  E *= 0.5;
}

// The errors of a solve at one refinement: the Green strain in L2 over
// the body and the potential's change in L2 over the body less its mean,
// neither of which a rigid motion of the solution changes.
std::pair<double, double> ManufacturedErrors(int order, int refinements) {
  Body b(2, order, refinements);
  IdentityDiffeomorphism identity(2);
  CallableDiffeomorphism exact(
      2,
      [](const Vector& x, Vector& y) {
        ExactDisplacement(x, y, nullptr);
        y += x;
      },
      [](const Vector& x, DenseMatrix& F) {
        Vector u;
        ExactDisplacement(x, u, &F);
        F(0, 0) += 1.0;
        F(1, 1) += 1.0;
      });
  FirstPiolaKirchhoffStressCoefficient elastic(b.cubic, exact, &identity);
  // The discrete reference potential, on the ball and on the body.
  GridFunction zeta0(&b.fes_zeta);
  zeta0 = b.Problem(b.cubic)->ReferencePotential();
  FiniteElementSpace fes_zeta_body(&b.body, &b.fec);
  GridFunction zeta0_body(&fes_zeta_body);
  SubMesh::Transfer(zeta0, zeta0_body);
  GradientGridFunctionCoefficient g0_ball(&zeta0), g0_body(&zeta0_body);

  GravityStressLoad gravity(g0_body);
  MatrixSumCoefficient stress(elastic, gravity);
  PotentialFluxLoad flux(g0_ball);

  auto p = b.Problem(b.cubic);
  p->SetBodyLoad(nullptr, &stress);
  p->SetPotentialLoad(nullptr, &flux);
  p->Options().refresh_contraction = 0.0;
  EXPECT_TRUE(p->Solve());

  // The strain.
  double strain = 0.0;
  DenseMatrix grad_h, grad, E_h, E;
  Vector x, u;
  for (int e = 0; e < b.body.GetNE(); e++) {
    ElementTransformation* T = b.body.GetElementTransformation(e);
    const IntegrationRule& rule =
        IntRules.Get(T->GetGeometryType(), 2 * order + 4);
    for (int q = 0; q < rule.GetNPoints(); q++) {
      const IntegrationPoint& ip = rule.IntPoint(q);
      T->SetIntPoint(&ip);
      T->Transform(ip, x);
      p->Displacement().GetVectorGradient(*T, grad_h);
      ExactDisplacement(x, u, &grad);
      Strain(grad_h, E_h);
      Strain(grad, E);
      E_h -= E;
      strain += ip.weight * T->Weight() * (E_h * E_h);
    }
  }
  // The potential's change on the body, where it goes with the points of
  // the body whatever rigid motion the solution holds (in the buffer it
  // does not: the extension is not a rigid motion). The mean of the
  // difference is taken off: in two dimensions the potential is fixed up
  // to a constant.
  GridFunction d(p->Potential());
  d -= p->ReferencePotential();
  double sum = 0.0, square = 0.0, volume = 0.0;
  for (int e = 0; e < b.parent.GetNE(); e++) {
    if (b.parent.GetAttribute(e) != 1) {
      continue;
    }
    ElementTransformation* T = b.parent.GetElementTransformation(e);
    const IntegrationRule& rule =
        IntRules.Get(T->GetGeometryType(), 2 * order + 4);
    for (int q = 0; q < rule.GetNPoints(); q++) {
      const IntegrationPoint& ip = rule.IntPoint(q);
      T->SetIntPoint(&ip);
      T->Transform(ip, x);
      const double w = ip.weight * T->Weight();
      const double value =
          d.GetValue(*T, ip) - ExactPotentialChange(x, nullptr);
      sum += w * value;
      square += w * value * value;
      volume += w;
    }
  }
  return {std::sqrt(strain), std::sqrt(square - sum * sum / volume)};
}

TEST(NonlinearSelfGravitatingManufactured, ConvergesAtTheOptimalRate) {
  for (const int order : {1, 2}) {
    const auto coarse = ManufacturedErrors(order, 0);
    const auto fine = ManufacturedErrors(order, 1);
    const double strain_rate = std::log2(coarse.first / fine.first);
    const double potential_rate = std::log2(coarse.second / fine.second);
    EXPECT_GT(strain_rate, order - 0.2);
    EXPECT_GT(potential_rate, order + 0.7);
  }
}


// ---------------------------------------------------------------------------
// The same disc described from a relabelled reference: the mesh is taken
// to the body by a mapping that moves the points within r < 0.8 and
// leaves the rest, the surface and the buffer among it. The stress of
// the law is the body's at the image of each point and the density is
// per unit volume of the mesh, J_e rho. Outside r = 0.8 a point of the
// mesh is the same point of the body in both descriptions, so on the
// body there the reference potential and the potential's change are the
// same, and so is the mean radial displacement of the surface; they
// fall together as the mesh is refined.

CallableDiffeomorphism DiscRelabelling(double c) {
  auto bump = [](const Vector& x, Vector* gradient) {
    const double q = 1.0 - (x * x) / 0.64;
    if (gradient) {
      gradient->SetSize(2);
      *gradient = 0.0;
      if (q > 0.0) {
        gradient->Add(-6.0 * q * q / 0.64, x);
      }
    }
    return q > 0.0 ? q * q * q : 0.0;
  };
  return CallableDiffeomorphism(
      2,
      [c, bump](const Vector& x, Vector& y) {
        const double b = bump(x, nullptr);
        y = x;
        y[0] += c * b;
        y[1] += 0.4 * c * b;
      },
      [c, bump](const Vector& x, DenseMatrix& F) {
        Vector g;
        bump(x, &g);
        F.SetSize(2);
        F(0, 0) = 1.0 + c * g[0];
        F(0, 1) = c * g[1];
        F(1, 0) = 0.4 * c * g[0];
        F(1, 1) = 1.0 + 0.4 * c * g[1];
      });
}

// The largest difference of two fields on the ball at the nodes of the
// body outside r = 0.85, over the largest value of the first there, each
// field less its mean over those nodes: in two dimensions the potential
// is fixed up to a constant.
double OuterDifference(Body& b, const GridFunction& first,
                       const GridFunction& second) {
  GridFunction radius(&b.fes_zeta);
  FunctionCoefficient r_c([](const Vector& x) { return x.Norml2(); });
  radius.ProjectCoefficient(r_c);
  double mean_first = 0.0, mean_second = 0.0;
  int count = 0;
  for (int i = 0; i < radius.Size(); i++) {
    if (radius[i] > 0.85 && radius[i] < 1.0 + 1e-9) {
      mean_first += first[i];
      mean_second += second[i];
      count++;
    }
  }
  mean_first /= count;
  mean_second /= count;
  double difference = 0.0, size = 0.0;
  for (int i = 0; i < radius.Size(); i++) {
    if (radius[i] > 0.85 && radius[i] < 1.0 + 1e-9) {
      difference = std::max(difference,
                            std::abs((first[i] - mean_first) -
                                     (second[i] - mean_second)));
      size = std::max(size, std::abs(first[i] - mean_first));
    }
  }
  return difference / size;
}

TEST(NonlinearSelfGravitatingRelabelled, SameBodyThroughAnotherMapping) {
  const int order = 2;
  auto relabelling = DiscRelabelling(0.06);
  FunctionCoefficient sigma([](const Vector& x) {
    return 0.03 * (1.0 + 0.5 * Degree2Load(x));
  });
  std::vector<double> differences;
  for (const int refinements : {0, 1}) {
    Body b(2, order, refinements);
    auto plain = b.Problem(b.cubic, &sigma);
    ASSERT_TRUE(plain->Solve());

    TransformedMatrixFunctionCoefficient T0(
        2, relabelling, [](const Vector& x, DenseMatrix& S) {
          S.SetSize(2);
          S = 0.0;
          S(0, 0) = S(1, 1) = -Pressure(x);
        });
    QuadraticMaterialLaw quadratic(2, b.Xi, &T0);
    ThirdOrderIsotropicLaw cubic(quadratic, b.nu1, b.nu2, b.nu3);
    JacobianCoefficient jacobian(relabelling);
    ProductCoefficient density(kRho, jacobian);
    NonlinearSelfGravitatingProblem relabelled(
        &b.fes_u, &b.fes_zeta, &b.fes_buffer, *b.extension, cubic, density,
        kG, kDtNDegree, &relabelling);
    relabelled.SetSurfaceLoad(sigma, b.surface);
    ASSERT_TRUE(relabelled.Solve());

    LinearForm flux(&b.fes_u);
    ConstantCoefficient one(1.0);
    flux.AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(one),
                               b.surface);
    flux.Assemble();
    const double radial = flux(plain->Displacement());
    double worst = std::abs(flux(relabelled.Displacement()) - radial) /
                   std::abs(radial);
    const double d0 = OuterDifference(b, plain->ReferencePotential(),
                                      relabelled.ReferencePotential());
    GridFunction change(plain->Potential()), other(relabelled.Potential());
    change -= plain->ReferencePotential();
    other -= relabelled.ReferencePotential();
    const double d1 = OuterDifference(b, change, other);
    worst = std::max({worst, d0, d1});
    differences.push_back(worst);
  }
  // Measured: 2.5e-5 and 1.0e-6.
  EXPECT_LT(differences[1], 0.2 * differences[0]);
  EXPECT_LT(differences[1], 1e-5);
}

}  // namespace
