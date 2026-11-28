#include "NonlinearTestCommon.hpp"
#include "TestCommon.hpp"

/*
  Tests for material_law.hpp.

  - The Green strain as a Mandel vector, and the round trip between a
    symmetric matrix and its Mandel vector.
  - A law at a point: the stress against central differences of the
    energy, and the tangent against central differences of the stress,
    at a strain that is not small; for the quadratic law and for a law
    whose tangent depends on the strain.
  - The quadratic law at zero strain returns its equilibrium stress and
    its elastic tensor.
  - The coefficients at the deformation of a mapping with a constant
    gradient: the first Piola-Kirchhoff stress against central
    differences of the energy density in the components of the gradient;
    the second Piola-Kirchhoff stress and the tangent against the law;
    the reference terms taken off on request.
*/

namespace {

using namespace nonlinear_test;

// One element and a point in it, for laws whose coefficients vary.
struct Point {
  Mesh mesh;
  ElementTransformation* T;
  IntegrationPoint ip;

  explicit Point(int dim) : mesh(MakeBody(dim, 0, 1)) {
    T = mesh.GetElementTransformation(0);
    ip = Geometries.GetCenter(mesh.GetElementBaseGeometry(0));
    T->SetIntPoint(&ip);
  }
};

// A symmetric strain of size about 0.1, as a Mandel vector.
Vector Strain(int dim) {
  auto A = DenseMatrix(dim);
  for (auto i = 0; i < dim; i++) {
    for (auto j = 0; j < dim; j++) {
      A(i, j) = 0.1 * std::sin(1.0 + 2.0 * i + 3.0 * j);
    }
  }
  auto e = Vector();
  MatrixToMandel(A, e);
  return e;
}

class MaterialLawTest : public testing::TestWithParam<int> {};

TEST_P(MaterialLawTest, GreenStrainAndMandel) {
  const auto dim = GetParam();
  auto F = DenseMatrix(dim);
  for (auto i = 0; i < dim; i++) {
    for (auto j = 0; j < dim; j++) {
      F(i, j) = (i == j ? 1.0 : 0.0) + 0.2 * std::cos(1.0 + i + 2.0 * j);
    }
  }
  auto e = Vector();
  GreenStrain(F, e);
  auto E = DenseMatrix();
  MandelToMatrix(dim, e, E);
  auto expected = DenseMatrix(dim);
  MultAtB(F, F, expected);
  for (auto i = 0; i < dim; i++) {
    expected(i, i) -= 1.0;
  }
  expected *= 0.5;
  expected -= E;
  EXPECT_LT(expected.MaxMaxNorm(), 1e-14);

  // The Mandel vector is an isometry: E : E is the square of its norm.
  auto contraction = real_t{0};
  for (auto i = 0; i < dim; i++) {
    for (auto j = 0; j < dim; j++) {
      contraction += E(i, j) * E(i, j);
    }
  }
  EXPECT_NEAR(e * e, contraction, 1e-14);

  auto back = Vector();
  MatrixToMandel(E, back);
  back -= e;
  EXPECT_LT(back.Normlinf(), 1e-15);
}

TEST_P(MaterialLawTest, StressAndTangentAreDerivatives) {
  const auto dim = GetParam();
  auto point = Point(dim);
  auto material = Material(dim);
  const auto e = Strain(dim);
  const auto n = material.quadratic.StrainSize();
  const auto h = 1e-5;

  // A law about an equilibrium that is the image of the mesh under a
  // mapping of constant gradient, as a law about the mesh.
  auto equilibrium = CallableDiffeomorphism(
      dim,
      [](const Vector& x, Vector& y) {
        y = x;
        y[0] += 0.2 * x[x.Size() - 1];
        y *= 1.1;
      },
      [dim](const Vector&, DenseMatrix& F) {
        F = 0.0;
        for (auto i = 0; i < dim; i++) {
          F(i, i) = 1.1;
        }
        F(0, dim - 1) += 0.22;
      });
  auto mapped = MappedMaterialLaw(material.cubic, equilibrium);

  for (MaterialLaw* law :
       {static_cast<MaterialLaw*>(&material.quadratic),
        static_cast<MaterialLaw*>(&material.cubic),
        static_cast<MaterialLaw*>(&mapped)}) {
    auto S = Vector(n);
    auto C = DenseMatrix(n);
    law->Evaluate(*point.T, point.ip, e, &S, &C);
    for (auto a = 0; a < n; a++) {
      auto plus = e, minus = e;
      plus[a] += h;
      minus[a] -= h;
      auto S_plus = Vector(n), S_minus = Vector(n);
      const auto W_plus =
          law->Evaluate(*point.T, point.ip, plus, &S_plus, nullptr);
      const auto W_minus =
          law->Evaluate(*point.T, point.ip, minus, &S_minus, nullptr);
      EXPECT_NEAR((W_plus - W_minus) / (2.0 * h), S[a], 1e-8);
      for (auto b = 0; b < n; b++) {
        EXPECT_NEAR((S_plus[b] - S_minus[b]) / (2.0 * h), C(b, a), 1e-8);
      }
    }
    // The tangent has the classical symmetries: a symmetric matrix.
    auto Ct = C;
    Ct.Transpose();
    Ct -= C;
    EXPECT_LT(Ct.MaxMaxNorm(), 1e-14);
  }
}

// The mapped law is the law at the strain from equilibrium, per unit
// volume of the mesh: with F^ the gradient from the mesh and F_e the
// equilibrium's, J_e W(E(F^ F_e^-1)); and at F^ = F_e its first
// Piola-Kirchhoff stress is the pull-back J_e T0 F_e^-T of the
// equilibrium stress.
TEST_P(MaterialLawTest, MappedLawIsTheLawFromEquilibrium) {
  const auto dim = GetParam();
  auto point = Point(dim);
  auto material = Material(dim);
  auto Fe = DenseMatrix(dim);
  Fe = 0.0;
  for (auto i = 0; i < dim; i++) {
    Fe(i, i) = 1.1;
  }
  Fe(0, dim - 1) += 0.22;
  auto equilibrium = CallableDiffeomorphism(
      dim, [&Fe](const Vector& x, Vector& y) { Fe.Mult(x, y); },
      [&Fe](const Vector&, DenseMatrix& F) { F = Fe; });
  auto mapped = MappedMaterialLaw(material.cubic, equilibrium);

  auto F_hat = Fe;
  for (auto i = 0; i < dim; i++) {
    for (auto j = 0; j < dim; j++) {
      F_hat(i, j) += 0.05 * std::sin(1.0 + 2.0 * i + 0.7 * j);
    }
  }
  auto Fe_inverse = Fe;
  Fe_inverse.Invert();
  auto F = DenseMatrix(dim);
  Mult(F_hat, Fe_inverse, F);
  auto e_hat = Vector(), e = Vector();
  GreenStrain(F_hat, e_hat);
  GreenStrain(F, e);
  const auto W_hat = mapped.Evaluate(*point.T, point.ip, e_hat, nullptr, nullptr);
  const auto W =
      material.cubic.Evaluate(*point.T, point.ip, e, nullptr, nullptr);
  EXPECT_NEAR(W_hat, Fe.Det() * W, 1e-14);

  // At the equilibrium: P^ = F_e S^ = J_e T0 F_e^-T.
  GreenStrain(Fe, e_hat);
  auto s_hat = Vector();
  mapped.Evaluate(*point.T, point.ip, e_hat, &s_hat, nullptr);
  auto S_hat = DenseMatrix();
  MandelToMatrix(dim, s_hat, S_hat);
  auto P_hat = DenseMatrix(dim);
  Mult(Fe, S_hat, P_hat);
  auto T0 = DenseMatrix();
  material.T0.Eval(T0, *point.T, point.ip);
  auto expected = DenseMatrix(dim);
  MultABt(T0, Fe_inverse, expected);
  expected *= Fe.Det();
  expected -= P_hat;
  EXPECT_LT(expected.MaxMaxNorm(), 1e-14);
}

TEST_P(MaterialLawTest, QuadraticLawAtTheReference) {
  const auto dim = GetParam();
  auto point = Point(dim);
  auto material = Material(dim);
  const auto n = material.quadratic.StrainSize();
  auto zero = Vector(n);
  zero = 0.0;
  auto S = Vector(n);
  auto C = DenseMatrix(n);
  const auto W = material.quadratic.Evaluate(*point.T, point.ip, zero, &S, &C);
  EXPECT_EQ(W, 0.0);

  auto T0 = DenseMatrix();
  material.T0.Eval(T0, *point.T, point.ip);
  auto t0 = Vector();
  MatrixToMandel(T0, t0);
  t0 -= S;
  EXPECT_LT(t0.Normlinf(), 1e-15);

  auto Xi = DenseMatrix();
  material.Xi.Eval(Xi, *point.T, point.ip);
  Xi -= C;
  EXPECT_LT(Xi.MaxMaxNorm(), 1e-15);

  // Without an equilibrium stress the reference is natural.
  auto natural = QuadraticMaterialLaw(dim, material.Xi);
  natural.Evaluate(*point.T, point.ip, zero, &S, nullptr);
  EXPECT_EQ(S.Normlinf(), 0.0);
}

TEST_P(MaterialLawTest, CoefficientsAtADeformation) {
  const auto dim = GetParam();
  auto point = Point(dim);
  auto material = Material(dim);
  auto& law = material.cubic;

  // The mapping x + A x, of constant gradient F = 1 + A.
  auto A = DenseMatrix(dim);
  for (auto i = 0; i < dim; i++) {
    for (auto j = 0; j < dim; j++) {
      A(i, j) = 0.15 * std::sin(0.5 + 1.7 * i + 0.9 * j);
    }
  }
  auto map = CallableDiffeomorphism(
      dim,
      [&A](const Vector& x, Vector& xi) {
        xi = x;
        A.AddMult(x, xi);
      },
      [&A, dim](const Vector&, DenseMatrix& F) {
        F = A;
        for (auto i = 0; i < dim; i++) {
          F(i, i) += 1.0;
        }
      });

  auto W = StrainEnergyDensityCoefficient(law, map);
  auto P = FirstPiolaKirchhoffStressCoefficient(law, map);
  auto P_at = DenseMatrix();
  P.Eval(P_at, *point.T, point.ip);
  const auto h = 1e-5;
  for (auto i = 0; i < dim; i++) {
    for (auto j = 0; j < dim; j++) {
      const auto a = A(i, j);
      A(i, j) = a + h;
      const auto plus = W.Eval(*point.T, point.ip);
      A(i, j) = a - h;
      const auto minus = W.Eval(*point.T, point.ip);
      A(i, j) = a;
      EXPECT_NEAR((plus - minus) / (2.0 * h), P_at(i, j), 1e-8);
    }
  }

  // S and the tangent are the law's at the Green strain of F, and
  // P = F S.
  auto F = A;
  for (auto i = 0; i < dim; i++) {
    F(i, i) += 1.0;
  }
  auto e = Vector();
  GreenStrain(F, e);
  const auto n = law.StrainSize();
  auto s = Vector(n);
  auto C = DenseMatrix(n);
  const auto energy = law.Evaluate(*point.T, point.ip, e, &s, &C);
  EXPECT_NEAR(W.Eval(*point.T, point.ip), energy, 1e-15);
  auto S = DenseMatrix();
  MandelToMatrix(dim, s, S);
  auto S_at = DenseMatrix();
  auto S_coefficient = SecondPiolaKirchhoffStressCoefficient(law, map);
  S_coefficient.Eval(S_at, *point.T, point.ip);
  auto FS = DenseMatrix(dim);
  Mult(F, S, FS);
  FS -= P_at;
  EXPECT_LT(FS.MaxMaxNorm(), 1e-14);
  S_at -= S;
  EXPECT_LT(S_at.MaxMaxNorm(), 1e-15);
  auto C_at = DenseMatrix();
  auto C_coefficient = MaterialTangentCoefficient(law, map);
  C_coefficient.Eval(C_at, *point.T, point.ip);
  C_at -= C;
  EXPECT_LT(C_at.MaxMaxNorm(), 1e-15);

  // With the reference taken off: P - S(0), and W - S(0) : (F - 1).
  auto identity = IdentityDiffeomorphism(dim);
  auto zero = Vector(n);
  zero = 0.0;
  auto s0 = Vector(n);
  law.Evaluate(*point.T, point.ip, zero, &s0, nullptr);
  auto S0 = DenseMatrix();
  MandelToMatrix(dim, s0, S0);
  auto P_incremental = FirstPiolaKirchhoffStressCoefficient(law, map, &identity);
  auto P_inc = DenseMatrix();
  P_incremental.Eval(P_inc, *point.T, point.ip);
  P_inc += S0;
  P_inc -= P_at;
  EXPECT_LT(P_inc.MaxMaxNorm(), 1e-14);
  auto W_incremental = StrainEnergyDensityCoefficient(law, map, &identity);
  auto linear = real_t{0};
  for (auto i = 0; i < dim; i++) {
    for (auto j = 0; j < dim; j++) {
      linear += S0(i, j) * A(i, j);
    }
  }
  EXPECT_NEAR(W_incremental.Eval(*point.T, point.ip), energy - linear, 1e-14);
}

INSTANTIATE_TEST_SUITE_P(MaterialLaw, MaterialLawTest, testing::Values(2, 3));

}  // namespace
