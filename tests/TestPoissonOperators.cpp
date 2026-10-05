#include <numbers>

#include "MixedProblemTestCommon.hpp"
#include "TestCommon.hpp"

/*
  Tests for the multipole operators of poisson.hpp on the body-in-buffer
  test meshes of MixedProblemTestCommon.hpp: a unit ball (disc), attribute
  1, about the origin inside a buffer with a spherical outer boundary of
  radius b.

  With nabla^2 u = rho, a density rho = r^l Y_i in the unit ball has the
  single multipole moment q = int r^{2l} Y_i^2 dV, and on the outer boundary

    du/dn = (l + 1) / (2 l + 1) q / b^{l+2} Y_i    (3-D),
    du/dn = q / (2 b^{l+1}) Y_i                    (2-D, l >= 1),

  so that pairing with v = Y_i on the boundary isolates the degree.

  - Mass: v = 1 and rho = 1 give the mass of the ball in either dimension.
  - Static operator: v^T A rho against the closed form, degree by degree.
  - Linearised operator: the displacement u = grad(r^l Y_i) of a uniform
    ball changes the moment by int |grad(r^l Y_i)|^2 dV = l a^{2l+d-2}.
  - Transposes: z.(A x) = x.(A^T z), with A^T z sized to the trial space.
*/

namespace {

using namespace self_grav_test;

constexpr double kPi = std::numbers::pi;

int MaxDegree(int dim) { return dim == 2 ? 4 : 3; }
// Order-2 fields on the order-2 curved meshes.
double Tol(int dim) { return dim == 2 ? 1e-4 : 2e-2; }

struct PoissonCase {
  std::unique_ptr<Mesh> mesh;
  H1_FECollection h1;
  L2_FECollection l2;
  std::unique_ptr<FiniteElementSpace> phi, rho, disp;
  Array<int> body;
  int dim, L;
  SurfaceHarmonics basis;
  Vector origin;

  explicit PoissonCase(int d)
      : h1(2, d), l2(2, d), dim(d), L(MaxDegree(d)), basis(d, L), origin(d) {
    mesh = std::make_unique<Mesh>(MeshFile(dim).c_str(), 1, 1);
    phi = std::make_unique<FiniteElementSpace>(mesh.get(), &h1);
    rho = std::make_unique<FiniteElementSpace>(mesh.get(), &l2);
    disp = std::make_unique<FiniteElementSpace>(mesh.get(), &l2, dim);
    body = BodyMarker(*mesh);
    origin = 0.0;
  }

  // The interpolant of Y_i extended along rays.
  GridFunction Harmonic(int i) const {
    Vector c(basis.Size());
    c = 0.0;
    c[i] = 1.0;
    HarmonicExpansionCoefficient Yi(basis, c, origin, 1.0);
    GridFunction v(phi.get());
    v.ProjectCoefficient(Yi);
    return v;
  }
};

class PoissonOperatorsTest : public testing::TestWithParam<int> {};

TEST_P(PoissonOperatorsTest, Mass) {
  PoissonCase s(GetParam());
  PoissonMultipoleOperator A(s.rho.get(), s.phi.get(), s.L, s.body);
  A.Assemble();

  GridFunction rho(s.rho.get()), v(s.phi.get());
  rho = 1.0;
  v = 1.0;
  Vector y(A.Height());
  A.Mult(rho, y);
  const double mass = s.dim == 2 ? kPi : 4.0 * kPi / 3.0;
  EXPECT_NEAR(v * y, mass, Tol(s.dim) * mass);
}

TEST_P(PoissonOperatorsTest, StaticMoments) {
  PoissonCase s(GetParam());
  const int dim = s.dim;
  PoissonMultipoleOperator A(s.rho.get(), s.phi.get(), s.L, s.body);
  A.Assemble();
  PoissonDtNOperator dtn(s.phi.get(), s.L);
  const double b = dtn.BoundaryRadius();

  for (int i = 1; i < s.basis.Size(); i++) {
    const int l = s.basis.Degree(i);
    // rho = r^l Y_i: the interior harmonic of unit radius.
    Vector c(s.basis.Size());
    c = 0.0;
    c[i] = 1.0;
    HarmonicExpansionCoefficient harmonic(s.basis, c, s.origin, 1.0, true);
    GridFunction rho(s.rho.get());
    rho.ProjectCoefficient(harmonic);
    auto v = s.Harmonic(i);

    Vector y(A.Height());
    A.Mult(rho, y);
    // q = 1 / (2 l + d); int_S Y_i^2 dS = b^{d-1}.
    const double q = 1.0 / (2 * l + dim);
    const double expected =
        dim == 2 ? q / (2.0 * std::pow(b, l))
                 : (l + 1.0) / (2 * l + 1.0) * q / std::pow(b, l);
    EXPECT_NEAR(v * y, expected, Tol(dim) * expected) << "harmonic " << i;
  }
}

TEST_P(PoissonOperatorsTest, LinearisedMoments) {
  PoissonCase s(GetParam());
  const int dim = s.dim;
  PoissonLinearisedMultipoleOperator A(s.disp.get(), s.phi.get(), s.L, s.body);
  A.Assemble();
  PoissonDtNOperator dtn(s.phi.get(), s.L);
  const double b = dtn.BoundaryRadius();

  for (int i = 1; i < s.basis.Size(); i++) {
    const int l = s.basis.Degree(i);
    // u = grad(r^l Y_i) = r^{l-1} (l Y_i x_hat + grad_1 Y_i).
    VectorFunctionCoefficient grad(dim, [&](const Vector& x, Vector& u) {
      Vector Y;
      DenseMatrix gradY;
      s.basis.EvalWithGradient(x, Y, gradY);
      const double r = x.Norml2();
      const double rpow = std::pow(r, l - 1);
      for (int a = 0; a < dim; a++) {
        // At the centre, the direction SurfaceHarmonics takes there.
        const double xhat =
            r > 0 ? x[a] / r : (a == (dim == 2 ? 0 : 2) ? 1.0 : 0.0);
        u[a] = rpow * (l * Y[i] * xhat + gradY(a, i));
      }
    });
    GridFunction u(s.disp.get());
    u.ProjectCoefficient(grad);
    auto v = s.Harmonic(i);

    Vector y(A.Height());
    A.Mult(u, y);
    // int |grad(r^l Y_i)|^2 dV = l over the unit ball.
    const double expected =
        dim == 2 ? l / (2.0 * std::pow(b, l))
                 : (l + 1.0) / (2 * l + 1.0) * l / std::pow(b, l);
    EXPECT_NEAR(v * y, expected, Tol(dim) * expected) << "harmonic " << i;
  }
}

TEST_P(PoissonOperatorsTest, Transposes) {
  PoissonCase s(GetParam());
  auto fill = [](Vector& x, double a) {
    for (int i = 0; i < x.Size(); i++) {
      x[i] = std::sin(a * i + 0.3);
    }
  };
  auto check = [&](const Operator& A) {
    Vector x(A.Width()), z(A.Height()), Ax, Atz;
    fill(x, 1.3);
    fill(z, 0.7);
    A.Mult(x, Ax);
    A.MultTranspose(z, Atz);
    ASSERT_EQ(Ax.Size(), A.Height());
    ASSERT_EQ(Atz.Size(), A.Width());
    const double scale = Ax.Norml2() * z.Norml2();
    EXPECT_NEAR(z * Ax, x * Atz, 1e-12 * scale);
  };

  PoissonMultipoleOperator A(s.rho.get(), s.phi.get(), s.L, s.body);
  A.Assemble();
  check(A);

  ConstantCoefficient density(2.0);
  PoissonLinearisedMultipoleOperator B(s.disp.get(), s.phi.get(), density, s.L,
                                       s.body);
  B.Assemble();
  check(B);

  PoissonDtNOperator C(s.phi.get(), s.L);
  C.Assemble();
  check(C);
}

INSTANTIATE_TEST_SUITE_P(Poisson, PoissonOperatorsTest, testing::Values(2, 3),
                         [](const testing::TestParamInfo<int>& info) {
                           return std::to_string(info.param) + "D";
                         });

}  // namespace
