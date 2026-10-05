#include "TestCommon.hpp"

/*
  Tests for the relaxation laws of relaxation_law.hpp:

  - LocalState::DeviatoricNorm against a direct tensor computation.
  - PowerLawRelaxation: the factor 1 / (1 + gamma (|T| / 2 mu0)^(n-1)),
    its linear limits (gamma = 0, n = 1), the transition stress tau_e =
    2 mu0 gamma^(-1/(n-1)) where the two mechanisms contribute equally, and
    the gradient against central finite differences.
*/

namespace {

DenseMatrix Tensor(int dim, const Vector& c) {
  DenseMatrix A(dim);
  for (int s = 0; s < c.Size(); s++) {
    int j, k;
    SymmetricTensorBasis::Component(dim, s, j, k);
    A(j, k) = A(k, j) = c[s];
  }
  return A;
}

class RelaxationLawTest : public testing::TestWithParam<int> {};

TEST_P(RelaxationLawTest, DeviatoricNorm) {
  const int dim = GetParam();
  const int ns = SymmetricTensorBasis::Size(dim);
  Vector c = RandomVector(ns);
  DenseMatrix A = Tensor(dim, c);
  const double tr = A.Trace();
  double s = 0.0;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      const double d = A(i, j) - (i == j ? tr / dim : 0.0);
      s += d * d;
    }
  }
  EXPECT_NEAR(LocalState::DeviatoricNorm(dim, c.GetData()), std::sqrt(s),
              1e-14);
  LocalState state(dim);
  state.stress = c;
  EXPECT_NEAR(state.DeviatoricStressNorm(), std::sqrt(s), 1e-14);
}

TEST_P(RelaxationLawTest, PowerLaw) {
  const int dim = GetParam();
  const int ns = SymmetricTensorBasis::Size(dim);
  ConstantCoefficient gamma(2.0), n(3.0), mu0(0.8);
  PowerLawRelaxation law(gamma, n, mu0);
  EXPECT_TRUE(law.IsStateDependent());
  EXPECT_EQ(law.NumParameters(), 3);
  EXPECT_EQ(&law.Parameter(0), &gamma);
  EXPECT_EQ(&law.Parameter(2), &mu0);
  EXPECT_TRUE(law.HasGradient());

  LocalState s(dim);
  s.stress = RandomVector(ns);
  const double T = s.DeviatoricStressNorm();
  const double params[3] = {2.0, 3.0, 0.8};
  const double expected = 1.0 / (1.0 + 2.0 * std::pow(T / 1.6, 2.0));
  EXPECT_NEAR(law.Factor(params, s), expected, 1e-14);

  // Linear limits.
  const double p_gamma0[3] = {0.0, 3.0, 0.8}, p_n1[3] = {2.0, 1.0, 0.8};
  EXPECT_EQ(law.Factor(p_gamma0, s), 1.0);
  EXPECT_EQ(law.Factor(p_n1, s), 1.0);

  // At |T| = tau_e = 2 mu0 gamma^(-1/(n-1)) the factor is 1/2.
  const double tau_e = 1.6 * std::pow(2.0, -0.5);
  s.stress *= tau_e / T;
  EXPECT_NEAR(law.Factor(params, s), 0.5, 1e-13);

  // Gradient against central differences.
  Vector g;
  law.Gradient(params, s, g);
  ASSERT_EQ(g.Size(), ns);
  const double h = 1e-6;
  for (int i = 0; i < ns; i++) {
    LocalState sp(s), sm(s);
    sp.stress[i] += h;
    sm.stress[i] -= h;
    const double fd = (law.Factor(params, sp) - law.Factor(params, sm)) /
                      (2.0 * h);
    EXPECT_NEAR(g[i], fd, 1e-7) << "component " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(RelaxationLaw, RelaxationLawTest,
                         testing::Values(2, 3));

}  // namespace
