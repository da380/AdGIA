#include <random>

#include "MixedProblemTestCommon.hpp"
#include "TestCommon.hpp"

/*
  Tests for the descent toolkit (descent.hpp) on an analytic toy
  problem over an L2 space: the quadratic J(c) = 1/2 c^T M c - f . c
  under the constraint ell . c = 0, whose constrained minimiser
  satisfies M c = f - mu ell — so at the solution the metric-projected
  gradient vanishes and the constraint holds. Checks: both loops reach
  that optimality system (projected gradient at the solver floor,
  constraint at round-off, value below the unconstrained start); the
  prior shifts the stationarity to (1 + lambda) M c = f - mu ell, which
  scales the solution by 1/(1 + lambda) when K = M; and
  Levenberg-Marquardt, whose Gauss-Newton model is exact for a
  quadratic, converges in a handful of outer iterations.
*/

namespace {

using namespace self_grav_test;

struct Toy {
  FiniteElementSpace* fes;
  SparseMatrix* M;
  Vector* f;

  double Value(const Vector& c) const {
    Vector Mc(c.Size());
    M->Mult(c, Mc);
    return 0.5 * (Mc * c) - (*f * c);
  }

  DescentFunctional Functional() const {
    DescentFunctional d;
    d.evaluate = [this](const Vector& c, Vector* dual) {
      if (dual) {
        dual->SetSize(c.Size());
        M->Mult(c, *dual);
        *dual -= *f;
      }
      return Value(c);
    };
    d.gauss_newton = [this](const Vector& x, Vector& Hx) {
      Hx.SetSize(x.Size());
      M->Mult(x, Hx);
    };
    return d;
  }
};

}  // namespace

TEST(Descent, LoopsReachTheConstrainedOptimum) {
  Mesh mesh("../data/elastogravity_two_layer_2d.msh", 1, 1);
  L2_FECollection l2(1, 2);
  FiniteElementSpace fes(&mesh, &l2);

  BilinearForm m(&fes);
  m.AddDomainIntegrator(new MassIntegrator());
  m.Assemble();
  m.Finalize();

  FunctionCoefficient fc(
      [](const Vector& x) { return 1.0 + x[0] + std::sin(2.0 * x[1]); });
  LinearForm flf(&fes);
  flf.AddDomainIntegrator(new DomainLFIntegrator(fc));
  flf.Assemble();
  Vector f(flf);

  ConstantCoefficient one(1.0);
  LinearForm ell_lf(&fes);
  ell_lf.AddDomainIntegrator(new DomainLFIntegrator(one));
  ell_lf.Assemble();
  Vector ell(ell_lf);

  Toy toy{&fes, &m.SpMat(), &f};
  L2RieszMap riesz(fes);
  ConstrainedMetric metric(riesz);
  metric.SetConstraint(ell);

  const double f_scale = std::sqrt(f * f);

  auto check = [&](const Vector& c, const char* what) {
    // The constraint holds, and the metric-projected gradient (the
    // optimality system's residual) is at the solver floor.
    EXPECT_LT(std::abs(ell * c), 1e-9 * f_scale) << what;
    Vector dual(c.Size()), g(c.Size());
    toy.Functional().evaluate(c, &dual);
    metric.Gradient(dual, g);
    Vector Mg(g.Size());
    m.SpMat().Mult(g, Mg);
    EXPECT_LT(std::sqrt(Mg * g), 1e-6 * f_scale) << what;
  };

  DescentOptions options;
  options.max_iterations = 400;
  options.tolerance = 0.0;  // disabled: run to the gradient floor

  {
    Vector c(fes.GetTrueVSize());
    c = 0.0;
    auto r = NonlinearCG(toy.Functional(), metric, c, options);
    EXPECT_GT(r.iterations, 0);
    EXPECT_LT(r.final, 0.0);  // below the start J(0) = 0
    check(c, "cg");
  }
  {
    Vector c(fes.GetTrueVSize());
    c = 0.0;
    auto r = LevenbergMarquardt(toy.Functional(), metric, m.SpMat(), c,
                                options);
    EXPECT_LT(r.iterations, 25);  // Gauss-Newton is exact: a few outers
    EXPECT_LT(r.final, 0.0);
    check(c, "lm");
  }
}

TEST(Descent, PriorScalesTheQuadraticSolution) {
  Mesh mesh("../data/elastogravity_two_layer_2d.msh", 1, 1);
  L2_FECollection l2(1, 2);
  FiniteElementSpace fes(&mesh, &l2);

  BilinearForm m(&fes);
  m.AddDomainIntegrator(new MassIntegrator());
  m.Assemble();
  m.Finalize();

  FunctionCoefficient fc(
      [](const Vector& x) { return x[0] - 0.3 * x[1]; });
  LinearForm flf(&fes);
  flf.AddDomainIntegrator(new DomainLFIntegrator(fc));
  flf.Assemble();
  Vector f(flf);

  ConstantCoefficient one(1.0);
  LinearForm ell_lf(&fes);
  ell_lf.AddDomainIntegrator(new DomainLFIntegrator(one));
  ell_lf.Assemble();
  Vector ell(ell_lf);

  Toy toy{&fes, &m.SpMat(), &f};
  L2RieszMap riesz(fes);

  const double lambda = 0.7;
  DescentOptions options;
  options.max_iterations = 400;
  options.tolerance = 0.0;

  Vector c_plain(fes.GetTrueVSize()), c_prior(fes.GetTrueVSize());
  c_plain = 0.0;
  c_prior = 0.0;
  {
    ConstrainedMetric metric(riesz);
    metric.SetConstraint(ell);
    LevenbergMarquardt(toy.Functional(), metric, m.SpMat(), c_plain,
                       options);
  }
  {
    ConstrainedMetric metric(riesz);
    metric.SetConstraint(ell);
    metric.SetPrior(m.SpMat(), lambda);
    LevenbergMarquardt(toy.Functional(), metric, m.SpMat(), c_prior,
                       options);
  }
  // With K = M the prior scales the constrained minimiser by
  // 1/(1 + lambda).
  Vector scaled(c_plain);
  scaled *= 1.0 / (1.0 + lambda);
  Vector diff(c_prior);
  diff -= scaled;
  EXPECT_LT(diff.Norml2() / scaled.Norml2(), 1e-5);
}
