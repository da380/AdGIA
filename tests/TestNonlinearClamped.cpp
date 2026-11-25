#include <cstdlib>

#include "NonlinearTestCommon.hpp"
#include "TestCommon.hpp"

/*
  Tests for NonlinearClampedProblem (nonlinear_problem.hpp), serial: a
  square or cube clamped on its bottom and loaded on its top, of a
  material that varies over the body, with an equilibrium stress.

  - The residual is the first variation of the energy and the tangent the
    first variation of the residual, by central differences at a state
    that is not the reference.
  - An affine patch: with an affine displacement prescribed on the whole
    boundary of a homogeneous body, the solution is that affine
    displacement, at finite strain.
  - The linear limit: the first iteration from the reference is the
    solution of the linear clamped problem with the law's tensor and
    stress at the reference, and the difference between the nonlinear
    solution and that one falls as the load squared.
  - With the tangent assembled at every iteration the corrections
    converge quadratically; with the tangent of the reference kept they
    contract at a rate proportional to the load.
  - Without loads the reference is the solution whatever the equilibrium
    stress.
*/

namespace {

using namespace nonlinear_test;

using Param = std::tuple<int, int, int>;  // dim, order, element type

// Diagnostic output of the passing tests is printed only when the
// environment variable ADGIA_TEST_VERBOSE is set.
bool Verbose() { return std::getenv("ADGIA_TEST_VERBOSE") != nullptr; }

real_t L2Norm(const GridFunction& u) {
  auto zero = Vector(u.FESpace()->GetVDim());
  zero = 0.0;
  auto z = VectorConstantCoefficient(zero);
  return u.ComputeL2Error(z);
}

// The body, its space and markers, and a problem on them.
struct Body {
  Mesh mesh;
  H1_FECollection fec;
  FiniteElementSpace fes;
  Material material;
  Array<int> bottom, top, none, all;

  Body(int dim, int order, int type, int n)
      : mesh(MakeBody(dim, type, n)),
        fec(order, dim),
        fes(&mesh, &fec, dim),
        material(dim),
        bottom(Marker(mesh, 1)),
        top(Marker(mesh, TopAttribute(dim))),
        none(Marker(mesh, 0)),
        all(mesh.bdr_attributes.Max()) {
    all = 1;
  }
};

class NonlinearClampedTest : public testing::TestWithParam<Param> {};

TEST_P(NonlinearClampedTest, ResidualAndTangentAreDerivatives) {
  const auto [dim, order, type] = GetParam();
  auto body = Body(dim, order, type, dim == 2 ? 4 : 2);
  auto traction = VectorConstantCoefficient(TopTraction(dim, 0.05));
  auto force = VectorFunctionCoefficient(
      dim, [](const Vector& x, Vector& f) { SmoothField(x, f, 0.03, 2.0); });
  auto problem = NonlinearClampedProblem(&body.fes, body.material.cubic,
                                         body.bottom, traction, body.top);
  problem.SetBodyForce(force);

  // A state, and two directions that vanish on the clamped part.
  auto state = VectorFunctionCoefficient(
      dim, [](const Vector& x, Vector& u) { SmoothField(x, u, 0.04, 0.3); });
  auto zero_vector = Vector(dim);
  zero_vector = 0.0;
  auto zero = VectorConstantCoefficient(zero_vector);
  auto u0 = GridFunction(&body.fes);
  u0.ProjectCoefficient(state);
  auto v = GridFunction(&body.fes), w = GridFunction(&body.fes);
  auto v_c = VectorFunctionCoefficient(
      dim, [](const Vector& x, Vector& u) { SmoothField(x, u, 1.0, 1.1); });
  auto w_c = VectorFunctionCoefficient(
      dim, [](const Vector& x, Vector& u) { SmoothField(x, u, 1.0, 2.7); });
  v.ProjectCoefficient(v_c);
  w.ProjectCoefficient(w_c);
  v.ProjectBdrCoefficient(zero, body.bottom);
  w.ProjectBdrCoefficient(zero, body.bottom);

  // Derivatives along a direction by the fourth-order central
  // difference, which leaves only rounding at this step.
  auto& u = problem.Displacement();
  const auto h = 1e-4;
  auto derivative = [&](const GridFunction& direction, auto&& value) {
    auto at = [&](real_t step) {
      u = u0;
      u.Add(step, direction);
      return value();
    };
    return (8.0 * (at(h) - at(-h)) - (at(2.0 * h) - at(-2.0 * h))) /
           (12.0 * h);
  };

  u = u0;
  auto r = Vector();
  problem.Residual(r);
  const auto dE = derivative(v, [&] { return problem.Energy(); });
  EXPECT_NEAR(dE, r * v, 1e-8 * std::abs(dE));

  // v^T K w with the eliminated matrix, v and w vanishing on the clamped
  // part.
  u = u0;
  const auto& K = *problem.Tangent().SystemMatrix().As<SparseMatrix>();
  auto Kw = Vector(w.Size());
  K.Mult(w, Kw);
  const auto vKw = v * Kw;
  const auto dr = derivative(w, [&] {
    auto at = Vector();
    problem.Residual(at);
    return at * v;
  });
  EXPECT_NEAR(dr, vKw, 1e-8 * std::abs(vKw));
  // The tangent is symmetric.
  auto Kv = Vector(v.Size());
  K.Mult(v, Kv);
  EXPECT_NEAR(w * Kv, vKw, 1e-11 * std::abs(vKw));
}

TEST_P(NonlinearClampedTest, AffinePatch) {
  const auto [dim, order, type] = GetParam();
  auto body = Body(dim, order, type, dim == 2 ? 4 : 2);
  // A homogeneous law, with a uniform hydrostatic equilibrium stress and
  // the bare tensor that goes with it.
  auto lambda = ConstantCoefficient(1.2), mu = ConstantCoefficient(0.9);
  auto pressure = ConstantCoefficient(0.3);
  auto C = IsotropicElasticTensorCoefficient(dim, lambda, mu);
  auto Xi = BareElasticTensorCoefficient(dim, C, pressure);
  auto stress = DenseMatrix(dim);
  stress = 0.0;
  for (auto j = 0; j < dim; j++) {
    stress(j, j) = -0.3;
  }
  auto T0 = MatrixConstantCoefficient(stress);
  auto law = QuadraticMaterialLaw(dim, Xi, &T0);

  auto A = DenseMatrix(dim);
  for (auto i = 0; i < dim; i++) {
    for (auto j = 0; j < dim; j++) {
      A(i, j) = 0.12 * std::sin(0.4 + 1.9 * i + 0.8 * j);
    }
  }
  auto affine = VectorFunctionCoefficient(
      dim, [&A](const Vector& x, Vector& u) { A.Mult(x, u); });
  auto no_traction_value = Vector(dim);
  no_traction_value = 0.0;
  auto no_traction = VectorConstantCoefficient(no_traction_value);
  auto problem = NonlinearClampedProblem(&body.fes, law, body.all, no_traction,
                                         body.none, &affine);
  problem.Options().refresh_contraction = 0.0;
  ASSERT_TRUE(problem.Solve());
  EXPECT_LT(problem.Displacement().ComputeL2Error(affine), 1e-9);
  EXPECT_GT(L2Norm(problem.Displacement()), 0.05);
}

TEST_P(NonlinearClampedTest, LinearLimit) {
  const auto [dim, order, type] = GetParam();
  auto body = Body(dim, order, type, dim == 2 ? 4 : 2);
  auto& law = body.material.cubic;

  // The linear problem about the reference: the law's tensor and stress
  // there, the identity mapping, a unit load.
  auto identity = IdentityDiffeomorphism(dim);
  auto rheology = ReferentialElasticRheology(dim, body.material.Xi,
                                             body.material.T0, identity);
  auto unit = VectorConstantCoefficient(TopTraction(dim, 1.0));
  auto linear = LinearQuasiStaticClampedProblem(&body.fes, rheology,
                                                body.bottom, unit, body.top);
  // The quadrature of the nonlinear problem, which differs from the
  // linear class's own where the material varies over an element.
  {
    auto* T = body.mesh.GetElementTransformation(0);
    const auto* fe = body.fes.GetFE(0);
    const auto& rule =
        IntRules.Get(fe->GetGeomType(), 2 * fe->GetOrder() + T->OrderW());
    for (auto* integrator : *linear.StiffnessIntegrators().GetDBFI()) {
      integrator->SetIntRule(&rule);
    }
  }
  linear.AssembleForce(0.0);
  ASSERT_TRUE(linear.Solve());
  const auto scale = L2Norm(linear.Displacement());

  auto difference = [&](real_t load, int iterations) {
    auto traction = VectorConstantCoefficient(TopTraction(dim, load));
    auto problem = NonlinearClampedProblem(&body.fes, law, body.bottom,
                                           traction, body.top);
    problem.Options().max_iterations = iterations;
    const auto converged = problem.Solve();
    EXPECT_EQ(converged, iterations > 1);
    auto d = GridFunction(&body.fes);
    d = linear.Displacement();
    d *= load;
    d -= problem.Displacement();
    return L2Norm(d) / (load * scale);
  };

  // The first iteration is the linear solution.
  EXPECT_LT(difference(1e-3, 1), 1e-9);
  // The nonlinear correction is of second order in the load.
  const auto coarse = difference(2e-3, 25);
  const auto fine = difference(1e-3, 25);
  EXPECT_GT(fine, 1e-6);
  EXPECT_NEAR(coarse / fine, 2.0, 0.05);
}

TEST_P(NonlinearClampedTest, Convergence) {
  const auto [dim, order, type] = GetParam();
  auto body = Body(dim, order, type, dim == 2 ? 4 : 2);
  auto& law = body.material.cubic;

  auto solve = [&](real_t load, real_t refresh, std::vector<real_t>& steps,
                   int& assemblies) {
    auto traction = VectorConstantCoefficient(TopTraction(dim, load));
    auto problem = NonlinearClampedProblem(&body.fes, law, body.bottom,
                                           traction, body.top);
    problem.Options().refresh_contraction = refresh;
    problem.Options().max_iterations = 60;
    problem.Options().print = Verbose();
    const auto converged = problem.Solve();
    steps = problem.Corrections();
    assemblies = problem.TangentAssemblies();
    return converged ? L2Norm(problem.Displacement()) : -1.0;
  };

  // Newton's method proper: each correction is bounded by a multiple of
  // the square of the one before, until rounding.
  auto steps = std::vector<real_t>();
  auto assemblies = 0;
  const auto size = solve(0.04, 0.0, steps, assemblies);
  ASSERT_GT(size, 0.0);
  EXPECT_EQ(assemblies, static_cast<int>(steps.size()));
  EXPECT_LE(steps.size(), 7u);
  for (auto k = 1u; k < steps.size(); k++) {
    if (steps[k] > 1e-11 * size) {
      EXPECT_LT(steps[k], 30.0 * steps[k - 1] * steps[k - 1] / size)
          << "iteration " << k + 1;
    }
  }

  // The tangent of the reference kept: one assembly, and a contraction
  // proportional to the load.
  auto contraction = [&](real_t load) {
    auto kept = std::vector<real_t>();
    auto count = 0;
    const auto norm = solve(load, 1e30, kept, count);
    EXPECT_GT(norm, 0.0);
    EXPECT_EQ(count, 1);
    EXPECT_GE(kept.size(), 4u);
    return kept.size() >= 4 ? kept[2] / kept[1] : 0.0;
  };
  const auto small = contraction(0.01);
  const auto large = contraction(0.02);
  EXPECT_GT(small, 0.0);
  EXPECT_LT(large, 0.5);
  EXPECT_NEAR(large / small, 2.0, 0.3);

  // The default: the tangent of the reference is kept while it serves.
  // At a small load it serves throughout; at a larger one it is given
  // up, and the answer is the same as Newton's.
  auto mixed = std::vector<real_t>();
  auto count = 0;
  EXPECT_GT(solve(0.002, 0.1, mixed, count), 0.0);
  EXPECT_EQ(count, 1);
  const auto same = solve(0.04, 0.1, mixed, count);
  EXPECT_NEAR(same, size, 1e-8 * size);
  EXPECT_GT(count, 1);
  EXPECT_LE(count, static_cast<int>(mixed.size()));
}

TEST_P(NonlinearClampedTest, ReferenceIsASolutionWithoutLoads) {
  const auto [dim, order, type] = GetParam();
  auto body = Body(dim, order, type, dim == 2 ? 4 : 2);
  auto no_traction_value = Vector(dim);
  no_traction_value = 0.0;
  auto no_traction = VectorConstantCoefficient(no_traction_value);
  auto problem = NonlinearClampedProblem(&body.fes, body.material.cubic,
                                         body.bottom, no_traction, body.top);
  auto r = Vector();
  problem.Residual(r);
  EXPECT_EQ(r.Normlinf(), 0.0);
  EXPECT_EQ(problem.Energy(), 0.0);
  ASSERT_TRUE(problem.Solve());
  EXPECT_EQ(problem.Iterations(), 1);
  EXPECT_EQ(L2Norm(problem.Displacement()), 0.0);
}

// A manufactured solution. With u* a smooth displacement, the dead load
// given by the stress P(F*) - S0 and u* prescribed on the clamped part,
// the solution is u*: the residual at u* is int (P(F*) - S0 - M) : grad v
// for every v, with M the load. The error of the discrete solution then
// falls at the rate of the interpolation, order + 1 in L2.
//
// With u* prescribed on the whole boundary, at strains of a tenth. And
// with the bottom alone clamped, where the load's action on the free
// boundary is its traction, at strains of a hundredth: at the larger
// amplitude that body under that dead load is not stable (the iteration
// wanders), which is the body's and not the method's.
real_t ManufacturedRate(int dim, int order, int type, real_t amplitude,
                        bool all_prescribed, real_t* finest = nullptr) {
  auto exact = SmoothConfiguration(dim, amplitude, 0.3);
  auto u_exact = VectorFunctionCoefficient(
      dim, [amplitude](const Vector& x, Vector& u) {
        SmoothField(x, u, amplitude, 0.3);
      });
  auto identity = IdentityDiffeomorphism(dim);
  auto errors = std::vector<real_t>();
  for (const auto n :
       dim == 2 ? std::vector<int>{4, 8} : std::vector<int>{2, 4}) {
    auto body = Body(dim, order, type, n);
    auto load = FirstPiolaKirchhoffStressCoefficient(body.material.cubic,
                                                     exact, &identity);
    auto no_traction_value = Vector(dim);
    no_traction_value = 0.0;
    auto no_traction = VectorConstantCoefficient(no_traction_value);
    auto problem = NonlinearClampedProblem(
        &body.fes, body.material.cubic,
        all_prescribed ? body.all : body.bottom, no_traction, body.none,
        &u_exact);
    problem.SetStressLoad(load);
    problem.Options().refresh_contraction = 0.0;
    EXPECT_TRUE(problem.Solve());
    errors.push_back(problem.Displacement().ComputeL2Error(u_exact));
  }
  if (finest) {
    *finest = errors[1];
  }
  return std::log2(errors[0] / errors[1]);
}

TEST(NonlinearClampedManufactured, ConvergesAtTheOptimalRate) {
  for (const auto& [dim, order, type] :
       {Param{2, 1, 0}, Param{2, 1, 1}, Param{2, 2, 0}, Param{2, 2, 1},
        Param{3, 1, 0}}) {
    auto finest = real_t{0};
    const auto rate = ManufacturedRate(dim, order, type, 0.03, true, &finest);
    EXPECT_GT(rate, order + 0.8)
        << "dim " << dim << " order " << order << " type " << type;
    EXPECT_LT(finest, dim == 3 ? 6e-3 : (order == 1 ? 1e-3 : 2e-5));
  }
}

TEST(NonlinearClampedManufactured, WithAFreeBoundary) {
  for (const auto type : {0, 1}) {
    EXPECT_GT(ManufacturedRate(2, 2, type, 0.003, false), 2.8)
        << "type " << type;
  }
}

// The same body described from a relabelled reference: the mesh is taken
// to the body by a mapping that moves its interior and leaves its
// boundary, the material is the body's at the image of each point, and
// the load is on the boundary. The displacement of a point of the body is
// then the same; compared through the mean displacement of the top and
// the integral of its square over the body, which fall together as the
// mesh is refined.
TEST(NonlinearClampedRelabelled, SameBodyThroughAnotherMapping) {
  for (const auto dim : {2, 3}) {
    const auto order = 2;
    auto relabelling = InteriorRelabelling(dim, 0.08);
    auto traction = VectorConstantCoefficient(TopTraction(dim, 0.05));
    auto differences = std::vector<real_t>();
    for (const auto n : dim == 2 ? std::vector<int>{4, 8}
                                 : std::vector<int>{2, 4}) {
      auto body = Body(dim, order, 0, n);
      auto measure = [&](NonlinearClampedProblem& problem, Coefficient& J) {
        auto values = std::vector<real_t>();
        // The mean displacement of the top, component by component.
        for (auto i = 0; i < dim; i++) {
          auto e = Vector(dim);
          e = 0.0;
          e[i] = 1.0;
          auto e_c = VectorConstantCoefficient(e);
          auto top = LinearForm(&body.fes);
          top.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(e_c),
                                    body.top);
          top.Assemble();
          values.push_back(top(problem.Displacement()));
        }
        // int |u|^2 over the body, the mesh's volume element times J.
        auto square = real_t{0};
        auto u = Vector();
        for (auto e = 0; e < body.mesh.GetNE(); e++) {
          auto* T = body.mesh.GetElementTransformation(e);
          const auto& rule = IntRules.Get(T->GetGeometryType(), 2 * order + 4);
          for (auto q = 0; q < rule.GetNPoints(); q++) {
            const auto& ip = rule.IntPoint(q);
            T->SetIntPoint(&ip);
            problem.Displacement().GetVectorValue(*T, ip, u);
            square += ip.weight * T->Weight() * J.Eval(*T, ip) * (u * u);
          }
        }
        values.push_back(std::sqrt(square));
        return values;
      };

      auto one = ConstantCoefficient(1.0);
      auto plain = NonlinearClampedProblem(&body.fes, body.material.cubic,
                                           body.bottom, traction, body.top);
      EXPECT_TRUE(plain.Solve());
      const auto a = measure(plain, one);

      auto material = RelabelledMaterial(dim, relabelling);
      auto jacobian = JacobianCoefficient(relabelling);
      auto relabelled =
          NonlinearClampedProblem(&body.fes, material.cubic, body.bottom,
                                  traction, body.top, nullptr, &relabelling);
      EXPECT_TRUE(relabelled.Solve());
      const auto b = measure(relabelled, jacobian);

      auto worst = real_t{0};
      for (auto k = 0u; k < a.size(); k++) {
        worst = std::max(worst, std::abs(a[k] - b[k]) / a.back());
      }
      differences.push_back(worst);
    }
    EXPECT_LT(differences[1], 0.5 * differences[0])
        << "dim " << dim << ": " << differences[0] << ", " << differences[1];
    EXPECT_LT(differences[1], dim == 2 ? 2e-4 : 2e-3);
  }
}

INSTANTIATE_TEST_SUITE_P(
    NonlinearClamped, NonlinearClampedTest,
    testing::Values(Param{2, 1, 0}, Param{2, 2, 1}, Param{2, 3, 0},
                    Param{3, 1, 1}, Param{3, 2, 0}));

}  // namespace
