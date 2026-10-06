#include <algorithm>
#include <random>

#include "MixedProblemTestCommon.hpp"
#include "TestCommon.hpp"

/*
  Tests for the density feasibility functional (equilibrium_figures.hpp)
  on the 3-D three-layer model (inner core, fluid outer core, mantle,
  buffer): the envelope-theorem derivative against central finite
  differences of the value, in the fluid-only variant (no-slip Stokes on
  the fluid SubMesh) and the weighted whole-body variant; the exact
  discrete quartic homogeneity of J in the density (the Euler identity
  rho . J' = 4 J, checkable because the test densities are polynomial
  and exactly representable in the control space); and the basic
  structure (J > 0 for a laterally varying, non-barotropic core; the
  potential and adjoint solves converge).

  The densities are polynomial by attribute: inner core 1.3, fluid
  1.2 - 0.3 r^2 + 0.1 y (non-barotropic through the lateral term),
  mantle 1.0, buffer 0. The FD step sits on the central-difference
  plateau; the tolerances allow for the quadrature differences between
  the load assembly inside the saddle and the derivative's own assembly
  on the curved elements.
*/

namespace {

using namespace self_grav_test;

constexpr int kDtNDegree = 8;
constexpr double kGravG = 0.05;

double FluidRho(const Vector& x) {
  const double r2 = x * x;
  return 1.2 - 0.3 * r2 + 0.1 * x[1];
}

// The density by attribute: polynomial within each region, so that it
// is exactly representable in an order-2 L2 control space.
class ModelDensity {
 public:
  ModelDensity()
      : inner_(1.3), fluid_(FluidRho), mantle_(1.0), buffer_(0.0) {
    pw_.UpdateCoefficient(1, inner_);
    pw_.UpdateCoefficient(2, fluid_);
    pw_.UpdateCoefficient(3, mantle_);
    pw_.UpdateCoefficient(4, buffer_);
  }
  Coefficient& Get() { return pw_; }

 private:
  ConstantCoefficient inner_, mantle_, buffer_;
  FunctionCoefficient fluid_;
  PWCoefficient pw_;
};

// One evaluation of the functional for a control perturbation c (a
// GridFunction on the control space over the Stokes SubMesh), in either
// variant. Returns the value; optionally the derivative dual.
struct ProblemSetup {
  Mesh* parent;
  SubMesh* stokes;
  FiniteElementSpace* fes_phi;
  FiniteElementSpace* fes_u;
  FiniteElementSpace* fes_p;
  FiniteElementSpace* fes_rho;        // control, on the Stokes mesh
  FiniteElementSpace* fes_rho_parent; // its parent twin
  Array<int> stokes_attributes;
  Coefficient* mu = nullptr;
  const Array<int>* essential_bdr = nullptr;
};

double Evaluate(const ProblemSetup& s, const GridFunction& c,
                Vector* dual = nullptr) {
  // The perturbation on the parent (element-local L2 transfer: exact).
  GridFunction c_parent(s.fes_rho_parent);
  c_parent = 0.0;
  SubMesh::Transfer(c, c_parent);
  GridFunctionCoefficient dc_parent(&c_parent);
  GridFunctionCoefficient dc_stokes(&c);

  ModelDensity rho0;
  SumCoefficient rho_parent(rho0.Get(), dc_parent);
  ModelDensity rho0_stokes;
  SumCoefficient rho_stokes(rho0_stokes.Get(), dc_stokes);

  DensityFeasibility J(*s.fes_phi, kDtNDegree, kGravG, rho_parent,
                       s.stokes_attributes, *s.fes_u, *s.fes_p, rho_stokes,
                       s.mu, s.essential_bdr);
  if (dual) {
    J.Derivative(*s.fes_rho, *dual);
  }
  return J.Value();
}

// Central-difference check of the dual against random directions.
// Returns the base value and dual, so a caller can reuse them; `trials`
// bounds the cost (each is two further solves of everything).
double CheckDerivative(const ProblemSetup& s, double step, double tol,
                       int trials, Vector& dual) {
  GridFunction c(s.fes_rho);
  c = 0.0;
  const double j0 = Evaluate(s, c, &dual);
  EXPECT_GT(j0, 0.0);

  std::mt19937 gen(1234);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (int trial = 0; trial < trials; trial++) {
    Vector d(s.fes_rho->GetTrueVSize());
    for (int i = 0; i < d.Size(); i++) {
      d[i] = dist(gen);
    }
    GridFunction cp(s.fes_rho), cm(s.fes_rho);
    cp = d;
    cp *= step;
    cm = d;
    cm *= -step;
    const double jp = Evaluate(s, cp);
    const double jm = Evaluate(s, cm);
    const double fd = (jp - jm) / (2.0 * step);
    const double predicted = dual * d;
    EXPECT_NEAR(fd, predicted,
                tol * std::max({std::abs(fd), std::abs(predicted), j0}))
        << "trial " << trial << ": fd " << fd << " vs dual.d " << predicted;
  }
  return j0;
}

}  // namespace

TEST(EquilibriumFigures, FluidOnlyDerivativeMatchesFD) {
  Mesh parent(ThreeLayerMeshFile(3), 1, 1);
  auto fluid = SubMesh::CreateFromDomain(parent, Array<int>({2}));

  H1_FECollection h1_phi(2, 3), h1_u(2, 3), h1_p(1, 3);
  L2_FECollection l2(2, 3);
  FiniteElementSpace fes_phi(&parent, &h1_phi);
  FiniteElementSpace fes_u(&fluid, &h1_u, 3);
  FiniteElementSpace fes_p(&fluid, &h1_p);
  FiniteElementSpace fes_rho(&fluid, &l2);
  FiniteElementSpace fes_rho_parent(&parent, &l2);

  // The fluid SubMesh inherits the ICB and CMB boundary attributes:
  // clamp the velocity on all of them (the fluid-only certificate).
  Array<int> ess(fluid.bdr_attributes.Max());
  ess = 1;

  ProblemSetup s{&parent,   &fluid,   &fes_phi,       &fes_u, &fes_p,
          &fes_rho,  &fes_rho_parent, Array<int>({2}), nullptr, &ess};
  Vector dual;
  CheckDerivative(s, 1e-3, 2e-5, 3, dual);
}

TEST(EquilibriumFigures, WholeBodyDerivativeAndEulerIdentity) {
  Mesh parent(ThreeLayerMeshFile(3), 1, 1);
  auto body = SubMesh::CreateFromDomain(parent, Array<int>({1, 2, 3}));

  H1_FECollection h1_phi(2, 3), h1_u(2, 3), h1_p(1, 3);
  L2_FECollection l2(2, 3);
  FiniteElementSpace fes_phi(&parent, &h1_phi);
  FiniteElementSpace fes_u(&body, &h1_u, 3);
  FiniteElementSpace fes_p(&body, &h1_p);
  FiniteElementSpace fes_rho(&body, &l2);
  FiniteElementSpace fes_rho_parent(&parent, &l2);

  // The weighted whole-body variant: the solid a hundred times as
  // viscous as the fluid, traction on the body surface.
  ConstantCoefficient mu_fluid(0.5), mu_solid(50.0);
  PWCoefficient mu;
  mu.UpdateCoefficient(1, mu_solid);
  mu.UpdateCoefficient(2, mu_fluid);
  mu.UpdateCoefficient(3, mu_solid);

  // One FD trial: this variant's saddle (viscosity contrast 100) is
  // the expensive one, and the fluid-only test already exercises the
  // derivative three times.
  ProblemSetup s{&parent,   &body,    &fes_phi,       &fes_u, &fes_p,
          &fes_rho,  &fes_rho_parent, Array<int>({1, 2, 3}), &mu, nullptr};
  Vector dual;
  const double j0 = CheckDerivative(s, 1e-3, 2e-5, 1, dual);

  // J is an exactly homogeneous discrete quartic in the density, so
  // rho . J' = 4 J; the model density is polynomial by attribute and
  // order 2, hence exactly representable in the control space.
  ModelDensity rho0;
  GridFunction rho_dofs(&fes_rho);
  rho_dofs.ProjectCoefficient(rho0.Get());
  const double euler = dual * rho_dofs;
  EXPECT_NEAR(euler, 4.0 * j0, 2e-5 * std::abs(j0))
      << "rho . J' = " << euler << " against 4 J = " << 4.0 * j0;
}
