/*
  Parallel tests for the pressure-split (u, p) solve
  (SetPressureSplitSolver). Run with 1, 2 and 4 ranks; a standalone MPI
  program that exits non-zero if any check fails.

  The checks are the partition-independent ones of the serial test:
  constant strain prescribed on the whole boundary is reproduced exactly
  (displacement and pressure p = kappa tr A alike, through MINRES on the
  block system with the AMG / weighted-mass block preconditioner); the
  modulus-switch chain (set, clear, re-set relaxation weights) stays
  consistent on a two-region composite with a 1e-6 weak-branch weight (the
  collapsed effective shear of dt >> tau), agreeing with the primal twin;
  and backward-Euler stepping under a prescribed uniform strain stays
  exact in space.
*/

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "QuasiStaticTestCommon.hpp"
#include "mfem.hpp"
#include "AdGIA.hpp"

using namespace mfem;
using namespace AdGIA;
using namespace elastic_test;

namespace {

int num_checks = 0;
int num_fails = 0;

double GlobalMax(double v) {
  double g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return g;
}

void Check(double err, double tol, const std::string& what) {
  num_checks++;
  if (!(err <= tol)) {
    num_fails++;
    if (Mpi::Root()) {
      std::cout << "FAIL: " << what << "  (err = " << err << ", tol = " << tol
                << ")\n";
    }
  }
}

DenseMatrix StrainMatrix(int dim) {
  DenseMatrix A(dim);
  A = 0.0;
  A(0, 0) = 0.01;
  A(0, 1) = 0.004;
  A(1, 0) = -0.002;
  A(1, 1) = -0.006;
  if (dim == 3) {
    A(2, 2) = 0.003;
    A(0, 2) = 0.005;
  }
  return A;
}

// The traction sigma n of a constant stress sigma on boundary elements.
class StressTraction : public VectorCoefficient {
 public:
  explicit StressTraction(const DenseMatrix& sigma)
      : VectorCoefficient(sigma.Height()), sigma_(sigma) {}
  void Eval(Vector& V, ElementTransformation& T,
            const IntegrationPoint& ip) override {
    Vector n(vdim);
    T.SetIntPoint(&ip);
    CalcOrtho(T.Jacobian(), n);
    n /= n.Norml2();
    V.SetSize(vdim);
    sigma_.Mult(n, V);
  }

 private:
  DenseMatrix sigma_;
};

double RelMaxDiff(const Vector& a, const Vector& b) {
  Vector d(a);
  d -= b;
  return GlobalMax(d.Normlinf()) / (GlobalMax(b.Normlinf()) + 1e-300);
}

void RunCase(int dim, int elementType, const std::string& label) {
  constexpr int kOrder = 2;
  auto smesh = MakeSmallMesh(dim, elementType);
  // Two horizontal layers for the composite leg.
  {
    Vector c;
    for (int e = 0; e < smesh.GetNE(); e++) {
      auto* T = smesh.GetElementTransformation(e);
      T->Transform(Geometries.GetCenter(T->GetGeometryType()), c);
      smesh.SetAttribute(e, c[dim - 1] < 0.5 ? 1 : 2);
    }
    smesh.SetAttributes();
  }
  const auto bottom_attr = BdrAttributeAt(smesh, dim - 1, 0.0);
  const auto top_attr = BdrAttributeAt(smesh, dim - 1, 1.0);
  const auto nbdr = smesh.bdr_attributes.Max();
  int nxyz[3] = {Mpi::WorldSize(), 1, 1};
  int* partitioning = smesh.CartesianPartitioning(nxyz);
  ParMesh pmesh(MPI_COMM_WORLD, smesh, partitioning);
  delete[] partitioning;
  H1_FECollection fec(kOrder, dim);
  ParFiniteElementSpace pfes(&pmesh, &fec, dim);
  ConstantCoefficient kappa(Kappa(dim)), mu(kMu);

  const DenseMatrix A = StrainMatrix(dim);
  VectorFunctionCoefficient dirichlet(
      dim, [&](const Vector& x, Vector& u) { A.Mult(x, u); });
  VectorFunctionCoefficient exact(
      dim, [&](const Vector& x, Vector& u) { A.Mult(x, u); });
  Vector zero(dim);
  zero = 0.0;
  VectorConstantCoefficient no_traction(zero);
  Array<int> all(nbdr), none(nbdr);
  all = 1;
  none = 0;

  // Constant strain: exact displacement and pressure.
  {
    IsotropicElasticRheology rheology(dim, kappa, mu);
    LinearQuasiStaticClampedProblem problem(&pfes, rheology, all, no_traction,
                                            none, &dirichlet);
    problem.SetPressureSplitSolver();
    problem.SetRelTol(1e-12);
    problem.AssembleForce(0.0);
    Check(problem.Solve() ? 0.0 : 1.0, 0.0, label + ": constant-strain solve");
    Check(GlobalMax(problem.Displacement().ComputeMaxError(exact)), 1e-9,
          label + ": constant-strain displacement");
    double tr = 0.0;
    for (int i = 0; i < dim; i++) {
      tr += A(i, i);
    }
    ConstantCoefficient p_exact(Kappa(dim) * tr);
    Check(GlobalMax(problem.Pressure()->ComputeMaxError(p_exact)),
          1e-9 * std::abs(Kappa(dim) * tr), label + ": constant-strain pressure");
  }

  // Modulus switches and the 1e-6 contrast on a two-region composite.
  {
    ConstantCoefficient mu_inf(0.0), mu1(kMu), mu2(0.6 * kMu);
    ConstantCoefficient tau1(1.0), tau2(2.0);
    std::vector<MaxwellBranch> b1{{&mu1, &tau1}}, b2{{&mu2, &tau2}};
    IsotropicMaxwellRheology r1(dim, kappa, mu_inf, b1);
    IsotropicMaxwellRheology r2(dim, kappa, mu_inf, b2);
    const int na = pmesh.attributes.Max();
    std::vector<RheologyRegion> regions{{Marker(na, {1}), &r1, "healthy"},
                                        {Marker(na, {2}), &r2, "weak"}};
    CompositeRheology rheology(dim, regions);
    VectorFunctionCoefficient pull(dim, PullTraction);
    const Array<int> clamp = Marker(nbdr, {bottom_attr});
    const Array<int> top = Marker(nbdr, {top_attr});
    auto make = [&](bool split_route) {
      auto p = std::make_unique<LinearQuasiStaticClampedProblem>(
          &pfes, rheology, clamp, pull, top);
      if (split_route) {
        p->SetPressureSplitSolver();
      }
      p->SetRelTol(1e-12);
      p->SetPrintLevel(IterativeSolver::PrintLevel().None());
      return p;
    };
    auto solve = [&](LinearQuasiStaticProblemBase& p, const std::string& what) {
      p.AssembleForce(0.0);
      Check(p.Solve() ? 0.0 : 1.0, 0.0, what);
      return Vector(p.Displacement());
    };
    auto split = make(true);
    auto primal = make(false);
    const Vector u0 = solve(*split, label + ": unrelaxed split solve");
    const Vector u0_primal = solve(*primal, label + ": unrelaxed primal solve");
    Check(RelMaxDiff(u0, u0_primal), 0.05, label + ": unrelaxed agreement");

    // Moderate contrast: both discretisations are sound and agree.
    ConstantCoefficient beta_healthy(1.0), beta_mild(0.3), beta_weak(1e-6);
    const std::vector<Coefficient*> mild{&beta_healthy, &beta_mild};
    split->SetRelaxationWeights(mild);
    primal->SetRelaxationWeights(mild);
    Check(RelMaxDiff(solve(*split, label + ": mild split solve"),
                     solve(*primal, label + ": mild primal solve")),
          0.10, label + ": mild-contrast agreement");

    // Extreme contrast: the primal discretisation locks (kappa/mu_eff ~
    // 1e6+), so robustness only; iteration counts feed solvers.md.
    const std::vector<Coefficient*> beta{&beta_healthy, &beta_weak};
    split->SetRelaxationWeights(beta);
    primal->SetRelaxationWeights(beta);
    const long its_before_split = split->TotalIterations();
    const long its_before_primal = primal->TotalIterations();
    const Vector ue = solve(*split, label + ": contrast split solve");
    solve(*primal, label + ": contrast primal solve");
    if (Mpi::Root()) {
      std::cout << label << ": contrast 1e-6 iterations, split "
                << split->TotalIterations() - its_before_split << ", primal "
                << primal->TotalIterations() - its_before_primal << "\n";
    }

    split->ClearRelaxationWeights();
    Check(RelMaxDiff(solve(*split, label + ": cleared solve"), u0), 1e-5,
          label + ": cleared weights match unrelaxed");
    split->SetRelaxationWeights(beta);
    Check(RelMaxDiff(solve(*split, label + ": re-set solve"), ue), 1e-5,
          label + ": re-set weights match effective");
  }

  // A purely elastic region beside a viscoelastic one (the production
  // shape): the composite split carries the elastic lid's fixed moduli
  // next to the Maxwell mantle's live effective one.
  {
    ConstantCoefficient mu_inf(0.0), mu1(kMu), tau1(1.0), mu_lid(1.5 * kMu);
    std::vector<MaxwellBranch> b1{{&mu1, &tau1}};
    IsotropicMaxwellRheology mantle(dim, kappa, mu_inf, b1);
    IsotropicElasticRheology lid(dim, kappa, mu_lid);
    const int na = pmesh.attributes.Max();
    std::vector<RheologyRegion> regions{{Marker(na, {1}), &mantle, "mantle"},
                                        {Marker(na, {2}), &lid, "lid"}};
    CompositeRheology rheology(dim, regions);
    VectorFunctionCoefficient pull(dim, PullTraction);
    const Array<int> clamp = Marker(nbdr, {bottom_attr});
    const Array<int> top = Marker(nbdr, {top_attr});
    LinearQuasiStaticClampedProblem split(&pfes, rheology, clamp, pull, top);
    split.SetPressureSplitSolver();
    split.SetRelTol(1e-12);
    LinearQuasiStaticClampedProblem primal(&pfes, rheology, clamp, pull, top);
    primal.SetRelTol(1e-12);
    auto solve = [&](LinearQuasiStaticProblemBase& p, const std::string& what) {
      p.AssembleForce(0.0);
      Check(p.Solve() ? 0.0 : 1.0, 0.0, what);
      return Vector(p.Displacement());
    };
    const Vector u0 = solve(split, label + ": mixed-composite split solve");
    Check(RelMaxDiff(u0, solve(primal, label + ": mixed-composite primal")),
          0.10, label + ": mixed-composite agreement");
    ConstantCoefficient beta(0.3);  // mild: the primal does not lock
    const std::vector<Coefficient*> weights{&beta};
    split.SetRelaxationWeights(weights);
    primal.SetRelaxationWeights(weights);
    // 0.2: the softened mantle sharpens the interface pressure jump the
    // continuous Taylor-Hood pressure smears on this coarse mesh.
    Check(RelMaxDiff(solve(split, label + ": mixed relaxed split"),
                     solve(primal, label + ": mixed relaxed primal")),
          0.20, label + ": mixed-composite relaxed agreement");
    split.ClearRelaxationWeights();
    Check(RelMaxDiff(solve(split, label + ": mixed cleared solve"), u0), 1e-5,
          label + ": mixed cleared weights match unrelaxed");
  }

  // Free body: a uniform stress applied as the traction on the whole
  // boundary of the unclamped body. The strain is uniform, so the split
  // (rigid modes projected out of its displacement block) and the primal
  // route agree to solver precision in the same rigid-mode gauge, and the
  // pressure is exact: (kappa - theta 2 mu / d) tr eps.
  {
    DenseMatrix eps(dim), sigma(dim);
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        eps(i, j) = 0.5 * (A(i, j) + A(j, i));
      }
    }
    double tr = 0.0;
    for (int i = 0; i < dim; i++) {
      tr += eps(i, i);
    }
    const double lambda = Kappa(dim) - 2.0 * kMu / dim;
    sigma = eps;
    sigma *= 2.0 * kMu;
    for (int i = 0; i < dim; i++) {
      sigma(i, i) += lambda * tr;
    }
    StressTraction traction(sigma);
    IsotropicElasticRheology rheology(dim, kappa, mu);
    LinearQuasiStaticTractionProblem primal(&pfes, rheology, traction, all);
    primal.SetRelTol(1e-12);
    primal.AssembleForce(0.0);
    Check(primal.Solve() ? 0.0 : 1.0, 0.0, label + ": free-body primal solve");
    const Vector u_primal(primal.Displacement());
    for (const double theta : {0.0, 1.0}) {
      const std::string t = " (theta " + std::to_string(int(theta)) + ")";
      LinearQuasiStaticTractionProblem split(&pfes, rheology, traction, all);
      split.SetPressureSplitSolver(theta);
      split.SetRelTol(1e-12);
      split.AssembleForce(0.0);
      Check(split.Solve() ? 0.0 : 1.0, 0.0, label + ": free-body split solve" + t);
      Check(RelMaxDiff(split.Displacement(), u_primal), 1e-8,
            label + ": free-body split matches primal" + t);
      const double p = (Kappa(dim) - theta * 2.0 * kMu / dim) * tr;
      ConstantCoefficient p_exact(p);
      Check(GlobalMax(split.Pressure()->ComputeMaxError(p_exact)),
            1e-8 * std::abs(p), label + ": free-body pressure" + t);
    }
  }

  // Per-region pressure (one continuous pressure per element attribute):
  // exact under a prescribed uniform strain, like Taylor-Hood.
  {
    IsotropicElasticRheology rheology(dim, kappa, mu);
    LinearQuasiStaticClampedProblem problem(&pfes, rheology, all, no_traction,
                                            none, &dirichlet);
    problem.SetPressureSplitPerRegion();
    problem.SetPressureSplitSolver();
    problem.SetRelTol(1e-12);
    problem.AssembleForce(0.0);
    Check(problem.Solve() ? 0.0 : 1.0, 0.0, label + ": per-region solve");
    Check(GlobalMax(problem.Displacement().ComputeMaxError(exact)), 1e-9,
          label + ": per-region displacement");
    double tr = 0.0;
    for (int i = 0; i < dim; i++) {
      tr += A(i, i);
    }
    ConstantCoefficient p_exact(Kappa(dim) * tr);
    Check(GlobalMax(problem.Pressure()->ComputeMaxError(p_exact)),
          1e-9 * std::abs(Kappa(dim) * tr), label + ": per-region pressure");
  }

  // A shear-modulus jump across the layer interface, through the
  // per-region pressure (and, for comparison, Taylor-Hood): both agree
  // with the primal solution at discretisation-error level. (Taylor-Hood
  // smears the interface pressure jump, a first-order error the
  // per-region pressure removes — measured on the box benchmark's
  // refinement ladders; on these small meshes the clamped corners
  // dominate both gaps, so the gaps are only printed.) The region-mean
  // coarse correction of the Schur block changes the preconditioner
  // only, not the solution.
  {
    Vector mu_layers({kMu, 0.05 * kMu});
    PWConstCoefficient mu_jump(mu_layers);
    IsotropicElasticRheology rheology(dim, kappa, mu_jump);
    VectorFunctionCoefficient pull(dim, PullTraction);
    const Array<int> clamp = Marker(nbdr, {bottom_attr});
    const Array<int> top = Marker(nbdr, {top_attr});
    auto run = [&](bool split, bool region, bool coarse, const std::string& w) {
      LinearQuasiStaticClampedProblem p(&pfes, rheology, clamp, pull, top);
      if (split) {
        p.SetPressureSplitPerRegion(region);
        p.SetPressureSplitCoarse(coarse);
        p.SetPressureSplitSolver(1.0);
      }
      p.SetRelTol(1e-12);
      p.AssembleForce(0.0);
      Check(p.Solve() ? 0.0 : 1.0, 0.0, label + ": " + w + " solve");
      return Vector(p.Displacement());
    };
    const Vector u_primal = run(false, false, false, "jump primal");
    const Vector u_th = run(true, false, false, "jump Taylor-Hood");
    const Vector u_region = run(true, true, false, "jump per-region");
    const Vector u_coarse = run(true, true, true, "jump per-region coarse");
    const double gap_th = RelMaxDiff(u_th, u_primal);
    const double gap_region = RelMaxDiff(u_region, u_primal);
    if (Mpi::Root()) {
      std::cout << label << ": shear-jump gap to primal, Taylor-Hood "
                << gap_th << ", per-region " << gap_region << "\n";
    }
    Check(gap_th, 0.25, label + ": shear-jump Taylor-Hood agreement");
    Check(gap_region, 0.25, label + ": shear-jump per-region agreement");
    Check(RelMaxDiff(u_coarse, u_region), 1e-8,
          label + ": coarse correction leaves the solution unchanged");
  }

  // Backward-Euler stepping under a prescribed uniform strain: exact in
  // space through the split route at every step.
  {
    ConstantCoefficient mu_inf(0.3), mu1(0.7), tau1(1.0);
    std::vector<MaxwellBranch> branches{{&mu1, &tau1}};
    IsotropicMaxwellRheology rheology(dim, kappa, mu_inf, branches);
    LinearQuasiStaticClampedProblem problem(&pfes, rheology, all, no_traction,
                                            none, &dirichlet);
    problem.SetPressureSplitSolver();
    problem.SetRelTol(1e-12);
    ViscoelasticOperator visco(problem);
    BackwardEulerSolver ode;
    ode.Init(visco);
    Vector m(visco.Height());
    m = 0.0;
    double t = 0.0, dt = 0.5;
    for (int step = 0; step < 4; step++) {
      ode.Step(m, t, dt);
    }
    Check(visco.SolveElastic(m, t) ? 0.0 : 1.0, 0.0,
          label + ": stepping solve");
    Check(GlobalMax(problem.Displacement().ComputeMaxError(exact)), 1e-9,
          label + ": stepping displacement");
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();

  RunCase(2, 1, "2D tensor");
  RunCase(2, 0, "2D simplex");
  RunCase(3, 1, "3D tensor");

  if (Mpi::Root()) {
    std::cout << (num_fails == 0 ? "OK" : "FAILED") << ": " << num_checks
              << " checks, " << num_fails << " failures on "
              << Mpi::WorldSize() << " ranks\n";
  }
  return num_fails == 0 ? 0 : 1;
}
