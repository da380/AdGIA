/*
  Parallel tests for CompositeRheology on ParFiniteElementSpaces. Run with
  1, 2 and 4 ranks; a standalone MPI program that exits with status 1 if
  any check fails.

  The unit square/cube of the quasi-static tests is split into two
  attribute regions (x < 0.4 and x > 0.4) and partitioned into slabs along
  x, so that a region may be absent from a rank.
  - A Maxwell body split into two regions with the same rheology equals the
    unsplit body (exponential trapezoid; displacement and internal
    variables).
  - An elastic region beside a Maxwell region equals the global Maxwell
    body with a piecewise branch modulus.
  - Two elastic regions assemble the action of the piecewise-moduli
    elastic stiffness.
*/

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>

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

constexpr double kSplit = 0.4;

void SplitAttributes(Mesh& mesh) {
  Vector c;
  for (int e = 0; e < mesh.GetNE(); e++) {
    mesh.GetElementCenter(e, c);
    mesh.SetAttribute(e, c[0] < kSplit ? 1 : 2);
  }
  mesh.SetAttributes();
}

Array<int> Region(int attr) {
  Array<int> m(2);
  m = 0;
  m[attr - 1] = 1;
  return m;
}

void ConstantUniaxial(const Vector& x, Vector& f) {
  UniaxialTraction(x, 0.0, f);
}

// Global max |a - b| / max |b| for local vectors of the same layout.
double RelMaxDiff(const Vector& a, const Vector& b) {
  Vector d(a);
  d -= b;
  return GlobalMax(d.Normlinf()) / (GlobalMax(b.Normlinf()) + 1e-300);
}

Vector Full(const ViscoelasticOperator& op, const Vector& m, int k) {
  Vector full;
  op.BranchToFull(m, k, full);
  return full;
}

double RelMaxDiffInRegion(ViscoelasticOperator& op, const Vector& a,
                          const Vector& b, int attr) {
  auto& sfes = op.InternalScalarSpace();
  Mesh* mesh = sfes.GetMesh();
  const int nd = sfes.GetVSize(), nc = op.NumComponents();
  double diff = 0.0, scale = 0.0;
  Array<int> dofs;
  for (int e = 0; e < mesh->GetNE(); e++) {
    if (mesh->GetAttribute(e) != attr) {
      continue;
    }
    sfes.GetElementDofs(e, dofs);
    for (int p : dofs) {
      for (int c = 0; c < nc; c++) {
        diff = std::max(diff, std::abs(a[c * nd + p] - b[c * nd + p]));
        scale = std::max(scale, std::abs(b[c * nd + p]));
      }
    }
  }
  return GlobalMax(diff) / (GlobalMax(scale) + 1e-300);
}

Vector Run(ViscoelasticOperator& visco, int steps, double dt) {
  ExponentialTrapezoidSolver ode;
  ode.Init(visco);
  Vector m(visco.Height());
  m = 0.0;
  double t = 0.0;
  for (int s = 0; s < steps; s++) {
    ode.Step(m, t, dt);
  }
  Check(visco.SolveElastic(m, t) ? 0.0 : 1.0, 0.0, "final elastic solve");
  return m;
}

void RunCase(int dim, int elementType, int order, const std::string& label) {
  auto smesh = MakeSmallMesh(dim, elementType);
  SplitAttributes(smesh);
  const auto x0_attr = BdrAttributeAt(smesh, 0, 0.0);
  const auto x1_attr = BdrAttributeAt(smesh, 0, 1.0);
  auto marker = Marker(smesh.bdr_attributes.Max(), {x0_attr, x1_attr});
  int nxyz[3] = {Mpi::WorldSize(), 1, 1};
  int* partitioning = smesh.CartesianPartitioning(nxyz);
  ParMesh pmesh(MPI_COMM_WORLD, smesh, partitioning);
  delete[] partitioning;
  H1_FECollection fec(order, dim);
  ParFiniteElementSpace pfes(&pmesh, &fec, dim);
  VectorFunctionCoefficient traction(dim, ConstantUniaxial);

  ConstantCoefficient kappa(2.1), mu(0.8), tau(1.0), zero(0.0);
  Array<int> attrs({1, 2});
  Array<Coefficient*> mu_inf_c({&mu, &zero}), mu_k_c({&zero, &mu});
  PWCoefficient mu_inf(attrs, mu_inf_c), mu_k(attrs, mu_k_c);

  // Split homogeneous body.
  {
    auto maxwell = IsotropicMaxwellRheology::Maxwell(dim, kappa, mu, tau);
    CompositeRheology split(dim, {{Region(1), &maxwell, ""},
                                  {Region(2), &maxwell, ""}});
    LinearQuasiStaticTractionProblem plain(&pfes, maxwell, traction, marker);
    LinearQuasiStaticTractionProblem comp(&pfes, split, traction, marker);
    ViscoelasticOperator v_plain(plain), v_comp(comp);
    const Vector m_plain = Run(v_plain, 4, 0.7);
    const Vector m_comp = Run(v_comp, 4, 0.7);
    Check(GlobalMax(plain.Displacement().Normlinf()) > 0.0 ? 0.0 : 1.0, 0.0,
          label + ": nonzero displacement");
    Check(RelMaxDiff(comp.Displacement(), plain.Displacement()), 1e-12,
          label + ": split body displacement");
    Check(v_comp.Height() == v_plain.Height() ? 0.0 : 1.0, 0.0,
          label + ": split body state size");
    const Vector p0 = v_plain.Branch(m_plain, 0);
    Check(RelMaxDiffInRegion(v_plain, Full(v_comp, m_comp, 0), p0, 1), 1e-12,
          label + ": split body state, region 1");
    Check(RelMaxDiffInRegion(v_plain, Full(v_comp, m_comp, 1), p0, 2), 1e-12,
          label + ": split body state, region 2");
  }

  // Elastic region beside a Maxwell region.
  {
    IsotropicElasticRheology elastic(dim, kappa, mu);
    auto maxwell = IsotropicMaxwellRheology::Maxwell(dim, kappa, mu, tau);
    CompositeRheology composite(dim, {{Region(1), &elastic, ""},
                                      {Region(2), &maxwell, ""}});
    std::vector<MaxwellBranch> branches{{&mu_k, &tau}};
    IsotropicMaxwellRheology global(dim, kappa, mu_inf, branches);
    LinearQuasiStaticTractionProblem plain(&pfes, global, traction, marker);
    LinearQuasiStaticTractionProblem comp(&pfes, composite, traction, marker);
    ViscoelasticOperator v_plain(plain), v_comp(comp);
    const Vector m_plain = Run(v_plain, 4, 0.7);
    const Vector m_comp = Run(v_comp, 4, 0.7);
    Check(RelMaxDiff(comp.Displacement(), plain.Displacement()), 1e-12,
          label + ": masked branch displacement");
    Check(RelMaxDiffInRegion(v_plain, Full(v_comp, m_comp, 0),
                             v_plain.Branch(m_plain, 0), 2),
          1e-12, label + ": masked branch state");

    // The composite's rate comes through the union-restricted strain path
    // (rank-local: a rank entirely inside one region may keep the full
    // path or hold no branch nodes at all); rebuild it from the
    // whole-mesh ComputeStrain() and compare.
    {
      Vector m(v_comp.Height()), k(v_comp.Height());
      for (int i = 0; i < m.Size(); i++) {
        m[i] = std::sin(1.0 + i + 1000.0 * Mpi::WorldRank());
      }
      v_comp.Mult(m, k);
      Check(v_comp.SolveElastic(m, v_comp.GetTime()) ? 0.0 : 1.0, 0.0,
            label + ": restricted-strain elastic solve");
      Vector d_full;
      v_comp.ComputeStrain(comp.Displacement(), d_full);
      const Array<int>& nodes = v_comp.BranchNodes(0);
      const int nd = v_comp.InternalScalarSpace().GetVSize();
      const int nb = v_comp.NumBranchNodes(0), nc = v_comp.NumComponents();
      const Vector mb = v_comp.Branch(m, 0), kb = v_comp.Branch(k, 0);
      double err = 0.0;
      for (int c = 0; c < nc; c++) {
        for (int q = 0; q < nb; q++) {
          // Rate (d - m) / tau with tau = 1.
          const double rate = d_full[c * nd + nodes[q]] - mb[c * nb + q];
          err = std::max(err, std::abs(kb[c * nb + q] - rate));
        }
      }
      Check(GlobalMax(err), 1e-13, label + ": restricted strain rate");
    }
  }

  // Stiffness of two elastic regions against piecewise moduli.
  {
    ConstantCoefficient k1(2.1), k2(3.3), m1(0.8), m2(0.4);
    IsotropicElasticRheology e1(dim, k1, m1), e2(dim, k2, m2);
    CompositeRheology composite(dim, {{Region(1), &e1, ""},
                                      {Region(2), &e2, ""}});
    Array<Coefficient*> kc({&k1, &k2}), mc({&m1, &m2});
    PWCoefficient kpw(attrs, kc), mpw(attrs, mc);
    IsotropicElasticRheology global(dim, kpw, mpw);

    auto assemble = [&](const Rheology& r) {
      auto s = r.MakeStiffness();
      ParBilinearForm a(&pfes);
      s->AddIntegrators(a);
      a.Assemble();
      a.Finalize();
      return std::unique_ptr<HypreParMatrix>(a.ParallelAssemble());
    };
    auto A = assemble(composite), B = assemble(global);
    Vector x(pfes.GetTrueVSize()), ya(x.Size()), yb(x.Size());
    for (int i = 0; i < x.Size(); i++) {
      x[i] = std::sin(0.37 * (i + 1 + 1000 * Mpi::WorldRank()));
    }
    A->Mult(x, ya);
    B->Mult(x, yb);
    Check(RelMaxDiff(ya, yb), 1e-13, label + ": stiffness action");
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();

  for (auto dim : {2, 3}) {
    for (auto elementType : {0, 1}) {
      for (auto order : {1, 2}) {
        auto label = "dim=" + std::to_string(dim) +
                     " et=" + std::to_string(elementType) +
                     " p=" + std::to_string(order);
        RunCase(dim, elementType, order, label);
      }
    }
  }

  if (Mpi::Root()) {
    if (num_fails == 0) {
      std::cout << "All " << num_checks << " checks passed on "
                << Mpi::WorldSize() << " ranks.\n";
    } else {
      std::cout << num_fails << " of " << num_checks << " checks FAILED on "
                << Mpi::WorldSize() << " ranks.\n";
    }
  }
  return num_fails == 0 ? 0 : 1;
}
