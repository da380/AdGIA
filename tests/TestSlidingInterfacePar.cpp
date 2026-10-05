/*
  Parallel test for the SubMesh true-dof pairing
  (NewSubMeshPairingTrueDofMatrix): the identification of the solid and
  fluid interface dofs must hold across partition boundaries, where the two
  sides of a parent interface dof live on different ranks. Run with 1, 2
  and 4 ranks; a standalone MPI program returning 0 if every check
  passes, 1 otherwise.

  A smooth field interpolated on both SubMeshes has equal traces, so
  J x_f must equal J J^T x_s (the paired part of x_s) to round-off; the
  pairing must be non-empty, and J J^T idempotent.
*/

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>

#include "mfem.hpp"
#include "AdGIA.hpp"

using namespace mfem;
using namespace AdGIA;

namespace {

int num_checks = 0;
int num_fails = 0;

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

double GlobalMax(double v) {
  double g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return g;
}

void RunCase(const char* mesh_file, int fluid_attr, int order,
             const std::string& label) {
  Mesh smesh(mesh_file, 1, 1);
  const int dim = smesh.Dimension();
  ParMesh pmesh(MPI_COMM_WORLD, smesh);
  smesh.Clear();

  Array<int> fluid_attrs({fluid_attr});
  Array<int> solid_attrs;
  for (int a = 1; a <= pmesh.attributes.Max() - 1; a++) {  // buffer excluded
    if (a != fluid_attr) {
      solid_attrs.Append(a);
    }
  }
  ParSubMesh solid(ParSubMesh::CreateFromDomain(pmesh, solid_attrs));
  ParSubMesh fluid(ParSubMesh::CreateFromDomain(pmesh, fluid_attrs));

  H1_FECollection fec(order, dim);
  ParFiniteElementSpace fes_parent(&pmesh, &fec, dim);
  auto fes_s = SubMeshDofInjection::MakeShadowSpace(fes_parent, solid);
  auto fes_f = SubMeshDofInjection::MakeShadowSpace(fes_parent, fluid);
  SubMeshDofInjection inj_s(*fes_s, fes_parent), inj_f(*fes_f, fes_parent);
  auto J = NewSubMeshPairingTrueDofMatrix(inj_s, inj_f);
  std::unique_ptr<HypreParMatrix> Jt(J->Transpose());

  VectorFunctionCoefficient f(dim, [](const Vector& x, Vector& v) {
    v.SetSize(x.Size());
    v[0] = std::sin(x[0]) + x[1] * x[1];
    v[1] = std::cos(x[1]) - 2.0 * x[0];
    if (x.Size() == 3) {
      v[2] = x[0] * x[2] - std::sin(x[1]);
    }
  });
  ParGridFunction u_s(fes_s.get()), u_f(fes_f.get());
  u_s.ProjectCoefficient(f);
  u_f.ProjectCoefficient(f);
  Vector x_s, x_f;
  u_s.GetTrueDofs(x_s);
  u_f.GetTrueDofs(x_f);

  // J x_f = (paired part of) x_s, across ranks.
  Vector z(J->Height()), t(J->Width()), w(J->Height());
  J->Mult(x_f, z);
  Jt->Mult(x_s, t);
  J->Mult(t, w);  // J J^T x_s
  const double znorm = std::sqrt(InnerProduct(MPI_COMM_WORLD, z, z));
  Check(znorm > 0.0 ? 0.0 : 1.0, 0.0, label + " pairing non-empty");
  w -= z;
  Check(GlobalMax(w.Normlinf()), 1e-12, label + " trace identification");

  // Idempotence of J J^T: apply twice to a random-ish vector.
  Vector r(J->Height());
  for (int i = 0; i < r.Size(); i++) {
    r[i] = std::sin(0.7 * i + 0.3);
  }
  Vector s1(J->Height()), s2(J->Height());
  Jt->Mult(r, t);
  J->Mult(t, s1);  // S r
  Jt->Mult(s1, t);
  J->Mult(t, s2);  // S^2 r
  s2 -= s1;
  Check(GlobalMax(s2.Normlinf()), 1e-12, label + " J J^T idempotent");
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();

  RunCase("../data/elastogravity_two_layer_2d.msh", 1, 2, "2d-o2");
  RunCase("../data/elastogravity_three_layer_3d.msh", 2, 2, "3d-o2");

  if (Mpi::Root()) {
    if (num_fails == 0) {
      std::cout << "All " << num_checks << " checks passed on "
                << Mpi::WorldSize() << " ranks.\n";
    } else {
      std::cout << num_fails << " of " << num_checks << " checks failed on "
                << Mpi::WorldSize() << " ranks.\n";
    }
  }
  return num_fails == 0 ? 0 : 1;  // an exit status is taken modulo 256
}
