/*
  Parallel tests for SubMeshDofInjection::NewTrueDofMatrix: the algebraic
  identities of an injection on true dofs and agreement with
  ParSubMesh::Transfer. Run with 1, 2 and 4 ranks. Not a gtest: a
  standalone MPI program that exits non-zero if any check fails.

  The parent meshes are Cartesian, partitioned into slabs along x by
  CartesianPartitioning, and the submesh regions are chosen so that across
  the 1/2/4-rank runs we exercise: every rank holding submesh elements;
  ranks holding none; and submesh boundaries aligned with rank boundaries
  (the case in which shared parent dofs on the submesh boundary may be
  owned by a rank with no submesh elements there).
*/

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <string>

#include "mfem.hpp"
#include "AdGIA.hpp"

using namespace mfem;
using namespace AdGIA;

namespace {

int num_checks = 0;
int num_fails = 0;

double GlobalMax(double v) {
  double g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return g;
}

// err must be globally consistent (use GlobalMax first).
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

Vector DeterministicRandomVector(int n, int seed) {
  std::mt19937 gen(1234 + 17 * seed + Mpi::WorldRank());
  std::normal_distribution<> dist(0.0, 1.0);
  Vector v(n);
  for (auto i = 0; i < n; i++) {
    v[i] = dist(gen);
  }
  return v;
}

double TestFunction(const Vector& x) {
  auto v = 1.0;
  for (auto i = 0; i < x.Size(); i++) {
    v *= std::sin(2.0 * x[i] + 0.5 * i) + 0.25 * x[i] * x[i];
  }
  return v;
}

Mesh MakeSerialMesh(int dim, int elementType, int region) {
  auto mesh =
      dim == 2
          ? Mesh::MakeCartesian2D(4, 4,
                                  elementType == 0 ? Element::TRIANGLE
                                                   : Element::QUADRILATERAL)
          : Mesh::MakeCartesian3D(3, 3, 3,
                                  elementType == 0 ? Element::TETRAHEDRON
                                                   : Element::HEXAHEDRON);
  // region 0: x < 0.5; region 1: x > 0.5; region 2: a single corner
  // element (so that with several ranks most hold no submesh elements).
  for (auto i = 0; i < mesh.GetNE(); i++) {
    Vector c(dim);
    mesh.GetElementCenter(i, c);
    auto in = false;
    if (region == 0) {
      in = c[0] < 0.5;
    } else if (region == 1) {
      in = c[0] > 0.5;
    } else {
      in = i == 0;
    }
    mesh.SetAttribute(i, in ? 2 : 1);
  }
  mesh.SetAttributes();
  return mesh;
}

ParMesh MakeParMesh(Mesh& smesh) {
  int nxyz[3] = {Mpi::WorldSize(), 1, 1};
  int* partitioning = smesh.CartesianPartitioning(nxyz);
  auto pmesh = ParMesh(MPI_COMM_WORLD, smesh, partitioning);
  delete[] partitioning;
  return pmesh;
}

// The boundary attribute of the face lying in the plane x = 0 (used for a
// boundary submesh that only rank 0 touches under slab partitioning).
int BdrAttributeAtXZero(Mesh& mesh) {
  const auto dim = mesh.Dimension();
  for (auto i = 0; i < mesh.GetNBE(); i++) {
    auto* tr = mesh.GetBdrElementTransformation(i);
    Vector c(dim);
    tr->Transform(Geometries.GetCenter(mesh.GetBdrElementGeometry(i)), c);
    if (std::abs(c[0]) < 1e-12) {
      return mesh.GetBdrAttribute(i);
    }
  }
  return -1;
}

void RunCase(ParMesh& pmesh, bool boundary, const Array<int>& attrs,
             FiniteElementCollection& fec, int vdim, Ordering::Type ordering,
             const std::string& label) {
  auto parent_fes = ParFiniteElementSpace(&pmesh, &fec, vdim, ordering);

  auto psub = boundary ? ParSubMesh::CreateFromBoundary(pmesh, attrs)
                       : ParSubMesh::CreateFromDomain(pmesh, attrs);
  auto shadow = SubMeshDofInjection::MakeShadowSpace(parent_fes, psub);
  auto injection = SubMeshDofInjection(*shadow, parent_fes);

  const auto n = injection.SubVSize();
  const auto m = injection.ParentVSize();

  // L-vector operator against the explicit local matrix.
  {
    auto P = injection.NewSparseMatrix();
    auto x = DeterministicRandomVector(n, 1);
    auto w = DeterministicRandomVector(m, 2);
    Vector y1(m), y2(m), z1(n), z2(n);
    injection.Mult(x, y1);
    P->Mult(x, y2);
    y2 -= y1;
    injection.MultTranspose(w, z1);
    P->MultTranspose(w, z2);
    z2 -= z1;
    const auto err =
        GlobalMax(std::max(y2.Normlinf(), n > 0 ? z2.Normlinf() : 0.0));
    Check(err, 0.0, label + ": Mult/MultTranspose vs NewSparseMatrix");
  }

  auto Pi = injection.NewTrueDofMatrix();

  const auto nt = shadow->TrueVSize();
  const auto mt = parent_fes.TrueVSize();
  Check(GlobalMax(std::abs(Pi->Height() - mt) + std::abs(Pi->Width() - nt)),
        0.0, label + ": Pi local sizes");

  // Pi^T Pi = I.
  {
    auto x = DeterministicRandomVector(nt, 3);
    Vector y(mt), z(nt);
    Pi->Mult(x, y);
    Pi->MultTranspose(y, z);
    z -= x;
    const auto err = GlobalMax(nt > 0 ? z.Normlinf() : 0.0);
    Check(err, 0.0, label + ": Pi^T Pi = I");
  }

  // Pi Pi^T idempotent.
  {
    auto w = DeterministicRandomVector(mt, 4);
    Vector z(nt), w1(mt), w2(mt);
    Pi->MultTranspose(w, z);
    Pi->Mult(z, w1);
    Pi->MultTranspose(w1, z);
    Pi->Mult(z, w2);
    w2 -= w1;
    const auto err = GlobalMax(mt > 0 ? w2.Normlinf() : 0.0);
    Check(err, 0.0, label + ": Pi Pi^T idempotent");
  }

  // Pi^T of a projected analytic parent function equals ParSubMesh::Transfer
  // of it into the shadow space, compared on true dofs.
  {
    auto f = FunctionCoefficient(TestFunction);
    auto fv = VectorFunctionCoefficient(vdim,
                                        [vdim](const Vector& x, Vector& u) {
                                          u.SetSize(vdim);
                                          for (auto i = 0; i < vdim; i++) {
                                            u[i] = (i + 1.0) * TestFunction(x);
                                          }
                                        });
    auto pgf = ParGridFunction(&parent_fes);
    if (vdim == 1) {
      pgf.ProjectCoefficient(f);
    } else {
      pgf.ProjectCoefficient(fv);
    }
    auto sgf = ParGridFunction(shadow.get());
    sgf = 0.0;
    ParSubMesh::Transfer(pgf, sgf);

    Vector xt(mt), st(nt), st_ref(nt);
    pgf.GetTrueDofs(xt);
    sgf.GetTrueDofs(st_ref);
    Pi->MultTranspose(xt, st);
    st -= st_ref;
    const auto scale = GlobalMax(nt > 0 ? st_ref.Normlinf() : 0.0) + 1.0;
    const auto err = GlobalMax(nt > 0 ? st.Normlinf() : 0.0) / scale;
    Check(err, 1e-13, label + ": Pi^T vs ParSubMesh::Transfer");
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();

  for (auto dim : {2, 3}) {
    for (auto elementType : {0, 1}) {
      for (auto region : {0, 1, 2}) {
        auto smesh = MakeSerialMesh(dim, elementType, region);
        const auto x0_attr = BdrAttributeAtXZero(smesh);
        auto pmesh = MakeParMesh(smesh);
        for (auto order : {1, 2, 3}) {
          for (auto useL2 : {false, true}) {
            std::unique_ptr<FiniteElementCollection> fec;
            if (useL2) {
              fec = std::make_unique<L2_FECollection>(order, dim,
                                                      BasisType::GaussLobatto);
            } else {
              fec = std::make_unique<H1_FECollection>(order, dim);
            }
            for (auto vdim : {1, dim}) {
              for (auto byVDIM : {false, true}) {
                const auto ordering =
                    byVDIM ? Ordering::byVDIM : Ordering::byNODES;
                auto label = "dim=" + std::to_string(dim) +
                             " et=" + std::to_string(elementType) +
                             " region=" + std::to_string(region) +
                             " p=" + std::to_string(order) +
                             (useL2 ? " L2" : " H1") +
                             " vdim=" + std::to_string(vdim) +
                             (byVDIM ? " byVDIM" : " byNODES");

                auto attrs = Array<int>({2});
                RunCase(pmesh, false, attrs, *fec, vdim, ordering,
                        label + " [domain]");

                // Boundary submeshes: H1 only, and only once per mesh
                // configuration (region is irrelevant to the boundary).
                if (!useL2 && region != 1) {
                  // region 0: a boundary section every rank touches;
                  // region 2: the x = 0 face, so with several ranks only
                  // rank 0 holds boundary submesh elements.
                  auto battrs = region == 0 ? Array<int>({1})
                                            : Array<int>({x0_attr});
                  RunCase(pmesh, true, battrs, *fec, vdim, ordering,
                          label + " [boundary]");
                }
              }
            }
          }
        }
      }
    }
  }

  const auto total_fails = num_fails;
  if (Mpi::Root()) {
    if (total_fails == 0) {
      std::cout << "All " << num_checks << " checks passed on "
                << Mpi::WorldSize() << " ranks.\n";
    } else {
      std::cout << total_fails << " of " << num_checks
                << " checks FAILED on " << Mpi::WorldSize() << " ranks.\n";
    }
  }
  return total_fails == 0 ? 0 : 1;
}
