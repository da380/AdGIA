/*
  Parallel tests for ParSubMeshMixedBilinearForm. Run with 1, 2 and 4
  ranks. Not a gtest: a standalone MPI program that exits non-zero if any
  check fails.

  Every rank holds the full serial mesh as well, and evaluates the serial
  SubMeshMixedBilinearForm there as the reference. The check is the value
  of the bilinear form on projected analytic functions, g^T A f, which is
  independent of dof numbering and partitioning: the parallel value from
  ParallelAssemble() (and from FormRectangularSystemMatrix() with no
  essential dofs) must equal the serial one to round-off.

  As in TestSubMeshDofInjectionPar, the Cartesian parent meshes are
  partitioned into slabs along x and the submesh regions are chosen so
  that across the 1/2/4-rank runs some ranks hold no submesh elements and
  submesh boundaries coincide with rank boundaries. Integrators: a domain
  integrator on the submesh and, for the scalar family, boundary
  integrators on both the cut (internal to the parent) and an exterior
  boundary section.
*/

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <list>
#include <memory>
#include <string>
#include <vector>

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

double TestFunction(const Vector& x) {
  auto v = 1.0;
  for (auto i = 0; i < x.Size(); i++) {
    v *= std::sin(2.0 * x[i] + 0.5 * i) + 0.25 * x[i] * x[i];
  }
  return v;
}

double OtherFunction(const Vector& x) {
  auto v = 0.5;
  for (auto i = 0; i < x.Size(); i++) {
    v += std::cos(1.5 * x[i] - 0.3 * i) * (1.0 + x[i]);
  }
  return v;
}

Mesh MakeSerialMesh(int dim, int elementType, int region) {
  auto mesh =
      dim == 2
          ? Mesh::MakeCartesian2D(
                4, 4,
                elementType == 0 ? Element::TRIANGLE : Element::QUADRILATERAL)
          : Mesh::MakeCartesian3D(
                3, 3, 3,
                elementType == 0 ? Element::TETRAHEDRON : Element::HEXAHEDRON);
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

// The configuration of one case: the parent-side space is scalar H1 of
// the given order; the submesh side is family 0 (scalar H1), 1 (vector H1
// byVDIM, vdim = submesh dimension) or 2 (scalar L2 of order - 1).
struct Config {
  int order;
  bool parentIsTrial;
  int family;
  int cut_attr;  // boundary attribute of the cut (parent max + 1)
  int ext_attr;  // an exterior boundary attribute inside the submesh
  bool boundary_submesh;
};

struct Spaces {
  std::vector<std::unique_ptr<FiniteElementCollection>> fecs;
  std::unique_ptr<FiniteElementSpace> parent, sub;
};

// Builds the parent-side and submesh-side spaces (serial or parallel
// according to the mesh types).
Spaces MakeSpaces(Mesh& parent_mesh, Mesh& submesh, const Config& cfg) {
  Spaces s;
  const auto pdim = parent_mesh.Dimension();
  const auto sdim = submesh.Dimension();
  s.fecs.push_back(std::make_unique<H1_FECollection>(cfg.order, pdim));
  auto* pfec = s.fecs.back().get();
  if (cfg.family == 2) {
    s.fecs.push_back(std::make_unique<L2_FECollection>(cfg.order - 1, sdim));
  } else {
    s.fecs.push_back(std::make_unique<H1_FECollection>(cfg.order, sdim));
  }
  auto* sfec = s.fecs.back().get();
  const auto vdim = cfg.family == 1 ? sdim : 1;
  const auto ordering = cfg.family == 1 ? Ordering::byVDIM : Ordering::byNODES;

#ifdef MFEM_USE_MPI
  if (auto* pm = dynamic_cast<ParMesh*>(&parent_mesh)) {
    auto* psm = dynamic_cast<ParMesh*>(&submesh);
    s.parent = std::make_unique<ParFiniteElementSpace>(pm, pfec);
    s.sub = std::make_unique<ParFiniteElementSpace>(psm, sfec, vdim, ordering);
    return s;
  }
#endif
  s.parent = std::make_unique<FiniteElementSpace>(&parent_mesh, pfec);
  s.sub = std::make_unique<FiniteElementSpace>(&submesh, sfec, vdim, ordering);
  return s;
}

// Coefficients and attribute markers; the forms keep pointers to the
// markers, so this must outlive them.
struct CaseData {
  CaseData()
      : rho([](const Vector& x) { return 1.0 + x[0] + 0.5 * x[x.Size() - 1]; }),
        sigma([](const Vector& x) { return 2.0 - x[x.Size() - 1]; }) {}
  FunctionCoefficient rho, sigma;
  std::list<Array<int>> markers;
};

void AddIntegrators(MixedBilinearForm& f, Mesh& submesh, const Config& cfg,
                    CaseData& c) {
  auto& dom_marker = c.markers.emplace_back(submesh.attributes.Max());
  dom_marker = 1;
  BilinearFormIntegrator* integ = nullptr;
  if (cfg.family == 1) {
    integ = cfg.parentIsTrial
                ? static_cast<BilinearFormIntegrator*>(
                      new GradientIntegrator(c.rho))
                : new TransposeIntegrator(new GradientIntegrator(c.rho));
  } else {
    integ = new MassIntegrator(c.rho);
  }
  f.AddDomainIntegrator(integ, dom_marker);

  if (cfg.family == 0 && !cfg.boundary_submesh) {
    // Marker sized by the parent's max + 1 (the cut attribute), which is
    // an upper bound for every rank's local submesh boundary attributes.
    auto& bdr_marker = c.markers.emplace_back(cfg.cut_attr);
    bdr_marker = 0;
    bdr_marker[cfg.cut_attr - 1] = 1;
    bdr_marker[cfg.ext_attr - 1] = 1;
    f.AddBoundaryIntegrator(new BoundaryMassIntegrator(c.sigma), bdr_marker);
  }
}

// Projects the analytic trial/test functions on the two spaces.
void ProjectFunctions(FiniteElementSpace& parent_fes,
                      FiniteElementSpace& sub_fes, const Config& cfg,
                      GridFunction& fp, GridFunction& gs) {
  auto f = FunctionCoefficient(TestFunction);
  fp.SetSpace(&parent_fes);
  fp.ProjectCoefficient(f);

  const auto vdim = sub_fes.GetVDim();
  gs.SetSpace(&sub_fes);
  if (vdim == 1) {
    auto g = FunctionCoefficient(OtherFunction);
    gs.ProjectCoefficient(g);
  } else {
    auto gv =
        VectorFunctionCoefficient(vdim, [vdim](const Vector& x, Vector& u) {
          u.SetSize(vdim);
          for (auto i = 0; i < vdim; i++) {
            u[i] = (i + 1.0) * OtherFunction(x);
          }
        });
    gs.ProjectCoefficient(gv);
  }
}

// Serial reference: g^T A f on the full mesh.
double SerialValue(Mesh& smesh, const Array<int>& attrs, const Config& cfg,
                   CaseData& c) {
  auto submesh = cfg.boundary_submesh
                     ? SubMesh::CreateFromBoundary(smesh, attrs)
                     : SubMesh::CreateFromDomain(smesh, attrs);
  auto s = MakeSpaces(smesh, submesh, cfg);
  auto form = SubMeshMixedBilinearForm(
      cfg.parentIsTrial ? s.parent.get() : s.sub.get(),
      cfg.parentIsTrial ? s.sub.get() : s.parent.get());
  AddIntegrators(form, submesh, cfg, c);
  form.Assemble();
  form.Finalize();

  GridFunction fp, gs;
  ProjectFunctions(*s.parent, *s.sub, cfg, fp, gs);
  Vector y(form.Height());
  if (cfg.parentIsTrial) {
    form.Mult(fp, y);
    return InnerProduct(gs, y);
  }
  form.Mult(gs, y);
  return InnerProduct(fp, y);
}

void RunCase(Mesh& smesh, ParMesh& pmesh, const Array<int>& attrs,
             const Config& cfg, const std::string& label) {
  CaseData c;
  const auto ref = SerialValue(smesh, attrs, cfg, c);
  const auto scale = std::abs(ref) + 1.0;

  auto psub = cfg.boundary_submesh
                  ? ParSubMesh::CreateFromBoundary(pmesh, attrs)
                  : ParSubMesh::CreateFromDomain(pmesh, attrs);
  auto s = MakeSpaces(pmesh, psub, cfg);
  auto* parent = static_cast<ParFiniteElementSpace*>(s.parent.get());
  auto* sub = static_cast<ParFiniteElementSpace*>(s.sub.get());

  auto form = ParSubMeshMixedBilinearForm(cfg.parentIsTrial ? parent : sub,
                                          cfg.parentIsTrial ? sub : parent);
  AddIntegrators(form, psub, cfg, c);
  form.Assemble();
  form.Finalize();

  GridFunction fp, gs;
  ProjectFunctions(*parent, *sub, cfg, fp, gs);
  Vector ft(parent->TrueVSize()), gt(sub->TrueVSize());
  parent->GetRestrictionMatrix()->Mult(fp, ft);
  sub->GetRestrictionMatrix()->Mult(gs, gt);

  auto& xt = cfg.parentIsTrial ? ft : gt;
  auto& yt = cfg.parentIsTrial ? gt : ft;

  {
    std::unique_ptr<HypreParMatrix> A(form.ParallelAssemble());
    Vector Ax(A->Height());
    A->Mult(xt, Ax);
    const auto val = InnerProduct(MPI_COMM_WORLD, yt, Ax);
    Check(GlobalMax(std::abs(val - ref)) / scale, 1e-12,
          label + ": ParallelAssemble vs serial");
  }

  {
    Array<int> empty;
    OperatorHandle Ah(Operator::Hypre_ParCSR);
    form.FormRectangularSystemMatrix(empty, empty, Ah);
    Vector Ax(Ah->Height());
    Ah->Mult(xt, Ax);
    const auto val = InnerProduct(MPI_COMM_WORLD, yt, Ax);
    Check(GlobalMax(std::abs(val - ref)) / scale, 1e-12,
          label + ": FormRectangularSystemMatrix vs serial");
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
        const auto cut_attr = smesh.bdr_attributes.Max() + 1;
        auto pmesh = MakeParMesh(smesh);
        for (auto order : {1, 2, 3}) {
          for (auto parentIsTrial : {false, true}) {
            for (auto family : {0, 1, 2}) {
              auto label = "dim=" + std::to_string(dim) +
                           " et=" + std::to_string(elementType) +
                           " region=" + std::to_string(region) +
                           " p=" + std::to_string(order) +
                           (parentIsTrial ? " parent=trial" : " parent=test") +
                           " family=" + std::to_string(family);

              // Domain submesh. Regions 0 and 2 contain the x = 0 face;
              // region 1 contains x = 1 instead, so use attribute 1
              // (y = 0 / z = 0 face, which every region touches).
              Config cfg{order,
                         parentIsTrial,
                         family,
                         cut_attr,
                         region == 1 ? 1 : x0_attr,
                         false};
              auto attrs = Array<int>({2});
              RunCase(smesh, pmesh, attrs, cfg, label + " [domain]");

              // Boundary submesh: scalar families only (see the serial
              // test), once per mesh configuration.
              if (family != 1 && region != 1) {
                Config bcfg{order, parentIsTrial, family, cut_attr, 1, true};
                auto battrs =
                    region == 0 ? Array<int>({1}) : Array<int>({x0_attr});
                RunCase(smesh, pmesh, battrs, bcfg, label + " [boundary]");
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
      std::cout << total_fails << " of " << num_checks << " checks FAILED on "
                << Mpi::WorldSize() << " ranks.\n";
    }
  }
  return total_fails == 0 ? 0 : 1;
}
