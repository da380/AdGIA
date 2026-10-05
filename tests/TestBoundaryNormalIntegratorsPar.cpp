/*
  Parallel tests for the boundary normal integrators on a ParSubMesh, and
  a check that BoomerAMG copes with a disconnected ParSubMesh. Run with 1,
  2 and 4 ranks. Not a gtest: a standalone MPI program returning the number
  of failed checks.

  Meshes: the three-layer disc/ball in ../data (attributes 1 inner core,
  2 outer core, 3 mantle, 4 buffer; boundary attributes 1 ICB, 2 CMB,
  3 surface, 4 outer), with ONE ParSubMesh for the two solid regions
  {1, 3}, which is disconnected.

  1. The values of the interface forms on interpolated fields,
       int_{ICB + CMB} q (m.u)(m.v) dS    (ParBilinearForm on the ParSubMesh),
       int_{ICB + CMB} q p (m.v) dS       (ParSubMeshMixedBilinearForm,
                                           scalar trial on the parent),
     with q = 1 and q = m . grad Phi0 (Phi0 = |x|^2/2 on the parent,
     restricted to the shadow space, evaluated through the boundary-element
     gradient path), equal the serial values to round-off, independently
     of the partition.
  2. CG preconditioned by BoomerAMG converges on the disconnected
     ParSubMesh, for a shifted Laplacian and for a shifted elasticity
     operator with the elasticity AMG options (the setup used by
     LinearQuasiStaticProblemBase), in an iteration count comparable to that
     on the connected mantle-only ParSubMesh. The counts are printed.
*/

#include <mpi.h>

#include <cmath>
#include <iostream>
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

void Check(bool ok, const std::string& what) {
  num_checks++;
  int fail = ok ? 0 : 1;
  int any = 0;
  MPI_Allreduce(&fail, &any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (any) {
    num_fails++;
    if (Mpi::Root()) {
      std::cout << "FAIL: " << what << "\n";
    }
  }
}

void CheckClose(double a, double b, double tol, const std::string& what) {
  const bool ok = std::abs(a - b) <= tol * std::abs(b);
  if (!ok && Mpi::Root()) {
    std::cout << "  " << what << ": " << a << " vs " << b << "\n";
  }
  Check(ok, what);
}

void Position(const Vector& x, Vector& v) { v = x; }
double HalfR2(const Vector& x) { return 0.5 * (x * x); }

std::string MeshFile(int dim) {
  return dim == 2 ? "../data/elastogravity_three_layer_2d.msh"
                  : "../data/elastogravity_three_layer_3d.msh";
}

Array<int> Marker(int max, std::initializer_list<int> attrs) {
  Array<int> m(max);
  m = 0;
  for (int a : attrs) {
    m[a - 1] = 1;
  }
  return m;
}

// --- serial reference values -----------------------------------------------

struct SerialValues {
  double nn = 0.0, nn_g = 0.0, ns = 0.0, ns_g = 0.0;
};

SerialValues Serial(Mesh& parent, const Array<int>& solid, int order) {
  const int dim = parent.Dimension();
  SubMesh sub(SubMesh::CreateFromDomain(parent, solid));
  H1_FECollection fec(order, dim);
  FiniteElementSpace fes_u(&sub, &fec, dim), fes_phi(&parent, &fec);
  auto shadow = SubMeshDofInjection::MakeShadowSpace(fes_phi, sub);
  SubMeshDofInjection inj(*shadow, fes_phi);

  VectorFunctionCoefficient xc(dim, Position);
  FunctionCoefficient half_r2(HalfR2);
  GridFunction u(&fes_u), phi0(&fes_phi), phi0_shadow(shadow.get()),
      p(&fes_phi);
  u.ProjectCoefficient(xc);
  phi0.ProjectCoefficient(half_r2);
  inj.MultTranspose(phi0, phi0_shadow);
  p = 1.0;
  GradientGridFunctionCoefficient grad(&phi0_shadow);
  BoundaryNormalDotCoefficient g(grad);
  Array<int> interfaces = Marker(sub.bdr_attributes.Max(), {1, 2});

  SerialValues v;
  auto nn = [&](Coefficient* q) {
    BilinearForm a(&fes_u);
    a.AddBoundaryIntegrator(new BoundaryNormalNormalIntegrator(q),
                            interfaces);
    a.Assemble();
    a.Finalize();
    Vector Au(a.Height());
    a.SpMat().Mult(u, Au);
    return Au * u;
  };
  auto ns = [&](Coefficient* q) {
    SubMeshMixedBilinearForm c(&fes_phi, &fes_u);
    c.AddBoundaryIntegrator(new BoundaryNormalScalarIntegrator(q),
                            interfaces);
    c.Assemble();
    Vector Cp(c.Height());
    c.SpMat().Mult(p, Cp);
    return Cp * u;
  };
  v.nn = nn(nullptr);
  v.nn_g = nn(&g);
  v.ns = ns(nullptr);
  v.ns_g = ns(&g);
  return v;
}

// --- parallel values ---------------------------------------------------------

void ParallelValues(ParMesh& pparent, const Array<int>& solid, int order,
                    const SerialValues& ref, const std::string& label) {
  const int dim = pparent.Dimension();
  ParSubMesh sub(ParSubMesh::CreateFromDomain(pparent, solid));
  H1_FECollection fec(order, dim);
  ParFiniteElementSpace fes_u(&sub, &fec, dim), fes_phi(&pparent, &fec);
  auto shadow = SubMeshDofInjection::MakeShadowSpace(fes_phi, sub);
  SubMeshDofInjection inj(*shadow, fes_phi);

  VectorFunctionCoefficient xc(dim, Position);
  FunctionCoefficient half_r2(HalfR2);
  ParGridFunction u(&fes_u), phi0(&fes_phi), phi0_shadow(shadow.get()),
      p(&fes_phi);
  u.ProjectCoefficient(xc);
  phi0.ProjectCoefficient(half_r2);
  inj.MultTranspose(phi0, phi0_shadow);
  p = 1.0;
  GradientGridFunctionCoefficient grad(&phi0_shadow);
  BoundaryNormalDotCoefficient g(grad);
  Array<int> interfaces = Marker(sub.bdr_attributes.Max(), {1, 2});

  Vector U, P;
  u.GetTrueDofs(U);
  p.GetTrueDofs(P);

  auto nn = [&](Coefficient* q) {
    ParBilinearForm a(&fes_u);
    a.AddBoundaryIntegrator(new BoundaryNormalNormalIntegrator(q),
                            interfaces);
    a.Assemble();
    a.Finalize();
    std::unique_ptr<HypreParMatrix> A(a.ParallelAssemble());
    Vector AU(A->Height());
    A->Mult(U, AU);
    return InnerProduct(MPI_COMM_WORLD, AU, U);
  };
  auto ns = [&](Coefficient* q) {
    ParSubMeshMixedBilinearForm c(&fes_phi, &fes_u);
    c.AddBoundaryIntegrator(new BoundaryNormalScalarIntegrator(q),
                            interfaces);
    c.Assemble();
    std::unique_ptr<HypreParMatrix> C(c.ParallelAssemble());
    Vector CP(C->Height());
    C->Mult(P, CP);
    return InnerProduct(MPI_COMM_WORLD, CP, U);
  };
  const double tol = 1e-12;
  CheckClose(nn(nullptr), ref.nn, tol, label + " normal-normal");
  CheckClose(nn(&g), ref.nn_g, tol, label + " normal-normal, q = m.grad Phi0");
  CheckClose(ns(nullptr), ref.ns, tol, label + " normal-scalar");
  CheckClose(ns(&g), ref.ns_g, tol, label + " normal-scalar, q = m.grad Phi0");
}

// --- BoomerAMG on a (disconnected) ParSubMesh ---------------------------------

struct AmgResult {
  int its_scalar = -1, its_vector = -1;
  bool ok_scalar = false, ok_vector = false;
};

AmgResult RunAmg(ParMesh& pparent, const Array<int>& solid, int order,
                 double eps) {
  const int dim = pparent.Dimension();
  ParSubMesh sub(ParSubMesh::CreateFromDomain(pparent, solid));
  H1_FECollection fec(order, dim);
  AmgResult r;
  ConstantCoefficient one(1.0), shift(eps), lambda(1.0), mu(0.5);

  // Shifted Laplacian, CG + BoomerAMG.
  {
    ParFiniteElementSpace fes(&sub, &fec);
    ParBilinearForm a(&fes);
    a.AddDomainIntegrator(new DiffusionIntegrator(one));
    a.AddDomainIntegrator(new MassIntegrator(shift));
    a.Assemble();
    a.Finalize();
    std::unique_ptr<HypreParMatrix> A(a.ParallelAssemble());
    ParLinearForm b(&fes);
    FunctionCoefficient f(HalfR2);
    b.AddDomainIntegrator(new DomainLFIntegrator(f));
    b.Assemble();
    std::unique_ptr<HypreParVector> B(b.ParallelAssemble());
    Vector X(A->Height());
    X = 0.0;
    HypreBoomerAMG amg(*A);
    amg.SetPrintLevel(0);
    CGSolver cg(MPI_COMM_WORLD);
    cg.SetOperator(*A);
    cg.SetPreconditioner(amg);
    cg.SetRelTol(1e-10);
    cg.SetAbsTol(0.0);
    cg.SetMaxIter(500);
    cg.SetPrintLevel(0);
    cg.Mult(*B, X);
    r.its_scalar = cg.GetNumIterations();
    r.ok_scalar = cg.GetConverged();
  }
  // Shifted elasticity, CG + BoomerAMG with the elasticity options.
  {
    ParFiniteElementSpace fes(&sub, &fec, dim);
    ParBilinearForm a(&fes);
    a.AddDomainIntegrator(new ElasticityIntegrator(lambda, mu));
    a.AddDomainIntegrator(new VectorMassIntegrator(shift));
    a.Assemble();
    a.Finalize();
    std::unique_ptr<HypreParMatrix> A(a.ParallelAssemble());
    ParLinearForm b(&fes);
    VectorFunctionCoefficient f(dim, Position);
    b.AddDomainIntegrator(new VectorDomainLFIntegrator(f));
    b.Assemble();
    std::unique_ptr<HypreParVector> B(b.ParallelAssemble());
    Vector X(A->Height());
    X = 0.0;
    HypreBoomerAMG amg(*A);
    amg.SetElasticityOptions(&fes);
    amg.SetPrintLevel(0);
    CGSolver cg(MPI_COMM_WORLD);
    cg.SetOperator(*A);
    cg.SetPreconditioner(amg);
    cg.SetRelTol(1e-10);
    cg.SetAbsTol(0.0);
    cg.SetMaxIter(500);
    cg.SetPrintLevel(0);
    cg.Mult(*B, X);
    r.its_vector = cg.GetNumIterations();
    r.ok_vector = cg.GetConverged();
  }
  return r;
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();

  for (int dim : {2, 3}) {
    Mesh smesh(MeshFile(dim).c_str(), 1, 1);
    if (smesh.Dimension() != dim) {
      Check(false, "mesh dimension");
      continue;
    }
    ParMesh pmesh(MPI_COMM_WORLD, smesh);
    Array<int> solid({1, 3}), mantle({3});
    const int order = 2;
    const std::string label = "dim=" + std::to_string(dim);

    // 1. Interface forms against the serial values.
    const auto ref = Serial(smesh, solid, order);
    ParallelValues(pmesh, solid, order, ref, label);

    // 2. BoomerAMG on the disconnected and on the connected submesh. With
    // the small shift the operators are nearly singular (the disconnected
    // mesh has twice the near-rigid modes, hence more elasticity
    // iterations); the shift 1 shows the AMG itself.
    for (double eps : {1e-3, 1.0}) {
      const auto disc = RunAmg(pmesh, solid, order, eps);
      const auto conn = RunAmg(pmesh, mantle, order, eps);
      if (Mpi::Root()) {
        std::cout << label << " np=" << Mpi::WorldSize() << " shift=" << eps
                  << "  CG+BoomerAMG iterations, disconnected {1,3} vs "
                     "connected {3}: Laplacian "
                  << disc.its_scalar << " vs " << conn.its_scalar
                  << ", elasticity " << disc.its_vector << " vs "
                  << conn.its_vector << "\n";
      }
      Check(disc.ok_scalar,
            label + " AMG Laplacian converged (disconnected)");
      Check(disc.ok_vector,
            label + " AMG elasticity converged (disconnected)");
      Check(disc.its_scalar <= 2 * conn.its_scalar + 10,
            label + " AMG Laplacian iterations comparable");
      Check(disc.its_vector <= 2 * conn.its_vector + 10,
            label + " AMG elasticity iterations comparable");
    }
  }

  if (Mpi::Root()) {
    std::cout << "TestBoundaryNormalIntegratorsPar (np=" << Mpi::WorldSize()
              << "): " << num_checks - num_fails << " / " << num_checks
              << " checks passed\n";
  }
  return num_fails == 0 ? 0 : 1;  // an exit status is taken modulo 256
}
