/*
  Parallel tests of LinearQuasiStaticClampedFaultProblem and
  FaultToSurfaceMap on the faulted boxes, in 2-D (orders 1 and 2) and 3-D
  (order 1). Run with 1, 2 and 4 ranks; a
  standalone MPI program that prints a summary and exits non-zero if any
  check fails.

  - Rigid block (as in the serial TestFaultProblem): the whole cut as
    fault, only the footwall clamped, a uniform slip; the hanging wall
    translates rigidly, on every rank's dofs.
  - Serial = parallel for the contained fault with a tapered slip: every
    rank also solves the serial problem and compares partition-independent
    quantities — the observable <G s, w> of a surface load w, its adjoint
    <s, G^T w>, the reaction energy <r(s), s> and the L2 norm of the
    physical displacement.
  - Reciprocity <G s, w> = <s, G^T w> on rank-dependent random vectors.
*/

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>

#include "FaultTestCommon.hpp"
#include "mfem.hpp"
#include "AdGIA.hpp"

using namespace mfem;
using namespace AdGIA;
using namespace fault_test;

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

void CheckTrue(bool ok, const std::string& what) {
  Check(ok ? 0.0 : 1.0, 0.0, what);
}

std::string Label(int dim, int order) {
  return " (" + std::to_string(dim) + "-D, p=" + std::to_string(order) + ")";
}

double RelDiff(double a, double b) {
  return std::abs(a - b) / std::max({std::abs(a), std::abs(b), 1e-300});
}

FunctionCoefficient Kappa(int dim) {
  return FunctionCoefficient([dim](const Vector& x) {
    return 2.0 * (1.0 - 0.3 * x(dim - 1)) * (1.0 + 0.05 * x(0));
  });
}

FunctionCoefficient Mu(int dim) {
  return FunctionCoefficient([dim](const Vector& x) {
    return (1.0 - 0.3 * x(dim - 1)) * (1.0 + 0.05 * x(0));
  });
}

// A surface observable: the vertical displacement weighted by a bump
// centred above the fault, plus a small horizontal part.
VectorFunctionCoefficient SurfaceWeight(int dim) {
  return VectorFunctionCoefficient(dim, [dim](const Vector& x, Vector& w) {
    w.SetSize(dim);
    auto r2 = x(0) * x(0);
    if (dim == 3) {
      r2 += x(1) * x(1);
    }
    const auto g = std::exp(-r2);
    w = 0.3 * g;
    w(dim - 1) = g;
  });
}

// The quantities compared between serial and parallel.
struct Observables {
  double map = 0.0, adjoint = 0.0, reaction_energy = 0.0, l2 = 0.0;
};

// The contained-fault problem on mesh (serial or parallel, by its type),
// the map, and the observables of the tapered slip and the surface
// weight. The submesh factory and dot product are the per-path pieces.
template <class MeshT, class SpaceT, class GridT, class LinearFormT,
          class SubMeshT>
Observables Solve(MeshT& mesh, int order, const FaultedBox& box) {
  const auto dim = box.dim;
  auto fec = H1_FECollection(order, dim);
  SpaceT fes(&mesh, &fec, dim);
  auto kappa = Kappa(dim);
  auto mu = Mu(dim);
  auto rheology = IsotropicElasticRheology(dim, kappa, mu);
  auto zero = Vector(dim);
  zero = 0.0;
  auto traction = VectorConstantCoefficient(zero);
  const auto nb = mesh.bdr_attributes.Max();
  const auto na = mesh.attributes.Max();
  auto problem = LinearQuasiStaticClampedFaultProblem(
      &fes, rheology,
      Marker(nb, {box.clamped_footwall, box.clamped_hanging_wall}), traction,
      Marker(nb, {}), Marker(na, {box.hanging_wall}), Marker(nb, {box.fault}),
      Marker(nb, {box.extension}));

  auto fault_attr = Array<int>({box.fault});
  auto surface_attr = Array<int>({box.surface});
  auto fault_mesh = SubMeshT::CreateFromBoundary(mesh, fault_attr);
  auto surface_mesh = SubMeshT::CreateFromBoundary(mesh, surface_attr);
  auto fault_fes = SubMeshDofInjection::MakeShadowSpace(fes, fault_mesh);
  auto surface_fes = SubMeshDofInjection::MakeShadowSpace(fes, surface_mesh);
  auto G = FaultToSurfaceMap(problem, *fault_fes, *surface_fes);

  auto slip_c = TaperedSlip(box, 0.01);
  GridT slip_f(fault_fes.get());
  slip_f.ProjectCoefficient(slip_c);
  auto s = Vector(fault_fes->GetTrueVSize());
  slip_f.GetTrueDofs(s);

  auto weight = SurfaceWeight(dim);
  LinearFormT w_form(surface_fes.get());
  w_form.AddDomainIntegrator(new VectorDomainLFIntegrator(weight));
  w_form.Assemble();
  auto w = Vector(surface_fes->GetTrueVSize());
  if (const Operator* P = surface_fes->GetProlongationMatrix()) {
    P->MultTranspose(w_form, w);
  } else {
    w = w_form;
  }

  auto dot = [&](const Vector& a, const Vector& b) {
#ifdef MFEM_USE_MPI
    if (auto* pf = dynamic_cast<ParFiniteElementSpace*>(&fes)) {
      return InnerProduct(pf->GetComm(), a, b);
    }
#endif
    return a * b;
  };

  auto out = Observables();
  auto Gs = Vector();
  auto Gtw = Vector();
  auto r = Vector();
  auto slip_true = Vector();
  G.Mult(s, Gs);
  CheckTrue(G.Converged(), "forward solve converged");
  out.map = dot(Gs, w);
  problem.FaultReaction(r);
  problem.Slip().GetTrueDofs(slip_true);
  out.reaction_energy = dot(r, slip_true);
  auto dg = L2_FECollection(order, dim, BasisType::GaussLobatto);
  SpaceT dg_fes(&mesh, &dg, dim);
  GridT u_dg(&dg_fes);
  u_dg.ProjectCoefficient(problem.PhysicalDisplacement());
  auto zero_c = VectorConstantCoefficient(zero);
  out.l2 = u_dg.ComputeL2Error(zero_c);
  G.MultTranspose(w, Gtw);
  CheckTrue(G.Converged(), "adjoint solve converged");
  out.adjoint = dot(s, Gtw);
  return out;
}

void RunRigidBlock(int dim, int order) {
  auto smesh = Mesh(MeshFile(dim).c_str(), 1, 1);
  const auto box = ReadFaultedBox(RecordFile(dim));
  auto pmesh = ParMesh(MPI_COMM_WORLD, smesh);
  auto fec = H1_FECollection(order, dim);
  auto fes = ParFiniteElementSpace(&pmesh, &fec, dim);
  auto kappa = ConstantCoefficient(2.0);
  auto mu = ConstantCoefficient(1.0);
  auto rheology = IsotropicElasticRheology(dim, kappa, mu);
  auto zero = Vector(dim);
  zero = 0.0;
  auto traction = VectorConstantCoefficient(zero);
  const auto nb = pmesh.bdr_attributes.Max();
  const auto cut = Marker(nb, {box.fault, box.extension});
  auto problem = LinearQuasiStaticClampedFaultProblem(
      &fes, rheology, Marker(nb, {box.clamped_footwall}), traction,
      Marker(nb, {}), Marker(pmesh.attributes.Max(), {box.hanging_wall}), cut,
      Marker(nb, {}));
  auto slip = Vector(dim);
  slip = 0.1;
  slip(0) = 0.3;
  slip(dim - 1) = -0.2;
  auto slip_c = VectorConstantCoefficient(slip);
  problem.SetSlip(slip_c);
  problem.AssembleForce(0.0);
  CheckTrue(problem.Solve(), "rigid block solve converged");
  const auto err = GlobalMax(
      RigidBlockError(problem.Displacement(), cut, box.hanging_wall, slip));
  Check(err, 1e-8 * slip.Norml2(), "rigid block" + Label(dim, order));
}

void RunSerialParallel(int dim, int order) {
  const auto box = ReadFaultedBox(RecordFile(dim));
  auto smesh = Mesh(MeshFile(dim).c_str(), 1, 1);
  const auto ser =
      Solve<Mesh, FiniteElementSpace, GridFunction, LinearForm, SubMesh>(
          smesh, order, box);
  auto pmesh = ParMesh(MPI_COMM_WORLD, smesh);
  const auto par = Solve<ParMesh, ParFiniteElementSpace, ParGridFunction,
                         ParLinearForm, ParSubMesh>(pmesh, order, box);
  const auto p = Label(dim, order);
  Check(RelDiff(par.map, ser.map), 1e-8, "serial = parallel <G s, w>" + p);
  Check(RelDiff(par.adjoint, ser.adjoint), 1e-8,
        "serial = parallel <s, G^T w>" + p);
  Check(RelDiff(par.reaction_energy, ser.reaction_energy), 1e-8,
        "serial = parallel reaction energy" + p);
  Check(RelDiff(par.l2, ser.l2), 1e-8,
        "serial = parallel physical displacement" + p);
  Check(RelDiff(par.map, par.adjoint), 1e-9,
        "parallel Volterra reciprocity" + p);
  CheckTrue(par.reaction_energy > 0.0, "reaction energy positive" + p);
}

void RunReciprocity(int dim, int order) {
  auto smesh = Mesh(MeshFile(dim).c_str(), 1, 1);
  const auto box = ReadFaultedBox(RecordFile(dim));
  auto pmesh = ParMesh(MPI_COMM_WORLD, smesh);
  auto fec = H1_FECollection(order, dim);
  auto fes = ParFiniteElementSpace(&pmesh, &fec, dim);
  auto kappa = Kappa(dim);
  auto mu = Mu(dim);
  auto rheology = IsotropicElasticRheology(dim, kappa, mu);
  auto zero = Vector(dim);
  zero = 0.0;
  auto traction = VectorConstantCoefficient(zero);
  const auto nb = pmesh.bdr_attributes.Max();
  auto problem = LinearQuasiStaticClampedFaultProblem(
      &fes, rheology,
      Marker(nb, {box.clamped_footwall, box.clamped_hanging_wall}), traction,
      Marker(nb, {}), Marker(pmesh.attributes.Max(), {box.hanging_wall}),
      Marker(nb, {box.fault}), Marker(nb, {box.extension}));
  auto fault_attr = Array<int>({box.fault});
  auto surface_attr = Array<int>({box.surface});
  auto fault_mesh = ParSubMesh::CreateFromBoundary(pmesh, fault_attr);
  auto surface_mesh = ParSubMesh::CreateFromBoundary(pmesh, surface_attr);
  auto fault_fes = SubMeshDofInjection::MakeShadowSpace(fes, fault_mesh);
  auto surface_fes = SubMeshDofInjection::MakeShadowSpace(fes, surface_mesh);
  auto G = FaultToSurfaceMap(problem, *fault_fes, *surface_fes);

  auto gen =
      std::mt19937(1234u + 17u * static_cast<unsigned>(Mpi::WorldRank()));
  auto distrib = std::normal_distribution<>(0, 1);
  auto s = Vector(G.Width());
  auto w = Vector(G.Height());
  for (int i = 0; i < s.Size(); i++) {
    s(i) = distrib(gen);
  }
  for (int i = 0; i < w.Size(); i++) {
    w(i) = distrib(gen);
  }
  auto Gs = Vector();
  auto Gtw = Vector();
  G.Mult(s, Gs);
  G.MultTranspose(w, Gtw);
  const auto lhs = InnerProduct(MPI_COMM_WORLD, Gs, w);
  const auto rhs = InnerProduct(MPI_COMM_WORLD, s, Gtw);
  const auto scale = std::sqrt(InnerProduct(MPI_COMM_WORLD, Gs, Gs) *
                               InnerProduct(MPI_COMM_WORLD, w, w));
  Check(std::abs(lhs - rhs) / scale, 1e-9,
        "random reciprocity" + Label(dim, order));
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();

  for (auto [dim, order] :
       {std::pair{2, 1}, std::pair{2, 2}, std::pair{3, 1}}) {
    RunRigidBlock(dim, order);
    RunSerialParallel(dim, order);
    RunReciprocity(dim, order);
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
