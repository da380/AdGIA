/*
  Serial tests of LinearQuasiStaticClampedFaultProblem and
  FaultToSurfaceMap on the faulted boxes (meshes/faulted_box.py), in 2-D
  (orders 1 and 2) and 3-D (order 1):

  - The mesh and its record: the advertised attributes, the fault and its
    extension as interior faces, the hanging wall on the side the normal
    points to.
  - Rigid block: the whole cut as fault, only the footwall clamped, a
    uniform slip. The hanging wall must translate rigidly, u_w = s on its
    dofs off the cut and zero elsewhere, and the physical displacement is
    exactly s on the hanging wall and zero on the footwall.
  - The contained fault: the slip is zero on the fault-edge dofs and
    interpolates the coefficient on the others.
  - The fault reaction is symmetric and positive in the slip (the Schur
    complement of the broken stiffness), and equals minus the footwall's
    reaction (equilibrium of the fault rows).
  - FaultToSurfaceMap: Mult equals a direct solve with the same slip, the
    adjoint satisfies <G s, w> = <s, G^T w>, and G^T w vanishes on the
    fault-edge dofs.
*/

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "FaultTestCommon.hpp"
#include "mfem.hpp"
#include "AdGIA.hpp"

using namespace mfem;
using namespace AdGIA;
using namespace fault_test;

namespace {

// (dimension, order) of the cases.
const std::vector<std::pair<int, int>> kCases = {{2, 1}, {2, 2}, {3, 1}};

std::string Label(int dim, int order) {
  return std::to_string(dim) + "-D, order " + std::to_string(order);
}

Vector RandomVector(int n, unsigned seed) {
  auto gen = std::mt19937(seed);
  auto distrib = std::normal_distribution<>(0, 1);
  auto v = Vector(n);
  for (int i = 0; i < n; i++) {
    v(i) = distrib(gen);
  }
  return v;
}

// Depth-stiffened moduli with a lateral gradient: the tests must not lean
// on a homogeneous body.
struct Material {
  explicit Material(int dim)
      : kappa([dim](const Vector& x) {
          return 2.0 * (1.0 - 0.3 * x(dim - 1)) * (1.0 + 0.05 * x(0));
        }),
        mu([dim](const Vector& x) {
          return (1.0 - 0.3 * x(dim - 1)) * (1.0 + 0.05 * x(0));
        }),
        rheology(dim, kappa, mu) {}
  FunctionCoefficient kappa, mu;
  IsotropicElasticRheology rheology;
};

// The problem on a canned mesh: either the contained fault (the example's
// configuration, every artificial side clamped) or the whole cut as fault
// with only the footwall clamped.
struct FaultCase {
  FaultCase(int dim, int order, bool whole_cut = false)
      : mesh(MeshFile(dim).c_str(), 1, 1),
        box(ReadFaultedBox(RecordFile(dim))),
        fec(order, dim),
        fes(&mesh, &fec, dim),
        material(dim),
        zero(dim) {
    zero = 0.0;
    traction = std::make_unique<VectorConstantCoefficient>(zero);
    const auto nb = mesh.bdr_attributes.Max();
    const auto na = mesh.attributes.Max();
    ess = whole_cut
              ? Marker(nb, {box.clamped_footwall})
              : Marker(nb, {box.clamped_footwall, box.clamped_hanging_wall});
    no_traction = Marker(nb, {});
    hanging_wall = Marker(na, {box.hanging_wall});
    fault = whole_cut ? Marker(nb, {box.fault, box.extension})
                      : Marker(nb, {box.fault});
    edge = whole_cut ? Marker(nb, {}) : Marker(nb, {box.extension});
    problem = std::make_unique<LinearQuasiStaticClampedFaultProblem>(
        &fes, material.rheology, ess, *traction, no_traction, hanging_wall,
        fault, edge);
  }

  Mesh mesh;
  FaultedBox box;
  H1_FECollection fec;
  FiniteElementSpace fes;
  Material material;
  Vector zero;
  std::unique_ptr<VectorConstantCoefficient> traction;
  Array<int> ess, no_traction, hanging_wall, fault, edge;
  std::unique_ptr<LinearQuasiStaticClampedFaultProblem> problem;
};

Vector UniformSlip(int dim) {
  auto s = Vector(dim);
  s(0) = 0.3;
  s(dim - 1) = -0.2;
  if (dim == 3) {
    s(1) = 0.1;
  }
  return s;
}

}  // namespace

TEST(FaultProblem, MeshRecord) {
  for (auto dim : {2, 3}) {
    auto mesh = Mesh(MeshFile(dim).c_str(), 1, 1);
    const auto box = ReadFaultedBox(RecordFile(dim));
    EXPECT_EQ(box.dim, dim);
    EXPECT_EQ(mesh.Dimension(), dim);
    EXPECT_EQ(mesh.attributes.Max(), 2);
    EXPECT_EQ(mesh.bdr_attributes.Max(), 5);

    int wrong_side = 0;
    auto centre = Vector(dim);
    for (int e = 0; e < mesh.GetNE(); e++) {
      mesh.GetElementCenter(e, centre);
      centre -= box.centre;
      const bool above = centre * box.normal > 0.0;
      wrong_side += above != (mesh.GetAttribute(e) == box.hanging_wall);
    }
    EXPECT_EQ(wrong_side, 0) << dim << "-D";

    int fault_faces = 0, extension_faces = 0, exterior_cut_faces = 0;
    for (int b = 0; b < mesh.GetNBE(); b++) {
      const auto a = mesh.GetBdrAttribute(b);
      if (a != box.fault && a != box.extension) {
        continue;
      }
      (a == box.fault ? fault_faces : extension_faces)++;
      int e1, e2;
      mesh.GetFaceElements(mesh.GetBdrElementFaceIndex(b), &e1, &e2);
      exterior_cut_faces += e2 < 0;
    }
    EXPECT_GT(fault_faces, 0) << dim << "-D";
    EXPECT_GT(extension_faces, 0) << dim << "-D";
    EXPECT_EQ(exterior_cut_faces, 0) << dim << "-D";
  }
}

TEST(FaultProblem, RigidBlock) {
  for (auto [dim, order] : kCases) {
    auto s = FaultCase(dim, order, /*whole_cut=*/true);
    const auto slip = UniformSlip(dim);
    auto slip_c = VectorConstantCoefficient(slip);
    s.problem->SetSlip(slip_c);
    s.problem->AssembleForce(0.0);
    ASSERT_TRUE(s.problem->Solve());
    EXPECT_LT(RigidBlockError(s.problem->Displacement(), s.fault,
                              s.box.hanging_wall, slip),
              1e-8 * slip.Norml2())
        << Label(dim, order);

    // The physical displacement, projected onto a discontinuous space
    // with the same nodes: s on the hanging wall, zero on the footwall.
    auto dg = L2_FECollection(order, dim, BasisType::GaussLobatto);
    auto dg_fes = FiniteElementSpace(&s.mesh, &dg, dim);
    auto u_dg = GridFunction(&dg_fes);
    u_dg.ProjectCoefficient(s.problem->PhysicalDisplacement());
    auto zero_c = VectorConstantCoefficient(s.zero);
    Array<VectorCoefficient*> pieces(2);
    pieces[s.box.footwall - 1] = &zero_c;
    pieces[s.box.hanging_wall - 1] = &slip_c;
    auto attrs = Array<int>({1, 2});
    auto exact = PWVectorCoefficient(dim, attrs, pieces);
    EXPECT_LT(u_dg.ComputeL2Error(exact), 1e-8 * slip.Norml2())
        << Label(dim, order);
  }
}

TEST(FaultProblem, SlipVanishesOnTheFaultEdge) {
  for (auto [dim, order] : kCases) {
    auto s = FaultCase(dim, order);
    auto slip = Vector(dim);
    slip = 0.4;
    slip(0) = 0.7;
    auto slip_c = VectorConstantCoefficient(slip);
    s.problem->SetSlip(slip_c);
    auto fault_vdofs = Array<int>();
    auto edge_vdofs = Array<int>();
    s.fes.GetEssentialVDofs(s.fault, fault_vdofs);
    s.fes.GetEssentialVDofs(s.edge, edge_vdofs);
    const auto& sh = s.problem->Slip();
    int edge_vdof_count = 0, carrying_dofs = 0;
    real_t err = 0.0;
    for (int i = 0; i < s.fes.GetNDofs(); i++) {
      for (int c = 0; c < dim; c++) {
        const auto v = s.fes.DofToVDof(i, c);
        const bool on_edge = fault_vdofs[v] && edge_vdofs[v];
        const bool carries = fault_vdofs[v] && !on_edge;
        edge_vdof_count += on_edge;
        carrying_dofs += carries;
        err = std::max(err, std::abs(sh(v) - (carries ? slip(c) : 0.0)));
      }
    }
    // The edge: the two tips in 2-D; in 3-D at order 1, the mesh vertices
    // the fault shares with its extension.
    int edge_vertices = 2;
    if (dim == 3) {
      auto on_fault = std::set<int>();
      auto shared = std::set<int>();
      auto v = Array<int>();
      for (int b = 0; b < s.mesh.GetNBE(); b++) {
        if (s.mesh.GetBdrAttribute(b) == s.box.fault) {
          s.mesh.GetBdrElementVertices(b, v);
          on_fault.insert(v.begin(), v.end());
        }
      }
      for (int b = 0; b < s.mesh.GetNBE(); b++) {
        if (s.mesh.GetBdrAttribute(b) == s.box.extension) {
          s.mesh.GetBdrElementVertices(b, v);
          for (auto i : v) {
            if (on_fault.count(i)) {
              shared.insert(i);
            }
          }
        }
      }
      edge_vertices = static_cast<int>(shared.size());
      EXPECT_GT(edge_vertices, 4);
    }
    EXPECT_EQ(edge_vdof_count, dim * edge_vertices) << Label(dim, order);
    EXPECT_LT(err, 1e-14 * slip.Normlinf()) << Label(dim, order);
    EXPECT_EQ(s.problem->SlipTrueDofs().Size(), carrying_dofs);
  }
}

TEST(FaultProblem, ReactionSymmetricPositiveBalanced) {
  for (auto [dim, order] : kCases) {
    auto s = FaultCase(dim, order);
    auto slip1 = TaperedSlip(s.box, 0.01);
    auto slip2 =
        VectorFunctionCoefficient(dim, [dim](const Vector& x, Vector& v) {
          v.SetSize(dim);
          for (int i = 0; i < dim; i++) {
            v(i) = std::sin(3.0 * x(0) + i) + 0.5 * x(dim - 1) * (i + 1);
          }
        });

    auto reaction = [&](VectorCoefficient& c, Vector& slip_true, Vector& r) {
      s.problem->SetSlip(c);
      s.problem->AssembleForce(0.0);
      EXPECT_TRUE(s.problem->Solve());
      s.problem->Slip().GetTrueDofs(slip_true);
      s.problem->FaultReaction(r);
    };
    auto s1 = Vector();
    auto s2 = Vector();
    auto r1 = Vector();
    auto r2 = Vector();
    reaction(slip1, s1, r1);

    // Equilibrium of the fault rows: the hanging wall's reaction is minus
    // the footwall's.
    auto footwall = Marker(s.mesh.attributes.Max(), {s.box.footwall});
    auto fw = LinearForm(&s.fes);
    fw.AddDomainIntegrator(
        new DomainLFFormActionIntegrator(s.problem->StiffnessIntegrators(),
                                         s.problem->Displacement()),
        footwall);
    fw.Assemble();
    real_t balance = 0.0;
    for (auto i : s.problem->SlipTrueDofs()) {
      balance = std::max(balance, std::abs(r1(i) + fw[i]));
    }
    EXPECT_LT(balance, 1e-8 * r1.Normlinf()) << Label(dim, order);

    reaction(slip2, s2, r2);
    const auto a = r1 * s2;
    const auto b = r2 * s1;
    EXPECT_LT(std::abs(a - b), 1e-9 * r1.Norml2() * s2.Norml2())
        << Label(dim, order);
    EXPECT_GT(r1 * s1, 0.0) << Label(dim, order);
    EXPECT_GT(r2 * s2, 0.0) << Label(dim, order);
  }
}

TEST(FaultProblem, FaultToSurfaceMap) {
  for (auto [dim, order] : kCases) {
    auto s = FaultCase(dim, order);
    auto fault_attr = Array<int>({s.box.fault});
    auto surface_attr = Array<int>({s.box.surface});
    auto fault_mesh = SubMesh::CreateFromBoundary(s.mesh, fault_attr);
    auto surface_mesh = SubMesh::CreateFromBoundary(s.mesh, surface_attr);
    auto fault_fes = SubMeshDofInjection::MakeShadowSpace(s.fes, fault_mesh);
    auto surface_fes =
        SubMeshDofInjection::MakeShadowSpace(s.fes, surface_mesh);
    auto G = FaultToSurfaceMap(*s.problem, *fault_fes, *surface_fes);
    ASSERT_EQ(G.Width(), fault_fes->GetTrueVSize());
    ASSERT_EQ(G.Height(), surface_fes->GetTrueVSize());

    // Mult against a direct solve with the same (interpolated) slip.
    auto slip_c = TaperedSlip(s.box, 0.01);
    auto slip_f = GridFunction(fault_fes.get());
    slip_f.ProjectCoefficient(slip_c);
    auto u_map = Vector();
    G.Mult(slip_f, u_map);
    EXPECT_TRUE(G.Converged());
    s.problem->SetSlip(slip_c);
    s.problem->AssembleForce(0.0);
    ASSERT_TRUE(s.problem->Solve());
    auto surface_injection = SubMeshDofInjection(*surface_fes, s.fes);
    auto u_direct = Vector(surface_fes->GetTrueVSize());
    surface_injection.MultTranspose(s.problem->Displacement(), u_direct);
    u_direct -= u_map;
    EXPECT_LT(u_direct.Normlinf(), 1e-9 * u_map.Normlinf())
        << Label(dim, order);

    // Reciprocity on random vectors, and no adjoint traction on the
    // fault-edge dofs.
    const auto sv = RandomVector(G.Width(), 11u + order + 7u * dim);
    const auto wv = RandomVector(G.Height(), 23u + order + 7u * dim);
    auto Gs = Vector();
    auto Gtw = Vector();
    G.Mult(sv, Gs);
    G.MultTranspose(wv, Gtw);
    EXPECT_TRUE(G.Converged());
    const auto lhs = Gs * wv;
    const auto rhs = sv * Gtw;
    EXPECT_LT(std::abs(lhs - rhs), 1e-9 * Gs.Norml2() * wv.Norml2())
        << Label(dim, order);

    auto fault_injection = SubMeshDofInjection(*fault_fes, s.fes);
    auto edge_vdofs = Array<int>();
    s.fes.GetEssentialVDofs(s.edge, edge_vdofs);
    real_t on_edge = 0.0;
    for (int i = 0; i < Gtw.Size(); i++) {
      if (edge_vdofs[fault_injection.ParentVDofs()[i]]) {
        on_edge = std::max(on_edge, std::abs(Gtw(i)));
      }
    }
    EXPECT_EQ(on_edge, 0.0) << Label(dim, order);
  }
}
