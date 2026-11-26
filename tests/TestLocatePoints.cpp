/*
  Tests for LocatePoints (mesh.hpp): the elements that hold given points.

  On the curved disc and ball: points made inside known elements, the
  mesh's own vertices (each on the faces of several elements) and points
  outside the mesh. A point found must be reproduced by its element and
  reference coordinates; mfem's Mesh::FindPoints must find the same
  points.
*/

#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "AdGIA.hpp"
#include "mfem.hpp"

namespace {

using namespace mfem;
using namespace AdGIA;

class LocatePointsTest : public testing::TestWithParam<const char*> {};

TEST_P(LocatePointsTest, FindsTheElementsThatHoldThePoints) {
  auto mesh = Mesh(GetParam(), 1, 1);
  const auto dim = mesh.Dimension();
  auto random = std::mt19937(7);
  auto unit = std::uniform_real_distribution<double>(0.0, 1.0);

  // A point inside each element, every vertex, and points outside.
  const auto n_inside = mesh.GetNE(), n_vertices = mesh.GetNV(), n_outside = 5;
  auto points = DenseMatrix(dim, n_inside + n_vertices + n_outside);
  auto x = Vector(dim);
  for (auto e = 0; e < n_inside; e++) {
    // A random point of the reference simplex, away from its faces.
    auto ip = IntegrationPoint();
    double w[4], sum = 0.0;
    for (auto k = 0; k <= dim; k++) {
      w[k] = 0.1 + unit(random);
      sum += w[k];
    }
    ip.Set3(w[0] / sum, w[1] / sum, dim == 3 ? w[2] / sum : 0.0);
    mesh.GetElementTransformation(e)->Transform(ip, x);
    points.SetCol(e, x);
  }
  for (auto v = 0; v < n_vertices; v++) {
    for (auto d = 0; d < dim; d++) {
      x[d] = mesh.GetNodes() ? (*mesh.GetNodes())(mesh.GetNodalFESpace()
                                                      ->DofToVDof(v, d))
                             : mesh.GetVertex(v)[d];
    }
    points.SetCol(n_inside + v, x);
  }
  for (auto k = 0; k < n_outside; k++) {
    x = 0.0;
    x[k % dim] = 5.0 + k;
    points.SetCol(n_inside + n_vertices + k, x);
  }

  auto elements = Array<int>();
  auto ips = Array<IntegrationPoint>();
  const auto found = LocatePoints(mesh, points, elements, ips);
  EXPECT_EQ(found, n_inside + n_vertices);

  auto back = Vector(dim);
  for (auto p = 0; p < points.Width(); p++) {
    if (p >= n_inside + n_vertices) {
      EXPECT_EQ(elements[p], -1);
      continue;
    }
    ASSERT_GE(elements[p], 0) << "point " << p;
    if (p < n_inside) {
      EXPECT_EQ(elements[p], p);
    }
    mesh.GetElementTransformation(elements[p])->Transform(ips[p], back);
    points.GetColumn(p, x);
    back -= x;
    EXPECT_LT(back.Normlinf(), 1e-9);
  }

  auto mfem_elements = Array<int>();
  auto mfem_ips = Array<IntegrationPoint>();
  EXPECT_EQ(mesh.FindPoints(points, mfem_elements, mfem_ips, false), found);
}

INSTANTIATE_TEST_SUITE_P(Meshes, LocatePointsTest,
                         testing::Values("../data/disc_buffer_2d.msh",
                                         "../data/ball_buffer_coarse_3d.msh"));

}  // namespace
