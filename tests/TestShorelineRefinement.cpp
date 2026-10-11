/*
  Tests for RefineToShoreline (mesh.hpp): conforming shoreline-band
  refinement of the curved ball mesh with sphere snapping.

  The configuration is a wavy polar continent (super-Gaussian cap with a
  longitude-modulated width), so the shoreline is a genuinely 2-D curve
  on the sphere. Thresholds encode the measured values noted at each
  gate.
*/

#include <algorithm>
#include <cmath>
#include <numbers>

#include "AdGIA.hpp"
#include "gtest/gtest.h"
#include "mfem.hpp"

using namespace mfem;
using namespace AdGIA;

namespace {

constexpr char kMeshFile[] = "../data/coupled_poisson.msh";

double ShorelineColatitude(double phi) {
  return 0.9 + 0.15 * std::sin(3.0 * phi);
}

double Ocean(const Vector& x) {
  const double r = x.Norml2();
  const double th = std::acos(std::min(1.0, std::max(-1.0, x[2] / r)));
  const double q = th / ShorelineColatitude(std::atan2(x[1], x[0]));
  return 1.0 - std::exp(-q * q * q * q);
}

// Max geometric deviation |r - 1| over surface quadrature points of
// boundary elements within an angular band of the shoreline.
double BandSurfaceDeviation(Mesh& mesh) {
  double dev = 0.0;
  for (int be = 0; be < mesh.GetNBE(); be++) {
    if (mesh.GetBdrAttribute(be) != 1) {
      continue;
    }
    ElementTransformation* T = mesh.GetBdrElementTransformation(be);
    const IntegrationRule& ir =
        IntRules.Get(mesh.GetBdrElementGeometry(be), 4);
    Vector p;
    double d = 0.0;
    bool in_band = false;
    for (int j = 0; j < ir.GetNPoints(); j++) {
      T->Transform(ir.IntPoint(j), p);
      const double r = p.Norml2();
      d = std::max(d, std::abs(r - 1.0));
      const double th = std::acos(std::min(1.0, std::max(-1.0, p[2] / r)));
      if (std::abs(th - ShorelineColatitude(std::atan2(p[1], p[0]))) < 0.1) {
        in_band = true;
      }
    }
    if (in_band) {
      dev = std::max(dev, d);
    }
  }
  return dev;
}

}  // namespace

TEST(ShorelineRefinement, RefinesConformingly) {
  Mesh mesh(kMeshFile, 1, 1);
  const int ne0 = mesh.GetNE();
  FunctionCoefficient ocean(Ocean);
  ShorelineRefinementOptions opt;
  opt.levels = 2;
  const int refined = RefineToShoreline(mesh, ocean, opt);
  EXPECT_GT(refined, 0);
  EXPECT_GT(mesh.GetNE(), ne0);
  EXPECT_TRUE(mesh.Conforming());
  EXPECT_EQ(mesh.CheckElementOrientation(false), 0);
  // Uniform refinement composes AFTER shoreline refinement (the marking
  // precondition is the other way round).
  mesh.UniformRefinement();
  EXPECT_EQ(mesh.CheckElementOrientation(false), 0);
}

TEST(ShorelineRefinement, SnapsToSphere) {
  Mesh mesh(kMeshFile, 1, 1);
  FunctionCoefficient ocean(Ocean);
  const double dev0 = BandSurfaceDeviation(mesh);
  ShorelineRefinementOptions opt;
  opt.levels = 3;
  RefineToShoreline(mesh, ocean, opt);
  // Measured: 4.1e-4 -> 6.1e-6 over three passes.
  EXPECT_GT(dev0, 1e-4);
  EXPECT_LT(BandSurfaceDeviation(mesh), 1e-5);
  // Snapped nodes land on the unit sphere exactly (measured 4.4e-16).
  const GridFunction* nodes = mesh.GetNodes();
  ASSERT_NE(nodes, nullptr);
  const FiniteElementSpace& nfes = *nodes->FESpace();
  double node_dev = 0.0;
  for (int i = 0; i < nfes.GetNDofs(); i++) {
    Vector p(3);
    for (int c = 0; c < 3; c++) {
      p[c] = (*nodes)(nfes.DofToVDof(i, c));
    }
    const double r = p.Norml2();
    if (std::abs(r - 1.0) < 5e-3) {
      node_dev = std::max(node_dev, std::abs(r - 1.0));
    }
  }
  EXPECT_LT(node_dev, 1e-14);
}

TEST(ShorelineRefinement, GeometricWidthControlsBand) {
  // The width option is the direct geometric knob: a wider target marks
  // more of the sphere, independently of the ocean function's steepness.
  auto refine_with_width = [](double w) {
    Mesh mesh(kMeshFile, 1, 1);
    FunctionCoefficient ocean(Ocean);
    ShorelineRefinementOptions opt;
    opt.levels = 2;
    opt.width = w;
    RefineToShoreline(mesh, ocean, opt);
    return mesh.GetNE();
  };
  const int ne_narrow = refine_with_width(0.08);
  const int ne_wide = refine_with_width(0.3);
  EXPECT_GT(ne_narrow, 1756);  // the base mesh's element count
  EXPECT_GT(ne_wide, ne_narrow);
}

TEST(ShorelineRefinement, TargetSizeIsMet) {
  // target_h is the resolution knob: passes continue until shoreline
  // elements reach the target, and the stop is exact by construction
  // (marking tests the same size measure).
  Mesh mesh(kMeshFile, 1, 1);
  FunctionCoefficient ocean(Ocean);
  ShorelineRefinementOptions opt;
  opt.levels = 12;
  opt.target_h = 0.1;
  RefineToShoreline(mesh, ocean, opt);
  // Every element the shoreline crosses (sampled as in the marking) is
  // at or below the target.
  double h_straddle = 0.0;
  Vector pt;
  for (int i = 0; i < mesh.GetNE(); i++) {
    ElementTransformation* T = mesh.GetElementTransformation(i);
    const IntegrationRule& ir =
        GlobGeometryRefiner.Refine(mesh.GetElementBaseGeometry(i), 2, 1)
            ->RefPts;
    double cmin = 2.0, cmax = -1.0;
    for (int j = 0; j < ir.GetNPoints(); j++) {
      T->Transform(ir.IntPoint(j), pt);
      const double r = pt.Norml2();
      if (r < 0.8 || r > 1.2) {
        continue;
      }
      const double c = Ocean(pt);
      cmin = std::min(cmin, c);
      cmax = std::max(cmax, c);
    }
    if (cmin <= 0.5 && 0.5 <= cmax) {
      h_straddle = std::max(h_straddle, mesh.GetElementSize(i));
    }
  }
  EXPECT_GT(h_straddle, 0.0);
  EXPECT_LE(h_straddle, 0.1);
}

TEST(ShorelineRefinement, RadialWindowConfines) {
  Mesh mesh(kMeshFile, 1, 1);
  auto deep_count = [&mesh]() {
    int n = 0;
    Vector c(3);
    for (int i = 0; i < mesh.GetNE(); i++) {
      mesh.GetElementCenter(i, c);
      if (c.Norml2() < 0.5) {
        n++;
      }
    }
    return n;
  };
  EXPECT_GT(deep_count(), 0);
  FunctionCoefficient ocean(Ocean);
  ShorelineRefinementOptions opt;
  opt.levels = 2;
  RefineToShoreline(mesh, ocean, opt);
  // Conforming bisection closure cascades into the interior (measured:
  // 13 deep elements become 216 at two passes), but the deep share of
  // the refined mesh stays marginal (measured 1%).
  EXPECT_LT(deep_count(), 0.03 * mesh.GetNE());
}
