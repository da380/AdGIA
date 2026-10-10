#include <cstdio>
#include <fstream>
#include <numbers>

#include "MixedProblemTestCommon.hpp"
#include "TestCommon.hpp"

/*
  Tests for SeaLevelOperator (the WP1 surface layer of the sea-level
  machinery) on the canned two-layer meshes: body attribute 1 (radius 1)
  inside a buffer, the body surface the SubMesh's only boundary.

  1. Trace restriction is nodally exact: a smooth scalar projected on the
     body (one hop) or on the parent (two hops, the mixed class's
     potential route) restricts to exactly its direct surface projection
     (H1 projection is nodal interpolation, so the agreement is to
     round-off for any smooth function).
  2. Manufactured sea-level change: with Phi0 = |x|^2/2 (grad Phi0 = x,
     g = 1 on the unit surface), u = a x, phi = b x0 and psi = c, the raw
     change is -(a + b x0 + c) and the mass-conserving uniform shift
     cancels exactly the constant part: SL1 = -b x0.
  3. The ocean function of an analytic initial state: a half-flooded
     surface (SL0 = -x_last) has half the surface area as ocean, to
     quadrature accuracy across the shoreline-crossing elements.
  4. Conservation diagnostics: surface integrals of a constant and of a
     zero-mean degree-2 pattern against closed forms (curved-geometry
     accuracy).
  5. A mixed-class smoke: the degree-2 surface-load solve, sea level from
     the parent-mesh potential (two hops) and from the body shadow (one
     hop) agree; the ocean mean vanishes after the shift.
*/

namespace {

using namespace self_grav_test;

// (dim, order)
using Param = std::tuple<int, int>;

struct Case {
  std::unique_ptr<Mesh> parent;
  std::unique_ptr<SubMesh> body;
  std::unique_ptr<H1_FECollection> fec_u;
  std::unique_ptr<FiniteElementSpace> fes_u;
  Array<int> surface;

  Case(int dim, int order) {
    parent = std::make_unique<Mesh>(MeshFile(dim).c_str(), 1, 1);
    EXPECT_EQ(parent->Dimension(), dim);
    body = std::make_unique<SubMesh>(
        SubMesh::CreateFromDomain(*parent, BodyMarker(*parent)));
    fec_u = std::make_unique<H1_FECollection>(order, dim);
    fes_u = std::make_unique<FiniteElementSpace>(body.get(), fec_u.get(), dim);
    surface = SurfaceMarker(*body);
  }
};

double Smooth(const Vector& x) {
  double s = 1.0;
  for (int i = 0; i < x.Size(); i++) {
    s += (i + 1) * x[i] + 0.3 * std::sin(x[i]);
  }
  return s;
}

class SeaLevelTest : public testing::TestWithParam<Param> {};

TEST_P(SeaLevelTest, RestrictionIsNodallyExact) {
  const auto [dim, order] = GetParam();
  Case s(dim, order);
  SeaLevelOperator sea(*s.fes_u, s.surface);
  FunctionCoefficient f(Smooth);

  GridFunction direct(&sea.SurfaceSpace()), restricted(&sea.SurfaceSpace());
  direct.ProjectCoefficient(f);

  // One hop: a body field.
  H1_FECollection fec(order, dim);
  FiniteElementSpace body_s(s.body.get(), &fec);
  GridFunction on_body(&body_s);
  on_body.ProjectCoefficient(f);
  sea.Restrict(on_body, restricted);
  restricted -= direct;
  EXPECT_LT(restricted.Normlinf(), 1e-12);

  // Two hops: a parent field (the potential route).
  FiniteElementSpace parent_s(s.parent.get(), &fec);
  GridFunction on_parent(&parent_s);
  on_parent.ProjectCoefficient(f);
  sea.Restrict(on_parent, restricted);
  restricted -= direct;
  EXPECT_LT(restricted.Normlinf(), 1e-12);
}

TEST_P(SeaLevelTest, ManufacturedSeaLevelChange) {
  const auto [dim, order] = GetParam();
  Case s(dim, order);
  SeaLevelOperator sea(*s.fes_u, s.surface);

  constexpr double kA = 0.3, kB = 0.7, kC = 0.2;
  // u = a x on the body.
  GridFunction u(s.fes_u.get());
  VectorFunctionCoefficient uc(dim, [](const Vector& x, Vector& v) {
    v = x;
    v *= kA;
  });
  u.ProjectCoefficient(uc);
  // phi = b x0 on the parent (exercises the two-hop route with data).
  H1_FECollection fec(order, dim);
  FiniteElementSpace parent_s(s.parent.get(), &fec);
  GridFunction phi(&parent_s);
  FunctionCoefficient phic([](const Vector& x) { return kB * x[0]; });
  phi.ProjectCoefficient(phic);
  // Phi0 = |x|^2 / 2: grad Phi0 = x, g = |x| = 1 on the unit surface.
  VectorFunctionCoefficient grad_phi0(dim,
                                      [](const Vector& x, Vector& v) { v = x; });
  ConstantCoefficient psi(kC);

  auto info = sea.SeaLevelChange(u, grad_phi0, phi, &psi);

  // Raw change -(a + b x0 + c); the all-ocean mass shift removes the
  // constants (int x0 dS = 0 on the sphere), leaving -b x0.
  FunctionCoefficient exact([](const Vector& x) { return -kB * x[0]; });
  GridFunction err(sea.SeaLevelChangeField());
  GridFunction ex(&sea.SurfaceSpace());
  ex.ProjectCoefficient(exact);
  err -= ex;
  // The surface mesh is curved order-`order`: |x| = 1 only to geometric
  // accuracy, so g and the node positions carry small geometry errors.
  EXPECT_LT(err.Normlinf(), 1e-3);
  EXPECT_NEAR(info.uniform, kA + kC, 2e-3);
  // The ocean mean vanishes after the shift by construction.
  GridFunctionCoefficient slc(&sea.SeaLevelChangeField());
  EXPECT_LT(std::abs(sea.OceanIntegral(slc)), 1e-10 + 1e-8 * info.ocean_area);
}

TEST_P(SeaLevelTest, OceanFunctionHalfFlooded) {
  const auto [dim, order] = GetParam();
  Case s(dim, order);
  SeaLevelOperator sea(*s.fes_u, s.surface);

  const double full = sea.OceanArea();  // all-ocean before an initial state
  const double exact_full = dim == 2 ? 2.0 * std::numbers::pi : 4.0 * std::numbers::pi;
  EXPECT_NEAR(full, exact_full, 1e-2 * exact_full);

  // Ocean where -x_last > 0: half the surface, cut through elements.
  FunctionCoefficient sl0(
      [](const Vector& x) { return -x[x.Size() - 1]; });
  ConstantCoefficient no_ice(0.0);
  sea.SetInitialState(sl0, no_ice);
  EXPECT_NEAR(sea.OceanArea(), 0.5 * exact_full, 0.05 * exact_full);

  // The nodal display indicator agrees with the rule at the nodes.
  GridFunction c(&sea.SurfaceSpace());
  sea.OceanFunction(c);
  EXPECT_LE(c.Normlinf(), 1.0);
  GridFunctionCoefficient cc(&c);
  const double display_area = sea.SurfaceIntegral(cc);
  EXPECT_NEAR(display_area, 0.5 * exact_full, 0.1 * exact_full);
}

TEST_P(SeaLevelTest, SurfaceIntegralsAgainstClosedForms) {
  const auto [dim, order] = GetParam();
  Case s(dim, order);
  SeaLevelOperator sea(*s.fes_u, s.surface);

  // A zero-mean degree-2 pattern: d c^2 - 1 with c = x_last / |x|
  // integrates to zero over the circle and the sphere alike.
  FunctionCoefficient p2([dim](const Vector& x) {
    const double c = x[x.Size() - 1] / x.Norml2();
    return dim * c * c - 1.0;
  });
  const double exact_full = dim == 2 ? 2.0 * std::numbers::pi : 4.0 * std::numbers::pi;
  EXPECT_LT(std::abs(sea.SurfaceIntegral(p2)), 1e-2 * exact_full);
}

TEST_P(SeaLevelTest, MixedProblemSmoke) {
  const auto [dim, order] = GetParam();
  if (dim == 3 && order > 1) {
    GTEST_SKIP() << "3-D order-2 solve is covered by the mixed suites";
  }
  Case s(dim, order);
  H1_FECollection fec_phi(order, dim);
  FiniteElementSpace fes_phi(s.parent.get(), &fec_phi);
  ConstantCoefficient kappa(kKappa), mu(kMu), rho(kRho);
  IsotropicElasticRheology rheology(dim, kappa, mu);
  LinearQuasiStaticMixedSelfGravitatingProblem prob(s.fes_u.get(), &fes_phi,
                                                    rheology, rho, kG,
                                                    kDtNDegree);
  FunctionCoefficient sigma(SurfaceLoad);
  prob.SetSurfaceLoad(sigma, s.surface);
  prob.SetRelTol(1e-11);
  prob.AssembleForce(0.0);
  ASSERT_TRUE(prob.Solve());

  // Uniform-ball background: grad Phi0 = (2 pi G rho) x in 2-D,
  // (4 pi G rho / 3) x in 3-D (outward, physicist convention).
  const double g0 =
      dim == 2 ? 2.0 * std::numbers::pi * kG * kRho : 4.0 * std::numbers::pi * kG * kRho / 3.0;
  VectorFunctionCoefficient grad_phi0(dim, [g0](const Vector& x, Vector& v) {
    v = x;
    v *= g0;
  });

  SeaLevelOperator sea(*s.fes_u, s.surface);
  auto info = sea.SeaLevelChange(prob.Displacement(), grad_phi0,
                                 prob.Potential());
  GridFunction from_parent(sea.SeaLevelChangeField());
  EXPECT_GT(from_parent.Normlinf(), 0.0);
  // The same from the body-shadow potential (one hop).
  sea.SeaLevelChange(prob.Displacement(), grad_phi0,
                     prob.PotentialOnBody());
  GridFunction diff(from_parent);
  diff -= sea.SeaLevelChangeField();
  EXPECT_LT(diff.Normlinf(), 1e-10 * from_parent.Normlinf() + 1e-14);
  // Ocean mean zero after the shift.
  GridFunctionCoefficient slc(&sea.SeaLevelChangeField());
  EXPECT_LT(std::abs(sea.OceanIntegral(slc)),
            1e-8 * from_parent.Normlinf() * info.ocean_area + 1e-12);
}

TEST(SeaLevelExport, NodalCsvRoundTrips) {
  Case s(3, 2);
  SeaLevelOperator sea(*s.fes_u, s.surface);
  FunctionCoefficient f(Smooth);
  GridFunction field(&sea.SurfaceSpace());
  field.ProjectCoefficient(f);
  const std::string path = "sea_level_export_test.csv";
  sea.WriteSurfaceField(field, path);

  std::ifstream in(path);
  ASSERT_TRUE(in.good());
  std::string header;
  std::getline(in, header);
  EXPECT_EQ(header, "x,y,z,value");
  int rows = 0;
  double worst = 0.0;
  std::string line;
  while (std::getline(in, line)) {
    double x[3], v;
    ASSERT_EQ(std::sscanf(line.c_str(), "%lf,%lf,%lf,%lf", &x[0], &x[1],
                          &x[2], &v),
              4);
    Vector p(x, 3);
    worst = std::max(worst, std::abs(v - Smooth(p)));
    rows++;
  }
  // One row per node, values exact at the nodes.
  EXPECT_EQ(rows, sea.SurfaceSpace().GetVSize());
  EXPECT_LT(worst, 1e-12);

  // The ingest leg: reading the file back (coordinate-matched) returns
  // the nodal values exactly.
  GridFunction back(&sea.SurfaceSpace());
  back = 0.0;
  const auto names = sea.ReadSurfaceField(back, path);
  ASSERT_EQ(names.size(), 1u);
  EXPECT_EQ(names[0], "value");
  back -= field;
  EXPECT_LT(back.Normlinf(), 1e-12);
  std::remove(path.c_str());
}

TEST(SeaLevelExport, IceHistoryInterpolates) {
  Case s(3, 2);
  SeaLevelOperator sea(*s.fes_u, s.surface);
  FunctionCoefficient f(Smooth);
  GridFunction field(&sea.SurfaceSpace());
  field.ProjectCoefficient(f);
  const std::string single = "ice_history_single_test.csv";
  sea.WriteSurfaceField(field, single);

  // A two-time stack from the written file: I(t=1) = the field,
  // I(t=3) = 2 field + 5 (the header names the columns by their times).
  const std::string stacked = "ice_history_stack_test.csv";
  {
    std::ifstream in(single);
    std::ofstream out(stacked);
    out.precision(16);
    std::string line;
    std::getline(in, line);
    out << "x,y,z,1,3\n";
    while (std::getline(in, line)) {
      double x[3], v;
      ASSERT_EQ(std::sscanf(line.c_str(), "%lf,%lf,%lf,%lf", &x[0], &x[1],
                            &x[2], &v),
                4);
      out << x[0] << ',' << x[1] << ',' << x[2] << ',' << v << ','
          << 2.0 * v + 5.0 << '\n';
    }
  }

  IceHistory ice(sea, stacked);
  ASSERT_EQ(ice.NumTimes(), 2);
  EXPECT_DOUBLE_EQ(ice.Time(0), 1.0);
  EXPECT_DOUBLE_EQ(ice.Time(1), 3.0);
  {
    GridFunction d(ice.Field(0));
    d -= field;
    EXPECT_LT(d.Normlinf(), 1e-12);
  }

  // The interpolant against closed forms of its surface integral:
  // S(t) = (1 + (t-1)/2) S0 + 5 A (t-1)/2 on [1,3], clamped outside;
  // the change is the increment from t = 1.
  ConstantCoefficient one(1.0);
  const double area = sea.SurfaceIntegral(one);
  // The stack stores the NODAL field, so the reference integral is the
  // field's own (the analytic coefficient differs by interpolation
  // error on the curved surface).
  GridFunctionCoefficient fc(&field);
  const double s0 = sea.SurfaceIntegral(fc);
  auto integral_at = [&](Coefficient& c, double t) {
    c.SetTime(t);
    return sea.SurfaceIntegral(c);
  };
  Coefficient& I = ice.Interpolant();
  EXPECT_NEAR(integral_at(I, 0.0), s0, 1e-10 * std::abs(s0));  // clamped
  EXPECT_NEAR(integral_at(I, 1.0), s0, 1e-10 * std::abs(s0));
  EXPECT_NEAR(integral_at(I, 2.0), 1.5 * s0 + 2.5 * area,
              1e-10 * (std::abs(s0) + area));
  EXPECT_NEAR(integral_at(I, 3.0), 2.0 * s0 + 5.0 * area,
              1e-10 * (std::abs(s0) + area));
  EXPECT_NEAR(integral_at(I, 9.0), 2.0 * s0 + 5.0 * area,
              1e-10 * (std::abs(s0) + area));  // clamped
  Coefficient& dI = ice.Change();
  EXPECT_NEAR(integral_at(dI, 1.0), 0.0, 1e-10 * (std::abs(s0) + area));
  EXPECT_NEAR(integral_at(dI, 3.0), s0 + 5.0 * area,
              1e-10 * (std::abs(s0) + area));

  // Body-side evaluation (the Extend leg): the same integral assembled
  // as a boundary form on the body mesh.
  {
    H1_FECollection sfec(2, 3);
    FiniteElementSpace sfes(s.body.get(), &sfec);
    LinearForm lf(&sfes);
    I.SetTime(3.0);
    lf.AddBoundaryIntegrator(new BoundaryLFIntegrator(I), s.surface);
    lf.Assemble();
    EXPECT_NEAR(lf.Sum(), 2.0 * s0 + 5.0 * area,
                1e-10 * (std::abs(s0) + area));
  }

  std::remove(single.c_str());
  std::remove(stacked.c_str());
}

INSTANTIATE_TEST_SUITE_P(SeaLevel, SeaLevelTest,
                         testing::Values(Param{2, 1}, Param{2, 2}, Param{3, 1},
                                         Param{3, 2}));

}  // namespace
