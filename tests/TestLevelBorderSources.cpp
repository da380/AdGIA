#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>

#include "MixedProblemTestCommon.hpp"
#include "TestCommon.hpp"

/*
  Gates for the level border's source hook (AddLevelBorderForce) and its
  exact-levels option (SetLevelBorder(levels, true, grad Phi0)) on the
  mixed class, on the canned fluid-core meshes in 3-D: the three-layer
  ball (a fluid shell between the inner core and the mantle: two walls)
  and the two-layer ball (a fluid core with a centre: one wall, U_0 = 0
  removed).

  - Reciprocity. The bordered system [K B; B^T Db] [x; y] = [b; r] is
    symmetric, so for any two right-hand sides f, g the solutions satisfy
    <x(f), g> = <x(g), f>. Pairing a source on each of the border's rows
    (U, s, nu, the wall rows) with a load on the potential or the solid,
    or with another border row, checks that each source lands on its row
    with the right sign and that the rows are the transposes of the
    columns. The border side of each pairing reads the solution through
    LevelBorderSolution(). 3-D only: in 2-D the class's compatibility
    treatment moves a potential load's mean onto the outer boundary,
    which pairs the potential rows and the border's nu columns
    asymmetrically by design (as for the monolithic route). Measured on
    the canned 2-D annulus with the same data: relative asymmetries of
    0.51 for the two pairings with a potential load, 5.5e-3 (U row /
    traction) and 1.0e-2 (s row / nu row) for the others.
  - Zero sources leave the solution bitwise unchanged; AssembleForce()
    clears the sources; at a centre the U_0 datum is ignored.
  - Exact levels: the border's histogram read off the analytic Phi0 of
    the three-layer model (ThreeLayerBackground) instead of its
    projection. The element volumes sum to the fluid volume either way;
    the cumulative histogram moves toward the analytic one (printed with
    the per-element volumes). The degree-0 agreement with the gauged
    fluid under the switch is gated in TestMixedProblemGauged.
*/

namespace {

using namespace self_grav_test;

bool Verbose() { return std::getenv("ADGIA_TEST_VERBOSE") != nullptr; }

// Integrals int N_i q dc of a function of the level against the hat
// functions of the grid c (5-point Gauss-Legendre per element: exact for
// the polynomial data used below up to degree 8).
Vector HatIntegrals(const Vector& c, const std::function<double(double)>& q) {
  const int n = c.Size() - 1;
  const IntegrationRule& ir = IntRules.Get(Geometry::SEGMENT, 9);
  Vector out(n + 1);
  out = 0.0;
  for (int e = 0; e < n; e++) {
    const double h = c[e + 1] - c[e];
    for (int k = 0; k < ir.GetNPoints(); k++) {
      const IntegrationPoint& ip = ir.IntPoint(k);
      const double v = q(c[e] + h * ip.x) * ip.weight * h;
      out[e] += (1.0 - ip.x) * v;
      out[e + 1] += ip.x * v;
    }
  }
  return out;
}

enum class Model { ThreeLayer, TwoLayer };

// A fluid-core problem closed by the level border, no load registered.
// As DahlenBorderCase of TestMixedProblemGauged: on the three-layer ball
// the core's bulk modulus is Adams-Williamson on the problem's own
// background gravity; on the two-layer ball it is constant, as in
// TestMonolithicBorder. The background is solved for by the class unless
// an analytic one is given; with @p exact_levels the border reads it.
struct BorderCase {
  Model model;
  int dim;
  std::unique_ptr<Mesh> parent;
  std::unique_ptr<SubMesh> solid;
  std::unique_ptr<H1_FECollection> fec;
  std::unique_ptr<FiniteElementSpace> fes_u, fes_phi;
  ConstantCoefficient kappa{kKappa}, mu{kMu}, rho_const{kRho},
      kappa_const{kKappa};
  AWBulkModulus kappa_aw;
  FunctionCoefficient rho_s{SolidDensity}, rho_f{FluidDensity};
  std::unique_ptr<IsotropicElasticRheology> rheology;
  std::vector<FluidRegion> fluids;
  Array<int> fluid_parent, cmb_solid, surface;
  std::unique_ptr<ThreeLayerBackground> bg;
  std::unique_ptr<FunctionCoefficient> phi0;
  std::unique_ptr<VectorFunctionCoefficient> grad_phi0;
  std::unique_ptr<GradientGridFunctionCoefficient> g_parent;
  std::unique_ptr<LinearQuasiStaticMixedSelfGravitatingProblem> problem;

  BorderCase(Model m, int order, int levels, bool analytic = false,
             bool exact_levels = false, int dim_ = 3)
      : model(m), dim(dim_) {
    const bool three = model == Model::ThreeLayer;
    parent = std::make_unique<Mesh>(
        (three ? ThreeLayerMeshFile(dim) : TwoLayerMeshFile(dim)).c_str(), 1,
        1);
    Array<int> attrs = three ? Array<int>({1, 3}) : Array<int>({2});
    solid =
        std::make_unique<SubMesh>(SubMesh::CreateFromDomain(*parent, attrs));
    fec = std::make_unique<H1_FECollection>(order, dim);
    fes_u = std::make_unique<FiniteElementSpace>(solid.get(), fec.get(), dim);
    fes_phi = std::make_unique<FiniteElementSpace>(parent.get(), fec.get());
    rheology = std::make_unique<IsotropicElasticRheology>(dim, kappa, mu);
    surface = SurfaceMarker(*solid);
    fluid_parent.SetSize(parent->attributes.Max());
    fluid_parent = 0;
    cmb_solid.SetSize(solid->bdr_attributes.Max());
    cmb_solid = 0;
    if (three) {
      fluids.push_back(OuterCore(*solid, rho_f));
      fluids.back().bulk_modulus = &kappa_aw;
      fluid_parent[1] = 1;  // the outer core
      cmb_solid[1] = 1;     // boundary attribute 2
    } else {
      FluidRegion core;
      core.attributes = Array<int>({1});
      core.density = &rho_f;
      core.interface_marker.SetSize(solid->bdr_attributes.Max());
      core.interface_marker = 0;
      core.interface_marker[0] = 1;  // the CMB
      core.bulk_modulus = &kappa_const;
      fluids.push_back(core);
      fluid_parent[0] = 1;
      cmb_solid[0] = 1;
    }
    if (analytic) {
      MFEM_VERIFY(three, "the analytic background is the three-layer one");
      bg = std::make_unique<ThreeLayerBackground>(dim);
      phi0 = std::make_unique<FunctionCoefficient>(
          [this](const Vector& x) { return (*bg)(x); });
      grad_phi0 = std::make_unique<VectorFunctionCoefficient>(
          dim, [this](const Vector& x, Vector& g) { bg->Grad(x, g); });
    }
    problem = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        fes_u.get(), fes_phi.get(), *rheology,
        three ? static_cast<Coefficient&>(rho_s) : rho_const, kG, kDtNDegree,
        phi0.get(), fluids);
    g_parent = std::make_unique<GradientGridFunctionCoefficient>(
        &problem->BackgroundPotential());
    kappa_aw.g = g_parent.get();
    if (three) {
      problem->AddRegionRotations(Array<int>({1}));
    }
    problem->SetRelTol(1e-13);
    if (exact_levels) {
      problem->SetLevelBorder(levels, true, grad_phi0.get());
    } else {
      problem->SetLevelBorder(levels);
    }
  }
};

// A solve's state: the fields (L-vectors) and the border unknowns.
struct State {
  Vector u, phi, U, s, nu, lambda;
};

State Capture(LinearQuasiStaticMixedSelfGravitatingProblem& p) {
  State st;
  st.u = p.Displacement();
  st.phi = p.Potential();
  Vector c;
  p.LevelBorderSolution(0, c, st.U, st.s, st.nu, st.lambda);
  return st;
}

// The border data of one solve, as AddLevelBorderForce takes them.
struct BorderData {
  Vector rU, rs, rnu, rlambda;
  double Pair(const State& st) const {
    double v = 0.0;
    if (rU.Size()) v += rU * st.U;  // U_0 = 0 at a centre
    if (rs.Size()) v += rs * st.s;
    if (rnu.Size()) v += rnu * st.nu;
    if (rlambda.Size()) v += rlambda * st.lambda;
    return v;
  }
};

// One solve under exactly one kind of right-hand side.
State SolveWith(BorderCase& bc, const BorderData* border,
                const Vector* potential_load, const Vector* solid_load) {
  auto& p = *bc.problem;
  p.AssembleForce(0.0);
  if (border) {
    p.AddLevelBorderForce(0, border->rU, border->rs, border->rnu,
                          border->rlambda);
  }
  if (potential_load) {
    p.AddPotentialForce(*potential_load);
  }
  if (solid_load) {
    p.AddForce(*solid_load);
  }
  EXPECT_TRUE(p.Solve());
  return Capture(p);
}

double Rel(double a, double b) {
  return std::abs(a - b) / std::max(std::abs(a), std::abs(b));
}

class LevelBorderSources : public testing::TestWithParam<Model> {};

// Reciprocity of the bordered operator, one pairing per row kind.
// Measured (3-D, order 1, 12 levels, solver tolerance 1e-13): relative
// asymmetries between 1e-16 and 7e-15 on both meshes (printed under
// ADGIA_TEST_VERBOSE); the gate 1e-10 leaves room for looser solves.
void Reciprocity(Model model, int dim, double tol) {
  BorderCase bc(model, 1, 12, false, false, dim);
  auto& p = *bc.problem;
  Vector c;
  bool centre;
  std::vector<int> wall_node;
  p.LevelBorderGrid(0, c, centre, wall_node);
  ASSERT_EQ(p.NumLevelBorderRegions(), 1);
  EXPECT_EQ(centre, model == Model::TwoLayer);
  const int n = c.Size() - 1;
  const int nwalls = static_cast<int>(wall_node.size());
  EXPECT_EQ(nwalls, model == Model::TwoLayer ? 1 : 2);
  EXPECT_EQ(wall_node[0], n);
  const double c0 = c[0], c1 = c[n];
  auto xi = [&](double v) { return (v - c0) / (c1 - c0); };

  // Border sources: smooth functions of the level against the hats, and
  // the wall data.
  BorderData r_nu, r_U, r_s, r_l;
  r_nu.rnu = HatIntegrals(c, [&](double v) {
    const double t = xi(v);
    return 1.0 + t - 0.5 * t * t;
  });
  r_U.rU = HatIntegrals(c, [&](double v) {
    const double t = xi(v);
    return 0.3 + t * (1.0 - t) * (1.0 + t);
  });
  r_s.rs = HatIntegrals(c, [&](double v) {
    const double t = xi(v);
    return 1.0 - 2.0 * t * t + t * t * t;
  });
  r_l.rlambda.SetSize(nwalls);
  r_l.rlambda[0] = 1.0;
  if (nwalls > 1) {
    r_l.rlambda[1] = -0.5;
  }

  // A potential load over the fluid and a traction on the solid's CMB.
  FunctionCoefficient q_phi([](const Vector& x) {
    const double z = x[x.Size() - 1];
    return 1.0 + z * z + 0.3 * x[0];
  });
  LinearForm ell(bc.fes_phi.get());
  ell.AddDomainIntegrator(new DomainLFIntegrator(q_phi), bc.fluid_parent);
  ell.Assemble();
  VectorFunctionCoefficient q_t(dim, [](const Vector& x, Vector& t) {
    const double r = x.Norml2();
    t = x;
    t *= 0.1 * (1.0 + 0.5 * x[x.Size() - 1] / r) / r;
    t[0] += 0.02;
  });
  LinearForm trac(bc.fes_u.get());
  trac.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(q_t), bc.cmb_solid);
  trac.Assemble();

  const State x_nu = SolveWith(bc, &r_nu, nullptr, nullptr);
  const State x_ell = SolveWith(bc, nullptr, &ell, nullptr);
  const State x_U = SolveWith(bc, &r_U, nullptr, nullptr);
  const State x_t = SolveWith(bc, nullptr, nullptr, &trac);
  const State x_s = SolveWith(bc, &r_s, nullptr, nullptr);
  const State x_l = SolveWith(bc, &r_l, nullptr, nullptr);

  struct Pairing {
    const char* name;
    double a, b;
  };
  const Pairing pairs[] = {
      {"nu row / potential load", ell * x_nu.phi, r_nu.Pair(x_ell)},
      {"U row / CMB traction", trac * x_U.u, r_U.Pair(x_t)},
      {"s row / nu row", r_nu.Pair(x_s), r_s.Pair(x_nu)},
      {"wall rows / potential load", ell * x_l.phi, r_l.Pair(x_ell)},
  };
  for (const auto& pr : pairs) {
    if (Verbose()) {
      std::cout << (model == Model::TwoLayer ? "two-layer" : "three-layer")
                << " " << dim << "-D " << pr.name << ": " << pr.a << " vs "
                << pr.b << ", relative asymmetry " << Rel(pr.a, pr.b) << "\n";
    }
    EXPECT_GT(std::abs(pr.a), 0.0) << pr.name;
    EXPECT_LT(Rel(pr.a, pr.b), tol) << pr.name;
  }
}

TEST_P(LevelBorderSources, Reciprocity) { Reciprocity(GetParam(), 3, 1e-10); }

bool BitwiseEqual(const Vector& a, const Vector& b) {
  return a.Size() == b.Size() &&
         std::memcmp(a.GetData(), b.GetData(), sizeof(double) * a.Size()) == 0;
}

bool BitwiseEqual(const State& a, const State& b) {
  return BitwiseEqual(a.u, b.u) && BitwiseEqual(a.phi, b.phi) &&
         BitwiseEqual(a.U, b.U) && BitwiseEqual(a.s, b.s) &&
         BitwiseEqual(a.nu, b.nu) && BitwiseEqual(a.lambda, b.lambda);
}

// Zero sources (explicit zeros or empty vectors) leave a loaded solve
// bitwise unchanged; AssembleForce() clears a non-zero source; at a
// centre the U_0 datum is dropped. Separate problems, so that no warm
// start differs between the compared solves.
TEST_P(LevelBorderSources, ZeroSourcesAreInert) {
  const Model model = GetParam();
  FunctionCoefficient sigma(SurfaceLoad);
  auto loaded = [&](const std::function<void(BorderCase&)>& extra) {
    BorderCase bc(model, 1, 12);
    bc.problem->SetSurfaceLoad(sigma, bc.surface);
    bc.problem->AssembleForce(0.0);
    extra(bc);
    EXPECT_TRUE(bc.problem->Solve());
    return Capture(*bc.problem);
  };
  const State base = loaded([](BorderCase&) {});
  const State zeros = loaded([](BorderCase& bc) {
    Vector c;
    bool centre;
    std::vector<int> wall_node;
    bc.problem->LevelBorderGrid(0, c, centre, wall_node);
    Vector z(c.Size()), zl(static_cast<int>(wall_node.size()));
    z = 0.0;
    zl = 0.0;
    bc.problem->AddLevelBorderForce(0, z, z, z, zl);
    bc.problem->AddLevelBorderForce(0, Vector(), Vector(), Vector(), Vector());
  });
  EXPECT_TRUE(BitwiseEqual(base, zeros)) << "zero sources changed the solve";
  const State cleared = loaded([](BorderCase& bc) {
    Vector c;
    bool centre;
    std::vector<int> wall_node;
    bc.problem->LevelBorderGrid(0, c, centre, wall_node);
    Vector one(c.Size());
    one = 1.0;
    bc.problem->AddLevelBorderForce(0, one, one, one, Vector());
    bc.problem->AssembleForce(0.0);
  });
  EXPECT_TRUE(BitwiseEqual(base, cleared))
      << "AssembleForce() did not clear the border sources";
  if (model == Model::TwoLayer) {
    const State u0 = loaded([](BorderCase& bc) {
      Vector c;
      bool centre;
      std::vector<int> wall_node;
      bc.problem->LevelBorderGrid(0, c, centre, wall_node);
      Vector rU(c.Size());
      rU = 0.0;
      rU[0] = 1.0;
      bc.problem->AddLevelBorderForce(0, rU, Vector(), Vector(), Vector());
    });
    EXPECT_TRUE(BitwiseEqual(base, u0)) << "the centre's U_0 datum entered";
  }
}

INSTANTIATE_TEST_SUITE_P(Mesh, LevelBorderSources,
                         testing::Values(Model::ThreeLayer, Model::TwoLayer),
                         [](const testing::TestParamInfo<Model>& info) {
                           return info.param == Model::ThreeLayer
                                      ? std::string("ThreeLayer")
                                      : std::string("TwoLayer");
                         });

// The exact-levels option on the three-layer ball with its analytic
// background: the border built from the projected Phi0 and from the
// analytic one (both problems constructed with the analytic coefficient,
// so only the border's reading of it differs). The element volumes
// partition the fluid either way (each fluid quadrature point lands in
// exactly one element), so both sum to the fluid volume to round-off.
// The per-element and cumulative volumes are printed against the
// analytic shell volumes (4 pi / 3)(r(c_{e+1})^3 - r(c_e)^3) between each
// grid's levels.
TEST(LevelBorderExactLevels, HistogramFromTheAnalyticBackground) {
  const int levels = 12;
  BorderCase proj(Model::ThreeLayer, 1, levels, true, false);
  BorderCase exact(Model::ThreeLayer, 1, levels, true, true);
  const LevelBorder& Lp = proj.problem->LevelBorderData(0);
  const LevelBorder& Le = exact.problem->LevelBorderData(0);
  ASSERT_EQ(Lp.n, Le.n);

  // The fluid volume, by a rule exact for the curved (quadratic) tets'
  // Jacobian (Mesh::GetElementVolume integrates at the Jacobian's order
  // only, which is not exact on curved elements).
  double fluid_volume = 0.0;
  Mesh& mesh = *proj.parent;
  for (int e = 0; e < mesh.GetNE(); e++) {
    if (!proj.fluid_parent[mesh.GetAttribute(e) - 1]) {
      continue;
    }
    ElementTransformation* T = mesh.GetElementTransformation(e);
    const IntegrationRule& ir = IntRules.Get(T->GetGeometryType(), 8);
    for (int q = 0; q < ir.GetNPoints(); q++) {
      T->SetIntPoint(&ir.IntPoint(q));
      fluid_volume += ir.IntPoint(q).weight * T->Weight();
    }
  }
  const double sum_p = Lp.vol.Sum(), sum_e = Le.vol.Sum();
  EXPECT_LT(std::abs(sum_p - fluid_volume) / fluid_volume, 1e-12);
  EXPECT_LT(std::abs(sum_e - fluid_volume) / fluid_volume, 1e-12);

  // The analytic shell volumes between a grid's levels.
  const ThreeLayerBackground& bg = *exact.bg;
  auto radius = [&](double c) {  // Phi0 increases with r in the fluid
    double lo = kMeshRIcb, hi = kMeshRCmb;
    for (int it = 0; it < 200; it++) {
      const double mid = 0.5 * (lo + hi);
      (bg.Phi0(mid) < c ? lo : hi) = mid;
    }
    return 0.5 * (lo + hi);
  };
  auto shell = [&](const LevelBorder& L, int e) {
    const double r0 = radius(L.c[e]), r1 = radius(L.c[e + 1]);
    return 4.0 * M_PI / 3.0 * (r1 * r1 * r1 - r0 * r0 * r0);
  };
  // The cumulative histogram Vol(c_k) = sum_{e<k} Vol_e against the
  // analytic cumulative volume above the inner wall, as a fraction of the
  // fluid volume.
  auto cumulative_error = [&](const LevelBorder& L) {
    double worst = 0.0, sum = 0.0, exact_sum = 0.0;
    for (int e = 0; e < L.n; e++) {
      sum += L.vol[e];
      exact_sum += shell(L, e);
      worst = std::max(worst, std::abs(sum - exact_sum) / fluid_volume);
    }
    return worst;
  };
  double dvol = 0.0, dcum = 0.0, sum_pk = 0.0, sum_ek = 0.0;
  for (int e = 0; e < Lp.n; e++) {
    dvol = std::max(dvol, std::abs(Le.vol[e] - Lp.vol[e]) / Lp.vol[e]);
    sum_pk += Lp.vol[e];
    sum_ek += Le.vol[e];
    dcum = std::max(dcum, std::abs(sum_ek - sum_pk) / fluid_volume);
  }
  const double cum_p = cumulative_error(Lp), cum_e = cumulative_error(Le);
  const double range = Le.c[Le.n] - Le.c[0];
  if (Verbose()) {
    for (int e = 0; e < Lp.n; e++) {
      std::cout << "  element " << e << ": projected " << Lp.vol[e]
                << " (analytic shell " << shell(Lp, e) << "), exact "
                << Le.vol[e] << " (analytic shell " << shell(Le, e) << ")\n";
    }
    std::cout << "fluid volume " << fluid_volume
              << ": histogram sums, relative deviation "
              << (sum_p - fluid_volume) / fluid_volume << " (projected), "
              << (sum_e - fluid_volume) / fluid_volume << " (exact)\n"
              << "end levels: projected [" << Lp.c[0] << ", " << Lp.c[Lp.n]
              << "], exact [" << Le.c[0] << ", " << Le.c[Le.n]
              << "], analytic walls [" << bg.Phi0(kMeshRIcb) << ", "
              << bg.Phi0(kMeshRCmb) << "]\n"
              << "end levels against the analytic walls / level range: "
              << (Le.c[0] - bg.Phi0(kMeshRIcb)) / range << ", "
              << (Le.c[Le.n] - bg.Phi0(kMeshRCmb)) / range << " (exact), "
              << (Lp.c[0] - bg.Phi0(kMeshRIcb)) / range << ", "
              << (Lp.c[Lp.n] - bg.Phi0(kMeshRCmb)) / range << " (projected)\n"
              << "element volumes, exact vs projected: max relative change "
              << dvol << " per element, " << dcum
              << " of the fluid volume cumulatively\n"
              << "cumulative histogram against the analytic one (fraction "
                 "of the fluid volume): projected "
              << cum_p << ", exact " << cum_e << "\n"
              << "wall spreads / level range: projected "
              << (Lp.wall_level_max[0] - Lp.wall_level_min[0]) / range << ", "
              << (Lp.wall_level_max[1] - Lp.wall_level_min[1]) / range
              << "; exact "
              << (Le.wall_level_max[0] - Le.wall_level_min[0]) / range << ", "
              << (Le.wall_level_max[1] - Le.wall_level_min[1]) / range << "\n";
  }
  // The switch moves the histogram, and the exact route's cumulative
  // histogram is the closer to the analytic one. Measured: per element
  // the volumes move by up to 100 % (the bins are lumpy at 12 levels on
  // this coarse mesh: each fluid quadrature point carries a finite
  // volume, and both routes scatter by about 30 % around the analytic
  // shells), cumulatively by 7.1 % of the fluid volume; the cumulative
  // distance to the analytic histogram is 6.2 % projected, 2.1 % exact.
  // The exact grid's end levels sit at the analytic wall values to
  // 6e-4 (inner) and 6e-5 (outer) of the level range (the walls are
  // taken at the area mean of Phi0 over the curved faces).
  EXPECT_GT(dcum, 0.0);
  EXPECT_LT(cum_e, cum_p);
  EXPECT_LT(std::abs(Le.c[Le.n] - bg.Phi0(kMeshRCmb)), 2e-3 * range);
  EXPECT_LT(std::abs(Le.c[0] - bg.Phi0(kMeshRIcb)), 2e-3 * range);
}

// The exact-levels switch is refused without a background coefficient or
// without its gradient, and a gradient without the switch is refused.
TEST(LevelBorderExactLevels, Refusals) {
  Vector z(3);
  z = 0.0;
  VectorConstantCoefficient g(z);
  BorderCase solved(Model::TwoLayer, 1, 12);  // the class solved for Phi0
  EXPECT_DEATH(solved.problem->SetLevelBorder(12, true, &g),
               "exact levels need the background potential");
  EXPECT_DEATH(solved.problem->SetLevelBorder(12, false, &g),
               "a gradient is read only with exact levels");
  BorderCase analytic(Model::ThreeLayer, 1, 12, true);
  EXPECT_DEATH(analytic.problem->SetLevelBorder(12, true, nullptr),
               "exact levels need the gradient");
}

}  // namespace
