// ============================================================================
// self_gravitating_solvers.cpp
//
// One physical problem, the main linear-solver architectures of the
// library: a two-layer self-gravitating body (fluid core, solid mantle)
// under a degree-2 surface mass load, solved through
//
//   1. dahlen / Schur CG        the Eulerian formulation with the fluid
//                               eliminated (doc/self_gravitation.md;
//                               doc/quasi_static_models.tex, "Fluid
//                               regions and the ladder of CMB
//                               approximations"): the potential block is
//                               eliminated and CG runs on the symmetric
//                               Schur complement in the displacement
//                               (positive on the complement of the rigid
//                               modes) — few outer iterations, one inner
//                               potential solve each;
//   2. dahlen / block MINRES    the same discrete system as one symmetric
//                               indefinite block operator under MINRES
//                               with a block-diagonal preconditioner — no
//                               nested solves, more (cheaper) iterations;
//   3. gauged / block MINRES    the fluid joins the displacement space
//                               with its bulk modulus and a deviatoric
//                               gauge penalty eps Q (doc/gauged_fluid.md);
//                               the solver interleaves Tikhonov gauge
//                               refinements, each one more linear solve,
//                               removing the O(eps) bias;
//   4. referential              the welded gauged REFERENTIAL formulation
//                               (doc/gravitating_elasticity.md): bare
//                               moduli, S_e = -p0 1, potential unknown
//                               zeta; projected block MINRES — the rigid
//                               modes are projected out of the Krylov
//                               space rather than pinned — plus the gauge
//                               refinements;
//   5. maxwell                  the referential problem of 4 with the
//                               core an artificial MAXWELL solid,
//                               relaxed to the secular static state
//                               under the Heaviside load
//                               (SetMaxwellFluid; doc/
//                               static_fluid_core.tex): backward Euler
//                               on the memory displacement, stopping on
//                               the SOLID increment, beta escalation on
//                               stagnation — no gauge refinements, no
//                               slip machinery, works on any geometry;
//   6. slip                     the slipping fluid-solid interface
//                               (doc/slip_interface.tex): broken
//                               displacement pair, single-valued zeta,
//                               the normal-jump constraint by penalty +
//                               augmented Lagrangian — every AL iteration
//                               is a full projected-MINRES solve of the
//                               three-block system;
//   7. slip broken-zeta         the same with the potential broken too
//                               (per-region zeta, the scalar-jump
//                               constraint joining the AL loop): the
//                               four-block system, no fluid extension.
//
// Not included: the KKT multiplier enforcement of the slip constraint
// (EnableKKT; see slipping_interface.cpp, -enforce kkt, and
// doc/slip_interface.tex, "KKT enforcement" — a multiplier space one order
// below the displacement can make the gauge refinements diverge in 2-D,
// "The multiplier space"), and the exact-gauge KKT of the mixed problem
// (EnableGaugeKKT), which is not competitive (doc/slip_interface.tex,
// "Why the fluid gauge stays a penalty").
//
// All seven must agree on the observables (the mantle displacement, modulo
// rigid modes) at the level of the discretisations; the table prints the
// unknowns, the outer iterations, the wall times and that agreement, so
// the cost of each architecture can be read against what it buys:
// robustness (MINRES), a tangential slip at the core boundary that the
// welded space suppresses (slip; needed where the slip cannot be removed
// by relabelling, e.g. on aspherical fluid regions), mapped/aspherical
// generality (referential family).
//
// With -vis (the default) GLVis shows the mantle displacement of the first
// architecture and, beside it, the DIFFERENCE field of the architecture
// that agrees least (rigid modes projected out): where the formulations
// part company, typically at the core-mantle boundary.
//
// Besides the solver comparison, -model turns the example into the
// interactive physics lab of the 2-D/3-D fluid-core programme
// (benchmarks/disc, doc/planning/disc_gravity_reference.md): -model fc
// is the non-neutral core, where the welded, slipping and Dahlen
// architectures genuinely differ (the welded-vs-slip gap is the
// suppressed non-removable slip; the full-elastic answers carry the
// non-neutral resolution floor); -model aw is the closed-form
// Adams-Williamson twin, where every tangential slip is removable and
// the Dahlen reduction is exact, so ALL architectures must coincide at
// discretisation level. -l picks the load degree, -kappa-scale dials
// N^2 on fc, and a 3-D mesh makes the same comparisons on the ball.
//
// One source serves the serial and the parallel build, and the same
// source serves 2-D and 3-D (the mesh decides).
//
// Sample runs (with mpiexec -np N in front in a parallel build):
//    ./self_gravitating_solvers
//    ./self_gravitating_solvers -model aw            (all must agree)
//    ./self_gravitating_solvers -model fc -l 3
//    ./self_gravitating_solvers -model fc -kappa-scale 10
//    ./self_gravitating_solvers -m ../data/elastogravity_two_layer_3d.msh \
//                               -model fc -no-slip -o 1
// ============================================================================

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

#include "AdGIA.hpp"
#include "visualisation.hpp"

using namespace mfem;
using namespace AdGIA;

namespace {

#ifdef MFEM_USE_MPI
using MeshType = ParMesh;
using SubMeshType = ParSubMesh;
using SpaceType = ParFiniteElementSpace;
using FieldType = ParGridFunction;
bool Root() { return Mpi::Root(); }
#else
using MeshType = Mesh;
using SubMeshType = SubMesh;
using SpaceType = FiniteElementSpace;
using FieldType = GridFunction;
bool Root() { return true; }
#endif

using Clock = std::chrono::steady_clock;
double Seconds(Clock::time_point since) {
  return std::chrono::duration<double>(Clock::now() - since).count();
}

long long TrueSize(SpaceType& fes) {
#ifdef MFEM_USE_MPI
  return fes.GlobalTrueVSize();
#else
  return fes.GetTrueVSize();
#endif
}

// Models (unit radius, fluid core below r_cmb):
//   uniform  the non-dimensional solver-demo body of the tests: rho = 1
//            and kappa = 1 everywhere, mu = 0.5 in the mantle, weak
//            coupling G = 0.05 — the historical default of this example;
//   fc       the benchmark twins' fluid_core in the disc family's units
//            (densities in 5000 kg/m^3, G = 1): uniform core with
//            N^2 < 0, where welded, slipping and Dahlen genuinely
//            differ (benchmarks/disc, doc/planning/
//            disc_gravity_reference.md);
//   aw       the closed-form Adams-Williamson neutral twin (N^2 = 0,
//            dimension-aware kappa; every tangential slip is removable
//            and the Dahlen reduction is exact, so ALL architectures
//            must coincide at discretisation level).
constexpr int kDtNDegree = 12;
constexpr double kRc = 3483.0 / 6371.0;
double kEps = -1.0;   // < 0: the model default (1e-2 uniform; 1e-1 on
                      // fc/aw, whose non-neutral or strongly coupled
                      // fluid sectors need it — the slip classes
                      // cannot warn about a semi-convergent gauge)
double kTheta = -1.0;  // < 0: the model default (1e2 uniform, 1e3 fc/aw
                       // — the strong-coupling models scale the
                       // interface stiffnesses ~20x)
int kALIterations = 12;
int kGaugeRefinements = 3;
constexpr double kAwAlpha = 0.2;

struct Model {
  std::function<double(double)> rho, kappa, mu, drho;  // radial fields
  double G = 0.05;  // model-default coupling (overridable with -G)
};

Model MakeModel(const std::string& name, int dim, double G_opt,
                double kappa_scale) {
  Model m;
  if (name == "uniform") {
    m.G = G_opt > 0 ? G_opt : 0.05;
    m.rho = [](double) { return 1.0; };
    m.kappa = [](double) { return 1.0; };
    m.mu = [](double r) { return r < kRc ? 0.0 : 0.5; };
    m.drho = [](double) { return 0.0; };
    return m;
  }
  // the disc family's units: densities / 5000 kg/m^3, moduli from the
  // fluid_core velocities (benchmarks/disc/disc_models.py)
  const double rho_mantle = 4500.0 / 5000.0;
  const double rho_core_mean = 11000.0 / 5000.0;
  const double kap_mantle =
      rho_mantle * (1.0 - 4.0 / 3.0 * std::pow(6000.0 / 11000.0, 2)) * 10.0;
  const double mu_mantle = rho_mantle * std::pow(6000.0 / 11000.0, 2) * 10.0;
  m.G = G_opt > 0 ? G_opt : 1.0;
  m.mu = [mu_mantle](double r) { return r < kRc ? 0.0 : mu_mantle; };
  if (name == "fc") {
    const double kap_core =
        rho_core_mean * std::pow(9000.0 / 11000.0, 2) * 10.0 * kappa_scale;
    m.rho = [rho_mantle, rho_core_mean](double r) {
      return r < kRc ? rho_core_mean : rho_mantle;
    };
    m.kappa = [kap_mantle, kap_core](double r) {
      return r < kRc ? kap_core : kap_mantle;
    };
    m.drho = [](double) { return 0.0; };
    return m;
  }
  MFEM_VERIFY(name == "aw", "unknown model");
  MFEM_VERIFY(kappa_scale == 1.0,
              "-kappa-scale breaks the AW identity; use -model fc");
  // mass-matched central density and the closed-form neutral kappa,
  // dimension-aware (doc/planning/disc_gravity_reference.md)
  const double a = kAwAlpha;
  const double rho0 = dim == 2 ? rho_core_mean / (1.0 - a / 2.0)
                               : rho_core_mean / (1.0 - 3.0 * a / 5.0);
  const double pi = std::numbers::pi;
  const double G = m.G;
  const double kap0 = dim == 2
                          ? pi * G * rho0 * rho0 * kRc * kRc / a
                          : 2.0 * pi * G * rho0 * rho0 * kRc * kRc / (3.0 * a);
  m.rho = [rho_mantle, rho0, a](double r) {
    if (r >= kRc) {
      return rho_mantle;
    }
    const double x2 = (r / kRc) * (r / kRc);
    return rho0 * (1.0 - a * x2);
  };
  m.kappa = [kap_mantle, kap0, a, dim](double r) {
    if (r >= kRc) {
      return kap_mantle;
    }
    const double x2 = (r / kRc) * (r / kRc);
    const double tail = dim == 2 ? 1.0 - 0.5 * a * x2
                                 : 1.0 - 0.6 * a * x2;
    return kap0 * (1.0 - a * x2) * (1.0 - a * x2) * tail;
  };
  m.drho = [rho0, a](double r) {
    return r < kRc ? -2.0 * rho0 * a * r / (kRc * kRc) : 0.0;
  };
  return m;
}

// Pure degree-l surface mass load (no degree-0 part, where the Dahlen
// fluid treatment differs by design — doc/gauged_fluid.md, "Degree 0"):
// cos(l theta) in 2-D, P_l(cos theta) in 3-D.
int LoadDegree = 2;
double SurfaceLoad(const Vector& x) {
  const double r = x.Norml2();
  if (x.Size() == 2) {
    return 0.02 * std::cos(LoadDegree * std::atan2(x[1], x[0]));
  }
  const double c = x[2] / r;
  double pm1 = 1.0, pl = c;  // P_0, P_1
  for (int k = 2; k <= LoadDegree; k++) {
    const double pk = ((2 * k - 1) * c * pl - (k - 1) * pm1) / k;
    pm1 = pl;
    pl = pk;
  }
  return 0.02 * (LoadDegree == 0 ? 1.0 : pl);
}

// Marker for the boundary attributes whose centre lies at radius in
// (r_min, r_max).
Array<int> RadialBdrMarker(Mesh& mesh, double r_min, double r_max) {
  Array<int> marker(mesh.bdr_attributes.Max());
  marker = 0;
  for (int i = 0; i < mesh.GetNBE(); i++) {
    auto* tr = mesh.GetBdrElementTransformation(i);
    Vector c(mesh.Dimension());
    tr->Transform(Geometries.GetCenter(mesh.GetBdrElementGeometry(i)), c);
    const double r = c.Norml2();
    if (r > r_min && r < r_max) {
      marker[mesh.GetBdrAttribute(i) - 1] = 1;
    }
  }
  return marker;
}

double L2Norm(const GridFunction& u) {
  Vector zero(u.FESpace()->GetVDim());
  zero = 0.0;
  VectorConstantCoefficient z(zero);
  return const_cast<GridFunction&>(u).ComputeL2Error(z);
}

// One run of one architecture.
struct Entry {
  std::string name;
  long long unknowns = 0;
  int iterations = 0;
  double setup_seconds = 0.0, solve_seconds = 0.0;
  double difference = NAN;  // vs the first entry, mantle, modulo rigid
};

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif

  const char* mesh_file = "../data/elastogravity_two_layer_2d.msh";
  const char* model_name = "uniform";
  int order = 2;
  bool with_slip = true;
  double rel_tol = 1e-10;
  double G_opt = -1.0;
  double kappa_scale = 1.0;
  bool visualization = true;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
                 "Two-layer mesh, 2-D or 3-D (fluid core 1, mantle 2, "
                 "buffer 3).");
  args.AddOption(&model_name, "-model", "--model",
                 "uniform (the solver demo), fc (non-neutral core) or aw "
                 "(the Adams-Williamson neutral twin).");
  args.AddOption(&LoadDegree, "-l", "--load-degree",
                 "Degree of the surface mass load (>= 1).");
  args.AddOption(&G_opt, "-G", "--gravitational-constant",
                 "Gravitational constant (< 0: the model's default).");
  args.AddOption(&kappa_scale, "-kappa-scale", "--kappa-scale",
                 "Scale the fc core bulk modulus (N^2 ~ 1/scale).");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&rel_tol, "-rt", "--rel-tol", "Relative solver tolerance.");
  args.AddOption(&kEps, "-geps", "--gauge-epsilon",
                 "Fluid gauge penalty factor.");
  args.AddOption(&kGaugeRefinements, "-gref", "--gauge-refinements",
                 "Tikhonov gauge refinements per solve.");
  args.AddOption(&kTheta, "-theta", "--theta",
                 "Slip normal-jump penalty.");
  args.AddOption(&kALIterations, "-nal", "--al-iterations",
                 "Augmented-Lagrangian iterations (slip).");
  args.AddOption(&with_slip, "-slip", "--slip", "-no-slip", "--no-slip",
                 "Run the slipping-interface architectures as well.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Show the solution and the largest disagreement in GLVis.");
  args.Parse();
  if (!args.Good()) {
    if (Root()) {
      args.PrintUsage(std::cout);
    }
    return 1;
  }
  if (Root()) {
    args.PrintOptions(std::cout);
  }

  Mesh smesh(mesh_file, 1, 1);
  const int dim = smesh.Dimension();
#ifdef MFEM_USE_MPI
  MeshType parent(MPI_COMM_WORLD, smesh);
#else
  MeshType& parent = smesh;
#endif
  Array<int> fluid_attr({1}), solid_attr({2}), buffer_attr({3}),
      body_attr({1, 2}), outer_attr({2, 3});
  auto solid = SubMeshType::CreateFromDomain(parent, solid_attr);
  auto fluid = SubMeshType::CreateFromDomain(parent, fluid_attr);
  auto buffer = SubMeshType::CreateFromDomain(parent, buffer_attr);
  auto body = SubMeshType::CreateFromDomain(parent, body_attr);
  auto outer = SubMeshType::CreateFromDomain(parent, outer_attr);
  H1_FECollection fec(order, dim);
  SpaceType fes_s(&solid, &fec, dim), fes_f(&fluid, &fec, dim);
  SpaceType fes_b(&buffer, &fec, dim), fes_body(&body, &fec, dim);
  SpaceType fes_zeta(&parent, &fec), fes_phi(&parent, &fec);
  Vector bb_min, bb_max;
  parent.GetBoundingBox(bb_min, bb_max);
  const double r_out = bb_max.Normlinf();

  // The shared background: hydrostatic two-layer disc/ball of the
  // chosen model.
  MFEM_VERIFY(LoadDegree >= 1, "degree-0 loads are excluded by design");
  const Model model = MakeModel(model_name, dim, G_opt, kappa_scale);
  const double kG = model.G;
  if (kTheta < 0) {
    kTheta = std::string(model_name) == "uniform" ? 1.0e2 : 1.0e3;
  }
  if (kEps < 0) {
    // fc's non-neutral fluid sector needs the larger penalty; on aw and
    // uniform the welded classes do better in the small-eps window
    kEps = std::string(model_name) == "fc" ? 1.0e-1 : 1.0e-2;
  }
  RadialHydrostaticBackground bg(dim, model.rho, model.kappa, model.mu,
                                 kG, 1.0);
  FunctionCoefficient sigma(SurfaceLoad);
  FunctionCoefficient rho_c(
      [&model](const Vector& x) { return model.rho(x.Norml2()); });
  // the Dahlen interface terms take the FLUID-side density: interface
  // quadrature points sit at r = r_cmb to round-off, where the radial
  // branch above would hand them the mantle value (invisible on the
  // uniform model, a ~2x error on fc/aw)
  FunctionCoefficient rho_fluid_side([&model](const Vector& x) {
    const double r = x.Norml2();
    return model.rho(r <= kRc * (1.0 + 1e-9) ? std::min(r, kRc * (1.0 - 1e-12))
                                             : r);
  });
  FunctionCoefficient kappa_c(
      [&model](const Vector& x) { return model.kappa(x.Norml2()); });
  FunctionCoefficient mu_c(
      [&model](const Vector& x) { return model.mu(x.Norml2()); });
  // rho'_F = d rho / d Phi_0 = rho'(r) / g(r) for the Dahlen F1 term
  // (finite at the centre: both vanish linearly)
  FunctionCoefficient drho_dPhi([&model, &bg](const Vector& x) {
    const double r = std::max(x.Norml2(), 1e-8);
    const double g = bg.State().Gravity(r);
    return g > 0 ? model.drho(r) / g : 0.0;
  });
  FunctionCoefficient mu_gauge(
      [&model](const Vector& x) { return model.kappa(x.Norml2()); });
  auto interface_s = RadialBdrMarker(solid, 0.9 * kRc, 1.1 * kRc);
  auto surface_s = RadialBdrMarker(solid, 0.9, 1.1);
  auto surface_body = RadialBdrMarker(body, 0.9, 1.1);

  std::vector<Entry> table;
  auto rigid_proj = MakeRigidModeProjector(fes_s);
  // FieldType, not GridFunction: a plain GridFunction copy of a parallel
  // field computes rank-local norms.
  std::unique_ptr<FieldType> reference;  // mantle displacement of run 1
  std::unique_ptr<FieldType> worst;      // largest difference from it
  std::string worst_name;
  double worst_difference = 0.0;

  // The mantle part of a solution, on the solid space.
  auto on_mantle = [&](const GridFunction& u) {
#ifdef MFEM_USE_MPI
    auto out = std::make_unique<ParGridFunction>(&fes_s);
    if (u.FESpace() == &fes_s) {
      *out = u;
    } else {
      SpaceType parent_v(&parent, &fec, dim);
      ParGridFunction on_parent(&parent_v);
      on_parent = 0.0;
      ParSubMesh::Transfer(static_cast<const ParGridFunction&>(u),
                           on_parent);
      ParSubMesh::Transfer(on_parent, *out);
    }
    return out;
#else
    auto out = std::make_unique<GridFunction>(&fes_s);
    if (u.FESpace() == &fes_s) {
      *out = u;
    } else {
      FiniteElementSpace parent_v(&parent, &fec, dim);
      GridFunction on_parent(&parent_v);
      on_parent = 0.0;
      SubMesh::Transfer(u, on_parent);
      SubMesh::Transfer(on_parent, *out);
    }
    return out;
#endif
  };

  auto record = [&](const std::string& name, long long unknowns,
                    int iterations, double setup, double solve,
                    const GridFunction& u) {
    Entry e{name, unknowns, iterations, setup, solve, NAN};
    auto mantle = on_mantle(u);
    if (!reference) {
      reference = std::move(mantle);
      e.difference = 0.0;
    } else {
      // The projector acts on true-dof vectors (in parallel the local
      // vector is longer: it repeats the shared dofs).
      auto remove_rigid = [&](FieldType& f) {
        Vector t;
        f.GetTrueDofs(t);
        rigid_proj->Project(t);
        f.SetFromTrueDofs(t);
      };
      FieldType d(*mantle);
      d -= *reference;
      remove_rigid(d);
      FieldType ref(*reference);
      remove_rigid(ref);
      e.difference = L2Norm(d) / L2Norm(ref);
      if (!worst || e.difference > worst_difference) {
        worst = std::make_unique<FieldType>(d);
        worst_name = name;
        worst_difference = e.difference;
      }
    }
    table.push_back(e);
    if (Root()) {
      std::cout << "  " << name << ": done (" << e.solve_seconds
                << " s solve)\n";
    }
  };

  // --- 1 & 2: the Eulerian formulation, fluid eliminated (Dahlen), with
  // its two solvers.
  {
    IsotropicElasticRheology rheology(dim, kappa_c, mu_c);
    FluidRegion core;
    core.attributes = fluid_attr;
    core.density = &rho_fluid_side;
    core.density_gradient = &drho_dPhi;
    core.interface_marker = interface_s;
    std::vector<FluidRegion> fluids{core};
    for (const bool schur : {true, false}) {
      auto t0 = Clock::now();
      LinearQuasiStaticMixedSelfGravitatingProblem dahlen(
          &fes_s, &fes_phi, rheology, rho_c, kG, kDtNDegree, nullptr,
          fluids);
      dahlen.SetSurfaceLoad(sigma, surface_s);
      dahlen.SetSolverType(
          schur ? LinearQuasiStaticMixedSelfGravitatingProblem::SolverType::SchurCG
                : LinearQuasiStaticMixedSelfGravitatingProblem::SolverType::
                      BlockMINRES);
      dahlen.SetRelTol(rel_tol);
      dahlen.AssembleForce(0.0);
      const double setup = Seconds(t0);
      t0 = Clock::now();
      if (!dahlen.Solve() && Root()) {
        std::cout << "  (not converged)\n";
      }
      record(schur ? "dahlen / Schur CG" : "dahlen / block MINRES",
             TrueSize(fes_s) + TrueSize(fes_phi),
             dahlen.LastOuterIterations(), setup, Seconds(t0),
             dahlen.Displacement());
    }
  }

  // --- 3: the Eulerian gauged fluid.
  {
    auto t0 = Clock::now();
    IsotropicElasticRheology rheology(dim, kappa_c, mu_c);
    LinearQuasiStaticMixedSelfGravitatingProblem gauged(
        &fes_body, &fes_phi, rheology, rho_c, kG, kDtNDegree);
    Array<int> gauge_marker(body.attributes.Max());
    gauge_marker = 0;
    gauge_marker[0] = 1;
    gauged.SetGaugedFluid(gauge_marker, mu_gauge, kEps,
                          kGaugeRefinements);
    gauged.SetSurfaceLoad(sigma, surface_body);
    gauged.SetRelTol(rel_tol);
    gauged.AssembleForce(0.0);
    const double setup = Seconds(t0);
    t0 = Clock::now();
    gauged.Solve();
    record("gauged / block MINRES",
           TrueSize(fes_body) + TrueSize(fes_phi),
           gauged.LastOuterIterations(), setup, Seconds(t0),
           gauged.Displacement());
  }

  // --- 4: the welded gauged referential formulation.
  {
    auto t0 = Clock::now();
    LinearQuasiStaticReferentialSelfGravitatingProblem referential(
        &fes_body, &fes_zeta, bg.Rheology(), bg.Density(), kG, kDtNDegree);
    auto Evac = NewRadialVacuumExtension(fes_body, fes_b, 1.0, r_out);
    referential.SetPrescribedVacuumExtension(fes_b, *Evac);
    Array<int> fluid_marker(body.attributes.Max());
    fluid_marker = 0;
    fluid_marker[0] = 1;
    referential.SetGaugedFluid(fluid_marker, mu_gauge, kEps,
                               kGaugeRefinements);
    referential.SetSurfaceLoad(sigma, surface_body);
    referential.SetRelTol(rel_tol);
    referential.AssembleForce(0.0);
    const double setup = Seconds(t0);
    t0 = Clock::now();
    referential.Solve();
    record("referential / projected MINRES",
           TrueSize(fes_body) + TrueSize(fes_zeta),
           referential.LastOuterIterations(), setup, Seconds(t0),
           referential.Displacement());
  }

  // --- 5: the Maxwell (secular) relaxation of the referential problem.
  {
    auto t0 = Clock::now();
    LinearQuasiStaticReferentialSelfGravitatingProblem maxwell(
        &fes_body, &fes_zeta, bg.Rheology(), bg.Density(), kG, kDtNDegree);
    auto Evac = NewRadialVacuumExtension(fes_body, fes_b, 1.0, r_out);
    maxwell.SetPrescribedVacuumExtension(fes_b, *Evac);
    Array<int> fluid_marker(body.attributes.Max());
    fluid_marker = 0;
    fluid_marker[0] = 1;
    // The artificial core shear: the mantle's value (the relaxed state
    // is exactly independent of it; it only sets the clock).
    ConstantCoefficient mu_core_art(model.mu(1.0));
    maxwell.SetMaxwellFluid(fluid_marker, mu_core_art);
    maxwell.SetSurfaceLoad(sigma, surface_body);
    maxwell.SetRelTol(rel_tol);
    maxwell.AssembleForce(0.0);
    const double setup = Seconds(t0);
    t0 = Clock::now();
    maxwell.Solve();
    const auto& rep = maxwell.MaxwellReport();
    if (Root()) {
      std::cout << "  maxwell relaxation: " << rep.steps << " steps, "
                << rep.operators << " operators, stop = " << rep.stop
                << ", solid increment " << std::scientific
                << std::setprecision(1) << rep.delta_returned
                << std::defaultfloat << "\n";
    }
    record("maxwell relaxed / projected MINRES",
           TrueSize(fes_body) + TrueSize(fes_zeta),
           static_cast<int>(maxwell.TotalIterations()), setup, Seconds(t0),
           maxwell.Displacement());
  }

  // --- 6 & 7: the slipping interface, single-valued and broken zeta.
  if (with_slip) {
    for (const bool broken : {false, true}) {
      auto t0 = Clock::now();
      LinearQuasiStaticReferentialSelfGravitatingSlipProblem slip(
          &fes_s, &fes_f, &fes_zeta, bg.Rheology(), bg.Density(),
          bg.Pressure(), interface_s, kG, kDtNDegree);
      auto Evac = NewRadialVacuumExtension(fes_s, fes_b, 1.0, r_out);
      slip.SetPrescribedVacuumExtension(fes_b, *Evac);
      slip.SetFluidGauge(mu_gauge, kEps);
      slip.SetConstraint(kTheta, kALIterations);
      std::unique_ptr<SpaceType> fes_zo;
      if (broken) {
        fes_zo = SubMeshDofInjection::MakeShadowSpace(fes_zeta, outer);
        slip.EnableBrokenZeta(fes_zo.get(), kTheta);
      } else {
        auto Ef = NewRadialFluidExtension(fes_s, fes_f, kRc);
        slip.SetFluidExtension(*Ef);
      }
      slip.SetSurfaceLoad(sigma, surface_s);
      slip.SetRelTol(rel_tol);
      slip.AssembleForce(0.0);
      const double setup = Seconds(t0);
      t0 = Clock::now();
      slip.Solve();
      // Single-valued: u_s, u_f and zeta on the ball. Broken: u_s, u_f,
      // zeta on the solid shell and buffer, zeta on the core (the four
      // spaces of BrokenSpace()).
      long long unknowns = 0;
      if (broken) {
        for (int i = 0; i < 4; i++) {
          unknowns += TrueSize(static_cast<SpaceType&>(slip.BrokenSpace(i)));
        }
      } else {
        unknowns = TrueSize(fes_s) + TrueSize(fes_f) + TrueSize(fes_zeta);
      }
      record(broken ? "slip broken-zeta / AL + proj. MINRES"
                    : "slip / AL + projected MINRES",
             unknowns,
             slip.LastOuterIterations(), setup, Seconds(t0),
             slip.Displacement());
    }
  }

  if (Root()) {
    std::cout << "\n  architecture                          unknowns   its"
              << "   setup s   solve s   vs dahlen/Schur\n";
    for (const auto& e : table) {
      std::cout << "  " << std::left << std::setw(38) << e.name
                << std::right << std::setw(8) << e.unknowns << std::setw(6)
                << e.iterations << std::setw(10) << std::fixed
                << std::setprecision(2) << e.setup_seconds << std::setw(10)
                << e.solve_seconds << std::setw(14) << std::scientific
                << std::setprecision(2) << e.difference << "\n"
                << std::defaultfloat;
    }
    std::cout << "\n";
    if (std::string(model_name) == "aw") {
      std::cout << "Model aw (N^2 = 0): every tangential slip is "
                   "removable and the Dahlen reduction is exact, so all "
                   "architectures must agree at discretisation level — "
                   "any spread beyond it is a bug.\n";
    } else if (std::string(model_name) == "fc") {
      std::cout << "Model fc (N^2 < 0): the spreads are physics — the "
                   "welded architectures suppress a non-removable slip, "
                   "Dahlen commits to the secular interface closure, the "
                   "Maxwell relaxation reaches the welded limit (or its "
                   "band-level plateau) through time alone, and "
                   "the full-elastic answers carry the non-neutral "
                   "resolution floor (rerun at another order and watch "
                   "them move while the physics stays inside the band; "
                   "doc/planning/disc_gravity_reference.md).\n";
    } else {
      std::cout << "The architectures agree on the mantle displacement "
                   "(modulo rigid modes) at the level of the "
                   "discretisations; the costs differ by their "
                   "structure: nested solves (Schur), iteration counts "
                   "(MINRES), gauge refinements, and full solves per AL "
                   "iteration (slip).\n";
    }
  }
  if (visualization && reference) {
    examples::GLVisWindow("mantle displacement: " + table[0].name,
                          examples::DefaultKeys(dim))
        .Send(solid, *reference);
    if (worst) {
      examples::GLVisWindow("difference, " + worst_name + " - " +
                                table[0].name + " (rigid modes removed)",
                            examples::DefaultKeys(dim))
          .Send(solid, *worst);
    }
  }
  return 0;
}
