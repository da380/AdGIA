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
//   5. slip                     the slipping fluid-solid interface
//                               (doc/slip_interface.tex): broken
//                               displacement pair, single-valued zeta,
//                               the normal-jump constraint by penalty +
//                               augmented Lagrangian — every AL iteration
//                               is a full projected-MINRES solve of the
//                               three-block system;
//   6. slip broken-zeta         the same with the potential broken too
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
// All six must agree on the observables (the mantle displacement, modulo
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
// One source serves the serial and the parallel build.
//
// Sample runs (with mpiexec -np N in front in a parallel build):
//    ./self_gravitating_solvers
//    ./self_gravitating_solvers -o 3
//    ./self_gravitating_solvers -no-slip        (the Eulerian trio alone)
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

// Non-dimensional two-layer model of the tests: unit radius, strong
// coupling, fluid core below r_cmb.
constexpr double kG = 0.05;
constexpr double kRho = 1.0;
constexpr double kKappa = 1.0;
constexpr double kMu = 0.5;
constexpr int kDtNDegree = 12;
constexpr double kRc = 3483.0 / 6371.0;
constexpr double kEps = 1.0e-2;
constexpr double kTheta = 1.0e2;
constexpr int kALIterations = 8;

// PURE degree-2 surface mass load: no degree-0 part, where the Dahlen
// fluid treatment differs from the compressible descriptions by design
// (doc/gauged_fluid.md, "Degree 0") and the architectures would rightly
// disagree.
double SurfaceLoad(const Vector& x) {
  const double r = x.Norml2();
  const double c = x[x.Size() - 1] / r;
  return 0.02 * (2.0 * c * c - 1.0);
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
  int order = 2;
  bool with_slip = true;
  double rel_tol = 1e-10;
  bool visualization = true;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
                 "Two-layer mesh (fluid core 1, mantle 2, buffer 3).");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&rel_tol, "-rt", "--rel-tol", "Relative solver tolerance.");
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

  // The shared background: hydrostatic two-layer disc/ball.
  RadialHydrostaticBackground bg(
      dim, [](double) { return kRho; }, [](double) { return kKappa; },
      [](double r) { return r < kRc ? 0.0 : kMu; }, kG, 1.0);
  FunctionCoefficient sigma(SurfaceLoad);
  ConstantCoefficient rho_c(kRho), kappa_c(kKappa), mu_c(kMu), zero(0.0);
  ConstantCoefficient mu_gauge(kKappa);
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
    core.density = &rho_c;
    core.density_gradient = &zero;
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
    FunctionCoefficient mu_layered(
        [](const Vector& x) { return x.Norml2() < kRc ? 0.0 : kMu; });
    IsotropicElasticRheology rheology(dim, kappa_c, mu_layered);
    LinearQuasiStaticMixedSelfGravitatingProblem gauged(
        &fes_body, &fes_phi, rheology, rho_c, kG, kDtNDegree);
    Array<int> gauge_marker(body.attributes.Max());
    gauge_marker = 0;
    gauge_marker[0] = 1;
    gauged.SetGaugedFluid(gauge_marker, mu_gauge, kEps, 3);
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
    referential.SetGaugedFluid(fluid_marker, mu_gauge, kEps, 3);
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

  // --- 5 & 6: the slipping interface, single-valued and broken zeta.
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
    std::cout << "\nThe architectures agree on the mantle displacement "
                 "(modulo rigid modes) at the level of the "
                 "discretisations; the costs differ by their structure: "
                 "nested solves (Schur), iteration counts (MINRES), gauge "
                 "refinements, and full solves per AL iteration (slip).\n";
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
