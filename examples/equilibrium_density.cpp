// ============================================================================
// equilibrium_density.cpp
//
// Density restoration — the first milestone of the equilibrium-figures
// programme (doc/equilibrium_figures.tex §8). A layered model starts
// from a fluid density with no static state, and the fluid-only
// feasibility functional
//
//   J = min { 1/2 int_fluid |dev T|^2 : Div T = rho grad Phi in the
//             fluid, interface traction free }
//
// is positive, with the minimising multiplier u the creeping flow the
// unbalanced buoyancy would drive.
//
// The model. By default the GEOMETRY denies the equilibrium: the
// starting density is a simple radial profile, but the container is
// aspherical — -shape flat (the default) flattens every interface
// (ellipse in 2-D, oblate ellipsoid in 3-D), -shape cmb puts
// oscillatory topography on the CMB alone, -shape bump a single
// Gaussian topographic bump there — so rho(r) is not constant
// on the equipotentials, and the restoration has to GENERATE the
// non-spherical equilibrium density of the container (the before/after
// density windows show it). -shape sphere keeps the spherical meshes,
// where the disequilibrium is instead the hand-inserted lateral
// density term -amp (its default: 0.1 on the sphere, 0 — pure radial —
// on the aspherical shapes; on a sphere -amp 0 starts at the mesh's
// own discretisation imbalance). -blob adds a fixed Gaussian density
// anomaly in the mantle: not part of the control, but its gravity
// stresses the mantle and shifts the equipotentials, so the core's
// equilibrium density moves in response. -dim picks the dimension (2,
// the default, for the disc section; 3 for the ball — an order of
// magnitude dearer) and -ic / -no-ic the layering (the three-layer
// Earth with a solid inner core, the default, or the two-layer one
// whose fluid core reaches the centre); together with -shape they name
// one of the sixteen
// {flattened,cmb_topo,cmb_bump}_{two,three}_layer_{2,3}d.mesh /
// elastogravity_{two,three}_layer_{2,3}d.msh meshes, and -m overrides
// with any layered mesh of the same conventions (meshes/README.md:
// the layering is recognised from the attribute count;
// equilibrium_bodies.py makes the aspherical ones and takes
// --flattening, --amplitude, --degree, --width and --scale for
// variants). With an inner core, the
// enclosed solid's force and torque balance joins the constraints
// through the saddle's rigid border by default (-core rigid), and the
// run reports the core's multiplier motion (its force/torque
// signature, decaying as balance is restored; the core_motion column
// of the CSV) alongside the connected-solid (clamped) functional,
// whose gap to J is exactly the core imbalance the clamped certificate
// cannot see; -core clamped shows that optimistic functional driving
// the loop instead. In two dimensions the exterior potential's
// logarithmic monopole makes the Laplace-DtN operator singular in the
// constant; the library makes the loads compatible, projects the
// solves and gauges the potentials to zero boundary mean, in which
// gauge the envelope formulas are discretely exact as written
// (equilibrium_figures.hpp).
//
// Discretisation: -o sets the velocity and control order (pressure one
// below, Taylor-Hood), -deg the truncation degree of the DtN expansion
// on the outer boundary. The default order is 3 on a disc and 2 on a
// ball (cost): MEASURED, the aspherical shapes' geometric
// disequilibrium is weak — eta scales like 8e-3 x flattening, the
// degree-2 potential perturbation decaying into the fluid — and sits
// BELOW the order-2 floor (2e-3 in 2-D), while at order 3 (floor
// 2e-4) the flattened disc starts at eta = 8e-4 and J falls 18x in 16
// iterations. A 3-D aspherical run at its default order 2 is
// floor-bound: give it -o 3 and patience, or -shape sphere. The cmb
// signal is an order of magnitude weaker still (the CMB density jump
// is 0.1 against the surface's 1.0) and at order 3 is limited by the
// order-2 GEOMETRY of the wavy interface — the qualitative variant.
//
// The example drives J to its floor by moving the fluid density, three
// ways (the default is the study's measured recipe, doc/planning/
// equilibrium_figures.md: Levenberg-Marquardt Gauss-Newton with a
// roughness prior, stopped at the discretisation floor):
//
//   -loop gn      Levenberg-Marquardt Gauss-Newton (L2 control): the
//                 inner PCG solves (H_GN + lambda M) s = -j with
//                 Hessian-vector products from the envelope machinery
//                 (one saddle and two Poisson solves each, on the
//                 persistent operators); the LM damping is iterated
//                 Tikhonov — the semi-convergence protection — and the
//                 roughness prior (-prior) selects the smooth member
//                 of J's near-null family, so the loop converges onto
//                 the regularised solution instead of fitting
//                 discretisation error. The prior defaults to 1e-7,
//                 EXCEPT for -shape cmb, where it defaults to 0: the
//                 wavy CMB demands an oscillatory, boundary-following
//                 correction that the H1 prior blocks (measured: 189
//                 iterations stuck at the start with it, 29 to the
//                 floor without; the damping alone protects there).
//
//   -loop cg      nonlinear conjugate gradients (Polak-Ribiere+, a
//                 forward-tracking line search) on the density, the
//                 derivative by the envelope theorem
//                 (DensityFeasibility) and the gradient through a
//                 choosable Riesz map:
//                   -metric l2   the L2 identification on the fluid
//                   -metric h1   the Sobolev metric on the whole mesh
//                   -metric h2   its iterated (order-2) form, the
//                                working default of the programme
//                 (-length sets the smoothing length sqrt(beta/alpha)).
//                 The fluid mass is held fixed by projecting the search
//                 direction in the metric. An order of magnitude more
//                 iterations than gn for the same floor.
//
//   -loop advect  the advection flow: rho stepped along u itself — the
//                 gradient flow of the gravitational energy in the
//                 dissipation metric, mass- and (up to projection)
//                 distribution-preserving, with J its decay rate.
//                 Steps are accepted on the energy (J need not fall
//                 monotonically along the flow), under a CFL-type cap
//                 and a ceiling on J: the projected Eulerian step
//                 leaks the rearrangement constraint through its
//                 discretisation error, and off that manifold the
//                 energy is unbounded below (discrete gravitational
//                 collapse — in 2-D this channel dominates and the
//                 flow runs straight into the ceiling). Exploratory;
//                 a range-preserving (semi-Lagrangian) step is the
//                 known structural fix (the planning document).
//
// Stopping: J's absolute size is dominated by the base model's own
// discretisation imbalance, so a relative-J tolerance is the wrong
// rule (it is either unreachable or fits discretisation error). The
// run watches the dimensionless infeasibility eta = |dev T| / |p|
// (the dev_over_p column of the CSV) and stops, by default, when eta
// stagnates — the discretisation floor, wherever this mesh puts it —
// or at -iters. -eta sets an explicit threshold instead (the
// discrepancy rule of the study), and -tol restores the relative-J
// stop for J-decay experiments.
//
// At J's floor the fluid is barotropic: rho constant on equipotentials.
// The (Phi, rho) scatter over the fluid collapses onto a single curve —
// the before/after scatter is the example's physical verdict.
//
// Outputs: the -csv table (default equilibrium_density.csv; "" writes
// none) with J, the step or damping, the mass drift, eta, the energy
// and the core motion by iteration, and a *_barotropy.csv named after
// it with the scatter, both for plot_csv.py. With -vis (start `glvis`
// first), five windows: the whole-body density and |dev T|/p_rms maps
// before and after (|dev T| against the RMS of the PHYSICAL fluid
// pressure, anchored by the recovered datum — a single scalar, since
// any pointwise pressure division is singular where the pressure
// crosses zero: the gauged field inside the fluid, the physical one at
// the free surface — so a map value reads as "deviatoric stress as
// this fraction of the actual pressure", O(flattening) for the
// mantle), and the initial relaxation flow |u| on the fluid. The
// INITIAL body stress window shows the
// unweighted GLOBAL minimiser — the one equilibrium stress field the
// whole body admits away from feasibility, whose fluid share includes
// the solid's leakage (the leakage proposition; the printed
// fluid/solid split quantifies it). The FINAL body window shows the
// exact two-piece recovery — the certificate's own stress in the
// fluid, and per solid component (mantle, and the inner core when
// there is one) its minimum-deviatoric generator under the state's
// potential, loaded on its fluid interface by the certificate's
// pressure (T n = -p n) and traction-free at the surface — a genuine
// equilibrium field to numerical convergence, since the neglected
// viscous interface traction is O(sqrt J). There the fluid is
// hydrostatic by construction while the aspherical solid keeps its
// unavoidable share (Love's obstruction); the printed ||dev T|| split
// (fluid = sqrt(2J), exactly) says so in numbers, and a -blob shows
// as a stressed halo in the mantle. Both stress windows share the
// fluid p_rms scale and are elementwise — |dev T| genuinely jumps at
// the CMB and ICB, which a continuous field renders as an interface
// artefact (refine the mesh, equilibrium_bodies.py --scale, for a
// finer image).
//
// One source serves the serial and the parallel build, as in
// equilibrium_stress.cpp; the barotropy scatter is gathered to the root.
//
// Sample runs (with mpiexec -np N in front in a parallel build):
//    ./equilibrium_density -vis                the flattened disc: the
//                                              geometry-driven restoration
//    ./equilibrium_density -shape cmb          wavy CMB, spherical surface
//    ./equilibrium_density -shape bump -vis    one Gaussian CMB bump (the
//                                              strongest geometric signal;
//                                              floor-bound without -ic)
//    ./equilibrium_density -no-ic              no inner core (two layers)
//    ./equilibrium_density -blob 0.5 -vis      a mantle anomaly stressing
//                                              the solid, moving the core
//    ./equilibrium_density -core clamped       the optimistic certificate
//    ./equilibrium_density -amp 0.1            flattening AND the lateral
//                                              density term together
//    ./equilibrium_density -dim 3 -shape sphere
//                                              the 3-D ball (aspherical
//                                              shapes in 3-D need -o 3)
//    ./equilibrium_density -shape sphere -amp 0.2 -eta 5e-3
//                                              spherical container, larger
//                                              lateral start, explicit stop
//    ./equilibrium_density -shape sphere -amp 0
//                                              the symmetric model's floor
//    ./equilibrium_density -loop cg -metric h2 -length 0.2 -iters 100
//    ./equilibrium_density -loop cg -metric l2 -prior 0 -tol 0.1
//                                              unregularised J-decay study
//    ./equilibrium_density -shape sphere -loop advect -iters 200
//    ./equilibrium_density -o 2                the order-2 floor hides the
//                                              geometric signal (measured)
//    ./equilibrium_density -m ../data/flattened_three_layer_2d_f20.mesh
//                                              a hand-built variant
//                                              (equilibrium_bodies.py)
// ============================================================================

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
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
using GridType = ParGridFunction;
using FormType = ParBilinearForm;
bool Root() { return Mpi::Root(); }
#else
using MeshType = Mesh;
using SubMeshType = SubMesh;
using SpaceType = FiniteElementSpace;
using GridType = GridFunction;
using FormType = BilinearForm;
bool Root() { return true; }
#endif

double GlobalSum(double v) {
#ifdef MFEM_USE_MPI
  double g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return g;
#else
  return v;
#endif
}

// The layered meshes' conventions (meshes/README.md; also the test
// meshes). Three layers: attributes 1 inner core, 2 fluid outer core,
// 3 mantle, 4 buffer; boundary attributes 1 ICB, 2 CMB, 3 surface,
// 4 outer. Two layers (no inner core): attributes 1 fluid core,
// 2 mantle, 3 buffer; boundary attributes 1 CMB, 2 surface, 3 outer.
// The layering is recognised from the number of domain attributes.
struct Layering {
  bool has_core;
  int fluid, mantle, buffer;  // domain attributes
};

Layering RecogniseLayering(int max_attr) {
  MFEM_VERIFY(max_attr == 3 || max_attr == 4,
              "equilibrium_density: a two- or three-layer mesh with a "
              "buffer is expected (meshes/README.md).");
  const bool core = max_attr == 4;
  return {core, core ? 2 : 1, core ? 3 : 2, max_attr};
}

constexpr double kG = 0.05;

double lateral_amplitude = 0.1;
double blob_amplitude = 0.0;

// The fluid's starting density: a radial base — which on the
// aspherical shapes is already non-barotropic, the geometry's doing —
// plus an optional lateral part of amplitude -amp, which no
// rearrangement-free relabelling can remove (the spherical meshes'
// disequilibrium).
double FluidRho(const Vector& x) {
  const double r = x.Norml2();
  return 1.2 - 0.3 * r * r + lateral_amplitude * (r > 0.0 ? x[1] / r : 0.0);
}

// The mantle's density: 1, plus the -blob Gaussian anomaly (width 0.1
// at mid-mantle radius on the +x axis) — part of the FIXED solid
// density, not of the control: its gravity stresses the mantle and
// shifts the equipotentials, so the fluid core's equilibrium density
// moves in response.
double MantleRho(const Vector& x) {
  if (blob_amplitude == 0.0) {
    return 1.0;
  }
  constexpr double kBlobRadius = 0.77;  // mid-mantle (CMB 0.55, surface 1)
  constexpr double kBlobWidth = 0.1;
  Vector c(x.Size());
  c = 0.0;
  c[0] = kBlobRadius;
  Vector d(x);
  d -= c;
  return 1.0 + blob_amplitude * std::exp(-(d * d) / (kBlobWidth * kBlobWidth));
}

// Transfer between the parent and its SubMesh (either direction).
void Transfer(const GridFunction& src, GridFunction& dst) {
#ifdef MFEM_USE_MPI
  ParSubMesh::Transfer(static_cast<const ParGridFunction&>(src),
                       static_cast<ParGridFunction&>(dst));
#else
  SubMesh::Transfer(src, dst);
#endif
}

// The density of the whole model about the current fluid perturbation:
// the parent-side and the Stokes-side coefficients DensityFeasibility
// takes, kept in step by Sync (the fluid field is the primary).
struct Density {
  Density(const Layering& layers, SpaceType& fes_fluid,
          SpaceType& fes_fluid_on_parent)
      : fluid_base(FluidRho),
        mantle(MantleRho),
        inner(1.3),
        buffer(0.0),
        c_fluid(&fes_fluid),
        c_parent(&fes_fluid_on_parent),
        c_fluid_coeff(&c_fluid),
        c_parent_coeff(&c_parent),
        stokes(fluid_base, c_fluid_coeff),
        fluid_on_parent(fluid_base, c_parent_coeff) {
    c_fluid = 0.0;
    c_parent = 0.0;
    if (layers.has_core) {
      parent.UpdateCoefficient(1, inner);
    }
    parent.UpdateCoefficient(layers.fluid, fluid_on_parent);
    parent.UpdateCoefficient(layers.mantle, mantle);
    parent.UpdateCoefficient(layers.buffer, buffer);
  }

  // The fluid perturbation from its true dofs (on the fluid space).
  void Sync(const Vector& c) {
    c_fluid.SetFromTrueDofs(c);
    c_parent = 0.0;
    Transfer(c_fluid, c_parent);
  }

  FunctionCoefficient fluid_base, mantle;
  ConstantCoefficient inner, buffer;
  GridType c_fluid, c_parent;
  GridFunctionCoefficient c_fluid_coeff, c_parent_coeff;
  SumCoefficient stokes, fluid_on_parent;
  PWCoefficient parent;
};

struct Spaces {
  MeshType* parent;
  SubMeshType* fluid;
  Layering layers;
  SpaceType *phi, *u, *p, *ctrl_fluid, *ctrl_parent;
  Array<int>* ess;
  int dtn_degree;
  // The persistent half: operators and preconditioners assembled once,
  // solves per evaluation (the loops evaluate hundreds of times).
  DensityFeasibilityProblem* problem = nullptr;
};

// One evaluation of the fluid-only functional at the density's current
// state. The object is kept alive by the caller for its fields.
std::unique_ptr<DensityFeasibility> Evaluate(const Spaces& s, Density& rho) {
  return s.problem->Evaluate(rho.parent, rho.stokes);
}

// The control's plumbing: where it lives, its Riesz map, the fixed-mass
// constraint, the roughness prior and the mass matrix — the metric
// itself (gradients, projections, prior bookkeeping) is the library's
// ConstrainedMetric; the loops are the library's.
struct Control {
  SpaceType* fes;            // the control space (fluid or parent)
  bool on_parent;            // Sobolev metrics live on the parent
  std::unique_ptr<RieszMap> riesz;
  std::unique_ptr<FormType> prior_form, mass_form;
  OperatorHandle prior_K, mass_M;
  Vector mass_dual;          // l(v) = int_fluid v
  std::unique_ptr<ConstrainedMetric> metric;
};

Control MakeControl(const std::string& name, double length, double lambda,
                    const Spaces& s) {
  Control m;
  m.on_parent = name != "l2";
  m.fes = m.on_parent ? s.ctrl_parent : s.ctrl_fluid;
  Array<int> empty;
  if (m.on_parent) {
    const int order = name == "h2" ? 2 : 1;
    // Dirichlet on the outer boundary of the extension domain.
    Array<int> outer(s.parent->bdr_attributes.Max());
    outer = 0;
    outer[s.parent->bdr_attributes.Max() - 1] = 1;
    m.riesz = std::make_unique<SobolevRieszMap>(
        *m.fes, 1.0, length * length, order, &outer);
    // The map copies the marker's dofs at construction.
  } else {
    m.riesz = std::make_unique<L2RieszMap>(*m.fes);
  }
  m.metric = std::make_unique<ConstrainedMetric>(*m.riesz);

  // The roughness prior: lambda/2 c^T K c with K the H1-seminorm Gram
  // on the control space — a selection within the near-null directions
  // of J, biased toward smoothness rather than toward a known answer.
  if (lambda != 0.0) {
    m.prior_form = std::make_unique<FormType>(m.fes);
    m.prior_form->AddDomainIntegrator(new DiffusionIntegrator());
    m.prior_form->Assemble();
    m.prior_form->FormSystemMatrix(empty, m.prior_K);
    m.metric->SetPrior(*m.prior_K.Ptr(), lambda);
  }

  // The control's mass matrix: the Levenberg-Marquardt damping Gram.
  m.mass_form = std::make_unique<FormType>(m.fes);
  m.mass_form->AddDomainIntegrator(new MassIntegrator());
  m.mass_form->Assemble();
  m.mass_form->FormSystemMatrix(empty, m.mass_M);

  // The mass functional over the fluid, on the control space.
  ConstantCoefficient one(1.0);
  Array<int> fluid_marker;
  auto lf = [&]() {
#ifdef MFEM_USE_MPI
    auto plf = std::make_unique<ParLinearForm>(m.fes);
#else
    auto plf = std::make_unique<LinearForm>(m.fes);
#endif
    if (m.on_parent) {
      fluid_marker.SetSize(s.parent->attributes.Max());
      fluid_marker = 0;
      fluid_marker[s.layers.fluid - 1] = 1;
      plf->AddDomainIntegrator(new DomainLFIntegrator(one), fluid_marker);
    } else {
      plf->AddDomainIntegrator(new DomainLFIntegrator(one));
    }
    plf->Assemble();
    return plf;
  }();
  m.mass_dual.SetSize(m.fes->GetTrueVSize());
#ifdef MFEM_USE_MPI
  lf->ParallelAssemble(m.mass_dual);
#else
  m.mass_dual = *lf;
#endif
  m.metric->SetConstraint(m.mass_dual);
  return m;
}

// The control seen by the density: on the parent the model reads the
// restriction, which Sync takes from the fluid field, so the fluid
// twin must follow the parent control.
void SyncDensity(const Control& m, const Vector& c, Density& rho,
                 GridType& scratch_parent, GridType& scratch_fluid) {
  if (!m.on_parent) {
    rho.Sync(c);
    return;
  }
  scratch_parent.SetFromTrueDofs(c);
  scratch_fluid = 0.0;
  Transfer(scratch_parent, scratch_fluid);
  Vector cf(rho.c_fluid.FESpace()->GetTrueVSize());
  scratch_fluid.GetTrueDofs(cf);
  rho.Sync(cf);
}

// The (Phi, rho) scatter over the fluid's control dofs, gathered to the
// root: at the barotropic end state it collapses onto one curve.
void BarotropyRows(const Spaces& s, Density& rho,
                   DensityFeasibility& J, const std::string& label,
                   examples::CsvTable& table) {
  // Phi on the fluid control space (an H1 transfer of the potential;
  // the spaces share order and mesh family).
  GridType phi_fluid(s.ctrl_fluid);
  phi_fluid = 0.0;
  Transfer(J.Potential(), phi_fluid);
  GridType rho_fluid(s.ctrl_fluid);
  rho_fluid.ProjectCoefficient(rho.stokes);

  Vector phi_t(s.ctrl_fluid->GetTrueVSize()), rho_t(phi_t.Size());
  phi_fluid.GetTrueDofs(phi_t);
  rho_fluid.GetTrueDofs(rho_t);

  std::vector<double> local;
  local.reserve(2 * phi_t.Size());
  for (int i = 0; i < phi_t.Size(); i++) {
    local.push_back(phi_t[i]);
    local.push_back(rho_t[i]);
  }
#ifdef MFEM_USE_MPI
  int n = static_cast<int>(local.size());
  std::vector<int> counts(Mpi::WorldSize()), offsets(Mpi::WorldSize(), 0);
  MPI_Gather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  std::vector<double> all;
  if (Root()) {
    for (int r = 1; r < Mpi::WorldSize(); r++) {
      offsets[r] = offsets[r - 1] + counts[r - 1];
    }
    all.resize(offsets.back() + counts.back());
  }
  MPI_Gatherv(local.data(), n, MPI_DOUBLE, all.data(), counts.data(),
              offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (Root()) {
    local = std::move(all);
  }
#endif
  if (Root()) {
    for (std::size_t i = 0; i + 1 < local.size(); i += 2) {
      table.Row(label, {local[i], local[i + 1]});
    }
  }
}

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif

  const char* mesh_file = "";
  int mesh_dim = 2;
  bool inner_core = true;
  const char* shape = "flat";
  const char* loop = "gn";
  const char* core = "rigid";
  const char* metric_name = "h2";
  double length = 0.3;
  double prior = -1.0;  // resolved by -shape: 0 for cmb, else 1e-7
  int order = -1;       // resolved by the mesh: 3 on a disc, 2 on a ball
  int dtn_degree = 8;
  int iters = 40;
  double amplitude = -1.0;  // resolved by -shape: 0.1 on sphere, else 0
  double blob = 0.0;
  double tol = 0.0;   // the relative-J stop, off by default
  double eta = 0.0;   // 0: stop when eta stagnates; > 0: eta threshold
  bool visualisation = false;
  const char* csv_file = "equilibrium_density.csv";

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
                 "Mesh file (a layered model with a buffer; overrides "
                 "-dim and -ic/-no-ic).");
  args.AddOption(&mesh_dim, "-dim", "--dimension",
                 "Dimension of the default mesh: 2 (the default) for "
                 "the disc section, 3 for the ball — an order of "
                 "magnitude dearer, and at its default order 2 the "
                 "aspherical shapes' signal sits below the "
                 "discretisation floor (use -o 3, or -shape sphere).");
  args.AddOption(&inner_core, "-ic", "--inner-core", "-no-ic",
                 "--no-inner-core",
                 "Default mesh with a solid inner core (three layers) "
                 "or without one (two layers).");
  args.AddOption(&shape, "-shape", "--shape",
                 "Shape of the default mesh: flat (flattened "
                 "interfaces), cmb (oscillatory CMB topography), bump "
                 "(one Gaussian CMB bump) — the geometry denies a "
                 "radial density equilibrium — or sphere (equilibrium "
                 "needs the -amp lateral term).");
  args.AddOption(&loop, "-loop", "--loop", "gn, cg or advect.");
  args.AddOption(&metric_name, "-metric", "--metric",
                 "Gradient metric of the cg loop: l2, h1 or h2.");
  args.AddOption(&length, "-length", "--length",
                 "Smoothing length of the Sobolev metrics.");
  args.AddOption(&core, "-core", "--core",
                 "rigid (the exact certificate: the inner core's force "
                 "and torque balance joins the constraints) or clamped "
                 "(the connected-solid functional, optimistic here). "
                 "Moot without an inner core.");
  args.AddOption(&prior, "-prior", "--prior",
                 "Roughness (H1-seminorm) prior weight: selects the "
                 "smooth member of the near-null family (0: off; "
                 "negative: 1e-7, except 0 for -shape cmb, whose "
                 "oscillatory correction the prior blocks).");
  args.AddOption(&order, "-o", "--order",
                 "Velocity and control order (negative: 3 on a disc — "
                 "the aspherical signals need the lower floor — and 2 "
                 "on a ball, for cost).");
  args.AddOption(&dtn_degree, "-deg", "--degree", "DtN truncation degree.");
  args.AddOption(&iters, "-iters", "--iterations", "Iteration budget.");
  args.AddOption(&amplitude, "-amp", "--amplitude",
                 "Lateral (non-barotropic) amplitude of the starting "
                 "density (negative: 0.1 on the sphere, where it is "
                 "the only disequilibrium, 0 on the aspherical "
                 "shapes).");
  args.AddOption(&blob, "-blob", "--blob",
                 "Amplitude of a Gaussian density anomaly fixed in the "
                 "mantle (width 0.1 at mid-mantle radius on the +x "
                 "axis): stresses the mantle and shifts the core's "
                 "equilibrium (0: off).");
  args.AddOption(&eta, "-eta", "--eta",
                 "Stop when |dev T|/|p| falls below this (0: stop when "
                 "it stagnates — the discretisation floor).");
  args.AddOption(&tol, "-tol", "--tolerance",
                 "Stop when J falls below this fraction of its start "
                 "(0: off; the floor makes small values unreachable).");
  args.AddOption(&visualisation, "-vis", "--visualization", "-no-vis",
                 "--no-visualization", "GLVis visualisation.");
  args.AddOption(&csv_file, "-csv", "--csv",
                 "History table for plot_csv.py (\"\": none).");
  args.Parse();
  if (!args.Good()) {
    if (Root()) {
      args.PrintUsage(std::cout);
    }
    return 1;
  }
  const std::string shape_s = shape;
  MFEM_VERIFY(shape_s == "flat" || shape_s == "cmb" || shape_s == "bump" ||
                  shape_s == "sphere",
              "-shape must be flat, cmb, bump or sphere.");
  if (amplitude < 0.0) {
    amplitude = shape_s == "sphere" ? 0.1 : 0.0;
  }
  if (prior < 0.0) {
    // The measured calibration (doc/planning/equilibrium_figures.md):
    // the H1 prior selects the smooth member of the near-null family,
    // but the wavy CMB demands an OSCILLATORY correction, which the
    // prior blocks (189 iterations stuck at the start against 29 to
    // the floor without it); the LM damping alone is the
    // semi-convergence protection there.
    prior = shape_s == "cmb" ? 0.0 : 1e-7;
  }
  lateral_amplitude = amplitude;
  blob_amplitude = blob;
  std::string mesh_path = mesh_file;
  if (mesh_path.empty()) {
    MFEM_VERIFY(mesh_dim == 2 || mesh_dim == 3, "-dim must be 2 or 3.");
    const std::string layers_s = inner_core ? "three" : "two";
    const std::string dim_s = std::to_string(mesh_dim);
    if (shape_s == "sphere") {
      mesh_path = "../data/elastogravity_" + layers_s + "_layer_" + dim_s +
                  "d.msh";
    } else {
      const std::string stem = shape_s == "flat"  ? "flattened_"
                               : shape_s == "cmb" ? "cmb_topo_"
                                                  : "cmb_bump_";
      mesh_path =
          "../data/" + stem + layers_s + "_layer_" + dim_s + "d.mesh";
    }
  }

  // The meshes: the parent ball (or disc) with its buffer, and the
  // fluid SubMesh; the layering read off the attribute count.
#ifdef MFEM_USE_MPI
  Mesh serial(mesh_path.c_str(), 1, 1);
  MeshType parent(MPI_COMM_WORLD, serial);
  serial.Clear();
#else
  MeshType parent(mesh_path.c_str(), 1, 1);
#endif
  const int dim = parent.Dimension();
  const Layering layers = RecogniseLayering(parent.attributes.Max());
  if (order < 0) {
    // By the loaded mesh's dimension, not -dim: -m may override it.
    order = dim == 2 ? 3 : 2;
  }
  auto fluid =
      SubMeshType::CreateFromDomain(parent, Array<int>({layers.fluid}));

  // The body SubMesh — everything inside the surface — for the
  // whole-body windows (-vis): construction is cheap, the generator
  // solve on it runs only when asked.
  Array<int> body_attrs(layers.buffer - 1);
  for (int a = 1; a < layers.buffer; a++) {
    body_attrs[a - 1] = a;
  }
  auto body = SubMeshType::CreateFromDomain(parent, body_attrs);

  H1_FECollection h1(order, dim), h1_p(order - 1, dim);
  SpaceType fes_phi(&parent, &h1);
  SpaceType fes_u(&fluid, &h1, dim);
  SpaceType fes_p(&fluid, &h1_p);
  SpaceType ctrl_fluid(&fluid, &h1);
  SpaceType ctrl_parent(&parent, &h1);
  // An H1 twin on the body for the transferred fields (the fluid
  // perturbation and, for the initial window's global minimiser, the
  // potential), and the global generator's own Taylor-Hood pair.
  SpaceType fes_h1_body(&body, &h1);
  SpaceType fes_u_body(&body, &h1, dim);
  SpaceType fes_p_body(&body, &h1_p);

  Array<int> ess(fluid.bdr_attributes.Max());
  ess = 1;  // the certificate's no-slip condition on the whole boundary

  // The inner core as an enclosed rigid component (the default when the
  // mesh has one): its force and torque balance joins the certificate's
  // constraints, which the clamped functional misses (the class notes;
  // `-core clamped` shows the difference). Without an inner core every
  // solid is connected to the surface and the clamped functional IS the
  // certificate.
  std::vector<RigidComponent> rigid;
  if (layers.has_core && std::string(core) == "rigid") {
    rigid.resize(1);
    rigid[0].fluid_bdr_marker.SetSize(fluid.bdr_attributes.Max());
    rigid[0].fluid_bdr_marker = 0;
    rigid[0].fluid_bdr_marker[0] = 1;  // the ICB
    rigid[0].parent_attributes.SetSize(1);
    rigid[0].parent_attributes[0] = 1;  // the inner core
  }
  DensityFeasibilityProblem problem(fes_phi, dtn_degree, kG,
                                    Array<int>({layers.fluid}), fes_u,
                                    fes_p, nullptr, &ess,
                                    rigid.empty() ? nullptr : &rigid);
  Spaces s{&parent,      &fluid,       layers, &fes_phi,   &fes_u, &fes_p,
           &ctrl_fluid,  &ctrl_parent, &ess,   dtn_degree, &problem};

  Density rho(layers, ctrl_fluid, ctrl_parent);

  // dev_over_p = |dev T| / |p| over the fluid: the dimensionless
  // infeasibility ("how non-fluid is the state"), the quantity a
  // stopping rule should watch rather than the dimensional J.
  auto dev_over_p = [](const DensityFeasibility& J) {
    ConstantCoefficient zero(0.0);
    const double p = J.Stress().Pressure().ComputeL2Error(zero);
    return p > 0.0 ? std::sqrt(2.0 * J.Value()) / p : 0.0;
  };
  examples::CsvTable history(csv_file, {"iteration", "J", "step",
                                        "mass_drift", "dev_over_p",
                                        "energy", "core_motion"});
  history.Meta("title", "density restoration: the feasibility functional")
      .Meta("x", "iteration")
      .Meta("y", "J")
      .Meta("logy", "1")
      .Meta("note", std::string("loop ") + loop + ", metric " + metric_name);
  // The scatter beside the history, named after it.
  std::string scatter_file;
  if (csv_file && *csv_file) {
    scatter_file = csv_file;
    const auto dot = scatter_file.rfind(".csv");
    scatter_file =
        scatter_file.substr(0, dot == std::string::npos ? scatter_file.size()
                                                        : dot) +
        "_barotropy.csv";
  }
  examples::CsvTable scatter(scatter_file, {"state", "Phi", "rho"});
  scatter.Meta("title", "the (Phi, rho) scatter over the fluid")
      .Meta("x", "Phi")
      .Meta("y", "rho")
      .Meta("group", "state")
      .Meta("note", "barotropy restored = the scatter collapses to a curve");

  // Five windows: the whole-body density and stress before and after,
  // and the initial relaxation flow on the fluid.
  examples::GLVisWindow flow("initial relaxation flow |u|",
                             examples::DefaultKeys(dim));
  examples::GLVisWindow body_rho_before("initial density (body)",
                                        examples::DefaultKeys(dim));
  examples::GLVisWindow body_rho_after("final density (body)",
                                       examples::DefaultKeys(dim));
  examples::GLVisWindow body_stress_before(
      "initial |dev T| / p_rms (body, global minimiser)",
      examples::DefaultKeys(dim));
  examples::GLVisWindow body_stress_after("final |dev T| / p_rms (body)",
                                          examples::DefaultKeys(dim));

  // The pointwise |dev T| against a SINGLE pressure scale — where the
  // body is being asked to carry shear. The scale must be one number:
  // a pointwise division is singular wherever the pressure crosses
  // zero — the gauged fluid pressure inside the fluid (measured as
  // mesh-dependent spikes three orders over the background), and the
  // PHYSICAL pressure at the free surface. The maps use the RMS of the
  // physical fluid pressure (the gauged certificate pressure plus the
  // recovered datum), so a map value reads directly as "deviatoric
  // stress as this fraction of the actual pressure" — O(flattening)
  // for the mantle, the Love-obstruction scaling. The global stopping
  // diagnostic eta keeps its own scale, the pressure VARIATION
  // (gauge-invariant without any datum).
  struct InfeasibilityCoefficient : public Coefficient {
    MatrixCoefficient* T;
    double scale;
    mutable DenseMatrix M;
    double Eval(ElementTransformation& tr,
                const IntegrationPoint& ip) override {
      T->Eval(M, tr, ip);
      const int d = M.Height();
      double trace = 0.0;
      for (int i = 0; i < d; i++) {
        trace += M(i, i);
      }
      double dev2 = 0.0;
      for (int i = 0; i < d; i++) {
        for (int k = 0; k < d; k++) {
          const double v = M(i, k) - (i == k ? trace / d : 0.0);
          dev2 += v * v;
        }
      }
      return std::sqrt(dev2) / scale;
    }
  };
  // Mean, variation RMS and area of a pressure field over its own
  // mesh: mean and area through the pressure space's mass functional,
  // the variation norm through ComputeL2Error (which reduces globally
  // itself). The RMS of the field itself is sqrt(variation^2 + mean^2).
  struct PressureStats {
    double variation, mean, area;
  };
  auto pressure_stats = [&](const GridFunction& p_in, SpaceType& fes) {
    auto& p = const_cast<GridFunction&>(p_in);
    ConstantCoefficient one_c(1.0);
#ifdef MFEM_USE_MPI
    ParLinearForm mass_lf(&fes);
#else
    LinearForm mass_lf(&fes);
#endif
    mass_lf.AddDomainIntegrator(new DomainLFIntegrator(one_c));
    mass_lf.Assemble();
    Vector mass_dual(fes.GetTrueVSize()), p_true(fes.GetTrueVSize()),
        ones_true(fes.GetTrueVSize());
#ifdef MFEM_USE_MPI
    mass_lf.ParallelAssemble(mass_dual);
#else
    mass_dual = mass_lf;
#endif
    p.GetTrueDofs(p_true);
    GridType ones_gf(&fes);
    ones_gf = 1.0;
    ones_gf.GetTrueDofs(ones_true);
    PressureStats s;
    s.area = GlobalSum(mass_dual * ones_true);
    s.mean = GlobalSum(mass_dual * p_true) / s.area;
    ConstantCoefficient mean_c(s.mean);
    s.variation = p.ComputeL2Error(mean_c) / std::sqrt(s.area);
    return s;
  };

  // The initial relaxation flow |u| on the fluid.
  auto send_flow = [&](DensityFeasibility& J) {
    VectorGridFunctionCoefficient u(&J.Velocity());
    InnerProductCoefficient u2(u, u);
    PowerCoefficient umag(u2, 0.5);
    GridType u_f(&ctrl_fluid);
    u_f.ProjectCoefficient(umag);
    flow.Send(fluid, u_f);
  };

  // The whole-body view (-vis), in TWO EXACT PIECES rather than one
  // weighted approximation (whose finite contrast leaks the solid's
  // deviatoric stress into the fluid — the leakage proposition,
  // doc/equilibrium_figures.tex §2 — burying the fluid's own share):
  //
  //   fluid   the certificate's stress itself;
  //   solid   per CONNECTED component — the mantle, and the inner core
  //           when there is one — its minimum-deviatoric equilibrium
  //           stress under the state's potential, loaded on its fluid
  //           interface by the certificate's pressure (T n = -p n; the
  //           enclosed fluid's pressure gauge adds no net force on a
  //           closed interface and cannot move dev T), traction-free
  //           at the surface. At the restored state the fluid is
  //           hydrostatic by construction while the aspherical solid
  //           keeps its unavoidable share (Love's obstruction: its
  //           boundaries are not equipotentials); a -blob shows as a
  //           stressed halo in the mantle. The printed ||dev T|| split
  //           is the same statement in numbers (the fluid's value is
  //           sqrt(2 J), exactly).
  //
  // Every piece shares the fluid p_rms scale and the combined map is
  // shown over the whole body.
  GridType c_body(&fes_h1_body);
  GridFunctionCoefficient c_body_coeff(&c_body);
  SumCoefficient fluid_on_body(rho.fluid_base, c_body_coeff);
  PWCoefficient rho_body;
  if (layers.has_core) {
    rho_body.UpdateCoefficient(1, rho.inner);
  }
  rho_body.UpdateCoefficient(layers.fluid, fluid_on_body);
  rho_body.UpdateCoefficient(layers.mantle, rho.mantle);

  // The fields the combined whole-body map rides on: a parent
  // accumulator every piece writes into, and the body-side twin that
  // is sent. ELEMENTWISE (L2): |dev T| genuinely jumps at the CMB and
  // ICB, which a continuous field can only render as an interface
  // artefact — refine the mesh for a finer picture. (The element-local
  // dofs also make the pieces' transfers order-independent.)
  L2_FECollection l2_map(order - 1, dim);
  SpaceType fes_map_parent(&parent, &l2_map);
  SpaceType fes_map_body(&body, &l2_map);
  SpaceType fes_pf_parent(&parent, &h1_p);  // the fluid pressure's twin

  // int dev A : dev B over a mesh, by quadrature (the deviatoric Gram
  // the gauge optimisation and the printed norms use); with a marker,
  // over the marked attributes only.
  auto dev_pair = [&](MatrixCoefficient& A, MatrixCoefficient& B,
                      Mesh& on, const Array<int>* marker = nullptr) {
    DenseMatrix MA, MB;
    double pair = 0.0;
    const bool same = &A == &B;
    for (int e = 0; e < on.GetNE(); e++) {
      if (marker && !(*marker)[on.GetAttribute(e) - 1]) {
        continue;
      }
      auto* tr = on.GetElementTransformation(e);
      const auto& ir = IntRules.Get(on.GetElementGeometry(e), 2 * order);
      for (int q = 0; q < ir.GetNPoints(); q++) {
        const auto& ip = ir.IntPoint(q);
        tr->SetIntPoint(&ip);
        const double w = ip.weight * tr->Weight();
        A.Eval(MA, *tr, ip);
        if (same) {
          MB = MA;
        } else {
          B.Eval(MB, *tr, ip);
        }
        double tra = 0.0, trb = 0.0;
        for (int i = 0; i < dim; i++) {
          tra += MA(i, i);
          trb += MB(i, i);
        }
        for (int i = 0; i < dim; i++) {
          for (int k = 0; k < dim; k++) {
            pair += w * (MA(i, k) - (i == k ? tra / dim : 0.0)) *
                    (MB(i, k) - (i == k ? trb / dim : 0.0));
          }
        }
      }
    }
    return GlobalSum(pair);
  };

  // One solid component's generator and its contribution to the
  // combined map: returns ||dev T|| over the component. The enclosed
  // fluid's pressure is gauged (its constant is projected), and on a
  // component whose boundary the interface only PARTLY covers — the
  // mantle, whose other boundary is the free surface — a constant
  // interface pressure carries real deviatoric stress (the thick-shell
  // Lame solution), so the gauge matters there. The constant is a
  // genuine parameter of the equilibrium family, and the generator's
  // own principle recovers it: solve once more for the unit-pressure
  // (Lame) response L and minimise ||dev(T0 + c L)|| over c. The
  // recovered c is the physical pressure datum the free surface
  // imposes. A fully covered boundary (the inner core's ICB) is
  // gauge-invariant and skips the extra solve.
  // The maps are projected at UNIT scale and divided by the physical
  // pressure RMS at the end, once the datum that anchors it is known.
  auto solid_piece = [&](int attr, int interface_attr, Coefficient& rho_c,
                         DensityFeasibility& J, const GridType& p_parent,
                         GridType& map_parent, double* gauge = nullptr) {
    auto comp = SubMeshType::CreateFromDomain(parent, Array<int>({attr}));
    SpaceType u_fes(&comp, &h1, dim);
    SpaceType p_fes(&comp, &h1_p);
    SpaceType s_fes(&comp, &h1);

    GridType phi_c(&s_fes);
    phi_c = 0.0;
    Transfer(J.Potential(), phi_c);
    GridType p_c(&p_fes);
    p_c = 0.0;
    Transfer(p_parent, p_c);
    GridFunctionCoefficient p_c_coeff(&p_c);

    Array<int> interface(comp.bdr_attributes.Max());
    interface = 0;
    interface[interface_attr - 1] = 1;

    GradientGridFunctionCoefficient grad_phi(&phi_c);
    ScalarVectorProductCoefficient f_c(rho_c, grad_phi);
    MinimumDeviatoricEquilibriumStress T0(u_fes, p_fes, f_c, nullptr,
                                          nullptr, nullptr, nullptr,
                                          Vector(), nullptr, &p_c_coeff,
                                          &interface);

    // The gauge: only when part of the boundary is pressure-free.
    std::unique_ptr<MinimumDeviatoricEquilibriumStress> L;
    std::unique_ptr<MatrixSumCoefficient> combined;
    MatrixCoefficient* T = &T0;
    double c = 0.0;
    if (comp.bdr_attributes.Max() > 1) {
      Vector zero_v(dim);
      zero_v = 0.0;
      VectorConstantCoefficient no_force(zero_v);
      ConstantCoefficient unit(1.0);
      L = std::make_unique<MinimumDeviatoricEquilibriumStress>(
          u_fes, p_fes, no_force, nullptr, nullptr, nullptr, nullptr,
          Vector(), nullptr, &unit, &interface);
      const double ll = dev_pair(*L, *L, comp);
      c = ll > 0.0 ? -dev_pair(T0, *L, comp) / ll : 0.0;
      combined = std::make_unique<MatrixSumCoefficient>(T0, *L, 1.0, c);
      T = combined.get();
    }
    if (gauge) {
      *gauge = c;
    }

    InfeasibilityCoefficient eta_c;
    eta_c.T = T;
    eta_c.scale = 1.0;
    SpaceType map_fes(&comp, &l2_map);
    GridType map_c(&map_fes);
    map_c.ProjectCoefficient(eta_c);
    Transfer(map_c, map_parent);
    return std::sqrt(dev_pair(*T, *T, comp));
  };

  // The whole-body stress window. At an unrestored state the two-piece
  // field is NOT a viable stress field — the fluid's optimal stress
  // carries an O(sqrt J) viscous interface traction the solid is not
  // given — so the INITIAL window shows the one equilibrium field the
  // whole body does admit there: the unweighted GLOBAL minimiser,
  // whose fluid share includes the solid's leakage (the leakage
  // proposition; the printed fluid/solid split shows it). The FINAL
  // window shows the exact two-piece recovery, a genuine equilibrium
  // stress field to numerical convergence (the interface mismatch is
  // O(sqrt J) at the floor).
  auto send_body_state = [&](DensityFeasibility& J, const char* when,
                             bool global, examples::GLVisWindow& rho_w,
                             examples::GLVisWindow& stress_w) {
    c_body = 0.0;
    Transfer(rho.c_parent, c_body);

    L2_FECollection rho_fec(order, dim);
    SpaceType rho_fes(&body, &rho_fec);
    GridType rho_gf(&rho_fes);
    rho_gf.ProjectCoefficient(rho_body);
    rho_w.Send(body, rho_gf);

    GridType map_body(&fes_map_body);
    map_body = 0.0;

    if (global) {
      GridType phi_b(&fes_h1_body);
      phi_b = 0.0;
      Transfer(J.Potential(), phi_b);
      GradientGridFunctionCoefficient grad_phi(&phi_b);
      ScalarVectorProductCoefficient f_b(rho_body, grad_phi);
      MinimumDeviatoricEquilibriumStress T(fes_u_body, fes_p_body, f_b);

      Array<int> fluid_marker(body.attributes.Max());
      fluid_marker = 0;
      fluid_marker[layers.fluid - 1] = 1;
      Array<int> solid_marker(body.attributes.Max());
      for (int a = 0; a < solid_marker.Size(); a++) {
        solid_marker[a] = 1 - fluid_marker[a];
      }
      if (Root()) {
        std::cout << "  ||dev T|| (" << when << ", global minimiser): ";
      }
      const double dev_f = std::sqrt(dev_pair(T, T, body, &fluid_marker));
      const double dev_s = std::sqrt(dev_pair(T, T, body, &solid_marker));
      if (Root()) {
        std::cout << "fluid " << dev_f << " (leakage included), solid "
                  << dev_s << "\n";
      }
      InfeasibilityCoefficient eta_c;
      eta_c.T = &T;
      eta_c.scale = 1.0;
      map_body.ProjectCoefficient(eta_c);
      // The global solve's pressure is datum-determined by the free
      // surface: its RMS over the fluid is the map's physical scale
      // (hoisted to the fluid submesh through the parent).
      {
        GridType p_par(&fes_pf_parent);
        p_par = 0.0;
        Transfer(T.Pressure(), p_par);
        GridType p_fl(&fes_p);
        p_fl = 0.0;
        Transfer(p_par, p_fl);
        const PressureStats s = pressure_stats(p_fl, fes_p);
        const double p_phys = std::max(
            std::sqrt(s.variation * s.variation + s.mean * s.mean), 1e-300);
        map_body *= 1.0 / p_phys;
      }
      stress_w.Send(body, map_body);
      return;
    }

    // The certificate's pressure, hoisted to the parent for the
    // solid components.
    GridType p_parent(&fes_pf_parent);
    p_parent = 0.0;
    Transfer(J.Stress().Pressure(), p_parent);

    GridType map_parent(&fes_map_parent);
    map_parent = 0.0;

    // The fluid piece: the certificate's own map.
    {
      InfeasibilityCoefficient eta_c;
      eta_c.T = &const_cast<MinimumDeviatoricEquilibriumStress&>(J.Stress());
      eta_c.scale = 1.0;
      SpaceType map_fes(&fluid, &l2_map);
      GridType map_f(&map_fes);
      map_f.ProjectCoefficient(eta_c);
      Transfer(map_f, map_parent);
    }

    const int cmb_bdr = layers.has_core ? 2 : 1;
    double gauge = 0.0;
    const double dev_mantle = solid_piece(layers.mantle, cmb_bdr,
                                          rho.mantle, J, p_parent,
                                          map_parent, &gauge);
    double dev_core = 0.0;
    if (layers.has_core) {
      dev_core = solid_piece(1, 1, rho.inner, J, p_parent, map_parent);
    }

    // The physical fluid pressure: the certificate's gauged field plus
    // the recovered datum; its RMS is the map's scale.
    const PressureStats s = pressure_stats(J.Stress().Pressure(), fes_p);
    const double shifted_mean = s.mean + gauge;
    const double p_phys = std::max(
        std::sqrt(s.variation * s.variation + shifted_mean * shifted_mean),
        1e-300);
    if (Root()) {
      std::cout << "  ||dev T|| (" << when
                << "): fluid " << std::sqrt(2.0 * J.Value())
                << ", mantle " << dev_mantle;
      if (layers.has_core) {
        std::cout << ", core " << dev_core;
      }
      std::cout << "  (fluid pressure datum " << gauge << ", rms "
                << p_phys << ")\n";
    }
    Transfer(map_parent, map_body);
    map_body *= 1.0 / p_phys;
    stress_w.Send(body, map_body);
  };

  // What the enriched certificate sees that the clamped one does not:
  // the core's multiplier motion (its force/torque signature), and the
  // gap between the two functionals.
  auto report_core = [&](DensityFeasibility& J, const char* when) {
    if (rigid.empty() || !Root()) {
      return;
    }
    const Vector& a = J.RigidCoefficients();
    const int modes_per = dim == 3 ? 6 : 3;  // translations first
    double t2 = 0.0, r2 = 0.0;
    for (int k = 0; k < a.Size(); k++) {
      (k % modes_per < dim ? t2 : r2) += a[k] * a[k];
    }
    std::cout << "  core multiplier motion (" << when
              << "): |translation| = " << std::sqrt(t2)
              << ", |rotation| = " << std::sqrt(r2) << "\n";
  };
  std::unique_ptr<DensityFeasibilityProblem> clamped_problem;
  auto report_gap = [&](const char* when) {
    if (rigid.empty()) {
      return;
    }
    if (!clamped_problem) {
      clamped_problem = std::make_unique<DensityFeasibilityProblem>(
          fes_phi, dtn_degree, kG, Array<int>({layers.fluid}), fes_u,
          fes_p, nullptr, &ess);
    }
    const double j_clamped =
        clamped_problem->Evaluate(rho.parent, rho.stokes)->Value();
    if (Root()) {
      std::cout << "  clamped (connected-solid) functional (" << when
                << "): " << j_clamped
                << " — the certificate gap is the core's unbalanced "
                   "force and torque\n";
    }
  };

  // The stopping rule on the dimensionless infeasibility (the header
  // comment): an explicit -eta threshold, or, at -eta 0, stagnation —
  // two consecutive accepted iterations improving eta by less than
  // 0.1% mark the discretisation floor. Stagnation only counts once
  // eta has fallen 10% from its start: the Levenberg-Marquardt warm-up
  // (lambda from 1) also moves slowly, and must not trip it. (The
  // advect loop keeps only the explicit threshold: its decay is slow
  // by nature, and it has its own honest stall detection at the
  // explicit-step floor.)
  double eta_start = 0.0;
  double eta_prev = std::numeric_limits<double>::max();
  int eta_stalled = 0;
  auto eta_stop = [&](double eta_now) {
    if (eta > 0.0) {
      return eta_now < eta;
    }
    if (eta_now > 0.9 * eta_start) {
      return false;
    }
    eta_stalled = eta_now > (1.0 - 1e-3) * eta_prev ? eta_stalled + 1 : 0;
    eta_prev = eta_now;
    return eta_stalled >= 2;
  };

  // The initial state.
  auto J0 = Evaluate(s, rho);
  const double j_start = J0->Value();
  eta_start = dev_over_p(*J0);
  if (Root()) {
    std::cout << "J at the start: " << std::scientific << j_start
              << "  (eta = " << eta_start << ")\n";
  }
  report_core(*J0, "initial");
  report_gap("initial");
  BarotropyRows(s, rho, *J0, "initial", scatter);
  if (visualisation) {
    send_flow(*J0);
    send_body_state(*J0, "initial", /*global=*/true, body_rho_before,
                    body_stress_before);
  }

  const bool advect = std::string(loop) == "advect";
  const bool gauss_newton = std::string(loop) == "gn";
  double j_final = j_start;
  int done = 0;
  std::unique_ptr<DensityFeasibility> J_last = std::move(J0);

  if (!advect) {
    // --- The library loops (descent.hpp) -----------------------------------
    // -loop cg: projected Polak-Ribiere+ CG in the chosen metric;
    // -loop gn: Levenberg-Marquardt Gauss-Newton (which needs the L2
    // control for its Hessian plumbing). The adapter syncs the density,
    // evaluates through the persistent problem, and serves duals and
    // Gauss-Newton products from the last state; the metric — Riesz
    // map, fixed-mass plane, roughness prior — and the loops are the
    // library's.
    Control control = MakeControl(gauss_newton ? "l2" : metric_name,
                                  length, prior, s);
    GridType scratch_parent(&ctrl_parent), scratch_fluid(&ctrl_fluid);
    GridType d_fluid(&ctrl_fluid), d_parent(&ctrl_parent);
    GridFunctionCoefficient dc_fluid(&d_fluid), dc_parent(&d_parent);

    DescentFunctional f;
    f.evaluate = [&](const Vector& c, Vector* dual) {
      SyncDensity(control, c, rho, scratch_parent, scratch_fluid);
      J_last = Evaluate(s, rho);
      if (dual) {
        if (control.on_parent) {
          J_last->DerivativeOnParent(*control.fes, *dual);
        } else {
          J_last->Derivative(*control.fes, *dual);
        }
      }
      return J_last->Value();
    };
    f.gauss_newton = [&](const Vector& d, Vector& Hd) {
      d_fluid.SetFromTrueDofs(d);
      d_parent = 0.0;
      Transfer(d_fluid, d_parent);
      J_last->HessianAction(rho.parent, rho.stokes, dc_parent, dc_fluid,
                            *control.fes, Hd, /*gn_only=*/true);
    };

    DescentOptions options;
    options.max_iterations = iters;
    options.tolerance = tol;
    // With the prior on, the LM inner solve is mesh-independent only
    // under a preconditioner spectrally equivalent to lambda M +
    // lambda_p K: a Sobolev Riesz map on the control space (the
    // descent.hpp note; beta is the prior weight against the initial
    // lambda = 1).
    std::unique_ptr<SobolevRieszMap> inner_prec;
    if (gauss_newton && prior > 0.0) {
      inner_prec = std::make_unique<SobolevRieszMap>(ctrl_fluid, 1.0,
                                                     prior, 1, nullptr);
      options.inner_preconditioner = inner_prec.get();
    }
    double eta_now = 0.0;
    options.monitor = [&](int it, double value, double step) {
      eta_now = dev_over_p(*J_last);
      history.Row({double(it), value, step, 0.0, eta_now,
                   J_last->GravitationalEnergy(),
                   J_last->RigidCoefficients().Norml2()});
      if (Root()) {
        std::cout << (gauss_newton ? "lm " : "cg ") << std::setw(4) << it
                  << ": J = " << value
                  << (gauss_newton ? ", lambda " : ", step ") << step
                  << ", eta " << eta_now << "\n";
      }
    };
    // The loops consult the stop after the monitor, so eta_now is the
    // accepted iterate's.
    options.stop = [&](int, double) { return eta_stop(eta_now); };

    Vector c(control.fes->GetTrueVSize());
    c = 0.0;
    const DescentResult r =
        gauss_newton
            ? LevenbergMarquardt(f, *control.metric, *control.mass_M.Ptr(),
                                 c, options)
            : NonlinearCG(f, *control.metric, c, options);
    SyncDensity(control, c, rho, scratch_parent, scratch_fluid);
    done = r.iterations;
    j_final = r.final;
  } else {
    // --- The advection flow ------------------------------------------------
    // rho stepped along the Stokes multiplier: an explicit step of the
    // energy's gradient flow, the increment L2-projected back onto the
    // fluid control space; the step adapts so J never increases.
    L2RieszMap mass(ctrl_fluid);
    GridType rho_f(&ctrl_fluid);
    rho_f.ProjectCoefficient(rho.stokes);
    Vector c(ctrl_fluid.GetTrueVSize());
    // The control is the perturbation about the analytic base.
    GridType base_f(&ctrl_fluid);
    base_f.ProjectCoefficient(rho.fluid_base);
    rho_f -= base_f;
    rho_f.GetTrueDofs(c);
    rho.Sync(c);
    J_last = Evaluate(s, rho);
    double jval = J_last->Value();

    // The fluid mass functional, for the conservation diagnostic.
    Vector m_dual(c.Size());
    {
      ConstantCoefficient one(1.0);
#ifdef MFEM_USE_MPI
      ParLinearForm mlf(&ctrl_fluid);
#else
      LinearForm mlf(&ctrl_fluid);
#endif
      mlf.AddDomainIntegrator(new DomainLFIntegrator(one));
      mlf.Assemble();
#ifdef MFEM_USE_MPI
      mlf.ParallelAssemble(m_dual);
#else
      m_dual = mlf;
#endif
    }
    const double mass_start = mass.Pair(m_dual, c);

    // The CFL-type step cap: the update is a genuine rearrangement only
    // to discretisation level (div u = 0 holds weakly, against the
    // pressure space), and OFF the rearrangement manifold the energy is
    // unbounded below — mass piling toward the centre lowers E while J
    // explodes (discrete gravitational collapse, observed without this
    // cap). Advective-scale steps keep the off-manifold drift at the
    // discretisation level the Armijo test can police.
    double h_min = std::numeric_limits<double>::max();
    for (int e = 0; e < fluid.GetNE(); e++) {
      h_min = std::min(h_min, fluid.GetElementSize(e));
    }
#ifdef MFEM_USE_MPI
    {
      double global = 0.0;
      MPI_Allreduce(&h_min, &global, 1, MPI_DOUBLE, MPI_MIN,
                    MPI_COMM_WORLD);
      h_min = global;
    }
#endif

    double dt = 1.0;
    for (int it = 1; it <= iters; it++) {
      // The advective increment in its WEAK form: with div u = 0 and
      // u = 0 on the boundary, (d rho, v) = dt int rho u . grad v —
      // the derivative lands on the test function, the control is
      // never differentiated (no numerical-differentiation noise), and
      // v = 1 shows the discrete mass is conserved exactly.
      VectorGridFunctionCoefficient u(&J_last->Velocity());
      ScalarVectorProductCoefficient rho_u(rho.stokes, u);
#ifdef MFEM_USE_MPI
      ParLinearForm lf(&ctrl_fluid);
#else
      LinearForm lf(&ctrl_fluid);
#endif
      lf.AddDomainIntegrator(new DomainLFGradIntegrator(rho_u));
      lf.Assemble();
      Vector j_adv(c.Size());
#ifdef MFEM_USE_MPI
      lf.ParallelAssemble(j_adv);
#else
      j_adv = lf;
#endif
      Vector delta(c.Size());
      mass.Mult(j_adv, delta);

      // Step acceptance is on the ENERGY: the flow is E's gradient
      // flow with decay rate 2J, and J itself need not fall
      // monotonically along it.
      const double e_old = J_last->GravitationalEnergy();
      double u_max = J_last->Velocity().Normlinf();
#ifdef MFEM_USE_MPI
      {
        double global = 0.0;
        MPI_Allreduce(&u_max, &global, 1, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);
        u_max = global;
      }
#endif
      dt = std::min(2.0 * dt,
                    u_max > 0.0 ? 0.5 * h_min / u_max
                                : 2.0 * dt);
      std::unique_ptr<DensityFeasibility> trial;
      double j_new = jval;
      Vector c_trial(c.Size());
      while (true) {
        c_trial = c;
        c_trial.Add(dt, delta);
        rho.Sync(c_trial);
        trial = Evaluate(s, rho);
        j_new = trial->Value();
        // Armijo on the energy (along the flow dE/dt = -2J exactly),
        // PLUS a guard on J's growth: the projected Eulerian step
        // leaks the rearrangement constraint through its compressive
        // discretisation error, and OFF the rearrangement manifold E
        // is unbounded below (discrete gravitational collapse, which
        // raises J monotonically as it feeds). The guard forbids that
        // channel; a range-preserving (semi-Lagrangian composition)
        // step is the structural fix, noted in the planning document.
        if ((trial->GravitationalEnergy() <= e_old - 0.2 * dt * jval &&
             j_new <= 2.0 * j_start) ||
            dt < 1e-14) {
          break;
        }
        dt *= 0.5;
      }
      if (dt < 1e-14) {
        // No step of the explicit scheme decreases J: the advection
        // flow has reached its discrete floor.
        if (Root()) {
          std::cout << "advect: stalled at the explicit-step floor, "
                       "iteration " << it << "\n";
        }
        break;
      }
      c = c_trial;
      jval = j_new;
      J_last = std::move(trial);
      // Mass drift: advection conserves int rho only to discretisation.
      const double drift = mass.Pair(m_dual, c) - mass_start;
      const double eta_now = dev_over_p(*J_last);
      history.Row({double(it), jval, dt, drift, eta_now,
                   J_last->GravitationalEnergy(),
                   J_last->RigidCoefficients().Norml2()});
      if (Root() && (it % 10 == 0 || it == 1)) {
        std::cout << "advect " << std::setw(4) << it << ": J = " << jval
                  << ", dt " << dt << ", eta " << eta_now << "\n";
      }
      done = it;
      if (eta > 0.0 && eta_now < eta) {
        break;
      }
      if (tol > 0.0 && jval < tol * j_start) {
        break;
      }
    }
    rho.Sync(c);
    j_final = jval;
  }

  // Collective (ComputeL2Error), so computed on every rank before the
  // root-only print.
  const double eta_final = dev_over_p(*J_last);
  if (Root()) {
    std::cout << "J: " << j_start << " -> " << j_final << "  ("
              << j_final / j_start << " of the start, " << done
              << " iterations, final eta = " << eta_final << ")\n";
  }
  report_core(*J_last, "final");
  report_gap("final");
  BarotropyRows(s, rho, *J_last, "final", scatter);
  if (visualisation) {
    send_body_state(*J_last, "final", /*global=*/false, body_rho_after,
                    body_stress_after);
  }
  history.Write();
  scatter.Write();
  return 0;
}
