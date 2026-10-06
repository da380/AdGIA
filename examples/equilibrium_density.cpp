// ============================================================================
// equilibrium_density.cpp
//
// Density restoration — the first milestone of the equilibrium-figures
// programme (doc/equilibrium_figures.tex §8). The fluid outer core of the
// three-layer model starts from a laterally varying, NON-barotropic
// density: no static state exists, and the fluid-only feasibility
// functional
//
//   J = min { 1/2 int_fluid |dev T|^2 : Div T = rho grad Phi in the
//             fluid, interface traction free }
//
// is positive, with the minimising multiplier u the creeping flow the
// unbalanced buoyancy would drive. The example drives J to its floor by
// moving the fluid density, two ways:
//
//   -loop cg      nonlinear conjugate gradients (Polak-Ribiere+, Armijo
//                 backtracking) on the density, the derivative by the
//                 envelope theorem (DensityFeasibility) and the gradient
//                 through a choosable Riesz map:
//                   -metric l2   the L2 identification on the fluid
//                   -metric h1   the Sobolev metric on the whole mesh
//                   -metric h2   its iterated (order-2) form, the
//                                working default of the programme
//                 (-length sets the smoothing length sqrt(beta/alpha)).
//                 The fluid mass is held fixed by projecting the search
//                 direction in the metric.
//
//   -loop advect  the advection flow: rho stepped along u itself — the
//                 gradient flow of the gravitational energy in the
//                 dissipation metric, mass- and (up to projection)
//                 distribution-preserving; J is its decay rate, and the
//                 step adapts so J never increases.
//
// At J's floor the fluid is barotropic: rho constant on equipotentials.
// The (Phi, rho) scatter over the fluid collapses onto a single curve —
// the before/after scatter is the example's physical verdict.
//
// Outputs: equilibrium_density.csv (J, the flow and step diagnostics by
// iteration) and equilibrium_density_barotropy.csv (the scatter), both
// for plot_csv.py; with -vis, GLVis windows of the initial and final
// fluid density (start `glvis` first).
//
// One source serves the serial and the parallel build, as in
// equilibrium_stress.cpp; the barotropy scatter is gathered to the root.
//
// Sample runs (with mpiexec -np N in front in a parallel build):
//    ./equilibrium_density
//    ./equilibrium_density -metric l2 -iters 40
//    ./equilibrium_density -metric h2 -length 0.2
//    ./equilibrium_density -loop advect -iters 200
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

// The three-layer mesh's conventions (meshes/README.md; also the test
// meshes): attributes 1 inner core, 2 fluid outer core, 3 mantle,
// 4 buffer; boundary attributes 1 ICB, 2 CMB, 3 surface, 4 outer.
constexpr int kFluidAttr = 2;
constexpr double kG = 0.05;

double lateral_amplitude = 0.1;

// The fluid's starting density: barotropic base plus a lateral part of
// amplitude -amp, which no rearrangement-free relabelling can remove.
double FluidRho(const Vector& x) {
  const double r = x.Norml2();
  return 1.2 - 0.3 * r * r + lateral_amplitude * (r > 0.0 ? x[1] / r : 0.0);
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
  Density(SpaceType& fes_fluid, SpaceType& fes_fluid_on_parent)
      : fluid_base(FluidRho),
        inner(1.3),
        mantle(1.0),
        buffer(0.0),
        c_fluid(&fes_fluid),
        c_parent(&fes_fluid_on_parent),
        c_fluid_coeff(&c_fluid),
        c_parent_coeff(&c_parent),
        stokes(fluid_base, c_fluid_coeff),
        fluid_on_parent(fluid_base, c_parent_coeff) {
    c_fluid = 0.0;
    c_parent = 0.0;
    parent.UpdateCoefficient(1, inner);
    parent.UpdateCoefficient(kFluidAttr, fluid_on_parent);
    parent.UpdateCoefficient(3, mantle);
    parent.UpdateCoefficient(4, buffer);
  }

  // The fluid perturbation from its true dofs (on the fluid space).
  void Sync(const Vector& c) {
    c_fluid.SetFromTrueDofs(c);
    c_parent = 0.0;
    Transfer(c_fluid, c_parent);
  }

  FunctionCoefficient fluid_base;
  ConstantCoefficient inner, mantle, buffer;
  GridType c_fluid, c_parent;
  GridFunctionCoefficient c_fluid_coeff, c_parent_coeff;
  SumCoefficient stokes, fluid_on_parent;
  PWCoefficient parent;
};

struct Spaces {
  MeshType* parent;
  SubMeshType* fluid;
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

// The metric: where the control lives, how the dual is assembled, and
// the Riesz map identifying gradients.
struct Metric {
  SpaceType* fes;            // the control space (fluid or parent)
  bool on_parent;            // Sobolev metrics live on the parent
  std::unique_ptr<RieszMap> riesz;
  Vector mass_dual;          // l(v) = int_fluid v: the constraint's dual
  Vector mass_gradient;      // its Riesz representative

  // The roughness prior: lambda/2 c^T K c with K the H1-seminorm Gram
  // on the control space — a selection within the near-null directions
  // of J, biased toward smoothness rather than toward a known answer.
  double lambda = 0.0;
  std::unique_ptr<FormType> prior_form;
  OperatorHandle prior_K;

  double PriorValue(const Vector& c) const {
    if (lambda == 0.0) {
      return 0.0;
    }
    Vector Kc(c.Size());
    prior_K.Ptr()->Mult(c, Kc);
    return 0.5 * lambda * riesz->Pair(Kc, c);
  }

  void AssembleDual(DensityFeasibility& J, const Vector& c,
                    Vector& j) const {
    if (on_parent) {
      J.DerivativeOnParent(*fes, j);
    } else {
      J.Derivative(*fes, j);
    }
    if (lambda != 0.0) {
      Vector Kc(c.Size());
      prior_K.Ptr()->Mult(c, Kc);
      j.Add(lambda, Kc);
    }
  }

  // Remove the mass component of a direction (the metric-orthogonal
  // projection onto the fixed-fluid-mass plane).
  void ProjectMass(Vector& d) const {
    const double scale = riesz->Pair(mass_dual, mass_gradient);
    const double along = riesz->Pair(mass_dual, d);
    d.Add(-along / scale, mass_gradient);
  }
};

Metric MakeMetric(const std::string& name, double length, double lambda,
                  const Spaces& s) {
  Metric m;
  m.on_parent = name != "l2";
  m.fes = m.on_parent ? s.ctrl_parent : s.ctrl_fluid;
  m.lambda = lambda;
  if (lambda != 0.0) {
    m.prior_form = std::make_unique<FormType>(m.fes);
    m.prior_form->AddDomainIntegrator(new DiffusionIntegrator());
    m.prior_form->Assemble();
    Array<int> empty;
    m.prior_form->FormSystemMatrix(empty, m.prior_K);
  }
  if (m.on_parent) {
    const int order = name == "h2" ? 2 : 1;
    // Dirichlet on the outer boundary of the extension domain.
    auto outer = std::make_unique<Array<int>>(
        s.parent->bdr_attributes.Max());
    *outer = 0;
    (*outer)[s.parent->bdr_attributes.Max() - 1] = 1;
    m.riesz = std::make_unique<SobolevRieszMap>(
        *m.fes, 1.0, length * length, order, outer.get());
    // The map copies the marker's dofs at construction; the Array may go.
  } else {
    m.riesz = std::make_unique<L2RieszMap>(*m.fes);
  }

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
      fluid_marker[kFluidAttr - 1] = 1;
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
  m.mass_gradient.SetSize(m.mass_dual.Size());
  m.riesz->Mult(m.mass_dual, m.mass_gradient);
  return m;
}

// The control seen by the density: on the parent the model reads the
// restriction, which Sync takes from the fluid field, so the fluid
// twin must follow the parent control.
void SyncDensity(const Metric& m, const Vector& c, Density& rho,
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

  const char* mesh_file = "../data/elastogravity_three_layer_3d.msh";
  const char* loop = "cg";
  const char* metric_name = "h2";
  double length = 0.3;
  double prior = 0.0;
  int order = 2;
  int dtn_degree = 8;
  int iters = 60;
  double amplitude = 0.1;
  double tol = 1e-3;  // stop when J has fallen by this factor
  bool visualisation = false;
  const char* csv_file = "equilibrium_density.csv";

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file (three-layer).");
  args.AddOption(&loop, "-loop", "--loop", "cg or advect.");
  args.AddOption(&metric_name, "-metric", "--metric",
                 "Gradient metric of the cg loop: l2, h1 or h2.");
  args.AddOption(&length, "-length", "--length",
                 "Smoothing length of the Sobolev metrics.");
  args.AddOption(&prior, "-prior", "--prior",
                 "Roughness (H1-seminorm) prior weight of the cg loop: "
                 "selects the smooth member of the near-null family.");
  args.AddOption(&order, "-o", "--order", "Velocity and control order.");
  args.AddOption(&dtn_degree, "-deg", "--degree", "DtN truncation degree.");
  args.AddOption(&iters, "-iters", "--iterations", "Iteration budget.");
  args.AddOption(&amplitude, "-amp", "--amplitude",
                 "Lateral (non-barotropic) amplitude of the start.");
  args.AddOption(&tol, "-tol", "--tolerance",
                 "Stop when J falls below this fraction of its start.");
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
  lateral_amplitude = amplitude;

  // The meshes: the parent ball with its buffer, and the fluid SubMesh.
#ifdef MFEM_USE_MPI
  Mesh serial(mesh_file, 1, 1);
  MeshType parent(MPI_COMM_WORLD, serial);
  serial.Clear();
#else
  MeshType parent(mesh_file, 1, 1);
#endif
  const int dim = parent.Dimension();
  MFEM_VERIFY(dim == 3, "the example is for balls (the 2-D exterior "
                        "problem has the known infrared caveats).");
  auto fluid = SubMeshType::CreateFromDomain(parent, Array<int>({kFluidAttr}));

  H1_FECollection h1(order, dim), h1_p(order - 1, dim);
  SpaceType fes_phi(&parent, &h1);
  SpaceType fes_u(&fluid, &h1, dim);
  SpaceType fes_p(&fluid, &h1_p);
  SpaceType ctrl_fluid(&fluid, &h1);
  SpaceType ctrl_parent(&parent, &h1);

  Array<int> ess(fluid.bdr_attributes.Max());
  ess = 1;  // the certificate's no-slip condition on the whole boundary

  DensityFeasibilityProblem problem(fes_phi, dtn_degree, kG,
                                    Array<int>({kFluidAttr}), fes_u, fes_p,
                                    nullptr, &ess);
  Spaces s{&parent,      &fluid, &fes_phi,    &fes_u, &fes_p,
           &ctrl_fluid,  &ctrl_parent, &ess,  dtn_degree, &problem};

  Density rho(ctrl_fluid, ctrl_parent);

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
                                        "energy"});
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

  examples::GLVisWindow before("initial fluid density",
                               examples::DefaultKeys(dim));
  examples::GLVisWindow after("final fluid density",
                              examples::DefaultKeys(dim));

  // The initial state.
  auto J0 = Evaluate(s, rho);
  const double j_start = J0->Value();
  if (Root()) {
    std::cout << "J at the start: " << std::scientific << j_start << "\n";
  }
  BarotropyRows(s, rho, *J0, "initial", scatter);
  if (visualisation) {
    GridType rho_f(&ctrl_fluid);
    rho_f.ProjectCoefficient(rho.stokes);
    before.Send(fluid, rho_f);
  }

  const bool advect = std::string(loop) == "advect";
  double j_final = j_start;
  int done = 0;
  std::unique_ptr<DensityFeasibility> J_last = std::move(J0);

  if (!advect) {
    // --- Nonlinear CG in the chosen metric ---------------------------------
    // The loop descends J_tot = J + prior; J alone (the certificate) is
    // what the history records and the stopping test reads.
    Metric metric = MakeMetric(metric_name, length, prior, s);
    GridType scratch_parent(&ctrl_parent), scratch_fluid(&ctrl_fluid);

    Vector c(metric.fes->GetTrueVSize());
    c = 0.0;
    Vector j(c.Size()), g(c.Size()), d(c.Size()), j_old(c.Size()),
        g_old(c.Size()), c_trial(c.Size());

    // The gradient is projected onto the fixed-mass plane as soon as it
    // is identified; the projector is metric-self-adjoint, so every
    // dual-gradient pairing below is then the manifold's own inner
    // product and the PR conjugacy is the constrained one.
    metric.AssembleDual(*J_last, c, j);
    metric.riesz->Mult(j, g);
    metric.ProjectMass(g);
    d = g;
    d *= -1.0;
    double jraw = j_start;
    double jval = j_start;  // the total
    double t = 1.0;

    bool restarted = false;
    for (int it = 1; it <= iters; it++) {
      const double slope = metric.riesz->Pair(j, d);
      if (slope >= 0.0) {
        if (restarted) {
          // The projected steepest descent itself does not descend:
          // the constrained optimum (to the solver floor).
          break;
        }
        // Not a descent direction (a stale conjugacy): restart.
        d = g;
        d *= -1.0;
        restarted = true;
        continue;
      }
      restarted = false;
      // Line search: J is an exact quartic along the ray, so track
      // forward — double the (warm-started) step while J keeps
      // falling, backtrack while it does not. The accepted point is
      // within a factor two of the quartic's minimiser.
      auto at = [&](double step, double& total) {
        c_trial = c;
        c_trial.Add(step, d);
        SyncDensity(metric, c_trial, rho, scratch_parent, scratch_fluid);
        auto df = Evaluate(s, rho);
        total = df->Value() + metric.PriorValue(c_trial);
        return df;
      };
      double j_new;
      std::unique_ptr<DensityFeasibility> trial = at(t, j_new);
      while (j_new >= jval && t > 1e-14) {
        t *= 0.5;
        trial = at(t, j_new);
      }
      while (t > 1e-14) {
        double j_next;
        auto next = at(2.0 * t, j_next);
        if (j_next >= j_new) {
          break;
        }
        t *= 2.0;
        trial = std::move(next);
        j_new = j_next;
      }
      // Leave the density at the accepted point (the last Evaluate may
      // have been the rejected probe).
      c_trial = c;
      c_trial.Add(t, d);
      SyncDensity(metric, c_trial, rho, scratch_parent, scratch_fluid);
      c = c_trial;
      jval = j_new;
      J_last = std::move(trial);
      jraw = J_last->Value();
      history.Row({double(it), jraw, t, 0.0, dev_over_p(*J_last),
                   J_last->GravitationalEnergy()});
      if (Root()) {
        std::cout << "cg " << std::setw(4) << it << ": J = " << jraw
                  << ", step " << t << "\n";
      }
      done = it;
      if (jraw < tol * j_start) {
        break;
      }
      // Roundoff hygiene: hold the iterate itself on the mass plane.
      metric.ProjectMass(c);
      j_old = j;
      g_old = g;
      metric.AssembleDual(*J_last, c, j);
      metric.riesz->Mult(j, g);
      metric.ProjectMass(g);
      // Polak-Ribiere+, in dual-gradient pairings (the projected
      // gradients make these the manifold inner products).
      Vector dg(g);
      dg -= g_old;
      const double beta =
          std::max(0.0, metric.riesz->Pair(j, dg) /
                            metric.riesz->Pair(j_old, g_old));
      d *= beta;
      d -= g;
    }
    // Leave the density at the accepted state.
    SyncDensity(metric, c, rho, scratch_parent, scratch_fluid);
    j_final = jraw;
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
      history.Row({double(it), jval, dt, drift, dev_over_p(*J_last),
                   J_last->GravitationalEnergy()});
      if (Root() && (it % 10 == 0 || it == 1)) {
        std::cout << "advect " << std::setw(4) << it << ": J = " << jval
                  << ", dt " << dt << "\n";
      }
      done = it;
      if (jval < tol * j_start) {
        break;
      }
    }
    rho.Sync(c);
    j_final = jval;
  }

  if (Root()) {
    std::cout << "J: " << j_start << " -> " << j_final << "  ("
              << j_final / j_start << " of the start, " << done
              << " iterations)\n";
  }
  BarotropyRows(s, rho, *J_last, "final", scatter);
  if (visualisation) {
    GridType rho_f(&ctrl_fluid);
    rho_f.ProjectCoefficient(rho.stokes);
    after.Send(fluid, rho_f);
  }
  history.Write();
  scatter.Write();
  return 0;
}
