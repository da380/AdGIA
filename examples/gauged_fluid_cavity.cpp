// ============================================================================
// gauged_fluid_cavity.cpp
//
// The gauged treatment of an inviscid fluid region on the simplest problem
// that has one: a purely elastic body with a fluid core under an external
// surface pressure, with no gravity. A demonstration of the gauged-fluid
// option of LinearQuasiStaticProblemBase (SetFluid with GaugePenaltyOptions;
// the method is in
// doc/gauged_fluid.md, "Gauge fixing: penalty plus iterated refinement").
//
// The fluid carries a displacement like the solid, with its physical bulk
// modulus and zero shear. That displacement is determined only up to a
// linearised relabelling (a volume-preserving rearrangement of fluid
// particles), so the discrete operator has a large near-kernel; the gauge
// is fixed by a small shear penalty eps * 2 mu_g dev e(u) : dev e(u') on
// the fluid, and the O(eps) bias this puts into the observables is removed
// by iterated Tikhonov refinement: each step solves the regularised system
// for the physical residual, which contracts by O(eps mu_g / mu_solid).
// The solid displacement, and in the fluid the pressure p = -kappa div u,
// are observables; the fluid displacement itself is gauge.
//
// The program prints three checks of the converged solution:
//   1. the refinement residuals (their decay is the observed contraction);
//   2. the fluid pressure, which must be *uniform* for a static fluid
//      without gravity (its spread is a discretisation observable);
//   3. for the uniform load (-P2 0) the exact Lame solution with a fluid
//      core: u = gamma x in the fluid (conformal, so it is also the
//      Q-minimal gauge and the whole field can be compared).
//
// One source serves the serial and the parallel build, as in
// elastogravity_layered.cpp.
//
// Meshes (../data, from meshes/layered_earth.py): the body is every layer
// but the outer buffer shell; attribute 1 is the fluid (with the three-layer
// meshes, attributes 1 and 2 together: the merged core keeps the example
// free of a floating solid inner core, which without gravity has no
// restoring force).
//
// With -vis (the default, needs a running GLVis server) the displacement
// and the fluid pressure are shown: the pressure window is the instructive
// one — flat in the fluid, however the (gauge) displacement there looks.
//
// Options (defaults in brackets):
//   -m     mesh file [../data/elastogravity_two_layer_2d.msh]: a
//          layered mesh from meshes/layered_earth.py (attribute 1,
//          and 2 as well with three layers, is the fluid).
//   -o     finite element order [2].
//   -eps   gauge penalty factor epsilon [1e-2]: sets the O(eps) bias
//          of the penalised solve and the refinement contraction
//          O(eps mu_g / mu_solid); with 3 refinements this is the
//          recommended operating point.
//   -nref  Tikhonov refinement steps per solve [3]: each step removes
//          a factor of the O(eps) gauge-penalty bias.
//   -P0    uniform external pressure amplitude [0.01].
//   -P2    degree-2 pressure pattern amplitude [0]; 0 keeps the exact
//          Lame comparison.
//   -vis / -no-vis   GLVis windows of the displacement and the fluid
//          pressure [on].
//
// Sample runs (with mpiexec -np N in front in a parallel build):
//    ./gauged_fluid_cavity          (eps = 1e-2, 3 refinements: the
//                                    recommended operating point)
//    ./gauged_fluid_cavity -P2 0.01
//    ./gauged_fluid_cavity -m ../data/elastogravity_three_layer_3d.msh
// ============================================================================

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>

#include "AdGIA.hpp"

using namespace mfem;
using namespace AdGIA;

namespace {

#ifdef MFEM_USE_MPI
using MeshType = ParMesh;
using SubMeshType = ParSubMesh;
using SpaceType = ParFiniteElementSpace;
using FieldType = ParGridFunction;
bool Root() { return Mpi::Root(); }
double GlobalSum(double v) {
  double g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return g;
}
double GlobalMax(double v) {
  double g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return g;
}
#else
using MeshType = Mesh;
using SubMeshType = SubMesh;
using SpaceType = FiniteElementSpace;
using FieldType = GridFunction;
bool Root() { return true; }
double GlobalSum(double v) { return v; }
double GlobalMax(double v) { return v; }
#endif

// Send a field to GLVis.
void Show(Mesh& mesh, const GridFunction& f, const char* title) {
  char vishost[] = "localhost";
  socketstream sock(vishost, 19916);
  sock.precision(8);
#ifdef MFEM_USE_MPI
  sock << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() << "\n";
#endif
  sock << "solution\n"
       << mesh << f << "window_title '" << title << "'"
       << (mesh.Dimension() == 2 ? "\nkeys Rjlbc\n" : "\nkeys RRRilc\n")
       << std::flush;
}

// Non-dimensional material and load of the example.
constexpr double kKappaSolid = 2.0;
constexpr double kMuSolid = 1.0;
constexpr double kKappaFluid = 1.0;
constexpr double kRCmb = 3483.0 / 6371.0;

double P0 = 0.01;  // uniform external pressure
double P2 = 0.0;   // degree-2 pressure pattern

void PressureTraction(const Vector& x, Vector& f) {
  const double r = x.Norml2();
  const double c = (x.Size() == 2 ? x[1] : x[2]) / r;
  const double P =
      P0 + P2 * (x.Size() == 2 ? 2.0 * c * c - 1.0 : 3.0 * c * c - 1.0);
  f = x;
  f *= -P / r;
}

// Exact solution for the uniform pressure P0 with a fluid core of radius a:
// u = (alpha + beta / r^d) x in the solid shell, gamma x in the fluid.
struct LameSolution {
  int dim;
  double a, alpha, beta, gamma;

  LameSolution(int dim, double a, double b) : dim(dim), a(a) {
    DenseMatrix M(2);
    Vector rhs(2), sol(2);
    const double ad = std::pow(a, dim), bd = std::pow(b, dim);
    if (dim == 2) {
      M(0, 0) = 2.0 * (kKappaSolid - kKappaFluid);
      M(0, 1) = -2.0 * (kMuSolid + kKappaFluid) / ad;
      M(1, 0) = 2.0 * kKappaSolid;
      M(1, 1) = -2.0 * kMuSolid / bd;
    } else {
      M(0, 0) = 3.0 * (kKappaSolid - kKappaFluid);
      M(0, 1) = -(4.0 * kMuSolid + 3.0 * kKappaFluid) / ad;
      M(1, 0) = 3.0 * kKappaSolid;
      M(1, 1) = -4.0 * kMuSolid / bd;
    }
    rhs(0) = 0.0;
    rhs(1) = -P0;
    M.Invert();
    M.Mult(rhs, sol);
    alpha = sol(0);
    beta = sol(1);
    gamma = alpha + beta / ad;
  }

  void Eval(const Vector& x, Vector& u) const {
    const double r = x.Norml2();
    u = x;
    u *= r < a ? gamma : alpha + beta / std::pow(r, dim);
  }
};

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif

  const char* mesh_file = "../data/elastogravity_two_layer_2d.msh";
  int order = 2;
  double eps = 1.0e-2;
  int nref = 3;
  bool visualization = true;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&eps, "-eps", "--epsilon", "Gauge penalty factor epsilon.");
  args.AddOption(&nref, "-nref", "--refinements",
                 "Tikhonov refinement steps per solve.");
  args.AddOption(&P0, "-P0", "--pressure", "Uniform external pressure.");
  args.AddOption(&P2, "-P2", "--pressure-degree2",
                 "Degree-2 external pressure pattern (0 keeps the exact "
                 "Lame comparison).");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "GLVis visualisation of the displacement and the fluid "
                 "pressure.");
  args.Parse();
  if (!args.Good()) {
    if (Root()) {
      args.PrintUsage(std::cout);
    }
    return 1;
  }

  // The body: every layer but the buffer, on one SubMesh; attribute 1 (and
  // 2 with an inner core) is the fluid, the last layer the solid mantle.
  Mesh smesh(mesh_file, 1, 1);
  const int dim = smesh.Dimension();
#ifdef MFEM_USE_MPI
  MeshType mesh(MPI_COMM_WORLD, smesh);
  smesh.Clear();
#else
  MeshType& mesh = smesh;
#endif
  const int n_layers = mesh.attributes.Max() - 1;
  Array<int> body_attrs(n_layers);
  for (int i = 0; i < n_layers; i++) {
    body_attrs[i] = i + 1;
  }
  auto body = SubMeshType::CreateFromDomain(mesh, body_attrs);
  H1_FECollection fec(order, dim);
  SpaceType fes(&body, &fec, dim);

  // Piecewise material: the fluid has its bulk modulus and NO shear; the
  // gauge penalty is not part of the rheology.
  Vector kappa_vals(n_layers), mu_vals(n_layers);
  Array<int> fluid(n_layers);
  kappa_vals = kKappaFluid;
  mu_vals = 0.0;
  fluid = 1;
  kappa_vals(n_layers - 1) = kKappaSolid;
  mu_vals(n_layers - 1) = kMuSolid;
  fluid[n_layers - 1] = 0;
  PWConstCoefficient kappa(kappa_vals), mu(mu_vals);
  IsotropicElasticRheology rheology(dim, kappa, mu);

  // Pressure load on the surface (the SubMesh's largest boundary
  // attribute).
  Array<int> surface(body.bdr_attributes.Max());
  surface = 0;
  surface[body.bdr_attributes.Max() - 1] = 1;
  VectorFunctionCoefficient traction(dim, PressureTraction);

  LinearQuasiStaticTractionProblem problem(&fes, rheology, traction, surface);

  // The gauged fluid: mu_g of the order of the fluid's own modulus, and the
  // mass-weighted rigid gauge (the exact solution has zero net momentum).
  ConstantCoefficient mu_gauge(kKappaFluid);
  problem.SetFluid(fluid, mu_gauge, GaugePenaltyOptions{eps, nref});
  problem.SetMassWeightedGauge();

  problem.AssembleForce(0.0);
  if (!problem.Solve()) {
    if (Root()) {
      std::cout << "solver did not converge\n";
    }
    return 1;
  }
  const auto& u = problem.Displacement();

  if (Root()) {
    std::cout << "gauged fluid: eps = " << eps << ", " << nref
              << " refinement steps\n"
              << "refinement residuals |eps Q delta| (decay = observed "
                 "contraction):\n  ";
    for (auto r : problem.GaugeResiduals()) {
      std::cout << std::setprecision(3) << r << "  ";
    }
    std::cout << "\n";
  }

  // The fluid pressure p = -kappa div u must be uniform (no gravity): its
  // spread about the mean is pure discretisation error, and it is a
  // gauge-invariant observable of the fluid.
  {
    DivergenceGridFunctionCoefficient div_u(&u);
    double p_int = 0.0, vol = 0.0;
    double p_max = -std::numeric_limits<double>::infinity();
    double p_min = std::numeric_limits<double>::infinity();
    for (int i = 0; i < body.GetNE(); i++) {
      if (!fluid[body.GetAttribute(i) - 1]) {
        continue;
      }
      auto* T = body.GetElementTransformation(i);
      const auto& ir = IntRules.Get(body.GetElementGeometry(i), 2 * order);
      for (int q = 0; q < ir.GetNPoints(); q++) {
        const auto& ip = ir.IntPoint(q);
        T->SetIntPoint(&ip);
        const double w = ip.weight * T->Weight();
        const double p = -kKappaFluid * div_u.Eval(*T, ip);
        p_int += w * p;
        vol += w;
        p_max = std::max(p_max, p);
        p_min = std::min(p_min, p);
      }
    }
    p_int = GlobalSum(p_int);
    vol = GlobalSum(vol);
    p_max = GlobalMax(p_max);
    p_min = -GlobalMax(-p_min);
    const double p_mean = p_int / vol;
    if (Root()) {
      std::cout << "fluid pressure: mean " << std::setprecision(6) << p_mean
                << ", relative spread "
                << (p_max - p_min) / std::abs(p_mean)
                << " (uniform in the continuum)\n";
    }
  }

  // Exact Lame comparison for the uniform load.
  if (P2 == 0.0) {
    LameSolution lame(dim, kRCmb, 1.0);
    VectorFunctionCoefficient exact(
        dim, [&lame](const Vector& x, Vector& u) { lame.Eval(x, u); });
    Vector zero(dim);
    zero = 0.0;
    VectorConstantCoefficient z(zero);
    auto& u_mut = const_cast<GridFunction&>(u);
    const double err = u_mut.ComputeL2Error(exact) / u_mut.ComputeL2Error(z);
    if (Root()) {
      std::cout << "relative L2 error against the exact Lame solution: "
                << std::setprecision(3) << err << "\n";
    }
  }

  if (visualization) {
    Show(body, u, "Displacement");
    // p = -kappa div u on an L2 space, zero outside the fluid: uniform in
    // the continuum, and gauge-invariant where the displacement is not.
    L2_FECollection pfec(order - 1, dim);
    SpaceType pfes(&body, &pfec);
    FieldType p_gf(&pfes);
    DivergenceGridFunctionCoefficient div_u(&u);
    ProductCoefficient minus_kappa_div(-kKappaFluid, div_u);
    p_gf.ProjectCoefficient(minus_kappa_div);
    Array<int> dofs;
    for (int i = 0; i < body.GetNE(); i++) {
      if (!fluid[body.GetAttribute(i) - 1]) {
        pfes.GetElementDofs(i, dofs);
        p_gf.SetSubVector(dofs, 0.0);
      }
    }
    Show(body, p_gf, "Fluid pressure");
  }

  return 0;
}
