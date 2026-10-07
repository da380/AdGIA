// ============================================================================
// coseismic_deformation.cpp
//
// Static coseismic deformation from a moment-tensor source: linear
// elasticity on a Cartesian box with a free top surface, no gravity, and
// the earthquake entered as a stress glut — the linear form
//   b(v) = sum_p  m0_p  (u n^T + n u^T) : grad v (x_p)
// assembled by AdGIA::DomainLFDeformationGradientIntegrator from one
// MatrixDeltaCoefficient per fault patch (MFEM's delta machinery locates
// each centre; on a ParMesh exactly one rank assembles each point). The
// planar fault is split into patches, each a point source at its centre
// with moment  m0 = mu(x_p) * slip * dA  from the LOCAL shear modulus,
// so the elastic structure changes the source as well as the medium; a
// single patch is a point source carrying the whole fault's moment. In
// 2-D the slip is pure dip-slip, its sign the sense; in 3-D the rake
// sets the slip direction in the fault plane. The box, the fault and the
// elastic structure live in fault_box.hpp, shared with
// postseismic_deformation.cpp.
//
// The box is x in [-W/2, W/2] (3-D: also y), z in [-D, 0]: the top is the
// free surface and the other boundaries are artificial, clamped (u = 0)
// and meshed far enough away that they do not matter visually — the
// static field of a moment tensor decays algebraically, so keep the
// fault dimensions and depth small against W and D. Everything is
// nondimensional (mu0 = lambda0 = 1); the displacement is linear in the
// slip, so its size is a free choice. The elastic structure scales both
// moduli by one factor (fixed Poisson ratio): depth stiffening, a smooth
// lateral gradient, and a soft Gaussian basin; the topography is mapped
// directly onto the Cartesian mesh nodes by blending
// z -> z + h(x,y)(1 + z/D), so the bottom stays flat and element faces
// follow the surface — no remeshing.
//
// The arguments (defaults in brackets; -h prints the same list):
//   -d       space dimension, 2 or 3                           [2]
//   -o       finite element order                              [2]
//   -nx      elements across the width (3-D: each horizontal
//            direction)                                        [64 / 24]
//   -nz      elements through the depth                        [32 / 12]
//   -W       width of the box                                  [8]
//   -D       depth of the box                                  [4]
// the fault:
//   -dip     dip in degrees from horizontal, towards +x        [30]
//   -rake    rake in degrees, anticlockwise in the fault plane
//            from the strike: 90 thrust, 0 left-lateral
//            strike-slip (3-D; 2-D is pure dip-slip)           [90]
//   -strike  strike in degrees clockwise from the y axis (3-D) [0]
//   -fd      depth of the fault centre below z = 0             [1]
//   -fx      x of the fault centre                             [0]
//   -fl      along-dip length                                  [1.5]
//   -fw      along-strike width (3-D)                          [1.5]
//   -s       slip; in 2-D > 0 thrust, < 0 normal; the PEAK
//            value when tapered                                [0.01]
//   -taper / -no-taper  cos^2 slip taper, zero at the fault
//            ends (3-D: all edges) — a uniform distribution's
//            abrupt edges are themselves an artifact. Midpoint
//            patches leave -nd 1 a full point source           [on]
//   -nd      point-source patches along dip                    [12]
//   -ns      point-source patches along strike (3-D); with
//            -nd 1 -ns 1 the fault is a single point source    [12]
//   -smooth  replace each point source by a normalised Gaussian
//            of this sigma in ELEMENT widths: same moment, same
//            far field to O(sigma^2/r^2), but a resolved density
//            instead of mesh-scale artifacts near the fault; the
//            printed capture is < 100% when a tail leaves the
//            body (0: point sources)                           [0]
// the elastic structure (all off by default):
//   -beta    depth stiffening, moduli *= 1 + beta |z| / D      [0]
//   -alpha   lateral gradient, moduli *= 1 + alpha x / W
//            (|alpha| < 2 keeps them positive)                 [0]
//   -gamma   Gaussian basin: moduli *= 1 - gamma exp(-r^2 /
//            2 br^2), gamma < 1 the softening at its centre    [0]
//   -bx      x of the basin centre                             [0]
//   -bz      depth of the basin centre below z = 0             [0]
//   -br      Gaussian radius of the basin                      [1]
// the topography:
//   -topo    height of the Gaussian surface topography         [0]
//   -tx      x of the topography centre                        [0]
//   -tr      Gaussian radius of the topography                 [1.5]
//   -hill    asymmetric hill: right of the crest a smooth,
//            roughly linear ramp keeping this fraction of the
//            height at the right edge (negative: symmetric)    [-1]
// the output:
//   -vis / -no-vis   u_z, the vector displacement, kappa and
//                    mu in GLVis                               [on]
//   -csv     surface-profile table ("": off)
//                                       [coseismic_deformation.csv]
//
// Output: the total scalar moment and the peak surface uplift and
// subsidence on the screen; coseismic_deformation.csv, the surface
// displacement profile along the x axis (3-D: at y = 0), via
// FindPoints/GetVectorValue (python3 plot_csv.py coseismic_deformation.csv);
// with -vis (the default), four GLVis windows: the SCALAR u_z, the
// colour scale fixed from the surface peaks — the field saturates near
// the source, where the delta's 1/r^(d-1) singularity is resolved only
// to the mesh scale —, the VECTOR displacement, autoscaled, mesh off,
// black arrows scaled to the mesh spacing (v cycles the arrow modes; in
// 2-D, d displaces the mesh by the field), and the moduli
// kappa = lambda + 2 mu / dim and mu, to see the elastic structure.
//
// One source serves the serial and the parallel build; the genuine
// differences are the mesh partitioning (the mesh is built and mapped
// serially first), the preconditioner (Gauss-Seidel serially, BoomerAMG
// in parallel) and the reduction of the sampled profile, each point of
// which is found on exactly one rank (ParMesh::FindPoints).
//
// Sample runs (with mpiexec -np N in front in a parallel build):
//    ./coseismic_deformation
//    ./coseismic_deformation -dip 60 -s -0.01            (normal fault)
//    ./coseismic_deformation -nd 1 -ns 1                 (point source)
//    ./coseismic_deformation -smooth 1.5                 (Gaussian sources)
//    ./coseismic_deformation -beta 2 -gamma 0.5 -topo 0.3
//    ./coseismic_deformation -alpha 1 -bz 0.5 -gamma 0.6 -bx -1
//    ./coseismic_deformation -d 3 -o 1 -rake 0           (strike-slip)
//    ./coseismic_deformation -d 3 -fd 0.75 -fl 1 -fw 2 -strike 30
// ============================================================================

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "AdGIA.hpp"
#include "fault_box.hpp"
#include "visualisation.hpp"

using namespace std;
using namespace mfem;
using namespace AdGIA;

namespace {

#ifdef MFEM_USE_MPI
using MeshType = ParMesh;
using SpaceType = ParFiniteElementSpace;
using FieldType = ParGridFunction;
using FormType = ParBilinearForm;
using LFType = ParLinearForm;
bool Root() { return Mpi::Root(); }
#else
using MeshType = Mesh;
using SpaceType = FiniteElementSpace;
using FieldType = GridFunction;
using FormType = BilinearForm;
using LFType = LinearForm;
bool Root() { return true; }
#endif

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif

  // Set the default options.
  int dim = 2;
  int order = 2;
  examples::FaultBox box;
  examples::Fault fault;
  bool visualization = true;
  const char* csv_file = "coseismic_deformation.csv";

  // Read in command line options and process.
  OptionsParser args(argc, argv);
  args.AddOption(&dim, "-d", "--dimension", "Space dimension (2 or 3).");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order for the displacement.");
  box.AddOptions(args);
  fault.AddOptions(args);
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Show u_z, the vector displacement, kappa and mu in GLVis.");
  args.AddOption(&csv_file, "-csv", "--csv",
                 "Table of the surface profile for plot_csv.py (\"\": none).");
  args.Parse();
  if (!args.Good()) {
    if (Root()) {
      args.PrintUsage(cout);
    }
    return 1;
  }
  if (Root()) {
    args.PrintOptions(cout);
  }
  MFEM_VERIFY(dim == 2 || dim == 3, "Dimension must be 2 or 3.");
  box.Finalize(dim);

  // Build the box mesh, mapped and centred (on the serial mesh, before
  // partitioning). The top is the free surface; everything else is
  // artificial and clamped.
  Mesh smesh = box.MakeMesh(dim);
  const int top_attr = box.TopAttribute(dim);
#ifdef MFEM_USE_MPI
  MeshType mesh(MPI_COMM_WORLD, smesh);
  smesh.Clear();
#else
  MeshType& mesh = smesh;
#endif

  // Displacement space.
  H1_FECollection fec(order, dim);
  SpaceType fes(&mesh, &fec, dim);
#ifdef MFEM_USE_MPI
  const auto n_u = fes.GlobalTrueVSize();
#else
  const auto n_u = fes.GetTrueVSize();
#endif
  if (Root()) {
    cout << "Displacement unknowns: " << n_u << "\n";
  }

  // Material: isotropic, both moduli scaled by the same factor (fixed
  // Poisson ratio). Under topography z > 0 is possible and the depth
  // factor drops slightly below 1 there.
  const real_t lambda0 = 1.0, mu0 = 1.0;
  auto modulus_factor = box.ModulusFactor(dim);
  FunctionCoefficient lambda_c(
      [=](const Vector& x) { return lambda0 * modulus_factor(x); });
  FunctionCoefficient mu_c(
      [=](const Vector& x) { return mu0 * modulus_factor(x); });

  // The fault as point-source patches, each with the local shear modulus
  // in its moment.
  auto sources = fault.Build(
      dim, box, [&](const Vector& xp) { return mu0 * modulus_factor(xp); });
  if (Root()) {
    cout << "Fault patches: " << fault.nd * (dim == 2 ? 1 : fault.ns)
         << (sources.smoothed ? " (smoothed)" : "")
         << ", total scalar moment M0 = " << sources.moment
         << (dim == 2 ? " (per unit strike length)" : "") << "\n";
  }

  // The stress-glut right-hand side: one point source per patch, or the
  // single smoothed density with a rule fine enough for a Gaussian of
  // about an element width. All ranks add every source; each delta
  // centre is assembled on exactly one rank (ParMesh::FindPoints inside
  // Assemble is collective).
  LFType b(&fes);
  if (sources.smoothed) {
    b.AddDomainIntegrator(new DomainLFDeformationGradientIntegrator(
        *sources.smoothed,
        &IntRules.Get(dim == 2 ? Geometry::SQUARE : Geometry::CUBE,
                      2 * order + 6)));
  } else {
    for (auto& src : sources.patches) {
      b.AddDomainIntegrator(new DomainLFDeformationGradientIntegrator(*src));
    }
  }
  b.Assemble();
  if (sources.smoothed) {
    const double captured = examples::MomentCapture(
        b, fes, sources.couple, sources.moment_signed);
    if (Root()) {
      cout << "Smoothed sources: sigma = " << sources.sigma << " ("
           << fault.smooth << " element widths), moment captured: "
           << 100.0 * captured << "%\n";
    }
  }

  // Clamp every boundary but the free surface, and solve. The one solver
  // difference: Gauss-Seidel serially, BoomerAMG in parallel.
  Array<int> ess_bdr(mesh.bdr_attributes.Max()), ess_tdof_list;
  ess_bdr = 1;
  ess_bdr[top_attr - 1] = 0;
  fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

  FormType a(&fes);
  a.AddDomainIntegrator(new ElasticityIntegrator(lambda_c, mu_c));
  a.Assemble();
  FieldType u(&fes);
  u = 0.0;
  OperatorPtr Amat;
  Vector X, B;
  a.FormLinearSystem(ess_tdof_list, u, b, Amat, X, B);
#ifdef MFEM_USE_MPI
  HypreBoomerAMG prec(*Amat.As<HypreParMatrix>());
  prec.SetPrintLevel(0);
  prec.SetSystemsOptions(dim);
  CGSolver cg(MPI_COMM_WORLD);
#else
  GSSmoother prec(*Amat.As<SparseMatrix>());
  CGSolver cg;
#endif
  cg.SetPreconditioner(prec);
  cg.SetOperator(*Amat);
  cg.SetRelTol(1e-10);
  cg.SetMaxIter(10000);
  cg.SetPrintLevel(IterativeSolver::PrintLevel().Summary());
  cg.Mult(B, X);
  a.RecoverFEMSolution(X, b, u);

  // Sample the displacement along the free surface.
  examples::SurfaceProfile profile(mesh, box, dim);
  const auto values = profile.Sample(u);
  const int npts = profile.NumPoints();

  // u_z as a scalar field (the H1 vertical component, interpolated
  // nodally into the scalar space of the same order: exact), for the
  // surface peaks and the plot.
  SpaceType fes_z(&mesh, &fec);
  FieldType u_z(&fes_z);
  VectorGridFunctionCoefficient u_c(&u);
  Vector e_z(dim);
  e_z = 0.0;
  e_z(dim - 1) = 1.0;
  VectorConstantCoefficient e_z_c(e_z);
  InnerProductCoefficient u_z_c(u_c, e_z_c);
  u_z.ProjectCoefficient(u_z_c);
  const auto [uz_min, uz_max] = examples::SurfacePeaks(u_z, fes_z, top_attr);
  if (Root()) {
    cout << "Peak surface uplift:     " << uz_max
         << "\nPeak surface subsidence: " << uz_min << "\n";
  }

  vector<string> cols = dim == 2 ? vector<string>{"x", "u_x", "u_z"}
                                 : vector<string>{"x", "u_x", "u_y", "u_z"};
  examples::CsvTable table(csv_file, cols);
  table.Meta("title", "Coseismic surface displacement")
      .Meta("note", dim == 2 ? "profile along the free surface"
                             : "profile along the free surface at y = 0")
      .Meta("xlabel", "x")
      .Meta("ylabel", "u");
  for (int i = 0; i < npts; i++) {
    vector<double> row{profile.X(i)};
    for (int c = 0; c < dim; c++) {
      row.push_back(values[c * npts + i]);
    }
    table.Row(row);
  }
  table.Write();

  // Show u_z, the colour scale fixed from the surface peaks — the source
  // region saturates by design — and the moduli kappa = lambda + 2 mu /
  // dim and mu, to see the elastic structure (one factor scales both, so
  // the two maps share their pattern at a fixed ratio).
  if (visualization) {
    examples::GLVisWindow window("u_z (coseismic)", examples::DefaultKeys(dim));
    const real_t span =
        std::max({std::abs(uz_min), std::abs(uz_max), real_t{1e-30}});
    window.SetValueRange(-1.2 * span, 1.2 * span);
    window.Send(mesh, u_z);

    // The vector displacement itself, autoscaled, with its own keys: no
    // 'b', which the 2-D vector scene rebinds from the boundary to the
    // displaced-mesh animation, and 'vvvvv' for arrow mode 5 — length
    // proportional to the field, capped at the mesh spacing, drawn BLACK
    // on top (the coloured modes vanish against the field below).
    // Further 'v' presses cycle the arrow modes; in 2-D, 'd' displaces
    // the mesh by the field.
    examples::GLVisWindow vec_window(
        "displacement", dim == 2 ? "Rjlcvvvvv" : "RRRilcvvvvv");
    vec_window.Send(mesh, u);

    FunctionCoefficient kappa_c([=](const Vector& x) {
      return (lambda0 + 2.0 * mu0 / dim) * modulus_factor(x);
    });
    FieldType kappa_gf(&fes_z), mu_gf(&fes_z);
    kappa_gf.ProjectCoefficient(kappa_c);
    mu_gf.ProjectCoefficient(mu_c);
    examples::GLVisWindow kappa_window("kappa", examples::DefaultKeys(dim));
    kappa_window.Send(mesh, kappa_gf);
    examples::GLVisWindow mu_window("mu", examples::DefaultKeys(dim));
    mu_window.Send(mesh, mu_gf);
  }
  return 0;
}
