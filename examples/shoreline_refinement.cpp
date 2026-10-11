// ============================================================================
// shoreline_refinement.cpp
//
// IN-C++ SHORELINE MESH REFINEMENT (RefineToShoreline, mesh.hpp): the
// MFEM `shaper` miniapp's pattern specialised to the sea-level problem.
// The ocean function drives conforming local refinement of the curved
// ball mesh toward the shoreline — a band about the flotation
// threshold, so migrating shorelines stay inside the refined region —
// with new nodes snapped back onto the bounding spheres. No gmsh round
// trip: the band follows whatever ocean function the run itself uses.
//
// The geography is analytic and deliberately non-axisymmetric: a polar
// continent whose super-Gaussian width is modulated in longitude, so
// the shoreline is a genuinely 2-D curve the refiner must chase. After
// refinement the example solves the ELASTIC sea-level fingerprint of a
// melting polar ice blob on the refined mesh, exporting
//
//   shoreline_h.csv   the surface resolution map (local element size:
//                     the refined ring is the money picture);
//   shoreline_sl.csv  the fingerprint on the refined mesh;
//   shoreline_refined.mesh  the refined mesh (GLVis / reuse).
//
// IMPORTANT ordering: conforming tet bisection needs the loaded mesh's
// refinement marking, which UniformRefinement clears — so shoreline
// refinement runs FIRST and -r uniform refinements compose after.
//
// Maps:  <build>/earth_data/plot_fingerprint_kernel shoreline_h.csv \
//            --no-mask --global-only --cbar-label "element size"
//
// One source serves the serial and the parallel build.
//
// Options (defaults in brackets):
//   -m     ball-in-buffer mesh [../data/coupled_poisson.msh].
//   -o     order [2];  -r  uniform refinements AFTER shaping [0].
//   -l     shoreline refinement passes [2];  -sd  sampling subdivisions
//          per element [2];  -band  marked band about the flotation
//          threshold, in ocean-function units [0.2].
//   -width target half-width of the refined zone, radians of arc on
//          the surface [0: off]. When set it replaces -band: elements
//          within this angular distance of the shoreline are marked,
//          independently of how steep the ocean function is at the
//          coast. NOTE: -width/-band set the zone's EXTENT; how fine
//          its elements get is -hs (or the pass count -l). A width
//          below the current element size cannot show.
//   -hs    target element size at the shoreline [0: off]. When set,
//          passes continue until the shoreline elements are at or
//          below this size (-l becomes a safety cap and is raised to
//          12 if left at its default) — the resolution knob.
//   -cont-width  continent super-Gaussian width (rad) [0.9].
//   -wave        longitudinal width modulation amplitude (rad) [0.15].
//   -blob-width  ice-blob Gaussian width (rad) [0.35].
//   -rhow / -rhoi  water and ice densities [0.05, 0.045].
//   -rt    solver tolerance [1e-10];  -vis / -no-vis  GLVis [on].
//
// Sample runs:
//   ./shoreline_refinement -no-vis
//   ./shoreline_refinement -l 3 -no-vis
//   mpiexec -np 4 ./shoreline_refinement -l 3 -no-vis
// ============================================================================

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <numbers>
#include <string>

#include "mfem.hpp"
#include "AdGIA.hpp"
#include "AdGIA/detail/fem_factory.hpp"
#include "visualisation.hpp"

using namespace mfem;
using namespace AdGIA;
using namespace examples;

namespace {

constexpr double kG = 0.05;
constexpr double kRho = 1.0;
constexpr double kKappa = 1.0;
constexpr double kMu = 0.5;
constexpr int kDtNDegree = 12;

double GlobalSum(double v) {
#ifdef MFEM_USE_MPI
  double g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return g;
#else
  return v;
#endif
}

double Theta(const Vector& x) {
  const double r = x.Norml2();
  return std::acos(std::min(1.0, std::max(-1.0, x[2] / r)));
}

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif

  const char* mesh_file = "../data/coupled_poisson.msh";
  int order = 2, ref_levels = 0, levels = 2, subdivisions = 2;
  double band = 0.2, width = 0.0, target_h = 0.0;
  double cont_width = 0.9, wave = 0.15, blob_width = 0.35;
  double rho_w = 0.05, rho_i = 0.045;
  double rel_tol = 1e-10;
  bool visualization = true;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Ball-in-buffer mesh.");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&ref_levels, "-r", "--refine",
                 "Uniform refinements AFTER shoreline refinement.");
  args.AddOption(&levels, "-l", "--levels", "Shoreline refinement passes.");
  args.AddOption(&subdivisions, "-sd", "--sub-divisions",
                 "Element sampling subdivisions.");
  args.AddOption(&band, "-band", "--band",
                 "Marked band about the flotation threshold.");
  args.AddOption(&width, "-width", "--width",
                 "Target angular half-width of the refined zone (rad; "
                 "0 keeps the -band semantics).");
  args.AddOption(&target_h, "-hs", "--shoreline-size",
                 "Target element size at the shoreline (0: off; -l "
                 "becomes a safety cap).");
  args.AddOption(&cont_width, "-cont-width", "--continent-width",
                 "Continent super-Gaussian width (rad).");
  args.AddOption(&wave, "-wave", "--wave",
                 "Longitudinal width modulation (rad).");
  args.AddOption(&blob_width, "-blob-width", "--blob-width",
                 "Ice-blob Gaussian width (rad).");
  args.AddOption(&rho_w, "-rhow", "--water-density", "Water density.");
  args.AddOption(&rho_i, "-rhoi", "--ice-density", "Ice density.");
  args.AddOption(&rel_tol, "-rt", "--solver-tolerance",
                 "Relative solver tolerance.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization", "GLVis windows.");
  args.Parse();
  if (!args.Good()) {
    if (IsRoot()) {
      args.PrintUsage(std::cout);
    }
    return 1;
  }
  if (IsRoot()) {
    args.PrintOptions(std::cout);
  }

  // The analytic geography: wavy polar continent, ocean elsewhere.
  const double cw = cont_width, wv = wave, bw = blob_width;
  auto continent = [cw, wv](const Vector& x) {
    const double ph = std::atan2(x[1], x[0]);
    const double q = Theta(x) / (cw + wv * std::sin(3.0 * ph));
    return std::exp(-q * q * q * q);
  };
  auto ocean = [continent](const Vector& x) {
    return 1.0 - continent(x);
  };
  FunctionCoefficient ocean_c(ocean);

  Mesh smesh(mesh_file, 1, 1);
  MFEM_VERIFY(smesh.Dimension() == 3, "a 3-D ball mesh is required");
#ifdef MFEM_USE_MPI
  ParMesh parent(MPI_COMM_WORLD, smesh);
  smesh.Clear();
#else
  Mesh& parent = smesh;
#endif

  // The shaping pass: one level at a time for the pass table. The outer
  // Poisson sphere stays snapped too (closure can reach it).
  ShorelineRefinementOptions opt;
  opt.levels = 1;
  opt.subdivisions = subdivisions;
  opt.band = band;
  opt.width = width;
  opt.target_h = target_h;
  if (target_h > 0.0 && levels == 2) {
    levels = 12;  // the default cap would stop short of most targets
  }
  {
    SphericalMeshHelper outer;
    outer.SetBoundaryMarker(&parent);
    opt.snap_radii = {1.0, outer.bdr_radius_};
  }
  if (IsRoot()) {
    if (target_h > 0.0) {
      std::cout << "\nshoreline refinement (target element size "
                << target_h << ", cap " << levels << " passes):\n";
    } else if (width > 0.0) {
      std::cout << "\nshoreline refinement (target width " << width
                << " rad, " << subdivisions << " subdivisions):\n";
    } else {
      std::cout << "\nshoreline refinement (band " << band << ", "
                << subdivisions << " subdivisions):\n";
    }
  }
  for (int l = 0; l < levels; l++) {
    const int refined = RefineToShoreline(parent, ocean_c, opt);
    long long ne = parent.GetNE();
#ifdef MFEM_USE_MPI
    ne = parent.GetGlobalNE();
#endif
    if (IsRoot()) {
      std::cout << "  pass " << l + 1 << ": refined " << refined
                << " elements -> " << ne << "\n";
    }
    if (refined == 0) {
      break;
    }
  }
  for (int l = 0; l < ref_levels; l++) {
    parent.UniformRefinement();
  }

  // The refined mesh for GLVis and reuse.
  {
#ifdef MFEM_USE_MPI
    std::ofstream mf(IsRoot() ? "shoreline_refined.mesh" : "/dev/null");
    mf.precision(16);
    parent.PrintAsOne(mf);
#else
    std::ofstream mf("shoreline_refined.mesh");
    mf.precision(16);
    parent.Print(mf);
#endif
  }

  // The sea-level stack on the refined mesh.
  Array<int> body_attrs_list;
  body_attrs_list.Append(1);
  const int dim = 3;
  H1_FECollection fec(order, dim);
#ifdef MFEM_USE_MPI
  ParSubMesh body(ParSubMesh::CreateFromDomain(parent, body_attrs_list));
  ParFiniteElementSpace fes_u(&body, &fec, dim), fes_phi(&parent, &fec);
#else
  SubMesh body(SubMesh::CreateFromDomain(parent, body_attrs_list));
  FiniteElementSpace fes_u(&body, &fec, dim), fes_phi(&parent, &fec);
#endif
  Array<int> surface(body.bdr_attributes.Max());
  surface = 0;
  surface[body.bdr_attributes.Max() - 1] = 1;
  SeaLevelOperator sea(fes_u, surface);
  sea.SetDensities(rho_w, rho_i);

  // The resolution map: per surface node, the smallest adjacent surface
  // element's size — the refined ring made visible.
  {
    Mesh& smesh_s = sea.SurfaceMesh();
    GridFunction h(&sea.SurfaceSpace());
    h = std::numeric_limits<real_t>::max();
    Array<int> dofs;
    for (int e = 0; e < smesh_s.GetNE(); e++) {
      const double he = smesh_s.GetElementSize(e);
      sea.SurfaceSpace().GetElementDofs(e, dofs);
      for (int j = 0; j < dofs.Size(); j++) {
        h[dofs[j]] = std::min(h[dofs[j]], he);
      }
    }
    sea.WriteSurfaceField(h, "shoreline_h.csv");
  }

  // The elastic fingerprint of the fully melted polar blob on the
  // refined mesh (frozen shorelines; gia_kernels' load at t = t1).
  const double g0 = 4.0 * std::numbers::pi * kG * kRho / 3.0;
  FunctionCoefficient w_ocean(
      [ocean, g0, rho_w](const Vector& x) {
        return rho_w * ocean(x) / g0;
      });
  FunctionCoefficient sigma_d(
      [ocean, bw, rho_i](const Vector& x) {
        const double q = Theta(x) / bw;
        return -rho_i * (1.0 - ocean(x)) * std::exp(-q * q);
      });
  VectorFunctionCoefficient grad_phi0(dim,
                                      [g0](const Vector& x, Vector& v) {
                                        v = x;
                                        v *= g0;
                                      });
  ConstantCoefficient kappa(kKappa), mu(kMu), rho(kRho);
  auto rheology = IsotropicElasticRheology(dim, kappa, mu);
  LinearQuasiStaticMixedSelfGravitatingProblem prob(&fes_u, &fes_phi,
                                                    rheology, rho, kG,
                                                    kDtNDegree);
  prob.SetWaterLoad(w_ocean, sigma_d, surface);
  prob.SetRelTol(rel_tol);
  prob.AssembleForce(0.0);
  if (!prob.Solve()) {
    if (IsRoot()) {
      std::cout << "solve failed\n";
    }
    return 1;
  }
  sea.SeaLevelChangeFrom(prob.Displacement(), grad_phi0, prob.Potential(),
                         prob.UniformPotentialTerm(), nullptr);
  sea.WriteSurfaceField(sea.SeaLevelChangeField(), "shoreline_sl.csv");

  if (IsRoot()) {
    std::cout << "\nwrote shoreline_h.csv (resolution ring), "
                 "shoreline_sl.csv (fingerprint),\n"
                 "shoreline_refined.mesh (glvis -m ...).\n"
                 "maps: <build>/earth_data/plot_fingerprint_kernel "
                 "shoreline_h.csv --no-mask \\\n"
                 "          --global-only --cbar-label \"element size\" "
                 "--out shoreline_h.png\n";
  }

  if (visualization) {
    GLVisWindow wh("surface resolution", DefaultKeys(2));
    GridFunction h(&sea.SurfaceSpace());
    {
      Mesh& smesh_s = sea.SurfaceMesh();
      h = std::numeric_limits<real_t>::max();
      Array<int> dofs;
      for (int e = 0; e < smesh_s.GetNE(); e++) {
        const double he = smesh_s.GetElementSize(e);
        sea.SurfaceSpace().GetElementDofs(e, dofs);
        for (int j = 0; j < dofs.Size(); j++) {
          h[dofs[j]] = std::min(h[dofs[j]], he);
        }
      }
    }
    wh.Send(sea.SurfaceMesh(), h);
    GLVisWindow ws("sea-level change on the refined mesh",
                   DefaultKeys(2));
    ws.Send(sea.SurfaceMesh(), sea.SeaLevelChangeField());
  }
  return 0;
}
