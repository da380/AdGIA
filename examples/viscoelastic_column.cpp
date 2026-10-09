// ============================================================================
// viscoelastic_column.cpp
//
// Why material interfaces belong on element faces: a two-layer Maxwell
// column with an exact solution, run twice, once with the interface on an
// element face and once with it cutting through an element.
//
// The column [0, W] x [0, H] (3-D: [0, W]^2 x [0, H]) is periodic sideways,
// clamped at the bottom and loaded on top by a uniform downward traction p0
// switched on at t = 0 and held. Both layers share kappa and mu; they
// differ only in the Maxwell time (tau_bottom below z_i, tau_top above).
// By symmetry the solution is uniaxial strain, uniform within each layer:
//
//   u_z(z, t) = -p0 [ min(z, z_i) c_b(t) + max(z - z_i, 0) c_t(t) ],
//
// with c the constrained creep compliance of a Maxwell layer (one
// exponential, from the elastic 1/M to the relaxed 1/kappa):
//
//   c(t) = 1/kappa - (M - kappa) / (kappa M) exp(-t kappa / (M tau)),
//   M = kappa + 2 (1 - 1/dim) mu
//
// (the library's 2-D continuum has the 2-D deviator, so M = kappa + mu in
// 2-D and kappa + 4 mu / 3 in 3-D). The displacement is piecewise linear in
// z with a kink at z_i:
//   - interface ON a face: the kink lies between elements, the finite-
//     element space holds the solution exactly, and the only error left is
//     the time stepping's (second order, tiny);
//   - interface INSIDE an element: no polynomial on that element can make
//     the kink, and the error shrinks only like the element size (order 2:
//     halved per halving of the elements; order 3 starts lower but
//     converges more slowly still). It is invisible at t = 0+, where the
//     layers are elastically identical and there is no kink; it peaks
//     while the layers are in different states (the top relaxed, the
//     bottom not) and fades as the bottom catches up and the kink goes.
// The rheology is the same in both runs: the Maxwell time is evaluated
// pointwise from z, so the cut run's quadrature points know exactly where
// the interface is. What fails is the displacement space, not the material
// description. (A CompositeRheology by element attribute fares worse on a
// cutting mesh: it moves the interface to the element boundary.)
//
// Outputs: the table of the top displacement on the screen;
// viscoelastic_column.csv, both histories with their exact values and
// their errors (python3 plot_csv.py viscoelastic_column.csv); with -vis, the
// error u_z - exact at the final time for both runs, in GLVis, on one
// common colour scale.
//
// One source serves the serial and the parallel build; the genuine
// difference is the partitioning of the (periodic) mesh.
//
// Options (defaults in brackets):
//   -d      space dimension, 2 or 3 [2].
//   -o      finite element order [2]; higher order does not cure
//           the kink in the cutting run.
//   -nx     elements across the (periodic) width, >= 3 [3].
//   -nz     elements up the column, even so that z = H/2 is a face
//           [4]; finer, the cut run's error shrinks like 1/nz.
//   -kappa  bulk modulus [1].
//   -mu     shear modulus, both layers [1].
//   -tb     Maxwell time of the bottom layer, the time unit [1].
//   -tt     Maxwell time of the top layer [0.1].
//   -p0     traction on the top [0.01].
//   -tf     final time [5].
//   -n      time steps per unit time [16].
//   -vis / -no-vis   final errors in GLVis on or off [on].
//   -csv    table of the histories for plot_csv.py
//           [viscoelastic_column.csv]; "" writes none.
//
// Sample runs (with mpiexec -np N in front in a parallel build):
//    ./viscoelastic_column
//    ./viscoelastic_column -nz 16       (finer: the cut error shrinks ~1/nz)
//    ./viscoelastic_column -o 3         (higher order does not cure a kink)
//    ./viscoelastic_column -n 4         (coarser steps: the face run's error
//                                        is the time stepping's)
//    ./viscoelastic_column -d 3 -nz 4
// ============================================================================
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "AdGIA.hpp"
#include "visualisation.hpp"

using namespace mfem;
using namespace AdGIA;

namespace {

#ifdef MFEM_USE_MPI
using MeshType = ParMesh;
using SpaceType = ParFiniteElementSpace;
using FieldType = ParGridFunction;
bool Root() { return Mpi::Root(); }
real_t GlobalMax(real_t v) {
  real_t g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                MPI_COMM_WORLD);
  return g;
}
#else
using MeshType = Mesh;
using SpaceType = FiniteElementSpace;
using FieldType = GridFunction;
bool Root() { return true; }
real_t GlobalMax(real_t v) { return v; }
#endif

// The constrained creep compliance of a Maxwell layer.
struct Compliance {
  real_t kappa, M, tau;
  real_t operator()(real_t t) const {
    return 1.0 / kappa -
           (M - kappa) / (kappa * M) * std::exp(-t * kappa / (M * tau));
  }
};

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif

  int dim = 2;
  int order = 2;
  int nx = 3, nz = 4;
  real_t W = 0.5, H = 1.0;
  real_t kappa = 1.0, mu = 1.0, tau_bottom = 1.0, tau_top = 0.1;
  real_t p0 = 0.01;
  real_t t_final = 5.0;
  int steps_per_unit_time = 16;
  bool visualization = true;
  const char* csv_file = "viscoelastic_column.csv";

  OptionsParser args(argc, argv);
  args.AddOption(&dim, "-d", "--dimension", "Space dimension (2 or 3).");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&nx, "-nx", "--num-elements-x",
                 "Elements across the (periodic) width, >= 3.");
  args.AddOption(&nz, "-nz", "--num-elements-z",
                 "Elements up the column; even, so that z = H/2 is a face.");
  args.AddOption(&kappa, "-kappa", "--bulk-modulus", "Bulk modulus.");
  args.AddOption(&mu, "-mu", "--shear-modulus", "Shear modulus (both layers).");
  args.AddOption(&tau_bottom, "-tb", "--tau-bottom",
                 "Maxwell time of the bottom layer (the time unit).");
  args.AddOption(&tau_top, "-tt", "--tau-top", "Maxwell time of the top layer.");
  args.AddOption(&p0, "-p0", "--load", "Traction on the top.");
  args.AddOption(&t_final, "-tf", "--t-final", "Final time.");
  args.AddOption(&steps_per_unit_time, "-n", "--steps-per-unit-time",
                 "Time steps per unit time.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization", "Show the final errors in GLVis.");
  args.AddOption(&csv_file, "-csv", "--csv",
                 "Table of the histories for plot_csv.py (\"\": none).");
  args.Parse();
  if (!args.Good() || (dim != 2 && dim != 3) || nx < 3 || nz < 2 ||
      nz % 2 != 0) {
    if (Root()) {
      args.PrintUsage(std::cout);
      std::cout << "(need -d 2 or 3, -nx >= 3, -nz even)\n";
    }
    return 1;
  }
  if (Root()) {
    args.PrintOptions(std::cout);
  }

  // The periodic column. Boundary attributes of the Cartesian mesh: the
  // bottom is 1; the top is 3 (2-D) or 6 (3-D). The side faces become
  // interior faces of the periodic mesh.
  Mesh serial = dim == 2 ? Mesh::MakeCartesian2D(nx, nz, Element::QUADRILATERAL,
                                                 true, W, H)
                         : Mesh::MakeCartesian3D(nx, nx, nz, Element::HEXAHEDRON,
                                                 W, W, H);
  {
    std::vector<Vector> shifts;
    for (int i = 0; i < dim - 1; i++) {
      Vector v(dim);
      v = 0.0;
      v[i] = W;
      shifts.push_back(v);
    }
    serial = Mesh::MakePeriodic(serial,
                                serial.CreatePeriodicVertexMapping(shifts));
  }
#ifdef MFEM_USE_MPI
  MeshType mesh(MPI_COMM_WORLD, serial);
  serial.Clear();
#else
  MeshType& mesh = serial;
#endif
  const int top_attr = dim == 2 ? 3 : 6;
  Array<int> bottom(mesh.bdr_attributes.Max()), top(mesh.bdr_attributes.Max());
  bottom = 0;
  bottom[0] = 1;
  top = 0;
  top[top_attr - 1] = 1;

  H1_FECollection fec(order, dim);
  SpaceType fes(&mesh, &fec, dim), fes_z(&mesh, &fec);

  const real_t M = kappa + 2.0 * (1.0 - 1.0 / dim) * mu;
  const Compliance c_bottom{kappa, M, tau_bottom}, c_top{kappa, M, tau_top};
  auto exact = [&](real_t z, real_t z_i, real_t t) {
    return -p0 * (std::min(z, z_i) * c_bottom(t) +
                  std::max(z - z_i, real_t(0)) * c_top(t));
  };

  // The two interface heights: on the face z = H/2, and half an element
  // above it, through the middle of an element.
  const real_t dz = H / nz;
  const std::vector<real_t> heights{0.5 * H, 0.5 * H + 0.5 * dz};
  const std::vector<std::string> names{"on a face", "cutting an element"};

  // The results of each run, recorded at the common output times (four
  // per unit time, and t_final).
  struct Run {
    std::vector<real_t> top;  // u_z at the top, per output time
    FieldType error;          // u_z - exact at the final time
  };
  std::vector<Run> runs;
  std::vector<real_t> times;
  const int n_steps = static_cast<int>(std::round(t_final * steps_per_unit_time));
  const real_t dt = t_final / n_steps;
  const int out_every = std::max(1, steps_per_unit_time / 4);

  for (std::size_t r = 0; r < heights.size(); r++) {
    const real_t z_i = heights[r];
    ConstantCoefficient kappa_c(kappa), mu_c(mu);
    FunctionCoefficient tau_c([&](const Vector& x) {
      return x[dim - 1] < z_i ? tau_bottom : tau_top;
    });
    auto rheology = IsotropicMaxwellRheology::Maxwell(dim, kappa_c, mu_c, tau_c);
    VectorFunctionCoefficient traction(dim, [&](const Vector&, Vector& f) {
      f = 0.0;
      f[dim - 1] = -p0;
    });
    LinearQuasiStaticClampedProblem problem(&fes, rheology, bottom, traction,
                                            top);
    problem.SetPrintLevel(IterativeSolver::PrintLevel().None());
    ViscoelasticOperator visco(problem);
    ExponentialTrapezoidSolver ode;
    ode.Init(visco);

    // u_z at the top: any top vertex (the solution is uniform sideways);
    // the rank holding one reports it, the others the -inf sentinel.
    int top_vertex = -1;
    for (int v = 0; v < mesh.GetNV(); v++) {
      if (std::abs(mesh.GetVertex(v)[dim - 1] - H) < 1e-12) {
        top_vertex = v;
        break;
      }
    }
    auto top_value = [&]() {
      real_t v = -std::numeric_limits<real_t>::infinity();
      if (top_vertex >= 0) {
        v = problem.Displacement()(fes.DofToVDof(top_vertex, dim - 1));
      }
      return GlobalMax(v);
    };

    Run run{{}, FieldType(&fes_z)};
    Vector m(visco.Height());
    m = 0.0;
    real_t t = 0.0;
    visco.SolveElastic(m, t);
    run.top.push_back(top_value());
    if (r == 0) {
      times.push_back(t);
    }
    for (int s = 1; s <= n_steps; s++) {
      real_t h = dt;
      ode.Step(m, t, h);
      t = s * dt;
      if (s % out_every == 0 || s == n_steps) {
        visco.SolveElastic(m, t);
        run.top.push_back(top_value());
        if (r == 0) {
          times.push_back(t);
        }
      }
    }

    // The error field at the final time: the vertical component, minus the
    // exact solution interpolated at the nodes.
    VectorGridFunctionCoefficient u_c(&problem.Displacement());
    Vector e_z(dim);
    e_z = 0.0;
    e_z(dim - 1) = 1.0;
    VectorConstantCoefficient e_z_c(e_z);
    InnerProductCoefficient u_z_c(u_c, e_z_c);
    run.error.ProjectCoefficient(u_z_c);
    FieldType ex(&fes_z);
    FunctionCoefficient ex_c(
        [&](const Vector& x) { return exact(x[dim - 1], z_i, t); });
    ex.ProjectCoefficient(ex_c);
    run.error -= ex;
    runs.push_back(std::move(run));
  }

  // The table: top displacement and its error, both runs.
  examples::CsvTable table(csv_file, {"t", "on_face", "on_face_exact", "cut",
                                      "cut_exact", "error_on_face",
                                      "error_cut"});
  table.Meta("title", "Two-layer Maxwell column: interface on a face vs "
                      "inside an element")
      .Meta("note", "top displacement (dashed: exact) and its error")
      .Meta("xlabel", "time / Maxwell time of the bottom layer")
      .Meta("y", "on_face,cut|error_on_face,error_cut")
      .Meta("ylabel", "u_z at the top|abs. error")
      .Meta("logy", "false|true");
  if (Root()) {
    std::cout << "\nConstrained modulus M = " << M << ", relaxed kappa = "
              << kappa << "; interface at z = " << heights[0] << " ("
              << names[0] << ") and " << heights[1] << " (" << names[1]
              << ")\n\n      t     u_top(face)   error       u_top(cut)    "
                 "error\n";
  }
  for (std::size_t k = 0; k < times.size(); k++) {
    const real_t t = times[k];
    std::vector<double> row{t};
    std::vector<double> errs;
    for (std::size_t r = 0; r < runs.size(); r++) {
      const real_t ex = exact(H, heights[r], t);
      row.insert(row.end(), {runs[r].top[k], ex});
      errs.push_back(std::abs(runs[r].top[k] - ex));
    }
    row.insert(row.end(), errs.begin(), errs.end());
    table.Row(row);
    if (Root()) {
      std::cout << std::fixed << std::setprecision(3) << std::setw(7) << t
                << std::scientific << std::setprecision(4);
      for (std::size_t r = 0; r < runs.size(); r++) {
        std::cout << std::setw(14) << runs[r].top[k] << std::setprecision(1)
                  << std::setw(10) << errs[r] << std::setprecision(4);
      }
      std::cout << "\n";
    }
  }
  table.Write();

  if (visualization) {
    // One colour scale for both, set by the larger error.
    real_t emax = 0.0;
    for (const auto& run : runs) {
      emax = std::max(emax, GlobalMax(run.error.Normlinf()));
    }
    for (std::size_t r = 0; r < runs.size(); r++) {
      examples::GLVisWindow w("u_z - exact at t = " +
                                  std::to_string(t_final).substr(0, 4) +
                                  ", interface " + names[r],
                              examples::DefaultKeys(dim));
      w.SetValueRange(-emax, emax);
      w.Send(mesh, runs[r].error);
    }
  }
  return 0;
}
