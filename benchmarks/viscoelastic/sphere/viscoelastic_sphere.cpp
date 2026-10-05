// ============================================================================
// viscoelastic_sphere.cpp
//
// Non-gravitating, solid-only viscoelastic benchmarks on spherically layered
// balls (3-D) and discs (2-D), for comparison with the exact references of
// sphere/reference.py or, with lateral viscosity variations, with a run of
// a much smaller step on the same mesh. A case (sphere/cases.py) names a
// mesh manifest (sphere/meshes.py: planetmodel, every layer interface a
// surface of the mesh), gives each layer of the manifest a generalised
// Maxwell rheology (fields of the radius, or varying laterally, as in
// viscoelastic_common.hpp), and a surface load with its history:
//
//   a normal pressure  amplitude * S(t) * Y_l(x/|x|)  on the surface,
//
// Y_l the real orthonormal harmonic of degree l and order 0 in 3-D (polar
// axis the last coordinate), cos(l theta)/sqrt(pi) in 2-D (the library's
// SurfaceHarmonics). The body is free (LinearQuasiStaticTractionProblem,
// rigid modes projected out): l >= 2, so the load has no net force or
// torque, and there is no gravity, so no isostasy either. A body with no
// elastic part therefore flows under the load.
//
// Observables at each output time (and t = 0+): the surface coefficients of
// the radial and the tangential displacement, u = sum_d U_d Y_d xhat +
// V_d grad_1 Y_d, for every degree d <= lmax at order 0 (3-D) / the cosine
// (2-D): a radial model answers at d = l only, a laterally varying one
// couples the degrees.
//
// Stepping (the schemes, the step grid, jumps) as viscoelastic_box: the
// shared Evolve() of viscoelastic_common.hpp.
//
// Sample runs:
//    mpiexec -np 4 ./viscoelastic_sphere -c case.json -o 2 -scheme exptrap
//        -dt 0.05 -out results.json
// ============================================================================

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#include "benchmark_case.hpp"
#include "viscoelastic_common.hpp"

using namespace mfem;
using namespace AdGIA;
using namespace benchmark;
using namespace vebench;

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();
  const bool root = Mpi::Root();

  const char* case_file = "case.json";
  const char* out_file = "results.json";
  int order = 2;
  int refine = 0;
  StepOptions step;
  const char* scheme = "exptrap";

  OptionsParser args(argc, argv);
  args.AddOption(&case_file, "-c", "--case", "Case file (cases.py).");
  args.AddOption(&out_file, "-out", "--output", "Results file (JSON).");
  args.AddOption(&order, "-o", "--order", "Displacement order.");
  args.AddOption(&refine, "-r", "--refine",
                 "Uniform refinements of the mesh (curved: the nodes "
                 "follow the mesh's own geometry).");
  args.AddOption(&scheme, "-scheme", "--scheme",
                 "exptrap (default), sdirk23, be, etd1, rk4 or adaptive.");
  args.AddOption(&step.dt, "-dt", "--step",
                 "Fixed-step schemes: largest step; adaptive: first step.");
  args.AddOption(&step.min_steps, "-min-steps", "--min-steps",
                 "Fixed-step schemes: least number of steps per interval.");
  args.AddOption(&step.align, "-align", "--align-breakpoints", "-no-align",
                 "--no-align-breakpoints",
                 "Put the load history's breakpoints on the step grid.");
  args.AddOption(&step.rtol, "-rtol", "--adaptive-rtol",
                 "Adaptive scheme: relative tolerance.");
  args.AddOption(&step.atol, "-atol", "--adaptive-atol",
                 "Adaptive scheme: absolute tolerance.");
  args.Parse();
  if (!args.Good()) {
    if (root) args.PrintUsage(std::cout);
    return 1;
  }
  if (root) args.PrintOptions(std::cout);
  step.scheme = scheme;
  MFEM_VERIFY(KnownScheme(step.scheme),
              "-scheme must be exptrap, sdirk23, be, etd1, rk4 or adaptive.");

  // --- the case and the mesh --------------------------------------------------

  const Json json = ReadJsonFile(case_file);
  const std::string case_name = Member(json, "name").string;
  const MeshManifest manifest(Member(json, "mesh").string);
  Mesh serial = manifest.LoadMesh();
  const int dim = serial.Dimension();
  MFEM_VERIFY(dim == static_cast<int>(Number(json, "dim")),
              "Case: dim does not match the mesh.");
  for (int i = 0; i < refine; i++) serial.UniformRefinement();
  ParMesh mesh(MPI_COMM_WORLD, serial);
  serial.Clear();

  // The case's layers, centre outward, matched to the manifest's.
  std::vector<Layer> layers;
  {
    std::vector<const MeshManifest::Layer*> body;
    for (const auto& l : manifest.Layers()) {
      if (l.in_geometry) body.push_back(&l);
    }
    const auto& entries = Member(json, "layers").array;
    MFEM_VERIFY(entries.size() == body.size(),
                "Case: " << entries.size() << " layers for a mesh of "
                         << body.size() << ".");
    for (std::size_t j = 0; j < body.size(); j++) {
      Layer layer =
          ParseLayer(entries[j], body[j]->r_inner, body[j]->r_outer);
      layer.attribute = body[j]->attribute;
      layers.push_back(std::move(layer));
    }
  }

  const Json& load = Member(json, "load");
  const int degree = static_cast<int>(Number(load, "degree"));
  const real_t amplitude = Number(load, "amplitude");
  MFEM_VERIFY(degree >= 2, "The load needs degree >= 2 (a free body).");
  LoadHistory history(Member(load, "history"));
  const int lmax = static_cast<int>(NumberOr(json, "lmax", degree));
  MFEM_VERIFY(lmax >= degree, "Case: lmax below the load's degree.");
  const std::vector<real_t> times = Numbers(Member(json, "times"));

  H1_FECollection fec(order, dim);
  ParFiniteElementSpace fes(&mesh, &fec, dim);
  const HYPRE_BigInt ndofs = fes.GlobalTrueVSize();

  // --- the rheology: one region per layer -------------------------------------

  const Coordinate radius = [](const Vector& x) { return x.Norml2(); };
  std::vector<std::unique_ptr<Coefficient>> coefs;
  auto single = [&](const Field& f) -> Coefficient& {
    std::vector<Field> fs(layers.size(), f);
    coefs.push_back(
        std::make_unique<LayeredCoefficient>(&layers, std::move(fs), radius));
    return *coefs.back();
  };
  std::vector<std::unique_ptr<IsotropicMaxwellRheology>> maxwell;
  std::vector<std::unique_ptr<IsotropicElasticRheology>> elastic;
  std::vector<RheologyRegion> regions;
  const int na = mesh.attributes.Max();
  for (const Layer& l : layers) {
    Array<int> marker(na);
    marker = 0;
    marker[l.attribute - 1] = 1;
    Coefficient& kappa = single(l.kappa);
    Coefficient& mu_inf = single(l.mu_inf);
    if (l.branches.empty()) {
      elastic.push_back(
          std::make_unique<IsotropicElasticRheology>(dim, kappa, mu_inf));
      regions.push_back({marker, elastic.back().get(), l.name});
    } else {
      std::vector<MaxwellBranch> branches;
      for (const Branch& b : l.branches) {
        branches.push_back({&single(b.mu), &single(b.tau), nullptr});
      }
      maxwell.push_back(std::make_unique<IsotropicMaxwellRheology>(
          dim, kappa, mu_inf, branches));
      regions.push_back({marker, maxwell.back().get(), l.name});
    }
  }
  CompositeRheology rheology(dim, regions);

  // --- the problem ------------------------------------------------------------

  const SurfaceHarmonics basis(dim, lmax);
  auto order_of = [dim](int d) { return dim == 3 ? 0 : d; };  // m = 0 / cos
  const int load_index = basis.Index(degree, order_of(degree));
  VectorFunctionCoefficient traction(
      dim, [&](const Vector& x, real_t t, Vector& f) {
        Vector Y;
        basis.Eval(x, Y);
        f = x;
        f /= x.Norml2();
        f *= -amplitude * history(t) * Y[load_index];
      });
  const int nbdr = mesh.bdr_attributes.Max();
  const Array<int> surface =
      MeshManifest::Marker(Array<int>({manifest.SurfaceAttribute()}), nbdr);
  LinearQuasiStaticTractionProblem problem(&fes, rheology, traction,
                                           surface);
  problem.SetPrintLevel(IterativeSolver::PrintLevel().None());
  ViscoelasticOperator visco(problem);

  BoundaryHarmonicCoefficients radial(
      fes, surface, lmax, BoundaryHarmonicCoefficients::Component::Radial);
  BoundaryHarmonicCoefficients tangential(
      fes, surface, lmax, BoundaryHarmonicCoefficients::Component::Tangential);

  struct Snapshot {
    real_t time = 0.0;
    std::vector<real_t> U, V;
  };
  Snapshot elastic_snapshot;
  std::vector<Snapshot> snapshots;
  auto observe = [&](const Vector&, real_t t, int k) {
    Vector cu, cv;
    radial.Coefficients(problem.Displacement(), cu);
    tangential.Coefficients(problem.Displacement(), cv);
    Snapshot s;
    s.time = t;
    for (int d = 0; d <= lmax; d++) {
      const int i = basis.Index(d, order_of(d));
      s.U.push_back(cu[i]);
      s.V.push_back(cv[i]);
    }
    if (k < 0) {
      elastic_snapshot = s;
    } else {
      snapshots.push_back(s);
      if (root) {
        std::cout << std::setw(12) << std::setprecision(5) << t
                  << std::setw(16) << std::setprecision(8) << s.U[degree]
                  << std::setw(16) << s.V[degree] << "\n";
      }
    }
  };

  if (root) {
    std::cout << "\nCase " << case_name << ": " << dim << "-D, "
              << layers.size() << " layer(s), degree " << degree << ", "
              << visco.NumBranches() << " branch(es), " << ndofs
              << " dofs, scheme " << step.scheme << "\n";
  }
  const EvolveResult run =
      Evolve(problem, visco, history, times, step, observe);

  if (root) {
    std::cout << std::setprecision(4) << "\n" << run.steps << " steps, "
              << run.stepping.solves << " stepping solves + "
              << run.observation.solves << " observation, "
              << run.total.assemblies << " assemblies, " << run.total.setups
              << " preconditioner setups, " << run.total.its
              << " iterations, " << run.seconds << " s"
              << (run.ok ? "" : ", SOME SOLVES FAILED") << "\n";
    std::ofstream os(out_file);
    MFEM_VERIFY(os.good(), "Cannot write " << out_file << ".");
    auto quantities = [&](const Snapshot& s) {
      return "\"U\": " + List(s.U) + ", \"V\": " + List(s.V);
    };
    os << "{\n  \"case\": \"" << case_name << "\",\n  \"case_file\": \""
       << case_file << "\",\n  \"dim\": " << dim
       << ",\n  \"kind\": \"sphere\",\n  \"load\": \"surface\""
       << ",\n  \"degree\": " << degree << ",\n  \"lmax\": " << lmax
       << ",\n  \"order\": " << order << ",\n  \"refine\": " << refine
       << ",\n  \"dofs\": " << ndofs << ",\n  \"ranks\": "
       << Mpi::WorldSize() << ",\n  \"branches\": " << visco.NumBranches()
       << ",\n  \"scheme\": \"" << step.scheme << "\",\n  \"dt\": "
       << Num(step.dt) << ",\n  \"min_steps\": " << step.min_steps
       << ",\n  \"align\": " << (step.align ? "true" : "false")
       << ",\n  \"rtol\": " << Num(step.rtol) << ",\n  \"atol\": "
       << Num(step.atol) << ",\n  \"times\": " << List(times)
       << ",\n  \"elastic\": {" << quantities(elastic_snapshot)
       << "},\n  \"histories\": [";
    for (std::size_t n = 0; n < snapshots.size(); n++) {
      const OutputCost& c = run.outputs[n];
      os << (n ? "," : "") << "\n    {\"time\": " << Num(snapshots[n].time)
         << ", " << quantities(snapshots[n])
         << ", \"solves\": " << c.cost.solves
         << ", \"assemblies\": " << c.cost.assemblies
         << ", \"iterations\": " << c.cost.its
         << ", \"seconds\": " << Num(c.seconds) << "}";
    }
    os << "],\n  \"cost\": {\"steps\": " << run.steps
       << ", \"rejected_steps\": " << run.rejected
       << ", \"stepping_solves\": " << run.stepping.solves
       << ", \"observation_solves\": " << run.observation.solves
       << ", \"assemblies\": " << run.total.assemblies
       << ", \"preconditioner_setups\": " << run.total.setups
       << ", \"iterations\": " << run.total.its
       << ", \"seconds\": " << Num(run.seconds)
       << ", \"converged\": " << (run.ok ? "true" : "false") << "}\n}\n";
    std::cout << "Wrote " << out_file << "\n";
  }
  return 0;
}
