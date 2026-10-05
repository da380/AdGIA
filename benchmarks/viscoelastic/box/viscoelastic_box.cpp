// ============================================================================
// viscoelastic_box.cpp
//
// Non-gravitating, solid-only viscoelastic benchmarks on rectangular meshes,
// for comparison with the exact references of reference.py. A case
// (cases.py) gives the geometry, a stack of horizontal layers (each a
// generalised Maxwell body: bulk modulus, long-term shear modulus, Prony
// branches; no branches is an elastic layer), a load with its spatial
// pattern and its time history, and the output times. This driver adds the
// discretisation and the time integrator, and writes the observables at
// every output time with the cost of reaching it.
//
// Geometry: the box [0, Lx] x [0, H] (2-D) or [0, Lx] x [0, Ly] x [0, H]
// (3-D), z the last coordinate. A "slab" is periodic in the horizontal
// directions; a "box" is not. Layers are stacked from z = 0 upwards.
//
// Loads (the case's load.kind), each scaled by amplitude * S(t):
//   uniaxial_stress  box, pure traction: +-e_x on the faces x = Lx, 0
//                    (LinearQuasiStaticTractionProblem). A homogeneous
//                    state: the finite elements are exact in space and only
//                    the time integration errs.
//   uniaxial_strain  box, the whole boundary prescribed: u = S(t) x_1 e_x
//                    (LinearQuasiStaticClampedProblem with time-dependent
//                    Dirichlet data). Homogeneous as well.
//   surface          slab, clamped at z = 0, a normal pressure
//                    amplitude * S(t) * cos(kx x) cos(ky y) on z = H
//                    (k = 2 pi n / L for the case's mode n; n = 0 is a
//                    uniform pressure: the laterally uniform "column").
//
// S(t) is a sum of pieces, each c0 + c1 s + a cos(w s) + b sin(w s) with
// s = t - start on [start, end), right-continuous; the case lists the
// breakpoints (piece ends, flagged when the history jumps there). With
// -align (the default) the step grid includes every breakpoint, and a step
// ending at a jump sees the load's LEFT limit, the next one starting from
// the right limit (ViscoelasticOperator::InvalidateDisplacement). With
// -no-align the breakpoints fall wherever the steps put them: the
// measurement of the order lost to an unresolved jump or kink.
//
// Mesh: uniform. Material discontinuities must lie on element faces, so
// the driver refuses an nz for which a layer interface cuts elements;
// -no-conform allows it, as the deliberate misfit of the interface study.
//
// Rheology (-rheology): composite (default) gives each layer its own
// region of a CompositeRheology, the layer of an element being that of its
// centre, so a layer interface that cuts through elements (-no-conform) is
// staircased;
// pointwise gives one IsotropicMaxwellRheology whose coefficients look the
// layer up at every quadrature point and internal-variable node, so an
// interface may cut elements (branches padded to a common count with zero
// moduli). Moduli and times may vary geometrically in z within a layer
// ({"kind": "geometric_z", "bottom": v0, "top": v1}).
//
// Observables at each output time (and at t = 0+):
//   box:   the mean strain tensor and, per branch, the mean internal
//          variable (full d x d tensors, row-major);
//   slab:  the surface amplitudes W = (u_z, c) / (c, c) and
//          U = (u_h, g) / (g, g) over z = H, with c = cos(kx x) cos(ky y)
//          and g = -grad_h c / |k| (the horizontal pattern of a poloidal
//          field), i.e. u_z = W c, u_h = -(U / |k|) grad_h c.
// with the cumulative stepping solves, assemblies, preconditioner setups,
// Krylov iterations and seconds.
//
// Schemes (-scheme): exptrap, sdirk23 (MFEM's L-stable variant), be, etd1,
// rk4, adaptive (-rtol, -atol). Fixed-step schemes divide every interval
// of the step grid into max(-min-steps, ceil(length / -dt)) equal steps.
//
// Sample runs:
//    mpiexec -np 1 ./viscoelastic_box -c case.json -scheme exptrap -dt 0.1
//    mpiexec -np 4 ./viscoelastic_box -c slab.json -o 2 -nx 16 -nz 16
// ============================================================================

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>

#include "benchmark_case.hpp"
#include "viscoelastic_common.hpp"

using namespace mfem;
using namespace AdGIA;
using namespace benchmark;
using namespace vebench;

namespace {

// The integral of cos^2 (sin^2) of k x over a period of length L, or over
// [0, L] for k = 0.
real_t CosSquared(real_t k, real_t L) { return k != 0.0 ? 0.5 * L : L; }
real_t SinSquared(real_t k, real_t L) { return k != 0.0 ? 0.5 * L : 0.0; }

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();
  const bool root = Mpi::Root();

  const char* case_file = "case.json";
  const char* out_file = "results.json";
  const char* scheme = "exptrap";
  const char* rheology_mode = "composite";
  int order = 1;
  int nx = 4, nz = 4;
  real_t dt = 0.1;
  int min_steps = 1;
  bool align = true;
  bool conform = true;
  real_t rtol = 1e-4, atol = 1e-10;

  OptionsParser args(argc, argv);
  args.AddOption(&case_file, "-c", "--case", "Case file (cases.py).");
  args.AddOption(&out_file, "-out", "--output", "Results file (JSON).");
  args.AddOption(&order, "-o", "--order", "Displacement order.");
  args.AddOption(&nx, "-nx", "--nx",
                 "Elements along each horizontal direction.");
  args.AddOption(&nz, "-nz", "--nz", "Elements along z.");
  args.AddOption(&scheme, "-scheme", "--scheme",
                 "exptrap (default), sdirk23, be, etd1, rk4 or adaptive.");
  args.AddOption(&rheology_mode, "-rheology", "--rheology",
                 "composite (one region per layer, by element centre) or "
                 "pointwise (layers looked up at every point).");
  args.AddOption(&dt, "-dt", "--step",
                 "Fixed-step schemes: largest step (every interval of the "
                 "step grid is divided into equal steps no longer than "
                 "this). Adaptive: the first trial step.");
  args.AddOption(&min_steps, "-min-steps", "--min-steps",
                 "Fixed-step schemes: least number of steps per interval.");
  args.AddOption(&align, "-align", "--align-breakpoints", "-no-align",
                 "--no-align-breakpoints",
                 "Put the load history's breakpoints on the step grid.");
  args.AddOption(&conform, "-conform", "--conform", "-no-conform",
                 "--no-conform",
                 "Require every layer interface on element faces (default); "
                 "-no-conform allows interfaces inside elements (the "
                 "deliberate misfit of the interface study).");
  args.AddOption(&rtol, "-rtol", "--adaptive-rtol",
                 "Adaptive scheme: relative tolerance.");
  args.AddOption(&atol, "-atol", "--adaptive-atol",
                 "Adaptive scheme: absolute tolerance.");
  args.Parse();
  if (!args.Good()) {
    if (root) args.PrintUsage(std::cout);
    return 1;
  }
  if (root) args.PrintOptions(std::cout);
  const std::string scheme_name(scheme), mode(rheology_mode);
  MFEM_VERIFY(scheme_name == "exptrap" || scheme_name == "sdirk23" ||
                  scheme_name == "be" || scheme_name == "etd1" ||
                  scheme_name == "rk4" || scheme_name == "adaptive",
              "-scheme must be exptrap, sdirk23, be, etd1, rk4 or adaptive.");
  MFEM_VERIFY(mode == "composite" || mode == "pointwise",
              "-rheology must be composite or pointwise.");
  MFEM_VERIFY(dt > 0.0 && min_steps >= 1, "Need -dt > 0, -min-steps >= 1.");

  // --- the case ---------------------------------------------------------------

  const Json json = ReadJsonFile(case_file);
  const std::string case_name = Member(json, "name").string;
  const int dim = static_cast<int>(Number(json, "dim"));
  MFEM_VERIFY(dim == 2 || dim == 3, "Case: dim must be 2 or 3.");
  const int zdim = dim - 1;
  const Json& geometry = Member(json, "geometry");
  const std::string kind = Member(geometry, "kind").string;
  MFEM_VERIFY(kind == "box" || kind == "slab",
              "Case: geometry.kind must be box or slab.");
  const bool periodic = kind == "slab";
  const std::vector<real_t> lengths = Numbers(Member(geometry, "length"));
  MFEM_VERIFY(static_cast<int>(lengths.size()) == dim - 1,
              "Case: geometry.length needs dim - 1 entries.");
  const real_t H = Number(geometry, "height");

  std::vector<Layer> layers;
  {
    real_t bottom = 0.0;
    for (const Json& l : Member(json, "layers").array) {
      const real_t top = Number(l, "top");
      MFEM_VERIFY(top > bottom, "Case: layers must stack upwards.");
      layers.push_back(ParseLayer(l, bottom, top));
      bottom = top;
    }
    MFEM_VERIFY(!layers.empty() && std::abs(bottom - H) < 1e-12 * H,
                "Case: the layers must fill [0, height].");
  }

  const Json& load = Member(json, "load");
  const std::string load_kind = Member(load, "kind").string;
  const real_t amplitude = Number(load, "amplitude");
  MFEM_VERIFY((load_kind == "surface") == periodic,
              "Case: the surface load goes with a slab, the uniaxial loads "
              "with a box.");
  MFEM_VERIFY(load_kind == "surface" || load_kind == "uniaxial_stress" ||
                  load_kind == "uniaxial_strain",
              "Case: unknown load kind '" << load_kind << "'.");
  std::vector<real_t> wavenumber(dim - 1, 0.0);
  if (periodic) {
    const std::vector<real_t> mode_n = Numbers(Member(load, "mode"));
    MFEM_VERIFY(static_cast<int>(mode_n.size()) == dim - 1,
                "Case: load.mode needs dim - 1 entries.");
    for (int i = 0; i < dim - 1; i++) {
      wavenumber[i] = 2.0 * kPi * mode_n[i] / lengths[i];
    }
  }
  real_t kmod = 0.0;
  for (const real_t k : wavenumber) kmod += k * k;
  kmod = std::sqrt(kmod);

  LoadHistory history(Member(load, "history"));
  const std::vector<real_t> times = Numbers(Member(json, "times"));
  for (std::size_t j = 0; j < times.size(); j++) {
    MFEM_VERIFY(times[j] > (j ? times[j - 1] : 0.0),
                "Case: output times must be positive and increasing.");
  }

  // --- the mesh ---------------------------------------------------------------
  //
  // Built serially and identically on every rank, then partitioned. Element
  // attribute = 1 + the layer of the element's centre; boundary attributes
  // 1 bottom, 2 top, 3/4 x = 0/Lx, 5/6 y = 0/Ly, from the face centres.

  if (periodic) {
    MFEM_VERIFY(nx >= 3, "A periodic slab needs -nx >= 3.");
  }
  Mesh serial =
      dim == 2 ? Mesh::MakeCartesian2D(nx, nz, Element::QUADRILATERAL, true,
                                       lengths[0], H)
               : Mesh::MakeCartesian3D(nx, nx, nz, Element::HEXAHEDRON,
                                       lengths[0], lengths[1], H);
  if (conform) {
    // Material discontinuities must sit on element faces: the cases put
    // every layer interface on the coarsest grid of their mesh ladder
    // (multiples of H/5), so a uniform nz conforms. Refuse a mesh that
    // would cut a layer; -no-conform allows it, for the misfit study.
    for (std::size_t j = 0; j + 1 < layers.size(); j++) {
      const real_t cell = layers[j].top * nz / H;
      MFEM_VERIFY(std::abs(cell - std::round(cell)) < 1e-9,
                  "The interface at z = " << layers[j].top
                      << " is not on an element face for nz = " << nz
                      << " (use an nz that conforms, or -no-conform for "
                         "the deliberate misfit).");
    }
  }
  {
    Vector c;
    for (int e = 0; e < serial.GetNE(); e++) {
      ElementTransformation* T = serial.GetElementTransformation(e);
      T->Transform(Geometries.GetCenter(T->GetGeometryType()), c);
      int j = 0;
      while (j + 1 < static_cast<int>(layers.size()) && c[zdim] >= layers[j].top)
        j++;
      serial.SetAttribute(e, j + 1);
    }
    for (int b = 0; b < serial.GetNBE(); b++) {
      ElementTransformation* T = serial.GetBdrElementTransformation(b);
      T->Transform(Geometries.GetCenter(T->GetGeometryType()), c);
      const real_t tol = 1e-10 * std::max<real_t>(H, lengths[0]);
      int a = 0;
      if (std::abs(c[zdim]) < tol) {
        a = 1;
      } else if (std::abs(c[zdim] - H) < tol) {
        a = 2;
      } else if (std::abs(c[0]) < tol) {
        a = 3;
      } else if (std::abs(c[0] - lengths[0]) < tol) {
        a = 4;
      } else if (dim == 3 && std::abs(c[1]) < tol) {
        a = 5;
      } else {
        a = 6;
      }
      serial.SetBdrAttribute(b, a);
    }
    serial.SetAttributes();
  }
  if (periodic) {
    // The horizontal translations identify the side faces; their boundary
    // elements stay in the mesh as interior faces that nothing marks.
    std::vector<Vector> shifts;
    for (int i = 0; i < dim - 1; i++) {
      Vector v(dim);
      v = 0.0;
      v[i] = lengths[i];
      shifts.push_back(v);
    }
    serial = Mesh::MakePeriodic(serial,
                                serial.CreatePeriodicVertexMapping(shifts));
  }
  ParMesh mesh(MPI_COMM_WORLD, serial);
  serial.Clear();
  // The highest boundary attribute of the full mesh (the periodic one
  // keeps its side faces), the same on every rank.
  const int nbdr = 2 * dim;

  H1_FECollection fec(order, dim);
  ParFiniteElementSpace fes(&mesh, &fec, dim);
  const HYPRE_BigInt ndofs = fes.GlobalTrueVSize();

  // --- the rheology -----------------------------------------------------------

  std::vector<std::unique_ptr<Coefficient>> coefs;
  const Coordinate height = [zdim](const Vector& x) { return x[zdim]; };
  auto layered = [&](auto pick) -> Coefficient& {
    std::vector<Field> f;
    for (const Layer& l : layers) f.push_back(pick(l));
    coefs.push_back(
        std::make_unique<LayeredCoefficient>(&layers, std::move(f), height));
    return *coefs.back();
  };
  // One layer's field everywhere (a composite region only evaluates it on
  // its own elements).
  auto single = [&](const Field& f) -> Coefficient& {
    std::vector<Field> fs(layers.size(), f);
    coefs.push_back(
        std::make_unique<LayeredCoefficient>(&layers, std::move(fs), height));
    return *coefs.back();
  };

  std::vector<std::unique_ptr<IsotropicMaxwellRheology>> maxwell;
  std::vector<std::unique_ptr<IsotropicElasticRheology>> elastic;
  std::unique_ptr<CompositeRheology> composite;
  const Rheology* rheology = nullptr;
  constexpr real_t kNever = 1e300;  // tau of a padded, zero-modulus branch
  if (mode == "composite") {
    std::vector<RheologyRegion> regions;
    const int na = mesh.attributes.Max();
    for (std::size_t j = 0; j < layers.size(); j++) {
      const Layer& l = layers[j];
      Array<int> marker(na);
      marker = 0;
      if (static_cast<int>(j) < na) marker[j] = 1;
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
    composite = std::make_unique<CompositeRheology>(dim, regions);
    rheology = composite.get();
  } else {
    std::size_t nb = 0;
    for (const Layer& l : layers) nb = std::max(nb, l.branches.size());
    MFEM_VERIFY(nb > 0, "Every layer is elastic: nothing relaxes.");
    Coefficient& kappa = layered([](const Layer& l) { return l.kappa; });
    Coefficient& mu_inf = layered([](const Layer& l) { return l.mu_inf; });
    std::vector<MaxwellBranch> branches;
    for (std::size_t k = 0; k < nb; k++) {
      const Field zero = Field::Constant(0.0), never = Field::Constant(kNever);
      Coefficient& mu = layered([&](const Layer& l) {
        return k < l.branches.size() ? l.branches[k].mu : zero;
      });
      Coefficient& tau = layered([&](const Layer& l) {
        return k < l.branches.size() ? l.branches[k].tau : never;
      });
      branches.push_back({&mu, &tau, nullptr});
    }
    maxwell.push_back(std::make_unique<IsotropicMaxwellRheology>(
        dim, kappa, mu_inf, branches));
    rheology = maxwell.back().get();
  }

  // --- the problem ------------------------------------------------------------

  VectorFunctionCoefficient traction(
      dim, [&](const Vector& x, real_t t, Vector& f) {
        f = 0.0;
        const real_t s = amplitude * history(t);
        if (load_kind == "uniaxial_stress") {
          const real_t tol = 1e-10 * lengths[0];
          if (std::abs(x[0] - lengths[0]) < tol) {
            f[0] = s;
          } else if (std::abs(x[0]) < tol) {
            f[0] = -s;
          }
        } else if (load_kind == "surface") {
          real_t c = 1.0;
          for (int i = 0; i < dim - 1; i++) c *= std::cos(wavenumber[i] * x[i]);
          f[zdim] = -s * c;
        }
      });
  VectorFunctionCoefficient dirichlet(
      dim, [&](const Vector& x, real_t t, Vector& u) {
        u = 0.0;
        if (load_kind == "uniaxial_strain") {
          u[0] = amplitude * history(t) * x[0];
        }
      });

  std::unique_ptr<LinearQuasiStaticProblemBase> problem;
  Array<int> none(nbdr), all(nbdr), bottom(nbdr), top(nbdr), xfaces(nbdr);
  none = 0;
  all = 1;
  bottom = 0;
  bottom[0] = 1;
  top = 0;
  top[1] = 1;
  xfaces = 0;
  xfaces[2] = xfaces[3] = 1;
  if (load_kind == "uniaxial_stress") {
    problem = std::make_unique<LinearQuasiStaticTractionProblem>(
        &fes, *rheology, traction, xfaces);
  } else if (load_kind == "uniaxial_strain") {
    problem = std::make_unique<LinearQuasiStaticClampedProblem>(
        &fes, *rheology, all, traction, none, &dirichlet);
  } else {
    problem = std::make_unique<LinearQuasiStaticClampedProblem>(
        &fes, *rheology, bottom, traction, top);
  }
  problem->SetPrintLevel(IterativeSolver::PrintLevel().None());

  ViscoelasticOperator visco(*problem);
  const int nb = visco.NumBranches();

  // --- observables ------------------------------------------------------------

  // Slab: the surface pattern c and the horizontal pattern g as boundary
  // linear forms on z = H, applied to the true dofs of u.
  std::unique_ptr<HypreParVector> form_w, form_u;
  real_t norm_w = 1.0, norm_u = 1.0;
  if (periodic) {
    auto make = [&](bool vertical) {
      VectorFunctionCoefficient g(dim, [&, vertical](const Vector& x,
                                                     Vector& v) {
        v = 0.0;
        std::vector<real_t> cs(dim - 1), sn(dim - 1);
        for (int i = 0; i < dim - 1; i++) {
          cs[i] = std::cos(wavenumber[i] * x[i]);
          sn[i] = std::sin(wavenumber[i] * x[i]);
        }
        if (vertical) {
          real_t c = 1.0;
          for (int i = 0; i < dim - 1; i++) c *= cs[i];
          v[zdim] = c;
        } else if (kmod > 0.0) {
          // g = -grad_h c / |k|
          for (int i = 0; i < dim - 1; i++) {
            real_t d = wavenumber[i] * sn[i];
            for (int j = 0; j < dim - 1; j++) {
              if (j != i) d *= cs[j];
            }
            v[i] = d / kmod;
          }
        }
      });
      ParLinearForm lf(&fes);
      lf.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(g), top);
      lf.Assemble();
      return std::unique_ptr<HypreParVector>(lf.ParallelAssemble());
    };
    form_w = make(true);
    form_u = make(false);
    norm_w = 1.0;
    for (int i = 0; i < dim - 1; i++) {
      norm_w *= CosSquared(wavenumber[i], lengths[i]);
    }
    if (kmod > 0.0) {
      norm_u = 0.0;
      for (int i = 0; i < dim - 1; i++) {
        real_t term = wavenumber[i] * wavenumber[i] *
                      SinSquared(wavenumber[i], lengths[i]);
        for (int j = 0; j < dim - 1; j++) {
          if (j != i) term *= CosSquared(wavenumber[j], lengths[j]);
        }
        norm_u += term;
      }
      norm_u /= kmod * kmod;
    }
  }

  struct Snapshot {
    real_t time = 0.0;
    std::vector<real_t> strain;                 // box: mean strain, d x d
    std::vector<std::vector<real_t>> internal;  // box: per branch, d x d
    real_t W = 0.0, U = 0.0;                    // slab
  };

  auto observe = [&](const Vector& m, real_t t) {
    Snapshot s;
    s.time = t;
    const auto& u = static_cast<const ParGridFunction&>(problem->Displacement());
    if (periodic) {
      std::unique_ptr<HypreParVector> ut(u.GetTrueDofs());
      s.W = InnerProduct(*form_w, *ut) / norm_w;
      s.U = kmod > 0.0 ? InnerProduct(*form_u, *ut) / norm_u : 0.0;
      return s;
    }
    // The mean strain over the element centres (a homogeneous state).
    std::vector<real_t> E(dim * dim, 0.0);
    DenseMatrix grad;
    for (int e = 0; e < mesh.GetNE(); e++) {
      ElementTransformation* T = mesh.GetElementTransformation(e);
      const IntegrationPoint& ip = Geometries.GetCenter(T->GetGeometryType());
      T->SetIntPoint(&ip);
      u.GetVectorGradient(*T, grad);
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
          E[i * dim + j] += 0.5 * (grad(i, j) + grad(j, i));
        }
      }
    }
    const real_t ne = GlobalSum(mesh.GetNE());
    for (real_t& v : E) v = GlobalSum(v) / ne;
    s.strain = E;
    // The mean internal variables, as full tensors.
    visco.SyncFields(m);
    const int nd = visco.InternalScalarSpace().GetNDofs();
    const int nc = visco.NumComponents();
    const real_t nodes = GlobalSum(nd);
    for (int k = 0; k < nb; k++) {
      const GridFunction& g = visco.InternalVariable(k);
      std::vector<real_t> comp(nc, 0.0);
      for (int c = 0; c < nc; c++) {
        real_t sum = 0.0;
        for (int q = 0; q < nd; q++) sum += g[c * nd + q];
        comp[c] = GlobalSum(sum) / nodes;
      }
      std::vector<real_t> M(dim * dim, 0.0);
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
          const int o = SymmetricComponentOrder::Offset(dim, i, j);
          if (o < nc) M[i * dim + j] = comp[o];
        }
      }
      if (visco.TraceFree()) {
        real_t tr = 0.0;
        for (int i = 0; i < dim - 1; i++) tr += M[i * dim + i];
        M[(dim - 1) * dim + dim - 1] = -tr;
      }
      s.internal.push_back(M);
    }
    return s;
  };

  // --- the evolution ----------------------------------------------------------
  //
  // The shared stepping of viscoelastic_common.hpp: output times and, with
  // -align, the breakpoints on the step grid; left limits at jumps.

  if (root) {
    std::cout << "\nCase " << case_name << ": " << dim << "-D " << kind
              << ", " << layers.size() << " layer(s), load " << load_kind
              << ", " << nb << " branch(es), " << ndofs << " dofs\nScheme "
              << scheme_name << ", " << times.size() << " outputs to "
              << times.back() << (align ? " (breakpoints aligned)" : "")
              << "\n";
  }
  Snapshot elastic_snapshot;
  std::vector<Snapshot> snapshots;
  auto record = [&](const Vector& m, real_t t, int k) {
    Snapshot s = observe(m, t);
    if (k < 0) {
      elastic_snapshot = std::move(s);
      return;
    }
    snapshots.push_back(std::move(s));
    if (root) {
      const Snapshot& b = snapshots.back();
      std::cout << std::setw(12) << std::setprecision(5) << t;
      if (periodic) {
        std::cout << std::setw(16) << std::setprecision(8) << b.W
                  << std::setw(16) << b.U;
      } else {
        std::cout << std::setw(16) << std::setprecision(8) << b.strain[0]
                  << std::setw(16) << b.strain[dim + 1];
      }
      std::cout << "\n";
    }
  };
  const StepOptions step{scheme_name, dt, min_steps, align, rtol, atol};
  const EvolveResult run =
      Evolve(*problem, visco, history, times, step, record);

  // --- the results ------------------------------------------------------------

  auto tensors = [&](const std::vector<std::vector<real_t>>& v) {
    std::string s = "[";
    for (std::size_t k = 0; k < v.size(); k++) {
      s += (k ? ", " : "") + List(v[k]);
    }
    return s + "]";
  };
  auto quantities = [&](std::ostream& os, const Snapshot& s) {
    if (periodic) {
      os << "\"W\": " << Num(s.W) << ", \"U\": " << Num(s.U);
    } else {
      os << "\"strain\": " << List(s.strain)
         << ", \"internal\": " << tensors(s.internal);
    }
  };

  if (root) {
    std::cout << std::setprecision(4) << "\n" << run.steps << " steps";
    if (run.rejected) {
      std::cout << " (+" << run.rejected << " rejected)";
    }
    std::cout << ", " << run.stepping.solves << " stepping solves + "
              << run.observation.solves << " observation, "
              << run.total.assemblies << " assemblies, " << run.total.setups
              << " preconditioner setups, " << run.total.its
              << " iterations, " << run.seconds << " s"
              << (run.ok ? "" : ", SOME SOLVES FAILED") << "\n";

    std::ofstream os(out_file);
    MFEM_VERIFY(os.good(), "Cannot write " << out_file << ".");
    os << "{\n  \"case\": \"" << case_name << "\",\n  \"case_file\": \""
       << case_file << "\",\n  \"dim\": " << dim << ",\n  \"kind\": \""
       << kind << "\",\n  \"load\": \"" << load_kind
       << "\",\n  \"order\": " << order << ",\n  \"nx\": " << nx
       << ",\n  \"nz\": " << nz << ",\n  \"dofs\": " << ndofs
       << ",\n  \"ranks\": " << Mpi::WorldSize()
       << ",\n  \"conform\": " << (conform ? "true" : "false")
       << ",\n  \"rheology\": \""
       << mode << "\",\n  \"branches\": " << nb << ",\n  \"scheme\": \""
       << scheme_name << "\",\n  \"dt\": " << Num(dt)
       << ",\n  \"min_steps\": " << min_steps << ",\n  \"align\": "
       << (align ? "true" : "false") << ",\n  \"rtol\": " << Num(rtol)
       << ",\n  \"atol\": " << Num(atol)
       << ",\n  \"wavenumber\": " << List(wavenumber)
       << ",\n  \"times\": " << List(times) << ",\n  \"elastic\": {";
    quantities(os, elastic_snapshot);
    os << "},\n  \"histories\": [";
    for (std::size_t n = 0; n < snapshots.size(); n++) {
      const Snapshot& s = snapshots[n];
      const OutputCost& c = run.outputs[n];
      os << (n ? "," : "") << "\n    {\"time\": " << Num(s.time) << ", ";
      quantities(os, s);
      os << ", \"solves\": " << c.cost.solves
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
