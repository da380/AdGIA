// ============================================================================
// benchmark_case.hpp
//
// What the drivers of the Love-number benchmark share: a case read from its
// manifest and set up for the formulation chosen by -method (the Eulerian
// LinearQuasiStaticMixedSelfGravitatingProblem for 'dahlen' and 'gauged',
// the referential and slipping classes for the others), the harmonic
// analysis of its solution on the interfaces of the solid, the translation
// that takes a solution to the centre-of-mass frame, and the writing of
// numbers as JSON. The formulations and the cases are described in
// doc/benchmarks.tex, "The Love-number family".
//
// A case is what make_case.py writes: the mesh of the body and its buffer
// shell, the density and the bulk and shear moduli as L2 GridFunctions on
// it, and the manifest saying which layers are solid and which fluid, where
// the interfaces are and what G is in the model's units.
// ============================================================================

#pragma once

#include <mpi.h>

#include <chrono>
#include <cmath>
#include <functional>
#include <optional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

#include "AdGIA.hpp"
#include "relabelling.hpp"

namespace benchmark {

using namespace mfem;
using namespace AdGIA;

inline constexpr real_t kPi = std::numbers::pi_v<real_t>;

using Clock = std::chrono::steady_clock;
using BHC = BoundaryHarmonicCoefficients;
using Problem = LinearQuasiStaticMixedSelfGravitatingProblem;

// The fields of the model a case holds.
inline constexpr const char* kFields[] = {"rho", "kappa", "mu"};

// Where the parts of a case for a number of ranks are (partition_case.cpp).
inline std::string PartsDirectory(const MeshManifest& manifest, int ranks) {
  const std::string& path = manifest.Path();
  const auto slash = path.find_last_of('/');
  return (slash == std::string::npos ? "" : path.substr(0, slash + 1)) +
         "parts_" + std::to_string(ranks);
}

inline double Seconds(Clock::time_point since) {
  return std::chrono::duration<double>(Clock::now() - since).count();
}

// A number as JSON: null when not finite. 17 significant digits make the
// double round-trip exact, so differencing scripts see the solver's
// numbers, not 1-ulp reading noise.
inline std::string Num(real_t x) {
  if (!std::isfinite(x)) {
    return "null";
  }
  std::ostringstream os;
  os << std::setprecision(17) << x;
  return os.str();
}

template <class V>
std::string List(const V& v) {
  std::string s = "[";
  for (int i = 0; i < static_cast<int>(v.size()); i++) {
    s += (i ? ", " : "") + Num(v[i]);
  }
  return s + "]";
}

inline std::string List(const Vector& v) {
  return List(std::vector<real_t>(v.begin(), v.end()));
}

// The density on the fluid side of each interface, from the manifest's
// one-sided values: entry b - 1 for the boundary attribute b, zero for a
// boundary that is not a fluid-solid interface. The interface terms of the
// weak form want the fluid's density on boundary elements of the solid,
// which hold no fluid element to ask; the manifest holds the model's exact
// value there.
inline Vector FluidSideDensity(const MeshManifest& manifest, int size) {
  Vector rho(size);
  rho = 0.0;
  const int first = manifest.Interfaces().front().attribute;
  for (const int fluid : manifest.FluidAttributes()) {
    for (const int b : manifest.FluidSolidInterfaces(fluid)) {
      rho[b - 1] = manifest.Interfaces()[b - first].ValueBeside("rho", fluid);
    }
  }
  return rho;
}

// A scalar field carried on two SubMeshes at once: the slipping methods
// evaluate one set of constitutive coefficients on both the solid and
// the fluid spaces, and a GridFunctionCoefficient is bound to one mesh.
// Dispatched on the mesh of the transformation.
class TwoRegionCoefficient : public Coefficient {
 public:
  TwoRegionCoefficient(GridFunction& a, GridFunction& b)
      : a_(&a), ca_(&a), cb_(&b) {}
  real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    return T.mesh == a_->FESpace()->GetMesh() ? ca_.Eval(T, ip)
                                              : cb_.Eval(T, ip);
  }

 private:
  GridFunction* a_;
  GridFunctionCoefficient ca_, cb_;
};

// The radial functions of one harmonic in one layer: for a field f and the
// harmonic Y_i, the polynomial a(r) of degree modes - 1 that is closest to
// the field in the layer,
//
//   scalar      minimising int (f - a Y_i)^2 dV
//   radial      minimising int (f . x^ - a Y_i)^2 dV
//   tangential  minimising int |f - (f . x^) x^ - a grad_1 Y_i|^2 dV
//
// over the elements of the given attribute. The polynomial is a sum of
// Chebyshev polynomials in the radius taken to [-1, 1] over the layer; what
// is returned are its values at the given radii. For a field that is a
// radial function times the harmonic this recovers the radial function.
enum class RadialPart { Scalar, Radial, Tangential };

inline Vector RadialFunction(ParMesh& mesh, int attribute, real_t r_inner,
                             real_t r_outer, const SurfaceHarmonics& basis,
                             int i, RadialPart part, Coefficient* scalar,
                             VectorCoefficient* vector, const Vector& radii,
                             int modes = 8, int quadrature_order = 12) {
  const int dim = mesh.Dimension();
  DenseMatrix A(modes);
  Vector b(modes), x(dim), Y, f(dim), T_k(modes);
  DenseMatrix gradY;
  A = 0.0;
  b = 0.0;
  auto chebyshev = [&](real_t r, Vector& T) {
    const real_t xi = std::min(
        1.0, std::max(-1.0, (2.0 * r - r_inner - r_outer) /
                                (r_outer - r_inner)));
    T[0] = 1.0;
    if (modes > 1) {
      T[1] = xi;
    }
    for (int k = 2; k < modes; k++) {
      T[k] = 2.0 * xi * T[k - 1] - T[k - 2];
    }
  };
  for (int e = 0; e < mesh.GetNE(); e++) {
    if (mesh.GetAttribute(e) != attribute) {
      continue;
    }
    ElementTransformation* T = mesh.GetElementTransformation(e);
    const IntegrationRule& ir =
        IntRules.Get(T->GetGeometryType(), quadrature_order);
    for (int q = 0; q < ir.GetNPoints(); q++) {
      const IntegrationPoint& ip = ir.IntPoint(q);
      T->SetIntPoint(&ip);
      T->Transform(ip, x);
      const real_t r = x.Norml2(), w = ip.weight * T->Weight();
      basis.EvalWithGradient(x, Y, gradY);
      chebyshev(r, T_k);
      real_t weight = 0.0, value = 0.0;
      if (part == RadialPart::Scalar) {
        weight = Y[i] * Y[i];
        value = scalar->Eval(*T, ip) * Y[i];
      } else {
        vector->Eval(f, *T, ip);
        if (part == RadialPart::Radial) {
          weight = Y[i] * Y[i];
          value = (f * x) / r * Y[i];
        } else {
          for (int c = 0; c < dim; c++) {
            weight += gradY(c, i) * gradY(c, i);
            value += f[c] * gradY(c, i);
          }
        }
      }
      for (int k = 0; k < modes; k++) {
        b[k] += w * value * T_k[k];
        for (int j = 0; j < modes; j++) {
          A(k, j) += w * weight * T_k[k] * T_k[j];
        }
      }
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, A.GetData(), modes * modes,
                MPITypeMap<real_t>::mpi_type, MPI_SUM, mesh.GetComm());
  MPI_Allreduce(MPI_IN_PLACE, b.GetData(), modes,
                MPITypeMap<real_t>::mpi_type, MPI_SUM, mesh.GetComm());
  Vector values(radii.Size());
  values = 0.0;
  if (A.Trace() <= 0.0) {
    return values;
  }
  Vector a(modes);
  DenseMatrixInverse inverse(A);
  inverse.Mult(b, a);
  for (int s = 0; s < radii.Size(); s++) {
    chebyshev(radii[s], T_k);
    values[s] = a * T_k;
  }
  return values;
}

// The analysis of U, V and phi on one interface of the solid.
struct InterfaceAnalysis {
  MeshManifest::Interface interface;
  std::unique_ptr<BHC> radial, tangential, scalar;
  // The gravity of the background at the interface.
  real_t gravity = 0.0;
};

// What a driver asks of a case.
struct CaseOptions {
  const char* manifest = "case.json";
  int order = 2;
  int dtn_degree = 16;
  int lmax = 6;
  int solver = 1;
  real_t rel_tol = 1e-10;
  bool no_fluid_gradient = false;
  bool diagnostics = false;
  bool gauged = false;
  real_t gauge_eps = 1e-2;
  int gauge_refinements = 3;
  const char* cmb = "full";
  const char* method = "dahlen";
  real_t slip_theta = 1e2;
  real_t gs_scale = 1.0;
  int al_iterations = 8;
  real_t sweep_tol = 1e-3;
  bool kkt = false;
  int kkt_order = 0;
  int gauge_kkt = 0;  // 0: penalty; 1: KKT; 2: KKT with the MGW prec
  real_t map_amplitude = 0.0;
  bool map_interp = false;
  real_t map_shift = 0.0;
  int map_shift_interface = 1;
  const char* profiles = "";

  void Add(OptionsParser& args) {
    args.AddOption(&method, "-method", "--method",
                   "The formulation solved: 'dahlen' (Eulerian, fluid "
                   "eliminated; -cmb picks its interface treatment), "
                   "'gauged' (Eulerian, gauged fluid in the displacement "
                   "space), 'referential' (welded gauged referential: bare "
                   "moduli from p0, S_e = -p0 I), 'slip' (broken "
                   "displacement pair, single-valued zeta) or "
                   "'slip_broken' (broken zeta as well; "
                   "doc/slip_interface.tex). The referential methods need "
                   "a case with the p0 field.");
    args.AddOption(&map_amplitude, "-map", "--map-amplitude",
                   "Amplitude of the interior relabelling of the "
                   "relabelled 3-D benchmark (relabelling.hpp): the same "
                   "spherical physical problem described from laterally "
                   "relabelled coordinates, so the reference stays exact "
                   "while every mapped code path acts. Referential "
                   "methods only (referential, slip_broken); needs the "
                   "case's radial_profiles.txt. Zero (default): off.");
    args.AddOption(&map_interp, "-map-interp", "--map-interpolated-f",
                   "-map-exact", "--map-exact-f",
                   "Use the nodal interpolant of the relabelling (the "
                   "discrete change-of-variables mode) instead of the "
                   "exact analytic mapping.");
    args.AddOption(&map_shift, "-map-shift", "--map-shift",
                   "Degree-0 interface shift eps (relabelling.hpp, the "
                   "perturbation family): the named reference "
                   "interface moves radially by eps, the physical model "
                   "being the perturbed spherical one — pair with "
                   "-profiles of THAT model. Methods referential and "
                   "slip_broken only. "
                   "Zero (default): off.");
    args.AddOption(&map_shift_interface, "-map-shift-interface",
                   "--map-shift-interface",
                   "Which interior interface the shift moves (1 = the "
                   "innermost).");
    args.AddOption(&profiles, "-profiles", "--profiles",
                   "Radial-profiles file overriding the case's own: the "
                   "PERTURBED model's profiles for a shift run.");
    args.AddOption(&slip_theta, "-theta", "--slip-theta",
                   "Constraint penalty of the slipping methods (both the "
                   "normal and, broken, the scalar jump).");
    args.AddOption(&al_iterations, "-al", "--al-iterations",
                   "Augmented-Lagrangian iterations per solve (slipping "
                   "methods).");
    args.AddOption(&gs_scale, "-gs-scale", "--gravity-interface-scale",
                   "DIAGNOSTIC (slip_broken): scale on the broken-zeta "
                   "gravity interface form G_Sigma and its vz couplings; "
                   "0 drops them. Changes the physics — for differential "
                   "mapped-vs-unmapped leakage tests only.");
    args.AddOption(&gauge_kkt, "-gauge-kkt", "--gauge-kkt",
                   "Gauged method only: enforce the fluid gauge by the "
                   "exact KKT saddle instead of the epsilon penalty "
                   "(1: diagonal preconditioner, 2: with the "
                   "Murphy-Golub-Wathen composite on the multiplier "
                   "slot; 0: penalty, the default).");
    args.AddOption(&kkt_order, "-kkt-order", "--kkt-multiplier-order",
                   "Polynomial order of the KKT multiplier space "
                   "(0: the potential space's order). Lower orders "
                   "shrink the multiplier block and improve the "
                   "inf-sup margin of equal-order traces.");
    args.AddOption(&kkt, "-kkt", "--kkt", "-no-kkt", "--no-kkt",
                   "Enforce the slip constraint by a monolithic KKT "
                   "multiplier block (single-valued slip only) instead "
                   "of penalty + augmented Lagrangian; -al then bounds "
                   "the gauge refinements (2-3 suffice).");
    args.AddOption(&sweep_tol, "-sweep-tol", "--al-sweep-tolerance",
                   "Loose relative tolerance of the early AL sweeps "
                   "(geometric tightening to -rt, final sweep at -rt; "
                   "SetSweepTolerance). Roughly halves the slip cost at "
                   "a solver-endpoint shift ~1e-3, an order below the "
                   "mesh error. 0: every sweep at full tolerance — "
                   "required for finite-difference (-map-shift) studies "
                   "and strict solver comparisons.");
    args.AddOption(&cmb, "-cmb", "--cmb-approximation",
                   "Fluid-interface treatment of the Dahlen path: 'full' "
                   "(stratified, F1+F2+F3), 'nomass' (drop the fluid mass "
                   "term F1), 'uniform' (nomass with a constant fluid-side "
                   "density, the region's outermost interface value: the "
                   "standard unmeshed-core condition of the GIA codes), "
                   "'winkler' (uniform without the interface potential "
                   "coupling F3: buoyancy alone).");
    args.AddOption(&gauged, "-gauged", "--gauged", "-dahlen", "--dahlen",
                   "Gauged fluid treatment: the fluid layers join the "
                   "displacement SubMesh with their bulk modulus and a gauge "
                   "shear penalty (doc/gauged_fluid.md), instead of Dahlen's "
                   "interface and fluid-mass terms.");
    args.AddOption(&gauge_eps, "-geps", "--gauge-epsilon",
                   "Gauge penalty factor epsilon (gauged fluid).");
    args.AddOption(&gauge_refinements, "-gref", "--gauge-refinements",
                   "Tikhonov refinement steps per solve (gauged fluid).");
    args.AddOption(&manifest, "-c", "--case", "Manifest of the case.");
    args.AddOption(&order, "-o", "--order", "Finite element order.");
    args.AddOption(&dtn_degree, "-deg", "--dtn-degree",
                   "DtN expansion degree.");
    args.AddOption(&lmax, "-lmax", "--max-degree", "Highest harmonic degree.");
    args.AddOption(&solver, "-s", "--solver",
                   "0: Schur-complement CG, 1: block MINRES.");
    args.AddOption(&rel_tol, "-rt", "--rel-tol", "Relative solver tolerance.");
    args.AddOption(&no_fluid_gradient, "-no-fluid-gradient",
                   "--no-fluid-gradient", "-fluid-gradient",
                   "--fluid-gradient",
                   "Set d rho / d Phi_0 to zero in the fluid, as it is for a "
                   "uniform layer, instead of computing it from the fields.");
    args.AddOption(&diagnostics, "-diag", "--diagnostics", "-no-diag",
                   "--no-diagnostics",
                   "Print rigid-mode residuals and potential-block Ritz "
                   "values.");
  }
};

// A case set up for solving. When the case has been partitioned for the
// number of ranks each rank reads its own part; else the mesh and the fields
// are read in serial on every rank and handed to the parallel mesh by the
// partitioning.
class Case {
 public:
  // A rheology to build the Eulerian problem with in place of the elastic
  // one of the case's moduli (the viscoelastic family's Maxwell layers):
  // called once, during construction, with the displacement SubMesh and
  // the case's bulk and shear moduli on it. The rheology returned must
  // outlive the Case; null keeps the elastic one.
  using RheologyFactory = std::function<const Rheology*(
      ParSubMesh& displacement_mesh, Coefficient& kappa, Coefficient& mu)>;

  explicit Case(const CaseOptions& options,
                const RheologyFactory& rheology_factory = {})
      : options_(options), manifest(options.manifest) {
    const bool root = Mpi::Root();
    const auto start = Clock::now();
    MFEM_VERIFY(options.dtn_degree >= options.lmax,
                "The DtN degree must reach lmax.");
    if (root) manifest.Print(std::cout);
    G = manifest.G();
    const int rank = Mpi::WorldRank();
    const std::string parts = PartsDirectory(manifest, Mpi::WorldSize());
    if (std::ifstream(MakeParFilename(parts + "/mesh.", rank)).good()) {
      if (root) {
        std::cout << "Reading the parts in " << parts << "\n";
      }
      std::ifstream mesh_in(MakeParFilename(parts + "/mesh.", rank));
      parent = std::make_unique<ParMesh>(MPI_COMM_WORLD, mesh_in, false, 1,
                                         false);
      elements = parent->GetGlobalNE();
      auto read = [&](const char* name) {
        std::ifstream in(
            MakeParFilename(parts + "/" + name + ".", rank));
        MFEM_VERIFY(in.good(), "A part of " << name << " is missing.");
        return std::make_unique<ParGridFunction>(parent.get(), in);
      };
      rho = read("rho");
      kappa = read("kappa");
      mu = read("mu");
      if (manifest.HasField("p0")) {
        p0 = read("p0");
      }
    } else {
      Mesh serial = manifest.LoadMesh();
      elements = serial.GetNE();
      auto rho_s = manifest.LoadField(serial, "rho");
      auto kappa_s = manifest.LoadField(serial, "kappa");
      auto mu_s = manifest.LoadField(serial, "mu");
      auto p0_s = manifest.HasField("p0") ? manifest.LoadField(serial, "p0")
                                          : nullptr;
      std::unique_ptr<int[]> partitioning(
          serial.GeneratePartitioning(Mpi::WorldSize()));
      parent = std::make_unique<ParMesh>(MPI_COMM_WORLD, serial,
                                         partitioning.get());
      auto distribute = [&](const GridFunction& f) {
        return std::make_unique<ParGridFunction>(parent.get(), &f,
                                                 partitioning.get());
      };
      rho = distribute(*rho_s);
      kappa = distribute(*kappa_s);
      mu = distribute(*mu_s);
      if (p0_s) {
        p0 = distribute(*p0_s);
      }
    }
    dim = parent->Dimension();
    MFEM_VERIFY(dim == 3, "The benchmark is for balls.");

    // The method solved: the Eulerian pair share the
    // LinearQuasiStaticMixedSelfGravitatingProblem, the referential family
    // its own construction below.
    MFEM_VERIFY(!options.gauged || std::string(options.method) == "dahlen" ||
                    std::string(options.method) == "gauged",
                "-gauged chooses the gauged Eulerian method: it cannot be "
                "combined with -method "
                    << options.method << ".");
    method = options.gauged ? "gauged" : options.method;
    MFEM_VERIFY(method == "dahlen" || method == "gauged" ||
                    method == "referential" || method == "slip" ||
                    method == "slip_broken",
                "-method must be dahlen, gauged, referential, slip or "
                "slip_broken.");
    eulerian = method == "dahlen" || method == "gauged";

    // The displacement regions and the material on them: the solid
    // layers, and for the welded treatments of the fluid (gauged,
    // referential) the fluid layers as well (the member keeps the name
    // `solid` as "the displacement SubMesh"; the slipping methods give
    // the fluid its own SubMesh).
    solid_attributes = manifest.SolidAttributes();
    fluid_attributes = manifest.FluidAttributes();
    Array<int> u_attributes(solid_attributes);
    if (method == "gauged" || method == "referential") {
      u_attributes.Append(fluid_attributes);
      u_attributes.Sort();
    }
    solid = std::make_unique<ParSubMesh>(
        ParSubMesh::CreateFromDomain(*parent, u_attributes));
    rho_solid_ = OnSolid(*rho);
    kappa_solid_ = OnSolid(*kappa);
    mu_solid_ = OnSolid(*mu);
    rho_c_ = std::make_unique<GridFunctionCoefficient>(rho_solid_.get());
    kappa_c_ = std::make_unique<GridFunctionCoefficient>(kappa_solid_.get());
    mu_c_ = std::make_unique<GridFunctionCoefficient>(mu_solid_.get());
    rho_parent_c_ = std::make_unique<GridFunctionCoefficient>(rho.get());

    // The mass of the body and the gravity on its surface.
    {
      ParLinearForm m(rho->ParFESpace());
      m.AddDomainIntegrator(new DomainLFIntegrator(*rho_parent_c_));
      m.Assemble();
      real_t local = m.Sum();
      MPI_Allreduce(&local, &mass, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                    MPI_COMM_WORLD);
    }
    radius = manifest.SurfaceRadius();
    gravity = G * mass / (radius * radius);

    fec_ = std::make_unique<H1_FECollection>(options.order, dim);
    fes_u = std::make_unique<ParFiniteElementSpace>(solid.get(), fec_.get(),
                                                    dim);
    fes_phi = std::make_unique<ParFiniteElementSpace>(parent.get(),
                                                      fec_.get());
    displacement_unknowns = fes_u->GlobalTrueVSize();
    potential_unknowns = fes_phi->GlobalTrueVSize();
    if (root) {
      std::cout << "Ranks " << Mpi::WorldSize() << ", elements " << elements
                << ", displacement unknowns " << displacement_unknowns
                << ", potential unknowns " << potential_unknowns << "\nG "
                << G << ", radius " << radius << ", mass " << mass
                << ", surface gravity " << gravity << "\n";
    }

    // The fluid layers: the density on the parent's fluid elements, and on
    // the solid's boundary elements the density of the fluid beyond them.
    const std::string cmb(options.cmb);
    MFEM_VERIFY(cmb == "full" || cmb == "nomass" || cmb == "uniform" ||
                    cmb == "winkler",
                "-cmb must be full, nomass, uniform or winkler.");
    MFEM_VERIFY(method == "dahlen" || cmb == "full",
                "-cmb applies to the Dahlen path only.");
    const int n_bdr = solid->bdr_attributes.Max();
    std::vector<FluidRegion> fluids;
    if (method == "dahlen") {
      fluid_side_ = FluidSideDensity(manifest, n_bdr);
      rho_interface_c_ = std::make_unique<PWConstCoefficient>(fluid_side_);
      for (const int attribute : fluid_attributes) {
        FluidRegion f;
        f.attributes = Array<int>({attribute});
        f.density = rho_parent_c_.get();
        f.interface_density = rho_interface_c_.get();
        f.interface_marker = MeshManifest::Marker(
            manifest.FluidSolidInterfaces(attribute), n_bdr);
        if (options.no_fluid_gradient || cmb != "full") {
          f.density_gradient = &zero_;
        }
        if (cmb == "uniform" || cmb == "winkler") {
          // The constant fluid density of the approximate condition: the
          // fluid-side value at the region's outermost interface (the
          // core-top density for a core).
          const int first = manifest.Interfaces().front().attribute;
          real_t rho_c = 0.0, r_max = -1.0;
          for (const int b : manifest.FluidSolidInterfaces(attribute)) {
            const auto& itf = manifest.Interfaces()[b - first];
            if (itf.radius > r_max) {
              r_max = itf.radius;
              rho_c = itf.ValueBeside("rho", attribute);
            }
          }
          cmb_rho_.push_back(std::make_unique<ConstantCoefficient>(rho_c));
          f.interface_density = cmb_rho_.back().get();
          if (root) {
            std::cout << "CMB approximation '" << cmb << "': fluid layer "
                      << attribute << " with constant interface density "
                      << rho_c << "\n";
          }
        }
        f.interface_potential_coupling = cmb != "winkler";
        fluids.push_back(f);
      }
    }

    if (eulerian) {
      rheology_ = std::make_unique<IsotropicElasticRheology>(dim, *kappa_c_,
                                                             *mu_c_);
      const Rheology* chosen =
          rheology_factory ? rheology_factory(*solid, *kappa_c_, *mu_c_)
                           : nullptr;
      problem = std::make_unique<Problem>(fes_u.get(), fes_phi.get(),
                                          chosen ? *chosen : *rheology_,
                                          *rho_c_, G,
                                          options.dtn_degree, nullptr,
                                          fluids);
      if (method == "gauged" && fluid_attributes.Size() > 0) {
        // The gauge shear scale is the model's own bulk modulus (the
        // fluid's kappa on the fluid layers).
        gauge_marker_.SetSize(solid->attributes.Max());
        gauge_marker_ = 0;
        for (const int a : fluid_attributes) {
          gauge_marker_[a - 1] = 1;
        }
        if (options.gauge_kkt > 0) {
          problem->EnableGaugeKKT(gauge_marker_, *kappa_c_,
                                  options.gauge_kkt == 2);
        } else {
          problem->SetGaugedFluid(gauge_marker_, *kappa_c_,
                                  options.gauge_eps,
                                  options.gauge_refinements);
        }
        if (root) {
          if (options.gauge_kkt > 0) {
            std::cout << "Gauged fluid: exact KKT gauge"
                      << (options.gauge_kkt == 2 ? " (MGW preconditioner)"
                                                 : "")
                      << ".\n";
          } else {
          std::cout << "Gauged fluid: eps " << options.gauge_eps << ", "
                    << options.gauge_refinements << " refinements.\n";
          }
        }
      }
      // A solid layer with fluid all around it turns freely in a
      // spherical model; with the gauged fluid the penalty owns those
      // modes and they must not be projected (doc/gauged_fluid.md).
      if (method == "dahlen") {
        for (const int attribute : solid_attributes) {
          bool enclosed = true;
          for (const auto& f : manifest.Interfaces()) {
            if (f.below == attribute || f.above == attribute) {
              const int other = f.below == attribute ? f.above : f.below;
              enclosed = enclosed && fluid_attributes.Find(other) >= 0;
            }
          }
          if (enclosed) {
            problem->AddRegionRotations(Array<int>({attribute}));
            if (root) {
              std::cout << "Layer " << attribute
                        << " is enclosed by fluid: its rotations are "
                           "projected.\n";
            }
          }
        }
      }
      problem->SetSolverType(options.solver == 0
                                 ? Problem::SolverType::SchurCG
                                 : Problem::SolverType::BlockMINRES);
      problem->SetRelTol(options.rel_tol);
    } else {
      MFEM_VERIFY(!rheology_factory,
                  "A rheology other than the elastic one needs an Eulerian "
                  "method (dahlen or gauged).");
      BuildReferential(options);
    }

    // Harmonic analysis on every interface bounding a solid layer, the
    // surface among them.
    for (const auto& f : manifest.Interfaces()) {
      if (solid_attributes.Find(f.below) < 0 &&
          solid_attributes.Find(f.above) < 0) {
        continue;
      }
      const Array<int> marker =
          MeshManifest::Marker(Array<int>({f.attribute}), n_bdr);
      InterfaceAnalysis an;
      an.interface = f;
      an.radial = std::make_unique<BHC>(*fes_u, marker, options.lmax,
                                        BHC::Component::Radial);
      an.tangential = std::make_unique<BHC>(*fes_u, marker, options.lmax,
                                            BHC::Component::Tangential);
      an.scalar = std::make_unique<BHC>(AnalysisPotentialSpace(), marker,
                                        options.lmax,
                                        BHC::Component::Scalar);
      if (f.attribute == manifest.SurfaceAttribute()) {
        surface = static_cast<int>(analyses.size());
      }
      analyses.push_back(std::move(an));
    }
    MFEM_VERIFY(surface >= 0, "The surface of the body is not solid.");
    // The gravity of the background at the radius of each interface,
    // G M(r) / r^2 with M(r) the mass of the layers within.
    {
      const int n_layers = parent->attributes.Max();
      Vector local(n_layers), layer_mass(n_layers);
      for (int attribute = 1; attribute <= n_layers; attribute++) {
        Array<int> marker(n_layers);
        marker = 0;
        marker[attribute - 1] = 1;
        ParLinearForm m(rho->ParFESpace());
        m.AddDomainIntegrator(new DomainLFIntegrator(*rho_parent_c_), marker);
        m.Assemble();
        local[attribute - 1] = m.Sum();
      }
      MPI_Allreduce(local.GetData(), layer_mass.GetData(), n_layers,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
      for (auto& an : analyses) {
        real_t within = 0.0;
        for (int attribute = 1; attribute <= an.interface.below; attribute++) {
          within += layer_mass[attribute - 1];
        }
        const real_t r = an.interface.radius;
        an.gravity = G * within / (r * r);
      }
    }
    // A mapped run describes a DIFFERENT physical model (the perturbed
    // profiles through the map): its mass and gravities come from the
    // exact profiles at the PHYSICAL interface radii — the base fields'
    // mesh integrals would keep the base model's (a 2 percent error in
    // g at a 0.02 interface shift, fatal to the derivative benchmark).
    if (profiles_) {
      mass = profiles_->EnclosedMass(radius);
      gravity = G * mass / (radius * radius);
      int shifted_attribute = -1;
      if (options.map_shift != 0.0) {
        int k = 0;
        for (const auto& f : manifest.Interfaces()) {
          if (++k == options.map_shift_interface) {
            shifted_attribute = f.attribute;
          }
        }
      }
      for (auto& an : analyses) {
        const real_t r_phys =
            an.interface.radius +
            (an.interface.attribute == shifted_attribute
                 ? options.map_shift
                 : 0.0);
        an.gravity = G * profiles_->EnclosedMass(r_phys) /
                     (r_phys * r_phys);
      }
      if (root) {
        std::cout << "Mapped model: mass " << mass << ", surface gravity "
                  << gravity << "\n";
      }
    }

    // One load and one tidal potential, their coefficients set per solve.
    Vector zero(Basis().Size());
    zero = 0.0;
    sigma = analyses[surface].radial->Expansion(zero, false);
    psi = analyses[surface].scalar->Expansion(zero, true);
    if (eulerian) {
      problem->SetSurfaceLoad(*sigma, analyses[surface].radial->Marker());
      problem->SetTidalPotential(*psi);
    } else {
      ref_problem->SetSurfaceLoad(*sigma,
                                  analyses[surface].radial->Marker());
    }

    if (options.diagnostics && eulerian) {
      const auto residuals = problem->RigidModeResiduals();
      real_t hi = 0.0;
      const real_t lo = fluids.empty()
                            ? 0.0
                            : problem->PotentialBlockMinEigenvalue(40, &hi);
      if (root) {
        if (!fluids.empty()) {
          std::cout << "Potential block Ritz values: " << lo << " .. " << hi
                    << (lo > 0.0 ? "" : "  (INDEFINITE)") << "\n";
        }
        std::cout << "Rigid-mode residuals:";
        for (const auto r : residuals) {
          std::cout << " " << r;
        }
        std::cout << "\n";
      }
    }
    if (options.diagnostics && !eulerian) {
      // The slipping methods' rigid null pairs (common translations,
      // then the independent solid and fluid mapped rotations): their
      // residuals under the full block operator say whether the
      // equilibrium-condition cancellation that the degree-1 response
      // rests on survives the assembly (a mapped run that breaks it
      // leaks into the unforced degree-1 orders).
      if (auto* sp = dynamic_cast<
              LinearQuasiStaticReferentialSelfGravitatingSlipProblem*>(
              ref_problem.get())) {
        const auto residuals = sp->SlipRigidPairResiduals();
        if (root) {
          std::cout << "Slip rigid-pair residuals:";
          for (const auto r : residuals) {
            std::cout << " " << r;
          }
          std::cout << "\n";
        }
      }
    }
    setup_seconds = Seconds(start);
  }

  const SurfaceHarmonics& Basis() const {
    return analyses[surface].radial->Basis();
  }
  const InterfaceAnalysis& Surface() const { return analyses[surface]; }
  bool HasFluid() const { return fluid_attributes.Size() > 0; }

  // What the method supports: the referential family is driven by
  // surface loads only (Solve() refuses a tide), and its potential is
  // meaningful (an observable) on the displacement region, not by layer.
  bool SupportsTide() const { return eulerian; }
  bool SupportsProfiles() const { return eulerian; }

  int OuterIterations() const {
    return eulerian ? problem->LastOuterIterations()
                    : ref_problem->LastOuterIterations();
  }
  int InnerIterations() const {
    return eulerian ? problem->LastInnerIterations() : 0;
  }

  const GridFunction& Displacement() const {
    return eulerian ? problem->Displacement() : ref_problem->Displacement();
  }

  // The potential field analysed on the interfaces: the Eulerian
  // methods' own phi1, or the REFERENTIAL zeta1 on the body, whose
  // interface coefficients the driver converts by the change of
  // variables phi_l = zeta_l - g u_l with the interface's own gravity
  // (exact from the layer masses; a projected grad-zeta0 trace loses an
  // order).
  const GridFunction& AnalysisPotential() const {
    return eulerian ? problem->PotentialOnBody()
                    : ref_problem->PotentialOnBody();
  }
  FiniteElementSpace& AnalysisPotentialSpace() {
    return eulerian ? problem->PotentialSpaceOnBody()
                    : ref_problem->PotentialSpaceOnBody();
  }
  // True when AnalysisPotential() is the referential zeta1, whose
  // coefficients want the change of variables.
  bool PotentialIsReferential() const { return !eulerian; }

  // Solve for the surface load or the tidal potential of the given
  // coefficients; returns the solver's convergence.
  bool Solve(const Vector& coefficients, bool load) {
    Vector zero(coefficients.Size());
    zero = 0.0;
    sigma->SetCoefficients(load ? coefficients : zero);
    psi->SetCoefficients(load ? zero : coefficients);
    if (eulerian) {
      problem->AssembleForce(0.0);
      return problem->Solve();
    }
    MFEM_VERIFY(load,
                "The referential methods solve the load problems only.");
    // Independent forcings: do not warm-start across degrees (the gauge
    // refinement would carry the previous gauge component forward).
    ref_problem->ResetSolution();
    ref_problem->AssembleForce(0.0);
    return ref_problem->Solve();
  }

  // Forget the previous solution, so that the next solve starts cold: the
  // referential classes' own reset; the Eulerian class keeps its warm start
  // privately and has none, but a solve with no forcing leaves it at zero
  // (MINRES and CG both return zero, without iterating, for a zero
  // right-hand side, and the gauge refinement of a zero solution is zero).
  void ResetSolution() {
    if (eulerian) {
      Vector zero(Basis().Size());
      zero = 0.0;
      Solve(zero, true);
    } else {
      ref_problem->ResetSolution();
    }
  }

  // The load's own potential on the displacement region, for the k'
  // normalisation: the Eulerian classes' member solve, or the same
  // Laplace-DtN problem assembled standalone (identical forms, the
  // fluid mass never enters either).
  bool SolveLoadPotential(GridFunction& phi_body) {
    if (eulerian) {
      return problem->SolveLoadPotential(phi_body);
    }
    if (!load_cg_) {
      SetupLoadPotential();
    }
    ParLinearForm b(fes_phi.get());
    Array<int> marker(parent->bdr_attributes.Max());
    marker = 0;
    marker[manifest.SurfaceAttribute() - 1] = 1;
    b.AddBoundaryIntegrator(new BoundaryLFIntegrator(*sigma), marker);
    b.Assemble();
    Vector B(fes_phi->GetTrueVSize()), Phi(fes_phi->GetTrueVSize());
    b.ParallelAssemble(B);
    B *= -1.0;
    Phi = 0.0;
    load_cg_->Mult(B, Phi);
    load_phi_ball_->SetFromTrueDofs(Phi);
    auto* body = dynamic_cast<ParGridFunction*>(&phi_body);
    MFEM_VERIFY(body, "SolveLoadPotential: a parallel field.");
    ParSubMesh::Transfer(*load_phi_ball_, *body);
    return load_cg_->GetConverged();
  }

  // The rigid translation that takes the solution to the frame of the
  // centre of mass of the body and its load, in which the potential
  // perturbation outside has no part of degree one: the coefficients d of
  // d . x^ = sum_m d_m Y_1m, from the degree-one coefficients of phi on the
  // surface. A translation d adds d to U and to V at degree one and takes
  // g(r) d from phi, with g the gravity at that radius.
  Vector CentreOfMassShift(const Vector& phi_surface) const {
    const auto& basis = Basis();
    Vector d(basis.Size());
    d = 0.0;
    if (basis.MaxDegree() >= 1) {
      for (int m = -1; m <= 1; m++) {
        const int i = basis.Index(1, m);
        d[i] = phi_surface[i] / gravity;
      }
    }
    return d;
  }

  // The vector t with t . x^ = sum_m d_m Y_1m(x^).
  Vector TranslationVector(const Vector& d) const {
    const auto& basis = Basis();
    Vector t(dim), e(dim), Y;
    t = 0.0;
    if (basis.MaxDegree() < 1) {
      return t;
    }
    for (int j = 0; j < dim; j++) {
      e = 0.0;
      e[j] = 1.0;
      basis.Eval(e, Y);
      for (int m = -1; m <= 1; m++) {
        const int i = basis.Index(1, m);
        t[j] += d[i] * Y[i];
      }
    }
    return t;
  }

  // The radial functions U, V and phi of the harmonic i of a solution, in
  // every layer of the mesh, at `samples` radii of each: the displacement
  // in the solid layers, the potential in all of them and in the buffer.
  struct LayerProfile {
    int attribute = 0;
    Vector radius, u, v, phi;
  };

  std::vector<LayerProfile> Profiles(const GridFunction& u,
                                     const GridFunction& phi, int i,
                                     int samples = 17) {
    VectorGridFunctionCoefficient u_c(&u);
    GridFunctionCoefficient phi_c(&phi);
    const auto& basis = Basis();
    std::vector<LayerProfile> out;
    for (const auto& layer : manifest.Layers()) {
      LayerProfile p;
      p.attribute = layer.attribute;
      p.radius.SetSize(samples);
      for (int s = 0; s < samples; s++) {
        p.radius[s] = layer.r_inner +
                      (layer.r_outer - layer.r_inner) * s / (samples - 1.0);
      }
      p.phi = RadialFunction(*parent, layer.attribute, layer.r_inner,
                             layer.r_outer, basis, i, RadialPart::Scalar,
                             &phi_c, nullptr, p.radius);
      if (solid_attributes.Find(layer.attribute) >= 0) {
        p.u = RadialFunction(*solid, layer.attribute, layer.r_inner,
                             layer.r_outer, basis, i, RadialPart::Radial,
                             nullptr, &u_c, p.radius);
        if (basis.Degree(i) > 0) {
          p.v = RadialFunction(*solid, layer.attribute, layer.r_inner,
                               layer.r_outer, basis, i,
                               RadialPart::Tangential, nullptr, &u_c,
                               p.radius);
        } else {
          p.v.SetSize(samples);
          p.v = 0.0;
        }
      }
      out.push_back(std::move(p));
    }
    return out;
  }

  // The gravity of the background at the radii of the profiles, from the
  // radial derivative of the background potential.
  void ProfileGravity(std::vector<LayerProfile>& profiles,
                      std::vector<Vector>& gravity) {
    GradientGridFunctionCoefficient grad(&problem->BackgroundPotential());
    gravity.clear();
    for (const auto& p : profiles) {
      const auto& layer = manifest.Layers()[p.attribute - 1];
      Vector g = RadialFunction(*parent, layer.attribute, layer.r_inner,
                                layer.r_outer, Basis(), 0,
                                RadialPart::Radial, nullptr, &grad, p.radius);
      g *= 1.0 / std::sqrt(4.0 * kPi);
      gravity.push_back(g);
    }
  }

  // Common entries of a results file (no collective call: the root alone
  // writes).
  void WriteHeader(std::ostream& os) const {
    os << "  \"case\": \"" << options_.manifest << "\",\n  \"ranks\": "
       << Mpi::WorldSize() << ",\n  \"order\": " << options_.order
       << ",\n  \"dtn_degree\": " << options_.dtn_degree
       << ",\n  \"rel_tol\": " << Num(options_.rel_tol)
       << ",\n  \"solver\": \""
       << (options_.solver == 0 ? "schur_cg" : "block_minres")
       << "\",\n  \"method\": \"" << method
       << "\",\n  \"fluid_treatment\": \""
       // dahlen keeps Dahlen's fluid; the gauged Eulerian and the welded
       // referential methods gauge it; the slip pair lets it slip
       << (method == "dahlen"       ? "dahlen"
           : method == "slip" || method == "slip_broken" ? "slip"
                                                         : "gauged")
       << "\""
       << ",\n  \"cmb\": \"" << options_.cmb << "\""
       << (options_.map_amplitude != 0.0
               ? ",\n  \"map_amplitude\": " + Num(options_.map_amplitude) +
                     ",\n  \"map_mode\": \"" +
                     (options_.map_interp ? "interpolated" : "exact") + "\""
               : std::string())
       << (options_.map_shift != 0.0
               ? ",\n  \"map_shift\": " + Num(options_.map_shift) +
                     ",\n  \"map_shift_interface\": " +
                     std::to_string(options_.map_shift_interface)
               : std::string())
       << (method == "slip" || method == "slip_broken"
               ? ",\n  \"slip_theta\": " + Num(options_.slip_theta) +
                     ",\n  \"al_iterations\": " +
                     std::to_string(options_.al_iterations) +
                     ",\n  \"sweep_tol\": " +
                     Num(options_.map_shift != 0.0 ? 0.0
                                                   : options_.sweep_tol) +
                     ",\n  \"kkt\": " +
                     std::string(options_.kkt ? "true" : "false")
               : std::string())
       << (method != "dahlen"
               ? ",\n  \"gauge_epsilon\": " + Num(options_.gauge_eps) +
                     ",\n  \"gauge_refinements\": " +
                     std::to_string(options_.gauge_refinements)
               : std::string())
       << ",\n  \"elements\": " << elements
       << ",\n  \"displacement_unknowns\": " << displacement_unknowns
       << ",\n  \"potential_unknowns\": " << potential_unknowns
       << ",\n  \"G\": " << Num(G) << ",\n  \"radius\": " << Num(radius)
       << ",\n  \"mass\": " << Num(mass) << ",\n  \"surface_gravity\": "
       << Num(gravity) << ",\n  \"setup_seconds\": " << Num(setup_seconds)
       << ",\n  \"surface\": " << surface << ",\n  \"interfaces\": [";
    for (std::size_t j = 0; j < analyses.size(); j++) {
      const auto& f = analyses[j].interface;
      os << (j ? ", " : "") << "\n    {\"attribute\": " << f.attribute
         << ", \"name\": \"" << f.name << "\", \"radius\": " << Num(f.radius)
         << ", \"measured_radius\": " << Num(analyses[j].radial->Radius())
         << ", \"gravity\": " << Num(analyses[j].gravity) << "}";
    }
    os << "]";
  }

 private:
  std::unique_ptr<ParGridFunction> OnSolid(const ParGridFunction& f) {
    spaces_.push_back(std::make_unique<ParFiniteElementSpace>(
        solid.get(), f.ParFESpace()->FEColl()));
    auto g = std::make_unique<ParGridFunction>(spaces_.back().get());
    ParSubMesh::Transfer(f, *g);
    return g;
  }

  std::unique_ptr<ParGridFunction> OnMesh(const ParGridFunction& f,
                                          ParSubMesh& sub) {
    spaces_.push_back(std::make_unique<ParFiniteElementSpace>(
        &sub, f.ParFESpace()->FEColl()));
    auto g = std::make_unique<ParGridFunction>(spaces_.back().get());
    ParSubMesh::Transfer(f, *g);
    return g;
  }

  // The welded referential and the two slipping methods
  // (doc/gravitating_elasticity.md, doc/slip_interface.tex): bare moduli
  // from the hydrostatic pressure p0, S_e = -p0 I, phi_e = id.
  void BuildReferential(const CaseOptions& options) {
    const bool root = Mpi::Root();
    MFEM_VERIFY(p0,
                "The referential methods need the p0 field: re-make the "
                "case (make_case.py exports it).");
    const bool slip = method == "slip" || method == "slip_broken";

    // The buffer: every parent attribute that is not a model layer.
    Array<int> buffer_attributes;
    for (int a = 1; a <= parent->attributes.Max(); a++) {
      if (solid_attributes.Find(a) < 0 && fluid_attributes.Find(a) < 0) {
        buffer_attributes.Append(a);
      }
    }
    MFEM_VERIFY(buffer_attributes.Size() > 0,
                "The referential methods need the buffer shell.");
    buffer_sub_ = std::make_unique<ParSubMesh>(
        ParSubMesh::CreateFromDomain(*parent, buffer_attributes));
    fes_buffer_ = std::make_unique<ParFiniteElementSpace>(
        buffer_sub_.get(), fec_.get(), dim);
    Vector bb_min, bb_max;
    parent->GetBoundingBox(bb_min, bb_max);
    const real_t r_out = bb_max.Normlinf();

    // The relabelling of the mapped benchmark: built before the
    // coefficients, which it composes.
    const bool shifted = options.map_shift != 0.0;
    const bool mapped = options.map_amplitude != 0.0 || shifted;
    Diffeomorphism* map_use = nullptr;
    if (mapped) {
      MFEM_VERIFY(method == "referential" || method == "slip_broken",
                  "-map: the mapped benchmark runs through the "
                  "referential organisations that are assembled mapped "
                  "(referential, slip_broken); the single-valued slip "
                  "refuses maps by design and the Eulerian classes are "
                  "unmapped.");
      MFEM_VERIFY(!(shifted && options.map_amplitude != 0.0),
                  "-map and -map-shift are separate benchmarks.");
      const std::string& mpath = manifest.Path();
      const auto slash = mpath.find_last_of('/');
      const std::string dir =
          slash == std::string::npos ? std::string()
                                     : mpath.substr(0, slash + 1);
      profiles_ = std::make_unique<RadialProfiles>(
          std::string(options.profiles).empty()
              ? dir + "radial_profiles.txt"
              : std::string(options.profiles));
      // The map's REFERENCE boundaries are the mesh's own: the base
      // case's, not the (possibly perturbed) profiles'.
      base_profiles_ =
          std::make_unique<RadialProfiles>(dir + "radial_profiles.txt");
      xi_analytic_ = std::make_unique<CallableDiffeomorphism>(
          shifted ? InterfaceShift(dim, base_profiles_->Boundaries(),
                                   options.map_shift_interface,
                                   options.map_shift)
                  : InteriorRelabelling(dim, base_profiles_->Boundaries(),
                                        options.map_amplitude));
      // Shift maps MUST run interpolated: the exact analytic map's
      // gradient is discontinuous AT the moving interface, and the
      // one-sided interface kernels evaluate it on the wrong side
      // there (systematically so on curved facets, whose quadrature
      // points sit O(h^2) off the analytic radius), which slows the
      // augmented-Lagrangian sweeps and pollutes the Love numbers. The
      // per-SubMesh interpolant is side-consistent by construction
      // (benchmarks/perturbation/README.md; doc/benchmarks.tex, "The
      // perturbation family").
      bool interp = options.map_interp;
      if (shifted && !interp) {
        if (root) {
          std::cout << "map-shift: forcing interpolated F (exact "
                       "branching maps evaluate their interface kink "
                       "on the wrong side; see perturbation/README)."
                    << std::endl;
        }
        interp = true;
      }
      if (interp) {
        xi_interp_ = std::make_unique<MultiMeshDiffeomorphism>(
            *xi_analytic_, *parent);
        xi_interp_->AddMesh(*solid);
        xi_interp_->AddMesh(*buffer_sub_);
        map_use = xi_interp_.get();
      } else {
        map_use = xi_analytic_.get();
      }
      map_use_ = map_use;
      // Seatbelts: identity on and outside the DtN sphere (the layer's
      // convention) and on every interface (this benchmark's own: the
      // physical problem must stay the referenced spherical one and the
      // interface analyses untouched).
      {
        auto outer_marker = ExternalBoundaryMarker(parent.get());
        const real_t d_out =
            MaxIdentityDeviation(*xi_analytic_, *parent, outer_marker);
        Array<int> itf_marker(parent->bdr_attributes.Max());
        itf_marker = 0;
        int k = 0;
        for (const auto& f : manifest.Interfaces()) {
          k++;
          if (shifted && k == options.map_shift_interface) {
            continue;  // the shifted interface moves by design
          }
          itf_marker[f.attribute - 1] = 1;
        }
        const real_t d_itf =
            MaxIdentityDeviation(*xi_analytic_, *parent, itf_marker);
        // The interface check sees the DISCRETE facets: quadrature
        // points sit off the exact spheres by the geometric error. The
        // interior relabelling's bump vanishes QUADRATICALLY there, so
        // its floor is that error squared (~1e-8); the piecewise-LINEAR
        // shift map only reaches the identity with a slope jump
        // eps / (layer width), so its floor is the facet error times
        // that jump (~1e-5 at eps = 0.02 on a coarse mesh).
        const real_t itf_floor = shifted ? 1e-4 : 1e-6;
        MFEM_VERIFY(d_out < 1e-12 && d_itf < itf_floor,
                    "-map: the relabelling must be the identity on the "
                    "sphere and on every unshifted interface (deviations "
                        << d_out << ", " << d_itf << ").");
        if (Mpi::Root()) {
          if (shifted) {
            std::cout << "Interface shift: eps " << options.map_shift
                      << " at interface " << options.map_shift_interface
                      << ", other interfaces identity to " << d_itf
                      << "\n";
          } else {
            std::cout << "Relabelled: amplitude " << options.map_amplitude
                      << (options.map_interp ? ", interpolated F"
                                             : ", exact F")
                      << ", interface identity to " << d_itf << "\n";
          }
        }
      }
    }

    // The constitutive coefficients: on the displacement SubMesh for the
    // welded method, on the solid AND fluid SubMeshes (mesh-dispatched)
    // for the slipping ones; in the mapped benchmark, the exact radial
    // profiles composed with the relabelling (position-based, valid on
    // every SubMesh) with the density carrying the Jacobian.
    p0_solid_ = OnSolid(*p0);
    p0_c_ = std::make_unique<GridFunctionCoefficient>(p0_solid_.get());
    Coefficient *kappa_use = kappa_c_.get(), *mu_use = mu_c_.get(),
                *p0_use = p0_c_.get(), *rho_use = rho_c_.get();
    if (mapped) {
      auto profile = [this](const char* name) {
        return [this, name](const Vector& y) {
          return profiles_->Eval(name, y.Norml2());
        };
      };
      comp_rho_ = std::make_unique<TransformedFunctionCoefficient>(
          *map_use, profile("rho"));
      comp_kappa_ = std::make_unique<TransformedFunctionCoefficient>(
          *map_use, profile("kappa"));
      comp_mu_ = std::make_unique<TransformedFunctionCoefficient>(
          *map_use, profile("mu"));
      comp_p0_ = std::make_unique<TransformedFunctionCoefficient>(
          *map_use, profile("p0"));
      map_jac_ = std::make_unique<JacobianCoefficient>(*map_use);
      rho_tilde_ =
          std::make_unique<ProductCoefficient>(*map_jac_, *comp_rho_);
      kappa_use = comp_kappa_.get();
      mu_use = comp_mu_.get();
      p0_use = comp_p0_.get();
      rho_use = rho_tilde_.get();
    }
    if (slip) {
      MFEM_VERIFY(
          fluid_attributes.Size() == 1 &&
              manifest.FluidSolidInterfaces(fluid_attributes[0]).Size() == 1,
          "The slipping methods support one fluid core inside a solid "
          "shell (nested shells are not supported).");
      fluid_sub_ = std::make_unique<ParSubMesh>(
          ParSubMesh::CreateFromDomain(*parent, fluid_attributes));
      fes_f_ = std::make_unique<ParFiniteElementSpace>(fluid_sub_.get(),
                                                       fec_.get(), dim);
    }
    if (slip && mapped) {
      kappa_fluid_c_dispatch_ = comp_kappa_.get();
      if (xi_interp_) {
        xi_interp_->AddMesh(*fluid_sub_);
      }
    }
    if (slip && !mapped) {
      rho_fluid_ = OnMesh(*rho, *fluid_sub_);
      kappa_fluid_ = OnMesh(*kappa, *fluid_sub_);
      mu_fluid_ = OnMesh(*mu, *fluid_sub_);
      p0_fluid_ = OnMesh(*p0, *fluid_sub_);
      two_rho_ = std::make_unique<TwoRegionCoefficient>(*rho_solid_,
                                                        *rho_fluid_);
      two_kappa_ = std::make_unique<TwoRegionCoefficient>(*kappa_solid_,
                                                          *kappa_fluid_);
      two_mu_ = std::make_unique<TwoRegionCoefficient>(*mu_solid_,
                                                       *mu_fluid_);
      two_p0_ = std::make_unique<TwoRegionCoefficient>(*p0_solid_,
                                                       *p0_fluid_);
      kappa_use = two_kappa_.get();
      mu_use = two_mu_.get();
      p0_use = two_p0_.get();
      rho_use = two_rho_.get();
      kappa_fluid_c_ =
          std::make_unique<GridFunctionCoefficient>(kappa_fluid_.get());
      kappa_fluid_c_dispatch_ = kappa_fluid_c_.get();
    }

    id_map_ = std::make_unique<IdentityDiffeomorphism>(dim);
    C_eff_.emplace(IsotropicElasticTensorCoefficient::FromBulkModulus(
        dim, *kappa_use, *mu_use));
    C_bare_ = std::make_unique<BareElasticTensorCoefficient>(dim, *C_eff_,
                                                             *p0_use);
    neg_p0_ = std::make_unique<ProductCoefficient>(minus_one_, *p0_use);
    id_mat_ = std::make_unique<IdentityMatrixCoefficient>(dim);
    S_e_ = std::make_unique<ScalarMatrixProductCoefficient>(*neg_p0_,
                                                            *id_mat_);
    MatrixCoefficient* C_final = C_bare_.get();
    MatrixCoefficient* S_final = S_e_.get();
    Diffeomorphism* phi_e = id_map_.get();
    if (mapped) {
      // The pulled-back constitutive state: C~ and S~ from the composed
      // (bare) tensors, phi_e the relabelling (doc/mappings.md).
      C_rel_ = std::make_unique<RelabelledElasticTensorCoefficient>(
          dim, *C_bare_, *map_use);
      S_rel_ = std::make_unique<PullbackStressCoefficient>(dim, *S_e_,
                                                           *map_use);
      C_final = C_rel_.get();
      S_final = S_rel_.get();
      phi_e = map_use;
    }
    ref_rheology_ = std::make_unique<ReferentialElasticRheology>(
        dim, *C_final, *S_final, *phi_e);

    if (!slip) {
      ref_problem = std::make_unique<LinearQuasiStaticReferentialSelfGravitatingProblem>(
          fes_u.get(), fes_phi.get(), *ref_rheology_, *rho_use, G,
          options.dtn_degree);
      Evac_ = NewRadialVacuumExtension(*fes_u, *fes_buffer_, radius, r_out);
      ref_problem->SetPrescribedVacuumExtension(*fes_buffer_, *Evac_);
      if (fluid_attributes.Size() > 0) {
        gauge_marker_.SetSize(solid->attributes.Max());
        gauge_marker_ = 0;
        for (const int a : fluid_attributes) {
          gauge_marker_[a - 1] = 1;
        }
        ref_problem->SetGaugedFluid(gauge_marker_, *kappa_c_,
                                    options.gauge_eps,
                                    options.gauge_refinements);
      }
      if (root) {
        std::cout << "Referential (welded, gauged fluid): eps "
                  << options.gauge_eps << ", "
                  << options.gauge_refinements << " refinements.\n";
      }
    } else {
      // The interface pressure and the interface marker on the solid
      // SubMesh, from the manifest's one-sided values.
      const int fluid_attr = fluid_attributes[0];
      const int b_itf = manifest.FluidSolidInterfaces(fluid_attr)[0];
      const int first = manifest.Interfaces().front().attribute;
      const auto& itf = manifest.Interfaces()[b_itf - first];
      const int n_bdr = solid->bdr_attributes.Max();
      Vector pi_values(n_bdr);
      pi_values = 0.0;
      pi_values[b_itf - 1] = itf.ValueBeside("p0", fluid_attr);
      pi_c_ = std::make_unique<PWConstCoefficient>(pi_values);
      Coefficient* pi_use = pi_c_.get();
      if (mapped) {
        // The interface pressure of the (possibly shifted) PHYSICAL
        // interface: the composed p0 profile at |phi_e(x)| — equal to
        // the manifest value when the map fixes the interface.
        pi_use = comp_p0_.get();
      }
      interface_marker_ =
          MeshManifest::Marker(Array<int>({b_itf}), n_bdr);
      if (root) {
        std::cout << "Slipping interface " << itf.name << " at r = "
                  << itf.radius << ", pi = " << pi_values[b_itf - 1]
                  << (mapped ? " (composed profile used)" : "")
                  << ", theta = " << options.slip_theta << ", "
                  << options.al_iterations << " AL iterations"
                  << (method == "slip_broken" ? ", broken zeta" : "")
                  << ".\n";
      }

      auto slip_problem =
          std::make_unique<LinearQuasiStaticReferentialSelfGravitatingSlipProblem>(
              fes_u.get(), fes_f_.get(), fes_phi.get(), *ref_rheology_,
              *rho_use, *pi_use, interface_marker_, G, options.dtn_degree);
      Evac_ = NewRadialVacuumExtension(*fes_u, *fes_buffer_, radius, r_out);
      slip_problem->SetPrescribedVacuumExtension(*fes_buffer_, *Evac_);
      slip_problem->SetFluidGauge(*kappa_fluid_c_dispatch_,
                                  options.gauge_eps);
      slip_problem->SetConstraint(options.slip_theta,
                                  options.al_iterations);
      if (options.gs_scale != 1.0) {
        slip_problem->ScaleBrokenGravityInterface(options.gs_scale);
        if (root) {
          std::cout << "DIAGNOSTIC: G_Sigma scaled by " << options.gs_scale
                    << "\n";
        }
      }
      // Shift (finite-difference) runs need the reproducible endpoint:
      // every sweep at full tolerance, whatever -sweep-tol says.
      slip_problem->SetSweepTolerance(
          options.map_shift != 0.0 ? 0.0 : options.sweep_tol);
      if (options.kkt) {
        MFEM_VERIFY(method == "slip",
                    "-kkt: single-valued slip only.");
        auto* solid_mesh = static_cast<ParMesh*>(fes_u->GetMesh());
        if (options.kkt_order > 0) {
          fec_lam_ = std::make_unique<H1_FECollection>(
              options.kkt_order, solid_mesh->Dimension());
          fes_lam_bench_ = std::make_unique<ParFiniteElementSpace>(
              solid_mesh, fec_lam_.get());
        } else {
          fes_lam_bench_ = std::make_unique<ParFiniteElementSpace>(
              solid_mesh, fes_phi->FEColl());
        }
        slip_problem->EnableKKT(fes_lam_bench_.get());
      }
      if (method == "slip") {
        Ef_ = NewRadialFluidExtension(*fes_u, *fes_f_, itf.radius);
        slip_problem->SetFluidExtension(*Ef_);
      } else {
        Array<int> outer_attributes(solid_attributes);
        outer_attributes.Append(buffer_attributes);
        outer_attributes.Sort();
        outer_sub_ = std::make_unique<ParSubMesh>(
            ParSubMesh::CreateFromDomain(*parent, outer_attributes));
        if (xi_interp_) {
          xi_interp_->AddMesh(*outer_sub_);
        }
        fes_zo_ = SubMeshDofInjection::MakeShadowSpace(
            *static_cast<ParFiniteElementSpace*>(fes_phi.get()),
            *outer_sub_);
        slip_problem->EnableBrokenZeta(fes_zo_.get(), options.slip_theta);
      }
      ref_problem = std::move(slip_problem);
    }
    ref_problem->SetRelTol(options.rel_tol);
  }

  // The Laplace-DtN solve of the load's own potential, standalone (the
  // referential classes have no member for it): (K + DtN) Phi / 4 pi G
  // = -(sigma, chi)_surface, as the Eulerian classes solve it.
  void SetupLoadPotential() {
    auto* pfes = static_cast<ParFiniteElementSpace*>(fes_phi.get());
    load_dtn_ = std::make_unique<PoissonDtNOperator>(
        pfes->GetComm(), pfes, options_.dtn_degree);
    load_dtn_->Assemble();
    load_dtn_rap_ = std::make_unique<RAPOperator>(load_dtn_->RAP());
    load_k_ = std::make_unique<ParBilinearForm>(pfes);
    load_k_->AddDomainIntegrator(new DiffusionIntegrator());
    load_k_->Assemble();
    Array<int> empty;
    load_k_->FormSystemMatrix(empty, load_K_);
    const real_t c = 1.0 / (4.0 * kPi * G);
    load_A_ = std::make_unique<SumOperator>(load_K_.Ptr(), c,
                                            load_dtn_rap_.get(), c, false,
                                            false);
    // Precondition with the SHIFTED Laplacian: AMG on the singular K
    // alone can go indefinite (a failed k' normalisation looks like
    // phi_s = 0 and infinite Love numbers).
    load_kshift_ = std::make_unique<ParBilinearForm>(pfes);
    load_kshift_->AddDomainIntegrator(new DiffusionIntegrator());
    load_shift_.constant = 1e-3;
    load_kshift_->AddDomainIntegrator(new MassIntegrator(load_shift_));
    load_kshift_->Assemble();
    load_kshift_->FormSystemMatrix(empty, load_Kshift_);
    auto amg = std::make_unique<HypreBoomerAMG>(
        *load_Kshift_.As<HypreParMatrix>());
    amg->SetPrintLevel(0);
    load_prec_ = std::move(amg);
    load_cg_ = std::make_unique<CGSolver>(pfes->GetComm());
    load_cg_->SetOperator(*load_A_);
    load_cg_->SetPreconditioner(*load_prec_);
    load_cg_->SetRelTol(1e-12);
    load_cg_->SetAbsTol(0.0);
    load_cg_->SetMaxIter(10000);
    load_cg_->SetPrintLevel(0);
    load_cg_->iterative_mode = false;
    load_phi_ball_ = std::make_unique<ParGridFunction>(pfes);
  }

  CaseOptions options_;

 public:
  MeshManifest manifest;
  real_t G = 0.0, mass = 0.0, radius = 0.0, gravity = 0.0;
  int dim = 3;
  long long elements = 0;
  HYPRE_BigInt displacement_unknowns = 0, potential_unknowns = 0;
  double setup_seconds = 0.0;
  Array<int> solid_attributes, fluid_attributes;
  std::string method;
  bool eulerian = true;
  std::unique_ptr<ParMesh> parent;
  std::unique_ptr<ParSubMesh> solid;
  std::unique_ptr<ParGridFunction> rho, kappa, mu, p0;
  std::unique_ptr<ParFiniteElementSpace> fes_u, fes_phi;
  std::unique_ptr<Problem> problem;
  std::unique_ptr<LinearQuasiStaticReferentialSelfGravitatingProblem> ref_problem;
  std::vector<InterfaceAnalysis> analyses;
  int surface = -1;
  std::unique_ptr<HarmonicExpansionCoefficient> sigma, psi;

 private:
  std::vector<std::unique_ptr<ParFiniteElementSpace>> spaces_;
  std::unique_ptr<ParGridFunction> rho_solid_, kappa_solid_, mu_solid_;
  std::unique_ptr<GridFunctionCoefficient> rho_c_, kappa_c_, mu_c_,
      rho_parent_c_;
  Vector fluid_side_;
  std::vector<std::unique_ptr<ConstantCoefficient>> cmb_rho_;
  Array<int> gauge_marker_;
  std::unique_ptr<PWConstCoefficient> rho_interface_c_;
  ConstantCoefficient zero_{0.0};
  std::unique_ptr<H1_FECollection> fec_;
  std::unique_ptr<IsotropicElasticRheology> rheology_;

  // The referential family (BuildReferential).
  std::unique_ptr<ParSubMesh> buffer_sub_, fluid_sub_, outer_sub_;
  std::unique_ptr<ParFiniteElementSpace> fes_buffer_, fes_f_, fes_zo_,
      fes_lam_bench_;
  std::unique_ptr<H1_FECollection> fec_lam_;
  std::unique_ptr<ParGridFunction> p0_solid_, p0_fluid_, rho_fluid_,
      kappa_fluid_, mu_fluid_;
  std::unique_ptr<GridFunctionCoefficient> p0_c_, kappa_fluid_c_;
  Coefficient* kappa_fluid_c_dispatch_ = nullptr;
  // The relabelled (mapped) benchmark.
  std::unique_ptr<RadialProfiles> profiles_, base_profiles_;
  std::unique_ptr<CallableDiffeomorphism> xi_analytic_;
  std::unique_ptr<MultiMeshDiffeomorphism> xi_interp_;
  Diffeomorphism* map_use_ = nullptr;
  std::unique_ptr<TransformedFunctionCoefficient> comp_rho_, comp_kappa_,
      comp_mu_, comp_p0_;
  std::unique_ptr<JacobianCoefficient> map_jac_;
  std::unique_ptr<ProductCoefficient> rho_tilde_;
  std::unique_ptr<RelabelledElasticTensorCoefficient> C_rel_;
  std::unique_ptr<PullbackStressCoefficient> S_rel_;
  std::unique_ptr<TwoRegionCoefficient> two_rho_, two_kappa_, two_mu_,
      two_p0_;
  std::unique_ptr<IdentityDiffeomorphism> id_map_;
  std::optional<IsotropicElasticTensorCoefficient> C_eff_;
  std::unique_ptr<BareElasticTensorCoefficient> C_bare_;
  ConstantCoefficient minus_one_{-1.0};
  std::unique_ptr<ProductCoefficient> neg_p0_;
  std::unique_ptr<IdentityMatrixCoefficient> id_mat_;
  std::unique_ptr<ScalarMatrixProductCoefficient> S_e_;
  std::unique_ptr<ReferentialElasticRheology> ref_rheology_;
  std::unique_ptr<PWConstCoefficient> pi_c_;
  Array<int> interface_marker_;
  std::unique_ptr<HypreParMatrix> Evac_, Ef_;

  // The standalone load-potential solve (SetupLoadPotential).
  std::unique_ptr<PoissonDtNOperator> load_dtn_;
  std::unique_ptr<RAPOperator> load_dtn_rap_;
  std::unique_ptr<ParBilinearForm> load_k_, load_kshift_;
  OperatorHandle load_K_, load_Kshift_;
  ConstantCoefficient load_shift_{1e-3};
  std::unique_ptr<SumOperator> load_A_;
  std::unique_ptr<Solver> load_prec_;
  std::unique_ptr<CGSolver> load_cg_;
  std::unique_ptr<ParGridFunction> load_phi_ball_;
};

}  // namespace benchmark
