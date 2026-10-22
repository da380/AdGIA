// ============================================================================
// traction_benchmark.cpp
//
// The E1 (non-gravitating elastic) leg of the cross-code hierarchy: the
// traction numbers of a solid, spherically layered case. The body is a case
// written by make_case.py (its mesh, kappa and mu; nothing about the model
// is set here); the problem is LinearQuasiStaticTractionProblem on the body
// alone -- no gravity, no potential -- with the rigid modes projected in
// the solver.
//
// For each degree l (order 0) the surface traction is set to -Y_l0 r^ (the
// normal channel) and, for l >= 2, to -grad_1 Y_l0 (the tangential
// channel); the displacement u = U Y r^ + V grad_1 Y is analysed into
// harmonic coefficients on the surface, and the (l, 0) coefficients per
// unit traction are the traction numbers h_u, l_u (normal) and h_v, l_v
// (tangential), written in SI (m per Pa) through the manifest's scales as
// one cross-code exchange file (common/exchange.py states the format;
// values are shortest-round-trip). Degree 1 is excluded -- a surface
// traction there has a net force against the rigid translations -- and
// written NaN; the tangential channel of degree 0 is zero.
//
// The exact reference is common/analytic.py (reference.py writes it);
// compare.py sets this driver's file against it. Self-adjointness gives
// h_v = l(l+1) l_u, reported as a residual, and the harmonic leakage into
// unforced coefficients is reported as `spurious`.
//
// Sample run:
//    mpiexec -np 8 ./traction_benchmark -c cases/two_solid/case.json \
//        -lmax 8 -out e1_two_solid_adgia.txt
// ============================================================================

#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <limits>

#include "AdGIA/manifest.hpp"
#include "AdGIA/quasi_static_problem.hpp"
#include "AdGIA/rheology.hpp"
#include "AdGIA/spherical_harmonics.hpp"
#include "mfem.hpp"
#include "planetmodel/manifest.hpp"
#include "planetmodel/units.hpp"

#ifndef ADGIA_GIT_COMMIT
#define ADGIA_GIT_COMMIT "unknown"
#endif

using namespace mfem;
using namespace AdGIA;

namespace {

// The surface traction of one channel: t = -sum_i c_i Y_i x^ (normal,
// pressing) or t = -sum_i c_i grad_1 Y_i (tangential), evaluated at the
// direction of the integration point.
class SurfaceTractionCoefficient : public VectorCoefficient {
 public:
  SurfaceTractionCoefficient(const SurfaceHarmonics& basis, int dim)
      : VectorCoefficient(dim), basis_(&basis), c_(basis.Size()) {
    c_ = 0.0;
  }

  void Set(const Vector& c, bool tangential) {
    c_ = c;
    tangential_ = tangential;
  }

  void Eval(Vector& V, ElementTransformation& T,
            const IntegrationPoint& ip) override {
    T.Transform(ip, x_);
    basis_->EvalWithGradient(x_, Y_, gradY_);
    V.SetSize(vdim);
    if (tangential_) {
      gradY_.Mult(c_, V);
      V.Neg();
    } else {
      const real_t r = x_.Norml2();
      const real_t f = -(Y_ * c_) / (r > 0.0 ? r : 1.0);
      V = x_;
      V *= f;
    }
  }

 private:
  const SurfaceHarmonics* basis_;
  Vector c_;
  bool tangential_ = false;
  Vector x_, Y_;
  DenseMatrix gradY_;
};

// The largest unforced coefficient of either analysis, relative to the
// degree's own scale (the larger forced coefficient).
real_t Spurious(const Vector& cu, const Vector& cv, int main) {
  real_t off = 0.0;
  for (int j = 0; j < cu.Size(); j++) {
    if (j != main) {
      off = std::max({off, std::abs(cu[j]), std::abs(cv[j])});
    }
  }
  return off / (std::max(std::abs(cu[main]), std::abs(cv[main])) + 1e-300);
}

}  // namespace

int main(int argc, char* argv[]) try {
  Mpi::Init(argc, argv);
  Hypre::Init();
  const bool root = Mpi::Root();
  using Clock = std::chrono::steady_clock;

  const char* case_file = "case.json";
  const char* out_file = "e1_adgia.txt";
  int lmax = 8;
  int order = 2;
  real_t rel_tol = 1e-12;

  OptionsParser args(argc, argv);
  args.AddOption(&case_file, "-c", "--case", "Case manifest (case.json).");
  args.AddOption(&out_file, "-out", "--output", "Exchange file to write.");
  args.AddOption(&lmax, "-lmax", "--max-degree", "Highest harmonic degree.");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&rel_tol, "-rtol", "--relative-tolerance",
                 "Solver relative tolerance.");
  args.Parse();
  if (!args.Good()) {
    if (root) args.PrintUsage(std::cout);
    return 1;
  }
  if (root) args.PrintOptions(std::cout);

  const planetmodel::MeshManifest manifest(case_file);
  if (root) Print(manifest, std::cout);
  MFEM_VERIFY(manifest.FluidAttributes().empty(),
              "The non-gravitating problem takes a solid model: a fluid "
              "region has no restoring force without gravity.");

  // The body alone, with its moduli; the buffer shell (there for the
  // gravitating problems' potential) is not meshed into the problem.
  Mesh serial = LoadMesh(manifest);
  MFEM_VERIFY(serial.Dimension() == 3, "The benchmark is for balls.");
  auto kappa_s = LoadField(manifest, serial, "kappa");
  auto mu_s = LoadField(manifest, serial, "mu");
  std::unique_ptr<int[]> partitioning(
      serial.GeneratePartitioning(Mpi::WorldSize()));
  ParMesh parent(MPI_COMM_WORLD, serial, partitioning.get());
  ParGridFunction kappa_p(&parent, kappa_s.get(), partitioning.get());
  ParGridFunction mu_p(&parent, mu_s.get(), partitioning.get());

  ParSubMesh body = ParSubMesh::CreateFromDomain(
      parent, ToArray(manifest.BodyAttributes()));
  ParFiniteElementSpace kappa_space(&body, kappa_p.ParFESpace()->FEColl());
  ParFiniteElementSpace mu_space(&body, mu_p.ParFESpace()->FEColl());
  ParGridFunction kappa(&kappa_space), mu(&mu_space);
  ParSubMesh::Transfer(kappa_p, kappa);
  ParSubMesh::Transfer(mu_p, mu);
  GridFunctionCoefficient kappa_c(&kappa), mu_c(&mu);

  const int dim = body.Dimension();
  H1_FECollection fec(order, dim);
  ParFiniteElementSpace fes(&body, &fec, dim);
  IsotropicElasticRheology rheology(dim, kappa_c, mu_c);

  const Array<int> surface = Marker(Array<int>({manifest.SurfaceAttribute()}),
                                    body.bdr_attributes.Max());
  using BHC = BoundaryHarmonicCoefficients;
  BHC radial(fes, surface, lmax, BHC::Component::Radial);
  BHC tangential(fes, surface, lmax, BHC::Component::Tangential);
  const SurfaceHarmonics& basis = radial.Basis();

  SurfaceTractionCoefficient traction(basis, dim);
  LinearQuasiStaticTractionProblem problem(&fes, rheology, traction, surface);
  problem.SetRelTol(rel_tol);

  const auto elements = parent.GetGlobalNE();
  const auto unknowns = fes.GlobalTrueVSize();
  if (root) {
    std::cout << "Ranks " << Mpi::WorldSize() << ", elements " << elements
              << ", displacement unknowns " << unknowns
              << "\n\n  l           h_u"
              << "           l_u           h_v           l_v    spurious"
              << "   time\n";
  }

  // The scales of the case's units: the numbers are converted to SI.
  const auto& scales = manifest.GetScales();
  const double h_si =
      scales.Factor(planetmodel::Dimensions{.mass = -1, .length = 2,
                                            .time = 2});
  const double radius_si =
      manifest.SurfaceRadius() * scales.Factor(planetmodel::Dimensions{
                                     .length = 1});

  const real_t nan = std::numeric_limits<real_t>::quiet_NaN();
  std::vector<real_t> h_u(lmax + 1, nan), l_u(lmax + 1, nan),
      h_v(lmax + 1, nan), l_v(lmax + 1, nan);
  bool all_converged = true;

  for (int l = 0; l <= lmax; l++) {
    if (l == 1) continue;
    const int i = basis.Index(l, 0);
    Vector c(basis.Size());
    real_t spurious = 0.0;
    double seconds = 0.0;
    for (const bool tan : {false, true}) {
      if (tan && l == 0) {
        h_v[0] = 0.0;
        l_v[0] = 0.0;
        continue;
      }
      c = 0.0;
      c[i] = 1.0;
      traction.Set(c, tan);
      const auto t0 = Clock::now();
      problem.AssembleForce(0.0);
      const bool converged = problem.Solve();
      seconds += std::chrono::duration<double>(Clock::now() - t0).count();
      all_converged = all_converged && converged;
      Vector cu, cv;
      radial.Coefficients(problem.Displacement(), cu);
      tangential.Coefficients(problem.Displacement(), cv);
      (tan ? h_v : h_u)[l] = cu[i] * h_si;
      (tan ? l_v : l_u)[l] = cv[i] * h_si;
      spurious = std::max(spurious, Spurious(cu, cv, i));
    }
    if (root) {
      std::cout << std::setw(3) << l << std::scientific
                << std::setprecision(6) << std::setw(14) << h_u[l]
                << std::setw(14) << l_u[l] << std::setw(14) << h_v[l]
                << std::setw(14) << l_v[l] << std::setprecision(1)
                << std::setw(12) << spurious << std::defaultfloat
                << std::setprecision(3) << std::setw(7) << seconds
                << (all_converged ? "" : "   NOT CONVERGED") << "\n";
    }
  }

  real_t reciprocity = 0.0;
  for (int l = 2; l <= lmax; l++) {
    const real_t k2 = l * (l + 1.0);
    const real_t scale = std::max(std::abs(h_v[l]), std::abs(k2 * l_u[l]));
    if (scale > 0.0) {
      reciprocity =
          std::max(reciprocity, std::abs(h_v[l] - k2 * l_u[l]) / scale);
    }
  }

  if (root) {
    const auto today = std::chrono::floor<std::chrono::days>(
        std::chrono::system_clock::now());
    std::ofstream os(out_file);
    MFEM_VERIFY(os.good(), "Cannot write " << out_file << ".");
    os << "# E1 traction numbers from AdGIA (non-gravitating traction "
          "problem)\n";
    os << std::format("# code adgia commit {} date {:%F}\n", ADGIA_GIT_COMMIT,
                      today);
    os << std::format(
        "# case {} order {} lmax {} rtol {}; SI (m per Pa)\n", case_file,
        order, lmax, rel_tol);
    os << std::format("radius {}\n", radius_si);
    os << "gravitating 0\n";
    os << "columns l h_u l_u h_v l_v\n";
    for (int l = 0; l <= lmax; l++) {
      os << std::format("{} {} {} {} {}\n", l, h_u[l], l_u[l], h_v[l],
                        l_v[l]);
    }
    std::cout << "\nreciprocity " << std::scientific << std::setprecision(3)
              << reciprocity << (all_converged ? "" : ", NOT CONVERGED")
              << "\nWrote " << out_file << "\n";
  }
  return all_converged ? 0 : 2;
} catch (const planetmodel::Error& e) {
  std::cerr << e.what() << "\n";
  return 1;
}
