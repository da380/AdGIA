// ============================================================================
// aspherical_reference.cpp
//
// Leg C of the relabelling family (doc/benchmarks.tex, "Leg C: an
// aspherical reference body"; doc/mappings.md): the SAME spherical
// physical problem, described from an ASPHERICAL reference body whose
// mesh comes from its own generator (meshes/aspherical_body.py), not
// from the nodal image of the case's mesh — so the mapped machinery
// runs on aspherical mesh geometry, with a mapping that is NOT the
// identity on the surface, and the exact pyslfp reference still
// applies. The generator's mesh is itself the image under D (below) of
// a spherical gmsh mesh, so this is not a test on a mesh generated on
// the aspherical body directly.
//
// The generator builds the body by the radial stretch
//
//   D(x) = (r + h) x^,   h = eps r s(theta, phi)          r <= 1
//                        h = eps s q^2, q = (1+B-r)/B     1 <= r <= 1+B
//   s = P2(cos theta) + beta sin^2(theta) cos(2 phi)
//
// (identity on and outside the DtN sphere at 1+B). Here the mesh IS the
// reference body and phi_e = D^{-1} maps it onto the unit ball: the
// radial inversion is exact per point (linear inside, one stable
// quadratic root in the buffer), and F = [F_D(phi_e(y))]^{-1} with F_D
// the analytic radial-stretch gradient — exact-F mode throughout.
//
// The constitutive coefficients are the case's exact radial profiles
// composed with phi_e (radial_profiles.txt), the density carries the
// Jacobian, and the surface load is the physical harmonic pulled back
// with its Nanson area factor. The solution is compared as FIELDS
// against the pyslfp radial solutions (reference_fields.txt) composed
// through phi_e: relative L2 errors of u over the body and of the
// EULERIAN potential phi1 = zeta1 - b.u over the body, per
// degree. Degree one is skipped (frame bookkeeping adds nothing this
// driver tests). The model must be a single solid layer (homogeneous,
// linear_solid): the gauge-free regime in which the identity of
// relabelled_identity is strict.
//
// Sample run (the case is a SPHERICAL one of the campaign, made by
// make_case.py; it supplies the model data and the reference only):
//    mpiexec -np 8 ./aspherical_reference \
//        -m ../../data/aspherical_buffer_3d.mesh \
//        -c runs/homogeneous/h0.3/case.json -o 2
// ============================================================================

#include <fstream>
#include <iostream>
#include <memory>

#include "benchmark_case.hpp"
#include "reference_field.hpp"
#include "relabelling.hpp"

using namespace mfem;
using namespace AdGIA;
using namespace benchmark;

namespace {

// The inverse of the generator's radial stretch: phi_e = D^{-1}, exact.
class InverseShape : public Diffeomorphism {
 public:
  InverseShape(int dim, real_t eps, real_t beta, real_t buffer)
      : Diffeomorphism(dim), eps_(eps), beta_(beta), B_(buffer) {}

  void Eval(Vector& V, ElementTransformation& T,
            const IntegrationPoint& ip) override {
    Vector y(vdim);
    T.Transform(ip, y);
    Map(y, V);
  }

  void EvalGradient(DenseMatrix& F, ElementTransformation& T,
                    const IntegrationPoint& ip) override {
    Vector y(vdim), x(vdim);
    T.Transform(ip, y);
    Map(y, x);
    DenseMatrix FD(vdim);
    ForwardGradient(x, FD);
    F.SetSize(vdim);
    CalcInverse(FD, F);
  }

  // The reference radius of the physical radius rt in the direction of
  // the angular factor s: the inversion of r + h(r) = rt.
  real_t Radius(real_t rt, real_t s) const {
    const real_t inside = rt / (1.0 + eps_ * s);
    if (inside <= 1.0) {
      return inside;
    }
    if (rt >= 1.0 + B_) {
      return rt;
    }
    // r + eps s q^2 = rt, q = (1+B-r)/B: with u = 1+B-r and
    // a = eps s / B^2, a u^2 - u + c = 0, c = 1+B-rt; the root that
    // goes to u = c as a -> 0, in its cancellation-stable form.
    const real_t a = eps_ * s / (B_ * B_), c = 1.0 + B_ - rt;
    const real_t disc = std::max(0.0, 1.0 - 4.0 * a * c);
    const real_t u = 2.0 * c / (1.0 + std::sqrt(disc));
    return 1.0 + B_ - u;
  }

 private:
  void Angular(const Vector& y, real_t& s, real_t& s_th,
               real_t& s_ph_over_sin) const {
    const real_t r = y.Norml2();
    const real_t ct = vdim == 3 ? y[2] / std::max(r, real_t(1e-300))
                                     : 0.0;
    const real_t st = std::sqrt(std::max(0.0, 1.0 - ct * ct));
    real_t c2p = 1.0, s2p = 0.0;
    const real_t rho2 = vdim == 3 ? y[0] * y[0] + y[1] * y[1]
                                       : r * r;
    if (rho2 > 1e-300) {
      // cos 2phi, sin 2phi from the equatorial components.
      c2p = (y[0] * y[0] - y[1] * y[1]) / rho2;
      s2p = 2.0 * y[0] * y[1] / rho2;
    }
    const real_t p2 = 0.5 * (3.0 * ct * ct - 1.0);
    s = p2 + beta_ * st * st * c2p;
    s_th = -3.0 * ct * st + 2.0 * beta_ * st * ct * c2p;
    s_ph_over_sin = -2.0 * beta_ * st * s2p;
  }

  void Map(const Vector& y, Vector& x) const {
    const real_t rt = y.Norml2();
    x.SetSize(vdim);
    if (rt < 1e-300) {
      x = y;
      return;
    }
    real_t s, s_th, s_ph;
    Angular(y, s, s_th, s_ph);
    const real_t r = Radius(rt, s);
    x = y;
    x *= r / rt;
  }

  // F_D at the REFERENCE point x (the radial-stretch gradient, exact),
  // written pole-regular in Cartesian form:
  //   F_D = (1 + dh/dr) r^ r^T + (1 + h/r)(I - r^ r^T) + r^ (x) v_t,
  //   v_t = (c1/r) [ A (sin th th^) - 2 beta s2p (sin th ph^) ],
  // with h = c1(r) s(th, ph), A = ds/dth / sin th = ct (2 beta c2p - 3),
  // sin th th^ = (ct x0/r, ct x1/r, -sin^2 th) and
  // sin th ph^ = (-x1/r, x0/r, 0) — every factor smooth on the axis.
  void ForwardGradient(const Vector& x, DenseMatrix& FD) const {
    const int dim = vdim;
    const real_t r = x.Norml2();
    FD.SetSize(dim);
    FD = 0.0;
    for (int i = 0; i < dim; i++) {
      FD(i, i) = 1.0;
    }
    if (r < 1e-300 || r >= 1.0 + B_) {
      return;
    }
    real_t s, s_th, s_ph;
    Angular(x, s, s_th, s_ph);
    real_t h, hr, c1;  // h = c1 s, dh/dr = hr
    if (r <= 1.0) {
      h = eps_ * r * s;
      hr = eps_ * s;
      c1 = eps_ * r;
    } else {
      const real_t q = (1.0 + B_ - r) / B_;
      h = eps_ * s * q * q;
      hr = -2.0 * eps_ * s * q / B_;
      c1 = eps_ * q * q;
    }
    Vector er(x);
    er /= r;
    const real_t Frr = 1.0 + hr, Ftt = 1.0 + h / r;
    // The pole-regular pieces of the off-diagonal row.
    const real_t ct = dim == 3 ? er[2] : 0.0;
    const real_t st2 = std::max(0.0, 1.0 - ct * ct);
    real_t c2p = 1.0, s2p = 0.0;
    const real_t rho2 = dim == 3 ? x[0] * x[0] + x[1] * x[1] : r * r;
    if (rho2 > 1e-300) {
      c2p = (x[0] * x[0] - x[1] * x[1]) / rho2;
      s2p = 2.0 * x[0] * x[1] / rho2;
    }
    Vector vt(dim);
    vt = 0.0;
    if (dim == 3) {
      const real_t A = ct * (2.0 * beta_ * c2p - 3.0);
      // sin th th^ and sin th ph^, Cartesian.
      const real_t stth[3] = {ct * x[0] / r, ct * x[1] / r, -st2};
      const real_t stph[3] = {-x[1] / r, x[0] / r, 0.0};
      for (int i = 0; i < 3; i++) {
        vt[i] = (c1 / r) * (A * stth[i] - 2.0 * beta_ * s2p * stph[i]);
      }
    } else {
      // 2-D plane theta = pi/2: s = -1/2 + beta cos 2phi.
      const real_t stph[2] = {-x[1] / r, x[0] / r};
      for (int i = 0; i < 2; i++) {
        vt[i] = (c1 / r) * (-2.0 * beta_ * s2p * stph[i]);
      }
    }
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        FD(i, j) = Ftt * ((i == j) - er[i] * er[j]) +
                   Frr * er[i] * er[j] + er[i] * vt[j];
      }
    }
  }

  real_t eps_, beta_, B_;
};

// One harmonic as a coefficient, Y_i of the direction; the index is
// mutable so ONE registered load integrator serves every degree.
class HarmonicCoefficient : public Coefficient {
 public:
  explicit HarmonicCoefficient(const SurfaceHarmonics& basis)
      : basis_(basis) {}
  void SetIndex(int i) { i_ = i; }
  real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    T.Transform(ip, x_);
    basis_.Eval(x_, Y_);
    return Y_[i_];
  }

 private:
  const SurfaceHarmonics& basis_;
  int i_ = 0;
  Vector x_, Y_;
};

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();
  const bool root = Mpi::Root();

  const char* mesh_file = "../../data/aspherical_buffer_3d.mesh";
  const char* case_file = "case.json";
  const char* out_file = "aspherical_results.json";
  double eps = 0.05, beta = 0.5, buffer = 0.2;
  int order = 2, dtn_degree = 12, lmax = 4, lmin = 0;
  double rel_tol = 1e-10;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
                 "The aspherical reference mesh (aspherical_body.py, with "
                 "its buffer).");
  args.AddOption(&case_file, "-c", "--case",
                 "Manifest of a SPHERICAL case of the same model: supplies "
                 "radial_profiles.txt and reference_fields.txt.");
  args.AddOption(&out_file, "-out", "--output", "Results file (JSON).");
  args.AddOption(&eps, "-eps", "--epsilon", "Shape amplitude of the mesh.");
  args.AddOption(&beta, "-beta", "--beta", "Elliptical part of the shape.");
  args.AddOption(&buffer, "-buffer", "--buffer", "Buffer thickness.");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&dtn_degree, "-deg", "--dtn-degree", "DtN degree.");
  args.AddOption(&lmin, "-lmin", "--min-degree", "Lowest degree.");
  args.AddOption(&lmax, "-lmax", "--max-degree", "Highest degree.");
  args.AddOption(&rel_tol, "-rt", "--rel-tol", "Solver tolerance.");
  args.Parse();
  if (!args.Good()) {
    if (root) args.PrintUsage(std::cout);
    return 1;
  }
  if (root) args.PrintOptions(std::cout);

  // The shape parameters: aspherical_body.py records eps, beta and the
  // buffer in the manifest beside the mesh, which is then authoritative,
  // so that a run cannot pair a mesh with the wrong flags. The flags
  // serve meshes without that meta; a flag that contradicts the
  // manifest is refused unless it still holds its default.
  {
    std::string mmpath(mesh_file);
    const auto ext = mmpath.rfind(".mesh");
    if (ext != std::string::npos) {
      mmpath = mmpath.substr(0, ext) + ".json";
      if (std::ifstream(mmpath).good()) {
        const MeshManifest mesh_manifest(mmpath);
        auto adopt = [&](const char* key, double& value, double def) {
          if (!mesh_manifest.HasMeta(key)) {
            return;
          }
          const double recorded = mesh_manifest.MetaNumber(key);
          MFEM_VERIFY(value == def || std::abs(value - recorded) < 1e-12,
                      "-" << key << " " << value
                          << " contradicts the mesh manifest (" << recorded
                          << " in " << mmpath
                          << "); drop the flag, the manifest is "
                             "authoritative.");
          value = recorded;
        };
        adopt("eps", eps, 0.05);
        adopt("beta", beta, 0.5);
        adopt("buffer", buffer, 0.2);
        if (root && mesh_manifest.HasMeta("eps")) {
          std::cout << "shape from the mesh manifest: eps " << eps
                    << ", beta " << beta << ", buffer " << buffer << "\n";
        }
      }
    }
  }

  // The model data of the spherical case: profiles, reference, G.
  const MeshManifest manifest(case_file);
  const real_t G = manifest.G();
  const std::string& mpath = manifest.Path();
  const auto slash = mpath.find_last_of('/');
  const std::string dir =
      slash == std::string::npos ? std::string() : mpath.substr(0, slash + 1);
  RadialProfiles profiles(dir + "radial_profiles.txt");
  RadialReference reference(dir + "reference_fields.txt");
  MFEM_VERIFY(manifest.FluidAttributes().Size() == 0 &&
                  manifest.SolidAttributes().Size() == 1,
              "aspherical_reference: a single solid layer (the strict, "
              "gauge-free regime).");

  // The aspherical reference mesh: body 1, buffer 2 by construction.
  Mesh smesh(mesh_file, 1, 1);
  const int dim = smesh.Dimension();
  MFEM_VERIFY(dim == 3, "The benchmark is for balls.");
  ParMesh parent(MPI_COMM_WORLD, smesh);
  Array<int> body_attr({1}), buffer_attr({2});
  ParSubMesh body(ParSubMesh::CreateFromDomain(parent, body_attr));
  ParSubMesh shell(ParSubMesh::CreateFromDomain(parent, buffer_attr));
  H1_FECollection fec(order, dim);
  ParFiniteElementSpace fes_u(&body, &fec, dim), fes_b(&shell, &fec, dim);
  ParFiniteElementSpace fes_zeta(&parent, &fec);

  // The mapping and its seatbelts: identity on the DtN sphere, and the
  // body's surface must land on the unit sphere (its nodes were moved
  // there analytically; quadrature points carry the geometric error).
  InverseShape xi(dim, eps, beta, buffer);
  {
    auto outer_marker = ExternalBoundaryMarker(&parent);
    const real_t d_out = MaxIdentityDeviation(xi, parent, outer_marker);
    // The outer facets' quadrature points sit off the exact sphere by
    // the geometric error, where the taper is not exactly zero: the
    // convention holds to that floor squared.
    MFEM_VERIFY(d_out < 1e-8,
                "The mapping must be the identity at the DtN sphere "
                "(deviation " << d_out << ").");
    // The body's surface must land on the unit sphere: its nodes were
    // moved there analytically, quadrature points carry the geometric
    // interpolation error of the order-2 facets.
    real_t d_surf = 0.0;
    Vector x(dim);
    for (int be = 0; be < body.GetNBE(); be++) {
      if (body.GetBdrAttribute(be) != 1) {
        continue;  // attribute 1 is the surface by construction
      }
      auto* T = body.GetBdrElementTransformation(be);
      const auto& ir =
          IntRules.Get(body.GetBdrElementGeometry(be), 2 * order + 2);
      for (int q = 0; q < ir.GetNPoints(); q++) {
        const auto& ip = ir.IntPoint(q);
        T->SetIntPoint(&ip);
        xi.Eval(x, *T, ip);
        d_surf = std::max(d_surf, std::abs(x.Norml2() - 1.0));
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, &d_surf, 1, MPITypeMap<real_t>::mpi_type,
                  MPI_MAX, MPI_COMM_WORLD);
    if (root) {
      std::cout << "surface lands on the unit sphere to " << d_surf
                << " (the geometric floor of the facets)\n";
    }
    MFEM_VERIFY(d_surf < 1e-2, "The mesh does not match -eps/-beta.");
  }

  // The composed constitutive chain: exact profiles at |phi_e(y)|.
  auto profile = [&](const char* name) {
    return [&profiles, name](const Vector& y) {
      return profiles.Eval(name, y.Norml2());
    };
  };
  TransformedFunctionCoefficient rho_c(xi, profile("rho"));
  TransformedFunctionCoefficient kappa_c(xi, profile("kappa"));
  TransformedFunctionCoefficient mu_c(xi, profile("mu"));
  TransformedFunctionCoefficient p0_c(xi, profile("p0"));
  JacobianCoefficient jac(xi);
  ProductCoefficient rho_tilde(jac, rho_c);
  auto C_eff = IsotropicElasticTensorCoefficient::FromBulkModulus(
      dim, kappa_c, mu_c);
  BareElasticTensorCoefficient C_bare(dim, C_eff, p0_c);
  ConstantCoefficient minus_one(-1.0);
  ProductCoefficient neg_p0(minus_one, p0_c);
  IdentityMatrixCoefficient id_mat(dim);
  ScalarMatrixProductCoefficient S_e(neg_p0, id_mat);
  RelabelledElasticTensorCoefficient C_rel(dim, C_bare, xi);
  PullbackStressCoefficient S_rel(dim, S_e, xi);
  ReferentialElasticRheology rheology(dim, C_rel, S_rel, xi);

  LinearQuasiStaticReferentialSelfGravitatingProblem problem(&fes_u, &fes_zeta, rheology,
                                              rho_tilde, G, dtn_degree);
  Vector bb_min, bb_max;
  parent.GetBoundingBox(bb_min, bb_max);
  auto Evac = NewRadialVacuumExtension(fes_u, fes_b, 1.0,
                                       bb_max.Normlinf());
  problem.SetPrescribedVacuumExtension(fes_b, *Evac);
  problem.SetRelTol(rel_tol);

  // The load: the physical harmonic on the surface, pulled back with
  // its Nanson area factor (the direction is preserved by the radial
  // map, so Y at the referential point IS Y at the physical one).
  SurfaceHarmonics basis(dim, lmax);
  Array<int> surface_marker(body.bdr_attributes.Max());
  surface_marker = 0;
  surface_marker[0] = 1;  // the body's surface (generator convention)
  NansonAreaCoefficient area(xi);
  HarmonicCoefficient sigma(basis);
  ProductCoefficient sigma_area(sigma, area);
  ConstantCoefficient minus_one_load(-1.0);
  ProductCoefficient minus_sigma_area(minus_one_load, sigma_area);
  problem.ExternalPotentialLoad().AddBoundaryIntegrator(
      new BoundaryLFIntegrator(minus_sigma_area), surface_marker);

  std::ostringstream degrees;
  if (root) {
    std::cout << "\n  l     |u - u_ref| / |u_ref|   |phi - phi_ref| / "
                 "|phi_ref|   its    time\n";
  }
  bool first = true;
  for (int l = lmin; l <= lmax; l++) {
    if (l == 1) {
      continue;  // frame bookkeeping; nothing this driver tests
    }
    const int i = basis.Index(l, 0);
    sigma.SetIndex(i);
    const auto t0 = Clock::now();
    problem.ResetSolution();
    problem.AssembleForce(0.0);
    const bool ok = problem.Solve();
    const double seconds = Seconds(t0);

    // Relative L2 errors against the composed reference: u and the
    // Eulerian phi1 = zeta1 - b.u, both over the body.
    Vector cs(basis.Size());
    cs = 0.0;
    cs[i] = 1.0;
    const GridFunction& u = problem.Displacement();
    const GridFunction& zeta_body = problem.PotentialOnBody();
    auto& b_c = problem.BackgroundGravity();
    real_t nums[2] = {0.0, 0.0}, dens[2] = {0.0, 0.0};
    Vector y(dim), x(dim), uh(dim), ur(dim), Y, bvec(dim);
    DenseMatrix gradY, F(dim);
    // Body: u and phi.
    for (int e = 0; e < body.GetNE(); e++) {
      auto* T = body.GetElementTransformation(e);
      const auto& ir = IntRules.Get(body.GetElementGeometry(e),
                                    2 * order + 3);
      for (int q = 0; q < ir.GetNPoints(); q++) {
        const auto& ip = ir.IntPoint(q);
        T->SetIntPoint(&ip);
        T->Transform(ip, y);
        xi.Eval(x, *T, ip);  // the physical point phi_e(y)
        const real_t w = ip.weight * T->Weight() * jac.Eval(*T, ip);
        const real_t r_phys = x.Norml2();
        basis.EvalWithGradient(x, Y, gradY);
        // reference u at the physical point (layer 1).
        const real_t U =
            reference.Eval(1, RadialReference::U, l, r_phys);
        const real_t V =
            reference.Eval(1, RadialReference::V, l, r_phys);
        for (int d = 0; d < dim; d++) {
          ur[d] = V * gradY(d, i) +
                  (r_phys > 0 ? U * Y[i] * x[d] / r_phys : 0.0);
        }
        u.GetVectorValue(*T, ip, uh);
        real_t du2 = 0.0, un2 = 0.0;
        for (int d = 0; d < dim; d++) {
          du2 += (uh[d] - ur[d]) * (uh[d] - ur[d]);
          un2 += ur[d] * ur[d];
        }
        nums[0] += w * du2;
        dens[0] += w * un2;
        // phi1 = zeta1 - b.u, b = F^{-T} grad zeta0 (the referential
        // gradient the class holds), vs the reference phi at x.
        const real_t zeta = zeta_body.GetValue(*T, ip);
        Vector gz(dim);
        b_c.Eval(gz, *T, ip);
        xi.EvalGradient(F, *T, ip);
        DenseMatrix Fi(dim);
        CalcInverse(F, Fi);
        Fi.MultTranspose(gz, bvec);
        real_t phi_h = zeta;
        for (int d = 0; d < dim; d++) {
          phi_h -= bvec[d] * uh[d];
        }
        const real_t phi_r =
            reference.Eval(1, RadialReference::Phi, l, r_phys);
        const real_t pr = phi_r * Y[i];
        nums[1] += w * (phi_h - pr) * (phi_h - pr);
        dens[1] += w * pr * pr;
      }
    }
    // The potential is compared on the BODY alone: off it zeta carries
    // the extension field (b . vtilde), which is gauge.
    MPI_Allreduce(MPI_IN_PLACE, nums, 2, MPITypeMap<real_t>::mpi_type,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, dens, 2, MPITypeMap<real_t>::mpi_type,
                  MPI_SUM, MPI_COMM_WORLD);
    const real_t eu = std::sqrt(nums[0] / std::max(dens[0], real_t(1e-300)));
    const real_t ep = std::sqrt(nums[1] / std::max(dens[1], real_t(1e-300)));
    if (root) {
      std::cout << std::setw(3) << l << std::setw(22) << eu << std::setw(28)
                << ep << std::setw(6) << problem.LastOuterIterations()
                << std::setw(8) << std::setprecision(3) << seconds
                << std::setprecision(6) << (ok ? "" : "  NOT CONVERGED")
                << "\n";
      degrees << (first ? "" : ",\n")
              << "    {\"degree\": " << l << ", \"u_error\": " << Num(eu)
              << ", \"phi_error\": " << Num(ep)
              << ", \"iterations\": " << problem.LastOuterIterations()
              << ", \"seconds\": " << Num(seconds)
              << ", \"converged\": " << (ok ? "true" : "false") << "}";
      first = false;
    }
  }

  if (root) {
    std::ofstream os(out_file);
    os << "{\n  \"mesh\": \"" << mesh_file << "\",\n  \"case\": \""
       << case_file << "\",\n  \"order\": " << order
       << ",\n  \"eps\": " << Num(eps) << ",\n  \"beta\": " << Num(beta)
       << ",\n  \"ranks\": " << Mpi::WorldSize()
       << ",\n  \"degrees\": [\n" << degrees.str() << "]\n}\n";
    std::cout << "\nWrote " << out_file << "\n";
  }
  return 0;
}
