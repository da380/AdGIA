// ============================================================================
// prestress_loading.cpp
//
// Does the deviatoric part of the equilibrium stress ever matter for
// loading? A self-contained benchmark on a body that CANNOT be
// hydrostatic: the homogeneous ellipse (semi-axes a = 1 + e, b = 1/a).
//
// The pipeline is the general referential machinery end to end, on the
// FIXED reference disc:
//
//   1. The equilibrium mapping phi_e carries the shape: the exact linear
//      ellipse map on the body, blended to the identity at the DtN
//      sphere by tapering the ellipse parameter across the buffer
//      (ParameterTaperedEllipseMap below), or, where that blend is not
//      diffeomorphic, by the harmonic extension of the body's map
//      (NewHarmonicExtensionMapping). The referential density is rho (the
//      map is area-preserving on the body).
//   2. A realistic equilibrium stress from the relabelled (mapped)
//      minimum-deviatoric generator (AW10 Stokes problem pulled back to
//      the reference body): S_e = J F^{-1}(-p 1 + 2 mu grad_s u) F^{-T},
//      with the body force the EXACT self-gravity of the homogeneous
//      elliptical cylinder, rho grad Phi = 4 pi G rho^2/(a+b) (b x, a y).
//   3. Loading responses from LinearQuasiStaticReferentialSelfGravitatingProblem
//      (mapped Poisson background, referential gravity, prescribed
//      vacuum extension, surface load mapped with the Nanson area
//      factor) computed TWICE, differing ONLY in the equilibrium
//      stress:
//        full:        S_e as generated (pressure + deviatoric part);
//        hydrostatic: S_e = -p J F^{-1} F^{-T}, the same field with its
//                     deviatoric part dropped in the physical frame --
//                     the quasi-hydrostatic approximation of standard
//                     practice (which is NOT an equilibrium stress here:
//                     no hydrostatic equilibrium exists off e = 0).
//
// The BARE elastic tensor is held fixed between the two runs: the
// comparison isolates the geometric-stiffness term S_e : (Du^T Dv).
// (If instead the SEISMOLOGICAL moduli were held fixed, the bare tensor
// would also change with the pre-stress -- the Maitra & Al-Attar 2021
// subtlety; that is a separate, material question.)
//
// The table prints, per ellipticity: the deviatoric fraction of S_e,
// the relative differences in displacement and potential between the
// two runs, the solver iterations and the taper rule used. Earth's non-hydrostatic geoid
// argues e_effective is tiny; planetary applications (fossil figures,
// fast rotators) reach e ~ 0.1 and beyond.
//
// One source serves the serial and the parallel build.
//
// Outputs: the table on the screen; prestress_loading.csv, the three
// relative measures against e (python3 plot_csv.py prestress_loading.csv);
// with -vis (the default), for the largest ellipticity that ran, GLVis
// windows of |dev S_e| (the deviatoric part the approximation drops) and of
// u_full - u_hydro (what dropping it costs), drawn on the PHYSICAL
// ellipse: a copy of the reference body moved by the ellipse map.
//
// Options (defaults in brackets):
//   -m    mesh file [../data/elastogravity_2d_wide.msh]: 2-D, body
//         (attribute 1) plus buffer (attribute 2) out to the DtN
//         sphere.
//   -o    finite element order [2]; the stress generator uses
//         Taylor-Hood spaces of orders o+1 / o.
//   -G    gravitational constant [0.1]: sets the pre-stress level
//         p/mu ~ pi G rho^2 / mu, with which the difference rows
//         scale.
//   -s    surface mass load amplitude [0.02].
//   -e    run a single ellipticity instead of the sweep [-1, meaning
//         the sweep 0.005 ... 0.3].
//   -vis / -no-vis   GLVis windows of |dev S_e| and u_full - u_hydro
//         on the physical ellipse [on].
//   -csv  CSV file of the sweep for plot_csv.py
//         [prestress_loading.csv]; "" writes none.
//
// Sample runs (with mpiexec -np N in front in a parallel build):
//    ./prestress_loading
//    ./prestress_loading -G 0.2 -o 3
//    ./prestress_loading -e 0.3     (single ellipticity)
// ============================================================================

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
using FieldType = ParGridFunction;
bool Root() { return Mpi::Root(); }
double GlobalSum(double v) {
  double g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return g;
}
double GlobalMin(double v) {
  double g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  return g;
}
#else
using MeshType = Mesh;
using SubMeshType = SubMesh;
using SpaceType = FiniteElementSpace;
using FieldType = GridFunction;
bool Root() { return true; }
double GlobalSum(double v) { return v; }
double GlobalMin(double v) { return v; }
#endif

constexpr double kRho = 1.0;
constexpr double kKappaBare = 2.0;
constexpr double kMuBare = 1.0;
constexpr int kDtNDegree = 12;
constexpr double kPi = std::numbers::pi;

double G = 0.1;       // gravitational constant (p/mu ~ pi G rho^2 / mu)
double Sigma0 = 0.02; // surface mass load amplitude

// Physical surface mass load: degree-2 zonal pattern about the y axis.
double SurfaceLoad(const Vector& y) {
  const double r = y.Norml2();
  const double c = y[1] / r;
  return Sigma0 * (1.0 + (2.0 * c * c - 1.0));
}

// The exact linear ellipse map (ball-wide formula; used pure on the
// body for the stress generator).
CallableDiffeomorphism EllipseMap(int dim, double a) {
  const double b = 1.0 / a;
  return CallableDiffeomorphism(
      dim,
      [a, b](const Vector& x, Vector& y) {
        y.SetSize(2);
        y(0) = a * x(0);
        y(1) = b * x(1);
      },
      [a, b](const Vector&, DenseMatrix& F) {
        F = 0.0;
        F(0, 0) = a;
        F(1, 1) = b;
      });
}

// The buffer taper for the loading problem: blend the ellipse PARAMETER,
// phi = (a(r) x, y / a(r)) with a(r) = 1 + e t(r) and t the cubic
// smoothstep (1 on the body, 0 at the DtN sphere). Its exact Jacobian is
//   J = 1 + (a'/(a r)) (x^2 - y^2),
// far better conditioned than blending the displacement (the
// TaperedDiffeomorphism rule), so larger ellipticities survive a thin
// buffer. Exact ellipse on the body, identity with F = I at the sphere.
CallableDiffeomorphism ParameterTaperedEllipseMap(int dim, double e,
                                                  double r0, double r1) {
  auto af = [e, r0, r1](double r, double& a, double& da) {
    if (r <= r0) {
      a = 1.0 + e;
      da = 0.0;
      return;
    }
    if (r >= r1) {
      a = 1.0;
      da = 0.0;
      return;
    }
    const double w = r1 - r0;
    const double s = (r - r0) / w;
    a = 1.0 + e * (1.0 - s * s * (3.0 - 2.0 * s));
    da = -e * 6.0 * s * (1.0 - s) / w;
  };
  return CallableDiffeomorphism(
      dim,
      [af](const Vector& x, Vector& y) {
        double a, da;
        af(x.Norml2(), a, da);
        y.SetSize(2);
        y(0) = a * x(0);
        y(1) = x(1) / a;
      },
      [af](const Vector& x, DenseMatrix& F) {
        const double r = x.Norml2();
        double a, da;
        af(r, a, da);
        F.SetSize(2);
        if (r > 0.0 && da != 0.0) {
          F(0, 0) = a + da * x(0) * x(0) / r;
          F(0, 1) = da * x(0) * x(1) / r;
          F(1, 0) = -da * x(0) * x(1) / (a * a * r);
          F(1, 1) = 1.0 / a - da * x(1) * x(1) / (a * a * r);
        } else {
          F = 0.0;
          F(0, 0) = a;
          F(1, 1) = 1.0 / a;
        }
      });
}

// L2 norm over the mesh of (f - its mean), by quadrature: the 2-D
// potential comparison must ignore the constant gauge.
double MeanFreeNorm(const GridFunction& f, Mesh& mesh, int order) {
  double s = 0.0, s2 = 0.0, vol = 0.0;
  for (int e = 0; e < mesh.GetNE(); e++) {
    auto* T = mesh.GetElementTransformation(e);
    const auto& ir = IntRules.Get(mesh.GetElementGeometry(e), 2 * order);
    for (int q = 0; q < ir.GetNPoints(); q++) {
      const auto& ip = ir.IntPoint(q);
      T->SetIntPoint(&ip);
      const double w = ip.weight * T->Weight();
      const double v = f.GetValue(*T, ip);
      s += w * v;
      s2 += w * v * v;
      vol += w;
    }
  }
  s = GlobalSum(s);
  s2 = GlobalSum(s2);
  vol = GlobalSum(vol);
  const double mean = s / vol;
  return std::sqrt(std::max(0.0, s2 - vol * mean * mean));
}

double L2Norm(const GridFunction& u) {
  const int vdim = u.FESpace()->GetVDim();
  Vector zero(vdim);
  zero = 0.0;
  VectorConstantCoefficient z(zero);
  return const_cast<GridFunction&>(u).ComputeL2Error(z);
}

// |dev S| pointwise (Frobenius), for plotting.
class DeviatoricNormCoefficient : public Coefficient {
 public:
  explicit DeviatoricNormCoefficient(MatrixCoefficient& S) : S_(S) {}
  real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    S_.Eval(M_, T, ip);
    const int d = M_.Height();
    real_t tr = 0.0;
    for (int i = 0; i < d; i++) {
      tr += M_(i, i);
    }
    real_t n2 = 0.0;
    for (int i = 0; i < d; i++) {
      for (int j = 0; j < d; j++) {
        const real_t v = M_(i, j) - (i == j ? tr / d : 0.0);
        n2 += v * v;
      }
    }
    return std::sqrt(n2);
  }

 private:
  MatrixCoefficient& S_;
  DenseMatrix M_;
};

void StressNorms(MatrixCoefficient& S, Mesh& mesh, int order, double& full,
                 double& dev) {
  const int dim = mesh.Dimension();
  DenseMatrix T;
  double full2 = 0.0, dev2 = 0.0;
  for (int e = 0; e < mesh.GetNE(); e++) {
    auto* Tr = mesh.GetElementTransformation(e);
    const auto& ir = IntRules.Get(mesh.GetElementGeometry(e), 2 * order);
    for (int q = 0; q < ir.GetNPoints(); q++) {
      const auto& ip = ir.IntPoint(q);
      Tr->SetIntPoint(&ip);
      const double w = ip.weight * Tr->Weight();
      S.Eval(T, *Tr, ip);
      double tr = 0.0;
      for (int i = 0; i < dim; i++) {
        tr += T(i, i);
      }
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
          const double d = T(i, j) - (i == j ? tr / dim : 0.0);
          full2 += w * T(i, j) * T(i, j);
          dev2 += w * d * d;
        }
      }
    }
  }
  full = std::sqrt(GlobalSum(full2));
  dev = std::sqrt(GlobalSum(dev2));
}

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif

  const char* mesh_file = "../data/elastogravity_2d_wide.msh";
  int order = 2;
  double single_e = -1.0;
  bool visualization = true;
  const char* csv_file = "prestress_loading.csv";

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use (2-D).");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&G, "-G", "--gravitational-constant",
                 "Gravitational constant (sets p/mu).");
  args.AddOption(&Sigma0, "-s", "--sigma", "Surface load amplitude.");
  args.AddOption(&single_e, "-e", "--ellipticity",
                 "Run a single ellipticity instead of the sweep.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Show |dev S_e| and the response difference in GLVis.");
  args.AddOption(&csv_file, "-csv", "--csv",
                 "Table of the sweep for plot_csv.py (\"\": none).");
  args.Parse();
  if (!args.Good()) {
    if (Root()) {
      args.PrintUsage(std::cout);
    }
    return 1;
  }

  Mesh smesh(mesh_file, 1, 1);
  MFEM_VERIFY(smesh.Dimension() == 2, "a 2-D benchmark");
  const int dim = 2;
#ifdef MFEM_USE_MPI
  MeshType mesh(MPI_COMM_WORLD, smesh);
  smesh.Clear();
#else
  MeshType& mesh = smesh;
#endif
  Vector bb_min, bb_max;
  mesh.GetBoundingBox(bb_min, bb_max);
  const double r_out = bb_max.Normlinf();

  Array<int> body_attr({1}), buffer_attr({2});
  auto body = SubMeshType::CreateFromDomain(mesh, body_attr);
  auto buffer = SubMeshType::CreateFromDomain(mesh, buffer_attr);
  H1_FECollection fec(order, dim);
  SpaceType fes_u(&body, &fec, dim), fes_zeta(&mesh, &fec);
  SpaceType fes_buffer(&buffer, &fec, dim);
  // Taylor-Hood spaces for the stress generator.
  H1_FECollection fec_gu(order + 1, dim), fec_gp(order, dim);
  SpaceType fes_gu(&body, &fec_gu, dim), fes_gp(&body, &fec_gp);

  Array<int> surface(body.bdr_attributes.Max());
  surface = 0;
  surface[body.bdr_attributes.Max() - 1] = 1;

  ConstantCoefficient kappa(kKappaBare), mu(kMuBare), rho(kRho);
  auto C =
      IsotropicElasticTensorCoefficient::FromBulkModulus(dim, kappa, mu);

  std::vector<double> sweep;
  if (single_e >= 0.0) {
    sweep = {single_e};
  } else {
    sweep = {0.005, 0.02, 0.05, 0.1, 0.2, 0.3};
  }

  examples::CsvTable table(csv_file, {"e", "dev_ratio", "du_rel", "dzeta_rel"});
  table.Meta("title", "Pre-stress in loading: full vs quasi-hydrostatic")
      .Meta("note", "homogeneous ellipse, G = " + std::to_string(G).substr(0, 5))
      .Meta("xlabel", "ellipticity e")
      .Meta("ylabel", "relative size")
      .Meta("logx", "true")
      .Meta("logy", "true");
  // The last (largest) ellipticity's fields, for GLVis.
  L2_FECollection fec_l2(order, dim);
  SpaceType fes_dev(&body, &fec_l2);
  FieldType dev_field(&fes_dev), du_field(&fes_u);
  double shown_e = -1.0;

  if (Root()) {
    std::cout << "pre-stress in loading on the homogeneous ellipse "
                 "(G = "
              << G << ", p(0)/mu = " << kPi * G * kRho * kRho / kMuBare
              << ", order " << order << ")\n"
              << "full = generated S_e; hydro = its pressure part only "
                 "(quasi-hydrostatic approximation)\n\n"
              << "    e    dev(S_e)/|S_e|   |du|/|u|    |dzeta|/|zeta|  "
                 "its(full/hydro)\n";
  }

  for (double e : sweep) {
    const double a = 1.0 + e, b = 1.0 / a;

    // The shape: exact on the body, blended to the identity at the DtN
    // sphere -- the analytic smoothstep when it stays diffeomorphic, the
    // harmonic-extension rule otherwise (its buffer gradients are
    // gentler; on the body the interpolant of the linear map is exact
    // either way).
    auto lin = EllipseMap(dim, a);
    auto min_jacobian = [&](Diffeomorphism& map) {
      double mj = std::numeric_limits<double>::infinity();
      DenseMatrix F;
      for (int el = 0; el < mesh.GetNE(); el++) {
        auto* T = mesh.GetElementTransformation(el);
        const auto& ip = Geometries.GetCenter(mesh.GetElementGeometry(el));
        T->SetIntPoint(&ip);
        map.EvalGradient(F, *T, ip);
        mj = std::min(mj, F.Det());
      }
      return GlobalMin(mj);
    };
    auto tapered = ParameterTaperedEllipseMap(dim, e, 1.0, r_out);
    std::unique_ptr<GridFunctionDiffeomorphism> harmonic;
    Diffeomorphism* phi_ptr = &tapered;
    const char* rule = "parameter";
    double min_J = min_jacobian(tapered);
    if (min_J <= 0.05) {
      harmonic = std::make_unique<GridFunctionDiffeomorphism>(
          NewHarmonicExtensionMapping(mesh, order, lin, body_attr,
                                      buffer_attr));
      phi_ptr = harmonic.get();
      rule = "harmonic";
      min_J = min_jacobian(*phi_ptr);
    }
    if (min_J <= 0.05) {
      if (Root()) {
        std::cout << std::setw(7) << e
                  << "   (skipped: no diffeomorphic taper, min J = "
                  << min_J << " -- widen the buffer or reduce e)\n";
      }
      continue;
    }
    Diffeomorphism& phi_e = *phi_ptr;

    // The realistic equilibrium stress: relabelled minimum-deviatoric
    // generator, exact elliptical-cylinder self-gravity.
    auto force = [a, b](const Vector& y, Vector& v) {
      const double c = 4.0 * kPi * G * kRho * kRho / (a + b);
      v.SetSize(2);
      v[0] = c * b * y[0];
      v[1] = c * a * y[1];
    };
    TransformedVectorFunctionCoefficient f_comp(lin, force);
    MinimumDeviatoricEquilibriumStress S_full(fes_gu, fes_gp, f_comp,
                                              nullptr, &lin);

    // The quasi-hydrostatic approximation: the same field with its
    // deviatoric part dropped (physically), S = -p J F^{-1} F^{-T}.
    GridFunctionCoefficient p_c(&S_full.Pressure());
    ProductCoefficient minus_p(-1.0, p_c);
    PullbackDiffusionCoefficient a_e(lin);
    ScalarMatrixProductCoefficient S_hydro(minus_p, a_e);

    double full_n, dev_n;
    StressNorms(S_full, body, order + 1, full_n, dev_n);

    // The referential surface load: physical sigma per physical area,
    // composed and weighted with the Nanson area factor.
    TransformedFunctionCoefficient sigma_comp(lin, SurfaceLoad);
    NansonAreaCoefficient nu(lin);
    ProductCoefficient sigma_ref(sigma_comp, nu);

    // The two runs differ ONLY in S_e.
    auto run = [&](MatrixCoefficient& S, GridFunction& u_out,
                   GridFunction& z_out, int& its) -> bool {
      ReferentialElasticRheology rheology(dim, C, S, phi_e);
      LinearQuasiStaticReferentialSelfGravitatingProblem problem(
          &fes_u, &fes_zeta, rheology, rho, G, kDtNDegree);
      auto E = NewRadialVacuumExtension(fes_u, fes_buffer, 1.0, r_out);
      problem.SetPrescribedVacuumExtension(fes_buffer, *E);
      problem.SetSurfaceLoad(sigma_ref, surface);
      problem.SetRelTol(1e-10);
      problem.AssembleForce(0.0);
      if (!problem.Solve()) {
        return false;
      }
      u_out = problem.Displacement();
      z_out = problem.Potential();
      its = problem.LastOuterIterations();
      return true;
    };

    FieldType uA(&fes_u), uB(&fes_u), zA(&fes_zeta), zB(&fes_zeta);
    int itsA = 0, itsB = 0;
    if (!run(S_full, uA, zA, itsA) || !run(S_hydro, uB, zB, itsB)) {
      if (Root()) {
        std::cout << std::setw(7) << e << "   (solver failure)\n";
      }
      continue;
    }

    FieldType du(uA);
    du -= uB;
    FieldType dz(zA);
    dz -= zB;
    const double du_rel = L2Norm(du) / L2Norm(uA);
    const double dz_rel =
        MeanFreeNorm(dz, mesh, order) / MeanFreeNorm(zA, mesh, order);

    table.Row({e, dev_n / full_n, du_rel, dz_rel});
    if (visualization) {
      DeviatoricNormCoefficient dev_c(S_full);
      dev_field.ProjectCoefficient(dev_c);
      du_field = du;
      shown_e = e;
    }
    if (Root()) {
      std::cout << std::setw(7) << std::setprecision(3) << e << "   "
                << std::setw(10) << dev_n / full_n << "   " << std::setw(10)
                << du_rel << "   " << std::setw(10) << dz_rel << "     "
                << itsA << "/" << itsB << "  (" << rule << ")\n";
    }
  }

  if (Root()) {
    std::cout << "\n(the difference rows scale with e and with p/mu: the "
                 "quasi-hydrostatic\napproximation is safe when both are "
                 "small, and only then)\n";
  }
  table.Write();
  if (visualization && shown_e >= 0.0) {
    // The physical ellipse: a copy of the reference body moved by the
    // linear ellipse map (exact on the body). The fields are referential
    // (functions of the reference point), so they go across unchanged.
    MeshType ellipse(body);
    const double a = 1.0 + shown_e;
    VectorFunctionCoefficient map(dim, [a](const Vector& x, Vector& y) {
      y.SetSize(2);
      y(0) = a * x(0);
      y(1) = x(1) / a;
    });
    ellipse.Transform(map);
    const std::string tag = " (e = " + std::to_string(shown_e).substr(0, 5) + ")";
    examples::GLVisWindow("|dev S_e|" + tag, examples::DefaultKeys(dim))
        .Send(ellipse, dev_field);
    examples::GLVisWindow("u_full - u_hydro" + tag, examples::DefaultKeys(dim))
        .Send(ellipse, du_field);
  }
  return 0;
}
