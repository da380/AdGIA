// ============================================================================
// equilibrium_stress.cpp
//
// The equilibrium-stress generators of Al-Attar & Woodhouse (2010, GJI
// 181, 567) on a homogeneous elliptical planet: the equilibrium equations
// Div T = rho grad Phi0 with a traction-free surface underdetermine the
// stress, and the two generators pick the distinguished solutions
//
//   minimum norm       T = 2 mu grad_s u,  an elastic BVP (their §3.3);
//   minimum deviatoric T = -p 1 + 2 mu grad_s u,  the steady
//                      incompressible Stokes problem (their §3.4),
//                      solved with Taylor-Hood elements — the pressure
//                      space one order below the velocity space, since
//                      equal-order interpolation is inf-sup (LBB)
//                      unstable and produces spurious pressure modes.
//
// The geometry is the body of the canned disc mesh deformed to an
// ellipse with semi-axes a = 1 + e, b = 1/a. The self-gravity of the
// homogeneous elliptical cylinder is exact and linear in position,
//
//   rho grad Phi0 = 4 pi G rho^2 / (a + b) * (b x, a y),
//
// which reduces to the disc's 2 pi G rho^2 x at e = 0. The physics on
// display is Love's classical obstruction: a homogeneous ellipse admits
// NO hydrostatic equilibrium (its equipotentials are not parallel to its
// surface), so even the minimum-deviatoric stress carries genuine
// deviatoric content, of order the ellipticity — run with -e 0 and the
// deviatoric fraction collapses and the pressure reproduces the
// hydrostatic p0 = pi G rho^2 (1 - r^2). The printed norms also exhibit
// the two optimality properties: the minimum-norm field has the smaller
// full norm, the minimum-deviatoric field the (much) smaller deviatoric
// norm.
//
// With -vis (needs a running GLVis server): the minimum-deviatoric
// pressure, and the deviatoric magnitude |dev T| of both fields.
//
// One source serves the serial and the parallel build, as in
// gauged_fluid_cavity.cpp.
//
// Sample runs (with mpiexec -np N in front in a parallel build):
//    ./equilibrium_stress
//    ./equilibrium_stress -e 0
//    ./equilibrium_stress -e 0.4 -o 4
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
bool Root() { return Mpi::Root(); }
double GlobalSum(double v) {
  double g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return g;
}
#else
using MeshType = Mesh;
using SubMeshType = SubMesh;
using SpaceType = FiniteElementSpace;
bool Root() { return true; }
double GlobalSum(double v) { return v; }
#endif

constexpr double kG = 0.05;
constexpr double kRho = 1.0;
constexpr double kPi = std::numbers::pi;

double A = 1.2;  // semi-axis a; b = 1/a

// |dev S| of a matrix coefficient, for visualisation.
class DeviatoricNormCoefficient : public Coefficient {
 public:
  explicit DeviatoricNormCoefficient(MatrixCoefficient& S) : S_(&S) {}

  double Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    S_->Eval(M_, T, ip);
    const int d = M_.Height();
    double tr = 0.0;
    for (int i = 0; i < d; i++) {
      tr += M_(i, i);
    }
    double n2 = 0.0;
    for (int i = 0; i < d; i++) {
      for (int j = 0; j < d; j++) {
        const double v = M_(i, j) - (i == j ? tr / d : 0.0);
        n2 += v * v;
      }
    }
    return std::sqrt(n2);
  }

 private:
  MatrixCoefficient* S_;
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

void Show(Mesh& mesh, const GridFunction& f, const char* title) {
  char vishost[] = "localhost";
  socketstream sock(vishost, 19916);
  sock.precision(8);
#ifdef MFEM_USE_MPI
  sock << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() << "\n";
#endif
  sock << "solution\n"
       << mesh << f << "window_title '" << title << "'"
       << "\nkeys Rjlbc\n"
       << std::flush;
}

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif

  const char* mesh_file = "../data/elastogravity_2d.msh";
  int order = 3;
  double ellipticity = 0.2;
  bool visualization = true;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use (2-D).");
  args.AddOption(&order, "-o", "--order",
                 "Velocity/displacement order (pressure one lower).");
  args.AddOption(&ellipticity, "-e", "--ellipticity",
                 "Semi-axis a = 1 + e, b = 1/a (0 recovers the disc).");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization", "GLVis visualisation.");
  args.Parse();
  if (!args.Good()) {
    if (Root()) {
      args.PrintUsage(std::cout);
    }
    return 1;
  }
  A = 1.0 + ellipticity;
  const double b = 1.0 / A;

  Mesh smesh(mesh_file, 1, 1);
  MFEM_VERIFY(smesh.Dimension() == 2, "a 2-D example");
  const int dim = 2;
  smesh.Transform([](const Vector& x, Vector& y) {
    y.SetSize(2);
    y[0] = A * x[0];
    y[1] = x[1] / A;
  });
#ifdef MFEM_USE_MPI
  MeshType mesh(MPI_COMM_WORLD, smesh);
  smesh.Clear();
#else
  MeshType& mesh = smesh;
#endif
  Array<int> body_attr({1});
  auto body = SubMeshType::CreateFromDomain(mesh, body_attr);

  // Exact self-gravity of the homogeneous elliptical cylinder.
  const double a = A;
  VectorFunctionCoefficient f(dim, [a, b](const Vector& x, Vector& v) {
    const double c = 4.0 * kPi * kG * kRho * kRho / (a + b);
    v.SetSize(2);
    v[0] = c * b * x[0];
    v[1] = c * a * x[1];
  });

  H1_FECollection fec_u(order, dim), fec_p(order - 1, dim);
  SpaceType fes_u(&body, &fec_u, dim), fes_p(&body, &fec_p);

  MinimumNormEquilibriumStress T_mn(fes_u, f);
  MinimumDeviatoricEquilibriumStress T_md(fes_u, fes_p, f);

  double full_mn, dev_mn, full_md, dev_md;
  StressNorms(T_mn, body, order, full_mn, dev_mn);
  StressNorms(T_md, body, order, full_md, dev_md);

  if (Root()) {
    std::cout << "ellipse a = " << a << ", b = " << b << " (e = "
              << ellipticity << "), order " << order << "/" << order - 1
              << " Taylor-Hood\n"
              << "solver iterations: minimum-norm CG "
              << T_mn.SolverIterations() << ", minimum-deviatoric MINRES "
              << T_md.SolverIterations() << "\n"
              << std::setprecision(4)
              << "                     ||T||      ||dev T||   dev fraction\n"
              << "minimum norm:        " << std::setw(9) << full_mn << "  "
              << std::setw(9) << dev_mn << "  " << dev_mn / full_mn << "\n"
              << "minimum deviatoric:  " << std::setw(9) << full_md << "  "
              << std::setw(9) << dev_md << "  " << dev_md / full_md << "\n"
              << "(optimality: the first column is smallest in row 1, the "
                 "second in row 2)\n";
  }

  if (ellipticity == 0.0) {
    // The disc admits the hydrostatic state: the minimum-deviatoric
    // pressure must reproduce p0 = pi G rho^2 (1 - r^2).
    FunctionCoefficient p0([](const Vector& x) {
      return kPi * kG * kRho * kRho * (1.0 - (x * x));
    });
    ConstantCoefficient zero(0.0);
    auto& p = const_cast<GridFunction&>(T_md.Pressure());
    const double err = p.ComputeL2Error(p0) / p.ComputeL2Error(zero);
    if (Root()) {
      std::cout << "disc: relative L2 error of p against the hydrostatic "
                   "p0: "
                << err << "\n";
    }
  } else if (Root()) {
    std::cout << "Love's obstruction: no hydrostatic equilibrium exists on "
                 "the ellipse;\nthe minimum-deviatoric fraction "
              << std::setprecision(3) << dev_md / full_md
              << " is the unavoidable deviatoric stress (O(e); -e 0 "
                 "collapses it).\n";
  }

  if (visualization) {
    Show(body, const_cast<GridFunction&>(T_md.Pressure()),
         "Minimum-deviatoric pressure");
    L2_FECollection dfec(order - 1, dim);
    SpaceType dfes(&body, &dfec);
    DeviatoricNormCoefficient dev_md_c(T_md), dev_mn_c(T_mn);
#ifdef MFEM_USE_MPI
    ParGridFunction dev_gf(&dfes), dev_gf2(&dfes);
#else
    GridFunction dev_gf(&dfes), dev_gf2(&dfes);
#endif
    dev_gf.ProjectCoefficient(dev_md_c);
    Show(body, dev_gf, "|dev T|, minimum deviatoric");
    dev_gf2.ProjectCoefficient(dev_mn_c);
    Show(body, dev_gf2, "|dev T|, minimum norm");
  }

  return 0;
}
