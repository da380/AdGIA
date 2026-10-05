// ============================================================================
// love_numbers.cpp
//
// Load and tidal Love numbers of a homogeneous self-gravitating elastic
// sphere (3-D) or disc (2-D), one solve per degree, through
// LinearQuasiStaticMixedSelfGravitatingProblem and the harmonic analysis of
// spherical_harmonics.hpp.
//
// For each degree l (order m = 0; in 2-D the cosine mode) the surface load
// is set to sigma = Y_l0 (unit coefficient) and, separately, the tidal
// potential to psi = (r/a)^l Y_l0. The radial surface displacement and the
// potential perturbation on the surface are analysed into harmonic
// coefficients, and with the code's sign convention (force = -grad phi, so
// the potential of a positive mass is negative):
//
//   load:   h'_l = -g u_l / phi_sigma            k'_l = phi_l / phi_sigma - 1
//   tidal:  h_l  = -g u_l                        k_l  = phi_l
//
// where u_l, phi_l are the (l, 0) coefficients of u_r and phi on the surface,
// g = 4 pi G M / |surface| the surface gravity, and phi_sigma the load's own
// potential on the surface. The latter is computed on the same mesh for the
// load alone (SolveLoadPotential), not from its exact value
// -4 pi G a / (2l + 1) (3-D) or -2 pi G a / l (2-D): phi_l is dominated by
// the load's direct potential, and k' is its small remainder, so the
// discretisation error of the direct potential must cancel. The ratio of
// the discrete to the exact direct potential is printed as "phi_s". Both
// sets come out positive (h, k) or negative (h', k') as usual.
//
// In 3-D the incompressible homogeneous sphere has closed-form Love numbers
// (Wu & Peltier 1982): with mu_l = (2l^2 + 4l + 3) mu / (l rho g a),
//   h_l = (2l+1)/(2(l-1)) / (1 + mu_l)     k_l = 3/(2(l-1)) / (1 + mu_l)
//   h'_l = -(2l+1)/3 / (1 + mu_l)          k'_l = -1 / (1 + mu_l)
// which the run compares against when the bulk modulus is large compared
// with the shear modulus (-kappa). Degree 1 load Love numbers depend on the
// rigid gauge (here the default: u orthogonal to the rigid modes in the
// true-dof inner product) and are flagged.
//
// The largest coefficient of the solution at any other (l', m') is printed
// as a measure of the mesh's departure from spherical symmetry.
//
// One source serves the serial and the parallel build; the harmonic
// analysis layer is parallel-aware by itself, so the genuine differences
// are only the mesh partitioning, the total-mass reduction and the
// guarded printing.
//
// Outputs: the table on the screen; love_numbers.csv, the Love numbers by
// degree (with the incompressible sphere's in 3-D; python3 plot_csv.py
// love_numbers.csv); with -vis, the displacement and the potential
// perturbation of the load solve at degree -vl in GLVis.
//
// Sample runs (with mpiexec -np N in front in a parallel build):
//    ./love_numbers -o 2 -lmax 6
//    ./love_numbers -m ../data/coupled_poisson.msh -o 2 -lmax 4 -kappa 100
// ============================================================================
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
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
bool Root() { return Mpi::Root(); }
#else
using MeshType = Mesh;
using SubMeshType = SubMesh;
using SpaceType = FiniteElementSpace;
bool Root() { return true; }
#endif

constexpr real_t kPi = std::numbers::pi_v<real_t>;

// Sum of a linear functional over the whole mesh: through the true-dof
// vector (shared dofs counted once), then reduced over the ranks that
// own the pieces.
real_t TrueSum(FiniteElementSpace& fes, LinearForm& lf) {
  const Operator* P = fes.GetProlongationMatrix();
  real_t local = 0.0;
  if (!P) {
    local = lf.Sum();
  } else {
    Vector T(fes.GetTrueVSize());
    P->MultTranspose(lf, T);
    local = T.Sum();
  }
#ifdef MFEM_USE_MPI
  real_t global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                MPI_COMM_WORLD);
  return global;
#else
  return local;
#endif
}

struct Analytic {
  real_t h, k, h_load, k_load;
};

// Incompressible homogeneous sphere, l >= 2.
Analytic IncompressibleSphere(int l, real_t mu, real_t rho, real_t g,
                              real_t a) {
  const real_t mu_l = (2.0 * l * l + 4.0 * l + 3.0) * mu / (l * rho * g * a);
  const real_t f = 1.0 / (1.0 + mu_l);
  return {(2.0 * l + 1.0) / (2.0 * (l - 1.0)) * f,
          3.0 / (2.0 * (l - 1.0)) * f, -(2.0 * l + 1.0) / 3.0 * f, -f};
}

// The largest |c_i| over i != main, relative to |c_main|.
real_t Spurious(const Vector& c, int main) {
  real_t m = 0.0;
  for (int i = 0; i < c.Size(); i++) {
    if (i != main) {
      m = std::max(m, std::abs(c[i]));
    }
  }
  return m / (std::abs(c[main]) + 1e-300);
}

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif

  const char* mesh_file = "../data/elastogravity_2d.msh";
  int order = 2;
  int dtn_degree = 12;
  int lmin = 2, lmax = 6;
  real_t G = 0.05, rho = 1.0, kappa = 100.0, mu = 0.5, rel_tol = 1e-10;
  bool analytic = true;
  bool visualization = true;
  int vis_degree = 2;
  const char* csv_file = "love_numbers.csv";

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file (ball in a ball).");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&dtn_degree, "-deg", "--dtn-degree", "DtN expansion degree.");
  args.AddOption(&lmin, "-lmin", "--min-degree", "Lowest harmonic degree.");
  args.AddOption(&lmax, "-lmax", "--max-degree", "Highest harmonic degree.");
  args.AddOption(&G, "-G", "--gravitational-constant", "G.");
  args.AddOption(&rho, "-rho", "--density", "Density.");
  args.AddOption(&kappa, "-kappa", "--bulk-modulus", "Bulk modulus.");
  args.AddOption(&mu, "-mu", "--shear-modulus", "Shear modulus.");
  args.AddOption(&rel_tol, "-rt", "--rel-tol", "Relative solver tolerance.");
  args.AddOption(&analytic, "-analytic", "--analytic", "-no-analytic",
                 "--no-analytic",
                 "Compare with the incompressible homogeneous sphere (3-D).");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization", "Show one degree's load solution in GLVis.");
  args.AddOption(&vis_degree, "-vl", "--vis-degree",
                 "The degree whose load solution is shown.");
  args.AddOption(&csv_file, "-csv", "--csv",
                 "Table of the Love numbers for plot_csv.py (\"\": none).");
  args.Parse();
  if (!args.Good()) {
    if (Root()) {
      args.PrintUsage(std::cout);
    }
    return 1;
  }
  if (Root()) {
    args.PrintOptions(std::cout);
  }

  Mesh smesh(mesh_file, 1, 1);
  const int dim = smesh.Dimension();
#ifdef MFEM_USE_MPI
  MeshType parent(MPI_COMM_WORLD, smesh);
  smesh.Clear();
#else
  MeshType& parent = smesh;
#endif
  Array<int> body_marker(parent.attributes.Max());
  body_marker = 0;
  body_marker[0] = 1;
  auto body = SubMeshType::CreateFromDomain(parent, body_marker);
  Array<int> surface(body.bdr_attributes.Max());
  surface = 0;
  surface[body.bdr_attributes.Max() - 1] = 1;

  H1_FECollection fec(order, dim);
  SpaceType fes_u(&body, &fec, dim), fes_phi(&parent, &fec);
  ConstantCoefficient kappa_c(kappa), mu_c(mu), rho_c(rho);
  IsotropicElasticRheology rheology(dim, kappa_c, mu_c);
  LinearQuasiStaticMixedSelfGravitatingProblem problem(&fes_u, &fes_phi, rheology,
                                                  rho_c, G, dtn_degree);
  problem.SetRelTol(rel_tol);

  // Harmonic analysis of u_r and phi on the surface.
  using BHC = BoundaryHarmonicCoefficients;
  BHC radial(fes_u, surface, lmax, BHC::Component::Radial);
  BHC scalar(problem.PotentialSpaceOnBody(), surface, lmax,
             BHC::Component::Scalar);
  const auto& basis = radial.Basis();
  const real_t a = radial.Radius();

  // Surface gravity g = 4 pi G M / |S| (mass reduced over ranks).
  SpaceType fes_s(&body, &fec);
  LinearForm mass(&fes_s);
  mass.AddDomainIntegrator(new DomainLFIntegrator(rho_c));
  mass.Assemble();
  const real_t M = TrueSum(fes_s, mass);
  const real_t area = dim == 2 ? 2.0 * kPi * a : 4.0 * kPi * a * a;
  const real_t g = 4.0 * kPi * G * M / area;
  if (Root()) {
    std::cout << dim << "-D body: radius " << a << ", mass " << M
              << ", surface gravity " << g << ", rho g a / mu = "
              << rho * g * a / mu << "\n";
  }

  // One load and one tidal potential, with coefficients switched per degree.
  Vector zero(basis.Size());
  zero = 0.0;
  auto sigma = radial.Expansion(zero, false);
  auto psi = scalar.Expansion(zero, true);
  problem.SetSurfaceLoad(*sigma, surface);
  problem.SetTidalPotential(*psi);

  auto solve = [&](int i, bool load) -> std::pair<Vector, Vector> {
    Vector c(basis.Size());
    c = 0.0;
    c[i] = 1.0;
    sigma->SetCoefficients(load ? c : zero);
    psi->SetCoefficients(load ? zero : c);
    problem.AssembleForce(0.0);
    if (!problem.Solve() && Root()) {
      std::cerr << "Solve failed at coefficient " << i << "\n";
    }
    Vector cu, cphi;
    radial.Coefficients(problem.Displacement(), cu);
    scalar.Coefficients(problem.PotentialOnBody(), cphi);
    return {cu, cphi};
  };

  const bool compare = analytic && dim == 3;
  if (Root()) {
    std::cout << std::setprecision(6);
  }
  GridFunction phi_direct(&problem.PotentialSpaceOnBody());
  std::vector<std::string> columns{"l", "h'", "k'", "h", "k"};
  if (compare) {
    columns.insert(columns.end(), {"h'_exact", "k'_exact", "h_exact", "k_exact"});
  }
  examples::CsvTable table(csv_file, columns);
  table.Meta("title", std::string("Love numbers of a homogeneous ") +
                          (dim == 2 ? "disc" : "sphere"))
      .Meta("x", "l")
      .Meta("xlabel", "degree l")
      .Meta("y", "h',k'|h,k")
      .Meta("ylabel", "load|tidal");
  if (compare) {
    table.Meta("note", "dashed: incompressible sphere (Wu & Peltier 1982)");
  }
  if (Root()) {
    std::cout << "\n  l         h'          k'           h           k"
              << "   spurious   phi_s";
    if (compare) {
      std::cout << "   |  incompressible sphere: h' k' h k";
    }
    std::cout << "\n";
  }
  for (int l = std::max(lmin, dim == 2 ? 1 : 0); l <= lmax; l++) {
    const int i = basis.Index(l, dim == 2 ? l : 0);
    real_t h_load = NAN, k_load = NAN, h = NAN, k = NAN, spurious = 0.0,
           phi_ratio = NAN;
    {
      auto [cu, cphi] = solve(i, true);
      if (visualization && l == vis_degree) {
        const std::string tag = " (load, degree " + std::to_string(l) + ")";
        examples::GLVisWindow("displacement" + tag, examples::DefaultKeys(dim))
            .Send(body, problem.Displacement());
        examples::GLVisWindow("potential perturbation" + tag,
                              examples::DefaultKeys(dim))
            .Send(parent, problem.Potential());
      }
      problem.SolveLoadPotential(phi_direct);
      Vector cdirect;
      scalar.Coefficients(phi_direct, cdirect);
      const real_t phi_sigma = cdirect[i];
      const real_t phi_exact = dim == 2 ? -2.0 * kPi * G * a / l
                                        : -4.0 * kPi * G * a / (2.0 * l + 1.0);
      phi_ratio = phi_sigma / phi_exact;
      h_load = -g * cu[i] / phi_sigma;
      k_load = cphi[i] / phi_sigma - 1.0;
      spurious = std::max(Spurious(cu, i), Spurious(cphi, i));
    }
    if (l >= 2) {
      auto [cu, cphi] = solve(i, false);
      h = -g * cu[i];
      k = cphi[i];
      spurious = std::max({spurious, Spurious(cu, i), Spurious(cphi, i)});
    }
    if (l >= 2) {
      std::vector<double> row{static_cast<double>(l), h_load, k_load, h, k};
      if (compare) {
        const auto ref = IncompressibleSphere(l, mu, rho, g, a);
        row.insert(row.end(), {ref.h_load, ref.k_load, ref.h, ref.k});
      }
      table.Row(row);
    }
    if (Root()) {
      std::cout << std::setw(3) << l << std::setw(12) << h_load
                << std::setw(12) << k_load << std::setw(12) << h
                << std::setw(12) << k << std::setw(11)
                << std::setprecision(2) << spurious << std::setw(8)
                << std::setprecision(4) << phi_ratio << std::setprecision(6);
      if (compare && l >= 2) {
        const auto ref = IncompressibleSphere(l, mu, rho, g, a);
        std::cout << "   | " << std::setw(10) << ref.h_load << std::setw(10)
                  << ref.k_load << std::setw(10) << ref.h << std::setw(10)
                  << ref.k;
      }
      if (l == 1) {
        std::cout << "   (degree 1: gauge dependent)";
      }
      std::cout << "\n";
    }
  }
  table.Write();
  return 0;
}
