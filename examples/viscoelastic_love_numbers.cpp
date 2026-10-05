// ============================================================================
// viscoelastic_love_numbers.cpp
//
// Load Love numbers of a homogeneous self-gravitating Maxwell sphere as
// functions of time, against the closed form.
//
// A surface mass load with one unit (l, 0) coefficient per degree l =
// lmin..lmax is switched on at t = 0 and held (a Heaviside load); the
// degrees are orthogonal, so one time integration serves them all. The
// sphere answers elastically at t = 0+ and then relaxes towards the fluid
// limit. ViscoelasticOperator steps the internal variables of
// IsotropicMaxwellRheology on LinearQuasiStaticMixedSelfGravitatingProblem,
// and at each output time the surface displacement and potential are
// analysed into harmonic coefficients, as in love_numbers.cpp:
//
//   h'_l(t) = -g u_l(t) / phi_sigma,     k'_l(t) = phi_l(t) / phi_sigma - 1.
//
// The reference: for the INCOMPRESSIBLE homogeneous sphere the elastic
// load Love numbers are (Wu & Peltier 1982)
//
//   h'_l = -(2l + 1)/3 f,  k'_l = -f,  f = 1 / (1 + mu_l),
//   mu_l = (2l^2 + 4l + 3) mu / (l rho g a),
//
// and by the correspondence principle the Maxwell sphere's Laplace
// transforms follow from mu -> mu s / (s + 1/tau). Inverting for a
// Heaviside load gives one exponential per degree:
//
//   f_l(t) = 1 - mu_l / (1 + mu_l) exp(-t / tau_l),  tau_l = (1 + mu_l) tau,
//
// from the elastic value 1 / (1 + mu_l) at t = 0+ to the fluid value 1.
// The degree's relaxation time tau_l is longer than the material's tau: the
// elastic stiffness mu_l, measured against gravity, holds the surface up
// while the mantle creeps.
//
// The finite-element sphere is COMPRESSIBLE (bulk modulus kappa), so it
// only approaches the closed form as kappa / mu grows: at kappa / mu = 200
// the elastic h'_2 is still 7% off, at the default 2000 it agrees to a few
// parts in a thousand. Two cautions:
//   - order 3 is needed. At order 2 the relaxed state is badly resolved on
//     this coarse mesh (h'_2 drifts to 12% off by t = 8 tau, while the
//     elastic value is within 3%); at order 3 the whole history agrees to
//     0.3% (l = 2), 0.1% (l = 3) and 1% (l = 4, the coarse mesh's limit).
//   - a uniform compressible sphere is unstably stratified, so once relaxed
//     it carries slowly GROWING buoyancy modes, weak at large kappa: keep
//     t_final to a few tau_l.
// The solver tolerance is looser than the library's default 1e-12: the
// comparison does not need more, and the solves (near-incompressible,
// order 3) are the cost: about a minute on 8 ranks at the defaults.
//
// Outputs: a table on the screen; viscoelastic_love_numbers.csv, the
// histories with their closed forms (python3 plot_csv.py
// viscoelastic_love_numbers.csv); with -vis, GLVis animations of the
// displacement and of the potential perturbation.
//
// One source serves the serial and the parallel build; the genuine
// differences are the mesh partitioning and the reductions of the mass.
//
// Sample runs (with mpiexec -np N in front in a parallel build):
//    ./viscoelastic_love_numbers
//    ./viscoelastic_love_numbers -kappa 100        (compressibility shows)
//    ./viscoelastic_love_numbers -o 2              (the relaxed state drifts)
//    ./viscoelastic_love_numbers -s sdirk23 -n 8   (another integrator)
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
using FieldType = ParGridFunction;
bool Root() { return Mpi::Root(); }
#else
using MeshType = Mesh;
using SubMeshType = SubMesh;
using SpaceType = FiniteElementSpace;
using FieldType = GridFunction;
bool Root() { return true; }
#endif

constexpr real_t kPi = std::numbers::pi_v<real_t>;

// Sum of a linear functional over the whole mesh: through the true-dof
// vector (shared dofs counted once), then reduced over the ranks.
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

// The incompressible Maxwell sphere's load Love numbers at time t after a
// Heaviside load.
struct Exact {
  real_t mu_l, tau_l;
  real_t f(real_t t) const {
    return 1.0 - mu_l / (1.0 + mu_l) * std::exp(-t / tau_l);
  }
};

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif

  const char* mesh_file = "../data/coupled_poisson.msh";
  int order = 3;
  int dtn_degree = 12;
  int lmin = 2, lmax = 4;
  real_t G = 1.0, rho = 1.0, kappa = 1000.0, mu = 0.5, tau = 1.0;
  real_t t_final = 6.0;
  int steps_per_tau = 4;
  int outputs_per_tau = 4;
  real_t rel_tol = 1e-6;
  const char* scheme = "exptrap";
  bool visualization = true;
  const char* csv_file = "viscoelastic_love_numbers.csv";

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
                 "Mesh file: a 3-D ball (attribute 1) in a buffer shell.");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&dtn_degree, "-deg", "--dtn-degree", "DtN expansion degree.");
  args.AddOption(&lmin, "-lmin", "--min-degree", "Lowest load degree (>= 2).");
  args.AddOption(&lmax, "-lmax", "--max-degree", "Highest load degree.");
  args.AddOption(&G, "-G", "--gravitational-constant", "G.");
  args.AddOption(&rho, "-rho", "--density", "Density.");
  args.AddOption(&kappa, "-kappa", "--bulk-modulus", "Bulk modulus.");
  args.AddOption(&mu, "-mu", "--shear-modulus", "Unrelaxed shear modulus.");
  args.AddOption(&tau, "-tau", "--maxwell-time", "Maxwell time eta / mu.");
  args.AddOption(&t_final, "-tf", "--t-final", "Final time, in Maxwell times.");
  args.AddOption(&scheme, "-s", "--scheme",
                 "Time integrator: exptrap (exponential trapezoid, one solve "
                 "per step) or sdirk23 (two).");
  args.AddOption(&steps_per_tau, "-n", "--steps-per-tau",
                 "Steps per Maxwell time.");
  args.AddOption(&outputs_per_tau, "-no", "--outputs-per-tau",
                 "Output times per Maxwell time (divides -n).");
  args.AddOption(&rel_tol, "-rt", "--rel-tol", "Relative solver tolerance.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization", "Animate in GLVis.");
  args.AddOption(&csv_file, "-csv", "--csv",
                 "Table of the histories for plot_csv.py (\"\": none).");
  args.Parse();
  if (!args.Good() || lmin < 2 || lmax < lmin ||
      steps_per_tau % outputs_per_tau != 0) {
    if (Root()) {
      args.PrintUsage(std::cout);
      std::cout << "(need 2 <= lmin <= lmax and -no dividing -n)\n";
    }
    return 1;
  }
  if (Root()) {
    args.PrintOptions(std::cout);
  }

  Mesh smesh(mesh_file, 1, 1);
  const int dim = smesh.Dimension();
  MFEM_VERIFY(dim == 3, "the closed form is for the 3-D sphere");
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
  ConstantCoefficient kappa_c(kappa), mu_c(mu), rho_c(rho), tau_c(tau);
  auto rheology = IsotropicMaxwellRheology::Maxwell(dim, kappa_c, mu_c, tau_c);
  LinearQuasiStaticMixedSelfGravitatingProblem problem(
      &fes_u, &fes_phi, rheology, rho_c, G, dtn_degree);
  problem.SetRelTol(rel_tol);

  // Harmonic analysis of u_r and phi on the surface.
  using BHC = BoundaryHarmonicCoefficients;
  BHC radial(fes_u, surface, lmax, BHC::Component::Radial);
  BHC scalar(problem.PotentialSpaceOnBody(), surface, lmax,
             BHC::Component::Scalar);
  const auto& basis = radial.Basis();
  const real_t a = radial.Radius();

  // Surface gravity g = 4 pi G M / |S|, with the mesh's mass.
  SpaceType fes_s(&body, &fec);
  LinearForm mass(&fes_s);
  mass.AddDomainIntegrator(new DomainLFIntegrator(rho_c));
  mass.Assemble();
  const real_t M = TrueSum(fes_s, mass);
  const real_t g = 4.0 * kPi * G * M / (4.0 * kPi * a * a);

  // The load: a unit (l, 0) coefficient at every degree, held from t = 0.
  std::vector<int> degrees, index;
  std::vector<Exact> exact;
  Vector unit(basis.Size());
  unit = 0.0;
  for (int l = lmin; l <= lmax; l++) {
    degrees.push_back(l);
    index.push_back(basis.Index(l, 0));
    unit[index.back()] = 1.0;
    const real_t mu_l = (2.0 * l * l + 4.0 * l + 3.0) * mu / (l * rho * g * a);
    exact.push_back({mu_l, (1.0 + mu_l) * tau});
  }
  auto sigma = radial.Expansion(unit, false);
  problem.SetSurfaceLoad(*sigma, surface);
  problem.AssembleForce(0.0);

  // The load's own potential on the same mesh (see love_numbers.cpp: k' is
  // a small remainder of phi, so the discretisation of the direct
  // potential must cancel).
  Vector phi_sigma;
  {
    FieldType phi_direct(
        static_cast<SpaceType*>(&problem.PotentialSpaceOnBody()));
    problem.SolveLoadPotential(phi_direct);
    scalar.Coefficients(phi_direct, phi_sigma);
  }

  if (Root()) {
    std::cout << "\nHomogeneous Maxwell sphere: radius " << a << ", surface "
              << "gravity " << g << ", rho g a / mu = " << rho * g * a / mu
              << ", mu / kappa = " << mu / kappa << "\n";
    for (std::size_t j = 0; j < degrees.size(); j++) {
      std::cout << "  degree " << degrees[j] << ": mu_l = " << exact[j].mu_l
                << ", relaxation time tau_l = " << exact[j].tau_l / tau
                << " tau\n";
    }
  }

  // The CSV columns: t, then h'_l with its closed form for every degree,
  // then the same for k'_l.
  std::vector<std::string> columns{"t"};
  std::string h_cols, k_cols;
  for (int l : degrees) {
    const std::string h = "h'_" + std::to_string(l);
    columns.push_back(h);
    columns.push_back(h + "_exact");
    h_cols += (h_cols.empty() ? "" : ",") + h;
  }
  for (int l : degrees) {
    const std::string k = "k'_" + std::to_string(l);
    columns.push_back(k);
    columns.push_back(k + "_exact");
    k_cols += (k_cols.empty() ? "" : ",") + k;
  }
  examples::CsvTable table(csv_file, columns);
  table.Meta("title", "Load Love numbers of a homogeneous Maxwell sphere")
      .Meta("note", "finite elements (solid, kappa/mu = " +
                        std::to_string(static_cast<int>(kappa / mu)) +
                        ") vs incompressible closed form (dashed)")
      .Meta("x", "t")
      .Meta("xlabel", "time after the load / Maxwell time")
      .Meta("y", h_cols + "|" + k_cols)
      .Meta("ylabel", "h'_l|k'_l");

  examples::GLVisWindow u_window("displacement (Heaviside load)",
                                 examples::DefaultKeys(dim));
  examples::GLVisWindow phi_window("potential perturbation",
                                   examples::DefaultKeys(dim));

  // Observe: the Love numbers of the current solution, printed and
  // tabulated, and the fields sent to GLVis.
  auto observe = [&](real_t t) {
    Vector cu, cphi;
    radial.Coefficients(problem.Displacement(), cu);
    scalar.Coefficients(problem.PotentialOnBody(), cphi);
    std::vector<double> row{t / tau}, h_row, k_row;
    std::vector<double> err(degrees.size());
    for (std::size_t j = 0; j < degrees.size(); j++) {
      const int i = index[j];
      const real_t h = -g * cu[i] / phi_sigma[i];
      const real_t k = cphi[i] / phi_sigma[i] - 1.0;
      const real_t f = exact[j].f(t);
      const real_t h_ex = -(2.0 * degrees[j] + 1.0) / 3.0 * f, k_ex = -f;
      h_row.insert(h_row.end(), {h, h_ex});
      k_row.insert(k_row.end(), {k, k_ex});
      err[j] = std::abs(h / h_ex - 1.0);
    }
    row.insert(row.end(), h_row.begin(), h_row.end());
    row.insert(row.end(), k_row.begin(), k_row.end());
    table.Row(row);
    if (Root()) {
      std::cout << std::fixed << std::setprecision(3) << std::setw(7)
                << t / tau;
      for (std::size_t j = 0; j < degrees.size(); j++) {
        std::cout << std::setw(10) << std::setprecision(4) << h_row[2 * j]
                  << std::setw(9) << h_row[2 * j + 1] << std::scientific
                  << std::setw(9) << std::setprecision(1) << err[j]
                  << std::fixed;
      }
      std::cout << "\n";
    }
    if (visualization) {
      u_window.Send(body, problem.Displacement());
      phi_window.Send(parent, problem.Potential());
    }
  };

  if (Root()) {
    std::cout << "\n  t/tau";
    for (int l : degrees) {
      std::cout << "     h'_" << l << "    exact  rel.err";
    }
    std::cout << "\n";
  }

  // Time stepping at a fixed step from rest, the elastic response at t = 0+
  // first. Both schemes are second order here; the exponential trapezoid
  // costs one solve per step against SDIRK23's two (it loses order only on
  // stiff bodies with several relaxation branches, not on one Maxwell
  // branch).
  ViscoelasticOperator visco(problem);
  std::unique_ptr<ODESolver> ode;
  if (std::string(scheme) == "sdirk23") {
    ode = std::make_unique<SDIRK23Solver>(2);
  } else {
    ode = std::make_unique<ExponentialTrapezoidSolver>();
  }
  ode->Init(visco);
  Vector m(visco.Height());
  m = 0.0;
  real_t t = 0.0;
  if (!visco.SolveElastic(m, t)) {
    if (Root()) {
      std::cerr << "Elastic solve failed.\n";
    }
    return 2;
  }
  observe(t);
  const int n_out = static_cast<int>(std::round(t_final * outputs_per_tau));
  const int sub = steps_per_tau / outputs_per_tau;
  const real_t dt_out = tau / outputs_per_tau;
  for (int k = 1; k <= n_out; k++) {
    for (int s = 0; s < sub; s++) {
      real_t dt = dt_out / sub;
      ode->Step(m, t, dt);
    }
    t = k * dt_out;  // exact, whatever the rounding of the sum
    if (!visco.SolveElastic(m, t)) {
      if (Root()) {
        std::cerr << "Elastic solve failed at t = " << t << "\n";
      }
      return 2;
    }
    observe(t);
  }
  if (Root()) {
    std::cout << "\nSolves " << problem.NumSolves() << ", assemblies "
              << problem.NumAssemblies() << "\n";
  }
  table.Write();
  return 0;
}
