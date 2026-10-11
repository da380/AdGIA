// ============================================================================
// gia_kernels.cpp
//
// THE ADJOINT DEMONSTRATOR: first-order GIA sensitivity kernels on an
// idealised, well-resolved configuration — everything analytic, nothing
// ingested — showing the adjoint side of the library end to end
// (doc/planning/adjoint_plan.md; the machinery is adjoint.hpp, gated in
// TestAdjoint*).
//
// The model: a homogeneous Maxwell planet (uniform kappa, mu, rho; a
// 1-D viscosity background tau) with the monolithic water load. The
// geography is a smooth polar CONTINENT (super-Gaussian cap, the ocean
// function C0 = 1 - continent, frozen shorelines) carrying a Gaussian
// ICE BLOB that melts linearly to time t1 — a simple analytical load.
// The observation is the present-day (t = t1) windowed-mean sea level
// at a distant ocean site. One adjoint sweep then yields:
//
//  - the VISCOSITY kernel dJ/d eta(x): a 3-D volume field, exported as
//    lon-lat maps on depth shells, as a great-circle SLICE through the
//    observation meridian (which also contains the blob), and as
//    mesh + .gf for GLVis cutting planes;
//  - the quelled ICE-RATE kernels D_Idot(x, t) of Yu et al. (2025,
//    eq. 84) at chosen epochs (with frozen shorelines the I0 kernel
//    vanishes identically);
//  - with -taylor, the TAYLOR TEST: exact J(m0 + s dm) against the
//    affine prediction J0 + s dJ[dm] for a mantle viscosity blob and
//    an ice-rate pattern (taylor_test.csv / taylor_remainder.csv for
//    plot_csv.py).
//
// Defaults resolve the kernels honestly: order 2 (order 1's
// internal-variable space breaks localised viscoelastic kernels — the
// adjoint note's order finding; the run warns) and one uniform
// refinement of the ball.
//
// Maps (the exported CSVs are WriteSurfaceField's format):
//   <build>/earth_data/plot_fingerprint_kernel visc_kernel_r*.csv ...
//       --no-mask --global-only --side-by-side
//   glvis -m visc_kernel.mesh -g visc_kernel.gf     (cutting planes)
//
// One source serves the serial and the parallel build; the depth
// shells and the slice are exported by serial runs (the .gf gathers in
// parallel too).
//
// Options (defaults in brackets):
//   -m    ball-in-buffer mesh [../data/coupled_poisson.msh].
//   -o    order [2];  -r  uniform refinements [1].
//   -tau  Maxwell time [1];  -tf  final time, Maxwell times [3].
//   -n    time steps [24].
//   -rhow / -rhoi  water and ice densities [0.05, 0.045].
//   -cont-width    continent super-Gaussian width (rad) [0.9].
//   -blob-width    ice-blob Gaussian width (rad) [0.35].
//   -melt          melted blob amplitude by t1 [1.0].
//   -obs-lat -obs-lon  observation site (deg) [25, 0].
//   -obs-width     observation window width (rad) [0.2].
//   -nk   D_Idot epochs exported [4];  -nd  depth shells [3].
//   -taylor  Taylor points per side [0: off];  -smax  largest |s| [1].
//   -rt   solver tolerance [1e-10];  -vis / -no-vis  GLVis [on].
//
// Sample runs:
//   ./gia_kernels -no-vis
//   ./gia_kernels -taylor 3 -no-vis
//   mpiexec -np 8 ./gia_kernels -r 2 -no-vis   (finer; shells skipped)
// ============================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "AdGIA.hpp"
#include "AdGIA/detail/fem_factory.hpp"
#include "visualisation.hpp"

using namespace mfem;
using namespace AdGIA;
using namespace examples;

namespace {

constexpr double kG = 0.05;
constexpr double kRho = 1.0;
constexpr double kKappa = 1.0;
constexpr double kMu = 0.5;
constexpr int kDtNDegree = 12;
constexpr double kPi = std::numbers::pi;

double GlobalSum(double v) {
#ifdef MFEM_USE_MPI
  double g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return g;
#else
  return v;
#endif
}

// Colatitude from the +z pole.
double Theta(const Vector& x) {
  const double r = x.Norml2();
  return std::acos(std::min(1.0, std::max(-1.0, x[2] / r)));
}

// Angle from a unit direction n.
double AngleFrom(const Vector& x, const Vector& n) {
  const double c = (x * n) / x.Norml2();
  return std::acos(std::min(1.0, std::max(-1.0, c)));
}

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif

  const char* mesh_file = "../data/coupled_poisson.msh";
  int order = 2, ref_levels = 1, n_steps = 24;
  double tau_m = 1.0, t_final = 3.0;
  double rho_w = 0.05, rho_i = 0.045;
  double cont_width = 0.9, blob_width = 0.35, melt = 1.0;
  double obs_lat = 25.0, obs_lon = 0.0, obs_width = 0.2;
  int n_epochs = 4, n_depths = 3;
  int taylor_points = 0;
  double taylor_smax = 1.0;
  double rel_tol = 1e-10;
  bool visualization = true;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Ball-in-buffer mesh.");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&ref_levels, "-r", "--refine", "Uniform refinements.");
  args.AddOption(&tau_m, "-tau", "--maxwell-time", "Maxwell time.");
  args.AddOption(&t_final, "-tf", "--t-final", "Final time (tau units).");
  args.AddOption(&n_steps, "-n", "--steps", "Time steps.");
  args.AddOption(&rho_w, "-rhow", "--water-density", "Water density.");
  args.AddOption(&rho_i, "-rhoi", "--ice-density", "Ice density.");
  args.AddOption(&cont_width, "-cont-width", "--continent-width",
                 "Continent super-Gaussian width (rad).");
  args.AddOption(&blob_width, "-blob-width", "--blob-width",
                 "Ice-blob Gaussian width (rad).");
  args.AddOption(&melt, "-melt", "--melt-amplitude",
                 "Melted blob amplitude by t1.");
  args.AddOption(&obs_lat, "-obs-lat", "--observation-latitude",
                 "Observation latitude (deg).");
  args.AddOption(&obs_lon, "-obs-lon", "--observation-longitude",
                 "Observation longitude (deg).");
  args.AddOption(&obs_width, "-obs-width", "--observation-width",
                 "Observation window width (rad).");
  args.AddOption(&n_epochs, "-nk", "--kernel-epochs",
                 "D_Idot epochs exported.");
  args.AddOption(&n_depths, "-nd", "--kernel-depths",
                 "Viscosity-kernel depth shells (serial runs).");
  args.AddOption(&taylor_points, "-taylor", "--taylor-points",
                 "Taylor points per side (0: off).");
  args.AddOption(&taylor_smax, "-smax", "--taylor-smax", "Largest |s|.");
  args.AddOption(&rel_tol, "-rt", "--solver-tolerance",
                 "Relative solver tolerance.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization", "GLVis windows.");
  args.Parse();
  if (!args.Good()) {
    if (IsRoot()) {
      args.PrintUsage(std::cout);
    }
    return 1;
  }
  if (IsRoot()) {
    args.PrintOptions(std::cout);
  }
  if (order < 2 && IsRoot()) {
    std::cout << "WARNING: order 1 internal variables break localised "
                 "viscoelastic kernels\n(doc/planning/adjoint_note.md); "
                 "use -o 2.\n";
  }

  Mesh smesh(mesh_file, 1, 1);
  for (int l = 0; l < ref_levels; l++) {
    smesh.UniformRefinement();
  }
  MFEM_VERIFY(smesh.Dimension() == 3, "a 3-D ball mesh is required");
#ifdef MFEM_USE_MPI
  ParMesh parent(MPI_COMM_WORLD, smesh);
  smesh.Clear();
#else
  Mesh& parent = smesh;
#endif
  Array<int> body_attrs_list;
  body_attrs_list.Append(1);
#ifdef MFEM_USE_MPI
  ParSubMesh body(ParSubMesh::CreateFromDomain(parent, body_attrs_list));
  ParFiniteElementSpace* pfes = nullptr;
#else
  SubMesh body(SubMesh::CreateFromDomain(parent, body_attrs_list));
#endif
  const int dim = 3;
  H1_FECollection fec(order, dim);
#ifdef MFEM_USE_MPI
  ParFiniteElementSpace fes_u(&body, &fec, dim), fes_phi(&parent, &fec);
#else
  FiniteElementSpace fes_u(&body, &fec, dim), fes_phi(&parent, &fec);
#endif
  Array<int> surface(body.bdr_attributes.Max());
  surface = 0;
  surface[body.bdr_attributes.Max() - 1] = 1;

  SeaLevelOperator sea(fes_u, surface);
  sea.SetDensities(rho_w, rho_i);

  // The analytic geography: polar continent, ocean elsewhere, ice blob
  // at the pole melting linearly.
  const double g0 = 4.0 * kPi * kG * kRho / 3.0;
  const double cw = cont_width, bw = blob_width;
  auto continent = [cw](const Vector& x) {
    const double q = Theta(x) / cw;
    return std::exp(-q * q * q * q);
  };
  auto ocean = [continent](const Vector& x) {
    return 1.0 - continent(x);
  };
  auto blob = [bw](const Vector& x) {
    const double q = Theta(x) / bw;
    return std::exp(-q * q);
  };
  FunctionCoefficient w_ocean(
      [ocean, g0, rho_w](const Vector& x) {
        return rho_w * ocean(x) / g0;
      });
  const double tf = t_final, melt_amp = melt;
  FunctionCoefficient sigma_d(
      [ocean, blob, rho_i, melt_amp, tf](const Vector& x, double t) {
        // sigma_d = rho_i (1 - C0) dI(t), dI = -melt blob t / t1.
        return rho_i * (1.0 - ocean(x)) * (-melt_amp) * blob(x) * (t / tf);
      });
  VectorFunctionCoefficient grad_phi0(dim,
                                      [g0](const Vector& x, Vector& v) {
                                        v = x;
                                        v *= g0;
                                      });

  // The Maxwell body with the monolithic water load (1-D viscosity).
  ConstantCoefficient kappa(kKappa), mu(kMu), rho(kRho), tau(tau_m);
  auto rheology = IsotropicMaxwellRheology::Maxwell(dim, kappa, mu, tau);
  LinearQuasiStaticMixedSelfGravitatingProblem prob(&fes_u, &fes_phi,
                                                    rheology, rho, kG,
                                                    kDtNDegree);
  prob.SetWaterLoad(w_ocean, sigma_d, surface);
  prob.SetRelTol(rel_tol);

  ViscoelasticOperator visco(prob);
  ExponentialTrapezoidSolver ode;
  ode.Init(visco);
  Vector m(visco.Height());
  m = 0.0;
  double t = 0.0;
  double dt = t_final / n_steps;

  ForwardTrajectory trajectory(visco, true);
  trajectory.Record(m, t);
  for (int step = 0; step < n_steps; step++) {
    ode.Step(m, t, dt);
    trajectory.Record(m, t);
  }
  if (!visco.SolveElastic(m, t)) {
    if (IsRoot()) {
      std::cout << "forward solve failed\n";
    }
    return 1;
  }

  // The observation: windowed-mean present-day sea level at the site
  // (unit-mass window; the strict three-slot dual).
  constexpr double deg = kPi / 180.0;
  const double cl = std::cos(obs_lat * deg), slat = std::sin(obs_lat * deg);
  const double cn = std::cos(obs_lon * deg), sn = std::sin(obs_lon * deg);
  Vector n_obs(3);
  n_obs[0] = cl * cn;
  n_obs[1] = cl * sn;
  n_obs[2] = slat;
  const double ow = obs_width;
  auto window = [n_obs, ow](const Vector& x) {
    const double a = AngleFrom(x, n_obs);
    return std::exp(-a * a / (ow * ow));
  };
  FunctionCoefficient h_raw(window);
  Array<int> parent_surface(parent.bdr_attributes.Max());
  parent_surface = 0;
  parent_surface[0] = 1;
  auto lf_mass = detail::MakeLinearForm(&fes_phi);
  lf_mass->AddBoundaryIntegrator(new BoundaryLFIntegrator(h_raw),
                                 parent_surface);
  lf_mass->Assemble();
  const double h_mass = GlobalSum(lf_mass->Sum());
  auto h_u = VectorFunctionCoefficient(
      dim, [window, h_mass](const Vector& x, Vector& v) {
        v = x;
        v *= -window(x) / (h_mass * x.Norml2());
      });
  auto g_u = detail::MakeLinearForm(&fes_u);
  g_u->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(h_u), surface);
  g_u->Assemble();
  FunctionCoefficient h_phi([window, h_mass, g0](const Vector& x) {
    return -window(x) / (h_mass * g0);
  });
  auto g_phi = detail::MakeLinearForm(&fes_phi);
  g_phi->AddBoundaryIntegrator(new BoundaryLFIntegrator(h_phi),
                               parent_surface);
  g_phi->Assemble();
  const double J0 = GlobalSum(*g_u * prob.Displacement()) +
                    GlobalSum(*g_phi * prob.Potential()) +
                    prob.UniformPotentialTerm() / g0;
  if (IsRoot()) {
    std::cout << "\nwindowed-mean present-day sea level at ("
              << obs_lat << " N, " << obs_lon << " E): J = " << J0
              << "\n";
  }

  AdjointForce force;
  AdjointLoad load;
  load.u = g_u.get();
  load.phi = g_phi.get();
  load.uniform = 1.0 / g0;
  force.AddObservation(trajectory.Time(trajectory.Size() - 1), load);

  // The zero-data adjoint twin.
  FunctionCoefficient w_adj(
      [ocean, g0, rho_w](const Vector& x) {
        return rho_w * ocean(x) / g0;
      });
  ConstantCoefficient zero_sigma(0.0);
  LinearQuasiStaticMixedSelfGravitatingProblem prob_adj(
      &fes_u, &fes_phi, rheology, rho, kG, kDtNDegree);
  prob_adj.SetWaterLoad(w_adj, zero_sigma, surface);
  prob_adj.SetRelTol(rel_tol);
  ViscoelasticOperator visco_adj(prob_adj);

  // Kernels: viscosity (volume) and the quelled ice rate.
  H1_FECollection sfec(order, dim);
  auto sfes_s = detail::MakeFESpace(fes_u, &sfec);
  IsotropicKernel K_eta(IsotropicKernel::Parameter::Eta, *sfes_s, &tau);
  L2_FECollection kfec(order - 1, dim);
  auto kfes = detail::MakeFESpace(fes_u, &kfec);
  K_eta.SetField(*kfes);
  SeaLevelLoadKernel K_ice(prob_adj, sea, grad_phi0, g0, false);
  FunctionCoefficient C_t([ocean](const Vector& x) { return ocean(x); });
  FunctionCoefficient C_0([ocean](const Vector& x) { return ocean(x); });
  K_ice.EnableQuelled(C_t, C_0, rho_i);

  // Taylor directions: a mantle viscosity blob in the observation
  // meridian plane, and an ice-rate pattern on the continent flank.
  Vector c_blob(3);
  {
    // Under the ice blob (the pole), mid-mantle: where the viscous
    // flow of the unloading concentrates, so the Taylor test probes a
    // direction J actually bends along.
    c_blob = 0.0;
    c_blob[2] = 0.7;
  }
  auto vblob = [c_blob](const Vector& x) {
    Vector d(x);
    d -= c_blob;
    return std::exp(-(d * d) / (0.3 * 0.3));
  };
  FunctionCoefficient vblob_c(vblob);
  ProductCoefficient d_eta(kMu * tau_m, vblob_c);
  const int i_eta_dir = K_eta.AddDirection(d_eta);
  const double bw2 = 2.0 * blob_width;
  auto icap = [bw2](const Vector& x) {
    const double q = Theta(x) / bw2;
    return std::exp(-q * q);
  };
  FunctionCoefficient dsig_ice(
      [ocean, icap, rho_i](const Vector& x, double t) {
        // delta I = icap t (a unit ice-rate pattern).
        return rho_i * (1.0 - ocean(x)) * icap(x) * t;
      });
  const int i_ice_dir = K_ice.AddDirection(dsig_ice);

  AdjointSolver solver(visco_adj, trajectory);
  solver.SetForce(force);
  solver.AddKernel(K_eta);
  solver.AddKernel(K_ice);
  if (!solver.Run()) {
    if (IsRoot()) {
      std::cout << "adjoint sweep failed\n";
    }
    return 1;
  }
  K_ice.Finalize();

  // === Exports ===
  int world = 1;
#ifdef MFEM_USE_MPI
  world = Mpi::WorldSize();
#endif
  std::vector<std::string> epoch_names, shell_names;
  for (int k = 0; k < n_epochs; k++) {
    const int nt = K_ice.NumTimes();
    const int idx =
        nt <= 1 ? 0 : (k * (nt - 1)) / std::max(1, n_epochs - 1);
    std::ostringstream name;
    name << "ice_kernel_t" << std::fixed << std::setprecision(2)
         << K_ice.Time(idx) << ".csv";
    sea.WriteSurfaceField(K_ice.IdotKernel(idx), name.str());
    epoch_names.push_back(name.str());
  }
  if (world == 1) {
    // Depth shells and the observation-meridian slice (which contains
    // the blob at the pole).
    Mesh& smesh_s = sea.SurfaceMesh();
    FiniteElementSpace vfes_s(&smesh_s, sea.SurfaceSpace().FEColl(), 3);
    GridFunction xyz(&vfes_s);
    VectorFunctionCoefficient idc(3,
                                  [](const Vector& x, Vector& v) { v = x; });
    xyz.ProjectCoefficient(idc);
    const int nd_s = sea.SurfaceSpace().GetVSize();
    for (int k = 0; k < n_depths; k++) {
      const double rk =
          n_depths == 1 ? 0.95
                        : 0.45 + (0.95 - 0.45) * k / (n_depths - 1.0);
      DenseMatrix pts(3, nd_s);
      for (int i = 0; i < nd_s; i++) {
        Vector dir(3);
        dir[0] = xyz[i];
        dir[1] = xyz[i + nd_s];
        dir[2] = xyz[i + 2 * nd_s];
        dir /= dir.Norml2();
        for (int c = 0; c < 3; c++) {
          pts(c, i) = rk * dir[c];
        }
      }
      Array<int> elems;
      Array<IntegrationPoint> ips;
      body.FindPoints(pts, elems, ips, false);
      GridFunction shell(&sea.SurfaceSpace());
      shell = 0.0;
      for (int i = 0; i < nd_s; i++) {
        if (elems[i] >= 0) {
          shell[i] = K_eta.Field().GetValue(elems[i], ips[i]);
        }
      }
      std::ostringstream name;
      name << "visc_kernel_r" << std::fixed << std::setprecision(2) << rk
           << ".csv";
      sea.WriteSurfaceField(shell, name.str());
      shell_names.push_back(name.str());
    }
    // The slice: the plane through the observation and the pole.
    {
      Vector e1(n_obs), ez(3), e2(3);
      ez = 0.0;
      ez[2] = 1.0;
      const double z1 = ez * e1;
      e2 = ez;
      e2.Add(-z1, e1);
      e2 /= e2.Norml2();
      const int na = 181, nr = 48;
      DenseMatrix pts(3, na * nr);
      for (int ia = 0; ia < na; ia++) {
        const double a = -kPi + 2.0 * kPi * ia / (na - 1);
        for (int ir = 0; ir < nr; ir++) {
          const double rr = 0.05 + (0.995 - 0.05) * ir / (nr - 1.0);
          const int col = ia * nr + ir;
          for (int c = 0; c < 3; c++) {
            pts(c, col) =
                rr * (std::cos(a) * e1[c] + std::sin(a) * e2[c]);
          }
        }
      }
      Array<int> elems;
      Array<IntegrationPoint> ips;
      body.FindPoints(pts, elems, ips, false);
      std::ofstream slice("visc_kernel_slice.csv");
      slice << "alpha_deg,r,value\n";
      slice.precision(16);
      for (int ia = 0; ia < na; ia++) {
        const double ad = -180.0 + 360.0 * ia / (na - 1);
        for (int ir = 0; ir < nr; ir++) {
          const double rr = 0.05 + (0.995 - 0.05) * ir / (nr - 1.0);
          const int col = ia * nr + ir;
          slice << ad << "," << rr << ","
                << (elems[col] >= 0
                        ? K_eta.Field().GetValue(elems[col], ips[col])
                        : 0.0)
                << "\n";
        }
      }
    }
  }
  // The kernel as mesh + .gf (gathered in parallel): GLVis slicing.
  {
#ifdef MFEM_USE_MPI
    std::ofstream mf(IsRoot() ? "visc_kernel.mesh" : "/dev/null");
    mf.precision(16);
    body.PrintAsOne(mf);
    std::ofstream gf(IsRoot() ? "visc_kernel.gf" : "/dev/null");
    gf.precision(16);
    static_cast<const ParGridFunction&>(K_eta.Field()).SaveAsOne(gf);
#else
    std::ofstream mf("visc_kernel.mesh");
    mf.precision(16);
    body.Print(mf);
    std::ofstream gf("visc_kernel.gf");
    gf.precision(16);
    K_eta.Field().Save(gf);
#endif
  }
#ifdef MFEM_USE_MPI
  if (world > 1 && IsRoot() && n_depths > 0) {
    // Parallel runs: rank 0 reloads the gathered kernel and samples
    // the shells and the slice serially, so a fine -np run yields the
    // same figure set as a serial one. The surface node set for the
    // shell CSVs comes from the gathered body mesh's surface cut.
    Mesh km("visc_kernel.mesh");
    std::ifstream gfs("visc_kernel.gf");
    GridFunction kg(&km, gfs);
    Array<int> sattr({km.bdr_attributes.Max()});
    auto ssm = SubMesh::CreateFromBoundary(km, sattr);
    H1_FECollection sf(order, 2);
    FiniteElementSpace ssp(&ssm, &sf);
    FiniteElementSpace svp(&ssm, &sf, 3);
    GridFunction xyz(&svp);
    VectorFunctionCoefficient idc(3,
                                  [](const Vector& x, Vector& v) { v = x; });
    xyz.ProjectCoefficient(idc);
    const int nd_s = ssp.GetVSize();
    for (int k = 0; k < n_depths; k++) {
      const double rk =
          n_depths == 1 ? 0.95
                        : 0.45 + (0.95 - 0.45) * k / (n_depths - 1.0);
      DenseMatrix pts(3, nd_s);
      for (int i = 0; i < nd_s; i++) {
        Vector dir(3);
        dir[0] = xyz[i];
        dir[1] = xyz[i + nd_s];
        dir[2] = xyz[i + 2 * nd_s];
        dir /= dir.Norml2();
        for (int c = 0; c < 3; c++) {
          pts(c, i) = rk * dir[c];
        }
      }
      Array<int> elems;
      Array<IntegrationPoint> ips;
      km.FindPoints(pts, elems, ips, false);
      std::ostringstream name;
      name << "visc_kernel_r" << std::fixed << std::setprecision(2) << rk
           << ".csv";
      std::ofstream out(name.str());
      out << "x,y,z,value\n";
      out.precision(16);
      for (int i = 0; i < nd_s; i++) {
        out << xyz[i] << "," << xyz[i + nd_s] << "," << xyz[i + 2 * nd_s]
            << ","
            << (elems[i] >= 0 ? kg.GetValue(elems[i], ips[i]) : 0.0)
            << "\n";
      }
      shell_names.push_back(name.str());
    }
    {
      Vector e1(n_obs), ez(3), e2(3);
      ez = 0.0;
      ez[2] = 1.0;
      const double z1 = ez * e1;
      e2 = ez;
      e2.Add(-z1, e1);
      e2 /= e2.Norml2();
      const int na = 181, nr = 48;
      DenseMatrix pts(3, na * nr);
      for (int ia = 0; ia < na; ia++) {
        const double a = -kPi + 2.0 * kPi * ia / (na - 1);
        for (int ir = 0; ir < nr; ir++) {
          const double rr = 0.05 + (0.995 - 0.05) * ir / (nr - 1.0);
          const int col = ia * nr + ir;
          for (int c = 0; c < 3; c++) {
            pts(c, col) =
                rr * (std::cos(a) * e1[c] + std::sin(a) * e2[c]);
          }
        }
      }
      Array<int> elems;
      Array<IntegrationPoint> ips;
      km.FindPoints(pts, elems, ips, false);
      std::ofstream slice("visc_kernel_slice.csv");
      slice << "alpha_deg,r,value\n";
      slice.precision(16);
      for (int ia = 0; ia < na; ia++) {
        const double ad = -180.0 + 360.0 * ia / (na - 1);
        for (int ir = 0; ir < nr; ir++) {
          const double rr = 0.05 + (0.995 - 0.05) * ir / (nr - 1.0);
          const int col = ia * nr + ir;
          slice << ad << "," << rr << ","
                << (elems[col] >= 0 ? kg.GetValue(elems[col], ips[col])
                                    : 0.0)
                << "\n";
        }
      }
    }
  }
#endif
  if (IsRoot()) {
    std::cout << "\nkernels of the present-day windowed sea-level "
                 "observation:\n  D_Idot epochs:";
    for (const auto& n : epoch_names) {
      std::cout << " " << n;
    }
    std::cout << "\n    <build>/earth_data/plot_fingerprint_kernel "
              << epoch_names.back() << " "
              << epoch_names[epoch_names.size() / 2]
              << " \\\n        --no-mask --global-only --side-by-side "
                 "--obs-label \"sensitivity of present-day\n        "
                 "sea level at the site to the ice-loading rate\" "
                 "--out ice_kernels.png\n";
    if (!shell_names.empty()) {
      std::cout << "  viscosity shells:";
      for (const auto& n : shell_names) {
        std::cout << " " << n;
      }
      std::cout << " (+ visc_kernel_slice.csv)\n"
                << "    <build>/earth_data/plot_fingerprint_kernel "
                << shell_names.front() << " " << shell_names.back()
                << " \\\n        --no-mask --global-only --side-by-side "
                   "--out visc_kernels.png\n";
    }
    std::cout << "  GLVis slicing: glvis -m visc_kernel.mesh -g "
                 "visc_kernel.gf\n";
  }

  // === The Taylor test ===
  if (taylor_points > 0) {
    const double D_v = K_eta.Directional(i_eta_dir);
    const double D_i = K_ice.Directional(i_ice_dir);
    auto run_J = [&](Coefficient& tau_run, Coefficient& sigma_run) {
      auto rh = IsotropicMaxwellRheology::Maxwell(dim, kappa, mu, tau_run);
      LinearQuasiStaticMixedSelfGravitatingProblem p(
          &fes_u, &fes_phi, rh, rho, kG, kDtNDegree);
      FunctionCoefficient w_run(
          [ocean, g0, rho_w](const Vector& x) {
            return rho_w * ocean(x) / g0;
          });
      p.SetWaterLoad(w_run, sigma_run, surface);
      p.SetRelTol(rel_tol);
      ViscoelasticOperator v(p);
      ExponentialTrapezoidSolver o;
      o.Init(v);
      Vector mm(v.Height());
      mm = 0.0;
      double tt = 0.0, dts = t_final / n_steps;
      if (!v.SolveElastic(mm, tt)) {
        return std::numeric_limits<double>::quiet_NaN();
      }
      for (int st = 0; st < n_steps; st++) {
        o.Step(mm, tt, dts);
      }
      if (!v.SolveElastic(mm, tt)) {
        return std::numeric_limits<double>::quiet_NaN();
      }
      return GlobalSum(*g_u * p.Displacement()) +
             GlobalSum(*g_phi * p.Potential()) +
             p.UniformPotentialTerm() / g0;
    };
    CsvTable tcurv("taylor_test.csv",
                   {"s", "J_visc", "J_visc_affine", "J_ice",
                    "J_ice_affine"});
    tcurv.Meta("title",
               "Taylor test: exact vs first-order affine prediction")
        .Meta("xlabel", "perturbation amplitude s")
        .Meta("y", "J_visc,J_visc_affine|J_ice,J_ice_affine")
        .Meta("ylabel", "J (viscosity blob)|J (ice-rate pattern)");
    CsvTable trem("taylor_remainder.csv", {"s", "R_visc", "R_ice",
                                           "ref_s2"});
    trem.Meta("title",
              "Taylor remainders |J - affine| (slope-2 reference)")
        .Meta("xlabel", "s")
        .Meta("logx", "true")
        .Meta("y", "R_visc,R_ice,ref_s2")
        .Meta("logy", "true");
    std::vector<std::array<double, 5>> rows;
    for (int j = -taylor_points; j <= taylor_points; j++) {
      const double sv = taylor_smax * j / taylor_points;
      double Jv = J0, Ji = J0;
      if (j != 0) {
        const double tml = tau_m;
        FunctionCoefficient tau_run([vblob, sv, tml](const Vector& x) {
          return tml * (1.0 + sv * vblob(x));
        });
        Jv = run_J(tau_run, sigma_d);
        FunctionCoefficient sig_run(
            [ocean, blob, icap, rho_i, melt_amp, tf, sv](const Vector& x,
                                                         double t) {
              const double dI =
                  (-melt_amp) * blob(x) * (t / tf) + sv * icap(x) * t;
              return rho_i * (1.0 - ocean(x)) * dI;
            });
        Ji = run_J(tau, sig_run);
      }
      rows.push_back({sv, Jv, J0 + sv * D_v, Ji, J0 + sv * D_i});
      if (IsRoot()) {
        std::cout << "  taylor s = " << sv << "  J_visc " << Jv
                  << " (affine " << J0 + sv * D_v << ")  J_ice " << Ji
                  << " (affine " << J0 + sv * D_i << ")\n";
      }
    }
    for (auto& r : rows) {
      tcurv.Row({r[0], r[1], r[2], r[3], r[4]});
    }
    const double Rv_max = std::abs(rows.back()[1] - rows.back()[2]);
    const double cref =
        Rv_max > 0.0 ? Rv_max / (taylor_smax * taylor_smax) : 1.0;
    for (auto& r : rows) {
      if (r[0] > 0.0) {
        trem.Row({r[0], std::abs(r[1] - r[2]), std::abs(r[3] - r[4]),
                  cref * r[0] * r[0]});
      }
    }
    if (IsRoot()) {
      std::cout << "  wrote taylor_test.csv and taylor_remainder.csv "
                   "(plot_csv.py)\n";
    }
  }

  if (visualization) {
    auto view = detail::MakeGridFunction(kfes.get());
    *view = K_eta.Field();
    GLVisWindow wk("viscosity kernel (volume)", DefaultKeys(dim));
    wk.Send(body, *view);
    GLVisWindow ws("adjoint-side sea-level kernel surface",
                   DefaultKeys(2));
    GridFunction last(&sea.SurfaceSpace());
    last = K_ice.IdotKernel(K_ice.NumTimes() - 1);
    ws.Send(sea.SurfaceMesh(), last);
  }
  return 0;
}
