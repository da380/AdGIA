// ============================================================================
// coseismic_fault.cpp
//
// Static coseismic deformation with the fault MESHED: the slip is a
// displacement discontinuity across an interior fault surface, imposed by
// LinearQuasiStaticClampedFaultProblem (the split-node load: the stiffness
// of the hanging-wall elements acting on the slip), on the faulted box of
// meshes/faulted_box.py — the box of coseismic_deformation.cpp with the
// fault plane cut through it, in 2-D or 3-D (the mesh decides). Linear
// elasticity, no gravity, the top a free surface and the other sides
// clamped. The slip has peak -s, tapered as cos^2 to zero at the fault
// edge (as in fault_box.hpp: along dip, and in 3-D along strike too);
// without the taper it is uniform and drops to zero over the element at
// the edge, where the displacement is single-valued. In 2-D it is pure
// dip-slip, the sign of -s the sense (thrust > 0); in 3-D the rake sets
// its direction in the fault plane. The elastic structure scales both
// moduli by the ParametricFactor of the box examples (fixed Poisson
// ratio).
//
// Three uses of the same problem:
//   - the forward solve, with the physical displacement (discontinuous
//     across the fault), the surface profile (3-D: along y = 0) and, on
//     the fault (3-D: along dip through its centre), the slip and the
//     traction change it causes, tau = sigma nu (nu the normal into the
//     hanging wall) from the fault reaction: shear component along the
//     slip direction (a stress drop is negative) and normal component;
//   - with -compare, the same slip as the stress glut of coseismic_
//     deformation (smoothed moment-tensor patches with the local shear
//     modulus) on the same mesh, the two surface profiles side by side and
//     their largest difference relative to the peak: two discretisations
//     of one source, which agree away from the fault and converge together
//     under refinement (meshes/faulted_box.py --scale);
//   - FaultToSurfaceMap, the linear map from fault slip to surface
//     displacement, and its adjoint: the observable J = mean uplift in a
//     Gaussian window (centre x = -sx, y = 0, width -sw) is computed
//     forward, J = <G s, w>, and by the adjoint, J = <s, G^T w>, where
//     G^T w is the traction the surface load w induces on the fault — the
//     sensitivity kernel of J to slip, whose shear part is written along
//     the fault; with -green k, the surface uplift of unit slip on each of
//     k equal sub-faults along dip (the columns of the Green's matrix of a
//     slip inversion), whose sum is checked against the response to
//     uniform slip.
//
// Output: the screen reports the moment, the surface peaks, the glut
// comparison, J both ways and the Green's-function linearity check. CSV
// tables (python3 plot_csv.py <file>): <csv>.csv, the surface displacement
// (with the glut's beside it); <csv>_fault.csv, along dip: the slip, the
// traction change and the kernel; <csv>_green.csv, the Green's functions.
// The traction and the kernel are their dual vectors divided by the
// lumped fault mass (the mass matrix's diagonal scaled to the fault's
// area, which stays positive at any order). With -vis (the default) GLVis
// shows the physical u_z, discontinuous at the fault, on the colour scale
// of the surface peaks, and the physical displacement vector (2-D: d
// displaces the mesh by it, which shows the offset).
//
// One source serves the serial and the parallel build; the differences are
// the partitioned mesh and submeshes, the preconditioner inside the
// problem class and the true-dof injection and mass matrices of the fault
// space, all in the #ifdef MFEM_USE_MPI blocks.
//
// Options (defaults in brackets):
//   -m       faulted-box mesh; its record is the .json of the same name
//            [../data/faulted_box_2d.msh].
//   -o       finite element order [2].
//   -s       peak slip; in 2-D > 0 thrust, < 0 normal [0.01].
//   -rake    rake in degrees, anticlockwise in the fault plane from the
//            strike (3-D; 90 thrust, 0 left-lateral strike-slip) [90].
//   -taper / -no-taper   cos^2 taper of the slip to zero at the fault
//            edge [on].
//   -beta, -alpha, -gamma, -bx, -bz, -br   the moduli factor of
//            coseismic_deformation (depth stiffening, lateral gradient,
//            Gaussian basin) [0, 0, 0, 0, 0, 1].
//   -compare / -no-compare   the stress-glut comparison [on].
//   -nd      glut patches along dip [12].
//   -ns      glut patches along strike (3-D) [12].
//   -smooth  glut Gaussian width in fault element sizes (0: point
//            sources) [1].
//   -sx      x of the centre of the observation window [-0.5].
//   -sw      width (sigma) of the observation window [0.25].
//   -green   number of sub-faults for the Green's functions (0: none) [5].
//   -vis / -no-vis   GLVis windows [on].
//   -csv     prefix of the CSV tables ("": none) [coseismic_fault].
//
// Sample runs (with mpiexec -np N in front in a parallel build):
//    ./coseismic_fault
//    ./coseismic_fault -s -0.01                          (normal fault)
//    ./coseismic_fault -no-taper                         (uniform slip)
//    ./coseismic_fault -beta 2 -gamma 0.5 -bz 1
//    ./coseismic_fault -sx 1 -green 8
//    ./coseismic_fault -m ../data/faulted_box_3d.msh -o 1
//    ./coseismic_fault -m ../data/faulted_box_3d.msh -o 1 -rake 0 -sw 0.5
//
// Mesh requirements: a faulted box of meshes/faulted_box.py, 2-D or 3-D,
// with its JSON record beside it: the footwall and hanging wall as domain
// attributes, the fault patch, its extension to the box boundary (welded:
// the fault edge is where they meet), the free surface and the two clamped
// sides as boundary attributes, and the fault frame. Nondimensional
// (mu0 = lambda0 = 1).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

#include "AdGIA.hpp"
#include "fault_box.hpp"
#include "visualisation.hpp"

using namespace std;
using namespace mfem;
using namespace AdGIA;

namespace {

#ifdef MFEM_USE_MPI
using MeshType = ParMesh;
using SubMeshType = ParSubMesh;
using SpaceType = ParFiniteElementSpace;
using FieldType = ParGridFunction;
using LFType = ParLinearForm;
using BFType = ParBilinearForm;
bool Root() { return Mpi::Root(); }
real_t Dot(const Vector& a, const Vector& b) {
  return InnerProduct(MPI_COMM_WORLD, a, b);
}
#else
using MeshType = Mesh;
using SubMeshType = SubMesh;
using SpaceType = FiniteElementSpace;
using FieldType = GridFunction;
using LFType = LinearForm;
using BFType = BilinearForm;
bool Root() { return true; }
real_t Dot(const Vector& a, const Vector& b) { return a * b; }
#endif

// The fault of the record, in the frame of fault_box.hpp (width, strike
// and its direction in 3-D only).
struct FaultRecord {
  int dim = 0;
  real_t W = 0.0, D = 0.0;
  real_t x = 0.0, depth = 0.0, dip = 0.0, strike = 0.0;
  real_t length = 0.0, width = 0.0;
  real_t fault_size = 0.0;
  Vector centre, normal, down_dip, strike_dir;
  int footwall = 0, hanging_wall = 0;
  int fault = 0, extension = 0, surface = 0, clamped_footwall = 0,
      clamped_hanging_wall = 0;
};

Vector JsonVector(const Json& j) {
  Vector v(static_cast<int>(j.array.size()));
  for (int i = 0; i < v.Size(); i++) {
    v(i) = j.array[i].number;
  }
  return v;
}

FaultRecord ReadRecord(const string& mesh_file) {
  const string path = mesh_file.substr(0, mesh_file.rfind('.')) + ".json";
  const Json rec = ReadJsonFile(path);
  auto find = [&](const Json& j, const char* key) -> const Json& {
    const Json* v = j.Find(key);
    MFEM_VERIFY(v, path << ": missing \"" << key << "\".");
    return *v;
  };
  auto num = [&](const Json& j, const char* key) {
    return find(j, key).number;
  };
  auto attr = [&](const Json& j, const char* key) {
    return static_cast<int>(num(j, key));
  };
  FaultRecord r;
  r.dim = attr(rec, "dimension");
  const Json& box = find(rec, "box");
  r.W = num(box, "width");
  r.D = num(box, "depth");
  const Json& fault = find(rec, "fault");
  r.x = num(fault, "x");
  r.depth = num(fault, "depth");
  r.dip = num(fault, "dip");
  r.length = num(fault, "length");
  r.centre = JsonVector(find(fault, "centre"));
  r.normal = JsonVector(find(fault, "normal"));
  r.down_dip = JsonVector(find(fault, "down_dip"));
  if (r.dim == 3) {
    r.strike = num(fault, "strike");
    r.width = num(fault, "width");
    r.strike_dir = JsonVector(find(fault, "strike_direction"));
  }
  r.fault_size = num(find(rec, "sizes"), "fault");
  const Json& attrs = find(rec, "attributes");
  const Json& dom = find(attrs, "domain");
  const Json& bdr = find(attrs, "boundary");
  r.footwall = attr(dom, "footwall");
  r.hanging_wall = attr(dom, "hanging_wall");
  r.fault = attr(bdr, "fault");
  r.extension = attr(bdr, "fault_extension");
  r.surface = attr(bdr, "surface");
  r.clamped_footwall = attr(bdr, "clamped_footwall");
  r.clamped_hanging_wall = attr(bdr, "clamped_hanging_wall");
  return r;
}

Array<int> Marker(int size, std::initializer_list<int> attrs) {
  Array<int> m(size);
  m = 0;
  for (int a : attrs) {
    m[a - 1] = 1;
  }
  return m;
}

// The fault coordinates (xi, eta) in [-1/2, 1/2]^2 over the patch: along
// dip and (3-D) along strike, in units of its length and width.
std::pair<real_t, real_t> FaultCoordinates(const FaultRecord& f,
                                           const Vector& x) {
  real_t xi = 0.0, eta = 0.0;
  for (int i = 0; i < x.Size(); i++) {
    xi += (x(i) - f.centre(i)) * f.down_dip(i);
    if (f.dim == 3) {
      eta += (x(i) - f.centre(i)) * f.strike_dir(i);
    }
  }
  return {xi / f.length, f.dim == 3 ? eta / f.width : 0.0};
}

// The slip: amplitude(xi, eta) times the unit slip direction.
VectorFunctionCoefficient Slip(
    const FaultRecord& f, const Vector& direction,
    std::function<real_t(real_t, real_t)> amplitude) {
  return VectorFunctionCoefficient(
      f.dim, [f, direction, amplitude](const Vector& x, Vector& s) {
        const auto [xi, eta] = FaultCoordinates(f, x);
        s = direction;
        s *= amplitude(xi, eta);
      });
}

// The true-dof dual vector of an assembled linear form.
Vector TrueDual(LinearForm& lf) {
  const FiniteElementSpace& fes = *lf.FESpace();
  Vector t(fes.GetTrueVSize());
  if (const Operator* P = fes.GetProlongationMatrix()) {
    P->MultTranspose(lf, t);
  } else {
    t = lf;
  }
  return t;
}

// The lumped (true-dof) mass of a mass form: the diagonal of the matrix,
// scaled so that it sums to the row sums' total (the measure). Unlike the
// row sums it is positive at any order (the vertex functions of quadratic
// triangles integrate to zero).
Vector LumpedMass(BFType& m) {
  m.Assemble();
  m.Finalize();
  Vector diag, rows;
#ifdef MFEM_USE_MPI
  std::unique_ptr<HypreParMatrix> M(m.ParallelAssemble());
  M->GetDiag(diag);
  Vector ones(M->Width());
  ones = 1.0;
  rows.SetSize(M->Height());
  M->Mult(ones, rows);
#else
  m.SpMat().GetDiag(diag);
  Vector ones(m.SpMat().Width());
  ones = 1.0;
  rows.SetSize(m.SpMat().Height());
  m.SpMat().Mult(ones, rows);
#endif
  Vector unit(diag.Size());
  unit = 1.0;
  diag *= Dot(rows, unit) / Dot(diag, unit);
  return diag;
}

// Vector values of fields at fixed points (FindPoints once, collective;
// each point on exactly one rank, the samples reduced everywhere).
class PointSampler {
 public:
  PointSampler(Mesh& mesh, const DenseMatrix& points)
      : dim_(points.Height()), n_(points.Width()), points_(points) {
    mesh.FindPoints(points_, elem_, ips_);
  }

  int NumPoints() const { return n_; }

  // Component-major: component c of point i at c * NumPoints() + i.
  vector<double> Sample(const GridFunction& u) const {
    vector<double> values(static_cast<size_t>(dim_) * n_, 0.0);
    Vector val(dim_);
    for (int i = 0; i < n_; i++) {
      if (elem_[i] >= 0) {
        u.GetVectorValue(elem_[i], ips_[i], val);
        for (int c = 0; c < dim_; c++) {
          values[c * n_ + i] = val(c);
        }
      }
    }
#ifdef MFEM_USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, values.data(), values.size(), MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
#endif
    return values;
  }

 private:
  int dim_, n_;
  DenseMatrix points_;
  Array<int> elem_;
  Array<IntegrationPoint> ips_;
};

// A field on the parent space from a true-dof vector of the fault space.
void FaultToParent(const Operator& injection, const Vector& fault_true,
                   FieldType& parent) {
  Vector parent_true(parent.FESpace()->GetTrueVSize());
  injection.Mult(fault_true, parent_true);
  parent.SetFromTrueDofs(parent_true);
}

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif
  constexpr real_t pi = std::numbers::pi_v<real_t>;
  constexpr real_t deg = pi / 180.0;

  // Set the default options.
  const char* mesh_file = "../data/faulted_box_2d.msh";
  int order = 2;
  real_t slip = 0.01, rake = 90.0;
  bool taper = true;
  examples::ParametricFactor moduli;
  bool compare = true;
  int nd = 12, ns = 12;
  real_t smooth = 1.0;
  real_t station_x = -0.5, station_w = 0.25;
  int n_green = 5;
  bool visualization = true;
  const char* csv_prefix = "coseismic_fault";

  // Read in command line options and process.
  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Faulted-box mesh file.");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&slip, "-s", "--slip",
                 "Peak slip; in 2-D > 0 thrust, < 0 normal faulting.");
  args.AddOption(&rake, "-rake", "--rake",
                 "Rake in degrees (3-D; 90 thrust, 0 left-lateral "
                 "strike-slip).");
  args.AddOption(&taper, "-taper", "--taper-slip", "-no-taper",
                 "--no-taper-slip",
                 "Taper the slip as cos^2 to zero at the fault edge.");
  args.AddOption(&moduli.beta, "-beta", "--depth-stiffening",
                 "Moduli grow by 1 + beta |z| / D (0: homogeneous).");
  args.AddOption(&moduli.alpha, "-alpha", "--lateral-gradient",
                 "Moduli grow by 1 + alpha x / W (|alpha| < 2).");
  args.AddOption(&moduli.gamma, "-gamma", "--basin-amplitude",
                 "Relative moduli reduction at the basin centre (< 1).");
  args.AddOption(&moduli.cx, "-bx", "--basin-x", "x of the basin centre.");
  args.AddOption(&moduli.cz, "-bz", "--basin-depth",
                 "Depth of the basin centre below z = 0.");
  args.AddOption(&moduli.radius, "-br", "--basin-radius",
                 "Gaussian radius of the basin.");
  args.AddOption(&compare, "-compare", "--compare", "-no-compare",
                 "--no-compare",
                 "Solve the same slip as a stress glut on the same mesh.");
  args.AddOption(&nd, "-nd", "--num-dip-patches", "Glut patches along dip.");
  args.AddOption(&ns, "-ns", "--num-strike-patches",
                 "Glut patches along strike (3-D).");
  args.AddOption(&smooth, "-smooth", "--source-smoothing",
                 "Glut Gaussian width in fault element sizes (0: point "
                 "sources).");
  args.AddOption(&station_x, "-sx", "--station-x",
                 "x of the centre of the surface observation window.");
  args.AddOption(&station_w, "-sw", "--station-width",
                 "Width (sigma) of the surface observation window.");
  args.AddOption(&n_green, "-green", "--green-subfaults",
                 "Sub-faults for the Green's functions (0: none).");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization", "Show the displacement in GLVis.");
  args.AddOption(&csv_prefix, "-csv", "--csv",
                 "Prefix of the CSV tables (\"\": none).");
  args.Parse();
  if (!args.Good()) {
    if (Root()) {
      args.PrintUsage(cout);
    }
    return 1;
  }
  if (Root()) {
    args.PrintOptions(cout);
  }

  const FaultRecord f = ReadRecord(mesh_file);
  moduli.Verify(f.D, 0.0, "modulus");
  Mesh smesh(mesh_file, 1, 1);
  const int dim = smesh.Dimension();
  MFEM_VERIFY(dim == f.dim, "The mesh and its record differ in dimension.");
#ifdef MFEM_USE_MPI
  MeshType mesh(MPI_COMM_WORLD, smesh);
  smesh.Clear();
#else
  MeshType& mesh = smesh;
#endif
  const int na = mesh.attributes.Max(), nb = mesh.bdr_attributes.Max();

  H1_FECollection fec(order, dim);
  SpaceType fes(&mesh, &fec, dim);
#ifdef MFEM_USE_MPI
  const auto n_u = fes.GlobalTrueVSize();
#else
  const auto n_u = fes.GetTrueVSize();
#endif

  // Material: both moduli scaled by one factor (fixed Poisson ratio).
  const real_t lambda0 = 1.0, mu0 = 1.0;
  auto factor = moduli.Make(dim, f.W, f.D);
  FunctionCoefficient kappa_c(
      [=](const Vector& x) { return (lambda0 + 2.0 * mu0 / dim) * factor(x); });
  FunctionCoefficient mu_c([=](const Vector& x) { return mu0 * factor(x); });
  IsotropicElasticRheology rheology(dim, kappa_c, mu_c);

  // The problem: every artificial side clamped, no surface traction, the
  // slip carried by the fault patch and pinned to zero where the patch
  // meets its extension.
  const Array<int> ess =
      Marker(nb, {f.clamped_footwall, f.clamped_hanging_wall});
  const Array<int> no_traction = Marker(nb, {});
  Vector zero(dim);
  zero = 0.0;
  VectorConstantCoefficient zero_c(zero);
  LinearQuasiStaticClampedFaultProblem problem(
      &fes, rheology, ess, zero_c, no_traction, Marker(na, {f.hanging_wall}),
      Marker(nb, {f.fault}), Marker(nb, {f.extension}));

  // The unit slip direction of the hanging wall: up-dip in 2-D (thrust for
  // slip > 0), the rake in the fault plane in 3-D; and its sense, the
  // direction of the slip itself, for the profiles.
  Vector direction(f.down_dip);
  direction.Neg();
  if (dim == 3) {
    add(std::cos(rake * deg), f.strike_dir, -std::sin(rake * deg), f.down_dip,
        direction);
  }
  Vector slip_dir(direction);
  if (slip < 0.0) {
    slip_dir.Neg();
  }
  auto window1 = [=](real_t t) -> real_t {
    if (std::abs(t) >= 0.5) {
      return 0.0;
    }
    return taper ? std::pow(std::cos(pi * t), 2) : 1.0;
  };
  auto amplitude = [=](real_t xi, real_t eta) {
    return slip * window1(xi) * (dim == 3 ? window1(eta) : 1.0);
  };
  auto slip_c = Slip(f, direction, amplitude);
  problem.SetSlip(slip_c);
  problem.AssembleForce(0.0);
  const bool ok = problem.Solve();
  if (Root()) {
    cout << "Displacement unknowns: " << n_u << ", solve "
         << (ok ? "converged" : "DID NOT CONVERGE") << "\n";
  }

  // The physical displacement, discontinuous across the fault, on a
  // discontinuous space with the nodes of the continuous one.
  L2_FECollection dg(order, dim, BasisType::GaussLobatto);
  SpaceType dg_fes(&mesh, &dg, dim);
  FieldType u_dg(&dg_fes);
  u_dg.ProjectCoefficient(problem.PhysicalDisplacement());

  // The scalar moment, the integral of mu |s| over the fault with the
  // local shear modulus: the first component of the dual of mu |s| e_0
  // paired with the field of ones.
  Vector unit(dim);
  unit = 1.0;
  VectorConstantCoefficient unit_c(unit);
  Array<int> fault_marker = Marker(nb, {f.fault});
  {
    FunctionCoefficient mu_slip_c([&](const Vector& x) {
      const auto [xi, eta] = FaultCoordinates(f, x);
      return mu0 * factor(x) * std::abs(amplitude(xi, eta));
    });
    Vector e0(dim);
    e0 = 0.0;
    e0(0) = 1.0;
    VectorConstantCoefficient e0_c(e0);
    ScalarVectorProductCoefficient m_c(mu_slip_c, e0_c);
    LFType moment_form(&fes);
    moment_form.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(m_c),
                                      fault_marker);
    moment_form.Assemble();
    FieldType ones(&fes);
    ones.ProjectCoefficient(unit_c);
    Vector ones_true(fes.GetTrueVSize());
    ones.GetTrueDofs(ones_true);
    const real_t moment = Dot(TrueDual(moment_form), ones_true);
    if (Root()) {
      cout << "Scalar moment: M0 = " << moment
           << (dim == 2 ? " (per unit strike length)" : "") << "\n";
    }
  }

  // The surface: the profile along it and the peaks of u_z over its dofs.
  examples::FaultBox box;
  box.W = f.W;
  box.D = f.D;
  box.nx = std::max(1, static_cast<int>(std::lround(f.W / f.fault_size)));
  box.nz = std::max(1, static_cast<int>(std::lround(f.D / f.fault_size)));
  examples::SurfaceProfile profile(mesh, box, dim);
  const int npts = profile.NumPoints();
  const auto surface_u = profile.Sample(u_dg);

  SpaceType fes_z(&mesh, &fec);
  FieldType u_z(&fes_z);
  VectorGridFunctionCoefficient u_w_c(&problem.Displacement());
  Vector e_z(dim);
  e_z = 0.0;
  e_z(dim - 1) = 1.0;
  VectorConstantCoefficient e_z_c(e_z);
  InnerProductCoefficient u_z_c(u_w_c, e_z_c);
  u_z.ProjectCoefficient(u_z_c);
  const auto [uz_min, uz_max] = examples::SurfacePeaks(u_z, fes_z, f.surface);
  if (Root()) {
    cout << "Peak surface uplift:     " << uz_max
         << "\nPeak surface subsidence: " << uz_min << "\n";
  }

  // On the fault: the slip and the traction change, tau = -r / m from the
  // fault reaction r and the lumped fault mass m, sampled along dip (3-D:
  // through the centre of the patch).
  BFType fault_mass_form(&fes);
  fault_mass_form.AddBoundaryIntegrator(new VectorMassIntegrator(),
                                        fault_marker);
  const Vector m_fault = LumpedMass(fault_mass_form);
  Vector r_true;
  problem.FaultReaction(r_true);
  Vector tau_true(fes.GetTrueVSize());
  tau_true = 0.0;
  for (int i : problem.SlipTrueDofs()) {
    tau_true(i) = -r_true(i) / m_fault(i);
  }
  FieldType tau(&fes);
  tau.SetFromTrueDofs(tau_true);
  const int nfault = 101;
  DenseMatrix fault_pts(dim, nfault);
  vector<double> xi_pts(nfault);
  for (int j = 0; j < nfault; j++) {
    xi_pts[j] = (j + 0.5) / nfault - 0.5;
    for (int c = 0; c < dim; c++) {
      fault_pts(c, j) = f.centre(c) + xi_pts[j] * f.length * f.down_dip(c);
    }
  }
  PointSampler on_fault(mesh, fault_pts);
  const auto slip_s = on_fault.Sample(problem.Slip());
  const auto tau_s = on_fault.Sample(tau);
  auto along = [&](const vector<double>& v, int j, const Vector& d) {
    double a = 0.0;
    for (int c = 0; c < dim; c++) {
      a += v[c * nfault + j] * d(c);
    }
    return a;
  };

  // The same slip as a stress glut on the same mesh (the patches of
  // fault_box.hpp, centred on the fault, with the local shear modulus).
  vector<double> glut_u;
  if (compare) {
    examples::Fault glut;
    glut.dip = f.dip;
    glut.strike = f.strike;
    glut.rake = rake;
    glut.depth = f.depth;
    glut.x = f.x;
    glut.length = f.length;
    glut.width = f.width;
    glut.slip = slip;
    glut.nd = nd;
    glut.ns = ns;
    glut.smooth = smooth;
    glut.taper = taper;
    auto sources = glut.Build(
        dim, box, [&](const Vector& xp) { return mu0 * factor(xp); });
    LinearQuasiStaticClampedProblem glut_problem(&fes, rheology, ess, zero_c,
                                                 no_traction);
    if (sources.smoothed) {
      glut_problem.ExternalLoad().AddDomainIntegrator(
          new DomainLFDeformationGradientIntegrator(
              *sources.smoothed,
              &IntRules.Get(mesh.GetTypicalElementGeometry(), 2 * order + 6)));
    } else {
      for (auto& src : sources.patches) {
        glut_problem.ExternalLoad().AddDomainIntegrator(
            new DomainLFDeformationGradientIntegrator(*src));
      }
    }
    glut_problem.AssembleForce(0.0);
    glut_problem.Solve();
    glut_u = profile.Sample(glut_problem.Displacement());
    double peak = 0.0, diff = 0.0;
    for (size_t i = 0; i < glut_u.size(); i++) {
      peak = std::max(peak, std::abs(surface_u[i]));
      diff = std::max(diff, std::abs(surface_u[i] - glut_u[i]));
    }
    if (Root()) {
      cout << "Stress glut (" << nd * (dim == 3 ? ns : 1) << " patches"
           << (sources.smoothed ? ", sigma = " + to_string(sources.sigma)
                                : string(", point sources"))
           << ", M0 = " << sources.moment
           << "): largest profile difference / peak = " << diff / peak << "\n";
    }
  }

  // The map from fault slip to surface displacement and its adjoint.
  Array<int> fault_attr({f.fault}), surface_attr({f.surface});
  auto fault_mesh = SubMeshType::CreateFromBoundary(mesh, fault_attr);
  auto surface_mesh = SubMeshType::CreateFromBoundary(mesh, surface_attr);
  auto fault_fes = SubMeshDofInjection::MakeShadowSpace(fes, fault_mesh);
  auto surface_fes = SubMeshDofInjection::MakeShadowSpace(fes, surface_mesh);
  FaultToSurfaceMap G(problem, *fault_fes, *surface_fes);
  SubMeshDofInjection fault_injection(*fault_fes, fes);
#ifdef MFEM_USE_MPI
  std::unique_ptr<Operator> pi_fault = fault_injection.NewTrueDofMatrix();
#else
  std::unique_ptr<Operator> pi_fault = fault_injection.NewSparseMatrix();
#endif

  // The observable: the mean uplift in a Gaussian window, J = <u, w> /
  // integral of the window, w the surface dual of the window weight.
  FunctionCoefficient window_c([=](const Vector& x) {
    real_t r2 = (x(0) - station_x) * (x(0) - station_x);
    if (dim == 3) {
      r2 += x(1) * x(1);
    }
    return std::exp(-0.5 * r2 / (station_w * station_w));
  });
  ScalarVectorProductCoefficient weight_c(window_c, e_z_c);
  LFType w_form(surface_fes.get());
  w_form.AddDomainIntegrator(new VectorDomainLFIntegrator(weight_c));
  w_form.Assemble();
  const Vector w = TrueDual(w_form);
  FieldType e_z_surface(surface_fes.get());
  e_z_surface.ProjectCoefficient(e_z_c);
  Vector e_z_true(surface_fes->GetTrueVSize());
  e_z_surface.GetTrueDofs(e_z_true);
  const real_t window = Dot(w, e_z_true);

  FieldType slip_f(fault_fes.get());
  slip_f.ProjectCoefficient(slip_c);
  Vector s_true(fault_fes->GetTrueVSize());
  slip_f.GetTrueDofs(s_true);
  Vector Gs, Gtw;
  G.Mult(s_true, Gs);
  const real_t J_forward = Dot(Gs, w) / window;
  G.MultTranspose(w, Gtw);
  const real_t J_adjoint = Dot(s_true, Gtw) / window;
  if (Root()) {
    cout << "Mean uplift in the window at x = " << station_x
         << ": J = " << J_forward << " (forward), " << J_adjoint
         << " (adjoint); relative difference "
         << std::abs(J_forward - J_adjoint) /
                std::max(std::abs(J_forward), real_t{1e-300})
         << "\n";
  }

  // The sensitivity kernel of J: G^T w / (window integral) per unit fault
  // area (the lumped fault-space mass), on the parent space for sampling.
  BFType fault_space_mass(fault_fes.get());
  fault_space_mass.AddDomainIntegrator(new VectorMassIntegrator());
  const Vector m_f = LumpedMass(fault_space_mass);
  Vector kernel_true(Gtw.Size());
  for (int i = 0; i < Gtw.Size(); i++) {
    kernel_true(i) = Gtw(i) / (window * m_f(i));
  }
  FieldType kernel(&fes);
  FaultToParent(*pi_fault, kernel_true, kernel);
  const auto kernel_s = on_fault.Sample(kernel);

  // Green's functions: the surface uplift of unit slip (in the slip
  // direction) on each of n_green equal sub-faults along dip, and their
  // sum against the response to unit slip on the whole fault.
  vector<vector<double>> green;
  if (n_green > 0) {
    const real_t sense = slip < 0.0 ? -1.0 : 1.0;
    auto unit_slip = [&](int k) {
      return Slip(f, direction, [=](real_t xi, real_t eta) -> real_t {
        if (std::abs(xi) >= 0.5 || std::abs(eta) >= 0.5) {
          return 0.0;
        }
        const int patch = std::min(
            n_green - 1, static_cast<int>(std::floor((xi + 0.5) * n_green)));
        return (k < 0 || patch == k) ? sense : 0.0;
      });
    };
    vector<double> sum(npts, 0.0);
    for (int k = -1; k < n_green; k++) {
      auto c = unit_slip(k);
      slip_f.ProjectCoefficient(c);
      slip_f.GetTrueDofs(s_true);
      G.Mult(s_true, Gs);
      const auto u = profile.Sample(problem.Displacement());
      vector<double> col(u.begin() + (dim - 1) * npts, u.begin() + dim * npts);
      if (k >= 0) {
        for (int i = 0; i < npts; i++) {
          sum[i] += col[i];
        }
      }
      green.push_back(std::move(col));
    }
    double peak = 0.0, diff = 0.0;
    for (int i = 0; i < npts; i++) {
      peak = std::max(peak, std::abs(green[0][i]));
      diff = std::max(diff, std::abs(green[0][i] - sum[i]));
    }
    if (Root()) {
      cout << "Green's functions: " << n_green
           << " sub-faults; their sum against the whole fault: relative "
              "difference "
           << diff / peak << "\n";
    }
  }

  // The tables.
  const string prefix = csv_prefix;
  if (!prefix.empty()) {
    vector<string> comps = dim == 2 ? vector<string>{"u_x", "u_z"}
                                    : vector<string>{"u_x", "u_y", "u_z"};
    vector<string> cols = {"x"};
    string y_spec, y_labels;
    for (const auto& c : comps) {
      cols.push_back(c);
      y_spec += (y_spec.empty() ? "" : "|") + c;
      y_labels += (y_labels.empty() ? "" : "|") + c;
      if (compare) {
        y_spec += "," + c + "_glut";
      }
    }
    if (compare) {
      for (const auto& c : comps) {
        cols.push_back(c + "_glut");
      }
    }
    examples::CsvTable table(prefix + ".csv", cols);
    table.Meta("title", "Coseismic surface displacement, meshed fault")
        .Meta("note", string(dim == 2 ? "along the surface"
                                      : "along the surface at y = 0") +
                          (compare ? "; glut: the same slip as a stress "
                                     "glut on the same mesh"
                                   : ""))
        .Meta("xlabel", "x")
        .Meta("ylabel", y_labels)
        .Meta("y", y_spec);
    for (int i = 0; i < npts; i++) {
      vector<double> row{profile.X(i)};
      for (int c = 0; c < dim; c++) {
        row.push_back(surface_u[c * npts + i]);
      }
      if (compare) {
        for (int c = 0; c < dim; c++) {
          row.push_back(glut_u[c * npts + i]);
        }
      }
      table.Row(row);
    }
    table.Write();

    examples::CsvTable fault_table(prefix + "_fault.csv",
                                   {"xi", "slip", "tau_shear", "tau_normal",
                                    "kernel_shear", "kernel_normal"});
    fault_table.Meta("title", "Along dip: slip, traction change, kernel")
        .Meta("note",
              "shear: along the slip; normal: into the hanging wall; "
              "kernel: dJ/ds per unit area")
        .Meta("xlabel", "xi")
        .Meta("y", "slip|tau_shear,tau_normal|kernel_shear,kernel_normal")
        .Meta("ylabel", "slip|traction change|dJ/ds");
    for (int j = 0; j < nfault; j++) {
      fault_table.Row({xi_pts[j], along(slip_s, j, slip_dir),
                       along(tau_s, j, slip_dir), along(tau_s, j, f.normal),
                       along(kernel_s, j, slip_dir),
                       along(kernel_s, j, f.normal)});
    }
    fault_table.Write();

    if (n_green > 0) {
      vector<string> gcols = {"x", "whole_fault"};
      for (int k = 0; k < n_green; k++) {
        gcols.push_back("subfault_" + to_string(k + 1));
      }
      examples::CsvTable green_table(prefix + "_green.csv", gcols);
      green_table.Meta("title", "Green's functions: uplift per unit slip")
          .Meta("note", "sub-faults numbered up-dip to down-dip")
          .Meta("xlabel", "x")
          .Meta("ylabel", "u_z");
      for (int i = 0; i < npts; i++) {
        vector<double> row{profile.X(i)};
        for (const auto& col : green) {
          row.push_back(col[i]);
        }
        green_table.Row(row);
      }
      green_table.Write();
    }
  }

  // The physical u_z on the colour scale of the surface peaks (the field
  // jumps at the fault), and the physical displacement vector.
  if (visualization) {
    L2_FECollection dg_scalar(order, dim, BasisType::GaussLobatto);
    SpaceType dg_z_fes(&mesh, &dg_scalar);
    FieldType u_dg_z(&dg_z_fes);
    VectorGridFunctionCoefficient u_dg_c(&u_dg);
    InnerProductCoefficient u_dg_z_c(u_dg_c, e_z_c);
    u_dg_z.ProjectCoefficient(u_dg_z_c);
    examples::GLVisWindow window_z("u_z (meshed fault)",
                                   examples::DefaultKeys(dim));
    const real_t span =
        std::max({std::abs(uz_min), std::abs(uz_max), real_t{1e-30}});
    window_z.SetValueRange(-1.2 * span, 1.2 * span);
    window_z.Send(mesh, u_dg_z);
    examples::GLVisWindow vec_window("displacement (meshed fault)",
                                     dim == 2 ? "Rjlcvvvvv" : "RRRilcvvvvv");
    vec_window.Send(mesh, u_dg);
  }
  return ok && G.Converged() ? 0 : 1;
}
