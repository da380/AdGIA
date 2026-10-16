// ============================================================================
// coseismic_fault.cpp
//
// Static coseismic deformation with the fault MESHED: the slip is a
// displacement discontinuity across an interior fault surface, imposed by
// LinearQuasiStaticClampedFaultProblem (the split-node load: the stiffness
// of the hanging-wall elements acting on the slip), on the faulted box of
// meshes/faulted_box.py — the box of coseismic_deformation.cpp with the
// fault plane cut through it, in 2-D or 3-D (the mesh decides), under the
// script's mountain-range topography over the hanging wall (the _flat
// meshes: a flat surface, as in coseismic_deformation). Linear
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
//     across the fault), the surface profile along the topography (3-D:
//     along y = 0) and, on
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
//     G^T w is the traction the surface load w induces on the fault, in
//     consistent nodal form: a dual vector, whose part tangential to the
//     fault, mapped to a field by a Riesz map, is the gradient of J with
//     respect to (tangential) slip; with -green k, the surface uplift of
//     unit slip on each of
//     k equal sub-faults along dip (the columns of the Green's matrix of a
//     slip inversion), whose sum is checked against the response to
//     uniform slip.
//
// Output: the screen reports the moment, the surface peaks, the glut
// comparison, J both ways and the Green's-function linearity check. CSV
// tables (python3 plot_csv.py <file>): <csv>.csv, the surface displacement
// (with the glut's beside it) and the surface height; <csv>_fault.csv,
// along dip: the slip, the
// traction change and the kernel; <csv>_green.csv, the Green's functions.
// The traction and the kernel come out of the solves as dual vectors (the
// fault reaction and G^T w) and are mapped to fields by Riesz maps of the
// fault space (riesz.hpp), the slip pinned to zero on the fault edge (the
// boundary of the fault SubMesh). The traction, a physical density, by the
// L2 map: the field is the L2 projection of tau onto the slip space (the
// reaction is assembled from the stress of the element layer on the
// hanging-wall side, so on an unstructured mesh the projection carries
// mesh-scale noise; a Sobolev map would smooth it, but would also damp its
// amplitude and, through the pinned edge, the concentrations at the tips).
// The kernel — its dual first projected onto the fault plane, the slip
// being tangential (on a planar fault the projection commutes with the
// component-wise metric) — by the Sobolev map of smoothing length -sl and
// order -so: the gradient of J in that metric (-sl 0: in L2). With -vis (the
// default) GLVis shows the physical u_z, discontinuous at the fault, on the
// colour scale of the surface peaks, and the physical displacement vector (2-D:
// d displaces the mesh by it, which shows the offset). In 3-D both are cut open
// along the dip section through the fault centre and seen along the strike, so
// the fault shows its dip; and three vector fields are shown on the fault
// itself, its boundary SubMesh drawn as a surface seen face on from the
// hanging wall: the slip, the shear (tangential) traction change and the
// kernel, each with black arrows, coloured by magnitude (F cycles the colour
// through the x, y and z components).
//
// One source serves the serial and the parallel build; the differences are
// the partitioned mesh and submeshes, the preconditioner inside the
// problem class and the true-dof injection of the fault space, all in the
// #ifdef MFEM_USE_MPI blocks.
//
// Options (defaults in brackets):
//   -m       faulted-box mesh; its record is the .json of the same name
//            [../data/faulted_box_2d.msh].
//   -o       finite element order [2].
//   -rs      uniform refinements of the mesh before it is partitioned [0].
//   -rp      uniform refinements of the partitioned mesh; a serial build
//            adds them to -rs [0]. The canned meshes have straight-sided
//            elements, so refinement halves the elements but keeps the
//            faceted surface: for a finer topography, regenerate the mesh
//            (meshes/faulted_box.py --scale).
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
//   -sl      Sobolev smoothing length of the kernel's Riesz map (0: L2)
//            [0.1].
//   -so      Sobolev order s of the kernel's Riesz map [2].
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
//    ./coseismic_fault -m ../data/faulted_box_2d_flat.msh     (flat surface)
//    ./coseismic_fault -rs 1                             (refined once)
//
// Mesh requirements: a faulted box of meshes/faulted_box.py, 2-D or 3-D,
// with its JSON record beside it: the footwall and hanging wall as domain
// attributes, the fault patch, its extension to the box boundary (welded:
// the fault edge is where they meet), the free surface and the two clamped
// sides as boundary attributes, the fault frame, and the topography's
// profile along y = 0 with its largest curvature (the surface samples sit
// just inside the faceted mesh surface). Nondimensional (mu0 = lambda0 =
// 1).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <numbers>
#include <utility>
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
  real_t fault_size = 0.0, surface_size = 0.0;
  real_t z_max = 0.0, max_curvature = 0.0;
  vector<real_t> profile_x, profile_z;
  Vector centre, normal, down_dip, strike_dir;
  int footwall = 0, hanging_wall = 0;
  int fault = 0, extension = 0, surface = 0, clamped_footwall = 0,
      clamped_hanging_wall = 0;
};

Vector JsonVector(const Json& j) {
  auto v = Vector(static_cast<int>(j.array.size()));
  for (int i = 0; i < v.Size(); i++) {
    v(i) = j.array[i].number;
  }
  return v;
}

FaultRecord ReadRecord(const string& mesh_file) {
  const auto path = mesh_file.substr(0, mesh_file.rfind('.')) + ".json";
  const auto rec = ReadJsonFile(path);
  auto find = [&](const Json& j, const char* key) -> const Json& {
    const auto* v = j.Find(key);
    MFEM_VERIFY(v, path << ": missing \"" << key << "\".");
    return *v;
  };
  auto num = [&](const Json& j, const char* key) {
    return find(j, key).number;
  };
  auto attr = [&](const Json& j, const char* key) {
    return static_cast<int>(num(j, key));
  };
  auto r = FaultRecord();
  r.dim = attr(rec, "dimension");
  const auto& box = find(rec, "box");
  r.W = num(box, "width");
  r.D = num(box, "depth");
  const auto& fault = find(rec, "fault");
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
  const auto& sizes = find(rec, "sizes");
  r.fault_size = num(sizes, "fault");
  r.surface_size = num(sizes, "surface");
  const auto& topography = find(rec, "topography");
  r.z_max = num(topography, "z_max");
  r.max_curvature = num(topography, "max_curvature");
  const auto& profile = find(topography, "profile");
  for (const auto& v : find(profile, "x").array) {
    r.profile_x.push_back(v.number);
  }
  for (const auto& v : find(profile, "z").array) {
    r.profile_z.push_back(v.number);
  }
  MFEM_VERIFY(
      r.profile_x.size() == r.profile_z.size() && r.profile_x.size() > 1,
      path << ": the topography profile is malformed.");
  const auto& attrs = find(rec, "attributes");
  const auto& dom = find(attrs, "domain");
  const auto& bdr = find(attrs, "boundary");
  r.footwall = attr(dom, "footwall");
  r.hanging_wall = attr(dom, "hanging_wall");
  r.fault = attr(bdr, "fault");
  r.extension = attr(bdr, "fault_extension");
  r.surface = attr(bdr, "surface");
  r.clamped_footwall = attr(bdr, "clamped_footwall");
  r.clamped_hanging_wall = attr(bdr, "clamped_hanging_wall");
  return r;
}

// The height of the free surface along y = 0 (2-D: everywhere), linearly
// interpolated in the record's profile.
real_t SurfaceHeight(const FaultRecord& f, real_t x) {
  const auto& px = f.profile_x;
  const auto upper = std::upper_bound(px.begin(), px.end(), x);
  if (upper == px.begin()) {
    return f.profile_z.front();
  }
  if (upper == px.end()) {
    return f.profile_z.back();
  }
  const auto i = static_cast<std::size_t>(upper - px.begin());
  const auto t = (x - px[i - 1]) / (px[i] - px[i - 1]);
  return (1.0 - t) * f.profile_z[i - 1] + t * f.profile_z[i];
}

Array<int> Marker(int size, std::initializer_list<int> attrs) {
  auto m = Array<int>(size);
  m = 0;
  for (auto a : attrs) {
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
  const auto& fes = *lf.FESpace();
  auto t = Vector(fes.GetTrueVSize());
  if (const Operator* P = fes.GetProlongationMatrix()) {
    P->MultTranspose(lf, t);
  } else {
    t = lf;
  }
  return t;
}

// The part of a true-dof vector of a byNODES vector space tangential to
// the plane of unit normal nu, node by node: v - (v . nu) nu.
void ProjectTangential(Vector& v, const FiniteElementSpace& fes,
                       const Vector& nu) {
  MFEM_VERIFY(fes.GetOrdering() == Ordering::byNODES,
              "ProjectTangential: byNODES spaces only.");
  const auto vdim = fes.GetVDim(), n = v.Size() / vdim;
  for (int i = 0; i < n; i++) {
    real_t vn = 0.0;
    for (int c = 0; c < vdim; c++) {
      vn += v(c * n + i) * nu(c);
    }
    for (int c = 0; c < vdim; c++) {
      v(c * n + i) -= vn * nu(c);
    }
  }
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

  // The number of points no rank found in the mesh (sampled as zero).
  int NumMissing() const {
    auto found = 0;
    for (int i = 0; i < n_; i++) {
      found += elem_[i] >= 0;
    }
#ifdef MFEM_USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &found, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
    return n_ - found;
  }

  // Component-major: component c of point i at c * NumPoints() + i.
  vector<double> Sample(const GridFunction& u) const {
    auto values = vector<double>(static_cast<size_t>(dim_) * n_, 0.0);
    auto val = Vector(dim_);
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
  auto parent_true = Vector(parent.FESpace()->GetTrueVSize());
  injection.Mult(fault_true, parent_true);
  parent.SetFromTrueDofs(parent_true);
}

// A field on the fault space from a true-dof vector of the parent space.
void ParentToFault(const Operator& injection, const Vector& parent_true,
                   FieldType& fault) {
  auto fault_true = Vector(fault.FESpace()->GetTrueVSize());
  injection.MultTranspose(parent_true, fault_true);
  fault.SetFromTrueDofs(fault_true);
}

// The GLVis view angles (degrees) of a camera looking along the unit
// vector d (GLVisWindow::SetView).
std::pair<real_t, real_t> ViewAngles(const Vector& d) {
  constexpr auto deg = 180.0 / std::numbers::pi_v<real_t>;
  const auto theta = std::acos(std::clamp(-d(2), real_t{-1}, real_t{1}));
  return {deg * theta, deg * std::atan2(-d(0), d(1))};
}

// The dip section through the fault centre in a 3-D GLVis window: the keys
// that turn the cutting plane onto it, and the camera direction, along the
// strike. GLVis starts the plane with normal -x through the centre of the
// bounding box, turns its azimuth in 5-degree steps (y, Y), moves it along
// its normal in steps of 1/200 of the box diagonal (Z forward, z back) and
// cuts away the side its normal points to: the normal here is minus the
// strike direction, so the half between the camera and the section goes.
struct DipSection {
  string keys;
  Vector view;
};

DipSection DipSectionView(const FaultRecord& f) {
  constexpr auto pi = std::numbers::pi_v<real_t>;
  const auto step = pi / 36;
  const auto target = std::atan2(-f.strike_dir(1), -f.strike_dir(0));
  const auto turns =
      static_cast<int>(std::lround(std::remainder(target - pi, 2 * pi) / step));
  const auto azimuth = pi + turns * step;
  auto normal = Vector(3);
  normal(0) = std::cos(azimuth);
  normal(1) = std::sin(azimuth);
  normal(2) = 0.0;
  auto offset = Vector(f.centre);
  offset(2) -= 0.5 * (f.z_max - f.D);
  const auto height = f.D + f.z_max;
  const auto diagonal = std::sqrt(2.0 * f.W * f.W + height * height);
  const auto moves =
      static_cast<int>(std::lround((offset * normal) / (diagonal / 200)));
  auto section = DipSection{string(std::abs(turns), turns > 0 ? 'y' : 'Y') +
                                string(std::abs(moves), moves > 0 ? 'Z' : 'z'),
                            normal};
  section.view.Neg();
  return section;
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
  auto moduli = examples::ParametricFactor();
  bool compare = true;
  int nd = 12, ns = 12;
  real_t smooth = 1.0;
  real_t station_x = -0.5, station_w = 0.25;
  int n_green = 5;
  int serial_refinements = 0, parallel_refinements = 0;
  real_t sobolev_length = 0.1;
  int sobolev_order = 2;
  bool visualization = true;
  const char* csv_prefix = "coseismic_fault";

  // Read in command line options and process.
  auto args = OptionsParser(argc, argv);
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
  args.AddOption(&serial_refinements, "-rs", "--refine-serial",
                 "Uniform refinements of the mesh before it is partitioned.");
  args.AddOption(&parallel_refinements, "-rp", "--refine-parallel",
                 "Uniform refinements of the partitioned mesh (a serial "
                 "build adds them to -rs).");
  args.AddOption(&n_green, "-green", "--green-subfaults",
                 "Sub-faults for the Green's functions (0: none).");
  args.AddOption(&sobolev_length, "-sl", "--sobolev-length",
                 "Sobolev smoothing length of the kernel's Riesz map "
                 "(0: L2).");
  args.AddOption(&sobolev_order, "-so", "--sobolev-order",
                 "Sobolev order s of the kernel's Riesz map.");
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

  const auto f = ReadRecord(mesh_file);
  moduli.Verify(f.D, 0.0, "modulus");
  // Uniform refinements before (-rs) and after (-rp) the partitioning; a
  // serial build applies both to its one mesh.
  auto smesh = Mesh(mesh_file, 1, 1);
  const auto dim = smesh.Dimension();
  MFEM_VERIFY(dim == f.dim, "The mesh and its record differ in dimension.");
  for (int l = 0; l < serial_refinements; l++) {
    smesh.UniformRefinement();
  }
#ifdef MFEM_USE_MPI
  auto mesh = MeshType(MPI_COMM_WORLD, smesh);
  smesh.Clear();
#else
  auto& mesh = smesh;
#endif
  for (int l = 0; l < parallel_refinements; l++) {
    mesh.UniformRefinement();
  }
  const auto fault_size =
      f.fault_size / (1 << (serial_refinements + parallel_refinements));
  const auto na = mesh.attributes.Max(), nb = mesh.bdr_attributes.Max();

  auto fec = H1_FECollection(order, dim);
  auto fes = SpaceType(&mesh, &fec, dim);
#ifdef MFEM_USE_MPI
  const auto n_u = fes.GlobalTrueVSize();
#else
  const auto n_u = fes.GetTrueVSize();
#endif

  // Material: both moduli scaled by one factor (fixed Poisson ratio).
  const real_t lambda0 = 1.0, mu0 = 1.0;
  auto factor = moduli.Make(dim, f.W, f.D);
  auto kappa_c = FunctionCoefficient(
      [=](const Vector& x) { return (lambda0 + 2.0 * mu0 / dim) * factor(x); });
  auto mu_c =
      FunctionCoefficient([=](const Vector& x) { return mu0 * factor(x); });
  auto rheology = IsotropicElasticRheology(dim, kappa_c, mu_c);

  // The problem: every artificial side clamped, no surface traction, the
  // slip carried by the fault patch and pinned to zero where the patch
  // meets its extension.
  const auto ess = Marker(nb, {f.clamped_footwall, f.clamped_hanging_wall});
  const auto no_traction = Marker(nb, {});
  auto zero = Vector(dim);
  zero = 0.0;
  auto zero_c = VectorConstantCoefficient(zero);
  auto problem = LinearQuasiStaticClampedFaultProblem(
      &fes, rheology, ess, zero_c, no_traction, Marker(na, {f.hanging_wall}),
      Marker(nb, {f.fault}), Marker(nb, {f.extension}));

  // The unit slip direction of the hanging wall: up-dip in 2-D (thrust for
  // slip > 0), the rake in the fault plane in 3-D; and its sense, the
  // direction of the slip itself, for the profiles.
  auto direction = Vector(f.down_dip);
  direction.Neg();
  if (dim == 3) {
    add(std::cos(rake * deg), f.strike_dir, -std::sin(rake * deg), f.down_dip,
        direction);
  }
  auto slip_dir = Vector(direction);
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
  auto dg = L2_FECollection(order, dim, BasisType::GaussLobatto);
  auto dg_fes = SpaceType(&mesh, &dg, dim);
  auto u_dg = FieldType(&dg_fes);
  u_dg.ProjectCoefficient(problem.PhysicalDisplacement());

  // The scalar moment, the integral of mu |s| over the fault with the
  // local shear modulus: the first component of the dual of mu |s| e_0
  // paired with the field of ones.
  auto unit = Vector(dim);
  unit = 1.0;
  auto unit_c = VectorConstantCoefficient(unit);
  auto fault_marker = Marker(nb, {f.fault});
  {
    auto mu_slip_c = FunctionCoefficient([&](const Vector& x) {
      const auto [xi, eta] = FaultCoordinates(f, x);
      return mu0 * factor(x) * std::abs(amplitude(xi, eta));
    });
    auto e0 = Vector(dim);
    e0 = 0.0;
    e0(0) = 1.0;
    auto e0_c = VectorConstantCoefficient(e0);
    auto m_c = ScalarVectorProductCoefficient(mu_slip_c, e0_c);
    auto moment_form = LFType(&fes);
    moment_form.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(m_c),
                                      fault_marker);
    moment_form.Assemble();
    auto ones = FieldType(&fes);
    ones.ProjectCoefficient(unit_c);
    auto ones_true = Vector(fes.GetTrueVSize());
    ones.GetTrueDofs(ones_true);
    const auto moment = Dot(TrueDual(moment_form), ones_true);
    if (Root()) {
      cout << "Scalar moment: M0 = " << moment
           << (dim == 2 ? " (per unit strike length)" : "") << "\n";
    }
  }

  // The surface: the profile along it (3-D: along y = 0) and the peaks of
  // u_z over its dofs. The mesh surface is the piecewise-linear interpolant
  // of the topography, which it undercuts between nodes by up to
  // max|h''| s^2 / 8 (s the surface element size), so the sample points sit
  // below the recorded profile by twice that, plus a round-off floor.
  const auto npts = 201;
  const auto inset =
      1e-6 * f.D + 0.25 * f.max_curvature * f.surface_size * f.surface_size;
  auto surface_x = vector<double>(npts);
  auto surface_pts = DenseMatrix(dim, npts);
  surface_pts = 0.0;
  for (int i = 0; i < npts; i++) {
    const auto x = std::clamp(-0.5 * f.W + i * (f.W / (npts - 1)),
                              -0.5 * f.W + inset, 0.5 * f.W - inset);
    surface_x[i] = x;
    surface_pts(0, i) = x;
    surface_pts(dim - 1, i) = SurfaceHeight(f, x) - inset;
  }
  auto profile = PointSampler(mesh, surface_pts);
  if (const auto missing = profile.NumMissing(); missing > 0 && Root()) {
    cout << "Warning: " << missing << " of " << npts
         << " surface sample points lie outside the mesh (sampled as 0).\n";
  }
  const auto surface_u = profile.Sample(u_dg);

  auto fes_z = SpaceType(&mesh, &fec);
  auto u_z = FieldType(&fes_z);
  auto u_w_c = VectorGridFunctionCoefficient(&problem.Displacement());
  auto e_z = Vector(dim);
  e_z = 0.0;
  e_z(dim - 1) = 1.0;
  auto e_z_c = VectorConstantCoefficient(e_z);
  auto u_z_c = InnerProductCoefficient(u_w_c, e_z_c);
  u_z.ProjectCoefficient(u_z_c);
  const auto [uz_min, uz_max] = examples::SurfacePeaks(u_z, fes_z, f.surface);
  if (Root()) {
    cout << "Peak surface uplift:     " << uz_max
         << "\nPeak surface subsidence: " << uz_min << "\n";
  }

  // The fault as a space of its own (the shadow space on its boundary
  // SubMesh) and its Riesz maps, the slip pinned to zero on the fault edge,
  // which is the whole boundary of the SubMesh: L2 for the traction, the
  // Sobolev map of -sl, -so for the kernel (-sl 0: L2).
  auto fault_attr = Array<int>({f.fault});
  auto fault_mesh = SubMeshType::CreateFromBoundary(mesh, fault_attr);
  auto fault_fes = SubMeshDofInjection::MakeShadowSpace(fes, fault_mesh);
  auto fault_injection = SubMeshDofInjection(*fault_fes, fes);
#ifdef MFEM_USE_MPI
  auto pi_fault = std::unique_ptr<Operator>(fault_injection.NewTrueDofMatrix());
#else
  auto pi_fault = std::unique_ptr<Operator>(fault_injection.NewSparseMatrix());
#endif
  auto fault_edge = Array<int>(fault_mesh.bdr_attributes.Max());
  fault_edge = 1;
  auto l2_riesz = SobolevRieszMap(*fault_fes, 1.0, 0.0, 1, &fault_edge);
  auto kernel_riesz =
      SobolevRieszMap(*fault_fes, 1.0, sobolev_length * sobolev_length,
                      sobolev_length > 0.0 ? sobolev_order : 1, &fault_edge);

  // On the fault: the slip and the traction change tau = sigma nu, the L2
  // representative of minus the fault reaction (a dual vector, on the
  // fault space through the transpose of the injection), sampled along dip
  // (3-D: through the centre of the patch).
  auto r_true = Vector();
  problem.FaultReaction(r_true);
  r_true.Neg();
  auto tau_dual = Vector(fault_fes->GetTrueVSize());
  auto tau_f = Vector();
  pi_fault->MultTranspose(r_true, tau_dual);
  l2_riesz.Mult(tau_dual, tau_f);
  auto tau = FieldType(&fes);
  FaultToParent(*pi_fault, tau_f, tau);
  auto slip_true_parent = Vector(fes.GetTrueVSize());
  problem.Slip().GetTrueDofs(slip_true_parent);
  const int nfault = 101;
  auto fault_pts = DenseMatrix(dim, nfault);
  auto xi_pts = vector<double>(nfault);
  for (int j = 0; j < nfault; j++) {
    xi_pts[j] = (j + 0.5) / nfault - 0.5;
    for (int c = 0; c < dim; c++) {
      fault_pts(c, j) = f.centre(c) + xi_pts[j] * f.length * f.down_dip(c);
    }
  }
  auto on_fault = PointSampler(mesh, fault_pts);
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
  auto glut_u = vector<double>();
  if (compare) {
    auto box = examples::FaultBox();
    box.W = f.W;
    box.D = f.D;
    box.nx = std::max(1, static_cast<int>(std::lround(f.W / fault_size)));
    box.nz = std::max(1, static_cast<int>(std::lround(f.D / fault_size)));
    auto glut = examples::Fault();
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
    auto glut_problem = LinearQuasiStaticClampedProblem(&fes, rheology, ess,
                                                        zero_c, no_traction);
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
  auto surface_attr = Array<int>({f.surface});
  auto surface_mesh = SubMeshType::CreateFromBoundary(mesh, surface_attr);
  auto surface_fes = SubMeshDofInjection::MakeShadowSpace(fes, surface_mesh);
  auto G = FaultToSurfaceMap(problem, *fault_fes, *surface_fes);

  // The observable: the mean uplift in a Gaussian window, J = <u, w> /
  // integral of the window, w the surface dual of the window weight.
  auto window_c = FunctionCoefficient([=](const Vector& x) {
    auto r2 = (x(0) - station_x) * (x(0) - station_x);
    if (dim == 3) {
      r2 += x(1) * x(1);
    }
    return std::exp(-0.5 * r2 / (station_w * station_w));
  });
  auto weight_c = ScalarVectorProductCoefficient(window_c, e_z_c);
  auto w_form = LFType(surface_fes.get());
  w_form.AddDomainIntegrator(new VectorDomainLFIntegrator(weight_c));
  w_form.Assemble();
  const auto w = TrueDual(w_form);
  auto e_z_surface = FieldType(surface_fes.get());
  e_z_surface.ProjectCoefficient(e_z_c);
  auto e_z_true = Vector(surface_fes->GetTrueVSize());
  e_z_surface.GetTrueDofs(e_z_true);
  const auto window = Dot(w, e_z_true);

  auto slip_f = FieldType(fault_fes.get());
  slip_f.ProjectCoefficient(slip_c);
  auto s_true = Vector(fault_fes->GetTrueVSize());
  slip_f.GetTrueDofs(s_true);
  auto Gs = Vector();
  auto Gtw = Vector();
  G.Mult(s_true, Gs);
  const auto J_forward = Dot(Gs, w) / window;
  G.MultTranspose(w, Gtw);
  const auto J_adjoint = Dot(s_true, Gtw) / window;
  if (Root()) {
    cout << "Mean uplift in the window at x = " << station_x
         << ": J = " << J_forward << " (forward), " << J_adjoint
         << " (adjoint); relative difference "
         << std::abs(J_forward - J_adjoint) /
                std::max(std::abs(J_forward), real_t{1e-300})
         << "\n";
  }

  // The gradient of J with respect to tangential slip: the dual G^T w /
  // (window integral), its tangential part mapped by the kernel's Riesz
  // map; on the parent space for sampling.
  auto kernel_dual = Vector(Gtw);
  auto kernel_f = Vector();
  kernel_dual /= window;
  ProjectTangential(kernel_dual, *fault_fes, f.normal);
  kernel_riesz.Mult(kernel_dual, kernel_f);
  auto kernel = FieldType(&fes);
  FaultToParent(*pi_fault, kernel_f, kernel);
  const auto kernel_s = on_fault.Sample(kernel);

  // Green's functions: the surface uplift of unit slip (in the slip
  // direction) on each of n_green equal sub-faults along dip, and their
  // sum against the response to unit slip on the whole fault.
  auto green = vector<vector<double>>();
  if (n_green > 0) {
    const auto sense = slip < 0.0 ? -1.0 : 1.0;
    auto unit_slip = [&](int k) {
      return Slip(f, direction, [=](real_t xi, real_t eta) -> real_t {
        if (std::abs(xi) >= 0.5 || std::abs(eta) >= 0.5) {
          return 0.0;
        }
        const auto patch = std::min(
            n_green - 1, static_cast<int>(std::floor((xi + 0.5) * n_green)));
        return (k < 0 || patch == k) ? sense : 0.0;
      });
    };
    auto sum = vector<double>(npts, 0.0);
    for (int k = -1; k < n_green; k++) {
      auto c = unit_slip(k);
      slip_f.ProjectCoefficient(c);
      slip_f.GetTrueDofs(s_true);
      G.Mult(s_true, Gs);
      const auto u = profile.Sample(problem.Displacement());
      auto col =
          vector<double>(u.begin() + (dim - 1) * npts, u.begin() + dim * npts);
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
  const auto prefix = string(csv_prefix);
  if (!prefix.empty()) {
    auto comps = dim == 2 ? vector<string>{"u_x", "u_z"}
                          : vector<string>{"u_x", "u_y", "u_z"};
    auto cols = vector<string>{"x"};
    auto y_spec = string();
    auto y_labels = string();
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
    cols.push_back("height");
    y_spec += "|height";
    y_labels += "|surface height";
    auto table = examples::CsvTable(prefix + ".csv", cols);
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
      auto row = vector<double>{surface_x[i]};
      for (int c = 0; c < dim; c++) {
        row.push_back(surface_u[c * npts + i]);
      }
      if (compare) {
        for (int c = 0; c < dim; c++) {
          row.push_back(glut_u[c * npts + i]);
        }
      }
      row.push_back(SurfaceHeight(f, surface_x[i]));
      table.Row(row);
    }
    table.Write();

    const auto metric = sobolev_length > 0.0
                            ? "H^" + to_string(sobolev_order) + ", length " +
                                  to_string(sobolev_length)
                            : string("L2");
    auto fault_table =
        examples::CsvTable(prefix + "_fault.csv",
                           {"xi", "slip", "tau_shear", "tau_normal", "kernel"});
    fault_table.Meta("title", "Along dip: slip, traction change, kernel")
        .Meta("note",
              "shear: along the slip; normal: into the hanging wall; "
              "traction in L2; kernel: gradient of J along the slip in " +
                  metric)
        .Meta("xlabel", "xi")
        .Meta("y", "slip|tau_shear,tau_normal|kernel")
        .Meta("ylabel", "slip|traction change|dJ/ds");
    for (int j = 0; j < nfault; j++) {
      fault_table.Row({xi_pts[j], along(slip_s, j, slip_dir),
                       along(tau_s, j, slip_dir), along(tau_s, j, f.normal),
                       along(kernel_s, j, slip_dir)});
    }
    fault_table.Write();

    if (n_green > 0) {
      auto gcols = vector<string>{"x", "whole_fault"};
      for (int k = 0; k < n_green; k++) {
        gcols.push_back("subfault_" + to_string(k + 1));
      }
      auto green_table = examples::CsvTable(prefix + "_green.csv", gcols);
      green_table.Meta("title", "Green's functions: uplift per unit slip")
          .Meta("note", "sub-faults numbered up-dip to down-dip")
          .Meta("xlabel", "x")
          .Meta("ylabel", "u_z");
      for (int i = 0; i < npts; i++) {
        auto row = vector<double>{surface_x[i]};
        for (const auto& col : green) {
          row.push_back(col[i]);
        }
        green_table.Row(row);
      }
      green_table.Write();
    }
  }

  // The physical u_z on the colour scale of the surface peaks (the field
  // jumps at the fault), the physical displacement vector, and in 3-D the
  // slip, the shear traction change and the kernel on the fault surface.
  if (visualization) {
    auto dg_scalar = L2_FECollection(order, dim, BasisType::GaussLobatto);
    auto dg_z_fes = SpaceType(&mesh, &dg_scalar);
    auto u_dg_z = FieldType(&dg_z_fes);
    auto u_dg_c = VectorGridFunctionCoefficient(&u_dg);
    auto u_dg_z_c = InnerProductCoefficient(u_dg_c, e_z_c);
    u_dg_z.ProjectCoefficient(u_dg_z_c);
    // In 3-D both volume windows are cut open along the dip section through
    // the fault centre and seen along the strike, so the fault shows its
    // dip.
    const auto section = dim == 3 ? DipSectionView(f) : DipSection{};
    const auto [view_theta, view_phi] =
        dim == 3 ? ViewAngles(section.view) : std::pair<real_t, real_t>{};
    auto window_z = examples::GLVisWindow(
        "u_z (meshed fault)",
        dim == 2 ? examples::DefaultKeys(2) : "i" + section.keys + "lc");
    const auto span =
        std::max({std::abs(uz_min), std::abs(uz_max), real_t{1e-30}});
    window_z.SetValueRange(-1.2 * span, 1.2 * span);
    auto vec_window = examples::GLVisWindow(
        "displacement (meshed fault)",
        dim == 2 ? string("Rjlcvvvvv") : "i" + section.keys + "lcvvvvv");
    if (dim == 3) {
      window_z.SetView(view_theta, view_phi);
      vec_window.SetView(view_theta, view_phi);
    }
    window_z.Send(mesh, u_dg_z);
    vec_window.Send(mesh, u_dg);

    if (dim == 3) {
      auto slip_fault = FieldType(fault_fes.get());
      auto tau_fault = FieldType(fault_fes.get());
      auto kernel_fault = FieldType(fault_fes.get());
      ParentToFault(*pi_fault, slip_true_parent, slip_fault);
      auto shear = Vector(tau_f);
      ProjectTangential(shear, *fault_fes, f.normal);
      tau_fault.SetFromTrueDofs(shear);
      kernel_fault.SetFromTrueDofs(kernel_f);
      // Seen face on, from the hanging-wall side.
      auto towards_fault = Vector(f.normal);
      towards_fault.Neg();
      const auto [face_theta, face_phi] = ViewAngles(towards_fault);
      const auto* fault_keys = "lcvvvvv";
      auto slip_window = examples::GLVisWindow("fault: slip", fault_keys);
      auto tau_window =
          examples::GLVisWindow("fault: shear traction change", fault_keys);
      auto kernel_window =
          examples::GLVisWindow("fault: kernel dJ/ds", fault_keys);
      for (auto* window : {&slip_window, &tau_window, &kernel_window}) {
        window->SetView(face_theta, face_phi);
      }
      slip_window.Send(fault_mesh, slip_fault);
      tau_window.Send(fault_mesh, tau_fault);
      kernel_window.Send(fault_mesh, kernel_fault);
    }
  }
  return ok && G.Converged() ? 0 : 1;
}
