//------------------------------------------------------------------------------
//
// PURPOSE:
//
// This example (relabelled_elasticity) demonstrates the particle-relabelling
// symmetry at the heart of the library's referential problems: ONE physical
// elastic body, TWO descriptions.
//
// The physical body is a two-layer "planet" whose core-mantle boundary and
// outer surface are genuinely aspherical (a quadrupole bump). It is solved
// twice:
//
// 1. **The natural description.** The computational mesh IS the deformed
//    body (the reference disc mesh with its nodes pushed through the
//    relabelling, giving curved elements), the coefficients are the plain
//    physical ones (piecewise-constant moduli and density by layer), and
//    the problem is an ordinary traction-free elastic solve under a body
//    force.
//
// 2. **The relabelled description.** The computational mesh is the pristine
//    CIRCULAR reference disc; the aspherical shape lives entirely in the
//    equilibrium mapping phi_e = xi carried by the rheology. The
//    constitutive data are pulled back (RelabelledElasticTensorCoefficient:
//    the isotropic tensor acquires an anisotropic-looking referential
//    expression), the density becomes rho~ = J rho (the Jacobian carries
//    the volume change of the labels), and the body force is composed with
//    the map. Nothing about the solver changes: the referential rheology's
//    mapped integrators handle every geometric factor.
//
// Because the physical mesh's nodes are exactly the images of the
// reference mesh's nodes, the two discrete solutions share one dof layout
// and must agree NODE BY NODE (u_rel(X) = u_nat(xi(X)); for nodal H1
// elements the vdof vectors coincide at discretisation level) -- and their
// strain energies, being relabelling-invariant, must match. Both checks
// are printed. The rigid-mode gauges of the two solves coincide too: the
// referential projector uses rotations of the MAPPED positions, which are
// precisely the physical mesh's rotations.
//
// Visualisation (GLVis; -no-vis to skip):
//   a. the physical density on the deformed body, and the relabelled
//      density rho~ = J rho on the circular reference (same body, the
//      labels' bookkeeping made visible);
//   b. the displacement magnitude of both descriptions;
//   c. the relabelled solution pushed forward onto the deformed body --
//      the natural picture reappears.
//
// Self-gravitating relabelled runs, where the potential joins the game,
// live in the relabelled benchmark (benchmarks/relabelling,
// doc/benchmarks.tex "The relabelling family"); this example keeps gravity
// off so the symmetry itself is the whole story.
//
// One source serves the serial and the parallel build, as throughout.
//
// Options (defaults in brackets):
//   -m    mesh file [../data/elastogravity_two_layer_2d.msh]: the
//         two-layer disc; only its inner layer is used here.
//   -o    finite element order [2]; raising it tightens the agreement
//         checks (the map is resolved by the geometry order).
//   -r    uniform refinements of the serial mesh [0]; likewise.
//   -a    amplitude of the quadrupole relabelling map [0.1]: how far
//         the deformed body departs from the circular reference; the
//         identity of the two descriptions holds at ANY amplitude, so
//         pushing it up is a stress test, not a small-parameter limit.
//   -vis / -no-vis   GLVis windows on or off [on].
//
// Sample runs:  ./relabelled_elasticity
//               ./relabelled_elasticity -a 0.15 -o 3
//               mpiexec -np 4 ./relabelled_elasticity -r 1  (parallel build)
//
//------------------------------------------------------------------------------

#include <cmath>
#include <iostream>
#include <memory>

#include "AdGIA.hpp"
#include "mfem.hpp"
#include "visualisation.hpp"

using namespace mfem;
using namespace AdGIA;

#ifdef MFEM_USE_MPI
using MeshType = ParMesh;
using SubMeshType = ParSubMesh;
using SpaceType = ParFiniteElementSpace;
using FieldType = ParGridFunction;
#else
using MeshType = Mesh;
using SubMeshType = SubMesh;
using SpaceType = FiniteElementSpace;
using FieldType = GridFunction;
#endif

namespace {

bool Root() {
#ifdef MFEM_USE_MPI
  return Mpi::Root();
#else
  return true;
#endif
}

real_t GlobalDot(const Vector& x, const Vector& y) {
#ifdef MFEM_USE_MPI
  return InnerProduct(MPI_COMM_WORLD, x, y);
#else
  return InnerProduct(x, y);
#endif
}

// The two-layer disc of the test data: core (attribute 1) and mantle
// (attribute 2) under a vacuum buffer (attribute 3, excluded here).
constexpr real_t kRCmb = 3483.0 / 6371.0;
constexpr real_t kRhoCore = 2.2, kRhoMantle = 0.9;     // benchmark-style
constexpr real_t kKappaCore = 7.4, kMuCore = 2.0;      // a SOLID core:
constexpr real_t kKappaMantle = 5.4, kMuMantle = 2.7;  // no fluid gauge

// The body force per unit mass, a smooth deviatoric pattern of the
// PHYSICAL position (f = f0 (y, x): a pure-shear forcing).
void BodyForce(const Vector& x, Vector& f) {
  f.SetSize(2);
  f[0] = 0.05 * x[1];
  f[1] = 0.05 * x[0];
}

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif

  const char* mesh_file = "../data/elastogravity_two_layer_2d.msh";
  int order = 2;
  int refinement = 0;
  real_t amplitude = 0.1;
  bool visualization = true;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Two-layer disc mesh.");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&refinement, "-r", "--refinement",
                 "Uniform refinements of the serial mesh.");
  args.AddOption(&amplitude, "-a", "--amplitude",
                 "Amplitude of the quadrupole relabelling (diffeomorphic "
                 "for |a| well below ~0.2).");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization", "Send the fields to GLVis.");
  args.Parse();
  if (!args.Good()) {
    if (Root()) args.PrintUsage(std::cout);
    return 1;
  }
  if (Root()) args.PrintOptions(std::cout);

  // === The relabelling xi and its exact deformation gradient ===
  //
  // xi(x) = (1 + g) x with the quadrupole g = a (x^2 - y^2): a smooth map
  // that inflates the body along one axis and deflates it along the
  // other, moving BOTH the core-mantle boundary and the outer surface
  // (each layer maps onto its deformed self, so element faces stay on
  // the material interface on both meshes). Its gradient is
  // F = (1 + g) I + x (grad g)^T, exact, with grad g = 2a (x, -y).
  const real_t a = amplitude;
  auto xi_map = [a](const Vector& x, Vector& y) {
    y.SetSize(2);
    const real_t g = a * (x[0] * x[0] - x[1] * x[1]);
    y[0] = (1.0 + g) * x[0];
    y[1] = (1.0 + g) * x[1];
  };
  auto xi_grad = [a](const Vector& x, DenseMatrix& F) {
    F.SetSize(2);
    const real_t g = a * (x[0] * x[0] - x[1] * x[1]);
    F(0, 0) = 1.0 + g + x[0] * 2.0 * a * x[0];
    F(0, 1) = x[0] * (-2.0 * a * x[1]);
    F(1, 0) = x[1] * 2.0 * a * x[0];
    F(1, 1) = 1.0 + g + x[1] * (-2.0 * a * x[1]);
  };
  CallableDiffeomorphism xi(2, xi_map, xi_grad);

  // === The reference mesh and the two bodies ===
  auto serial_mesh = Mesh(mesh_file, 1, 1);
  for (int l = 0; l < refinement; l++) {
    serial_mesh.UniformRefinement();
  }
#ifdef MFEM_USE_MPI
  auto parent_ref = ParMesh(MPI_COMM_WORLD, serial_mesh);
  auto parent_phys = ParMesh(MPI_COMM_WORLD, serial_mesh);
  serial_mesh.Clear();
#else
  Mesh& parent_ref = serial_mesh;
  auto parent_phys = Mesh(serial_mesh);
#endif
  Array<int> body_attrs({1, 2});
  auto body_ref = SubMeshType::CreateFromDomain(parent_ref, body_attrs);
  auto body_phys = SubMeshType::CreateFromDomain(parent_phys, body_attrs);

  const int dim = body_ref.Dimension();
  MFEM_VERIFY(dim == 2, "This example is two-dimensional.");
  auto fec = H1_FECollection(order, dim);

  // Deform the physical body: give it order-matched curved geometry and
  // push every node through the relabelling. Its nodes are now exactly
  // the images of the reference body's nodes.
  auto xi_coeff = VectorFunctionCoefficient(dim, xi_map);
  auto nodes_phys = SpaceType(&body_phys, &fec, dim);
  body_phys.SetNodalFESpace(&nodes_phys);
  {
    auto y = FieldType(&nodes_phys);
    y.ProjectCoefficient(xi_coeff);
    *body_phys.GetNodes() = y;
  }

  auto fes_ref = SpaceType(&body_ref, &fec, dim);
  auto fes_phys = SpaceType(&body_phys, &fec, dim);

  // === Shared data ===
  // Layer-wise physical constants: attribute-based, so they attach to the
  // material LABELS and need no composition under the relabelling.
  Vector rho_vals({kRhoCore, kRhoMantle});
  Vector kappa_vals({kKappaCore, kKappaMantle});
  Vector mu_vals({kMuCore, kMuMantle});
  auto rho = PWConstCoefficient(rho_vals);
  auto kappa = PWConstCoefficient(kappa_vals);
  auto mu = PWConstCoefficient(mu_vals);
  auto zero_traction_vec = Vector(dim);
  zero_traction_vec = 0.0;
  auto zero_traction = VectorConstantCoefficient(zero_traction_vec);
  auto surface = Array<int>(body_ref.bdr_attributes.Max());
  surface = 1;  // every body boundary is traction-free

  // === 1. The natural description, on the deformed body ===
  auto rheology_nat = IsotropicElasticRheology(dim, kappa, mu);
  auto nat = LinearQuasiStaticTractionProblem(&fes_phys, rheology_nat,
                                              zero_traction, surface);
  auto f_phys = VectorFunctionCoefficient(dim, BodyForce);
  auto load_nat = ScalarVectorProductCoefficient(rho, f_phys);
  nat.ExternalLoad().AddDomainIntegrator(
      new VectorDomainLFIntegrator(load_nat));
  nat.AssembleForce(0.0);
  if (!nat.Solve() && Root()) {
    std::cout << "natural solve: NOT CONVERGED\n";
  }

  // === 2. The relabelled description, on the circular reference ===
  //
  // The constitutive state (C~, S~ = 0, phi_e = xi): the attribute-wise
  // isotropic tensor is already its own composition with xi (labels), and
  // RelabelledElasticTensorCoefficient supplies the pull-back algebra.
  auto C_iso = IsotropicElasticTensorCoefficient::FromBulkModulus(
      dim, kappa, mu);
  auto C_rel = RelabelledElasticTensorCoefficient(dim, C_iso, xi);
  auto S_zero_mat = DenseMatrix(dim);
  S_zero_mat = 0.0;
  auto S_zero = MatrixConstantCoefficient(S_zero_mat);
  auto rheology_rel = ReferentialElasticRheology(dim, C_rel, S_zero, xi);
  auto rel = LinearQuasiStaticTractionProblem(&fes_ref, rheology_rel,
                                              zero_traction, surface);
  // The pulled-back load: rho~(X) f(xi(X)) with rho~ = J rho. On the
  // reference body the layer of a label is set by its radius, so the
  // whole load is one analytic coefficient.
  auto load_rel =
      VectorFunctionCoefficient(dim, [a](const Vector& X, Vector& v) {
        Vector x_phys;
        DenseMatrix F;
        // reuse the map and gradient inline
        const real_t g = a * (X[0] * X[0] - X[1] * X[1]);
        x_phys.SetSize(2);
        x_phys[0] = (1.0 + g) * X[0];
        x_phys[1] = (1.0 + g) * X[1];
        F.SetSize(2);
        F(0, 0) = 1.0 + g + 2.0 * a * X[0] * X[0];
        F(0, 1) = -2.0 * a * X[0] * X[1];
        F(1, 0) = 2.0 * a * X[0] * X[1];
        F(1, 1) = 1.0 + g - 2.0 * a * X[1] * X[1];
        const real_t J = F.Det();
        const real_t rho0 = X.Norml2() < kRCmb ? kRhoCore : kRhoMantle;
        BodyForce(x_phys, v);
        v *= J * rho0;
      });
  rel.ExternalLoad().AddDomainIntegrator(
      new VectorDomainLFIntegrator(load_rel));
  rel.AssembleForce(0.0);
  if (!rel.Solve() && Root()) {
    std::cout << "relabelled solve: NOT CONVERGED\n";
  }

  // === 3. The checks ===
  //
  // (i) Node-by-node displacement agreement: the two solves share one dof
  // layout and the physical nodes are the mapped reference nodes, so
  // u_rel(X_i) = u_nat(xi(X_i)) makes the true-dof vectors agree at
  // discretisation level.
  Vector U_nat, U_rel;
  nat.Displacement().GetTrueDofs(U_nat);
  rel.Displacement().GetTrueDofs(U_rel);
  Vector d(U_rel);
  d -= U_nat;
  const real_t rel_diff = std::sqrt(GlobalDot(d, d)) /
                          std::sqrt(GlobalDot(U_nat, U_nat));
  // (ii) The strain energy is relabelling-invariant.
  auto energy = [](LinearQuasiStaticTractionProblem& p,
                   const Vector& U) {
    Vector AU(U.Size());
    p.SystemMatrix().Ptr()->Mult(U, AU);
    return 0.5 * GlobalDot(U, AU);
  };
  const real_t e_nat = energy(nat, U_nat);
  const real_t e_rel = energy(rel, U_rel);
  if (Root()) {
    std::cout << "\nOne physical body, two descriptions (amplitude " << a
              << "):\n"
              << "  displacement agreement |u_rel - u_nat| / |u_nat|  "
              << rel_diff << "\n"
              << "  strain energy, natural description               "
              << e_nat << "\n"
              << "  strain energy, relabelled description            "
              << e_rel << "  (relative difference "
              << std::abs(e_rel / e_nat - 1.0) << ")\n\n";
  }

  // === 4. Visualisation ===
  if (visualization) {
    const std::string keys = examples::DefaultKeys(dim);
    // The densities: the physical one on the deformed body, and the
    // relabelled rho~ = J rho the circular description carries.
    auto l2 = L2_FECollection(order - 1, dim);
    auto dfes_phys = SpaceType(&body_phys, &l2);
    auto dfes_ref = SpaceType(&body_ref, &l2);
    auto rho_phys_gf = FieldType(&dfes_phys);
    rho_phys_gf.ProjectCoefficient(rho);
    auto Jc = JacobianCoefficient(xi);
    auto rho_rel_c = ProductCoefficient(Jc, rho);
    auto rho_rel_gf = FieldType(&dfes_ref);
    rho_rel_gf.ProjectCoefficient(rho_rel_c);
    examples::GLVisWindow("density: the physical body", keys)
        .Send(body_phys, rho_phys_gf);
    examples::GLVisWindow("density: relabelled, rho~ = J rho", keys)
        .Send(body_ref, rho_rel_gf);
    // The displacements.
    examples::GLVisWindow("displacement: natural description", keys)
        .Send(body_phys, nat.Displacement());
    examples::GLVisWindow("displacement: relabelled description", keys)
        .Send(body_ref, rel.Displacement());
    // Push the relabelled solution forward onto the deformed body: the
    // natural picture reappears (its data already lives on the shared
    // dof layout).
    auto pushed = FieldType(&fes_phys);
    pushed = rel.Displacement();
    examples::GLVisWindow("relabelled, pushed forward", keys)
        .Send(body_phys, pushed);
  }
  return 0;
}
