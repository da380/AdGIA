// ============================================================================
// referential_elastogravity.cpp
//
// The general linearised referential problem
// (LinearQuasiStaticReferentialSelfGravitatingProblem, doc/gravitating_elasticity.md)
// beside the traditional Eulerian one
// (LinearQuasiStaticMixedSelfGravitatingProblem, doc/self_gravitation.md), on a
// uniform self-gravitating body under a degree-2 surface mass load: two
// mathematically equivalent formalisms solving one physical problem, so
// their agreement checks both.
//
// The referential side is assembled from the ingredients described in
// doc/gravitating_elasticity.md ("2. The linearised quasi-static problem"
// and "3.1 The linearised referential system"):
//   - the BARE moduli, converted from the seismological ones with the
//     hydrostatic pressure (BareElasticTensorCoefficient; the moduli in
//     Earth models are NOT the strain-energy moduli — the conversion is
//     ~p0 in lambda and mu, not small at depth);
//   - the equilibrium stress S_e = -p0(r) 1 acting on the FULL
//     displacement gradient (GeometricStiffnessIntegrator);
//   - referential gravity: the potential unknown is zeta = phi o phi_e,
//     no density gradients or grad grad Phi0 anywhere, and rigid modes
//     carry no potential partners (with a rigid-preserving extension the
//     translations are EXACT discrete null pairs; the tapered extension
//     below does not preserve them, so they are merely near-null,
//     decreasing with order — printed for inspection);
//   - the vacuum extension: the buffer's gravity terms folded through a
//     prescribed radial extension E (NewRadialVacuumExtension) — a gauge
//     choice, so any other smooth taper must give the same observables.
//
// The two solutions are compared through the change of variables
// zeta1 = phi1 + u . grad(Phi0) (modulo the constant in 2-D): agreement is
// at the level of the two discretisations, improving with order.
//
// One source serves the serial and the parallel build. The two problems
// live on two separately-partitioned copies of one mesh; partitioning
// the same serial mesh twice in one run gives identical partitions, so
// the field-by-field comparison carries over. The other genuine
// differences: comparison copies are made with the build's field type
// (a plain GridFunction on a parallel space would compute rank-local
// norms), and the 2-D constant gauge is removed with a global mean.
//
// Output: the rigid pair residuals, the relative differences of the
// displacements and of the potentials, the iteration counts, and with -vis
// (on by default) both displacements and the referential potential in
// GLVis.
//
// Sample runs (with mpiexec -np N in front in a parallel build):
//    ./referential_elastogravity
//    ./referential_elastogravity -o 3
//    ./referential_elastogravity -m ../data/coupled_poisson.msh -o 2
// ============================================================================

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>

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

// A field of the build's type on a space of either type: comparison
// copies must be ParGridFunctions in parallel or their norms come out
// rank-local.
std::unique_ptr<GridFunction> MakeField(FiniteElementSpace& fes) {
#ifdef MFEM_USE_MPI
  if (auto* pfes = dynamic_cast<ParFiniteElementSpace*>(&fes)) {
    return std::make_unique<ParGridFunction>(pfes);
  }
#endif
  return std::make_unique<GridFunction>(&fes);
}

// Non-dimensional model: unit radius, strong coupling.
constexpr double kG = 0.05;
constexpr double kRho = 1.0;
constexpr double kKappa = 1.0;
constexpr double kMu = 0.5;
constexpr int kDtNDegree = 12;
constexpr double kPi = std::numbers::pi;

// Hydrostatic pressure of the uniform body, 2-D and 3-D conventions.
double Pressure(const Vector& x) {
  const double r2 = x * x;
  const double p = x.Size() == 2 ? kPi * kG * kRho * kRho * (1.0 - r2)
                                 : 2.0 * kPi * kG * kRho * kRho *
                                       (1.0 - r2) / 3.0;
  return std::max(0.0, p);
}

// Degree-2 surface mass load.
double SurfaceLoad(const Vector& x) {
  const double r = x.Norml2();
  const double c = (x.Size() == 2 ? x[1] : x[2]) / r;
  return 0.02 * (1.0 + 3.0 * c * c);
}

CallableDiffeomorphism IdentityMap(int dim) {
  return CallableDiffeomorphism(
      dim, [](const Vector& x, Vector& y) { y = x; },
      [](const Vector&, DenseMatrix& F) {
        F = 0.0;
        for (int i = 0; i < F.Height(); i++) {
          F(i, i) = 1.0;
        }
      });
}

double L2Norm(const GridFunction& u) {
  const int vdim = u.FESpace()->GetVDim();
  Vector zero(vdim);
  zero = 0.0;
  if (vdim == 1) {
    ConstantCoefficient z(0.0);
    return const_cast<GridFunction&>(u).ComputeL2Error(z);
  }
  VectorConstantCoefficient z(zero);
  return const_cast<GridFunction&>(u).ComputeL2Error(z);
}

// Send a field to GLVis.
void Show(Mesh& mesh, const GridFunction& f, const char* title) {
  char vishost[] = "localhost";
  socketstream sock(vishost, 19916);
  sock.precision(8);
#ifdef MFEM_USE_MPI
  sock << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() << "\n";
#endif
  sock << "solution\n"
       << mesh << f << "window_title '" << title << "'"
       << (mesh.Dimension() == 2 ? "\nkeys Rjlbc\n" : "\nkeys RRRilc\n")
       << std::flush;
}

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif

  const char* mesh_file = "../data/elastogravity_2d.msh";
  int order = 2;
  bool visualization = true;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization", "GLVis visualisation.");
  args.Parse();
  if (!args.Good()) {
    if (Root()) {
      args.PrintUsage(std::cout);
    }
    return 1;
  }

  Mesh smesh(mesh_file, 1, 1);
  const int dim = smesh.Dimension();
#ifdef MFEM_USE_MPI
  MeshType parent(MPI_COMM_WORLD, smesh);
#else
  MeshType& parent = smesh;
#endif
  Array<int> body_attr({1}), buffer_attr({2});
  auto body = SubMeshType::CreateFromDomain(parent, body_attr);
  H1_FECollection fec(order, dim);
  SpaceType fes_u(&body, &fec, dim);
  SpaceType fes_phi(&parent, &fec);

  // Surface marker: the body SubMesh's largest boundary attribute.
  Array<int> surface(body.bdr_attributes.Max());
  surface = 0;
  surface[body.bdr_attributes.Max() - 1] = 1;
  FunctionCoefficient sigma(SurfaceLoad);
  ConstantCoefficient rho(kRho);

  // --- The Eulerian problem: seismological moduli go in directly. -----------
  ConstantCoefficient kappa(kKappa), mu(kMu);
  IsotropicElasticRheology e_rheology(dim, kappa, mu);
  LinearQuasiStaticMixedSelfGravitatingProblem eulerian(
      &fes_u, &fes_phi, e_rheology, rho, kG, kDtNDegree);
  eulerian.SetSurfaceLoad(sigma, surface);
  eulerian.SetRelTol(1e-11);
  eulerian.AssembleForce(0.0);
  if (!eulerian.Solve()) {
    if (Root()) {
      std::cout << "Eulerian solve failed\n";
    }
    return 1;
  }

  // --- The referential problem: bare moduli, S_e, mapping, extension. -------
  // A second copy of the mesh (partitioned identically in parallel: same
  // serial mesh, same partitioner, one run).
  Mesh smesh2(mesh_file, 1, 1);
#ifdef MFEM_USE_MPI
  MeshType parent2(MPI_COMM_WORLD, smesh2);
  smesh2.Clear();
#else
  MeshType& parent2 = smesh2;
#endif
  auto body2 = SubMeshType::CreateFromDomain(parent2, body_attr);
  auto buffer2 = SubMeshType::CreateFromDomain(parent2, buffer_attr);
  SpaceType fes_u2(&body2, &fec, dim);
  SpaceType fes_zeta(&parent2, &fec);
  SpaceType fes_buffer(&buffer2, &fec, dim);

  auto phi_e = IdentityMap(dim);
  FunctionCoefficient p0(Pressure);
  auto C_eff = IsotropicElasticTensorCoefficient::FromBulkModulus(dim, kappa,
                                                                  mu);
  BareElasticTensorCoefficient C_bare(dim, C_eff, p0);
  MatrixFunctionCoefficient S_e(dim, [](const Vector& x, DenseMatrix& S) {
    S.SetSize(x.Size());
    S = 0.0;
    const double p = Pressure(x);
    for (int i = 0; i < x.Size(); i++) {
      S(i, i) = -p;
    }
  });
  ReferentialElasticRheology r_rheology(dim, C_bare, S_e, phi_e);

  Vector bb_min, bb_max;
  parent2.GetBoundingBox(bb_min, bb_max);
  const double r_out = bb_max.Normlinf();

  LinearQuasiStaticReferentialSelfGravitatingProblem referential(
      &fes_u2, &fes_zeta, r_rheology, rho, kG, kDtNDegree);
  auto E = NewRadialVacuumExtension(fes_u2, fes_buffer, 1.0, r_out);
  referential.SetPrescribedVacuumExtension(fes_buffer, *E);
  FunctionCoefficient sigma2(SurfaceLoad);
  referential.SetSurfaceLoad(sigma2, surface);
  referential.SetRelTol(1e-11);
  referential.AssembleForce(0.0);
  if (!referential.Solve()) {
    if (Root()) {
      std::cout << "referential solve failed\n";
    }
    return 1;
  }

  // --- The comparison. ------------------------------------------------------
  std::cout << std::setprecision(3);

  // Rigid null pairs: near-null under the tapered extension, which does
  // not preserve the translations.
  const auto rigid = referential.RigidPairResiduals();
  if (Root()) {
    std::cout << "\nrigid pair residuals (translations then rotations;\n  near-null under the tapered extension, decreasing with order):\n  ";
    for (const auto r : rigid) {
      std::cout << r << "  ";
    }
    std::cout << "\n";
  }

  // Displacement: same space, same rigid gauge at phi_e = id. Comparison
  // copies carry the build's field type so their norms are global.
  {
    auto d = MakeField(fes_u2);
    *d = referential.Displacement();
    *d -= eulerian.Displacement();
    const double rel = L2Norm(*d) / L2Norm(eulerian.Displacement());
    if (Root()) {
      std::cout << "|u_ref - u_eul| / |u_eul|                = " << rel
                << "\n";
    }
  }
  // Potential through the change of variables zeta1 = phi1 + u . grad Phi0.
  {
    VectorGridFunctionCoefficient u_c(&eulerian.Displacement());
    InnerProductCoefficient advect(u_c, eulerian.BackgroundGravity());
    GridFunctionCoefficient phi1(&eulerian.PotentialOnBody());
    SumCoefficient zeta_expected(phi1, advect);
    auto d = MakeField(referential.PotentialSpaceOnBody());
    *d = referential.PotentialOnBody();
    auto z = MakeField(referential.PotentialSpaceOnBody());
    z->ProjectCoefficient(zeta_expected);
    *d -= *z;
    if (dim == 2) {
      // Both potentials are constant-gauged: remove the global mean.
      *d -= GlobalSum(d->Sum()) / GlobalSum(d->Size());
    }
    const double rel =
        L2Norm(*d) /
        std::max(1e-30, L2Norm(referential.PotentialOnBody()));
    if (Root()) {
      std::cout << "|zeta1 - (phi1 + u.grad Phi0)| / |zeta1| = " << rel
                << "\n";
    }
  }
  if (Root()) {
    std::cout << "(both at the level of the two discretisations: rerun with "
                 "-o 3 to watch them fall)\n";
    std::cout << "iterations: Eulerian " << eulerian.LastOuterIterations()
              << ", referential " << referential.LastOuterIterations()
              << "\n";
  }

  if (visualization) {
    Show(body, eulerian.Displacement(), "Displacement (Eulerian)");
    Show(body2, referential.Displacement(), "Displacement (referential)");
    Show(parent2, referential.Potential(),
         "Referential potential perturbation");
  }

  return 0;
}
