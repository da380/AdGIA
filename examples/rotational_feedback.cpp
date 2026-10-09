// ============================================================================
// rotational_feedback.cpp
//
// Rotational feedback on a loaded self-gravitating body
// (RotationalFeedback, rotation.hpp; Yu, Al-Attar, Syvret & Lloyd 2025,
// §2.3–2.4 and Appendix A3; doc/planning/sea_level_plan.md, WP2).
//
// Surface loading perturbs the inertia tensor; conservation of angular
// momentum then perturbs the rotation vector, and the centrifugal
// potential of that perturbation feeds back on the deformation. The
// coupled system is the ordinary mixed self-gravitating problem plus a
// small symmetric border — the spin rate in 2-D, spin plus polar wander
// in 3-D — solved exactly by block elimination over the class's own
// tidal machinery: the unit centrifugal responses are computed once and
// reused by linearity, so a feedback solve costs one base solve, a tiny
// dense solve and one composed final solve.
//
// The example solves a degree-2 surface mass load (with an order-1 part
// in 3-D, so that polar wander is activated) without and with the
// feedback, and prints the angular-velocity perturbation, the relative
// change it induces in the displacement, and the residual of the
// angular-momentum row re-evaluated at the composed state (the
// consistency diagnostic; solver-tolerance small when all is well).
//
// The equilibrium principal moments are DATA, not derived from the
// model: the traditional rotational theory assumes C1 <= C2 < C3, which
// a spherically symmetric model violates (its own moments are equal) —
// real calculations use observed values (paper §2.4).
//
// One source serves the serial and the parallel build; the only genuine
// difference is the mesh partitioning.
//
// Options (defaults in brackets):
//   -m       mesh file [../data/elastogravity_2d.msh]; the 3-D ball is
//            ../data/coupled_poisson.msh.
//   -o       finite element order [2].
//   -Omega   equilibrium rotation rate about e3 (out of plane in 2-D)
//            [0.1]; the feedback scales linearly with it.
//   -C1 -C2 -C3   principal moments [1.6, 1.7, 2.0]; 2-D uses -C3 only.
//   -vis / -no-vis   GLVis windows on or off [on].
//
// Sample runs (with mpiexec -np N in front in a parallel build):
//    ./rotational_feedback
//    ./rotational_feedback -Omega 0.3
//    ./rotational_feedback -m ../data/coupled_poisson.msh -o 1
// ============================================================================

#include <iostream>
#include <memory>

#include "mfem.hpp"
#include "AdGIA.hpp"
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

// Degree-2 surface mass load, with an order-1 part in 3-D so that the
// equatorial (polar-wander) components of the feedback are driven too.
double SurfaceLoad(const Vector& x) {
  const double r = x.Norml2();
  if (r == 0.0) {
    return 0.0;
  }
  const double c = (x.Size() == 2 ? x[1] : x[2]) / r;
  double s = 0.02 * (1.0 + 3.0 * c * c);
  if (x.Size() == 3) {
    s += 0.01 * x[0] * x[2] / (r * r);
  }
  return s;
}

double L2NormOf(const GridFunction& u) {
  Vector zero(u.FESpace()->GetVDim());
  zero = 0.0;
  VectorConstantCoefficient z(zero);
  return u.ComputeL2Error(z);
}

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif

  const char* mesh_file = "../data/elastogravity_2d.msh";
  int order = 2;
  double Omega = 0.1;
  double C1 = 1.6, C2 = 1.7, C3 = 2.0;
  bool visualization = true;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
                 "Body-in-buffer mesh (2-D disc or 3-D ball).");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&Omega, "-Omega", "--rotation-rate",
                 "Equilibrium rotation rate about e3.");
  args.AddOption(&C1, "-C1", "--moment-1", "Principal moment C1 (3-D).");
  args.AddOption(&C2, "-C2", "--moment-2", "Principal moment C2 (3-D).");
  args.AddOption(&C3, "-C3", "--moment-3", "Principal moment C3.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization", "GLVis windows.");
  args.Parse();
  if (!args.Good()) {
    if (IsRoot()) {
      args.PrintUsage(std::cout);
    }
    return 1;
  }

  // The mesh: body attribute 1 inside a buffer, the body surface the
  // SubMesh's only boundary.
  Mesh smesh(mesh_file, 1, 1);
  const int dim = smesh.Dimension();
#ifdef MFEM_USE_MPI
  ParMesh parent(MPI_COMM_WORLD, smesh);
  smesh.Clear();
#else
  Mesh& parent = smesh;
#endif
  Array<int> body_marker(parent.attributes.Max());
  body_marker = 0;
  body_marker[0] = 1;
#ifdef MFEM_USE_MPI
  ParSubMesh body(ParSubMesh::CreateFromDomain(parent, body_marker));
#else
  SubMesh body(SubMesh::CreateFromDomain(parent, body_marker));
#endif
  H1_FECollection fec(order, dim);
#ifdef MFEM_USE_MPI
  ParFiniteElementSpace fes_u(&body, &fec, dim), fes_phi(&parent, &fec);
#else
  FiniteElementSpace fes_u(&body, &fec, dim), fes_phi(&parent, &fec);
#endif
  Array<int> surface(body.bdr_attributes.Max());
  surface = 0;
  surface[body.bdr_attributes.Max() - 1] = 1;

  ConstantCoefficient kappa(kKappa), mu(kMu), rho(kRho);
  IsotropicElasticRheology rheology(dim, kappa, mu);
  FunctionCoefficient sigma(SurfaceLoad);

  auto make_problem = [&] {
    auto p = std::make_unique<LinearQuasiStaticMixedSelfGravitatingProblem>(
        &fes_u, &fes_phi, rheology, rho, kG, kDtNDegree);
    p->SetSurfaceLoad(sigma, surface);
    p->SetRelTol(1e-11);
    return p;
  };

  // Without the feedback.
  auto p0 = make_problem();
  p0->AssembleForce(0.0);
  if (!p0->Solve()) {
    if (IsRoot()) {
      std::cout << "base solve failed\n";
    }
    return 1;
  }
  // The comparison copy must carry the build's field type: a plain
  // GridFunction copy of a parallel field would compute rank-local
  // norms below.
#ifdef MFEM_USE_MPI
  ParGridFunction u0(static_cast<const ParGridFunction&>(p0->Displacement()));
#else
  GridFunction u0(p0->Displacement());
#endif

  // With the feedback: the border solved by block elimination.
  Vector moments(dim == 2 ? 1 : 3);
  if (dim == 2) {
    moments[0] = C3;
  } else {
    moments[0] = C1;
    moments[1] = C2;
    moments[2] = C3;
  }
  auto p1 = make_problem();
  RotationalFeedback rot(*p1, Omega, moments);
  if (!rot.Solve(0.0, sigma, surface)) {
    if (IsRoot()) {
      std::cout << "feedback solve failed\n";
    }
    return 1;
  }

  const Vector& omega = rot.AngularVelocityPerturbation();
#ifdef MFEM_USE_MPI
  ParGridFunction du(static_cast<const ParGridFunction&>(p1->Displacement()));
#else
  GridFunction du(p1->Displacement());
#endif
  du -= u0;
  const double rel = L2NormOf(du) / L2NormOf(u0);
  const double residual = rot.ConsistencyResidual(sigma, surface);
  if (IsRoot()) {
    std::cout << "\nrotational feedback (Omega = " << Omega << ")\n";
    std::cout << "  omega =";
    for (int k = 0; k < omega.Size(); k++) {
      std::cout << " " << omega[k];
    }
    std::cout << (dim == 2 ? "   (spin rate)\n" : "   (wander_1 wander_2 spin)\n");
    std::cout << "  displacement change |du|/|u|   " << rel << "\n";
    std::cout << "  angular-momentum row residual  " << residual
              << "   (solver-tolerance small)\n";
  }

  if (visualization) {
    GLVisWindow wu("displacement (no rotation)", DefaultKeys(dim));
    GLVisWindow wr("displacement (with feedback)", DefaultKeys(dim));
    GLVisWindow wd("feedback change", DefaultKeys(dim));
    wu.Send(body, u0);
    wr.Send(body, const_cast<GridFunction&>(p1->Displacement()));
    wd.Send(body, du);
  }
  return 0;
}
