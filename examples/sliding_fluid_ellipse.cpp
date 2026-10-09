// ============================================================================
// sliding_fluid_ellipse.cpp
//
// The sliding fluid-solid interface (a tangential displacement
// discontinuity) on an ELLIPTICAL body with a fluid core. Purely elastic,
// with no gravity and no pre-stress: a demonstration of the discretisation
// and of the constraint enforcement, not a physical Earth model.
// Companion to gauged_fluid_cavity.cpp, which welds the interface (one
// continuous displacement space). The method is in doc/slip_interface.tex,
// "Discretisation of the slipping interface" and "Constraint enforcement";
// when the weld is admissible is discussed in doc/gauged_fluid.md,
// "Tangential slip and the welded space".
//
// Two displacement fields live on the solid and fluid SubMeshes of one
// parent; the pairing J = Pi_s^T Pi_f identifies the two interface traces
// nodally, and continuity of the NORMAL component alone is enforced by a
// penalty on the normal jump,
//
//     theta [ B, -B J; -J^T B, J^T B J ],   B = normal-normal form,
//
// driven to zero by augmented-Lagrangian iterations at moderate theta,
// interleaved with the Tikhonov refinement of the fluid's gauge penalty.
// The tangential jump stays free: the slip.
//
// Why the ellipse: tangential slip is not gauge in general. A slip can be
// removed by relabelling the fluid only where it extends into the fluid
// as a relabelling field; on a circular interface the rigid rotation of
// the fluid is such a field, and under a uniform load the response is
// conformal with no slip at all. On an interface that is not a level
// surface, such as the ellipse, the slip generally cannot be removed by
// relabelling, and the welded space would suppress part of the response:
//
//   1. even the UNIFORM radial load drives a tangential slip (run with
//      -e 0 and again with the default to see the tangential jump appear
//      at O(ellipticity));
//   2. the frictionless interface no longer decouples the rotations:
//      a circular interface transmits no torque, so shell and core rotate
//      independently (with -e 0 the two separate rotations replace the
//      common one in the projected null space); an elliptical interface
//      transmits torque through the normal forces alone, and the relative
//      rotation becomes a stiff physical mode.
//
// The geometry is made by deforming the canned two-layer disc with the
// area-preserving map (x, y) -> (a x, y / a), a = 1 + e: the interface
// and surface become concentric ellipses, and every piece of machinery
// (pairing, boundary normals, loads) simply acts on the deformed mesh.
//
// Printed diagnostics: AL contraction of the normal jump, the final
// normal and tangential jumps against the interface trace, and the fluid
// pressure spread (p = -kappa div u must still be uniform without
// gravity - the gauge-invariant observable of the fluid, ellipse or not).
//
// One source serves the serial and the parallel build: in parallel the
// pairing runs on true dofs (NewSubMeshPairingTrueDofMatrix, hypre
// products for the penalty blocks) and the diagonal blocks are
// preconditioned with AMG instead of Gauss-Seidel.
//
// Options (defaults in brackets):
//   -m      mesh file [../data/elastogravity_two_layer_2d.msh]: the
//           2-D two-layer disc (attribute 1 fluid, attribute 2 solid).
//   -o      finite element order [2].
//   -e      ellipticity [0.2]: semi-axes a = 1 + e, b = 1/a; 0
//           recovers the disc (no slip, independent rotations).
//   -eps    gauge penalty factor epsilon on the fluid shear [1e-2].
//   -theta  normal-jump penalty parameter [1e2]: the AL iterations
//           drive the jump to zero at this moderate value.
//   -nal    augmented-Lagrangian iterations [8].
//   -P0     uniform radial traction amplitude [0.01].
//   -P2     degree-2 traction pattern amplitude [0]; even with 0 the
//           ellipse slips at O(e).
//   -vis / -no-vis   GLVis windows (solid and fluid displacement,
//           fluid pressure) [on].
//
// Sample runs (with mpiexec -np N in front in a parallel build):
//    ./sliding_fluid_ellipse
//    ./sliding_fluid_ellipse -e 0
//    ./sliding_fluid_ellipse -e 0.3 -P2 0.01
//    ./sliding_fluid_ellipse -theta 1e3 -nal 12
// ============================================================================

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "AdGIA.hpp"

using namespace mfem;
using namespace AdGIA;

namespace {

#ifdef MFEM_USE_MPI
using MeshType = ParMesh;
using SubMeshType = ParSubMesh;
using SpaceType = ParFiniteElementSpace;
using FieldType = ParGridFunction;
using FormType = ParBilinearForm;
bool Root() { return Mpi::Root(); }
double Dot(const Vector& a, const Vector& b) {
  return InnerProduct(MPI_COMM_WORLD, a, b);
}
double GlobalSum(double v) {
  double g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return g;
}
double GlobalMax(double v) {
  double g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return g;
}
#else
using MeshType = Mesh;
using SubMeshType = SubMesh;
using SpaceType = FiniteElementSpace;
using FieldType = GridFunction;
using FormType = BilinearForm;
bool Root() { return true; }
double Dot(const Vector& a, const Vector& b) { return InnerProduct(a, b); }
double GlobalSum(double v) { return v; }
double GlobalMax(double v) { return v; }
#endif

constexpr double kKappaSolid = 2.0;
constexpr double kMuSolid = 1.0;
constexpr double kKappaFluid = 1.0;
constexpr double kRCmb = 3483.0 / 6371.0;
constexpr double kPi = std::numbers::pi;

double P0 = 0.01;  // uniform radial traction amplitude
double P2 = 0.0;   // degree-2 pattern
double A = 1.2;    // ellipse semi-axis a (b = 1 / a)

// Radial surface traction f = -P(x) x/|x| (torque-free about the origin
// for any pattern, so the common rotation stays a null mode).
void RadialTraction(const Vector& x, Vector& f) {
  const double r = x.Norml2();
  const double c = x[1] / r;
  f = x;
  f *= -(P0 + P2 * (2.0 * c * c - 1.0)) / r;
}

// Boundary attributes whose elements sit at scaled radius
// s = sqrt((x/a)^2 + (y a)^2) in (s_min, s_max): the level sets of the
// undeformed radius, which the map carries onto concentric ellipses.
Array<int> ScaledRadialBdrMarker(Mesh& mesh, double a, double s_min,
                                 double s_max) {
  Array<int> marker(mesh.bdr_attributes.Max());
  marker = 0;
  for (int i = 0; i < mesh.GetNBE(); i++) {
    auto* tr = mesh.GetBdrElementTransformation(i);
    Vector c(mesh.Dimension());
    tr->Transform(Geometries.GetCenter(mesh.GetBdrElementGeometry(i)), c);
    const double sx = c[0] / a, sy = c[1] * a;
    const double s = std::sqrt(sx * sx + sy * sy);
    if (s > s_min && s < s_max) {
      marker[mesh.GetBdrAttribute(i) - 1] = 1;
    }
  }
#ifdef MFEM_USE_MPI
  // A rank may own no elements of the marked surface: unify the marker.
  Array<int> global(marker.Size());
  MPI_Allreduce(marker.GetData(), global.GetData(), marker.Size(), MPI_INT,
                MPI_MAX, MPI_COMM_WORLD);
  return global;
#else
  return marker;
#endif
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

  const char* mesh_file = "../data/elastogravity_two_layer_2d.msh";
  int order = 2;
  double ellipticity = 0.2;
  double eps = 1.0e-2;
  double theta = 1.0e2;
  int n_al = 8;
  bool visualization = true;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use (2-D).");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&ellipticity, "-e", "--ellipticity",
                 "Semi-axis a = 1 + e, b = 1/a (0 recovers the disc).");
  args.AddOption(&eps, "-eps", "--epsilon", "Gauge penalty factor epsilon.");
  args.AddOption(&theta, "-theta", "--theta",
                 "Normal-jump penalty parameter.");
  args.AddOption(&n_al, "-nal", "--al-iterations",
                 "Augmented-Lagrangian iterations.");
  args.AddOption(&P0, "-P0", "--traction", "Uniform radial traction.");
  args.AddOption(&P2, "-P2", "--traction-degree2",
                 "Degree-2 traction pattern.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "GLVis visualisation (solid and fluid displacement, fluid "
                 "pressure).");
  args.Parse();
  if (!args.Good()) {
    if (Root()) {
      args.PrintUsage(std::cout);
    }
    return 1;
  }
  A = 1.0 + ellipticity;
  const bool circular = ellipticity == 0.0;

  // The elliptical body: deform the whole canned mesh by the
  // area-preserving map (x, y) -> (a x, y/a), then partition.
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

  // Two independent displacement spaces on the solid and fluid SubMeshes
  // (shadow spaces of one parent space, as the dof injections require),
  // and the interface pairing J on (true) dofs.
  Array<int> fluid_attr({1}), solid_attr({2});
  auto solid = SubMeshType::CreateFromDomain(mesh, solid_attr);
  auto fluid = SubMeshType::CreateFromDomain(mesh, fluid_attr);
  H1_FECollection fec(order, dim);
  SpaceType fes_parent(&mesh, &fec, dim);
  auto fes_s = SubMeshDofInjection::MakeShadowSpace(fes_parent, solid);
  auto fes_f = SubMeshDofInjection::MakeShadowSpace(fes_parent, fluid);
  SubMeshDofInjection inj_s(*fes_s, fes_parent), inj_f(*fes_f, fes_parent);
  const int ns = fes_s->GetTrueVSize(), nf = fes_f->GetTrueVSize();

  // Stiffness: solid elasticity; fluid bulk modulus plus the eps
  // deviatoric gauge penalty (assembled separately too, for the gauge
  // source of the refinement).
  ConstantCoefficient kappa_s(kKappaSolid), mu_s(kMuSolid);
  ConstantCoefficient kappa_f(kKappaFluid), one(1.0);
  ConstantCoefficient eps_mu(eps * kKappaFluid);
  FormType a_s(fes_s.get());
  a_s.AddDomainIntegrator(new ElasticityIntegrator(kappa_s, 1.0, 0.0));
  a_s.AddDomainIntegrator(new ElasticityIntegrator(mu_s, -2.0 / dim, 1.0));
  a_s.Assemble();
  a_s.Finalize();
  FormType a_f(fes_f.get());
  a_f.AddDomainIntegrator(new ElasticityIntegrator(kappa_f, 1.0, 0.0));
  a_f.AddDomainIntegrator(new ElasticityIntegrator(eps_mu, -2.0 / dim, 1.0));
  a_f.Assemble();
  a_f.Finalize();
  FormType q_f(fes_f.get());
  q_f.AddDomainIntegrator(new ElasticityIntegrator(eps_mu, -2.0 / dim, 1.0));
  q_f.Assemble();
  q_f.Finalize();

  // Interface forms, assembled ONCE on the solid side: B for the normal
  // jump, M for the full jump (tangential = difference).
  auto cmb = ScaledRadialBdrMarker(solid, A, 0.9 * kRCmb, 1.1 * kRCmb);
  FormType b_form(fes_s.get());
  b_form.AddBoundaryIntegrator(new BoundaryNormalNormalIntegrator(one), cmb);
  b_form.Assemble();
  b_form.Finalize();
  FormType m_form(fes_s.get());
  m_form.AddBoundaryIntegrator(new VectorMassIntegrator(one), cmb);
  m_form.Assemble();
  m_form.Finalize();

  // The penalty blocks theta [B, -BJ; -J^T B, J^T B J] and the two
  // regularised diagonal blocks, on the build's matrix type.
#ifdef MFEM_USE_MPI
  auto J = NewSubMeshPairingTrueDofMatrix(inj_s, inj_f);
  std::unique_ptr<HypreParMatrix> Jt(J->Transpose());
  std::unique_ptr<HypreParMatrix> As(a_s.ParallelAssemble());
  std::unique_ptr<HypreParMatrix> Af(a_f.ParallelAssemble());
  std::unique_ptr<HypreParMatrix> Qf(q_f.ParallelAssemble());
  std::unique_ptr<HypreParMatrix> B(b_form.ParallelAssemble());
  std::unique_ptr<HypreParMatrix> M(m_form.ParallelAssemble());
  std::unique_ptr<HypreParMatrix> BJ(ParMult(B.get(), J.get(), true));
  std::unique_ptr<HypreParMatrix> JtB(ParMult(Jt.get(), B.get(), true));
  std::unique_ptr<HypreParMatrix> JtBJ(ParMult(JtB.get(), J.get(), true));
  std::unique_ptr<HypreParMatrix> A00(Add(1.0, *As, theta, *B));
  std::unique_ptr<HypreParMatrix> A11(Add(1.0, *Af, theta, *JtBJ));
  HypreBoomerAMG prec0(*A00), prec1(*A11);
  prec0.SetPrintLevel(0);
  prec1.SetPrintLevel(0);
  prec0.SetSystemsOptions(dim);
  prec1.SetSystemsOptions(dim);
#else
  auto J = NewSubMeshPairingMatrix(inj_s, inj_f);
  std::unique_ptr<SparseMatrix> Jt(Transpose(*J));
  const SparseMatrix* As = &a_s.SpMat();
  const SparseMatrix* Af = &a_f.SpMat();
  const SparseMatrix* Qf = &q_f.SpMat();
  const SparseMatrix* B = &b_form.SpMat();
  const SparseMatrix* M = &m_form.SpMat();
  std::unique_ptr<SparseMatrix> BJ(mfem::Mult(*B, *J));
  std::unique_ptr<SparseMatrix> JtB(mfem::Mult(*Jt, *B));
  std::unique_ptr<SparseMatrix> JtBJ(mfem::Mult(*JtB, *J));
  std::unique_ptr<SparseMatrix> A00(Add(1.0, *As, theta, *B));
  std::unique_ptr<SparseMatrix> A11(Add(1.0, *Af, theta, *JtBJ));
  GSSmoother prec0(*A00), prec1(*A11);
#endif

  Array<int> offsets({0, ns, nf});
  offsets.PartialSum();
  BlockOperator block_op(offsets);
  block_op.SetBlock(0, 0, A00.get());
  block_op.SetBlock(0, 1, BJ.get(), -theta);
  block_op.SetBlock(1, 0, JtB.get(), -theta);
  block_op.SetBlock(1, 1, A11.get());
  BlockDiagonalPreconditioner block_prec(offsets);
  block_prec.SetDiagonalBlock(0, &prec0);
  block_prec.SetDiagonalBlock(1, &prec1);

  // Null space: common translations and the common rotation always. The
  // INDEPENDENT rotations of shell and core are null only for a circular
  // interface (it transmits no torque); an elliptical interface locks
  // them through the normal forces, and the relative rotation is a
  // physical mode.
#ifdef MFEM_USE_MPI
  NullSpaceProjector P(MPI_COMM_WORLD);
#else
  NullSpaceProjector P;
#endif
  {
    BlockVector n(offsets);
    FieldType gs(fes_s.get()), gf(fes_f.get());
    Vector ts, tf;
    for (int c = 0; c < dim; c++) {
      Vector e(dim);
      e = 0.0;
      e[c] = 1.0;
      VectorConstantCoefficient t(e);
      gs.ProjectCoefficient(t);
      gf.ProjectCoefficient(t);
      gs.GetTrueDofs(ts);
      gf.GetTrueDofs(tf);
      n.GetBlock(0) = ts;
      n.GetBlock(1) = tf;
      P.Add(n);
    }
    RigidRotation rot(dim, 2);
    gs.ProjectCoefficient(rot);
    gf.ProjectCoefficient(rot);
    gs.GetTrueDofs(ts);
    gf.GetTrueDofs(tf);
    if (circular) {
      n.GetBlock(0) = ts;
      n.GetBlock(1) = 0.0;
      P.Add(n);
      n.GetBlock(0) = 0.0;
      n.GetBlock(1) = tf;
      P.Add(n);
    } else {
      n.GetBlock(0) = ts;
      n.GetBlock(1) = tf;
      P.Add(n);
    }
  }

  // The load: radial traction on the outer (elliptical) surface.
  auto surf = ScaledRadialBdrMarker(solid, A, 0.9, 1.1);
  VectorFunctionCoefficient traction(dim, RadialTraction);
  LinearForm lf(fes_s.get());
  lf.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(traction), surf);
  lf.Assemble();
  BlockVector F(offsets);
  {
    const Operator* Pr = fes_s->GetProlongationMatrix();
    if (Pr) {
      Pr->MultTranspose(lf, F.GetBlock(0));
    } else {
      F.GetBlock(0) = lf;
    }
  }
  F.GetBlock(1) = 0.0;

  // Solver on the projected regularised operator.
  ProjectedSolver prec_p(P);
  prec_p.SetSolver(block_prec);
  ProjectedOperator op_p(block_op, P);
#ifdef MFEM_USE_MPI
  CGSolver cg(MPI_COMM_WORLD);
#else
  CGSolver cg;
#endif
  cg.SetOperator(op_p);
  cg.SetPreconditioner(prec_p);
  cg.SetRelTol(1e-12);
  cg.SetAbsTol(0.0);
  cg.SetMaxIter(20000);
  cg.iterative_mode = true;
  ProjectedSolver solver(P);
  solver.SetSolver(cg);
  solver.iterative_mode = true;

  // The augmented-Lagrangian loop, interleaved with the gauge source:
  //   R U_{k+1} = f - w_k + eps Q u_f,k ;  w_{k+1} = w_k + theta C U_{k+1},
  // with R the regularised block operator and C = [B, -BJ; -J^T B, J^T B J]
  // the unscaled penalty matrix. At the fixed point the normal jump
  // vanishes at finite theta and the eps shear drops out of the
  // observables (doc/slip_interface.tex, "Penalty and augmented Lagrangian,
  // interleaved with the gauge refinement").
  BlockVector U(offsets), rhs(offsets), w(offsets), t(offsets);
  U = 0.0;
  w = 0.0;
  auto normal_jump2 = [&](const BlockVector& X) {
    Vector js(ns);
    J->Mult(X.GetBlock(1), js);
    js -= X.GetBlock(0);
    Vector Bj(ns);
    B->Mult(js, Bj);
    return Dot(js, Bj);
  };
  std::vector<double> jumps;
  for (int k = 0; k < n_al; k++) {
    rhs = F;
    rhs -= w;
    Vector qf(nf);
    Qf->Mult(U.GetBlock(1), qf);
    rhs.GetBlock(1) += qf;
    solver.Mult(rhs, U);
    if (!cg.GetConverged()) {
      if (Root()) {
        std::cout << "inner CG did not converge at AL iteration " << k
                  << "\n";
      }
      return 1;
    }
    Vector js(ns);
    J->Mult(U.GetBlock(1), js);
    js -= U.GetBlock(0);
    Vector Bj(ns);
    B->Mult(js, Bj);
    t = 0.0;
    t.GetBlock(0) -= Bj;
    Jt->AddMult(Bj, t.GetBlock(1));
    t *= theta;
    w += t;
    jumps.push_back(std::sqrt(normal_jump2(U)));
  }

  // Diagnostics: jump norms against the interface trace.
  FieldType u_s(fes_s.get()), u_f(fes_f.get());
  u_s.SetFromTrueDofs(U.GetBlock(0));
  u_f.SetFromTrueDofs(U.GetBlock(1));
  {
    Vector Bu(ns);
    B->Mult(U.GetBlock(0), Bu);
    const double trace = std::sqrt(Dot(U.GetBlock(0), Bu));
    Vector js(ns);
    J->Mult(U.GetBlock(1), js);
    js -= U.GetBlock(0);
    Vector Mj(ns);
    M->Mult(js, Mj);
    const double full = std::sqrt(Dot(js, Mj));
    const double normal = jumps.back();
    const double tangential =
        std::sqrt(std::max(0.0, full * full - normal * normal));
    if (Root()) {
      std::cout << "ellipse a = " << A << ", b = " << 1.0 / A << " (e = "
                << ellipticity << "), theta = " << theta << ", " << n_al
                << " AL iterations, eps = " << eps << "\n"
                << "normal-jump decay: ";
      for (auto j : jumps) {
        std::cout << std::setprecision(3) << j << "  ";
      }
      std::cout << "\nnormal jump / normal trace:     "
                << std::setprecision(3) << normal / trace
                << "  (constrained away)\n"
                << "tangential jump / normal trace: " << tangential / trace
                << "  (the slip; O(e) under the uniform load, 0 on the "
                   "disc)\n"
                << "null modes projected: " << P.Size()
                << (circular ? "  (disc: independent rotations)\n"
                             : "  (ellipse: rotations locked by the normal "
                               "forces)\n");
    }
  }

  // The fluid pressure p = -kappa div u must be uniform without gravity,
  // ellipse or not: its spread is discretisation error, and it is the
  // gauge-invariant observable of the fluid.
  {
    DivergenceGridFunctionCoefficient div_u(&u_f);
    double p_int = 0.0, vol = 0.0;
    double p_max = -std::numeric_limits<double>::infinity();
    double p_min = std::numeric_limits<double>::infinity();
    for (int i = 0; i < fluid.GetNE(); i++) {
      auto* T = fluid.GetElementTransformation(i);
      const auto& ir = IntRules.Get(fluid.GetElementGeometry(i), 2 * order);
      for (int q = 0; q < ir.GetNPoints(); q++) {
        const auto& ip = ir.IntPoint(q);
        T->SetIntPoint(&ip);
        const double wq = ip.weight * T->Weight();
        const double p = -kKappaFluid * div_u.Eval(*T, ip);
        p_int += wq * p;
        vol += wq;
        p_max = std::max(p_max, p);
        p_min = std::min(p_min, p);
      }
    }
    p_int = GlobalSum(p_int);
    vol = GlobalSum(vol);
    p_max = GlobalMax(p_max);
    p_min = -GlobalMax(-p_min);
    const double p_mean = p_int / vol;
    if (Root()) {
      std::cout << "fluid pressure: mean " << std::setprecision(6) << p_mean
                << ", relative spread "
                << (p_max - p_min) / std::abs(p_mean)
                << " (uniform in the continuum)\n";
    }
  }

  if (visualization) {
    Show(solid, u_s, "Solid displacement");
    Show(fluid, u_f, "Fluid displacement (gauge; its trace slips)");
    L2_FECollection pfec(order - 1, dim);
    SpaceType pfes(&fluid, &pfec);
    FieldType p_gf(&pfes);
    DivergenceGridFunctionCoefficient div_u(&u_f);
    ProductCoefficient minus_kappa_div(-kKappaFluid, div_u);
    p_gf.ProjectCoefficient(minus_kappa_div);
    Show(fluid, p_gf, "Fluid pressure");
  }

  return 0;
}
