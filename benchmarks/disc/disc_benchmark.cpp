// ============================================================================
// disc_benchmark.cpp
//
// Rung 0 of the no-gravity disc family (doc/planning/no_gravity_analytics.md):
// a fluid disc (bulk modulus only) inside a solid annulus, no gravity, no
// pre-stress, loaded degree by degree on the outer circle. The exact
// solution is closed form (disc_reference.py): for l >= 1 the interface is
// exactly traction-free and welded == slip identically, so the run is a
// machinery identity test — the broken space, the pairing, the normal
// constraint and the null-space handling against exact values, with the
// welded and slipping answers also compared to each other.
//
// Loads, with t = 1 and theta the polar angle:
//   l = 0:   sigma.rhat = -t rhat
//   l >= 2:  sigma.rhat = -t cos(l theta) rhat                (shear-free)
//   l = 1:   T_r = -t cos(theta), T_theta = -t sin(theta): the
//            self-equilibrated combination, Cartesian -t (cos 2theta,
//            sin 2theta); a pure cos(theta) normal load carries net force.
//
// Responses: the amplitudes U, V of u_r = U cos(l theta),
// u_theta = V sin(l theta) on the surface and on the interface (solid
// side). They are extracted basis-convention-free: the raw harmonic
// coefficient of the field is divided by the raw coefficient of the
// *known* pattern (rhat cos(l theta), resp. thetahat sin(l theta))
// analysed on the same circle, so every normalisation cancels.
//
// Methods:
//   welded  one continuous displacement space; the fluid carries kappa_f
//           plus the eps deviatoric gauge penalty, refined by Tikhonov
//           iterations (the observables are gauge-invariant).
//   slip    broken spaces on the solid and fluid SubMeshes, the nodal
//           pairing J, the normal-jump penalty theta driven by
//           augmented-Lagrangian iterations interleaved with the gauge
//           source (examples/sliding_fluid_ellipse.cpp is the template).
//
// One source serves the serial and the parallel build. Results go to a
// JSON file for benchmarks/disc/check.py, which compares both methods
// against disc_reference.py, absolute and (later rungs) in the interface
// derivative.
//
// Sample runs (with mpiexec -np N in front in a parallel build):
//    ./disc_benchmark -o 2 -ref 1 -out results.json
//    ./disc_benchmark -lmax 6 -nal 12 -theta 1e3
// ============================================================================

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
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
using LinearFormType = ParLinearForm;
bool Root() { return Mpi::Root(); }
double Dot(const Vector& a, const Vector& b) {
  return InnerProduct(MPI_COMM_WORLD, a, b);
}
#else
using MeshType = Mesh;
using SubMeshType = SubMesh;
using SpaceType = FiniteElementSpace;
using FieldType = GridFunction;
using FormType = BilinearForm;
using LinearFormType = LinearForm;
bool Root() { return true; }
double Dot(const Vector& a, const Vector& b) { return InnerProduct(a, b); }
#endif

constexpr double kRCmb = 3483.0 / 6371.0;

int L = 0;  // degree of the current load

// The degree-l boundary traction (see the header comment).
void Traction(const Vector& x, Vector& f) {
  const double theta = std::atan2(x[1], x[0]);
  f.SetSize(2);
  if (L == 1) {
    f[0] = -std::cos(2.0 * theta);
    f[1] = -std::sin(2.0 * theta);
    return;
  }
  const double c = std::cos(L * theta);
  const double r = x.Norml2();
  f[0] = -c * x[0] / r;
  f[1] = -c * x[1] / r;
}

// Known patterns on a circle, for the normalisation-free extraction.
void RadialPattern(const Vector& x, Vector& f) {
  const double theta = std::atan2(x[1], x[0]);
  const double r = x.Norml2();
  const double c = std::cos(L * theta);
  f.SetSize(2);
  f[0] = c * x[0] / r;
  f[1] = c * x[1] / r;
}
void TangentialPattern(const Vector& x, Vector& f) {
  const double theta = std::atan2(x[1], x[0]);
  const double r = x.Norml2();
  const double s = std::sin(L * theta);
  f.SetSize(2);
  f[0] = -s * x[1] / r;  // s * thetahat
  f[1] = s * x[0] / r;
}

// Boundary attributes whose element centres sit at radius in
// (r_min, r_max) — on the given (sub)mesh, rank-unified.
Array<int> RadialBdrMarker(Mesh& mesh, double r_min, double r_max) {
  Array<int> marker(mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max()
                                               : 0);
  marker = 0;
  for (int i = 0; i < mesh.GetNBE(); i++) {
    auto* tr = mesh.GetBdrElementTransformation(i);
    Vector c(mesh.Dimension());
    tr->Transform(Geometries.GetCenter(mesh.GetBdrElementGeometry(i)), c);
    const double r = c.Norml2();
    if (r > r_min && r < r_max) {
      marker[mesh.GetBdrAttribute(i) - 1] = 1;
    }
  }
#ifdef MFEM_USE_MPI
  Array<int> global(marker.Size());
  MPI_Allreduce(marker.GetData(), global.GetData(), marker.Size(), MPI_INT,
                MPI_MAX, MPI_COMM_WORLD);
  return global;
#else
  return marker;
#endif
}

// U (and V, l >= 1) of the field on the circle marked in `marker`:
// harmonic coefficients of the field divided by those of the known
// patterns, so the basis normalisation cancels.
struct Amplitudes {
  double U = 0.0, V = 0.0;
};
Amplitudes ResponseAmplitudes(SpaceType& fes, const FieldType& u,
                              const Array<int>& marker, int l) {
  Amplitudes out;
  const int lmax = std::max(l, 1);
  BoundaryHarmonicCoefficients radial(
      fes, marker, lmax, BoundaryHarmonicCoefficients::Component::Radial);
  Vector cu, cp;
  radial.Coefficients(u, cu);
  VectorFunctionCoefficient rpat(2, RadialPattern);
  radial.Coefficients(rpat, cp);
  const int i_cos = radial.Basis().Index(l, l);
  out.U = cu[i_cos] / cp[i_cos];
  if (l >= 1) {
    BoundaryHarmonicCoefficients tang(
        fes, marker, lmax,
        BoundaryHarmonicCoefficients::Component::Tangential);
    Vector tu, tp;
    tang.Coefficients(u, tu);
    VectorFunctionCoefficient tpat(2, TangentialPattern);
    tang.Coefficients(tpat, tp);
    // In 2-D the tangential basis member of the COS scalar index is
    // proportional to sin(l theta) thetahat: the sin thetahat pattern
    // (and u_theta = V sin(l theta)) lives at Index(l, +l).
    out.V = tu[i_cos] / tp[i_cos];
  }
  return out;
}

struct Result {
  int l = 0;
  Amplitudes surface, interface_;
  double p1 = 0.0;          // fluid pressure (meaningful at l = 0)
  double normal_jump = 0.0;  // final sqrt(j^T B j) (slip only)
  int iterations = 0;        // total inner CG iterations
};

void WriteJson(const std::string& path, const std::string& method, int order,
               int refinements, double kappa_s, double mu_s, double kappa_f,
               const std::vector<Result>& results) {
  if (!Root()) {
    return;
  }
  std::ofstream f(path);
  f << std::setprecision(16);
  f << "{\n  \"method\": \"" << method << "\",\n"
    << "  \"order\": " << order << ",\n"
    << "  \"refinements\": " << refinements << ",\n"
    << "  \"kappa_s\": " << kappa_s << ",\n  \"mu_s\": " << mu_s << ",\n"
    << "  \"kappa_f\": " << kappa_f << ",\n"
    << "  \"r_cmb\": " << kRCmb << ",\n  \"degrees\": [\n";
  for (size_t i = 0; i < results.size(); i++) {
    const auto& r = results[i];
    f << "    {\"l\": " << r.l << ", \"ur_a\": " << r.surface.U
      << ", \"ut_a\": " << r.surface.V << ", \"ur_c\": " << r.interface_.U
      << ", \"ut_c\": " << r.interface_.V << ", \"p1\": " << r.p1
      << ", \"normal_jump\": " << r.normal_jump
      << ", \"iterations\": " << r.iterations << "}"
      << (i + 1 < results.size() ? ",\n" : "\n");
  }
  f << "  ]\n}\n";
}

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
#endif

  const char* mesh_file = "../../data/elastogravity_two_layer_2d.msh";
  const char* method = "slip";
  const char* out_file = "results.json";
  int order = 2;
  int refinements = 0;
  int lmin = 0, lmax = 4;
  double kappa_s = 2.0, mu_s = 1.0, kappa_f = 1.5;
  double eps = 1.0e-2, theta = 1.0e2, rel_tol = 1.0e-12;
  int n_al = 10;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "2-D two-layer disc mesh.");
  args.AddOption(&method, "-method", "--method", "welded or slip.");
  args.AddOption(&out_file, "-out", "--output", "Results JSON file.");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&refinements, "-ref", "--refinements",
                 "Uniform refinements of the canned mesh.");
  args.AddOption(&lmin, "-lmin", "--lmin", "Lowest degree.");
  args.AddOption(&lmax, "-lmax", "--lmax", "Highest degree.");
  args.AddOption(&kappa_s, "-ks", "--kappa-solid", "Solid bulk modulus.");
  args.AddOption(&mu_s, "-ms", "--mu-solid", "Solid shear modulus.");
  args.AddOption(&kappa_f, "-kf", "--kappa-fluid", "Fluid bulk modulus.");
  args.AddOption(&eps, "-eps", "--epsilon", "Fluid gauge penalty factor.");
  args.AddOption(&theta, "-theta", "--theta", "Normal-jump penalty (slip).");
  args.AddOption(&n_al, "-nal", "--al-iterations",
                 "Augmented-Lagrangian / gauge refinement iterations.");
  args.AddOption(&rel_tol, "-rt", "--rel-tol", "Inner CG relative tolerance.");
  args.Parse();
  if (!args.Good()) {
    if (Root()) {
      args.PrintUsage(std::cout);
    }
    return 1;
  }
  const bool slip = std::string(method) == "slip";
  const int dim = 2;

  Mesh smesh(mesh_file, 1, 1);
  MFEM_VERIFY(smesh.Dimension() == 2, "a 2-D benchmark");
  for (int i = 0; i < refinements; i++) {
    smesh.UniformRefinement();
  }
#ifdef MFEM_USE_MPI
  MeshType mesh(MPI_COMM_WORLD, smesh);
  smesh.Clear();
#else
  MeshType& mesh = smesh;
#endif

  H1_FECollection fec(order, dim);
  ConstantCoefficient c_kappa_s(kappa_s), c_mu_s(mu_s);
  ConstantCoefficient c_kappa_f(kappa_f), one(1.0);
  ConstantCoefficient c_eps_mu(eps * kappa_f);

  std::vector<Result> results;

  if (!slip) {
    // ------- Welded: one continuous space on the body SubMesh -------
    // (the canned mesh carries a vacuum buffer, attribute 3, which has
    // no elasticity: it must stay outside the displacement space).
    Array<int> body_attr({1, 2});
    auto body = SubMeshType::CreateFromDomain(mesh, body_attr);
    SpaceType fes(&body, &fec, dim);
    const int n = fes.GetTrueVSize();
    Array<int> fluid_attr(body.attributes.Max()),
        solid_attr(body.attributes.Max());
    fluid_attr = 0;
    solid_attr = 0;
    fluid_attr[0] = 1;  // attribute 1: core
    solid_attr[1] = 1;  // attribute 2: mantle

    FormType a(&fes);
    a.AddDomainIntegrator(new ElasticityIntegrator(c_kappa_s, 1.0, 0.0),
                          solid_attr);
    a.AddDomainIntegrator(new ElasticityIntegrator(c_mu_s, -2.0 / dim, 1.0),
                          solid_attr);
    a.AddDomainIntegrator(new ElasticityIntegrator(c_kappa_f, 1.0, 0.0),
                          fluid_attr);
    a.AddDomainIntegrator(new ElasticityIntegrator(c_eps_mu, -2.0 / dim, 1.0),
                          fluid_attr);
    a.Assemble();
    a.Finalize();
    FormType q(&fes);
    q.AddDomainIntegrator(new ElasticityIntegrator(c_eps_mu, -2.0 / dim, 1.0),
                          fluid_attr);
    q.Assemble();
    q.Finalize();
#ifdef MFEM_USE_MPI
    std::unique_ptr<HypreParMatrix> A(a.ParallelAssemble());
    std::unique_ptr<HypreParMatrix> Q(q.ParallelAssemble());
    HypreBoomerAMG prec(*A);
    prec.SetPrintLevel(0);
    prec.SetSystemsOptions(dim);
    NullSpaceProjector P(MPI_COMM_WORLD);
    CGSolver cg(MPI_COMM_WORLD);
#else
    SparseMatrix* A = &a.SpMat();
    SparseMatrix* Q = &q.SpMat();
    GSSmoother prec(*A);
    NullSpaceProjector P;
    CGSolver cg;
#endif
    {
      FieldType g(&fes);
      Vector t;
      for (int c = 0; c < dim; c++) {
        Vector e(dim);
        e = 0.0;
        e[c] = 1.0;
        VectorConstantCoefficient tc(e);
        g.ProjectCoefficient(tc);
        g.GetTrueDofs(t);
        P.Add(t);
      }
      RigidRotation rot(dim, 2);
      g.ProjectCoefficient(rot);
      g.GetTrueDofs(t);
      P.Add(t);
    }
    ProjectedOperator op_p(*A, P);
    ProjectedSolver prec_p(P);
    prec_p.SetSolver(prec);
    cg.SetOperator(op_p);
    cg.SetPreconditioner(prec_p);
    cg.SetRelTol(rel_tol);
    cg.SetAbsTol(0.0);
    cg.SetMaxIter(50000);
    cg.iterative_mode = true;
    ProjectedSolver solver(P);
    solver.SetSolver(cg);
    solver.iterative_mode = true;

    auto surf = RadialBdrMarker(body, 0.9, 1.1);
    auto cmb = RadialBdrMarker(body, 0.9 * kRCmb, 1.1 * kRCmb);

    for (L = lmin; L <= lmax; L++) {
      VectorFunctionCoefficient traction(dim, Traction);
      LinearFormType lf(&fes);
      lf.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(traction),
                               surf);
      lf.Assemble();
      Vector F(n);
      {
        const Operator* Pr = fes.GetProlongationMatrix();
        if (Pr) {
          Pr->MultTranspose(lf, F);
        } else {
          F = lf;
        }
      }
      Vector U(n), rhs(n), qu(n);
      U = 0.0;
      int its = 0;
      for (int k = 0; k < n_al; k++) {  // gauge refinement
        rhs = F;
        Q->Mult(U, qu);
        rhs += qu;
        solver.Mult(rhs, U);
        MFEM_VERIFY(cg.GetConverged(), "welded CG did not converge");
        its += cg.GetNumIterations();
      }
      FieldType u(&fes);
      u.SetFromTrueDofs(U);
      Result r;
      r.l = L;
      r.iterations = its;
      r.surface = ResponseAmplitudes(fes, u, surf, L);
      r.interface_ = ResponseAmplitudes(fes, u, cmb, L);
      // p1 = -kappa_f dA/A with dA = 2 pi c U_0(c): the l = 0 interface
      // amplitude already in hand (zero for l >= 1 up to discretisation).
      r.p1 = L == 0 ? -2.0 * kappa_f * r.interface_.U / kRCmb : 0.0;
      results.push_back(r);
      if (Root()) {
        std::cout << "welded l=" << L << "  ur_a=" << r.surface.U
                  << "  ut_a=" << r.surface.V << "  its=" << its << "\n";
      }
    }
    WriteJson(out_file, method, order, refinements, kappa_s, mu_s, kappa_f,
              results);
    return 0;
  }

  // ---------------- Slip: broken spaces and the pairing. ----------------
  Array<int> fluid_attr({1}), solid_attr({2});
  auto solid = SubMeshType::CreateFromDomain(mesh, solid_attr);
  auto fluid = SubMeshType::CreateFromDomain(mesh, fluid_attr);
  SpaceType fes_parent(&mesh, &fec, dim);
  auto fes_s = SubMeshDofInjection::MakeShadowSpace(fes_parent, solid);
  auto fes_f = SubMeshDofInjection::MakeShadowSpace(fes_parent, fluid);
  SubMeshDofInjection inj_s(*fes_s, fes_parent), inj_f(*fes_f, fes_parent);
  const int ns = fes_s->GetTrueVSize(), nf = fes_f->GetTrueVSize();

  FormType a_s(fes_s.get());
  a_s.AddDomainIntegrator(new ElasticityIntegrator(c_kappa_s, 1.0, 0.0));
  a_s.AddDomainIntegrator(new ElasticityIntegrator(c_mu_s, -2.0 / dim, 1.0));
  a_s.Assemble();
  a_s.Finalize();
  FormType a_f(fes_f.get());
  a_f.AddDomainIntegrator(new ElasticityIntegrator(c_kappa_f, 1.0, 0.0));
  a_f.AddDomainIntegrator(new ElasticityIntegrator(c_eps_mu, -2.0 / dim, 1.0));
  a_f.Assemble();
  a_f.Finalize();
  FormType q_f(fes_f.get());
  q_f.AddDomainIntegrator(new ElasticityIntegrator(c_eps_mu, -2.0 / dim, 1.0));
  q_f.Assemble();
  q_f.Finalize();

  auto cmb = RadialBdrMarker(solid, 0.9 * kRCmb, 1.1 * kRCmb);
  FormType b_form(fes_s.get());
  b_form.AddBoundaryIntegrator(new BoundaryNormalNormalIntegrator(one), cmb);
  b_form.Assemble();
  b_form.Finalize();

#ifdef MFEM_USE_MPI
  auto J = NewSubMeshPairingTrueDofMatrix(inj_s, inj_f);
  std::unique_ptr<HypreParMatrix> Jt(J->Transpose());
  std::unique_ptr<HypreParMatrix> As(a_s.ParallelAssemble());
  std::unique_ptr<HypreParMatrix> Af(a_f.ParallelAssemble());
  std::unique_ptr<HypreParMatrix> Qf(q_f.ParallelAssemble());
  std::unique_ptr<HypreParMatrix> B(b_form.ParallelAssemble());
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
  NullSpaceProjector P(MPI_COMM_WORLD);
  CGSolver cg(MPI_COMM_WORLD);
#else
  auto J = NewSubMeshPairingMatrix(inj_s, inj_f);
  std::unique_ptr<SparseMatrix> Jt(Transpose(*J));
  const SparseMatrix* As = &a_s.SpMat();
  const SparseMatrix* Af = &a_f.SpMat();
  const SparseMatrix* Qf = &q_f.SpMat();
  const SparseMatrix* B = &b_form.SpMat();
  std::unique_ptr<SparseMatrix> BJ(mfem::Mult(*B, *J));
  std::unique_ptr<SparseMatrix> JtB(mfem::Mult(*Jt, *B));
  std::unique_ptr<SparseMatrix> JtBJ(mfem::Mult(*JtB, *J));
  std::unique_ptr<SparseMatrix> A00(Add(1.0, *As, theta, *B));
  std::unique_ptr<SparseMatrix> A11(Add(1.0, *Af, theta, *JtBJ));
  GSSmoother prec0(*A00), prec1(*A11);
  NullSpaceProjector P;
  CGSolver cg;
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

  {  // common translations; independent rotations (circular interface)
    BlockVector nvec(offsets);
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
      nvec.GetBlock(0) = ts;
      nvec.GetBlock(1) = tf;
      P.Add(nvec);
    }
    RigidRotation rot(dim, 2);
    gs.ProjectCoefficient(rot);
    gf.ProjectCoefficient(rot);
    gs.GetTrueDofs(ts);
    gf.GetTrueDofs(tf);
    nvec.GetBlock(0) = ts;
    nvec.GetBlock(1) = 0.0;
    P.Add(nvec);
    nvec.GetBlock(0) = 0.0;
    nvec.GetBlock(1) = tf;
    P.Add(nvec);
  }

  ProjectedOperator op_p(block_op, P);
  ProjectedSolver prec_p(P);
  prec_p.SetSolver(block_prec);
  cg.SetOperator(op_p);
  cg.SetPreconditioner(prec_p);
  cg.SetRelTol(rel_tol);
  cg.SetAbsTol(0.0);
  cg.SetMaxIter(50000);
  cg.iterative_mode = true;
  ProjectedSolver solver(P);
  solver.SetSolver(cg);
  solver.iterative_mode = true;

  auto surf = RadialBdrMarker(solid, 0.9, 1.1);

  for (L = lmin; L <= lmax; L++) {
    VectorFunctionCoefficient traction(dim, Traction);
    LinearFormType lf(fes_s.get());
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

    BlockVector U(offsets), rhs(offsets), w(offsets);
    U = 0.0;
    w = 0.0;
    int its = 0;
    double jump = 0.0;
    for (int k = 0; k < n_al; k++) {
      rhs = F;
      rhs -= w;
      Vector qf(nf);
      Qf->Mult(U.GetBlock(1), qf);
      rhs.GetBlock(1) += qf;
      solver.Mult(rhs, U);
      MFEM_VERIFY(cg.GetConverged(), "slip CG did not converge");
      its += cg.GetNumIterations();
      // w <- w + theta C U, with C the unscaled penalty matrix
      // (C U = (-B j; J^T B j) for j = J u_f - u_s).
      Vector js(ns), Bj(ns), tmp(nf);
      J->Mult(U.GetBlock(1), js);
      js -= U.GetBlock(0);
      B->Mult(js, Bj);
      jump = std::sqrt(std::max(0.0, Dot(js, Bj)));
      w.GetBlock(0).Add(-theta, Bj);
      Jt->Mult(Bj, tmp);
      w.GetBlock(1).Add(theta, tmp);
    }
    FieldType us(fes_s.get());
    us.SetFromTrueDofs(U.GetBlock(0));
    Result r;
    r.l = L;
    r.iterations = its;
    r.normal_jump = jump;
    r.surface = ResponseAmplitudes(*fes_s, us, surf, L);
    r.interface_ = ResponseAmplitudes(*fes_s, us, cmb, L);
    r.p1 = L == 0 ? -2.0 * kappa_f * r.interface_.U / kRCmb : 0.0;
    results.push_back(r);
    if (Root()) {
      std::cout << "slip   l=" << L << "  ur_a=" << r.surface.U
                << "  ut_a=" << r.surface.V << "  jump=" << jump
                << "  its=" << its << "\n";
    }
  }
  WriteJson(out_file, method, order, refinements, kappa_s, mu_s, kappa_f,
            results);
  return 0;
}
