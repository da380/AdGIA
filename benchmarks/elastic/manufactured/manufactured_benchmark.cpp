// ============================================================================
// manufactured_benchmark.cpp
//
// The method of manufactured solutions for the static self-gravitating
// elastic problem (LinearQuasiStaticMixedSelfGravitatingProblem) on a
// solid body: the exact pair (u*, phi*) and the loads that make it the
// solution are those of fields.hpp, generated symbolically by
// manufacture.py from the class's weak form (see its docstring for the
// derivation). The body force and the surface traction load the
// displacement row (AddForce), the volume and surface sources the
// potential row (ExternalPotentialLoad); the background potential is the
// exact one of the homogeneous ball, supplied analytically, so that the
// rigid modes are exact null pairs of the continuous problem.
//
// The discrete solution is gauged by the solver (no rigid component in
// the true-dof inner product), the exact one is not, so the rigid part of
// u_h - u* is fitted on the discrete rigid modes and removed from the
// displacement error, and the matching potential shift -r.grad Phi0 of
// the null pair is removed from the potential error (in 2-D also the
// constant the class projects out). Reported, relative to the exact
// fields: the L2 error of u, the L2 error of the stress sigma(u) (the
// energy-type measure, one order lower), and the L2 error of phi over the
// whole computational ball, with the mesh's size range and the costs.
//
// Sample run:
//    mpiexec -np 4 ./manufactured_benchmark -m ../../data/ball_buffer_coarse_3d.msh \
//        -o 2 -out manufactured_o2.json
// ============================================================================

#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "AdGIA/mixed_problem.hpp"
#include "AdGIA/null_space.hpp"
#include "AdGIA/rheology.hpp"
#include "fields.hpp"
#include "mfem.hpp"

using namespace mfem;
using namespace AdGIA;

namespace {

// The dimension-specific functions of fields.hpp behind one interface.
struct Exact {
  int dim;
  explicit Exact(int d) : dim(d) {}

  bool Inside(const Vector& x) const {
    return x * x <= manufactured::kRadius * manufactured::kRadius;
  }
  void U(const Vector& x, Vector& v) const {
    v.SetSize(dim);
    dim == 3 ? manufactured::d3::u(x.GetData(), v.GetData())
             : manufactured::d2::u(x.GetData(), v.GetData());
  }
  double Phi(const Vector& x) const {
    if (dim == 3) {
      return Inside(x) ? manufactured::d3::phi_in(x.GetData())
                       : manufactured::d3::phi_out(x.GetData());
    }
    return Inside(x) ? manufactured::d2::phi_in(x.GetData())
                     : manufactured::d2::phi_out(x.GetData());
  }
  double Phi0(const Vector& x) const {
    if (dim == 3) {
      return Inside(x) ? manufactured::d3::phi0_in(x.GetData())
                       : manufactured::d3::phi0_out(x.GetData());
    }
    return Inside(x) ? manufactured::d2::phi0_in(x.GetData())
                     : manufactured::d2::phi0_out(x.GetData());
  }
  void GradPhi0(const Vector& x, Vector& g) const {
    g.SetSize(dim);
    if (dim == 3) {
      Inside(x) ? manufactured::d3::grad_phi0_in(x.GetData(), g.GetData())
                : manufactured::d3::grad_phi0_out(x.GetData(), g.GetData());
    } else {
      Inside(x) ? manufactured::d2::grad_phi0_in(x.GetData(), g.GetData())
                : manufactured::d2::grad_phi0_out(x.GetData(), g.GetData());
    }
  }
  double Rho(const Vector& x) const {
    return dim == 3 ? manufactured::d3::rho(x.GetData())
                    : manufactured::d2::rho(x.GetData());
  }
  double Kappa(const Vector& x) const {
    return dim == 3 ? manufactured::d3::kappa(x.GetData())
                    : manufactured::d2::kappa(x.GetData());
  }
  double Mu(const Vector& x) const {
    return dim == 3 ? manufactured::d3::mu(x.GetData())
                    : manufactured::d2::mu(x.GetData());
  }
  void BodyForce(const Vector& x, Vector& f) const {
    f.SetSize(dim);
    dim == 3 ? manufactured::d3::body_force(x.GetData(), f.GetData())
             : manufactured::d2::body_force(x.GetData(), f.GetData());
  }
  void Traction(const Vector& x, const Vector& n, Vector& t) const {
    t.SetSize(dim);
    dim == 3 ? manufactured::d3::traction(x.GetData(), n.GetData(),
                                          t.GetData())
             : manufactured::d2::traction(x.GetData(), n.GetData(),
                                          t.GetData());
  }
  double PotentialSource(const Vector& x) const {
    if (!Inside(x)) {
      return 0.0;
    }
    return dim == 3 ? manufactured::d3::potential_source(x.GetData())
                    : manufactured::d2::potential_source(x.GetData());
  }
  double SurfaceSource(const Vector& x, const Vector& n) const {
    return dim == 3
               ? manufactured::d3::surface_source(x.GetData(), n.GetData())
               : manufactured::d2::surface_source(x.GetData(), n.GetData());
  }
  void Stress(const Vector& x, DenseMatrix& s) const {
    s.SetSize(dim);
    Vector flat(dim * dim);
    dim == 3 ? manufactured::d3::stress(x.GetData(), flat.GetData())
             : manufactured::d2::stress(x.GetData(), flat.GetData());
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        s(i, j) = flat[i * dim + j];  // row-major from the generator
      }
    }
  }
};

// The surface sources evaluated with the facet's own unit normal (the
// discrete domain's boundary, not the exact sphere's: an order-2 facet's
// normal is O(h^2) off the sphere's, and a traction posed with the exact
// normal is inconsistent with the discrete boundary at that order).
struct FacetNormal {
  static void Of(ElementTransformation& T, const IntegrationPoint& ip,
                 Vector& n) {
    T.SetIntPoint(&ip);
    n.SetSize(T.GetSpaceDim());
    CalcOrtho(T.Jacobian(), n);
    n /= n.Norml2();
  }
};

class TractionCoefficient : public VectorCoefficient {
 public:
  explicit TractionCoefficient(const Exact& ex)
      : VectorCoefficient(ex.dim), ex_(ex) {}
  void Eval(Vector& V, ElementTransformation& T,
            const IntegrationPoint& ip) override {
    T.Transform(ip, x_);
    FacetNormal::Of(T, ip, n_);
    ex_.Traction(x_, n_, V);
  }

 private:
  const Exact& ex_;
  Vector x_, n_;
};

class SurfaceSourceCoefficient : public Coefficient {
 public:
  explicit SurfaceSourceCoefficient(const Exact& ex) : ex_(ex) {}
  double Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    T.Transform(ip, x_);
    FacetNormal::Of(T, ip, n_);
    return ex_.SurfaceSource(x_, n_);
  }

 private:
  const Exact& ex_;
  Vector x_, n_;
};

// Relative L2 error of the discrete stress sigma(u_h) against the exact
// one, over the body: sqrt(sum w |sigma_h - sigma*|^2 / sum w |sigma*|^2).
// With `layer` set, over the elements with a vertex within 0.05 of the
// surface only (the curved ones), or (-1) over the rest.
double StressError(const ParGridFunction& u, const Exact& ex, int order,
                   int layer = 0) {
  const auto& fes = *u.ParFESpace();
  Mesh& mesh = *fes.GetMesh();
  const int dim = mesh.Dimension();
  double num = 0.0, den = 0.0;
  Vector x;
  DenseMatrix grad(dim), sig(dim), sig_ex(dim);
  Array<int> verts;
  for (int e = 0; e < mesh.GetNE(); e++) {
    if (layer != 0) {
      mesh.GetElementVertices(e, verts);
      bool near = false;
      for (int v : verts) {
        const double* c = mesh.GetVertex(v);
        double r2 = 0.0;
        for (int i = 0; i < dim; i++) r2 += c[i] * c[i];
        near = near || r2 > 0.95 * 0.95;
      }
      if ((layer > 0) != near) continue;
    }
    ElementTransformation& T = *mesh.GetElementTransformation(e);
    const IntegrationRule& ir =
        IntRules.Get(mesh.GetElementGeometry(e), 2 * order + 2);
    for (int q = 0; q < ir.GetNPoints(); q++) {
      const IntegrationPoint& ip = ir.IntPoint(q);
      T.SetIntPoint(&ip);
      T.Transform(ip, x);
      u.GetVectorGradient(T, grad);
      const double kappa = ex.Kappa(x), mu = ex.Mu(x);
      double divu = 0.0;
      for (int i = 0; i < dim; i++) divu += grad(i, i);
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
          const double eps = 0.5 * (grad(i, j) + grad(j, i));
          sig(i, j) = 2.0 * mu * (eps - (i == j ? divu / dim : 0.0)) +
                      (i == j ? kappa * divu : 0.0);
        }
      }
      ex.Stress(x, sig_ex);
      const double w = ip.weight * T.Weight();
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
          const double d = sig(i, j) - sig_ex(i, j);
          num += w * d * d;
          den += w * sig_ex(i, j) * sig_ex(i, j);
        }
      }
    }
  }
  double v[2] = {num, den};
  MPI_Allreduce(MPI_IN_PLACE, v, 2, MPI_DOUBLE, MPI_SUM, fes.GetComm());
  return std::sqrt(v[0] / v[1]);
}

}  // namespace

int main(int argc, char* argv[]) try {
  Mpi::Init(argc, argv);
  Hypre::Init();
  const bool root = Mpi::Root();
  using Clock = std::chrono::steady_clock;

  const char* mesh_file = "../../data/ball_buffer_coarse_3d.msh";
  const char* out_file = "manufactured.json";
  int order = 2, order_phi = 0, dtn_degree = 12;
  double rel_tol = 1e-11;
  bool schur = false, mass_gauge = false, exact_hessian = false,
       interpolant = false;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
                 "A body (attribute 1) of unit radius inside a buffer "
                 "(attribute 2) with a spherical outer boundary.");
  args.AddOption(&out_file, "-out", "--output", "Results file (JSON).");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&order_phi, "-ophi", "--order-phi",
                 "Order of the potential space (0: the displacement's). "
                 "The background potential is projected onto it and "
                 "differentiated, so its order sets the accuracy of the "
                 "background gravity in the displacement row.");
  args.AddOption(&dtn_degree, "-deg", "--dtn-degree", "DtN degree.");
  args.AddOption(&rel_tol, "-rt", "--rel-tol", "Solver tolerance.");
  args.AddOption(&schur, "-schur", "--schur-cg", "-minres", "--block-minres",
                 "Schur-complement CG instead of block MINRES.");
  args.AddOption(&exact_hessian, "-exact-hessian", "--exact-hessian",
                 "-discrete-hessian", "--discrete-hessian",
                 "Assemble the gravity term by the product rule with the "
                 "EXACT Hessian of Phi0 (SetBackgroundHessian) instead of "
                 "the class's default, the Hessian of the discrete Phi0.");
  args.AddOption(&interpolant, "-interpolant", "--interpolant",
                 "-hessian", "--hessian",
                 "The nodal interpolant of u.grad Phi0 (the default before "
                 "5 November 2026) instead of the product rule.");
  args.AddOption(&mass_gauge, "-mass-gauge", "--mass-weighted-gauge",
                 "-euclid-gauge", "--euclidean-gauge",
                 "Gauge the rigid modes by zero momentum instead of the "
                 "true-dof inner product.");
  args.Parse();
  if (!args.Good()) {
    if (root) args.PrintUsage(std::cout);
    return 1;
  }
  if (root) args.PrintOptions(std::cout);

  const auto t_start = Clock::now();
  Mesh smesh(mesh_file, 1, 1);
  const int dim = smesh.Dimension();
  MFEM_VERIFY(dim == 2 || dim == 3, "two or three dimensions");
  MFEM_VERIFY(smesh.attributes.Max() >= 2,
              "the mesh must hold a body (attribute 1) in a buffer (2)");
  ParMesh parent(MPI_COMM_WORLD, smesh);
  smesh.Clear();
  Array<int> body_attr({1});
  ParSubMesh body(ParSubMesh::CreateFromDomain(parent, body_attr));
  if (order_phi <= 0) order_phi = order;
  H1_FECollection fec(order, dim), fec_phi(order_phi, dim);
  ParFiniteElementSpace fes_u(&body, &fec, dim), fes_phi(&parent, &fec_phi);
  Array<int> surface(body.bdr_attributes.Max());
  surface = 0;
  surface[body.bdr_attributes.Max() - 1] = 1;

  const Exact ex(dim);
  FunctionCoefficient rho([&](const Vector& x) { return ex.Rho(x); });
  FunctionCoefficient kappa([&](const Vector& x) { return ex.Kappa(x); });
  FunctionCoefficient mu([&](const Vector& x) { return ex.Mu(x); });
  FunctionCoefficient phi0([&](const Vector& x) { return ex.Phi0(x); });
  IsotropicElasticRheology rheology(dim, kappa, mu);
  LinearQuasiStaticMixedSelfGravitatingProblem problem(
      &fes_u, &fes_phi, rheology, rho, manufactured::kG, dtn_degree, &phi0);
  problem.SetRelTol(rel_tol);
  if (schur) {
    problem.SetSolverType(
        LinearQuasiStaticMixedSelfGravitatingProblem::SolverType::SchurCG);
  }
  if (mass_gauge) {
    problem.SetMassWeightedGauge();
  }
  MatrixFunctionCoefficient hess(dim, [&](const Vector& x, DenseMatrix& H) {
    H.SetSize(dim);
    Vector flat(dim * dim);
    dim == 3 ? manufactured::d3::hess_phi0_in(x.GetData(), flat.GetData())
             : manufactured::d2::hess_phi0_in(x.GetData(), flat.GetData());
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) H(i, j) = flat[i * dim + j];
    }
  });
  MFEM_VERIFY(!(exact_hessian && interpolant), "one gravity route");
  if (exact_hessian) {
    problem.SetBackgroundHessian(&hess);
  } else if (interpolant) {
    problem.SetBackgroundHessian(nullptr);
  }

  // The loads: the potential row's volume and surface sources through
  // the class's own linear form (on the body's shadow of the potential
  // space; the buffer source vanishes by construction), the displacement
  // row's body force and traction through AddForce().
  FunctionCoefficient q([&](const Vector& x) { return ex.PotentialSource(x); });
  SurfaceSourceCoefficient s(ex);
  problem.ExternalPotentialLoad().AddDomainIntegrator(
      new DomainLFIntegrator(q));
  problem.ExternalPotentialLoad().AddBoundaryIntegrator(
      new BoundaryLFIntegrator(s), surface);
  VectorFunctionCoefficient f(dim, [&](const Vector& x, Vector& v) {
    ex.BodyForce(x, v);
  });
  TractionCoefficient tr(ex);
  ParLinearForm lf(&fes_u);
  lf.AddDomainIntegrator(new VectorDomainLFIntegrator(f));
  lf.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(tr), surface);
  lf.Assemble();
  problem.AssembleForce(0.0);
  problem.AddForce(lf);

  const auto t_solve = Clock::now();
  MFEM_VERIFY(problem.Solve(), "the solve failed");
  const double solve_s =
      std::chrono::duration<double>(Clock::now() - t_solve).count();

  const auto& uh = dynamic_cast<const ParGridFunction&>(problem.Displacement());
  const auto& ph = dynamic_cast<const ParGridFunction&>(problem.Potential());

  // The rigid part of u_h - I_h u*, fitted on the discrete rigid modes
  // (translations e_k, rotations W_k x) in the true-dof inner product;
  // its coefficients give the rigid motion r = a + b x x.
  VectorFunctionCoefficient u_ex(dim, [&](const Vector& x, Vector& v) {
    ex.U(x, v);
  });
  ParGridFunction u_star(&fes_u);
  u_star.ProjectCoefficient(u_ex);
  Vector e_t, us_t;
  uh.GetTrueDofs(e_t);
  u_star.GetTrueDofs(us_t);
  e_t -= us_t;
  const int nmodes = dim == 3 ? 6 : 3;
  std::vector<Vector> modes(nmodes);
  for (int k = 0; k < nmodes; k++) {
    VectorFunctionCoefficient mc(dim, [&, k](const Vector& x, Vector& v) {
      v.SetSize(dim);
      v = 0.0;
      if (k < dim) {
        v[k] = 1.0;
      } else if (dim == 2) {
        v[0] = -x[1];
        v[1] = x[0];
      } else {
        const int a = k - 3;  // rotation about axis a
        const int i = (a + 1) % 3, j = (a + 2) % 3;
        v[i] = -x[j];
        v[j] = x[i];
      }
    });
    ParGridFunction m(&fes_u);
    m.ProjectCoefficient(mc);
    m.GetTrueDofs(modes[k]);
  }
  DenseMatrix gram(nmodes);
  Vector rhs(nmodes), coef(nmodes);
  for (int i = 0; i < nmodes; i++) {
    rhs[i] = InnerProduct(fes_u.GetComm(), modes[i], e_t);
    for (int j = 0; j < nmodes; j++) {
      gram(i, j) = InnerProduct(fes_u.GetComm(), modes[i], modes[j]);
    }
  }
  DenseMatrixInverse(gram).Mult(rhs, coef);
  Vector a(dim), b(3);
  b = 0.0;
  for (int k = 0; k < dim; k++) a[k] = coef[k];
  if (dim == 2) {
    b[2] = coef[2];
  } else {
    for (int k = 0; k < 3; k++) b[k] = coef[3 + k];
  }
  // u_h is compared with u* + r: the corrected discrete field u_h - r_h.
  Vector r_t(e_t.Size());
  r_t = 0.0;
  for (int k = 0; k < nmodes; k++) r_t.Add(coef[k], modes[k]);
  Vector w_t;
  uh.GetTrueDofs(w_t);
  w_t -= r_t;
  ParGridFunction w(&fes_u);
  w.SetFromTrueDofs(w_t);
  Vector zero_v(dim);
  zero_v = 0.0;
  VectorConstantCoefficient zero_vc(zero_v);
  const double u_norm = u_star.ComputeL2Error(zero_vc);
  const double u_err = w.ComputeL2Error(u_ex) / u_norm;
  const double stress_err = StressError(w, ex, order);
  const double stress_layer = StressError(w, ex, order, 1);
  const double stress_interior = StressError(w, ex, order, -1);
  // Cross-check of the strain error: MFEM's own H1-seminorm error of
  // each component against the exact gradient (relative to its norm).
  double grad_err = 0.0, grad_norm = 0.0;
  {
    ParFiniteElementSpace fes_s(&body, &fec);
    const int nd = fes_s.GetVSize();
    for (int i = 0; i < dim; i++) {
      Vector comp(const_cast<double*>(w.GetData()) + i * nd, nd);
      ParGridFunction wi(&fes_s, comp.GetData());
      VectorFunctionCoefficient gi(dim, [&, i](const Vector& x, Vector& g) {
        Vector flat(dim * dim);
        dim == 3 ? manufactured::d3::grad_u(x.GetData(), flat.GetData())
                 : manufactured::d2::grad_u(x.GetData(), flat.GetData());
        g.SetSize(dim);
        for (int j = 0; j < dim; j++) g[j] = flat[i * dim + j];
      });
      Vector zero_g(dim);
      zero_g = 0.0;
      VectorConstantCoefficient zg(zero_g);
      ParGridFunction zgf(&fes_s);
      zgf = 0.0;
      const double e = wi.ComputeGradError(&gi), n = zgf.ComputeGradError(&gi);
      grad_err += e * e;
      grad_norm += n * n;
    }
    grad_err = std::sqrt(grad_err / grad_norm);
  }
  const double rigid_norm = std::sqrt(InnerProduct(fes_u.GetComm(), r_t, r_t) /
                                      InnerProduct(fes_u.GetComm(), us_t, us_t));

  // The potential: phi_h ~ phi* - r.grad Phi0 (the null pair's partner),
  // plus, in 2-D, the constant the class projects out.
  auto rigid = [&](const Vector& x, Vector& v) {
    v.SetSize(dim);
    v = a;
    if (dim == 2) {
      v[0] += -b[2] * x[1];
      v[1] += b[2] * x[0];
    } else {
      v[0] += b[1] * x[2] - b[2] * x[1];
      v[1] += b[2] * x[0] - b[0] * x[2];
      v[2] += b[0] * x[1] - b[1] * x[0];
    }
  };
  double shift = 0.0;
  FunctionCoefficient phi_ex([&](const Vector& x) {
    Vector v, g;
    rigid(x, v);
    ex.GradPhi0(x, g);
    return ex.Phi(x) - (v * g) + shift;
  });
  if (dim == 2) {
    GridFunctionCoefficient phc(&ph);
    SumCoefficient diff(phc, phi_ex, 1.0, -1.0);
    ConstantCoefficient one(1.0);
    ParLinearForm lv(&fes_phi), ld(&fes_phi);
    lv.AddDomainIntegrator(new DomainLFIntegrator(one));
    ld.AddDomainIntegrator(new DomainLFIntegrator(diff));
    lv.Assemble();
    ld.Assemble();
    double v[2] = {ld.Sum(), lv.Sum()};
    MPI_Allreduce(MPI_IN_PLACE, v, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    shift = v[0] / v[1];
  }
  ParGridFunction phi_star(&fes_phi);
  FunctionCoefficient phi_plain([&](const Vector& x) { return ex.Phi(x); });
  phi_star.ProjectCoefficient(phi_plain);
  ConstantCoefficient zero(0.0);
  const double phi_norm = phi_star.ComputeL2Error(zero);
  const double phi_err = ph.ComputeL2Error(phi_ex) / phi_norm;

  double h_min, h_max, k_min, k_max;
  parent.GetCharacteristics(h_min, h_max, k_min, k_max);
  const HYPRE_BigInt ndof_u = fes_u.GlobalTrueVSize(),
                     ndof_phi = fes_phi.GlobalTrueVSize();
  const long ne = parent.GetGlobalNE();
  const double total_s =
      std::chrono::duration<double>(Clock::now() - t_start).count();

  if (root) {
    std::cout << std::format(
        "dim {} order {} (phi {}) elements {} dofs u {} phi {} h [{:.4g}, {:.4g}]\n"
        "  |u_h - u*|/|u*|           {:.3e}\n"
        "  |sigma_h - sigma*|/|sigma*| {:.3e} (surface layer {:.3e}, interior {:.3e}; grad cross-check {:.3e})\n"
        "  |phi_h - phi*|/|phi*|     {:.3e}\n"
        "  rigid part removed        {:.3e} (relative to |u*|)\n"
        "  iterations {} solve {:.2f} s total {:.2f} s\n",
        dim, order, order_phi, ne, ndof_u, ndof_phi, h_min, h_max, u_err, stress_err,
        stress_layer, stress_interior, grad_err,
        phi_err, rigid_norm, problem.LastOuterIterations(), solve_s, total_s);
    std::ofstream out(out_file);
    out << std::format(
        "{{\n  \"mesh\": \"{}\",\n  \"dim\": {},\n  \"order\": {},\n  \"order_phi\": {},\n"
        "  \"dtn_degree\": {},\n  \"solver\": \"{}\",\n  \"gauge\": \"{}\",\n"
        "  \"gravity\": \"{}\",\n"
        "  \"elements\": {},\n  \"dofs_u\": {},\n  \"dofs_phi\": {},\n"
        "  \"h_min\": {:.17g},\n  \"h_max\": {:.17g},\n"
        "  \"u_l2\": {:.17g},\n  \"stress_l2\": {:.17g},\n"
        "  \"stress_l2_layer\": {:.17g},\n  \"stress_l2_interior\": {:.17g},\n"
        "  \"grad_l2\": {:.17g},\n"
        "  \"phi_l2\": {:.17g},\n  \"rigid\": {:.17g},\n"
        "  \"iterations\": {},\n  \"solve_seconds\": {:.6g},\n"
        "  \"total_seconds\": {:.6g},\n  \"ranks\": {}\n}}\n",
        mesh_file, dim, order, order_phi, dtn_degree, schur ? "schur" : "minres",
        mass_gauge ? "mass" : "euclid",
        exact_hessian ? "exact-hessian"
        : interpolant ? "interpolant"
                      : "discrete-hessian",
        ne, ndof_u, ndof_phi,
        h_min, h_max,
        u_err, stress_err, stress_layer, stress_interior, grad_err, phi_err,
        rigid_norm, problem.LastOuterIterations(),
        solve_s, total_s, Mpi::WorldSize());
  }
  return 0;
} catch (const std::exception& e) {
  std::cerr << "manufactured_benchmark: " << e.what() << "\n";
  return 1;
}
