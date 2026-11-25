/*
  Tests for the surface terms of a vacuum written in the Eulerian
  potential (VacuumSurfaceCouplingIntegrator,
  VacuumSurfaceGravityIntegrator in bilininteg.hpp).

  1. The two forms on interpolated fields against the quadrature of
     their integrands from the analytic fields, on the surface of the
     disc and of the ball.
  2. The identity they come from: on the buffer, for an extended
     displacement u and a potential increment phi of no particular kind,
     with zeta1 = phi + u . g0 and g0 the gradient of a potential that is
     harmonic there,
         Q_V[u, zeta1] = (1 / 8 pi G) int_V |grad phi|^2 + T_s[u, phi],
     Q_V the referential functional of the buffer as the referential
     integrators assemble it, T_s the surface functional.
  Both fall as the mesh is refined.
*/

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

#include "AdGIA.hpp"
#include "mfem.hpp"

namespace {

using namespace mfem;
using namespace AdGIA;

const double kPi = std::numbers::pi;

// The vacuum's equilibrium gradient outside a body of unit radius and
// density (G = 1), and its Hessian.
void VacuumGradient(const Vector& x, Vector& g) {
  const int dim = x.Size();
  const double r = x.Norml2();
  g = x;
  g *= dim == 3 ? (4.0 * kPi / 3.0) / (r * r * r) : 2.0 * kPi / (r * r);
}

void VacuumHessian(const Vector& x, DenseMatrix& H) {
  const int dim = x.Size();
  const double r = x.Norml2();
  const double c =
      dim == 3 ? (4.0 * kPi / 3.0) / (r * r * r) : 2.0 * kPi / (r * r);
  H.SetSize(dim);
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      H(i, j) = c * ((i == j ? 1.0 : 0.0) - dim * x[i] * x[j] / (r * r));
    }
  }
}

// A smooth displacement times a taper t(r) that vanishes at r_out, with
// its gradient; t = 1 gives the displacement itself.
void Displacement(const Vector& x, Vector& u, DenseMatrix* grad,
                  double r_out) {
  const int dim = x.Size();
  const double r = x.Norml2();
  double t = 1.0, dt = 0.0;
  if (r_out > 0.0) {
    const double q = (r_out - r) / (r_out - 1.0);
    t = q * q;
    dt = -2.0 * q / (r_out - 1.0);
  }
  u.SetSize(dim);
  if (grad) {
    grad->SetSize(dim);
  }
  Vector ds(dim);
  for (int i = 0; i < dim; i++) {
    double s = 0.0;
    for (int j = 0; j < dim; j++) {
      const double k = 1.1 + 0.4 * i - 0.3 * j;
      const double a = k * x[j] + 0.3 + 0.9 * i + 0.5 * j;
      s += std::sin(a);
      ds[j] = k * std::cos(a);
    }
    u[i] = 0.05 * t * s;
    for (int j = 0; grad && j < dim; j++) {
      (*grad)(i, j) = 0.05 * (dt * x[j] / r * s + t * ds[j]);
    }
  }
}

double Increment(const Vector& x, Vector* grad) {
  double a = 0.3;
  for (int j = 0; j < x.Size(); j++) {
    a += (1.1 - 0.35 * j) * x[j];
  }
  if (grad) {
    grad->SetSize(x.Size());
    for (int j = 0; j < x.Size(); j++) {
      (*grad)[j] = 0.4 * (1.1 - 0.35 * j) * std::cos(a);
    }
  }
  return 0.4 * std::sin(a);
}

Mesh Ball(int dim, int refinements) {
  Mesh mesh(dim == 2 ? "../data/disc_buffer_2d.msh"
                     : "../data/ball_buffer_coarse_3d.msh",
            1, 1);
  for (int l = 0; l < refinements; l++) {
    mesh.UniformRefinement();
  }
  return mesh;
}

// T_s[u, phi] by quadrature of the analytic fields over the boundary
// elements of attribute 1 of a mesh, with the normal `sign` times the
// mesh's own: the coupling and the quadratic part.
std::pair<double, double> SurfaceFunctional(Mesh& mesh, double sign,
                                            double r_out, int order) {
  const int dim = mesh.Dimension();
  double coupling = 0.0, quadratic = 0.0;
  Vector x, u, g, n(dim), w(dim), Hw(dim), uu(dim), grad_phi;
  DenseMatrix grad_u, H;
  for (int be = 0; be < mesh.GetNBE(); be++) {
    if (mesh.GetBdrAttribute(be) != 1) {
      continue;
    }
    ElementTransformation* T = mesh.GetBdrElementTransformation(be);
    const IntegrationRule& rule =
        IntRules.Get(T->GetGeometryType(), 2 * order + 6);
    for (int q = 0; q < rule.GetNPoints(); q++) {
      const IntegrationPoint& ip = rule.IntPoint(q);
      T->SetIntPoint(&ip);
      T->Transform(ip, x);
      CalcOrtho(T->Jacobian(), n);
      n *= sign / n.Norml2();
      Displacement(x, u, &grad_u, r_out);
      VacuumGradient(x, g);
      VacuumHessian(x, H);
      Increment(x, &grad_phi);
      const double gn = g * n, un = u * n;
      for (int i = 0; i < dim; i++) {
        w[i] = gn * u[i] - un * g[i];
      }
      H.Mult(w, Hw);
      grad_u.Mult(u, uu);  // (u . grad) u
      const double D = un * grad_u.Trace() - (n * uu);
      const double dS = ip.weight * T->Weight();
      coupling += dS * (w * grad_phi) / (4.0 * kPi);
      quadratic += dS * ((u * Hw) - 0.5 * (g * g) * D) / (8.0 * kPi);
    }
  }
  return {coupling, quadratic};
}

class VacuumSurfaceTest : public testing::TestWithParam<int> {};

TEST_P(VacuumSurfaceTest, FormsAgainstTheirIntegrands) {
  const int dim = GetParam(), order = 2;
  std::vector<double> errors;
  for (const int refinements : {0, 1}) {
    Mesh parent = Ball(dim, refinements);
    Array<int> body_attr({1});
    SubMesh body(SubMesh::CreateFromDomain(parent, body_attr));
    H1_FECollection fec(order, dim);
    FiniteElementSpace fes_u(&body, &fec, dim), fes_z(&body, &fec);
    // On the body the fields need only be smooth: the vacuum's gradient
    // is singular at the centre, so one that is smooth there, equal to
    // it on the surface.
    VectorFunctionCoefficient g_c(dim, [](const Vector& x, Vector& g) {
      const double c = x.Size() == 3 ? 4.0 * kPi / 3.0 : 2.0 * kPi;
      g = x;
      g *= c * (2.0 - (x * x));
    });
    VectorFunctionCoefficient u_c(dim, [](const Vector& x, Vector& u) {
      Displacement(x, u, nullptr, -1.0);
    });
    FunctionCoefficient phi_c(
        [](const Vector& x) { return Increment(x, nullptr); });
    GridFunction g(&fes_u), u(&fes_u), phi(&fes_z);
    g.ProjectCoefficient(g_c);
    u.ProjectCoefficient(u_c);
    phi.ProjectCoefficient(phi_c);

    Array<int> surface(body.bdr_attributes.Max());
    surface = 0;
    surface[0] = 1;
    MixedBilinearForm b(&fes_z, &fes_u);
    b.AddBoundaryIntegrator(
        new VacuumSurfaceCouplingIntegrator(g, 1.0 / (4.0 * kPi)), surface);
    b.Assemble();
    b.Finalize();
    BilinearForm a(&fes_u);
    a.AddBoundaryIntegrator(
        new VacuumSurfaceGravityIntegrator(g, 1.0 / (8.0 * kPi)), surface);
    a.Assemble();
    a.Finalize();
    Vector b_phi(fes_u.GetVSize());
    b.Mult(phi, b_phi);
    const double coupling = u * b_phi;
    const double quadratic = 0.5 * a.InnerProduct(u, u);

    // The reference: the same integrands with the smooth gradient above,
    // whose Hessian on a tangent vector of the unit sphere is that of
    // c (2 - r^2) x.
    double coupling_ref = 0.0, quadratic_ref = 0.0;
    Vector x, uu, gg, n(dim), w(dim), Hw(dim), gradu_u(dim), grad_phi;
    DenseMatrix grad_u, H(dim);
    for (int be = 0; be < body.GetNBE(); be++) {
      if (body.GetBdrAttribute(be) != 1) {
        continue;
      }
      ElementTransformation* T = body.GetBdrElementTransformation(be);
      const IntegrationRule& rule =
          IntRules.Get(T->GetGeometryType(), 2 * order + 6);
      for (int q = 0; q < rule.GetNPoints(); q++) {
        const IntegrationPoint& ip = rule.IntPoint(q);
        T->SetIntPoint(&ip);
        T->Transform(ip, x);
        CalcOrtho(T->Jacobian(), n);
        n /= n.Norml2();
        Displacement(x, uu, &grad_u, -1.0);
        const double c = dim == 3 ? 4.0 * kPi / 3.0 : 2.0 * kPi;
        gg = x;
        gg *= c * (2.0 - (x * x));
        for (int i = 0; i < dim; i++) {
          for (int j = 0; j < dim; j++) {
            H(i, j) = c * ((i == j ? 2.0 - (x * x) : 0.0) - 2.0 * x[i] * x[j]);
          }
        }
        Increment(x, &grad_phi);
        const double gn = gg * n, un = uu * n;
        for (int i = 0; i < dim; i++) {
          w[i] = gn * uu[i] - un * gg[i];
        }
        H.Mult(w, Hw);
        grad_u.Mult(uu, gradu_u);
        const double D = un * grad_u.Trace() - (n * gradu_u);
        const double dS = ip.weight * T->Weight();
        coupling_ref += dS * (w * grad_phi) / (4.0 * kPi);
        quadratic_ref +=
            dS * ((uu * Hw) - 0.5 * (gg * gg) * D) / (8.0 * kPi);
      }
    }
    EXPECT_GT(std::abs(coupling_ref), 1e-3);
    EXPECT_GT(std::abs(quadratic_ref), 1e-3);
    errors.push_back(std::max(std::abs(coupling / coupling_ref - 1.0),
                              std::abs(quadratic / quadratic_ref - 1.0)));
  }
  EXPECT_LT(errors[1], 0.5 * errors[0]);
  EXPECT_LT(errors[1], dim == 2 ? 1e-4 : 2e-2);
}

TEST_P(VacuumSurfaceTest, TheVacuumsFunctionalIsTheDirichletIntegralAndThem) {
  const int dim = GetParam(), order = 3;
  std::vector<double> errors;
  for (const int refinements : {0, 1}) {
    Mesh parent = Ball(dim, refinements);
    Vector lo, hi;
    parent.GetBoundingBox(lo, hi);
    const double r_out = hi.Normlinf();
    Array<int> buffer_attr({2});
    SubMesh V(SubMesh::CreateFromDomain(parent, buffer_attr));
    H1_FECollection fec(order, dim);
    FiniteElementSpace fes_u(&V, &fec, dim), fes_z(&V, &fec);
    VectorFunctionCoefficient u_c(dim, [r_out](const Vector& x, Vector& u) {
      Displacement(x, u, nullptr, r_out);
    });
    VectorFunctionCoefficient g_c(dim, VacuumGradient);
    FunctionCoefficient zeta_c([r_out](const Vector& x) {
      Vector u, g;
      Displacement(x, u, nullptr, r_out);
      VacuumGradient(x, g);
      return Increment(x, nullptr) + u * g;
    });
    GridFunction u(&fes_u), zeta(&fes_z);
    u.ProjectCoefficient(u_c);
    zeta.ProjectCoefficient(zeta_c);

    // Q_V, as the referential class assembles the buffer.
    IdentityDiffeomorphism identity(dim);
    BilinearForm gravity(&fes_u);
    gravity.AddDomainIntegrator(new ReferentialGravityIntegrator(
        identity, g_c, 1.0 / (8.0 * kPi)));
    gravity.Assemble();
    gravity.Finalize();
    MixedBilinearForm coupling(&fes_z, &fes_u);
    coupling.AddDomainIntegrator(new ReferentialGravityCouplingIntegrator(
        identity, g_c, 1.0 / (4.0 * kPi)));
    coupling.Assemble();
    coupling.Finalize();
    ConstantCoefficient scale(1.0 / (4.0 * kPi));
    BilinearForm laplacian(&fes_z);
    laplacian.AddDomainIntegrator(new DiffusionIntegrator(scale));
    laplacian.Assemble();
    laplacian.Finalize();
    Vector c_zeta(fes_u.GetVSize());
    coupling.Mult(zeta, c_zeta);
    const double Q = 0.5 * gravity.InnerProduct(u, u) + (u * c_zeta) +
                     0.5 * laplacian.InnerProduct(zeta, zeta);

    double dirichlet = 0.0;
    Vector x, grad_phi;
    for (int e = 0; e < V.GetNE(); e++) {
      ElementTransformation* T = V.GetElementTransformation(e);
      const IntegrationRule& rule =
          IntRules.Get(T->GetGeometryType(), 2 * order + 6);
      for (int q = 0; q < rule.GetNPoints(); q++) {
        const IntegrationPoint& ip = rule.IntPoint(q);
        T->SetIntPoint(&ip);
        T->Transform(ip, x);
        Increment(x, &grad_phi);
        dirichlet +=
            ip.weight * T->Weight() * (grad_phi * grad_phi) / (8.0 * kPi);
      }
    }
    // The buffer's own normal on the body's surface points into the body.
    const auto [t_coupling, t_quadratic] =
        SurfaceFunctional(V, -1.0, r_out, order);
    const double surface = t_coupling + t_quadratic;
    // The surface terms are most of it: the identity is not carried by
    // the Dirichlet integral.
    EXPECT_GT(std::abs(surface), std::abs(dirichlet));
    errors.push_back(std::abs(Q - dirichlet - surface) / std::abs(surface));
  }
  // Measured: 4e-7 and 2e-8 on the disc, 1e-3 and 6e-5 on the ball.
  EXPECT_LT(errors[1], 0.2 * errors[0]);
  EXPECT_LT(errors[1], dim == 2 ? 1e-7 : 2e-4);
}

INSTANTIATE_TEST_SUITE_P(Dimensions, VacuumSurfaceTest, testing::Values(2, 3));

}  // namespace
