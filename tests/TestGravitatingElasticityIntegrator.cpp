#include <cstdlib>
#include <memory>
#include <numbers>

#include "TestCommon.hpp"

/*
  GravitatingElasticityIntegrator, the displacement block of the
  self-gravitating problem in one integrator.

  - Its element matrices assemble to the matrix of the integrators it
    stands for (the elastic tensor form, the two gravity forms with their
    transposes, a mass term), on tetrahedra and hexahedra.
  - On a device that uses libCEED (ADGIA_TEST_DEVICE, e.g. ceed-cpu or
    ceed-cuda:/gpu/cuda/gen), its partial assembly applies that matrix
    and gives its diagonal; skipped on other devices.
*/

namespace {

constexpr double kPi = std::numbers::pi;

class DeviceEnvironment : public testing::Environment {
 public:
  void SetUp() override {
    const char* name = std::getenv("ADGIA_TEST_DEVICE");
    device_ = std::make_unique<Device>(name ? name : "cpu");
  }
  void TearDown() override { device_.reset(); }

 private:
  std::unique_ptr<Device> device_;
};
const auto* const device_environment =
    testing::AddGlobalTestEnvironment(new DeviceEnvironment);

void Jiggle(const Vector& x, Vector& y) {
  y = x;
  const double s = 0.06;
  y[0] += s * std::sin(kPi * x[0]) * std::sin(kPi * x[1]);
  y[1] += s * std::cos(0.5 * kPi * x[0]) * std::sin(kPi * x[1]);
  y[2] += s * std::sin(kPi * x[2]) * std::cos(kPi * x[0]);
}

// A background potential with its gradient and Hessian:
// Phi0 = a |x|^2 / 2 + b x0 x1 x2 + c sin(x0 + 2 x2).
constexpr double kA = 1.3, kB = 0.7, kC = 0.4;
void GradPhi0(const Vector& x, Vector& g) {
  const double co = kC * std::cos(x[0] + 2.0 * x[2]);
  g[0] = kA * x[0] + kB * x[1] * x[2] + co;
  g[1] = kA * x[1] + kB * x[0] * x[2];
  g[2] = kA * x[2] + kB * x[0] * x[1] + 2.0 * co;
}
void HessianPhi0(const Vector& x, DenseMatrix& H) {
  const double si = kC * std::sin(x[0] + 2.0 * x[2]);
  H(0, 0) = kA - si;
  H(1, 1) = kA;
  H(2, 2) = kA - 4.0 * si;
  H(0, 1) = H(1, 0) = kB * x[2];
  H(0, 2) = H(2, 0) = kB * x[1] - 2.0 * si;
  H(1, 2) = H(2, 1) = kB * x[0];
}

struct Fields {
  FunctionCoefficient rho{[](const Vector& x) { return 2.0 + 0.5 * x[0] - 0.3 * x[2] * x[2]; }};
  VectorFunctionCoefficient grad{3, GradPhi0};
  MatrixFunctionCoefficient hessian{3, HessianPhi0};
  FunctionCoefficient shift{[](const Vector& x) { return 0.7 + 0.2 * x[1]; }};
  FunctionCoefficient lambda{[](const Vector& x) { return 1.3 + 0.4 * std::sin(2.0 * x[0] + x[2]); }};
  FunctionCoefficient mu{[](const Vector& x) { return 0.8 + 0.3 * x[1] * x[1]; }};
  IsotropicElasticTensorCoefficient iso{3, lambda, mu};
  ConstantCoefficient A{3.1}, C{2.7}, F{1.1}, L{0.9}, N{1.2};
  Vector x0;
  std::unique_ptr<RadialUnitVectorCoefficient> axis;
  std::unique_ptr<TransverselyIsotropicElasticTensorCoefficient> ti;
  Fields() : x0(3) {
    x0 = -0.3;
    axis = std::make_unique<RadialUnitVectorCoefficient>(3, x0);
    ti = std::make_unique<TransverselyIsotropicElasticTensorCoefficient>(
        3, A, C, F, L, N, *axis);
  }
};

Mesh CurvedMesh(bool hexahedra) {
  auto mesh = Mesh::MakeCartesian3D(2, 2, 2,
                                    hexahedra ? Element::HEXAHEDRON
                                              : Element::TETRAHEDRON);
  mesh.SetCurvature(2);
  mesh.Transform(Jiggle);
  return mesh;
}

using Param = std::tuple<bool, int, bool, bool>;  // hexahedra, order, isotropic, mass

class GravitatingElasticity : public testing::TestWithParam<Param> {
 protected:
  Fields f;
  MatrixCoefficient& Tensor() {
    return std::get<2>(GetParam()) ? static_cast<MatrixCoefficient&>(f.iso)
                                   : *f.ti;
  }
  std::unique_ptr<GravitatingElasticityIntegrator> Make(
      const IntegrationRule* ir) {
    auto integ = std::make_unique<GravitatingElasticityIntegrator>(
        Tensor(), f.rho, f.grad, f.hessian, ir);
    if (std::get<3>(GetParam())) {
      integ->SetMassTerm(&f.shift);
    }
    return integ;
  }
};

TEST_P(GravitatingElasticity, IsTheSumOfItsTerms) {
  const auto [hexahedra, order, isotropic, mass] = GetParam();
  auto mesh = CurvedMesh(hexahedra);
  H1_FECollection fec(order, 3);
  FiniteElementSpace fes(&mesh, &fec, 3);
  const IntegrationRule& ir =
      IntRules.Get(mesh.GetTypicalElementGeometry(), 2 * order + 2);

  BilinearForm a(&fes);
  a.AddDomainIntegrator(Make(&ir).release());
  a.Assemble();
  a.Finalize();

  ProductCoefficient half_rho(0.5, f.rho), minus_half_rho(-0.5, f.rho);
  ScalarVectorProductCoefficient minus_half_rho_grad(minus_half_rho, f.grad);
  BilinearForm b(&fes);
  b.AddDomainIntegrator(new ElasticTensorIntegrator(Tensor(), &ir));
  auto* g1 = new DomainVectorGradVectorIntegrator(f.grad, half_rho, &ir);
  g1->SetGradient(&f.hessian);
  auto* g2 = new DomainVectorDivVectorIntegrator(minus_half_rho_grad, &ir);
  b.AddDomainIntegrator(g1);
  b.AddDomainIntegrator(g2);
  b.AddDomainIntegrator(new TransposeIntegrator(g1, 0));
  b.AddDomainIntegrator(new TransposeIntegrator(g2, 0));
  if (mass) {
    auto* m = new VectorMassIntegrator(f.shift);
    m->SetIntRule(&ir);
    b.AddDomainIntegrator(m);
  }
  b.Assemble();
  b.Finalize();

  std::unique_ptr<SparseMatrix> D(Add(1.0, a.SpMat(), -1.0, b.SpMat()));
  EXPECT_LT(D->MaxNorm(), 1e-12 * b.SpMat().MaxNorm());
}

TEST_P(GravitatingElasticity, PartialAssemblyThroughLibCeed) {
  if (!DeviceCanUseCeed()) {
    GTEST_SKIP() << "the device does not use libCEED";
  }
  const auto [hexahedra, order, isotropic, mass] = GetParam();
  auto mesh = CurvedMesh(hexahedra);
  H1_FECollection fec(order, 3);
  FiniteElementSpace fes(&mesh, &fec, 3);
  for (const bool given_rule : {false, true}) {
    const IntegrationRule* ir =
        given_rule ? &IntRules.Get(mesh.GetTypicalElementGeometry(),
                                   2 * order + 2)
                   : nullptr;
    BilinearForm a(&fes);
    a.AddDomainIntegrator(Make(ir).release());
    a.Assemble();
    a.Finalize();

    BilinearForm b(&fes);
    b.SetAssemblyLevel(AssemblyLevel::PARTIAL);
    b.AddDomainIntegrator(Make(ir).release());
    b.Assemble();

    const double scale = a.SpMat().MaxNorm();
    for (int trial = 0; trial < 3; trial++) {
      auto x = RandomVector(fes.GetVSize());
      Vector ya(x.Size()), yb(x.Size());
      a.SpMat().Mult(x, ya);
      b.Mult(x, yb);
      yb -= ya;
      EXPECT_LT(yb.Normlinf(), 1e-11 * scale * x.Normlinf());
    }
    Vector da, db(fes.GetVSize());
    a.SpMat().GetDiag(da);
    b.AssembleDiagonal(db);
    db -= da;
    EXPECT_LT(db.Normlinf(), 1e-11 * scale);
  }
}

INSTANTIATE_TEST_SUITE_P(
    Cases, GravitatingElasticity,
    testing::Combine(testing::Bool(), testing::Values(1, 2, 3),
                     testing::Bool(), testing::Bool()));

}  // namespace
