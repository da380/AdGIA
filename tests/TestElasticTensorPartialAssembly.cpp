#include <cstdlib>
#include <memory>
#include <numbers>

#include "TestCommon.hpp"

/*
  ElasticTensorIntegrator's partial assembly (3-D; sum-factorised on
  hexahedra, dense in the element on tetrahedra) against its element
  matrices, which the other tests of the integrator cover:

  - the action of the form with no matrix equals the assembled matrix's,
    for an isotropic tensor with varying moduli (kept as two moduli), a
    transversely isotropic tensor with a varying axis (kept as 21
    components), and the pull-back through a mapping; on a curved mesh, at
    orders 1 to 4, in both orderings of the space, with the default rule
    and with a given one;
  - the diagonal equals the assembled matrix's.

  - the same on tetrahedra, at orders 1 to 3.

  Set ADGIA_TEST_DEVICE (e.g. to cuda) to run the kernels on a device. On a
  device that uses libCEED (e.g. ceed-cpu, ceed-cuda:/gpu/cuda/gen) the
  form is a libCEED operator in place of the library's kernels.
*/

namespace {

constexpr double kPi = std::numbers::pi;

void Jiggle(const Vector& x, Vector& y) {
  y = x;
  const double s = 0.08;
  y[0] += s * std::sin(kPi * x[0]) * std::sin(kPi * x[1]);
  y[1] += s * std::cos(0.5 * kPi * x[0]) * std::sin(kPi * x[1]);
  y[2] += s * std::sin(kPi * x[2]) * std::cos(kPi * x[0]);
}

// Geometry of order 2 whatever the order of the space: the default rule
// follows the geometry's order, and the element matrices' cost with it.
Mesh CurvedHexMesh() {
  auto mesh = Mesh::MakeCartesian3D(3, 2, 2, Element::HEXAHEDRON);
  mesh.SetCurvature(2);
  mesh.Transform(Jiggle);
  return mesh;
}

// The kernels run on the mfem device named by ADGIA_TEST_DEVICE (the host
// if unset), so that a device build can run this test on its device.
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

// The two assemblies of one integrator compared on random vectors and on
// the diagonal; make() returns a new integrator each time it is called.
template <class Make>
void Compare(FiniteElementSpace& fes, Make make) {
  BilinearForm a(&fes);
  a.AddDomainIntegrator(make());
  a.Assemble();
  a.Finalize();

  BilinearForm b(&fes);
  b.SetAssemblyLevel(AssemblyLevel::PARTIAL);
  b.AddDomainIntegrator(make());
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

using Param = std::tuple<int, int>;  // (order, ordering)

class ElasticTensorPartialAssembly : public testing::TestWithParam<Param> {
 protected:
  void SetUp() override {
    order = std::get<0>(GetParam());
    ordering = std::get<1>(GetParam()) == 0 ? Ordering::byNODES
                                            : Ordering::byVDIM;
  }
  int order;
  Ordering::Type ordering;
};

TEST_P(ElasticTensorPartialAssembly, Isotropic) {
  auto mesh = CurvedHexMesh();
  H1_FECollection fec(order, 3);
  FiniteElementSpace fes(&mesh, &fec, 3, ordering);
  FunctionCoefficient lambda(
      [](const Vector& x) { return 1.3 + 0.4 * std::sin(2.0 * x[0] + x[2]); });
  FunctionCoefficient mu(
      [](const Vector& x) { return 0.8 + 0.3 * x[1] * x[1]; });
  IsotropicElasticTensorCoefficient C(3, lambda, mu);
  Compare(fes, [&]() { return new ElasticTensorIntegrator(C); });
}

TEST_P(ElasticTensorPartialAssembly, TransverselyIsotropic) {
  auto mesh = CurvedHexMesh();
  H1_FECollection fec(order, 3);
  FiniteElementSpace fes(&mesh, &fec, 3, ordering);
  ConstantCoefficient A(3.1), Cc(2.7), F(1.1), L(0.9), N(1.2);
  Vector x0(3);
  x0 = -0.3;
  RadialUnitVectorCoefficient axis(3, x0);
  TransverselyIsotropicElasticTensorCoefficient ti(3, A, Cc, F, L, N, axis);
  Compare(fes, [&]() { return new ElasticTensorIntegrator(ti); });
  // With a rule of the caller's.
  const IntegrationRule& ir = IntRules.Get(Geometry::CUBE, 2 * order + 2);
  Compare(fes, [&]() { return new ElasticTensorIntegrator(ti, &ir); });
}

TEST_P(ElasticTensorPartialAssembly, Mapped) {
  auto mesh = CurvedHexMesh();
  H1_FECollection fec(order, 3);
  FiniteElementSpace fes(&mesh, &fec, 3, ordering);
  const double c = 0.05;
  CallableDiffeomorphism xi(
      3,
      [c](const Vector& x, Vector& y) {
        for (int i = 0; i < 3; i++) {
          y(i) = x(i) + c * std::sin(kPi * x((i + 1) % 3));
        }
      },
      [c](const Vector& x, DenseMatrix& F) {
        F = 0.0;
        for (int i = 0; i < 3; i++) {
          F(i, i) = 1.0;
          F(i, (i + 1) % 3) = c * kPi * std::cos(kPi * x((i + 1) % 3));
        }
      });
  ConstantCoefficient A(3.1), Cc(2.7), F(1.1), L(0.9), N(1.2);
  Vector x0(3);
  x0 = -0.3;
  RadialUnitVectorCoefficient axis(3, x0);
  TransverselyIsotropicElasticTensorCoefficient ti(3, A, Cc, F, L, N, axis);
  const IntegrationRule& ir = IntRules.Get(Geometry::CUBE, 2 * order + 3);
  Compare(fes, [&]() { return new ElasticTensorIntegrator(ti, xi, &ir); });
}

// Tetrahedra: the kernel dense in the element, or libCEED's.
TEST(ElasticTensorPartialAssemblyDense, Tetrahedra) {
  for (int order = 1; order <= 3; order++) {
    auto mesh = Mesh::MakeCartesian3D(2, 2, 2, Element::TETRAHEDRON);
    mesh.SetCurvature(2);
    mesh.Transform(Jiggle);
    H1_FECollection fec(order, 3);
    FiniteElementSpace fes(&mesh, &fec, 3);
    FunctionCoefficient lambda(
        [](const Vector& x) { return 1.3 + 0.4 * std::sin(2.0 * x[0] + x[2]); });
    FunctionCoefficient mu(
        [](const Vector& x) { return 0.8 + 0.3 * x[1] * x[1]; });
    IsotropicElasticTensorCoefficient iso(3, lambda, mu);
    Compare(fes, [&]() { return new ElasticTensorIntegrator(iso); });
    ConstantCoefficient A(3.1), Cc(2.7), F(1.1), L(0.9), N(1.2);
    Vector x0(3);
    x0 = -0.3;
    RadialUnitVectorCoefficient axis(3, x0);
    TransverselyIsotropicElasticTensorCoefficient ti(3, A, Cc, F, L, N, axis);
    Compare(fes, [&]() { return new ElasticTensorIntegrator(ti); });
  }
}

INSTANTIATE_TEST_SUITE_P(Orders, ElasticTensorPartialAssembly,
                         testing::Combine(testing::Values(1, 2, 3, 4),
                                          testing::Values(0, 1)));

}  // namespace
