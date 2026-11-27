#include <cstdlib>
#include <memory>
#include <numbers>

#include "TestCommon.hpp"

/*
  GradientCouplingOperator: (phi, v) -> int c grad phi . v between a scalar
  and a vector H1 space, with no matrix. Against the matrix of mfem's
  GradientIntegrator: the action and the transpose's, on curved meshes of
  tetrahedra (the kernel dense in the element) and hexahedra
  (sum-factorised), the two spaces of one order and of different orders,
  with the default rule and with a given one.

  Set ADGIA_TEST_DEVICE (e.g. to cuda) to run on a device.
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

using Param = std::tuple<bool, int, int>;  // hexahedra, scalar order, vector order

class GradientCoupling : public testing::TestWithParam<Param> {};

TEST_P(GradientCoupling, IsTheMatrixOfTheGradientIntegrator) {
  const auto [hexahedra, scalar_order, vector_order] = GetParam();
  auto mesh = Mesh::MakeCartesian3D(2, 2, 2,
                                    hexahedra ? Element::HEXAHEDRON
                                              : Element::TETRAHEDRON);
  mesh.SetCurvature(2);
  mesh.Transform(Jiggle);
  H1_FECollection scalar_fec(scalar_order, 3), vector_fec(vector_order, 3);
  FiniteElementSpace scalar_fes(&mesh, &scalar_fec);
  FiniteElementSpace vector_fes(&mesh, &vector_fec, 3);
  FunctionCoefficient c(
      [](const Vector& x) { return 1.3 + 0.4 * std::sin(2.0 * x[0] + x[2]); });

  for (const bool given_rule : {false, true}) {
    const IntegrationRule* ir =
        given_rule ? &IntRules.Get(mesh.GetTypicalElementGeometry(),
                                   scalar_order + vector_order + 2)
                   : nullptr;
    MixedBilinearForm a(&scalar_fes, &vector_fes);
    auto* integ = new GradientIntegrator(c);
    if (ir) {
      integ->SetIntRule(ir);
    }
    a.AddDomainIntegrator(integ);
    a.Assemble();
    a.Finalize();
    const double scale = a.SpMat().MaxNorm();

    GradientCouplingOperator C(scalar_fes, vector_fes, c, ir);
    ASSERT_EQ(C.Height(), vector_fes.GetVSize());
    ASSERT_EQ(C.Width(), scalar_fes.GetVSize());
    for (int trial = 0; trial < 3; trial++) {
      auto phi = RandomVector(scalar_fes.GetVSize());
      Vector ua(vector_fes.GetVSize()), ub(vector_fes.GetVSize());
      a.SpMat().Mult(phi, ua);
      C.Mult(phi, ub);
      ub -= ua;
      EXPECT_LT(ub.Normlinf(), 1e-11 * scale * phi.Normlinf());

      auto u = RandomVector(vector_fes.GetVSize());
      Vector pa(scalar_fes.GetVSize()), pb(scalar_fes.GetVSize());
      a.SpMat().MultTranspose(u, pa);
      C.MultTranspose(u, pb);
      pb -= pa;
      EXPECT_LT(pb.Normlinf(), 1e-11 * scale * u.Normlinf());
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Cases, GradientCoupling,
    testing::Values(Param{false, 2, 2}, Param{false, 3, 3}, Param{false, 2, 3},
                    Param{false, 4, 3}, Param{true, 2, 2}, Param{true, 3, 3},
                    Param{true, 2, 3}, Param{true, 4, 3}));

}  // namespace
