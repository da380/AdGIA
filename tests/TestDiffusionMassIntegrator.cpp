#include <cstdlib>
#include <memory>
#include <numbers>

#include "TestCommon.hpp"

/*
  DiffusionMassIntegrator: a diffusion form and a mass term on a scalar
  space in one integrator.

  - Its element matrices assemble to the matrix of mfem's
    DiffusionIntegrator and MassIntegrator on the same rule.
  - Its partial assembly applies that matrix and gives its diagonal, on
    tetrahedra (the kernel dense in the element) and on hexahedra
    (sum-factorised), on curved meshes at orders 1 to 4, with and without
    the mass term, with the default rule and with a given one.

  Set ADGIA_TEST_DEVICE (e.g. to cuda, ceed-cpu) to run on a device.
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

using Param = std::tuple<bool, int, bool>;  // hexahedra, order, mass

class DiffusionMass : public testing::TestWithParam<Param> {};

TEST_P(DiffusionMass, MatchesMfemAndItsPartialAssembly) {
  const auto [hexahedra, order, mass] = GetParam();
  auto mesh = Mesh::MakeCartesian3D(2, 2, 2,
                                    hexahedra ? Element::HEXAHEDRON
                                              : Element::TETRAHEDRON);
  mesh.SetCurvature(2);
  mesh.Transform(Jiggle);
  H1_FECollection fec(order, 3);
  FiniteElementSpace fes(&mesh, &fec);
  FunctionCoefficient k(
      [](const Vector& x) { return 1.3 + 0.4 * std::sin(2.0 * x[0] + x[2]); });
  FunctionCoefficient s([](const Vector& x) { return 0.7 + 0.2 * x[1]; });

  for (const bool given_rule : {false, true}) {
    const IntegrationRule* ir =
        given_rule ? &IntRules.Get(mesh.GetTypicalElementGeometry(),
                                   2 * order + 2)
                   : nullptr;
    const auto make = [&]() {
      return new DiffusionMassIntegrator(k, mass ? &s : nullptr, ir);
    };
    BilinearForm a(&fes);
    a.AddDomainIntegrator(make());
    a.Assemble();
    a.Finalize();
    const double scale = a.SpMat().MaxNorm();

    if (given_rule) {
      BilinearForm m(&fes);
      auto* d = new DiffusionIntegrator(k);
      d->SetIntRule(ir);
      m.AddDomainIntegrator(d);
      if (mass) {
        auto* mi = new MassIntegrator(s);
        mi->SetIntRule(ir);
        m.AddDomainIntegrator(mi);
      }
      m.Assemble();
      m.Finalize();
      std::unique_ptr<SparseMatrix> D(Add(1.0, a.SpMat(), -1.0, m.SpMat()));
      EXPECT_LT(D->MaxNorm(), 1e-12 * scale);
    }

    BilinearForm b(&fes);
    b.SetAssemblyLevel(AssemblyLevel::PARTIAL);
    b.AddDomainIntegrator(make());
    b.Assemble();
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

INSTANTIATE_TEST_SUITE_P(Cases, DiffusionMass,
                         testing::Combine(testing::Bool(),
                                          testing::Values(1, 2, 3, 4),
                                          testing::Bool()));

}  // namespace
