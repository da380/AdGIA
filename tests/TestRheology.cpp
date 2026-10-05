#include "TestCommon.hpp"

/*
  Tests for IsotropicMaxwellRheology: the derived coefficients evaluate to
  the right combinations of the inputs at quadrature points.
  For the purely elastic IsotropicElasticRheology and
  AnisotropicElasticRheology: unrelaxed moduli and stiffness against MFEM's
  ElasticityIntegrator, a branchless Maxwell body and each other, with the
  (no-op) relaxation weights; and the elastic limits of the Maxwell bodies.
  And for AnisotropicMaxwellRheology: the unrelaxed tensor, the branch
  moduli, and the stiffness objects of the two rheologies assembling the
  same matrix for an isotropic tensor, with and without relaxation weights.
*/

namespace {

// Evaluate c at every quadrature point of an order-3 rule on every element
// and apply `check(value, x)`.
template <class F>
void ForEachQuadraturePoint(Mesh& mesh, Coefficient& c, F check) {
  for (auto e = 0; e < mesh.GetNE(); e++) {
    auto* T = mesh.GetElementTransformation(e);
    const auto& ir = IntRules.Get(mesh.GetElementGeometry(e), 3);
    for (auto q = 0; q < ir.GetNPoints(); q++) {
      const auto& ip = ir.IntPoint(q);
      T->SetIntPoint(&ip);
      Vector x(mesh.Dimension());
      T->Transform(ip, x);
      check(c.Eval(*T, ip), x);
    }
  }
}

double Kappa(const Vector& x) { return 2.0 + x[0]; }
double MuInf(const Vector& x) { return 0.5 + 0.25 * x[x.Size() - 1]; }
double Mu1(const Vector& x) { return 1.0 + x[0] * x[0]; }
double Mu2(const Vector& x) { return 0.3 + x[1]; }

class RheologyTest : public testing::TestWithParam<int> {
 protected:
  void SetUp() override {
    dim = GetParam();
    mesh = std::make_unique<Mesh>(
        dim == 2 ? Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL)
                 : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON));
  }
  int dim = 2;
  std::unique_ptr<Mesh> mesh;
  FunctionCoefficient kappa{Kappa}, mu_inf{MuInf}, mu1{Mu1}, mu2{Mu2};
  ConstantCoefficient tau1{1.0}, tau2{100.0};
};

TEST_P(RheologyTest, UnrelaxedModuli) {
  std::vector<MaxwellBranch> branches{{&mu1, &tau1}, {&mu2, &tau2}};
  IsotropicMaxwellRheology r(dim, kappa, mu_inf, branches);

  EXPECT_EQ(r.SpaceDim(), dim);
  EXPECT_EQ(r.NumBranches(), 2);
  EXPECT_EQ(r.Branch(0).mu, &mu1);
  EXPECT_EQ(r.Branch(1).tau, &tau2);
  EXPECT_EQ(&r.BulkModulus(), &kappa);
  EXPECT_EQ(&r.LongTermShearModulus(), &mu_inf);

  ForEachQuadraturePoint(*mesh, r.UnrelaxedShearModulus(),
                         [](double v, const Vector& x) {
                           EXPECT_NEAR(v, MuInf(x) + Mu1(x) + Mu2(x), 1e-14);
                         });
  ForEachQuadraturePoint(*mesh, r.UnrelaxedLame(),
                         [this](double v, const Vector& x) {
                           const auto mu_u = MuInf(x) + Mu1(x) + Mu2(x);
                           EXPECT_NEAR(v, Kappa(x) - 2.0 * mu_u / dim, 1e-14);
                         });
}

TEST_P(RheologyTest, IsotropicElastic) {
  IsotropicElasticRheology r(dim, kappa, mu1);
  EXPECT_EQ(r.SpaceDim(), dim);
  EXPECT_EQ(r.NumBranches(), 0);
  EXPECT_TRUE(r.IsLinear());
  EXPECT_TRUE(r.TraceFreeInternalVariables());
  EXPECT_EQ(&r.BulkModulus(), &kappa);
  EXPECT_EQ(&r.ShearModulus(), &mu1);
  ForEachQuadraturePoint(*mesh, r.Lame(), [this](double v, const Vector& x) {
    EXPECT_NEAR(v, Kappa(x) - 2.0 * Mu1(x) / dim, 1e-14);
  });

  // The unrelaxed modulus is the isotropic Mandel tensor, and equals that
  // of a branchless Maxwell body with the same moduli.
  IsotropicMaxwellRheology maxwell(dim, kappa, mu1, {});
  auto C_ref =
      IsotropicElasticTensorCoefficient::FromBulkModulus(dim, kappa, mu1);
  const int ns = SymmetricTensorBasis::Size(dim);
  DenseMatrix Ca(ns), Cb(ns);
  for (auto e = 0; e < mesh->GetNE(); e++) {
    auto* T = mesh->GetElementTransformation(e);
    const auto& ir = IntRules.Get(mesh->GetElementGeometry(e), 3);
    for (auto q = 0; q < ir.GetNPoints(); q++) {
      const auto& ip = ir.IntPoint(q);
      T->SetIntPoint(&ip);
      r.UnrelaxedModulus(*T, ip, Ca);
      C_ref.Eval(Cb, *T, ip);
      Ca -= Cb;
      EXPECT_LT(Ca.MaxMaxNorm(), 1e-13 * Cb.MaxMaxNorm());
      maxwell.UnrelaxedModulus(*T, ip, Ca);
      Ca -= Cb;
      EXPECT_LT(Ca.MaxMaxNorm(), 1e-13 * Cb.MaxMaxNorm());
    }
  }
}

// Assemble the stiffness of `r` on `fes`, optionally after setting or
// clearing (empty) relaxation weights, and return the matrix.
SparseMatrix AssembleStiffness(FiniteElementSpace& fes, const Rheology& r,
                               bool set_and_clear_weights = false) {
  auto s = r.MakeStiffness();
  EXPECT_FALSE(s->IsRelaxed());
  if (set_and_clear_weights) {
    s->SetRelaxationWeights({});
    EXPECT_FALSE(s->IsRelaxed());
    s->ClearRelaxationWeights();
    EXPECT_FALSE(s->IsRelaxed());
  }
  BilinearForm a(&fes);
  s->AddIntegrators(a);
  a.Assemble();
  a.Finalize();
  return a.SpMat();
}

TEST_P(RheologyTest, ElasticStiffnessAgrees) {
  H1_FECollection fec(2, dim);
  FiniteElementSpace fes(mesh.get(), &fec, dim);

  // Reference: MFEM's integrator in (lambda, mu) form.
  IsotropicElasticRheology iso(dim, kappa, mu1);
  BilinearForm ref(&fes);
  ref.AddDomainIntegrator(new ElasticityIntegrator(iso.Lame(), mu1));
  ref.Assemble();
  ref.Finalize();
  const double norm = ref.SpMat().MaxNorm();
  EXPECT_GT(norm, 0.0);

  EXPECT_LT(MaxDiff(AssembleStiffness(fes, iso, true), ref.SpMat()),
            1e-13 * norm);

  // The same solid as a branchless Maxwell body.
  IsotropicMaxwellRheology maxwell(dim, kappa, mu1, {});
  EXPECT_LT(MaxDiff(AssembleStiffness(fes, maxwell), ref.SpMat()),
            1e-13 * norm);

  // And as an anisotropic elastic body with an isotropic tensor.
  auto C = IsotropicElasticTensorCoefficient::FromBulkModulus(dim, kappa, mu1);
  AnisotropicElasticRheology aniso(dim, C);
  EXPECT_EQ(aniso.SpaceDim(), dim);
  EXPECT_EQ(aniso.NumBranches(), 0);
  EXPECT_TRUE(aniso.IsLinear());
  EXPECT_FALSE(aniso.TraceFreeInternalVariables());
  EXPECT_EQ(&aniso.Tensor(), &C);
  EXPECT_LT(MaxDiff(AssembleStiffness(fes, aniso, true), ref.SpMat()),
            1e-13 * norm);
}

TEST_P(RheologyTest, ElasticLimits) {
  std::vector<MaxwellBranch> branches{{&mu1, &tau1}, {&mu2, &tau2}};
  IsotropicMaxwellRheology r(dim, kappa, mu_inf, branches);
  auto unrelaxed = r.UnrelaxedElastic();
  auto relaxed = r.LongTermElastic();
  EXPECT_EQ(unrelaxed.SpaceDim(), dim);
  EXPECT_EQ(&unrelaxed.BulkModulus(), &kappa);
  EXPECT_EQ(&unrelaxed.ShearModulus(), &r.UnrelaxedShearModulus());
  EXPECT_EQ(&relaxed.BulkModulus(), &kappa);
  EXPECT_EQ(&relaxed.ShearModulus(), &mu_inf);
  ForEachQuadraturePoint(*mesh, unrelaxed.Lame(),
                         [this](double v, const Vector& x) {
                           const auto mu_u = MuInf(x) + Mu1(x) + Mu2(x);
                           EXPECT_NEAR(v, Kappa(x) - 2.0 * mu_u / dim, 1e-14);
                         });

  auto C_inf =
      IsotropicElasticTensorCoefficient::FromBulkModulus(dim, kappa, mu_inf);
  ConstantCoefficient zero(0.0);
  auto C_1 = IsotropicElasticTensorCoefficient::FromBulkModulus(dim, zero, mu1);
  auto C_2 = IsotropicElasticTensorCoefficient::FromBulkModulus(dim, zero, mu2);
  std::vector<AnisotropicBranch> abranches{{&C_1, &tau1}, {&C_2, &tau2}};
  AnisotropicMaxwellRheology a(dim, C_inf, abranches);
  auto a_unrelaxed = a.UnrelaxedElastic();
  auto a_relaxed = a.LongTermElastic();
  EXPECT_EQ(&a_unrelaxed.Tensor(), &a.UnrelaxedTensor());
  EXPECT_EQ(&a_relaxed.Tensor(), &C_inf);

  // Both limits assemble the same stiffness as their isotropic twins.
  H1_FECollection fec(1, dim);
  FiniteElementSpace fes(mesh.get(), &fec, dim);
  auto Ku = AssembleStiffness(fes, unrelaxed),
       Kr = AssembleStiffness(fes, relaxed);
  const double nu = Ku.MaxNorm();
  EXPECT_LT(MaxDiff(AssembleStiffness(fes, a_unrelaxed), Ku), 1e-13 * nu);
  EXPECT_LT(MaxDiff(AssembleStiffness(fes, a_relaxed), Kr), 1e-13 * nu);
  EXPECT_LT(MaxDiff(AssembleStiffness(fes, r), Ku), 1e-13 * nu);
  EXPECT_GT(MaxDiff(Ku, Kr), 1e-3 * nu);
}

TEST_P(RheologyTest, MaxwellFactory) {
  auto r = IsotropicMaxwellRheology::Maxwell(dim, kappa, mu1, tau1);
  EXPECT_EQ(r.NumBranches(), 1);
  EXPECT_EQ(r.Branch(0).mu, &mu1);
  EXPECT_EQ(r.Branch(0).tau, &tau1);
  // mu_inf = 0, so mu_U = mu1 (the factory's owned zero must survive the
  // move out of the factory).
  ForEachQuadraturePoint(*mesh, r.LongTermShearModulus(),
                         [](double v, const Vector&) { EXPECT_EQ(v, 0.0); });
  ForEachQuadraturePoint(
      *mesh, r.UnrelaxedShearModulus(),
      [](double v, const Vector& x) { EXPECT_NEAR(v, Mu1(x), 1e-14); });
}

TEST_P(RheologyTest, AnisotropicTensors) {
  auto C_inf = IsotropicElasticTensorCoefficient::FromBulkModulus(dim, kappa,
                                                                  mu_inf);
  ConstantCoefficient zero(0.0);
  auto C_1 = IsotropicElasticTensorCoefficient::FromBulkModulus(dim, zero, mu1);
  auto C_2 = IsotropicElasticTensorCoefficient::FromBulkModulus(dim, zero, mu2);
  std::vector<AnisotropicBranch> branches{{&C_1, &tau1}, {&C_2, &tau2}};
  AnisotropicMaxwellRheology r(dim, C_inf, branches);
  EXPECT_EQ(r.SpaceDim(), dim);
  EXPECT_EQ(r.NumBranches(), 2);
  EXPECT_FALSE(r.TraceFreeInternalVariables());
  EXPECT_EQ(&r.RelaxationTime(1), &tau2);
  EXPECT_EQ(&r.LongTermTensor(), &C_inf);

  // C_U = C_inf + C_1 + C_2 = the isotropic tensor with mu_U.
  std::vector<MaxwellBranch> ibranches{{&mu1, &tau1}, {&mu2, &tau2}};
  IsotropicMaxwellRheology iso(dim, kappa, mu_inf, ibranches);
  auto C_u = IsotropicElasticTensorCoefficient::FromBulkModulus(
      dim, kappa, iso.UnrelaxedShearModulus());
  const int ns = SymmetricTensorBasis::Size(dim);
  DenseMatrix Ca(ns), Cb(ns), P(ns);
  SymmetricTensorBasis::DeviatoricProjector(dim, P);
  for (auto e = 0; e < mesh->GetNE(); e++) {
    auto* T = mesh->GetElementTransformation(e);
    const auto& ir = IntRules.Get(mesh->GetElementGeometry(e), 3);
    for (auto q = 0; q < ir.GetNPoints(); q++) {
      const auto& ip = ir.IntPoint(q);
      T->SetIntPoint(&ip);
      r.UnrelaxedTensor().Eval(Ca, *T, ip);
      C_u.Eval(Cb, *T, ip);
      Ca -= Cb;
      EXPECT_LT(Ca.MaxMaxNorm(), 1e-13 * Cb.MaxMaxNorm());
      // Branch moduli: the anisotropic C_1 equals the isotropic body's
      // 2 mu_1 P_dev, and the isotropic branch 1 is 2 mu_2 P_dev.
      r.BranchModulus(0, *T, ip, Ca);
      iso.BranchModulus(0, *T, ip, Cb);
      Ca -= Cb;
      EXPECT_LT(Ca.MaxMaxNorm(), 1e-13 * Cb.MaxMaxNorm());
      Vector x(dim);
      T->Transform(ip, x);
      iso.BranchModulus(1, *T, ip, Cb);
      Cb.Add(-2.0 * Mu2(x), P);
      EXPECT_LT(Cb.MaxMaxNorm(), 1e-13);
    }
  }
}

TEST_P(RheologyTest, StiffnessObjectsAgree) {
  H1_FECollection fec(1, dim);
  FiniteElementSpace fes(mesh.get(), &fec, dim);
  std::vector<MaxwellBranch> ibranches{{&mu1, &tau1}, {&mu2, &tau2}};
  IsotropicMaxwellRheology iso(dim, kappa, mu_inf, ibranches);
  auto C_inf = IsotropicElasticTensorCoefficient::FromBulkModulus(dim, kappa,
                                                                  mu_inf);
  ConstantCoefficient zero(0.0);
  auto C_1 = IsotropicElasticTensorCoefficient::FromBulkModulus(dim, zero, mu1);
  auto C_2 = IsotropicElasticTensorCoefficient::FromBulkModulus(dim, zero, mu2);
  std::vector<AnisotropicBranch> branches{{&C_1, &tau1}, {&C_2, &tau2}};
  AnisotropicMaxwellRheology aniso(dim, C_inf, branches);

  auto si = iso.MakeStiffness();
  auto sa = aniso.MakeStiffness();
  // Template forms holding the integrators; each assembly borrows them.
  BilinearForm ti(&fes), ta(&fes);
  si->AddIntegrators(ti);
  sa->AddIntegrators(ta);
  double norm_i = 0.0;
  auto assemble = [&]() {
    BilinearForm ai(&fes, &ti), aa(&fes, &ta);
    ai.Assemble();
    aa.Assemble();
    ai.Finalize();
    aa.Finalize();
    norm_i = ai.SpMat().MaxNorm();
    return MaxDiff(ai.SpMat(), aa.SpMat()) / norm_i;
  };
  EXPECT_FALSE(si->IsRelaxed());
  EXPECT_LT(assemble(), 1e-13);
  const double norm_u = norm_i;

  FunctionCoefficient b1([](const Vector& x) { return 0.2 + 0.5 * x[0]; });
  ConstantCoefficient b2(0.7);
  si->SetRelaxationWeights({&b1, &b2});
  sa->SetRelaxationWeights({&b1, &b2});
  EXPECT_TRUE(si->IsRelaxed());
  EXPECT_TRUE(sa->IsRelaxed());
  EXPECT_LT(assemble(), 1e-13);
  EXPECT_GT(std::abs(norm_i - norm_u), 1e-3 * norm_u);

  si->ClearRelaxationWeights();
  sa->ClearRelaxationWeights();
  EXPECT_LT(assemble(), 1e-13);
  EXPECT_NEAR(norm_i, norm_u, 1e-13 * norm_u);
}

INSTANTIATE_TEST_SUITE_P(Rheology, RheologyTest, testing::Values(2, 3));

}  // namespace
