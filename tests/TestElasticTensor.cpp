#include "TestCommon.hpp"

/*
  Pointwise tests for SymmetricTensorBasis and the elastic tensor
  coefficients, against closed-form Voigt matrices, explicit index
  contractions and rotated copies of the same tensor.
*/

namespace {

constexpr double kTol = 1e-12;

double MaxDiff(const DenseMatrix& A, const DenseMatrix& B) {
  if (A.Height() != B.Height() || A.Width() != B.Width() || A.Height() == 0) {
    return std::numeric_limits<double>::infinity();
  }
  DenseMatrix D(A);
  D -= B;
  return D.MaxMaxNorm();
}

// A random rotation (Gram-Schmidt of a random matrix, det = +1).
DenseMatrix RandomRotation(int dim) {
  auto A = RandomMatrix(dim);
  DenseMatrix R(dim);
  for (int c = 0; c < dim; c++) {
    Vector v(dim);
    for (int i = 0; i < dim; i++) {
      v[i] = A(i, c);
    }
    for (int p = 0; p < c; p++) {
      double dot = 0.0;
      for (int i = 0; i < dim; i++) {
        dot += R(i, p) * v[i];
      }
      for (int i = 0; i < dim; i++) {
        v[i] -= dot * R(i, p);
      }
    }
    v /= v.Norml2();
    for (int i = 0; i < dim; i++) {
      R(i, c) = v[i];
    }
  }
  if (R.Det() < 0.0) {
    for (int i = 0; i < dim; i++) {
      R(i, dim - 1) = -R(i, dim - 1);
    }
  }
  return R;
}

// A random symmetric positive definite Voigt matrix.
DenseMatrix RandomSPD(int n) {
  auto A = RandomMatrix(n);
  DenseMatrix S(n);
  MultAAt(A, S);
  for (int i = 0; i < n; i++) {
    S(i, i) += n;
  }
  return S;
}

// Evaluate a MatrixCoefficient at a fixed point near the centre of one
// element of a small mesh.
struct Point {
  explicit Point(int dim)
      : mesh(dim == 2 ? Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL)
                      : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON)) {
    T = mesh.GetElementTransformation(3);
    ip = Geometries.GetCenter(T->GetGeometryType());
    ip.x += 0.1;
    ip.y -= 0.07;
    T->SetIntPoint(&ip);
  }
  DenseMatrix Eval(MatrixCoefficient& c) {
    DenseMatrix K;
    c.Eval(K, *T, ip);
    return K;
  }
  Mesh mesh;
  ElementTransformation* T = nullptr;
  IntegrationPoint ip;
};

class ElasticTensorTest : public testing::TestWithParam<int> {
 protected:
  void SetUp() override {
    dim = GetParam();
    n = SymmetricTensorBasis::Size(dim);
  }
  int dim = 3, n = 6;
};

TEST_P(ElasticTensorTest, IndexConventions) {
  // Index/Component are inverse; Voigt indices are a permutation.
  std::vector<int> seen(n, 0), voigt(n, 0);
  for (int k = 0; k < dim; k++) {
    for (int j = k; j < dim; j++) {
      const int s = SymmetricTensorBasis::Index(dim, j, k);
      EXPECT_EQ(s, SymmetricTensorBasis::Index(dim, k, j));
      int jj, kk;
      SymmetricTensorBasis::Component(dim, s, jj, kk);
      EXPECT_EQ(jj, j);
      EXPECT_EQ(kk, k);
      seen[s]++;
      voigt[SymmetricTensorBasis::VoigtIndex(dim, j, k)]++;
    }
  }
  for (int s = 0; s < n; s++) {
    EXPECT_EQ(seen[s], 1);
    EXPECT_EQ(voigt[s], 1);
  }
  if (dim == 3) {
    EXPECT_EQ(SymmetricTensorBasis::VoigtIndex(3, 1, 2), 3);
    EXPECT_EQ(SymmetricTensorBasis::VoigtIndex(3, 0, 2), 4);
    EXPECT_EQ(SymmetricTensorBasis::VoigtIndex(3, 0, 1), 5);
    EXPECT_EQ(SymmetricTensorBasis::Index(3, 2, 1), 4);
  }
}

TEST_P(ElasticTensorTest, ConversionsRoundTrip) {
  auto Cv = RandomSPD(n);
  DenseMatrix Cm, Cv2, Cm2;
  SymmetricTensorBasis::FromVoigt(dim, Cv, Cm);
  SymmetricTensorBasis::ToVoigt(dim, Cm, Cv2);
  EXPECT_LT(MaxDiff(Cv, Cv2), kTol);

  std::vector<real_t> full(dim * dim * dim * dim);
  SymmetricTensorBasis::Unpack(dim, Cm, full.data());
  SymmetricTensorBasis::Pack(dim, full.data(), Cm2);
  EXPECT_LT(MaxDiff(Cm, Cm2), kTol);

  // Minor and major symmetries of the unpacked tensor.
  auto at = [&](int i, int j, int k, int l) {
    return full[((i * dim + j) * dim + k) * dim + l];
  };
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      for (int k = 0; k < dim; k++) {
        for (int l = 0; l < dim; l++) {
          EXPECT_NEAR(at(i, j, k, l), at(j, i, k, l), kTol);
          EXPECT_NEAR(at(i, j, k, l), at(i, j, l, k), kTol);
          EXPECT_NEAR(at(i, j, k, l), at(k, l, i, j), kTol);
        }
      }
    }
  }

  // Apply agrees with the contraction sigma_jk = C_jklm eps_lm.
  auto eps = RandomVector(n);
  Vector sig;
  SymmetricTensorBasis::Apply(Cm, eps, sig);
  for (int s = 0; s < n; s++) {
    int j, k;
    SymmetricTensorBasis::Component(dim, s, j, k);
    double v = 0.0;
    for (int l = 0; l < dim; l++) {
      for (int m = 0; m < dim; m++) {
        v += at(j, k, l, m) * eps[SymmetricTensorBasis::Index(dim, l, m)];
      }
    }
    EXPECT_NEAR(sig[s], v, kTol * (1.0 + std::abs(v)));
  }

  // The volumetric and deviatoric projectors are idempotent, mutually
  // orthogonal and complementary.
  DenseMatrix Pv, Pd, PP(n);
  SymmetricTensorBasis::VolumetricProjector(dim, Pv);
  SymmetricTensorBasis::DeviatoricProjector(dim, Pd);
  Mult(Pv, Pv, PP);
  EXPECT_LT(MaxDiff(PP, Pv), kTol);
  Mult(Pd, Pd, PP);
  EXPECT_LT(MaxDiff(PP, Pd), kTol);
  Mult(Pv, Pd, PP);
  EXPECT_LT(PP.MaxMaxNorm(), kTol);
  // Complementary: Pv + Pd = I.
  PP = Pv;
  PP += Pd;
  for (int i = 0; i < PP.Height(); i++) {
    PP(i, i) -= 1.0;
  }
  EXPECT_LT(PP.MaxMaxNorm(), kTol);
}

TEST_P(ElasticTensorTest, Isotropic) {
  const double lambda = 1.3, mu = 0.8;
  ConstantCoefficient lam(lambda), m(mu), kappa(lambda + 2.0 * mu / dim);
  IsotropicElasticTensorCoefficient iso(dim, lam, m);
  auto iso_k =
      IsotropicElasticTensorCoefficient::FromBulkModulus(dim, kappa, m);
  Point pt(dim);
  auto Cm = pt.Eval(iso);
  EXPECT_LT(MaxDiff(Cm, pt.Eval(iso_k)), kTol);

  // Textbook Voigt matrix.
  DenseMatrix Cv(n);
  Cv = 0.0;
  for (int a = 0; a < dim; a++) {
    for (int b = 0; b < dim; b++) {
      Cv(a, b) = lambda + (a == b ? 2.0 * mu : 0.0);
    }
  }
  for (int s = dim; s < n; s++) {
    Cv(s, s) = mu;
  }
  DenseMatrix Cm_ref;
  SymmetricTensorBasis::FromVoigt(dim, Cv, Cm_ref);
  EXPECT_LT(MaxDiff(Cm, Cm_ref), kTol);

  // Eigen-stiffnesses: d kappa on volumetric tensors, 2 mu on deviatoric.
  DenseMatrix Pv, Pd, X(n), Y;
  SymmetricTensorBasis::VolumetricProjector(dim, Pv);
  SymmetricTensorBasis::DeviatoricProjector(dim, Pd);
  Mult(Cm, Pv, X);
  Y = Pv;
  Y *= dim * (lambda + 2.0 * mu / dim);
  EXPECT_LT(MaxDiff(X, Y), kTol);
  Mult(Cm, Pd, X);
  Y = Pd;
  Y *= 2.0 * mu;
  EXPECT_LT(MaxDiff(X, Y), kTol);

  // The deviatoric split reproduces those parts.
  DeviatoricProjectionElasticTensorCoefficient dev(dim, iso, true),
      vol(dim, iso, false);
  Y = Pd;
  Y *= 2.0 * mu;
  EXPECT_LT(MaxDiff(pt.Eval(dev), Y), kTol);
  Y = Pv;
  Y *= dim * (lambda + 2.0 * mu / dim);
  EXPECT_LT(MaxDiff(pt.Eval(vol), Y), kTol);
}

TEST_P(ElasticTensorTest, TransverselyIsotropic) {
  const double A = 3.1, C = 2.7, F = 1.1, L = 0.9, N = 1.2;
  ConstantCoefficient cA(A), cC(C), cF(F), cL(L), cN(N);
  Point pt(dim);

  // Axis e_d: the canonical Voigt matrix.
  Vector ez(dim);
  ez = 0.0;
  ez[dim - 1] = 1.0;
  VectorConstantCoefficient axis_z(ez);
  TransverselyIsotropicElasticTensorCoefficient ti(dim, cA, cC, cF, cL, cN,
                                                   axis_z);
  DenseMatrix Cv;
  SymmetricTensorBasis::ToVoigt(dim, pt.Eval(ti), Cv);
  if (dim == 3) {
    EXPECT_NEAR(Cv(0, 0), A, kTol);
    EXPECT_NEAR(Cv(1, 1), A, kTol);
    EXPECT_NEAR(Cv(2, 2), C, kTol);
    EXPECT_NEAR(Cv(0, 2), F, kTol);
    EXPECT_NEAR(Cv(1, 2), F, kTol);
    EXPECT_NEAR(Cv(0, 1), A - 2.0 * N, kTol);
    EXPECT_NEAR(Cv(3, 3), L, kTol);
    EXPECT_NEAR(Cv(4, 4), L, kTol);
    EXPECT_NEAR(Cv(5, 5), N, kTol);
    EXPECT_NEAR(Cv(0, 3), 0.0, kTol);
    EXPECT_NEAR(Cv(3, 5), 0.0, kTol);
  } else {
    // Plane strain in the (x, z) plane: 11 = A, 22 = C, 12 = F, shear L.
    EXPECT_NEAR(Cv(0, 0), A, kTol);
    EXPECT_NEAR(Cv(1, 1), C, kTol);
    EXPECT_NEAR(Cv(0, 1), F, kTol);
    EXPECT_NEAR(Cv(2, 2), L, kTol);
    EXPECT_NEAR(Cv(0, 2), 0.0, kTol);
  }

  // Isotropic parameters with a random axis give the isotropic tensor.
  const double lambda = 1.3, mu = 0.8;
  ConstantCoefficient iA(lambda + 2.0 * mu), iF(lambda), iM(mu), lam(lambda);
  auto nr = RandomVector(dim);
  VectorConstantCoefficient axis_r(nr);
  TransverselyIsotropicElasticTensorCoefficient ti_iso(dim, iA, iA, iF, iM, iM,
                                                       axis_r);
  IsotropicElasticTensorCoefficient iso(dim, lam, iM);
  EXPECT_LT(MaxDiff(pt.Eval(ti_iso), pt.Eval(iso)), kTol);

  // Rotation covariance: TI with axis R e_d equals the rotated e_d tensor.
  auto R = RandomRotation(dim);
  Vector Rez(dim);
  R.Mult(ez, Rez);
  VectorConstantCoefficient axis_R(Rez);
  TransverselyIsotropicElasticTensorCoefficient ti_R(dim, cA, cC, cF, cL, cN,
                                                     axis_R);
  MatrixConstantCoefficient Rc(R);
  RotatedElasticTensorCoefficient rotated(ti, Rc);
  EXPECT_LT(MaxDiff(pt.Eval(ti_R), pt.Eval(rotated)), kTol);

  // FromVelocities with PREM-like numbers is positive definite and has
  // A = rho vph^2 in the 11 slot for the axis e_d.
  ConstantCoefficient rho(3.3), vpv(8.0), vph(8.2), vsv(4.4), vsh(4.6),
      eta(0.95);
  auto prem = TransverselyIsotropicElasticTensorCoefficient::FromVelocities(
      dim, rho, vpv, vph, vsv, vsh, eta, axis_z);
  auto Cp = pt.Eval(prem);
  for (int trial = 0; trial < 5; trial++) {
    auto x = RandomVector(n);
    Vector y(n);
    Cp.Mult(x, y);
    EXPECT_GT(x * y, 0.0);
  }
  SymmetricTensorBasis::ToVoigt(dim, Cp, Cv);
  EXPECT_NEAR(Cv(0, 0), 3.3 * 8.2 * 8.2, 1e-10);  // A = rho vph^2
}

TEST_P(ElasticTensorTest, RotationOfGeneralTensor) {
  // Unpack-rotate-Pack equals the Mandel orthogonal transform, and Q is
  // orthogonal.
  auto Cv = RandomSPD(n);
  DenseMatrix Cm, Q, QQt(n), tmp(n), Cq(n);
  SymmetricTensorBasis::FromVoigt(dim, Cv, Cm);
  auto R = RandomRotation(dim);
  SymmetricTensorBasis::RotationMatrix(dim, R, Q);
  MultAAt(Q, QQt);
  DenseMatrix I(n);
  I = 0.0;
  for (int s = 0; s < n; s++) {
    I(s, s) = 1.0;
  }
  EXPECT_LT(MaxDiff(QQt, I), kTol);
  Mult(Q, Cm, tmp);
  MultABt(tmp, Q, Cq);

  std::vector<real_t> full(dim * dim * dim * dim), rot(full.size());
  SymmetricTensorBasis::Unpack(dim, Cm, full.data());
  auto at = [&](const std::vector<real_t>& v, int i, int j, int k, int l) {
    return v[((i * dim + j) * dim + k) * dim + l];
  };
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      for (int k = 0; k < dim; k++) {
        for (int l = 0; l < dim; l++) {
          double v = 0.0;
          for (int a = 0; a < dim; a++) {
            for (int b = 0; b < dim; b++) {
              for (int c = 0; c < dim; c++) {
                for (int d = 0; d < dim; d++) {
                  v += R(i, a) * R(j, b) * R(k, c) * R(l, d) *
                       at(full, a, b, c, d);
                }
              }
            }
          }
          rot[((i * dim + j) * dim + k) * dim + l] = v;
        }
      }
    }
  }
  DenseMatrix Cr;
  SymmetricTensorBasis::Pack(dim, rot.data(), Cr);
  EXPECT_LT(MaxDiff(Cq, Cr), kTol * Cm.MaxMaxNorm());

  // The coefficient classes agree with this.
  MatrixConstantCoefficient Cvc(Cv), Rc(R);
  VoigtElasticTensorCoefficient voigt(dim, Cvc);
  RotatedElasticTensorCoefficient rotated(voigt, Rc);
  Point pt(dim);
  EXPECT_LT(MaxDiff(pt.Eval(voigt), Cm), kTol);
  EXPECT_LT(MaxDiff(pt.Eval(rotated), Cr), kTol * Cm.MaxMaxNorm());
}

// RadialUnitVectorCoefficient returns the unit position vector.
TEST_P(ElasticTensorTest, RadialAxis) {
  Point pt(dim);
  RadialUnitVectorCoefficient radial(dim);
  Vector n, x(dim);
  radial.Eval(n, *pt.T, pt.ip);
  pt.T->Transform(pt.ip, x);
  EXPECT_NEAR(n.Norml2(), 1.0, kTol);
  x /= x.Norml2();
  x -= n;
  EXPECT_LT(x.Norml2(), kTol);
}

INSTANTIATE_TEST_SUITE_P(ElasticTensor, ElasticTensorTest,
                         testing::Values(2, 3));

// The 2-D tensor with an in-plane axis is the plane-strain restriction of
// the 3-D tensor with the same axis (compared through the static Build).
TEST(ElasticTensorPlaneStrain, MatchesThreeD) {
  const double A = 3.1, C = 2.7, F = 1.1, L = 0.9, N = 1.2;
  ConstantCoefficient cA(A), cC(C), cF(F), cL(L), cN(N);
  Vector n2(2), n3(3);
  n2[0] = 0.6;
  n2[1] = 0.8;
  n3[0] = 0.6;
  n3[1] = 0.8;
  n3[2] = 0.0;
  DenseMatrix C2, C3;
  TransverselyIsotropicElasticTensorCoefficient::Build(2, A, C, F, L, N, n2,
                                                       C2);
  TransverselyIsotropicElasticTensorCoefficient::Build(3, A, C, F, L, N, n3,
                                                       C3);
  const int map2[3][2] = {{0, 0}, {1, 0}, {1, 1}};
  for (int s = 0; s < 3; s++) {
    for (int t = 0; t < 3; t++) {
      const int s3 = SymmetricTensorBasis::Index(3, map2[s][0], map2[s][1]);
      const int t3 = SymmetricTensorBasis::Index(3, map2[t][0], map2[t][1]);
      EXPECT_NEAR(C2(s, t), C3(s3, t3), kTol);
    }
  }
}

}  // namespace
