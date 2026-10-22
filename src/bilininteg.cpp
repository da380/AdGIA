#include "AdGIA/bilininteg.hpp"

#include <cmath>
#include <numbers>

namespace AdGIA {

const mfem::IntegrationRule& DomainVectorScalarIntegrator::GetRule(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    const mfem::ElementTransformation& Trans) {
  const auto order = trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW();
  return mfem::IntRules.Get(trial_fe.GetGeomType(), order);
}

void DomainVectorScalarIntegrator::AssembleElementMatrix2(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat) {
  using namespace mfem;
  auto space_dim = Trans.GetSpaceDim();
  auto trial_dof = trial_fe.GetDof();
  auto test_dof = test_fe.GetDof();

  auto same_shape = &trial_fe == &test_fe;

  elmat.SetSize(space_dim * test_dof, trial_dof);
  elmat = 0.;

#ifdef MFEM_THREAD_SAFE
  Vector trial_shape, test_shape, qv;
  DenseMatrix part_elmat;
#endif
  trial_shape.SetSize(trial_dof);
  qv.SetSize(space_dim);
  part_elmat.SetSize(test_dof, trial_dof);

  if (same_shape) {
    test_shape.NewDataAndSize(trial_shape.GetData(), test_dof);
  } else {
    test_shape.SetSize(test_dof);
  }

  const auto* ir = IntRule ? IntRule : &GetRule(trial_fe, test_fe, Trans);

  for (auto i = 0; i < ir->GetNPoints(); i++) {
    const auto& ip = ir->IntPoint(i);
    Trans.SetIntPoint(&ip);
    auto w = Trans.Weight() * ip.weight;

    trial_fe.CalcShape(ip, trial_shape);
    if (!same_shape) {
      test_fe.CalcShape(ip, test_shape);
    }

    if (map_) {
      w *= map_->Jacobian(Trans, ip);
    }

    QV->Eval(qv, Trans, ip);
    MultVWt(test_shape, trial_shape, part_elmat);
    for (auto j = 0; j < space_dim; j++) {
      elmat.AddMatrix(w * qv(j), part_elmat, test_dof * j, 0);
    }
  }
}

const mfem::IntegrationRule& DomainVectorGradScalarIntegrator::GetRule(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    const mfem::ElementTransformation& Trans) {
  const auto order =
      trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW() - 1;
  return mfem::IntRules.Get(trial_fe.GetGeomType(), order);
}

void DomainVectorGradScalarIntegrator::AssembleElementMatrix2(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat) {
  using namespace mfem;
  auto space_dim = Trans.GetSpaceDim();
  auto trial_dof = trial_fe.GetDof();
  auto test_dof = test_fe.GetDof();

  elmat.SetSize(space_dim * test_dof, trial_dof);
  elmat = 0.;

#ifdef MFEM_THREAD_SAFE
  Vector test_shape, qv;
  DenseMatrix trial_dshape, part_elmat, qm, tm;
  DenseMatrix F_, dtmp_;
#endif
  test_shape.SetSize(test_dof);
  trial_dshape.SetSize(trial_dof, space_dim);
  part_elmat.SetSize(test_dof, trial_dof);

  if (QM) {
    qm.SetSize(space_dim);
    tm.SetSize(trial_dof, space_dim);
  } else if (QV) {
    qv.SetSize(space_dim);
  }

  const auto* ir = IntRule ? IntRule : &GetRule(trial_fe, test_fe, Trans);

  for (auto i = 0; i < ir->GetNPoints(); i++) {
    const auto& ip = ir->IntPoint(i);
    Trans.SetIntPoint(&ip);
    auto w = Trans.Weight() * ip.weight;

    test_fe.CalcShape(ip, test_shape);
    trial_fe.CalcPhysDShape(Trans, trial_dshape);

    if (map_) {
      map_->EvalGradient(F_, Trans, ip);
      w *= F_.Det();
      F_.Invert();
      dtmp_ = trial_dshape;
      Mult(dtmp_, F_, trial_dshape);
    }

    if (QM) {
      QM->Eval(qm, Trans, ip);
      MultABt(trial_dshape, qm, tm);
      for (auto j = 0; j < space_dim; j++) {
        auto tm_column = Vector(tm.GetColumn(j), trial_dof);
        MultVWt(test_shape, tm_column, part_elmat);
        elmat.AddMatrix(w, part_elmat, j * test_dof, 0);
      }
    } else if (QV) {
      QV->Eval(qv, Trans, ip);
      qv *= w;
      for (auto j = 0; j < space_dim; j++) {
        auto trial_dshape_column = Vector(trial_dshape.GetColumn(j), trial_dof);
        MultVWt(test_shape, trial_dshape_column, part_elmat);
        elmat.AddMatrix(qv(j), part_elmat, j * test_dof, 0);
      }
    } else {
      if (Q) {
        w *= Q->Eval(Trans, ip);
      }
      for (auto j = 0; j < space_dim; j++) {
        auto trial_dshape_column = Vector(trial_dshape.GetColumn(j), trial_dof);
        MultVWt(test_shape, trial_dshape_column, part_elmat);
        elmat.AddMatrix(w, part_elmat, j * test_dof, 0);
      }
    }
  }
}

const mfem::IntegrationRule& DomainDivVectorScalarIntegrator::GetRule(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    const mfem::ElementTransformation& Trans) {
  const auto order =
      trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW() - 1;
  return mfem::IntRules.Get(trial_fe.GetGeomType(), order);
}

void DomainDivVectorScalarIntegrator::AssembleElementMatrix2(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat) {
  auto space_dim = Trans.GetSpaceDim();
  auto trial_dof = trial_fe.GetDof();
  auto test_dof = test_fe.GetDof();

  elmat.SetSize(space_dim * test_dof, trial_dof);
  elmat = 0.;

#ifdef MFEM_THREAD_SAFE
  auto test_dshape = mfem::DenseMatrix();
  auto trial_shape = mfem::Vector();
  auto part_elmat = mfem::DenseMatrix();
  auto F_ = mfem::DenseMatrix();
  auto dtmp_ = mfem::DenseMatrix();
#endif
  test_dshape.SetSize(test_dof, space_dim);
  trial_shape.SetSize(trial_dof);
  part_elmat.SetSize(test_dof, trial_dof);

  const auto* ir = IntRule ? IntRule : &GetRule(trial_fe, test_fe, Trans);

  for (auto i = 0; i < ir->GetNPoints(); i++) {
    const auto& ip = ir->IntPoint(i);
    Trans.SetIntPoint(&ip);

    test_fe.CalcPhysDShape(Trans, test_dshape);
    trial_fe.CalcShape(ip, trial_shape);

    auto w = Trans.Weight() * ip.weight;
    if (Q) {
      w *= Q->Eval(Trans, ip);
    }

    if (map_) {
      map_->EvalGradient(F_, Trans, ip);
      w *= F_.Det();
      F_.Invert();
      dtmp_ = test_dshape;
      mfem::Mult(dtmp_, F_, test_dshape);
    }

    for (auto j = 0; j < space_dim; j++) {
      auto test_dshape_column =
          mfem::Vector(test_dshape.GetColumn(j), test_dof);
      mfem::MultVWt(test_dshape_column, trial_shape, part_elmat);
      elmat.AddMatrix(w, part_elmat, j * test_dof, 0);
    }
  }
}

const mfem::IntegrationRule& DomainDivVectorDivVectorIntegrator::GetRule(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    const mfem::ElementTransformation& Trans) {
  const auto order =
      trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW() - 2;
  return mfem::IntRules.Get(trial_fe.GetGeomType(), order);
}

void DomainDivVectorDivVectorIntegrator::AssembleElementMatrix2(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto space_dim = Trans.GetSpaceDim();
  auto trial_dof = trial_fe.GetDof();
  auto test_dof = test_fe.GetDof();

  auto same_spaces = &test_fe == &trial_fe;

  elmat.SetSize(space_dim * test_dof, space_dim * trial_dof);
  elmat = 0.;

#ifdef MFEM_THREAD_SAFE
  auto trial_dshape = DenseMatrix();
  auto test_dshape = DenseMatrix();
  auto F_ = DenseMatrix();
  auto dtmp_ = DenseMatrix();
#endif
  trial_dshape.SetSize(trial_dof, space_dim);

  if (same_spaces) {
    test_dshape.Reset(trial_dshape.GetData(), test_dof, space_dim);
  } else {
    test_dshape.SetSize(test_dof, space_dim);
  }

  const auto* ir = IntRule ? IntRule : &GetRule(trial_fe, test_fe, Trans);

  const auto test_dshape_vector =
      Vector(test_dshape.GetData(), space_dim * test_dof);
  const auto trial_dshape_vector =
      Vector(trial_dshape.GetData(), space_dim * trial_dof);

  for (auto i = 0; i < ir->GetNPoints(); i++) {
    const auto& ip = ir->IntPoint(i);
    Trans.SetIntPoint(&ip);

    trial_fe.CalcPhysDShape(Trans, trial_dshape);
    if (!same_spaces) {
      test_fe.CalcPhysDShape(Trans, test_dshape);
    }

    auto w = Trans.Weight() * ip.weight;
    if (Q) {
      w *= Q->Eval(Trans, ip);
    }

    if (map_) {
      map_->EvalGradient(F_, Trans, ip);
      w *= F_.Det();
      F_.Invert();
      dtmp_ = trial_dshape;
      Mult(dtmp_, F_, trial_dshape);
      if (!same_spaces) {
        dtmp_ = test_dshape;
        Mult(dtmp_, F_, test_dshape);
      }
    }

    AddMult_a_VWt(w, test_dshape_vector, trial_dshape_vector, elmat);
  }
}

const mfem::IntegrationRule& DomainVectorGradVectorIntegrator::GetRule(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    const mfem::ElementTransformation& Trans) {
  const auto order =
      trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW() - 1;
  return mfem::IntRules.Get(trial_fe.GetGeomType(), order);
}

void DomainVectorGradVectorIntegrator::AssembleElementMatrix2(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto space_dim = Trans.GetSpaceDim();
  auto trial_dof = trial_fe.GetDof();
  auto test_dof = test_fe.GetDof();

  elmat.SetSize(space_dim * test_dof, space_dim * trial_dof);
  elmat = 0.;

#ifdef MFEM_THREAD_SAFE
  Vector qv, test_shape;
  DenseMatrix trial_dshape, left_elmat, right_elmat_trans, part_elmat;
  DenseMatrix F_, dtmp_;
#endif

  qv.SetSize(space_dim);
  right_elmat_trans.SetSize(space_dim * trial_dof, trial_dof);
  right_elmat_trans = 0.;

  const auto& trial_nodes = trial_fe.GetNodes();
  for (auto i = 0; i < trial_dof; i++) {
    const auto& ip = trial_nodes.IntPoint(i);
    Trans.SetIntPoint(&ip);
    QV->Eval(qv, Trans, ip);
    for (auto j = 0; j < space_dim; j++) {
      right_elmat_trans(i + trial_dof * j, i) = qv(j);
    }
  }

  part_elmat.SetSize(test_dof, trial_dof);
  left_elmat.SetSize(space_dim * test_dof, trial_dof);
  left_elmat = 0.;

  trial_dshape.SetSize(trial_dof, space_dim);
  test_shape.SetSize(test_dof);

  const auto* ir = IntRule ? IntRule : &GetRule(trial_fe, test_fe, Trans);

  for (auto i = 0; i < ir->GetNPoints(); i++) {
    const auto& ip = ir->IntPoint(i);
    Trans.SetIntPoint(&ip);
    auto w = Trans.Weight() * ip.weight;

    trial_fe.CalcPhysDShape(Trans, trial_dshape);
    test_fe.CalcShape(ip, test_shape);

    if (Q) {
      w *= Q->Eval(Trans, ip);
    }

    if (map_) {
      map_->EvalGradient(F_, Trans, ip);
      w *= F_.Det();
      F_.Invert();
      dtmp_ = trial_dshape;
      Mult(dtmp_, F_, trial_dshape);
    }

    for (auto j = 0; j < space_dim; j++) {
      auto trial_dshape_column = Vector(trial_dshape.GetColumn(j), trial_dof);
      MultVWt(test_shape, trial_dshape_column, part_elmat);
      left_elmat.AddMatrix(w, part_elmat, test_dof * j, 0);
    }
  }
  // right_elmat_trans has one nonzero per row, (i + trial_dof j, i), so the
  // product with its transpose scales column i of left_elmat into column
  // i + trial_dof j of elmat.
  for (auto j = 0; j < space_dim; j++) {
    for (auto i = 0; i < trial_dof; i++) {
      const auto scale = right_elmat_trans(i + trial_dof * j, i);
      for (auto r = 0; r < left_elmat.Height(); r++) {
        elmat(r, i + trial_dof * j) = left_elmat(r, i) * scale;
      }
    }
  }
}

const mfem::IntegrationRule& DomainVectorDivVectorIntegrator::GetRule(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    const mfem::ElementTransformation& Trans) {
  const auto order =
      trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW() - 1;
  return mfem::IntRules.Get(trial_fe.GetGeomType(), order);
}

void DomainVectorDivVectorIntegrator::AssembleElementMatrix2(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto space_dim = Trans.GetSpaceDim();
  auto trial_dof = trial_fe.GetDof();
  auto test_dof = test_fe.GetDof();

  elmat.SetSize(space_dim * test_dof, space_dim * trial_dof);
  elmat = 0.;

#ifdef MFEM_THREAD_SAFE
  Vector qv, test_shape;
  DenseMatrix trial_dshape, part_elmat;
  DenseMatrix F_, dtmp_;
#endif

  qv.SetSize(space_dim);
  part_elmat.SetSize(test_dof, trial_dof);

  trial_dshape.SetSize(trial_dof, space_dim);
  test_shape.SetSize(test_dof);

  const auto* ir = IntRule ? IntRule : &GetRule(trial_fe, test_fe, Trans);

  for (auto i = 0; i < ir->GetNPoints(); i++) {
    const auto& ip = ir->IntPoint(i);
    Trans.SetIntPoint(&ip);
    auto w = Trans.Weight() * ip.weight;

    if (map_) {
      map_->EvalGradient(F_, Trans, ip);
      w *= F_.Det();
      F_.Invert();
    }

    QV->Eval(qv, Trans, ip);
    qv *= w;

    test_fe.CalcShape(ip, test_shape);
    trial_fe.CalcPhysDShape(Trans, trial_dshape);

    if (map_) {
      dtmp_ = trial_dshape;
      Mult(dtmp_, F_, trial_dshape);
    }

    for (auto k = 0; k < space_dim; k++) {
      auto trial_dshape_column = Vector(trial_dshape.GetColumn(k), trial_dof);
      MultVWt(test_shape, trial_dshape_column, part_elmat);
      for (auto j = 0; j < space_dim; j++) {
        elmat.AddMatrix(qv(j), part_elmat, j * test_dof, k * trial_dof);
      }
    }
  }
}

const mfem::IntegrationRule& DomainMatrixDeformationGradientIntegrator::GetRule(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    const mfem::ElementTransformation& Trans) {
  const auto order =
      trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW() - 1;
  return mfem::IntRules.Get(trial_fe.GetGeomType(), order);
}

void DomainMatrixDeformationGradientIntegrator::AssembleElementMatrix2(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto space_dim = Trans.GetSpaceDim();
  auto trial_dof = trial_fe.GetDof();
  auto test_dof = test_fe.GetDof();

  auto vectorIndex = VectorIndex(space_dim, trial_dof);
  auto matrixIndex = MatrixIndex(space_dim, test_dof);

  elmat.SetSize(matrixIndex.Size(), vectorIndex.Size());
  elmat = 0.;

  const auto* ir = IntRule ? IntRule : &GetRule(trial_fe, test_fe, Trans);

#ifdef MFEM_THREAD_SAFE
  Vector test_shape;
  DenseMatrix trial_dshape, part_elmat;
#endif
  test_shape.SetSize(test_dof);
  trial_dshape.SetSize(trial_dof, space_dim);
  part_elmat.SetSize(test_dof, trial_dof);

  for (auto i = 0; i < ir->GetNPoints(); i++) {
    const auto& ip = ir->IntPoint(i);
    Trans.SetIntPoint(&ip);
    auto w = Trans.Weight() * ip.weight;

    trial_fe.CalcPhysDShape(Trans, trial_dshape);
    test_fe.CalcShape(ip, test_shape);

    if (Q) {
      w *= Q->Eval(Trans, ip);
    }

    for (auto k = 0; k < space_dim; k++) {
      auto trial_dshape_column = Vector(trial_dshape.GetColumn(k), trial_dof);
      MultVWt(test_shape, trial_dshape_column, part_elmat);
      for (auto j = 0; j < space_dim; j++) {
        elmat.AddMatrix(w, part_elmat, matrixIndex.Offset(j, k),
                        vectorIndex.Offset(j));
      }
    }
  }
}

const mfem::IntegrationRule& DomainSymmetricMatrixStrainIntegrator::GetRule(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    const mfem::ElementTransformation& Trans) {
  const auto order =
      trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW() - 1;
  return mfem::IntRules.Get(trial_fe.GetGeomType(), order);
}

void DomainSymmetricMatrixStrainIntegrator::AssembleElementMatrix2(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto space_dim = Trans.GetSpaceDim();
  auto trial_dof = trial_fe.GetDof();
  auto test_dof = test_fe.GetDof();

  auto vectorIndex = VectorIndex(space_dim, trial_dof);
  auto matrixIndex = SymmetricMatrixIndex(space_dim, test_dof);

  elmat.SetSize(matrixIndex.Size(), vectorIndex.Size());
  elmat = 0.;

#ifdef MFEM_THREAD_SAFE
  Vector test_shape;
  DenseMatrix trial_dshape, part_elmat;
#endif
  test_shape.SetSize(test_dof);
  trial_dshape.SetSize(trial_dof, space_dim);
  part_elmat.SetSize(test_dof, trial_dof);

  const auto* ir = IntRule ? IntRule : &GetRule(trial_fe, test_fe, Trans);

  for (auto i = 0; i < ir->GetNPoints(); i++) {
    const auto& ip = ir->IntPoint(i);
    Trans.SetIntPoint(&ip);
    auto w = Trans.Weight() * ip.weight;

    if (Q) {
      w *= Q->Eval(Trans, ip);
    }

    trial_fe.CalcPhysDShape(Trans, trial_dshape);
    test_fe.CalcShape(ip, test_shape);

    for (auto k = 0; k < space_dim; k++) {
      auto trial_dshape_column = Vector(trial_dshape.GetColumn(k), trial_dof);
      MultVWt(test_shape, trial_dshape_column, part_elmat);
      for (auto j = 0; j < space_dim; j++) {
        elmat.AddMatrix(w, part_elmat, matrixIndex.Offset(j, k),
                        vectorIndex.Offset(j));
      }
    }
  }
}

const mfem::IntegrationRule&
DomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator::GetRule(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    const mfem::ElementTransformation& Trans) {
  const auto order =
      trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW() - 1;
  return mfem::IntRules.Get(trial_fe.GetGeomType(), order);
}

void DomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator::
    AssembleElementMatrix2(const mfem::FiniteElement& trial_fe,
                           const mfem::FiniteElement& test_fe,
                           mfem::ElementTransformation& Trans,
                           mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto space_dim = Trans.GetSpaceDim();
  auto trial_dof = trial_fe.GetDof();
  auto test_dof = test_fe.GetDof();

  auto vectorIndex = VectorIndex(space_dim, trial_dof);
  auto matrixIndex = TraceFreeSymmetricMatrixIndex(space_dim, test_dof);

  elmat.SetSize(matrixIndex.Size(), vectorIndex.Size());
  elmat = 0.;

#ifdef MFEM_THREAD_SAFE
  Vector test_shape;
  DenseMatrix trial_dshape, part_elmat;
#endif
  test_shape.SetSize(test_dof);
  trial_dshape.SetSize(trial_dof, space_dim);
  part_elmat.SetSize(test_dof, trial_dof);

  const auto* ir = IntRule ? IntRule : &GetRule(trial_fe, test_fe, Trans);

  for (auto i = 0; i < ir->GetNPoints(); i++) {
    const auto& ip = ir->IntPoint(i);
    Trans.SetIntPoint(&ip);
    auto w = Trans.Weight() * ip.weight;

    if (Q) {
      w *= Q->Eval(Trans, ip);
    }

    trial_fe.CalcPhysDShape(Trans, trial_dshape);
    test_fe.CalcShape(ip, test_shape);

    for (auto k = 0; k < space_dim - 1; k++) {
      auto trial_dshape_column = Vector(trial_dshape.GetColumn(k), trial_dof);
      MultVWt(test_shape, trial_dshape_column, part_elmat);

      for (auto j = 0; j < space_dim; j++) {
        elmat.AddMatrix(w, part_elmat, matrixIndex.Offset(j, k),
                        vectorIndex.Offset(j));
      }
    }

    auto k = space_dim - 1;
    auto trial_dshape_column = Vector(trial_dshape.GetColumn(k), trial_dof);
    MultVWt(test_shape, trial_dshape_column, part_elmat);

    for (auto j = 0; j < space_dim - 1; j++) {
      elmat.AddMatrix(w, part_elmat, matrixIndex.Offset(j, k),
                      vectorIndex.Offset(j));
      elmat.AddMatrix(-w, part_elmat, matrixIndex.Offset(j, j),
                      vectorIndex.Offset(k));
    }
  }
}

void ElasticTensorIntegrator::StrainDisplacementMatrix(
    int dim, const mfem::DenseMatrix& gshape, mfem::DenseMatrix& B) {
  using namespace mfem;
  const auto dof = gshape.Height();
  const auto uidx = VectorIndex(dim, dof);
  const auto sidx = SymmetricMatrixIndex(dim, dof);
  const real_t inv_sqrt2 = 1.0 / std::numbers::sqrt2_v<real_t>;
  B.SetSize(sidx.ComponentSize(), uidx.Size());
  B = 0.0;
  for (auto k = 0; k < dim; k++) {
    for (auto j = k; j < dim; j++) {
      const auto s = sidx.ComponentOffset(j, k);
      if (j == k) {
        for (auto i = 0; i < dof; i++) {
          B(s, uidx(i, j)) = gshape(i, j);
        }
      } else {
        for (auto i = 0; i < dof; i++) {
          B(s, uidx(i, j)) = inv_sqrt2 * gshape(i, k);
          B(s, uidx(i, k)) = inv_sqrt2 * gshape(i, j);
        }
      }
    }
  }
}

void ElasticTensorIntegrator::AssembleElementMatrix(
    const mfem::FiniteElement& el, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  using namespace mfem;
  const auto dof = el.GetDof();
  const auto dim = el.GetDim();
  const auto n = SymmetricMatrixIndex(dim, dof).ComponentSize();
  MFEM_VERIFY(dim == Trans.GetSpaceDim(),
              "ElasticTensorIntegrator: manifold elements are not "
              "supported.");
  MFEM_VERIFY(C_->GetHeight() == n && C_->GetWidth() == n,
              "ElasticTensorIntegrator: the tensor coefficient must be "
              "n_s x n_s with n_s = d(d+1)/2.");

#ifdef MFEM_THREAD_SAFE
  DenseMatrix dshape_, gshape_, B_, Cq_, CB_;
  DenseMatrix F_, gshape_map_;
#endif
  dshape_.SetSize(dof, dim);
  gshape_.SetSize(dof, dim);
  CB_.SetSize(n, dim * dof);
  elmat.SetSize(dof * dim);
  elmat = 0.0;

  if (map_) {
    F_.SetSize(dim);
    gshape_map_.SetSize(dof, dim);
  }

  const IntegrationRule* ir = IntRule;
  if (ir == nullptr) {
    ir = &IntRules.Get(el.GetGeomType(), 2 * Trans.OrderGrad(&el));
  }

  for (auto q = 0; q < ir->GetNPoints(); q++) {
    const auto& ip = ir->IntPoint(q);
    Trans.SetIntPoint(&ip);
    el.CalcDShape(ip, dshape_);
    Mult(dshape_, Trans.InverseJacobian(), gshape_);
    auto w = ip.weight * Trans.Weight();
    if (map_) {
      // Pull-back: derivatives w.r.t. the mapped coordinates and the
      // Jacobian in the weight; the assembly below is unchanged.
      map_->EvalGradient(F_, Trans, ip);
      w *= F_.Det();
      F_.Invert();
      Mult(gshape_, F_, gshape_map_);
      StrainDisplacementMatrix(dim, gshape_map_, B_);
    } else {
      StrainDisplacementMatrix(dim, gshape_, B_);
    }
    C_->Eval(Cq_, Trans, ip);
    Mult(Cq_, B_, CB_);
    AddMult_a_AtB(w, B_, CB_, elmat);
  }
}

void GeometricStiffnessIntegrator::AssembleElementMatrix(
    const mfem::FiniteElement& el, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  using namespace mfem;
  const auto dof = el.GetDof();
  const auto dim = el.GetDim();
  MFEM_VERIFY(dim == Trans.GetSpaceDim(),
              "GeometricStiffnessIntegrator: manifold elements are not "
              "supported.");
  MFEM_VERIFY(S_->GetHeight() == dim && S_->GetWidth() == dim,
              "GeometricStiffnessIntegrator: the stress coefficient must be "
              "d x d.");

#ifdef MFEM_THREAD_SAFE
  DenseMatrix dshape_, gshape_, Sq_, tmp_, G_;
  DenseMatrix F_, gshape_map_;
#endif
  dshape_.SetSize(dof, dim);
  gshape_.SetSize(dof, dim);
  Sq_.SetSize(dim);
  tmp_.SetSize(dof, dim);
  G_.SetSize(dof);
  G_ = 0.0;
  if (map_) {
    F_.SetSize(dim);
    gshape_map_.SetSize(dof, dim);
  }

  const IntegrationRule* ir = IntRule;
  if (ir == nullptr) {
    ir = &IntRules.Get(el.GetGeomType(), 2 * Trans.OrderGrad(&el));
  }

  for (auto q = 0; q < ir->GetNPoints(); q++) {
    const auto& ip = ir->IntPoint(q);
    Trans.SetIntPoint(&ip);
    el.CalcDShape(ip, dshape_);
    Mult(dshape_, Trans.InverseJacobian(), gshape_);
    auto w = ip.weight * Trans.Weight();
    const DenseMatrix* g = &gshape_;
    if (map_) {
      // Relabelling pull-back: derivatives w.r.t. the mapped coordinates
      // and the Jacobian in the weight.
      map_->EvalGradient(F_, Trans, ip);
      w *= F_.Det();
      F_.Invert();
      Mult(gshape_, F_, gshape_map_);
      g = &gshape_map_;
    }
    S_->Eval(Sq_, Trans, ip);
    Mult(*g, Sq_, tmp_);
    AddMult_a_ABt(w, tmp_, *g, G_);
  }

  // One copy of G per displacement component.
  const auto uidx = VectorIndex(dim, dof);
  elmat.SetSize(dof * dim);
  elmat = 0.0;
  for (auto k = 0; k < dim; k++) {
    for (auto i = 0; i < dof; i++) {
      for (auto j = 0; j < dof; j++) {
        elmat(uidx(i, k), uidx(j, k)) = G_(i, j);
      }
    }
  }
}

void MaterialStiffnessIntegrator::StrainDisplacementMatrix(
    int dim, const mfem::DenseMatrix& gshape, const mfem::DenseMatrix& F,
    mfem::DenseMatrix& B) {
  using namespace mfem;
  const auto dof = gshape.Height();
  const auto uidx = VectorIndex(dim, dof);
  const auto sidx = SymmetricMatrixIndex(dim, dof);
  const real_t inv_sqrt2 = 1.0 / std::numbers::sqrt2_v<real_t>;
  B.SetSize(sidx.ComponentSize(), uidx.Size());
  B = 0.0;
  // sym(F^T Du)_{AB} = (F_{kA} d_B u_k + F_{kB} d_A u_k) / 2, Mandel
  // scaled: every displacement component contributes to every row.
  for (auto A = 0; A < dim; A++) {
    for (auto Bb = 0; Bb <= A; Bb++) {
      const auto s = sidx.ComponentOffset(A, Bb);
      if (A == Bb) {
        for (auto k = 0; k < dim; k++) {
          for (auto i = 0; i < dof; i++) {
            B(s, uidx(i, k)) = F(k, A) * gshape(i, A);
          }
        }
      } else {
        for (auto k = 0; k < dim; k++) {
          for (auto i = 0; i < dof; i++) {
            B(s, uidx(i, k)) =
                inv_sqrt2 * (F(k, A) * gshape(i, Bb) + F(k, Bb) * gshape(i, A));
          }
        }
      }
    }
  }
}

void MaterialStiffnessIntegrator::AssembleElementMatrix(
    const mfem::FiniteElement& el, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  using namespace mfem;
  const auto dof = el.GetDof();
  const auto dim = el.GetDim();
  const auto n = SymmetricMatrixIndex(dim, dof).ComponentSize();
  MFEM_VERIFY(dim == Trans.GetSpaceDim(),
              "MaterialStiffnessIntegrator: manifold elements are not "
              "supported.");
  MFEM_VERIFY(C_->GetHeight() == n && C_->GetWidth() == n,
              "MaterialStiffnessIntegrator: the tensor coefficient must be "
              "n_s x n_s with n_s = d(d+1)/2.");

#ifdef MFEM_THREAD_SAFE
  DenseMatrix dshape_, gshape_, B_, Cq_, CB_, F_;
#endif
  dshape_.SetSize(dof, dim);
  gshape_.SetSize(dof, dim);
  F_.SetSize(dim);
  CB_.SetSize(n, dim * dof);
  elmat.SetSize(dof * dim);
  elmat = 0.0;

  const IntegrationRule* ir = IntRule;
  if (ir == nullptr) {
    ir = &IntRules.Get(el.GetGeomType(), 2 * Trans.OrderGrad(&el));
  }

  for (auto q = 0; q < ir->GetNPoints(); q++) {
    const auto& ip = ir->IntPoint(q);
    Trans.SetIntPoint(&ip);
    el.CalcDShape(ip, dshape_);
    Mult(dshape_, Trans.InverseJacobian(), gshape_);
    // The equilibrium mapping enters the strain operator alone: no
    // Jacobian factor (the strain energy is per referential volume).
    map_->EvalGradient(F_, Trans, ip);
    StrainDisplacementMatrix(dim, gshape_, F_, B_);
    const auto w = ip.weight * Trans.Weight();
    C_->Eval(Cq_, Trans, ip);
    Mult(Cq_, B_, CB_);
    AddMult_a_AtB(w, B_, CB_, elmat);
  }
}

void ReferentialGravityIntegrator::AssembleElementMatrix(
    const mfem::FiniteElement& el, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  using namespace mfem;
  const auto dof = el.GetDof();
  const auto dim = el.GetDim();
  MFEM_VERIFY(dim == Trans.GetSpaceDim(),
              "ReferentialGravityIntegrator: manifold elements are not "
              "supported.");

#ifdef MFEM_THREAD_SAFE
  DenseMatrix dshape_, gshape_, F_, a_, M_, P_, ag_;
  Vector g0v_, w_, beta_, gamma_;
#endif
  dshape_.SetSize(dof, dim);
  gshape_.SetSize(dof, dim);
  F_.SetSize(dim);
  a_.SetSize(dim);
  M_.SetSize(dof, dim);
  P_.SetSize(dof);
  ag_.SetSize(dof, dim);
  g0v_.SetSize(dim);
  w_.SetSize(dim);
  beta_.SetSize(dof);
  gamma_.SetSize(dim);
  const auto uidx = VectorIndex(dim, dof);
  elmat.SetSize(dof * dim);
  elmat = 0.0;

  const IntegrationRule* ir = IntRule;
  if (ir == nullptr) {
    ir = &IntRules.Get(el.GetGeomType(), 2 * Trans.OrderGrad(&el));
  }

  for (auto q = 0; q < ir->GetNPoints(); q++) {
    const auto& ip = ir->IntPoint(q);
    Trans.SetIntPoint(&ip);
    el.CalcDShape(ip, dshape_);
    Mult(dshape_, Trans.InverseJacobian(), gshape_);
    const auto wq = scale_ * ip.weight * Trans.Weight();

    map_->EvalGradient(F_, Trans, ip);
    const auto J = F_.Det();
    F_.Invert();  // F_ now holds F_e^{-1}
    MultAAt(F_, a_);
    a_ *= J;  // a_e = J F^{-1} F^{-T}
    g0_->Eval(g0v_, Trans, ip);
    a_.Mult(g0v_, w_);               // w = a_e g0
    const auto c0 = g0v_ * w_;       // <a_e g0, g0>
    Mult(gshape_, F_, M_);           // M(a,k) = grad(phi_a) . f_k = tr H
    gshape_.Mult(w_, beta_);         // beta_a = grad(phi_a) . w
    F_.MultTranspose(g0v_, gamma_);  // gamma_k = f_k . g0
    Mult(gshape_, a_, ag_);          // a_e grad(phi_a)
    MultABt(ag_, gshape_, P_);       // P(a,b) = grad(phi_a) . a_e grad(phi_b)

    // <a''(u,v) g0, g0> for the rank-one H of each basis pair (a,k),(b,l):
    //   c0 [M_ak M_bl - M_al M_bk]
    //   - 2 M_ak beta_b gamma_l - 2 M_bl beta_a gamma_k
    //   + 2 M_al beta_b gamma_k + 2 M_bk beta_a gamma_l
    //   + 2 gamma_k gamma_l P_ab.
    for (auto k = 0; k < dim; k++) {
      for (auto a = 0; a < dof; a++) {
        const auto row = uidx(a, k);
        for (auto l = 0; l < dim; l++) {
          for (auto b = 0; b < dof; b++) {
            const auto val = c0 * (M_(a, k) * M_(b, l) - M_(a, l) * M_(b, k)) -
                             2.0 * (M_(a, k) * beta_(b) * gamma_(l) +
                                    M_(b, l) * beta_(a) * gamma_(k)) +
                             2.0 * (M_(a, l) * beta_(b) * gamma_(k) +
                                    M_(b, k) * beta_(a) * gamma_(l)) +
                             2.0 * gamma_(k) * gamma_(l) * P_(a, b);
            elmat(row, uidx(b, l)) += wq * val;
          }
        }
      }
    }
  }
}

void ReferentialGravityCouplingIntegrator::AssembleElementMatrix2(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat) {
  using namespace mfem;
  const auto dof_p = trial_fe.GetDof();  // scalar potential space
  const auto dof_u = test_fe.GetDof();   // vector displacement space
  const auto dim = test_fe.GetDim();
  MFEM_VERIFY(dim == Trans.GetSpaceDim(),
              "ReferentialGravityCouplingIntegrator: manifold elements are "
              "not supported.");

#ifdef MFEM_THREAD_SAFE
  DenseMatrix dshape_u_, gshape_u_, dshape_p_, gshape_p_, F_, a_, M_, agp_;
  DenseMatrix AGP_, FW_;
  Vector g0v_, w_, beta_, gamma_, wg_;
#endif
  dshape_u_.SetSize(dof_u, dim);
  gshape_u_.SetSize(dof_u, dim);
  dshape_p_.SetSize(dof_p, dim);
  gshape_p_.SetSize(dof_p, dim);
  F_.SetSize(dim);
  a_.SetSize(dim);
  M_.SetSize(dof_u, dim);
  g0v_.SetSize(dim);
  w_.SetSize(dim);
  beta_.SetSize(dof_u);
  gamma_.SetSize(dim);
  agp_.SetSize(dof_u, dim);
  AGP_.SetSize(dof_u, dof_p);
  FW_.SetSize(dof_p, dim);
  wg_.SetSize(dof_p);
  const auto uidx = VectorIndex(dim, dof_u);
  elmat.SetSize(dof_u * dim, dof_p);
  elmat = 0.0;

  const IntegrationRule* ir = IntRule;
  if (ir == nullptr) {
    ir = &IntRules.Get(
        trial_fe.GetGeomType(),
        trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderGrad(&test_fe));
  }

  for (auto q = 0; q < ir->GetNPoints(); q++) {
    const auto& ip = ir->IntPoint(q);
    Trans.SetIntPoint(&ip);
    trial_fe.CalcDShape(ip, dshape_p_);
    Mult(dshape_p_, Trans.InverseJacobian(), gshape_p_);
    test_fe.CalcDShape(ip, dshape_u_);
    Mult(dshape_u_, Trans.InverseJacobian(), gshape_u_);
    const auto wq = scale_ * ip.weight * Trans.Weight();

    map_->EvalGradient(F_, Trans, ip);
    const auto J = F_.Det();
    F_.Invert();
    MultAAt(F_, a_);
    a_ *= J;
    g0_->Eval(g0v_, Trans, ip);
    a_.Mult(g0v_, w_);
    Mult(gshape_u_, F_, M_);
    gshape_u_.Mult(w_, beta_);
    F_.MultTranspose(g0v_, gamma_);

    // a_e grad(phi_a), and its pairing with grad(psi_c), once per point.
    Mult(gshape_u_, a_, agp_);
    MultABt(agp_, gshape_p_, AGP_);  // AGP(a,c) = a_e grad phi_a . grad psi_c
    gshape_p_.Mult(w_, wg_);         // wg_c = w . grad psi_c
    // f_k . grad psi_c: column k of F^{-1} dotted with the gradient.
    Mult(gshape_p_, F_, FW_);

    // <a'(phi_a e_k) g0, grad psi_c>
    //   = M_ak (w . grad psi_c) - beta_a (f_k . grad psi_c)
    //     - gamma_k (a_e grad phi_a . grad psi_c).
    for (auto c = 0; c < dof_p; c++) {
      for (auto k = 0; k < dim; k++) {
        for (auto a = 0; a < dof_u; a++) {
          const auto val =
              M_(a, k) * wg_(c) - beta_(a) * FW_(c, k) - gamma_(k) * AGP_(a, c);
          elmat(uidx(a, k), c) += wq * val;
        }
      }
    }
  }
}

const mfem::IntegrationRule& TransformedDiffusionIntegrator::GetRule(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    const mfem::ElementTransformation& Trans) {
  const auto order = trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW();
  return mfem::IntRules.Get(trial_fe.GetGeomType(), order);
}

void TransformedDiffusionIntegrator::AssembleElementMatrix2(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto dim = Trans.GetSpaceDim();
  auto trial_dof = trial_fe.GetDof();
  auto test_dof = test_fe.GetDof();

  auto same_spaces = &test_fe == &trial_fe;

  elmat.SetSize(test_dof, trial_dof);
  elmat = 0.;

#ifdef MFEM_THREAD_SAFE
  Vector fs, df, x;
  DenseMatrix trial_dshape, test_dshape, xis, F, a, trial_dshape_trans;
#endif
  trial_dshape.SetSize(trial_dof, dim);

  if (same_spaces) {
    test_dshape.Reset(trial_dshape.GetData(), test_dof, dim);
  } else {
    test_dshape.SetSize(test_dof, dim);
  }

  if (Q || QV || D) {
    F.SetSize(dim, dim);
  }

  if (Q || QV || QM || D) {
    a.SetSize(dim, dim);
    trial_dshape_trans.SetSize(trial_dof, dim);
  }

  if (Q) {
    // Evaluate radial function at trial nodes.
    x.SetSize(dim);
    df.SetSize(dim);
    fs.SetSize(trial_dof);
    const auto& ir = trial_fe.GetNodes();
    for (auto i = 0; i < ir.GetNPoints(); i++) {
      const auto& ip = ir.IntPoint(i);
      Trans.SetIntPoint(&ip);
      fs(i) = Q->Eval(Trans, ip);
    }
  }

  if (QV) {
    // Evaluate mapping at all trial nodes.
    const auto& ir = trial_fe.GetNodes();
    QV->Eval(xis, Trans, ir);
  }

  const auto* ir = IntRule ? IntRule : &GetRule(trial_fe, test_fe, Trans);

  for (auto i = 0; i < ir->GetNPoints(); i++) {
    const auto& ip = ir->IntPoint(i);
    Trans.SetIntPoint(&ip);

    trial_fe.CalcPhysDShape(Trans, trial_dshape);
    if (!same_spaces) {
      test_fe.CalcPhysDShape(Trans, test_dshape);
    }

    auto w = Trans.Weight() * ip.weight;

    if (Q) {
      // Compute F at the integration point from the radial mapping.
      Trans.Transform(ip, x);
      auto f = Q->Eval(Trans, ip);
      trial_dshape.MultTranspose(fs, df);
      for (auto k = 0; k < dim; k++) {
        for (auto j = 0; j < dim; j++) {
          F(j, k) = x(j) * df(k);
        }
        F(k, k) += f;
      }
    }

    if (QV) {
      // Compute F at the integration point from the mapping.
      Mult(xis, trial_dshape, F);
    }

    if (D) {
      // F directly from the mapping.
      D->EvalGradient(F, Trans, ip);
    }

    if (Q || QV || D) {
      // Form the matrix a = J F^{-1} F^{-T}
      auto J = F.Det();
      F.Invert();
      MultAAt(F, a);
      a *= J;
    }

    if (QM) {
      // Evaluate a.
      QM->Eval(a, Trans, ip);
    }

    // Form the contribution to the local element matrix.
    if (Q || QV || QM || D) {
      Mult(trial_dshape, a, trial_dshape_trans);
      AddMult_a_ABt(w, test_dshape, trial_dshape_trans, elmat);
    } else {
      AddMult_a_ABt(w, test_dshape, trial_dshape, elmat);
    }
  }
}

void DeformationGradientInterpolator::AssembleElementMatrix2(
    const mfem::FiniteElement& in_fe, const mfem::FiniteElement& out_fe,
    mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto space_dim = in_fe.GetDim();
  auto in_dof = in_fe.GetDof();
  auto out_dof = out_fe.GetDof();

  auto vectorIndex = VectorIndex(space_dim, in_dof);
  auto matrixIndex = MatrixIndex(space_dim, out_dof);

#ifdef MFEM_THREAD_SAFE
  DenseMatrix dshape;
#endif
  dshape.SetSize(in_dof, space_dim);
  elmat.SetSize(matrixIndex.Size(), vectorIndex.Size());
  elmat = 0.;

  const auto& nodes = out_fe.GetNodes();

  for (auto i = 0; i < out_dof; i++) {
    const auto& ip = nodes.IntPoint(i);
    Trans.SetIntPoint(&ip);
    in_fe.CalcPhysDShape(Trans, dshape);
    for (auto l = 0; l < space_dim; l++) {
      for (auto j = 0; j < in_dof; j++) {
        for (auto k = 0; k < space_dim; k++) {
          elmat(matrixIndex(i, k, l), vectorIndex(j, k)) += dshape(j, l);
        }
      }
    }
  }
}

void StrainInterpolator::AssembleElementMatrix2(
    const mfem::FiniteElement& in_fe, const mfem::FiniteElement& out_fe,
    mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat) {
  using namespace mfem;

  constexpr real_t half = 0.5;

  auto space_dim = in_fe.GetDim();
  auto in_dof = in_fe.GetDof();
  auto out_dof = out_fe.GetDof();

  auto vectorIndex = VectorIndex(space_dim, in_dof);
  auto matrixIndex = SymmetricMatrixIndex(space_dim, out_dof);

#ifdef MFEM_THREAD_SAFE
  DenseMatrix dshape;
#endif
  dshape.SetSize(in_dof, space_dim);
  elmat.SetSize(matrixIndex.Size(), vectorIndex.Size());
  elmat = 0.;

  const auto& nodes = out_fe.GetNodes();

  for (auto i = 0; i < out_dof; i++) {
    const IntegrationPoint& ip = nodes.IntPoint(i);
    Trans.SetIntPoint(&ip);
    in_fe.CalcPhysDShape(Trans, dshape);

    for (auto l = 0; l < space_dim; l++) {
      for (auto j = 0; j < in_dof; j++) {
        for (auto k = l; k < space_dim; k++) {
          elmat(matrixIndex(i, k, l), vectorIndex(j, l)) += half * dshape(j, k);
          elmat(matrixIndex(i, k, l), vectorIndex(j, k)) += half * dshape(j, l);
        }
      }
    }
  }
}

void DeviatoricStrainInterpolator::AssembleElementMatrix2(
    const mfem::FiniteElement& in_fe, const mfem::FiniteElement& out_fe,
    mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto space_dim = in_fe.GetDim();
  auto in_dof = in_fe.GetDof();
  auto out_dof = out_fe.GetDof();

  auto vectorIndex = VectorIndex(space_dim, in_dof);
  auto matrixIndex = TraceFreeSymmetricMatrixIndex(space_dim, out_dof);

#ifdef MFEM_THREAD_SAFE
  auto dshape = DenseMatrix();
#endif
  dshape.SetSize(in_dof, space_dim);
  elmat.SetSize(matrixIndex.Size(), vectorIndex.Size());
  elmat = 0.;

  constexpr real_t half = 0.5;
  const auto space_dim_inverse = static_cast<real_t>(1) / space_dim;

  const auto& nodes = out_fe.GetNodes();

  for (auto i = 0; i < out_dof; i++) {
    const IntegrationPoint& ip = nodes.IntPoint(i);
    Trans.SetIntPoint(&ip);
    in_fe.CalcPhysDShape(Trans, dshape);

    for (auto l = 0; l < space_dim - 1; l++) {
      for (auto j = 0; j < in_dof; j++) {
        for (auto k = l; k < space_dim; k++) {
          elmat(matrixIndex(i, k, l), vectorIndex(j, k)) += half * dshape(j, l);
          elmat(matrixIndex(i, k, l), vectorIndex(j, l)) += half * dshape(j, k);
        }
        for (auto k = 0; k < space_dim; k++) {
          elmat(matrixIndex(i, l, l), vectorIndex(j, k)) -=
              space_dim_inverse * dshape(j, k);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Boundary normal integrators

namespace {

// Unit normal of a boundary element at the current integration point of
// Trans (CalcOrtho of the Jacobian, normalised); false if degenerate.
bool BoundaryUnitNormal(mfem::ElementTransformation& Trans,
                        mfem::Vector& normal) {
  normal.SetSize(Trans.GetSpaceDim());
  mfem::CalcOrtho(Trans.Jacobian(), normal);
  const mfem::real_t nrm = normal.Norml2();
  if (nrm <= 0.0) {
    return false;
  }
  normal /= nrm;
  return true;
}

}  // namespace

const mfem::IntegrationRule& BoundaryNormalNormalIntegrator::GetRule(
    const mfem::FiniteElement& el, const mfem::ElementTransformation& Trans) {
  const auto order = 2 * el.GetOrder() + Trans.OrderW();
  return mfem::IntRules.Get(el.GetGeomType(), order);
}

void BoundaryNormalNormalIntegrator::AssembleElementMatrix(
    const mfem::FiniteElement& el, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  using namespace mfem;
  const auto dim = Trans.GetSpaceDim();
  const auto dof = el.GetDof();

#ifdef MFEM_THREAD_SAFE
  Vector shape, normal, nshape;
  Vector nu_;
#endif
  shape.SetSize(dof);
  nshape.SetSize(dim * dof);
  elmat.SetSize(dim * dof);
  elmat = 0.0;

  const auto* ir = IntRule ? IntRule : &GetRule(el, Trans);
  for (auto q = 0; q < ir->GetNPoints(); q++) {
    const auto& ip = ir->IntPoint(q);
    Trans.SetIntPoint(&ip);
    if (!BoundaryUnitNormal(Trans, normal)) {
      continue;
    }
    if (map_) {
      // Two unit normals, one measure: (nu.u)(nu.u')/|nu| dS.
      map_->MapNormal(normal, Trans, ip, nu_);
      const auto s = nu_.Norml2();
      if (s <= 0.0) {
        continue;
      }
      normal = nu_;
      normal /= std::sqrt(s);
    }
    el.CalcShape(ip, shape);
    for (auto d = 0; d < dim; d++) {
      for (auto i = 0; i < dof; i++) {
        nshape[i + d * dof] = shape[i] * normal[d];
      }
    }
    auto w = ip.weight * Trans.Weight();
    if (Q) {
      w *= Q->Eval(Trans, ip);
    }
    AddMult_a_VVt(w, nshape, elmat);
  }
}

namespace {

/// The covariant direction slot of the interface kernels:
/// dir = P_T F^{-1} (1 - nu nu^T / |nu|^2). The deformed-face kernel
/// sanitises its direction with its own tangential projector, and the
/// pullback of that projector is the nu-orthogonal projection applied
/// before F^{-1} (F^{-1} maps nu-perp onto the referential tangent plane,
/// by Nanson's formula). P_T F^{-1} alone agrees on the constraint set
/// (nu . jump = 0) but is not invariant under face-fixing shears of F off
/// it; the projected form is, which preserves the discrete
/// change-of-variables identity through the slipping interface
/// (doc/slip_interface.tex). Unmapped kernels are unchanged: there nu = n
/// and the extra projector is contained in P_T.
void SlipDirection(const mfem::DenseMatrix& PT, const mfem::DenseMatrix& Fi,
                   const mfem::Vector& nu, bool mapped,
                   mfem::DenseMatrix& dir) {
  using namespace mfem;
  const int dim = PT.Height();
  MFEM_ASSERT(dim <= 3, "SlipDirection: space dimension at most 3.");
  dir.SetSize(dim);
  Mult(PT, Fi, dir);
  if (!mapped) {
    return;
  }
  const real_t n2 = nu * nu;
  if (n2 <= 0.0) {
    return;
  }
  // dir <- dir (1 - nu nu^T / n2), row by row.
  real_t row[3];
  for (int i = 0; i < dim; i++) {
    for (int k = 0; k < dim; k++) {
      row[k] = dir(i, k);
    }
    for (int j = 0; j < dim; j++) {
      real_t sum = 0.0;
      for (int k = 0; k < dim; k++) {
        sum += row[k] * ((k == j ? 1.0 : 0.0) - nu[k] * nu[j] / n2);
      }
      dir(i, j) = sum;
    }
  }
}

/// Shared surface machinery of the slip-interface kernels: physical
/// tangential shape gradients gshape, tangential projector P_T, the
/// (mapped) Nanson normal nu and F^{-1} at the current point. Returns
/// false where the normal is unavailable.
bool SlipSurfaceData(const mfem::FiniteElement& el,
                     mfem::ElementTransformation& Trans,
                     const mfem::IntegrationPoint& ip, Diffeomorphism* map,
                     mfem::Vector& shape, mfem::DenseMatrix& dshape,
                     mfem::DenseMatrix& gshape, mfem::Vector& normal,
                     mfem::Vector& nu, mfem::DenseMatrix& F,
                     mfem::DenseMatrix& Fi, mfem::DenseMatrix& PT,
                     mfem::DenseMatrix& Jt, mfem::DenseMatrix& JtJ,
                     mfem::DenseMatrix& JtJi) {
  using namespace mfem;
  const auto dim = Trans.GetSpaceDim();
  const auto sdim = el.GetDim();
  Trans.SetIntPoint(&ip);
  if (!BoundaryUnitNormal(Trans, normal)) {
    return false;
  }
  el.CalcShape(ip, shape);
  el.CalcDShape(ip, dshape);
  const DenseMatrix& Js = Trans.Jacobian();
  JtJ.SetSize(sdim);
  MultAtB(Js, Js, JtJ);
  JtJi.SetSize(sdim);
  CalcInverse(JtJ, JtJi);
  // Jt = (J^T J)^{-1} J^T
  Jt.SetSize(sdim, dim);
  MultABt(JtJi, Js, Jt);
  gshape.SetSize(el.GetDof(), dim);
  Mult(dshape, Jt, gshape);
  PT.SetSize(dim);
  Mult(Js, Jt, PT);
  F.SetSize(dim);
  Fi.SetSize(dim);
  if (map) {
    map->MapNormal(normal, Trans, ip, nu);
    map->EvalGradient(F, Trans, ip);
    CalcInverse(F, Fi);
  } else {
    nu = normal;
    Fi = 0.0;
    for (int d = 0; d < dim; d++) {
      Fi(d, d) = 1.0;
    }
  }
  return true;
}

}  // namespace

const mfem::IntegrationRule& SlipInterfacePressureIntegrator::GetRule(
    const mfem::FiniteElement& el, const mfem::ElementTransformation& Trans) {
  const auto order = 2 * el.GetOrder() + Trans.OrderW();
  return mfem::IntRules.Get(el.GetGeomType(), order);
}

const mfem::IntegrationRule& BoundaryNormalSurfaceDivergenceIntegrator::GetRule(
    const mfem::FiniteElement& el, const mfem::ElementTransformation& Trans) {
  const auto order = 2 * el.GetOrder() + Trans.OrderW();
  return mfem::IntRules.Get(el.GetGeomType(), order);
}

void BoundaryNormalSurfaceDivergenceIntegrator::AssembleElementMatrix(
    const mfem::FiniteElement& el, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  using namespace mfem;
  const auto dim = Trans.GetSpaceDim();
  const auto sdim = el.GetDim();
  const auto dof = el.GetDof();

#ifdef MFEM_THREAD_SAFE
  Vector shape_, normal_, nu_;
  DenseMatrix dshape_, gshape_, Jt_, JtJ_, JtJi_, F_, Fi_, PT_;
#endif
  shape_.SetSize(dof);
  dshape_.SetSize(dof, sdim);
  elmat.SetSize(dim * dof);
  elmat = 0.0;

  const auto* ir = IntRule ? IntRule : &GetRule(el, Trans);
  for (auto q = 0; q < ir->GetNPoints(); q++) {
    const auto& ip = ir->IntPoint(q);
    if (!SlipSurfaceData(el, Trans, ip, nullptr, shape_, dshape_, gshape_,
                         normal_, nu_, F_, Fi_, PT_, Jt_, JtJ_, JtJi_)) {
      continue;
    }
    // (1/2) pi [ (m.u) div_S v + (m.v) div_S u ]: for u = phi_a e_i,
    // v = phi_b e_j the entry is
    // (w pi / 2) [ m_i phi_a gshape(b, j) + gshape(a, i) m_j phi_b ].
    const auto w = 0.5 * ip.weight * Trans.Weight() * pi_->Eval(Trans, ip);
    for (int i = 0; i < dim; i++) {
      for (int a = 0; a < dof; a++) {
        const real_t ma = w * normal_[i] * shape_[a];
        const real_t ga = w * gshape_(a, i);
        for (int j = 0; j < dim; j++) {
          for (int b = 0; b < dof; b++) {
            elmat(a + i * dof, b + j * dof) +=
                ma * gshape_(b, j) + ga * normal_[j] * shape_[b];
          }
        }
      }
    }
  }
}

void SlipInterfacePressureIntegrator::AssembleElementMatrix(
    const mfem::FiniteElement& el, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  using namespace mfem;
  const auto dim = Trans.GetSpaceDim();
  const auto sdim = el.GetDim();
  const auto dof = el.GetDof();

#ifdef MFEM_THREAD_SAFE
  Vector shape_, normal_, nu_;
  DenseMatrix dshape_, gshape_, Jt_, JtJ_, JtJi_, F_, Fi_, PT_, dir_, T_;
#endif
  shape_.SetSize(dof);
  dshape_.SetSize(dof, sdim);
  elmat.SetSize(dim * dof);
  elmat = 0.0;

  T_.SetSize(dof, dim);  // T(p, j) = gshape_p . dir_col_j

  const auto* ir = IntRule ? IntRule : &GetRule(el, Trans);
  for (auto q = 0; q < ir->GetNPoints(); q++) {
    const auto& ip = ir->IntPoint(q);
    if (!SlipSurfaceData(el, Trans, ip, map_, shape_, dshape_, gshape_, normal_,
                         nu_, F_, Fi_, PT_, Jt_, JtJ_, JtJi_)) {
      continue;
    }
    SlipDirection(PT_, Fi_, nu_, map_ != nullptr, dir_);
    Mult(gshape_, dir_, T_);

    const auto w = ip.weight * Trans.Weight() * pi_->Eval(Trans, ip);
    for (int i = 0; i < dim; i++) {
      for (int p = 0; p < dof; p++) {
        const real_t row = w * nu_[i];
        for (int j = 0; j < dim; j++) {
          const real_t tv = row * T_(p, j);
          for (int qq = 0; qq < dof; qq++) {
            elmat(p + i * dof, qq + j * dof) += tv * shape_[qq];
          }
        }
      }
    }
  }
}

const mfem::IntegrationRule& SlipInterfaceGravityIntegrator::GetRule(
    const mfem::FiniteElement& el, const mfem::ElementTransformation& Trans) {
  const auto order = 2 * el.GetOrder() + Trans.OrderW();
  return mfem::IntRules.Get(el.GetGeomType(), order);
}

void SlipInterfaceGravityIntegrator::AssembleElementMatrix(
    const mfem::FiniteElement& el, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  using namespace mfem;
  const auto dim = Trans.GetSpaceDim();
  const auto sdim = el.GetDim();
  const auto dof = el.GetDof();
  constexpr real_t kPi = std::numbers::pi_v<real_t>;

#ifdef MFEM_THREAD_SAFE
  Vector shape_, normal_, nu_, gz_, b_, A_;
  DenseMatrix dshape_, gshape_, Jt_, JtJ_, JtJi_, F_, Fi_, PT_, dir_, T_;
#endif
  shape_.SetSize(dof);
  dshape_.SetSize(dof, sdim);
  elmat.SetSize(dim * dof);
  elmat = 0.0;

  T_.SetSize(dof, dim);  // T(p, j) = gshape_p . dir_col_j
  gz_.SetSize(dim);
  b_.SetSize(dim);
  A_.SetSize(dim);
  const real_t eight_pi_g = 8.0 * kPi * G_;
  const real_t four_pi_g = 4.0 * kPi * G_;

  const auto* ir = IntRule ? IntRule : &GetRule(el, Trans);
  for (auto q = 0; q < ir->GetNPoints(); q++) {
    const auto& ip = ir->IntPoint(q);
    if (!SlipSurfaceData(el, Trans, ip, map_, shape_, dshape_, gshape_, normal_,
                         nu_, F_, Fi_, PT_, Jt_, JtJ_, JtJi_)) {
      continue;
    }
    SlipDirection(PT_, Fi_, nu_, map_ != nullptr, dir_);
    Mult(gshape_, dir_, T_);

    // b = F^{-T} grad zeta0 (identity map: b = grad zeta0), and
    // A = (|b|^2 / 8 pi G) nu - ((b.nu) / 4 pi G) b.
    grad_zeta0_->Eval(gz_, Trans, ip);
    if (map_) {
      Fi_.MultTranspose(gz_, b_);
    } else {
      b_ = gz_;
    }
    const real_t b2 = b_ * b_;
    const real_t bnu = b_ * nu_;
    for (int d = 0; d < dim; d++) {
      A_[d] = b2 * nu_[d] / eight_pi_g - bnu * b_[d] / four_pi_g;
    }

    const auto w = ip.weight * Trans.Weight();
    for (int i = 0; i < dim; i++) {
      const real_t row = w * A_[i];
      for (int p = 0; p < dof; p++) {
        for (int j = 0; j < dim; j++) {
          const real_t tv = row * T_(p, j);
          for (int qq = 0; qq < dof; qq++) {
            elmat(p + i * dof, qq + j * dof) += tv * shape_[qq];
          }
        }
      }
    }
  }
}

const mfem::IntegrationRule& SlipInterfaceGravityScalarIntegrator::GetRule(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    const mfem::ElementTransformation& Trans) {
  const auto order = trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW();
  return mfem::IntRules.Get(trial_fe.GetGeomType(), order);
}

void SlipInterfaceGravityScalarIntegrator::AssembleElementMatrix2(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat) {
  using namespace mfem;
  const auto dim = Trans.GetSpaceDim();
  const auto sdim = trial_fe.GetDim();
  const auto dof_z = trial_fe.GetDof();  // scalar zeta (surface gradient)
  const auto dof_v = test_fe.GetDof();   // vector slip slot
  constexpr real_t kPi = std::numbers::pi_v<real_t>;

#ifdef MFEM_THREAD_SAFE
  Vector shape_, normal_, nu_, gz_, b_, test_shape_;
  DenseMatrix dshape_, gshape_, Jt_, JtJ_, JtJi_, F_, Fi_, PT_, dir_, T_;
#endif
  shape_.SetSize(dof_z);
  dshape_.SetSize(dof_z, sdim);
  elmat.SetSize(dim * dof_v, dof_z);
  elmat = 0.0;

  test_shape_.SetSize(dof_v);
  T_.SetSize(dof_z, dim);  // T(p, j) = gshape^zeta_p . dir_col_j
  gz_.SetSize(dim);
  b_.SetSize(dim);
  const real_t four_pi_g = 4.0 * kPi * G_;

  const auto* ir = IntRule ? IntRule : &GetRule(trial_fe, test_fe, Trans);
  for (auto q = 0; q < ir->GetNPoints(); q++) {
    const auto& ip = ir->IntPoint(q);
    if (!SlipSurfaceData(trial_fe, Trans, ip, map_, shape_, dshape_, gshape_,
                         normal_, nu_, F_, Fi_, PT_, Jt_, JtJ_, JtJi_)) {
      continue;
    }
    SlipDirection(PT_, Fi_, nu_, map_ != nullptr, dir_);
    Mult(gshape_, dir_, T_);
    test_fe.CalcShape(ip, test_shape_);

    grad_zeta0_->Eval(gz_, Trans, ip);
    if (map_) {
      Fi_.MultTranspose(gz_, b_);
    } else {
      b_ = gz_;
    }
    const real_t qcoef = (b_ * nu_) / four_pi_g;

    const auto w = ip.weight * Trans.Weight() * qcoef;
    for (int p = 0; p < dof_z; p++) {
      for (int j = 0; j < dim; j++) {
        const real_t tv = w * T_(p, j);
        for (int qq = 0; qq < dof_v; qq++) {
          elmat(qq + j * dof_v, p) += tv * test_shape_[qq];
        }
      }
    }
  }
}

const mfem::IntegrationRule& BoundaryNormalScalarIntegrator::GetRule(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    const mfem::ElementTransformation& Trans) {
  const auto order = trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW();
  return mfem::IntRules.Get(trial_fe.GetGeomType(), order);
}

void BoundaryNormalScalarIntegrator::AssembleElementMatrix2(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat) {
  using namespace mfem;
  const auto dim = Trans.GetSpaceDim();
  const auto trial_dof = trial_fe.GetDof();
  const auto test_dof = test_fe.GetDof();

#ifdef MFEM_THREAD_SAFE
  Vector trial_shape, test_shape, normal, nshape;
  Vector nu_;
#endif
  trial_shape.SetSize(trial_dof);
  test_shape.SetSize(test_dof);
  nshape.SetSize(dim * test_dof);
  elmat.SetSize(dim * test_dof, trial_dof);
  elmat = 0.0;

  const auto* ir = IntRule ? IntRule : &GetRule(trial_fe, test_fe, Trans);
  for (auto q = 0; q < ir->GetNPoints(); q++) {
    const auto& ip = ir->IntPoint(q);
    Trans.SetIntPoint(&ip);
    if (!BoundaryUnitNormal(Trans, normal)) {
      continue;
    }
    if (map_) {
      // Nanson exactly: m dS -> nu dS, no norm factor.
      map_->MapNormal(normal, Trans, ip, nu_);
      normal = nu_;
    }
    trial_fe.CalcShape(ip, trial_shape);
    test_fe.CalcShape(ip, test_shape);
    auto w = ip.weight * Trans.Weight();
    if (Q) {
      w *= Q->Eval(Trans, ip);
    }
    for (auto d = 0; d < dim; d++) {
      for (auto i = 0; i < test_dof; i++) {
        nshape[i + d * test_dof] = w * test_shape[i] * normal[d];
      }
    }
    AddMultVWt(nshape, trial_shape, elmat);
  }
}

const mfem::IntegrationRule& BoundaryVectorScalarIntegrator::GetRule(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    const mfem::ElementTransformation& Trans) {
  const auto order = trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW();
  return mfem::IntRules.Get(trial_fe.GetGeomType(), order);
}

void BoundaryVectorScalarIntegrator::AssembleElementMatrix2(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat) {
  using namespace mfem;
  const auto dim = Trans.GetSpaceDim();
  const auto trial_dof = trial_fe.GetDof();
  const auto test_dof = test_fe.GetDof();

#ifdef MFEM_THREAD_SAFE
  Vector trial_shape, test_shape, cvec_, cshape_;
#endif
  trial_shape.SetSize(trial_dof);
  test_shape.SetSize(test_dof);
  cvec_.SetSize(dim);
  cshape_.SetSize(dim * test_dof);
  elmat.SetSize(dim * test_dof, trial_dof);
  elmat = 0.0;

  const auto* ir = IntRule ? IntRule : &GetRule(trial_fe, test_fe, Trans);
  for (auto q = 0; q < ir->GetNPoints(); q++) {
    const auto& ip = ir->IntPoint(q);
    Trans.SetIntPoint(&ip);
    c_->Eval(cvec_, Trans, ip);
    trial_fe.CalcShape(ip, trial_shape);
    test_fe.CalcShape(ip, test_shape);
    const auto w = ip.weight * Trans.Weight();
    for (auto d = 0; d < dim; d++) {
      for (auto i = 0; i < test_dof; i++) {
        cshape_[i + d * test_dof] = w * test_shape[i] * cvec_[d];
      }
    }
    AddMultVWt(cshape_, trial_shape, elmat);
  }
}

}  // namespace AdGIA
