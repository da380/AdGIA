// The integrators' forms as libCEED operators; see
// AdGIA/detail/ceed_operators.hpp. Without libCEED in mfem this file
// holds the functions' refusals alone.
#include "AdGIA/detail/ceed_operators.hpp"

#include "AdGIA/detail/pa_kernels.hpp"

#ifdef MFEM_USE_CEED
#include <ceed.h>

#include <string>

#include "displacement_block_qf.h"
#include "potential_block_qf.h"
#include "fem/ceed/interface/basis.hpp"
#include "fem/ceed/interface/restriction.hpp"

#define ADGIA_CEED_CHECK(call) \
  MFEM_VERIFY((call) == CEED_ERROR_SUCCESS, "AdGIA: a libCEED call failed.")
#endif

namespace AdGIA::detail {

#ifdef MFEM_USE_CEED

bool HasCeed() { return true; }

namespace {

// A form on an H1 space whose work at the quadrature points is a
// QFunction of the reference gradients, with or without the values, and
// of data kept at the points.
class PointDataCeedOperator : public mfem::ceed::Operator {
 public:
  // source: the QFunction's file in ADGIA_CEED_QFUNCTION_DIR and its name,
  // "file.h:Name". n: the data at a point. values: the QFunction takes
  // and gives values as well as gradients.
  PointDataCeedOperator(const mfem::FiniteElementSpace& fes,
                        const mfem::IntegrationRule& ir,
                        const mfem::Vector& data, int n,
                        CeedQFunctionUser function, const char* source,
                        bool values) {
    Ceed ceed = mfem::internal::ceed;
    const CeedInt ne = fes.GetNE();
    const CeedInt nq = ir.GetNPoints();
    const CeedInt vdim = fes.GetVDim();
    const CeedInt dim = fes.GetMesh()->Dimension();
    MFEM_VERIFY(data.Size() == static_cast<long long>(n) * nq * ne,
                "PointDataCeedOperator: size of the data.");

    // mfem's, and kept by mfem: not destroyed here.
    CeedBasis basis;
    CeedElemRestriction restriction;
    mfem::ceed::InitBasis(fes, ir, ceed, &basis);
    mfem::ceed::InitRestriction(fes, ceed, &restriction);

    // The data as they are laid out: component, point, element.
    const CeedInt strides[3] = {n, 1, n * nq};
    const CeedSize size = static_cast<CeedSize>(n) * nq * ne;
    ADGIA_CEED_CHECK(CeedElemRestrictionCreateStrided(
        ceed, ne, nq, n, size, strides, &data_restriction_));
    ADGIA_CEED_CHECK(CeedVectorCreate(ceed, size, &data_));
    ADGIA_CEED_CHECK(CeedVectorSetArray(
        data_, CEED_MEM_HOST, CEED_COPY_VALUES,
        const_cast<CeedScalar*>(data.HostRead())));

    const std::string path =
        std::string(ADGIA_CEED_QFUNCTION_DIR "/") + source;
    ADGIA_CEED_CHECK(CeedQFunctionCreateInterior(ceed, 1, function,
                                                 path.c_str(), &qfunction_));
    if (values) {
      ADGIA_CEED_CHECK(
          CeedQFunctionAddInput(qfunction_, "u", vdim, CEED_EVAL_INTERP));
    }
    ADGIA_CEED_CHECK(
        CeedQFunctionAddInput(qfunction_, "du", vdim * dim, CEED_EVAL_GRAD));
    ADGIA_CEED_CHECK(
        CeedQFunctionAddInput(qfunction_, "data", n, CEED_EVAL_NONE));
    if (values) {
      ADGIA_CEED_CHECK(
          CeedQFunctionAddOutput(qfunction_, "v", vdim, CEED_EVAL_INTERP));
    }
    ADGIA_CEED_CHECK(
        CeedQFunctionAddOutput(qfunction_, "dv", vdim * dim, CEED_EVAL_GRAD));

    ADGIA_CEED_CHECK(CeedOperatorCreate(ceed, qfunction_, CEED_QFUNCTION_NONE,
                                        CEED_QFUNCTION_NONE, &oper));
    if (values) {
      ADGIA_CEED_CHECK(CeedOperatorSetField(oper, "u", restriction, basis,
                                            CEED_VECTOR_ACTIVE));
      ADGIA_CEED_CHECK(CeedOperatorSetField(oper, "v", restriction, basis,
                                            CEED_VECTOR_ACTIVE));
    }
    ADGIA_CEED_CHECK(CeedOperatorSetField(oper, "du", restriction, basis,
                                          CEED_VECTOR_ACTIVE));
    ADGIA_CEED_CHECK(CeedOperatorSetField(oper, "data", data_restriction_,
                                          CEED_BASIS_NONE, data_));
    ADGIA_CEED_CHECK(CeedOperatorSetField(oper, "dv", restriction, basis,
                                          CEED_VECTOR_ACTIVE));
    ADGIA_CEED_CHECK(CeedOperatorCheckReady(oper));

    height = width = fes.GetVSize();
    ADGIA_CEED_CHECK(CeedVectorCreate(ceed, height, &v));
    ADGIA_CEED_CHECK(CeedVectorCreate(ceed, width, &u));
  }

  ~PointDataCeedOperator() override {
    CeedQFunctionDestroy(&qfunction_);
    CeedVectorDestroy(&data_);
    CeedElemRestrictionDestroy(&data_restriction_);
  }

 private:
  CeedQFunction qfunction_ = nullptr;
  CeedVector data_ = nullptr;
  CeedElemRestriction data_restriction_ = nullptr;
};

}  // namespace

std::unique_ptr<mfem::ceed::Operator> MakeDisplacementBlockCeedOperator(
    const mfem::FiniteElementSpace& fes, const mfem::IntegrationRule& ir,
    const mfem::Vector& data, bool isotropic, int extras) {
  CeedQFunctionUser function = nullptr;
  const char* source = nullptr;
#define ADGIA_QFUNCTION(name)            \
  {                                      \
    function = name;                     \
    source = "displacement_block_qf.h:" #name; \
  }
  if (extras == pa::kNoExtras) {
    if (isotropic) ADGIA_QFUNCTION(AdGIAElasticIsotropic)
    else ADGIA_QFUNCTION(AdGIAElasticGeneral)
  } else if (extras == pa::kMassExtras) {
    if (isotropic) ADGIA_QFUNCTION(AdGIAElasticMassIsotropic)
    else ADGIA_QFUNCTION(AdGIAElasticMassGeneral)
  } else if (extras == pa::kGravityExtras) {
    if (isotropic) ADGIA_QFUNCTION(AdGIAElasticGravityIsotropic)
    else ADGIA_QFUNCTION(AdGIAElasticGravityGeneral)
  } else {
    MFEM_ABORT("MakeDisplacementBlockCeedOperator: extras.");
  }
#undef ADGIA_QFUNCTION
  return std::make_unique<PointDataCeedOperator>(
      fes, ir, data, pa::DataSize(isotropic, extras), function, source,
      extras != pa::kNoExtras);
}

std::unique_ptr<mfem::ceed::Operator> MakePotentialBlockCeedOperator(
    const mfem::FiniteElementSpace& fes, const mfem::IntegrationRule& ir,
    const mfem::Vector& data) {
  return std::make_unique<PointDataCeedOperator>(
      fes, ir, data, pa::kPotentialDataSize, AdGIAPotential,
      "potential_block_qf.h:AdGIAPotential", true);
}

#else

bool HasCeed() { return false; }

std::unique_ptr<mfem::ceed::Operator> MakePotentialBlockCeedOperator(
    const mfem::FiniteElementSpace&, const mfem::IntegrationRule&,
    const mfem::Vector&) {
  MFEM_ABORT("AdGIA: mfem was built without libCEED (MFEM_USE_CEED).");
  return nullptr;
}

std::unique_ptr<mfem::ceed::Operator> MakeDisplacementBlockCeedOperator(
    const mfem::FiniteElementSpace&, const mfem::IntegrationRule&,
    const mfem::Vector&, bool, int) {
  MFEM_ABORT("AdGIA: mfem was built without libCEED (MFEM_USE_CEED).");
  return nullptr;
}

#endif

}  // namespace AdGIA::detail
