// The elastic tensor form as a libCEED operator; see
// AdGIA/detail/ceed_operators.hpp. Without libCEED in mfem this file
// holds the functions' refusals alone.
#include "AdGIA/detail/ceed_operators.hpp"

#include "AdGIA/detail/pa_kernels.hpp"

#ifdef MFEM_USE_CEED
#include <ceed.h>

#include <string>

#include "elastic_tensor_qf.h"
#include "fem/ceed/interface/basis.hpp"
#include "fem/ceed/interface/restriction.hpp"

#define ADGIA_CEED_CHECK(call) \
  MFEM_VERIFY((call) == CEED_ERROR_SUCCESS, "AdGIA: a libCEED call failed.")
#endif

namespace AdGIA::detail {

#ifdef MFEM_USE_CEED

bool HasCeed() { return true; }

namespace {

class ElasticTensorCeedOperator : public mfem::ceed::Operator {
 public:
  ElasticTensorCeedOperator(const mfem::FiniteElementSpace& fes,
                            const mfem::IntegrationRule& ir,
                            const mfem::Vector& data, bool isotropic) {
    Ceed ceed = mfem::internal::ceed;
    const CeedInt ne = fes.GetNE();
    const CeedInt nq = ir.GetNPoints();
    const CeedInt n =
        9 + (isotropic ? pa::kIsotropicTensorSize : pa::kGeneralTensorSize);
    MFEM_VERIFY(data.Size() == static_cast<long long>(n) * nq * ne,
                "ElasticTensorCeedOperator: size of the data.");

    // mfem's, and kept by mfem: not destroyed here.
    CeedBasis basis;
    CeedElemRestriction restriction;
    mfem::ceed::InitBasis(fes, ir, ceed, &basis);
    mfem::ceed::InitRestriction(fes, ceed, &restriction);

    // The data as they are laid out: component, point, element.
    const CeedInt strides[3] = {n, 1, n * nq};
    const CeedSize size = static_cast<CeedSize>(n) * nq * ne;
    ADGIA_CEED_CHECK(CeedElemRestrictionCreateStrided(ceed, ne, nq, n, size, strides,
                                              &data_restriction_));
    ADGIA_CEED_CHECK(CeedVectorCreate(ceed, size, &data_));
    ADGIA_CEED_CHECK(CeedVectorSetArray(
        data_, CEED_MEM_HOST, CEED_COPY_VALUES,
        const_cast<CeedScalar*>(data.HostRead())));

    const std::string source =
        std::string(ADGIA_CEED_QFUNCTION_DIR "/elastic_tensor_qf.h:") +
        (isotropic ? "AdGIAElasticTensorIsotropic"
                   : "AdGIAElasticTensorGeneral");
    ADGIA_CEED_CHECK(CeedQFunctionCreateInterior(
        ceed, 1,
        isotropic ? AdGIAElasticTensorIsotropic : AdGIAElasticTensorGeneral,
        source.c_str(), &qfunction_));
    ADGIA_CEED_CHECK(CeedQFunctionAddInput(qfunction_, "du", 9, CEED_EVAL_GRAD));
    ADGIA_CEED_CHECK(CeedQFunctionAddInput(qfunction_, "data", n, CEED_EVAL_NONE));
    ADGIA_CEED_CHECK(CeedQFunctionAddOutput(qfunction_, "dv", 9, CEED_EVAL_GRAD));

    ADGIA_CEED_CHECK(CeedOperatorCreate(ceed, qfunction_, CEED_QFUNCTION_NONE,
                                CEED_QFUNCTION_NONE, &oper));
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

  ~ElasticTensorCeedOperator() override {
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

std::unique_ptr<mfem::ceed::Operator> MakeElasticTensorCeedOperator(
    const mfem::FiniteElementSpace& fes, const mfem::IntegrationRule& ir,
    const mfem::Vector& data, bool isotropic) {
  return std::make_unique<ElasticTensorCeedOperator>(fes, ir, data, isotropic);
}

#else

bool HasCeed() { return false; }

std::unique_ptr<mfem::ceed::Operator> MakeElasticTensorCeedOperator(
    const mfem::FiniteElementSpace&, const mfem::IntegrationRule&,
    const mfem::Vector&, bool) {
  MFEM_ABORT("AdGIA: mfem was built without libCEED (MFEM_USE_CEED).");
  return nullptr;
}

#endif

}  // namespace AdGIA::detail
