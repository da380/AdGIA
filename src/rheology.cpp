/**
 * @file rheology.cpp
 * @brief Implementation of the rheologies and their stiffness objects.
 */

#include "AdGIA/rheology.hpp"

#include <cstdlib>
#include <functional>
#include <string>

#include "AdGIA/bilininteg.hpp"

namespace AdGIA {

using namespace mfem;

bool Rheology::IsLinear() const {
  for (int k = 0; k < NumBranches(); k++) {
    const RelaxationLaw* law = Law(k);
    if (law && law->IsStateDependent()) {
      return false;
    }
  }
  return true;
}

std::string Rheology::BranchLabel(int k) const {
  return "branch" + std::to_string(k);
}

Coefficient& Rheology::BranchShearModulus(int /*k*/) const {
  MFEM_ABORT("Rheology::BranchShearModulus: this rheology has no scalar "
             "branch moduli (TraceFreeInternalVariables() is false).");
  return *static_cast<Coefficient*>(nullptr);
}

// ---------------------------------------------------------------------------
// Shared pieces

namespace {

/// The branch methods of a branchless rheology.
[[noreturn]] void NoBranches(const char* cls, const char* method) {
  MFEM_ABORT(cls << "::" << method
                 << ": a purely elastic rheology has no branches.");
  std::abort();
}

/// form.AddDomainIntegrator(integ[, *marker]).
void AddDomain(BilinearForm& form, BilinearFormIntegrator* integ,
               Array<int>* marker) {
  if (marker) {
    form.AddDomainIntegrator(integ, *marker);
  } else {
    form.AddDomainIntegrator(integ);
  }
}

/// kappa div-div + 2 mu dev-dev.
void AddIsotropicIntegrators(BilinearForm& form, int dim, Coefficient& kappa,
                             Coefficient& mu, Array<int>* marker) {
  AddDomain(form, new ElasticityIntegrator(kappa, 1.0, 0.0), marker);
  AddDomain(form, new ElasticityIntegrator(mu, -2.0 / dim, 1.0), marker);
}

/// lambda 1 1^T + 2 mu I in Mandel form, 1 1^T = d P_vol.
void IsotropicMandelTensor(int dim, real_t lambda, real_t mu, DenseMatrix& C) {
  const int ns = SymmetricTensorBasis::Size(dim);
  SymmetricTensorBasis::VolumetricProjector(dim, C);
  C *= dim * lambda;
  for (int i = 0; i < ns; i++) {
    C(i, i) += 2.0 * mu;
  }
}

/// Stiffness of a branchless rheology: fixed integrators, weights no-ops.
class ElasticOnlyStiffness : public ElasticStiffness {
 public:
  explicit ElasticOnlyStiffness(
      std::function<void(BilinearForm&, Array<int>*)> add)
      : add_(std::move(add)) {}

  void AddIntegrators(BilinearForm& form, Array<int>* marker) override {
    add_(form, marker);
  }

  void SetRelaxationWeights(const std::vector<Coefficient*>& beta) override {
    MFEM_VERIFY(beta.empty(),
                "SetRelaxationWeights: a purely elastic rheology takes no "
                "weights.");
  }

  void ClearRelaxationWeights() override {}

  bool IsRelaxed() const override { return false; }

 private:
  std::function<void(BilinearForm&, Array<int>*)> add_;
};

}  // namespace

// ---------------------------------------------------------------------------
// Isotropic elastic

IsotropicElasticRheology::IsotropicElasticRheology(int dim, Coefficient& kappa,
                                                   Coefficient& mu)
    : dim_(dim), kappa_(&kappa), mu_(&mu) {
  MFEM_VERIFY(dim == 2 || dim == 3,
              "IsotropicElasticRheology: dim must be 2 or 3.");
  lambda_ = std::make_unique<SumCoefficient>(*kappa_, *mu_, 1.0,
                                             -2.0 / static_cast<real_t>(dim_));
}

Coefficient& IsotropicElasticRheology::RelaxationTime(int /*k*/) const {
  NoBranches("IsotropicElasticRheology", "RelaxationTime");
}

const RelaxationLaw* IsotropicElasticRheology::Law(int /*k*/) const {
  NoBranches("IsotropicElasticRheology", "Law");
}

void IsotropicElasticRheology::BranchModulus(int /*k*/,
                                             ElementTransformation& /*T*/,
                                             const IntegrationPoint& /*ip*/,
                                             DenseMatrix& /*Ck*/) const {
  NoBranches("IsotropicElasticRheology", "BranchModulus");
}

void IsotropicElasticRheology::UnrelaxedModulus(ElementTransformation& T,
                                                const IntegrationPoint& ip,
                                                DenseMatrix& CU) const {
  IsotropicMandelTensor(dim_, lambda_->Eval(T, ip), mu_->Eval(T, ip), CU);
}

std::unique_ptr<ElasticStiffness> IsotropicElasticRheology::MakeStiffness()
    const {
  return std::make_unique<ElasticOnlyStiffness>(
      [this](BilinearForm& form, Array<int>* marker) {
        AddIsotropicIntegrators(form, dim_, *kappa_, *mu_, marker);
      });
}

// ---------------------------------------------------------------------------
// Anisotropic elastic

AnisotropicElasticRheology::AnisotropicElasticRheology(int dim,
                                                       MatrixCoefficient& C)
    : dim_(dim), C_(&C) {
  MFEM_VERIFY(dim == 2 || dim == 3,
              "AnisotropicElasticRheology: dim must be 2 or 3.");
  const int ns = SymmetricTensorBasis::Size(dim);
  MFEM_VERIFY(C.GetHeight() == ns && C.GetWidth() == ns,
              "AnisotropicElasticRheology: C must be n_s x n_s.");
}

Coefficient& AnisotropicElasticRheology::RelaxationTime(int /*k*/) const {
  NoBranches("AnisotropicElasticRheology", "RelaxationTime");
}

const RelaxationLaw* AnisotropicElasticRheology::Law(int /*k*/) const {
  NoBranches("AnisotropicElasticRheology", "Law");
}

void AnisotropicElasticRheology::BranchModulus(int /*k*/,
                                               ElementTransformation& /*T*/,
                                               const IntegrationPoint& /*ip*/,
                                               DenseMatrix& /*Ck*/) const {
  NoBranches("AnisotropicElasticRheology", "BranchModulus");
}

std::unique_ptr<ElasticStiffness> AnisotropicElasticRheology::MakeStiffness()
    const {
  return std::make_unique<ElasticOnlyStiffness>(
      [this](BilinearForm& form, Array<int>* marker) {
        AddDomain(form, new ElasticTensorIntegrator(*C_), marker);
      });
}

// ---------------------------------------------------------------------------
// Isotropic Maxwell

namespace {

/// kappa div-div + 2 mu dev-dev with mu redirectable between mu_U and
/// mu_inf + sum_k beta_k mu_k.
class IsotropicStiffness : public ElasticStiffness {
 public:
  explicit IsotropicStiffness(const IsotropicMaxwellRheology& r)
      : r_(&r), mu_current_(&r.UnrelaxedShearModulus()) {}

  void AddIntegrators(BilinearForm& form, Array<int>* marker) override {
    AddIsotropicIntegrators(form, r_->SpaceDim(), r_->BulkModulus(),
                            mu_current_, marker);
  }

  void SetRelaxationWeights(const std::vector<Coefficient*>& beta) override {
    MFEM_VERIFY(static_cast<int>(beta.size()) == r_->NumBranches(),
                "SetRelaxationWeights: one weight per branch.");
    // mu_eff = mu_inf + sum_k beta_k mu_k as a chain of coefficients; the
    // old chain is released only after the target has been moved.
    std::vector<std::unique_ptr<Coefficient>> chain;
    Coefficient* mu = &r_->LongTermShearModulus();
    for (int k = 0; k < r_->NumBranches(); k++) {
      chain.push_back(
          std::make_unique<ProductCoefficient>(*beta[k], *r_->Branch(k).mu));
      chain.push_back(std::make_unique<SumCoefficient>(*mu, *chain.back()));
      mu = chain.back().get();
    }
    mu_current_.SetTarget(mu);
    chain_ = std::move(chain);
    relaxed_ = true;
  }

  void ClearRelaxationWeights() override {
    mu_current_.SetTarget(&r_->UnrelaxedShearModulus());
    chain_.clear();
    relaxed_ = false;
  }

  bool IsRelaxed() const override { return relaxed_; }

 private:
  const IsotropicMaxwellRheology* r_;
  detail::RedirectableCoefficient mu_current_;
  std::vector<std::unique_ptr<Coefficient>> chain_;
  bool relaxed_ = false;
};

}  // namespace

IsotropicMaxwellRheology::IsotropicMaxwellRheology(
    int dim, Coefficient& kappa, Coefficient& mu_inf,
    const std::vector<MaxwellBranch>& branches)
    : dim_(dim), kappa_(&kappa), mu_inf_(&mu_inf), branches_(branches) {
  MFEM_VERIFY(dim == 2 || dim == 3,
              "IsotropicMaxwellRheology: dim must be 2 or 3.");
  for (const auto& b : branches_) {
    MFEM_VERIFY(b.mu && b.tau,
                "IsotropicMaxwellRheology: every branch needs mu and tau.");
  }

  // mu_U = mu_inf + sum_k mu_k, as a chain of SumCoefficients.
  mu_u_ = mu_inf_;
  for (const auto& b : branches_) {
    owned_.push_back(std::make_unique<SumCoefficient>(*mu_u_, *b.mu));
    mu_u_ = owned_.back().get();
  }

  // lambda_U = kappa - 2 mu_U / dim.
  owned_.push_back(std::make_unique<SumCoefficient>(
      *kappa_, *mu_u_, 1.0, -2.0 / static_cast<real_t>(dim_)));
  lambda_u_ = owned_.back().get();
}

IsotropicMaxwellRheology IsotropicMaxwellRheology::Maxwell(
    int dim, Coefficient& kappa, Coefficient& mu, Coefficient& tau,
    const RelaxationLaw* law) {
  // mu_inf = 0 is an owned constant; move it into the result so that the
  // pointer stays valid.
  auto zero = std::make_unique<ConstantCoefficient>(0.0);
  auto* zero_ptr = zero.get();
  std::vector<MaxwellBranch> branches{MaxwellBranch{&mu, &tau, law}};
  IsotropicMaxwellRheology r(dim, kappa, *zero_ptr, branches);
  r.owned_.push_back(std::move(zero));
  return r;
}

void IsotropicMaxwellRheology::BranchModulus(int k, ElementTransformation& T,
                                               const IntegrationPoint& ip,
                                               DenseMatrix& Ck) const {
  // 2 mu_k P_dev in Mandel form.
  SymmetricTensorBasis::DeviatoricProjector(dim_, Ck);
  Ck *= 2.0 * branches_[k].mu->Eval(T, ip);
}

void IsotropicMaxwellRheology::UnrelaxedModulus(ElementTransformation& T,
                                                const IntegrationPoint& ip,
                                                DenseMatrix& CU) const {
  IsotropicMandelTensor(dim_, lambda_u_->Eval(T, ip), mu_u_->Eval(T, ip), CU);
}

std::unique_ptr<ElasticStiffness> IsotropicMaxwellRheology::MakeStiffness()
    const {
  return std::make_unique<IsotropicStiffness>(*this);
}

IsotropicElasticRheology IsotropicMaxwellRheology::UnrelaxedElastic() const {
  return IsotropicElasticRheology(dim_, *kappa_, *mu_u_);
}

IsotropicElasticRheology IsotropicMaxwellRheology::LongTermElastic() const {
  return IsotropicElasticRheology(dim_, *kappa_, *mu_inf_);
}

// ---------------------------------------------------------------------------
// Anisotropic Maxwell

namespace {

/// One ElasticTensorIntegrator with a redirectable tensor: C_U, or
/// C_inf + sum_k beta_k C_k.
class AnisotropicStiffness : public ElasticStiffness {
 public:
  explicit AnisotropicStiffness(const AnisotropicMaxwellRheology& r)
      : r_(&r), C_current_(&r.UnrelaxedTensor()) {}

  void AddIntegrators(BilinearForm& form, Array<int>* marker) override {
    AddDomain(form, new ElasticTensorIntegrator(C_current_), marker);
  }

  void SetRelaxationWeights(const std::vector<Coefficient*>& beta) override {
    MFEM_VERIFY(static_cast<int>(beta.size()) == r_->NumBranches(),
                "SetRelaxationWeights: one weight per branch.");
    std::vector<std::unique_ptr<MatrixCoefficient>> chain;
    MatrixCoefficient* C = &r_->LongTermTensor();
    for (int k = 0; k < r_->NumBranches(); k++) {
      chain.push_back(std::make_unique<ScalarMatrixProductCoefficient>(
          *beta[k], *r_->Branch(k).C));
      chain.push_back(
          std::make_unique<MatrixSumCoefficient>(*C, *chain.back(), 1.0, 1.0));
      C = chain.back().get();
    }
    C_current_.SetTarget(C);
    chain_ = std::move(chain);
    relaxed_ = true;
  }

  void ClearRelaxationWeights() override {
    C_current_.SetTarget(&r_->UnrelaxedTensor());
    chain_.clear();
    relaxed_ = false;
  }

  bool IsRelaxed() const override { return relaxed_; }

 private:
  const AnisotropicMaxwellRheology* r_;
  detail::RedirectableMatrixCoefficient C_current_;
  std::vector<std::unique_ptr<MatrixCoefficient>> chain_;
  bool relaxed_ = false;
};

}  // namespace

AnisotropicMaxwellRheology::AnisotropicMaxwellRheology(
    int dim, MatrixCoefficient& C_inf,
    const std::vector<AnisotropicBranch>& branches)
    : dim_(dim), C_inf_(&C_inf), branches_(branches) {
  MFEM_VERIFY(dim == 2 || dim == 3,
              "AnisotropicMaxwellRheology: dim must be 2 or 3.");
  const int ns = SymmetricTensorBasis::Size(dim);
  MFEM_VERIFY(C_inf.GetHeight() == ns && C_inf.GetWidth() == ns,
              "AnisotropicMaxwellRheology: C_inf must be n_s x n_s.");
  for (const auto& b : branches_) {
    MFEM_VERIFY(b.C && b.tau,
                "AnisotropicMaxwellRheology: every branch needs C and tau.");
    MFEM_VERIFY(b.C->GetHeight() == ns && b.C->GetWidth() == ns,
                "AnisotropicMaxwellRheology: C_k must be n_s x n_s.");
  }
  C_u_ = C_inf_;
  for (const auto& b : branches_) {
    owned_.push_back(
        std::make_unique<MatrixSumCoefficient>(*C_u_, *b.C, 1.0, 1.0));
    C_u_ = owned_.back().get();
  }
}

AnisotropicMaxwellRheology AnisotropicMaxwellRheology::DeviatoricMaxwell(
    int dim, MatrixCoefficient& C, Coefficient& tau, const RelaxationLaw* law) {
  auto dev = std::make_unique<DeviatoricProjectionElasticTensorCoefficient>(
      dim, C, true);
  auto rest = std::make_unique<DeviatoricProjectionElasticTensorCoefficient>(
      dim, C, false);
  std::vector<AnisotropicBranch> branches{
      AnisotropicBranch{dev.get(), &tau, law}};
  AnisotropicMaxwellRheology r(dim, *rest, branches);
  r.owned_.push_back(std::move(dev));
  r.owned_.push_back(std::move(rest));
  return r;
}

std::unique_ptr<ElasticStiffness> AnisotropicMaxwellRheology::MakeStiffness()
    const {
  return std::make_unique<AnisotropicStiffness>(*this);
}

AnisotropicElasticRheology AnisotropicMaxwellRheology::UnrelaxedElastic()
    const {
  return AnisotropicElasticRheology(dim_, *C_u_);
}

AnisotropicElasticRheology AnisotropicMaxwellRheology::LongTermElastic()
    const {
  return AnisotropicElasticRheology(dim_, *C_inf_);
}

// ---------------------------------------------------------------------------
// Composite

namespace {

/// A scalar coefficient equal to `inner` on the marked attributes and to a
/// constant elsewhere.
class RegionCoefficient : public Coefficient {
 public:
  RegionCoefficient(const Array<int>& marker, Coefficient& inner,
                    real_t outside)
      : marker_(&marker), inner_(&inner), outside_(outside) {}

  real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    const int a = T.Attribute;
    if (a >= 1 && a <= marker_->Size() && (*marker_)[a - 1]) {
      return inner_->Eval(T, ip);
    }
    return outside_;
  }

 private:
  const Array<int>* marker_;
  Coefficient* inner_;
  real_t outside_;
};

/// The regions' stiffnesses, each restricted to its marker; weights sliced
/// per region.
class CompositeStiffness : public ElasticStiffness {
 public:
  explicit CompositeStiffness(const CompositeRheology& r) : r_(&r) {
    for (int i = 0; i < r_->NumRegions(); i++) {
      markers_.emplace_back(r_->RegionMarker(i));
      parts_.push_back(r_->Region(i).MakeStiffness());
    }
  }

  void AddIntegrators(BilinearForm& form, Array<int>* marker) override {
    if (marker) {
      // Nested composite: restrict every region to the outer marker.
      for (auto& m : markers_) {
        MFEM_VERIFY(m.Size() == marker->Size(),
                    "CompositeRheology: marker sizes differ.");
        for (int a = 0; a < m.Size(); a++) {
          m[a] = m[a] && (*marker)[a];
        }
      }
    } else {
      r_->VerifyCoverage(*form.FESpace()->GetMesh());
    }
    for (int i = 0; i < r_->NumRegions(); i++) {
      parts_[i]->AddIntegrators(form, &markers_[i]);
    }
  }

  void SetRelaxationWeights(const std::vector<Coefficient*>& beta) override {
    MFEM_VERIFY(static_cast<int>(beta.size()) == r_->NumBranches(),
                "SetRelaxationWeights: one weight per (global) branch.");
    for (int i = 0; i < r_->NumRegions(); i++) {
      const int begin = r_->RegionBranchOffset(i);
      const int end = begin + r_->Region(i).NumBranches();
      parts_[i]->SetRelaxationWeights(
          std::vector<Coefficient*>(beta.begin() + begin, beta.begin() + end));
    }
    relaxed_ = true;
  }

  void ClearRelaxationWeights() override {
    for (auto& p : parts_) {
      p->ClearRelaxationWeights();
    }
    relaxed_ = false;
  }

  bool IsRelaxed() const override { return relaxed_; }

 private:
  const CompositeRheology* r_;
  std::vector<Array<int>> markers_;  ///< owned copies, referenced by forms
  std::vector<std::unique_ptr<ElasticStiffness>> parts_;
  bool relaxed_ = false;
};

}  // namespace

CompositeRheology::CompositeRheology(int dim,
                                     std::vector<RheologyRegion> regions)
    : dim_(dim), regions_(std::move(regions)) {
  MFEM_VERIFY(dim == 2 || dim == 3, "CompositeRheology: dim must be 2 or 3.");
  MFEM_VERIFY(!regions_.empty(), "CompositeRheology: no regions.");
  const int na = regions_[0].marker.Size();
  MFEM_VERIFY(na > 0, "CompositeRheology: empty region marker.");
  attribute_region_.assign(na, -1);
  int k = 0;
  for (int r = 0; r < NumRegions(); r++) {
    auto& region = regions_[r];
    MFEM_VERIFY(region.rheology, "CompositeRheology: region "
                                     << r << " has no rheology.");
    MFEM_VERIFY(region.rheology->SpaceDim() == dim_,
                "CompositeRheology: region " << r
                                             << " has the wrong dimension.");
    MFEM_VERIFY(region.marker.Size() == na,
                "CompositeRheology: region markers must have equal size.");
    int marked = 0;
    for (int a = 0; a < na; a++) {
      if (region.marker[a]) {
        MFEM_VERIFY(attribute_region_[a] < 0,
                    "CompositeRheology: attribute " << a + 1
                                                    << " is in two regions.");
        attribute_region_[a] = r;
        marked++;
      }
    }
    MFEM_VERIFY(marked > 0, "CompositeRheology: region " << r << " is empty.");
    if (region.name.empty()) {
      region.name = "region" + std::to_string(r);
    }
    region_offset_.push_back(k);
    for (int j = 0; j < region.rheology->NumBranches(); j++, k++) {
      branch_region_.push_back(r);
      local_branch_.push_back(j);
    }
    if (!region.rheology->TraceFreeInternalVariables()) {
      tracefree_ = false;
    }
  }
  for (int b = 0; b < NumBranches(); b++) {
    const auto& region = regions_[branch_region_[b]];
    tau_.push_back(std::make_unique<RegionCoefficient>(
        region.marker, region.rheology->RelaxationTime(local_branch_[b]),
        kOutsideRelaxationTime));
    if (tracefree_) {
      mu_.push_back(std::make_unique<RegionCoefficient>(
          region.marker, region.rheology->BranchShearModulus(local_branch_[b]),
          0.0));
    }
  }
}

int CompositeRheology::RegionOf(int attribute) const {
  if (attribute < 1 || attribute > static_cast<int>(attribute_region_.size())) {
    return -1;
  }
  return attribute_region_[attribute - 1];
}

void CompositeRheology::VerifyCoverage(const Mesh& mesh) const {
  for (int i = 0; i < mesh.attributes.Size(); i++) {
    const int a = mesh.attributes[i];
    MFEM_VERIFY(RegionOf(a) >= 0, "CompositeRheology: element attribute "
                                      << a << " belongs to no region.");
  }
}

void CompositeRheology::UnrelaxedModulus(ElementTransformation& T,
                                         const IntegrationPoint& ip,
                                         DenseMatrix& CU) const {
  const int r = RegionOf(T.Attribute);
  MFEM_VERIFY(r >= 0, "CompositeRheology: element attribute "
                          << T.Attribute << " belongs to no region.");
  Region(r).UnrelaxedModulus(T, ip, CU);
}

Coefficient& CompositeRheology::BranchShearModulus(int k) const {
  if (!tracefree_) {
    return Rheology::BranchShearModulus(k);
  }
  return *mu_[k];
}

void CompositeRheology::BranchModulus(int k, ElementTransformation& T,
                                      const IntegrationPoint& ip,
                                      DenseMatrix& Ck) const {
  const int r = branch_region_[k];
  if (RegionOf(T.Attribute) == r) {
    Region(r).BranchModulus(local_branch_[k], T, ip, Ck);
  } else {
    const int ns = SymmetricTensorBasis::Size(dim_);
    Ck.SetSize(ns);
    Ck = 0.0;
  }
}

std::unique_ptr<ElasticStiffness> CompositeRheology::MakeStiffness() const {
  return std::make_unique<CompositeStiffness>(*this);
}

std::string CompositeRheology::BranchLabel(int k) const {
  const int r = branch_region_[k];
  return regions_[r].name + "_" + Region(r).BranchLabel(local_branch_[k]);
}

}  // namespace AdGIA
