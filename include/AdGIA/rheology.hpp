/**
 * @file rheology.hpp
 * @brief Material description shared by the elastic and viscoelastic layers:
 * the abstract Rheology (a generalised Maxwell body with branch moduli that
 * may be scalar or tensorial), the per-problem ElasticStiffness it creates,
 * the purely elastic IsotropicElasticRheology and
 * AnisotropicElasticRheology, the IsotropicMaxwellRheology and
 * AnisotropicMaxwellRheology, and the CompositeRheology assigning different
 * rheologies to different regions (element attributes) of one body.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "AdGIA/elastic_tensor.hpp"
#include "AdGIA/relaxation_law.hpp"

namespace AdGIA {

class Diffeomorphism;

namespace detail {

/**
 * @brief Scalar coefficient forwarding to a switchable target, so that an
 * integrator's coefficient can be redirected without rebuilding the
 * integrator.
 */
class RedirectableCoefficient : public mfem::Coefficient {
 public:
  explicit RedirectableCoefficient(mfem::Coefficient* target)
      : target_(target) {}

  void SetTarget(mfem::Coefficient* target) { target_ = target; }
  mfem::Coefficient* Target() const { return target_; }

  mfem::real_t Eval(mfem::ElementTransformation& T,
                    const mfem::IntegrationPoint& ip) override {
    return target_->Eval(T, ip);
  }

 private:
  mfem::Coefficient* target_;
};

/** @brief Matrix counterpart of RedirectableCoefficient. */
class RedirectableMatrixCoefficient : public mfem::MatrixCoefficient {
 public:
  explicit RedirectableMatrixCoefficient(mfem::MatrixCoefficient* target)
      : mfem::MatrixCoefficient(target->GetHeight(), target->GetWidth()),
        target_(target) {}

  void SetTarget(mfem::MatrixCoefficient* target) { target_ = target; }
  mfem::MatrixCoefficient* Target() const { return target_; }

  void Eval(mfem::DenseMatrix& K, mfem::ElementTransformation& T,
            const mfem::IntegrationPoint& ip) override {
    target_->Eval(K, T, ip);
  }

 private:
  mfem::MatrixCoefficient* target_;
};

}  // namespace detail

/**
 * @brief The integrators a quasi-static problem assembles its
 * stiffness with, together with the switch between the unrelaxed modulus
 * and an effective one. One object per problem, created by
 * Rheology::MakeStiffness(), so that several problems may share a rheology
 * without interfering.
 *
 * The stiffness assembled is @f$C_\infty + \sum_k \beta_k C_k@f$ with the
 * relaxation weights @f$\beta_k@f$ set by SetRelaxationWeights() (all one,
 * i.e. the unrelaxed @f$C_U@f$, when unset or cleared). Implementations
 * point the integrators at a redirectable coefficient, so that setting
 * weights never rebuilds integrators; the problem reassembles its matrix.
 * For a purely elastic rheology (no branches) the two moduli coincide and
 * the weight calls are no-ops.
 */
class ElasticStiffness {
 public:
  virtual ~ElasticStiffness() = default;

  /**
   * @brief Add the stiffness integrators to @p form (called once per form;
   * the form takes ownership of them, and other forms may borrow them from
   * it). With @p marker (element attributes, not owned, must outlive the
   * form and every form borrowing its integrators) they act on the marked
   * elements only.
   */
  virtual void AddIntegrators(mfem::BilinearForm& form,
                              mfem::Array<int>* marker = nullptr) = 0;

  /**
   * @brief Point the stiffness at @f$C_\infty + \sum_k \beta_k C_k@f$. One
   * coefficient per branch, not owned; they must outlive the next call to
   * SetRelaxationWeights() or ClearRelaxationWeights().
   */
  virtual void SetRelaxationWeights(
      const std::vector<mfem::Coefficient*>& beta) = 0;

  /** @brief Restore the unrelaxed modulus. */
  virtual void ClearRelaxationWeights() = 0;

  /** @brief Whether relaxation weights are currently set (never for a
   * purely elastic rheology). */
  virtual bool IsRelaxed() const = 0;
};

/**
 * @brief Abstract generalised Maxwell (Prony series) rheology,
 * @f[
 *   \sigma = C_U\,\varepsilon - \sum_k C_k m_k, \qquad
 *   C_U = C_\infty + \sum_k C_k, \qquad
 *   \dot m_k = (\varepsilon - m_k)/\tau_k ,
 * @f]
 * with @f$C_k@f$ the relaxable modulus of branch @f$k@f$ (a scalar
 * @f$2\mu_k@f$ on deviatoric tensors for the isotropic body, a fourth-order
 * tensor in general) and scalar relaxation times @f$\tau_k = \tau_{k0}(x)
 * F_k(\text{state})@f$: a branch may carry a RelaxationLaw making
 * @f$\tau_k@f$ depend on the local stress, strain and internal variable
 * (Crawford et al. 2017, Appendix A); without one, or with a
 * state-independent one, the branch is linear. The elastic part is always
 * linear. A purely elastic solid has no branches
 * (IsotropicElasticRheology, AnisotropicElasticRheology); the branch
 * methods must not be called on it.
 *
 * The material data has one owner: the quasi-static problem assembles its
 * operator with the integrators of MakeStiffness(), and the viscoelastic
 * operator reads the branch data from here, so the two layers cannot
 * disagree. When TraceFreeInternalVariables() is true the internal
 * variables are trace-free and BranchShearModulus() supplies the scalar
 * @f$\mu_k@f$ of @f$C_k = 2\mu_k P_{dev}@f$; otherwise they are full symmetric tensors and
 * BranchModulus() supplies @f$C_k@f$ pointwise (Mandel form, see
 * SymmetricTensorBasis).
 */
class Rheology {
 public:
  virtual ~Rheology() = default;

  virtual int SpaceDim() const = 0;
  virtual int NumBranches() const = 0;

  /** @brief The reference relaxation time @f$\tau_{k0}@f$ of branch
   * @p k. */
  virtual mfem::Coefficient& RelaxationTime(int k) const = 0;

  /** @brief The relaxation law of branch @p k, or null for @f$F_k = 1@f$. */
  virtual const RelaxationLaw* Law(int k) const = 0;

  /** @brief True if no branch has a state-dependent law. */
  bool IsLinear() const;

  /** @brief @f$C_U@f$ at a point as a Mandel matrix (for the nodal stress
   * of state-dependent laws). */
  virtual void UnrelaxedModulus(mfem::ElementTransformation& T,
                                const mfem::IntegrationPoint& ip,
                                mfem::DenseMatrix& CU) const = 0;

  /** @brief True if @f$C_k m@f$ is @f$2\mu_k\,\mathrm{dev}\,m@f$ for every
   * branch, so that trace-free internal variables and a scalar branch
   * weight suffice. */
  virtual bool TraceFreeInternalVariables() const = 0;

  /** @brief @f$\mu_k@f$ of branch @p k (only when
   * TraceFreeInternalVariables()). */
  virtual mfem::Coefficient& BranchShearModulus(int k) const;

  /** @brief @f$C_k@f$ at a point as an @f$n_s \times n_s@f$ Mandel matrix
   * (for the isotropic body @f$2\mu_k P_{dev}@f$). */
  virtual void BranchModulus(int k, mfem::ElementTransformation& T,
                             const mfem::IntegrationPoint& ip,
                             mfem::DenseMatrix& Ck) const = 0;

  /** @brief The stiffness integrators for one problem. */
  virtual std::unique_ptr<ElasticStiffness> MakeStiffness() const = 0;

  /** @brief A label for branch @p k, used in output field names; default
   * "branch<k>". */
  virtual std::string BranchLabel(int k) const;

  /** @brief Element attributes on which branch @p k lives, or null for the
   * whole mesh. The viscoelastic operator stores and evolves the branch's
   * internal variable on the marked elements only; the branch modulus must
   * vanish outside them. */
  virtual const mfem::Array<int>* BranchMarker(int k) const { return nullptr; }

  /** @brief The equilibrium mapping @f$\varphi_e@f$ when the reference
   * state is *non-natural* in the sense of Al-Attar & Crawford 2016 (the
   * particle label is not the equilibrium position); null means natural
   * labels, @f$\varphi_e = \mathrm{id}@f$. The stiffness integrators of
   * MakeStiffness() carry the mapping themselves; problems consult this
   * for everything else that depends on it (rigid modes of the mapped
   * positions, boundary-area factors). */
  virtual Diffeomorphism* EquilibriumMapping() const { return nullptr; }
};

// ---------------------------------------------------------------------------
// Purely elastic rheologies

/**
 * @brief Isotropic linear elastic solid, @f$\sigma = \kappa\,\mathrm{tr}
 * (\varepsilon) I + 2\mu\,\mathrm{dev}\,\varepsilon@f$: a Rheology with no
 * branches.
 *
 * The stiffness is the split form @f$\kappa\,\mathrm{div}\,u\,\mathrm{div}\,
 * v + 2\mu\,\mathrm{dev}\,\varepsilon(u) : \mathrm{dev}\,\varepsilon(v)@f$
 * with two mfem::ElasticityIntegrators; relaxation weights are no-ops. In
 * 2-D the deviator is the two-dimensional one, @f$\lambda = \kappa -
 * \mu@f$ (see IsotropicMaxwellRheology).
 *
 * Holds pointers to the caller's coefficients, which must outlive it, plus
 * the @f$\lambda@f$ it builds. Movable, not copyable. A problem with this
 * rheology passes through the viscoelastic operator with an empty internal
 * state (a purely elastic evolution under time-dependent loads).
 */
class IsotropicElasticRheology : public Rheology {
 public:
  /**
   * @param dim Space dimension (2 or 3).
   * @param kappa Bulk modulus.
   * @param mu Shear modulus.
   */
  IsotropicElasticRheology(int dim, mfem::Coefficient& kappa,
                           mfem::Coefficient& mu);

  IsotropicElasticRheology(IsotropicElasticRheology&&) = default;
  IsotropicElasticRheology& operator=(IsotropicElasticRheology&&) = default;
  IsotropicElasticRheology(const IsotropicElasticRheology&) = delete;
  IsotropicElasticRheology& operator=(const IsotropicElasticRheology&) = delete;

  int SpaceDim() const override { return dim_; }
  int NumBranches() const override { return 0; }
  mfem::Coefficient& RelaxationTime(int k) const override;
  const RelaxationLaw* Law(int k) const override;
  bool TraceFreeInternalVariables() const override { return true; }
  void BranchModulus(int k, mfem::ElementTransformation& T,
                     const mfem::IntegrationPoint& ip,
                     mfem::DenseMatrix& Ck) const override;
  void UnrelaxedModulus(mfem::ElementTransformation& T,
                        const mfem::IntegrationPoint& ip,
                        mfem::DenseMatrix& CU) const override;
  std::unique_ptr<ElasticStiffness> MakeStiffness() const override;

  mfem::Coefficient& BulkModulus() const { return *kappa_; }
  mfem::Coefficient& ShearModulus() const { return *mu_; }

  /** @brief @f$\lambda = \kappa - 2\mu/d@f$, for (lambda, mu) form. */
  mfem::Coefficient& Lame() const { return *lambda_; }

 private:
  int dim_;
  mfem::Coefficient* kappa_;
  mfem::Coefficient* mu_;
  std::unique_ptr<mfem::Coefficient> lambda_;
};

/**
 * @brief Anisotropic linear elastic solid, @f$\sigma = C\varepsilon@f$ with
 * @f$C@f$ an @f$n_s \times n_s@f$ Mandel MatrixCoefficient in
 * SymmetricTensorBasis ordering (see elastic_tensor.hpp): a Rheology with
 * no branches.
 *
 * The stiffness is one ElasticTensorIntegrator; relaxation weights are
 * no-ops. Holds a pointer to the caller's coefficient, which must outlive
 * it. Movable, not copyable. As for IsotropicElasticRheology, a field with
 * this rheology passes through the viscoelastic operator with an empty
 * internal state.
 */
class AnisotropicElasticRheology : public Rheology {
 public:
  /**
   * @param dim Space dimension (2 or 3).
   * @param C Elastic tensor.
   */
  AnisotropicElasticRheology(int dim, mfem::MatrixCoefficient& C);

  AnisotropicElasticRheology(AnisotropicElasticRheology&&) = default;
  AnisotropicElasticRheology& operator=(AnisotropicElasticRheology&&) = default;
  AnisotropicElasticRheology(const AnisotropicElasticRheology&) = delete;
  AnisotropicElasticRheology& operator=(const AnisotropicElasticRheology&) =
      delete;

  int SpaceDim() const override { return dim_; }
  int NumBranches() const override { return 0; }
  mfem::Coefficient& RelaxationTime(int k) const override;
  const RelaxationLaw* Law(int k) const override;
  bool TraceFreeInternalVariables() const override { return false; }
  void BranchModulus(int k, mfem::ElementTransformation& T,
                     const mfem::IntegrationPoint& ip,
                     mfem::DenseMatrix& Ck) const override;
  void UnrelaxedModulus(mfem::ElementTransformation& T,
                        const mfem::IntegrationPoint& ip,
                        mfem::DenseMatrix& CU) const override {
    C_->Eval(CU, T, ip);
  }
  std::unique_ptr<ElasticStiffness> MakeStiffness() const override;

  mfem::MatrixCoefficient& Tensor() const { return *C_; }

 private:
  int dim_;
  mfem::MatrixCoefficient* C_;
};

// ---------------------------------------------------------------------------
// Generalised Maxwell rheologies

/**
 * @brief One Prony branch of an isotropic generalised Maxwell body: a
 * relaxable shear modulus @f$\mu_k@f$, its relaxation time @f$\tau_k =
 * \eta_k/\mu_k@f$ (the reference time @f$\tau_{k0}@f$ when a law is set)
 * and an optional RelaxationLaw (null: linear). All three are non-owning.
 */
struct MaxwellBranch {
  mfem::Coefficient* mu = nullptr;
  mfem::Coefficient* tau = nullptr;
  const RelaxationLaw* law = nullptr;
};

/**
 * @brief Isotropic generalised Maxwell rheology.
 *
 * With @f$d = \mathrm{dev}\,\varepsilon(u)@f$ and @f$K@f$ branches,
 * @f[
 *   \sigma = \kappa\,\mathrm{tr}(\varepsilon) I + 2\mu_\infty d
 *          + \sum_k 2\mu_k (d - m_k), \qquad
 *   \dot m_k = (d - m_k)/\tau_k .
 * @f]
 * The classical Maxwell body is @f$\mu_\infty = 0@f$ with one branch. A
 * purely elastic solid is IsotropicElasticRheology (an empty branch list
 * here is legal but not the intended spelling).
 *
 * The stiffness is assembled in the split form @f$\kappa\,\mathrm{div}\,u\,
 * \mathrm{div}\,v + 2\mu\,\mathrm{dev}\,\varepsilon(u) :
 * \mathrm{dev}\,\varepsilon(v)@f$ with two mfem::ElasticityIntegrators,
 * @f$\mu = \mu_U@f$ or, with relaxation weights, @f$\mu_\infty + \sum_k
 * \beta_k \mu_k@f$.
 *
 * The space dimension fixes the deviatoric convention: in 2-D the library
 * models a two-dimensional continuum (2-D deviator, @f$\lambda = \kappa -
 * \mu@f$), not plane strain of a 3-D body.
 *
 * The object holds only pointers to the caller's coefficients, which must
 * outlive it, plus the sum coefficients it builds itself. It is movable but
 * not copyable.
 */
class IsotropicMaxwellRheology : public Rheology {
 public:
  /**
   * @param dim Space dimension (2 or 3).
   * @param kappa Bulk modulus.
   * @param mu_inf Long-term (fully relaxed) shear modulus.
   * @param branches Prony branches.
   */
  IsotropicMaxwellRheology(int dim, mfem::Coefficient& kappa,
                             mfem::Coefficient& mu_inf,
                             const std::vector<MaxwellBranch>& branches);

  /** @brief Classical Maxwell body: @f$\mu_\infty = 0@f$ and one branch,
   * optionally with a relaxation law on it. */
  static IsotropicMaxwellRheology Maxwell(int dim, mfem::Coefficient& kappa,
                                          mfem::Coefficient& mu,
                                          mfem::Coefficient& tau,
                                          const RelaxationLaw* law = nullptr);

  IsotropicMaxwellRheology(IsotropicMaxwellRheology&&) = default;
  IsotropicMaxwellRheology& operator=(IsotropicMaxwellRheology&&) = default;
  IsotropicMaxwellRheology(const IsotropicMaxwellRheology&) = delete;
  IsotropicMaxwellRheology& operator=(const IsotropicMaxwellRheology&) =
      delete;

  int SpaceDim() const override { return dim_; }
  int NumBranches() const override {
    return static_cast<int>(branches_.size());
  }
  mfem::Coefficient& RelaxationTime(int k) const override {
    return *branches_[k].tau;
  }
  const RelaxationLaw* Law(int k) const override { return branches_[k].law; }
  bool TraceFreeInternalVariables() const override { return true; }
  mfem::Coefficient& BranchShearModulus(int k) const override {
    return *branches_[k].mu;
  }
  void BranchModulus(int k, mfem::ElementTransformation& T,
                     const mfem::IntegrationPoint& ip,
                     mfem::DenseMatrix& Ck) const override;
  void UnrelaxedModulus(mfem::ElementTransformation& T,
                        const mfem::IntegrationPoint& ip,
                        mfem::DenseMatrix& CU) const override;
  std::unique_ptr<ElasticStiffness> MakeStiffness() const override;

  mfem::Coefficient& BulkModulus() const { return *kappa_; }

  /** @brief @f$\mu_\infty@f$. */
  mfem::Coefficient& LongTermShearModulus() const { return *mu_inf_; }

  /** @brief @f$\mu_U = \mu_\infty + \sum_k \mu_k@f$; the modulus the elastic
   * operator is assembled with when unrelaxed. */
  mfem::Coefficient& UnrelaxedShearModulus() const { return *mu_u_; }

  /** @brief @f$\lambda_U = \kappa - 2\mu_U/d@f$, for (lambda, mu) form. */
  mfem::Coefficient& UnrelaxedLame() const { return *lambda_u_; }

  /** @brief The instantaneous (@f$t = 0^+@f$) elastic solid @f$(\kappa,
   * \mu_U)@f$. Refers to this object's coefficients, which must outlive
   * it. */
  IsotropicElasticRheology UnrelaxedElastic() const;

  /** @brief The fully relaxed (@f$t \to \infty@f$) elastic solid
   * @f$(\kappa, \mu_\infty)@f$. Refers to this object's coefficients, which
   * must outlive it. */
  IsotropicElasticRheology LongTermElastic() const;

  const MaxwellBranch& Branch(int k) const { return branches_[k]; }

 private:
  int dim_;
  mfem::Coefficient* kappa_;
  mfem::Coefficient* mu_inf_;
  std::vector<MaxwellBranch> branches_;
  std::vector<std::unique_ptr<mfem::Coefficient>> owned_;
  mfem::Coefficient* mu_u_ = nullptr;
  mfem::Coefficient* lambda_u_ = nullptr;
};

/**
 * @brief One Prony branch of an anisotropic body: the relaxable tensor
 * @f$C_k@f$ (an @f$n_s \times n_s@f$ Mandel MatrixCoefficient in
 * SymmetricTensorBasis ordering, e.g. a
 * DeviatoricProjectionElasticTensorCoefficient or a
 * TransverselyIsotropicElasticTensorCoefficient with only L and N) and its
 * relaxation time, with an optional RelaxationLaw as for MaxwellBranch.
 * Non-owning.
 */
struct AnisotropicBranch {
  mfem::MatrixCoefficient* C = nullptr;
  mfem::Coefficient* tau = nullptr;
  const RelaxationLaw* law = nullptr;
};

/**
 * @brief Anisotropic generalised Maxwell rheology,
 * @f$\sigma = C_U \varepsilon - \sum_k C_k m_k@f$ with @f$C_U = C_\infty +
 * \sum_k C_k@f$, all tensors given as Mandel MatrixCoefficients (see
 * elastic_tensor.hpp) and the internal variables full symmetric tensors
 * evolving as @f$\dot m_k = (\varepsilon - m_k)/\tau_k@f$. A purely
 * elastic solid is AnisotropicElasticRheology.
 *
 * The stiffness is one ElasticTensorIntegrator with @f$C_U@f$ or, with
 * relaxation weights, @f$C_\infty + \sum_k \beta_k C_k@f$ through MFEM's
 * matrix-coefficient algebra. Which part of a tensor relaxes is the
 * modelling choice of the branch coefficient; DeviatoricMaxwell() makes the
 * choice that reproduces the isotropic Maxwell body for an isotropic
 * @f$C@f$ (relax @f$P_{dev} C P_{dev}@f$).
 *
 * Holds pointers to the caller's coefficients plus the sums it builds;
 * movable, not copyable.
 */
class AnisotropicMaxwellRheology : public Rheology {
 public:
  /**
   * @param dim Space dimension.
   * @param C_inf Long-term tensor @f$C_\infty@f$.
   * @param branches Relaxable tensors and relaxation times.
   */
  AnisotropicMaxwellRheology(int dim, mfem::MatrixCoefficient& C_inf,
                             const std::vector<AnisotropicBranch>& branches);

  /**
   * @brief Maxwell body relaxing the deviatoric part of the tensor
   * @f$C@f$, with one branch @f$C_1 = P_{dev} C P_{dev}@f$ and
   * @f$C_\infty = C - C_1@f$ (owned coefficients). For an isotropic
   * @f$C@f$ this is IsotropicMaxwellRheology::Maxwell().
   */
  static AnisotropicMaxwellRheology DeviatoricMaxwell(
      int dim, mfem::MatrixCoefficient& C, mfem::Coefficient& tau,
      const RelaxationLaw* law = nullptr);

  AnisotropicMaxwellRheology(AnisotropicMaxwellRheology&&) = default;
  AnisotropicMaxwellRheology& operator=(AnisotropicMaxwellRheology&&) = default;
  AnisotropicMaxwellRheology(const AnisotropicMaxwellRheology&) = delete;
  AnisotropicMaxwellRheology& operator=(const AnisotropicMaxwellRheology&) =
      delete;

  int SpaceDim() const override { return dim_; }
  int NumBranches() const override {
    return static_cast<int>(branches_.size());
  }
  mfem::Coefficient& RelaxationTime(int k) const override {
    return *branches_[k].tau;
  }
  const RelaxationLaw* Law(int k) const override { return branches_[k].law; }
  bool TraceFreeInternalVariables() const override { return false; }
  void BranchModulus(int k, mfem::ElementTransformation& T,
                     const mfem::IntegrationPoint& ip,
                     mfem::DenseMatrix& Ck) const override {
    branches_[k].C->Eval(Ck, T, ip);
  }
  void UnrelaxedModulus(mfem::ElementTransformation& T,
                        const mfem::IntegrationPoint& ip,
                        mfem::DenseMatrix& CU) const override {
    C_u_->Eval(CU, T, ip);
  }
  std::unique_ptr<ElasticStiffness> MakeStiffness() const override;

  mfem::MatrixCoefficient& LongTermTensor() const { return *C_inf_; }

  /** @brief @f$C_U = C_\infty + \sum_k C_k@f$. */
  mfem::MatrixCoefficient& UnrelaxedTensor() const { return *C_u_; }

  /** @brief The instantaneous elastic solid with tensor @f$C_U@f$. Refers
   * to this object's coefficients, which must outlive it. */
  AnisotropicElasticRheology UnrelaxedElastic() const;

  /** @brief The fully relaxed elastic solid with tensor @f$C_\infty@f$.
   * Refers to this object's coefficients, which must outlive it. */
  AnisotropicElasticRheology LongTermElastic() const;

  const AnisotropicBranch& Branch(int k) const { return branches_[k]; }

 private:
  int dim_;
  mfem::MatrixCoefficient* C_inf_;
  std::vector<AnisotropicBranch> branches_;
  std::vector<std::unique_ptr<mfem::MatrixCoefficient>> owned_;
  mfem::MatrixCoefficient* C_u_ = nullptr;
};

// ---------------------------------------------------------------------------
// Composite rheology

/**
 * @brief One region of a CompositeRheology: a set of element attributes of
 * the displacement mesh and the rheology that holds there. The rheology is
 * not owned and must outlive the composite. The optional name appears in
 * the output field names of the region's internal variables (default
 * "region<r>").
 */
struct RheologyRegion {
  mfem::Array<int> marker;             ///< sized to mesh.attributes.Max()
  const Rheology* rheology = nullptr;  ///< not owned
  std::string name;
};

/**
 * @brief Different rheologies in different regions of one body.
 *
 * Regions are sets of element attributes of the displacement mesh; they
 * must be disjoint (checked at construction) and cover every attribute
 * present on the mesh (checked when the stiffness is attached to a form).
 * A region's rheology may be elastic (no branches), a Maxwell body with any
 * number of branches, isotropic or anisotropic, with or without relaxation
 * laws; its coefficients need only be meaningful inside the region.
 *
 * The composite's branches are the concatenation of the regions' branches
 * in region order: global branch @f$k@f$ is local branch LocalBranch(k) of
 * region BranchRegion(k). Its relaxable modulus vanishes outside its region
 * (BranchShearModulus() and BranchModulus() are masked), so the internal
 * variable there never enters the stress; the relaxation time outside is a
 * large dummy, so that the internal variable stays put and never limits an
 * explicit step. Internal variables are trace-free only when every region's
 * are; otherwise the isotropic regions present their branches as full
 * tensors @f$2\mu_k P_{dev}@f$. The stiffness is the sum of the regions'
 * stiffnesses, each restricted to its marker; SetRelaxationWeights() takes
 * one weight per global branch.
 *
 * The viscoelastic operator stores and evolves each branch's internal
 * variable on its region only (BranchMarker()), so the state and the
 * per-step work are the sum over regions of their own branches. Movable,
 * not copyable.
 */
class CompositeRheology : public Rheology {
 public:
  /**
   * @param dim Space dimension (2 or 3); every region's rheology must agree.
   * @param regions Disjoint regions; the markers must have equal size.
   */
  CompositeRheology(int dim, std::vector<RheologyRegion> regions);

  CompositeRheology(CompositeRheology&&) = default;
  CompositeRheology& operator=(CompositeRheology&&) = default;
  CompositeRheology(const CompositeRheology&) = delete;
  CompositeRheology& operator=(const CompositeRheology&) = delete;

  // --- regions --------------------------------------------------------------

  int NumRegions() const { return static_cast<int>(regions_.size()); }
  const Rheology& Region(int r) const { return *regions_[r].rheology; }
  const mfem::Array<int>& RegionMarker(int r) const {
    return regions_[r].marker;
  }
  const std::string& RegionName(int r) const { return regions_[r].name; }

  /** @brief Region containing element attribute @p attribute, or -1. */
  int RegionOf(int attribute) const;

  /** @brief Region owning global branch @p k. */
  int BranchRegion(int k) const { return branch_region_[k]; }

  /** @brief Index of global branch @p k within its region. */
  int LocalBranch(int k) const { return local_branch_[k]; }

  /** @brief Global index of the first branch of region @p r (the region's
   * branches are contiguous). */
  int RegionBranchOffset(int r) const { return region_offset_[r]; }

  /** @brief Abort unless every element attribute present on @p mesh belongs
   * to a region (attributes absent from the mesh need not). */
  void VerifyCoverage(const mfem::Mesh& mesh) const;

  // --- Rheology -------------------------------------------------------------

  int SpaceDim() const override { return dim_; }
  int NumBranches() const override {
    return static_cast<int>(branch_region_.size());
  }
  mfem::Coefficient& RelaxationTime(int k) const override { return *tau_[k]; }
  const RelaxationLaw* Law(int k) const override {
    return Region(branch_region_[k]).Law(local_branch_[k]);
  }
  void UnrelaxedModulus(mfem::ElementTransformation& T,
                        const mfem::IntegrationPoint& ip,
                        mfem::DenseMatrix& CU) const override;
  bool TraceFreeInternalVariables() const override { return tracefree_; }
  mfem::Coefficient& BranchShearModulus(int k) const override;
  void BranchModulus(int k, mfem::ElementTransformation& T,
                     const mfem::IntegrationPoint& ip,
                     mfem::DenseMatrix& Ck) const override;
  std::unique_ptr<ElasticStiffness> MakeStiffness() const override;

  /** @brief "<region name>_<branch label of the region's rheology>". */
  std::string BranchLabel(int k) const override;

  /** @brief The marker of the branch's region. */
  const mfem::Array<int>* BranchMarker(int k) const override {
    return &regions_[branch_region_[k]].marker;
  }

  /** @brief The relaxation time used outside a branch's region. */
  static constexpr mfem::real_t kOutsideRelaxationTime = 1e300;

 private:
  int dim_;
  std::vector<RheologyRegion> regions_;
  std::vector<int> attribute_region_;  ///< by attribute - 1; -1 = none
  std::vector<int> branch_region_, local_branch_, region_offset_;
  bool tracefree_ = true;
  // Per global branch: the region's tau (dummy outside) and, when
  // trace-free, the region's mu_k (zero outside).
  std::vector<std::unique_ptr<mfem::Coefficient>> tau_, mu_;
};

}  // namespace AdGIA
