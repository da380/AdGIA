/**
 * @file mixed_problem.cpp
 * @brief Implementation of LinearQuasiStaticMixedSelfGravitatingProblem.
 */

#include "AdGIA/mixed_problem.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numbers>
#include <random>

#include "AdGIA/bilininteg.hpp"
#include "AdGIA/centrifugal.hpp"
#include "AdGIA/detail/fem_factory.hpp"
#include "AdGIA/mesh.hpp"

namespace AdGIA {

using namespace mfem;

namespace {
constexpr real_t kPi = std::numbers::pi_v<real_t>;
constexpr real_t kMinInnerRelTol = 1e-13;
}

// ---------------------------------------------------------------------------
// Construction

LinearQuasiStaticMixedSelfGravitatingProblem::
    LinearQuasiStaticMixedSelfGravitatingProblem(
        FiniteElementSpace* fes_u, FiniteElementSpace* fes_phi,
        const AdGIA::Rheology& rheology, Coefficient& density,
        real_t gravitational_constant, int dtn_degree,
        Coefficient* background_potential,
        const std::vector<FluidRegion>& fluids)
    : LinearQuasiStaticProblemBase(fes_u, rheology),
      dim_(fes_u->GetMesh()->Dimension()),
      fes_phi_(fes_phi),
      rho_(&density),
      G_(gravitational_constant),
      four_pi_G_(4.0 * kPi * gravitational_constant),
      dtn_degree_(dtn_degree),
      one_(1.0),
      inv_four_pi_G_(1.0 / (4.0 * kPi * gravitational_constant)),
      shift_coef_(shift_ / (4.0 * kPi * gravitational_constant)),
      fluids_(fluids),
      K_phi_(Operator::MFEM_SPARSEMAT),
      K_shift_(Operator::MFEM_SPARSEMAT),
      M_fluid_(Operator::MFEM_SPARSEMAT),
      C_(Operator::MFEM_SPARSEMAT) {
  MFEM_VERIFY(G_ > 0.0,
              "LinearQuasiStaticMixedSelfGravitatingProblem: G must be positive.");
  MFEM_VERIFY(
      fes_phi_->GetVDim() == 1,
      "LinearQuasiStaticMixedSelfGravitatingProblem: the potential space must be "
      "scalar.");
  MFEM_VERIFY(
      fes_phi_->GetMesh()->Dimension() == dim_,
      "LinearQuasiStaticMixedSelfGravitatingProblem: mesh dimensions differ.");

#ifdef MFEM_USE_MPI
  pfes_phi_ = dynamic_cast<ParFiniteElementSpace*>(fes_phi_);
  MFEM_VERIFY(
      (pfes_ != nullptr) == (pfes_phi_ != nullptr),
      "LinearQuasiStaticMixedSelfGravitatingProblem: the displacement and potential "
      "spaces must both be serial or both be parallel.");
  if (pfes_phi_) {
    auto* psub = dynamic_cast<ParSubMesh*>(pfes_->GetParMesh());
    MFEM_VERIFY(psub,
                "LinearQuasiStaticMixedSelfGravitatingProblem: the displacement "
                "space must live on a ParSubMesh.");
    MFEM_VERIFY(
        psub->GetParent() == pfes_phi_->GetParMesh(),
        "LinearQuasiStaticMixedSelfGravitatingProblem: the potential space must "
        "live on the parent of the displacement SubMesh.");
    shadow_phi_ = SubMeshDofInjection::MakeShadowSpace(*pfes_phi_, *psub);
    K_phi_.SetType(Operator::Hypre_ParCSR);
    K_shift_.SetType(Operator::Hypre_ParCSR);
    M_fluid_.SetType(Operator::Hypre_ParCSR);
    C_.SetType(Operator::Hypre_ParCSR);
  } else
#endif
  {
    auto* sub = dynamic_cast<SubMesh*>(fes_->GetMesh());
    MFEM_VERIFY(
        sub,
        "LinearQuasiStaticMixedSelfGravitatingProblem: the displacement space "
        "must live on a SubMesh.");
    MFEM_VERIFY(
        sub->GetParent() == fes_phi_->GetMesh(),
        "LinearQuasiStaticMixedSelfGravitatingProblem: the potential space must "
        "live on the parent of the displacement SubMesh.");
    shadow_phi_ = SubMeshDofInjection::MakeShadowSpace(*fes_phi_, *sub);
  }
  injection_ = std::make_unique<SubMeshDofInjection>(*shadow_phi_, *fes_phi_);

  // Fluid regions: markers on the parent's attributes and the SubMesh's
  // boundary attributes.
  for (auto& f : fluids_) {
    MFEM_VERIFY(f.density,
                "LinearQuasiStaticMixedSelfGravitatingProblem: a fluid region "
                "needs a density.");
    MFEM_VERIFY(f.attributes.Size() > 0,
                "LinearQuasiStaticMixedSelfGravitatingProblem: a fluid region needs "
                "parent-mesh attributes.");
    MFEM_VERIFY(
        f.interface_marker.Size() == fes_->GetMesh()->bdr_attributes.Max(),
        "LinearQuasiStaticMixedSelfGravitatingProblem: a fluid region's "
        "interface_marker must be sized to the SubMesh's "
        "bdr_attributes.Max().");
    // The CMB-approximation ladder: a non-Full treatment owns its
    // degradations, so the fields it prescribes must be left at their
    // defaults (the ladder is defined in doc/quasi_static_models.tex,
    // "Fluid regions and the ladder of CMB approximations").
    if (f.treatment == FluidRegion::Treatment::Full) {
      MFEM_VERIFY(f.uniform_density == 0.0,
                  "LinearQuasiStaticMixedSelfGravitatingProblem: "
                  "uniform_density requires Treatment::UniformCore or "
                  "Treatment::Winkler.");
    } else {
      MFEM_VERIFY(!f.density_gradient,
                  "LinearQuasiStaticMixedSelfGravitatingProblem: a non-Full "
                  "treatment prescribes density_gradient; leave it null.");
      MFEM_VERIFY(f.interface_potential_coupling,
                  "LinearQuasiStaticMixedSelfGravitatingProblem: leave "
                  "interface_potential_coupling true; Treatment::Winkler "
                  "drops it itself.");
      treatment_coefs_.push_back(std::make_unique<ConstantCoefficient>(0.0));
      f.density_gradient = treatment_coefs_.back().get();
      if (f.treatment == FluidRegion::Treatment::NoMass) {
        MFEM_VERIFY(f.uniform_density == 0.0,
                    "LinearQuasiStaticMixedSelfGravitatingProblem: "
                    "uniform_density requires Treatment::UniformCore or "
                    "Treatment::Winkler.");
      } else {
        MFEM_VERIFY(!f.interface_density,
                    "LinearQuasiStaticMixedSelfGravitatingProblem: "
                    "Treatment::UniformCore/Winkler prescribe "
                    "interface_density; supply uniform_density instead.");
        MFEM_VERIFY(f.uniform_density > 0.0,
                    "LinearQuasiStaticMixedSelfGravitatingProblem: "
                    "Treatment::UniformCore/Winkler need a positive "
                    "uniform_density.");
        treatment_coefs_.push_back(
            std::make_unique<ConstantCoefficient>(f.uniform_density));
        f.interface_density = treatment_coefs_.back().get();
      }
      f.interface_potential_coupling =
          (f.treatment != FluidRegion::Treatment::Winkler);
    }
    Array<int> marker(fes_phi_->GetMesh()->attributes.Max());
    marker = 0;
    for (int a : f.attributes) {
      MFEM_VERIFY(
          a >= 1 && a <= marker.Size(),
          "LinearQuasiStaticMixedSelfGravitatingProblem: fluid attribute out of "
          "range.");
      marker[a - 1] = 1;
    }
    fluid_markers_.push_back(marker);
  }

  // Potential fields.
  phi0_ = detail::MakeGridFunction(fes_phi_);
  phi0_shadow_ = detail::MakeGridFunction(shadow_phi_.get());
  phi_ = detail::MakeGridFunction(fes_phi_);
  phi_shadow_ = detail::MakeGridFunction(shadow_phi_.get());
  *phi0_ = 0.0;
  *phi0_shadow_ = 0.0;
  *phi_ = 0.0;
  *phi_shadow_ = 0.0;
  grad_phi0_ = std::make_unique<GradientGridFunctionCoefficient>(phi0_.get());
  grad_phi0_shadow_ =
      std::make_unique<GradientGridFunctionCoefficient>(phi0_shadow_.get());
  Phi_true_.SetSize(fes_phi_->GetTrueVSize());
  Phi_true_ = 0.0;

  // The DtN operator on the outer boundary.
#ifdef MFEM_USE_MPI
  if (pfes_phi_) {
    dtn_ = std::make_unique<PoissonDtNOperator>(pfes_phi_->GetComm(),
                                                pfes_phi_, dtn_degree_);
    dtn_->Assemble();
    dtn_rap_ = std::make_unique<RAPOperator>(dtn_->RAP());
    dtn_op_ = dtn_rap_.get();
  } else
#endif
  {
    dtn_ = std::make_unique<PoissonDtNOperator>(fes_phi_, dtn_degree_);
    dtn_->Assemble();
    dtn_op_ = dtn_.get();
  }

  // Data for the 2-D compatibility condition: the constant and the outer
  // boundary functional, on true dofs.
  if (dim_ == 2) {
    auto ones = detail::MakeGridFunction(fes_phi_);
    *ones = 1.0;
    ones->GetTrueDofs(ones_);
    auto marker = ExternalBoundaryMarker(fes_phi_->GetMesh());
    auto l = detail::MakeLinearForm(fes_phi_);
    l->AddBoundaryIntegrator(new BoundaryLFIntegrator(one_), marker);
    l->Assemble();
    ToTrueDofs(*fes_phi_, *l, L_outer_);
    outer_length_ = Dot(L_outer_, ones_);
    MFEM_VERIFY(
        outer_length_ > 0.0,
        "LinearQuasiStaticMixedSelfGravitatingProblem: empty outer boundary.");
  }

  SetupPotentialOperators();
  ComputeBackgroundPotential(background_potential);
  SetupFluidMass();
  SetupCoupling();
  SetupGravityIntegrators();
  SetupRigidModes();
  SetupCoupledNullSpace();

  b_phi_ = detail::MakeLinearForm(shadow_phi_.get());
  B_phi_.SetSize(fes_phi_->GetTrueVSize());
  B_phi_ = 0.0;
}

bool LinearQuasiStaticMixedSelfGravitatingProblem::ParallelPotential() const {
#ifdef MFEM_USE_MPI
  return pfes_phi_ != nullptr;
#else
  return false;
#endif
}

void LinearQuasiStaticMixedSelfGravitatingProblem::ToTrueDofs(
    const FiniteElementSpace& fes, const Vector& L, Vector& T) const {
  const Operator* P = fes.GetProlongationMatrix();
  if (P) {
    T.SetSize(P->Width());
    P->MultTranspose(L, T);
  } else {
    T = L;
  }
}

void LinearQuasiStaticMixedSelfGravitatingProblem::SetupPotentialOperators() {
  Array<int> empty;

  // K = (grad phi, grad psi) on the ball; A_phiphi = (K + DtN) / (4 pi G).
  k_phi_form_ = detail::MakeBilinearForm(fes_phi_);
  k_phi_form_->AddDomainIntegrator(new DiffusionIntegrator(one_));
  k_phi_form_->Assemble();
  k_phi_form_->FormSystemMatrix(empty, K_phi_);
  const real_t c = 1.0 / four_pi_G_;
  A_lap_ = std::make_unique<SumOperator>(K_phi_.Ptr(), c, dtn_op_, c, false,
                                         false);
  A_phiphi_ = A_lap_.get();

  SetupPotentialPreconditioner();
  SetupPotentialSolver();
}

void LinearQuasiStaticMixedSelfGravitatingProblem::SetupPotentialSolver() {
  // CG on the current potential block A_phiphi_ (the operator must be set
  // before the preconditioner: BoomerAMG's SetOperator only accepts a
  // HypreParMatrix). In 2-D the constant is projected out on both sides,
  // i.e. CG runs on P A_phiphi P, the same restriction as in the block
  // solver.
#ifdef MFEM_USE_MPI
  if (pfes_phi_) {
    cg_phi_ = std::make_unique<CGSolver>(pfes_phi_->GetComm());
    if (dim_ == 2 && !projector_c_) {
      projector_c_ = std::make_unique<NullSpaceProjector>(pfes_phi_->GetComm());
    }
  } else
#endif
  {
    cg_phi_ = std::make_unique<CGSolver>();
    if (dim_ == 2 && !projector_c_) {
      projector_c_ = std::make_unique<NullSpaceProjector>();
    }
  }
  if (dim_ == 2) {
    if (projector_c_->Size() == 0) {
      projector_c_->Add(ones_);
    }
    projected_phi_op_ =
        std::make_unique<ProjectedOperator>(*A_phiphi_, *projector_c_);
    cg_phi_->SetOperator(*projected_phi_op_);
    // The preconditioner is projected too: (K + eps M)^{-1} amplifies the
    // round-off constant component of the residual by 1/eps per iteration,
    // and CG on the singular projected operator then diverges once the
    // residual reaches round-off (seen with BoomerAMG).
    projected_prec_phi_ = std::make_unique<ProjectedSolver>(*projector_c_);
    projected_prec_phi_->SetSolver(*prec_phi_);
    cg_phi_->SetPreconditioner(*projected_prec_phi_);
  } else {
    cg_phi_->SetOperator(*A_phiphi_);
    cg_phi_->SetPreconditioner(*prec_phi_);
  }
  cg_phi_->SetRelTol(inner_rel_tol_);
  cg_phi_->SetAbsTol(0.0);
  cg_phi_->SetMaxIter(10000);
  cg_phi_->SetPrintLevel(inner_print_level_);
  cg_phi_->iterative_mode = false;
  if (dim_ == 2) {
    projected_phi_ = std::make_unique<ProjectedSolver>(*projector_c_);
    projected_phi_->SetSolver(*cg_phi_);
    projected_phi_->iterative_mode = false;
    phi_solver_ = projected_phi_.get();
  } else {
    phi_solver_ = cg_phi_.get();
  }
}

void LinearQuasiStaticMixedSelfGravitatingProblem::SetupPotentialPreconditioner() {
  // (Re)build the shifted-Laplacian preconditioner; the solver, if it
  // exists, is rebuilt on it by the caller through SetupPotentialSolver().
  Array<int> empty;
  shift_coef_.constant = shift_ / four_pi_G_;
  k_shift_form_ = detail::MakeBilinearForm(fes_phi_);
  k_shift_form_->AddDomainIntegrator(new DiffusionIntegrator(inv_four_pi_G_));
  k_shift_form_->AddDomainIntegrator(new MassIntegrator(shift_coef_));
  k_shift_form_->Assemble();
  k_shift_form_->FormSystemMatrix(empty, K_shift_);
#ifdef MFEM_USE_MPI
  if (pfes_phi_) {
    auto amg = std::make_unique<HypreBoomerAMG>(*K_shift_.As<HypreParMatrix>());
    amg->SetPrintLevel(0);
    prec_phi_ = std::move(amg);
  } else
#endif
  {
    prec_phi_ = std::make_unique<GSSmoother>(*K_shift_.As<SparseMatrix>());
  }
  if (cg_phi_) {
    SetupPotentialSolver();
  }
}

void LinearQuasiStaticMixedSelfGravitatingProblem::MakeCompatible(
    Vector& B_phi) const {
  if (dim_ != 2) {
    return;
  }
  const real_t mass = Dot(B_phi, ones_);
  B_phi.Add(-mass / outer_length_, L_outer_);
}

void LinearQuasiStaticMixedSelfGravitatingProblem::AddPotentialForce(
    const Vector& f) {
  MFEM_VERIFY(f.Size() == fes_phi_->GetVSize(),
              "AddPotentialForce: the dual must be in the L-vector layout "
              "of the potential space.");
  Vector ft(fes_phi_->GetTrueVSize());
  ToTrueDofs(*fes_phi_, f, ft);
  // The increment deliberately does NOT feed the border data (the mass
  // row's sea_Sd_ or the omega rows' rot_r_): the bordered system is
  // symmetric with (Phi_g, omega) among its unknowns, so the exact
  // discrete adjoint of an observation J = f . Phi loads ONLY the
  // potential block — the continuum reading of Yu et al. (2025,
  // eq. 58), where a D_phi J datum stands beside the surface mass in
  // the jump condition, is realised automatically through the
  // symmetric border coupling columns. (Feeding the rows as if the
  // dual were a registered surface load double-counts: measured as a
  // 10 % error on the full-physics potential-observation FD gate of
  // TestAdjointFingerprint, which gates this choice.) The forward's
  // surface loads DO feed those rows, as data of those equations.
  MakeCompatible(ft);
  B_phi_ += ft;
}

void LinearQuasiStaticMixedSelfGravitatingProblem::AddBorderForce(
    real_t uniform_row, const Vector* omega_rows) {
  if (uniform_row != 0.0) {
    MFEM_VERIFY(sea_enabled_,
                "AddBorderForce: a uniform-term datum needs the water "
                "border (SetWaterLoad).");
    border_r0_add_ += uniform_row;
  }
  if (omega_rows) {
    MFEM_VERIFY(rot_enabled_,
                "AddBorderForce: omega-row data need the rotational "
                "border (SetRotation).");
    MFEM_VERIFY(omega_rows->Size() == rot_r_.Size(),
                "AddBorderForce: omega data sized to the border width.");
    if (border_rot_add_.Size() != rot_r_.Size()) {
      border_rot_add_.SetSize(rot_r_.Size());
      border_rot_add_ = 0.0;
    }
    border_rot_add_ += *omega_rows;
  }
}

bool LinearQuasiStaticMixedSelfGravitatingProblem::SolvePotential(const Vector& b,
                                                             Vector& x) const {
  x.SetSize(b.Size());
  x = 0.0;
  phi_solver_->Mult(b, x);
  inner_its_ += cg_phi_->GetNumIterations();
  return cg_phi_->GetConverged();
}

bool LinearQuasiStaticMixedSelfGravitatingProblem::SolveLoadPotential(
    GridFunction& phi_body) {
  MFEM_VERIFY(phi_body.FESpace() == shadow_phi_.get(),
              "SolveLoadPotential: the field must be on PotentialSpaceOnBody().");
  EnsureOperator();
  // With the plain Laplace-DtN operator: the fluid mass term is part of the
  // body's response, not of the load's own potential. The swap rebuilds
  // only the CG wrapper and its pointer wiring — the preconditioner is
  // the shifted-Laplacian AMG, shared by every potential operator, so
  // no multigrid setup is paid here.
  Vector Phi;
  bool ok;
  if (A_phiphi_ == A_lap_.get()) {
    ok = SolvePotential(B_phi_, Phi);
  } else {
    auto* saved = A_phiphi_;
    A_phiphi_ = A_lap_.get();
    SetupPotentialSolver();
    ok = SolvePotential(B_phi_, Phi);
    A_phiphi_ = saved;
    SetupPotentialSolver();
  }
  auto tmp = detail::MakeGridFunction(fes_phi_);
  tmp->SetFromTrueDofs(Phi);
  injection_->MultTranspose(*tmp, phi_body);
  return ok;
}

void LinearQuasiStaticMixedSelfGravitatingProblem::DistributePotential(
    const Vector& Phi) {
  phi_->SetFromTrueDofs(Phi);
  injection_->MultTranspose(*phi_, *phi_shadow_);
}

void LinearQuasiStaticMixedSelfGravitatingProblem::ComputeBackgroundPotential(
    Coefficient* phi0) {
  if (phi0) {
    phi0_->ProjectCoefficient(*phi0);
  } else {
    // (K + DtN) Phi0 = -4 pi G (rho, psi)_M, i.e. A_lap Phi0 = -(rho, psi)_M
    // with the solid density integrated on the SubMesh and injected into
    // the ball, and the fluid densities integrated on the parent. Solved
    // with the plain Laplace-DtN operator A_lap: the fluid mass term is a
    // perturbation quantity and does not enter the background problem. At
    // construction A_phiphi is A_lap; once the fluid mass term has been
    // assembled the potential solver is switched to A_lap for this solve.
    auto rho_form = detail::MakeLinearForm(shadow_phi_.get());
    rho_form->AddDomainIntegrator(new DomainLFIntegrator(*rho_));
    rho_form->Assemble();
    Vector bL(fes_phi_->GetVSize()), B;
    injection_->Mult(*rho_form, bL);
    if (!fluids_.empty()) {
      auto fluid_form = detail::MakeLinearForm(fes_phi_);
      auto marker = fluid_markers_.begin();
      for (auto& f : fluids_) {
        fluid_form->AddDomainIntegrator(new DomainLFIntegrator(*f.density),
                                        *marker);
        ++marker;
      }
      fluid_form->Assemble();
      bL += *fluid_form;
    }
    ToTrueDofs(*fes_phi_, bL, B);
    B *= -1.0;
    MakeCompatible(B);
    Vector Phi0;
    bool ok;
    if (A_phiphi_ == A_lap_.get()) {
      ok = SolvePotential(B, Phi0);
    } else {
      auto* saved = A_phiphi_;
      A_phiphi_ = A_lap_.get();
      SetupPotentialSolver();
      ok = SolvePotential(B, Phi0);
      A_phiphi_ = saved;
      SetupPotentialSolver();
    }
    MFEM_VERIFY(ok,
                "LinearQuasiStaticMixedSelfGravitatingProblem: the background "
                "potential solve did not converge.");
    phi0_->SetFromTrueDofs(Phi0);
  }
  injection_->MultTranspose(*phi0_, *phi0_shadow_);
}

void LinearQuasiStaticMixedSelfGravitatingProblem::SetupFluidMass() {
  // M_F = int_{M_F} rho'_F phi chi on the parent's fluid elements, with
  // rho'_F from the region or, by default, from the element-wise L2
  // projection of the density and the discrete grad Phi0. Then
  // A_phiphi = A_lap + M_F and the potential solver is rebuilt on it.
  if (fluids_.empty()) {
    return;
  }
  rho_fluid_l2_.clear();
  fluid_coefs_.clear();
  fluid_rho_prime_.clear();
  m_fluid_form_ = detail::MakeBilinearForm(fes_phi_);
  auto marker = fluid_markers_.begin();
  for (auto& f : fluids_) {
    Coefficient* rho_prime = f.density_gradient;
    if (!rho_prime) {
      if (!l2_fes_) {
        l2_fec_ = std::make_unique<L2_FECollection>(
            fes_phi_->GetMaxElementOrder(), dim_);
        l2_fes_ = detail::MakeFESpace(*fes_phi_, l2_fec_.get());
      }
      auto rho_l2 = detail::MakeGridFunction(l2_fes_.get());
      rho_l2->ProjectCoefficient(*f.density);
      auto rp = std::make_unique<BarotropicDensityGradientCoefficient>(
          *rho_l2, *phi0_);
      rho_prime = rp.get();
      rho_fluid_l2_.push_back(std::move(rho_l2));
      fluid_coefs_.push_back(std::move(rp));
    }
    m_fluid_form_->AddDomainIntegrator(new MassIntegrator(*rho_prime),
                                       *marker);
    fluid_rho_prime_.push_back(rho_prime);
    ++marker;
  }
  m_fluid_form_->Assemble();
  Array<int> empty;
  m_fluid_form_->FormSystemMatrix(empty, M_fluid_);
  A_full_ = std::make_unique<SumOperator>(A_lap_.get(), 1.0, M_fluid_.Ptr(),
                                          1.0, false, false);
  A_phiphi_ = A_full_.get();
  SetupPotentialSolver();
}

void LinearQuasiStaticMixedSelfGravitatingProblem::SetupCoupling() {
  // C = int_{M_S} rho grad phi . v  -  sum_F int_{Sigma_F} rho_F phi (m.v),
  // trial phi on B, test v on M_S; C^T by transposition.
  Array<int> empty;
  auto add_interfaces = [&](MixedBilinearForm& form) {
    for (auto& f : fluids_) {
      if (!f.interface_potential_coupling) {
        continue;  // buoyancy-only (Winkler) interface: no potential coupling
      }
      auto minus_rho = std::make_unique<ProductCoefficient>(
          -1.0, InterfaceDensity(f));
      form.AddBoundaryIntegrator(
          new BoundaryNormalScalarIntegrator(*minus_rho), f.interface_marker);
      fluid_coefs_.push_back(std::move(minus_rho));
    }
    if (sea_enabled_) {
      // The u-phi water block: -int w (m.grad Phi0) phi (m.v) dS; its
      // transpose (the phi-row) comes with C^T below.
      auto q_g =
          std::make_unique<BoundaryNormalDotCoefficient>(*grad_phi0_shadow_);
      auto minus_wqg =
          std::make_unique<ProductCoefficient>(*sea_w_, *q_g);
      auto m_wqg = std::make_unique<ProductCoefficient>(-1.0, *minus_wqg);
      form.AddBoundaryIntegrator(new BoundaryNormalScalarIntegrator(*m_wqg),
                                 sea_marker_);
      sea_coefs_.push_back(std::move(q_g));
      sea_coefs_.push_back(std::move(minus_wqg));
      sea_coefs_.push_back(std::move(m_wqg));
    }
  };
#ifdef MFEM_USE_MPI
  if (pfes_phi_) {
    auto form =
        std::make_unique<ParSubMeshMixedBilinearForm>(pfes_phi_, pfes_);
    form->AddDomainIntegrator(new GradientIntegrator(*rho_));
    add_interfaces(*form);
    form->Assemble();
    form->FormRectangularSystemMatrix(empty, empty, C_);
    c_form_ = std::move(form);
    Ct_owned_.reset(C_.As<HypreParMatrix>()->Transpose());
  } else
#endif
  {
    auto form = std::make_unique<SubMeshMixedBilinearForm>(fes_phi_, fes_);
    form->AddDomainIntegrator(new GradientIntegrator(*rho_));
    add_interfaces(*form);
    form->Assemble();
    form->FormRectangularSystemMatrix(empty, empty, C_);
    c_form_ = std::move(form);
    Ct_owned_.reset(Transpose(*C_.As<SparseMatrix>()));
  }
  C_op_ = C_.Ptr();
  Ct_op_ = Ct_owned_.get();
}

void LinearQuasiStaticMixedSelfGravitatingProblem::SetupGravityIntegrators() {
  half_rho_ = std::make_unique<ProductCoefficient>(0.5, *rho_);
  minus_half_rho_ = std::make_unique<ProductCoefficient>(-0.5, *rho_);
  minus_half_rho_grad_ = std::make_unique<ScalarVectorProductCoefficient>(
      *minus_half_rho_, *grad_phi0_shadow_);
  auto* g1 =
      new DomainVectorGradVectorIntegrator(*grad_phi0_shadow_, *half_rho_);
  auto* g2 = new DomainVectorDivVectorIntegrator(*minus_half_rho_grad_);
  auto& integs = StiffnessIntegrators();
  integs.AddDomainIntegrator(g1);
  integs.AddDomainIntegrator(g2);
  integs.AddDomainIntegrator(new TransposeIntegrator(g1, 0));
  integs.AddDomainIntegrator(new TransposeIntegrator(g2, 0));

  // Interface term -int_{Sigma_F} rho_F (m . grad Phi0) (m.u)(m.v).
  if (!fluids_.empty()) {
    m_dot_grad_phi0_ =
        std::make_unique<BoundaryNormalDotCoefficient>(*grad_phi0_shadow_);
    for (auto& f : fluids_) {
      auto q = std::make_unique<ProductCoefficient>(InterfaceDensity(f),
                                                    *m_dot_grad_phi0_);
      auto minus_q = std::make_unique<ProductCoefficient>(-1.0, *q);
      integs.AddBoundaryIntegrator(new BoundaryNormalNormalIntegrator(*minus_q),
                                   f.interface_marker);
      fluid_coefs_.push_back(std::move(q));
      fluid_coefs_.push_back(std::move(minus_q));
    }
  }
}

void LinearQuasiStaticMixedSelfGravitatingProblem::SetupRigidModes() {
  projector_u_ = MakeRigidModeProjector(*fes_);
  num_global_modes_ = projector_u_->Size();
}

void LinearQuasiStaticMixedSelfGravitatingProblem::AddRegionRotations(
    const Array<int>& solid_attributes) {
  Mesh* mesh = fes_->GetMesh();
  Array<int> attr_marker(mesh->attributes.Max());
  attr_marker = 0;
  for (int a : solid_attributes) {
    MFEM_VERIFY(a >= 1 && a <= attr_marker.Size(),
                "AddRegionRotations: SubMesh attribute out of range.");
    attr_marker[a - 1] = 1;
  }
  // Mask of the vdofs of the region's elements.
  Array<int> in_region(fes_->GetVSize());
  in_region = 0;
  Array<int> vdofs;
  for (int e = 0; e < mesh->GetNE(); e++) {
    if (attr_marker[mesh->GetAttribute(e) - 1]) {
      fes_->GetElementVDofs(e, vdofs);
      for (int v : vdofs) {
        in_region[FiniteElementSpace::DecodeDof(v)] = 1;
      }
    }
  }
  auto gf = detail::MakeGridFunction(fes_);
  int added = 0;
  auto add = [&](VectorCoefficient& c) {
    gf->ProjectCoefficient(c);
    for (int i = 0; i < in_region.Size(); i++) {
      if (!in_region[i]) {
        (*gf)[i] = 0.0;
      }
    }
    Vector t;
    gf->GetTrueDofs(t);
    if (projector_u_->Add(t)) {
      added++;
    }
  };
  if (dim_ == 2) {
    RigidRotation rot(2, 2);
    add(rot);
  } else {
    for (int c = 0; c < 3; c++) {
      RigidRotation rot(3, c);
      add(rot);
    }
  }
  MFEM_VERIFY(added > 0, "AddRegionRotations: no new mode (empty region, or "
                         "its rotations are already in the null space).");
  SetupCoupledNullSpace();
  MarkOperatorDirty();
}

void LinearQuasiStaticMixedSelfGravitatingProblem::SetupCoupledNullSpace() {
  // The block system is restricted to displacements orthogonal to the
  // rigid modes (and, in 2-D, potentials orthogonal to the constant): the
  // projector removes (u_r, 0) and (0, 1). On that subspace the operator is
  // symmetric and nonsingular (its Schur complement is P S P, positive for
  // a gravitationally stable body), and it is exactly the system the Schur
  // solver solves, so the two solvers agree to solver tolerance and no
  // gauge fixing is needed afterwards. (Projecting the near-null vectors
  // (u_r, -A_phiphi^{-1} C^T u_r) instead and re-gauging the displacement
  // afterwards costs a residual of the size of the rigid-mode residual
  // times the rigid content of the iterate, which soft physical modes such
  // as the translation of an inner core amplify.)
#ifdef MFEM_USE_MPI
  if (pfes_) {
    projector_block_ =
        std::make_unique<NullSpaceProjector>(pfes_->GetComm());
  } else
#endif
  {
    projector_block_ = std::make_unique<NullSpaceProjector>();
  }
  offsets_.SetSize(3);
  offsets_[0] = 0;
  offsets_[1] = fes_->GetTrueVSize();
  offsets_[2] = fes_phi_->GetTrueVSize();
  offsets_.PartialSum();

  BlockVector n(offsets_);
  for (int i = 0; i < projector_u_->Size(); i++) {
    n.GetBlock(0) = projector_u_->Basis(i);
    n.GetBlock(1) = 0.0;
    projector_block_->Add(n);
  }
  if (dim_ == 2) {
    n.GetBlock(0) = 0.0;
    n.GetBlock(1) = ones_;
    projector_block_->Add(n);
  }
}

// ---------------------------------------------------------------------------
// Loads

void LinearQuasiStaticMixedSelfGravitatingProblem::SetSurfaceLoad(
    Coefficient& sigma, const Array<int>& bdr_marker) {
  MFEM_VERIFY(bdr_marker.Size() == fes_->GetMesh()->bdr_attributes.Max(),
              "SetSurfaceLoad: the marker must be sized to the SubMesh's "
              "bdr_attributes.Max().");
  RegisterTimeDependent(sigma);
  load_markers_.push_back(bdr_marker);
  auto& marker = load_markers_.back();
  auto minus_sigma = std::make_unique<ProductCoefficient>(-1.0, sigma);
  auto minus_sigma_grad = std::make_unique<ScalarVectorProductCoefficient>(
      *minus_sigma, *grad_phi0_shadow_);
  b_->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(*minus_sigma_grad),
                            marker);
  b_phi_->AddBoundaryIntegrator(new BoundaryLFIntegrator(*minus_sigma),
                                marker);
  load_coefs_.push_back(std::move(minus_sigma));
  load_vcoefs_.push_back(std::move(minus_sigma_grad));
}

void LinearQuasiStaticMixedSelfGravitatingProblem::SetTidalPotential(
    Coefficient& psi) {
  MFEM_VERIFY(!rot_enabled_,
              "SetTidalPotential: incompatible with SetRotation (the "
              "centrifugal potential is handled by the border).");
  psi_ = &psi;
  RegisterTimeDependent(psi);
  psi_gf_ = detail::MakeGridFunction(fes_phi_);
  *psi_gf_ = 0.0;
}

real_t LinearQuasiStaticMixedSelfGravitatingProblem::TidalCoupling(
    Coefficient& psi) {
  EnsureOperator();
  auto psig = detail::MakeGridFunction(fes_phi_);
  psig->ProjectCoefficient(psi);
  Vector Psi(fes_phi_->GetTrueVSize());
  psig->GetTrueDofs(Psi);
  Vector cu(fes_->GetTrueVSize());
  C_op_->Mult(Psi, cu);
  Vector U(fes_->GetTrueVSize());
  Displacement().GetTrueDofs(U);
  real_t v = Dot(cu, U);
  if (!fluids_.empty()) {
    Vector mp(Psi.Size());
    M_fluid_.Ptr()->Mult(Psi, mp);
    Vector Phi(fes_phi_->GetTrueVSize());
    Potential().GetTrueDofs(Phi);
    v += Dot(mp, Phi);
  }
  return v;
}

real_t LinearQuasiStaticMixedSelfGravitatingProblem::TidalTidalCoupling(
    Coefficient& a, Coefficient& b) {
  if (fluids_.empty()) {
    return 0.0;
  }
  EnsureOperator();
  auto ga = detail::MakeGridFunction(fes_phi_);
  auto gb = detail::MakeGridFunction(fes_phi_);
  ga->ProjectCoefficient(a);
  gb->ProjectCoefficient(b);
  Vector Pa(fes_phi_->GetTrueVSize()), Pb(fes_phi_->GetTrueVSize());
  ga->GetTrueDofs(Pa);
  gb->GetTrueDofs(Pb);
  Vector mp(Pb.Size());
  M_fluid_.Ptr()->Mult(Pb, mp);
  return Dot(Pa, mp);
}

void LinearQuasiStaticMixedSelfGravitatingProblem::SetWaterLoad(
    Coefficient& ocean_weight, Coefficient& sigma_data,
    const Array<int>& surface_marker) {
  MFEM_VERIFY(!sea_enabled_, "SetWaterLoad: already set.");
  MFEM_VERIFY(dim_ == 3,
              "SetWaterLoad: the sea-level equation is 3-D only (a 2-D "
              "ocean is not supported, and the 2-D compatibility "
              "relocation would make the bordered system asymmetric).");
  MFEM_VERIFY(!gauge_kkt_,
              "SetWaterLoad: incompatible with the gauge-KKT saddle.");
  sea_enabled_ = true;
  sea_w_ = &ocean_weight;
  sea_sigma_data_ = &sigma_data;
  sea_marker_ = surface_marker;
  RegisterTimeDependent(sigma_data);

  // The uu water block: -int w (m.grad Phi0)^2 (m.u)(m.v) dS on the
  // stiffness form (the normal-projected tau-tau trace block; exact on
  // an equipotential surface).
  {
    auto q_g =
        std::make_unique<BoundaryNormalDotCoefficient>(*grad_phi0_shadow_);
    auto wq = std::make_unique<ProductCoefficient>(*sea_w_, *q_g);
    auto wqq = std::make_unique<ProductCoefficient>(*wq, *q_g);
    auto m_wqq = std::make_unique<ProductCoefficient>(-1.0, *wqq);
    StiffnessIntegrators().AddBoundaryIntegrator(
        new BoundaryNormalNormalIntegrator(*m_wqq), sea_marker_);
    sea_coefs_.push_back(std::move(q_g));
    sea_coefs_.push_back(std::move(wq));
    sea_coefs_.push_back(std::move(wqq));
    sea_coefs_.push_back(std::move(m_wqq));
  }

  // The u-phi block: rebuild the coupling with the water term (see
  // SetupCoupling); C^T follows by transposition.
  SetupCoupling();

  // The phi-phi block and the Phi_g border, shared with
  // RefreshWaterLoad().
  sea_phiphi_base_ = A_phiphi_;
  BuildWaterPhiPhi();
  BuildWaterBorder();

  // The data load goes through the ordinary surface-load slot; its
  // surface integral (the mass-row data) is refreshed per
  // AssembleForce.
  SetSurfaceLoad(sigma_data, surface_marker);
  BuildBorderCrossData();
  MarkOperatorDirty();
}

void LinearQuasiStaticMixedSelfGravitatingProblem::BuildWaterPhiPhi() {
  // -int w phi chi dS on the parent mesh's surface boundary elements
  // (their attributes are inherited, so the caller's marker lists the
  // right attributes; the parent's marker is just the same list at the
  // parent's size).
  Array<int> parent_marker(fes_phi_->GetMesh()->bdr_attributes.Max());
  parent_marker = 0;
  for (int a = 1; a <= sea_marker_.Size(); a++) {
    if (sea_marker_[a - 1]) {
      MFEM_VERIFY(a <= parent_marker.Size(),
                  "SetWaterLoad: surface attribute missing on the "
                  "parent mesh.");
      parent_marker[a - 1] = 1;
    }
  }
  auto minus_w = std::make_unique<ProductCoefficient>(-1.0, *sea_w_);
  sea_phiphi_form_ = detail::MakeBilinearForm(fes_phi_);
  sea_phiphi_form_->AddBoundaryIntegrator(new MassIntegrator(*minus_w),
                                          parent_marker);
  sea_phiphi_form_->Assemble();
  Array<int> empty;
  sea_phiphi_form_->FormSystemMatrix(empty, S_sea_);
  sea_coefs_.push_back(std::move(minus_w));
  A_sea_sum_ = std::make_unique<SumOperator>(
      sea_phiphi_base_, 1.0, S_sea_.Ptr(), 1.0,
      false, false);
  A_phiphi_ = A_sea_sum_.get();
  SetupPotentialSolver();
}

void LinearQuasiStaticMixedSelfGravitatingProblem::BuildWaterBorder() {
  // The Phi_g border c = int w tau' dS: the surface-mass column of
  // density w, and m = int w dS.
  BuildSurfaceMassColumn(*sea_w_, sea_cu_, sea_cphi_);
  sea_cphi_compat_ = sea_cphi_;
  MakeCompatible(sea_cphi_compat_);
  ConstantCoefficient one_c(1.0);
  ProductCoefficient w_one(*sea_w_, one_c);
  sea_m_ = SurfaceIntegralOnMarker(w_one);
  MFEM_VERIFY(sea_m_ > 0.0, "SetWaterLoad: the ocean has zero area.");
}

void LinearQuasiStaticMixedSelfGravitatingProblem::RefreshWaterLoad() {
  MFEM_VERIFY(sea_enabled_, "RefreshWaterLoad: SetWaterLoad first.");
  // The coupling block holds the mutated coefficient through its
  // integrator; rebuilding re-evaluates it. The stiffness's uu block
  // re-evaluates at the next operator assembly.
  SetupCoupling();
  BuildWaterPhiPhi();
  BuildWaterBorder();
  BuildBorderCrossData();
  MarkOperatorDirty();
}

void LinearQuasiStaticMixedSelfGravitatingProblem::BuildSurfaceMassColumn(
    Coefficient& sdens, Vector& cu, Vector& cphi) {
  // The tau'-pairing of a surface mass density s: the u row
  // +int s (m.grad Phi0)(m.v) dS — the normal-projected form, matching
  // the water Q-blocks' route — and the phi row +int s chi dS
  // (assembled like the loads: on the shadow space, injected).
  auto q_g = std::make_unique<BoundaryNormalDotCoefficient>(
      *grad_phi0_shadow_);
  ProductCoefficient sq(sdens, *q_g);
  auto lu = detail::MakeLinearForm(fes_);
  lu->AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(sq),
                            sea_marker_);
  lu->Assemble();
  ToTrueDofs(*fes_, *lu, cu);
  auto lphi = detail::MakeLinearForm(shadow_phi_.get());
  lphi->AddBoundaryIntegrator(new BoundaryLFIntegrator(sdens),
                              sea_marker_);
  lphi->Assemble();
  Vector bL(fes_phi_->GetVSize());
  injection_->Mult(*lphi, bL);
  ToTrueDofs(*fes_phi_, bL, cphi);
}

real_t LinearQuasiStaticMixedSelfGravitatingProblem::SurfaceIntegralOnMarker(
    Coefficient& f) {
  auto lphi = detail::MakeLinearForm(shadow_phi_.get());
  lphi->AddBoundaryIntegrator(new BoundaryLFIntegrator(f), sea_marker_);
  lphi->Assemble();
  real_t v = lphi->Sum();
#ifdef MFEM_USE_MPI
  if (pfes_phi_) {
    real_t g = 0.0;
    MPI_Allreduce(&v, &g, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                  pfes_phi_->GetComm());
    v = g;
  }
#endif
  return v;
}

void LinearQuasiStaticMixedSelfGravitatingProblem::SetRotation(
    real_t Omega, const Vector& principal_moments) {
  MFEM_VERIFY(!rot_enabled_, "SetRotation: already set.");
  MFEM_VERIFY(!gauge_kkt_,
              "SetRotation: incompatible with the gauge-KKT saddle.");
  MFEM_VERIFY(!psi_,
              "SetRotation: incompatible with SetTidalPotential (the "
              "centrifugal potential is handled by the border).");
  rot_enabled_ = true;
  rot_Omega_ = Omega;
  const int nw = dim_ == 2 ? 1 : 3;
  MFEM_VERIFY(principal_moments.Size() == nw,
              "SetRotation: " << nw << " principal moment(s) expected.");
  rot_D_ = InertiaMatrix(principal_moments);
  for (int k = 0; k < nw; k++) {
    auto psi = std::make_unique<CentrifugalPotential>(dim_, Omega);
    psi->SetUnit(k);
    rot_psi_.push_back(std::move(psi));
  }
  omega_.SetSize(nw);
  omega_ = 0.0;
  rot_r_.SetSize(nw);
  rot_r_ = 0.0;
  {
    auto total = std::make_unique<CentrifugalPotential>(dim_, Omega);
    total->SetZero();
    rot_psi_total_ = std::move(total);
  }
  BuildBorderCrossData();
  MarkOperatorDirty();
}

Coefficient&
LinearQuasiStaticMixedSelfGravitatingProblem::SolutionCentrifugalPotential() {
  MFEM_VERIFY(rot_enabled_,
              "SolutionCentrifugalPotential: SetRotation first.");
  return *rot_psi_total_;
}

void LinearQuasiStaticMixedSelfGravitatingProblem::BuildBorderCrossData() {
  // The water-rotation cross data, when both features are on: the
  // columns c^psi_k = int w tau' psi_k dS, the couplings
  // m^psi_k = int w psi_k dS and Pw_kj = int w psi_k psi_j dS.
  rot_cache_version_ = -1;
  if (!sea_enabled_ || !rot_enabled_) {
    return;
  }
  const int nw = static_cast<int>(rot_psi_.size());
  rot_cpsi_u_.resize(nw);
  rot_cpsi_phi_.resize(nw);
  rot_mpsi_.SetSize(nw);
  rot_Pw_.SetSize(nw);
  // psi through its interpolant (on the shadow space, whose surface
  // nodes are the potential space's): the convention of the tidal
  // columns, kept everywhere so the border stays symmetric.
  std::vector<std::unique_ptr<GridFunction>> psig(nw);
  std::vector<std::unique_ptr<GridFunctionCoefficient>> psii(nw);
  for (int k = 0; k < nw; k++) {
    psig[k] = detail::MakeGridFunction(shadow_phi_.get());
    psig[k]->ProjectCoefficient(*rot_psi_[k]);
    psii[k] = std::make_unique<GridFunctionCoefficient>(psig[k].get());
  }
  for (int k = 0; k < nw; k++) {
    ProductCoefficient wpsi(*sea_w_, *psii[k]);
    BuildSurfaceMassColumn(wpsi, rot_cpsi_u_[k], rot_cpsi_phi_[k]);
    rot_mpsi_[k] = SurfaceIntegralOnMarker(wpsi);
    for (int j = 0; j <= k; j++) {
      ProductCoefficient wpp(wpsi, *psii[j]);
      rot_Pw_(k, j) = rot_Pw_(j, k) = SurfaceIntegralOnMarker(wpp);
    }
  }
}

void LinearQuasiStaticMixedSelfGravitatingProblem::AssembleTidalLoad() {
  // Psi = interpolant of psi; loads -C Psi (displacement) and -M_F Psi
  // (potential), the latter absent without fluid regions.
  tidal_u_.SetSize(fes_->GetTrueVSize());
  if (!psi_) {
    tidal_u_ = 0.0;
    return;
  }
  psi_gf_->ProjectCoefficient(*psi_);
  psi_gf_->GetTrueDofs(Psi_true_);
  C_op_->Mult(Psi_true_, tidal_u_);
  if (!fluids_.empty()) {
    Vector t(B_phi_.Size());
    M_fluid_.Ptr()->Mult(Psi_true_, t);
    B_phi_ -= t;
  }
}

void LinearQuasiStaticMixedSelfGravitatingProblem::AssembleForce(real_t t) {
  LinearQuasiStaticProblemBase::AssembleForce(t);
  border_r0_add_ = 0.0;
  if (border_rot_add_.Size() > 0) {
    border_rot_add_ = 0.0;
  }
  b_phi_->Assemble();
  Vector bL(fes_phi_->GetVSize());
  injection_->Mult(*b_phi_, bL);
  ToTrueDofs(*fes_phi_, bL, B_phi_);
  if (rot_enabled_) {
    // The omega-row data -int sigma psi_k dS, read off the raw
    // surface-load form (B_phi_ holds -int sigma chi for every
    // registered surface load), before the tidal and compatibility
    // modifications below.
    auto psig = detail::MakeGridFunction(fes_phi_);
    Vector Psi(fes_phi_->GetTrueVSize());
    for (int k = 0; k < static_cast<int>(rot_psi_.size()); k++) {
      psig->ProjectCoefficient(*rot_psi_[k]);
      psig->GetTrueDofs(Psi);
      rot_r_[k] = Dot(B_phi_, Psi);
    }
  }
  AssembleTidalLoad();
  MakeCompatible(B_phi_);
  if (sea_enabled_) {
    // The mass-row data int sigma_data dS at this time.
    if (!sea_mass_form_) {
      sea_mass_form_ = detail::MakeLinearForm(shadow_phi_.get());
      sea_mass_form_->AddBoundaryIntegrator(
          new BoundaryLFIntegrator(*sea_sigma_data_), sea_marker_);
    }
    sea_mass_form_->Assemble();
    real_t sd = sea_mass_form_->Sum();
#ifdef MFEM_USE_MPI
    if (pfes_phi_) {
      real_t g = 0.0;
      MPI_Allreduce(&sd, &g, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                    pfes_phi_->GetComm());
      sd = g;
    }
#endif
    sea_Sd_ = sd;
  }
}

// ---------------------------------------------------------------------------
// Controls

void LinearQuasiStaticMixedSelfGravitatingProblem::SetMassWeightedGauge(bool on) {
  if (on && !gauge_M_.Ptr()) {
    gauge_form_ = AssembleMassOperator(rho_, gauge_M_);
  }
  if (on != mass_gauge_) {
    mass_gauge_ = on;
    MarkOperatorDirty();  // the solvers are rebuilt with the gauge
  }
}

void LinearQuasiStaticMixedSelfGravitatingProblem::SetSolverType(SolverType type) {
  if (type != type_) {
    type_ = type;
    MarkOperatorDirty();
  }
}

void LinearQuasiStaticMixedSelfGravitatingProblem::SetInnerRelTol(real_t tol) {
  // CG stops on (B r, r) <= tol^2 (B r0, r0); below tol ~ 1e-13 that is
  // round-off, where the projected iteration loses orthogonality and
  // diverges rather than stagnates.
  inner_rel_tol_ = std::max<real_t>(kMinInnerRelTol, tol);
  inner_tol_set_ = true;
  cg_phi_->SetRelTol(inner_rel_tol_);
}

void LinearQuasiStaticMixedSelfGravitatingProblem::SetPreconditionerShift(
    real_t eps) {
  shift_ = eps;
  SetupPotentialPreconditioner();
  MarkOperatorDirty();
}

void LinearQuasiStaticMixedSelfGravitatingProblem::SetInnerPrintLevel(
    IterativeSolver::PrintLevel level) {
  inner_print_level_ = level;
  cg_phi_->SetPrintLevel(level);
}

void LinearQuasiStaticMixedSelfGravitatingProblem::SetBackgroundPotential(
    Coefficient& phi0) {
  ComputeBackgroundPotential(&phi0);
  if (!fluids_.empty()) {
    // The fluid mass term may depend on grad Phi0. SetupFluidMass
    // resets A_phiphi_ to the plain assembled block, so the water
    // machinery (which sums its boundary mass onto A_phiphi_ and
    // caches border and cross data built from grad Phi0) must be
    // rebuilt afterwards.
    SetupFluidMass();
  }
  if (sea_enabled_) {
    RefreshWaterLoad();
  }
  MarkOperatorDirty();
}

void LinearQuasiStaticMixedSelfGravitatingProblem::RegisterFields(
    DataCollection& dc) {
  LinearQuasiStaticProblemBase::RegisterFields(dc);
  dc.RegisterField("potential", phi_shadow_.get());
  dc.RegisterField("background_potential", phi0_shadow_.get());
}

// ---------------------------------------------------------------------------
// Diagnostics

real_t LinearQuasiStaticMixedSelfGravitatingProblem::ModeResidual(const Vector& u) {
  EnsureOperator();
  const Operator& A_uu = *A_.Ptr();
  real_t a_max = 0.0;
#ifdef MFEM_USE_MPI
  if (pfes_) {
    auto* hyp = A_.As<HypreParMatrix>();
    SparseMatrix diag, offd;
    HYPRE_BigInt* cmap = nullptr;
    hyp->GetDiag(diag);
    hyp->GetOffd(offd, cmap);
    real_t local = std::max(diag.MaxNorm(), offd.MaxNorm());
    MPI_Allreduce(&local, &a_max, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                  pfes_->GetComm());
  } else
#endif
  {
    a_max = A_.As<SparseMatrix>()->MaxNorm();
  }
  const int saved_inner = inner_its_;
  Vector t(Ct_op_->Height()), w, y(A_uu.Height()), cw(A_uu.Height());
  Ct_op_->Mult(u, t);
  SolvePotential(t, w);
  A_uu.Mult(u, y);
  C_op_->Mult(w, cw);
  y -= cw;
  inner_its_ = saved_inner;
  return std::sqrt(Dot(y, y)) / (a_max * std::sqrt(Dot(u, u)));
}

std::vector<real_t>
LinearQuasiStaticMixedSelfGravitatingProblem::RigidModeResiduals() {
  std::vector<real_t> residuals;
  for (int i = 0; i < projector_u_->Size(); i++) {
    residuals.push_back(ModeResidual(projector_u_->Basis(i)));
  }
  return residuals;
}

namespace {

// Extreme eigenvalues of the symmetric tridiagonal (alpha, beta) by Sturm
// bisection.
int SturmCount(const std::vector<real_t>& alpha,
               const std::vector<real_t>& beta, real_t x) {
  // Number of eigenvalues < x.
  int count = 0;
  real_t q = alpha[0] - x;
  if (q < 0.0) count++;
  for (size_t i = 1; i < alpha.size(); i++) {
    const real_t b2 = beta[i - 1] * beta[i - 1];
    q = alpha[i] - x - (q != 0.0 ? b2 / q : b2 / 1e-300);
    if (q < 0.0) count++;
  }
  return count;
}

void TridiagonalExtremes(const std::vector<real_t>& alpha,
                         const std::vector<real_t>& beta, real_t& lo,
                         real_t& hi) {
  // Gershgorin bounds, then bisection for the first and the last eigenvalue.
  const int n = static_cast<int>(alpha.size());
  real_t a = alpha[0], b = alpha[0];
  for (int i = 0; i < n; i++) {
    real_t r = (i > 0 ? std::abs(beta[i - 1]) : 0.0) +
               (i < n - 1 ? std::abs(beta[i]) : 0.0);
    a = std::min(a, alpha[i] - r);
    b = std::max(b, alpha[i] + r);
  }
  auto bisect = [&](int k) {  // k-th smallest (0-based)
    real_t l = a, u = b;
    for (int it = 0; it < 200 && (u - l) > 1e-14 * (std::abs(l) + std::abs(u));
         it++) {
      const real_t m = 0.5 * (l + u);
      if (SturmCount(alpha, beta, m) > k) {
        u = m;
      } else {
        l = m;
      }
    }
    return 0.5 * (l + u);
  };
  lo = bisect(0);
  hi = bisect(n - 1);
}

}  // namespace

real_t LinearQuasiStaticMixedSelfGravitatingProblem::PotentialBlockMinEigenvalue(
    int lanczos_steps, real_t* largest) {
  // Lanczos on A_phiphi with full reorthogonalisation, Euclidean inner
  // product on true dofs, deterministic start.
  const int n = fes_phi_->GetTrueVSize();
  std::vector<Vector> V;
  std::vector<real_t> alpha, beta;
  Vector v(n), w(n);
  // In 2-D the constant is projected out, as in every potential solve.
  auto deflate = [&](Vector& x) {
    if (dim_ == 2) {
      x.Add(-Dot(x, ones_) / Dot(ones_, ones_), ones_);
    }
  };
  {
    std::mt19937 gen(12345 + n);
    std::uniform_real_distribution<real_t> dist(-1.0, 1.0);
    for (int i = 0; i < n; i++) {
      v[i] = dist(gen);
    }
    deflate(v);
    v /= std::sqrt(Dot(v, v));
  }
  V.push_back(v);
  for (int k = 0; k < lanczos_steps; k++) {
    A_phiphi_->Mult(V[k], w);
    deflate(w);
    alpha.push_back(Dot(w, V[k]));
    for (int pass = 0; pass < 2; pass++) {  // full reorthogonalisation
      for (const auto& vj : V) {
        w.Add(-Dot(w, vj), vj);
      }
    }
    const real_t b = std::sqrt(Dot(w, w));
    if (b <= 1e-14 * std::abs(alpha[0]) || k == lanczos_steps - 1) {
      break;
    }
    beta.push_back(b);
    w /= b;
    V.push_back(w);
  }
  real_t lo, hi;
  TridiagonalExtremes(alpha, beta, lo, hi);
  if (largest) {
    *largest = hi;
  }
  return lo;
}

// ---------------------------------------------------------------------------
// Solvers

LinearQuasiStaticMixedSelfGravitatingProblem::SchurOperator::SchurOperator(
    const LinearQuasiStaticMixedSelfGravitatingProblem& p, const Operator& A_uu)
    : Operator(A_uu.Height(), A_uu.Width()), p_(&p), A_uu_(&A_uu) {}

void LinearQuasiStaticMixedSelfGravitatingProblem::SchurOperator::Mult(
    const Vector& x, Vector& y) const {
  t_.SetSize(p_->Ct_op_->Height());
  p_->Ct_op_->Mult(x, t_);
  p_->SolvePotential(t_, w_);
  A_uu_->Mult(x, y);
  cw_.SetSize(y.Size());
  p_->C_op_->Mult(w_, cw_);
  y -= cw_;
}

void LinearQuasiStaticMixedSelfGravitatingProblem::SetupSolver(OperatorHandle& A) {
  if (!inner_tol_set_) {
    inner_rel_tol_ = std::max<real_t>(kMinInnerRelTol, 1e-2 * rel_tol_);
    cg_phi_->SetRelTol(inner_rel_tol_);
  }
  // Preconditioner (prec_) and CG (cg_) for the displacement block.
  SetupDefaultCG(A);
  if (type_ == SolverType::SchurCG) {
    SetupSchur(A);
  } else {
    SetupMinres(A);
  }
  if (gauge_kkt_) {
    MFEM_VERIFY(!GaugePreconditionerOnly(),
                "SetGaugePreconditionerOnly: incompatible with the "
                "gauge-KKT saddle.");
    SetupKKTGauge();
  }
}

void LinearQuasiStaticMixedSelfGravitatingProblem::SetupSchur(OperatorHandle& A) {
  MFEM_VERIFY(!GaugePreconditionerOnly(),
              "SetGaugePreconditionerOnly: block-MINRES (or BlockCG) only; "
              "the Schur path eliminates through the regularised block.");
  schur_ = std::make_unique<SchurOperator>(*this, *A.Ptr());
  projected_op_ = std::make_unique<ProjectedOperator>(*schur_, *projector_u_);
  projected_prec_ = std::make_unique<ProjectedSolver>(*projector_u_);
  projected_prec_->SetSolver(*prec_);
  SetupCG(*projected_op_, *projected_prec_);

  projected_ = std::make_unique<ProjectedSolver>(*projector_u_);
  projected_->SetSolver(*cg_);
  projected_->iterative_mode = true;
  if (mass_gauge_) {
    projected_->SetGauge(gauge_M_.Ptr());
  }
}

void LinearQuasiStaticMixedSelfGravitatingProblem::SetupMinres(OperatorHandle& A) {
  block_op_ = std::make_unique<BlockOperator>(offsets_);
  // The solid-everywhere preconditioner mode: the Krylov operator
  // keeps the CLEAN displacement block while prec_ (built by
  // SetupDefaultCG on the handle passed in) sits on A + eps_prec Q;
  // MINRES ignores the exact gauge kernel, the near-gauge modes move
  // into the stopping rule (SetGaugePreconditionerOnly).
  block_op_->SetBlock(0, 0,
                      GaugePreconditionerOnly() ? A_.Ptr() : A.Ptr());
  block_op_->SetBlock(0, 1, const_cast<Operator*>(C_op_));
  block_op_->SetBlock(1, 0, const_cast<Operator*>(Ct_op_));
  block_op_->SetBlock(1, 1, A_phiphi_);

  block_prec_ = std::make_unique<BlockDiagonalPreconditioner>(offsets_);
  block_prec_->SetDiagonalBlock(0, prec_.get());
  block_prec_->SetDiagonalBlock(1, prec_phi_.get());

#ifdef MFEM_USE_MPI
  if (pfes_) {
    if (type_ == SolverType::BlockCG) {
      minres_ = std::make_unique<CGSolver>(pfes_->GetComm());
    } else {
      minres_ = std::make_unique<MINRESSolver>(pfes_->GetComm());
    }
  } else
#endif
  {
    if (type_ == SolverType::BlockCG) {
      minres_ = std::make_unique<CGSolver>();
    } else {
      minres_ = std::make_unique<MINRESSolver>();
    }
  }
  projected_op_ =
      std::make_unique<ProjectedOperator>(*block_op_, *projector_block_);
  minres_->SetOperator(*projected_op_);
  // P M P as for the inner solves (the potential block's preconditioner
  // amplifies the constant in 2-D, the displacement's the rigid modes).
  projected_prec_ = std::make_unique<ProjectedSolver>(*projector_block_);
  projected_prec_->SetSolver(*block_prec_);
  minres_->SetPreconditioner(*projected_prec_);
  minres_->SetRelTol(rel_tol_);
  minres_->SetAbsTol(0.0);
  minres_->SetMaxIter(10000);
  minres_->SetPrintLevel(print_level_);
  minres_->iterative_mode = true;

  projected_ = std::make_unique<ProjectedSolver>(*projector_block_);
  projected_->SetSolver(*minres_);
  projected_->iterative_mode = true;
  if (mass_gauge_) {
    // The gauge weights the u block only; the phi part of a null vector
    // follows its u part.
    gauge_block_ = std::make_unique<BlockOperator>(offsets_);
    gauge_block_->SetBlock(0, 0, gauge_M_.Ptr());
    projected_->SetGauge(gauge_block_.get());
  }

  // The MINRES iterate is the warm-start state; keep it across operator
  // rebuilds (a change of shear modulus) when the sizes are unchanged.
  if (!X_block_ || X_block_->Size() != offsets_.Last()) {
    X_block_ = std::make_unique<BlockVector>(offsets_);
    *X_block_ = 0.0;
  }
  B_block_ = std::make_unique<BlockVector>(offsets_);
}

void LinearQuasiStaticMixedSelfGravitatingProblem::ConfigureFluidOperator(
    const Array<int>& marker, Coefficient& mu, real_t epsilon,
    GaugePenalty form, Diffeomorphism* map) {
  MFEM_VERIFY(fluids_.empty(),
              "SetFluid: the gauged formulation carries the fluid "
              "inside the displacement SubMesh; construct the problem "
              "without FluidRegions.");
  LinearQuasiStaticProblemBase::ConfigureFluidOperator(marker, mu, epsilon,
                                                       form, map);
}

namespace {

// y = P (M (P x)): with P symmetric positive and M SPD this is an SPD
// preconditioner. Used as the Murphy-Golub-Wathen multiplier-slot
// approximation (A Qt^{-1} A)^{-1} ~ P(A) * Qt * P(A) of the gauge-KKT
// saddle, with Qt the (A + Q) proxy of the regularised primal block.
class SymmetricProductSolver : public Solver {
 public:
  SymmetricProductSolver(const Solver& P, const Operator& M)
      : Solver(M.Height()), P_(&P), M_(&M), t1_(M.Height()),
        t2_(M.Height()) {}
  void SetOperator(const Operator&) override {}
  void Mult(const Vector& x, Vector& y) const override {
    P_->Mult(x, t1_);
    M_->Mult(t1_, t2_);
    P_->Mult(t2_, y);
  }

 private:
  const Solver* P_;
  const Operator* M_;
  mutable Vector t1_, t2_;
};

}  // namespace

void LinearQuasiStaticMixedSelfGravitatingProblem::EnableGaugeKKT(
    const Array<int>& fluid_marker, Coefficient& mu_gauge, bool mgw_prec) {
  MFEM_VERIFY(type_ != SolverType::SchurCG,
              "EnableGaugeKKT: block organisation only.");
  gauge_kkt_ = true;
  kkt_mgw_ = mgw_prec;
  // Unit-scale penalty: the base then assembles Q_ itself and
  // A_solve_ = A + Q, whose preconditioner the KKT diagonal slots
  // reuse; zero refinements (the KKT solve needs none). The gauge
  // form is assembled WITHOUT an equilibrium mapping by design: the
  // gauge-KKT path is an unmapped instrument of this class, and no
  // mapped configuration reaches it.
  GaugePenaltyOptions o;
  o.epsilon = 1.0;
  o.refinements = 0;
  SetFluid(fluid_marker, mu_gauge, o);
}

void LinearQuasiStaticMixedSelfGravitatingProblem::SetupKKTGauge() {
  const int nu = offsets_[1] - offsets_[0];
  const int nphi = offsets_[2] - offsets_[1];
  offsets_kkt4_.SetSize(5);
  offsets_kkt4_[0] = 0;
  offsets_kkt4_[1] = nu;
  offsets_kkt4_[2] = nphi;
  offsets_kkt4_[3] = nu;
  offsets_kkt4_[4] = nphi;
  offsets_kkt4_.PartialSum();

  // [ Q 0 | A ] with the CLEAN physical blocks (A_, not the penalised
  // [ 0 0 |   ]  A_solve_) in the constraint rows and columns.
  // [  A  | 0 ]
  block_op_kkt4_ = std::make_unique<BlockOperator>(offsets_kkt4_);
  Operator* Auu = A_.Ptr();
  block_op_kkt4_->SetBlock(0, 0, GaugeQ().Ptr());
  block_op_kkt4_->SetBlock(0, 2, Auu);
  block_op_kkt4_->SetBlock(0, 3, const_cast<Operator*>(C_op_));
  block_op_kkt4_->SetBlock(1, 2, const_cast<Operator*>(Ct_op_));
  block_op_kkt4_->SetBlock(1, 3, A_phiphi_);
  block_op_kkt4_->SetBlock(2, 0, Auu);
  block_op_kkt4_->SetBlock(2, 1, const_cast<Operator*>(C_op_));
  block_op_kkt4_->SetBlock(3, 0, const_cast<Operator*>(Ct_op_));
  block_op_kkt4_->SetBlock(3, 1, A_phiphi_);

  // Null pairs: each coupled-system pair appears in the primal slots
  // and, independently, in the multiplier slots (v shares A's kernel).
#ifdef MFEM_USE_MPI
  if (pfes_) {
    projector_kkt4_ = std::make_unique<NullSpaceProjector>(pfes_->GetComm());
  } else
#endif
  {
    projector_kkt4_ = std::make_unique<NullSpaceProjector>();
  }
  {
    const int n2 = offsets_[2];
    BlockVector nb(offsets_kkt4_);
    for (int i = 0; i < projector_block_->Size(); i++) {
      const Vector& b = projector_block_->Basis(i);
      nb = 0.0;
      for (int j = 0; j < n2; j++) {
        nb(j) = b(j);
      }
      projector_kkt4_->Add(nb);
      nb = 0.0;
      for (int j = 0; j < n2; j++) {
        nb(n2 + j) = b(j);
      }
      projector_kkt4_->Add(nb);
    }
  }

  // Clean-A preconditioner for the MGW composite of the v_u slot:
  // (A Qt^{-1} A)^{-1} = A^{-1} Qt A^{-1} ~ P(A) (A+Q) P(A), with the
  // (A+Q) matrix available as the solve operator of the unit-scale
  // gauged assembly.
#ifdef MFEM_USE_MPI
  if (pfes_) {
    auto amg = std::make_unique<HypreBoomerAMG>(*A_.As<HypreParMatrix>());
    // order_bynodes = true: the displacement space is Ordering::byNODES.
    amg->SetSystemsOptions(fes_->GetVDim(), true);
    amg->SetPrintLevel(0);
    prec_clean_u_ = std::move(amg);
  } else
#endif
  {
    prec_clean_u_ = std::make_unique<GSSmoother>(*A_.As<SparseMatrix>());
  }
  prec_vu_ =
      std::make_unique<SymmetricProductSolver>(*prec_clean_u_, *A_solve_.Ptr());

  block_prec_kkt4_ =
      std::make_unique<BlockDiagonalPreconditioner>(offsets_kkt4_);
  block_prec_kkt4_->SetDiagonalBlock(0, prec_.get());
  block_prec_kkt4_->SetDiagonalBlock(1, prec_phi_.get());
  // The MGW composite needs component inverses of AMG grade; with
  // smoother-grade P it DEGRADES the solve (measured serially) and is
  // therefore opt-in.
  block_prec_kkt4_->SetDiagonalBlock(2,
                                     kkt_mgw_ ? prec_vu_.get() : prec_.get());
  block_prec_kkt4_->SetDiagonalBlock(3, prec_phi_.get());

#ifdef MFEM_USE_MPI
  if (pfes_) {
    minres_kkt4_ = std::make_unique<MINRESSolver>(pfes_->GetComm());
  } else
#endif
  {
    minres_kkt4_ = std::make_unique<MINRESSolver>();
  }
  projected_op_kkt4_ =
      std::make_unique<ProjectedOperator>(*block_op_kkt4_, *projector_kkt4_);
  minres_kkt4_->SetOperator(*projected_op_kkt4_);
  projected_prec_kkt4_ = std::make_unique<ProjectedSolver>(*projector_kkt4_);
  projected_prec_kkt4_->SetSolver(*block_prec_kkt4_);
  minres_kkt4_->SetPreconditioner(*projected_prec_kkt4_);
  minres_kkt4_->SetRelTol(rel_tol_);
  minres_kkt4_->SetAbsTol(0.0);
  minres_kkt4_->SetMaxIter(30000);
  minres_kkt4_->SetPrintLevel(print_level_);
  minres_kkt4_->iterative_mode = true;

  projected_kkt4_ = std::make_unique<ProjectedSolver>(*projector_kkt4_);
  projected_kkt4_->SetSolver(*minres_kkt4_);
  projected_kkt4_->iterative_mode = true;

  if (!Xk4_ || Xk4_->Size() != offsets_kkt4_.Last()) {
    Xk4_ = std::make_unique<BlockVector>(offsets_kkt4_);
    *Xk4_ = 0.0;
  }
  Bk4_ = std::make_unique<BlockVector>(offsets_kkt4_);
}

std::pair<real_t, real_t>
LinearQuasiStaticMixedSelfGravitatingProblem::KKTResiduals() const {
  MFEM_VERIFY(gauge_kkt_ && Xk4_ && Bk4_, "KKTResiduals: no KKT solve yet.");
  BlockVector r(offsets_kkt4_);
  block_op_kkt4_->Mult(*Xk4_, r);
  r -= *Bk4_;
  real_t prim = 0.0, prim_f = 0.0, stat = 0.0, qun = 0.0;
  for (int b : {2, 3}) {
    prim += r.GetBlock(b) * r.GetBlock(b);
    prim_f += Bk4_->GetBlock(b) * Bk4_->GetBlock(b);
  }
  for (int b : {0, 1}) {
    stat += r.GetBlock(b) * r.GetBlock(b);
  }
  Vector qu(offsets_kkt4_[1]);
  GaugeQ().Ptr()->Mult(Xk4_->GetBlock(0), qu);
  qun = qu * qu;
  return {std::sqrt(prim / std::max(prim_f, real_t{1e-300})),
          std::sqrt(stat / std::max(qun, real_t{1e-300}))};
}


bool LinearQuasiStaticMixedSelfGravitatingProblem::SolveLinearSystem(
    const Vector& B_in, Vector& X) {
  if (!sea_enabled_ && !rot_enabled_ && !level_enabled_) {
    return SolveUnbordered(B_in, X);
  }
  // The bordered solve:
  // with y = (Phi_g[, omega]) the system is
  //   [K  B; B^T  Db] [x; y] = [b; r],
  // K the (water-modified) operator, the border columns
  //   B = [c_Phi, (L_k - c^psi_k)...]
  // (c_Phi the Phi_g column, L_k the centrifugal coupling = the tidal
  // operators applied to the unit potential psi_k, c^psi_k the water's
  // psi column), and the dense block
  //   Db = [[-m, m^psi], [m^psi, D + P - Pw]].
  // Block elimination: K Y_j = B_j (psi_ masked, B_phi_ swapped per
  // column), then (Db - B^T Y) y = r - B^T x0 and x = x0 - Y y.
  const int ns = sea_enabled_ ? 1 : 0;
  const int nw = rot_enabled_ ? static_cast<int>(rot_psi_.size()) : 0;
  const int nb0 = ns + nw;
  int nl = 0;
  for (const auto& L : level_) {
    nl += L.size;
  }
  const int nb = nb0 + nl;

  // The border columns.
  std::vector<Vector> Bu(nb), Bphi(nb);
  if (sea_enabled_) {
    Bu[0] = sea_cu_;
    Bphi[0] = sea_cphi_;
  }
  if (rot_enabled_) {
    EnsureOperator();
    // The rotation columns and the psi-psi fluid block do not depend on the
    // load: rebuilt only when the operator or the water data change.
    if (rot_cache_version_ != OperatorVersion()) {
      rot_Bu_.assign(nw, Vector());
      rot_Bphi_.assign(nw, Vector());
      rot_TT_.SetSize(nw);
      rot_TT_ = 0.0;
      auto psig = detail::MakeGridFunction(fes_phi_);
      std::vector<Vector> Psi(nw, Vector(fes_phi_->GetTrueVSize()));
      for (int k = 0; k < nw; k++) {
        psig->ProjectCoefficient(*rot_psi_[k]);
        psig->GetTrueDofs(Psi[k]);
        rot_Bu_[k].SetSize(fes_->GetTrueVSize());
        C_op_->Mult(Psi[k], rot_Bu_[k]);
        rot_Bphi_[k].SetSize(fes_phi_->GetTrueVSize());
        rot_Bphi_[k] = 0.0;
        if (!fluids_.empty()) {
          M_fluid_.Ptr()->Mult(Psi[k], rot_Bphi_[k]);
        }
      }
      if (!fluids_.empty()) {
        // psi_k . M_F psi_j is symmetric: each pair evaluated once.
        for (int k = 0; k < nw; k++) {
          for (int j = k; j < nw; j++) {
            rot_TT_(k, j) = rot_TT_(j, k) = Dot(Psi[k], rot_Bphi_[j]);
          }
        }
      }
      if (sea_enabled_) {
        for (int k = 0; k < nw; k++) {
          rot_Bu_[k] -= rot_cpsi_u_[k];
          rot_Bphi_[k] -= rot_cpsi_phi_[k];
        }
      }
      rot_cache_version_ = OperatorVersion();
    }
    for (int k = 0; k < nw; k++) {
      Bu[ns + k] = rot_Bu_[k];
      Bphi[ns + k] = rot_Bphi_[k];
    }
  }

  // The dense block and the border rhs.
  DenseMatrix Db(nb);
  Db = 0.0;
  Vector r(nb);
  r = 0.0;
  if (sea_enabled_) {
    Db(0, 0) = -sea_m_;
    r[0] = sea_Sd_ + border_r0_add_;
  }
  if (rot_enabled_) {
    for (int k = 0; k < nw; k++) {
      r[ns + k] = rot_r_[k];
      if (border_rot_add_.Size() == nw) {
        r[ns + k] += border_rot_add_[k];
      }
      for (int j = 0; j < nw; j++) {
        Db(ns + k, ns + j) = rot_D_(k, j) + rot_TT_(k, j);
        if (sea_enabled_) {
          Db(ns + k, ns + j) -= rot_Pw_(k, j);
        }
      }
      if (sea_enabled_) {
        Db(0, ns + k) = Db(ns + k, 0) = rot_mpsi_[k];
      }
    }
  }

  // The level borders (SetLevelBorder): the nu unknowns couple to the
  // potential, the wall multipliers to the solid, the rest only within
  // the 1-D block; their columns are zero and the engine skips them.
  if (level_enabled_) {
    int off = nb0;
    for (const auto& L : level_) {
      for (int j = 0; j < L.size; j++) {
        Bu[off + j].SetSize(fes_->GetTrueVSize());
        Bu[off + j] = 0.0;
        Bphi[off + j].SetSize(fes_phi_->GetTrueVSize());
        Bphi[off + j] = 0.0;
      }
      for (int k = 0; k <= L.n; k++) {
        Bphi[off + L.onu + k] = L.nu_phi[k];
      }
      for (size_t w = 0; w < L.wall_u.size(); w++) {
        Bu[off + L.ol + static_cast<int>(w)] = L.wall_u[w];
      }
      for (int i = 0; i < L.size; i++) {
        for (int j = 0; j < L.size; j++) {
          Db(off + i, off + j) = L.Db(i, j);
        }
      }
      off += L.size;
    }
  }

  if (picard_border_) {
    bool converged = false;
    if (!SolveBorderedPicard(Bu, Bphi, Db, r, B_in, X, converged)) {
      return false;  // a base solve failed
    }
    if (converged) {
      return true;
    }
    // The loop gain did not contract within the pass cap (possible as
    // the moments approach neutrality): fall back to the elimination.
    picard_passes_ = -1;
    bool root = true;
#ifdef MFEM_USE_MPI
    if (pfes_) {
      root = pfes_->GetMyRank() == 0;
    }
#endif
    if (root) {
      mfem::out << "SetPicardBorder WARNING: no contraction to " << picard_tol_
                << " in " << picard_max_
                << " passes; falling back to the border elimination.\n";
    }
  }

  if (monolithic_border_) {
    MFEM_VERIFY(type_ != SolverType::SchurCG && !gauge_kkt_ && !mass_gauge_,
                "SetMonolithicBorder: the plain BlockMINRES/BlockCG path "
                "only (experimental).");
    return SolveBorderedMonolithic(Bu, Bphi, Db, r, B_in, X);
  }

  // Base solve, then the shared elimination engine: one solve per
  // border column, cached across solves with an unchanged operator
  // (the viscoelastic stepping economy) and only re-solved when the
  // operator changed or a tighter tolerance is demanded than they were
  // built at. The column solve swaps the raw potential-row data in
  // (psi masked, compatibility fix applied) — the class-side
  // conventions live in this callback, the algebra in the engine.
  bool ok = SolveUnbordered(B_in, X);
  if (!border_elim_) {
    border_elim_ = std::make_unique<BorderedElimination>(
        2,
        [this](const BorderedElimination::State& rhs,
               BorderedElimination::State& xs) {
          Vector Bphi_save(B_phi_);
          Coefficient* psi_save = psi_;
          psi_ = nullptr;
          Vector col(rhs[1]);
          MakeCompatible(col);
          B_phi_ = col;
          xs[0].SetSize(rhs[0].Size());
          xs[0] = 0.0;
          const bool converged = SolveUnbordered(rhs[0], xs[0]);
          xs[1] = Phi_true_;
          B_phi_ = Bphi_save;
          psi_ = psi_save;
          return converged;
        },
        [this](const Vector& a, const Vector& b) { return Dot(a, b); });
  }
  std::vector<BorderedElimination::State> columns(nb);
  for (int j = 0; j < nb; j++) {
    columns[j] = {Bu[j], Bphi[j]};
  }
  BorderedElimination::State xs = {X, Phi_true_};
  Vector y;
  // (With the sea border alone this reduces to Sherman-Morrison,
  // Phi_g = (c.x0 - Sd)/(m + c.y).)
  ok = border_elim_->Eliminate(columns, Db, r, OperatorVersion(), RelTol(),
                               xs, y) &&
       ok;
  phi_g_ = sea_enabled_ ? y[0] : 0.0;
  for (int k = 0; k < nw; k++) {
    omega_[k] = y[ns + k];
  }
  if (level_enabled_) {
    int off = nb0;
    for (auto& L : level_) {
      L.y.SetSize(L.size);
      for (int j = 0; j < L.size; j++) {
        L.y[j] = y[off + j];
      }
      off += L.size;
    }
  }
  if (rot_enabled_) {
    static_cast<CentrifugalPotential*>(rot_psi_total_.get())
        ->SetAmplitudes(omega_);
  }
  X = xs[0];
  Phi_true_ = xs[1];
  // The column solves distributed their own potentials, so redistribute
  // the combined field.
  DistributePotential(Phi_true_);
  return ok;
}

bool LinearQuasiStaticMixedSelfGravitatingProblem::SolveBorderedPicard(
    const std::vector<Vector>& Bu, const std::vector<Vector>& Bphi,
    const DenseMatrix& Db, const Vector& r, const Vector& B_in, Vector& X,
    bool& converged) {
  // Block Gauss-Seidel on [K B; B^T Db]: K x = b - B y with the current
  // border scalars, then Db y = r - B^T x. Each pass is one
  // warm-started base solve; the contraction is the physical loop gain
  // rho(Db^{-1} B^T K^{-1} B). The scalars start from the previous
  // Solve() (phi_g_, omega_), so successive solves of a time
  // integration enter far down the contraction.
  const int nb = static_cast<int>(Bu.size());
  const int ns = sea_enabled_ ? 1 : 0;
  const int nw = nb - ns;
  picard_passes_ = 0;
  converged = false;

  Vector y(nb), g(nb), rhs_y(nb);
  if (ns) {
    y[0] = phi_g_;
  }
  for (int k = 0; k < nw; k++) {
    y[ns + k] = omega_.Size() == nw ? omega_[k] : 0.0;
  }
  DenseMatrix Db_f(Db);
  DenseMatrixInverse Db_inv(Db_f);

  // Anderson mixing on the scalar fixed point y -> G(y): the map is
  // AFFINE in the nb border scalars, so the mixing (GMRES on the
  // nb-dimensional Schur system, one base solve per matvec) terminates
  // in about nb + 1 passes regardless of the plain loop gain — plain
  // Gauss-Seidel stalls at near-neutral moments, where the gain
  // reaches 1 — while a warm-started solve still stops after the one
  // pass that verifies its residual. The difference pairs are NOT
  // recycled across solves: measured, the warm-start-tolerance noise
  // of nearly-converged pairs pollutes later solves, and at a frozen
  // operator the elimination's exactly-probed cached columns are the
  // better algorithm anyway — this route targets the
  // operator-changing regimes.
  const int depth = nb + 1;
  Vector f(nb);
  std::vector<Vector> dY, dF;
  Vector prev_y(nb), prev_f(nb);
  bool have_prev = false;
  auto anderson_next = [&]() {
    if (have_prev) {
      Vector dy(nb), df(nb);
      for (int c = 0; c < nb; c++) {
        dy[c] = y[c] - prev_y[c];
        df[c] = f[c] - prev_f[c];
      }
      if (df.Norml2() > 0.0) {
        dY.push_back(dy);
        dF.push_back(df);
        while (static_cast<int>(dY.size()) > depth) {
          dY.erase(dY.begin());
          dF.erase(dF.begin());
        }
      }
    }
    prev_y = y;
    prev_f = f;
    have_prev = true;
    const int m = static_cast<int>(dY.size());
    if (m == 0) {
      y = g;
      return;
    }
    // min ||f - sum_j a_j dF_j||, then y <- g - sum_j a_j (dY_j + dF_j)
    // (Anderson type II in differences; dG = dY + dF).
    DenseMatrix N(m);
    Vector rhs(m), alpha(m);
    for (int i = 0; i < m; i++) {
      for (int j = i; j < m; j++) {
        const real_t v = dF[i] * dF[j];
        N(i, j) = v;
        N(j, i) = v;
      }
      rhs[i] = dF[i] * f;
    }
    for (int i = 0; i < m; i++) {
      N(i, i) += 1e-12 * (N(i, i) + 1.0);  // near-collinearity guard
    }
    DenseMatrixInverse Ninv(N);
    Ninv.Mult(rhs, alpha);
    y = g;
    for (int j = 0; j < m; j++) {
      for (int c = 0; c < nb; c++) {
        y[c] -= alpha[j] * (dY[j][c] + dF[j][c]);
      }
    }
  };

  const Vector Bphi_save(B_phi_);
  Vector bu_mod(B_in.Size()), col(B_phi_.Size());
  bool ok = true;
  for (int k = 0; k < picard_max_; k++) {
    bu_mod = B_in;
    col = 0.0;
    for (int j = 0; j < nb; j++) {
      bu_mod.Add(-y[j], Bu[j]);
      col.Add(y[j], Bphi[j]);
    }
    // The 2-D compatibility fix on the summed border load, as the
    // elimination applies it per column (linear, so equivalent).
    MakeCompatible(col);
    B_phi_ = Bphi_save;
    B_phi_ -= col;
    ok = SolveUnbordered(bu_mod, X);
    if (!ok) {
      break;
    }
    picard_passes_++;
    // The plain update from the dense rows, and the fixed-point
    // residual it implies for the CURRENT pass.
    for (int i = 0; i < nb; i++) {
      rhs_y[i] = r[i] - Dot(Bu[i], X) - Dot(Bphi[i], Phi_true_);
    }
    Db_inv.Mult(rhs_y, g);
    real_t df = 0.0, ymax = 0.0;
    for (int i = 0; i < nb; i++) {
      f[i] = g[i] - y[i];
      df = std::max(df, std::abs(f[i]));
      ymax = std::max(ymax, std::abs(g[i]));
    }
    if (df <= picard_tol_ * std::max(ymax, real_t{1e-300})) {
      y = g;  // row-2-exact for the returned state
      converged = true;
      break;
    }
    anderson_next();
  }
  B_phi_ = Bphi_save;
  if (!ok || !converged) {
    return ok;  // SolveLinearSystem falls back to the elimination
  }

  phi_g_ = sea_enabled_ ? y[0] : 0.0;
  for (int k = 0; k < nw; k++) {
    omega_[k] = y[ns + k];
  }
  if (rot_enabled_) {
    static_cast<CentrifugalPotential*>(rot_psi_total_.get())
        ->SetAmplitudes(omega_);
  }
  DistributePotential(Phi_true_);
  return true;
}

namespace {

/** The bordered block system [A2 B; B^T Db] as one operator over the
 * monolithic vector (u, phi | y): A2 is the (projected) two-block
 * operator, the columns are pre-projected, and the nb border scalars
 * live on rank 0 alone so the solver's global inner products count them
 * once; every rank keeps the current y by a broadcast per apply. */
class BorderedBlockOperator : public Operator {
 public:
  BorderedBlockOperator(const Operator& A2, const std::vector<Vector>& cols,
                        const DenseMatrix& Db, int rank, bool parallel,
                        const mfem::FiniteElementSpace* fes)
      : Operator(A2.Height() + (rank == 0 ? Db.Height() : 0)),
        A2_(A2),
        cols_(cols),
        Db_(Db),
        n2_(A2.Height()),
        nb_(Db.Height()),
        rank_(rank),
        parallel_(parallel),
        y_(Db.Height()),
        dots_(Db.Height()) {
#ifdef MFEM_USE_MPI
    if (parallel_) {
      comm_ = static_cast<const ParFiniteElementSpace*>(fes)->GetComm();
    }
#else
    (void)fes;
#endif
  }

  void Mult(const Vector& x, Vector& out) const override {
    const Vector x2(const_cast<real_t*>(x.GetData()), n2_);
    Vector out2(out.GetData(), n2_);
    // The border scalars, known to every rank.
    for (int j = 0; j < nb_; j++) {
      y_[j] = rank_ == 0 ? x[n2_ + j] : 0.0;
    }
#ifdef MFEM_USE_MPI
    if (parallel_) {
      MPI_Bcast(y_.GetData(), nb_, MPITypeMap<real_t>::mpi_type, 0, comm_);
    }
#endif
    A2_.Mult(x2, out2);
    for (int j = 0; j < nb_; j++) {
      out2.Add(y_[j], cols_[j]);
      dots_[j] = cols_[j] * x2;
    }
#ifdef MFEM_USE_MPI
    if (parallel_) {
      MPI_Allreduce(MPI_IN_PLACE, dots_.GetData(), nb_,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, comm_);
    }
#endif
    if (rank_ == 0) {
      for (int i = 0; i < nb_; i++) {
        real_t v = dots_[i];
        for (int j = 0; j < nb_; j++) {
          v += Db_(i, j) * y_[j];
        }
        out[n2_ + i] = v;
      }
    }
  }

 private:
  const Operator& A2_;
  const std::vector<Vector>& cols_;
  const DenseMatrix& Db_;
  int n2_, nb_, rank_;
  bool parallel_;
#ifdef MFEM_USE_MPI
  MPI_Comm comm_ = MPI_COMM_NULL;
#endif
  mutable Vector y_, dots_;
};

/** |S| of a small symmetric matrix by cyclic Jacobi: MINRES needs an
 * SPD preconditioner, and the border Schur block is indefinite, so its
 * absolute value V |Lambda| V^T is the standard surrogate. */
DenseMatrix SymmetricAbs(const DenseMatrix& S) {
  const int n = S.Height();
  DenseMatrix A(S), V(n);
  V = 0.0;
  for (int i = 0; i < n; i++) {
    V(i, i) = 1.0;
  }
  for (int sweep = 0; sweep < 30; sweep++) {
    real_t off = 0.0;
    for (int pp = 0; pp < n; pp++) {
      for (int q = pp + 1; q < n; q++) {
        off += A(pp, q) * A(pp, q);
      }
    }
    if (off < 1e-30 * A.FNorm2()) {
      break;
    }
    for (int pp = 0; pp < n; pp++) {
      for (int q = pp + 1; q < n; q++) {
        if (A(pp, q) == 0.0) {
          continue;
        }
        const real_t theta = (A(q, q) - A(pp, pp)) / (2.0 * A(pp, q));
        const real_t t = (theta >= 0 ? 1.0 : -1.0) /
                         (std::abs(theta) +
                          std::sqrt(theta * theta + 1.0));
        const real_t cth = 1.0 / std::sqrt(t * t + 1.0), sth = t * cth;
        for (int k = 0; k < n; k++) {
          const real_t akp = A(k, pp), akq = A(k, q);
          A(k, pp) = cth * akp - sth * akq;
          A(k, q) = sth * akp + cth * akq;
        }
        for (int k = 0; k < n; k++) {
          const real_t apk = A(pp, k), aqk = A(q, k);
          A(pp, k) = cth * apk - sth * aqk;
          A(q, k) = sth * apk + cth * aqk;
          const real_t vkp = V(k, pp), vkq = V(k, q);
          V(k, pp) = cth * vkp - sth * vkq;
          V(k, q) = sth * vkp + cth * vkq;
        }
      }
    }
  }
  DenseMatrix out(n);
  out = 0.0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        out(i, j) += V(i, k) * std::abs(A(k, k)) * V(j, k);
      }
    }
  }
  return out;
}

/** diag(P2, |S~|^{-1}): the two-block preconditioner on (u, phi) and the
 * dense preconditioner-probed border Schur complement on y (its
 * absolute value: MINRES needs SPD). */
class BorderedBlockPreconditioner : public Solver {
 public:
  BorderedBlockPreconditioner(const Solver& P2, const DenseMatrix& Sb,
                              int rank)
      : Solver(P2.Height() + (rank == 0 ? Sb.Height() : 0)),
        P2_(P2),
        Sabs_(SymmetricAbs(Sb)),
        Sinv_(Sabs_),
        n2_(P2.Height()),
        nb_(Sb.Height()),
        rank_(rank) {}

  void SetOperator(const Operator&) override {}

  void Mult(const Vector& x, Vector& out) const override {
    const Vector x2(const_cast<real_t*>(x.GetData()), n2_);
    Vector out2(out.GetData(), n2_);
    P2_.Mult(x2, out2);
    if (rank_ == 0 && nb_ > 0) {
      Vector xb(const_cast<real_t*>(x.GetData()) + n2_, nb_);
      Vector ob(out.GetData() + n2_, nb_);
      Sinv_.Mult(xb, ob);
    }
  }

 private:
  const Solver& P2_;
  DenseMatrix Sabs_;
  DenseMatrixInverse Sinv_;
  int n2_, nb_, rank_;
};

}  // namespace

bool LinearQuasiStaticMixedSelfGravitatingProblem::SolveBorderedMonolithic(
    const std::vector<Vector>& Bu, const std::vector<Vector>& Bphi,
    const DenseMatrix& Db, const Vector& r, const Vector& B_in, Vector& X) {
  EnsureOperator();
  const int nb = Db.Height();
  const int n2 = offsets_[2];
  int rank = 0;
#ifdef MFEM_USE_MPI
  if (IsParallel()) {
    MPI_Comm_rank(static_cast<ParFiniteElementSpace*>(fes_)->GetComm(),
                  &rank);
  }
#endif
  const FiniteElementSpace* dfes = fes_;

  // The columns as projected two-block vectors (compatibility applied to
  // the potential parts, as the elimination does).
  std::vector<Vector> cols(nb);
  for (int j = 0; j < nb; j++) {
    cols[j].SetSize(n2);
    Vector cu(cols[j].GetData(), offsets_[1]);
    Vector cphi(cols[j].GetData() + offsets_[1], offsets_[2] - offsets_[1]);
    cu = Bu[j];
    Vector col(Bphi[j]);
    MakeCompatible(col);
    cphi = col;
    projector_block_->Project(cols[j]);
  }

  // Symmetric equilibration of the border: the scalars' natural scales
  // sit orders away from the field rows' (inertia entries O(C) against
  // field columns O(coupling)), so an unscaled global residual
  // tolerance leaves the border rows' errors large against their own
  // scale — percent-grade omega on the near-cancelling 2-D spin row.
  // Scaling column j by s_j = 1/max(|col_j|, sqrt(|Db_jj|)) keeps the
  // system symmetric and puts every border row at unit scale; the
  // solution scalars are y_j = s_j y'_j.
  Vector bscale(nb);
  DenseMatrix Dbs(Db);
  Vector rs(r);
  for (int j = 0; j < nb; j++) {
    real_t cn2 = cols[j] * cols[j];
#ifdef MFEM_USE_MPI
    if (IsParallel()) {
      MPI_Allreduce(MPI_IN_PLACE, &cn2, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM,
                    static_cast<ParFiniteElementSpace*>(fes_)->GetComm());
    }
#endif
    const real_t scale =
        std::max(std::sqrt(cn2), std::sqrt(std::abs(Db(j, j))));
    // A border scalar with no field column and no diagonal (a
    // constraint row coupled only off-diagonally, e.g. the level
    // border's s rows on a uniform core) keeps unit scale.
    bscale[j] = scale > 0.0 ? 1.0 / scale : 1.0;
    cols[j] *= bscale[j];
    rs[j] *= bscale[j];
  }
  for (int i = 0; i < nb; i++) {
    for (int j = 0; j < nb; j++) {
      Dbs(i, j) *= bscale[i] * bscale[j];
    }
  }

  // The border block of the preconditioner: the preconditioner-probed
  // Schur complement S~ = Db - B^T P^{-1} B — the cancellation structure
  // of the exact border Schur at preconditioner grade, for a handful of
  // preconditioner applications (no solves).
  DenseMatrix Sb(nb);
  {
    Vector z(n2);
    for (int j = 0; j < nb; j++) {
      projected_prec_->Mult(cols[j], z);
      for (int i = 0; i < nb; i++) {
        real_t d = cols[i] * z;
#ifdef MFEM_USE_MPI
        if (IsParallel()) {
          MPI_Allreduce(MPI_IN_PLACE, &d, 1, MPITypeMap<real_t>::mpi_type,
                        MPI_SUM,
                        static_cast<ParFiniteElementSpace*>(fes_)->GetComm());
        }
#endif
        Sb(i, j) = Dbs(i, j) - d;
      }
    }
  }

  BorderedBlockOperator op(*projected_op_, cols, Dbs, rank, IsParallel(),
                           dfes);
  BorderedBlockPreconditioner prec(*projected_prec_, Sb, rank);

  // The monolithic right-hand side and iterate.
  const int nloc = n2 + (rank == 0 ? nb : 0);
  Vector rhs(nloc), sol(nloc);
  {
    Vector r2(rhs.GetData(), n2);
    Vector ru(r2.GetData(), offsets_[1]);
    Vector rphi(r2.GetData() + offsets_[1], offsets_[2] - offsets_[1]);
    B_eff_ = B_in;
    if (psi_) {
      B_eff_ -= tidal_u_;
    }
    ru = B_eff_;
    rphi = B_phi_;
    projector_block_->Project(r2);
    if (rank == 0) {
      for (int j = 0; j < nb; j++) {
        rhs[n2 + j] = rs[j];
      }
    }
  }
  sol = 0.0;

  std::unique_ptr<MINRESSolver> minres;
#ifdef MFEM_USE_MPI
  if (IsParallel()) {
    minres = std::make_unique<MINRESSolver>(
        static_cast<ParFiniteElementSpace*>(fes_)->GetComm());
  } else
#endif
  {
    minres = std::make_unique<MINRESSolver>();
  }
  minres->SetRelTol(rel_tol_);
  minres->SetAbsTol(0.0);
  minres->SetMaxIter(10000);
  minres->SetPrintLevel(print_level_);
  minres->SetOperator(op);
  minres->SetPreconditioner(prec);
  // Iterative refinement on the TRUE residual: with the border Schur
  // preconditioner carrying the (physical) near-neutral directions,
  // MINRES's recurrence-tracked residual can drift from the true one
  // and report convergence at a stagnation floor; recomputing the
  // residual with the exact operator and re-solving for the correction
  // removes the floor at the cost of a short extra solve per pass.
  auto global_dot = [&](const Vector& a, const Vector& b) {
    real_t v = a * b;
#ifdef MFEM_USE_MPI
    if (IsParallel()) {
      MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM,
                    static_cast<ParFiniteElementSpace*>(fes_)->GetComm());
    }
#endif
    return v;
  };
  Vector res(rhs), dsol(nloc);
  const real_t rhs_norm = std::sqrt(global_dot(rhs, rhs));
  bool ok = false;
  outer_its_ = 0;
  for (int pass = 0; pass < 4; pass++) {
    dsol = 0.0;
    minres->Mult(res, dsol);
    sol += dsol;
    outer_its_ += minres->GetNumIterations();
    op.Mult(sol, res);
    res -= rhs;
    res.Neg();
    const real_t rn = std::sqrt(global_dot(res, res));
    ok = rn <= rel_tol_ * rhs_norm;
    if (ok) {
      break;
    }
  }
  NoteIterations(outer_its_);

  // The gauge: the operator is blind to the projected-out modes, so
  // the preconditioner's rounding lets the iterates drift along them
  // without any residual signature — project the solution back, as the
  // elimination path does.
  {
    Vector s2(sol.GetData(), n2);
    projector_block_->Project(s2);
  }

  // Unpack: fields, then the border scalars to every rank. The views
  // are named lvalues so the assignments COPY: an rvalue Vector view
  // would move-assign, leaving the caller's vector aliasing the dying
  // local `sol`.
  const Vector sol_u(sol.GetData(), offsets_[1]);
  const Vector sol_phi(sol.GetData() + offsets_[1],
                       offsets_[2] - offsets_[1]);
  X = sol_u;
  Phi_true_ = sol_phi;
  Vector y(nb);
  if (rank == 0) {
    for (int j = 0; j < nb; j++) {
      y[j] = bscale[j] * sol[n2 + j];
    }
  }
#ifdef MFEM_USE_MPI
  if (IsParallel()) {
    MPI_Bcast(y.GetData(), nb, MPITypeMap<real_t>::mpi_type, 0,
              static_cast<ParFiniteElementSpace*>(fes_)->GetComm());
  }
#endif
  const int ns = sea_enabled_ ? 1 : 0;
  phi_g_ = sea_enabled_ ? y[0] : 0.0;
  for (int k = 0; k < omega_.Size(); k++) {
    omega_[k] = y[ns + k];
  }
  if (rot_enabled_) {
    static_cast<CentrifugalPotential*>(rot_psi_total_.get())
        ->SetAmplitudes(omega_);
  }
  DistributePotential(Phi_true_);
  return ok;
}

bool LinearQuasiStaticMixedSelfGravitatingProblem::SolveUnbordered(
    const Vector& B_in, Vector& X) {
  inner_its_ = 0;
  outer_its_ = 0;
  bool ok = true;

  // Displacement load with the tidal term: B = B_in - C Psi.
  B_eff_ = B_in;
  if (psi_) {
    B_eff_ -= tidal_u_;
  }
  const Vector& B = B_eff_;

  if (type_ == SolverType::SchurCG) {
    // rhs_S = B - C A_phiphi^{-1} B_phi.
    ok = SolvePotential(B_phi_, w_) && ok;
    rhs_s_.SetSize(B.Size());
    C_op_->Mult(w_, rhs_s_);
    rhs_s_ *= -1.0;
    rhs_s_ += B;
    projector_u_->Project(rhs_s_);
    if (!SetWarmStartTolerance(*cg_, *projected_prec_, rhs_s_)) {
      X = 0.0;
      Phi_true_ = w_;
    } else {
      projected_->Mult(rhs_s_, X);
      ok = cg_->GetConverged() && ok;
      outer_its_ = cg_->GetNumIterations();
      NoteIterations(outer_its_);
      // phi = A_phiphi^{-1} (B_phi - C^T u).
      Vector t(B_phi_.Size());
      Ct_op_->Mult(X, t);
      t *= -1.0;
      t += B_phi_;
      ok = SolvePotential(t, Phi_true_) && ok;
    }
  } else if (gauge_kkt_) {
    Bk4_->GetBlock(0) = 0.0;
    Bk4_->GetBlock(1) = 0.0;
    Bk4_->GetBlock(2) = B;
    Bk4_->GetBlock(3) = B_phi_;
    if (!SetWarmStartTolerance(*minres_kkt4_, *projected_prec_kkt4_,
                               *Bk4_)) {
      X = 0.0;
      Phi_true_ = 0.0;
      *Xk4_ = 0.0;
    } else {
      projected_kkt4_->Mult(*Bk4_, *Xk4_);
      ok = minres_kkt4_->GetConverged();
      outer_its_ = minres_kkt4_->GetNumIterations();
      NoteIterations(outer_its_);
      X = Xk4_->GetBlock(0);
      Phi_true_ = Xk4_->GetBlock(1);
    }
  } else {
    B_block_->GetBlock(0) = B;
    B_block_->GetBlock(1) = B_phi_;
    if (!SetWarmStartTolerance(*minres_, *projected_prec_, *B_block_)) {
      X = 0.0;
      Phi_true_ = 0.0;
      *X_block_ = 0.0;
    } else {
      if (GaugePreconditionerOnly() && gauge_plateau_ratio_ > 0.0) {
        // Chunked restarts with the stagnation stop (PlateauMult).
        bool conv = false;
        outer_its_ = PlateauMult(*projected_, *minres_, *B_block_,
                                 *X_block_, conv);
        ok = conv;
      } else {
        // Warm start from the previous MINRES iterate.
        projected_->Mult(*B_block_, *X_block_);
        ok = minres_->GetConverged();
        outer_its_ = minres_->GetNumIterations();
      }
      NoteIterations(outer_its_);
      X = X_block_->GetBlock(0);
      if (mass_gauge_) {
        // The gauge shifted the body by a rigid motion, whose potential
        // (of the rigid modes' near-null partners) the block solve does not
        // carry: solve phi from the gauged u, as the Schur path does.
        Vector t(B_phi_.Size());
        Ct_op_->Mult(X, t);
        t *= -1.0;
        t += B_phi_;
        ok = SolvePotential(t, Phi_true_) && ok;
      } else {
        // The iterate is already in the output gauge (u orthogonal to the
        // rigid modes; in 2-D phi orthogonal to the constant).
        projector_u_->Project(X);
        Phi_true_ = X_block_->GetBlock(1);
      }
    }
  }
  DistributePotential(Phi_true_);
  return ok;
}


// ---------------------------------------------------------------------------
// The level border (SetLevelBorder): doc/dahlen_closure.tex, "The closure in
// the cumulative-volume variable" and "Implementation".

namespace {

// Clamp a level into the grid's range.
real_t ClampLevel(const Vector& c, real_t v) {
  return std::min(std::max(v, c[0]), c[c.Size() - 1]);
}

// The element of the grid containing v (clamped), and its hat values.
void LocateLevel(const Vector& c, real_t v, int& e, real_t& N0, real_t& N1) {
  const int n = c.Size() - 1;
  v = ClampLevel(c, v);
  e = 0;
  while (e < n - 1 && v > c[e + 1]) {
    e++;
  }
  const real_t h = c[e + 1] - c[e];
  N1 = (v - c[e]) / h;
  N0 = 1.0 - N1;
}

// -(m . grad Phi0) / |grad Phi0| on the solid's boundary elements: the
// wall column's coefficient, minus the sign that tells a fluid below the
// solid (outer wall of the region, m . grad Phi0 < 0) from one above.
class MinusWallSignCoefficient : public Coefficient {
 public:
  explicit MinusWallSignCoefficient(VectorCoefficient& g) : g_(&g) {}
  real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    T.SetIntPoint(&ip);
    n_.SetSize(T.GetSpaceDim());
    CalcOrtho(T.Jacobian(), n_);
    g_->Eval(v_, T, ip);
    const real_t nn = n_.Norml2(), vv = v_.Norml2();
    if (nn <= 0.0 || vv <= 0.0) {
      return 0.0;
    }
    return -(v_ * n_) / (nn * vv);
  }

 private:
  VectorCoefficient* g_;
  Vector v_, n_;
};

}  // namespace

void LinearQuasiStaticMixedSelfGravitatingProblem::SetLevelBorder(int levels) {
  MFEM_VERIFY(levels >= 3, "SetLevelBorder: at least three levels.");
  MFEM_VERIFY(!fluids_.empty(), "SetLevelBorder: no fluid regions.");
  for (const auto& f : fluids_) {
    MFEM_VERIFY(f.treatment == FluidRegion::Treatment::Full,
                "SetLevelBorder: every region must use Treatment::Full.");
    MFEM_VERIFY(f.bulk_modulus,
                "SetLevelBorder: every region needs its bulk_modulus.");
  }
  MFEM_VERIFY(!picard_border_,
              "SetLevelBorder: incompatible with the Picard border route.");
  MFEM_VERIFY(!rot_enabled_, "SetLevelBorder: the border's centrifugal "
                             "columns are not wired; no SetRotation().");
  MFEM_VERIFY(!gauge_kkt_, "SetLevelBorder: incompatible with the KKT gauge.");
  MFEM_VERIFY(fluid_rho_prime_.size() == fluids_.size(),
              "SetLevelBorder: the fluid mass term is not set up.");
  level_n_ = levels - 1;
  level_enabled_ = true;
  BuildLevelBorders();
  if (border_elim_) {
    border_elim_->InvalidateCache();
  }
}

void LinearQuasiStaticMixedSelfGravitatingProblem::BuildLevelBorders() {
  level_.clear();
  level_coefs_.clear();
  Mesh* ms = fes_->GetMesh();        // the solid SubMesh
  Mesh* mp = fes_phi_->GetMesh();    // the parent ball
  const int n = level_n_;
  const int qorder = 2 * fes_phi_->GetMaxElementOrder() + 2;
#ifdef MFEM_USE_MPI
  MPI_Comm comm = pfes_phi_ ? pfes_phi_->GetComm() : MPI_COMM_NULL;
  auto allreduce = [&](real_t* a, int count, MPI_Op op) {
    if (pfes_phi_) {
      MPI_Allreduce(MPI_IN_PLACE, a, count, MPITypeMap<real_t>::mpi_type, op,
                    comm);
    }
  };
#else
  auto allreduce = [](real_t*, int, int) {};
  constexpr int MPI_SUM = 0, MPI_MIN = 1, MPI_MAX = 2;
#endif
  BoundaryNormalDotCoefficient m_dot_g(*grad_phi0_shadow_);
  auto marker = fluid_markers_.begin();
  for (size_t r = 0; r < fluids_.size(); r++, ++marker) {
    const FluidRegion& f = fluids_[r];
    LevelBorder L;
    L.n = n;

    // 1. The walls: per interface attribute of the region, the area, the
    // area integrals of Phi0 and of m . grad Phi0 on the solid's boundary
    // elements. The sign of the latter classifies the wall.
    const int nattr = f.interface_marker.Size();
    std::vector<real_t> w_area(nattr, 0.0), w_phi(nattr, 0.0),
        w_mg(nattr, 0.0);
    for (int be = 0; be < ms->GetNBE(); be++) {
      const int a = ms->GetBdrAttribute(be);
      if (a > nattr || !f.interface_marker[a - 1]) {
        continue;
      }
      ElementTransformation* T = ms->GetBdrElementTransformation(be);
      const IntegrationRule& ir = IntRules.Get(T->GetGeometryType(), qorder);
      for (int q = 0; q < ir.GetNPoints(); q++) {
        const IntegrationPoint& ip = ir.IntPoint(q);
        T->SetIntPoint(&ip);
        const real_t w = ip.weight * T->Weight();
        w_area[a - 1] += w;
        w_phi[a - 1] += w * phi0_shadow_->GetValue(*T, ip);
        w_mg[a - 1] += w * m_dot_g.Eval(*T, ip);
      }
    }
    allreduce(w_area.data(), nattr, MPI_SUM);
    allreduce(w_phi.data(), nattr, MPI_SUM);
    allreduce(w_mg.data(), nattr, MPI_SUM);
    Array<int> outer_marker(nattr), inner_marker(nattr);
    outer_marker = 0;
    inner_marker = 0;
    real_t top_area = 0.0, top_phi = 0.0, bot_area = 0.0, bot_phi = 0.0;
    for (int a = 0; a < nattr; a++) {
      if (!f.interface_marker[a] || w_area[a] <= 0.0) {
        continue;
      }
      if (w_mg[a] < 0.0) {
        outer_marker[a] = 1;
        top_area += w_area[a];
        top_phi += w_phi[a];
      } else {
        inner_marker[a] = 1;
        bot_area += w_area[a];
        bot_phi += w_phi[a];
      }
    }
    MFEM_VERIFY(top_area > 0.0,
                "SetLevelBorder: region " << r << " has no outer wall.");
    const real_t c_top = top_phi / top_area;
    L.centre = bot_area <= 0.0;

    // 2. The level range and the foliation assertion on the parent's fluid
    // elements: the extremes of Phi0 and the largest |grad Phi0|, then the
    // smallest |grad Phi0| away from a centre.
    real_t cmin = std::numeric_limits<real_t>::max(), cmax = -cmin;
    real_t gmax = 0.0;
    Vector gv(dim_);
    auto for_each_fluid_point =
        [&](const std::function<void(ElementTransformation&,
                                     const IntegrationPoint&, real_t)>& fn) {
          for (int e = 0; e < mp->GetNE(); e++) {
            if (!(*marker)[mp->GetAttribute(e) - 1]) {
              continue;
            }
            ElementTransformation* T = mp->GetElementTransformation(e);
            const IntegrationRule& ir =
                IntRules.Get(T->GetGeometryType(), qorder);
            for (int q = 0; q < ir.GetNPoints(); q++) {
              const IntegrationPoint& ip = ir.IntPoint(q);
              T->SetIntPoint(&ip);
              fn(*T, ip, ip.weight * T->Weight());
            }
          }
        };
    for_each_fluid_point([&](ElementTransformation& T,
                             const IntegrationPoint& ip, real_t) {
      const real_t c = phi0_->GetValue(T, ip);
      cmin = std::min(cmin, c);
      cmax = std::max(cmax, c);
      grad_phi0_->Eval(gv, T, ip);
      gmax = std::max(gmax, gv.Norml2());
    });
    allreduce(&cmin, 1, MPI_MIN);
    allreduce(&cmax, 1, MPI_MAX);
    allreduce(&gmax, 1, MPI_MAX);
    const real_t c_bot = L.centre ? cmin : bot_phi / bot_area;
    MFEM_VERIFY(c_top > c_bot,
                "SetLevelBorder: region " << r << " has no level range.");
    real_t gmin_away = gmax;
    const real_t range = c_top - c_bot;
    for_each_fluid_point([&](ElementTransformation& T,
                             const IntegrationPoint& ip, real_t) {
      const real_t c = phi0_->GetValue(T, ip);
      if (L.centre && c - c_bot < 0.05 * range) {
        return;  // near the centre g -> 0 legitimately
      }
      grad_phi0_->Eval(gv, T, ip);
      gmin_away = std::min(gmin_away, gv.Norml2());
    });
    allreduce(&gmin_away, 1, MPI_MIN);
    MFEM_VERIFY(gmin_away > 0.05 * gmax,
                "SetLevelBorder: |grad Phi0| vanishes inside fluid region "
                    << r << " away from a centre (" << gmin_away << " against "
                    << gmax << "); the level foliation is not regular.");

    // 3. The grid, uniform in the level.
    L.c.SetSize(n + 1);
    for (int k = 0; k <= n; k++) {
      L.c[k] = c_bot + (c_top - c_bot) * k / n;
    }

    // 4. The volume integrals: per-element volume, mass and kappa-volume
    // (the level-mean coefficients), and the hat-function blocks
    // K_ss = -int rho_Phi N_i N_j dV and K_snu = int N_i N_j dV.
    L.vol.SetSize(n);
    L.mass.SetSize(n);
    L.kap.SetSize(n);
    L.vol = 0.0;
    L.mass = 0.0;
    L.kap = 0.0;
    DenseMatrix Kss(n + 1), Ksnu(n + 1);
    Kss = 0.0;
    Ksnu = 0.0;
    Coefficient* rho_prime = fluid_rho_prime_[r];
    for_each_fluid_point([&](ElementTransformation& T,
                             const IntegrationPoint& ip, real_t w) {
      int e;
      real_t N0, N1;
      LocateLevel(L.c, phi0_->GetValue(T, ip), e, N0, N1);
      const real_t rho = f.density->Eval(T, ip);
      const real_t rp = rho_prime->Eval(T, ip);
      const real_t kap = f.bulk_modulus->Eval(T, ip);
      L.vol[e] += w;
      L.mass[e] += rho * w;
      L.kap[e] += kap * w;
      const real_t N[2] = {N0, N1};
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
          Kss(e + i, e + j) -= rp * N[i] * N[j] * w;
          Ksnu(e + i, e + j) += N[i] * N[j] * w;
        }
      }
    });
    allreduce(L.vol.GetData(), n, MPI_SUM);
    allreduce(L.mass.GetData(), n, MPI_SUM);
    allreduce(L.kap.GetData(), n, MPI_SUM);
    allreduce(Kss.Data(), (n + 1) * (n + 1), MPI_SUM);
    allreduce(Ksnu.Data(), (n + 1) * (n + 1), MPI_SUM);
    for (int e = 0; e < n; e++) {
      MFEM_VERIFY(L.vol[e] > 0.0,
                  "SetLevelBorder: level element " << e << " of region " << r
                  << " holds no fluid quadrature point; use fewer levels.");
    }

    // 5. The nu columns, int N_k(Phi0) chi dV on the potential's true
    // dofs, assembled on the SAME quadrature points as the 1-D block so
    // that sum_j (b_k)_j = int N_k dV holds exactly and the discrete mass
    // balance of the border is an identity.
    {
      std::vector<Vector> colL(n + 1, Vector(fes_phi_->GetVSize()));
      for (auto& v : colL) {
        v = 0.0;
      }
      Array<int> vdofs;
      Vector shape;
      for (int e = 0; e < mp->GetNE(); e++) {
        if (!(*marker)[mp->GetAttribute(e) - 1]) {
          continue;
        }
        const FiniteElement* fe = fes_phi_->GetFE(e);
        fes_phi_->GetElementVDofs(e, vdofs);
        shape.SetSize(fe->GetDof());
        ElementTransformation* T = mp->GetElementTransformation(e);
        const IntegrationRule& ir = IntRules.Get(T->GetGeometryType(), qorder);
        for (int q = 0; q < ir.GetNPoints(); q++) {
          const IntegrationPoint& ip = ir.IntPoint(q);
          T->SetIntPoint(&ip);
          const real_t w = ip.weight * T->Weight();
          fe->CalcShape(ip, shape);
          int e1;
          real_t N0, N1;
          LocateLevel(L.c, phi0_->GetValue(*T, ip), e1, N0, N1);
          for (int j = 0; j < vdofs.Size(); j++) {
            const int d = vdofs[j];
            const int dof = d >= 0 ? d : -1 - d;
            const real_t sgn = d >= 0 ? 1.0 : -1.0;
            colL[e1][dof] += sgn * w * N0 * shape[j];
            colL[e1 + 1][dof] += sgn * w * N1 * shape[j];
          }
        }
      }
      L.nu_phi.resize(n + 1);
      for (int k = 0; k <= n; k++) {
        ToTrueDofs(*fes_phi_, colL[k], L.nu_phi[k]);
      }
    }

    // 6. The wall columns, -int (m . grad Phi0 / |grad Phi0|) (m . v) dS per
    // wall, outer first; the constrained node is n for the outer wall and 0
    // for the inner one.
    auto sign = std::make_unique<MinusWallSignCoefficient>(*grad_phi0_shadow_);
    auto add_wall = [&](const Array<int>& wm, int node) {
      auto lf = detail::MakeLinearForm(fes_);
      lf->AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(*sign),
                                const_cast<Array<int>&>(wm));
      lf->Assemble();
      Vector col;
      ToTrueDofs(*fes_, *lf, col);
      L.wall_u.push_back(col);
      L.wall_node.push_back(node);
    };
    add_wall(outer_marker, n);
    if (!L.centre) {
      add_wall(inner_marker, 0);
    }
    level_coefs_.push_back(std::move(sign));

    // 7. The 1-D block in the local ordering (U | s | nu | lambda).
    L.nU = L.centre ? n : n + 1;
    L.oU = 0;
    L.os = L.nU;
    L.onu = L.os + n + 1;
    L.ol = L.onu + n + 1;
    L.size = L.ol + static_cast<int>(L.wall_u.size());
    L.Db.SetSize(L.size);
    L.Db = 0.0;
    auto U = [&](int node) { return L.oU + L.UIndex(node); };
    auto S = [&](int node) { return L.os + node; };
    auto NU = [&](int node) { return L.onu + node; };
    for (int e = 0; e < n; e++) {
      const real_t h = L.c[e + 1] - L.c[e];
      const real_t rho_e = L.mass[e] / L.vol[e];
      // int kappa N_i' N_j' / A dc over the element: with the level-mean
      // kappa_e = kap_e / Vol_e and <A> = Vol_e / h, the integral is
      // kappa_e h / <A> = kappa_e h^2 / Vol_e, and the derivatives
      // contribute 1/h^2: kappa_e / Vol_e = kap_e / Vol_e^2. At a centre
      // A ~ sqrt(c - c_min) in three dimensions and the element mean of
      // 1/A is 4/3 of 1/<A>.
      real_t kuu = L.kap[e] / (L.vol[e] * L.vol[e]);
      if (L.centre && e == 0 && dim_ == 3) {
        kuu *= 4.0 / 3.0;
      }
      for (int i = 0; i < 2; i++) {
        const int ni = e + i;
        const real_t di = (i == 0) ? -1.0 : 1.0;  // sign of N_ni' on e
        for (int j = 0; j < 2; j++) {
          const int nj = e + j;
          const real_t dj = (j == 0) ? -1.0 : 1.0;
          // K_U nu: the 1-D mass matrix.
          const real_t m = h / 6.0 * ((i == j) ? 2.0 : 1.0);
          if (!(L.centre && ni == 0)) {
            L.Db(U(ni), NU(nj)) += m;
            L.Db(NU(nj), U(ni)) += m;
            // K_Us = int rho N_i' N_j dc = rho_e d_i / 2.
            L.Db(U(ni), S(nj)) += rho_e * di / 2.0;
            L.Db(S(nj), U(ni)) += rho_e * di / 2.0;
            if (!(L.centre && nj == 0)) {
              L.Db(U(ni), U(nj)) += kuu * di * dj;
            }
          }
        }
      }
    }
    for (int i = 0; i <= n; i++) {
      for (int j = 0; j <= n; j++) {
        L.Db(S(i), S(j)) += Kss(i, j);
        L.Db(S(i), NU(j)) += Ksnu(i, j);
        L.Db(NU(i), S(j)) += Ksnu(i, j);
      }
    }
    for (size_t w = 0; w < L.wall_u.size(); w++) {
      const int lam = L.ol + static_cast<int>(w);
      L.Db(U(L.wall_node[w]), lam) = 1.0;
      L.Db(lam, U(L.wall_node[w])) = 1.0;
    }
    level_.push_back(std::move(L));
  }
}

Vector LinearQuasiStaticMixedSelfGravitatingProblem::LevelBorderDiagnostics(
    int region) const {
  MFEM_VERIFY(level_enabled_ && region >= 0 &&
                  region < static_cast<int>(level_.size()),
              "LevelBorderDiagnostics: no such level border.");
  const LevelBorder& L = level_[region];
  MFEM_VERIFY(L.y.Size() == L.size, "LevelBorderDiagnostics: no solve yet.");
  Vector U_true;
  Displacement().GetTrueDofs(U_true);
  const int nwalls = static_cast<int>(L.wall_u.size());
  Vector out(2 * nwalls + 2);
  real_t wall_mass = 0.0;
  // The fluid-side density at each wall: the level-mean density of the
  // adjacent element of the grid.
  for (int w = 0; w < nwalls; w++) {
    const int node = L.wall_node[w];
    const real_t Uw = (L.centre && node == 0) ? 0.0 : L.y[L.oU + L.UIndex(node)];
    const real_t flux = -Dot(L.wall_u[w], U_true);
    out[2 * w] = Uw;
    out[2 * w + 1] = flux;
    const int e = (node == L.n) ? L.n - 1 : 0;
    const real_t rho_w = L.mass[e] / L.vol[e];
    wall_mass += (node == L.n ? -rho_w : rho_w) * flux;
  }
  // Bulk: sum_k nu_k int N_k dV + int rho_Phi phi dV over the region.
  Vector ones(fes_phi_->GetTrueVSize());
  ones = 1.0;
  real_t bulk = 0.0;
  for (int k = 0; k <= L.n; k++) {
    bulk += L.y[L.onu + k] * Dot(L.nu_phi[k], ones);
  }
  auto marker = fluid_markers_.begin();
  std::advance(marker, region);
  Mesh* mp = fes_phi_->GetMesh();
  const int qorder = 2 * fes_phi_->GetMaxElementOrder() + 2;
  Coefficient* rho_prime = fluid_rho_prime_[region];
  const GridFunction& phi = Potential();
  real_t f1 = 0.0;
  for (int e = 0; e < mp->GetNE(); e++) {
    if (!(*marker)[mp->GetAttribute(e) - 1]) {
      continue;
    }
    ElementTransformation* T = mp->GetElementTransformation(e);
    const IntegrationRule& ir = IntRules.Get(T->GetGeometryType(), qorder);
    for (int q = 0; q < ir.GetNPoints(); q++) {
      const IntegrationPoint& ip = ir.IntPoint(q);
      T->SetIntPoint(&ip);
      f1 += ip.weight * T->Weight() * rho_prime->Eval(*T, ip) *
            phi.GetValue(*T, ip);
    }
  }
#ifdef MFEM_USE_MPI
  if (pfes_phi_) {
    MPI_Allreduce(MPI_IN_PLACE, &f1, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                  pfes_phi_->GetComm());
  }
#endif
  out[2 * nwalls] = bulk + f1;
  out[2 * nwalls + 1] = wall_mass;
  // Debug tail: the border rows' residual max |B^T x + Db y|, the pieces
  // f1 = int rho_Phi phi dV, nu_mass = sum nu_k int N_k dV, and
  // s_mass = int rho_Phi s A dc.
  Vector phi_true;
  Potential().GetTrueDofs(phi_true);
  Vector Dby(L.size);
  L.Db.Mult(L.y, Dby);
  real_t rmax = 0.0;
  for (int j = 0; j < L.size; j++) {
    real_t rj = Dby[j];
    if (j >= L.onu && j < L.ol) {
      rj += Dot(L.nu_phi[j - L.onu], phi_true);
    } else if (j >= L.ol) {
      rj += Dot(L.wall_u[j - L.ol], U_true);
    }
    rmax = std::max(rmax, std::abs(rj));
  }
  real_t s_mass = 0.0;
  for (int i = 0; i <= L.n; i++) {
    for (int j = 0; j <= L.n; j++) {
      // -Kss(i,j) = int rho_Phi N_i N_j dV; Db(S(i),S(j)) holds Kss.
      s_mass -= L.Db(L.os + i, L.os + j) * L.y[L.os + j];
    }
  }
  Vector dbg(4);
  dbg[0] = rmax;
  dbg[1] = f1;
  dbg[2] = bulk;
  dbg[3] = s_mass;
  Vector all(out.Size() + 4);
  for (int i = 0; i < out.Size(); i++) all[i] = out[i];
  for (int i = 0; i < 4; i++) all[out.Size() + i] = dbg[i];
  return all;
}

void LinearQuasiStaticMixedSelfGravitatingProblem::LevelBorderSolution(
    int region, Vector& c, Vector& U, Vector& s, Vector& nu,
    Vector& lambda) const {
  MFEM_VERIFY(level_enabled_ && region >= 0 &&
                  region < static_cast<int>(level_.size()),
              "LevelBorderSolution: no such level border.");
  const LevelBorder& L = level_[region];
  MFEM_VERIFY(L.y.Size() == L.size, "LevelBorderSolution: no solve yet.");
  const int n = L.n;
  c = L.c;
  U.SetSize(n + 1);
  s.SetSize(n + 1);
  nu.SetSize(n + 1);
  for (int k = 0; k <= n; k++) {
    U[k] = (L.centre && k == 0) ? 0.0 : L.y[L.oU + L.UIndex(k)];
    s[k] = L.y[L.os + k];
    nu[k] = L.y[L.onu + k];
  }
  lambda.SetSize(static_cast<int>(L.wall_u.size()));
  for (int w = 0; w < lambda.Size(); w++) {
    lambda[w] = L.y[L.ol + w];
  }
}

}  // namespace AdGIA
