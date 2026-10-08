/**
 * @file mixed_problem.cpp
 * @brief Implementation of LinearQuasiStaticMixedSelfGravitatingProblem.
 */

#include "AdGIA/mixed_problem.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>

#include "AdGIA/bilininteg.hpp"
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
  // body's response, not of the load's own potential.
  Vector Phi;
  bool ok;
  if (A_phiphi_ == A_lap_.get()) {
    ok = SolvePotential(B_phi_, Phi);
  } else {
    const Operator* saved = A_phiphi_;
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
      const Operator* saved = A_phiphi_;
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
        continue;  // buoyancy-only (Winkler) interface: no (F3)
      }
      auto minus_rho = std::make_unique<ProductCoefficient>(
          -1.0, InterfaceDensity(f));
      form.AddBoundaryIntegrator(
          new BoundaryNormalScalarIntegrator(*minus_rho), f.interface_marker);
      fluid_coefs_.push_back(std::move(minus_rho));
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
  operator_dirty_ = true;
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
  psi_ = &psi;
  RegisterTimeDependent(psi);
  psi_gf_ = detail::MakeGridFunction(fes_phi_);
  *psi_gf_ = 0.0;
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
  b_phi_->Assemble();
  Vector bL(fes_phi_->GetVSize());
  injection_->Mult(*b_phi_, bL);
  ToTrueDofs(*fes_phi_, bL, B_phi_);
  AssembleTidalLoad();
  MakeCompatible(B_phi_);
}

// ---------------------------------------------------------------------------
// Controls

void LinearQuasiStaticMixedSelfGravitatingProblem::SetMassWeightedGauge(bool on) {
  if (on && !gauge_M_.Ptr()) {
    gauge_form_ = AssembleMassOperator(rho_, gauge_M_);
  }
  if (on != mass_gauge_) {
    mass_gauge_ = on;
    operator_dirty_ = true;  // the solvers are rebuilt with the gauge
  }
}

void LinearQuasiStaticMixedSelfGravitatingProblem::SetSolverType(SolverType type) {
  if (type != type_) {
    type_ = type;
    operator_dirty_ = true;
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
  operator_dirty_ = true;
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
    // The fluid mass term may depend on grad Phi0.
    SetupFluidMass();
  }
  operator_dirty_ = true;
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
real_t SturmCount(const std::vector<real_t>& alpha,
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
    for (size_t j = 0; j < V.size(); j++) {  // full reorthogonalisation
      w.Add(-Dot(w, V[j]), V[j]);
    }
    for (size_t j = 0; j < V.size(); j++) {
      w.Add(-Dot(w, V[j]), V[j]);
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
  // The solid-everywhere preconditioner experiment: the Krylov operator
  // keeps the CLEAN displacement block while prec_ (built by
  // SetupDefaultCG on the handle passed in) sits on A + eps_prec Q;
  // MINRES ignores the exact gauge kernel, the near-gauge modes move
  // into the stopping rule (SetGaugePreconditionerOnly).
  block_op_->SetBlock(0, 0,
                      GaugePreconditionerOnly() ? A_.Ptr() : A.Ptr());
  block_op_->SetBlock(0, 1, const_cast<Operator*>(C_op_));
  block_op_->SetBlock(1, 0, const_cast<Operator*>(Ct_op_));
  block_op_->SetBlock(1, 1, const_cast<Operator*>(A_phiphi_));

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
  // reuse; zero refinements (the KKT solve needs none).
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
  block_op_kkt4_->SetBlock(1, 3, const_cast<Operator*>(A_phiphi_));
  block_op_kkt4_->SetBlock(2, 0, Auu);
  block_op_kkt4_->SetBlock(2, 1, const_cast<Operator*>(C_op_));
  block_op_kkt4_->SetBlock(3, 0, const_cast<Operator*>(Ct_op_));
  block_op_kkt4_->SetBlock(3, 1, const_cast<Operator*>(A_phiphi_));

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
    amg->SetSystemsOptions(fes_->GetMesh()->Dimension());
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

}  // namespace AdGIA
