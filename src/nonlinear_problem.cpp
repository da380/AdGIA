/**
 * @file nonlinear_problem.cpp
 * @brief Implementation of nonlinear_problem.hpp.
 */

#include "AdGIA/nonlinear_problem.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

#include "AdGIA/bilininteg.hpp"
#include "AdGIA/detail/fem_factory.hpp"
#include "AdGIA/lininteg.hpp"

namespace AdGIA {

using namespace mfem;

namespace {

// The quadrature of the energy and of the residual on an element: the
// same rule for both, so that the residual is the first variation of the
// discrete energy and not of another quadrature of it.
int QuadratureOrder(const FiniteElement& fe, ElementTransformation& T) {
  return 2 * fe.GetOrder() + T.OrderW();
}

bool Root(const FiniteElementSpace& fes) {
#ifdef MFEM_USE_MPI
  if (const auto* pfes = dynamic_cast<const ParFiniteElementSpace*>(&fes)) {
    return pfes->GetMyRank() == 0;
  }
#endif
  (void)fes;
  return true;
}

// The pairing of a dual vector assembled as a linear form with a field.
real_t Pair(LinearForm& b, GridFunction& u) {
#ifdef MFEM_USE_MPI
  if (auto* pb = dynamic_cast<ParLinearForm*>(&b)) {
    return (*pb)(*static_cast<ParGridFunction*>(&u));
  }
#endif
  return b(u);
}

}  // namespace

NonlinearClampedProblem::NonlinearClampedProblem(
    FiniteElementSpace* fes, MaterialLaw& law, const Array<int>& ess_bdr,
    VectorCoefficient& traction, const Array<int>& traction_marker,
    VectorCoefficient* dirichlet, Diffeomorphism* equilibrium)
    : fes_(fes),
      law_(&law),
      ess_bdr_(ess_bdr),
      traction_marker_(traction_marker),
      traction_(&traction),
      dirichlet_(dirichlet) {
  const auto dim = fes->GetMesh()->SpaceDimension();
  MFEM_VERIFY(fes->GetVDim() == dim && law.SpaceDim() == dim,
              "NonlinearClampedProblem: the space and the law must have the "
              "dimension of the mesh.");
  u_ = detail::MakeGridFunction(fes);
  *u_ = 0.0;
  // The configuration is the equilibrium mapping displaced by u, and the
  // law one about the mesh.
  identity_ = std::make_unique<IdentityDiffeomorphism>(dim);
  auto& base = equilibrium ? *equilibrium : *identity_;
  auto* about_mesh = &law;
  if (!base.IsIdentity()) {
    mapped_law_ = std::make_unique<MappedMaterialLaw>(law, base);
    about_mesh = mapped_law_.get();
  }
  map_ = std::make_unique<DisplacedDiffeomorphism>(base);
  map_->AddField(*u_);
  energy_c_ = std::make_unique<StrainEnergyDensityCoefficient>(
      *about_mesh, *map_, &base);
  stress_c_ = std::make_unique<FirstPiolaKirchhoffStressCoefficient>(
      *about_mesh, *map_, &base);
  tangent_stress_c_ = std::make_unique<SecondPiolaKirchhoffStressCoefficient>(
      *about_mesh, *map_);
  tangent_c_ =
      std::make_unique<MaterialTangentCoefficient>(*about_mesh, *map_);
  rheology_ = std::make_unique<ReferentialElasticRheology>(
      dim, *tangent_c_, *tangent_stress_c_, *map_);
  auto none = Vector(dim);
  none = 0.0;
  zero_ = std::make_unique<VectorConstantCoefficient>(none);
  // The tangent problem solves for corrections: no loads of its own, and
  // on the clamped part what the displacement still lacks of its
  // prescribed value. That is the whole of the prescribed value at a
  // first iteration from the reference, where the tangent then carries
  // the boundary data into the body, and zero afterwards.
  state_c_ = std::make_unique<VectorGridFunctionCoefficient>(u_.get());
  lacking_c_ = std::make_unique<VectorSumCoefficient>(
      dirichlet_ ? *dirichlet_ : *zero_, *state_c_, 1.0, -1.0);
  tangent_ = std::make_unique<LinearQuasiStaticClampedProblem>(
      fes, *rheology_, ess_bdr_, *zero_, traction_marker_, lacking_c_.get());

  // The stiffness is integrated with the rule of the residual, where one
  // rule serves every element, so that the tangent is the derivative of
  // the residual as assembled.
  auto* mesh = fes->GetMesh();
  const auto generator = mesh->MeshGenerator();
  if (mesh->GetNE() > 0 && (generator == 1 || generator == 2)) {
    auto* T = mesh->GetElementTransformation(0);
    const auto* fe = fes->GetFE(0);
    const auto& rule =
        IntRules.Get(fe->GetGeomType(), QuadratureOrder(*fe, *T));
    auto& integrators = *tangent_->StiffnessIntegrators().GetDBFI();
    for (auto* integrator : integrators) {
      integrator->SetIntRule(&rule);
    }
  }

  internal_ = detail::MakeLinearForm(fes);
  internal_->AddDomainIntegrator(
      new DomainLFDeformationGradientIntegrator(*stress_c_));
  load_ = detail::MakeLinearForm(fes);
  load_->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(*traction_),
                               traction_marker_);
}

NonlinearClampedProblem::~NonlinearClampedProblem() = default;

void NonlinearClampedProblem::SetBodyForce(VectorCoefficient& force) {
  MFEM_VERIFY(!body_force_ && !load_assembled_,
              "NonlinearClampedProblem::SetBodyForce: set the body force "
              "once, before the first evaluation.");
  body_force_ = &force;
  load_->AddDomainIntegrator(new VectorDomainLFIntegrator(force));
}

void NonlinearClampedProblem::SetStressLoad(MatrixCoefficient& stress) {
  MFEM_VERIFY(!stress_load_ && !load_assembled_,
              "NonlinearClampedProblem::SetStressLoad: set the load once, "
              "before the first evaluation.");
  stress_load_ = &stress;
  load_->AddDomainIntegrator(
      new DomainLFDeformationGradientIntegrator(stress));
}

bool NonlinearClampedProblem::IsParallel() const {
  return detail::IsParallel(*fes_);
}

int NonlinearClampedProblem::TangentAssemblies() const {
  return tangent_->NumAssemblies();
}

real_t NonlinearClampedProblem::Norm(const GridFunction& v) const {
  return v.ComputeL2Error(*zero_);
}

real_t NonlinearClampedProblem::Energy() {
  auto* mesh = fes_->GetMesh();
  auto energy = real_t{0};
  for (auto e = 0; e < mesh->GetNE(); e++) {
    auto* T = mesh->GetElementTransformation(e);
    const auto* fe = fes_->GetFE(e);
    const auto& rule =
        IntRules.Get(fe->GetGeomType(), QuadratureOrder(*fe, *T));
    for (auto q = 0; q < rule.GetNPoints(); q++) {
      const auto& ip = rule.IntPoint(q);
      T->SetIntPoint(&ip);
      energy += ip.weight * T->Weight() * energy_c_->Eval(*T, ip);
    }
  }
#ifdef MFEM_USE_MPI
  if (auto* pfes = dynamic_cast<ParFiniteElementSpace*>(fes_)) {
    auto local = energy;
    MPI_Allreduce(&local, &energy, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                  pfes->GetComm());
  }
#endif
  if (!load_assembled_) {
    load_->Assemble();
    load_assembled_ = true;
  }
  return energy - Pair(*load_, *u_);
}

void NonlinearClampedProblem::Residual(Vector& r) {
  if (!load_assembled_) {
    load_->Assemble();
    load_assembled_ = true;
  }
  internal_->Assemble();
  r.SetSize(internal_->Size());
  subtract(*internal_, *load_, r);
}

LinearQuasiStaticClampedProblem& NonlinearClampedProblem::Tangent() {
  tangent_->MarkOperatorDirty();
  return *tangent_;
}

bool NonlinearClampedProblem::Solve() {
  corrections_.clear();
  tangent_->SetRelTol(options_.linear_rel_tol);
  const auto print = options_.print && Root(*fes_);
  auto residual = Vector();
  // The tangent is assembled at the first iteration, and at every one
  // when the method is Newton's proper.
  auto newton = options_.refresh_contraction <= 0.0;
  auto refresh = true;
  auto previous = real_t{0};
  for (auto it = 0; it < options_.max_iterations; it++) {
    Residual(residual);
    residual.Neg();
    if (refresh) {
      tangent_->MarkOperatorDirty();
    }
    tangent_->AssembleForce(0.0);
    tangent_->AddForce(residual);
    if (!tangent_->Solve()) {
      if (print) {
        std::cout << "NonlinearClampedProblem: the linear solve of iteration "
                  << it + 1 << " did not converge.\n";
      }
      return false;
    }
    const auto& correction = tangent_->Displacement();
    const auto size = Norm(correction);
    if (!refresh && size > options_.refresh_contraction * previous) {
      // The kept tangent has stopped serving: its correction is not
      // taken, and the rest of the solve is Newton's method proper.
      if (print) {
        std::cout << "  Newton " << std::setw(2) << it + 1 << ": correction "
                  << std::scientific << std::setprecision(3) << size
                  << " of the kept tangent not taken\n"
                  << std::defaultfloat;
      }
      newton = true;
      refresh = true;
      continue;
    }
    *u_ += correction;
    const auto scale = Norm(*u_);
    corrections_.push_back(size);
    if (print) {
      std::cout << "  Newton " << std::setw(2) << it + 1 << ": correction "
                << std::scientific << std::setprecision(3) << size
                << ", displacement " << scale
                << (refresh ? ", tangent assembled" : "") << "\n"
                << std::defaultfloat;
    }
    if (size <= std::max(options_.rel_tol * scale, options_.abs_tol)) {
      return true;
    }
    refresh = newton;
    previous = size;
  }
  return false;
}

// ---------------------------------------------------------------------------
// NonlinearSelfGravitatingProblem

NonlinearSelfGravitatingProblem::NonlinearSelfGravitatingProblem(
    FiniteElementSpace* fes_u, FiniteElementSpace* fes_zeta,
    FiniteElementSpace* fes_buffer, const SparseMatrix& extension,
    MaterialLaw& law, Coefficient& density, real_t gravitational_constant,
    int dtn_degree, Diffeomorphism* equilibrium)
    : fes_u_(fes_u),
      fes_zeta_(fes_zeta),
      fes_buffer_(fes_buffer),
      extension_(&extension),
      law_(&law),
      rho_(&density),
      G_(gravitational_constant),
      dtn_degree_(dtn_degree) {
  MFEM_VERIFY(!detail::IsParallel(*fes_u),
              "NonlinearSelfGravitatingProblem: parallel spaces take the "
              "extension on true dofs.");
  Initialise(equilibrium);
}

#ifdef MFEM_USE_MPI
NonlinearSelfGravitatingProblem::NonlinearSelfGravitatingProblem(
    ParFiniteElementSpace* fes_u, ParFiniteElementSpace* fes_zeta,
    ParFiniteElementSpace* fes_buffer, const HypreParMatrix& extension,
    MaterialLaw& law, Coefficient& density, real_t gravitational_constant,
    int dtn_degree, Diffeomorphism* equilibrium)
    : fes_u_(fes_u),
      fes_zeta_(fes_zeta),
      fes_buffer_(fes_buffer),
      pextension_(&extension),
      law_(&law),
      rho_(&density),
      G_(gravitational_constant),
      dtn_degree_(dtn_degree) {
  Initialise(equilibrium);
}
#endif

NonlinearSelfGravitatingProblem::~NonlinearSelfGravitatingProblem() = default;

void NonlinearSelfGravitatingProblem::Initialise(Diffeomorphism* equilibrium) {
  const auto dim = fes_u_->GetMesh()->SpaceDimension();
  MFEM_VERIFY(fes_u_->GetVDim() == dim && law_->SpaceDim() == dim &&
                  fes_buffer_->GetVDim() == dim && fes_zeta_->GetVDim() == 1,
              "NonlinearSelfGravitatingProblem: vector spaces of the mesh's "
              "dimension for the displacement and its extension, a scalar "
              "one for the potential, and a law of that dimension.");
  identity_ = std::make_unique<IdentityDiffeomorphism>(dim);
  auto& base = equilibrium ? *equilibrium : *identity_;
  auto* about_mesh = law_;
  if (!base.IsIdentity()) {
    mapped_law_ = std::make_unique<MappedMaterialLaw>(*law_, base);
    about_mesh = mapped_law_.get();
  }

  // The displacement on the body, in the buffer and on the ball.
#ifdef MFEM_USE_MPI
  if (auto* pzeta = dynamic_cast<ParFiniteElementSpace*>(fes_zeta_)) {
    fes_ball_ = std::make_unique<ParFiniteElementSpace>(
        pzeta->GetParMesh(), fes_u_->FEColl(), dim);
  } else
#endif
  {
    fes_ball_ = std::make_unique<FiniteElementSpace>(fes_zeta_->GetMesh(),
                                                     fes_u_->FEColl(), dim);
  }
  u_ = detail::MakeGridFunction(fes_u_);
  u_buffer_ = detail::MakeGridFunction(fes_buffer_);
  u_ball_ = detail::MakeGridFunction(fes_ball_.get());
  zeta_ = detail::MakeGridFunction(fes_zeta_);
  zeta0_ = detail::MakeGridFunction(fes_zeta_);
  *u_ = 0.0;
  *u_buffer_ = 0.0;
  *u_ball_ = 0.0;
  *zeta_ = 0.0;
  *zeta0_ = 0.0;
  map_ = std::make_unique<DisplacedDiffeomorphism>(base);
  map_->AddField(*u_);
  map_->AddField(*u_buffer_);
  map_->AddField(*u_ball_);

  energy_c_ = std::make_unique<StrainEnergyDensityCoefficient>(
      *about_mesh, *map_, &base);
  stress_c_ = std::make_unique<FirstPiolaKirchhoffStressCoefficient>(
      *about_mesh, *map_, &base);
  tangent_stress_c_ = std::make_unique<SecondPiolaKirchhoffStressCoefficient>(
      *about_mesh, *map_);
  tangent_c_ =
      std::make_unique<MaterialTangentCoefficient>(*about_mesh, *map_);
  rheology_ = std::make_unique<ReferentialElasticRheology>(
      dim, *tangent_c_, *tangent_stress_c_, *map_);
  zeta_c_ = std::make_unique<GridFunctionCoefficient>(zeta_.get());
  auto none = Vector(dim);
  none = 0.0;
  zero_ = std::make_unique<VectorConstantCoefficient>(none);

  // The rule: of an order for the displacement's and the potential's
  // elements both, on a mesh of one element type. It is the same on
  // every rank, those that hold no element of the body among them: the
  // linear problems are made with it by all the ranks together.
  auto* mesh = fes_u_->GetMesh();
  const auto generator = mesh->MeshGenerator();
  MFEM_VERIFY(generator == 1 || generator == 2,
              "NonlinearSelfGravitatingProblem: a mesh of one element type.");
  auto weight_order =
      mesh->GetNE() > 0 ? mesh->GetElementTransformation(0)->OrderW() : -1;
#ifdef MFEM_USE_MPI
  if (auto* pfes = dynamic_cast<ParFiniteElementSpace*>(fes_u_)) {
    MPI_Allreduce(MPI_IN_PLACE, &weight_order, 1, MPI_INT, MPI_MAX,
                  pfes->GetComm());
  }
#endif
  MFEM_VERIFY(weight_order >= 0,
              "NonlinearSelfGravitatingProblem: the body has no elements.");
  const auto geometry =
      generator == 1 ? (dim == 2 ? Geometry::TRIANGLE : Geometry::TETRAHEDRON)
                     : (dim == 2 ? Geometry::SQUARE : Geometry::CUBE);
  rule_ = &IntRules.Get(
      geometry, 2 * std::max(fes_u_->GetMaxElementOrder(),
                             fes_zeta_->GetMaxElementOrder()) +
                    weight_order);

  internal_ = detail::MakeLinearForm(fes_u_);
  auto* stress = new DomainLFDeformationGradientIntegrator(*stress_c_);
  if (rule_) {
    stress->SetIntRule(rule_);
  }
  internal_->AddDomainIntegrator(stress);
}

void NonlinearSelfGravitatingProblem::SetSurfaceLoad(Coefficient& sigma,
                                                     const Array<int>& marker) {
  MFEM_VERIFY(!sigma_ && !reference_ready_,
              "NonlinearSelfGravitatingProblem::SetSurfaceLoad: set the "
              "load once, before the first evaluation.");
  sigma_ = &sigma;
  surface_marker_ = marker;
}

void NonlinearSelfGravitatingProblem::SetBodyLoad(VectorCoefficient* force,
                                                  MatrixCoefficient* stress) {
  MFEM_VERIFY(!body_force_ && !stress_load_ && !reference_ready_,
              "NonlinearSelfGravitatingProblem::SetBodyLoad: set the load "
              "once, before the first evaluation.");
  body_force_ = force;
  stress_load_ = stress;
}

void NonlinearSelfGravitatingProblem::SetPotentialLoad(
    Coefficient* source, VectorCoefficient* flux) {
  MFEM_VERIFY(!potential_source_ && !potential_flux_ && !reference_ready_,
              "NonlinearSelfGravitatingProblem::SetPotentialLoad: set the "
              "load once, before the first evaluation.");
  potential_source_ = source;
  potential_flux_ = flux;
}

bool NonlinearSelfGravitatingProblem::IsParallel() const {
  return detail::IsParallel(*fes_u_);
}

GridFunction& NonlinearSelfGravitatingProblem::Displacement() {
  EnsureReference();
  return *u_;
}

GridFunction& NonlinearSelfGravitatingProblem::Potential() {
  EnsureReference();
  return *zeta_;
}

const GridFunction& NonlinearSelfGravitatingProblem::ReferencePotential() {
  EnsureReference();
  return *zeta0_;
}

real_t NonlinearSelfGravitatingProblem::Dot(const Vector& a,
                                            const Vector& b) const {
#ifdef MFEM_USE_MPI
  if (auto* pfes = dynamic_cast<ParFiniteElementSpace*>(fes_u_)) {
    return InnerProduct(pfes->GetComm(), a, b);
  }
#endif
  return a * b;
}

real_t NonlinearSelfGravitatingProblem::Norm(const GridFunction& v) const {
  return v.ComputeL2Error(*zero_);
}

void NonlinearSelfGravitatingProblem::SyncFields() {
#ifdef MFEM_USE_MPI
  if (pextension_) {
    auto& u = static_cast<ParGridFunction&>(*u_);
    auto& buffer = static_cast<ParGridFunction&>(*u_buffer_);
    auto& ball = static_cast<ParGridFunction&>(*u_ball_);
    auto on_body = Vector(pextension_->Width());
    auto on_buffer = Vector(pextension_->Height());
    u.GetTrueDofs(on_body);
    pextension_->Mult(on_body, on_buffer);
    buffer.SetFromTrueDofs(on_buffer);
    ball = 0.0;
    ParSubMesh::Transfer(buffer, ball);
    ParSubMesh::Transfer(u, ball);
    return;
  }
#endif
  extension_->Mult(*u_, *u_buffer_);
  *u_ball_ = 0.0;
  SubMesh::Transfer(*u_buffer_, *u_ball_);
  SubMesh::Transfer(*u_, *u_ball_);
}

std::unique_ptr<LinearQuasiStaticReferentialSelfGravitatingProblem>
NonlinearSelfGravitatingProblem::NewLinear(bool reference) {
  auto linear =
      std::make_unique<LinearQuasiStaticReferentialSelfGravitatingProblem>(
          fes_u_, fes_zeta_, *rheology_, *rho_, G_, dtn_degree_,
          reference ? nullptr : zeta_c_.get(), std::vector<FluidRegion>{},
          rule_);
  // One rule for every gravity form, given above, and for the elastic
  // stiffness: the rule of the elastic residual and energy. The operator
  // is then the second variation of the energy as that rule integrates
  // it, and the gravity terms of the residual, read from the operator,
  // its first.
  if (rule_) {
    for (auto* integrator : *linear->StiffnessIntegrators().GetDBFI()) {
      if (dynamic_cast<MaterialStiffnessIntegrator*>(integrator) ||
          dynamic_cast<GeometricStiffnessIntegrator*>(integrator)) {
        integrator->SetIntRule(rule_);
      }
    }
  }
#ifdef MFEM_USE_MPI
  if (pextension_) {
    linear->SetPrescribedVacuumExtension(
        *static_cast<ParFiniteElementSpace*>(fes_buffer_), *pextension_);
  } else
#endif
  {
    linear->SetPrescribedVacuumExtension(*fes_buffer_, *extension_);
  }
  linear->SetRelTol(options_.linear_rel_tol);
  return linear;
}

void NonlinearSelfGravitatingProblem::EnsureReference() {
  if (reference_ready_) {
    return;
  }
  // The reference, with nothing displaced: its discrete potential, the
  // gravity parts of its residual, and the load of the potential's row.
  const auto u_kept = Vector(*u_);
  *u_ = 0.0;
  SyncFields();
  auto reference = NewLinear(true);
  if (sigma_) {
    reference->SetSurfaceLoad(*sigma_, surface_marker_);
  }
  reference->AssembleForce(0.0);
  load_zeta_ = reference->PotentialForce();
  if (potential_source_ || potential_flux_) {
    auto load = detail::MakeLinearForm(fes_zeta_);
    if (potential_source_) {
      load->AddDomainIntegrator(new DomainLFIntegrator(*potential_source_));
    }
    if (potential_flux_) {
      load->AddDomainIntegrator(new DomainLFGradIntegrator(*potential_flux_));
    }
    load->Assemble();
    auto on_true = Vector(fes_zeta_->GetTrueVSize());
    if (const auto* P = fes_zeta_->GetProlongationMatrix()) {
      P->MultTranspose(*load, on_true);
    } else {
      on_true = *load;
    }
    load_zeta_ += on_true;
  }
  load_u_.SetSize(fes_u_->GetTrueVSize());
  load_u_ = 0.0;
  if (body_force_ || stress_load_) {
    auto load = detail::MakeLinearForm(fes_u_);
    if (body_force_) {
      load->AddDomainIntegrator(new VectorDomainLFIntegrator(*body_force_));
    }
    if (stress_load_) {
      load->AddDomainIntegrator(
          new DomainLFDeformationGradientIntegrator(*stress_load_));
    }
    load->Assemble();
    if (const auto* P = fes_u_->GetProlongationMatrix()) {
      P->MultTranspose(*load, load_u_);
    } else {
      load_u_ = *load;
    }
  }
  *zeta0_ = reference->BackgroundPotential();
  *zeta_ = *zeta0_;
  auto zeta0 = Vector(fes_zeta_->GetTrueVSize());
  zeta0_->GetTrueDofs(zeta0);
  reference->ApplyPotentialColumn(zeta0, reference_u_, reference_zeta_);
  reference_u_ *= 0.5;
  *u_ = u_kept;
  reference_ready_ = true;
}

void NonlinearSelfGravitatingProblem::StateResidual(
    LinearQuasiStaticReferentialSelfGravitatingProblem& at, Vector& r_u,
    Vector& r_zeta, Vector* a_zeta) {
  // The gravity terms from the operator at the state: its coupling
  // applied to the potential is twice the first variation of the gravity
  // energy in the displacement, and its potential row applied to the
  // potential the first variation in the potential.
  auto zeta = Vector(fes_zeta_->GetTrueVSize());
  zeta_->GetTrueDofs(zeta);
  auto coupling = Vector();
  at.ApplyPotentialColumn(zeta, coupling, r_zeta);
  if (a_zeta) {
    *a_zeta = r_zeta;
  }
  r_zeta -= reference_zeta_;
  r_zeta -= load_zeta_;

  internal_->Assemble();
  r_u.SetSize(fes_u_->GetTrueVSize());
  if (const auto* P = fes_u_->GetProlongationMatrix()) {
    P->MultTranspose(*internal_, r_u);
  } else {
    r_u = *internal_;
  }
  r_u.Add(0.5, coupling);
  r_u -= reference_u_;
  r_u -= load_u_;
}

void NonlinearSelfGravitatingProblem::Residual(Vector& r_u, Vector& r_zeta) {
  EnsureReference();
  SyncFields();
  auto at = NewLinear(false);
  StateResidual(*at, r_u, r_zeta);
}

real_t NonlinearSelfGravitatingProblem::Energy() {
  EnsureReference();
  SyncFields();
  auto at = NewLinear(false);
  auto r_u = Vector(), r_zeta = Vector(), a_zeta = Vector();
  StateResidual(*at, r_u, r_zeta, &a_zeta);

  auto* mesh = fes_u_->GetMesh();
  auto energy = real_t{0};
  for (auto e = 0; e < mesh->GetNE(); e++) {
    auto* T = mesh->GetElementTransformation(e);
    const auto& rule = *rule_;
    for (auto q = 0; q < rule.GetNPoints(); q++) {
      const auto& ip = rule.IntPoint(q);
      T->SetIntPoint(&ip);
      energy += ip.weight * T->Weight() * energy_c_->Eval(*T, ip);
    }
  }
#ifdef MFEM_USE_MPI
  if (auto* pfes = dynamic_cast<ParFiniteElementSpace*>(fes_u_)) {
    auto local = energy;
    MPI_Allreduce(&local, &energy, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                  pfes->GetComm());
  }
#endif
  auto u = Vector(fes_u_->GetTrueVSize());
  u_->GetTrueDofs(u);
  auto zeta = Vector(fes_zeta_->GetTrueVSize());
  zeta_->GetTrueDofs(zeta);
  return energy + 0.5 * Dot(zeta, a_zeta) - Dot(zeta, reference_zeta_) -
         Dot(zeta, load_zeta_) - Dot(u, reference_u_) - Dot(u, load_u_);
}

LinearQuasiStaticReferentialSelfGravitatingProblem&
NonlinearSelfGravitatingProblem::Tangent() {
  EnsureReference();
  SyncFields();
  tangent_ = NewLinear(false);
  return *tangent_;
}

bool NonlinearSelfGravitatingProblem::Solve() {
  EnsureReference();
  corrections_.clear();
  const auto print = options_.print && Root(*fes_u_);
  auto r_u = Vector(), r_zeta = Vector();
  auto force = Vector(fes_u_->GetVSize());
  // The tangent is made at the first iteration, and at every one when the
  // method is Newton's proper.
  auto newton = options_.refresh_contraction <= 0.0;
  auto refresh = true;
  auto previous = real_t{0};
  for (auto it = 0; it < options_.max_iterations; it++) {
    const auto started = std::chrono::steady_clock::now();
    SyncFields();
    auto at = NewLinear(false);
    StateResidual(*at, r_u, r_zeta);
    const auto assembled = std::chrono::steady_clock::now();
    if (refresh) {
      tangent_ = std::move(at);
      tangents_++;
    }
    r_u.Neg();
    r_zeta.Neg();
    // A dual vector on true dofs in the layout the linear class takes.
    if (const auto* R = fes_u_->GetRestrictionMatrix()) {
      R->MultTranspose(r_u, force);
    } else {
      force = r_u;
    }
    tangent_->ResetSolution();
    tangent_->AssembleForce(0.0);
    tangent_->AddForce(force);
    tangent_->AddPotentialForce(r_zeta);
    if (!tangent_->Solve()) {
      if (print) {
        std::cout << "NonlinearSelfGravitatingProblem: the linear solve of "
                     "iteration "
                  << it + 1 << " did not converge.\n";
      }
      return false;
    }
    const auto solved = std::chrono::steady_clock::now();
    const auto seconds = [](auto from, auto to) {
      return std::chrono::duration<double>(to - from).count();
    };
    const auto& correction = tangent_->Displacement();
    const auto size = Norm(correction);
    if (!refresh && size > options_.refresh_contraction * previous) {
      // The kept tangent has stopped serving: its correction is not
      // taken, and the rest of the solve is Newton's method proper.
      if (print) {
        std::cout << "  Newton " << std::setw(2) << it + 1 << ": correction "
                  << std::scientific << std::setprecision(3) << size
                  << " of the kept tangent not taken\n"
                  << std::defaultfloat;
      }
      newton = true;
      refresh = true;
      continue;
    }
    *u_ += correction;
    *zeta_ += tangent_->Potential();
    const auto scale = Norm(*u_);
    corrections_.push_back(size);
    if (print) {
      std::cout << "  Newton " << std::setw(2) << it + 1 << ": correction "
                << std::scientific << std::setprecision(3) << size
                << ", displacement " << scale
                << (refresh ? ", tangent made" : "") << std::fixed
                << std::setprecision(2) << "; residual "
                << seconds(started, assembled) << " s, solve "
                << seconds(assembled, solved) << " s ("
                << tangent_->LastOuterIterations() << " iterations)\n"
                << std::defaultfloat;
    }
    if (size <= std::max(options_.rel_tol * scale, options_.abs_tol)) {
      SyncFields();
      return true;
    }
    refresh = newton;
    previous = size;
  }
  return false;
}

}  // namespace AdGIA
