/**
 * @file nonlinear_problem.cpp
 * @brief Implementation of nonlinear_problem.hpp.
 */

#include "AdGIA/nonlinear_problem.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>

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
    VectorCoefficient* dirichlet)
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
  map_ = std::make_unique<GridFunctionDiffeomorphism>(*u_);
  energy_c_ =
      std::make_unique<StrainEnergyDensityCoefficient>(law, *map_, true);
  stress_c_ =
      std::make_unique<FirstPiolaKirchhoffStressCoefficient>(law, *map_, true);
  tangent_stress_c_ =
      std::make_unique<SecondPiolaKirchhoffStressCoefficient>(law, *map_);
  tangent_c_ = std::make_unique<MaterialTangentCoefficient>(law, *map_);
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

}  // namespace AdGIA
