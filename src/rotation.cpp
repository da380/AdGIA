/**
 * @file rotation.cpp
 * @brief Implementation of RotationalFeedback.
 */

#include "AdGIA/rotation.hpp"

#include "AdGIA/detail/fem_factory.hpp"

namespace AdGIA {

using namespace mfem;

RotationalFeedback::RotationalFeedback(
    LinearQuasiStaticMixedSelfGravitatingProblem& problem, real_t Omega,
    const Vector& principal_moments)
    : problem_(problem), Omega_(Omega), psi_(0, 0.0) {
  FiniteElementSpace& fes = problem_.DisplacementSpace();
  dim_ = fes.GetMesh()->Dimension();
  nw_ = dim_ == 2 ? 1 : 3;
  MFEM_VERIFY(principal_moments.Size() == nw_,
              "RotationalFeedback: " << nw_ << " principal moment(s) "
              "expected ([C3] in 2-D, [C1, C2, C3] in 3-D).");
  // D = diag(C3 - C1, C3 - C2, -C3) (eq. 16); its 2-D reduction is the
  // spin row -C3 alone.
  D_.SetSize(nw_);
  D_ = 0.0;
  if (dim_ == 2) {
    D_(0, 0) = -principal_moments[0];
  } else {
    const real_t c1 = principal_moments[0], c2 = principal_moments[1],
                 c3 = principal_moments[2];
    D_(0, 0) = c3 - c1;
    D_(1, 1) = c3 - c2;
    D_(2, 2) = -c3;
  }
  psi_ = CentrifugalPotential(dim_, Omega_);
  omega_.SetSize(nw_);
  omega_ = 0.0;
  for (int k = 0; k < nw_; k++) {
    psi_unit_.push_back(std::make_unique<CentrifugalPotential>(dim_, Omega_));
    psi_unit_.back()->SetUnit(k);
  }
  problem_.SetTidalPotential(psi_);

  sfec_ = std::make_unique<H1_FECollection>(fes.GetMaxElementOrder(), dim_);
  sfes_ = detail::MakeFESpace(fes, sfec_.get());
  parallel_ = detail::IsParallel(fes);
#ifdef MFEM_USE_MPI
  if (parallel_) {
    comm_ = static_cast<ParFiniteElementSpace&>(fes).GetComm();
  }
#endif
}

real_t RotationalFeedback::GlobalSum(real_t v) const {
#ifdef MFEM_USE_MPI
  if (parallel_) {
    real_t g = 0.0;
    MPI_Allreduce(&v, &g, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM, comm_);
    return g;
  }
#endif
  return v;
}

real_t RotationalFeedback::SurfacePairing(Coefficient& sigma,
                                          const Array<int>& marker, int k) {
  ProductCoefficient sp(sigma, *psi_unit_[k]);
  auto lf = detail::MakeLinearForm(sfes_.get());
  lf->AddBoundaryIntegrator(new BoundaryLFIntegrator(sp),
                            const_cast<Array<int>&>(marker));
  lf->Assemble();
  return GlobalSum(lf->Sum());
}

void RotationalFeedback::EnsureResponses(real_t t) {
  if (cached_) {
    return;
  }
  // Base state X (zero centrifugal amplitude), its pairings a_k, then
  // the unit responses by linearity: Y_j = Z_j - X with Z_j the solve
  // at unit amplitude; only the pairings are stored.
  psi_.SetZero();
  problem_.AssembleForce(t);
  MFEM_VERIFY(problem_.Solve(),
              "RotationalFeedback: the base solve failed.");
  Vector a(nw_);
  for (int k = 0; k < nw_; k++) {
    a[k] = problem_.TidalCoupling(*psi_unit_[k]);
  }
  S_.SetSize(nw_);
  for (int j = 0; j < nw_; j++) {
    psi_.SetUnit(j);
    problem_.AssembleForce(t);
    MFEM_VERIFY(problem_.Solve(),
                "RotationalFeedback: unit response solve failed.");
    for (int k = 0; k < nw_; k++) {
      // T(psi_k, Y_j) = T(psi_k, Z_j) - T(psi_k, X).
      S_(k, j) = problem_.TidalCoupling(*psi_unit_[k]) - a[k];
    }
  }
  // S = D + P + T(psi_k, Y_j), with P the fluid psi-psi block.
  for (int k = 0; k < nw_; k++) {
    for (int j = 0; j < nw_; j++) {
      S_(k, j) += D_(k, j) +
                  problem_.TidalTidalCoupling(*psi_unit_[k], *psi_unit_[j]);
    }
  }
  cached_ = true;
}

bool RotationalFeedback::Solve(real_t t, Coefficient& sigma,
                               const Array<int>& surface_marker) {
  EnsureResponses(t);
  // Base solve at this time/load.
  psi_.SetZero();
  problem_.AssembleForce(t);
  if (!problem_.Solve()) {
    return false;
  }
  // The border row: S omega = -(T(psi_k, X) + int sigma psi_k dS).
  Vector rhs(nw_);
  for (int k = 0; k < nw_; k++) {
    rhs[k] = -(problem_.TidalCoupling(*psi_unit_[k]) +
               SurfacePairing(sigma, surface_marker, k));
  }
  DenseMatrixInverse Sinv(S_);
  Sinv.Mult(rhs, omega_);
  // The composed final solve leaves the problem at the full solution.
  psi_.SetAmplitudes(omega_);
  problem_.AssembleForce(t);
  return problem_.Solve();
}

real_t RotationalFeedback::ConsistencyResidual(
    Coefficient& sigma, const Array<int>& surface_marker) {
  // r_k = T(psi_k, x_final) + ((D + P) omega)_k + int sigma psi_k dS,
  // evaluated afresh at the problem's current state.
  real_t rmax = 0.0, scale = 0.0;
  for (int k = 0; k < nw_; k++) {
    const real_t tk = problem_.TidalCoupling(*psi_unit_[k]);
    real_t dp = 0.0;
    for (int j = 0; j < nw_; j++) {
      dp += (D_(k, j) +
             problem_.TidalTidalCoupling(*psi_unit_[k], *psi_unit_[j])) *
            omega_[j];
    }
    const real_t sk = SurfacePairing(sigma, surface_marker, k);
    rmax = std::max(rmax, std::abs(tk + dp + sk));
    scale = std::max({scale, std::abs(tk), std::abs(dp), std::abs(sk)});
  }
  return rmax / (scale + 1e-300);
}

}  // namespace AdGIA
