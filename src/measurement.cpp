#include "AdGIA/measurement.hpp"

#include <cmath>

#include "AdGIA/centrifugal.hpp"
#include "AdGIA/detail/fem_factory.hpp"
#include "AdGIA/mixed_problem.hpp"
#include "AdGIA/spherical_harmonics.hpp"

using namespace mfem;

namespace AdGIA {

namespace {

// <dual, u> for an L-vector dual (the rank's element-assembled partial
// sums) and a consistent u: through the prolongation transpose so that a
// shared dof counts once (TangentLinearSolver::Dot).
real_t DotGlobal(const Vector& dual, const GridFunction& u,
                 FiniteElementSpace& fes) {
#ifdef MFEM_USE_MPI
  if (auto* pfes = dynamic_cast<ParFiniteElementSpace*>(&fes)) {
    Vector dt(pfes->GetTrueVSize()), ut(pfes->GetTrueVSize());
    pfes->GetProlongationMatrix()->MultTranspose(dual, dt);
    u.GetTrueDofs(ut);
    return InnerProduct(pfes->GetComm(), dt, ut);
  }
#endif
  return dual * u;
}

// The sum over ranks of a scalar built from element-local sums.
real_t SumOverRanks(real_t v, FiniteElementSpace& fes) {
#ifdef MFEM_USE_MPI
  if (auto* pfes = dynamic_cast<ParFiniteElementSpace*>(&fes)) {
    MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                  pfes->GetComm());
  }
#else
  (void)fes;
#endif
  return v;
}

// int_S f dS over the marked boundary of fes's mesh: the Sum() of the
// element-assembled load vector counts every element once (the H1
// partition of unity), so the global sum over ranks is the integral.
real_t BoundaryIntegral(Coefficient& f, FiniteElementSpace& fes,
                        const Array<int>& marker) {
  auto lf = detail::MakeLinearForm(&fes);
  lf->AddBoundaryIntegrator(new BoundaryLFIntegrator(f),
                            const_cast<Array<int>&>(marker));
  lf->Assemble();
  return SumOverRanks(lf->Sum(), fes);
}

// |v(x)| of a vector coefficient.
class MagnitudeCoefficient : public Coefficient {
 public:
  explicit MagnitudeCoefficient(VectorCoefficient& v) : v_(v) {}
  real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    v_.Eval(x_, T, ip);
    return x_.Norml2();
  }

 private:
  VectorCoefficient& v_;
  Vector x_;
};

LinearQuasiStaticMixedSelfGravitatingProblem* Gravitating(
    LinearQuasiStaticProblem& problem) {
  return dynamic_cast<LinearQuasiStaticMixedSelfGravitatingProblem*>(&problem);
}

}  // namespace

// --- MeasurementDual, Evaluate ----------------------------------------------------

AdjointLoad MeasurementDual::Load() const {
  AdjointLoad load;
  if (u.Size() > 0) load.u = &u;
  if (phi.Size() > 0) load.phi = &phi;
  load.uniform = uniform;
  if (omega.Size() > 0) load.omega = &omega;
  return load;
}

real_t Evaluate(const MeasurementDual& dual, LinearQuasiStaticProblem& problem) {
  real_t J = 0.0;
  if (dual.u.Size() > 0) {
    J += DotGlobal(dual.u, problem.Displacement(), problem.DisplacementSpace());
  }
  if (dual.OnPotential() || dual.OnBorder()) {
    auto* g = Gravitating(problem);
    MFEM_VERIFY(g, "Evaluate: a potential or border dual on a problem "
                   "without them.");
    if (dual.OnPotential()) {
      J += DotGlobal(dual.phi, g->Potential(), g->PotentialForceSpace());
    }
    J += dual.uniform * g->UniformPotentialTerm();
    if (dual.omega.Size() > 0) {
      MFEM_VERIFY(g->AngularVelocity().Size() == dual.omega.Size(),
                  "Evaluate: an omega dual on a problem without rotation.");
      J += dual.omega * g->AngularVelocity();
    }
  }
  return J;
}

// --- window helpers ----------------------------------------------------------

CapWindowCoefficient::CapWindowCoefficient(const Vector& direction,
                                           real_t width, real_t amplitude,
                                           const Vector& centre)
    : n_(direction), x0_(centre), width2_(width * width),
      amplitude_(amplitude) {
  n_ /= n_.Norml2();
  if (x0_.Size() == 0) {
    x0_.SetSize(n_.Size());
    x0_ = 0.0;
  }
}

real_t CapWindowCoefficient::Eval(ElementTransformation& T,
                                  const IntegrationPoint& ip) {
  T.Transform(ip, x_);
  d_ = x_;
  d_ -= x0_;
  d_ /= d_.Norml2();
  d_ -= n_;
  return amplitude_ * std::exp(-(d_ * d_) / width2_);
}

RadialVectorCoefficient::RadialVectorCoefficient(int dim, Coefficient& window,
                                                 const Vector& centre)
    : VectorCoefficient(dim), w_(window), x0_(centre) {
  if (x0_.Size() == 0) {
    x0_.SetSize(dim);
    x0_ = 0.0;
  }
}

void RadialVectorCoefficient::Eval(Vector& v, ElementTransformation& T,
                                   const IntegrationPoint& ip) {
  T.Transform(ip, v);
  v -= x0_;
  v *= w_.Eval(T, ip) / v.Norml2();
}

// --- surface measurements ------------------------------------------------------

SurfaceDisplacementMeasurement::SurfaceDisplacementMeasurement(
    VectorCoefficient& weight, const Array<int>& marker, std::string name)
    : w_(weight), marker_(marker), name_(std::move(name)) {}

void SurfaceDisplacementMeasurement::Assemble(LinearQuasiStaticProblem& problem,
                                              MeasurementDual& dual) const {
  auto lf = detail::MakeLinearForm(&problem.DisplacementSpace());
  lf->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(w_),
                            const_cast<Array<int>&>(marker_));
  lf->Assemble();
  dual.u = *lf;
}

SurfacePotentialMeasurement::SurfacePotentialMeasurement(
    Coefficient& weight, const Array<int>& marker, std::string name)
    : w_(weight), marker_(marker), name_(std::move(name)) {}

void SurfacePotentialMeasurement::Assemble(LinearQuasiStaticProblem& problem,
                                           MeasurementDual& dual) const {
  auto* slot = dynamic_cast<PotentialForceSlot*>(&problem);
  MFEM_VERIFY(slot, "SurfacePotentialMeasurement: the problem has no "
                    "potential rows.");
  auto lf = detail::MakeLinearForm(&slot->PotentialForceSpace());
  lf->AddBoundaryIntegrator(new BoundaryLFIntegrator(w_),
                            const_cast<Array<int>&>(marker_));
  lf->Assemble();
  dual.phi = *lf;
}

// --- harmonic coefficients -------------------------------------------------------

HarmonicCoefficientMeasurement::HarmonicCoefficientMeasurement(
    Field field, const Array<int>& marker, int degree, int order,
    const Vector& centre, real_t radius_tolerance)
    : field_(field), marker_(marker), l_(degree), m_(order), centre_(centre),
      radius_tolerance_(radius_tolerance) {}

std::string HarmonicCoefficientMeasurement::Name() const {
  return std::string(field_ == Field::Potential ? "potential" : "radial u") +
         " harmonic (" + std::to_string(l_) + ", " + std::to_string(m_) + ")";
}

void HarmonicCoefficientMeasurement::Assemble(LinearQuasiStaticProblem& problem,
                                              MeasurementDual& dual) const {
  using BHC = BoundaryHarmonicCoefficients;
  if (field_ == Field::RadialDisplacement) {
    BHC bhc(problem.DisplacementSpace(), marker_, l_, BHC::Component::Radial,
            centre_, radius_tolerance_);
    Vector c(bhc.Size());
    c = 0.0;
    c[bhc.Basis().Index(l_, m_)] = 1.0;
    bhc.LoadVector(c, dual.u);
  } else {
    auto* slot = dynamic_cast<PotentialForceSlot*>(&problem);
    MFEM_VERIFY(slot, "HarmonicCoefficientMeasurement: the problem has no "
                      "potential rows.");
    BHC bhc(slot->PotentialForceSpace(), marker_, l_, BHC::Component::Scalar,
            centre_, radius_tolerance_);
    Vector c(bhc.Size());
    c = 0.0;
    c[bhc.Basis().Index(l_, m_)] = 1.0;
    bhc.LoadVector(c, dual.phi);
  }
}

// --- rotation ---------------------------------------------------------------------

RotationMeasurement::RotationMeasurement(Kind kind, int component)
    : kind_(kind), k_(component) {}

std::string RotationMeasurement::Name() const {
  switch (kind_) {
    case Kind::Component:
      return "omega_" + std::to_string(k_);
    case Kind::PolarMotion:
      return "polar motion m_" + std::to_string(k_);
    case Kind::LengthOfDay:
      return "length-of-day change";
  }
  return "rotation";
}

void RotationMeasurement::Assemble(LinearQuasiStaticProblem& problem,
                                   MeasurementDual& dual) const {
  auto* g = Gravitating(problem);
  MFEM_VERIFY(g && g->AngularVelocity().Size() > 0 && g->RotationRate() != 0.0,
              "RotationMeasurement: the problem is not a rotating mixed "
              "problem.");
  const int nc = g->AngularVelocity().Size();
  dual.omega.SetSize(nc);
  dual.omega = 0.0;
  switch (kind_) {
    case Kind::Component:
      MFEM_VERIFY(k_ >= 0 && k_ < nc, "RotationMeasurement: component.");
      dual.omega[k_] = 1.0;
      break;
    case Kind::PolarMotion:
      MFEM_VERIFY(nc == 3 && (k_ == 0 || k_ == 1),
                  "RotationMeasurement: polar motion needs 3-D, k = 0 or 1.");
      dual.omega[k_] = 1.0 / g->RotationRate();
      break;
    case Kind::LengthOfDay:
      dual.omega[nc - 1] = -1.0 / g->RotationRate();
      break;
  }
}

// --- sea level ---------------------------------------------------------------------

SeaLevelMeasurement::SeaLevelMeasurement(Coefficient& window,
                                         const Array<int>& surface,
                                         const Array<int>& parent_surface,
                                         std::string name)
    : h_(window), surface_(surface), parent_surface_(parent_surface),
      name_(std::move(name)) {}

void SeaLevelMeasurement::Assemble(LinearQuasiStaticProblem& problem,
                                   MeasurementDual& dual) const {
  auto* g = Gravitating(problem);
  MFEM_VERIFY(g, "SeaLevelMeasurement: the problem is not a mixed "
                 "self-gravitating problem.");
  // On the body: -(h/g) grad Phi0 against the displacement.
  {
    MagnitudeCoefficient gmag(g->BackgroundGravity());
    RatioCoefficient h_over_g(h_, gmag);
    ScalarVectorProductCoefficient w(h_over_g, g->BackgroundGravity());
    auto lf = detail::MakeLinearForm(&problem.DisplacementSpace());
    lf->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(w),
                              const_cast<Array<int>&>(surface_));
    lf->Assemble();
    dual.u = *lf;
    dual.u *= -1.0;
  }
  // The potential rows, the uniform row and the omega rows are
  // integrals over the same surface weighted by h/g; g is the body's
  // discrete |grad Phi0|, which is evaluable on the body's boundary but
  // not on the parent mesh's interior faces. So they are assembled on a
  // scalar space of the body (the potential space's collection) and the
  // phi dual is carried to the potential space node by node: the surface
  // nodes are the same nodes in both spaces, and the integrals attach to
  // them.
  {
    FiniteElementSpace& fes_phi = g->PotentialForceSpace();
    auto body_sfes = detail::MakeFESpace(
        problem.DisplacementSpace(),
        const_cast<FiniteElementCollection*>(fes_phi.FEColl()), 1);
    MagnitudeCoefficient gmag(g->BackgroundGravity());
    RatioCoefficient h_over_g(h_, gmag);
    auto lf = detail::MakeLinearForm(body_sfes.get());
    lf->AddBoundaryIntegrator(new BoundaryLFIntegrator(h_over_g),
                              const_cast<Array<int>&>(surface_));
    lf->Assemble();
    dual.uniform = SumOverRanks(lf->Sum(), *body_sfes);
    auto on_body = detail::MakeGridFunction(body_sfes.get());
    *on_body = *lf;
    *on_body *= -1.0;
    auto on_parent = detail::MakeGridFunction(&fes_phi);
    *on_parent = 0.0;
    Mesh* body_mesh = problem.DisplacementSpace().GetMesh();
#ifdef MFEM_USE_MPI
    if (dynamic_cast<ParSubMesh*>(body_mesh)) {
      ParSubMesh::Transfer(static_cast<ParGridFunction&>(*on_body),
                           static_cast<ParGridFunction&>(*on_parent));
    } else
#endif
    {
      MFEM_VERIFY(dynamic_cast<SubMesh*>(body_mesh),
                  "SeaLevelMeasurement: the body is not a SubMesh of the "
                  "potential mesh.");
      SubMesh::Transfer(*on_body, *on_parent);
    }
    dual.phi = *on_parent;
    const int nc = g->AngularVelocity().Size();
    if (nc > 0) {
      const int dim = body_mesh->Dimension();
      dual.omega.SetSize(nc);
      for (int k = 0; k < nc; k++) {
        CentrifugalPotential psi_k(dim, g->RotationRate());
        psi_k.SetUnit(k);
        ProductCoefficient hpsi(h_over_g, psi_k);
        dual.omega[k] = -BoundaryIntegral(hpsi, *body_sfes, surface_);
      }
    }
  }
}

// --- sets ----------------------------------------------------------------------------

int MeasurementSet::Add(const Measurement& m, real_t t) {
  obs_.push_back({&m, t, nullptr});
  problem_ = nullptr;
  return Size() - 1;
}

int MeasurementSet::AddIntegrated(const Measurement& m,
                                  std::function<real_t(real_t)> weight) {
  MFEM_VERIFY(weight, "MeasurementSet::AddIntegrated: a weight is needed.");
  obs_.push_back({&m, 0.0, std::move(weight)});
  problem_ = nullptr;
  return Size() - 1;
}

void MeasurementSet::Assemble(LinearQuasiStaticProblem& problem) {
  duals_.assign(obs_.size(), MeasurementDual());
  for (int i = 0; i < Size(); i++) {
    obs_[i].m->Assemble(problem, duals_[i]);
    if (obs_[i].w) {
      MFEM_VERIFY(!duals_[i].OnBorder(),
                  "MeasurementSet: a time-integrated observation on the "
                  "border rows (" + obs_[i].m->Name() + ") is not served.");
    }
  }
  problem_ = &problem;
}

void MeasurementSet::Data(ForwardTrajectory& trajectory, Vector& d) {
  MFEM_VERIFY(problem_, "MeasurementSet: Assemble() first.");
  LinearQuasiStaticProblem& problem = trajectory.Op().Problem();
  MFEM_VERIFY(&problem == problem_ ||
                  &problem.DisplacementSpace() == &problem_->DisplacementSpace(),
              "MeasurementSet::Data: the trajectory's problem is on other "
              "spaces than the assembled duals.");
  const int N = trajectory.Size() - 1;
  const real_t span = trajectory.Time(N) - trajectory.Time(0);
  const real_t tol = 1e-8 * (span > 0.0 ? span : 1.0);
  d.SetSize(Size());
  d = 0.0;
  // Node by node: one re-solve per node that any observation needs.
  for (int n = 0; n <= N; n++) {
    const real_t t = trajectory.Time(n);
    const real_t wR = n < N ? 0.5 * (trajectory.Time(n + 1) - t) : 0.0;
    const real_t wL = n > 0 ? 0.5 * (t - trajectory.Time(n - 1)) : 0.0;
    bool needed = false;
    for (int i = 0; i < Size() && !needed; i++) {
      needed = obs_[i].w ? true : std::abs(obs_[i].t - t) <= tol;
    }
    if (!needed) continue;
    MFEM_VERIFY(trajectory.Op().SolveElastic(trajectory.State(n), t),
                "MeasurementSet::Data: the forward re-solve failed.");
    for (int i = 0; i < Size(); i++) {
      if (obs_[i].w) {
        d[i] += (wL + wR) * obs_[i].w(t) * Evaluate(duals_[i], problem);
      } else if (std::abs(obs_[i].t - t) <= tol) {
        d[i] = Evaluate(duals_[i], problem);
      }
    }
  }
  for (int i = 0; i < Size(); i++) {
    if (obs_[i].w) continue;
    bool found = false;
    for (int n = 0; n <= N && !found; n++) {
      found = std::abs(obs_[i].t - trajectory.Time(n)) <= tol;
    }
    MFEM_VERIFY(found, "MeasurementSet::Data: observation " +
                           obs_[i].m->Name() + " is off the grid.");
  }
}

void MeasurementSet::Data(LinearQuasiStaticProblem& problem, Vector& d) {
  MFEM_VERIFY(problem_, "MeasurementSet: Assemble() first.");
  d.SetSize(Size());
  for (int i = 0; i < Size(); i++) {
    MFEM_VERIFY(!obs_[i].w, "MeasurementSet::Data: a time-integrated "
                            "observation on a static state.");
    d[i] = Evaluate(duals_[i], problem);
  }
}

AdjointForce MeasurementSet::Force(const Vector* c) const {
  MFEM_VERIFY(problem_, "MeasurementSet: Assemble() first.");
  MFEM_VERIFY(!c || c->Size() == Size(), "MeasurementSet::Force: c's size.");
  auto coef = [&](int i) { return c ? (*c)[i] : 1.0; };
  AdjointForce force;
  // Point terms, merged by time into one load each: the merged duals
  // are owned by the force's copies, so scratch here is enough.
  std::vector<bool> done(obs_.size(), false);
  for (int i = 0; i < Size(); i++) {
    if (obs_[i].w || done[i]) continue;
    MeasurementDual sum;
    auto add_into = [&](Vector& acc, const Vector& v, real_t a) {
      if (v.Size() == 0 || a == 0.0) return;
      if (acc.Size() == 0) {
        acc.SetSize(v.Size());
        acc = 0.0;
      }
      acc.Add(a, v);
    };
    for (int j = i; j < Size(); j++) {
      if (obs_[j].w || done[j] || obs_[j].t != obs_[i].t) continue;
      done[j] = true;
      const real_t a = coef(j);
      add_into(sum.u, duals_[j].u, a);
      add_into(sum.phi, duals_[j].phi, a);
      add_into(sum.omega, duals_[j].omega, a);
      sum.uniform += a * duals_[j].uniform;
    }
    force.AddObservation(obs_[i].t, sum.Load());
  }
  // Integrated terms: continuous fillers referring to this set's duals.
  for (int i = 0; i < Size(); i++) {
    if (!obs_[i].w) continue;
    const real_t a = coef(i);
    if (a == 0.0) continue;
    const MeasurementDual* dual = &duals_[i];
    const auto w = obs_[i].w;
    if (dual->u.Size() > 0) {
      force.AddContinuous([dual, w, a](real_t t, Vector& f) {
        f.Add(a * w(t), dual->u);
      });
    }
    if (dual->phi.Size() > 0) {
      force.AddContinuousPotential([dual, w, a](real_t t, Vector& f) {
        f.Add(a * w(t), dual->phi);
      });
    }
  }
  return force;
}

}  // namespace AdGIA
