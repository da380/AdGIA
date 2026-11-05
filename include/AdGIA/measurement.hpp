/**
 * @file measurement.hpp
 * @brief Observables of the quasi-static problems as measurements: each
 * yields its value on a solved state and its dual (the AdjointLoad of
 * adjoint.hpp), so that the data vector of a forward run and the adjoint
 * force of a misfit come from one definition.
 *
 * A Measurement is a linear functional J of the state (u, phi, Phi_g,
 * omega). Assemble() builds its dual once on a problem's spaces; the
 * value is the pairing of that dual with the state (Evaluate()), the same
 * pairing the tangent-linear solver uses, so J(u) and <dJ, du> are
 * consistent to rounding. A MeasurementSet orders observations in time,
 * gives the data vector of a trajectory (one elastic re-solve per
 * observation node) and the AdjointForce of any weighted sum of its
 * observations; time-integrated observations enter as continuous terms
 * with the forward grid's trapezoid weights, which is exactly how the
 * tangent and the discrete adjoints sample them.
 *
 * Built-in measurements: a weighted surface integral of the displacement
 * (stations, caps, components: SurfaceDisplacementMeasurement) and of the
 * potential (SurfacePotentialMeasurement); a surface-harmonic coefficient
 * of the radial displacement or of the potential
 * (HarmonicCoefficientMeasurement); the angular velocity, polar motion
 * and length-of-day change (RotationMeasurement); and the sea-level
 * change weighted by a surface window (SeaLevelMeasurement, all four
 * slots). The window helpers CapWindowCoefficient and
 * RadialVectorCoefficient build the usual station and cap weights.
 */

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "AdGIA/adjoint.hpp"
#include "AdGIA/quasi_static_problem.hpp"
#include "mfem.hpp"

namespace AdGIA {

/** @brief The assembled dual of one measurement: owned slot data, empty
 * where the slot is unused (AdjointLoad's layouts). */
struct MeasurementDual {
  mfem::Vector u, phi, omega;
  mfem::real_t uniform = 0.0;
  bool OnPotential() const { return phi.Size() > 0; }
  bool OnBorder() const { return uniform != 0.0 || omega.Size() > 0; }
  /** @brief A view of the dual as an AdjointLoad (valid while this
   * object lives and is not resized). */
  AdjointLoad Load() const;
};

/** @brief A linear observable of the state. */
class Measurement {
 public:
  virtual ~Measurement() = default;
  virtual std::string Name() const = 0;
  /** @brief Build the dual on @p problem's spaces (the problem's mesh,
   * spaces and background are read; its state is not). */
  virtual void Assemble(LinearQuasiStaticProblem& problem,
                        MeasurementDual& dual) const = 0;
};

/** @brief The value of a dual on @p problem's current state: the
 * displacement rows, and, on a gravitating problem, the potential rows
 * and the border scalars. Global in parallel. */
mfem::real_t Evaluate(const MeasurementDual& dual,
                      LinearQuasiStaticProblem& problem);

// --- window helpers ----------------------------------------------------------

/** @brief A Gaussian cap on the unit sphere of directions:
 * amplitude * exp(-|x_hat - n|^2 / width^2) with x_hat the unit vector
 * from @p centre (default the origin). */
class CapWindowCoefficient : public mfem::Coefficient {
 public:
  CapWindowCoefficient(const mfem::Vector& direction, mfem::real_t width,
                       mfem::real_t amplitude = 1.0,
                       const mfem::Vector& centre = mfem::Vector());
  mfem::real_t Eval(mfem::ElementTransformation& T,
                    const mfem::IntegrationPoint& ip) override;

 private:
  mfem::Vector n_, x0_, x_, d_;
  mfem::real_t width2_, amplitude_;
};

/** @brief w(x) x_hat: a scalar window times the unit radial vector about
 * @p centre (default the origin); the weight of a radial station or cap. */
class RadialVectorCoefficient : public mfem::VectorCoefficient {
 public:
  RadialVectorCoefficient(int dim, mfem::Coefficient& window,
                          const mfem::Vector& centre = mfem::Vector());
  void Eval(mfem::Vector& v, mfem::ElementTransformation& T,
            const mfem::IntegrationPoint& ip) override;

 private:
  mfem::Coefficient& w_;
  mfem::Vector x0_;
};

// --- measurements -------------------------------------------------------------

/** @brief J = int_S w . u dS over the boundary attributes of @p marker
 * (sized to the body mesh's bdr_attributes.Max()). */
class SurfaceDisplacementMeasurement : public Measurement {
 public:
  SurfaceDisplacementMeasurement(mfem::VectorCoefficient& weight,
                                 const mfem::Array<int>& marker,
                                 std::string name = "surface displacement");
  std::string Name() const override { return name_; }
  void Assemble(LinearQuasiStaticProblem& problem,
                MeasurementDual& dual) const override;

 private:
  mfem::VectorCoefficient& w_;
  mfem::Array<int> marker_;
  std::string name_;
};

/** @brief J = int_S w phi dS on the potential rows (PotentialForceSlot):
 * @p marker is sized to the POTENTIAL space's mesh (the mixed class's
 * parent mesh). */
class SurfacePotentialMeasurement : public Measurement {
 public:
  SurfacePotentialMeasurement(mfem::Coefficient& weight,
                              const mfem::Array<int>& marker,
                              std::string name = "surface potential");
  std::string Name() const override { return name_; }
  void Assemble(LinearQuasiStaticProblem& problem,
                MeasurementDual& dual) const override;

 private:
  mfem::Coefficient& w_;
  mfem::Array<int> marker_;
  std::string name_;
};

/** @brief One surface-harmonic coefficient (degree, order; the
 * SurfaceHarmonics basis of spherical_harmonics.hpp) of the radial
 * displacement on the body's spherical surface, or of the potential on
 * the potential space's spherical boundary. The boundary must be a
 * sphere about @p centre (BoundaryHarmonicCoefficients measures and
 * checks the radius). */
class HarmonicCoefficientMeasurement : public Measurement {
 public:
  enum class Field { RadialDisplacement, Potential };
  HarmonicCoefficientMeasurement(Field field, const mfem::Array<int>& marker,
                                 int degree, int order,
                                 const mfem::Vector& centre = mfem::Vector(),
                                 mfem::real_t radius_tolerance = 1e-2);
  std::string Name() const override;
  void Assemble(LinearQuasiStaticProblem& problem,
                MeasurementDual& dual) const override;
  int Degree() const { return l_; }
  int Order() const { return m_; }

 private:
  Field field_;
  mfem::Array<int> marker_;
  int l_, m_;
  mfem::Vector centre_;
  mfem::real_t radius_tolerance_;
};

/** @brief The angular-velocity perturbation of a rotating mixed problem
 * (BorderForceSlot): a component omega_k (Component), polar motion
 * m_k = omega_k / Omega for k = 0, 1 (PolarMotion), or the relative
 * length-of-day change -omega_3 / Omega (LengthOfDay; in 2-D the single
 * component). */
class RotationMeasurement : public Measurement {
 public:
  enum class Kind { Component, PolarMotion, LengthOfDay };
  explicit RotationMeasurement(Kind kind, int component = 0);
  std::string Name() const override;
  void Assemble(LinearQuasiStaticProblem& problem,
                MeasurementDual& dual) const override;

 private:
  Kind kind_;
  int k_;
};

/** @brief J = int_S h SL_1 dS: the sea-level change
 * SL_1 = -(u . grad Phi0 + phi + psi(omega)) / g + Phi_g / g of a mixed
 * problem with the water load, weighted by the surface window @p h
 * (a site, a cap, or the ocean function for the mean). Fills all four
 * slots: u = -(h/g) grad Phi0, phi = -h/g, uniform = int h/g dS,
 * omega_k = -int h psi_k / g dS (psi_k the unit centrifugal potentials;
 * omitted without rotation). @p surface marks the body's surface and
 * @p parent_surface the same surface on the potential space's mesh. */
class SeaLevelMeasurement : public Measurement {
 public:
  SeaLevelMeasurement(mfem::Coefficient& window,
                      const mfem::Array<int>& surface,
                      const mfem::Array<int>& parent_surface,
                      std::string name = "sea level");
  std::string Name() const override { return name_; }
  void Assemble(LinearQuasiStaticProblem& problem,
                MeasurementDual& dual) const override;

 private:
  mfem::Coefficient& h_;
  mfem::Array<int> surface_, parent_surface_;
  std::string name_;
};

// --- sets ----------------------------------------------------------------------

/**
 * @brief An ordered set of observations: measurements at forward times
 * (point terms) and time-integrated measurements int w(t) J(t) dt
 * (continuous terms; displacement and potential rows only). Assemble()
 * builds every dual on the problem; Data() evaluates the set on a
 * trajectory or a static state; Force() is the AdjointForce of
 * sum_i c_i J_i for coefficients c (the misfit's residual, a unit vector
 * for one kernel, ...), point terms at the same time merged into one
 * observation.
 */
class MeasurementSet {
 public:
  /** @brief A point observation of @p m at forward time @p t. Returns
   * its index. */
  int Add(const Measurement& m, mfem::real_t t);
  /** @brief A time-integrated observation int w(t) J_m(t) dt, sampled
   * with the forward grid's trapezoid weights. Returns its index. */
  int AddIntegrated(const Measurement& m,
                    std::function<mfem::real_t(mfem::real_t)> weight);

  int Size() const { return static_cast<int>(obs_.size()); }
  const Measurement& Of(int i) const { return *obs_[i].m; }
  bool IsIntegrated(int i) const { return static_cast<bool>(obs_[i].w); }
  mfem::real_t Time(int i) const { return obs_[i].t; }

  /** @brief Build the duals on @p problem (once; Data() and Force() need
   * it). */
  void Assemble(LinearQuasiStaticProblem& problem);
  const MeasurementDual& Dual(int i) const { return duals_[i]; }

  /** @brief The data vector on a forward trajectory: every point term at
   * its node, every integrated term by the trapezoid rule over the grid.
   * The trajectory's problem is re-solved at each node visited (its
   * state is left at the last one). Global in parallel. */
  void Data(ForwardTrajectory& trajectory, mfem::Vector& d);
  /** @brief The data vector on @p problem's current (static) state;
   * integrated terms are refused. */
  void Data(LinearQuasiStaticProblem& problem, mfem::Vector& d);

  /** @brief The adjoint force of sum_i c_i J_i (c sized to Size(); null:
   * all ones). The returned force refers to this set's duals: keep the
   * set alive while the force is used. */
  AdjointForce Force(const mfem::Vector* c = nullptr) const;

 private:
  struct Obs {
    const Measurement* m;
    mfem::real_t t;
    std::function<mfem::real_t(mfem::real_t)> w;  // set: integrated
  };
  std::vector<Obs> obs_;
  std::vector<MeasurementDual> duals_;
  LinearQuasiStaticProblem* problem_ = nullptr;
};

}  // namespace AdGIA
