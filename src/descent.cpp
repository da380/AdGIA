#include "AdGIA/descent.hpp"

#include <algorithm>
#include <iostream>

namespace AdGIA {

using namespace mfem;

ConstrainedMetric::ConstrainedMetric(RieszMap& riesz) : riesz_(&riesz) {}

void ConstrainedMetric::SetConstraint(const Vector& ell) {
  ell_ = ell;
  ell_gradient_.SetSize(ell.Size());
  riesz_->Mult(ell_, ell_gradient_);
  ell_scale_ = riesz_->Pair(ell_, ell_gradient_);
  MFEM_VERIFY(ell_scale_ > 0.0,
              "ConstrainedMetric: the constraint's metric norm vanishes.");
}

void ConstrainedMetric::SetPrior(const Operator& K, real_t weight) {
  prior_ = &K;
  weight_ = weight;
}

real_t ConstrainedMetric::PriorValue(const Vector& c) const {
  if (!HasPrior()) {
    return 0.0;
  }
  Vector Kc(c.Size());
  prior_->Mult(c, Kc);
  return 0.5 * weight_ * riesz_->Pair(Kc, c);
}

void ConstrainedMetric::AddPriorDual(const Vector& c, Vector& j) const {
  if (!HasPrior()) {
    return;
  }
  Vector Kc(c.Size());
  prior_->Mult(c, Kc);
  j.Add(weight_, Kc);
}

void ConstrainedMetric::AddPriorHessian(const Vector& d, Vector& Hd) const {
  if (!HasPrior()) {
    return;
  }
  Vector Kd(d.Size());
  prior_->Mult(d, Kd);
  Hd.Add(weight_, Kd);
}

void ConstrainedMetric::Gradient(const Vector& dual, Vector& g) const {
  g.SetSize(dual.Size());
  riesz_->Mult(dual, g);
  Project(g);
}

void ConstrainedMetric::Project(Vector& v) const {
  if (ell_.Size() == 0) {
    return;
  }
  const real_t along = riesz_->Pair(ell_, v);
  v.Add(-along / ell_scale_, ell_gradient_);
}

DescentResult NonlinearCG(const DescentFunctional& f,
                          const ConstrainedMetric& metric, Vector& c,
                          const DescentOptions& options) {
  DescentResult result;
  Vector j(c.Size()), g(c.Size()), d(c.Size()), j_old(c.Size()),
      g_old(c.Size()), c_trial(c.Size());

  // The descended objective is value + prior; the reported values are
  // the functional's own.
  auto total = [&](const Vector& at, Vector* dual, real_t& raw) {
    raw = f.evaluate(at, dual);
    if (dual) {
      metric.AddPriorDual(at, *dual);
    }
    return raw + metric.PriorValue(at);
  };

  real_t raw;
  real_t jval = total(c, &j, raw);
  result.initial = raw;
  metric.Gradient(j, g);
  d = g;
  d *= -1.0;

  real_t t = 1.0;
  bool restarted = false;
  for (int it = 1; it <= options.max_iterations; it++) {
    const real_t slope = metric.Riesz().Pair(j, d);
    if (slope >= 0.0) {
      if (restarted) {
        // Projected steepest descent itself does not descend: the
        // constrained optimum, to the solver floor.
        break;
      }
      d = g;
      d *= -1.0;
      restarted = true;
      continue;
    }
    restarted = false;

    // Forward-tracking line search: the objective is smooth (for the
    // feasibility functional, an exact quartic along the ray), so
    // double while it falls, halve while it does not; warm-started.
    real_t raw_new = raw;
    auto at = [&](real_t step, real_t& raw_out) {
      c_trial = c;
      c_trial.Add(step, d);
      real_t r;
      const real_t v = total(c_trial, nullptr, r);
      raw_out = r;
      return v;
    };
    real_t j_new = at(t, raw_new);
    while (j_new >= jval && t > 1e-14) {
      t *= 0.5;
      j_new = at(t, raw_new);
    }
    while (t > 1e-14) {
      real_t raw_next;
      const real_t j_next = at(2.0 * t, raw_next);
      if (j_next >= j_new) {
        break;
      }
      t *= 2.0;
      j_new = j_next;
      raw_new = raw_next;
    }
    if (t <= 1e-14) {
      break;
    }
    c.Add(t, d);
    // Re-evaluate at the accepted point before anything observes it:
    // the line search's last call may have been a rejected probe, and
    // the monitor (and the next conjugacy) read the functional's state.
    j_old = j;
    g_old = g;
    jval = total(c, &j, raw);
    result.iterations = it;
    if (options.monitor) {
      options.monitor(it, raw, t);
    }
    if (options.print) {
      std::cout << "cg " << it << ": value " << raw << ", step " << t
                << "\n";
    }
    if (options.tolerance > 0 && raw < options.tolerance * result.initial) {
      result.converged = true;
      break;
    }

    // Polak-Ribiere+, in dual-gradient pairings; the projected
    // gradients make these the constraint manifold's inner products.
    metric.Gradient(j, g);
    Vector dg(g);
    dg -= g_old;
    const real_t beta =
        std::max(real_t{0}, metric.Riesz().Pair(j, dg) /
                                metric.Riesz().Pair(j_old, g_old));
    d *= beta;
    d -= g;
  }
  result.final = raw;
  return result;
}

namespace {

// The Riesz map as the inner CG's preconditioner.
class RieszPreconditioner : public Solver {
 public:
  explicit RieszPreconditioner(RieszMap& m)
      : Solver(m.Height()), map_(&m) {}
  void Mult(const Vector& x, Vector& y) const override { map_->Mult(x, y); }
  void SetOperator(const Operator&) override {}

 private:
  RieszMap* map_;
};

// The Levenberg-Marquardt model operator H_GN + lambda G + lambda_p K.
class LmOperator : public Operator {
 public:
  LmOperator(const DescentFunctional& f, const ConstrainedMetric& metric,
             const Operator& damping, int n)
      : Operator(n), f_(&f), metric_(&metric), damping_(&damping) {}

  real_t lambda = 1.0;

  void Mult(const Vector& x, Vector& y) const override {
    f_->gauss_newton(x, y);
    Vector Gx(x.Size());
    damping_->Mult(x, Gx);
    y.Add(lambda, Gx);
    metric_->AddPriorHessian(x, y);
  }

 private:
  const DescentFunctional* f_;
  const ConstrainedMetric* metric_;
  const Operator* damping_;
};

}  // namespace

DescentResult LevenbergMarquardt(const DescentFunctional& f,
                                 const ConstrainedMetric& metric,
                                 const Operator& damping, Vector& c,
                                 const DescentOptions& options) {
  MFEM_VERIFY(f.gauss_newton,
              "LevenbergMarquardt: the functional supplies no Gauss-Newton "
              "product.");
  DescentResult result;
  Vector j(c.Size()), c_trial(c.Size());

  auto total = [&](const Vector& at, Vector* dual, real_t& raw) {
    raw = f.evaluate(at, dual);
    if (dual) {
      metric.AddPriorDual(at, *dual);
    }
    return raw + metric.PriorValue(at);
  };

  real_t raw;
  real_t jval = total(c, &j, raw);
  result.initial = raw;

  LmOperator lm(f, metric, damping, c.Size());
  RieszPreconditioner prec(options.inner_preconditioner
                               ? *options.inner_preconditioner
                               : metric.Riesz());

  real_t lambda = 1.0;
  for (int it = 1; it <= options.max_iterations; it++) {
    bool accepted = false;
    real_t raw_new = raw, j_new = jval;
    for (int attempt = 0; attempt < 8 && !accepted; attempt++) {
      lm.lambda = lambda;
      std::unique_ptr<CGSolver> cg;
#ifdef MFEM_USE_MPI
      cg = std::make_unique<CGSolver>(MPI_COMM_WORLD);
#else
      cg = std::make_unique<CGSolver>();
#endif
      cg->SetOperator(lm);
      cg->SetPreconditioner(prec);
      cg->SetRelTol(1e-2);  // inexact Newton: the model is local anyway
      cg->SetAbsTol(0.0);
      cg->SetMaxIter(30);
      cg->SetPrintLevel(0);
      Vector mj(j);
      mj *= -1.0;
      Vector step(c.Size());
      step = 0.0;
      cg->Mult(mj, step);

      c_trial = c;
      c_trial += step;
      metric.Project(c_trial);
      j_new = total(c_trial, nullptr, raw_new);
      if (j_new < jval) {
        accepted = true;
        lambda = std::max(real_t{1e-12}, lambda / 3.0);
      } else {
        lambda *= 4.0;
        // The trial evaluation moved the functional's state off c; the
        // next attempt's Gauss-Newton products must act at c again.
        real_t raw_pin;
        total(c, nullptr, raw_pin);
      }
    }
    if (!accepted) {
      break;
    }
    c = c_trial;
    jval = total(c, &j, raw);  // re-evaluate: the GN closure's state
    result.iterations = it;
    if (options.monitor) {
      options.monitor(it, raw, lambda);
    }
    if (options.print) {
      std::cout << "lm " << it << ": value " << raw << ", lambda "
                << lambda << "\n";
    }
    if (options.tolerance > 0 && raw < options.tolerance * result.initial) {
      result.converged = true;
      break;
    }
  }
  result.final = raw;
  return result;
}

}  // namespace AdGIA
