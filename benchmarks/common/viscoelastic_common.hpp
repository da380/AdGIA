// ============================================================================
// viscoelastic_common.hpp
//
// What the viscoelastic benchmark drivers with exact references share
// (viscoelastic/box/viscoelastic_box.cpp, viscoelastic/sphere/
// viscoelastic_sphere.cpp): reading a case file, the load history S(t) of
// histories.py, the layer fields and the coefficient that looks a layer up
// at a point, and the cost counters.
//
// A layer has a coordinate interval [bottom, top]: z for the box slabs, the
// radius for the spheres. Its fields are evaluated at a point x with that
// coordinate s:
//   number                         constant
//   {"kind": "geometric_z" | "geometric_r", "bottom": v0, "top": v1}
//                                  v0 (v1/v0)^((s - bottom)/(top - bottom)),
//                                  the profile of a viscosity varying by
//                                  decades with depth
//   {"kind": "lateral_tanh", "value": v, "contrast": C, "width": w,
//    "axis": a}                    v C^((1 + tanh(x_a / w)) / 2): v on the
//                                  side x_a << 0, C v on x_a >> 0, a smooth
//                                  lateral variation (no discontinuity, so
//                                  no interface for the mesh to follow)
// ============================================================================

#pragma once

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "AdGIA.hpp"

namespace vebench {

using namespace mfem;
using AdGIA::Json;
using AdGIA::LinearQuasiStaticProblemBase;

// --- the case ----------------------------------------------------------------

inline const Json& Member(const Json& obj, const std::string& key) {
  const Json* v = obj.Find(key);
  MFEM_VERIFY(v, "Case: missing member '" << key << "'.");
  return *v;
}

inline real_t Number(const Json& obj, const std::string& key) {
  const Json& v = Member(obj, key);
  MFEM_VERIFY(v.type == Json::Type::Number,
              "Case: '" << key << "' is not a number.");
  return v.number;
}

inline real_t NumberOr(const Json& obj, const std::string& key,
                       real_t fallback) {
  const Json* v = obj.Find(key);
  return v && v->type == Json::Type::Number ? v->number : fallback;
}

inline std::vector<real_t> Numbers(const Json& v) {
  std::vector<real_t> out;
  for (const Json& e : v.array) out.push_back(e.number);
  return out;
}

// The load history S(t): a sum of pieces c0 + c1 s + a cos(w s) +
// b sin(w s), s = t - start, on [start, end), right-continuous; at the
// time left_at it is evaluated as the left limit instead (a step ending at
// a jump).
class LoadHistory {
 public:
  struct Piece {
    real_t start, end, c0, c1, a, b, omega;
  };
  struct Breakpoint {
    real_t time;
    bool jump;
  };

  /// A constant load: no pieces and no breakpoints (the history is then
  /// only consulted for its breakpoints, of which there are none).
  LoadHistory() = default;

  explicit LoadHistory(const Json& h) {
    for (const Json& p : Member(h, "pieces").array) {
      const Json* e = p.Find("end");
      pieces_.push_back(
          {Number(p, "start"),
           e && e->type == Json::Type::Number
               ? e->number
               : std::numeric_limits<real_t>::infinity(),
           NumberOr(p, "c0", 0.0), NumberOr(p, "c1", 0.0),
           NumberOr(p, "a", 0.0), NumberOr(p, "b", 0.0),
           NumberOr(p, "omega", 0.0)});
    }
    if (const Json* bp = h.Find("breakpoints")) {
      for (const Json& b : bp->array) {
        const Json* j = b.Find("jump");
        breakpoints_.push_back(
            {Number(b, "time"), j && j->type == Json::Type::Bool && j->boolean});
      }
    }
  }

  real_t operator()(real_t t) const {
    const bool left = std::abs(t - left_at_) <=
                      1e-12 * std::max<real_t>(1.0, std::abs(t));
    // A stepped time is the jump time up to rounding, on either side of
    // it: evaluate AT the jump, or a t an ulp past it would fall outside
    // the piece that ends there.
    if (left) t = left_at_;
    real_t v = 0.0;
    for (const Piece& p : pieces_) {
      const bool in = left ? (p.start < t && t <= p.end)
                           : (p.start <= t && t < p.end);
      if (!in) continue;
      const real_t s = t - p.start;
      v += p.c0 + p.c1 * s + p.a * std::cos(p.omega * s) +
           p.b * std::sin(p.omega * s);
    }
    return v;
  }

  void SetLeftLimitAt(real_t t) { left_at_ = t; }
  void ClearLeftLimit() { left_at_ = std::numeric_limits<real_t>::quiet_NaN(); }
  const std::vector<Breakpoint>& Breakpoints() const { return breakpoints_; }

 private:
  std::vector<Piece> pieces_;
  std::vector<Breakpoint> breakpoints_;
  real_t left_at_ = std::numeric_limits<real_t>::quiet_NaN();
};

// A material field of a layer [s0, s1] (header comment for the kinds).
struct Field {
  enum class Kind { Constant, Geometric, LateralTanh } kind = Kind::Constant;
  real_t v0 = 0.0, v1 = 0.0, s0 = 0.0, s1 = 1.0;
  real_t contrast = 1.0, width = 1.0;
  int axis = 0;

  static Field Constant(real_t v) {
    Field f;
    f.v0 = f.v1 = v;
    return f;
  }

  static Field Parse(const Json& v, real_t s0, real_t s1) {
    Field f;
    f.s0 = s0;
    f.s1 = s1;
    if (v.type == Json::Type::Number) {
      f.v0 = f.v1 = v.number;
      return f;
    }
    MFEM_VERIFY(v.type == Json::Type::Object,
                "Case: a field is a number or an object.");
    const std::string& kind = Member(v, "kind").string;
    if (kind == "geometric_z" || kind == "geometric_r") {
      f.kind = Kind::Geometric;
      f.v0 = Number(v, "bottom");
      f.v1 = Number(v, "top");
      MFEM_VERIFY(f.v0 > 0.0 && f.v1 > 0.0,
                  "Case: a geometric field must be positive.");
    } else if (kind == "lateral_tanh") {
      f.kind = Kind::LateralTanh;
      f.v0 = f.v1 = Number(v, "value");
      f.contrast = Number(v, "contrast");
      f.width = Number(v, "width");
      f.axis = static_cast<int>(Number(v, "axis"));
      MFEM_VERIFY(f.v0 > 0.0 && f.contrast > 0.0 && f.width > 0.0,
                  "Case: a lateral_tanh field needs value, contrast, "
                  "width > 0.");
    } else {
      MFEM_ABORT("Case: unknown field kind '" << kind << "'.");
    }
    return f;
  }

  bool IsConstant() const { return kind == Kind::Constant; }

  real_t operator()(const Vector& x, real_t s) const {
    switch (kind) {
      case Kind::Constant:
        return v0;
      case Kind::Geometric: {
        const real_t u =
            std::clamp((s - s0) / (s1 - s0), real_t(0), real_t(1));
        return v0 * std::pow(v1 / v0, u);
      }
      case Kind::LateralTanh: {
        const real_t u = 0.5 * (1.0 + std::tanh(x[axis] / width));
        return v0 * std::pow(contrast, u);
      }
    }
    return v0;
  }
};

struct Branch {
  Field mu, tau;
};

struct Layer {
  std::string name;
  int attribute = 0;  // element attribute (spheres: from the manifest)
  real_t bottom = 0.0, top = 0.0;
  Field kappa, mu_inf;
  std::vector<Branch> branches;
};

// A layer's fields from its case entry ({"name", "kappa", "mu_inf",
// "branches": [{"mu", "tau"}]}), on [bottom, top].
inline Layer ParseLayer(const Json& l, real_t bottom, real_t top) {
  Layer layer;
  layer.name = Member(l, "name").string;
  layer.bottom = bottom;
  layer.top = top;
  layer.kappa = Field::Parse(Member(l, "kappa"), bottom, top);
  layer.mu_inf = Field::Parse(Member(l, "mu_inf"), bottom, top);
  for (const Json& b : Member(l, "branches").array) {
    layer.branches.push_back({Field::Parse(Member(b, "mu"), bottom, top),
                              Field::Parse(Member(b, "tau"), bottom, top)});
  }
  return layer;
}

using Coordinate = std::function<real_t(const Vector&)>;

// A coefficient over the layered body: at a point, fields[j] of the layer j
// containing its coordinate. With one field for every layer it is that
// field everywhere (a composite region evaluates it on its own elements).
class LayeredCoefficient : public Coefficient {
 public:
  LayeredCoefficient(const std::vector<Layer>* layers,
                     std::vector<Field> fields, Coordinate coordinate)
      : layers_(layers),
        fields_(std::move(fields)),
        coordinate_(std::move(coordinate)) {}

  real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    T.Transform(ip, x_);
    const real_t s = coordinate_(x_);
    std::size_t j = 0;
    while (j + 1 < layers_->size() && s >= (*layers_)[j].top) j++;
    return fields_[j](x_, s);
  }

 private:
  const std::vector<Layer>* layers_;
  std::vector<Field> fields_;
  Coordinate coordinate_;
  Vector x_;
};

// --- counting ----------------------------------------------------------------

struct Counters {
  int solves = 0, assemblies = 0, setups = 0;
  long its = 0;
  static Counters Of(const LinearQuasiStaticProblemBase& p) {
    return {p.NumSolves(), p.NumAssemblies(), p.NumPreconditionerSetups(),
            p.TotalIterations()};
  }
  Counters operator-(const Counters& o) const {
    return {solves - o.solves, assemblies - o.assemblies, setups - o.setups,
            its - o.its};
  }
  Counters operator+(const Counters& o) const {
    return {solves + o.solves, assemblies + o.assemblies, setups + o.setups,
            its + o.its};
  }
};

// --- the evolution -------------------------------------------------------------
//
// The step grid is the output times plus, aligned, the breakpoints inside
// (0, t_final); fixed-step schemes divide each interval into
// max(min_steps, ceil(length / dt)) equal steps, a step equal to the last
// one up to rounding IS the last one (one effective operator for a fixed
// dt), a step ending at a jump sees the load's left limit and the next
// starts from the right limit. At each output time observe(m, t, k) is
// called with the displacement consistent with (m, t); k counts the
// outputs from 0. The elastic t = 0+ observation is k = -1. An observer
// taking a fourth argument, const OutputCost&, also receives the stepping
// cost and wall time up to that output (zero at k = -1).

struct StepOptions {
  std::string scheme = "exptrap";  // exptrap sdirk23 be etd1 rk4 adaptive
  real_t dt = 0.1;
  int min_steps = 1;
  bool align = true;
  real_t rtol = 1e-4, atol = 1e-10;
};

struct OutputCost {
  Counters cost;  // cumulative, stepping only
  double seconds = 0.0;
};

struct EvolveResult {
  int steps = 0, rejected = 0;
  Counters stepping, observation, total;
  double seconds = 0.0;
  bool ok = true;
  std::vector<OutputCost> outputs;
};

inline bool KnownScheme(const std::string& s) {
  return s == "exptrap" || s == "sdirk23" || s == "be" || s == "etd1" ||
         s == "rk4" || s == "adaptive";
}

template <class Observe>
EvolveResult Evolve(LinearQuasiStaticProblemBase& problem,
                    AdGIA::ViscoelasticOperator& visco,
                    LoadHistory& history, const std::vector<real_t>& times,
                    const StepOptions& opt, Observe observe) {
  using Clock = std::chrono::steady_clock;
  MFEM_VERIFY(KnownScheme(opt.scheme), "Unknown scheme " << opt.scheme);
  struct Mark {
    real_t time;
    bool output = false, jump = false;
  };
  std::vector<Mark> marks;
  for (const real_t t : times) marks.push_back({t, true, false});
  if (opt.align) {
    for (const auto& b : history.Breakpoints()) {
      if (b.time <= 0.0 || b.time >= times.back()) continue;
      bool merged = false;
      for (Mark& m : marks) {
        if (std::abs(m.time - b.time) <= 1e-12 * std::max<real_t>(1, b.time)) {
          m.jump = m.jump || b.jump;
          merged = true;
        }
      }
      if (!merged) marks.push_back({b.time, false, b.jump});
    }
  }
  std::sort(marks.begin(), marks.end(),
            [](const Mark& a, const Mark& b) { return a.time < b.time; });

  std::unique_ptr<ODESolver> ode;
  AdGIA::AdaptiveExponentialTrapezoidSolver adaptive;
  if (opt.scheme == "sdirk23") {
    ode = std::make_unique<SDIRK23Solver>(2);  // the L-stable variant
  } else if (opt.scheme == "exptrap") {
    ode = std::make_unique<AdGIA::ExponentialTrapezoidSolver>();
  } else if (opt.scheme == "be") {
    ode = std::make_unique<BackwardEulerSolver>();
  } else if (opt.scheme == "etd1") {
    ode = std::make_unique<AdGIA::ExponentialEulerSolver>();
  } else if (opt.scheme == "rk4") {
    ode = std::make_unique<RK4Solver>();
  } else {
    adaptive.Init(visco);
    adaptive.SetTolerances(opt.rtol, opt.atol);
  }
  if (ode) ode->Init(visco);

  EvolveResult r;
  Vector m(visco.Height());
  m = 0.0;
  real_t t = 0.0;
  MPI_Barrier(MPI_COMM_WORLD);
  const auto start = Clock::now();
  const Counters c0 = Counters::Of(problem);
  // Call the observer with or without the cost, as it accepts.
  auto call = [&observe](const Vector& mm, real_t tt, int kk,
                         const OutputCost& oc) {
    if constexpr (std::is_invocable_v<Observe&, const Vector&, real_t, int,
                                      const OutputCost&>) {
      observe(mm, tt, kk, oc);
    } else {
      observe(mm, tt, kk);
    }
  };
  r.ok = visco.SolveElastic(m, t) && r.ok;
  call(m, real_t(0), -1, OutputCost{});
  Counters observed = Counters::Of(problem) - c0;
  real_t dt_adaptive = opt.dt, h_last = -1.0;
  int k = 0;
  for (const Mark& mark : marks) {
    if (mark.jump) history.SetLeftLimitAt(mark.time);
    if (ode) {
      const real_t span = mark.time - t;
      const int n = std::max(
          opt.min_steps, static_cast<int>(std::ceil(span / opt.dt - 1e-9)));
      real_t h = span / n;
      if (std::abs(h - h_last) <= 1e-12 * h) h = h_last;
      h_last = h;
      for (int s = 0; s < n; s++) {
        real_t step = h;
        ode->Step(m, t, step);
      }
      r.steps += n;
    } else {
      adaptive.Integrate(m, t, mark.time, dt_adaptive);
    }
    // t equals mark.time up to rounding and is not reset (the operator's
    // displacement cache is keyed on the exact (m, t)), except at a jump.
    if (mark.output) {
      const Counters before = Counters::Of(problem);
      r.ok = visco.SolveElastic(m, t) && r.ok;
      observed = observed + (Counters::Of(problem) - before);
      OutputCost oc;
      oc.cost = (Counters::Of(problem) - c0) - observed;
      oc.seconds = std::chrono::duration<double>(Clock::now() - start).count();
      r.outputs.push_back(oc);
      call(m, mark.time, k++, oc);
    }
    if (mark.jump) {
      history.ClearLeftLimit();
      visco.InvalidateDisplacement();
      t = mark.time;
    }
  }
  if (!ode) {
    r.steps = adaptive.NumAcceptedSteps();
    r.rejected = adaptive.NumRejectedSteps();
  }
  r.seconds = std::chrono::duration<double>(Clock::now() - start).count();
  r.total = Counters::Of(problem) - c0;
  r.observation = observed;
  r.stepping = r.total - observed;
  return r;
}

inline real_t GlobalSum(real_t v) {
  real_t g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                MPI_COMM_WORLD);
  return g;
}

}  // namespace vebench
