// ============================================================================
// relabelling.hpp
//
// The relabelled (mapped) 3-D benchmark's local pieces (doc/mappings.md;
// doc/benchmarks.tex, "The relabelling family" and "The perturbation
// family"):
//
//  - RadialProfiles: the model's exact radial fields, read from the
//    radial_profiles.txt that make_case.py writes (Chebyshev points of
//    the second kind per layer, barycentric interpolation). Composing a
//    radial coefficient with a mapping needs its value at |xi(x)|, which
//    the exported L2 fields cannot give.
//
//  - InteriorRelabelling: a genuinely three-dimensional relabelling of
//    the ball that is the POINTWISE IDENTITY on every interface and on
//    and outside the surface, C^1 across them (per-layer bumps vanish
//    with their first derivative at the layer boundaries, so F = I on
//    every interface too), and laterally wild in between:
//
//        xi(x) = x + A sum_layers h_i(r) w(x),
//        h_i   = [4 (r - r_i)(r_o - r) / (r_o - r_i)^2]^2,
//        w     = (sin(pi y) + c cos(pi z),
//                 sin(pi z) + c cos(pi x),
//                 sin(pi x) + c cos(pi y)),   c = 1/2
//
//    (2-D: w = (sin(pi y) + c cos(pi x), sin(pi x) - c cos(pi y))). The
//    PHYSICAL problem is therefore the unchanged spherical one — the
//    pyslfp reference stays exact — while every pulled-back coefficient
//    carries full lateral variation and F is non-radial everywhere
//    between the interfaces. Solved with the mapped referential
//    machinery, the solution composed back must reproduce the
//    reference; on the interfaces the composition is the identity, so
//    the harmonic analysis needs no change at all.
//
//  - InterfaceShift: the degree-0 shift of one interface used by the
//    perturbation family (below).
//
// The exactness mode is the mapping object's: the analytic
// CallableDiffeomorphism here gives convergence-to-reference runs (leg A,
// love_benchmark -map); its nodal interpolant
// (AdGIA::Interpolate) gives the discrete change-of-variables
// identity against the mapped mesh (leg B, relabelled_identity).
// ============================================================================

#pragma once

#include <array>
#include <cmath>
#include <fstream>
#include <map>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

#include "AdGIA.hpp"

namespace benchmark {

using namespace mfem;
using namespace AdGIA;

// The exact radial profiles of the model's fields, by layer.
class RadialProfiles {
 public:
  explicit RadialProfiles(const std::string& path) {
    std::ifstream in(path);
    MFEM_VERIFY(in.good(),
                "RadialProfiles: cannot read "
                    << path
                    << " — re-make the case (make_case.py writes it).");
    std::string word;
    int n_layers = 0, nodes = 0;
    in >> word >> n_layers >> word >> nodes >> word;
    std::string header_line;
    std::getline(in, header_line);
    std::istringstream header(header_line);
    std::vector<std::string> names;
    while (header >> word) {
      names.push_back(word);
    }
    layers_.resize(n_layers);
    for (auto& layer : layers_) {
      int attribute = 0, m = 0;
      in >> word >> attribute >> layer.lo >> layer.hi >> m;
      MFEM_VERIFY(word == "layer" && m == nodes,
                  "RadialProfiles: malformed " << path);
      layer.radius.resize(nodes);
      for (auto& r : layer.radius) {
        in >> r;
      }
      for (const auto& name : names) {
        auto& v = layer.field[name];
        v.resize(nodes);
        for (auto& x : v) {
          in >> x;
        }
      }
    }
    MFEM_VERIFY(in.good(), "RadialProfiles: truncated " << path);
  }

  /// The field at radius r: the layer holding r (a boundary radius goes
  /// with the layer below it, a radius beyond the surface with the
  /// outermost layer), interpolated barycentrically on its Chebyshev
  /// nodes.
  real_t Eval(const std::string& name, real_t r) const {
    const Layer* layer = &layers_.back();
    for (const auto& l : layers_) {
      if (r <= l.hi) {
        layer = &l;
        break;
      }
    }
    const auto& x = layer->radius;
    const auto it = layer->field.find(name);
    MFEM_VERIFY(it != layer->field.end(),
                "RadialProfiles: no field " << name);
    const auto& f = it->second;
    const int n = static_cast<int>(x.size());
    // Barycentric weights of Chebyshev points of the second kind:
    // (-1)^j, halved at the ends.
    real_t num = 0.0, den = 0.0;
    for (int j = 0; j < n; j++) {
      const real_t d = r - x[j];
      if (std::abs(d) < 1e-14 * (layer->hi - layer->lo)) {
        return f[j];
      }
      real_t w = (j % 2 == 0) ? 1.0 : -1.0;
      if (j == 0 || j == n - 1) {
        w *= 0.5;
      }
      num += w / d * f[j];
      den += w / d;
    }
    return num / den;
  }

  /// The layer boundaries 0 = r_0 < r_1 < ... < r_n (body layers).
  std::vector<real_t> Boundaries() const {
    std::vector<real_t> b{layers_.front().lo};
    for (const auto& l : layers_) {
      b.push_back(l.hi);
    }
    return b;
  }

  /// The mass enclosed within radius r, 4 pi int rho r'^2 dr' (balls),
  /// by 64-point Gauss-Legendre per segment of the exact profiles: the
  /// mapped runs need the PERTURBED model's gravity, which the mesh
  /// integrals of the base fields cannot give.
  real_t EnclosedMass(real_t r) const {
    constexpr int n = 64;
    static const auto nodes = [] {
      std::array<std::array<real_t, 2>, n> nw{};
      // Newton on Legendre P_n for the Gauss nodes and weights.
      for (int i = 0; i < n; i++) {
        real_t x = std::cos(std::numbers::pi_v<real_t> * (i + 0.75) /
                            (n + 0.5));
        real_t pp = 0.0;
        for (int it = 0; it < 100; it++) {
          real_t p0 = 1.0, p1 = x;
          for (int k = 2; k <= n; k++) {
            const real_t p2 = ((2.0 * k - 1.0) * x * p1 -
                               (k - 1.0) * p0) / k;
            p0 = p1;
            p1 = p2;
          }
          pp = n * (x * p1 - p0) / (x * x - 1.0);
          const real_t dx = p1 / pp;
          x -= dx;
          if (std::abs(dx) < 1e-16) {
            break;
          }
        }
        nw[i] = {x, 2.0 / ((1.0 - x * x) * pp * pp)};
      }
      return nw;
    }();
    constexpr real_t four_pi = 4.0 * std::numbers::pi_v<real_t>;
    real_t mass = 0.0;
    for (const auto& layer : layers_) {
      const real_t lo = layer.lo, hi = std::min(layer.hi, r);
      if (hi <= lo) {
        break;
      }
      const real_t mid = 0.5 * (lo + hi), half = 0.5 * (hi - lo);
      real_t sum = 0.0;
      for (const auto& [x, w] : nodes) {
        const real_t rr = mid + half * x;
        sum += w * Eval("rho", rr) * rr * rr;
      }
      mass += four_pi * half * sum;
    }
    return mass;
  }

 private:
  struct Layer {
    real_t lo = 0.0, hi = 0.0;
    std::vector<real_t> radius;
    std::map<std::string, std::vector<real_t>> field;
  };
  std::vector<Layer> layers_;
};

// The nodal interpolant of a mapping, usable on the parent mesh AND its
// SubMeshes: a GridFunctionDiffeomorphism's displacement lives on ONE
// mesh, but the problem classes evaluate phi_e on submesh
// transformations (stiffness and couplings on the displacement SubMesh,
// slip machinery on solid and fluid, buffer folds, broken-zeta blocks
// on the outer region), and a cross-mesh GetVectorGradient reads the
// wrong elements. This wrapper interpolates once on the parent and
// TRANSPLANTS the displacement to each registered SubMesh — the nodal
// transfer restricts the interpolant exactly, so the discrete
// change-of-variables identity is preserved — and dispatches on the
// transformation's mesh.
class MultiMeshDiffeomorphism : public Diffeomorphism {
 public:
  MultiMeshDiffeomorphism(Diffeomorphism& xi, ParMesh& parent)
      : Diffeomorphism(parent.Dimension()) {
    maps_.push_back(std::make_unique<GridFunctionDiffeomorphism>(
        Interpolate(xi, parent)));
    meshes_.push_back(&parent);
  }

  /// Register a SubMesh of the parent the mapping will be evaluated on.
  void AddMesh(ParSubMesh& sub) {
    const auto* h =
        dynamic_cast<const ParGridFunction*>(&maps_.front()->Displacement());
    MFEM_VERIFY(h, "MultiMeshDiffeomorphism: parallel interpolant.");
    auto fec = std::unique_ptr<FiniteElementCollection>(
        FiniteElementCollection::New(h->FESpace()->FEColl()->Name()));
    auto fes = std::make_unique<ParFiniteElementSpace>(&sub, fec.get(),
                                                       GetVDim());
    auto disp = std::make_unique<ParGridFunction>(fes.get());
    ParSubMesh::Transfer(*h, *disp);
    meshes_.push_back(&sub);
    maps_.push_back(std::make_unique<GridFunctionDiffeomorphism>(
        std::move(fec),
        std::unique_ptr<FiniteElementSpace>(fes.release()),
        std::unique_ptr<GridFunction>(disp.release())));
  }

  void Eval(Vector& V, ElementTransformation& T,
            const IntegrationPoint& ip) override {
    Pick(T).Eval(V, T, ip);
  }
  void EvalGradient(DenseMatrix& F, ElementTransformation& T,
                    const IntegrationPoint& ip) override {
    Pick(T).EvalGradient(F, T, ip);
  }

 private:
  GridFunctionDiffeomorphism& Pick(ElementTransformation& T) {
    for (std::size_t i = 0; i < meshes_.size(); i++) {
      if (T.mesh == meshes_[i]) {
        return *maps_[i];
      }
    }
    MFEM_ABORT(
        "MultiMeshDiffeomorphism: evaluated on an unregistered mesh "
        "(AddMesh every SubMesh the problem assembles on).");
    return *maps_.front();
  }

  std::vector<const Mesh*> meshes_;
  std::vector<std::unique_ptr<GridFunctionDiffeomorphism>> maps_;
};

// The interior relabelling (header note): identity, with identity
// gradient, on every layer boundary and on and outside the surface.
inline CallableDiffeomorphism InteriorRelabelling(
    int dim, const std::vector<real_t>& boundaries, real_t amplitude) {
  constexpr real_t c = 0.5;
  constexpr real_t pi = std::numbers::pi_v<real_t>;
  auto bump = [boundaries](real_t r, real_t& h, real_t& hp) {
    h = 0.0;
    hp = 0.0;
    for (std::size_t i = 0; i + 1 < boundaries.size(); i++) {
      const real_t lo = boundaries[i], hi = boundaries[i + 1];
      if (r <= lo || r >= hi) {
        continue;
      }
      const real_t s = 4.0 / ((hi - lo) * (hi - lo));
      const real_t q = s * (r - lo) * (hi - r);
      h = q * q;
      hp = 2.0 * q * s * (hi + lo - 2.0 * r);
      break;
    }
  };
  auto w_of = [dim](const Vector& x, Vector& w, DenseMatrix& Dw) {
    w.SetSize(dim);
    Dw.SetSize(dim);
    Dw = 0.0;
    if (dim == 3) {
      w[0] = std::sin(pi * x[1]) + c * std::cos(pi * x[2]);
      w[1] = std::sin(pi * x[2]) + c * std::cos(pi * x[0]);
      w[2] = std::sin(pi * x[0]) + c * std::cos(pi * x[1]);
      Dw(0, 1) = pi * std::cos(pi * x[1]);
      Dw(0, 2) = -c * pi * std::sin(pi * x[2]);
      Dw(1, 2) = pi * std::cos(pi * x[2]);
      Dw(1, 0) = -c * pi * std::sin(pi * x[0]);
      Dw(2, 0) = pi * std::cos(pi * x[0]);
      Dw(2, 1) = -c * pi * std::sin(pi * x[1]);
    } else {
      w[0] = std::sin(pi * x[1]) + c * std::cos(pi * x[0]);
      w[1] = std::sin(pi * x[0]) - c * std::cos(pi * x[1]);
      Dw(0, 0) = -c * pi * std::sin(pi * x[0]);
      Dw(0, 1) = pi * std::cos(pi * x[1]);
      Dw(1, 0) = pi * std::cos(pi * x[0]);
      Dw(1, 1) = c * pi * std::sin(pi * x[1]);
    }
  };
  return CallableDiffeomorphism(
      dim,
      [dim, amplitude, bump, w_of](const Vector& x, Vector& y) {
        const real_t r = x.Norml2();
        real_t h, hp;
        bump(r, h, hp);
        Vector w;
        DenseMatrix Dw;
        w_of(x, w, Dw);
        y.SetSize(dim);
        for (int i = 0; i < dim; i++) {
          y[i] = x[i] + amplitude * h * w[i];
        }
      },
      [dim, amplitude, bump, w_of](const Vector& x, DenseMatrix& F) {
        const real_t r = x.Norml2();
        real_t h, hp;
        bump(r, h, hp);
        Vector w;
        DenseMatrix Dw;
        w_of(x, w, Dw);
        F.SetSize(dim);
        F = 0.0;
        for (int i = 0; i < dim; i++) {
          F(i, i) = 1.0;
          for (int j = 0; j < dim; j++) {
            F(i, j) += amplitude * h * Dw(i, j);
            if (r > 1e-14) {
              F(i, j) += amplitude * hp * w[i] * x[j] / r;
            }
          }
        }
      });
}

// The degree-0 interface shift (the perturbation family,
// doc/benchmarks.tex): a purely radial, piecewise-linear map of the
// reference boundaries 0 = b_0 < ... < b_n onto shifted ones, moving
// ONE interface by eps and leaving the centre, the other interfaces,
// the surface and everything beyond fixed:
//
//   phi_e(x) = f(r) x^,  f affine per layer with f(b_j) = b_j + eps
//              delta_{jk} at the shifted interface k,
//   F = f'(r) r^ r^T + (f/r)(I - r^ r^T)   (exact; discontinuous
//                                           across interfaces, which
//                                           the per-region forms allow).
//
// The physical model is the SAME spherical one with that interface at
// b_k + eps, so pyslfp solves it exactly at every finite eps: the 3-D
// mapped solve on ONE fixed mesh has an exact reference per eps, and
// the fixed mesh cancels the discretisation bias in the finite
// difference [R(eps) - R(-eps)] / 2 eps against the 1-D derivative.
inline CallableDiffeomorphism InterfaceShift(
    int dim, const std::vector<real_t>& boundaries, int interface,
    real_t eps) {
  MFEM_VERIFY(interface >= 1 &&
                  interface + 1 < static_cast<int>(boundaries.size()),
              "InterfaceShift: an INTERIOR interface (1 .. n-1) moves; "
              "the centre and the surface stay.");
  auto f_of = [boundaries, interface, eps](real_t r, real_t& f,
                                           real_t& fp) {
    const auto& b = boundaries;
    const int n = static_cast<int>(b.size());
    f = r;
    fp = 1.0;
    if (r >= b[n - 1]) {
      return;  // on and beyond the surface: identity
    }
    // Side convention at the moving interface: the gradient's slope is
    // DISCONTINUOUS there, and the one-sided interface kernels are
    // assembled on the solid (outer) side, so radii within a roundoff
    // band of an interior boundary must take the OUTER segment's slope.
    // A plain `r <= b[j+1]` would hand those kernels the inner (fluid)
    // slope exactly on Sigma, which degrades the augmented-Lagrangian
    // sweeps of an outward shift (interpolated-F maps are immune: they
    // evaluate through the adjacent solid element).
    const real_t band = 1e-10 * b[n - 1];
    for (int j = 0; j + 1 < n; j++) {
      if (r <= b[j + 1] - (j + 2 < n ? band : real_t{0})) {
        const real_t lo = b[j], hi = b[j + 1];
        const real_t lo_t = lo + (j == interface ? eps : 0.0);
        const real_t hi_t = hi + (j + 1 == interface ? eps : 0.0);
        fp = (hi_t - lo_t) / (hi - lo);
        f = lo_t + fp * (r - lo);
        return;
      }
    }
  };
  return CallableDiffeomorphism(
      dim,
      [dim, f_of](const Vector& x, Vector& y) {
        const real_t r = x.Norml2();
        y.SetSize(dim);
        if (r < 1e-300) {
          y = x;
          return;
        }
        real_t f, fp;
        f_of(r, f, fp);
        y = x;
        y *= f / r;
      },
      [dim, f_of](const Vector& x, DenseMatrix& F) {
        const real_t r = x.Norml2();
        F.SetSize(dim);
        F = 0.0;
        real_t f = r, fp = 1.0;
        if (r > 1e-300) {
          f_of(r, f, fp);
        }
        const real_t t = r > 1e-300 ? f / r : 1.0;
        for (int i = 0; i < dim; i++) {
          F(i, i) = t;
          for (int j = 0; j < dim; j++) {
            if (r > 1e-300) {
              F(i, j) += (fp - t) * x[i] * x[j] / (r * r);
            }
          }
        }
      });
}

}  // namespace benchmark
