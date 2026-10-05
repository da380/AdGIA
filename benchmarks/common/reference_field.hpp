// ============================================================================
// reference_field.hpp
//
// The reference solution of a load problem as fields on the mesh. The file
// reference_fields.txt of a case (make_case.py) holds, for each layer of the
// model and each degree, the radial functions U, V and phi of the response
// to a surface density of unit coefficient, on Chebyshev points of the
// layer, and the gravity of the model there. For a load sigma = sum_i c_i Y_i the response is
//
//   u   = sum_i c_i [ U_l(r) Y_i x^ + V_l(r) grad_1 Y_i ]
//   phi = sum_i c_i phi_l(r) Y_i
//
// with l the degree of Y_i, and outside the body phi_l continues as
// phi_l(a) (a / r)^(l + 1). The radial functions are evaluated by the
// interpolating polynomial of their layer in its barycentric form; which
// layer a point belongs to is the attribute of its element, not its radius,
// so that the two sides of an interface keep their own values.
// ============================================================================

#pragma once

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "AdGIA.hpp"

namespace benchmark {

using namespace mfem;
using namespace AdGIA;

class RadialReference {
 public:
  explicit RadialReference(const std::string& path) {
    std::ifstream in(path);
    MFEM_VERIFY(in.good(), "RadialReference: cannot open " << path << ".");
    std::string word;
    int n_layers = 0;
    in >> word >> lmax_ >> word >> n_layers;
    MFEM_VERIFY(in.good() && lmax_ >= 0 && n_layers > 0,
                "RadialReference: " << path << " has no header.");
    layers_.resize(n_layers);
    for (Layer& layer : layers_) {
      int attribute = 0, fluid = 0, n = 0;
      in >> word >> attribute >> layer.r_inner >> layer.r_outer >> fluid >> n;
      MFEM_VERIFY(in.good() && word == "layer" && n > 1,
                  "RadialReference: " << path << " is malformed.");
      layer.fluid = fluid != 0;
      layer.r.SetSize(n);
      layer.gravity.SetSize(n);
      for (int j = 0; j < n; j++) {
        in >> layer.r[j];
      }
      for (int j = 0; j < n; j++) {
        in >> layer.gravity[j];
      }
      // Barycentric weights of the Chebyshev points of the second kind.
      layer.w.SetSize(n);
      for (int j = 0; j < n; j++) {
        layer.w[j] = (j % 2 ? -1.0 : 1.0) * (j == 0 || j == n - 1 ? 0.5 : 1.0);
      }
      layer.values.resize(3 * (lmax_ + 1));
      for (Vector& v : layer.values) {
        v.SetSize(n);
        for (int j = 0; j < n; j++) {
          in >> v[j];
        }
      }
      MFEM_VERIFY(!in.fail(), "RadialReference: " << path << " is short.");
    }
  }

  int MaxDegree() const { return lmax_; }
  int NumLayers() const { return static_cast<int>(layers_.size()); }
  real_t Radius() const { return layers_.back().r_outer; }
  bool Fluid(int attribute) const { return layers_[attribute - 1].fluid; }

  enum Component { U = 0, V = 1, Phi = 2 };

  // A radial function of degree l at the radius r, as the layer of the
  // given attribute has it; r is taken into the layer when outside it. An
  // attribute above the layers is the exterior, where phi continues
  // harmonically and the displacement is zero.
  real_t Eval(int attribute, Component c, int l, real_t r) const {
    if (attribute > NumLayers()) {
      if (c != Phi) {
        return 0.0;
      }
      const Layer& top = layers_.back();
      const real_t a = top.r_outer;
      return Interpolate(top, top.values[3 * l + c], a) *
             std::pow(a / std::max(r, a), l + 1);
    }
    return Interpolate(layers_[attribute - 1],
                       layers_[attribute - 1].values[3 * l + c], r);
  }

  // The gravity of the model at the radius r, in the layer of the given
  // attribute or outside the body.
  real_t Gravity(int attribute, real_t r) const {
    if (attribute > NumLayers()) {
      const Layer& top = layers_.back();
      const real_t a = top.r_outer, s = a / std::max(r, a);
      return Interpolate(top, top.gravity, a) * s * s;
    }
    return Interpolate(layers_[attribute - 1], layers_[attribute - 1].gravity,
                       r);
  }

 private:
  struct Layer {
    real_t r_inner = 0.0, r_outer = 0.0;
    bool fluid = false;
    Vector r, w, gravity;
    // U, V and phi of degree l at 3 l, 3 l + 1 and 3 l + 2.
    std::vector<Vector> values;
  };

  static real_t Interpolate(const Layer& layer, const Vector& f, real_t r) {
    const int n = layer.r.Size();
    r = std::min(std::max(r, layer.r[0]), layer.r[n - 1]);
    real_t num = 0.0, den = 0.0;
    for (int j = 0; j < n; j++) {
      const real_t d = r - layer.r[j];
      if (d == 0.0) {
        return f[j];
      }
      const real_t q = layer.w[j] / d;
      num += q * f[j];
      den += q;
    }
    return num / den;
  }

  int lmax_ = 0;
  std::vector<Layer> layers_;
};

// The reference displacement of a load with the given coefficients.
class ReferenceDisplacement : public VectorCoefficient {
 public:
  ReferenceDisplacement(const RadialReference& reference,
                        const SurfaceHarmonics& basis, const Vector& c)
      : VectorCoefficient(basis.Dim()),
        reference_(reference),
        basis_(basis),
        c_(c) {
    MFEM_VERIFY(basis.MaxDegree() <= reference.MaxDegree(),
                "ReferenceDisplacement: the reference stops at degree "
                    << reference.MaxDegree() << ".");
  }

  void Eval(Vector& u, ElementTransformation& T,
            const IntegrationPoint& ip) override {
    T.Transform(ip, x_);
    const real_t r = x_.Norml2();
    basis_.EvalWithGradient(x_, Y_, gradY_);
    u.SetSize(vdim);
    u = 0.0;
    real_t radial = 0.0;
    int degree = -1;
    real_t U = 0.0, V = 0.0;
    for (int i = 0; i < c_.Size(); i++) {
      if (c_[i] == 0.0) {
        continue;
      }
      if (basis_.Degree(i) != degree) {
        degree = basis_.Degree(i);
        U = reference_.Eval(T.Attribute, RadialReference::U, degree, r);
        V = reference_.Eval(T.Attribute, RadialReference::V, degree, r);
      }
      radial += c_[i] * U * Y_[i];
      for (int k = 0; k < vdim; k++) {
        u[k] += c_[i] * V * gradY_(k, i);
      }
    }
    if (r > 0.0) {
      u.Add(radial / r, x_);
    }
  }

 private:
  const RadialReference& reference_;
  const SurfaceHarmonics& basis_;
  Vector c_, x_, Y_;
  DenseMatrix gradY_;
};

// The reference potential perturbation of a load with the given
// coefficients, in the body and outside it.
class ReferencePotential : public Coefficient {
 public:
  ReferencePotential(const RadialReference& reference,
                     const SurfaceHarmonics& basis, const Vector& c)
      : reference_(reference), basis_(basis), c_(c) {
    MFEM_VERIFY(basis.MaxDegree() <= reference.MaxDegree(),
                "ReferencePotential: the reference stops at degree "
                    << reference.MaxDegree() << ".");
  }

  real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    T.Transform(ip, x_);
    const real_t r = x_.Norml2();
    basis_.Eval(x_, Y_);
    real_t phi = 0.0, radial = 0.0;
    int degree = -1;
    for (int i = 0; i < c_.Size(); i++) {
      if (c_[i] == 0.0) {
        continue;
      }
      if (basis_.Degree(i) != degree) {
        degree = basis_.Degree(i);
        radial = reference_.Eval(T.Attribute, RadialReference::Phi, degree, r);
      }
      phi += c_[i] * radial * Y_[i];
    }
    return phi;
  }

 private:
  const RadialReference& reference_;
  const SurfaceHarmonics& basis_;
  Vector c_, x_, Y_;
};

// The potential perturbation of the rigid translation t of the body,
// -t . grad Phi_0 = -g(r) t . x^, with the gravity of the reference.
class TranslationPotential : public Coefficient {
 public:
  TranslationPotential(const RadialReference& reference, const Vector& t)
      : reference_(reference), t_(t) {}

  real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    T.Transform(ip, x_);
    const real_t r = x_.Norml2();
    return r > 0.0 ? -reference_.Gravity(T.Attribute, r) * (t_ * x_) / r : 0.0;
  }

 private:
  const RadialReference& reference_;
  Vector t_, x_;
};

}  // namespace benchmark
