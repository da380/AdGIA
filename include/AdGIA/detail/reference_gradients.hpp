/**
 * @file reference_gradients.hpp
 * @brief The reference gradients of an element's shape functions, kept
 * by quadrature point.
 */

#pragma once

#include <cstddef>
#include <functional>
#include <unordered_map>
#include <utility>

#include "mfem.hpp"

namespace AdGIA {
namespace detail {

/**
 * @brief The gradients of a finite element's shape functions in
 * reference coordinates at given points, each computed once.
 *
 * At a point of a quadrature rule these are the same in every element
 * of one kind, and for a simplex of high order they are the costly part
 * of an integrand's evaluation. An integrator or a coefficient holds one
 * of these and asks it in place of FiniteElement::CalcDShape.
 *
 * An entry is found by the element's and the point's addresses and
 * checked against the point's coordinates and the element's order,
 * geometry and size, so a point held by value at a reused address is
 * not mistaken for another.
 */
class ReferenceGradients {
 public:
  /// The gradients (dof by dimension) of @p fe at @p ip.
  const mfem::DenseMatrix& At(const mfem::FiniteElement& fe,
                              const mfem::IntegrationPoint& ip) {
    if (held_.size() > max_held_) {
      held_.clear();
    }
    auto& entry = held_[{&fe, &ip}];
    if (entry.dshape.Height() != fe.GetDof() ||
        entry.order != fe.GetOrder() || entry.geometry != fe.GetGeomType() ||
        entry.x != ip.x || entry.y != ip.y || entry.z != ip.z) {
      entry.order = fe.GetOrder();
      entry.geometry = fe.GetGeomType();
      entry.x = ip.x;
      entry.y = ip.y;
      entry.z = ip.z;
      entry.dshape.SetSize(fe.GetDof(), fe.GetDim());
      fe.CalcDShape(ip, entry.dshape);
    }
    return entry.dshape;
  }

 private:
  using Key = std::pair<const mfem::FiniteElement*,
                        const mfem::IntegrationPoint*>;
  struct Hash {
    std::size_t operator()(const Key& key) const {
      const auto a = std::hash<const void*>()(key.first);
      const auto b = std::hash<const void*>()(key.second);
      return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
    }
  };
  struct Entry {
    int order = -1;
    int geometry = -1;
    mfem::real_t x = 0, y = 0, z = 0;
    mfem::DenseMatrix dshape;
  };
  static constexpr std::size_t max_held_ = 8192;
  std::unordered_map<Key, Entry, Hash> held_;
};

}  // namespace detail
}  // namespace AdGIA
