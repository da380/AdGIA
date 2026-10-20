#pragma once

/*
  Helpers shared by the fault-problem tests (serial gtests and the MPI
  program): the faulted-box mesh and its record (meshes/faulted_box.py),
  the markers of the two configurations (the contained fault, and the
  whole cut as fault for the rigid-block test), a tapered dip-slip
  coefficient and the rigid-block error.
*/

#include <cmath>
#include <numbers>
#include <string>

#include "mfem.hpp"
#include "AdGIA.hpp"
#include "planetmodel/json.hpp"

namespace fault_test {

using namespace mfem;

inline std::string MeshFile(int dim) {
  return "../data/faulted_box_" + std::to_string(dim) + "d.msh";
}
inline std::string RecordFile(int dim) {
  return "../data/faulted_box_" + std::to_string(dim) + "d.json";
}

// The record's fault frame and attribute dictionary (width and strike
// direction in 3-D only).
struct FaultedBox {
  int dim = 0;
  real_t length = 0.0, width = 0.0;
  Vector centre, normal, down_dip, strike;
  int footwall = 0, hanging_wall = 0;
  int fault = 0, extension = 0, surface = 0, clamped_footwall = 0,
      clamped_hanging_wall = 0;
};

inline Vector JsonVector(const planetmodel::Json& j) {
  auto v = Vector(static_cast<int>(j.Size()));
  for (int i = 0; i < v.Size(); i++) {
    v(i) = j.At(static_cast<std::size_t>(i)).AsNumber();
  }
  return v;
}

inline FaultedBox ReadFaultedBox(const std::string& path) {
  const auto record = planetmodel::ReadJsonFile(path);
  auto box = FaultedBox();
  box.dim = record.At("dimension").AsInt();
  const auto& fault = record.At("fault");
  box.length = fault.At("length").AsNumber();
  box.centre = JsonVector(fault.At("centre"));
  box.normal = JsonVector(fault.At("normal"));
  box.down_dip = JsonVector(fault.At("down_dip"));
  if (box.dim == 3) {
    box.width = fault.At("width").AsNumber();
    box.strike = JsonVector(fault.At("strike_direction"));
  }
  const auto& attrs = record.At("attributes");
  const auto& dom = attrs.At("domain");
  const auto& bdr = attrs.At("boundary");
  auto attr = [](const planetmodel::Json& j, const char* key) {
    return j.At(key).AsInt();
  };
  box.footwall = attr(dom, "footwall");
  box.hanging_wall = attr(dom, "hanging_wall");
  box.fault = attr(bdr, "fault");
  box.extension = attr(bdr, "fault_extension");
  box.surface = attr(bdr, "surface");
  box.clamped_footwall = attr(bdr, "clamped_footwall");
  box.clamped_hanging_wall = attr(bdr, "clamped_hanging_wall");
  return box;
}

inline Array<int> Marker(int size, std::initializer_list<int> attrs) {
  auto m = Array<int>(size);
  m = 0;
  for (auto a : attrs) {
    m[a - 1] = 1;
  }
  return m;
}

// Thrust dip-slip (the hanging wall up-dip) of peak `peak` with the cos^2
// taper of examples/fault_box.hpp, zero at the fault edge (in 3-D along
// dip and along strike).
inline VectorFunctionCoefficient TaperedSlip(const FaultedBox& box,
                                             real_t peak) {
  const auto b = box;
  return VectorFunctionCoefficient(
      b.dim, [b, peak](const Vector& x, Vector& s) {
        auto taper = [](real_t t) {
          return std::abs(t) < 0.5
                     ? std::pow(std::cos(std::numbers::pi_v<real_t> * t), 2)
                     : 0.0;
        };
        real_t xi = 0.0, eta = 0.0;
        for (int i = 0; i < b.dim; i++) {
          xi += (x(i) - b.centre(i)) * b.down_dip(i);
          if (b.dim == 3) {
            eta += (x(i) - b.centre(i)) * b.strike(i);
          }
        }
        auto w = taper(xi / b.length);
        if (b.dim == 3) {
          w *= taper(eta / b.width);
        }
        s.SetSize(b.dim);
        for (int i = 0; i < b.dim; i++) {
          s(i) = -peak * w * b.down_dip(i);
        }
      });
}

// max |u_w - u_exact| over the local vdofs for uniform slip s across a
// cut that separates the body, the footwall clamped: the hanging wall
// translates rigidly, so u_w = s on the hanging wall's dofs off the cut
// and zero elsewhere (on the cut, the footwall and the clamped boundary).
inline real_t RigidBlockError(const GridFunction& u, const Array<int>& cut,
                              int hanging_wall_attr, const Vector& s) {
  const auto& fes = *u.FESpace();
  auto on_cut = Array<int>();
  fes.GetEssentialVDofs(cut, on_cut);
  auto on_hw = Array<int>(fes.GetNDofs());
  on_hw = 0;
  auto dofs = Array<int>();
  for (int e = 0; e < fes.GetNE(); e++) {
    if (fes.GetMesh()->GetAttribute(e) == hanging_wall_attr) {
      fes.GetElementDofs(e, dofs);
      for (auto d : dofs) {
        on_hw[d >= 0 ? d : -1 - d] = 1;
      }
    }
  }
  real_t err = 0.0;
  for (int i = 0; i < fes.GetNDofs(); i++) {
    for (int c = 0; c < fes.GetVDim(); c++) {
      const auto v = fes.DofToVDof(i, c);
      const auto expected = on_hw[i] && !on_cut[v] ? s(c) : 0.0;
      err = std::max(err, std::abs(u(v) - expected));
    }
  }
  return err;
}

}  // namespace fault_test
