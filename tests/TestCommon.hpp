#pragma once

#include <gtest/gtest.h>

#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <tuple>

#include "mfem.hpp"
#include "AdGIA.hpp"

using namespace mfem;
using namespace AdGIA;

// Test parameter (dim, order, elementType) of the integrator tests.
using DimOrderTypeTuple = std::tuple<int, int, int>;

// The unit interval, square or cube with 20 elements per side (spacing
// 0.05); elementType 0 gives simplices, 1 tensor-product elements.
inline Mesh MakeMesh(int dim, int elementType) {
  if (dim == 1) {
    return Mesh::MakeCartesian1D(20);
  } else if (dim == 2) {
    return Mesh::MakeCartesian2D(
        20, 20, elementType == 0 ? Element::TRIANGLE : Element::QUADRILATERAL);
  } else {
    return Mesh::MakeCartesian3D(
        20, 20, 20,
        elementType == 0 ? Element::TETRAHEDRON : Element::HEXAHEDRON);
  }
}

// Vector and matrix with independent standard normal entries. The
// generator is seeded with a fixed constant, so every call returns the
// same values and failures are reproducible; the checks that use them
// hold for any input.
inline Vector RandomVector(int dim) {
  std::mt19937 gen(12345u);
  std::normal_distribution<> distrib(0, 1);

  auto v = Vector(dim);
  for (auto j = 0; j < dim; j++) {
    v(j) = distrib(gen);
  }
  return v;
}

inline DenseMatrix RandomMatrix(int dim) {
  std::mt19937 gen(12345u);
  std::normal_distribution<> distrib(0, 1);

  auto A = DenseMatrix(dim);
  for (auto j = 0; j < dim; j++) {
    for (auto i = 0; i < dim; i++) {
      A(i, j) = distrib(gen);
    }
  }
  return A;
}

// Entrywise max |A - B| for finalized sparse matrices of equal size.
inline double MaxDiff(const SparseMatrix& A, const SparseMatrix& B) {
  EXPECT_EQ(A.Height(), B.Height());
  EXPECT_EQ(A.Width(), B.Width());
  if (A.Height() != B.Height() || A.Width() != B.Width()) {
    return std::numeric_limits<double>::infinity();
  }
  std::unique_ptr<SparseMatrix> D(mfem::Add(1.0, A, -1.0, B));
  return D->MaxNorm();
}
