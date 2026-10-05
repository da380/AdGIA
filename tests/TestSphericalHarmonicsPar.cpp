/*
  Parallel tests for BoundaryHarmonicCoefficients on a ParSubMesh body
  inside a ParMesh ball. Run with 1, 2 and 4 ranks; a standalone MPI program
  that exits non-zero if any check fails. Every rank also builds the serial
  operator on the full mesh; the parallel coefficients (scalar, radial, and
  from a coefficient) must equal the serial ones, the radius must agree, and
  the parallel load vector must reproduce the duality relation globally.
*/

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>

#include "MixedProblemTestCommon.hpp"
#include "mfem.hpp"
#include "AdGIA.hpp"

using namespace mfem;
using namespace AdGIA;
using namespace self_grav_test;

namespace {

int num_checks = 0;
int num_fails = 0;

void Check(double err, double tol, const std::string& what) {
  num_checks++;
  if (!(err <= tol)) {
    num_fails++;
    if (Mpi::Root()) {
      std::cout << "FAIL: " << what << "  (err = " << err << ", tol = " << tol
                << ")\n";
    }
  }
}

double MaxDiff(const Vector& a, const Vector& b) {
  Vector d(a);
  d -= b;
  return d.Normlinf();
}

void RunCase(int dim, const std::string& label) {
  using BHC = BoundaryHarmonicCoefficients;
  const int L = dim == 2 ? 6 : 3;
  Mesh smesh(MeshFile(dim).c_str(), 1, 1);
  SubMesh sbody(SubMesh::CreateFromDomain(smesh, BodyMarker(smesh)));
  ParMesh pmesh(MPI_COMM_WORLD, smesh);
  ParSubMesh body(ParSubMesh::CreateFromDomain(pmesh, BodyMarker(pmesh)));
  H1_FECollection fec(2, dim);
  FiniteElementSpace sscalar(&sbody, &fec), svector(&sbody, &fec, dim);
  ParFiniteElementSpace pscalar(&body, &fec), pvector(&body, &fec, dim);
  auto surface = SurfaceMarker(sbody);
  auto psurface = SurfaceMarker(body);

  BHC s_scalar(sscalar, surface, L, BHC::Component::Scalar);
  BHC p_scalar(pscalar, psurface, L, BHC::Component::Scalar);
  BHC s_radial(svector, surface, L, BHC::Component::Radial);
  BHC p_radial(pvector, psurface, L, BHC::Component::Radial);
  Check(std::abs(p_scalar.Radius() - s_scalar.Radius()), 1e-12,
        label + ": radius");

  FunctionCoefficient f([dim](const Vector& x) {
    return std::exp(0.5 * x[0]) * (1.0 + 0.3 * x[1]) +
           (dim == 3 ? 0.7 * x[2] * x[0] : 0.0);
  });
  VectorFunctionCoefficient u(dim, [dim](const Vector& x, Vector& v) {
    v = x;
    v *= std::cos(x[0]) + 0.2 * x[1];
    v[0] += 0.3 * x[1];
  });

  Vector cs, cp;
  s_scalar.Coefficients(f, cs);
  p_scalar.Coefficients(f, cp);
  Check(MaxDiff(cs, cp), 1e-12 * cs.Normlinf(), label + ": scalar coefficient");
  s_radial.Coefficients(u, cs);
  p_radial.Coefficients(u, cp);
  Check(MaxDiff(cs, cp), 1e-12 * cs.Normlinf(), label + ": radial coefficient");

  GridFunction sg(&sscalar);
  sg.ProjectCoefficient(f);
  ParGridFunction pg(&pscalar);
  pg.ProjectCoefficient(f);
  s_scalar.Coefficients(sg, cs);
  p_scalar.Coefficients(pg, cp);
  Check(MaxDiff(cs, cp), 1e-12 * cs.Normlinf(), label + ": scalar field");

  GridFunction su(&svector);
  su.ProjectCoefficient(u);
  ParGridFunction pu(&pvector);
  pu.ProjectCoefficient(u);
  s_radial.Coefficients(su, cs);
  p_radial.Coefficients(pu, cp);
  Check(MaxDiff(cs, cp), 1e-12 * cs.Normlinf(), label + ": radial field");

  // Duality with the parallel load vector: the local dots sum over the
  // ranks to R^{d-1} c . coefficients(g), each boundary element once.
  Vector c(p_scalar.Size());
  for (int i = 0; i < c.Size(); i++) {
    c[i] = std::cos(1.3 * i + 0.2);
  }
  Vector b;
  p_scalar.LoadVector(c, b);
  double local = pg * b, global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  p_scalar.Coefficients(pg, cp);
  const double expected = std::pow(p_scalar.Radius(), dim - 1) * (c * cp);
  Check(std::abs(global - expected), 1e-12 * std::abs(expected),
        label + ": load vector duality");
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();
  for (int dim : {2, 3}) {
    RunCase(dim, "dim=" + std::to_string(dim));
  }
  if (Mpi::Root()) {
    if (num_fails == 0) {
      std::cout << "All " << num_checks << " checks passed on "
                << Mpi::WorldSize() << " ranks.\n";
    } else {
      std::cout << num_fails << " of " << num_checks << " checks FAILED on "
                << Mpi::WorldSize() << " ranks.\n";
    }
  }
  return num_fails == 0 ? 0 : 1;
}
