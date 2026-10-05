
#include "AdGIA/mesh.hpp"

#include <optional>

namespace AdGIA {

mfem::Array<int> ExternalBoundaryMarker(mfem::Mesh* mesh) {
  auto bdr_marker = mfem::Array<int>(mesh->bdr_attributes.Max());
  bdr_marker = 0;
  mesh->MarkExternalBoundaries(bdr_marker);
  return bdr_marker;
}

mfem::Array<int> AllDomainsMarker(mfem::Mesh* mesh) {
  auto dom_marker = mfem::Array<int>(mesh->attributes.Max());
  dom_marker = 1;
  return dom_marker;
}

mfem::Array<int> AllBoundariesMarker(mfem::Mesh* mesh) {
  auto bdr_marker = mfem::Array<int>(mesh->bdr_attributes.Max());
  bdr_marker = 1;
  return bdr_marker;
}

std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::Mesh* mesh, const mfem::Array<int>& bdr_marker,
    const mfem::Vector& x0) {
  using namespace mfem;

  const auto rtol = 1e-6;

  auto dim = mesh->Dimension();
  auto fec = H1_FECollection(1, dim);
  auto fes = FiniteElementSpace(mesh, &fec);

  auto radius = real_t{-1};
  auto x = Vector(dim);
  auto found = 0;
  auto same = 1;
  for (auto i = 0; i < mesh->GetNBE(); i++) {
    if (same == 0) break;
    const auto attr = mesh->GetBdrAttribute(i);

    if (bdr_marker[attr - 1] == 1) {
      found = 1;
      const auto* fe = fes.GetBE(i);
      auto* Trans = fes.GetBdrElementTransformation(i);
      const auto& ir = fe->GetNodes();
      for (auto j = 0; j < ir.GetNPoints(); j++) {
        const auto& ip = ir.IntPoint(j);
        Trans->SetIntPoint(&ip);
        Trans->Transform(ip, x);
        auto d = x.DistanceTo(x0);
        if (radius < 0) {
          radius = d;
        } else {
          same = static_cast<int>(std::abs(radius - d) < rtol * radius);
          if (same == 0) break;
        }
      }
    }
  }
  return {found, same, radius};
}

std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::Mesh* mesh, const mfem::Array<int>& bdr_marker) {
  auto x0 = mfem::Vector(mesh->Dimension());
  x0 = 0.0;
  return SphericalBoundaryRadius(mesh, bdr_marker, x0);
}

std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::Mesh* mesh, const mfem::Vector& x0) {
  return SphericalBoundaryRadius(mesh, ExternalBoundaryMarker(mesh), x0);
}

std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(mfem::Mesh* mesh) {
  return SphericalBoundaryRadius(mesh, ExternalBoundaryMarker(mesh));
}

#ifdef MFEM_USE_MPI
std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::ParMesh* mesh, const mfem::Array<int>& bdr_marker,
    const mfem::Vector& x0) {
  using namespace mfem;

  const auto rtol = 1e-6;
  auto comm = mesh->GetComm();
  auto rank = mesh->GetMyRank();
  auto size = mesh->GetNRanks();

  auto [local_found, local_same, local_radius] =
      SphericalBoundaryRadius(dynamic_cast<Mesh*>(mesh), bdr_marker, x0);

  // As in the serial version: radius -1 when no rank holds the boundary.
  real_t radius = -1;
  auto found = 0;
  auto same = 1;

  if (rank == 0) {
    auto founds = std::vector<int>(size);
    auto sames = std::vector<int>(size);
    auto radii = std::vector<real_t>(size);

    MPI_Gather(&local_found, 1, MPI_INT, founds.data(), 1, MPI_INT, 0, comm);
    MPI_Gather(&local_same, 1, MPI_INT, sames.data(), 1, MPI_INT, 0, comm);
    MPI_Gather(&local_radius, 1, MFEM_MPI_REAL_T, radii.data(), 1,
               MFEM_MPI_REAL_T, 0, comm);

    // The boundary is a sphere only if every rank holding part of it
    // found its part spherical, all with the same radius.
    for (auto i = 0; i < size; i++) {
      if (founds[i] == 1) {
        found = 1;
        radius = radii[i];
        break;
      }
    }

    for (auto i = 0; i < size; i++) {
      if (founds[i] == 1) {
        if (sames[i] == 0 || std::abs(radius - radii[i]) > rtol * radius) {
          same = 0;
          break;
        }
      }
    }

  } else {
    MPI_Gather(&local_found, 1, MPI_INT, nullptr, 0, MPI_INT, 0, comm);
    MPI_Gather(&local_same, 1, MPI_INT, nullptr, 0, MPI_INT, 0, comm);
    MPI_Gather(&local_radius, 1, MFEM_MPI_REAL_T, nullptr, 0, MFEM_MPI_REAL_T,
               0, comm);
  }

  MPI_Bcast(&found, 1, MPI_INT, 0, comm);
  MPI_Bcast(&same, 1, MPI_INT, 0, comm);
  MPI_Bcast(&radius, 1, MFEM_MPI_REAL_T, 0, comm);

  return {found, same, radius};
}

std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::ParMesh* mesh, const mfem::Array<int>& bdr_marker) {
  auto x0 = mfem::Vector(mesh->Dimension());
  x0 = 0.0;
  return SphericalBoundaryRadius(mesh, bdr_marker, x0);
}

std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::ParMesh* mesh, const mfem::Vector& x0) {
  auto bdr_marker = ExternalBoundaryMarker(mesh);
  return SphericalBoundaryRadius(mesh, bdr_marker, x0);
}

std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::ParMesh* mesh) {
  auto bdr_marker = ExternalBoundaryMarker(mesh);
  return SphericalBoundaryRadius(mesh, bdr_marker);
}

#endif

mfem::Vector MeshCentroid(mfem::Mesh* mesh, const mfem::Array<int>& dom_marker,
                          int order) {
  using namespace mfem;
  // The linear forms keep a pointer to a non-const marker.
  auto marker = Array<int>(dom_marker);
  auto dim = mesh->Dimension();
  auto x0 = Vector(dim);
  x0 = 0.0;

  auto fec = L2_FECollection(order, dim);
  auto fes = FiniteElementSpace(mesh, &fec);

  auto c = ConstantCoefficient(1);
  auto f = LinearForm(&fes);
  f.AddDomainIntegrator(new DomainLFIntegrator(c), marker);
  f.Assemble();

  auto u = GridFunction(&fes);
  u.ProjectCoefficient(c);
  auto m = f(u);

  for (auto i = 0; i < dim; i++) {
    auto f = LinearForm(&fes);
    auto c = FunctionCoefficient([i](const Vector& x) { return x[i]; });
    f.AddDomainIntegrator(new DomainLFIntegrator(c), marker);
    f.Assemble();
    x0(i) = f(u) / m;
  }

  return x0;
}

mfem::Vector MeshCentroid(mfem::Mesh* mesh, int order) {
  return MeshCentroid(mesh, AllDomainsMarker(mesh), order);
}

#ifdef MFEM_USE_MPI

mfem::Vector MeshCentroid(mfem::ParMesh* mesh,
                          const mfem::Array<int>& dom_marker, int order) {
  using namespace mfem;
  // The linear forms keep a pointer to a non-const marker.
  auto marker = Array<int>(dom_marker);
  auto dim = mesh->Dimension();
  auto x0 = Vector(dim);
  x0 = 0.0;

  auto fec = L2_FECollection(order, dim);
  auto fes = ParFiniteElementSpace(mesh, &fec);

  auto c = ConstantCoefficient(1);
  auto f = ParLinearForm(&fes);
  f.AddDomainIntegrator(new DomainLFIntegrator(c), marker);
  f.Assemble();

  auto u = ParGridFunction(&fes);
  u.ProjectCoefficient(c);
  auto m = f(u);

  for (auto i = 0; i < dim; i++) {
    auto f = ParLinearForm(&fes);
    auto c = FunctionCoefficient([i](const Vector& x) { return x[i]; });
    f.AddDomainIntegrator(new DomainLFIntegrator(c), marker);
    f.Assemble();
    x0(i) = f(u) / m;
  }

  return x0;
}

mfem::Vector MeshCentroid(mfem::ParMesh* mesh, int order) {
  return MeshCentroid(mesh, AllDomainsMarker(mesh), order);
}
#endif

void SphericalMeshHelper::SetBoundaryMarker(mfem::Mesh* mesh) {
  x0_ = MeshCentroid(mesh);
  bdr_marker_ = ExternalBoundaryMarker(mesh);
  auto [found, same, radius] = SphericalBoundaryRadius(mesh, bdr_marker_, x0_);
  assert(found == 1 && same == 1);
  bdr_radius_ = radius;
}

#ifdef MFEM_USE_MPI
void SphericalMeshHelper::SetBoundaryMarker(mfem::ParMesh* mesh) {
  x0_ = MeshCentroid(mesh);
  bdr_marker_ = ExternalBoundaryMarker(mesh);
  auto [found, same, radius] = SphericalBoundaryRadius(mesh, bdr_marker_, x0_);
  assert(found == 1 && same == 1);
  bdr_radius_ = radius;
}
#endif

#ifdef MFEM_USE_MPI

std::tuple<MPI_Comm, bool, int> SplitBoundaryCommunicator(
    mfem::ParMesh* mesh, const mfem::Array<int>& bdr_marker) {
  MPI_Comm parent_comm = mesh->GetComm();
  int my_rank = mesh->GetMyRank();

  int local_has_boundary = 0;
  for (int i = 0; i < mesh->GetNBE(); i++) {
    int attr = mesh->GetBdrAttribute(i);
    if (bdr_marker[attr - 1] == 1) {
      local_has_boundary = 1;
      break;
    }
  }
  bool is_boundary_rank = (local_has_boundary == 1);
  int color = is_boundary_rank ? 1 : MPI_UNDEFINED;

  MPI_Comm bdr_comm;
  MPI_Comm_split(parent_comm, color, my_rank, &bdr_comm);

  int bdr_local_rank = -1;
  if (is_boundary_rank) {
    MPI_Comm_rank(bdr_comm, &bdr_local_rank);
  }

  int bdr_root_global_rank = -1;
  if (bdr_local_rank == 0) {
    bdr_root_global_rank = my_rank;
  }

  MPI_Allreduce(MPI_IN_PLACE, &bdr_root_global_rank, 1, MPI_INT, MPI_MAX,
                parent_comm);

  return {bdr_comm, is_boundary_rank, bdr_root_global_rank};
}

std::tuple<MPI_Comm, bool, int> SplitBoundaryCommunicator(mfem::ParMesh* mesh) {
  auto bdr_marker = ExternalBoundaryMarker(mesh);
  return SplitBoundaryCommunicator(mesh, bdr_marker);
}

#endif

}  // namespace AdGIA