
#include "AdGIA/mesh.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace AdGIA {

namespace {

// Projects nodes lying within the relative capture window of a listed
// radius back onto that sphere (about the origin). Acts on the nodes grid
// function when the mesh is curved and on the vertices otherwise; the
// projection is pointwise and deterministic, so shared parallel nodes
// stay consistent without communication.
void SnapToSpheres(mfem::Mesh& mesh, const ShorelineRefinementOptions& opt) {
  if (opt.snap_radii.empty()) {
    return;
  }
  auto snap = [&opt](mfem::real_t* x, int sdim) {
    mfem::real_t r2 = 0.0;
    for (int c = 0; c < sdim; c++) {
      r2 += x[c] * x[c];
    }
    const mfem::real_t r = std::sqrt(r2);
    if (r == 0.0) {
      return;
    }
    for (const auto R : opt.snap_radii) {
      if (std::abs(r - R) < opt.snap_tol * R) {
        const mfem::real_t s = R / r;
        for (int c = 0; c < sdim; c++) {
          x[c] *= s;
        }
        return;
      }
    }
  };
  const int sdim = mesh.SpaceDimension();
  if (mfem::GridFunction* nodes = mesh.GetNodes()) {
    // Nodes are stored byNODES or byVDIM; go through the dof points.
    const mfem::FiniteElementSpace& nfes = *nodes->FESpace();
    const int nd = nfes.GetNDofs();
    mfem::Vector p(sdim);
    for (int i = 0; i < nd; i++) {
      for (int c = 0; c < sdim; c++) {
        p[c] = (*nodes)(nfes.DofToVDof(i, c));
      }
      snap(p.GetData(), sdim);
      for (int c = 0; c < sdim; c++) {
        (*nodes)(nfes.DofToVDof(i, c)) = p[c];
      }
    }
  } else {
    for (int i = 0; i < mesh.GetNV(); i++) {
      snap(mesh.GetVertex(i), sdim);
    }
  }
}

}  // namespace

int RefineToShoreline(mfem::Mesh& mesh, mfem::Coefficient& ocean,
                      const ShorelineRefinementOptions& options) {
  // Conforming tet bisection needs refinement-marked tets. Meshes are
  // marked at load (Mesh(file, 1, 1)) but UniformRefinement clears the
  // flags, so shoreline refinement goes FIRST and uniform refinements
  // after; fail here with the recipe rather than deep inside MFEM.
  if (mesh.Dimension() == 3 && mesh.GetNE() > 0 &&
      mesh.GetElement(0)->GetType() == mfem::Element::TETRAHEDRON) {
    MFEM_VERIFY(
        static_cast<mfem::Tetrahedron*>(mesh.GetElement(0))
                ->GetRefinementFlag() != 0,
        "RefineToShoreline: tets are not marked for refinement; call it "
        "straight after loading the mesh, before UniformRefinement");
  }
  const mfem::real_t lo = options.threshold - options.band;
  const mfem::real_t hi = options.threshold + options.band;
  int total = 0;
  for (int pass = 0; pass < options.levels; pass++) {
    mfem::Array<mfem::Refinement> refs;
    mfem::Vector pt;
    mfem::Vector centre;
    for (int i = 0; i < mesh.GetNE(); i++) {
      const mfem::Geometry::Type geom = mesh.GetElementBaseGeometry(i);
      mfem::ElementTransformation* T = mesh.GetElementTransformation(i);
      const mfem::IntegrationRule& ir =
          mfem::GlobGeometryRefiner.Refine(geom, options.subdivisions, 1)
              ->RefPts;
      mfem::real_t cmin = std::numeric_limits<mfem::real_t>::max();
      mfem::real_t cmax = std::numeric_limits<mfem::real_t>::lowest();
      for (int j = 0; j < ir.GetNPoints(); j++) {
        const mfem::IntegrationPoint& ip = ir.IntPoint(j);
        T->Transform(ip, pt);
        const mfem::real_t r = pt.Norml2();
        if (r < options.r_min || r > options.r_max) {
          continue;
        }
        T->SetIntPoint(&ip);
        const mfem::real_t c = ocean.Eval(*T, ip);
        cmin = std::min(cmin, c);
        cmax = std::max(cmax, c);
      }
      // Marked when the element's sampled range meets the band about the
      // flotation threshold (cmin > cmax means no sample in the window);
      // with a geometric width, when the element straddles the threshold
      // or its estimated angular distance to the shoreline is below it.
      const mfem::real_t h = mesh.GetElementSize(i);
      bool mark = false;
      if (cmin <= cmax) {
        if (options.width > 0.0) {
          if (cmin <= options.threshold && options.threshold <= cmax) {
            mark = true;
          } else if (cmax > cmin) {
            mesh.GetElementCenter(i, centre);
            const mfem::real_t rc = std::max(centre.Norml2(), options.r_min);
            const mfem::real_t ang = h / rc;
            const mfem::real_t gap =
                std::min(std::abs(cmin - options.threshold),
                         std::abs(cmax - options.threshold));
            mark = gap * ang / (cmax - cmin) <= options.width;
          }
        } else {
          mark = cmax >= lo && cmin <= hi;
        }
      }
      if (mark && options.target_h > 0.0 &&
          h <= options.target_h) {
        mark = false;
      }
      if (mark) {
        refs.Append(mfem::Refinement(i));
      }
    }
    int refined = refs.Size();
#ifdef MFEM_USE_MPI
    if (auto* pmesh = dynamic_cast<mfem::ParMesh*>(&mesh)) {
      MPI_Allreduce(MPI_IN_PLACE, &refined, 1, MPI_INT, MPI_SUM,
                    pmesh->GetComm());
    }
#endif
    if (refined == 0) {
      break;
    }
    mesh.GeneralRefinement(refs, -1, 0);
    SnapToSpheres(mesh, options);
    total += refined;
  }
  return total;
}

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
    mfem::Mesh* mesh, const mfem::Array<int>& bdr_marker, const mfem::Vector& x0,
    mfem::real_t* deviation) {
  using namespace mfem;

  auto dim = mesh->Dimension();
  auto fec = H1_FECollection(1, dim);
  auto fes = FiniteElementSpace(mesh, &fec);

  auto radius = real_t{-1};
  auto dev = real_t{0};
  auto x = Vector(dim);
  auto found = 0;
  for (auto i = 0; i < mesh->GetNBE(); i++) {
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
          dev = std::max(dev, std::abs(radius - d));
        }
      }
    }
  }
  const auto rel = radius > 0 ? dev / radius : real_t{0};
  if (deviation) {
    *deviation = rel;
  }
  const auto same = static_cast<int>(rel <= kSphericalBoundaryRTol);
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
    const mfem::Vector& x0, mfem::real_t* deviation) {
  using namespace mfem;

  auto comm = mesh->GetComm();
  auto rank = mesh->GetMyRank();
  auto size = mesh->GetNRanks();

  real_t local_dev = 0;
  auto [local_found, local_same, local_radius] = SphericalBoundaryRadius(
      dynamic_cast<Mesh*>(mesh), bdr_marker, x0, &local_dev);

  // As in the serial version: radius -1 when no rank holds the boundary.
  real_t radius = -1;
  real_t dev = 0;
  auto found = 0;
  auto same = 1;

  if (rank == 0) {
    auto founds = std::vector<int>(size);
    auto devs = std::vector<real_t>(size);
    auto radii = std::vector<real_t>(size);

    MPI_Gather(&local_found, 1, MPI_INT, founds.data(), 1, MPI_INT, 0, comm);
    MPI_Gather(&local_dev, 1, MFEM_MPI_REAL_T, devs.data(), 1,
               MFEM_MPI_REAL_T, 0, comm);
    MPI_Gather(&local_radius, 1, MFEM_MPI_REAL_T, radii.data(), 1,
               MFEM_MPI_REAL_T, 0, comm);

    // The boundary is a sphere only if every rank holding part of it
    // found its part spherical, all with the same radius; the verdict
    // pools the per-rank spreads with the cross-rank radius spread.
    for (auto i = 0; i < size; i++) {
      if (founds[i] == 1) {
        found = 1;
        radius = radii[i];
        break;
      }
    }

    for (auto i = 0; i < size; i++) {
      if (founds[i] == 1) {
        dev = std::max(dev, devs[i]);
        if (radius > 0) {
          dev = std::max(dev, std::abs(radius - radii[i]) / radius);
        }
      }
    }
    same = static_cast<int>(dev <= kSphericalBoundaryRTol);

  } else {
    MPI_Gather(&local_found, 1, MPI_INT, nullptr, 0, MPI_INT, 0, comm);
    MPI_Gather(&local_dev, 1, MFEM_MPI_REAL_T, nullptr, 0, MFEM_MPI_REAL_T, 0,
               comm);
    MPI_Gather(&local_radius, 1, MFEM_MPI_REAL_T, nullptr, 0, MFEM_MPI_REAL_T,
               0, comm);
  }
  (void)local_same;  // the pooled deviation carries the verdict

  MPI_Bcast(&found, 1, MPI_INT, 0, comm);
  MPI_Bcast(&same, 1, MPI_INT, 0, comm);
  MPI_Bcast(&radius, 1, MFEM_MPI_REAL_T, 0, comm);
  MPI_Bcast(&dev, 1, MFEM_MPI_REAL_T, 0, comm);
  if (deviation) {
    *deviation = dev;
  }

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
    auto fi = LinearForm(&fes);
    auto xi = FunctionCoefficient([i](const Vector& x) { return x[i]; });
    fi.AddDomainIntegrator(new DomainLFIntegrator(xi), marker);
    fi.Assemble();
    x0(i) = fi(u) / m;
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
    auto fi = ParLinearForm(&fes);
    auto xi = FunctionCoefficient([i](const Vector& x) { return x[i]; });
    fi.AddDomainIntegrator(new DomainLFIntegrator(xi), marker);
    fi.Assemble();
    x0(i) = fi(u) / m;
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
  mfem::real_t dev = 0;
  auto [found, same, radius] =
      SphericalBoundaryRadius(mesh, bdr_marker_, x0_, &dev);
  MFEM_VERIFY(found == 1 && same == 1,
              "SphericalMeshHelper: the external boundary is not a single "
              "sphere about the mesh centroid (relative radius spread "
                  << dev << ", tolerance " << kSphericalBoundaryRTol << ")");
  bdr_radius_ = radius;
}

#ifdef MFEM_USE_MPI
void SphericalMeshHelper::SetBoundaryMarker(mfem::ParMesh* mesh) {
  x0_ = MeshCentroid(mesh);
  bdr_marker_ = ExternalBoundaryMarker(mesh);
  mfem::real_t dev = 0;
  auto [found, same, radius] =
      SphericalBoundaryRadius(mesh, bdr_marker_, x0_, &dev);
  MFEM_VERIFY(found == 1 && same == 1,
              "SphericalMeshHelper: the external boundary is not a single "
              "sphere about the mesh centroid (relative radius spread "
                  << dev << ", tolerance " << kSphericalBoundaryRTol << ")");
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