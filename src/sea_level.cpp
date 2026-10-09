/**
 * @file sea_level.cpp
 * @brief Implementation of SeaLevelOperator (the WP1 surface layer).
 */

#include "AdGIA/sea_level.hpp"

#include <cmath>
#include <cstddef>
#include <fstream>
#include <vector>

#include "AdGIA/detail/fem_factory.hpp"

namespace AdGIA {

using namespace mfem;

namespace {

// Boundary attributes selected by a marker, as the list
// (Par)SubMesh::CreateFromBoundary expects.
Array<int> MarkedAttributes(const Array<int>& marker) {
  Array<int> attrs;
  for (int a = 1; a <= marker.Size(); a++) {
    if (marker[a - 1]) {
      attrs.Append(a);
    }
  }
  return attrs;
}

// The pointwise ocean indicator (eq. 29) from the surface fields.
class OceanIndicatorCoefficient : public Coefficient {
 public:
  OceanIndicatorCoefficient(const GridFunction& sl0, const GridFunction& ice,
                            real_t rho_water, real_t rho_ice)
      : sl0_(&sl0), ice_(&ice), rho_water_(rho_water), rho_ice_(rho_ice) {}

  real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    return rho_water_ * sl0_->GetValue(T, ip) -
                       rho_ice_ * ice_->GetValue(T, ip) >
                   0.0
               ? 1.0
               : 0.0;
  }

 private:
  const GridFunction* sl0_;
  const GridFunction* ice_;
  real_t rho_water_, rho_ice_;
};

}  // namespace

SeaLevelOperator::SeaLevelOperator(FiniteElementSpace& body_fes,
                                   const Array<int>& surface_marker,
                                   int order)
    : body_fes_(&body_fes) {
  body_mesh_ = body_fes.GetMesh();
  MFEM_VERIFY(surface_marker.Size() == body_mesh_->bdr_attributes.Max(),
              "SeaLevelOperator: the surface marker must be sized to the "
              "body mesh's bdr_attributes.Max().");
  parallel_ = detail::IsParallel(body_fes);
  order_ = order < 0 ? body_fes.GetMaxElementOrder() : order;
  const int dim = body_mesh_->Dimension();
  const Array<int> attrs = MarkedAttributes(surface_marker);
  MFEM_VERIFY(attrs.Size() > 0, "SeaLevelOperator: empty surface marker.");

  // MFEM's CreateFromBoundary crashes when its parent is itself a
  // SubMesh (the internal node transfer maps through the wrong parent),
  // so when the body is a SubMesh the surface is cut from the body's own
  // parent instead: the surface boundary elements exist there with the
  // same inherited attributes (the meshes carry boundary elements on
  // every interface). Fields then hop body -> top -> surface.
  top_mesh_ = body_mesh_;
#ifdef MFEM_USE_MPI
  if (parallel_) {
    if (auto* psm = dynamic_cast<ParSubMesh*>(body_mesh_)) {
      top_mesh_ = const_cast<ParMesh*>(psm->GetParent());
    }
  } else
#endif
  {
    if (auto* sm = dynamic_cast<SubMesh*>(body_mesh_)) {
      top_mesh_ = const_cast<Mesh*>(sm->GetParent());
    }
  }
  for (int a : attrs) {
    MFEM_VERIFY(a <= top_mesh_->bdr_attributes.Max(),
                "SeaLevelOperator: surface attribute " << a << " does not "
                "exist on the top-level mesh (the surface must be a "
                "boundary-element interface there).");
  }

  fec_ = std::make_unique<H1_FECollection>(order_, dim - 1);
  body_fec_ = std::make_unique<H1_FECollection>(order_, dim);
#ifdef MFEM_USE_MPI
  if (parallel_) {
    auto* ptop = static_cast<ParMesh*>(top_mesh_);
    comm_ = ptop->GetComm();
    auto psurf = std::make_unique<ParSubMesh>(
        ParSubMesh::CreateFromBoundary(*ptop, attrs));
    sfes_ = std::make_unique<ParFiniteElementSpace>(psurf.get(), fec_.get());
    surf_mesh_ = std::move(psurf);
    top_sfes_ = std::make_unique<ParFiniteElementSpace>(ptop, body_fec_.get());
    if (top_mesh_ != body_mesh_) {
      body_sfes_ = std::make_unique<ParFiniteElementSpace>(
          static_cast<ParMesh*>(body_mesh_), body_fec_.get());
    }
  } else
#endif
  {
    surf_mesh_ = std::make_unique<SubMesh>(
        SubMesh::CreateFromBoundary(*top_mesh_, attrs));
    sfes_ = std::make_unique<FiniteElementSpace>(surf_mesh_.get(),
                                                 fec_.get());
    top_sfes_ = std::make_unique<FiniteElementSpace>(top_mesh_,
                                                     body_fec_.get());
    if (top_mesh_ != body_mesh_) {
      body_sfes_ = std::make_unique<FiniteElementSpace>(body_mesh_,
                                                        body_fec_.get());
    }
  }
  sl1_ = detail::MakeGridFunction(sfes_.get());
  *sl1_ = 0.0;
}

GridFunction& SeaLevelOperator::BodyScratch() const {
  if (!body_scratch_) {
    body_scratch_ = detail::MakeGridFunction(
        body_sfes_ ? body_sfes_.get() : top_sfes_.get());
  }
  return *body_scratch_;
}

GridFunction& SeaLevelOperator::TopScratch() const {
  if (!top_scratch_) {
    top_scratch_ = detail::MakeGridFunction(top_sfes_.get());
  }
  return *top_scratch_;
}

void SeaLevelOperator::SetInitialState(Coefficient& sea_level,
                                       Coefficient& ice_thickness) {
  if (!sl0_) {
    sl0_ = detail::MakeGridFunction(sfes_.get());
    ice_ = detail::MakeGridFunction(sfes_.get());
  }
  sl0_->ProjectCoefficient(sea_level);
  ice_->ProjectCoefficient(ice_thickness);
}

GridFunction& SeaLevelOperator::InitialSeaLevel() {
  MFEM_VERIFY(sl0_, "SeaLevelOperator: no initial state set.");
  return *sl0_;
}

GridFunction& SeaLevelOperator::IceThickness() {
  MFEM_VERIFY(ice_, "SeaLevelOperator: no initial state set.");
  return *ice_;
}

GridFunction& SeaLevelOperator::SeaLevelChangeField() { return *sl1_; }

void SeaLevelOperator::OceanFunction(GridFunction& c) const {
  MFEM_VERIFY(c.FESpace() == sfes_.get(),
              "SeaLevelOperator::OceanFunction: a surface field expected.");
  if (!sl0_) {
    c = 1.0;
    return;
  }
  for (int i = 0; i < c.Size(); i++) {
    c[i] = rho_water_ * (*sl0_)[i] - rho_ice_ * (*ice_)[i] > 0.0 ? 1.0 : 0.0;
  }
}

void SeaLevelOperator::TransferPair(const GridFunction& src,
                                    GridFunction& dst) const {
#ifdef MFEM_USE_MPI
  if (parallel_) {
    ParSubMesh::Transfer(static_cast<const ParGridFunction&>(src),
                         static_cast<ParGridFunction&>(dst));
    return;
  }
#endif
  SubMesh::Transfer(src, dst);
}

void SeaLevelOperator::Restrict(const GridFunction& scalar,
                                GridFunction& out) const {
  MFEM_VERIFY(out.FESpace() == sfes_.get(),
              "SeaLevelOperator::Restrict: the output must live on the "
              "surface space.");
  const Mesh* src_mesh = scalar.FESpace()->GetMesh();
  if (src_mesh == surf_mesh_.get()) {
    out = scalar;
    return;
  }
  if (src_mesh == top_mesh_) {
    TransferPair(scalar, out);
    return;
  }
  // A body field when the body is a SubMesh of the top mesh: hop up to
  // the top scratch (only the body's dofs are written, which include
  // every surface node), then down to the surface.
  MFEM_VERIFY(src_mesh == body_mesh_,
              "SeaLevelOperator::Restrict: the field must live on the "
              "surface, body or top-level mesh.");
  GridFunction& hop = TopScratch();
  hop = 0.0;
  TransferPair(scalar, hop);
  TransferPair(hop, out);
}

SeaLevelOperator::SeaLevelChangeInfo SeaLevelOperator::SeaLevelChange(
    const GridFunction& u, VectorCoefficient& grad_phi0,
    const GridFunction& phi, Coefficient* psi, real_t water_mass_change) {
  MFEM_VERIFY(u.FESpace()->GetMesh() == body_mesh_,
              "SeaLevelOperator::SeaLevelChange: u must live on the body "
              "mesh.");
  // u . grad(Phi0) interpolated nodally on the body, then restricted.
  auto ugf = detail::MakeGridFunction(sfes_.get());
  {
    VectorGridFunctionCoefficient uc(&u);
    InnerProductCoefficient ug(uc, grad_phi0);
    BodyScratch().ProjectCoefficient(ug);
    Restrict(BodyScratch(), *ugf);
  }
  // The potential trace.
  auto phis = detail::MakeGridFunction(sfes_.get());
  Restrict(phi, *phis);
  // g = |grad(Phi0)| nodally on the body, then restricted.
  auto g = detail::MakeGridFunction(sfes_.get());
  {
    InnerProductCoefficient gg(grad_phi0, grad_phi0);
    PowerCoefficient gmag(gg, 0.5);
    BodyScratch().ProjectCoefficient(gmag);
    Restrict(BodyScratch(), *g);
  }
  // The optional surface potential.
  std::unique_ptr<GridFunction> psis;
  if (psi) {
    psis = detail::MakeGridFunction(sfes_.get());
    psis->ProjectCoefficient(*psi);
  }

  for (int i = 0; i < sl1_->Size(); i++) {
    const real_t gi = (*g)[i];
    MFEM_VERIFY(gi > 0.0,
                "SeaLevelOperator::SeaLevelChange: |grad Phi0| vanishes on "
                "the surface.");
    (*sl1_)[i] =
        -((*ugf)[i] + (*phis)[i] + (psis ? (*psis)[i] : 0.0)) / gi;
  }

  // The uniform constant: rho_w int_O (SL1) dS = water_mass_change.
  SeaLevelChangeInfo info;
  info.ocean_area = OceanArea();
  MFEM_VERIFY(info.ocean_area > 0.0,
              "SeaLevelOperator::SeaLevelChange: the ocean has zero area.");
  GridFunctionCoefficient slc(sl1_.get());
  const real_t raw = OceanIntegral(slc);
  info.uniform = (water_mass_change / rho_water_ - raw) / info.ocean_area;
  *sl1_ += info.uniform;
  return info;
}

void SeaLevelOperator::WriteSurfaceField(const GridFunction& f,
                                         const std::string& path) const {
  MFEM_VERIFY(f.FESpace() == sfes_.get(),
              "SeaLevelOperator::WriteSurfaceField: a surface field "
              "expected.");
  const int sdim = surf_mesh_->SpaceDimension();
  // Nodal coordinates of the scalar space: the identity projected on a
  // matching vector space (byNODES: component c of node i at i + c nd).
  FiniteElementCollection& fec = *fec_;
  std::unique_ptr<mfem::FiniteElementSpace> vfes;
  std::unique_ptr<GridFunction> coords;
#ifdef MFEM_USE_MPI
  if (parallel_) {
    vfes = std::make_unique<ParFiniteElementSpace>(
        static_cast<ParMesh*>(surf_mesh_.get()), &fec, sdim,
        Ordering::byNODES);
  } else
#endif
  {
    vfes = std::make_unique<mfem::FiniteElementSpace>(
        surf_mesh_.get(), &fec, sdim, Ordering::byNODES);
  }
  coords = detail::MakeGridFunction(vfes.get());
  VectorFunctionCoefficient identity(
      sdim, [](const Vector& x, Vector& v) { v = x; });
  coords->ProjectCoefficient(identity);

  const int nd = sfes_->GetVSize();
  // Pack the rows this rank owns (true dofs, so shared nodes are
  // written once), then gather to the root.
  std::vector<real_t> rows;
  rows.reserve(static_cast<std::size_t>(nd) * (sdim + 1));
  for (int i = 0; i < nd; i++) {
#ifdef MFEM_USE_MPI
    if (parallel_ &&
        static_cast<ParFiniteElementSpace*>(sfes_.get())
                ->GetLocalTDofNumber(i) < 0) {
      continue;
    }
#endif
    for (int c = 0; c < sdim; c++) {
      rows.push_back((*coords)[i + c * nd]);
    }
    rows.push_back(f[i]);
  }

  std::vector<real_t> all = rows;
#ifdef MFEM_USE_MPI
  if (parallel_) {
    int rank = 0, size = 1;
    MPI_Comm_rank(comm_, &rank);
    MPI_Comm_size(comm_, &size);
    const int nloc = static_cast<int>(rows.size());
    std::vector<int> counts(size), displs(size);
    MPI_Gather(&nloc, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm_);
    int total = 0;
    if (rank == 0) {
      for (int r = 0; r < size; r++) {
        displs[r] = total;
        total += counts[r];
      }
      all.resize(total);
    }
    MPI_Gatherv(rows.data(), nloc, MPITypeMap<real_t>::mpi_type, all.data(),
                counts.data(), displs.data(), MPITypeMap<real_t>::mpi_type,
                0, comm_);
    if (rank != 0) {
      return;
    }
  }
#endif
  std::ofstream out(path);
  MFEM_VERIFY(out, "SeaLevelOperator::WriteSurfaceField: cannot open "
                       << path);
  out.precision(16);
  out << (sdim == 2 ? "x,y,value\n" : "x,y,z,value\n");
  const std::size_t stride = sdim + 1;
  for (std::size_t r = 0; r + stride <= all.size(); r += stride) {
    for (std::size_t c = 0; c < stride; c++) {
      out << all[r + c] << (c + 1 < stride ? ',' : '\n');
    }
  }
}

real_t SeaLevelOperator::SurfaceIntegral(Coefficient& f) const {
  // The dual of f against the H1 partition of unity: Sum() of the
  // element-assembled L-vector counts every element once, so the global
  // sum over ranks is the integral.
  auto lf = detail::MakeLinearForm(sfes_.get());
  lf->AddDomainIntegrator(new DomainLFIntegrator(f));
  lf->Assemble();
  return GlobalSum(lf->Sum());
}

real_t SeaLevelOperator::OceanIntegral(Coefficient& f) const {
  if (!sl0_) {
    return SurfaceIntegral(f);  // all-ocean configuration
  }
  // The pointwise ocean indicator from the surface fields (never a
  // projection), times f.
  OceanIndicatorCoefficient flooded(*sl0_, *ice_, rho_water_, rho_ice_);
  ProductCoefficient cf(flooded, f);
  return SurfaceIntegral(cf);
}

real_t SeaLevelOperator::OceanArea() const {
  ConstantCoefficient one(1.0);
  return OceanIntegral(one);
}

real_t SeaLevelOperator::GlobalSum(real_t v) const {
#ifdef MFEM_USE_MPI
  if (parallel_) {
    real_t g = 0.0;
    MPI_Allreduce(&v, &g, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM, comm_);
    return g;
  }
#endif
  return v;
}

}  // namespace AdGIA
