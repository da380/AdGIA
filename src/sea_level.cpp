/**
 * @file sea_level.cpp
 * @brief Implementation of SeaLevelOperator (the WP1 surface layer).
 */

#include "AdGIA/sea_level.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
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

void SeaLevelOperator::SeaLevelChangeFrom(const GridFunction& u,
                                          VectorCoefficient& grad_phi0,
                                          const GridFunction& phi,
                                          real_t phi_g, Coefficient* psi) {
  MFEM_VERIFY(u.FESpace()->GetMesh() == body_mesh_,
              "SeaLevelOperator::SeaLevelChangeFrom: u must live on the "
              "body mesh.");
  auto ugf = detail::MakeGridFunction(sfes_.get());
  {
    VectorGridFunctionCoefficient uc(&u);
    InnerProductCoefficient ug(uc, grad_phi0);
    BodyScratch().ProjectCoefficient(ug);
    Restrict(BodyScratch(), *ugf);
  }
  auto phis = detail::MakeGridFunction(sfes_.get());
  Restrict(phi, *phis);
  auto g = detail::MakeGridFunction(sfes_.get());
  {
    InnerProductCoefficient gg(grad_phi0, grad_phi0);
    PowerCoefficient gmag(gg, 0.5);
    BodyScratch().ProjectCoefficient(gmag);
    Restrict(BodyScratch(), *g);
  }
  std::unique_ptr<GridFunction> psis;
  if (psi) {
    psis = detail::MakeGridFunction(sfes_.get());
    psis->ProjectCoefficient(*psi);
  }
  for (int i = 0; i < sl1_->Size(); i++) {
    const real_t gi = (*g)[i];
    MFEM_VERIFY(gi > 0.0,
                "SeaLevelOperator::SeaLevelChangeFrom: |grad Phi0| "
                "vanishes on the surface.");
    (*sl1_)[i] =
        (-((*ugf)[i] + (*phis)[i] + (psis ? (*psis)[i] : 0.0)) + phi_g) / gi;
  }
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

namespace {

// Nodal coordinates of a scalar surface space: the identity projected
// on a matching vector space (byNODES: component c of node i at
// i + c * nd). The CSV writer and reader must form them identically.
std::unique_ptr<GridFunction> NodalCoordinates(
    Mesh& mesh, FiniteElementCollection& fec, int sdim, bool parallel,
    std::unique_ptr<mfem::FiniteElementSpace>& vfes) {
#ifdef MFEM_USE_MPI
  if (parallel) {
    vfes = std::make_unique<ParFiniteElementSpace>(
        static_cast<ParMesh*>(&mesh), &fec, sdim, Ordering::byNODES);
  } else
#endif
  {
    vfes = std::make_unique<mfem::FiniteElementSpace>(&mesh, &fec, sdim,
                                                      Ordering::byNODES);
  }
  auto coords = detail::MakeGridFunction(vfes.get());
  VectorFunctionCoefficient identity(
      sdim, [](const Vector& x, Vector& v) { v = x; });
  coords->ProjectCoefficient(identity);
  return coords;
}

}  // namespace

void SeaLevelOperator::WriteSurfaceField(const GridFunction& f,
                                         const std::string& path) const {
  MFEM_VERIFY(f.FESpace() == sfes_.get(),
              "SeaLevelOperator::WriteSurfaceField: a surface field "
              "expected.");
  const int sdim = surf_mesh_->SpaceDimension();
  std::unique_ptr<mfem::FiniteElementSpace> vfes;
  auto coords = NodalCoordinates(*surf_mesh_, *fec_, sdim, parallel_, vfes);

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

std::vector<std::string> SeaLevelOperator::ReadSurfaceField(
    GridFunction& f, const std::string& path, int column) const {
  MFEM_VERIFY(f.FESpace() == sfes_.get(),
              "SeaLevelOperator::ReadSurfaceField: a surface field "
              "expected.");
  const int sdim = surf_mesh_->SpaceDimension();
  std::ifstream in(path);
  MFEM_VERIFY(in,
              "SeaLevelOperator::ReadSurfaceField: cannot open " << path);
  std::string line;
  MFEM_VERIFY(std::getline(in, line),
              "SeaLevelOperator::ReadSurfaceField: " << path << " is empty");
  std::vector<std::string> names;
  {
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      names.push_back(tok);
    }
  }
  MFEM_VERIFY(static_cast<int>(names.size()) > sdim + column,
              "SeaLevelOperator::ReadSurfaceField: " << path
                  << " has no value column " << column);
  names.erase(names.begin(), names.begin() + sdim);

  // Quantized-coordinate table of the file's rows: the quantum sits far
  // below any node separation and far above the 16-digit print
  // round-trip noise, and the lookup searches the neighbouring bins, so
  // a coordinate on a bin boundary cannot be missed. Matching by
  // coordinates makes the file independent of row order and of the rank
  // count that wrote it.
  const real_t q = 1e-9;
  using Key = std::array<long long, 3>;
  auto key_of = [&](const real_t* x) {
    Key k{0, 0, 0};
    for (int c = 0; c < sdim; c++) {
      k[c] = llround(x[c] / q);
    }
    return k;
  };
  struct KeyHash {
    std::size_t operator()(const Key& k) const {
      std::size_t h = 1469598103934665603ull;
      for (long long v : k) {
        h ^= static_cast<std::size_t>(v);
        h *= 1099511628211ull;
      }
      return h;
    }
  };
  std::unordered_map<Key, real_t, KeyHash> table;
  const std::size_t ncols = names.size() + sdim;
  std::vector<real_t> row(ncols);
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    std::stringstream ss(line);
    std::string tok;
    std::size_t c = 0;
    while (std::getline(ss, tok, ',') && c < ncols) {
      row[c++] = std::stod(tok);
    }
    MFEM_VERIFY(c == ncols,
                "SeaLevelOperator::ReadSurfaceField: short row in " << path);
    table[key_of(row.data())] = row[sdim + column];
  }

  std::unique_ptr<mfem::FiniteElementSpace> vfes;
  auto coords = NodalCoordinates(*surf_mesh_, *fec_, sdim, parallel_, vfes);
  const int nd = sfes_->GetVSize();
  for (int i = 0; i < nd; i++) {
    real_t x[3] = {0.0, 0.0, 0.0};
    for (int c = 0; c < sdim; c++) {
      x[c] = (*coords)[i + c * nd];
    }
    const Key k0 = key_of(x);
    bool found = false;
    real_t value = 0.0;
    const long long dlo = sdim == 3 ? -1 : 0, dhi = sdim == 3 ? 1 : 0;
    for (long long dx = -1; dx <= 1 && !found; dx++) {
      for (long long dy = -1; dy <= 1 && !found; dy++) {
        for (long long dz = dlo; dz <= dhi && !found; dz++) {
          const Key k{k0[0] + dx, k0[1] + dy, k0[2] + dz};
          auto it = table.find(k);
          if (it != table.end()) {
            value = it->second;
            found = true;
          }
        }
      }
    }
    MFEM_VERIFY(found, "SeaLevelOperator::ReadSurfaceField: no row of "
                           << path
                           << " matches a surface node (a different mesh?)");
    f[i] = value;
  }
  return names;
}

void SeaLevelOperator::Extend(const GridFunction& surface_field,
                              GridFunction& out) const {
  MFEM_VERIFY(surface_field.FESpace() == sfes_.get(),
              "SeaLevelOperator::Extend: a surface field expected.");
  Mesh* m = out.FESpace()->GetMesh();
  out = 0.0;
  if (m == top_mesh_) {
    TransferPair(surface_field, out);
    return;
  }
  MFEM_VERIFY(m == body_mesh_,
              "SeaLevelOperator::Extend: the target lives on neither the "
              "body nor the top mesh.");
  GridFunction& hop = TopScratch();
  hop = 0.0;
  TransferPair(surface_field, hop);
  TransferPair(hop, out);
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


// ---------------------------------------------------------------------------
// Shoreline migration

namespace detail {

/** The migrating ocean state: the fraction C of eq. (29) with
 * SL = SL0 + SL1, SL1 = (-tau - psi + Phi_g)/g read from stored copies
 * of the problem's solution fields (zero before the first solve, so the
 * initial fraction IS C0); and the two coefficients the water load
 * needs, w = rho_w C / g and the C-dependent data load. All evaluation
 * is on body (boundary) elements. */
class MigratingOceanState {
 public:
  MigratingOceanState(VectorCoefficient& grad_phi0, Coefficient& sl0,
                      Coefficient& ice0, Coefficient& dice, real_t rho_w,
                      real_t rho_i, real_t shore)
      : grad_phi0_(&grad_phi0),
        sl0_(&sl0),
        ice0_(&ice0),
        dice_(&dice),
        rho_w_(rho_w),
        rho_i_(rho_i),
        shore_(shore) {}

  /** Nodal SL1 on the body and on the parent (the water blocks
   * assemble on both meshes); evaluation dispatches on the
   * transformation's mesh. */
  void SetSeaLevelChange(std::unique_ptr<GridFunction> body,
                         std::unique_ptr<GridFunction> parent,
                         bool replace = false) {
    if (!replace) {
      sl1_prev_ = std::move(sl1_body_);
    }
    sl1_body_ = std::move(body);
    sl1_parent_ = std::move(parent);
  }

  const GridFunction* Current() const { return sl1_body_.get(); }
  const GridFunction* Previous() const { return sl1_prev_.get(); }

  real_t SeaLevelChange(ElementTransformation& T,
                        const IntegrationPoint& ip) const {
    if (!sl1_body_) {
      return 0.0;
    }
    if (T.mesh == sl1_body_->FESpace()->GetMesh()) {
      return sl1_body_->GetValue(T, ip);
    }
    MFEM_ASSERT(T.mesh == sl1_parent_->FESpace()->GetMesh(),
                "MigratingOceanState: unknown evaluation mesh");
    return sl1_parent_->GetValue(T, ip);
  }

  real_t Fraction(ElementTransformation& T, const IntegrationPoint& ip,
                  bool initial) const {
    // Before the first state update the current fraction IS the
    // initial one, C0 = C(SL0, I0): the frozen-shoreline pass (and the
    // off-switch) linearise about t0, as the load law does. Live, the
    // flotation criterion uses the current sea level AND ice.
    const bool as_initial = initial || !sl1_body_;
    const real_t sl =
        sl0_->Eval(T, ip) + (as_initial ? 0.0 : SeaLevelChange(T, ip));
    const real_t ice =
        ice0_->Eval(T, ip) + (as_initial ? 0.0 : dice_->Eval(T, ip));
    const real_t q = rho_w_ * sl - rho_i_ * ice;
    return 0.5 * (1.0 + std::tanh(q / shore_));
  }

  real_t Weight(ElementTransformation& T, const IntegrationPoint& ip) const {
    Vector g;
    grad_phi0_->Eval(g, T, ip);
    return rho_w_ * Fraction(T, ip, false) / g.Norml2();
  }

  real_t Data(ElementTransformation& T, const IntegrationPoint& ip) const {
    // sigma_d = rho_w (C - C0) SL0 + rho_i [(1-C)(I0+dI) - (1-C0) I0].
    const real_t c = Fraction(T, ip, false);
    const real_t c0 = Fraction(T, ip, true);
    const real_t sl0 = sl0_->Eval(T, ip);
    const real_t i0 = ice0_->Eval(T, ip);
    const real_t di = dice_->Eval(T, ip);
    return rho_w_ * (c - c0) * sl0 +
           rho_i_ * ((1.0 - c) * (i0 + di) - (1.0 - c0) * i0);
  }

 private:
  VectorCoefficient* grad_phi0_;
  Coefficient *sl0_, *ice0_, *dice_;
  real_t rho_w_, rho_i_, shore_;
  std::unique_ptr<GridFunction> sl1_body_, sl1_parent_, sl1_prev_;
};

namespace {

class StateCoefficient : public Coefficient {
 public:
  enum class Kind { Weight, Data, Fraction };
  StateCoefficient(MigratingOceanState& s, Kind kind)
      : state_(&s), kind_(kind) {}
  real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    switch (kind_) {
      case Kind::Weight:
        return state_->Weight(T, ip);
      case Kind::Data:
        return state_->Data(T, ip);
      default:
        return state_->Fraction(T, ip, false);
    }
  }

 private:
  MigratingOceanState* state_;
  Kind kind_;
};

}  // namespace
}  // namespace detail

ShorelineMigration::ShorelineMigration(
    LinearQuasiStaticMixedSelfGravitatingProblem& problem,
    VectorCoefficient& grad_phi0, Coefficient& initial_sea_level,
    Coefficient& initial_ice, Coefficient& ice_change, real_t rho_water,
    real_t rho_ice, const Array<int>& surface_marker, Options options)
    : problem_(problem),
      options_(options),
      marker_(surface_marker),
      grad_phi0_(&grad_phi0) {
  state_ = std::make_unique<detail::MigratingOceanState>(
      grad_phi0, initial_sea_level, initial_ice, ice_change, rho_water,
      rho_ice, options.shore);
  weight_ = std::make_unique<detail::StateCoefficient>(
      *state_, detail::StateCoefficient::Kind::Weight);
  data_ = std::make_unique<detail::StateCoefficient>(
      *state_, detail::StateCoefficient::Kind::Data);
  frac_coef_ = std::make_unique<detail::StateCoefficient>(
      *state_, detail::StateCoefficient::Kind::Fraction);
  problem_.SetWaterLoad(*weight_, *data_, surface_marker);

  FiniteElementSpace& fes = problem_.DisplacementSpace();
  const int dim = fes.GetMesh()->Dimension();
  sfec_ = std::make_unique<H1_FECollection>(fes.GetMaxElementOrder(), dim);
  sfes_ = detail::MakeFESpace(fes, sfec_.get());
  // A matching scalar space on the parent mesh (the potential's), for
  // the up-transferred state.
  pfes_ = detail::MakeFESpace(
      *const_cast<GridFunction&>(problem_.Potential()).FESpace(),
      sfec_.get());
  parallel_ = detail::IsParallel(fes);
#ifdef MFEM_USE_MPI
  if (parallel_) {
    comm_ = static_cast<ParFiniteElementSpace&>(fes).GetComm();
  }
#endif
}

ShorelineMigration::~ShorelineMigration() = default;

Coefficient& ShorelineMigration::OceanFraction() { return *frac_coef_; }

real_t ShorelineMigration::ShorelineChange() {
  // The relative surface-L2 increment of SL1 between consecutive
  // passes: smooth, so it resolves convergence even when the moving
  // shoreline strip is narrower than the mesh (a nodal or quadrature
  // snapshot of C itself cannot).
  const GridFunction* cur = state_->Current();
  if (!cur) {
    return 1.0;
  }
  GridFunction diff(*cur);
  if (state_->Previous()) {
    diff -= *state_->Previous();
  }
  auto norm = [&](const GridFunction& f) {
    GridFunctionCoefficient fc(const_cast<GridFunction*>(&f));
    ProductCoefficient f2(fc, fc);
    auto lf = detail::MakeLinearForm(sfes_.get());
    lf->AddBoundaryIntegrator(new BoundaryLFIntegrator(f2), marker_);
    lf->Assemble();
    real_t v = lf->Sum();
#ifdef MFEM_USE_MPI
    if (parallel_) {
      real_t g = 0.0;
      MPI_Allreduce(&v, &g, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                    comm_);
      v = g;
    }
#endif
    return std::sqrt(std::max(v, real_t{0}));
  };
  return norm(diff) / (norm(*cur) + 1e-300);
}

void ShorelineMigration::UpdateState(bool replace) {
  // Nodal SL1 = (-(u.grad Phi0 + phi + psi) + Phi_g)/g on the body
  // scalar space, then transferred up to the parent, so the state is
  // evaluable wherever the water blocks assemble.
  auto tau = detail::MakeGridFunction(sfes_.get());
  {
    VectorGridFunctionCoefficient uc(&problem_.Displacement());
    InnerProductCoefficient ug(uc, *grad_phi0_);
    tau->ProjectCoefficient(ug);
  }
  auto phi = detail::MakeGridFunction(sfes_.get());
  {
    GridFunctionCoefficient pc(
        const_cast<GridFunction*>(&problem_.PotentialOnBody()));
    phi->ProjectCoefficient(pc);
  }
  std::unique_ptr<GridFunction> psi;
  if (problem_.AngularVelocity().Size() > 0) {
    psi = detail::MakeGridFunction(sfes_.get());
    psi->ProjectCoefficient(problem_.SolutionCentrifugalPotential());
  }
  auto g = detail::MakeGridFunction(sfes_.get());
  {
    InnerProductCoefficient gg(*grad_phi0_, *grad_phi0_);
    PowerCoefficient gmag(gg, 0.5);
    g->ProjectCoefficient(gmag);
  }
  auto body = detail::MakeGridFunction(sfes_.get());
  const real_t phi_g = problem_.UniformPotentialTerm();
  for (int i = 0; i < body->Size(); i++) {
    (*body)[i] = (-((*tau)[i] + (*phi)[i] + (psi ? (*psi)[i] : 0.0)) +
                  phi_g) /
                 (*g)[i];
  }
  auto parent = detail::MakeGridFunction(pfes_.get());
  *parent = 0.0;
#ifdef MFEM_USE_MPI
  if (parallel_) {
    ParSubMesh::Transfer(static_cast<const ParGridFunction&>(*body),
                         static_cast<ParGridFunction&>(*parent));
  } else
#endif
  {
    SubMesh::Transfer(*body, *parent);
  }
  state_->SetSeaLevelChange(std::move(body), std::move(parent), replace);
}

bool ShorelineMigration::Solve(real_t t) {
  iterations_ = 0;
  last_change_ = 0.0;
  outer_its_total_ = 0;
  const real_t base = problem_.RelTol();
  const bool loose = options_.inexact > 0.0 && options_.max_iterations > 0;
  auto solve_at = [&](real_t tol) {
    problem_.SetRelTol(tol);
    problem_.AssembleForce(t);
    const bool ok = problem_.Solve();
    outer_its_total_ += problem_.LastOuterIterations();
    return ok;
  };
  auto finish = [&](bool ok) {
    problem_.SetRelTol(base);
    return ok;
  };
  auto root = [&]() {
#ifdef MFEM_USE_MPI
    if (parallel_) {
      int rank = 0;
      MPI_Comm_rank(comm_, &rank);
      return rank == 0;
    }
#endif
    return true;
  };
  auto trace = [&](const char* what, real_t value) {
    if (options_.verbose && root()) {
      mfem::out << "shoreline migration: " << what << " " << value
                << ", Phi_g " << problem_.UniformPotentialTerm() << "\n";
    }
  };

  // The frozen-C pass (the state is empty, so C = C0): the production
  // answer when migration is off, the Picard seed otherwise.
  if (!solve_at(loose ? options_.inexact_max : base)) {
    return finish(false);
  }
  trace("seed solved, tolerance",
        loose ? options_.inexact_max : base);
  real_t prev_change = std::numeric_limits<real_t>::infinity();
  bool inexact_abandoned = false;
  for (int it = 0; it < options_.max_iterations; it++) {
    UpdateState();
    last_change_ = ShorelineChange();
    // The contraction guard (the Options note): a loose pass whose SL1
    // increment fails to contract may have been poisoned by the
    // border-amplified solver error. Redo this iterate at full
    // tolerance, measured against the same baseline — and abandon
    // inexactness for the rest of the loop: with this operator and
    // solver the amplification is a property of the configuration, so
    // every further loose pass would poison the state again (observed:
    // a non-sticky guard leaves the loop bouncing between flooded and
    // dried shorelines). The tight loop is globally attracted — C is
    // bounded and the water feedback is a weak contraction — so it
    // recovers even from a poisoned seed.
    // The absolute floor: an honest loose pass can move SL1 by about
    // the loosest solver tolerance, so the benign rattle of the
    // increments near convergence sits at ~inexact_max and must not
    // fire the guard; a non-contracting move an order of magnitude
    // bigger than any honest loose move is poison.
    const real_t floor = 10.0 * options_.inexact_max;
    if (loose && !inexact_abandoned && options_.guard > 0.0 &&
        state_->Previous() && last_change_ > options_.guard * prev_change &&
        last_change_ > floor) {
      trace("guard fired at change", last_change_);
      inexact_abandoned = true;
      if (!solve_at(base)) {
        return finish(false);
      }
      UpdateState(/*replace=*/true);
      last_change_ = ShorelineChange();
    }
    trace("pass change", last_change_);
    if (last_change_ <= options_.tol) {
      break;
    }
    prev_change = last_change_;
    problem_.RefreshWaterLoad();
    const real_t tol =
        loose && !inexact_abandoned
            ? std::min(std::max(options_.inexact * last_change_, base),
                       options_.inexact_max)
            : base;
    if (!solve_at(tol)) {
      return finish(false);
    }
    iterations_++;
  }
  if (loose) {
    // The polish: one full-tolerance solve whose operator carries the
    // final ocean function (the loose passes solved with the previous
    // pass's shoreline; warm-started, so it is cheap).
    problem_.RefreshWaterLoad();
    if (!solve_at(base)) {
      return finish(false);
    }
  }
  return finish(true);
}

// --- IceHistory --------------------------------------------------------------

namespace detail {

/** The stack's storage: each time's field on the surface space and its
 * nodal extensions to the top and (when distinct) body meshes, so the
 * coefficients evaluate wherever the load machinery assembles. */
class IceStack {
 public:
  IceStack(SeaLevelOperator& sea, const std::string& path) {
    auto first = detail::MakeGridFunction(&sea.SurfaceSpace());
    const auto names = sea.ReadSurfaceField(*first, path, 0);
    for (const auto& name : names) {
      std::size_t pos = 0;
      real_t t = 0.0;
      bool ok = true;
      try {
        t = std::stod(name, &pos);
      } catch (...) {
        ok = false;
      }
      MFEM_VERIFY(ok && pos == name.size(),
                  "IceHistory: the column name '"
                      << name << "' of " << path << " is not a time.");
      MFEM_VERIFY(times_.empty() || t > times_.back(),
                  "IceHistory: the times of " << path << " must ascend.");
      times_.push_back(t);
    }
    const int n = static_cast<int>(times_.size());
    surface_.resize(n);
    top_.resize(n);
    body_.resize(n);
    mfem::FiniteElementSpace* bfes = sea.BodyScalarSpace();
    for (int k = 0; k < n; k++) {
      surface_[k] = detail::MakeGridFunction(&sea.SurfaceSpace());
      if (k == 0) {
        *surface_[k] = *first;
      } else {
        sea.ReadSurfaceField(*surface_[k], path, k);
      }
      top_[k] = detail::MakeGridFunction(&sea.TopScalarSpace());
      sea.Extend(*surface_[k], *top_[k]);
      if (bfes) {
        body_[k] = detail::MakeGridFunction(bfes);
        sea.Extend(*surface_[k], *body_[k]);
      }
    }
  }

  int NumTimes() const { return static_cast<int>(times_.size()); }
  real_t Time(int k) const { return times_[k]; }
  const GridFunction& Surface(int k) const { return *surface_[k]; }

  /** Bracketing pair and weight of time @p t, clamped to the stack. */
  void Bracket(real_t t, int& k0, int& k1, real_t& a) const {
    const int n = NumTimes();
    if (t <= times_.front() || n == 1) {
      k0 = k1 = 0;
      a = 0.0;
      return;
    }
    if (t >= times_.back()) {
      k0 = k1 = n - 1;
      a = 0.0;
      return;
    }
    int k = 1;
    while (times_[k] < t) {
      k++;
    }
    k0 = k - 1;
    k1 = k;
    a = (t - times_[k0]) / (times_[k1] - times_[k0]);
  }

  real_t Value(ElementTransformation& T, const IntegrationPoint& ip,
               int k) const {
    const GridFunction* g = nullptr;
    if (T.mesh == surface_[k]->FESpace()->GetMesh()) {
      g = surface_[k].get();
    } else if (T.mesh == top_[k]->FESpace()->GetMesh()) {
      g = top_[k].get();
    } else if (body_[k] && T.mesh == body_[k]->FESpace()->GetMesh()) {
      g = body_[k].get();
    }
    MFEM_ASSERT(g, "IceStack: unknown evaluation mesh");
    return g->GetValue(T, ip);
  }

 private:
  std::vector<real_t> times_;
  std::vector<std::unique_ptr<GridFunction>> surface_, top_, body_;
};

namespace {

class IceStackCoefficient : public Coefficient {
 public:
  /** @p fixed >= 0 evaluates that field regardless of the time. */
  IceStackCoefficient(IceStack& stack, bool change, int fixed = -1)
      : stack_(stack), change_(change), fixed_(fixed) {}

  real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    if (fixed_ >= 0) {
      return stack_.Value(T, ip, fixed_);
    }
    int k0 = 0, k1 = 0;
    real_t a = 0.0;
    stack_.Bracket(GetTime(), k0, k1, a);
    real_t v = (1.0 - a) * stack_.Value(T, ip, k0);
    if (a != 0.0) {
      v += a * stack_.Value(T, ip, k1);
    }
    if (change_) {
      v -= stack_.Value(T, ip, 0);
    }
    return v;
  }

 private:
  IceStack& stack_;
  bool change_;
  int fixed_;
};

}  // namespace

}  // namespace detail

IceHistory::IceHistory(SeaLevelOperator& sea, const std::string& path)
    : stack_(std::make_unique<detail::IceStack>(sea, path)),
      interp_(std::make_unique<detail::IceStackCoefficient>(*stack_, false)),
      change_(std::make_unique<detail::IceStackCoefficient>(*stack_, true)) {}

IceHistory::~IceHistory() = default;

int IceHistory::NumTimes() const { return stack_->NumTimes(); }

real_t IceHistory::Time(int k) const { return stack_->Time(k); }

const GridFunction& IceHistory::Field(int k) const {
  return stack_->Surface(k);
}

Coefficient& IceHistory::Interpolant() { return *interp_; }

Coefficient& IceHistory::Change() { return *change_; }

Coefficient& IceHistory::FieldCoefficient(int k) {
  MFEM_VERIFY(k >= 0 && k < NumTimes(),
              "IceHistory::FieldCoefficient: no field " << k);
  if (fixed_.size() < static_cast<std::size_t>(NumTimes())) {
    fixed_.resize(NumTimes());
  }
  if (!fixed_[k]) {
    fixed_[k] =
        std::make_unique<detail::IceStackCoefficient>(*stack_, false, k);
  }
  return *fixed_[k];
}

}  // namespace AdGIA
