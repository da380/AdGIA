#include "AdGIA/profile_border.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

#include "AdGIA/detail/fem_factory.hpp"

namespace AdGIA {

using namespace mfem;

namespace {

// Local dofs to true dofs through the space's prolongation.
void ToTrueDofs(const FiniteElementSpace& fes, const Vector& L, Vector& T) {
  const Operator* P = fes.GetProlongationMatrix();
  if (P) {
    T.SetSize(P->Width());
    P->MultTranspose(L, T);
  } else {
    T = L;
  }
}

// The layer containing the level c (clamped to the stack's range).
int LocateLayer(const std::vector<ProfileBorder::Layer>& layers, real_t c) {
  const int n = static_cast<int>(layers.size());
  int j = 0;
  while (j < n - 1 && c > layers[j].c1) {
    j++;
  }
  return j;
}

// The unscaled basis of gamma and its level derivative at c: on a neutral
// layer the one end value; on a stratified layer the two linear end
// functions and the interior modes P_k - P_{k-2}, k = 2..degree, in the
// layer's coordinate xi in [-1, 1], whose derivatives are (2k-1) P_{k-1}.
void RawBasis(const std::vector<ProfileBorder::Layer>& layers, int size,
              real_t c, Vector& b, Vector& db) {
  b.SetSize(size);
  db.SetSize(size);
  b = 0.0;
  db = 0.0;
  const auto& L = layers[LocateLayer(layers, c)];
  if (L.neutral) {
    b[L.v0] = 1.0;
    return;
  }
  const real_t h = L.c1 - L.c0;
  const real_t xi =
      std::min(real_t{1}, std::max(real_t{-1}, 2.0 * (c - L.c0) / h - 1.0));
  const real_t dxi = 2.0 / h;
  b[L.v0] = 0.5 * (1.0 - xi);
  db[L.v0] = -0.5 * dxi;
  b[L.v1] = 0.5 * (1.0 + xi);
  db[L.v1] = 0.5 * dxi;
  real_t Pkm2 = 1.0, Pkm1 = xi;  // P_{k-2}, P_{k-1}
  for (int k = 2; k <= L.degree; k++) {
    const real_t Pk = ((2 * k - 1) * xi * Pkm1 - (k - 1) * Pkm2) / k;
    b[L.bubble + k - 2] = Pk - Pkm2;
    db[L.bubble + k - 2] = (2 * k - 1) * Pkm1 * dxi;
    Pkm2 = Pkm1;
    Pkm1 = Pk;
  }
}

// rho_F b_k(Phi0) on the solid's boundary elements: the wall column's
// coefficient for the basis function k.
class WallProfileCoefficient : public Coefficient {
 public:
  WallProfileCoefficient(const ProfileBorder& border, Coefficient& rho,
                         std::function<real_t(ElementTransformation&,
                                              const IntegrationPoint&)> level)
      : border_(&border), rho_(&rho), level_(std::move(level)) {}
  void SetIndex(int k) { k_ = k; }
  real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    T.SetIntPoint(&ip);
    border_->Basis(level_(T, ip), b_, db_);
    return rho_->Eval(T, ip) * b_[k_];
  }

 private:
  const ProfileBorder* border_;
  Coefficient* rho_;
  std::function<real_t(ElementTransformation&, const IntegrationPoint&)> level_;
  int k_ = 0;
  Vector b_, db_;
};

}  // namespace

void ProfileBorder::Basis(real_t c, Vector& b, Vector& db) const {
  RawBasis(layers, size, c, b, db);
  for (int k = 0; k < size; k++) {
    b[k] *= scale[k];
    db[k] *= scale[k];
  }
}

real_t ProfileBorder::Gamma(real_t c) const {
  MFEM_VERIFY(y.Size() == size, "ProfileBorder::Gamma: no solve yet.");
  Vector b, db;
  Basis(c, b, db);
  return b * y;
}

std::vector<ProfileBorder> BuildProfileBorders(
    const ProfileBorderContext& ctx, const std::vector<FluidRegion>& fluids,
    const std::list<mfem::Array<int>>& fluid_markers,
    const std::vector<mfem::Coefficient*>& rho_prime) {
  const ProfileBorderOptions& opt = ctx.options;
  MFEM_VERIFY(opt.degree >= 1, "SetProfileBorder: degree at least one.");
  MFEM_VERIFY(opt.bins >= 1, "SetProfileBorder: at least one level bin.");
  MFEM_VERIFY(opt.neutral_tol > 0.0,
              "SetProfileBorder: a positive neutral tolerance.");
  std::vector<real_t> breaks(opt.breaks);
  std::sort(breaks.begin(), breaks.end());
  MFEM_VERIFY(std::adjacent_find(breaks.begin(), breaks.end()) == breaks.end(),
              "SetProfileBorder: a break is given twice.");
  std::vector<ProfileBorder> borders;
  Mesh* ms = ctx.fes_u->GetMesh();    // the solid SubMesh
  Mesh* mp = ctx.fes_pot->GetMesh();  // the parent ball
  const int qorder = 2 * ctx.fes_pot->GetMaxElementOrder() + 2;
#ifdef MFEM_USE_MPI
  auto* pfes_pot = dynamic_cast<ParFiniteElementSpace*>(ctx.fes_pot);
  MPI_Comm comm = pfes_pot ? pfes_pot->GetComm() : MPI_COMM_NULL;
  auto allreduce = [&](real_t* a, int count, MPI_Op op) {
    if (pfes_pot) {
      MPI_Allreduce(MPI_IN_PLACE, a, count, MPITypeMap<real_t>::mpi_type, op,
                    comm);
    }
  };
#else
  auto allreduce = [](real_t*, int, int) {};
  constexpr int MPI_SUM = 0, MPI_MIN = 1, MPI_MAX = 2;
#endif
  bool root = true;
#ifdef MFEM_USE_MPI
  if (pfes_pot) {
    root = pfes_pot->GetMyRank() == 0;
  }
#endif
  auto pot_parent = [&](ElementTransformation& T,
                        const IntegrationPoint& ip) -> real_t {
    return ctx.pot0_coef ? ctx.pot0_coef->Eval(T, ip)
                         : ctx.pot0->GetValue(T, ip);
  };
  auto pot_shadow = [&](ElementTransformation& T,
                        const IntegrationPoint& ip) -> real_t {
    return ctx.pot0_coef ? ctx.pot0_coef->Eval(T, ip)
                         : ctx.pot0_shadow->GetValue(T, ip);
  };

  auto marker = fluid_markers.begin();
  for (size_t r = 0; r < fluids.size(); r++, ++marker) {
    const FluidRegion& f = fluids[r];
    Coefficient* rp_coef = rho_prime[r];
    Coefficient& rho_wall =
        f.interface_density ? *f.interface_density : *f.density;
    ProfileBorder B;

    auto for_each_fluid_point =
        [&](const std::function<void(ElementTransformation&,
                                     const IntegrationPoint&, real_t)>& fn) {
          for (int e = 0; e < mp->GetNE(); e++) {
            if (!(*marker)[mp->GetAttribute(e) - 1]) {
              continue;
            }
            ElementTransformation* T = mp->GetElementTransformation(e);
            const IntegrationRule& ir =
                IntRules.Get(T->GetGeometryType(), qorder);
            for (int q = 0; q < ir.GetNPoints(); q++) {
              const IntegrationPoint& ip = ir.IntPoint(q);
              T->SetIntPoint(&ip);
              fn(*T, ip, ip.weight * T->Weight());
            }
          }
        };

    // 1. The level range of the region: the fluid's quadrature points and
    // the points of its wall.
    real_t cmin = std::numeric_limits<real_t>::max(), cmax = -cmin;
    for_each_fluid_point(
        [&](ElementTransformation& T, const IntegrationPoint& ip, real_t) {
          const real_t c = pot_parent(T, ip);
          cmin = std::min(cmin, c);
          cmax = std::max(cmax, c);
        });
    const int nattr = f.interface_marker.Size();
    for (int be = 0; be < ms->GetNBE(); be++) {
      const int a = ms->GetBdrAttribute(be);
      if (a > nattr || !f.interface_marker[a - 1]) {
        continue;
      }
      ElementTransformation* T = ms->GetBdrElementTransformation(be);
      const IntegrationRule& ir = IntRules.Get(T->GetGeometryType(), qorder);
      for (int q = 0; q < ir.GetNPoints(); q++) {
        const IntegrationPoint& ip = ir.IntPoint(q);
        T->SetIntPoint(&ip);
        const real_t c = pot_shadow(*T, ip);
        cmin = std::min(cmin, c);
        cmax = std::max(cmax, c);
      }
    }
    allreduce(&cmin, 1, MPI_MIN);
    allreduce(&cmax, 1, MPI_MAX);
    MFEM_VERIFY(cmax > cmin,
                "SetProfileBorder: region " << r << " has no level range.");

    // 2. The survey: per level bin the integrals of beta and of
    // rho^2 / kappa, whose ratio is the excess of the Bullen parameter
    // over one. The excess is the region's own where it supplies one,
    // and otherwise follows from its density gradient and bulk modulus.
    const int nb = opt.bins;
    const real_t hb = (cmax - cmin) / nb;
    auto bin_of = [&](real_t c) {
      return std::min(nb - 1, std::max(0, static_cast<int>((c - cmin) / hb)));
    };
    Vector s_beta(nb), s_ref(nb);
    s_beta = 0.0;
    s_ref = 0.0;
    for_each_fluid_point(
        [&](ElementTransformation& T, const IntegrationPoint& ip, real_t w) {
          const real_t rho = f.density->Eval(T, ip);
          const real_t ref = rho * rho / f.bulk_modulus->Eval(T, ip);
          const int b = bin_of(pot_parent(T, ip));
          s_beta[b] += w * (f.stratification
                                ? f.stratification->Eval(T, ip) * ref
                                : -(rp_coef->Eval(T, ip) + ref));
          s_ref[b] += w * ref;
        });
    allreduce(s_beta.HostReadWrite(), nb, MPI_SUM);
    allreduce(s_ref.HostReadWrite(), nb, MPI_SUM);
    B.bin_level.SetSize(nb);
    B.bin_excess.SetSize(nb);
    // The kind of each bin: 0 neutral, 1 stratified; a bin without fluid
    // takes the kind of the nearest bin below that has some (above, for
    // the lowest bins).
    std::vector<int> kind(nb, -1);
    real_t worst = 0.0;
    int n_unstable = 0, n_filled = 0;
    for (int b = 0; b < nb; b++) {
      B.bin_level[b] = cmin + (b + 0.5) * hb;
      B.bin_excess[b] = 0.0;
      if (s_ref[b] <= 0.0) {
        continue;
      }
      n_filled++;
      const real_t e = s_beta[b] / s_ref[b];
      B.bin_excess[b] = e;
      if (e < -opt.neutral_tol) {
        n_unstable++;
        worst = std::min(worst, e);
        kind[b] = 0;
      } else {
        kind[b] = e > opt.neutral_tol ? 1 : 0;
      }
    }
    MFEM_VERIFY(n_filled > 0,
                "SetProfileBorder: region " << r << " holds no fluid.");
    MFEM_VERIFY(
        n_unstable == 0 || opt.unstable_as_neutral,
        "SetProfileBorder: fluid region "
            << r << " is unstably stratified in " << n_unstable << " of "
            << n_filled << " level bins (the excess of the Bullen parameter "
            << "over one reaches " << worst << "). The static closure has no "
            << "solution there that a relaxation would reach: make the model "
            << "stable, or set ProfileBorderOptions::unstable_as_neutral to "
            << "treat those levels as neutral.");
    for (int b = 1; b < nb; b++) {
      if (kind[b] < 0) {
        kind[b] = kind[b - 1];
      }
    }
    for (int b = nb - 2; b >= 0; b--) {
      if (kind[b] < 0) {
        kind[b] = kind[b + 1];
      }
    }
    // The excess as a function of the level: the bin means, joined
    // linearly between the centres of the bins that hold fluid. The
    // stratified layers read their stratification from this and not
    // point by point: a density gradient taken numerically is a ratio of
    // two vanishing quantities next to a centre, and wild at single
    // points there.
    std::vector<real_t> tab_c, tab_e;
    for (int b = 0; b < nb; b++) {
      if (s_ref[b] > 0.0) {
        tab_c.push_back(B.bin_level[b]);
        tab_e.push_back(B.bin_excess[b]);
      }
    }
    auto mean_excess = [&](real_t c) {
      const auto it = std::upper_bound(tab_c.begin(), tab_c.end(), c);
      if (it == tab_c.begin()) {
        return tab_e.front();
      }
      if (it == tab_c.end()) {
        return tab_e.back();
      }
      const size_t i = it - tab_c.begin();
      const real_t t = (c - tab_c[i - 1]) / (tab_c[i] - tab_c[i - 1]);
      return (1.0 - t) * tab_e[i - 1] + t * tab_e[i];
    };

    // 3. The layers, and the numbering of the unknowns: the end values
    // first (the two ends of a neutral layer are one unknown), then the
    // interior modes. With breaks inside the region's range the layers
    // are the intervals between them; otherwise neighbouring bins of one
    // kind are merged.
    std::vector<real_t> edges{cmin};
    {
      const real_t margin = 1e-10 * (cmax - cmin);
      for (const real_t c : breaks) {
        if (c > cmin + margin && c < cmax - margin) {
          edges.push_back(c);
        }
      }
      edges.push_back(cmax);
    }
    B.declared = edges.size() > 2;
    // The mean excess of the stratified fluid of each declared layer.
    std::vector<real_t> layer_excess;
    if (B.declared) {
      const int ni = static_cast<int>(edges.size()) - 1;
      auto interval_of = [&](real_t c) {
        return static_cast<int>(
            std::upper_bound(edges.begin() + 1, edges.end() - 1, c) -
            (edges.begin() + 1));
      };
      // An interval is stratified when most of its fluid is, by the
      // weight rho^2 / kappa, point by point, and not by its mean: where
      // a break is inside elements the levels of the discrete background
      // put a sliver of the fluid of one side on the other, and a mean
      // would take a sliver of stratified fluid for a stratified layer.
      Vector i_ref(ni), i_strat(ni), i_excess(ni);
      i_ref = 0.0;
      i_strat = 0.0;
      i_excess = 0.0;
      for_each_fluid_point(
          [&](ElementTransformation& T, const IntegrationPoint& ip, real_t w) {
            const real_t rho = f.density->Eval(T, ip);
            const real_t ref = rho * rho / f.bulk_modulus->Eval(T, ip);
            const real_t excess = f.stratification
                                      ? f.stratification->Eval(T, ip)
                                      : -(rp_coef->Eval(T, ip) + ref) / ref;
            const int j = interval_of(pot_parent(T, ip));
            i_ref[j] += w * ref;
            if (excess > opt.neutral_tol) {
              i_strat[j] += w * ref;
              i_excess[j] += w * ref * excess;
            }
          });
      allreduce(i_ref.HostReadWrite(), ni, MPI_SUM);
      allreduce(i_strat.HostReadWrite(), ni, MPI_SUM);
      allreduce(i_excess.HostReadWrite(), ni, MPI_SUM);
      // An interval without a quadrature point takes the kind of the one
      // below it (above, for the lowest).
      std::vector<int> ikind(ni, -1);
      for (int j = 0; j < ni; j++) {
        if (i_ref[j] > 0.0) {
          ikind[j] = i_strat[j] > 0.5 * i_ref[j] ? 1 : 0;
        }
      }
      for (int j = 1; j < ni; j++) {
        if (ikind[j] < 0) {
          ikind[j] = ikind[j - 1];
        }
      }
      for (int j = ni - 2; j >= 0; j--) {
        if (ikind[j] < 0) {
          ikind[j] = ikind[j + 1];
        }
      }
      for (int j = 0; j < ni; j++) {
        if (ikind[j] == 0 && !B.layers.empty() && B.layers.back().neutral) {
          B.layers.back().c1 = edges[j + 1];
          continue;
        }
        ProfileBorder::Layer L;
        L.c0 = edges[j];
        L.c1 = edges[j + 1];
        L.neutral = ikind[j] == 0;
        L.degree = L.neutral ? 0 : opt.degree;
        B.layers.push_back(L);
        layer_excess.push_back(
            i_strat[j] > 0.0 ? i_excess[j] / i_strat[j] : opt.neutral_tol);
      }
    } else {
      for (int b = 0; b < nb;) {
        int e = b, filled = 0;
        while (e < nb && kind[e] == kind[b]) {
          filled += s_ref[e] > 0.0 ? 1 : 0;
          e++;
        }
        ProfileBorder::Layer L;
        L.c0 = cmin + b * hb;
        L.c1 = e == nb ? cmax : cmin + e * hb;
        L.neutral = kind[b] == 0;
        L.degree = L.neutral ? 0 : std::max(1, std::min(opt.degree, filled));
        B.layers.push_back(L);
        b = e;
      }
    }
    int next = 0;
    for (size_t j = 0; j < B.layers.size(); j++) {
      auto& L = B.layers[j];
      L.v0 = j == 0 ? next++ : B.layers[j - 1].v1;
      L.v1 = L.neutral ? L.v0 : next++;
    }
    for (auto& L : B.layers) {
      L.bubble = next;
      next += L.neutral ? 0 : L.degree - 1;
    }
    B.size = next;
    const int m = B.size;

    // 4. The form a and the potential columns, on the same quadrature
    // points. With varpi_k = rho b_k,
    //   a_kl = int [ -rho' b_k b_l - rho (b_k' b_l + b_k b_l')
    //                + rho^2 b_k' b_l' / beta ] dV,
    // the last term on stratified layers only, and the column is
    //   -int (rho' b_k + rho b_k') chi dV.
    // In this form the constant gamma = 1 gives the conservation of the
    // region's mass as an identity of the discrete system.
    DenseMatrix A(m);
    A = 0.0;
    std::vector<Vector> colL(m, Vector(ctx.fes_pot->GetVSize()));
    for (auto& v : colL) {
      v = 0.0;
    }
    {
      Array<int> vdofs;
      Vector shape, b, db;
      for (int e = 0; e < mp->GetNE(); e++) {
        if (!(*marker)[mp->GetAttribute(e) - 1]) {
          continue;
        }
        const FiniteElement* fe = ctx.fes_pot->GetFE(e);
        ctx.fes_pot->GetElementVDofs(e, vdofs);
        shape.SetSize(fe->GetDof());
        ElementTransformation* T = mp->GetElementTransformation(e);
        const IntegrationRule& ir = IntRules.Get(T->GetGeometryType(), qorder);
        for (int q = 0; q < ir.GetNPoints(); q++) {
          const IntegrationPoint& ip = ir.IntPoint(q);
          T->SetIntPoint(&ip);
          const real_t w = ip.weight * T->Weight();
          const real_t c = pot_parent(*T, ip);
          const real_t rho = f.density->Eval(*T, ip);
          const real_t rp = rp_coef->Eval(*T, ip);
          const real_t ref = rho * rho / f.bulk_modulus->Eval(*T, ip);
          const int layer = LocateLayer(B.layers, c);
          const bool neutral = B.layers[layer].neutral;
          // Within a stratified layer the stratification is the
          // region's own where it supplies one and the level mean of the
          // survey otherwise, kept above the neutral tolerance. A point
          // of neutral fluid in a declared stratified layer is of the
          // sliver that the discrete levels put on the wrong side of a
          // break, and takes the layer's mean.
          real_t excess = f.stratification ? f.stratification->Eval(*T, ip)
                                           : mean_excess(c);
          if (B.declared && excess <= opt.neutral_tol) {
            excess = layer_excess[layer];
          }
          const real_t beta = std::max(excess, opt.neutral_tol) * ref;
          RawBasis(B.layers, m, c, b, db);
          fe->CalcShape(ip, shape);
          for (int k = 0; k < m; k++) {
            if (b[k] == 0.0 && db[k] == 0.0) {
              continue;
            }
            for (int l = 0; l < m; l++) {
              real_t v = -rp * b[k] * b[l] - rho * (db[k] * b[l] + b[k] * db[l]);
              if (!neutral) {
                v += rho * rho * db[k] * db[l] / beta;
              }
              A(k, l) += w * v;
            }
            const real_t s = -w * (rp * b[k] + rho * db[k]);
            for (int j = 0; j < vdofs.Size(); j++) {
              const int d = vdofs[j];
              const int dof = d >= 0 ? d : -1 - d;
              colL[k][dof] += (d >= 0 ? s : -s) * shape[j];
            }
          }
        }
      }
    }
    allreduce(A.Data(), m * m, MPI_SUM);

    // 5. The scaling to unit diagonal, the dense block and the columns.
    B.scale.SetSize(m);
    for (int k = 0; k < m; k++) {
      MFEM_VERIFY(A(k, k) > 0.0,
                  "SetProfileBorder: unknown " << k << " of region " << r
                  << " has no positive energy (" << A(k, k) << "); the "
                  << "stratification is not resolved by the quadrature.");
      B.scale[k] = 1.0 / std::sqrt(A(k, k));
    }
    B.Db.SetSize(m);
    for (int k = 0; k < m; k++) {
      for (int l = 0; l < m; l++) {
        B.Db(k, l) = -0.5 * (A(k, l) + A(l, k)) * B.scale[k] * B.scale[l];
      }
    }
    B.phi_cols.resize(m);
    for (int k = 0; k < m; k++) {
      colL[k] *= B.scale[k];
      ToTrueDofs(*ctx.fes_pot, colL[k], B.phi_cols[k]);
    }

    // 6. The wall columns, int rho_F b_k(Phi0) (m . v) dS over the
    // region's interfaces, m the outward normal of the solid.
    {
      WallProfileCoefficient wall(B, rho_wall, pot_shadow);
      B.u_cols.resize(m);
      for (int k = 0; k < m; k++) {
        wall.SetIndex(k);
        auto lf = detail::MakeLinearForm(ctx.fes_u);
        lf->AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(wall),
                                  const_cast<Array<int>&>(f.interface_marker));
        lf->Assemble();
        ToTrueDofs(*ctx.fes_u, *lf, B.u_cols[k]);
      }
    }

    if (root) {
      std::cout << "Profile border, region " << r << ": " << m
                << " unknowns on " << B.layers.size()
                << (B.declared ? " declared" : "") << " layer"
                << (B.layers.size() == 1 ? "" : "s") << " (";
      for (size_t j = 0; j < B.layers.size(); j++) {
        const auto& L = B.layers[j];
        std::cout << (j ? ", " : "")
                  << (L.neutral ? "neutral"
                                : "stratified, degree " +
                                      std::to_string(L.degree));
      }
      std::cout << ")";
      if (n_unstable > 0) {
        std::cout << "; " << n_unstable << " unstable level bins of "
                  << n_filled << " treated as neutral (excess to " << worst
                  << ")";
      }
      std::cout << ".\n";
    }
    borders.push_back(std::move(B));
  }
  return borders;
}

}  // namespace AdGIA
