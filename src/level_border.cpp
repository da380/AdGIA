#include "AdGIA/level_border.hpp"

#include <algorithm>
#include <functional>
#include <limits>

#include "AdGIA/coefficient.hpp"
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

real_t ClampLevel(const Vector& c, real_t v) {
  return std::min(std::max(v, c[0]), c[c.Size() - 1]);
}

// The element of the grid containing v (clamped), and its hat values.
void LocateLevel(const Vector& c, real_t v, int& e, real_t& N0, real_t& N1) {
  const int n = c.Size() - 1;
  v = ClampLevel(c, v);
  e = 0;
  while (e < n - 1 && v > c[e + 1]) {
    e++;
  }
  const real_t h = c[e + 1] - c[e];
  N1 = (v - c[e]) / h;
  N0 = 1.0 - N1;
}

// -(m . grad Phi0) / |grad Phi0| on the solid's boundary elements: the
// wall column's coefficient, minus the sign that tells a fluid below the
// solid (outer wall of the region, m . grad Phi0 < 0) from one above.
class MinusWallSignCoefficient : public Coefficient {
 public:
  explicit MinusWallSignCoefficient(VectorCoefficient& g) : g_(&g) {}
  real_t Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
    T.SetIntPoint(&ip);
    n_.SetSize(T.GetSpaceDim());
    CalcOrtho(T.Jacobian(), n_);
    g_->Eval(v_, T, ip);
    const real_t nn = n_.Norml2(), vv = v_.Norml2();
    if (nn <= 0.0 || vv <= 0.0) {
      return 0.0;
    }
    return -(v_ * n_) / (nn * vv);
  }

 private:
  VectorCoefficient* g_;
  Vector v_, n_;
};

}  // namespace

void LevelBorder::Solution(Vector& c_out, Vector& U, Vector& s, Vector& nu,
                           Vector& lambda) const {
  MFEM_VERIFY(y.Size() == size, "LevelBorder::Solution: no solve yet.");
  c_out = c;
  U.SetSize(n + 1);
  s.SetSize(n + 1);
  nu.SetSize(n + 1);
  for (int k = 0; k <= n; k++) {
    U[k] = (centre && k == 0) ? 0.0 : y[oU + UIndex(k)];
    s[k] = y[os + k];
    nu[k] = y[onu + k];
  }
  lambda.SetSize(static_cast<int>(wall_u.size()));
  for (int w = 0; w < lambda.Size(); w++) {
    lambda[w] = y[ol + w];
  }
}

std::vector<LevelBorder> BuildLevelBorders(
    const LevelBorderContext& ctx, const std::vector<FluidRegion>& fluids,
    const std::list<mfem::Array<int>>& fluid_markers,
    const std::vector<mfem::Coefficient*>& rho_prime,
    std::vector<std::unique_ptr<mfem::Coefficient>>& coefs) {
  std::vector<LevelBorder> level;
  Mesh* ms = ctx.fes_u->GetMesh();   // the solid SubMesh
  Mesh* mp = ctx.fes_pot->GetMesh();  // the parent ball
  const int n = ctx.n;
  const int dim_ = ms->Dimension();
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
  BoundaryNormalDotCoefficient m_dot_g(*ctx.grad_pot0_shadow);
  auto marker = fluid_markers.begin();
  for (size_t r = 0; r < fluids.size(); r++, ++marker) {
    const FluidRegion& f = fluids[r];
    LevelBorder L;
    L.n = n;

    // 1. The walls: per interface attribute of the region, the area, the
    // area integrals of Phi0 and of m . grad Phi0 on the solid's boundary
    // elements. The sign of the latter classifies the wall.
    const int nattr = f.interface_marker.Size();
    std::vector<real_t> w_area(nattr, 0.0), w_phi(nattr, 0.0),
        w_mg(nattr, 0.0);
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
        const real_t w = ip.weight * T->Weight();
        w_area[a - 1] += w;
        w_phi[a - 1] += w * ctx.pot0_shadow->GetValue(*T, ip);
        w_mg[a - 1] += w * m_dot_g.Eval(*T, ip);
      }
    }
    allreduce(w_area.data(), nattr, MPI_SUM);
    allreduce(w_phi.data(), nattr, MPI_SUM);
    allreduce(w_mg.data(), nattr, MPI_SUM);
    Array<int> outer_marker(nattr), inner_marker(nattr);
    outer_marker = 0;
    inner_marker = 0;
    real_t top_area = 0.0, top_phi = 0.0, bot_area = 0.0, bot_phi = 0.0;
    for (int a = 0; a < nattr; a++) {
      if (!f.interface_marker[a] || w_area[a] <= 0.0) {
        continue;
      }
      if (w_mg[a] < 0.0) {
        outer_marker[a] = 1;
        top_area += w_area[a];
        top_phi += w_phi[a];
      } else {
        inner_marker[a] = 1;
        bot_area += w_area[a];
        bot_phi += w_phi[a];
      }
    }
    MFEM_VERIFY(top_area > 0.0,
                "SetLevelBorder: region " << r << " has no outer wall.");
    const real_t c_top = top_phi / top_area;
    L.centre = bot_area <= 0.0;

    // 2. The level range and the foliation assertion on the parent's fluid
    // elements: the extremes of Phi0 and the largest |grad Phi0|, then the
    // smallest |grad Phi0| away from a centre.
    real_t cmin = std::numeric_limits<real_t>::max(), cmax = -cmin;
    real_t gmax = 0.0;
    Vector gv(dim_);
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
    for_each_fluid_point([&](ElementTransformation& T,
                             const IntegrationPoint& ip, real_t) {
      const real_t c = ctx.pot0->GetValue(T, ip);
      cmin = std::min(cmin, c);
      cmax = std::max(cmax, c);
      ctx.grad_pot0->Eval(gv, T, ip);
      gmax = std::max(gmax, gv.Norml2());
    });
    allreduce(&cmin, 1, MPI_MIN);
    allreduce(&cmax, 1, MPI_MAX);
    allreduce(&gmax, 1, MPI_MAX);
    const real_t c_bot = L.centre ? cmin : bot_phi / bot_area;
    MFEM_VERIFY(c_top > c_bot,
                "SetLevelBorder: region " << r << " has no level range.");
    real_t gmin_away = gmax;
    const real_t range = c_top - c_bot;
    for_each_fluid_point([&](ElementTransformation& T,
                             const IntegrationPoint& ip, real_t) {
      const real_t c = ctx.pot0->GetValue(T, ip);
      if (L.centre && c - c_bot < 0.05 * range) {
        return;  // near the centre g -> 0 legitimately
      }
      ctx.grad_pot0->Eval(gv, T, ip);
      gmin_away = std::min(gmin_away, gv.Norml2());
    });
    allreduce(&gmin_away, 1, MPI_MIN);
    MFEM_VERIFY(gmin_away > 0.05 * gmax,
                "SetLevelBorder: |grad Phi0| vanishes inside fluid region "
                    << r << " away from a centre (" << gmin_away << " against "
                    << gmax << "); the level foliation is not regular.");

    // 3. The grid, uniform in the level.
    L.c.SetSize(n + 1);
    for (int k = 0; k <= n; k++) {
      L.c[k] = c_bot + (c_top - c_bot) * k / n;
    }

    // 4. The volume integrals: per-element volume, mass and kappa-volume
    // (the level-mean coefficients), and the hat-function blocks
    // K_ss = -int rho_Phi N_i N_j dV and K_snu = int N_i N_j dV.
    L.vol.SetSize(n);
    L.mass.SetSize(n);
    L.kap.SetSize(n);
    L.vol = 0.0;
    L.mass = 0.0;
    L.kap = 0.0;
    DenseMatrix Kss(n + 1), Ksnu(n + 1);
    Kss = 0.0;
    Ksnu = 0.0;
    Coefficient* rp_coef = rho_prime[r];
    for_each_fluid_point([&](ElementTransformation& T,
                             const IntegrationPoint& ip, real_t w) {
      int e;
      real_t N0, N1;
      LocateLevel(L.c, ctx.pot0->GetValue(T, ip), e, N0, N1);
      const real_t rho = f.density->Eval(T, ip);
      const real_t rp = rp_coef->Eval(T, ip);
      const real_t kap = f.bulk_modulus->Eval(T, ip);
      L.vol[e] += w;
      L.mass[e] += rho * w;
      L.kap[e] += kap * w;
      const real_t N[2] = {N0, N1};
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
          Kss(e + i, e + j) -= rp * N[i] * N[j] * w;
          Ksnu(e + i, e + j) += N[i] * N[j] * w;
        }
      }
    });
    allreduce(L.vol.GetData(), n, MPI_SUM);
    allreduce(L.mass.GetData(), n, MPI_SUM);
    allreduce(L.kap.GetData(), n, MPI_SUM);
    allreduce(Kss.Data(), (n + 1) * (n + 1), MPI_SUM);
    allreduce(Ksnu.Data(), (n + 1) * (n + 1), MPI_SUM);
    for (int e = 0; e < n; e++) {
      MFEM_VERIFY(L.vol[e] > 0.0,
                  "SetLevelBorder: level element " << e << " of region " << r
                  << " holds no fluid quadrature point; use fewer levels.");
    }

    // 5. The nu columns, int N_k(Phi0) chi dV on the potential's true
    // dofs, assembled on the SAME quadrature points as the 1-D block so
    // that sum_j (b_k)_j = int N_k dV holds exactly and the discrete mass
    // balance of the border is an identity.
    {
      std::vector<Vector> colL(n + 1, Vector(ctx.fes_pot->GetVSize()));
      for (auto& v : colL) {
        v = 0.0;
      }
      Array<int> vdofs;
      Vector shape;
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
          fe->CalcShape(ip, shape);
          int e1;
          real_t N0, N1;
          LocateLevel(L.c, ctx.pot0->GetValue(*T, ip), e1, N0, N1);
          for (int j = 0; j < vdofs.Size(); j++) {
            const int d = vdofs[j];
            const int dof = d >= 0 ? d : -1 - d;
            const real_t sgn = d >= 0 ? 1.0 : -1.0;
            colL[e1][dof] += sgn * w * N0 * shape[j];
            colL[e1 + 1][dof] += sgn * w * N1 * shape[j];
          }
        }
      }
      L.nu_phi.resize(n + 1);
      for (int k = 0; k <= n; k++) {
        ToTrueDofs(*ctx.fes_pot, colL[k], L.nu_phi[k]);
      }
    }

    // 6. The wall columns, -int (m . grad Phi0 / |grad Phi0|) (m . v) dS per
    // wall, outer first; the constrained node is n for the outer wall and 0
    // for the inner one.
    auto sign = std::make_unique<MinusWallSignCoefficient>(*ctx.grad_pot0_shadow);
    auto add_wall = [&](const Array<int>& wm, int node) {
      auto lf = detail::MakeLinearForm(ctx.fes_u);
      lf->AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(*sign),
                                const_cast<Array<int>&>(wm));
      lf->Assemble();
      Vector col;
      ToTrueDofs(*ctx.fes_u, *lf, col);
      L.wall_u.push_back(col);
      L.wall_node.push_back(node);
    };
    add_wall(outer_marker, n);
    if (!L.centre) {
      add_wall(inner_marker, 0);
    }
    coefs.push_back(std::move(sign));

    // 7. The 1-D block in the local ordering (U | s | nu | lambda).
    L.nU = L.centre ? n : n + 1;
    L.oU = 0;
    L.os = L.nU;
    L.onu = L.os + n + 1;
    L.ol = L.onu + n + 1;
    L.size = L.ol + static_cast<int>(L.wall_u.size());
    L.Db.SetSize(L.size);
    L.Db = 0.0;
    auto U = [&](int node) { return L.oU + L.UIndex(node); };
    auto S = [&](int node) { return L.os + node; };
    auto NU = [&](int node) { return L.onu + node; };
    for (int e = 0; e < n; e++) {
      const real_t h = L.c[e + 1] - L.c[e];
      const real_t rho_e = L.mass[e] / L.vol[e];
      // int kappa N_i' N_j' / A dc over the element: with the level-mean
      // kappa_e = kap_e / Vol_e and <A> = Vol_e / h, the integral is
      // kappa_e h / <A> = kappa_e h^2 / Vol_e, and the derivatives
      // contribute 1/h^2: kappa_e / Vol_e = kap_e / Vol_e^2. At a centre
      // A ~ sqrt(c - c_min) in three dimensions and the element mean of
      // 1/A is 4/3 of 1/<A>.
      real_t kuu = L.kap[e] / (L.vol[e] * L.vol[e]);
      if (L.centre && e == 0 && dim_ == 3) {
        kuu *= 4.0 / 3.0;
      }
      for (int i = 0; i < 2; i++) {
        const int ni = e + i;
        const real_t di = (i == 0) ? -1.0 : 1.0;  // sign of N_ni' on e
        for (int j = 0; j < 2; j++) {
          const int nj = e + j;
          const real_t dj = (j == 0) ? -1.0 : 1.0;
          // K_U nu: the 1-D mass matrix.
          const real_t m = h / 6.0 * ((i == j) ? 2.0 : 1.0);
          if (!(L.centre && ni == 0)) {
            L.Db(U(ni), NU(nj)) += m;
            L.Db(NU(nj), U(ni)) += m;
            // K_Us = int rho N_i' N_j dc = rho_e d_i / 2.
            L.Db(U(ni), S(nj)) += rho_e * di / 2.0;
            L.Db(S(nj), U(ni)) += rho_e * di / 2.0;
            if (!(L.centre && nj == 0)) {
              L.Db(U(ni), U(nj)) += kuu * di * dj;
            }
          }
        }
      }
    }
    for (int i = 0; i <= n; i++) {
      for (int j = 0; j <= n; j++) {
        L.Db(S(i), S(j)) += Kss(i, j);
        L.Db(S(i), NU(j)) += Ksnu(i, j);
        L.Db(NU(i), S(j)) += Ksnu(i, j);
      }
    }
    for (size_t w = 0; w < L.wall_u.size(); w++) {
      const int lam = L.ol + static_cast<int>(w);
      L.Db(U(L.wall_node[w]), lam) = 1.0;
      L.Db(lam, U(L.wall_node[w])) = 1.0;
    }
    level.push_back(std::move(L));
  }
  return level;
}

}  // namespace AdGIA
