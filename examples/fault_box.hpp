// ============================================================================
// fault_box.hpp
//
// The pieces coseismic_deformation.cpp and postseismic_deformation.cpp
// share:
//
//   ParametricFactor  the parametric spatial variation the examples give
//                     their material fields: depth dependence, a lateral
//                     gradient and a Gaussian anomaly, multiplied.
//   FaultBox          the Cartesian box with its free surface at z = 0,
//                     optional surface topography (a Gaussian, or an
//                     asymmetric hill) mapped directly onto
//                     the mesh nodes (no remeshing), the elastic-moduli
//                     factor, and an optional elastic lid assigned by
//                     element rows, so the material discontinuity falls
//                     on element faces.
//   Fault             a planar fault laid out as moment-tensor point
//                     sources (MatrixDeltaCoefficient, to be paired with
//                     DomainLFDeformationGradientIntegrator), each patch's
//                     moment from the local shear modulus.
//   SurfaceProfile    the displacement sampled along the free surface
//                     (the x axis; y = 0 in 3-D): FindPoints once, then
//                     any number of fields. Collective; each point is
//                     found on exactly one rank (ParMesh::FindPoints) and
//                     a zero-elsewhere sum reduces the samples everywhere.
//   SurfacePeaks      the extreme values of a scalar field over the WHOLE
//                     free surface, from its top-boundary dofs (a profile
//                     alone misses a strike-slip quadrant pattern, whose
//                     u_z vanishes on y = 0 by antisymmetry).
//
// The flags the AddOptions methods register, their meanings and their
// defaults are listed in the headers of the two examples.
// ============================================================================

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <numbers>
#include <vector>

#include "AdGIA.hpp"

namespace examples {

// (1 + beta |z| / D)(1 + alpha x / W)(1 - gamma exp(-r^2 / 2 radius^2))
// about the centre (cx, 0, -cz); checked positive on the box by Verify().
struct ParametricFactor {
  mfem::real_t beta = 0.0, alpha = 0.0;
  mfem::real_t gamma = 0.0, cx = 0.0, cz = 0.0, radius = 1.0;

  void Verify(mfem::real_t D, mfem::real_t h0, const char* name) const {
    MFEM_VERIFY(gamma < 1.0 && std::abs(alpha) < 2.0 && 1.0 + beta > 0.0 &&
                    1.0 - beta * std::max(h0, mfem::real_t{0}) / D > 0.0,
                "The " << name
                       << " factor must stay positive over the box, from "
                          "the bottom to the highest topography.");
  }

  std::function<mfem::real_t(const mfem::Vector&)> Make(int dim,
                                                        mfem::real_t W,
                                                        mfem::real_t D) const {
    const auto p = *this;
    return [p, dim, W, D](const mfem::Vector& x) {
      mfem::real_t f =
          (1.0 - p.beta * x(dim - 1) / D) * (1.0 + p.alpha * x(0) / W);
      if (p.gamma != 0.0) {
        const mfem::real_t dx = x(0) - p.cx;
        const mfem::real_t dz = x(dim - 1) + p.cz;
        mfem::real_t r2 = dx * dx + dz * dz;
        if (dim == 3) {
          r2 += x(1) * x(1);
        }
        f *= 1.0 - p.gamma * std::exp(-r2 / (2.0 * p.radius * p.radius));
      }
      return f;
    };
  }
};

// The box x in [-W/2, W/2] (3-D: also y), z in [-D, 0], its topography
// and the factor scaling both elastic moduli (fixed Poisson ratio).
// Boundary attributes of the Cartesian factories: 2-D bottom/right/top/
// left = 1/2/3/4; 3-D bottom 1, sides 2-5, top 6.
struct FaultBox {
  int nx = -1, nz = -1;  // -1: 64 x 32 in 2-D, 24 x 12 in 3-D
  mfem::real_t W = 8.0, D = 4.0;
  ParametricFactor moduli;                        // -beta, -alpha, -gamma...
  mfem::real_t h0 = 0.0, tx = 0.0, tr = 1.5;      // topography
  mfem::real_t hill = -1.0;                       // < 0: symmetric Gaussian

  void AddOptions(mfem::OptionsParser& args) {
    args.AddOption(&nx, "-nx", "--num-elements-x",
                   "Elements across the width (3-D: each horizontal "
                   "direction); -1 for a default per dimension.");
    args.AddOption(&nz, "-nz", "--num-elements-z",
                   "Elements through the depth D; -1 for a default per "
                   "dimension.");
    args.AddOption(&W, "-W", "--width", "Width of the box.");
    args.AddOption(&D, "-D", "--depth", "Depth of the box.");
    args.AddOption(&moduli.beta, "-beta", "--depth-stiffening",
                   "Moduli grow by 1 + beta |z| / D (0: homogeneous).");
    args.AddOption(&moduli.alpha, "-alpha", "--lateral-gradient",
                   "Moduli grow by 1 + alpha x / W (|alpha| < 2).");
    args.AddOption(&moduli.gamma, "-gamma", "--basin-amplitude",
                   "Relative moduli reduction at the basin centre (< 1).");
    args.AddOption(&moduli.cx, "-bx", "--basin-x", "x of the basin centre.");
    args.AddOption(&moduli.cz, "-bz", "--basin-depth",
                   "Depth of the basin centre below z = 0.");
    args.AddOption(&moduli.radius, "-br", "--basin-radius",
                   "Gaussian radius of the basin.");
    args.AddOption(&h0, "-topo", "--topography-height",
                   "Height of the Gaussian surface topography (0: flat).");
    args.AddOption(&tx, "-tx", "--topography-x",
                   "x of the topography centre.");
    args.AddOption(&tr, "-tr", "--topography-radius",
                   "Gaussian radius of the topography.");
    args.AddOption(&hill, "-hill", "--hill-far-fraction",
                   "Asymmetric hill: right of the crest the Gaussian flank "
                   "becomes a smooth, roughly linear ramp that keeps this "
                   "fraction of the height at the right edge of the box "
                   "(negative: symmetric Gaussian).");
  }

  // Per-dimension mesh defaults and the range checks.
  void Finalize(int dim) {
    if (nx < 0) {
      nx = dim == 2 ? 64 : 24;
    }
    if (nz < 0) {
      nz = dim == 2 ? 32 : 12;
    }
    moduli.Verify(D, h0, "modulus");
    MFEM_VERIFY(hill < 0.0 || (hill <= 1.0 && tx < 0.5 * W),
                "The hill fraction must lie in [0, 1] and its crest left of "
                "the right edge of the box.");
  }

  // Topography height at (x, y) (y ignored in 2-D). The cross-section
  // through the crest is a Gaussian flank for x <= tx and, in hill mode,
  // a half-cosine ramp for x > tx — zero slope at the crest (so the
  // profile stays C^1 there) and at the right edge of the box, where a
  // fraction `hill` of the height remains: the asymmetric range-front /
  // long-back-slope shape of active tectonics. In 3-D the cross-section
  // is modulated by the Gaussian in y; with hill < 0 the product is
  // exactly the original radially symmetric Gaussian.
  mfem::real_t Topography(int dim, mfem::real_t x, mfem::real_t y) const {
    const mfem::real_t dx = x - tx;
    mfem::real_t h;
    if (hill >= 0.0 && dx > 0.0) {
      constexpr mfem::real_t pi = std::numbers::pi_v<mfem::real_t>;
      const mfem::real_t u = std::min(dx / (0.5 * W - tx), mfem::real_t{1});
      h = h0 * (hill + (1.0 - hill) * 0.5 * (1.0 + std::cos(pi * u)));
    } else {
      h = h0 * std::exp(-dx * dx / (2.0 * tr * tr));
    }
    if (dim == 3) {
      h *= std::exp(-y * y / (2.0 * tr * tr));
    }
    return h;
  }

  // The factor scaling both moduli.
  std::function<mfem::real_t(const mfem::Vector&)> ModulusFactor(
      int dim) const {
    return moduli.Make(dim, W, D);
  }

  // The serial mesh, mapped: horizontal coordinates centred, the free
  // surface at z = 0, the topography blended in so the bottom stays flat
  // (z -> z + h(x,y)(1 + z/D)) and element faces follow the surface.
  // With lid_rows > 0 the top that many element rows get attribute 2
  // (the rest keep 1), assigned BEFORE the mapping so the lid base is a
  // plane of element faces, bent with the mesh.
  mfem::Mesh MakeMesh(int dim, int lid_rows = 0) const {
    using mfem::Element;
    using mfem::Mesh;
    using mfem::Vector;
    MFEM_VERIFY(lid_rows >= 0 && lid_rows < nz,
                "The lid must be between 0 and nz - 1 element rows.");
    Mesh mesh = dim == 2 ? Mesh::MakeCartesian2D(
                               nx, nz, Element::QUADRILATERAL, false, W, D)
                         : Mesh::MakeCartesian3D(
                               nx, nx, nz, Element::HEXAHEDRON, W, W, D);
    if (lid_rows > 0) {
      const mfem::real_t z_lid = D * (1.0 - mfem::real_t(lid_rows) / nz);
      Vector centre(dim);
      for (int e = 0; e < mesh.GetNE(); e++) {
        mesh.GetElementCenter(e, centre);
        mesh.SetAttribute(e, centre(dim - 1) > z_lid ? 2 : 1);
      }
      mesh.SetAttributes();
    }
    mesh.Transform([this, dim](const Vector& p, Vector& q) {
      q.SetSize(dim);
      q(0) = p(0) - 0.5 * W;
      if (dim == 3) {
        q(1) = p(1) - 0.5 * W;
      }
      const mfem::real_t z = p(dim - 1) - D;  // [-D, 0]
      q(dim - 1) =
          z + Topography(dim, q(0), dim == 3 ? q(1) : 0.0) * (1.0 + z / D);
    });
    return mesh;
  }

  int TopAttribute(int dim) const { return dim == 2 ? 3 : 6; }
};

// The moment density of a smoothed fault: the fixed double couple times
// a sum of normalised Gaussians, one per patch,
//   m(x) = couple * sum_p m0_p G_sigma(x - x_p),
// the smooth stand-in for the point sources through the SAME integrator's
// volume path (DomainLFDeformationGradientIntegrator on a non-delta
// coefficient). Away from the fault — a few sigma — it agrees with the
// point sources to O(sigma^2 / r^2); near the fault it replaces the
// mesh-scale delta artifacts by a resolved density. A Gaussian tail
// falling outside the body (a shallow fault) is simply lost: the load
// then carries slightly less moment, which the examples report.
class SmoothedMomentDensity : public mfem::MatrixCoefficient {
 public:
  SmoothedMomentDensity(const mfem::DenseMatrix& couple,
                        std::vector<mfem::Vector> centres,
                        std::vector<mfem::real_t> moments, mfem::real_t sigma)
      : mfem::MatrixCoefficient(couple.Height()),
        couple_(couple),
        centres_(std::move(centres)),
        moments_(std::move(moments)),
        sigma_(sigma),
        norm_(std::pow(2.0 * std::numbers::pi_v<mfem::real_t> * sigma *
                           sigma,
                       -0.5 * couple.Height())) {}

  void Eval(mfem::DenseMatrix& M, mfem::ElementTransformation& T,
            const mfem::IntegrationPoint& ip) override {
    const int dim = couple_.Height();
    mfem::Vector x(dim);
    T.Transform(ip, x);
    mfem::real_t rho = 0.0;
    for (std::size_t p = 0; p < centres_.size(); p++) {
      mfem::real_t r2 = 0.0;
      for (int c = 0; c < dim; c++) {
        const mfem::real_t d = x(c) - centres_[p](c);
        r2 += d * d;
      }
      rho += moments_[p] * std::exp(-r2 / (2.0 * sigma_ * sigma_));
    }
    M = couple_;
    M *= norm_ * rho;
  }

 private:
  mfem::DenseMatrix couple_;
  std::vector<mfem::Vector> centres_;
  std::vector<mfem::real_t> moments_;
  mfem::real_t sigma_, norm_;
};

// The planar fault: centre, orientation, extent, slip and its layout as
// point-source patches, or as the smoothed density when smooth > 0.
struct Fault {
  mfem::real_t dip = 30.0, rake = 90.0, strike = 0.0;
  mfem::real_t depth = 1.0, x = 0.0;
  mfem::real_t length = 1.5, width = 1.5;
  mfem::real_t slip = 0.01;
  int nd = 12, ns = 12;
  mfem::real_t smooth = 0.0;  // Gaussian sigma in element widths; 0: deltas
  bool taper = true;          // cos^2 slip taper to zero at the fault ends

  void AddOptions(mfem::OptionsParser& args) {
    args.AddOption(&dip, "-dip", "--dip",
                   "Dip of the fault in degrees from horizontal, towards "
                   "+x.");
    args.AddOption(&rake, "-rake", "--rake",
                   "Rake in degrees (3-D; 90 thrust, 0 left-lateral "
                   "strike-slip). 2-D is pure dip-slip.");
    args.AddOption(&strike, "-strike", "--strike",
                   "Strike in degrees clockwise from the y axis (3-D).");
    args.AddOption(&depth, "-fd", "--fault-depth",
                   "Depth of the fault centre below z = 0.");
    args.AddOption(&x, "-fx", "--fault-x", "x of the fault centre.");
    args.AddOption(&length, "-fl", "--fault-length",
                   "Along-dip length of the fault.");
    args.AddOption(&width, "-fw", "--fault-width",
                   "Along-strike width of the fault (3-D).");
    args.AddOption(&slip, "-s", "--slip",
                   "Slip; in 2-D the sign selects thrust (> 0) or normal.");
    args.AddOption(&nd, "-nd", "--num-dip-patches",
                   "Point-source patches along dip (with -ns 1: a point "
                   "source).");
    args.AddOption(&ns, "-ns", "--num-strike-patches",
                   "Point-source patches along strike (3-D).");
    args.AddOption(&smooth, "-smooth", "--source-smoothing",
                   "Replace each point source by a normalised Gaussian of "
                   "this sigma in element widths (0: point sources).");
    args.AddOption(&taper, "-taper", "--taper-slip", "-no-taper",
                   "--no-taper-slip",
                   "Taper the slip as cos^2 to zero at the fault ends "
                   "(3-D: all edges); the slip is then the peak, at the "
                   "centre. Patch centres sit at midpoints, so -nd 1 is a "
                   "single full-amplitude point source.");
  }

  struct Sources {
    // Point-source mode: one MatrixDeltaCoefficient per patch. Smoothed
    // mode (smooth > 0): the single volume coefficient instead, with
    // sigma its Gaussian width.
    std::vector<std::unique_ptr<AdGIA::MatrixDeltaCoefficient>> patches;
    std::unique_ptr<SmoothedMomentDensity> smoothed;
    mfem::real_t sigma = 0.0;
    mfem::DenseMatrix couple;   // the unit double couple u n^T + n u^T
    mfem::real_t moment = 0.0;  // total scalar moment (2-D: per unit strike)
    mfem::real_t moment_signed = 0.0;  // sum of the signed patch moments
  };

  // The fault frame — down-dip direction d, upward unit normal n and the
  // hanging wall's slip direction u — and one MatrixDeltaCoefficient per
  // patch, scaled by its moment mu(x_p) * slip * dA from the local shear
  // modulus @p mu. In 3-D the frame is built from the strike s and the
  // horizontal dip direction by cross products (x east, y north, z up;
  // strike clockwise from north, rake anticlockwise in the fault plane
  // from the strike); 2-D is the strike = 0, rake = 90 section, with the
  // sense carried by the sign of the slip.
  Sources Build(int dim, const FaultBox& box,
                const std::function<mfem::real_t(const mfem::Vector&)>& mu)
      const {
    using mfem::DenseMatrix;
    using mfem::real_t;
    using mfem::Vector;
    using AdGIA::MatrixDeltaCoefficient;
    MFEM_VERIFY(dip >= 0.0 && dip <= 90.0,
                "Dip must lie in [0, 90] degrees.");
    MFEM_VERIFY(nd >= 1 && ns >= 1, "At least one patch in each direction.");

    constexpr real_t deg = std::numbers::pi_v<real_t> / 180.0;
    const real_t cd = std::cos(dip * deg), sd = std::sin(dip * deg);
    Vector d_hat(dim), n_hat(dim), u_hat(dim), s_hat(dim);
    s_hat = 0.0;
    if (dim == 2) {
      d_hat(0) = cd;
      d_hat(1) = -sd;
      n_hat(0) = sd;
      n_hat(1) = cd;
      u_hat = d_hat;
      u_hat *= -1.0;  // up-dip: thrust for slip > 0
    } else {
      const real_t cp = std::cos(strike * deg), sp = std::sin(strike * deg);
      const real_t cr = std::cos(rake * deg), sr = std::sin(rake * deg);
      s_hat(0) = sp;
      s_hat(1) = cp;
      d_hat(0) = cd * cp;
      d_hat(1) = -cd * sp;
      d_hat(2) = -sd;
      // n = d x s, the upward normal.
      n_hat(0) = d_hat(1) * s_hat(2) - d_hat(2) * s_hat(1);
      n_hat(1) = d_hat(2) * s_hat(0) - d_hat(0) * s_hat(2);
      n_hat(2) = d_hat(0) * s_hat(1) - d_hat(1) * s_hat(0);
      add(cr, s_hat, -sr, d_hat, u_hat);
    }

    // The unit double couple u n^T + n u^T; each patch scales it by its
    // moment, through the delta's scale or the Gaussian's weight.
    Sources out;
    out.couple.SetSize(dim);
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        out.couple(i, j) = u_hat(i) * n_hat(j) + n_hat(i) * u_hat(j);
      }
    }

    // Patch centres offset along dip and (3-D) strike from the fault
    // centre; in 2-D the moment is per unit strike length.
    const int n_strike = dim == 2 ? 1 : ns;
    const real_t dA = (length / nd) * (dim == 3 ? width / n_strike : 1.0);
    Vector centre(dim);
    centre = 0.0;
    centre(0) = x;
    centre(dim - 1) = -depth;
    std::vector<Vector> centres;
    std::vector<real_t> moments;
    Vector xp(dim);
    constexpr real_t pi = std::numbers::pi_v<real_t>;
    for (int k = 0; k < nd; k++) {
      for (int j = 0; j < n_strike; j++) {
        const real_t xi = (k + 0.5) / nd - 0.5;
        const real_t eta = (j + 0.5) / n_strike - 0.5;
        add(centre, xi * length, d_hat, xp);
        if (dim == 3) {
          xp.Add(eta * width, s_hat);
        }
        MFEM_VERIFY(xp(dim - 1) > -box.D && xp(dim - 1) < 0.0 &&
                        std::abs(xp(0)) < 0.5 * box.W &&
                        (dim == 2 || std::abs(xp(1)) < 0.5 * box.W),
                    "Fault patch outside the box: shrink the fault (-fl, "
                    "-fw, -fd, -dip) or enlarge the box (-W, -D).");
        // The slip of the patch: the peak value shaped by the cos^2
        // taper, which vanishes at the fault ends (the abrupt edge of a
        // uniform distribution is itself an artifact) and leaves a
        // single patch (xi = 0) at full amplitude.
        real_t w = 1.0;
        if (taper) {
          const real_t cx_t = std::cos(pi * xi);
          w = cx_t * cx_t;
          if (dim == 3) {
            const real_t cy_t = std::cos(pi * eta);
            w *= cy_t * cy_t;
          }
        }
        const real_t m0 = mu(xp) * slip * w * dA;
        out.moment += std::abs(m0);
        out.moment_signed += m0;
        centres.push_back(xp);
        moments.push_back(m0);
      }
    }

    // The representation: one delta per patch, or, with smooth > 0, the
    // single Gaussian density of width sigma = smooth element widths.
    if (smooth > 0.0) {
      out.sigma = smooth * std::max(box.W / box.nx, box.D / box.nz);
      out.smoothed = std::make_unique<SmoothedMomentDensity>(
          out.couple, std::move(centres), std::move(moments), out.sigma);
    } else {
      for (std::size_t p = 0; p < centres.size(); p++) {
        const Vector& c = centres[p];
        out.patches.push_back(
            dim == 2 ? std::make_unique<MatrixDeltaCoefficient>(
                           out.couple, c(0), c(1), moments[p])
                     : std::make_unique<MatrixDeltaCoefficient>(
                           out.couple, c(0), c(1), c(2), moments[p]));
      }
    }
    return out;
  }
};

// b(v): the value of an assembled linear form on a grid function, with
// the true-dof pairing in parallel (the naive local dot product would
// count shared dofs once per rank).
inline double Pair(mfem::LinearForm& b, mfem::GridFunction& v) {
#ifdef MFEM_USE_MPI
  if (auto* pb = dynamic_cast<mfem::ParLinearForm*>(&b)) {
    auto& pv = dynamic_cast<mfem::ParGridFunction&>(v);
    mfem::Vector tb(pb->ParFESpace()->GetTrueVSize());
    pb->ParallelAssemble(tb);
    mfem::Vector tv(pv.ParFESpace()->GetTrueVSize());
    pv.GetTrueDofs(tv);
    return mfem::InnerProduct(pb->ParFESpace()->GetComm(), tb, tv);
  }
#endif
  return b(v);
}

// The moment a stress-glut linear form actually carries: b paired with
// the linear field v = C x, C = couple / (couple : couple), for which
// b(v) = integral of the scalar moment density — exactly the summed
// patch moments for point sources, and slightly less for a smoothed
// fault whose Gaussian tails leave the body. Reported as the fraction of
// @p moment_signed.
inline double MomentCapture(mfem::LinearForm& b,
                            mfem::FiniteElementSpace& fes,
                            const mfem::DenseMatrix& couple,
                            double moment_signed) {
  const int dim = couple.Height();
  mfem::DenseMatrix C(couple);
  C *= 1.0 / (couple * couple);  // couple : couple
  mfem::VectorFunctionCoefficient lin(
      dim, [C, dim](const mfem::Vector& x, mfem::Vector& y) {
        y.SetSize(dim);
        C.Mult(x, y);
      });
#ifdef MFEM_USE_MPI
  mfem::ParGridFunction v(dynamic_cast<mfem::ParFiniteElementSpace*>(&fes));
#else
  mfem::GridFunction v(&fes);
#endif
  v.ProjectCoefficient(lin);
  return Pair(b, v) / moment_signed;
}

// The displacement along the free surface (the x axis; y = 0 in 3-D),
// just inside the topography. FindPoints runs once, in the constructor
// (collective); Sample() evaluates a field at the points and reduces.
class SurfaceProfile {
 public:
  SurfaceProfile(mfem::Mesh& mesh, const FaultBox& box, int dim,
                 int npts = 201)
      : dim_(dim), npts_(npts), pts_(dim, npts) {
    // The sample points sit on the ANALYTIC topography, but the mesh
    // surface is its nodal interpolant, which sags below the curve
    // between nodes by up to max|h''| dx^2 / 8; inset by that bound
    // (doubled: in 3-D the section at y = 0 can also sag in y, and the
    // ramp bound is taken at the crest) plus a round-off floor, so
    // every sample lands inside the discrete mesh.
    mfem::real_t curv = std::abs(box.h0) / (box.tr * box.tr);
    if (box.hill >= 0.0) {
      constexpr mfem::real_t pi = std::numbers::pi_v<mfem::real_t>;
      const mfem::real_t L = 0.5 * box.W - box.tx;
      curv = std::max(curv, std::abs(box.h0) * (1.0 - box.hill) * 0.5 * pi *
                                pi / (L * L));
    }
    const mfem::real_t dx = box.W / box.nx;
    const mfem::real_t inset = 1e-6 * box.D + 0.25 * curv * dx * dx;
    for (int i = 0; i < npts; i++) {
      const mfem::real_t x = -0.5 * box.W + i * (box.W / (npts - 1));
      pts_(0, i) = std::clamp(x, -0.5 * box.W + inset, 0.5 * box.W - inset);
      if (dim == 3) {
        pts_(1, i) = 0.0;
      }
      pts_(dim - 1, i) = box.Topography(dim, pts_(0, i), 0.0) - inset;
    }
    mesh.FindPoints(pts_, elem_, ips_);
  }

  int NumPoints() const { return npts_; }
  mfem::real_t X(int i) const { return pts_(0, i); }

  // The index of the sample point nearest x.
  int Nearest(mfem::real_t x) const {
    int best = 0;
    for (int i = 1; i < npts_; i++) {
      if (std::abs(pts_(0, i) - x) < std::abs(pts_(0, best) - x)) {
        best = i;
      }
    }
    return best;
  }

  // u at the points, component-major (component c of point i at
  // c * NumPoints() + i), identical on every rank.
  std::vector<double> Sample(const mfem::GridFunction& u) const {
    std::vector<double> values(static_cast<std::size_t>(dim_) * npts_, 0.0);
    mfem::Vector val(dim_);
    for (int i = 0; i < npts_; i++) {
      if (elem_[i] >= 0) {
        u.GetVectorValue(elem_[i], ips_[i], val);
        for (int c = 0; c < dim_; c++) {
          values[c * npts_ + i] = val(c);
        }
      }
    }
#ifdef MFEM_USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, values.data(), values.size(), MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
#endif
    return values;
  }

 private:
  int dim_, npts_;
  mfem::DenseMatrix pts_;
  mfem::Array<int> elem_;
  mfem::Array<mfem::IntegrationPoint> ips_;
};

// The (min, max) of a scalar field over the free surface: its values on
// the top-boundary dofs, reduced over the ranks.
inline std::pair<mfem::real_t, mfem::real_t> SurfacePeaks(
    const mfem::GridFunction& u_z, mfem::FiniteElementSpace& fes_z,
    int top_attr) {
  mfem::Array<int> top_marker(
      fes_z.GetMesh()->bdr_attributes.Max()),
      on_top;
  top_marker = 0;
  top_marker[top_attr - 1] = 1;
  fes_z.GetEssentialVDofs(top_marker, on_top);
  mfem::real_t mn = 0.0, mx = 0.0;
  for (int i = 0; i < u_z.Size(); i++) {
    if (on_top[i]) {
      mn = std::min(mn, u_z(i));
      mx = std::max(mx, u_z(i));
    }
  }
#ifdef MFEM_USE_MPI
  MPI_Allreduce(MPI_IN_PLACE, &mn, 1,
                mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_MIN,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &mx, 1,
                mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_MAX,
                MPI_COMM_WORLD);
#endif
  return {mn, mx};
}

}  // namespace examples
