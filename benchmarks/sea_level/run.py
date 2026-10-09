"""Run the sea-level fingerprint benchmark: AdGIA's monolithic solve
against pyslfp's pseudo-spectral solution of the same problem (rung 1 of
doc/planning/sea_level_plan.md).

One spherically layered model (homogeneous by default), one smooth
analytical initial state (continent at the north pole, ice cap on it)
and one melt load, defined once here in the case's non-dimensional
units and handed to both sides:

  - AdGIA: a case made with make_case.py (reused if present), solved by
    sea_level_benchmark under MPI (SetWaterLoad; frozen shorelines, no
    rotation); the fingerprint comes back as nodal CSV.
  - pyslfp: Love numbers computed directly from the same planetmodel
    model (EarthModel.from_planet_model), the same state as SHGrids,
    LinearSeaLevelEquation without rotational feedbacks.

Both fingerprints are converted to metres; the AdGIA nodes are
interpolated onto pyslfp's grid (local thin-plate RBF on the unit
directions) and compared: relative RMS and maximum over the ocean, the
eustatic values, and the per-degree amplitudes of the difference.
Everything lands in <out>/<model>/h<h>/ as sea_level.{csv,json},
reference.json and report.json.

At low resolution the two sides need not agree closely (this script
builds the machinery; the resolved comparison belongs to the server
campaign). The degree-1 row is listed separately: the reference-frame
convention there has not yet been reconciled.

    python run.py --h 0.4 --order 2 --np 4 --lmax 32
    python run.py --model homogeneous --h 0.2 --order 2 --np 8

The build makes a launcher, <build>/benchmarks/sea_level/run, meant to
be started there so that `runs` is in the build tree.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))

import models  # noqa: E402
from drivers import find_programs, run  # noqa: E402
from outputs import outside_source  # noqa: E402

# The analytical state, in the case's non-dimensional units (length in
# planet radii, density in units of models.DENSITY_SCALE). The physical
# intent: a 4 km ocean, a 9 km continent, a 3 km polar ice cap.
STATE = {
    "ocean_depth": 4000.0 / models.RADIUS,
    "cont_amp": 9000.0 / models.RADIUS,
    "cont_width": 0.7,
    "cap_amp": 3000.0 / models.RADIUS,
    "cap_width": 0.35,
    "melt": 0.5,
    "melt_width": 0.15,
}


def state_grids(em, rho_w_case, rho_i_case, shore):
    """The initial state and the melt load on pyslfp's grid, in the
    CASE's units (conversion happens at the comparison)."""
    lats, lons = np.meshgrid(em.lats(), em.lons(), indexing="ij")
    colat = np.deg2rad(90.0 - lats)
    t6 = (colat / STATE["cont_width"]) ** 6
    sl0 = STATE["ocean_depth"] - STATE["cont_amp"] * np.exp(-0.5 * t6)
    t2 = (colat / STATE["cap_width"]) ** 2
    ice = STATE["cap_amp"] * np.exp(-0.5 * t2)
    q = rho_w_case * sl0 - rho_i_case * ice
    frac = 0.5 * (1.0 + np.tanh(q / shore))
    # The melt unloads the +x hemisphere of the cap, smoothly (regular
    # at the pole: the factor is in the Cartesian coordinate).
    xdir = np.cos(np.deg2rad(lats)) * np.cos(np.deg2rad(lons))
    hemi = 0.5 * (1.0 + np.tanh(xdir / STATE["melt_width"]))
    melt_load = -(1.0 - frac) * rho_i_case * STATE["melt"] * hemi * ice
    return sl0, ice, melt_load


def rbf_to_grid(xyz, values, em):
    from scipy.interpolate import RBFInterpolator
    d = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    lats, lons = np.meshgrid(np.deg2rad(em.lats()), np.deg2rad(em.lons()),
                             indexing="ij")
    target = np.column_stack([
        (np.cos(lats) * np.cos(lons)).ravel(),
        (np.cos(lats) * np.sin(lons)).ravel(),
        np.sin(lats).ravel(),
    ])
    rbf = RBFInterpolator(d, values, neighbors=min(64, values.size),
                          kernel="thin_plate_spline")
    return rbf(target).reshape(lats.shape)



INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ADGIA_C, PYSLFP_C, DIFF_C = "#2a78d6", "#eb6834", "#52514e"


def figure(path, em, adgia, ref, frac, report):
    """Maps of the two fingerprints (shared diverging scale), their
    difference, and the per-degree amplitudes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lats, lons = em.lats(), em.lons()
    lim = max(np.max(np.abs(adgia)), np.max(np.abs(ref)))
    dlim = np.max(np.abs(adgia - ref))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, field, title, vlim in (
            (axes[0, 0], adgia, "AdGIA (monolithic)", lim),
            (axes[0, 1], ref, "pyslfp (pseudo-spectral)", lim),
            (axes[1, 0], adgia - ref, "difference", dlim)):
        m = ax.pcolormesh(lons, lats, field, cmap="RdBu_r",
                          vmin=-vlim, vmax=vlim, rasterized=True)
        ax.contour(lons, lats, frac, levels=[0.5], colors=INK,
                   linewidths=0.6)
        ax.set_title(title, color=INK, fontsize=10)
        ax.set_xlabel("longitude", color=MUTED, fontsize=8)
        ax.set_ylabel("latitude", color=MUTED, fontsize=8)
        ax.tick_params(colors=MUTED, labelsize=7)
        fig.colorbar(m, ax=ax, shrink=0.85, label="sea-level change (m)")
    ax = axes[1, 1]
    ls = [row["l"] for row in report["degrees"]]
    for key, colour, label in (("adgia", ADGIA_C, "AdGIA"),
                               ("pyslfp", PYSLFP_C, "pyslfp"),
                               ("diff", DIFF_C, "difference")):
        ax.semilogy(ls, [row[key] for row in report["degrees"]],
                    marker="o", ms=4, lw=1.5, color=colour, label=label)
    ax.set_xlabel("degree l", color=MUTED, fontsize=8)
    ax.set_ylabel("amplitude (m)", color=MUTED, fontsize=8)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(color=GRID, lw=0.5)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("per-degree amplitudes", color=INK, fontsize=10)
    fig.suptitle(
        f"{report['model']}, h = {report['h']:g}, order {report['order']}, "
        f"pyslfp lmax {report['lmax']}: ocean-RMS rel diff "
        f"{report['diff_ocean_rel']:.3f}", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="homogeneous",
                   help="model name from common/models.py")
    p.add_argument("--h", type=float, default=0.4, help="element size")
    p.add_argument("--order", type=int, default=2)
    p.add_argument("--np", type=int, default=4)
    p.add_argument("--lmax", type=int, default=32,
                   help="pyslfp truncation degree")
    p.add_argument("--buffer", type=float, default=0.5)
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--programs", type=Path, default=None)
    p.add_argument("--mpiexec", default=None)
    p.add_argument("--rot", action="store_true",
                   help="rotational feedback on both sides, with pyslfp's "
                        "Earth rotation rate and principal moments as the "
                        "shared data")
    p.add_argument("--remake", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    programs = find_programs(args.programs)
    out = outside_source(args.out)
    case = out / args.model / f"h{args.h:g}"
    case.mkdir(parents=True, exist_ok=True)

    # pyslfp first: it also supplies the densities, so that the flotation
    # conventions of the two sides agree by construction.
    import pyslfp
    em = pyslfp.EarthModel.from_planet_model(
        models.model(args.model), args.lmax)
    ps = em.parameters
    L = models.RADIUS            # case length scale, metres
    D = models.DENSITY_SCALE     # case density scale, kg/m^3
    rho_w_si = ps.water_density * ps.density_scale
    rho_i_si = ps.ice_density * ps.density_scale
    rho_w_case = rho_w_si / D
    rho_i_case = rho_i_si / D
    shore = 0.05 * rho_w_case * STATE["ocean_depth"]

    # The rotation data, shared by both sides: pyslfp's Earth rotation
    # rate and principal moments (the traditional theory takes them as
    # data), in case units for the driver. The case time scale is
    # 1/sqrt(G rho_scale), its inertia scale rho_scale L^5.
    G_SI = ps.raw_gravitational_constant
    T_case = 1.0 / np.sqrt(G_SI * D)
    Omega_case = ps.raw_rotation_frequency * T_case
    C_case = ps.raw_polar_moment_of_inertia / (D * L**5)
    A_case = ps.raw_equatorial_moment_of_inertia / (D * L**5)

    # The case and the AdGIA solve.
    if args.remake or not (case / "case.json").exists():
        ok = run([sys.executable,
                  str(HERE.parent / "common" / "make_case.py"), args.model,
                  "--h", f"{args.h:g}", "--buffer", f"{args.buffer:g}",
                  "--lmax", "10", "--out", str(case)],
                 log=None, dry_run=args.dry_run)
        if not ok:
            raise SystemExit(f"{case}: make_case.py failed")
    tag = f"o{args.order}" + ("_rot" if args.rot else "")
    slcsv = case / f"sea_level_{tag}.csv"
    sljson = case / f"sea_level_{tag}.json"
    if args.force or not sljson.exists():
        mpiexec = args.mpiexec or "mpiexec"
        cmd = [mpiexec, "-np", str(args.np),
               str(programs / "sea_level_benchmark"),
               "-c", str(case / "case.json"), "-o", str(args.order),
               "-method", "dahlen",
               "-rhow", f"{rho_w_case}", "-rhoi", f"{rho_i_case}",
               "-ocean-depth", f"{STATE['ocean_depth']}",
               "-cont-amp", f"{STATE['cont_amp']}",
               "-cont-width", f"{STATE['cont_width']}",
               "-cap-amp", f"{STATE['cap_amp']}",
               "-cap-width", f"{STATE['cap_width']}",
               "-melt", f"{STATE['melt']}",
               "-melt-width", f"{STATE['melt_width']}",
               "-shore", f"{shore}",
               "-slcsv", str(slcsv), "-out", str(sljson)]
        if args.rot:
            cmd += ["-Omega", f"{Omega_case}", "-C1", f"{A_case}",
                    "-C2", f"{A_case}", "-C3", f"{C_case}"]
        ok = run(cmd, log=case / f"log_{tag}.txt",
                 dry_run=args.dry_run)
        if not ok:
            raise SystemExit("sea_level_benchmark failed")
    if args.dry_run:
        return

    # The pyslfp fingerprint of the same problem.
    sl0, ice, melt_load = state_grids(em, rho_w_case, rho_i_case, shore)
    to_ps_len = L / ps.length_scale
    to_ps_sigma = (D * L) / (ps.density_scale * ps.length_scale)
    import pyshtools
    sl0_g = pyshtools.SHGrid.from_array(sl0 * to_ps_len, grid=em.grid)
    ice_g = pyshtools.SHGrid.from_array(ice * to_ps_len, grid=em.grid)
    load_g = pyshtools.SHGrid.from_array(melt_load * to_ps_sigma,
                                         grid=em.grid)
    state = pyslfp.EarthState(ice_g, sl0_g, em, exclude_caspian=False)
    lin = pyslfp.LinearSeaLevelEquation(state)
    sl_ps, _, _, omega_ps = lin.solve_sea_level_equation(
        load_g, rotational_feedbacks=args.rot)
    ref = sl_ps.data * ps.length_scale  # metres

    # The AdGIA fingerprint, gridded and in metres.
    rows = np.genfromtxt(slcsv, delimiter=",", names=True)
    xyz = np.column_stack([rows["x"], rows["y"], rows["z"]])
    adgia = rbf_to_grid(xyz, np.asarray(rows["value"]), em) * L

    # The comparison, over the ocean (the state's own fraction).
    q = rho_w_case * sl0 - rho_i_case * ice
    frac = 0.5 * (1.0 + np.tanh(q / shore))
    wlat = np.cos(np.deg2rad(np.meshgrid(em.lats(), em.lons(),
                                         indexing="ij")[0]))
    diff = adgia - ref
    def wrms(f, mask):
        return float(np.sqrt(np.sum(mask * wlat * f * f) /
                             np.sum(mask * wlat)))
    scale = wrms(ref, frac)
    report = {
        "model": args.model, "h": args.h, "order": args.order,
        "lmax": args.lmax,
        "ref_ocean_rms_m": scale,
        "diff_ocean_rms_m": wrms(diff, frac),
        "diff_ocean_rel": wrms(diff, frac) / scale,
        "diff_max_m": float(np.max(np.abs(diff))),
        "adgia": json.loads(sljson.read_text()),
    }
    if args.rot:
        # omega / Omega on both sides (dimensionless; components
        # [wander_1, wander_2, spin]). The wander rows carry the
        # near-neutral amplification of any convention or
        # discretisation difference (C - A is small for the Earth
        # data), so they are reported, not gated.
        m_ps = (np.asarray(omega_ps) / ps.rotation_frequency).tolist()
        om_case = report["adgia"].get("omega", [])
        m_ad = [o / Omega_case for o in om_case]
        report["omega_over_Omega"] = {"adgia": m_ad, "pyslfp": m_ps}
    # Per-degree amplitudes of both fields and the difference.
    ca = pyshtools.SHGrid.from_array(adgia, grid=em.grid).expand(
        normalization="ortho")
    cr = pyshtools.SHGrid.from_array(ref, grid=em.grid).expand(
        normalization="ortho")
    power_a = ca.spectrum()
    power_r = cr.spectrum()
    power_d = (ca - cr).spectrum()
    lshow = min(8, args.lmax)
    report["degrees"] = [
        {"l": l, "adgia": float(np.sqrt(power_a[l])),
         "pyslfp": float(np.sqrt(power_r[l])),
         "diff": float(np.sqrt(power_d[l]))}
        for l in range(lshow + 1)
    ]
    (case / f"report_{tag}.json").write_text(
        json.dumps(report, indent=2) + "\n")
    figure(case / f"fingerprint_{tag}.png", em, adgia, ref, frac,
           report)

    print(f"\n{args.model}, h = {args.h:g}, order {args.order}, "
          f"pyslfp lmax {args.lmax}")
    print(f"  ocean RMS   pyslfp {scale:.4g} m,  "
          f"diff {report['diff_ocean_rms_m']:.4g} m  "
          f"(rel {report['diff_ocean_rel']:.3f})")
    print("  l    |SL|_adgia    |SL|_pyslfp   |diff|")
    for row in report["degrees"]:
        note = "   (frame convention unreconciled)" if row["l"] == 1 else ""
        print(f"  {row['l']:<3} {row['adgia']:<13.4g} "
              f"{row['pyslfp']:<13.4g} {row['diff']:.4g}{note}")
    if args.rot:
        mm = report["omega_over_Omega"]
        print("  omega/Omega  adgia  " +
              " ".join(f"{v: .4g}" for v in mm["adgia"]))
        print("               pyslfp " +
              " ".join(f"{v: .4g}" for v in mm["pyslfp"]) +
              "   (wander rows amplification-prone, reported not gated)")
    print(f"  report: {case / f'report_{tag}.json'}")
    print(f"  figure: {case / f'fingerprint_{tag}.png'}")


if __name__ == "__main__":
    main()
