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

--nonlinear (rung 3) adds shoreline migration: AdGIA's Picard loop
(-mig) against pyslfp's nonlinear solver, both from the same initial
state and ice change. The frozen/linear pair is solved too, so the
report carries the migration effect delta = SL(migrating) - SL(frozen)
on each side — the sharp probe, since the common linear part cancels.
Convention note: pyslfp's nonlinear ocean updates are sharp where ours
are smoothed over the `shore` width, so the delta comparison is
shoreline-dominated at toy resolution and tightens as shore shrinks
with h on the resolved ladder.

--earth swaps the analytic state for the real present-day geography:
the ICE-7G sea level and ice at --lmax (loaded by pyslfp), a SMALL
melt of one ice sheet (--melt-region greenland | west-antarctica, the
smooth windows of examples/sea_level_fingerprint.cpp; --melt-fraction,
default 1% — both sides are linear, so the relative comparison is
independent of it and the stated problem stays in the fingerprint
regime, flotation changes negligible), and a case mesh
refined along the real coastlines (the committed grid of
meshes/earth_coastlines.py through make_case.py --coast-grid). The
state and melt are built once here on pyslfp's grid; the driver gets
them sampled at its surface nodes (-state/-ice/-dice; the node set is
mesh- and order-specific, so the sampling is per case). Everything
downstream — the comparison, the report, the figure — is the simple
leg's, so the two geographies read side by side. Linear and frozen-C
only for now (no --nonlinear, --timings or --coast with it).

--timings (no comparison) times the solve across the feature axis —
the same melt load as a plain elastic solve (-no-water), with the
water feedback, with rotation, and with shoreline migration — min and
mean of --repeat runs, with ratios against the elastic control.

    python run.py --h 0.4 --order 2 --np 4 --lmax 32
    python run.py --model homogeneous --h 0.2 --order 2 --np 8
    python run.py --nonlinear
    python run.py --timings --repeat 5
    python run.py --earth --lmax 64 --np 8
    python run.py --earth --melt-region west-antarctica --h 0.3 --np 8

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

# The real-geography leg (--earth): the present-day ICE-7G state and a
# melt of one ice sheet, confined by the smooth geographic windows of
# examples/sea_level_fingerprint.cpp (degrees: lat_min, lat_max,
# lon_min, lon_max; the edge smoothing in radians).
REGIONS = {
    "greenland": (59.0, 84.0, 285.0, 350.0),
    "west-antarctica": (-85.0, -72.0, 190.0, 300.0),
}
REGION_EDGE = 0.05


def earth_state_grids(em, rho_w_case, rho_i_case, shore, region, lmax,
                      melt_fraction):
    """The ICE-7G initial state, the sheet melt and the frozen-C melt
    load on pyslfp's grid, in case units — the --earth counterpart of
    state_grids(), defined once here for both sides. The melt fraction
    is SMALL by default: both sides are linear, so the relative
    comparison is independent of it, and a small melt keeps the stated
    problem in the fingerprint regime, where the frozen flotation is
    the physics and not just the linearisation."""
    from pyslfp.ice import IceNG
    ice_g, sl_g = IceNG().get_ice_thickness_and_sea_level(0.0, lmax)
    assert ice_g.data.shape == (em.lats().size, em.lons().size)
    sl0 = sl_g.data / models.RADIUS
    ice = ice_g.data / models.RADIUS
    lats, lons = np.meshgrid(np.deg2rad(em.lats()), np.deg2rad(em.lons()),
                             indexing="ij")

    def smooth_edge(d):
        return 0.5 * (1.0 + np.tanh(d / REGION_EDGE))

    lat0, lat1, lon0, lon1 = (np.deg2rad(v) for v in REGIONS[region])
    mask = (smooth_edge(lats - lat0) * smooth_edge(lat1 - lats) *
            smooth_edge(lons - lon0) * smooth_edge(lon1 - lons))
    dice = -melt_fraction * mask * ice
    q = rho_w_case * sl0 - rho_i_case * ice
    frac = 0.5 * (1.0 + np.tanh(q / shore))
    melt_load = (1.0 - frac) * rho_i_case * dice
    return sl0, ice, dice, melt_load


def write_node_stacks(em, nodes_csv, grids):
    """Sample wrapped em-grid fields at the driver's exported surface
    nodes (bilinear; the grids include the poles and the 360-degree
    column) and write each as a one-column IceHistory stack CSV."""
    from scipy.interpolate import RegularGridInterpolator
    rows = np.genfromtxt(nodes_csv, delimiter=",", names=True)
    xyz = np.column_stack([rows["x"], rows["y"], rows["z"]])
    d = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    lat = np.rad2deg(np.arcsin(np.clip(d[:, 2], -1.0, 1.0)))
    lon = np.rad2deg(np.arctan2(d[:, 1], d[:, 0])) % 360.0
    order = np.argsort(em.lats())
    for path, grid in grids.items():
        interp = RegularGridInterpolator(
            (np.asarray(em.lats())[order], np.asarray(em.lons())),
            np.asarray(grid)[order, :], bounds_error=False,
            fill_value=None)
        values = interp(np.column_stack([lat, lon]))
        with open(path, "w") as f:
            f.write("x,y,z,0\n")
            for p, v in zip(xyz, values):
                f.write(f"{p[0]:.16g},{p[1]:.16g},{p[2]:.16g},{v:.16g}\n")


def state_grids(em, rho_w_case, rho_i_case, shore):
    """The initial state, the ice-thickness change and the (frozen-C)
    melt load on pyslfp's grid, in the CASE's units (conversion happens
    at the comparison)."""
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
    dice = -STATE["melt"] * hemi * ice
    melt_load = (1.0 - frac) * rho_i_case * dice
    return sl0, ice, dice, melt_load


def shoreline_colatitude(rho_w_case, rho_i_case):
    """The colatitude of the state's shoreline, rho_w SL0 = rho_i I0
    (axisymmetric: one ring; the melt does not move the frozen C0), by
    bisection between the dry pole and the open ocean."""
    def flot(theta):
        sl0 = (STATE["ocean_depth"] - STATE["cont_amp"] *
               np.exp(-0.5 * (theta / STATE["cont_width"]) ** 6))
        ice = STATE["cap_amp"] * np.exp(
            -0.5 * (theta / STATE["cap_width"]) ** 2)
        return rho_w_case * sl0 - rho_i_case * ice
    lo, hi = 1e-3, 0.5 * np.pi
    assert flot(lo) < 0.0 < flot(hi)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if flot(mid) < 0.0 else (lo, mid)
    return 0.5 * (lo + hi)


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


def degree_rows(adgia, ref, em, lmax):
    """Per-degree amplitudes of the two gridded fields and their
    difference (orthonormalised power), through min(8, lmax)."""
    import pyshtools
    ca = pyshtools.SHGrid.from_array(adgia, grid=em.grid).expand(
        normalization="ortho")
    cr = pyshtools.SHGrid.from_array(ref, grid=em.grid).expand(
        normalization="ortho")
    power_a, power_r = ca.spectrum(), cr.spectrum()
    power_d = (ca - cr).spectrum()
    return [
        {"l": l, "adgia": float(np.sqrt(power_a[l])),
         "pyslfp": float(np.sqrt(power_r[l])),
         "diff": float(np.sqrt(power_d[l]))}
        for l in range(min(8, lmax) + 1)
    ]


def timings(args, case, solve, rot_args):
    """Time the solve across the feature axis: the same melt load as a
    plain elastic solve, with the water feedback, with rotation, with
    shoreline migration. Best (and mean) of --repeat runs each, ratios
    against the elastic control; no pyslfp side."""
    rows = [("elastic", ["-no-water"]),
            ("water", []),
            ("water_mono", ["-feedback", "monolithic"]),
            ("water_rotation", rot_args),
            ("water_rot_mono", rot_args + ["-feedback", "monolithic"]),
            # Tight migration, as the comparison runs it on this stack
            # (the loose path only pays the contraction guard's rescue
            # here).
            ("water_migration", ["-mig", "-mig-inexact", "0"])]
    results = {}
    for name, extra in rows:
        secs, info = [], None
        for _ in range(max(1, args.repeat)):
            _, sljson = solve(f"timing_{name}_o{args.order}", extra,
                              force=True)
            if args.dry_run:
                continue
            info = json.loads(sljson.read_text())
            secs.append(info["solve_seconds"])
        if args.dry_run:
            continue
        row = {"solve_seconds_min": min(secs),
               "solve_seconds_mean": sum(secs) / len(secs),
               "iterations": info["iterations"]}
        if info.get("migrate"):
            row["mig_passes"] = info["mig_passes"]
            row["mig_outer_iterations"] = info["mig_outer_iterations"]
        results[name] = row
    if args.dry_run:
        return
    base = results["elastic"]["solve_seconds_min"]
    for row in results.values():
        row["ratio_vs_elastic"] = row["solve_seconds_min"] / base
    out = case / f"timings_o{args.order}.json"
    out.write_text(json.dumps(
        {"model": args.model, "h": args.h, "order": args.order,
         "np": args.np, "repeat": args.repeat, "rows": results},
        indent=2) + "\n")
    print(f"\nsolve timings: {args.model}, h = {args.h:g}, "
          f"order {args.order}, np {args.np}, best of {args.repeat}")
    print(f"  {'configuration':<16} {'min (s)':>9} {'mean (s)':>9} "
          f"{'x elastic':>10}  iterations")
    for name, row in results.items():
        note = (f" ({row['mig_passes']} extra passes, "
                f"{row['mig_outer_iterations']} outer)"
                if "mig_passes" in row else "")
        print(f"  {name:<16} {row['solve_seconds_min']:>9.3f} "
              f"{row['solve_seconds_mean']:>9.3f} "
              f"{row['ratio_vs_elastic']:>10.2f}  "
              f"{row['iterations']}{note}")
    print(f"  written: {out}")



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
    p.add_argument("--nonlinear", action="store_true",
                   help="rung 3: shoreline migration (AdGIA -mig) against "
                        "pyslfp's nonlinear solver; the frozen/linear pair "
                        "is solved too, for the migration-effect delta")
    p.add_argument("--timings", action="store_true",
                   help="time the solve across the feature axis (elastic / "
                        "water / rotation / migration) instead of comparing")
    p.add_argument("--repeat", type=int, default=3,
                   help="timing repetitions per configuration (--timings)")
    p.add_argument("--coast", action="store_true",
                   help="coastline-refined mesh: a near_curve ring at the "
                        "state's shoreline (planetmodel >= 1.2.4); the case "
                        "lands in its own h<h>_coast directory")
    p.add_argument("--earth", action="store_true",
                   help="real geography: the present-day ICE-7G state, a "
                        "melt of --melt-region, and the committed "
                        "real-coastline refinement; the case lands in its "
                        "own h<h>_earth directory")
    p.add_argument("--melt-region", default="greenland",
                   choices=sorted(REGIONS),
                   help="the melted ice sheet (--earth)")
    p.add_argument("--melt-fraction", type=float, default=0.01,
                   help="melted fraction of the sheet (--earth). Both "
                        "sides are linear, so the relative comparison "
                        "does not depend on it; small keeps the stated "
                        "problem in the fingerprint regime, flotation "
                        "changes negligible")
    p.add_argument("--remake", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.earth and (args.nonlinear or args.timings or args.coast):
        raise SystemExit("--earth is the linear comparison on the "
                         "real-coastline mesh: no --nonlinear, --timings "
                         "or --coast with it (yet)")
    programs = find_programs(args.programs)
    out = outside_source(args.out)
    case = out / args.model / (f"h{args.h:g}" +
                               ("_coast" if args.coast else "") +
                               ("_earth" if args.earth else ""))
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
    if args.earth:
        # The same 5% convention as the analytic state, on the data's
        # own mean ocean depth (the loader caches, so the second read
        # in earth_state_grids is free).
        from pyslfp.ice import IceNG
        _, _sl = IceNG().get_ice_thickness_and_sea_level(0.0, args.lmax)
        _sl0 = _sl.data / models.RADIUS
        shore = 0.05 * rho_w_case * float(np.mean(_sl0[_sl0 > 0.0]))
    else:
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
        cmd = [sys.executable,
               str(HERE.parent / "common" / "make_case.py"), args.model,
               "--h", f"{args.h:g}", "--buffer", f"{args.buffer:g}",
               "--lmax", "10", "--out", str(case)]
        if args.coast:
            theta = shoreline_colatitude(rho_w_case, rho_i_case)
            print(f"shoreline colatitude {theta:.4f} rad "
                  f"({np.degrees(theta):.1f} deg): ring refinement on")
            cmd += ["--refine-rings", f"{theta:.12g}"]
        if args.earth:
            cmd += ["--coast-grid",
                    str(HERE.parent.parent / "meshes" /
                        "ice7g_topography_lmax64.npz")]
        ok = run(cmd, log=None, dry_run=args.dry_run)
        if not ok:
            raise SystemExit(f"{case}: make_case.py failed")
    rot_args = ["-Omega", f"{Omega_case}", "-C1", f"{A_case}",
                "-C2", f"{A_case}", "-C3", f"{C_case}"]
    state_args = [] if args.earth else [
        "-ocean-depth", f"{STATE['ocean_depth']}",
        "-cont-amp", f"{STATE['cont_amp']}",
        "-cont-width", f"{STATE['cont_width']}",
        "-cap-amp", f"{STATE['cap_amp']}",
        "-cap-width", f"{STATE['cap_width']}",
        "-melt", f"{STATE['melt']}",
        "-melt-width", f"{STATE['melt_width']}"]
    base_cmd = [args.mpiexec or "mpiexec", "-np", str(args.np),
                str(programs / "sea_level_benchmark"),
                "-c", str(case / "case.json"), "-o", str(args.order),
                "-method", "dahlen",
                "-rhow", f"{rho_w_case}", "-rhoi", f"{rho_i_case}"] + \
        state_args + ["-shore", f"{shore}"]

    # The sampled state of the earth leg: the node set of this case and
    # order (exact nodal data — mesh- and order-specific), the grids of
    # earth_state_grids sampled there, three stack CSVs for the driver.
    earth_grids = None
    if args.earth:
        earth_grids = earth_state_grids(em, rho_w_case, rho_i_case,
                                        shore, args.melt_region, args.lmax,
                                        args.melt_fraction)
        # The node set and the state are shared between melt regions;
        # the ice change (and the run tag below) carry the region.
        region = args.melt_region.replace("-", "_")
        nodes = case / f"nodes_o{args.order}.csv"
        dice_csv = case / f"dice_o{args.order}_{region}.csv"
        stacks = {case / f"state_o{args.order}.csv": earth_grids[0],
                  case / f"ice_o{args.order}.csv": earth_grids[1],
                  dice_csv: earth_grids[2]}
        if args.remake or args.force or not nodes.exists():
            ok = run([args.mpiexec or "mpiexec", "-np", "1",
                      str(programs / "sea_level_benchmark"),
                      "-c", str(case / "case.json"), "-o", str(args.order),
                      "-method", "dahlen",
                      "-export-nodes", str(nodes)],
                     log=case / "log_nodes.txt", dry_run=args.dry_run)
            if not ok:
                raise SystemExit("node export failed")
        if not args.dry_run:
            write_node_stacks(em, nodes, stacks)
        base_cmd += ["-state", str(case / f"state_o{args.order}.csv"),
                     "-ice", str(case / f"ice_o{args.order}.csv"),
                     "-dice", str(dice_csv)]

    def solve(tag, extra, force=None):
        """One driver run (files reused unless forced); the paths of the
        fingerprint CSV and the scalars JSON."""
        slcsv = case / f"sea_level_{tag}.csv"
        sljson = case / f"sea_level_{tag}.json"
        if (args.force if force is None else force) or not sljson.exists():
            ok = run(base_cmd + extra +
                     ["-slcsv", str(slcsv), "-out", str(sljson)],
                     log=case / f"log_{tag}.txt", dry_run=args.dry_run)
            if not ok:
                raise SystemExit("sea_level_benchmark failed")
        return slcsv, sljson

    if args.timings:
        timings(args, case, solve, rot_args)
        return

    tag = (f"o{args.order}" + ("_rot" if args.rot else "") +
           (f"_{args.melt_region.replace('-', '_')}" if args.earth else ""))
    slcsv, sljson = solve(tag, rot_args if args.rot else [])
    if args.nonlinear:
        # The migration runs tight (-mig-inexact 0): this solver stack
        # amplifies a loose residual into an O(tolerance) absolute error
        # in Phi_g (the library's contraction guard would rescue the
        # loop, at more passes than solving tight from the start).
        slcsv_nl, sljson_nl = solve(
            tag + "_nl",
            (rot_args if args.rot else []) + ["-mig", "-mig-inexact", "0"])
    if args.dry_run:
        return

    # The pyslfp fingerprint of the same problem.
    if args.earth:
        sl0, ice, dice, melt_load = earth_grids
    else:
        sl0, ice, dice, melt_load = state_grids(em, rho_w_case,
                                                rho_i_case, shore)
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
    if args.earth:
        report["earth"] = {"melt_region": args.melt_region,
                           "melt_fraction": args.melt_fraction,
                           "ice_model": "ICE7G", "date_ka": 0.0}
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
    report["degrees"] = degree_rows(adgia, ref, em, args.lmax)
    (case / f"report_{tag}.json").write_text(
        json.dumps(report, indent=2) + "\n")
    figure(case / f"fingerprint_{tag}.png", em, adgia, ref, frac,
           report)

    print(f"\n{args.model}, h = {args.h:g}, order {args.order}, "
          f"pyslfp lmax {args.lmax}" +
          (f", ICE-7G melting {args.melt_region}" if args.earth else ""))
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

    if not args.nonlinear:
        return

    # Rung 3: shoreline migration against pyslfp's nonlinear solver, from
    # the same initial state and ice-thickness change. The primary row is
    # the migrating-fingerprint comparison; the sharp probe is the
    # migration effect delta = SL(migrating) - SL(frozen) on each side,
    # where the common linear part cancels. pyslfp's ocean updates are
    # sharp where ours are smoothed over `shore`, so the delta difference
    # is shoreline-dominated at toy resolution.
    sle = pyslfp.SeaLevelEquation(em)
    dice_g = pyshtools.SHGrid.from_array(dice * to_ps_len, grid=em.grid)
    _, sl_nl, _, _, omega_nl = sle.solve_nonlinear_equation(
        state, ice_thickness_change=dice_g, rotational_feedbacks=args.rot)
    ref_nl = sl_nl.data * ps.length_scale

    rows_nl = np.genfromtxt(slcsv_nl, delimiter=",", names=True)
    xyz_nl = np.column_stack([rows_nl["x"], rows_nl["y"], rows_nl["z"]])
    adgia_nl = rbf_to_grid(xyz_nl, np.asarray(rows_nl["value"]), em) * L

    diff_nl = adgia_nl - ref_nl
    scale_nl = wrms(ref_nl, frac)
    delta_a, delta_p = adgia_nl - adgia, ref_nl - ref
    delta_scale = wrms(delta_p, frac)
    report_nl = {
        "model": args.model, "h": args.h, "order": args.order,
        "lmax": args.lmax, "rot": args.rot,
        "ref_ocean_rms_m": scale_nl,
        "diff_ocean_rms_m": wrms(diff_nl, frac),
        "diff_ocean_rel": wrms(diff_nl, frac) / scale_nl,
        "diff_max_m": float(np.max(np.abs(diff_nl))),
        "delta_ocean_rms_m": {"adgia": wrms(delta_a, frac),
                              "pyslfp": delta_scale},
        "delta_diff_ocean_rel": wrms(delta_a - delta_p, frac) / delta_scale,
        "adgia": json.loads(sljson_nl.read_text()),
        "degrees": degree_rows(adgia_nl, ref_nl, em, args.lmax),
    }
    if args.rot:
        report_nl["omega_over_Omega"] = {
            "adgia": [o / Omega_case
                      for o in report_nl["adgia"].get("omega", [])],
            "pyslfp": (np.asarray(omega_nl) /
                       ps.rotation_frequency).tolist()}
    (case / f"report_{tag}_nl.json").write_text(
        json.dumps(report_nl, indent=2) + "\n")
    figure(case / f"fingerprint_{tag}_nl.png", em, adgia_nl, ref_nl, frac,
           report_nl)
    # The migration effect, drawn with the same four panels (per-degree
    # rows and the headline number are the delta's own).
    report_delta = {
        "model": args.model, "h": args.h, "order": args.order,
        "lmax": args.lmax,
        "diff_ocean_rel": report_nl["delta_diff_ocean_rel"],
        "degrees": degree_rows(delta_a, delta_p, em, args.lmax),
    }
    figure(case / f"migration_effect_{tag}.png", em, delta_a, delta_p,
           frac, report_delta)

    mg = report_nl["adgia"]
    print(f"\nnonlinear (rung 3): AdGIA {mg.get('mig_passes', '?')} extra "
          f"passes (relative SL1 increment {mg.get('mig_last_change', 0):.2e}, "
          f"{mg.get('mig_outer_iterations', '?')} outer iterations)")
    print(f"  migrating fingerprints: ocean-RMS rel diff "
          f"{report_nl['diff_ocean_rel']:.3f}")
    print(f"  migration effect:  adgia {report_nl['delta_ocean_rms_m']['adgia']:.4g} m, "
          f"pyslfp {delta_scale:.4g} m  "
          f"(rel diff {report_nl['delta_diff_ocean_rel']:.3f}; "
          f"shoreline-convention-dominated at toy resolution)")
    if args.rot:
        mm = report_nl["omega_over_Omega"]
        print("  omega/Omega  adgia  " +
              " ".join(f"{v: .4g}" for v in mm["adgia"]))
        print("               pyslfp " +
              " ".join(f"{v: .4g}" for v in mm["pyslfp"]))
    print(f"  report: {case / f'report_{tag}_nl.json'}")
    print(f"  figures: {case / f'fingerprint_{tag}_nl.png'}, "
          f"{case / f'migration_effect_{tag}.png'}")


if __name__ == "__main__":
    main()
