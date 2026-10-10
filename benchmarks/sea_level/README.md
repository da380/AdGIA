# The sea-level fingerprint benchmark

AdGIA's monolithic sea-level solve (`SetWaterLoad`: the sea-level
equation folded into the elastic operator, frozen shorelines, no
rotation) against pyslfp's pseudo-spectral solution of the same problem
— rung 1 of `doc/planning/sea_level_plan.md`. `--rot` adds the
rotational border on both sides (rung 2), `--nonlinear` adds shoreline
migration against pyslfp's nonlinear solver (rung 3), and `--timings`
times the solve across the feature axis instead of comparing.

One spherically layered model (homogeneous by default), one smooth
analytical state (polar continent, ice cap, melt), defined once in
`run.py` and handed to both sides; pyslfp computes its own Love numbers
from the same planetmodel model (`EarthModel.from_planet_model`), so
Love-number agreement is assumed here and tested by the Love-number
family instead. `run.py` builds the case, runs `sea_level_benchmark`
under MPI, runs pyslfp, converts both fingerprints to metres, grids the
FE nodes onto pyslfp's grid and writes `report_o<p>.json`: ocean-RMS
difference and per-degree amplitudes.

Caveats recorded with the machinery: low-resolution runs need not agree
closely (the resolved comparison belongs to the server campaign), and
the degree-1 reference-frame convention between the two sides is not
yet reconciled — the report prints that row with a flag.

The `--nonlinear` leg solves the frozen/linear pair too and reports the
migration effect delta = SL(migrating) − SL(frozen) on each side — the
probe in which the common linear part cancels. Two resolution caveats
make its numbers server-campaign material: the
shoreline strip a realistic melt moves is tens of km, far below both a
toy FE mesh and a sharp ocean function on an lmax ≈ 16 grid (pyslfp's
nonlinear updates are sharp where ours are smoothed over `shore`), so
at toy resolution only the machinery is validated — the migrating
fingerprints agree at the frozen pair's grade while the deltas are
resolution-starved. The resolved comparison also wants the
coastline-refined meshes planned in
`doc/planning/planetmodel_coastline_sizing_plan.md`. The AdGIA
migration runs tight here (`-mig-inexact 0`): this solver stack
amplifies a loose residual into an O(tolerance) absolute error in
Phi_g, and the library's contraction guard
(`ShorelineMigration::Options::guard`) would rescue the loop at more
passes than solving tight from the start.

`--timings` runs the same melt load as a plain elastic solve
(`-no-water`), with the water feedback (elimination and monolithic),
with rotation (both again), and with migration — best/mean of
`--repeat` runs and ratios against the elastic control, written to
`timings_o<p>.json`. Toy-size measurement (h 0.4, o2, np 4):
elimination water ≈ 1.6×, rotation ≈ 3.8×, migration ≈ 5.3×; the
monolithic border (`-feedback monolithic`, one MINRES on the extended
system instead of one solve per border column) brings the composed
water–rotation case to ≈ 2.3× while the single-border water case stays
with the elimination. Each elimination border column is one extra
unbordered solve, so its scaling is structural rather than
resolution-bound; the monolithic cost is flat in the border count.

`--coast` (planetmodel >= 1.2.4) builds the case on a
coastline-refined mesh: a `near_curve` ring refinement at the state's
shoreline colatitude (computed from the same flotation the state
defines), in its own `h<h>_coast` directory. The global comparison is
unchanged at toy resolution (spectral/FE floor), but the shoreline
strip is where it bites: measured at h 0.4/o2, the migration loop
converges in 1 pass instead of 2 and the AdGIA-side migration-effect
delta drops 4x — most of it was shoreline-quadrature noise.

    cd <build>/benchmarks/sea_level
    ./run --h 0.4 --order 2 --np 4 --lmax 32
    ./run --nonlinear
    ./run --timings --repeat 5
    ./run --coast --nonlinear
