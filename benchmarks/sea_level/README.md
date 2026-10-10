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
make its numbers server-campaign material (agreed 10 Oct 2026): the
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
Phi_g, and the library's contraction guard (`ShorelineMigration::
Options::guard`, found by this benchmark) would rescue the loop at more
passes than solving tight from the start.

`--timings` runs the same melt load as a plain elastic solve
(`-no-water`), with the water feedback, with rotation, and with
migration — best/mean of `--repeat` runs and ratios against the elastic
control, written to `timings_o<p>.json`. Toy-size measurement (h 0.4,
o2, np 4): water ≈ 1.4–1.7×, rotation ≈ 3.7×, migration ≈ 5× the
elastic solve; each border column is one extra unbordered solve, so the
scaling is structural rather than resolution-bound.

    cd <build>/benchmarks/sea_level
    ./run --h 0.4 --order 2 --np 4 --lmax 32
    ./run --nonlinear
    ./run --timings --repeat 5
