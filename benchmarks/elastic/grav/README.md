# E2: the gravitating elastic tier

The second rung of the cross-code hierarchy (`../../README.md`): the
self-gravitating elastic response of spherical models — solid and
fluid–solid — compared across the three codes as conventional Love
numbers: the load numbers h' l' k' (Farrell's, referred to the load's
direct potential, geodetic sign) and the geodetic tidal numbers h l k
(NaN below degree 2). Degree 1 is in the centre-of-mass frame in every
code (k'_1 = -1 identically); degree 0 of AdGIA's Dahlen fluid is
excluded by design and NaN — the Maxwell-core route covers it
(`doc/static_fluid_core.tex`).

**The reference here is pyslfp**: the most thoroughly tested of the
three (checked in its own suite against closed forms and an independent
Love-number code). There is no closed form for the compressible
gravitating ladder, so the tier is code-against-code with pyslfp as the
anchor; the incompressible homogeneous closed forms live in pyslfp's
own tests.

## Producing the exchange files

Columns `l hp lp kp hT lT kT`, dimensionless:

- **pyslfp** — from its checkout: `cd ~/dev/pyslfp && poetry run python
  .../benchmarks/elastic/grav/pyslfp_reference.py --out <runs dir>`.
- **sl3dpp** — its `love_numbers` driver
  (`-DSL3DPP_BUILD_BENCHMARKS=ON`): `love_numbers <deck> <lmax>
  <out.txt>`, on the decks `common/decks.py` writes.
- **AdGIA** — the existing `love_benchmark` (the love_numbers family,
  any method), converted: `./from_results results.json --out
  runs/e2_<model>_adgia.txt`. The JSON's numbers are already the
  conventional ones; the converter copies, never recomputes.

## Comparing

`./compare` (shared with E1). Measured at lmax 8:

- pyslfp ↔ sl3dpp (independent meshes, deck vs model object): worst
  5e-11 over homogeneous, two_solid and fluid_core — both converged.
- AdGIA (dahlen, h 0.3, order 2, lmax 6) ↔ pyslfp: worst 2e-2 at the
  degree-resolution edge, ~1e-3 in the resolved channels; degree 1
  agrees at 1.4e-3 with k'_1 = -1 to rounding in both, the first
  cross-code check of the centre-of-mass frame conventions.
- fluid_core, same resolution: Dahlen agrees with pyslfp at the same
  FEM level for l >= 1 while its degree 0 is wrong by design (h'_0 off
  by a factor 10, k'_0 by 40 — the known arbitrary function, exhibited
  rather than benchmarked); the Maxwell-core method agrees at every
  degree *including* 0 (h'_0 to 4.5e-3), at several times Dahlen's
  cost per solve.
