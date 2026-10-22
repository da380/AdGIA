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
- pyslfp ↔ sl3dpp on the fluid-core variants (21 Oct 2026): inner_core
  worst 8.6e-12 (k'_0; h' 3.8e-12), stratified_core 8.2e-12 (h'_1; l',
  k' 6e-12), tidal at most 2.1e-12 on both; sl3dpp's reciprocity
  residual 4–8e-16 on all three variants.
- aw_core: 8.2e-8, entirely in h'_0; every degree l >= 1 below 1e-11,
  every other entry at most 3e-11 (k'_0). This is the deck, not the
  codes: aw_core's core has the closed-form Adams–Williamson vp, the one
  non-polynomial field of the ladder, so the default eight-knot deck
  reproduces the model only to spline accuracy (`decks.py` reports a
  round trip of 9.5e-7). Sixteen knots per layer (`decks.py aw_core
  --knots 16`, round trip 5.4e-8) bring h'_0 to 2.0e-9 with l >= 1
  unchanged: degree 0 is the channel sensitive to the core's
  compressibility profile, which is what the deck samples. Above degree
  0 every code makes the static-fluid reduction (the spectral codes'
  potential-only fluid regions as much as AdGIA's Dahlen elimination),
  with a stratification term built from rho and drho/dr only, so kappa,
  the one field the deck approximates, enters only at degree 0, where
  the full compressible equations apply; the core's quadratic density
  the deck reproduces exactly. (AdGIA meshes the model directly and
  never sees the deck.) Verdict: pyslfp (model object) and sl3dpp
  (deck) agree to the deck's own accuracy; write aw_core's deck with more
  knots when degree 0 is at stake, and quote `decks.py`'s round-trip
  error with the comparison.
- AdGIA on the fluid-core variants (21 Oct 2026; h 0.3, order 2,
  degrees 0–6, 8 ranks; cases by the love_numbers family's `run` script,
  `make_case <model> --h 0.3 --buffer 0.2 --angular 0.3 --thin 4 --lmax
  10`, solved by `love_benchmark -o 2 -rt 1e-10 -s 1 -method
  dahlen|maxwell -lmin 0 -lmax 6 -deg 16`): the fluid_core pattern
  repeats on every variant. `from_results` leaves degree 0 out (NaN) of
  the exchange file of a Dahlen run on a model with a fluid layer, so the
  Dahlen worst is over l >= 1; the Dahlen degree-0 values quoted below
  are from the results JSON. The worst entry
  is h'_6 in every run; seconds on 8 ranks:

  | model | route | worst | hp2 | hp1, lp1 | hp0 | s |
  |---|---|---|---|---|---|---|
  | aw_core | dahlen | 3.6e-2 | 7.3e-3 | 1.8e-3, 1.7e-3 | – | 6.8 |
  | | maxwell | 3.3e-2 | 1.4e-2 | 6.1e-4, 4.0e-5 | 4.4e-3 | 118.6 |
  | inner_core | dahlen | 3.6e-2 | 6.9e-3 | 2.8e-3, 5.6e-3 | – | 20.0 |
  | | maxwell | 3.2e-2 | 2.0e-2 | 1.5e-2, 4.1e-3 | 4.5e-3 | 274.2 |
  | stratified_core | dahlen | 2.9e-2 | 7.4e-4 | 7.4e-4, 5.9e-4 | – | 7.0 |
  | | maxwell | 3.3e-2 | 4.8e-3 | 8.7e-3, 1.1e-3 | 5.0e-3 | 172.5 |

  - Dahlen degree 0, wrong by design: h'_0 = −0.5223, −1.3657, −0.7800
    against pyslfp's −0.12972, −0.13236, −0.15229 (4.0, 10.3, 5.1
    times), k'_0 = −2.20, −5.02, −3.03 against 0. The Maxwell core gets
    h'_0 to 4.4–5.0e-3 and k'_0 ≈ −7e-6 on all three, at 14–25 times
    Dahlen's wall time.
  - aw_core and inner_core (same mantle, fluid_core's): the Dahlen
    errors at degrees 3–6 coincide to 3e-4 or better (5e-5 at 5–6),
    differing only at degrees 1–2 (up to 4e-3, 1e-3) — at h 0.3 the
    high-degree error is the mantle's discretisation at the resolution
    edge, not the core's.
  - stratified_core: Dahlen unusually close at degree 2 (h'_2 7.4e-4,
    k'_2 7.3e-6).
  - Maxwell degree 1: h'_1 6.1e-4 on the neutral aw_core against 1.5e-2
    (inner_core), 8.7e-3 (stratified_core) and 1.6e-2 (fluid_core) — the
    increment floor on non-neutral cores; on aw_core both routes are
    within 2e-3 of pyslfp at degree 1. k'_1 = −1 to rounding in every
    run.
