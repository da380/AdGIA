# Viscoelastic: Love-number histories

Maxwell load Love-number histories of gravitating spherical models: the
finite-element driver `viscoelastic_love` against pyslfp through the
correspondence principle (`laplace_reference.py`), compared by
`compare.py`. Launchers in `<build>/benchmarks/viscoelastic/love/`
(`laplace_reference`, `compare`, `make_case`). The method of the
reference (correspondence principle, Gaver–Stehfest inversion, its
accuracy and the instability horizon), the comparison metrics and the
measured results are in `doc/benchmarks.tex` ("The Laplace-domain
reference" and "The finite-element histories").

`references/` holds committed reference histories (`fluid_core` with
tau = 1, `homogeneous` with tau = 1, `homogeneous_lithosphere` with
tau = 1, inf).

## The Laplace reference

For a layered Maxwell body under a Heaviside load the Laplace transform
of the response is the elastic response with the shear modulus
`mu_j(s) = mu_j s tau_j / (1 + s tau_j)` in every Maxwell layer, divided
by `s`; Gaver–Stehfest inversion (order `--stehfest`, default 12) needs
the transform at real positive `s` only, so every evaluation is a pyslfp
elastic solve of a modified model. Supported: the uniform-layer models
(`homogeneous`, `homogeneous_lithosphere`, `two_solid`, `fluid_core`,
`inner_core`).

Every solid layer has its own Maxwell time, centre outward; `inf`
keeps a layer elastic:

```
./laplace_reference homogeneous --tau 1                    # all Maxwell
./laplace_reference homogeneous_lithosphere --tau 1 inf    # elastic lid
./laplace_reference fluid_core --tau 1 --lmax 4 --out ref.json
```

`homogeneous_lithosphere` is the homogeneous body cut at 5700 km:
elastically identical, but its ~670 km outer shell is a layer of its own
that can stay elastic.

Options: `--times` (default 13 geometric times `0.011 * 2^(5k/6)
tau_min`, k = 0..12, 0.011 to 11.3 tau), `--lmax` (default 4),
`--stehfest`, and the pole scan that sets the instability horizon,
`--scan S_MIN S_MAX` and `--scan-density` (a log grid, in units of
1/tau_min; density 0 turns it off) followed by `--fine-scan S_MIN S_MAX
DENSITY` (a dense grid where the horizon is decided; density 0: none).
Times beyond the horizon are marked invalid. `--out` names the results
file (default `laplace_<model>_tau<taus>.json`).

### Schema

```
{"model", "layers": [{"name", "fluid", "tau"}],   # tau: number, "inf", null (fluid)
 "lmax", "stehfest", "degree": [l...], "times": [t...],
 "elastic": {"h_load": [per degree], "l_load", "k_load", "h_tide", "l_tide", "k_tide"},
 "relaxed": {same, or nulls}, "relaxed_note",
 "poles": [{"degree", "s", "efold_time"}], "horizon",
 "histories": [{"time", "valid", "h_load": [per degree], ...}],
 "sanity": {"t0", "t0_error", "tolerance", "monotone": [...]}}
```

`viscoelastic_love` writes the same `degree`, `times`, `elastic` (its
t = 0+ solve) and `histories` entries (load numbers; the tidal ones with
`-tide`), plus the case header, the layers and taus, the scheme, and
per output time the cumulative stepping solves and seconds, and totals
under `cost`.

The reference stores, and prints, its checks: the inversion at
`1e-6 tau_min` against the elastic numbers (the inversion's accuracy),
the monotonicity of h', k' (and h, k) over the valid times, and, where a
relaxed (static-fluid) value exists, the fraction of the way to it,
which is information, not a check: a non-neutral compressible Maxwell
layer does not relax to pyslfp's static fluid. With no elastic layer the
relaxed limit is unavailable (the body relaxes to a fluid ball).

## The finite-element driver

`viscoelastic_love` loads a case exactly as `love_benchmark` does
(`-c case.json`, `-o`, `-deg`, `-lmax`, `-method dahlen|gauged`), gives
every solid layer the Maxwell time of `-tau` (comma separated, centre
outward, one value per *solid* layer or one for all, `inf` elastic; a
fluid layer takes none, unlike the spectral V1 drivers below), switches
on the unit loads of degrees
`-lmin..-lmax` (default `-lmin 2`: with a Dahlen fluid degree 0 differs
from the reference by design) together at t = 0, and evolves with
`ViscoelasticOperator`:

| `-scheme` | |
|---|---|
| `sdirk23` | default: L-stable, 2 solves/step |
| `exptrap` | exponential trapezoid, 1 solve/step |
| `be`, `etd1` | first order (for convergence studies) |
| `adaptive` | adaptive exponential trapezoid, `-rtol`, `-atol` |

Fixed-step schemes take, within each interval between output times, the
same number of equal steps: at least `-steps` (4), more to keep
`dt <= -dt-max` (0.25 tau_min). Output times (`-times`, comma separated;
default the reference's 13, rounded to six decimals in units of tau_min
exactly as the reference writes them, so that every leg's exchange file
has the same time column) are hit exactly. `-tide` evolves the tidal
forcing as well.

The `-method gauged` histories are clean only for a neutrally stratified
fluid core: at N² ≠ 0 (`fluid_core`'s uniform core) the gauge penalty is
defined only to the stratification band, and the comparison with the
Laplace reference there carries that band (`doc/static_fluid_core.tex`).

```
mpiexec -np 8 <build>/benchmarks/bin/viscoelastic_love \
    -c case.json -o 2 -lmax 4 -tau 1 -out results.json
./compare results.json --reference laplace_fluid_core_tauf_1.json
```

The case can be any Love-number case (`./make_case <model> --h 0.3
--out <dir>` makes one). `compare.py` prints and writes, per degree and
number, the history error (the worst over the valid output times and
t = 0+, relative to the largest reference value), the elastic error at
t = 0+ and the error of the relaxation f(t) - f(0), as
`<stem>_errors.md`/`.json`, with the figures `<stem>_history.png`,
`<stem>_error.png`, and `<stem>_series.png` (linear time axis) when a
run has 20 or more output times; `--out` names the directory,
`--no-plots` skips the figures. A step-convergence check is two runs at
different `-steps`/`-dt-max` compared through the same reference.

The opt-in campaign stage runs the local version (`fluid_core`, h = 0.3,
order 2, degrees 2..4, SDIRK23) and plots it:

```
./campaign --profile local --stages viscoelastic plot
```

## The cross-code tier (V1)

This family is also the home of the V1 tier of the cross-code hierarchy
(`../../README.md`): the same Heaviside-load histories computed in the
time domain by all three codes and set against each other and the
Laplace reference through the exchange format. The compared file is one
table per model: `t` (in units of the shortest Maxwell time, the 13
geometric times above, rounded to six decimals as the reference writes
them) and the conventional load numbers `hp<l> lp<l> kp<l>` per degree.
The stepping rule is shared — exponential trapezoid, equal steps between
output times, at least 4 and none longer than 0.25 tau_min — and the two
spectral codes also share the uniform lmax mesh rule, so their mutual
difference isolates nothing but arithmetic, while differences from this
library's `viscoelastic_love` isolate the spatial discretisation.

- **pyslfp** — `pyslfp_histories.py`, run from the pyslfp checkout (see
  its docstring).
- **sl3dpp** — its `love_histories` driver
  (`-DSL3DPP_BUILD_BENCHMARKS=ON`): `love_histories <deck> <lmax>
  <taus> <out.txt> [steps] [dtmax]`, taus in seconds per layer (the
  fluid included), `inf` elastic.
- **AdGIA and the Laplace reference** — `./from_results <json> --out
  <file>` converts either a `viscoelastic_love` results file or a
  `laplace_reference` output (invalid times become NaN); `--channel
  tide` writes the tidal triple.
- Comparison by the shared exchange comparer,
  `<build>/benchmarks/elastic/nongrav/compare` (this directory's own
  `compare` is the JSON comparer of the Laplace leg); it keys on any
  first column, `t` here. On a `t`-keyed file its degree scale is the
  degree's largest reference value over the whole history (every time and
  every channel of the degree), so each degree's history is judged on one
  scale.

### Measured (21 October 2026)

`homogeneous` (tau 1e11 s), `homogeneous_lithosphere` (interior 1e11 s,
lithosphere elastic) and `fluid_core` (mantle 1e11 s), lmax 4, degrees
0–4. AdGIA: Dahlen method, degree 1 in the centre-of-mass frame, h = 0.3
(1903 / 3110 / 3733 elements in the body), orders 2 and 3, 8 MPI ranks
on a laptop; degree 0 left out on `fluid_core` (a Dahlen fluid differs
there by design). Every figure is the worst over the 13 times on the
exchange comparer's degree scale (above): degree 0 (|hp0| ≈ 0.14–0.17)
on its own scale, degrees 1–4 on theirs (about 1.3 to 3.6). The
write-up is tier V1 of `doc/benchmark_programme.tex`.

- **sl3dpp against pyslfp**: 1.2e-11 (`homogeneous`), 1.3e-11
  (`homogeneous_lithosphere`), 1.1e-11 (`fluid_core`), each in kp0; h'
  at most 1.1e-11, l' and k' at degrees 2–4 below 5e-12 — the identical
  discrete problem in space and time.
- **Both spectral codes against the Laplace reference**: worst 1.5e-4
  (hp3) on `homogeneous`, 1.6e-4 (hp4) on `homogeneous_lithosphere`,
  1.5e-4 (hp4) on `fluid_core`; hp2 1.0–1.5e-4, l' and k' at most 9e-5,
  degrees 0–1 at most 8e-5 (lp1 on `fluid_core`), kp1 2.5e-11. The
  family's earlier stepping refinement (40 steps per interval, dt <=
  0.025 tau_min, on `homogeneous`) brought it down to the reference's
  own Gaver–Stehfest floor, about 3e-5 (measured then on a different
  scale): the stepping error the three legs share is about 1e-4 at the
  default rule, and it largely cancels between AdGIA and pyslfp, which
  take the same steps with the same weights.
- **AdGIA against pyslfp** (seconds on 8 ranks; each run is 79 stepping
  solves and one observation solve):

  | model | order | worst | degree 0: hp0 | degree 1: hp1, lp1 | s |
  |---|---|---|---|---|---|
  | `homogeneous` | 2 | 2.3e-2 (hp2) | 5.8e-4 | 3.9e-4, 1.1e-3 | 11 |
  | | 3 | 5.2e-4 (hp4) | 2.4e-4 | 2.8e-5, 2.9e-4 | 38 |
  | `homogeneous_lithosphere` | 2 | 1.0e-2 (hp4) | 3.8e-3 | 1.3e-3, 4.4e-3 | 19 |
  | | 3 | 7.9e-4 (hp2) | 6.8e-5 | 1.8e-5, 1.5e-4 | 133 |
  | `fluid_core` | 2 | 1.5e-2 (lp1) | – | 1.6e-3, 1.5e-2 | 29 |
  | | 3 | 8.2e-4 (lp2) | – | 1.7e-4, 5.8e-4 | 178 |

  After the worst: on `homogeneous`, kp2 1.4e-2, lp2 1.3e-2, lp4
  1.3e-2, lp3 9.3e-3, hp4 8.7e-3 at order 2 (kp0 2.8e-5), and lp4
  4.0e-4, lp2 3.0e-4, hp2 5.1e-5 at order 3; on
  `homogeneous_lithosphere`, hp3 4.5e-3, kp4 4.4e-3 at order 2 and kp2
  4.7e-4, hp3 4.6e-4 at order 3; on `fluid_core`, hp2 1.1e-2, hp4
  8.9e-3, lp4 6.3e-3 at order 2 and lp1 5.8e-4, hp3 4.2e-4 at order 3.
  Order 3 improves on order 2 by a factor 13–44 (`homogeneous` 44,
  `homogeneous_lithosphere` 13, `fluid_core` 19), at 3.5–7 times the
  cost. kp1 = −1 to rounding (1e-15) at
  every time in all three codes: the centre-of-mass frame conventions
  agree in the viscoelastic setting too (E2 had confirmed them
  elastically).

Along the history (this directory's `compare`, which judges each channel
on its own scale, against the Laplace reference): on `homogeneous` at
order 2, h'_2 is 3.9e-3 from the reference at t = 0+ but 2.3e-2 worst
over the history, and its relaxation f(t) − f(0) is off by 3.5e-2 of
the drift; at order 3 the three coincide (1.4e-4, 1.6e-4, 1.6e-4). So
at order 2 the error grows along the history: h'_2 goes from −0.65 to
−2.00 by 11.3 tau_min (−2.04 at order 2), and the late, largely viscous
response is what h = 0.3 at order 2 under-resolves; at order 3 the
relative error is the elastic one throughout. The exchange comparer
shows it on the degree scale (for hp2 the same scale as `compare`'s):
on `homogeneous` at order 2 the hp2 entry rises from about 1e-3 at the
first time to 2.3e-2 at the last, the worst entry there, and on
`fluid_core` the worst order-2 entry, lp1, rises from 1.7e-3 to 1.5e-2
as l'_1 goes from −0.89 to −1.34, falling to 5.8e-4 at order 3.

### Reproducing

From `<build>/benchmarks/viscoelastic/love/` (`mpiexec` is the build's
MPICH), for `homogeneous` at order 2:

```
<adgia python> <repo>/benchmarks/common/decks.py homogeneous --out runs   # sl3dpp's deck
(cd ~/dev/pyslfp && poetry run python \
    <repo>/benchmarks/viscoelastic/love/pyslfp_histories.py homogeneous \
    --tau 1e11 --out <build>/benchmarks/viscoelastic/love/runs)
<sl3dpp build>/benchmarks/love_histories runs/homogeneous.elastic.deck 4 \
    1e11 runs/v1_homogeneous_sl3dpp.txt
./from_results --out runs/v1_homogeneous_laplace.txt \
    <repo>/benchmarks/viscoelastic/love/references/laplace_homogeneous_tau1.json
./make_case homogeneous --h 0.3 --out cases/homogeneous_h03
mpiexec -np 8 ../../bin/viscoelastic_love -c cases/homogeneous_h03/case.json \
    -o 2 -lmin 0 -lmax 4 -tau 1 -scheme exptrap \
    -out runs/v1_homogeneous_adgia_h03_o2.json
./from_results runs/v1_homogeneous_adgia_h03_o2.json \
    --out runs/v1_homogeneous_adgia_h03_o2.txt
../../elastic/nongrav/compare runs/v1_homogeneous_pyslfp.txt \
    runs/v1_homogeneous_sl3dpp.txt runs/v1_homogeneous_laplace.txt \
    runs/v1_homogeneous_adgia_h03_o2.txt
```

The Maxwell times follow two conventions. `viscoelastic_love`, like
`laplace_reference`, takes one per *solid* layer: `-tau 1,inf` for
`homogeneous_lithosphere`, `-tau 1` for `fluid_core` (there with
`-lmin 1`). The spectral drivers take one per layer, the fluid
included: `1e11,inf` for `homogeneous_lithosphere` and `inf,1e11` for
`fluid_core` (`--tau` of `pyslfp_histories.py`, the third argument of
`love_histories`).
