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
outward, `inf` elastic), switches on the unit loads of degrees
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
default the reference's 13) are hit exactly. `-tide` evolves the tidal
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
  <taus> <out.txt> [steps] [dtmax]`, taus in seconds per layer, `inf`
  elastic.
- **AdGIA and the Laplace reference** — `./from_results <json> --out
  <file>` converts either a `viscoelastic_love` results file or a
  `laplace_reference` output (invalid times become NaN); `--channel
  tide` writes the tidal triple.
- Comparison by the shared exchange comparer,
  `<build>/benchmarks/elastic/nongrav/compare` (this directory's own
  `compare` is the JSON comparer of the Laplace leg); it keys on any
  first column, `t` here.

Measured on `homogeneous`, tau 1e11 s, lmax 4: pyslfp and sl3dpp agree
to 7e-12 at every time and degree (identical discrete problem in space
and time); against the Laplace reference both sit at 1.2e-4 worst
(hp2) under the default stepping, falling to 2.8e-5 with 40 steps per
interval and dt <= 0.025 tau_min — the remainder being the reference's
own Gaver-Stehfest accuracy, not the steppers'.
