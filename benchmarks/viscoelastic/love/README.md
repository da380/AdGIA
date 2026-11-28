# Viscoelastic: Love-number histories

Maxwell load Love-number histories of gravitating spherical models: the
finite-element driver `viscoelastic_love` against the same histories
stepped in time by the spectral codes, pyslfp and sl3dpp. Launchers in
`<build>/benchmarks/viscoelastic/love/` (`make_case`, `from_results`).

The time stepping on its own, without gravity or fluids, is tested
against exact references by the box and sphere sub-families (`../box`,
`../sphere`), and the static solve by the elastic Love numbers
(`../../love_numbers`, `../../elastic/grav`). This family is the check
that the two come together: the relaxation of a self-gravitating body
with a fluid core, the fluid's closure carried through time.

## The finite-element driver

`viscoelastic_love` loads a case exactly as `love_benchmark` does
(`-c case.json`, `-o`, `-deg`, `-lmax`, `-method dahlen|gauged`), gives
every solid layer the Maxwell time of `-tau` (comma separated, centre
outward, one value per *solid* layer or one for all, `inf` elastic; a
fluid layer takes none, unlike the spectral drivers below), switches on
the unit loads of degrees `-lmin..-lmax` (default `-lmin 2`) together at
t = 0, and evolves with `ViscoelasticOperator`:

| `-scheme` | |
|---|---|
| `sdirk23` | default: L-stable, 2 solves/step |
| `exptrap` | exponential trapezoid, 1 solve/step |
| `be`, `etd1` | first order (for convergence studies) |
| `adaptive` | adaptive exponential trapezoid, `-rtol`, `-atol` |

Fixed-step schemes take, within each interval between output times, the
same number of equal steps: at least `-steps` (4), more to keep
`dt <= -dt-max` (0.25 tau_min). Output times (`-times`, comma separated;
default 13 geometric times `0.011 * 2^(5k/6) tau_min`, k = 0..12, from
0.011 to 11.3 tau_min, rounded to six decimals so that every code's
exchange file has the same time column) are hit exactly. `-tide` evolves
the tidal forcing as well.

On a model with a fluid layer `-level-border <n>` closes the Dahlen
fluid by the level border of `doc/theory/dahlen_closure.tex` (the default
static route), so that the degree-0 history is benchmarked; without it
the naked reduction's degree 0 is wrong by design. The results file
records `"level_border"`.

```
mpiexec -np 8 <build>/benchmarks/bin/viscoelastic_love \
    -c case.json -o 2 -lmin 0 -lmax 4 -tau 1 -level-border 16 \
    -out results.json
```

The case can be any Love-number case (`./make_case <model> --h 0.3
--out <dir>` makes one). The results are JSON: the case header, the
layers and their taus, the scheme, `degree`, `times`, `elastic` (the
t = 0+ solve) and `histories` (per output time the load numbers per
degree; the tidal ones with `-tide`), with the cumulative stepping
solves and seconds per output time and the totals under `cost`.

## The cross-code comparison (V1)

The compared file is one exchange table per model and code: `t` (in
units of the shortest Maxwell time) and the conventional load numbers
`hp<l> lp<l> kp<l>` per degree. The stepping rule is shared —
exponential trapezoid, equal steps between output times, at least 4 and
none longer than 0.25 tau_min — and the two spectral codes also share
the uniform lmax mesh rule, so their mutual difference is arithmetic
alone, while differences from `viscoelastic_love` isolate the spatial
discretisation. Because the codes take the same steps with the same
weights, their agreement says nothing about the error of the time
stepping: the spectral reference is to be shown converged in its own
step (`--steps`, `--dtmax`) before a finite-element run is judged
against it.

- **pyslfp** — `pyslfp_histories.py`, run from the pyslfp checkout (see
  its docstring).
- **sl3dpp** — its `love_histories` driver
  (`-DSL3DPP_BUILD_BENCHMARKS=ON`): `love_histories <deck> <lmax>
  <taus> <out.txt> [steps] [dtmax]`, taus in seconds per layer (the
  fluid included), `inf` elastic.
- **AdGIA** — `./from_results <json> --out <file>` converts a
  `viscoelastic_love` results file; `--channel tide` writes the tidal
  triple.
- Comparison by the shared exchange comparer,
  `<build>/benchmarks/elastic/nongrav/compare`; it keys on any first
  column, `t` here. On a `t`-keyed file its degree scale is the degree's
  largest reference value over the whole history (every time and every
  channel of the degree), so each degree's history is judged on one
  scale.

From `<build>/benchmarks/viscoelastic/love/` (`mpiexec` is the build's
MPICH), for a model `<model>` at order 2:

```
<adgia python> <repo>/benchmarks/common/decks.py <model> --out runs   # sl3dpp's deck
(cd <pyslfp checkout> && poetry run python \
    <repo>/benchmarks/viscoelastic/love/pyslfp_histories.py <model> \
    --tau <taus> --out <build>/benchmarks/viscoelastic/love/runs)
<sl3dpp build>/benchmarks/love_histories runs/<model>.elastic.deck 4 \
    <taus> runs/v1_<model>_sl3dpp.txt
./make_case <model> --h 0.3 --out cases/<model>_h03
mpiexec -np 8 ../../bin/viscoelastic_love -c cases/<model>_h03/case.json \
    -o 2 -lmin 0 -lmax 4 -tau <taus> -scheme exptrap -level-border 16 \
    -out runs/v1_<model>_adgia_h03_o2.json
./from_results runs/v1_<model>_adgia_h03_o2.json \
    --out runs/v1_<model>_adgia_h03_o2.txt
../../elastic/nongrav/compare runs/v1_<model>_pyslfp.txt \
    runs/v1_<model>_sl3dpp.txt runs/v1_<model>_adgia_h03_o2.txt
```

The Maxwell times follow two conventions. `viscoelastic_love` takes one
per *solid* layer, in the case's units: `-tau 1,inf` for a Maxwell
interior under an elastic lithosphere, `-tau 1` for a Maxwell mantle
over a fluid core. The spectral drivers take one per layer, the fluid
included, in seconds: `1e11,inf` and `inf,1e11` for the same two
(`--tau` of `pyslfp_histories.py`, the third argument of
`love_histories`).
