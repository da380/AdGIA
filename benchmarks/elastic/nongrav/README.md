# E1: the non-gravitating elastic tier

The bottom rung of the cross-code hierarchy (`../../README.md`, "The
cross-code hierarchy"): the static elastic response of solid,
spherically layered models to unit surface tractions, with gravity off
everywhere. On the piecewise-homogeneous ladder models the exact
solution is a per-degree power-law propagator (`common/analytic.py`),
so every code is judged against a closed form, not against another
code. Degree 1 is excluded (a surface traction there has a net force
against the rigid translations) and marked NaN throughout; degree 0 is
uniform compression, U(a) = -a / (3 kappa) on the homogeneous model.

The compared quantities are the traction numbers: U(a) and V(a) per
unit normal and per unit tangential traction, SI (m per Pa), per unit
coefficient of the fully normalised Y_lm — pyslfp's
`traction_numbers(..., gravitating=False)`, sl3dpp's
`ComputeTractionNumbers(material, lmax, false)`, and (to come) AdGIA's
traction problem analysed on the surface. Self-adjointness gives
h_v = l(l+1) l_u in every code.

## Producing the exchange files

Each code reduces its answer to one exchange file per model
(`common/exchange.py`; columns `l h_u l_u h_v l_v`):

- **analytic** — `reference` (a launcher in this directory's build
  twin): `./reference --lmax 32 --out runs`.
- **pyslfp** — from the pyslfp checkout, which owns its environment:
  `cd ~/dev/pyslfp && poetry run python
  .../benchmarks/elastic/nongrav/pyslfp_reference.py --out <build
  runs dir>` (`--gravitating` writes the full-gravity twin under the
  same forcing, the E2 diagnostic).
- **sl3dpp** — its own driver, built in the sl3dpp repository with
  `-DSL3DPP_BUILD_BENCHMARKS=ON`: `traction_numbers <deck> <lmax>
  <out.txt>`, fed the decks `common/decks.py` writes (its E1 gate
  `TestTractionNumbers` already holds it to pyslfp at 1e-10 on the
  committed fixtures).
- **AdGIA** — `traction_benchmark` (built into `<build>/benchmarks/bin`
  with the other drivers): the non-gravitating traction problem on a
  case of `make_case.py`, the surface displacement analysed into
  harmonic coefficients, rigid modes projected in the solver. Degree-l
  tractions are applied one at a time, order 0; `spurious` reports the
  discrete leakage into the unforced harmonics.

      # both from <build>/benchmarks
      ./love_numbers/make_case two_solid --h 0.15 \
          --out elastic/nongrav/cases/two_solid
      mpiexec -np 8 ./bin/traction_benchmark \
          -c elastic/nongrav/cases/two_solid/case.json -lmax 8 -o 3 \
          -out elastic/nongrav/e1_two_solid_adgia.txt

## Comparing

`./compare <reference.txt> <other.txt> [...] [--tol 1e-8] [--verbose]`
reports the worst relative difference per channel, measured against the
largest reference value of each degree. Expected tiers: the spectral
codes against the analytic reference at ~1e-10 (mesh-rule dependent);
the FEM against the analytic reference at its discretisation level,
converging under refinement — measured on `two_solid`, lmax 8: the
worst channel (h_v, at l = 8, whose relative error is l(l+1) times
l_u's) falls from 6.3e-1 at h 0.3 order 2 to 1.4e-2 at h 0.15 order 3;
h_u's worst from 6.6e-2 to 1.6e-3, its l = 2 value from 5.6e-4 to
2.1e-6; and the reciprocity residual from 1.2e-2 to 8.5e-8.
