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
- **sl3dpp** — its own driver, in the sl3dpp repository (its E1 gate
  `TestTractionNumbers` already holds it to pyslfp at 1e-10 on the
  committed fixtures).
- **AdGIA** — the FEM driver of this family (to come): the traction
  problem per ladder mesh, surface fields analysed into coefficients.

## Comparing

`./compare <reference.txt> <other.txt> [...] [--tol 1e-8] [--verbose]`
reports the worst relative difference per channel, measured against the
largest reference value of each degree. Expected tiers: the spectral
codes against the analytic reference at ~1e-10 (mesh-rule dependent);
the FEM against the analytic reference at its discretisation level,
converging under refinement.
