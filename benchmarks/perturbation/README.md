# Perturbation

Derivatives of the response with respect to the model, checked against
the radial solver on perturbed models. The method, its assumptions and
the measured results are in `doc/benchmarks.tex` (section "The
perturbation family").

## The degree-0 interface shift

One fixed spherical mesh; a family of physical models with one interior
interface at r_k + eps, described from it by the piecewise-linear radial
map of `common/relabelling.hpp` (`love_benchmark -map-shift`; the family
has no driver of its own). Every member of the family is still spherical,
so pyslfp solves it exactly at every finite eps; and because the mesh is
fixed, the discretisation bias largely cancels in the finite difference,
so

    [R_3D(+eps) - R_3D(-eps)] / 2 eps   vs   the same of pyslfp

compares the derivative of the response with respect to the interface
radius, through the mapped assembly. The 1-D side is a finite difference
of pyslfp solves, not an analytic kernel.

```
cd <build>/benchmarks/perturbation
./perturbation_check ../love_numbers/runs/fluid_core/h0.3 \
    --eps 0.01 0.02 --method referential slip_broken --np 8
```

The case must exist (made by `love_numbers/run`, whose unmapped run of
each method is the eps = 0 point). For each eps of the ladder the script
builds the perturbed model's exact radial profiles and pyslfp reference
(under `--out`, default beside the case: `shift_<+-eps>/`), runs
`love_benchmark -map-shift eps -profiles <perturbed profiles>` for each
method and sign (`results_o<p>_<method>_shift<+-eps>.json`), and prints,
per degree and Love number, the absolute agreement at each eps and the
derivative comparison, the eps-ladder showing the O(eps^2) remainder.
Defaults: `--eps 0.01 0.02`, `--method referential`, `--np 8`,
`--lmax 4`; `--interface` picks the interface (1 = the innermost),
`--program-args` passes further driver options. The models with a
shiftable interface are named in `models.SHIFTABLE` (`fluid_core` moves
its CMB, `two_solid` its mid-mantle discontinuity). The methods are the
mapped ones, `referential` and `slip_broken`.

`plot.py` draws what the tables say, one figure per method
(`perturbation_<method>_o<p>.png` beside the results): the absolute
agreement with pyslfp by degree, one line per eps (the unmapped base run
labelled eps = 0); the derivative check as an identity plot, the 3-D
central difference against the 1-D one, one point per degree and Love
number (marker the number, colour the degree); and the relative
discrepancy of each derivative:

```
./plot ../love_numbers/runs/fluid_core/h0.3 --out <root> \
    --method referential slip_broken
```

The campaign runs the check (stage `perturbation`) on `fluid_core`
(`referential`, `slip_broken`) and `two_solid` (`referential`), at
eps = 0.02 in the local profile and 0.01, 0.02 in the server profile.

## What the driver does for a shift run

- **Mass and gravity from the perturbed model.** The perturbed model's
  mass differs from the base model's at O(eps) (dM/dr_k = 4 pi r_k^2
  [rho]), so the driver takes the mapped model's mass, surface gravity and
  interface gravities from the exact radial profiles
  (`RadialProfiles::EnclosedMass`), not from mesh integrals of the base
  fields; the normalisations of the Love numbers depend on it.
- **Interpolated F.** The exact shift map has a discontinuous gradient at
  the moving interface, and on curved facets the quadrature points of the
  one-sided slip interface kernels sit O(h²) below the analytic radius,
  so with the exact map most of each facet would receive the fluid-side
  slope where the solid-side value belongs. The per-SubMesh interpolated
  map (`-map-interp`) samples only its own elements, and the map's value
  is continuous, so it is side-consistent by construction. The driver
  therefore forces interpolated F for any `-map-shift` run. The
  interpolation perturbs the represented model by O(h^p) consistently at
  both signs, which cancels in the central difference.
- **Full-tolerance sweeps.** Every augmented-Lagrangian sweep of the
  slipping methods runs at full tolerance in a shift run, whatever
  `-sweep-tol` says, so that the endpoints are reproducible.

The referential derivatives agree with the reference to about 1 % at
degree 0 and 1–2 % in k'; the slip_broken derivatives at degrees 2 and 3
are not verified (they are several times further off than the
referential ones on the same mesh). On `fluid_core` the degree-2 numbers
carry an absolute offset against pyslfp at every eps, for both methods:
the O(N²) mismatch of the compressible methods on its non-neutral core.
The numbers are in `doc/benchmarks.tex`.
