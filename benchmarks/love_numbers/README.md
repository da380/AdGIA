# Love numbers

Load and tidal Love numbers of the spherically layered models, against
the radial solver of pyslfp, across formulations, solvers and CMB
treatments, with the response compared as fields as well. What is
compared and how, the models, and the measured results are in
`doc/benchmarks.tex` (section "The Love-number family"); this README says
how to run it.

```
cd <build>/benchmarks/love_numbers
./run homogeneous --h 0.3 0.2 0.14 --order 2 3 --np 8
./run all --h 0.2 --order 2 3 --np 8 --field
./plot runs
```

`run.py` makes a case for each model and element size (the mesh, the model
on it and the reference, through `common/make_case.py`), runs the drivers
on it for each order, and skips what is already there (`--force` reruns,
`--remake` remakes the cases), so a sweep can be extended later or on
another machine. `all` is every model of `common/models.py`, including
`prem_6`, whose 21 km crust makes a mesh of millions of elements, and
`homogeneous_lithosphere`, meant for the viscoelastic family; name the
models to leave them out. `plot.py` prints the relative errors and writes
the figures, for one model or for all of them, with a summary in
`runs/summary.md`. On a larger machine only the numbers change:

```
./run earth_like prem_4 --h 0.1 0.07 0.05 --order 2 3 --np 128 --lmax 10 \
    --field --partition --out /scratch/love
./plot /scratch/love
```

With `--partition` each case is cut into the parts of the ranks beforehand
(`partition_case`, a serial program), and each rank reads its own; without
it every rank reads the whole mesh, which is simpler and costs each the
memory of the whole.

`--launcher-args` passes the MPI launcher further arguments (binding, host
files), `--mpiexec` names another launcher than the build's,
`--program-args` passes further options to the drivers (e.g.
`"-geps 1e-1"` or `"-kkt -kkt-order 1"`), and `--dry-run` prints the
commands of a sweep without running them. Each script has `--help`, and
so has each driver (`../bin/love_benchmark --help`).

## Outputs

```
runs/<model>/h<h>/                       the case: mesh, fields, manifest, reference
runs/<model>/h<h>/results_o<p><suffix>.json   love_benchmark (Love numbers, coefficients, radial functions, cost)
runs/<model>/h<h>/log_o<p><suffix>.txt
runs/<model>/h<h>/field_o<p><suffix>.json     field_benchmark, with --field
runs/<model>/h<h>/paraview_o<p><suffix>/      with --field --paraview
runs/<model>/h<h>/parts_<np>/                 with --partition
runs/<model>/*.png                       plot.py: errors, love_numbers, timing, profiles,
                                         field_<run>, field_spectrum, convergence, field_convergence
runs/summary.md                          plot.py: one row per run
runs/cmb_summary.md                      cmb_report.py
```

The suffix names the variant: `_<method>` for a method other than
`dahlen`, `_<cmb>` for a CMB treatment other than `full`, `_map<A>` for a
mapped run, `_schur` for the Schur solver, `_combined` for combined
solves.

## The methods

`--method` sweeps the formulations over the same case, one results file
(and one series in every figure) each, with the wall times and iteration
counts compared in `timing.png` and in the summary:

```
./run fluid_core --h 0.3 --order 2 --np 8 \
    --method dahlen gauged referential slip slip_broken --solver 1 0
./plot runs
```

| method | formulation |
|---|---|
| `dahlen` | Eulerian, fluid eliminated (`doc/self_gravitation.md`); `--cmb` picks its fluid-interface treatment |
| `gauged` | Eulerian, the fluid in the displacement space with a gauge shear penalty (`doc/gauged_fluid.md`) |
| `referential` | the welded gauged referential formulation (`doc/gravitating_elasticity.md`): bare moduli from the exported hydrostatic pressure `p0`, equilibrium stress `-p0 I` |
| `slip` | the slipping fluid–solid interface, single-valued potential (`doc/slip_interface.tex`): broken displacement pair, normal-jump constraint by penalty and augmented Lagrangian, or by a KKT multiplier (`-kkt`) |
| `slip_broken` | the same with the potential broken per region as well |

`--solver 1 0` runs the Eulerian methods under both of their linear
solvers (block MINRES, and the Schur-complement CG with the `_schur`
suffix); the referential and slipping methods have their own solvers and
ignore the choice. The referential and slipping methods solve the load
problems only (no tidal numbers, no radial functions) and need a case
with the `p0` field; the slipping methods accept one fluid core inside
one solid shell, so no model with an inner core. The driver options of
the formulations (`-geps`, `-gref`, `-gauge-kkt`, `-theta`, `-al`,
`-sweep-tol`, `-kkt`, `-kkt-order`) and their defaults are in
`doc/benchmarks.tex` ("The formulations compared").

On a fluid core that is not neutrally stratified (`fluid_core`'s uniform
core) the compressible methods differ from the Dahlen/pyslfp response by
O(N²) at every degree, so converged cross-method ladders split there; at
h ~ 0.3 the split is below the mesh error (`doc/gauged_fluid.md`).

`--map A` runs the referential methods from relabelled coordinates — the
relabelling family's benchmark, described in `../relabelling/README.md`.

## The CMB treatments

`--cmb full nomass uniform winkler` runs the Dahlen path under each
fluid-interface treatment, from the full stratified-fluid conditions to
the standard unmeshed-core condition of the GIA codes (`uniform`) and
buoyancy alone (`winkler`); `doc/benchmarks.tex` ("The CMB treatments")
states each with its published source. `cmb_report.py` collates their
cost against their accuracy — setup and solve time, iterations and
potential unknowns, against the error relative to the reference and to
the full treatment on the same mesh (the approximation error alone) —
into `runs/cmb_summary.md`:

```
./run fluid_core --h 0.3 --order 2 --np 8 --cmb full nomass uniform winkler
./cmb_report runs
```

The treatments coincide on a uniform core (`fluid_core`) and separate on
the stratified ones (`stratified_core`, `earth_like`, the PREM models);
the documented comparison runs `prem_4`:

```
./run prem_4 --h 0.2 --order 2 --lmax 5 --np 8 --cmb full nomass uniform winkler
./run prem_4 --h 0.2 --order 2 --lmax 5 --np 8 --method gauged
./cmb_report runs
```

The core stays meshed in every treatment, so the potential unknowns do
not change and the gain is solver-side only.

## The models

| name | layers |
|---|---|
| `homogeneous` | one uniform solid |
| `two_solid` | two uniform solids, every parameter jumping between them |
| `fluid_core` | a uniform fluid core under a uniform solid mantle |
| `inner_core` | a solid inner core, a fluid outer core and a mantle, each uniform |
| `linear_solid` | one solid, density and velocities linear in radius |
| `stratified_core` | a fluid core whose density falls with radius, under a mantle, every parameter linear in radius |
| `earth_like` | inner core, fluid outer core and mantle, every parameter linear in radius within each |
| `prem_4` | PREM without its ocean, isotropic, on four layers: inner core, outer core, lower mantle, upper mantle with the crust |
| `prem_6` | the same on six: the transition zone and the crust are layers of their own |
| `homogeneous_lithosphere` | `homogeneous` cut at 5700 km, so that the outer shell can stay elastic (viscoelastic family) |

The values are in `doc/benchmarks.tex` ("The models"). A new model is a
function in `common/models.py` and an entry in `MODELS`; every one is
planetmodel's `LayeredIsotropicElastic`, whose layers take a constant or
polynomial coefficients in the radius. The PREM models keep the named
boundaries and merge what lies between them (`Model.coarsened`), each
parameter within a merged layer being the cubic closest to PREM's.

A case is solved in the units of `models.scales` (the outer radius, 5000
kg m^-3, and by default the time unit that makes G one);
`make_case --time-scale` sets another time unit, and G then takes the value
the manifest records, which is the one the driver uses.

## Radial functions and fields

`love_benchmark` writes, for the Eulerian methods, the radial functions
of each solution by layer (`-radial`, the default; `-no-radial` leaves
them out); `profiles.png` draws them over the reference's, with the
harmonic coefficients on the interfaces as points.

`--field` (or `--field-only`) runs `field_benchmark` as well, for the
Eulerian methods: an off-axis cap load summed over the degrees up to
`--field-lmax` (default 8), compared with the reference as fields (u over
the solid, phi over body and buffer). Its outputs give the maps
`field_<run>.png` and the error by degree `field_spectrum.png`;
`--paraview` writes the fields, the reference and their difference.
With a fluid layer the summed load starts at degree 1 by default;
`field_benchmark -lmin 0` (run by hand) includes degree zero for either
method and shows Dahlen's degree-zero difference from the reference
(`doc/benchmarks.tex`, "Degree zero with a fluid layer"). The Love-number
plots show Dahlen's degree-zero point for the same reason.

## Combined solves

`run.py --combined` runs `love_benchmark -combined`: one solve for the
sum of the unit loads of every degree from `lmin` to `lmax`, and one for
the sum of the tidal potentials of degree two and above, every degree's
numbers analysed from the one solution. The results,
`results_o<p>[_<method>]_combined.json`, keep their per-degree form with
`"combined": true`; the outer iterations and wall time of the two solves
are under `"combined_solves"`, and the per-degree counts and times are
null. The collating scripts read such files alongside those by degree
(`plot.py`; `cmb_report.py`, `campaign.py`'s scaling summary and
`talk_figures.py` through `common/costs.py`): a combined run's cost is
its one load solve for all the degrees, attributed once to the run and
labelled combined. `campaign.py --combined` runs the methods and CMB
stages this way (`../README.md`).

The discrete operator is not exactly rotationally invariant, so a load of
degree l leaks into the responses of the other degrees, which the solves
by degree discard and the combined solve adds to each degree's number. On
`fluid_core` at h = 0.3, order 2, the difference is 2e-5 to 4.5e-3
relative, below the error against the reference in all entries but one;
`doc/benchmarks.tex` ("Combined solves") has the numbers and the cost.
Dahlen's fluid leaves degree zero out of the combined load, since its
degree-zero response differs from the reference by design (it does not
represent the fluid's compressibility there); the other methods keep it.

## Resolution and h-refinement

The element size `--h` is that on every interface, in units of the outer
radius, capped at `--angular` (default 0.3) times the interface's radius,
so that a small inner core is still a sphere, and at `--thin` (default 4)
times the thickness of the layers it bounds; the size grows to twice
`--h` away from the interfaces. gmsh's Netgen optimiser removes the
slivers the mesher leaves. The geometry is of order two.

Each `--h` of a sweep is a case of its own, meshed afresh at that size:
refinement is by re-meshing, not by MFEM's uniform refinement, so a ladder
can take steps finer than a factor of two. `plot.py` draws the relative
error of each Love number against the element size at fixed order in
`convergence.png`, and of the cap-load fields in `field_convergence.png`,
with the observed rate fitted over the ladder and a table of the rates
printed alongside. The angular cap holds the elements of a deep interface
at a fixed size for every `--h` above it (0.165 on `fluid_core`'s CMB), so
a convergence ladder is run with `--angular 1.0`:

```
./run fluid_core --h 0.3 0.24 0.19 0.15 0.12 --order 2 3 --lmax 4 \
    --angular 1.0 --field --out ladder
./plot ladder
```
