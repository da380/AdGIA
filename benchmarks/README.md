# Benchmarks

Comparisons of the library's solvers with independent reference solutions.
An example shows how a class is used; a benchmark says how accurate the
answer is, and at what cost.

The benchmarks are organised by family, one directory each with its own
scripts, drivers and README; `common/` holds what they share (the models,
the case builder, the case and mapping layers of the drivers), and
`campaign.py` is the master script that runs the families through stages.
A new family is a directory and a campaign stage.

What each benchmark tests, its mathematics, how its reference is
obtained, and the measured results are in `doc/benchmarks.tex`; the
READMEs here say how to run them.

| family | problem | reference |
|---|---|---|
| `love_numbers/` | load and tidal Love numbers of spherically layered, self-gravitating elastic bodies, across formulations, solvers and CMB treatments, with the response as fields | the radial solver of [pyslfp](https://github.com/da380/pyslfp) |
| `relabelling/` | the mapped (aspherical) machinery: the same spherical physics from relabelled coordinates, the discrete change-of-variables identity, an aspherical reference body | the same, exact under the relabelling |
| `perturbation/` | derivatives of the response with respect to the model: the degree-0 interface-shift check | pyslfp on the perturbed models |
| `viscoelastic/` | the viscoelastic time stepping, in four sub-families: non-gravitating box and sphere problems, the stepper survey, Maxwell Love-number histories | exact modal/Talbot references; RK4 at a small step; pyslfp through the correspondence principle |

## Setting up

The scripts run in the poetry environment of this directory, which is made
once:

```
cd benchmarks
poetry install        # planetmodel (gmsh, PyMFEM), pyslfp, matplotlib
```

The drivers are parallel programs, built with the library when it is
configured with `-DUSE_MPI=ON -DBUILD_BENCHMARKS=ON`, into
`<build>/benchmarks/bin/`. A benchmark is run from the build tree, like
the examples: in `<build>/benchmarks/<family>/` the build puts launchers
of that family's scripts, which start them with the Python environment
found at configuration (`BENCHMARKS_PYTHON` names another), the drivers
and the MPI launcher of that build:

| directory under `<build>/benchmarks/` | launchers |
|---|---|
| `.` | `campaign` |
| `love_numbers/` | `run`, `plot`, `cmb_report`, `make_case` |
| `relabelling/` | `plot` |
| `perturbation/` | `perturbation_check`, `plot` |
| `viscoelastic/box/` | `cases`, `reference`, `compare`, `study` |
| `viscoelastic/sphere/` | `meshes`, `cases`, `reference`, `study` |
| `viscoelastic/love/` | `laplace_reference`, `compare`, `make_case` |
| `viscoelastic/stepping/` | `survey` |

What a script writes goes where it is started, or to `--out`: `runs/`
beside the Love-number launchers for the sweeps, `runs_campaign/` for the
campaign, one directory per model holding its cases, results and figures.
Run the launchers in the build tree so that the outputs stay there. The
scripts of `viscoelastic/box` and `viscoelastic/sphere` refuse an output
path inside the source tree (`common/outputs.py`); the others do not check.
The reference histories in `viscoelastic/love/references/` are committed
reference data, not outputs.

## The master script

`<build>/benchmarks/campaign` runs the families stage by stage, behind two
profiles whose every default is a flag:

```
cd <build>/benchmarks
./campaign --profile local                    # rehearsal on this machine
./campaign --profile server --out /scratch/love
./campaign --profile server --stages methods field --lmax 20
./campaign --profile server --stages scaling --dry-run
./campaign --profile local --stages viscoelastic plot
./campaign --profile local --stages methods cmb plot --combined
```

- `--profile local`: the models `homogeneous`, `two_solid` and
  `fluid_core`, h = 0.3, order 2, degrees 0–4, 8 ranks; the default stages
  run end to end in about an hour — the shape to validate before a server
  run.
- `--profile server`: eight models (`homogeneous`, `two_solid`,
  `linear_solid`, `fluid_core`, `inner_core`, `stratified_core`,
  `earth_like`, `prem_4`), an h-ladder 0.2/0.15/0.1 at orders 2 and 3,
  degrees to 16, 100 ranks (sized for a 128-core shared-memory machine),
  both Eulerian solvers, and a weak-scaling stage whose rungs hold the
  unknowns per rank roughly constant (h ~ np^(-1/3)); consider
  `--partition` so that no rank reads a whole fine mesh.

The default stages are `methods`, `cmb`, `field`, `mapped`, `identity`,
`aspherical`, `perturbation`, `scaling` and `plot`. The stages
`viscoelastic` (Maxwell Love-number histories), `viscoelastic_box` and
`viscoelastic_sphere` (every study of those sub-families at the campaign's
profile) are opt-in: they run only when named with `--stages`. Everything
delegates to the family scripts, which skip work whose results exist, so
an interrupted campaign resumes where it stopped. The tree mirrors the
families:

```
<out>/love_numbers/<model>/h<h>/    cases, Love and field results
<out>/love_numbers/<model>/*.png    the family's figures (plot.py)
<out>/love_numbers/cmb_summary.md   the CMB cost-accuracy tables
<out>/relabelling/                  identity logs, aspherical results, aspherical.png
<out>/perturbation/<model>/         shifted profiles, references, runs, figures
<out>/scaling/                      the weak-scaling rungs
<out>/scaling_summary.md            their timings
<out>/viscoelastic/<model>/         (opt-in) case, Laplace reference, FE histories, errors, figures
<out>/viscoelastic_box/<study>/     (opt-in) the box studies, summaries and figures
<out>/viscoelastic_sphere/<study>/  (opt-in) the sphere studies
<out>/campaign_log.md               every command and stage outcome (appended)
```

`--combined` (opt-in) runs the Love numbers of the `methods` and `cmb`
stages combined (`run.py --combined`, `love_numbers/README.md`: one load
solve for all the degrees and one tidal solve), the results suffixed
`_combined` beside, not over, those by degree. It leaves the other stages
as they are: `field` has no Love solve, `mapped` checks an agreement of
the order of the combined solves' leakage between degrees, `perturbation`
differences exact solves by degree, `identity` and `aspherical` run other
drivers, and `scaling` times the solves by degree. The collating scripts
(`plot.py`, `cmb_report.py`, the scaling summary, `talk_figures.py`)
read either kind; a combined run's cost is its one load solve for all
the degrees, shown once and labelled combined, and `cmb_report.py`
tabulates the combined treatments apart from those by degree.

The campaign does not make every run that `doc/benchmarks.tex` reports:
the all-model sweep at h = 0.2, the uncapped `fluid_core` h-ladder, the
`prem_4` CMB runs and the dense viscoelastic histories are made with the
family scripts; the commands are in the section "Reproducing the figures"
of `doc/benchmarks.tex`.

## Figures for talks and for the documentation

`talk_figures.py` draws single-message figures (`methods.png`,
`cmb.png`, `identity.png`, `derivative.png`, `aspherical.png`,
`ladder.png`, `models.png`, `viscoelastic.png`, `fieldmap.png`) from the
build-tree runs; every input is a flag whose default is the campaign's
location, so it is started from `<build>/benchmarks` with the poetry
environment's Python:

```
cd <build>/benchmarks
poetry -C <repo>/benchmarks run python <repo>/benchmarks/talk_figures.py --out talk
```

`talk_render.py` is a ParaView render of a cap-load solve, run with
`pvbatch` on the output of `run.py --field --paraview`. The figures of
`doc/benchmarks.tex` are copies of these and of the family scripts'
figures in `doc/figures/benchmarks/`, with a prefix naming their source
(`talk_`, `fc_`, `pert_`, `box_`, `sphere_`, `ve_`).

## What the benchmarks establish

In brief (the measurements are in `doc/benchmarks.tex`):

- Every formulation agrees with the radial reference at the
  discretisation level on its domain of validity, and the cross-method
  spread at coarse resolution is mesh error, not physics. The welded
  mapped assembly passes a solver-level change-of-variables identity to
  ~1e-6; the slipping-interface forms under a mapping are not verified.
- For loading at l >= 1 the Dahlen path is the cheapest and as accurate
  as anything here. The dearer formulations buy physics it cannot
  represent: the compressible fluid at degree zero, non-neutral core
  stratification, the slipping interface, and the referential family's
  mapped machinery. The gauged and referential methods cost several
  times the Dahlen path per load solve, the slipping methods one to two
  orders of magnitude more.
- The standard unmeshed-core CMB condition (`uniform`) is safe for
  loading at l >= 2 (h' within 0.1 %, l'_2 and k'_2 ~0.3 %), about ten
  times worse on tides and so on the rotational feedback, which runs
  through the degree-2 tidal response; buoyancy alone (`winkler`) is safe
  for neither. At degree zero every Dahlen-family treatment is wrong and
  the gauged one is right.

## The shared pieces (`common/`)

| file | what it does |
|---|---|
| `models.py` | the models, as planetmodel models in SI, and the units they are solved in |
| `make_case.py` | one case: mesh, fields and manifest through planetmodel, reference through pyslfp |
| `benchmark_case.hpp` | what the Love-number drivers share: a case set up as a problem, the analysis of its solution, the common options |
| `relabelling.hpp` | the mapped layer of the drivers: relabellings, interface shifts, exact radial profiles |
| `reference_field.hpp` | the reference solution as fields on the mesh |
| `viscoelastic_common.hpp` | what the box and sphere drivers share: the step grid, the schemes, the cost bookkeeping |
| `costs.py` | the cost of a run (by degree or combined) for the collating scripts |
| `drivers.py` | where a build's drivers are, and how the scripts run and log commands |
| `outputs.py` | the guard that keeps the box and sphere scripts' outputs out of the source tree |

A model is defined once, in `models.py`, and both solvers are given that
object. planetmodel meshes its skeleton, with a buffer shell outside the
surface, and writes the density and the bulk and shear moduli as L2
GridFunctions on the mesh, so that a discontinuity at an interface stays
one; the manifest beside the mesh names the layers and interfaces, records
the units and G, says which layers are fluid, and holds the model's exact
one-sided field values on each interface. The driver reads all of it
through `MeshManifest` (`mesh_manifest.hpp`) and sets nothing about the
model itself. pyslfp solves the same model on its own radial mesh.
