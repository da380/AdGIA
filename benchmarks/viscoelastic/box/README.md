# Viscoelastic: the box benchmarks

Non-gravitating, solid-only viscoelastic problems on rectangular meshes
(2-D and 3-D, `Mesh::MakeCartesian*`, periodic sideways for slabs), each
against an exact reference. They test the time integrators and the
viscoelastic discretisation without gravity, fluids or spherical
geometry, which the quasi-static elastic benchmarks test separately.
The problems, the models, the derivation of the references and the
measured results are in `doc/benchmarks.tex` (the box subsections of
"The viscoelastic family").

| file | what it is |
|---|---|
| `viscoelastic_box.cpp` | the driver: a case, a mesh, a scheme -> observables in time with their cost |
| `cases.py` | the models and the case writer (schema in its docstring) |
| `histories.py` | load histories S(t) as pieces, and their exact convolutions (`python histories.py` runs its self-test) |
| `reference.py` | the exact references |
| `compare.py` | history / elastic / relax errors of one run, and a figure |
| `study.py` | the sweeps and figures: `steppers`, `slab`, `stehfest`, `contrast` |

## The problems, in short

- **Homogeneous box** (`--load uniaxial_stress`: pure traction;
  `uniaxial_strain`: the whole boundary prescribed): finite elements exact
  in space, so every error is the integrator's. Models `maxwell`, `sls`,
  `burgers`, `prony_wide`.
- **Column** (`--mode 0`, a slab under a uniform pressure) and **Fourier
  slab** (`--mode 1 ...`, pressure cos(kx x) cos(ky y) on top, clamped
  base, periodic sides): models `maxwell_slab`, `lid`, `channel`,
  `sls_stack`, `burgers_lid`, `prony_lid`, `contrast`, `gradient`
  (`--contrast C` sets their viscosity contrast).
- **Histories** (`--history`): `heaviside`, `ramp`, `smooth_step`,
  `load_unload`, `sawtooth`, `periodic`, `multi_sine`, `ramped_periodic`.

The references are finite sums of real relaxation modes with closed-form
convolutions: the modes of the branch-variable system (homogeneous box,
column), or of the correspondence-principle transfer function recovered
by AAA (slab), checked against fixed-Talbot inversion. When the modal
form fails its checks at large viscosity contrasts and the load is a
Heaviside, the reference takes the Talbot history (`"method": "talbot"`);
other histories are then flagged `"reliable": false`.
`reference.py --stehfest 12 16` stores Gaver–Stehfest inversions as well,
for the inversion study.

## Running

From `<build>/benchmarks/viscoelastic/box`:

```
./cases lid --dim 2 --mode 1 --history periodic --out lid.json
./reference lid.json --out lid_ref.json
mpiexec -np 4 ../../bin/viscoelastic_box -c lid.json -o 2 -nx 20 -nz 10 \
    -scheme exptrap -dt 0.05 -out lid_res.json
./compare lid_res.json --reference lid_ref.json --plot lid.png

./study steppers            # 5933 runs on one element, ~15 min on 10 cores
./study slab                # profile "local" (default): laptop-sized
./study stehfest
./study contrast            # 488 runs, ~2 h on 10 cores
./study all --quick
./study all --profile server   # + the long 3-D runs (p = 2 at nz = 10,
                               #   over 1.5 h each), more ranks
```

`study.py` takes `--out` (default `.`), `--jobs` (parallel runs),
`--np` (ranks per run), `--quick`, `--figures-only` (redraw from the
stored summaries) and `--dry-run`. Each study writes
`<out>/<study>/summary.json` (one record per run: errors, solves,
assemblies, preconditioner setups, iterations) and its figures beside it
(`steppers_dt_<model>_<load>.png`, `steppers_cost_<history>.png`,
`steppers_alignment.png`, `steppers_transient.png`,
`slab_convergence.png`, `slab_interfaces.png`, `stehfest_*.png`,
`contrast_n<mode>.png`); the per-run cases, references and results go
under `<out>/<study>/runs/`. Runs are cached by their options, so delete
`runs/` after changing the driver.

The campaign runs every study as its opt-in stage `viscoelastic_box`
(`./campaign --profile server --stages viscoelastic_box`).

## Driver options and rules

- `-scheme` (`etd1`, `exptrap`, `be`, `sdirk23`, `rk4`, `adaptive`),
  `-dt`, `-rtol`, `-atol`; `-nx` (elements along each horizontal
  direction), `-nz`, `-o`.
- **The mesh comes first**: every layer interface is a multiple of 0.2
  (the coarsest grid, nz = 5), so every uniform mesh with nz a multiple
  of 5 has the material discontinuities on element faces, and the driver
  refuses a mesh that cuts a layer. `-no-conform` allows it, for the
  interface study only; `-rheology pointwise` replaces the per-layer
  composite rheology by one rheology that looks the layer up at every
  point.
- **The step grid**: fixed-step schemes divide each interval between
  output times (and aligned breakpoints) into equal steps no longer than
  `-dt` (at least `-min-steps`). `-align` (the default) puts the history's
  breakpoints on the step grid; `-no-align` lets them fall inside steps.
- **At a jump** the step ending there sees the load's left limit and the
  next starts from the right limit (`ViscoelasticOperator::
  InvalidateDisplacement`); the evaluation snaps the stepped time to the
  jump time, since rounding can put it an ulp past.
