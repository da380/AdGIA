# Viscoelastic: the sphere benchmarks

Non-gravitating, solid-only viscoelastic problems on spherically layered
balls (3-D) and discs (2-D) under a surface pressure `S(t) Y_l`, against
exact references for radial models, and against a much smaller step on
the same mesh for laterally varying viscosity. The companion of the box
sub-family (`../box`), whose histories, comparison and AAA/Talbot
machinery it reuses. The problem, the models, the Euler-type radial
reference and the measured results are in `doc/benchmarks.tex` (the
sphere subsections of "The viscoelastic family").

| file | what it is |
|---|---|
| `viscoelastic_sphere.cpp` | the driver: a case, a mesh, a scheme -> surface coefficients of every degree in time, with their cost |
| `meshes.py` | layered balls and discs by planetmodel, interfaces as mesh surfaces |
| `cases.py` | the models and the case writer (schema in its docstring) |
| `reference.py` | the exact references of radial models |
| `derive_euler.py` | derives (sympy) and checks the Euler matrix the reference uses |
| `study.py` | the sweeps and figures: `radial`, `lateral` |

Models: `maxwell_ball`, `lid`, `core_lid`, `channel`, `burgers_lid`,
`prony_lid` (radial), `lateral` and `lateral_weak` (the mantle's
relaxation time varying smoothly across the polar axis, from 1 to C and
from 1/C to 1; `--contrast C`). The body is free and the load has degree
l >= 2, so there is no net force or torque; without gravity a body with
nothing elastic in it flows, and a free body whose elastic core sits in a
fully relaxing shell has an unrestrained relative mode — which is why
`lateral_weak` has a standard-linear-solid mantle.

**The mesh comes first**: every interface radius is a multiple of 0.2
and planetmodel makes it a surface of the mesh. Meshes are uniform (one
size `h` everywhere), curved of order 2, built once and cached
(`cases.py --meshes`, default `./meshes`; the studies use
`<out>/meshes`).

## Running

From `<build>/benchmarks/viscoelastic/sphere` (outputs refuse to go into
the source tree):

```
./cases core_lid --dim 2 --degree 2 --h 0.1 --out cl.json
./reference cl.json --out cl_ref.json
mpiexec -np 2 ../../bin/viscoelastic_sphere -c cl.json -o 2 -dt 0.05 \
    -out cl_res.json
../box/compare cl_res.json --reference cl_ref.json

./study radial                 # profile "local": discs, coarse balls
./study lateral                # discs, lateral and lateral_weak
./study all --profile server   # finer discs and balls, the 3-D lateral study
```

The driver takes `-scheme` (`exptrap` default, `sdirk23`, `be`, `etd1`,
`rk4`, `adaptive`), `-dt`, `-min-steps`, `-align`/`-no-align`, `-rtol`,
`-atol`, `-o`, and `-r` (uniform refinements of the curved mesh).
`study.py` takes `--out` (default `.`), `--jobs`, `--np`,
`--figures-only` and `--dry-run`; each study writes
`<out>/<study>/summary.json` and its figures (`radial_<dim>d.png`;
`<model>_dt_<dim>d.png` and `<model>_cost_<dim>d.png` for the lateral
models), with the per-run cases, references and results under
`<out>/<study>/runs/`. The campaign runs every study as its
opt-in stage `viscoelastic_sphere`.
