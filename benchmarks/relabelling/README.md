# Relabelling

The aspherical verification (`doc/mappings.md`): the same spherical
physical problem described from laterally relabelled coordinates, so the
pyslfp reference stays exact while every mapped code path acts. The
interior relabelling (`common/relabelling.hpp`) is the identity, with
identity gradient, on every interface and on and outside the surface, and
genuinely three-dimensional in between. The maps, the tests and the
measured results are in `doc/benchmarks.tex` (section "The relabelling
family"). Three legs:

```
cd <build>/benchmarks/love_numbers
./run fluid_core --h 0.3 --order 2 --np 8 \
    --method referential slip_broken --map 0.02
mpiexec -np 8 ../bin/relabelled_identity \
    -c runs/two_solid/h0.3/case.json -o 2 -map 0.02
mpiexec -np 8 ../bin/relabelled_identity \
    -c runs/fluid_core/h0.3/case.json -o 2 -map 0.02 -method slip_broken
mpiexec -np 8 ../bin/aspherical_reference \
    -m ../../data/aspherical_buffer_3d.mesh \
    -c runs/homogeneous/h0.3/case.json -o 2 -out aspherical_o2.json
```

The campaign runs all three (stages `mapped`, `identity`, `aspherical`)
and writes the identity logs and aspherical results to
`<out>/relabelling/`.

- **`--map A`** (exact analytic F, through the Love-number family's
  `run.py`; results suffixed `_map<A>`): the mapped solve must reproduce
  the pyslfp numbers of the unmapped one at the discretisation level, and
  its `spurious` column measures lateral leakage. Referential methods only
  (`referential`, `slip_broken`): the single-valued `slip` refuses maps,
  and the Eulerian classes are unmapped. Needs a case with
  `radial_profiles.txt`. Mapped `referential` passes; mapped
  `slip_broken` is not verified (its leakage is far above the unmapped
  level).
- **`relabelled_identity`** (interpolated F): the discrete
  change-of-variables identity at the level of the full coupled solve —
  the mapped assembly on the reference mesh against the standard
  assembly on the nodal-image mesh, compared entrywise; it prints the
  relative differences of u and zeta, per-attribute differences and
  operator-action probes, and `IDENTITY HOLDS` when both are below 1e-5.
  Strict on every welded case, the gauged fluid included. With
  `-method slip_broken` the run is informational: it also compares the
  sixteen assembled blocks of the broken organisation
  (`ApplyBrokenBlock`) and the stored constraint kernels
  (`ApplyBrokenKernel`).
- **`aspherical_reference`** (exact F): the same spherical physics
  described from an aspherical reference body, whose mesh is the image of
  a spherical gmsh mesh under the radial stretch of
  `meshes/aspherical_body.py`, with the exact inverse stretch as the map.
  The map is not the identity on the surface, the load carries its Nanson
  area factor, and the solution is compared as fields against the pyslfp
  radial solutions composed through the map (relative L2 of u over the
  body and of the Eulerian potential; degree one skipped). Single solid
  layers (`homogeneous`, `linear_solid`). The mesh manifest records the
  shape parameters (`--eps`, `--beta`, the buffer), and the driver adopts
  them; a contradicting `-eps`, `-beta` or `-buffer` aborts. Meshes for a
  sweep in the amplitude or the resolution:

      python <repo>/meshes/aspherical_body.py --dim 3 --buffer \
          --eps 0.02 --name _e0.02 --out <dir>
      python <repo>/meshes/aspherical_body.py --dim 3 --buffer \
          --scale 0.5 --name _s0.5 --out <dir>

  `plot.py` draws the runs' relative L2 field errors against degree, one
  series per results file, into `aspherical.png` beside them
  (non-converged degrees hollow):

      cd <build>/benchmarks/relabelling
      ./plot <dir-with-aspherical*.json>
