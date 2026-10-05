# Viscoelastic

Benchmarks of the viscoelastic time stepping (`ViscoelasticOperator`)
and of the rheologies, in four sub-families. Each has its own README,
and its launchers in `<build>/benchmarks/viscoelastic/<sub-family>/`.

| sub-family | what it tests | reference |
|---|---|---|
| [`box/`](box/README.md) | non-gravitating, solid-only problems on rectangular meshes, 2-D and 3-D: homogeneous bodies (the integrators alone), laterally uniform columns, Fourier-mode loading of layered slabs; general load histories, multi-branch rheologies, viscosity contrasts to 1e8 | exact: closed-form modes and convolutions, AAA + Talbot |
| [`sphere/`](sphere/README.md) | non-gravitating, solid-only layered balls (3-D) and discs (2-D) under surface loads of degree l; radial models, and smooth lateral viscosity contrasts to 1e4 for the integrators | exact for radial models (Euler-type radial system, AAA + Talbot); a much smaller step on the same mesh for lateral ones |
| [`love/`](love/README.md) | Maxwell load Love-number histories of gravitating spherical models (fluid cores, elastic lithosphere) | pyslfp through the correspondence principle, Gaver–Stehfest inversion |
| [`stepping/`](stepping/README.md) | the stepper survey on the clamped beam of `examples/viscoelastic_schemes` | RK4 at a small step |

The box and sphere sub-families separate the time integration and the
viscoelastic discretisation from gravity and fluids (and, for the box,
spherical geometry), which the quasi-static elastic benchmarks
(`love_numbers/`) test on their own; their references are exact to
~1e-10, so that a mismatch is the code's. The Love-number histories
remain the integrated test of the whole self-gravitating machinery.

`doc/benchmarks.tex` (section "The viscoelastic family") describes all
four — problems, references, the mathematics of the comparisons — with
the measured results; `doc/viscoelasticity.md` ("Time stepping") describes
the integrators and how to choose between them. The campaign runs the box,
sphere and Love-number sub-families as its opt-in stages
`viscoelastic_box`, `viscoelastic_sphere` and `viscoelastic`. Every script
of `box/` and `sphere/` refuses to write into the source tree
(`common/outputs.py`); run them, and the others, from the build tree.
