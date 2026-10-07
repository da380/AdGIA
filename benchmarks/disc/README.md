# The disc family (2-D)

The 2-D testing ground of the slip programme
(`doc/planning/no_gravity_analytics.md`): the machinery of the slipping
interface against exact references, at a cost that allows resolution
ladders as a matter of course. Rung 0 (this directory, gravity-free) is
implemented; rungs 1 (uniform pre-pressure) and 2 (gravity, against the
radial-ODE reference) follow once it is clean.

## Rung 0: no gravity, no pre-stress

A fluid disc (bulk modulus `kappa_f` only) inside a plane-strain solid
annulus, loaded degree by degree with a normal traction
`-cos(l theta)` on the outer circle (at l = 1 the self-equilibrated
variant; see the note). The exact solution is closed form
(`disc_reference.py`): the fluid reduces to an area spring that is
inert for l >= 1, the annulus is a Michell problem, and the interface
derivative d(response)/d r_cmb is exact. Without gravity every
tangential slip is removable, so **welded == slip exactly**: the
welded-slip gap measures the constraint machinery and nothing else.

- `disc_reference.py` — the exact solutions and derivatives, with an
  internal validation suite (`python disc_reference.py`).
- `disc_benchmark.cpp` — the FE driver (one source, serial and
  parallel): `-method welded` (one continuous space on the body
  SubMesh, Tikhonov gauge refinements) or `-method slip`
  (broken spaces, nodal pairing, normal-jump penalty + augmented
  Lagrangian, as `examples/sliding_fluid_ellipse.cpp`). Responses are
  extracted convention-free: the harmonic coefficient of the field
  divided by that of the known pattern on the same circle.
- `check.py` — runs the (order, refinement) ladder for both methods
  and checks: agreement with the reference converging down the
  ladder, and the welded-slip identity holding near solver tolerance
  on every rung. At l = 1 comparisons use the translation-invariant
  U + V per circle.

```
cd <build>/benchmarks/disc
./check --orders 2 3 --refinements 0 1 --lmax 4
```

The mesh is the canned `elastogravity_two_layer_2d.msh` (core
attribute 1, mantle 2, vacuum buffer 3 — the buffer stays outside the
displacement spaces), interface at r = 3483/6371.

Not yet here: the shift-map leg (the mapped assembly under a jumping
F against the exact derivative — the next unit of work), rung 1's
pre-stressed exponent basis, and rung 2's gravitating twins with the
radial-ODE reference.
