# The disc family (2-D)

The 2-D testing ground of the slip programme
(`doc/planning/no_gravity_analytics.md`): the machinery of the slipping
interface against exact references, at a cost that allows resolution
ladders as a matter of course. Rung 0 (this directory, gravity-free) is
implemented, as are rung 1 (uniform pre-pressure) and rung 2's
radial reference (gravity; the gravitating FE driver is pending).

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

## The shift leg

`-shift eps` solves the model with the interface at `r_cmb + eps`
through the mapped assembly on the same fixed mesh (the exact
`InterfaceShift` map of `benchmarks/common/relabelling.hpp`, with its
solid-side band at the kink): the covariant `ElasticTensorIntegrator`
volume forms, the Nanson constraint row, and the mapped rotation null
vectors. The map is the identity on and beyond the surface, so loads
and surface extraction are unmapped. `check --shift 0.02` runs both
methods at +/- eps and compares the central difference of the surface
responses against the exact interface derivative of the reference —
the welded run isolates the mapped volume operators under a jumping
F, the slip run adds the mapped constraint row on top.

## Rung 1: uniform surface pressure

`-P0 p` solves about the pre-stressed reference state S_e = -P0 I
(uniform surface pressure, no gravity): bare moduli
(lam - P0, mu + P0), the geometric stiffness everywhere, mu_b = P0 in
the fluid, and B_Sigma with pi = P0 on the slip side
(NewSlipInterfaceMatrix) — the first exact test of B_Sigma. The
reference collapses in effective variables (the note, rung 1): the
volume operator and interface conditions are the rung-0 ones, only
the loaded-surface rows change to the dead-traction amplitudes of
delta P . N. `check --P0 0.3` runs the ladder. Not combined with
`-shift` yet.

Not yet here: the shift x P0 combination, and a gravitating FE driver
for rung 2 (the radial reference below exists).

## Rung 2: gravity, the per-degree radial reference

`doc/planning/disc_gravity_reference.md`. Two Python modules, with no
FE side yet:

- `disc_models.py` — the gravitating twins in benchmark units (G = 1,
  radius 1): `fluid_core_2d` (uniform fluid core, N^2 < 0) and
  `aw_core_2d` (the closed-form Adams-Williamson core,
  rho = rho0 (1 - alpha x^2), kappa = -rho^2 g / rho', mass-matched so
  the mantle state is identical; N^2 = 0). Each carries rho, kappa, mu
  and the hydrostatic state (g, p0); `python disc_models.py` checks the
  AW identity, the mass match and the hydrostatic residual.
- `disc_radial.py` — `solve_degree(model, l, method=...)`: the
  theta-reduction of the code's weak forms under a dead surface
  traction, solved by 1-D hp-FEM, with three methods: `welded` (full
  fluid, continuous displacement), `slip` (full fluid, free-slip CMB
  with the B_Sigma and G_Sigma interface forms) and `dahlen` (the
  Eulerian mixed-class system with F1-F3, fluid displacement
  eliminated; at l = 0 it uses the welded full-fluid solve). Degrees
  l = 0 and l >= 2; l = 1 is pending.

The gate is the aw null test: on `aw_core_2d` the three methods must
agree to solver precision (welded exact through the N^2 = 0
relabelling class, Dahlen exact because the reduction is the full
static fluid there); they agree to ~1e-10 at every degree, and every
interface-form sign is pinned by it. The validation also checks the
G -> 0 limit against rung 0 (`disc_reference.py`), strict p-ladder
convergence on `aw_core_2d`, and records the `fluid_core_2d` p-ladder
drift, loosely bounded: on the non-neutral core the welded and slip
solves wander within a ~0.3 % envelope under refinement (a resolution
floor by physics; the Dahlen solve converges), so fixed-resolution
splits between the methods there are not decompositions.

```
cd benchmarks/disc
python disc_radial.py
```
