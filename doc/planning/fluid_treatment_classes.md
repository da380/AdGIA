# Fluid-treatment class structure: making Maxwell the visible default

**Status:** COMPLETE — all four steps implemented 8 Oct (step 4 on
David's instruction the same evening), test gates green in both
builds at every stage, example numbers unchanged. Context:
doc/static_fluid_core.tex ("The Maxwell relaxation method" and "The
status of the formulations").

**As built (8 Oct, second session).** `FluidRegionOperator` now exists at namespace
scope in quasi_static_problem.hpp: it owns the marker, the scaled
coefficients, the (covariant) penalty integrator, the assembled
`eps mu Q`, the unit-epsilon cache and the Rescale fast path, with
`Configure/Clear/SetEpsilon/Assemble/Rescale/ApplyQ/Q/Integrator/
Marker`. The problem base keeps the composition (`a_solve_form_`,
`A_solve_`), the solvers, and the two drivers; its mode flags
collapsed to one slot, `enum class FluidTreatment { None, Penalty,
Maxwell }` (`SetGaugedFluid` selects Penalty, `SetMaxwellFluid`
re-selects Maxwell after it, `ClearGaugedFluid` resets — the
treatment-switching bug class is structurally gone). `GaugePenalty`
moved to namespace scope with a nested alias for the historical
spelling; derived drivers reach the engine's matrix through the
protected `GaugeQ()` (four sites migrated: the referential refine
loop, the mixed gauge-KKT block, its residuals and its refine loop).
Step 3 resolved into this structure: the vacuum extension and the
preconditioner-only mode configure the same engine through the
`SetGaugedFluid` internals, while every public verb
(`SetFluid`, `SetVacuumExtension`, `SetGaugePreconditionerOnly`)
names its own intent. En route, the slip solver-setup audit for the
referential keep-alive bug came back clean: all three slip paths
already pin `prec_stale_ = true` before their preconditioner calls
(reuse deliberately defeated), so the referential path was the only
instance.

## The problem

`LinearQuasiStaticProblemBase` has accumulated three fluid-handling
modes as flat members and mode flags:

1. gauge penalty + Tikhonov refinements (`SetGaugedFluid`,
   `GaugeRefine`, the `gauge_*` members);
2. gauge preconditioner-only + plateau stop
   (`SetGaugePreconditionerOnly`, `PlateauMult`);
3. Maxwell relaxation (`SetMaxwellFluid`, `MaxwellSolve`, the
   `maxwell_*` members) — implemented *as a call into*
   `SetGaugedFluid`.

The reuse is right (one deviatoric fluid operator, read two ways:
gauge-fixing penalty, or per-step effective Maxwell shear — and the
Tikhonov refinement IS the beta→infinity Maxwell step). But the
*shape* of the code says "gauge penalty is the treatment, Maxwell is
a variant riding it", which is backwards from where the method
landed: the relaxation is the default general method; the gauge
penalty is the shared engine and a specialist alternative. A fourth
user of the same machinery (the referential ball-wide vacuum
extension, Harmonic form, which is gauge *data*, not a fluid) adds
to the muddle.

## Step 1 — the front door (DONE, 8 Oct)

Public API on the base class:

```cpp
// the DEFAULT treatment: Maxwell relaxation (the general method)
problem.SetFluid(fluid_marker, mu_scale);                       // or
problem.SetFluid(fluid_marker, mu_scale, MaxwellRelaxationOptions{...});

// the specialist alternative: relabelling-gauge penalty
problem.SetFluid(fluid_marker, mu_scale, GaugePenaltyOptions{...});
```

with a block comment at the top of the fluid-region section stating
the one-engine-two-treatments picture and when each applies
(Maxwell: any stratification, any geometry, any number of regions;
gauge penalty: the cheaper specialist for neutral/near-neutral
models with a validated epsilon window, and the Maxwell step's own
engine). `SetGaugedFluid`/`SetMaxwellFluid` remain as the detailed
spellings (both dispatch virtually, so the referential covariant
default and the slip-class refusal still apply); `SetGaugedFluid`
now clears the Maxwell flag, closing a latent treatment-switching
bug. The flagship example and one test use the `SetFluid` spelling.

## Step 2 — extract the engine (proposed)

A member component, working name `FluidRegionOperator`, owning what
is currently scattered through the base class: the fluid marker, the
mu coefficient and covariant tensor, the penalty integrator/template
form, the assembled `eps mu Q`, the cached unit `Q` and the
rescale-by-sparse-addition path, and `ApplyQ` with essential-row
zeroing. The base class keeps load/solve plumbing and the `A + eps
mu Q` composition; the two drivers (`MaxwellSolve`,
`GaugeRefine`/`PlateauMult`) become clearly two loops over one
engine. Mechanical, behaviour-preserving; the point is legibility
and that the mutual-exclusion `MFEM_VERIFY`s become structural (one
treatment slot) instead of flag checks.

## Step 3 — re-route the non-fluid users (proposed)

The referential ball-wide `SetVacuumExtension` (Harmonic penalty)
and `SetGaugePreconditionerOnly` use the engine explicitly rather
than through `SetGaugedFluid`, so "fluid treatment" and "gauge data
regularisation" stop sharing a public verb.

## Step 4 — later, optional

- Unify `GaugeRefine` as the beta = infinity, fixed-epsilon,
  no-escalation case of the Maxwell driver (today they share the
  operator but not the loop; the refinement's zero-potential-load
  increment solves are an optimisation the unified driver would need
  to keep).
- Retire the old spellings from drivers/benchmarks once the
  love_benchmark campaign has moved over; whether to keep them
  permanently is David's call.
- The slip classes keep their own `SetFluidGauge` (separate fluid
  space); they are laboratory instruments now and are not part of
  this surface.

## Non-goals

No change to the Dahlen (`FluidRegion`, mixed-class constructor)
path, to the viscoelastic layer, or to any weak form. Behaviour of
every existing call is preserved through steps 1–3.
