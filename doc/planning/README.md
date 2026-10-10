# Planning: open issues and future work

This directory holds what the library does **not** yet do or has not yet
settled: defects, unverified results, review items, solver plans,
proposals, parked ideas and future extensions. It is kept separate from
the reference documentation in `doc/`, which describes only what the
library does and why.

Rules of the directory:

- Each item is one section with a short title, a **Status** line
  (open / proposal / parked / decided, and what is decided), a
  self-contained description with the mathematics, measurements and
  arguments that matter, and a **See** line pointing to the reference
  document (by section name) and to the code.
- Measurements are quoted with their settings (model, mesh size `h`,
  order, solver options) so that they can be rerun.
- When an item is resolved it is removed from here, and the reference
  documents in `doc/` are updated to describe the outcome. A decision
  that leaves nothing to do is recorded in the reference docs, not here;
  a decision is kept here only while some part of the item remains open.

## Files

| File | Contents |
|---|---|
| [`open_issues.md`](open_issues.md) | Correctness and verification: known defects, unverified results, open theoretical questions on the formulations, code and build-system defects, benchmark housekeeping and editorial checks. |
| [`solvers.md`](solvers.md) | Solver and performance plans: slip-constraint enforcement (AL and KKT), fluid-gauge preconditioning and the penalty window, near-incompressible and high-contrast preconditioning, stopping tests and warm starts, adaptive time stepping, combined-degree solves, parallel efficiency. |
| [`future_work.md`](future_work.md) | Physics and capability extensions: nonlinear slip and buffer, stratified fluids, nested shells, the two-potential fluid and the essential spectrum, background-state generators, shape derivatives, adjoints and inversion, Earth-model benchmarks, deferred physics, benchmark extensions and figures. |
| [`equilibrium_figures.md`](equilibrium_figures.md) | Formulation of the planned equilibrium-state and hydrostatic-figure solver: the fluid feasibility functional, adjoints, Sobolev gradients, interface-shape parameterisation, implementation steps. |
| [`sea_level_plan.md`](sea_level_plan.md) | The sea-level equation: settled record (WP1–WP6 delivered), the condensed bordered formulation and the WP3 derivation that code comments cite, settled decisions; its open items point back into the three queue files. |
| [`planetmodel_coastline_sizing_plan.md`](planetmodel_coastline_sizing_plan.md) | Settled record of lateral mesh refinement near sea level: planetmodel's Refinement protocol (v1.2.4) and the two AdGIA consumers (benchmark shoreline rings, the real-topography band field). |
