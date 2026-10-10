# Sea-level equation: plan and settled record

*Status: **delivered** (WP1–WP6, October 2026). This file keeps the
formulation, the derivation that code comments cite, the settled
decisions and the work-package record; everything still open lives in
[`open_issues.md`](open_issues.md), [`solvers.md`](solvers.md) and
[`future_work.md`](future_work.md). The reference exposition is
`doc/quasi_static_models.tex` (the mixed class's sea-level section);
the benchmark family is documented in `benchmarks/sea_level/README.md`.*

Formulation of record: Yu, Al-Attar, Syvret & Lloyd (2025, GJI 240,
329–348 = `doc/Elasticity/ggae388.pdf`), §2 — "the paper" below.
`pyslfp` (`~/dev/pyslfp`) is the cross-validation reference and the
source of the analytical-test-state trick and the ice-ng data files.

## The formulation of record

The paper's eq. (36) states the *entire* forward problem — elasticity +
gravitation (2)–(3), rotation (13)–(17), internal variable (25), load
definition (32), sea-level definition (34), mass conservation (33) — as
**one symmetric weak form**. "Building the sea-level iteration into the
elastic solve" is therefore the discretisation of eq. (36) with the
ocean function C frozen. The only genuine nonlinearity is shoreline
migration, C = C(SL) (eq. 29), handled by an outer Picard loop on C
alone; each frozen-C problem is linear and symmetric. Appendix A: fluid
regions leave the form unchanged — with `SetFluid` (Maxwell treatment)
the fluid core needs no special handling at all.

## The condensed system in our variables

The load pairing in eq. (2) is `∫_∂M (u'·∇Φ + φ') σ dS`, and the sea
level (eq. 34) is `SL = SL₀ − (1/g)(u·∇Φ + φ + ψ) + Φ_g/g`; both
classes' `SetSurfaceLoad` already assemble exactly this pairing (in the
referential variables the combination `u·∇Φ + φ` *is* ζ, and the
feedback collapses to a single ζζ boundary mass). The whole frozen-C
problem is the existing system plus

1. the symmetric PSD boundary form `−(ρ_w C/g) ∫ τ' τ dS` with
   `τ = u·∇Φ₀ + φ` (four uu/uφ/φu/φφ trace blocks);
2. a rank-one border for the uniform Φ_g with the mass-conservation row
   `∫ σ dS = 0` (eq. 33) as its transpose;
3. (with rotation) a rank-three ω border assembled directly as linear
   forms from eqs. (15)–(17) — no MacCullagh or degree-2 extraction, so
   the construction is unchanged on aspherical surfaces — with the
   inertia matrix D (16) as data;
4. the data loads (`ρ_i(1−C₀)I₁` and the initial-state terms) on the
   ordinary `SetSurfaceLoad` slot.

Sea level itself is postprocessing (`SL₁ = −(τ+ψ)/g + Φ_g/g`,
`SeaLevelOperator`); no SL unknown lives in the solve. By eq. (36)'s
symmetry the adjoint problem is the same operator with different loads
(eqs. 48–50), so the condensed operator loses nothing for the adjoint
bookkeeping later. The pyslfp-style fixed-point iteration is kept as a
test instrument only: monolithic ≡ fixed point is a standing gate.

## The WP3 derivation (the discrete terms, term by term)

Frozen shoreline (`C = C₀`, first-order exact) and no rotation
(`ψ = 0`). Write `τ(x) := u·∇Φ₀ + φ` for the surface trace combination
(the pairing the class's `SetSurfaceLoad` assembles) and
`w := ρ_w C₀ / g ≥ 0` for the ocean weight. The load (eq. 32) with
eqs. (34)–(35) becomes

    σ = ρ_w C₀ SL₁ + ρ_i (1−C₀) I₁ ,   SL₁ = −τ/g + Φ_g/g ,

with `I₁ = I − I₀` the ice-thickness change (data). Substituting into
the loaded weak form `A(x|x′) + ∫ τ′ σ dS = 0` and adjoining the mass
row `∫ σ dS = 0` (times −1, to make the border symmetric):

    A(x|x′) − ∫ w τ′ τ dS + Φ_g ∫ w τ′ dS = −∫ τ′ σ_data dS ,
    ∫ w τ dS − Φ_g ∫ w dS = ∫ σ_data dS ,

with `σ_data = ρ_i(1−C₀) I₁`. In block form, with `Q(x,x′) = ∫ w τ′ τ`
(symmetric PSD boundary form), `c(x′) = ∫ w τ′ dS` and `m = ∫ w dS`:

    [ A − Q    c  ] [ x  ]   [ −L(σ_data) ]
    [ c^T     −m  ] [ Φ_g ] = [ −F_I      ] ,   F_I = −∫ σ_data dS .

(Sign check: the −Q is the destabilising water feedback — water fills
depressions; smallness `ρ_w/ρ̄` keeps the system solvable.) The border
is rank one and is eliminated by Sherman–Morrison; `SL₁ = −τ/g + Φ_g/g`
is postprocessing, and mass conservation `∫σ dS = 0` holds identically —
the discrete certificate. Per degree `l ≥ 1` on an all-ocean spherical
model the system closes against the class's own measured response:

    SL₁_l = − T_l σ̂_l / (1 − ρ_w T_l) ,

the classic SLE spectral solution — the rung-0 self-consistency gate.
The Φ_g row is inert for `l ≥ 1` patterns and activates with land or
degree-0 content.

## Settled decisions

- **Classes and order:** mixed/Dahlen first (delivered); the
  referential leg (the ζζ collapse) and the Maitra & Al-Attar
  generalised rotational theory are the WP7 extension
  (`future_work.md`).
- **Signs:** the physicist convention throughout (Φ < 0 attractive,
  gravity = −∇Φ, `u·∇Φ₀ = g u_r` at the surface).
- **Surface representation:** H1 fields at the displacement trace order
  for output; boundary integrals and the shoreline update evaluate
  their coefficients pointwise at quadrature — shorelines are data, not
  mesh lines, and never projected into the operator.
- **ψ convention:** the centrifugal/tidal potential enters through its
  interpolant on the potential space *everywhere* — rows, columns,
  cross data, instruments (mixing interpolant and exact evaluations is
  a route-grade input error that near-neutral polar wander amplifies).
- **Moments are data:** the traditional theory takes the equilibrium
  principal moments as observed values; a spherical model's own moments
  degenerate the wander rows (paper §2.4).
- **Shoreline migration:** strictly second order instantaneously
  (Crawford: the load integrand vanishes at the shoreline), finite over
  glacial cycles — so production runs migrate shorelines by default
  (`ShorelineMigration`, Picard on C with inexact per-pass tolerances
  and a sticky contraction guard), with frozen C₀ as the first-order
  control and off-switch. The quadratic-scaling certificate is the
  standing gate.
- **Three-dimensional only:** 2-D elastic solves with gravity stay; a
  2-D ocean is not needed, even for testing — `SetWaterLoad` refuses
  the dimension (which also avoids the 2-D compatibility relocation's
  deliberately asymmetric border pairing).
- **Solvers:** the bordered elimination is the reference (one
  unbordered solve per border column, columns cached while the operator
  stands still); the monolithic bordered MINRES (`SetMonolithicBorder`,
  probed-|Schur| border preconditioner, equilibrated columns,
  true-residual refinement) is the gated option that wins multi-border
  one-shots. Measured (h 0.4/o2/np4): elastic 1.00×, water 1.64×
  (monolithic 2.31×), water+rotation 3.80× (monolithic 2.34×),
  migration 5.31×. Default and promotion questions: `solvers.md`.
- **Ice-ng data flow:** pyslfp's downloader and `IceNG` loader are
  borrowed wholesale (nothing native, raw NetCDF never copied); the
  exchange format is `WriteSurfaceField`'s nodal CSV, read back by
  `ReadSurfaceField`/`IceHistory` and produced by
  `postprocess/ice_ng_to_surface.py`.
- **Mesh refinement at shorelines:** a planetmodel (≥ 1.2.4) Refinement
  field sized by |topography| — shorelines and shallow shelves refine
  together, no coastline isolation (`planetmodel_coastline_sizing_plan.md`).

## The work packages, as delivered

- **WP1** — the surface layer: `SeaLevelOperator` (boundary SubMesh of
  the exterior surface, trace restriction, SL₁ postprocessing, nodal
  CSV export/ingest) and `postprocess/surface_to_netcdf.py`. Tests:
  `TestSeaLevel(+Par)`.
- **WP2** — rotational feedback on prescribed loading:
  `RotationalFeedback` (cached unit responses over the tidal
  machinery), `CentrifugalPotential`/`InertiaMatrix`. Tests:
  `TestRotation(+Par)`; rung R rides the rung-2 benchmark.
- **WP3** — sea level without rotation: `SetWaterLoad` (the −Q blocks
  and the Φ_g border above). Tests: `TestSeaLevelCoupling(+Par)`
  (monolithic ≡ fixed point, the spectral identity, mass certificates,
  land and fluid-core cases). Rung 1 vs pyslfp: ~1 % ocean-RMS at
  h 0.4/o2/lmax 16.
- **WP4** — the composed water–rotation border (`SetRotation`,
  multi-column elimination; ω-row data read off the raw surface-load
  form). Tests: `TestSeaLevelRotation(+Par)`. Rung 2 vs pyslfp:
  rotating fingerprints at the non-rotating grade; wander within the
  near-neutral amplification budget.
- **WP5** — shoreline migration: `ShorelineMigration` (frozen-C₀
  off-switch, relative-SL₁ stop, inexact Picard with the sticky
  contraction guard). Tests: `TestShorelineMigration(+Par)` including
  the quadratic certificate. The resolved rung-3 comparison is
  server-gated (`open_issues.md`).
- **WP6** — viscoelastic composition and the ice history:
  composition holds with no library change (the water blocks ride the
  registered integrators; every stepper solve goes through the bordered
  path) — `TestSeaLevelViscoelastic(+Par)`; `IceHistory` +
  `ice_ng_to_surface.py` + `examples/ice_age_loading.cpp` are the
  end-to-end chain, with `earth_coastlines.msh` as the refined-mesh
  option.

Benchmarks: `benchmarks/sea_level/` (rungs 1–3 vs pyslfp, `--rot`,
`--nonlinear`, `--coast`, `--timings`, `-feedback
border|monolithic|picard`).

## Open items (the queues hold the detail)

- Degree-1 reference-frame convention vs pyslfp: unconfirmed
  (`open_issues.md`).
- Resolved rung-3 (shoreline migration vs pyslfp's nonlinear solver)
  and the resolution ladders: server campaign (`open_issues.md`).
- Border-solver questions — monolithic promotion, refinement economy,
  Picard-feedback mode in the stepper, the loose-tolerance Φ_g
  amplification: `solvers.md`.
- WP7 referential leg + generalised rotational theory, equilibrium
  backgrounds with water (the foundational-caveat fix), ice-age
  production runs: `future_work.md`.
