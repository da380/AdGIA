# Sea-level equation: implementation plan

*Status: draft for review, 9 October 2026.*

## Scope and sources

A sea-level problem class extending the self-gravitating elastic problem
layer, solving the gravitationally self-consistent water-load problem of
Yu, Al-Attar, Syvret & Lloyd (2025, GJI 240, 329–348 =
`doc/Elasticity/ggae388.pdf`), §2 — hereafter "the paper". Rotational
feedback and shoreline migration are staged options. The viscoelastic
version comes for free through `ViscoelasticOperator` in the usual way.
`pyslfp` (`~/dev/pyslfp`) is the cross-validation reference (pseudo-
spectral, Anderson-accelerated load iteration, linear and nonlinear
variants, rotation via MacCullagh + tidal Love numbers) and the source of
the analytical-test-state trick and the ice-ng data files. NetCDF
ingestion is deferred.

## The formulation of record

The paper's eq. (36) already states the *entire* forward problem —
elasticity + gravitation (2)–(3), rotation (13)–(17), internal variable
(25), load definition (32), sea-level definition (34), mass conservation
(33) — as **one symmetric weak form** ("sign choices … made so as to
maximize the symmetry"). So "building the sea-level iteration into the
elastic solve" is not a trick we need to invent: it is the discretisation
of eq. (36) with the ocean function C frozen. The only genuine
nonlinearity is shoreline migration, C = C(SL) (eq. 29), handled by an
outer Picard loop on C alone; each frozen-C problem is linear and
symmetric. Appendix A: fluid regions leave the form of the results
unchanged — with `SetFluid` (Maxwell treatment) the fluid core needs no
special handling at all.

## Which classes, in which order (decided 9 Oct)

**Both formulations get the sea-level machinery; the Dahlen/mixed class
goes first.** It is the conventional formulation that we and most users
will actually run, and most runs will be spherical elastic models where
it is the natural tool. The referential leg follows for generality —
aspherical models, and the route to the **generalised rotational theory**
of Maitra & Al-Attar (2024, `doc/Elasticity/ggae092.pdf`): Yu §2.4 shows
the traditional rotational theory implicitly assumes an isotropic
background inertia tensor, which bites for 3-D models, and the M&A
approach avoids that assumption. The traditional theory is what rungs
0–3 implement; the generalised one is a flagged extension of the
referential leg.

**A foundational caveat, flagged for later (David, 9 Oct):** all sea-
level theory to date rests on a sleight of hand — the applied forces
belong to a rotating model with oceans, but their effect is computed on
a non-rotating model without them. There is no consistent equilibrium
state behind the linearisation: the reference sea level is an
equipotential of nothing. Acceptable for now (it is what the whole
field does), but to be fixed down the line. The fix has a natural home
here: the equilibrium-figures machinery builds self-consistent rotating
backgrounds (extended to carry surface water), and the M&A generalised
rotational theory is the matching perturbation framework — both on the
referential leg.

## The key simplification in our variables

The load pairing in eq. (2) is `∫_∂M (u'·∇Φ + φ') σ dS`, and the sea
level (eq. 34) is `SL = SL₀ − (1/g)(u·∇Φ + φ + ψ) + Φ_g/g`. Both
classes' `SetSurfaceLoad` already assemble exactly this pairing:

- **mixed class**: both rows explicitly — `−σ ∇Φ₀·u'` through
  `VectorBoundaryLFIntegrator` with the existing `grad_phi0_shadow_`,
  and `−σ φ'` through `BoundaryLFIntegrator`
  (`mixed_problem.cpp::SetSurfaceLoad`). Substituting the load law, the
  frozen-C feedback operator is the symmetric four-block boundary form
  `−(ρ_w C/g) ∫ (u'·∇Φ₀ + φ')(u·∇Φ₀ + φ) dS`: a Winkler-like uu block
  weighted by `(∇Φ₀⊗∇Φ₀)/g`, two cross blocks, and a φφ boundary mass.
  SL postprocessing reads both traces: `SL₁ = −(u·∇Φ₀ + φ + ψ)/g +
  Φ_g/g`.
- **referential class**: the combination `u·∇Φ + φ` **is the
  referential potential ζ** — which is why its `SetSurfaceLoad`
  assembles on the ζ row only ("the u-row load … absorbed by the change
  of variables"). The same feedback collapses to a single boundary mass
  `−(ρ_w C/g) ζ ζ'` on the ζζ block, and `SL₁ = −(ζ + ψ)/g + Φ_g/g`.

In either set of variables, the whole frozen-C sea-level problem is the
existing system plus:

1. **a boundary mass term** `−(ρ_w C/g) ∫_∂M ζ ζ' dS` on the ζζ block
   (the ocean-load feedback; one scalar boundary mass matrix weighted by
   the ocean function and surface gravity);
2. **a rank-one border** for the uniform constant Φ_g, with the mass-
   conservation row `∫_∂M σ dS = 0` (eq. 33) as its transpose;
3. **(with rotation) a rank-three border** for ω, assembled directly
   from the deformation fields (David, 9 Oct): for each component of
   ω′, the couplings in eq. (15) — the body term `∫ ρ u·∇ψ′` and the
   surface term `∫ σ ψ′`, with `ψ′ = −(Ω×x)·(ω′×x)` — are LinearForms
   assembled once; the border columns are those vectors, the rows their
   transposes by the symmetry of eq. (17), and the 3×3 diagonal block
   is the inertia matrix D (16). No MacCullagh / degree-2 extraction,
   so the construction is unchanged on aspherical surfaces. The
   centrifugal forcing of the u-row rides the `SetTidalLoad` substrate
   (built 8 Oct; the "rotational feedback door" noted there);
4. **data loads**: the ice term `ρ_i(1−C)I` and the initial-state terms
   `ρ_w C₀ SL₀`, `ρ_i(1−C₀)I₀` on the slot `SetSurfaceLoad` already
   serves.

Sea level itself is then **postprocessing** (formulas above per class)
restricted to the surface. No SL unknown needs to live in the solve.
This is the condensed form; the paper's full form with (SL₁, σ, Φ_g) as
explicit unknowns is equivalent and matters later for the adjoint
bookkeeping — but by eq. (36)'s symmetry the adjoint problem is the same
operator with different loads (eqs. 48–50), so the condensed operator
loses nothing.

Expected solver impact: the added ζζ term is boundary-only, negative
semi-definite and small (ρ_w against bulk densities); physically the
ocean-loaded planet is stable, so no definiteness surprise, and the
MINRES/projected stack should be essentially unaffected — this is the
quantitative form of "little effect on the overall convergence". The
rigid-motion null space is untouched (paper eq. 21); SL is rigid-motion
invariant (remark under eq. 34), which becomes a discrete certificate.

Known-correct alternative kept as an *instrument*: the pyslfp-style
fixed-point iteration (assumed SL → load → solve → new SL), a few lines
on top of `SetSurfaceLoad`. Each iteration costs a full elastic solve, so
it is not the production path, but monolithic == fixed-point at
convergence is a strong independent gate, and it mirrors the reference
implementation one-to-one.

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
(symmetric PSD boundary form — the four uu/uφ/φu/φφ trace blocks),
`c(x′) = ∫ w τ′ dS` and `m = ∫ w dS`:

    [ A − Q    c  ] [ x  ]   [ −L(σ_data) ]
    [ c^T     −m  ] [ Φ_g ] = [ −F_I      ] ,   F_I = −∫ σ_data dS .

(Sign check: the −Q is the destabilising water feedback — water fills
depressions; smallness `ρ_w/ρ̄` keeps the system solvable.) The border
is rank one and is eliminated by Sherman–Morrison: `Φ_g = (cᵀx + F_I)/m`
and the x-system becomes `A − Q + (1/m) c cᵀ` — one extra solve
`(A−Q)⁻¹c` per operator assembly (load-independent, cached), then a
rank-one combination per load. `SL₁ = −τ/g + Φ_g/g` is postprocessing
as before, and mass conservation `∫σ dS = 0` holds identically — the
discrete certificate.

Per degree `l ≥ 1` on an all-ocean spherical model the system closes:
with `T_l` the τ/g-amplitude response to a unit surface-mass load of
degree l (measured by a plain `SetSurfaceLoad` solve of the same
class — no external reference), the monolithic answer must satisfy

    SL₁_l = − T_l σ̂_l / (1 − ρ_w T_l) ,

the classic SLE spectral solution — the rung-0 self-consistency gate.
The Φ_g row is inert for `l ≥ 1` patterns and activates with land or
degree-0 content.

## Class design (sketch for review)

`SeaLevelProblem` wrapping/extending
`LinearQuasiStaticReferentialSelfGravitatingProblem` (production class:
Maxwell fluid treatment, tides, gauge machinery all inherited).

- **Surface fields** live on a boundary `SubMesh` of the exterior
  surface (MFEM `CreateFromBoundary` + transfer maps; our submesh layer
  is domain-submesh-tested — verifying the boundary leg is an explicit
  early work item). Fields: topography/initial sea level SL₀, ice
  thickness I(t), ocean function C, and the output SL₁. Space: scalar on
  the surface mesh at the displacement order (H1 trace; final choice at
  review).
- **Shorelines are coefficient data, not mesh lines**: C enters only as
  a weight in boundary integrals, evaluated at quadrature points from
  the surface fields. The mesh-conforms-to-material-interfaces rule is
  about volume material discontinuities; a shoreline-crossing boundary
  element just integrates a discontinuous coefficient (quadrature-
  accuracy caveat to measure; shoreline-aware surface meshing is a
  possible later refinement, not a prerequisite).
- **API sketch**: `SetTopography(...)` / `SetInitialState(SL0, I0)`,
  `SetIceThickness(provider over time)`, `EnableRotationalFeedback(Ω,
  inertia)`, `EnableShorelineMigration(opts)` (Picard tolerance/cap),
  overridden assembly adding the three items above, `SeaLevelChange()`
  returning the surface-submesh field, plus conservation and invariance
  diagnostics (`∫σ dS` to round-off; rigid-shift invariance).
- **Background data**: surface gravity g and ∇Φ from the class's
  hydrostatic background (kept as Coefficients — aspherical surfaces
  make g non-constant on ∂M).
- **Shoreline migration is strictly second order in the instantaneous
  perturbation** (Crawford — structurally: the load integrand
  `ρ_w SL − ρ_i I` *vanishes at the shoreline*, cf. the δC line
  distribution of paper eq. 51), so frozen C = C₀ is first-order exact
  and is the natural validation configuration and off-switch. **But it
  is required machinery for ice-age work (David, 9 Oct):** over a
  glacial cycle the shoreline excursions are finite, so production
  runs migrate shorelines by default, with the frozen-C option kept
  for linearised studies and as the control. The theorem supplies the
  implementation's certificate — `|SL(migrating) − SL(C₀)|` must
  scale quadratically with load amplitude — and at first order the δC
  shoreline terms drop from the adjoints.
- **Inexact Picard** (when the loop does run; David, 9 Oct: never
  re-solve full linear problems per C-update): warm-started Krylov
  across iterates (the C-update is a small boundary-only operator
  perturbation), the Maxwell layer's increment-validated inexact
  tolerance schedule reused (loose early solves clamped against the
  measured shoreline increment, validation against warm-start fake
  convergence, full-tolerance final polish), preconditioner frozen
  across the loop via the existing reuse machinery, and boundary-only
  reassembly of the C-weighted forms sparse-added into the operator
  (the `RescaleGaugeOperator` pattern, with reassembly in place of
  rescaling since C changes support, not scale). Anderson on the C
  iterates if ever needed (pyslfp converges in a handful of steps).
- **Viscoelastic composition**: `ViscoelasticOperator` drives the class
  through the virtual `Solve()`/`SetRelaxationWeights` path as usual;
  the per-step external load update includes I(t) through the existing
  time-dependent-coefficient registration. One check: the effective-
  modulus reassembly must preserve the sea-level boundary terms (they
  are modulus-independent, so this is bookkeeping, not maths).

## Testing ladder (laptop-sized by design)

The pyslfp trick throughout: smooth analytical states (super-Gaussian
continents/caps, `pyslfp/ice/analytical_ice.py`,
`EarthState.for_testing`) sampled as exact Coefficients — no aliasing,
honest at low resolution. Development on the laptop (2-D + coarse 3-D);
resolution ladders and ice-ng runs go to the server.

- **Rung 0 — all-ocean planet (C ≡ 1), elastic, no rotation.** The SLE
  closes per degree in terms of *our own measured load Love numbers*
  (h′, k′) plus the uniform Φ_g term: a self-consistency gate against
  the love_benchmark numbers needing no external reference. Also the
  monolithic == fixed-point identity, and the conservation/invariance
  certificates. A 2-D disc analogue (ocean = full boundary circle) gives
  the house-style cheap first leg with closed forms.
- **Rung 1 — fixed shorelines, analytical continents.** Elastic sea-
  level fingerprints vs `pyslfp.LinearSeaLevelEquation` on a matched
  spherically-symmetric model (the Love-number benchmark already matches
  models to pyslfp — same wiring). Partial-ocean 2-D arcs as the cheap
  leg (no pyslfp there; invariants + fixed-point identity instead).
- **Rung R — rotational feedback on *prescribed* loading (no sea
  level; built and gated BEFORE the sea-level rungs, see work
  packages).** A simple surface load with the ω border active, against
  pyslfp, where rotational feedback is implicit and usable for
  benchmarking in isolation. Degree-2 order-1 response; the inertia-
  isotropy caveat (paper §2.4) noted where the background model makes
  it relevant.
- **Rung 2 — sea level and rotation composed.** Rotating fingerprints
  vs pyslfp; the eq.-(40)-style invariance checks.
- **Rung 3 — shoreline migration.** Vs pyslfp's nonlinear solver on
  analytical states; Picard iteration counts recorded.
- **Then viscoelastic**: composition smoke + conservation over a loading
  history; first ice-ng demo (data via pyslfp's local files, exported to
  our sampling by a small Python step — no C++ NetCDF dependency yet).

## Work packages (chunked as usual; review+commit per chunk)

Three separately-testable couplings, built in order of complexity
(David, 9 Oct): prescribed loading **with** rotation, sea level
**without** rotation, then the two composed — each with its own gate.

- **WP0** — this note.
- **WP1** — boundary SubMesh plumbing + trace restriction + SL
  postprocessing on *existing* solves (no feedback yet); diagnostics.
  Class-agnostic. Both legs, serial and parallel, same unit of work.
- **WP2** — **rotational feedback on prescribed loading** (mixed
  class): the ω border by direct linear-form assembly + D block +
  centrifugal forcing via the tidal substrate; establishes the
  bordered-assembly plumbing; rung R vs pyslfp.
- **WP3** — **sea level without rotation** (mixed class): the
  four-block boundary pairing + Φ_g border, frozen C; the fixed-point
  instrument; rung 0 (all-ocean, self-consistent vs our Love numbers)
  and rung 1 (analytical continents vs pyslfp linear fingerprints).
- **WP4** — **sea level with rotation**: compose the two borders;
  rung 2 vs pyslfp rotating fingerprints.
- **WP5** — shoreline migration (Picard on C) + rung 3 — *required
  for ice-age production (switchable; frozen C₀ = the off-switch and
  first-order control)*; includes the quadratic-scaling certificate
  and the inexact-iteration machinery above.
- **WP6** — viscoelastic composition + ice-history demo.
- **WP7** — the **referential leg**: port the boundary terms (the ζζ
  collapse), gate against the mixed results; the M&A generalised
  rotational theory hangs off this leg as the future extension.

## Open questions for David

1. ~~Base class~~ **ANSWERED 9 Oct: both, mixed/Dahlen first** (see
   "Which classes, in which order").
2. ~~Sign conventions~~ **ANSWERED 9 Oct: the physicist convention
   throughout** (Φ < 0 attractive, gravity = −∇Φ, so ∇Φ₀ points outward
   and `u·∇Φ₀ = g u_r` at the surface), consistent across David's
   papers and codes — double-check welcomed. Staging: a half-page
   term-by-term derivation (paper eqs. 32/34 onto the class variables)
   opens WP2 for review, and the rung-0 all-ocean gate enforces the
   signs numerically regardless.
3. ~~SL surface space~~ **ANSWERED 9 Oct: H1 at the displacement trace
   order** — sea level is physically continuous. The split stands: H1
   fields for output/comparison (SL₁, SL₀, I, C), while the boundary
   integrals and the Picard shoreline update evaluate their
   coefficients directly at quadrature points from the traces and
   data, so no projection error enters the operator or the C
   iteration.
4. ~~Rotation ordering~~ **ANSWERED 9 Oct: three stages in order of
   complexity** — prescribed loading with rotational feedback (gated
   alone vs pyslfp, where it is implicit and benchmarkable), sea level
   without rotation, sea level with rotation. See the work packages.
5. ~~Class shape~~ **ANSWERED 9 Oct: inheritance by composition** — a
   `SeaLevelOperator` engine component (surface submesh, H1 fields,
   C-weighted boundary forms, Φ_g/ω borders) behind a `SetSeaLevel`-
   style front door on the problem layer, with one small per-class
   virtual hook for how the pairing lands in that class's variables
   (mixed: the four-block (u, φ) form; referential: the ζζ mass) —
   the SetFluid/FluidRegionOperator pattern. Guiding principles
   (David): a passably simple presented API — few classes, options
   behind them — and maximal reuse of existing code and methods
   (`SetSurfaceLoad` slots, the tidal substrate, submesh layer,
   bordered/null-space machinery, preconditioner reuse) over new
   structure.

   *Refined after David's Lego musing (9 Oct, post-read):* the
   bolt-on-by-multiple-inheritance picture (mixin stacking,
   `SeaLevel<Rotation<Mixed>>`) was considered and set aside for
   David's own reason — each layer would see only a narrow hook
   interface, losing the class-level global view needed to build ONE
   operator, ONE preconditioner and one bordered elimination over all
   active capabilities (plus MI type explosion and friction with the
   virtual `Solve()`/`SetRelaxationWeights` hooks the viscoelastic and
   Maxwell layers use). The decided structure keeps the Lego but at
   runtime: **inheritance for formulation** (mixed vs referential),
   **components for capabilities** (fluid treatment, sea level,
   rotation, tides), with assembly/solver setup owned by the problem
   class as the one point of global visibility — and a **thin
   convenience subclass** (`SeaLevelProblem`: the mixed class with the
   sea-level and rotation doors pre-wired and a tidy constructor) so
   the presented API still reads as named bricks.
