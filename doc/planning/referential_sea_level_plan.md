# The referential sea-level leg (WP7): design

*Status: **delivered** (11 Oct 2026) — phases A and B implemented and
gated through G8d; the per-WP entries below and the Verification table
of [`../generalised_rotation.md`](../generalised_rotation.md) carry
the record (the fluid-core frame choice is settled there: the
Tisserand gauge pins the mantle frame). Still open: the Mitrovica &
Wahr viscoelastic numerical check. This is WP7 of
[`sea_level_plan.md`](sea_level_plan.md): the sea-level boundary terms
on the referential class, and the Maitra & Al-Attar generalised
rotational theory that hangs off it. Formulation of record as before:
Yu, Al-Attar, Syvret & Lloyd (2025 = `doc/Elasticity/ggae388.pdf`,
"the paper"); the generalised rotation is Maitra & Al-Attar (2024, GJI
237, 1301–1338 = `doc/Elasticity/ggae092.pdf`, "M&A"). The reference
exposition to extend is `doc/quasi_static_models.tex` (the referential
class's section and the mixed class's sea-level section).*

## Why this leg exists

Two reasons, in order of weight:

1. **Generality.** The mixed/Dahlen sea level (WP1–WP6) is production
   for spherical elastic models. The referential class is where the
   mapped (aspherical) machinery lives, and where the sea-level terms
   take their simplest form — so aspherical GIA with self-consistent
   sea level belongs here.
2. **The generalised rotational theory.** The paper's §2.4 shows the
   traditional rotational-feedback theory (our WP4 border) implicitly
   assumes the *elastic model's* inertia tensor is isotropic: eq. (20),
   `∫ (a + b×x)·ρ∇ψ' dV = 0` for every test angular velocity ω', is
   required for the ω of eq. (15) to be well defined against the rigid
   non-uniqueness of u. Spherical models satisfy it; 3-D models do
   not, and the error has no small parameter. M&A's Tisserand-frame
   formulation conserves angular momentum with the *exact*
   deformation-dependent inertia and needs no such assumption. That
   theory is referential by construction (M&A §3.7–3.8), which is why
   it hangs off this leg and not off the mixed class.

## The ζζ collapse

In the referential variables the potential unknown is
`ζ¹ = φ + u·∇Φ₀` (quasi_static_models.tex, the referential section) —
exactly the combination `τ` that the mixed class's water feedback
`−(ρ_w C/g) ∫ τ'τ dS` had to assemble from four uu/uφ/φu/φφ trace
blocks (the WP3 derivation). The referential water feedback is
therefore **a single boundary mass on the ζζ block**:

    −(ρ_w C̃ / g̃) ∫_∂B ζ' ζ  J_Σ dS,

and the class's `SetSurfaceLoad` being a ζ-row-only load (its header
says why) is the same fact seen from the load side. The conventions to
pin in the half-page derivation that opens the work (the WP3 pattern):

- **Area element.** The integral above is the physical-surface integral
  pulled back: `J_Σ = J_e |F_e^{-T} n̂|` is the equilibrium mapping's
  surface Jacobian. `SetSurfaceLoad`'s σ is *per referential area*, so
  the data loads need no factor; the feedback operator does. For every
  current model the equilibrium mapping is the identity at the surface
  (`J_Σ = 1`) and the term is literally the mixed one; the factor is
  carried so that the mapped gates below are honest.
- **Surface gravity.** `g̃ = |γ|` with `γ = −F_e^{-T}∇ζ⁰` the
  referential gravitational acceleration (M&A eq. around their (205));
  at an identity-at-surface mapping this is the mixed class's
  `|∇Φ₀|`.
- **Ocean function.** C̃ = C∘φ_e, pointwise as in the mixed class
  (never projected); with the surface mapping the identity, C̃ = C.
- **Sign.** Physicist convention throughout as decided for WP2; the
  rung-0 spectral-identity gate enforces it (below).

The condensed referential system is then the existing (u, ζ) system
plus the four items of the mixed record
(`sea_level_plan.md`, "The condensed system in our variables") with
these replacements:

1. the ζζ boundary mass above (one block, not four);
2. the rank-one Φ_g border with the mass-conservation row
   `∫ σ dS = 0` (σ per referential area, so the row is referential);
3. the rank-three ω border — phase A traditional, phase B generalised
   (below); the column forcing enters through the class's
   `SetTidalLoad` door, whose header already anticipates exactly this
   (`∇ψ_k ∘ φ_e` as plain analytic data — the centrifugal r² part is
   not an interior harmonic);
4. the data loads (`ρ_i(1−C₀)I₁`, initial-state terms) on the ordinary
   `SetSurfaceLoad` slot.

Sea level stays postprocessing: `SL₁ = −(ζ¹ + ψ)/g + Φ_g/g` — the
referential `SeaLevelOperator` entry takes ζ¹ directly instead of
rebuilding `u·∇Φ₀ + φ` from pieces.

## Scope

- **Class:** the welded referential class
  (`LinearQuasiStaticReferentialSelfGravitatingProblem`). Fluid cores
  through the inherited `SetFluid` (Maxwell/gauge-penalty) routes. The
  slipping classes are *out*: their `SetTidalLoad` is refused today,
  and water on a slipping outer surface is not a current need.
- **Dimension:** 3-D only, as ruled for the mixed class.
- **Both legs** (serial + parallel) in the same unit of work, as
  always; the ψ-interpolant convention from the WP4 bug hunt carries
  over verbatim (interpolant everywhere: rows, columns, instruments).

## Phases and work packages

### Phase A — water, Φ_g and the traditional rotation border

**Status: delivered 11 Oct 2026** (all four WPs; gates G1–G7b in the
table below measured at their stated grades).

- **WP7.1** The derivation half-page (conventions above) into
  `quasi_static_models.tex`; then `SetWaterLoad(ocean_weight,
  sigma_data, marker)` on the referential class: ζζ boundary mass on
  the parent-surface marker summed onto the ζ block (the mixed class's
  `SumOperator` + solver re-setup pattern), Φ_g border by
  Sherman–Morrison at the `SolveLinearSystem` level.
- **WP7.2** `SetRotation(Omega, moments)`: the traditional eq-(15) row
  (user-supplied principal moments, direct linear-form assembly as in
  WP2/WP4) with the composed multi-column elimination. Valid on
  spherical models — which is where the mixed-class gate lives — and
  explicitly *labelled* as requiring the paper's eq. (20) otherwise.
- **WP7.3** Postprocessing + orchestration: the referential
  `SeaLevelChangeFrom(ζ¹, ψ, Φ_g)` overload; `ShorelineMigration`
  generalised to drive either class (`RefreshWaterLoad()` becomes the
  shared hook); the fingerprint benchmark and example gain
  `-method referential`.
- **WP7.4** The mapped gates (the family's reason to exist): the
  solver-level change-of-variables identity extended to a water solve —
  relabelled sphere with water ≡ unrelabelled, at the strict
  (~1e-6) grade of the welded identity. The covariance lesson from the
  gauge penalty applies: the surface mass must be assembled with the
  mapped integrators and the `J_Σ` factor, not with a relabelled
  coefficient on the unmapped form.

### Phase B — the generalised (M&A) rotational theory

- **WP7.5** A derivation note first (the WP3-derivation pattern; this
  one is genuinely new) — **written, and since graduated to the
  reference docs (13 Oct 2026):
  [`../generalised_rotation.md`](../generalised_rotation.md)**.
  Headlines: the traditional ω's rigid-gauge shift in closed form,
  δω = D⁻¹[ad_b, I]Ω (zero iff the model inertia is isotropic — the
  quantitative certificate for the G8 demonstration); the linearised
  Tisserand condition ∫ρ x×u dV = 0 is *exactly* the existing
  mass-weighted gauge (`SetMassWeightedGauge`, F4), so design
  question 4 resolves DRY — no new constraint machinery, the gauge
  supplies the representative and the generalised ω equation is a
  3-column border on the shared engine; the input-reference-inertia
  option keeps uniqueness (it comes from the gauge, not the data) and
  costs only the conservation certificates, at O(|I_obs − I_model|) —
  it strictly dominates the traditional theory.

  **The inertia-data question (David, 11 Oct).** In the traditional
  theory the elastic calculation is done in a *non-rotating* (often
  spherical) model, and the inertia tensor in the ω equation is not
  that model's but observed values in some form — which is exactly our
  WP2/WP4 door (user-supplied principal moments; a spherical model's
  own moments degenerate D). In the M&A theory the inertia tensor
  really is the *model's*: self-consistent, which is its virtue — but
  applied to a simplified model it inherits that model's degenerate
  rotational response, so the generalised door likely also wants an
  option to input a reference inertia tensor. That option raises a
  genuine compatibility question the note must settle: an imposed
  inertia inconsistent with the model's density breaks the exact
  conservation structure the theory is built on, so which invariances
  (and which of phase B's certificates) survive under an input
  reference inertia must be derived, not assumed. The note should also
  digest the Mitrovica & Wahr treatment of the *viscoelastic* rotation
  problem (the background figure/fossil bulge in the
  rotational-stability feedback; their review is
  `doc/Elasticity/annurev-earth-040610-133404.pdf`) — read, per
  David's assessment, as valuable plastering of the traditional theory
  rather than anything fundamental: with the reference earth model's
  structure properly known it would not be needed. The long-run
  resolution is exactly that knowledge — the equilibrium-figures work,
  suitably extended: a rotating equilibrium background carries a
  realistic inertia tensor (and figure) of its own, and the M&A theory
  then applies self-consistently with nothing imposed
  (`future_work.md`, "Equilibrium backgrounds with surface water").
- **WP7.6** Implementation + gates: on spherical models the
  generalised route must reproduce the traditional border (that *is*
  the regime where the traditional theory is exact); on a deliberately
  triaxial model (ellipsoidal density through the mapped machinery)
  the traditional row's eq-(20) violation is demonstrated (ω shifts
  under a rigid-motion re-gauge of u) while the generalised ω is
  invariant — an invariance certificate, no external reference needed.
  **Status: delivered through G8d + G8c phase 1** (11 Oct 2026; the
  note's gates table has the measurements): the generalised door
  (`SetRotation(Omega)` = model inertia + Tisserand gauge, the
  input-inertia option = `SetRotation(Omega, moments)` +
  `SetTisserandGauge(true)`), `TestGeneralisedRotation` +
  `GeneralisedRotationDoor` + `ConservationCertificate` (+ the
  `water+generalised` parallel-twin case), and the demonstration
  example `examples/generalised_rotation.cpp` (decision 3's
  deliverable). The fluid-core frame choice is settled and gated
  (`../generalised_rotation.md`: the Tisserand gauge pins the mantle
  frame); open: the Mitrovica & Wahr viscoelastic numerical check.

## Gates (each phase separately gated, the house ladder)

| gate | statement | grade |
|---|---|---|
| G1 | rung-0 spectral identity `s = −T/(1+ρ_w T)` per degree, referential water solve on the homogeneous sphere | route (~1e-6 at o2) |
| G2 | referential ≡ mixed: same spherical model, water (+ fluid core via SetFluid), fields and Φ_g | route grade |
| G3 | monolithic ≡ fixed point (the `IteredWaterLoad` instrument re-pointed) | standing |
| G4 | mass certificates: assembled row exact; Φ_g inert for zero-mean load | exact / route |
| G5 | fingerprint benchmark `-method referential` vs pyslfp | ~1 % at toy res (mixed's grade) |
| G6 | rotation A: SetRotation ≡ mixed on sphere; composed ≡ joint fixed point | route grade |
| G7 | mapped identity with water: relabelled ≡ unrelabelled | strict ~1e-6 |
| G7b | water on a genuinely aspherical solid body (independently meshed, the `aspherical_reference` pattern): certificates (mass row, Φ_g, monolithic ≡ fixed point) with no spherical reference | certificate grade |
| G8 | rotation B: generalised ≡ traditional on sphere; rigid-re-gauge invariance on the triaxial model (the traditional ω's shift is itself the headline demonstration, decision 3) | route / certificate |

## Design decisions (reviewed 11 Oct 2026)

1. **Border engine extraction — decided: extract.** The composed
   multi-column elimination (columns, dense block, data rows,
   version-keyed caching) is pulled out of the mixed class's
   `SolveLinearSystem` into a shared component (the
   SeaLevelOperator/RotationalFeedback pattern) and both classes drive
   it; the WP4 bug-hunt lessons stay encoded once. The monolithic
   MINRES path can follow later or stay mixed-only.

2. **Phase A rotation — decided: build as the gate instrument,**
   decide whether to expose it as a user-facing door when phase B
   lands.

3. **Mapped scope — decided: a genuinely aspherical body is in the
   gates,** built up in complexity from the relabelled-sphere identity;
   an entirely solid simple aspherical body suffices. Moreover the
   numerical demonstration of the traditional theory's failure is
   promoted to a *deliverable* in its own right, not only a gate
   (G8): the existence argument is mathematically simple but
   practically unknown in geophysics, so showing ω shift under a
   rigid re-gauge on a concrete model makes the point as nothing else
   does — a quotable figure/table, example- or benchmark-grade.

4. **Phase B ownership — open by design,** settled in the WP7.5 note;
   preference for the DRY outcome (one mechanism) if the mathematics
   permits.

5. **Fluid-core inertia — decided:** within the referential
   formulation the fluid has actual displacements and they are used
   as-is in the inertia-tensor terms (the solid formula stands on the
   Maxwell route). For the gauged treatment, the inertia tensor is
   invariant under the fluid relabelling group, as it must be — which
   doubles as a phase-B certificate: the computed inertia terms must
   not move when the gauge representative does. Slip-route caveat
   recorded for the future.

**See:** `sea_level_plan.md` (the condensed system; the WP3
derivation); `include/AdGIA/referential_problem.hpp`
(`SetSurfaceLoad`, `SetTidalLoad` headers); `doc/Elasticity/ggae388.pdf`
§2.3–2.4; `doc/Elasticity/ggae092.pdf` §3.7–3.8;
`doc/planning/future_work.md`, "Sea level".
