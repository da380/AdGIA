# The generalised rotational theory

*The Maitra & Al-Attar (2024) rotational theory as the library
implements it: why the traditional rotational-feedback border is
gauge-ambiguous off spherical models, the Tisserand-frame resolution,
its secular reduction onto the existing bordered machinery, the
input-reference-inertia option, and the fluid-core frame choice.
Implemented on the mixed and referential classes —
`SetRotation(Omega)` is the self-consistent door (model inertia +
Tisserand gauge), `SetRotation(Omega, moments)` with
`SetTisserandGauge(true)` the input-inertia one; the demonstration
example is `examples/generalised_rotation.cpp`, and the measurements
are under "Verification" below.
Sources: Yu, Al-Attar, Syvret & Lloyd (2025, "the paper",
`doc/Elasticity/ggae388.pdf`) §2.3–2.4 for the traditional theory and
its diagnosis; Maitra & Al-Attar (2024, "M&A",
`doc/Elasticity/ggae092.pdf`) §2.4–2.5, §3.7–3.8 for the resolution;
Mitrovica & Wahr (review,
`doc/Elasticity/annurev-earth-040610-133404.pdf`) for the viscoelastic
plastering this supersedes.*

## The traditional theory, and its two defects

The traditional rotational feedback (the library's traditional
rotation border, `SetRotation(Omega, moments)`) solves the
elastostatic loading problem in a **non-rotating** model for (u, φ)
given the load σ and a centrifugal potential perturbation
ψ(ω), and determines ω from the paper's eq. (15),

    ω'·D·ω + ∫_M ρ u·∇ψ' dV + ∫_∂M σ ψ' dS = 0   for all ω',

with ψ' = −(Ω×x)·(ω'×x) and D = diag(C₃−C₁, C₃−C₂, −C₃) built from
**observed** principal moments. Two structural defects:

1. **The two-model split.** The elastic calculation lives in one model
   (often spherical); D comes from observations, because the model's
   own moments would degenerate it (a spherical model has C₃−C₁ = 0
   and no finite polar-wander response). The theory is internally
   inconsistent by construction, and the viscoelastic patches
   (fossil bulge, rotational stability — the Mitrovica & Wahr review)
   exist to manage that inconsistency.

2. **Rigid-gauge sensitivity.** u is defined only modulo linearised
   rigid motions, and the ω of eq. (15) is invariant under
   u → u + a + b×x only when the paper's eq. (20) holds — which is
   equivalent to the *elastic model's* inertia tensor being isotropic
   (§2.4). The failure is exactly computable. Under u → u + b×x the
   ω-row functional shifts by

       Δ(ω') = ∫ ρ (b×x)·∇ψ' dV = −ω'·[ J(b×Ω) − b×(JΩ) ],

   with J = ∫ρ x⊗x the model's second-moment tensor (expand
   ψ' = −(Ω·ω')|x|² + (Ω·x)(ω'·x), take the gradient, use the triple
   product twice). Writing I = tr(J)1 − J for the inertia tensor this
   is the commutator action, so the traditional ω shifts by

       δω = D⁻¹ [ad_b, I] Ω,     [ad_b, I]Ω = b×(IΩ) − I(b×Ω),

   which vanishes for every b **iff I ∝ 1**. On a spherical model the
   gauge freedom is harmless; on a 3-D model it is an O(1) ambiguity
   with no small parameter — and the formula above *predicts* the
   shift from the model's inertia tensor alone, which is the
   quantitative certificate of the triaxial demonstration (G8b under
   "Verification"; `examples/generalised_rotation.cpp`).

## The M&A resolution

M&A decompose the motion **exactly** into a Euclidean motion (the
frame: a translation and a rotation R(t)) and an internal deformation,
and make the decomposition unique by the Tisserand condition: the
internal motion carries no net linear or angular momentum (their
eqs. 88, 100). The rotation is then a *dynamical variable*, not a
gauge direction: the spin angular momentum takes the rigid-body form
h = IΩ with the exact, deformation-dependent inertia (their eq. 108),
and conservation of h is what determines the rotational response — no
isotropy assumption anywhere, and the inertia tensor really is the
model's.

Linearised about a steadily rotating equilibrium (M&A §3.8), the
pieces that matter for quasi-statics are:

- **The Tisserand gauge-fixing** (their eqs. 220b, 222): the
  perturbation u carries zero relative angular momentum,

      ∫_B ρ x × u dV = 0

  (three linear conditions; the linear-momentum partner for the
  translations). This *spends* the rigid non-uniqueness — under an
  attempted re-gauge u → u + b×x the condition re-absorbs b into the
  frame angle θ⁽¹⁾, and the pair (u, ω) is unique. The eq.-(20)
  solvability question never arises: the three rigid-rotation kernel
  directions have become the three frame unknowns, matched by the
  three constraint rows. The system is square by construction.

- **The linearised Euler equation** (their eq. 224):

      I⁽⁰⁾ ω̇ + ad_Ω (I⁽⁰⁾ − λ) ω + (∂t + ad_Ω) I⁽¹⁾ Ω = 0,

  with I⁽⁰⁾ the **model's own** equilibrium inertia and I⁽¹⁾(u, σ) the
  exact inertia perturbation (the load's direct contribution plus the
  deformation's). The quasi-static (secular) reading for GIA drops the
  wobble dynamics (ω̇ and the Chandler-band terms) and is derived
  under "The secular reduction" below; the full eq. (224)
  is there when the wobble is ever wanted. In fluid regions the
  inertia integrals are FRL-invariant — as they must be, since the
  relabelling is unobservable — so the gauged fluid treatments
  evaluate them on any representative, and the referential class uses
  its actual fluid displacements directly. For
  Dahlen-style treatments the paper's appendix gives the Eulerian
  fluid-region formula.

Why the two defects disappear: (1) one model, exact inertia — nothing
observed is imposed, so nothing is inconsistent, and the viscoelastic
plastering is unnecessary *provided the model's figure is realistic*
(which is the real content of the inertia-data question below);
(2) the rigid ambiguity is pinned physically by the Tisserand
condition before ω is ever computed, so ω is manifestly re-gauge
invariant — the G8 invariance certificate.

## The secular reduction, worked

Two identities pin the quasi-static form of eq. (224) completely.

**The δI identity.** For ψ' = −(Ω×x)·(ω'×x), expanding the gradient
and using the triple product twice,

    ∫ ρ u·∇ψ' dV = −ω'·δI(u)·Ω,
    δI(u)_ij = ∫ ρ (2(x·u)δ_ij − x_i u_j − x_j u_i) dV,

and likewise ∫σψ' dS = −ω'·δI(σ)·Ω with
δI(σ)_ij = ∫σ(|x|²δ_ij − x_i x_j) dS: δI(u) + δI(σ) is exactly the
linearised inertia perturbation I⁽¹⁾. The traditional row (paper
eq. 15) therefore reads **D·ω = I⁽¹⁾(u,σ)·Ω** — the ω-row data the
border machinery assembles through the ψ pairing *is* the inertia
perturbation, no separate assembly needed.

**The secular limit of eq. (224).** Take Ω = Ω e₃ along a principal
axis of I⁽⁰⁾ = diag(C₁, C₂, C₃) (the equilibrium Euler equation
requires I⁽⁰⁾Ω ∥ Ω, and λ = C₃ is the multiplier). Dropping ω̇ and
∂ₜI⁽¹⁾, eq. (224) becomes
Ω×[(I⁽⁰⁾ − C₃·1)ω] + Ω×[I⁽¹⁾Ω] = 0 for the equatorial components,
and the axial component integrates to conservation. Componentwise:

    (C₃ − C₁) ω₁ = Ω I⁽¹⁾₁₃,
    (C₃ − C₂) ω₂ = Ω I⁽¹⁾₂₃,
    −C₃ ω₃      = Ω I⁽¹⁾₃₃.

**This is exactly the paper's eq. (15) with D = diag(C₃−C₁, C₃−C₂,
−C₃)** — the (C₃−C₁) wander denominators come from the frame term
ad_Ω(I⁽⁰⁾−λ), not from bare conservation (naive δ(IΩ) = 0 would give
C₁ in place of C₃−C₁ and is *not* the right secular statement).

**The consequence for the implementation.** The generalised
quasi-static door is the traditional border machinery *verbatim*,
with exactly three deltas:

1. **D is built from the model's own principal moments** (assembled
   I⁽⁰⁾, referentially ∫ρ̃(|φ_e|²δ − φ_e⊗φ_e) dV), or from input
   reference moments (the input-inertia option) — same
   `InertiaMatrix` structure either way;
2. **u is pinned by the Tisserand gauge** (the mass-weighted gauge,
   below), which makes I⁽¹⁾(u)·Ω — and hence ω — rigid-unambiguous on
   any model;
3. **Ω must lie along a principal axis of I⁽⁰⁾** (an equilibrium
   consistency condition the door verifies; off-principal equilibria
   are not steady and are out of scope).

Gate G8a is thereby trivialised (input moments + the gauge reproduce
the traditional border identically on models where eq. 20 holds), and
the substantive gates are the gauge-invariance demonstration (G8b)
and the certificates (G8c, G8d), all under "Verification" below.

## Reduction to our variables

The key structural fact: **the discrete Tisserand condition already
exists in the library.** `NullSpaceProjector::SetGauge` with the
ρ-weighted vector mass matrix (the `SetMassWeightedGauge` door, F4)
chooses the rigid-mode representative with zero net momentum *and
angular momentum* — exactly the linearised Tisserand gauge. The
implementation shape:

1. The Krylov solve is unchanged (Euclidean projector for the
   symmetric PAP, as `null_space.hpp` prescribes); the **mass-weighted
   gauge supplies the Tisserand representative** of the solution — one
   mechanism, no new constraint machinery. Under a non-identity
   equilibrium mapping the null basis is the mapped rotations
   (b×φ_e), so the ρ̃-mass metric pins exactly ∫ρ̃ φ_e×u dV = 0 with
   no extra wiring. The gauge is applied before any ω row is read,
   and the referential solve's tail skips its Euclidean rigid
   clean-up when the Tisserand gauge is on, so the representative
   survives the solve.
2. The ω-row data I⁽¹⁾(u, σ) assembles as direct linear forms (no
   MacCullagh, no degree-2 extraction, aspherical-safe),
   referentially with ρ̃ and composition with φ_e; the referential
   row is read off the raw assembled surface-load form before the
   compatibility projection, matching the mixed class.
3. I⁽⁰⁾ is assembled once from the model (a volume integral of the
   density; on the equilibrium mapping's image where one is set).
4. The centrifugal feedback columns enter through `SetTidalLoad`
   (∇ψ_k ∘ φ_e as analytic data — the door its header anticipates),
   and the whole thing is a 3-column border on the shared
   `BorderedElimination` engine, composable with the water border.

The spherical gate: with I⁽⁰⁾ isotropic the balance loses its wander
restoring terms just as the traditional D does with model moments —
which is why the spherical-model comparison against the traditional
border (the verification's spherical leg) is run with the
*input-inertia option* below.

## The input-reference-inertia option

Applied to a simplified model, the self-consistent theory inherits
that model's degenerate rotational response, so the generalised door
also provides an option to input a reference inertia (observed
values) in place of I⁽⁰⁾. What survives, derived rather than
assumed:

- **Uniqueness survives intact.** The re-gauge invariance of ω comes
  from the Tisserand gauge-fixing of u, not from the inertia data;
  with I⁽⁰⁾ replaced by I_obs the pair (u, ω) is still unique. The
  input-inertia generalised door therefore *strictly dominates* the
  traditional theory: same data requirements, no gauge pathology.
  Concretely it is `SetRotation(Omega, moments)` +
  `SetTisserandGauge(true)` — no third API; the model-inertia
  degeneracy refusal points here.
- **Exactness does not.** Conservation-based certificates (δ(IΩ) = 0
  re-integrated at the solution) degrade at O(|I_obs − I_model|): the
  option reintroduces a two-model split, confined to the dense border
  block. The certificates must therefore be gated on the
  self-consistent configuration and only *reported* on the
  input-inertia one.
- The long-run resolution is a model whose own figure is realistic —
  the equilibrium-figures programme (`doc/equilibrium_figures.tex`;
  its surface-water extension, with the observed-sea-level density
  inverse, is future work), after which nothing is imposed.

## The fluid-core frame choice (G8c)

A fluid relabelling (FRL) of a fluid region F is v with div(ρv) = 0 in
F and v·n̂ = 0 on ∂F: unobservable, so every physical reading must be
invariant under u → u + v. Two computations pin what G8c can and
cannot certify:

- **The ω-row IS FRL-invariant.** ∫_F ρ v·∇ψ' dV =
  ∮_∂F ρψ' v·n̂ dS − ∫_F ψ' div(ρv) dV = 0, exactly, for every
  admissible v. The inertia-perturbation data are safe on any fluid
  representative — this part of G8c is a quadrature identity, gated in
  `TestGeneralisedRotation` (`FluidRelabellingCertificate`).

- **The relative angular momentum is NOT.** For ρv = curl W with W
  compactly supported in F, ∫_F ρ x×v dV = 2∫_F W dV ≠ 0; and on a
  spherical fluid core with radial ρ the core's rigid swirls v = b×x
  are themselves FRLs (∇ρ ∥ x ⊥ b×x; v·n̂ = 0 on the sphere) carrying
  h = I_F·b. So the WHOLE-BODY Tisserand condition ∫ρ x×u = 0 is
  ill-defined on a gauged-fluid model: its core contribution is pure
  gauge. (The degree-1 FRL non-removability seen in the slip
  benchmarks is the same freedom in different clothes.)

**The resolution: fix the frame via the mantle.** The
rigid representative is pinned by the SOLID-region Tisserand
condition — the mantle frame of Earth-rotation theory:
`SetTisserandGauge` keeps the global rigid basis but assembles its
metric as the ρ̃-mass restricted to the solid attributes whenever a
fluid treatment is configured (lazily, at solver setup, so the
SetFluid/SetRotation call order is free). The ω-row integrals stay
whole-body — the invariance identity above is exactly what licenses
that: the fluid displacements enter only through a reading the
relabelling cannot touch, so no separate Eulerian fluid-row assembly
is needed. Gated (`FluidCoreMantleTisserand` + the par twin's
`fluid-core mantle-frame` case): on the two-layer fluid-core ball with
the input-moments door composed with the gauged core, the mantle
momentum vanishes at the gauged solution (7.1e-9 of the mantle trace;
the ungauged twin is 333× larger) and ω is invariant across fluid
representatives (ε 1e-2 vs 3e-3: 3.1e-7 relative — the refined
penalty debiases and the rows are relabelling-invariant). The
traditional border with fluid cores was never affected (its row is
the invariant integral; its gauge pathology is the solid one G8b
demonstrates).

Why this is a frame choice and not a physics change: the "core
follows the mantle on secular
timescales" physics lives in the balance itself — D and the ω-row
data are WHOLE-BODY integrals, and dropping ω̇ is precisely the
assumption of one shared rigid rotation. The mantle restriction only
decides how the computed displacement splits into frame + deformation,
declining to let the core's relabelling-undefined ∫ρx×u vote on that
split; every physical reading was already independent of it. A core
that never follows (mantle-only moments C_m in the denominators) would
be a different physical model — the rapid-wobble regime, full M&A
eq. (224) territory, outside the secular door by construction. The
same reasoning covers the Maxwell-relaxation treatment: its core
displacement is solve-determinate but artificial-path-dependent, so
its momentum is a representative's, not physics.

## Verification

| gate | statement | status |
|---|---|---|
| G8a | spherical model, input inertia: generalised ≡ traditional border (route grade) | subsumed: the plumbing identity (`GeneralisedRotationDoor` (i)) is bitwise, and the spherical control runs in the example |
| G8b | triaxial solid model: ω invariant under a rigid re-gauge of u; the traditional row's shift equals the closed form D⁻¹[ad_b, I]Ω (the demonstration deliverable) | done — `TestGeneralisedRotation` (closed form 1e-10, solve-level shift O(1) of ω, pinning 4.7e-8, re-pin invariance) + `examples/generalised_rotation.cpp` (the gauge-sweep figure: traditional ≡ closed form, pinned at 1e-14, spherical control three orders down; ω_gen = ω_trad − D⁻¹[ad_bf,I]Ω at 1.2e-5) |
| G8c | gauged-fluid configuration: inertia integrals invariant under the fluid-gauge representative (FRL certificate) | done — the row invariance is exact and gated as a quadrature identity; the momentum is NOT invariant, so the gauge pins the MANTLE frame (the fluid-core frame choice above), gated at solve level: mantle momentum 7.1e-9 of trace vs 333× ungauged, ω across fluid representatives 3.1e-7 |
| G8d | self-consistent configuration: the secular balance (D ω)_k + ∫ρu·∇ψ_k + ∫σψ_k = 0 re-integrated at the solution, σ the TOTAL load including the water response (conservation certificate; reported, not gated, under input inertia) | done — `ConservationCertificate`, two tiers: STRICT via test-side reassembly on the library's pairing convention (8.0e-8 data load, 4.9e-6 composed water) and ROUTE against the exact ψ (3.1e-3 / 9.8e-4: the assembled data integrate σ against the interpolant with the forms' rules; D and the volume row agree at 1e-8); parallel twin case `water+generalised`, serial ≡ parallel at 1e-5, Tisserand 4.0e-5 at o1 → 1e-7-grade at o2 |

What remains open — the secular reduction against the viscoelastic
time-stepping layer (with the Mitrovica & Wahr comparison as a
numerical check, not a theory input) and whether the translations'
momentum condition ever needs the mantle-frame treatment — is tracked
in `doc/planning/future_work.md` (the referential sea-level residue).

**See:** `doc/quasi_static_models.tex` (the referential class's
sea-level and rotation section);
`include/AdGIA/null_space.hpp` (`SetGauge`);
`include/AdGIA/quasi_static_problem.hpp` (`SetMassWeightedGauge`);
`examples/generalised_rotation.cpp`;
`tests/TestGeneralisedRotation.cpp`;
`doc/Elasticity/ggae388.pdf` §2.4; `doc/Elasticity/ggae092.pdf`
eqs. 88, 100, 222, 224.
