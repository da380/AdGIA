# The generalised rotational theory (WP7.5): derivation note

*Status: **derivation for review** (no code; WP7.5 of
[`referential_sea_level_plan.md`](referential_sea_level_plan.md)).
Sources: Yu, Al-Attar, Syvret & Lloyd (2025, "the paper",
`doc/Elasticity/ggae388.pdf`) §2.3–2.4 for the traditional theory and
its diagnosis; Maitra & Al-Attar (2024, "M&A",
`doc/Elasticity/ggae092.pdf`) §2.4–2.5, §3.7–3.8 for the resolution;
Mitrovica & Wahr (review,
`doc/Elasticity/annurev-earth-040610-133404.pdf`) for the viscoelastic
plastering this supersedes.*

## The traditional theory, and its two defects

The traditional rotational feedback (our WP2/WP4 border) solves the
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
   quantitative certificate for the WP7.6 demonstration (gate G8, the
   deliverable of plan decision 3).

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
  wobble dynamics (ω̇ and the Chandler-band terms) and is the
  conservation statement δ(IΩ) = 0 resolved for ω; the full eq. (224)
  is there when the wobble is ever wanted. In fluid regions the
  inertia integrals are FRL-invariant — as they must be, since the
  relabelling is unobservable — so the gauged fluid treatments
  evaluate them on any representative, and the referential class uses
  its actual fluid displacements directly (plan decision 5). For
  Dahlen-style treatments the paper's appendix gives the Eulerian
  fluid-region formula.

Why the two defects disappear: (1) one model, exact inertia — nothing
observed is imposed, so nothing is inconsistent, and the viscoelastic
plastering is unnecessary *provided the model's figure is realistic*
(which is the real content of the inertia-data question below);
(2) the rigid ambiguity is pinned physically by the Tisserand
condition before ω is ever computed, so ω is manifestly re-gauge
invariant — the G8 invariance certificate.

## Reduction to our variables

The key structural fact: **the discrete Tisserand condition already
exists in the library.** `NullSpaceProjector::SetGauge` with the
ρ-weighted vector mass matrix (the `SetMassWeightedGauge` door, F4)
chooses the rigid-mode representative with zero net momentum *and
angular momentum* — exactly the linearised Tisserand gauge. So the
expected implementation shape, to be verified in WP7.6:

1. The Krylov solve is unchanged (Euclidean projector for the
   symmetric PAP, as `null_space.hpp` prescribes); the **mass-weighted
   gauge supplies the Tisserand representative** of the solution. One
   mechanism, no new constraint machinery — the DRY outcome of plan
   decision 4. The gauge must be applied before any ω row is read.
2. The ω-row data I⁽¹⁾(u, σ) assembles as direct linear forms (the
   WP2 pattern: no MacCullagh, no degree-2 extraction,
   aspherical-safe), referentially with ρ̃ and composition with φ_e.
3. I⁽⁰⁾ is assembled once from the model (a volume integral of the
   density; on the equilibrium mapping's image where one is set).
4. The centrifugal feedback columns enter through `SetTidalLoad`
   (∇ψ_k ∘ φ_e as analytic data — the door its header anticipates),
   and the whole thing is a 3-column border on the shared
   `BorderedElimination` engine, composable with the water border
   exactly as in WP4.

The spherical gate: with I⁽⁰⁾ isotropic the balance loses its wander
restoring terms just as the traditional D does with model moments —
which is why the spherical-model comparison against the traditional
border (gate G8's first leg) is run with the *input-inertia option*
below.

## The input-reference-inertia option

David's requirement (plan, WP7.5 insert): applied to a simplified
model the self-consistent theory inherits that model's degenerate
rotational response, so the generalised door also wants an option to
input a reference inertia (observed values) in place of I⁽⁰⁾. What
survives, derived rather than assumed:

- **Uniqueness survives intact.** The re-gauge invariance of ω comes
  from the Tisserand gauge-fixing of u, not from the inertia data;
  with I⁽⁰⁾ replaced by I_obs the pair (u, ω) is still unique. The
  input-inertia generalised door therefore *strictly dominates* the
  traditional theory: same data requirements, no gauge pathology.
- **Exactness does not.** Conservation-based certificates (δ(IΩ) = 0
  re-integrated at the solution) degrade at O(|I_obs − I_model|): the
  option reintroduces a two-model split, confined to the dense border
  block. The certificates must therefore be gated on the
  self-consistent configuration and only *reported* on the
  input-inertia one.
- The long-run resolution is a model whose own figure is realistic —
  the equilibrium-figures ladder (`future_work.md`, "Equilibrium
  backgrounds with surface water", including the observed-sea-level
  density inverse), after which nothing is imposed.

## What WP7.6 gates

| gate | statement |
|---|---|
| G8a | spherical model, input inertia: generalised ≡ traditional border (route grade) |
| G8b | triaxial solid model (mapped machinery): ω invariant under a rigid re-gauge of u; the traditional row's shift equals the closed form D⁻¹[ad_b, I]Ω (the quotable demonstration) |
| G8c | gauged-fluid configuration: inertia integrals invariant under the fluid-gauge representative (FRL certificate) |
| G8d | self-consistent configuration: δ(IΩ) = 0 re-integrated at the solution (conservation certificate; reported, not gated, under input inertia) |

## To verify in implementation

- That the mass-weighted gauge's basis and metric realise exactly
  ∫ρ̃ (φ_e×u) = 0 under a non-identity equilibrium mapping (the F4
  door was built at φ_e = id).
- The quasi-static secular reduction of eq. (224) against the
  time-stepping layer (what the viscoelastic stepper holds fixed per
  step; the Wahr & Mitrovica comparison belongs here as a *numerical*
  check, not as theory input).
- Whether the translations' momentum condition needs the same
  treatment in practice (loads with net force are excluded by the
  existing machinery; expected inert).

**See:** `referential_sea_level_plan.md` (phases, decisions);
`include/AdGIA/null_space.hpp` (`SetGauge`);
`include/AdGIA/quasi_static_problem.hpp` (`SetMassWeightedGauge`);
`doc/Elasticity/ggae388.pdf` §2.4; `doc/Elasticity/ggae092.pdf`
eqs. 88, 100, 222, 224.
