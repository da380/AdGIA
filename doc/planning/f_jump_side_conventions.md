# The broken-ζ forms under a jumping F: theory pass and audit

**Status:** theory pass and implementation audit complete, 7 Oct 2026;
awaiting the order-3 perturbation rung. Verdict: the derivation
contains no implicit `[[F]] = 0` assumption, and the implementation's
one-sided data are internally consistent, so the side-convention
hypothesis for the `slip_broken` shift-derivative disagreement is
retired at both levels. The two sign computations below carry the
argument and are the items to check on paper.

Companion of `doc/planning/open_issues.md`, "Broken-ζ interface forms
when F jumps across Σ" and "`slip_broken` interface-shift derivatives
disagree with the 1-D reference"; derivation references are to
`doc/slip_interface.tex`.

## Setting

Interface-shift mappings are the only configurations with a jumping
mapping gradient across Σ. The map φ_e itself is continuous, so the
jump is Hadamard-rigid:

    [[F_e]] = a ⊗ N        (rank-one normal; radial shifts have a ∥ N).

Every side-dependence question below reduces to this structure.

## Five side-independence facts

1. **F_e t is side-free for tangential t**: `[[F_e]]t = a (N·t) = 0`.
   The two sides agree as linear maps on TΣ.
2. **The Nanson normal ν = cof(F_e)N is side-free, exactly**:
   `cof(F)N = Ft₁ × Ft₂` uses tangential images only. This holds
   discretely too — for per-SubMesh interpolated maps the face
   restriction is fixed by the shared face nodes, and for the exact
   branching map by the continuity of φ_e — which is why the
   Nanson-built constraint row (`B_n`) was covariant by construction.
3. **The slip slot is side-free**: on the constraint set
   `[[v]] = −F_e s ∈ F_e(TΣ)`, where the two sides are the same map,
   so both inverses recover the same s. Off the constraint set the
   fixed direction slot `P_T F⁻¹(1 − ν̂ν̂ᵀ)` first projects onto
   ν^⊥ = F_e(TΣ) and is therefore side-free for arbitrary input: the
   covariance fix of 7 Oct bought side-independence as a corollary.
4. **b = F_e^{−T}∇ζ⁰ = ∇Φ₀∘φ_e is continuous — as a same-side
   composition only.** `∇ζ⁰ = F_eᵀ(∇Φ₀∘φ_e)` and `F_e^{−T}` jump
   individually; the jumps cancel iff both factors come from one side
   (either side). Needs Φ₀ ∈ C¹ at the deformed interface (gravity is
   continuous). This is the single side-hazardous datum of the whole
   construction: a mixed-side composition errs at O([[F]]) = O(ε).
5. **q = (1/4πG) b·ν is continuous** given 2 and 4; per side it is the
   natural flux of the mapped Poisson operator, whose continuity is the
   background interface condition.

What genuinely jumps: `J_e`, `F_e⁻¹` and `F_e^{−T}` on general
vectors, `∇ζ⁰`, `∇∇ζ⁰`, `D²φ_e`, and the Maxwell tensor `T_g`.

## The two computations that carry the argument

**(1) The kinematic datum b₂ is side-free as a sum.** Since
σ_ε : Σ → Σ, the composition φ_e∘σ_ε reads only φ_e|_Σ (single-valued),
so the geometric part of b₂ must be side-free as a whole even though
its pieces are not. Explicitly, with the repo convention ∇_s N·s =
+II(s,s) (circle of radius r, outward N: II(s,s) = |s|²/r, matching
`eq:sigma2`'s N·σ₂ = −|s|²/r): differentiating `[[F_e]] = a⊗N`
tangentially along s,

    [[D²φ_e[s,s]]] = (∇_Σa[s])(N·s) + a(∇_s N·s) = + a II(s,s),
    [[F_e σ₂]]     = a (N·σ₂)                     = − a II(s,s),

and the sum cancels. The Weingarten collapse (`rem:wein`) then lands on
tangential derivatives of the single-valued surface function
F_e s = −[[v]], side-free by fact 1. Hence **B_Σ needs no side
convention**: π continuous (traction continuity), ν by fact 2, traces
intrinsic, slip slot by fact 3.

**(2) The second-variation cancellations are internally solid-side and
survive any jump.** The `[[ζ²]]` datum comes from expanding
ζ_f = ζ_s∘σ — all derivatives of solid-side fields — and the σ₂ term of
the Maxwell pairing comes from b₂, which `eq:expand` builds from the
solid motion. So the advertised cancellation ∇ζ⁰·σ₂ ↔ b·F_eσ₂ is F_s
against F_s: exact for any [[F]]. The scalar analogue of (1), with
`[[∇ζ⁰]] = cN`:

    [[∇∇ζ⁰[s,s]]] = + c II(s,s),      [[∇ζ⁰·σ₂]] = − c II(s,s),

so the jump datum is side-free as a sum and side-dependent piecewise —
an implementation must take its pieces from one side together, never
mixed. The surviving object of term (ii) is the Maxwell **traction**

    T_g N = (|b|²/8πG) ν − q b,

continuous by facts 2, 4, 5 although T_g itself jumps. Every
coefficient of the final `G_Σ` (ν, b, q, |b|²) is in the continuous
list.

## What each form must consume when F jumps

| form | data | requirement |
|---|---|---|
| constraint row `B_n` | ν | none — Nanson is side-free, also discretely |
| `B_Σ` | π, ν, slip slot | none (post direction-slot fix) |
| `[[ζ¹]] = b·[[v]]` row | b | same-side composition `F^{−T}∇ζ⁰`, either side |
| `G_Σ` | b, q = b·ν, |b|², ν, slip slot | same-side b; the collapsed traction combination, never raw T_g (no explicit J or F⁻¹) |

No side is preferred; the only requirement is internal consistency of
b's two factors and use of the collapsed (continuous) coefficient
combinations.

## Implementation audit (7 Oct 2026) — PASSES

- `SlipInterfaceGravityIntegrator` assembles exactly
  `A = (|b|²/8πG)ν − (b·ν/4πG)b` and
  `SlipInterfaceGravityScalarIntegrator` exactly `q = (b·ν)/4πG`: the
  collapsed combinations; no raw `T_g`, no bare `J` or `F⁻¹`
  (`src/bilininteg.cpp`).
- b is built in-kernel as `F^{−T}∇ζ⁰` with `F = map->EvalGradient` and
  `∇ζ⁰ = GradientGridFunctionCoefficient(zeta0_shadow)` — the discrete
  solid-side background field — both evaluated through the same
  (solid) element transformation at the same point: same-side by
  construction. The constraint row's `MappedBackgroundField` does the
  identical composition for `Kvz`/`Pb`
  (`src/referential_problem.cpp`).
- ν is `Diffeomorphism::MapNormal` (Nanson) everywhere; the direction
  slot is the fixed `P_T F⁻¹(1 − ν̂ν̂ᵀ)` in all three kernels.
- All `G_Σ` blocks (solid and fluid test/trial slots alike) use the
  solid-side b — a consistent one-sided choice, which is all the
  derivation requires.

Residual discrete caveats, all convergent and none O(ε)-systematic:
∇ζ⁰ is the one-sided gradient of the discrete background solve
(O(h^p) at the interface, smooth in ε since the mapped background is
re-solved per shift with a smooth map family); exact branching maps
remain side-hazardous at interface quadrature points (the known
hazard; the shift driver forces interpolated F).

## Consequences

- There is no missing interface term proportional to [[F]], and no
  mixed-side bug: the side-convention explanation of the `slip_broken`
  shift-derivative disagreement (h'₂ 25 %, k'₂ 19 %) is **retired**.
  Remaining candidates: discretisation (the order-3 rung of the
  perturbation family, running as of this note — the Leg-A playbook),
  and the physics content of the comparison itself.
- pyslfp's own numerics are exonerated as a contributor: an ngll
  5 → 12 ladder on `fluid_core` and its ±0.02 shifted models moves
  every Love number and every central-difference derivative by
  ≤ ~1e-10 relative (7 Oct 2026). The reference solves the
  Dahlen-reduced equations essentially exactly; all genuine
  disagreement with it is physics (the reduction vs the full static
  fluid) or 3-D discretisation. Note l'₂ = −0.0178: the 12–17 % l'
  offsets are percentages of a small denominator.

## To check on paper

The two displayed computations (the II sign convention does real work
in both), and the claim ν^⊥ = F_e(TΣ) behind fact 3 (dimension count
plus ν ⊥ F_e(TΣ)).
