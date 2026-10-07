# Rung 2: the gravitating disc and its radial reference

**Status:** derivation/design note, 7 Oct 2026; updated 8 Oct 2026.
Implemented and validated for welded, slip and Dahlen at l = 0 and
l ≥ 2 (`benchmarks/disc/disc_models.py`, `disc_radial.py`;
`python disc_radial.py` runs the validation). The Dahlen switch is the
Eulerian mixed-class path (eq. (mixedA) + F1-F3, continuous φ), after
the referential ζ-glue version failed the aw_core_2d null test; at
l = 0 it dispatches to the welded full-fluid solve. l = 1 pending. On
fluid_core_2d the welded and slip solves sit on a non-neutral
resolution floor (~0.3 % envelope at l = 2) while Dahlen converges, so
the welded-slip and full-vs-Dahlen splits are not resolved there; see
`open_issues.md`, "Which tangential slips are removable by
relabelling". Companion to `no_gravity_analytics.md` (rungs 0-1, both
implemented and passing) and `benchmarks/disc/README.md`.

**Purpose.** The 2-D twins of `fluid_core` and `aw_core` solved by the
disc FE driver (welded / slip), against a per-degree radial reference
that implements, as switches in one code path:

  - the **full static fluid** with a **welded or free-slip** CMB,
  - the **Dahlen-reduced** fluid,

i.e. the discriminator stack pyslfp cannot provide: welded-vs-slip
(the FRL gap) and full-vs-Dahlen (the O(N²) question) at 1-D
precision, each isolable per degree, at resolution-ladder cost of
seconds. On `aw_core_2d` (N² = 0) the strict prediction is
welded ≡ slip ≡ Dahlen; on `fluid_core_2d` (N² < 0) all three split,
and in 2-D the FRL split is *generic across degrees* (the
surface-divergence-free slips of a circle are the rigid rotations
only), which is what makes the disc the mechanism lab.

**Scope statement.** The reference is the θ-reduction of the code's
own documented weak forms (the welded gauged referential functional
and the slipping organisation of `doc/gravitating_elasticity.md` §2-3
and `doc/slip_interface.tex`), solved by an independent 1-D
discretisation. It therefore tests *discretisation and the
2-D/3-D assembly*, and *discriminates the physics switches*; it does
not re-derive the formulation — that validation belongs to rungs 0-1
(exact), to the identity tests, and to pyslfp in 3-D. This keeps the
derivation transcription-shaped rather than invention-shaped, which
today's sign-error ledger strongly recommends.

## Conventions

Codebase Poisson convention in either dimension: ∇²Φ₀ = 4πGρ, so in
2-D g(r) = (4πG/r)∫₀^r ρ s ds and the hydrostatic state is
`RadialHydrostaticState(dim = 2)`. Harmonics cos lθ / sin lθ with the
same component conventions as the disc driver (u_r = U cos lθ,
u_θ = V sin lθ, φ = Z cos lθ). Exterior potential per degree:
Z ∝ r^{−l} for l ≥ 1 (DtN row Z′(a⁺) = −(l/a) Z(a)); at l = 0 the
exterior is logarithmic, Z′(a⁺) = 2G δM / a with δM the perturbation
mass — the 2-D DtN's special degree, matching the code's 2-D
conventions (`poisson.hpp`; the zero-boundary-mean gauge fixes the
constant).

## The models

- `fluid_core_2d`: uniform fluid disc (the 3-D `fluid_core` values:
  ρ 11000, κ from vp 9000) under the uniform solid mantle annulus,
  CMB at 3483/6371 — N² < 0 in the core, as in 3-D.
- `aw_core_2d`: the closed-form neutral twin, dimension-adapted:

      ρ(r)   = ρ₀ (1 − α x²),            x = r / r_c,
      g(r)   = 2πG ρ₀ r (1 − (α/2) x²),
      κ(r)   = −ρ² g / ρ′
             = (πG ρ₀² r_c² / α)(1 − αx²)² (1 − (α/2) x²),

  N² = 0 identically; mass-matched to `fluid_core_2d`'s core
  (ρ₀(1 − α/2) = 11000, α = 0.2 ⇒ ρ₀ = 12222.2…), same mantle, so the
  mantle state is identical between the twins, as in 3-D. The same
  SHIFTABLE convention: a shifted r_c re-derives κ with (ρ₀, α)
  frozen, keeping every family member neutral.

Both live beside the disc benchmark (its own small models module —
they are not planetmodel models), with ρ, κ, p₀ as callables the
driver projects and the reference consumes directly.

## The per-degree functional

Unknowns on [0, a] (solid on [c, a], fluid on [0, c], potential
ball-wide + DtN): U, V per region and Z. The pieces, each the exact
θ-integral of the code's form (the π from ∫cos² dθ divides out):

1. **Material (bare) + geometric**, hydrostatic natural state
   (S_e = −p₀(r) I, bare moduli from the physical ones by the
   dictionary λ_b = λ − p₀, μ_b = μ + p₀; fluid μ_b = p₀):
   with the 2-D strain amplitudes

       e_rr = U′,   e_θθ = (U + lV)/r,   2 e_rθ = V′ − (V + lU)/r,

   the material term is λ_b (e_rr + e_θθ)² + 2μ_b (e_rr² + e_θθ²
   + 2 e_rθ²), and the geometric term −p₀ tr(DuᵀDu) with the gradient
   amplitudes (Du)_rr = U′, (Du)_rθ = −(lU + V)/r, (Du)_θr = V′,
   (Du)_θθ = (U + lV)/r (the rθ/θr pair carries the sin-channel
   pairing; the reduction table is the one checkable item here and
   gets its own verification against the 2-D FE forms on one
   element).

2. **Gravity**, in the referential ζ-organisation the code solves
   (ζ = φ + u·∇Φ₀, `doc/gravitating_elasticity.md` §3): the Poisson
   block (1/4πG)|∇ζ|² with the DtN closure, the ρ u·∇ζ coupling, and
   the ρ ∇∇Φ₀ mass term, each transcribed and θ-reduced; radial
   weights r dr throughout. The transcription is from the reference
   doc, term for term — no independent rederivation.

3. **Interface switches** at r = c:
   - *welded*: one (U, V) through; no interface terms (continuous
     trace, continuous p₀ via the hydrostatic state).
   - *slip*: independent (U_s, V_s | U_f, V_f) with the single
     constraint U_s(c) = U_f(c) (1-D multiplier — no penalty, no AL),
     plus the θ-reduced B_Σ and broken-ζ G_Σ / ζ-jump terms of
     `doc/slip_interface.tex` in their amplitude forms (the B_Σ
     contraction is already in `no_gravity_analytics.md` rung 1; the
     G_Σ one is its gravity analogue with coefficients b, q from the
     background). Per degree l ≥ 1 the fluid's strict-class null
     space is empty in 2-D, so the system is non-degenerate with no
     gauge machinery at all.
   - *Dahlen*: fluid eliminated; the F1/F2/F3 interface and volume
     terms of `doc/benchmarks.tex` ("The CMB treatments") θ-reduced,
     unknowns (U, V) on the solid only plus Z ball-wide.

4. **Loads and observables**: degree-l surface mass load σ = cos lθ
   (its own potential by the exterior solution) and/or dead normal
   traction as in rungs 0-1; observables U(a), V(a), Z(a) and the
   disc "Love numbers" they define. Degree 1 in the
   centre-of-mass-style gauge (translation-invariant combinations
   compared, as in rung 0).

## Implementation

A 1-D hp finite-element solve of the functional (Python, numpy:
Gauss-Legendre elements, order ~8-10, graded toward r = c; a few
hundred dof per degree reaches near machine precision and a
resolution ladder costs milliseconds). One assembly routine takes the
switch (welded | slip | dahlen) and the model; `solve_degree` returns
amplitudes and observables. No shooting, no ODE-system hand algebra:
the only hand-derived content is the reduction table in 1., which is
verified directly against the 2-D FE bilinear forms on a one-element
mesh before use.

## Validation ladder (before any use as a reference)

1. **G → 0, p₀ → 0 limit** must reproduce `disc_reference.py`
   (rung 0) to solver precision, including the interface derivative
   by its c-parameterisation; with p₀ = P₀ constant, rung 1.
2. **Reduction-table check**: each 1-D form against the 2-D FE form
   on a single annular element ring, term by term.
3. **aw_core_2d null test**: welded ≡ slip ≡ Dahlen to solver
   precision (N² = 0, class K removes every slip; Dahlen exact when
   neutral). Any split is a transcription bug.
4. Then the physics runs: `fluid_core_2d` splits (welded vs slip vs
   Dahlen per degree — the 2-D FRL gap is predicted at *every*
   degree), N²-scaling via a κ-scaled core family, and the shift
   derivative d/dr_c against the 2-D FE driver's mapped runs.

## What rung 2 feeds back to 3-D

The welded-slip split per degree on `fluid_core_2d` is the 2-D
measurement of the FRL suppression error (the degree-1 2.6 % question
in 3-D); the full-vs-Dahlen split on the same model is the 1-D
measurement of the O(N²) reference mismatch that pyslfp cannot
separate from its own physics; and the 1-D shift derivatives give the
gravitating analogue of the rung-0 exact-derivative test for the
mapped assembly — the remaining suspect sector for the 3-D 25 % gap
(pending the gauge-corrected o3 rerun).
