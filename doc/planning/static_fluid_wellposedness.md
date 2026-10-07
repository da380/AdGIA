# Static well-posedness of the non-neutral fluid core

**Status:** consolidation note for review, 7 Oct 2026. The argument
below was assembled in discussion and is supported by the
measurements listed at the end; the functional-analytic statements
are argued, not proved — turning them into theorems is flagged where
it matters. This note is the single home of the "fluid issues";
the scattered fragments (open_issues: the h-ladder and removability
items; benchmarks.tex: the attribution caveat;
disc_gravity_reference.md: the status warnings) point at pieces of
it.

## 1. The slaving

For the full static-elastic fluid (inviscid barotropic-elastic
energy V(x, J), hydrostatic radial background), combine the
linearised momentum balance with mass conservation
ρ¹ = −∇·(ρu) and the constitutive law Δp = −κ∇·u. The tangential
balance slaves the aspherical Eulerian pressure, p̂¹ = −ρφ; the
radial balance then slaves the density, ρ¹ = (∂_rρ/g) φ; and
eliminating ∇·u between the two kinematic relations gives

    u_r (ρN²/g) = −(ρN²/g²) φ,

so for **N² ≠ 0 the radial displacement is slaved pointwise,
u_r = −φ/g**: fluid particles ride the equipotentials. At N² = 0 the
relation degenerates to 0 = 0 and u_r is free. (Dahlen's reduction
derives the first two slavings without ever introducing u; they are
exact interior physics for l ≥ 1 on a spherical background. The
third is what the full-elastic description adds.)

## 2. No classical static solution for N² ≠ 0

Normal continuity at a fluid–solid interface demands
u_r(c) = U_solid(c); the slaving demands u_r(c) = −φ(c)/g(c). For a
generic load these disagree: the static problem is over-determined
at the interface, and **no solution exists in any formulation whose
fluid displacement has a normal trace required to match the
solid's** (classical or H¹-conforming alike). The numerical face of
this (measured, below): the full-elastic fluid fields never
converge — not to a slaved interior plus a boundary layer, but to
nothing (regulariser- and resolution-dependent volume noise), while
the Eulerian potential stays smooth and the solid observables hover
in a band.

## 3. The gauge-theoretic statement (where Maitra–Al-Attar sits)

Every fluid relabelling has w·n = 0 on the boundary, and for N² ≠ 0
the strict class is tangent to level surfaces, so w_r = 0 pointwise:
**u_r is gauge-invariant exactly when the slaving pins it**, and the
obstruction of §2 lives entirely in the invariant sector. At N² = 0,
where class K makes u_r gauge-variable, the pinning equation
vanishes — the solvability of the equations and the richness of the
group switch together.

In MA language the dichotomy is between two degeneracies:

- **kernel** (N² = 0): the zero modes are a clean null space, loads
  are automatically compatible (gauge directions are energy- and
  load-neutral), solutions exist as the FRL orbit, and the MA
  prescription — enforce normal continuity, leave the slip free,
  select a representative by minimising a suitable quantity — is a
  well-posed selection among existing solutions;
- **essential spectrum at zero** (N² ≠ 0): the stratification
  continuum accumulates at zero; there is no clean kernel to quotient
  by and the range is not closed, so a generic load has no solution
  *even modulo gauge*. The selection step has nothing to select
  from: minimising sequences lose compactness (the slaving violation
  concentrates at the interface), the infimum is not attained, and
  the discrete, regularised answers chase the escaping sequence —
  which is why they drift with h and with the regulariser instead of
  converging. For N² < 0 the same in saddle form.

So the static problem is well-posed on the FRL quotient **iff**
N² = 0; otherwise the quotient problem itself is over-determined and
the resolution is dynamical (viscosity, diffusion, genuine time
dependence — physics outside statics decides what the interface
region does).

## 4. What Dahlen's method is, in this light

Dahlen's elimination never carries a fluid displacement, so it never
faces §2: its interior fields are the exact slaved ones, and its
interface bookkeeping — the F3 surface-mass term ρ_F(m·u)φ′ — is
precisely the account of the fluid displaced *normally across* the
boundary. Reconstructing the implied displacement from Dahlen's
fields gives the equipotential-riding u_r = −φ/g, which does not
match the solid's CMB displacement: **the method implicitly contains
a normal-displacement jump at the CMB**, booked as surface mass.
(The historical irony is noted: Dahlen was scathing about Crossley &
Gubbins' suggestion of exactly such a jump.) This makes Dahlen's
static problem well-posed — it is the secular closure, the state the
fluid reaches after rearranging — and κ-blind at l ≥ 1 by
construction. Its "failing" is epistemic: the number is one closure
of a question whose answer carries an irreducible N²-controlled
ambiguity band, presented as unconditional. On a near-neutral outer
core the band is small, which is why this goes unnoticed; on
stratified models or wherever core stratification is the object of
study, it is not ignorable.

Consequences for comparisons: a welded method against a Dahlen
reference (pyslfp) bundles the welded-suppression (FRL) error, the
static-vs-secular closure difference and the non-neutral band; none
of the three is separately resolved on fluid_core-type models, and
all three vanish together on aw_core-type models. The clean
instrument for everything in this note is the Adams–Williamson twin.

## 5. Evidence (all 7 Oct 2026)

On the 2-D per-degree radial reference (`benchmarks/disc/
disc_radial.py`, validated against the rung-0 closed forms at
1e-9..1e-13):

1. aw_core_2d (N² = 0): welded, slip and Dahlen agree to ~1e-10 at
   every degree; the welded/slip p-ladder converges to ~1e-12.
2. fluid_core_2d (N² < 0): the Dahlen solve converges to 7 digits;
   welded and slip wander non-monotonically within a ~0.3 % envelope
   under p-refinement and move at ~1e-3 with the fluid regulariser.
   Fixed-resolution splits inside that envelope are not
   decompositions.
3. The fluid displacement field never tracks the slaving, at any
   regularisation tried — the §2 non-existence displayed; the
   Eulerian φ stays smooth throughout.
4. κ-scaling: Dahlen's answer is unchanged to 1e-11 under a 30-fold
   scaling of the core bulk modulus (the M4 elimination never sees
   κ at l ≥ 1); the full-elastic band about it shrinks ~14x over the
   same N² range (a noisy envelope estimator).
5. The FE replica (`examples/self_gravitating_solvers -model aw`):
   all six architectures coincide at ≲1.2e-3; `-model fc` shows the
   genuine spreads. Found en route and worth remembering: the Dahlen
   interface terms take the FLUID-side density (a radial branch
   hands interface quadrature points the mantle value — invisible on
   equal-density models), and the gauge/AL windows are model- and
   formulation-dependent (fc needs ε = 0.1 and θ = 1e3 where the
   uniform demo wanted 1e-2 and 1e2) with the slip classes unable to
   warn.

In 3-D: the h = 0.3 perturbation-family residuals (order-dependent,
non-converging, mixed in sign between methods — benchmarks.tex, "The
perturbation family") are consistent with this picture; the 3-D
aw_core decomposition and ladder are the pending confirmations.

## 6. Open

- The functional-analytic theorem behind §2–3 (range not closed, 0
  at the edge of the essential spectrum for the static operator on
  the strict-class quotient; non-attainment for N² > 0) — argued
  here, not proved.
- The N²-scaling of the band as a clean measurement (the 1-D tool
  does it in seconds; the estimator needs more than two resolutions).
- The 3-D aw_core runs: the degree-1 welded–slip null test, the
  h-ladder on the Dahlen path (which the 2-D result predicts
  converges — deciding whether the floor explains the 3-D h-ladder
  residual items), and the welded/slip/Dahlen decomposition of the
  pyslfp offsets.
- Whether any of this warrants a paper-level write-up is David's
  call; the pieces (slaving, dichotomy, Dahlen-as-closure, the
  measured band and its N²-scaling) are assembled here.
