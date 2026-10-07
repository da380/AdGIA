# The no-gravity slip family: exact references, 2-D first

**Status:** derivation note, 7 Oct 2026; updated 8 Oct 2026. Rungs 0
and 1 are implemented and passing (`benchmarks/disc/`). Rung 2 is
implemented and validated for welded, slip and Dahlen at l = 0 and
l ≥ 2 (`disc_models.py`, `disc_radial.py`; design in
[disc_gravity_reference.md](disc_gravity_reference.md)), the Dahlen
switch via the Eulerian mixed-class path after the referential glue
failed the aw_core_2d null test; l = 1 pending. The 3-D twins (Love's
shell solution, the κ/V fluid spring) follow the identical structure
and are deferred.

**Purpose.** With gravity off, the slipping problem's physics becomes
trivial while its *machinery* — the broken space, the pairing, the
normal constraint (penalty/AL/KKT), the enlarged null space, the
mapped assembly under `[[F]] ≠ 0` — remains fully engaged. Closed-form
solutions then test the machinery in isolation, including the
interface-shift derivative with an exact reference (no pyslfp, no
finite differences on the reference side). This is the separator for
the shift-derivative question: clean here ⇒ what remains lives in the
gravity forms; dirty here ⇒ the defect is in the constraint/elastic
machinery, with closed forms to debug against. It is also the
machinery control for the FRL question: without gravity every
tangential slip is removable (a zero-flux div-free extension always
exists), so **welded ≡ slip exactly, on any geometry** — any gap
between them in this family is implementation error by definition.

## Rung 0: 2-D, no gravity, no pre-stress

Geometry: fluid disc 0 ≤ r ≤ r_c (bulk modulus κ_f, no shear, no
pre-stress), solid annulus r_c ≤ r ≤ a (plane strain, Lamé λ, μ).
Load: prescribed normal traction on r = a, one Fourier degree at a
time, σ·n̂ = −t cos(lθ) n̂ (plus the shear-free condition σ_rθ = 0 at
r = a; a tangential load variant is the same algebra). Response
coefficients: the cos lθ / sin lθ coefficients of u_r, u_θ on r = a
(and on r = c when wanted) — the family's "Love numbers", ours to
define since there is no conventional reference.

### The fluid reduces to one number

Static equilibrium of the fluid without body force forces its pressure
perturbation p₁ = −κ_f div u to be **constant**, so the fluid responds
only to its net area change:

    p₁ = −κ_f ΔA / A,     ΔA = ∮_{r=c} u·n ds,   A = π r_c².

- **l ≥ 1**: the load pattern's area change vanishes degree by degree
  (∮ u·n picks the l = 0 moment only), so p₁ = 0 and the CMB is
  **exactly traction-free**: the solid solves a free-inner-boundary
  annulus problem and the fluid displacement is pure null space.
- **l = 0**: the fluid is an area spring of stiffness κ_f; see below.

The FE slip solve must reproduce this through the full constraint
machinery, with the fluid's enormous null space (every div-free field)
handled by the projection/gauge path — this family is the "no-gravity
slip class as KKT testbed" anticipated in the class-naming refactor.

### l = 0: Lamé plus the area spring

Axisymmetric plane strain in the annulus: u_r = A r + B / r,

    σ_rr = 2(λ + μ) A − 2μ B / r²            (plane strain),

with the two constants fixed by σ_rr(a) = −t and σ_rr(r_c) = −p₁,
and the spring closure p₁ = −κ_f · 2 u_r(r_c) / r_c (from
ΔA = 2π r_c u_r(r_c)). Three linear equations, closed form. The limit
κ_f → 0 is the empty-hole annulus, κ_f → ∞ the incompressible-core
(u_r(r_c) = 0) annulus — both ends are classical checks.

### l ≥ 2: the Michell annulus

Airy stress function Φ = f(r) cos lθ with the Michell radial basis

    f(r) = A r^{l+2} + B r^{l} + C r^{-l+2} + D r^{-l},

stresses by the standard polar formulas, displacements by the
plane-strain Michell integration (the displacement basis is power-law
too; the implementation carries the standard tables rather than this
note). The four constants solve the 4 × 4 system of boundary
conditions

    σ_rr(a) = −t,  σ_rθ(a) = 0,  σ_rr(r_c) = 0,  σ_rθ(r_c) = 0,

exact linear algebra on exact matrix entries (rational in r_c, a,
powers of each). No special functions anywhere.

### l = 1: the compatibility projection

A cos θ normal traction carries net force, which a non-gravitating
static body cannot balance: the problem is solvable only on the
quotient by translations, which is exactly what the projected
rigid-mode solver does on the FE side. The reference solves the
Michell l = 1 system (which acquires the r log r term) for the
*compatible* part of the load — t cos θ minus the traction pattern of
a uniform force — or equivalently applies the self-equilibrated load
t (cos θ − pattern) directly. Degree 1 is where the slip family
misbehaves in 3-D, so this rung is worth the extra care rather than
skipping.

### The shift derivative, exactly

Every boundary-condition system above is M(r_c) x = b(r_c) with
entries polynomial (or power-law) in r_c, so

    dx/dr_c = M⁻¹ (db/dr_c − dM/dr_c · x)

is exact — the interface-shift derivative of every response
coefficient in closed form. The FE side runs the same shift maps as
the 3-D perturbation family (`-map-shift`, interpolated F per
SubMesh), giving the mapped assembly under `[[F]] ≠ 0` an exact
derivative reference for the first time. The comparison isolates the
constraint row and the mapped elastic volume operators: the gravity
interface forms (G_Σ, the ζ-jump row, b, q) are absent by
construction.

### What rung 0 tests, explicitly

1. welded ≡ slip ≡ exact, every degree (machinery identity; any gap
   is a bug — the 2-D FRL control).
2. Penalty vs AL vs KKT endpoints against exact values (the KKT 2-D
   lower-order-multiplier divergence gets an exact-answer testbed).
3. Null-space handling: fluid div-free space, independent rotations,
   translation quotient at l = 1.
4. Shift derivatives vs exact, h- and order-ladders at 2-D cost.

## Rung 1: uniform surface pressure P₀ (still no gravity)

A uniform normal pressure P₀ on r = a with no body force equilibrates
with the homogeneous hydrostatic pre-stress S_e = −P₀ I in solid and
fluid alike: π = P₀ constant, the non-natural reference state of the
Traction classes (dead referential tractions; stiffness = material +
geometric only — `doc/gravitating_elasticity.md` §2). This switches
on B_Σ (coefficient π) and the fluid's bare shear μ_b = P₀. The
derivation collapses further than first expected — three structural
facts (each to be checked):

**(1) The volume operator is classical Navier with the effective
moduli.** The perturbation first Piola is δP = λ_b div u I + 2μ_b ε −
P₀ Du, and div(−P₀ Du) = −P₀ Δu, so the strong form is Navier with

    λ_eff = λ_b + P₀,    μ_eff = μ_b − P₀

— the WD effective-moduli dictionary with its gravity terms switched
off (∇p⁰ = 0 exactly). The rung-0 power-law basis therefore carries
over verbatim with (λ_eff, μ_eff); no new exponent eigenproblem is
needed after all.

**(2) The fluid's P₀-stiffness is a null Lagrangian.** The fluid's
effective shear is μ_eff,f = P₀ − P₀ = 0 (the dictionary's fluid
statement), and the residual geometric+material combination

    ∫_fluid P₀ [ Du : Duᵀ − (div u)² ] dA

is a pure boundary functional of the interface trace, so the fluid
again enters only through (i) the κ_f area spring at l = 0 and (ii)
an explicit P₀-weighted boundary quadratic in its trace amplitudes
(U_f, V_f), computable in closed form on the circle. The fluid
displacement's volume indeterminacy survives exactly as in rung 0.

**(3) The traction rows gain closed-form cof corrections.** With
cof(A) = tr(A) I − Aᵀ in 2-D,

    δP·N = σ_eff·N − P₀ cof(Du) N,

and cof(Du)N is tangential-derivative-only on the boundary: in
amplitudes, the radial traction row gains −P₀ (U + lV)/r and the
tangential row −P₀ (lU + V)/r. (Sanity at l = 0: this reproduces
λ_b(U′+U/r) + (2μ_b−P₀)U′ directly from δP.) Both the loaded surface
rows and the interface rows carry these corrections. The slipping
problem's interface conditions then follow from 1-D stationarity of
the per-degree functional — solid and fluid volume terms, the fluid's
boundary quadratic from (2), and B_Σ with π = P₀, whose 2-D
amplitude form on the circle of radius c is explicit
(ν̂·∇_Σv[s] on the circle = s_θ(∂_θ v_r − v_θ)/r, so B_Σ contracts
to −(π_pres/c)·S·[l(U_s+U_f) + (V_s+V_f)] per unit angular
normalisation, S = V_f − V_s the tangential-jump amplitude). The
natural interface conditions of that stationarity are exactly what
the FE slip solve assembles (constraint row + B_Σ + cof-corrected
tractions): rung 1 tests B_Σ and the multiplier physics with no
potential block anywhere.

The welded variant keeps the volume geometric terms and the
cof-corrected surface rows but has no interface term (continuous
trace and continuous π: the two sides cancel); welded ≡ slip remains
exact (the slip is still removable without gravity), so the identity
stays the machinery test — now with B_Σ active on the slip side.
Rotations are near-null through moment balance (dead loads), matching
the Traction class's RigidPairResiduals expectation.

## Rung 2: gravity on the disc — the radial-ODE reference

The 2-D gravitating twins (uniform-density disc core = fluid_core-2D;
the closed-form AW disc core, g = 2πGρ₀r(1 − αx²/2),
κ = (πGρ₀²r_c²/α)(1 − αx²)²(1 − αx²/2), mass-matched the same way)
against a small radial-ODE solver, per Fourier degree. Structural
bonus making this easy: on a circle the strict-class fluid null space
per degree l ≥ 1 is **empty** (tangent-to-level-circles + div-free
forces rigid rotation, which is l = 0), so the full-fluid radial BVP
per degree needs no gauge. The solver therefore implements, as
switches in one place: the full static fluid with a **welded or
free-slip CMB**, and the **Dahlen-reduced** fluid — the discriminator
stack (welded-vs-slip = the FRL gap; full-vs-Dahlen = the O(N²)
question) at 1-D precision, independent of all FE code. Its own
derivation note comes when rungs 0-1 are clean.

A 2-D caveat to carry: on a circle the surface-divergence-free slips
are the rigid rotations *only*, so with gravity and N² ≠ 0 every
θ-dependent slip is non-removable — the welded-slip gap should be
generic across degrees in 2-D, unlike 3-D where the toroidal sector
is removable. 2-D is the mechanism lab, not a quantitative proxy; the
3-D confirmation runs on the already-built aw_core.

## Implementation shape (for discussion)

- `benchmarks/disc/` (or tests first): the exact reference as one
  Python module (`disc_reference.py`: rung-0 closed forms + the
  rung-1 exponent eigenproblem + later the rung-2 BVP), validated
  internally against its own limits (κ_f → 0, ∞; P₀ → 0; thick/thin
  annulus asymptotics).
- The FE side reuses the existing two-layer-disc construction of the
  2-D lab/tests; a thin driver applies the degree-l boundary load,
  reads Fourier coefficients on the boundary circles, and runs the
  shift maps. Serial and parallel in the same source, as always.
- The 2-D model constants live beside the 3-D ones so the AW recipe
  is stated once.
