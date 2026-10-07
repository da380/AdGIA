# Open issues: correctness and verification

Defects, unverified results, open theoretical questions and review items.
Measurements are on the benchmark model `fluid_core` (uniform-density
compressible fluid core, `N² < 0`) unless stated otherwise; `h` is the
nominal mesh size of the benchmark meshes, "order" the displacement order.

---

## The slipping interface

### Mapped `slip_broken`: the slipping-interface forms are not covariant

**Status:** symptom 1 (Leg B) RESOLVED 7 Oct 2026 — an implementation
defect, not a derivation defect, found and fixed; symptom 2 (Leg A)
remains open, now isolated as a separate mechanism (below).

**The defect and the fix.** The continuous `B_Σ`/`G_Σ` of
`doc/slip_interface.tex` are covariant (as they must be: MA24 and AC18
derive on an arbitrary reference body). The implementation was not: the
kernels' direction slot sanitised the jump as `P_T F⁻¹`, which agrees
with the derivation ON the constraint set (`ν·[[v]] = 0`) but is not
invariant under face-fixing shears of F (`F = 1 + a⊗N` on a face —
exactly what adjacent-element evaluation of an interpolated map
produces) off it, where the block probes and the AL/penalty iterates
live. The pullback of the deformed face's own tangential projector is
ν-orthogonal projection BEFORE `F⁻¹`; the slot is now
`P_T F⁻¹(1 − ν̂⊗ν̂)` in all three kernels (`SlipDirection`,
`src/bilininteg.cpp`), identical unmapped. Instrument: the proposed
element-level identity test now exists —
`TestSlipInterface.InterfaceKernelsFaceShearCovariance` assembles each
kernel mapped on the reference mesh against unmapped on the
nodally-deformed mesh (the isoparametric identity): 0.165 / 0.155 /
0.091 relative on the old slot, < 1e-10 on the new one. Measured on
Leg B (`relabelled_identity -method slip_broken`, `fluid_core`,
`h = 0.3`, A = 0.02): δu 1.2e-2 → 1.6e-6, δζ 8.1e-3 → 7.2e-7 — welded
level; the u-block probes fell from 1–2.5e-4 to the 1.45e-7
interpolation floor of the certified constraint kernels. The slip leg
of `relabelled_identity` is now STRICT (same 1e-5 criterion as the
welded runs); the welded legs are unchanged (u 1.6e-7).

**Leg A (exact F, `love_benchmark -map 0.02`): RESOLVED 8 Oct 2026 —
convergent discretisation error, not a defect.** The symptom: spurious
0.36 at degree 1 and 6–8.5e-2 at degree ≥ 2 at order 2, h = 0.3
(identical before and after the kernel fix to every digit — provably
disjoint from the covariance defect, since the exact
`InteriorRelabelling`'s bumps vanish WITH their first derivative at
the layer boundaries, so `F = 1` exactly on every interface and the
interface kernels assemble as unmapped). The chain that closed it:

1. *The amplification hypothesis was falsified first*: perturbing
   `G_Σ` by a known amount on the unmapped run (`-gs-scale` 0.999 /
   0.99) moves h'₁ LINEARLY with slope ≈ 0.8 and leaves `spurious`
   untouched — the degree-1 response is not a sharp amplifier of
   small coefficient errors (the 0.46 at scale 0.5 is a
   large-perturbation effect). So the mapped background's b-error
   cannot reach 0.36 through `G_Σ`'s coefficients.
2. *The order ladder settles it*: at ORDER 3 the exact-F mapped run
   collapses to spurious 0.0071 / 0.016 / 0.0047 at degrees 0–2
   (from 0.033 / 0.36 / 0.085), and the matched-resolution identity
   is restored — mapped vs unmapped order-3 `slip_broken` agree to
   1.3–1.6e-3 in every Love number with the SAME spurious level
   (l = 1: 0.016 vs 0.017). The mapped referential control behaves
   identically (3e-4). INTERPOLATED F was already clean at order 2
   (spurious 0.006–0.016 at all degrees, ~10 % fewer iterations).

Conclusion: the exact-F mapped `slip_broken` at order 2, h = 0.3 is
simply under-resolved in its exact-F-specific channels (the
non-polynomial mapped integrands and the mapped-background data in
the broken constraint row), and degree 1 displays it loudly.
Production guidance: mapped slip runs use INTERPOLATED F by default
(side-consistent, identity-exact with the fixed kernels, cheaper);
exact F is fine at adequate resolution. With this and the kernel fix,
the mapped slipping-interface results are certified at the method's
own level; the reference docs' "mapped slipping results are
unverified" can be retired once the shift-derivative re-measurement
below lands.

*Residual observation (separate, unmapped):* at order 3 the
`slip_broken` degree-1 response sits 2.6 % from the welded
referential's (h'₁ −1.281 vs −1.248) with spurious 0.017 vs 0.0023 —
the slip family's degree-1 behaviour (near-null translation pairs of
the broken organisation?) deserves its own look; it is unrelated to
mapping (present unmapped). The rigid null-pair
residuals of the slip class (`love_benchmark -diag`) survive the map
essentially unchanged (translations 2.4e-3 vs 2.6e-3; rotations about
doubled, ~1e-3).

Certification state after the fix: the KKT constraint row was already
covariant by construction (Nanson-exact flux kernel); `B_n` 2.2e-16,
`P_b` 1.5e-7, `K_vz` 9.5e-8; `B_Σ` and `G_Σ` are now certified at
kernel level (the isoparametric element test) and at solution level
through Leg B. The AL normal–normal penalty with its 1/|ν| factor
passes inside the strict Leg-B identity. The single-valued
organisation's mismatch gravity pieces are assembled at φ_e = id and
refuse maps outright (`Diffeomorphism::IsIdentity` guard); that
limitation is separate. The interface-shift derivative item below was re-measured with the
fixed kernels (unchanged — the fix is provably inert for radial
shifts) and subsequently resolved as gauge pollution plus the physics
envelope (its own item); the "mapped slipping results unverified"
wording in the reference docs can be retired on the strength of the
order-3 ladder and the resolved derivative item.

A data point from `examples/slipping_interface.cpp`: on the
aspherical fluid-core meshes (radial-stretch reference, eps = 0.05,
degree-2 load), mapped `slip_broken` reproduces the spherical answer
(h', k' within 6e-4 in 3-D) with the same off-degree leakage as the
certified welded referential problem (3.8e-4 against 3.6e-4 excluding
degree 1). The leakage above was measured with a lateral interior
relabelling and degree-1 loads; the example with a lateral map would
be the instrument to localise it.

**See:** `doc/benchmarks.tex`, "The relabelling family" (Leg A, Leg B);
`doc/mappings.md` §6 "Verification"; `doc/slip_interface.tex`,
"Gravity: the broken-ζ organisation", "One-sided assembly of the
interface forms"; `benchmarks/relabelling/relabelled_identity.cpp`
(block probes); `benchmarks/common/benchmark_case.hpp` (`-gs-scale`,
`-diag`); `include/AdGIA/bilininteg.hpp`
(`SlipInterfacePressureIntegrator`, `SlipInterfaceGravityIntegrator`);
`include/AdGIA/referential_problem.hpp` ("Limitations" of the
slip class); `tests/TestSlipProblem.cpp`.

### KKT with a lower-order multiplier diverges in 2-D

**Status:** open.

In `examples/slipping_interface.cpp` (2-D two-layer disc, quadratic
displacements, single-valued ζ, `-enforce kkt`) a multiplier one order
below the displacement makes the gauge refinements diverge: the normal
jump grows 1e-3 → 7e-2 → 14.5 over three refinements (order 3 with a
linear multiplier: 1.5e3; order 3 with a quadratic multiplier also
grows). With one refinement the jump stays small but h' is biased
(−1.009 against −1.091). Equal order converges. The 3-D `fluid_core`
measurements that motivated the lower-order recommendation (order-1
multiplier, three refinements, endpoint shift 4e-4) did not show this.
Candidates: the inf-sup margin of the lower-order pairing in 2-D, the
interplay of the multiplier space with the fixed-point form of the KKT
outer loop (full solves with right-hand side f + εQU⁽ᵏ⁾), or the gauge
increment entering the constraint row. The docs recommend the lower
order in 3-D only, with a check of the normal-jump history.

**See:** `doc/slip_interface.tex`, "The multiplier space";
`examples/slipping_interface.cpp` (`-enforce kkt -kkt-order`);
`src/referential_problem.cpp` (`SolveLinearSystemKKT`).

### Exact-F maps with a kink at the surface: one-sided evaluation

**Status:** open (worked around in the example).

The aspherical meshes' radial stretch is tapered across the buffer and
kinked at the body surface. At surface quadrature points the physical
radius lands within ~1e-6 of the knot, and round-off can select the
buffer branch of F, giving b = F⁻ᵀ∇ζ⁰ and the Eulerian potential the
wrong side's gradient (k' errors of 20 % in the slip example before the
fix; the Nanson factor is unaffected). `examples/slipping_interface.cpp`
takes the body side within 1e-4 of the knot. The same exact-F inverse
stretch is used by `benchmarks/relabelling/aspherical_reference.cpp`
(check it), and any Diffeomorphism with a kink on a mesh face has the
issue: a library-level rule (evaluate F from the element's own side, e.g.
by passing the element attribute or a side flag) would remove it.

**See:** `doc/mappings.md`, "Pitfalls"; the `InverseShape` maps in the
example and the benchmark.

### Slipping class: missing diagnostics and noise

**Status:** open; item (1) is a NEEDED FIX (7 Oct 2026), items (2)–(4)
small.

(1) **Needed fix.** The slipping class records no per-refinement gauge
residuals (`GaugeResiduals()` stays empty) and exposes no way to apply
its fluid gauge penalty, so the gauge convergence of the slip path
cannot be inspected as for the welded classes, and no
`WarnGaugeContraction` semi-convergence warning can fire. Evidence: the
`slip_broken` interface-shift derivatives were gauge-polluted
(semi-convergent at order 3, biased at order 2) with no warning, while
the welded runs at the same settings reported it (next item). Fix:
record the residuals per refinement and route them through the same
contraction check as the welded classes. (2) `NewRadialVacuumExtension`
prints MFEM "points were not found" warnings on the aspherical meshes
(the retries resolve them; it is noise on every run). (3)
`MaxIdentityDeviation` lives only in `benchmarks/common/relabelling.hpp`;
the example carries its own check. (4) `SubMesh::CreateFromBoundary`
segfaults in MFEM's node transfer on the curved two-layer meshes, so the
example draws the interface slip on the mantle instead of on an
interface submesh.

**See:** `include/AdGIA/referential_problem.hpp`;
`examples/slipping_interface.cpp`.

### `slip_broken` interface-shift derivatives disagree with the 1-D reference

**Status:** RESOLVED 7 Oct 2026 — gauge pollution plus the physics
envelope of the comparison; no formulation-level anomaly remains.

**What it was.** The recorded disagreement (`slip_broken` h'₂ 25 %,
l'₂ 12 %, k'₂ 19 %, h'₃ 5.6 %, l'₃ 23 %, k'₃ 4.9 %, against referential
6.4 / 12 / 1.7 / 1.9 / 14 / 0.8 %; `fluid_core`, `h = 0.3`, order 2,
ε = ±0.02) was measured at the Love-number gauge setting, penalty
ε_g = 0.01 with 3 Tikhonov refinements. At order 3 that setting is
semi-convergent on this mesh (GaugeRefine contraction ~1.0 at degrees
1–3); at order 2 it is silently biased. Only the welded classes could
report it: the slipping classes record no gauge residuals
(`GaugeResiduals()` stays empty), so the `slip_broken` runs carried no
warning — the missing-diagnostics item above proved load-bearing. The
earlier candidates were eliminated on the way: the direction-slot
covariance fix is provably inert for radial shift maps (their
interface shears are normal–normal, `a ∥ N`), and the side-convention
hypothesis is retired by the theory pass and audit
([f_jump_side_conventions.md](f_jump_side_conventions.md); "Broken-ζ
interface forms when F jumps across Σ" below).

**Definitive numbers** (gauge ε_g = 0.1, 7 refinements; ε = 0.02,
lmax 3, `h = 0.3`; derivative relative difference against the pyslfp
central difference):

| | order | h'₀ | h'₂ | l'₂ | k'₂ | h'₃ | l'₃ | k'₃ |
|---|---|---|---|---|---|---|---|---|
| referential | 2 | 0.79 % | 5.62 % | 11.1 % | 1.23 % | 1.58 % | 13.6 % | 0.53 % |
| `slip_broken` | 2 | 0.78 % | 8.67 % | 7.85 % | 3.97 % | 2.95 % | 16.0 % | 1.97 % |
| referential | 3 | 0.0027 % | 7.90 % | 7.30 % | 3.32 % | 2.86 % | 10.4 % | 1.19 % |
| `slip_broken` | 3 | 0.0027 % | 11.6 % | 2.79 % | 6.36 % | 4.97 % | 13.9 % | 3.20 % |

Absolute agreement with pyslfp at order 3 (worst compared Love number
per degree, ε = −0.02 / 0 / +0.02): referential l = 2
8.2 / 7.9 / 7.8e-2, `slip_broken` l = 2 5.4 / 4.8 / 4.3e-2; degree 0
4.4e-5 / 1.5e-4 / 4.3e-5 for both. pyslfp's numerics are exonerated
(ngll 5 → 12 moves every Love number and central-difference derivative
by ≤ ~1e-10 relative); l'₂ = −0.0178, so the l' percentages sit on a
small denominator.

**Residual observations.** With a converged gauge the degree-2/3
discrepancies are order-dependent and non-converging for BOTH methods,
and the slip-vs-referential differences are mixed in sign across
quantities and lie within that envelope. The envelope is consistent
with the genuine physics difference between the 3-D full-fluid solves
and the Dahlen-reduced reference together with the non-neutral
resolution floor ("`fluid_core` h-ladder: residual items" below); the
2-D disc family cannot yet split these on a non-neutral core ("Which
tangential slips are removable by relabelling" below). `slip_broken`
is the closer of the two to pyslfp in absolute terms at degree 2; that
is an observation, not an attribution. What remains open is nothing
formulation-level: the aw_core comparisons (where the 2-D twin shows
all three methods coincide) are pending in 3-D.

**Practical guidance.** Shift runs (full-tolerance AL sweeps) need
per-order gauge settings: ε_g = 0.01 with 3 refinements is
semi-convergent at order 3 on this mesh; ε_g = 0.1 with ~7 refinements
converges (its final refinements hit the solver tolerance floor, which
the contraction tripwire mis-reports — see "GaugeRefine contraction
tripwire: false positive at the tolerance floor"). The old runs'
results files are kept beside the new ones with a `_geps001` suffix.

**See:** `doc/benchmarks.tex`, "The perturbation family" (Results;
"Assumptions and implementation details" for the gauge settings) and
"The formulations compared" (the decomposition caveat);
`benchmarks/perturbation/README.md`, `perturbation_check.py`.

### Broken-ζ interface forms when F jumps across Σ

**Status:** decided for the benchmarks (interface-shift maps are run with
interpolated F, forced by the driver); theory pass and implementation
audit complete 7 Oct 2026 ([f_jump_side_conventions.md](f_jump_side_conventions.md)):
the derivation needs no `[[F]] = 0` and the implemented one-sided data
are internally consistent, so no side convention is missing — the
remaining open item here is the library-level rule for exact branching
maps (below), and the shift-derivative question moves to the
discretisation/physics axes.

Interface-shift mappings are the only configurations in which the
mapping gradient F jumps across Σ (the relabelling maps have identity
gradient on every interface). The broken-ζ forms consume one-sided data:
the scalar-jump constraint kernels use b = F⁻ᵀ∇ζ⁰ and `G_Σ` uses the
gravity vector, both evaluated from the solid side. With an exact,
analytically branching shift map the evaluation is side-hazardous: on Σ
the map returns the fluid-side slope while the kernels are assembled
solid-side; on curved order-2 facets interface quadrature points lie
O(h²) ≈ 1e-2 inside the nominal radius and solid elements dipping below
it see slope kinks inside the element. The measured consequence was an
outward-shift (+ε) pathology of `slip_broken` (about 9600 iterations
against 4000–5200 for −ε, h'₁ = −72, spurious 1.5), localised in a 2-D
lab to a θ-independent floor of the ζ-jump contraction (0.175 at θ = 100,
0.178 at θ = 1000; 0.010 at ε = 0): a discretely inconsistent constraint,
not conditioning. With per-submesh interpolated F
(`MultiMeshDiffeomorphism`, `-map-interp`), which is side-consistent by
construction, +ε is healthy (4044–5244 iterations, h'₁ = −1.288, spurious
0.003–0.009). The equilibrium-geometry interface terms of Al-Attar et al.
(2018), eq. 121, are present (they are exactly `B_Σ` after the curvature
collapse) and were excluded as the cause.

Open: the physical gravity is continuous across Σ, but the default
one-sided discrete composition of b is not when F jumps. Either settle
side conventions for the broken-ζ forms that are correct for any F
(checked against the broken-ζ derivation), or provide an
attribute-keyed exact map whose interface evaluation is side-consistent.
`SetBrokenConstraintGravity` exists as a diagnostic override of b.
Analytic continuous b in the constraint kernels had no effect in the lab.

**See:** `doc/slip_interface.tex`, "Gravity: the broken-ζ organisation";
`benchmarks/perturbation/README.md`; `benchmarks/common/benchmark_case.hpp`
(shift maps force interpolated F); `benchmarks/common/relabelling.hpp`
(interface side band); `include/AdGIA/referential_problem.hpp`
(`SetBrokenConstraintGravity`).

### Which tangential slips are removable by relabelling (including N² = 0)

**Status:** open; the N² = 0 premise is confirmed numerically in 2-D
(7 Oct 2026, below), the 3-D `aw_core` degree-1 null test remains. The
reference documents state the conservative
position — tangential slip is not gauge in general; the slipping
formulation is the general one; the welded gauged formulation is used
where the suppressed slip is absent or removable, e.g. on spherically
symmetric models, where welded and slipping solutions agree to
discretisation level (1.1e-3 solid displacement on the 2-D two-layer
disc). The N² = 0 case is deliberately left open there.

*The question.* A tangential slip j on a fluid–solid interface Σ is
removable by relabelling iff it extends into the fluid as an
energy-neutral displacement field w with tangential trace j (and w·m = 0
on the fluid boundary). Which fields are energy-neutral depends on the
stratification.

*N² ≠ 0* (a second advected parameter; this includes the uniform-density
`fluid_core`). The energy-neutral displacements are divergence-free and
tangent to the level surfaces of the equilibrium potential (which are
those of pressure and density in hydrostatic equilibrium):
div w = 0, w·∇Φ₀ = 0. On a level-surface interface this restricts j: for
a spherical core, shell by shell w must be a surface-divergence-free
tangential field, so rigid rotations and toroidal slips qualify and a
general slip does not. On an interface that is not a level surface (a
general ellipse) the level surfaces cut Σ and the construction generally
fails. So slip is not gauge in general.

*N² = 0* (Adams–Williamson, one thermodynamic parameter). The
energy-neutral directions appear to be the larger set

  K = { w : div(ρ₀ w) = 0 in the fluid, w·m = 0 on its boundary },

the trivial displacements of Friedman & Schutz (1978), which need not be
tangent to the level surfaces. If so, every tangential slip is removable,
on any geometry, by an immediate lifting argument: put v = ρ₀ w; the
problem is div v = 0 in the fluid with v = ρ₀ j on the boundary (j
extended by zero where there is no slip). A boundary datum has a
divergence-free H¹ extension iff its flux through each connected
component of the boundary vanishes (Girault & Raviart 1986, Ch. I,
Lemma 2.2; Galdi, via the Bogovskii right inverse of the divergence),
and a tangential datum has zero flux through every component — also for
a shell (slip on the CMB, zero on the ICB). Then w = v/ρ₀. An equivalent
construction: extend j as w̃ with w̃·m = 0, and correct by z with
div(ρ₀z) = −div(ρ₀w̃), z = 0 on the boundary (compatibility
∮ρ₀ w̃·m dS = 0 holds).

*What remains to check* is the premise: that for N² = 0 the kernel of the
linearised referential operator is exactly div(ρ₀w) = 0 with w·m = 0, with
no further condition — in particular whether fields in K that are not
tangent to level surfaces count as relabellings (they change the
referential density and pressure fields to first order), and whether
anything in the linearised operator (the pre-stress terms) breaks the
larger symmetry. Then: characterise the removable slips for N² ≠ 0 as a
function of geometry, and decide when the welded formulation is exact
for aspherical models.

A direct numerical test is available: `examples/sliding_fluid_ellipse.cpp`
(purely elastic, gravity-free, no background pressure, where the energy
depends only on div u) states that even the uniform load drives a genuine
tangential slip on the ellipse; welded and sliding observables there can
be compared directly as a test of removability.

*2-D numerical adjudication (7 Oct 2026).* The per-degree radial
reference of the gravitating disc (`benchmarks/disc/disc_radial.py`,
the θ-reduction of the code's weak forms with welded, free-slip and
Dahlen switches) settles the mechanism on the neutral case. On
`aw_core_2d` (N² = 0) welded, slip and Dahlen agree to ~1e-10 at every
degree: welded is exact through the enlarged (class-K) relabelling
group, and Dahlen is exact because the reduction coincides with the
full static fluid at N² = 0. This is the strict prediction, and it
holds. On `fluid_core_2d` (N² < 0, l = 2, dead surface traction) the
measurement does NOT resolve the welded-suppression split: the Dahlen
solve converges (u_r(a) = −0.1782503 at p8 / p11 / p12, seven digits),
while the full-fluid welded (−0.1784474 / −0.1789929 / −0.1782873) and
slip (−0.1782625 / −0.1782305 / −0.1783079) solves wander
non-monotonically within a ~0.3 % envelope — the non-neutral resolution
floor ("`fluid_core` h-ladder: residual items"). Fixed-resolution
welded-slip and slip-Dahlen differences sit inside that envelope and
are not decompositions. The 2-D geometry also limits what can be read
across: the circle's surface-divergence-free slips are the rotations
only, so for N² ≠ 0 the strict class removes nothing generic, whereas
3-D re-weights toward the toroidal sector and the soft degree-1 mode.
Consequences: (i) the class-K removability at N² = 0 stands confirmed
at mechanism level in 2-D; (ii) on non-neutral models a welded method
compared against a Dahlen reference bundles the welded-suppression
error, the static-vs-secular closure difference and the resolution
floor, none separately resolved at current resolutions — 3-D offsets
against pyslfp previously read as one of these are not attributable;
(iii) remaining: the 3-D `aw_core` degree-1 null test (welded ≡ slip ≡
Dahlen expected), and the N²-scaling of the envelope, which the 2-D
instrument can measure but has not run.

**See:** `doc/gravitating_elasticity.md` §5, §5.1, §5.2;
`doc/gauged_fluid.md` §2 "Tangential slip and the welded space",
"Equivalence and the Adams–Williamson condition";
`doc/slip_interface.tex` (introduction);
`doc/quasi_static_models.tex`, "The mixed problem with a gauged fluid",
"The slipping-interface problem" ("When it is needed; assumptions");
`examples/sliding_fluid_ellipse.cpp`; `tests/TestSlipProblem.cpp`
(`TwoLayerBarotropicCrossCheck`); `tests/TestReferentialProblem.cpp`
(`FluidRelabellingNullPair`); `benchmarks/disc/disc_radial.py`,
`doc/planning/disc_gravity_reference.md`.

---

## The gauged fluid

### Gauged method: degree-1 interior potential

**Status:** open. The reference text states only that the interior
degree-1 response of the gauged method is not verified.

The gauged method's degree-1 φ profile departs from the reference inside
the fluid core: relative RMS deviation 0.119 (core) / 0.053 (mantle) at
`h = 0.3` and 0.120 / 0.056 at `h = 0.2` — flat in h, flat in
ε ∈ [1e-2, 1e-1] and in the number of gauge refinements — whereas
Dahlen's falls with refinement (0.013 → 0.010 in the core). φ is invariant
under exact relabellings, so this is a systematic ~12 % defect of the
gauged formulation's degree-1 interior response, not of the plotting;
surface values agree to about 1 %. Candidate: the degree-one frame
correction (applied with the background gravity of the profile layer)
against the formulation's own degree-1 terms. Review the gauged degree-1
terms before the method is used for interior fields.

**See:** `doc/benchmarks.tex`, "The Love-number family" → Results,
"Radial profiles"; `doc/gauged_fluid.md`;
`benchmarks/common/benchmark_case.hpp` (degree-one frame correction).

### Gauge refinement semi-convergence on `fluid_core` at h = 0.3

**Status:** open.

The gauged runs on `fluid_core`, `h = 0.3`, print the contraction-rate
warning of `WarnGaugeContraction`: 0.95 in `relabelled_identity` and 1.013
in the gauged field run — the iterated Tikhonov refinement is not
contracting at ε = 1e-2 on that mesh. The gauged solid-field error is
insensitive to ε ∈ [1e-2, 1] and to the refinement count, and the
identity itself is unaffected (strict, 1.6e-7). Decide whether ε = 1e-2 is
outside the window for this mesh (see [solvers.md](solvers.md), "The
gauge penalty window") and what the warning should trigger (warn only at
present; thresholds 0.9 for semi-convergence, 0.2 for residual bias).

**See:** `doc/gauge_penalty_iteration.tex`, "Semi-convergence and the
operating point"; `doc/benchmarks.tex`, "Solver settings: measured
behaviour"; `src/quasi_static_problem.cpp` (`WarnGaugeContraction`).

### GaugeRefine contraction tripwire: false positive at the tolerance floor

**Status:** open (small; library), 7 Oct 2026.

Once the Tikhonov corrections reach the linear-solver tolerance floor,
successive corrections stop shrinking and `WarnGaugeContraction` reports
a contraction near 1 — semi-convergence — although the refinement has
converged. Seen on the gauge-converged interface-shift runs
(`fluid_core`, `h = 0.3`, penalty 0.1, 7 refinements), whose final
refinements sit at the floor. Fix: ignore corrections below the solver
tolerance (relative to the solution) when estimating the contraction,
so the warning fires only on genuine semi-convergence.

**See:** `src/quasi_static_problem.cpp` (`WarnGaugeContraction`);
`doc/benchmarks.tex`, "The perturbation family" ("Assumptions and
implementation details").

### Definiteness of the mixed system: the negative control

**Status:** open (minor).

`SolverType::BlockCG` on the mixed class gives the same solution as MINRES
to 1e-11 at identical cost on a stable and a steep 2-D three-layer model
(356/358 and 376/377 iterations), supporting the reading that the
symmetric mixed system is congruent to a positive-definite one. The steep
model may not actually cross the indefiniteness threshold it was meant to
probe; `PotentialBlockMinEigenvalue` on that model would settle whether it
is a valid negative control.

**See:** `doc/self_gravitation.md` §3 "Solvers";
`include/AdGIA/mixed_problem.hpp` (`SolverType::BlockCG`).

---

## Viscoelastic problems

### Order 2 under-resolves relaxation

**Status:** open (a finding to act on in defaults, and an unexplained
convergence behaviour).

On the `fluid_core` ladder the order-2 relaxation error is not
asymptotic in h (h'₂ error 2.4e-2 at `h = 0.3`, 3.7e-2 at `h = 0.2`),
while order 3 at `h = 0.3` gives about 1e-3 (l'₄ 9.4e-4, 25× better than
order 2). The homogeneous-sphere example shows the same: the order-2
relaxed state is 12 % off by 8 τ, order 3 0.3 %. Act: use order 3 for
viscoelastic Love-number runs (`benchmarks/viscoelastic/love/
viscoelastic_love.cpp` defaults to `-o 2`; `examples/
viscoelastic_love_numbers.cpp` already uses 3). Open: why the order-2
relaxed state converges so poorly in h — the internal-variable
representation, or the near-incompressible relaxed response (the
effective shear modulus collapses while κ stays).

**See:** `doc/benchmarks.tex`, "The finite-element histories";
`doc/viscoelasticity.md` §4 "Internal variables and the strain map".

### Residual of the gravitating Maxwell Love-number comparison

**Status:** open.

Finite-element histories against the Laplace-domain reference differ by
about 1e-2 in h' and k' and up to 0.46 in l'₄ (`fluid_core`, `h = 0.3`).
Excluded: Gaver–Stehfest truncation (≤ 1e-4 on smooth spectra) and random
noise in pyslfp's transform values (bounded near 1e-12 by the n = 12
against 16 test). Not excluded: smooth systematic errors in R(s) from
pyslfp's radial-solver tolerance. Direct test: re-evaluate a few
transform values at tighter pyslfp tolerances. The h-convergence on
`homogeneous` (h'₂ 2.2e-2 → 1.0e-2 from `h = 0.3` to 0.2) already
indicates that the finite-element side dominates. The references are
valid only inside the instability horizon ln 2 / s_max set by the real
positive poles of compressible uniform layers (N² < 0): 35 τ for
`homogeneous`, 136 τ with an elastic lithosphere, 82 τ for `fluid_core`.

**See:** `doc/benchmarks.tex`, "Box: Gaver–Stehfest against exact
histories", "The Laplace-domain reference", "The finite-element
histories"; `benchmarks/viscoelastic/love/`.

### Viscoelastic Love-number driver: paths without a recorded comparison

**Status:** open.

`viscoelastic_love` accepts `-method gauged`, `-tide`, and degree 0 with a
fluid (`-lmin 0`), but no comparison against the Laplace-domain reference
is recorded for the gauged method, for the tidal histories, or for degree
0 with a fluid core.

**See:** `benchmarks/viscoelastic/love/README.md`.

### A new `ViscoelasticOperator` inherits stale relaxation weights

**Status:** open (library).

A `ViscoelasticOperator` constructed on a problem that a previous operator
has used does not clear the relaxation weights the previous operator left
in the problem's stiffness: the new operator assumes the unrelaxed
operator but the problem still assembles the old effective moduli. The
viscoelastic Love-number driver works around this by calling
`problem.ClearRelaxationWeights()` before the tidal run. Fix in the
library: clear (or verify) the weights when an operator is constructed or
first stepped.

**See:** `doc/viscoelasticity.md` §3 "Driving the operator";
`src/viscoelastic.cpp` (`UseUnrelaxedOperator`);
`benchmarks/viscoelastic/love/viscoelastic_love.cpp` (tidal run).

### Exponential trapezoid: order reduction under stiff stress control

**Status:** open (characterised, not analysed).

Under stress control on multi-branch bodies with τ ≪ Δt the exponential
trapezoid drops to order ≈ 1.5 while SDIRK23 keeps 2: box `prony_wide`
orders 1.52 (periodic) / 1.48 (Heaviside); slab `prony_lid` SDIRK23 16×
more accurate at the same step; sphere `prony_lid` and `burgers_lid` show
time-error floors. Proposed mechanism: the stiff branches put components
into δ(t) that vary inside a step, which the scheme's linear
interpolation misses. Decide whether to analyse it or to recommend SDIRK23
by default for stiff multi-branch rheologies.

**See:** `doc/benchmarks.tex`, "Box: the integrators on a homogeneous
body", "Two floors that are time error"; `doc/viscoelasticity.md` §3
"Choosing a scheme".

### `self_gravitating_relaxation`: no visible approach to isostasy

**Status:** open (check).

The example (order 1, 10 steps, 2-D two-layer) shows the pole displacement
and ‖u‖ still growing almost linearly at 5 Maxwell times. This may be
correct for the model's timescales; it has not been checked.

**See:** `examples/self_gravitating_relaxation.cpp`.

---

## Benchmark results not yet explained

### `fluid_core` h-ladder: residual items

**Status:** partly resolved (the angular-cap arithmetic is confirmed; the
uncapped ladder is run and documented).

1. The geometry-order-3 ladder (`fluid_core_geom3`) was *less* accurate at
   displacement order 3 than geometry order 2 at the same h and showed no
   drop at `h = 0.12`; unexplained, not rerun uncapped (`--angular 1.0`)
   on the current code.
2. The capped ladder's order-3 plateau followed by a drop at `h = 0.12` is
   explained in part by the cap (CMB elements fixed at 0.165 for
   h ≥ 0.165), but not the size of the drop; candidates not separately
   tested: DtN degree 16, the reference, the geometry.
3. The uncapped ladder has a non-monotone rung (`h = 0.15`), and the
   degree-2 Love-number column falls at an implied rate of about 6
   (cancellation); field rates 1.6–2.7 are not a clean asymptotic regime.
4. The uncapped ladder has no campaign stage (see "Benchmark
   housekeeping" below).

*Candidate explanation for 1–3 (7 Oct 2026): a resolution floor by
physics.* On `fluid_core_2d` (N² < 0) the 1-D radial reference's l = 2
response drifts ~3e-3 non-monotonically under p-refinement and moves
~1e-3 with the fluid regulariser, while `aw_core_2d` (N² = 0) converges
to ~1e-12. For N² ≠ 0 the static operator is inverted at the edge of
its essential spectrum: the full-elastic fluid's interior slaving
u_r = −φ/g conflicts with normal continuity at the interface, and the
conflict is resolved in a boundary layer whose discrete representation
never settles. The Dahlen-reduced solve on the same disc converges (its
closure is well-posed); the welded and slip full-fluid solves wander
within a ~0.3 % envelope. Caveat: the ladder above is the Dahlen path, and
the 2-D Dahlen solve converges; whether the floor reaches the 3-D
Dahlen path, or only the full-fluid methods, is part of what the 3-D
test (the same ladder on `aw_core` against `fluid_core`) decides. The
2-D floor is recorded (loosely bounded, strict on the neutral twin) in
the validation of `benchmarks/disc/disc_radial.py`
(`python disc_radial.py`).

**See:** `doc/benchmarks.tex`, "Resolution and the h-ladder", "Results";
`benchmarks/disc/disc_radial.py` (`validate`).

### Combined-degree solves: unverified paths

**Status:** open (minor).

The combined-degree mode (`love_benchmark -combined`, `run.py
--combined`) resets the potential between the load and the tidal solve by
a zero-forcing solve. This is verified for block MINRES
(`SetWarmStartTolerance` zeroes the block iterate); the Schur-CG path is
unverified and relies on the zero guard in `SolvePotential`. The measured
inter-degree leakage (2e-5 to 4.5e-3 relative, tolerance-independent) is
below the mesh error everywhere except one cancellation case (gauged
tidal h at degree 3, whose mesh error is anomalously small, 2e-4).

**See:** `benchmarks/love_numbers/README.md`; `doc/benchmarks.tex`,
"Forcings and the Love numbers read from a 3-D solution".

### `methods.png`: missing referential degree-4 point

**Status:** open (minor, uninvestigated).

The referential series of the cross-method figure (`methods.png`) lacks
its degree-4 point; the gap predates the combined-degree mode.

**See:** `benchmarks/talk_figures.py`; `benchmarks/campaign.py`.

---

## Known code and build defects

### `viscoelastic_loading`: the load switch-off

**Status:** open.

The step ending at `t_load` is taken with the load already off at its end,
so the jump is spread over one step (an O(dt) error in the history near
`t_load`; the example's header says so). Fix: make the load test inclusive
at `t_load` and call `ViscoelasticOperator::InvalidateDisplacement()` at
the jump, as the box benchmark driver does.

**See:** `examples/viscoelastic_loading.cpp`; `doc/viscoelasticity.md`
§3 "Driving the operator".

### Build system

**Status:** open.

- `CMakeLists.txt:47-49`: `if (NOT CMAKE_CXX_COMPILER AND
  MFEM_CXX_COMPILER)` comes after `project()`, where `CMAKE_CXX_COMPILER`
  is always set, so it never fires.
- `CMakeLists.txt:215`: the Doxygen install copies `<build>/doc` into
  `share/<project>/html`, nesting `html/doc/html`; probably meant
  `<build>/doc/html`.
- `meshes/make_all.py` omits `disc_with_wide_buffer.py` and
  `aspherical_body.py --all`.

**See:** `CMakeLists.txt`; `meshes/make_all.py`; `meshes/README.md`.

### Code questions from the library comment pass

**Status:** open; each needs a decision (all are code changes, none made).

1. **`SetVacuumExtension` default** (`referential_problem.hpp`):
   `refinements = 3`, but `doc/gauge_penalty_iteration.tex` ("When it must
   fail: the spectral condition") shows refinement diverges for the
   ball-wide buffer. Default to 0?
2. **`EnableBrokenZeta` after `EnableKKT`**: `EnableKKT` refuses broken ζ,
   but `EnableBrokenZeta` does not check `kkt_`; in that order the broken
   organisation runs silently. Refuse both ways?
3. **`EnableGaugeKKT` is never covariant**: it calls `SetGaugedFluid`
   with `map = nullptr`. Intended?
4. **`ClearGaugedFluid()`** does not reset `gauge_lambda_eps_` /
   `gauge_Cdev_` (harmless: reassigned on the next `SetGaugedFluid`).
5. **`SphericalMeshHelper`** (`src/mesh.cpp:243, 252`) checks sphericity
   with `assert` only (skipped in release builds): `MFEM_VERIFY`?
6. **`PoissonDtNOperator`** creates `bdr_comm_` with `MPI_Comm_split`
    and never frees it.
7. **Manifest fluid rule** (`src/mesh_manifest.cpp:256`): a layer listed
    in `meta.fluid_layers` is fluid even when `layers[].fluid` is false
    (the two are combined). Should the list apply only where
    `layers[].fluid` is null?
8. **Doxygen and out-of-class definitions**: the "no matching class
    member" warnings in `src/` are silenced with `/// @cond` around five
    constructor and three `Coefficients` definitions (their types are
    unqualified under `using namespace mfem`); qualifying the types
    (`mfem::`) would let Doxygen list them instead.
9. **Small documentation questions**: the use to name for
    `DomainLFDeformationGradientIntegrator` (only tests use it; the
    comment says a prescribed-stress source); the reason
    `RelabelledBackground` requires ξ to be the identity at and outside
    the surface; the equation number cited in `coefficient.hpp:81`
    (Al-Attar & Tromp 2014, eq. 2.8) for the fluid mass term.

---

## Benchmark housekeeping

### Benchmark runs and outputs

**Status:** open.

1. Runs documented but made by hand, not by the campaign: the all-model
   sweep (`h = 0.2`, orders 2 and 3, eight models), the uncapped
   `fluid_core` ladder, the `prem_4` CMB runs (and gauged), the aspherical
   amplitude sweep (`talk_data/`) and the dense viscoelastic histories
   (`viscoelastic_series/`). Campaign stages (or a `--stages docs` group)
   would make the whole documentation reproducible with one command.
2. ~~Scripts that write relative to the working directory without the
   source-tree guard.~~ Done (6 Oct 2026): every benchmark script now
   passes its output root through `common/outputs.py::outside_source`.
   Regenerating the committed references of
   `viscoelastic/love/references/` is now a build-tree write plus a
   copy.

**See:** `benchmarks/campaign.py`; `benchmarks/common/outputs.py`;
`doc/benchmarks.tex`, "Reproducing the figures".

### Numbers in `doc/benchmarks.tex` without a stored run tree

**Status:** open; rerun into the build tree (or add a script that
regenerates them) when convenient.

Kept from written sources with their run settings but not re-measured
during the rewrite: the gauge-ε window (bias 2.6e-2 at ε = 1e-1;
3.7k / 6.3k / 28.6k / 138k iterations; h'₂ = −3.17 at 1e-4;
`fluid_core`, `h = 0.3`); the AL/KKT measurements (θ cliff;
16.9k / 27.7k / 49.5k; 22.9k; 79.4k; the AL-against-KKT table); the
gauge-KKT 30k iterations; the combined-solve numbers
(`love_numbers/README.md`); the 2-D Schur/MINRES timing
(`doc/self_gravitation.md`); the 3 % near-null leakage and the 6e-5 AL
floor (2-D test meshes). The prestress ellipse table and every number of
the CMB, ladder, model-sweep, cross-method, identity and perturbation
tables were re-read from current runs.

**See:** `doc/benchmarks.tex`, "Solver settings: measured behaviour",
"Measurements from the examples and the unit tests".

---

## Editorial

### Bibliography entries to verify

**Status:** partly checked against the PDFs in `doc/Elasticity/` and
`doc/BenchmarkPapers/`.

Verified from the PDFs: Al-Attar, Wahr & Zhong 2013; Al-Attar et al. 2018;
Al-Attar & Crawford 2016; Al-Attar & Woodhouse 2010; Huang et al. 2023;
Latychev et al. 2005; Maitra & Al-Attar 2021, 2024; Martinec 1999; Spada
et al. 2011 (full author list); CitcomSVE-3.0 (pages); Yu et al. 2025;
Woodhouse & Deuss 2007 (start page 31; end page 65 and the editor from the
printed headers).

Not verifiable from the repository: Al-Attar & Tromp 2014; Crawford et al.
2017; Dahlen & Tromp 1998; Dahlen 1974; Engl, Hanke & Neubauer 1996;
Fortin & Glowinski 1983; Friedman & Schutz 1978; Golub & Greif 2003;
Hestenes 1969; Lardy 1975; Marsden & Hughes 1983; Martinec 2000; Murphy,
Golub & Wathen 2000; Powell 1969 (pages missing: pp. 283–298 in Fletcher
(ed.)); Simo & Hughes 1998; Tromp & Mitrovica 1999; Valette 1986;
Valette 1991 (only the first page, 555, is known; title as in
`doc/gravitating_elasticity.md`); Wohlmuth 2001; Wu & Ni 1996; Wu &
Peltier 1982; Zhong et al. 2003. Girault & Raviart 1986 and Galdi (cited
above for the lifting lemma) are not in the bibliographies.

Decided: the Valette (1986, 1991) PDFs are not needed in
`doc/Elasticity/` — the slip-interface derivation covers their content
(the Weingarten-operator treatment of the interface terms) in greater
generality; the citation suffices. Only their bibliographic details
remain to check.

**See:** bibliographies of `doc/quasi_static_models.tex`,
`doc/slip_interface.tex`, `doc/benchmarks.tex`; sources list of
`doc/gravitating_elasticity.md`.
