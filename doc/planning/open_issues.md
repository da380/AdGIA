# Open issues: correctness and verification

Defects, unverified results, open theoretical questions and review items.
Measurements are on the benchmark model `fluid_core` (uniform-density
compressible fluid core, `N² < 0`) unless stated otherwise; `h` is the
nominal mesh size of the benchmark meshes, "order" the displacement order.

---

## The slipping interface

### Mapped `slip_broken`: the slipping-interface forms are not covariant

**Status:** open. Mapped slipping-interface results are unverified; this
blocks slipping-interface topography and aspherical slip runs.

Two distinct symptoms, both on `fluid_core` with the lateral relabelling
of amplitude `A = 0.02`:

1. **Interpolated F (relabelling family, Leg B,
   `relabelled_identity -method slip_broken`).** The discrete
   change-of-variables identity, which holds to about 1e-6 on every
   welded case including the gauged fluid (welded gauged: u 1.6e-7,
   ζ 4e-8 at `h = 0.3`), fails through the slipping interface:
   δu = 1.2e-2, δζ = 8.1e-3 at `h = 0.3`, and 4.5e-2 / 5.6e-2 at
   `h = 0.2`. It grows under refinement, so it is a genuine defect, not
   interpolation noise. The driver's block probes localise it: of the
   sixteen broken blocks every ζ–ζ block agrees to machine precision and
   every u-involving block differs at 1–2.5e-4. The constraint kernels
   are exonerated: `B_n` 2.2e-16 (the Nanson normal ν = adj(F)ᵀN is
   invariant under face-fixing shears of F), `P_b` 1.5e-7, `K_vz`
   9.5e-8; the base rows and the C couplings are certified by the
   welded identity. By elimination the non-covariant content is in
   `B_Σ` / `G_Σ` themselves: their `P_T F⁻¹` slot and the gravity
   vector are not invariant under face-fixing shears of the interpolated
   F, which MFEM evaluates from the adjacent volume element. The
   relabelling is the identity on the interfaces (no jump of F across
   Σ), so this is unrelated to the side-selection defect of the shift
   maps (next items).
2. **Exact F (relabelling family, Leg A, `love_benchmark -map 0.02`).**
   Lateral leakage in the Love numbers: the `spurious` column is
   0.36–0.4 at degree 1 and 6–9e-2 at degree ≥ 2 on the current code
   (an earlier build gave 0.63 / 6–7e-2), against 6e-3–1e-2 unmapped and
   2–8e-3 for the mapped referential method. A separate mechanism is
   suspected: `G_Σ` sits inside a sharp degree-1 cancellation —
   `-gs-scale 0.5` raises the *unmapped* degree-1 spurious value to 0.46
   and makes the mapped solve diverge — so the mapped leakage is
   plausibly a small relative perturbation of `G_Σ`, strongly amplified.
   The rigid null-pair residuals of the slip class (`love_benchmark
   -diag`) survive the map essentially unchanged (translations 2.4e-3
   vs 2.6e-3; rotations about doubled, ~1e-3).

Which forms are uncertified: the KKT constraint row is covariant by
construction (Nanson-exact flux kernel); the mapped broken solver
reproduces the identity solution under an interface-fixing twist
relabelling to 5e-4 / 2e-4. Uncertified: `B_Σ`
(`SlipInterfacePressureIntegrator`), `G_Σ`
(`SlipInterfaceGravityIntegrator`, `SlipInterfaceGravityScalarIntegrator`),
and possibly the AL normal–normal penalty with its 1/|ν| factor. The
single-valued organisation's mismatch gravity pieces are assembled at
φ_e = id and refuse maps outright (`Diffeomorphism::IsIdentity` guard); that
limitation is separate.

Proposed instruments: a `B_Σ` counterpart of the `-gs-scale` hook; a
difference-field map of mapped-vs-unmapped solutions; an element-level
identity test for each interface form on a mapped mesh, like those of the
domain and Nanson boundary integrators; a theory pass on the mapped
`B_Σ`/`G_Σ` derivation, using the degree-1 cancellation to identify the
combination the mapped assembly must preserve. Decide afterwards which
forms are certified and update the reference docs (they currently call
mapped slipping results unverified).

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

**Status:** open (small).

(1) The slipping class records no per-refinement gauge residuals
(`GaugeResiduals()` stays empty) and exposes no way to apply its fluid
gauge penalty, so the gauge convergence of the slip path cannot be
inspected as for the welded classes. (2) `NewRadialVacuumExtension`
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

**Status:** open; probably the same defect as the previous item, to be
covered by the same diagnostic.

On `fluid_core`, `h = 0.3`, order 2, shifts ε = ±0.02 of the CMB
(interpolated-F shift maps, exact sweeps), the derivatives of the Love
numbers with respect to the interface radius disagree with the 1-D finite
difference of pyslfp solves far more for `slip_broken` than for the
referential method on the same mesh and map:

| | h'₂ | l'₂ | k'₂ | h'₃ | l'₃ | k'₃ |
|---|---|---|---|---|---|---|
| `slip_broken` | 25 % | 12 % | 19 % | 5.6 % | 23 % | 4.9 % |
| referential | 6.4 % | 12 % | 1.7 % | 1.9 % | 14 % | 0.8 % |

An earlier build gave k'₂ 39 %, k'₃ 11 %, l'₂ 2.8 % for `slip_broken`: the
numbers move between builds, itself a symptom. Degree 0 agrees for both
methods (0.8 %). The two methods share the shift map and the reference,
so the difference sits in the slipping forms under a moving interface.
The degree-2 rows of both methods also carry the ~17 % method-independent
offset that is the O(N²) stratification mismatch between the 3-D
compressible fluid and pyslfp's neutral fluid (see
[future_work.md](future_work.md), "Neutral (Adams–Williamson) core
model").

**See:** `doc/benchmarks.tex`, "The perturbation family" (Results);
`benchmarks/perturbation/README.md`, `perturbation_check.py`.

### Broken-ζ interface forms when F jumps across Σ

**Status:** decided for the benchmarks (interface-shift maps are run with
interpolated F, forced by the driver); open in the library (the side
conventions of the broken-ζ forms under a jump of F are not settled).

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

**Status:** open. The reference documents state the conservative
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

**See:** `doc/gravitating_elasticity.md` §5, §5.1, §5.2;
`doc/gauged_fluid.md` §2 "Tangential slip and the welded space",
"Equivalence and the Adams–Williamson condition";
`doc/slip_interface.tex` (introduction);
`doc/quasi_static_models.tex`, "The mixed problem with a gauged fluid",
"The slipping-interface problem" ("When it is needed; assumptions");
`examples/sliding_fluid_ellipse.cpp`; `tests/TestSlipProblem.cpp`
(`TwoLayerBarotropicCrossCheck`); `tests/TestReferentialProblem.cpp`
(`FluidRelabellingNullPair`).

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

**See:** `doc/benchmarks.tex`, "Resolution and the h-ladder", "Results".

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
2. Scripts that write relative to the working directory without the
   source-tree guard: only the viscoelastic box and sphere scripts pass
   output paths through `common/outputs.py`. `love_numbers/run.py`
   (`--out runs`), `campaign.py` (`--out runs_campaign`),
   `viscoelastic/stepping/survey.py` (`--out survey`), `talk_figures.py`
   (`--out talk`), `perturbation/perturbation_check.py` (default beside
   the case), `viscoelastic/love/laplace_reference.py` / `compare.py` and
   `common/make_case.py` write wherever they are started, including into
   the source tree. Apply `outside_source` to their defaults; outputs
   belong in the build tree.

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
