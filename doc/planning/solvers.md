# Solver and performance plans

Unless stated otherwise, measurements are on `fluid_core`, `h = 0.3`,
order 2, run in parallel, with pyslfp as the external reference.
Production runs are parallel: solver decisions are taken on parallel
measurements, and the serial path is kept working but is not tuned.

Cost per load solve on that case (current code), for orientation:

| method | wall | iterations |
|---|---|---|
| Dahlen | 0.55 s | 117 |
| gauged | 4.1 s | 783 |
| referential | 5.8 s | 802 |
| slip | 17 s | 1306 |
| slip_broken | 23 s | 2714 |

The gauged, referential and slipping methods pay the fluid-gauge premium
(about 7× Dahlen), the slipping methods in addition the interface
constraint.

---

## The slip constraint

### AL and KKT: both paths are kept

**Status:** decided — both enforcement paths stay, behind one interface;
the AL path is the default. Open: the KKT path for the broken-ζ
organisation, and a remaining multiplier-block preconditioner variant.

The two paths solve the same discrete problem and agree within
discretisation error: degree-0 h' identical to 2e-7 with identical error
against pyslfp (5.87e-4); endpoint differences between sweep counts and
paths (up to 2e-2 at degree 1) lie below the mesh error. The KKT path
gives an exact, θ-independent constraint (endpoint invariant to 5e-10
over θ ∈ [1, 300], a sharp regression check), the multiplier as a solved
field, and an instrument for constraint-rank questions; the AL path
serves both organisations.

Measured settings (θ = 100 throughout; KKT with an order-1 multiplier at
displacement order 2):

| solver | iterations | wall |
|---|---|---|
| AL, 3 sweeps, exact | 4.4k | 76 s |
| AL, 8 sweeps, exact | 11.4k | 200 s |
| KKT, order-1 multiplier, 3 refinements, exact | 10.7k | 178 s |
| KKT, same, inexact refinement sweeps | 7.4k | 124 s |

What has been established about the settings:

- *θ.* The dimensional argument θ ~ C/h (≈ 10 here) points the wrong
  way: θ = 10 ran more than 39 minutes against about 4 at θ = 100
  (stopped unfinished). The penalty also regularises the near-null
  relative normal motion at the interface, so there is a conditioning
  cliff below θ ≈ 100 and a flat plateau above (θ = 300: 11344 against
  11364 iterations, the constraint converging further in the same sweeps).
  In the KKT path, too, lowering θ hurts monotonically (16.9k / 27.7k /
  49.5k iterations at θ = 100 / 10 / 1; θ = 300 gains 5 %), because the
  augmentation regularises the Krylov iteration beyond Schur clustering.
- *KKT augmentation.* Augmenting with the consistent
  θ Cᵀ M̂⁻¹ C instead of the AL penalty's full L² trace form removes a
  kernel mismatch between equal-order traces and the multiplier space:
  22.9k → 16.9k iterations, endpoint moved 7e-6.
- *Multiplier-block scaling.* The Schur complement is ≈ (1/θ) M_Σ, so
  the multiplier block must be preconditioned by θ M̂⁻¹ (entries θ/mᵢ);
  θ M̂ mis-scales it by θ² m̂² (about 1e4 at `h = 0.3`). Correcting it cut
  79.4k to 22.9k iterations.
- *Multiplier order.* One below the displacement (`-kkt-order 1`, the
  mortar-style choice): 16.9k → 10.7k iterations, endpoint shift 4e-4,
  below the mesh error. Recommended benchmark KKT settings:
  `-kkt -kkt-order 1 -al 3`, θ = 100.

Open:

- A CG-on-mass (rather than lumped-mass) multiplier-block preconditioner;
  likely marginal now.
- KKT for the broken-ζ organisation: `EnableKKT` and `EnableBrokenZeta`
  are mutually exclusive; a second multiplier for the scalar-jump
  constraint ⟦ζ¹⟧ = b·⟦v⟧ would allow the broken organisation to be
  solved monolithically too.
- The KKT multiplier against the AL `ConstraintMultiplier()` is a free
  consistency observable not yet used in any test.

**See:** `doc/slip_interface.tex`, "Constraint enforcement" ("KKT
enforcement", "Choosing between the two paths"); `doc/benchmarks.tex`,
"Solver settings: measured behaviour";
`include/AdGIA/referential_problem.hpp` (`EnableKKT`,
`EnableBrokenZeta`, `KKTMultiplier`, `ConstraintMultiplier`);
`benchmarks/common/benchmark_case.hpp` (`-kkt`, `-kkt-order`, `-al`).

### Default number of AL sweeps

**Status:** open; library default `al_iterations_ = 8` and benchmark
default `-al 8` unchanged.

At `h = 0.3`, 3-sweep AL runs are as accurate against pyslfp as 8-sweep
ones (at degrees 1 and 2 they are closer: 2.5–3.1e-2 against 5e-2), at
4.4k against 11.4k iterations (76 s against 200 s). AL-3 and KKT-3 agree to
5e-3; AL-8 and AL-3 differ by 2.5e-2 at degree 1, all inside the mesh-error
band. So the default oversolves at this resolution. Re-examine at finer h
before changing it: the AL constraint floor and the mesh error fall at
different rates.

**See:** `doc/slip_interface.tex`, "Penalty and augmented Lagrangian,
interleaved with the gauge refinement";
`include/AdGIA/referential_problem.hpp` (AL settings).

### Inexact sweeps

**Status:** decided and implemented (`SetSweepTolerance`; library default
exact, benchmark default `-sweep-tol 1e-3`, exact mode forced on
`-map-shift` runs). Open: whether the library default should become
inexact.

The schedule tightens the inner tolerance geometrically in the sweep
index from the loose value to the solver tolerance, with the final sweep
at full tolerance and warm-started sweeps on an absolute target anchored
to the first sweep's initial residual. Two alternatives were tried and
rejected: an inner tolerance coupled to the measured jump deadlocks (the
measured jump cannot fall below the solver-error floor the loose
tolerance sets, the multiplier error is amplified by θ, and the Love
numbers ended 1e-2 off despite 6× fewer iterations; small serial tests
pass either way, the trap shows only on the 3-D case); and a
plateau-based early exit triggers falsely on the same noise (healthy AL
contracts about 0.6 per sweep) and truncates multiplier updates the AL
contraction needs. Measured effect: endpoint shift about 1e-3 of the
spread between exact endpoints, roughly half the cost.

**See:** `doc/slip_interface.tex`, "The sweep schedule";
`src/referential_problem.cpp` (sweep loop).

### Multiplier warm starts

**Status:** proposal.

The AL multiplier (and the KKT multiplier) may be carried over between
solves with the same constraint and correlated forcing — across
viscoelastic time steps — but not across degrees in a Love-number sweep,
whose forcings are independent (the same trap as `ResetSolution`). Not
implemented; it becomes relevant once the slipping classes are driven by
`ViscoelasticOperator`.

**See:** `include/AdGIA/referential_problem.hpp`
(`ResetSolution`, `ConstraintMultiplier`).

---

## The fluid gauge

### The gauge penalty window

**Status:** decided — ε = 1e-2 with 2–3 iterated Tikhonov refinements is
the production operating point; no further ε study unless the
contraction warning fires on finer meshes. Open: the semi-convergence at
`fluid_core h = 0.3` ([open_issues.md](open_issues.md)).

Measured (gauged, three refinements):

| ε | iterations | h'₂ (load) |
|---|---|---|
| 1e-1 | 3.7k | −0.9745 (bias 2.6e-2 survives the refinements) |
| 1e-2 | 6.3k | −0.9894 |
| 1e-3 | 28.6k | −1.0007 |
| 1e-4 | 138k | −3.17 (semi-convergent: polluted although every solve reports convergence) |

So the penalty window is two-sided: O(ε) bias above, about 4.5× cost per
decade and semi-convergent pollution below; ε ≈ 1e-2 is cost-forced. It
also serves PREM structure (`prem_4`, `h = 0.2`: h'₀ within 6e-4 of
pyslfp). `WarnGaugeContraction` (called from all three `GaugeRefine`
implementations) guards finer meshes: warn-only, thresholds 0.9
(semi-convergence) and 0.2 (residual bias).

Alternatives considered and not pursued: a mixed (u, p) fluid block, i.e.
a pressure-like auxiliary field making the fluid block a Stokes-like
saddle with a parameter-robust block preconditioner (it would take ε out
of the conditioning for the gauged, referential and slipping methods);
Schöberl-type parameter-robust multigrid (heavy without MFEM support).
Changes to the fluid formulation are not pursued — the penalty with
iterated refinement is the accepted formulation; only numerical levers
(the next items) remain open.

**See:** `doc/gauge_penalty_iteration.tex`, "Semi-convergence and the
operating point"; `doc/gauged_fluid.md` §1 "Gauge fixing: penalty plus
iterated refinement"; `doc/benchmarks.tex`, "Solver settings: measured
behaviour"; `src/quasi_static_problem.cpp` (`WarnGaugeContraction`).

### Solid-everywhere preconditioner for the clean gauged system

**Status:** IMPLEMENTED, MEASURED AND MADE HANDS-OFF 7 Oct 2026 —
with the stagnation stop (below) one setting serves every degree at
7–10× over the penalty path with penalty-grade quality, degree 1
included. EXTENDED TO THE SLIP CLASSES 8 Oct 2026 (next paragraph).
Open items: the `prem_4` near-neutral check on the server profile,
and the default-ness call (finer-h validation before any production
default changes, as for AL-3).

**The slip classes** (8 Oct 2026): the slip override of
`SetGaugePreconditionerOnly` keeps the AL constraint penalty θB_n in
the operator, drops εQ_f from the (1,1) operator block only (the
fluid-block smoother/AMG stays on the +Q_f matrix), and drops the
per-sweep Tikhonov term; both organisations, serial and parallel;
EnableKKT refuses the mode. Measured (`fluid_core` h = 0.3, order 2,
np 8, `-gpe 1 -gps 0.7 -rt 1e-8`): slip 425–450 its per degree vs
513–1674 AL-8 penalty, slip_broken 475–525 vs 2462–3036 (≈ 4–6×
iterations, ≈ 3–4× wall), spurious 0.004–0.008 vs 0.003–0.015.
TWO ESSENTIAL FINDINGS: (1) the plateau stop is LOAD-BEARING for
slip, not an optimisation — without it the clean-operator sweeps
semi-converge catastrophically (rt 1e-4: 55k iterations, h'₂ −0.42,
spurious 2.5); (2) the endpoints move by up to 3e-2 from the AL-8
penalty answers — but this is WITHIN the slip method's own AL
enforcement floor at this mesh: against the θ-independent KKT
reference (h'₂ = −0.99299, 1707 its) the AL-8 penalty sits 1.4e-2 on
one side and gpe+plateau 1.6e-2 on the other. The two modes bracket
the discrete solution equally well; KKT remains the precision
referee. One blemish to watch: slip_broken degree-0 k' comes out
−2.4e-3 under gpe (penalty: −7e-6, physical ≈ 0) — the degree-0
scalar-jump channel is plateau-sensitive; keep the penalty path for
slip_broken l = 0 until understood.

**The stagnation stop** (`SetGaugePlateauStop(ratio, chunk)`,
benchmark `-gps`): the block solve runs as warm-started MINRES
restarts of `chunk` (default 25) iterations and stops when a chunk's
final residual exceeds `ratio` times the previous chunk's — the
plateau at the load's near-kernel content, which is the intended
stopping point. HANDS-OFF RECIPE, measured on `fluid_core` h = 0.3,
order 2, np 8, degrees 0–4 (`-gpe 1 -gps 0.7 -rt 1e-8`):

- gauged: 75–100 its per degree against 754–816 penalty (sweep 475 vs
  ~3900 its), every Love number within a few e-4 of the penalty
  truth, spurious at or below penalty levels at every degree;
- referential: 75–150 its against 786–885 (7× wall), spurious
  IDENTICAL to the penalty path degree by degree;
- degree 1, the tolerance-stopped mode's failure case, is FIXED by
  the stop + cross-degree warm start: 50–75 its, spurious 1.6–2.1e-3
  (penalty: 2.0–2.3e-3) — no penalty fallback needed;
- ratio sensitivity: 0.3 and 0.5 stop one chunk earlier and leak at
  high degrees (spurious 0.13–0.14 at l = 4); 0.7 is the measured
  default. The earlier manual-tolerance scan (kept below for the
  record) is superseded by the stop.

*Implementation.* `SetGaugePreconditionerOnly(eps_prec)` on the base
class (call after `SetGaugedFluid`): the gauge ε becomes the
preconditioner value, the refinements are disabled, and the
block-MINRES setups of the mixed and referential classes put the CLEAN
A in the (0,0) operator block while `prec_` is built on
A + ε_prec Q (the referential class augments both by EᵀGE). The Schur,
base-CG and gauge-KKT paths refuse the mode. Benchmark flag `-gpe`.

*Measured (`fluid_core`, h = 0.3, order 2, np 8, loads+tides).*
Penalty production (ε = 1e-2, 3 refinements): 754–816 its, 3.3–7.0 s
per degree. Solid-everywhere, ε_prec = 1:

| degree | rt | its | s | endpoint vs penalty truth |
|---|---|---|---|---|
| 0 | 1e-6 | 79 | 0.71 | h'₀ to 2e-4 (1e-5: 32 its, 2e-4) |
| 1 | 1e-4 | 1665 | 7.4 | h'₁ 0.6 %, spurious 4.1e-3 — NO WIN |
| 2 | 1e-3 | 49 | 0.45 | h'₂ 1.6e-3, spurious identical |
| 3 | 1e-3 | 53 | 0.43 | h'₃ 5e-4 |
| 4 | 1e-3 | 40 | 0.34 | h'₄ 4e-4, spurious 0.048 vs 0.023 |

ε_prec ∈ {1, 0.3, 0.1} barely matters (49/55/87 its at l = 2): the
plateau, not the condition number, is the controlling feature, as the
analysis below predicted. Over-solving is the failure mode on cue:
l = 2 at rt 1e-5 burns 5911 its and semi-converges into lateral junk
(spurious 0.13); l = 1 pollutes already at rt 1e-5 (0.45). The
referential class: 885 → 100 its (4.9 → 0.93 s) at l = 2, endpoint to
1.9e-4. Degree 1 is the one load whose near-kernel (rigid-adjacent)
content is large, exactly the known worst case of the penalty
analysis; the penalty path stays the degree-1 production answer.
Production recipe candidate: gpe at rt 1e-3 for l ≥ 2, rt ~1e-5–1e-6
for l = 0, penalty for l = 1 — pending the stagnation stop and the
`prem_4` check.

*Idea.* Solve the clean system A u = f (no ε in the operator) with
MINRES preconditioned by L = A + ε_prec Q — the model made solid
everywhere by giving the fluid the gauge shear modulus μ_g — with ε_prec
of order one.

*Analysis* in the generalised eigenbasis A v = λ Q v. The preconditioned
operator (A + ε_prec Q)⁻¹A has eigenvalues λ/(λ + ε_prec):

- solid-dominated modes (Q vanishes on the solid) → 1;
- physical fluid modes lie in [λ_min/(λ_min + ε_prec), 1]; the measured
  refinement contraction ρ ≈ 0.1 at ε = 1e-2 with μ_g = κ_f puts
  λ_min ≈ 0.1, so the condition number is about 10 at ε_prec = 1;
- exact gauge modes have eigenvalue 0 and zero right-hand side, and MINRES
  with an SPD preconditioner ignores them;
- near-gauge modes (λ ~ h^p, with right-hand-side content f ~ h^p) keep
  eigenvalue ≈ λ.

*Relation to production.* Iterated Tikhonov refinement *is* Richardson
iteration on A u = f preconditioned by A + εQ, contracting mode-wise by
ε/(λ + ε). Richardson forces ε small, and small ε is what makes the solves
expensive: the fluid block at κ/(εμ_g) = 100 is nearly incompressible,
and AMG on A + εQ takes about 800 iterations against Dahlen's 120. Krylov
needs the spectrum only away from 0, so ε_prec can be order one, where the
fluid in A + Q has μ = κ (Poisson ratio ≈ 0.13), an easy solid. ε_prec is
a knob between the two regimes.

*Caveat.* The near-gauge modes move from the operator into the stopping
rule. The clean solution carries fᵢ/λᵢ on them; early-stopped Krylov
regularises, with junk ≈ fᵢ |p_k′(0)| — a few hundred fᵢ for k ~ 30,
comparable to the three-refinement production path's 3fᵢ/ε = 300 fᵢ.
Expected history: the residual falls to the near-kernel content of f
(about 1e-3–1e-4) in tens to a hundred iterations, then plateaus, then
drifts. (Direct MINRES on the singular system *to convergence* is a dead
end — it re-derives semi-convergence; early-stopped Krylov is a different
question.)

*Questions.* Is the plateau clean enough for a stagnation-based stop
(the `WarnGaugeContraction` logic inverted)? Do plateau observables match
the penalty endpoint at mesh level?

*Measurement plan.* Split ε into an operator value (0) and a
preconditioner value ε_prec in `SetGaugedFluid` / the MINRES setup (A, Q,
A + εQ and its preconditioner already exist; a handful of lines plus a
benchmark flag). On `fluid_core`, `h = 0.3`, order 2, parallel: iteration
history and plateau level for ε_prec ∈ {1, 0.3, 0.1}; endpoint Love
numbers against the penalty endpoint (h'₂ = −0.989424 at degree 2, 819
iterations) and pyslfp; repeat on `prem_4` (near-neutral core). Success =
a clean plateau at about 1e-4, which would mean a 5–10× saving on every
gauged, referential and slipping solve.

**See:** `doc/gauge_penalty_iteration.tex` ("Convergence, mode by mode",
"Semi-convergence and the operating point"); `doc/gauged_fluid.md` §1;
`include/AdGIA/quasi_static_problem.hpp` (`SetGaugedFluid`);
`src/quasi_static_problem.cpp` (`WarnGaugeContraction`).

### Exact-gauge KKT

**Status:** parked. Implemented on the mixed class
(`EnableGaugeKKT(fluid_marker, mu_gauge, mgw_prec)`, `KKTResiduals()`,
benchmark `-gauge-kkt 0|1|2`), correct, not competitive; the penalty path
is production.

The gauge is fixed exactly by minimising ½(Qu, u) subject to the full
coupled equations A u = f, with Q the unit-scale deviatoric gauge form:
the doubled saddle [diag(Q, 0), A; A, 0][u; v] = [0; f]. No ε, no bias, no
refinements. The iterates reach mesh-level physics early and satisfy
stationarity to 1e-5, but MINRES does not converge algebraically: about
30k iterations (capped) against about 800 per solve for the penalty path,
on the 2-D three-layer gauged test problem and on `fluid_core` at
`h = 0.3` (degree 2: h'₂ −0.989262 block-diagonal, −0.989023 MGW, against
−0.989424). Preconditioners tried: block-diagonal with serial
Gauss–Seidel and with parallel AMG; the Murphy–Golub–Wathen composite on
the multiplier slot (P_v⁻¹ ≈ P̂(A) Q̃ P̂(A), `SymmetricProductSolver`),
which degrades with smoother-grade components (primal residual 3.5e-2)
and still caps with AMG-grade ones. In the 2-D lab the stationarity
residual reaches 2.6e-5 while the primal residual stalls at 7.3e-4: the
preconditioner fails on the constraint rows.

Why: the "constraint" operator A is square and near-singular, so
saddle-point preconditioning theory (which assumes a full-rank
rectangular constraint) does not apply, and the near-kernel reappears
wherever the scheme needs solves with A. KKT pays for full-rank interface
constraints (the slip constraint); for selection over a near-kernel the
penalty with iterated refinement is the appropriate regularisation.

The remaining untried rung: constraint preconditioning (Keller, Gould &
Wathen 2000) with a flexible outer iteration (MFEM `FGMRESSolver`),
about a day's work; its two inner A-solves per outer iteration are solves
of the same near-singular system, so success is expected at best at
penalty-path cost. Early stopping on `KKTResiduals` is capped at the
measured 1e-3/1e-4 plateau, below verification grade. The open research
question: a parameter-free Krylov formulation of minimum-energy selection
over an *approximate* kernel whose cost does not degenerate with the
near-kernel.

**See:** `doc/slip_interface.tex`, "Why the fluid gauge stays a penalty";
`doc/quasi_static_models.tex`, "The exact-gauge KKT alternative";
`doc/benchmarks.tex`, "Solver settings: measured behaviour";
`include/AdGIA/mixed_problem.hpp` (`EnableGaugeKKT`);
`src/mixed_problem.cpp`.

### Reduced potential fluid as a preconditioner ingredient

**Status:** proposal (idea only).

The fluid's physical content modulo relabellings is one scalar
(div(ρu)) plus the interface normal trace, so a potential representation
u_f = ∇ψ spans it in the continuous problem (Neumann solvability;
relabellings lie in the radical of A). In the self-gravitating problem it
does not collapse to a single potential — Dahlen's φ-only fluid is exact
only for neutral stratification; in general ψ and φ both remain, and the
fourth-order ψ operator needs a mixed (ψ, p) pair — so it is not a
replacement formulation. It could serve as the fluid part of a
preconditioner for the gauged system.

**See:** `doc/self_gravitation.md` §1; `doc/gauged_fluid.md` §3.

---

## Elasticity preconditioning

### Near-incompressible and high-contrast effective operators

**Status:** open problem.

Per-solve cost grows steeply as the shear modulus becomes small against
κ, from two directions with one root (the displacement preconditioner,
BoomerAMG on the elasticity block, degrades):

- *Near incompressibility.* 2.5× per solve from κ/μ = 200 to 2000 on a
  coarse ball at orders 2–3; about 11 s per solve at order 3 on
  `coupled_poisson.msh` on 4 ranks.
- *Viscosity contrast in the effective operator.* With Δt ≫ τ in weak
  regions the effective modulus collapses there. CG iterations per solve
  (BoomerAMG on the effective operator) rise from about 55 (contrast
  C = 1) to 150–225 (column) and 230–340 (slab) at C ≥ 1e4, and to
  500–2500 on the graded column at C ≥ 1e6; sphere `lateral_weak` 105–180
  against 90–125. ETD1 and RK4, which solve with the unrelaxed operator,
  stay at 50–65.
- The fluid gauge block κ div div + ε μ_g dev is the same phenomenon
  (previous section).

Options: elasticity-aware BoomerAMG settings (nodal systems AMG,
near-null-space rigid-mode vectors); a mixed (u, p) formulation of the
solid with a pressure-robust block preconditioner; deflation of the
relaxed region's near-null space.

**See:** `doc/benchmarks.tex`, "Box: viscosity contrasts", "Sphere: the
integrators with lateral viscosity variations";
`src/quasi_static_problem.cpp` (`SetupDefaultPreconditioner`).

---

## Stopping tests and warm starts

### Increment-relative stopping test

**Status:** proposal (library change; a design decision is needed).

The main solve is warm-started from the previous solution, but its
stopping test is absolute, rel_tol · ‖B‖_{M⁻¹} (`SetWarmStartTolerance`),
so at rel_tol = 1e-10 a warm start saves only the digits the step did not
change. A test relative to the step's increment (the residual of the warm
start) would let successive viscoelastic and AL solves stop early.
Measured: `viscoelastic_love_numbers` at rel_tol 1e-6 gave the same Love
numbers to the printed digits as at 1e-10, at about a quarter of the cost;
looser defaults in the benchmarks are a cheap first step. The AL sweeps
already use an absolute target anchored to the first sweep's residual
(see "Inexact sweeps"); the question is what the right anchor is for a
time step.

**See:** `include/AdGIA/quasi_static_problem.hpp`
(`SetWarmStartTolerance`); `doc/viscoelasticity.md` §3;
`examples/viscoelastic_love_numbers.cpp`.

---

## Time stepping

### Adaptive exponential trapezoid: a sharper error estimate

**Status:** proposal, not implemented.

`AdaptiveExponentialTrapezoidSolver` estimates its local error from the
ETD1 companion of each trapezoid step (nodal work only, no extra solve):
err = RMS of (m − m̂)/(atol + rtol · max(|mⁿ|, |mⁿ⁺¹|)), controller
dt ← dt · clamp(0.9 err^{−1/2}, 0.2, 4). Being the estimate of the
first-order companion, it is conservative for the propagated
second-order solution by a factor of order τ/dt. A sharp estimate needs a
second-order companion, e.g. step doubling (about three elastic solves per
step instead of one).

The cost of the conservative estimate is measured:

- Under sustained periodic forcing (box benchmarks) the adaptive scheme
  is the most expensive choice, 1400–23000 solves, because the estimate
  keeps every step small.
- At contrast C = 1e8 (box column) it takes 35000 solves to reach
  rtol 1e-1, because the estimate tracks the transient of the τ = 1e-8
  branch; on sphere `lateral_weak` runs took over an hour per disc, so the
  study omits the adaptive scheme.
- On the viscoelastic Love-number driver it took 512 reassemblies and
  310 s; SDIRK23 is the default there.

Options, cheapest first: use the solver's existing `SetStepBounds`
(dt_min) in the drivers (no driver uses it at present); an error norm
weighting branches by their effect on the observables; a second-order
companion.

**See:** `doc/viscoelasticity.md` §3 "Choosing a scheme", §5
"State-dependent relaxation times"; `doc/benchmarks.tex`, "Box: the
integrators on a homogeneous body" (cost), "Box: viscosity contrasts";
`include/AdGIA/viscoelastic.hpp`
(`AdaptiveExponentialTrapezoidSolver`, `SetStepBounds`);
`src/viscoelastic.cpp` (`ErrorEstimate`).

### Caching unrelaxed and effective operators

**Status:** proposal (wait and see).

Adjoint calculations will apply load jumps that need the unrelaxed moduli
at every observation time, possibly many, interleaved with effective-
modulus steps. That is the use case for caching both stiffness versions
(matrix and preconditioner keyed by relaxation state) instead of
reassembling on every switch (`UseUnrelaxedOperator`). Decide when the
adjoint work starts.

**See:** `doc/viscoelasticity.md` §3; `src/viscoelastic.cpp`
(`UseUnrelaxedOperator`); `include/AdGIA/quasi_static_problem.hpp`
(`SetPreconditionerReuse`).

---

## Multiple right-hand sides

### Combined-degree solves

**Status:** decided and implemented as an opt-in benchmark mode
(`love_benchmark -combined`, `run.py --combined`, `campaign --combined`
for the methods and CMB stages); it replaces the earlier plan of Krylov
recycling across the degree sweep. Open: whether it becomes the campaign
default.

By linearity, one solve with all degree-l unit loads superposed (and one
for the tides) replaces the per-degree solves; the Love numbers are
extracted per degree from one spectral decomposition of the boundary
fields. Measured: 3–5× fewer iterations and less wall time; a combined
solve costs about one single-degree solve. The discrete operator is not
exactly SO(3)-equivariant, so leakage between degrees, discarded in
per-degree solves, adds in the combined one: measured 2e-5 to 4.5e-3
relative, tolerance-independent, below the mesh error except in one
cancellation case (see [open_issues.md](open_issues.md), "Combined-degree
solves: unverified paths"). Degree 0 stays excluded for Dahlen with a
fluid; the degree-1 frame correction is linear post-processing and works
unchanged. The perturbation and identity stages stay per-degree by design
(their finite differences and identities are below the leakage level).
For the slipping methods the saving multiplies through the AL sweeps.

**See:** `benchmarks/love_numbers/README.md`; `benchmarks/README.md`;
`benchmarks/campaign.py`.

---

## Parallel efficiency

### Parallel efficiency of the coupled solves

**Status:** open.

Production runs are parallel (the server campaign profile uses 100 ranks
and includes a weak-scaling ladder); solver decisions should be measured
in parallel. No strong- or weak-scaling results are recorded yet for the
gauged, referential and slipping methods, whose cost is dominated by the
elasticity AMG on the gauge-penalised fluid block and, for the slip
classes, by the interface constraint. Items above that bear directly on
parallel cost: the solid-everywhere preconditioner, elasticity-aware AMG
settings, combined-degree solves, increment-relative stopping.

**See:** `doc/benchmarks.tex`, "Organisation" (campaign profiles);
`benchmarks/campaign.py` (`scaling` stage).
