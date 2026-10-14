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

## Where to start (the measured ordering, 12 October 2026)

The AMG rigid-rotation route is measured out (four configurations,
no win — "Near-incompressible and high-contrast effective operators"
below), which orders the remaining work:

1. the frozen moderate-ε preconditioner for the Maxwell continuation
   ("The Maxwell endgame's own lever") — **measured, no win locally
   (12 October 2026)**: the default reuse already holds the
   physical-phase preconditioner to the end of the continuation, and
   freezing lower degrades cost and quality both; the knob
   (`MaxwellRelaxationOptions::prec_eps`, `-mxpe`) stays for the
   server h-ladder;
2. the border-column cache per operator flavour ("Border-column cache
   against the stepper's modulus switches") — **done 12 October 2026**,
   merged with the unrelaxed/effective operator caching into the
   modulus-flavour cache (alternation gate: 13 → 2 assemblies);
3. the mixed (u, p) pressure-robust block preconditioner, prototyped
   on the box contrast case before any gravitating class ("Options"
   of the near-incompressible item; deflation is the fallback) —
   **prototype delivered and measured 14 October 2026** (the
   pressure-split solve, below): parameter-robust as predicted, a
   ~2x saddle overhead where the primal AMG copes, 2.5–3x the other
   way on the graded column where it does not; the promotion
   decision is re-aimed at the production targets, graded 3-D
   viscosity and the Maxwell fluid-core endgame (see the item's
   "Measured" paragraph);
4. the Picard feedback inside the time stepper ("Picard feedback
   inside the time stepper") — **delivered 13 October 2026** as the
   gated `SetPicardBorder()` route; wins 25 % in operator-changing
   stepping, loses to the flavour-cached elimination at frozen
   operators; measured in its real regime the same day (stage 2 of
   the Picard item below: 16 % win on the rotating migrating
   history), so the promotion decision now sits with the route
   policy — server scale pending;
5. the decisions already measured: combined-degree solves as the
   campaign default, the monolithic border with a border-aware
   refinement stop, the increment-relative stopping test;
6. the calibration harness (below) once 1–4 exist;
7. server-scale confirmations: the `-amg-rbm` h-ladder, the
   strong/weak-scaling ladder, `prem_6`.

Not pursued: partial assembly / matrix-free operators (they conflict
with the AMG-on-assembled-matrices stack at production orders), and
fluid-formulation changes in pursuit of conditioning.

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
iterated refinement is the accepted gauge-fixing formulation for the
penalty path; the Maxwell relaxation is the general fluid treatment
(`doc/static_fluid_core.tex`) and the penalty is the specialist for
neutrally stratified fluids; only numerical levers (the next items)
remain open.

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

*Rigid-mode interpolation vectors: measured, no win (12 Oct 2026).*
`SetElasticityNearNull()` attaches the rigid rotations as hypre GM-2
interpolation vectors on top of the nodal-systems default (the
byNODES-compatible equivalent of mfem's `SetElasticityOptions`;
benchmark flag `-amg-rbm` in the love and box drivers). Measured at
np 4: `fluid_core` h 0.3, referential, l = 2 — endpoints
digit-identical, iterations 155 both ways, wall 7.8 → 11.7 s; box
`contrast` C = 1e4 — 3593 → 3474 iterations, wall +20 %; box
`gradient` C = 1e6 — 6231 → 6166 iterations, wall +10 %. fluid_core maxwell, l = 2 — the harshest small-μ
configuration (the continuation drives the fluid's effective shear
ε μ_c toward zero): 2706 → 2873 outer iterations, wall 15.5 → 25.4 s,
the re-setup on every escalation operator pure cost. The nodal
coarsening already captures what the rotations add on these
operators, so AMG-side vector tuning is not the lever here; the
remaining candidates are the mixed (u, p) block preconditioner and
deflation. The flag stays for h-ladder confirmation on the server
profile (the local meshes are small, and the degradation grows
with resolution).

*The Maxwell endgame's own lever: measured, no win (12 Oct 2026).*
The escalation already reuses its AMG across ε changes until the
iteration count spikes past `prec_reuse_` × baseline; the proposed
refinement was to build the preconditioner DELIBERATELY at a moderate
ε_prec and hold it for the whole continuation
(`MaxwellRelaxationOptions::prec_eps`, benchmark flag `-mxpe`; 0 keeps
the heuristic). Measured at np 4, order 2, h = 0.3, l = 2: pinning at
the physical-phase ε₀ = 1/(1 + β₀) = 0.25 is digit-identical to the
default on BOTH `fluid_core` (2706 iterations, 32 steps) and `aw_core`
(1583 iterations, 15 steps) — the spike heuristic never fires on these
continuations, so the default reuse already holds the moderate-ε
preconditioner to the end and there is no rebuild-on-a-degraded-operator
cost to remove. Freezing deliberately lower is harmful both ways: at
ε_prec = 1e-1–1e-3 on `fluid_core` the iterations rise to 2904–3080
(+7–14 %) and the returned state degrades (solid increment
1.26e-4 → ~2.5e-3, spurious 0.0044 → 0.0099). The spectrum bound is
uniform only in the operator's ε ≤ ε_prec; the physical phase at
ε₀ > ε_prec is the bad direction, its steps cost more iterations under
the mismatched preconditioner, and the iteration-sensitive stopping
rules (`iter_budget`, the solver floor) then end the escalation at
step 16 instead of 32 — preconditioner quality feeds the returned
physics through the stopping heuristics, not just the wall time. The
knob stays for the server h-ladder (a spike that never fires at
h = 0.3 may fire at production resolution); locally the endgame lever
moves to the remaining candidates, the mixed (u, p) block
preconditioner and deflation.

*The pressure-split (u, p) solve: prototype delivered and measured
(14 Oct 2026).* `SetPressureSplitSolver()` on the clamped solid routes
Solve() through the mixed reformulation with `p = kappa div u`
eliminated from the stress: MINRES on the symmetric block system
`[[A_mu, B^T], [B, -M_{1/kappa}]]` with `A_mu` the deviatoric stiffness,
`B` the divergence coupling onto a continuous pressure space one order
below the displacement (Taylor–Hood), preconditioned block-diagonally by
the house nodal-systems AMG on the kappa-free full-gradient operator
`2 mu eps:eps` and the diagonal of the pressure mass weighted by
`1/kappa + 1/(2 mu)` (the Schur equivalent). The AMG never sees kappa:
both degradation regimes — near incompressibility and collapsed
effective shear — move into the Schur block, where a weighted mass
matrix absorbs them. The rheology side supplies `(kappa, mu_eff)`
through the new `ElasticStiffness::IsotropicSplit()` (the live
redirectable effective shear; piecewise attribute dispatch for
composites, purely elastic regions beside viscoelastic ones included —
the production lithosphere/mantle shape, gated in `TestPressureSplit`).
Prototype scope: clamped solid only (no rigid-mode projection, no
fluid treatments, order >= 2); the modulus-flavour cache is bypassed
(every weight switch reassembles the blocks); the discrete solution is
the Taylor–Hood one — locking-free, but its continuous pressure smears
the pressure jump at material interfaces at discretisation level.
Benchmark flag `-up` (box driver).

Measured at np 4, order 2, exptrap, dt 0.25, 40 solves (`contrast` and
`gradient` cases of the box family; iterations per solve, wall for the
run):

| case | primal its | split its | primal wall | split wall |
|---|---|---|---|---|
| C = 1, column, nz 10 | 55 | 129 | 0.57 s | 1.34 s |
| C = 1e4, column | 140 | 203 | 1.7 s | 2.0 s |
| C = 1e4, slab | 267 | 424 | 5.6 s | 7.3 s |
| C = 1e6, column | 210 | 202 | 2.0 s | 2.4 s |
| C = 1e6, slab | 287 | 365 | 3.8 s | 5.3 s |
| gradient C = 1e6, nz 20 | 1102 | 367 | 7.7 s | 3.1 s |
| gradient, nz 40 | 2337 | 760 | 21.8 s | 7.8 s |
| gradient, nz 80 | 4115 | 1756 | 41.1 s | 19.8 s |

The reading: the split is parameter-robust — its column count is flat
across the whole ladder (129 → 203 → 202) — but carries a ~2x saddle
overhead per solve, so on the LAYERED cases, where the nodal AMG
copes with a weak layer sandwiched by elastic support, the primal
route stays ahead. It wins exactly where the primal collapses: the
smoothly GRADED column (2.5–3x in iterations and wall, and the stakes
grow with resolution — the primal reaches 4115 iterations per solve at
nz 80). On the harsher two-region test configuration (weak-branch
weight 1e-6, no stiff support) the gap is 5–7x (168 vs 932 at np 1,
flat to np 4), and there the primal does not merely slow down, it
LOCKS — the discretisations disagree at O(1), which is a correctness
point for the primal route, not just a cost one.

The promotion decision (David's framing, 14 Oct): the layered box
cases are not the production shape. The production targets are
(a) smoothly graded / 3-D viscosity in the mantle — the gradient
case's regime, where the split already wins — and above all (b) the
Maxwell fluid-core treatment, which drives the core's effective shear
`eps mu_c -> 0` BY DESIGN: the most extreme kappa/mu contrast in the
codebase (the 2706-iteration continuation measured on `fluid_core`).
There `p = kappa div u` stays healthy in the fluid while the vanishing
shear moves into the Schur weight. The next rung is therefore the
extension to the mixed gravitating class — the displacement block of
the (u, phi) solve with the fluid region's effective shear as one more
piecewise split region — measured on `fluid_core` against the Maxwell
continuation's current cost. Open questions for that rung: the
potential block rides along (the split changes only the u-block solve
inside SolveLinearSystem's block structure), the bordered rows must
not notice, and the near-zero-mu A_mu block leaves the fluid
relabelling null space to the continuation exactly as today. Deflation
remains the fallback if the extension disappoints.

*The region-restricted split (David's question, 14 Oct 2026: primal
solid solver in the solid, pressure split in the fluid, continuity
weak at the interface).* Yes — and there is a lighter version that
needs no interface machinery at all. Nothing forces the pressure to
live everywhere: keep ONE conforming displacement space, keep the
primal form kappa div·div + 2 mu dev·dev on the solid elements, and
split only the fluid's volumetric term — p on a pressure space over
the FLUID elements only, B the divergence coupling restricted to the
fluid marker. Continuity of u is then automatic (the space is
conforming), no weak coupling, no mortar, and the saddle overhead and
the Taylor–Hood pressure are paid only where the collapse lives. The
preconditioner follows: the house AMG on [solid primal + fluid
full-gradient] plus the fluid-only weighted pressure mass. The open
technical point is the inf-sup condition for the restricted pair
(divergence of fluid-supported conforming fields against the fluid
pressure space, with the interface acting as that subproblem's
boundary) — standard domain-restricted Taylor–Hood territory, to be
checked rather than assumed. The weak-continuity (Nitsche / mortar)
variant David describes becomes necessary only when the two regions
also want DIFFERENT displacement discretisations or a genuinely
slipping interface — and there it would meet the existing slip-class
multiplier machinery rather than new code. Since the Maxwell fluid
treatment runs on the welded space (slip emerges as its inviscid
asymptotics), the conforming region-restricted split is the natural
first implementation for the fluid-core target, and it composes with
the stepping question above: the solid's solves stay well-conditioned
for ANY stepper, the fluid keeps its continuation.

*Round 2 (14 Oct, after David's framing): the viscosity-limit
lithosphere and multiple time scales.* The new box model `visco_lid` is
the elastic-lid-as-viscosity-limit case: ONE Maxwell material, weak
mantle (tau = 1/C) on [0, 0.6], a rapid geometric ramp to tau = 100
over [0.6, 0.8], and a tau = 100 lid, effectively elastic over the
window by viscosity alone. Measured (np 4, o2, exptrap, dt 0.25,
iterations per solve):

| case | primal | split |
|---|---|---|
| C = 1e2 column / slab | 119 / 162 | 161 / 318 |
| C = 1e4 column / slab | 348 / 921 | 226 / 664 |
| C = 1e6 column / slab | 527 / 2865 | 287 / 1208 |
| C = 1e6 column, nz 40 | 4217 | 788 |
| C = 1e6 slab, nz 20 / 40 | **CG FAILS (10000-cap)** | see below |

Unlike the sharp layered `contrast` cases, here the split wins on BOTH
column and slab from C = 1e4 up (1.5–2.4x at nz 10, 5x at nz 40), and
the primal route does not merely degrade — on the C = 1e6 slab it FAILS
OUTRIGHT at nz 20/40 (CG hits the iteration cap and the step aborts).
The production reading: an effectively-elastic lithosphere realised by
viscosity is the graded regime, and it is where the primal breaks first.

Multiple time scales behave as the mu_inf floor predicts: `prony_lid`
(mu_inf = 0.1) keeps kappa/mu_eff <= ~17 at every dt — primal 66–89
its/solve flat, split ~250 (the saddle overhead, no win); `burgers_lid`
(mu_inf = 0) grows toward the Maxwell regime as dt collapses both
branches (primal 76 -> 228, split ~250–365). A nonzero long-term shear
modulus caps the hazard; the hard cases are mu_inf ~ 0 materials and
graded/rapid transitions.

**Open issue — the outer Krylov on the extreme split systems
(14 Oct):** on some `visco_lid` configurations (first seen at C = 1e6,
column, nz 15–30) MINRES ABORTS with `beta = -nan` a few iterations in:
the P-inner-product recurrence evaluates sqrt((v, Pv)) by cancellation
across the preconditioner's ~1e6 coefficient spread and roundoff flips
the sign. The preconditioned SYSTEM is healthy — full (unrestarted)
GMRES with the same block preconditioner solves the worst config at
~200 iterations/solve, right in the split's robust band — and
restarted GMRES(60) stalls (restart truncation on the indefinite
spectrum), so this is a Krylov-numerics engineering choice, not a
method defect. Candidate resolutions, to measure: (a) GMRES with a
large restart as the route's solver (memory cost at production sizes),
(b) hardening MINRES against the sign flip (clamp the Dot at zero /
detect breakdown and restart), (c) symmetric diagonal equilibration of
the block system so the Lanczos vectors see commensurate components.
Fixed along the way and kept: the preconditioner block's essential
diagonal is now KEPT at its assembled value rather than set to one
(unit rows in a collapsed-shear clamped region put an artificial 1e6
jump under the AMG); the operator blocks stay DIAG_ONE.

Implementation notes recorded along the way: serial
`RecoverFEMSolution` assumes the solver wrote through FormLinearSystem's
aliased true-dof vector — a block solver must copy its solution back
into that vector before recovering (see `doc/mfem_notes.md`); the
flavour-cache guards live in `StashFlavor`/`TryRestoreFlavor`.

**See:** `doc/benchmarks.tex`, "Box: viscosity contrasts", "Sphere: the
integrators with lateral viscosity variations";
`src/quasi_static_problem.cpp` (`SetupDefaultPreconditioner`,
`AssemblePressureSplitOperator`); `tests/TestPressureSplit.cpp`.

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
  is the most expensive choice, 1400–13000 solves, because the estimate
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

### Implicit against explicit stepping: revisit per model

**Status:** question recorded (David, 14 Oct 2026), to revisit once the
pressure-split promotion is settled; no new measurements yet.

The house default is implicit (L-stable / exponential-trapezoid)
stepping, chosen for accuracy at large dt. The solver-side cost
structure cuts the other way: implicit steps solve with the EFFECTIVE
operator, whose conditioning collapses exactly where dt is large
against tau, while the explicit and exponential-explicit schemes (ETD1,
RK4) solve with the UNRELAXED operator and sit at the flat ~50–65
iterations per solve at every contrast (the box measurements). So the
total-cost comparison — few expensive ill-conditioned solves against
many cheap well-conditioned ones — is MODEL-DEPENDENT, and the answer
shifts with everything this item's work changes: the pressure-split
route flattens the implicit solve cost (weakening the explicit case),
while a model with mu_inf > 0 everywhere caps the implicit penalty by
itself (the prony_lid finding). The revisit must carry the known
accuracy/stability constraints from the stepper survey, which are not
solver-side folklore but measured: RK4 is stability-bound at
dt < 2.8/C, and ETD1 does not converge at large contrast on slabs
(error 0.5–0.8 at C >= 1e4 — its start-of-step relaxation misses the
slow redistribution; column-only convergence). A serious explicit
contender therefore needs either resolved fast time scales (many
steps) or a better exponential corrector. A related half-way option is
an IMEX / region-split step: implicit only where dt >> tau, explicit
(unrelaxed) elsewhere — which meets the region-splitting idea of the
pressure-split item above.

**See:** `doc/benchmarks.tex`, "Box: viscosity contrasts" (the ETD1 and
RK4 findings); "The pressure-split (u, p) solve" above.

### Caching unrelaxed and effective operators

**Status:** DONE 12 October 2026 — the modulus-flavour cache; one
mechanism with the border-column item, see "Border-column cache
against the stepper's modulus switches" (which also records why the
preconditioner stays shared rather than per-flavour).

**See:** `doc/viscoelasticity.md` §3; `src/viscoelastic.cpp`
(`UseUnrelaxedOperator`, `SetEffectiveModulus`);
`include/AdGIA/quasi_static_problem.hpp`
(`RestoreRelaxationWeights`).

---

## Bordered solves (sea level and rotation)

**Route policy (David, 13 October 2026): all routes stay, and the
defaults stay — block-MINRES for the coupled (u, φ) solve, the
ELIMINATION for the borders — until server-scale measurements exist
(none of the local np 4 verdicts is trusted at production rank counts;
the scaling ladder of "Parallel efficiency" below is where they could
invert). Monolithic and Picard remain gated options; the per-problem
route choice is a calibration-harness knob.**

The bordered system `[K B; Bᵀ Db]` (the Φ_g and ω borders of the
sea-level machinery) has three solve routes, all gated to agree:
the **elimination** (one unbordered solve per border column, the
reference; columns cached while `OperatorVersion()` stands still, so
repeated solves at a frozen operator pay one base solve), the
**monolithic MINRES** (`SetMonolithicBorder`, 3-D BlockMINRES path:
probed-|Schur| border preconditioner, equilibrated columns,
true-residual refinement; cost flat in the border count), and the
**Picard feedback** instrument (the loads iterated around the plain
operator; `sea_level_benchmark -feedback picard`). Measured at
h 0.4/o2/np4: elastic 1.00×, water 1.64× (monolithic 2.31×),
water+rotation 3.80× (monolithic 2.34×).

### Monolithic border: default and refinement economy

**Status:** decided — the elimination stays the default; the monolithic
route is the gated option for multi-border one-shots. Open: its
refinement economy, and server soak before promoting it for rotating
sweeps.

The true-residual refinement guards against the MINRES recurrence
drifting at the border preconditioner's conditioning, but at tight
tolerances it costs a second full Krylov pass (pass 0 reaches ~2e-9,
pass 1 ~3e-15 at h 0.4/o2), which is what loses the single-border water
case to the elimination. A border-aware stop (refine only until the
border rows' own residuals meet their scale, rather than the global
norm) would reclaim most of that pass. Related: the loose-tolerance
Φ_g amplification (`open_issues.md`, "The sea-level equation") is the
same near-cancellation seen from the elimination side.

**See:** `src/mixed_problem.cpp::SolveBorderedMonolithic`;
`tests/TestMonolithicBorder(.Par).cpp`;
`benchmarks/sea_level/run.py --timings`.

### Picard feedback inside the time stepper

**Status:** DELIVERED 13 October 2026 as the problem-level
`SetPicardBorder()` route (gated, elimination stays the default);
promotion decision deferred to the migration stage, where its target
regime lives.

*What was built.* Block Gauss–Seidel on the already-assembled bordered
system — @f$K x = b - B y@f$ with the current border scalars, then the
dense rows @f$D_b y = r - B^T x@f$ — i.e. the water and rotation
feedbacks as explicit loads over the SAME discrete system as the
elimination, which makes the routes agree at solver grade by
construction (gated in `TestMonolithicBorder(.Par)`, np 1/2/4, with
the near-neutral moments) and removes the earlier instrument's
ψ-convention and mass-bookkeeping requirements entirely. The border
scalars warm-start from the previous Solve(). Plain Gauss–Seidel
STALLS at near-neutral moments (the loop gain reaches 1; measured:
50-pass cap hit), so the pass loop is Anderson-mixed on the nb ≤ 4
border scalars — GMRES on the tiny affine Schur map, converging in
about nb + 1 passes regardless of the gain (rank-replicated scalars:
no communication, and the dots are the plain update's own). On
non-convergence the route warns and falls back to the elimination.

*Measured (np 4, two-layer ball).* One-shot composed (water +
rotation, homogeneous h 0.4 o2): elimination 0.580 s, monolithic
0.332 s, picard-border 0.459 s (8 passes), the fully-explicit
instrument 0.949 s (13 passes) — cold, the monolithic stays the
one-shot winner. Cold pass counts: water 3, rotation 9–10, composed 8.
Stepping (the small sea-level test, 6 trapezoid steps): at FIXED dt
the elimination + the modulus-flavour cache wins 2× (560 vs 1060
iterations) — the cached columns are exact probes of the frozen
operator, the optimal algorithm there; at VARYING dt (the operator
reassembles each step: adaptive stepping, nonlinear correctors,
migration) picard-border wins 25 % (1397 → 1049). ANDERSON-RECYCLING
ACROSS SOLVES was tried and reverted: residual differences are
load-independent in exact arithmetic, but the warm-start-tolerance
noise of nearly-converged pairs pollutes later solves (fixed-dt
stepping 1060 → 1520 iterations), and the frozen-operator regime
belongs to the elimination anyway.

*Stage 2 — migration merged (13 October 2026).* The migration loop
now runs INSIDE the stepper: `ViscoelasticOperator::SetStepCorrector`
fires a load-law corrector in the implicit and trapezoid pass loops,
and `ShorelineMigration::Correct()` is that corrector — one combined
fixed point per step (C, the borders and, when nonlinear, the
relaxation times together), settling in a few passes per step. The
converged ocean is recorded per step (`RecordStep`) and played back by
`SetRecordedTime` (left-limit, memoised) — the recorded-C stack the
adjoint's reversed playback consumes (WP8.4). Gates
(`TestShorelineMigration(.Par)`): two migrating trapezoid steps at
dt ≪ τ land on the static migrating solve (1e-4); the per-step
corrector settles within 6 passes on a relaxing rotating history;
playback reproduces the forward ocean exactly; routes agree at 1e-6.
MEASURED on the rotating migrating history (5 steps, serial, the
small two-layer ball): elimination 1846 iterations, picard-border
1548 — a 16 % win in the production-shaped regime (no rotation the
single border still belongs to the elimination, measured 796 vs
1299). The promotion decision stays with the route policy above
(server scale pending); production-scale timing on `ice_age_loading`
when migration lands there.

**See:** `include/AdGIA/mixed_problem.hpp` (`SetPicardBorder`,
`LastPicardPasses`); `src/mixed_problem.cpp::SolveBorderedPicard`;
`tests/TestMonolithicBorder(.Par).cpp`;
`tests/TestSeaLevelViscoelastic(.Par).cpp` (`PicardBorderStepping`);
`benchmarks/sea_level/sea_level_benchmark.cpp`
(`-feedback picard-border`; `-feedback picard` remains the
fully-explicit instrument).

### Anderson mixing: where it helps and where it hurts

**Status:** record (consolidated 13 October 2026 — the findings were
previously only in the API docs).

- *Maxwell memory fixed point* (`MaxwellRelaxationOptions::anderson`,
  depth 4, measured before 9 October 2026): mixing the FIELD-SIZED
  memory vectors gave ~40 % fewer steps and wall −10–35 % in serial
  2-D and on the 3-D ball (347 vs 539 iterations at the default β),
  but **+70 % on a small parallel 2-D case** — the mixed updates
  degrade the inner solves' warm starts and tighten the inexact
  tolerances sooner. Opt-in; a calibration-harness decision.
- *Picard border scalars* (13 October 2026): mixing the nb ≤ 4
  rank-replicated border scalars is load-bearing, not an optimisation
  — plain Gauss–Seidel stalls at near-neutral moments, mixed it
  converges in ~nb + 1 passes — and has no parallel cost (no
  communication; the dots are the plain update's own). Measured at
  np 4 throughout.
- *Recycling Anderson pairs across solves*: reverted — see the Picard
  item above (warm-start-tolerance noise in nearly-converged pairs).

**See:** `include/AdGIA/quasi_static_problem.hpp`
(`MaxwellRelaxationOptions::anderson`);
`src/mixed_problem.cpp::SolveBorderedPicard`.

### Border-column cache against the stepper's modulus switches

**Status:** DONE 12 October 2026, together with "Caching unrelaxed and
effective operators" — one mechanism, the modulus-flavour cache.

The problem base keeps the assembled operator state of the unrelaxed
and the last relaxed stiffness in two slots
(`RestoreRelaxationWeights()`; `ViscoelasticOperator` re-engages an
unchanged weight set through it), so a stepper alternating between the
two — effective steps interleaved with unrelaxed observation or
adjoint impulse solves, the adjoint sweep against epoch-resolved
observations being the production case — swaps cached matrices and
operator versions instead of reassembling. The revived version revives
every version-keyed cache; `BorderedElimination` keeps two column
sets (most recently used first), so both flavours hold their columns.
Any non-flavour mutation goes through `MarkOperatorDirty()` and drops
the slots — a revived version never outlives the state it keyed.
Steady-dt stepping alone never alternates (the trapezoid's end state
is unrelaxed-consistent, so the plain history was already at two
assemblies); the win is wherever observation solves break the step
cache: the alternation gate
(`TestSeaLevelViscoelastic.AlternationRestoresFlavors` + the par
twin's section 4) runs water + rotation with a per-step invalidated
observation solve at 2 assemblies against 13 before (wall 640 → 334 ms
on the gate; mass row checked at every flip, fresh-problem identity at
1e-8).

THE PRECONDITIONER IS DELIBERATELY NOT PER-FLAVOUR: it keeps the
shared-reuse policy. A first per-flavour version exposed a LATENT
PRECONDITIONER FRAGILITY (now in `open_issues.md`): a fresh BoomerAMG
built on the power-law EFFECTIVE traction operator (pure Neumann,
small effective shear) returns NaN at np 2 — reachable in the
pre-cache library through `SetPreconditionerReuse(1.0)`, and masked in
production only because the reuse heuristic happens never to rebuild
on that operator. Until the near-incompressible preconditioning item
above lands, cross-flavour prec reuse is the behaviour-preserving
choice; the slots carry explicit keep-alives (`prec_ref_slot_`, the
referential fold hook) so the shared preconditioner's matrices
survive slot swaps.

**See:** `src/quasi_static_problem.cpp`
(`StashFlavor`/`TryRestoreFlavor`/`ReleaseFlavorSlot`);
`include/AdGIA/bordered.hpp`; `src/viscoelastic.cpp`
(`SetEffectiveModulus`); `tests/TestSeaLevelViscoelastic(.Par).cpp`.

---

## Automated solver calibration

### A per-problem calibration harness

**Status:** proposal (agreed 8 October 2026); queued behind the
preconditioning and stepping items above.

Applications are viscoelastic loops re-solving the same elastic system
hundreds of times, so a ~20-solve one-off calibration amortises
immediately. Sketch: `CalibrateSolver(t0, opts)` (its own header,
driven through the public interface) draws its candidate set FROM the
active configuration — the `FluidTreatment` slot makes the knobs
discoverable: penalty ε grid with refinement counts derived from the
measured contraction, Maxwell β₀ ∈ {3, 10, 30} and the inexact factor,
the relative tolerance, preconditioner reuse, Anderson depth on/off —
scored by wall time per solve under an observable-drift guard against
the tightest candidate. Calibrate at BOTH the unrelaxed and the
relaxed effective moduli, since the operator drifts over a time
integration; apply the winner and report. This is also the natural
home of per-problem decisions the measurements left open (the
Anderson default, β₀ on non-neutral cores).

**See:** `include/AdGIA/quasi_static_problem.hpp` (`SetFluid`,
`SetPreconditionerReuse`, the Anderson options);
`doc/static_fluid_core.tex`, "The Maxwell relaxation method".

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
