# The adjoint GIA problem (WP8.1): theory note and common-structure design

*Status: **reviewed by David 11 Oct 2026** (the constrained-SL₀
formulation confirmed parked); the common structure of WP8.2 is
implemented from this design (`include/AdGIA/adjoint.hpp`, gate GA0 —
measurements in [`adjoint_plan.md`](adjoint_plan.md), WP8.2). Sources:
Yu, Al-Attar, Syvret & Lloyd (2025, "the paper",
`doc/Elasticity/ggae388.pdf`) §3–4; `examples/adjoint_elasticity.cpp`
and `examples/viscoelastic_adjoint.cpp` (the two FD-gated prototypes
the structure was extracted from); the design decisions recorded in
`adjoint_plan.md` ("Design decisions", 11 Oct 2026).*

## The adjoint problem in our variables

The paper's §3.3 observation, restated as an implementation dictionary:
the adjoint GIA problem is the forward problem run in reversed time
with **four force slots** opened and the ocean function **prescribed**.
Nothing is transposed anywhere — the elastostatic operator is
self-adjoint, the bordered blocks (water, rotation) are symmetric by
construction, and the Maxwell evolution reverses cleanly.

With T the time-reversal map (T f)(t) = f(t₁ − t + t₀) (involutory,
anticommutes with d/dt, self-adjoint on L²(I); paper eqs. 43–45), the
adjoint fields (u†, ζ†, ω†, m†, SL†₁, σ†) satisfy, in reversed time:

- the **same elastostatic subproblem**, with the surface conditions
  (paper eqs. 57–58 vs the forward 55–56) acquiring two extra terms:
  −T D_u J in the traction condition and +T D_φ J in the
  potential-jump condition;
- the **same Maxwell evolution** τ ṁ† + m† = d†, with m†(t₀) = 0
  (paper eq. 49) — zero initial state in reversed time;
- the **same water column**: SL†₁ = −(u†·∇Φ + ζ† + ψ†)/g + Λ†_g/g and
  ∫σ† dS = 0 (paper eqs. 76–78 / 49–50) are structurally the forward
  sea-level readout and mass row, except that the ocean function is
  C† = T C — the time-reversed forward one, *prescribed* rather than
  dynamic (the adjoint is linear with time-dependent coefficients);
- the **same rotation border**: the ω-row reads
  δω·(D·ω† + T D_ω J) = 0 (paper eq. 48) — the forward row with an
  additive datum.

The dictionary onto the library (both classes, mixed and referential):

| paper slot | enters as | our mechanism | new plumbing? |
|---|---|---|---|
| D_SL J | −(1/g) T D_SL J inside the load σ† (eq. 53), exactly where the ice/data load enters σ | `SetSurfaceLoad` with a time-dependent coefficient | no |
| D_u J | a traction-like dual on the displacement rows | `AddForce` (an assembled boundary linear form — exactly the examples' observation vector g) | no |
| D_φ J | a dual on the potential rows | a boundary linear form added to the ζ-block right-hand side | **yes** — small: the ζ-row analogue of `AddForce` |
| D_ω J | a 3-vector datum on the rotation border's rows | added to the border right-hand side | **yes** — trivial: the bordered engine's r-vector takes an increment |
| C† | prescribed time-reversed ocean function | `RefreshWaterLoad()` driven by a stored C history (playback mode of the water border) instead of the dynamic update | **yes** — a mode switch, not new assembly |

So the honest inventory of genuinely new forward-code plumbing is: a
ζ-row dual slot, an ω-row datum, and a prescribed-C mode. Everything
else is calling what exists with different data.

Two conventions to fix once:

- **Time convention.** Adjoint force data are naturally specified at
  *forward* observation times t′; the adjoint evolution runs in the
  reversed variable. The force class below therefore speaks forward
  time, and the adjoint driver owns the reversal (t ↔ t₁ − t + t₀) —
  users never apply T by hand. (The paper time-reverses the adjoint
  variables inside the Lagrangian for exactly this convenience.)
- **Shoreline-migration-off is not a separate theory.** With migration
  off the forward load is eq. (59) and the forward problem is linear;
  the adjoint equations are *unchanged in form* — only C† becomes the
  constant C. The common structure never branches on it.

## Existence, rigid modes, gauges

The paper's existence condition (eq. 54) is rigid-compatibility of the
adjoint loads: taking rigid test functions, the D_u J/D_φ J surface
terms must vanish — which holds exactly when the objective functional
J is invariant under linearised rigid motions, i.e. physically well
defined. Our projector machinery already enforces the discrete version
on every solve (and the examples' windowed baseline functionals were
built rigid-invariant for this reason). The kernels are rigid-invariant
combinations (paper, after eq. 65), so the rigid non-uniqueness of
(u†, ζ†) never reaches them. With the generalised rotational door the
Tisserand gauge carries over verbatim — the gauge rows are the same
invariant integrals, and the mantle-frame restriction applies to the
adjoint solve unchanged.

## Observation times: the generalized fingerprint solve and the m† jump

When J depends on the state at discrete times t′ (in practice almost
always — decision 2), the functional derivatives are impulsive,
D_* J = h̃†_* δ(t − t′) (paper eqs. 70–73), and at the reversed
observation time every adjoint variable *except m†* carries a
δ-singularity. Integrating the adjoint equations across the time and
taking the limit gives one static solve — the **generalized
fingerprint problem** (paper eqs. 75–78): the elastostatic + water
(+ rotation) system loaded impulsively through all four slots by the
h̃† data. Its displacement d̃† then **jumps the adjoint internal
variable** (paper eq. 80):

    [m†]⁺₋ = d̃† / τ    (pointwise at the internal-variable nodes),

after which the adjoint relaxes freely until the next observation.
Between observation times the adjoint steps with the standard
machinery and every kernel integrand is bounded.

`viscoelastic_adjoint` implements precisely this with gravity off: the
fingerprint solve degenerates to one traction solve loaded by the
observation dual (`AddForce(g)`), and the jump is
m† += dev ε(u†_g)/τ via `InverseRelaxationTimes`. The GIA version
replaces that solve by the monolithic water(+rotation) solve — which
is machinery we have. A **static adjoint** is the degenerate case:
point terms only, no stepping — the fingerprint solve *is* the whole
problem (`adjoint_elasticity`).

The elastic fingerprint problem has its own adjoint theory in another
hat: the reciprocity paper (`doc/Elasticity/ggad434.pdf`; Al-Attar,
Syvret, Crawford, Mitrovica & Lloyd 2024) proves the loading
reciprocity (its eq. 8 — the discrete version is the rung-3 GA1
certificate), the self-adjointness of the fingerprint operator
ζ → SL (eqs. 15/17 with the Green-function symmetry, eq. 21), and the
rotational generalisation (§3.2). These are ready-made certificates
for the water rung, and the laterally-heterogeneous-structure case had
never been tested numerically — gate GA1b, DONE 11 Oct 2026 on the 3-D
ball with Gaussian melt caps (defects 2.5e-5 spherical / 3.5e-5 with
3-D structure / 1.9e-5 with rotation + structure, all the readout's
projection gap, converging with order); a quotable example of the
3-D-structure case remains a candidate deliverable.

## The kernels, in our conventions

**Viscosity η** (the workhorse; paper eq. 61):

    D_η Ĵ = ∫_I (2μ/η²)·η (d − m) : (T m†) dt
          = ∫_I (2μ/η) (d − m) : (T m†) dt,

bounded integrand, standard quadrature between observation times. The
internal-variable-free form (eq. 69), −∫ (1/2μη) τ : (T τ†_v) dt with
τ_v the viscous deviatoric stress, is the cross-check against codes
without internal variables — worth printing once in a gate. Per
decision 5 this kernel rides the **mixed** class in production;
everything below is referential-only machinery.

**The (κ, μ, η) convention.** The paper parameterises η = μτ *with μ
held fixed*; our examples take (κ, μ, η) as the independent
parameters, so perturbing μ at fixed η also perturbs τ, and K_μ picks
up the corresponding extra term (recorded in the
`viscoelastic_adjoint` header, FD-gated). The library kernels keep the
(κ, μ, η) convention — state it wherever a kernel is exported.

**Ice history** (decision 4: quelled AND raw). The raw derivative
(paper eq. 64) has δ-singularities at t₀ and at every observation time
— real physics, the instantaneous elastic response to instantaneous
loading, not an artefact. It stays available as an exportable datum.
The optimisation-facing standard is the Backus quelling through the
rate parameterisation (İ, I₀) (paper eq. 84):

    D_İ Ĵ(x, t) = −ρᵢ g ∫_t^{t₁} [1 − C(x, s)] SL†₁(x, t₁ − s + t₀) ds,
    D_I₀ Ĵ(x)   =  ρᵢ g ∫_I [C(x, t) − C₀(x)] SL†₁(x, t₁ − t + t₀) dt,

with only finite jumps at the observation times: both are
time-integrals of the adjoint sea level against ocean-function
factors, accumulated during the adjoint sweep like any other kernel.
The Sobolev-embedding route (Zuberi & Pratt 2017; Syvret 2022) is the
stated upgrade and composes with the metric layer — nothing in the
structure below commits against it.

**Initial sea level SL₀** (paper eq. 65):

    D_SL₀ Ĵ = ∫_I [ D_SL J − ρ_w g (C − C₀) T SL†₁ ] dt,

with the D_SL J part contributing its impulses at the observation
times (evaluated through the fingerprint solves) and the rest smooth.
§3.9's observation is worth keeping in view: for the present-day
matching functional, the traditional iterative initial-sea-level
update is the steepest-descent step on this kernel — so the kernel
doubles as a diagnostic of that iteration. The *constrained*
formulation (paper eqs. 89–92) makes the adjoint load non-time-local
(it needs ∫ SL†₁ dt at the reversed present) and is therefore
iterative; parked as a later option, not in the common structure.

**κ, μ** — "just to look" (low effort): the static parts are
`adjoint_elasticity` verbatim, the viscoelastic parts
`viscoelastic_adjoint` verbatim, lifted through the same slots.

**Density ρ** (decision 3, the split stated once): what is included is
the equilibrium Poisson coupling — δρ perturbs the background
potential Φ₀, adding one adjoint Poisson solve's worth of chain to the
kernel. What is **parked** is the hydrostatic momentum-balance
consistency: δρ also changes p₀ and the equilibrium stress S_e, and
that constraint is not imposed. The kernel's documentation and the
export both carry this statement; the referential form keeps ρ̃ linear
in every operator, which is what makes the included chain easy.

**Lithospheric thickness** (the novel leg, WP8.5): in the referential
frame an interface perturbation is a diffeomorphism direction δφ_e
shifting the reference interface — default convention (open Q4): a
scalar normal-shift field on the reference interface, lithosphere base
first. The derivative of the forward operator in that direction
assembles from the mapped machinery the relabelling family certifies,
and the adjoint pairing turns it into a surface kernel. The FD gate
pre-exists: the perturbation benchmark family already differentiates
through radial interface maps on a fixed mesh (GA5).

## The rheology axis: Prony series and nonlinear laws

David's check (11 Oct 2026): the structure must think, from the start,
about the general rheologies the library has or can host — the general
linear Prony series, and the Prony-like laws whose evolution equations
are nonlinear (one example implemented; the general form easy). Worked
through, the news is good: the rheology axis is orthogonal to the
ladder, and the four bricks are already shaped for it at all but one
marked point.

**General linear Prony (multi-branch).** The adjoint of the
multi-branch evolution is branchwise: each branch k carries its own
adjoint internal variable with the SAME evolution τ_k ṁ†_k + m†_k =
d†, and the elastic solve sums the branch forces exactly as the
forward one — so the adjoint evolution operator is the forward
`ViscoelasticOperator` at any branch count, and the driver's jump,
[m†_k]⁺₋ = d̃†/τ_k per branch at its nodes, is already implemented
generally. The one branch-0-bound piece is the KERNEL layer: the
pairing struct carries one set of internal fields and
`IsotropicKernel` evaluates single-branch Maxwell densities (the
driver's guard marks exactly this boundary). The extension is
contained there: per-branch kernels — eq. (61) branchwise,
D_{η_k} Ĵ = ∫ (2μ_k/η_k)(d − m_k) : (T m†_k) dt, with D_{μ_k} getting
the branch's share of the elastic pairing — accumulators taking a
branch index, integration restricted to `BranchMarker(k)` for
region-confined branches, and the unrelaxed moduli (κ, μ_U) keeping
the elastic densities. Scientifically these are kernels with respect
to the relaxation spectrum. To be built with the first multi-branch
gate (an SLS/Burgers FD table, serial + parallel).

**Nonlinear laws.** The library's forward form (relaxation_law.hpp):
ṁ_k = (d − m_k)/τ_k with τ_k = τ_k0 F_k(ε, σ, m_k) pointwise at the
internal nodes. The adjoint of a nonlinear forward problem is LINEAR,
with coefficients linearised about the stored forward trajectory:

- the adjoint evolution carries ∂(rate)/∂m ᵀ evaluated at the forward
  state — per node a small nc×nc matrix, the scalar relaxation rate
  plus a rank-one correction from ∂τ/∂m;
- the branch-to-elastic coupling weight becomes ∂(rate)/∂d ᵀ at the
  forward state, and the observation-time jump weight generalises from
  1/τ to the same linearised coupling at the observation state;
- the kernels of the law's parameters pair ∂(rate)/∂(parameter) with
  m†_k — through `RelaxationLaw::Gradient`, the hook the forward side
  already declares "for adjoint problems" (implemented for the power
  law).

The structural consequences the design already accommodates: (i) the
adjoint evolution operator is a constructor argument of the driver,
and `ViscoelasticOperator::Rate`/`LocalExponentialUpdate` are virtual
precisely so a tangent-adjoint subclass can override them with the
linearised nodal coefficients; (ii) the driver gains one sync point —
feed that subclass the forward state at the matching reversed time —
which the trajectory store ALREADY serves, because the linearisation
coefficients are functions of (m, load(t)) and the slaving solve
reconstructs them anywhere: **m-only checkpointing survives
nonlinearity unchanged**, the important structural fact. The honest
complications: the nodal linearisation is a matrix, so the exponential
steppers need matrix updates (or the implicit route via effective
moduli), and the adjoint visits the forward state at every step, which
raises the re-solve traffic of lean trajectory policies. Nonlinearity
sits in the RHEOLOGY only — the elastic operator stays linear and
self-adjoint, so the elastic half of every rung, the force slots and
the fingerprint solves are untouched, and the `IsLinear()` guard in
today's driver marks the boundary exactly.

## Second order, summarised

The paper's §4: the Hessian-vector action H·δm costs ~4 simulations.
The second-order functional J₂ (eq. 103) is the first-order kernel
paired with the given model perturbation; making its Lagrangian
stationary shows the second-order state and adjoint-state variables
satisfy **the same forward and adjoint equations** with source terms
built from the first-order fields. Structural consequence, and all
WP8.1 needs from it: the second-order pair is *one more instance of
the same classes*, provided the force structure admits (a) sources in
every slot and (b) volume/internal-variable source terms built from
stored first-order trajectories. The force class below keeps its term
set extensible for exactly that; nothing else is built now (WP8.8).

## The common structure (the design)

Extracted from the two gated examples, generalised only where the
ladder's upper rungs demand it. Four pieces; a "specialisation" is
what binds them to one problem class. Proposed names are for review.

### 1. `AdjointForce` — the force object (decision 2)

Carries the data of D_u J, D_φ J, D_SL J, D_ω J in **two kinds of
terms, kept separate** because they are handled differently:

- **Continuous-in-time terms**: time-dependent data on any slot
  (a dual-vector-valued function of forward time for the u- and
  ζ-slots; a coefficient for the SL slot; a 3-vector function for the
  ω slot). They add into the assembled force at every stage of the
  reversed stepping — ordinary loads.
- **Point-in-time terms**: a forward observation time t′ plus the
  impulse data h̃† on any subset of the slots. Each triggers, at the
  reversed time, one generalized fingerprint solve and the m† jump —
  the special handling that justifies the separation.

Conventions: the class speaks *forward* time (the driver owns the
reversal); duals are L-vectors in the `AddForce` layout (the house
parallel convention — forces assemble locally, the solve reduces);
terms accumulate. A static adjoint is point terms only. The slot set
is a small enum, extensible (second order adds a volume term kind;
nothing else changes).

### 2. The adjoint driver

One object that owns the sweep; per decision 1 it is built against the
`LinearQuasiStaticProblem` interface (`AssembleForce`/`AddForce`/
`Solve`/`Displacement`) plus a `ViscoelasticOperator` when time is
present — never against class names (the house dispatch rule). Its
loop, in reversed time:

1. at an observation time: load every slot of the point term, run the
   fingerprint solve (which, per rung, is the plain traction solve or
   the monolithic bordered solve), apply the m† jump, hand the
   impulsive fields to the kernel accumulators;
2. between observations: step m† with the existing steppers
   (`ExponentialTrapezoidStep` as the workhorse), with continuous
   terms entering through `AssembleForce`-time data and the prescribed
   C† playback driving `RefreshWaterLoad`;
3. after each step: fetch the forward state at the matching forward
   time from the trajectory store, and push the (forward, adjoint)
   pair into every registered kernel accumulator.

Point 3 fixes the sweep's shape: **kernels accumulate during the
adjoint sweep; the adjoint trajectory is consumed as produced and
never stored.** Only the forward trajectory is checkpointed. (The
examples store both and pair afterwards — fine for a tutorial, wrong
as the library default.) The driver also serves the degenerate static
case: no stepping, one fingerprint solve per point term.

### 3. The trajectory store (the checkpoint interface, WP8.6 designed in)

The structural fact (plan, "Checkpointing"): the quasi-static state is
the internal-variable field **m alone** — u and ζ are slaved to
(m, load(t)) through one bordered solve (`SolveElastic`). So the store
interface is keyed on the step index and holds m only:

- `Record(n, m)` during the forward run; `State(n)` during the adjoint
  sweep; the slaved fields on demand via `SolveElastic` (with the
  caching economy applying to re-solves).
- Policy objects behind one interface: **store-all** (rungs 1–2;
  optionally memoising u to skip re-solves — a space/time knob, not a
  structure change), **revolve** (binomial schedules with forward
  re-stepping from the nearest checkpoint), **compression** on top
  (m is restrictable to the viscoelastic regions by the
  union-restricted strain map, and reduced-order storage has a
  kernel-error budget measurable against the FD gates).
- Measured at rung 2 already (post-seismic runs are long too), so the
  expensive rungs inherit a proven mechanism.

The budget question (open Q6) does not touch the interface.

### 4. The kernel accumulators

Per-parameter objects registered with the driver; each receives the
(forward, adjoint) pair plus the quadrature weight per pairing step,
and the impulsive fields at observation times, and accumulates

- the **dual vector** (the derivative's components against the model
  space's basis — the export, per the pygeoinf contract: never
  pre-Riesz'd on the wrapped path), and
- optionally an L2 field for display/tutorials.

The pointwise densities are the examples' coefficient classes
librarified (the η/κ/μ logic of `DensityCoefficient`); the quadrature
follows the assembly rule of the elastic forms (the examples' measured
lesson: a higher-order rule is a less faithful pairing, not a more
accurate one). The ice and SL₀ accumulators consume the adjoint
sea-level readout instead of volume fields, and the ice one
accumulates both the quelled (İ, I₀) pair and the raw singular datum
(decision 4). Riesz identification stays downstream (`riesz.hpp` for
in-house use, pygeoinf's Gram solves on the wrapped path — decision 6).

### What a specialisation owns

A rung of the ladder = one specialisation, owning exactly three
things: (i) the **slot binding** — how each AdjointForce slot maps
onto its class's API, including refusing slots the physics lacks
(rung 1–2: u-slot only; rung 3 adds the ζ-slot; rung 4 the SL slot,
C† playback and the mass row; rung 5 the ω datum and the Tisserand
gauge); (ii) its **fingerprint solve** (plain, gravitating, bordered);
(iii) its **kernel set and gates**. The driver, force class, store and
accumulators are shared bricks. The purely viscoelastic no-gravity
adjoint (local post-seismic) is a first-class specialisation, not a
by-product — decision 1's point.

### What WP8.2 does with this

Librarify `adjoint_elasticity` + `viscoelastic_adjoint` onto the four
pieces; the examples' FD tables become regression gate GA0; the
examples themselves shrink to the tutorial layer. Serial and parallel
in the same unit of work, as always (the duals are L-vectors and every
reduction is already in the examples' lambdas).

## To verify in implementation

- ~~The ζ-row dual slot's interaction with the compatibility projection
  (mixed class)~~ — decided and gated at rung 3 (WP8.3): the slot is
  the capability interface `PotentialForceSlot`, implemented by the
  mixed class as an AddForce-analogue on the assembled potential load
  with the compatibility treatment applied to the increment; its sign
  and scale are pinned by the u–φ reciprocity certificate (2e-9). Two
  residues: (i) the REFERENTIAL class's ζ-row version (its load
  projection) — decide and gate when the referential rungs open;
  (ii) a rung-5 trap recorded: the ω-row data are read off the RAW
  assembled potential load at AssembleForce, so slot increments
  deliberately do not feed them — the adjoint's ω-row datum is the
  separate T D_ω J input, but any future term that SHOULD pair with
  ψ_k must not arrive through the slot.
- The prescribed-C playback — briefly thought dissolved (frozen
  shorelines give C† = C₀), but David's correction (11 Oct 2026)
  restores it to the critical path: **shoreline migration is the norm
  in deglacial sea-level work** (second-order formally, not
  negligible at deglacial amplitudes; only mm/cm modern studies may
  freeze C), so the migrating forward is the production
  configuration, frozen-C the option. The adjoint over a migrating
  forward is linear with time-dependent coefficients: store the
  forward ocean-function history and play it back reversed through
  the water machinery — confirm `RefreshWaterLoad`/the frozen-C
  operator path can be driven per reversed step by a stored C(t)
  without re-deriving C from the adjoint state.
- ~~The fingerprint solve's mass row under impulsive loading~~ — and
  the whole eq.-58 question — resolved at rung 4 by the FD gate, the
  other way round from the note's first guess: the DISCRETE bordered
  system is symmetric with the uniform term and ω among its unknowns,
  so the exact discrete adjoint of a potential observation loads ONLY
  the potential block; the continuum reading (the D_φ J datum standing
  beside σ† in the jump condition) is realised automatically through
  the symmetric border columns. Feeding `sea_Sd_`/`rot_r_` as if the
  dual were a surface load double-counts — measured as a 10 % error on
  the full-physics potential-observation gate, then removed. The
  forward's surface loads DO feed those rows, as data of those
  equations. (An SL observation is different: its functional touches
  the uniform-term and ω COMPONENTS of the solution, so its dual has
  genuine entries in those slots — the remaining WP8.4 plumbing.)
- ~~Whether the adjoint stepper needs `InvalidateDisplacement` at every
  observation time~~ — resolved at rung 2: the driver owns it (the
  impulse solve leaves the problem's displacement singular, and the m†
  jump invalidates the cache anyway); no specialisation can forget it.
- The quelled ice kernel's time integrals under adaptive stepping
  (the accumulators assume the driver's grid; the İ kernel is a
  *backward* cumulative integral, cheapest accumulated as a running
  sum during the sweep — confirm the jump bookkeeping at observation
  times).

## The migrating-forward adjoint: the C† playback (design sketch)

Per David (11 Oct 2026), migration is the production configuration,
so the adjoint over a migrating forward is WP8.4's remaining backbone.
The structure, for implementation:

- **Store the ocean function with the trajectory.** The migrating
  forward (`ShorelineMigration`'s Picard) converges C at each step;
  the trajectory store gains a per-step surface-field record of the
  converged ocean function (or equivalently the ocean weight w =
  ρ_w C/g) alongside m. This does not disturb the m-only slaving
  argument: (m, load(t), C(t)) determines everything, and C(t) is
  data-sized (one surface scalar per step).
- **The adjoint operator is time-dependent.** w enters the SYMMETRIC
  boundary blocks folded into the operator, so the adjoint solve at
  reversed time s must carry w(t₁ − s): the water blocks and border
  columns need a per-step refresh path (the analogue of the effective-
  modulus reassembly the steppers already do — the reassembly cadence
  exists; the water-block update rides it). The adjoint stays LINEAR:
  C is played back, never re-derived from the adjoint state.
- **Jumps and kernels inherit the stored history.** The fingerprint
  solve at an observation time uses that time's operator; the ice
  kernels' (1 − C(t)) and (C(t) − C₀) factors read the stored fields —
  with migration on, the D_I₀ term is live.
- **Frozen-C remains the option** (one C for all t; the machinery
  gated at GA2), per David: standard for modern mm/cm amplitudes,
  not for deglacial work.

Implemented 11 Oct 2026: `AdjointSolver::SetTimeSync` (the playback
callback, forward time, fired before every reversed-stepping solve
and every impulse solve), with the same force-hook pattern on the
forward operator as the stepping/migration interleave. Gated with a
prescribed moving ocean against sharp FD, frozen-C control 11×
worse (`PlaybackKernels`; measurements in the plan). The remaining
migrating-forward piece is the per-step recording of the Picard-
converged C and the δC terms' smallness assessment on a real
deglacial configuration.

## WP8.2 implementation findings (11 Oct 2026)

- **Interior observation times** split the node's trapezoid weight
  across the jump exactly as the note prescribes: the forward-right
  subinterval pairs the arrived (pre-jump) adjoint state, the
  forward-left the post-jump one; at the sweep's first node the right
  weight vanishes, recovering the examples' convention. FD-gated at
  3.3e-6 — the same grade as the final-time rows, so the split costs
  no accuracy.
- **Continuous terms** enter through a new
  `ViscoelasticOperator::SetForceHook` — one callback fired after
  every `AssembleForce(t)` the operator issues (unrelaxed, implicit
  and trapezoid effective solves alike), through which the driver adds
  the fillers' duals with the time mapped back to forward time. The
  impulsive solves deliberately bypass the hook: bounded loads carry
  no δ part. One measured floor: pairing a trapezoid-sampled
  time-integrated J against the stepper is not its exact discrete
  adjoint, an O(dt²) mismatch the η rows feel (9.8e-4 at n = 32,
  2.4e-4 at n = 64; the κ/μ rows sit at 3.3e-6).
- **Store-all ≡ re-solve** at 1.1e-11: the m-only state recovery
  through `SolveElastic` reproduces the memoised fields to solver
  roundoff — the slaved-state identity measured, and the seed of GA6.
- **Order matters for viscoelastic kernels:** at order 1 the
  internal-variable space has order zero, and the localised kernel
  rows are not merely inaccurate but WRONG (η-blob at 0.59 of FD,
  μ-blob sign-flipped on the 3-D ball); order 2 restores sign and
  magnitude with the documented O(h) floor on blob rows. Elastic
  kernels are safe at order 1 (pointwise gradients only).
- **A test-configuration trap worth remembering:** an observation
  functional mirror-symmetric to the load makes J vanish identically,
  and every body-wide FD row becomes 0/0 roundoff while localised rows
  still pass (they break the symmetry themselves). Break the symmetry
  in the stations, not the load.

**See:** `adjoint_plan.md` (decisions, work packages, gates);
`doc/Elasticity/ggae388.pdf` §3.3 (slots), §3.7 (eqs. 75–80), §3.8
(eq. 84), §3.9, §4; `examples/adjoint_elasticity.cpp`;
`examples/viscoelastic_adjoint.cpp`;
`include/AdGIA/quasi_static_problem.hpp` (the interface the driver
binds to); `include/AdGIA/viscoelastic.hpp` (`SolveElastic`,
`InverseRelaxationTimes`); `include/AdGIA/riesz.hpp`.
