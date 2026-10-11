# Adjoint GIA: sensitivity kernels and the inverse-problem interface (WP8)

**Status:** design REVIEWED (11 Oct 2026): David's decisions 1-6
recorded below; Q4/Q6/Q8 open with stated defaults. WP8.1 (the theory
note and common-structure design, [`adjoint_note.md`](adjoint_note.md))
reviewed by David 11 Oct 2026, with the constrained-SL₀ formulation
confirmed parked. WP8.2 delivered the same day (gate GA0 and the
measurements under the WP8.2 entry below); next = WP8.3 (rung 3,
+ self-gravity on the mixed class).

## Theory base and what already exists

The theory is settled: Yu, Al-Attar, Syvret & Lloyd (2025,
`doc/Elasticity/ggae388.pdf`) gives the complete mixed-formulation
first- AND second-order adjoint GIA problem including rotational
feedbacks, on a chain (Al-Attar & Tromp 2014; Martinec et al. 2015;
Crawford et al. 2018) that has been checked numerically at various
stages. The referential translation is expected to be small — the
same functional structure in the (u, ζ) variables.

A second theory source (David, 11 Oct 2026):
`doc/Elasticity/ggad434.pdf` (Al-Attar, Syvret, Crawford, Mitrovica &
Lloyd 2024, "Reciprocity and sensitivity kernels for sea level
fingerprints") — the adjoint method in another hat, for the ELASTIC
fingerprint problem. Its reciprocity theorems are ready-made
certificates for our machinery: the loading reciprocity (eq. 8; the
discrete version is gated at rung 3 already), the sea-level
reciprocity ∫SL†ζ dS = ∫SL ζ† dS — self-adjointness of the fingerprint
operator ζ → SL (eqs. 15/17, §2.5) with the Green-function symmetry
G(x,x′) = G(x′,x) (eq. 21) — and the generalized theorems including
rotational feedbacks (§3.2). The reciprocity-with-3-D-ELASTIC-STRUCTURE
case has NEVER been tested numerically: gating it on the water-loaded
mixed solve with laterally heterogeneous moduli is both a WP8.4
certificate and a publishable first (a natural example alongside it).

In-repo groundwork (more than a skeleton):

- `examples/viscoelastic_adjoint.cpp` is Yu §3.7 in miniature: the
  time-reversed adjoint on the SAME forward operators, the
  observation entering as an impulsive elastic solve that JUMPS the
  adjoint internal variable (m† += dev ε(u†_g)/τ), kernels for
  (κ, μ, η) verified against central finite differences, and
  store-everything checkpointing with the honest note that reverse
  reconstruction is unstable (the essential difference from the
  seismological trick).
- `examples/adjoint_elasticity.cpp`: static kernels, FD gates, and
  the derivative-vs-gradient distinction done properly.
- `include/AdGIA/riesz.hpp`: `L2RieszMap` and `SobolevRieszMap`
  (integer order s, (αM + βK)^s Gram solves; order 2 the
  equilibrium-figures default since H2 embeds in C0 in 3-D) — the
  seed of the Hilbert-space numerics pygeoinf consumes.
- `relaxation_law.hpp` already declares the `Gradient()` hook "for
  adjoint problems"; `descent.hpp` has metric-self-adjoint projection
  machinery (equilibrium figures).
- The bordered systems (water, rotation) are SYMMETRIC by
  construction, and the elastostatic operator is self-adjoint.

## Structural observations (why the runway is short)

1. **Adjoint = forward code + four force slots** (Yu §3.3). The
   adjoint problem differs from the forward one only by: (i) a
   sea-level-like load from D_SL J (the ice-load slot); (ii) an
   angular-momentum-like datum from D_ω J (the bordered engine's
   r-vector on the ω rows); (iii) surface terms from D_u J and
   D_φ J (inhomogeneous boundary data on the u- and potential-row
   load slots); and (iv) the ocean function C† being the
   TIME-REVERSED forward one, prescribed rather than dynamic — the
   adjoint is linear with time-dependent coefficients, and
   `RefreshWaterLoad()` driven by a stored C history is exactly that
   mechanism. No transpose assembly anywhere: the operators are
   self-adjoint and the dense border blocks symmetric.
2. **Rigid modes and gauges are already right.** Yu's existence
   condition (their eq. 54) is rigid-compatibility of the adjoint
   loads — the projector machinery's business — and the kernels are
   rigid-invariant combinations; with the generalised door the
   Tisserand gauge carries over verbatim (the rows are the same
   invariant integrals).
3. **Observation-time singularities have a working prototype.**
   Between observation times the adjoint steps normally; at each
   observation time one "generalized fingerprint problem" (an
   elastic + sea-level solve — machinery we have) produces the m†
   jump and the singular parts. `viscoelastic_adjoint` implements
   precisely this without gravity; the GIA version replaces the
   static solve by the monolithic water(+rotation) solve.
4. **The singular ice kernel is handled by the rate formulation**
   (Yu §3.8, the Backus quelling): treat (I₀, İ) as the model
   variables; D_İ Ĵ (their eq. 84, a time-integral of the adjoint
   sea level) has only finite jumps and is optimisation-ready. The
   Sobolev-embedding route (Zuberi & Pratt 2017; Syvret 2022) is the
   stated upgrade path and composes with the metric layer below.

## Kernel set and priorities (David's brief, 11 Oct 2026)

| kernel | priority | notes |
|---|---|---|
| viscosity η | primary | eq. (61): (2μ/η)(d − m) : (T m†); bounded integrand, standard quadrature between observations; lateral viscosity is the point of the whole exercise |
| ice history | needed | via the rate formulation (İ, I₀); the GIA inverse problem is ice + viscosity jointly |
| initial sea level SL₀ | needed | eq. (65) + the §3.9 present-day-matching strategies |
| κ, μ | low priority, low effort | "best determined via seismology"; implement **just to look** — the machinery is shared |
| density ρ | low effort, scientifically interesting | seismology is weak on density (the normal-modes conversation); referential form keeps ρ̃ linear in every operator. **Q3** below: background consistency |
| lithospheric thickness | NOVEL — not done before | an explicit-elastic-lid interface kernel via the referential mapped machinery; see below |

**The lithospheric-thickness kernel.** The two historical reasons it
was not done: (a) the lithosphere can instead be modelled through
lateral viscosity (a high-η lid, effectively elastic over the run) —
which the η kernel already covers; (b) boundary-perturbation theory
in spatial formulations is a faff. Reason (b) has now dissolved: in
the referential frame an interface perturbation IS a perturbation of
the equilibrium mapping — a diffeomorphism direction δφ shifting the
interface — and the derivative of the forward operator in that
direction is assembled from the SAME mapped machinery the
relabelling family certifies. Better still, the perturbation
benchmark family (`doc/benchmarks.tex`, "The perturbation family")
already computes interface-shift derivatives by finite differences
through radial maps on a fixed mesh: the FD gate for the adjoint
shape kernel exists before the kernel does. A useful scientific
by-product: with both the thickness kernel and the η kernel in one
framework, the two lithosphere parameterisations (explicit lid vs
high-η lid) can be compared like for like.

## Design decisions (reviewed 11 Oct 2026)

David's answers to the design pass; Q4, Q6 and Q8 remain open with
tolerable defaults.

1. **Build up slowly; think lego, but optimised (Q1).** Do NOT jump
   to the full GIA problem. Design a COMMON adjoint structure (akin
   to the viscoelastic problem class) with specialisations, so that
   generic problems can be formed — e.g. a purely viscoelastic,
   non-gravitating adjoint for local post-seismic studies is a
   first-class citizen, not a by-product. The ladder: static elastic
   → viscoelastic no-gravity (post-seismic) → + self-gravity →
   + water → + rotation (the full Yu configuration), with kernels
   and FD gates at every rung. Rungs 1–2 are
   `adjoint_elasticity`/`viscoelastic_adjoint` librarified — the
   common structure is extracted from code that already passes FD
   gates, not invented.
2. **The adjoint force class (Q2).** Observations are, in practice,
   almost always POINTWISE IN TIME; but the design must also serve
   STATIC adjoints where there is no time axis at all. The adjoint
   force object therefore carries two kinds of terms, kept separate
   because they are handled differently: CONTINUOUS-in-time terms
   (ordinary time-dependent loads on the four force slots) and
   POINT-in-time terms, each triggering the impulsive observation
   solve and the m† jump at its time. A static adjoint is the
   degenerate case: point terms only, no stepping.
3. **The ρ kernel: equilibrium Poisson yes, momentum balance parked
   (Q3).** The full background-consistency question is complicated
   and practically irrelevant for now. What IS included: the
   equilibrium Poisson coupling — δρ changes the background
   potential Φ₀, and that chain is easy (one adjoint Poisson solve's
   worth of work). What is parked: the hydrostatic momentum-balance
   constraint (δρ changing p₀ and S_e). The kernel's documentation
   states exactly this split.
4. **The ice kernel: quelled AND raw (Q5 refined).** The rate
   formulation (İ, I₀) is the optimisation-facing standard, but the
   RAW singular derivative D_I Ĵ (δ-functions at the observation
   times and all) stays available as an exportable datum — it is the
   honest derivative, and downstream processing choices belong
   downstream.
5. **Mixed is the viscosity workhorse; everything else referential
   (Q7).** Basic η kernels ride the implemented mixed class — likely
   the production vehicle. Interface kernels and all other new
   adjoint machinery are referential only.
6. **pygeoinf handoff stays thin either way.** The spaces and their
   Riesz mappings are set up HERE regardless; pygeoinf (which may
   shift to accept gradients directly where available) then just
   directs processing of data it sees through a pointer — no real
   cost in the division, so the plan does not over-commit to a
   dual-vector-only export: AdGIA owns the Gram/Riesz numerics, and
   whether the boundary hands over derivative components or a
   gradient is a one-line adapter.

## The rheology axis (David's check, 11 Oct 2026)

The ladder below is the GRAVITY axis; the rheology axis is orthogonal
and rides every rung (worked in `adjoint_note.md`, "The rheology
axis"). Two steps, each with its own gate:

- **R1 — general linear Prony. Done 11 Oct 2026.** The pairing
  struct and `IsotropicKernel` are branch-indexed (per-branch m_k,
  m†_k fields from the driver), with the new `MuLongTerm` mode for
  μ_∞ (stiffness-only density) alongside the per-branch Mu (fixed
  η_k) and Eta kernels; the kernel split is documented in
  adjoint.hpp against the stress form σ = κ tr(ε)1 + 2μ_∞d +
  Σ2μ_k(d − m_k). Gate GA8 done: a two-branch body (μ_∞ = 1,
  τ = 1 and 0.3) FD-gated for all six parameters, serial +
  parallel; every deviation is the O(dt²) time floor scaling as
  (dt/τ_k)² (factor-4 under n-doubling; the fast branch dominates
  at 5.5e-3 for n = 64). Region-restricted branches remain guarded
  out of the kernel layer (marker-restricted quadrature when the
  composite-rheology kernels are wanted).
- **R2 — nonlinear laws (nonlinearity in the rheology only).** The
  adjoint is linear with trajectory-dependent coefficients: a
  tangent-adjoint subclass of `ViscoelasticOperator` overriding the
  virtual `Rate`/`LocalExponentialUpdate` with nodal linearisations
  fed from the trajectory store (m-only checkpointing survives — the
  coefficients are functions of (m, load)); parameter kernels through
  `RelaxationLaw::Gradient` (already implemented for the power law).
  Gate GA9 = FD table against nonlinear forward re-runs on the
  power-law body. Slot: after first order is gated on the ladder;
  independent of the gravity rungs.

## The metric layer and the pygeoinf boundary

David's design rule (11 Oct 2026): **don't build the inverse problem
in house.** AdGIA's job ends at defining the numerics of the relevant
Hilbert spaces and operators; `pygeoinf` (v2 refactor,
`~/dev/pygeoinf`, refactor branch / pygeoinf2) handles the generic
parts — priors, optimisation, function-space MCMC (planned for
v2.0.0). Concretely AdGIA exposes:

The v2 contract (surveyed 11 Oct 2026 on the `refactor` branch;
`pygeoinf2/CURRENT_STATE.md` and `DECISIONS.md` are the authorities —
David should sanity-check this digest):

- **Spaces.** Everything is a `HilbertSpace` that knows its own
  inner product; a backend supplies either a `CoordinateSpace`
  (`dim`, `to_components`/`from_components`, `apply_gram`/
  `solve_gram`, `_key`) or, coordinate-free, a `HilbertSpace` plus a
  mass operator wrapped as `MassWeightedSpace` (the
  `backends/mfem_hilbert.py` pattern: "this library conducts, MFEM
  computes"). For AdGIA: true-dof spaces with the mass (or Sobolev)
  Gram applied and solved by AdGIA's own operators — `riesz.hpp` is
  exactly the `apply_gram`/`solve_gram` numerics. Note also
  `pygeoinf2/sem1d` (GLL radial/ball/layered Sobolev spaces over
  `planetmodel.randomfield`): the natural model space for RADIAL
  viscosity profiles before any 3-D inversion.
- **Operators.** Subclass `Operator` with `_value` and `_linearize`
  (one call yielding value + derivative — matching one forward
  solve); the Jacobian is a `LinearOperator` whose adjoint comes in
  through `from_derivative_callables` — AdGIA hands over the
  derivative COMPONENT arrays (dJ/dc, what an adjoint solve
  accumulates) and pygeoinf applies the Gram inverse itself.
  pygeoinf's D-26 is categorical: the derivative is the primitive,
  the gradient is derived through the adjoint, and handing over a
  pre-Riesz'd "gradient" is the error the design exists to make
  hard. So AdGIA EXPORTS KERNELS AS DUAL VECTORS and never
  pre-applies a metric on the wrapped path (`riesz.hpp` stays for
  in-house examples). `ForwardProblem(op, error=GaussianMeasure)`
  closes the contract; `pygeoinf2/testing.py` (`check_space`,
  `check_operator`, `check_gradient`, ...) is gate GA7 ready-made.
- **The metric IS the preconditioner.** The v2 optimizers take no
  preconditioner argument; Sobolev-preconditioned descent falls out
  of posing the model space with the Sobolev metric. David's
  "Sobolev descent-direction functionality" therefore reduces, on
  the wrapped path, to supplying the right Gram pair — no descent
  code in AdGIA.
- **MCMC** is deferred in v2 but its seeds are laid (`log_density`/
  `grad_log_density` hooks, pCN-ready `MaximumAPosteriori`); adding
  the sampler later "means adding the sampler and nothing else" —
  consistent with wrapping now.
- **Packaging.** One distribution ships both packages; AdGIA would
  take a git dependency on the `refactor` branch and
  `import pygeoinf2`, isolated behind one adapter module since the
  package renames to `pygeoinf` at 2.0. `requires-python >=3.12`,
  poetry-core — compatible with the root project. PyMFEM-based
  backends are NOT the route (AdGIA runs C++ MFEM 4.10; PyMFEM is
  4.8): the custom-HilbertSpace route with AdGIA's own bindings is
  the documented alternative, which raises **Q8** below.

## Checkpointing and reconstruction (WP8.6)

Store-everything will not survive production runs. The structural
fact to exploit: **the quasi-static state is the internal-variable
field m alone** — u(t) and ζ(t) are slaved to (m(t), load(t)) through
one bordered solve. So:

- checkpoint m (not u, not ζ): it lives only on the viscoelastic
  regions (the union-restricted strain map machinery), and one
  bordered solve recovers everything else at any checkpoint;
- a revolve-style (binomial) schedule over m-checkpoints trades
  recomputation (forward re-stepping from the nearest checkpoint)
  against memory, with the caching economy (−43 % stepping) applying
  to the re-solves;
- compression is a genuine option ON TOP: m is smooth in the regions
  that matter, so storing it at reduced order / in a coarser space
  with a certified kernel-error budget is worth measuring (the
  kernels pair m against m†, so storage error enters linearly and is
  measurable against the FD gates).

**Q6** below asks for the memory budget to target.

## Work packages (rephased per the decisions: the lego ladder)

- **WP8.1 Theory note + common structure design** (the
  `generalised_rotation_note.md` pattern): digest Yu 2025 into repo
  conventions — the force slots on both classes, the reversed ocean
  function through the migration machinery, the ρ split (equilibrium
  Poisson in, momentum balance parked), the interface shape calculus,
  second order summarised — AND the design of the common adjoint
  structure: the adjoint force class (continuous + point-in-time
  terms, decision 2), the state/checkpoint interface, what a
  "specialisation" owns. **Drafted:**
  [`adjoint_note.md`](adjoint_note.md) (for review).
- **WP8.2 Rungs 1–2 (no gravity):** librarify
  `adjoint_elasticity`/`viscoelastic_adjoint` onto the common
  structure — the static elastic adjoint and the purely viscoelastic
  (post-seismic) adjoint as library citizens with their κ/μ/η
  kernels; the existing FD tables become the regression gates. The
  examples remain as the tutorial layer on top.
  **Status: delivered (11 Oct 2026).** `adjoint.hpp`/`adjoint.cpp`:
  `AdjointForce` (point + continuous terms; continuous terms reach the
  stepping through the new `ViscoelasticOperator::SetForceHook`, the
  driver mapping reversed time back to forward), `ForwardTrajectory`
  (m-only store with store-all and re-solve policies behind one
  interface), `AdjointSolver` (static + quasi-static sweeps; kernels
  accumulate during the sweep; the m† jump and the trapezoid weight
  SPLIT at interior observation times), `IsotropicKernel` /
  `IsotropicKernelDensity` (dimension-general component contractions;
  directional derivatives, dual export, display field). Gate GA0:
  both examples rewritten onto the structure reproduce their FD
  tables DIGIT FOR DIGIT (the static example's Riesz slopes move ~1%
  — its duals now come from the accumulated L2 field, the
  viscoelastic pattern). `TestAdjoint` + `TestAdjointPar` (np 1/2/4)
  gate: static κ/μ 2-D (1.1e-6 measured) and 3-D (4.1e-3, coarse),
  viscoelastic κ/μ/η for final-time (3.3e-6), INTERIOR (3.3e-6; the
  weight split) and CONTINUOUS observations (κ/μ 3.3e-6; η 9.8e-4 at
  n = 32 → 2.4e-4 at n = 64, the O(dt²) quadrature floor of pairing
  a trapezoid-sampled J against the stepper), and store-all ≡
  re-solve at 1.1e-11 (the GA6 seed).
- **WP8.3 Rung 3 (+ self-gravity, mixed class):** force-slot
  plumbing, the adjoint-identity gate (GA1 restricted to the
  gravitating operator), η kernel FD gate, the 1-D spectral
  cross-check (GA3).
  **Status: delivered 11 Oct 2026 except GA3 (open).** The D_φ J slot
  is `PotentialForceSlot` (quasi_static_problem.hpp), a capability
  interface the mixed class implements as `AddPotentialForce` — the
  AddForce analogue on the assembled potential load, with the 2-D
  compatibility treatment applied to the increment; the driver
  discovers it by capability, never by class name. `AdjointForce`
  gains potential observations and continuous potential terms.
  Certificates (`TestAdjoint` + the par leg, np 1/2/4): u–u and u–φ
  reciprocity at 2e-9 (the slot's sign pinned by symmetry — no flip);
  κ/μ/η FD gates on the gravitating Maxwell disc for BOTH a
  displacement and a potential final-time observation: every row at
  or below 1.6e-4 except μ-blob at 1.3e-3 (the curved-element
  strain-map floor of the examples), threshold 5e-3. One convention
  trap recorded for rung 5: the rotational border's ω-row data are
  read off the RAW assembled potential load, so `AddPotentialForce`
  increments deliberately do not feed them.
- **WP8.4 Rungs 4–5 (+ water, + rotation; mixed as the viscosity
  workhorse):** reversed C†, ω-row data, the full-physics η kernel
  (GA2), ice kernels (quelled İ/I₀ + the raw singular derivative,
  decision 4), SL₀; κ/μ/ρ "to look" (ρ per decision 3). Opens with the
  **fingerprint reciprocity certificates** (ggad434: ∫SL†ζ = ∫SLζ†
  on the water-loaded solve; rotation per its §3.2) — spherical
  moduli first, then LATERALLY HETEROGENEOUS 3-D elastic structure,
  the never-tested case (gate GA1b; candidate example).
- **WP8.5 Interface kernels** (referential only, the novel leg):
  shape derivative via the mapped machinery, FD-gated against the
  perturbation family (GA5); lithospheric thickness first; the
  lid-vs-high-η comparison as the demonstration deliverable.
- **WP8.6 Checkpointing:** m-only state storage + revolve schedule +
  the compression measurement; designed into the common structure
  from WP8.1, measured at rung 2 already (post-seismic runs are
  long too).
- **WP8.7 pygeoinf interface:** spaces + Riesz numerics here,
  operators per the thin-handoff decision 6; the sem1d radial
  inversion as the GA7 milestone.
- **WP8.8 Second order** (Yu §4): Hessian-vector action — "much of a
  muchness" structurally; after first order is gated; the common
  structure should not preclude it (the second adjoint pair is one
  more instance of the same classes).

## Open questions (the answered ones moved to the decisions above)

- **Q4 (interface-kernel convention).** Default taken unless
  overruled: a scalar normal-shift field on the reference interface,
  lithosphere base first; CMB and internal viscosity discontinuities
  later.
- **Q6 (checkpoint budget).** To be measured when a production-scale
  run exists; the design (m-only state) does not depend on the
  number.
- **Q8 (the binding route).** Deferred behind the sem1d radial
  milestone (even a slow bridge exercises the whole contract);
  pybind11 under the root project is the presumptive route when
  in-process speed is needed.

## Gates (provisional)

| gate | statement |
|---|---|
| GA0 | rungs 1-2 librarified: the `adjoint_elasticity`/`viscoelastic_adjoint` FD tables reproduced through the common structure (regression grade) — **done 11 Oct 2026**: tables digit-identical; `TestAdjoint`(`Par`) gates them plus interior-time, continuous-in-time and store-all≡re-solve (measurements in WP8.2) |
| GA1 | discrete adjoint identity through the full bordered solve (water + rotation): ⟨Lu, v⟩ = ⟨u, Lv⟩ at solver grade — gravitating (unbordered) part **done 11 Oct 2026**: `GravitatingReciprocity` + par twin, u–u and u–φ reciprocity at 2e-9 (the potential slot's sign certificate) |
| GA1b | fingerprint reciprocity (ggad434 eqs. 15/17, §3.2 with rotation): ∫SL†ζ = ∫SLζ† on the water-loaded solve, spherical AND laterally heterogeneous 3-D elastic structure (the never-tested case) — **done 11 Oct 2026**: `TestAdjointFingerprint`(`Par`, np 1/2/4), Gaussian melt caps on the 3-D ball; defects 2.5e-5 (spherical, → 7.6e-6 at order 2: the readout's projection gap, converging), 3.5e-5 (3-D structure — its first numerical test), 1.9e-5 (rotation + structure); the EXAMPLE is `sea_level_fingerprint -kernel` + `earth_data/plot_fingerprint_kernel` (Boston kernel, 1-D vs `-lateral` 3-D, ice-masked pyslfp maps with Greenland/Antarctica zooms; delivered 11 Oct 2026 per David's figure spec) |
| GA2 | FD gate on J: η kernel on the Yu configuration (spherical, water + rotation), the `viscoelastic_adjoint` table lifted to full physics |
| GA3 | η kernel 1-D cross-check: layered sphere, kernel integrated against radial perturbations vs spectral/Love-number FD |
| GA4 | ice kernel (quelled İ, I₀): FD against ice-history perturbations through the full migration-on forward |
| GA5 | interface kernel vs the perturbation family's FD derivatives (the ready-made gate) |
| GA6 | checkpointed kernels ≡ store-everything kernels (bitwise at matching schedules; certified error under compression) |
| GA7 | pygeoinf round trip: an AdGIA-defined space + operator driving a pygeoinf optimisation on a toy inversion |
| GA8 | multi-branch Prony kernels (R1): per-branch FD table on an SLS/Burgers body, serial + parallel — **done 11 Oct 2026**: two-branch + μ_∞ body, six parameters, all rows at the (dt/τ_k)² time floor (measurements in R1) |
| GA9 | nonlinear-rheology tangent adjoint (R2): FD table against nonlinear forward re-runs on the power-law body |

**See:** `doc/Elasticity/ggae388.pdf` (Yu et al. 2025);
`examples/viscoelastic_adjoint.cpp`; `examples/adjoint_elasticity.cpp`;
`include/AdGIA/riesz.hpp`; `doc/benchmarks.tex` ("The perturbation
family"); `future_work.md` ("Fully nonlinear forward models" — the
second-variation synergy); `~/dev/pygeoinf` (refactor branch).
