# Equilibrium states by constrained optimisation

**Status: formulation agreed, not implemented.** The building blocks it
uses exist (`MinimumDeviatoricEquilibriumStress`, the Poisson
machinery, the mapping layer, the interface-shift perturbation
benchmark); the functional, the adjoints, the Riesz solve and the outer
loop do not.

Formulation of a planned equilibrium-state (and, in its
generalisation, hydrostatic-figure) solver: given a model geometry and
constitution, find a referential density — and in stage 2 also an
equilibrium mapping — for which a valid static state exists, by
minimising the deviatoric stress that the fluid regions would otherwise
be forced to carry. Companions: `doc/gravitating_elasticity.md` §6 (the
equilibrium background generators this formulation builds on) and §1
(continuity of the equilibrium mapping), `doc/slip_interface.tex` (the
slip machinery describes the dynamics about the state, not the state
itself), `benchmarks/perturbation/README.md` (the interface-shift
derivatives that serve as stage 2's gradient check).

## 1. The equilibrium equations and the role of the mapping

The static equilibrium of a general self-gravitating body with material
discontinuities is, at a *natural* reference (particle label = position
at equilibrium), simply the Cauchy-stress balance

```
Div T + ρ ∇Φ = 0  in each region,   [[T n]] = 0 and [[Φ]], [[∇Φ·n]]
continuous across interfaces,
```

with the shape of the body an implicit parameter. To *parameterise* the
shape, pull the state back to a fixed reference through an equilibrium
mapping φ_e; the stress becomes its Piola–Kirchhoff form and the
equations take the referential shapes of Al-Attar & Woodhouse (2010)
or Maitra & Al-Attar (2024, eq. 205, with the caveat below it) — but
nothing more.

**The mapping may be taken continuous without loss of generality.** The
physical positions are continuous across a fluid–solid boundary (no
cavitation or overlap), so a continuous reference admits a continuous
map; the tangential-slip freedom in the labelling of a *static* state
is pure gauge and can be absorbed by relabelling the fluid side. The
fluid-slip terms of the referential formulations are therefore not
needed for describing or parameterising the figure — they are physics
of the *perturbations about* it (where, with the data fixed on the
reference, tangential slip is in general not removable by relabelling;
`doc/gravitating_elasticity.md` §5), and reappear only in that layer. One
scope condition makes this exact: the constitutive data (Ĉ, S_e) must
be understood as transported with the chosen map (the construction
above guarantees this — the data are *defined* at the natural reference
and pulled back); pinning them to a fixed reference while varying the
map would make the map physical rather than gauge.

## 2. The feasibility functional

Given (ρ, φ_e) and the self-consistent Φ (one Poisson solve), the
*minimum-deviatoric-stress* equilibrium field T_min is the solution of
the quadratic minimisation over all equilibrium-compatible stress
fields — the Stokes-like saddle problem already implemented as
`MinimumDeviatoricEquilibriumStress` (Taylor–Hood). Define

```
J(ρ, φ_e) = ½ ∫_{fluid} |dev T_min|² dV   (+ priors, §3).
```

- A solid supports deviatoric stress; a fluid cannot. J = 0 is
  therefore necessary and sufficient for the model to possess a valid
  static state, and J > 0 at the minimiser *quantifies* infeasibility.
- In the fluid, dev T = 0 forces ∇p = ρ∇Φ, whose integrability is the
  classical barotropic structure: ρ constant on equipotentials of the
  self-consistent field. The solid imposes no obstruction — it absorbs
  whatever interface traction results — which is exactly why the
  functional penalises the fluid region alone.
- Generic attainability, with one honest failure mode: if an
  equipotential meets the fluid in disconnected components, ρ(Φ)
  forces equal density on all of them; for geometries violating this,
  no valid model exists and J's positive minimum says so.
- At J = 0 the minimiser is non-unique (the whole barotropic family);
  the priors of §3 select.

## 3. Stage 1: fixed shape, density control

Controls: the referential density on the fluid regions, the solid (and
hence the CMB shape) held fixed. Constraints and priors: total mass
(hard constraint, by projection in the gradient metric of §4), and a
Tikhonov pull toward a reference profile (PREM) that both selects
within the J = 0 family and convexifies the tail of the descent.

**Adjoints.** J depends on ρ through two solves: Poisson (Φ from ρ)
and the stress generator (T_min from ρ, Φ). Both are linear; the
derivative of J is assembled from two transpose solves — the adjoint of
the Stokes-like saddle (differentiation through a quadratic minimiser:
the same saddle, transposed data) and the Poisson adjoint carrying the
self-gravity chain dΦ/dρ. Hessian actions are the corresponding
second-order adjoints of the same pair. No new solver classes are
required; the new object is the outer loop.

## 4. Derivatives are not gradients: the Sobolev metric

The derivative J′(ρ) is a functional (an element of the dual); a
*gradient* requires a metric, and the choice is the implicit
preconditioner of the whole scheme. Pose the density optimisation in a
Sobolev space on the fluid, with inner product

```
⟨u, v⟩ = ∫_{fluid} ( α u v + β ∇u · ∇v ) dV ,
```

and obtain the gradient g from the derivative by one elliptic solve
⟨g, v⟩ = J′(ρ)[v] for all v — a Helmholtz-type problem the existing
Poisson machinery covers. The smoothing length √(β/α) is a principled
knob: it filters the mesh-scale components of the raw derivative into
smooth descent directions. Expected behaviour (from the analysis; not
yet measured in this library): the ill-posedness of the underlying
problem does not disappear, but the updates are not appreciably
polluted by it — the high-frequency null- and near-null content that
would otherwise accumulate over iterations is suppressed at source. The mass constraint is enforced by projection
*in this metric* (one scalar correction per step); the PREM prior adds
its term to the derivative before the Riesz solve.

**Pose the metric on a larger domain than the control.** In practice
the parameter field is regarded as living on the *whole body* (or any
convenient enclosing domain), with the forward model reading only its
restriction to the fluid region. This removes the need to impose any
boundary condition on the model parameters at the fluid–solid
boundary — precisely where the density action concentrates and where
an artificial Dirichlet or Neumann choice would bias the updates. The
Riesz solve then carries natural conditions only on the outer boundary
(Dirichlet there), the smoothing extends the derivative smoothly
across the CMB, and nothing changes in the forward chain.

**Metric orders beyond H¹.** Even-integer-order Sobolev metrics come
for free by applying the (Dirichlet-on-the-outer-boundary) inverse
Laplacian the appropriate number of times — each application one solve
of existing machinery. Non-integer orders are obtained by fractional
powers of the same operator through Dunford–Schwartz-type operational
calculus: in practice the resolvent integral representation
(Balakrishnan-type), realised as a quadrature over shifted Helmholtz
solves (A + t I)⁻¹ — again nothing but the solvers already in the
library, composed. The order of the metric is then a continuous
regularisation parameter. This Riesz/fractional-operator machinery is
deliberately general: it is shared infrastructure for the
inverse-problem programme the library is ultimately pointing at, where
the same derivative-to-gradient discipline, extension domains and
fractional metrics recur for every parameter field.

## 5. Stage 2: letting the shape vary

Generalisation to hydrostatic figures: admit φ_e among the controls.
Two structural points, both grounded in measurements documented
elsewhere (gauge semi-convergence: `doc/gauge_penalty_iteration.tex`,
"Semi-convergence and the operating point"; side selection of
shift-type maps: `benchmarks/perturbation/README.md`):

- **Gauge.** (ρ, φ_e) carries the relabelling gauge (ρ̃ = J ρ∘ξ with
  φ_e∘ξ describing the same model). J is gauge-invariant, so its
  derivative annihilates gauge directions and the gradient — in any
  metric — is orthogonal to them: exact-arithmetic descent never
  drifts along the gauge. *Discretely* the invariance is only
  approximate, and small spurious gauge components accumulate over
  many iterations (the optimisation analogue of the gauge
  semi-convergence measured in the penalty studies). Rather than
  projecting them away, **eliminate the redundancy: parameterise φ_e
  explicitly by interface shapes** (CMB and surface topography
  coefficients, identity at the DtN sphere) — which also matches the
  mapping layer's architecture and keeps the computational interfaces
  spherical.
- **Discretisation discipline.** The map's gradient kinks must remain
  attribute-aligned on the mesh, and shift-type maps must be used in
  their interpolated-F form (the side-selection defect and its cure are
  documented in `benchmarks/perturbation/README.md`). The shape
  derivative of J carries moving-domain transport terms — the fluid
  region, where dev is penalised, moves with φ_e — assembled with the
  Nanson layer; they must not be forgotten.

The derivative of the state with respect to interface location is
exactly what the interface-shift perturbation benchmark certifies
(degree-0 derivative agreement with 1-D theory at the sub-percent
level), so stage 2's gradients arrive with their verification harness
pre-built.

## 6. Implementation steps

1. **Outer-loop driver** (a `benchmarks/figures/` family, or `tools/`):
   inner solves = Poisson + `MinimumDeviatoricEquilibriumStress`;
   evaluate J.
2. **Adjoints**: transpose saddle solve and Poisson adjoint; assemble
   J′(ρ); finite-difference checks of J′ against directional
   differences (the standard kernel-vs-FD validation; this would be the
   library's first adjoint computation).
3. **Sobolev gradient**: the Riesz elliptic solve, mass projection,
   PREM prior; a smoothing-length study on the first test case.
4. **Milestone 1 — fixed-shape density restoration**: on fluid_core,
   start from a deliberately non-barotropic fluid density and verify
   the loop recovers the known hydrostatic profile with J driven to
   the discretisation floor.
5. **Stage 2**: interface-shape parameters, the moving-domain terms,
   gradient checks against the shift-derivative benchmark; then
   equilibrium *figures* proper once rotation (centrifugal potential)
   is added to the background (equilibrium figures with rotation), with
   the thin-lithosphere limit as a special case.
6. **Hessian actions** (second-order adjoints) once gradient descent
   is validated, for Newton–Krylov in the tail; the Krylov iterations
   inherit the gauge-orthogonality discussion of §5.

Deliberately deferred: rotation and Coriolis in the perturbation layer
(Tisserand-frame machinery), finite-strain reference changes, and any
coupling to the viscoelastic evolution.
