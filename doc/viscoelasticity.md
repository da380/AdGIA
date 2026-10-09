# Quasi-static problems, rheologies and viscoelastic time stepping

Method notes for `quasi_static_problem.hpp`, `rheology.hpp`,
`relaxation_law.hpp` and `viscoelastic.hpp`.

The layering is this. A **quasi-static problem** is anything that, at a time
`t`, solves a linear elliptic system whose displacement part can receive
extra dual-vector forces. A **rheology** owns the material data and says how
to assemble the (effective) elastic stiffness. The **viscoelastic operator**
owns internal variables on the displacement mesh and drives the problem
through a small interface. Self-gravitation, fluid cores and whatever else
the equilibrium involves live *inside* the problem, so the viscoelastic layer
runs unchanged on coupled problems and never sees a potential.
`viscoelastic.hpp` depends only on the interface and on the strain
integrators; that dependency direction is the point of the design.

## 1. The model

Generalised Maxwell (Prony series) body, isotropic or anisotropic:

```
σ = C_U ε − Σ_k C_k m_k,    C_U = C_∞ + Σ_k C_k,    ṁ_k = (ε − m_k)/τ_k,   m_k(0) = 0.
```

`C_k` is the relaxable tensor of branch `k`. For the isotropic body
`C_k = 2μ_k P_dev`; only the deviator `d = dev ε` matters, the internal
variables are trace-free, and
`σ = κ tr(ε) I + 2μ_∞ d + Σ_k 2μ_k (d − m_k)`. The classical Maxwell body is
μ_∞ = 0 with one branch (τ = η/μ); Burgers is μ_∞ = 0 with two. For an
anisotropic body *which part of the tensor relaxes is a modelling choice*
made by the branch coefficient: `DeviatoricMaxwell()` relaxes
`P_dev C P_dev`, which reproduces the isotropic Maxwell body for an isotropic
`C`; a transversely isotropic coefficient with only L and N relaxes the shear
moduli alone. The internal variables are then full symmetric tensors, since
`C_k m_k` is not trace-free in general.

The elastic operator is assembled with the **unrelaxed** modulus `C_U`, and
the branches appear only as forces:

```
K_U u = f_ext(t) + Σ_k Bᵀ (C_k m_k),        ṁ_k = (D u − m_k)/τ_k,
```

with `B_IJ = ∫ Φ_I : ε(φ_J)` assembled once with a unit coefficient, all
material weighting applied pointwise at the internal-variable nodes, and `D`
the strain map (§4). The material data has one owner, the `Rheology`: the
problem assembles with the integrators of `MakeStiffness()`, and the operator
reads the branch data from the same object, so the two layers cannot
disagree.

**Two dimensions.** The deviator is that of the space dimension, so in 2-D
the isotropic rheologies model a two-dimensional continuum (λ = κ − μ), not
plane strain of a 3-D body. The transversely isotropic tensor coefficient, by
contrast, gives the plane-strain restriction of the 3-D tensor
(`elastic_tensors.md`); the two conventions should not be mixed in one model.

**The bulk/deviatoric split needs no custom integrator.**
`κ div u div v + 2μ dev ε(u) : dev ε(v) = λ div u div v + 2μ ε(u) : ε(v)`
with λ = κ − 2μ/d, so MFEM's `ElasticityIntegrator(coef, q_λ, q_μ)` (which
sets λ = q_λ·coef, μ = q_μ·coef) gives both parts: `(κ, 1, 0)` and
`(μ, −2/d, 1)`.

### The rheology classes

| Class | Material | Stiffness integrators | Internal variables |
|---|---|---|---|
| `IsotropicElasticRheology(dim, κ, μ)` | isotropic elastic, no branches | two `ElasticityIntegrator`s (κ/μ split) | none |
| `AnisotropicElasticRheology(dim, C)` | elastic with a Mandel tensor (`elastic_tensors.md`), no branches | one `ElasticTensorIntegrator` | none |
| `IsotropicMaxwellRheology(dim, κ, μ_∞, branches)` | isotropic generalised Maxwell; each `MaxwellBranch` is (μ_k, τ_k, optional law); `Maxwell(dim, κ, μ, τ [, law])` is the classical body | the κ/μ split with μ_U or μ_∞ + Σ β_k μ_k | trace-free |
| `AnisotropicMaxwellRheology(dim, C_∞, branches)` | anisotropic generalised Maxwell; each `AnisotropicBranch` is (C_k, τ_k, optional law); `DeviatoricMaxwell(dim, C, τ [, law])` relaxes P_dev C P_dev | one `ElasticTensorIntegrator` with C_U or C_∞ + Σ β_k C_k | full symmetric |
| `CompositeRheology(dim, regions)` | different rheologies on disjoint sets of element attributes (§6) | each region's, restricted to its marker | trace-free only if every region's are |
| `ReferentialElasticRheology(dim, Ĉ, S_e, φ_e)` (`referential_problem.hpp`) | the general pre-stressed elastic state of the referential formulation, no branches | `MaterialStiffnessIntegrator` + `GeometricStiffnessIntegrator` (`elastic_tensors.md`) | none |

A purely elastic rheology passes through the viscoelastic operator with an
empty internal state (a purely elastic evolution under time-dependent
loads); its relaxation-weight calls are no-ops. The Maxwell classes provide
`UnrelaxedElastic()` and `LongTermElastic()`, the instantaneous (t = 0⁺) and
fully relaxed (t → ∞) elastic solids, which are the limits a time-domain run
must approach. All rheologies hold pointers to the caller's coefficients,
which must outlive them; they are movable, not copyable.

**Reference state.** `Rheology::EquilibriumMapping()` returns the
equilibrium mapping φ_e when the reference state is non-natural in the sense
of Al-Attar & Crawford (2016) (the particle label is not the equilibrium
position), and null for natural labels (φ_e = id). Every class above except
`ReferentialElasticRheology` returns null; that one always returns its
mapping (the identity for a natural state). The stiffness integrators carry
the mapping themselves; problems consult it for everything else that depends
on it, such as the rigid modes of the mapped positions (`null_space.md`) and
boundary-area factors. Whether the equilibrium stress is hydrostatic is
likewise a property of the rheology's data, not of the problem class.

## 2. The problem interface

Per evaluation time `t`:

```
AssembleForce(t);   // time-dependent data to t; external loads; increments cleared
AddForce(f); ...    // superpose dual vectors on the displacement (accumulates)
Solve();            // displacement <- K⁻¹ (external + increments)
```

- `AssembleForce(t)` is called at every stage of a time integrator, with
  possibly non-monotone `t`; it must be cheap and idempotent. MFEM's explicit
  solvers call `SetTime(t + cᵢ dt)` and then `Mult`, so the operator uses
  `GetTime()` there, never a stored time, and time-dependent load
  coefficients must be registered with the problem rather than sampled once.
- `AddForce` takes the vdof (L-vector) layout, that of a `LinearForm` before
  `FormLinearSystem`. In parallel the problem applies `Pᵀ` once inside
  `Solve()`. `Bᵀ` of a parallel mixed form in the local layout produces
  exactly such a vector, so the force path needs no `ParallelAssemble`.
- `Solve()` may be iterative inside but is a black box; linearity in the
  *forces* is part of the contract.
- The problem holds no history: the state `m` together with `t` is
  everything, and `SolveElastic(m, t)` recovers a consistent displacement
  after a restart.
- There is one displacement field. Several solid regions share it on a
  possibly disconnected SubMesh, and regional material differences belong to
  the rheology (§6).
- `Solve()` returns false if the linear solver did not converge. Problems
  carrying further unknowns (a potential) keep them internal.

### The problem classes

`LinearQuasiStaticProblem` is the abstract interface above;
`LinearQuasiStaticProblemBase` implements it on a serial or parallel
displacement space (the space decides) with the rheology's stiffness,
time-dependent load registration (`RegisterTimeDependent`), lazy
reassembly, the default preconditioned CG (Gauss–Seidel in serial,
systems BoomerAMG with nodal coarsening in parallel; default relative tolerance
1e-12) and the gauged-fluid option. The concrete classes:

| Class | Header | Problem |
|---|---|---|
| `LinearQuasiStaticTractionProblem` | `quasi_static_problem.hpp` | traction on marked boundaries, no essential conditions; CG on P A P with the rigid-mode projector (`null_space.md`); optional mass-weighted gauge (`SetMassWeightedGauge`) |
| `LinearQuasiStaticClampedProblem` | `quasi_static_problem.hpp` | prescribed displacement on one set of boundary attributes (time-dependent or homogeneous), traction on another |
| `LinearQuasiStaticMixedSelfGravitatingProblem` | `mixed_problem.hpp` | self-gravitating body, referential displacement with the spatial potential perturbation on an enclosing ball (`self_gravitation.md`) |
| `LinearQuasiStaticReferentialSelfGravitatingProblem` | `referential_problem.hpp` | self-gravitating body, fully referential (displacement and potential), general reference state through a `ReferentialElasticRheology` (`gravitating_elasticity.md`) |
| `LinearQuasiStaticReferentialSelfGravitatingSlipProblem` | `referential_problem.hpp` | the referential problem with a slipping fluid–solid interface (`slip_interface.tex`) |

The class names are built from fixed slots, `[Linear|Nonlinear]
[QuasiStatic|Dynamic] [Mixed|Referential]? [SelfGravitating]? [variant]
Problem`. The elasticity is referential in every class; the `Mixed` /
`Referential` slot says how the *gravity* is described and so appears only
for self-gravitating problems. Properties of the reference state belong to
the rheology, not to the class name; a class that requires a special
reference state says so (the mixed formulation requires a hydrostatic,
natural one). The traction and clamped problems accept a
`ReferentialElasticRheology` and are then the non-gravitating referential
problem; their tractions and prescribed values are referential fields.

The viscoelastic operator needs `SupportsRelaxationWeights()` for the
trapezoid and implicit schemes. The base class and the mixed problem
support it; the referential classes take a `ReferentialElasticRheology`,
which has no branches, so they are used elastically.

**Gauged fluids.** `SetGaugedFluid(marker, μ_g, ε, refinements, penalty,
map)` treats the marked attributes as an inviscid fluid in the relabelling
formulation: the rheology supplies the fluid's bulk stiffness (zero shear),
and a gauge-fixing shear penalty ε Q is added to the *solver* operator only.
`Solve()` removes the O(ε) bias by iterated Tikhonov refinement, each
refinement solve starting from zero. With a non-identity `map` the
deviatoric penalty is assembled covariantly with
`ElasticTensorIntegrator(C_dev, map)` (`elastic_tensors.md`, section
"Material and geometric stiffness"). The formulation and its verification
are in `gauged_fluid.md`.

**Relaxation weights.** The implicit and exponential-trapezoid schemes
eliminate `m^{n+1}` and need the stiffness reassembled as
`C_∞ + Σ_k β_k(x) C_k` with pointwise weights β_k, which vary in space
because τ_k does. `SetRelaxationWeights(beta)` retargets a redirectable
coefficient inside the rheology's `ElasticStiffness` and marks the operator
stale; no integrator is rebuilt. One stiffness object exists per problem, so
several problems may share a rheology without interfering. In the
self-gravitating problem only the displacement block is reassembled; the
potential block, the coupling and the DtN are built once.

**Reassembly.** Each assembly builds a fresh `BilinearForm` that borrows the
integrators from a never-assembled template form, rather than calling
`Update()` on the old one: `Update()` only zeroes the matrix in place, and
`Finalize(skip_zeros)` may have dropped entries the new coefficient needs
(`mfem_notes.md`).

**Warm starts.** The main solve of a problem is warm-started from its
previous solution. MFEM's relative tolerance is relative to the *initial*
residual. A warm-started solve, the normal case in time stepping, starts
from a residual of about 1e-12 ‖b‖, so the target becomes about 1e-24 and CG
stalls at its iteration limit. `SetWarmStartTolerance` sets an absolute
tolerance `rel_tol · √(M B, B)` in the preconditioner norm, the target a cold
start would have had, and reports `B = 0` so that the solve is skipped.

**Preconditioner reuse.** With a variable step or state-dependent relaxation
times the effective operator changes at every step, and the BoomerAMG setup
is the dominant cost of an assembly. `SetPreconditionerReuse(factor)`
(default 2) keeps the preconditioner across reassemblies while the iteration
count stays within `factor` of its count at setup, then rebuilds. The matrix
is always the current one; only the preconditioner is allowed to lag. The
form and matrix the preconditioner was built on are kept alive as long as it
is in use, since the preconditioner refers to its matrix.

## 3. Time stepping

With h_k = dt/τ_k and dⁿ = D uⁿ, and the elastic solve of the trapezoid and
implicit schemes taken with the effective modulus `C_∞ + Σ_k β_k C_k` of §2:

| Scheme | Update of m_k | Elastic solve | Order, stability |
|---|---|---|---|
| explicit RK (`Mult`) | any explicit MFEM solver | one per stage, `K_U` | RK order; dt ≲ 2.8 τ_min for RK4 |
| ETD1 (`ExponentialEulerStep`) | `e^{−h} m + (1−e^{−h}) dⁿ` | one per step, `K_U` | 1st; unconditionally stable |
| exponential trapezoid (`ExponentialTrapezoidStep`) | `e^{−h} m + a dⁿ + b d^{n+1}`, a = (1−e^{−h})/h − e^{−h}, b = 1 − (1−e^{−h})/h | one per step with weights β_k = 1 − b_k and force `Bᵀ Σ C_k (e^{−h_k} m_kⁿ + a_k dⁿ)` | 2nd; exact for a strain linear in time; no step restriction |
| backward Euler (`ImplicitSolve`) | `(m + h d^{n+1})/(1+h)` | weights β_k = 1/(1+h_k), force `Bᵀ Σ C_k m_kⁿ/(1+h_k)` | 1st; L-stable |
| SDIRK (`ImplicitSolve`) | as backward Euler with γ·dt per stage | one per stage | `SDIRK23Solver(2)`: 2nd, L-stable |

The exponential trapezoid and SDIRK23 are the second-order fixed-step
schemes for loading problems with dt ≫ τ in part of the domain: the
exponential trapezoid takes one solve per step and keeps a constant
operator while dt is constant; SDIRK23 takes two and keeps its order on
stiff multi-branch bodies ("Choosing a scheme" below). Crank–Nicolson is deliberately
absent; it is not L-stable and oscillates for h ≫ 1.

### Driving the operator

`ViscoelasticOperator` is an `mfem::TimeDependentOperator`, so MFEM's
`ODESolver`s drive it directly for the explicit and implicit schemes. The
exponential schemes are reached through adaptors in `viscoelastic.hpp`:

| Scheme | Solver object |
|---|---|
| explicit RK | any explicit MFEM solver (e.g. `RK4Solver`) on `Mult` |
| backward Euler | `mfem::BackwardEulerSolver` on `ImplicitSolve` |
| SDIRK23 | `mfem::SDIRK23Solver(2)`, the L-stable second-order variant (MFEM's default `SDIRK23Solver()` is the third-order, A- but not L-stable one; `mfem_notes.md`) |
| ETD1 | `ExponentialEulerSolver` |
| exponential trapezoid | `ExponentialTrapezoidSolver` |
| adaptive exponential trapezoid | `AdaptiveExponentialTrapezoidSolver` (§5) |

`SolveElastic(m, t)` makes the problem's displacement consistent with any
`(m, t)` and is how a driver observes the displacement between steps;
`SyncFields(m)` copies the state into the output fields registered by
`RegisterFields(dc)`, and `MinRelaxationTime()` gives the explicit stability
scale.

**Load jumps.** A step that ends at a jump of the external load is taken
with the load's left limit; the next step must start from the right limit.
The displacement cache is keyed on `(m, t)` and cannot tell the two apart,
so a driver calls `InvalidateDisplacement()` at the jump. Jumps should also
lie on the step grid: a jump inside a step costs every scheme its order,
while a kink (a jump in the load's derivative) costs a second-order scheme
nothing and caps RK4 at second order.

### Choosing a scheme

The schemes trade order, stability and the number of operator assemblies
(cost is best counted in elastic solves, since each is one quasi-static
system):

- **Nothing stiff** (τ_max/τ_min modest, dt limited by accuracy rather
  than by τ_min): RK4 is the cheapest at tight tolerances; its operator
  never changes, so it never reassembles.
- **Stiff bodies** (a wide range of relaxation times, laterally varying
  viscosity, a power law): RK4 is bound by dt ≲ 2.8 τ_min whatever the
  accuracy wanted. The exponential trapezoid and SDIRK23 are the robust
  fixed-step choices and cost about the same per unit accuracy (one solve
  per step against two with a smaller error constant). The exponential
  trapezoid wins when the strain is close to linear over a step (it is
  exact for a piecewise-linear strain); SDIRK23 wins on a stiff
  multi-branch body under stress control, where the exponential
  trapezoid's order drops to about 1.5 (stiff branches put components
  into the strain that vary inside a step; SDIRK23's L-stable stages damp
  them).
- **Relaxation after a step load** (a Heaviside load, load and unload):
  the adaptive trapezoid is cheapest, taking small steps through each
  transient and striding once the response is quiet.
- **Sustained periodic forcing**: adaptivity is the worst choice, because
  the conservative error estimate keeps every step small for the whole run;
  use a fixed-step second-order scheme.
- **First-order schemes** (ETD1, backward Euler) are not competitive
  beyond errors of about 1e-3. ETD1 is exact for a piecewise-constant
  strain and needs no reassembly; it lags systematically under creep.
- **Assembly cost.** The adaptive trapezoid reassembles the effective
  operator at every accepted step, a fixed-step implicit or trapezoid
  scheme once per step size, RK4 and ETD1 never. Where assembly and
  preconditioner setup dominate, this counts against adaptivity
  (preconditioner reuse, §2, softens it).

The evidence is `examples/viscoelastic_schemes.cpp`, the stepping survey of
`benchmarks/viscoelastic/stepping/`, and the box benchmarks of
`benchmarks/viscoelastic/box/`, documented in `benchmarks.tex` (the
viscoelastic family).

Details that matter:

- `ImplicitSolve` must return the *rate* `k` with `k = f(m + dt·k)`, not
  `m^{n+1}`. SDIRK schemes call it with γ·dt, so several effective operators
  may be needed per step.
- The switch between the unrelaxed and an effective operator is lazy and
  costs one reassembly, so mixing schemes works but is not free.
- After a trapezoid or backward-Euler step the problem's displacement is
  already consistent with the new state; after an SDIRK step it belongs to
  the last stage state, so an observation costs an extra unrelaxed solve.
  The operator caches the `(m, t)` for which the displacement is consistent
  (compared exactly, with an all-reduce in parallel), and the next step's
  dⁿ or a call to `SolveElastic` reuses it. ETD1 and explicit stages
  invalidate the cache, and so does `InvalidateDisplacement()`.
- There is one elastic solve per stage, in `ElasticUpdate`, and never one
  inside the pointwise kernels, so that the solve count is predictable.

## 4. Internal variables and the strain map

The internal variables live on an L2 nodal space with `Ordering::byNODES`,
so component `c` of a branch occupies `[c·n_d, (c+1)·n_d)`, in the component
convention of `TraceFreeSymmetricMatrixIndex` / `SymmetricMatrixIndex`
(`index.hpp`). L2 nodes are element-interior, so attribute-wise discontinuous
material data are sampled at them without ambiguity.

**Two strain maps** `D : u ↦ ε(u)` (or its deviator) at the nodes: nodal
interpolation, and the Galerkin projection. The projection makes the
discrete adjoint of a step exactly the transposed step, and is the default.

**The projection is `(G⁻¹ ⊗ M⁻¹) B`, not `M⁻¹ B`.** The basis tensors `E_c`
of the component convention are not orthonormal. Off the diagonal
`E = e_j e_kᵀ + e_k e_jᵀ` has |E|² = 2. In the trace-free basis the diagonal
ones are `e_j e_jᵀ − e_{d−1} e_{d−1}ᵀ`, again with |E|² = 2, and in 3-D the
two of them overlap (`E₀ : E₃ = 1`). `G_{cc′} = E_c : E_{c′}` is their
Frobenius metric; without `G⁻¹` the strain comes out exactly twice too large
in 2-D. For full symmetric tensors `G = diag(1, 2)`. `M⁻¹` is the exact
element-block inverse of the scalar L2 mass matrix.

**The internal order must resolve ε(u) exactly.** Then the projection is the
identity on representable strains, the two maps coincide, and the
effective-modulus elimination is exact, since `Bᵀ β C D = K(βC)`. On
simplices that order is p − 1. On tensor-product elements the gradient of a
Q_p field has Q_{p,p−1} components, so the order must be p: with p − 1 on
quadrilaterals the long-time limit of a clamped body misses the μ_∞ elastic
solution by 0.6 %. The default `internal_order < 0` picks p − 1 on simplex
meshes and p when the mesh has any other geometry (reduced across ranks).

**The effective modulus between nodes.** The weights β_k are nodal
`GridFunction`s on the scalar L2 space, and the effective modulus is
evaluated as the coefficient chain `C_∞ + Σ β_k C_k` at quadrature points,
not interpolated as a single nodal field: the two agree at the nodes, and
the chain is exact in between.

**Anisotropic branch data** are stored per node as the n_s × n_s matrix that
acts on unscaled tensor components, i.e. the Mandel tensor with the √2
scalings folded in.

## 5. State-dependent relaxation times

Following Crawford et al. (2017, Appendix A), after Simo & Hughes (1998), the
elastic part stays linear and the relaxation times depend on the state,

```
τ_k = τ_k0(x) · F_k(ε, σ, m_k),
power law:  F = 1 / (1 + γ (‖dev σ‖ / 2μ₀)^{n−1}),
```

so that the slow-deformation limit is a composite Newtonian and power-law
fluid: diffusion creep at low stress, dislocation creep (n = 3) at high
stress. The nonlinearity is *diagonal in space and scalar*, so no nonlinear
global system arises and everything global remains a linear elastic solve.
A `RelaxationLaw` is therefore a pointwise factor with parameter fields
sampled at the internal nodes, plus an optional gradient with respect to the
stress (for adjoints). There is one operator, and linearity is a property of
the rheology (`IsLinear()`): a linear body skips the re-evaluation and the
corrector. The nodal stress is σ = C_U ε − Σ_k C_k m_k from the unrelaxed
and branch moduli sampled at the nodes; in the trace-free isotropic case its
deviator is T = 2μ_∞ d + Σ_k 2μ_k (d − m_k) = 2μ_U d − Σ_k 2μ_k m_k, the full
deviatoric stress (the long-term part included), which is what drives the
effective relaxation time.

τ can drop by one or two orders of magnitude at stresses a few times the
transition stress, which rules out explicit stepping and changes the
accuracy scale during a run.

**Predictor–corrector.** For the trapezoid the weights β_k need τ over the
step:

1. take τ* at the start state and step;
2. re-evaluate τ* at the midpoint state `((mⁿ + m^{n+1})/2, (dⁿ + d^{n+1})/2)`
   and repeat the step;
3. iterate, `SetCorrectorIterations(max, tol)`, with an early exit when the
   nodal times stop changing.

One corrector pass (the default) gives second order for smooth τ(t).
Backward Euler does the same with the end state. Each pass is an elastic
solve with a *different* effective operator, which is what preconditioner
reuse (§2) is for.

**Adaptive stepping** (`AdaptiveExponentialTrapezoidSolver`). Exponential
integrators have no stability limit here,
so adaptivity is purely about accuracy: resolving the times when τ collapses
and striding across quiet periods. After a trapezoid step its ETD1 companion
(nodal work only, no solve) gives an embedded first-order estimate,

```
err = RMS of (m − m̂) / (atol + rtol · max(|mⁿ|, |m^{n+1}|)),
dt ← dt · clamp(0.9 err^{−1/2}, 0.2, 4),    reject and retry when err > 1.
```

The estimate is that of the first-order companion, so the tolerance is
conservative for the second-order solution that is propagated, by a factor
of order τ/dt; choose rtol accordingly. A linear body benefits in the same
way, since its effective operator depends on dt. The controls are
`SetTolerances(rtol, atol)` (defaults 1e-4, 1e-10), `SetStepBounds(dt_min,
dt_max)` and `SetStepFactors(shrink, grow, safety)` (defaults 0.2, 4, 0.9).
`Step(x, t, dt)` takes a step of at most `dt` and returns the proposed next
step in `dt`; `Integrate(x, t, t_final, dt)` runs to `t_final`, hitting it
exactly; `NumAcceptedSteps()`, `NumRejectedSteps()` and
`LastErrorEstimate()` report.

## 6. Composite rheologies

`CompositeRheology` assigns different rheologies to different regions of one
displacement space: different numbers of branches, elastic regions inside a
viscoelastic body, anisotropic regions inside an isotropic one, a relaxation
law in one region only.

- **Regions are sets of element attributes**, disjoint (checked at
  construction) and covering every attribute present on the mesh (checked
  when the stiffness is attached to a form). On a SubMesh the attributes are
  inherited from the parent, so regions are the gmsh physical volumes. An
  elastic region is a region with an elastic rheology; it contributes
  integrators and nothing else.
- **Branches** are the concatenation of the regions' branches, in region
  order. A branch's modulus is masked to zero outside its region. Its
  relaxation time outside is a large dummy (1e300), so that the variable
  there neither moves nor limits an explicit step.
- **Stiffness**: one `ElasticStiffness` per region, each added with its
  element marker through MFEM's marked domain integrators. The borrowing
  form constructor copies marker *pointers*, so the composite stiffness owns
  copies of the markers.
- **Trace-free or full.** Internal variables are trace-free only when every
  region's are. Otherwise the isotropic regions present their branches as
  full tensors `2μ_k P_dev`, at a storage cost of n_s/(n_s − 1), i.e. 4/3 in
  2-D and 6/5 in 3-D, in that mixed case only.
- **Region-restricted state.** Each branch is stored and evolved on the L2
  nodes of its region's elements only (`Rheology::BranchMarker`), so the
  state and the per-step work are the sum over regions of their own
  branches. Because an L2 node's row of `B` involves its own element only,
  the per-branch coupling rows are a plain row extraction from the one
  global `B`. For the same reason a region SubMesh with a dof injection
  would add nothing: the submesh L2 space is a row selection of the parent
  one. The per-step strain map is restricted the same way: when every
  branch is confined to a region, the Galerkin (or interpolation) map is
  applied on the union of the branch nodes only — an exact row restriction
  by the same element-locality, shared by all branches and decided per
  rank in parallel (`RestrictedStrainMap()`); the public `ComputeStrain()`
  always evaluates on the whole mesh. A whole-mesh rheology has exactly
  the unrestricted layout.
- Output fields. `RegisterFields` names branch k's field
  `internal_variable_<label>` with the rheology's `BranchLabel(k)`:
  `branch<k>` by default, and `<region>_branch<j>` for a composite (region
  names default to `region<r>`). A rheology with a single branch under the
  default label keeps the plain name `internal_variable`.

## References

- Al-Attar, D. and Crawford, O. (2016). Particle relabelling
  transformations in elastodynamics. *Geophysical Journal International*,
  205(1), 575–593.
- Crawford, O., Al-Attar, D., Tromp, J. and Mitrovica, J. X. (2017). Forward
  and inverse modelling of post-seismic deformation. *Geophysical Journal
  International*, 208(2), 845–876.
- Simo, J. C. and Hughes, T. J. R. (1998). *Computational Inelasticity*.
  Springer, New York.
