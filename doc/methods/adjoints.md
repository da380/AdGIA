# Derivatives of the quasi-static problem: adjoints, tangents and second order

Method notes for the derivative machinery: the continuous first-order
adjoint (`adjoint.hpp`), the tangent-linear problem (`tangent.hpp`), the
continuous second-order adjoint (`second_order.hpp`) and the discrete
first- and second-order adjoints (`discrete_adjoint.hpp`,
`discrete_second_order.hpp`). The first-order theory is that of Yu,
Al-Attar, Syvret & Lloyd (2025, §3–4; "the paper" below); the forward
scheme that every sweep differentiates or transposes is the exponential
trapezoid of `doc/methods/viscoelasticity.md`. This note is the account of what
the code does and what its gates measure.

## 1. What is differentiated

**The forward scheme.** The forward run is a `ViscoelasticOperator` on a
`LinearQuasiStaticProblem`, stepped by `ExponentialTrapezoidStep` from the
unrelaxed start. With h_k = dt/τ_k, the weights e_k = e^{−h_k},
a_k = (1 − e^{−h_k})/h_k − e^{−h_k}, b_k = 1 − (1 − e^{−h_k})/h_k, and
dⁿ = D uⁿ the strain at the internal nodes (D = P B the Galerkin map,
P = G⁻¹ ⊗ M⁻¹ symmetric; `doc/methods/viscoelasticity.md` §4), a step and the start
are, in effective-solve form,

```
(S1)   K_eff uⁿ⁺¹ = fⁿ⁺¹ + Σ_k B_kᵀ C_k (e_k m_kⁿ + a_k dⁿ),   K_eff = K(C_∞ + Σ_k β_k C_k),  β_k = 1 − b_k
(S2)   m_kⁿ⁺¹ = e_k m_kⁿ + a_k dⁿ + b_k dⁿ⁺¹
start: K_U u⁰ = f⁰ + Σ_k B_kᵀ C_k m_k⁰                     (m_k⁰ = 0)
```

The β_k are interpolated nodal coefficients inside the assembled stiffness,
so (S1) is the step the forward takes. It equals the unrelaxed form
K_U u = f + Σ_k B_kᵀ C_k m_k only where the internal space resolves the
strain exactly (tensor-product elements, affine simplices), not on curved
tetrahedra (§3). A static problem is one solve with no internal variables.
`ForwardTrajectory` records the state after every step (§2). The
quasi-static drivers verify a linear rheology; the tangent and discrete ones
also isotropic (trace-free) branches, the discrete ones the Galerkin strain
map and whole-mesh branches.

**The observation functional** is linear in the response and given as an
`AdjointForce` in forward time. *Point terms* sit at times on the forward
grid and carry any subset of the slots of `AdjointLoad`: a dual on the
displacement rows (`AddForce` layout), one on the potential rows
(`PotentialForceSlot`), and data on the border rows of the uniform potential
term and the angular velocity (`BorderForceSlot`); a sea-level observation
J = ∫ h SL₁ dS fills all four (u-dual −(1/g) h ∇Φ₀, φ-dual −(1/g) h, uniform
(1/g) ∫ h dS, ω_k −(1/g) ∫ h ψ_k dS). *Continuous terms*
J = ∫ ⟨f(t), u(t)⟩ dt act on the displacement or potential rows
(`AddContinuous`, `AddContinuousPotential`); the forward run samples them
with its grid's trapezoid weights, and that sampled sum is what the tangent
and the discrete adjoints differentiate exactly.

**The parameters.** The isotropic generalised Maxwell body
(`doc/methods/viscoelasticity.md` §1) is parameterised as (κ, μ_∞, {μ_k, η_k}),
τ_k = η_k/μ_k — the library's (κ, μ, η) convention, not the paper's, which
holds μ fixed when it perturbs η = μτ. κ and μ_∞ enter the stiffness alone
(on an elastic rheology the shear modulus is μ_∞); a branch modulus μ_k is
perturbed **at fixed η_k**, so it enters the stiffness, the branch force and,
through τ_k, the evolution; η_k enters the evolution alone.
`TangentDirection` carries absolute perturbations (coefficients, null for
zero) in this convention. The **load** is not in it: the problem is linear in
its load, so a load direction is the tangent twin's own load, and the
adjoint's derivative with respect to a data surface load on the mixed
class is `SeaLevelLoadKernel` (§2) in the continuous flavour and
`DiscreteAdjointSolver::DirectionalLoad` with the load duals in the
discrete one (§5). **Not parameters**: the density (its kernel is not
built), the parameters of
state-dependent relaxation laws (nonlinear rheologies are refused),
anisotropic moduli, and a migrating ocean function, whose history the
first-order adjoint can play back but whose δC terms are not
differentiated.

**Twin problems.** Every driver solves on twins: further instances of the
forward's class sharing its spaces and rheology, with zero load, loaded by
the adjoint force, by a Born force or (a load direction) by their own load;
the second-order and discrete drivers take further twins for derivative
solves and source builders, each engaging its effective modulus
independently. A twin carries the forward's potential block and water and
rotation borders with zero data: these systems are linear and symmetric, so
the same feedbacks are their own adjoint and tangent.

## 2. The continuous first-order adjoint (`adjoint.hpp`)

`AdjointSolver` is the adjoint problem of the paper (§3.3): the forward
problem in reversed time with the force slots opened. Nothing is transposed: the elastostatic operator is self-adjoint,
the bordered blocks are symmetric, and the adjoint evolution
τ_k ṁ†_k + m†_k = d† is the forward one from a zero state, so it is a
`ViscoelasticOperator` on the zero-load twin. The driver owns the time
reversal (t = t₁ − s). **Static configuration**
(`AdjointSolver(adjoint_problem, forward_displacement)`): one solve of the
twin per point term, paired with the forward displacement with weight one;
continuous terms are refused. **Quasi-static configuration**
(`AdjointSolver(adjoint_op, forward)`), walking the forward grid backwards:

1. nodes before the first observation (in reversed time) are skipped unless
   continuous terms drive the adjoint from the start;
2. between nodes m† is stepped by `ExponentialTrapezoidSolver`; continuous
   terms enter every elastic solve through
   `ViscoelasticOperator::SetForceHook`, at the forward time;
3. at an observation the driver solves the **impulsive problem** — the slot
   data on the unrelaxed operator alone, no branch forces and no continuous
   loads (they carry no δ part): the generalised fingerprint problem, paper
   eqs. 75–78 — passes the impulsive pairing on with weight one, **jumps** the
   adjoint state branch by branch, [m†_k]⁺₋ = (1/τ_k) D u†_imp at the
   branch's nodes (paper eq. 80), invalidates the displacement cache and
   re-solves the post-jump field;
4. every node's (forward, adjoint) pair goes to the accumulators with the
   trapezoid weight of the forward grid; at an observation the weight splits
   across the jump, the forward-right half pairing the arrived state and the
   forward-left half the post-jump one.

Kernels accumulate *during* the sweep; the adjoint trajectory is never
stored. `SetTimeSync` registers a callback, in forward time, run before
every elastic solve of the stepping and every impulsive solve: the C†
playback of a moving ocean function (`RefreshWaterLoad()` on a stored or
prescribed C(t)), making the adjoint the linear problem with the forward's
time-dependent coefficients played back in reverse.

**The trajectory store.** The state is m alone (the displacement is slaved
to (m, load(t)) through one solve), so `ForwardTrajectory` records m and t
per step under two policies: store-all (`keep_fields = true`) and re-solve
(`Displacement(n)` by one forward solve, valid until the next call), equal
to 1.1e-11. Revolve schedules and compression are not built.

**Kernel densities** (`IsotropicKernelDensity`), per pairing and branch k:

```
Kappa:        − div u div u†
MuLongTerm:   − 2 dev ε(u) : dev ε(u†)
Mu  (k):      − 2 [ dev ε(u) : dev ε(u†) − m_k : dev ε(u†) − (d − m_k) : m†_k ]     (fixed η_k)
Eta (k):      − (2/τ_k) (d − m_k) : m†_k
```

following where each parameter enters σ = κ tr(ε) 1 + 2μ_∞ d +
Σ_k 2μ_k (d − m_k); dev ε is pointwise from the displacement gradients, and
d, m_k, m†_k are trace-free fields of the internal space. Null internal
fields are zero: then Kappa and Mu are the static kernels, with m† null the
impulsive parts; Eta has neither. The kernel of p is the weighted sum of its
density over the sweep plus the impulsive pairings, δJ = ∫ K_p δp dV.

**Outputs.** `IsotropicKernel` accumulates one parameter (one branch, for
Mu and Eta) into three optional outputs: **directional values**
(`AddDirection(δp)`, `Directional(i)` = ∫ K δp dV by the default rule of a
`DomainLFIntegrator` on a scalar space of the mesh, which the header calls
the assembly rule of the elastic forms — the same order on affine
quadrilaterals — summed over ranks); the **dual vector** j_i = ∫ K φ_i on
a scalar space of the same mesh (`SetDualSpace`, optionally restricted to
attributes; `Dual()` local, `TrueDual()` on true dofs), the export of the
pygeoinf contract, never pre-Riesz'd (`riesz.hpp` acts downstream); and a
**display field** (`SetField`, the L2 projection of the density, plot
grade). With accumulators registered the driver refuses region-restricted
branches.

**The sea-level kernel.** `SeaLevelLoadKernel` accumulates the derivative
with respect to the data surface load on the mixed class,
δJ = +g ∫∫ δσ(x, t) SL₁†(x, reversed t) dS dt in the library's conventions
(paper eq. 63 has the opposite sign in its own), reading the twin's adjoint
sea level through `SeaLevelOperator`: directional values along time-aware
δσ coefficients and, after `EnableQuelled`, the quelled ice kernels of paper
eq. 84 — D_I0 = −ρ_i g ∫ (C(t) − C₀) SL₁† dt (live only with a moving ocean)
and the İ stack, integrated backwards by `Finalize()` into D_İ(x, t_n), the
impulses kept as exportable atoms. The discrete counterpart (frozen
shoreline, no quelled kernels) is the load dual of §5.

## 3. The tangent-linear problem (`tangent.hpp`)

`TangentLinearSolver` is the forward-mode twin of `AdjointSolver`: the
derivative (δu, δm) of the forward solution along a `TangentDirection`.
**Static**: one solve of the twin with the Born force −δK u of the forward
displacement (`ModuliBornForce`). **Quasi-static**:
`TangentViscoelasticOperator`, a subclass of the forward operator on the
twin with the forward's internal order and strain map, steps with the
forward's own exponential trapezoid along the recorded grid and injects the
sources at every elastic solve through the force hook. Differentiating
(S1)–(S2), with δh_k = dt δ(1/τ_k), δ(1/τ_k) = (1/τ_k)(δμ_k/μ_k − δη_k/η_k)
and e′, a′, b′ the h-derivatives of the weights (e′ = −e; series for small
h, `detail::TrapezoidWeightDerivatives`):

```
effective solve of the step n → n+1:
  Born force      −K(δκ, δμ_∞ + Σ_k [β_k δμ_k + δβ_k μ_k]) uⁿ⁺¹,       δβ_k = −b′_k δh_k
  branch source   Σ_k B_kᵀ [ 2δμ_k (e_k m_kⁿ + a_k dⁿ) + C_k (e′_k m_kⁿ + a′_k dⁿ) δh_k ]
unrelaxed solve (the start; the restart after a jump in §4):
  Born force      −δK_U u,   δμ_U = δμ_∞ + Σ_k δμ_k;     branch source   Σ_k B_kᵀ 2δμ_k m_k
after the step:   δm_kⁿ⁺¹ += s_k,   s_k = δh_k (e′_k m_kⁿ + a′_k dⁿ + b′_k dⁿ⁺¹)
```

The Born forms use the isotropic rheology's own integrators (κ div–div,
2μ dev–dev, `ElasticityIntegrator` with the default rules) and the forward's
own β_k coefficients, so each force is the exact derivative of the assembled
operator.

**Why the effective form.** The two forms agree only where the internal
space resolves the strain, for then Bᵀ β C D = K(βC). Differentiating the
unrelaxed form instead is exact on the quadrilateral square, 3e-5 off on
the curved disc, and 0.5 % off in μ and η on the coarse 3-D ball of curved
tetrahedra, halving with the step count — the b_k-weighted projection
defect. With the weights and their derivative in the Born force the 3-D
rows agree with finite differences; `Tangent.CurvedElements3D` gates the
case.

**`Pair(force)`** evaluates δJ for an `AdjointForce` on the recorded
`TangentTrajectory` (state, displacement and, on a gravitating twin,
potential, uniform term and ω): point terms, border data, and continuous
terms with the forward grid's trapezoid weights — the exact derivative of
the sampled functional, so the tangent's continuous rows are as sharp as its
point rows (the continuous adjoint's carry an O(dt²) floor, §6). The
operator also steps on any `TangentData` (`Start`, `Step`) for the second
order, and exposes its sources (`PrepareStep`, `ComputeSources`,
`StateSource`) for the discrete adjoints.

## 4. The continuous second-order adjoint (`second_order.hpp`)

`SecondOrderAdjointSolver` computes J″(x)v as the derivative of the
first-order kernels along v, in `AdjointSolver`'s static and quasi-static
configurations: it runs the first-order sweep and, beside it, the tangent of
that sweep (statically, the twin solve with the Born force −δK u† of v). The gradient is a sum of pairings K(u, u†); its
derivative is K(δ_v u, u†) + K(u, δ_v u†) plus the densities' explicit
parameter dependence, with δ_v u v's tangent trajectory
(`TangentLinearSolver`, run first) and δ_v u† the derivative of the sweep:

- between observations a `TangentViscoelasticOperator` on a further
  zero-load twin steps the derivative state with the adjoint's own reversed
  trajectory as data, served a window of nodes at a time (`AdjointWindow`):
  the adjoint evolution is the forward operator on the zero-load twin, so
  its tangent is the same tangent step with the adjoint fields as data;
- at an observation the derivative of the impulsive solve is
  K_U δu†_imp = −δK_U u†_imp, and the derivative state jumps by
  (1/τ_k) D δu†_imp + δ(1/τ_k) D u†_imp;
- the observation duals carry no model dependence.

The impulsive pairings and trapezoid weights are the first-order sweep's.
The only explicit dependence is the Eta density's 1/τ_k: its derivative,
(δ(1/τ_k)/(1/τ_k)) times the first-order pairing, goes to the Eta
accumulators alone through `AdjointPair::scale`. Accumulators added with
`AddKernel` return J″v in the kernels' own form (⟨J″v, δ_i⟩, duals,
fields); those added with `AddFirstOrderKernel` receive the first-order
pairings, the gradient as a by-product. With continuous forcing the
derivative state starts at the first node.

**The continuous flavour.** This is the exact derivative of the first-order
kernels *as computed*; its gate is central differences of those kernels
along v. The kernels are O(dt²) from the exact discrete gradient (§6), so
this Hessian is O(dt²) from the discrete Hessian and symmetric,
⟨J″v, w⟩ = ⟨J″w, v⟩, only to that order. The model is sl3dpp's continuous
second-order sweep (its `docs/dev/adjoints.md` §9). Restrictions: linear
isotropic rheologies, whole-mesh branches, no time sync (no ocean-function
playback), and no derivative of `SeaLevelLoadKernel`.

## 5. The discrete adjoints (`discrete_adjoint.hpp`, `discrete_second_order.hpp`)

The discrete adjoint is the exact transpose of (S1)–(S2) — of the tangent,
their exact derivative — and gives the exact gradient of the discrete
objective J = Σ_n ⟨g_n, uⁿ⟩, g_n the point duals at node n plus the
continuous terms times node n's trapezoid weight. With λⁿ the dual of uⁿ and
μ_kⁿ that of m_kⁿ, `DiscreteAdjointOperator::Run` performs on the zero-load
twin, in reversed order,

```
μ_kᴺ = 0,   K_eff^(N−1→N) λᴺ = g_N
μ_kⁿ = e_k (μ_kⁿ⁺¹ + C_k B_k λⁿ⁺¹)                                                       (A2)
K_eff^(n−1→n) λⁿ = g_n + Bᵀ P Σ_k [ a_k^(n→n+1) (C_k B_k λⁿ⁺¹ + μ_kⁿ⁺¹) + b_k^(n−1→n) μ_kⁿ ]   (A1)
n = 0:   K_U λ⁰ = g_0 + Bᵀ P Σ_k a_k (C_k B_k λ¹ + μ_k¹)
```

(Dᵀ = Bᵀ P since P is symmetric). Each reversed step is one solve with the
step's own symmetric operator; the potential and border parts of g_n go to
the twin's slots. There are no impulsive solves and no jumps — the
continuous adjoint's rendering of the same observations — and the duals are
kept per node. `DiscreteAdjointSolver::Directional(δ)` pairs them with the
tangent's sources along δ, built on a second zero-load twin:

```
δJ = Σ_n [ ⟨λⁿ⁺¹, F_effⁿ⁺¹(δ)⟩ + Σ_k ⟨B_k λⁿ⁺¹, ζ_kⁿ(δ)⟩ + Σ_k ⟨μ_kⁿ⁺¹, s_kⁿ(δ)⟩ ]
     + ⟨λ⁰, F_unrel⁰(δ)⟩ + Σ_k ⟨B_k λ⁰, 2δμ_k m_k⁰⟩
```

with F_eff the effective Born force, ζ_k the known-part branch source and
s_k the state source of §3; the identity ⟨g, δu⟩_tangent = δJ_discrete
holds to solver tolerance, with no finite differences and no time-quadrature
floor. The **static configuration** (`DiscreteAdjointSolver(twin,
forward_displacement)`) solves the twin once per point observation and pairs
each dual with −δK u (`ModuliBornForce`); it builds no tangent operator, so
branchless rheologies are served. In parallel the displacement pairings are
taken on true dofs and the internal-space pairings summed once over ranks.

**Kernels as dual vectors** (quasi-static configuration). `AddKernel` and
`AssembleKernels()` give the dual of J with respect to a parameter's nodal
values on the internal scalar space (L2 nodal, its dofs true dofs);
`Dual(i)` returns it, `Pair(i, δ)` pairs it with a nodal direction. The
assembly mirrors the sources: the elastic densities (−div u div λ;
−2 dev ε(u) : dev ε(λ)), with the step's β_k on a branch modulus and, for the
weights' dependence on τ_k, μ_k times the shear density with the nodal
factor −b′ dt ∂(1/τ_k)/∂p, integrated against the L2 basis; plus the
pointwise terms of the branch sources against B λ and of the state source
against μ. The integration uses the forward stiffness's own rule
(2·`OrderGrad` of the displacement element), so that the dual paired with a
nodal direction is the derivative of the assembled stiffness to rounding; a
higher rule leaves a 9e-5 mismatch on κ. The rule is taken from the first
element: a mesh of one element type is assumed.

**The data surface load** (frozen shoreline; both configurations). The
problem is linear in its load, so the derivative along a load direction
δσ(x, t) is the sum over the solves of the dual state paired with the
full assembled load of δσ at the solve's time — the transpose of the
tangent's load direction, whose twin carries δσ as its water-load data.
On the mixed class the load touches four rows, each paired with its
dual:

```
δJ = Σ_n [ ⟨λⁿ, f_u(δσ(t_n))⟩ + ⟨φ†ⁿ, ℓ_φ(δσ(t_n))⟩ + Φ_g†ⁿ ∫ δσ(t_n) dS + Σ_k ω†ⁿ_k r_k(δσ(t_n)) ]
f_u(σ) = −∫ σ ∇Φ₀·v dS,   ℓ_φ(σ) = −∫ σ χ dS,   r_k(σ) = −∫ σ Πψ_k dS
```

with Πψ_k the potential space's interpolant of the unit centrifugal
potential. φ†ⁿ, Φ_g†ⁿ and ω†ⁿ are the twin's `Potential()`,
`UniformPotentialTerm()` and `AngularVelocity()` after the dual solve of
node n (the bordered system is symmetric); `DiscreteAdjointOperator::Run`
keeps them per node beside λⁿ, the static configuration per
observation. The sum runs over the unrelaxed start and the effective
solve (S1) of every step, the structure of the Born-force pairing; in
the static configuration over the point observations, with the load
assembled at t = 0 as the static tangent assembles it.
`DirectionalLoad(load_twin)` calls the load twin's `AssembleForce(t_n)`
and reads the rows off it (`ExternalLoad()`, `PotentialLoad()`,
`UniformRowLoad()`, `AngularVelocityRowLoad()`); on a class without a
potential only the displacement rows enter. A tidal potential on the load
twin and the referential class are refused; Dirichlet data are not read.

The **load duals** (`AssembleLoadDuals(marker)`, `LoadDual(n)`) are the
transpose of that assembly applied to the dual state, on the body's scalar
trace space (`LoadDualSpace()`, the class's `PotentialSpaceOnBody()`, on
which it assembles its surface-load forms):

```
jⁿ_j = ∫ ψ_j g SL₁(dualⁿ) dS,    g SL₁(dual) = −(λ·∇Φ₀ + φ† + Πψ(ω†) − Φ_g†)
```

the discrete counterpart of `SeaLevelLoadKernel`'s +g SL₁†, in the same
sign convention. Each term is integrated with the rule of the row it
transposes (2 p_u for the displacement rows' `VectorBoundaryLFIntegrator`,
the default `BoundaryLFIntegrator` rule of the trace space for the
potential, mass and ω rows), so Σ_n ⟨jⁿ, δσ(t_n)⟩ (`PairLoad`) equals
`DirectionalLoad` to rounding for a load read as a
`GridFunctionCoefficient` of trace-space fields. No quadrature weights
enter: the forward samples the load at the nodes, and
`IntegratedLoadDual()` = Σ_n jⁿ is the dual of a load constant in time.
The migrating ocean (the δC terms) and the quelled ice kernels of
`SeaLevelLoadKernel::EnableQuelled` are not built in the discrete flavour.

**The discrete second order.** `DiscreteSecondOrderSolver` is the derivative
of the discrete gradient along v, the exact Hessian-vector product. The
derivative of the transposed recursion is the same recursion with sources
from its own v-dependence applied to the first-order duals: −δ_v K_eff λⁿ
(the tangent's effective Born form of v applied to the duals, served as
`TangentData` by `DualData`) and the derivatives of the weights and branch
moduli. `Directional(δ)` returns H[v, δ]: the derivative duals against δ's
sources on the forward trajectory, the first-order duals against δ's sources
on v's tangent trajectory (`TangentTrajectoryData`), and the parameter cross
terms (the scheme's second derivative in the model). The static
configuration solves K δλ_i = −K(v) λ_i and returns
H[v, δ] = Σ_i ⟨δλ_i, −K(δ) u⟩ + ⟨λ_i, −K(δ) δ_v u⟩. The equations of the
derivative recursion and of the cross terms are those of the discrete
first order differentiated along v.

**The Hessian-vector product as a dual vector** (quasi-static
configuration). `DiscreteSecondOrderSolver::AddKernel`,
`AssembleKernels()`, `Dual(i)` and `Pair(i, δ)` give the dual of
H[v, ·] with respect to a parameter's nodal values p_q on the internal
scalar space, as the first-order kernels give that of the gradient (κ,
μ_∞, μ_k at fixed η_k, η_k). Every term of H[v, δ] is linear in δ. The two
source pairings are the first-order assembly above twice: with the
derivative duals (δλ, δμ) in place of (λ, μ) on the forward trajectory,
and with v's tangent trajectory (δ_v u, δ_v m, D δ_v u) in place of the
forward one against (λ, μ). The cross terms (branch parameters only) are
the Born cross form, with coefficient
δ_vβ δμ_δ + δ_δβ δ_vμ + δ²β μ_k, δ²β = −b″ δ_vh δ_δh − b′ dt δ²(1/τ)[v, δ],
and the derivatives along v of the known-part branch source and of the
state source; they depend on δ through δμ_δ = χ p, δ_δ(1/τ) = c p and
δ²(1/τ)[v, δ] = g p (1/τ = μ/η, whose second derivatives are 0 in μμ,
−1/η² in μη and 2μ/η³ in ηη), so the nodal factors and the per-step cross
part of the dual are

```
             χ = ∂μ/∂p     c = ∂(1/τ)/∂p     g = ∂²(1/τ)/∂p ∂v
μ_k (η_k fixed):   1         (1/τ)/μ           −δ_vη/η²
η_k:               0        −(1/τ)/η           (2μ δ_vη/η − δ_vμ)/η²

j_q += χ ∫ φ_q δ_vβ ρ₁ − b′ dt c ∫ φ_q δ_vμ ρ₁ + (−b″ dt c δ_vh − b′ dt g) ∫ φ_q μ_k ρ₁
     + Σ_c (B λⁿ⁺¹)_c [2χ δ_vh r₁ + 2 δ_vμ dt c r₁ + 2μ dt (g r₁ + c δ_vh r₂)]_c
     + Σ_c (μ_kⁿ⁺¹)_c dt [g (r₁ + b′ dⁿ⁺¹) + c δ_vh (r₂ + b″ dⁿ⁺¹)]_c
```

at node q, with r₁ = e′ m_kⁿ + a′ dⁿ, r₂ = e″ m_kⁿ + a″ dⁿ,
ρ₁ = −2 dev ε(uⁿ⁺¹) : dev ε(λⁿ⁺¹), v's nodal data δ_vμ, δ_vh = dt δ_v(1/τ),
δ_vβ = −b′ δ_vh, δ_vη = η (δ_vμ/μ − τ δ_v(1/τ)), μ, η, τ the nodal branch
values, and the integrals taken against the L2 basis φ_q with the forward
stiffness's rule. The header `discrete_second_order.hpp` carries the
whole dual, source pairings included. Paired with a nodal direction it
equals `Directional` along that direction read as a
`GridFunctionCoefficient` to rounding.

**Limits.** The parameter kernel duals, of the gradient and of H[v, ·],
are quasi-static only (the load duals serve both configurations). Both
discrete drivers require the Galerkin strain map,
whole-mesh branches and linear isotropic rheologies; neither plays back a
migrating shoreline.

## 5a. Measurements (`measurement.hpp`)

A `Measurement` is a linear observable of the state (u, φ, Φ_g, ω) that
yields both its value and its dual: `Assemble` builds the dual once on a
problem's spaces as a `MeasurementDual` (owned slot data in the
`AdjointLoad` layouts), and the value is the pairing of that dual with
the state (`Evaluate`), the same pairing the tangent uses in `Pair`, so
J(u) and ⟨dJ, δu⟩ are consistent to rounding. Built in:
`SurfaceDisplacementMeasurement` (∫ w·u dS: stations, caps, components,
with `CapWindowCoefficient` and `RadialVectorCoefficient` for the usual
weights), `SurfacePotentialMeasurement` (∫ w φ dS on the potential rows),
`HarmonicCoefficientMeasurement` (one surface-harmonic coefficient of the
radial displacement or of the potential, the dual being
`BoundaryHarmonicCoefficients::LoadVector` of the unit coefficient scaled
by R^{1−d}; the analysis `Coefficients` of a field on the space is the
exact transpose of `LoadVector`, so the value is the analysis coefficient
c_i of the state to rounding, with R the radius the operator measures on
the discrete boundary, not the nominal one: R − 1 = −2.45·10⁻⁴ on the
coarse order-2 test ball, where the unscaled pairing differs from c_i by
4.9·10⁻⁴),
`RotationMeasurement`
(ω_k, polar motion ω_k/Ω, the relative length-of-day change −ω_3/Ω) and
`SeaLevelMeasurement` (∫ h SL₁ dS with all four slots: u = −(h/g)∇Φ₀,
φ = −h/g, uniform ∫ h/g dS, ω_k = −∫ h ψ_k/g dS, with g the body's
discrete |∇Φ₀|; the potential-row integrals are assembled on the body's
surface and carried to the potential space node by node).

A `MeasurementSet` orders observations: point terms at forward times and
time-integrated terms ∫ w(t) J(t) dt (displacement and potential rows
only). `Data` gives the data vector of a forward trajectory (one elastic
re-solve per node an observation needs; integrated terms by the grid's
trapezoid rule, which is how the tangent and the discrete adjoints sample
them) or of a static state; `Force(c)` is the `AdjointForce` of Σ c_i J_i
(point terms at one time merged into one observation, integrated terms as
continuous fillers), so a misfit's residual, a unit vector for one kernel
or any weighting goes straight to every adjoint driver.

## 6. Choosing between the continuous and the discrete flavours

The continuous flavour is the default and the discrete one the user's
option (the parity sl3dpp has). The choice is the
driver constructed
— `AdjointSolver` or `DiscreteAdjointSolver`, `SecondOrderAdjointSolver` or
`DiscreteSecondOrderSolver` — with the same `AdjointForce`, twins and
`TangentDirection`s.

| | continuous | discrete |
|---|---|---|
| what it is | the paper's adjoint equations, discretised: reversed stepping, impulsive solves and jumps | the transpose of (S1)–(S2) |
| gradient | O(dt²) from the discrete one (1e-3 on the η row of a continuous functional, 1e-6 elsewhere, in `DiscreteAdjoint.PairsWithTangent2D`) | exact to solver tolerance (gated at 1e-8) |
| Hessian | exact derivative of the continuous kernels; symmetric to O(dt²) | symmetric to rounding (gated at 1e-9) |
| solves per sweep | one per reversed step, plus an impulsive solve and a re-solve per observation | one per reversed step |
| storage | adjoint consumed as produced; kernels accumulated in the sweep | duals kept per node; kernels assembled after |
| kernels | directional values, duals on any scalar space, fields; the sea-level kernel | duals on the internal scalar space (gradient and H[v, ·]); the data-load duals on the body's trace space (frozen shoreline); directional values |
| playback (time sync) | first order (`SetTimeSync`) | none |

In the static case the two coincide: the same solve paired through the same
assembled form (1.5e-14 and 4.9e-14 apart in `DiscreteAdjoint.Static2D`).
In time the continuous flavour matches the published derivation, its
distance from the discrete gradient a time-discretisation error converging
as dt² (9.8e-4 at n = 32, 2.4e-4 at n = 64 on the η row of a continuous
functional); the discrete gives exact identities and symmetric Hessians. The
cost per sweep is the same up to the observation-time solves.

## 7. Support matrix

"Built, ungated": the code path exists and is not refused, but no test
exercises it.

| | continuous adjoint | tangent | continuous second order | discrete adjoint | discrete second order |
|---|---|---|---|---|---|
| static | gated by `Adjoint.StaticKernels2D`, `Adjoint.StaticKernels3D` | gated by `Tangent.Static2D` | gated by `SecondOrder.Static2D` | gated by `DiscreteAdjoint.Static2D` | gated by `DiscreteSecondOrder.Static2D` |
| viscoelastic | gated by `Adjoint.ViscoelasticKernels` | gated by `Tangent.Viscoelastic2D` | gated by `SecondOrder.Viscoelastic2D` | gated by `DiscreteAdjoint.PairsWithTangent2D`, `DiscreteAdjoint.KernelDualsPairExactly2D`; a traction load direction by `DiscreteAdjoint.TractionLoad2D` | gated by `DiscreteSecondOrder.SymmetricAndMatchesFD2D`, `DiscreteSecondOrder.KernelDualsPairExactly2D` |
| gravitating mixed class | gated by `Adjoint.GravitatingReciprocity`, `Adjoint.GravitatingViscoelasticKernels` | gated by `Tangent.GravitatingStatic2D`, `Tangent.GravitatingViscoelastic2D`, `Tangent.CurvedElements3D` | built, ungated | gated by `DiscreteAdjoint.GravitatingStatic2D`, `DiscreteAdjoint.WaterLoadFrozen3D` | built, ungated |
| water load, frozen shoreline | gated by `AdjointFingerprint.FullPhysicsKernels`; played-back moving ocean: `AdjointFingerprint.PlaybackKernels`, `AdjointFingerprint.IceKernels` | gated by `Tangent.WaterLoadFrozen3D` | built, ungated (no sea-level kernel derivative) | gated by `DiscreteAdjoint.WaterLoadFrozen3D`; the data-load derivative and its duals by `DiscreteAdjoint.WaterLoadKernel3D`, `DiscreteAdjoint.WaterLoadKernelStatic3D` | gated by `DiscreteSecondOrder.WaterLoadFrozen3D` |
| rotation | gated, with the water load, by `AdjointFingerprint.FullPhysicsKernels` | gated, with the water load, by `Tangent.WaterLoadFrozen3D` | built, ungated | gated, with the water load, by `DiscreteAdjoint.WaterLoadFrozen3D` | gated, with the water load, by `DiscreteSecondOrder.WaterLoadFrozen3D` |
| Prony multi-branch | gated by `Adjoint.PronyKernels` | gated by `Tangent.Prony2D` | built, ungated | built, ungated | built, ungated |
| parallel | gated by `TestAdjointPar` | gated by `TestTangentPar` | built, ungated | gated by `TestDiscreteAdjointPar` (the identity, the data-load derivative and its duals; the parameter kernel duals ungated) | gated by `TestDiscreteSecondOrderPar` (the identity against FD of the parallel discrete gradient; symmetry; kernel duals ungated) |

## 8. Verification

Gates are finite differences (FD) of fresh forward runs, identities between
flavours, and reciprocity. Unless stated, the configuration is the free unit
square (12 × 12 quadrilaterals, order 2, rigid modes projected) under a
zero-mean normal traction stepped on at t = 0, observed by a windowed
two-station baseline; Maxwell with τ = 1 to t = 2; "blob" and "all" are a
localised and a uniform perturbation. Grades are relative, quoted from the
test comments; "gated" gives a threshold where no measurement is recorded.

`TestAdjoint`:

- `StaticKernels2D`: κ, μ against FD, 2.6e-7 (blob), 1.1e-6 (all); the dual
  against the directional value, 6e-4 (interpolation grade).
- `StaticKernels3D`: coarse hexahedra at order 1: 4.1e-3 (blob), 1.1e-6
  (all); the dual check scale-only.
- `ViscoelasticKernels`: κ, μ, η, 32 steps, final-time, interior and
  continuous observations: point rows ≤ 3.3e-6; continuous η rows at the
  O(dt²) floor, 9.8e-4 (n = 32), 2.4e-4 (n = 64); store-all = re-solve to
  1.1e-11.
- `GravitatingReciprocity`: the two-layer disc of the mixed class, u–u and
  u–φ reciprocity, the `PotentialForceSlot` sign certificate (gated 2e-9).
- `GravitatingViscoelasticKernels`: the disc, 24 steps, displacement and
  potential observations: ≤ 1.6e-4, μ blob 1.3e-3 (curved-element floor).
- `PronyKernels`: two branches and μ_∞, 64 steps: κ 1.5e-6, μ_∞ 1.3e-4,
  μ₁ 6e-5, η₁ 1.5e-4, η₂ 7e-4, μ₂ 5.5e-3, the O(dt²) floor ∝ (dt/τ_k)².

`TestAdjointFingerprint` (the canned 3-D ball, water load):

- `SphericalReciprocity`: ∫ SL₁ ζ₂ dS = ∫ SL₂ ζ₁ dS (Al-Attar et al. 2024),
  2.5e-5 at order 1, 7.6e-6 at order 2.
- `LateralStructureReciprocity`: laterally varying κ, μ, 3.5e-5.
- `RotationWithStructureReciprocity`: rotation and 3-D structure, 1.9e-5.
- `FullPhysicsKernels`: Maxwell, water and rotation, frozen shoreline, 12
  steps: η rows 8e-5 (u), 9e-7 (φ: `AddPotentialForce` loads only the
  potential block), 1.6e-5 (four-slot sea level, the `BorderForceSlot`
  certificate); blob rows at the ball's floor (η 2.7e-2/6.4e-2, μ 0.15).
- `PlaybackKernels`: a prescribed moving ocean played back: η 0.9617
  (blob), 0.99951 (all), μ blob 1.152; the frozen-C control eleven times
  further off (0.9944).
- `IceKernels`: ice-rate and initial-thickness directions 1.2–1.9e-2 of FD;
  the D_I0 field 7e-2, the İ stack 3.3e-4 against the directional route.

`TestTangent`:

- `Static2D`: κ, μ against FD (1e-6) and the adjoint (gated 1e-8); a load
  direction reproduces J (gated 1e-10).
- `Viscoelastic2D`: κ, μ, η, 32 steps, three observation types, all within
  2e-5 of FD; against the adjoint to solver tolerance on the point
  functionals (gated 1e-6), and 3e-6 (κ, μ) and 1e-3 (η) on the continuous
  one.
- `GravitatingStatic2D`: κ, μ on the disc against FD (gated 1e-4).
- `Prony2D`: every parameter within 2e-5 of FD; the adjoint within its
  O(dt²) floor (up to 5.5e-3).
- `GravitatingViscoelastic2D`: the disc, 24 steps, u and φ observations,
  every row within 3e-5 of FD.
- `WaterLoadFrozen3D`: water and rotation, 8 steps; η blob for u and the
  four-slot sea level (gated 1e-4); an ice-load direction (gated 1e-7).
- `CurvedElements3D`: the ball of curved tetrahedra, 8 steps, κ, μ, η
  (gated 1e-5); the case of §3.

`TestTangentPar` (1, 2, 4 ranks): static κ/μ (1e-5) and viscoelastic κ/μ/η,
final-time and continuous (2e-5), against FD.

`TestSecondOrder` (against FD of the first-order kernels along v):

- `Static2D`: v = μ blob, κ all; κ blob and μ all kernels (gated 1e-5, the
  note records 1e-6); by-product gradient = `AdjointSolver`'s (gated 1e-10).
- `Viscoelastic2D`: 32 steps, final-time and continuous, v = η blob and μ
  blob: every row within 1e-6 of FD on the Hessian's scale; the small μ-η
  cross term (1.2e-8 against 7e-6) at 4e-4 relative. The header mentions a
  symmetry defect; the body does not check it.

`TestDiscreteAdjoint`:

- `PairsWithTangent2D`: 32 steps, κ blob, μ all, η blob, three observation
  types: the identity to 1e-8; the continuous adjoint within 5e-3 (its η
  row on the continuous functional at 1e-3).
- `Static2D`: against the tangent 4.5e-10 (κ blob, round-off floor) and
  3.6e-12; against the static continuous adjoint 1.5e-14 and 4.9e-14.
- `GravitatingStatic2D`: the disc, κ and μ blobs (gated 1e-8).
- `WaterLoadFrozen3D`: water and rotation, η blob, 8 steps, u and four-slot
  sea level: 1.0e-9 and 4.1e-9 at solver tolerance 1e-12.
- `WaterLoadKernel3D`: the data surface load on the same ball, 8 steps:
  `DirectionalLoad` against the tangent's load direction for a melt cap,
  the cap times a ramp in t (0.5 at the start) and nodal caps on the trace
  space (constant and ramped), for the displacement and the four-slot sea
  level: 1.6e-10 and 5.3e-12 at solver tolerance 1e-12 (gated 1e-8); every
  row carries weight (the uniform-term and ω rows 1–6 % of the total). The
  load duals against `DirectionalLoad`, ramped and integrated: 6.7e-16
  (gated 1e-9).
- `WaterLoadKernelStatic3D`: the static configuration, the displacement and
  sea-level observations at t = 0 (two duals): 2.6e-12 against the static
  tangent; the load duals 1.3e-15.
- `TractionLoad2D`: a ramped traction direction on the unit square, 16
  steps, final-time and continuous observations: `DirectionalLoad` against
  the tangent's load direction 6.7e-12 and 1.3e-11 (gated 1e-8), the
  displacement rows alone.
- `KernelDualsPairExactly2D`: 16 steps; the κ, μ, η duals paired with a
  nodal blob equal `Directional()` to 1e-9.

`TestDiscreteAdjointPar` (1, 2, 4 ranks): the static and viscoelastic
identities to 1e-8; the static pairing as local sums to 1e-12; the
data-load leg on the water-loaded ball (4 steps, solver tolerance 1e-13,
which the identity follows: 1.0e-8 at 1e-12 on one rank): 6.9e-10
(displacement), 3.0e-11 (sea level) against the tangent's load direction,
the load duals to 6.7e-16.

`TestDiscreteSecondOrder` (against FD of the discrete gradient):

- `SymmetricAndMatchesFD2D`: 16 steps, v, w, δ among κ, μ, η blobs,
  final-time and continuous: symmetric to 1e-9, within 1e-5 of FD on the
  block's scale; every entry agrees to six digits.
- `KernelDualsPairExactly2D`: 16 steps, final-time and continuous,
  v among κ, μ, η blobs: the κ, μ_∞, μ, η duals of H[v, ·] paired with a
  nodal blob equal `Directional()` to 3.5e-13 at most (gated 1e-9); the
  μ and η rows along v = μ, η, where the cross terms enter, to 2.3e-14,
  and off by 8e-2 to 12 without the nodal cross terms.
- `Static2D`: κ and μ blobs: symmetric to 1e-9, FD to 1e-5 (gated).
- `WaterLoadFrozen3D`: the ball with water and rotation, 8 steps, η and μ
  blobs, the displacement and the four-slot sea-level functionals: every
  entry of the 2 × 2 blocks agrees with FD to six digits (gated 1e-5) and
  the blocks are symmetric (gated 1e-7); solver tolerance 1e-12.

`TestDiscreteSecondOrderPar` (1, 2, 4 ranks): the 2-D evolution case
against FD of the parallel discrete gradient (1e-5) and symmetry (1e-9).

`TestMeasurement`:

- `WaterLoadedBall3D`: the ball with water and rotation, 8 steps. The
  radial-cap and harmonic duals equal the hand-built ones to 1e-13; the
  sea-level four slots agree with the hand-built ones (which take g = g₀)
  to the background's discretisation error, 0.7–1.0e-2 on this coarse
  mesh; the identity ⟨dJ_i, δu⟩_tangent = FD of J_i along an η blob holds
  to 1e-7 for the radial cap, the sea-level cap, polar motion, the length
  of day and the (2,0) potential and (2,1) radial-u coefficients (gated
  1e-4), and for the merged set.
- `TimeIntegrated2D`: the unit-square Maxwell body, a point station and a
  time-integrated station: the data equal the hand-computed pairing and
  trapezoid sum (solver tolerance); tangent = FD (1e-5) and the discrete
  adjoint = tangent (1e-8) for both, through `Force`.

Also: `TestAdjointLove` (the η kernel on the gravitating Maxwell sphere
against the Love-number closed form, 1.0037 at order 3; a radial direction
against FD, 1.1e-3), `TestAdjointPar` (the static, viscoelastic,
gravitating and Prony legs at 1, 2, 4 ranks), `TestAdjointFingerprintPar`.

## 9. What is not covered

Nonlinear rheologies (the quasi-static drivers refuse them); anisotropic
branches and moduli; the density and interface (lithospheric-thickness)
kernels (not built); the δC terms of migrating shorelines, and
playback beyond the first-order continuous adjoint; the quelled ice
kernels in the discrete flavour (its data-load derivative is the
frozen-shoreline one); region-restricted
branches in the kernels, the second order and the discrete adjoints;
checkpointing beyond store-all and re-solve; the referential classes (the
sea-level accumulator and the discrete load duals read the mixed class,
and the referential potential-row slot is open); the ice kernels' Hessian and a symmetry gate of
the continuous second order; and forward schemes other than the exponential
trapezoid, which the tangent and discrete adjoints differentiate and
transpose (the gates use fixed steps).

## References

- Al-Attar, D., Syvret, F., Crawford, O., Mitrovica, J. X. and Lloyd, A. J.
  (2024). Reciprocity and sensitivity kernels for sea level fingerprints.
  *Geophysical Journal International*, 236, 362–378.
- Yu, Z., Al-Attar, D., Syvret, F. and Lloyd, A. J. (2025). Application of
  first- and second-order adjoint methods to glacial isostatic adjustment
  incorporating rotational feedbacks. *Geophysical Journal International*,
  240, 329–348.
