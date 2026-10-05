# The gauged treatment of fluid regions

Method notes for the gauged-fluid option of `LinearQuasiStaticProblemBase`
(`SetGaugedFluid()`, `quasi_static_problem.hpp`) and its self-gravitating
uses: the mixed class `LinearQuasiStaticMixedSelfGravitatingProblem`
(`mixed_problem.hpp`) and the referential class
`LinearQuasiStaticReferentialSelfGravitatingProblem`
(`referential_problem.hpp`). The formulation follows Maitra & Al-Attar
(2024, §3.7.3): the fluid keeps its displacement in a fully referential
description, and the relabelling gauge is fixed by a small shear penalty
whose bias is removed by iterated Tikhonov refinement. It is the
alternative to Dahlen's treatment of `doc/self_gravitation.md`, which
eliminates the fluid displacement in favour of the potential.

Companion documents: `doc/gauge_penalty_iteration.tex` (the mathematics of
the penalty and the refinement: spectral analysis, failure condition,
semi-convergence, relation to augmented Lagrangians) and
`doc/slip_interface.tex` (the slipping fluid–solid interface, for which the
welded space of this note is not enough).

## 1. Formulation

### The fluid as a solid without shear

An inviscid fluid region `M_F` carries a displacement like the solid, with
the isotropic stiffness `kappa (div u)(div u')` and zero shear; the
gravity, coupling and load terms are the same integrands as in the solid
(the elastic-gravitational form is structure-blind). What distinguishes a
fluid is a symmetry: the solution is defined only up to a **linearised
relabelling** — a rearrangement of fluid particles with no Eulerian effect.
For a materially barotropic fluid (one thermodynamic parameter: the stored
energy depends on position only through the density) these are the fields

```
K = { w on M_F :  div(rho w) = 0,   w.m = 0 on the interfaces },
```

the "trivial displacements" of Friedman & Schutz (1978). A fluid that
carries a second parameter (an entropy or compositional label, i.e.
`N^2 != 0`; in hydrostatic equilibrium its level surfaces are those of the
density, pressure and potential) keeps only the relabellings that also
preserve that parameter: the fields of `K` tangent to the level surfaces,
which are then divergence-free. In either case the relabellings lie in the
kernel of the continuous operator (loads do no work on them), so the fluid
displacement is gauge; the observables are the solid displacement, the
potential, and in the fluid anything built from `div(rho u)` and the
interface normal displacement (e.g. the pressure perturbation).

### Gauge fixing: penalty plus iterated refinement

Discretely the finite element space does not contain the relabellings
exactly: the operator has a large *near*-kernel of "almost relabellings"
with eigenvalues `O(h^p)` instead of zero, and a Krylov method on the raw
system amplifies their `O(h^q)` load components into mesh-dependent fluid
displacements that pollute the solid through the coupling. The remedy:

1. **Penalty.** Add `eps * 2 mu_g dev e(u) : dev e(u')` on the fluid — a
   fluid with shear modulus `eps mu_g`. The solver operates on
   `A + eps Q`, whose near-kernel eigenvalues are bounded below by
   `eps mu_g` times the shear eigenvalue; the observables acquire an
   `O(eps)` bias.
2. **Iterated Tikhonov refinement.** `U_{k+1} = U_k + (A + eps Q)^{-1}
   (f - A U_k)`. After an exact step the physical residual is
   `f - A U = eps Q delta` with `delta` the last increment (for the coupled
   self-gravitating system, `[eps Q delta_u; 0]`), so **no second operator
   is needed**: each refinement solves the regularised system against
   `eps Q` applied to the previous increment. The error contracts by
   `O(eps mu_g / mu_solid)` per step, and the iterates converge to the
   `Q`-minimal solution of the *physical* discrete system — the observables
   become independent of `eps` (to ~1e-4 in the tests).

**Semi-convergence.** The refinement must not be over-driven: each step
also adds `O(h-residual / (eps mu_g))` of the near-kernel content that the
penalty exists to suppress (iterated Tikhonov on a consistent-but-discrete
singular system is semi-convergent). Too small an `eps` or too many
refinements bring it back; `eps ~ 1e-2` with **2–3 refinements** balances
the `O(eps^k)` bias against the `O(k h^q / eps)` pollution, and both
improve with mesh refinement. The window is bounded on both sides in the
Love-number benchmarks (`fluid_core`, and PREM structure in `prem_4`):
`eps = 1e-1` leaves an observable bias at the percent level, while below
`1e-2` the cost grows several-fold per decade of `eps` and by `eps = 1e-4`
the solution is semi-convergently polluted although the solver reports
convergence (the measurements are in `doc/benchmarks.tex`). Every
`GaugeRefine()` carries a contraction-rate tripwire
(`WarnGaugeContraction()`: a warning above 0.9 — the semi-convergence
signature, `eps` too small for the mesh and model; a note above 0.2 —
residual bias beyond the refinement budget). The worst case is a load
aligned with a near-null mode: a *uniform* tidal gradient (a rigid-mode
load) leaks ~3% of a comparable physical response into the solid on the
coarse test mesh at `eps = 1e-2`, k = 3, growing like `k / eps` beyond
(`TestMixedProblemGauged`); physical (degree-2) loads do not sit on the
near-kernel. The mathematics is `doc/gauge_penalty_iteration.tex`.

## 2. Tangential slip and the welded space

The gauged fluid uses **one continuous H1 displacement space** over solid
and fluid together. This welds the fluid–solid interface: the normal
displacement is continuous, as it must be, and so is the tangential one,
which a fluid does not require. Whether the weld costs anything depends on
whether the tangential slip it suppresses could have been removed by a
relabelling.

**When slip is gauge.** In a hydrostatic fluid the linearised energy is
unchanged by displacements that are divergence-free and tangent to the
level surfaces of the equilibrium potential, pressure and density (which
coincide in hydrostatic equilibrium): these are relabelling (gauge)
directions, and for a fluid with a second advected parameter they are all
of them (§1). A tangential slip
on the fluid–solid interface can be removed by relabelling only if it
extends into the fluid as such a field. On an interface that is itself a
level surface this restricts the slip: for a spherical core the rigid
rotations of the fluid are such fields, but a general tangential slip need
not be. On an interface that is not a level surface (a general ellipse)
the level surfaces meet the interface and the construction generally
fails. **Tangential slip is therefore not gauge in general**, and the
welded space is not an admissible gauge condition in general. (For a
materially barotropic fluid the larger relabelling class `K` of §1 is
available; which slips it removes on which geometries is not established
here, and the library does not rely on it.)

**What the library does.** The slipping-interface formulation
(`doc/slip_interface.tex`) is the general one: it allows a tangential jump
and enforces only normal continuity, and allowing a slip that is not needed
costs nothing in correctness. The welded gauged formulation of this note is
used where the slip it suppresses is absent or removable by relabelling —
as for the spherically symmetric models of the benchmarks, where the welded
and slipping formulations agree to discretisation level (`TestSlipProblem`
`TwoLayerBarotropicCrossCheck`; the `gauged` and `slip` columns of the
Love-number benchmark, `doc/benchmarks.tex`). For aspherical fluid regions
the slipping formulation should be used. A two-parameter (stratified)
fluid and the non-linear slip map are further cases outside the welded
formulation.

Two consequences of the welded space are worth noting where it is used.
The tangential traction it transmits is the fluid's deviatoric stress,
`O(eps)`, and dies with the refinement like the rest of the bias. And the
rotations of a spherical core, which the gauge functional leaves
ambiguous (conformal-Killing modes), are absorbed by the interface trace:
an inner-core rotation must drag the fluid at `O(eps)` cost, so no
Tisserand-type constraint is needed — and `AddRegionRotations()` should
*not* be used in this mode (the penalty owns those modes; projecting them
would bias the solution by the `O(eps)` restoring they actually feel).

## 3. What changes against Dahlen's treatment

In `LinearQuasiStaticMixedSelfGravitatingProblem` the gauged mode is selected by
construction: **no FluidRegions** — the fluid attributes join the
displacement SubMesh, the rheology gives them `kappa_f` with zero shear,
the `density` coefficient covers them, and `SetGaugedFluid()` adds the
penalty. Nothing else changes, because every operator of the class
integrates over the whole SubMesh:

| Dahlen (`doc/self_gravitation.md`) | Gauged |
|---|---|
| displacement on the *solid* SubMesh | displacement on the whole body |
| fluid mass term (F1), `A_phiphi = (K+DtN)/4piG + M_F`, `M_F <= 0`, near-indefinite for steep cores | no F1: the potential block is the plain Laplace–DtN operator, SPD with margin |
| interface terms (F2), (F3) with `m.grad Phi0` signs | none: interface conditions are natural |
| coupling `c(phi,v) = int_{M_S} rho grad phi . v - int_Sigma rho_F phi (m.v)` | `c(phi,v) = int_M rho grad phi . v` |
| tidal load on the potential row, `-M_F Psi` | none (the fluid's density perturbation is `-div(rho u)`) |
| 2-D: constant-mode inconsistency through the interface coupling | consistent (the constant is exactly null, as without fluids) |
| needs `rho'_F`, level surfaces, interface conventions | needs `kappa_f` and the penalty |
| fluid block absent; well-conditioned | displacement block has the `1/eps` fluid near-kernel |

The refinement on the coupled system is `GaugeRefine()` overridden: zero
potential load and no tidal term in the refinement solves, the potential
accumulated alongside the displacement, and the accumulated pair left as
the warm start of the next `Solve()`.

### Equivalence and the Adams–Williamson condition

The two treatments describe the same physics **iff the fluid is materially
barotropic**: one thermodynamic parameter, so the constitutive `d rho/d p`
equals the background's, i.e.

```
N^2 = 0   <=>   kappa_f = rho^2 |grad Phi0| / |d rho / d r|      (Adams–Williamson)
```

Then the static `kappa`-response coincides with Dahlen's secular response,
and `TestMixedProblemGauged` (which builds `kappa_f` from the
Adams–Williamson condition on the discrete background gravity) confirms
agreement of solid displacement and potential to the discretisation level
under a zero-mean load. For a two-parameter fluid (`N^2 != 0`) the static
`kappa`-response is the "frozen" elastic one and *differs* from the
secular Dahlen response by `O(N^2)`; the gauge freedom also shrinks (§1),
though the penalty machinery still regularises the nearly-neutral case.
Benchmark comparisons against Dahlen or pyslfp at `l >= 1` must therefore
use neutrally stratified cores (PREM's outer core is Adams–Williamson by
construction; a uniform-density core is not).

### Degree 0

At degree 0 the treatments differ **by design**: Dahlen's fluid is
described by the potential alone and never sees `kappa_f`, while pyslfp
uses the fluid's bulk modulus at `l = 0`. The gauged fluid carries
`kappa_f` at every degree, restoring the compressible degree-0 physics;
`TestMixedProblemGauged.Degree0DiffersFromDahlen2D` pins the difference to
the fluid treatment, and the degree-0 Love numbers of the benchmark
(`doc/benchmarks.tex`, the Love-number family) compare the gauged result
with pyslfp.

## 4. Implementation

The machinery lives in `LinearQuasiStaticProblemBase`
(`quasi_static_problem.hpp`, `src/quasi_static_problem.cpp`):

- `SetGaugedFluid(const Array<int>& fluid_marker, Coefficient& mu_gauge,
  real_t epsilon, int refinements = 2,
  GaugePenalty penalty = GaugePenalty::Deviatoric,
  Diffeomorphism* map = nullptr)` builds a template form holding one
  penalty integrator on the marked attributes (`mu_gauge` is not owned and
  must outlive the problem):
  - `GaugePenalty::Deviatoric` (a fluid): without a map,
    `ElasticityIntegrator(eps mu_g, -2/dim, 1)`; with a map (the identity
    included) the **covariant** form, `ElasticTensorIntegrator(C, map)`
    with `C` the isotropic tensor of `lambda = -2 eps mu_g / dim`,
    `mu = eps mu_g` pulled back through the map, so that the penalty of a
    relabelled problem is the exact pull-back of the unmapped one and both
    sides of a change-of-variables identity use the same integrator class
    (`doc/mappings.md`).
  - `GaugePenalty::Harmonic` (a vacuum-extension field):
    `VectorDiffusionIntegrator(eps mu_g)`; it is gauge data shared between
    descriptions and refuses a non-identity map.

  Assembly then produces, next to the physical `A` (which `SystemMatrix()`
  still returns), the penalty `eps Q` on true dofs (no elimination;
  residuals are zeroed on essential dofs) and the regularised `A + eps Q`
  in one extra assembly — a form borrowing the physical integrators with
  the penalty appended (MFEM's borrowing forms own no integrators). The
  solver and preconditioner are built on `A + eps Q`;
  `RegularizedMatrix()` exposes it. `SetGaugeEpsilon()`,
  `SetGaugeRefinements()` and `ClearGaugedFluid()` adjust or remove it.
- `Solve()` runs `SolveLinearSystem` once, warm-started from the previous
  solution, then `GaugeRefine()`: `r = eps Q delta`, one further
  `SolveLinearSystem` per refinement, each **cold-started** from zero (the
  increment, not the solution, is solved for). `GaugeResiduals()` reports
  `|eps Q delta|` per step — their decay is the observed contraction, and
  `WarnGaugeContraction()` checks it at the end of every `GaugeRefine()`
  (base, mixed and referential).
- Block overrides of `GaugeRefine()`: the mixed class
  (`src/mixed_problem.cpp`) and the referential class
  (`src/referential_problem.cpp`) solve the coupled increment with zero
  potential load, accumulate the potential alongside the displacement and
  leave the accumulated pair as the next warm start. The referential class
  also overrides `SetGaugedFluid()` to assemble the Deviatoric penalty
  covariantly through the rheology's equilibrium mapping when no map is
  passed, and uses the same machinery with the Harmonic penalty for its
  ball-wide vacuum extension (`SetVacuumExtension`; see
  `doc/gauge_penalty_iteration.tex` for why that mode fails to converge
  under refinement).
- The slipping-interface class
  `LinearQuasiStaticReferentialSelfGravitatingSlipProblem` keeps the fluid
  on its own space and refuses `SetGaugedFluid()`; its fluid gauge is set
  with `SetFluidGauge(mu_gauge, epsilon)` (covariant Deviatoric penalty)
  and refined inside its constraint loop (`doc/slip_interface.tex`).
- Essential boundary conditions must not touch the gauged attributes
  (the penalty's column elimination is not folded into the loads); the
  geophysical use cases have none.
- `mu_gauge` sets the scale of the penalty; the fluid's own bulk modulus
  is a natural choice. The contraction factor is `~ eps mu_g / mu_solid`.

The mass-weighted rigid gauge (`SetMassWeightedGauge`) matters for
comparisons against exact radial solutions: the Euclidean true-dof gauge
leaves a mesh-asymmetry rigid translation (~1e-5 on the test meshes) that
uniform refinement does not remove, and which dominates e.g. a Lamé
comparison at the 0.5% level (`TestFluidGauge`).

## 5. The slipping interface

When a genuine tangential discontinuity is wanted (§2), the fluid and solid
carry separate displacement spaces on two SubMeshes of one parent, paired
nodally on the interface, with normal continuity enforced weakly by penalty
plus augmented-Lagrangian iterations or by a KKT multiplier. The
formulation, the discretisation (the pairing `J`, why the coupling is weak
rather than nodal), the constraint enforcement and the verification are in
`doc/slip_interface.tex`, sections "Discretisation of the slipping
interface", "Constraint enforcement" and "Verification". The gravity-free
version is exercised by `TestSlidingInterface`, `TestSlidingInterfacePar`
and `examples/sliding_fluid_ellipse.cpp`.

## 6. Verification

- **Purely elastic, exact references** (`TestFluidGauge`,
  `TestFluidGaugePar`, example `gauged_fluid_cavity`). A gravity-free
  static fluid supports a uniform pressure, so its exact response
  condenses onto the solid as the rank-one cavity term
  `(kappa_f/V) b(u) b(v)`, `b(v) = ∮ v.m dS`: the gauged solution matches
  the Sherman–Morrison solve of the condensed problem to 2e-3 under a
  degree-2 pressure. Uniform pressure has the exact Lamé solution with a
  fluid core (matched to 3e-5 in 2-D, 5e-4 in 3-D; the fluid's exact
  response `gamma x` is conformal, hence also the `Q`-minimal gauge, so
  the *whole* field is compared). The fluid pressure's uniformity is a
  further gauge-invariant observable (1e-4 relative spread on the coarse
  mesh under uniform load).
- **Contraction and eps-independence** (`TestFluidGauge`
  `ObservablesIndependentOfEpsilon2D`). Residual decay ~0.1 per step at
  `eps = 1e-2, mu_g = kappa_f`; observables agree to ~1e-4..1e-5 between
  `eps = 1e-2` and `1e-3` after refinement.
- **Self-gravitating equivalence** (`TestMixedProblemGauged`,
  `TestMixedProblemGaugedPar`): against the Dahlen path on the
  three-layer model with an Adams–Williamson core, solid displacement and
  potential to 2e-2 (the two discretisations' own error) under a zero-mean
  load; the degree-0 difference asserted present; SchurCG and BlockMINRES
  agree to 1e-6 within the gauged problem; serial and parallel agree to
  1e-6.
- **Welded against slipping**: on the two-layer hydrostatic disc the
  slipping-interface solver reproduces the welded gauged solution to
  discretisation level (`TestSlipProblem` `TwoLayerBarotropicCrossCheck`,
  `doc/slip_interface.tex`, "Verification").
- **Known leakage**: the rigid-tidal worst case of §1 (semi-convergence).

## References

- Friedman, J. L. & Schutz, B. F. (1978). Lagrangian perturbation theory
  of nonrelativistic fluids. *Astrophys. J.* 221, 937–957.
- Maitra, M. & Al-Attar, D. (2024). On the elastodynamics of rotating
  planets. *Geophys. J. Int.* 237, 1301–1338.
