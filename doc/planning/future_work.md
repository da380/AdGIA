# Future work: physics and capability extensions

---

## Nonlinear problems

### Nonlinear slip map and the mortar route

**Status:** future work; not implemented. The linearised slip is
implemented (penalty + AL for both organisations, and KKT with a
lower-order mortar-style multiplier for the single-valued organisation).

For finite slip the interface map σ ∈ Diff(Σ) is no longer
infinitesimal and the welded space cannot represent the configuration at
all. A mortar treatment (Lagrange-multiplier trace space, Wohlmuth 2001)
is the natural nonlinear path and a cross-check of the penalty/AL
machinery; the KKT path (`EnableKKT`: multiplier on a scalar space of the
solid SubMesh, Nanson-exact flux row) is its linear prototype. The
broken-ζ organisation (region-wise composition of the potential) is the
form a nonlinear slip map requires. In a nonlinear first-variation AL the
converged multiplier is the referential pressure π, which gives the
consistency check P_s N = −π ν on Σ; the linear solve's multiplier is the
first-order traction ϖ¹ instead, so the check does not apply there. The
same check can be applied directly to generated backgrounds.

**See:** `doc/slip_interface.tex`, "First variation: equilibrium and the
multiplier", "Constraint enforcement";
`include/AdGIA/referential_problem.hpp` (`EnableKKT`,
`EnableBrokenZeta`).

### Buffer motion in the nonlinear referential problem

**Status:** future work; not started.

In a finite-deformation referential problem the motion in the vacuum
buffer becomes a genuine unknown carrying mesh-motion energy. A ball-wide
displacement field with a physical buffer stiffness would then replace
the prescribed linear extension E (`NewRadialVacuumExtension`), and a
penalised extension regains a well-posed role. The ball-wide penalised
extension that exists (`SetVacuumExtension`, `GaugePenalty::Harmonic`)
fails in the linear problem because the vacuum has no stiffness against
a harmonic penalty scaling like 1/h²: the spectral condition of the
iterated refinement — λ_min of the pencil (A, Q) bounded away from zero
uniformly in h — fails with λ ~ h² (measured: order-2 bias 73 % in u,
14 % in ζ at ε_eff = 0.1). Any nonlinear penalised extension must respect
that condition.

**See:** `doc/gravitating_elasticity.md` §3.1;
`doc/gauge_penalty_iteration.tex`, "When it must fail: the spectral
condition"; `doc/quasi_static_models.tex`, "Discretisation: the vacuum
extension"; `include/AdGIA/referential_problem.hpp`
(`SetVacuumExtension`).

### Hybrid Eulerian variables outside the body

**Status:** parked.

Instead of extending the displacement into the vacuum buffer, the
potential outside the body could be carried as an Eulerian variable,
removing the vacuum extension. Not pursued: the prescribed extension is
exact (the extension is gauge), adds no unknowns and is implemented in
serial and parallel.

**See:** `doc/gravitating_elasticity.md` §3.1.

### Nonlinear elasticity

**Status:** future work.

The problem interface carries "Linear" in its class names to leave room
for nonlinear elastic problems; in the referential formulation φ_e is the
slot a Newton iterate would occupy. Not started.

**See:** `doc/quasi_static_models.tex`, "Naming and the problem-class
grid"; `doc/gravitating_elasticity.md` (preamble).

---

## Fluids and interfaces

### Two-parameter (stratified) fluids

**Status:** future work; the rheologies carry no second advected
parameter.

An independently advected entropy or compositional label s adds the
constraint w·∇s = 0 to the relabellings, changing which mismatch fields
are gauge and adding an N²-type volume term to the second variation; no
new interface integral arises. Needs a two-parameter constitutive model.
The slipping interface is the formulation for such fluids: with N² ≠ 0
slip is not removable by relabelling in general (see
[open_issues.md](open_issues.md)), and the interface terms (`B_Σ` ∝ π) are
genuine static physics.

**See:** `doc/slip_interface.tex`, Remark "Stratification";
`doc/gauged_fluid.md` §1–3; `doc/gravitating_elasticity.md` §5.

### Nested shells: two-sided fluid extensions

**Status:** future work; the slipping classes support one fluid region
inside one solid shell.

An inner core inside the fluid needs two-sided fluid extensions: the
prescribed `NewRadialFluidExtension` extends from one interface towards
the centre with t = (r/r_c)^{p_t}; with an inner core the fluid shell has
two solid boundaries and the extension must interpolate between two
independent solid motions (the inner core's rotation is then free for a
spherical ICB/CMB pair, so the null space grows, as with
`AddRegionRotations`). The theory is per interface and unchanged
(ν·w = 0 per boundary component, `B_Σ` per interface). The broken-ζ
organisation needs no fluid extension and scales trivially with topology,
so it is the natural route for multi-interface models.

Benchmark consequence: `slip` and `slip_broken` refuse more than one
fluid core (`benchmark_case.hpp`), so the inner-core models
(`inner_core`, `earth_like`, `prem_4`, `prem_6`) are outside their
cross-method comparison; nested shells would let the near-neutral PREM
cores serve as the cross-method ladder.

**See:** `doc/slip_interface.tex`, Remark "Topology, the extension, and
the broken-ζ alternative"; `doc/benchmarks.tex`, "The formulations
compared"; `include/AdGIA/referential_problem.hpp`
(`NewRadialFluidExtension`); `benchmarks/common/benchmark_case.hpp`.

### Implicit fluid extension for aspherical single-valued use

**Status:** proposal; relevant only if the single-valued organisation is
to be used on aspherical or mapped backgrounds (it refuses them via the
`Diffeomorphism::IsIdentity` guard; the broken-ζ organisation covers
them).

For aspherical fluid regions a prescribed geometric extension has no
natural recipe. Proposed: carry the fluid extension implicitly as an
auxiliary field with a fixed elliptic rule (a well-posed extra block
eliminated in the solver). Since the extension is gauge, the cheapest
rule may be chosen per geometry; unlike the ball-wide penalised vacuum
field, the rule is fixed, not penalised. The mapped forms of the
single-valued mismatch pieces (assembled at φ_e = id now) would also be
needed.

**See:** `doc/slip_interface.tex`, "The gravity Hessian of the broken
motion, explicitly", "Implementation"; `src/referential_problem.cpp`
(`AssembleSlipBlocks`).

### Rigid-preserving vacuum extension

**Status:** proposal.

With the tapered prescribed vacuum extension the rigid translations are
only near-null pairs of the referential classes (the taper does not
extend rigid motions rigidly; about 7e-3 for the slip class's common
translations). An extension rule that maps rigid motions of the body to
rigid motions of the buffer would make them exact discrete null pairs.

**See:** `doc/gravitating_elasticity.md` §3.1;
`include/AdGIA/referential_problem.hpp`
(`NewRadialVacuumExtension`).

### Chaljub–Valette two-potential fluid and the essential spectrum

**Status:** open question; not used as a formulation (the penalty gauged
formulation is the accepted one); the spectral prediction is untested.

Chaljub & Valette (2004) write the fluid displacement as
u = ∇χ + ξ s, s = ∇ρ/ρ − g/c², N² = s·g. For a non-rotating hydrostatic
fluid the null space of the operator is the divergence-free motion
tangential to level surfaces, whose L²-complement is the ∇χ + ξg form, so
the ansatz parameterises the complement of the gauge: one potential when
N² = 0 (Dahlen's barotropic fluid), two otherwise. Caveats:
orthogonality to the null space is a property of the ansatz, not
enforced; it fails once rotation makes the null space geostrophic, and
holds only approximately after discretisation. The formulation is
dynamic: the static limit of their eqs (9)–(10) gives div u = 0 and
u·g = ψ, which is right for degree ≥ 1 but loses the degree-0
compressible response, and ξ = (ψ − g·∇χ)/N² is singular in neutral
layers. A static version would need its own derivation.

*Prediction to test (a diagnostic, not a formulation change).* For
N² ≠ 0 the essential spectrum of the fluid operator is the interval
between 0 and the extremal values of N², σ_ess = [min(0, N²_inf),
max(0, N²_sup)], which touches 0. The static problem then sits at the
edge of a continuum, and the discrete near-kernel on the non-neutral
`fluid_core` (N² < 0) samples genuine slow physical modes with non-zero
right-hand-side projections, not only the discretised relabelling
kernel. So the gauged ε-sweep, the gauge-KKT stall and the plateau of
the solid-everywhere preconditioner ([solvers.md](solvers.md)) should be
markedly cleaner on an Adams–Williamson core than on `fluid_core`. Needs
the neutral-core model below.

**See:** `doc/gauged_fluid.md` §1; `doc/gauge_penalty_iteration.tex`,
"Semi-convergence and the operating point";
`doc/Elasticity/158-1-131.pdf`.

### Background states admissible in fluid regions

**Status:** (a) decided, not implemented; (b) proposal.

(a) In `MinimumDeviatoricEquilibriumStress`, fluid attributes should be
constrained to carry zero deviatoric stress (dev-free subdomains), so that
a generated background is admissible in the fluid regions rather than
merely minimising the deviatoric stress globally. This is the quantity
the equilibrium-figure functional measures
(J = ½∫_fluid |dev T_min|²).

(b) A mixed equilibrium-stress generator: a pure-pressure state imposed
exactly in the fluid, minimum-norm or minimum-deviatoric stress over the
solid regions only, coupled by Div T = ρ∇Φ and the interface traction
T_s N = −p ν. It is not in Al-Attar & Woodhouse (2010), whose generators
minimise over the whole body. The obstruction: the fluid pressure is
determined by ∇p = ρ∇Φ only when the fluid density is
equipotential-compatible, so aspherical fluid heterogeneity cannot be
free. The workable form inverts the problem: fix the solid shape (the CMB
in particular), adjust the fluid density to a self-consistent hydrostatic
form (admissible because the CMB can support non-uniform pressure), then
give the solid the minimum stress against the resulting −pν traction.
That is the density-control problem of the equilibrium-figure
formulation rather than a stress-space minimisation. It gates only
aspherical fluid-core backgrounds; radial backgrounds are unaffected.

**See:** `doc/gravitating_elasticity.md` §6 "Equilibrium stress in
general models"; [equilibrium_figures.md](equilibrium_figures.md);
`include/AdGIA/background.hpp`
(`MinimumDeviatoricEquilibriumStress`).

---

## Inverse problems

The library's forms carry the deformation gradient explicitly, which
points towards inverse problems: shape and density sensitivities with
full self-gravity. The items below are the groundwork.

### Equilibrium states and hydrostatic figures by constrained optimisation

**Status:** proposal; formulation agreed, not implemented. The full
formulation is in [equilibrium_figures.md](equilibrium_figures.md).

Minimise J(ρ, φ_e) = ½∫_fluid |dev T_min|², with T_min the
minimum-deviatoric equilibrium stress for density ρ and the
self-consistent potential; J = 0 iff a valid static state exists. Stage 1:
fixed shape, fluid density as control, mass constraint and a PREM
Tikhonov prior; gradients from the Stokes-saddle and Poisson adjoints,
smoothed by a Sobolev (Riesz) metric posed on the whole body, fractional
orders by resolvent quadrature. Milestone 1: recover the hydrostatic
profile on `fluid_core` from a perturbed density — the library's first
adjoint computation. Stage 2: the shape enters through interface-shape
coefficients of φ_e, with moving-domain terms and gradient checks against
the interface-shift perturbation benchmark; then rotation and figures
proper; Hessian actions for Newton–Krylov.

**See:** [equilibrium_figures.md](equilibrium_figures.md);
`doc/gravitating_elasticity.md` §1, §6.

### Shape derivatives and the shape-inversion variable

**Status:** future work; not started.

Because the mapping appears explicitly in every pulled-back form (and in
the mapped equilibrium-stress generators), derivatives of the solution
with respect to interface and surface shape can be assembled analytically,
and a `GridFunctionDiffeomorphism` can serve as the discrete shape
variable of an inversion. This underpins stage 2 of the equilibrium-figure
formulation and any shape inversion. Two lessons constrain the design:
shape parameterisations must keep F side-consistent at interfaces
(interpolated F or attribute-aligned kinks; see
[open_issues.md](open_issues.md), "Broken-ζ interface forms when F jumps
across Σ"), and the discrete relabelling gauge is only approximate, so an
explicit interface-shape parameterisation is preferred to projecting
gradients onto gauge complements. The interface-shift perturbation
benchmark provides the degree-0 derivative check.

`GridFunctionDiffeomorphism` is the intended inversion variable, and
both equilibrium-stress generators pull every form back with an explicit
F, so their derivatives with respect to the mapping can be formed
analytically.

**See:** `doc/mappings.md` (preamble, §4 "The `Diffeomorphism`
interface"); `doc/gravitating_elasticity.md` §6;
`benchmarks/perturbation/README.md`.

### Adjoints and sensitivity kernels with full gravity

**Status:** non-gravitating first-order adjoints exist as examples;
no library-level adjoint machinery yet.

What exists: `examples/adjoint_elasticity.cpp` (static isotropic
kernels for κ and μ from a windowed surface-station observable, FD
verification at roundoff grade, and the derivative-versus-gradient
distinction made concrete through the L2/H1/H2 Riesz maps of
`riesz.hpp` on a buffered mesh) and `examples/viscoelastic_adjoint.cpp`
(final-time observation, kernels for κ, μ AND the Maxwell viscosity η;
the adjoint is the same forward operator in reversed time with the
internal-variable jump `m⁺(0⁺) = dev ε(u⁺_g)/τ`; store-all
checkpointing, because the dissipative forward field cannot be
recovered by backward integration — the essential difference from the
seismological trick; FD verification with an O(dt²) time floor and an
O(h) nodal-sampling floor, both documented in the header).

Library work this implies: discrete interpolators for forming kernel
fields (the examples use element-wise coefficients), checkpointing
beyond store-all (windowed/binomial), and the exact discrete adjoint
of the nodal μ/τ sampling if the O(h) verification floor ever matters.

The gravitating extension: for MODULI kernels nothing changes beyond
the forward and adjoint problems being gravitating. A DENSITY kernel
must remember that the density also sources the equilibrium Poisson
equation, so the zeroth- and first-order dynamics both enter the
Lagrangian — same idea, more work. The subtle open point is that the
equilibrium stress depends on the density too, and non-uniquely (the
equilibrium-stress degeneracy of `equilibrium_stress.cpp` / the AW10
generators). Theory: Yu, Al-Attar, Syvret & Lloyd (2025,
`doc/Elasticity/ggae388.pdf`), whose rotational feedbacks enter
through the `SetTidalLoad` substrate.

Beyond that, as before: analytic shape derivatives combined with full
self-gravity are a combination no surveyed code offers (only the
spectral code of Lloyd et al. 2024 has kernels, 1-D structure;
G-ADOPT has derived adjoints but no self-gravity); targets are the
adjoints of the equilibrium-figure stage 1 and l ≥ 1
interface-topography kernels verified against finite differences of
the kind the degree-0 shift check uses. Until such kernels exist, the
1-D side of the present perturbation check is a finite difference of
pyslfp solves and stays labelled so. Viscoelastic adjoints at scale
need unrelaxed-operator applications at many observation times (see
[solvers.md](solvers.md), "Caching unrelaxed and effective
operators").

**See:** `examples/adjoint_elasticity.cpp`,
`examples/viscoelastic_adjoint.cpp`;
`doc/BenchmarkPapers/code_survey.md` ("Cross-cutting reading");
`doc/benchmarks.tex`, "The perturbation family";
`benchmarks/perturbation/perturbation_check.py`.

---

## Earth models and benchmarks

### PREM Love numbers at production resolution

**Status:** partly done. Elastic `prem_4` runs against pyslfp exist at
`h = 0.2` (CMB treatments, Dahlen and gauged; h'₀ within 6e-4 of
pyslfp); the merged PREM models' load Love numbers are within 0.2 % of
PREM's.

Remaining: the server-profile ladder (`h = 0.2, 0.15, 0.1`, orders 2 and
3, degrees to 16); `prem_6`, whose 21 km crust needs a mesh of millions of
elements and a large machine; the slipping methods on PREM (needs nested
shells, above); viscoelastic PREM histories.

**See:** `doc/benchmarks.tex`, "The models", "The CMB approximations:
results"; `benchmarks/common/models.py`.

### Neutral (Adams–Williamson) core model

**Status:** proposal; no such model in `benchmarks/common/models.py`.

A model with an N² ≈ 0 fluid core and no inner core, so that the
compressible methods (gauged, referential, slip) converge to the same
secular response as Dahlen and pyslfp. On `fluid_core` the O(N²)
stratification split pollutes both the cross-method h-ladder and the
derivative ladder (a method-independent ≈ 17 % absolute offset in the
degree-2 shift derivatives at every ε). A neutral core would be the clean
cross-method h-ladder, the clean derivative ladder, and the comparison
case for the essential-spectrum prediction above.

**See:** `doc/gauged_fluid.md`, "Equivalence and the Adams–Williamson
condition"; `doc/benchmarks.tex`, "The perturbation family".

### Aspherical reference body with a fluid core; an independently generated aspherical mesh

**Status:** future work.

`aspherical_reference` runs single solid layers only (`homogeneous`,
`linear_solid`). Its aspherical mesh is the image under the radial stretch
of a spherical mesh, so the pulled-back problem is the spherical problem
on the original connectivity and amplitude independence is expected; a
mesh generated on the aspherical body directly would be a genuinely
independent-mesh test. A fluid core needs the mapped slipping forms
certified first ([open_issues.md](open_issues.md)) or the welded gauged
method.

**See:** `doc/benchmarks.tex`, "Leg C: an aspherical reference body";
`meshes/aspherical_body.py`.

### Reproduce the Spada et al. (2011) benchmark tables

**Status:** future work; not started.

Tables 3–14 of Spada et al. (2011) are the standard 1-D reference numbers
for validating a GIA code on spherical models: a 3-layer incompressible
Maxwell model with an elastic lithosphere and a uniform inviscid fluid
core (M3–L70–V01), with the community conventions (Wu & Ni 1996 CMB
conditions, degree-1 Love numbers in the CM frame with k₁ᴸ = −1). Their
viscoelastic content is normal-mode (Laplace-domain) based, whereas the
library's runs are time-domain, so a comparison needs either
Laplace-domain output from the library or time-domain reconstructions of
the reference responses (the Laplace reference machinery of the
viscoelastic Love-number family does the latter).

**See:** `doc/BenchmarkPapers/code_survey.md` ("The published
benchmarks"); `benchmarks/viscoelastic/love/`.

### 3-D laterally varying viscosity: the time-stepping advantage

**Status:** future work; not started.

Across the surveyed 3-D GIA codes the explicit-Euler (or explicit RK2)
step restriction Δt ≲ min Maxwell time recurs, and it forces viscosity
floors (e.g. 2×10¹⁹ Pa s in Lloyd et al. 2024). The library's exponential
trapezoid, SDIRK23 and adaptive steppers have no stability limit. A 3-D
run with laterally varying viscosity (contrast 1e3 or more) would show the
advantage where it matters; the stepping survey covers the
relaxation-time contrast axis only on a 2-D beam, and the sphere
`lateral_weak` study is limited by the cost of high-contrast solves
([solvers.md](solvers.md)).

**See:** `doc/BenchmarkPapers/code_survey.md` ("Cross-cutting reading");
`doc/viscoelasticity.md` §3 "Choosing a scheme";
`benchmarks/viscoelastic/stepping/README.md`.

### Stepping survey: power-law and elastic-region axes

**Status:** future work.

The beam example has a power-law branch (`-gamma`) and a long-term
modulus (`-mu-inf`); a survey axis for each would follow once
correspondence-principle (or other) references exist for them.

**See:** `benchmarks/viscoelastic/stepping/README.md`;
`examples/viscoelastic_schemes.cpp`.

### Further examples

**Status:** proposal.

A moment-tensor post-seismic relaxation example (the seismic-cycle use
case of the library) is not written.

**See:** `examples/README.md`.

### Figure suggestions

**Status:** proposal.

1. 3-D field render: `talk/field3d.png` (ParaView, `talk_render.py`) is a
   faceted half-ball with an unlabelled colour bar; a whole-sphere render
   coloured by U with a diverging map and displacement arrows would read
   better (`talk_fieldmap.png` carries the message meanwhile).
2. Accuracy–cost scatter: error against seconds per solve, one point per
   method and mesh, once a cross-method h-ladder exists (needs the
   neutral-core model).
3. CMB report across h on a stratified core: deviation-from-full against
   the full treatment's own error as a function of h, to show when the CMB
   approximation becomes the bottleneck.

**See:** `benchmarks/talk_figures.py`, `benchmarks/talk_render.py`,
`benchmarks/love_numbers/cmb_report.py`.

---

### Viscoelastic referential problems

**Status:** future work.

`ReferentialElasticRheology` carries the elastic tensor and the
equilibrium stress of a general reference state but no relaxing
branches, so the viscoelastic layer runs only on the traction, clamped
and mixed classes. A referential viscoelastic rheology would supply
branch tensors (relabelled with the equilibrium mapping, as the elastic
tensor is) and let `ViscoelasticOperator` drive the referential and
slipping problems, e.g. for relaxation of an aspherical body.

**See:** `include/AdGIA/referential_problem.hpp`
(`ReferentialElasticRheology`); `doc/viscoelasticity.md`.

## Sea level

### The referential sea-level leg and the generalised rotational theory

**Status:** proposal (the plan's WP7).

Port the sea-level boundary terms to the referential class, where the
combination `u·∇Φ + φ` is the referential potential ζ and the water
feedback collapses to a single ζζ boundary mass (the condensed system
of `sea_level_plan.md`); gate against the mixed-class results. The
Maitra & Al-Attar (2024) generalised rotational theory hangs off this
leg: the traditional theory implicitly assumes an isotropic background
inertia tensor (Yu et al. §2.4), which bites for 3-D models.

**See:** `sea_level_plan.md` ("The condensed system in our
variables"); `doc/Elasticity/ggae092.pdf`.

### Equilibrium backgrounds with surface water

**Status:** parked (the foundational caveat).

All sea-level theory to date computes forces that belong to a rotating
model with oceans on a non-rotating model without them — the reference
sea level is an equipotential of nothing. The consistent fix: the
equilibrium-figures machinery builds self-consistent rotating
backgrounds extended to carry surface water, and the generalised
rotational theory is the matching perturbation framework — both on the
referential leg.

**See:** `equilibrium_figures.md`; `sea_level_plan.md`.

### Ice-age production runs

**Status:** open — the server campaign.

The pieces exist end to end (`IceHistory`, the ice-ng export, the
sea-level-band refined meshes, `examples/ice_age_loading.cpp` as the
chain's demonstration on a toy rheology): a production run is a
realistic layered viscoelastic model, the resolved meshes, the full
ICE-NG date list and migration on — plus the resolution ladders of the
sea-level benchmark family (rungs 1–3, `--coast`, `--timings` on the
server profile).

**See:** `benchmarks/sea_level/README.md`;
`examples/ice_age_loading.cpp`; `open_issues.md` ("The sea-level
equation").

## Deferred physics

### Out-of-scope physics of the linearised theory

**Status:** future work; not implemented.

1. Time dependence and Coriolis forces, via the Tisserand-frame machinery
   of Maitra & Al-Attar (2024); with them, rotating fluid cores (the gauge
   becomes geostrophic) and time-domain Chandler-wobble runs with a
   drift-free referential fluid core.
2. The gravitational-stress-tensor form of gravity (the N of Maitra &
   Al-Attar 2024), equivalent to the body-force form up to an external
   boundary term.
3. Finite deformation, where the referential potential and the exact slip
   map χ ∈ Diff(Σ) become essential (see "Nonlinear problems" above).
4. The stress dependence of the moduli themselves (the Π tensor of
   Maitra & Al-Attar 2021), a constitutive question for Earth models.
5. Oceans: the free-surface condition T⁰·N̂ = 0 of the background against
   the background surface-load convention is undecided; an ocean is the
   seafloor-constraint-only case of the slipping interface.
6. Surface topography from CRUST-1.0, and mesh recipes as data, are
   deferred to planetmodel.

**See:** `doc/gravitating_elasticity.md` (preamble, §3, §4, §7);
`meshes/README.md`.
