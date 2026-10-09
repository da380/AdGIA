# Equilibrium figures, stage 2: shape optimisation — implementation plan

*Status: draft for discussion, 9 October 2026. The theory is already in
`doc/equilibrium_figures.tex` (§stage2, §second, §shape-second); this
note maps it onto the library and sequences the work. Questions for
David at the end.*

## What the theory hands us (tex summary, for orientation)

- **Gauge eliminated by construction**: the shape enters only through an
  explicit parameterisation `φ_e = φ[s]` — interface topographies lifted
  into the volume by a fixed rule, identity at the DtN sphere — so no
  relabelling redundancy survives discretely (the 1 Oct lesson against
  projected gradients).
- **The shape derivative is a volume form**: eq. (shape-derivative) and
  its field dual `Γ(δφ) = ∫ Ψ : Dδφ + ∫ Ψ_g : Dδφ`, with Ψ, Ψ_g the
  Eshelby tensors of the Stokes and Poisson Lagrangians, assembled from
  fields already computed for the density gradient (u, p, ζ, w) — **no
  additional solve per iteration**. Per-parameter derivatives are dot
  products `∂J/∂s_i = Γ(θ_i)` with the tangent lifts θ_i.
- **Lift dependence is physical, not a bug**: at fixed referential
  density the derivative depends on the lift (the gauge identity,
  eq. gauge-identity); the Eulerian pairing `Γ_E` is the lift-free
  Hadamard object. Which to use is a choice of coordinates on the joint
  (ρ, s) space; referential conserves region masses automatically.
- **Second order is routine**: fixed reference mesh, F-dependence
  analytic at quadrature points, symmetric second derivatives; GN shape
  block costs ~1 Poisson + 1 saddle per parameter, a dense `N_s × N_s`
  block beside the field density block.
- **Rotation**: centrifugal ψ is analytic data; compatibility = two new
  linear constraints on ρ (centre of mass on the axis, axis principal).
- **What stage 2 is for**: `J_F`'s shape control matters where fluid
  supports its own boundary or an inner core must balance; the
  hydrostatic-figure functional `J_hyd` (every density interface an
  equipotential) is the real stage-2 target, with fixed-aspherical
  `J_hyd > 0` at the stage-1 minimiser as the intermediate diagnostic.
- David's prior (6 Oct): for proper hydrostatic bodies this is
  **benign** — nearly spherical family, mild parameterisation.

## Gaps between tex and library (verified 9 Oct)

1. `DensityFeasibilityProblem` is identity-map only — stage 1 handled
   asphericity through meshes, never through `φ_e`. The mapped assembly
   (tex "Mapped form": `ε_φ`, `a(F)` Poisson, mapped loads) is the
   largest new piece. Heavy reuse available: the covariant mapped
   integrators, `a(F)` transformed-Poisson machinery, and the
   interpolated-F discipline from the slip/relabelling work.
2. No shape-parameterisation object: need `φ[s]` with tangents θ_i,
   built on the mapping layer (`RadialDiffeomorphism`,
   `TaperedDiffeomorphism`, `GridFunctionDiffeomorphism`) and matching
   `meshes/equilibrium_bodies.py`'s smoothstep taper so maps and meshes
   describe the same bodies.
3. No Γ/Eshelby assembly, no GN shape block.
4. `ConstrainedMetric` takes ONE linear constraint; rotation needs
   three more (mass + 2 compatibility) and the joint problem wants a
   product control space (ρ field ⊕ s ∈ R^{N_s}) in the metric, CG and
   LM — a descent-module extension.
5. `J_hyd` exists only in the tex; wiring it as a `DescentFunctional`
   over the existing whole-body generator is a small, separable item.

## Gates (in the order they become available)

- **Mesh-vs-map identity** (the WP-S1 gate): the same aspherical body
  realised two ways — aspherical mesh with identity map (stage 1 as-is)
  vs spherical mesh with `φ[s]` — must agree on J and the density
  gradient to discretisation accuracy. The strongest possible check of
  the mapped assembly, and it reuses the stage-1 examples unchanged.
- **FD on ∂J/∂s_i**, and the externally certified interface-shift
  derivative benchmark (`benchmarks/perturbation/`) as the independent
  reference for interface-location derivatives.
- **The gauge identity** (eq. gauge-identity) as a discrete certificate:
  `Γ(F_e η)` vs `−∫ J'_ρ Div(ρη)` for interface-tangent η — catches
  Eshelby-assembly bugs with no FD cost. Likewise lift-independence of
  `Γ_E` (two lifts, same trace).
- **Maclaurin** (rotation on, constant density): the exact rotating
  figure — in 2-D the ellipse with its closed-form Ω²(ellipticity)
  relation (laptop-fast house-style first leg), then the 3-D spheroid.
  Shape descent from a sphere at given Ω must land on the exact figure.
- **Clairaut/Radau** (layered, small Ω): flattening ratios of a
  two/three-layer model vs the Clairaut ODE solved radially (a
  `disc_radial`-style 1-D tool, seconds per point).
- **Fixed-aspherical `J_hyd` diagnostic**: on the bump/flat meshes,
  `J_hyd > 0` at the stage-1 minimiser, driven to the floor once shape
  moves. Inner-core interface term sign (tex flags it "to be fixed with
  the stage-2 implementation") pinned by FD + an interface-continuity
  style oracle, per the stage-1 sign lessons.

## Work packages

- **WP-S0** — this note; resolve the open questions below.
- **WP-S1** — **mapped stage 1**: `φ_e`-aware assembly in
  `DensityFeasibilityProblem` (fixed map, density control as today);
  mesh-vs-map identity gate. De-risks everything downstream and is
  useful alone (density restoration on mapped bodies).
- **WP-S2** — **shape parameterisation layer**: `φ[s]` + tangents θ_i;
  interface-harmonic coefficients first (the mild nearly-spherical
  family), the interface-FE-field variant later (same Γ projection
  serves both); lift rule shared with `equilibrium_bodies.py`.
- **WP-S3** — **the shape derivative**: Eshelby-tensor assembly for Γ,
  `∂J/∂s = Γ(θ_i)`, the Γ_E variant; FD + gauge-identity +
  lift-independence + interface-shift-benchmark gates. Centrifugal term
  included from the start (it is one extra `δφ`-term).
- **WP-S4** — **joint descent**: product control space in the descent
  module, multi-constraint `ConstrainedMetric`, GN shape block (dense)
  + cross blocks; `J_hyd` wired as a functional; fixed-aspherical
  diagnostic then full shape+density descent, non-rotating.
- **WP-S5** — **rotation**: ψ, the two compatibility constraints,
  Ω-continuation if needed; Maclaurin (2-D, 3-D) and Clairaut gates.
  This is "figures proper".
- **WP-S6** — **applications**: PREM-pull prior, hydrostatic reference
  figures for the 3-D programme (the original motivation: valid initial
  states for aspherical models), handoff of backgrounds to the
  perturbation layer.

**Recorded stretch goal, not planned (David, 9 Oct):** self-consistent
surface water / sea level as part of the figure — interesting but
subtle: disconnected water bodies at the surface and topology change as
the system approaches equilibrium. Ties to the foundational caveat in
`doc/planning/sea_level_plan.md` (the reference sea level as an
equipotential of nothing); the fix route named there (equilibrium
backgrounds carrying water + M&A perturbation theory) starts here, but
only after WP-S6.

## Open questions for David

1. **Control coordinates for the joint descent**: referential pairing
   (region masses conserved automatically, matches the rearrangement
   semantics) vs the Eulerian `Γ_E` (lift-free, classical Hadamard)?
   The tex notes a joint stationary point is the same in both; my
   inclination is referential as primary with `Γ_E` implemented as a
   cheap derived check, but this is a modelling choice.
2. **Which interfaces move first**: outer surface + CMB (the
   equipotential pair that defines figures), with the ICB joining when
   the inner-core term's sign is pinned? Or all at once?
3. **Parameterisation scope at first**: axisymmetric low-degree
   harmonics only (degree 2, maybe 4 — enough for Maclaurin/Clairaut
   gates), generalising later?
4. **Rotation entry**: direct solve at target Ω or continuation in Ω²?
   (Maclaurin suggests direct is fine at Earth-like slowness; the 2-D
   gate will measure it.)
5. **Where `J_hyd` sits**: as a separate `DescentFunctional` beside
   `J_F` (my assumption), or folded into the existing problem class as
   an option?
6. **Descent-module extension shape**: product control space as a
   first-class `DescentFunctional` concept (field ⊕ parameters), or a
   thin adapter that flattens (ρ, s) into one vector with a block
   metric? The latter is less invasive; the former is cleaner if more
   joint problems are coming (they are — the inverse-problem road).
