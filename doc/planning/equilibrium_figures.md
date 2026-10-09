# Equilibrium states by constrained optimisation

**Status: theory written up, stage-1 value and derivative implemented
and validated.** The theory — the feasibility functionals, their
derivatives, the Sobolev gradients, the advection route and the stage-2
shape programme — is in `doc/equilibrium_figures.tex` (the reference;
this note is the roadmap and the open items). The stage-1 functional
and its envelope-theorem derivative are `DensityFeasibility`
(`equilibrium_figures.hpp`), validated against central finite
differences and the quartic Euler identity, serial and parallel
(`tests/TestEquilibriumFigures.cpp`, `TestEquilibriumFiguresPar.cpp`).

## The formulation, in brief

Given a density (stage 1) and later also an equilibrium mapping
(stage 2), quantify the distance from possessing a valid static state
by the deviatoric stress the fluid would have to carry. Three
functionals, compared in the reference document:

- **`J_F` (fluid-only, the primary)**: minimise the fluid's deviatoric
  norm over stress fields satisfying equilibrium in the fluid alone,
  interface traction free (the solid absorbs it). The multiplier is a
  no-slip Stokes flow on the fluid SubMesh — small, well-conditioned,
  one cheap solve per iteration, exact certificate for a connected
  solid. The whole-body generator then runs **once**, at the final
  density, for the model's background stress.
- **`J_mu` (weighted whole-body, the cross-check)**: the same saddle
  over the body with a two-region viscosity; approaches `J_F` with a
  floor `O(1/mu_s)`, extrapolatable in the contrast. Measured cost of
  the contrast (three-layer test, contrast 100): about twenty times
  the fluid-only solve.
- The *unweighted* whole-body restriction is **not** a certificate
  (solid deviatoric stress leaks into the fluid at the minimiser;
  the leakage proposition is in the reference document).

The derivative comes by the envelope theorem — the primal multiplier
is the adjoint variable; no transposed saddle solve — plus one Poisson
adjoint with the same operator as the potential solve:
`J' = -u . grad(Phi) + 4 pi G w`. Discretely exact (the FD check and
the Euler identity `rho . J' = 4 J` both pass at 2e-5 with solver
tolerances 1e-11).

## Implementation roadmap

1. ~~Value and derivative machinery~~ — done: `DensityFeasibility`
   (Poisson + DtN, the Stokes saddle through
   `MinimumDeviatoricEquilibriumStress` with the new essential-velocity
   option, value as the saddle's own value function, derivative as an
   L2 dual against a control space), with the FD/Euler test harness.
2. **Sobolev/Riesz tools**, standalone and general (shared
   inverse-problem infrastructure): L2 mass solve, H1 Helmholtz,
   iterated-Gram H2 (the default — H1 does not embed in C0 in 3-D),
   marker-restricted control with the metric on the full mesh (the
   buffer supplies the extension domain), fractional orders by
   Balakrishnan quadrature later.
3. **Outer loops**: nonlinear CG (Polak–Ribière, Armijo; the exact
   quartic line search of the reference document as an option) with
   the mass constraint and bounds (`SLBQPOptimizer` is the Euclidean
   projection; L2 exactly only with a lumped mass), **and** the
   advection loop (density advected along the Stokes flow: gradient
   flow of the energy, mass- and distribution-preserving); race them
   on milestone 1. L2 identification first, the Sobolev metrics as the
   first measured study.
4. ~~**Milestone 1**~~ — demonstrated (examples/equilibrium_density.cpp,
   three-layer model, lateral non-barotropic start, measurements under
   `build_parallel/examples/eqd*_*.csv`): every regularised loop
   removes the lateral content (best run: within-equipotential density
   spread 3.4e-2 -> 1.3e-2, L2 metric with roughness prior 1e-7), and
   the study MEASURED the semi-convergence the Sobolev-gradient
   analysis predicted — the unregularised L2 loop drives J down 9x
   while sending the density to [-4.5, 8.4] (mesh-scale pollution), and
   even the H2 loop, run 400 iterations, pushes J BELOW the base
   model's own discretisation floor (the unperturbed model evaluates to
   J = 9.4e-9, two thirds of the perturbed start) and the physical
   scatter grows again past that point: descending below the floor is
   fitting discretisation error. Open: a principled stopping rule (the
   floor is measurable by evaluating the reference model, or by
   h-refinement; a discrepancy-principle stop or keeping the prior on
   are the standard answers). The advection loop conserves the
   distribution but stalls at its explicit-step floor (~0.7 of the
   start); a semi-Lagrangian or flux-form transport step is the known
   upgrade. The exact rearrangement prediction (reference document §5)
   remains to be tested against an upgraded transport step.
5. ~~**Gauss–Newton**~~ — done: `DensityFeasibility::HessianAction`
   (the GN/residual split of the reference document §8; the GN dual is
   the gradient assembly with (u, w) replaced by the sensitivities;
   one saddle + two Poisson solves per product, three with the residual
   term), validated by FD against the gradient, symmetry and the
   serial–parallel pairing check. The example's `-loop gn` is
   Levenberg–Marquardt on it: with the roughness prior it reaches the
   best physical restoration of the study (within-equipotential spread
   3.4e-2 → 1.4e-2) in 15 iterations — an order of magnitude fewer
   than CG — converging onto the regularised solution; undamped it
   overfits past the floor like every other loop, measured.
6. **Stage 2**: interface-shape parameters, moving-domain terms,
   gradient checks against the shift-derivative benchmark; rotation.

## Open items

- **Promote the descent loop to a module.** The example's metric /
  mass-projection / projected-PR-CG core is self-contained; once the
  basic method is demonstrated, move it into a small library module
  (beside `riesz.hpp`) so the stage-2 and joint loops reuse it.

- ~~**A principled stopping rule / usable example defaults.**~~ Done:
  the example's defaults are now the measured recipe — Levenberg–
  Marquardt Gauss–Newton with the roughness prior (1e-7) — stopped
  where the dimensionless eta = |dev T|/|p| stagnates (two consecutive
  accepted iterations improving it by under 0.1%, counted only after a
  10% fall so the damped LM warm-up cannot trip it; `-eta` is an
  explicit discrepancy threshold instead, `-tol` the old relative-J
  stop). On the three-layer ball this stops at 16 of the former
  60-iteration grind at the same floor; the stop predicate is
  `DescentOptions::stop`, serial and parallel. The example also gained
  `-dim 2` (the disc models, an order of magnitude cheaper — the
  library's Poisson solves handle the 2-D singular constant by the
  uniform-flux compatibility correction and a zero-boundary-mean
  gauge, which keeps the envelope formulas discretely exact; FD-tested
  to the 3-D tolerances) and `-ic`/`-no-ic` (the three- or two-layer
  model; `elastogravity_two_layer_3d.msh` added to the build). The 2-D
  advection loop runs straight into the known off-manifold growth and
  is held by the J ceiling — further evidence for the semi-Lagrangian
  upgrade below.

- ~~**Geometry-driven defaults** (David's suggestion)~~. Done: the
  example's default model is now ASPHERICAL with a purely radial
  starting density — `equilibrium_bodies.py` builds
  `{flattened,cmb_topo}_{two,three}_layer_{2,3}d.mesh` (planetmodel
  stretch mappings, smoothstep-tapered across the buffer so the DtN
  keeps a spherical outer boundary; a quadratic taper folds elements at
  f = 0.1) — so the restoration generates the container's non-spherical
  equilibrium density; `-shape sphere` keeps the lateral-term
  experiment, `-blob` adds a fixed mantle anomaly, and `-vis` gained
  whole-body windows: the stress recovered EXACTLY in two solves — the
  fluid certificate's own T, and per solid component a
  minimum-deviatoric generator loaded by the certificate's interface
  pressure, the fluid pressure datum recovered by minimising the
  deviatoric norm over the gauge constant (a constant inner pressure on
  the mantle's partly-free boundary is real Lamé stress; on the fully
  loaded core it is dev-invariant). The weighted (scaled-viscosity)
  recovery is dropped: approximate at any contrast, and its leakage
  buries the fluid's share. Verification oracle: on the spherically
  symmetric model every piece sits at the discretisation floor and
  the recovered datum is the free-surface value. Measured
  calibration: the geometric signal is weak — eta roughly 8e-3 x
  flattening in 2-D, the degree-2 potential perturbation decaying
  inward — so it sits BELOW the order-2 floor (2e-3) and the default
  order is 3 on a disc (floor 2e-4; the flattened disc then starts at
  eta 8e-4 and J falls 18x in 16 iterations) but 2 on a ball for cost
  (3-D aspherical runs need `-o 3`). The wavy CMB demands an
  oscillatory, boundary-following correction that the H1 prior BLOCKS
  (189 iterations stuck with it, 29 to the floor without), so the
  prior's default is 0 for that shape — the first measured case where
  the smooth prior is the wrong selection, worth remembering for
  stage 2.

- **Per-component pressure constants.** The fluid-only solve removes
  one pressure constant; several fluid regions enclosed by solid need
  one per component (`background.cpp`, the clamped null-space branch).
- ~~**Disconnected solid (inner core).**~~ Done: rigid boundary groups
  on `StokesSaddleSolver` (bordered saddle, the lifted fields' base
  solves precomputed so a bordered solve costs one base solve plus
  dense algebra) and `RigidComponent` on the feasibility problem (the
  core's weight and torque on the border right-hand side, the adjoint
  "velocity" extended into the core as its rigid multiplier motion),
  FD-validated with the border and serial–parallel checked. Measured
  on the three-layer lateral model: the corrected certificate's start
  is 5.7x the clamped one — the net force on the inner core was the
  model's largest infeasibility, invisible to the clamped functional —
  the symmetric base agrees between the two to four digits (balance
  automatic), and the Gauss–Newton restoration under the corrected
  certificate collapses the within-equipotential spread to 4.1e-3,
  three times better than the best clamped run. The example's
  `-core rigid` is the default.
- **Value from an inexact solve**: evaluate through the load
  functional (error quadratic in the solve error), as the reference
  document recommends — the class already does this.
- **Solid planets.** For purely solid bodies there is no feasibility
  question; the deliverable is the background state itself. Wire a
  one-stop path (self-consistent potential + minimum-norm or
  minimum-deviatoric generator) through the same class family, so
  solid and fluid–solid models share a pipeline.
- **Parameterising the equilibrium-stress family.** The affine family
  is a particular solution plus the self-equilibrated fields
  (Div S = 0, traction-free, interface-continuous). Numerically:
  project an arbitrary symmetric tensor field tau through one
  elasticity solve, S = tau - def z(tau), with z solving the traction
  problem loaded by Div tau (weakly, int tau : def v) — the same solve
  as `MinimumNormEquilibriumStress`, and the gauge (tau -> tau + def w
  gives the same S) is quotiented exactly. The classical potentials
  (Airy in 2-D, Beltrami/Maxwell–Morera in 3-D, gauge def(v), with
  C1 or elasticity-complex elements) are the alternative if an
  explicit basis is ever wanted.
