# Linearised self-gravitating elasticity: theory

The linearised theory of a self-gravitating, pre-stressed elastic body
on which the referential problem classes (`referential_problem.hpp`)
and the background-state module (`background.hpp`) are built: a
general reference body, non-hydrostatic equilibrium stress, and
fluid–solid interfaces that may slip. The mixed class of
`doc/self_gravitation.md` is its hydrostatic, natural specialisation.
Only the quasi-static linearised problem is treated; rotation enters
solely through the centrifugal part of the background potential.

Sources (the PDFs are in `doc/Elasticity/` where a file name is given):

- **Maitra & Al-Attar (2021)**, On the stress dependence of the elastic
  tensor, *Geophys. J. Int.* 225, 378–415 (`ggaa591.pdf`): the notation
  adopted here (close to Marsden & Hughes 1983) and the treatment of the
  elastic-tensor/initial-stress relations of §1–§2.
- **Maitra & Al-Attar (2024)**, On the elastodynamics of rotating
  planets, *Geophys. J. Int.* 237, 1301–1338 (`ggae092.pdf`): the
  equations of motion, the referential gravity (their eq. 21) and the
  exact tangential-slip machinery. Its *linearised* fluid–solid boundary
  integral drops a surface-Jacobian factor (harmless for their static
  applications); the linearised boundary terms are taken from AC18
  instead.
- **Al-Attar, Crawford, Valentine & Trampert (2018)**, Hamilton's
  principle and normal mode coupling in an aspherical planet with a
  fluid core, *Geophys. J. Int.* 214, 485–507 (`ggy141.pdf`, "AC18"):
  the linearised equations in weak form about a general equilibrium
  (their eqs. 108–137), including the fluid–solid surface terms and the
  particle-relabelling transformation laws.
- **Al-Attar & Crawford (2016)**, Particle relabelling transformations
  in elastodynamics, *Geophys. J. Int.* 205, 575–593
  (`doc/BenchmarkPapers/ggw032.pdf`): the natural/non-natural and
  hydrostatic/non-hydrostatic terminology of §7.
- **Woodhouse & Deuss (2007)**, Theory and observations — Earth's free
  oscillations, *Treatise on Geophysics* vol. 1, Elsevier
  (`Woodhouse-Deuss.pdf`, "WD"): the mixed (Eulerian-potential)
  formulation, and the explicit statement of the pressure-modified
  moduli (their eqs. 59–64).
- **Al-Attar & Woodhouse (2010)**, On the parametrization of
  equilibrium stress fields in the Earth, *Geophys. J. Int.* 181,
  567–576 (`181-1-567.pdf`, "AW10"): the parametrisation of equilibrium
  stress fields in general models, and the modern rendering of Dahlen's
  hydrostatic-region argument (§6).
- **Valette (1986)**, About the influence of pre-stress upon adiabatic
  perturbations of the Earth, *Geophys. J. R. astr. Soc.* 85, 179–208,
  and **Valette (1991)**, Gravito-elastodynamics of a pre-stressed
  elastic earth, *Geophys. J. Int.* 104, 555: the geometric
  (Weingarten-operator) treatment of the fluid–solid interface terms —
  the content of Woodhouse & Dahlen with the interface curvature
  explicit.

## 1. Notation and kinematics

Following Maitra & Al-Attar (2021): reference body `B` with referential
coordinates `x`; motion `φ(x, t)`; deformation gradient `F = Dφ`
(`F_ij = ∂φ_i/∂x_j`, second index referential), Jacobian `J = det F`,
right Cauchy–Green tensor `C = Fᵀ F`. Referential density
`ρ(x) = J ϱ∘φ`. Stress measures: first Piola–Kirchhoff `P` (traction per
referential area, `T = P·N̂`), second Piola–Kirchhoff `S`, Cauchy `σ`,
related by `P = F S = J (σ∘φ) F⁻ᵀ`. Hyperelasticity: strain energy
`W(x, F) = V(x, C)` (frame indifference), and

```
P = D_F W,     S = 2 D_C V.
```

Two elastic tensors:

```
A = D²_F W     "first":  hyperelastic (major) symmetry only, A_iAjB = A_jBiA
C = 4 D²_C V   "second": classical symmetries, 21 components
```

related by `A = L_F C L_Fᵀ + R_S` — the `R` term carrying the stress
explicitly. A reference configuration is **natural** when the equilibrium
is `φ_e(x) = x` (`F_e = 1`); then `P_e = S_e = σ_e =: T⁰`, and

```
A_iAjB = C_iAjB + δ_ij T⁰_AB .                                   (*)
```

An **equilibrium** satisfies `Div P_e + ρ γ_e = 0` with `γ_e` the (self-)
gravitational acceleration plus centrifugal force; nothing requires `T⁰`
hydrostatic. The reference body itself is arbitrary (it need not be a
configuration the body ever occupies): all physical statements are
invariant under particle relabellings `ξ`, with

```
ρ̃ = J_ξ ρ∘ξ,   W̃(x̃, F̃) = J_ξ W(ξ(x̃), F̃ F_ξ⁻¹),
Ã_ijkl = J_ξ (F_ξ⁻¹)_jm (F_ξ⁻¹)_ln A_imkn ∘ ξ,   ϖ̃ = J_{ξ|Σ} ϖ∘ξ,
```

(AC18 eqs. 83, 134–136), the surface multiplier picking up the
*surface* Jacobian. The mapping layer (`doc/mappings.md`) implements
these laws; for the second tensor and the equilibrium stress they are
`RelabelledElasticTensorCoefficient` and `PullbackStressCoefficient`,
and `RelabelledBackground` owns the whole transformed chain.

**The equilibrium mapping may be taken continuous.** In a static state
the physical positions are continuous across a fluid–solid boundary (no
cavitation or overlap), so a continuous reference admits a continuous
equilibrium mapping `φ_e`: any tangential discontinuity in the labelling
of the *equilibrium* can be removed by relabelling the fluid side. This
holds because the constitutive data `(Ĉ, S_e, ρ)` are transported with
the chosen labelling (they are defined at the natural reference and
pulled back by the laws above); pinning the data to a fixed reference
while varying the map would make the map physical rather than a choice
of description. The statement concerns the description of the
equilibrium only. For *perturbations* about it, with the data held fixed
on the reference, a tangential slip is in general not removable by
relabelling (§5), and the slipping-interface formulation
(`doc/slip_interface.tex`) is the general one.

## 2. The linearised quasi-static problem

Write `φ = φ_e + u` and expand the action to second order. About a
natural reference the quadratic elastic-plus-initial-stress energy is (WD
eqs. 12–16, AC18 eq. 97)

```
E₂(u) = 1/2 ∫_B  C ε(u) : ε(u)  +  T⁰_AB ∂_A u_k ∂_B u_k  dV,      (E)
```

i.e. the classical-symmetry tensor `C` acting on the symmetric strain,
plus the initial stress acting on the **full** gradient (it enters
through the second-order part of the Green strain). Equivalently
`E₂ = 1/2⟨A·Du, Du⟩` with `A` from (*). Both terms are symmetric in
`u ↔ v`; `A` has no minor symmetries, the same "major symmetry only"
structure as the pulled-back tensors of the mapping layer. About a
general (non-natural) reference the material term becomes
`Ĉ sym(F_eᵀDu) : sym(F_eᵀDv)` (the linearised Green strain) and the
geometric term `S_e : (DuᵀDv)`; these are `MaterialStiffnessIntegrator`
and `GeometricStiffnessIntegrator`, combined by
`ReferentialElasticRheology`.

The linearised momentum equation in weak form, about a general
equilibrium, with self-gravity and a steadily rotating frame
(AC18 eq. 137, quasi-static: drop `∂²_t u` and Coriolis):

```
∫_B { ⟨A·Du, Dw⟩ + ρ⟨Ω×(Ω×u) − γ¹(u), w⟩ } dV + (surface terms, §4)
    = loads,                                                        (M)
```

where `γ¹` is the linearised gravitational acceleration (§3) and the
centrifugal term can be absorbed into the background potential
`Φ₀ → Φ₀ + ψ` for constant `Ω`.

### The elastic tensor dictionary

Seismology does not tabulate `C`. With hydrostatic initial stress
`T⁰ = −p⁰ 1`, WD (eqs. 59–61) rearrange the equations so that pressure
never appears explicitly: the `p⁰`-parts of `A` combine with the
equilibrium condition `∇p⁰ = −ρ∇Φ₀` into the familiar gravity terms
`ρ[∇(u·∇Φ₀) − ∇Φ₀ div u]`, leaving a stiffness term `C^eff ε(u):ε(v)`
with the **effective** tensor

```
C^eff_ijkl = C_ijkl + p⁰ ( δ_ij δ_kl − δ_il δ_jk − δ_ik δ_jl ),
```

which again has all classical symmetries. **PREM's moduli are the
isotropic components of `C^eff`, not of `C`** (WD eqs. 62–64 define
κ, μ, A, C, N, L, F from `C^eff`). The conversion is:

```
λ_bare = λ_PREM − p⁰,    μ_bare = μ_PREM + p⁰,    κ_bare = κ_PREM − p⁰/3 .
```

This is not a small correction: at the CMB `p⁰ ≈ 136 GPa` against
`μ_PREM ≈ 290 GPa` — a ~45 % shift in the bare shear modulus. In a fluid
(`W = V(x, J)`, `P = −pJF⁻ᵀ`): `μ_PREM = 0` but `μ_bare = p⁰` — the bare
strain-energy tensor of a fluid has *pressure-sized shear components*,
which combine with the explicit `T⁰` term of (*) to make the incremental
traction purely normal. In the library:

- `LinearQuasiStaticMixedSelfGravitatingProblem` is exactly WD eq. (59):
  it assembles `C^eff` plus the rearranged gravity terms, so PREM moduli
  are the correct input there.
- The referential classes assemble (E) and need the bare `C`.
  `BareElasticTensorCoefficient(dim, C_eff, p0)` applies
  `C = C^eff − p⁰(δδ − δδ − δδ)` at the coefficient level, so model
  files keep seismological moduli and the conversion cannot be silently
  forgotten; `RadialHydrostaticBackground` applies it with the `p⁰` it
  computes. `BareTensorIsotropicConversion` (`TestGeneralisedStiffness`)
  checks the formula, and `FluidRelabellingNullPair` (§5.1) shows that
  the conversion is load-bearing.
- For hydrostatic `T⁰` the two assemblies describe the same physics.
  The systems are not identical — they are related by the change of
  variables `ζ¹ = φ¹ + u·∇Φ₀` (§3.1) — so the check is at solution
  level: `HydrostaticCrossCheck2D` (`TestReferentialProblem`) maps the
  referential solution onto the mixed class's. At integrator level,
  `GeometricEqualsVectorDiffusionForIsotropicS` and
  `MaterialIdentityMapEqualsElasticTensor` check the two stiffness pieces
  against their classical counterparts.

The heuristic §1.1 of Maitra & Al-Attar (2021) also parametrises how the
*moduli themselves* respond to incremental stress (their `Π` tensor,
Dahlen and Tromp & Trampert as special cases); that is a constitutive
question about Earth models, not a formulation question, and does not
enter the linearised solver.

## 3. Gravity: three formulations

Let `Φ₀` be the background potential (with centrifugal contribution when
rotating) and `g = ∇Φ₀`.

1. **Mixed, Eulerian potential** (WD eqs. 21–22): keep the perturbation
   `φ¹` of the *spatial* potential as an unknown:

   ```
   (1/4πG) ∫_ℝ³ ∇φ¹·∇χ dV = ∫_B ρ ∇χ·u dV      (Poisson row)
   coupling in (M):  ∫_B ρ ⟨∇φ¹ + (∇∇Φ₀)·u, w⟩ dV
   ```

   Constant-coefficient Laplacian (assembled once, DtN on a fixed outer
   sphere), local coupling `∫ρ∇φ¹·w`. This is what
   `LinearQuasiStaticMixedSelfGravitatingProblem` does, with the `∇∇Φ₀`
   term integrated by parts into the symmetrised form.

2. **Referential potential** (Maitra & Al-Attar 2024, eq. 21):
   `ζ = φ∘φ` extended over all space via a diffeomorphic extension of
   the motion; the Poisson operator becomes `(1/4πG)⟨a ∇ζ, ∇χ⟩` with

   ```
   a = J F⁻¹ F⁻ᵀ = J C⁻¹,
   ```

   identical to the pulled-back diffusion operator of the mapping layer
   (`PullbackDiffusionCoefficient`, `TransformedDiffusionIntegrator`).
   Linearised about a natural reference: `ζ¹ = φ¹ + u·g` and
   `δa(u) = (div u)1 − Du − Duᵀ`, so the coupling moves into the Poisson
   row as `(1/4πG)∫⟨δa(u)∇Φ₀, ∇χ⟩` — derivatives shift from `φ¹` onto
   `u` and the background. Same content, different sparsity.

3. **Eliminated, non-local** (AC18 eqs. 99/122): substitute the Green's
   function, giving a dense double integral `γ¹(u)`. A device for
   derivations; as a numerical formulation it produces a dense operator,
   and the library does not use it.

**The referential classes use (2).** It costs only assembly (the
operator is the `TransformedDiffusionIntegrator` machinery), and it
leaves the DtN untouched: the equilibrium mapping is the identity at and
beyond the DtN sphere — the motion of the exterior is gauge — so `a = 1`
there and the outer closure is the plain one. The mapping is *not* the
identity throughout the buffer in general: unless `φ_e` is the identity
on the body's physical surface, its surface values must be tapered
smoothly (and diffeomorphically) to the identity at the buffer's external
boundary. Any smooth extension is admissible — the extension is gauge —
but a rule is needed. Two are provided: `TaperedDiffeomorphism`
(`mappings.hpp`) blends any ball-wide analytic mapping to the identity
through a cubic smoothstep, `C¹` at both seams with `F = I` at the DtN
sphere; `NewHarmonicExtensionMapping` (`background.hpp`, serial and
parallel) interpolates a mapping given on the body and extends it by a
vector Laplace solve in the buffer, returning a
`GridFunctionDiffeomorphism` on the ball.

There is also a physical reason for (2) (Maitra & Al-Attar 2024): the
Eulerian form's gravity coupling contains the *bare displacement*, not a
strain (the `u·∇∇Φ⁰`-type terms sample the reference potential at
displaced positions). If secular motions are excited — true polar
wander, slow rotations, any drift that grows displacement without
growing strain — that linearisation fails on an aspherical reference,
where the sampled potential actually changes; a spherically symmetric
reference is immune only because its potential is invariant under the
drift. The referential form's gravity enters through `Du` and
composition alone, with no bare displacement anywhere, and is
indifferent to secular motion.

### 3.1 The linearised referential system

Write the static part of the action (Maitra & Al-Attar 2024, eq. 21) as

```
V[φ, ζ] = ∫_B W(x, F) dV + ∫_B ρ ζ dV + (1/8πG) ∫_{ℝ³} ⟨a(F) ∇ζ, ∇ζ⟩ dV,
a(F) = J F⁻¹F⁻ᵀ,
```

with the motion extended diffeomorphically to ℝ³ (§3's taper). Stationarity
in ζ is the referential Poisson equation; stationarity in φ the momentum
equation. Perturb `φ = φ_e + u`, `ζ = ζ⁰ + ζ¹` and expand to second
order. With `H_u = F_e⁻¹ Du` (all indices referential; for a vector basis
function `u = φ_a e_k`, `H` is rank-one), the derivatives of `a` are

```
a′(u)   = (tr H_u) a_e − H_u a_e − a_e H_uᵀ,
a″(u,v) = [tr H_u tr H_v − tr(H_u H_v)] a_e
          − tr H_u (H_v a_e + a_e H_vᵀ) − tr H_v (H_u a_e + a_e H_uᵀ)
          + (H_u H_v + H_v H_u) a_e + a_e (H_uᵀH_vᵀ + H_vᵀH_uᵀ)
          + H_u a_e H_vᵀ + H_v a_e H_uᵀ,
```

symmetric in `u ↔ v` as it must be. The `∫ρζ` term is linear in `ζ` and
contributes only to the background equations; `ρ` is never differentiated
anywhere. The linearised system in `(u, ζ¹)`, test functions `(v, χ)`,
with `g₀ = ∇ζ⁰` and the **referential gravity flux** `w = a_e g₀`
(the Nanson-transported background gravity):

```
u-row:  ∫_B ⟨A·Du, Dv⟩ dV                         [material + geometric, §2]
        + (1/8πG) ∫ ⟨a″(u,v) g₀, g₀⟩ dV            [gravity–gravity]
        + (1/4πG) ∫ ⟨a′(v) g₀, ∇ζ¹⟩ dV             [coupling, transpose]
        = ℓ_u(v)

ζ-row:  (1/4πG) ∫ ⟨a_e ∇ζ¹, ∇χ⟩ dV                 [mapped Laplace + DtN]
        + (1/4πG) ∫ ⟨a′(u) g₀, ∇χ⟩ dV              [coupling]
        = ℓ_ζ(χ)
```

The coupling expands to
`⟨a′(u) g₀, ∇χ⟩ = (tr H_u)(w·∇χ) − (H_u w)·∇χ − (H_uᵀ g₀)·(a_e ∇χ)`,
and the gravity–gravity term reduces to products of per-dof scalars:
for the basis function `u = φ_a e_k`, `H` is rank-one and every
contraction in `⟨a″ g₀, g₀⟩` is built from `m_a = F_e⁻ᵀ∇φ_a`,
`∇φ_a·w`, `F_e⁻ᵀg₀` and `F_e⁻¹w` — O(d) work per dof pair, no
fourth-order coefficient ever formed (the assembly pattern of the mapped
elastic recipe). The two gravity terms are `ReferentialGravityIntegrator`
(scale `1/8πG`) and `ReferentialGravityCouplingIntegrator` (scale
`1/4πG`); `ReferentialGravityMatchesFiniteDifference`
(`TestGeneralisedStiffness`) checks them against finite differences of
the exact energy.

Structural remarks:

- **Loads.** A surface mass load `σ` (per referential area) loads the
  potential row alone, `ℓ_ζ(χ) = −∫_∂B σ χ dS`; the Eulerian
  `−∫σ∇Φ₀·v` is absorbed by the change of variables. An applied
  (tidal) potential `ψ` loads the *displacement* row alone,
  `ℓ_u(v) = −∫_B ρ̃ (∇ψ∘φ_e)·v dV` (`SetTidalLoad`): by the chain rule
  `F_e⁻ᵀ∇_X(ψ∘φ_e) = (∇ψ)∘φ_e`, so composing the physical tidal
  gradient with the equilibrium mapping
  (`HarmonicExpansionGradientCoefficient` takes the mapping directly)
  is the same as differentiating the referential tidal potential —
  the "factor of `F⁻ᵀ`" is already folded in. Nothing loads the
  potential row: the mixed class's tidal `−M_F Ψ` term belongs to its
  fluid-density unknown, which this formulation does not carry. The
  centrifugal potential's force fits the same door (its degree-0 `r²`
  part is not an interior harmonic — pass its gradient as plain
  analytic data; `TidalCrossCheck2D` checks exactly this piece), and
  the `u`-dependent rotational feedback `ψ′ = −(Ω×x)·(ω×x)` (Maitra &
  Al-Attar 2024) will enter as an operator term on the same substrate.
  The slipping subclasses refuse the tidal load for now: their fluid
  displacement row carries no tidal force, and a solid-row-only load
  under-drives a fluid core by tens of percent (measured).
- **At `φ_e = id`, hydrostatic**: `H = Du`, `a_e = 1`, `w = g₀ = ∇Φ₀`,
  and the coupling reduces to
  `⟨[(div u)1 − Du − Duᵀ]∇Φ₀, ∇χ⟩/4πG`. The system is related to the
  Eulerian one of the mixed class by the invertible change of variables
  `ζ¹ = φ¹ + u·∇Φ₀`: congruent, not identical. The hydrostatic
  cross-check therefore maps solutions across; it does not compare
  operators entry-wise.
- **No gravity interface terms — but the perturbation needs the
  buffer.** `ρ` is a fixed referential field, so no
  `[∇φ¹·n̂ + 4πGρ u·n̂]`-type jump conditions arise, and second
  derivatives of `Φ₀` never appear. However, the gravity terms of the
  second variation live wherever `∇ζ⁰ ≠ 0` — the buffer included —
  through the *extended* perturbation: exactly as `φ_e` needs a taper
  (§3), the linearised motion `u` must be extended from `∂B` to the DtN
  sphere, and `δa(u_ext)` contributes there. The extension is gauge
  (observables are extension-independent; `ζ¹` itself is
  extension-dependent in the vacuum), but it cannot be omitted:
  truncating the gravity terms at `∂B` is not a gauge choice and produces
  O(1) errors in both `u` and `ζ¹`; a load applied to the potential row
  alone is likewise extension-dependent. Three ways to supply the
  extension:
  (a) carry the displacement on the whole ball, the buffer part a
  pure-gauge field under a small harmonic penalty
  (`SetVacuumExtension`, ball-wide mode). This is accurate only in the
  biased `O(ε)` regime: the Tikhonov refinement that removes the
  analogous bias for the gauged fluid cannot work here, because the
  vacuum has no physical stiffness (only the zeroth-order gravity terms)
  while the harmonic penalty scales like `1/h²`, so the refinement's
  contraction factor tends to one under mesh or order refinement
  (`doc/gauge_penalty_iteration.tex`, section "Iterated refinement",
  subsection "When it must fail: the spectral condition").
  `HydrostaticCrossCheckBallWide2D` exercises this mode.
  (b) a *prescribed* linear extension `u_ext = E(u|_{∂B})` with the
  buffer blocks folded through `E` by sparse triple products: exact (any
  `E` is a gauge choice), no extra unknowns, no near-kernel, no ε. This
  is the library's route: `NewRadialVacuumExtension` builds a radial
  taper `E` that is exact on the trace through the SubMesh dof pairing
  and gradient-free at the DtN sphere, and
  `SetPrescribedVacuumExtension` folds it in (serial and parallel).
  With it the hydrostatic cross-check maps displacement and potential
  onto the mixed class's solution at discretisation level, improving
  with order, and two different tapers agree in observables (a built-in
  gauge-invariance test). In 2-D the comparison must respect the
  constant gauge: `ζ¹` and `φ¹` are both projected orthogonal to
  constants, but `φ¹ + u·∇Φ₀` is not.
  (c) hybrid Eulerian variables outside the body: not implemented.
- **Saddle structure**: symmetric, elastic-positive in `u`, Laplace-type
  in `ζ¹`; the block solvers and projectors of the mixed class carry
  over.
- **Rigid modes have no potential partners.** The referential potential
  `ζ = φ∘φ` is *invariant* under rigid motions of the body (the spatial
  potential co-moves), so the null pairs are `(t, 0)` for translations
  (`Du = 0`: every term vanishes identically) and `(Wφ_e, 0)` for
  rotations (`Du = W F_e`, so `sym(F_eᵀ D u) = sym(F_eᵀ W F_e) = 0`
  identically; the geometric and gravity terms cancel by the rotational
  invariance of the equilibrium energy). `MappedRotation`
  (`null_space.hpp`) supplies `Wφ_e`. Contrast the Eulerian
  formulation's coupled near-null pairs `(u_r, −u_r·∇Φ₀)`: the projector
  machinery simplifies, and rigid near-nullity is not degraded by secular
  drift. `TranslationsAreExactNullPairs` and
  `RotationResidualDecreasesWithOrder` (`TestReferentialProblem`) check
  both.

**The gravitational stress tensor** (Maitra & Al-Attar 2024 write
gravity as a stress `N` alongside `T`, giving momentum-equation terms
`⟨N, Dw⟩ + ∫_∂B ⟨N n̂, w⟩`): equivalent after integration by parts to the
body-force form, at the price of an extra external boundary term. The
library uses the body-force form throughout, which keeps the free
surface natural.

## 4. Boundaries and interfaces

- **Free surface / welded interfaces**: natural conditions of the weak
  form; nothing to assemble.
- **Fluid–solid interfaces**: the exact theory constrains the motion by
  `φ₋ ∘ χ = φ₊` with a slip map `χ ∈ Diff(Σ)`, and the tangential-slip
  constraint on test functions is enforced by a scalar multiplier `ϖ`
  paired with the Nanson vector `J F⁻ᵀ n̂` (Maitra & Al-Attar 2024,
  eqs. 164–171); the traction condition across the interface carries the
  *surface* Jacobian, `t₋ = J_χ t₊∘χ`. Linearised about a general
  equilibrium (AC18 eqs. 108–116; their weak form eq. 137):

  ```
  − ∫_Σ ϖ⁰ ⟨Q·Δu, Dw₁⟩ − ∫_Σ ϖ⁰ ⟨Du₁, Q·Δw⟩ + ∫_Σ ϖ⁰ ⟨S·Δu, Δw⟩ dS,
  with the constraint  ⟨F₁⁻¹ Δu, n̂⟩ = 0  on Σ,   Δu = u₂ − u₁,
  ```

  where `ϖ⁰` is the equilibrium multiplier (minus the equilibrium
  pressure on `Σ`), and `Q`, `S` are equilibrium-geometry operators
  (AC18 eqs. 107/109) built from a level-set function of `Σ`. At a
  natural reference these reduce to the Woodhouse & Dahlen surface
  terms — WD eq. 23's `π⁰`-terms with tangential surface derivatives —
  and for a hydrostatic equilibrium further to the (F2)-type terms of
  `LinearQuasiStaticMixedSelfGravitatingProblem`
  (`doc/self_gravitation.md`). The perturbation `ϖ¹` of the multiplier
  is the Lagrange multiplier of the linearised slip constraint. In the
  referential organisation these terms collapse to a single interface
  form, derived in `doc/slip_interface.tex` and summarised in §5.2.
- **Gravity across interfaces**: `[φ¹] = 0` and `[∇φ¹·n̂ + 4πGρu·n̂] = 0`
  are natural in the mixed weak forms (WD eq. 24); no assembly.

## 5. Fluids, stratification and the gauge

A fluid region has `W = V(x, J)`: bulk response only, `P = −pJF⁻ᵀ`,
`p = −D_J V`. In the linearisation this supplies `C` with
`κ_bare = κ_PREM − p⁰/3` **and** `μ_bare = p⁰` (§2) — the general-form
statement of "a fluid at pressure is not a shear-free solid".

The quasi-static solution in a fluid is defined only up to linearised
relabellings. In a hydrostatic, barotropic fluid the linearised energy is
unchanged by displacements that are divergence-free **and** tangent to
the level surfaces (of the equilibrium potential, pressure and density,
which coincide in hydrostatic equilibrium): these are the relabelling
(gauge) directions. `doc/gauged_fluid.md` (gauge penalty, Tikhonov
refinement, the continuous-space gauge for barotropic fluids) applies
with the general operators of this note, and its discussion of the
Adams–Williamson condition identifies when the hydrostatic short-cuts
remain exact.

**Tangential slip is not gauge in general.** A tangential slip on a
fluid–solid interface can be removed by relabelling only if it extends
into the fluid as a divergence-free field tangent to the level surfaces.
On an interface that is a level surface this restricts the slip: for a
spherical core, rigid rotations of the core are such fields, but a
general tangential slip need not be. On an interface that is not a level
surface (an elliptical interface in a non-hydrostatic model, say) the
level surfaces meet the interface and the construction generally fails.
Hence:

- the slipping-interface formulation (§5.2) is the general one; allowing
  slip that is not needed costs nothing in correctness;
- the welded gauged formulation (§5.1) is used where the slip it
  suppresses is absent or removable by relabelling — as for the
  spherically symmetric models of the benchmarks, where the welded and
  slipping formulations agree to discretisation level.

Which slips are removable in a general geometry is not characterised.

### 5.1 The welded (gauged) case in the referential class

The gauged treatment transfers to the referential class without further
machinery: the fluid's inputs come from the background module (bare
tensor with `μ_b = p⁰` via `BareElasticTensorCoefficient`,
`S_e = −p⁰1`), and the base-class `SetGaugedFluid` penalty and
refinement act on the fluid attributes unchanged. On a mapped problem the
referential class assembles the deviatoric penalty covariantly through
`ElasticTensorIntegrator(C, map)` (see `doc/mappings.md`, "Pitfalls").
Two tests in `TestReferentialProblem` verify it:

- **The joint relabelling-invariance identity**
  (`FluidRelabellingNullPair`): an azimuthal relabelling `w = curl ψ`
  supported inside the core is annihilated by the u-row operator only
  *jointly* — material `μ_b = p⁰` term, geometric `−p⁰` term and gravity
  second variation cancelling through the equilibrium condition. The
  energy along the orbit, normalised by the elastic energy of `w`,
  converges to zero with element order when the bare dictionary is used;
  with the seismological moduli used directly it converges to the finite
  defect `∫2p⁰|sym Dw|²`, nearly two orders of magnitude larger at order
  3. This is the sharpest single test of the §2 dictionary. The full
  block operator (discrete `ζ⁰`) is near-null on `(w, 0)`, the residual
  falling with order.
- **Cross-check against the gauged mixed class**
  (`GaugedFluidCrossCheck2D`): the same two-layer fluid-core problem
  through both formalisms agrees in the mantle displacement and in
  `ζ¹ = φ¹ + u·∇Φ₀` (2-D constant removed) at the discretisation level
  of the coarse test mesh.

### 5.2 The slipping interface

`LinearQuasiStaticReferentialSelfGravitatingSlipProblem` carries a
broken displacement pair `(u_s, u_f)` on solid and fluid SubMeshes,
constrained by `ν·[u] = 0` on `Σ` with the Nanson normal
`ν = cof(F_e)N`. The complete derivation, discretisation, constraint
enforcement (penalty with augmented-Lagrangian iterations, or the
monolithic KKT system of `EnableKKT`), implementation and verification
are in `doc/slip_interface.tex`; the per-class model summary is
`doc/quasi_static_models.tex`, section "The slipping-interface problem".
The results the rest of this note relies on:

- Interface integrals pair the Nanson normal with the referential
  measure; mixing the unit physical normal with the referential measure
  silently loses `|ν|` (the Jacobian warning of `doc/slip_interface.tex`,
  section "Conventions, and one warning").
- The constraint multiplier is the referential pressure, `λ = π`
  (`= p⁰∘φ_e`), and the equilibrium must hand the interface a purely
  normal traction `−π ν` per referential area.
- The AC18 `ϖ⁰`-weighted `Q`/`S` terms collapse to one interface form
  `B_Σ` (multiplier × second-order constraint), in which the interface
  curvature enters and cancels to a trace-only expression;
  `SlipInterfacePressureIntegrator` and `NewSlipInterfaceMatrix` assemble
  it.
- Gravity: in the single-valued organisation `ζ` stays single-valued on
  the ball (composed with a smooth solid-side extension over the fluid,
  `NewRadialFluidExtension`), so the DtN and the potential block are
  untouched, and the slip's gravitational effect is volume-only, through
  the fluid-mismatch coupling — there is no gravity interface term to
  the order kept. In the broken-`ζ` organisation (`EnableBrokenZeta`)
  the potential is composed region-wise, the fluid source is exact, and
  an interface form `G_Σ` with a scalar-jump constraint on `ζ¹` takes the
  mismatch terms' place (`SlipInterfaceGravityIntegrator`,
  `SlipInterfaceGravityScalarIntegrator`,
  `NewSlipGravityInterfaceMatrix`).
- The fluid displacement remains determined only up to relabellings, and
  the fluid gauge penalty with Tikhonov refinement (`SetFluidGauge`) is
  interleaved with the constraint iterations.

## 6. Equilibrium stress in general models

The general form (E) takes `T⁰` as data, but `T⁰` is not free: it must
satisfy `Div T⁰ = ρ∇Φ₀` with continuous tractions, and be hydrostatic in
fluid regions. AW10 organise the solution set completely:

- **Affine structure.** Any two equilibrium stress fields differ by an
  element of `ker(Div₀)` (divergence-free, traction-free); so
  `T⁰ = T_m + Σ aᵢ Sᵢ` with one particular solution and a basis of
  divergence-free fields (constructed via generalised spherical harmonics
  for spherical models, AW10 §4).
- **Particular solutions with desirable properties.** The minimum-norm
  equilibrium stress solves a *linear elastic* boundary-value problem
  (AW10 eqs. 56–57: shear modulus `μ`, `λ = 0`); the
  minimum-**deviatoric** equilibrium stress solves the steady
  *incompressible Stokes* problem (AW10 eqs. 72–74). `background.hpp`
  provides both as `MinimumNormEquilibriumStress` and
  `MinimumDeviatoricEquilibriumStress`: the objects are
  `MatrixCoefficient`s, usable directly as `S_e`, serial and parallel.
  The Stokes solve uses **Taylor–Hood** elements, the pressure one order
  below the velocity: equal-order interpolation violates the inf–sup
  (LBB) condition and produces spurious pressure modes, and the
  constructor refuses it. On a homogeneous ellipse of ellipticity 0.2
  with exact elliptical-cylinder gravity (`examples/equilibrium_stress.cpp`)
  the minimum-deviatoric field carries an unavoidable deviatoric fraction
  of about 0.18 (Love's obstruction); on the disc it falls to the
  discretisation level, the pressure reproducing the hydrostatic `p⁰`.
  The two optimality orderings hold discretely
  (`EquilibriumStressOptimalityOrdering`, `EllipseNeedsDeviatoricStress`,
  `MinimumDeviatoricRecoversHydrostatic` in `TestBackground`).
- **The generators admit relabellings** (mapped mode, optional trailing
  `Diffeomorphism*`): the elastic form pulls back with the standard
  recipe, the divergence coupling becomes `∫ tr(∇u F⁻¹) q J`, the kernel
  becomes translations + `MappedRotation`, and `Eval()` returns the
  second Piola–Kirchhoff pullback `J F⁻¹(T∘φ)F⁻ᵀ` — the `S_e` of the
  general problem, generated on the fixed reference body with `F`
  explicit in every form. Shape updates are then field updates, and
  shape derivatives are analytic. `MappedGeneratorsMatchTransformedMesh`
  verifies the change-of-variables identity (mapped generators on the
  reference disc against unmapped generators on the exactly transformed
  mesh, to about 1e-7).
- **Effect of the deviatoric pre-stress on loading**
  (`examples/prestress_loading.cpp`, test `EllipticalPrestressLoading`).
  On the homogeneous ellipse, loading responses computed with the full
  generated `S_e` and with its pressure part alone (the
  quasi-hydrostatic approximation of standard practice, which is not an
  equilibrium stress off sphericity) differ in displacement by about
  `0.13 e` at `p/μ = 0.31`, scaling like the product `e·(p/μ)`, with the
  potential an order smaller. The effect is negligible at the Earth's
  non-hydrostatic figure, but reaches 1–7 % in displacement for
  `e = 0.1–0.3` (fossil figures, fast rotators) and more where `p/μ` is
  larger. The bare tensor is held fixed in this comparison; holding the
  seismological moduli fixed instead would add the Maitra & Al-Attar
  (2021) conversion difference. Large ellipticities need taper room: the
  `elastogravity_2d_wide` mesh (buffer to radius 2) and the
  parameter-blended ellipse taper `φ = (a(r)x, y/a(r))`, whose exact
  Jacobian `J = 1 + (a′/ar)(x² − y²)` degrades far more slowly than a
  displacement blend.
- Solvability requires the body force and boundary tractions to exert no
  net force or torque (AW10 eq. 58) — the same compatibility the
  projected solvers enforce.
- AW10 eqs. 76–79 further read the deviatoric part of `T⁰` as
  *stress-induced anisotropy* of the seismological tensor (the
  Dahlen & Tromp `Υ = Γ + stress terms` split), making the
  min-deviatoric field also the minimiser of stress-induced anisotropy.
  That reading rests on Dahlen's pre-stress decomposition of the elastic
  tensor, which Maitra & Al-Attar (2021) showed to be incomplete. The
  generators are used here purely as equilibrium-stress constructors;
  nothing relies on eq. 79.

The minimum-deviatoric field is also the object behind a
hydrostatic-figure formulation (a body admits a valid static state
exactly when the minimum-deviatoric field vanishes in its fluid regions);
that solver is not implemented.

### Dahlen's hydrostatic-region argument, and where it stops

AW10 §2.2–2.3 give the modern form of Dahlen's argument, for
perturbations of a *spherical* hydrostatic reference. In a fluid
region the linearised balance `−∇p¹ = ρ⁰∇φ¹ + ρ¹∇Φ₀` is processed in
three moves: (i) cross with `r̂` — using that `∇Φ₀ = g r̂` exactly — to
conclude that `p̂¹ + ρ⁰φ¹` is a function of `r` alone; (ii) normalise
away the degree-0 parts of `ρ¹` and the boundary perturbations by
absorbing them into the spherical reference (their eq. 11), turning
"function of `r` with zero spherical mean" into zero, so the aspherical
pressure is slaved, `p̂¹ = −ρ⁰φ¹`; (iii) curl again to slave the fluid
density pointwise, `ρ¹ = g⁻¹∂_rρ⁰ φ¹` — precisely the `ρ'_F`
coefficient of the Dahlen loading treatment (`doc/self_gravitation.md`).
What remains free is one *constant* pressure perturbation per
connected fluid region (their conclusion (iii)): the degree-0 hole,
quarantined at `l = 0` because spherical harmonics decouple the Poisson
equation degree by degree.

**About a general (aspherical) hydrostatic reference the argument does
not readily extend, and the failure is structural, not technical.**
Step (i) survives in level-surface form: with `ρ⁰ = ρ(Φ₀)` (forced by
hydrostatics), crossing with `n̂ = ∇Φ₀/g` still slaves the *tangential*
structure, `δp + ρ⁰δφ = f(Φ₀)` on each equipotential. But step (ii) has
no analogue: the free data is now a whole function `f(Φ₀)` per fluid
region, and there is neither a symmetry-generated family of reference
models to absorb it into, nor a spectral decomposition in which it
decouples — level-surface averaging does not commute with the Laplacian
on aspherical equipotentials, so the "degree-0-like" free field couples
to *all* harmonics through the Poisson equation. In the spherical case
the free function collapses to a constant via the decoupled `l = 0`
radial balance; aspherically the corresponding reduction is an implicit,
globally coupled problem on the equipotential foliation (hydrostatic
ellipticity theory being its perturbative instance for rotation). The
elimination that defines Dahlen's fluid treatment therefore stops being
an elimination, and the underdetermination that shows up spherically as
the isolated `l = 0` gap is no longer quarantined. This is the
equilibrium-theory counterpart of the loading-problem analysis in
`doc/gauged_fluid.md`: the gauged and referential formulations, which
keep the fluid's displacement and bulk modulus and never perform the
elimination, are indifferent to all of this — there is no privileged
degree anywhere in them.

A corollary for model building (AW10, below their eq. 39): in a
hydrostatic model, lateral density variations in the fluid core are
*slaved* to the potential perturbation generated elsewhere; freely
specified core heterogeneity is inconsistent with a hydrostatic core.

## 7. Implementation map

### Problem classes and terminology

The problem-class names are built from axis slots —
`[Linear|Nonlinear] [QuasiStatic|Dynamic] [Mixed|Referential]?
[SelfGravitating]? [variant] Problem` — with the conventions:

- The elasticity is referential in **every** class; the formulation slot
  names how the *gravity* is described, and so exists only when the
  problem self-gravitates. `Mixed` is the referential–spatial
  organisation (referential displacement, Eulerian potential
  perturbation `φ¹`: Dahlen's arrangement,
  `LinearQuasiStaticMixedSelfGravitatingProblem`); `Referential` is the
  fully referential one (`ζ¹`;
  `LinearQuasiStaticReferentialSelfGravitatingProblem`, slip variant
  appending `Slip`). A non-gravitating problem is referential by
  default and carries no slot
  (`LinearQuasiStaticTractionProblem`/`...ClampedProblem`).
- Reference-state properties are carried by the rheology, not the
  names. Two independent distinctions, in the terminology of Al-Attar &
  Crawford (2016): **hydrostatic** vs non-hydrostatic equilibrium stress
  (`S_e = −p⁰1` vs general), and **natural** vs non-natural particle
  labels (natural: the label is the equilibrium position,
  `φ_e = id`; non-natural: any other labelling, `φ_e ≠ id`). The
  referential classes take the general `(Ĉ, S_e, φ_e)` through
  `ReferentialElasticRheology`; the mixed class *requires* a
  hydrostatic, natural reference state, and it is this restriction, not
  the gravity description alone, that the referential classes lift.

### Terms of the general problem and where they are assembled

| Term | Class / function | Verified by |
|---|---|---|
| material stiffness `Ĉ sym(F_eᵀDu):sym(F_eᵀDv)` (bare tensor, Mandel) | `MaterialStiffnessIntegrator` (equilibrium mapping `φ_e`); `ElasticTensorIntegrator` at `φ_e = id` | `MaterialIdentityMapEqualsElasticTensor`, `MaterialAffineEnergyPatch` |
| PREM → bare conversion `C = C^eff − p⁰(δδ−δδ−δδ)` | `BareElasticTensorCoefficient` | `BareTensorIsotropicConversion`, `FluidRelabellingNullPair` |
| initial-stress (geometric) term `∫ S_e : (DuᵀDv)` | `GeometricStiffnessIntegrator` | `GeometricEnergyPatch`, `GeometricEqualsVectorDiffusionForIsotropicS`, `GeometricPullbackMatchesMappedMesh` |
| constitutive state `(Ĉ, S_e, φ_e)` | `ReferentialElasticRheology` | — |
| hydrostatic background `ρ, g, p⁰, Ĉ, S_e = −p⁰1, φ_e = id` | `RadialHydrostaticState`, `RadialHydrostaticBackground` | `UniformStateMatchesAnalytic`, `StratifiedStateMatchesAnalytic`, `HydrostaticCoefficientsMatchHandRolled` |
| relabelled background (AC18 eqs. 134–136) | `RelabelledBackground`, `RelabelledElasticTensorCoefficient`, `PullbackStressCoefficient` | `RelabelledCoefficientsMatchHandRolled`, `TransformationLawCoefficients`, `RelabelledEquilibrium2D` |
| general aspherical `S_e` (AW10) | `MinimumNormEquilibriumStress`, `MinimumDeviatoricEquilibriumStress` (or any `MatrixCoefficient`; equilibrium consistency is then the caller's responsibility) | `MinimumNormEquilibriumSatisfiesWeakForm`, `MinimumDeviatoricRecoversHydrostatic`, `MappedGeneratorsMatchTransformedMesh` |
| buffer taper of `φ_e` | `TaperedDiffeomorphism`, `NewHarmonicExtensionMapping` | `TaperedDiffeomorphismBlends`, `HarmonicExtensionMapping` |
| gravity, Eulerian (mixed) | `LinearQuasiStaticMixedSelfGravitatingProblem` (`mixed_problem.hpp`) | `doc/self_gravitation.md`, section "Verification" |
| gravity, referential: mapped Poisson block | `TransformedDiffusionIntegrator` + `PoissonDtNOperator` | `doc/mappings.md`, section "Verification" |
| gravity, referential: `a″` and `a′` terms | `ReferentialGravityIntegrator`, `ReferentialGravityCouplingIntegrator` | `ReferentialGravityMatchesFiniteDifference` |
| vacuum extension of `u` | `NewRadialVacuumExtension` + `SetPrescribedVacuumExtension` (option (b)); `SetVacuumExtension` (option (a)) | `HydrostaticCrossCheck2D`, `HydrostaticCrossCheckBallWide2D` |
| rigid modes | translations, `MappedRotation` | `TranslationsAreExactNullPairs`, `RotationResidualDecreasesWithOrder` |
| welded gauged fluid | `SetGaugedFluid` (base class; covariant on mapped problems) | `FluidRelabellingNullPair`, `GaugedFluidCrossCheck2D` |
| slipping interface: constraint, `B_Σ` | `LinearQuasiStaticReferentialSelfGravitatingSlipProblem`, `SlipInterfacePressureIntegrator`, `NewSlipInterfaceMatrix`, `BoundaryNormalNormalIntegrator`/`BoundaryNormalScalarIntegrator` (mapped) | `TestSlipInterface`, `TestSlipProblem` |
| slipping interface: gravity | `NewRadialFluidExtension` (single-valued `ζ`); `EnableBrokenZeta`, `SlipInterfaceGravityIntegrator`, `SlipInterfaceGravityScalarIntegrator`, `NewSlipGravityInterfaceMatrix` (broken `ζ`) | `GravityHessianIdentity`, `BrokenZetaGravityIdentity`, `BrokenZetaHeadToHead` |
| centrifugal term | part of `Φ₀` (absorbed into the background potential) | — |

Test names refer to `tests/TestGeneralisedStiffness.cpp`,
`TestBackground.cpp`, `TestReferentialProblem.cpp`,
`TestSlipInterface.cpp` and `TestSlipProblem.cpp`.

Not part of this linearised quasi-static theory: time dependence and
Coriolis forces, the gravitational-stress-tensor form, finite
deformation, and the stress dependence of the moduli (`Π`).
