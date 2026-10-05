# Mapped (relabelled) forms

Method notes for the mapping-aware integrator layer: the pull-back of
the elasto-gravity weak forms through a diffeomorphism of the reference
(spherical) domain, the single assembly recipe that implements every
pulled-back term, and the `Diffeomorphism` interface (`mappings.hpp`).
The layer serves the referential problem classes (whose equilibrium
mapping `φ_e` is a `Diffeomorphism`), aspherical models with interface
and surface topography, and the relabelled 3-D benchmarks
(`benchmarks/relabelling/`). Because the mapping appears explicitly in
every form, derivatives with respect to the shape are analytic.

The construction follows the particle-relabelling picture of Al-Attar &
Crawford (2016; `doc/BenchmarkPapers/ggw032.pdf`, with a dictionary of
conventions at the end of Section 3); every formula below is
self-contained (chain rule, change of variables, Nanson's relation) so it
can be checked line by line.

## 1. Setup and conventions

- `M` is the reference body inside the reference computational ball `B`,
  all boundaries and internal interfaces genuine spheres. `ξ : B → B̃` is a
  diffeomorphism onto the physical domain; indices `A, B` label reference
  coordinates `x`, indices `i, j` physical coordinates `ξ`.
- Deformation gradient, Jacobian, right Cauchy–Green tensor of the mapping:

  ```
  F_iA = ∂ξ_i/∂x_A,     J = det F > 0,     C = Fᵀ F.
  ```

- **ξ is the identity on and outside the DtN sphere** (in practice: from
  somewhere inside the buffer outward). The outer boundary is then a
  genuine sphere and `SurfaceHarmonics`, the DtN map and the boundary
  harmonic analysis are used unchanged. Interface and surface topography
  live entirely in ξ; the reference fluid–solid interfaces remain spheres,
  so the submesh coupling is also untouched.
- Fields pull back **componentwise** (no rotation of components; vectors
  keep their ambient Cartesian components):

  ```
  u_i(x) = ũ_i(ξ(x)),     φ(x) = φ̃(ξ(x)).
  ```

- The three transformation rules everything follows from:

  ```
  gradients:   ∂ũ_i/∂ξ_j = (∂u_i/∂x_A) F⁻¹_Aj        (∇̃ = F⁻ᵀ ∇)
  volumes:     dξ = J dx
  surfaces:    m̃ dS̃ = ν dS,  ν = J F⁻ᵀ n             (Nanson)
               i.e.  dS̃ = |ν| dS,   m̃ = ν/|ν|.
  ```

- **Coefficient convention.** Integrators take the *physical* material
  fields expressed in reference coordinates — `ρ` below always means
  `ρ̃ ∘ ξ` — and the geometric factors `J`, `F` appear explicitly in the
  form. Nothing is absorbed into "relabelled" densities. Two consequences:
  with ξ = id the forms reduce term-by-term to the standard ones, and in
  the relabelled benchmark (physical model *defined* as the push-forward of
  a spherical model) the referential coefficients are exactly the radial
  coefficients of the unmapped spherical problem.
- Coefficients that are physically gradients (`∇̃Φ̃₀`) are supplied as the
  referential gradient and mapped by `F⁻ᵀ` (`PullbackGradientCoefficient`).

## 2. The pulled-back forms

Each term of the bilinear form `A` (see `self_gravitation.md`) pulls back
by applying the three rules. Writing `∇` and `div` for reference-domain
operators:

```
Poisson       ∫ ∇̃φ̃·∇̃φ̃′ dξ            = ∫ J C⁻¹ ∇φ·∇φ′ dx
elastic       ∫ c̃_ijkl ∂_jũ_i ∂_lũ′_k dξ = ∫ c′_iAkB ∂_Au_i ∂_Bu′_k dx,
                                          c′_iAkB = J c_ijkl F⁻¹_Aj F⁻¹_Bl
mass / L2     ∫ ρ̃ φ̃ φ̃′ dξ              = ∫ J ρ φ φ′ dx
advective     ∫ ρ̃ ∇̃φ̃·ũ′ dξ            = ∫ J ρ (F⁻ᵀ∇φ)·u′ dx
divergence    d̃iv ũ                     = tr(∇u F⁻¹) = ∂_Au_j F⁻¹_Aj
surface       ∫_Σ̃ f̃ (m̃·ũ)(m̃·ũ′) dS̃   = ∫_Σ f (m̃·u)(m̃·u′) |ν| dS,
                                          m̃ = ν/|ν|, ν = J F⁻ᵀ n
DtN           unchanged (ξ = id on the DtN sphere)
```

The pulled-back elasticity tensor `c′` retains only the **major** symmetry
`c′_iAkB = c′_kBiA` (hyperelasticity survives the pull-back); the minor
symmetries are lost because `iA` mixes a physical with a referential index.
In 3-D that is 45 independent components against the 21 of `c` — the reason
`c′` is never stored as a field, and in fact (Section 3) never formed at
all.

## 3. One assembly recipe

Every domain term above is its standard counterpart with two
substitutions at each quadrature point:

```
dshape  →  dshape · F⁻¹          (shape derivatives w.r.t. ξ, by chain rule)
w       →  J · w                 (volume element)
```

after which the *standard* assembly runs. For the elastic term this means:
compute `dshape · F⁻¹`, then build the usual Mandel strain–displacement
matrix `B` from it and contract with the **referential** tensor `ĉ`
(21 components, `ElasticTensorCoefficient` unchanged). Symmetrising in the
ξ-derivatives is exact because `c̃` has its minor symmetries in physical
indices — so the Mandel machinery is reused as-is and the 45-component
`c′` never appears. Per quadrature point the extra cost is one `d × d`
inversion and one small matrix product; the extra storage over the
spherical problem is the mapping itself.

Boundary terms use Nanson's relation instead: the normal-and-area factor
`m dS` is replaced by `ν dS = J F⁻ᵀ n dS`, with unit normal `ν/|ν|` and
area factor `|ν|` wherever the integrand needs them separately.

For the Poisson term the recipe is `TransformedDiffusionIntegrator`
(`a = J C⁻¹ = J F⁻¹F⁻ᵀ`); the other pull-back integrators of Section 4
apply it in the same way.

### Material symmetry survives the pull-back

If the physical body is isotropic, the pulled-back tensor is not
isotropic in the naive sense, but it takes the closed form

```
c′_iAkB = J [ λ G_iA G_kB + μ ( δ_ik (C⁻¹)_AB + G_kA G_iB ) ],   G = F⁻ᵀ,
```

with λ, μ the referential Lamé fields: it is parametrised entirely by the
two scalars and the mapping. Likewise a physically TI body pulls back to
a form parametrised by its five Love moduli and the referential axis
field ñ∘ξ (components untouched — relabelling does not rotate ambient
components). The recipe of this section computes exactly these
contractions without forming them: material symmetry enters through the
unchanged referential `ElasticTensorCoefficient`, geometry through F, and
storage never exceeds the referential model plus the mapping. Al-Attar &
Crawford (2016) state both versions of this fact: eq. (143) is the
linearised "apparent anisotropy of a specific form" of an isotropic body
relative to a non-natural reference configuration (with the corollary
that relabelling-induced anisotropy cannot produce shear-wave
splitting), and eq. (75) the finite-strain analogue for a modified
Saint Venant–Kirchhoff material.

### Dictionary to Al-Attar & Crawford (2016)

The paper (`doc/BenchmarkPapers/ggw032.pdf`; GJI 205, 575–593) writes the
relabelling ξ : M̃ → M from the *new* reference body to the old, and its
eq. (71) absorbs the Jacobian into the relabelled material parameters:
ρ̃ = J_ξ ρ∘ξ, W̃(x̃, F̃) = J_ξ W[ξ, F̃ F_ξ⁻¹]. The forms here keep J
explicit and take coefficients as the physical fields composed with the
mapping (Section 1), so the paper's relabelled density is `J ρ` in this
note's symbols. The linearised first Piola–Kirchhoff elastic tensor of
its Section 4 carries the same major-only symmetry as c′ above.


## 4. The `Diffeomorphism` interface (`mappings.hpp`)

A mapping is asked for exactly two things at a quadrature point — ξ and F
(J follows from F). The base class derives from `mfem::VectorCoefficient`
with `Eval` returning ξ, so a mapping is directly usable everywhere a
coefficient of position goes (`TransformedFunctionCoefficient`,
`Mesh::Transform`, projection onto a nodal space); it adds `EvalGradient`
returning F, a `Jacobian` helper for det F, `MapNormal` (below) and
`IsIdentity`.

### Concrete mappings

- `IdentityDiffeomorphism` — ξ = x, F = I exactly; `IsIdentity()` is
  true. The equilibrium mapping of every unmapped (natural-reference)
  problem, as a first-class object.
- `CallableDiffeomorphism` — ξ and F from callables of position: analytic
  mappings with exact derivatives.
- `RadialDiffeomorphism` — ξ = f(x)·x with exact `F = f I + x (∇f)ᵀ`,
  from coefficients for f and ∇f, or from a radial profile f(r), f′(r)
  (needing f′(0) = 0 for smoothness at the origin). Both constructors
  require the derivative: the analytic classes are exact by design, and a
  caller with f alone goes through `Interpolate`, which makes explicit
  that the gradient is then a discrete one.
- `TaperedDiffeomorphism(xi, r_inner, r_outer)` — the radial blend
  `x + t(r)(ξ(x) − x)` of a ball-wide mapping to the identity, with the
  cubic smoothstep t: equal to ξ (values and gradient) inside `r_inner`,
  the identity outside `r_outer`, `C¹` across both seams, so `F = I` at
  the DtN sphere. The analytic buffer-taper rule for an equilibrium
  mapping that is non-trivial on the physical surface
  (`doc/gravitating_elasticity.md`, section "Gravity: three formulations").
- `GridFunctionDiffeomorphism` — ξ = x + h for a displacement grid
  function, `F = I + ∇h` through the element transformation. The
  discrete representative: interpolated mappings and mappings supplied
  in discretised form (planetmodel can export these — the same entry
  point). `NewHarmonicExtensionMapping` (`background.hpp`, serial and
  parallel) returns one: it interpolates a mapping on the body and
  extends it into the buffer by a vector Laplace solve with the identity
  on the DtN sphere — the elliptic buffer-taper rule, for mappings known
  only on the body.

### Free functions

Serial, with parallel overloads under `MFEM_USE_MPI`:

- `Interpolate(xi, mesh)` — the nodal interpolant ξ_h of ξ on the mesh's
  geometric space (its nodal collection, order and ordering), returned as
  an owning `GridFunctionDiffeomorphism`.
- `MappedMesh(mesh, xi)` — a copy of the mesh with nodes moved to ξ_h:
  the element-by-element image mesh. Set the reference mesh's curvature
  before calling.
- `MaxIdentityDeviation(xi, mesh, bdr_marker)` — the max of |ξ(x) − x|
  over quadrature points of the marked boundary; drivers wrap it in an
  `MFEM_VERIFY` to assert the identity-outside-the-body convention
  (Section 1), which mappings satisfy by construction and this detects
  the taper that does not quite reach zero.

### Coefficients

- `JacobianCoefficient` (J): multiplies volume densities and sources for
  the stock `MassIntegrator`/`DomainLFIntegrator`.
- `PullbackDiffusionCoefficient` (J C⁻¹): the Poisson tensor.
- `PullbackGradientCoefficient` (F⁻ᵀ v): referential gradients such as
  ∇Φ₀ mapped to physical ones.
- `TransformedFunctionCoefficient`, `TransformedVectorFunctionCoefficient`,
  `TransformedMatrixFunctionCoefficient`: a physical scalar, vector or
  matrix function composed with ξ, i.e. its referential expression.
- `PullbackStressCoefficient`: the relabelling (Piola) transformation of
  a symmetric stress, `J F⁻¹ (S∘ξ) F⁻ᵀ` (AC18 eq. 136; the inner
  coefficient must already be the composed `S∘ξ`).
- `RelabelledElasticTensorCoefficient` (`elastic_tensor.hpp`): the
  relabelling transformation of the second elastic tensor,
  `J Q⁻ᵀ (Ĉ∘ξ) Q⁻¹` with `Q` the Mandel congruence by F (the inner
  coefficient again composed). With `PullbackStressCoefficient` and
  `J ρ∘ξ` it makes up the transformed chain that `RelabelledBackground`
  owns.
- `MappedRotation` (`null_space.hpp`): the rigid rotation of the mapped
  positions, `W ξ(x)` — the strain-free rotational mode of a problem
  posed on a fixed reference body with equilibrium mapping ξ, used by
  the referential classes' rigid-mode projectors and by the equilibrium
  stress generators in mapped mode.

Note the distinction between the last three and the rest: the
relabelling coefficients *transform data* (they describe the same
physical body from another reference), whereas the pull-back recipe of
Section 3 keeps the data physical and puts `J`, `F` in the form.

### Boundary (Nanson) machinery

`Diffeomorphism::MapNormal` returns ν = cof(F)·n for a reference unit
normal n (adjugate-based, no inversion of F); on a boundary
transformation of a `GridFunctionDiffeomorphism` the gradient comes from
the adjacent volume element (`GridFunction::GetVectorGradient` does
this), which is what makes the discrete identity hold on boundary terms —
the mapped mesh's boundary geometry is the trace of its volume geometry.
Built on it: `NansonAreaCoefficient` (|ν| = dS̃/dS, composing physical
surface loads onto the stock boundary linear-form integrators, as
`JacobianCoefficient` does for volume densities) and
`MappedBoundaryNormalDotCoefficient` (m̃·V = (ν·V)/|ν|, the mapped
counterpart of `BoundaryNormalDotCoefficient` for the m̃·∇̃Φ̃₀ factor of
the interface terms). The boundary integrators own exactly the normals of
their own formulas plus the measure: `BoundaryNormalScalarIntegrator`
maps (m·v) dS → (ν·v) dS (Nanson exactly — the |ν| factors cancel), and
`BoundaryNormalNormalIntegrator` maps (m·u)(m·u′) dS →
(ν·u)(ν·u′)/|ν| dS (two unit normals, one measure); the problem layer
remains the only composer of coefficient and integrator.

### Mapping-aware integrators

Every mapping-aware integrator has, beside its standard constructors,
overloaded constructors taking a `Diffeomorphism&` (not owned); a
constructor without one stores no mapping and assembles the standard
form at no extra cost. Coefficients are always referential fields
(Section 1). The pulled-back form is selected by overload rather than by
a parallel family of `Transformed*` classes because the recipe touches
two lines of each element assembly and leaves the coefficient semantics
unchanged.

| Integrator (`bilininteg.hpp`) | Form | Role of the mapping |
|---|---|---|
| `DomainVectorScalarIntegrator` | `(q·v) u` | pull-back |
| `DomainVectorGradScalarIntegrator` | `v·q·∇u` (e.g. the coupling `ρ∇φ·v`) | pull-back |
| `DomainDivVectorScalarIntegrator` | `q (div v) u` | pull-back |
| `DomainDivVectorDivVectorIntegrator` | `q div u div v` | pull-back |
| `DomainVectorGradVectorIntegrator` | `q v·∇(w·u)` (gravity terms) | pull-back |
| `DomainVectorDivVectorIntegrator` | `(q·v) div u` (gravity terms) | pull-back |
| `ElasticTensorIntegrator` | `Ĉ ε(u):ε(v)` (Mandel) | pull-back |
| `GeometricStiffnessIntegrator` | `S : (DuᵀDv)` | pull-back |
| `TransformedDiffusionIntegrator` | `a ∇φ·∇χ`, `a = J C⁻¹` | pull-back (also from a radial scalar, a vector ξ, or `a` directly) |
| `BoundaryNormalScalarIntegrator` | `q p (m·v) dS` (the (F3) coupling) | Nanson |
| `BoundaryNormalNormalIntegrator` | `q (m·u)(m·v) dS` (the (F2) term, slip penalty) | Nanson |
| `SlipInterfacePressureIntegrator` | slip-interface pressure kernel | equilibrium mapping on Σ |
| `SlipInterfaceGravityIntegrator`, `SlipInterfaceGravityScalarIntegrator` | broken-ζ gravity interface kernels | equilibrium mapping on Σ |
| `MaterialStiffnessIntegrator` | `Ĉ sym(F_eᵀDu):sym(F_eᵀDv)` | equilibrium mapping (required) |
| `ReferentialGravityIntegrator` | `⟨a″(u,v) g₀, g₀⟩` | equilibrium mapping (required) |
| `ReferentialGravityCouplingIntegrator` | `⟨a′(u) g₀, ∇χ⟩` | equilibrium mapping (required) |

"Pull-back" integrators assemble the form of Section 2 through the
recipe of Section 3: the mapping is a relabelling and the coefficients
are the physical fields composed with it. The last three take the
*equilibrium mapping* `φ_e` of the linearised referential problem
(`doc/gravitating_elasticity.md`, sections "The linearised quasi-static
problem" and "The linearised referential system"); their forms are not plain
pull-backs, and a relabelling of such a problem composes into `φ_e` and
transforms the coefficients (`RelabelledBackground`) while leaving these
forms unchanged. The three integrators have no identity-only
constructor; `IdentityDiffeomorphism` gives the natural-reference case.

The mapped Poisson pieces other than the stiffness need no dedicated
integrators: `∫ J ρ φ φ′` and `∫ J ρ φ′` are `MassIntegrator` and
`DomainLFIntegrator` with `ρ` multiplied by `JacobianCoefficient`, and
surface loads use `NansonAreaCoefficient` likewise. The equilibrium
stress generators `MinimumNormEquilibriumStress` and
`MinimumDeviatoricEquilibriumStress` (`background.hpp`) take an optional
trailing `Diffeomorphism*` and then solve the pulled-back problem
(`doc/gravitating_elasticity.md`, section "Equilibrium stress in general
models").

## 5. The discrete change-of-variables identity

The layer supports two evaluation modes, used for different purposes:

- **Exact mode** (`CallableDiffeomorphism`, `RadialDiffeomorphism`,
  `TaperedDiffeomorphism` over an analytic mapping): F is analytic at
  each quadrature point — no projection error in the operator. This is
  the production mode and the one for convergence tests: the pulled-back
  solution converges to the mapped exact solution under refinement.

- **Interpolated mode** (`Interpolate` + `GridFunctionDiffeomorphism`):
  F is the gradient of the nodal interpolant ξ_h of ξ on the mesh's
  geometric space. Let the mapped mesh be the reference mesh with nodes
  moved to ξ_h (`MappedMesh`). Then for any element and quadrature point,
  the mapped element map is ξ_h composed with the reference element map,
  so by the chain rule its Jacobian is `F_h · J_ref` and its weight
  `J_h · |J_ref|` — precisely the recipe of Section 3 with `F = F_h`.
  Assembling the standard integrators on the mapped mesh and the mapped
  integrators on the reference mesh with ξ_h therefore produces the
  **same element matrices to machine precision**, provided the same
  integration rule is passed to both (the default rules must not be
  relied on, since they need not coincide). This is the
  change-of-variables identity as a unit test: no physics, no reference
  solution, every mapped integrator checked against its standard
  counterpart on an arbitrarily deformed mesh.

The distinction matters: with *exact* F the two discrete systems are not
identical — they differ by the geometric interpolation error of ξ_h — so
exact-mode agreement with the mapped mesh is a convergence statement, not
an identity. Machine precision belongs to the interpolated mode; superior
accuracy belongs to the exact mode.

These identity tests sit alongside the integrators' other unit tests
(known-solution actions, symmetry, patch and covariance tests) and are
part of the verification chain in the same sense: each mapped operator is
certified at the element-matrix level, so the verification runs
continuously from element matrices to Love numbers.

## 6. Verification and pitfalls

### Verification

Element and operator level (`tests/`):

- `TestMappings`: F exact for the analytic mappings
  (`RadialGradientsAreExact`), the factorisation of `MappedMesh` through
  `Interpolate` at geometric orders 1–3 in 2-D and 3-D
  (`InterpolateMatchesMappedMesh`, `GridFunctionMapReproducesPolynomial`),
  and `MaxIdentityDeviation`.
- `TestTransformedDiffusionIntegrator`: the Poisson identity against
  `DiffusionIntegrator` on the mapped mesh (`MatchesMappedMesh`,
  `NonAffineMatchesMappedMesh`, `MatchesMappedMeshSolve`), agreement of
  the four ways of specifying the mapping (`MappingPathsAgree`), and a
  mapped solve converging to a mapped exact solution
  (`ExactMappingConverges`).
- `TestElasticTensorIntegrator` (`ElasticTensorIntegratorMapped`),
  `TestMappedDomainIntegrators` (the six generalised domain integrators)
  and `TestMappedBoundaryIntegrators` (the (F3) coupling, the composed
  (F2) term, and the Nanson load compositions): the identity of
  Section 5 for every pull-back integrator.
- `TestGeneralisedStiffness`: `GeometricPullbackMatchesMappedMesh`, and
  finite-difference checks of the equilibrium-mapping integrators.
- `TestBackground`: `TaperedDiffeomorphismBlends`,
  `HarmonicExtensionMapping`, `RelabelledCoefficientsMatchHandRolled`,
  `MappedGeneratorsMatchTransformedMesh`.
- `TestReferentialProblem`: `TransformationLawCoefficients` and
  `RelabelledEquilibrium2D` (a relabelled problem reproduces the
  unrelabelled solution under composition).

Solver level: the relabelling benchmark family
(`benchmarks/relabelling/README.md`; documented in `doc/benchmarks.tex`,
section "The relabelling family") describes the same spherical physical
problem from laterally relabelled coordinates. `love_benchmark -map`
(exact F) must reproduce the unmapped Love numbers; `relabelled_identity`
(interpolated F) applies the identity of Section 5 to the full coupled
solve, mapped assembly on the reference mesh against standard assembly
on the image mesh; `aspherical_reference` solves on an independently
generated aspherical mesh (`meshes/aspherical_body.py`). On every welded
case, the gauged fluid included, the solve-level identity holds to about
1e-6, limited only by the DtN centring on the shifted mesh centroid. This
certifies the whole mapped chain at once: stiffness, equilibrium stress,
gravity couplings, Poisson block, loads, extensions and projectors.
Through a slipping interface the identity is not met to that level (the
slip-interface terms are not certified covariant). What is verified: the
broken-ζ organisation on a relabelled two-layer background whose map
twists across the interface (`TestSlipProblem.BrokenZetaRelabelledEquilibrium`,
agreement with the identity description to 5e-3, the fixed mesh's
geometric-interpolation floor), and the radial-stretch aspherical
fluid-core meshes of `examples/slipping_interface.cpp`, where mapped
broken ζ matches the spherical solution and the welded problem's leakage.
Mapped slipping-interface results beyond these cases should be treated as
unverified.

### Pitfalls

- **Every term of a mapped problem must be assembled covariantly,
  including gauge-fixing terms.** A gauge subspace has stiffness `O(ε)`,
  so a non-covariant `O(ε)` penalty difference produces `O(1)`
  differences in the gauge representative. The fluid gauge penalty of a
  mapped problem is therefore the pull-back of the unmapped one.
- **The covariant gauge penalty uses `ElasticTensorIntegrator(C, map)`**
  with the deviatoric tensor, not `MaterialStiffnessIntegrator`. The
  latter's mapping is an equilibrium mapping, and its form is the plain
  pull-back only when the tensor is the *relabelled* one; given the
  unrelabelled tensor it assembles a different form.
- **Identity tests need identical integrator classes and rules on both
  sides.** When both sides of an identity assemble the same term, both
  use the same integrator class (the identity-map case included) and the
  same integration rule, so that the quadrature coincides.
- **Gauge data must be shared, not rebuilt.** The two sides of a
  solve-level identity must share one vacuum-extension matrix: the
  builder (`NewRadialVacuumExtension`) samples the body side of the
  surface, which differs at `O(A)` between a mesh and its image, and the
  extension is gauge data, not a mapped quantity. The harmonic (vacuum)
  gauge penalty likewise refuses a map.
- **A nodal-interpolant mapping lives on one mesh.** A
  `GridFunctionDiffeomorphism` can be evaluated only on element
  transformations of the mesh its grid function lives on; evaluating it
  through a SubMesh's transformations gives wrong values without an
  error. Problems that assemble on SubMeshes of the parent need the
  interpolant transplanted onto every SubMesh (the benchmark's
  `MultiMeshDiffeomorphism` in `benchmarks/common/relabelling.hpp` does
  this); analytic mappings have no such restriction.
