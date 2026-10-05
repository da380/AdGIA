# Elastic tensors: convention, coefficients and integrators

Notes for `elastic_tensor.hpp` and the `ElasticTensorIntegrator` (with its
total-Lagrangian companions `MaterialStiffnessIntegrator` and
`GeometricStiffnessIntegrator`) of `bilininteg.hpp`. MFEM's
`ElasticityIntegrator` is isotropic (two scalar coefficients) and nothing
in MFEM handles a fourth-order stiffness tensor.
The library adds one integrator that takes the tensor as a matrix
coefficient, and a family of coefficients that produce that matrix in one
fixed convention.

## Convention: a Mandel matrix in the library's component order

A symmetric second-order tensor in `d` dimensions has `n_s = d(d+1)/2`
components. The library orders them by `SymmetricComponentOrder::Offset`
(`index.hpp`), lower triangle, column-major:

```
d = 3:  (11, 21, 31, 22, 32, 33)  ->  0 … 5
d = 2:  (11, 21, 22)              ->  0 … 2
```

The strain interpolators and the viscoelastic internal variables use this
order, so the elastic tensor uses it too; otherwise applying `C` to an
internal variable would need a permutation at every node. It is *not* Voigt
order (11, 22, 33, 23, 13, 12).

The scaling is **Mandel** (an orthonormal reduced basis), not Voigt's
engineering strain. With `a_s = 1` for diagonal components and `√2` for
shear ones,

```
ε̂_s = a_s ε_jk,   σ̂_s = a_s σ_jk,   ε : σ = Σ_s ε̂_s σ̂_s,   Ĉ_st = a_s a_t C_(jk)(lm).
```

What this buys:

- Ĉ is a genuine symmetric matrix representing ε ↦ σ in an orthonormal
  basis. Its eigenvalues are the eigen-stiffnesses, positive-definiteness is
  a plain matrix property, and a rotation `R` acts as `Q Ĉ Qᵀ` with `Q`
  orthogonal (`SymmetricTensorBasis::RotationMatrix`).
- The strain energy is `½ ε̂ᵀ Ĉ ε̂` with no factors of 2 or 4, and the element
  matrix is `Bᵀ Ĉ B` with a `B` that produces ε̂ directly.
- Isotropy is `Ĉ = λ 1̂1̂ᵀ + 2μ I`, and the bulk/deviatoric split is the pair
  of orthogonal projectors `P_vol = 1̂1̂ᵀ/d`, `P_dev = I − P_vol`, which is
  what viscoelastic relaxation needs.

Voigt is what geophysicists write (Love's A, C, F, L, N; tomographic C_ij),
so Voigt is an **input** convention with explicit conversions
(`SymmetricTensorBasis::FromVoigt` / `ToVoigt`, `Pack` / `Unpack` for the
full C_ijkl). Mandel in library order is the **internal** convention that
every coefficient emits and the integrator consumes.

`SymmetricTensorBasis` collects the helpers of this convention:

| Helper | What it does |
|---|---|
| `Size(d)`, `Index(d, j, k)`, `Component(d, s, j, k)`, `Scale(j, k)` | n_s, the reduced index of (j, k) in either order, its inverse, and the Mandel scale a_s |
| `VoigtIndex`, `FromVoigt`, `ToVoigt` | Voigt numbering and the Voigt ↔ Mandel-in-library-order conversions |
| `Pack`, `Unpack` | Mandel matrix ↔ full C_ijkl (stored with index ((i d + j) d + k) d + l) |
| `Apply(Ĉ, ε, σ)` | σ_jk = C_jklm ε_lm for *unscaled* tensor-component vectors in library order |
| `VolumetricProjector`, `DeviatoricProjector` | P_vol = 1̂1̂ᵀ/d and P_dev = I − P_vol |
| `RotationMatrix(d, R, Q)` | the orthogonal Mandel matrix of ε ↦ R ε Rᵀ |
| `CongruenceMatrix(d, F, Q)` | the Mandel matrix of ε ↦ Fᵀ ε F for a general invertible F (`RotationMatrix` is the orthogonal case R = Fᵀ); used by the relabelled tensor below |

## Coefficients

All derive from `ElasticTensorCoefficient`, an `mfem::MatrixCoefficient` of
size n_s × n_s whose `Eval` returns a Mandel matrix in library order and
which records the space dimension (`SpaceDim()`). They therefore compose
with MFEM's algebra: `MatrixSumCoefficient`,
`ScalarMatrixProductCoefficient`, `PWMatrixCoefficient`.

- **`IsotropicElasticTensorCoefficient`**: from (λ, μ) or (κ, μ). It provides
  the isotropic limit for tests and lets the anisotropic path serve
  everywhere.
- **`TransverselyIsotropicElasticTensorCoefficient`**: Love's A, C, F, L, N
  and a symmetry-axis field `n`, or PREM-style velocities and η,

  ```
  C_ijkl = (A − 2N) δ_ij δ_kl + N (δ_ik δ_jl + δ_il δ_jk)
         + (F − A + 2N)(δ_ij n_k n_l + n_i n_j δ_kl)
         + (L − N)(δ_ik n_j n_l + δ_il n_j n_k + δ_jk n_i n_l + δ_jl n_i n_k)
         + (A + C − 2F − 4L) n_i n_j n_k n_l .
  ```

  For n = e₃ this is the canonical Voigt matrix (C₁₁ = A, C₃₃ = C, C₁₃ = F,
  C₄₄ = L, C₆₆ = N, C₁₂ = A − 2N). The axis is normalised inside `Eval`.
  `RadialUnitVectorCoefficient` gives radial anisotropy. The formula is
  written with δ and n only, so it is dimension-generic: in 2-D with an
  in-plane axis it yields the **plane-strain** restriction of the 3-D
  tensor. Plane stress is not supported.
- **`VoigtElasticTensorCoefficient`**: general anisotropy from any MFEM
  matrix coefficient in Voigt form and order.
- **`RotatedElasticTensorCoefficient`**: a tensor given in a material frame
  together with a rotation field, applied as `Q Ĉ Qᵀ`.
- **`DeviatoricProjectionElasticTensorCoefficient`**: `P_dev C P_dev` or its
  complement. For an isotropic `C` these are exactly 2μ dev–dev and
  κ div–div. Together with a TI coefficient holding only L and N, these are
  the two ready-made choices of "the part that relaxes" for an anisotropic
  Maxwell body (`viscoelasticity.md`).

Two further coefficients serve the referential (general initial-stress)
formulation; the physics is in `gravitating_elasticity.md` (section "The
linearised quasi-static problem", subsection "The elastic tensor
dictionary") and `mappings.md`:

- **`BareElasticTensorCoefficient`**: the bare (strain-energy) tensor from
  the seismological effective one under hydrostatic pre-stress p⁰,

  ```
  C_ijkl = C^eff_ijkl − p⁰ (δ_ij δ_kl − δ_il δ_jk − δ_ik δ_jl),
  Ĉ = Ĉ^eff − p⁰ 1̂1̂ᵀ + 2 p⁰ I        (Mandel),
  ```

  the inverse of Woodhouse & Deuss (2007, eq. 61). Tabulated (PREM) moduli
  are components of C^eff, while the referential form assembles the bare C;
  the decorator keeps model data seismological and makes the conversion
  impossible to forget. Isotropically λ = λ^eff − p⁰, μ = μ^eff + p⁰,
  κ = κ^eff − p⁰/3, so a fluid has bare shear modulus p⁰. The conversion is
  three-dimensional physics; in 2-D the same Mandel formula is applied
  formally.
- **`RelabelledElasticTensorCoefficient`**: the transformation of the
  second elastic tensor under a relabelling ξ (Al-Attar et al. 2018,
  eq. 136),

  ```
  Ĉ~(x~) = J_ξ Q_ξ⁻ᵀ Ĉ(ξ(x~)) Q_ξ⁻¹,
  ```

  with Q_ξ the Mandel congruence by F_ξ (`CongruenceMatrix`). The inner
  coefficient must already be the referential expression Ĉ∘ξ (composed
  analytic data, or a constant); the class supplies only the algebra. This
  is the tensor `MaterialStiffnessIntegrator` expects when a relabelling has
  been composed into the equilibrium mapping.

`examples/referential_elastogravity.cpp` uses the bare tensor, and
`benchmarks/relabelling/aspherical_reference.cpp` chains the two (bare,
then relabelled).

## The integrators

`ElasticTensorIntegrator(MatrixCoefficient& C)` assembles ∫ ε(v) : C : ε(u)
on a vector H1 space in byNODES layout. It accepts any `MatrixCoefficient`
of size n_s, so sums and products of the classes above work; the convention
is then the caller's responsibility, and the classes above guarantee it. Per
quadrature point, with `g` the physical shape gradients (dof × d) and
columns indexed by (component c, dof i):

```
diagonal row s(j,j):     B[s, (j,i)] = g(i,j)
shear row s(j,k), j > k: B[s, (j,i)] = g(i,k)/√2,   B[s, (k,i)] = g(i,j)/√2
elmat += w · Bᵀ Ĉ B
```

so that `ε̂ = B u` by construction. The default quadrature is MFEM's
`2·OrderGrad(el)`; a rapidly varying `C` may want more through the usual
`IntRule`.

### The pull-back constructor

`ElasticTensorIntegrator(MatrixCoefficient& C, Diffeomorphism& map)`
assembles the pull-back of the same form through a mapping (`mappings.md`):
the shape gradients become derivatives with respect to the mapped
coordinates (`gshape → gshape F⁻¹`), the weight gains the Jacobian
(`w → J w`), and the assembly is otherwise unchanged. The coefficient stays
the *referential* tensor (21 components in 3-D); the pulled-back tensor
`c'_iAkB = J c_ijkl F⁻¹_Aj F⁻¹_Bl`, which has only the major symmetry, is
never formed. With the mapping interpolated on the mesh's geometric space
and one integration rule on both sides, this equals the plain integrator on
the mapped mesh: that is the discrete change-of-variables identity
described in `mappings.md`.

### Material and geometric stiffness

For the total-Lagrangian split about an equilibrium mapping φ_e
(`gravitating_elasticity.md`, section "The linearised quasi-static
problem"):

- `MaterialStiffnessIntegrator(C, φ_e)` assembles
  ∫_B ⟨Ĉ sym(F_eᵀ Du)^, sym(F_eᵀ Dv)^⟩ dV, the second elastic tensor acting
  on the linearised Green strain, with no Jacobian factor (the strain
  energy is per referential volume). With the identity mapping it is
  `ElasticTensorIntegrator`. Its `Diffeomorphism` is the *equilibrium
  mapping*, not a relabelling pull-back: a relabelling composes into the
  mapping and transforms the coefficients (Al-Attar et al. 2018,
  eqs. 134–136), so the tensor passed must already be the relabelled one.
- `GeometricStiffnessIntegrator(S [, map])` assembles
  ∫_B S_AB ∂_A u_k ∂_B v_k dV with S the second Piola–Kirchhoff equilibrium
  stress (S = T⁰ at a natural reference). Its optional mapping is the
  relabelling pull-back, as for `ElasticTensorIntegrator`.

The two mapped constructors are therefore not interchangeable. The
covariant gauge penalty of a gauged fluid
(`LinearQuasiStaticProblemBase::SetGaugedFluid` with a non-identity map,
and the fluid penalty of the slip problem) is assembled with
`ElasticTensorIntegrator(C_dev, map)`, where `C_dev` is the isotropic
deviatoric tensor of the penalty (λ = −2εμ_g/d, μ = εμ_g): this form is the
exact pull-back of the unmapped penalty, so a relabelled problem's gauge is
the pull-back of the unmapped one. `MaterialStiffnessIntegrator` would be
wrong here because it expects a relabelled tensor and carries no Jacobian.
The distinction matters because the gauge subspace has stiffness O(ε): an
O(εA) difference in the penalty between a problem and its relabelling of
amplitude A moves the gauge representative by O(A), not by the
discretisation error. Where both sides of an identity assemble the
same term, both use the same integrator class (the identity map included),
so the quadrature defaults coincide.

### Internal-variable coupling

The internal-variable coupling ∫ (C_k m) : ε(v) does not go through this
integrator. It uses the unit-coefficient strain integrators with `C_k`
applied pointwise at the internal-variable nodes, which is why the
component orders must match.

## How it is tested

The isotropic tensor reproduces `mfem::ElasticityIntegrator` to round-off on
curved meshes. A TI tensor whose moduli are isotropic equals the isotropic
class for a random axis. Rotating the axis agrees with rotating the tensor,
and rotating mesh and axis together rotates the element matrix. A linear
displacement field gives the exact energy `|Ω| ½ ε̂ᵀ Ĉ ε̂`. Rigid modes lie in
the null space for a radially anisotropic material, on isoparametric
geometry (`null_space.md` explains why). A plane-strain sheet matches an
extruded slab with u_z = 0 (`tests/TestElasticTensor.cpp`,
`tests/TestElasticTensorIntegrator.cpp`). The mapped integrator on the
reference mesh equals the plain integrator on the mapped mesh, for a
spatially varying isotropic and a constant TI tensor, in 2-D and 3-D at
orders 1–3 (`ElasticTensorIntegratorMapped.MatchesMappedMesh`).
`tests/TestGeneralisedStiffness.cpp` checks the bare-tensor isotropic
conversion, that the material stiffness with the identity map equals
`ElasticTensorIntegrator`, an affine energy patch test for it, and the
geometric stiffness (equal to `VectorDiffusionIntegrator` for isotropic S,
energy patch, pull-back against the mapped mesh).

## References

- Al-Attar, D., Crawford, O., Valentine, A. P. and Trampert, J. (2018).
  Hamilton's principle and normal mode coupling in an aspherical planet with
  a fluid core. *Geophysical Journal International*, 214(1), 485–507.
- Woodhouse, J. H. and Deuss, A. (2007). Theory and observations — Earth's
  free oscillations. In *Treatise on Geophysics*, vol. 1 (Seismology and
  Structure of the Earth), ed. G. Schubert, pp. 31–65. Elsevier.
