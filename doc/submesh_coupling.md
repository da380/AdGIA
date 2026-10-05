# Coupling forms across a mesh and its SubMesh

The self-gravitating problem couples a displacement on the solid body `M` (a
`SubMesh`) to a potential on the enclosing ball `B ⊃ M` (the parent mesh),
through integrals over `M`, or over its boundary, in which one of the two
fields lives on the parent. MFEM has no form for this: `MixedBilinearForm`
indexes trial and test by the same element number, so both spaces must be on
one mesh, and `SubMesh::Transfer` moves fields but gives no operator that a
Krylov method could see. `submesh.hpp` fills the gap.

## Principle

Assemble on the submesh with MFEM's own `MixedBilinearForm`, between the
submesh space and a *shadow* of the parent space (the parent's
`FiniteElementCollection` placed on the submesh). Then re-index one side of
the resulting sparse matrix from shadow vdofs to parent vdofs. The integrand
only ever sees one mesh, so every integrator that `MixedBilinearForm`
accepts works unchanged, including boundary integrators on the submesh
boundary (the fluid–solid interface terms). Nothing else is custom.

| Layer | Object | Responsibility |
|---|---|---|
| A | `SubMeshDofInjection` | the signed vdof map shadow → parent; vector transfer; row and column re-indexing of sparse matrices; the true-dof matrix Π in parallel |
| A′ | `NewSubMeshPairingMatrix`, `NewSubMeshPairingTrueDofMatrix` | the signed pairing of two sibling submesh spaces through their injections into one parent |
| B | `SubMeshMixedBilinearForm`, `ParSubMeshMixedBilinearForm` | a `MixedBilinearForm` whose two spaces live on a mesh and its SubMesh; `Assemble()` is a helper form on the submesh followed by the re-indexing; everything else is inherited |

## The injection

The map comes from MFEM's `SubMeshUtils::BuildVdofToVdofMap` (the map inside
`TransferMap` is private), decoded into a parent vdof and a sign per shadow
vdof. It requires identical per-element dof layouts on the two sides, which
sharing the collection *object* guarantees; `MakeShadowSpace()` builds such a
space with the parent's vdim and ordering. Sharing the collection also keeps
`SubMesh::Transfer` usable on the shadow. The construction is element-local,
so it works unchanged on `ParFiniteElementSpace`s: a `ParSubMesh` inherits
the parent's partition, so the parent of every local submesh element is
local. Domain and boundary (`CreateFromBoundary`) submeshes are both
supported; for the latter the shadow is the trace space.

As an operator `P` (parent vsize × sub vsize, entries ±1, one per column):
`P` extends by zero, `Pᵀ` is the exact restriction, `PᵀP = I`, and `PPᵀ` is
the indicator of the parent dofs lying in the submesh. `RemapRows(M)` and
`RemapColumns(M)` equal `P M` and `M Pᵀ` but are O(nnz) re-indexings that
preserve the sparsity pattern; no sparse product is formed.

### The true-dof matrix Π in parallel

Block operators and Schur complements act on true dofs, so the parallel path
needs Π : sub true dofs → parent true dofs as a `HypreParMatrix`.

The product `Π = R_parent · P_loc · P_sub` is **wrong**. A
shared parent dof on the submesh boundary may be owned by a rank whose local
elements there all lie outside the submesh. `R_parent` selects only owned
parent dofs, and on the owning rank the corresponding row of `P_loc` is
empty, so the entry is silently lost. This is not exotic: a 4×4 quad mesh
with the submesh `x > 0.5`, slab-partitioned on two ranks at order 2,
already fails.

Instead Πᵀ is built row by row over *owned sub* true dofs. The owner of a sub
true dof always holds a submesh element containing it, and
`ParFiniteElementSpace::GetGlobalTDofNumber(parent_ldof)` returns the correct
global parent true dof even for parent dofs the rank does not own (conforming
spaces), so no ownership case is missed:

```
for each sub ldof l with local true dof lt >= 0:
    J[lt]    = parent_pfes.GetGlobalTDofNumber(parent_vdof[l])
    data[lt] = sign[l]
Πᵀ = HypreParMatrix(one entry per row, sub tdof offsets, parent tdof offsets)
Π  = transpose(Πᵀ)
```

Π is a boolean injection, so Πᵀ is at once the exact primal restriction and
the correct dual prolongation; the two uses never need separate operators.

## Pairing two sibling submeshes

Where the two sides of an interface carry *separate* spaces, each on its own
SubMesh of one parent, the two sides must be identified on the shared
interface. Two injections into the same parent space give that pairing
directly:

```
J = P_aᵀ P_b        (serial, vdofs:      NewSubMeshPairingMatrix(a, b))
J = Π_aᵀ Π_b        (parallel, true dofs: NewSubMeshPairingTrueDofMatrix(a, b))
```

`J_ij = s_i s_j` when the i-th dof of `a` and the j-th dof of `b` map to the
same parent dof (signs from the two injections), and zero otherwise. For two
domain SubMeshes meeting along an interface the shared parent dofs are
exactly the interface dofs, so `J` identifies the two traces nodally: for
fields of equal trace, `(J u_b)_i = (u_a)_i` on the interface dofs of `a`.
The pairing is one-to-one, so `J Jᵀ` is the identity on the shared dofs of
`a`. Both injections must target the same parent space (checked by
`MFEM_VERIFY`). In parallel the product is formed by hypre from the two
`NewTrueDofMatrix()` operators, so the two sides of a shared parent dof may
live on different ranks.

Its use is that an interface form need be assembled only once: with `B` an
interface bilinear form assembled on `a`'s boundary elements, the same form
on `b`'s trace is `Jᵀ B J` and the cross terms are `B J`, and the orientation
bookkeeping is the injections'.

## The form

`Assemble()` builds a helper `MixedBilinearForm` between the shadow and the
submesh-side space with MFEM's borrowing constructor, which shares the
integrators and markers without owning them; assembles and finalizes it; and
replaces the form's matrix by the re-indexed one. `SpMat()`, `Mult`,
elimination and `FormRectangularSystemMatrix` are inherited and correct,
because they read only the matrix and the two real spaces.

The parallel class uses the same serial helper on the `ParFiniteElementSpace`s
(assembly is element-local); the inherited `ParallelAssemble()` then forms
`P_testᵀ · mat · P_trial` with each real space's own prolongation. Ranks
without submesh elements contribute an empty local matrix of the right size.
The two classes share the shadow space and injection
(`detail::SubMeshFormSetup`, built from the trial/test pair) and the
assembly routine (`detail::AssembleOnSubMesh`).

Things to know when using it:

- `MixedBilinearForm::Assemble` is not virtual, so the derived `Assemble`
  hides it (as `DiscreteLinearOperator` does in MFEM). Call it through the
  derived type. It *replaces* the matrix rather than accumulating.
- Markers refer to the *submesh's* attributes and are sized against its
  `attributes.Max()` / `bdr_attributes.Max()`. Domain attributes are
  inherited from the parent. Boundary attributes are inherited where the
  parent had a boundary element, and equal `max(parent bdr attr) + 1` where
  the submesh was cut out of the parent's interior. In parallel the
  attribute lists are the globally reduced ones, so ranks without submesh
  elements pass the size check.
- MFEM's borrowing constructor does not copy boundary-face integrators; they
  are handed to the helper explicitly. Interior-face integrators are refused:
  `ParallelAssemble` takes a face-neighbour path for them that a re-indexed
  matrix would break.
- Refused by `MFEM_VERIFY`: nonconforming parents, variable order, spaces
  other than H1 and L2 (orientation-dependent dof transformations of
  H(curl)/H(div) on the submesh need not agree with the parent's), and any
  assembly level other than `LEGACY`. After mesh refinement rebuild the
  object.

## What it is used for

| Coupling | Spaces | How |
|---|---|---|
| ∫_M ρ ∇φ·v | u on the solid submesh, φ on the parent | domain integrator, shadow of φ |
| fluid–solid interface −∫_Σ ρ_F φ (m·v) | the same; Σ part of the submesh boundary | boundary integrator (`BoundaryNormalScalarIntegrator`) with an interface marker |
| loads and fields on a surface | a `CreateFromBoundary` submesh | domain integrator on the surface mesh; the shadow is the trace space (supported and tested; no library class uses it) |

Welded solid regions need no coupling to each other: a `SubMesh` may be
disconnected, so one displacement space covers every solid region (see
`self_gravitation.md`, section "Discretisation"). Separate spaces, and with
them the pairing, are needed where the two sides of an interface are
distinct unknowns or where one space is extended into another:

| Pairing | Spaces | Used by |
|---|---|---|
| solid × fluid displacement on the slipping interface Σ | solid and fluid SubMeshes | `LinearQuasiStaticReferentialSelfGravitatingSlipProblem` (the broken pair `(u_s, u_f)`; see `slip_interface.tex`) and `examples/sliding_fluid_ellipse.cpp` |
| buffer × body displacement on ∂B | buffer shell and body SubMeshes | `NewRadialVacuumExtension` (trace rows of the vacuum extension, `referential_problem.hpp`) |
| fluid × solid displacement on Σ | fluid and solid SubMeshes | `NewRadialFluidExtension` (trace rows of the fluid extension) |
| potential shadows on the solid × outer/fluid regions | shadows of the ζ space | `EnableBrokenZeta` of the slip problem |

The forms themselves couple the potential on the ball to the displacement on
the solid in `LinearQuasiStaticMixedSelfGravitatingProblem` (the ∫ρ∇φ·v
block) and `LinearQuasiStaticReferentialSelfGravitatingProblem` and its slip
variant (the ζ–displacement coupling, and the buffer and fluid couplings).

`examples/submesh_injection.cpp` tours the injection, and
`examples/coupled_poisson.cpp` solves a coupled pair of Poisson problems
monolithically with the form; each is one source for the serial and parallel
builds.
