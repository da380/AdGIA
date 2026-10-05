# MFEM facts and pitfalls

MFEM behaviours (checked against MFEM 4.9) that the library relies on or
works around. Each entry says where the library deals with it; the subject
notes (`submesh_coupling.md`, `null_space.md`, `viscoelasticity.md`) give
the context.

## Solvers

- **The relative tolerance of the Krylov solvers is relative to the
  *initial* residual**, not to ‖b‖. A warm-started solve that begins near
  the solution therefore chases a target far below round-off and runs to its
  iteration limit. Set an absolute tolerance from the preconditioned norm of b,
  rel_tol·√(M b, b), whenever `iterative_mode` is on
  (`LinearQuasiStaticProblemBase::SetWarmStartTolerance`).
- **CG's stopping criterion is on the squared residual** (the preconditioned
  inner product), so a relative tolerance below about 1e-13 is beyond
  round-off. The inner potential solves floor theirs there.
- **An unprojected preconditioner on a projected singular system diverges
  at round-off.** BoomerAMG or a shifted Laplacian amplifies the round-off
  null component of the residual at every iteration. Project the
  preconditioner as well as the operator (`null_space.md`). It is observed in
  parallel; it is not observed with serial Gauss–Seidel.
- **`OrthoSolver` is not a projected operator.** It runs the inner solver on
  `A` with a projected right-hand side. When the constant is only
  approximately null (the 2-D potential block with a fluid mass term) that
  is a different regularisation from CG on `P A P`, and the two give
  different answers. The library uses `P A P` throughout; only
  `examples/poisson_dtn.cpp` uses `OrthoSolver`, for the singular 2-D case
  of a stand-alone Poisson problem.
- **Setting the operator of an `IterativeSolver` also sets it on its
  preconditioner.** When the two must differ, set the operator first and the
  preconditioner second (`SetupCG`).
- **BoomerAMG copes with a disconnected domain.** Iteration counts on a
  `ParSubMesh` made of inner core and mantle match those on the mantle
  alone, for the Laplacian and for elasticity with the elasticity options.
- **`SDIRK23Solver()` defaults to the third-order, A-stable but not
  L-stable variant.** The second-order L-stable scheme is
  `SDIRK23Solver(2)`, which the examples and benchmarks use.
- **`ImplicitSolve(dt, x, k)` must return the rate** `k` with
  `k = f(x + dt·k)`, not the new state, and SDIRK schemes call it with γ·dt.
- **Explicit `ODESolver`s call `SetTime(t + cᵢ dt)` before each `Mult`.** Use
  `GetTime()` inside `Mult`, not a stored time.

## Forms and assembly

- **`MixedBilinearForm::Assemble` needs both spaces on one mesh**: trial and
  test are indexed by the same element number. Coupling a mesh to its
  SubMesh is what `submesh.hpp` exists for.
- **`TransferMap` is a signed vdof permutation**, with the map private. The
  public route to the same map is `SubMeshUtils::BuildVdofToVdofMap`.
- **`MixedBilinearForm::Assemble` is not virtual.** A derived class can only
  hide it, as `DiscreteLinearOperator` does; call through the derived type.
- **The borrowing constructor `MixedBilinearForm(tr, te, mbf)`** shares the
  domain, boundary, trace-face and boundary-trace-face integrators and their
  markers, but omits the boundary-face and interior-face lists. It copies
  marker *pointers*, so markers must outlive every form that borrows them.
- **Reassembling with a changed coefficient: build a fresh form.**
  `BilinearForm::Update()` only zeroes the matrix in place, and
  `Finalize(skip_zeros)` may have dropped exact zeros from the pattern that
  the new coefficient needs.
- **Markers are sized to `attributes.Max()` of the mesh the form integrates
  over**; for a SubMesh form, the SubMesh's. On a `ParSubMesh` the attribute
  lists are globally reduced, so a rank with no submesh elements still sizes
  its markers correctly.
- **`ElasticityIntegrator(coef, q_λ, q_μ)`** sets λ = q_λ·coef and
  μ = q_μ·coef, which gives the κ div–div and 2μ dev–dev parts separately
  with no custom integrator.
- **`GradientIntegrator::AssembleElementMatrix2` sizes the Jacobian
  adjugate by the reference dimension** and corrupts memory on manifold
  (boundary-submesh) elements. A scalar–vector coupling *on* a surface mesh
  needs its own integrator.
- **`FormLinearSystem` in serial legacy assembly returns an `X` that aliases
  the grid function**, and `RecoverFEMSolution` relies on that aliasing.
  After solving in a separate `BlockVector`, copy the block back explicitly
  instead of calling `RecoverFEMSolution`.

## Linear algebra

- **The dense products `Mult`, `MultABt`, `MultAAt`, `AddMult_a_AtB` do not
  resize their output.** The size check is an `MFEM_ASSERT`, which vanishes
  in release builds, so an unsized output silently yields an empty result.
  Size every output explicitly.

## Meshes and SubMeshes

- **`SubMesh::CreateFromDomain` works for a disconnected set of
  attributes**, although its documentation asks for a connected subset.
  Nothing in the implementation uses connectivity: elements are copied
  attribute by attribute, the vdof map is element-based, and the shared
  groups of a `ParSubMesh` are per entity. Transfer, the dof injection and
  forms with boundary integrators on both interfaces behave identically at
  every rank count. The library relies on this to hold all solid regions in
  one displacement space.
- **SubMesh boundary attributes** are inherited where the parent had
  boundary elements, and are `max(parent bdr attr) + 1` on a boundary cut
  through the parent's interior. Interior interfaces a form must address
  should therefore be physical surfaces in the parent mesh.
- **`CalcOrtho` of a boundary transformation is the outward normal** of the
  mesh the boundary element belongs to, on inherited and cut SubMesh
  boundaries alike.
- **A `ParSubMesh` keeps the parent's partition**; there is no
  rebalancing. The displacement problem is balanced only as well as the body
  is spread over ranks, which is one reason to keep the buffer shell thin.
- **A rigid rotation on a curved element is representable only when the
  geometry order does not exceed the field order** (`null_space.md`).

## Parallel

- **`ParFiniteElementSpace::GlobalTrueVSize()` is collective.** Calling it
  inside a root-only block hangs.
- **`GetGlobalTDofNumber(ldof)` is valid for local dofs the rank does not
  own** (conforming spaces), which makes the true-dof injection buildable
  without communication (`submesh_coupling.md`).
