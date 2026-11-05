# Null spaces, projection and gauge

Notes for `null_space.hpp`: the rigid-mode coefficients (`RigidTranslation`,
`RigidRotation`, `MappedRotation`), the basis `NullSpaceProjector` with
`AddRigidModes()` and `MakeRigidModeProjector()`, and `ProjectedOperator` and
`ProjectedSolver`. Their users are the pure traction problem
(`LinearQuasiStaticTractionProblem::RigidModes()`), the mixed and referential
self-gravitating problems (displacement modes and their block versions), and
the background-state generator problems of `background.hpp`.

## Two different projections

A free body under traction, or a self-gravitating body, has the rigid
motions as null vectors of its symmetric operator `A`. Two operations are
involved, and they use different inner products.

- **Range compatibility.** Because `A` is symmetric, range(A) = null(A)^⊥ in
  the *Euclidean* true-dof inner product. Projecting the right-hand side with
  the Euclidean projector `P = I − Σ nᵢnᵢᵀ` is what makes CG or MINRES
  well posed. For physically consistent loads (no net force or torque) it is
  a no-op up to round-off.
- **Solution gauge.** Which representative `u + (a + b×x)` to return is a
  separate choice. The default is the Euclidean one, `nᵢᵀx = 0`. The
  physically meaningful one is zero net momentum and angular momentum, which
  is orthogonality in the ρ-weighted L² inner product, `nᵢᵀ M x = 0` with
  `M` the vector mass matrix.

`ProjectedSolver::SetGauge(M)` applies the second *after* a solve done with
the first: the null-space component of the solution is replaced so that
`nᵢᵀ M x = 0` for every basis vector of nonzero `M`-norm, while vectors of
zero `M`-norm (the 2-D constant potential in a block system) keep the
Euclidean condition. The inner solve is untouched on purpose: an `M`-weighted
projector would make `P A P` non-symmetric, and the gauge is only a choice of
representative. The strains, and every rigid-motion-invariant quantity, are
the same in both gauges.

## Solve the projected system, not the projected data

`ProjectedSolver` hands its inner Krylov solver `P A P`, projects the
right-hand side and the warm-start guess before the solve, and projects the
solution after it. Projecting only the data and the solution while iterating
on the unprojected `A` is weaker, for three reasons.

- **Discrete rigid modes are only near-null vectors** when the geometry is
  curved or the problem is coupled to gravity; see `self_gravitation.md` for
  the measured residuals. With `P A P` the iteration never sees that
  direction; with `A` it does, and the result is sensitive to it.
- **The preconditioner must be projected as well** (`P M P`, a second
  `ProjectedSolver` around the preconditioner). An unprojected BoomerAMG or
  shifted-Laplacian preconditioner amplifies the round-off null component of
  the residual at every iteration, and CG diverges once the residual reaches
  round-off. This is observed in parallel; it is not observed with serial
  Gauss–Seidel. `SetupCG()` sets the operator before the preconditioner so
  that the latter is not reset onto the projected operator.
- **A warm start may carry a rigid component**, which the unprojected
  iteration would keep.

`NullSpaceProjector::Add` orthonormalises by modified Gram–Schmidt and drops
a vector that is numerically dependent on those already present, so a caller
may add modes without checking for duplicates (region rotations on top of
global ones, for instance).

## Rigid modes of a mapped reference body

When a problem is posed on a fixed reference body with a non-natural
reference state, the rheology carries an equilibrium mapping φ_e
(`Rheology::EquilibriumMapping()`, `viscoelasticity.md`) and the stiffness
acts on the linearised Green strain sym(F_eᵀ Du). A plain rotation u = W x
(W skew) is then not strain-free, since F_eᵀ W is not skew in general. The
rotation of the *mapped* positions, u = W φ_e(x), is: Du = W F_e and
F_eᵀ W F_e is skew. `MappedRotation(map, component)` is this vector
coefficient (in 2-D the single in-plane rotation, component 2); at the
identity map it equals `RigidRotation`. Translations are unchanged and
are exact null vectors of the stiffness; with a pre-stressed reference
state the mapped rotations are null through the moment balance of the
background state, and discretely near-null, improving with refinement
(`LinearQuasiStaticTractionProblem::RigidPairResiduals()` measures both).

`AddRigidModes(P, fes, map)` appends the translations and the rotations (the
mapped ones when `map` is non-null) of a displacement space of vdim 2 or 3
to `P` as true-dof vectors and returns how many were added (modes already
spanned are dropped). `MakeRigidModeProjector(fes, map)` returns a projector
holding exactly these modes, on the space's communicator in parallel. The
traction problem passes its rheology's equilibrium mapping (null for a
natural reference state), the referential problems build the same modes from
the reference rheology's mapping, and the background generator problems use
the mapped rotations of the pulled-back problem.

## Element order on curved meshes

A rigid rotation of a curved element is representable only when the geometry
order does not exceed the displacement order. With order-1 displacements on
an order-2 mesh the rotations are not in the space: the discrete rigid modes
are poor null vectors and results shift by far more than the
discretisation error; with matched orders they do not.
Use a displacement order at least that of the geometry. The integrator tests
that check rigid modes in the null space use isoparametric geometry for the
same reason.
