# Finite strain about a given equilibrium

> **Where this sits.** The linear problems of the library are
> linearisations about a given equilibrium state. The classes described
> here solve the static problem of a solid body at finite strain about
> the same kind of state, with the linear classes as their tangent:
> without gravity (§1 to §5), and self-gravitating (§6). Fluid regions
> and time dependence are not part of them.

Method notes for `material_law.hpp`, `NonlinearClampedProblem` and
`NonlinearSelfGravitatingProblem` (`nonlinear_problem.hpp`).

## 1. Formulation

The reference configuration is a given equilibrium. The unknown is the
displacement `u` from it, with `F = 1 + grad u` the deformation gradient
and `E = (F^T F - 1) / 2` the Green strain. A material law gives the
strain energy `W(X, E)` per unit reference volume, the second
Piola-Kirchhoff stress `S = dW/dE` and the tensor `dS/dE`; the first
Piola-Kirchhoff stress is `P = F S`. No stress-free state appears: at
`E = 0` the stress is the equilibrium stress of the reference
configuration and the tensor is its second elastic tensor.

The law built is the quadratic one with a prescribed equilibrium stress,

    W = T0 : E + (1/2) E : Xi : E,        S = T0 + Xi : E,

with `T0` the equilibrium stress and `Xi` the second elastic tensor, the
bare tensor (`BareElasticTensorCoefficient` converts seismological moduli
under a hydrostatic stress). It keeps every geometric effect of a finite
deformation, the rotation and convection of the equilibrium stress among
them, and says nothing of how the moduli depend on strain: its
second-order response in the interior is its own and not a material's.

`ThirdOrderIsotropicLaw` adds to another law the isotropic terms of third
order in the strain,
`nu_1 (tr E)^3 / 6 + nu_2 tr E tr E^2 + 4 nu_3 tr E^3 / 3`, the general
such function. Its three constants fix, to first order, how the bulk and
the shear modulus change with pressure. It is there for trying whether
that dependence matters to a result; the library says nothing of what
the constants are. Any other law is a class that returns the energy
density, the stress and the tangent at a strain.

The body is clamped on part of its boundary, with a prescribed
displacement there, and carries dead loads: a traction `t` per unit
reference area on another part and a force `b` per unit reference
volume, fixed in direction and magnitude. The problem is the
stationarity of

    Pi[u] = int_B [W(X, E) - S0 : grad u] dV - int_B b . u dV
            - int_Gt t . u dS,

with `S0` the law's stress at the reference. Its first variation is the
residual,

    r[u](v) = int_B (F S - S0) : grad v dV - int_B b . v dV
              - int_Gt t . v dS,

and its second the tangent,

    K[u](v, w) = int_B [dS/dE sym(F^T grad v) : sym(F^T grad w)
                        + S : (grad v^T grad w)] dV,

the material and geometric stiffness about the current state.

A load may also be given by a stress field `M`,
`int_B M : grad v dV`: the body force `-Div M` with the traction `M N`
on the boundary, in one (`SetStressLoad`).

The term in `S0` makes the residual incremental: without loads the
reference is a solution whatever the equilibrium stress. A given `T0`
balances the body only to the accuracy of the mesh, and without the term
that imbalance would load it.

### A reference that is not the equilibrium

Both problem classes take an equilibrium mapping `phi_e`. The mesh is
then a reference of labels and the equilibrium configuration its image,
with gradient `F_e` and Jacobian `J_e`; the displacement is from that
image, and a configuration has the gradient `F^ = F_e + grad u` from the
mesh. The deformation from equilibrium is `F = F^ F_e^-1`, and the law
about the equilibrium becomes a law about the mesh
(`MappedMaterialLaw`): with `G = F_e^-1`,

    E = G^T E^ G + (G^T G - 1) / 2,    W^ = J_e W(E),
    S^ = J_e G S G^T,

and the tangent likewise. Everything below is then as for the identity,
with `F^` in place of `F`, the reference stress that of the equilibrium
mapping, and densities and loads per unit volume and area of the mesh.
Where the law varies over the body its coefficients are those of a
point's image.

## 2. Discretisation

Symmetric tensors are Mandel vectors in the ordering of
`SymmetricTensorBasis`, in which a contraction is a dot product and the
tensor `dS/dE` a symmetric matrix. A `MaterialLaw` returns, at a point
and from a Green strain, the energy density, the stress and the tangent.
Four coefficients evaluate a law at the deformation of a mapping:
`StrainEnergyDensityCoefficient`, `FirstPiolaKirchhoffStressCoefficient`,
`SecondPiolaKirchhoffStressCoefficient` and `MaterialTangentCoefficient`.
With the mapping the equilibrium mapping displaced by the displacement
(`DisplacedDiffeomorphism`) they follow the state as it changes.

Nothing is assembled that the linear classes do not assemble. The
residual is `DomainLFDeformationGradientIntegrator` with the coefficient
`F S - S0`. The tangent is a `LinearQuasiStaticClampedProblem` owned by
the nonlinear problem, with a `ReferentialElasticRheology` whose tensor
is `dS/dE`, whose stress is `S` and whose mapping is the state:
`MaterialStiffnessIntegrator` and `GeometricStiffnessIntegrator`. The
linear class's solver and preconditioner are the nonlinear problem's.
`LinearQuasiStaticProblemBase::MarkOperatorDirty()` tells it that the
state has moved.

The energy, the residual and the tangent are integrated with one rule on
each element, of order `2 p + w` with `p` the order of the element and
`w` that of the weight of its transformation, so that the residual is the
derivative of the discrete energy and the tangent the derivative of the
discrete residual. The tangent takes that rule where the mesh has one
kind of element; on a mixed mesh it keeps the linear class's own.

## 3. The iteration

`Solve()` starts from the current displacement and repeats: the residual
at the state, one solve with the tangent, the correction added. A
correction takes on the clamped part what the displacement lacks of its
prescribed value, so that from the reference the first iteration is the
linear problem with the boundary data, and later ones vanish there.

The tangent is assembled at the first state and kept while each
correction is at most `NewtonOptions::refresh_contraction` (default 0.1)
of the one before. The first correction that is not is discarded, and
from then on the tangent is assembled at every iteration, which is
Newton's method. A value of zero is Newton's method from the start. The
iteration stops when the L2 norm of a correction is at most `rel_tol`
times that of the displacement.

With the tangent of the reference kept the corrections contract by a
factor proportional to the strain, about ten times it on the test body,
so that at small strain a solve is one assembly and a few solves with
the operator of the linear problem.

There is no line search and no load stepping.

## 4. Verification

`tests/TestMaterialLaw.cpp`: the Green strain and the Mandel round trip;
for the quadratic law and for the law with the terms of third order,
whose tangent depends on the strain, the stress against differences of the
energy and the tangent against differences of the stress; the quadratic
law at zero strain; the four coefficients at a mapping of constant
gradient.

`tests/TestNonlinearClamped.cpp` and `tests/TestNonlinearClampedPar.cpp`,
on a square and a cube clamped below and loaded above, of a material
that varies over the body, with a hydrostatic equilibrium stress that
vanishes on the free boundary and the bare tensor that goes with it,
orders 1 to 3, simplices and tensor elements:

- the residual against fourth-order differences of the energy, and the
  tangent against those of the residual, at a state that is not the
  reference, to 1e-8;
- an affine patch: an affine displacement of strain 0.1 prescribed on
  the whole boundary of a homogeneous body is recovered to 1e-9;
- the first iteration from the reference equals the linear clamped
  problem with the law's tensor and stress there, and the difference
  between the nonlinear solution and that one falls as the load squared;
- with the tangent assembled at every iteration, each correction is
  bounded by a multiple of the square of the one before; with the
  tangent of the reference kept, the contraction doubles with the load;
- without loads the residual and the energy vanish at the reference and
  the solve returns it;
- in parallel, on 1, 2 and 4 ranks, the displacement, the energy and the
  corrections of each iteration equal those of the serial problem.

- a manufactured solution: with `u*` a smooth displacement, the load
  given by the stress `P(F*) - S0` and `u*` prescribed on the boundary,
  the error falls at order + 1 in L2 at strains of a tenth (2.00 and
  2.99 at orders 1 and 2, on simplices and tensor elements); with the
  bottom alone clamped, at strains of a hundredth;
- the same body through another mapping: an interior relabelling of the
  square and the cube gives the displacement of the top and its square
  integral over the body to within what the mesh gives, falling as it
  is refined; and the mapped law is the law at the strain from
  equilibrium, point by point;
- a relabelled body under a stress load in parallel against serial.

`examples/finite_strain.cpp` bends a clamped beam under a ladder of
loads and prints the linear and nonlinear displacements.

## 5. Limits of the clamped problem

- No gravity, no fluid regions, no time dependence.
- Dead loads only. A load that follows the surface, a pressure for one,
  is not built.
- The tangent must be positive definite on the displacements that vanish
  on the clamped part, and the linear solver does not detect that it is
  not. With an equilibrium stress this asks that the stress be the
  equilibrium of the body under its own loads. A stress that pushes on a
  boundary which is free or carries a dead load is held, in the true
  problem, by a load that follows the surface; without that load the
  tangent can lose definiteness and the iteration then fails.
- The equilibrium stress of a hydrostatic state goes with the bare
  tensor. A seismological tensor used in its place gives a body that is
  softer in rotation than the model's.
- The quadratic law's response at second order in the interior is not a
  material's (§1).
- No line search: a first iteration that leaves the neighbourhood of the
  solution is not recovered from.
- A body with free sides under a dead load can lose stability at
  moderate strain: the manufactured solution with the bottom alone
  clamped converges at strains of a hundredth, and at a tenth the
  iteration wanders.

## 6. The self-gravitating body

### Formulation

The body `B` sits in a ball whose outer sphere carries the exterior
potential's Dirichlet-to-Neumann condition, with a buffer shell of vacuum
between. The unknowns are the displacement `u` on the body and the
referential potential `zeta` on the ball: the potential at the place a
point of the reference has moved to (the whole of it, not an increment).
With `a(F) = J F^-1 F^-T`, the problem is the stationarity in both of

    V[u, zeta] = int_B W(X, E) dV + int_B rho zeta dV
                 + (1/8 pi G) int <a(F) grad zeta, grad zeta> dV
                 + int_dB sigma zeta dS,

the third integral over the ball with the exterior's energy on its
sphere (`doc/theory/gravitating_elasticity.md`, §3.1). `rho` is the
referential density, which no deformation changes. Stationarity in
`zeta` is Poisson's equation pulled back to the reference, and is linear
in `zeta`.

The load is a mass `sigma` per unit reference area carried by the points
of the surface. It enters through the potential alone: its weight
reaches the body through the gravity term, and that is exact at finite
strain, the load's pull on itself included.

Other loads: on the displacement a force per unit volume and a load
given by a stress (`SetBodyLoad`), both dead; on the potential's row a
source and a flux (`SetPotentialLoad`), `int q chi + int g . grad chi`
over the ball, on the right of the potential's equation as the surface
mass's `-int sigma chi dS` is.

In the buffer `F` is that of a prescribed extension of the displacement,
`u_ext = E u` (`NewRadialVacuumExtension`), which is the identity on the
outer sphere, so that the exterior condition is untouched. The extension
is a gauge at any amplitude at which it stays invertible.

The residual is incremental. The residual of the reference state
`(0, zeta_0)` without load is subtracted from both rows, `zeta_0` being
the discrete potential of the reference. In the potential's row this
removes the density's source identically. In the displacement's it
removes the imbalance of the given equilibrium stress and gravity on the
mesh. Without load the reference is then the solution.

### Discretisation

The tangent is `LinearQuasiStaticReferentialSelfGravitatingProblem` at
the state: the current configuration its mapping, the law's stress and
tangent its rheology, the current potential its background. Its block
solve gives the corrections of both unknowns, with its preconditioner
and its rigid modes, which are those of the mapped body.

The current configuration is a `DisplacedDiffeomorphism`: the
equilibrium mapping displaced by `u`, held as one field on the body's
SubMesh, one on the buffer's (`E u`) and one on the ball, and served by
whichever the element belongs to.

Nothing of gravity is assembled that the linear class does not assemble.
Its coupling block applied to `zeta` is twice the first variation of the
gravity energy in the displacement, and its potential block applied to
`zeta` the first variation in the potential; the residual reads both
from a linear problem made at the state. For that to be the derivative
of one discrete energy, every gravity form of that problem, the elastic
stiffness, the elastic residual and the energy are integrated with one
rule on each element, of order `2 p + w`
(the linear class takes the rule at construction for the gravity
forms: `a(F)` is a rational function of `F`, no rule integrates the
three exactly on a deformed body, and their own rules differ).

### The iteration

`Solve()` is the iteration of §3 on the pair: a linear problem is made at
each state for the residual; it becomes the tangent at the first
iteration and whenever the kept tangent stops contracting by
`refresh_contraction`. The residual takes the potential's column of that
problem's operator (`ApplyPotentialColumn`): the coupling and the
potential's own block. The displacement's block, the buffer's
gravity-gravity block and the preconditioners are assembled only when
the problem is solved, so a problem made for a residual alone does not
pay for them.

### Verification

`tests/TestNonlinearSelfGravitating.cpp` and
`tests/TestNonlinearSelfGravitatingPar.cpp`, on a homogeneous disc and
ball in hydrostatic equilibrium, with the bare tensor and the stress
`-p0 1`, the quadratic law and the law with the terms of third order:

- without load the residual vanishes at the reference and the solve
  returns it;
- the residual against fourth-order differences of the energy, and the
  tangent against those of the residual, in both unknowns at once, at a
  state that is not the reference, to 1e-7, at orders 1 and 2;
- the first iteration from the reference equals the linear referential
  problem under the same load (1e-8), and the difference between the
  nonlinear solution and that one falls as the load squared;
- with a tangent at every iteration each correction is bounded by a
  multiple of the square of the one before; at a small load the tangent
  of the reference serves throughout and gives the same answer;
- a manufactured solution on the disc: a displacement and a change of
  the potential that vanish towards the surface, with the stress load
  and the flux that make them the solution; the strain converges at the
  order of the space and the potential at one more (0.99 and 1.98 at
  order 1; 1.99 and 2.99 at order 2 over three meshes);
- the same disc through another mapping, which moves the points within
  r < 0.8: the mean radial displacement of the surface agrees to 2e-8,
  and the reference potential and the potential's change on the body
  outside to 1e-6, from 2.5e-5 on the mesh before;
- a ball under a uniform surface mass of 0.05 against the radial
  solution of the same problem (`tests/radial_nonlinear_solid.py`:
  Lagrangian equations with gravity `G m / r^2`, no potential field):
  the surface's displacement to 1.2% and the nonlinear part of it to 2%
  on a coarse mesh at order 2;
- in parallel, on 1, 2 and 4 ranks, the displacement, the potential's
  change and the corrections of each iteration equal those of the
  serial problem, also for a relabelled disc under every kind of
  load.

`examples/finite_strain_gravity.cpp` loads a ball over a ladder of loads
and prints the linear and nonlinear displacements.

### Limits

- A solid body: no fluid regions.
- The rigid motions of the body are taken out of every correction. The
  surface mass and a load of the potential's row inside the body leave
  them free. A dead load on the displacement does not: it is not
  unchanged by a rotation of the body, so that where its answer holds a
  rotation the class returns a state that differs from it by more than
  a rigid motion, at the size of the rotation. A load with a net force
  has no equilibrium at all.
- With an equilibrium mapping, the mapping must be the identity on the
  surface and through the buffer, where the vacuum extension is given in
  the mesh's coordinates.
- A linear problem is made at every state at which the residual is
  wanted, also where a kept tangent then does the solve: its coupling
  and its potential block are assembled as matrices to be applied once.
- A mesh of one element type.
- The buffer has to resolve the extension: the potential there depends
  on it.
- No line search and no load stepping.
