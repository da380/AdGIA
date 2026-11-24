# Finite strain about a given equilibrium: the solid without gravity

> **Where this sits.** The linear problems of the library are
> linearisations about a given equilibrium state. The classes described
> here solve the static problem of a solid body at finite strain about
> the same kind of state, with the linear classes as their tangent.
> Gravity, fluid regions and time dependence are not part of them.

Method notes for `material_law.hpp` and `NonlinearClampedProblem`
(`nonlinear_problem.hpp`).

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

The term in `S0` makes the residual incremental: without loads the
reference is a solution whatever the equilibrium stress. A given `T0`
balances the body only to the accuracy of the mesh, and without the term
that imbalance would load it.

## 2. Discretisation

Symmetric tensors are Mandel vectors in the ordering of
`SymmetricTensorBasis`, in which a contraction is a dot product and the
tensor `dS/dE` a symmetric matrix. A `MaterialLaw` returns, at a point
and from a Green strain, the energy density, the stress and the tangent.
Four coefficients evaluate a law at the deformation of a mapping:
`StrainEnergyDensityCoefficient`, `FirstPiolaKirchhoffStressCoefficient`,
`SecondPiolaKirchhoffStressCoefficient` and `MaterialTangentCoefficient`.
With the mapping a `GridFunctionDiffeomorphism` of the displacement they
follow the state as it changes.

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
for the quadratic law and for a law with a cubic volumetric term, whose
tangent depends on the strain, the stress against differences of the
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

`examples/finite_strain.cpp` bends a clamped beam under a ladder of
loads and prints the linear and nonlinear displacements.

## 5. Limits

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
