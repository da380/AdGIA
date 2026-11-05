# Manufactured solutions

The method of manufactured solutions for the static self-gravitating
elastic problem on a solid body (`LinearQuasiStaticMixedSelfGravitatingProblem`):
an exact pair (u*, φ*) is chosen, the loads that make it the solution of
the class's weak form are derived from it symbolically, and the discrete
solution is compared with the exact one. Unlike the spherical Love-number
ladders, whose reference is another code's answer to the same physics,
this measures the discretisation alone against a closed form, on any mesh
and with variable coefficients, and gives the convergence rates in h and
in the element order directly.

## The problem

`manufacture.py` (sympy) holds the chosen fields and the derivation, and
writes `fields.hpp`, the generated C++ header that both the driver and the
test `tests/TestManufactured.cpp` read. Its docstring states the weak form
and the sources that follow from it:

- displacement row: the body force f = −div σ + ∇(ρ u·g) − div(ρu) g + ρ∇φ
  (the symmetrised weak form's own Euler–Lagrange form carries the gravity
  terms with halves and the natural traction σn + ½ρ(u·n)g − ½ρ(u·g)n;
  they coincide with the standard form and with T = σn on a hydrostatic
  background, and the derivation works from the weak form either way);
- potential row: the volume source q = −Δφ/4πG − div(ρu) in the body and
  the surface source s = (∂ₙφ_in − ∂ₙφ_out)/4πG + ρu·n on the surface.

Choices that keep the test clean:

- The potential is exactly harmonic in the buffer, a sum of exterior
  multipoles about the centre, so the buffer carries no source and the
  DtN condition at the outer sphere holds exactly whatever the buffer's
  radius. Inside the body it is the interior continuation of the
  multipoles' traces plus (a² − r²)w(x), continuous across the surface
  with a kink that the surface source carries.
- The background is the self-consistent potential of the homogeneous
  ball (quadratic inside), so the body is in hydrostatic equilibrium and
  the rigid modes are exact null pairs of the continuous problem, as the
  solver's projection assumes. The exact displacement carries a rigid
  component; the driver fits the rigid part of u_h − u* on the discrete
  rigid modes and removes it, and removes the null pair's partner
  −r·∇Φ₀ from the potential (and in 2-D the constant the class projects
  out).
- The moduli vary smoothly (κ = 1 + 0.2xy, μ = 0.5 + 0.1z²), the
  displacement is trigonometric and exponential, so no element order
  represents the solution exactly.

Units are the non-dimensional ones of the tests (unit radius, G = 0.05,
ρ = 1; ρga/μ ≈ 0.4 at the surface).

## Running

```
cd <build>/benchmarks/elastic/manufactured
./run --dim 2 --h 0.1 0.05 0.025 --order 1 2 3 --np 2
./run --dim 3 --h 0.3 0.2 0.15 --order 1 2 3 --np 4
```

`run` builds the meshes with `meshes/homogeneous_body.py` (order-2
geometry, buffer to 1.5 radii, element size 2h in the far field) into
`runs/meshes/`, runs `manufactured_benchmark` on each, and prints the
table of relative L2 errors of u, of the stress σ(u) and of φ with the
observed rates (`runs/d<dim>/table.txt`). Expected on a smooth solution:
p + 1 for u and φ, p for the stress, down to the floor set by the order-2
geometry and the solver tolerance. The driver alone:

```
mpiexec -np 4 ../bin/manufactured_benchmark -m ../../data/ball_buffer_coarse_3d.msh -o 2 -out o2.json
```

To change the fields, edit `manufacture.py` and regenerate the header
with `./manufacture` (the launcher) or the repository's Python
environment; the test's thresholds record the measured errors on the
canned meshes and will need re-reading if the fields change.

## What it covers, and what it does not

Covered: the mixed class's elastic form with variable isotropic moduli,
the symmetrised gravity terms, the potential coupling, the whole-space
Poisson block with the DtN closure, the surface and body loads of both
rows, the rigid-mode projection and gauge, serial and parallel, on
curved meshes in two and three dimensions.

Not covered: fluid regions (a non-neutral fluid's response is determined
by the closure and cannot be manufactured; a neutral fluid admits the
welded description and could be added), the referential class and the
mapped (aspherical) machinery (its weak form differs, and an aspherical
hydrostatic background needs rotation: a separate derivation), and the
viscoelastic stepping.

## Findings

Meshes from `homogeneous_body.py` (order-2 geometry, buffer to 1.5 radii,
far size 2h), MINRES to 1e-11, relative L2 errors of u, of the stress
σ(u) and of φ over the ball, with the rates between consecutive sizes.

**Order 1 does not converge on curved geometry** (2-D: u 4.9e-2, 4.6e-2,
4.5e-2 at h 0.1, 0.05, 0.025; 3-D slowly, rates 1.4 and 0.7): the rigid
rotations of an order-2 geometry are not in the order-1 space
(`doc/self_gravitation.md`, "Element order on curved meshes"). Use a
displacement order at least that of the geometry.

**The background gravity limits the displacement to O(h^p).** The class
projects Φ₀ onto the potential space and differentiates it; the gradient
error O(h^p) enters the gravity terms and the displacement's L2 rate is
p, not p + 1 (2-D order 2: 2.1; order 3: 3.0). With the potential space
one order above the displacement's (`-ophi`, `run --phi-offset 1`) the
rate is p + 1 (order 2: 3.0). The potential itself converges at p + 1
either way. Recommendation: potential order p + 1 whenever the
displacement's rates matter (it costs little: the potential block is a
scalar Laplacian).

**The gravity integrator's nodal interpolant limits the energy norm to
second order whatever the element order.** `DomainVectorGradVectorIntegrator`
differentiates the nodal interpolant of u·∇Φ₀ on the trial space; in the
symmetrised form the interpolant is also applied to the (oscillatory)
test functions, where its error is O(h²) at any order, so the stress
converges at rate 2 at orders 2, 3 and 4 alike (2-D, potential one order
up: 1.9, 2.0, 2.1; the displacement is one or two orders better by
duality). The zero-density problem is clean (order 3: u 3.9, stress 3.0),
and so is the gravitating one once the integrator uses the product rule
with the Hessian of Φ₀ (`SetBackgroundHessian`, driver `-exact-hessian`;
or the Hessian of the discrete Φ₀ itself, `UseDiscreteBackgroundHessian`,
driver `-discrete-hessian`, identical results with the potential one
order up):

```
2-D, potential order p+1, Hessian route      2-D, same, default interpolant
order  h      u_l2 rate  stress rate         order  h      u_l2 rate  stress rate
  2  0.025  3.14e-07 2.98  4.69e-05 1.87        2  0.025  3.14e-07 2.98  4.69e-05 1.87
  3  0.025  1.04e-09 4.00  1.78e-07 3.00        3  0.025  7.22e-09 3.05  1.49e-06 2.05
  4  0.025  9.76e-10 3.66  1.83e-09 3.32        4  0.025  1.05e-09 3.61  1.42e-07 2.09
```

Order 4 at h 0.025 meets the order-2 geometry's floor (errors ~1e-9).
In 3-D (h 0.3 to 0.1, order 3, potential order 4): exact Hessian u 4.05,
stress 3.09 at the last step; interpolant 3.77 and 2.38, the stress error
1.8 times larger at h 0.1. The 3-D ladder is pre-asymptotic at these
sizes.

With the discrete Hessian at the SAME order the stress is optimal (order
3: 3.0) and the displacement is held at p by the gradient (3.0); at
order 2 nothing changes (second order was already optimal).

**The class default is the discrete Hessian.** Costs on the 3-D h 0.15
mesh (4 ranks): interpolant 2.1 s
/ 10.0 s at orders 2 / 3; discrete Hessian 2.4 s / 10.8 s, same
iteration counts; with the potential one order up 4.2 s / 18.3 s, the
potential unknowns tripling. The driver's `-interpolant` (`run
--interpolant`) selects the interpolant; the potential order is the
drivers' `-ophi`.

**Consequence for the benchmark programme.** With the interpolant every
gravitating solve of the mixed class is second-order accurate in the
energy norm; surface observables (Love numbers) gain the duality orders
(about 20× from order 2 to 3 at h 0.3, consistent with an L2-like rate
of 3 rather than 4). Results computed with the interpolant move at the
discretisation level under the default (fluid_core at h 0.3, order 3:
h'_0 from 2.1e-5 to 1.4e-5; the l ≥ 1 entries, mesh-dominated, unchanged).

Tables of every run: `runs_facet/d2_g2*/table.txt`, `d3_g2*/table.txt`
in the build tree (the ladder script writes `table.txt` beside each
set's results).
