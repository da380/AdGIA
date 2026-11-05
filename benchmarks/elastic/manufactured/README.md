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

Not covered here: fluid regions (their closed problem, the mixed form
with the level border, is the separate family below, with its own
driver and test), the referential class and the mapped (aspherical) machinery (its
weak form differs, and an aspherical hydrostatic background needs
rotation: a separate derivation), and the viscoelastic stepping.

## The fluid-core family

`manufacture_fluid.py` writes `fields_fluid.hpp` (namespace
`manufactured_fluid::d3`): the exact fields of a two-layer ball with a
closed fluid core, and the loads that make them the solution of the
bordered weak form of the Dahlen closure as the class builds it (the
mixed form with (F1)–(F3), plus the level border's rows in its unknowns
U, s, ν on the level interval and the wall multiplier λ;
`doc/theory/dahlen_closure.tex`, "Implementation"). The geometry is that of
`data/earth_fluidcore_3d.msh` (CMB at 0.5467, buffer to 1.2); the core
is stably stratified, ρ = ρ_c(1 − βr²) with κ_f = 10 (D < 0 throughout,
N² > 0 away from the centre); the shell has the solid family's moduli
and displacement plus a uniform dilation, so that the CMB flux S[u] is
nonzero. The script's docstring states the weak form and every source:

- displacement row: the shell's body force and surface traction as
  above; the CMB traction adds the buoyancy (F2), the potential stress
  (F3) and the wall pressure −λ(m·ĝ)m;
- potential row: −Δφ/4πG − div(ρu) in the shell, −Δφ/4πG + ρ′φ + ν(Φ₀)
  in the core, and the flux sources on the surface and on the CMB, the
  latter with the density jump (ρ_s − ρ_F) u·m;
- border rows, densities against dc: r_U = −(κU′/A + ρs)′ + ν with the
  point source κU′/A + ρ_F s + λ on the wall node, r_s = ρU′ − ρ′As + Aν,
  r_ν = U + sA + Φ̄, and U(c_w) − S[u] on the multiplier's row.

The level functions are closed form: Φ₀ is a quadratic in r² in the
core, so r(c) is explicit, and the core potential is a radial part plus
harmonic polynomials of degree ≥ 1 plus (b² − r²) times an odd function,
so its level integral is Φ̄ = A a₀. The border functions are chosen
freely (polynomials in r²), so every row carries a source. Three
dimensions only: the class's two-dimensional removal of the potential's
constant is not a null-space fact once a fluid is present
(`doc/methods/self_gravitation.md`, section 5), and no field can match it.

`./manufacture_fluid --check` evaluates both sides of every row's weak
identity by quadrature, with random polynomial tests on the ball and
random hat-function tests on random level grids, the fluid volume
integrals assembled as the class assembles them: the largest relative
residual is 4e-14. `./manufacture_fluid --galerkin` solves the border
rows alone by P1 Galerkin with exact level integrals (the exact
potential's level integral and wall flux as data): the one-dimensional
grid's own accuracy (below).

### Driver, test and the placement of the sources

`manufactured_fluid_benchmark` (parallel) and `tests/TestManufacturedFluid.cpp`
(serial, on `data/earth_fluidcore_3d.msh`) share `fluid_case.hpp`, which
sets the problem up, applies every source, solves and measures. The mesh
is a two-layer ball from `meshes/layered_earth.py --layers 2 --dim 3`
(attributes 1 core, 2 shell, 3 buffer; boundary attributes 1 CMB,
2 surface). The displacement lives on the shell's SubMesh (attribute 2),
which keeps the boundary attributes 1 (CMB) and 2 (surface). The class
gets the analytic Φ₀ (`phi0`, by radius), the shell's density, and the
core as a `FluidRegion` (attribute 1; `rho_core`, `rho_prime_fluid`,
`kappa_fluid`; interface marker the SubMesh's attribute 1); the level
border is `SetLevelBorder(levels, exact, grad_phi0)` (`-levels N`,
`-exact-levels` / `-projected-levels`); the gravity term is assembled as
in the solid driver (the class default, the Hessian of the discrete Φ₀;
`-exact-hessian` with the shell's analytic Hessian; `-interpolant`).

| row | source of `fields_fluid.hpp` | where | how |
|---|---|---|---|
| u | `body_force` | the shell | `VectorDomainLFIntegrator`, `AddForce` |
| u | `traction_surface` | SubMesh boundary 2 | `VectorBoundaryLFIntegrator`, n ≈ +x/r |
| u | `traction_cmb` | SubMesh boundary 1 | `VectorBoundaryLFIntegrator`, n ≈ −x/r |
| φ | `potential_source_core` | parent attribute 1 | `DomainLFIntegrator`, `AddPotentialForce` |
| φ | `potential_source_shell` | parent attribute 2 | `DomainLFIntegrator` (nothing on 3) |
| φ | `surface_source` | parent boundary 2 | `BoundaryLFIntegrator`, n ≈ +x/r |
| φ | `cmb_source` | parent boundary 1 (interior) | `BoundaryLFIntegrator`, n ≈ −x/r |
| U, s, ν, λ | border data | the border's grid | `AddLevelBorderForce` after `AssembleForce` |

The surface data take the facet's own unit normal (`CalcOrtho`), oriented
by the sign of n·x so that it is the solid's outward normal: on the
canned mesh the SubMesh's boundary elements already are (0 of 516 CMB
and 0 of 1700 surface facets flipped), the parent's CMB elements all
point away from the centre (516 of 516), against the sources'
convention. `-sphere-normal` takes ±x/|x| instead; on the canned mesh at
order 2 the errors agree to every printed digit.

The border data, on the class's own grid (`LevelBorderGrid`):
r_U[i] = ∫N_i′(κU*′/A + ρs*) dc + ∫N_i ν* dc + λ* N_i(c_w) (the weak
residual; `-strong-border`: ∫N_i r_U dc plus the wall's point source,
the same data), r_s[i] = ∫N_i r_s dc, r_ν[i] = ∫N_i r_ν dc, and
U*(c_w) − S[u*] on the wall row. They are integrated over the exact core
[c_min, c_w] with the hats clamped beyond the grid's end nodes (the class
locates its fluid quadrature points so, and its grid runs from the
lowest Φ₀ among them to the CMB facets' area mean of Φ₀), per interval
in r between the nodes' radii with dc = g dr, by 20-point Gauss–Legendre:
the integrands are smooth in r and behave like √(c − c_min) in c.

The measures: the rigid part of u_h − u* is fitted on the discrete rigid
modes and removed, and the potential compared with φ* − r·∇Φ₀ over the
whole ball (the translation pair stays null with zero border unknowns;
the rotations have r·∇Φ₀ = 0). Relative L² errors of u and of the stress
over the shell, of φ over the ball and over the core; the border's nodal
U, s, ν against U*, s*, ν* at the grid's nodes (largest error over the
largest exact value, the RMS ratio, and the volume-weighted L² error
∫A(v_h − v*)² dc), and λ. The driver also prints the border rows'
residual at the exact solution and the deviation of the border's
element volumes (binned fluid quadrature points) from the analytic
shells between its levels.

### Running the fluid-core family

```
cd <build>/benchmarks/elastic/manufactured
<mpiexec> -np 4 ../../bin/manufactured_fluid_benchmark \
    -m ../../../data/earth_fluidcore_3d.msh -o 2 -levels 16 -out o2.json
./run_fluid --h 0.3 0.2 --order 2 3 --np 4
./run_fluid --h --order 2 3 --ladder-h 0.2 --ladder-levels 4 6 9 12 16 24 --np 4
./manufacture_fluid --galerkin
```

`run_fluid` builds the meshes with `meshes/layered_earth.py` at the
scale h/0.135 (the canned interface size, `common.SIZING`), runs the
h ladder with the level count proportional to 1/h (`--levels-at`, 6 at
h = 0.3: 6, 9, 12, 18 at 0.3, 0.2, 0.15, 0.1) and the levels ladder at
`--ladder-h`, and writes `runs_fluid/<route>/table_h.txt` and
`table_levels_h<h>.txt`. A level count whose grid has an element without
a fluid quadrature point is refused by the class and recorded so.

The server's share (16 ranks; extrapolating the element counts, 2048,
5346 and 13 302 at h = 0.3, 0.2 and 0.135, the h = 0.1 mesh has about
27 000 elements and 200 000 displacement unknowns at order 3):

```
./run_fluid --h 0.3 0.2 0.15 0.1 --order 2 3 --np 16 --ladder-levels
./run_fluid --h --order 2 3 --ladder-h 0.135 --ladder-levels 4 8 12 16 24 32 --np 16
./run_fluid --h --order 2 3 --ladder-h 0.1 --ladder-levels 6 9 12 18 24 36 48 --np 16
./run_fluid --h 0.3 0.2 0.15 0.1 --order 2 3 --np 16 --ladder-levels --projected-levels
./run_fluid --h 0.3 0.2 0.15 0.1 --order 2 3 --np 16 --ladder-levels --exact-hessian
```

### Findings (fluid core)

Exact levels and the class default for the gravity term unless said;
MINRES to 1e-11.

**Every source is in place.** The potential falls steeply with the
order and with h (canned mesh, 16 levels: 1.9e-2, 1.0e-3, 8.1e-5 at
orders 1, 2, 3; rates 4.4 to 4.6 from h = 0.3 to 0.2), and dropping a
source leaves an O(1) residual (order 1, 16 levels: without the border
data U 0.87 and λ 1.66 against 3.2e-2 and 4.5e-2, φ over the core 0.11
against 1.8e-2; without the CMB traction u 0.33 against 5.7e-2).

**The shell's displacement error is the wall multiplier's.** In every
run at order 3, on every mesh and at 3 to 24 levels, the relative error
of u is 0.086 to 0.091 times the relative error of λ (1.7 times its
absolute error), and the stress's 0.11 times; at order 2 the ratio is
0.070 to 0.089, the mesh's own error showing on the coarsest mesh. The
border's level-mean wall pressure sets the displacement's accuracy. It
comes out of the one-dimensional grid, whose own errors fall like
n^-1.5 in the level count n (`--galerkin`: U 3.2e-2, 3.0e-3, 3.6e-4 at
4, 16, 64 levels), and on the canned mesh at order 2 λ falls at that
pace (9.3e-2, 3.1e-2, 1.2e-2 at 4, 8, 16 levels: rates 1.6, 1.4).

**The binned level integrals cap the level count.** The border's volume
integrals (element volumes, the s and ν blocks, the ν columns) are sums
over the fluid quadrature points binned by level; once the bins are
thinner than the points' spacing they scatter. On the canned mesh at
order 2 the element volumes deviate from the analytic shells by 1.3 %,
6.3 %, 40 % and 53 % at 4, 8, 16 and 32 levels (9 % at 16 levels with
order 3's denser rule), and past the cap every border-coupled measure
gets worse (32 levels: λ 3.0e-2, u 2.45e-3 against 1.2e-2 and 1.03e-3 at
16). Beyond it the class refuses the grid: on the h = 0.3 mesh above 6
levels at order 2 and above 12 at order 3. The best count is the largest
the bins resolve, and the displacement does not get below about 1e-3
on any of these meshes:

| mesh (interface h) | order | best levels | u | λ | φ |
|---|---|---|---|---|---|
| 0.3 | 2 | 6 | 2.7e-3 | 3.5e-2 | 1.4e-2 |
| 0.3 | 3 | 8 | 8.9e-4 | 9.8e-3 | 1.3e-3 |
| 0.2 | 2 | 12 | 2.7e-3 | 3.3e-2 | 2.1e-3 |
| 0.2 | 3 | 12 | 9.3e-4 | 1.1e-2 | 1.6e-4 |
| 0.135 (canned) | 2 | 16 | 1.0e-3 | 1.2e-2 | 1.0e-3 |
| 0.135 (canned) | 3 | 16 | 6.5e-4 | 7.6e-3 | 8.1e-5 |

So the border's error never stops mattering on these meshes, and with
the level count proportional to 1/h the displacement does not converge
in h at the elements' rate while the potential does:

```
order  h    levels  u        rate   stress   rate  phi      rate  U        lambda   vol
  2   0.3     6    2.71e-3    -    1.64e-2    -   1.35e-2    -   1.39e-1  3.54e-2  0.90
  2   0.2     9    3.16e-3  -0.38  5.08e-3  2.89  2.07e-3  4.61  1.59e-2  3.94e-2  0.22
  3   0.3     6    3.03e-3    -    4.17e-3    -   1.30e-3    -   7.96e-2  3.49e-2  0.31
  3   0.2     9    2.13e-3   0.87  2.77e-3  1.00  2.17e-4  4.40  1.88e-2  2.49e-2  0.13
```

(vol: the largest deviation of the border's element volumes from the
analytic shells.) The levels ladders, u and λ with the volume deviation:

```
h = 0.3   levels:        3        4        6        8        12
order 2   u        1.58e-2  7.63e-3  2.71e-3  refused  refused
          lambda   2.08e-1  1.09e-1  3.54e-2
          vol      5.2e-3   0.34     0.90
order 3   u        1.89e-2  1.37e-2  3.03e-3  8.85e-4  3.50e-2
          lambda   2.20e-1  1.59e-1  3.49e-2  9.76e-3  4.08e-1
          vol      0.08     0.22     0.31     0.58     1.02
h = 0.2   levels:        4        6        9        12       16       24
order 2   u        7.70e-3  5.06e-3  3.16e-3  2.67e-3  2.98e-3  8.53e-3
          lambda   9.33e-2  6.23e-2  3.94e-2  3.32e-2  3.73e-2  1.03e-1
          vol      0.06     0.11     0.22     0.49     0.69     0.91
order 3   u        7.57e-3  3.10e-3  2.13e-3  9.32e-4  9.92e-4  1.43e-3
          lambda   8.81e-2  3.60e-2  2.49e-2  1.09e-2  1.16e-2  1.67e-2
          vol      0.04     0.06     0.13     0.35     0.40     0.78
canned    levels:        4        8        16       32
order 2   u        7.81e-3  2.53e-3  1.03e-3  2.45e-3
          lambda   9.25e-2  3.06e-2  1.17e-2  2.96e-2
          vol      0.013    0.063    0.40     0.53
```

**The lift s and its multiplier ν are resolved at a few levels only.**
They are set through the border's volume blocks, whose scatter the ν row
amplifies (its data are dominated five-fold by the level integral of
the potential): at 16 levels on the canned mesh s is off by 3.6 and ν
by 0.88 (largest nodal error over the largest value), at 4 levels by
6.4e-2 and 0.12. ν's nodal error next to the centre is the grid's own:
the exact-integral Galerkin border has 0.10 to 0.12 there for any count
from 4 to 64 levels (the s row's weight A vanishes like √(c − c_min) at
the centre), while its volume-weighted error falls (7.3e-2 to 6.4e-3)
and s converges at first order (5.8e-3 at 4 levels, ten times below the
class's).

**The two level routes agree at order 2** to 0.3 % in every measure (the
projection of this Φ₀ is accurate); at order 1 the projected
(piecewise-linear) levels bin the fluid worse (volumes up to 75 % off).
**The two Hessian routes** give u 9.9e-4 (exact) and 1.0e-3 (default) at
order 2 and 16 levels: the border dominates either way. **Order 1** holds
u at 5.7 % whatever the level count (finding 1 of the solid family).
**Cost:** the border's elimination makes one base solve per ν node and
wall, so a solve costs about levels + 2 single solves: on the canned
mesh at 4 ranks 20 s at order 2 and 16 levels, 37 s at 32 levels, 164 s
at order 3 and 16 levels.

## Findings

Meshes from `homogeneous_body.py` (order-2 geometry, buffer to 1.5 radii,
far size 2h), MINRES to 1e-11, relative L2 errors of u, of the stress
σ(u) and of φ over the ball, with the rates between consecutive sizes.

**Order 1 does not converge on curved geometry** (2-D: u 4.9e-2, 4.6e-2,
4.5e-2 at h 0.1, 0.05, 0.025; 3-D slowly, rates 1.4 and 0.7): the rigid
rotations of an order-2 geometry are not in the order-1 space
(`doc/methods/self_gravitation.md`, "Element order on curved meshes"). Use a
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
