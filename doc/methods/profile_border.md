# The profile border: the closure in the pressure-profile variable

> **Where this sits.** Dahlen's reduction eliminates the displacement of a
> hydrostatic fluid region and leaves one function of the level of the
> background potential undetermined; the closure supplies it
> (`doc/theory/dahlen_closure.tex`). The level border solves the closure
> in its primal form, in which the unknown is the cumulative volume
> beneath each level. The profile border described here solves the same
> closure in its dual form, in which the unknown is the perturbation of
> the fluid's pressure profile. The two are alternatives on the mixed
> class; the referential class has the level border only.

Method notes for `SetProfileBorder()` of
`LinearQuasiStaticMixedSelfGravitatingProblem` (`mixed_problem.hpp`), built
by `BuildProfileBorders()` (`profile_border.hpp`).

## 1. Formulation

Let `c = Phi_0` be the level, and `rho(c)`, `kappa(c)` the density and
bulk modulus of the fluid as functions of it. With a prime for `d/dc`,
the stratification is

    beta = -(rho' + rho^2 / kappa) >= 0,

zero for a neutrally stratified fluid and positive for a stable one;
`beta kappa / rho^2` is the excess of the Bullen parameter over one. An
unstably stratified fluid (`beta < 0`) has no static state that a
relaxation would reach and is refused.

The perturbation of the pressure profile at a fixed level is written
`varpi = rho gamma`, with `gamma` the change of the fluid's Bernoulli
function. The Eulerian perturbations in the fluid are

    delta p = varpi - rho phi,        delta rho = rho' phi - varpi',

so Dahlen's relations are the case `varpi = 0`. The linearised static
problem is the stationarity of the functional of the mixed class, which
holds Dahlen's three terms, with three terms added,

    - (1/2) a(varpi, varpi)
    + int_Sigma varpi (m . u) dS
    - int_Omega varpi' phi dV,

where `Omega` is the fluid region, `Sigma` its wall, `m` the outward
normal of the solid, and

    a(varpi, psi) = int_Omega [ d_a varpi d_a psi / beta + varpi psi / kappa ] dV,
    d_a varpi = varpi' + (rho / kappa) varpi = rho gamma' - beta gamma.

In a neutral layer `gamma` is constant and the first term of `a` is
absent. A region that is neutral throughout carries one number, and its
equation is the conservation of the region's mass. An applied potential
(tidal, centrifugal) enters the last term with `phi`.

Nothing in these terms refers to the geometry of the level surfaces. The
hypsometry is not needed, and a wall that cuts the level surfaces is
treated by the same integrals as a level wall.

## 2. Discretisation

A region is a stack of layers in the level, each neutral or stratified.
`gamma` is continuous, constant on a neutral layer and a polynomial of
degree `ProfileBorderOptions::degree` on a stratified one. The unknowns
are the values at the ends of the layers, the two ends of a neutral layer
sharing one, and the interior polynomial modes of the stratified layers;
the basis functions are scaled to unit diagonal of `a`.

The form is assembled as

    a_kl = int [ -rho' b_k b_l - rho (b_k' b_l + b_k b_l') + rho^2 b_k' b_l' / beta ] dV,

which equals the form above when `beta = -(rho' + rho^2 / kappa)`, and
the columns are

    on the potential:     - int_Omega (rho' b_k + rho b_k') chi dV,
    on the displacement:    int_Sigma rho b_k (m . v) dS,

all by quadrature, with the basis evaluated at the level of each point.
The form and the potential columns use the same quadrature points and the
same `rho'` as the fluid mass term of the class. The row of the constant
`gamma = 1` is then the conservation of the region's mass, as an identity
of the discrete system.

The bordered system is symmetric. It is solved by the elimination engine
(`bordered.hpp`), with one column solve per unknown, or by the monolithic
route (`SetMonolithicBorder()`).

## 3. The stratification

The layers are found by a survey: the level range is divided into
`ProfileBorderOptions::bins` bins, and the mean excess of the Bullen
parameter in each decides whether it is neutral (within `neutral_tol` of
zero), stratified (above) or unstable (below). Neighbouring bins of one
kind form a layer. Within a stratified layer `beta` is taken from the bin
means, joined linearly in the level, not from the value at each point.

The excess comes from one of two places.

- **Declared** (`FluidRegion::stratification`): the region supplies it.
  This is the route for a model that is neutral or weakly stratified.
  With `density_gradient` set to `-(1 + excess) rho^2 / kappa` as well, no
  derivative is taken on the mesh anywhere.
- **Computed**, from `density_gradient` and `bulk_modulus`. When the
  density gradient is the class's numerical one, its errors are of the
  size of the discretisation and are large next to a centre, where it is
  a ratio of two vanishing quantities. A strongly stratified region
  survives this; a neutral one does not (it is found unstable in places)
  and a weakly stratified one may not.

## 4. Verification

`tests/TestProfileBorder.cpp` and `tests/TestProfileBorderPar.cpp`, on a
ball with a fluid core of density `rho_c - a r^2` and bulk modulus
`1 + delta` times the neutral value:

- agreement with the level border under a uniform load, to the
  discretisation error of the two (displacement 5e-4 to 7e-4, potential
  2e-5), where the reduction without a border is wrong by a factor of
  about two;
- the response departs from the neutral one in proportion to `delta`;
- the border is not excited by a load of degree two on a sphere;
- the conservation of the region's mass, evaluated from the solution;
- the elimination against the monolithic route, with and without
  rotation, at solver tolerance;
- the refusal of an unstable region, and the declared stratification.

The degree-0 load Love number against the radial reference
(`benchmarks/love_numbers`, method `dahlen`, order 3; signed relative
error of h'_0 at element sizes 0.3, 0.2 and 0.15):

| model | profile border, degree 6 | level border, 16 nodes | level border, 32 nodes |
|---|---|---|---|
| `aw_core` (neutral) | -2.2e-4, -6.8e-5, +2.3e-5 | -3.1e-5, +4.1e-4, +8.2e-4 | -9.2e-4, -5.0e-4, -1.1e-3 |
| `bullen_core_weak` (excess 0.02) | -2.2e-4, -6.9e-5, +2.2e-5 | -2.9e-5, +4.0e-4, +8.0e-4 | -8.9e-4, -4.8e-4, -1.0e-3 |
| `bullen_core_strong` (excess 0.5) | -2.5e-4, -9.3e-5, +1.5e-5 | -3.2e-5, +2.7e-4, +5.6e-4 | -5.1e-4, -2.3e-4, -4.6e-4 |
| `stable_core` | -4.5e-4, +8.3e-5, -1.6e-4 | +2.0e-4, +5.6e-4, +8.1e-4 | -1.5e-4, +2.1e-4, +2.0e-4 |
| `stable_mantle` | -5.6e-4, -1.1e-5, -2.2e-4 | +5.9e-5, +4.4e-4, +6.9e-4 | -2.8e-4, +1.0e-4, +1.1e-4 |

The first three rows have the stratification declared; on them degree 10
gives the same figures as degree 6. On these models the profile border's
error falls with the mesh, and the level border's does not: it depends on
the number of nodes and stays between 3e-5 and 1e-3. At order 2 the
profile border's error is about 8e-3, 4e-3 to 5e-3 and 1.5e-3 to 2e-3 on
the three meshes.

`stable_core` and `stable_mantle` have a density linear in radius down to
the centre, so their stratification is singular there (the excess of the
Bullen parameter grows without bound as the radius goes to zero) and
`gamma` is not smooth in the level at the centre. On them the profile
border stays at 1e-4 to 2e-4 and depends on the degree (degree 10:
-6e-6, +1.4e-4, +1.8e-4 on `stable_core`). Their stratification is
computed, not declared.

The profile border has fewer column solves than the level border: one
for a neutral region and `degree + 1` for a stratified layer, against one
for each node of the level border. In these runs its solves took
roughly a tenth to two thirds of the level border's time at 16 nodes,
the least on the neutral model.

## 5. Limits

- The mixed class only.
- One profile for each region: fluid in separate pockets of one level
  (a wall with topography) shares it.
- A model whose stratification is singular at a centre is resolved
  slowly by polynomials in the level.
- The stratification is uniform in the benchmark driver's declaration
  (`-bullen-excess`); a case does not yet carry a stratification field.
