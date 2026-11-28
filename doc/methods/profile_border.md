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

The layers of a region come from one of two places.

- **Declared** (`ProfileBorderOptions::breaks`): the levels at which the
  model says the stratification changes. A region with any of them
  inside its level range has for layers the intervals between them. An
  interval is stratified when most of its fluid is, by the weight
  `rho^2 / kappa`, and neutral otherwise; neighbouring neutral intervals
  are one layer, and two stratified ones stay two, the break a knot of
  `gamma`. A stratified layer takes the degree asked for.
- **Surveyed**, in a region with no break inside its range: the range is
  divided into `ProfileBorderOptions::bins` bins, and the mean excess of
  the Bullen parameter in each decides whether it is neutral (within
  `neutral_tol` of zero) or stratified (above). Neighbouring bins of one
  kind form a layer, of degree no more than the number of bins it spans.
  A boundary between a stratified and a neutral stretch is then at an
  edge of a bin and not where the model has it, which costs accuracy
  (§4).

The survey is made in every region, and a bin whose mean excess is below
`-neutral_tol` is unstable: the region is refused, or the bin taken as
neutral.

A break is a level, the value of the background potential, because the
level is the profile's own coordinate. For a spherical model a knot at a
radius is the discrete background potential at a point of that radius
(`ValueAtPoint`, `mesh.hpp`). The benchmark drivers declare, for each
fluid layer of a case's model, the radii at which its Bullen excess goes
from zero to positive or jumps; they are edges of the model file's
radial elements (`StratificationChanges`, `benchmark_case.hpp`;
`-profile-survey` leaves the layers to the survey).

A break need not be an interface of the mesh. Where it is inside
elements the levels of the discrete potential put a sliver of the fluid
of one side on the other; that is why an interval's kind goes by most of
its fluid and not by its mean, and a point of neutral fluid in a
declared stratified layer takes the layer's mean excess.

The excess itself comes from one of two places.

- **Declared** (`FluidRegion::stratification`): the region supplies it.
  This is the route for a model that is neutral or weakly stratified,
  and the route of the benchmark drivers: a model states the Bullen
  excess of each fluid layer (planetmodel's `bullen_excess`, zero where
  the layer is neutral), a case carries it in its model file, and the
  driver hands it to the region. The driver refuses the profile border
  on a fluid layer that states none. Unless the region gives a density
  gradient of its own, the problem class then takes
  `rho' = -(1 + excess) rho^2 / kappa`
  (`DeclaredDensityGradientCoefficient`), and no derivative is taken on
  the mesh anywhere. Within a stratified layer `beta` is then the
  declared value at each point, not below `neutral_tol`.
- **Computed**, from `density_gradient` and `bulk_modulus`. Within a
  stratified layer `beta` is then taken from the survey's bin means,
  joined linearly in the level, not from the value at each point. When the
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
- the refusal of an unstable region, and the declared stratification;
- on a core stratified below a knot and neutral above it, with the
  knot's level declared: the two layers the model has, the mass of the
  region, the two routes, and the level border's answer to 3e-4 in the
  displacement, where the survey's layers are 7e-3 from it;
- a break where nothing changes: two stratified layers give the one
  layer's answer (4e-7), two neutral intervals are one layer, a break
  outside the region's range is ignored, a break given twice refused;
- in parallel the knot's level and the displacement are the serial
  problem's.

The degree-0 load Love number against the radial reference
(`benchmarks/love_numbers`, method `dahlen`, order 3; signed
relative error of h'_0 at element sizes 0.3, 0.2, 0.15 and 0.1, each
model's stratification read from its case):

| model | its fluid | profile border, degree 6 | level border, 16 nodes |
|---|---|---|---|
| `aw_core` | neutral | -2.2e-4, -6.8e-5, +2.3e-5, +1.2e-5 | -3.3e-5, +4.1e-4, +8.1e-4, +8.7e-4 |
| `bullen_core_weak` | excess 0.02 | -2.2e-4, -6.9e-5, +2.2e-5, +1.2e-5 | -3.1e-5, +4.0e-4, +8.0e-4, +8.5e-4 |
| `bullen_core_strong` | excess 0.5 | -2.5e-4, -9.3e-5, +1.5e-5, +9.2e-6 | -3.3e-5, +2.7e-4, +5.6e-4, +5.7e-4 |
| `bullen_mantle` | excess 0.02 | -2.8e-4, -1.2e-4, -2.9e-5, -4.5e-6 | -6.4e-5, +4.0e-4, +8.2e-4, +9.2e-4 |
| `inner_core` | neutral, a shell between two walls | -2.2e-4, -6.5e-5, +2.0e-5, +8.4e-6 | -9.5e-4, -9.7e-4, -1.9e-4, +8.0e-5 |
| `bullen_mantle_stacked` | excess 0.02 below half the core's radius, neutral above | -2.7e-4, -1.2e-4, -2.9e-5, -4.1e-6 | -5.6e-5, +4.0e-4, +8.2e-4, +9.3e-4 |
| `prem_4` | PREM's outer core made stable: four layers | +7.8e-4, +5.3e-4, +4.7e-4, +4.3e-4 | +2.4e-4, -1.9e-4, +2.3e-4, +5.4e-4 |

On the first six the profile border's error falls with the mesh and is
the same at degree 10 as at degree 6; the level border's does not fall
and depends on its number of nodes. At an element size of 0.07 the
profile border gives +5.9e-6 on `aw_core` and -1.2e-6 on
`bullen_mantle`, the level border +8.7e-4 and +9.3e-4. At order 2 the
profile border's error on `bullen_mantle` is -1.9e-3, -1.9e-3, -1.4e-3
and -3.7e-4 on the four meshes.

`bullen_mantle_stacked` has its two layers from the knot of its model,
which is not an interface of the mesh. With the layers left to the
survey its error is -3.3e-4, -1.7e-4, -9.1e-5, -6.1e-5 at degree 6 and
-3.0e-4, -1.4e-4, -5.7e-5, -2.9e-5 at degree 10: slower, and dependent
on the degree. On `prem_4` the border has the four layers of the
stabilised core (stratified, neutral, stratified, neutral) and its error
is the same at degree 10; the other Love numbers of that model are at
1e-3 on these meshes, which do not resolve it.

The profile border has fewer column solves than the level border: one
for a neutral region and `degree + 1` for a stratified layer, against one
for each node of the level border. For degrees 0 to 6 at an element size
of 0.1 the runs above took 18 to 27 s with the profile border at degree
6 and 39 to 42 s with the level border at 16 nodes on the models with a
core to the centre, on 32 ranks, and 29 s against 68 s on `inner_core`,
on 24.

## 5. Limits

- The mixed class only.
- One profile for each region: fluid in separate pockets of one level
  (a wall with topography) shares it.
- A model whose stratification is singular at a centre is resolved
  slowly by polynomials in the level.
- Without declared breaks the layers are bins of the survey (§3).
- A break is declared as a level. On a body that is not spherical the
  caller has to know the levels of its model's knots.
