# 3-D GIA modelling codes: a survey

A survey of the numerical codes for 3-D glacial isostatic adjustment and
self-gravitating (visco)elastic loading, drawn from the published papers
listed under Sources and from the codes' public repositories. It is
background research for the library's verification benchmarks: what each
code does, what it assumes, what it cannot do, and where AdGIA's
treatments (full gravity, meshed and possibly stratified fluid cores, exact
DtN, aspherical boundaries by relabelling) sit relative to the field. The
availability statements record the public state of each code when the
survey was compiled and will age. Al-Attar & Crawford
(2016), the theoretical basis of the relabelling benchmarks, is not a GIA
code paper and is not surveyed.

## The published benchmarks (both 1-D)

**Spada et al. 2011** (GJI 185, 106–132). Eight codes —
normal-mode (TABOO, FastLove-HiDeg, MHPLove, VEENT), Post–Widder Laplace
inversion (ALMA), spectral–finite-element (VILMA), ABAQUS FE (flat and
spherical) — on a 3-layer incompressible Maxwell model with elastic
lithosphere and uniform inviscid fluid core (M3–L70–V01). Agreement to
6+ significant figures among the semi-analytic codes; FE within ~10%
(grid effects). Fixes community conventions worth knowing: Wu & Ni (1996)
CMB conditions, degree-1 Love numbers in the CM frame (k₁ᴸ = −1),
hydrostatic initial rotational state. Deliberately excludes
compressibility (dense normal-mode spectra) and the sea-level equation.
Its Tables 3–14 remain the standard reference numbers for validating a
new code on spherical models; their viscoelastic content is normal-mode
based (Laplace domain).

**Martinec et al. 2018** (GJI 215, 389–414). Ten codes on the
sea-level equation: five synthetic cases from no-SLE through fixed
coastlines to fully moving coastlines with floating ice, on the same
incompressible non-rotating 1-D model. Agreement ~1.5% max in U and N;
main discrepancy sources are spatial discretisation and Gibbs behaviour
at load margins. Only five of the ten codes could run moving coastlines.
Also 1-D only; 3-D structure explicitly deferred.

**A 3-D benchmark does not exist in print.** The initiative is EGU
abstract EGU22-1447 (Klemann and 16 co-authors, out of the 2021
PALSEA-SERCE workshop) proposing a catalogue of synthetic experiments;
no paper or preprint has followed it (a 2025 GIA workshop held a
benchmarking breakout, without a published outcome). Every 3-D code below is verified only against 1-D
semi-analytic references. This is the gap the relabelled-benchmark
methodology addresses: exact aspherical reference solutions without
needing a second 3-D code.

## The 3-D codes

### Martinec spectral–finite-element lineage (VILMA, VEGA)
- Theory: Martinec 1999 (GJI 137, 469–488) — tensor
  spherical harmonics in angle, FE in radius, explicit time stepping
  with a viscous "memory" term; lateral viscosity couples degrees/orders
  through Clebsch–Gordan machinery. Implemented as Martinec (2000) and
  productionised at GFZ as VILMA (SLE, moving coastlines, rotational
  feedback, climate-model coupling).
- Assumes: elastic moduli and density **radial only** (lateral variation
  in viscosity alone), all boundaries spherical, incompressible (a
  compressible prototype exists), Maxwell, explicit Euler (Δt limited by
  the minimum Maxwell time), practical lateral resolution ~degree 50–128.
- Fluid core: free-slip CMB conditions on an unmeshed core.
- Availability: Martinec's own implementation and GFZ's VILMA are not
  distributed ("open source in preparation" per the natESM
  documentation). A "VILMA v2" rewrite is public at
  github.com/fesmc/vilma, without a licence and described by its authors
  as not yet ready for scientific production use.

### Latychev finite-volume code (Seakon)
- Latychev et al. 2005 (GJI 161, 421–444): node-centred
  finite volumes on an unstructured tetrahedral grid, all four unknowns
  (u, φ₁) in one monolithic system, GMRES+ILU, MPI. Full first-order
  gravity; elastically compressible but incompressible in the fluid
  limit; arbitrary 3-D variations in viscosity *and* elastic parameters
  across arbitrary internal surfaces. Explicit Euler in time (first
  order, Δt ≤ min Maxwell time). Grids honour radial PREM
  discontinuities — no boundary topography. Core unmeshed: uniform
  incompressible inviscid fluid via CMB condition. No SLE in the 2005
  paper (added later via the Mitrovica–Milne theory); rotation added
  later; extended to transient rheology (Lau et al. 2026, JGR; not
  checked against the paper). Later versions are reported to use an
  exponential (Bailey-type) explicit time-stepper in place of forward
  Euler, easing the Δt ≤ Maxwell-time constraint; this is unpublished and
  not in the 2005 paper.
- Availability: the code was long available to collaborators only. A
  snapshot is deposited at the Brown Digital Repository
  (doi 10.26300/y2ct-jp25, CC BY-NC): about 100 GB, configured for 192
  CPUs, accompanying one paper and distributed via Globus. It is an
  archival release rather than a maintained distribution, without public
  documentation.

### Zhong / A / Yuan finite-element lineage (→ CitcomSVE-3.0)
- A, Wahr & Zhong 2013 (GJI 192, 557–572): compressible 3-D FE
  (CitcomS heritage), trapezoid-rule Maxwell stepping, full first-order
  gravity — but the Poisson equation is *not* solved by FE: φ comes from
  spherical-harmonic Green's-function integrals, which **requires the
  background density to be layered (1-D)**; self-gravity enters through
  an iteration (6–8 per step). 3-D viscosity and Lamé parameters; polar
  wander feedback; SLE with partial ocean-function time dependence.
  Degree-dependent errors ~1% (low degrees) to ~3–5% (short wavelengths,
  point GPS rates).
- CitcomSVE-3.0 (Yuan, Zhong & A 2025, GMD 18, 1445–1461): the
  packaged, parallel (multigrid, >75% efficiency to 6144 cores), open
  release. Compressible since 3.0; fully 3-D viscosity (linear or
  nonlinear) and elastic moduli; full Kendall et al. (2005) SLE with
  moving coastlines; polar wander; CM frame with degree-1 handled. The
  potential is still computed in spherical harmonics to a cutoff degree
  (20–64) — the dominant cost (¼–½ of runtime) and the resolution limit
  on geoid outputs; density must be radially layered; core unmeshed
  (incompressible-core CMB condition); regular spherical grid (no
  unstructured refinement, no boundary topography); trilinear elements,
  second-order convergence; benchmark errors <0.1% to degree 4, <2% to
  degree 16 at ~50 km resolution, degree (2,1)/polar-wander term the
  least accurate (1.4–17.5%).
- Availability: open — github.com/shjzhong/CitcomSVE (LGPL-3.0) and
  Zenodo (10.5281/zenodo.13932410), maintained, with a manual; the most
  complete openly available 3-D GIA code in this survey.

### ABAQUS lineage (Wu → van der Wal → FEMIBSF)
- Huang et al. 2023 (GJI 235, 2231–2256), the FEMIBSF approach:
  the commercial FE kernel solves ∇·σ = 0, and *all* gravity terms
  (pre-stress advection, perturbed potential, compressibility buoyancy)
  are applied as iteratively updated body/surface forces, with the
  authors' own OpenMP Poisson solver doing direct Green's-function
  volume integration over all elements (no spherical harmonics —
  deliberate, to avoid degree leakage; expensive). Compressible,
  spherical, CM-frame degree-1 handled; core unmeshed (potential stress
  + Winkler buoyancy at the CMB; homogeneous incompressible core).
  Designed for 3-D structure but **validated only on 1-D models**, vs
  analytic homogeneous solutions, VILMA-C and ICEAGE (<1–4% typical).
  Reproduces the Rayleigh–Taylor instabilities of homogeneous
  compressible models. No rotation (sketched only), no SLE in the paper.
- Assumptions/limits: accuracy strongly mesh-dependent (coarse grids
  systematically underestimate the driving potential; degree-15 errors
  can exceed 100% before refinement); hanging nodes forbidden; the
  earlier Wu-method caveat that ABAQUS compressibility lacks
  buoyancy-consistency is what the iterative scheme repairs. Not
  releasable as a whole (commercial kernel); scripts not published. The
  same family includes van der Wal's composite-rheology Fennoscandia and
  Antarctica models and van Calcar's coupled ice-sheet–GIA model.

### Cambridge adjoint spectral code (Lloyd et al. 2024)
- Lloyd et al. 2024 (GJI 236, 1139–1171): the Al-Attar & Tromp
  (2013) / Crawford et al. (2018) rate formulation, generalized
  spherical harmonics to degree 64 × radial spectral elements,
  pseudo-spectral lateral-viscosity terms, non-iterative SLE with
  shoreline migration, and the distinctive capability: adjoint Fréchet
  kernels for 3-D viscosity and initial sea level (verified against
  finite differences). Compressible, full gravity, fluid outer core in
  the formulation.
- Assumes: **1-D elastic and density structure** (3-D in viscosity
  only), no rotation yet, Maxwell (transient implementable), explicit
  RK2 stepping with Δt ~ half the minimum Maxwell time — hence viscosity
  floored at 2×10¹⁹ Pa s, and ~TB-scale storage of forward fields for
  kernels. The degree-64 truncation of the paper is not intrinsic to
  the method: forward runs are feasible at degree ~256 on a server, and
  the truncation reflects the storage of forward fields for kernel
  construction (unpublished).
- Availability: public as **sl3d** (1-D and 3-D versions); it runs on a
  workstation and needs a server for production work.

### SPECFEMX (Gharti; not a GIA code yet, but adjacent)
- Spectral-infinite-element method: spectral elements on unstructured
  hex meshes with infinite elements carrying the gravitational potential
  to infinity — the one code family solving the unbounded Poisson
  problem on genuinely 3-D density without spherical harmonics. Papers:
  Gharti & Tromp 2017 (SIEM Poisson), Gharti et al. 2018 (gravity
  anomalies), 2019a (coseismic + post-earthquake viscoelastic
  deformation), 2019b (earthquake-induced gravity perturbations), 2023
  (self-gravitating rotating wave propagation).
- Availability: github.com/homnath/SPECFEMX (GPL-3.0), actively
  developed. Post-earthquake viscoelastic and gravity-perturbation
  capabilities are marked "experimental" in the manual. The README
  documents in-progress work on sea-level change via the Crawford et
  al. (2018) rate formulation, with open items (ice term missing from
  the weak form, load/BC questions, memory-integral time marching): a
  quasi-static self-gravitating loading + SLE capability is in
  development and explicitly incomplete. No published GIA application.

### Others, briefly
- **G-ADOPT viscoelastic** (Scott et al. 2026, GMD 19, 2717):
  Firedrake-based FE with *automatically derived adjoints*, compressible,
  lateral and nonlinear/transient rheology — but **no self-gravity yet**.
  Open source; the closest thing to an adjoint-capable community
  competitor, and its missing gravity is exactly the hard part.
- **ASPECT-based GIA** (Weerdesteijn et al. 2023): viscoelastic loading
  in a very active open FEM framework, but Cartesian boxes, no
  self-gravity, no SLE.
- **FastIsostasy** (GMD 2024, Julia, active): deliberately approximate
  regional 2-D (LV-ELVA) solver for coupling to ice-sheet models; not a
  field-equation competitor.
- **ML surrogates** (GMD 2024): emulators trained on 3-D runs, not
  solvers.

## Cross-cutting reading

Setting AdGIA's design choices against the field:

1. **Gravity.** All serious codes keep the three first-order gravity
   terms, but the Poisson strategies split into: spherical-harmonic /
   Green's-function solves that force a **1-D background density**
   (A13, CitcomSVE, the spectral codes), direct volume integration that
   is exact but expensive (FEMIBSF, Seakon's boundary integrals), and
   genuine unbounded-domain discretisations (SPECFEMX's infinite
   elements). An FE Poisson solve on the same mesh with an exact DtN
   closure — the library's arrangement — handles 3-D density at no structural
   cost; in the survey only SPECFEMX is comparably general, and it has
   no working GIA capability yet.
2. **Fluid cores.** Universally unmeshed: every GIA code reduces the
   core to a CMB boundary condition for a uniform (or neutrally
   stratified) incompressible inviscid fluid. None solves for the
   potential inside a stratified fluid core (the ρ′ term). The submesh
   fluid machinery of AdGIA has no counterpart in the field, and
   the degree-0 limitation of the eliminated-fluid (Dahlen) treatment
   (`doc/gauged_fluid.md`, section "Degree 0") concerns physics none of
   these codes represents. The published approximate CMB conditions are
   available in the Love-number benchmarks as option `-cmb`, for
   comparison against the meshed core; the conditions and their translation
   into the library's interface terms are in `doc/quasi_static_models.tex`
   (section "Fluid regions and the ladder of CMB approximations"), their
   measured cost and accuracy in `doc/benchmarks.tex` (section "The CMB
   approximations: results").
3. **Boundary and interface topography.** No surveyed code handles
   aspherical internal boundaries: grids and spectral expansions honour
   spherical PREM interfaces (Seakon's unstructured grid could in
   principle, but the published model does not; CitcomSVE's grid is a
   regular spherical shell). Relabelling carries topography exactly on
   every interface while the computational domain stays spherical —
   both a modelling capability and, via the pulled-back forms, the
   verification methodology.
4. **Verification.** Every 3-D code above is validated against 1-D
   semi-analytic references only, typically to ~1% at low degrees and
   several per cent at short wavelengths; a published 3-D benchmark does
   not exist. Relabelling gives aspherical references of two kinds: a
   discrete change-of-variables identity that holds to ~1e-6 for the
   welded formulations, and an aspherical reference body checked by convergence
   (`doc/benchmarks.tex`, section "The relabelling family").
5. **Time stepping.** The explicit-Euler Δt ≤ Maxwell-time constraint
   recurs across the field (Martinec lineage, the published Seakon,
   Lloyd et al.) and is what forces viscosity floors; CitcomSVE and A13
   use slightly better trapezoid stepping, and modern Seakon reportedly
   an exponential (Bailey-type) explicit integrator (see above). The
   library's exponential trapezoid and adaptive steppers have no such
   limit (`doc/viscoelasticity.md`, section "Time stepping").
6. **Adjoints.** Only the Cambridge line has published GIA sensitivity
   kernels; G-ADOPT has automatic adjoints without gravity. AdGIA
   implements no adjoint; its mapping-aware forms carry the deformation
   gradient F explicitly (`doc/mappings.md`).

## Sources

Published papers (local PDF file names in brackets, as a convenience; the
PDFs are not in the repository):

- A, G., Wahr, J. and Zhong, S. (2013). Computations of the viscoelastic
  response of a 3-D compressible Earth to surface loading: an application
  to Glacial Isostatic Adjustment in Antarctica and Canada. *Geophys. J.
  Int.*, 192, 557–572, doi:10.1093/gji/ggs030. [`ggs030.pdf`]
- Al-Attar, D. and Crawford, O. (2016). Particle relabelling
  transformations in elastodynamics. *Geophys. J. Int.*, 205, 575–593,
  doi:10.1093/gji/ggw032. [`ggw032.pdf`]
- Huang, P., Steffen, R., Steffen, H., Klemann, V., Wu, P., van der Wal,
  W., Martinec, Z. and Tanaka, Y. (2023). A commercial finite element
  approach to modelling Glacial Isostatic Adjustment on spherical
  self-gravitating compressible earth models. *Geophys. J. Int.*, 235,
  2231–2256, doi:10.1093/gji/ggad354. [`ggad354.pdf`]
- Latychev, K., Mitrovica, J. X., Tromp, J., Tamisiea, M. E., Komatitsch,
  D. and Christara, C. C. (2005). Glacial isostatic adjustment on 3-D
  Earth models: a finite-volume formulation. *Geophys. J. Int.*, 161,
  421–444, doi:10.1111/j.1365-246X.2005.02536.x. [`161-2-421.pdf`]
- Lloyd, A. J., Crawford, O., Al-Attar, D., Austermann, J., Hoggard, M.
  J., Richards, F. D. and Syvret, F. (2024). GIA imaging of 3-D mantle
  viscosity based on palaeo sea level observations – Part I: Sensitivity
  kernels for an Earth with laterally varying viscosity. *Geophys. J.
  Int.*, 236, 1139–1171, doi:10.1093/gji/ggad455. [`ggad455.pdf`]
- Martinec, Z. (1999). Spectral, initial value approach for viscoelastic
  relaxation of a spherical earth with a three-dimensional viscosity — I.
  Theory. *Geophys. J. Int.*, 137, 469–488. [`137-2-469.pdf`]
- Martinec, Z., Klemann, V., van der Wal, W., Riva, R. E. M., Spada, G.,
  Sun, Y., Melini, D., Kachuck, S. B., Barletta, V., Simon, K., A, G. and
  James, T. S. (2018). A benchmark study of numerical implementations of
  the sea level equation in GIA modelling. *Geophys. J. Int.*, 215,
  389–414, doi:10.1093/gji/ggy280. [`ggy280.pdf`]
- Spada, G. et al. (2011). A benchmark study for glacial isostatic
  adjustment codes. *Geophys. J. Int.*, 185, 106–132,
  doi:10.1111/j.1365-246X.2011.04952.x. [`185-1-106.pdf`]
- Yuan, T., Zhong, S. and A, G. (2025). CitcomSVE-3.0: a
  three-dimensional finite-element software package for modeling
  load-induced deformation and glacial isostatic adjustment for an Earth
  with a viscoelastic and compressible mantle. *Geosci. Model Dev.*, 18,
  1445–1461, doi:10.5194/gmd-18-1445-2025. [`gmd-18-1445-2025.pdf`]

Papers cited above by author and year but not read for this survey:
Martinec (2000), Al-Attar & Tromp (2013), Crawford et al. (2018), Gharti &
Tromp (2017), Gharti et al. (2018, 2019a, 2019b, 2023), Weerdesteijn et
al. (2023), Scott et al. (2026, G-ADOPT), Lau et al. (2026), the EGU22-1447
abstract, and the GMD (2024) FastIsostasy and machine-learning surrogate
papers.

Public repositories and pages consulted for availability: CitcomSVE
(GitHub and Zenodo); the Seakon deposit at the Brown Digital Repository
(bdr:429nt562); the natESM VILMA page; github.com/fesmc/vilma;
github.com/homnath/SPECFEMX.
