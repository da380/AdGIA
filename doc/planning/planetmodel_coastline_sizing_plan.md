# Lateral mesh refinement near sea level: settled record

*Status: **delivered** — planetmodel v1.2.4 (the library side) and the
AdGIA consumers below. Nothing is open; this file records what was
decided and where it lives.*

## What planetmodel provides (v1.2.4)

Sizing in `planetmodel.mesh3d` was purely radial (per-interface
closed-form distance-to-sphere fields combined by a `Min` background
field). v1.2.4 added lateral refinement through a **`Refinement`
protocol**: any callable size field over the reference domain with
`size`/`far_size` bounds, composed into the sizing by minimum (so a
refinement can only refine, and the radial span/scale guards stand).
Refinements go through gmsh's size callback, with the minimum taken in
one place. Two shipped kinds: `NearPoints` (KD-tree point cloud) and
`near_curve` (a polyline resampled to spacing ≤ `size` by construction,
which keeps the distance ripple inside the ramp's plateau). The ramp
holds the size flat for one `size` of distance before growing — against
a ramp of gradient g the mesher realises only about size/(1 − g/2), so
the plateau is what makes the asked size real. `MeshSpec.refinements`,
the manifest (digests, not point clouds), `Mesh.MeshSizeMin` and the
scale check are all threaded.

## The AdGIA consumers

- **Benchmark shoreline rings** — `benchmarks/common/make_case.py
  --refine-rings` (colatitude rings via `near_curve`) and
  `benchmarks/sea_level/run.py --coast`, which computes the analytic
  state's shoreline colatitude and builds the refined case in its own
  `h<h>_coast` directory. Measured at h 0.4/o2: the global comparison
  is unchanged (spectral/FE floor), the migration loop converges in 1
  pass instead of 2, and the migration-effect delta drops 4× —
  shoreline-quadrature noise removed.
- **Real geography** — no coastline isolation at all: size by the sea
  level itself. `postprocess/topography_grid.py` exports the
  lmax-truncated ICE-NG topography (truncation is what smooths the
  shorelines to a mesh-followable resolution), and
  `meshes/earth_coastlines.py` wraps it as a `Refinement` field with
  effective distance `max(|topo|/band, |r−1|/depth_width)` — real
  shorelines and the shallow shelves, where the flotation criterion is
  delicate, refine together, with the band (metres of topography) and
  the element sizes as the only knobs. Laptop defaults build
  `data/earth_coastlines.msh` at ~3 k tetrahedra (19 % of the surface
  inside a 400 m band); the server runs shrink `--coast-size`.
  Consumed by `examples/ice_age_loading.cpp` through `-m`.

The delivered design elaborates the original proposal in four ways —
the protocol in place of a point-cloud class, `near_curve`'s
constructive spacing, the ramp plateau, and the topography field in
place of extracted coastlines — each strictly simpler or more general
than what was first planned.
