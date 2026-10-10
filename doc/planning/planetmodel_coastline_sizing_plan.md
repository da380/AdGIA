# planetmodel: lateral (coastline) mesh refinement — plan

*Status: draft for David, 10 October 2026. This note is written to be
carried over to the planetmodel repository; the facts about planetmodel
below were read from its source at `~/dev/planetmodel` (the version the
AdGIA projects pin, `planetmodel[meshing,mfem] >= 1.2.3`). The last
section is the AdGIA-side consumer and stays here.*

## Why

The sea-level solver's boundary machinery (the water-load operator, the
shoreline Picard update, the ocean-masked integrals) lives on the outer
surface, and its error concentrates at coastlines: in the rung-1
fingerprint benchmark the AdGIA-vs-pyslfp difference is visibly
concentrated at the melted sector's shoreline, and the shoreline-
migration stop criterion is limited by the mesh's ability to resolve a
migrating strip. Radial refinement cannot help with this; what is
wanted is a surface mesh that is finer near coastlines and coarser in
the open ocean and continental interiors, with the refinement decaying
with depth so it does not drill fine columns through the planet.

## What planetmodel does today

Sizing is per-interface and purely radial (`mesh3d/_sizing.py`,
`mesh3d/spec.py`):

- A `SizingRule` is a callable mapping the computational domain's
  interfaces to one `InterfaceSizing(size, far_size, decay_width)` per
  interface index.
- `apply_size_fields` turns each into a gmsh `MathEval` field holding
  the closed-form distance `| |x - c| - R |` to that sphere, wrapped in
  a `Threshold` (size at the interface, `far_size` beyond
  `decay_width`), and combines all of them with a single `Min` set as
  the background field.
- `apply_mesh_options` deliberately switches off every competing gmsh
  size source (`MeshSizeExtendFromBoundary`, `FromPoints`,
  `FromCurvature`) so the background field is the single authority, and
  sets `Mesh.MeshSizeMin/Max`.
- `check_sizing_resolves_spans` / `check_sizing_scale` guard the radial
  sizing against untetrahedralisable thin shells and absolute-scale
  mistakes.
- The module docstring explicitly rejects gmsh's own `Distance` field
  for *interfaces*: distance to a point sampling of a surface ripples
  (zero at the samples, ~half the spacing between them), so the
  realised size along the interface is rougher and on average coarser
  than asked.

Nothing in the sizing depends on direction. That is the gap.

## Proposed design

### The spec object: a point-cloud feature, not a coastline

Add one new frozen dataclass to `mesh3d/spec.py`:

```python
@dataclass(frozen=True)
class FeatureSizing:
    """Refine towards a cloud of points on an interface.

    `directions`: (n, d) array of unit vectors; the points are placed at
    `directions * R_k` on interface k in the reference configuration.
    `interface`: the interface index the cloud sits on (default: outer).
    `size`, `far_size`, `decay_width`: the Threshold vocabulary, in the
    geometry's own lengths, exactly as for InterfaceSizing.
    """
```

and let `MeshSpec` carry `features: Sequence[FeatureSizing] = ()`.

Keeping the input a *point cloud on an interface* (rather than
"coastline", or an arbitrary size function) keeps planetmodel generic —
the same object refines towards an ice margin, a seismic source region,
or a station network — and keeps all geographic knowledge (pyslfp,
ocean functions, contour extraction) out of planetmodel. Points are
given as unit directions and planetmodel applies the interface radius,
because the mesh is always built in the reference configuration (the
mapping is applied at export), so the caller never needs to know the
drawn radius or worry about topography.

### The gmsh mechanism: `Distance`(points) + `Threshold` into the Min

For each `FeatureSizing`, add a gmsh `Distance` field with `PointsList`
= the scaled cloud, wrap it in a `Threshold` exactly like the radial
ones, and append it to the existing `Min`. Two properties make this the
right primitive:

1. **Depth decay comes free.** The 3-D Euclidean distance to a cloud on
   the surface grows with depth at the same rate as it grows laterally,
   so the refined region is a tube of radius ~`decay_width` around the
   coastline curve — precisely the wanted shape, with no separate
   depth-ramp machinery.
2. **Composition is already solved.** The `Min` over fields means a
   feature can only refine, never coarsen: the radial sizing (and its
   span/scale guards) remains a valid ceiling everywhere, so
   `check_sizing_resolves_spans` needs no change.

gmsh's `Distance` field evaluates point clouds through an ANN kd-tree,
so ~1e4–1e5 coastline points is cheap per evaluation. (API note: the
option is `PointsList` in current gmsh; it was `NodesList` long ago —
irrelevant at the versions the meshing extra pins.)

### The ripple objection, answered

The docstring's rejection of `Distance` is about *interfaces*, where
the asked size must hold uniformly along the whole surface and a
rippled, on-average-coarser size breaks the span guarantees. A feature
cloud only has to deliver "fine near, coarse far": the ripple amplitude
is ~half the point spacing, so the rule

> sample the feature at spacing ≤ `FeatureSizing.size`

makes the ripple sub-element and invisible. Enforce it softly: a
validation warning when the cloud's maximum nearest-neighbour gap
(cheap from the same kd-tree logic, or left to the caller) exceeds
`size`. The radial interfaces keep their closed-form `MathEval` route
unchanged.

### Threading through the existing plumbing

- `apply_mesh_options`: `size_min` passed by the builder must become
  `min(radial sizes, feature sizes)` so `Mesh.MeshSizeMin` does not
  silently floor the feature refinement.
- `check_sizing_scale`: extend to feature sizes (same absolute-scale
  sanity).
- **Manifest**: record each feature's count, `size/far_size/
  decay_width`, interface index, and a content hash of the directions
  array — not the cloud itself (it can be 1e5 points; the manifest is
  for provenance and change detection).
- **Serialisation/printing**: `FeatureSizing` holds an array, so it is
  not naturally `==`-comparable or printable like the shipped frozen
  rules; store directions as a read-only array and implement
  `__eq__`/`__repr__` via the hash + count. (Open question 2.)
- **2-D**: nothing special. A disc's "coastline" is a set of isolated
  shoreline points on the circle; the same `Distance` + `Threshold`
  works verbatim with (n, 2) directions. Both legs land in the same
  unit of work, as usual.
- **Element-count budget** (document in the docstring so sizes are
  chosen deliberately): the refined tube contributes roughly
  `L_coast · decay_width² / size³` extra tets (2-D:
  `L_coast · decay_width / size²` triangles, with `L_coast` the number
  of shoreline points × spacing). Global coastlines at `size` ≪ ocean
  `far_size` are where the mesh budget will actually go.

### Alternatives considered (recorded, not chosen)

- **`gmsh.model.mesh.setSizeCallback`** — an arbitrary Python
  `f(x) → size`, min'd with the field value inside the callback. Fully
  general (handles smooth size *fields*, not just curves), but it is a
  scalar Python call per evaluation point, and its interplay with the
  background field has to be verified per gmsh version. Keep as the
  generalisation if a smooth driver (e.g., sizing by ice thickness
  gradient) is ever wanted.
- **Sampled background field** (`PostView` or `Structured` field from a
  precomputed grid) — the performance fallback if a pathological cloud
  (≫1e5 points) ever makes the ANN route slow. More plumbing, sampling
  resolution to choose; not needed at coastline scales.

### Acceptance gates (planetmodel-side tests)

1. **Synthetic feature, measured ramp**: a great-circle (3-D) / two
   antipodal points (2-D) cloud on a homogeneous ball; measured surface
   edge lengths near the feature ≈ `size`, far from it ≈ the radial
   sizing, the transition within the `Threshold` ramp's tolerance.
2. **No-feature equivalence**: `features=()` reproduces today's meshes
   (same field graph; ideally bit-identical .msh on a fixed seed).
3. **Min property**: with a coarse feature (`size` ≥ radial size) the
   mesh is unchanged — features never coarsen.
4. **Quality**: minSICN stays above `QUALITY_FLOOR` on the refined
   meshes; the high-order optimiser path exercised once.
5. **Depth decay**: element sizes at depth > `decay_width` under the
   coastline match the radial-only mesh.
6. Both legs (2-D and 3-D) for each gate; a timing note at ~1e5 points.

## The AdGIA-side consumer (stays in this repo)

A `meshes/` script (pyslfp added to `meshes/pyproject.toml`, probably
as an optional extra alongside the existing `planetmodel[meshing,mfem]`
dependency) builds the cloud:

1. Ocean function from pyslfp: present-day topography via its data
   machinery (ice-7G topography or ETOPO; `ensure_data` downloads and
   caches), `C = (topography < 0)` on the DH grid — or the analytical
   super-Gaussian states for the benchmark meshes, so the benchmark
   coastline and the benchmark load come from the same functions.
2. Extract the `C = 1/2` contour on the lat-lon grid (matplotlib's
   contour generator or `skimage.measure.find_contours`; longitude wrap
   handled by padding), resample each contour polyline to spacing ≤ the
   target `size`, map to unit vectors.
3. Pass as `FeatureSizing(directions, interface=outer, size, far_size,
   decay_width)` through the house mesh scripts; sizes chosen from the
   element-count budget above.

The rung-1/rung-2 fingerprint benchmarks then get a
coastline-refined mesh axis (same solve, fewer DOFs for the same
shoreline resolution), and the shoreline-migration stop criterion gets
the strip resolution it is currently starved of.

## Open questions for David

1. **Version/placement**: next planetmodel minor (1.3.0)? And is
   `mesh3d/spec.py` + `_sizing.py` the right home, or do you want a
   separate `features` module given the cloud-building helpers that may
   accrete around it?
2. **Array-in-frozen-dataclass convention**: hash-based equality as
   proposed, or do you prefer the directions passed as a plain tuple of
   tuples to keep the shipped-rules serialisation story exact?
3. **Validation strictness**: warn or refuse when the cloud's point
   spacing exceeds `size`?
4. Should the soft warning also check that the cloud actually lies on
   the named interface (|direction| = 1 is cheap; "on the interface the
   caller meant" is not knowable) — or is unit-norm enough?
