# Post-processing

Python tools for AdGIA outputs, a poetry project in the style of
`meshes/` and `benchmarks/` (`poetry install` here; add `--extras maps`
for cartopy and pyshtools).

- `surface_to_netcdf.py` — grids the nodal surface-field CSV written by
  `SeaLevelOperator::WriteSurfaceField` (sea level, ice, any surface
  scalar) onto a regular lon–lat array and writes NetCDF: the row/column
  convention matches pyshtools' DH grids (drop the south-pole row for
  `SHGrid.from_array`), and cartopy or any standard mapping stack takes
  the file directly. `--plot` writes a quick-look map (Robinson with
  coastlines when cartopy is installed). GLVis and ParaView have no
  cartographic projections; this is the route to map-quality figures.

- `ice_ng_to_surface.py` — the ingest leg: samples an ICE-5G/6G/7G
  field (ice thickness, topography or sea level) at the nodes of an
  exported surface CSV for a list of dates, and writes the time-stack
  CSV that `IceHistory` reads (one value column per time, the header
  naming each column by ascending model time). The data files are
  pyslfp's: its Zenodo downloader fetches and caches them in its own
  data directory on first use — nothing is bundled or copied here — and
  its `IceNG` loader does the per-date file resolution and time
  interpolation. `--length-scale` (metres per length unit) and
  `--time-scale` (ka per time unit) put the output in a run's own
  units. `examples/ice_age_loading.cpp` is the end-to-end chain.

- `topography_grid.py` — exports the lmax-truncated ICE-NG topography
  as a wrapped lat-lon grid (.npz) for mesh refinement:
  `meshes/earth_coastlines.py` turns it into a planetmodel `Refinement`
  that sizes elements by how close the sea level is to zero — no
  polyline isolation; shorelines and shallow shelves, where the
  flotation criterion is delicate, refine together. pyslfp downloads
  and caches the data on first use.
