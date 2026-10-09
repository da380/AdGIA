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
