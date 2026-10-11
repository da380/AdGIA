# Earth data

Python tools that exchange surface fields between real-Earth datasets
and AdGIA: pyslfp's ICE-NG ice histories and topography in, gridded
NetCDF and maps out. The scripts live here and run from the build tree,
in the style of `meshes/` and `benchmarks/`; more datasets will join
them as the need arises (tomographic models for conversion into
viscosity, say).

## Running

The build puts a launcher for each script in `<build>/earth_data/`
(`surface_to_netcdf`, `ice_ng_to_surface`, `topography_grid`): each
starts the script of the source tree with the Python found at
configuration — the shared poetry environment of the repository root, or
the one `EARTH_DATA_PYTHON` names. The environment is made once (see
INSTALL.md, "The Python environment"), and a build configured before it
existed is configured again so the launchers pick it up:

```
cd <repo>
poetry env use python3.12   # only if your python3 is older than 3.12
poetry install
cd <build> && cmake .       # regenerates the launchers
./earth_data/topography_grid --lmax 24
```

`--extras maps` on the install adds cartopy and pyshtools for the map
figures. When no environment is found at configuration the launchers
fall back on `python3` and refuse to run, with this recipe in their
message. A script writes where it is started (or to `-o`), so run the
launchers from the build tree; every script's `-h` and docstring give
its options, defaults and a sample run.

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
  plus a pre-processed shoreline (.npz, wrapped lat-lon grids):
  `coast_rad` is the great-circle distance to the coast of the
  near-native-resolution topography (its sign changes, plus the
  shallow shelves within `--band`, where the flotation criterion is
  delicate), a unit-gradient field that defines the shoreline evenly
  at cliff coasts and shelf seas alike; `meshes/earth_coastlines.py`
  refines on it directly — no polyline isolation. pyslfp downloads
  and caches the data on first use. The build's default mesh reads a
  committed grid instead, so this tool is only needed for a
  re-dated or re-banded one (`meshes/README.md`, "The coastline
  mesh").
