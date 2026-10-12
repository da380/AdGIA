"""A slide-grade 3-D render of a cap-load solve, through ParaView.

Run with pvbatch (ParaView's own Python), not in the benchmarks venv:

    pvbatch talk_render.py <case>/paraview_o2/solid/solid.pvd talk/field3d.png

Because pvbatch's interpreter has no access to the benchmark modules, the
output-location guard of common/outputs.py is not applied here: write the
image into the build tree (or outside the repository).

Loads the solid's ParaView output from field_benchmark --paraview,
clips half the ball away, warps the surface by the displacement
(auto-exaggerated to ~15% of the radius) and colours it by |u| —
the off-axis cap load makes an asymmetric, recognisably 3-D picture.
"""
import sys
from pathlib import Path

USAGE = "usage: pvbatch talk_render.py <solid.pvd> <output.png>"
if len(sys.argv) != 3:
    raise SystemExit(USAGE)
if not Path(sys.argv[1]).is_file():
    raise SystemExit(f"{sys.argv[1]}: no such file\n{USAGE}")
if not sys.argv[2].lower().endswith(".png"):
    raise SystemExit(f"{sys.argv[2]}: the output must be a .png file\n"
                     f"{USAGE}")
if not Path(sys.argv[2]).resolve().parent.is_dir():
    raise SystemExit(f"{sys.argv[2]}: its directory does not exist")

from paraview.simple import (Clip, ColorBy, GetActiveViewOrCreate,
                             GetColorTransferFunction, GetScalarBar,
                             Hide, PVDReader, Render, ResetCamera,
                             SaveScreenshot, Show, WarpByVector)

pvd, out = sys.argv[1], sys.argv[2]

reader = PVDReader(FileName=pvd)
reader.UpdatePipeline()
rng = reader.PointData["u"].GetRange(-1)  # the magnitude range
scale = 0.15 / max(rng[1], 1e-12)

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [2000, 1500]
view.UseColorPaletteForBackground = 0
view.Background = [1.0, 1.0, 1.0]
view.OrientationAxesVisibility = 0

clip = Clip(Input=reader)
clip.ClipType = "Plane"
clip.ClipType.Origin = [0.0, 0.0, 0.0]
clip.ClipType.Normal = [0.0, 1.0, 0.0]
clip.Invert = 0

warp = WarpByVector(Input=clip)
warp.Vectors = ["POINTS", "u"]
warp.ScaleFactor = scale

display = Show(warp, view)
ColorBy(display, ("POINTS", "u", "Magnitude"))
display.SetScalarBarVisibility(view, True)
lut = GetColorTransferFunction("u")
lut.ApplyPreset("Viridis (matplotlib)", True)
display.RescaleTransferFunctionToDataRange(True, False)
bar = GetScalarBar(lut, view)
bar.TitleColor = [0.0, 0.0, 0.0]
bar.LabelColor = [0.0, 0.0, 0.0]
bar.Title = "|u|"
bar.ComponentTitle = ""
Hide(reader, view)

ResetCamera(view)
camera = view.GetActiveCamera()
camera.Elevation(20)
camera.Azimuth(85)
camera.Zoom(1.2)
Render(view)
SaveScreenshot(out, view, ImageResolution=[2000, 1500])
print("wrote " + out)
