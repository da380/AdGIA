"""A slide-grade 3-D render of a cap-load solve, through ParaView.

Run with pvbatch (not the benchmarks venv):

    pvbatch talk_render.py <case>/paraview_o2/solid/solid.pvd talk/field3d.png

Loads the solid's ParaView output from field_benchmark --paraview,
clips half the ball away, warps the surface by the displacement
(auto-exaggerated to ~15% of the radius) and colours it by |u| —
the off-axis cap load makes an asymmetric, recognisably 3-D picture.
"""
import sys

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
