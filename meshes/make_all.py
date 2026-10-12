"""Generate every mesh of the build (gmsh `.msh` and MFEM `.mesh`) by running
each script of SCRIPTS in turn.

    poetry run python make_all.py --out DIR  # e.g. a build tree's data/

The build does this itself through CMakeLists.txt; this is for running the
set by hand — for instance `--scale 0.5` for a higher-resolution version
of every mesh (forwarded to each script). SCRIPTS is the set the build
makes (CMakeLists.txt here), `--all` forms of the multi-mesh scripts
included: layered_earth.py and aspherical_body.py.

The unit ball and the three-layer 3D Earth take a minute or two each.
"""
import subprocess
import sys
from pathlib import Path

from common import parser

HERE = Path(__file__).resolve().parent

#: Each script with the arguments that produce the files in data/.
SCRIPTS = [
    ["homogeneous_body.py", "--preset", "disc"],
    ["homogeneous_body.py", "--preset", "disc-buffer"],
    ["homogeneous_body.py", "--preset", "disc-buffer-wide"],
    ["homogeneous_body.py", "--preset", "ball-buffer-coarse"],
    ["homogeneous_body.py", "--preset", "ball"],
    ["offset_disc.py"],
    ["layered_earth.py", "--all"],
    ["aspherical_body.py", "--all"],
    ["earth_coastlines.py", "--field", "earth"],
    ["earth_coastlines.py", "--field", "analytic"],
]


def main() -> None:
    args = parser(__doc__).parse_args()
    extra = ["--out", str(args.out), "--scale", str(args.scale)]
    extra += ["--verbose"] if args.verbose else []
    for script, *options in SCRIPTS:
        print(f"== {script} {' '.join(options)}", flush=True)
        subprocess.run([sys.executable, str(HERE / script), *options, *extra],
                       check=True)


if __name__ == "__main__":
    main()
