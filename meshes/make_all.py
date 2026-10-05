"""Generate the gmsh meshes by running each script of SCRIPTS in turn.

    poetry run python make_all.py --out DIR  # e.g. a build tree's data/

The build does this itself through CMakeLists.txt; this is for running the
set by hand. SCRIPTS leaves out disc_with_wide_buffer.py and
aspherical_body.py, which the build runs; run those two by hand as well
for the full set.

The unit ball and the three-layer 3D Earth take a minute or two each.
"""
import subprocess
import sys
from pathlib import Path

from common import parser

HERE = Path(__file__).resolve().parent

#: Each script with the arguments that produce the files in data/.
SCRIPTS = [
    ["unit_disc.py"],
    ["offset_disc.py"],
    ["disc_with_buffer.py"],
    ["ball_with_buffer.py"],
    ["layered_earth.py", "--all"],
    ["unit_ball.py"],
]


def main() -> None:
    args = parser(__doc__).parse_args()
    extra = ["--out", str(args.out)] + (["--verbose"] if args.verbose else [])
    for script, *options in SCRIPTS:
        print(f"== {script} {' '.join(options)}", flush=True)
        subprocess.run([sys.executable, str(HERE / script), *options, *extra],
                       check=True)


if __name__ == "__main__":
    main()
