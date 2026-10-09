# The sea-level fingerprint benchmark

AdGIA's monolithic sea-level solve (`SetWaterLoad`: the sea-level
equation folded into the elastic operator, frozen shorelines, no
rotation) against pyslfp's pseudo-spectral solution of the same problem
— rung 1 of `doc/planning/sea_level_plan.md`.

One spherically layered model (homogeneous by default), one smooth
analytical state (polar continent, ice cap, melt), defined once in
`run.py` and handed to both sides; pyslfp computes its own Love numbers
from the same planetmodel model (`EarthModel.from_planet_model`), so
Love-number agreement is assumed here and tested by the Love-number
family instead. `run.py` builds the case, runs `sea_level_benchmark`
under MPI, runs pyslfp, converts both fingerprints to metres, grids the
FE nodes onto pyslfp's grid and writes `report_o<p>.json`: ocean-RMS
difference and per-degree amplitudes.

Caveats recorded with the machinery: low-resolution runs need not agree
closely (the resolved comparison belongs to the server campaign), and
the degree-1 reference-frame convention between the two sides is not
yet reconciled — the report prints that row with a flag.

    cd <build>/benchmarks/sea_level
    ./run --h 0.4 --order 2 --np 4 --lmax 32
