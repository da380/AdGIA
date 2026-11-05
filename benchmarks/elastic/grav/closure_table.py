"""Collate the closed-E2 sweep: convert every Dahlen results file of the
fluid-core models (naked and bordered, orders 2/3) to exchange files and
tabulate degree 0 and the worst l>=1 entry against the pyslfp reference."""
import json, subprocess, sys
from pathlib import Path
import numpy as np

P = "/home/david/.cache/pypoetry/virtualenvs/adgia-python--ifwnOFe-py3.12/bin/python"
SRC = Path("/home/david/dev/AdGIA/benchmarks")
B = Path("/home/david/dev/AdGIA/build_parallel/benchmarks")
RUNS = B / "elastic/grav/runs"
sys.path.insert(0, str(SRC / "common"))
import exchange  # noqa

def read(path):
    _, _, table = exchange.read_table(path) if hasattr(exchange, "read_table") else exchange.read(path)
    return table

def load(path):
    rows = [l.split() for l in Path(path).read_text().splitlines()
            if l and not l.startswith("#") and l[0].isdigit()]
    return {int(r[0]): [float(x) for x in r[1:]] for r in rows}

models = sys.argv[1:] or ["fluid_core", "aw_core", "inner_core", "stratified_core",
                           "stable_core_mild", "stable_core", "stable_core_strong"]
cols = ["hp", "lp", "kp", "hT", "lT", "kT"]
print(f"{'model':18s} {'h':>4s} {'o':>2s} {'lb':>3s} {'h0_ref':>10s} {'h0_adgia':>10s} {'dh0':>9s} {'k0_adgia':>10s} "
      f"{'worst l>=1 (chan)':>18s} {'hp1':>8s} {'hp2':>8s} {'s':>7s}")
for model in models:
    ref = load(RUNS / f"e2_{model}_pyslfp.txt")
    case_dirs = sorted((RUNS / "love" / model).glob("h*")) + sorted((B / "love_numbers/runs" / model).glob("h*"))
    for case in case_dirs:
        h = case.name[1:]
        for res in sorted(case.glob("results_o*.json")):
            r = json.loads(res.read_text())
            if r.get("method") != "dahlen" or r.get("combined"):
                continue
            lb = int(r.get("level_border", 0))
            if r.get("level_monolithic"):
                continue
            out = RUNS / f"e2_{model}_adgia_dahlen{'_lb'+str(lb) if lb else ''}_o{r['order']}_h{h}.txt"
            subprocess.run([P, str(SRC / "elastic/grav/from_results.py"), str(res), "--out", str(out)],
                           check=True, capture_output=True)
            a = load(out)
            d0 = [d for d in r["degrees"] if d["degree"] == 0][0]["load"]
            h0 = d0["h"]; k0 = d0["k"]
            dh0 = abs(h0 - ref[0][0]) / abs(ref[0][0])
            worst = (0.0, "")
            for l in sorted(a):
                if l < 1 or l not in ref:
                    continue
                for j, c in enumerate(cols):
                    x, y = a[l][j], ref[l][j]
                    if np.isnan(x) or np.isnan(y) or y == 0:
                        continue
                    e = abs(x - y) / abs(y)
                    if e > worst[0]:
                        worst = (e, f"{c}{l}")
            hp = lambda l: abs(a[l][0] - ref[l][0]) / abs(ref[l][0])
            secs = sum(d["load"].get("seconds") or 0 for d in r["degrees"]) + r.get("setup_seconds", 0)
            print(f"{model:18s} {h:>4s} {r['order']:>2d} {lb:>3d} {ref[0][0]:>10.5f} {h0:>10.5f} {dh0:>9.2e} {k0:>10.2e} "
                  f"{worst[0]:>9.2e} ({worst[1]:>5s}) {hp(1):>8.1e} {hp(2):>8.1e} {secs:>7.1f}")
