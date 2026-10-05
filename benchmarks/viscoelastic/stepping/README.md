# Viscoelastic: the stepping survey

`survey.py` asks which integrator of `ViscoelasticOperator` is worth
using, and where, by driving `examples/viscoelastic_schemes` (a clamped
beam, `data/beam-quad.mesh`, order 2, a Maxwell body under a periodic
pull; every integrator; a cost-to-target search) over two axes:

- **stiffness** (`--contrasts`, default 1 10 100 1000): a second Maxwell
  branch with tau / r, the relaxation-time contrast r = tau_max/tau_min
  standing in for laterally varying viscosity. The explicit stability
  limit dt < ~2.8 tau_min binds; the implicit and exponential schemes do
  not notice it.
- **regime** (`--periods`, default 0.13 0.42 1.3 4.2 13 42, in units of
  tau): the load period over the relaxation time (a Deborah number).

For each point and scheme the example finds the coarsest step (halving
from one step per tau), or the loosest adaptive tolerance, that reaches
each target error (`--targets`, default 1e-3) of the displacement
history — the worst relative error over t_final (`--t-final`, default 4)
and four interior checkpoints — against RK4 at a small step. The load
runs with a phase offset and periods incommensurate with the checkpoints,
so that no scheme can score stroboscopically. Cost is counted in elastic
solves, the unit that transfers to the 3-D self-gravitating problems,
where one solve is one quasi-static system.

```
cd <build>/benchmarks/viscoelastic/stepping
./survey                                   # writes into survey/
./survey --targets 1e-2 1e-3 --contrasts 1 10 100
./survey --out <dir> --figures-only        # redraw from the stored runs
```

Outputs, in `--out` (default `survey/`, relative to where it is started):
one JSON file per point (`stiffness_r<r>.json`, `regime_tp<Tp>.json`;
existing ones are skipped) and the figures `ve_stiffness.png` (solves to
the target against the contrast) and `ve_regimes.png` (against the load
period). By default the example is taken from the first `build*` directory of
the repository that has it; `--program` names another binary.
`--dry-run` prints the commands.

The results are in `doc/benchmarks.tex` ("The stepping survey"), and the
conclusions for choosing a scheme in `doc/viscoelasticity.md` ("Choosing
a scheme"). The example also has a power-law branch (`-gamma`) and a
long-term modulus (`-mu-inf`), which the survey does not sweep.
