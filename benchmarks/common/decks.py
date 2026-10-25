"""The ladder models as decks: the bridge to the spectral codes.

sl3dpp reads tabulated radial models (planetmodel decks); pyslfp reads
the planetmodel models of `models.py` directly.  This module samples a
ladder model onto per-layer knots and writes the elastic deck format
the cross-code comparisons share -- `r rho vpv vsv vph vsh eta`, SI,
radius first -- the same format sl3dpp's stage-1 fixtures use, so one
file feeds both spectral codes while the finite-element side builds its
mesh from the model directly.

Every field of every ladder layer is a polynomial of degree at most
three (constants, linears, the PREM cubics) except aw_core's
closed-form vp, so the cubic (not-a-knot) spline a deck is read with
reproduces the model exactly from eight knots per layer; aw_core alone
comes back to spline accuracy.  `roundtrip_error` measures the worst
relative disagreement of model and deck, so a generation script can
state it in the deck's provenance.

Run as a script it writes `<name>.elastic.deck` for the named models
(all of them by default) into --out, which must lie outside the source
tree; committed copies of the decks, where a consumer wants them, are
build-tree regenerations copied in, like the committed reference
histories.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from planetmodel import Deck, DeckFormat, deck_layers, read_deck, write_deck

sys.path.insert(0, str(Path(__file__).resolve().parent))

import models  # noqa: E402
import outputs  # noqa: E402

def _full_precision(deck: Deck, names: tuple[str, ...]) -> list[str]:
    rows = np.column_stack([deck.radius] + [deck[n] for n in names])
    return [" ".join(f"{v:.17g}" for v in row) for row in rows]


#: The elastic deck format of the cross-code comparisons (sl3dpp's
#: stage-1 convention): no viscosity column, eta the anisotropy
#: parameter (one for the isotropic ladder).  Written at full precision
#: (planetmodel's default table writer keeps 8 digits), so that for the
#: polynomial layers the deck *is* the ladder model to rounding and all
#: three codes share one definition.
ELASTIC = DeckFormat(("rho", "vpv", "vsv", "vph", "vsh", "eta"),
                     name="elastic", write_table=_full_precision)

#: Knots per layer: enough for the not-a-knot cubic to reproduce a
#: cubic layer exactly, with headroom for aw_core's closed form.
KNOTS_PER_LAYER = 8

#: Deck column -> the names tried on a layer, in order (the ladder is
#: isotropic, so the TI columns fall back to the isotropic fields).
_FALLBACKS = {"rho": ("rho",), "vpv": ("vpv", "vp"), "vsv": ("vsv", "vs"),
              "vph": ("vph", "vpv", "vp"), "vsh": ("vsh", "vsv", "vs")}


def _column(layer, name: str, r: np.ndarray) -> np.ndarray:
    if name == "eta":
        for key in ("eta",):
            if key in layer:
                return np.asarray(layer[key](r), dtype=float)
        return np.ones(r.size)
    for key in _FALLBACKS[name]:
        if key in layer:
            return np.asarray(layer[key](r), dtype=float)
    raise KeyError(f"layer {layer.index} has none of {_FALLBACKS[name]}")


def as_deck(model, *, knots_per_layer: int = KNOTS_PER_LAYER) -> Deck:
    """`model` sampled per layer onto uniform knots, as an elastic deck
    in SI (repeated radii mark the internal boundaries)."""
    radius, columns = [], {name: [] for name in ELASTIC.columns}
    for layer in model.layers:
        lo, hi = layer.interval
        r = np.linspace(lo, hi, knots_per_layer)
        radius.append(r)
        for name in ELASTIC.columns:
            columns[name].append(_column(layer, name, r))
    return Deck(np.concatenate(radius),
                {n: np.concatenate(v) for n, v in columns.items()})


def write_model_deck(name: str, out: Path, *,
                     knots_per_layer: int = KNOTS_PER_LAYER) -> Path:
    """The ladder model `name` written as `<name>.elastic.deck` under
    `out`; returns the path written."""
    deck = as_deck(models.model(name), knots_per_layer=knots_per_layer)
    return write_deck(out / f"{name}.elastic.deck", deck, ELASTIC)


def roundtrip_error(name: str, path: Path, *,
                    samples_per_layer: int = 64) -> float:
    """The worst relative disagreement, over every deck column and
    layer, between the ladder model `name` and the deck at `path` read
    back the way a consumer reads it (cubic, not-a-knot)."""
    si = models.model(name)
    skeleton, layers = deck_layers(read_deck(path, ELASTIC), kind="cubic")
    worst = 0.0
    for ours, theirs in zip(si.layers, layers):
        lo, hi = ours.interval
        r = np.linspace(lo, hi, samples_per_layer + 2)[1:-1]
        for column in ELASTIC.columns:
            a = _column(ours, column, r)
            b = np.asarray(theirs[column](r), dtype=float)
            scale = np.max(np.abs(a))
            if scale > 0.0:
                worst = max(worst, float(np.max(np.abs(a - b)) / scale))
    return worst


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="write ladder models as elastic decks for sl3dpp")
    parser.add_argument("names", nargs="*", default=sorted(models.MODELS),
                        help="ladder models (default: all)")
    parser.add_argument("--out", type=Path, required=True,
                        help="output directory, outside the source tree")
    parser.add_argument("--knots", type=int, default=KNOTS_PER_LAYER,
                        help=f"knots per layer (default {KNOTS_PER_LAYER})")
    args = parser.parse_args(argv)
    out = outputs.outside_source(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        path = write_model_deck(name, out, knots_per_layer=args.knots)
        error = roundtrip_error(name, path)
        print(f"{path}  roundtrip {error:.3g}")


if __name__ == "__main__":
    main()
