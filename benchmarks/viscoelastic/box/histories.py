"""Load histories S(t) of the box benchmarks, and their exact convolutions.

A history is a sum of PIECES, each

    w(s) = c0 + c1 s + a cos(omega s) + b sin(omega s),   s = t - start,

on [start, end) (end may be infinite), right-continuous, zero before the
first piece. The case file carries the pieces and the BREAKPOINTS (every
piece end and interior start, flagged when S jumps there); the driver
(viscoelastic_box.cpp) evaluates the same pieces, so the two sides cannot
disagree about the load.

Every reference of this sub-family is a sum of an instantaneous term and
relaxation modes, y(t) = R_inf S(t) + sum_i r_i (e^{p_i .} * S)(t), so the
one primitive needed is the exact convolution

    I(p, t) = int_0^t e^{p (t - u)} S(u) du,

which `convolve` evaluates piece by piece in closed form (phi-functions;
no quadrature), for real p <= 0 and, for the odd spurious pole of a
rational fit, complex p.

The presets (all times in the case's own unit, normally the shortest
Maxwell time) are the forcings of the stepper studies. Their breakpoints
sit at irrational times (sqrt 2, e, ...), off any uniform step grid of a
rational step, so that "aligned" and "unaligned" stepping differ:

  heaviside       S = 1 from t = 0
  ramp            linear rise over [0, T], then constant (a kink)
  smooth_step     raised cosine over [0, T], then constant (C^1)
  load_unload     1 on [0, T), 0 after (a jump at T: GIA-like removal)
  sawtooth        glacial cycles: linear growth over Tg, linear melt over
                  Tm, repeated (kinks only)
  periodic        mean + sin(omega t + phase), switched on at t = 0: an
                  initial transient decaying into the periodic state
  multi_sine      a sum of sines at incommensurate frequencies
  ramped_periodic a periodic load whose amplitude ramps on over [0, T]
                  (no switch-on jump; a smooth onset of the transient)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Piece:
    start: float
    end: float = math.inf
    c0: float = 0.0
    c1: float = 0.0
    a: float = 0.0
    b: float = 0.0
    omega: float = 0.0

    def to_json(self) -> dict:
        d = {"start": self.start,
             "end": None if math.isinf(self.end) else self.end}
        for k in ("c0", "c1", "a", "b", "omega"):
            v = getattr(self, k)
            if v != 0.0:
                d[k] = v
        return d

    def value(self, s):
        """w(s), s = t - start."""
        return (self.c0 + self.c1 * s + self.a * np.cos(self.omega * s)
                + self.b * np.sin(self.omega * s))


@dataclass
class History:
    name: str
    pieces: list[Piece] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    # --- evaluation ----------------------------------------------------------

    def __call__(self, t, left: bool = False):
        """S(t), right-continuous (left limit with left=True)."""
        t = np.asarray(t, dtype=float)
        v = np.zeros_like(t)
        for p in self.pieces:
            inside = ((p.start < t) & (t <= p.end) if left
                      else (p.start <= t) & (t < p.end))
            v = v + np.where(inside, p.value(t - p.start), 0.0)
        return v

    def breakpoints(self) -> list[dict]:
        """Piece boundaries in (0, inf), with whether S jumps there."""
        times = sorted({p.start for p in self.pieces if p.start > 0.0}
                       | {p.end for p in self.pieces
                          if math.isfinite(p.end)})
        out = []
        for t in times:
            jump = abs(float(self(t)) - float(self(t, left=True)))
            out.append({"time": t, "jump": bool(jump > 1e-14)})
        return out

    def to_json(self) -> dict:
        return {"name": self.name, "params": self.params,
                "pieces": [p.to_json() for p in self.pieces],
                "breakpoints": self.breakpoints()}

    @staticmethod
    def from_json(d: dict) -> "History":
        pieces = [Piece(start=p["start"],
                        end=math.inf if p.get("end") is None else p["end"],
                        c0=p.get("c0", 0.0), c1=p.get("c1", 0.0),
                        a=p.get("a", 0.0), b=p.get("b", 0.0),
                        omega=p.get("omega", 0.0))
                  for p in d["pieces"]]
        return History(d.get("name", "history"), pieces,
                       d.get("params", {}))

    # --- the convolution -----------------------------------------------------

    def convolve(self, p: complex, t) -> np.ndarray:
        """I(p, t) = int_0^t e^{p (t - u)} S(u) du, exactly, for every t."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.zeros(t.shape, dtype=complex)
        for piece in self.pieces:
            for j, tj in enumerate(t):
                if tj <= piece.start:
                    continue
                T1 = min(piece.end, tj) - piece.start
                tail = tj - piece.start - T1  # time since the piece ended
                out[j] += np.exp(p * tail) * _piece_integral(piece, p, T1)
        if np.all(np.abs(out.imag) <= 1e-300) and np.isrealobj(p):
            return out.real
        return out


def _phi1(z: complex) -> complex:
    """(e^z - 1) / z."""
    if abs(z) < 1e-5:
        return 1.0 + z / 2.0 + z * z / 6.0
    if isinstance(z, complex) or np.iscomplexobj(z):
        return (np.exp(z) - 1.0) / z
    return math.expm1(z) / z


def _phi2(z: complex) -> complex:
    """(e^z - 1 - z) / z^2."""
    if abs(z) < 1e-2:
        return (0.5 + z / 6.0 + z ** 2 / 24.0 + z ** 3 / 120.0
                + z ** 4 / 720.0 + z ** 5 / 5040.0)
    return (_phi1(z) - 1.0) / z


def _piece_integral(piece: Piece, p: complex, T: float) -> complex:
    """J = int_0^T e^{p (T - v)} w(v) dv for one piece."""
    if T <= 0.0:
        return 0.0
    z = p * T
    J = piece.c0 * T * _phi1(z) + piece.c1 * T * T * _phi2(z)
    if piece.a != 0.0 or piece.b != 0.0:
        # cos and sin through e^{+-i w v}: K(nu) = int_0^T e^{p(T-v)} e^{nu v}
        def K(nu: complex) -> complex:
            d = (nu - p) * T
            if abs(d) < 1e-8:
                return T * np.exp(p * T)
            return (np.exp(nu * T) - np.exp(p * T)) / (nu - p)
        w = piece.omega
        kp, km = K(1j * w), K(-1j * w)
        J += piece.a * 0.5 * (kp + km) + piece.b * (kp - km) / 2j
    if np.isrealobj(p) and not isinstance(p, complex):
        return complex(J).real
    return J


# --- presets -------------------------------------------------------------------


def heaviside() -> History:
    return History("heaviside", [Piece(0.0, c0=1.0)])


def ramp(T: float = math.sqrt(2.0)) -> History:
    return History("ramp", [Piece(0.0, T, c1=1.0 / T), Piece(T, c0=1.0)],
                   {"T": T})


def smooth_step(T: float = math.sqrt(3.0)) -> History:
    w = math.pi / T
    return History("smooth_step",
                   [Piece(0.0, T, c0=0.5, a=-0.5, omega=w),
                    Piece(T, c0=1.0)], {"T": T})


def load_unload(T: float = 2.0 * math.sqrt(2.0)) -> History:
    return History("load_unload", [Piece(0.0, T, c0=1.0)], {"T": T})


def sawtooth(grow: float = math.e, melt: float = 1.0 / math.pi,
             cycles: int = 2) -> History:
    pieces, t = [], 0.0
    for _ in range(cycles):
        pieces.append(Piece(t, t + grow, c1=1.0 / grow))
        pieces.append(Piece(t + grow, t + grow + melt, c0=1.0,
                            c1=-1.0 / melt))
        t += grow + melt
    return History("sawtooth", pieces,
                   {"grow": grow, "melt": melt, "cycles": cycles})


def periodic(omega: float = 2.0, phase: float = 0.3,
             mean: float = 0.0) -> History:
    # sin(omega t + phase) = sin(phase) cos(omega t) + cos(phase) sin(omega t)
    return History("periodic",
                   [Piece(0.0, c0=mean, a=math.sin(phase),
                          b=math.cos(phase), omega=omega)],
                   {"omega": omega, "phase": phase, "mean": mean})


def multi_sine(omegas=(1.0, math.sqrt(2.0) * 2.0, math.pi * 3.0),
               amplitudes=(1.0, 0.5, 0.25),
               phases=(0.3, 1.1, 2.0)) -> History:
    pieces = [Piece(0.0, a=A * math.sin(f), b=A * math.cos(f), omega=w)
              for w, A, f in zip(omegas, amplitudes, phases)]
    return History("multi_sine", pieces,
                   {"omegas": list(omegas), "amplitudes": list(amplitudes),
                    "phases": list(phases)})


def ramped_periodic(omega: float = 2.0, T: float = math.sqrt(5.0)) -> History:
    """sin(omega t) with its amplitude ramped linearly over [0, T]:
    (t/T) sin(omega t) is not a piece type, so the ramp is approximated by
    the smooth envelope 0.5(1 - cos(pi t / T)) sin(omega t), expanded into
    sines of omega and omega +- pi/T."""
    nu = math.pi / T
    # 0.5 sin(w t) - 0.25 [sin((w+nu) t) + sin((w-nu) t)] on [0, T)
    pieces = [Piece(0.0, T, b=0.5, omega=omega),
              Piece(0.0, T, b=-0.25, omega=omega + nu),
              Piece(0.0, T, b=-0.25, omega=omega - nu)]
    # sin(w t) after T, written about s = t - T
    pieces.append(Piece(T, a=math.sin(omega * T), b=math.cos(omega * T),
                        omega=omega))
    return History("ramped_periodic", pieces, {"omega": omega, "T": T})


PRESETS = {
    "heaviside": heaviside,
    "ramp": ramp,
    "smooth_step": smooth_step,
    "load_unload": load_unload,
    "sawtooth": sawtooth,
    "periodic": periodic,
    "multi_sine": multi_sine,
    "ramped_periodic": ramped_periodic,
}


def make(name: str, **params) -> History:
    if name not in PRESETS:
        raise SystemExit(f"unknown history {name!r}; "
                         f"one of {', '.join(PRESETS)}")
    return PRESETS[name](**params)


def _self_test() -> None:
    """The closed-form convolution against adaptive quadrature."""
    from scipy.integrate import quad

    for name in PRESETS:
        h = make(name)
        for p in (0.0, -0.3, -7.0, -1e3):
            for t in (0.37, 2.9, 7.3):
                pts = [bp["time"] for bp in h.breakpoints()
                       if bp["time"] < t]
                exact = h.convolve(p, t)[0]
                num, _ = quad(lambda u: math.exp(p * (t - u))
                              * float(h(u)), 0.0, t,
                              points=pts or None, limit=400,
                              epsabs=1e-14, epsrel=1e-13)
                err = abs(exact - num) / max(1e-300, abs(num), 1e-12)
                assert err < 1e-8, (name, p, t, exact, num)
    print("histories: convolutions agree with quadrature")


if __name__ == "__main__":
    _self_test()
