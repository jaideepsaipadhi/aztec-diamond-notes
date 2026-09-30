"""
Bounded complement components (gas phases) of the amoeba, WITHOUT assuming any
symmetry.

hole_tune.py locates holes by min_z |log|w(z)||, i.e. by distance from r2 = 0.
That is only valid when the holes actually lie on r2 = 0, which follows from
w+ w- = C/A = 1 -- and C = A only because the first weight family satisfies
WV[1] = WV[0].  Break that symmetry (which is the whole point of a second test
configuration) and the detector reports zero holes for a model that plainly has
three: a 5% weight change appeared to destroy three gas phases of width 0.64.

General method.  For mv = 2 the curve is quadratic in w, so at fixed |z| = e^{r1}
the amoeba slice is exactly

        S(r1) = { log|w+(z)| } U { log|w-(z)| },   |z| = e^{r1},

two closed curves' worth of values.  The complement of S(r1) in R is a union of
intervals; the BOUNDED ones are the gas phases at that r1.  Sampling z densely
and looking for gaps in the sorted values finds them with no symmetry assumed --
and correctly reproduces the symmetric case.

A hole of the 2-D amoeba is a connected family of such gaps over an r1-interval;
we report, per r1, the gaps found, and then group them.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np

from hole_tune import quad_np


def slice_gaps(r1, wu, wv, mu=2, mv=4, N=400, gap_tol=None):
    """Bounded gaps of the amoeba slice at |z| = e^{r1}, as (lo, hi) in r2."""
    th = 2 * np.pi * np.arange(N) / N
    zs = np.exp(r1) * np.exp(1j * th)
    vals = []
    for z in zs:
        A, B, C = quad_np(z, wu, wv, mu, mv)
        d = np.sqrt(B * B - 4 * A * C + 0j)
        for s in (1, -1):
            w = (-B + s * d) / (2 * A)
            if w != 0 and np.isfinite(abs(w)):
                vals.append(np.log(abs(w)))
    if len(vals) < 4:
        return []
    v = np.sort(np.array(vals))
    # a genuine gap must be much wider than the local sampling spacing
    if gap_tol is None:
        span = v[-1] - v[0]
        gap_tol = max(8.0 * span / len(v), 1e-3)
    out = []
    for i in range(len(v) - 1):
        if v[i + 1] - v[i] > gap_tol:
            out.append((v[i], v[i + 1]))
    return out


def holes2d(wu, wv, mu=2, mv=4, lo=-6.0, hi=6.0, step=0.06, N=300):
    """Group per-r1 gaps into 2-D bounded components.  Returns a list of dicts
    with the r1-range, the r2-range, and the maximum r2-width."""
    rs = np.arange(lo, hi + 1e-9, step)
    per = {}
    for r1 in rs:
        per[float(r1)] = slice_gaps(float(r1), wu, wv, mu, mv, N=N)

    comps = []          # each: dict(r1s=[], r2mid=[], width=[])
    for r1 in rs:
        for (a, b) in per[float(r1)]:
            mid = 0.5 * (a + b)
            placed = False
            for c in comps:
                if abs(c["r1s"][-1] - r1) <= 1.5 * step and \
                        abs(c["r2mid"][-1] - mid) < 0.5 * (c["width"][-1] + (b - a)) + 0.25:
                    c["r1s"].append(float(r1))
                    c["r2mid"].append(mid)
                    c["width"].append(b - a)
                    placed = True
                    break
            if not placed:
                comps.append(dict(r1s=[float(r1)], r2mid=[mid], width=[b - a]))

    out = []
    for c in comps:
        r1lo, r1hi = min(c["r1s"]), max(c["r1s"])
        bounded = r1lo > lo + step / 2 and r1hi < hi - step / 2
        out.append(dict(r1=(r1lo, r1hi), r1width=r1hi - r1lo,
                        r2mid=float(np.mean(c["r2mid"])),
                        r2width=float(max(c["width"])),
                        bounded=bounded, n=len(c["r1s"])))
    out = [c for c in out if c["bounded"] and c["n"] >= 3]
    out.sort(key=lambda c: c["r1"][0])
    return out


if __name__ == "__main__":
    import mpmath as mp
    W = np.load(os.path.join(HERE, 'g3_best_weights.npy'))
    U0, V0 = W[0].tolist(), W[1].tolist()
    print("=== symmetric configuration (must reproduce 3 holes of r1-width 0.64) ===")
    for c in holes2d(U0, V0):
        print(f"   r1 in [{c['r1'][0]:+.2f},{c['r1'][1]:+.2f}] width {c['r1width']:.2f}"
              f"   r2 centre {c['r2mid']:+.3f} width {c['r2width']:.3f}")

    print("\n=== symmetry broken: WV[1] = WV[0] * exp(delta) ===")
    rng = np.random.default_rng(9)
    for s in (0.05, 0.10, 0.20):
        V = [list(V0[0]), [x * float(np.exp(rng.normal(0, s))) for x in V0[0]]]
        cs = holes2d(U0, V)
        print(f"  sigma {s}: {len(cs)} bounded components")
        for c in cs:
            print(f"     r1 [{c['r1'][0]:+.2f},{c['r1'][1]:+.2f}] w {c['r1width']:.2f}"
                  f"   r2 centre {c['r2mid']:+.3f} w {c['r2width']:.3f}")
