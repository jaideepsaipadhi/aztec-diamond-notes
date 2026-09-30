"""
Choose (2,4)-periodic weights for which the genus-3 discrete component is
actually MEASURABLE.  Two requirements pull against each other:

  * the three gas facets must be wide enough to spatially average over, which
    wants LARGE amoeba holes;
  * the discrete component must put visible mass on its neighbouring atoms,
    P(+-1)/P(0) = exp(-pi(1-2e)/B_ii), which wants LARGE diagonal B_ii.

A wide hole is a strongly gas-like phase, hence a rigid one, hence a SMALL B_ii.
The first tuning (maximise the smallest hole) drove the holes to ~1.8 and B_ii
down to 0.39, giving P(0,0,0) = 0.983 -- three facets, none measurable.

So: require every bounded hole to be at least MINHOLE wide (enough facet at
n ~ 128-192) and then maximise the smallest B_ii.  The hole test is cheap
float64; the period matrix is computed only for candidates that pass it.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
import time

import mpmath as mp
import numpy as np

from hole_tune import holes
from hyper_curve import Dz, curve
from hyper_g3 import period_matrix

MINHOLE = 0.50


def diag_B(WU, WV, dps=30):
    mp.mp.dps = dps
    wu = [[mp.mpf(x) for x in r] for r in WU]
    wv = [[mp.mpf(x) for x in r] for r in WV]
    g, r0, interior = curve(wu, wv, 2, 4, verbose=False)
    if g != 3 or len(r0) != 8:
        return None, None
    f = lambda t: mp.re(Dz(t, wu, wv, 2, 4))
    r = [mp.findroot(f, mp.mpf(str(x)), tol=mp.mpf('1e-50')) for x in r0]
    try:
        ImB = period_matrix(r, verbose=False, tol=1e-9)
    except Exception:
        return None, None
    return ImB, [float(ImB[i, i]) for i in range(3)]


def evaluate(WU, WV):
    h = holes(WU, WV)
    if len(h) != 3:
        return None
    hw = [b - a for a, b in h]
    if min(hw) < MINHOLE:
        return dict(ok=False, holes=hw)
    ImB, d = diag_B(WU, WV)
    if d is None:
        return dict(ok=False, holes=hw)
    return dict(ok=True, holes=hw, diag=d, ImB=ImB)


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    rng = np.random.default_rng(seed)

    known = [
        ("baseline", [[1.0, 0.55, 1.0, 0.55], [0.8, 1.3, 0.8, 1.3]],
         [[0.6, 1.2, 0.9, 1.4], [0.6, 1.2, 0.9, 1.4]]),
        ("hole-tuned",
         [[0.73362, 0.40972, 0.43319, 1.09429], [0.46608, 0.58536, 0.74074, 0.46156]],
         [[0.42979, 2.15644, 0.88992, 0.58789], [0.42979, 2.15644, 0.88992, 0.58789]]),
    ]
    best = None
    for name, WU, WV in known:
        t0 = time.time()
        r = evaluate(WU, WV)
        if r and r["ok"]:
            print(f"{name:<12} holes {[round(x,2) for x in r['holes']]}  "
                  f"diag B {[round(x,4) for x in r['diag']]}  "
                  f"min diag {min(r['diag']):.4f}  [{time.time()-t0:.0f}s]",
                  flush=True)
            if best is None or min(r["diag"]) > best[0]:
                best = (min(r["diag"]), WU, WV, r)
        else:
            print(f"{name:<12} rejected: holes "
                  f"{[round(x,2) for x in (r or {}).get('holes',[])]}", flush=True)

    for t in range(trials):
        WU = [[float(np.exp(rng.uniform(-0.8, 0.8))) for _ in range(4)]
              for _ in range(2)]
        WV = [[float(np.exp(rng.uniform(-0.8, 0.8))) for _ in range(4)]
              for _ in range(2)]
        WV[1] = list(WV[0])
        r = evaluate(WU, WV)
        if r and r["ok"] and (best is None or min(r["diag"]) > best[0]):
            best = (min(r["diag"]), WU, WV, r)
            print(f"  trial {t:3d}: min diag B = {min(r['diag']):.4f}  "
                  f"diag {[round(x,3) for x in r['diag']]}  "
                  f"holes {[round(x,2) for x in r['holes']]}", flush=True)
            print(f"    WU = {[[round(x,5) for x in q] for q in WU]}", flush=True)
            print(f"    WV = {[[round(x,5) for x in q] for q in WV]}", flush=True)

    if best:
        s, WU, WV, r = best
        print(f"\nBEST min diag B = {s:.4f}")
        print(f"  diag  {[round(x,5) for x in r['diag']]}")
        print(f"  holes {[round(x,3) for x in r['holes']]}")
        print(f"  WU = {[[round(x,5) for x in q] for q in WU]}")
        print(f"  WV = {[[round(x,5) for x in q] for q in WV]}")
        np.save(os.path.join(HERE, 'g3_best_weights.npy'),
                np.array([WU, WV], dtype=float))
        np.save(os.path.join(HERE, 'g3_best_ImB.npy'),
                np.array([[float(r['ImB'][i, j]) for j in range(3)]
                          for i in range(3)]))
        print("  saved g3_best_weights.npy, g3_best_ImB.npy")
