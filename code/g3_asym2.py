"""
Second genus-3 configuration, asymmetric -- selection by the criteria that
actually matter, not by an amoeba-hole census.

Why not the hole census: hole_tune locates holes by distance from r2 = 0, valid
only when C = A, which holds only for the symmetric family WV[1] = WV[0] -- the
very symmetry to be broken.  A general 2-D component tracker (amoeba_holes.py)
works but fragments holes and is fiddly to tune.

What is used instead:
  * genus 3 with 8 real branch points -- the curve is Harnack, so this already
    guarantees three bounded complement components, i.e. three gas phases;
  * min_i Im B_ii >= MINDIAG -- the discrete component must put visible mass on
    its neighbouring atoms;
  * facet sizes checked afterwards by a short sampling run, which is the ground
    truth anyway.

Symmetry breaking is done in the one direction that does it: WV[1] away from
WV[0].  Perturbing all sixteen weights leaves the gas regime almost immediately.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
import sys

import mpmath as mp
import numpy as np

from hyper_curve import Dz, curve
from hyper_g3 import period_matrix

MINDIAG = 0.50


def period_of(WU, WV, dps=30):
    mp.mp.dps = dps
    wu = [[mp.mpf(str(x)) for x in r] for r in WU]
    wv = [[mp.mpf(str(x)) for x in r] for r in WV]
    try:
        g, r0, interior = curve(wu, wv, 2, 4, verbose=False)
    except Exception:
        return None
    if g != 3 or len(r0) != 8:
        return None
    f = lambda t: mp.re(Dz(t, wu, wv, 2, 4))
    try:
        r = [mp.findroot(f, mp.mpf(str(x)), tol=mp.mpf('1e-50')) for x in r0]
        ImB = period_matrix(r, verbose=False, tol=1e-9)
    except Exception:
        return None
    d = [float(ImB[i, i]) for i in range(3)]
    b12, b23 = float(ImB[0, 1]), float(ImB[1, 2])
    return dict(ImB=ImB, diag=d, b12=b12, b23=b23,
                asym=abs(b12 - b23) / (abs(b12) + abs(b23) + 1e-12))


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    rng = np.random.default_rng(seed)
    W = np.load(os.path.join(HERE, 'g3_best_weights.npy'))
    U0, V0 = W[0].tolist(), W[1].tolist()

    r = period_of(U0, V0)
    print(f"symmetric start: asym {r['asym']:.4f}  B12={r['b12']:.4f} "
          f"B23={r['b23']:.4f}  diag {[round(x,3) for x in r['diag']]}", flush=True)
    best = (0.0, U0, V0, r)
    tried = ok = 0
    for t in range(steps):
        s = (0.10, 0.20, 0.35, 0.55)[t % 4]
        V = [list(V0[0]),
             [x * float(np.exp(rng.normal(0, s))) for x in V0[0]]]
        tried += 1
        rr = period_of(U0, V)
        if rr is None or min(rr['diag']) < MINDIAG:
            continue
        ok += 1
        if rr['asym'] > best[0]:
            best = (rr['asym'], U0, V, rr)
            print(f"  step {t:3d} (s={s}): asym {rr['asym']:.4f}  "
                  f"B12={rr['b12']:.4f} B23={rr['b23']:.4f}  "
                  f"diag {[round(x,3) for x in rr['diag']]}", flush=True)
            print(f"    WV = {[[round(x,5) for x in q] for q in V]}", flush=True)
    print(f"\n{ok}/{tried} perturbations gave a usable genus-3 curve")
    a, U, V, rr = best
    print(f"BEST asym {a:.4f}   B12={rr['b12']:.5f}  B23={rr['b23']:.5f}")
    print(f"  diag {[round(x,5) for x in rr['diag']]}")
    print("  Im B =")
    for i in range(3):
        print("    " + "  ".join(f"{float(rr['ImB'][i,j]):>12.7f}" for j in range(3)))
    print(f"  WU = {[[round(x,5) for x in q] for q in U]}")
    print(f"  WV = {[[round(x,5) for x in q] for q in V]}")
    np.save(os.path.join(HERE, 'g3b_weights.npy'), np.array([U, V], dtype=float))
    np.save(os.path.join(HERE, 'g3b_ImB.npy'),
            np.array([[float(rr['ImB'][i, j]) for j in range(3)] for i in range(3)]))
    print("  saved g3b_weights.npy, g3b_ImB.npy")
