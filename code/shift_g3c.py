"""
The shift's velocity at genus 3 -- is v rational?

Equation (19) of Berggren-Nicoletti says the shift advances linearly on the
Jacobian,  e^{(kN)} = e^{(0)} + N v,  and the paper's headline is that in general
this never closes up: it "evolves quasi-periodically in R^g/Z^g".  Rational v
gives a finite cycle (Section 4.6 is v = 1/4, hence Remark 4.18's four
alignments and nothing more); irrational v gives genuine quasi-periodicity, and
nobody has watched it.

Validated at genus 1 by shift_vs_n.py: the measured advance was
0.2503 +- 0.0026 against the v = 1/4 that Remark 4.18 forces, with e cycling
0.246 -> 0.502 -> 0.743 -> 0.002.  So the method measures what (19) describes.

TWO ARTIFACTS had to be removed first, both of which would have manufactured a
convincing fake of the phenomenon:

1. The facet mask in g3_measure.find_facets thresholds on median(V) over all
   faces, which is 3.71 at n = 96 while the facets sit at 1.7-4.3.  The
   threshold cuts through the signal, so facets drop in and out as n moves the
   median -- sizes bounced between 25 and 510 with three collapsing to zero.
   g3_facets.detect uses slope plateaus and connectivity instead.

2. The Thurston height's global orientation flips with the parity of n (even n
   has slope -4 at negative y, odd n has +4), so h -> -h, Z -> -Z and e -> -e
   between consecutive n, and labelling facets by slope target swaps facets 1
   and 3 every step.  g3_facets.detect_oriented orders facets by position and
   returns the sign to put every n in one frame.

What is fitted: three shifts per n, with Im B held FIXED at the spectral curve's
value.  No reference to (19) and no identification of the angles.
"""
import itertools
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

import numpy as _np
import g3_sample as GS
_W = _np.load('/home/claude/g3c_weights.npy')
GS.WU = _W[0].tolist()
GS.WV = _W[1].tolist()
from g3_facets import detect_oriented
from g3_measure import height_profile, tree_arrays
from spatial_avg import STEP, bfs_tree, coeffs_for, face_graph

HERE = os.path.dirname(os.path.abspath(__file__))
PTS = list(itertools.product(range(-2, 3), repeat=3))


def measure(n, K, seed, Kprof=400):
    idx, E, F, steps, adj = face_graph(n)
    root, parent, order = bfs_tree(adj)
    pidx, pe, ps, pp = tree_arrays(order, parent)
    levels, _, EE = GS.build_levels(n)
    nE = len(EE)

    Hp = height_profile(n, levels, E, order, pidx, pe, ps, pp, Kprof, seed - 1)
    V, Mn = Hp.var(0), Hp.mean(0)
    facets, sigma, bad = detect_oriented(order, pidx, V, Mn)
    if facets is None:
        return None, None, None

    co = []
    for sel, info in facets:
        A, c = coeffs_for(sel, parent, order, nE)
        co.append((len(sel), A, c))

    Z = np.zeros((K, 3))
    rng = np.random.default_rng(seed)
    for i in range(K):
        M = GS.sample(n, levels, nE, rng)
        for k, (nS, A, c) in enumerate(co):
            Z[i, k] = (A - 4 * (c @ M)) / nS
    Z *= sigma                                   # one frame for every n
    Zc = (Z - np.median(Z, 0)) / STEP
    R = np.round(Zc)
    R = (R - np.round(np.median(R, 0))).astype(int)
    obs = {tuple(int(x) for x in a): int(b)
           for a, b in zip(*np.unique(R, axis=0, return_counts=True))}
    return obs, sigma, [len(s) for s, _ in facets]


def fit_shift3(obs, ImB):
    Q = np.linalg.inv(ImB)
    items = list(obs.items())

    def nll(e):
        w = np.array([np.exp(-np.pi * ((np.array(v) - e) @ Q @ (np.array(v) - e)))
                      for v in PTS])
        w = w / w.sum()
        p = dict(zip(PTS, w))
        return -sum(c * np.log(max(p.get(k, 1e-300), 1e-300)) for k, c in items)

    best = None
    for st in itertools.product((0.0, 0.3, -0.3), repeat=3):
        r = minimize(nll, np.array(st), method='Nelder-Mead',
                     options=dict(xatol=3e-4, fatol=3e-3, maxiter=1500))
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def run(ns, K=6000, seed=20261020):
    ImB = np.load(os.path.join(HERE, 'g3c_ImB.npy'))
    print(f"genus 3, configuration C (different weights, different period matrix), K = {K} per n")
    print(f"Im B diag = {np.round(np.diag(ImB),4)};  three shifts fitted, "
          f"Im B fixed\n")
    print(f"{'n':>4} {'sigma':>6} {'facet sizes':>20}   "
          f"{'e1':>8} {'e2':>8} {'e3':>8}   time")
    out = []
    for n in ns:
        t0 = time.time()
        obs, sigma, sizes = measure(n, K, seed + 131 * n)
        if obs is None:
            print(f"{n:>4} {'--':>6} {'detection failed':>20}", flush=True)
            continue
        e = fit_shift3(obs, ImB)
        out.append((n, sigma, e))
        print(f"{n:>4} {sigma:>+6d} {str(sizes):>20}   "
              f"{e[0]:+8.4f} {e[1]:+8.4f} {e[2]:+8.4f}   "
              f"[{time.time()-t0:.0f}s]", flush=True)
        np.save(os.path.join(HERE, 'g3c_shift_vs_n.npy'),
                np.array([[a] + [b] + list(c) for a, b, c in out]))
    return out


def analyse(out):
    if len(out) < 4:
        print("\n  too few points")
        return
    print("\n  advance per step (mod 1), consecutive n only:")
    ds = []
    for (n0, _, e0), (n1, _, e1) in zip(out, out[1:]):
        if n1 - n0 != 1:
            continue
        d = (np.array(e1) - np.array(e0)) % 1.0
        ds.append(d)
        print(f"    {n0}->{n1}:  " + "  ".join(f"{x:.4f}" for x in d))
    if not ds:
        return
    ds = np.array(ds)
    # circular mean, since each component lives on R/Z
    ang = 2 * np.pi * ds
    m = (np.angle(np.exp(1j * ang).mean(0)) / (2 * np.pi)) % 1.0
    Rbar = np.abs(np.exp(1j * ang).mean(0))
    se = np.sqrt(np.maximum(-2 * np.log(np.maximum(Rbar, 1e-12)), 0)
                 / len(ds)) / (2 * np.pi)
    print("\n  circular mean velocity v:")
    for i in range(3):
        print(f"    v{i+1} = {m[i]:.4f} +- {se[i]:.4f}   "
              f"(concentration R = {Rbar[i]:.3f})")
    print("\n  distance to simple rationals, in sigma:")
    for i in range(3):
        row = []
        for q in (1, 2, 3, 4, 5, 6, 8, 10, 12):
            p = round(m[i] * q)
            cand = (p / q) % 1.0
            d = abs(((m[i] - cand + 0.5) % 1.0) - 0.5)
            row.append(f"{p}/{q}:{d/max(se[i],1e-9):.0f}s")
        print(f"    v{i+1}: " + " ".join(row))
    print("\n  A component many sigma from EVERY small rational is irrational")
    print("  to the resolution of this measurement: the orbit does not close.")


if __name__ == "__main__":
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 88
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else 103
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
    out = run(list(range(lo, hi + 1)), K=K)
    analyse(out)
