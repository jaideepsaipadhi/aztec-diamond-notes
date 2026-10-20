"""
The SHIFT as a function of N -- the half of Berggren-Nicoletti we skipped.

Theorem 1.2's scale matrix is B, and we measured that to death.  Its shift is
e^{(kN)}, and equation (19) says

    e^{(kN)} - e^{(0)}  =  N * k ( sum_i u(q_{0,i}) - l sum_i u(p_{0,i} ) )
                        =: N * v        in R^g / Z^g,

i.e. the shift advances LINEARLY on the Jacobian with a fixed velocity v.  The
paper's headline is that this does not converge: "In general, the shift
e^{(kN)}_{w_0,0} is not periodic in N", and it "evolves quasi-periodically in
R^g/Z^g".  Nobody has watched it move.

The important consequence of (19), which is easy to miss: quasi-periodicity does
NOT require quasi-periodic edge weights.  It requires v to be IRRATIONAL.  The
Section 4.6 example is periodic in N only because its v happens to be rational
(v = 1/4, hence Remark 4.18's four alignments and nothing more).  Ordinary
periodic weights with generic parameters should give an irrational v.

This file measures e(n) directly, with no reference to (19) and no need to
identify the angles q_{0,i}, p_{0,i} at all: for each n, sample, take the
spatial average over the gas facet (spatial_avg.py -- the single-face observable
is the one that produced the retracted factor of 2), and fit e from the pmf with
b held FIXED at the value the spectral curve gives.  One free parameter per n.

GENUS 1 IS THE FIXTURE.  There v = 1/4 is known independently, from Remark 4.18:
the four alignments are the four values e takes, so e(n) must advance by exactly
1/4 per step and cycle with period 4.  If this file does not reproduce that, it
is wrong and the genus-3 run means nothing.
"""
import sys
import time

import numpy as np
from scipy.optimize import minimize_scalar

from spatial_avg import STEP, bfs_tree, box, coeffs_for, face_graph
from vecshuffle import build_levels, edge_index, sample


def pmf_at(n, a, K, R, seed):
    """Spatial-average pmf of the discrete component at size n."""
    idx, E, F, steps, adj = face_graph(n)
    root, parent, order = bfs_tree(adj)
    S = box(R, order)
    levels, _, EE = build_levels(n, a)
    nE = len(EE)
    A, coef = coeffs_for(S, parent, order, nE)
    v = np.empty(K)
    for k in range(K):
        M = sample(n, levels, nE, np.random.default_rng(seed + k))
        v[k] = (A - 4 * (coef @ M)) / len(S)
    z = np.round((v - np.median(v)) / STEP)
    z = (z - np.round(np.median(z))).astype(int)
    u, c = np.unique(z, return_counts=True)
    return dict(zip(u.tolist(), (c / K).tolist())), len(S), K


def fit_shift(p, b, K):
    """ML fit of e in P(m) ~ exp(-pi (m-e)^2 / b), with b FIXED."""
    ms = np.arange(-4, 5)

    def nll(e):
        w = np.exp(-np.pi * (ms - e) ** 2 / b)
        w = w / w.sum()
        s = 0.0
        for m, q in p.items():
            if -4 <= m <= 4:
                s -= q * K * np.log(max(w[m + 4], 1e-300))
        return s

    best, bestv = None, np.inf
    for st in np.linspace(-0.5, 0.5, 21):
        r = minimize_scalar(nll, bracket=None, bounds=(st - 0.05, st + 0.05),
                            method='bounded')
        if r.fun < bestv:
            bestv, best = r.fun, r.x
    # curvature error
    h = 1e-3
    d2 = (nll(best + h) - 2 * nll(best) + nll(best - h)) / h ** 2
    se = 1.0 / np.sqrt(max(d2, 1e-12))
    return float(best), float(se)


def run(ns, a=0.6, K=6000, seed=555, Rfrac=8):
    from period_g1 import b_legendre
    b = float(b_legendre(a))
    print(f"genus 1, a = {a}, b = K'/K = {b:.6f}, K = {K} samples per n")
    print(f"Remark 4.18 => v = 1/4 exactly, so e(n) must advance by 1/4 per "
          f"step and cycle with period 4.\n")
    print(f"{'n':>5} {'|S|':>6} {'P(-1)':>8} {'P(0)':>8} {'P(+1)':>8}"
          f" {'e fitted':>12} {'4e mod 4':>10}")
    out = []
    for n in ns:
        t0 = time.time()
        R = max(2, n // Rfrac)
        p, nS, _ = pmf_at(n, a, K, R, seed + 97 * n)
        e, se = fit_shift(p, b, K)
        out.append((n, e, se))
        print(f"{n:>5} {nS:>6} {p.get(-1,0):8.4f} {p.get(0,0):8.4f} "
              f"{p.get(1,0):8.4f} {e:+8.4f}+-{se:.4f} {4*e % 4:10.3f}"
              f"   [{time.time()-t0:.0f}s]", flush=True)
    return out, b


def analyse(out):
    """Is the advance per step 1/4?  Differences are taken mod 1."""
    print("\n  step-to-step advance of e (mod 1):")
    ds = []
    for (n0, e0, _), (n1, e1, _) in zip(out, out[1:]):
        if n1 - n0 != 1:
            continue
        d = (e1 - e0) % 1.0
        ds.append(d)
        print(f"    n {n0} -> {n1}:  de = {d:.4f}")
    if ds:
        ds = np.array(ds)
        print(f"\n  mean advance {ds.mean():.4f} +- {ds.std(ddof=1)/np.sqrt(len(ds)):.4f}"
              f"   (1/4 = 0.2500)")
    print("\n  e by n mod 4:")
    for r in range(4):
        vals = [e % 1.0 for (n, e, _) in out if n % 4 == r]
        if vals:
            print(f"    n = {r} mod 4:  e mod 1 = "
                  + ", ".join(f"{v:.3f}" for v in vals))


if __name__ == "__main__":
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    a = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
    K = int(sys.argv[4]) if len(sys.argv) > 4 else 6000
    out, b = run(list(range(lo, hi + 1)), a=a, K=K)
    analyse(out)
