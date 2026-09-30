"""
Finite-n behaviour of the e = 1/2 alignment.

For e = 1/2 the discrete Gaussian puts EXACTLY equal mass on the atoms 0 and 1,
for every b, because (1-e)^2 = e^2.  At n = 96 we measure P(+1)/P(0) = 0.954,
which is 3.6 sigma below 1.  Either that is a finite-n correction that decays
with n, or it is a real deviation from Corollary 4.17.  The test is the
n-dependence: a 1/n correction halves from n to 2n.

The ratio is dimensionless and needs no parameter input, so this is the cleanest
possible finite-size probe in the whole family -- nothing has to be calibrated.
"""
import sys
import time

import numpy as np

from align_avg import build_levels_aligned, sample_levels
from spatial_avg import STEP, bfs_tree, box, coeffs_for, face_graph


def measure(n, a=0.7, K=12000, R=4, seed=90001, swap_e=0, swap_b=1):
    idx, E, F, steps, adj = face_graph(n)
    root, parent, order = bfs_tree(adj)
    S = box(R, order)
    t0 = time.time()
    levels, _, EE = build_levels_aligned(n, a, 0, swap_e, swap_b)
    nE = len(EE)
    A, coef = coeffs_for(S, parent, order, nE)
    tb = time.time() - t0

    v = np.empty(K)
    for k in range(K):
        M = sample_levels(n, levels, nE, np.random.default_rng(seed + k))
        v[k] = (A - 4 * (coef @ M)) / len(S)
    z = np.round((v - np.median(v)) / STEP).astype(int)
    u, c = np.unique(z, return_counts=True)
    p = dict(zip(u.tolist(), (c / K).tolist()))
    # anchor on the MODE, not the median: for e = 1/2 the two top atoms are
    # nearly tied and any centring rule picks between them arbitrarily.
    m = max(p, key=p.get)
    lo, hi = p.get(m - 1, 0.0), p.get(m + 1, 0.0)
    nb = m - 1 if lo >= hi else m + 1          # the dominant neighbour
    Pa, Pb = p[m], max(lo, hi)                 # Pb <= Pa by construction
    r = Pb / Pa
    sr = r * np.sqrt((1 - Pb) / (K * Pb) + (1 - Pa) / (K * Pa) + 2 / K)
    third = p.get(2 * m - nb, 0.0)             # atom on the far side of the mode
    print(f"  n={n:<5} |S|={len(S):<5} top pair {Pa:.5f} / {Pb:.5f}  "
          f"third atom {third:.5f}   ratio = {r:.5f} +- {sr:.5f}   "
          f"dev from 1 = {(r-1)*100:+.2f}%  ({abs(r-1)/sr:.1f} sigma)"
          f"   [build {tb:.0f}s]", flush=True)
    return r, sr


if __name__ == "__main__":
    ns = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 \
        else [32, 48, 64, 96, 144]
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 12000
    print(f"e = 1/2 alignment (swap_e=0, swap_b=1), a=0.7, R=4, K={K}")
    print("P(+1)/P(0) must equal 1 exactly in the limit, for any b.\n")
    out = []
    for n in ns:
        out.append((n,) + measure(n, K=K))
    ns_ = np.array([o[0] for o in out], float)
    rs = np.array([o[1] for o in out])
    ss = np.array([o[2] for o in out])
    d = 1 - rs
    print("\n  deviation 1 - r  vs n:")
    for n, di, si in zip(ns_, d, ss):
        print(f"    n={int(n):<5} 1-r = {di:.5f} +- {si:.5f}")
    ok = d > 3 * ss
    if ok.sum() >= 3:
        A = np.vstack([np.ones(ok.sum()), np.log(ns_[ok])]).T
        cp = np.linalg.lstsq(A, np.log(d[ok]), rcond=None)[0]
        rp = np.exp(A @ cp)
        A2 = np.vstack([np.ones(ok.sum()), ns_[ok]]).T
        ce = np.linalg.lstsq(A2, np.log(d[ok]), rcond=None)[0]
        re = np.exp(A2 @ ce)
        print(f"\n  power law  1-r ~ n^({cp[1]:.2f})   rms log resid "
              f"{np.std(np.log(d[ok])-np.log(rp)):.3f}")
        print(f"  exponential 1-r ~ exp({ce[1]:.4f} n)  (decay length "
              f"{-1/ce[1]:.1f})   rms log resid "
              f"{np.std(np.log(d[ok])-np.log(re)):.3f}")
        print(f"  extrapolated 1-r at n=384: power {np.exp(cp[0]+cp[1]*np.log(384)):.2e}"
              f"   exponential {np.exp(ce[0]+ce[1]*384):.2e}")
