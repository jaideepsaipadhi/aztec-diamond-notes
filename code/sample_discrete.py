"""
The discrete component by SAMPLING, at n = 192 (NOTES.md section 3.4).

The determinant route (conditioned.py) gives the exact distribution but is capped
at n ~ 60: cond(K) ~ 10^(0.252 n), because the frozen corners' deterministic
matchings produce exponentially small singular values.

But the discrete component does not need a determinant.  It is a signed count of
matching edges crossing the dual path, so on a sampled configuration it is one
dot product.  Sampling involves no matrix at all, so there is no ceiling.

At a = 0.7 this measures the disputed parameter directly, rather than carrying it
from small a by the functional form.

Cost on one core: ~91 s to build the shuffling levels (one-time per size and
weight set), then ~32 ms per sample.
"""
import time

import numpy as np

from conditioned import path_edges
from vecshuffle import build_levels, edge_index, sample

PREDICTED_R = 0.130972          # exp(-pi/(2b)) with b = K'/K, k^2 = 1 - a^4
B_TRUE = 0.772737661704


def measure(n=192, a=0.7, K=6000, seed=10000, verbose=True):
    t0 = time.time()
    levels, _, EE = build_levels(n, a)
    if verbose:
        print(f"build_levels({n}): {time.time()-t0:.0f}s, {len(EE)} edges",
              flush=True)

    idx, _ = edge_index(n)
    rows, sgn = [], []
    for e, s in path_edges(n):
        k = idx.get(tuple(sorted(e)))
        if k is not None:
            rows.append(k)
            sgn.append(s)
    rows = np.array(rows)
    sgn = np.array(sgn, float)

    t1 = time.time()
    vals = np.empty(K)
    for k in range(K):
        m = sample(n, levels, len(EE), np.random.default_rng(seed + k))
        vals[k] = (m[rows] * sgn).sum()
    if verbose:
        print(f"{K} samples in {time.time()-t1:.0f}s "
              f"({(time.time()-t1)/K*1000:.0f} ms each)")

    u, c = np.unique(vals, return_counts=True)
    p = c / K
    i = int(np.argmax(p))
    p0 = p[i]
    lo = p[i-1] if i > 0 else 0.0
    hi = p[i+1] if i+1 < len(p) else 0.0
    r = (lo + hi) / 2 / p0

    se = lambda q: np.sqrt(q*(1-q)/K)
    sr = r*np.sqrt((se(lo)**2 + se(hi)**2)/max((lo+hi)**2, 1e-30)
                   + (se(p0)/p0)**2)
    b = -np.pi/(2*np.log(r))
    sb = sr*np.pi/(2*np.log(r)**2*r)
    return dict(atoms=(u[i-1], u[i], u[i+1]), probs=(lo, p0, hi),
                r=r, sr=sr, b=b, sb=sb)


if __name__ == "__main__":
    res = measure()
    lo, p0, hi = res["probs"]
    print(f"\n  atoms {res['atoms']}")
    print(f"  P(-1)={lo:.5f}  P(0)={p0:.5f}  P(+1)={hi:.5f}")
    print(f"  r = {res['r']:.5f} +- {res['sr']:.5f}   predicted {PREDICTED_R}"
          f"   ({abs(res['r']-PREDICTED_R)/res['sr']:.1f} sigma)")
    print(f"  b = {res['b']:.5f} +- {res['sb']:.5f}   K'/K = {B_TRUE:.6f}"
          f"   ({abs(res['b']-B_TRUE)/res['sb']:.1f} sigma)")
