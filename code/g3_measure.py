"""
The genus-3 discrete component, measured: the joint distribution of the rigid
shifts of the THREE gas facets, against the 3x3 period matrix of the spectral
curve.

The facets are identified from the data, not assumed.  The mean height's local
slope dh/d(4y) takes exactly five plateau values, -8, -4, 0, +4, +8; dividing by
4 gives -2, -1, 0, +1, +2, which are the Newton polygon's two vertices and its
three interior points.  The +-8 plateaus have zero variance (frozen); the other
three are the gas facets, stacked along y.  Each is assigned as

        facet k  =  { faces : |dh/d(4y) - 4(k-2)| < 0.5,  |dh/d(2x)| < 0.4,
                      variance below the liquid level },

and then eroded to its interior so that no face near a facet boundary enters the
average.

The observable, as in genus 1, is the SPATIAL AVERAGE of h over each facet --
the local gas fluctuation is averaged away and the rigid shift survives -- and it
is again a single precomputed dot product per facet per sample, via a fixed BFS
tree of faces.

What is tested: not just the three marginals (each a 1-D discrete Gaussian of
scale B_kk) but the CORRELATIONS between facets, which come from the
off-diagonal entries of B and have no genus-1 analogue.  At e = (1/4,1/4,1/4)
the prediction is corr(Z1,Z2) = 0.324, corr(Z1,Z3) = 0.138, and a joint atom
P(1,1,0) fifteen times what independent components would give.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
import time

import numpy as np

from g3_sample import build_levels, sample
from spatial_avg import bfs_tree, coeffs_for, face_graph


def height_profile(n, levels, E, order, pidx, pe, ps, pp, K, seed):
    H = np.zeros((K, len(order)))
    rng = np.random.default_rng(seed)
    for k in range(K):
        M = sample(n, levels, len(E), rng)
        d = np.where(pe >= 0, ps * (1 - 4 * M[np.maximum(pe, 0)]), 0.0)
        h = np.zeros(len(order))
        for i in range(1, len(order)):
            h[i] = h[pp[i]] + d[i]
        H[k] = h
    return H


def tree_arrays(order, parent):
    pidx = {f: i for i, f in enumerate(order)}
    pe = np.full(len(order), -1)
    ps = np.zeros(len(order))
    pp = np.zeros(len(order), dtype=int)
    for f in order:
        pr = parent[f]
        if pr is None:
            continue
        p, ei, s = pr
        pe[pidx[f]] = ei
        ps[pidx[f]] = s
        pp[pidx[f]] = pidx[p]
    return pidx, pe, ps, pp


def find_facets(order, pidx, V, Mn, erode=3):
    """Assign faces to facets by the local slope of the mean height."""
    out = {}
    vliq = np.median(V)
    for k, target in ((0, -4.0), (1, 0.0), (2, +4.0)):
        sel = []
        for f in order:
            a = (f[0], f[1] + 4)
            b = (f[0] + 2, f[1])
            if a not in pidx or b not in pidx:
                continue
            sy = Mn[pidx[a]] - Mn[pidx[f]]
            sx = Mn[pidx[b]] - Mn[pidx[f]]
            if abs(sy - target) < 0.5 and abs(sx) < 0.4 and V[pidx[f]] > 1e-9 \
                    and V[pidx[f]] < vliq:
                sel.append(f)
        if not sel:
            out[k] = []
            continue
        xs = [f[0] for f in sel]
        ys = [f[1] for f in sel]
        # keep the dominant y-band, then erode in both directions.  The band
        # half-width must SCALE with the diamond: a fixed +-12 caps every facet
        # at 24 rows however large n is, so the facets stop growing and the
        # residual local noise stops shrinking.
        ymid = np.median(ys)
        half = max(12, int(0.11 * max(abs(f[1]) for f in order)))
        sel = [f for f in sel if abs(f[1] - ymid) <= half]
        ys = [f[1] for f in sel]
        xs = [f[0] for f in sel]
        x0, x1 = np.percentile(xs, 5), np.percentile(xs, 95)
        y0, y1 = min(ys), max(ys)
        out[k] = [f for f in sel
                  if x0 + erode <= f[0] <= x1 - erode
                  and y0 + erode <= f[1] <= y1 - erode]
    return out


def run(n=96, K=20000, seed=4242, Kprof=400):
    t0 = time.time()
    levels, idx, E = build_levels(n)
    idx2, E2, F, steps, adj = face_graph(n)
    root, parent, order = bfs_tree(adj)
    pidx, pe, ps, pp = tree_arrays(order, parent)
    print(f"n={n}  faces={len(order)}  edges={len(E)}  setup {time.time()-t0:.0f}s",
          flush=True)

    # --- locate the facets from a short profiling run
    Hp = height_profile(n, levels, E, order, pidx, pe, ps, pp, Kprof, seed - 1)
    V, Mn = Hp.var(0), Hp.mean(0)
    fac = find_facets(order, pidx, V, Mn)
    for k in range(3):
        if not fac[k]:
            print(f"  facet {k+1}: NOT FOUND")
            continue
        ys = [f[1] for f in fac[k]]
        xs = [f[0] for f in fac[k]]
        print(f"  facet {k+1}: {len(fac[k])} faces, "
              f"x in [{min(xs)},{max(xs)}], y in [{min(ys)},{max(ys)}], "
              f"mean var {np.mean([V[pidx[f]] for f in fac[k]]):.2f}")
    if any(not fac[k] for k in range(3)):
        return None

    # --- exact spatial-average coefficients
    co = {}
    for k in range(3):
        A, c = coeffs_for(fac[k], parent, order, len(E))
        co[k] = (len(fac[k]), A, c)
    # self-check against the profiling heights
    for k in range(3):
        nS, A, c = co[k]
        direct = np.mean([Hp[0][pidx[f]] for f in fac[k]])
        M0 = None
        print(f"  facet {k+1} coefficient vector: nnz {np.count_nonzero(c)}")

    t0 = time.time()
    Z = np.zeros((K, 3))
    rng = np.random.default_rng(seed)
    for i in range(K):
        M = sample(n, levels, len(E), rng)
        for k in range(3):
            nS, A, c = co[k]
            Z[i, k] = (A - 4 * (c @ M)) / nS
    print(f"  {K} samples in {time.time()-t0:.0f}s", flush=True)
    np.save(os.path.join(HERE, f"g3_Z_n{n}.npy"), Z)
    return Z


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 96
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    Z = run(n=n, K=K)
    if Z is not None:
        print("\n  raw spatial averages: mean "
              f"{np.round(Z.mean(0),3)}  sd {np.round(Z.std(0),3)}")
        print("  pairwise correlation of the RAW averages:")
        print(np.round(np.corrcoef(Z.T), 4))
