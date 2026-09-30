"""
Is the residual disagreement in corr(Z1,Z2) a facet-mask artifact?

At n = 192 the measured correlations are +0.344 / +0.114 / +0.215 against
+0.332 / +0.098 / +0.216 predicted: corr23 is exact, the two involving facet 1
sit ~2 sigma high.  A mask that leaks liquid into a facet average would do this,
because the liquid between two facets is shared and pulls their averages
together.

Conditioning on small rounding residual does NOT test this -- it selects
configurations, and makes the agreement worse (2.1 -> 8.0 sigma) purely through
that bias.  Eroding the mask does test it: erosion is a fixed linear functional
of the configuration, chosen before any sampling, so it introduces no selection.

All erosion levels are evaluated on the SAME sampled configurations, as a
separate precomputed dot product each, so the comparison carries no sampling
noise between levels and costs one pass.

If the discrepancy is mask leakage it shrinks monotonically with erosion and
plateaus; if it is a finite-n effect it does not move.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
import itertools
import sys
import time

import numpy as np
from scipy.optimize import minimize

from g3_sample import build_levels, sample
from g3_measure import height_profile, tree_arrays
from spatial_avg import bfs_tree, coeffs_for, face_graph


def facet_masks(order, pidx, V, Mn, erodes, slope_tol=0.5, band_frac=0.11):
    """Facet masks at several erosion levels, sharing one slope-based core."""
    vliq = np.median(V)
    ymax = max(abs(f[1]) for f in order)
    out = {}
    for k, target in ((0, -4.0), (1, 0.0), (2, +4.0)):
        sel = []
        for f in order:
            a, b = (f[0], f[1] + 4), (f[0] + 2, f[1])
            if a not in pidx or b not in pidx:
                continue
            sy = Mn[pidx[a]] - Mn[pidx[f]]
            sx = Mn[pidx[b]] - Mn[pidx[f]]
            if abs(sy - target) < slope_tol and abs(sx) < 0.4 \
                    and 1e-9 < V[pidx[f]] < vliq:
                sel.append(f)
        if not sel:
            for e in erodes:
                out[(k, e)] = []
            continue
        ymid = np.median([f[1] for f in sel])
        half = max(12, int(band_frac * ymax))
        sel = [f for f in sel if abs(f[1] - ymid) <= half]
        xs = [f[0] for f in sel]
        ys = [f[1] for f in sel]
        x0, x1 = np.percentile(xs, 5), np.percentile(xs, 95)
        y0, y1 = min(ys), max(ys)
        for e in erodes:
            out[(k, e)] = [f for f in sel
                           if x0 + e <= f[0] <= x1 - e
                           and y0 + e <= f[1] <= y1 - e]
    return out


def run(n=192, K=20000, seed=777, Kprof=400,
        erodes=(2, 6, 10, 14, 18)):
    t0 = time.time()
    levels, idx, E = build_levels(n)
    idx2, E2, F, steps, adj = face_graph(n)
    root, parent, order = bfs_tree(adj)
    pidx, pe, ps, pp = tree_arrays(order, parent)
    print(f"n={n} faces={len(order)} edges={len(E)}  setup {time.time()-t0:.0f}s",
          flush=True)

    Hp = height_profile(n, levels, E, order, pidx, pe, ps, pp, Kprof, seed - 1)
    V, Mn = Hp.var(0), Hp.mean(0)
    masks = facet_masks(order, pidx, V, Mn, erodes)
    co = {}
    for key, S in masks.items():
        if len(S) < 30:
            print(f"  facet {key[0]+1} erode {key[1]}: only {len(S)} faces -- skipped")
            continue
        A, c = coeffs_for(S, parent, order, len(E))
        co[key] = (len(S), A, c)
    for e in erodes:
        sizes = [len(masks[(k, e)]) for k in range(3)]
        print(f"  erode {e:<3} facet sizes {sizes}", flush=True)

    t0 = time.time()
    keys = sorted(co)
    Z = np.zeros((K, len(keys)))
    rng = np.random.default_rng(seed)
    for i in range(K):
        M = sample(n, levels, len(E), rng)
        for j, key in enumerate(keys):
            nS, A, c = co[key]
            Z[i, j] = (A - 4 * (c @ M)) / nS
    print(f"  {K} samples in {time.time()-t0:.0f}s", flush=True)
    np.save(os.path.join(HERE, f"g3_erode_n{n}.npy"), Z)
    np.save(os.path.join(HERE, f"g3_erode_keys_n{n}.npy"), np.array(keys))
    return Z, keys


# ------------------------------------------------------------------ analysis
def analyse(Z, keys, ImB):
    Q = np.linalg.inv(ImB)
    pts = list(itertools.product(range(-2, 3), repeat=3))

    def model(e):
        w = np.array([np.exp(-np.pi * ((np.array(v) - e) @ Q @ (np.array(v) - e)))
                      for v in pts])
        return dict(zip(pts, w / w.sum()))

    def corr_of(p):
        P = np.array(list(p.keys()))
        w = np.array(list(p.values()))
        m = (P * w[:, None]).sum(0)
        C = np.einsum('i,ij,ik->jk', w, P - m, P - m)
        d = np.sqrt(np.diag(C))
        return C / np.outer(d, d)

    erodes = sorted({k[1] for k in keys})
    print(f"\n{'erode':<7}{'e fitted':<26}"
          f"{'corr12 meas/pred':<26}{'corr13':<26}{'corr23':<26}chi2")
    for e in erodes:
        cols = [keys.index((k, e)) for k in range(3) if (k, e) in keys]
        if len(cols) != 3:
            continue
        Zs = Z[:, cols]
        K = len(Zs)
        Zc = (Zs - np.median(Zs, 0)) / 4.0
        R = np.round(Zc)
        R = (R - np.round(np.median(R, 0))).astype(int)
        obs = {tuple(int(x) for x in a): int(b)
               for a, b in zip(*np.unique(R, axis=0, return_counts=True))}
        f = lambda v: -sum(c * np.log(max(model(v).get(k, 1e-300), 1e-300))
                           for k, c in obs.items())
        bb = None
        for st in itertools.product((0.0, -0.25, 0.25, 0.5), repeat=3):
            r = minimize(f, np.array(st), method='Nelder-Mead',
                         options=dict(xatol=1e-4, fatol=1e-3, maxiter=3000))
            if bb is None or r.fun < bb.fun:
                bb = r
        cm = corr_of(model(bb.x))
        parts, chi = [], 0.0
        for (i, j) in ((0, 1), (0, 2), (1, 2)):
            mm = np.corrcoef(R[:, i], R[:, j])[0, 1]
            se = (1 - mm * mm) / np.sqrt(K)
            chi += ((mm - cm[i, j]) / se) ** 2
            parts.append(f"{mm:+.4f}/{cm[i,j]:+.4f}({abs(mm-cm[i,j])/se:.1f}s)")
        print(f"{e:<7}{str(np.round(bb.x,3)):<26}" + "".join(f"{p:<26}" for p in parts)
              + f"{chi:.1f}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 192
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    Z, keys = run(n=n, K=K)
    analyse(Z, keys, np.load(os.path.join(HERE, 'g3_best_ImB.npy')))
