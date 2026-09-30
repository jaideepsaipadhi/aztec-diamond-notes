"""
Second genus-3 configuration measured, and the SWAP TEST.

Configuration A (symmetric weights) gave diag Im B = (0.634, 0.834, 0.640) with
B12 = B23; its measured joint distribution of the three facet shifts matched to
chi2 = 0.4 on 3 dof for the correlations and 6.9 on ~7 for the atoms.

A good fit to one dataset with three fitted shifts is suggestive, not conclusive.
Configuration B has diag Im B = (0.517, 0.688, 0.858) -- monotone rather than
symmetric, B11 and B33 differing by 66%.  Two questions:

  1. does B's own period matrix fit B's data?
  2. does A's period matrix FAIL on B's data?

(2) is the one that matters.  Both matrices have the same shape and similar
magnitudes, and both get the same three free shifts, so if A also fits B's data
then the measurement is not really testing the period matrix at all.

The facets are found exactly as before, from the five slope plateaus of the mean
height, and the observable is the same eroded spatial average.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
import itertools
import sys
import time

import numpy as np
from scipy.optimize import minimize

import g3_sample as GS
from g3_erode import facet_masks
from g3_measure import height_profile, tree_arrays
from spatial_avg import bfs_tree, coeffs_for, face_graph


def run(tag, WU, WV, n=192, K=20000, seed=2024, Kprof=400, erodes=(4, 10)):
    t0 = time.time()
    levels, idx, E = GS.build_levels(n, WU, WV)
    idx2, E2, F, steps, adj = face_graph(n)
    root, parent, order = bfs_tree(adj)
    pidx, pe, ps, pp = tree_arrays(order, parent)
    print(f"[{tag}] n={n} faces={len(order)} setup {time.time()-t0:.0f}s", flush=True)

    Hp = height_profile(n, levels, E, order, pidx, pe, ps, pp, Kprof, seed - 1)
    V, Mn = Hp.var(0), Hp.mean(0)
    masks = facet_masks(order, pidx, V, Mn, erodes)
    co, keys = {}, []
    for e in erodes:
        sizes = [len(masks[(k, e)]) for k in range(3)]
        print(f"  erode {e}: facet sizes {sizes}", flush=True)
        if min(sizes) < 40:
            continue
        for k in range(3):
            A, c = coeffs_for(masks[(k, e)], parent, order, len(E))
            co[(k, e)] = (len(masks[(k, e)]), A, c)
            keys.append((k, e))
    if not keys:
        print("  no usable facets"); return None, None
    keys = sorted(set(keys))

    t0 = time.time()
    Z = np.zeros((K, len(keys)))
    rng = np.random.default_rng(seed)
    for i in range(K):
        M = GS.sample(n, levels, len(E), rng)
        for j, key in enumerate(keys):
            nS, A, c = co[key]
            Z[i, j] = (A - 4 * (c @ M)) / nS
    print(f"  {K} samples in {time.time()-t0:.0f}s", flush=True)
    np.save(os.path.join(HERE, f"g3_Z_{tag}.npy"), Z)
    np.save(os.path.join(HERE, f"g3_keys_{tag}.npy"), np.array(keys))
    return Z, keys


# ------------------------------------------------------------------ analysis
PTS = list(itertools.product(range(-2, 3), repeat=3))


def model(e, Q):
    w = np.array([np.exp(-np.pi * ((np.array(v) - e) @ Q @ (np.array(v) - e)))
                  for v in PTS])
    return dict(zip(PTS, w / w.sum()))


def corr_of(p):
    P = np.array(list(p.keys()))
    w = np.array(list(p.values()))
    m = (P * w[:, None]).sum(0)
    C = np.einsum('i,ij,ik->jk', w, P - m, P - m)
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def fit(R, ImB, perm=(0, 1, 2)):
    P = np.eye(3)[list(perm)]
    Q = np.linalg.inv(P @ ImB @ P.T)
    Rp = R[:, list(perm)]
    obs = {tuple(int(x) for x in a): int(b)
           for a, b in zip(*np.unique(Rp, axis=0, return_counts=True))}
    f = lambda v: -sum(c * np.log(max(model(v, Q).get(k, 1e-300), 1e-300))
                       for k, c in obs.items())
    bb = None
    for st in itertools.product((0.0, -0.25, 0.25, 0.5), repeat=3):
        r = minimize(f, np.array(st), method='Nelder-Mead',
                     options=dict(xatol=1e-4, fatol=1e-3, maxiter=3000))
        if bb is None or r.fun < bb.fun:
            bb = r
    p = model(bb.x, Q)
    K = len(Rp)
    chi = 0.0
    dof = 0
    for k, c in obs.items():
        ex = p.get(k, 0) * K
        if ex > 5:
            chi += (c - ex) ** 2 / ex
            dof += 1
    cm = corr_of(p)
    cc = 0.0
    parts = []
    for (i, j) in ((0, 1), (0, 2), (1, 2)):
        mm = np.corrcoef(Rp[:, i], Rp[:, j])[0, 1]
        se = (1 - mm * mm) / np.sqrt(K)
        cc += ((mm - cm[i, j]) / se) ** 2
        parts.append((mm, cm[i, j], abs(mm - cm[i, j]) / se))
    return dict(e=bb.x, nll=bb.fun, chi=chi, dof=max(dof - 3, 1),
                corr=parts, corrchi=cc)


def rounded(Z, keys, erode):
    cols = [keys.index((k, erode)) for k in range(3)]
    Zs = Z[:, cols]
    Zc = (Zs - np.median(Zs, 0)) / 4.0
    R = np.round(Zc)
    return (R - np.round(np.median(R, 0))).astype(int)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 192
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    wfile = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, 'g3c_weights.npy')
    mfile = sys.argv[4] if len(sys.argv) > 4 else os.path.join(HERE, 'g3c_ImB.npy')
    WB = np.load(wfile)
    Z, keys = run("C", WB[0].tolist(), WB[1].tolist(), n=n, K=K,
                  erodes=(4, 10))
    if Z is None:
        sys.exit(1)
    keys = [tuple(k) for k in keys]
    ImB_B = np.load(mfile)
    ImB_A = np.load(os.path.join(HERE, 'g3_best_ImB.npy'))
    for erode in sorted({k[1] for k in keys}):
        R = rounded(Z, keys, erode)
        print(f"\n=== configuration C data, erode {erode} ===")
        for lab, M in (("C (its own)", ImB_B), ("A (wrong one)", ImB_A)):
            best = None
            for perm in itertools.permutations(range(3)):
                r = fit(R, M, perm)
                if best is None or r["nll"] < best[1]["nll"]:
                    best = (perm, r)
            perm, r = best
            print(f"  period matrix {lab:<14} perm {perm}  e={np.round(r['e'],3)}"
                  f"  -logL {r['nll']:.1f}  atoms chi2 {r['chi']:.1f}/{r['dof']}"
                  f"  corr chi2 {r['corrchi']:.1f}/3")
            for (mm, pp, s), nm in zip(r["corr"], ("12", "13", "23")):
                print(f"      corr{nm}: measured {mm:+.4f}  predicted {pp:+.4f}"
                      f"   ({s:.1f} sigma)")
