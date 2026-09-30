"""
Falsification test: does the WRONG period matrix fit the data?

Configuration A's measured joint distribution of the three facet shifts matches
its own period matrix at chi2 = 0.4 on 3 dof for the correlations.  With three
fitted shifts that is suggestive but not conclusive on its own -- so the question
is whether a DIFFERENT genus-3 period matrix, from a different weight choice on
the same lattice, also fits.  It is given every advantage: all six facet
labellings and its own three free shifts.

Configuration C's period matrix differs from A's by up to 18% on the diagonal.
Both matrices have the same shape, similar magnitudes, and the same number of
free parameters, so this is a like-for-like comparison.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
import itertools
import sys

import numpy as np
from scipy.optimize import minimize

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


def fit(R, ImB, perm, starts):
    P = np.eye(3)[list(perm)]
    Q = np.linalg.inv(P @ ImB @ P.T)
    Rp = R[:, list(perm)]
    obs = {tuple(int(x) for x in a): int(b)
           for a, b in zip(*np.unique(Rp, axis=0, return_counts=True))}
    items = list(obs.items())

    def f(v):
        p = model(v, Q)
        return -sum(c * np.log(max(p.get(k, 1e-300), 1e-300)) for k, c in items)

    bb = None
    for st in starts:
        r = minimize(f, np.array(st), method='Nelder-Mead',
                     options=dict(xatol=2e-4, fatol=2e-3, maxiter=1500))
        if bb is None or r.fun < bb.fun:
            bb = r
    p = model(bb.x, Q)
    K = len(Rp)
    chi, dof = 0.0, 0
    for k, c in obs.items():
        ex = p.get(k, 0) * K
        if ex > 5:
            chi += (c - ex) ** 2 / ex
            dof += 1
    cm = corr_of(p)
    cc, parts = 0.0, []
    for (i, j) in ((0, 1), (0, 2), (1, 2)):
        mm = np.corrcoef(Rp[:, i], Rp[:, j])[0, 1]
        se = (1 - mm * mm) / np.sqrt(K)
        cc += ((mm - cm[i, j]) / se) ** 2
        parts.append((mm, cm[i, j], abs(mm - cm[i, j]) / se))
    return dict(e=bb.x, nll=bb.fun, chi=chi, dof=max(dof - 3, 1),
                corr=parts, corrchi=cc)


def seed_starts(R):
    """Coarse starts from the observed marginal asymmetries, plus a lattice."""
    st = [np.array(s) for s in itertools.product((0.0, -0.25, 0.25, 0.45), repeat=3)]
    return st[::3] + [np.zeros(3)]


if __name__ == "__main__":
    Z = np.load(os.path.join(HERE, 'g3_erode_n192.npy'))
    keys = [tuple(k) for k in np.load(os.path.join(HERE, 'g3_erode_keys_n192.npy'))]
    mats = {'A (its own)': np.load(os.path.join(HERE, 'g3_best_ImB.npy')),
            'C (wrong one)': np.load(os.path.join(HERE, 'g3c_ImB.npy'))}
    erode = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    cols = [keys.index((k, erode)) for k in range(3)]
    Zs = Z[:, cols]
    Zc = (Zs - np.median(Zs, 0)) / 4.0
    R = np.round(Zc)
    R = (R - np.round(np.median(R, 0))).astype(int)
    starts = seed_starts(R)
    print(f"=== configuration A data, n=192, K={len(R)}, erode {erode} ===")
    print(f"    ({len(starts)} optimiser starts per labelling, "
          f"all 6 labellings tried)\n", flush=True)
    for lab, M in mats.items():
        best = None
        for perm in itertools.permutations(range(3)):
            r = fit(R, M, perm, starts)
            if best is None or r['nll'] < best[1]['nll']:
                best = (perm, r)
        perm, r = best
        print(f"  period matrix {lab:<15} perm {perm}  e={np.round(r['e'],3)}")
        print(f"      -logL {r['nll']:.1f}   atoms chi2 {r['chi']:.1f}/{r['dof']}"
              f"   corr chi2 {r['corrchi']:.1f}/3")
        for (mm, pp, s), nm in zip(r['corr'], ('12', '13', '23')):
            print(f"      corr{nm}: measured {mm:+.4f}  predicted {pp:+.4f}"
                  f"   ({s:.1f} sigma)")
        print(flush=True)
