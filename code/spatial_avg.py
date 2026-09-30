"""
The discrete component as a SPATIAL AVERAGE of the height function over the gas
facet (Nicoletti, 2026-09: "You need the spatial average to get the right
distribution").  This supersedes the single-face measurement of
sample_discrete.py, and it reverses that file's conclusion.

WHAT WAS WRONG BEFORE.  sample_discrete.py measured the height at ONE face of
the gas facet.  That is (discrete component) + (local gas fluctuation).  The
local fluctuation is a nearly symmetric +-1 displacement of one face, with mass
about 0.05 on each side at a = 0.5.  It therefore fills in the P(-1) atom, which
the true discrete component essentially does not have, and inflates P(+1).  The
resulting three-atom shape is symmetric and looks like a discrete Gaussian of
scale 2b -- which is the "factor of 2" reported in NOTES.md section 3.3.  It is
an artifact of the observable, not a property of the model.

THE FIX.  The discrete component is a RIGID shift of the entire facet, so it
survives spatial averaging while the local fluctuation is averaged away.  We
average h over a box of faces centred on the diamond:

        Zbar(R) = (1/|S_R|) sum_{f in S_R} h(f) - h(root),
        S_R = { faces (x,y) : |x+1/2| <= R, |y+1/2| <= R }.

Units: the face height moves in steps of 4 across the facet, so the discrete
component in the normalisation of the paper is Zbar/4.  The noise atoms sit at
+-1, +-2, +-3 in raw units and vanish with R; the signal sits at multiples of 4
and does not.  That separation is itself the check that the average is working.

COST.  A per-sample BFS over ~(2n)^2 faces would dominate.  It is not needed:
the face topology and step signs are fixed, only the matching M varies.  Fix a
BFS tree of faces rooted at the frozen corner; h(f) = sum over the tree path of
s_e (1 - 4 M_e).  Summing over f in S, each tree edge appears with multiplicity
cnt[f] = number of faces of S in the subtree below it, so

        sum_{f in S} h(f) = A - 4 * (coef . M),   coef[e_f] = cnt[f] s_f,
                                                  A = sum_f cnt[f] s_f.

One dot product per sample and per R -- the same cost as the single-face
observable.  Verified against a direct BFS height on every run.
"""
import time
from collections import deque

import numpy as np

from ndep import faces_and_steps, height_at
from vecshuffle import build_levels, edge_index, sample

STEP = 4            # the facet height moves in units of 4 raw Thurston steps


# ------------------------------------------------------------------ topology
def face_graph(n):
    idx, E = edge_index(n)
    F, steps = faces_and_steps(n, idx)
    adj = {}
    for (f1, f2, ei, s) in steps:
        adj.setdefault(f1, []).append((f2, ei, s))
        adj.setdefault(f2, []).append((f1, ei, -s))
    return idx, E, F, steps, adj


def bfs_tree(adj):
    """Root at the extreme corner face (frozen, so h there is deterministic)."""
    root = min(adj)
    parent, order = {root: None}, [root]
    q = deque([root])
    while q:
        v = q.popleft()
        for (w, ei, s) in adj[v]:
            if w not in parent:
                parent[w] = (v, ei, s)
                order.append(w)
                q.append(w)
    return root, parent, order


def coeffs_for(S, parent, order, nedges):
    """(A, coef) with sum_{f in S} h(f) = A - 4 * coef . M,  h(root) = 0."""
    cnt = {f: 0 for f in order}
    for f in S:
        if f in cnt:
            cnt[f] += 1
    coef = np.zeros(nedges)
    A = 0.0
    for f in reversed(order):                   # children before parents
        pr = parent[f]
        if pr is None:
            continue
        p, ei, s = pr
        c = cnt[f]
        if c:
            coef[ei] += c * s
            A += c * s
            cnt[p] += c
    return A, coef


def box(R, order):
    s = set(order)
    if R == 0:                                   # the old single-face observable
        return [f for f in [(0, 0)] if f in s]
    return [f for f in s if abs(f[0] + 0.5) <= R and abs(f[1] + 0.5) <= R]


# ------------------------------------------------------- alignment predictions
def alignment_ratios(b, e):
    """P(+1)/P(0) and P(-1)/P(0) for P(n) ~ exp(-pi (n-e)^2 / b)."""
    f = lambda n: np.exp(-np.pi * (n - e) ** 2 / b)
    return f(1) / f(0), f(-1) / f(0)


def analyse(vals, b, K, label, nS):
    """vals are raw face-height averages.  Report the discrete-component pmf."""
    v = (vals - np.median(vals)) / STEP     # signal lattice has spacing 4 -> now 1
    z = np.round(v)
    resid = np.abs(v - z)                   # residual local fluctuation
    z = z - np.round(np.median(z))
    u, c = np.unique(z, return_counts=True)
    sig = dict(zip(u.astype(int), c / len(v)))
    noise = float(np.mean(resid > 0.25))    # fraction not cleanly on the lattice

    P0 = sig.get(0, 0.0)
    Pp = sig.get(1, 0.0)
    Pm = sig.get(-1, 0.0)
    rp = Pp / P0 if P0 else np.nan
    rm = Pm / P0 if P0 else np.nan
    # binomial errors on the counts
    err = lambda q: np.sqrt(max(q, 1e-12) * (1 - q) / K) / P0

    print(f"  {label:<20} |S|={nS:<6} residual |Zbar/4 - nearest|: "
          f"mean {resid.mean():.4f}  frac>0.25 {noise:.4f}")
    print(f"  {'':<20} Z pmf:  P(-1)={Pm:.5f}  P(0)={P0:.5f}  P(+1)={Pp:.5f}")
    print(f"  {'':<20} P(+1)/P(0) = {rp:.5f} +- {err(Pp):.5f}"
          f"     P(-1)/P(0) = {rm:.5f} +- {err(Pm):.5f}")
    return rp, rm, err(Pp), err(Pm), noise


# ------------------------------------------------------------------ main run
def run(n=96, a=0.5, K=4000, seed=777, Rs=(0, 2, 4, 6, 8, 12, 16),
        b=None, check=True):
    from period_g1 import b_legendre
    if b is None:
        b = float(b_legendre(a))

    idx, E, F, steps, adj = face_graph(n)
    root, parent, order = bfs_tree(adj)
    print(f"n={n}  a={a}  K={K}   faces={len(order)}  edges={len(E)}   "
          f"b = K'/K = {b:.6f}")

    t0 = time.time()
    levels, _, EE = build_levels(n, a)
    nE = len(EE)
    print(f"  build_levels {time.time()-t0:.0f}s")

    prep = {}
    for R in Rs:
        S = box(R, order)
        A, coef = coeffs_for(S, parent, order, nE)
        prep[R] = (len(S), A, coef)

    if check:
        rng = np.random.default_rng(seed - 1)
        ok = True
        for _ in range(2):
            M = sample(n, levels, nE, rng)
            _, h, _ = height_at(steps, F, M, (0, 0))
            for R in Rs:
                S = box(R, order)
                nS, A, coef = prep[R]
                if abs(np.mean([h[f] for f in S]) - (A - 4 * (coef @ M)) / nS) > 1e-9:
                    ok = False
        print(f"  self-check (dot product == BFS height): {'ok' if ok else 'FAILED'}")

    t0 = time.time()
    vals = {R: np.empty(K) for R in Rs}
    for k in range(K):
        M = sample(n, levels, nE, np.random.default_rng(seed + k))
        for R in Rs:
            nS, A, coef = prep[R]
            vals[R][k] = (A - 4 * (coef @ M)) / nS
    print(f"  {K} samples in {time.time()-t0:.0f}s\n")

    print("=" * 74)
    print("  predicted P(+1)/P(0), P(-1)/P(0) for the four alignments of Rmk 4.18")
    print("=" * 74)
    for e in (0.0, 0.25, 0.5, 0.75):
        rp, rm = alignment_ratios(b, e)
        print(f"    e = {e:<5} scale b :  {rp:10.6f}   {rm:12.3e}")
    rp2, rm2 = alignment_ratios(2 * b, 0.0)
    print(f"    (the old single-face reading: symmetric, scale 2b : {rp2:.6f})")

    print("\n" + "=" * 74)
    print(f"  measured, box half-width R   (R=0 is the old single-face observable)")
    print("=" * 74)
    out = {}
    for R in Rs:
        out[R] = analyse(vals[R], b, K, f"R = {R}", prep[R][0])
        print()
    return vals, out, b


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 96
    a = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    Rs = tuple(int(x) for x in sys.argv[4].split(",")) if len(sys.argv) > 4 \
        else (0, 2, 4, 6, 8, 12, 16)
    run(n=n, a=a, K=K, Rs=Rs)
