"""
Exact facet-height distribution, conditioned.

The r x r determinant det(I + D M) lost accuracy above n ~ 60.  Two fixes:

1. EQUILIBRATION.  Scaling K -> R K C multiplies D_k M_kl by R_{b_k}/R_{b_l},
   i.e. conjugates D M by diag(R_b), leaving det(I + D M) unchanged.  So we may
   balance K freely before factorizing.  This is exact, not an approximation.

2. DIAGNOSTIC.  phi(theta) is a characteristic function, so |phi| <= 1 and
   phi(0) = 1 exactly.  Any theta with |phi| > 1 + tol signals loss of accuracy;
   we report it rather than silently returning a corrupted pmf.  The pmf must
   also be non-negative and sum to 1.
"""
from collections import deque

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from fastshuffle import bridge
from lowrank import sparse_K
from ndep import faces_and_steps
from vecshuffle import edge_index


def path_edges(n):
    idx, E = edge_index(n)
    F, steps = faces_and_steps(n, idx)
    adj = {}
    for (f1, f2, ei, s) in steps:
        adj.setdefault(f1, []).append((f2, ei, s))
        adj.setdefault(f2, []).append((f1, ei, -s))
    start, goal = (0, 0), min(adj)
    prev = {start: None}
    q = deque([start])
    while q:
        v = q.popleft()
        if v == goal:
            break
        for (w, ei, s) in adj[v]:
            if w not in prev:
                prev[w] = (v, ei, s)
                q.append(w)
    out = []
    cur = goal
    while prev[cur] is not None:
        p, ei, s = prev[cur]
        out.append((ei, -s))
        cur = p
    Erev = {i: e for e, i in idx.items()}
    return [(Erev[ei], s) for ei, s in out]


def equilibrate(K, iters=6):
    """Simple Sinkhorn-style balancing on |K|; returns R, C as 1-D arrays."""
    A = abs(K).tocsr()
    m, n = A.shape
    R = np.ones(m)
    C = np.ones(n)
    for _ in range(iters):
        rs = np.asarray(A.multiply(C[None, :]).max(axis=1).todense()).ravel()
        rs[rs == 0] = 1
        R = 1.0 / rs
        cs = np.asarray(A.multiply(R[:, None]).max(axis=0).todense()).ravel()
        cs[cs == 0] = 1
        C = 1.0 / cs
    return R, C


def pmf(n, a, Mpts=256, tol=1e-6):
    K, wi, bi = sparse_K(n, a)
    R, C = equilibrate(K)
    Ks = sp.diags(R) @ K @ sp.diags(C)

    p, q_, src, Bset = bridge(n)
    inv = {((v[0]+v[1]-1)//2+p, (v[1]-v[0]+1)//2+q_): v for v in src}
    cr = []
    for (e, s) in path_edges(n):
        v1, v2 = inv[e[0]], inv[e[1]]
        b, w = (v1, v2) if v1 in Bset else (v2, v1)
        cr.append((b, w, s))

    lu = spl.splu(Ks.tocsc())
    col = {}
    for jb in {bi[b] for b, _, _ in cr}:
        e0 = np.zeros(K.shape[0], dtype=complex)
        e0[jb] = 1.0
        col[jb] = lu.solve(e0)

    r = len(cr)
    M = np.array([[col[bi[b2]][wi[w]] for (b2, _, _) in cr] for (_, w, _) in cr])
    d0 = np.array([Ks[bi[b], wi[w]] for (b, w, _) in cr])
    sg = np.array([s for _, _, s in cr])

    phi = np.empty(Mpts, dtype=complex)
    for k in range(Mpts):
        th = 2*np.pi*k/Mpts
        D = d0 * (np.exp(-4j*th*sg) - 1.0)
        phi[k] = np.linalg.det(np.eye(r) + D[:, None]*M)

    diag = {
        "phi0_err": abs(phi[0] - 1.0),
        "max_abs_phi": float(np.abs(phi).max()),
    }
    pm = np.fft.ifft(phi).real
    diag["min_pmf"] = float(pm.min())
    diag["sum_pmf"] = float(pm.sum())
    diag["ok"] = (diag["phi0_err"] < tol and diag["max_abs_phi"] < 1 + 1e-6
                  and diag["min_pmf"] > -1e-7 and abs(diag["sum_pmf"] - 1) < tol)
    return pm, diag


def ratio(n, a, Mpts=256):
    pm, d = pmf(n, a, Mpts)
    M = len(pm)
    i = int(np.argmax(pm))
    side = (pm[(i-4) % M] + pm[(i+4) % M]) / 2
    return side / pm[i], pm[i], d


def ratio_checked(a, ns=(24, 36, 48), Mpts=256, tol=0.25):
    """Ratio with a CROSS-n consistency check.

    The per-run diagnostic (|phi| <= 1, non-negative pmf, sums to 1) is
    necessary but NOT sufficient: at a=0.3, n=60 it returned 0.052415 flagged
    ok=True while n=36 and n=48 both gave 0.022546, and at n=84 a 40-digit
    computation returned a perfectly valid distribution with a wrong answer.
    Agreement across n is the only check that has actually caught these.

    Returns (value, ok, table). ok is False if any run fails its own
    diagnostic, if the sequence is not monotone, or if the last step changes
    by more than `tol` of the preceding one.
    """
    table = []
    for n in ns:
        r, pmode, d = ratio(n, a, Mpts)
        table.append((n, r, pmode, bool(d["ok"])))
    rs = [t[1] for t in table]
    ok = all(t[3] for t in table)
    if ok and len(rs) >= 3:
        diffs = np.diff(rs)
        ok = bool(np.all(diffs > 0) or np.all(diffs < 0))          # monotone
        if ok and abs(diffs[-2]) > 0:
            ok = bool(abs(diffs[-1]) <= tol * abs(diffs[-2]))      # settling
    return rs[-1], ok, table


if __name__ == "__main__":
    import time
    print("a=0.7   ours predicts 0.130972,  paper-B predicts 0.049283\n")
    print("  n     ratio      P(mode)   |phi|max   min pmf     ok    time")
    for n in (24, 48, 60, 72, 96, 120):
        t0 = time.time()
        try:
            r, pmode, d = ratio(n, 0.7)
            dt = time.time() - t0
            print(f"  {n:<5} {r:.5f}   {pmode:.5f}  {d['max_abs_phi']:.6f}  "
                  f"{d['min_pmf']:+.2e}  {str(d['ok']):<5} {dt:6.1f}s")
        except Exception as ex:
            print(f"  {n:<5} failed: {ex}")
