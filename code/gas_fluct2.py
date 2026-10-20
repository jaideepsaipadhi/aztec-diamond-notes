"""
The local gas fluctuation, with the height increment taken from the VALIDATED
step structure instead of a hand-rolled sign convention.

WHY THE FIRST ATTEMPT FAILED.  gas_fluct.py built its own dual path: a
horizontal ray, with both up-edges of each black vertex crossed and given the
SAME sign.  That is not a height increment, and the symptom was
Var[h(x) - h(y)] growing linearly in the separation.  I took the linear growth
for a bug in the gas inverse -- a gas phase has bounded height fluctuations --
but the EXACT finite-n inverse reproduces it (Var 1.92457 exact against 1.93450
gas at L = 8, n = 64), so the path was the problem, not the inverse.

ndep.faces_and_steps already carries the correct structure, and its signs
ALTERNATE with parity:

    right neighbour:  s = +1 if (x+1+y) even else -1
    up neighbour:     s = -1 if (x+y+1) even else +1

which is exactly what the hand-rolled version lacked.  That function underpins
the height machinery used throughout §3 and §5, so reusing it means the
observable here is the same one those sections measure, not a new guess.

WHAT THIS CAN AND CANNOT GIVE.  The discrete component of Theorem 1.2 is
intrinsically a FINITE-n object: it is the integer global height shift that the
diamond's frozen boundary forces on the facet.  The full-plane gas measure has
no boundary and therefore no discrete component, so no full-plane computation
can produce one.  What the gas inverse gives is the LOCAL fluctuation -- which
is precisely §3.3's contamination, the thing that produced the retracted factor
of 2, and whose size ("mass roughly 0.05 at a = 0.5 and 0.11 at a = 0.7 on each
side") currently rests on sampling alone.

THE TEST THAT DECIDES WHETHER THE OBSERVABLE IS RIGHT.  In a gas phase
Var[h(x) - h(y)] must SATURATE as |x - y| grows (bounded fluctuations is the
defining property of the smooth phase; a liquid grows like log, and linear
growth means no cancellation at all).  So the first thing reported below is the
variance against separation.  If it saturates, the observable is a height
increment and the per-side mass can be read off; if it grows, the construction
is still wrong and nothing further should be believed.

Run:  python3 gas_fluct2.py
"""
import sys
from collections import deque

import numpy as np
import scipy.sparse.linalg as spl

import gasinv2 as G2
from fastshuffle import bridge
from lowrank import sparse_K
from ndep import faces_and_steps
from vecshuffle import edge_index


def geometry(n):
    """Faces, the signed step graph, and the face -> twoperiodic edge map."""
    idx, E = edge_index(n)
    F, steps = faces_and_steps(n, idx)
    adj = {}
    for (f1, f2, ei, s) in steps:
        adj.setdefault(f1, []).append((f2, ei, s))
        adj.setdefault(f2, []).append((f1, ei, -s))
    p, q, src, Bset = bridge(n)
    inv = {((v[0] + v[1] - 1) // 2 + p, (v[1] - v[0] + 1) // 2 + q): v
           for v in src}
    Erev = {i: e for e, i in idx.items()}
    return F, adj, inv, Bset, Erev


def path_between(adj, f1, f2):
    """Shortest signed path f1 -> f2 as [(edge_index, sign)]."""
    prev = {f1: None}
    q = deque([f1])
    while q:
        v = q.popleft()
        if v == f2:
            break
        for (w, ei, s) in adj.get(v, ()):
            if w not in prev:
                prev[w] = (v, ei, s)
                q.append(w)
    if f2 not in prev:
        return None
    out, cur = [], f2
    while prev[cur] is not None:
        pv, ei, s = prev[cur]
        out.append((ei, s))
        cur = pv
    return out[::-1]


def crossed(path, inv, Bset, Erev):
    """[(black, white, sign)] in twoperiodic coordinates."""
    cr = []
    for (ei, s) in path:
        e = Erev[ei]
        v1, v2 = inv[e[0]], inv[e[1]]
        b, w = (v1, v2) if v1 in Bset else (v2, v1)
        cr.append((b, w, s))
    return cr


def pmf_from(cr, K, wi, bi, use_gas, a, Mpts=512):
    r = len(cr)
    if use_gas:
        M = np.array([[G2.gas_inverse(w, b2, a) for (b2, _, _) in cr]
                      for (_, w, _) in cr])
    else:
        lu = spl.splu(K.tocsc())
        cols = {}
        for (b, w, s) in cr:
            if bi[b] not in cols:
                e0 = np.zeros(K.shape[0], dtype=complex)
                e0[bi[b]] = 1.0
                cols[bi[b]] = lu.solve(e0)
        M = np.array([[cols[bi[b2]][wi[w]] for (b2, _, _) in cr]
                      for (_, w, _) in cr])
    d0 = np.array([K[bi[b], wi[w]] for (b, w, _) in cr])
    sg = np.array([s for (_, _, s) in cr])
    phi = np.empty(Mpts, dtype=complex)
    for k in range(Mpts):
        th = 2 * np.pi * k / Mpts
        D = d0 * (np.exp(-4j * th * sg) - 1.0)
        phi[k] = np.linalg.det(np.eye(r) + D[:, None] * M)
    return np.fft.ifft(phi).real


def atoms(p, step=4):
    """Atoms on the height lattice, wrapped into (-M/2, M/2] about the mode."""
    M = len(p)
    idx = [i for i, v in enumerate(p) if abs(v) > 1e-9]
    if not idx:
        return None
    c = max(idx, key=lambda i: p[i])
    out = {}
    for i in idx:
        off = ((i - c + M // 2) % M) - M // 2
        out[int(round(off / step))] = out.get(int(round(off / step)), 0.0) \
            + float(p[i])
    return out


def moments(d):
    t = sum(d.values())
    m = sum(k * v for k, v in d.items()) / t
    v = sum(k * k * val for k, val in d.items()) / t - m * m
    side = (d.get(1, 0.0) + d.get(-1, 0.0)) / 2 / t
    return t, m, v, d.get(0, 0.0) / t, side


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    a = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
    F, adj, inv, Bset, Erev = geometry(n)
    K, wi, bi = sparse_K(n, a)

    ctr = min(adj, key=lambda f: f[0] ** 2 + f[1] ** 2)
    print(f"n = {n}, a = {a}, centre face {ctr}, {len(F)} faces\n")
    print("THE DECIDING TEST: does Var[h(x)-h(y)] saturate with separation?")
    print("  (bounded variance is the defining property of the gas phase;")
    print("   the previous hand-rolled path gave Var growing like D/4)\n")
    print("{:>4} {:>5} {:>10} {:>10} {:>10}   {:>10} {:>10}".format(
        "D", "edges", "Var gas", "Var exact", "P(0) gas", "side gas", "side ex"))
    for D in (1, 2, 3, 4, 6, 8, 10, 12):
        f2 = (ctr[0] + D, ctr[1])
        pth = path_between(adj, ctr, f2)
        if pth is None:
            continue
        cr = crossed(pth, inv, Bset, Erev)
        try:
            dg = atoms(pmf_from(cr, K, wi, bi, True, a))
            de = atoms(pmf_from(cr, K, wi, bi, False, a))
        except Exception as exc:
            print(f"{D:>4}  failed: {exc}")
            continue
        if dg is None or de is None:
            continue
        _, _, vg, p0g, sg_ = moments(dg)
        _, _, ve, _, se_ = moments(de)
        print("{:>4} {:>5} {:10.5f} {:10.5f} {:10.6f}   {:10.6f} {:10.6f}"
              .format(D, len(cr), vg, ve, p0g, sg_, se_))
    print("\n  if Var saturates, the per-side mass at large D is the §3.3")
    print("  contamination, computed with no sampling and no facet observable.")
    print("  §3.3 claims ~0.05 at a = 0.5 and ~0.11 at a = 0.7, from sampling.")
