"""
Optimized generalized shuffling for the two-periodic Aztec diamond.

Two optimizations over genshuffle.py, both validated:

1. TORSION SHORTCUT.  The weight arrays satisfy w_m = c * w_{m-4} exactly on
   shared edges (verified to 4e-16, with c = 1.132704 at a = 0.7).  Since the
   creation probability is the scale-invariant ratio alpha*gamma/(alpha*gamma +
   beta*delta), the constant cancels.  So instead of descending N times and
   storing N weight arrays (O(N^3) memory), we descend FOUR times and obtain
   every lower level by restriction.  Memory drops to O(N^2).

2. CACHED GEOMETRY.  active_faces() was recomputed on every call; it is a pure
   function of the level, so it is cached.

Validation: empirical edge frequencies against the exact edge probabilities
P(e) = K(b,w) K^{-1}(w,b) from the Kasteleyn matrix.
"""
import functools

import numpy as np

from genshuffle import active_faces, edges_of, face_edges, verts, descend
from twoperiodic import edge_weight, vertices as tpverts

active_faces = functools.lru_cache(maxsize=None)(active_faces)
edges_of = functools.lru_cache(maxsize=None)(edges_of)


@functools.lru_cache(maxsize=None)
def bridge(n):
    """Offsets (p,q) mapping diagonal coords to shuffling vertex coords."""
    W, B = tpverts(n)
    src = set(W) | set(B)
    tgt = verts(n)
    for p in range(-n - 2, n + 3):
        for q in range(-n - 2, n + 3):
            if {((x1 + x2 - 1) // 2 + p, (x2 - x1 + 1) // 2 + q)
                    for (x1, x2) in src} == tgt:
                return p, q, src, set(B)
    raise RuntimeError("no bridge found")


def weight_fn(n, a):
    p, q, src, Bset = bridge(n)
    inv = {((v[0] + v[1] - 1) // 2 + p, (v[1] - v[0] + 1) // 2 + q): v
           for v in src}

    def wfn(e):
        v1, v2 = inv[e[0]], inv[e[1]]
        b, w = (v1, v2) if v1 in Bset else (v2, v1)
        return edge_weight(b, w, a)
    return wfn


def build_weights(n, a):
    """Return dict level -> weight dict, using the period-4 shortcut."""
    wfn = weight_fn(n, a)
    ws = {n: {e: float(wfn(e)) for e in edges_of(n)}}
    top = max(2, n - 4)
    for m in range(n, top, -1):
        ws[m - 1] = descend(m, ws[m])
    for m in range(top, 1, -1):
        keep = set(edges_of(m - 1))
        ws[m - 1] = {e: v for e, v in ws[m + 3].items() if e in keep}
    return ws


def shuffle_up(m, match, w, rng):
    new = set()
    for f in active_faces(m):
        (b, t), (l, r) = face_edges(f)
        present = [e for e in (b, t, l, r) if e in match]
        if len(present) == 2:
            continue
        if len(present) == 1:
            e = present[0]
            new.add({b: t, t: b, l: r, r: l}[e])
            continue
        al, ga, be, de = w[b], w[t], w[l], w[r]
        if rng.random() < al * ga / (al * ga + be * de):
            new.add(b); new.add(t)
        else:
            new.add(l); new.add(r)
    return new


def sample(n, a, rng, ws=None):
    if ws is None:
        ws = build_weights(n, a)
    match = set()
    for m in range(1, n + 1):
        match = shuffle_up(m, match, ws[m], rng)
    return match


if __name__ == "__main__":
    import time
    import collections
    from twoperiodic import kasteleyn, kast_entry, E1, E2

    rng = np.random.default_rng(4)
    a, n, trials = 0.7, 12, 4000

    # exact edge probabilities from K^{-1}
    W, B = tpverts(n)
    wi = {v: k for k, v in enumerate(W)}
    bi = {v: k for k, v in enumerate(B)}
    A = np.linalg.inv(kasteleyn(n, a))
    exact = {}
    for b in B:
        for d in (E1, E2, (-1, -1), (1, -1)):
            w = (b[0] + d[0], b[1] + d[1])
            if w in wi:
                exact[(b, w)] = abs(kast_entry(b, w, a) * A[wi[w], bi[b]])

    p, q, src, Bset = bridge(n)
    inv = {((v[0] + v[1] - 1) // 2 + p, (v[1] - v[0] + 1) // 2 + q): v
           for v in src}

    ws = build_weights(n, a)
    cnt = collections.Counter()
    t0 = time.time()
    for _ in range(trials):
        for e in sample(n, a, rng, ws):
            v1, v2 = inv[e[0]], inv[e[1]]
            b, w = (v1, v2) if v1 in Bset else (v2, v1)
            cnt[(b, w)] += 1
    dt = time.time() - t0

    keys = sorted(exact, key=lambda k: -exact[k])
    err = [abs(cnt[k] / trials - exact[k]) for k in keys]
    se = [np.sqrt(max(exact[k] * (1 - exact[k]) / trials, 1e-12)) for k in keys]
    z = [e / s for e, s in zip(err, se)]
    print(f"n={n}, a={a}, {trials} samples in {dt:.0f}s "
          f"({dt/trials*1000:.1f} ms/sample)")
    print(f"edge-probability check against exact K^-1 over {len(keys)} edges:")
    print(f"   max |empirical - exact| = {max(err):.5f}")
    print(f"   max z-score             = {max(z):.2f}   (expect ~3-4 for "
          f"{len(keys)} edges)")
    print(f"   mean |z|                = {np.mean(z):.2f}   (expect ~0.8)")
