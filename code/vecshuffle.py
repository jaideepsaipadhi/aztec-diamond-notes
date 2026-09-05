"""
Vectorized generalized domino shuffling.

Same algorithm as genshuffle.py / fastshuffle.py, restructured for numpy.

The enabling fact: every edge of A_m lies on the boundary of EXACTLY ONE active
face of A_m.  So the per-face updates write to disjoint edge slots and the whole
level is one vectorized assignment -- no collisions, no atomics, no loop.

Representation:
  * every edge of A_N gets a global integer index once
  * a matching is a boolean array over those indices
  * each level m caches four index arrays (ib, it, il, ir) giving the bottom,
    top, left, right edges of its active faces, plus the creation probability
    p = w_b w_t / (w_b w_t + w_l w_r) per face

Per level the update is:
    c = Mb + Mt + Ml + Mr
    c == 2 -> face empties
    c == 1 -> the domino slides to the opposite edge
    c == 0 -> a new opposite pair, chosen with probability p

The torsion shortcut (w_m = c w_{m-4}, exact to 4e-16, c cancels in p) means only
four descents are needed regardless of N.
"""
import functools

import numpy as np

from genshuffle import active_faces, edges_of, face_edges, descend
from fastshuffle import weight_fn


@functools.lru_cache(maxsize=None)
def edge_index(n):
    """Global index for every edge of A_n (canonical orientation)."""
    E = sorted(tuple(sorted(e)) for e in edges_of(n))
    return {e: i for i, e in enumerate(E)}, E


def build_levels(n, a):
    """Per-level index arrays and creation probabilities."""
    idx, E = edge_index(n)
    wfn = weight_fn(n, a)

    ws = {n: {e: float(wfn(e)) for e in edges_of(n)}}
    top = max(2, n - 4)
    for m in range(n, top, -1):
        ws[m - 1] = descend(m, ws[m])
    for m in range(top, 1, -1):
        keep = set(tuple(sorted(e)) for e in edges_of(m - 1))
        ws[m - 1] = {e: v for e, v in ws[m + 3].items()
                     if tuple(sorted(e)) in keep}

    levels = {}
    for m in range(1, n + 1):
        fs = active_faces(m)
        ib, it, il, ir = [], [], [], []
        pw = []
        w = ws[m]
        for f in fs:
            (b, t), (l, r) = face_edges(f)
            b, t, l, r = (tuple(sorted(x)) for x in (b, t, l, r))
            ib.append(idx[b]); it.append(idx[t])
            il.append(idx[l]); ir.append(idx[r])
            num = w[b] * w[t]
            pw.append(num / (num + w[l] * w[r]))
        levels[m] = (np.array(ib), np.array(it), np.array(il), np.array(ir),
                     np.array(pw))
    return levels, idx, E


def sample(n, levels, nedges, rng):
    M = np.zeros(nedges, dtype=bool)
    for m in range(1, n + 1):
        ib, it, il, ir, p = levels[m]
        Mb, Mt, Ml, Mr = M[ib], M[it], M[il], M[ir]
        c = Mb.astype(np.int8) + Mt + Ml + Mr
        one = c == 1
        zero = c == 0
        coin = rng.random(len(p)) < p
        newb = (one & Mt) | (zero & coin)
        newt = (one & Mb) | (zero & coin)
        newl = (one & Mr) | (zero & ~coin)
        newr = (one & Ml) | (zero & ~coin)
        M = np.zeros(nedges, dtype=bool)
        M[ib] = newb; M[it] = newt; M[il] = newl; M[ir] = newr
    return M


if __name__ == "__main__":
    import time
    from twoperiodic import kasteleyn, kast_entry, vertices as tpverts, E1, E2
    from fastshuffle import bridge

    rng = np.random.default_rng(8)
    a, n, trials = 0.7, 12, 4000

    levels, idx, E = build_levels(n, a)
    W, B = tpverts(n)
    wi = {v: k for k, v in enumerate(W)}
    bi = {v: k for k, v in enumerate(B)}
    A = np.linalg.inv(kasteleyn(n, a))
    p, q, src, Bset = bridge(n)
    inv = {((v[0] + v[1] - 1) // 2 + p, (v[1] - v[0] + 1) // 2 + q): v
           for v in src}

    exact = np.zeros(len(E))
    for k, e in enumerate(E):
        v1, v2 = inv[e[0]], inv[e[1]]
        b, w = (v1, v2) if v1 in Bset else (v2, v1)
        exact[k] = abs(kast_entry(b, w, a) * A[wi[w], bi[b]])

    acc = np.zeros(len(E))
    t0 = time.time()
    for _ in range(trials):
        acc += sample(n, levels, len(E), rng)
    dt = time.time() - t0
    emp = acc / trials
    se = np.sqrt(np.maximum(exact * (1 - exact) / trials, 1e-12))
    z = np.abs(emp - exact) / se
    print(f"n={n} a={a} {trials} samples in {dt:.1f}s "
          f"({dt/trials*1000:.2f} ms/sample)")
    print(f"  max |empirical-exact| = {np.abs(emp-exact).max():.5f}")
    print(f"  max z = {z.max():.2f},  mean |z| = {z.mean():.2f}  "
          f"(expect ~3-4 and ~0.8)")

    print("\nscaling:")
    for nn in (24, 48, 96, 192):
        t0 = time.time(); lv, ix, EE = build_levels(nn, a); tb = time.time()-t0
        k = max(3, min(50, 4000 // nn))
        t0 = time.time()
        for _ in range(k):
            sample(nn, lv, len(EE), rng)
        ms = (time.time()-t0)/k*1000
        print(f"  n={nn:<5} build {tb:6.1f}s   {ms:9.2f} ms/sample   "
              f"2000 samples: {ms*2000/1000/3600:6.2f} h")
