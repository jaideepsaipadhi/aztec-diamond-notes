"""
Fast exact sampler for the two-periodic Aztec diamond.

The slow version recomputed a growing determinant for every candidate edge,
costing about O(V^3.5) per sample.  Standard fact for determinantal dimer
measures: with A = K^{-1},

    P(edge bw present) = K(b,w) * A(w,b),

and these sum to 1 over the neighbours w of a fixed black vertex b.  After
committing to the edge bw, the conditional measure on the remaining graph is
obtained by a Schur complement (i.e. Gaussian elimination) update

    A(w',b') <- A(w',b') - A(w',b) A(w,b') / A(w,b).

So each step is one rank-one update of A: O(V^2) per step, O(V^3) per sample,
with no determinants at all.

Validated against the exact enumerated distribution at n=4.
"""
import numpy as np

from twoperiodic import E1, E2, vertices, kast_entry, edge_weight, kasteleyn

DIRS = (E1, E2, (-E1[0], -E1[1]), (-E2[0], -E2[1]))


def sample_fast(n, a, rng, Ainit=None):
    W, B = vertices(n)
    Wset = set(W)
    wi = {v: k for k, v in enumerate(W)}
    bi = {v: k for k, v in enumerate(B)}
    A = (np.linalg.inv(kasteleyn(n, a)) if Ainit is None else Ainit.copy())

    nbrs = {b: [(b[0] + d[0], b[1] + d[1]) for d in DIRS
                if (b[0] + d[0], b[1] + d[1]) in Wset] for b in B}

    match = []
    for b in sorted(B):
        jb = bi[b]
        cand = nbrs[b]
        p = np.array([abs(kast_entry(b, w, a) * A[wi[w], jb]) for w in cand])
        s = p.sum()
        if s <= 0:
            return None
        p = p / s
        w = cand[rng.choice(len(cand), p=p)]
        iw = wi[w]
        match.append((b, w))
        piv = A[iw, jb]
        if piv == 0:
            return None
        A -= np.outer(A[:, jb], A[iw, :]) / piv
    return match


if __name__ == "__main__":
    import collections
    import time

    from shuffle2 import in_ad

    rng = np.random.default_rng(5)
    n, a = 4, 0.7

    # exact distribution
    W, B = vertices(n)
    Wset = set(W)
    Bs = sorted(B)
    nb = {x: [(x[0]+d[0], x[1]+d[1]) for d in DIRS
              if (x[0]+d[0], x[1]+d[1]) in Wset] for x in Bs}
    exact = {}
    used = set()

    def rec(i, cur, wp):
        if i == len(Bs):
            exact[tuple(cur)] = wp
            return
        x = Bs[i]
        for y in nb[x]:
            if y not in used:
                used.add(y); cur.append((x, y))
                rec(i+1, cur, wp*edge_weight(x, y, a))
                cur.pop(); used.discard(y)
    rec(0, [], 1.0)
    Z = sum(exact.values())

    trials = 20000
    counts = collections.Counter()
    for _ in range(trials):
        m = sample_fast(n, a, rng)
        counts[tuple(sorted(m))] += 1
    keys = sorted(exact)
    obs = np.array([counts.get(k, 0) for k in keys], float)
    expc = np.array([exact[k]/Z for k in keys]) * trials
    chi2 = ((obs-expc)**2/np.maximum(expc, 1e-12)).sum()
    print(f"validation n=4 a=0.7: chi2 = {chi2:.1f} on {len(keys)-1} dof "
          f"(expect ~{len(keys)-1} +- {np.sqrt(2*(len(keys)-1)):.0f})")
    print(f"  hit {len(counts)}/{len(exact)}, outside enumeration "
          f"{len(set(counts)-set(exact))}")

    print("\ntiming (one sample, includes the K inverse):")
    for nn in (4, 8, 12, 16, 24, 32):
        W, B = vertices(nn)
        t = time.time(); sample_fast(nn, 0.5, rng); dt = time.time()-t
        print(f"  n={nn:<4} |B|={len(B):<6} {dt:7.2f}s   500 samples: "
              f"{dt*500/60:8.1f} min")
