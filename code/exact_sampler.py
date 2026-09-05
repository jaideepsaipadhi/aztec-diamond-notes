"""
Exact sampler for the two-periodic Aztec diamond via the inverse Kasteleyn matrix.

Kenyon's formula: for edges e_i = b_i w_i,
    P(e_1,...,e_r all present) = prod_i K(b_i,w_i) * det( K^{-1}(w_i, b_j) )_{i,j}

We sample sequentially: process black vertices in order; for the current black
vertex b, the conditional probability of matching it to each free neighbour w,
given the edges already fixed, is proportional to P(S union {bw}).  Normalising
over the candidates gives an exact draw.  After every black vertex is processed
we have a perfect matching distributed exactly according to the dimer measure.

No shuffling theory involved -- this is ground truth to validate the shuffling
sampler against, and it works right now.
"""
import numpy as np

from twoperiodic import (E1, E2, vertices, kast_entry, edge_weight, kasteleyn)

DIRS = (E1, E2, (-E1[0], -E1[1]), (-E2[0], -E2[1]))


def neighbours(x, Wset):
    return [(x[0] + d[0], x[1] + d[1]) for d in DIRS
            if (x[0] + d[0], x[1] + d[1]) in Wset]


def sample_exact(n, a, rng):
    """Return a perfect matching as a list of (black, white) pairs."""
    W, B = vertices(n)
    Wset = set(W)
    wi = {v: k for k, v in enumerate(W)}
    bi = {v: k for k, v in enumerate(B)}
    K = kasteleyn(n, a)
    Kinv = np.linalg.inv(K)          # Kinv[w_index, b_index]

    chosen = []                       # list of (b, w)
    used_w = set()

    for b in sorted(B):
        cands = [w for w in neighbours(b, Wset) if w not in used_w]
        if not cands:
            return None               # dead end (should not happen)
        weights = []
        for w in cands:
            S = chosen + [(b, w)]
            r = len(S)
            M = np.empty((r, r), dtype=complex)
            for i, (_, wi_v) in enumerate(S):
                for j, (bj_v, _) in enumerate(S):
                    M[i, j] = Kinv[wi[wi_v], bi[bj_v]]
            pref = 1.0 + 0j
            for (bb, ww) in S:
                pref *= kast_entry(bb, ww, a)
            weights.append(abs(pref * np.linalg.det(M)))
        tot = sum(weights)
        if tot <= 0:
            return None
        pick = rng.choice(len(cands), p=[x / tot for x in weights])
        w = cands[pick]
        chosen.append((b, w))
        used_w.add(w)
    return chosen


def matching_weight(match, a):
    p = 1.0
    for (b, w) in match:
        p *= edge_weight(b, w, a)
    return p


def canon(match):
    return tuple(sorted((b, w) for (b, w) in match))


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    n, a = 4, 0.7

    # exact distribution by enumeration
    W, B = vertices(n)
    Wset = set(W)
    Bs = sorted(B)
    nbrs = {x: neighbours(x, Wset) for x in Bs}
    exact = {}
    used = set()

    def rec(i, cur, wprod):
        if i == len(Bs):
            exact[tuple(cur)] = wprod
            return
        x = Bs[i]
        for y in nbrs[x]:
            if y not in used:
                used.add(y)
                cur.append((x, y))
                rec(i + 1, cur, wprod * edge_weight(x, y, a))
                cur.pop()
                used.discard(y)

    rec(0, [], 1.0)
    Z = sum(exact.values())
    print(f"n={n}, a={a}: {len(exact)} tilings, Z={Z:.6f}")

    trials = 20000
    counts = {}
    for _ in range(trials):
        m = sample_exact(n, a, rng)
        k = canon(m)
        counts[k] = counts.get(k, 0) + 1

    keys = sorted(exact)
    obs = np.array([counts.get(k, 0) for k in keys], float)
    expp = np.array([exact[k] / Z for k in keys])
    expc = expp * trials
    chi2 = ((obs - expc) ** 2 / np.maximum(expc, 1e-12)).sum()
    dof = len(keys) - 1
    print(f"sampler hit {len(counts)} of {len(exact)} tilings, "
          f"outside enumeration {len(set(counts) - set(exact))}")
    print(f"chi2 = {chi2:.1f} on {dof} dof   (expect ~{dof} +- {np.sqrt(2*dof):.0f})")

    # weight-class check: coarser and more powerful per bin
    import collections
    bins = collections.defaultdict(lambda: [0.0, 0])
    for k in keys:
        wv = round(exact[k], 9)
        bins[wv][0] += exact[k] / Z
        bins[wv][1] += counts.get(k, 0)
    print("\nby weight class:  weight   expected%   observed%")
    for wv in sorted(bins):
        e, o = bins[wv]
        print(f"   {wv:.6f}    {100*e:7.3f}    {100*o/trials:7.3f}")
