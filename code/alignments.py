"""
The four alignments of the two-periodic weights, and which shift each realizes.

Johansson (7.2) fixes one alignment.  Remark 4.18 says there are four, obtained
by shifting the weight pattern inside the fundamental domain.  In this
parameterization the two independent bits are:

    swap_j : which class of black vertex (x1+x2 mod 4) carries `a`
    swap_e : whether the pair {e1, -e2} or {-e1, e2} carries `a`

The shift e is diagnosed from the centre-height distribution: e=0 and e=1/2 are
symmetric about their mode, e=+-1/4 are strongly asymmetric.  At small n the
tails are fattened by finite-size effects, but the ASYMMETRY is a property of
the alignment and should survive.
"""
import collections

import numpy as np

E1 = (1, 1)
E2 = (-1, 1)
DIRS = (E1, E2, (-1, -1), (1, -1))


def make_kast(a, swap_j, swap_e):
    def kast(x, y):
        j = 0 if (x[0] + x[1]) % 4 == 1 else 1
        if swap_j:
            j = 1 - j
        d = (y[0] - x[0], y[1] - x[1])
        hi, lo = (a, 1.0) if not swap_e else (1.0, a)
        # pair {e1, -e2} carries `hi` for j=0, `lo` for j=1
        if d == E1:
            return hi*(1-j) + lo*j
        if d == (1, -1):                       # -e2
            return (hi*(1-j) + lo*j) * 1j
        if d == (-1, -1):                      # -e1
            return lo*(1-j) + hi*j
        if d == E2:
            return (lo*(1-j) + hi*j) * 1j
        return 0
    return kast


def vertices(n):
    W = [(x1, x2) for x1 in range(1, 2*n, 2) for x2 in range(0, 2*n+1, 2)]
    B = [(x1, x2) for x1 in range(0, 2*n+1, 2) for x2 in range(1, 2*n, 2)]
    return W, B


def build_K(n, kast):
    W, B = vertices(n)
    wi = {v: k for k, v in enumerate(W)}
    K = np.zeros((len(B), len(W)), dtype=complex)
    for r, x in enumerate(B):
        for d in DIRS:
            y = (x[0]+d[0], x[1]+d[1])
            if y in wi:
                K[r, wi[y]] = kast(x, y)
    return K, W, B


def sample(n, kast, rng, A0, W, B):
    wi = {v: k for k, v in enumerate(W)}
    bi = {v: k for k, v in enumerate(B)}
    Wset = set(W)
    A = A0.copy()
    match = []
    for b in sorted(B):
        jb = bi[b]
        cand = [(b[0]+d[0], b[1]+d[1]) for d in DIRS
                if (b[0]+d[0], b[1]+d[1]) in Wset]
        p = np.array([abs(kast(b, w) * A[wi[w], jb]) for w in cand])
        s = p.sum()
        if s <= 0:
            return None
        w = cand[rng.choice(len(cand), p=p/s)]
        iw = wi[w]
        match.append((b, w))
        piv = A[iw, jb]
        if piv == 0:
            return None
        A -= np.outer(A[:, jb], A[iw, :]) / piv
    return match


if __name__ == "__main__":
    from gasvar import matching_to_tiling
    from height import height

    a, n, ns = 0.5, 16, 300
    print(f"a={a}, n={n}, {ns} samples per alignment\n")
    for swap_j in (False, True):
        for swap_e in (False, True):
            kast = make_kast(a, swap_j, swap_e)
            K, W, B = build_K(n, kast)
            A0 = np.linalg.inv(K)
            rng = np.random.default_rng(31)
            hs = []
            for _ in range(ns):
                m = sample(n, kast, rng, A0, W, B)
                if m is None:
                    continue
                h = height(matching_to_tiling(m, n))
                if (0, 0) in h:
                    hs.append(h[(0, 0)])
            hs = np.array(hs)
            c = collections.Counter(hs)
            mode = c.most_common(1)[0][0]
            rel = collections.Counter(hs - mode)
            tot = len(hs)
            lo = 100*rel.get(-4, 0)/tot
            hi = 100*rel.get(4, 0)/tot
            mid = 100*rel.get(0, 0)/tot
            asym = abs(lo-hi)/max(lo+hi, 1e-9)
            print(f"  swap_j={swap_j!s:<5} swap_e={swap_e!s:<5}  "
                  f"-4:{lo:5.1f}%  0:{mid:5.1f}%  +4:{hi:5.1f}%   "
                  f"asymmetry={asym:.2f}")
