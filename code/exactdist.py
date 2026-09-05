"""
EXACT distribution of the facet height -- no Monte Carlo.

The height difference between two faces is a signed count of the dimers crossing
a dual path joining them:

    Delta h = sum over crossed edges e of  s_e * (1 - 4 * 1{e in M})

so, writing c(M) for the signed crossing count,

    E[ e^{i theta Delta h} ]  =  (1/Z) sum_M w(M) e^{i theta (...)}

and multiplying the Kasteleyn entry of each crossed edge e by e^{-4 i theta s_e}
produces exactly that sum as a determinant.  Since the Kasteleyn signs make every
matching contribute with a common phase, the ratio

    det K(theta) / det K(0)

is the characteristic function of Delta h (up to the deterministic prefactor
e^{i theta sum_e s_e}).  FFT over theta gives the exact atom probabilities.

Cost: one determinant per theta.  Dense is O(V^3), but K is sparse and planar,
so sparse LU is roughly O(V^1.5) and scales to large n.

Validated below against brute-force enumeration at n=4.
"""
import numpy as np

from twoperiodic import vertices, kasteleyn, kast_entry, edge_weight, E1, E2

DIRS = (E1, E2, (-E1[0], -E1[1]), (-E2[0], -E2[1]))


def dual_path_edges(n):
    """Edges crossed by a path from the centre out to the boundary, with signs.

    We use the straight ray in the +x1 direction from the central black vertex.
    Crossing the ray, the edges are those joining a vertex below to one above;
    the sign is +1/-1 by the colour of the lower endpoint.
    """
    W, B = vertices(n)
    Wset, Bset = set(W), set(B)
    c = min(B, key=lambda x: (x[0] - n) ** 2 + (x[1] - n) ** 2)
    crossed = []
    # walk outward in x1 from the centre; cross the edges of the vertical line
    x2 = c[1]
    for x1 in range(c[0], 2 * n + 1):
        for (u, v) in (((x1, x2), (x1 + 1, x2 + 1)),
                       ((x1, x2), (x1 - 1, x2 + 1))):
            if (u in Bset and v in Wset) or (u in Wset and v in Bset):
                b, w = (u, v) if u in Bset else (v, u)
                s = 1 if (b[0] + b[1]) % 4 == 1 else -1
                crossed.append((b, w, s))
    return c, crossed


def char_fn(n, a, thetas, crossed):
    W, B = vertices(n)
    wi = {v: k for k, v in enumerate(W)}
    bi = {v: k for k, v in enumerate(B)}
    K0 = kasteleyn(n, a)
    sign0, ld0 = np.linalg.slogdet(K0)
    out = []
    for th in thetas:
        K = K0.copy()
        for (b, w, s) in crossed:
            K[bi[b], wi[w]] *= np.exp(-4j * th * s)
        sg, ld = np.linalg.slogdet(K)
        out.append(sg / sign0 * np.exp(ld - ld0))
    return np.array(out)


def distribution(n, a, M=64):
    """Exact pmf of the (shifted) height difference, on the lattice 4Z."""
    c, crossed = dual_path_edges(n)
    # Delta h = S - 4 * (#crossings), S = sum of signs; so the random part is
    # a multiple of 4.  Characteristic function sampled at M points suffices.
    thetas = 2 * np.pi * np.arange(M) / M
    phi = char_fn(n, a, thetas, crossed)
    pmf = np.fft.ifft(phi).real
    return pmf, crossed


if __name__ == "__main__":
    import collections

    n, a = 4, 0.7
    print("validation at n=4 against brute-force enumeration\n")

    # brute force: exact distribution of the signed crossing count
    W, B = vertices(n)
    Wset = set(W)
    Bs = sorted(B)
    nbrs = {x: [(x[0]+d[0], x[1]+d[1]) for d in DIRS
                if (x[0]+d[0], x[1]+d[1]) in Wset] for x in Bs}
    c, crossed = dual_path_edges(n)
    cross_set = {(b, w): s for (b, w, s) in crossed}
    dist = collections.Counter()
    used = set()

    def rec(i, wp, cnt):
        if i == len(Bs):
            dist[cnt] += wp
            return
        x = Bs[i]
        for y in nbrs[x]:
            if y not in used:
                used.add(y)
                rec(i+1, wp*edge_weight(x, y, a), cnt + cross_set.get((x, y), 0))
                used.discard(y)
    rec(0, 1.0, 0)
    Z = sum(dist.values())
    exact = {k: v/Z for k, v in sorted(dist.items())}
    print("  crossings-count distribution, brute force:")
    for k, v in exact.items():
        print(f"    {k:+3d}: {v:.6f}")

    pmf, _ = distribution(n, a, M=64)
    top = sorted(((abs(p), i) for i, p in enumerate(pmf)), reverse=True)[:6]
    print("\n  from the determinant ratio (largest FFT coefficients):")
    for mag, i in sorted(top, key=lambda t: t[1]):
        print(f"    index {i:3d}: {pmf[i]:+.6f}")
