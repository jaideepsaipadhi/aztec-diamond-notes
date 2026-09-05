"""
The k x l doubly periodic Aztec diamond, built from Berggren-Nicoletti's own
Definition 2.1 (arXiv:2502.07241) rather than from Johansson's k=l=2 special case.

Vertices of the size M = k*l*N Aztec diamond:
    black  b_{l x + i, k y + j}   at  (2(lx+i),   2(ky+j)+1)
    white  w_{l x + i, k y + j}   at  (2(lx+i)+1, 2(ky+j)+2)

Kasteleyn matrix (their (9)), with row index w_{c1,r1} and column b_{c2,r2}:
    alpha_{r1+1, c1+1}   if (c2,r2) = (c1,   r1+1)
    gamma_{r1+1, c1+1}   if (c2,r2) = (c1,   r1)
    beta_{r1+1, c1+1}    if (c2,r2) = (c1+1, r1+1)
    -1                   if (c2,r2) = (c1+1, r1)
    0                    otherwise

so alpha/gamma/beta repeat with period l horizontally (in c) and k vertically
(in r).  The edges carrying -1 are the reference matching M_0 used for the
height function, and they are the ones with a negative Kasteleyn sign.

Genus of the spectral curve is g = (k-1)(l-1), i.e. that many gaseous facets.
For k = l = 2 this is 1 (what we have studied); k = 2, l = 3 gives g = 2, which
is the smallest genuinely multivariate case -- and since k = 2 makes P(z,w)
quadratic in w, that curve is still hyperelliptic and its periods are real
quadratures, exactly as in genus 1.
"""
import numpy as np


def index_sets(M):
    """Black and white index ranges (c, r) for a size M Aztec diamond.

    Determined by requiring |B| = |W| and det K to count matchings; verified
    below against 2^{M(M+1)/2} in the uniform case.
    """
    # matching the embedding: black at (2c, 2r+1), white at (2c+1, 2r+2).
    # In the (x1,x2) coordinates of the validated two-periodic code this is
    # black x1 even in [0,2M], x2 odd in [1,2M-1]; white x1 odd in [1,2M-1],
    # x2 even in [0,2M].  Hence the ranges differ between colours:
    B = [(c, r) for c in range(0, M + 1) for r in range(0, M)]
    W = [(c, r) for c in range(0, M) for r in range(-1, M)]
    return W, B


def kasteleyn(M, k, l, alpha, beta, gamma):
    W, B = index_sets(M)
    wi = {v: n for n, v in enumerate(W)}
    bi = {v: n for n, v in enumerate(B)}
    K = np.zeros((len(W), len(B)))
    for (c1, r1) in W:
        a = alpha[r1 % k][c1 % l]
        b = beta[r1 % k][c1 % l]
        g = gamma[r1 % k][c1 % l]
        for (c2, r2), val in (((c1, r1 + 1), a), ((c1, r1), g),
                              ((c1 + 1, r1 + 1), b), ((c1 + 1, r1), -1.0)):
            if (c2, r2) in bi:
                K[wi[(c1, r1)], bi[(c2, r2)]] = val
    return K, W, B


def uniform(k, l):
    o = [[1.0] * l for _ in range(k)]
    return o, [row[:] for row in o], [row[:] for row in o]


if __name__ == "__main__":
    print("uniform check: |det K| must equal 2^{M(M+1)/2}")
    for M in (1, 2, 3, 4):
        a, b, g = uniform(2, 2)
        K, W, B = kasteleyn(M, 2, 2, a, b, g)
        d = abs(np.linalg.det(K))
        print(f"  M={M}: |B|={len(B)} |W|={len(W)}  |det K|={d:12.4f}   "
              f"expected {2**(M*(M+1)//2):12d}")
