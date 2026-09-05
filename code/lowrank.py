"""
Exact height-difference distribution, stably and fast.

K(theta) differs from K(0) only on the crossed edges, where the entry is
multiplied by exp(-4 i theta s_e).  So

    K(theta) = K(0) + U D(theta) V^T,     D(theta)_kk = K0[b_k,w_k](e^{-4 i theta s_k} - 1)

with U, V the indicator columns of the crossed edges' black/white endpoints.
By the matrix determinant lemma,

    det K(theta) / det K(0) = det( I + D(theta) M ),   M_kl = (K0^{-1})[w_k, b_l]

which is an r x r determinant with r = number of crossed edges (~2n).  One sparse
LU of K(0) plus r sparse solves gives M; after that every theta costs one small
dense determinant.  No log-determinants, no branch tracking, no overflow.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from twoperiodic import vertices, kast_entry, E1, E2
from exactdist import dual_path_edges

DIRS = (E1, E2, (-E1[0], -E1[1]), (-E2[0], -E2[1]))


def sparse_K(n, a):
    W, B = vertices(n)
    wi = {v: k for k, v in enumerate(W)}
    bi = {v: k for k, v in enumerate(B)}
    Wset = set(W)
    rows, cols, vals = [], [], []
    for b in B:
        for d in DIRS:
            w = (b[0]+d[0], b[1]+d[1])
            if w in Wset:
                rows.append(bi[b]); cols.append(wi[w])
                vals.append(kast_entry(b, w, a))
    K = sp.csc_matrix((vals, (rows, cols)), shape=(len(B), len(W)))
    return K, wi, bi


def build_M(n, a):
    """Return crossed-edge data and the small matrix M."""
    K, wi, bi = sparse_K(n, a)
    _, crossed = dual_path_edges(n)
    r = len(crossed)
    lu = spl.splu(K.tocsc())
    # columns of K^{-1} for the black endpoints we need
    need_b = sorted({bi[b] for (b, w, s) in crossed})
    colmap = {}
    for jb in need_b:
        e = np.zeros(K.shape[0], dtype=complex)
        e[jb] = 1.0
        colmap[jb] = lu.solve(e)          # indexed by W
    M = np.empty((r, r), dtype=complex)
    d0 = np.empty(r, dtype=complex)
    for k, (b, w, s) in enumerate(crossed):
        d0[k] = K[bi[b], wi[w]]
        for l, (b2, w2, s2) in enumerate(crossed):
            M[k, l] = colmap[bi[b2]][wi[w]]
    signs = np.array([s for (_, _, s) in crossed])
    return M, d0, signs


def pmf(n, a, Mpts=64):
    M, d0, signs = build_M(n, a)
    r = len(signs)
    phi = np.empty(Mpts, dtype=complex)
    for k in range(Mpts):
        th = 2*np.pi*k/Mpts
        D = d0 * (np.exp(-4j*th*signs) - 1.0)
        phi[k] = np.linalg.det(np.eye(r) + D[:, None]*M)
    return np.fft.ifft(phi).real


if __name__ == "__main__":
    import time
    print("validation against the dense method / enumeration at n=4")
    p = pmf(4, 0.7, 64)
    nz = [(i, round(v, 6)) for i, v in enumerate(p) if abs(v) > 1e-8]
    print("  support:", nz, " sum =", round(p.sum(), 8))
    print("  expected: 0 -> 0.072772, 4 -> 0.436301, 56 -> 0.054626, "
          "60 -> 0.436301")

    print("\nlarger n (a=0.5): support width and timing")
    for n in (12, 24, 48, 96):
        t0 = time.time()
        p = pmf(n, 0.5, 64)
        dt = time.time() - t0
        nz = [(i, v) for i, v in enumerate(p) if abs(v) > 1e-6]
        print(f"  n={n:<4} {len(nz):3d} atoms, sum={p.sum():.6f}, "
              f"max atom={max(v for _, v in nz):.4f}  ({dt:.1f}s)")
