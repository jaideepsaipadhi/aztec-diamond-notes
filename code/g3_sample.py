"""
Sampler for the (2,4)-periodic Aztec diamond -- the genus-3 model whose period
matrix hyper_g3.py computes.

The two-periodic sampler (vecshuffle.build_levels) uses the torsion shortcut
w_m = c w_{m-4}, which holds only for the two-periodic weights.  Here the full
N-level descent is run instead; active_faces and edges_of are cached so the
descent stays the dominant-but-tolerable cost rather than the fatal one.

Weights, in the shuffler's own coordinates (which ARE the diagonal coordinates
of the Aztec diamond, up to a constant shift -- the shift is an alignment, so it
moves the shift vector e of the discrete component but not the period matrix):

    u-edge (x,y)-(x+1,y):  wu[x % 2][y % 4]
    v-edge (x,y)-(x,y+1):  wv[x % 2][y % 4]

VALIDATION (run as main): empirical edge frequencies from the shuffler are
compared against exact Kasteleyn edge probabilities P(e) = K_e (K^{-1})_{w,b} on
a small diamond.  A wrong weight-to-edge dictionary would otherwise give a
perfectly valid sampler for the WRONG model, silently.
"""
import functools
import time

import numpy as np

import genshuffle as G
from vecshuffle import edge_index

# the (2,4)-periodic weights used throughout
# tuned by g3_tune2.py: all three bounded amoeba holes 0.64 wide (so all three
# gas facets are visible) AND diag Im B = [0.634, 0.834, 0.640] (so the discrete
# component puts measurable mass on its neighbouring atoms).
WU = [[1.466377, 0.614494, 0.496125, 1.170496], [1.883652, 0.469123, 1.629409, 0.609127]]
WV = [[0.521337, 0.46243, 0.718033, 1.438181], [0.521337, 0.46243, 0.718033, 1.438181]]


@functools.lru_cache(maxsize=None)
def _verts(n):
    return G.verts(n)


@functools.lru_cache(maxsize=None)
def _edges(n):
    return tuple(G.edges_of(n))


@functools.lru_cache(maxsize=None)
def _faces(n):
    return tuple(G.active_faces(n))


def weight(e, wu=None, wv=None):
    wu = WU if wu is None else wu
    wv = WV if wv is None else wv
    (x1, y1), (x2, y2) = e
    if x2 == x1 + 1:                    # u-edge
        return wu[x1 % 2][y1 % 4]
    return wv[x1 % 2][y1 % 4]           # v-edge


def descend(n, w):
    out = {}
    for f in _faces(n):
        (b, t), (l, r) = G.face_edges(f)
        al, ga, be, de = w[b], w[t], w[l], w[r]
        DP = al * ga + be * de
        out[b], out[t] = ga / DP, al / DP
        out[l], out[r] = de / DP, be / DP
    keep = set(_edges(n - 1)) if n >= 2 else set()
    return {e: v for e, v in out.items() if e in keep}


def build_levels(n, wu=None, wv=None, verbose=False):
    """Full descent -- no torsion shortcut."""
    idx, E = edge_index(n)
    t0 = time.time()
    ws = {n: {e: float(weight(e, wu, wv)) for e in _edges(n)}}
    for m in range(n, 1, -1):
        ws[m - 1] = descend(m, ws[m])
    if verbose:
        print(f"  descent {time.time()-t0:.0f}s", flush=True)

    levels = {}
    for m in range(1, n + 1):
        ib, it, il, ir, pw = [], [], [], [], []
        w = ws[m]
        for f in _faces(m):
            (b, t), (l, r) = G.face_edges(f)
            bb, tt, ll, rr = (tuple(sorted(x)) for x in (b, t, l, r))
            ib.append(idx[bb]); it.append(idx[tt])
            il.append(idx[ll]); ir.append(idx[rr])
            num = w[b] * w[t]
            pw.append(num / (num + w[l] * w[r]))
        levels[m] = (np.array(ib), np.array(it), np.array(il), np.array(ir),
                     np.array(pw))
    return levels, idx, E


def sample(n, levels, nedges, rng):
    from vecshuffle import sample as vsample
    return vsample(n, levels, nedges, rng)


# ---------------------------------------------------------------- validation
def exact_edge_probs(n, wu=None, wv=None):
    """P(e) for every edge, from the Kasteleyn matrix of A_n directly."""
    V = sorted(_verts(n))
    B = [v for v in V if (v[0] + v[1]) % 2 == 0]
    W = [v for v in V if (v[0] + v[1]) % 2 == 1]
    assert len(B) == len(W)
    bi = {v: k for k, v in enumerate(B)}
    wi = {v: k for k, v in enumerate(W)}
    K = np.zeros((len(B), len(W)), dtype=complex)
    for e in _edges(n):
        (p, q) = e
        val = weight(e, wu, wv)
        # Kasteleyn phase: 1 on u-edges, i on v-edges
        ph = 1.0 if q[0] == p[0] + 1 else 1j
        b, w = (p, q) if (p[0] + p[1]) % 2 == 0 else (q, p)
        K[bi[b], wi[w]] += val * ph
    Ki = np.linalg.inv(K)
    out = {}
    for e in _edges(n):
        (p, q) = e
        val = weight(e, wu, wv)
        ph = 1.0 if q[0] == p[0] + 1 else 1j
        b, w = (p, q) if (p[0] + p[1]) % 2 == 0 else (q, p)
        out[tuple(sorted(e))] = abs(val * ph * Ki[wi[w], bi[b]])
    return out


if __name__ == "__main__":
    for n in (4, 6):
        print(f"=== n = {n}: shuffler vs exact Kasteleyn edge probabilities ===")
        levels, idx, E = build_levels(n)
        exact = exact_edge_probs(n)
        K = 40000
        cnt = np.zeros(len(E))
        rng = np.random.default_rng(3)
        for k in range(K):
            cnt += sample(n, levels, len(E), rng)
        emp = cnt / K
        ex = np.array([exact[e] for e in E])
        se = np.sqrt(np.maximum(ex * (1 - ex), 1e-12) / K)
        z = (emp - ex) / se
        print(f"  {len(E)} edges, {K} samples")
        print(f"  max |empirical - exact| = {np.abs(emp-ex).max():.5f}")
        print(f"  max |z| = {np.abs(z).max():.2f}   rms z = {np.sqrt((z**2).mean()):.2f}"
              f"   (expect rms ~ 1)")
        print(f"  sum of P(e) = {emp.sum():.4f} exact {ex.sum():.4f}  "
              f"(must equal #vertices/2 = {len(_verts(n))//2})\n")
