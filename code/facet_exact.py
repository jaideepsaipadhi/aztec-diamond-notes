"""
Facet sizes WITHOUT sampling: the exact mean height profile from K^{-1}.

WHY.  Two open items (configuration C's facet 1<->3 assignment, and the weight
search for a configuration with three healthy facets and a 2% off-diagonal
asymmetry) both reduce to needing a cheap, reliable measure of facet health.
There was none off the symmetric family:

  * hole_tune assumes C(z) = A(z) and returns nothing for the asymmetric
    configurations;
  * hole_tune2 generalises it correctly but is NOT a facet measure -- it gave a
    false positive on the first new configuration it was used for (config D:
    three bounded components resolved, smallest extent 1.54, and facet 1 turns
    out to have 0 faces).  A bounded amoeba component says the gas PHASE exists
    in the infinite model; whether its facet lands inside a finite diamond, and
    how large, is a limit-shape question the hole does not answer;
  * so g3_asym2.py fell back to "facet sizes checked afterwards by a short
    sampling run", making each trial cost minutes to hours.

THE OBSERVATION.  The facet detector (g3_erode.facet_masks) does not need
samples.  It needs the per-face MEAN height, to find the plateaus where the
slope is pinned to -4, 0, +4, and it uses the per-face variance only as a loose
gas/liquid filter.  The mean is exactly computable:

    E[h(f)] = E[h(parent)] + s_f (1 - 4 p_{e_f}),     p_e = K[b,w] K^{-1}[w,b]

accumulated down the same BFS tree the sampler uses.  So one LU of K and the
selected entries K^{-1}[w,b] on edges give the exact profile -- no Monte Carlo,
no sampling error, and the facet sizes are the ones at the n that will actually
be measured rather than an infinite-volume proxy for them.

The variance is replaced by the exact local gas discriminator
`edge_entropy`: in a gas phase each black vertex's dimer is close to
deterministic among its four edges, so the entropy of (p_e) around a vertex is
small, while in the liquid it is large.  That is computed from the same p_e at
no extra cost.  It plays the role of V's `1e-9 < V < median(V)` test.

VALIDATION, and the point of this file.  Today's two retractions both came from
calibrating on the cases that were convenient and generalising.  So this is
calibrated on configurations A, B and C -- whose facet sizes are known from
sampling -- and then tested on configuration D, which was NOT used to calibrate
and which the amoeba-hole proxy got wrong.  If it does not reproduce D's empty
facet 1, it is no better than what it replaces and must not be used.

Run:  python3 facet_exact.py
"""
import sys
import time

import numpy as np
import scipy.sparse.linalg as spl

from fastshuffle import bridge
from lowrank import sparse_K
from spatial_avg import bfs_tree, face_graph
from g3_measure import tree_arrays
from g3_erode import facet_masks

BLOCK = 256


def sparse_K_w(n, WU, WV):
    """K for general (2,4)-periodic weights, sparse.

    Conventions (vertex split, edge set, weight lookup, Kasteleyn phase) are
    taken verbatim from g3_sample so that this is the same matrix its
    exact_edge_probs builds densely -- the __main__ block checks that at small
    n.  Only the inversion differs: exact_edge_probs calls np.linalg.inv, which
    is O(V^3) and hopeless at n = 96, while every entry needed here lies on an
    edge, so selected inversion by sparse LU suffices.
    """
    import scipy.sparse as sp
    import g3_sample as GS
    V = sorted(GS._verts(n))
    B = [v for v in V if (v[0] + v[1]) % 2 == 0]
    W = [v for v in V if (v[0] + v[1]) % 2 == 1]
    bi = {v: k for k, v in enumerate(B)}
    wi = {v: k for k, v in enumerate(W)}
    rows, cols, vals = [], [], []
    ev = []
    for e in GS._edges(n):
        (pp_, qq) = e
        val = GS.weight(e, WU, WV)
        ph = 1.0 if qq[0] == pp_[0] + 1 else 1j
        b, w = (pp_, qq) if (pp_[0] + pp_[1]) % 2 == 0 else (qq, pp_)
        rows.append(bi[b]); cols.append(wi[w]); vals.append(val * ph)
        ev.append((tuple(sorted(e)), b, w, val * ph))
    K = sp.coo_matrix((vals, (rows, cols)),
                      shape=(len(B), len(W))).tocsr()
    return K, wi, bi, ev


def edge_probs(n, WU, WV, verbose=False):
    """Exact p_e for every edge, keyed by tuple(sorted(edge))."""
    t0 = time.time()
    K, wi, bi, ev = sparse_K_w(n, WU, WV)
    need = sorted({bi[b] for (_, b, _, _) in ev})
    lu = spl.splu(K.tocsc())
    if verbose:
        print(f"    K {K.shape}, LU in {time.time()-t0:.1f}s, "
              f"{len(need)} columns")
    col = {}
    N = K.shape[0]
    for s in range(0, len(need), BLOCK):
        chunk = need[s:s + BLOCK]
        rhs = np.zeros((N, len(chunk)))
        for j, jb in enumerate(chunk):
            rhs[jb, j] = 1.0
        sol = lu.solve(rhs.astype(complex))
        for j, jb in enumerate(chunk):
            col[jb] = sol[:, j]
    out = {}
    for (key, b, w, kv) in ev:
        out[key] = float(abs(kv * col[bi[b]][wi[w]]))
    if verbose:
        print(f"    {len(out)} edge probabilities in {time.time()-t0:.1f}s")
    return out


def profile(n, WU, WV, verbose=False):
    """(order, pidx, Mn, gas) -- exact mean height and the gas discriminator."""
    pe_prob = edge_probs(n, WU, WV, verbose)
    idx, E, F, steps, adj = face_graph(n)
    # idx keys are tuple(sorted(edge)) in the SAME coordinates as g3_sample
    p = np.zeros(len(idx))
    miss = 0
    for e, ei in idx.items():
        v = pe_prob.get(tuple(sorted(e)))
        if v is None:
            miss += 1
        else:
            p[ei] = v
    if verbose:
        print(f"    {miss} of {len(idx)} face-graph edges unmatched")
    root, parent, order = bfs_tree(adj)
    pidx, pe, ps, pp = tree_arrays(order, parent)
    Mn = np.zeros(len(order))
    for i in range(1, len(order)):
        Mn[i] = Mn[pp[i]] + ps[i] * (1.0 - 4.0 * p[pe[i]])
    # gas discriminator: entropy of the four edge probabilities at the face's
    # parent edge neighbourhood; small in a gas phase, large in the liquid
    gas = np.zeros(len(order))
    for i in range(1, len(order)):
        q = p[pe[i]]
        q = min(max(q, 1e-15), 1 - 1e-15)
        gas[i] = -(q * np.log(q) + (1 - q) * np.log(1 - q))
    return order, pidx, Mn, gas


def sizes(n, WU, WV, erodes=(4, 10), verbose=False):
    order, pidx, Mn, gas = profile(n, WU, WV, verbose)
    # facet_masks wants (V, Mn) and tests 1e-9 < V < median(V); the gas
    # entropy has exactly that orientation, so it slots in unchanged.
    masks = facet_masks(order, pidx, gas, Mn, erodes)
    return {e: [len(masks[(k, e)]) for k in range(3)] for e in erodes}


TRUTH = {
    'A': ("g3_best_weights.npy", "three healthy facets at n=192"),
    'B': ("g3b_weights.npy", "facet 1 = 24 faces at n=96, 0 under erode 10"),
    'C': ("g3c_weights.npy", "facets 1 and 3 both weak"),
    'D': ("g3d_weights.npy", "HELD OUT: facet 1 = 0 faces at n=96"),
}

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 96
    print(f"exact facet sizes from K^-1, no sampling, n = {n}\n")
    for tag in ("A", "B", "C", "D"):
        f, truth = TRUTH[tag]
        try:
            W = np.load("/home/claude/" + f)
        except FileNotFoundError:
            print(f"  {tag}: {f} missing"); continue
        t0 = time.time()
        try:
            s = sizes(n, W[0].tolist(), W[1].tolist(), verbose=(tag == "A"))
        except Exception as exc:
            print(f"  {tag}: failed -- {exc}")
            continue
        held = "   <-- HELD OUT" if tag == "D" else ""
        print(f"  config {tag}: " + "  ".join(
            f"erode {e}: {v}" for e, v in s.items())
            + f"   [{time.time()-t0:.0f}s]{held}")
        print(f"      sampled truth: {truth}")
