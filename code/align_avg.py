"""
Remark 4.18: the four alignments of the two-periodic weights, measured by the
SPATIAL AVERAGE of the height over the gas facet.

Remark 4.18 says the four ways of aligning the periodic edge weights with the
Aztec diamond give the same characteristic polynomial and the same limit shape,
but four different shifts of the discrete component:

        e = 1/4,  3/4,  0,  2/4   for weights 1..4.

Shifts 0 and 1/2 are symmetric about their mode; 1/4 and 3/4 are strongly
asymmetric and are mirror images of one another.  The e = 1/2 case is the sharp
one: P(n) ~ exp(-pi (n - 1/2)^2 / b) puts EXACTLY EQUAL mass on two neighbouring
atoms, whatever b is.  Nothing else in this family does that, so it is a
signature that cannot be faked by a mis-set parameter.

Parameterisation.  Johansson (7.2) fixes one alignment through

    kast_entry(x, y, a):  j = bclass(x) = 0 if (x1+x2) mod 4 == 1 else 1
                          e1 -> a(1-j) + j          -e1 -> aj + (1-j)
                         -e2 -> (a(1-j) + j) i       e2 -> (aj + (1-j)) i

The two independent shifts inside the fundamental domain are
    swap_j : exchange the two classes of black vertex  (j -> 1-j)
    swap_e : exchange which diagonal pair carries `a`  (a <-> 1 on e1/-e1)

so (swap_j, swap_e) in {0,1}^2 enumerates the four alignments.  Which of the
four corresponds to which of the paper's "weights 1..4" is a labelling question;
what is testable is that the SET of measured shifts is {0, 1/4, 1/2, 3/4}.

Measurement is spatial_avg.py's: average h over a box of faces in the gas facet,
which removes the local gas fluctuation and leaves the rigid facet shift.
"""
import functools
import sys
import time

import numpy as np

from fastshuffle import bridge
from genshuffle import active_faces, descend, edges_of, face_edges
from spatial_avg import STEP, bfs_tree, box, coeffs_for, face_graph
from twoperiodic import E1, E2, bclass
from vecshuffle import edge_index

DIRS = (E1, E2, (-E1[0], -E1[1]), (-E2[0], -E2[1]))


def kast_aligned(x, y, a, swap_j, swap_e, swap_b=0):
    """swap_e : a <-> 1 everywhere          (weights 1 <-> 2 of Remark 4.18)
    swap_j : flip the black-vertex class on BOTH edge families (= swap_e here)
    swap_b : flip it on the beta family (+-E2) ONLY   (weights 1 <-> 3)"""
    j0 = bclass(x)
    if swap_j:
        j0 = 1 - j0
    hi, lo = (a, 1.0) if not swap_e else (1.0, a)
    d = (y[0] - x[0], y[1] - x[1])
    ja = j0
    jb = 1 - j0 if swap_b else j0
    if d == E1:
        return hi * (1 - ja) + lo * ja
    if d == (-E1[0], -E1[1]):
        return lo * (1 - ja) + hi * ja
    if d == (-E2[0], -E2[1]):
        return hi * (1 - jb) + lo * jb
    if d == E2:
        return lo * (1 - jb) + hi * jb
    return 0.0


def weight_fn_aligned(n, a, swap_j, swap_e, swap_b=0):
    p, q, src, Bset = bridge(n)
    inv = {((v[0] + v[1] - 1) // 2 + p, (v[1] - v[0] + 1) // 2 + q): v
           for v in src}

    def wfn(e):
        v1, v2 = inv[e[0]], inv[e[1]]
        b, w = (v1, v2) if v1 in Bset else (v2, v1)
        return abs(kast_aligned(b, w, a, swap_j, swap_e, swap_b))
    return wfn


def build_levels_aligned(n, a, swap_j, swap_e, swap_b=0):
    """vecshuffle.build_levels with the alignment-dependent weight function."""
    idx, E = edge_index(n)
    wfn = weight_fn_aligned(n, a, swap_j, swap_e, swap_b)

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
        ib, it, il, ir, pw = [], [], [], [], []
        w = ws[m]
        for f in active_faces(m):
            (b, t), (l, r) = face_edges(f)
            b, t, l, r = (tuple(sorted(x)) for x in (b, t, l, r))
            ib.append(idx[b]); it.append(idx[t])
            il.append(idx[l]); ir.append(idx[r])
            num = w[b] * w[t]
            pw.append(num / (num + w[l] * w[r]))
        levels[m] = (np.array(ib), np.array(it), np.array(il), np.array(ir),
                     np.array(pw))
    return levels, idx, E


def sample_levels(n, levels, nedges, rng):
    from vecshuffle import sample
    return sample(n, levels, nedges, rng)


# ------------------------------------------------------------------ analysis
def shift_from_ratios(rp, rm, b):
    """Solve for e from P(+1)/P(0) alone: log rp = -pi(1-2e)/b."""
    return 0.5 * (1 + b * np.log(rp) / np.pi)


def run(n=96, a=0.7, K=6000, R=4, seed=31337):
    from period_g1 import b_legendre
    b = float(b_legendre(a))
    idx, E, F, steps, adj = face_graph(n)
    root, parent, order = bfs_tree(adj)
    S = box(R, order)
    print(f"n={n} a={a} K={K} R={R} (|S|={len(S)})   b = {b:.6f}")
    print(f"predicted P(+1)/P(0) by shift:  "
          + "  ".join(f"e={e}: {np.exp(-np.pi*((1-e)**2-e**2)/b):.5f}"
                      for e in (0.0, 0.25, 0.5, 0.75)))
    print()

    rows = []
    for swap_e in (0, 1):
        for swap_b in (0, 1):
            swap_j = 0
            t0 = time.time()
            levels, _, EE = build_levels_aligned(n, a, swap_j, swap_e, swap_b)
            nE = len(EE)
            A, coef = coeffs_for(S, parent, order, nE)
            v = np.empty(K)
            for k in range(K):
                M = sample_levels(n, levels, nE, np.random.default_rng(seed + k))
                v[k] = (A - 4 * (coef @ M)) / len(S)
            z = np.round((v - np.median(v)) / STEP)
            z = z - np.round(np.median(z))
            u, c = np.unique(z, return_counts=True)
            p = dict(zip(u.astype(int), c / K))
            P0, Pp, Pm = p.get(0, 0.), p.get(1, 0.), p.get(-1, 0.)
            se = lambda q: np.sqrt(max(q, 1e-12) * (1 - q) / K) / P0
            rp = Pp / P0
            rm = Pm / P0
            e_hat = shift_from_ratios(rp, rm, b)
            print(f"  swap_e={swap_e} swap_b={swap_b}   "
                  f"P(-1)={Pm:.5f}  P(0)={P0:.5f}  P(+1)={Pp:.5f}    "
                  f"[{time.time()-t0:.0f}s]")
            print(f"      P(+1)/P(0) = {rp:.5f} +- {se(Pp):.5f}    "
                  f"P(-1)/P(0) = {rm:.5f} +- {se(Pm):.5f}")
            print(f"      implied shift e = {e_hat:.4f} "
                  f"+- {b*se(Pp)/rp/np.pi/2:.4f}\n")
            rows.append((swap_e, swap_b, rp, rm, e_hat))
    return rows, b


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 96
    a = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
    R = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    run(n=n, a=a, K=K, R=R)
