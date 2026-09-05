"""
END TO END: Schottky data -> Fock weights -> edge weights -> sampled dimers.

Pipeline:
  1. Schottky data (A, mu, train tracks) gives the period matrix, Abel map and
     theta functions, hence the face weights W_f (eq 67) as a function of the
     discrete Abel map eta(f).
  2. eta is linear on the faces: eta(x,y) = x*de + y*dn with
     de = A(alpha^-) - A(alpha^+), dn = A(beta^-) - A(beta^+).
     Generically these are not in Z + BZ, so the weights are QUASI-periodic --
     which is the whole point of the M-curve generalization.
  3. GAUGE RECONSTRUCTION.  W_f is gauge invariant, so it does not determine
     edge weights; we fix a gauge.  Setting every horizontal edge to 1, the face
     relation W_f = (top*bottom)/(left*right) = 1/(v_x v_{x+1}) recurses along
     each row: v_{x+1} = 1/(W_f v_x).
  4. Feed those edge weights to the validated generalized shuffler and compare
     the sampled edge frequencies with the exact Kasteleyn probabilities.

Step 4 is the check that the whole chain is consistent: the shuffler and the
Kasteleyn matrix are independent consumers of the same weights.
"""
import mpmath as mp
import numpy as np

from fockweights import abel, face_weight, schottky_genus1
from genshuffle import edges_of, sample, verts

mp.mp.dps = 25


def build_edge_weights(n, A, B, Bper, pts, D):
    """Face weights from Schottky data, then a gauge reconstruction."""
    bm, ap, bp, am = [abel(p, A, B) for p in pts]
    de, dn = ap - am, bp - bm

    # face (x,y) is the unit square with lower-left corner (x,y)
    def Wf(x, y):
        eta = x * de + y * dn
        w = face_weight(eta, pts, A, B, Bper, D)
        return abs(float(mp.re(w)))   # |W_f|: the Kasteleyn sign is supplied separately

    V = verts(n)
    weights = {}
    # horizontal edges: all set to 1 (gauge choice)
    for (x, y) in V:
        if (x + 1, y) in V:
            weights[tuple(sorted(((x, y), (x + 1, y))))] = 1.0
    # vertical edges: recurse along each row of faces
    ys = sorted({y for (_, y) in V})
    for y in ys:
        xs = sorted(x for (x, yy) in V if yy == y and (x, y + 1) in V)
        if not xs:
            continue
        v = 1.0
        e0 = tuple(sorted(((xs[0], y), (xs[0], y + 1))))
        weights[e0] = v
        for x in xs[:-1]:
            w = Wf(x, y)
            v = 1.0 / (w * v)
            e = tuple(sorted(((x + 1, y), (x + 1, y + 1))))
            weights[e] = v
    return weights, de, dn


def kasteleyn_from_weights(n, weights):
    """Bipartite Kasteleyn matrix with the standard square-lattice signs
    (horizontal 1, vertical i)."""
    V = sorted(verts(n))
    Bv = [v for v in V if (v[0] + v[1]) % 2 == 0]
    Wv = [v for v in V if (v[0] + v[1]) % 2 == 1]
    bi = {v: k for k, v in enumerate(Bv)}
    wi = {v: k for k, v in enumerate(Wv)}
    K = np.zeros((len(Bv), len(Wv)), dtype=complex)
    for e, val in weights.items():
        (u, v) = e
        horiz = (u[1] == v[1])
        b, w = (u, v) if u in bi else (v, u)
        if b in bi and w in wi:
            K[bi[b], wi[w]] += val * (1.0 if horiz else 1j)
    return K, Bv, Wv, bi, wi


if __name__ == "__main__":
    import collections

    n = 8
    pts = (mp.mpf('-2.4'), mp.mpf('-0.4'), mp.mpf('0.4'), mp.mpf('2.4'))
    A, B, Bper = schottky_genus1(mp.mpc('0.1', '1.0'), '0.02')
    D = mp.mpf('0.3')

    weights, de, dn = build_edge_weights(n, A, B, Bper, pts, D)
    print(f"Schottky data -> weights on AD({n})")
    print(f"  b = {mp.nstr(mp.im(Bper), 10)},  de = {mp.nstr(de, 8)}, "
          f"dn = {mp.nstr(dn, 8)}")
    vals = np.array(list(weights.values()))
    print(f"  {len(weights)} edges, weights in "
          f"[{vals.min():.6f}, {vals.max():.6f}], all positive: "
          f"{bool((vals > 0).all())}")

    K, Bv, Wv, bi, wi = kasteleyn_from_weights(n, weights)
    print(f"  |B|={len(Bv)}, |W|={len(Wv)},  |det K| = "
          f"{abs(np.linalg.det(K)):.6e}")

    Ainv = np.linalg.inv(K)
    exact = {}
    for e, val in weights.items():
        (u, v) = e
        horiz = (u[1] == v[1])
        b, w = (u, v) if u in bi else (v, u)
        if b in bi and w in wi:
            exact[e] = abs(val * (1.0 if horiz else 1j) * Ainv[wi[w], bi[b]])

    rng = np.random.default_rng(3)
    trials = 4000
    cnt = collections.Counter()
    wfn = lambda e: weights[tuple(sorted(e))]
    for _ in range(trials):
        for e in sample(n, wfn, rng):
            cnt[tuple(sorted(e))] += 1

    keys = sorted(exact)
    emp = np.array([cnt[k] / trials for k in keys])
    ex = np.array([exact[k] for k in keys])
    se = np.sqrt(np.maximum(ex * (1 - ex) / trials, 1e-12))
    z = np.abs(emp - ex) / se
    print(f"\nshuffler vs exact Kasteleyn over {len(keys)} edges, "
          f"{trials} samples:")
    print(f"  max |empirical - exact| = {np.abs(emp-ex).max():.5f}")
    print(f"  max z = {z.max():.2f},  mean |z| = {z.mean():.2f}  "
          f"(expect ~3-4 and ~0.8)")
