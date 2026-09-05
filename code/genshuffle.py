"""
Generalized domino shuffling for weighted Aztec diamonds.

Following Janvresse-de la Rue-Velenik (Electron. J. Combin. 13 (2006) #R00),
which is Propp's generalized shuffling (math/0111034) with the zero-weight cases
made explicit.  We only need the nonzero-weight case.

Aztec diamond of order n (their convention):
    A_n = { (x,y) in Z^2 : |x - 1/2| + |y - 1/2| <= n }
with edges between vertices at Euclidean distance 1.  NOTE these are VERTICES of
the square lattice; dominoes are edges.  This is a different (rotated) picture
from the (W,B) diagonal coordinates used for the Kasteleyn work.

Faces are unit squares, labelled by their lower-left corner, chessboard-coloured
by (x+y) mod 2.  For each m, the ACTIVE faces are those coloured like the inner
boundary of A_m; the defining property we use to pin the colour down is that
every edge of A_m lies on the boundary of exactly one active face.

Weight recursion (Step 1), for an active face with boundary weights
alpha, gamma opposite and beta, delta opposite, DP = alpha*gamma + beta*delta:
    alpha -> gamma/DP,  gamma -> alpha/DP,  beta -> delta/DP,  delta -> beta/DP

Shuffling up: for each active face of A_{m+1},
    one domino on e      -> move it to the opposite edge e'
    two dominoes         -> remove both
    no domino            -> add two dominoes on an opposite pair, choosing the
                            alpha/gamma pair with probability
                            alpha*gamma / (alpha*gamma + beta*delta)
"""
import numpy as np


def verts(n):
    return {(x, y) for x in range(-n, n + 1) for y in range(-n, n + 1)
            if abs(x - 0.5) + abs(y - 0.5) <= n}


def edges_of(n):
    V = verts(n)
    E = []
    for (x, y) in V:
        for d in ((1, 0), (0, 1)):
            w = (x + d[0], y + d[1])
            if w in V:
                E.append(((x, y), w))
    return E


def face_edges(f):
    """The four boundary edges of the unit face with lower-left corner f,
    returned as two opposite PAIRS: ((bottom, top), (left, right))."""
    x, y = f
    bottom = ((x, y), (x + 1, y))
    top = ((x, y + 1), (x + 1, y + 1))
    left = ((x, y), (x, y + 1))
    right = ((x + 1, y), (x + 1, y + 1))
    return (bottom, top), (left, right)


def active_faces(n):
    """Faces of A_n whose four corners are all in A_n, of the colour for which
    every edge of A_n lies on exactly one active face."""
    V = verts(n)
    E = set(edges_of(n))
    cands = [f for f in ((x, y) for x in range(-n - 1, n + 1)
                         for y in range(-n - 1, n + 1))
             if all(c in V for c in ((f[0], f[1]), (f[0]+1, f[1]),
                                     (f[0], f[1]+1), (f[0]+1, f[1]+1)))]
    for colour in (0, 1):
        fs = [f for f in cands if (f[0] + f[1]) % 2 == colour]
        cover = {}
        ok = True
        for f in fs:
            for pair in face_edges(f):
                for e in pair:
                    if e in cover:
                        ok = False
                    cover[e] = f
        if ok and set(cover) == E:
            return fs
    return None


def descend(n, w):
    """One step of the weight recursion: weights on A_n -> weights on A_{n-1}."""
    out = {}
    for f in active_faces(n):
        (b, t), (l, r) = face_edges(f)
        al, ga, be, de = w[b], w[t], w[l], w[r]
        DP = al * ga + be * de
        if DP == 0:
            raise ZeroDivisionError("vanishing double product")
        out[b], out[t] = ga / DP, al / DP
        out[l], out[r] = de / DP, be / DP
    # keep only edges that survive into A_{n-1}
    keep = set(edges_of(n - 1)) if n >= 2 else set()
    return {e: v for e, v in out.items() if e in keep}


def shuffle_up(n, match, w, rng):
    """Matching of A_{n-1} -> matching of A_n, using weights w on A_n."""
    new = set()
    for f in active_faces(n):
        (b, t), (l, r) = face_edges(f)
        present = [e for e in (b, t, l, r) if e in match]
        if len(present) == 2:
            continue                                   # full -> empty
        if len(present) == 1:
            e = present[0]
            opp = {b: t, t: b, l: r, r: l}[e]
            new.add(opp)                               # slide to opposite edge
            continue
        if len(present) == 0:
            al, ga, be, de = w[b], w[t], w[l], w[r]
            p = al * ga / (al * ga + be * de)
            if rng.random() < p:
                new.add(b); new.add(t)
            else:
                new.add(l); new.add(r)
    return new


def sample(n, weight_fn, rng):
    """weight_fn(edge) -> weight on A_n.  Returns a matching of A_n."""
    ws = [None] * (n + 1)
    ws[n] = {e: float(weight_fn(e)) for e in edges_of(n)}
    for m in range(n, 1, -1):
        ws[m - 1] = descend(m, ws[m])
    match = set()
    for m in range(1, n + 1):
        match = shuffle_up(m, match, ws[m], rng)
    return match


if __name__ == "__main__":
    import collections
    rng = np.random.default_rng(0)

    print("active-face colour resolves uniquely?")
    for n in range(1, 7):
        fs = active_faces(n)
        print(f"  n={n}: {'yes' if fs is not None else 'NO'}  "
              f"({len(fs) if fs else 0} active faces, colour "
              f"{(fs[0][0]+fs[0][1]) % 2 if fs else '-'})")

    print("\nuniform weights: matching counts and uniformity")
    for n in (1, 2, 3):
        cnt = collections.Counter()
        trials = 20000
        for _ in range(trials):
            m = sample(n, lambda e: 1.0, rng)
            cnt[tuple(sorted(m))] += 1
        exp = 2 ** (n * (n + 1) // 2)
        obs = np.array(list(cnt.values()), float)
        e = trials / len(cnt)
        chi2 = ((obs - e) ** 2 / e).sum()
        sizes = {len(k) for k in cnt}
        print(f"  n={n}: {len(cnt)} distinct matchings (expect {exp}), "
              f"sizes {sizes} (expect {{{n*(n+1)}}}), "
              f"chi2={chi2:.1f} on {len(cnt)-1} dof")
