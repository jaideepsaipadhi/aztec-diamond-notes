"""
N-dependence of the facet height distribution.

Height in the shuffling picture: vertices are lattice points, dominoes are edges,
so the height lives on FACES (unit squares).  Crossing an edge between adjacent
faces the height changes by +1 if the edge is unmatched and -3 if matched, with
a sign fixed by the colour of a chosen endpoint.

Self-validating: around any vertex, exactly one of its four incident edges is
matched, so the circulation of the four face-to-face steps around that vertex is
3*(+1) + (-3) = 0.  We check that numerically and flip the convention if needed.
"""
from collections import deque

import numpy as np

from genshuffle import verts
from vecshuffle import build_levels, sample, edge_index


def faces_and_steps(n, idx):
    """Faces of A_n and, for each adjacent pair, the index of the shared edge
    plus the sign of the height step."""
    V = verts(n)
    F = [(x, y) for x in range(-n - 1, n + 1) for y in range(-n - 1, n + 1)
         if any(c in V for c in ((x, y), (x+1, y), (x, y+1), (x+1, y+1)))]
    Fset = set(F)
    steps = []          # (f1, f2, edge_index, sign)
    for (x, y) in F:
        # right neighbour: shared edge is the vertical edge at x+1
        f2 = (x + 1, y)
        if f2 in Fset:
            e = tuple(sorted(((x+1, y), (x+1, y+1))))
            if e in idx:
                s = 1 if (x + 1 + y) % 2 == 0 else -1
                steps.append(((x, y), f2, idx[e], s))
        # up neighbour: shared edge is the horizontal edge at y+1
        f2 = (x, y + 1)
        if f2 in Fset:
            e = tuple(sorted(((x, y+1), (x+1, y+1))))
            if e in idx:
                s = -1 if (x + y + 1) % 2 == 0 else 1
                steps.append(((x, y), f2, idx[e], s))
    return F, steps


def height_at(steps, F, M, target):
    """BFS the height from an arbitrary root; return h(target) - h(root)."""
    adj = {}
    for (f1, f2, ei, s) in steps:
        d = s * (1 - 4 * int(M[ei]))
        adj.setdefault(f1, []).append((f2, d))
        adj.setdefault(f2, []).append((f1, -d))
    root = min(adj)
    h = {root: 0}
    q = deque([root])
    while q:
        v = q.popleft()
        for w, d in adj[v]:
            if w not in h:
                h[w] = h[v] + d
                q.append(w)
    return h.get(target), h, adj


def check_circulation(steps, M, nsample=2000):
    """Verify the height is path independent by BFS-then-check-all-edges."""
    adj = {}
    for (f1, f2, ei, s) in steps:
        d = s * (1 - 4 * int(M[ei]))
        adj.setdefault(f1, []).append((f2, d))
        adj.setdefault(f2, []).append((f1, -d))
    root = min(adj)
    h = {root: 0}
    q = deque([root])
    while q:
        v = q.popleft()
        for w, d in adj[v]:
            if w not in h:
                h[w] = h[v] + d
                q.append(w)
    bad = 0
    for (f1, f2, ei, s) in steps:
        d = s * (1 - 4 * int(M[ei]))
        if f1 in h and f2 in h and h[f2] - h[f1] != d:
            bad += 1
    return bad, len(h)


if __name__ == "__main__":
    import collections
    import time
    rng = np.random.default_rng(12)
    a = 0.7

    print("validating the height convention (violated edges must be 0)")
    for n in (8, 12, 16):
        idx, E = edge_index(n)
        levels, _, _ = build_levels(n, a)
        F, steps = faces_and_steps(n, idx)
        M = sample(n, levels, len(E), rng)
        bad, nf = check_circulation(steps, M)
        print(f"  n={n:<4} faces={nf:<6} violated={bad}")
