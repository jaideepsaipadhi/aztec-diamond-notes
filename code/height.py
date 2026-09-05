"""
Thurston height function on Aztec diamond tilings.

Vertices are the CORNERS of cells: integer points (i,j).  Cell (i,j) occupies
[i,i+1] x [j,j+1] and has colour (i+j) mod 2  (0 = black).

Thurston's rule: orient every lattice edge so the BLACK cell lies on its left.
Along that orientation the height changes by
        +1  if the edge is NOT covered by a domino
        -3  if it IS covered (both adjacent cells belong to the same domino).

Consistency is automatic and is the reason this is well defined: going around
any one cell, exactly one of its four edges is covered (the one shared with its
partner), and the black-on-left rule orients all four consistently around the
face, so the circulation is 1+1+1-3 = 0.  We verify this numerically rather
than trusting it.

Coordinates: i increases DOWNWARD, j increases RIGHT.  With (x,y) = (j,-i),
rotating a direction 90 degrees counterclockwise sends (dx,dy) -> (-dy,dx).
  * horizontal edge (i,j)->(i,j+1) separates cell (i-1,j) above from (i,j) below;
    travelling +j puts the UPPER cell on the left.
  * vertical edge (i,j)->(i+1,j) separates cell (i,j-1) left from (i,j) right;
    travelling +i puts the RIGHT cell on the left.
"""
from collections import deque

from shuffle2 import Tiling, sample, in_ad, HORIZ


def covering_map(t):
    """cell -> anchor of the domino covering it."""
    m = {}
    for a, typ in t.dom.items():
        m[a] = a
        m[t.second_cell(a, typ)] = a
    return m


def black(i, j):
    return (i + j) % 2 == 0


def edges(t):
    """Yield (v1, v2, delta) meaning h(v2) = h(v1) + delta, already oriented."""
    n, cov = t.n, covering_map(t)

    def same_domino(cA, cB):
        a, b = cov.get(cA), cov.get(cB)
        return a is not None and a == b

    # horizontal edges: (i,j)-(i,j+1); cells (i-1,j) above, (i,j) below
    for i in range(-n, n + 2):
        for j in range(-n, n + 1):
            up, dn = (i - 1, j), (i, j)
            if not (in_ad(*up, n) or in_ad(*dn, n)):
                continue
            delta = -3 if same_domino(up, dn) else 1
            v1, v2 = (i, j), (i, j + 1)
            # travelling +j puts the UPPER cell on the left
            yield (v1, v2, delta) if black(*up) else (v2, v1, delta)

    # vertical edges: (i,j)-(i+1,j); cells (i,j-1) left, (i,j) right
    for i in range(-n, n + 1):
        for j in range(-n, n + 2):
            lf, rt = (i, j - 1), (i, j)
            if not (in_ad(*lf, n) or in_ad(*rt, n)):
                continue
            delta = -3 if same_domino(lf, rt) else 1
            v1, v2 = (i, j), (i + 1, j)
            # travelling +i puts the RIGHT cell on the left
            yield (v1, v2, delta) if black(*rt) else (v2, v1, delta)


def height(t):
    """BFS the height from a boundary corner.  Returns dict vertex -> h."""
    adj = {}
    for v1, v2, d in edges(t):
        adj.setdefault(v1, []).append((v2, d))
        adj.setdefault(v2, []).append((v1, -d))
    root = min(adj)
    h = {root: 0}
    q = deque([root])
    while q:
        v = q.popleft()
        for w, d in adj[v]:
            if w not in h:
                h[w] = h[v] + d
                q.append(w)
    return h


def check_consistent(t):
    """Every edge must be satisfied by the BFS height -- i.e. path independence."""
    h = height(t)
    bad = 0
    for v1, v2, d in edges(t):
        if v1 in h and v2 in h and h[v2] - h[v1] != d:
            bad += 1
    return bad, len(h)


def circulations(t):
    """Directly verify the face circulation is 0 around every cell."""
    n = t.n
    lookup = {}
    for v1, v2, d in edges(t):
        lookup[(v1, v2)] = d
        lookup[(v2, v1)] = -d
    worst = 0
    for i in range(-n, n):
        for j in range(-n, n):
            if not in_ad(i, j, n):
                continue
            loop = [(i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j), (i, j)]
            s = sum(lookup[(loop[k], loop[k + 1])] for k in range(4))
            worst = max(worst, abs(s))
    return worst


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(7)

    print("height function consistency (uniform a=1 sampler)")
    for n in (1, 2, 3, 6, 12, 25):
        t = sample(n, rng)
        bad, nv = check_consistent(t)
        worst = circulations(t)
        print(f"  n={n:<3} vertices={nv:<6} violated edges={bad:<4} "
              f"max |face circulation|={worst}")

    print("\nheight range and boundary behaviour")
    for n in (4, 8, 16):
        t = sample(n, rng)
        h = height(t)
        vals = list(h.values())
        print(f"  n={n:<3} min={min(vals):<5} max={max(vals):<5} "
              f"span={max(vals)-min(vals):<5} (4n = {4*n})")
