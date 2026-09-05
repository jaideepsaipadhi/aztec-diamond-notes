"""
Domino shuffling for the Aztec diamond -- STAGE 1: uniform weights.

Design note (learned the hard way): a domino's direction is NOT a function of
the parity of its position.  The empty 2x2 blocks left after a slide do not sit
at a fixed parity class -- they alternate.  So the type is carried on the domino:
creation fills a block with an outward-facing pair, and that determines the type
once and for all.  No global parity bookkeeping anywhere.

Cells (i,j) are unit squares; AD(n) = { (i,j) : |i+1/2| + |j+1/2| <= n }.

  creation in block anchored (i,j):  N at (i,j) over S at (i+1,j)      [horizontal]
                                 or  W at (i,j) left of E at (i,j+1)   [vertical]
  destruction:  S at (i,j) over N at (i+1,j),  or  E at (i,j) left of W at (i,j+1)

Directions: N=(-1,0) up, S=(1,0) down, W=(0,-1) left, E=(0,1) right.
"""
import numpy as np

N, S, E, W = 1, 2, 3, 4
HORIZ = (N, S)
DIRS = {N: (-1, 0), S: (1, 0), W: (0, -1), E: (0, 1)}
NAME = {N: "N", S: "S", E: "E", W: "W"}


def in_ad(i, j, n):
    return abs(i + 0.5) + abs(j + 0.5) <= n


class Tiling:
    def __init__(self, n=0):
        self.n = n
        self.dom = {}          # anchor (top-left cell) -> type

    def second_cell(self, ij, typ):
        i, j = ij
        return (i, j + 1) if typ in HORIZ else (i + 1, j)

    def occupied(self):
        occ = set()
        for a, t in self.dom.items():
            occ.add(a)
            occ.add(self.second_cell(a, t))
        return occ


def destroy(t):
    doomed = set()
    for (i, j), typ in t.dom.items():
        if typ == S and t.dom.get((i + 1, j)) == N:
            doomed |= {(i, j), (i + 1, j)}
        if typ == E and t.dom.get((i, j + 1)) == W:
            doomed |= {(i, j), (i, j + 1)}
    r = Tiling(t.n)
    r.dom = {k: v for k, v in t.dom.items() if k not in doomed}
    return r


def slide(t):
    r = Tiling(t.n + 1)
    for (i, j), typ in t.dom.items():
        di, dj = DIRS[typ]
        r.dom[(i + di, j + dj)] = typ
    return r


def create(t, rng):
    n = t.n
    occ = t.occupied()
    for i in range(-n, n):
        for j in range(-n, n):
            blk = [(i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1)]
            if not all(in_ad(a, b, n) for a, b in blk):
                continue
            if any(c in occ for c in blk):
                continue
            if rng.random() < 0.5:
                t.dom[(i, j)] = N
                t.dom[(i + 1, j)] = S
            else:
                t.dom[(i, j)] = W
                t.dom[(i, j + 1)] = E
            occ |= set(blk)
    return t


def sample(n, rng):
    t = Tiling(0)
    for _ in range(n):
        t = create(slide(destroy(t)), rng)
    return t


# ------------------------------------------------------------------ validation
def is_valid(t):
    n = t.n
    cover = set()
    for a, typ in t.dom.items():
        for c in (a, t.second_cell(a, typ)):
            if c in cover:
                return False, f"double cover at {c}"
            if not in_ad(c[0], c[1], n):
                return False, f"outside AD({n}) at {c}"
            cover.add(c)
    want = sum(1 for i in range(-n, n) for j in range(-n, n) if in_ad(i, j, n))
    if len(cover) != want:
        return False, f"covered {len(cover)} of {want}"
    return True, "ok"


def canon(t):
    """Shape only: (anchor, orientation).  Types are internal bookkeeping."""
    return tuple(sorted((i, j, "H" if typ in HORIZ else "V")
                        for (i, j), typ in t.dom.items()))


def brute_force(n):
    cells = sorted((i, j) for i in range(-n, n) for j in range(-n, n) if in_ad(i, j, n))
    cellset = set(cells)
    found = set()

    def rec(used, placed):
        if len(used) == len(cells):
            found.add(tuple(sorted(placed)))
            return
        c = next(x for x in cells if x not in used)
        i, j = c
        for horiz in (True, False):
            c2 = (i, j + 1) if horiz else (i + 1, j)
            if c2 in cellset and c2 not in used:
                rec(used | {c, c2}, placed + [(i, j, "H" if horiz else "V")])

    rec(frozenset(), [])
    return found


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("validity")
    for n in (1, 2, 3, 5, 10, 25, 60):
        bad = None
        for _ in range(30):
            ok, msg = is_valid(sample(n, rng))
            if not ok:
                bad = msg
                break
        t = sample(n, rng)
        print(f"  n={n:<3} {'FAIL ' + bad if bad else 'ok'}   "
              f"dominoes={len(t.dom)} (expect {n*(n+1)})")

    print("\nuniformity vs brute force")
    for n in (1, 2, 3):
        all_t = brute_force(n)
        trials = 24000
        counts = {}
        for _ in range(trials):
            k = canon(sample(n, rng))
            counts[k] = counts.get(k, 0) + 1
        obs = np.array([counts.get(k, 0) for k in sorted(all_t)], float)
        e = trials / len(all_t)
        chi2 = ((obs - e) ** 2 / e).sum()
        dof = len(all_t) - 1
        print(f"  n={n}: {len(all_t)} tilings (expect {2**(n*(n+1)//2)}), "
              f"hit {len(counts)}, outside enumeration {len(set(counts)-all_t)}, "
              f"chi2={chi2:.1f} on {dof} dof")
