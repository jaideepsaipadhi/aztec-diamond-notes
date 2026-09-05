"""
Exact oracle for the two-periodic Aztec diamond.

Two independent computations of the partition function:
  (1) brute-force enumeration of every tiling, summing products of domino weights
  (2) the Kasteleyn determinant

Agreement between them validates BOTH the weight convention and the Kasteleyn
signs, and gives us exact tiling probabilities to test a weighted sampler
against.  This is rungs 2 and 3 of the validation ladder.

The four alignments of Remark 4.18 are encoded as the four ways of positioning
the 2x2 periodic pattern of weights inside the diamond -- i.e. the two bits
(di, dj) of offset.  Which alignment the sampler realizes is exactly this choice.
"""
import itertools

import numpy as np

from shuffle2 import in_ad


def cells(n):
    return sorted((i, j) for i in range(-n, n) for j in range(-n, n) if in_ad(i, j, n))


def domino_weight(i, j, horiz, a, off):
    """Weight of the domino anchored at (i,j).

    Two-periodic pattern: the weight depends on the position of the anchor
    modulo 2 in both directions, shifted by the alignment offset `off`=(di,dj).
    Weights take the two values a and 1/a (gamma = 1 throughout), matching the
    alpha/beta structure of Remark 4.18.
    """
    di, dj = off
    p = ((i + di) % 2, (j + dj) % 2)
    if horiz:
        return a if p in ((0, 0), (1, 1)) else 1.0 / a
    return a if p in ((0, 1), (1, 0)) else 1.0 / a


def enumerate_tilings(n):
    cs = cells(n)
    cset = set(cs)
    out = []

    def rec(used, placed):
        if len(used) == len(cs):
            out.append(tuple(placed))
            return
        c = next(x for x in cs if x not in used)
        i, j = c
        for horiz in (True, False):
            c2 = (i, j + 1) if horiz else (i + 1, j)
            if c2 in cset and c2 not in used:
                rec(used | {c, c2}, placed + [(i, j, horiz)])

    rec(frozenset(), [])
    return out


def partition_enumerated(n, a, off):
    tot = 0.0
    for t in enumerate_tilings(n):
        w = 1.0
        for (i, j, h) in t:
            w *= domino_weight(i, j, h, a, off)
        tot += w
    return tot


def kasteleyn(n, a, off):
    """|det K| for the Aztec diamond.  Square-lattice convention: horizontal
    edges get factor 1, vertical edges get factor i."""
    cs = cells(n)
    black = [c for c in cs if (c[0] + c[1]) % 2 == 0]
    white = [c for c in cs if (c[0] + c[1]) % 2 == 1]
    assert len(black) == len(white), (len(black), len(white))
    bi = {c: k for k, c in enumerate(black)}
    wi = {c: k for k, c in enumerate(white)}
    K = np.zeros((len(black), len(white)), dtype=complex)
    for (i, j) in cs:
        for horiz, (di, dj) in ((True, (0, 1)), (False, (1, 0))):
            c2 = (i + di, j + dj)
            if c2 not in bi and c2 not in wi:
                continue
            w = domino_weight(i, j, horiz, a, off)
            fac = 1.0 if horiz else 1j
            b, wc = ((i, j), c2) if (i + j) % 2 == 0 else (c2, (i, j))
            K[bi[b], wi[wc]] += w * fac
    return abs(np.linalg.det(K))


if __name__ == "__main__":
    print("uniform check (a=1): partition function must be 2^(n(n+1)/2)")
    for n in (1, 2, 3, 4):
        z_enum = partition_enumerated(n, 1.0, (0, 0))
        z_kast = kasteleyn(n, 1.0, (0, 0))
        print(f"  n={n}: enumerated={z_enum:12.4f}  kasteleyn={z_kast:12.4f}  "
              f"expected={2**(n*(n+1)//2):12d}")

    print("\ntwo-periodic, a=0.7, all four alignments")
    for off in itertools.product((0, 1), repeat=2):
        print(f"  offset {off}")
        for n in (1, 2, 3, 4):
            z_enum = partition_enumerated(n, 0.7, off)
            z_kast = kasteleyn(n, 0.7, off)
            ok = "ok" if abs(z_enum - z_kast) < 1e-8 * max(1, z_enum) else "MISMATCH"
            print(f"    n={n}: enumerated={z_enum:14.6f}  "
                  f"kasteleyn={z_kast:14.6f}  {ok}")
