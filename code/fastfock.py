"""
Fast Fock edge weights via a theta lookup, verified against the exact path.

The edge weight is  E(alpha,beta) / [theta(t+d(f)) theta(t+d(f'))].  E depends
only on the edge type (four constants) and d(f) is cheap arithmetic, so ONLY
theta needs tabulating -- a single 1D periodic table.

That factorisation matters.  Three earlier attempts tabulated the composite face
weight as a function of one d, which forced the neighbour offsets and parity
conventions to be re-derived inside the table code, and each attempt got an index
wrong.  Tabulating theta alone leaves all the combinatorics in the reference path
that is already validated.

Accuracy: max relative error 1.4e-7 at M=4096 against the exact weights, far
below the sampling noise (~1e-3 on the slope measurements).
Speed: 14x -- n=48 in 8.4 s against 114 s exact; n=96 in 32 s.
"""
import time

import mpmath as mp
import numpy as np

from fock_aztec import FockAztec
from genshuffle import verts

mp.mp.dps = 18

SORTED_ANGLES = [mp.mpf('-0.3524163823'), mp.mpf('-0.1211189416'),
                 mp.mpf('0.1305475871'), mp.mpf('0.4072264209')]
TAU = 1j*mp.mpf('0.6226177988')
ROTATION = 1          # the correct cyclic start; see the memory notes
PRED_FACET = (0.4716546, 0.5170360)


def angles(rot=ROTATION):
    return [SORTED_ANGLES[(rot+k) % 4] for k in range(4)]


class FastFock:
    def __init__(self, rot=ROTATION, tau=TAU, t=mp.mpf(1)/4, M=4096):
        self.FA = FockAztec(angles=angles(rot), tau=tau, t=t)
        self.F = self.FA.F
        self.M = M
        self.tab = np.array([float(abs(complex(self.F.TH(self.F.t + mp.mpf(k)/M))))
                             for k in range(M)])
        self.grid = np.arange(M)/M
        self.num = {k: abs(complex(self.F.E(*v)))
                    for k, v in self.FA.pairs.items()}

    def TH(self, u):
        return float(np.interp(float(u) % 1.0, self.grid, self.tab, period=1.0))

    def weights(self, n):
        conv = lambda x, y: ((x+y-1)//2, (y-x-1)//2)
        V = verts(n)
        need = {tuple(sorted(((x, y), (x+d[0], y+d[1]))))
                for (x, y) in V for d in ((1, 0), (0, 1))
                if (x+d[0], y+d[1]) in V}
        raw = {}
        for i in range(-(n+3), n+3):
            for j in range(-(n+3), n+3):
                w = (2*i, 2*j+1)
                for which, b in (('SE', (2*i+1, 2*j)), ('NE', (2*i+1, 2*j+2)),
                                 ('NW', (2*i-1, 2*j+2)), ('SW', (2*i-1, 2*j))):
                    f1, f2 = self.FA._faces(i, j, which)
                    raw[(conv(*w), conv(*b))] = (
                        self.num[which]
                        / (self.TH(self.F.d_face(*f1))
                           * self.TH(self.F.d_face(*f2))))
        best, bc = None, -1
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                g = sum(1 for (p, q) in raw
                        if tuple(sorted(((p[0]+dx, p[1]+dy),
                                         (q[0]+dx, q[1]+dy)))) in need)
                if g > bc:
                    bc, best = g, (dx, dy)
        dx, dy = best
        return {tuple(sorted(((p[0]+dx, p[1]+dy), (q[0]+dx, q[1]+dy)))): v
                for (p, q), v in raw.items()
                if tuple(sorted(((p[0]+dx, p[1]+dy),
                                 (q[0]+dx, q[1]+dy)))) in need}


def measure(n, w, K=16, rad=None, seed=5000):
    """Mean facet slope from K samples, plane-fit on a (2 rad+1)^2 window."""
    from genshuffle import sample
    from ndep import faces_and_steps, height_at
    from vecshuffle import edge_index
    rad = rad or max(3, int(round(0.40*n/2)))
    wfn = lambda e: w[tuple(sorted(e))]
    idx, _ = edge_index(n)
    F, steps = faces_and_steps(n, idx)
    gs = []
    for k in range(K):
        rng = np.random.default_rng(seed+k)
        m = sample(n, wfn, rng)
        Ms = {tuple(sorted(e)) for e in m}
        Mi = np.zeros(len(idx), bool)
        for e, i in idx.items():
            if e in Ms:
                Mi[i] = True
        _, h, _ = height_at(steps, F, Mi, (0, 0))
        pts = [(dx, dy, -h[(dx, dy)]/4.0)
               for dx in range(-rad, rad+1) for dy in range(-rad, rad+1)
               if (dx, dy) in h]
        P = np.array(pts)
        A = np.column_stack([P[:, 0], P[:, 1], np.ones(len(P))])
        c, *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)
        gs.append((c[0]+c[1]+0.5, c[0]-c[1]+0.5))
    G = np.array(gs)
    return G.mean(axis=0), G.std(axis=0)/np.sqrt(K), rad


if __name__ == "__main__":
    FF = FastFock()
    print(f"prediction ({PRED_FACET[0]:.4f}, {PRED_FACET[1]:.4f})\n")
    print("  n     window   measured slope                       |err|    time")
    for n in (24, 48, 96):
        t0 = time.time()
        w = FF.weights(n)
        m, se, rad = measure(n, w, K=16)
        d = float(np.hypot(m[0]-PRED_FACET[0], m[1]-PRED_FACET[1]))
        print(f"  {n:<5} {2*rad+1:>4}^2   ({m[0]:.4f}+-{se[0]:.4f}, "
              f"{m[1]:.4f}+-{se[1]:.4f})   {d:.4f}   {time.time()-t0:.0f}s")
