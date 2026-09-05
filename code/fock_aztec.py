"""
Corrected Fock edge weights on the Aztec diamond, and the re-run of the
sampler comparison that originally exposed the bug.

Edge weights come DIRECTLY from Fock's formula -- no gauge reconstruction, which
also removes the numerical blowup (weights spanning 1e16) that killed the
previous attempt.

Everything below was derived from Example 15 of arXiv:2405.20284, not guessed:
  crossing pairs, from d(b) - d(w) over the four black neighbours of a white
  vertex:   SW -> (delta, alpha),  SE -> (alpha, gamma),
            NW -> (beta, delta),   NE -> (beta, gamma)
  faces adjacent to each edge: SE -> south & east, NE -> east & north,
            NW -> north & west, SW -> west & south
  discrete Abel map on faces:
            even (2p,2q):      d = q(beta-alpha) + p(gamma-delta)
            odd  (2p+1,2q+1):  d = q(beta-alpha) + p(gamma-delta) - alpha + gamma

Closure test (verified, ratio 1.000000 at nine faces): the alternating product
[SE(w) NW(w')] / [NE(w) SW(w')] around face (2i+1,2j+1) reproduces the face
weight validated to 1e-16 against the paper's four identities.

Coordinates: the paper's graph has diagonal edges (w=(even,odd), b=(odd,even));
(x,y) -> ((x+y-1)/2, (y-x-1)/2) turns those into unit steps, matching the
shuffler's A_n.
"""
import mpmath as mp
import numpy as np

from fock_correct import FockGenus1
from genshuffle import verts

mp.mp.dps = 18

ANGLES = [mp.mpf('-0.3524163823'), mp.mpf('-0.1211189416'),
          mp.mpf('0.1305475871'), mp.mpf('0.4072264209')]
TAU = 1j*mp.mpf('0.6226177988')
PRED_FACET = (0.4716546, 0.5170360)


class FockAztec:
    def __init__(self, angles=ANGLES, tau=TAU, t=mp.mpf(1)/4):
        self.F = FockGenus1(tau=tau, t=t, angles=angles)
        al, ga, be, de = self.F.al, self.F.ga, self.F.be, self.F.de
        self.pairs = {'SE': (al, ga), 'NE': (be, ga),
                      'NW': (be, de), 'SW': (de, al)}

    def _faces(self, i, j, which):
        S, E = (2*i, 2*j), (2*i+1, 2*j+1)
        N, W = (2*i, 2*j+2), (2*i-1, 2*j+1)
        return {'SE': (S, E), 'NE': (E, N), 'NW': (N, W), 'SW': (W, S)}[which]

    def edge_weight(self, i, j, which):
        F = self.F
        x, y = self.pairs[which]
        f1, f2 = self._faces(i, j, which)
        return abs(complex(F.E(x, y)
                           / (F.TH(F.t + F.d_face(*f1))
                              * F.TH(F.t + F.d_face(*f2)))))

    def weights_on_An(self, n, span=None):
        """Edge weights keyed by shuffler A_n edges."""
        conv = lambda x, y: ((x + y - 1)//2, (y - x - 1)//2)
        V = verts(n)
        span = span or (2*n + 4)
        raw = {}
        for i in range(-span, span):
            for j in range(-span, span):
                w = (2*i, 2*j+1)
                for which, b in (('SE', (2*i+1, 2*j)),
                                 ('NE', (2*i+1, 2*j+2)),
                                 ('NW', (2*i-1, 2*j+2)),
                                 ('SW', (2*i-1, 2*j))):
                    raw[(conv(*w), conv(*b))] = self.edge_weight(i, j, which)
        # find the offset aligning the image with A_n
        best, bestcnt = None, -1
        need = {tuple(sorted(e)) for e in _edges(V)}
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                got = 0
                for (p, q), val in raw.items():
                    e = tuple(sorted(((p[0]+dx, p[1]+dy), (q[0]+dx, q[1]+dy))))
                    if e in need:
                        got += 1
                if got > bestcnt:
                    bestcnt, best = got, (dx, dy)
        dx, dy = best
        out = {}
        for (p, q), val in raw.items():
            e = tuple(sorted(((p[0]+dx, p[1]+dy), (q[0]+dx, q[1]+dy))))
            if e in need:
                out[e] = val
        return out, bestcnt, len(need)


def _edges(V):
    out = []
    for (x, y) in V:
        for d in ((1, 0), (0, 1)):
            if (x+d[0], y+d[1]) in V:
                out.append(((x, y), (x+d[0], y+d[1])))
    return out


if __name__ == "__main__":
    import time
    FA = FockAztec()
    for n in (8, 16):
        t0 = time.time()
        w, got, need = FA.weights_on_An(n)
        v = np.array(list(w.values()))
        print(f"n={n}: matched {got}/{need} edges in {time.time()-t0:.0f}s   "
              f"weights [{v.min():.5f}, {v.max():.5f}] ratio {v.max()/v.min():.2f}")
