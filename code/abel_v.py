"""
The shift velocity from the CURVE, not from sampling.

Equation (19) writes v as an integer combination of Abel images of the angles --
the points at infinity of the spectral curve.  Applying it verbatim needs
Berggren-Nicoletti's (k, l) parameterisation, which does not fit the models used
here: their angle count 2(k+l) needs a Newton polygon with that many boundary
lattice points, and the diagonal-coordinate family used here gives diamonds with
only four.  (For genus 3 that is area 4 with 4 boundary points, against the area
8 and 12 boundary points their k=2, l=4 would need; no (mu, mv) in the diamond
family gives genus 3 at area 8.)

But the SHAPE of (19) survives without their labels.  On a hyperelliptic curve
the angles are the two points over z = 0 and the two over z = infinity, and the
hyperelliptic involution relates them: with base point infinity_+,

    u(inf_+) = 0,   u(inf_-) = A,   u(0_+) = B,   u(0_-) = A - B.

So ANY integer combination of angle images is  m A + n B  for integers m, n.
That is a two-parameter test against a measured 3-vector v -- heavily
overdetermined, and so a real test rather than a fit.

The integrals, with all 2g+2 branch points real and negative (so (0, +inf) is
free of them and lies on one sheet):

    u(0_+)  ~  - int_0^inf      z^{j-1} dz / sqrt|D|
    u(inf_-) ~ -2 int_{e_last}^inf z^{j-1} dz / sqrt|D|

the second because the outer oval runs from inf_+ in along y > 0 to the last
branch point and back out along y < 0 to inf_-.  Both converge: y ~ z^{g+1} at
infinity and j-1 <= g-1.

Normalisation: with Ahat_{ik} = oint_{A_i} omega_k over the compact ovals (the
alternating branch signs of hyper_g3.py), u(P) = Ihat(P)^T Ahat^{-1}.

GENUS 1 IS THE FIXTURE: there v = 1/4 is known from Remark 4.18 and was measured
at 0.2503 +- 0.0026, so the combination must come out to 1/4.
"""
import itertools
import os
import sys

import mpmath as mp
import numpy as np

mp.mp.dps = 30
HERE = os.path.dirname(os.path.abspath(__file__))


def abel_data(roots):
    """Return (A, B) in R^g/Z^g: the normalised Abel images of inf_- and 0_+."""
    e = [mp.mpf(r) for r in sorted(roots)]
    g = len(e) // 2 - 1

    def absD(z):
        p = mp.mpf(1)
        for r in e:
            p *= (z - r)
        return abs(p)

    # A-periods over the g compact ovals, alternating branch sign
    Ahat = mp.zeros(g, g)
    for k in range(g):
        lo, hi = e[2 * k + 1], e[2 * k + 2]
        for j in range(g):
            f = lambda z: z ** j / mp.sqrt(absD(z))
            Ahat[k, j] = (-1) ** k * 2 * mp.quad(f, [lo, (lo + hi) / 2, hi])

    # unnormalised Abel integrals
    Iinf = mp.matrix(g, 1)
    Izero = mp.matrix(g, 1)
    last = e[-1]
    for j in range(g):
        f = lambda z: z ** j / mp.sqrt(absD(z))
        Iinf[j] = -2 * mp.quad(f, [last, last + 1, mp.inf])
        Izero[j] = -mp.quad(f, [0, 1, mp.inf])

    Ainv = Ahat ** -1
    # u(P)_j = sum_k Ihat_k (Ainv)_{kj}
    A = [sum(Iinf[k] * Ainv[k, j] for k in range(g)) for j in range(g)]
    B = [sum(Izero[k] * Ainv[k, j] for k in range(g)) for j in range(g)]
    return (np.array([float(mp.re(x)) for x in A]),
            np.array([float(mp.re(x)) for x in B]))


def search(v, sev, A, B, M=12):
    """Integer m, n with m A + n B = v (mod 1)."""
    hits = []
    for m in range(-M, M + 1):
        for n in range(-M, M + 1):
            if m == 0 and n == 0:
                continue
            cand = (m * A + n * B) % 1.0
            d = np.abs(((cand - v + 0.5) % 1.0) - 0.5)
            hits.append((d.max(), m, n, cand))
    hits.sort()
    return hits


def genus1_roots(a):
    a = mp.mpf(str(a))
    c = 2 * (1 + a ** 2) / a
    r = []
    for s in (-2, 2):
        bb = c + s
        d = mp.sqrt(bb ** 2 - 4)
        r += [(-bb - d) / 2, (-bb + d) / 2]
    return sorted(r)


if __name__ == "__main__":
    print("=== FIXTURE: genus 1, two-periodic Aztec, a = 0.7, v = 1/4 known ===")
    r = genus1_roots(0.7)
    A, B = abel_data(r)
    print(f"  branch points {[float(mp.nstr(x,8)) for x in r]}")
    print(f"  A = u(inf_-) = {A[0]:.8f}      B = u(0_+) = {B[0]:.8f}")
    v1 = np.array([0.25])
    hits = search(v1, np.array([0.003]), A, B, M=12)
    print(f"\n  closest integer combinations to v = 1/4:")
    for d, m, n, cand in hits[:6]:
        print(f"    {m:+3d} A {n:+3d} B = {cand[0]:.8f}   |diff| = {d:.2e}")

    print("\n=== genus 3, configuration A ===")
    import mpmath
    from hyper_curve import Dz, curve
    W = np.load(os.path.join(HERE, 'g3_best_weights.npy'))
    WU, WV = W[0].tolist(), W[1].tolist()
    mp.mp.dps = 40
    wu = [[mp.mpf(str(x)) for x in q] for q in WU]
    wv = [[mp.mpf(str(x)) for x in q] for q in WV]
    g, r0, interior = curve(wu, wv, 2, 4, verbose=False)
    f = lambda t: mp.re(Dz(t, wu, wv, 2, 4))
    r3 = [mp.findroot(f, mp.mpf(str(x)), tol=mp.mpf('1e-50')) for x in r0]
    mp.mp.dps = 30
    A3, B3 = abel_data(r3)
    print(f"  A = u(inf_-) = {np.round(A3,8)}")
    print(f"  B = u(0_+)   = {np.round(B3,8)}")
    v = np.array([0.28548, 0.41538, 0.28539])
    se = np.array([0.00014, 0.00014, 0.00014])
    print(f"\n  measured v = {v}  +- {se[0]}")
    hits = search(v, se, A3, B3, M=12)
    print(f"\n  closest integer combinations m A + n B:")
    for d, m, n, cand in hits[:8]:
        print(f"    {m:+3d} A {n:+3d} B = {np.round(cand,6)}   "
              f"max|diff| = {d:.5f} = {d/se[0]:.0f} sigma")
