"""
Is v rational?  Now decidable, because v comes from the curve.

abel_v.py found that the shift velocity is an explicit multiple of the Abel
image of the angle over z = 0:

    genus 1, (2,2):   v = -B        B = u(0_+) = -1/4  EXACTLY
    genus 3, (2,4):   v = -2B       matched the measurement at 1 sigma

and that A - B is exactly (-1/4, -1/2, -3/4), a 4-torsion relation.  At genus 1
B is a quarter-integer, so v is rational, the orbit closes with period 4, and
that is precisely why Section 4.6 shows nothing quasi-periodic.

Sampling can never prove a number irrational.  A computed B can be tested far
harder: if B = p/q with modest q, its continued fraction terminates and a
convergent reproduces it to full working precision.  If instead the continued
fraction keeps producing small partial quotients and no convergent with
q < 10^6 matches to 20 digits, B is irrational for every practical purpose and
the orbit does not close.

Everything is carried at 40 digits; the genus-1 case is used to confirm that the
quadrature really does deliver exact quarter-integers, i.e. that the precision
claimed here is real.
"""
import os

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def abel_hi(roots, dps=40):
    """A = u(inf_-), B = u(0_+) at high precision, as mpmath values."""
    mp.mp.dps = dps
    e = [mp.mpf(r) for r in sorted(roots)]
    g = len(e) // 2 - 1

    def absD(z):
        p = mp.mpf(1)
        for r in e:
            p *= (z - r)
        return abs(p)

    Ahat = mp.zeros(g, g)
    for k in range(g):
        lo, hi = e[2 * k + 1], e[2 * k + 2]
        for j in range(g):
            f = lambda z: z ** j / mp.sqrt(absD(z))
            Ahat[k, j] = (-1) ** k * 2 * mp.quad(f, [lo, (lo + hi) / 2, hi])

    Iinf = mp.matrix(g, 1)
    Izero = mp.matrix(g, 1)
    for j in range(g):
        f = lambda z: z ** j / mp.sqrt(absD(z))
        Iinf[j] = -2 * mp.quad(f, [e[-1], e[-1] + 1, mp.inf])
        Izero[j] = -mp.quad(f, [0, 1, mp.inf])

    Ainv = Ahat ** -1
    A = [sum(Iinf[k] * Ainv[k, j] for k in range(g)) for j in range(g)]
    B = [sum(Izero[k] * Ainv[k, j] for k in range(g)) for j in range(g)]
    return [mp.re(x) for x in A], [mp.re(x) for x in B]


def cfrac(x, n=25):
    """Continued fraction of x, with the convergents."""
    x = mp.mpf(x)
    terms, convs = [], []
    p0, q0, p1, q1 = 0, 1, 1, 0
    y = x
    for _ in range(n):
        a = int(mp.floor(y))
        terms.append(a)
        p0, q0, p1, q1 = p1, q1, a * p1 + p0, a * q1 + q0
        convs.append((p1, q1, abs(x - mp.mpf(p1) / q1)))
        fr = y - a
        if fr < mp.mpf(10) ** (-(mp.mp.dps - 5)):
            break
        y = 1 / fr
    return terms, convs


def report(label, vals, dps):
    print(f"\n  {label}")
    for i, x in enumerate(vals):
        xx = mp.fmod(mp.fmod(x, 1) + 1, 1)
        terms, convs = cfrac(xx)
        p, q, err = convs[-1]
        exact = err < mp.mpf(10) ** (-(dps - 6))
        print(f"    component {i+1}: {mp.nstr(xx, 22)}")
        print(f"      continued fraction {terms[:12]}"
              + (" ..." if len(terms) >= 12 else ""))
        if exact and q < 10 ** 8:
            print(f"      -> RATIONAL: {p}/{q}  (error {mp.nstr(err,3)})")
        else:
            best = min((c for c in convs if c[1] < 10 ** 6),
                       key=lambda c: c[2], default=None)
            if best:
                print(f"      -> no exact rational; best with q < 1e6 is "
                      f"{best[0]}/{best[1]}, off by {mp.nstr(best[2],3)}")


if __name__ == "__main__":
    from abel_v import genus1_roots
    DPS = 40

    print("=" * 70)
    print("  FIXTURE: genus 1, two-periodic Aztec, a = 0.7")
    print("=" * 70)
    A1, B1 = abel_hi(genus1_roots(0.7), DPS)
    report("A = u(inf_-)", A1, DPS)
    report("B = u(0_+)", B1, DPS)
    print("\n  v = -B should be exactly 1/4:")
    print(f"    -B = {mp.nstr(mp.fmod(-B1[0] + 1, 1), 25)}")

    print("\n" + "=" * 70)
    print("  genus 3, configuration A   (v = -2B)")
    print("=" * 70)
    from hyper_curve import Dz, curve
    W = np.load(os.path.join(HERE, 'g3_best_weights.npy'))
    wu = [[mp.mpf(str(x)) for x in q] for q in W[0].tolist()]
    wv = [[mp.mpf(str(x)) for x in q] for q in W[1].tolist()]
    mp.mp.dps = DPS
    g, r0, _ = curve(wu, wv, 2, 4, verbose=False)
    f = lambda t: mp.re(Dz(t, wu, wv, 2, 4))
    r3 = [mp.findroot(f, mp.mpf(str(x)), tol=mp.mpf('1e-55')) for x in r0]
    A3, B3 = abel_hi(r3, DPS)
    print("\n  A - B (expect exactly -1/4, -1/2, -3/4):")
    for i in range(3):
        print(f"    {mp.nstr(A3[i] - B3[i], 25)}")
    report("B = u(0_+)", B3, DPS)
    v = [mp.fmod(mp.fmod(-2 * b, 1) + 1, 1) for b in B3]
    print("\n  v = -2B:")
    for i, x in enumerate(v):
        print(f"    v{i+1} = {mp.nstr(x, 22)}")
    print(f"\n  measured    (0.28548, 0.41538, 0.28539) +- 0.00014")
    report("v = -2B", v, DPS)
