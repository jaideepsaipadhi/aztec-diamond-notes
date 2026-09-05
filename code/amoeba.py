"""
Amoeba map, polygon map and Ronkin data from Schottky uniformization.

Following Bobenko-Bobenko-Suris (arXiv:2402.08798), Section 8 and Prop 41.
For a Schottky group G in U2 form with Harnack data on the real line ordered

    beta^- < alpha^+ < beta^+ < alpha^-

the Abelian integrals of the third kind are Poincare series (their (66)):

    zeta^{alpha_i}(z) = sum_{sigma in G} log{ z, sigma alpha_i^-, z0, sigma alpha_i^+ }

with the cross-ratio {z1,z2,z3,z4} = (z1-z2)/(z2-z3) * (z3-z4)/(z4-z1).
They take z0 = infinity (base point between alpha^- and beta^-), and then the
cross-ratio degenerates to the simple ratio

    {z, u, infinity, v} = (z - u)/(z - v)

so that   zeta^{alpha}(z) = sum_sigma log[ (z - sigma alpha^-)/(z - sigma alpha^+) ].

Then (Definition 24) the AMOEBA map and POLYGON map are

    A(P)     = (Re zeta_1, Re zeta_2)
    Delta(P) = (1/pi)(-Im zeta_2, Im zeta_1)

with zeta_1 = sum_i zeta^{alpha_i}, zeta_2 = sum_j zeta^{beta_j}.

SHARP CHECK (their Prop 31): the differentials dzeta_k are real on the real
ovals, so Im zeta_k is CONSTANT on each compact oval -- those constants are the
b-periods (their (49)), which are also the interior points of the Newton polygon.
That is a strong, falsifiable prediction and is what we test here.
"""
import mpmath as mp

mp.mp.dps = 25


def group_elements(A, B, mu, nmax):
    """Genus-1 Schottky group <sigma>, sigma with fixed points A,B, multiplier mu.
    Returns the maps sigma^n for |n| <= nmax as functions."""
    def sigma_n(n):
        m = mu ** n
        # (w - A)/(w - B) = m (z - A)/(z - B)
        def f(z):
            t = m * (z - A) / (z - B)
            return (A - B * t) / (1 - t)
        return f
    return [sigma_n(n) for n in range(-nmax, nmax + 1)]


def zeta(z, minus, plus, elems):
    """sum_sigma log[(z - sigma(minus))/(z - sigma(plus))]"""
    tot = mp.mpc(0)
    for s in elems:
        u, v = s(minus), s(plus)
        tot += mp.log((z - u) / (z - v))
    return tot


def maps(z, pts, A, B, mu, nmax=12):
    """pts = (beta_m, alpha_p, beta_p, alpha_m); returns (zeta1, zeta2)."""
    bm, ap, bp, am = pts
    el = group_elements(A, B, mu, nmax)
    z1 = zeta(z, am, ap, el)      # alpha pair
    z2 = zeta(z, bm, bp, el)      # beta pair
    return z1, z2


if __name__ == "__main__":
    # Bobenko et al. Fig. 21 (left) data
    pts = (mp.mpf('-2.4'), mp.mpf('-0.4'), mp.mpf('0.4'), mp.mpf('2.4'))
    A = mp.mpc('0.1', '1.0')
    B = mp.conj(A)
    mu = mp.mpf('0.02')

    print("Schottky data: A = 0.1+i, mu = 0.02, "
          "train tracks (b-,a+,b+,a-) = (-2.4,-0.4,0.4,2.4)\n")

    print("test: Im zeta_k must be CONSTANT on the compact oval")
    print("(the compact oval is the circle |z - c| = r of the isometric circle)")
    c = (A - mu * mp.conj(A)) / (1 - mu)
    r = 2 * mp.im(A) / (1 - mu) / 2      # radius of the isometric circle
    print(f"  isometric circle centre {mp.nstr(c,8)}, radius {mp.nstr(r,8)}")
    print("   theta      Im zeta_1        Im zeta_2")
    for kk in range(8):
        th = 2 * mp.pi * kk / 8
        z = c + r * mp.e ** (1j * th)
        z1, z2 = maps(z, pts, A, B, mu)
        print(f"   {float(th):5.2f}   {mp.nstr(mp.im(z1),10):>15} "
              f"{mp.nstr(mp.im(z2),10):>15}")
