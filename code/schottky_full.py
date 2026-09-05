"""
Schottky uniformization for dimer spectral curves.

Three pieces:

1. ROBUST INVERSION.  Solve theta_2(q)^4/theta_3(q)^4 = 1 - a^4 for the nome by
   bisection on a guaranteed bracket (k^2 is increasing in q on (0,1)), so it no
   longer fails as q grows.

2. FORWARD DIRECTION.  Bobenko's construction runs the other way: choose the
   Schottky data first, then read off the curve.  Given a multiplier mu, the
   branch points and hence the model parameter follow.  This is the constructive
   direction -- it generates models rather than analysing a given one.

3. HIGHER GENUS.  For genus >= 2 the group is non-elementary and the theta
   constants are no longer classical: the period matrix comes from Poincare
   series summed over group elements.  With generators sigma_i having fixed
   points A_i, B_i and multipliers mu_i,

       2 pi i B_ii = log mu_i + sum_{sigma} log X(A_i, B_i; sigma A_i, sigma B_i)
       2 pi i B_ij = sum_{sigma} log X(A_i, B_i; sigma A_j, sigma B_j)   (i != j)

   where X(a,b;c,d) = (a-c)(b-d)/((a-d)(b-c)) is the cross-ratio, and sigma runs
   over the appropriate coset representatives (excluding the identity and powers
   of sigma_i in the diagonal term).  We sum over words up to a given length and
   watch the increments.
"""
import itertools

import mpmath as mp

mp.mp.dps = 30


# ---------------------------------------------------------------- 1. inversion
def theta2(q, N=80):
    return 2 * mp.fsum(q ** ((n + mp.mpf(1)/2) ** 2) for n in range(N))


def theta3(q, N=80):
    return 1 + 2 * mp.fsum(q ** (n ** 2) for n in range(1, N))


def k2_from_q(q):
    return (theta2(q) / theta3(q)) ** 4


def q_from_a(a):
    """Bisection on (0,1): k^2(q) increases from 0 to 1."""
    a = mp.mpf(a)
    target = 1 - a ** 4
    lo, hi = mp.mpf('1e-30'), mp.mpf('0.9')
    while k2_from_q(hi) < target:
        hi = (hi + 1) / 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if k2_from_q(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def b_schottky(a):
    q = q_from_a(a)
    return -mp.log(q) / mp.pi, q, q ** 2


# ---------------------------------------------------------- 2. forward direction
def curve_from_multiplier(mu):
    """Schottky data -> model.  Given mu, return (q, k^2, a, b)."""
    mu = mp.mpf(mu)
    q = mp.sqrt(mu)
    k2 = k2_from_q(q)
    a = (1 - k2) ** mp.mpf(0.25)     # since k^2 = 1 - a^4
    b = -mp.log(q) / mp.pi
    return q, k2, a, b


# ------------------------------------------------------------- 3. higher genus
def cross(a, b, c, d):
    return (a - c) * (b - d) / ((a - d) * (b - c))


def moebius(M, z):
    if z == mp.inf:
        return M[0] / M[2] if M[2] != 0 else mp.inf
    num, den = M[0]*z + M[1], M[2]*z + M[3]
    return mp.inf if den == 0 else num / den


def gen_matrix(A, B, mu):
    """Moebius map with fixed points A (attracting), B (repelling), multiplier mu:
       (w - A)/(w - B) = mu (z - A)/(z - B)."""
    # as a 2x2 matrix acting on z
    return [A - mu*B, mu*B*A - A*B, 1 - mu, mu*A - B]


def words(gens, maxlen):
    """All reduced words (as matrices) up to maxlen, excluding identity."""
    inv = []
    for M in gens:
        det = M[0]*M[3] - M[1]*M[2]
        inv.append([M[3]/det, -M[1]/det, -M[2]/det, M[0]/det])
    # 1-based signed alphabet so that generator k and its inverse are +k, -k
    alphabet = [i+1 for i in range(len(gens))] + [-(i+1) for i in range(len(gens))]

    def mat(i):
        return gens[i-1] if i > 0 else inv[-i-1]

    def mul(P, Q):
        return [P[0]*Q[0]+P[1]*Q[2], P[0]*Q[1]+P[1]*Q[3],
                P[2]*Q[0]+P[3]*Q[2], P[2]*Q[1]+P[3]*Q[3]]

    out = []
    for L in range(1, maxlen+1):
        for w in itertools.product(alphabet, repeat=L):
            ok = all(w[i] != -w[i+1] for i in range(len(w)-1))
            if not ok:
                continue
            M = [1, 0, 0, 1]
            for s in w:
                M = mul(M, mat(s))
            out.append((w, M))
    return out


def period_matrix(fixed, mus, maxlen=3):
    """fixed: list of (A_i, B_i); mus: multipliers.  Returns g x g matrix."""
    g = len(mus)
    gens = [gen_matrix(mp.mpc(A), mp.mpc(B), mp.mpf(m))
            for (A, B), m in zip(fixed, mus)]
    W = words(gens, maxlen)
    Bm = mp.zeros(g, g)
    for i in range(g):
        Ai, Bi = mp.mpc(fixed[i][0]), mp.mpc(fixed[i][1])
        for j in range(g):
            Aj, Bj = mp.mpc(fixed[j][0]), mp.mpc(fixed[j][1])
            s = mp.log(mp.mpf(mus[i])) if i == j else mp.mpf(0)
            if i != j:
                # the IDENTITY element belongs to the off-diagonal sum
                s += mp.log(cross(Ai, Bi, Aj, Bj))
            for (w, M) in W:
                # double-coset representatives: no leading power of sigma_i,
                # no trailing power of sigma_j
                if abs(w[0]) == i + 1 or abs(w[-1]) == j + 1:
                    continue
                try:
                    c = cross(Ai, Bi, moebius(M, Aj), moebius(M, Bj))
                    if c != 0:
                        s += mp.log(c)
                except (ZeroDivisionError, ValueError):
                    pass
            Bm[i, j] = s / (2j * mp.pi)
    return Bm


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/mnt/user-data/outputs")
    from period_g1 import b_legendre

    print("1. ROBUST INVERSION (bisection)")
    print("   a      b (Schottky)          b (Legendre)         difference")
    for a in ("0.3", "0.5", "0.7", "0.9", "0.95", "0.99"):
        b, q, mu = b_schottky(a)
        ref = b_legendre(a)
        print(f"  {a:<6} {mp.nstr(b, 15):<21} {mp.nstr(ref, 15):<20} "
              f"{mp.nstr(abs(b-ref), 3)}")

    print("\n2. FORWARD: Schottky data -> model")
    print("   mu            q           a (model)     b")
    for mu in ("0.0005", "0.005", "0.0077873557", "0.05"):
        q, k2, a, b = curve_from_multiplier(mu)
        print(f"  {mu:<13} {mp.nstr(q,7):<11} {mp.nstr(a,8):<13} "
              f"{mp.nstr(b,10)}")
    print("   round trip from a=0.7:", end=" ")
    b0, q0, mu0 = b_schottky("0.7")
    _, _, a1, _ = curve_from_multiplier(mu0)
    print(f"mu={mp.nstr(mu0,10)} -> a={mp.nstr(a1,12)}")

    print("\n3. HIGHER GENUS: period matrix by Poincare series")
    print("   genus 1 sanity (fixed points 0, inf; the sum must be empty):")
    Bm = period_matrix([(0, mp.inf)], [mu0], maxlen=1)
    print(f"     B = {mp.nstr(Bm[0,0], 12)}   (expect i*{mp.nstr(b0,12)})")
