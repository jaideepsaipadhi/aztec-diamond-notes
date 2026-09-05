"""
Genus-2 amoeba and polygon maps: TWO gas facets.

Why this is attackable now.  The multivariate test was blocked on (a) the
canonical homology basis of the algebraic curve and (b) the conditioning wall at
n~60 in the determinant method.  Both belong to a route we no longer need.  The
Fock/Schottky route bypasses them: the period matrix comes from the Schottky
group (validated against jtem to 1e-11), and the measurement comes from sampling
with the validated shuffler rather than from determinants.

Structural prediction to test (BBS Prop 31, the genus-1 version of which we
verified to 10 digits): Im zeta_k is CONSTANT on each compact oval, and the two
ovals give two DISTINCT interior points of the Newton polygon -- the two gas
facet slopes.

zeta^{a}(z) = sum_{sigma in Gamma} log[(z - sigma a^-)/(z - sigma a^+)]
(the z0 = infinity degeneration of the cross-ratio, as in genus 1).
"""
import itertools

import mpmath as mp

mp.mp.dps = 20


def gen_matrix(A, B, mu):
    return [A - mu*B, mu*B*A - A*B, 1 - mu, mu*A - B]


def inv_matrix(M):
    det = M[0]*M[3] - M[1]*M[2]
    return [M[3]/det, -M[1]/det, -M[2]/det, M[0]/det]


def mul(P, Q):
    return [P[0]*Q[0]+P[1]*Q[2], P[0]*Q[1]+P[1]*Q[3],
            P[2]*Q[0]+P[3]*Q[2], P[2]*Q[1]+P[3]*Q[3]]


def moebius(M, z):
    num, den = M[0]*z + M[1], M[2]*z + M[3]
    return mp.inf if den == 0 else num/den


def group(fixed, mus, maxlen=4):
    """All reduced words up to maxlen, as matrices (identity included)."""
    gens = [gen_matrix(A, B, mu) for (A, B), mu in zip(fixed, mus)]
    invs = [inv_matrix(g) for g in gens]
    alpha = [i+1 for i in range(len(gens))] + [-(i+1) for i in range(len(gens))]
    mat = lambda i: gens[i-1] if i > 0 else invs[-i-1]
    out = [[1, 0, 0, 1]]
    for L in range(1, maxlen+1):
        for wd in itertools.product(alpha, repeat=L):
            if any(wd[i] == -wd[i+1] for i in range(len(wd)-1)):
                continue
            M = [1, 0, 0, 1]
            for s in wd:
                M = mul(M, mat(s))
            out.append(M)
    return out


def zeta(z, minus, plus, elems):
    tot = mp.mpc(0)
    for M in elems:
        u, v = moebius(M, minus), moebius(M, plus)
        tot += mp.log((z - u)/(z - v))
    return tot


def iso_circle(A, B, mu):
    a, b, c, d = gen_matrix(A, B, mu)
    s = mp.sqrt(a*d - b*c)
    a, b, c, d = a/s, b/s, c/s, d/s
    return mp.conj(-d/c), 1/abs(c)


if __name__ == "__main__":
    fixed = [(mp.mpc('1.2', '0.4'), mp.mpc('1.2', '-0.4')),
             (mp.mpc('-1.2', '0.4'), mp.mpc('-1.2', '-0.4'))]
    mus = [mp.mpf('0.03'), mp.mpf('0.03')]
    el = group(fixed, mus, maxlen=3)
    print(f"group elements used: {len(el)}")

    # train tracks on the real axis, interleaved with the two ovals' projections
    tracks = {'a+': mp.mpf('-2.6'), 'a-': mp.mpf('2.6'),
              'b+': mp.mpf('-0.5'), 'b-': mp.mpf('0.5')}
    print(f"train tracks: {[(k, float(v)) for k, v in tracks.items()]}\n")

    print("Prop 31 test: Im zeta_k must be CONSTANT on each compact oval")
    for k, ((A, B), mu) in enumerate(zip(fixed, mus)):
        cen, rad = iso_circle(A, B, mu)
        vals = []
        for j in range(6):
            z = cen + rad*mp.e**(2j*mp.pi*j/6)
            z1 = zeta(z, tracks['a-'], tracks['a+'], el)
            z2 = zeta(z, tracks['b-'], tracks['b+'], el)
            vals.append((mp.im(z1), mp.im(z2)))
        s1 = max(abs(v[0]-vals[0][0]) for v in vals)
        s2 = max(abs(v[1]-vals[0][1]) for v in vals)
        D = (-vals[0][1]/mp.pi, vals[0][0]/mp.pi)
        print(f"  oval {k+1}: spread of Im zeta over the circle "
              f"{mp.nstr(s1,3)}, {mp.nstr(s2,3)}")
        print(f"           polygon map Delta = ({mp.nstr(D[0],10)}, "
              f"{mp.nstr(D[1],10)})")
