"""
Genus-2 spectral curve for the k=2, l=3 periodic Aztec diamond, and its period
matrix -- the smallest genuinely MULTIVARIATE case (g = (k-1)(l-1) = 2).

From Berggren-Nicoletti (10), (11), (13) and Lemma 2.6, with k = 2:

    phi_{2i-1}(z) = [[gamma_{1,i}, alpha_{2,i}/z], [alpha_{1,i}, gamma_{2,i}]]
    phi_{2i}(z)   = 1/(1 - bv_i/z) * [[1, beta_{2,i}/z], [beta_{1,i}, 1]]
    bv_i          = beta_{1,i} beta_{2,i}
    Phi(z)        = phi_1 phi_2 ... phi_{2l}
    P(z,w)        = prod_i (1 - bv_i/z) * det(Phi(z) - w I)

Since k = 2, P is quadratic in w, so the curve is hyperelliptic:

    y^2 = D(z) := tr(Phi)^2 - 4 det(Phi)

For an M-curve of genus 2 the relevant part of D has six real branch points.
With e_1 < ... < e_6, the compact ovals (the A cycles) sit over the segments
where D > 0 that do not run off to infinity, and the B cycles over the
complementary segments -- exactly as in the genus-1 case, but now 2x2.

Period matrix: with the unnormalised holomorphic differentials z^{j-1} dz / y,
    A_ij = oint_{A_i} z^{j-1} dz/y,   Bt_ij = oint_{B_i} z^{j-1} dz/y
the normalised period matrix is  B = A^{-1} Bt  (then symmetrised in the usual
way); for an M-curve it must be symmetric with positive definite imaginary part.
"""
import mpmath as mp

mp.mp.dps = 30


def Phi(z, k, l, alpha, beta, gamma):
    """Transfer matrix product, k = 2 only."""
    assert k == 2
    M = mp.matrix([[1, 0], [0, 1]])
    pref = mp.mpf(1)
    for i in range(l):
        a1, a2 = alpha[0][i], alpha[1][i]
        g1, g2 = gamma[0][i], gamma[1][i]
        b1, b2 = beta[0][i], beta[1][i]
        odd = mp.matrix([[g1, a2 / z], [a1, g2]])
        bv = b1 * b2
        even = mp.matrix([[1, b2 / z], [b1, 1]]) / (1 - bv / z)
        pref *= (1 - bv / z)
        M = M * odd * even
    return M, pref


def Dz(z, k, l, alpha, beta, gamma):
    M, _ = Phi(z, k, l, alpha, beta, gamma)
    tr = M[0, 0] + M[1, 1]
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    return tr * tr - 4 * det


def branch_points(k, l, alpha, beta, gamma, lo=-60.0, hi=60.0, n=40000):
    """Sign changes of D(z) on the real axis."""
    f = lambda z: Dz(mp.mpf(z), k, l, alpha, beta, gamma)
    xs = [lo + (hi - lo) * i / n for i in range(n + 1)]
    roots = []
    prev = None
    for x in xs:
        if abs(x) < 1e-9:
            prev = None
            continue
        try:
            v = mp.re(f(x))
        except ZeroDivisionError:
            prev = None
            continue
        if prev is not None and mp.sign(v) != mp.sign(prev[1]):
            try:
                r = mp.findroot(lambda t: mp.re(f(t)), (prev[0] + x) / 2)
                if all(abs(r - q) > 1e-8 for q in roots):
                    roots.append(r)
            except Exception:
                pass
        prev = (x, v)
    return sorted(roots)


if __name__ == "__main__":
    k, l = 2, 3
    alpha = [[mp.mpf('1.0'), mp.mpf('0.6'), mp.mpf('1.4')],
             [mp.mpf('1.3'), mp.mpf('1.0'), mp.mpf('0.7')]]
    beta = [[mp.mpf('0.8'), mp.mpf('1.2'), mp.mpf('1.0')],
            [mp.mpf('1.1'), mp.mpf('0.9'), mp.mpf('1.3')]]
    gamma = [[mp.mpf(1)] * 3, [mp.mpf(1)] * 3]

    r = branch_points(k, l, alpha, beta, gamma)
    print(f"real branch points of y^2 = D(z):  ({len(r)} found)")
    for x in r:
        print(f"   {mp.nstr(x, 12)}")
    print("\nexpect 6 for a genus-2 M-curve "
          "(g = (k-1)(l-1) = 2, so 2g+2 = 6 branch points)")
