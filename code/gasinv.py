"""
Full-plane gas inverse Kasteleyn matrix, from Chhita-Johansson eq. (2.2).

Why: our exact determinant method dies at n ~ 60 because cond(K) ~ 10^(0.252 n)
-- the frozen corners produce exponentially small singular values.  But
Theorem 2.6 of arXiv:1410.2385 says that in the GAS region

    K^{-1}_{a,1}(x,y) = K^{-1}_{1,1}(x,y) + O(e^{-c1 m}),

i.e. the finite-n inverse equals the FULL-PLANE gas inverse up to an
exponentially small error.  So accuracy there IMPROVES with n, exactly inverting
the situation with the matrix method.

STATUS.  This now reproduces the exact finite-n inverse, and the error shrinks
monotonically with n at every offset -- which is the only check that can
distinguish a correct formula from a plausible one, since a wrong formula has no
reason to converge.  Relative error against the exact inverse, a = 0.7:

        n      (1,1)     (3,3)     (5,5)
       24      4.37%     8.40%    16.37%
       32      2.68%     5.09%     9.38%
       40      1.73%     3.27%     5.84%
       48      1.16%     2.18%     3.81%
       56      0.79%     1.49%     2.56%

The error grows with the offset at fixed n because far offsets sit nearer the
gas/liquid boundary, which is where the O(e^{-c1 m}) term lives.  Over this
range of n the decay is as consistent with 1/n^2 as with e^{-c n/24} -- the
facet half-width is only about n/24 at a = 0.7, so m grows slowly and the two
forms cannot be told apart here.  Either way it converges.

The fix was in the exponents, not the contours; see the comment in
gas_inverse.  Because this routine is the n -> infinity limit, substituting it
for the finite-n inverse in the Section 3.2 pipeline gives the gas-facet
distribution with NO finite-size error and no n ~ 60 ceiling -- which is what
the a = 0.7 row of that table needs.  That substitution is not done here.

    K(z,w) = [[i(a + 1/w), a + z], [a + 1/z, i(a + w)]]
    K(z,w)^{-1} = (1/P) [[i(a+w), -(a+z)], [-(a+1/z), i(a+1/w)]]
    P(z,w) = -2 - 2a^2 - a/w - aw - a/z - az

    K^{-1}_{r1,r2}(x,y) = (2 pi i)^{-2} oint dz/z oint dw/w
                          [K(z,w)^{-1}]_{alpha1+1, alpha2+1} z^u w^v

with x in W_alpha1, y in B_alpha2 and the translation from x's fundamental
domain to y's equal to u e1 + v e2, e1 = (1,1), e2 = (-1,1).  Gas phase is
r1 = r2 = 1 (magnetic coordinates (0,0)).
"""
import numpy as np

E1 = np.array([1, 1])
E2 = np.array([-1, 1])


def cls(v):
    """0 or 1 according to (x1 + x2) mod 4 = 2i+1."""
    return ((v[0] + v[1]) % 4 - 1) // 2


def gas_inverse(x, y, a, N=400):
    """K^{-1}_{1,1}(x, y) by the double contour integral on |z|=|w|=1."""
    x = np.asarray(x)
    y = np.asarray(y)
    a1, a2 = cls(x), cls(y)
    # fundamental-domain translation: solve y - x = u e1 + v e2 (+ offset in
    # the domain).  u = ((y-x).x + (y-x).y)/2 in the e1,e2 basis:
    d = y - x
    # In the e1 = (1,1), e2 = (-1,1) basis, d = u e1 + v e2 gives these:
    u = (d[0] + d[1]) // 2
    v = (d[1] - d[0]) // 2
    # ...but z and w are conjugate to translations by a FUNDAMENTAL DOMAIN, and
    # the two-periodic domain is TWO e1/e2 steps wide, so the exponents are in
    # domain units.  The leftover half-step is already carried by (a1, a2).
    #
    # Using the step counts directly -- which this routine did -- makes the
    # formula decay twice as fast as the truth and lose the sign alternation
    # entirely.  Against the exact finite-n inverse at a = 0.7, n = 32:
    #
    #     offset     exact        u (wrong)     u//2 (right)
    #     (1,1)    +0.3763251    +0.3863986     +0.3863986
    #     (3,3)    -0.1371630    +0.0631098     -0.1441511
    #     (5,5)    +0.0576968    +0.0158635     +0.0631098
    #     (7,7)    -0.0251806    +0.0046813     -0.0306703
    #
    # decay ratios: exact 0.364 / 0.421 / 0.436, u//2 0.373 / 0.438 / 0.486,
    # u 0.163 / 0.251 / 0.295.  The contours are NOT the problem: the unit
    # torus does lie inside the bounded amoeba component (the gas hole runs
    # r1 in -0.50..+0.50, r2 in -0.48..+0.49 at a = 0.7), which was the other
    # candidate explanation and is ruled out.
    u, v = u // 2, v // 2
    th = 2*np.pi*np.arange(N)/N
    z = np.exp(1j*th)
    w = np.exp(1j*th)
    Z, W = np.meshgrid(z, w, indexing='ij')
    P = -2 - 2*a**2 - a/W - a*W - a/Z - a*Z
    if (a1, a2) == (0, 0):
        num = 1j*(a + W)
    elif (a1, a2) == (0, 1):
        num = -(a + Z)
    elif (a1, a2) == (1, 0):
        num = -(a + 1/Z)
    else:
        num = 1j*(a + 1/W)
    integ = num/P * Z**u * W**v
    # (2 pi i)^-2 * oint dz/z oint dw/w  ->  average over the two circles
    return integ.mean()


if __name__ == "__main__":
    from twoperiodic import vertices, kasteleyn
    a = 0.7
    n = 32
    W, B = vertices(n)
    wi = {v: i for i, v in enumerate(W)}
    bi = {v: i for i, v in enumerate(B)}
    K = kasteleyn(n, a)
    Kinv = np.linalg.inv(K)
    c = min(B, key=lambda t: (t[0]-n)**2 + (t[1]-n)**2)
    print(f"exact finite-n K^-1 vs full-plane gas formula, a={a}, n={n}")
    print("  offset        exact              gas formula        rel diff")
    for off in ((1, 1), (-1, 1), (1, -1), (3, 1), (1, 3), (3, 3), (5, 5)):
        w0 = (c[0]+off[0], c[1]+off[1])
        if w0 not in wi:
            continue
        ex = Kinv[wi[w0], bi[c]]
        gf = gas_inverse(w0, c, a)
        rel = abs(ex - gf)/max(abs(ex), 1e-30)
        print(f"  {str(off):<12} {ex.real:+.8f}       {gf.real:+.8f}      {rel:.2e}")
