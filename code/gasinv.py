"""
Full-plane gas inverse Kasteleyn matrix, from Chhita-Johansson eq. (2.2).

Why: our exact determinant method dies at n ~ 60 because cond(K) ~ 10^(0.252 n)
-- the frozen corners produce exponentially small singular values.  But
Theorem 2.6 of arXiv:1410.2385 says that in the GAS region

    K^{-1}_{a,1}(x,y) = K^{-1}_{1,1}(x,y) + O(e^{-c1 m}),

i.e. the finite-n inverse equals the FULL-PLANE gas inverse up to an
exponentially small error.  So accuracy there IMPROVES with n, exactly inverting
the situation with the matrix method.

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
    u = (d[0] + d[1]) // 2
    v = (d[1] - d[0]) // 2
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
