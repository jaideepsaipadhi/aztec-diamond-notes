"""
Generalized Ronkin function from Schottky data.

Krichever's framework (arXiv:1310.8472), which BBS follow: the gradient of the
generalized Ronkin function IS the polygon map,

    grad rho (x) = Delta,     x = A(P) = (Re zeta_1, Re zeta_2)
                              Delta   = (1/pi)(-Im zeta_2, Im zeta_1)

with rho convex, affine linear on each component of the amoeba complement, and
Hess rho positive definite in the interior of the amoeba.  So rho is recovered
by integrating the exact 1-form

    d rho = Delta_1 dx_1 + Delta_2 dx_2

along paths in the amoeba.  We parameterise by z in the upper half plane (which
the amoeba map sends onto the amoeba) and integrate in z:

    d rho = Delta . dA/dz  dz.

Checks available without any extra theory:
  * PATH INDEPENDENCE.  d rho is exact, so integrating around a closed loop in
    the upper half plane must give zero.  This is a real test of the whole
    construction -- it fails if either map is wrong.
  * AFFINE ON THE HOLE.  Delta is constant on the compact oval (verified in
    amoeba.py), so rho is affine there, as Krichever requires.
  * CONVEXITY.  Hess rho must be positive definite inside the amoeba; since
    Hess rho = d Delta / d x, we can check it by finite differences.
"""
import mpmath as mp

from amoeba import maps

mp.mp.dps = 20

PTS = (mp.mpf('-2.4'), mp.mpf('-0.4'), mp.mpf('0.4'), mp.mpf('2.4'))
A0 = mp.mpc('0.1', '1.0')
B0 = mp.conj(A0)
MU = mp.mpf('0.02')


def AD(z):
    """Amoeba point x = (Re z1, Re z2) and polygon point Delta."""
    z1, z2 = maps(z, PTS, A0, B0, MU)
    x = (mp.re(z1), mp.re(z2))
    D = (-mp.im(z2) / mp.pi, mp.im(z1) / mp.pi)
    return x, D


def dAdz(z, h=mp.mpf('1e-8')):
    """d/dz of the amoeba map (complex derivative of each component)."""
    xp, _ = AD(z + h)
    xm, _ = AD(z - h)
    return ((xp[0] - xm[0]) / (2 * h), (xp[1] - xm[1]) / (2 * h))


def drho(z):
    """The 1-form d rho expressed in z: Delta . dA/dz."""
    _, D = AD(z)
    d1, d2 = dAdz(z)
    return D[0] * d1 + D[1] * d2


def integrate(path, N=200):
    """Integrate d rho along a polyline of complex points."""
    tot = mp.mpf(0)
    for a, b in zip(path[:-1], path[1:]):
        for n in range(N):
            t0 = mp.mpf(n) / N
            t1 = mp.mpf(n + 1) / N
            za = a + (b - a) * t0
            zb = a + (b - a) * t1
            zm = (za + zb) / 2
            tot += mp.re(drho(zm) * (zb - za))
    return tot


if __name__ == "__main__":
    print("PATH INDEPENDENCE: d rho is exact, so a closed loop must give 0")
    loops = [
        [mp.mpc('0.5', '2.0'), mp.mpc('1.5', '2.0'),
         mp.mpc('1.5', '3.0'), mp.mpc('0.5', '3.0'), mp.mpc('0.5', '2.0')],
        [mp.mpc('-1.0', '1.5'), mp.mpc('0.0', '2.5'),
         mp.mpc('1.0', '1.5'), mp.mpc('-1.0', '1.5')],
    ]
    for k, lp in enumerate(loops):
        v = integrate(lp)
        print(f"   loop {k+1}: closed integral = {mp.nstr(v, 6)}")

    print("\nCONVEXITY: Hess rho = d Delta / d x must be positive definite")
    print("   z                x = A(z)                    Hess (via finite diff)")
    for z in (mp.mpc('0.3', '2.2'), mp.mpc('-0.8', '1.8'), mp.mpc('1.2', '2.6')):
        h = mp.mpf('1e-5')
        x0, D0 = AD(z)
        xr, Dr = AD(z + h)
        xi, Di = AD(z + 1j * h)
        # dDelta/dx via the 2x2 Jacobians in the (Re z, Im z) chart
        J_x = mp.matrix([[(xr[0]-x0[0])/h, (xi[0]-x0[0])/h],
                         [(xr[1]-x0[1])/h, (xi[1]-x0[1])/h]])
        J_D = mp.matrix([[(Dr[0]-D0[0])/h, (Di[0]-D0[0])/h],
                         [(Dr[1]-D0[1])/h, (Di[1]-D0[1])/h]])
        H = J_D * J_x ** -1
        tr = H[0, 0] + H[1, 1]
        det = H[0, 0]*H[1, 1] - H[0, 1]*H[1, 0]
        print(f"   {mp.nstr(z,6):<18} ({mp.nstr(x0[0],7)}, {mp.nstr(x0[1],7)})  "
              f"tr={mp.nstr(tr,6)} det={mp.nstr(det,6)} "
              f"sym={mp.nstr(abs(H[0,1]-H[1,0]),3)}")
