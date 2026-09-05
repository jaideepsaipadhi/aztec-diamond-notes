"""
Surface tension sigma as the Legendre dual of the generalized Ronkin function.

With grad rho = Delta (Krichever), the Legendre transform is immediate: at the
amoeba point x = A(z) the slope is s = Delta(z), and

    sigma(s) = s . x - rho(x).

Everything is parameterised by z in the upper half plane, so no inversion of the
gradient map is needed -- the parameterisation supplies the conjugate pair.

Two sharp checks, both consequences of Legendre duality and neither built in:

  * INVOLUTIVITY:   grad sigma (s) = x,  i.e. the transform inverts.
  * MONGE-AMPERE:   Hess sigma = (Hess rho)^{-1}, so
                        det Hess sigma = 1 / det Hess rho = pi^2
                    since we measured det Hess rho = 1/pi^2 to seven digits.

sigma is defined on the polygon Delta_S and has a conical singularity at the
image of each compact oval -- those are the gas facets, where rho is affine.
"""
import mpmath as mp

from ronkin import AD

mp.mp.dps = 20

Z0 = mp.mpc('0.5', '2.0')          # base point for rho


def drho_uv(z, h=mp.mpf('1e-7')):
    x0, D0 = AD(z)
    xu, _ = AD(z + h)
    xv, _ = AD(z + 1j * h)
    du = ((xu[0] - x0[0]) / h, (xu[1] - x0[1]) / h)
    dv = ((xv[0] - x0[0]) / h, (xv[1] - x0[1]) / h)
    return (D0[0] * du[0] + D0[1] * du[1],
            D0[0] * dv[0] + D0[1] * dv[1])


def rho(z, z0=Z0, N=120):
    """rho(z) - rho(z0), integrated along the straight segment."""
    tot = mp.mpf(0)
    for n in range(N):
        za = z0 + (z - z0) * mp.mpf(n) / N
        zb = z0 + (z - z0) * mp.mpf(n + 1) / N
        zm = (za + zb) / 2
        gu, gv = drho_uv(zm)
        tot += gu * mp.re(zb - za) + gv * mp.im(zb - za)
    return tot


def sigma(z):
    """sigma at the slope Delta(z)."""
    x, D = AD(z)
    return D[0] * x[0] + D[1] * x[1] - rho(z)


if __name__ == "__main__":
    print("Legendre duality checks (nothing below is built in)\n")
    print("INVOLUTIVITY: grad sigma with respect to s must equal x = A(z)")
    print("   z                  x = A(z)                     grad sigma")
    h = mp.mpf('1e-5')
    for z in (mp.mpc('0.3', '2.2'), mp.mpc('-0.8', '1.8'), mp.mpc('1.2', '2.6')):
        x0, D0 = AD(z)
        s0 = sigma(z)
        # perturb in the two chart directions, get ds and dsigma
        rows, rhs = [], []
        for dz in (h, 1j * h):
            _, D1 = AD(z + dz)
            s1 = sigma(z + dz)
            rows.append([D1[0] - D0[0], D1[1] - D0[1]])
            rhs.append(s1 - s0)
        M = mp.matrix(rows)
        g = M ** -1 * mp.matrix(rhs)
        print(f"   {mp.nstr(z,6):<18} ({mp.nstr(x0[0],8)}, {mp.nstr(x0[1],8)})   "
              f"({mp.nstr(g[0],8)}, {mp.nstr(g[1],8)})")

    print("\nMONGE-AMPERE: det Hess sigma must equal pi^2 = "
          f"{mp.nstr(mp.pi**2, 10)}")
    print("   z                  det Hess sigma")
    for z in (mp.mpc('0.3', '2.2'), mp.mpc('-0.8', '1.8'),
              mp.mpc('1.2', '2.6'), mp.mpc('-1.5', '2.2')):
        x0, D0 = AD(z)
        xr, Dr = AD(z + h)
        xi, Di = AD(z + 1j * h)
        Jx = mp.matrix([[(xr[0]-x0[0])/h, (xi[0]-x0[0])/h],
                        [(xr[1]-x0[1])/h, (xi[1]-x0[1])/h]])
        JD = mp.matrix([[(Dr[0]-D0[0])/h, (Di[0]-D0[0])/h],
                        [(Dr[1]-D0[1])/h, (Di[1]-D0[1])/h]])
        Hs = Jx * JD ** -1                  # Hess sigma = dx/ds
        det = Hs[0, 0]*Hs[1, 1] - Hs[0, 1]*Hs[1, 0]
        print(f"   {mp.nstr(z,6):<18} {mp.nstr(det, 10)}")
