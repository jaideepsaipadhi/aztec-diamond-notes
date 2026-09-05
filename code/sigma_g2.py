"""
Genus-2 surface tension: sigma over a polygon with TWO interior points.

Everything transfers from genus 1: grad rho = Delta, so rho comes from
integrating d rho = Delta . dA, and sigma = Delta . A - rho, with the
z-parameterisation supplying the conjugate pair for free.

What changes: the fundamental domain is the upper half plane minus TWO isometric
disks, so integration paths must avoid both.  We route every path over the top
(up to height H above both disks, across, then down), which is safe as long as
the descent column misses the disks; points near a circle are reached by a short
radial leg from directly above it.
"""
import mpmath as mp
import numpy as np

from genus2_amoeba import group, zeta, iso_circle

mp.mp.dps = 18

FIXED = [(mp.mpc('1.4', '0.35'), mp.mpc('1.4', '-0.35')),
         (mp.mpc('-0.9', '0.5'), mp.mpc('-0.9', '-0.5'))]
MUS = [mp.mpf('0.02'), mp.mpf('0.045')]
TRACKS = {'a+': mp.mpf('-2.8'), 'a-': mp.mpf('2.4'),
          'b+': mp.mpf('-0.35'), 'b-': mp.mpf('0.6')}
H = mp.mpf('2.5')          # routing height, above both disks
EL = group(FIXED, MUS, maxlen=3)
CIRC = [iso_circle(A, B, mu) for (A, B), mu in zip(FIXED, MUS)]


def AD(z):
    z1 = zeta(z, TRACKS['a-'], TRACKS['a+'], EL)
    z2 = zeta(z, TRACKS['b-'], TRACKS['b+'], EL)
    return ((mp.re(z1), mp.re(z2)), (-mp.im(z2)/mp.pi, mp.im(z1)/mp.pi))


def drho_uv(z, h=mp.mpf('1e-6')):
    (x0, _), (D0a, D0b) = AD(z)[0], AD(z)[1]
    x0 = AD(z)[0]
    D0 = AD(z)[1]
    xu = AD(z + h)[0]
    xv = AD(z + 1j*h)[0]
    du = ((xu[0]-x0[0])/h, (xu[1]-x0[1])/h)
    dv = ((xv[0]-x0[0])/h, (xv[1]-x0[1])/h)
    return (D0[0]*du[0] + D0[1]*du[1], D0[0]*dv[0] + D0[1]*dv[1])


def seg(a, b, N=14):
    tot = mp.mpf(0)
    for n in range(N):
        p = a + (b-a)*mp.mpf(n)/N
        q = a + (b-a)*mp.mpf(n+1)/N
        m = (p+q)/2
        gu, gv = drho_uv(m)
        tot += gu*mp.re(q-p) + gv*mp.im(q-p)
    return tot


BASE = mp.mpc('0', H)


def rho(z):
    """Route over the top: BASE -> (Re z, H) -> z."""
    top = mp.mpc(mp.re(z), H)
    return seg(BASE, top, N=16) + seg(top, z, N=20)


def sample_points(n_ang=14, n_rad=7):
    """Points around each circle (radially, from outside) plus a high band."""
    pts = []
    for (cen, rad) in CIRC:
        for k in range(n_ang):
            th = 2*mp.pi*k/n_ang
            for j in range(n_rad):
                r = rad*(mp.mpf('1.06') + mp.mpf('0.5')*j)
                z = cen + r*mp.e**(1j*th)
                if mp.im(z) > mp.mpf('0.08'):
                    pts.append(z)
    for u in np.linspace(-3.5, 3.5, 12):
        for v in (0.9, 1.4, 2.0):
            pts.append(mp.mpc(str(u), str(v)))
    return pts


if __name__ == "__main__":
    import time
    print("genus-2 sigma table")
    print(f"  circles: {[(mp.nstr(c,6), mp.nstr(r,6)) for c, r in CIRC]}")
    t0 = time.time()
    rows = []
    for z in sample_points():
        try:
            x, D = AD(z)
            s = D[0]*x[0] + D[1]*x[1] - rho(z)
            rows.append((float(D[0]), float(D[1]), float(s),
                         float(x[0]), float(x[1])))
        except Exception:
            continue
    tab = np.array(rows)
    np.save('/home/claude/sigma_g2.npy', tab)
    print(f"  {len(tab)} points in {time.time()-t0:.0f}s")
    print(f"  slope coverage s1 [{tab[:,0].min():.3f},{tab[:,0].max():.3f}]  "
          f"s2 [{tab[:,1].min():.3f},{tab[:,1].max():.3f}]")
    print("  the two facet slopes (from genus2_amoeba): "
          "(-0.0672, 0.8677) and (-0.1321, 0.8708)")
