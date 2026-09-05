"""
Sample sigma UNIFORMLY IN SLOPE SPACE.

The polar sweep in z maps very non-uniformly onto the polygon, so the tangent
planes cluster and the LP underestimator converges only like O(h).  Here we pick
target slopes on a uniform grid and solve Delta(z) = s by Newton, so the tangent
planes are placed where the LP needs them.

Each accepted point still costs one rho path integral, so the per-point cost is
comparable; the gain is in WHERE the points land.

Test: build tables of matched size by both methods and compare the LP
representation error (spread about the base-point constant) on the same probes.
"""
import mpmath as mp
import numpy as np

from ronkin import AD
from tabulate import integrate_seg

mp.mp.dps = 15

A0 = mp.mpc('0.1', '1.0')
MU = mp.mpf('0.02')


def iso_circle():
    A = A0; B = mp.conj(A); mu = MU
    a = A - mu*B; b = mu*B*A - A*B; c = 1 - mu; d = mu*A - B
    s = mp.sqrt(a*d - b*c)
    a, b, c, d = a/s, b/s, c/s, d/s
    return mp.conj(-d/c), 1/abs(c)


CEN, RAD = iso_circle()
BASE = CEN + RAD*mp.mpf('1.03')


def slope(z):
    _, D = AD(z)
    return np.array([float(D[0]), float(D[1])])


def invert(target, z_start, tol=1e-9, itmax=30):
    """Newton solve Delta(z) = target, z in the upper half plane outside the circle."""
    z = z_start
    h = mp.mpf('1e-6')
    for _ in range(itmax):
        f = slope(z) - target
        if np.hypot(*f) < tol:
            return z
        du = (slope(z + h) - slope(z - h)) / (2*float(h))
        dv = (slope(z + 1j*h) - slope(z - 1j*h)) / (2*float(h))
        J = np.column_stack([du, dv])
        try:
            step = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            return None
        # damped, and keep z legal
        for lam in (1.0, 0.5, 0.25, 0.1):
            zn = z + mp.mpc(lam*step[0], lam*step[1])
            if mp.im(zn) > mp.mpf('0.03') and abs(zn - CEN) > RAD*mp.mpf('1.01'):
                z = zn
                break
        else:
            return None
    return None


def polar_path(z, N=22):
    """Integrate rho from BASE to z along an ANGULAR ARC then RADIALLY.

    A straight segment from BASE to z can cross the isometric disk, where the
    form is not defined; about half the points of a slope-space table are
    reached that way and come out corrupted.  The tell is that polar paths give
    a CONSISTENT offset from the direct sigma at every point (0.391 at all
    probes, crossing ones included) while straight paths gave 0.672 at one.
    Any path integral here must respect the fundamental domain.
    """
    rb = abs(BASE - CEN)
    thb = mp.arg(BASE - CEN)
    thz = mp.arg(z - CEN)
    acc = mp.mpf(0)
    prev = BASE
    for n in range(1, N + 1):
        cur = CEN + rb * mp.e ** (1j * (thb + (thz - thb) * mp.mpf(n) / N))
        acc += integrate_seg(prev, cur, N=4)
        prev = cur
    return acc + integrate_seg(prev, z, N=18)


def build_slope_table(n=20, seeds=None):
    """Uniform grid of target slopes; returns rows (s1,s2,sigma,x1,x2)."""
    if seeds is None:
        seeds = []
        for k in range(16):
            th = 2*mp.pi*k/16
            for j in range(8):
                z = CEN + RAD*(mp.mpf('1.05') + mp.mpf('0.6')*j)*mp.e**(1j*th)
                if mp.im(z) > mp.mpf('0.05'):
                    seeds.append((slope(z), z))
    out = []
    grid = np.linspace(0.06, 0.94, n)
    for s1 in grid:
        for s2 in grid:
            t = np.array([s1, s2])
            z0 = min(seeds, key=lambda p: np.hypot(*(p[0]-t)))[1]
            z = invert(t, z0)
            if z is None:
                continue
            x, D = AD(z)
            acc = polar_path(z)
            out.append((float(D[0]), float(D[1]),
                        float(D[0]*x[0] + D[1]*x[1] - acc),
                        float(x[0]), float(x[1])))
    return np.array(out)


if __name__ == "__main__":
    import time
    from lp_solve import sigma_hat
    from tension import sigma as sigma_direct

    t0 = time.time()
    tabS = build_slope_table(n=20)
    print(f"slope-space table: {len(tabS)} points in {time.time()-t0:.0f}s")
    print(f"   coverage s1 [{tabS[:,0].min():.3f},{tabS[:,0].max():.3f}] "
          f"s2 [{tabS[:,1].min():.3f},{tabS[:,1].max():.3f}]")
    np.save('/home/claude/sigma_table_slope.npy', tabS)

    tabP = np.load('/home/claude/sigma_table.npy')
    step = max(1, len(tabP)//len(tabS))
    tabPm = tabP[::step][:len(tabS)]
    print(f"   matched polar subset: {len(tabPm)} points\n")

    def probe(T, n=16):
        rr = np.random.default_rng(4)
        out = []
        for _ in range(n):
            th = mp.mpf(str(rr.uniform(0, 6.28)))
            r = RAD*mp.mpf(str(rr.uniform(1.1, 4.0)))
            z = CEN + r*mp.e**(1j*th)
            if mp.im(z) < 0.05:
                continue
            _, D = AD(z)
            out.append(sigma_hat(T, (float(D[0]), float(D[1])))
                       - float(sigma_direct(z)))
        return np.array(out)

    print("LP representation error (spread about the base-point constant):")
    for T, lab in ((tabPm, f"polar  {len(tabPm)}"), (tabS, f"slope  {len(tabS)}")):
        d = probe(T); off = np.median(d)
        print(f"   {lab}: median |dev| = {np.median(abs(d-off)):.3e}   "
              f"max = {np.abs(d-off).max():.3e}")
