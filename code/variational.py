"""
Variational minimiser for the quasi-periodic limit shape.

    h = argmin  integral sigma(grad h)   over Lipschitz h with given boundary data

This is the first piece of the project with NO independent identity to check
against -- there is no closed-form limit shape for quasi-periodic weights, which
is why BBS wrote the paper.  So the solver is built against tests whose answers
we know for other reasons:

  TEST 1 (linear boundary data).  sigma is convex, so if h is prescribed to be
  an affine function on the boundary, the minimiser is that same affine function
  everywhere.  Any deviation is solver error, measurable exactly.

  TEST 2 (convergence under refinement).  The energy and the solution must
  settle as the grid is refined.

  TEST 3 (the real one, later).  Compare against a large sample from the
  validated shuffler using the same Schottky weights.

sigma comes from tabulate.py: sweeping z gives scattered (slope, value) pairs
plus the gradient for free.  We interpolate linearly on that scatter.
"""
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import minimize


def build_table(n_ang=24, n_rad=14):
    """Scattered (s1, s2, sigma) over the polygon, via the polar sweep."""
    import mpmath as mp
    from ronkin import AD
    from tabulate import integrate_seg

    A = mp.mpc('0.1', '1.0'); B = mp.conj(A); mu = mp.mpf('0.02')
    a = A - mu*B; b = mu*B*A - A*B; c = 1 - mu; d = mu*A - B
    s = mp.sqrt(a*d - b*c); a, b, c, d = a/s, b/s, c/s, d/s
    cen = mp.conj(-d/c); rad = 1/abs(c)

    base = cen + rad*mp.mpf('1.03')
    pts = []
    for k in range(n_ang):
        th = 2*mp.pi*k/n_ang
        prev = None; acc = mp.mpf(0)
        for j in range(n_rad):
            r = rad*(mp.mpf('1.03') + mp.mpf('0.42')*j)
            z = cen + r*mp.e**(1j*th)
            if mp.im(z) < mp.mpf('0.04'):
                break
            acc = (integrate_seg(base, z, N=25) if prev is None
                   else acc + integrate_seg(prev, z, N=8))
            x, D = AD(z)
            # store the GRADIENT too: grad sigma = x, free from the sweep
            pts.append((float(D[0]), float(D[1]),
                        float(D[0]*x[0] + D[1]*x[1] - acc),
                        float(x[0]), float(x[1])))
            prev = z
    return np.array(pts)


class Sigma:
    """sigma and its gradient. grad sigma = x is tabulated, not differenced."""

    def __init__(self, table):
        self.pts = table[:, :2]
        self.interp = LinearNDInterpolator(self.pts, table[:, 2])
        self.gx = LinearNDInterpolator(self.pts, table[:, 3])
        self.gy = LinearNDInterpolator(self.pts, table[:, 4])
        self.fill = float(table[:, 2].max())

    def __call__(self, s1, s2):
        v = self.interp(s1, s2)
        return np.where(np.isnan(v), self.fill, v)

    def grad(self, s1, s2):
        a = self.gx(s1, s2)
        b = self.gy(s1, s2)
        return np.nan_to_num(a), np.nan_to_num(b)


def energy_and_grad(hflat, shape, hbnd, mask, sig, hgrid):
    h = hbnd.copy()
    h[mask] = hflat
    g1 = (np.diff(h, axis=0) / hgrid)[:, :-1]
    g2 = (np.diff(h, axis=1) / hgrid)[:-1, :]
    E = float(np.sum(sig(g1, g2))) * hgrid * hgrid
    # dE/dh: discrete divergence of grad sigma evaluated at grad h
    p, q = sig.grad(g1, g2)
    dE = np.zeros_like(h)
    # g1[i,j] = (h[i+1,j]-h[i,j])/hgrid
    dE[1:, :-1] += p * hgrid
    dE[:-1, :-1] -= p * hgrid
    # g2[i,j] = (h[i,j+1]-h[i,j])/hgrid
    dE[:-1, 1:] += q * hgrid
    dE[:-1, :-1] -= q * hgrid
    return E, dE[mask]


if __name__ == "__main__":
    import time
    t0 = time.time()
    tab = build_table()
    print(f"sigma table: {len(tab)} points in {time.time()-t0:.0f}s")
    print(f"  slope coverage s1 [{tab[:,0].min():.3f},{tab[:,0].max():.3f}]  "
          f"s2 [{tab[:,1].min():.3f},{tab[:,1].max():.3f}]")
    sig = Sigma(tab)

    # interpolation check against the table itself
    err = max(abs(float(sig(p[0], p[1])) - p[2]) for p in tab[::7])
    print(f"  interpolation reproduces table nodes to {err:.1e}\n")

    print("TEST 1: affine boundary data -> the minimiser must be that affine map")
    s_star = (0.40, 0.45)          # a slope inside the polygon
    for N in (9, 13):
        xs = np.linspace(0, 1, N)
        X, Y = np.meshgrid(xs, xs, indexing='ij')
        hex_ = s_star[0]*X + s_star[1]*Y
        mask = np.zeros_like(hex_, dtype=bool)
        mask[1:-1, 1:-1] = True
        hb = hex_.copy()
        x0 = hex_[mask] + 0.02*np.random.default_rng(0).normal(size=mask.sum())
        hgrid = xs[1] - xs[0]
        res = minimize(energy_and_grad, x0, jac=True,
                       args=((N, N), hb, mask, sig, hgrid),
                       method='L-BFGS-B',
                       options=dict(maxiter=2000, ftol=1e-15, gtol=1e-13))
        h = hb.copy(); h[mask] = res.x
        dev = np.abs(h - hex_).max()
        print(f"   N={N:<3} max |h - affine| = {dev:.2e}   "
              f"energy {res.fun:.6f}  (exact {float(sig(*s_star)):.6f})")
