"""
A CONSISTENT surface-tension representation.

The previous version interpolated sigma linearly and grad sigma linearly, as two
separate objects.  Those disagree by construction -- the gradient of a linear
interpolant is not the linear interpolant of the gradient -- and the measured
discrepancy was 67% of the gradient magnitude, which trapped the optimiser.

Here sigma is piecewise linear on a Delaunay triangulation of the tabulated
slopes, and the gradient is that triangle's own constant, computed from the same
three vertex values.  So sigma and grad sigma come from ONE object and agree to
machine precision.

Independent accuracy check: the triangle gradients can be compared against the
tabulated grad sigma = x, which the sweep provides for free and which was never
used to build the interpolant.  That measures the representation error rather
than merely its self-consistency.
"""
import numpy as np
from scipy.spatial import Delaunay


class SigmaPL:
    def __init__(self, table):
        self.pts = table[:, :2]
        self.val = table[:, 2]
        self.xg = table[:, 3:5]           # tabulated grad sigma, held out
        self.tri = Delaunay(self.pts)
        # per-triangle affine coefficients: sigma = c0 + c1 s1 + c2 s2
        simp = self.tri.simplices
        P = self.pts[simp]                # (T,3,2)
        V = self.val[simp]                # (T,3)
        ones = np.ones((len(simp), 3, 1))
        M = np.concatenate([ones, P], axis=2)      # (T,3,3)
        self.coef = np.linalg.solve(M, V[:, :, None])[:, :, 0]   # (T,3)
        self.fill = float(self.val.max())

    def locate(self, s1, s2):
        pts = np.column_stack([np.ravel(s1), np.ravel(s2)])
        return self.tri.find_simplex(pts), pts

    def value_and_grad(self, s1, s2):
        shape = np.shape(s1)
        idx, pts = self.locate(s1, s2)
        inside = idx >= 0
        c = np.zeros((len(pts), 3))
        c[inside] = self.coef[idx[inside]]
        val = np.where(inside, c[:, 0] + c[:, 1]*pts[:, 0] + c[:, 2]*pts[:, 1],
                       self.fill)
        g1 = np.where(inside, c[:, 1], 0.0)
        g2 = np.where(inside, c[:, 2], 0.0)
        return val.reshape(shape), g1.reshape(shape), g2.reshape(shape)

    def representation_error(self):
        """Compare triangle gradients with the tabulated grad sigma = x."""
        centres = self.pts[self.tri.simplices].mean(axis=1)
        _, g1, g2 = self.value_and_grad(centres[:, 0], centres[:, 1])
        xt = self.xg[self.tri.simplices].mean(axis=1)
        err = np.hypot(g1 - xt[:, 0], g2 - xt[:, 1])
        mag = np.hypot(xt[:, 0], xt[:, 1])
        return float(np.median(err / np.maximum(mag, 1e-12)))


def energy_and_grad(hflat, hbnd, mask, sig, hgrid):
    h = hbnd.copy()
    h[mask] = hflat
    g1 = (np.diff(h, axis=0) / hgrid)[:, :-1]
    g2 = (np.diff(h, axis=1) / hgrid)[:-1, :]
    v, p, q = sig.value_and_grad(g1, g2)
    E = float(v.sum()) * hgrid * hgrid
    dE = np.zeros_like(h)
    dE[1:, :-1] += p * hgrid
    dE[:-1, :-1] -= p * hgrid
    dE[:-1, 1:] += q * hgrid
    dE[:-1, :-1] -= q * hgrid
    return E, dE[mask]


if __name__ == "__main__":
    import time
    from scipy.optimize import minimize
    from variational import build_table

    t0 = time.time()
    tab = build_table(n_ang=28, n_rad=16)
    sig = SigmaPL(tab)
    print(f"table {len(tab)} points, {len(sig.tri.simplices)} triangles, "
          f"{time.time()-t0:.0f}s")
    print(f"representation error vs the held-out tabulated gradient: "
          f"{sig.representation_error()*100:.2f}% (median relative)\n")

    print("CONSISTENCY: analytic gradient vs finite differences")
    rng = np.random.default_rng(0)
    s_star = (0.40, 0.45)
    N = 11
    xs = np.linspace(0, 1, N)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    hex_ = s_star[0]*X + s_star[1]*Y
    mask = np.zeros_like(hex_, bool); mask[1:-1, 1:-1] = True
    hb = hex_.copy(); hg = xs[1]-xs[0]
    x0 = hex_[mask] + 0.01*rng.normal(size=mask.sum())
    E0, g0 = energy_and_grad(x0, hb, mask, sig, hg)
    errs = []
    for k in range(0, len(x0), 4):
        d = np.zeros_like(x0); d[k] = 1e-7
        Ep, _ = energy_and_grad(x0+d, hb, mask, sig, hg)
        Em, _ = energy_and_grad(x0-d, hb, mask, sig, hg)
        errs.append(abs((Ep-Em)/2e-7 - g0[k]))
    print(f"   max |analytic - FD| = {max(errs):.2e}   "
          f"typical |g| = {np.abs(g0).mean():.2e}\n")

    print("TEST 1 from GENERIC starts (this is what failed before)")
    for N in (9, 13, 17):
        xs = np.linspace(0, 1, N)
        X, Y = np.meshgrid(xs, xs, indexing='ij')
        hex_ = s_star[0]*X + s_star[1]*Y
        mask = np.zeros_like(hex_, bool); mask[1:-1, 1:-1] = True
        hb = hex_.copy(); hg = xs[1]-xs[0]
        for noise in (0.02, 0.05):
            x0 = hex_[mask] + noise*rng.normal(size=mask.sum())
            r = minimize(energy_and_grad, x0, jac=True, args=(hb, mask, sig, hg),
                         method='L-BFGS-B',
                         options=dict(maxiter=5000, ftol=1e-16, gtol=1e-14))
            h = hb.copy(); h[mask] = r.x
            v, _, _ = sig.value_and_grad(np.array(s_star[0]), np.array(s_star[1]))
            print(f"   N={N:<3} noise={noise}: dev={np.abs(h-hex_).max():.2e}  "
                  f"E={r.fun:.6f}  exact={float(v):.6f}  nit={r.nit}")
