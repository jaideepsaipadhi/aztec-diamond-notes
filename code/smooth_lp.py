"""
Pinning the minimiser: a strictly convex smoothing of the tangent-plane sigma.

The LP formulation gives the exact ENERGY but a non-unique minimiser, because
sigma_hat = max_k L_k is piecewise linear, hence convex but not strictly convex:
on each flat piece the minimiser can slide.  Measured deviation from the known
affine answer was ~5e-2 even with the energy correct to six digits.

Fix: replace the max by a log-sum-exp, which is smooth and strictly convex,

    sigma_beta(s) = (1/beta) log sum_k exp( beta L_k(s) ),
    L_k(s) = sigma_k + x_k . (s - s_k)

with sigma_beta -> sigma_hat from above as beta -> infinity (the gap is at most
log(K)/beta).  The gradient is the softmax-weighted average of the x_k, so both
value and gradient come from one object -- no repeat of the inconsistency that
broke the earlier attempt.

Being smooth, this is also usable with L-BFGS, which the polyhedral objective
was not.  Larger beta is sharper but stiffer; the test below sweeps it.
"""
import numpy as np
from scipy.optimize import minimize


class SigmaSmooth:
    def __init__(self, table, beta=200.0):
        self.s = table[:, :2]
        self.v = table[:, 2]
        self.x = table[:, 3:5]
        self.beta = beta
        # L_k(s) = c_k + x_k . s  with c_k = sigma_k - x_k . s_k
        self.c = self.v - (self.x * self.s).sum(axis=1)

    def value_and_grad(self, g1, g2):
        shape = np.shape(g1)
        S = np.column_stack([np.ravel(g1), np.ravel(g2)])
        L = self.c[None, :] + S @ self.x.T            # (N, K)
        m = L.max(axis=1, keepdims=True)
        e = np.exp(self.beta * (L - m))
        Z = e.sum(axis=1, keepdims=True)
        val = (m[:, 0] + np.log(Z[:, 0]) / self.beta)
        wgt = e / Z
        gx = wgt @ self.x[:, 0]
        gy = wgt @ self.x[:, 1]
        return val.reshape(shape), gx.reshape(shape), gy.reshape(shape)


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
    tab = np.load('/home/claude/sigma_table_slope2.npy')
    s_star = (0.40, 0.45)
    rng = np.random.default_rng(0)

    print("gradient consistency (analytic vs finite difference)")
    sig = SigmaSmooth(tab, beta=200.0)
    N = 11
    xs = np.linspace(0, 1, N)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    hex_ = s_star[0]*X + s_star[1]*Y
    mask = np.zeros_like(hex_, bool); mask[1:-1, 1:-1] = True
    hb = hex_.copy(); hg = xs[1]-xs[0]
    x0 = hex_[mask] + 0.02*rng.normal(size=mask.sum())
    E0, g0 = energy_and_grad(x0, hb, mask, sig, hg)
    errs = []
    for k in range(0, len(x0), 4):
        d = np.zeros_like(x0); d[k] = 1e-6
        Ep, _ = energy_and_grad(x0+d, hb, mask, sig, hg)
        Em, _ = energy_and_grad(x0-d, hb, mask, sig, hg)
        errs.append(abs((Ep-Em)/2e-6 - g0[k]))
    print(f"   max |analytic - FD| = {max(errs):.2e}  "
          f"typical |g| = {np.abs(g0).mean():.2e}\n")

    print("affine test from GENERIC starts (the LP left this at ~5e-2)")
    print("   beta    noise   deviation    energy      exact sigma_hat")
    exact = float(np.max(sig.c + np.array(s_star) @ sig.x.T))
    for beta in (50.0, 200.0, 800.0):
        sig = SigmaSmooth(tab, beta=beta)
        for noise in (0.02, 0.05):
            x0 = hex_[mask] + noise*rng.normal(size=mask.sum())
            r = minimize(energy_and_grad, x0, jac=True,
                         args=(hb, mask, sig, hg), method='L-BFGS-B',
                         options=dict(maxiter=4000, ftol=1e-16, gtol=1e-14))
            h = hb.copy(); h[mask] = r.x
            print(f"   {beta:6.0f}  {noise:5.2f}   {np.abs(h-hex_).max():.2e}   "
                  f"{r.fun:.6f}    {exact:.6f}")
