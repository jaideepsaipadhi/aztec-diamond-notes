"""
Arctic curves from the computed limit shape.

Provenance note: sigma comes from the SCHOTTKY data (amoeba map -> polygon map
-> Ronkin -> Legendre dual), not from the Fock weights.  amoeba.py imports only
mpmath.  So the limit shape was never affected by the weight bug -- that bug
touched only the sampler side.

Two boundaries to find:
  * the OUTER arctic curve, separating frozen corners from the liquid region,
    where grad h reaches a VERTEX of the polygon;
  * the INNER boundary, separating liquid from the gas facet, where grad h
    reaches the facet slope Delta(compact oval).

The solver is the strictly convex smoothing (smooth_lp), so the minimiser is
unique -- necessary here, because a sliding minimiser would move the boundaries.
"""
import numpy as np
from scipy.optimize import minimize

from smooth_lp import SigmaSmooth, energy_and_grad

GAS = (0.4716546, 0.5170360)
VERTS = [(0, 0), (1, 0), (0, 1), (1, 1)]


def aztec_boundary(N):
    xs = np.linspace(0, 1, N)
    h = np.zeros((N, N))
    h[:, 0] = 0.0
    h[0, :] = 0.0
    h[-1, :] = xs
    h[:, -1] = xs
    return h, xs


def solve(tab, N, beta=600.0):
    sig = SigmaSmooth(tab, beta=beta)
    hb, xs = aztec_boundary(N)
    hg = xs[1] - xs[0]
    mask = np.zeros((N, N), bool)
    mask[1:-1, 1:-1] = True
    x0 = hb[mask].copy()
    r = minimize(energy_and_grad, x0, jac=True, args=(hb, mask, sig, hg),
                 method='L-BFGS-B',
                 options=dict(maxiter=8000, ftol=1e-16, gtol=1e-14))
    h = hb.copy()
    h[mask] = r.x
    g1 = (np.diff(h, axis=0)/hg)[:, :-1]
    g2 = (np.diff(h, axis=1)/hg)[:-1, :]
    return h, g1, g2, r.fun


def distances(g1, g2):
    dfro = np.min([np.hypot(g1-vx, g2-vy) for vx, vy in VERTS], axis=0)
    dgas = np.hypot(g1-GAS[0], g2-GAS[1])
    return dfro, dgas


def trace(field, level, xs):
    """Crossings of `field` = level along each row, by linear interpolation."""
    pts = []
    n = field.shape[0]
    for i in range(n):
        row = field[i]
        for j in range(len(row)-1):
            a, b = row[j], row[j+1]
            if (a-level)*(b-level) < 0:
                w = (level-a)/(b-a)
                pts.append((xs[i], xs[j] + w*(xs[j+1]-xs[j])))
    return np.array(pts) if pts else np.empty((0, 2))


if __name__ == "__main__":
    import time
    tab = np.load('/home/claude/sigma_table_slope2.npy')
    for N in (25, 33):
        t0 = time.time()
        h, g1, g2, E = solve(tab, N)
        dfro, dgas = distances(g1, g2)
        xs = np.linspace(0, 1, g1.shape[0])
        outer = trace(dfro, 0.12, xs)
        inner = trace(dgas, 0.08, xs)
        fro = (dfro < 0.12).mean()
        gas = ((dgas < 0.08) & (dfro >= 0.12)).mean()
        print(f"N={N} ({time.time()-t0:.0f}s)  energy {E:.6f}   "
              f"frozen {100*fro:.1f}%  gas {100*gas:.1f}%  "
              f"liquid {100*(1-fro-gas):.1f}%")
        print(f"   outer arctic curve: {len(outer)} crossing points")
        print(f"   inner (gas) boundary: {len(inner)} crossing points")
        if N == 33 and len(outer):
            mid = np.abs(outer[:, 0]-0.5) < 0.03
            print(f"   outer curve at x=0.5 crosses y = "
                  f"{np.round(np.sort(outer[mid][:,1]),4)}")
            midg = np.abs(inner[:, 0]-0.5) < 0.03
            print(f"   gas boundary at x=0.5 crosses y = "
                  f"{np.round(np.sort(inner[midg][:,1]),4)}")
