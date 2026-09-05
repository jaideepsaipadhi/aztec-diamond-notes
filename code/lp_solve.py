"""
The limit-shape variational problem as a LINEAR PROGRAM.

sigma's non-smoothness is the physics, not a nuisance: the conical singularities
are exactly where the gas facets are.  So rather than smoothing sigma to suit a
quasi-Newton method, use a method that handles polyhedral convex objectives
natively.

sigma is convex and we have BOTH sigma and grad sigma at every tabulated slope
(the sweep gives the conjugate pair for free).  For a convex function each
tangent plane is a global underestimator, so

    sigma_hat(s) = max_k [ sigma_k + x_k . (s - s_k) ]

is a convex piecewise-linear underapproximation that converges to sigma as the
table refines.  Minimising the discrete energy over that is a linear program:

    minimise    sum_cells t_c * hgrid^2
    subject to  t_c >= sigma_k + x_k . (grad h|_c - s_k)   for every k, c

linear in the nodal heights h and the epigraph variables t.  Kinks are handled
exactly -- they are just active constraints.

Test 1 (affine boundary data) has a known answer: the minimiser is that affine
map and the energy is sigma_hat(s*).
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog


def solve(sig_table, N, s_star, hbnd=None, verbose=True):
    """LP solve on an N x N grid with affine boundary data of slope s_star
    (or supplied boundary array hbnd)."""
    S = sig_table[:, :2]
    V = sig_table[:, 2]
    Xg = sig_table[:, 3:5]
    K = len(S)

    xs = np.linspace(0, 1, N)
    hg = xs[1] - xs[0]
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    if hbnd is None:
        hbnd = s_star[0]*X + s_star[1]*Y

    mask = np.zeros((N, N), bool)
    mask[1:-1, 1:-1] = True
    idx = -np.ones((N, N), int)
    idx[mask] = np.arange(mask.sum())
    nh = int(mask.sum())
    cells = [(i, j) for i in range(N-1) for j in range(N-1)]
    nc = len(cells)
    nvar = nh + nc

    rows, cols, vals, rhs = [], [], [], []
    r = 0
    # rhs constant from the boundary values folded in
    const = V - (Xg * S).sum(axis=1)          # sigma_k - x_k . s_k
    for ci, (i, j) in enumerate(cells):
        # grad h|_c = ((h[i+1,j]-h[i,j])/hg, (h[i,j+1]-h[i,j])/hg)
        terms = [((i+1, j), 0, 1.0/hg), ((i, j), 0, -1.0/hg),
                 ((i, j+1), 1, 1.0/hg), ((i, j), 1, -1.0/hg)]
        for k in range(K):
            b = -const[k]
            for (node, comp, coef) in terms:
                w = Xg[k, comp] * coef
                if mask[node]:
                    rows.append(r); cols.append(idx[node]); vals.append(w)
                else:
                    b -= w * hbnd[node]
            rows.append(r); cols.append(nh + ci); vals.append(-1.0)
            rhs.append(b)
            r += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, nvar))
    c = np.zeros(nvar); c[nh:] = hg*hg
    res = linprog(c, A_ub=A, b_ub=np.array(rhs), method='highs')
    if verbose:
        print(f"   LP: {nvar} vars, {r} constraints, status={res.status} "
              f"({res.message.split('.')[0]})")
    h = hbnd.copy()
    if res.x is not None:
        h[mask] = res.x[:nh]
    return h, (res.fun if res.fun is not None else np.nan), hbnd


def sigma_hat(table, s):
    S = table[:, :2]; V = table[:, 2]; Xg = table[:, 3:5]
    return float(np.max(V + (Xg * (np.asarray(s) - S)).sum(axis=1)))


if __name__ == "__main__":
    import time
    from variational import build_table

    t0 = time.time()
    tab = build_table(n_ang=20, n_rad=12)
    print(f"table {len(tab)} points in {time.time()-t0:.0f}s\n")

    s_star = (0.40, 0.45)
    exact = sigma_hat(tab, s_star)
    print(f"TEST 1: affine boundary data, slope {s_star}")
    print(f"   exact energy = sigma_hat(s*) = {exact:.6f}\n")
    for N in (7, 9, 11):
        t0 = time.time()
        h, E, hb = solve(tab, N, s_star)
        dev = np.abs(h - hb).max()
        print(f"   N={N:<3} dev={dev:.2e}  E={E:.6f}  "
              f"(exact {exact:.6f})  {time.time()-t0:.0f}s")
