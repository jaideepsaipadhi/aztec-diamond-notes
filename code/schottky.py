"""
The period B from SCHOTTKY UNIFORMIZATION -- independent of elliptic integrals.

Genus-1 Schottky group: a single generator, normalized to z -> mu z, with
fundamental domain the annulus mu <= |z| <= 1.  The normalized differential is
omega = dz/(2 pi i z), so

    B = log(mu) / (2 pi i) = i log(1/mu) / (2 pi),      b = -log(mu)/(2 pi).

log(1/mu) is the modulus of the annulus, and the modulus of a doubly connected
domain is a pure potential-theory quantity: if u is harmonic with u = 0 on one
boundary component and 1 on the other, then

    Dirichlet energy  E = 2 pi / log(R/r) = 2 pi / log(1/mu),

hence simply  b = 1 / E.  No K, no K', no AGM, no theta -- just Laplace.

The domain: our curve y^2 = z(z+a^2)(z+a^-2) is the Schottky double of the
z-sphere slit along the two real ovals, i.e. along [-a^-2,-a^2] and [0,infinity].
The Moebius map w = 1/(z-d) with d = -0.25 (outside both slits) makes both slits
bounded, giving the plane minus two real segments.

Discretization: finite differences on a square box, Dirichlet on the two slits,
Neumann on the far box boundary (u tends to a constant at infinity, zero flux).
Slit endpoints carry square-root singularities, so convergence in h is slow; we
Richardson-extrapolate in h.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


def slits(a, d=-0.25):
    T = lambda z: 1.0 / (z - d)
    A = sorted([T(-1/a**2), T(-a**2)])
    B = sorted([T(0.0), 0.0])          # T(inf) = 0
    return A, B


def energy(a, L=40.0, N=600, d=-0.25):
    """Dirichlet energy of the harmonic function (0 on slit A, 1 on slit B)."""
    A, B = slits(a, d)
    xs = np.linspace(-L, L, N)
    ys = np.linspace(-L, L, N)
    h = xs[1] - xs[0]
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    j0 = np.argmin(np.abs(ys))          # the row nearest y = 0
    onA = np.zeros((N, N), bool)
    onB = np.zeros((N, N), bool)
    onA[:, j0] = (xs >= A[0]) & (xs <= A[1])
    onB[:, j0] = (xs >= B[0]) & (xs <= B[1])
    fixed = onA | onB

    idx = -np.ones((N, N), int)
    free = ~fixed
    idx[free] = np.arange(free.sum())
    nf = free.sum()

    rows, cols, vals = [], [], []
    rhs = np.zeros(nf)
    for i in range(N):
        for j in range(N):
            if not free[i, j]:
                continue
            k = idx[i, j]
            diag = 0.0
            for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i+di, j+dj
                if not (0 <= ii < N and 0 <= jj < N):
                    continue            # Neumann: skip, reduces the diagonal
                diag += 1.0
                if free[ii, jj]:
                    rows.append(k); cols.append(idx[ii, jj]); vals.append(-1.0)
                else:
                    rhs[k] += 1.0 if onB[ii, jj] else 0.0
            rows.append(k); cols.append(k); vals.append(diag)
    Lap = sp.csr_matrix((vals, (rows, cols)), shape=(nf, nf))
    sol = spl.spsolve(Lap.tocsc(), rhs)

    u = np.zeros((N, N))
    u[free] = sol
    u[onB] = 1.0
    # Dirichlet energy, sum of squared differences (h factors cancel in 2D)
    E = ((np.diff(u, axis=0))**2).sum() + ((np.diff(u, axis=1))**2).sum()
    return E


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/mnt/user-data/outputs")
    from period_g1 import b_legendre
    import mpmath as mp
    mp.mp.dps = 15

    a = 0.7
    print(f"a = {a}   elliptic-integral value of b: "
          f"{float(b_legendre(str(a))):.6f}   (paper's would be 0.521828)\n")
    print("  N      E          b = 1/E")
    res = []
    for N in (300, 450, 600):
        E = energy(a, L=40.0, N=N)
        res.append((N, E, 1/E))
        print(f"  {N:<6} {E:9.5f}  {1/E:.6f}")
    Ns = np.array([r[0] for r in res], float)
    bs = np.array([r[2] for r in res])
    c = np.polyfit(1/Ns, bs, 1)
    print(f"\n  Richardson (linear in 1/N): b -> {c[1]:.6f}")
