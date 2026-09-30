"""
Periodic dimer model on the Aztec diamond, general period, from its OWN
Kasteleyn matrix -- no dependence on Berggren-Nicoletti's alpha/beta/gamma
labelling.

Coordinates.  Write a vertex (x1,x2) of the Aztec diamond graph in diagonal
coordinates  (u,v) = ((x1+x2-1)/2, (x2-x1+1)/2).  The four diagonal neighbours
E1, E2, -E1, -E2 become (u+1,v), (u,v+1), (u-1,v), (u,v-1): the Aztec diamond
graph IS the square lattice Z^2, with black/white the two parities of u+v.  So a
periodic Aztec weighting is just a periodic square-lattice dimer model, and the
whole Kenyon-Okounkov-Sheffield apparatus applies verbatim.

Weights.  wu[u % mu][v % mv] on the edge (u,v)-(u+1,v)
         wv[u % mu][v % mv] on the edge (u,v)-(u,v+1)
Kasteleyn phases: 1 on u-edges, i on v-edges (the flat convention that
twoperiodic.py uses; checked below to reproduce the genus-1 answer).

P(z,w) = det K(z,w) on the (mu x mv) fundamental domain, z conjugate to u and w
to v.  With mv = 2 the w-degree is 2 and the curve is hyperelliptic,

        y^2 = D(z) = tr(Phi)^2 - 4 det(Phi)   in the w-quadratic form,

which is what makes the period matrix computable by real-axis quadrature exactly
as at genus 1.

VALIDATION (run as main): with mu = mv = 2 and the Johansson weights this must
reproduce the genus-1 period b = K'/K = 0.772738 at a = 0.7, end to end.
"""
import itertools

import mpmath as mp
import numpy as np

mp.mp.dps = 30


# ------------------------------------------------------------------ the matrix
def fundamental(mu, mv):
    """Black and white representatives of the mu x mv fundamental domain."""
    cells = [(u, v) for u in range(mu) for v in range(mv)]
    B = [c for c in cells if (c[0] + c[1]) % 2 == 0]
    W = [c for c in cells if (c[0] + c[1]) % 2 == 1]
    assert len(B) == len(W), "fundamental domain must be balanced"
    return B, W


def Kzw(z, w, wu, wv, mu, mv):
    """Magnetically altered Kasteleyn matrix, rows = black, cols = white."""
    B, W = fundamental(mu, mv)
    wi = {c: k for k, c in enumerate(W)}
    K = mp.zeros(len(B), len(W))
    for r, (u, v) in enumerate(B):
        for (du, dv) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            uu, vv = u + du, v + dv
            cu, cv = uu % mu, vv % mv
            mx, my = (uu - cu) // mu, (vv - cv) // mv      # winding
            if du:                                          # u-edge
                a = min(u, uu) % mu
                val = wu[a][v % mv]
                ph = mp.mpf(1)
            else:                                           # v-edge
                b = min(v, vv) % mv
                val = wv[u % mu][b]
                ph = mp.mpc(0, 1)
            K[r, wi[(cu, cv)]] += val * ph * z ** mx * w ** my
    return K


def P(z, w, wu, wv, mu, mv):
    return mp.det(Kzw(z, w, wu, wv, mu, mv))


# --------------------------------------------------- Newton polygon / amoeba
def newton(wu, wv, mu, mv, zdeg=6, wdeg=6):
    """Laurent coefficients of P by 2-D FFT on a torus of radius 1."""
    Nz, Nw = 4 * zdeg + 8, 4 * wdeg + 8
    A = np.zeros((Nz, Nw), dtype=complex)
    for i in range(Nz):
        for j in range(Nw):
            z = mp.expjpi(mp.mpf(2 * i) / Nz)
            w = mp.expjpi(mp.mpf(2 * j) / Nw)
            A[i, j] = complex(P(z, w, wu, wv, mu, mv))
    C = np.fft.fft2(A) / (Nz * Nw)
    pts = {}
    for i in range(Nz):
        for j in range(Nw):
            if abs(C[i, j]) > 1e-9:
                a = i if i <= Nz // 2 else i - Nz
                b = j if j <= Nw // 2 else j - Nw
                pts[(-a, -b)] = C[i, j]
    return pts


def hull_and_interior(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    lo_x, hi_x, lo_y, hi_y = min(xs), max(xs), min(ys), max(ys)
    from scipy.spatial import ConvexHull
    arr = np.array(sorted(pts))
    try:
        h = ConvexHull(arr)
        verts = [tuple(arr[i]) for i in h.vertices]
    except Exception:
        verts = sorted(pts)
    interior = []
    for x in range(lo_x, hi_x + 1):
        for y in range(lo_y, hi_y + 1):
            if _strictly_inside((x, y), verts):
                interior.append((x, y))
    return verts, interior


def _strictly_inside(p, verts):
    n = len(verts)
    sgn = 0
    for i in range(n):
        a, b = verts[i], verts[(i + 1) % n]
        cr = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if cr == 0:
            return False
        s = 1 if cr > 0 else -1
        if sgn == 0:
            sgn = s
        elif s != sgn:
            return False
    return True


def amoeba_min(wu, wv, mu, mv, r1, r2, N=160):
    """min |P| on the torus |z|=e^{r1}, |w|=e^{r2}; >0 means a complement
    component (a gas facet or an unbounded component)."""
    m = mp.inf
    for i in range(N):
        z = mp.exp(r1) * mp.expjpi(mp.mpf(2 * i) / N)
        for j in range(N):
            w = mp.exp(r2) * mp.expjpi(mp.mpf(2 * j) / N)
            m = min(m, abs(P(z, w, wu, wv, mu, mv)))
    return m


# ------------------------------------------------------- the genus-1 fixture
def johansson_22(a):
    """mu = mv = 2 weights reproducing Johansson (7.2).

    DERIVED, not guessed: pushing twoperiodic.edge_weight through the diagonal
    coordinate change gives wu depending only on u and wv only on v.  (A guess
    that both depend on u+v gives a NODAL curve -- D(z) picks up a square
    factor and the genus drops to 0 -- so this is worth deriving.)"""
    wu = [[a if u % 2 == 0 else 1.0 for v in range(2)] for u in range(2)]
    wv = [[a if v % 2 == 0 else 1.0 for v in range(2)] for u in range(2)]
    return wu, wv


if __name__ == "__main__":
    a = mp.mpf('0.7')
    wu, wv = johansson_22(a)
    print("=== genus-1 fixture: mu = mv = 2, Johansson weights, a = 0.7 ===")
    pts = newton(wu, wv, 2, 2)
    verts, interior = hull_and_interior(pts)
    print(f"  Newton polygon vertices: {sorted(verts)}")
    print(f"  interior points ({len(interior)}): {sorted(interior)}   "
          f"[expect 1 -> genus 1]")
    for (x, y) in interior:
        m = amoeba_min(wu, wv, 2, 2, 0.0, 0.0, N=80)
        print(f"  min|P| on the unit torus = {mp.nstr(m, 6)}  "
              f"({'gas facet present' if m > 1e-9 else 'NO HOLE'})")
    print("\n  coefficients:")
    for k in sorted(pts):
        print(f"    z^{k[0]:<3} w^{k[1]:<3}  {complex(pts[k]):.6f}")
