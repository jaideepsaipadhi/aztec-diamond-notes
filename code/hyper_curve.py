"""
From a periodic Aztec/square-lattice dimer model to its hyperelliptic curve.

For periods (mu, mv) with mv = 2 the characteristic polynomial P(z,w) has
w-degree 2 after clearing w^{-1}, so

        w P(z,w) = A(z) w^2 + Bc(z) w + C(z),     D(z) = Bc^2 - 4 A C,

and the spectral curve is y^2 = D(z), hyperelliptic of genus g = (number of
interior Newton points).  Periodic dimer curves are Harnack, so all 2g+2 branch
points are real -- which is what makes hyper_g3.py's real-axis quadrature
applicable.  We verify the count rather than assume it.

A, Bc, C are recovered at full precision without symbolic algebra: for fixed z,
w P(z,w) is a quadratic in w, so evaluating at three values of w and solving the
3x3 Vandermonde gives the coefficients exactly.  (Reading them off the complex128
FFT instead would cap the branch points at ~1e-16 relative, which is not enough
for the near-degenerate outer roots.)
"""
import mpmath as mp

from periodic_dimer import P, newton, hull_and_interior

mp.mp.dps = 30


def quad_coeffs(z, wu, wv, mu, mv):
    """(A, Bc, C) with w*P(z,w) = A w^2 + Bc w + C."""
    ws = [mp.mpf(1), mp.mpf(2), mp.mpf(3)]
    V = mp.matrix([[w ** 2, w, mp.mpf(1)] for w in ws])
    rhs = mp.matrix([w * P(z, w, wu, wv, mu, mv) for w in ws])
    c = mp.lu_solve(V, rhs)
    return c[0], c[1], c[2]


def Dz(z, wu, wv, mu, mv):
    A, Bc, C = quad_coeffs(z, wu, wv, mu, mv)
    return Bc * Bc - 4 * A * C


def branch_points(wu, wv, mu, mv, lo=-400.0, hi=4.0, n=6000, expect=None):
    """Real roots of D by sign change plus refinement.  The curve is Harnack so
    all of them are real and negative for these models; the window is checked by
    counting."""
    f = lambda t: mp.re(Dz(mp.mpf(t), wu, wv, mu, mv))
    xs = []
    # geometric spacing: the roots span several decades in |z|
    import math
    for i in range(n + 1):
        t = -math.exp(math.log(abs(lo)) * (1 - i / n) + math.log(1e-4) * (i / n))
        xs.append(t)
    roots, prev = [], None
    for x in xs:
        try:
            v = f(x)
        except Exception:
            prev = None
            continue
        if prev is not None and mp.sign(v) != mp.sign(prev[1]):
            try:
                r = mp.findroot(f, (prev[0] + x) / 2)
                if mp.im(r) == 0 and all(abs(r - q) > 1e-12 * max(1, abs(r))
                                         for q in roots):
                    roots.append(mp.re(r))
            except Exception:
                pass
        prev = (x, v)
    roots = sorted(roots)
    if expect is not None and len(roots) != expect:
        print(f"    WARNING: found {len(roots)} real branch points, expected "
              f"{expect} -- widen the window or refine the grid")
    return roots


def curve(wu, wv, mu, mv, verbose=True):
    pts = newton(wu, wv, mu, mv, zdeg=mu + 2, wdeg=mv + 2)
    verts, interior = hull_and_interior(pts)
    g = len(interior)
    if verbose:
        print(f"  Newton polygon: z {min(p[0] for p in pts)}..{max(p[0] for p in pts)}"
              f"  w {min(p[1] for p in pts)}..{max(p[1] for p in pts)}")
        print(f"  interior points ({g}): {sorted(interior)}   -> genus {g}")
    r = branch_points(wu, wv, mu, mv, expect=2 * g + 2)
    if verbose:
        print(f"  real branch points ({len(r)}): "
              + ", ".join(mp.nstr(x, 8) for x in r))
    return g, r, interior


if __name__ == "__main__":
    from hyper_g3 import period_matrix
    from periodic_dimer import johansson_22

    print("=== fixture: (2,2) Johansson, a = 0.7 -> genus 1, b = 0.772738 ===")
    wu, wv = johansson_22(mp.mpf('0.7'))
    g, r, interior = curve(wu, wv, 2, 2)
    ImB = period_matrix(r)
    print(f"  b = {mp.nstr(ImB[0,0], 13)}   (expect 0.772737661704)\n")

    print("=== genus 3: periods (2,4) ===")
    wu = [[mp.mpf('1.0'), mp.mpf('0.55'), mp.mpf('1.0'), mp.mpf('0.55')],
          [mp.mpf('0.8'), mp.mpf('1.3'), mp.mpf('0.8'), mp.mpf('1.3')]]
    wv = [[mp.mpf('0.6'), mp.mpf('1.2'), mp.mpf('0.9'), mp.mpf('1.4')],
          [mp.mpf('0.6'), mp.mpf('1.2'), mp.mpf('0.9'), mp.mpf('1.4')]]
    g, r, interior = curve(wu, wv, 2, 4)
    if len(r) == 2 * g + 2:
        # refine the branch points first: they span three decades here, and the
        # sign-change scan alone is not accurate enough for the quadrature
        f = lambda t: mp.re(Dz(t, wu, wv, 2, 4))
        r = [mp.findroot(f, mp.mpf(str(x)), tol=mp.mpf('1e-50')) for x in r]
        ImB = period_matrix(r)
        print("\n  Im B =")
        for i in range(g):
            print("    " + "  ".join(f"{mp.nstr(ImB[i,j], 9):>13}" for j in range(g)))
