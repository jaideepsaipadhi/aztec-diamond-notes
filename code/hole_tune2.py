"""
Bounded amoeba complement components WITHOUT assuming the symmetric family.

hole_tune.holes locates the gas facets by finding, for each r1, the r2-interval
around r2 = 0 that the amoeba misses.  That is only correct when C(z) = A(z),
i.e. for the symmetric sub-family WV[1] = WV[0]; off it the holes are not
centred on r2 = 0 and the routine returns nothing at all.  Both of the
asymmetric genus-3 configurations hit exactly that:

    configuration A (symmetric)   3 holes, widths 0.64 / 0.64 / 0.64
    configuration B (asymmetric)  holes() -> empty
    configuration C (asymmetric)  holes() -> empty

This is why the weight search could never optimise for facet HEALTH off the
symmetric family -- it had no curve-side measure of it, and g3_asym2.py fell
back to "facet sizes checked afterwards by a short sampling run".  That makes
each trial cost a sampling run instead of a fraction of a second, which is why
only a handful of configurations were ever tried.

The fix needs no symmetry.  P(z, w) is quadratic in w, so for |z| = e^{r1} the
two roots w_pm(z) give exactly two values of r2 = log|w| per argument of z.
Sweeping arg z traces out the amoeba's slice at that r1 as a union of
r2-intervals, and the BOUNDED gaps between them are the facets. Nothing here
refers to r2 = 0.

Facet linear size scales with the hole's extent, so the usable measure of a
configuration is the smallest bounded component's r2-extent (and its r1-extent),
computed here on an (r1, r2) grid by flood-filling the uncovered cells and
discarding any component touching the grid boundary.

CALIBRATION against the sampled truth, which is what makes the proxy usable:
configuration A's three holes of r2-extent 0.64 correspond to three healthy
facets (smallest ~2000 faces at n = 192); configuration C's weakest facet (175
faces) and configuration B's (24 faces at n = 96, gone under erode 10) pin the
low end.  See the __main__ block, which prints the calibration table.
"""
import numpy as np

MU, MV = 2, 4
_WS = np.array([1.0, 2.0, 3.0])
_VINV = np.linalg.inv(np.array([[w ** 2, w, 1.0] for w in _WS]))


def _kast_batch(zs, w, wu, wv, mu=MU, mv=MV):
    """K(z, w) for every z in zs at once -> array (len(zs), |B|, |W|)."""
    cells = [(u, v) for u in range(mu) for v in range(mv)]
    Bl = [c for c in cells if (c[0] + c[1]) % 2 == 0]
    Wl = [c for c in cells if (c[0] + c[1]) % 2 == 1]
    wi = {c: k for k, c in enumerate(Wl)}
    K = np.zeros((len(zs), len(Bl), len(Wl)), dtype=complex)
    for r, (u, v) in enumerate(Bl):
        for (du, dv) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            uu, vv = u + du, v + dv
            cu, cv = uu % mu, vv % mv
            mx, my = (uu - cu) // mu, (vv - cv) // mv
            if du:
                val, ph = wu[min(u, uu) % mu][v % mv], 1.0
            else:
                val, ph = wv[u % mu][min(v, vv) % mv], 1j
            K[:, r, wi[(cu, cv)]] += val * ph * zs ** mx * w ** my
    return K


def quad_batch(zs, wu, wv, mu=MU, mv=MV):
    """(A, Bc, C) for every z at once, from w*det K at w = 1, 2, 3."""
    rhs = np.stack([w * np.linalg.det(_kast_batch(zs, w, wu, wv, mu, mv))
                    for w in _WS])                      # (3, nz)
    return _VINV @ rhs                                   # (3, nz)


def slice_r2(r1, wu, wv, mu=MU, mv=MV, nth=721):
    """All r2 = log|w| on the amoeba at |z| = e^{r1}."""
    th = np.linspace(0.0, 2.0 * np.pi, nth, endpoint=False)
    zs = np.exp(r1) * np.exp(1j * th)
    A, Bc, C = quad_batch(zs, wu, wv, mu, mv)
    ok = np.abs(A) > 1e-300
    A, Bc, C = A[ok], Bc[ok], C[ok]
    d = np.sqrt(Bc * Bc - 4.0 * A * C)
    w = np.concatenate([(-Bc + d) / (2.0 * A), (-Bc - d) / (2.0 * A)])
    w = w[np.abs(w) > 1e-300]
    return np.log(np.abs(w))


def components(wu, wv, mu=2, mv=4, r1lim=3.0, r2lim=3.0, n1=241, n2=301,
               nth=1441, pad=1, mincells=20):
    """Bounded complement components of the amoeba, on an (r1, r2) grid.

    The defaults are CALIBRATED, not arbitrary.  nth is the number of arg z
    samples per slice and it is the parameter that matters: at nth = 481 the
    sampled r2 values are sparse enough to leave spurious gaps inside genuinely
    covered regions, which fragments configuration A's three holes into twelve
    specks and reports the smallest as 0.02 wide.  At nth = 1441 with pad = 1
    the routine returns exactly three components for configuration A with
    r1-extents 0.65 / 0.62 / 0.67, against hole_tune's 0.64 / 0.64 / 0.64 for
    the same configuration -- so it reproduces the symmetric-family answer it
    has to agree with, and only then is it trusted off that family.

    Returns a list of dicts with the r1/r2 extents and the cell count.
    """
    r1s = np.linspace(-r1lim, r1lim, n1)
    r2s = np.linspace(-r2lim, r2lim, n2)
    h2 = r2s[1] - r2s[0]
    covered = np.zeros((n1, n2), dtype=bool)
    for i, r1 in enumerate(r1s):
        vals = slice_r2(float(r1), wu, wv, mu, mv, nth)
        if vals.size == 0:
            covered[i, :] = True
            continue
        j = np.clip(np.searchsorted(r2s, vals), 0, n2 - 1)
        covered[i, j] = True
        for k in range(1, pad + 1):          # widen by the grid resolution
            covered[i, np.clip(j - k, 0, n2 - 1)] = True
            covered[i, np.clip(j + k, 0, n2 - 1)] = True

    free = ~covered
    lab = -np.ones_like(free, dtype=int)
    out = []
    cid = 0
    for i in range(n1):
        for j in range(n2):
            if not free[i, j] or lab[i, j] >= 0:
                continue
            stack = [(i, j)]
            lab[i, j] = cid
            cells = []
            touches = False
            while stack:
                a, b = stack.pop()
                cells.append((a, b))
                if a == 0 or a == n1 - 1 or b == 0 or b == n2 - 1:
                    touches = True
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    p, q = a + da, b + db
                    if 0 <= p < n1 and 0 <= q < n2 and free[p, q] \
                            and lab[p, q] < 0:
                        lab[p, q] = cid
                        stack.append((p, q))
            cid += 1
            if touches or len(cells) < mincells:
                continue
            ii = np.array([c[0] for c in cells])
            jj = np.array([c[1] for c in cells])
            out.append(dict(
                n=len(cells),
                r1=(float(r1s[ii.min()]), float(r1s[ii.max()])),
                r2=(float(r2s[jj.min()]), float(r2s[jj.max()])),
                r2ext=float(r2s[jj.max()] - r2s[jj.min()] + h2),
                r1ext=float(r1s[ii.max()] - r1s[ii.min()]),
            ))
    out.sort(key=lambda d: d["r1"][0])
    return out


def health(WU, WV, **kw):
    """(number of bounded components, smallest r2-extent) -- the search score."""
    c = components(WU, WV, **kw)
    if not c:
        return 0, 0.0
    return len(c), min(d["r1ext"] for d in c)


if __name__ == "__main__":
    import time

    print("CALIBRATION of the curve-side facet proxy against sampled truth\n")
    truth = {
        'A': "3 healthy facets; smallest ~2000 faces at n=192",
        'B': "facet 1 = 24 faces at n=96, 0 under erode 10",
        'C': "facet 3 = 175 faces at n=192, atoms chi2 86.8/6",
    }
    for nm, f in (('A', 'g3_best_weights.npy'),
                  ('B', 'g3b_weights.npy'),
                  ('C', 'g3c_weights.npy')):
        W = np.load(f'/home/claude/{f}')
        t = time.time()
        c = components(W[0].tolist(), W[1].tolist())
        print(f"  configuration {nm}: {len(c)} bounded components"
              f"   [{time.time()-t:.1f}s]")
        for d in c:
            print(f"      r1 {d['r1'][0]:+.2f}..{d['r1'][1]:+.2f}"
                  f"   r2 {d['r2'][0]:+.2f}..{d['r2'][1]:+.2f}"
                  f"   r2-extent {d['r2ext']:.3f}  cells {d['n']}")
        if c:
            print(f"      smallest r2-extent {min(d['r2ext'] for d in c):.3f}")
        print(f"      sampled truth: {truth[nm]}\n")
