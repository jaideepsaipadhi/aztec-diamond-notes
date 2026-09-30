"""
Choose (2,4)-periodic weights whose THREE gas facets are all large enough to
measure.

The first weight guess gave bounded amoeba holes of width 0.30, 1.40, 0.30 in
r1, so at n = 64 only the middle facet is visible and the genus-3 discrete
component collapses to one measurable component.  Facet linear size scales with
hole width, so we search for weights maximising the SMALLEST bounded hole.

Exact and fast: P(z,w) is quadratic in w with w+ w- = C/A, so for |z| = e^{r1}
the amoeba slice is {log|w_pm(z)|}, and by the w -> 1/w symmetry of this family
it is symmetric in r2.  The bounded holes therefore all lie on r2 = 0 and are the
r1-intervals where min_z |log|w(z)|| > 0 -- computed by solving the quadratic, not
by minimising |P| on a grid (which misses the zeros entirely; a 48x48 torus scan
reported no amoeba at all).

Everything here is float64 numpy; the winning weights are re-checked at full
precision by hyper_g3.py.
"""
import numpy as np


def kzw_np(z, w, wu, wv, mu, mv):
    cells = [(u, v) for u in range(mu) for v in range(mv)]
    B = [c for c in cells if (c[0] + c[1]) % 2 == 0]
    W = [c for c in cells if (c[0] + c[1]) % 2 == 1]
    wi = {c: k for k, c in enumerate(W)}
    K = np.zeros((len(B), len(W)), dtype=complex)
    for r, (u, v) in enumerate(B):
        for (du, dv) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            uu, vv = u + du, v + dv
            cu, cv = uu % mu, vv % mv
            mx, my = (uu - cu) // mu, (vv - cv) // mv
            if du:
                val = wu[min(u, uu) % mu][v % mv]
                ph = 1.0
            else:
                val = wv[u % mu][min(v, vv) % mv]
                ph = 1j
            K[r, wi[(cu, cv)]] += val * ph * z ** mx * w ** my
    return K


def quad_np(z, wu, wv, mu, mv):
    """A, B, C with w det K = A w^2 + B w + C."""
    ws = np.array([1.0, 2.0, 3.0])
    V = np.array([[w ** 2, w, 1.0] for w in ws])
    rhs = np.array([w * np.linalg.det(kzw_np(z, w, wu, wv, mu, mv)) for w in ws])
    return np.linalg.solve(V, rhs)


def slice_min(r1, wu, wv, mu=2, mv=4, N=180):
    th = 2 * np.pi * np.arange(N) / N
    zs = np.exp(r1) * np.exp(1j * th)
    best = np.inf
    for z in zs:
        A, B, C = quad_np(z, wu, wv, mu, mv)
        d = np.sqrt(B * B - 4 * A * C + 0j)
        for s in (1, -1):
            w = (-B + s * d) / (2 * A)
            if w != 0:
                best = min(best, abs(np.log(abs(w))))
    return best


def holes(wu, wv, lo=-5.0, hi=4.0, step=0.04, **kw):
    rs = np.arange(lo, hi + 1e-9, step)
    v = np.array([slice_min(float(r), wu, wv, **kw) for r in rs])
    out, inh = [], False
    for r, x in zip(rs, v):
        if x > 1e-7 and not inh:
            start, inh = r, True
        if x <= 1e-7 and inh:
            out.append((start, r))
            inh = False
    if inh:
        out.append((start, rs[-1]))
    # drop the two unbounded components (they touch the scan ends)
    bounded = [h for h in out if h[0] > lo + step / 2 and h[1] < hi - step / 2]
    return bounded


def score(wu, wv):
    h = holes(wu, wv)
    if len(h) != 3:
        return -1.0, h
    return min(b - a for a, b in h), h


if __name__ == "__main__":
    import sys
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    base_u = [[1.0, 0.55, 1.0, 0.55], [0.8, 1.3, 0.8, 1.3]]
    base_v = [[0.6, 1.2, 0.9, 1.4], [0.6, 1.2, 0.9, 1.4]]
    s, h = score(base_u, base_v)
    print(f"baseline: min bounded hole {s:.3f}   holes "
          + ", ".join(f"[{a:+.2f},{b:+.2f}]w={b-a:.2f}" for a, b in h))

    best = (s, base_u, base_v, h)
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    for t in range(trials):
        wu = [[float(np.exp(rng.uniform(-0.9, 0.9))) for _ in range(4)]
              for _ in range(2)]
        wv = [[float(np.exp(rng.uniform(-0.9, 0.9))) for _ in range(4)]
              for _ in range(2)]
        wv[1] = list(wv[0])            # keep wv independent of u, as in genus 1
        s, h = score(wu, wv)
        if s > best[0]:
            best = (s, wu, wv, h)
            print(f"  trial {t:3d}: min hole {s:.3f}   "
                  + ", ".join(f"w={b-a:.2f}" for a, b in h), flush=True)
    s, wu, wv, h = best
    print(f"\nbest min bounded hole = {s:.3f}")
    print("  holes " + ", ".join(f"[{a:+.2f},{b:+.2f}] width {b-a:.2f}"
                                 for a, b in h))
    print(f"  WU = {[[round(x,5) for x in r] for r in wu]}")
    print(f"  WV = {[[round(x,5) for x in r] for r in wv]}")
