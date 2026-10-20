"""
A facet mask that is stable across n.

The detector in g3_measure.find_facets admits a face if its mean-height slope
matches a facet plateau AND its height variance is below median(V) over all
faces.  That second condition is the problem: median(V) is 3.71 at n = 96 while
the facets themselves sit at 1.7-4.3, so the threshold cuts straight through the
signal.  As n changes, the frozen fraction changes, the median moves, and whole
facets cross in and out -- which is why a sweep over n produced facet sizes
bouncing between 25 and 510 with three sizes collapsing to zero.

That does not matter much for a single measurement at one n (the surviving mask
is still inside the facet), but for a sweep over n it is fatal: a mask that
drifts towards a facet edge picks up liquid, which biases the fitted shift, and
the bias would masquerade as motion in e(n) -- the very thing being measured.

Here the variance is used only to exclude the frozen phase (V == 0 exactly).
Facets are identified by slope plateau alone, then reduced to the largest
contiguous y-band, then shrunk by a fixed fraction so the mask sits well inside.
No global threshold anywhere, so nothing moves when the frozen fraction does.
"""
import numpy as np

TARGETS = (-4.0, 0.0, 4.0)


def detect_oriented(order, pidx, V, Mn, **kw):
    """Facets ordered by POSITION (most negative y first), plus the orientation
    sign of the height function.

    Both are needed for a sweep over n.  The Thurston height's global
    orientation flips with the parity of n -- even n puts slope -4 at negative
    y, odd n puts +4 there -- so h -> -h between consecutive n.  Labelling
    facets by slope target therefore SWAPS facets 1 and 3 every step, and the
    sign flip sends Z -> -Z and hence e -> -e.  Left uncorrected, that alone
    would make the shift appear to move on the torus: a perfect fake of the
    phenomenon being looked for.

    Returns (facets_by_y, sigma) where sigma = +1 if the facet at the most
    negative y has slope -4 (standard), -1 if it has +4 (flipped).  Multiply the
    measured Z by sigma to put every n in the same frame.
    """
    raw = detect(order, pidx, V, Mn, **kw)
    found = [(k, sel, info) for k, (sel, info) in raw.items() if sel]
    if len(found) != 3:
        return None, None, {k: raw[k] for k in raw}
    found.sort(key=lambda t: t[2]["ymid"])
    sigma = 1 if TARGETS[found[0][0]] < 0 else -1
    facets = [(sel, info) for _, sel, info in found]
    return facets, sigma, None


def detect(order, pidx, V, Mn, slope_tol=0.35, sx_tol=0.30,
           keep_y=0.55, keep_x=0.60, min_row=6):
    """Return {k: (faces, info)} for the three facets, keyed by slope target."""
    out = {}
    for k, target in enumerate(TARGETS):
        cand = []
        for f in order:
            a, b = (f[0], f[1] + 4), (f[0] + 2, f[1])
            if a not in pidx or b not in pidx:
                continue
            if V[pidx[f]] <= 1e-9:            # frozen -- the ONLY use of V
                continue
            sy = Mn[pidx[a]] - Mn[pidx[f]]
            sx = Mn[pidx[b]] - Mn[pidx[f]]
            if abs(sy - target) < slope_tol and abs(sx) < sx_tol:
                cand.append(f)
        if not cand:
            out[k] = ([], dict(reason="no candidates"))
            continue

        # rows with enough qualifying faces, then the longest contiguous run
        rows = {}
        for f in cand:
            rows.setdefault(f[1], []).append(f)
        good = sorted(y for y, fs in rows.items() if len(fs) >= min_row)
        if not good:
            out[k] = ([], dict(reason="no dense rows"))
            continue
        runs, cur = [], [good[0]]
        for y in good[1:]:
            if y - cur[-1] <= 2:
                cur.append(y)
            else:
                runs.append(cur)
                cur = [y]
        runs.append(cur)
        run = max(runs, key=len)
        y0, y1 = run[0], run[-1]
        ymid = 0.5 * (y0 + y1)
        hy = max(1.0, keep_y * 0.5 * (y1 - y0))

        band = [f for f in cand if abs(f[1] - ymid) <= hy]
        xs = [f[0] for f in band]
        xmid = 0.5 * (min(xs) + max(xs))
        hx = max(1.0, keep_x * 0.5 * (max(xs) - min(xs)))
        sel = [f for f in band if abs(f[0] - xmid) <= hx]
        out[k] = (sel, dict(ymid=ymid, hy=hy, xmid=xmid, hx=hx,
                            yspan=(y0, y1), n=len(sel)))
    return out


if __name__ == "__main__":
    import sys
    import time

    import g3_sample as GS
    from g3_measure import height_profile, tree_arrays
    from spatial_avg import bfs_tree, face_graph

    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else 99
    Kp = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    print("stability of the facet mask across n "
          f"(profiling with {Kp} samples per n)\n")
    print(f"{'n':>4} | {'facet 1: size  y-centre':>26} | "
          f"{'facet 2':>24} | {'facet 3':>24}")
    for n in range(lo, hi + 1):
        t0 = time.time()
        idx, E, F, steps, adj = face_graph(n)
        root, parent, order = bfs_tree(adj)
        pidx, pe, ps, pp = tree_arrays(order, parent)
        levels, _, EE = GS.build_levels(n)
        Hp = height_profile(n, levels, E, order, pidx, pe, ps, pp, Kp, 7 + n)
        V, Mn = Hp.var(0), Hp.mean(0)
        fac = detect(order, pidx, V, Mn)
        cells = []
        for k in range(3):
            sel, info = fac[k]
            if not sel:
                cells.append(f"{'FAILED':>26}")
            else:
                cells.append(f"{len(sel):>8}  y={info['ymid']:+7.1f}"
                             f" ({info['ymid']/n:+.3f}n)")
        print(f"{n:>4} | " + " | ".join(cells) + f"   [{time.time()-t0:.0f}s]",
              flush=True)
