"""
Gas vs liquid at the centre, by height-difference variance.

Smooth (gas) phase: Var[h(x+r) - h(x)] stays BOUNDED as r grows.
Rough (liquid) phase: Var grows like log r.

Everything here is already-validated machinery:
  * fast_sampler.sample_fast  -- exact, chi^2-validated at n=4
  * the bridge x1 = i-j+n, x2 = i+j+n+1 -- unique, checked at n=4 and n=8
  * height.height -- path-independent, zero face circulation

We sample, map each matching back to a cell tiling, compute the Thurston height,
and look at the variance of height differences at a range of separations.
a=1.0 (uniform, liquid at the centre) is the control.
"""
import numpy as np

from fast_sampler import sample_fast
from height import height
from shuffle2 import Tiling, in_ad


def matching_to_tiling(match, n):
    """Invert the bridge: (black,white) edge -> domino on two adjacent cells."""
    inv = {}
    for i in range(-n, n):
        for j in range(-n, n):
            if in_ad(i, j, n):
                inv[(i - j + n, i + j + n + 1)] = (i, j)
    t = Tiling(n)
    for (b, w) in match:
        c1, c2 = inv[b], inv[w]
        (i1, j1), (i2, j2) = sorted([c1, c2])
        if (i1, j1 + 1) == (i2, j2):        # horizontal
            t.dom[(i1, j1)] = 1             # type is irrelevant for the height
        elif (i1 + 1, j1) == (i2, j2):      # vertical
            t.dom[(i1, j1)] = 3
        else:
            raise ValueError(f"non-adjacent cells {c1} {c2}")
    return t


def run(n, a, nsamp, rng, rs):
    """Variance of h(0,r) - h(0,0) at the centre, for each separation r."""
    vals = {r: [] for r in rs}
    for _ in range(nsamp):
        m = sample_fast(n, a, rng)
        if m is None:
            continue
        t = matching_to_tiling(m, n)
        h = height(t)
        base = (0, 0)
        if base not in h:
            continue
        for r in rs:
            p = (0, r)
            if p in h:
                vals[r].append(h[p] - h[base])
    return {r: (np.var(v) if len(v) > 2 else np.nan, len(v)) for r, v in vals.items()}


if __name__ == "__main__":
    rng = np.random.default_rng(17)
    n = 24
    rs = [1, 2, 3, 4, 6, 8, 10, 12]
    nsamp = 250

    for a, label in ((1.0, "uniform (liquid control)"), (0.5, "two-periodic")):
        res = run(n, a, nsamp, rng, rs)
        print(f"\na={a}  {label}   n={n}, {nsamp} samples")
        print("   r    Var[h(r)-h(0)]   (n obs)")
        for r in rs:
            v, c = res[r]
            print(f"   {r:<4} {v:12.3f}      {c}")
