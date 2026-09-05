"""
Sample the quasi-periodic (Schottky/Fock) dimer model at large n and extract the
empirical height profile -- the external check for the computed limit shape.

Weights: W_f depends on the face only through eta = x*de + y*dn and is exactly
1-periodic in eta, so the mpmath theta evaluation is replaced by a lookup in
fock_W_table.npy (1024 points, interpolation error 3.6e-8 relative).  That turns
~48 minutes of weight generation at n=200 into numpy work.

Gauge: |W_f| is the positive dimer face weight (W_f itself carries the Kasteleyn
sign).  Horizontal edges are set to 1 and the verticals follow by the per-row
recursion v_{x+1} = 1/(W_f v_x), which is contractive here (|W| ~ 0.95) and
measured stable: vertical-weight spread ratio 1.12 / 1.25 / 1.29 at n = 8/16/24.

The torsion shortcut does NOT apply to quasi-periodic weights, so genshuffle's
full N-level descent is used.
"""
import numpy as np

from genshuffle import verts, edges_of, sample
from ndep import faces_and_steps, height_at
from vecshuffle import edge_index

TAB = np.load('/home/claude/fock_W_table.npy')
WTAB = TAB[0]
DE, DN = TAB[1][0], TAB[1][1]
M = len(WTAB)


def Wf(x, y):
    eta = (x*DE + y*DN) % 1.0
    return float(np.interp(eta, np.arange(M)/M, WTAB, period=1.0))


def fock_weights(n):
    """Edge weights on A_n from the Fock construction, via the eta lookup."""
    V = verts(n)
    w = {}
    for (x, y) in V:
        if (x+1, y) in V:
            w[tuple(sorted(((x, y), (x+1, y))))] = 1.0
    ys = sorted({y for (_, y) in V})
    for y in ys:
        xs = sorted(x for (x, yy) in V if yy == y and (x, y+1) in V)
        if not xs:
            continue
        v = 1.0
        w[tuple(sorted(((xs[0], y), (xs[0], y+1))))] = v
        for x in xs[:-1]:
            v = 1.0 / (Wf(x, y) * v)
            w[tuple(sorted(((x+1, y), (x+1, y+1))))] = v
    return w


def sample_height(n, rng):
    w = fock_weights(n)
    wfn = lambda e: w[tuple(sorted(e))]
    match = sample(n, wfn, rng)
    idx, _ = edge_index(n)
    M_ = {tuple(sorted(e)) for e in match}
    Mi = np.zeros(len(idx), bool)
    for e, i in idx.items():
        if e in M_:
            Mi[i] = True
    F, steps = faces_and_steps(n, idx)
    _, h, _ = height_at(steps, F, Mi, (0, 0))
    return h


if __name__ == "__main__":
    import time
    rng = np.random.default_rng(0)
    print(f"eta increments: de={DE:.6f}  dn={DN:.6f}")
    print(f"W(eta) in [{WTAB.min():.6f}, {WTAB.max():.6f}]\n")

    for n in (16, 32, 60):
        t0 = time.time()
        w = fock_weights(n)
        t1 = time.time()
        h = sample_height(n, rng)
        t2 = time.time()
        vals = np.array(list(w.values()))
        print(f"n={n:<4} weights {len(w):>6} in [{vals.min():.4f},{vals.max():.4f}] "
              f"({t1-t0:.1f}s)  sample+height {t2-t1:.1f}s  faces {len(h)}")

    n = 60
    h = sample_height(n, rng)
    print(f"\nheight profile along the diagonal at n={n} "
          f"(paper units = -(h_Thurston - h(0,0))/4):")
    h0 = h[(0, 0)]
    for t in range(-n, n+1, max(1, n//8)):
        f = (t, t)
        if f in h:
            print(f"   face ({t:+4d},{t:+4d})   h_paper = {-(h[f]-h0)/4:+8.2f}")
