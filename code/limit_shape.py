"""
The limit shape for the Aztec diamond with quasi-periodic (Schottky) weights.

Boundary data on the unit square, the four frozen phases sitting at the four
polygon vertices (0,0), (1,0), (0,1), (1,1):

    bottom (y=0): h = 0        left  (x=0): h = 0
    right  (x=1): h = y        top   (y=1): h = x

Solved by the LP of lp_solve.py against the slope-space sigma table.  The result
carries all three phases: frozen corners, a liquid annulus, and a gas facet where
grad h sits at the interior point of the polygon.

CAVEAT: the phase classification thresholds below are arbitrary.  The existence
and arrangement of the three phases is robust to them; the reported fractions are
not.  And none of this is validated against anything external yet -- the sampler
comparison is what would do that.
"""
import numpy as np

from lp_solve import solve

GAS_SLOPE = (0.4716546, 0.5170360)      # Delta(compact oval), the gas facet
VERTS = [(0, 0), (1, 0), (0, 1), (1, 1)]


def aztec_boundary(N):
    xs = np.linspace(0, 1, N)
    h = np.zeros((N, N))
    h[:, 0] = 0.0
    h[0, :] = 0.0
    h[-1, :] = xs
    h[:, -1] = xs
    return h


def phases(h, N, fro_tol=0.15, gas_tol=0.10):
    hg = 1.0 / (N - 1)
    g1 = np.diff(h, axis=0)[:, :-1] / hg
    g2 = np.diff(h, axis=1)[:-1, :] / hg
    dfro = np.min([np.hypot(g1-vx, g2-vy) for vx, vy in VERTS], axis=0)
    dgas = np.hypot(g1-GAS_SLOPE[0], g2-GAS_SLOPE[1])
    frozen = dfro < fro_tol
    gas = (dgas < gas_tol) & (~frozen)
    return frozen, gas, (g1, g2)


def run(table, N):
    hb = aztec_boundary(N)
    h, E, _ = solve(table, N, (0.5, 0.5), hbnd=hb, verbose=False)
    frozen, gas, grads = phases(h, N)
    return h, E, frozen, gas, grads


if __name__ == "__main__":
    import time
    tab = np.load('/home/claude/sigma_table_slope2.npy')
    print(f"sigma table: {len(tab)} slope-space points\n")
    print("  N    energy      frozen%  gas%  liquid%   time")
    for N in (15, 19, 25):
        t0 = time.time()
        h, E, fro, gas, _ = run(tab, N)
        print(f"  {N:<4} {E:.6f}   {100*fro.mean():5.1f}  {100*gas.mean():5.1f} "
              f"  {100*(1-fro.mean()-gas.mean()):5.1f}    {time.time()-t0:.0f}s")

    print("\nphase map at N=25   F=frozen  G=gas  .=liquid")
    h, E, fro, gas, _ = run(tab, 25)
    for i in range(fro.shape[0]):
        print("   " + "".join('F' if fro[i, j] else ('G' if gas[i, j] else '.')
                              for j in range(fro.shape[1])))
