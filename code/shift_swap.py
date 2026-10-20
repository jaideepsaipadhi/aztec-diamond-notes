"""
Configuration C: resolving the facet 1 <-> 3 interchange, and recovering the
individual velocities it hides.

THE SYMPTOM.  With the parity term included (shift_global.py) configuration A
matches the spectral-curve prediction v = -2 u(0_+) at 1.4 / 0.6 / 0.5 sigma,
chi^2 = 2.6 on 3 dof.  Configuration C does not: components 1 and 3 land 3.8 and
4.0 sigma away, chi^2 = 30.

THE DIAGNOSIS.  C's measured v1 = 0.270372 and v3 = 0.270010 are nearly equal,
and the mean of the two predicted values is (0.265853 + 0.274804)/2 = 0.270329.
Both components are sitting on the average of the pair -- the signature of the
two facets being interchanged at random from one n to the next, so that each
labelled component is a mixture of both true ones.  C's facets 1 and 3 are the
weak ones (facet 3 has 175 faces at n = 192), which is why the orientation-based
ordering of g3_facets.py separates them unreliably while facet 2 is never in
doubt.

It is confirmed by a parameter-free test.  The sum e1 + e3 is INVARIANT under
the interchange, and against the curve's own v1 + v3, with nothing fitted but
the offsets:

                      e1 alone     e3 alone     e1 + e3
    configuration C   sd 0.1037    sd 0.1056    sd 0.0359     <- 3x better
    configuration A   sd 0.0072    sd 0.0057    sd 0.0099     <- slightly worse

For A, whose facets are cleanly separated, summing merely adds noise, which is
what the third column shows.  For C the sum is where the signal is.

THE FIT.  Since v1 + v3 is well determined by the swap-invariant channel, only
the difference D = v3 - v1 is at issue, so this is a ONE-dimensional scan rather
than a search over two velocities.  For each candidate D, each n is assigned the
interchange that fits better, the per-parity offsets are profiled out, and the
pooled concentration is recorded.  The scan is run with S = v1 + v3 taken from
the data (not from the curve), so the curve's predicted D = 0.008951 is never
supplied and the peak location is a genuine recovery.

Guard against the obvious circularity: allowing a free per-n binary choice can
manufacture agreement, so the scan is also run on configuration A, where the
true interchange rate is zero and the swap freedom should buy almost nothing,
and on label-shuffled surrogates of C to measure what the freedom is worth by
itself.

Run:  python3 shift_swap.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from shift_ladder import gather, advance, climb   # noqa: E402

CURVE = {
    "A": np.array([0.2855890797123944, 0.4154709760294156, 0.2853120189148280]),
    "C": np.array([0.2658531592890836, 0.3971450056522778, 0.2748041555597199]),
}
FILES = {
    "A": ['run_shift2.log', 'run_shift_far.log', 'run_shift_traj.log'],
    "C": ['run_shift_c.log', 'run_c_far.log'],
}
RNG = np.random.default_rng(20261020)


def offsets(ph, par):
    """Per-parity circular mean, and the wrapped residuals about it."""
    c = np.zeros(2)
    r = np.empty_like(ph)
    for p in (0, 1):
        m = par == p
        if m.any():
            c[p] = np.angle(np.exp(2j * np.pi * ph[m]).mean()) / (2 * np.pi)
            r[m] = ((ph[m] - c[p] + 0.5) % 1.0) - 0.5
    return c, r


def fit_sum(ns, e1, e3, v0, half, ngrid=40001):
    """S = v1 + v3 from the swap-invariant channel."""
    par = ns % 2
    k = ns / 2.0
    grid = np.linspace(v0 - half, v0 + half, ngrid)
    best, bS = -1.0, v0
    for S in grid:
        _, r = offsets(e1 + e3 - k * S, par)
        sc = np.abs(np.exp(2j * np.pi * r).mean())
        if sc > best:
            best, bS = sc, S
    return bS, best


def scan_diff(ns, e1, e3, S, Dgrid, niter=6):
    """For each D, assign the interchange per n and score the pooled fit."""
    par = ns % 2
    k = ns / 2.0
    out = np.empty(len(Dgrid))
    swaps = np.empty((len(Dgrid), len(ns)), dtype=bool)
    for j, D in enumerate(Dgrid):
        v1, v3 = (S - D) / 2.0, (S + D) / 2.0
        sw = np.zeros(len(ns), dtype=bool)
        for _ in range(niter):
            a1 = np.where(sw, e3, e1) - k * v1
            a3 = np.where(sw, e1, e3) - k * v3
            c1, _ = offsets(a1, par)
            c3, _ = offsets(a3, par)
            # cost of each assignment for every n, given the offsets
            def cost(x, v, c):
                d = x - k * v - c[par]
                return 1.0 - np.cos(2 * np.pi * d)
            keep = cost(e1, v1, c1) + cost(e3, v3, c3)
            flip = cost(e3, v1, c1) + cost(e1, v3, c3)
            new = flip < keep
            if np.array_equal(new, sw):
                break
            sw = new
        a1 = np.where(sw, e3, e1) - k * v1
        a3 = np.where(sw, e1, e3) - k * v3
        _, r1 = offsets(a1, par)
        _, r3 = offsets(a3, par)
        out[j] = 0.5 * (np.abs(np.exp(2j * np.pi * r1).mean())
                        + np.abs(np.exp(2j * np.pi * r3).mean()))
        swaps[j] = sw
    return out, swaps


def run(name, verbose=True):
    a = gather(FILES[name])
    ns = a[:, 0].astype(int)
    e1, e2, e3 = a[:, 2], a[:, 3], a[:, 4]
    cu = CURVE[name]

    v, se, _ = advance(ns, a[:, 2:], 2)
    v, se = climb(v, se, ns, a[:, 2:], 14, "", verbose=False)
    S0 = v[0] + v[2]
    S, Rs = fit_sum(ns, e1, e3, S0, 6 * se.max())
    if verbose:
        print(f"\nconfiguration {name}: {len(ns)} points, n = {ns.min()}..{ns.max()}")
        print(f"  swap-invariant sum: v1 + v3 = {S:.6f}  (R = {Rs:.4f})")
        print(f"                      curve says {cu[0]+cu[2]:.6f}"
              f"   -> {abs(S-(cu[0]+cu[2])):.6f} apart")

    Dg = np.linspace(-0.05, 0.05, 4001)
    sc, sw = scan_diff(ns, e1, e3, S, Dg)
    j = int(np.argmax(sc))
    D = Dg[j]
    v1, v3 = (S - D) / 2.0, (S + D) / 2.0
    frac = sw[j].mean()
    if verbose:
        print(f"  swap-resolved scan peak at D = v3 - v1 = {D:+.6f}"
              f"   (curve: {cu[2]-cu[0]:+.6f})")
        print(f"    -> v1 = {v1:.6f}   curve {cu[0]:.6f}")
        print(f"       v3 = {v3:.6f}   curve {cu[2]:.6f}")
        print(f"    interchange rate {frac:.2f} of {len(ns)} points"
              f"   (score {sc[j]:.4f})")
    return dict(ns=ns, e1=e1, e3=e3, S=S, D=D, v1=v1, v3=v3,
                frac=frac, score=sc[j], Dg=Dg, sc=sc, cu=cu)


if __name__ == "__main__":
    print("=" * 72)
    print("  Resolving the facet interchange in configuration C")
    print("=" * 72)
    res = {n: run(n) for n in ("A", "C")}

    print("\n" + "=" * 72)
    print("  is the swap freedom doing the work?  (surrogate control)")
    print("=" * 72)
    for name in ("A", "C"):
        r = res[name]
        ns, e1, e3, S = r['ns'], r['e1'], r['e3'], r['S']
        null = []
        for _ in range(200):
            p = RNG.permutation(len(ns))
            s2, _ = scan_diff(ns, e1[p], e3[p], S,
                              np.linspace(-0.05, 0.05, 401))
            null.append(s2.max())
        null = np.array(null)
        z = (r['score'] - null.mean()) / null.std()
        print(f"  {name}: real score {r['score']:.4f} vs shuffled "
              f"{null.mean():.4f} +- {null.std():.4f}   -> {z:.1f} sigma")

    print("\n  Reading: for A the interchange rate should come out near zero and")
    print("  the recovered v1, v3 should match the per-component fit, since A's")
    print("  facets are cleanly separated.  For C a substantial rate together")
    print("  with recovery of the curve's D is the diagnosis confirmed.")
