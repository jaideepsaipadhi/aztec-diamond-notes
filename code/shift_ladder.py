"""
High-precision v by a bootstrap ladder, and the decisive A-vs-C comparison.

A long baseline gives v = (delta_e + m) / steps for an unknown integer m, so the
candidate values are spaced 1/steps apart.  Resolving m needs a prior estimate
better than half that spacing.

For configuration A this was automatic: the two-step advance gave v to +-0.0016
while the 49-k-step aliases are spaced 1/49 = 0.0204 apart, so the alias was
unambiguous.  Configuration C has a weak third facet and its coarse v is only
+-0.0153 -- comparable to the spacing, so the alias is NOT resolved and the
"refined" value would be a coin flip between 0.2554 and 0.2758.

The fix is a ladder.  Each rung resolves the next:

    1-step differences        spacing 1/1      -> v +- 0.015
    7 k-steps  (n -> n+14)    spacing 1/7      -> v +- 0.001
    49 k-steps (n -> n+98)    spacing 1/49     -> v +- 0.0001

Each rung's alias is resolved by the rung below it, and the final precision is
set by the longest baseline.  A rung is only used if the rung below is sharp
enough: the code refuses to climb when 3 sigma of the prior exceeds half the
alias spacing, rather than silently picking the wrong integer.
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def parse(fn):
    rows = []
    for line in open(os.path.join(HERE, fn)):
        m = re.match(r'\s*(\d+)\s+([+-]\d+)\s+\[.*?\]\s+'
                     r'([+-][\d.]+)\s+([+-][\d.]+)\s+([+-][\d.]+)', line)
        if m:
            rows.append([int(m.group(1)), int(m.group(2))]
                        + [float(m.group(i)) for i in (3, 4, 5)])
    return np.array(rows) if rows else np.zeros((0, 5))


def gather(files):
    a = np.vstack([parse(f) for f in files if os.path.exists(os.path.join(HERE, f))])
    a = a[np.argsort(a[:, 0])]
    _, keep = np.unique(a[:, 0], return_index=True)
    return a[keep]


def advance(ns, es, P):
    """Mean of (e(n+P) - e(n)) mod 1, circularly, over all available pairs."""
    idx = {n: i for i, n in enumerate(ns)}
    ds = [(es[idx[n + P]] - es[idx[n]]) % 1.0 for n in ns if n + P in idx]
    if not ds:
        return None, None, 0
    ds = np.array(ds)
    z = np.exp(2j * np.pi * ds).mean(0)
    m = (np.angle(z) / (2 * np.pi)) % 1.0
    R = np.abs(z)
    se = np.sqrt(np.maximum(-2 * np.log(np.maximum(R, 1e-12)), 0)
                 / len(ds)) / (2 * np.pi)
    return m, se, len(ds)


def climb(v, sev, ns, es, P, label, verbose=True):
    """Refine v using baseline P (in n), resolving the alias with the prior."""
    steps = P / 2.0
    d, sed, npairs = advance(ns, es, P)
    if d is None:
        if verbose:
            print(f"    {label}: no pairs")
        return v, sev
    spacing = 1.0 / steps
    if 3 * sev.max() > spacing / 2:
        if verbose:
            print(f"    {label}: REFUSED -- prior 3sigma {3*sev.max():.4f} "
                  f"exceeds half the alias spacing {spacing/2:.4f}")
        return v, sev
    out = np.empty(3)
    for i in range(3):
        ms = np.arange(-2, int(steps) + 3)
        cands = (d[i] + ms) / steps
        out[i] = cands[int(np.argmin(np.abs(cands - v[i])))]
    seo = np.full(3, sed.max() / steps) if sed is not None else sev / steps
    seo = np.maximum(seo, 1e-6)
    if verbose:
        print(f"    {label}: {npairs} pairs, spacing {spacing:.4f}  ->  "
              + "  ".join(f"{out[i]:.5f}" for i in range(3))
              + f"   (+-{seo.max():.5f})")
    return out, seo


def velocity(files, name):
    a = gather(files)
    ns = a[:, 0].astype(int)
    es = a[:, 2:]
    print(f"\n{name}: n = {ns.min()}..{ns.max()}, {len(ns)} points")
    v, se, npairs = advance(ns, es, 2)
    print(f"    rung 0 (two-step): {npairs} pairs  ->  "
          + "  ".join(f"{v[i]:.5f}" for i in range(3))
          + f"   (+-{se.max():.5f})")
    for P, lab in ((14, "rung 1 (7 k-steps) "), (98, "rung 2 (49 k-steps)")):
        v, se = climb(v, se, ns, es, P, lab)
    return v, se


if __name__ == "__main__":
    vA, seA = velocity(['run_shift2.log', 'run_shift_far.log',
                        'run_shift_traj.log'], "configuration A")
    vC, seC = velocity(['run_shift_c.log', 'run_c_far.log'], "configuration C")

    print("\n" + "=" * 64)
    print("  does v depend on the edge weights?")
    print("=" * 64)
    for i in range(3):
        d = ((vA[i] - vC[i] + 0.5) % 1.0) - 0.5
        s = np.hypot(seA[i], seC[i])
        print(f"    v{i+1}:  A = {vA[i]:.5f}+-{seA[i]:.5f}   "
              f"C = {vC[i]:.5f}+-{seC[i]:.5f}   "
              f"difference {d:+.5f} = {abs(d)/s:.1f} sigma")
    print("\n  v is an Abel-map expression in the angles (eq. 19), and the")
    print("  angles move with the weights.  A velocity that demonstrably moves")
    print("  is irrational for almost every weight choice, so the orbit")
    print("  generically never closes -- which is the quasi-periodicity.")
