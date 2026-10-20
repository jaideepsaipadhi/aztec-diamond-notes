"""
The shift velocity by a GLOBAL circular fit, replacing the bootstrap ladder.

The ladder of shift_ladder.py resolves the alias correctly but wastes the data
and mis-estimates its own error.  Two faults, both visible in configuration C:

  1. It measures the final rung from the ENDPOINT PAIRS only.  Configuration C
     has 18 values of n but rung 2 used 2 pairs, i.e. 4 numbers.
  2. It estimates the error from the circular concentration of those pairs.  A
     concentration from 2 samples is meaningless, and it came out such that the
     49-k-step rung (+-0.00531) was reported as WORSE than the 7-k-step rung
     (+-0.00170).  A longer lever arm cannot lose precision; the number is an
     artifact of the estimator, not of the data.

Propagating C's own per-n scatter (se_e ~ 0.038, from the rung-0 concentration
over 14 pairs) across 49 k-steps gives se_v ~ 0.0011.  So C's v is about 5x
better determined than the ladder reported.

THE MODEL, AND THE PARITY TERM.  A k-step is n -> n+2, so odd and even n are
two different sublattices and purely linear advance,

    e_i(n) = c_i + (n/2) v_i   (mod 1),                              [WRONG]

is misspecified.  Fitting it leaves residuals with lag-1 autocorrelation
rho = -0.997 (permutation p < 1e-4) -- a near-perfect alternation, not noise.
The correct model carries an n-parity term,

    e_i(n) = c_i + (n/2) v_i + (-1)^n d_i   (mod 1),

which is the alignment of the weight pattern with the diamond: the same
phenomenon as the orientation flip of the height function with n parity, and a
finite-n effect of the kind Remark 4.18 describes, not part of v.

This resolves what otherwise looks like a contradiction between estimators.
Both available long-baseline pairs, (88, 186) and (89, 187), have Dn = 98, which
is EVEN, so (-1)^n is unchanged and d cancels identically in both differences.
That is why they agree to 0.005 while the raw residual scatter is 0.11, and why
the endpoint ladder's small error is defensible while a fit that treats d as
noise reports an error ~20x larger.  The ladder is right for a reason it never
states; the misspecified global fit is conservative, not wrong.

For fixed v the best c and d follow in closed form and the profile likelihood is
a concentration

    R_i(v) = | mean_n exp(2 pi i (e_i(n) - n v / 2 - (-1)^n d_i)) |,

which is a Lomb-Scargle periodogram on the circle.  Maximising R uses every n
at once, and its peak sharpens like 1/(baseline * sqrt(N)) rather than like the
endpoint separation alone.

ALIAS.  R is quasi-periodic in v with peaks spaced ~1/(max baseline in k-steps)
apart, so the peak must be picked inside a window supplied by a coarser
estimate.  The window used here is +-3 sigma of the ladder's rung-1 value, which
for both configurations is narrower than half the peak spacing -- the code
asserts this rather than assuming it, and refuses the fit if it fails.

ERROR.  By RESIDUAL bootstrap: the n values are held fixed and the fitted
residuals are resampled onto the model prediction.

Resampling n itself -- the obvious bootstrap -- is wrong here, and wrong in a
way that is easy to miss because it errs conservatively.  This estimator's
precision comes from the LEVER ARM, so a resample that happens to omit the
extreme n collapses the baseline from 49 k-steps to a few and widens the peak
enormously.  Averaging over such resamples reported A's error as +-0.0020 when
the endpoint ladder, using less data, got +-0.00014.  An estimator cannot lose
precision by being given more data; the design must be held fixed and only the
noise resampled.

Run:  python3 shift_global.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from shift_ladder import gather, advance, climb   # noqa: E402

NBOOT = 4000
RNG = np.random.default_rng(20261020)


def _conc(ph, par):
    """Concentration of ph after removing the best per-parity offset.

    Profiling out c and d is the same as profiling out one offset per parity
    class, so each class is centred on its own circular mean and the pooled
    resultant is what remains.  ph may be (N,) or (N, G).
    """
    ph = np.atleast_2d(ph.T).T
    out = np.zeros(ph.shape[1])
    n = 0
    for p in (0, 1):
        m = par == p
        if not m.any():
            continue
        out += np.abs(np.exp(2j * np.pi * ph[m]).sum(axis=0))
        n += int(m.sum())
    return out / max(n, 1)


def periodogram(ns, es, v):
    """R_i(v): concentration with c and the parity offset d profiled out."""
    par = ns % 2
    return np.array([_conc(es[:, i] - ns / 2.0 * v[i], par)[0]
                     for i in range(3)])


def peak(ns, es, v0, half, ngrid=20001, nref=3):
    """Maximise R componentwise in [v0 - half, v0 + half], then refine."""
    par = ns % 2
    out = np.empty(3)
    for i in range(3):
        lo, hi = v0[i] - half[i], v0[i] + half[i]
        for _ in range(nref):
            grid = np.linspace(lo, hi, ngrid)
            ph = es[:, i][:, None] - np.outer(ns / 2.0, grid)
            R = _conc(ph, par)
            k = int(np.argmax(R))
            step = grid[1] - grid[0]
            lo, hi = grid[k] - 2 * step, grid[k] + 2 * step
        out[i] = 0.5 * (lo + hi)
    return out


def spacing(ns):
    """Peak spacing of the periodogram, in v: 1 / (baseline in k-steps)."""
    return 2.0 / (ns.max() - ns.min())


def fit(files, name, verbose=True):
    a = gather(files)
    ns = a[:, 0].astype(int)
    es = a[:, 2:]
    if verbose:
        print(f"\n{name}: n = {ns.min()}..{ns.max()}, {len(ns)} points, "
              f"baseline {(ns.max()-ns.min())/2:.1f} k-steps")

    # --- coarse prior: the ladder's first two rungs (alias-safe) ---
    v, se, _ = advance(ns, es, 2)
    v, se = climb(v, se, ns, es, 14, "  prior (7 k-steps)", verbose=verbose)

    sp = spacing(ns)
    half = 3 * se
    if half.max() >= sp / 2:
        print(f"  REFUSED: prior window +-{half.max():.5f} is not inside half "
              f"the peak spacing {sp/2:.5f}")
        return None, None, ns, es
    if verbose:
        print(f"  periodogram peak spacing {sp:.5f}; "
              f"searching +-{half.max():.5f} around the prior")

    vhat = peak(ns, es, v, half)

    # --- residual bootstrap: n held FIXED, residuals resampled ---
    N = len(ns)
    par = ns % 2
    resid = np.empty((N, 3))
    off = np.zeros((2, 3))                      # per-parity offset = c +- d
    for i in range(3):
        ph = es[:, i] - ns / 2.0 * vhat[i]
        for p in (0, 1):
            m = par == p
            if not m.any():
                continue
            off[p, i] = np.angle(np.exp(2j * np.pi * ph[m]).mean()) / (2 * np.pi)
            resid[m, i] = ((ph[m] - off[p, i] + 0.5) % 1.0) - 0.5

    boots = np.empty((NBOOT, 3))
    for b in range(NBOOT):
        eb = np.empty((N, 3))
        for i in range(3):
            eb[:, i] = (off[par, i] + ns / 2.0 * vhat[i]
                        + resid[RNG.integers(0, N, N), i]) % 1.0
        boots[b] = peak(ns, eb, vhat, np.full(3, sp / 6),
                        ngrid=2001, nref=2)
    sev = boots.std(axis=0)

    if verbose:
        R = periodogram(ns, es, vhat)
        d = (off[0] - off[1]) / 2.0
        print(f"  global fit (with the parity term):")
        for i in range(3):
            print(f"    v{i+1} = {vhat[i]:.6f} +- {sev[i]:.6f}   "
                  f"(R = {R[i]:.4f}, resid sd {resid[:,i].std():.4f}, "
                  f"parity d = {d[i]:+.4f})")
            print(f"           ladder gave {v[i]:.5f} +- {se[i]:.5f}")
    return vhat, sev, ns, es


CURVE = {
    "configuration A": np.array([0.2855890797123944,
                                 0.4154709760294156,
                                 0.2853120189148280]),
    "configuration C": np.array([0.2658531592890836,
                                 0.3971450056522778,
                                 0.2748041555597199]),
}

if __name__ == "__main__":
    res = {}
    res["configuration A"] = fit(['run_shift2.log', 'run_shift_far.log',
                                  'run_shift_traj.log'], "configuration A")
    res["configuration C"] = fit(['run_shift_c.log', 'run_c_far.log'],
                                 "configuration C")

    print("\n" + "=" * 70)
    print("  measured against the spectral curve  (v = -2 u(0_+))")
    print("=" * 70)
    for name, (vh, sev, _, _) in res.items():
        if vh is None:
            continue
        pred = CURVE[name]
        print(f"\n  {name}")
        chi2 = 0.0
        for i in range(3):
            d = ((vh[i] - pred[i] + 0.5) % 1.0) - 0.5
            z = d / sev[i]
            chi2 += z * z
            print(f"    v{i+1}: measured {vh[i]:.6f} +- {sev[i]:.6f}   "
                  f"curve {pred[i]:.6f}   {abs(z):.1f} sigma")
        print(f"    chi^2 = {chi2:.1f} on 3 dof")

    if all(r[0] is not None for r in res.values()):
        vA, seA, _, _ = res["configuration A"]
        vC, seC, _, _ = res["configuration C"]
        print("\n" + "=" * 70)
        print("  does v move with the weights?")
        print("=" * 70)
        pA, pC = CURVE["configuration A"], CURVE["configuration C"]
        for i in range(3):
            d = ((vA[i] - vC[i] + 0.5) % 1.0) - 0.5
            s = np.hypot(seA[i], seC[i])
            print(f"    v{i+1}: measured difference {d:+.6f} +- {s:.6f} "
                  f"= {abs(d)/s:.1f} sigma   (curve says {pA[i]-pC[i]:+.6f})")
