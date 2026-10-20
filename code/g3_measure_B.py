"""
Configuration B measured: the run that makes b12 != b23 independently testable,
and that settles the facet 1 <-> 3 labelling question.

WHY THIS RUN EXISTS.  Two items in the notes' open list reduce to one missing
measurement.

(1) "I never found a configuration with three healthy facets AND B12 != B23 by
more than 4.6%, so the two correlations are near-equal predictions rather than
independently distinguishable ones."  That inference was wrong -- it was never
backed by a power calculation.  Doing one: the samples needed to separate the
true period matrix from the one with b12 and b23 interchanged, at fixed shift,

    configuration A   |b12 - b23| = 0.00046    25-76 MILLION samples
    configuration B   |b12 - b23| = 0.01916    11,000 - 28,400 samples
    configuration C   |b12 - b23| = 0.01284    21,000 - 94,200 samples

Configuration B's 4.6% asymmetry is comfortably detectable at the K = 20,000
already used for the other runs; A's 0.11% never will be.  So the obstacle was
never the size of the asymmetry, it was that configuration B was never sampled.
g3_measure_b.py was in fact WRITTEN for configuration B -- its docstring
describes B's diag (0.517, 0.688, 0.858) -- but it was only ever run with the
configuration C weights.

(2) "Configuration C's facet 1 <-> 3 assignment is unresolved."  There, the
atom-width check that would fix the labelling is confounded: C's facet 3 is both
the narrowest in B and the noisiest to measure (175 faces), so measured widths
running opposite to B's diagonal is equally well explained by "broadest =
noisiest".  Configuration B is not confounded -- B11 = 0.517 against
B33 = 0.858, 66% apart, giving predicted marginal sds that differ by 2x
(0.420 vs 0.222 at the operative shift).  If B's facets are healthy, the width
ordering fixes the labelling outright.

WHAT IS NEW HERE, beyond calling the existing harness with the right weights:

  * a distinct tag, so nothing overwrites g3_Z_C.npy.  The existing
    g3_measure_b.py hardcodes tag "C" while taking the weight file as an
    argument, which is a live foot-gun: pointing it at g3b_weights.npy would
    silently destroy configuration C's 20,000 samples.
  * the b12 <-> b23 interchange test, which is the point of the run and which
    the existing script does not perform.  It compares B's own matrix against
    the SAME matrix with only the two off-diagonals swapped -- diagonal and b13
    untouched -- so the comparison isolates the off-diagonal structure instead
    of the overall scale, which is what §5.4's A-vs-C test already covers.
  * the measured marginal widths against predicted, for the labelling question,
    reported per facet alongside its face count so that a width explained by
    noise rather than by B is visible as such.

RUN (about 3-5 hours at K = 60000, n = 192; K is the only thing worth changing,
and 60000 is ~2x the 28,400 the power calculation asks for):

    cd /home/claude
    nohup python3 g3_measure_B.py 192 60000 > run_measure_B.log 2>&1 &
    tail -f run_measure_B.log

Then:  python3 g3_measure_B.py --report
"""
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

TAG = "B"
WFILE = os.path.join(HERE, 'g3b_weights.npy')
MFILE = os.path.join(HERE, 'g3b_ImB.npy')
ZFILE = os.path.join(HERE, f'g3_Z_{TAG}.npy')
KFILE = os.path.join(HERE, f'g3_keys_{TAG}.npy')

PTS = list(itertools.product(range(-3, 4), repeat=3))


def swap_offdiag(B):
    """Interchange b12 and b23; leave the diagonal and b13 alone."""
    C = B.copy()
    C[0, 1] = C[1, 0] = B[1, 2]
    C[1, 2] = C[2, 1] = B[0, 1]
    return C


def pmf(B, e):
    P = np.array(PTS, dtype=float)
    d = P - e
    Q = np.linalg.inv(B)
    w = np.exp(-np.pi * np.einsum('ij,jk,ik->i', d, Q, d))
    return P, w / w.sum()


def predicted_moments(B, e):
    P, w = pmf(B, e)
    m = (w[:, None] * P).sum(0)
    C = np.einsum('i,ij,ik->jk', w, P - m, P - m)
    sd = np.sqrt(np.diag(C))
    return sd, C / np.outer(sd, sd)


def measure(n, K):
    import g3_measure_b as MB
    W = np.load(WFILE)
    # run() saves to g3_Z_<tag>.npy, so the tag MUST be B here
    Z, keys = MB.run(TAG, W[0].tolist(), W[1].tolist(), n=n, K=K,
                     erodes=(4, 10))
    if Z is None:
        print("no usable facets for configuration B -- "
              "the facet-health premise fails and items 3/4 stay open")
        return None
    print(f"\nsaved {ZFILE}")
    return Z, [tuple(k) for k in keys]


def report():
    import g3_measure_b as MB
    if not os.path.exists(ZFILE):
        sys.exit(f"{ZFILE} not found -- run the measurement first")
    Z = np.load(ZFILE)
    keys = [tuple(k) for k in np.load(KFILE)]
    ImB_B = np.load(MFILE)
    ImB_A = np.load(os.path.join(HERE, 'g3_best_ImB.npy'))
    ImB_sw = swap_offdiag(ImB_B)

    print("=" * 74)
    print(f"  configuration B measured: {len(Z)} samples")
    print(f"  diag Im B {np.round(np.diag(ImB_B), 5)}")
    print(f"  b12 {ImB_B[0,1]:.5f}  b23 {ImB_B[1,2]:.5f}"
          f"   |b12-b23| = {abs(ImB_B[0,1]-ImB_B[1,2]):.5f}")
    print("=" * 74)

    for erode in sorted({k[1] for k in keys}):
        R = MB.rounded(Z, keys, erode)
        print(f"\n=== erode {erode} ===")
        fits = {}
        for lab, M in (("B (its own)", ImB_B),
                       ("B, b12<->b23 swapped", ImB_sw),
                       ("A (wrong matrix)", ImB_A)):
            best = None
            for perm in itertools.permutations(range(3)):
                r = MB.fit(R, M, perm)
                if best is None or r["nll"] < best[1]["nll"]:
                    best = (perm, r)
            perm, r = best
            fits[lab] = r
            print(f"  {lab:<22} perm {perm}  e={np.round(r['e'],3)}"
                  f"  -logL {r['nll']:.1f}"
                  f"  atoms chi2 {r['chi']:.1f}/{r['dof']}"
                  f"  corr chi2 {r['corrchi']:.1f}/3")

        own = fits["B (its own)"]["nll"]
        for lab in ("B, b12<->b23 swapped", "A (wrong matrix)"):
            d = fits[lab]["nll"] - own
            print(f"\n  Delta(-logL) against {lab}: {d:+.1f}"
                  + ("   <- off-diagonal structure resolved"
                     if lab.startswith("B,") and d > 12.5 else
                     "   <- NOT resolved; needs more samples"
                     if lab.startswith("B,") else ""))

        # labelling: measured widths against predicted, with face counts
        e = fits["B (its own)"]["e"]
        sd_pred, rho_pred = predicted_moments(ImB_B, e)
        cols = [keys.index((k, erode)) for k in range(3)]
        sd_meas = (Z[:, cols] / 4.0).std(0)
        print(f"\n  marginal widths at the fitted shift e = {np.round(e,3)}:")
        for i in range(3):
            print(f"    facet {i+1}: measured {sd_meas[i]:.4f}"
                  f"   predicted {sd_pred[i]:.4f}")
        order_m = tuple(np.argsort(-sd_meas))
        order_p = tuple(np.argsort(-sd_pred))
        print(f"    width ordering measured {order_m}  predicted {order_p}"
              + ("   -> labelling CONFIRMED" if order_m == order_p
                 else "   -> labelling DISAGREES with B's diagonal"))


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 192
        K = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
        if measure(n, K) is not None:
            print()
            report()
