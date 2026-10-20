"""
The weight search with the objective the right way round.

WHAT WENT WRONG BEFORE.  g3_asym.py searched for a configuration with
B12 != B23 by MAXIMISING |B12 - B23| subject to a floor on facet size.  It
reached 4.63% (configuration B), and the notes concluded that 4.6% was not
enough to make the two correlations independently distinguishable.  Both halves
of that are mistaken:

  * 4.6% IS enough.  The samples needed to separate the true period matrix from
    the one with b12 and b23 interchanged, at fixed shift, are ~11,000-28,400
    for configuration B against 25-76 MILLION for configuration A.  The runs
    already use K = 20,000.  The conclusion was drawn without a power
    calculation.
  * Maximising the asymmetry is the wrong objective, because it is what drives
    a facet to zero.  Configuration B's facet 1 is 24 faces at n = 96 and
    vanishes under erode 10 -- so B has the asymmetry but not three healthy
    facets, and cannot be measured.

So the real constraint is facet HEALTH, and the asymmetry only needs to clear
~2%.  This searches for max facet health subject to asym >= ASYM_MIN, which is
the reverse of what was tried.

WHY IT WAS NOT TRIED.  Off the symmetric family there was no cheap measure of
facet health: hole_tune assumes C(z) = A(z) and returns nothing for the
asymmetric configurations, so g3_asym2.py fell back to "facet sizes checked
afterwards by a short sampling run".  That makes each trial cost a sampling run
rather than a second, which is why only a handful of configurations were ever
tried.  hole_tune2.components removes the symmetry assumption, and is calibrated
so that the number of bounded components it resolves tracks the number of
usable facets:

    configuration A   3 components (extents 1.80, 2.72, 2.80)   3 healthy facets
    configuration B   2 components                              facet 1 = 24 faces
    configuration C   1 component                               facets 1 and 3 both weak

TWO-STAGE SCREEN, because the costs differ by 20x:
    stage 1  hole_tune2.components   ~1 s    require 3 components, min extent
    stage 2  g3_asym2.period_of     ~19 s    require asym >= ASYM_MIN, diag floor
Only stage-1 survivors reach stage 2.

The proposal moves are multiplicative (log-normal) so weights stay positive and
the model stays a dimer model, and the walk is seeded from configuration A,
which is the one known to have three healthy facets.

RUN (hours; it prints and saves every improvement, so it is safe to stop at any
time and use what it has found):

    cd /home/claude
    nohup python3 g3_search_healthy.py 1 4000 > run_search_healthy.log 2>&1 &
    tail -f run_search_healthy.log

Different first argument = different seed; run several in parallel.
Output: g3d_weights.npy / g3d_ImB.npy when something beats every prior
configuration, i.e. 3 healthy facets AND asym >= ASYM_MIN.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from hole_tune2 import components              # noqa: E402
from g3_asym2 import period_of                 # noqa: E402

ASYM_MIN = 0.020      # |b12-b23|/(b12+b23); B reached 0.0463, A has 0.0011
MINEXT = 1.20         # smallest bounded-component r2-extent; A's weakest is 1.80
MINDIAG = 0.50        # Im B_ii floor, as in g3_asym2


def stage1(WU, WV):
    """Cheap: three resolvable facets, and how healthy the weakest is."""
    try:
        c = components(WU, WV)
    except Exception:
        return None
    if len(c) != 3:
        return None
    ext = min(d["r2ext"] for d in c)
    if ext < MINEXT:
        return None
    return ext


def stage2(WU, WV):
    """Expensive: the period matrix, its asymmetry and diagonal."""
    r = period_of(WU, WV, dps=25)
    if r is None or min(r["diag"]) < MINDIAG:
        return None
    return r


def score(ext, r):
    """Maximise facet health; the asymmetry is a constraint, not the objective."""
    if r["asym"] < ASYM_MIN:
        return None
    return ext


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    rng = np.random.default_rng(seed)

    W = np.load(os.path.join(HERE, 'g3_best_weights.npy'))
    cur = (W[0].tolist(), W[1].tolist())
    e0 = stage1(*cur)
    r0 = stage2(*cur)
    print(f"seed {seed}: starting from configuration A -- "
          f"min extent {e0}, asym {r0['asym']:.4f}, diag "
          f"{[round(x,3) for x in r0['diag']]}", flush=True)
    print(f"target: 3 components, min extent >= {MINEXT}, "
          f"asym >= {ASYM_MIN}, min diag >= {MINDIAG}\n", flush=True)

    best = None                      # best FEASIBLE (asym >= ASYM_MIN)
    walk = cur                       # current point of the walk
    walk_ext = e0
    n1 = n2 = nfeas = 0
    t0 = time.time()
    for t in range(steps):
        scale = 0.10 if t % 4 else 0.25
        U = [[x * float(np.exp(rng.normal(0, scale))) for x in q]
             for q in walk[0]]
        V = [[x * float(np.exp(rng.normal(0, scale))) for x in q]
             for q in walk[1]]
        ext = stage1(U, V)
        n1 += 1
        if ext is None:
            continue
        # accept the walk on facet health alone, so it can drift towards
        # asymmetry without first losing its facets
        if ext >= 0.9 * walk_ext:
            walk, walk_ext = (U, V), max(ext, walk_ext * 0.98)
        r = stage2(U, V)
        n2 += 1
        if r is None:
            continue
        s = score(ext, r)
        if s is None:
            continue
        nfeas += 1
        if best is None or s > best[0]:
            best = (s, U, V, r)
            print(f"  step {t:5d}  FEASIBLE  min extent {ext:.3f}"
                  f"  asym {r['asym']:.4f}"
                  f"  b12 {r['b12']:.5f} b23 {r['b23']:.5f}"
                  f"  diag {[round(x,3) for x in r['diag']]}", flush=True)
            np.save(os.path.join(HERE, 'g3d_weights.npy'),
                    np.array([U, V], dtype=float))
            np.save(os.path.join(HERE, 'g3d_ImB.npy'),
                    np.array([[float(r['ImB'][i, j]) for j in range(3)]
                              for i in range(3)]))
            print("     saved g3d_weights.npy / g3d_ImB.npy", flush=True)
        if t % 200 == 0 and t:
            print(f"  [step {t}: {n1} stage-1 trials, {n2} reached stage 2, "
                  f"{nfeas} feasible, {time.time()-t0:.0f}s]", flush=True)

    if best is None:
        print(f"\nnothing feasible in {steps} steps "
              f"({n2} period matrices computed).")
        print("That is itself informative: it is evidence the (2,4) diamond "
              "family may not contain a configuration with three healthy "
              "facets AND a 2% off-diagonal asymmetry, which would mean the "
              "item should be retired as a property of the family rather than "
              "left open as a search that has not finished.")
    else:
        s, U, V, r = best
        print(f"\nBEST: min extent {s:.3f}, asym {r['asym']:.4f}")
        print(f"  diag {[round(x,5) for x in r['diag']]}")
        print(f"  b12 {r['b12']:.5f}  b23 {r['b23']:.5f}")
        print(f"  WU = {[[round(x,5) for x in q] for q in U]}")
        print(f"  WV = {[[round(x,5) for x in q] for q in V]}")
        print("\nNext: measure it with g3_measure_B.py pointed at "
              "g3d_weights.npy / g3d_ImB.npy (K >= 40000).")
