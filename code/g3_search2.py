"""
The weight search, using the GROUND TRUTH for facet health instead of a proxy.

THE PREMISE THAT WAS WRONG.  g3_asym2.py checks facet sizes "afterwards by a
short sampling run, which is the ground truth anyway", and I read that as a cost
problem -- each trial needing minutes to hours -- and went looking for a cheap
curve-side proxy.  Two attempts, both failures with clear causes:

  * amoeba holes (hole_tune2): a bounded complement component says the gas PHASE
    exists in the infinite model, not that its facet lands inside a finite
    diamond.  False positive on the first new configuration tried (config D:
    three components resolved, facet 1 empty by sampling).
  * exact finite-n facet sizes from K^{-1} (facet_exact.py): correct in
    principle and validated against g3_sample.exact_edge_probs to 5e-15, but
    double precision only survives to n = 40 -- sum_e p_e must equal |B|, and
    the relative error runs 1e-15, 2e-13, 2e-9, 9e-4 at n = 16, 32, 40, 48,
    with probabilities exceeding 1 beyond that (cond(K) ~ 10^{0.252 n}).  Exact
    equilibration buys a factor of ~70 at n = 40 and does not extend the range.
    And at n <= 40 the facets have not developed: EVERY configuration including
    A has an empty facet there, so the accurate regime and the informative
    regime do not overlap.

Then I measured what I had assumed.  Facet health by sampling at n = 96 costs
**9 to 17 seconds** -- the descent dominates and the height profile itself is
2-3 s -- against ~19 s for the period matrix.  The ground truth was never
expensive.  No proxy is needed, and none of the above was necessary.

THE SEARCH.  Two stages, cheap constraint first:

    stage 1  period_of        ~19 s   reject unless asym >= ASYM_MIN and
                                      min diag >= MINDIAG
    stage 2  sampled facets   ~15 s   require all three facets >= MINFACET

Most proposals fail stage 1, so the average trial is ~20 s and a few thousand
trials is an overnight job.  Stage 2 uses g3_erode.facet_masks on a sampled
height profile -- the same detector and the same observable as the measurements
in §5, so a configuration that passes here is one that can actually be measured.

Why ASYM_MIN = 0.02 rather than "as large as possible": the samples needed to
separate the true period matrix from the one with b12 and b23 interchanged are
~11,000-28,400 at configuration B's 4.6% against 25-76 MILLION at A's 0.11%, and
the runs already use K = 20,000.  Maximising the asymmetry is what starves a
facet; 2% is enough and leaves room for healthy facets.

RUN (hours; every improvement is printed and saved, so it is safe to stop):

    cd /home/claude
    nohup python3 g3_search2.py 1 3000 > run_search2.log 2>&1 &
    tail -f run_search2.log

Different first argument = different seed; run several in parallel. Output:
g3e_weights.npy / g3e_ImB.npy on each improvement.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ASYM_MIN = 0.020
MINDIAG = 0.50
MINFACET = 120          # faces per facet at n = 96, erode 4; A gives ~315-375
NPROF = 80
NSIZE = 96


def stage1(WU, WV):
    from g3_asym2 import period_of
    r = period_of(WU, WV, dps=25)
    if r is None or min(r["diag"]) < MINDIAG or r["asym"] < ASYM_MIN:
        return None
    return r


def stage2(WU, WV, n=NSIZE, Kp=NPROF, seed=7):
    """Real facet sizes, same detector and observable as the §5 measurements."""
    import g3_sample as GS
    from g3_erode import facet_masks
    from g3_measure import height_profile, tree_arrays
    from spatial_avg import bfs_tree, face_graph
    levels, idx, E = GS.build_levels(n, WU, WV)
    _, _, _, _, adj = face_graph(n)
    root, parent, order = bfs_tree(adj)
    pidx, pe, ps, pp = tree_arrays(order, parent)
    Hp = height_profile(n, levels, E, order, pidx, pe, ps, pp, Kp, seed)
    V, Mn = Hp.var(0), Hp.mean(0)
    m = facet_masks(order, pidx, V, Mn, (4,))
    return [len(m[(k, 4)]) for k in range(3)]


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    rng = np.random.default_rng(seed)

    W = np.load(os.path.join(HERE, 'g3_best_weights.npy'))
    walk = (W[0].tolist(), W[1].tolist())
    r0 = stage1(*walk)
    f0 = stage2(*walk)
    print(f"seed {seed}: configuration A -- asym "
          f"{r0['asym'] if r0 else float('nan'):.4f}, facets {f0}")
    print(f"target: asym >= {ASYM_MIN}, min diag >= {MINDIAG}, "
          f"all facets >= {MINFACET} at n = {NSIZE}\n", flush=True)

    best = None
    n1 = n2 = 0
    t0 = time.time()
    for t in range(steps):
        scale = 0.10 if t % 4 else 0.22
        U = [[x * float(np.exp(rng.normal(0, scale))) for x in q]
             for q in walk[0]]
        V_ = [[x * float(np.exp(rng.normal(0, scale))) for x in q]
              for q in walk[1]]
        n1 += 1
        r = stage1(U, V_)
        if r is None:
            continue
        n2 += 1
        try:
            fs = stage2(U, V_)
        except Exception as exc:
            print(f"  step {t}: stage 2 failed: {exc}", flush=True)
            continue
        score = min(fs)
        print(f"  step {t:5d}  asym {r['asym']:.4f}  diag "
              f"{[round(x,3) for x in r['diag']]}  facets {fs}", flush=True)
        if score >= MINFACET and (best is None or score > best[0]):
            best = (score, U, V_, r, fs)
            np.save(os.path.join(HERE, 'g3e_weights.npy'),
                    np.array([U, V_], dtype=float))
            np.save(os.path.join(HERE, 'g3e_ImB.npy'),
                    np.array([[float(r['ImB'][i, j]) for j in range(3)]
                              for i in range(3)]))
            print(f"      *** FEASIBLE, min facet {score} -- "
                  f"saved g3e_weights.npy / g3e_ImB.npy", flush=True)
        # drift towards asymmetry while keeping facets alive
        if score >= MINFACET // 2:
            walk = (U, V_)
        if t % 100 == 0 and t:
            print(f"  [step {t}: {n1} proposals, {n2} passed stage 1, "
                  f"{time.time()-t0:.0f}s]", flush=True)

    if best is None:
        print("\nnothing feasible. With the ground-truth facet check this is "
              "evidence about the (2,4) diamond family itself rather than "
              "about the search, and the item should be restated as a property "
              "of the family.")
    else:
        score, U, V_, r, fs = best
        print(f"\nBEST: min facet {score}, facets {fs}, asym {r['asym']:.4f}")
        print(f"  diag {[round(x,5) for x in r['diag']]}")
        print(f"  b12 {r['b12']:.5f}  b23 {r['b23']:.5f}")
        print(f"  WU = {[[round(x,5) for x in q] for q in U]}")
        print(f"  WV = {[[round(x,5) for x in q] for q in V_]}")
        print("\nNext: measure it with g3_measure_B.py pointed at "
              "g3e_weights.npy / g3e_ImB.npy (K >= 40000).")
