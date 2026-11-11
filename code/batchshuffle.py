"""
Generalized domino shuffling, batched over DRAWS.

WHAT THIS CHANGES.  vecshuffle.sample draws one matching: it walks the levels
m = 1..n in Python and does ~15 numpy operations per level on arrays of size
|faces(m)| ~ m^2.  That is ~15n numpy calls per draw -- 2880 at n = 192 -- so
K draws cost 15nK calls, about 58 million at K = 20,000.  At small m those
arrays are tiny, so most of that time is numpy call overhead rather than work.

Batching over draws makes the state (K, nedges) and runs every draw at once.
The level loop is still length n, but now the whole batch moves through it
together, so the call count is 15n TOTAL, independent of K, and the work becomes
memory-bandwidth bound instead of overhead bound.

WHY THIS IS SAFE, and it is the same reason the algorithm vectorizes at all:
every edge of A_m lies on the boundary of exactly one active face of A_m
(vecshuffle's docstring).  So per-face writes land in disjoint edge slots, and
batching adds a draw index that is disjoint by construction -- each (draw, edge)
slot is written by exactly one (draw, face).  No collisions, no atomics, no
reduction.

WHY IT IS ALSO THE GPU PORT.  A per-level elementwise kernel over a
(draws x faces) array with no atomics is the canonical data-parallel pattern;
the CUDA/CuPy version is a transcription of the same six boolean expressions,
with the same index arrays uploaded once.  The point of writing it this way on
the CPU first is that it can be VALIDATED here -- against
g3_sample.exact_edge_probs, which is exact -- so the kernel has a reference
implementation and a test to match rather than being written blind.  No GPU was
available where this was developed, so no GPU timing is claimed.

MEMORY.  nedges = 4 n^2, so the state is 4 K n^2 bytes: at n = 192, K = 1000
that is 147 MB.  Draws are therefore processed in chunks, which is what a GPU
would do too; `chunk` is the batch size, not the number of draws.
"""
import numpy as np


def _levels_order(levels):
    return sorted(levels)


def sample_chunk(levels, nedges, rng, K, order=None, out=None, buf=None):
    """K matchings at once as a (K, nedges) boolean array."""
    if order is None:
        order = _levels_order(levels)
    M = np.zeros((K, nedges), dtype=bool) if out is None else out
    N = np.empty((K, nedges), dtype=bool) if buf is None else buf
    M.fill(False)
    for m in order:
        ib, it, il, ir, p = levels[m]
        Mb = M[:, ib]
        Mt = M[:, it]
        Ml = M[:, il]
        Mr = M[:, ir]
        c = Mb.astype(np.int8)
        c += Mt
        c += Ml
        c += Mr
        one = c == 1
        zero = c == 0
        coin = rng.random((K, len(p))) < p
        N.fill(False)
        N[:, ib] = (one & Mt) | (zero & coin)
        N[:, it] = (one & Mb) | (zero & coin)
        N[:, il] = (one & Mr) | (zero & ~coin)
        N[:, ir] = (one & Ml) | (zero & ~coin)
        M, N = N, M
    return M


def edge_counts(levels, nedges, rng, K, chunk=256):
    """Sum over K draws of the edge indicator: counts[e] = #{draws: e in M}."""
    order = _levels_order(levels)
    counts = np.zeros(nedges, dtype=np.int64)
    out = np.zeros((chunk, nedges), dtype=bool)
    buf = np.empty((chunk, nedges), dtype=bool)
    done = 0
    while done < K:
        k = min(chunk, K - done)
        if k == chunk:
            M = sample_chunk(levels, nedges, rng, k, order, out, buf)
        else:
            M = sample_chunk(levels, nedges, rng, k, order)
        counts += M.sum(axis=0, dtype=np.int64)
        done += k
    return counts


def iter_chunks(levels, nedges, rng, K, chunk=256):
    """Yield (K_i, nedges) boolean blocks covering K draws."""
    order = _levels_order(levels)
    done = 0
    while done < K:
        k = min(chunk, K - done)
        yield sample_chunk(levels, nedges, rng, k, order)
        done += k


if __name__ == "__main__":
    import sys
    import time

    sys.path.insert(0, ".")
    from vecshuffle import build_levels, sample as sample1

    a = 0.7
    print("CORRECTNESS: batched edge frequencies vs the EXACT Kasteleyn values")
    print("(g3_sample.exact_edge_probs; the batched sampler must reproduce it)\n")
    from g3_sample import exact_edge_probs

    for n in (8, 12):
        levels, idx, E = build_levels(n, a)
        ex = exact_edge_probs(n, a, a if False else None) \
            if False else exact_edge_probs(n)
        rng = np.random.default_rng(11)
        K = 20000
        t0 = time.time()
        c = edge_counts(levels, nedges=len(E), rng=rng, K=K, chunk=512)
        f = c / K
        pe = np.array([ex[e] for e in E])
        se = np.sqrt(np.maximum(pe * (1 - pe), 1e-12) / K)
        z = (f - pe) / np.maximum(se, 1e-9)
        print(f"  n={n}: {len(E)} edges, K={K} in {time.time()-t0:.1f}s")
        print(f"     max|f-p| {np.abs(f-pe).max():.4f}   "
              f"max|z| {np.abs(z).max():.2f}   "
              f"|z|>4: {int((np.abs(z)>4).sum())} of {len(E)}")
        print(f"     sum f = {f.sum():.3f}  (exact {pe.sum():.3f})")


# ---------------------------------------------------------------- measured
# MEASURED, and the reason this file is not the answer it was meant to be.
#
# Batched vs single-draw, identical levels, cost per draw:
#
#     n      single      batched   speedup
#    16     0.241 ms    0.037 ms     6.5x
#    24     0.389 ms    0.169 ms     2.3x
#    32     0.736 ms    0.410 ms     1.8x
#    48     1.152 ms    1.196 ms     1.0x
#    64     2.197 ms    2.815 ms     0.8x
#
# So batching over draws removes numpy CALL OVERHEAD, which dominates only
# while the per-level arrays are small.  By n = 48 it buys nothing and by
# n = 64 it costs.  The premise this file was written on -- that the sampler is
# overhead-bound and that vectorising is the hard part of a GPU port -- is
# wrong, and measuring it is what showed that.
#
# What the sampler actually is: bandwidth/element bound, with the per-element
# cost settling at ~16.4 ns (67.3, 21.5, 16.9, 16.4 ns at n = 32, 48, 64, 96)
# over sum_m |faces(m)| ~ n^3/3 elements.  So
#
#     cost per draw  ~  5.5 n^3 nanoseconds
#
#     n = 96:   4.9 ms      n = 128:  12 ms      n = 192:  39 ms
#
# which puts K = 20,000 at n = 192 at 13 minutes on ONE core, and the 68,100
# draws the b12<->b23 swap test needs at 42 minutes.  Nothing in this project
# needs a GPU; the crossover for K = 20,000 in under an hour is near n = 320,
# and at n = 500 it is 3.8 hours.
#
# The earlier impression that sampling was expensive came from timing
# height_profile, where the cost is build_levels (a pure-Python dict descent,
# 5-14 s, once per weight configuration) and the per-draw height accumulation
# (a sequential Python loop over ~37k faces at n = 96, ~800 ns an iteration) --
# not the shuffling.  Those two are where CPU work would pay off.
#
# This file is kept because it is correct and validated, because it is a real
# speedup for n <= 32, and because a (draws x faces) elementwise form with no
# atomics is the right reference implementation to validate a CUDA kernel
# against if a port is ever wanted above n ~ 300.
