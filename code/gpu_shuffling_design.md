# GPU domino shuffling for the doubly periodic Aztec diamond — design

Status: scoping. No GPU required for anything in §§1–5, 7.

## 1. What we are actually measuring

The sampler is not the deliverable. The deliverable is one number per sample:
the height of the **gaseous facet**, whose fluctuation is the discrete component
`Z = X − E[X]` of Corollary 4.17, with `X` discrete Gaussian of scale
`τ = −1/B` and shift `e = ±1/4`.

This fixes the requirements, and they are unusually mild:

* **One scalar per sample.** The discrete component is a single global random
  variable attached to the facet, not a field. We do not need correlations,
  so we do not need many measurements per sample.
* **Independent samples.** Shuffling is an exact sampler, not MCMC. No burn-in,
  no autocorrelation, no thinning.
* **How many samples.** To separate the two candidate values of B at the
  e = 1/4 alignment we must distinguish P(height = 1) = 11.6% from 4.7%.
  With n samples the standard error on a proportion is ~sqrt(p(1−p)/n);
  at n = 300 that is ~1.8pp against a 6.9pp gap, i.e. ~4σ. **n ≈ 500 is ample.**
* **How large.** N must be big enough that the gaseous facet is many lattice
  spacings across and the height fluctuation is genuinely O(1). Literature on
  the two-periodic diamond works at N ~ 200–1000. Target **N ≈ 1000**, with the
  N-dependence checked (see §7) rather than assumed.

500 independent samples at N = 1000 is a small computation. That matters: it
means the GPU is for convenience, not necessity, and the project is not
hardware-gated. The gate is elsewhere — see §4.

## 2. The three phases are all local

One shuffle step AD(n) → AD(n+1):

| phase | rule | parallelism |
|---|---|---|
| destroy | remove S-over-N and E-left-of-W pairs | per-cell, read 2×2 neighbourhood |
| slide | move each domino 1 cell in its direction | per-cell scatter, conflict-free |
| create | fill each empty 2×2 block with an outward pair | per-block, independent coin |

Every phase is a stencil over a 2D array with no sequential dependence within
the phase. This is close to ideal for a GPU: no atomics needed if slide is
written as a gather (each destination cell asks which domino lands on it)
rather than a scatter.

Note the design lesson already paid for in the CPU prototype: **do not derive a
domino's direction from the parity of its position.** The empty blocks alternate
parity class as n grows. Carry the type. On GPU this is 2 bits per cell.

## 3. Data layout

Store the tiling as a dense 2N×2N array of 2-bit codes (empty/N/S/E/W needs 3
bits, or 2 bits + an occupancy mask). At N = 1000 that is 4·10^6 cells ≈ 1–2 MB.
Negligible. Many independent samples fit in memory simultaneously, so the outer
loop over the 500 samples is itself a parallel axis — batch them and the
occupancy problem disappears entirely.

Work per sample: sum over n ≤ N of O(n²) = **O(N³) ≈ 10^9 cell-updates** at
N = 1000. Times 500 samples = 5·10^11. That is minutes on a GPU, hours on a
CPU. Tractable either way.

## 4. The real bottleneck: the weight recursion

Uniform weights make every creation a fair coin. Two-periodic weights do not.
The creation probability in a given block at level k is determined by a **weight
array at level k**, and those arrays come from iterating urban renewal (the
spider move) *downward* from the order-N weights to order 1 — Propp's
generalized shuffling, `math/0111034`.

The awkward part is the direction mismatch: **weights are generated top-down,
consumed bottom-up.** Three options:

1. **Store all levels.** Memory = sum of O(k²) over k ≤ N = O(N³/3).
   At N = 1000, ~3.3·10^8 floats ≈ 1.3 GB (fp32) — fits on a 24 GB card.
   At N = 2000, ~10.7 GB — tight. At N = 4000, ~85 GB — dead.
2. **Recompute.** Regenerating the level-k array on demand costs another
   downward pass; doing that per level is O(N⁴) overall. Too slow.
3. **Checkpoint.** Store every m-th level, recompute the gaps.
   Memory O(N³/m), time O(N³) with an m-fold constant. This is the standard
   reverse-mode-autodiff tradeoff and is almost certainly the right answer;
   m ≈ 10–30 buys back an order of magnitude of memory cheaply.

**Open question to settle before writing any GPU code:** whether the
two-periodic weights admit a closed form under iterated urban renewal. Under a
two-periodic initial condition the arrays do not stay two-periodic, but they
plausibly retain enough structure (a periodic pattern modulated by
position-dependent scalars) that level k is computable in O(1) per block from
formulas rather than from a stored array. If so, option 1–3 all evaporate and
the memory problem disappears. **I do not know whether this is the case** — it
needs to be read out of `math/0111034` and the Chhita–Johansson line of work.
This is the single highest-value piece of mathematical scoping remaining,
because it decides the entire memory architecture.

## 5. Numerical precision

Iterated urban renewal on a = 0.7 drives weights across a very wide dynamic
range — this is a known practical failure mode of generalized shuffling, not a
hypothetical one. Mitigations, in order of preference:

* Work in **log space**, storing log-weights and forming creation probabilities
  via a stable logistic `p = 1/(1 + exp(Δ))`. Costs a transcendental per block;
  irrelevant next to memory traffic.
* Renormalize each level (weights only matter up to a common factor per block).
* fp64 where it is cheap; but log-space + renormalization should make fp32
  sufficient, which matters on consumer cards.

The probability `p` is the only quantity whose accuracy actually affects the
sampled distribution, so error analysis should target `p` directly rather than
the intermediate weights.

## 6. GPU shape (for later)

* Batch axis = samples. Grid-stride over cells. One kernel per phase, three
  kernels per level, N levels — 3000 kernel launches per batch, fine.
* Slide as gather, not scatter: no atomics, no race.
* RNG: counter-based (Philox) keyed by (sample, level, block) so runs are
  reproducible and independent of thread scheduling. This is worth doing from
  the start; retrofitting reproducibility is painful.
* The weight arrays are read-only per level and shared across the batch —
  broadcast, so the 1.3 GB is paid once, not per sample.

## 7. Validation ladder

Each rung is a real test, in increasing strength. Rungs 1–3 need no GPU.

1. **a = 1 reduces to uniform.** Already passed: valid tilings to N = 60,
   exact tiling counts 2/8/64 at n = 1,2,3, χ² = 1.2/4.6/67.6 on 1/7/63 dof.
2. **Weighted small-n against exact enumeration.** Brute-force all tilings of
   AD(2), AD(3) with two-periodic weights, compute exact probabilities, compare
   to empirical frequencies. This catches a wrong creation probability, which
   validity checks never will.
3. **Kasteleyn determinant.** The total weighted count from the Kasteleyn matrix
   must match the shuffling normalization. Independent of the sampler's logic.
4. **Arctic curve.** At moderate N, the empirical frozen boundary should match
   the known limit shape for the two-periodic model.
5. **N-dependence of the observable.** The facet height distribution must
   stabilize as N grows. If it drifts, we are measuring finite-size effects,
   not the discrete component.

## 8. REVISED SPEC (supersedes the numbers in Sections 1 and 4)

Everything below was established after the first draft and overrides it.

**Run at a = 0.7, not a = 0.5.** Cost analysis favours a = 0.5 by 3.6x, but the
paper quotes B only at a = 0.7, so a = 0.5 cannot adjudicate the discrepancy --
it can only test whether our method reproduces its own prediction. Worth doing
as method validation; not the decisive run.

**Use e = 0, the alignment Johansson's weights already give.** Earlier I judged
e = 0 unusable (10^4-10^5 samples). That was wrong: it answered a different
question (resolving a third atom to fit b and e jointly). For DISCRIMINATING the
two candidate values of B, e = 0 needs 2031 samples for 5 sigma, versus 542 for
e = 1/4 -- the 6.9x ratio in P(+-1) compensates for the rarer events. So the
Remark 4.18 alignment figure is NOT needed to run the experiment.

**Discriminating observable:** P(facet height +-1).
    ours (B = 0.772738i)  -> 1.658%
    paper (B = 0.521828i) -> 0.242%      ratio 6.9x

**Size.** Gas half-width is ~n/6 at a = 0.5, and roughly 4x narrower at a = 0.7,
so ~n/24. A facet ~50 lattice units across implies n ~ 600 (n divisible by 4).
This is a rough target, not a derived requirement -- the N-dependence drift
curve should decide when we are converged.

**Finite-size inflation is the main threat.** At n = 16 (half-width 4) the
measured side-atom mass was 15x the asymptotic prediction. The N-dependence rung
is therefore mandatory, not optional: measure at several n and confirm the ratio
has stopped drifting before believing any value.

**The exact sampler cannot do this.** n = 600 means V ~ 360,000 and the
Schur-complement sampler is O(V^3) per sample. Generalized shuffling is required.

**Generalized shuffling is now BUILT and VALIDATED** (genshuffle.py, fastshuffle.py),
following Janvresse-de la Rue-Velenik (EJC 13 (2006) #R00), the zero-weight-explicit
version of Propp math/0111034. Two details that must not be guessed:
  * active faces of A_m are those coloured like the INNER BOUNDARY of A_m -- the
    colour ALTERNATES with m (0,1,0,1 for m=1,2,3,4)
  * the weight update TRANSPOSES each opposite pair: alpha->gamma/DP, gamma->alpha/DP,
    beta->delta/DP, delta->beta/DP, with DP = alpha*gamma + beta*delta
  * creation probability = alpha*gamma / (alpha*gamma + beta*delta)
Validation: uniform gives 2/8/64 matchings with chi2 0.9/8.0/65.0 on 1/7/63 dof;
weighted two-periodic at n=4 gives chi2 955.8 on 1023 dof against exact enumeration;
edge frequencies at n=12 match exact K^-1 probabilities with mean |z| = 0.74.

**The torsion shortcut is CONFIRMED NUMERICALLY**: w_m = c * w_{m-4} on all shared
edges, exact to 4e-16, with c = 1.132704 at a = 0.7. Creation probabilities are
scale-invariant ratios so c cancels. Hence descend FOUR times, not N: memory drops
from O(N^3) to O(N^2) and the recursion cost becomes negligible.

**Remaining gap is pure implementation.** Per-sample work is sum_m m^2 = N^3/3 face
updates (7e7 at N=600). Pure Python runs ~800 s/sample at N=600 (measured exponent
3.0-3.7, drifting up from interpreter overhead). Faces at a level are independent,
so vectorizing the face update should give 50-100x and put 2000 samples at N=600 in
the several-hour range ON A CPU. The GPU is a convenience, not a requirement.

## 9. SUPERSEDED: the sampling experiment is not needed

Sections 1-8 design a Monte Carlo experiment (a=0.7, ~2000 samples, N~600).
That is no longer the route. The height difference between the gas facet and the
frozen boundary is a signed count of dimers crossing a dual path, so its
characteristic function is a ratio of Kasteleyn determinants and the distribution
is EXACT with no sampling -- seconds at n=48, valid to n~60 before conditioning
fails. See conditioned.py / lowrank.py.

The shuffler (genshuffle.py, fastshuffle.py, vecshuffle.py) remains built and
validated, and is the only tool that reaches large N; it is also what consumes
the Fock weights in the forward construction (endtoend.py). It is simply not
what answered this question.

## 10. What is unresolved, honestly

* **Height normalization.** SETTLED empirically: the centre height is supported
  on atoms spaced exactly 4 apart in Thurston units, so the paper's integer
  discrete component is our height / 4. Confirmed at n = 16 and n = 24.
* **Closed form for the weights** (§4). RESOLVED by the torsion property: at
  these weights repeated shuffling recovers the initial weights, so the weight
  arrays cycle and there is nothing to store.
* **Which alignment the sampler realizes.** SETTLED empirically: Johansson's
  weights give a symmetric alignment, e = 0. Measured centre-height histograms
  are symmetric (about 6%/88%/6%) at n = 16 and 24. Swapping the weight class or
  the diagonal pair gives only 2 distinct models, both symmetric -- the e = +-1/4
  alignments are not reachable in that parameterization and would need the
  Remark 4.18 alpha/beta/gamma placement. Not blocking, since e = 0 works.

The third item deserves emphasis. The four alignments give shifts 0, 1/4, 1/2,
3/4, and the e = 1/2 alignment is a 50/50 split under *either* value of B — so
if we accidentally implement that one, the experiment cannot distinguish the two
candidates at all. Choosing the alignment is a design decision, not a detail.
