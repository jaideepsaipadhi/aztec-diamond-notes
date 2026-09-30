# Code index

Python 3 with numpy/scipy/mpmath. `test_vs_jtem.py` additionally needs a JDK and
a clone of `github.com/nikolaibobenko/FockDimerSimulation`.

## The period (§2 of NOTES.md)
- `period_g1.py` — `B` by Legendre, direct quadrature and AGM; the discriminant check
- `period_g1_notes.md` — conventions, and what was excluded
- `schottky_theta.py` — `B` via the Schottky theta series (31-digit agreement)
- `schottky.py` — the conformal-modulus route (~10%, independent but coarse)

## Exact height distribution (§3)
- `twoperiodic.py` — Johansson (7.2) weights; validated against enumeration
- `lowrank.py` — the matrix-determinant-lemma reduction
- `conditioned.py` — the pmf, plus `ratio_checked` (cross-`n` convergence test)
- `ndep.py` — dual path with verified zero circulation; `n`-dependence
- `height.py` — Thurston height; zero violated edges, zero face circulation
- `exactdist.py`, `alignments.py`, `gasvar.py` — supporting checks
- `gasgibbs.py` — full-plane gas Gibbs measure from the spectral curve
- `sample_discrete.py` — the same measurement by sampling at n = 192, past the
  determinant method's ceiling (NOTES.md §3.4). **The observable is wrong** — it
  reads the height at one face, which is the discrete component plus the local
  gas fluctuation, and that is where the retracted factor of 2 came from. Kept
  because the sampling machinery is correct and still used.

## The discrete component, done right (§3.6)
- `spatial_avg.py` — **the correct observable**: the spatial average of the
  height over the gas facet, as one precomputed dot product per sample. Verified
  against a direct BFS height on every run
- `align_avg.py` — the four alignments of Remark 4.18; measured shifts
  `{1/4, 3/4, 1/2, 0}`
- `half_scaling.py` — finite-`n` behaviour of the `e = 1/2` alignment, where two
  atoms must carry exactly equal mass for any `b`

## Samplers
- `genshuffle.py` — generalised shuffling (Janvresse–de la Rue–Velenik)
- `fastshuffle.py`, `vecshuffle.py` — torsion shortcut; vectorised (233×)
- `exact_sampler.py`, `fast_sampler.py` — Schur-complement determinantal sampler
- `shuffle2.py`, `oracle.py` — uniform case and brute-force oracle

## Schottky / higher genus
- `schottky_full.py` — period matrices by Poincaré series over double cosets
- `test_vs_jtem.py`, `Reference.java` — regression fixture against jtem
- `poincare_map.py` — map of where the series is usable
- `genus2.py`, `genus2_amoeba.py`, `sigma_g2.py` — genus-2 curve, amoeba, sigma
- `genus2_fock.py` — genus-2 Fock face weights via BBS (10); the prime form's
  spinors cancel in the face weight, leaving only theta with odd characteristic
- `bn_model.py` — the general `k × l` model of [BN] Definition 2.1

## Limit shapes (§4)
- `amoeba.py` — amoeba and polygon maps
- `ronkin.py` — Ronkin function; the `det Hess ρ = 1/π²` check
- `tension.py` — surface tension by Legendre duality
- `tabulate.py`, `slope_sample.py` — σ tabulation (slope-space sampling is 120× better)
- `variational.py`, `sigma_pl.py` — earlier solver attempts, kept for the record
- `lp_solve.py` — the LP formulation (exact energy)
- `smooth_lp.py` — log-sum-exp smoothing (unique minimiser)
- `limit_shape.py`, `arctic.py` — phases and arctic curves

## Fock weights and the external check
- `fock_correct.py` — **the correct construction**; self-test reproduces the four
  genus-1 face-weight identities to `10⁻¹⁶`
- `fock_aztec.py` — edge weights on the Aztec diamond
- `fastfock.py` — θ-lookup version, 14× faster, `1.4×10⁻⁷`
- `fock_sample.py` — sampling and facet-slope measurement
- `endtoend.py`, `fockweights.py` — superseded; `fockweights.face_weight` is
  **wrong** (see NOTES.md §4) and is kept only so the history is legible

## Genus 3: the period matrix and its off-diagonal entries (§5)
- `periodic_dimer.py` — a periodic Aztec diamond from its OWN Kasteleyn matrix:
  in diagonal coordinates the graph is the square lattice, so `P(z,w) = det K`,
  the Newton polygon and the amoeba follow directly. No `α/β/γ` dictionary
- `hyper_curve.py` — the hyperelliptic curve `y² = D(z)` and its real branch
  points, with the quadratic-in-`w` coefficients recovered at full precision
- `hyper_g3.py` — **the period matrix**, with the three errors genus 1 cannot
  detect fixed and documented: `ℬ𝒜⁻¹` (not `𝒜⁻¹ℬ`), the alternating branch
  phases, and `B − Bᵀ` integral rather than zero. Self-checks: homology
  relations, symmetry, `Im B ≻ 0`, and the genus-1 value to `2×10⁻¹³`
- `g3_sample.py` — sampler for `(2,4)`-periodic weights (full descent; the
  two-periodic torsion shortcut does not apply). Validated against exact
  Kasteleyn edge probabilities
- `g3_measure.py` — facets located from the five slope plateaus of the mean
  height; the joint distribution of the three facet shifts
- `g3_erode.py` — erosion sweep of the facet mask, the valid mask-contamination
  test (conditioning on rounding residual is not — it selects configurations)
- `g3_swap.py` — **the falsification test**: a period matrix from different
  weights, given all six labellings and its own free shifts, rejected by
  `Δ(−log L) = 208`
- `g3_tune2.py`, `g3_asym2.py`, `hole_tune.py` — weight selection against the
  tension between facet size and measurable atoms
- `amoeba_holes.py` — symmetry-free hole finder. `hole_tune.py` locates holes by
  distance from `r₂ = 0`, valid only when `C = A`, i.e. only for the symmetric
  weight family; breaking that symmetry makes it report zero holes for a model
  that has three
- `g3_best_weights.npy`, `g3_best_ImB.npy`, `g3c_*.npy` — the two configurations

## Contour formula (incomplete — see NOTES.md §3.4)
- `exactinv.py`, `fastinv.py`, `gasinv.py` — reproduces the matrix inverse to
  `10⁻¹⁴` near the centre at cost independent of `n`, but has an unresolved
  systematic error away from it. Nothing in NOTES.md depends on these.
