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
- `sample_discrete.py` — **the same measurement by sampling at n = 192**, past
  the determinant method's ceiling (NOTES.md §3.4)

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

## Contour formula (incomplete — see NOTES.md §3.4)
- `exactinv.py`, `fastinv.py`, `gasinv.py` — reproduces the matrix inverse to
  `10⁻¹⁴` near the centre at cost independent of `n`, but has an unresolved
  systematic error away from it. Nothing in NOTES.md depends on these.
