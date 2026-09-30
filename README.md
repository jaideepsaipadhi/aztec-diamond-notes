# Doubly periodic Aztec diamond: period computation and the discrete component

Numerics accompanying a reading of Berggren–Nicoletti, *Geometry of the doubly
periodic Aztec dimer model* ([arXiv:2502.07241](https://arxiv.org/abs/2502.07241)).

**[NOTES.md](NOTES.md) is the write-up.** [code/README.md](code/README.md) indexes
the source.

> **Correction, 30 September 2026.** An earlier version reported a factor of 2
> between Theorem 1.2 and the dimer model. That was wrong — it came from measuring
> the height at a *single* face of the gas facet instead of a spatial average over
> it, which adds the local gas fluctuation and imitates a discrete Gaussian of
> scale `2B`. Theorem 1.2 is correct as stated. NOTES.md §3.3 keeps the record of
> the error; §3.6 has the corrected measurement. The period discrepancy is
> unaffected.

## What is here

1. **The Section 4.6 period.** For the genus-1 example at `a = 0.7` five
   independent routes give `B = 0.772737661704 i`, against `0.521828 i` in the
   text. Two of them use no elliptic function theory, and one makes no reference
   to the spectral curve at all.

2. **Theorem 1.2 and Remark 4.18 confirmed at genus 1.** Measured through a
   spatial average over the gas facet, the discrete component is a discrete
   Gaussian of scale `B` with shift `e = 1/4`, agreeing to 0.03σ on the
   discriminating ratio `P(−1)/P(0)` — the one the old observable got wrong by a
   factor of 58. The four alignments of Remark 4.18 give measured shifts
   `{1/4, 3/4, 1/2, 0}`.

3. **Theorem 1.2 confirmed at genus 3**, off-diagonal entries of the period
   matrix included — these predict correlations *between* gas facets and have no
   genus-1 analogue. Twelve atoms over three decades of probability, `χ² = 3.8`
   on 7 dof with only the shift vector fitted, and a period matrix from different
   weights rejected by `Δ(−log L) = 208`.

4. **A limit-shape pipeline from Schottky data**, validated end to end: amoeba
   and polygon maps, Ronkin function, surface tension, variational solve, and an
   external check against sampling agreeing to `8 × 10⁻⁴`.

Findings 1 and 2 are independent of each other in both directions.

## Reproducing

```bash
pip install -r requirements.txt

python code/period_g1.py        # the period, three ways
python code/schottky_theta.py   # the theta route (31-digit agreement)
python code/conditioned.py      # exact height distribution
python code/ronkin.py           # det Hess rho = 1/pi^2
python code/fock_correct.py     # Fock weights, self-test to 1e-16
python code/arctic.py           # limit shape and arctic curves
python code/spatial_avg.py      # the discrete component, correct observable
python code/align_avg.py        # the four alignments of Remark 4.18
python code/hyper_g3.py         # genus-3 period matrix, with its self-checks
```

`code/test_vs_jtem.py` additionally needs a JDK and a clone of
[FockDimerSimulation](https://github.com/nikolaibobenko/FockDimerSimulation);
see the header of that file.

## Caveats

Stated at length in NOTES.md §6. The one worth repeating here: every internal
consistency check passed while the observable behind the retracted factor of 2
was wrong, and the two computational routes agreed with each other only because
they shared that observable. Two methods agreeing is not independence when they
compute the same wrong thing.

Also: the genus-3 shift vector is fitted rather than predicted; the genus-2 limit
shape is not validated against sampling; and the contour-integral route in
`code/exactinv.py` is incomplete — nothing here depends on it.

Two files are kept only so the history is legible, and are **wrong**:
`code/fockweights.py` (superseded by `code/fock_correct.py`, NOTES.md §4) and the
observable in `code/sample_discrete.py` (superseded by `code/spatial_avg.py`,
NOTES.md §3.3).
