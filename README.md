# Two-periodic Aztec diamond: period computation and the discrete component

Numerics accompanying a reading of Berggren–Nicoletti, *Geometry of the doubly
periodic Aztec dimer model* ([arXiv:2502.07241](https://arxiv.org/abs/2502.07241)).

**[NOTES.md](NOTES.md) is the write-up.** [code/README.md](code/README.md) indexes
the source.

## What is here

1. **The Section 4.6 period.** For the genus-1 example at `a = 0.7` four
   independent routes give `B = 0.772737661704 i`, against `0.521828 i` in the
   text. One of the routes uses no elliptic function theory.

2. **A factor of 2 between Theorem 1.2 and the dimer model.** Measuring the
   discrete component from exact Kasteleyn determinant ratios — no Monte Carlo —
   the height fluctuation is a discrete Gaussian of scale `2B`, not `B`. The
   functional form of `B(a)` is confirmed to 6–7 digits across `a ∈ [0.3, 0.8]`,
   and independently across `a ∈ [0.4, 0.7]` by sampling — a route with no
   Kasteleyn matrix anywhere, and no size ceiling. Only the overall factor disagrees. I could
   not locate where it enters, and §3.3 of the notes lists the explanations I
   excluded.

3. **A limit-shape pipeline from Schottky data**, validated end to end: amoeba
   and polygon maps, Ronkin function, surface tension, variational solve, and an
   external check against sampling agreeing to `8 × 10⁻⁴`.

Findings 1 and 2 are independent; 2 does not rest on 1.

## Reproducing

```bash
pip install -r requirements.txt

python code/period_g1.py        # the period, three ways
python code/schottky_theta.py   # the theta route (31-digit agreement)
python code/conditioned.py      # exact height distribution
python code/ronkin.py           # det Hess rho = 1/pi^2
python code/fock_correct.py     # Fock weights, self-test to 1e-16
python code/arctic.py           # limit shape and arctic curves
python code/sample_discrete.py  # the same measurement by sampling, no matrix
```

`code/test_vs_jtem.py` additionally needs a JDK and a clone of
[FockDimerSimulation](https://github.com/nikolaibobenko/FockDimerSimulation);
see the header of that file.

## Caveats

Stated at length in NOTES.md §5. In short: the `a = 0.7` row is only 0.8%
converged by the determinant method, and its independent confirmation by
sampling carries a ±0.0036 statistical error (1.6σ) rather than six digits; the
genus-2 limit shape is not validated against sampling; and the contour-integral
route in `code/exactinv.py` is incomplete — nothing here depends on it.

`code/fockweights.py` is **wrong** and is kept only so the history is legible —
it is superseded by `code/fock_correct.py`. See NOTES.md §4.
