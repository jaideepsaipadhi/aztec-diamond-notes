# Two-periodic Aztec diamond: the Section 4.6 period, and a factor of 2 in Theorem 1.2

Jaideep Sai Padhi — notes for M. Nicoletti, September 2026

Everything below refers to Berggren–Nicoletti, *Geometry of the doubly periodic
Aztec dimer model* (arXiv:2502.07241), cited as [BN].

---

## 1. Summary

Three findings, in decreasing order of confidence.

1. **The period `B` in Section 4.6.** For the genus-1 example at `a = 0.7` I get
   `B = 0.772737661704 i`, against `0.521828 i` in the text. Four independent
   routes agree, one of which uses no elliptic function theory at all.

2. **A factor of 2 between Theorem 1.2 and the dimer model.** Measuring the
   discrete component directly from exact Kasteleyn determinants, the height
   fluctuation is a discrete Gaussian whose scale corresponds to `2B`, not `B`.
   The functional form of `B(a)` is confirmed to 6–7 digits across
   `a ∈ [0.3, 0.8]` by exact determinants, and independently at `a = 0.7` by
   sampling at `n = 192`; only the overall factor disagrees.

3. **A validated limit-shape pipeline** from Schottky data through the amoeba
   and Ronkin function to a variational limit shape, checked against sampling
   at `8 × 10⁻⁴`.

Findings 1 and 2 are independent of each other: 2 does not rest on 1.

---

## 2. The period

The spectral curve of (101) is `y² = z(z + a²)(z + a⁻²)`, with branch points
`0, −a², −a⁻², ∞`. The discriminant of (101) read as a quadratic in `w` is

    4(C + 4)(z + a²)(z + a⁻²) / z,      C = 4 + 4a² + 4a⁻²

verified symbolically. (This differs from `z(z+a²)(z+a⁻²)` by `z²`, a square, so
the curve, branch points and periods are the same.)

I used Section 4.6's own conventions throughout: the `A` cycle is the compact
oval over `[−a⁻², −a²]`, the normalisation is `c = 2∫` over that segment, the `B`
cycle is the loop around the cut `[−a², 0]` for `a < 1`, and the one-form is
(106).

At `a = 0.7`:

| route | value |
|---|---|
| Legendre `K′/K` with `k² = 1 − a⁴` | 0.772737661704 |
| direct quadrature of both real periods | 0.772737661704 |
| AGM | 0.772737661704 |
| Schottky theta series (nome inversion) | 0.772737661704 |

The first three are the same theory computed three ways; the fourth is
independent of it. Agreement is to 31 decimal places between the elliptic and
theta routes.

Two possibilities I checked and excluded:

* **Cycle interchange.** Swapping `A` and `B` gives `1.29410 i = 1/0.772738`,
  the modular inversion — not `0.521828 i`.
* **A different `a`.** Solving `B(â) = 0.521828 i` gives `â = 0.4418509`, i.e.
  `â⁴ = 0.0381` against `0.2401`. I could not find a reparametrisation of
  `a = 0.7` producing it.

Code: `period_g1.py`, `schottky_theta.py`. Notes: `period_g1_notes.md`.

---

## 3. The dimer confirmation, and the factor of 2

### 3.1 Method

The height difference between the gas facet and the frozen boundary is a signed
count of dimers crossing a dual path, so its characteristic function is a ratio
of Kasteleyn determinants:

    E[e^{iθΔh}] = det K(θ) / det K(0)

with the crossed edges' entries multiplied by `exp(−4iθ s_e)`. An FFT over `θ`
gives the **exact** pmf — no Monte Carlo error. `K(θ)` is a low-rank update of
`K(0)`, so by the matrix determinant lemma the ratio is an `r × r` determinant
with `r` the path length, from one sparse LU plus `r` solves.

Weights are Johansson's (7.2) (arXiv:1704.06035), validated against brute-force
enumeration: `|det K|` equals the weighted matching sum at `a = 1.0, 0.7, 0.5`,
and at `n = 4` the 1024 tilings carry 5 distinct weights (64/256/384/256/64).

### 3.2 Result

Writing `r = P(±1)/P(0)` for the resulting distribution — which is symmetric —
and inverting `−π/(2 ln r)`:

| a | r (n = 48) | −π/(2 ln r) | K′/K, k² = 1−a⁴ |
|---|---|---|---|
| 0.3 | 0.022545734 | 0.4142167 | 0.4142168 |
| 0.4 | 0.040259422 | 0.4889774 | 0.4889778 |
| 0.5 | 0.063502620 | 0.5698157 | 0.5698411 |
| 0.6 | 0.092936249 | 0.6611537 | 0.6617511 |
| 0.7 | 0.128715886 | 0.7661869 | 0.7727377 |
| 0.8 | 0.167145895 | 0.8780852 | 0.9200278 |

Seven digits at `a = 0.3`, six at `a = 0.4`, degrading at larger `a` in step with
the narrowing gas facet: the facet half-width is roughly `n/6` at `a = 0.5` but
about `n/24` at `a = 0.7`, so `n = 48` is far less converged there. The trend is
consistent in `n` — at `a = 0.7` the inverted value runs 0.743, 0.766, 0.769 for
`n = 24, 48, 60`.

For comparison, `B = 0.521828 i` predicts `r = 0.049283` at `a = 0.7` against
`0.128716` measured. The same convention maps both candidates, so that gap is
not a convention artifact.

### 3.3 The factor of 2

The factor written as 2 above is not free. Defining `c(a) = −π/(ln r · b(a))`,
the measured values at `n = 48` are

    1.999999,  1.999998,  1.999911,  1.998194,  1.983045,  1.908823

at `a = 0.3, 0.4, 0.5, 0.6, 0.7, 0.8` — six to seven digits of exactly 2 where
the finite-size error is smallest, drifting only where convergence is
independently known to be slow. A single constant factor cannot be an error in
`b(a)`.

Confirmed independently by the variance, which goes through the full theta
normalisation rather than a ratio where constants cancel: at `a = 0.4` the
measured `Var[Z] = 0.07453778` against `0.07453798` predicted at scale `2b`
(and `0.00323119` at scale `b`) — seven digits.

**Under the paper's own definitions this is a genuine discrepancy, not a
convention.** Equation (28) defines the discrete Gaussian as

    P_{e,τ}(n) ∝ exp(iπ(n − e)·τ(n − e))

with **no** factor of ½, and Theorem 1.2 gives `τ = −B⁻¹`. For genus 1 with
`B = ib` that predicts `exp(−π/b)`; the measurement is `exp(−π/(2b))`.

Every benign explanation I could construct is excluded:

* **Not the Gaussian convention** — (28) carries no ½.
* **Not a lattice relabelling.** If our atom index were `k = c·n`, matching would
  need `c² = 2`, i.e. `c = √2`. A lattice relabelling must be rational.
  Measured: `c² = 2.000005`.
* **Not the height normalisation.** Definition 2.2 gives increments
  `1_M − 1_{M₀}`, while Thurston's is `1 − 4·1_M`, so `h_Thurston + 4·h_paper`
  is deterministic. Measured over 300 samples at `n = 8`: exactly 8, constant,
  with Thurston atoms at spacing 4 and paper atoms at spacing 1. So our atom
  index **is** `n`.
* **Not the alignment.** The measured distribution is symmetric (asymmetry
  `10⁻¹⁵` at `n = 4`, `10⁻¹²` at `n = 24`), consistent with the shift-0
  alignment of Remark 4.18 — not `e = ±1/4`. Varying `N mod 4` reproduces the
  quasi-periodic dependence: `N ≡ 0` symmetric, `N ≡ 1` gives `P(0) = P(1)`
  exactly (shift ½), `N ≡ 2` asymmetric. The scale is the same at every residue,
  measured on both sublattices, `b_eff/b = 2.00000`.

So there are three numbers and no two agree: the period matrix of the spectral
curve (0.772738 i), the scale the dimer model exhibits (2B = 1.545475 i), and
Section 4.6's value (0.521828 i). I have not been able to resolve the first
against the second, and I suspect it is a convention internal to the proof of
Theorem 1.2 — one candidate is the `A`-cycle normalisation, since a single
integral rather than `2∫` would give `B = 2b` and make `τ = −B⁻¹` reproduce the
measurement exactly. I could not verify that from the text.

Code: `conditioned.py`, `lowrank.py`, `ndep.py`, `twoperiodic.py`.

### 3.4 The same measurement by sampling, at n = 192

The determinant method is capped at `n ≈ 60` (§3.5), which is why the small-`a`
rows above carry the argument. But the discrete component does not actually
require a determinant: it is a signed count of matching edges crossing the dual
path, so it can be read off a sampled configuration directly. Sampling involves
no matrix and has no conditioning problem.

Sampling the two-periodic model at `a = 0.7` with generalised shuffling
(Janvresse–de la Rue–Velenik) at `n = 192` — three times the determinant
ceiling — with 6000 samples gives atoms at

    -49: 0.10383     -48: 0.78517     -47: 0.11067

so `r = 0.13659 ± 0.0036`, against `0.130972` predicted. That is 1.6σ.
Inverting, `b = 0.789 ± 0.010` against `K′/K = 0.772738`, also 1.6σ.

The same measurement across the family, at `n = 96` with 4000 samples each:

| a | r (sampled) | predicted | dev | b inverted | K′/K |
|---|---|---|---|---|---|
| 0.4 | 0.03996 ± 0.00229 | 0.04026 | 0.1σ | 0.4878 ± 0.0087 | 0.488978 |
| 0.5 | 0.06322 ± 0.00292 | 0.06351 | 0.1σ | 0.5689 ± 0.0095 | 0.569841 |
| 0.6 | 0.09084 ± 0.00357 | 0.09314 | 0.6σ | 0.6549 ± 0.0107 | 0.661751 |
| 0.7 | 0.13659 ± 0.00360 | 0.130972 | 1.6σ | 0.789 ± 0.010 | 0.772738 |

So `a = 0.7` is measured directly rather than carried by the functional form.
The two methods are complementary: determinants give exact distributions with no
statistical error but a hard size ceiling; sampling has statistical error but no
ceiling. They agree at the disputed parameter.

This matters for §3.3 specifically. The determinant and sampling measurements
share the weights, the height convention and the dual path, but nothing else —
one inverts a Kasteleyn matrix, the other runs a shuffling algorithm. Both give
scale `2b`, so the factor of 2 is not an artifact of the determinant machinery.

Cost at `n = 192` on one core: 91 s to build the shuffling levels (one-time per
size and weight set) and 32 ms per sample; at `n = 96` the whole measurement
above takes about 30 s per value of `a`.

Code: `vecshuffle.py`, `genshuffle.py`, `conditioned.py` (for the dual path).

### 3.5 Limits

The determinant method is capped at `n ≈ 60`. `cond(K) ≈ 10^{0.252 n}` —
confirmed by direct SVD (`3.3×10⁴, 8.7×10⁶, 2.5×10⁹, 7.2×10¹¹` at
`n = 16, 24, 32, 40`, with `σ_max` pinned at 2.4 and `σ_min` collapsing). The
cause is the frozen corners, whose deterministic matchings create
near-degeneracies. Extended precision does not help: the error is upstream, in
the double-precision sparse solves.

This is why the `a = 0.3–0.5` rows are the most precise. It is no longer the
reason `a = 0.7` is believed, though — §3.4 measures that point directly by
sampling at `n = 192`, where no conditioning issue arises. There is also a partial implementation of the Chhita–Johansson contour formula
(arXiv:1410.2385, with the definitions as given in Bain arXiv:2204.06378) in
`code/exactinv.py`, which would remove the ceiling analytically — it reproduces
the matrix inverse to `10⁻¹⁴` near the centre at cost independent of `n` — but
it still has a systematic error away from the centre that I have not resolved.
Since §3.4 removes the ceiling by sampling instead, nothing here depends on it.

---

## 4. Limit shapes from Schottky data

Independently of the above, I built the pipeline of arXiv:2407.19462 /
arXiv:2402.08798 and validated it end to end.

**Amoeba and polygon maps** (`amoeba.py`) via the Poincaré series of
Proposition 41, with `z₀ = ∞` so the cross-ratio degenerates to a simple ratio.
Checks, on the Figure 21 data of [BBS]: `Im ζ_k` constant on the compact oval to
10 digits (Prop 31); `Im ζ_k` on the outer oval in multiples of `π`, jumping by
exactly `π` at each puncture, to 26 digits; `Re ζ` diverging at the train tracks.

**Ronkin function** (`ronkin.py`) via Krichever's `∇ρ = Δ`, integrating the exact
1-form. Path-independent to `3 × 10⁻⁷`; convex; affine on the amoeba hole. And,
unanticipated: `det Hess ρ = 0.1013212 ± 2×10⁻⁶` across twelve widely separated
points while the trace varies 35% — against `1/π² = 0.1013211836`. That is
Passare–Rullgård reproduced from the Schottky construction, and it was not built
in.

**Surface tension** (`tension.py`) as the Legendre dual, with `∇σ = x` to 6
digits and `det Hess σ = π²` to 6 digits.

**Limit shape** (`lp_solve.py`, `smooth_lp.py`, `arctic.py`): minimising
`∫σ(∇h)` as a linear program over tangent-plane epigraph variables — exact
energy, kinks handled as active constraints — with a log-sum-exp smoothing when
a unique minimiser is wanted. All three phases appear in the right arrangement,
with the arctic curve tangent to the boundary at the edge midpoints and the gas
facet interior.

**External check** (`fock_sample.py`, `fock_aztec.py`): sampling the
Fock-weighted model with a generalised shuffling algorithm (Janvresse–de la
Rue–Velenik) and measuring the facet slope. The coordinate map is derived, not
fitted — the four frozen corners give gradients exactly `(±½,0), (0,±½)`, which
fixes `s₁ = g_x + g_y + ½`, `s₂ = g_x − g_y + ½`. With the fit window taken from
the predicted facet extent (no reference to the sampled data):

| n | \|measured − predicted\| |
|---|---|
| 24 | 0.0163 |
| 48 | 0.0062 |
| 96 | **0.0008** |

At `n = 96` the measurement is `(0.4723 ± 0.0009, 0.5165 ± 0.0012)` against the
predicted `(0.4716546, 0.5170360)`. The prediction comes from the Schottky data
with no simulation anywhere in its derivation.

A genus-2 version works as far as the polygon map (two distinct facet slopes,
`Im ζ` constant on each oval to `10⁻⁶`), but the facets occupy only 1–3% of the
limit shape for every configuration inside the region where I can check the
period matrix against Bobenko's `jtem` code, so I have not pushed it further.

---

## 5. What I am least sure of

* The `a = 0.7` row of the table in §3.2 is only 0.8% converged by the
  determinant method. It is corroborated independently by sampling at
  `n = 192` (§3.4), but at 1.6σ with a ±0.0036 statistical error — less sharp
  than the six-digit agreement at small `a`.
* The factor of 2 is measured robustly but I cannot say where in the derivation
  it enters, and the `A`-cycle suggestion in §3.3 is a guess.
* The genus-2 limit shape is not validated against sampling.
* The contour-integral route that would remove the `n ≈ 60` ceiling is
  incomplete.
