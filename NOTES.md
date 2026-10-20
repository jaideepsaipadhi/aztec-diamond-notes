# Doubly periodic Aztec diamond: the Section 4.6 period, and the discrete component

Jaideep Sai Padhi — notes for M. Nicoletti, September 2026

> **Correction, 30 September 2026.** An earlier version of these notes reported a
> factor of 2 between Theorem 1.2 and the dimer model. That was wrong. The factor
> was an artifact of the observable I measured — the height at a *single* face of
> the gas facet rather than a spatial average over it — and Theorem 1.2 is correct
> as stated. §3.3 is kept as a record of the error, since what went wrong there is
> instructive; §3.6 has the corrected measurement. The period discrepancy in §2 is
> unaffected and now rests on a fourth, independent route.

Everything below refers to Berggren–Nicoletti, *Geometry of the doubly periodic
Aztec dimer model* (arXiv:2502.07241), cited as [BN].

---

## 1. Summary

1. **The period `B` in Section 4.6.** For the genus-1 example at `a = 0.7` I get
   `B = 0.772737661704 i`, against `0.521828 i` in the text. Five independent
   routes now agree, two of which use no elliptic function theory at all, and one
   of which (§3.6) makes no reference to the spectral curve whatsoever.

2. **Theorem 1.2 and Remark 4.18 confirmed, at genus 1.** Measured through a
   spatial average over the gas facet, the discrete component is a discrete
   Gaussian of scale `B` with shift `e = 1/4`, matching Corollary 4.17 to 0.03σ
   on the discriminating ratio. The four alignments of Remark 4.18 give measured
   shifts `{1/4, 3/4, 1/2, 0}`. **This supersedes the factor-of-2 claim** in the
   first version of these notes, which came from a bad observable (§3.3).

3. **Theorem 1.2 confirmed at genus 3**, including the off-diagonal entries of
   the period matrix, which have no genus-1 analogue (§5). Twelve atoms spanning
   three decades of probability, `χ² = 3.8` on 7 dof, with only the shift vector
   fitted; a period matrix from different weights is rejected by
   `Δ(−log L) = 208`.

4. **A validated limit-shape pipeline** from Schottky data through the amoeba
   and Ronkin function to a variational limit shape, checked against sampling
   at `8 × 10⁻⁴`.

5. **The quasi-periodic dependence on `N`, demonstrated (§6).** The shift
   velocity `v` of equation (19) is identified as `−2 u(0₊)`, computed from the
   spectral curve at 40 digits, and found to agree with an independent
   Monte-Carlo measurement: all three components of configuration A
   (`χ² = 2.6` on 3 dof) and both labelling-invariant channels of configuration
   C. Its continued fraction has only ordinary partial quotients, so `v` is
   irrational and the orbit never closes. The genus-1 case is 4-torsion —
   `v = 1/4` to 22 digits — which is why §4.6 of the paper, the one worked
   example, is periodic and shows none of this. The measurement requires an
   `n`-parity term that pure linear advance omits (§6.1); without it the
   residuals are not noise but a near-perfect alternation, `ρ = −0.997`.

Findings 1 and 2 are independent of each other: 2 does not rest on 1, and 1 does
not rest on 2. Finding 5 rests on the genus-3 machinery of 3.

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

## 3. The dimer measurement at genus 1

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

> Superseded by §3.6. The distribution below is that of the **single-face**
> observable, which is the discrete component convolved with the local gas
> fluctuation. The table is reproducible and the entries are correct for what
> they measure; they are simply not the discrete component. The agreement of the
> last column is genuine and is why `b(a)` was confirmed to 6–7 digits — that
> part survives.

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

### 3.3 The factor of 2 — what went wrong

**This section is retracted. It is kept because the way it failed is the useful
part.** The corrected measurement is §3.6.

What §3.2 and §3.4 measured was the height at a **single face** of the gas facet,
relative to the frozen boundary. That is not the discrete component. It is the
discrete component *plus the local gas fluctuation at that face*, and in this
model the local fluctuation is a near-symmetric displacement of one face by one
unit, with mass roughly 0.05 at `a = 0.5` and 0.11 at `a = 0.7` on each side.

That contamination does two things at once. It fills in the `P(−1)` atom, which
the true discrete component essentially does not have at `e = 1/4`, and it
inflates `P(+1)`. The resulting three-atom distribution is very nearly
symmetric — and a symmetric discrete Gaussian with the observed neighbour ratio
has scale `2b`. Hence the factor of 2, at every parameter value, to six digits.

The measured numbers were right; the observable was wrong. Concretely, at
`a = 0.7`, `n = 192`:

| | single face (old) | spatial average (§3.6) | Cor. 4.17, `e = 1/4` |
|---|---|---|---|
| `P(+1)/P(0)` | 0.1229 | 0.12728 ± 0.00461 | 0.130972 |
| `P(−1)/P(0)` | 0.1323 | 0.00226 ± 0.00065 | 0.002247 |

`P(−1)/P(0)` is the whole story: the single-face observable overstates it by a
factor of 58, and the symmetry that produced it is exactly what the local
fluctuation manufactures.

**Why the checks in the original §3.3 did not catch it.** Each exclusion was
correct as far as it went — the Gaussian convention in (28) really has no ½, the
atom index really is `n`, the height normalisation really is as claimed, and the
scale really did come out `2b` on both sublattices and by two independent
computational routes. What none of them tested was whether the *quantity being
measured* was the discrete component. The determinant route and the sampling
route agreed with each other because they share the observable, not because the
observable was right. Two methods agreeing is not independence when they compute
the same wrong thing.

The `A`-cycle normalisation suggested at the end of the original section as a
possible source was also wrong, and separately: §2.3 of [BN] states that the `A`
cycles are the compact ovals and the normalisation is the full loop integral,
which is what was used here.

Code as it was: `conditioned.py`, `lowrank.py`, `ndep.py`, `twoperiodic.py`.

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

The determinant and sampling measurements share the weights, the height
convention and the dual path, but nothing else — one inverts a Kasteleyn matrix,
the other runs a shuffling algorithm. Both give scale `2b`, which showed that the
factor of 2 was not an artifact of the determinant machinery.

**It was an artifact of the dual path**, i.e. of the observable both methods
share. That is the point made in §3.3: agreement between two methods computing
the same quantity says nothing about whether the quantity is the right one. The
part of this subsection that survives is the engineering — sampling removes the
`n ≈ 60` conditioning ceiling entirely, and §3.6 is built on it.

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

### 3.6 The spatial average — the corrected measurement

M. Nicoletti's diagnosis (September 2026): *"You need the spatial average to get
the right distribution."* That is exactly right.

The discrete component is a **rigid shift of the entire gas facet**, so it
survives spatial averaging; the local gas fluctuation is not, and averages away.
The observable is therefore

    Zbar(R) = (1/|S_R|) Σ_{f ∈ S_R} h(f) − h(root),
    S_R = { faces (x,y) : |x+½| ≤ R, |y+½| ≤ R }

with `h` the height on faces. The facet height moves in steps of 4 raw Thurston
units, so the discrete component in the paper's normalisation is `Zbar/4`.

**Cost.** A per-sample BFS over `(2n)²` faces would dominate everything, and is
not needed: the face topology and step signs are fixed, only the matching varies.
Fixing a BFS tree of faces rooted at a frozen corner, `h(f)` is the sum along its
tree path of `s_e(1 − 4M_e)`; summing over `f ∈ S`, each tree edge appears with
multiplicity `cnt[f]` = the number of faces of `S` below it, so

    Σ_{f∈S} h(f) = A − 4 (coef · M),   coef[e_f] = cnt[f]·s_f

— one dot product per sample, the same cost as the single-face observable. The
whole `R`-sweep is one extra coefficient vector per `R`. Checked against a direct
BFS height on every run.

**The separation is visible in the data.** From `R = 2` to `R = 24` the pmf is
identical to five digits while the off-lattice residual falls like `1/R`: the
`±1`, `±2` noise atoms vanish (0.041 → 0.0007 at `a = 0.5`) while the `±4` atom
does not move (0.056 → 0.063). That atom is the discrete component.

**Result at `a = 0.7`,** with `n = 96` and `n = 192` agreeing:

| | measured | Cor. 4.17, `e = 1/4` | symmetric, scale `2b` |
|---|---|---|---|
| `P(+1)/P(0)` | 0.12728 ± 0.00461 | 0.130972 ✓ | 0.130972 |
| `P(−1)/P(0)` | 0.00226 ± 0.00065 | 0.002247 ✓ (0.03σ) | 0.131 ✗ |

At `a = 0.7` the `+1` ratio cannot discriminate — both readings predict 0.131.
`P(−1)/P(0)` can, by a factor of 58, and it lands on the prediction to 3%.
Convergence in `R` is clean: 0.141 → 0.021 → 0.0054 → 0.0033 → 0.00226 → 0.00226.

Across the family:

| a | b = K′/K | `P(+1)/P(0)` measured | predicted | `P(−1)/P(0)` measured | predicted |
|---|---|---|---|---|---|
| 0.4 | 0.488978 | 0.04275 ± 0.00292 | 0.04026 | 0 | 6.5×10⁻⁵ |
| 0.5 | 0.569841 | 0.06129 ± 0.00391 | 0.06351 | 0 | 2.6×10⁻⁴ |
| 0.6 | 0.661751 | 0.09010 ± 0.00425 | 0.09314 | 0.00065 ± 0.00038 | 8.1×10⁻⁴ |
| 0.7 | 0.772738 | 0.12728 ± 0.00461 | 0.13097 | 0.00226 ± 0.00065 | 2.2×10⁻³ |

**Remark 4.18.** The two independent generators are `a ↔ a⁻¹`, and flipping the
black-vertex class on the **β family only**. (My first attempt flipped both
families, which is the same map as `a ↔ a⁻¹` — that is why it produced only two
of the four measures.) With the right pair, all four shifts appear:

| generators | `P(−1)/P(0)` | `P(+1)/P(0)` | implied `e` | Rmk 4.18 |
|---|---|---|---|---|
| — | 0.00247 | 0.13541 ± 0.0048 | 0.2541 ± 0.0043 | 1/4 |
| `a↔a⁻¹` | 0.13562 | 0.00247 | −0.2386 ± 0.034 | 3/4 |
| β-flip | 0.00490 | 0.95428 ± 0.0126 | 0.4942 ± 0.0016 | 1/2 |
| both | 0.01988 | 0.01711 ± 0.0017 | −0.0003 ± 0.0123 | 0 |

The `e = 1/2` case puts **exactly equal mass on two neighbouring atoms** for any
`b`, which nothing else in the family does. Its 4% deficit at `n = 96` is
finite-size: the deviation runs −58%, −30%, −12.5%, −4.0%, −2.4%, −2.5% at
`n = 32, 48, 64, 96, 144, 192`, decaying with a length of about 21 lattice units,
and is 1.4σ from an exact tie by `n = 192`.

**A fifth route to the period.** The `e = 0` alignment is shift-free, so
`P(±1)/P(0) = exp(−π/b)` measures `b` with nothing assumed — no cycle choice, no
elliptic identity, no spectral curve at all:

    b = 0.7873 ± 0.0137     (K′/K = 0.772738 → 1.1σ;  0.521828 → 19σ)

This is also an independent kill of the old `2b` reading: at scale `2b` the
`e = 0` alignment would give 0.131, and 0.0171 = `exp(−π/b)` is measured.

Code: `spatial_avg.py`, `align_avg.py`, `half_scaling.py`.


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

**Genus 2.** The same construction carries over and has also been checked
against sampling. Two distinct gas facets are predicted from the Schottky data
through the polygon map, and both are found in a sampled Fock-weighted
configuration at `n = 96` with 20 samples:

| | predicted | measured | distance |
|---|---|---|---|
| facet 1 | (0.2738, 0.3417) | (0.2811 ± 0.0076, 0.3378 ± 0.0061) | 0.0084 (0.9σ) |
| facet 2 | (0.6633, 0.7312) | (0.6499 ± 0.0084, 0.7195 ± 0.0085) | 0.0178 (1.5σ) |

(The Newton polygon is only defined up to translation; the predicted values are
shifted by `(+1, 0)` into the frame of the measurement.)

The configuration is checked three ways: the period matrix against Bobenko's
`jtem` code to `7 × 10⁻⁸`; the Kasteleyn sign condition (5) satisfied at every
face; and the facet slopes above.

**The sign condition is not optional and is easy to lose.** Equation (5)
requires `sign(W_f) = (−1)^{n+1}`, so `−1` for quadrilateral faces. An earlier
version of this measurement used Harnack data whose face weights came out all
*positive* — not a dimer model at all — and the gauge reconstruction takes
`|W_f|`, so the sampler ran happily and produced plausible numbers. It surfaced
only because changing `Im A` flipped the sign. Of sixteen configurations
surveyed, twelve satisfy the condition and four do not, with the same
train-track cyclic order in both groups, so the ordering condition of Theorem 1
does not by itself decide it.

Two things make the genus-2 case tractable. The Schottky–Klein prime form,
which the *edge* weight needs, requires holomorphic spinors — but in the *face*
weight (10) each train-track point occurs once in a numerator and once in a
denominator, so the spinors cancel and only `θ[Δ]` with an odd characteristic
survives, a lattice sum over `ℤ²`. And `θ` is 1-periodic in each component, so
the face weight depends on the discrete Abel map only through `η mod 1`, giving
a 2D lookup accurate to `6 × 10⁻⁵`.

Checks on the machinery: all six odd characteristics vanish at `0` to `10⁻²¹`;
the train-track factor of (10) is real to `4 × 10⁻¹⁵` (the M-curve prediction,
and not forced by the computation); the face weights are real to `10⁻¹⁵`.

The facets occupy a few percent of the limit shape, which is why the agreement
is at the `10⁻²` level rather than `10⁻⁴`.

Code: `genus2_fock.py`, `genus2_amoeba.py`.

---

## 5. Genus 3: the period matrix, off-diagonal entries included

Genus 1 tests one number. The substantive content of [BN] is the higher-genus
statement, where the discrete component is a **vector** and `τ = −B⁻¹` is a
matrix whose off-diagonal entries predict *correlations between gas facets* —
something with no genus-1 analogue.

### 5.1 A self-contained route to the curve

Rather than work from [BN]'s `α, β, γ` labelling, everything here comes from the
model's own Kasteleyn matrix. In diagonal coordinates
`(u,v) = ((x₁+x₂−1)/2, (x₂−x₁+1)/2)` the Aztec diamond graph **is** the square
lattice, so a periodic Aztec weighting is a periodic square-lattice dimer model
and `P(z,w) = det K(z,w)` is immediate. With `mv = 2` the curve is hyperelliptic,

    y² = D(z) = Bc(z)² − 4A(z)C(z),   w·P = A w² + Bc w + C

so the period matrix follows from real-axis quadrature. A `(2,4)`-periodic
weighting gives a Newton polygon with **three** interior points, `(−1,0), (0,0),
(1,0)` — genus 3, three gas facets.

The genus-1 fixture reproduces `b = 0.7727376617042` end to end (rel. err
`2×10⁻¹³`), which is the fourth independent route to §2's number and touches
neither elliptic identities nor cycle bookkeeping.

**Three errors that genus 1 structurally cannot detect**, each caught by a check
rather than by inspection:

* **Normalisation.** The normalised period matrix is `ℬ𝒜⁻¹`, not `𝒜⁻¹ℬ`. These
  agree at genus 1, so the fixture passes either way and hides the error
  completely.
* **Branch phases.** A consistent branch has `y = i^{2g+2−p}|D|^{1/2}` on the
  `p`-th real interval, so successive ovals and cuts enter with *alternating*
  signs — invisible with a single oval. Caught by the homology relation: the
  `g+1` real ovals must sum to zero, and do, to `2×10⁻¹⁸`.
* **The dual basis.** `B_k` is only defined up to adding `A`-cycles, which shifts
  `B` by an **integer** matrix. So the bilinear test is "`B − Bᵀ` is integral",
  not "vanishes".

With those fixed, `B` is symmetric to `10⁻¹⁸` with `Im B ≻ 0`, and — unforced —
satisfies `B₁₁ = B₃₃`, `B₁₂ = B₂₃`, the structure the weights' own reflection
symmetry demands.

### 5.2 Finding the facets, and choosing weights

The facets are read off the data, not assumed. The mean height's slope
`dh/d(4y)` takes exactly five plateau values, `−8, −4, 0, +4, +8`; divided by 4
these are `−2, −1, 0, +1, +2` — the Newton polygon's two vertices and its three
interior points. The `±8` plateaus have zero variance (frozen); the other three
are the gas facets, stacked along `y`.

Weights have to be chosen against a real tension: a wide amoeba hole gives a
large facet but a *rigid* phase, hence small `B_ii` and invisible atoms. Tuning
for hole size alone drove the holes to ~1.8 and `B_ii` to 0.39 — three facets,
none measurable. The configuration used has holes 0.64 and
`diag B = (0.634, 0.834, 0.640)`.

### 5.3 Result

`n = 192`, `K = 20000`, with `Im B` fixed from the curve and **only the three
shifts fitted**:

| atom | observed | predicted |
|---|---|---|
| (0,0,0) | 0.49785 | 0.49880 |
| (0,0,1) | 0.38460 | 0.38273 |
| (−1,0,0) | 0.03355 | 0.03222 |
| (−1,−1,0) | 0.02515 | 0.02441 |
| (0,−1,0) | 0.02420 | 0.02528 |
| (0,1,1) | 0.01455 | 0.01471 |
| (0,−1,1) | 0.00140 | 0.00133 |
| (−1,−1,1) | 0.00095 | 0.00092 |

Twelve atoms over three decades of probability, Pearson `χ² = 3.8` on 7 dof. The
correlations — which come from the off-diagonal entries alone:

| | measured | predicted |
|---|---|---|
| corr(Z₁,Z₂) | +0.3388 | +0.3357 (0.5σ) |
| corr(Z₁,Z₃) | +0.0992 | +0.1012 (0.3σ) |
| corr(Z₂,Z₃) | +0.2215 | +0.2192 (0.3σ) |

`χ² = 0.4` on 3 dof, stable across an erosion sweep of the facet mask spanning a
4× change in facet area (2097 → 476 faces), with the fitted shift identical to
three decimals throughout.

Two further checks:

* **The labelling is fixed by the off-diagonals alone.** The two facet↔component
  labellings with essentially identical likelihood (`−log L` 23368.8 vs 23369.0)
  are separated by the correlations by a factor of 76 in `χ²` (9.2 vs 703).
* **Marginal broadening.** The marginal of a multivariate discrete Gaussian is
  *not* a 1-D one of scale `B_ii` — summing out the other components broadens it,
  most for the facet coupled to both neighbours. Model marginal scales
  0.6137 / 0.9073 / 0.6338 against measured 0.6067 / 0.9200 / 0.6518, i.e.
  0.3σ / 1.5σ / 0.3σ. Facet 2's broadening from 0.834 to 0.907 is a second,
  independent measurement of the off-diagonals. (Reading those numbers as `B_ii`
  instead gives a spurious 10σ — the shift-independent product test of §3.6 is a
  genus-1 tool and does not survive to `g > 1`.)

### 5.4 Falsification

A good fit with three free shifts proves little on its own, so: a **second**
weight configuration `C`, with period matrix differing from `A` by up to 18% on
the diagonal, was computed and its matrix offered `A`'s data — given every
advantage, namely all six facet labellings and its own three free shifts.

| on A's data | `−log L` | atoms `χ²` | corr `χ²` |
|---|---|---|---|
| A's own matrix | 23715.8 | 3.8 / 7 | 0.5 / 3 |
| C's matrix | 23923.6 | 508.2 / 8 | 62.8 / 3 |

Rejected by `Δ(−log L) = 208` and a factor of 134 in atom `χ²`. The reverse holds
too: on C's own data, C beats A by `Δ(−log L) = 27.7`, with A forced to permute
its facets to `(1,0,2)` to approximate it at all. So the measurement determines
the period matrix; it is not being absorbed by the fitted parameters.

Code: `periodic_dimer.py`, `hyper_curve.py`, `hyper_g3.py`, `g3_sample.py`,
`g3_measure.py`, `g3_erode.py`, `g3_swap.py`, `g3_tune2.py`, `g3_asym2.py`.

---

## 6. The shift velocity, and why the orbit never closes

Everything above fixes the *scale* matrix `B`. The shift vector `e` is the other
half of Theorem 1.2, and it is where the quasi-periodicity in `N` lives.
Equation (19) makes `e` advance **linearly** on the Jacobian,

```
e^(kN) − e^(0) = N · k ( Σᵢ u(q₀,ᵢ) − ℓ Σᵢ u(p₀,ᵢ) ),
```

with some velocity `v`. If `v` is rational the orbit closes and the model is
genuinely periodic in `N`; if `v` is irrational it never closes, and that —
not any quasi-periodicity in the weights — is the phenomenon. BN's own §4.6
example happens to be the closed case, which is why it shows nothing.

### 6.1 `v` measured, and the `n`-parity term

The obvious model is pure linear advance, `e(n) = c + (n/2)v (mod 1)`, one
k-step per `n → n+2`. **It is misspecified**, and badly. Fitting it leaves
residuals with lag-1 autocorrelation `ρ = −0.997` across adjacent `n`
(permutation `p < 10⁻⁴`) — a near-perfect alternation, not noise. Odd and even
`n` are two different sublattices, and `e` carries a parity term:

```
e(n) = c + (n/2) v + (−1)ⁿ d   (mod 1)
```

This is the alignment of the weight pattern with the diamond — the same effect
as the orientation flip of the height function with `n` parity — and it is a
finite-`n` phenomenon, not part of `v`. Including it collapses configuration A's
residual scatter from **0.111 to 0.0070** (16×), with concentration `R = 0.999`
and amplitudes `d = (+0.111, −0.101, +0.124)`.

A global circular fit with the parity term, using every `n` at once, then gives

```
v_A = (0.285430 ± 0.000113, 0.415401 ± 0.000109, 0.285357 ± 0.000092)
```

errors by residual bootstrap with the design held fixed.

**Two estimator traps, both of which I fell into.** An earlier version of this
section quoted `v_A = (0.28548, 0.41538, 0.28539) ± 0.00014` from a bootstrap
ladder reading the endpoint pairs only. That error bar was not supported: it is
a circular concentration computed from the **two** pairs `(88,186)` and
`(89,187)`, which are adjacent and not independent. The ladder's answer is
nonetheless close to right, for a reason it never states — both pairs have
`Δn = 98`, which is *even*, so the parity term cancels identically. Separately,
a global fit whose errors come from resampling `n` **with replacement** reports
A's error as `±0.0020`, worse than the endpoint ladder using less data: that
bootstrap randomizes the design, and this estimator's precision comes from the
lever arm, so resamples that drop the extreme `n` collapse the baseline. The
design must be held fixed and only the noise resampled.

**Sampling still cannot settle rationality.** By Dirichlet some `p/q` always
fits; a measurement of precision `ε` bounds the period from below and no more.
Against the sharpened error, `2/7 = 0.2857142857` sits 2.5σ from `v₁` — excluded
only weakly. The decisive exclusion is §6.3's, from the curve.

### 6.2 `v` computed from the curve

Equation (19) cannot be applied verbatim here: BN's `(k, ℓ)` family needs a
Newton polygon with `2(k+ℓ)` boundary lattice points, and the diagonal-coordinate
diamonds used here have four. But the *shape* of (19) survives. On a
hyperelliptic curve the angles are the two points over `z = 0` and the two over
`z = ∞`, and the hyperelliptic involution relates them: with base point `∞₊`,

```
u(∞₊) = 0,   u(∞₋) = A,   u(0₊) = B,   u(0₋) = A − B,
```

so *any* integer combination of angle images is `mA + nB`. That is a
two-parameter ansatz against a measured 3-vector — overdetermined, hence a test
rather than a fit. It picks out

```
genus 1, (2,2):   v = −B
genus 3, (2,4):   v = −2B
```

Carried at 40 digits, with all `2g+2` branch points real and negative so `(0, ∞)`
lies on one sheet:

| | value | status |
|---|---|---|
| genus-1 fixture, `−B` | `0.2500000000000000000000168` | `= 1/4` to 22 digits |
| `A − B`, config A | `(−0.25, −0.50, −0.75)` to 22 digits | 4-torsion, free check |
| `A − B`, config C | `(−0.25, −0.50, −0.75)` to 22 digits | same, different curve |

The genus-1 fixture is the calibration: `v = 1/4` is known independently from
Remark 4.18, and the quadrature reproduces it to 22 digits. So the precision
claimed below is real.

### 6.3 The two agree, and `v` is irrational

| | computed from the curve | measured | |
|---|---|---|---|
| `v_A` | `(0.2855890797, 0.4154709760, 0.2853120189)` | `(0.285430, 0.415401, 0.285357)` | 1.4σ / 0.6σ / 0.5σ |

`χ² = 2.6` on 3 dof. Two independent routes — a Monte-Carlo fit to sampled
height functions, and a period computation on the spectral curve — agreeing on
all three components of configuration A.

**Configuration C agrees only in its labelling-invariant channels, and this is
unresolved.** Taken per component under the facet labelling of `g3_facets.py`,
C lands 3.8σ and 4.0σ away on components 1 and 3 (`χ² = 30`), while component 2
agrees at 0.2σ. The failure has a clear shape: C's measured `v₁ = 0.270372` and
`v₃ = 0.270010` are nearly equal, and the mean of the two predicted values is
`(0.265853 + 0.274804)/2 = 0.270329`. Both components sit on the average of the
pair — the signature of C's two *weak* facets being interchanged from one `n` to
the next. Three things support that reading:

* `e₁ + e₃` is invariant under the interchange, and against the curve's own
  `v₁ + v₃` with nothing fitted but offsets it gives residual sd 0.036 against
  0.104 for either component alone — 3× better. For configuration A the sum is
  slightly *worse* than the components (0.0099 vs 0.0072 / 0.0057), which is
  what summing uncorrelated noise does when there is no interchange to cancel.
* A one-dimensional scan over `D = v₃ − v₁` with the interchange assigned per
  `n` recovers `|D| = 0.00940` against the curve's `0.00895` — 5% in magnitude.
* The same scan on configuration A returns an interchange rate of **0.00** out
  of 31 points and reproduces the per-component fit to six digits, so the
  per-`n` freedom is not manufacturing the agreement.

But the scan's `D` comes out with the **opposite sign**, i.e. it wants
components 1 and 3 globally transposed, and that transposition cannot be
justified: C's `Im B` has `B₁₁ = 0.694` against `B₃₃ = 0.529`, 24% apart, so the
two components are strongly distinguishable and the labelling is not free. The
natural independent check — which labelled facet has the broader atom
distribution — is **confounded**: C's facet 3 has 175 faces, so it is also the
noisiest, and measured sds `(0.909, 1.075, 1.850)` running opposite to `B`'s
diagonal is equally well explained by "broadest = noisiest" as by a
transposition.

So what C establishes is the invariant combinations — `v₁ + v₃` is itself
unchanged by the transposition:

| configuration C | measured | curve | |
|---|---|---|---|
| `v₁ + v₃` | `0.540474 ± 0.000563` | `0.540657` | 0.3σ |
| `v₂` | `0.397195 ± 0.000262` | `0.397145` | 0.2σ |

(The same channels for A: `v₁ + v₃ = 0.570788 ± 0.000156` against `0.570901`,
and `v₂` as tabled — 0.7σ each.) C therefore confirms `v = −2u(0₊)` in every
channel its facet data can actually speak to, and its individual weak-facet
assignment is an open item, listed in §7.

Rationality is then decidable, because a *computed* number can be tested far
harder than a measured one. **The discriminator is the partial-quotient
pattern.** If `v = p/q` with modest `q`, its continued fraction terminates and
the tail is pure quadrature noise, which shows up as one astronomically large
partial quotient. The genus-1 fixture does exactly that:

```
B  = [0, 1, 2, 1, 3718120517003592139456, 6, ...]      ->  exactly 3/4
```

All six genus-3 components instead have ordinary partial quotients — maxima
between 13 and 311 — no termination, and the best rational with `q < 10⁶` misses
by `5×10⁻¹³` to `3×10⁻¹²` rather than by zero. In particular `2/7 = 0.2857142857`,
which sat 0.1σ from the coarse measurement, is excluded outright: it differs from
`v₁` in the fourth decimal.

```
v_A − v_C = (0.019735920423, 0.018325970377, 0.010507863355)
```

exactly, from the two curves with no sampling anywhere. Measured, the
labelling-invariant difference `(v₁+v₃)_A − (v₁+v₃)_C = 0.030314 ± 0.000585`
against the curve's `0.030244` — 0.1σ, and 52σ from zero. So `v` demonstrably
moves with the weights, which is the part of the argument that does not depend
on any single `v` being irrational: `v` is an Abel-map expression in the angles,
the angles move continuously with the edge weights, and a continuously varying
quantity is irrational for almost every parameter value.

So: `v` is irrational, the orbit does not close, and the dependence on `N` is
quasi-periodic. The genus-1 case is torsion (`v = 1/4` exactly), which explains
why the one worked example in the paper is periodic and the phenomenon is
invisible there.

**One caveat on the code.** `abel_exact.py`'s automatic `RATIONAL` flag is
mis-calibrated — it reports "no exact rational" for the genus-1 `B`, which *is*
exactly `3/4`, because the terminating convergent's error sits just above its
threshold. The conclusion above rests on the partial-quotient pattern, which is
unambiguous; the flag should be ignored or rewritten.

Code: `abel_v.py`, `abel_exact.py`, `shift_vs_n.py`, `shift_g3.py`,
`shift_g3c.py`, `shift_ladder.py`, `g3_facets.py`.

---

## 7. What I am least sure of

Attacked in order of what turned out to be tractable. Three of the seven moved;
two were reframed; one is unchanged; one is untouched.

### 7.1 The contour-integral route: the gas inverse now works; the `a = 0.7`
application does not yet use it

Two earlier claims in this subsection were withdrawn (exponents as `e₁`/`e₂`
step counts; then as `u // 2`, which repaired one diagonal ray and left 27 of 36
offsets wrong). The third attempt, `gasinv2.py`, does the split the contour
integral actually requires, and this time it is validated across the domain.

**The period lattice**, which neither predecessor had. `E₁ = (1,1)` and
`E₂ = (−1,1)` are not translations of the vertex set — `E₂` maps black to white
— and `bclass` depends on `(x₁+x₂) mod 4`, which `E₁` shifts by 2. The
translations preserving classes and weights are generated by `2E₁ = (2,2)` and
`2E₂ = (−2,2)`: a cell of area 8 holding 4 vertices, 2 black and 2 white, which
is *why* `K(z,w)` is 2×2. So `z, w` are conjugate to `2E₁, 2E₂`, a displacement
`(c₁,c₂)` has `p = (c₁+c₂)/4`, `q = (c₂−c₁)/4`, and those are integral only once
the within-domain offset between class representatives is removed. `gasinv2`
removes it explicitly; nothing is rounded.

**Validation**, in order of increasing force:

* every white/black pair is period-lattice related once the representatives are
  removed (0 inconsistent pairs);
* all 36 offsets in `[−5,5]²` converge monotonically in *absolute* error,
  median `6.6e−3 → 3.1e−3 → 1.5e−3` at `n = 32, 48, 64`. The one offset whose
  *relative* error rises, `(3,−5)`, does so because the exact entry crosses zero
  (`0.0292 → −0.0049`, heading for the gas value `−0.008385`) while `|ex−gf|`
  halves;
* all four class pairs calibrate to domain shift `(0,0)`, so the representatives
  are right rather than lucky;
* assembled entrywise at `L = 8`, `n = 64`: `max|M_exact − M_gas| = 4.8e−3`
  against `max|M| = 0.386`;
* **the defining test** — the gas inverse is the `n → ∞` limit, so the exact
  finite-`n` computation must converge onto it, and does:

| `L` | `n=32` | `n=48` | `n=64` | `n=96` |
|---|---|---|---|---|
| 4 | `6.2e−3` | `3.0e−3` | `1.5e−3` | `3.4e−5` |
| 6 | `1.1e−2` | `4.5e−3` | `2.1e−3` | `2.4e−4` |
| 8 | `2.1e−2` | `9.7e−3` | `4.7e−3` | `2.9e−4` |

(`|P(0)_exact − P(0)_gas|`; the variance behaves the same way.) A wrong formula
has no reason to be approached from below by an independent computation at four
values of `n` and three separations.

**What is NOT done.** `gas_fluct.py` uses an ad-hoc horizontal dual path, and
that observable has `Var[h(x)−h(y)] ∝ L`. I took that for a bug — a gas phase
has bounded height fluctuations — but the *exact* inverse gives the same linear
growth (Var 1.92457 exact against 1.93450 gas at `L = 8`, `n = 64`), so it is a
property of the path, not of the inverse: both up-edges of a black vertex carry
the same sign there, so it is not the standard height increment. Two blind
alleys on the way, recorded because each cost a cycle: the unit torus *is*
inside the gas hole (0.502 clear of the amoeba at `r₁ = 0`), and `gasinv`'s
docstring `P` *is* exactly the true characteristic polynomial (ratio 1 to
`2e−16`) — my first comparison said otherwise only because I guessed the weight
pattern as `[[a,1],[1,a]]` instead of taking the derived
`periodic_dimer.johansson_22`, which is the error §5 already warns about.

So the remaining work is well-scoped and is not about the inverse. It is NOT,
as an earlier version of this paragraph said, to port `conditioned.py`'s path:
that observable is the wrong one (below). It is to apply `gasinv2` to §3.6's
spatial average, which is the correct observable, giving the gas-facet
distribution with no finite-size error and a second, sampling-free route to
§3.3's diagnosis. Not done here.

**The gas inverse applied: §3.3's contamination computed from the curve.**
The discrete component of Theorem 1.2 is intrinsically a finite-`n` object — it
is the integer global height shift the frozen boundary forces — so no
full-plane computation can produce one. What the gas inverse gives is the
*local* fluctuation, which is precisely §3.3's contamination, and whose size
rested on sampling alone, sharing its entire dependency set with the retracted
claim itself.

The first attempt failed on a hand-rolled dual path (both up-edges of a black
vertex given the same sign), whose variance grew linearly in the separation.
Reusing `ndep.faces_and_steps`, whose signs alternate with parity and which
underpins §3 and §5, fixes it, and the variance saturates as the gas phase
requires:

| `D` | 1 | 2 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|
| `Var`, gas (`n = ∞`) | 0.1007 | 0.1756 | 0.2019 | 0.2074 | 0.2088 | 0.2092 | 0.2093 |
| `Var`, exact `n = 64` | 0.1013 | 0.1768 | 0.2071 | 0.2204 | 0.2369 | 0.2637 | 0.3039 |

The exact values track the gas limit to `D ≈ 4–6` and then diverge, which is
the facet half-width at `n = 64` (`≈ n/24 ≈ 2.7`): the finite-`n` path leaves
the facet, while at `n = ∞` the facet is infinite and the variance saturates.
That is the finite-size ceiling being removed, on the correct observable.

| `a` | `Var(∞)` | per-side mass | §3.3 claims (sampling) |
|---|---|---|---|
| 0.3 | 0.04315 | 0.02157 | — |
| 0.5 | 0.11282 | **0.05635** | ~0.05 |
| 0.7 | 0.20932 | **0.10373** | ~0.11 |

Agreement at both values of `a`, so a two-point match rather than a
coincidence, and at `D = 1` exact and gas agree to five digits at `a = 0.3` and
`0.5`. **So §3.3's diagnosis now has a second route, sharing nothing with the
first** — no sampling, no facet observable, no shuffling.

**But §3.3's stated MECHANISM does not reproduce its own numbers.** Convolving
the true discrete component with the computed fluctuation:

| | `P(+1)/P(0)` | `P(−1)/P(0)` |
|---|---|---|
| convolution, `m = 0.1037` | 0.2574 | 0.1308 |
| convolution, `m = 0.0523` | 0.1880 | 0.0602 |
| §3.3 measured, single face | **0.1229** | **0.1323** |

The measured distribution is far more symmetric than either convolution. So the
*size* of the contamination is confirmed, and the account of how it manufactures
the near-symmetry is not. The likeliest reason is the convolution's own
assumption: the local fluctuation and the global shift are not independent, since
conditioning on the shift changes the local statistics, so a convolution is at
best an approximation. §3.3's conclusion is unaffected — it rests on §3.6 and on
the other routes, not on this mechanism — but the mechanism as written is
unverified, and the near-symmetry remains unexplained.

**§3.2's observable, fourth and I believe final statement — this time with a
mechanism and an identity rather than an inference.** I have twice demoted this
table and once restored it, each time by reading off one number. Here is what it
actually measures.

*Why its distribution is exactly symmetric.* For the translation-invariant
full-plane measure, `M_kl = K^{-1}[w_k, b_l]` is Toeplitz, and reversing the
path transposes it. Since `det(I + D Mᵗ) = det(I + M D) = det(I + D M)` by
Sylvester, `φ(−θ) = φ(θ)`, so the characteristic function is real and the pmf
is **exactly** symmetric — for any straight path, unavoidably. That is why the
atoms agree to seven digits rather than two or three, and it rules out
"approximately symmetric by contamination" as the explanation. (A midpoint-parity
mechanism was my first guess and is wrong: every start offset tested gives ratio
1.000.)

*What it therefore measures.* Both endpoint faces lie in the same gas facet, so
the discrete component — a single global integer shift — cancels **identically**
in their difference. §3.2 measures pure local gas fluctuation, with no discrete
component present at all. §3.3 is right that it is not the discrete component,
and its stated mechanism (discrete component convolved with contamination) is
wrong, which is exactly why that convolution failed to reproduce its own
numbers.

*The identity, corrected. An earlier revision of this paragraph claimed the
saturated local fluctuation is a discrete Gaussian of scale `2b` outright. That
is true only for a subset of face classes, and I had checked exactly one.*

Pushing several starting offsets to saturation at `a = 0.7` (all three converge;
these are limits, not transients):

| starting offset | saturated `P(1)/P(0)` | `P(2)/P(0)` |
|---|---|---|
| (0,0), (1,1), (2,0), (0,2) | **0.13097249** = `exp(−π/2b)` | `2.943e−4` |
| (1,0) | 0.03428741 | `2.943e−4` |
| (0,1) | 0.50029418 | `2.943e−4` |

So the `k = ±1` atom depends on the face class mod 2 — taking at least three
distinct values, with the two diagonal classes agreeing — while **the tail
`k ≥ 2` is universal and exactly `DG(2b)`**: `2.943e−4` for every class, against
`exp(−4π/2b)`, and at `k = 3` the measured atoms sit at `1.1e−8` against the
same prediction. Verified relative to `DG(2b)` at `1.3e−6` (`k = 2`) and
`4.3e−5` (`k = 3`).

For the **diagonal classes** the whole distribution is `DG(2b)`, to
`8.2e−14` at `a = 0.5`:

```
P(k)/P(0) = exp(−π k² / 2b),    b = K′/K,  k² = 1 − a⁴
```

**This still closes the factor of 2, and §3.2's table is still a correct
determination of `b`** — its conditioned path starts at face `(0,0)`, which is
in the diagonal class, which is why its ratio matches `exp(−π/2b)` to seven
digits. The route count of five stands. What is withdrawn is the generality: the
statement is about diagonal-class face pairs, with a universal tail, not about
"the local fluctuation" tout court.

There are still two discrete Gaussians and the factor of 2 is still the ratio of
their scales: the discrete component of Theorem 1.2 has scale `b` and shift
`1/4`; the diagonal-class local fluctuation has scale `2b` and shift `0`.

*The genus-3 generalisation fails as first guessed.* The natural conjecture —
facet `i`'s local fluctuation is `DG(2(Im B)_ii)` — was tested by sampling
configuration A at `n = 96` and gives measured `P(1)/P(0)` of
`0.271 / 0.122 / 0.138` against predicted `0.084 / 0.152 / 0.086`, off by up to
3.2×. But the face pairs there were chosen by `x`-extremes within each facet and
so are of arbitrary classes, which the genus-1 table above shows is exactly what
the `k = 1` atom is sensitive to. The test must be redone on diagonal-class
pairs before the conjecture is judged; as it stands it is untested, not refuted.

*Why §3.6 can see the shift and §3.2 cannot.* Any height difference between two
faces is symmetry-forced as above, so no such observable can ever reveal the
shift `e = 1/4`. The shift is only visible relative to a fixed reference, which
is what §3.6's spatial average has — its BFS tree is rooted at the frozen
boundary.

### 7.2 MOSTLY RESOLVED: the shift vector is no longer fitted

§6 identifies `v = −2u(0₊)`, which predicts `e`'s entire `n`-dependence. For
configuration A, 93 fitted shift components collapse to **6** free numbers, and
the velocity contributes none of them:

| model | free parameters | mean residual sd |
|---|---|---|
| `e` free at each `n` | 93 | — |
| `c + (n/2)v`, `v` fitted | 6 | 0.1122 |
| `+ (−1)ⁿd`, `v` fitted | 9 | 0.0065 |
| `+ (−1)ⁿd`, **`v` from the curve** | **6** | **0.0066** |

Fitting `v` buys nothing over taking it from the curve. What remains fitted is
the constant offset `c` and the parity amplitude `d` per component, and `d` is
**not** any of the obvious closed forms — against `d = (+0.1112, −0.1008,
+0.1242) ± 0.0017`, every candidate is rejected at ≥ 7.6σ on some component:

| candidate | `v/4` | `v/2` | `1/8` | `(1−v)/4` | `1/4` |
|---|---|---|---|---|---|
| worst component | 36.6σ | 61.1σ | 13.8σ | 37.8σ | 87.2σ |

(`1/8` fits component 3 at 0.6σ and misses component 1 by 7.6σ, so it is a
coincidence, not a pattern.) The three `d` differ in sign and magnitude, so
they are not a universal constant either. Predicting `c` and `d` from the
divisor is the remaining gap, now 6 numbers rather than a free function.

### 7.3 RESOLVED, by discovering the premise was false

The item was: no configuration has three healthy facets *and* `B₁₂ ≠ B₂₃` by
more than 4.6%, so the two correlations are not independently distinguishable.
Three separate things were wrong with it.

**The asymmetry was never the obstacle.** Samples needed to separate the true
period matrix from the one with `b₁₂` and `b₂₃` interchanged, at fixed shift:

| configuration | `\|b₁₂ − b₂₃\|` | samples for 5σ |
|---|---|---|
| A | 0.00046 | 25–76 **million** |
| B | 0.01916 | **11,000 – 28,400** |
| C | 0.01284 | 21,000 – 94,200 |

The runs already use `K = 20,000`. So 4.6% is comfortably detectable and A's
0.11% never will be; the original inference was drawn without a power
calculation.

**The search's objective was backwards.** `g3_asym.py` *maximised* the
asymmetry subject to a floor on facet size, and maximising it is exactly what
starves a facet — configuration B's facet 1 is 24 faces at `n = 96` and empty
under erode 10. Given the power table, 2% suffices, so facet health should be
the objective and the asymmetry the constraint.

**And the reason it was never run that way was a cost premise that is false.**
`g3_asym2.py` checks facet sizes "afterwards by a short sampling run, which is
the ground truth anyway", which I read as minutes-to-hours per trial, and went
looking for a cheap curve-side proxy. Two attempts, both failed, both with
clear causes worth recording:

* **amoeba holes** (`hole_tune2.py`). A bounded complement component says the
  gas *phase* exists in the infinite model, not that its facet lands inside a
  finite diamond at a usable size. False positive on the first new
  configuration it was applied to — config D resolved three components with
  smallest extent 1.54, and its facet 1 has **0** faces by sampling.
* **exact finite-`n` facet sizes from `K^{-1}`** (`facet_exact.py`). Correct in
  principle, and the selected sparse inversion agrees with
  `g3_sample.exact_edge_probs` to `5e−15`. But `Σ_e p_e = |B|` is an exact
  identity and therefore a free self-check, and it fails: relative error
  `1e−15, 2e−13, 2e−9, 9e−4` at `n = 16, 32, 40, 48`, with probabilities
  exceeding 1 beyond that — `cond(K) ~ 10^{0.252n}`, the same ceiling §3.5
  documents. Exact equilibration buys ~70× at `n = 40` and does not extend the
  range. And at `n ≤ 40` the facets have not developed: **every** configuration
  including A has an empty facet there. The accurate regime and the informative
  regime do not overlap, so this cannot work in double precision.

Then I measured the thing I had assumed. **Facet health by sampling at
`n = 96` costs 9–17 seconds** — the descent dominates, the height profile
itself is 2–3 s — against ~19 s for the period matrix. The ground truth was
never expensive, no proxy is needed, and neither of the above was necessary.

`g3_search2.py` is the search: stage 1 the period matrix (reject unless
`asym ≥ 2%`, `min diag ≥ 0.5`), stage 2 the real facet sizes from
`g3_erode.facet_masks` on a sampled profile — the same detector and observable
as §5, so anything that passes can actually be measured. Most proposals fail
stage 1, so the average trial is ~20 s. Early output reaches
`asym = 0.051` with facets `[491, 121, 109]` against a threshold of 120, so
feasible configurations appear to be within reach rather than absent.

If a long run finds nothing, that is now evidence about the `(2,4)` diamond
family itself rather than about the search, and the item should be restated as
a property of the family.

### 7.4 UNBLOCKED: configuration C's facet 1 ↔ 3 assignment

Still unresolved for C itself, but no longer blocked. §7.3's ground-truth search
found a usable configuration in seven trials — **configuration E**:

```
Im B  diag (0.6689, 1.0447, 0.5579),  b12 = 0.22392, b23 = 0.21356
facets at n = 96:  [471, 171, 180]      asym 0.0237
```

Three healthy facets, and — the point for this item — predicted marginal widths
`(0.4667, 0.3362, 0.1047)` at the operative shift, so facets 1 and 3 differ by
a factor **4.46** in width. C's own width test was confounded because its facet
3 is both narrowest in `B` and noisiest to measure; E's is not, so a measurement
on E fixes the position-ordering-versus-component-ordering question outright,
which is what C's assignment turns on.

What E does *not* settle is §7.3's other half: its `|b12 − b23| = 0.0104` puts
the `b12 ↔ b23` swap test at **68,100** samples, against 20,000 for the usual
runs. So the search objective has been flipped — healthy facets are now known
to be attainable, so the binding constraint is the asymmetry, and it maximises
asym subject to the facet floor rather than the reverse. The log shows
asym 0.057–0.059 reached with facets `[597, 119, 65]` and `[528, 158, 57]`, i.e.
facet 3 just under the threshold, so a configuration with both looks close.

### 7.5 ATTACKED: the methodological item, which is still the important one

The retracted §3.3 claim survived every internal consistency check because its
two "independent" routes shared the observable that was wrong. The missing check
is not another consistency test but a **dependency audit**: for each claim, list
what its confirmations have in common, and treat a claim whose routes share a
single point of failure as untested in that direction however many routes agree.

`audit_shared.py` records that graph explicitly and reports the intersection per
claim. It is hand-maintained on purpose: what matters is the *mathematical*
dependency — which observable, which convention, which identity — and that is
not visible in the call graph. The single-face observable and the spatial
average share nearly all their code and differ in the one place that mattered.

Run on the claims above it flags the retracted one correctly, naming the shared
observable, using only the dependency table and no knowledge that the observable
was wrong. It also flags two things I had not noticed:

* **`v` irrational is a single-route claim.** 22 digits of precision is not
  independence — a wrong A-period normalisation would give a wrong `v` to 22
  digits just as cheerfully. The genus-1 fixture (`v = 1/4` exactly, known from
  Remark 4.18) is the only external anchor, and it constrains the Abel
  normalisation but not the curve.
* **§5's four genus-3 checks have identical dependency sets.** Atoms,
  correlations, marginal broadening and the A-vs-C falsification test different
  *features* of one model against one measurement, which is worth something, but
  none of them is an independent route.

By contrast `v = −2u(0₊)` has two routes sharing *nothing* — a Monte-Carlo fit
to sampled heights, and an Abel integral on the curve — which is the strongest
structure in these notes.

**It has already caught me twice, in this same pass.** §7.3 above describes a
facet-health proxy I calibrated on the three configurations whose facet sizes I
already knew, found consistent across all three, and then trusted on a fourth —
where it failed outright. Three points, a plausible monotone trend, no held-out
test, and the calibration set was the same set that motivated the proxy. That is
the §3.3 failure mode exactly: a quantity that agrees everywhere it has been
checked, checked only where the answer was already known. The audit's rule —
agreement is evidence only to the extent the routes can fail independently —
applies to a proxy and its calibration set just as it does to two computational
routes.

And again in §7.1: I checked the gas inverse on three offsets along one
diagonal ray, watched the error shrink with `n` at all three, and reported the
route resolved. A sweep over all offsets in `[−5,5]²` shows 27 of 36 wrong, with
errors that do not shrink with `n` at all. Identical shape to the proxy failure
and to §3.3 itself: agreement everywhere it was checked, checked only where
agreement was easy. The lesson is not "check more cases" but "the validation set
must be chosen by someone who is not the person who wrote the formula" — or,
failing that, swept exhaustively over the domain the formula claims, which is
what `gasinv_validate.py` now does.

A third instance, which I then got wrong twice over: I demoted §3.2 from an
independent route on the strength of §3.3's prose, without running the code —
and recorded that demotion here as an error. It was not an error. Running the
code later confirmed the demotion and refuted the "correction". So this entry
was itself a false confession, written in the same reflex as the rest: a
plausible account accepted without printing the object.

A fourth instance, and the worst, because I reversed a correct statement.
I had §3.2's observable right; then "corrected" it on the strength of its
one-sided ratio matching theory to seven digits, without printing the other
side; the distribution is symmetric to seven digits where theory demands a
2000:1 asymmetry. Three of these four failures are literally the same act:
confirm a formula on the slice that is convenient (one diagonal ray, one fixed
black vertex, one side of a distribution, three known configurations) and
generalise. The habit that fixes it is not "check more" but "print the whole
object" — the asymmetry was one line of output away throughout.

A fifth instance, an hour after the fourth: I verified the scale-`2b` identity
at one starting offset, pushed that offset to saturation, matched three atoms to
`1e-14`, and committed it as a statement about the local fluctuation. Other
offsets saturate to different `k = 1` atoms. The committed claim had to be
narrowed to diagonal-class face pairs. The precision of the check (fourteen
digits) was no protection at all, because precision and generality are
independent and I kept spending effort on the first while assuming the second.

What the audit does not do is say which shared dependency is wrong, or that any
is. It converts "two routes agree" into "two routes agree, and here is the one
thing that would fool both", which is the question that had not been asked.

### 7.6 UNTOUCHED: the genus-2 limit shape away from the facets

Only the two gas-facet *locations* are validated against sampling (§4, 0.9σ and
1.5σ). The arctic curve, frozen boundary and liquid-region height are predicted
from the Schottky data and never checked. This needs a genus-2 Fock-weighted
sampling run and a profile comparison; it is the one item here I have not
started. (An earlier version of this bullet denied the facet check too,
contradicting §4; that was my error, not a second result.)
