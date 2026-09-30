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

Findings 1 and 2 are independent of each other: 2 does not rest on 1, and 1 does
not rest on 2.

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

## 6. What I am least sure of

* **The methodological one, which is the important one.** Every internal
  consistency check I ran passed while the observable in §3.2–§3.4 was wrong, and
  two independent computational routes agreed with each other because they shared
  that observable. What caught it was a person who knew the theory pointing at the
  observable. I do not have a procedure that would have caught it on my own, and
  I should assume the same failure mode is live elsewhere in these notes.
* The shift vector `e` at genus 3 is fitted, not predicted. It differs between
  `n = 96` and `n = 192` — expected, since it depends on how the weight pattern
  aligns with the diamond while `B` does not — but I cannot predict it from the
  divisor, so the genus-3 test constrains `B` with `e` marginalised away rather
  than testing the pair.
* Configuration C's facet 3 has only 175 faces at `n = 192`, which is why its own
  absolute fit is poor (atoms `χ² = 86.8/6`) even though the swap comparison in
  §5.4 is clean. I never found a configuration with three healthy facets *and*
  `B₁₂ ≠ B₂₃` by more than 4.6%, so the two correlations are near-equal
  predictions rather than independently distinguishable ones.
* The `a = 0.7` row of the table in §3.2 is only 0.8% converged by the
  determinant method — though §3.6 now measures that point directly.
* The genus-2 limit shape is not validated against sampling.
* The contour-integral route that would remove the `n ≈ 60` ceiling is
  incomplete. Nothing above depends on it.
