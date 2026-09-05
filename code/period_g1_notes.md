# Genus-1 period computation — §4.6 of arXiv:2502.07241

Run: `python period_g1.py` (needs only `mpmath`). 30-digit working precision.

## Setup

Collecting the characteristic polynomial (101) as a quadratic in `w`, the discriminant is
`4(C+4)·(z+a²)(z+a⁻²)/z` (verified symbolically). That differs from `z(z+a²)(z+a⁻²)`
by a factor of z², a square, so both give the same curve, branch points and periods:

    y² = z (z + a²) (z + a⁻²),   branch points  e₁ = −a⁻² < e₂ = −a² < e₃ = 0 < ∞.

Sign of the cubic on the real line:

| interval | sign of f | real locus |
|---|---|---|
| (−a⁻², −a²) | + | **compact oval** → A cycle |
| (−a², 0) | − | y imaginary; carries the B cycle |
| (0, ∞) | + | outer (noncompact) oval |

With `ω = dz/(y·I_A)` normalized so `∮_A ω = 1`, we get `B = i·I_B/I_A`.

## Result at a = 0.7

| method | B |
|---|---|
| Legendre, k² = (e₂−e₁)/(e₃−e₁) = 1 − a⁴ | 0.772737661704 i |
| direct quadrature of both real periods | 0.772737661704 i |
| AGM, K′/K = AGM(1,k)/AGM(1,k′) | 0.772737661704 i |
| **§4.6** | **0.521828 i** |

Raw periods: `I_A = 6.08899192457`, `I_B = 4.70519338193`.

## Conventions verified against the paper

§4.6 states the A cycle as the loop over [-a^-2, -a^2], the normalisation
c = ±2∫ over that segment, and the B cycle as the loop containing the cut
[-a^2, 0] for a < 1. These are exactly the choices used here, so the
discrepancy is not a convention gap. The paper also notes B has no closed
form and is evaluated numerically.

## Two things this rules out

1. **Not a cycle swap.** Taking the *other* real cycle as A sends k² → 1 − k² and gives
   `B = 1.29410 i`, which is exactly `1/0.772738` — the modular inversion `B → −1/B`,
   as it must be. It is not `0.521828`. So the discrepancy cannot be explained by my
   having the A and B cycles interchanged, which was the failure mode I most expected.

2. **Not an obvious reparametrization.** `B` depends on `a` only through `a⁴`.
   Solving `B(â) = 0.521828 i` gives `â = 0.4418509`, i.e. `â⁴ = 0.0381156`
   against our `a⁴ = 0.2401`. That value of `â` does not match `a²`, `√a`, `a/(1+a²)`,
   `(1−a²)/(1+a²)`, `a²/(1+a²)`, `(1−a)/(1+a)`, or `a/(1+a)` at `a = 0.7`.
   Note `a² = 0.49` gives `B = 0.5613530 i`, close to 0.521828 but not equal.

Independent confirmation: the dimer model itself. Exact height-difference
distributions from Kasteleyn determinant ratios (no sampling) reproduce
b(a) = K'/K, k² = 1-a⁴ across a ∈ [0.3, 0.8], to six or seven digits where
the gas facet is widest. At a = 0.7 that measurement is 0.128716 against
0.130972 predicted from B = 0.772738i, and 0.049283 from B = 0.521828i.

## Facet-height masses (Remark 4.18 alignments)

g = 1, scale matrix τ = −B⁻¹ = i/b, so `P(n) ∝ exp(−π(n−e)²/b)` with shifts e = 0, ¼, ½, ¾.

| e | with b = 0.772738 (ours) | with b = 0.521828 (§4.6) |
|---|---|---|
| 0 | P(0) = 96.68% | P(0) = 99.52% |
| ¼ | P(0) = 88.24%, P(1) = 11.56% | P(0) = 95.29%, P(1) = 4.70% |
| ½ | P(0) = P(1) = 49.99% | P(0) = P(1) = 50.00% |
| ¾ | P(0) = 11.56%, P(1) = 88.24% | P(0) = 4.70%, P(1) = 95.29% |

The e = ½ alignment is a 50/50 split between heights 0 and 1 under either value of B —
that one is insensitive to the discrepancy and is the cleanest thing to test against a
sampler first.
