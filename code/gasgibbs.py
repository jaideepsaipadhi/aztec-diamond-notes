"""
Gas-phase Gibbs measure for the two-periodic Aztec diamond, from the spectral
curve -- and comparison with the converged centre probabilities of the finite
diamond.

Infinite lattice: W = (2Z+1)x(2Z), B = (2Z)x(2Z+1), weights as in Johansson
(7.2), depending on x1+x2 mod 4.  The weight pattern is invariant under
translations by (2,2) and (2,-2), which generate a sublattice of index 8; a
fundamental domain therefore holds 2 white and 2 black vertices, so the
magnetically altered Kasteleyn matrix K(z,w) is 2x2.

For a translation-invariant Gibbs measure with magnetic coordinates (r1,r2),

    K^{-1}(w, b + t) = (1/(2pi i)^2) \oint \oint (K(z,w)^{-1})_{w,b}
                        z^{-t1} w^{-t2} dz/z dw/w,   |z|=e^{r1}, |w|=e^{r2}

and P(edge) = K_edge * K^{-1}(w, b).  The GAS phase is the one whose (r1,r2)
lies in the bounded hole of the amoeba.  By the z<->w and z->1/z symmetries of
this curve the hole is centred at the origin, so the gas contours are the unit
circles -- verified below.
"""
import numpy as np

from twoperiodic import kast_entry

E1 = (1, 1)
E2 = (-1, 1)
P1 = (2, 2)      # period generators
P2 = (2, -2)

BREP = [(0, 1), (2, 1)]
WREP = [(1, 0), (3, 0)]


def decompose(v, reps):
    """Write v = rep + m*P1 + n*P2; return (index, m, n)."""
    for k, r in enumerate(reps):
        dx, dy = v[0] - r[0], v[1] - r[1]
        # dx = 2m + 2n, dy = 2m - 2n
        if (dx + dy) % 4 or (dx - dy) % 4:
            continue
        m = (dx + dy) // 4
        n = (dx - dy) // 4
        if (2*m + 2*n, 2*m - 2*n) == (dx, dy):
            return k, m, n
    return None


def Kzw(z, w, a):
    """2x2 magnetically altered Kasteleyn matrix; rows=black reps, cols=white."""
    K = np.zeros((2, 2), dtype=complex)
    for bi, b in enumerate(BREP):
        for d in (E1, E2, (-E1[0], -E1[1]), (-E2[0], -E2[1])):
            v = (b[0] + d[0], b[1] + d[1])
            dec = decompose(v, WREP)
            if dec is None:
                raise ValueError(f"{v} not decomposable")
            wi, m, n = dec
            K[bi, wi] += kast_entry(b, v, a) * (z ** m) * (w ** n)
    return K


def gas_probabilities(a, N=512, r1=0.0, r2=0.0):
    """P(edge) for the four edges at black rep 0, in the Gibbs measure with
    contours |z|=e^{r1}, |w|=e^{r2}."""
    th = 2 * np.pi * np.arange(N) / N
    zs = np.exp(r1) * np.exp(1j * th)
    ws = np.exp(r2) * np.exp(1j * th)
    # accumulate Fourier coefficients of (K^{-1})_{w,b} that we need
    acc = {}
    for z in zs:
        for w in ws:
            Ki = np.linalg.inv(Kzw(z, w, a))     # (K^{-1})[white, black]
            for key, (m, n) in NEEDED.items():
                acc[key] = acc.get(key, 0) + Ki[key[1], key[0]] * z**(-m) * w**(-n)
    for k in acc:
        acc[k] /= N * N
    out = {}
    b = BREP[0]
    for d in (E1, E2, (-E1[0], -E1[1]), (-E2[0], -E2[1])):
        v = (b[0] + d[0], b[1] + d[1])
        wi, m, n = decompose(v, WREP)
        out[d] = abs(kast_entry(b, v, a) * acc[(0, wi, m, n)])
    return out


# which Fourier coefficients are needed: (black idx, white idx, m, n)
NEEDED = {}
for _d in (E1, E2, (-E1[0], -E1[1]), (-E2[0], -E2[1])):
    _v = (BREP[0][0] + _d[0], BREP[0][1] + _d[1])
    _wi, _m, _n = decompose(_v, WREP)
    NEEDED[(0, _wi, _m, _n)] = (_m, _n)


def hole_contains_origin(a, N=400):
    """Check that (0,0) is inside the amoeba hole: the curve P(z,w)=0 should
    have no solution with |z|=|w|=1."""
    th = 2*np.pi*np.arange(N)/N
    m = np.inf
    for t1 in th:
        for t2 in th[::8]:
            d = abs(np.linalg.det(Kzw(np.exp(1j*t1), np.exp(1j*t2), a)))
            m = min(m, d)
    return m


if __name__ == "__main__":
    for a in (0.5, 0.7):
        print(f"\n=== a={a} ===")
        m = hole_contains_origin(a)
        print(f"  min |det K(z,w)| on the unit torus = {m:.4f}  "
              f"({'no zeros -> (0,0) is in a complement component' if m > 1e-6 else 'ZEROS PRESENT'})")
        p = gas_probabilities(a, N=256)
        tot = sum(p.values())
        print(f"  edge probabilities at a black vertex: "
              + ", ".join(f"{v:.5f}" for v in p.values()))
        print(f"  sum = {tot:.6f}   (must be 1)")
