"""
The local gas fluctuation at n = infinity, from the curve -- a second route to
the §3.3 retraction's own diagnosis.

WHAT THE RETRACTION CLAIMS.  §3.2 and §3.4 measured the height at a SINGLE face
of the gas facet, which is the discrete component plus the local gas fluctuation
at that face.  The retraction attributes the factor of 2 to that contamination,
stating it is "a near-symmetric displacement of one face by one unit, with mass
roughly 0.05 at a = 0.5 and 0.11 at a = 0.7 on each side".

That diagnosis currently rests on sampling, and on sampling alone -- the same
dependency (generalised shuffling + the facet observable) that the retracted
claim itself rested on.  audit_shared.py flags exactly this pattern.  So it is
worth computing independently.

WHY IT IS COMPUTABLE NOW.  gasinv.py's full-plane gas inverse was wrong until
the exponents were put in fundamental-domain units; with that fixed it
reproduces the exact finite-n inverse and the error shrinks with n at every
offset.  And at n = infinity the gas phase fills the plane, so ANY finite dual
path lies entirely in the gas region -- which is exactly the regime the
full-plane inverse describes, with no facet-width constraint and no finite-size
error.  So the same determinant-lemma machinery as lowrank.py applies, with M
built from the gas inverse instead of a sparse solve:

    det K(theta)/det K(0) = det(I + D(theta) M),   M_kl = K^{-1}[w_k, b_l]

over the edges crossed by a dual path joining two faces.  Both faces sit in the
gas phase, so this is the height difference between two faces of the SAME facet,
in which the discrete component cancels identically and only the local
fluctuation survives.  For well-separated faces the fluctuations are nearly
independent, so the difference distribution is close to the self-convolution of
the single-face fluctuation, and its variance is 2 Var[phi].

WHAT THIS CAN AND CANNOT SETTLE.  It gives the contamination without sampling
and without the facet observable, so it is a genuinely independent route to the
retraction's diagnosis.  It does NOT re-derive the §3.2 table: that table's dual
path runs from the facet out to the frozen boundary and crosses the liquid
region, where the gas inverse does not apply, so the a = 0.7 row cannot be
converged this way.  (That row is in any case not a convergence problem -- see
the note in NOTES.md §7.1.)

Run:  python3 gas_fluct.py
"""
import sys

import numpy as np

import gasinv as G
from twoperiodic import E1, E2, vertices

CLS = G.cls


def is_black(v):
    """twoperiodic.vertices: black is (even, odd), white is (odd, even)."""
    return v[0] % 2 == 0 and v[1] % 2 == 1


def is_white(v):
    return v[0] % 2 == 1 and v[1] % 2 == 0


def path_edges(L, x2=1, x1_0=0):
    """Edges crossed by a straight dual ray of L steps on the INFINITE lattice.

    Mirrors exactdist.dual_path_edges, with the boundary and the vertex lists
    removed: on the full plane the colour classes are decided arithmetically
    rather than by membership of a finite vertex set.

    Note the colour rule is NOT a parity of x1 + x2.  The edge directions are
    E1 = (1,1) and E2 = (-1,1), which change x1 + x2 by 2 or 0, so that parity
    is constant on each sublattice and a parity test never fires at all.  The
    ray therefore sits at ODD x2, where only EVEN x1 carry vertices, and each
    such x1 contributes two crossed edges.
    """
    crossed = []
    for x1 in range(x1_0, x1_0 + L):
        for (u, v) in (((x1, x2), (x1 + 1, x2 + 1)),
                       ((x1, x2), (x1 - 1, x2 + 1))):
            if (is_black(u) and is_white(v)) or (is_white(u) and is_black(v)):
                b, w = (u, v) if is_black(u) else (v, u)
                s = 1 if (b[0] + b[1]) % 4 == 1 else -1
                crossed.append((b, w, s))
    return crossed


def kast_entry_inf(b, w, a):
    """K(b, w) on the full plane, in gasinv's gauge."""
    d = (w[0] - b[0], w[1] - b[1])
    j = CLS(np.asarray(b))
    if d == tuple(E1):
        return (a if j == 0 else 1.0)
    if d == (-E1[0], -E1[1]):
        return (1.0 if j == 0 else a)
    if d == tuple(E2):
        return 1j * (1.0 if j == 0 else a)
    if d == (-E2[0], -E2[1]):
        return 1j * (a if j == 0 else 1.0)
    return 0.0


def pmf_inf(L, a, Mpts=64, N=600):
    """Atom probabilities of the height difference across a length-L gas path."""
    crossed = path_edges(L)
    r = len(crossed)
    if r == 0:
        return None
    M = np.empty((r, r), dtype=complex)
    d0 = np.empty(r, dtype=complex)
    cache = {}
    for k, (b, w, s) in enumerate(crossed):
        d0[k] = kast_entry_inf(b, w, a)
        for l, (b2, w2, s2) in enumerate(crossed):
            key = (w[0] - b2[0], w[1] - b2[1], CLS(np.asarray(w)),
                   CLS(np.asarray(b2)))
            if key not in cache:
                cache[key] = G.gas_inverse(w, b2, a, N=N)
            M[k, l] = cache[key]
    signs = np.array([s for (_, _, s) in crossed])
    phi = np.empty(Mpts, dtype=complex)
    for k in range(Mpts):
        th = 2 * np.pi * k / Mpts
        D = d0 * (np.exp(-4j * th * signs) - 1.0)
        phi[k] = np.linalg.det(np.eye(r) + D[:, None] * M)
    return np.fft.ifft(phi).real


def summarise(p, step=4):
    """Atoms on the height lattice (spacing `step`), centred on the mode."""
    idx = np.array([i for i, v in enumerate(p) if abs(v) > 1e-7])
    if idx.size == 0:
        return None
    vals = p[idx]
    c = idx[int(np.argmax(vals))]
    out = {}
    for i, v in zip(idx, vals):
        k = int(round((i - c) / step))
        out[k] = out.get(k, 0.0) + float(v)
    return out


if __name__ == "__main__":
    print("local gas fluctuation on the full plane, from the fixed gas inverse")
    print("(height difference between two faces of the same facet, so the")
    print(" discrete component cancels and only the fluctuation remains)\n")
    for a in (0.5, 0.7):
        print(f"  a = {a}   retraction claims per-side mass ~"
              f"{0.05 if a == 0.5 else 0.11}")
        for L in (2, 4, 6, 8):
            p = pmf_inf(L, a)
            if p is None:
                continue
            s = summarise(p)
            tot = sum(s.values())
            var = sum(k * k * v for k, v in s.items()) / tot \
                - (sum(k * v for k, v in s.items()) / tot) ** 2
            side = sum(v for k, v in s.items() if k != 0) / tot / 2
            shown = {k: round(v / tot, 5) for k, v in sorted(s.items())
                     if abs(v / tot) > 1e-4}
            print(f"    L={L:<3} atoms {shown}")
            print(f"         sum {tot:.6f}   Var {var:.5f}"
                  f"   per-side mass {side:.5f}")
        print()
