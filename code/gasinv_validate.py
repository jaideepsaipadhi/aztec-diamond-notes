"""
Broad-offset validation of the full-plane gas inverse -- the check that should
have existed before anything was built on it.

HISTORY, because it is the point of this file.  gasinv.py's own __main__
compared gas_inverse against the exact finite-n inverse at offsets (1,1), (3,1),
(1,3), (3,3), (5,5) and the two anti-diagonal ones -- but (3,1), (1,3) and the
anti-diagonals came back ZERO in both columns (wrong sublattice parity), so the
comparison was effectively a single diagonal ray.  On that ray the original code
had the wrong sign and roughly double the true decay rate; putting the exponents
in fundamental-domain units (u//2) fixed the ray, and the error then shrank
monotonically with n at (1,1), (3,3), (5,5).  That was reported as the route
being resolved.

It is not.  Sweeping ALL offsets in [-5,5]^2 instead of one ray:

    27 of 36 non-trivial offsets are wrong by more than 10% at n = 48,
    and their errors DO NOT shrink with n
    ((1,3): 1.88 -> 2.11 from n=32 to 48;  (3,1): 0.884 -> 0.887;
     (3,-3): 3.74 -> 3.71;  (5,-5): 3.50 -> 3.35)

while the nine that work are exactly the ones on or beside the tested ray
((3,3) 2.2%, (5,5) 3.8%, (1,5) 2.3%, (-3,5) 2.8%, (5,1) 5.8%, (3,-1) 6.1%).

So u//2 is not the fix either -- it repairs the ray and leaves most of the
lattice wrong. A plausible mechanism is that floor division drops the
within-domain half-step instead of absorbing it into the sublattice class, but
that is NOT established: the obvious consequence, distinct offsets collapsing
onto one (u//2, v//2, class) key, does not occur (0 collisions over the same
sweep). The cause is open.

WHAT TO TRUST.  Nothing downstream of gas_inverse, at any offset off the
diagonal ray. gas_fluct.py is built on it and its numbers are therefore void.
The honest statement of the underlying route is still Chhita-Johansson Theorem
2.6; what is unimplemented is the fundamental-domain bookkeeping, where the
translation must split as

    y - x = (whole-domain translation) + (offset within the domain)

with the first giving z^u w^v and the second selecting the entry of
K(z,w)^{-1}. Neither `u` nor `u//2` does that split.

THE RULE THIS FILE ENCODES.  A formula is validated on the set of inputs it was
tested on and nowhere else. Both times something was over-claimed in this
project -- the facet-health proxy and this -- the pattern was identical: check a
handful of cases, find agreement, generalise. Run this sweep, not a ray.

Run:  python3 gasinv_validate.py [a] [n1,n2]
"""
import sys

import numpy as np

import gasinv as G
from twoperiodic import kasteleyn, vertices

THRESH = 0.10


def sweep(a=0.7, ns=(32, 48), R=5, N=600):
    rows = {}
    for n in ns:
        W, B = vertices(n)
        wi = {v: i for i, v in enumerate(W)}
        bi = {v: i for i, v in enumerate(B)}
        Kinv = np.linalg.inv(kasteleyn(n, a))
        c = min(B, key=lambda t: (t[0] - n) ** 2 + (t[1] - n) ** 2)
        for d1 in range(-R, R + 1):
            for d2 in range(-R, R + 1):
                w0 = (c[0] + d1, c[1] + d2)
                if w0 not in wi:
                    continue
                ex = Kinv[wi[w0], bi[c]]
                gf = G.gas_inverse(w0, c, a, N=N)
                if abs(ex) < 1e-10 and abs(gf) < 1e-10:
                    continue
                rows.setdefault((d1, d2), {})[n] = (
                    ex, gf, abs(ex - gf) / max(abs(ex), 1e-30))
    return rows


def report(a=0.7, ns=(32, 48)):
    rows = sweep(a, ns)
    full = {k: v for k, v in rows.items() if all(n in v for n in ns)}
    print(f"gas_inverse vs the exact finite-n inverse, a = {a}\n")
    print(f"{'offset':>9} {'|exact|':>10} {'|gas|':>10} "
          + " ".join(f"{'rel n='+str(n):>11}" for n in ns))
    bad, worse = [], []
    for k in sorted(full, key=lambda t: (abs(t[0]) + abs(t[1]), t)):
        v = full[k]
        ex, gf, _ = v[ns[0]]
        errs = [v[n][2] for n in ns]
        flag = ""
        if errs[-1] > THRESH:
            bad.append(k)
            flag = " <-- WRONG"
            if errs[-1] >= errs[0] - 1e-12:
                worse.append(k)
                flag = " <-- WRONG, not improving with n"
        print(f"{str(k):>9} {abs(ex):10.6f} {abs(gf):10.6f} "
              + " ".join(f"{e:11.4f}" for e in errs) + flag)
    print(f"\n{len(bad)} of {len(full)} offsets wrong by more than "
          f"{THRESH:.0%} at n = {ns[-1]}")
    print(f"{len(worse)} of those do not improve with n, so the discrepancy is "
          f"not the O(e^-cm) term")
    if len(bad) == 0:
        print("\nAll offsets agree -- gas_inverse may be used downstream.")
    else:
        print("\nDo NOT use gas_inverse downstream. Anything built on it "
              "(gas_fluct.py) is void.")
    return rows


if __name__ == "__main__":
    a = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
    ns = tuple(int(x) for x in sys.argv[2].split(',')) \
        if len(sys.argv) > 2 else (32, 48)
    report(a, ns)
