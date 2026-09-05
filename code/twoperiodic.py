"""
The two-periodic Aztec diamond, with weights taken from Johansson's explicit
Kasteleyn matrix (arXiv:1704.06035, eq. (7.2)) -- no guessing.

Vertices of the Aztec diamond graph (size n, with n divisible by 4):
    W = {(x1,x2) in (2Z+1) x (2Z) : 1 <= x1 <= 2n-1, 0 <= x2 <= 2n}
    B = {(x1,x2) in (2Z) x (2Z+1) : 0 <= x1 <= 2n, 1 <= x2 <= 2n-1}
    B_j = {x in B : x1+x2 mod 4 = 2j+1},  same for W_j
    e1 = (1,1),  e2 = (-1,1)

Kasteleyn matrix, for x in B_j:
    y = x + e1 :  a(1-j) + j          y = x - e1 :  aj + (1-j)
    y = x + e2 : (aj + (1-j)) i       y = x - e2 : (a(1-j) + j) i

The i's are Kasteleyn signs; the DIMER WEIGHT is the modulus.  So the real
edge weight for x in B_j is
    x + e1 :  a(1-j) + j             x - e1 :  aj + (1-j)
    x + e2 :  aj + (1-j)             x - e2 :  a(1-j) + j
i.e. the two "diagonal" directions e1 and -e2 carry one value, and -e1, e2 the
other, with the roles swapped between B_0 and B_1.  That is the two-periodicity.

Validation: |det K| must equal the weighted sum over all perfect matchings.
At a = 1 both must equal 2^{n(n+1)/2}.
"""
import numpy as np

E1 = (1, 1)
E2 = (-1, 1)


def vertices(n):
    W = [(x1, x2) for x1 in range(1, 2*n, 2) for x2 in range(0, 2*n + 1, 2)]
    B = [(x1, x2) for x1 in range(0, 2*n + 1, 2) for x2 in range(1, 2*n, 2)]
    return W, B


def bclass(x):
    return 0 if (x[0] + x[1]) % 4 == 1 else 1


def kast_entry(x, y, a):
    """Complex Kasteleyn entry for black x -> white y, per (7.2); 0 if no edge."""
    j = bclass(x)
    d = (y[0] - x[0], y[1] - x[1])
    if d == E1:
        return a*(1 - j) + j
    if d == E2:
        return (a*j + (1 - j)) * 1j
    if d == (-E1[0], -E1[1]):
        return a*j + (1 - j)
    if d == (-E2[0], -E2[1]):
        return (a*(1 - j) + j) * 1j
    return 0


def edge_weight(x, y, a):
    """Real dimer weight = modulus of the Kasteleyn entry."""
    return abs(kast_entry(x, y, a))


def kasteleyn(n, a):
    W, B = vertices(n)
    wi = {v: k for k, v in enumerate(W)}
    K = np.zeros((len(B), len(W)), dtype=complex)
    for r, x in enumerate(B):
        for d in (E1, E2, (-E1[0], -E1[1]), (-E2[0], -E2[1])):
            y = (x[0] + d[0], x[1] + d[1])
            if y in wi:
                K[r, wi[y]] = kast_entry(x, y, a)
    return K


def enumerate_weighted(n, a):
    """Brute-force sum over perfect matchings of the product of edge weights."""
    W, B = vertices(n)
    Wset = set(W)
    nbrs = {}
    for x in B:
        nbrs[x] = [(x[0] + d[0], x[1] + d[1]) for d in
                   (E1, E2, (-E1[0], -E1[1]), (-E2[0], -E2[1]))
                   if (x[0] + d[0], x[1] + d[1]) in Wset]
    Bs = sorted(B)
    total = [0.0]
    used = set()

    def rec(i, wprod):
        if i == len(Bs):
            total[0] += wprod
            return
        x = Bs[i]
        for y in nbrs[x]:
            if y not in used:
                used.add(y)
                rec(i + 1, wprod * edge_weight(x, y, a))
                used.discard(y)

    rec(0, 1.0)
    return total[0]


def face_gauge_invariants(n, a):
    """Alternating product around each 4-cycle face of the Aztec diamond graph.
    Faces are centred at points with both coordinates odd or both even."""
    W, B = vertices(n)
    Wset, Bset = set(W), set(B)
    out = {}
    for cx in range(1, 2*n, 1):
        for cy in range(1, 2*n, 1):
            # a face has 4 corners at c +- e1, c +- e2 alternating B,W
            corners = [(cx + E1[0], cy + E1[1]), (cx + E2[0], cy + E2[1]),
                       (cx - E1[0], cy - E1[1]), (cx - E2[0], cy - E2[1])]
            if not all(c in Wset or c in Bset for c in corners):
                continue
            # edges around the face, in cyclic order
            cyc = [(corners[0], corners[1]), (corners[1], corners[2]),
                   (corners[2], corners[3]), (corners[3], corners[0])]
            ws = []
            ok = True
            for (u, v) in cyc:
                if u in Bset and v in Wset:
                    e = edge_weight(u, v, a)
                elif v in Bset and u in Wset:
                    e = edge_weight(v, u, a)
                else:
                    ok = False
                    break
                if e == 0:
                    ok = False
                    break
                ws.append(e)
            if ok:
                out[(cx % 4, cy % 4)] = round(ws[0]*ws[2]/(ws[1]*ws[3]), 9)
    return out


if __name__ == "__main__":
    n = 4
    print(f"Aztec diamond size n={n} (n must be divisible by 4)")
    W, B = vertices(n)
    print(f"  |W|={len(W)}  |B|={len(B)}  (must match)")

    for a in (1.0, 0.7, 0.5):
        det = abs(np.linalg.det(kasteleyn(n, a)))
        enum = enumerate_weighted(n, a)
        ok = "ok" if abs(det - enum) < 1e-6 * max(1, enum) else "MISMATCH"
        extra = f"   (2^{n*(n+1)//2} = {2**(n*(n+1)//2)})" if a == 1.0 else ""
        print(f"  a={a}:  |det K|={det:14.6f}   enumerated={enum:14.6f}   {ok}{extra}")

    print("\nface gauge invariants (alternating product), a=0.7")
    inv = face_gauge_invariants(n, 0.7)
    vals = sorted(set(inv.values()))
    print(f"  distinct values: {vals}")
    print(f"  -> {'NON-TRIVIAL (gas phase possible)' if len(vals) > 1 else 'GAUGE-TRIVIAL'}")
