"""
Where is the Poincare series usable?

Convergence of the period-matrix series is open in general (BBS Section 10);
it is known for sufficiently small |mu| and in the circle-decomposable case.
Their own implementation refuses at mu = (0.08, 0.03) -- the parameters of their
published Figure 21 -- so this is a live practical constraint, not just theory.

This scans genus-2 Schottky data over multiplier size and fixed-point separation,
and records for each point:
    OK      both agree within the truncation tolerance
    DIFFER  both produced a number but they disagree
    JTEM    jtem declined ("too many elements") -- reference unavailable
    OURS    our series failed a structural check (Im B not positive definite)

The U2 form requires A = conj(B), Im A > 0, 0 < mu < 1, and the isometric circles
must be disjoint; BBS give the explicit disjointness inequality

    |(A_n - mu_n conj(A_n))/(1-mu_n) - (A_m - mu_m conj(A_m))/(1-mu_m)|
        > 2 Im A_n/(1-mu_n) + 2 Im A_m/(1-mu_m)

which we check first, since outside it the group is not even a classical Schottky
group and neither computation means anything.
"""
import subprocess

import mpmath as mp

import schottky_full as S

REPO = "/home/claude/FockDimerSimulation"
CLASSES = "/tmp/ref"


def disjoint(A1, mu1, A2, mu2):
    """BBS disjointness condition for the isometric circles."""
    c = lambda A, m: (A - m * mp.conj(A)) / (1 - m)
    r = lambda A, m: 2 * mp.im(A) / (1 - m)
    return abs(c(A1, mu1) - c(A2, mu2)) > r(A1, mu1) + r(A2, mu2)


def jtem(cases):
    lines = []
    for data in cases:
        parts = [str(len(data))]
        for (A, B, mu) in data:
            parts += [str(mp.re(A)), str(mp.im(A)),
                      str(mp.re(B)), str(mp.im(B)), str(mu)]
        lines.append(" ".join(parts))
    out = subprocess.run(
        ["java", "-cp", f"{REPO}/lib/*:{REPO}/src/main/java:{CLASSES}",
         "Reference"],
        input="\n".join(lines), capture_output=True, text=True, timeout=1800)
    return [l for l in out.stdout.strip().splitlines()
            if l.startswith("OK") or l.startswith("FAIL")]


if __name__ == "__main__":
    mp.mp.dps = 25
    seps = [mp.mpf(s) for s in ("0.5", "0.7", "0.9", "1.2", "1.8")]
    mus = [mp.mpf(m) for m in ("0.005", "0.02", "0.05", "0.08", "0.15", "0.25")]

    cases, labels = [], []
    for sep in seps:
        for mu in mus:
            A1 = mp.mpc(sep, "0.4")
            A2 = mp.mpc(-sep, "0.4")
            if not disjoint(A1, mu, A2, mu):
                labels.append((sep, mu, "NOTCLASSICAL"))
                continue
            cases.append([(A1, mp.conj(A1), mu), (A2, mp.conj(A2), mu)])
            labels.append((sep, mu, None))

    ref = jtem(cases)
    ri = 0
    print(f"{'sep':>5} {'mu':>7}  {'status':<14} detail")
    for (sep, mu, pre) in labels:
        if pre:
            print(f"{float(sep):5.1f} {float(mu):7.3f}  {'not classical':<14} "
                  f"isometric circles overlap")
            continue
        line = ref[ri]; ri += 1
        L = 5
        Bm = S.period_matrix([(mp.mpc(sep, "0.4"), mp.mpc(sep, "-0.4")),
                              (mp.mpc(-sep, "0.4"), mp.mpc(-sep, "-0.4"))],
                             [mu, mu], maxlen=L)
        im = [[mp.im(Bm[i, j] * 2j * mp.pi) for j in range(2)] for i in range(2)]
        # our own structural check: Im of the normalized B must be pos. def.
        imn = [[mp.im(Bm[i, j]) for j in range(2)] for i in range(2)]
        posdef = imn[0][0] > 0 and (imn[0][0]*imn[1][1] - imn[0][1]*imn[1][0]) > 0
        if line.startswith("FAIL"):
            print(f"{float(sep):5.1f} {float(mu):7.3f}  {'jtem declined':<14} "
                  f"{'(ours: pos.def. ok)' if posdef else '(ours: also fails)'}")
            continue
        t = line.split(); vals = [float(x) for x in t[2:]]
        R = [[complex(vals[2*(i*2+j)], vals[2*(i*2+j)+1]) for j in range(2)]
             for i in range(2)]
        d = max(abs(complex(Bm[i, j] * 2j * mp.pi) - R[i][j])
                for i in range(2) for j in range(2))
        tol = max(1e-12, 50 * float(mu) ** L)
        print(f"{float(sep):5.1f} {float(mu):7.3f}  "
              f"{'agree' if d < tol else 'DISAGREE':<14} "
              f"dev {d:.1e}  tol {tol:.1e}")
