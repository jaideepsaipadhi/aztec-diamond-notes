"""
Regression fixture: our Schottky period matrix vs the jtem reference.

Why this exists.  Across this project, checks against our own expectations
repeatedly passed while the code was wrong -- a dual path validated against a
brute-force enumeration of the same wrong statistic; an alignment read off
histograms taken from an arbitrary reference face; a higher-genus period matrix
"validated" by a decoupling limit that the bug itself produced.  Structural
checks (symmetry, positive-definite Im B, |phi| <= 1, non-negativity) never
caught a real error here and twice gave false assurance.  Independent
implementations did.

Setup (once):
    apt-get install -y openjdk-21-jdk-headless
    git clone --depth 1 https://github.com/nikolaibobenko/FockDimerSimulation.git
    javac -nowarn -cp "FockDimerSimulation/lib/*:FockDimerSimulation/src/main/java" \
          -d /tmp/ref Reference.java

Convention: jtem's getPeriodMatrix() returns 2*pi*i times our normalized B.
"""
import subprocess

import mpmath as mp

import schottky_full as S

REPO = "/home/claude/FockDimerSimulation"
CLASSES = "/tmp/ref"

CASES = [
    # (label, [(A, B, mu), ...])
    ("g1 mu=0.02",   [(mp.mpc('0.1', '1.0'), mp.mpc('0.1', '-1.0'), '0.02')]),
    ("g1 mu=0.005",  [(mp.mpc('0.1', '1.0'), mp.mpc('0.1', '-1.0'), '0.005')]),
    ("g1 mu=0.0077873557",
     [(mp.mpc('0.25', '0.9'), mp.mpc('0.25', '-0.9'), '0.0077873557')]),
    ("g2 mu=(0.01,0.005)",
     [(mp.mpc('1.2', '1.3'), mp.mpc('1.2', '-1.3'), '0.01'),
      (mp.mpc('-0.4', '0.6'), mp.mpc('-0.4', '-0.6'), '0.005')]),
    ("g2 mu=(0.02,0.01)",
     [(mp.mpc('1.2', '1.3'), mp.mpc('1.2', '-1.3'), '0.02'),
      (mp.mpc('-0.4', '0.6'), mp.mpc('-0.4', '-0.6'), '0.01')]),
    ("g2 mu=(0.03,0.02)",
     [(mp.mpc('1.5', '1.5'), mp.mpc('1.5', '-1.5'), '0.03'),
      (mp.mpc('-1.0', '1.0'), mp.mpc('-1.0', '-1.0'), '0.02')]),
]


def reference(cases):
    lines = []
    for _, data in cases:
        g = len(data)
        parts = [str(g)]
        for (A, B, mu) in data:
            parts += [str(mp.re(A)), str(mp.im(A)),
                      str(mp.re(B)), str(mp.im(B)), str(mu)]
        lines.append(" ".join(parts))
    out = subprocess.run(
        ["java", "-cp", f"{REPO}/lib/*:{REPO}/src/main/java:{CLASSES}",
         "Reference"],
        input="\n".join(lines), capture_output=True, text=True, timeout=600)
    return out.stdout.strip().splitlines()


def ours(data, maxlen):
    fixed = [(A, B) for (A, B, _) in data]
    mus = [mp.mpf(m) for (_, _, m) in data]
    Bm = S.period_matrix(fixed, mus, maxlen=maxlen)
    g = len(data)
    return [[Bm[i, j] * 2j * mp.pi for j in range(g)] for i in range(g)]


if __name__ == "__main__":
    mp.mp.dps = 25
    ref = reference(CASES)
    print(f"{'case':<26} {'status':<8} max |ours - jtem|")
    worst = 0.0
    for (label, data), rline in zip(CASES, ref):
        if rline.startswith("FAIL"):
            print(f"{label:<26} {'jtem FAIL':<8} {rline[5:][:40]}")
            continue
        t = rline.split()
        g = int(t[1])
        vals = [float(x) for x in t[2:]]
        R = [[complex(vals[2*(i*g+j)], vals[2*(i*g+j)+1]) for j in range(g)]
             for i in range(g)]
        L = 5 if g > 1 else 1
        O = ours(data, maxlen=L)
        d = max(abs(complex(O[i][j]) - R[i][j])
                for i in range(g) for j in range(g))
        # The Poincare series is truncated at word length L, so the residual is
        # truncation, not disagreement. It falls geometrically in L at a rate set
        # by the multipliers, so the tolerance must scale with mu and L rather
        # than being a flat constant. (g2 mu=(0.03,0.02) sits at 1.45e-8 at L=5
        # and converges to jtem by ~16x per extra word: 2.4e-7, 1.5e-8, 8.9e-10.)
        mumax = max(float(m) for (_, _, m) in data)
        tol = max(1e-12, 50 * mumax ** L)
        status = 'ok' if d < tol else 'MISMATCH'
        worst = max(worst, d / tol)
        print(f"{label:<26} {status:<8} {d:.2e}   (tol {tol:.1e}, L={L})")
    print(f"\nworst deviation relative to tolerance: {worst:.2f}x"
          f"   {'ALL PASS' if worst <= 1 else 'FAILURES PRESENT'}")
