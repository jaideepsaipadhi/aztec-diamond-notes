"""
The period from the Schottky group via theta (Poincare) series.

Genus-1 Schottky group Gamma = <z -> mu z>.  The associated theta series are the
classical Jacobi theta constants in the nome q = sqrt(mu):

    theta_2(q) = 2 sum_{n>=0} q^{(n+1/2)^2}
    theta_3(q) = 1 + 2 sum_{n>=1} q^{n^2}
    theta_4(q) = 1 + 2 sum_{n>=1} (-1)^n q^{n^2}

The curve uniformized by Gamma has modulus  k = theta_2^2 / theta_3^2.  Our curve
y^2 = z(z+a^2)(z+a^-2) has k^2 = 1 - a^4 (from the branch points alone: the
cross-ratio).  So we solve

    theta_2(q)^4 / theta_3(q)^4 = 1 - a^4

for q, and then read off the period directly from the Schottky data:

    b = -log(mu) / (2 pi) = -log(q^2) / (2 pi) = -log(q) / pi.

This uses only geometric series in q -- no elliptic integrals, no AGM, no
quadrature.  It is the Schottky/theta route, and it converges geometrically:
q ~ 0.09 at a = 0.7, so a dozen terms give machine precision.
"""
import mpmath as mp

mp.mp.dps = 40


def theta2(q, N=60):
    return 2 * mp.fsum(q ** ((n + mp.mpf(1)/2) ** 2) for n in range(N))


def theta3(q, N=60):
    return 1 + 2 * mp.fsum(q ** (n ** 2) for n in range(1, N))


def theta4(q, N=60):
    return 1 + 2 * mp.fsum((-1) ** n * q ** (n ** 2) for n in range(1, N))


def k2_from_q(q):
    return (theta2(q) / theta3(q)) ** 4


def b_schottky(a):
    """Solve k^2(q) = 1 - a^4 for the Schottky nome, then read off b."""
    a = mp.mpf(a)
    target = 1 - a ** 4
    f = lambda q: k2_from_q(q) - target
    # bracket: k^2 -> 0 as q -> 0, k^2 -> 1 as q -> 1
    q = mp.findroot(f, mp.mpf('0.05'))
    mu = q ** 2
    return -mp.log(q) / mp.pi, q, mu


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/mnt/user-data/outputs")
    from period_g1 import b_legendre

    print("Schottky/theta route vs the elliptic-integral route\n")
    print("   a      q (nome)      mu = q^2       b (Schottky)     "
          "b (Legendre)     difference")
    for a in ("0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"):
        b, q, mu = b_schottky(a)
        ref = b_legendre(a)
        print(f"  {a}   {mp.nstr(q, 8):<12}  {mp.nstr(mu, 8):<13} "
              f"{mp.nstr(b, 12):<16} {mp.nstr(ref, 12):<16} "
              f"{mp.nstr(abs(b-ref), 3)}")

    print("\nat a = 0.7, the disputed value:")
    b, q, mu = b_schottky("0.7")
    print(f"   Schottky multiplier mu = {mp.nstr(mu, 12)}")
    print(f"   B = i * {mp.nstr(b, 12)}")
    print(f"   paper's value would need mu = "
          f"{mp.nstr(mp.e**(-2*mp.pi*mp.mpf('0.521828')), 12)}")
