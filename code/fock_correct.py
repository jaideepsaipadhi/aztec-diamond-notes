"""
Fock's weights, CORRECTED and validated against Boutillier-de Tiliere
(arXiv:2405.20284).  This replaces fockweights.py, which was wrong.

WHAT WAS WRONG.  fockweights.face_weight invented a FACE quantity built from
four guessed theta arguments.  No such object exists in the construction.  It
produced weights varying by 1.2% where the comparable two-periodic model varies
by a factor of 4, and that propagated through sigma, the LP and the limit shape
before the sampler comparison caught it.  It survived because endtoend.py
compared the shuffler against a Kasteleyn matrix built from the SAME weights --
self-consistency, not validation.

THE CONSTRUCTION.  Fock's weights are EDGE weights:

    K_{w,b} = E(alpha, beta) / [ theta(t + d(f)) theta(t + d(f')) ]

where the edge wb is crossed by train-tracks with angles alpha, beta, and f, f'
are the two faces adjacent to that edge.  Four train-track families
alpha, beta (horizontal) and gamma, delta (vertical) with cyclic order
alpha < gamma < beta < delta.  Genus 1: theta = theta_3(pi u | tau),
E(u,v) = theta_1(pi(v-u) | tau) up to a constant (gauge).

THE DISCRETE ABEL MAP IS DERIVED, NOT POSITED.  d changes by the crossing
train-track angle along each quad-graph edge, so the four faces around a white
vertex carry d(w) + alpha, beta, gamma, delta.  With alpha=0, gamma=rho,
beta=1/2, delta=1/2+rho that gives d(f) in {0, 1/2, rho, 1/2+rho}, which is
exactly the set appearing in the paper's formulas:

    even face (2p, 2q):      d = (q - p)/2
    odd  face (2p+1, 2q+1):  d = (q - p)/2 + 1/2 + rho

ORIENTATION.  The alternating product around a face is taken in the sense that
makes W_f match the paper; the opposite cyclic order gives 1/W_f.

VALIDATION (self-test below): all four of the paper's genus-1 face-weight
identities -- 1/a^2, 1/a^2, (ab)^2, a^2/b^2 with
a = theta_2(pi rho)/theta_1(pi rho) and
b = theta_3(pi(rho+1/4))/theta_4(pi(rho+1/4)) -- reproduced to 1e-16 at five
different (rho, tau).
"""
import mpmath as mp

mp.mp.dps = 20


def theta(j, u, tau, N=45):
    q = mp.e ** (1j * mp.pi * tau)
    if j == 1:
        return 2*mp.fsum((-1)**n * q**((n+mp.mpf(1)/2)**2) * mp.sin((2*n+1)*u)
                         for n in range(N))
    if j == 2:
        return 2*mp.fsum(q**((n+mp.mpf(1)/2)**2) * mp.cos((2*n+1)*u)
                         for n in range(N))
    if j == 3:
        return 1 + 2*mp.fsum(q**(n*n) * mp.cos(2*n*u) for n in range(1, N))
    if j == 4:
        return 1 + 2*mp.fsum((-1)**n * q**(n*n) * mp.cos(2*n*u)
                             for n in range(1, N))
    raise ValueError(j)


class FockGenus1:
    """Genus-1 Fock weights for GENERAL train-track angles alpha < gamma < beta
    < delta on A_0 = R/Z.  The 2x2 periodic case is the specialisation
    alpha=0, gamma=rho, beta=1/2, delta=1/2+rho, which is what the self-test
    validates against the paper's four identities."""

    def __init__(self, rho=None, tau=None, t=mp.mpf(1)/4, angles=None):
        self.tau, self.t = tau, mp.mpf(t)
        if angles is not None:
            self.al, self.ga, self.be, self.de = [mp.mpf(x) for x in angles]
            self.rho = self.ga - self.al
        else:
            self.rho = mp.mpf(rho)
            self.al, self.ga = mp.mpf(0), self.rho
            self.be, self.de = mp.mpf(1)/2, mp.mpf(1)/2 + self.rho

    def E(self, x, y):
        return theta(1, mp.pi*(y - x), self.tau)

    def TH(self, u):
        return theta(3, mp.pi*u, self.tau)

    def d_face(self, fx, fy):
        """Discrete Abel map on faces, GENERAL ANGLES.

        Derived from d(w) = q(beta-alpha) + p(gamma-delta) - alpha and the rule
        that the four faces around a white vertex carry d(w) + alpha (south),
        + beta (north), + gamma (EAST), + delta (WEST).  East/west matters: with
        them swapped the map is inconsistent (the east face of one vertex is the
        west face of its right neighbour, and the two routes must agree).  The
        2x2 case hides this because gamma - delta = -1/2 collapses mod 1.
        """
        al, ga, be, de = self.al, self.ga, self.be, self.de
        if fx % 2 == 0:
            p, q = fx // 2, fy // 2
            return q*(be - al) + p*(ga - de)
        p, q = (fx - 1) // 2, (fy - 1) // 2
        return q*(be - al) + p*(ga - de) - al + ga

    def prime_ratio(self):
        al, ga, be, de = self.al, self.ga, self.be, self.de
        return abs(complex((self.E(al, ga)*self.E(be, de))
                           / (self.E(ga, be)*self.E(de, al))))

    def face_weight(self, fx, fy):
        """Gauge-invariant face weight (the theta(t+d(f)) factors cancel)."""
        pf = self.prime_ratio()
        nb = [(fx+1, fy+1), (fx-1, fy+1), (fx-1, fy-1), (fx+1, fy-1)]
        ds = [self.d_face(*n) for n in nb]
        thp = abs(complex((self.TH(self.t+ds[0])*self.TH(self.t+ds[2]))
                          / (self.TH(self.t+ds[1])*self.TH(self.t+ds[3]))))
        w = (pf*thp) if (fx % 2 == 0) else (thp/pf)
        return 1.0 / w                      # orientation convention

    def ab(self):
        """The equivalent biased 2x2 periodic parameters."""
        a = mp.re(theta(2, mp.pi*self.rho, self.tau)
                  / theta(1, mp.pi*self.rho, self.tau))
        b = mp.re(theta(3, mp.pi*(self.rho+mp.mpf(1)/4), self.tau)
                  / theta(4, mp.pi*(self.rho+mp.mpf(1)/4), self.tau))
        return a, b


if __name__ == "__main__":
    print("self-test: all four genus-1 face-weight identities\n")
    print("   rho   tau      a        b        max rel. error")
    worst = 0.0
    for rho, bim in (('0.15', '0.35'), ('0.20', '0.6'), ('0.25', '0.75'),
                     ('0.30', '0.9'), ('0.35', '1.2')):
        F = FockGenus1(rho, 1j*mp.mpf(bim))
        a, b = F.ab()
        tgt = sorted([float(1/a**2), float(1/a**2),
                      float((a*b)**2), float(a**2/b**2)])
        got = sorted([F.face_weight(fx, fy)
                      for fx, fy in ((0, 0), (0, 2), (1, 1), (1, 3))])
        err = max(abs(g-t)/t for g, t in zip(got, tgt))
        worst = max(worst, err)
        print(f"   {rho}  {bim}i   {float(a):7.4f}  {float(b):7.4f}  {err:.2e}")
    print(f"\n   worst: {worst:.2e}")
    F = FockGenus1('0.20', 1j*mp.mpf('0.6'))
    w = [F.face_weight(fx, fy) for fx, fy in ((0, 0), (0, 2), (1, 1), (1, 3))]
    print(f"\n   face weights: {[round(x,5) for x in w]}")
    print(f"   variation ratio {max(w)/min(w):.3f}   "
          f"(old broken version gave 1.012)")
