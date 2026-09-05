"""
Genus-2 Fock face weights, from BBS (arXiv:2407.19462) equation (10).

The blocker I thought existed does not.  The Fock EDGE weight needs the prime
form E(alpha,beta) = theta[Delta](int_alpha^beta omega) / (h_Delta(alpha)
h_Delta(beta)), and the holomorphic spinors h_Delta are genuinely hard.  But the
FACE weight is

    W_f = prod_i  theta[Delta](int_{alpha_i}^{beta_i} omega)
                / theta[Delta](int_{beta_i}^{alpha_{i+1}} omega)
                * theta(eta(f_2i) + D) / theta(eta(f_2i-1) + D)

and each train-track point occurs once in a numerator and once in a denominator
around the face, so every h_Delta CANCELS.  Only theta with an odd characteristic
survives, which for genus 2 is a lattice sum over Z^2.

Genus 1 is the consistency check: there the odd characteristic gives theta_1,
which is exactly what fock_correct.py uses and which reproduces the paper's four
face-weight identities to 1e-16.

Ingredients, all previously validated:
  * period matrix B by Poincare series over double cosets (checked against jtem)
  * Abel map A(P) = int omega, likewise by Poincare series
  * theta[Delta]: all six odd characteristics vanish at 0 to ~1e-21
  * the discrete Abel map eta, whose increments come from train-track crossings
"""
import itertools

import mpmath as mp
import numpy as np

import schottky_full as S
from genus2_amoeba import group, iso_circle

mp.mp.dps = 18

HALF = mp.mpf(1)/2


def odd_characteristics(g=2):
    """Delta = [Delta1; Delta2] with 4<Delta1,Delta2> = 1 mod 2."""
    out = []
    for d in itertools.product([0, 1], repeat=2*g):
        D1 = [d[i]*HALF for i in range(g)]
        D2 = [d[g+i]*HALF for i in range(g)]
        if int(4*sum(D1[i]*D2[i] for i in range(g))) % 2 == 1:
            out.append((D1, D2))
    return out


def theta_char(z, B, D1, D2, N=6):
    """BBS (6): theta[Delta](z) for genus 2."""
    tot = mp.mpc(0)
    for m1 in range(-N, N+1):
        for m2 in range(-N, N+1):
            m = [m1 + D1[0], m2 + D1[1]]
            q = (m[0]*(B[0, 0]*m[0] + B[0, 1]*m[1])
                 + m[1]*(B[1, 0]*m[0] + B[1, 1]*m[1]))
            lin = (z[0] + D2[0])*m[0] + (z[1] + D2[1])*m[1]
            tot += mp.e ** (1j*mp.pi*q + 2j*mp.pi*lin)
    return tot


def theta(z, B, N=6):
    return theta_char(z, B, [mp.mpf(0)]*2, [mp.mpf(0)]*2, N)


class Genus2Fock:
    def __init__(self, fixed, mus, tracks, maxlen=3):
        self.fixed, self.mus, self.tracks = fixed, mus, tracks
        Bm = S.period_matrix(fixed, mus, maxlen=5)
        self.B = mp.matrix(2, 2)
        for i in range(2):
            for j in range(2):
                self.B[i, j] = Bm[i, j]
        self.D1, self.D2 = odd_characteristics()[0]
        self.el = group(fixed, mus, maxlen=maxlen)
        self.circ = [(complex(c), float(r))
                     for c, r in (iso_circle(A, Bf, m)
                                  for (A, Bf), m in zip(fixed, mus))]

    def abel(self, P, Q):
        """A(P) - A(Q) componentwise, by the Poincare series for omega_k.

        omega_k(z) = (1/2 pi i) sum_{sigma in G/G_k} [1/(z - sigma B_k)
                                                     - 1/(z - sigma A_k)] dz
        so the integral from Q to P is a sum of logs.
        """
        out = []
        for k in range(2):
            Ak, Bk = self.fixed[k]
            tot = mp.mpc(0)
            for M in self.el:
                num, den = M[0]*Bk + M[1], M[2]*Bk + M[3]
                sB = num/den if den != 0 else mp.inf
                num, den = M[0]*Ak + M[1], M[2]*Ak + M[3]
                sA = num/den if den != 0 else mp.inf
                tot += mp.log(((P - sB)*(Q - sA))/((P - sA)*(Q - sB)))
            out.append(tot/(2j*mp.pi))
        return out

    def theta_odd(self, z):
        return theta_char(z, self.B, self.D1, self.D2)

    def prime_ratio(self):
        """The train-track factor of (10): the spinors cancel, only theta[D]."""
        t = self.tracks
        a_p, b_p, a_m, b_m = t['a+'], t['b+'], t['a-'], t['b-']
        num = (self.theta_odd(self.abel(a_p, b_p))
               * self.theta_odd(self.abel(a_m, b_m)))
        den = (self.theta_odd(self.abel(b_p, a_m))
               * self.theta_odd(self.abel(b_m, a_p)))
        return num/den


if __name__ == "__main__":
    fixed = [(mp.mpc('2.0', '0.6'), mp.mpc('2.0', '-0.6')),
             (mp.mpc('-2.0', '0.6'), mp.mpc('-2.0', '-0.6'))]
    mus = [mp.mpf('0.07'), mp.mpf('0.10')]
    tracks = {'a+': mp.mpf('-5.0'), 'b+': mp.mpf('-1.5'),
              'a-': mp.mpf('1.5'), 'b-': mp.mpf('5.0')}
    F = Genus2Fock(fixed, mus, tracks)
    print("period matrix:")
    for i in range(2):
        print("   ", " ".join(mp.nstr(F.B[i, j], 10) for j in range(2)))
    print(f"\nodd characteristic used: D1={[float(x) for x in F.D1]}, "
          f"D2={[float(x) for x in F.D2]}")
    print(f"  theta[D](0) = {mp.nstr(abs(F.theta_odd([mp.mpf(0)]*2)), 4)} "
          f"(must vanish)")
    d = F.abel(tracks['a+'], tracks['b+'])
    print(f"\nAbel map A(a+) - A(b+) = ({mp.nstr(d[0],10)}, {mp.nstr(d[1],10)})")
    pr = F.prime_ratio()
    print(f"train-track factor of (10) = {mp.nstr(pr, 10)}")
    print(f"   |.| = {mp.nstr(abs(pr),8)}   arg/pi = {mp.nstr(mp.arg(pr)/mp.pi,6)}")
    print("   (must be REAL for an M-curve with all labels on X_0)")
