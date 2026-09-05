"""
EXACT inverse Kasteleyn for the two-periodic Aztec diamond, by contour integrals.

This bypasses the conditioning wall entirely: no matrix, no LU, cost independent
of n.  cond(K) ~ 10^(0.252 n) kills the matrix route at n ~ 60 (the cause is the
frozen corners' deterministic matchings, not the gas phase).

Chhita-Johansson arXiv:1410.2385 Thm 2.3 / Bain arXiv:2204.06378 Thm 3.2:

    K_a^{-1}(x,y) = K^{-1}_{a,0,0}(x,y) - (I^{0,0} - I^{1,0} - I^{0,1} + I^{1,1})

with, for sqrt(2c) < r < 1,

    I^{j,k} = i^{y1-x1}/(2 pi i)^2 * oint_{C_r} dw1/w1 oint_{C_{1/r}} dw2
                  V^{j,k}(w1,w2)/(w2-w1) * h_{j,k}(w1,w2)

All ingredients (Bain eqs 3.1-3.12):
    c = 1/(a + 1/a)
    sqrt(w^2+2c) = i sqrt(-i(w+i sqrt(2c))) sqrt(-i(w-i sqrt(2c)))   [principal]
    G(w) = (w - sqrt(w^2+2c))/sqrt(2c)
    t(w) = w sqrt(w^{-2}+2c)
    Htilde_{x1,x2}(w) = w^{2m} (-i G(w))^{2m-x1/2} / (i G(1/w))^{2m-x2/2}
    x^{e1,e2}_{g1,g2} = G(w1)G(w2)/prod_i sqrt(wi^2+2c) sqrt(wi^{-2}+2c)
                        * y^{e1,e2}_{g1,g2}(G(w1),G(w2)) * (1 - w1^2 w2^2)
    Q = (-1)^{e1+e2+e1 e2+g1(1+e2)+g2(1+e1)} t(w1)^{g1} t(1/w2)^{g2}
        G(w1)^{e1} G(1/w2)^{e2} x^{e1,e2}_{g1,g2}(w1, 1/w2)
    V^{j,k} = 1/2 sum_{g1,g2} (-1)^{g2 j + g1 k} (Q(w1,w2) + (-1)^{e2+1} Q(w1,-w2))

The gas term K^{-1}_{a,0,0} is the validated full-plane formula (same fundamental
domain, shifts in units of 2e1/2e2, 0-based entry indexing).
"""
import numpy as np

E1 = np.array([1, 1])
E2 = np.array([-1, 1])


def cls(v):
    return ((v[0] + v[1]) % 4 - 1) // 2


class TwoPeriodicInverse:
    def __init__(self, a, n):
        assert n % 4 == 0
        self.a = a
        self.n = n
        self.m = n // 4
        self.c = 1.0/(a + 1.0/a)

    # --- branch-tracked square root, Bain (3.1) ---
    def sq(self, w):
        s2c = np.sqrt(2*self.c)
        return 1j*np.sqrt(-1j*(w + 1j*s2c))*np.sqrt(-1j*(w - 1j*s2c))

    def G(self, w):
        return (w - self.sq(w))/np.sqrt(2*self.c)

    def t(self, w):
        return w*self.sq(1.0/w)

    def logHt(self, x1, x2, w):
        """log of Htilde. Forming Htilde directly overflows for large m
        (it contains w^{2m}), but the RATIO appearing in h is O(1), so work
        in the log domain and exponentiate only at the end."""
        m = self.m
        return (2*m*np.log(w)
                + (2*m - x1/2)*np.log(-1j*self.G(w))
                - (2*m - x2/2)*np.log(1j*self.G(1.0/w)))

    # --- the y functions, Bain (3.6)-(3.7) ---
    def y00(self, g1, g2, u, v, A, B):
        f = ((2*A**2*u*v + 2*B**2*u*v - A*B*(-1+u**2)*(-1+v**2))
             * (2*A**2*u*v + 2*B**2*u*v + A*B*(-1+u**2)*(-1+v**2)))
        if (g1, g2) == (0, 0):
            return (1/(4*(A**2+B**2)**2*f))*(
                2*A**7*u**2*v**2
                - A**5*B**2*(1+u**4+u**2*v**2-u**4*v**2+v**4-u**2*v**4)
                - A**3*B**4*(1+3*u**2+3*v**2+2*u**2*v**2+u**4*v**2+u**2*v**4-u**4*v**4)
                - A*B**6*(1+v**2+u**2+3*u**2*v**2))
        if (g1, g2) == (0, 1):
            return (A/(4*(A**2+B**2)*f))*(B**2+A**2*u**2)*(2*A**2*v**2+B**2*(1+v**2-u**2+u**2*v**2))
        if (g1, g2) == (1, 0):
            return (A/(4*(A**2+B**2)*f))*(B**2+A**2*v**2)*(2*A**2*u**2+B**2*(1-u**2+v**2+u**2*v**2))
        return (A/(4*f))*(2*A**2*u**2*v**2+B**2*(-1+v**2+u**2+u**2*v**2))

    def yfun(self, e1, e2, g1, g2, u, v):
        a = self.a
        if (e1, e2) == (0, 0):
            return self.y00(g1, g2, u, v, a, 1.0)
        if (e1, e2) == (0, 1):
            return self.y00(g1, g2, u, 1.0/v, 1.0, a)/v**2
        if (e1, e2) == (1, 0):
            return self.y00(g1, g2, 1.0/u, v, 1.0, a)/u**2
        return self.y00(g1, g2, 1.0/u, 1.0/v, a, 1.0)/(u**2*v**2)

    def xfun(self, e1, e2, g1, g2, w1, w2):
        den = (self.sq(w1)*self.sq(1.0/w1)*self.sq(w2)*self.sq(1.0/w2))
        return (self.G(w1)*self.G(w2)/den
                * self.yfun(e1, e2, g1, g2, self.G(w1), self.G(w2))
                * (1 - w1**2*w2**2))

    def Q(self, e1, e2, g1, g2, w1, w2):
        sgn = (-1.0)**(e1+e2+e1*e2+g1*(1+e2)+g2*(1+e1))
        return (sgn*self.t(w1)**g1*self.t(1.0/w2)**g2
                * self.G(w1)**e1*self.G(1.0/w2)**e2
                * self.xfun(e1, e2, g1, g2, w1, 1.0/w2))

    def V(self, j, k, e1, e2, w1, w2):
        tot = 0.0
        for g1 in (0, 1):
            for g2 in (0, 1):
                s = (-1.0)**(g2*j + g1*k)
                tot = tot + s*(self.Q(e1, e2, g1, g2, w1, w2)
                               + (-1.0)**(e2+1)*self.Q(e1, e2, g1, g2, w1, -w2))
        return 0.5*tot

    def h(self, j, k, x, y, w1, w2):
        n2 = 2*self.n
        x1, x2 = x
        y1, y2 = y
        ln = (self.logHt(x1+1, x2 if k == 0 else n2-x2, w1)
              - self.logHt(y1 if j == 0 else n2-y1, y2+1, w2))
        return np.exp(ln)
