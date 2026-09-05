"""
Fast exact inverse Kasteleyn: precompute the V grids once.

The integrand of I^{j,k} factorises as

    V^{j,k}_{e1,e2}(w1,w2) / (w2-w1)   x   h_{j,k}(x,y; w1,w2)

and V depends only on (j,k,e1,e2) -- NOT on the lattice points x,y.  So the 16
V-grids are built once, and every K^{-1} entry is then a single weighted sum over
the quadrature grid.  That takes an entry from ~seconds to ~milliseconds, which
is what makes assembling the whole r x r matrix M along a dual path feasible.

Cost is independent of n throughout (the n-dependence is only in the exponents
of h, computed in the log domain to avoid overflow).
"""
import numpy as np

from exactinv import TwoPeriodicInverse, E1, E2, cls


class FastInverse:
    def __init__(self, a, n, N=600, rfac=0.5, Nq=800):
        self.a, self.n, self.N = a, n, N
        self.T = TwoPeriodicInverse(a, n)
        s = np.sqrt(2*self.T.c)
        self.r = s + rfac*(1.0 - s)
        th = 2*np.pi*np.arange(N)/N
        self.A, self.B = np.meshgrid(self.r*np.exp(1j*th),
                                     (1.0/self.r)*np.exp(1j*th), indexing='ij')
        self.dt = 2*np.pi/N
        # precompute V/(B-A) * measure for every (j,k,e1,e2)
        meas = (1j*self.A*self.dt/self.A)*(1j*self.B*self.dt)/(2j*np.pi)**2
        self.Vg = {}
        for j in (0, 1):
            for k in (0, 1):
                for e1 in (0, 1):
                    for e2 in (0, 1):
                        self.Vg[(j, k, e1, e2)] = (
                            self.T.V(j, k, e1, e2, self.A, self.B)
                            / (self.B - self.A) * meas)
        # precompute the logs appearing in logHt: they depend only on the
        # contours, not on x,y.  logHt(x1,x2,w) = 2m log w + (2m-x1/2) log(-iG(w))
        #                                        - (2m-x2/2) log(iG(1/w))
        T = self.T
        self.LA = (np.log(self.A), np.log(-1j*T.G(self.A)),
                   np.log(1j*T.G(1.0/self.A)))
        self.LB = (np.log(self.B), np.log(-1j*T.G(self.B)),
                   np.log(1j*T.G(1.0/self.B)))
        # gas term grids
        tq = 2*np.pi*np.arange(Nq)/Nq
        Z, W = np.meshgrid(np.exp(1j*tq), np.exp(1j*tq), indexing='ij')
        P = -2 - 2*a**2 - a/W - a*W - a/Z - a*Z
        self.gasP, self.gasZ, self.gasW = P, Z, W
        self.gasINV = {(0, 0): 1j*(a+W), (0, 1): -(a+Z),
                       (1, 0): -(a+1/Z), (1, 1): 1j*(a+1/W)}

    def gas(self, x, y):
        e1, e2 = cls(x), cls(y)
        w0 = np.array(x) if e1 == 0 else np.array(x) - E1 - E2
        bt = w0 + E1 if e2 == 1 else w0 + E2
        d = np.array(y) - bt
        if (d[0]+d[1]) % 4 or (d[1]-d[0]) % 4:
            return None
        u = (d[0]+d[1])//4
        v = (d[1]-d[0])//4
        return (self.gasINV[(e1, e2)]/self.gasP
                * self.gasZ**int(u) * self.gasW**int(v)).mean()

    def entry(self, x, y):
        """K^{-1}(x, y) for white x, black y."""
        e1, e2 = cls(x), cls(y)
        n2 = 2*self.n
        m2 = 2*self.T.m
        la, ga, ga2 = self.LA
        lb, gb, gb2 = self.LB
        tot = 0
        for j in (0, 1):
            for k in (0, 1):
                x1, x2 = x[0]+1, (x[1] if k == 0 else n2-x[1])
                y1, y2 = (y[0] if j == 0 else n2-y[0]), y[1]+1
                ln = ((m2*la + (m2-x1/2)*ga - (m2-x2/2)*ga2)
                      - (m2*lb + (m2-y1/2)*gb - (m2-y2/2)*gb2))
                tot += ((-1)**(j+k))*1j**(y[0]-x[0]) * (
                    self.Vg[(j, k, e1, e2)]*np.exp(ln)).sum()
        g = self.gas(x, y)
        return None if g is None else g - tot


if __name__ == "__main__":
    import time
    import warnings
    warnings.filterwarnings('ignore')
    a, n = 0.7, 32
    Wv = [(i, j) for i in range(1, 2*n, 2) for j in range(0, 2*n+1, 2)]
    Bv = [(i, j) for i in range(0, 2*n+1, 2) for j in range(1, 2*n, 2)]
    wi = {v: k for k, v in enumerate(Wv)}
    bi = {v: k for k, v in enumerate(Bv)}
    K = np.zeros((len(Bv), len(Wv)), dtype=complex)
    for x in Bv:
        j = cls(x)
        for dd, val in ((tuple(E1), a*(1-j)+j), (tuple(E2), (a*j+(1-j))*1j),
                        (tuple(-E1), a*j+(1-j)), (tuple(-E2), (a*(1-j)+j)*1j)):
            y = (x[0]+dd[0], x[1]+dd[1])
            if y in wi:
                K[bi[x], wi[y]] = val
    Ki = np.linalg.inv(K)

    for N in (400, 800, 1600):
        t0 = time.time()
        F = FastInverse(a, n, N=N)
        setup = time.time()-t0
        errs, t1 = [], time.time()
        m = n//4
        for xi in (-0.03, -0.3, -0.6):
            t = int(4*m + 2*m*xi)
            for c0 in [p for p in Bv if abs(p[0]-t) < 3 and abs(p[1]-t) < 3]:
                for off in ((1, 1), (-1, -1)):
                    w0 = (c0[0]+off[0], c0[1]+off[1])
                    if w0 not in wi:
                        continue
                    v = F.entry(w0, c0)
                    if v is not None:
                        errs.append(abs(Ki[wi[w0], bi[c0]] - v))
        print(f"  N={N:<5} setup {setup:.0f}s   {len(errs)} entries in "
              f"{time.time()-t1:.1f}s   max |err| {max(errs):.2e}  "
              f"median {np.median(errs):.2e}")
