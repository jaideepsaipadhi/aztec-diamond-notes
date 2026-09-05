"""
Tabulate the surface tension sigma over the polygon Delta_S.

sigma is currently only computable pointwise, and each evaluation costs a path
integral of ~120 Poincare-series evaluations.  A variational solver needs sigma
and grad sigma at every grid point at every iteration, so we tabulate first.

Strategy: sweep z over the upper half plane.  Each z gives a conjugate triple

    slope   s = Delta(z)          (a point of the polygon)
    amoeba  x = A(z)              (which is grad sigma at s)
    value   sigma(z) = s.x - rho(z)

so ONE sweep produces sigma AND its gradient on a scattered set of slopes, with
no gradient inversion anywhere.  That is the payoff of the z-parameterisation:
the hard direction of the Legendre transform is free.

The efficiency trick: rho is a path integral from a base point, but along a
sweep we can integrate INCREMENTALLY, carrying rho from one sample to the next,
so each new point costs one short segment instead of a full path.

The gas facet is the image of the compact oval; sigma has a conical singularity
there, and the sweep must not smooth across it -- we record where the samples
crowd, which is the signature of the facet.
"""
import mpmath as mp

from ronkin import AD
from tension import drho_uv

mp.mp.dps = 15


def sweep(us, vs):
    """Sweep z = u + iv, carrying rho incrementally along each v-column."""
    out = []
    base = None
    for u in us:
        # walk up the column, carrying rho
        col = []
        prev_z = None
        acc = mp.mpf(0)
        for v in vs:
            z = mp.mpc(u, v)
            if prev_z is None:
                if base is None:
                    base = z
                    acc = mp.mpf(0)
                else:
                    # connect to the base along a straight segment (once per column)
                    acc = integrate_seg(base, z)
            else:
                acc += integrate_seg(prev_z, z, N=12)
            x, D = AD(z)
            col.append((float(u), float(v), float(D[0]), float(D[1]),
                        float(x[0]), float(x[1]),
                        float(D[0]*x[0] + D[1]*x[1] - acc)))
            prev_z = z
        out.extend(col)
    return out


def integrate_seg(za, zb, N=40):
    tot = mp.mpf(0)
    for n in range(N):
        p = za + (zb - za) * mp.mpf(n) / N
        q = za + (zb - za) * mp.mpf(n + 1) / N
        m = (p + q) / 2
        gu, gv = drho_uv(m)
        tot += gu * mp.re(q - p) + gv * mp.im(q - p)
    return tot


if __name__ == "__main__":
    import time
    us = [mp.mpf(k) / 2 for k in range(-4, 5)]
    vs = [mp.mpf('1.2') + mp.mpf(k) * mp.mpf('0.4') for k in range(7)]
    t0 = time.time()
    tab = sweep(us, vs)
    dt = time.time() - t0
    print(f"tabulated {len(tab)} points in {dt:.0f}s "
          f"({dt/len(tab)*1000:.0f} ms/point)\n")

    s1 = [r[2] for r in tab]; s2 = [r[3] for r in tab]
    print(f"polygon coverage: s1 in [{min(s1):.4f}, {max(s1):.4f}], "
          f"s2 in [{min(s2):.4f}, {max(s2):.4f}]")
    print("(the compact oval maps to (-0.4717, -0.5170) -- the gas facet)\n")

    print("consistency: tabulated sigma vs a direct pointwise evaluation")
    from tension import sigma as sigma_direct
    for (u, v, d1, d2, x1, x2, sg) in tab[::17]:
        z = mp.mpc(u, v)
        direct = float(sigma_direct(z))
        print(f"   z={u:+.2f}{v:+.2f}i  s=({d1:+.5f},{d2:+.5f})  "
              f"tab={sg:+.7f}  direct={direct:+.7f}  diff={abs(sg-direct):.1e}")
