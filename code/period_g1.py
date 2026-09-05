"""
Period of the genus-1 spectral curve, Berggren-Nicoletti (arXiv:2502.07241) Sec 4.6.

Curve (from the discriminant of char. poly (101) as a quadratic in w, which is
4(C+4)(z+a^2)(z+a^-2)/z with C = 4+4a^2+4a^-2; this differs from z(z+a^2)(z+a^-2)
by a square, so the two give the same curve, branch points and periods):
        y^2 = z (z + a^2) (z + a^{-2}),
branch points  e1 = -a^{-2} < e2 = -a^2 < e3 = 0 < e4 = infinity.

Real locus:
  * f(z) > 0 on (e1, e2)  -> COMPACT oval          (A cycle, per BN convention)
  * f(z) > 0 on (e3, oo)  -> noncompact/outer oval
  * f(z) < 0 on (e2, e3)  -> y pure imaginary here; this segment carries the B cycle

Normalized holomorphic form: omega = dz / (y * I_A) with I_A = oint_A dz/y, so that
int_A omega = 1 and  B = int_B omega = i * I_B / I_A.

Three independent evaluations: Legendre K, direct quadrature, AGM.
"""
import mpmath as mp

mp.mp.dps = 30


# ---------------------------------------------------------------- method 1
def b_legendre(a, convention="compact-oval"):
    """B = i K'(k)/K(k) with modulus fixed by which real segment is the A cycle."""
    a = mp.mpf(a)
    e1, e2, e3 = -1 / a**2, -a**2, mp.mpf(0)
    k2_compact = (e2 - e1) / (e3 - e1)      # = 1 - a^4
    if convention == "compact-oval":
        k2 = k2_compact
    elif convention == "swapped":            # A and B cycles interchanged
        k2 = 1 - k2_compact                  # = a^4
    else:
        raise ValueError(convention)
    k = mp.sqrt(k2)
    kp = mp.sqrt(1 - k2)
    return mp.ellipk(kp**2) / mp.ellipk(k**2)   # mpmath's ellipk takes m = k^2


# ---------------------------------------------------------------- method 2
def b_quadrature(a):
    """Direct numerical evaluation of the two real periods. No elliptic-integral
    identities used -- this is the check that the cycle bookkeeping is right."""
    a = mp.mpf(a)
    e1, e2, e3 = -1 / a**2, -a**2, mp.mpf(0)

    def absf(z):
        return abs(z * (z + a**2) * (z + 1 / a**2))

    # both integrands have inverse-square-root endpoint singularities;
    # mp.quad handles them if we split at the midpoint and let it adapt.
    I_A = 2 * mp.quad(lambda z: 1 / mp.sqrt(absf(z)), [e1, (e1 + e2) / 2, e2])
    I_B = 2 * mp.quad(lambda z: 1 / mp.sqrt(absf(z)), [e2, (e2 + e3) / 2, e3])
    return I_B / I_A, I_A, I_B


# ---------------------------------------------------------------- method 3
def b_agm(a):
    """K = pi/(2 AGM(1,k')), K' = pi/(2 AGM(1,k))  =>  K'/K = AGM(1,k')/AGM(1,k)."""
    a = mp.mpf(a)
    k2 = 1 - a**4
    k, kp = mp.sqrt(k2), mp.sqrt(1 - k2)
    return mp.agm(1, kp) / mp.agm(1, k)


# ---------------------------------------------------------------- discrete Gaussian
def facet_masses(b, e, nmax=40):
    """g=1 discrete Gaussian, scale tau = -B^{-1} = i/b, shift e.
    P(n) propto exp(i pi (n-e) tau (n-e)) = exp(-pi (n-e)^2 / b)."""
    b = mp.mpf(b)
    ns = list(range(-nmax, nmax + 1))
    w = [mp.e ** (-mp.pi * (n - mp.mpf(e)) ** 2 / b) for n in ns]
    Z = mp.fsum(w)
    return {n: wi / Z for n, wi in zip(ns, w) if wi / Z > mp.mpf("1e-6")}


# ---------------------------------------------------------------- inversion
def a_from_b(target):
    """Invert b(a) = K'/K on the compact-oval convention."""
    return mp.findroot(lambda a: b_legendre(a) - mp.mpf(target), mp.mpf("0.45"))


if __name__ == "__main__":
    a = mp.mpf("0.7")
    PAPER = mp.mpf("0.521828")

    print("=" * 68)
    print(f"a = {a}   -> curve y^2 = z(z+a^2)(z+a^-2),  branch pts "
          f"{-1/a**2}, {-a**2}, 0, oo")
    print("=" * 68)

    bq, IA, IB = b_quadrature(a)
    print(f"\nA-period (compact oval, real)      I_A = {mp.nstr(IA, 12)}")
    print(f"B-period (over [-a^2,0], imaginary) I_B = {mp.nstr(IB, 12)} i\n")

    print("  B = i * K'/K, A cycle = compact oval:")
    print(f"    Legendre     {mp.nstr(b_legendre(a), 12)} i")
    print(f"    quadrature   {mp.nstr(bq, 12)} i")
    print(f"    AGM          {mp.nstr(b_agm(a), 12)} i")
    print(f"\n  A cycle = other cycle (modulus swapped k^2 -> 1-k^2):")
    print(f"    Legendre     {mp.nstr(b_legendre(a, 'swapped'), 12)} i")
    print(f"\n  paper (Sec 4.6)  {PAPER} i")

    print("\n" + "-" * 68)
    print("Does 0.521828 come from a different value of the parameter?")
    print("-" * 68)
    a_hat = a_from_b(PAPER)
    print(f"  b(a_hat) = 0.521828  =>  a_hat = {mp.nstr(a_hat, 10)}")
    for name, val in [("a^2", a**2), ("a^4", a**4), ("sqrt(a)", mp.sqrt(a)),
                      ("2a/(1+a^2)", 2*a/(1+a**2)), ("a/(1+a^2)", a/(1+a**2)),
                      ("(1-a^2)/(1+a^2)", (1-a**2)/(1+a**2)),
                      ("a^2/(1+a^2)", a**2/(1+a**2)),
                      ("(1-a)/(1+a)", (1-a)/(1+a)),
                      ("a/(1+a)", a/(1+a)), ("a^2/(1+a^4)", a**2/(1+a**4))]:
        print(f"    {name:>16} = {mp.nstr(val, 8):<12}"
              f"{'   <== matches a_hat' if abs(val - a_hat) < mp.mpf('5e-3') else ''}")

    print("\n" + "-" * 68)
    print("Facet-height masses (Remark 4.18 alignments e = 0, 1/4, 1/2, 3/4)")
    print("-" * 68)
    for label, bval in [("b = 0.772738 (ours)", bq), ("b = 0.521828 (paper)", PAPER)]:
        print(f"\n  {label}")
        for e in ["0", "1/4", "1/2", "3/4"]:
            ev = mp.mpf(1)/4 if e == "1/4" else (mp.mpf(1)/2 if e == "1/2"
                 else (mp.mpf(3)/4 if e == "3/4" else mp.mpf(0)))
            m = facet_masses(bval, ev)
            s = ",  ".join(f"P({n})={mp.nstr(p*100, 4)}%" for n, p in sorted(m.items()))
            print(f"    e = {e:<4} {s}")
