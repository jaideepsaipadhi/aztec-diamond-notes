"""
Fock face weights from Schottky data -- the FORWARD direction.

Following Bobenko-Bobenko-Suris (arXiv:2402.08798), Section 11, eq (67):

    W_f = [theta[D](int_{a+}^{b+} w) / theta[D](int_{b+}^{a-} w)]
        * [theta[D](int_{a-}^{b-} w) / theta[D](int_{b-}^{a+} w)]
        * [theta(eta(f_e)+D) theta(eta(f_w)+D)]
        / [theta(eta(f_n)+D) theta(eta(f_s)+D)]

U2 uniformization (their Section 10): A_n = conj(B_n), Im A_n > 0, 0 < mu_n < 1,
and the Harnack data are cyclically ordered  beta^- < alpha^+ < beta^+ < alpha^-
on the real line.

For genus 1 everything is elementary, because G/G_1 (words not ending in
sigma_1) contains only the identity:

    omega(z) = (1/2 pi i) [ 1/(z-B) - 1/(z-A) ] dz
    Abel map  A(P) = (1/2 pi i) log[ (z-B)(z0-A) / ((z-A)(z0-B)) ]
    period    B_11 = log(mu) / (2 pi i)

The genus-1 theta functions with the odd characteristic Delta = [1/2, 1/2]:

    theta(z)      = sum_m exp(i pi B m^2 + 2 pi i z m)
    theta[D](z)   = sum_m exp(i pi B (m+1/2)^2 + 2 pi i (z+1/2)(m+1/2))

The test that matters: the resulting face weights must be POSITIVE.  That is the
Kasteleyn/positivity property the M-curve construction is designed to guarantee
(their Section 6), and it is a real check on the whole pipeline.
"""
import mpmath as mp

mp.mp.dps = 30


def schottky_genus1(A, mu):
    """U2 data: fixed points A and conj(A), multiplier mu."""
    A = mp.mpc(A)
    B = mp.conj(A)
    Bper = mp.log(mp.mpf(mu)) / (2j * mp.pi)     # period matrix (1x1)
    return A, B, Bper


def abel(z, A, B, z0=mp.inf):
    """A(P) = (1/2 pi i) log[(z-B)(z0-A)/((z-A)(z0-B))]; z0 = infinity gives 1."""
    z = mp.mpc(z)
    if z0 == mp.inf:
        r = (z - B) / (z - A)
    else:
        r = ((z - B) * (z0 - A)) / ((z - A) * (z0 - B))
    return mp.log(r) / (2j * mp.pi)


def theta(z, Bper, N=40):
    return mp.fsum(mp.e ** (1j*mp.pi*Bper*m**2 + 2j*mp.pi*z*m)
                   for m in range(-N, N+1))


def theta_odd(z, Bper, N=40):
    return mp.fsum(mp.e ** (1j*mp.pi*Bper*(m+mp.mpf(1)/2)**2
                            + 2j*mp.pi*(z + mp.mpf(1)/2)*(m+mp.mpf(1)/2))
                   for m in range(-N, N+1))


def face_weight(eta, pts, A, B, Bper, D):
    """eq (67).  pts = (beta_m, alpha_p, beta_p, alpha_m) in cyclic order
    beta^- < alpha^+ < beta^+ < alpha^-.  eta is the discrete Abel map at f."""
    bm, ap, bp, am = [abel(p, A, B) for p in pts]
    # prime-form ratios via odd theta
    num = theta_odd(bp - ap, Bper) * theta_odd(bm - am, Bper)
    den = theta_odd(am - bp, Bper) * theta_odd(ap - bm, Bper)
    # the eta-dependent factor: neighbours differ by the train-track increments
    de = (ap - am)          # east/west shift
    dn = (bp - bm)          # north/south shift
    fac = (theta(eta + de + D, Bper) * theta(eta - de + D, Bper)) / \
          (theta(eta + dn + D, Bper) * theta(eta - dn + D, Bper))
    return (num / den) * fac


if __name__ == "__main__":
    # Bobenko et al., Fig. 21 (left): genus 1
    pts = (mp.mpf('-2.4'), mp.mpf('-0.4'), mp.mpf('0.4'), mp.mpf('2.4'))
    A, B, Bper = schottky_genus1(mp.mpc('0.1', '1.0'), '0.02')
    print("Schottky data (their Fig. 21 left):")
    print(f"  A = {A},  mu = 0.02")
    print(f"  period B = {mp.nstr(Bper, 12)}   (b = {mp.nstr(mp.im(Bper), 12)})")
    print(f"  train tracks (beta-, alpha+, beta+, alpha-) = "
          f"{[float(p) for p in pts]}")

    print("\nAbel map of the four train-track points:")
    for nm, p in zip(("beta-", "alpha+", "beta+", "alpha-"), pts):
        print(f"  {nm:<7} A = {mp.nstr(abel(p, A, B), 10)}")

    print("\nface weight W_f as a function of eta (must be POSITIVE):")
    D = mp.mpf('0.3')
    neg = 0
    for k in range(12):
        eta = mp.mpf(k) / 12
        W = face_weight(eta, pts, A, B, Bper, D)
        if mp.im(W) != 0 and abs(mp.im(W)) > 1e-20 * abs(mp.re(W)):
            flag = "  <- complex!"
        else:
            flag = ""
        if mp.re(W) <= 0:
            neg += 1
            flag += "  <- NEGATIVE"
        print(f"  eta={mp.nstr(eta,4):<8} W_f = {mp.nstr(W, 10)}{flag}")
    print(f"\n  negative weights: {neg} of 12")
