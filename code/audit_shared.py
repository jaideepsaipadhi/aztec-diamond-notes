"""
The audit that would have caught the factor of 2.

THE FAILURE.  The retracted claim in Section 3.3 survived every internal
consistency check, and two "independent" computational routes agreed with each
other. They agreed because they shared the observable -- a single-face height
measurement -- and the observable was the thing that was wrong. Agreement
between routes is evidence only to the extent that the routes can fail
independently, and nothing in the checks being run measured that.

So the check that was missing is not another consistency test. It is a
DEPENDENCY audit: for each claim, list what its supposedly independent
confirmations have in common, and treat any claim whose confirmations share a
single point of failure as untested in that direction, however many routes
agree.

This file records the dependency graph explicitly and reports, per claim, the
intersection of its routes. It is deliberately a hand-maintained table rather
than something inferred from imports: what matters is the MATHEMATICAL
dependency -- which observable, which convention, which identity -- and that is
not visible in the call graph. The single-face observable and the spatial
average share almost all their code and differ in the one place that mattered.

Keeping it honest: a route may only be listed as not sharing a dependency if it
genuinely could have failed alone. "Different script" is not independence.
"""

# Dependencies are named at the level at which they can be WRONG.
DEPS = {
    "single_face_obs": "single-face height observable (RETRACTED, §3.3)",
    "spatial_avg_obs": "spatial average over the gas facet",
    "shuffling": "generalised domino shuffling sampler",
    "kasteleyn_det": "finite-n Kasteleyn determinant / inverse",
    "gas_inverse": "full-plane gas inverse (contour integral)",
    "curve": "spectral curve y^2 = D(z), branch points",
    "period_quad": "real-axis period quadrature (hyper_g3)",
    "schottky": "Schottky uniformisation / Poincare series",
    "theta": "theta functions with odd characteristic",
    "abel": "Abel map normalisation (A-periods)",
    "dg_convention": "discrete Gaussian convention P ~ exp(-pi n Q n), Q = (Im B)^-1",
    "facet_mask": "facet detection and erosion",
    "elliptic_id": "elliptic-integral identities (K'/K)",
    "exact_enum": "brute-force weighted matching enumeration",
}

CLAIMS = [
    dict(
        name="B = 0.7727 i at genus 1 (§2, §3)",
        routes={
            "determinant + single face": ["kasteleyn_det", "single_face_obs"],
            "determinant + spatial avg": ["kasteleyn_det", "spatial_avg_obs"],
            "sampling + spatial avg": ["shuffling", "spatial_avg_obs",
                                        "facet_mask"],
            "elliptic identity K'/K": ["elliptic_id"],
            "curve + period quadrature": ["curve", "period_quad"],
        },
        note="Five routes. The elliptic identity shares nothing with any "
             "other, which is what makes the value safe.",
    ),
    dict(
        name="factor of 2 in the discrete component (RETRACTED)",
        routes={
            "determinant + single face": ["kasteleyn_det", "single_face_obs"],
            "sampling + single face": ["shuffling", "single_face_obs"],
        },
        note="The failure. Two routes, agreeing, sharing the observable that "
             "was wrong. The audit below flags it without hindsight.",
    ),
    dict(
        name="Theorem 1.2 at genus 1, shift e = 1/4 (§3.6)",
        routes={
            "sampling + spatial avg": ["shuffling", "spatial_avg_obs",
                                        "facet_mask", "dg_convention"],
            "four alignments (Remark 4.18)": ["shuffling", "spatial_avg_obs",
                                               "facet_mask", "dg_convention"],
        },
        note="Both routes share the observable AND the convention. The four "
             "alignments vary the model, not the measurement.",
    ),
    dict(
        name="Theorem 1.2 at genus 3, off-diagonals (§5)",
        routes={
            "atoms": ["shuffling", "spatial_avg_obs", "facet_mask",
                      "dg_convention", "curve", "period_quad"],
            "correlations": ["shuffling", "spatial_avg_obs", "facet_mask",
                             "dg_convention", "curve", "period_quad"],
            "marginal broadening": ["shuffling", "spatial_avg_obs",
                                     "facet_mask", "dg_convention", "curve",
                                     "period_quad"],
            "A-vs-C falsification": ["shuffling", "spatial_avg_obs",
                                      "facet_mask", "dg_convention", "curve",
                                      "period_quad"],
        },
        note="Four checks, identical dependency sets. They test different "
             "FEATURES of one model against one measurement, which is worth "
             "something, but none of them is an independent route.",
    ),
    dict(
        name="v = -2 u(0+), the shift velocity (§6)",
        routes={
            "measured from e(n)": ["shuffling", "spatial_avg_obs",
                                    "facet_mask", "dg_convention"],
            "computed from the curve": ["curve", "abel"],
        },
        note="Two routes sharing NOTHING. This is the strongest structure in "
             "the notes: the measurement cannot be wrong in a way that moves "
             "the Abel integral, or vice versa.",
    ),
    dict(
        name="v is irrational / orbit never closes (§6.3)",
        routes={
            "continued fraction of computed v": ["curve", "abel"],
        },
        note="ONE route. The 22-digit precision is not independence: a wrong "
             "A-period normalisation would give a wrong v to 22 digits. The "
             "genus-1 fixture (v = 1/4 exactly, known from Remark 4.18) is "
             "the only external anchor, and it constrains `abel` but not "
             "`curve`.",
    ),
    dict(
        name="genus-2 gas facet locations (§4)",
        routes={
            "Schottky + polygon map": ["schottky", "theta"],
            "sampling": ["shuffling", "facet_mask"],
            "period matrix vs jtem": ["schottky"],
        },
        note="The jtem comparison checks `schottky` against an external "
             "implementation, so it is a real external check -- but of the "
             "period matrix only, not of the polygon map.",
    ),
]


def audit():
    print("=" * 78)
    print("  DEPENDENCY AUDIT: what do the 'independent' routes share?")
    print("=" * 78)
    worst = []
    for c in CLAIMS:
        routes = c["routes"]
        sets = [set(v) for v in routes.values()]
        shared = set.intersection(*sets) if sets else set()
        n = len(routes)
        print(f"\n{c['name']}")
        print(f"  {n} route(s):")
        for k, v in routes.items():
            print(f"    - {k}")
        if n == 1:
            verdict = "SINGLE ROUTE -- no cross-check at all"
        elif not shared:
            verdict = "routes share nothing -> genuinely independent"
        else:
            verdict = (f"ALL routes share {len(shared)}: "
                       + ", ".join(sorted(DEPS[d] for d in shared)))
        print(f"  verdict: {verdict}")
        print(f"  note: {c['note']}")
        if n == 1 or shared:
            worst.append((c["name"], n, sorted(shared)))

    print("\n" + "=" * 78)
    print("  CLAIMS WITH A SINGLE POINT OF FAILURE")
    print("=" * 78)
    for name, n, shared in worst:
        tag = "1 route" if n == 1 else f"{n} routes, all sharing {len(shared)}"
        print(f"  [{tag}] {name}")
        for d in shared:
            print(f"        {DEPS[d]}")

    print("\n" + "=" * 78)
    print("  WOULD IT HAVE CAUGHT THE RETRACTED CLAIM?")
    print("=" * 78)
    bad = next(c for c in CLAIMS if "RETRACTED" in c["name"])
    sh = set.intersection(*[set(v) for v in bad["routes"].values()])
    print(f"  Yes. Its two routes share: "
          + ", ".join(sorted(DEPS[d] for d in sh)))
    print("  The audit names the shared observable as the single point of")
    print("  failure, using only the dependency table -- no knowledge that")
    print("  the observable was in fact wrong.")
    print("\n  What it does NOT do: tell you WHICH shared dependency is")
    print("  wrong, or that any is. It converts 'two routes agree' into")
    print("  'two routes agree and here is the one thing that would fool")
    print("  both', which is the question that had not been asked.")


if __name__ == "__main__":
    audit()
