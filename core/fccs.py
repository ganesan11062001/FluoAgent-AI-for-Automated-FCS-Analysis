"""
core/fccs.py — FCCS bound-fraction (amplitude-ratio) calculation.

Standard formalism (Bacia & Schwille): for red/green channels with cross-correlation
amplitude Gx(0) and autocorrelation amplitudes G_r(0), G_g(0):
    Gx(0) = N_rg / (N_r * N_g)         (N_rg = number of doubly-labeled complexes)
    G_r(0) = 1/N_r,  G_g(0) = 1/N_g
=>  fraction of red-labeled molecules bound to green  = N_rg/N_r = Gx(0) / G_g(0)
    fraction of green-labeled molecules bound to red  = N_rg/N_g = Gx(0) / G_r(0)

i.e. bound fraction of a channel = Gx(0) / G(other channel)(0).

No spectral crosstalk / bleed-through correction is applied in this version (by
project decision) -- this is the uncorrected amplitude-ratio formula.
"""


def bound_fraction_from_cross(g_cross0, g_partner0):
    """Fraction of the OTHER channel's molecules that are in complex, i.e. bound
    to the species measured by g_partner0.

    g_cross0: fitted cross-correlation amplitude at tau->0 (1/N * sum(f_i) from the
              cross-correlation FitResult).
    g_partner0: fitted autocorrelation amplitude at tau->0 of the PARTNER channel
                (the channel whose species you want the bound fraction OF is the
                *other* one -- see module docstring).

    Returns NaN (rather than raising) if g_partner0 is ~0, since a zero/near-zero
    autocorrelation amplitude means that channel's fit is not usable for this ratio.
    """
    if g_partner0 is None or abs(g_partner0) < 1e-12:
        return float("nan")
    return g_cross0 / g_partner0


def amplitude_at_zero(fit_result):
    """The fitted diffusion amplitude at tau->0. Every diffusion_i(tau) term equals 1
    at tau=0 and the component fractions f_i sum to 1, so this reduces to 1/N
    regardless of component count -- used instead of the noisiest raw near-zero-lag
    data point. The baseline offset G_inf is deliberately excluded: it comes from
    fluctuations much slower than diffusion, not from the diffusing complexes the
    bound-fraction ratio is about."""
    if fit_result is None or not fit_result.success:
        return float("nan")
    return 1.0 / fit_result.N
