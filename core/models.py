"""
core/models.py — standard confocal 3D-diffusion FCS autocorrelation models.

G(tau) = (1/N) * sum_i f_i * diffusion_i(tau)   with   sum_i f_i = 1
diffusion_i(tau) = 1/(1+tau/tauD_i) * 1/sqrt(1+tau/(kappa^2 * tauD_i))

kappa = w_z/w_xy is the structure parameter (fixed, not floated -- see fitting.py).
An optional triplet/blinking prefactor can be multiplied in:
  (1 - T + T*exp(-tau/tau_trip)) / (1 - T)

An optional constant baseline G_inf is added on top:
  G(tau) = G_inf + (1/N) * [diffusion terms] * [triplet term]
Real traces often carry fluctuations much slower than diffusion (aggregates,
drift, focus wander). Over the fitted lag range these look like a constant
plateau. Without G_inf the fit can only explain that plateau by inflating
tau_D to seconds. G_inf is a nuisance term: it is excluded from the amplitude
1/N used for N and for the FCCS bound fraction.
"""

import numpy as np


def diffusion_term(tau, tauD, kappa):
    tau = np.asarray(tau, dtype=np.float64)
    return 1.0 / (1.0 + tau / tauD) / np.sqrt(1.0 + tau / (kappa**2 * tauD))


def triplet_term(tau, T, tau_trip):
    tau = np.asarray(tau, dtype=np.float64)
    if T <= 0:
        return np.ones_like(tau)
    return (1.0 - T + T * np.exp(-tau / tau_trip)) / (1.0 - T)


def fcs_1comp(tau, N, tauD, kappa=5.0, T=0.0, tau_trip=1e-6, G_inf=0.0):
    g = (1.0 / N) * diffusion_term(tau, tauD, kappa)
    return G_inf + g * triplet_term(tau, T, tau_trip)


def fcs_2comp(tau, N, f1, tauD1, tauD2, kappa=5.0, T=0.0, tau_trip=1e-6, G_inf=0.0):
    f2 = 1.0 - f1
    g = (1.0 / N) * (f1 * diffusion_term(tau, tauD1, kappa) + f2 * diffusion_term(tau, tauD2, kappa))
    return G_inf + g * triplet_term(tau, T, tau_trip)
