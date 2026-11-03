import numpy as np

from core.models import fcs_1comp, fcs_2comp
from core.fitting import fit_curve, multi_start_fit, FitResult


def test_1component_fit_recovers_noiseless_parameters():
    tau = np.logspace(-6, 0, 200)
    true_N, true_tauD = 2.5, 2e-4
    g = fcs_1comp(tau, true_N, true_tauD, kappa=5.0)

    fr = fit_curve(tau, g, n_components=1)
    assert fr.success
    assert abs(fr.N - true_N) / true_N < 1e-3
    assert abs(fr.tauD[0] - true_tauD) / true_tauD < 1e-3


def test_2component_fit_recovers_noiseless_parameters():
    tau = np.logspace(-6, 0, 200)
    g = fcs_2comp(tau, 3.0, 0.3, 1e-5, 5e-3, kappa=5.0)

    fr = fit_curve(tau, g, n_components=2)
    assert fr.success
    assert abs(fr.tauD[0] - 1e-5) / 1e-5 < 1e-2
    assert abs(fr.tauD[1] - 5e-3) / 5e-3 < 1e-2


def test_fit_result_calibration_defaults_to_none():
    tau = np.logspace(-6, 0, 100)
    g = fcs_1comp(tau, 2.0, 1e-4, kappa=5.0)
    fr = fit_curve(tau, g, n_components=1)
    assert fr.calibration is None


def test_multi_start_fit_stable_on_well_conditioned_curve():
    rng = np.random.default_rng(1)
    tau = np.logspace(-6, 0, 200)
    g = fcs_1comp(tau, 2.5, 2e-4, kappa=5.0)
    g_noisy = g + rng.normal(0, 0.002, size=g.shape)

    report = multi_start_fit(tau, g_noisy, n_components=1, n_starts=5)
    assert report.is_stable
    assert report.converged_fraction >= 0.6
    assert abs(report.chosen_result.tauD[0] - 2e-4) / 2e-4 < 0.2


def test_multi_start_fit_flags_instability_on_overparameterized_model():
    """A 2-component fit on essentially 1-component data is a classic unidentifiable
    (near-degenerate) case -- different starts should disagree wildly on which
    tau is the 'real' component, and this must be flagged, not silently accepted."""
    rng = np.random.default_rng(3)
    tau = np.logspace(-6, 0, 60)
    g = fcs_1comp(tau, 2.0, 1e-4, kappa=5.0)
    g_noisy = g + rng.normal(0, 0.01, size=g.shape)

    report = multi_start_fit(tau, g_noisy, n_components=2, n_starts=8)
    assert not report.is_stable


# --- Fit range, baseline offset, tau_D bounds, and quality verdict -----------
# Each case mirrors a failure seen on real VistaVision traces.

from core.fitting import QUALITY_GOOD, QUALITY_ROUGH, QUALITY_UNRESOLVED  # noqa: E402

TAU = np.logspace(np.log10(5e-6), np.log10(2e-2), 75)


def _noisy(g, sigma, seed=0):
    return g + np.random.default_rng(seed).normal(0, sigma, len(g))


def test_offset_recovers_tauD_under_slow_plateau():
    """A slow fluctuation adds a flat plateau. Without an offset the fit inflates
    tau_D toward the end of the lag range; with one, tau_D is recovered."""
    g = fcs_1comp(TAU, 200.0, 4e-4) + 0.03
    sigma = np.full_like(TAU, 2e-4)
    data = _noisy(g, sigma)

    with_offset = multi_start_fit(TAU, data, weights=1 / sigma, offset=True).chosen_result
    without = multi_start_fit(TAU, data, weights=1 / sigma, offset=False).chosen_result

    assert np.isclose(with_offset.tauD[0], 4e-4, rtol=0.15)
    assert np.isclose(with_offset.G_inf, 0.03, rtol=0.05)
    assert without.tauD[0] > 10 * 4e-4


def test_tau_min_excludes_afterpulsing_spike():
    g = fcs_1comp(TAU, 50.0, 3e-4)
    g[TAU < 1.5e-5] += 0.1  # detector artifact at the first lags
    fr = multi_start_fit(TAU, g, tau_min=2e-5).chosen_result
    assert np.isclose(fr.tauD[0], 3e-4, rtol=0.02)
    assert fr.fit_tau_min >= 2e-5
    assert len(fr.fit_curve) == len(TAU)  # still evaluated on the full grid for plotting


def test_tauD_bounded_to_fit_range_and_pure_noise_not_determined():
    """Pure noise used to produce tau_D values from 1e-9 s to months."""
    sigma = np.full_like(TAU, 3e-3)
    for seed in range(5):
        fr = multi_start_fit(TAU, _noisy(np.zeros_like(TAU), sigma, seed), weights=1 / sigma, tau_min=2e-5).chosen_result
        assert fr.fit_tau_min <= fr.tauD[0] <= fr.fit_tau_max
        assert fr.quality == QUALITY_UNRESOLVED, (seed, fr.tauD, fr.quality_reasons)


def test_clean_curve_is_well_determined_with_small_error():
    sigma = np.full_like(TAU, 2e-4)
    data = _noisy(fcs_1comp(TAU, 20.0, 3e-4), sigma)
    fr = multi_start_fit(TAU, data, weights=1 / sigma, tau_min=2e-5).chosen_result
    assert fr.quality == QUALITY_GOOD, fr.quality_reasons
    assert fr.tauD_rel_err[0] < 0.1


def test_decay_faster_than_first_lag_is_not_determined():
    """10 kHz recordings: first lag 100 us, comparable to a sub-ms tau_D."""
    tau = np.logspace(-4, 0, 60)
    sigma = np.full_like(tau, 2e-4)
    data = _noisy(fcs_1comp(tau, 20.0, 3e-5), sigma)
    fr = multi_start_fit(tau, data, weights=1 / sigma).chosen_result
    assert fr.quality == QUALITY_UNRESOLVED
    assert any("first fitted lag" in r for r in fr.quality_reasons)


def test_verdict_degrades_with_noise():
    """Same decay at rising noise: well determined -> rough -> not determined,
    with tau_D's relative error tracking the noise."""
    clean = fcs_1comp(TAU, 100.0, 5e-4) + 0.01
    verdicts = []
    for s in (5e-4, 4e-3, 1.5e-2):
        sigma = np.full_like(TAU, s)
        verdicts.append(multi_start_fit(TAU, _noisy(clean, sigma, seed=1), weights=1 / sigma, tau_min=2e-5).chosen_result.quality)
    assert verdicts == [QUALITY_GOOD, QUALITY_ROUGH, QUALITY_UNRESOLVED]


def test_fixed_tauD1_pins_fast_component_and_recovers_slow_one():
    """Locking the monomer tau_D (e.g. from a control) leaves only the slow
    component and the split to fit, which noisy data can still constrain."""
    import pytest

    tau = np.logspace(np.log10(2e-5), np.log10(0.3), 80)
    sigma = np.full_like(tau, 1e-3)
    data = _noisy(fcs_2comp(tau, 50.0, 0.6, 3e-4, 3e-2), sigma, seed=2)

    fixed = multi_start_fit(tau, data, n_components=2, weights=1 / sigma, fixed_tauD1=3e-4, offset=False).chosen_result
    assert fixed.tauD[0] == 3e-4
    assert fixed.fixed_tauD1 == 3e-4
    assert np.isclose(fixed.tauD[1], 3e-2, rtol=0.3)
    assert np.isclose(fixed.params["f1"], 0.6, atol=0.1)
    # The fixed parameter has no stderr by design; it must not make the fit "not determined".
    assert not any("tau_D1" in r for r in fixed.quality_reasons), fixed.quality_reasons

    with pytest.raises(ValueError):
        fit_curve(tau, data, n_components=1, fixed_tauD1=3e-4)
