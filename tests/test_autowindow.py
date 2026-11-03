import numpy as np

from core.autowindow import analyze_windows, sliding_window_starts, tiled_window_starts
from core.correlate import smooth_error

FAST = dict(segments=3, points_per_segment=8, base=4, n_blocks=3)


def test_sliding_window_starts_steps_by_one_second():
    starts = sliding_window_starts(t_min=0.0, t_max=10.0, window_length_s=3.0, step_s=1.0)
    # 0-3, 1-4, 2-5, ..., 7-10 -- stops once a full window no longer fits.
    assert starts == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


def test_sliding_window_starts_falls_back_to_whole_trace_when_too_long():
    starts = sliding_window_starts(t_min=0.0, t_max=5.0, window_length_s=10.0, step_s=1.0)
    assert starts == [0.0]


def test_tiled_window_starts_do_not_overlap():
    assert tiled_window_starts(0.0, 10.0, 3.0) == [0.0, 3.0, 6.0]


def _poisson_trace(rng, n=20000, rate=8, dt=1e-3):
    return np.arange(n) * dt, rng.poisson(rate, n).astype(float)


def test_default_curve_is_average_of_tiled_windows(rng):
    time_arr, ch1 = _poisson_trace(rng)  # ~20 s
    result = analyze_windows(time_arr, {"CH1": ch1}, window_length_s=2.0, **FAST)

    assert result.chosen is result.average
    assert result.average.kind == "average"
    assert len(result.tiled) == len(tiled_window_starts(float(time_arr[0]), float(time_arr[-1]), 2.0))
    accepted = [w for w in result.tiled if not w.rejected]
    assert result.average.n_averaged == len(accepted)
    assert np.isclose(result.average.duration_s, 2.0 * len(accepted))
    # Sliding windows are still all computed for inspection.
    assert len(result.candidates) == len(sliding_window_starts(float(time_arr[0]), float(time_arr[-1]), 2.0))


def test_average_error_is_sem_across_windows(rng):
    time_arr, ch1 = _poisson_trace(rng)
    result = analyze_windows(time_arr, {"CH1": ch1}, window_length_s=2.0, **FAST)

    accepted = [w for w in result.tiled if not w.rejected]
    n_tau = len(result.average.corr_results["acf_ch1"].g)
    gs = np.array([w.corr_results["acf_ch1"].g[:n_tau] for w in accepted])
    np.testing.assert_allclose(result.average.corr_results["acf_ch1"].g, gs.mean(axis=0))
    np.testing.assert_allclose(
        result.average.corr_errors["acf_ch1"], gs.std(axis=0, ddof=1) / np.sqrt(len(accepted))
    )


def test_aggregate_burst_window_is_rejected_and_excluded(rng):
    """A bright aggregate in one window raises both its count rate and its
    amplitude. The old best-SNR rule would have favored that window; the
    screening must reject it instead."""
    time_arr, ch1 = _poisson_trace(rng)
    burst = slice(10200, 10600)  # inside tiled window [10, 12) s
    ch1[burst] += 200 * np.hanning(400)

    result = analyze_windows(time_arr, {"CH1": ch1}, window_length_s=2.0, **FAST)

    by_t0 = {round(w.t0, 3): w for w in result.tiled}
    assert by_t0[10.0].rejected
    assert by_t0[10.0].reject_reason
    assert result.average.n_averaged == len(result.tiled) - sum(w.rejected for w in result.tiled)
    assert result.average.n_averaged < len(result.tiled)


def test_bleaching_step_windows_are_rejected(rng):
    time_arr, ch1 = _poisson_trace(rng, rate=20)
    ch1[14000:] = rng.poisson(4, 6000)  # last 6 s drop to a fifth of the rate

    result = analyze_windows(time_arr, {"CH1": ch1}, window_length_s=2.0, **FAST)

    by_t0 = {round(w.t0, 3): w for w in result.tiled}
    assert by_t0[14.0].rejected and by_t0[16.0].rejected
    assert "count rate" in by_t0[14.0].reject_reason
    assert not by_t0[0.0].rejected


def test_falls_back_to_full_trace_when_window_longer_than_trace(rng):
    time_arr, ch1 = _poisson_trace(rng, n=5000)  # ~5 s
    result = analyze_windows(time_arr, {"CH1": ch1}, window_length_s=10.0, **FAST)

    assert result.average is None
    assert result.chosen is result.full_trace
    assert result.full_trace.kind == "full"


def test_dual_channel_includes_cross(rng):
    time_arr, ch1 = _poisson_trace(rng)
    ch2 = rng.poisson(4, len(ch1)).astype(float)
    result = analyze_windows(time_arr, {"CH1": ch1, "CH2": ch2}, window_length_s=2.0, **FAST)

    assert set(result.chosen.corr_results.keys()) == {"acf_ch1", "acf_ch2", "cross"}
    assert set(result.chosen.corr_errors.keys()) == {"acf_ch1", "acf_ch2", "cross"}


def test_window_too_short_for_sub_blocks_still_usable(rng):
    """Sub-block errors failing must not mark the window failed -- it can still
    be averaged, it just has no per-window error bars."""
    time_arr, ch1 = _poisson_trace(rng, n=10000)
    result = analyze_windows(
        time_arr, {"CH1": ch1}, window_length_s=3.0, segments=2, points_per_segment=3, base=4, n_blocks=2000
    )
    assert not any(c.failed for c in result.candidates)
    assert all(c.corr_errors == {} for c in result.candidates)
    assert result.average is not None


def test_smooth_error_keeps_nans_and_reduces_scatter(rng):
    true_sigma = np.linspace(1.0, 2.0, 60)
    noisy = true_sigma * rng.uniform(0.6, 1.4, 60)
    noisy[[10, 40]] = np.nan

    smoothed = smooth_error(noisy)

    assert np.isnan(smoothed[10]) and np.isnan(smoothed[40])
    finite = np.isfinite(noisy)
    rms = lambda x: np.sqrt(np.mean((x[finite] / true_sigma[finite] - 1) ** 2))  # noqa: E731
    assert rms(smoothed) < rms(noisy)
