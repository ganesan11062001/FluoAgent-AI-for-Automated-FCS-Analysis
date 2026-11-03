"""
core/autowindow.py — split a raw trace into windows, screen out bad ones, and
average the rest.

The user picks a window length L (e.g. "3 s"). Two sets of windows are built:

- Sliding windows: [0, L), [1, 1+L), [2, 2+L), ... in 1-second steps. These are
  for inspection only. Each can be viewed or picked by hand.
- Tiled windows: [0, L), [L, 2L), [2L, 3L), ... non-overlapping. Their curves are
  averaged into the default curve.

Each window is screened against the trace as a whole, using a robust z-score
(median/MAD) of its mean count rate per channel and its correlation amplitude
per curve. Count-rate outliers catch bleaching, drift, and focus loss.
Amplitude outliers catch bright aggregates passing through the volume. A window
is rejected if any of its metrics exceeds OUTLIER_Z.

Why not keep the "cleanest" window? Scoring windows by |G|/sigma and keeping
the best one favors windows with a higher G, i.e. fewer molecules or an
aggregate burst. Taking the best of many windows also biases every downstream
number toward whichever stretch happened to look cleanest. Averaging all
acceptable windows uses all the good data. Because tiled windows don't overlap,
the spread across them is also an independent per-point error estimate.
"""

from dataclasses import dataclass, field

import numpy as np

from .binning import apply_point_binning
from .correlate import CorrelationResult, compute_all_correlations, compute_correlation_error
from .engine import select_window

STEP_S = 1.0
OUTLIER_Z = 3.5
MIN_WINDOWS_FOR_SCREENING = 3
N_AMPLITUDE_POINTS = 5


@dataclass
class WindowCandidate:
    label: str
    t0: float
    t1: float
    corr_results: dict
    corr_errors: dict
    eff_dt: float = float("nan")
    mean_rates: list = field(default_factory=list)
    amplitudes: dict = field(default_factory=dict)
    failed: bool = False
    tiled: bool = False
    rejected: bool = False
    reject_reason: str = ""
    kind: str = "window"  # "window" | "average" | "full"
    duration_s: float = float("nan")
    n_averaged: int = 0


@dataclass
class AutoWindowResult:
    chosen: WindowCandidate
    candidates: list = field(default_factory=list)  # sliding windows
    average: WindowCandidate = None
    full_trace: WindowCandidate = None
    tiled: list = field(default_factory=list)


def sliding_window_starts(t_min, t_max, window_length_s, step_s=STEP_S):
    """t_min, t_min+step, t_min+2*step, ... for as long as [start, start+window_length_s]
    still fully fits inside [t_min, t_max], so every candidate is the same length
    and directly comparable. If window_length_s doesn't fit at all, falls back to
    a single candidate spanning the whole trace."""
    span = t_max - t_min
    if window_length_s >= span:
        return [t_min]
    starts = []
    start = t_min
    while start + window_length_s <= t_max:
        starts.append(start)
        start += step_s
    return starts


def tiled_window_starts(t_min, t_max, window_length_s):
    """Non-overlapping windows of window_length_s covering the trace: t_min,
    t_min+L, t_min+2L, ..."""
    return sliding_window_starts(t_min, t_max, window_length_s, step_s=window_length_s)


def _amplitude(result):
    """Correlation amplitude of a window: mean G over the first few lags. This is
    approximately 1/N, so it shifts when a bright aggregate enters the volume."""
    g = result.g[:N_AMPLITUDE_POINTS]
    g = g[np.isfinite(g)]
    return float(np.mean(g)) if len(g) else float("nan")


def _correlate_window(time_arr, channels, t0, t1, label, segments, points_per_segment, base, bin_points, n_blocks):
    w_time = None
    w_channels = {}
    for name, arr in channels.items():
        wt, wa = select_window(time_arr, arr, t0, t1)
        w_time = wt
        w_channels[name] = wa
    _, b_channels, eff_dt, _ = apply_point_binning(w_time, w_channels, bin_points)
    corr = compute_all_correlations(
        b_channels, eff_dt, segments=segments, points_per_segment=points_per_segment, base=base
    )
    # A window too short for sub-block errors is still a usable curve (and can
    # still go into the average); it just has no per-window error bars.
    try:
        err = compute_correlation_error(
            corr, b_channels, eff_dt,
            segments=segments, points_per_segment=points_per_segment, base=base, n_blocks=n_blocks,
        )
    except ValueError:
        err = {}
    mean_rates = [float(np.mean(arr)) / eff_dt for arr in b_channels.values()]
    amplitudes = {kind: _amplitude(r) for kind, r in corr.items()}
    return WindowCandidate(
        label, t0, t1, corr, err, eff_dt, mean_rates, amplitudes, duration_s=t1 - t0,
    )


def _robust_z(values):
    """|x - median| / (1.4826 * MAD); 1.4826 makes MAD match the standard deviation
    for normal data. Returns zeros if MAD is 0 (no spread to judge against)."""
    values = np.asarray(values, dtype=np.float64)
    med = np.nanmedian(values)
    mad = 1.4826 * np.nanmedian(np.abs(values - med))
    if not np.isfinite(mad) or mad == 0:
        return np.zeros_like(values)
    return np.abs(values - med) / mad


def screen_windows(windows, channel_names, z_threshold=OUTLIER_Z):
    """Flag outlier windows in place (rejected / reject_reason), judged against the
    median and MAD of all windows given. Fewer than MIN_WINDOWS_FOR_SCREENING
    windows is too few to estimate the spread, so nothing is rejected."""
    usable = [w for w in windows if not w.failed]
    if len(usable) < MIN_WINDOWS_FOR_SCREENING:
        return
    metrics = {}
    for i, name in enumerate(channel_names):
        metrics[f"{name} count rate"] = [w.mean_rates[i] for w in usable]
    for kind in usable[0].amplitudes:
        metrics[f"{kind} amplitude"] = [w.amplitudes.get(kind, np.nan) for w in usable]
    for metric_name, values in metrics.items():
        z = _robust_z(values)
        for w, zi in zip(usable, z):
            if zi > z_threshold:
                w.rejected = True
                w.reject_reason = "; ".join(filter(None, [w.reject_reason, f"{metric_name} (z={zi:.1f})"]))


def average_windows(windows, label="average"):
    """Average the correlation curves of several equal-length, non-overlapping
    windows. The per-point error is the standard error of the mean across
    windows (independent, since they don't overlap). Needs >= 2 windows."""
    if len(windows) < 2:
        raise ValueError("Need at least 2 windows to average.")
    first = windows[0]
    corr, err = {}, {}
    for kind, r0 in first.corr_results.items():
        n_tau = min(len(w.corr_results[kind].tau) for w in windows)
        gs = np.array([w.corr_results[kind].g[:n_tau] for w in windows])
        ns = np.array([w.corr_results[kind].n_samples[:n_tau] for w in windows])
        corr[kind] = CorrelationResult(
            r0.tau[:n_tau].copy(), np.mean(gs, axis=0), np.sum(ns, axis=0),
            kind, r0.dt, r0.segments, r0.points_per_segment, r0.base,
        )
        err[kind] = np.std(gs, axis=0, ddof=1) / np.sqrt(len(windows))
    mean_rates = list(np.mean([w.mean_rates for w in windows], axis=0))
    return WindowCandidate(
        label, min(w.t0 for w in windows), max(w.t1 for w in windows), corr, err,
        first.eff_dt, [float(r) for r in mean_rates],
        {kind: _amplitude(r) for kind, r in corr.items()},
        kind="average", duration_s=float(sum(w.t1 - w.t0 for w in windows)), n_averaged=len(windows),
    )


def analyze_windows(
    time_arr, channels, window_length_s, step_s=STEP_S,
    segments=5, points_per_segment=15, base=4, bin_points=1, n_blocks=5, z_threshold=OUTLIER_Z,
):
    """Correlate every sliding and tiled window, screen out outliers, and average
    the accepted tiled windows.

    chosen defaults to that average. If fewer than 2 tiled windows survive, it
    falls back to a single correlation of the full trace.
    """
    t_min, t_max = float(time_arr[0]), float(time_arr[-1])
    sliding_starts = sliding_window_starts(t_min, t_max, window_length_s, step_s)
    tiled_starts = tiled_window_starts(t_min, t_max, window_length_s)

    cache = {}

    def get(t0):
        key = round(t0, 9)
        if key not in cache:
            t1 = min(t0 + window_length_s, t_max)
            label = f"{t0:.3g}-{t1:.3g}s"
            try:
                cache[key] = _correlate_window(
                    time_arr, channels, t0, t1, label, segments, points_per_segment, base, bin_points, n_blocks
                )
            except ValueError:
                cache[key] = WindowCandidate(label, t0, t1, {}, {}, failed=True, duration_s=t1 - t0)
        return cache[key]

    sliding = [get(s) for s in sliding_starts]
    tiled = [get(s) for s in tiled_starts]
    for w in tiled:
        w.tiled = True

    # Judge every window (sliding and tiled) against the same reference spread.
    all_windows = list({id(w): w for w in sliding + tiled}.values())
    screen_windows(all_windows, list(channels.keys()), z_threshold)

    accepted = [w for w in tiled if not w.failed and not w.rejected]
    average = None
    if len(accepted) >= 2:
        average = average_windows(accepted, label=f"Average of {len(accepted)}/{len(tiled)} windows")

    full_trace = None
    try:
        full_trace = _correlate_window(
            time_arr, channels, t_min, t_max, f"Full trace ({t_min:.3g}-{t_max:.3g}s)",
            segments, points_per_segment, base, bin_points, n_blocks,
        )
        full_trace.kind = "full"
    except ValueError:
        pass

    chosen = average or full_trace or next((w for w in sliding if not w.failed), sliding[0])
    return AutoWindowResult(chosen, sliding, average, full_trace, tiled)
