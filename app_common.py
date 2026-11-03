"""
app_common.py — shared helpers used by app.py and pages/*.py.

Kept at the project root (not inside core/) since this is UI-specific glue code
(Streamlit caching, plotting conventions) that core/ should stay free of.
"""

import os
import tempfile

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.io import load_trace_auto
from core.engine import _coarsen, select_window
from core.autowindow import analyze_windows

DEFAULT_SETTINGS = {
    "segments": 5,
    "points_per_segment": 15,
    "base": 4,
    "w_xy_um": 0.2,
    "w_z_um": 1.0,
    "min_reliable_n_samples": 10,
}

# Validated categorical palette (CVD-safe, fixed order -- see dataviz skill).
# Assigned by entity, not cycled: CH1 always slot 1, CH2 always slot 2, cross
# always slot 3, everywhere in the app (charts, comparison plots, metrics).
SERIES_COLORS = {
    "acf_ch1": "#2a78d6",  # slot 1 -- blue
    "acf_ch2": "#eb6834",  # slot 2 -- orange
    "cross": "#1baf7a",  # slot 3 -- aqua
}
SERIES_LABELS = {
    "acf_ch1": "CH1 autocorrelation",
    "acf_ch2": "CH2 autocorrelation",
    "cross": "Cross-correlation",
}
MUTED_INK = "#898781"  # axis/label text token -- never used for a data mark's identity
GRIDLINE = "#e1e0d9"  # hairline gridline color
REFERENCE_LINE = "#0b0b0b"  # neutral (non-categorical) color for "true value" annotation lines


def init_defaults():
    if "defaults" not in st.session_state:
        st.session_state["defaults"] = dict(DEFAULT_SETTINGS)


def kappa_from_w(w_xy_um, w_z_um):
    """kappa = w_z/w_xy, the structure parameter that actually enters the fit
    model's diffusion term -- the model has no other dependence on the beam
    waist, so w_xy/w_z are only ever used to derive this ratio, never directly."""
    return w_z_um / w_xy_um


def sidebar_about():
    """Small, consistent wayfinding blurb shown in the sidebar on every page --
    matters once the app is a public URL that people may land on directly via
    a shared link to a sub-page, with no context from the Home page."""
    with st.sidebar:
        st.caption(
            "**General-purpose FCS/FCCS data analysis**\n\n"
            "Analyzes ISS VistaVision trace exports (CSV). See the Home page "
            "for an overview of the pipeline."
        )


def format_seconds(x):
    """Human-scale formatting for a time constant (tau_D, dt, etc.): picks
    us/ms/s so a labmate doesn't have to mentally parse 2.1e-04."""
    if x is None or not np.isfinite(x):
        return "n/a"
    if x == 0:
        return "0 s"
    ax = abs(x)
    if ax < 1e-3:
        scale, unit, next_scale, next_unit = 1e6, "µs", 1e3, "ms"
    elif ax < 1.0:
        scale, unit, next_scale, next_unit = 1e3, "ms", 1.0, "s"
    else:
        scale, unit, next_scale, next_unit = 1.0, "s", None, None
    val = x * scale
    # 3-sig-fig rounding can push a value like 999.95 up a full order of magnitude
    # (displayed as "1e+03"); bump to the next unit before that happens instead.
    if next_scale is not None and abs(val) >= 999.5:
        val, unit = x * next_scale, next_unit
    return f"{val:.3g} {unit}"


def apply_chart_style(fig):
    """Hairline, recessive gridlines -- applied to every chart for a consistent look."""
    fig.update_xaxes(gridcolor=GRIDLINE, zerolinecolor=GRIDLINE, linecolor=GRIDLINE)
    fig.update_yaxes(gridcolor=GRIDLINE, zerolinecolor=GRIDLINE, linecolor=GRIDLINE)
    return fig


@st.cache_resource(show_spinner="Parsing trace file...")
def load_trace_cached(file_bytes, filename):
    """cache_resource (not cache_data): a real trace can be tens of millions of
    points, and cache_data deep-copies its return value on every call -- on a
    12M-point dual-channel trace that copy alone costs ~200MB and real time,
    repeated on every widget interaction. Nothing downstream mutates trace.time
    or trace.channels in place (windowing/binning always slice into new arrays),
    so sharing the same object across reruns is safe."""
    suffix = os.path.splitext(filename)[1] or ".csv"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        return load_trace_auto(tmp_path)
    finally:
        os.unlink(tmp_path)


@st.cache_resource(show_spinner="Correlating every window and screening out outliers...")
def run_auto_window_search_cached(
    file_bytes, filename, window_length_s, segments, points_per_segment, base, bin_points, n_blocks
):
    """Cached so re-running the page (e.g. clicking a fit button) doesn't redo the
    window analysis -- only a change to the file or a settings value does."""
    trace = load_trace_cached(file_bytes, filename)
    return analyze_windows(
        trace.time, trace.channels, window_length_s=window_length_s,
        segments=segments, points_per_segment=points_per_segment, base=base,
        bin_points=bin_points, n_blocks=n_blocks,
    )


WINDOW_LENGTH_OPTIONS_S = [1, 2, 3, 5, 10, 15, 20, 30, 60]


def window_length_options_for_trace(t_min, t_max):
    """Only offer window lengths that actually fit at least twice in the trace
    (so the slide has room to move); falls back to the whole-trace span if
    even the smallest fixed option doesn't fit at all."""
    span = t_max - t_min
    options = [opt for opt in WINDOW_LENGTH_OPTIONS_S if opt <= span]
    return options or [span]


def decimate_for_display(time_arr, y_arr, max_points=5000):
    """Down-sample purely for plotting a raw multi-million-row trace in the browser.
    This is a DISPLAY-ONLY decimation -- it never touches the arrays used for the
    actual 'Binning points' analysis control or the correlation engine."""
    n = len(time_arr)
    if n <= max_points:
        return time_arr, y_arr
    factor = int(np.ceil(n / max_points))
    return _coarsen(time_arr, factor), _coarsen(y_arr, factor)


def window_channels(time_arr, channels, t0, t1):
    """Apply the same [t0, t1] time-window trim to every channel array, reusing
    engine.select_window (validated) once per channel."""
    windowed_time = None
    windowed_channels = {}
    for name, arr in channels.items():
        t_w, arr_w = select_window(time_arr, arr, t0, t1)
        windowed_time = t_w
        windowed_channels[name] = arr_w
    return windowed_time, windowed_channels


def time_window_selector(time_arr, key_prefix):
    """A double-ended range slider for trimming the acquisition window, with the
    kept region shaded on a companion plot via add_vrect. Returns (t0, t1)."""
    t_min, t_max = float(time_arr[0]), float(time_arr[-1])
    t0, t1 = st.slider(
        "Time window to analyze (seconds)",
        min_value=t_min,
        max_value=t_max,
        value=(t_min, t_max),
        key=f"{key_prefix}_window_slider",
        help=(
            "Keeps only the intensity data between t0 and t1; everything outside is dropped "
            "before binning/correlation. Impact: use this to cut out a bad segment (laser "
            "drift, a bubble, focus loss) -- including bad data biases G(tau) at every lag, "
            "since the correlator's normalization uses the whole kept trace's mean intensity."
        ),
    )
    return t0, t1


def plot_raw_trace_with_window(time_arr, channels, t0, t1, max_points=5000, kept_spans=None, rejected_spans=None):
    """Raw trace with the analyzed region shaded. kept_spans/rejected_spans
    (lists of (t0, t1)) override the single [t0, t1] span, e.g. to show which
    windows went into an average and which were screened out."""
    disp_time, _ = decimate_for_display(time_arr, next(iter(channels.values())), max_points)
    fig = go.Figure()
    channel_colors = {"CH1": SERIES_COLORS["acf_ch1"], "CH2": SERIES_COLORS["acf_ch2"]}
    for name, arr in channels.items():
        _, disp_y = decimate_for_display(time_arr, arr, max_points)
        fig.add_trace(
            go.Scatter(
                x=disp_time, y=disp_y, mode="lines", name=name,
                line=dict(width=2, color=channel_colors.get(name)),
            )
        )
    if kept_spans is None and rejected_spans is None:
        fig.add_vrect(x0=t0, x1=t1, fillcolor="#2a78d6", opacity=0.08, line_width=0, annotation_text="kept window")
    for k0, k1 in kept_spans or []:
        fig.add_vrect(x0=k0, x1=k1, fillcolor="#2a78d6", opacity=0.08, line_width=0)
    for r0, r1 in rejected_spans or []:
        fig.add_vrect(x0=r0, x1=r1, fillcolor="#d62a2a", opacity=0.15, line_width=0, annotation_text="rejected")
    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Photon counts / bin",
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h") if len(channels) > 1 else dict(),
        showlegend=len(channels) > 1,
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
    )
    return apply_chart_style(fig)


def plot_correlation_curve(results, min_reliable_n_samples, title="", height=400):
    """Semilog-x G(tau) plot (tau log-scale, G linear -- not log-log, since G can
    go negative at long lags from noise). Points below the reliability threshold
    are shown with a distinct marker so long-tau statistical unreliability is
    visually obvious, not hidden."""
    fig = go.Figure()
    any_unreliable = False
    for kind, result in results.items():
        color = SERIES_COLORS.get(kind, MUTED_INK)
        reliable = result.n_samples >= min_reliable_n_samples
        if not np.all(reliable):
            any_unreliable = True
        fig.add_trace(
            go.Scatter(
                x=result.tau[reliable],
                y=result.g[reliable],
                mode="markers",
                name=SERIES_LABELS.get(kind, kind),
                marker=dict(color=color, size=8, line=dict(width=1, color="#fcfcfb")),
            )
        )
        if np.any(~reliable):
            fig.add_trace(
                go.Scatter(
                    x=result.tau[~reliable],
                    y=result.g[~reliable],
                    mode="markers",
                    name=f"{SERIES_LABELS.get(kind, kind)} (low sample count)",
                    marker=dict(color=color, size=9, symbol="x", opacity=0.55),
                )
            )
    fig.update_xaxes(type="log", title="tau (s)")
    fig.update_yaxes(title="G(tau)")
    fig.update_layout(
        title=title, height=height, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h"), plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
    )
    return apply_chart_style(fig), any_unreliable


def plot_fit_overlay(tau, g, fit_curve_arr, title="", color=None, err=None, fit_tau_min=None, fit_tau_max=None):
    """Data (with error bars if err is given) and fitted curve. Lags outside
    [fit_tau_min, fit_tau_max] are shaded grey: shown, but not used by the fit."""
    color = color or SERIES_COLORS["acf_ch1"]
    fig = go.Figure()
    error_y = None
    if err is not None and len(err):
        error_y = dict(type="data", array=np.where(np.isfinite(err), err, 0), thickness=1, width=0, color=MUTED_INK)
    fig.add_trace(
        go.Scatter(
            x=tau, y=g, mode="markers", name="data", error_y=error_y,
            marker=dict(size=8, color=MUTED_INK, opacity=0.7),
        )
    )
    tau_pos = tau[tau > 0]
    if fit_tau_min is not None and len(tau_pos) and tau_pos.min() < fit_tau_min:
        fig.add_vrect(x0=tau_pos.min(), x1=fit_tau_min, fillcolor="#888888", opacity=0.12, line_width=0,
                      annotation_text="not fitted", annotation_position="top left")
    if fit_tau_max is not None and len(tau_pos) and tau_pos.max() > fit_tau_max:
        fig.add_vrect(x0=fit_tau_max, x1=tau_pos.max(), fillcolor="#888888", opacity=0.12, line_width=0)
    order = np.argsort(tau)
    fig.add_trace(
        go.Scatter(x=tau[order], y=fit_curve_arr[order], mode="lines", name="fit", line=dict(color=color, width=2))
    )
    fig.update_xaxes(type="log", title="tau (s)")
    fig.update_yaxes(title="G(tau)")
    fig.update_layout(
        title=title, height=350, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h"), plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
    )
    return apply_chart_style(fig)


def fit_settings_widgets(key_prefix, label):
    st.markdown(f"**{label} fit settings**")
    n_components = st.radio(
        "Components", [1, 2], key=f"{key_prefix}_ncomp", horizontal=True,
        help=(
            "How many diffusing species the model assumes.\n\n"
            r"Formula: $G(\tau) = \frac{1}{N}\sum_i f_i D_i(\tau)$, $\sum_i f_i = 1$, "
            r"$D_i(\tau) = \frac{1}{1+\tau/\tau_{Di}} \cdot \frac{1}{\sqrt{1+\tau/(\kappa^2 \tau_{Di})}}$"
            "\n\n"
            "Impact: 1 component fits a single tau_D (one population size). 2 components "
            "fits two independent tau_D's plus their amplitude split (f1/f2) -- use this if "
            "the curve visibly has a fast and a slow decay (e.g. free monomer + aggregate). "
            "More parameters also means more risk of an unstable/degenerate fit -- check the "
            "fit quality verdict after fitting."
        ),
    )
    triplet = st.checkbox(
        "Include triplet/blinking term", key=f"{key_prefix}_triplet",
        help=(
            "Adds a fast photophysical-blinking prefactor on top of the diffusion term.\n\n"
            r"Formula: multiplies $G(\tau)$ by $\frac{1-T+T e^{-\tau/\tau_{trip}}}{1-T}$, with "
            r"$\tau_{trip} \sim 1\,\mu s$."
            "\n\n"
            "Impact: only reshapes the fastest (leftmost) part of the curve. Turn on if you "
            "see a fast decay at short tau that the diffusion term alone can't explain "
            "(common with organic-dye triplet states). Leave off if your shortest tau is "
            "already much longer than ~1 us, or it just adds an unconstrained nuisance "
            "parameter."
        ),
    )
    n_starts = st.number_input(
        "Multi-start attempts", min_value=1, max_value=20, value=5, key=f"{key_prefix}_nstarts",
        help=(
            "How many independent random initial guesses are fit and compared.\n\n"
            "Impact: this is purely a stability check, not part of G(tau)'s formula -- it "
            "doesn't change a converged fit's best answer, but more starts make it more "
            "likely to catch a fit that's actually unstable (parameters swing >20% depending "
            "on starting guess, or <60% of starts converge at all). Increase this for "
            "2-component fits, which are more prone to degenerate solutions than 1-component."
        ),
    )
    offset = st.checkbox(
        "Fit a baseline offset G∞", value=True, key=f"{key_prefix}_offset",
        help=(
            "Adds a constant to the model.\n\n"
            r"Formula: $G(\tau) = G_\infty + \frac{1}{N}\sum_i f_i D_i(\tau)$"
            "\n\n"
            "Impact: absorbs fluctuations much slower than diffusion (aggregates, drift), "
            "which look like a flat plateau across the lag range. Without it the fit can only "
            "explain that plateau by stretching tau_D, often to seconds. Leave on unless the "
            "curve clearly decays to zero within the lag range."
        ),
    )
    tau_min_us = st.number_input(
        "Fit lags from (µs)", min_value=0.0, value=20.0, step=5.0, key=f"{key_prefix}_tau_min_us",
        help=(
            "Lags shorter than this are shown but left out of the fit.\n\n"
            "Impact: the earliest autocorrelation lags carry detector artifacts "
            "(afterpulsing, dead time) that add a spurious fast spike. 10-20 µs is a common "
            "cutoff. Cross-correlations between two detectors don't have this artifact."
        ),
    )
    fixed_tauD1 = None
    if n_components == 2:
        fix = st.checkbox(
            "Fix τ_D1 (fast component) at a known value", key=f"{key_prefix}_fix_tauD1",
            help=(
                "Holds the faster component's diffusion time at a value you enter, e.g. the "
                "monomer tau_D from a control sample, and fits only the slower component and "
                "the split between them.\n\n"
                "Impact: on noisy curves a free 2-component fit can trade the two components "
                "off against each other almost arbitrarily. Fixing one removes most of that. "
                "Only as good as the control: use a value that was itself well determined."
            ),
        )
        if fix:
            fixed_tauD1 = st.number_input(
                "τ_D1 (µs)", min_value=0.1, value=300.0, step=10.0, key=f"{key_prefix}_fixed_tauD1_us",
            ) * 1e-6
    return dict(
        n_components=n_components, triplet=triplet, n_starts=n_starts, offset=offset,
        tau_min=tau_min_us * 1e-6, fixed_tauD1=fixed_tauD1,
    )
