import numpy as np
import pandas as pd
import streamlit as st

from app_common import (
    init_defaults,
    sidebar_about,
    load_trace_cached,
    run_auto_window_search_cached,
    window_length_options_for_trace,
    plot_raw_trace_with_window,
    plot_correlation_curve,
    plot_fit_overlay,
    fit_settings_widgets,
    format_seconds,
    kappa_from_w,
    SERIES_COLORS,
)

from core.correlate import PLATEAU_TAU_S, plateau_level, smooth_error
from core.fitting import multi_start_fit, QUALITY_GOOD, QUALITY_ROUGH, QUALITY_UNRESOLVED, TAUD_EDGE_FACTOR, format_rel_err
from core.fccs import bound_fraction_from_cross, amplitude_at_zero
from core.export import (
    correlation_results_to_df,
    correlation_results_to_df_with_error,
    correlation_results_to_vistavision_csv_text,
    fit_result_to_df,
    stability_report_to_df,
    fccs_result_to_df,
    df_to_csv_bytes,
)

# Above this raw sampling interval, sub-millisecond decays are poorly sampled.
SLOW_SAMPLING_DT_S = 20e-6

st.set_page_config(page_title="Single File Analysis", layout="wide")
init_defaults()
sidebar_about()
d = st.session_state["defaults"]

st.title("Single File Analysis")
st.caption(
    "Upload a trace and pick a window length. The app splits the acquisition into "
    "windows of that length, screens out outlier windows (bleaching, drift, aggregates), "
    "and averages the rest. No manual trimming required."
)

uploaded = st.file_uploader("VistaVision trace export (single- or dual-channel CSV)", type=["csv"])
if uploaded is None:
    st.info("Upload a trace file to begin.")
    st.stop()

trace = load_trace_cached(uploaded.getvalue(), uploaded.name)

info_cols = st.columns(4)
info_cols[0].metric("Channels", trace.n_channels)
info_cols[1].metric("Rows", f"{trace.n_rows:,}")
info_cols[2].metric("Raw dt", format_seconds(trace.dt))
info_cols[3].metric("Raw sampling rate", f"{1.0 / trace.dt:,.0f} Hz")

if trace.dt > SLOW_SAMPLING_DT_S:
    st.warning(
        f"Sampled at {1.0 / trace.dt:,.0f} Hz, so the first lag is {format_seconds(trace.dt)}. "
        f"Diffusion times shorter than about {format_seconds(TAUD_EDGE_FACTOR * trace.dt)} can't be "
        "resolved from this file -- the decay happens before the first lag. If you expect "
        "sub-millisecond diffusion times, use a recording at a higher sampling rate."
    )

if trace.used_fallback_parser:
    st.warning(f"Fast parser was bypassed for this file: {trace.fallback_reason}")

t_min, t_max = float(trace.time[0]), float(trace.time[-1])
window_options = window_length_options_for_trace(t_min, t_max)
default_window_length_idx = window_options.index(3) if 3 in window_options else len(window_options) // 2
window_length_s = st.selectbox(
    "Time period per window (s)",
    options=window_options,
    index=default_window_length_idx,
    key="sf_window_length_s",
    help=(
        "The trace is split into non-overlapping windows of this length (0-L, L-2L, ...). "
        "Outlier windows are rejected and the rest are averaged into one curve, with error "
        "bars from the spread across windows. Shorter windows give more windows (better error "
        "bars, finer outlier rejection) but each must still be much longer than the slowest "
        "decay you want to resolve. Overlapping 1-second-step windows are also computed so "
        "you can inspect or pick a single stretch by hand."
    ),
)

with st.expander("Advanced settings (defaults match VistaVision manual sec. 13.4.1)"):
    c1, c2, c3 = st.columns(3)
    segments = c1.number_input(
        "Segments", min_value=1, max_value=20, value=d["segments"], key="sf_segments",
        help=(
            "How many multi-tau blocks are computed, each at a coarser lag spacing than "
            "the last.\n\n"
            r"Formula: total $\tau$ points $= \text{segments} \times \text{points\_per\_segment}$; "
            r"segment $k$'s lag step $\Delta t_k = \Delta t_{raw} \cdot \text{base}^k$."
            "\n\n"
            "Impact: more segments extend how far out in tau the curve reaches, but each "
            "extra segment needs proportionally more trace length to fill with real data."
        ),
    )
    points_per_segment = c2.number_input(
        "Points per segment", min_value=1, max_value=100, value=d["points_per_segment"], key="sf_pps",
        help=(
            "How many correlation points are computed at each segment's lag spacing before "
            "moving to the next (coarser) segment.\n\n"
            r"Formula: within segment $k$, points are computed at lags "
            r"$\tau = \Delta t_{raw} \cdot \text{base}^k \cdot i$, for $i = 1, \dots, \text{points\_per\_segment}$."
            "\n\n"
            "15 is VistaVision's own documented default (manual sec. 13.4.1)."
        ),
    )
    base = c3.number_input(
        "Grouping base", min_value=2, max_value=16, value=d["base"], key="sf_base",
        help=(
            "The coarsening factor applied to the lag spacing between successive segments.\n\n"
            r"Formula: segment $k$'s effective $\Delta t_k = \Delta t_{raw} \cdot \text{base}^k$."
            "\n\n"
            "base=4 (the standard multi-tau choice) roughly quadruples the tau spacing each "
            "segment, giving even log-spaced coverage across many decades of tau."
        ),
    )
    c4, c5 = st.columns(2)
    bin_points = c4.number_input(
        "Binning points (group N raw points together before correlating)", min_value=1, value=1, key="sf_bin_points",
        help=(
            "Groups N consecutive raw samples into one averaged point BEFORE correlating.\n\n"
            r"Formula: $\Delta t_{eff} = \Delta t_{raw} \cdot N$. N=1 means no binning is applied "
            "at all.\n\n"
            "Impact: binning reduces shot-noise per point but also blurs out any real "
            r"correlation decay faster than $\Delta t_{eff}$ -- most FCS analyses leave this at 1."
        ),
    )
    n_blocks = c5.number_input(
        "Sub-blocks for single-window error bars", min_value=2, max_value=50, value=5, key="sf_n_blocks",
        help=(
            "Only used when a single window or the full trace is selected: its error bars "
            "come from splitting it into this many sub-blocks. The averaged curve's error bars "
            "come from the spread across windows instead. More sub-blocks = a more stable "
            "error estimate but each sub-block is shorter."
        ),
    )
    wcol1, wcol2 = st.columns(2)
    w_xy_um = wcol1.number_input(
        "w_xy (µm, radial)", min_value=0.01, value=d["w_xy_um"], step=0.01, format="%.3f", key="sf_w_xy",
        help="The confocal detection volume's radial (lateral) beam waist, in microns. Only ever "
        "used to derive kappa below -- the fit model has no separate dependence on its absolute "
        "value. Overrides the Home page's global default for this session/page only.",
    )
    w_z_um = wcol2.number_input(
        "w_z (µm, axial)", min_value=0.01, value=d["w_z_um"], step=0.01, format="%.3f", key="sf_w_z",
        help="The confocal detection volume's axial beam waist, in microns. Only ever used to "
        "derive kappa below. Overrides the Home page's global default for this session/page only.",
    )
    kappa = kappa_from_w(w_xy_um, w_z_um)
    st.caption(
        f"kappa = w_z / w_xy = **{kappa:.3g}** -- the only quantity that actually enters the fit "
        r"model's diffusion term $\frac{1}{\sqrt{1+\tau/(\kappa^2 \tau_D)}}$, not in computing "
        "G(tau) itself. Fixed rather than floated because it's badly degenerate with tau_D "
        "without an independent volume calibration -- a wrong value biases the fitted tau_D, "
        "especially at long tau. Typical confocal setups: kappa ~ 3-6."
    )

run_id = (
    uploaded.name, uploaded.size, window_length_s, segments, points_per_segment, base, bin_points, n_blocks, kappa,
)

auto = run_auto_window_search_cached(
    uploaded.getvalue(), uploaded.name, window_length_s, segments, points_per_segment, base, bin_points, n_blocks
)
valid_candidates = [c for c in auto.candidates if not c.failed]
if not valid_candidates and auto.full_trace is None:
    st.error(
        "The trace couldn't be correlated with these settings. Try fewer segments or "
        "points per segment in Advanced settings."
    )
    st.stop()


def _option_label(c):
    if c.kind == "average":
        return f"{c.label} (recommended)"
    if c.kind == "full":
        return c.label
    return f"{c.label} -- outlier: {c.reject_reason}" if c.rejected else c.label


selectable = [c for c in (auto.average, auto.full_trace) if c is not None] + valid_candidates
option_labels = [_option_label(c) for c in selectable]
label_lookup = dict(zip(option_labels, selectable))
default_option = _option_label(auto.chosen)

if st.session_state.get("sf_run_id") != run_id:
    st.session_state["sf_run_id"] = run_id
    st.session_state.pop("sf_fit_results", None)
    st.session_state["sf_window_choice"] = default_option
    st.session_state["sf_fit_window_acf_ch1"] = default_option
    st.session_state["sf_fit_window_acf_ch2"] = default_option

tiled_ok = [w for w in auto.tiled if not w.failed]
tiled_rejected = [w for w in tiled_ok if w.rejected]
channel_names = list(trace.channels.keys())

with st.container(border=True):
    st.subheader("1. Time window")
    selected_option = st.selectbox(
        "Curve to use",
        options=option_labels,
        key="sf_window_choice",
        help=(
            "Defaults to the average of all non-outlier windows. You can also use the full "
            "trace as one curve, or any single sliding window. The curve, fit, and exports "
            "below all follow this choice."
        ),
    )
    chosen = label_lookup[selected_option]
    if chosen.kind == "average":
        kept = [(w.t0, w.t1) for w in tiled_ok if not w.rejected]
        st.plotly_chart(
            plot_raw_trace_with_window(
                trace.time, trace.channels, chosen.t0, chosen.t1,
                kept_spans=kept, rejected_spans=[(w.t0, w.t1) for w in tiled_rejected],
            ),
            width="stretch",
        )
    else:
        st.plotly_chart(plot_raw_trace_with_window(trace.time, trace.channels, chosen.t0, chosen.t1), width="stretch")

    if auto.average is not None:
        st.caption(
            f"Split the trace into {len(tiled_ok)} non-overlapping {window_length_s}s windows; "
            f"{len(tiled_rejected)} rejected as outliers, {auto.average.n_averaged} averaged."
        )
        if auto.average.n_averaged < 5:
            st.warning(
                f"Only {auto.average.n_averaged} windows went into the average, so its error bars "
                "are rough. A shorter window length gives more windows."
            )
        if tiled_ok and len(tiled_rejected) / len(tiled_ok) > 1 / 3:
            st.warning(
                f"{len(tiled_rejected)} of {len(tiled_ok)} windows were rejected. The trace may be "
                "unstable (bleaching, drift, or frequent aggregates). Check the raw trace above."
            )
    else:
        st.info(
            "Fewer than 2 usable windows of this length fit in the trace, so there's no average. "
            "Using the full trace as a single curve instead. Pick a shorter window length to "
            "enable averaging."
        )

    with st.expander(f"Outlier screening ({len(tiled_ok)} windows)"):
        screen_rows = []
        for w in tiled_ok:
            row = {"window": w.label, "t0 (s)": w.t0, "t1 (s)": w.t1}
            for name, rate in zip(channel_names, w.mean_rates):
                row[f"{name} rate (Hz)"] = rate
            for kind, amp in w.amplitudes.items():
                row[f"{kind} amplitude"] = amp
            row["in average"] = auto.average is not None and not w.rejected
            row["rejected because"] = w.reject_reason
            screen_rows.append(row)
        st.dataframe(pd.DataFrame(screen_rows), width="stretch", hide_index=True)
        st.caption(
            "A window is rejected if its mean count rate (any channel) or its correlation "
            "amplitude (mean G over the first 5 lags, any curve) is more than 3.5 robust "
            "standard deviations (1.4826 x MAD) from the median across windows. See the "
            "Methodology page for details."
        )

    with st.expander(f"View all {len(valid_candidates)} sliding-window curves"):
        st.caption(
            "Overlapping windows in 1 s steps, for inspection. Click \"Use this window\" under "
            "any plot to analyze just that stretch."
        )
        grid_cols = st.columns(3)
        for i, c in enumerate(valid_candidates):
            with grid_cols[i % 3]:
                with st.container(border=True):
                    tags = []
                    if c.rejected:
                        tags.append("outlier")
                    if c.label == chosen.label:
                        tags.append("selected")
                    title = c.label + (f" ({', '.join(tags)})" if tags else "")
                    fig, _ = plot_correlation_curve(
                        c.corr_results, d["min_reliable_n_samples"], title=title, height=260
                    )
                    st.plotly_chart(fig, width="stretch", key=f"sf_grid_plot_{c.label}")
                    if c.rejected:
                        st.caption(f"Outlier: {c.reject_reason}")
                    if c.label != chosen.label:
                        # Setting st.session_state[key] for an already-instantiated widget raises --
                        # must be done in an on_click callback, which runs before the rerun, not here.
                        st.button(
                            "Use this window", key=f"sf_grid_use_{c.label}",
                            on_click=lambda opt=_option_label(c): st.session_state.update(sf_window_choice=opt),
                        )
                    else:
                        st.caption("Currently selected")

results = chosen.corr_results
errors = chosen.corr_errors

with st.container(border=True):
    st.subheader("2. Correlation curves")
    st.caption(
        f"Window: **{chosen.label}** "
        f"(t = {format_seconds(chosen.t0)} to {format_seconds(chosen.t1)})"
    )
    fig, any_unreliable = plot_correlation_curve(
        results, d["min_reliable_n_samples"], title=f"{uploaded.name} -- {chosen.label}"
    )
    st.plotly_chart(fig, width="stretch")
    if any_unreliable:
        st.caption(
            f"Points marked with 'x' are based on fewer than {d['min_reliable_n_samples']} averaged samples "
            "(long-tau statistical unreliability, especially near the edge of the acquisition window)."
        )

    curve_download_labels = {"acf_ch1": "CH1 autocorrelation", "acf_ch2": "CH2 autocorrelation", "cross": "Cross-correlation"}

    if auto.full_trace is not None:
        st.markdown("**Slow-component plateau** (full trace, G at ~10 ms)")
        plateau_rows = []
        for kind, cr in auto.full_trace.corr_results.items():
            pl = plateau_level(cr, auto.full_trace.corr_errors.get(kind), PLATEAU_TAU_S)
            if pl is not None:
                plateau_rows.append(
                    {"curve": curve_download_labels[kind], "lag used": format_seconds(pl[0]), "G": pl[1], "error": pl[2]}
                )
        if plateau_rows:
            plateau_df = pd.DataFrame(plateau_rows)
            st.dataframe(plateau_df, hide_index=True, column_config={
                "G": st.column_config.NumberColumn(format="%.4f"),
                "error": st.column_config.NumberColumn(format="%.4f"),
            })
            st.download_button(
                "Download plateau levels (CSV)", df_to_csv_bytes(plateau_df.assign(file=uploaded.name)),
                file_name=f"{uploaded.name}_plateau.csv", key="sf_dl_plateau",
            )
            st.caption(
                "By ~10 ms sub-millisecond diffusion has decayed away, so this is the contribution of "
                "fluctuations slower than the lag range: large slow species (aggregates) or slow drift. "
                "A rising cross-correlation plateau means more slow particles carrying both labels. "
                "It's always taken from the full trace -- short windows each normalize by their own mean, "
                "which removes most of the slow component. It scales with 1/concentration, so compare "
                "samples of similar count rate. A common artifact (crosstalk, laser/focus drift) "
                "raises it too: check single-label and buffer controls. Error is from sub-blocks of the "
                "trace and is approximate for components this slow."
            )
        else:
            st.caption(
                "Slow-component plateau: the lag grid stops short of 10 ms. Raise 'Segments' in "
                "Advanced settings to reach it."
            )

    curve_choice = st.radio(
        "Curve to include in the CSV downloads below",
        options=["All curves"] + [curve_download_labels[k] for k in results],
        horizontal=True, key="sf_curve_download_choice",
        help="Restricts the two CSV downloads to just one curve, e.g. to get a CH1-only file. "
        "The VistaVision-format download always includes every curve, matching a real "
        "VistaVision export's whole-measurement structure.",
    )
    if curve_choice == "All curves":
        dl_results, dl_errors = results, errors
    else:
        kind_for_choice = next(k for k, v in curve_download_labels.items() if v == curve_choice)
        dl_results = {kind_for_choice: results[kind_for_choice]}
        dl_errors = {kind_for_choice: errors.get(kind_for_choice)}

    dl_cols = st.columns(3)
    dl_cols[0].download_button(
        "Download correlation curves (CSV)",
        df_to_csv_bytes(correlation_results_to_df(dl_results)),
        file_name=f"{uploaded.name}_correlation.csv",
        help="tau, G(tau), n_samples per curve -- no error column.",
    )
    dl_cols[1].download_button(
        "Download with error (CSV)",
        df_to_csv_bytes(correlation_results_to_df_with_error(dl_results, dl_errors)),
        file_name=f"{uploaded.name}_correlation_with_error.csv",
        help="tau, G(tau), error, n_samples per curve.",
    )
    dl_cols[2].download_button(
        "Download (VistaVision format)",
        correlation_results_to_vistavision_csv_text(
            results, errors, sampling_rate_hz=1.0 / chosen.eff_dt, mean_rates=chosen.mean_rates,
            measurement_time_s=chosen.duration_s, raw_data_file=uploaded.name,
        ),
        file_name=f"{uploaded.name}_correlation_vistavision.csv",
        help="Matches a real VistaVision [HeaderV]/[Data] export's structure (metadata header + one "
        "row per quantity: Tau, ChN AutoCorrelation, ChN AutoCorrelation Standard Deviation, "
        "Ch0x1 CrossCorrelation). Does NOT reproduce VistaVision's segment-boundary duplicate Tau "
        "values -- an undocumented internal quirk with no known rule to reproduce without the raw "
        "trace that produced it.",
    )

with st.container(border=True):
    st.subheader("3. Model fitting (optional)")
    st.caption(
        "CH1 and CH2 can each be fit from a different curve from step 1 (the average, the full "
        "trace, or a single window). Cross-correlation needs both channels sampled over the exact "
        "same data (it's a paired CH1(t)*CH2(t+tau) product), so it's only available when both "
        "channels currently use the same one."
    )
    fit_results = st.session_state.setdefault("sf_fit_results", {})
    fit_window_used = st.session_state.setdefault("sf_fit_window_used", {})

    fit_panels = [("acf_ch1", "CH1 autocorrelation"), ("acf_ch2", "CH2 autocorrelation"), ("cross", "Cross-correlation")]
    fit_cols = st.columns(sum(1 for kind, _ in fit_panels if kind in results))
    col_idx = 0
    for kind, label in fit_panels:
        if kind not in results:
            continue
        with fit_cols[col_idx]:
            if kind in ("acf_ch1", "acf_ch2"):
                fit_window_option = st.selectbox(
                    f"Window for {label} fit", options=option_labels, key=f"sf_fit_window_{kind}",
                    help="Defaults to the curve selected in step 1, but you can fit this channel "
                    "from a different one than the other channel.",
                )
                kind_result = label_lookup[fit_window_option].corr_results[kind]
                kind_error = label_lookup[fit_window_option].corr_errors.get(kind)
            else:
                ch1_window = st.session_state.get("sf_fit_window_acf_ch1")
                ch2_window = st.session_state.get("sf_fit_window_acf_ch2")
                if ch1_window != ch2_window:
                    fit_results.pop("cross", None)
                    fit_window_used.pop("cross", None)
                    st.info(
                        "CH1 and CH2 are currently using different windows "
                        f"({ch1_window} vs {ch2_window}), so cross-correlation isn't available -- "
                        "pick the same window for both to enable it."
                    )
                    col_idx += 1
                    continue
                fit_window_option = ch1_window
                kind_result = label_lookup[fit_window_option].corr_results[kind]
                kind_error = label_lookup[fit_window_option].corr_errors.get(kind)
                st.caption(f"Using window: {fit_window_option}")

            if fit_window_used.get(kind) != fit_window_option:
                # The window for this channel changed since the last fit -- drop the stale
                # result rather than show a fit overlaid on data from a different window.
                fit_results.pop(kind, None)

            fit_opts = fit_settings_widgets(f"sf_{kind}", label)
            if st.button(f"Fit {label}", key=f"sf_fit_btn_{kind}"):
                # Weight each tau point by 1/sigma_G so the optimizer (and the reported
                # reduced chi-squared) actually accounts for per-point uncertainty, instead
                # of treating every point as equally reliable. sigma is smoothed across tau
                # first so a few points with luckily small sigma don't dominate the fit.
                # Points with an unknown/zero sigma (too few sub-blocks/windows reached that
                # tau) get weight 0 -- excluded from the fit rather than assigned a made-up
                # uncertainty.
                if kind_error is not None and len(kind_error):
                    sigma = smooth_error(kind_error)
                    weights = np.where(np.isfinite(sigma) & (sigma > 0), 1.0 / sigma, 0.0)
                else:
                    weights = None
                with st.spinner(f"Fitting {label} ({fit_opts['n_starts']} starts)..."):
                    try:
                        report = multi_start_fit(
                            kind_result.tau, kind_result.g, kappa=kappa, weights=weights, **fit_opts,
                        )
                    except ValueError as exc:
                        report = None
                        st.error(f"Can't fit: {exc} Lower the 'Fit lags from' cutoff.")
                if report is not None:
                    fit_results[kind] = report
                    fit_window_used[kind] = fit_window_option
            if kind in fit_results:
                report = fit_results[kind]
                fr = report.chosen_result
                verdict_box = {QUALITY_GOOD: st.success, QUALITY_ROUGH: st.warning}.get(fr.quality, st.error)
                verdict = f"tau_D {fr.quality}"
                if fr.quality_reasons:
                    verdict += ": " + "; ".join(fr.quality_reasons)
                verdict_box(verdict)
                st.plotly_chart(
                    plot_fit_overlay(
                        kind_result.tau, kind_result.g, fr.fit_curve, title=label, color=SERIES_COLORS.get(kind),
                        err=kind_error, fit_tau_min=fr.fit_tau_min, fit_tau_max=fr.fit_tau_max,
                    ),
                    width="stretch",
                )
                for i, (td, rel) in enumerate(zip(fr.tauD, fr.tauD_rel_err)):
                    name = f"tau_D{'' if len(fr.tauD) == 1 else f' ({i + 1})'}"
                    if i == 0 and fr.fixed_tauD1 is not None:
                        st.metric(name, f"{format_seconds(td)} (fixed)")
                    else:
                        st.metric(
                            name,
                            f"{format_seconds(td)} ± {format_rel_err(rel)}" if np.isfinite(rel) else f"{format_seconds(td)} ± ?",
                        )
                if fr.n_components == 2 and "f1" in fr.params:
                    f1, f1_se = fr.params["f1"], fr.params_stderr.get("f1", float("nan"))
                    st.metric(
                        "Fast-component fraction f1",
                        f"{f1:.2f} ± {f1_se:.2f}" if np.isfinite(f1_se) else f"{f1:.2f} ± ?",
                    )
                st.metric("N (amplitude)", f"{fr.N:.3g}")
                st.caption(
                    f"Fitted {fr.n_points_fit} lags from {format_seconds(fr.fit_tau_min)} to "
                    f"{format_seconds(fr.fit_tau_max)}. Baseline G∞ = {fr.G_inf:.3g}. "
                    f"Reduced chi-squared = {fr.redchi:.4g}. Starting-guess check: "
                    + ("consistent" if report.is_stable else "inconsistent")
                    + f" ({report.converged_fraction:.0%} of {len(report.per_start_results)} starts converged)."
                )
                with st.expander("Per-start results (stability check)"):
                    st.dataframe(stability_report_to_df(report, label=label))
                st.download_button(
                    f"Download {label} fit parameters (CSV)",
                    df_to_csv_bytes(fit_result_to_df(fr, label=label)),
                    file_name=f"{uploaded.name}_{kind}_fit.csv",
                    key=f"sf_dl_fit_{kind}",
                )
        col_idx += 1

if "acf_ch1" in fit_results and "acf_ch2" in fit_results and "cross" in fit_results:
    with st.container(border=True):
        st.subheader("4. FCCS bound fraction")
        g_ch1_0 = amplitude_at_zero(fit_results["acf_ch1"].chosen_result)
        g_ch2_0 = amplitude_at_zero(fit_results["acf_ch2"].chosen_result)
        g_x_0 = amplitude_at_zero(fit_results["cross"].chosen_result)
        bound_ch2_species = bound_fraction_from_cross(g_x_0, g_ch1_0)  # CH2 species' bound fraction uses CH1 partner amplitude
        bound_ch1_species = bound_fraction_from_cross(g_x_0, g_ch2_0)  # CH1 species' bound fraction uses CH2 partner amplitude

        bcols = st.columns(2)
        bcols[0].metric("Bound fraction of CH2 species", f"{bound_ch2_species:.3f}")
        bcols[1].metric("Bound fraction of CH1 species", f"{bound_ch1_species:.3f}")
        undetermined = [
            curve_download_labels[k] for k in ("acf_ch1", "acf_ch2", "cross")
            if fit_results[k].chosen_result.quality == QUALITY_UNRESOLVED
        ]
        if undetermined:
            st.error(
                "Not meaningful: the fit for " + ", ".join(undetermined) + " is not determined, so "
                "its amplitude (and these ratios) are not reliable."
            )
        out_of_range = [v for v in (bound_ch2_species, bound_ch1_species) if np.isfinite(v) and not 0 <= v <= 1]
        if out_of_range:
            st.warning(
                "A bound fraction outside 0-1 is physically impossible. It points to noisy "
                "amplitudes, spectral crosstalk, or unequal detection volumes in the two channels."
            )
        st.caption("No spectral crosstalk correction applied (uncorrected amplitude-ratio formula).")
        st.download_button(
            "Download FCCS bound fractions (CSV)",
            df_to_csv_bytes(fccs_result_to_df(bound_ch2_species, bound_ch1_species, label=uploaded.name)),
            file_name=f"{uploaded.name}_fccs.csv",
        )
