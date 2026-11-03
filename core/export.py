"""
core/export.py — turn result dataclasses into pandas DataFrames / CSV text for
download at every stage (raw correlation curves, fit parameters, stability
reports, Kd fits, batch comparisons).
"""

import numpy as np
import pandas as pd


def correlation_result_to_df(result):
    return pd.DataFrame({"tau_seconds": result.tau, "G_tau": result.g, "n_samples": result.n_samples})


def correlation_results_to_df(results):
    """results: dict[kind -> CorrelationResult]. One combined long-format DataFrame."""
    frames = []
    for kind, result in results.items():
        df = correlation_result_to_df(result)
        df.insert(0, "kind", kind)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def correlation_results_to_df_with_error(results, errors):
    """Same long format as correlation_results_to_df, plus a per-point standard-error
    column from core.correlate.compute_correlation_error. errors: dict[kind -> ndarray],
    same length/order as each result's tau array."""
    frames = []
    for kind, result in results.items():
        df = correlation_result_to_df(result)
        df.insert(0, "kind", kind)
        df["error"] = errors.get(kind)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def correlation_results_to_vistavision_csv_text(
    results, errors, sampling_rate_hz, mean_rates, measurement_time_s, raw_data_file="", created=None,
):
    """Matches a real VistaVision [HeaderV]/[Data] export's structure: a metadata
    header block, then one label line + one comma-separated data row per quantity
    (Tau, ChN AutoCorrelation, ChN AutoCorrelation Standard Deviation, Ch0x1
    CrossCorrelation, Ch0x1 CrossCorrelation Standard Deviation) -- NOT the
    row-per-tau table this app used to export under this name.

    mean_rates / TotalPhotons: VistaVision's TotalPhotons is the actual summed raw
    photon count over the measurement window; this app only has the mean count rate
    (CPS) for the analyzed window, so TotalPhotons here is approximated as
    CPS * measurement_time_s rather than a true per-bin sum -- consistent with the
    CPS value on the same line, but not an independently-measured integer count.

    Does NOT reproduce VistaVision's segment-boundary duplicate Tau values (e.g. a
    repeated "2E-05, 2E-05" in a real export) -- that's an undocumented internal
    quirk of VistaVision's own correlator with no published rule behind it, and
    with no raw trace available that's known to produce it, reproducing it would
    mean guessing at a pattern rather than validating one.
    """
    import datetime

    any_result = next(iter(results.values()))
    n_channels = 2 if "acf_ch2" in results else 1
    created = created or datetime.datetime.now().isoformat(timespec="seconds")

    header = [
        "[HeaderV]",
        "Version, 3",
        f"RawDataFile, {raw_data_file}",
        f"Created,{created}",
        f"Sections,{any_result.segments}",
        f"PtsPerSection, {any_result.points_per_segment}",
        f"SampleFrequency, {sampling_rate_hz:.10g}",
        "TimeSeries Count, 1",
        "PositionSeries Count, 1",
        "Spectrum Count, 1",
        f"ChannelCount, {n_channels}",
        f"MeasurementTime(sec), {measurement_time_s:.10g}",
        "",
        "iT,1",
        "iP,1",
        "iS,1",
    ]
    if n_channels == 2:
        header.append("AutoChannelIDs, 0, 1")
        header.append("CPS, " + ", ".join(f"{r:.12g}" for r in mean_rates))
        header.append("TotalPhotons, " + ", ".join(f"{r * measurement_time_s:.0f}" for r in mean_rates))
        header.append("CrossChannelIDs,01")
    else:
        header.append("AutoChannelIDs, 0")
        header.append(f"CPS, {mean_rates[0]:.12g}")
        header.append(f"TotalPhotons, {mean_rates[0] * measurement_time_s:.0f}")
    header += ["", "[Data]", "", "iT=1, iP=1, iS=1", ""]

    def block(label, values):
        # NaN can appear in the error arrays (too few sub-blocks reached a given tau);
        # written as 0 here since this text format has no other way to express "unknown".
        return [label, ", ".join(f"{v:.12g}" if np.isfinite(v) else "0" for v in values), ""]

    lines = list(header)
    lines += block("Tau", results["acf_ch1"].tau)
    lines += block("Ch0 AutoCorrelation", results["acf_ch1"].g)
    err = errors.get("acf_ch1")
    lines += block("Ch0 AutoCorrelation Standard Deviation", err if err is not None else np.zeros_like(results["acf_ch1"].g))
    if n_channels == 2:
        lines += block("Ch1 AutoCorrelation", results["acf_ch2"].g)
        err = errors.get("acf_ch2")
        lines += block("Ch1 AutoCorrelation Standard Deviation", err if err is not None else np.zeros_like(results["acf_ch2"].g))
        lines += block("Ch0x1 CrossCorrelation", results["cross"].g)
        err = errors.get("cross")
        lines += block("Ch0x1 CrossCorrelation Standard Deviation", err if err is not None else np.zeros_like(results["cross"].g))

    return "\n".join(lines) + "\n"


def fit_result_to_df(fit_result, label=""):
    rows = []
    for name, value in fit_result.params.items():
        rows.append(
            {
                "label": label,
                "parameter": name,
                "value": value,
                "stderr": fit_result.params_stderr.get(name, float("nan")),
            }
        )
    summary = {
        "quality": fit_result.quality,
        "quality_reasons": "; ".join(fit_result.quality_reasons),
        "fit_tau_min_s": fit_result.fit_tau_min,
        "fit_tau_max_s": fit_result.fit_tau_max,
        "n_points_fit": fit_result.n_points_fit,
        "redchi": fit_result.redchi,
    }
    for name, value in summary.items():
        rows.append({"label": label, "parameter": name, "value": value, "stderr": float("nan")})
    df = pd.DataFrame(rows)
    df.attrs["n_components"] = fit_result.n_components
    df.attrs["redchi"] = fit_result.redchi
    df.attrs["success"] = fit_result.success
    return df


def stability_report_to_df(report, label=""):
    rows = []
    for i, r in enumerate(report.per_start_results):
        row = {"label": label, "start_index": i, "success": r.success, "redchi": r.redchi, "N": r.N}
        for j, td in enumerate(r.tauD):
            row[f"tauD{j + 1}"] = td
        rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["is_stable"] = report.is_stable
    df.attrs["converged_fraction"] = report.converged_fraction
    df.attrs["relative_spreads"] = report.relative_spreads
    return df


def fccs_result_to_df(bound_fraction_ch2_species, bound_fraction_ch1_species, label=""):
    return pd.DataFrame(
        [
            {"label": label, "quantity": "bound_fraction_ch2_species", "value": bound_fraction_ch2_species},
            {"label": label, "quantity": "bound_fraction_ch1_species", "value": bound_fraction_ch1_species},
        ]
    )


def kd_result_to_df(concentrations, bound_fractions, kd_fit_result):
    df = pd.DataFrame(
        {
            "concentration": concentrations,
            "bound_fraction": bound_fractions,
            "isotherm_fit": kd_fit_result.fit_curve,
        }
    )
    df.attrs["Kd"] = kd_fit_result.Kd
    df.attrs["Kd_stderr"] = kd_fit_result.Kd_stderr
    df.attrs["T_total"] = kd_fit_result.T_total
    return df


def batch_comparison_to_df(rows):
    """rows: list of dicts, one per processed file, e.g.
    {"label": "sample_1", "tauD_ch1": ..., "tauD_ch2": ..., "bound_fraction_ch2_species": ...,
     "stable_ch1": True, "stable_ch2": True}. Simple passthrough to a DataFrame,
    kept as a function so the app's row-building convention lives in one place."""
    return pd.DataFrame(rows)


def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")
