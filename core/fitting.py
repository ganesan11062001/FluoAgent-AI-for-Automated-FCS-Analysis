"""
core/fitting.py — fit standard 3D-diffusion FCS models to G(tau) curves, with a
multi-start stability check that FLAGS (rather than silently accepts) divergent fits,
and a quality verdict on whether tau_D is actually determined by the data.

Fit choices that matter on real data (see README "Fitting"):
- Lag range [tau_min, tau_max]: the earliest autocorrelation lags carry detector
  artifacts (afterpulsing, dead time), so callers should cut them off (the app
  defaults to 20 us). Cross-correlations between two detectors don't have this.
- Baseline offset G_inf (on by default): absorbs fluctuations much slower than
  diffusion, which otherwise get folded into an inflated tau_D.
- tau_D is bounded to the fitted lag range. A tau_D outside the measured lags is
  not measured by the data, and one that ends up AT a bound is flagged.

Scope note: no observation-volume calibration (w_xy) is applied, so results are
reported as tau_D (seconds) and fitted amplitude/N directly -- not converted to an
absolute diffusion coefficient or concentration. FitResult.calibration is an
explicit, currently-unset slot for that conversion.
"""

from dataclasses import dataclass, field

import numpy as np
import lmfit

from .models import fcs_1comp, fcs_2comp

STABILITY_REL_SPREAD_THRESHOLD = 0.20
STABILITY_MIN_CONVERGED_FRACTION = 0.6

# Quality verdict thresholds on tau_D's relative standard error.
TAUD_WELL_DETERMINED_REL_ERR = 0.25
TAUD_DETERMINED_REL_ERR = 1.0
# tau_D must sit this far inside the fitted lag range for the decay to actually
# be sampled on both sides (a factor 2 from either end).
TAUD_EDGE_FACTOR = 2.0

QUALITY_GOOD = "well determined"
QUALITY_ROUGH = "rough"
QUALITY_UNRESOLVED = "not determined"


@dataclass
class CalibrationConstants:
    """Explicit future slot for tau_D -> D and G(0) -> concentration conversion.
    Intentionally unset in v1 -- no w_xy/kappa calibration workflow exists yet."""

    wxy_um: float = None
    kappa_calibrated: float = None


@dataclass
class FitResult:
    n_components: int
    triplet: bool
    params: dict
    params_stderr: dict
    tauD: list
    N: float
    fit_curve: np.ndarray
    redchi: float
    success: bool
    message: str = ""
    calibration: CalibrationConstants = None
    stability: "StabilityReport" = None
    tauD_stderr: list = field(default_factory=list)
    G_inf: float = 0.0
    fit_tau_min: float = float("nan")
    fit_tau_max: float = float("nan")
    n_points_fit: int = 0
    quality: str = QUALITY_UNRESOLVED
    quality_reasons: list = field(default_factory=list)
    fixed_tauD1: float = None  # set when tauD1 was held fixed (e.g. monomer value from a control)

    @property
    def tauD_rel_err(self):
        return [
            (se / td if td and np.isfinite(se) else float("nan"))
            for td, se in zip(self.tauD, self.tauD_stderr)
        ]


@dataclass
class StabilityReport:
    is_stable: bool
    relative_spreads: dict
    converged_fraction: float
    per_start_results: list
    chosen_result: FitResult


def build_model(n_components=1, triplet=False, offset=True, tau_lo=1e-9, tau_hi=np.inf):
    """Returns (lmfit.Model, lmfit.Parameters) with bounds/constraints set, but no
    data-informed initial values yet -- fit_curve()/multi_start_fit() set those.
    tau_D is bounded to [tau_lo, tau_hi], normally the fitted lag range."""
    if n_components == 1:
        model = lmfit.Model(fcs_1comp, independent_vars=["tau"])
        params = model.make_params()
        params["N"].set(value=1.0, min=1e-6)
        params["tauD"].set(value=np.sqrt(tau_lo * min(tau_hi, 1.0)), min=tau_lo, max=tau_hi)
    elif n_components == 2:
        model = lmfit.Model(fcs_2comp, independent_vars=["tau"])
        params = model.make_params()
        params["N"].set(value=1.0, min=1e-6)
        params["f1"].set(value=0.5, min=0.0, max=1.0)
        params["tauD1"].set(value=tau_lo * 10, min=tau_lo, max=tau_hi)
        # tauD2 = tauD1 + delta (delta >= 0) enforces tauD2 >= tauD1, avoiding
        # component label-swap ambiguity between the two species.
        params.add("tauD2_delta", value=tau_lo * 100, min=0.0, max=tau_hi)
        params.add("tauD2", expr="tauD1 + tauD2_delta")
    else:
        raise ValueError("n_components must be 1 or 2.")

    params["kappa"].set(value=5.0, vary=False)
    params["G_inf"].set(value=0.0, vary=offset)
    if triplet:
        params["T"].set(value=0.1, min=0.0, max=0.9, vary=True)
        params["tau_trip"].set(value=1e-6, min=1e-9, vary=True)
    else:
        params["T"].set(value=0.0, vary=False)
        params["tau_trip"].set(value=1e-6, vary=False)
    return model, params


def fit_range_mask(tau, g, tau_min=None, tau_max=None):
    tau = np.asarray(tau, dtype=np.float64)
    mask = np.isfinite(tau) & np.isfinite(np.asarray(g, dtype=np.float64)) & (tau > 0)
    if tau_min is not None:
        mask &= tau >= tau_min
    if tau_max is not None:
        mask &= tau <= tau_max
    return mask


def _guess_amplitude_and_tauD(tau, g, offset=True):
    """Data-driven initial guess: baseline from the last 10% of lags (if fitting
    an offset), amplitude from the earliest 5% minus that baseline, tauD from
    where the curve has decayed to half that amplitude."""
    tau = np.asarray(tau, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    order = np.argsort(tau)
    tau_s, g_s = tau[order], g[order]
    n_head = max(1, len(g_s) // 20)
    n_tail = max(1, len(g_s) // 10)
    g_inf = float(np.mean(g_s[-n_tail:])) if offset else 0.0
    amp = float(np.mean(g_s[:n_head])) - g_inf
    if amp <= 1e-9:
        amp = 1e-9
    half = g_inf + amp / 2.0
    idx = int(np.argmin(np.abs(g_s - half)))
    tauD_guess = tau_s[idx] if tau_s[idx] > 0 else float(np.median(tau_s[tau_s > 0]))
    return 1.0 / amp, tauD_guess, g_inf


def format_rel_err(rel):
    """'37%', or 'over 1000%' once the number is too large to mean anything."""
    if not np.isfinite(rel):
        return "unknown"
    return "over 1000%" if rel > 10 else f"{rel:.0%}"


def assess_fit(fr):
    """Quality verdict on whether the data actually pin down tau_D (and the
    amplitude). Sets fr.quality and fr.quality_reasons in place."""
    reasons = []
    unresolved = False
    if not fr.success:
        fr.quality, fr.quality_reasons = QUALITY_UNRESOLVED, [f"fit failed: {fr.message}"]
        return fr
    lo, hi = fr.fit_tau_min, fr.fit_tau_max
    for i, (td, rel) in enumerate(zip(fr.tauD, fr.tauD_rel_err)):
        name = "tau_D" if len(fr.tauD) == 1 else f"tau_D{i + 1}"
        if i == 0 and fr.fixed_tauD1 is not None:
            continue  # held fixed: not estimated from this curve, so nothing to judge
        if td < TAUD_EDGE_FACTOR * lo:
            unresolved = True
            reasons.append(
                f"{name} is at or near the first fitted lag ({lo:.3g} s): the decay is faster "
                "than the data can sample"
            )
        elif td > hi / TAUD_EDGE_FACTOR:
            unresolved = True
            reasons.append(
                f"{name} is at or near the last fitted lag ({hi:.3g} s): no decay seen within the lag range"
            )
        if not np.isfinite(rel):
            unresolved = True
            reasons.append(f"{name} uncertainty could not be estimated (parameter not constrained by the data)")
        elif rel > TAUD_DETERMINED_REL_ERR:
            unresolved = True
            reasons.append(f"{name} relative error is {format_rel_err(rel)}")
        elif rel > TAUD_WELL_DETERMINED_REL_ERR:
            reasons.append(f"{name} relative error is {format_rel_err(rel)}")
    n_se = fr.params_stderr.get("N", float("nan"))
    if np.isfinite(n_se) and fr.N > 0 and n_se / fr.N > 1.0:
        unresolved = True
        reasons.append(f"amplitude 1/N is not distinguishable from zero (N = {fr.N:.3g} ± {n_se:.2g})")
    if unresolved:
        fr.quality = QUALITY_UNRESOLVED
    elif reasons:
        fr.quality = QUALITY_ROUGH
    else:
        fr.quality = QUALITY_GOOD
    fr.quality_reasons = reasons
    return fr


def fit_curve(
    tau, g, n_components=1, triplet=False, initial_guess=None, weights=None, kappa=5.0,
    tau_min=None, tau_max=None, offset=True, fixed_tauD1=None,
):
    """Fit one G(tau) curve over the lags in [tau_min, tau_max]. initial_guess is an
    optional dict of param_name->value overrides (used by multi_start_fit to probe
    different starting points). fit_curve in the result is evaluated on the full
    tau grid so it can be overlaid on the data; only the fitted range drives the fit.

    fixed_tauD1 (seconds, 2-component only) holds the faster component's tau_D at a
    known value, e.g. the monomer tau_D from a control sample. Fitting one fewer
    free parameter removes most of the trade-off between the two components, which
    on noisy curves otherwise lets the fit split the decay almost arbitrarily."""
    if fixed_tauD1 is not None and n_components != 2:
        raise ValueError("fixed_tauD1 only applies to 2-component fits.")
    tau = np.asarray(tau, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    mask = fit_range_mask(tau, g, tau_min, tau_max)
    if mask.sum() < 4:
        raise ValueError("Fewer than 4 lags inside the fit range.")
    tau_f, g_f = tau[mask], g[mask]
    w_f = None if weights is None else np.asarray(weights, dtype=np.float64)[mask]
    lo, hi = float(tau_f.min()), float(tau_f.max())

    model, params = build_model(n_components, triplet, offset=offset, tau_lo=lo, tau_hi=hi)
    params["kappa"].set(value=kappa)

    N_guess, tauD_guess, ginf_guess = _guess_amplitude_and_tauD(tau_f, g_f, offset=offset)
    tauD_guess = float(np.clip(tauD_guess, lo, hi))
    params["N"].set(value=N_guess)
    if offset:
        params["G_inf"].set(value=ginf_guess)
    if n_components == 1:
        params["tauD"].set(value=tauD_guess)
    else:
        params["tauD1"].set(value=max(lo, tauD_guess / 3.0))
        params["tauD2_delta"].set(value=min(hi, tauD_guess * 3.0))

    if initial_guess:
        for key, value in initial_guess.items():
            if key in params:
                p = params[key]
                params[key].set(value=float(np.clip(value, p.min, p.max)))

    if fixed_tauD1 is not None:
        # Applied after initial_guess so a multi-start probe can't move it. The fixed
        # value may lie outside the fitted lag range (it came from another curve).
        params["tauD1"].set(value=fixed_tauD1, vary=False, min=-np.inf, max=np.inf)

    base = dict(
        n_components=n_components, triplet=triplet, fit_tau_min=lo, fit_tau_max=hi,
        n_points_fit=int(mask.sum()), fixed_tauD1=fixed_tauD1,
    )
    try:
        result = model.fit(g_f, params, tau=tau_f, weights=w_f)
        success = bool(result.success) and np.isfinite(result.redchi)
        message = result.message or ""
    except Exception as exc:  # noqa: BLE001 - a failed fit is a valid, reportable outcome
        return assess_fit(FitResult(
            params={}, params_stderr={}, tauD=[], N=float("nan"),
            fit_curve=np.full_like(g, np.nan), redchi=float("inf"), success=False,
            message=f"fit raised: {exc}", **base,
        ))

    out_params = {name: p.value for name, p in result.params.items()}
    out_stderr = {name: (p.stderr if p.stderr is not None else float("nan")) for name, p in result.params.items()}
    td_names = ["tauD"] if n_components == 1 else ["tauD1", "tauD2"]

    fr = FitResult(
        params=out_params,
        params_stderr=out_stderr,
        tauD=[out_params[n] for n in td_names],
        tauD_stderr=[out_stderr[n] for n in td_names],
        N=out_params["N"],
        G_inf=out_params["G_inf"],
        fit_curve=result.eval(params=result.params, tau=tau),
        redchi=float(result.redchi),
        success=success,
        message=message,
        **base,
    )
    return assess_fit(fr)


def multi_start_fit(
    tau, g, n_components=1, triplet=False, n_starts=5, kappa=5.0, weights=None,
    tau_min=None, tau_max=None, offset=True, fixed_tauD1=None,
):
    """Run the same fit from n_starts different initial tauD guesses (log-spaced
    across the fitted lag range) and flag divergence rather than silently
    trusting whichever result the optimizer returns first.

    Unstable if: any parameter's (max-min)/median across CONVERGED starts exceeds
    STABILITY_REL_SPREAD_THRESHOLD, or fewer than STABILITY_MIN_CONVERGED_FRACTION
    of starts converge. An unstable fit can't be rated better than "rough".
    """
    tau = np.asarray(tau, dtype=np.float64)
    mask = fit_range_mask(tau, g, tau_min, tau_max)
    tau_pos = tau[mask]
    if len(tau_pos) == 0:
        raise ValueError("No positive tau values to fit.")

    lo = 3.0 * np.min(tau_pos)
    hi = np.max(tau_pos) / 3.0
    if hi <= lo:
        hi = lo * 10.0
    start_tauDs = np.exp(np.linspace(np.log(lo), np.log(hi), n_starts))

    per_start_results = []
    for s in start_tauDs:
        if n_components == 1:
            ig = {"tauD": s}
        elif fixed_tauD1 is not None:
            ig = {"tauD2_delta": max(s - fixed_tauD1, s / 10.0)}
        else:
            ig = {"tauD1": s / 3.0, "tauD2_delta": s * 3.0}
        fr = fit_curve(
            tau, g, n_components=n_components, triplet=triplet, initial_guess=ig, weights=weights,
            kappa=kappa, tau_min=tau_min, tau_max=tau_max, offset=offset, fixed_tauD1=fixed_tauD1,
        )
        per_start_results.append(fr)

    converged = [r for r in per_start_results if r.success]
    converged_fraction = len(converged) / len(per_start_results)

    param_names = ["N"] + (["tauD"] if n_components == 1 else (["tauD2"] if fixed_tauD1 is not None else ["tauD1", "tauD2"]))
    relative_spreads = {}
    for pname in param_names:
        vals = [r.params[pname] for r in converged if pname in r.params]
        if len(vals) >= 2:
            med = np.median(vals)
            relative_spreads[pname] = float((max(vals) - min(vals)) / med) if med != 0 else float("inf")
        else:
            relative_spreads[pname] = float("nan")

    is_stable = converged_fraction >= STABILITY_MIN_CONVERGED_FRACTION and all(
        spread <= STABILITY_REL_SPREAD_THRESHOLD for spread in relative_spreads.values() if not np.isnan(spread)
    )

    chosen = min(converged, key=lambda r: r.redchi) if converged else per_start_results[0]
    if not is_stable and chosen.quality == QUALITY_GOOD:
        chosen.quality = QUALITY_ROUGH
        chosen.quality_reasons = chosen.quality_reasons + ["different starting guesses give different results"]

    report = StabilityReport(
        is_stable=is_stable,
        relative_spreads=relative_spreads,
        converged_fraction=converged_fraction,
        per_start_results=per_start_results,
        chosen_result=chosen,
    )
    chosen.stability = report
    return report
