"""
Regression checks on real VistaVision recordings (Fazlur's FL/BSA/NT tau data).

The data are not in git (fcs_data/ is gitignored, ~1.9 GB), so these tests skip
when the files aren't present. They pin the two outcomes that matter on this
data set: the one consistently resolvable diffusion time (FL+NT 48 h, CH2) and
a recording where the fit must report "not determined" rather than a number.
"""

from pathlib import Path

import numpy as np
import pytest

from core.autowindow import analyze_windows
from core.correlate import plateau_level, smooth_error
from core.fitting import QUALITY_GOOD, QUALITY_ROUGH, QUALITY_UNRESOLVED, multi_start_fit
from core.io import load_trace_auto

DATA = Path(__file__).resolve().parent.parent / "fcs_data" / "Raw data csv file for Lee"
FL_NT_48H = DATA / "FL+N 48hr 100000hz 1_AllInOne_AllInOne_AllInOne.csv"
FL_BSA = DATA / "FL+BSA 100000HZ_AllInOne_AllInOne.csv"
FL_NT_SERIES = [
    DATA / "FL+NT 0hr 100000hz 1_AllInOne_AllInOne_AllInOne.csv",
    DATA / "FL+NT 3hr 100000hz_AllInOne_AllInOne.csv",
    FL_NT_48H,
]


def _fit_average(path, kind):
    """Same path as the app's defaults: 3 s windows averaged, smoothed-sigma
    weights, fit from 20 us with a baseline offset."""
    tr = load_trace_auto(str(path))
    avg = analyze_windows(tr.time, tr.channels, window_length_s=3.0).average
    sigma = smooth_error(avg.corr_errors[kind])
    weights = np.where(np.isfinite(sigma) & (sigma > 0), 1 / sigma, 0)
    cr = avg.corr_results[kind]
    return multi_start_fit(cr.tau, cr.g, weights=weights, tau_min=2e-5, offset=True).chosen_result


@pytest.mark.skipif(not FL_NT_48H.exists(), reason="real data not present")
def test_fl_nt_48h_ch2_gives_sub_ms_tauD():
    fr = _fit_average(FL_NT_48H, "acf_ch2")
    assert 2e-4 < fr.tauD[0] < 1.5e-3, fr.tauD
    assert fr.quality in (QUALITY_GOOD, QUALITY_ROUGH), fr.quality_reasons
    assert fr.tauD_rel_err[0] < 1.0


@pytest.mark.skipif(not FL_BSA.exists(), reason="real data not present")
def test_fl_bsa_ch1_is_reported_not_determined():
    """Previously reported tau_D ~ 3.5e4 s with no warning."""
    fr = _fit_average(FL_BSA, "acf_ch1")
    assert fr.quality == QUALITY_UNRESOLVED
    assert fr.fit_tau_min <= fr.tauD[0] <= fr.fit_tau_max


@pytest.mark.skipif(not all(p.exists() for p in FL_NT_SERIES), reason="real data not present")
def test_fl_nt_cross_plateau_rises_with_incubation_time():
    """The main positive finding: the cross-correlation plateau (slow particles
    carrying both labels) grows 0 h -> 3 h -> 48 h. Measured ~0.0028, 0.0082, 0.0106."""
    levels = []
    for path in FL_NT_SERIES:
        tr = load_trace_auto(str(path))
        full = analyze_windows(tr.time, tr.channels, window_length_s=3.0).full_trace
        _, g, sigma = plateau_level(full.corr_results["cross"], full.corr_errors["cross"])
        levels.append(g)
        assert sigma < 0.002
    assert levels[0] < levels[1] < levels[2], levels
    assert levels[2] > 2 * levels[0]
