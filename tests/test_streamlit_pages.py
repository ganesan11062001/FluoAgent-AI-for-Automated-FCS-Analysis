"""
Headless UI regression tests using Streamlit's AppTest -- exercises each page's
actual widget/callback code (not just the underlying core/ functions, which are
covered elsewhere) to catch Streamlit-API misuse, session_state key mistakes, or
undefined-variable bugs that unit tests on core/ can't see.
"""

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest


def _make_dual_channel_csv_bytes(n=3000, dt=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    ch1 = rng.poisson(8, n)
    ch2 = rng.poisson(4, n)
    lines = [f"\t  {i * dt:.6f},         {a},         {b}\n" for i, (a, b) in enumerate(zip(ch1, ch2))]
    return "".join(lines).encode("utf-8")


def _make_single_channel_csv_bytes(n=3000, dt=1e-4, seed=1):
    rng = np.random.default_rng(seed)
    ch1 = rng.poisson(5, n)
    lines = [f"\t  {i * dt:.6f},         {c}\n" for i, c in enumerate(ch1)]
    return "".join(lines).encode("utf-8")


def test_home_page_renders():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    assert not at.exception


def test_single_file_analysis_dual_channel_end_to_end():
    at = AppTest.from_file("pages/1_Single_File_Analysis.py")
    at.run(timeout=60)
    assert not at.exception

    at.file_uploader[0].set_value(("dual.csv", _make_dual_channel_csv_bytes(), "text/csv"))
    at.run(timeout=60)
    assert not at.exception  # correlation now runs automatically on upload, no button

    # Fit CH1, CH2, and cross
    for key in ["sf_fit_btn_acf_ch1", "sf_fit_btn_acf_ch2", "sf_fit_btn_cross"]:
        at.button(key=key).click()
        at.run(timeout=60)
        assert not at.exception, f"exception after clicking {key}: {at.exception}"


def test_single_file_analysis_averages_windows_by_default():
    """A trace long enough for several 3 s windows: the default curve must be the
    average, and fitting it plus the FCCS step must run cleanly."""
    at = AppTest.from_file("pages/1_Single_File_Analysis.py")
    at.run(timeout=60)
    at.file_uploader[0].set_value(("long.csv", _make_dual_channel_csv_bytes(n=12000, dt=1e-3), "text/csv"))
    at.run(timeout=120)
    assert not at.exception

    assert at.selectbox(key="sf_window_choice").value.startswith("Average of")
    for key in ["sf_fit_btn_acf_ch1", "sf_fit_btn_acf_ch2", "sf_fit_btn_cross"]:
        at.button(key=key).click()
        at.run(timeout=120)
        assert not at.exception, f"exception after clicking {key}: {at.exception}"


def test_two_component_fit_with_fixed_tauD1_runs():
    at = AppTest.from_file("pages/1_Single_File_Analysis.py")
    at.run(timeout=60)
    at.file_uploader[0].set_value(("long.csv", _make_dual_channel_csv_bytes(n=12000, dt=1e-3), "text/csv"))
    at.run(timeout=120)
    at.radio(key="sf_acf_ch1_ncomp").set_value(2)
    at.run(timeout=60)
    at.checkbox(key="sf_acf_ch1_fix_tauD1").check()
    at.run(timeout=60)
    at.number_input(key="sf_acf_ch1_tau_min_us").set_value(0.0)  # 1 ms sampling: keep every lag
    at.button(key="sf_fit_btn_acf_ch1").click()
    at.run(timeout=120)
    assert not at.exception
    assert any("(fixed)" in str(m.value) for m in at.metric)


def test_single_file_analysis_single_channel_end_to_end():
    at = AppTest.from_file("pages/1_Single_File_Analysis.py")
    at.run(timeout=60)
    at.file_uploader[0].set_value(("single.csv", _make_single_channel_csv_bytes(), "text/csv"))
    at.run(timeout=60)
    assert not at.exception  # correlation now runs automatically on upload, no button

    at.button(key="sf_fit_btn_acf_ch1").click()
    at.run(timeout=60)
    assert not at.exception


def test_batch_time_course_end_to_end():
    # Page lives outside pages/ (see disabled_pages/) so it's not exposed in the
    # deployed app's nav yet, but its code is still covered here.
    at = AppTest.from_file("disabled_pages/2_Batch_Time_Course.py")
    at.run(timeout=60)
    at.file_uploader[0].set_value(
        [
            ("0hr.csv", _make_dual_channel_csv_bytes(seed=1), "text/csv"),
            ("3hr.csv", _make_dual_channel_csv_bytes(seed=2), "text/csv"),
        ]
    )
    at.run(timeout=60)
    assert not at.exception

    run_btn = [b for b in at.button if b.label == "Run batch"][0]
    run_btn.click()
    at.run(timeout=120)
    assert not at.exception


def test_kd_fitting_manual_entry_default_table():
    # Note: Streamlit's AppTest (this version) exposes st.data_editor as a
    # read-only Dataframe test element (no .set_value()/editing support), so
    # this only exercises the page with its default placeholder table -- the
    # isotherm math itself is covered independently in tests/test_fccs_kd.py.
    at = AppTest.from_file("disabled_pages/3_Kd_Fitting.py")
    at.run(timeout=30)
    assert not at.exception

    fit_btn = [b for b in at.button if b.label == "Fit Kd"][0]
    fit_btn.click()
    at.run(timeout=30)
    assert not at.exception


def test_kd_fitting_pull_from_batch_without_batch_run_shows_warning():
    at = AppTest.from_file("disabled_pages/3_Kd_Fitting.py")
    at.run(timeout=30)
    pull_radio = at.radio[0]
    pull_radio.set_value("Pull from last Batch run")
    at.run(timeout=30)
    assert not at.exception
    assert len(at.warning) >= 1


def test_validation_page_synthetic_recovery():
    at = AppTest.from_file("disabled_pages/4_Validation.py")
    at.run(timeout=30)
    assert not at.exception

    gen_btn = [b for b in at.button if b.label == "Generate & fit synthetic trace"][0]
    gen_btn.click()
    at.run(timeout=60)
    assert not at.exception


def test_validation_page_fft_crosscheck():
    at = AppTest.from_file("disabled_pages/4_Validation.py")
    at.run(timeout=30)

    gen_synth_btn = [b for b in at.button if b.label == "Generate synthetic trace for FFT check"][0]
    gen_synth_btn.click()
    at.run(timeout=30)
    assert not at.exception

    run_btn = [b for b in at.button if b.label == "Run FFT cross-check"][0]
    run_btn.click()
    at.run(timeout=30)
    assert not at.exception
