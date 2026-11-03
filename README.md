# FCS/FCCS Analysis App

**Live app:** https://makowski-lab-fcs-analysis.streamlit.app/

A local, cross-platform (macOS Apple Silicon-native) tool for analyzing Fluorescence
Correlation/Cross-Correlation Spectroscopy (FCS/FCCS) traces exported from an ISS
VistaVision instrument, without needing VistaVision itself. General-purpose: single- or
dual-channel autocorrelation/cross-correlation, diffusion-model fitting, FCCS bound
fraction, Kd-from-concentration-series fitting, and batch/time-course processing -- not
tied to any one binding pair or biological system.

The live app above is deployed on Streamlit Community Cloud, tracking the `main` branch
of this repo -- pushes to `main` redeploy it automatically within a couple of minutes.

## Quick start (no programming required)

1. Double-click **`Launch_FCS_App.command`**.
   - First run: it sets up a local Python environment and installs dependencies
     (takes a minute or two; needs an internet connection once).
   - It then opens the app in your default web browser.
2. If macOS blocks the file the first time ("cannot be opened because it is from an
   unidentified developer"): right-click it → **Open** → **Open** again. This is a
   one-time step per machine.
3. Everything after that runs locally in the browser tab that opens — no data leaves
   your computer.

Requires Python 3.10+ already installed on macOS (check with `python3 --version` in
Terminal; install from [python.org](https://www.python.org/downloads/macos/) if needed).

## What each page does

- **Home** — global defaults (multi-tau segments/points-per-segment/base, beam waists
  w_xy and w_z, minimum sample count for a "reliable" tau point).
- **Methodology** — the exact formula behind every step of the pipeline.
- **Single File Analysis** — load one trace and pick a window length (default 3 s). The
  app:
  1. splits the trace into non-overlapping windows of that length and correlates each;
  2. rejects outlier windows, i.e. those whose count rate or correlation amplitude is
     more than 3.5 robust SDs (median/MAD) from the rest. This catches bleaching, drift,
     focus loss, and aggregates;
  3. averages the accepted curves into the default curve, with error bars from the spread
     across windows.

  Kept and rejected windows are shaded on the raw trace, and a table shows why each window
  was rejected. You can instead use the full trace as one curve, or any single 1 s-step
  sliding window. After that you can set "Binning points" (VistaVision's term for grouping
  N raw points before correlating), fit diffusion models to the CH1/CH2 autocorrelations
  and the cross-correlation independently, and compute the FCCS bound fraction. Every
  stage is CSV-exportable.

The following pages exist in `disabled_pages/` but are hidden from the deployed app for now:

- **Batch / Time Course** — process a set of files (e.g. across conditions, replicates,
  or timepoints) with one shared settings panel, producing a comparison table and plots.
- **Kd Fitting** — fit a 1:1 binding isotherm (full quadratic/ligand-depletion form) to
  bound-fraction-vs-ligand-concentration data from a concentration series to extract Kd.
  Concentrations are typed directly into an editable table per file.
- **Validation** — (A) re-run the exact-FFT-vs-multi-tau cross-check that the reference
  engine was originally validated against; (B) generate synthetic data with a known
  diffusion time and confirm the fitting pipeline recovers it, including an optional
  Monte Carlo repeat.

## Why windows are averaged, not cherry-picked

Earlier versions slid a window across the trace and kept the one with the highest
correlation signal-to-noise. That rule was biased. SNR rises with G(0) ≈ 1/N, so it
favored stretches with fewer molecules or a passing aggregate, which are exactly the
artifacts you want to exclude. Taking the best of dozens of windows also biased τ_D, N,
and the FCCS bound fraction toward whichever stretch happened to look cleanest. Averaging
every acceptable window uses all the good data. Because the windows don't overlap, their
spread is an independent error estimate (typically 10–20 windows, versus 5 sub-blocks
before).

Fit weights use σ smoothed with a 5-point running median across τ. Each raw σ is itself
noisy, and weighting by it directly lets a few lucky points dominate the fit. Exported
error bars stay unsmoothed.

## Fitting

Defaults chosen after running Fazlur's real recordings (FL tau, BSA, NT tau) through
the app. Each one fixes a failure seen on that data:

- **Baseline offset G∞** (on by default). The real curves sit on a large plateau from
  fluctuations much slower than diffusion; in FL+BSA CH2 it's about 10x the diffusion
  amplitude. Without an offset, the fit stretched τ_D to seconds. For FL+NT 48 h CH2
  (full trace) τ_D went from 1,450 ms to 0.6 ms.
- **Fit lags from 20 µs.** The earliest autocorrelation lags carry detector artifacts
  (afterpulsing). These made 500 kHz fits collapse to the τ_D lower bound.
- **τ_D bounded to the fitted lag range.** Previously, values from 10⁻⁹ s to months
  were reported.
- **τ_D ± error and a quality verdict** (well determined / rough / not determined)
  replace the "STABLE" badge. That badge only checked whether different starting guesses
  agreed, and fits to pure noise often passed. The FCCS bound fraction is marked
  "not meaningful" if any of its three fits is not determined, and warns outside 0–1.
- **Slow-sampling warning.** Files sampled slower than 50 kHz (first lag > 20 µs) can't
  resolve sub-millisecond diffusion times. This applies to the 10 kHz recordings.

- **Fix τ_D1 in 2-component fits.** This holds the fast (monomer) component at a value
  from a control and fits only the slow component and the split. On noisy curves a free
  2-component fit can trade the two components off almost arbitrarily. It's only as good
  as the control's own τ_D.
- **Slow-component plateau** (step 2): G at about 10 ms from the full trace, ± error.
  By 10 ms sub-millisecond diffusion has decayed, so this measures species slower than
  the lag range (aggregates) or slow drift, and it is much better measured than τ_D.
  In FL+NT the cross-correlation plateau rises 0.0028 → 0.0082 → 0.0106 over
  0 h → 3 h → 48 h (100 kHz recordings). But the FL+BSA control has the largest plateau
  (cross-correlation 0.013), so crosstalk or drift has to be ruled out before reading
  this as binding. It's always taken from the full trace, because short windows remove
  the slow component. It scales with 1/concentration.

On this data set the only consistently resolvable diffusion time is FL+NT 48 h CH2,
about 0.4–0.9 ms ± 40–60%, across the 100, 200 and 500 kHz recordings. The other
autocorrelations come out "not determined". No single 3 s window is well determined.

## Known limitations (by design, for this version)

- **No spectral crosstalk/bleed-through correction** in the FCCS bound-fraction
  calculation — it uses the standard uncorrected amplitude-ratio formula
  (`G_cross(0) / G_partner(0)`). Add crosstalk correction later using single-labeled
  (CH1-only / CH2-only) reference files if bleed-through turns out to matter.
- **No absolute diffusion coefficient or concentration** — you enter the beam waists
  `w_xy` and `w_z`, but they are only used to compute the structure parameter
  κ = w_z / w_xy, which is fixed in the fit. They are not used to convert τ_D into D or
  N into a concentration. Results are reported as τ_D and fitted amplitude/N directly.
  `FitResult.calibration` is an explicit, currently-unset field so this conversion can be
  added later without changing the output schema. Calibrating `w_xy` against a dye of
  known D is a prerequisite.
- **Outlier thresholds are fixed** — z > 3.5 on count rate and amplitude, with at least
  3 windows needed to screen. A trace that drifts steadily throughout won't be caught,
  because no single window stands out from the others. Check the raw trace plot.
- **Parser validated against the two documented formats and synthetic fixtures only** —
  no real VistaVision export files were available while building this. Run a real file
  through **Single File Analysis** and check for the "fast parser bypassed" warning; if
  it appears, the (slower but still correct) fallback parser handled it, but it's worth
  flagging so the fast path can be adjusted for whatever format quirk triggered it.

## Project layout

```
core/            correlation engine, fitting, FCCS, Kd, validation (no UI code)
  engine.py      the original validated multi-tau engine, extended only additively
                 (adds per-tau n_samples; do not modify the correlation math itself)
  autowindow.py  window tiling, outlier screening, and averaging
pages/           Streamlit pages (the GUI)
disabled_pages/  Batch, Kd, Validation pages, hidden from the deployed app
tests/           pytest suite: unit tests for core/, + headless UI tests (AppTest)
                 for pages/ using Streamlit's real script-execution engine
```

## Running the test suite

```bash
cd fcs_app
./.venv/bin/pytest -q
```

Covers: parser correctness/fallback, the `n_samples` addition and a regression pin
against a fixture (proving the additive change didn't alter the validated math),
FFT-vs-multi-tau exact-match, window outlier rejection (aggregate bursts, bleaching
steps) and averaging, fitting recovery on noiseless and synthetic data,
multi-start instability detection, baseline offset / lag cutoff / τ_D bounds and the
quality verdict, FCCS/Kd formulas, and headless UI runs of every page.
`tests/test_real_data.py` pins results on two of the real recordings. It skips unless
`fcs_data/` (gitignored) is present.
