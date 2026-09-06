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

- **Home** — global defaults (multi-tau segments/points-per-segment/base, structure
  parameter kappa, minimum sample count for a "reliable" tau point).
- **Single File Analysis** — load one trace file, trim the time window (draggable
  slider, shaded region on the raw trace plot), set "Binning points" (VistaVision's own
  term for grouping N raw points before correlating), run the multi-tau correlation
  engine, fit diffusion models to CH1/CH2 autocorrelation and the cross-correlation
  independently, and compute the FCCS bound fraction. Every stage is CSV-exportable.
- **Batch / Time Course** — process a set of files (e.g. across conditions, replicates,
  or timepoints) with one shared settings panel, producing a comparison table and plots.
- **Kd Fitting** — fit a 1:1 binding isotherm (full quadratic/ligand-depletion form) to
  bound-fraction-vs-ligand-concentration data from a concentration series to extract Kd.
  Concentrations are typed directly into an editable table per file.
- **Validation** — (A) re-run the exact-FFT-vs-multi-tau cross-check that the reference
  engine was originally validated against; (B) generate synthetic data with a known
  diffusion time and confirm the fitting pipeline recovers it, including an optional
  Monte Carlo repeat.

## Known limitations (by design, for this version)

- **No spectral crosstalk/bleed-through correction** in the FCCS bound-fraction
  calculation — it uses the standard uncorrected amplitude-ratio formula
  (`G_cross(0) / G_partner(0)`). Add crosstalk correction later using single-labeled
  (CH1-only / CH2-only) reference files if bleed-through turns out to matter.
- **No absolute diffusion coefficient or concentration** — no observation-volume
  calibration (beam waist `w_xy`) exists yet, so results are reported as diffusion time
  (`tau_D`) and fitted amplitude/N directly. `FitResult.calibration` is an explicit,
  currently-unset field so this conversion can be added later without changing the
  output schema.
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
pages/           Streamlit pages (the GUI)
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
FFT-vs-multi-tau exact-match, fitting recovery on noiseless and synthetic data,
multi-start instability detection, FCCS/Kd formulas, and headless UI runs of every page.
