import streamlit as st

from app_common import sidebar_about

st.set_page_config(page_title="Methodology", layout="wide")
sidebar_about()

st.title("Methodology & Formulas")
st.caption(
    "What each stage of the pipeline actually computes, in order. Every number the app "
    "reports traces back to one of the formulas below -- nothing here is instrument-specific, "
    "it's the standard math used across FCS/FCCS analysis generally."
)

with st.container(border=True):
    st.subheader("The pipeline, in order")
    st.markdown(
        """
1. **Load** — parse the CSV's time column + 1 or 2 intensity columns.
2. **Split into windows, reject outliers, average** — no manual trimming needed.
3. **Bin** *(optional)* — group N raw points together before correlating.
4. **Correlate** — compute the autocorrelation of each channel, and (if 2 channels) the
   symmetrized cross-correlation, on a multi-tau lag grid.
5. **Fit** — fit a diffusion model to each correlation curve to get diffusion time(s) and
   amplitude.
6. **FCCS bound fraction** *(2-channel only)* — combine the three fitted amplitudes.
7. **Kd** *(not yet available in this deployment)* — fit a binding isotherm across a
   concentration series of bound fractions.
        """
    )

with st.container(border=True):
    st.subheader("1. Correlation function")
    st.markdown("The core quantity computed at every lag time τ:")
    st.latex(r"G(\tau) = \frac{\langle \delta I(t)\,\delta I(t+\tau) \rangle}{\langle I \rangle^2}")
    st.markdown(
        r"where $\delta I(t) = I(t) - \langle I \rangle$. For a two-channel trace, the "
        r"**cross-correlation** replaces one $I$ with the other channel's intensity, and is "
        r"symmetrized to remove lag-direction bias:"
    )
    st.latex(r"G_\times(\tau) = \tfrac{1}{2}\left[G_{1\to2}(\tau) + G_{2\to1}(\tau)\right]")

with st.container(border=True):
    st.subheader("2. Time windows: screening and averaging")
    st.markdown(
        "You pick a **window length** L (a dropdown, e.g. 3 s). The trace is split into "
        "non-overlapping windows of that length, and each is correlated separately:"
    )
    st.latex(r"[0, L),\ [L, 2L),\ [2L, 3L),\ \dots\ \text{up to the end of the trace}")
    st.markdown(
        "Each window gets a robust z-score for its mean count rate in each channel and for "
        "its correlation amplitude in each curve. The amplitude is the mean G over the first "
        "5 lags, roughly 1/N:"
    )
    st.latex(
        r"z = \frac{|x - \operatorname{median}(x)|}{1.4826 \cdot \operatorname{MAD}(x)}, "
        r"\qquad \text{reject if } z > 3.5 \text{ for any metric}"
    )
    st.markdown(
        "Count-rate outliers catch bleaching, drift, and focus loss. Amplitude outliers catch "
        "bright aggregates passing through the volume. Median and MAD are used instead of "
        "mean and standard deviation so the outliers themselves can't hide by inflating the "
        "spread. Screening needs at least 3 windows. The accepted windows' curves are then "
        "averaged point by point:"
    )
    st.latex(r"\bar G(\tau) = \frac{1}{M}\sum_{m=1}^{M} G_m(\tau)")
    st.markdown(
        "This averaged curve is the default. The app deliberately does **not** pick the "
        "single \"cleanest\" window. Ranking windows by signal-to-noise favors those with a "
        "larger G, i.e. fewer molecules or an aggregate, and taking the best of many biases "
        "every result toward whichever stretch happened to look best. Overlapping windows in "
        "1 s steps are also computed, and the full trace as one curve, so you can inspect or "
        "pick any of these by hand. If fewer than 2 windows survive, the full trace is used."
    )

with st.container(border=True):
    st.subheader("3. Multi-tau lag grid")
    st.markdown(
        "Rather than evaluating every lag linearly (too slow for million-point traces), "
        "lags are grouped into **segments**, each coarser than the last by a fixed **base**:"
    )
    st.latex(r"\Delta t_k = \Delta t_0 \cdot \text{base}^{\,k}, \qquad k = 0, 1, \dots, \text{segments}-1")
    st.latex(r"\tau = \Delta t_k \cdot i, \qquad i = 1, \dots, \text{points\_per\_segment}")
    st.markdown(
        "This gives even log-spaced τ coverage across many decades without the O(n²) cost "
        "of a brute-force correlation. Each τ point also tracks how many terms were actually "
        "averaged into it (`n_samples`) — points near the tail of the trace are averaged over "
        "fewer samples and are flagged as statistically less reliable."
    )

with st.container(border=True):
    st.subheader("4. Per-point error")
    st.markdown(
        "For the **averaged curve**, the uncertainty at each τ is the standard error across "
        "the M accepted windows. They don't overlap, so they're independent:"
    )
    st.latex(r"\sigma_{G(\tau)} = \frac{\text{std}\left(\{G_m(\tau)\}_{m=1}^{M}\right)}{\sqrt{M}}")
    st.markdown(
        "For a **single window or the full trace**, the same formula is applied across N "
        "contiguous sub-blocks of it (the \"Sub-blocks\" setting), each correlated with "
        "identical settings."
    )
    st.markdown(
        "**Fit weights.** With only 5–20 blocks, each σ is itself uncertain by tens of "
        "percent. Weighting by 1/σ directly would let a few points with luckily small σ "
        "dominate the fit. So before fitting, σ is smoothed with a running median over 5 "
        "neighbouring τ points. The true uncertainty varies smoothly with τ, so this keeps its "
        "shape. Exported error bars are left unsmoothed."
    )
    st.caption(
        "A standard block-averaging convention used broadly across FCS correlator software "
        "— not a reproduction of any one instrument's internal error algorithm."
    )

with st.container(border=True):
    st.subheader("5. Diffusion model (the fit)")
    st.markdown("Standard confocal 3D-diffusion model, for a single diffusing species:")
    st.latex(
        r"G(\tau) = \frac{1}{N} \cdot \frac{1}{1+\tau/\tau_D} \cdot "
        r"\frac{1}{\sqrt{1+\tau/(\kappa^2 \tau_D)}}"
    )
    st.markdown(
        r"$N$ is the average number of particles in the detection volume (amplitude), "
        r"$\tau_D$ is the diffusion time, and $\kappa = w_z/w_{xy}$ is the fixed structure "
        r"parameter (axial:radial ratio of the detection volume). For **two** independently "
        r"diffusing species (component count = 2):"
    )
    st.latex(
        r"G(\tau) = \frac{1}{N}\left[f_1 \cdot D(\tau,\tau_{D1}) + (1-f_1)\cdot D(\tau,\tau_{D2})\right]"
    )
    st.markdown("An optional fast blinking/triplet-state prefactor can be multiplied in:")
    st.latex(r"G(\tau) \;\leftarrow\; G(\tau)\cdot \frac{1-T+T\,e^{-\tau/\tau_{trip}}}{1-T}")
    st.markdown("A constant **baseline offset** is added on top (on by default):")
    st.latex(r"G(\tau) \;\leftarrow\; G_\infty + G(\tau)")
    st.markdown(
        r"Fluctuations much slower than diffusion (aggregates, drift) look like a flat plateau "
        r"across the lag range. Without $G_\infty$ the fit can only explain that plateau by "
        r"stretching $\tau_D$, often to seconds. $G_\infty$ is left out of $1/N$ and out of the "
        r"FCCS amplitudes."
    )
    st.markdown(
        r"**Fit range.** Only lags from a cutoff (default 20 µs) up to the last lag are fitted. "
        r"The earliest autocorrelation lags carry detector artifacts (afterpulsing, dead time). "
        r"$\tau_D$ is bounded to the fitted lag range."
    )
    st.markdown(
        r"**Quality verdict.** Each fit reports $\tau_D \pm$ its standard error and a verdict:"
    )
    st.markdown(
        """
- **well determined** — relative error ≤ 25%, and different starting guesses agree.
- **rough** — relative error 25–100%, or starting guesses disagree.
- **not determined** — any of: relative error > 100% or not estimable; τ_D within a
  factor 2 of either end of the fitted lag range (the decay isn't sampled on both sides);
  amplitude 1/N not distinguishable from zero.
        """
    )

with st.container(border=True):
    st.subheader("6. FCCS bound fraction")
    st.markdown(
        "Combines the fitted zero-lag amplitudes of both autocorrelations and the "
        "cross-correlation (uncorrected amplitude-ratio form — no spectral crosstalk "
        "correction applied):"
    )
    st.latex(r"f_{\text{bound, CH2 species}} = \frac{G_\times(0)}{G_{\text{CH1}}(0)}")
    st.latex(r"f_{\text{bound, CH1 species}} = \frac{G_\times(0)}{G_{\text{CH2}}(0)}")

with st.container(border=True):
    st.subheader("7. Kd — binding isotherm")
    st.markdown(
        "Fits bound fraction vs. ligand concentration $L$ to the full quadratic 1:1 "
        "binding equation (accounts for ligand depletion, valid even when $L$ and the "
        "fixed tracer concentration $T$ are comparable in magnitude — not just the "
        "simple hyperbolic approximation valid only when $L \\gg T$):"
    )
    st.latex(
        r"f_{\text{bound}}(L) = F_{min} + (F_{max}-F_{min}) \cdot "
        r"\frac{(K_d+T+L) - \sqrt{(K_d+T+L)^2 - 4TL}}{2T}"
    )

st.caption(
    "Correctness of the correlation step (sections 1 and 3) is checked against a brute-force "
    "FFT correlation and a synthetic trace with a known diffusion time on the Validation page."
)
