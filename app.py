import streamlit as st

from app_common import init_defaults, sidebar_about, kappa_from_w

st.set_page_config(page_title="FCS/FCCS Analysis", layout="wide")


def home():
    init_defaults()
    sidebar_about()

    st.title("FCS/FCCS Data Analysis")
    st.caption(
        "A general-purpose analysis tool for Fluorescence Correlation/Cross-Correlation "
        "Spectroscopy (FCS/FCCS) traces exported from an ISS VistaVision instrument, without "
        "needing VistaVision itself."
    )

    st.subheader("Available now")

    page_cards = [
        (
            "Methodology",
            "The exact formula behind every step of the pipeline below.",
        ),
        (
            "Single File Analysis",
            "Load one trace, average its non-outlier time windows, run the multi-tau correlation engine, and "
            "fit diffusion models to get diffusion time and (for dual-channel data) the FCCS "
            "bound fraction.",
        ),
    ]

    cols = st.columns(2)
    for col, (title, desc) in zip(cols, page_cards):
        with col.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(desc)

    st.caption("Batch / Time Course, Kd Fitting, and Validation are temporarily unavailable.")

    with st.container(border=True):
        st.subheader("How a single file moves through the pipeline")
        st.markdown(
            """
1. **Load** — your CSV's time column + 1 or 2 intensity columns are read in.
2. **Split into windows and screen them** — the trace is split into windows of a length
   you pick; windows with an outlying count rate or amplitude (bleaching, drift, focus
   loss, aggregates) are rejected and the rest are averaged. Kept and rejected windows are
   shaded on the raw trace plot.
3. **Bin points** *(optional)* — group N consecutive raw samples together before
   correlating, to reduce shot noise. Leave at 1 to skip this.
4. **Run the multi-tau correlation** — computes G(τ) for each channel (and the
   cross-correlation, if dual-channel) on a log-spaced lag grid.
5. **Fit a diffusion model** — choose 1 or 2 diffusing species (and an optional
   triplet/blinking term) per channel, independently; get back τ_D and amplitude, with a
   multi-start stability check that flags unstable/degenerate fits rather than silently
   accepting them.
6. **FCCS bound fraction** *(dual-channel only)* — combines the three fitted amplitudes
   into the fraction of each species that's bound to the other.

Every stage is CSV-exportable. See **Methodology** in the sidebar for the exact formula
behind each step.
            """
        )

    st.subheader("Global defaults")
    st.caption(
        "Pre-fill settings on every page; each page lets you override them per file/run. "
        "Defaults match VistaVision manual section 13.4.1 (5 segments x 15 points/segment, base 4)."
    )

    with st.container(border=True):
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        d = st.session_state["defaults"]
        d["segments"] = col1.number_input(
            "Segments", min_value=1, max_value=20, value=d["segments"],
            help=r"How many multi-tau blocks are computed (total $\tau$ points "
            r"$= \text{segments} \times \text{points\_per\_segment}$). More segments extend how far "
            "in tau the curve reaches; too many for a short trace adds low-reliability tail points instead.",
        )
        d["points_per_segment"] = col2.number_input(
            "Points/segment", min_value=1, max_value=100, value=d["points_per_segment"],
            help="Correlation points computed at each segment's lag spacing before coarsening "
            "to the next segment. More = finer resolution per decade of tau. 15 is VistaVision's "
            "own documented default.",
        )
        d["base"] = col3.number_input(
            "Grouping base", min_value=2, max_value=16, value=d["base"],
            help=r"Coarsening factor between segments: segment $k$'s lag step "
            r"$\Delta t_k = \Delta t_{raw} \cdot \text{base}^k$. base=4 is the standard multi-tau "
            "choice -- roughly quadruples tau spacing per segment for even log coverage. Larger = "
            "fewer segments needed but coarser within-decade resolution.",
        )
        d["w_xy_um"] = col4.number_input(
            "w_xy (µm, radial)", min_value=0.01, value=d["w_xy_um"], step=0.01, format="%.3f",
            help=r"The confocal detection volume's radial (lateral) beam waist, in microns -- from "
            "your instrument's own calibration. Only ever used to derive kappa below; the fit "
            "model has no separate dependence on its absolute value.",
        )
        d["w_z_um"] = col5.number_input(
            "w_z (µm, axial)", min_value=0.01, value=d["w_z_um"], step=0.01, format="%.3f",
            help="The confocal detection volume's axial beam waist, in microns -- from your "
            "instrument's own calibration. Only ever used to derive kappa below.",
        )
        kappa = kappa_from_w(d["w_xy_um"], d["w_z_um"])
        st.caption(
            f"kappa = w_z / w_xy = **{kappa:.3g}** -- this is the only quantity that actually enters "
            "the fit model's diffusion term; typical confocal setups land around 3-6. w_xy/w_z "
            "themselves have no calibration workflow yet for converting tau_D to an absolute "
            "diffusion coefficient or concentration (see README)."
        )
        d["min_reliable_n_samples"] = col6.number_input(
            "Min samples for a reliable tau point", min_value=1, value=d["min_reliable_n_samples"],
            help="Below this many averaged terms, a tau point is marked unreliable ('x' marker) "
            "on the correlation plot. Display/QA only -- it never changes G(tau) itself, just "
            "flags noisy long-tau points near the edge of your trace length.",
        )


pg = st.navigation(
    [
        st.Page(home, title="Home", default=True),
        st.Page("pages/0_Methodology.py", title="Methodology"),
        st.Page("pages/1_Single_File_Analysis.py", title="Single File Analysis"),
    ]
)
pg.run()
