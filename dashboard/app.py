"""Dashboard of the fitted SPX and synthetic panels: surfaces, static-arbitrage maps and the
ranked mispricing flags of any day.

    make dashboard   # uv run streamlit run dashboard/app.py

It reads the per-quote tables and fits written by scripts/fit_panel.py under results/ and the
conditional-model checkpoints under artifacts/; both are produced by the Makefile targets listed
in the README.
"""

import streamlit as st
from common import sidebar

st.set_page_config(page_title="Volatility surfaces", layout="wide")
pages = st.navigation(
    [
        st.Page("views/surface.py", title="Surface", url_path="surface", default=True),
        st.Page("views/arbitrage.py", title="Arbitrage", url_path="arbitrage"),
        st.Page("views/mispricings.py", title="Mispricings", url_path="mispricings"),
    ]
)
sidebar()
pages.run()
