import streamlit as st

from utils.prosesing import apply_theme, render_header

apply_theme()

pages = [
    st.Page("pages/0_Dashboard.py", title="Dashboard", default=True),
    st.Page("pages/1_overview.py", title="Overview"),
    st.Page("pages/2_EDA.py", title="EDA"),
    st.Page("pages/3_Prosesing.py", title="Prosesing"),
    st.Page("pages/4_Modeling.py", title="Modeling"),
    st.Page("pages/5_evalution.py", title="Evalution"),
]

pg = st.navigation(pages, position="hidden")  # menu bawaan disembunyikan
render_header(pages, pg)                      # nama web + menu horizontal
pg.run()