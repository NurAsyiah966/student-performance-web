import pandas as pd
import streamlit as st

from utils.prosesing import require_dataset, setup_page, split_columns

setup_page("Dataset Overview", "🗂️", subtitle="Kenali kondisi dataset sebelum dianalisis lebih lanjut")

df = require_dataset()
st.caption(f"Sumber data: **{st.session_state.get('filename', '-')}**")

total_missing = int(df.isnull().sum().sum())
total_dup = int(df.duplicated().sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records", df.shape[0])
c2.metric("Total Features", df.shape[1])
c3.metric("Missing Values", total_missing)
c4.metric("Duplicate Data", total_dup)

st.subheader("Dataset Preview")
n_rows = st.slider("Jumlah baris ditampilkan", 5, 50, 10)
st.dataframe(df.head(n_rows))

tab1, tab2, tab3 = st.tabs(["Tipe Data & Missing", "Statistik Deskriptif", "Data Duplikat"])

with tab1:
    info = pd.DataFrame(
        {
            "Atribut": df.columns,
            "Tipe Data": [str(t) for t in df.dtypes],
            "Non-Null": df.notnull().sum().values,
            "Missing": df.isnull().sum().values,
            "Nilai Unik": df.nunique().values,
        }
    )
    st.dataframe(info, hide_index=True)
    if total_missing == 0:
        st.success("✅ Tidak ada data kosong (missing values) pada dataset ini.")
    else:
        st.warning(f"Terdapat {total_missing} data kosong. Penanganannya dilakukan di halaman Prosesing.")

with tab2:
    num_cols, cat_cols = split_columns(df)
    st.markdown("**Atribut numerik**")
    st.dataframe(df[num_cols].describe().T)
    if cat_cols:
        st.markdown("**Atribut kategorikal**")
        st.dataframe(df[cat_cols].describe().T)

with tab3:
    if total_dup == 0:
        st.success("✅ Tidak ada data duplikat.")
    else:
        st.warning(f"Terdapat {total_dup} baris duplikat.")
        st.dataframe(df[df.duplicated(keep=False)])
