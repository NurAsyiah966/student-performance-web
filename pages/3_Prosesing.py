import pandas as pd
import streamlit as st

from utils.prosesing import (
    LEAK_COLS,
    MISSING_DROP,
    MISSING_FILL,
    TARGET,
    make_split,
    require_dataset,
    setup_page,
    transformed_preview,
)

setup_page("Data Preprocessing", "⚙️", subtitle="Missing values, encoding, scaling, dan pembagian data training-testing")
st.caption("Alur: cek missing values → pisahkan fitur & target → split train/test → encoding → scaling")

df = require_dataset()

# ---------------------------------------------------------------- 1. Missing
st.subheader("1. Missing Values")
missing = df.isnull().sum()
if missing.sum() == 0:
    st.success("✅ Tidak ada data kosong pada dataset. Pipeline tetap menyertakan imputer sebagai pengaman.")
else:
    st.warning(f"Terdapat {int(missing.sum())} data kosong.")
    st.dataframe(missing[missing > 0].rename("Jumlah Missing"))
missing_method = st.radio("Metode penanganan missing values", [MISSING_FILL, MISSING_DROP])

# ---------------------------------------------------------------- 2. Fitur & target
st.subheader("2. Fitur dan Target")
st.write(f"Target prediksi: **{TARGET}** (nilai akhir siswa)")
leak_present = [c for c in LEAK_COLS if c in df.columns]
drop_leak = st.checkbox(
    f"Buang {', '.join(leak_present)} dari fitur (disarankan)",
    value=True,
    help="G1 dan G2 adalah nilai periode sebelumnya yang sangat berkorelasi dengan G3, "
    "sehingga model hanya 'meniru' nilai tersebut. Tanpa keduanya, model benar-benar "
    "memprediksi dari karakteristik siswa.",
)
other_options = [c for c in df.columns if c != TARGET and c not in LEAK_COLS]
extra_drop = st.multiselect("Buang fitur lain (opsional)", other_options)

# ---------------------------------------------------------------- 3. Split
st.subheader("3. Train-Test Split")
c1, c2 = st.columns(2)
test_pct = c1.slider("Persentase data testing (%)", 10, 40, 20, step=5)
random_state = c2.number_input("Random state", min_value=0, value=42, step=1)

# ---------------------------------------------------------------- 4. Encoding & scaling
st.subheader("4. Encoding dan Scaling")
st.markdown(
    "- **OneHotEncoder** untuk atribut kategorikal (school, sex, Mjob, dst.)\n"
    "- **StandardScaler** untuk atribut numerik (target tidak di-scale)\n"
    "- Encoding dan scaling di-**fit hanya pada data training**, lalu diterapkan ke data testing "
    "(mencegah *data leakage*)."
)

if st.button("🚀 Jalankan Preprocessing", type="primary"):
    drop_cols = (leak_present if drop_leak else []) + extra_drop
    with st.spinner("Memproses data..."):
        st.session_state["prep"] = make_split(df, drop_cols, test_pct / 100, random_state, missing_method)
        st.session_state.pop("models", None)  # model lama tidak berlaku lagi
    st.rerun()

# ---------------------------------------------------------------- Hasil
if "prep" in st.session_state:
    prep = st.session_state["prep"]
    cfg = prep["settings"]
    st.divider()
    st.subheader("✅ Hasil Preprocessing")
    st.caption(
        f"Pengaturan terakhir: test size {cfg['test_size']:.0%}, random state {cfg['random_state']}, "
        f"fitur dibuang: {', '.join(cfg['drop_cols']) or '-'}"
    )

    transformed = transformed_preview(prep)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Data Training", len(prep["X_train"]))
    m2.metric("Data Testing", len(prep["X_test"]))
    m3.metric("Fitur Awal", prep["X_train"].shape[1])
    m4.metric("Fitur Setelah Encoding", transformed.shape[1])

    t1, t2, t3 = st.tabs(["Daftar Fitur", "Sebelum vs Sesudah", "Verifikasi Scaling"])
    with t1:
        a, b = st.columns(2)
        a.markdown(f"**Fitur numerik ({len(prep['num_cols'])})**")
        a.write(", ".join(prep["num_cols"]))
        b.markdown(f"**Fitur kategorikal ({len(prep['cat_cols'])})**")
        b.write(", ".join(prep["cat_cols"]) or "-")
    with t2:
        st.markdown("**Sebelum (data training asli)**")
        st.dataframe(prep["X_train"].head())
        st.markdown("**Sesudah (encoding + scaling)**")
        st.dataframe(transformed.head())
    with t3:
        st.write("Setelah StandardScaler, fitur numerik pada data training memiliki mean ≈ 0 dan std ≈ 1.")
        stats = transformed[prep["num_cols"]].agg(["mean", "std"]).T.round(3)
        st.dataframe(stats)

    st.success("➡️ Lanjutkan ke halaman **Modeling** untuk melatih model.")
