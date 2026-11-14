import time

import pandas as pd
import streamlit as st

from utils.modeling import FOREST, LINEAR, MODEL_NAMES, train_model
from utils.prosesing import TARGET, require_prep, setup_page

setup_page("Modeling", "🤖", subtitle="Pilih algoritma, atur parameter, lalu latih model")

prep = require_prep()
st.session_state.setdefault("models", {})

# ---------------------------------------------------------------- Ringkasan data
st.subheader("Fitur Input dan Target")
c1, c2, c3 = st.columns(3)
c1.metric("Target", TARGET)
c2.metric("Jumlah Fitur Input", prep["X_train"].shape[1])
c3.metric("Training / Testing", f"{len(prep['X_train'])} / {len(prep['X_test'])}")
with st.expander("Lihat daftar fitur input"):
    st.write(", ".join(prep["X_train"].columns))

# ---------------------------------------------------------------- Pilih algoritma
st.subheader("Pilih Algoritma")
algo = st.selectbox("Algoritma", MODEL_NAMES)

params = {"random_state": 42}
if algo == LINEAR:
    st.caption("Mempelajari hubungan linear antara fitur siswa dan nilai akhir. Tidak ada hyperparameter.")
else:
    st.caption("Model ensemble dari banyak decision tree yang hasilnya dirata-rata.")
    p1, p2 = st.columns(2)
    params["n_estimators"] = p1.slider("Jumlah pohon (n_estimators)", 50, 500, 200, step=50)
    depth = p2.slider("Kedalaman maksimum (0 = tanpa batas)", 0, 30, 0)
    params["max_depth"] = depth or None

# Hyperparameter Random Forest dipakai juga untuk tombol "latih kedua model"
rf_params = params if algo == FOREST else {"random_state": 42, "n_estimators": 200, "max_depth": None}


def run_training(names):
    bar = st.progress(0.0, text="Memulai training...")
    for i, name in enumerate(names):
        bar.progress(i / len(names), text=f"Melatih {name}...")
        p = rf_params if name == FOREST else {}
        t0 = time.time()
        pipe = train_model(name, prep, **p)
        st.session_state["models"][name] = {
            "pipeline": pipe,
            "params": p,
            "seconds": time.time() - t0,
        }
    bar.progress(1.0, text="Training selesai ✅")


b1, b2 = st.columns(2)
if b1.button(f"▶️ Train Model ({algo})", type="primary"):
    run_training([algo])
if b2.button("⏩ Latih Kedua Algoritma"):
    run_training(MODEL_NAMES)

# ---------------------------------------------------------------- Model tersimpan
st.subheader("Model yang Sudah Dilatih (sesi ini)")
models = st.session_state["models"]
if not models:
    st.info("Belum ada model yang dilatih.")
else:
    rows = [
        {
            "Model": name,
            "Parameter": ", ".join(f"{k}={v}" for k, v in info["params"].items() if k != "random_state") or "-",
            "Waktu Training (detik)": round(info["seconds"], 2),
        }
        for name, info in models.items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.success("➡️ Lanjutkan ke halaman **Evalution** untuk melihat performa model.")
