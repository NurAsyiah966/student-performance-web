import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from utils.modeling import evaluate_model, feature_effects
from utils.prosesing import TARGET, require_prep, setup_page

setup_page("Model Evaluation", "📈", subtitle="Bandingkan performa model dengan metrik regresi dan grafik prediksi")

prep = require_prep()
models = st.session_state.get("models", {})
if not models:
    st.warning("⚠️ Belum ada model yang dilatih. Buka halaman **Modeling** terlebih dahulu.")
    st.stop()

X_train, X_test = prep["X_train"], prep["X_test"]
y_train, y_test = prep["y_train"], prep["y_test"]

# ---------------------------------------------------------------- Hitung metrik
rows, preds = [], {}
for name, info in models.items():
    m_test, pred = evaluate_model(info["pipeline"], X_test, y_test)
    m_train, _ = evaluate_model(info["pipeline"], X_train, y_train)
    preds[name] = pred
    rows.append({"Model": name, **m_test, "R2 (Train)": m_train["R2"]})
results = pd.DataFrame(rows).set_index("Model")

# ---------------------------------------------------------------- Tabel
st.subheader("Perbandingan Model (data testing)")
try:
    styled = (
        results.style.format("{:.3f}")
        .highlight_min(subset=["MAE", "MSE", "RMSE"], color="#FFE08A")
        .highlight_max(subset=["R2"], color="#FFE08A")
    )
    st.dataframe(styled)
except Exception:
    st.dataframe(results.round(3))

best = results["RMSE"].idxmin()
baseline_pred = np.full(len(y_test), y_train.mean())
baseline_rmse = float(np.sqrt(np.mean((y_test - baseline_pred) ** 2)))
baseline_mae = float(np.mean(np.abs(y_test - baseline_pred)))
st.success(
    f"🏆 Model terbaik berdasarkan RMSE terendah: **{best}** "
    f"(MAE {results.loc[best, 'MAE']:.2f}, RMSE {results.loc[best, 'RMSE']:.2f}, R² {results.loc[best, 'R2']:.3f})"
)
st.caption(
    f"Pembanding (baseline: selalu menebak rata-rata G3 training): MAE {baseline_mae:.2f}, "
    f"RMSE {baseline_rmse:.2f}, R² ≈ 0."
)
if (results["R2"] < 0).any():
    st.warning("R² negatif berarti model lebih buruk daripada sekadar menebak rata-rata target.")
if len(models) == 1:
    st.info("Latih algoritma kedua di halaman Modeling untuk membandingkan kedua model.")

with st.expander("Cara membaca metrik"):
    st.markdown(
        "- **MAE**: rata-rata kesalahan absolut prediksi (semakin kecil semakin baik).\n"
        "- **MSE**: rata-rata kuadrat kesalahan.\n"
        "- **RMSE**: akar MSE, dalam skala nilai G3; memberi penalti lebih besar pada kesalahan besar.\n"
        "- **R²**: seberapa besar variasi target yang dijelaskan model (bisa negatif).\n"
        "- **R² (Train)** jauh di atas R² testing menandakan *overfitting*."
    )

# ---------------------------------------------------------------- Grafik perbandingan
st.subheader("Model Performance Comparison")
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(results))
w = 0.35
b1 = ax.bar(x - w / 2, results["MAE"], w, label="MAE", color="#7FA8D6")
b2 = ax.bar(x + w / 2, results["RMSE"], w, label="RMSE", color="#E9B949")
ax.bar_label(b1, fmt="%.2f")
ax.bar_label(b2, fmt="%.2f")
ax.set_xticks(x)
ax.set_xticklabels(results.index)
ax.set_ylabel("Error (skala nilai G3)")
ax.legend()
st.pyplot(fig)
plt.close(fig)

# ---------------------------------------------------------------- Detail per model
st.subheader("Analisis Prediksi")
chosen = st.selectbox("Pilih model", list(models.keys()), index=list(models.keys()).index(best))
pred = preds[chosen]
residual = y_test.values - pred

g1, g2 = st.columns(2)
with g1:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_test, pred, alpha=0.6, color="#7FA8D6")
    lim = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    ax.plot(lim, lim, "--", color="#2B4162", label="Prediksi sempurna")
    ax.set_xlabel(f"{TARGET} aktual")
    ax.set_ylabel(f"{TARGET} prediksi")
    ax.set_title("Actual vs Predicted")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
with g2:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(pred, residual, alpha=0.6, color="#A8C6E7")
    ax.axhline(0, color="#2B4162", linestyle="--")
    ax.set_xlabel(f"{TARGET} prediksi")
    ax.set_ylabel("Residual (aktual − prediksi)")
    ax.set_title("Residual Plot")
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------- Pengaruh fitur
st.subheader("Fitur Paling Berpengaruh")
effects, label = feature_effects(models[chosen]["pipeline"], top_n=15)
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(effects["Fitur"][::-1], effects[label][::-1], color="#7FA8D6")
ax.set_xlabel(label)
ax.set_title(f"Top 15 fitur — {chosen}")
st.pyplot(fig)
plt.close(fig)
if label == "Koefisien":
    st.caption("Koefisien dihitung pada fitur yang sudah di-scale, sehingga ukurannya dapat dibandingkan antar fitur.")
