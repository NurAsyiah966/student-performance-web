import streamlit as st

from utils.prosesing import DEFAULT_DATASET, TARGET, load_dataset, set_dataset

# ------------------------------------------------------------------ Upload
with st.container(key="card_upload"):
    st.subheader("Upload Dataset")
    uploaded = st.file_uploader(
        "Upload Student Performance Dataset (student-mat.csv)", type=["csv"]
    )

    if uploaded is not None:
        upload_id = f"{uploaded.name}-{uploaded.size}"
        if st.session_state.get("last_upload") != upload_id:
            try:
                set_dataset(load_dataset(uploaded), uploaded.name)
                st.session_state["last_upload"] = upload_id
            except Exception as e:  # file kosong / format salah
                st.error(f"Gagal membaca dataset: {e}")

    if DEFAULT_DATASET.exists():
        if st.button("Gunakan student-mat.csv bawaan project"):
            try:
                set_dataset(load_dataset(DEFAULT_DATASET), DEFAULT_DATASET.name)
            except Exception as e:
                st.error(f"Gagal membaca dataset: {e}")

    if "df" in st.session_state:
        st.success(f"Dataset berhasil diupload: **{st.session_state.get('filename')}**")
    else:
        st.info("Upload file CSV atau klik tombol di atas untuk memulai.")

# ------------------------------------------------------------------ Ringkasan data
if "df" in st.session_state:
    df = st.session_state["df"]
    with st.container(key="card_data"):
        st.subheader("Ringkasan Dataset")
        c1, c2, c3 = st.columns(3)
        c1.metric("Jumlah Baris", df.shape[0])
        c2.metric("Jumlah Kolom", df.shape[1])
        c3.metric("Target Prediksi", TARGET)
        st.markdown("**Dataset Preview**")
        st.dataframe(df.head())

# ------------------------------------------------------------------ Alur (kartu flip)
FLOW_CSS = """
<style>
.flow-row { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 14px; margin-top: .4rem; }
@media (max-width: 900px) { .flow-row { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.flow-card { height: 180px; }
.flow-card summary { list-style: none; cursor: pointer; height: 100%; display: block; }
.flow-card summary::-webkit-details-marker { display: none; }
.flow-card summary::marker { content: ""; }
.flow-face { height: 100%; box-sizing: border-box; padding: 1rem; display: flex; flex-direction: column;
    align-items: center; justify-content: center; text-align: center; border-radius: 22px;
    box-shadow: 0 6px 18px rgba(127,168,214,.25); color: #1F3552;
    animation: flipIn .45s ease; }
@keyframes flipIn {
    from { transform: perspective(700px) rotateY(90deg); opacity: .2; }
    to   { transform: perspective(700px) rotateY(0deg); opacity: 1; }
}
.flow-card .flow-back { display: none; }
.flow-card[open] .flow-front { display: none; }
.flow-card[open] .flow-back { display: flex; }
.flow-card.yellow .flow-front { background: #FFF2B2; border: 2px solid #FFE08A; }
.flow-card.yellow .flow-back  { background: #FFE08A; border: 2px solid #E9B949; }
.flow-card.blue .flow-front   { background: #DCE9F7; border: 2px solid #A8C6E7; }
.flow-card.blue .flow-back    { background: #A8C6E7; border: 2px solid #7FA8D6; }
.flow-num { width: 38px; height: 38px; border-radius: 50%; display: flex; align-items: center;
    justify-content: center; background: #2B4162; color: #FFF7D6; font-weight: 800; margin-bottom: .6rem; }
.flow-name { font-size: 1.05rem; }
.flow-hint { margin-top: .35rem; font-size: .75rem; color: #4F6D94; }
.flow-badge { margin-top: .45rem; padding: .1rem .65rem; border-radius: 999px; background: #2B4162;
    color: #FFF7D6; font-size: .72rem; font-weight: 700; }
.flow-desc { font-size: .9rem; line-height: 1.5; color: #1F3552; }
</style>
"""

ss = st.session_state
has_df, has_prep, has_models = "df" in ss, "prep" in ss, bool(ss.get("models"))

# (nama, deskripsi singkat, sudah selesai?)
steps = [
    ("Overview", "Melihat jumlah data, missing values, duplikat, dan tipe data.", has_df),
    ("EDA", "Menjelajahi sebaran data, outlier, dan hubungan antar atribut lewat grafik.", has_df),
    ("Prosesing", "Menyiapkan data: encoding, scaling, dan pembagian data training-testing.", has_prep),
    ("Modeling", "Melatih model Linear Regression atau Random Forest.", has_models),
    ("Evalution", "Menilai model dengan MAE, RMSE, R², dan grafik prediksi.", has_models),
]

cards = ""
for i, (title, desc, done) in enumerate(steps):
    color = "yellow" if i % 2 == 0 else "blue"
    badge = '<div class="flow-badge">Selesai</div>' if done else ""
    cards += (
        f'<details class="flow-card {color}"><summary>'
        f'<div class="flow-face flow-front">'
        f'<div class="flow-num">{i + 1}</div><div class="flow-name">{title}</div>'
        f'{badge}<div class="flow-hint">Klik untuk detail</div></div>'
        f'<div class="flow-face flow-back"><div class="flow-desc">{desc}</div></div>'
        f"</summary></details>"
    )

st.subheader("Alur Penggunaan Aplikasi")
st.markdown(FLOW_CSS + f'<div class="flow-row">{cards}</div>', unsafe_allow_html=True)
st.caption("Dataset: UCI Student Performance (student-mat.csv)")