"""Fungsi preprocessing & helper yang dipakai bersama oleh semua halaman."""
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "G3"
LEAK_COLS = ["G1", "G2"]
DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "student-mat.csv"

MISSING_FILL = "Isi numerik dengan median & kategorikal dengan modus"
MISSING_DROP = "Hapus baris yang memiliki missing values"


# ----------------------------------------------------------------------------
# Tampilan & state
# ----------------------------------------------------------------------------
# Palet warna: Buttercup Sky, Dewy Blue, Sunwashed, Cloud Puff, Morning Breeze
PALETTE = {
    "buttercup": "#FFF2B2",
    "dewy": "#A8C6E7",
    "sunwashed": "#FFE08A",
    "cloud": "#FFF7D6",
    "breeze": "#7FA8D6",
    "gold": "#E9B949",
    "ink": "#2B4162",
}
SKY_CMAP = LinearSegmentedColormap.from_list(
    "sky", [PALETTE["breeze"], PALETTE["cloud"], PALETTE["gold"]]
)

# Font: Nunito (isi teks) dan Poppins (judul). Ganti nama font di sini jika ingin mengubahnya.
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=Poppins:wght@500;600;700&display=swap');
.stApp { background: linear-gradient(180deg, #FFF7D6 0%, #FFFDF2 260px, #FFFDF2 100%); }
.stApp *:not([data-testid="stIconMaterial"]):not(code):not(pre):not(kbd) {
    font-family: 'Nunito', 'Segoe UI', system-ui, sans-serif;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .site-title, .page-head-title, .flow-name {
    font-family: 'Poppins', 'Segoe UI', system-ui, sans-serif !important;
    color: #2B4162; font-weight: 700; letter-spacing: -0.01em;
}
.stApp h1 { font-size: 2.1rem; }
.stApp h2 { font-size: 1.7rem; }
.stApp h3 { font-size: 1.35rem; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1250px; padding-top: 2.2rem; }
/* sidebar tidak dipakai: menu ada di bagian atas halaman */
section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display: none !important; }

/* Header situs */
.site-header {
    background: linear-gradient(120deg, #FFF2B2 0%, #FFE08A 40%, #A8C6E7 100%);
    border-radius: 28px; padding: 1.8rem 2.2rem; margin-bottom: .9rem;
    box-shadow: 0 12px 30px rgba(127,168,214,.30); position: relative; overflow: hidden;
}
.site-header:after { content: ""; position: absolute; right: -60px; top: -70px; width: 240px; height: 240px;
    border-radius: 50%; background: rgba(255,255,255,.35); }
.site-badge { display: inline-block; background: rgba(255,255,255,.65); color: #2B4162;
    padding: .2rem .85rem; border-radius: 999px; font-weight: 700; font-size: .8rem; margin-bottom: .6rem;
    letter-spacing: .04em; text-transform: uppercase; }
.site-title { color: #1F3552 !important; font-size: 2.3rem; line-height: 1.15; }
.site-sub { margin-top: .4rem; color: #2B4162; font-size: 1.02rem; }

/* Menu horizontal */
div[class*="st-key-navbar"] {
    background: #FFFDF2; border: 2px solid #FFE08A; border-radius: 999px;
    padding: .4rem .6rem; margin-bottom: 1.4rem; box-shadow: 0 6px 18px rgba(127,168,214,.20);
}
div[class*="st-key-navbar"] [data-testid="stHorizontalBlock"] { align-items: center; gap: .4rem; }
div[class*="st-key-navbar"] a {
    display: flex; align-items: center; justify-content: center; min-height: 2.6rem;
    border-radius: 999px; padding: .4rem .8rem; color: #2B4162; text-decoration: none;
    transition: background .15s ease;
}
div[class*="st-key-navbar"] a p { color: inherit; font-weight: 700; margin: 0; }
div[class*="st-key-navbar"] a:hover { background: #FFF2B2; }
div[class*="st-key-navitem_active"] a {
    background: #FFE08A; color: #1F3552;
    box-shadow: inset 0 0 0 2px #E9B949, 0 3px 10px rgba(233,185,73,.35);
}
div[class*="st-key-navitem_active"] a:hover { background: #FFE08A; }
div[class*="st-key-navitem_active"] a p { font-weight: 800; }

/* Judul halaman */
.page-head { border-left: 6px solid #FFE08A; padding: .15rem 0 .15rem 1rem; margin: .2rem 0 1.2rem; }
.page-head-title { color: #1F3552 !important; font-size: 1.8rem; line-height: 1.2; }
.page-head-sub { color: #4F6D94; margin-top: .25rem; }

/* Kartu metrik */
div[data-testid="stMetric"] {
    background: #FFF7D6; border: 1.5px solid #FFE08A; border-left: 6px solid #7FA8D6;
    padding: 14px 18px; border-radius: 16px; box-shadow: 0 4px 14px rgba(127,168,214,.18);
}
div[data-testid="stMetricValue"] { color: #2B4162; font-weight: 700; }

/* Tombol */
.stButton > button, .stDownloadButton > button, [data-testid="stFileUploader"] button {
    border-radius: 999px; font-weight: 700; padding: .55rem 1.5rem;
    background: #A8C6E7; color: #1F3552; border: 2px solid #7FA8D6;
    box-shadow: 0 3px 10px rgba(127,168,214,.30); transition: all .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFileUploader"] button:hover {
    background: #7FA8D6; color: #1F3552; border-color: #7FA8D6; transform: translateY(-2px);
}
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
    background: #FFE08A; color: #1F3552; border: 2px solid #E9B949;
    box-shadow: 0 4px 12px rgba(233,185,73,.40);
}
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: #FFD35C; border-color: #E9B949; color: #1F3552;
}
.stButton > button p, .stDownloadButton > button p { color: inherit; }

/* Upload & alert */
[data-testid="stFileUploaderDropzone"] {
    background: #FFF7D6; border: 2.5px dashed #7FA8D6; border-radius: 18px;
}
div[data-testid="stAlert"] { border-radius: 14px; }

/* Container kartu: st.container(key="card_...") */
div[class*="st-key-card"] {
    background: rgba(255,247,214,.70); border: 2px solid #FFE08A; border-radius: 22px;
    padding: 1.2rem 1.5rem; box-shadow: 0 8px 24px rgba(127,168,214,.20); margin-bottom: .6rem;
}
</style>
"""


def apply_theme(title: str = "Student Performance Analysis", icon: str = "📊"):
    """Dipanggil SEKALI di app.py: konfigurasi halaman, CSS tema, dan gaya grafik."""
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    mpl.rcParams.update(
        {
            "figure.facecolor": "#FFFDF2",
            "axes.facecolor": "#FFFDF2",
            "axes.edgecolor": PALETTE["dewy"],
            "axes.labelcolor": PALETTE["ink"],
            "axes.titlecolor": PALETTE["ink"],
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "legend.frameon": False,
            "axes.prop_cycle": mpl.cycler(
                color=[PALETTE["breeze"], PALETTE["gold"], PALETTE["dewy"], PALETTE["sunwashed"]]
            ),
        }
    )


def render_header(pages, current):
    """Nama web + menu horizontal di bawahnya. Menu yang aktif ditandai oval kuning."""
    st.markdown(
        '<div class="site-header">'
        '<div class="site-badge">Data Science Web Application</div>'
        '<div class="site-title">Student Performance Analysis</div>'
        '<div class="site-sub">Analisis dan prediksi nilai akhir siswa (G3)</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    with st.container(key="navbar"):
        cols = st.columns(len(pages))
        for col, page in zip(cols, pages):
            with col:
                if page.title == current.title:
                    with st.container(key="navitem_active"):
                        st.page_link(page, width="stretch")
                else:
                    st.page_link(page, width="stretch")

def setup_page(title: str, icon: str = "", subtitle: str = None):
    """Dipanggil di tiap halaman: menampilkan judul halaman (jika subtitle diisi)."""
    if subtitle is not None:
        st.markdown(
            f'<div class="page-head"><div class="page-head-title">{title}</div>'
            f'<div class="page-head-sub">{subtitle}</div></div>',
            unsafe_allow_html=True,
        )


def set_dataset(df: pd.DataFrame, name: str):
    """Simpan dataset baru dan reset hasil tahap-tahap berikutnya."""
    st.session_state["df"] = df
    st.session_state["filename"] = name
    st.session_state.pop("prep", None)
    st.session_state.pop("models", None)


def require_dataset() -> pd.DataFrame:
    if "df" not in st.session_state:
        st.warning(
            "Dataset belum diupload. Buka menu **Dashboard** di bagian atas "
            "lalu upload `student-mat.csv` terlebih dahulu."
        )
        st.stop()
    return st.session_state["df"]


def require_prep() -> dict:
    require_dataset()
    if "prep" not in st.session_state:
        st.warning("Preprocessing belum dijalankan. Buka menu **Prosesing** di bagian atas terlebih dahulu.")
        st.stop()
    return st.session_state["prep"]


# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
def load_dataset(source) -> pd.DataFrame:
    """Baca CSV (pemisah ';' sesuai student-mat.csv, fallback ',')."""
    df = pd.read_csv(source, sep=";")
    if df.shape[1] == 1:  # kemungkinan pemisahnya koma
        if hasattr(source, "seek"):
            source.seek(0)
        df = pd.read_csv(source, sep=",")
    if df.empty:
        raise ValueError("Dataset kosong.")
    if TARGET not in df.columns:
        raise ValueError(f"Kolom target '{TARGET}' tidak ditemukan. Pastikan file yang diupload adalah student-mat.csv.")
    return df


def split_columns(df: pd.DataFrame):
    """Pisahkan kolom numerik dan kategorikal."""
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols]
    return num_cols, cat_cols


# ----------------------------------------------------------------------------
# Outlier (IQR)
# ----------------------------------------------------------------------------
def iqr_bounds(series: pd.Series):
    s = series.dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1, q3, iqr, q1 - 1.5 * iqr, q3 + 1.5 * iqr


def outlier_summary(df: pd.DataFrame, cols) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = df[c].dropna()
        q1, q3, iqr, lower, upper = iqr_bounds(s)
        n_out = int(((s < lower) | (s > upper)).sum())
        rows.append(
            {
                "Atribut": c,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "Batas Bawah": lower,
                "Batas Atas": upper,
                "Jumlah Outlier": n_out,
                "Persentase (%)": round(n_out / len(s) * 100, 2) if len(s) else 0.0,
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------
def build_preprocessor(num_cols, cat_cols) -> ColumnTransformer:
    """Imputer + StandardScaler (numerik) dan Imputer + OneHotEncoder (kategorikal).

    Semua transformer di-fit hanya pada data training (ketika pipeline di-fit),
    sehingga tidak terjadi data leakage.
    """
    num_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("num", num_pipe, list(num_cols)), ("cat", cat_pipe, list(cat_cols))],
        sparse_threshold=0,
    )


def make_split(df, drop_cols, test_size, random_state, missing_method) -> dict:
    """Pisahkan fitur/target lalu bagi menjadi data training dan testing."""
    features = [c for c in df.columns if c != TARGET and c not in drop_cols]
    data = df.dropna(subset=[TARGET]).copy()
    if missing_method == MISSING_DROP:
        data = data.dropna(subset=features + [TARGET])

    X, y = data[features], data[TARGET]
    num_cols, cat_cols = split_columns(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_state)
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "preprocessor": build_preprocessor(num_cols, cat_cols),
        "settings": {
            "missing_method": missing_method,
            "drop_cols": list(drop_cols),
            "test_size": test_size,
            "random_state": int(random_state),
            "rows_used": len(data),
        },
    }


def transformed_preview(prep: dict) -> pd.DataFrame:
    """Hasil transformasi data training (untuk ditampilkan saja)."""
    pre = clone(prep["preprocessor"])
    Xt = pre.fit_transform(prep["X_train"])
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    return pd.DataFrame(Xt, columns=names, index=prep["X_train"].index)